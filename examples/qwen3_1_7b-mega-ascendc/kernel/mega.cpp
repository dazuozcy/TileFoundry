// Qwen3-1.7B decode mega-kernel -- ONE AscendC MIX launch per token.
//
// 24 blocks = 24 AIC (cube) + 48 AIV (vector), single launch on a mix binary
// (KERNEL_TYPE_MIX_AIC_1_2, auto-identified -- NEVER compile with
// -cce-enable-mix). AIC blockIdx 0..23, AIV blockIdx 0..47;
// group g = AIC g + AIV 2g, 2g+1 (sub = bid&1).
//
// Decomposition:
//   * every GEMV splits N over 24 blocks in contiguous padded slabs
//     (qkv 4224 = 24x176 in 96+80 tiles; o/down 2304 = 24x96; gu 12288 =
//     24x512 in 4x128; lm 152064 = 24x6336 in 96 tiles), K in 256 chunks.
//   * attention mesh: block cid owns kv = cid%8, sid = cid/8; the live
//     prefix splits into 3 SLT-aligned thirds; scores = q2 * K^T per tile
//     (Mmad m=16, rows 0,1 = heads 2kv, 2kv+1), two-pass softmax on AIV,
//     PV = probs2 * V per tile, LSE combine on sid==0 leader AIVs.
//   * elementwise (rmsnorm, qk-norm, rope, silu, residuals, argmax)
//     replicated on AIVs; GM writes + seq writes done by sub==0 only.
//
// Layouts (verified on hardware):
//   * A-side GEMV fractal source: x broadcast 16 rows per k-fractal
//     (AIV builds via Copy{1,0,8,0} per src block, probe11).
//   * B-side fractal source: [k-block][n-block][16n x 16k] row-major.
//   * K cache KcP: [li*8+kv][d-block j][s-block i][16s x 16d] (d-major).
//   * V cache VcP: [li*8+kv][s-block i][d-block j][16d x 16s] (s-major).
//   * GEMV outputs are (16, N_pad) f32; consumers read row 0.
//   * scores16 (NB,16,STEP): rows 0,1 = the two heads' tile scores.
//     Score tail beyond the live prefix is EXACTLY 0 (host-zeroed K pads),
//     so probs tail x V-zero = 0; l/max only reduce the valid prefix.
//
// SYNC (hardware-verified recipe, see smoke3/NOTES.md):
//   * FFTS mode-2 cross-core flags are WEAK for multi-setter gates: each set
//     broadcasts +1 to every AIV's local flag level and a wait passes after
//     ONE set -- NOT after all 24 AICs. They are NOT used here.
//   * ALL cross-plane gates use GM seq words in p_syncws (verified probes):
//       - AIV->AIC: AIV sub0 writes row cid (MTE3 DataCopy of 16 int32);
//         AIC polls (dcci + volatile read). Rows 0..23.
//       - AIC->AIV: AIC writes row 24+cid (direct scalar store + dcci
//         CACHELINE_OUT publish, KFC pattern). Rows 24..47.
//       - full-row AIV reads (qkv/o/gu/d, all 24 AICs): each AIV polls ONE
//         assigned slot (24 + bid%24), then a plane SyncAll aggregates --
//         after the barrier all 24 AICs are confirmed. Per-group reads
//         (scores, lm span) poll the own-cid slot only; the combine
//         leaders poll the 3 slice-AIC slots (part_m/l readiness is
//         transitive via those AICs' own R+3 polls).
//   * tag increments by 512 per host launch (starting at 512: round 0 of
//     launch 0 must not collide with the zero-initialised slots); stale
//     words never match. AIV rounds/layer: 0 rms1 done, 2 appends done,
//     3 probs done, 5 B_ATT, 6 xn2 done, 7 h done; finals 224 rmsf.
//     AIC rounds/layer: 0 qkv done, 2 scores done, 4 partials done,
//     5 o done, 6 gu done, 7 down done; finals 224 logits done.
//   * plane barrier: SyncAll<false>(gmWs, ubWs, 48) -- the isAIVOnly=true
//     default HANGS in mix. GM ws 8x48 int32 host-zeroed once, monotonic.
//     5 plane barriers per layer (qkv gate, attn combine, o/gu/d gates).
//   * S_MTE3 event ids are one-shot per kernel (reuse hangs): the V-append
//     cycles 4 ids. V_MTE3 / MTE3_V / MTE2_S / MTE3_MTE2 ids are reusable.
//   * UB-source reuse after a GM DataCopy needs a trailing MTE3_V (or
//     MTE3_MTE2 before an MTE2 reload); scalar reads of MTE2-loaded data
//     need MTE2_S; scalar reads of vector data need PipeBarrier<PIPE_V>.
//
// lstop: number of layers to run (1..L) before the final lm -- validation
// slicing; all intermediates live in GM and are inspectable post-launch.
#include <kernel_operator.h>
using namespace AscendC;
auto __enable_feature_for_compile_default = KERNEL_TYPE_MIX_AIC_1_2;

// ---------------- model constants ----------------
constexpr int L = 28;
constexpr int H = 2048;
constexpr int KVH = 8;
constexpr int D = 128;
constexpr int II = 6144;
constexpr int VV = 151936;
constexpr int NB = 24;
constexpr int KTILE = 256;
constexpr int SLT = 256;
constexpr int NSL = 3;
constexpr int STEP = 13824;      // ceil(40961/3) rounded to 256
constexpr int SPADG = 41472;     // STEP*NSL
constexpr int NQKV = 4096;
constexpr int NQKV_PAD = 4224;   // NB*176
constexpr int NC_QKV = 96;       // tiles 96 + 80
constexpr int NO_PAD = 2304;     // NB*96
constexpr int NC_O = 96;
constexpr int NGU = 12288;       // NB*512
constexpr int NC_GU = 128;
constexpr int ND_PAD = 2304;     // NB*96
constexpr int NC_D = 96;
constexpr int SPAN = 6336;       // ceil(VV/NB) = 6331 -> 16-mult
constexpr int NC_LM = 96;
constexpr int NRD_LM = SPAN / NC_LM;  // 66
constexpr int SPANP = 8192;
constexpr int CH = 2048;
constexpr int NCH = SPAN / CH + 1;    // 4 (last chunk 192 valid)
constexpr int NTM = STEP / SLT;       // 54 max tiles per slice
constexpr float EPS = 1e-6f;
constexpr float NEG_INF = -1e30f;

// seq workspace layout (int32), carved out of p_syncws:
//   [0, 1536)                SyncAll GM ws (48 x 8)
//   [4096 ...]               rows 0..23 AIV slots, 24..47 AIC slots;
//                           slot = 16 int32 at SEQ_WANT(row, r)
//                           = 4096 + (row*256 + r)*16  (max 196608)
constexpr int SEQ_OFF = 4096;
constexpr int SEQ_ROUNDS = 256;        // 8*28 + finals headroom
__aicore__ inline int SEQ_WANT(int cid, int r) { return SEQ_OFF + (cid * 256 + r) * 16; }

// ---------------- cube helpers ----------------
struct CubeCtx {
    event_t ev0, ev1, ev2, ev3, ev4, ev5;
    LocalTensor<bfloat16_t> la1, lb1, la2, lb2;
    LocalTensor<float> lc1;
};

// AIC: poll a GM seq word until it equals want (dcci + volatile read).
__aicore__ inline void poll_seq(volatile __gm__ int32_t* p, int32_t want) {
    while (*p != want) {
        dcci(reinterpret_cast<__gm__ uint64_t*>((__gm__ int32_t*)p),
             cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
        __asm__ __volatile__("" ::: "memory");
    }
}

// AIC: publish a GM seq word via direct scalar store + dcci (KFC pattern,
// verified by probe p14aicw). The dcci(CACHELINE_OUT) pushes the store out
// of the cube's cache to L2 so other cores can poll it.
__aicore__ inline void aic_seq(volatile __gm__ int32_t* g, int slot, int32_t val) {
    volatile __gm__ int32_t* p = g + slot;
    __asm__ __volatile__("" ::: "memory");
    *p = val;
    __asm__ __volatile__("");
    dcci(reinterpret_cast<__gm__ uint64_t*>((__gm__ int32_t*)p),
         cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
    __asm__ __volatile__("");
}

// GEMV over one weight slab set; W packed [kb][nb][16n x 16k] with nb =
// nPad/16 n-fractals PER K-BLOCK (nPad >= nGlobal, host pads rows to nPad!);
// xPacked is the 16x broadcast A-fractal pack; out (16, outStride) f32,
// row 0 valid.
__aicore__ inline void gemv_slab(CubeCtx& c, GlobalTensor<bfloat16_t> wBase,
    GlobalTensor<bfloat16_t> xPacked, GlobalTensor<float> outBase,
    int nGlobal, int kTotal, int nc, int nrd, int slab, int cid, int outStride,
    int nPad) {
    const int NF = nPad / 16;
    const int KCC = KTILE / 16;
    bool l1Busy = false, l0Busy = false, l0cBusy = false;
    for (int nt = 0; nt < nrd; nt++) {
        const int n0 = cid * slab + nt * nc;
        if (n0 >= nGlobal) break;
        const int nlim = (cid * slab + slab < nGlobal) ? (cid * slab + slab) : nGlobal;
        const int ncEnd = (n0 + nc < nlim) ? (n0 + nc) : nlim;
        const int nce = ncEnd - n0;
        const int ncc = nce / 16;
        for (int k0 = 0; k0 < kTotal; k0 += KTILE) {
            if (l0cBusy) { WaitFlag<HardEvent::FIX_M>(c.ev4); l0cBusy = false; }
            if (l0Busy)  { WaitFlag<HardEvent::M_MTE1>(c.ev3); l0Busy = false; }
            if (l1Busy)  { WaitFlag<HardEvent::MTE1_MTE2>(c.ev5); l1Busy = false; }
            DataCopy(c.la1, xPacked[k0 * 16], 16 * KTILE);
            for (int jj = 0; jj < KCC; jj++) {
                const int j = k0 / 16 + jj;
                DataCopy(c.lb1[jj * ncc * 256], wBase[(int64_t)(j * NF + n0 / 16) * 256], ncc * 256);
            }
            SetFlag<HardEvent::MTE2_MTE1>(c.ev0);
            WaitFlag<HardEvent::MTE2_MTE1>(c.ev0);
            LoadData2DParams pa(0, (uint8_t)KCC, 1, 0, 0, false, 0);
            LoadData(c.la2, c.la1, pa);
            LoadData2DParams pb(0, (uint8_t)(KCC * ncc), 1, 0, 0, false, 0);
            LoadData(c.lb2, c.lb1, pb);
            SetFlag<HardEvent::MTE1_M>(c.ev1);
            WaitFlag<HardEvent::MTE1_M>(c.ev1);
            MmadParams mp; mp.m = 16; mp.n = nce; mp.k = KTILE; mp.cmatrixInitVal = (k0 == 0);
            Mmad(c.lc1, c.la2, c.lb2, mp);
            SetFlag<HardEvent::M_MTE1>(c.ev3);
            SetFlag<HardEvent::MTE1_MTE2>(c.ev5);
            l0Busy = true; l1Busy = true;
        }
        SetFlag<HardEvent::M_FIX>(c.ev2); WaitFlag<HardEvent::M_FIX>(c.ev2);
        FixpipeParamsV220 fp; fp.nSize = nce; fp.mSize = 16; fp.srcStride = 16; fp.dstStride = outStride;
        Fixpipe(outBase[n0], c.lc1, fp);
        SetFlag<HardEvent::FIX_MTE2>(c.ev0); WaitFlag<HardEvent::FIX_MTE2>(c.ev0);
        SetFlag<HardEvent::FIX_M>(c.ev4);
        l0cBusy = true;
    }
    if (l0Busy)  { WaitFlag<HardEvent::M_MTE1>(c.ev3); }
    if (l1Busy)  { WaitFlag<HardEvent::MTE1_MTE2>(c.ev5); }
    if (l0cBusy) { WaitFlag<HardEvent::FIX_M>(c.ev4); }
}

// ---------------- AIV helpers ----------------
// tree reduce over n (power-of-2 multiple of 8, n>=16); result in t[0..8)
__aicore__ inline float tree_sum(LocalTensor<float> t, int n) {
    for (int len = n / 2; len >= 8; len /= 2) { Add(t, t, t[len], len); }
    PipeBarrier<PIPE_V>();
    float s = t.GetValue(0);
    for (int i = 1; i < 8; i++) { s += t.GetValue(i); }
    return s;
}
__aicore__ inline float tree_max(LocalTensor<float> t, int n) {
    for (int len = n / 2; len >= 8; len /= 2) { Max(t, t, t[len], len); }
    PipeBarrier<PIPE_V>();
    float m = t.GetValue(0);
    for (int i = 1; i < 8; i++) { float x = t.GetValue(i); if (x > m) m = x; }
    return m;
}
// scalar exp via 16-el vector buffer
__aicore__ inline float vexp_scalar(LocalTensor<float> t16, float x) {
    Duplicate(t16, x, 16);
    Exp(t16, t16, 16);
    PipeBarrier<PIPE_V>();
    return t16.GetValue(0);
}

// rmsnorm: t (n f32, destroyed), s (n f32 scratch), gb (n bf16 gamma),
// o (n bf16 out), t16 (16 f32 scratch). n multiple of 8, >= 16.
__aicore__ inline void rms_norm(LocalTensor<float> t, LocalTensor<float> s,
    LocalTensor<bfloat16_t> gb, LocalTensor<bfloat16_t> o, int n, LocalTensor<float> t16) {
    Mul(s, t, t, n);
    const float ms = tree_sum(s, n) / (float)n;
    Duplicate(t16, ms + EPS, 16);
    Rsqrt(t16, t16, 16);
    PipeBarrier<PIPE_V>();
    const float r = t16.GetValue(0);
    Muls(t, t, r, n);
    Cast(o, t, RoundMode::CAST_RINT, n);      // land bf16 (HF numerics)
    Cast(t, o, RoundMode::CAST_NONE, n);      // back to f32
    Cast(s, gb, RoundMode::CAST_NONE, n);     // gamma
    Mul(t, t, s, n);
    Cast(o, t, RoundMode::CAST_RINT, n);
}

// rope: t (128 f32, in/out), cos/sinT (128 f32, sinT first half
// pre-negated), r (128 f32 scratch). pairs (i, i+64).
__aicore__ inline void rope_apply(LocalTensor<float> t, LocalTensor<float> cosr,
    LocalTensor<float> sinr, LocalTensor<float> r) {
    CopyRepeatParams cp; cp.dstStride = 1; cp.srcStride = 1; cp.dstRepeatSize = 8; cp.srcRepeatSize = 8;
    Copy(r, t[64], (uint64_t)64, (uint8_t)1, cp);
    Copy(r[64], t[0], (uint64_t)64, (uint8_t)1, cp);
    Mul(t, t, cosr, 128);
    Mul(r, r, sinr, 128);
    Add(t, t, r, 128);
}

// broadcast-pack: src (256 bf16) -> pack (4096 bf16): fractal j = 16 rows
// of src[j*16..+16] (probe11)
__aicore__ inline void bcast_pack(LocalTensor<bfloat16_t> pack, LocalTensor<bfloat16_t> src) {
    CopyRepeatParams cp; cp.dstStride = 1; cp.srcStride = 0; cp.dstRepeatSize = 8; cp.srcRepeatSize = 0;
    for (int j = 0; j < 16; j++) {
        Copy(pack[j * 256], src[j * 16], (uint64_t)128, (uint8_t)2, cp);
    }
}
// strided rows: dst fractal rows {base} of nblk fractals <- src (nblk*16
// bf16): dst block (16s + base) <- src block s (probe10-B)
__aicore__ inline void strided_rows(LocalTensor<bfloat16_t> dst, LocalTensor<bfloat16_t> src,
    int nblk, int base) {
    CopyRepeatParams cr; cr.dstStride = 16; cr.srcStride = 1; cr.dstRepeatSize = 128; cr.srcRepeatSize = 8;
    Copy(dst[base * 16], src, (uint64_t)128, (uint8_t)(nblk / 8), cr);
}

// padded x16 scalar GM write (buffer = slots of 16 f32); V->MTE3 produce
// sync + trailing MTE3_V so t16 is reusable immediately.
__aicore__ inline void put_scalar16(GlobalTensor<float> gm, int slot, float v,
    LocalTensor<float> t16, event_t eV3, event_t eM3V) {
    Duplicate(t16, v, 16);
    SetFlag<HardEvent::V_MTE3>(eV3); WaitFlag<HardEvent::V_MTE3>(eV3);
    DataCopy(gm[slot * 16], t16, 16);
    SetFlag<HardEvent::MTE3_V>(eM3V); WaitFlag<HardEvent::MTE3_V>(eM3V);
}

// pack an H-vector (bf16) from UB to a GM pack buffer in 256-el chunks
// (trailing MTE3_V per chunk: pack is reused next chunk)
__aicore__ inline void pack_out(LocalTensor<bfloat16_t> pack, LocalTensor<bfloat16_t> v,
    GlobalTensor<bfloat16_t> gmPack, int nChunks, event_t eV3, event_t eM3V) {
    for (int c = 0; c < nChunks; c++) {
        bcast_pack(pack, v[c * 256]);
        SetFlag<HardEvent::V_MTE3>(eV3); WaitFlag<HardEvent::V_MTE3>(eV3);
        DataCopy(gmPack[(int64_t)c * 4096], pack, 4096);
        SetFlag<HardEvent::MTE3_V>(eM3V); WaitFlag<HardEvent::MTE3_V>(eM3V);
    }
}

// AIV: write one GM seq word. `slot` = ELEMENT offset of the 16-int32 slot
// start (SEQ_WANT already includes SEQ_OFF and the *16 slot stride), so the
// AIC's poll address (p_syncws + SEQ_WANT) matches exactly. sq has room for 2
// slots; `which` alternates so the previous MTE3 copy's source is never
// clobbered; a trailing MTE3_V also retires the copy before sq reuse.
__aicore__ inline void seq_write(GlobalTensor<int32_t>& g, int32_t slot, int32_t val,
    LocalTensor<int32_t>& sq, int which, event_t eV3, event_t eM3V) {
    Duplicate(sq[which * 16], val, 16);
    SetFlag<HardEvent::V_MTE3>(eV3); WaitFlag<HardEvent::V_MTE3>(eV3);
    DataCopy(g[slot], sq[which * 16], 16);
    SetFlag<HardEvent::MTE3_V>(eM3V); WaitFlag<HardEvent::MTE3_V>(eM3V);
}

// ---------------- kernel ----------------
__global__ __aicore__ void megak(
    GM_ADDR p_ffts, GM_ADDR p_WqkvP, GM_ADDR p_WoP, GM_ADDR p_WguP, GM_ADDR p_WdP,
    GM_ADDR p_WlmP, GM_ADDR p_rms1w, GM_ADDR p_rms2w, GM_ADDR p_rmsfw, GM_ADDR p_qnw,
    GM_ADDR p_knw, GM_ADDR p_cosT, GM_ADDR p_sinT, GM_ADDR p_KcP, GM_ADDR p_VcP,
    GM_ADDR p_xEmb, GM_ADDR p_xn1P, GM_ADDR p_qkv_out, GM_ADDR p_q2P, GM_ADDR p_scores16,
    GM_ADDR p_probs2P, GM_ADDR p_partm, GM_ADDR p_partl, GM_ADDR p_partAccP, GM_ADDR p_attnP,
    GM_ADDR p_o_out, GM_ADDR p_xn2P, GM_ADDR p_gu_out, GM_ADDR p_hP, GM_ADDR p_d_out,
    GM_ADDR p_logits, GM_ADDR p_idxTab, GM_ADDR p_partMax, GM_ADDR p_partIdx,
    GM_ADDR p_syncws, GM_ADDR p_xnfP, int32_t sl, int32_t lstop, int32_t tag) {
    if ((uint64_t)p_ffts != 0) { SetSyncBaseAddr((uint64_t)p_ffts); }

    // dynamic slice split (SLT-aligned thirds)
    const int thirdsRaw = ((sl + 1) + NSL - 1) / NSL;
    const int thirds = ((thirdsRaw + SLT - 1) / SLT) * SLT;

    if ASCEND_IS_AIC {
        // ================= CUBE SIDE =================
        const int cid = GetBlockIdx();
        const int kv = cid % KVH;
        const int sid = cid / KVH;
        const int s0 = sid * thirds;
        const int e0 = (sl + 1 < s0 + thirds) ? (sl + 1) : (s0 + thirds);
        const int sblk = (e0 > s0) ? (e0 - s0) : 0;
        const int nloc = (sblk + SLT - 1) / SLT;

        TPipe pipe;
        TBuf<TPosition::A1> a1; TBuf<TPosition::B1> b1;
        TBuf<TPosition::A2> a2; TBuf<TPosition::B2> b2; TBuf<TPosition::CO1> c1;
        pipe.InitBuffer(a1, 16 * KTILE * sizeof(bfloat16_t));
        pipe.InitBuffer(b1, 64 * 1024);
        pipe.InitBuffer(a2, 16 * KTILE * sizeof(bfloat16_t));
        pipe.InitBuffer(b2, 64 * 1024);
        pipe.InitBuffer(c1, 16 * 256 * sizeof(float));
        pipe.FetchEventID<HardEvent::MTE2_MTE1>();
        CubeCtx C;
        C.ev0 = static_cast<event_t>(pipe.FetchEventID<HardEvent::MTE2_MTE1>());
        pipe.FetchEventID<HardEvent::MTE1_M>();
        C.ev1 = static_cast<event_t>(pipe.FetchEventID<HardEvent::MTE1_M>());
        pipe.FetchEventID<HardEvent::M_FIX>();
        C.ev2 = static_cast<event_t>(pipe.FetchEventID<HardEvent::M_FIX>());
        pipe.FetchEventID<HardEvent::M_MTE1>();
        C.ev3 = static_cast<event_t>(pipe.FetchEventID<HardEvent::M_MTE1>());
        pipe.FetchEventID<HardEvent::MTE1_MTE2>();
        C.ev5 = static_cast<event_t>(pipe.FetchEventID<HardEvent::MTE1_MTE2>());
        pipe.FetchEventID<HardEvent::FIX_M>();
        C.ev4 = static_cast<event_t>(pipe.FetchEventID<HardEvent::FIX_M>());
        C.la1 = a1.Get<bfloat16_t>();
        C.lb1 = b1.Get<bfloat16_t>();
        C.la2 = a2.Get<bfloat16_t>();
        C.lb2 = b2.Get<bfloat16_t>();
        C.lc1 = c1.Get<float>();

        __gm__ bfloat16_t* gWqkv = (__gm__ bfloat16_t*)p_WqkvP;
        __gm__ bfloat16_t* gWo = (__gm__ bfloat16_t*)p_WoP;
        __gm__ bfloat16_t* gWgu = (__gm__ bfloat16_t*)p_WguP;
        __gm__ bfloat16_t* gWd = (__gm__ bfloat16_t*)p_WdP;
        __gm__ bfloat16_t* gWlm = (__gm__ bfloat16_t*)p_WlmP;
        __gm__ bfloat16_t* gxn1P = (__gm__ bfloat16_t*)p_xn1P;
        __gm__ bfloat16_t* gq2P = (__gm__ bfloat16_t*)p_q2P;
        __gm__ bfloat16_t* gKcP = (__gm__ bfloat16_t*)p_KcP;
        __gm__ bfloat16_t* gVcP = (__gm__ bfloat16_t*)p_VcP;
        __gm__ bfloat16_t* gprobs2P = (__gm__ bfloat16_t*)p_probs2P;
        __gm__ float* gqkv_out = (__gm__ float*)p_qkv_out;
        __gm__ float* gscores = (__gm__ float*)p_scores16;
        __gm__ float* gpartAccP = (__gm__ float*)p_partAccP;
        __gm__ float* go_out = (__gm__ float*)p_o_out;
        __gm__ float* ggu_out = (__gm__ float*)p_gu_out;
        __gm__ float* gd_out = (__gm__ float*)p_d_out;
        __gm__ float* glogits = (__gm__ float*)p_logits;
        __gm__ bfloat16_t* gattnP = (__gm__ bfloat16_t*)p_attnP;
        __gm__ bfloat16_t* gxn2P = (__gm__ bfloat16_t*)p_xn2P;
        __gm__ bfloat16_t* ghP = (__gm__ bfloat16_t*)p_hP;
        __gm__ bfloat16_t* gxnfP = (__gm__ bfloat16_t*)p_xnfP;
        volatile __gm__ int32_t* gseq = (volatile __gm__ int32_t*)p_syncws;

        GlobalTensor<bfloat16_t> txn1P, tq2P, tKcP, tVcP, tprobs2P, tattnP, txn2P, thP, txnfP, tWlm;
        tWlm.SetGlobalBuffer(gWlm);
        GlobalTensor<float> tqkvOut, tScores, tPartAcc, toOut, tguOut, tdOut, tLogits;
        txn1P.SetGlobalBuffer(gxn1P);  tq2P.SetGlobalBuffer(gq2P);
        tKcP.SetGlobalBuffer(gKcP);    tVcP.SetGlobalBuffer(gVcP);
        tprobs2P.SetGlobalBuffer(gprobs2P);
        tattnP.SetGlobalBuffer(gattnP); txn2P.SetGlobalBuffer(gxn2P);
        thP.SetGlobalBuffer(ghP);      txnfP.SetGlobalBuffer(gxnfP);
        tqkvOut.SetGlobalBuffer(gqkv_out); tScores.SetGlobalBuffer(gscores);
        tPartAcc.SetGlobalBuffer(gpartAccP); toOut.SetGlobalBuffer(go_out);
        tguOut.SetGlobalBuffer(ggu_out);    tdOut.SetGlobalBuffer(gd_out);
        tLogits.SetGlobalBuffer(glogits);

        for (int li = 0; li < lstop; li++) {
            const int R = li * 8;
            // ---- qkv gemv (after own aiv wrote xn1P) ----
            poll_seq(gseq + SEQ_WANT(cid, R + 0), tag + R + 0);      // rms1 done
            {
                GlobalTensor<bfloat16_t> tw; tw.SetGlobalBuffer(gWqkv + (int64_t)li * NQKV_PAD * H);
                gemv_slab(C, tw, txn1P, tqkvOut, NQKV, H, NC_QKV, 2, 176, cid, NQKV_PAD,
                          NQKV_PAD);
            }
            aic_seq(gseq, SEQ_WANT(24 + cid, R + 0), tag + R + 0);   // qkv done

            {
                // ---- gemm1: scores per 256-tile (guarded busy-flag pattern,
                // exactly like gemv_slab: one k-chunk per output tile) ----
                poll_seq(gseq + SEQ_WANT(cid, R + 2), tag + R + 2);  // appends done
                if (sblk > 0) {
                    const int64_t khead = (int64_t)(li * KVH + kv) * 8 * (SPADG / 16) * 256;
                    const int i0 = s0 / 16;
                    bool l1Busy = false, l0Busy = false, l0cBusy = false;
                    for (int t = 0; t < nloc; t++) {
                        if (l0cBusy) { WaitFlag<HardEvent::FIX_M>(C.ev4); l0cBusy = false; }
                        if (l0Busy)  { WaitFlag<HardEvent::M_MTE1>(C.ev3); l0Busy = false; }
                        if (l1Busy)  { WaitFlag<HardEvent::MTE1_MTE2>(C.ev5); l1Busy = false; }
                        DataCopy(C.la1, tq2P[(int64_t)cid * 8 * 256], 8 * 256);
                        for (int j = 0; j < 8; j++) {
                            DataCopy(C.lb1[j * 16 * 256],
                                     tKcP[khead + (int64_t)j * (SPADG / 16) * 256 +
                                          (int64_t)(i0 + t * 16) * 256],
                                     16 * 256);
                        }
                        SetFlag<HardEvent::MTE2_MTE1>(C.ev0); WaitFlag<HardEvent::MTE2_MTE1>(C.ev0);
                        LoadData2DParams pa(0, 8, 1, 0, 0, false, 0);
                        LoadData(C.la2, C.la1, pa);
                        LoadData2DParams pb(0, 128, 1, 0, 0, false, 0);
                        LoadData(C.lb2, C.lb1, pb);
                        SetFlag<HardEvent::MTE1_M>(C.ev1); WaitFlag<HardEvent::MTE1_M>(C.ev1);
                        MmadParams mp; mp.m = 16; mp.n = 256; mp.k = D; mp.cmatrixInitVal = true;
                        Mmad(C.lc1, C.la2, C.lb2, mp);
                        SetFlag<HardEvent::M_MTE1>(C.ev3);
                        SetFlag<HardEvent::MTE1_MTE2>(C.ev5);
                        l0Busy = true; l1Busy = true;
                        SetFlag<HardEvent::M_FIX>(C.ev2); WaitFlag<HardEvent::M_FIX>(C.ev2);
                        FixpipeParamsV220 fp; fp.nSize = 256; fp.mSize = 16; fp.srcStride = 16;
                        fp.dstStride = STEP;
                        Fixpipe(tScores[((int64_t)cid * 16) * STEP + (int64_t)(s0 + t * SLT)],
                                C.lc1, fp);
                        SetFlag<HardEvent::FIX_MTE2>(C.ev0); WaitFlag<HardEvent::FIX_MTE2>(C.ev0);
                        SetFlag<HardEvent::FIX_M>(C.ev4);
                        l0cBusy = true;
                    }
                    if (l0Busy)  { WaitFlag<HardEvent::M_MTE1>(C.ev3); }
                    if (l1Busy)  { WaitFlag<HardEvent::MTE1_MTE2>(C.ev5); }
                    if (l0cBusy) { WaitFlag<HardEvent::FIX_M>(C.ev4); }
                }
                aic_seq(gseq, SEQ_WANT(24 + cid, R + 2), tag + R + 2);  // scores done
                // ---- gemm2: PV per tile (accumulates across tiles into ONE
                // CO1, single Fixpipe; guarded busy flags) ----
                if (sblk > 0) {
                    poll_seq(gseq + SEQ_WANT(cid, R + 3), tag + R + 3);  // probs done
                    const int64_t vhead = (int64_t)(li * KVH + kv) * (SPADG / 16) * 8 * 256;
                    const int i0 = s0 / 16;
                    bool l1Busy = false, l0Busy = false;
                    for (int t = 0; t < nloc; t++) {
                        if (l0Busy) { WaitFlag<HardEvent::M_MTE1>(C.ev3); l0Busy = false; }
                        if (l1Busy) { WaitFlag<HardEvent::MTE1_MTE2>(C.ev5); l1Busy = false; }
                        DataCopy(C.la1, tprobs2P[((int64_t)cid * NTM + t) * 16 * 256], 16 * 256);
                        for (int i = 0; i < 16; i++) {
                            DataCopy(C.lb1[i * 8 * 256],
                                     tVcP[vhead + (int64_t)(i0 + t * 16 + i) * 8 * 256],
                                     8 * 256);
                        }
                        SetFlag<HardEvent::MTE2_MTE1>(C.ev0); WaitFlag<HardEvent::MTE2_MTE1>(C.ev0);
                        LoadData2DParams pa(0, 16, 1, 0, 0, false, 0);
                        LoadData(C.la2, C.la1, pa);
                        LoadData2DParams pb(0, 128, 1, 0, 0, false, 0);
                        LoadData(C.lb2, C.lb1, pb);
                        SetFlag<HardEvent::MTE1_M>(C.ev1); WaitFlag<HardEvent::MTE1_M>(C.ev1);
                        MmadParams mp; mp.m = 16; mp.n = D; mp.k = SLT; mp.cmatrixInitVal = (t == 0);
                        Mmad(C.lc1, C.la2, C.lb2, mp);
                        SetFlag<HardEvent::M_MTE1>(C.ev3);
                        SetFlag<HardEvent::MTE1_MTE2>(C.ev5);
                        l0Busy = true; l1Busy = true;
                    }
                    if (l0Busy) { WaitFlag<HardEvent::M_MTE1>(C.ev3); }
                    if (l1Busy) { WaitFlag<HardEvent::MTE1_MTE2>(C.ev5); }
                    SetFlag<HardEvent::M_FIX>(C.ev2); WaitFlag<HardEvent::M_FIX>(C.ev2);
                    FixpipeParamsV220 fp; fp.nSize = D; fp.mSize = 16; fp.srcStride = 16;
                    fp.dstStride = D;
                    Fixpipe(tPartAcc[((int64_t)(kv * NSL + sid) * 16) * D], C.lc1, fp);
                    SetFlag<HardEvent::FIX_MTE2>(C.ev0); WaitFlag<HardEvent::FIX_MTE2>(C.ev0);
                    SetFlag<HardEvent::FIX_M>(C.ev4);
                    WaitFlag<HardEvent::FIX_M>(C.ev4);
                }
                aic_seq(gseq, SEQ_WANT(24 + cid, R + 4), tag + R + 4);  // partials done
                poll_seq(gseq + SEQ_WANT(cid, R + 5), tag + R + 5);     // B_ATT go
            }
            // ---- o gemv ----
            {
                GlobalTensor<bfloat16_t> tw; tw.SetGlobalBuffer(gWo + (int64_t)li * NO_PAD * H);
                gemv_slab(C, tw, tattnP, toOut, H, H, NC_O, 1, 96, cid, NO_PAD, NO_PAD);
            }
            aic_seq(gseq, SEQ_WANT(24 + cid, R + 5), tag + R + 5);      // o done
            // ---- gate_up gemv ----
            poll_seq(gseq + SEQ_WANT(cid, R + 6), tag + R + 6);      // xn2 done
            {
                GlobalTensor<bfloat16_t> tw; tw.SetGlobalBuffer(gWgu + (int64_t)li * NGU * H);
                gemv_slab(C, tw, txn2P, tguOut, NGU, H, NC_GU, 4, 512, cid, NGU, NGU);
            }
            aic_seq(gseq, SEQ_WANT(24 + cid, R + 6), tag + R + 6);      // gu done
            // ---- down gemv ----
            poll_seq(gseq + SEQ_WANT(cid, R + 7), tag + R + 7);      // h done
            {
                GlobalTensor<bfloat16_t> tw; tw.SetGlobalBuffer(gWd + (int64_t)li * ND_PAD * II);
                gemv_slab(C, tw, thP, tdOut, H, II, NC_D, 1, 96, cid, ND_PAD, ND_PAD);
            }
            aic_seq(gseq, SEQ_WANT(24 + cid, R + 7), tag + R + 7);      // down done
        }

        // ---- final: lm gemv + logits ----
        {
            poll_seq(gseq + SEQ_WANT(cid, 224), tag + 224);          // rmsf done
            for (int nt = 0; nt < NRD_LM; nt++) {
                const int n0 = cid * SPAN + nt * NC_LM;
                if (n0 >= VV) break;
                const int nlim = (cid * SPAN + SPAN < VV) ? (cid * SPAN + SPAN) : VV;
                const int ncEnd = (n0 + NC_LM < nlim) ? (n0 + NC_LM) : nlim;
                const int nce = ncEnd - n0;
                const int ncc = nce / 16;
                bool l1Busy = false, l0Busy = false, l0cBusy = false;
                for (int k0 = 0; k0 < H; k0 += KTILE) {
                    if (l0cBusy) { WaitFlag<HardEvent::FIX_M>(C.ev4); l0cBusy = false; }
                    if (l0Busy)  { WaitFlag<HardEvent::M_MTE1>(C.ev3); l0Busy = false; }
                    if (l1Busy)  { WaitFlag<HardEvent::MTE1_MTE2>(C.ev5); l1Busy = false; }
                    DataCopy(C.la1, txnfP[k0 * 16], 16 * KTILE);
                    for (int jj = 0; jj < KTILE / 16; jj++) {
                        const int j = k0 / 16 + jj;
                        DataCopy(C.lb1[jj * ncc * 256],
                                 tWlm[(int64_t)(j * (NB * SPAN / 16) + n0 / 16) * 256], ncc * 256);
                    }
                    SetFlag<HardEvent::MTE2_MTE1>(C.ev0); WaitFlag<HardEvent::MTE2_MTE1>(C.ev0);
                    LoadData2DParams pa(0, (uint8_t)(KTILE / 16), 1, 0, 0, false, 0);
                    LoadData(C.la2, C.la1, pa);
                    LoadData2DParams pb(0, (uint8_t)((KTILE / 16) * ncc), 1, 0, 0, false, 0);
                    LoadData(C.lb2, C.lb1, pb);
                    SetFlag<HardEvent::MTE1_M>(C.ev1); WaitFlag<HardEvent::MTE1_M>(C.ev1);
                    MmadParams mp; mp.m = 16; mp.n = nce; mp.k = KTILE; mp.cmatrixInitVal = (k0 == 0);
                    Mmad(C.lc1, C.la2, C.lb2, mp);
                    SetFlag<HardEvent::M_MTE1>(C.ev3);
                    SetFlag<HardEvent::MTE1_MTE2>(C.ev5);
                    l0Busy = true; l1Busy = true;
                }
                SetFlag<HardEvent::M_FIX>(C.ev2); WaitFlag<HardEvent::M_FIX>(C.ev2);
                FixpipeParamsV220 fp; fp.nSize = nce; fp.mSize = 16; fp.srcStride = 16;
                fp.dstStride = NB * SPANP;   // ROW stride of logits (16, NB*SPANP)!
                Fixpipe(tLogits[(int64_t)cid * SPANP + nt * NC_LM], C.lc1, fp);
                SetFlag<HardEvent::FIX_MTE2>(C.ev0); WaitFlag<HardEvent::FIX_MTE2>(C.ev0);
                SetFlag<HardEvent::FIX_M>(C.ev4);
                l0cBusy = true;
                if (l0Busy)  { WaitFlag<HardEvent::M_MTE1>(C.ev3); }
                if (l1Busy)  { WaitFlag<HardEvent::MTE1_MTE2>(C.ev5); }
                l0cBusy = false;
                WaitFlag<HardEvent::FIX_M>(C.ev4);
            }
            aic_seq(gseq, SEQ_WANT(24 + cid, 224), tag + 224);         // logits done
        }
    } else {
        // ================= VECTOR SIDE =================
        const int bid = GetBlockIdx();
        const int sub = bid & 1;
        const int cid = bid >> 1;
        const int kv = cid % KVH;
        const int sid = cid / KVH;
        const int s0 = sid * thirds;
        const int e0 = (sl + 1 < s0 + thirds) ? (sl + 1) : (s0 + thirds);
        const int sblk = (e0 > s0) ? (e0 - s0) : 0;
        const int nloc = (sblk + SLT - 1) / SLT;
        const int sidOwner = (sl / thirds < NSL) ? (sl / thirds) : (NSL - 1);
        const bool leader = (sid == 0);

        TPipe pipe;
        TBuf<TPosition::VECCALC> tXm, tXf, tXn, tPack, tQkv, tHead, tHb, tSm, tP2, tQ2,
            tCmb, tAm, tSig, tSync, tFr, tSq;
        pipe.InitBuffer(tXm, H * sizeof(bfloat16_t));            // residual carrier
        pipe.InitBuffer(tXf, 2 * H * sizeof(float));             // f32 H (val + sq)
        pipe.InitBuffer(tXn, H * sizeof(bfloat16_t));            // bf16 H out
        pipe.InitBuffer(tPack, 4096 * sizeof(bfloat16_t));       // broadcast chunk
        pipe.InitBuffer(tQkv, NQKV_PAD * sizeof(float));         // qkv row 0
        pipe.InitBuffer(tHead, 16 * 128 * sizeof(float));        // head f32 scratch
        pipe.InitBuffer(tHb, 512 * sizeof(bfloat16_t));          // head bf16 scratch
        pipe.InitBuffer(tSm, 12 * 256 * sizeof(float));          // softmax/silu f32
        pipe.InitBuffer(tP2, 16 * 256 * sizeof(bfloat16_t));     // probs2P tile
        pipe.InitBuffer(tQ2, 8 * 256 * sizeof(bfloat16_t));      // q2P tile
        pipe.InitBuffer(tCmb, 4 * 128 * sizeof(float));          // combine
        pipe.InitBuffer(tAm, 4 * 2048 * sizeof(float));          // argmax
        pipe.InitBuffer(tSig, 8192);                             // sigmoid tmp
        pipe.InitBuffer(tSync, 8 * 48 * sizeof(int32_t));        // SyncAll ub ws
        pipe.InitBuffer(tFr, 256 * sizeof(bfloat16_t));          // V append RMW
        pipe.InitBuffer(tSq, 2 * 16 * sizeof(int32_t));          // seq word (x2)
        LocalTensor<bfloat16_t> xm = tXm.Get<bfloat16_t>();
        LocalTensor<float> xf = tXf.Get<float>();
        LocalTensor<float> xs = tXf.Get<float>()[H];
        LocalTensor<bfloat16_t> xn = tXn.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> pack = tPack.Get<bfloat16_t>();
        LocalTensor<float> qkv = tQkv.Get<float>();
        LocalTensor<float> hd = tHead.Get<float>();
        LocalTensor<bfloat16_t> hb = tHb.Get<bfloat16_t>();
        LocalTensor<float> sm = tSm.Get<float>();
        LocalTensor<bfloat16_t> p2 = tP2.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> q2 = tQ2.Get<bfloat16_t>();
        LocalTensor<float> cmb = tCmb.Get<float>();
        LocalTensor<float> am = tAm.Get<float>();
        LocalTensor<uint8_t> sig = tSig.Get<uint8_t>();
        LocalTensor<int32_t> syncub = tSync.Get<int32_t>();
        LocalTensor<bfloat16_t> vfr = tFr.Get<bfloat16_t>();
        LocalTensor<int32_t> sq = tSq.Get<int32_t>();

        // named scratch views
        LocalTensor<float> q0f = hd[0], q1f = hd[128], kf = hd[256];
        LocalTensor<float> cosr = hd[640], sinr = hd[768], rot = hd[896];
        LocalTensor<float> t16 = hd[1408];
        LocalTensor<float> pml = hd[1664];                       // part_m/l loads
        LocalTensor<bfloat16_t> q0b = hb[0], q1b = hb[128], kb = hb[256], vb = hb[384];
        LocalTensor<float> sc0 = sm[0], sc1 = sm[256], ex0 = sm[512], ex1 = sm[768];
        // ping-pong reload buffers: the compiler/HW may hoist MTE2 issues past
        // PipeBarrier, so reload loops MUST alternate buffers (verified hard
        // way: silu chunks read one iteration ahead otherwise)
        LocalTensor<float> sc0b = sm[1536], sc1b = sm[1792];
        LocalTensor<float> ggu = sm[1024], uup = sm[1280];
        LocalTensor<float> ggub = sm[2048], uupb = sm[2304];
        LocalTensor<float> num = cmb[0], acc = cmb[128], t16c = cmb[256];
        LocalTensor<float> accb = cmb[384];   // ping-pong partner for acc
        LocalTensor<float> lgc = am[0], idxc = am[2048];
        LocalTensor<float> lgcb = am[4096], idxcb = am[6144];   // argmax ping-pong

        event_t eM2V = static_cast<event_t>(pipe.FetchEventID<HardEvent::MTE2_V>());
        event_t eV3 = static_cast<event_t>(pipe.FetchEventID<HardEvent::V_MTE3>());
        event_t eM3V = static_cast<event_t>(pipe.FetchEventID<HardEvent::MTE3_V>());
        event_t eM3M2 = static_cast<event_t>(pipe.FetchEventID<HardEvent::MTE3_MTE2>());
        event_t eM2S = static_cast<event_t>(pipe.FetchEventID<HardEvent::MTE2_S>());
        int sqWhich = 0;          // alternating seq-word buffer
        int s3cnt = 0;            // S_MTE3 id cycler (one-shot ids!)

        __gm__ bfloat16_t* gRms1w = (__gm__ bfloat16_t*)p_rms1w;
        __gm__ bfloat16_t* gRms2w = (__gm__ bfloat16_t*)p_rms2w;
        __gm__ bfloat16_t* gRmsfw = (__gm__ bfloat16_t*)p_rmsfw;
        __gm__ bfloat16_t* gQnw = (__gm__ bfloat16_t*)p_qnw;
        __gm__ bfloat16_t* gKnw = (__gm__ bfloat16_t*)p_knw;
        __gm__ float* gCosT = (__gm__ float*)p_cosT;
        __gm__ float* gSinT = (__gm__ float*)p_sinT;
        __gm__ bfloat16_t* gKcP = (__gm__ bfloat16_t*)p_KcP;
        __gm__ bfloat16_t* gVcP = (__gm__ bfloat16_t*)p_VcP;
        __gm__ bfloat16_t* gxn1P = (__gm__ bfloat16_t*)p_xn1P;
        __gm__ float* gqkv_out = (__gm__ float*)p_qkv_out;
        __gm__ bfloat16_t* gq2P = (__gm__ bfloat16_t*)p_q2P;
        __gm__ float* gscores = (__gm__ float*)p_scores16;
        __gm__ bfloat16_t* gprobs2P = (__gm__ bfloat16_t*)p_probs2P;
        __gm__ float* gpartm = (__gm__ float*)p_partm;
        __gm__ float* gpartl = (__gm__ float*)p_partl;
        __gm__ float* gpartAccP = (__gm__ float*)p_partAccP;
        __gm__ bfloat16_t* gattnP = (__gm__ bfloat16_t*)p_attnP;
        __gm__ float* go_out = (__gm__ float*)p_o_out;
        __gm__ bfloat16_t* gxn2P = (__gm__ bfloat16_t*)p_xn2P;
        __gm__ float* ggu_out = (__gm__ float*)p_gu_out;
        __gm__ bfloat16_t* ghP = (__gm__ bfloat16_t*)p_hP;
        __gm__ float* gd_out = (__gm__ float*)p_d_out;
        __gm__ float* glogits = (__gm__ float*)p_logits;
        __gm__ float* gIdxTab = (__gm__ float*)p_idxTab;
        __gm__ float* gPartMax = (__gm__ float*)p_partMax;
        __gm__ float* gPartIdx = (__gm__ float*)p_partIdx;
        GlobalTensor<bfloat16_t> tKcP, tVcP, txn1P, tq2P, tprobs2P, tattnP, txn2P, thP, txnfP;
        GlobalTensor<float> tQkvOut, tScores, tPartM, tPartL, tPartAcc, toOut, tguOut, tdOut,
            tLogits, tIdxTab, tPartMax, tPartIdx, tCosT, tSinT;
        GlobalTensor<bfloat16_t> tEmb;
        GlobalTensor<int32_t> gsync, gseq;
        tKcP.SetGlobalBuffer(gKcP);   tVcP.SetGlobalBuffer(gVcP);
        txn1P.SetGlobalBuffer(gxn1P); tq2P.SetGlobalBuffer(gq2P);
        tprobs2P.SetGlobalBuffer(gprobs2P); tattnP.SetGlobalBuffer(gattnP);
        txn2P.SetGlobalBuffer(gxn2P); thP.SetGlobalBuffer(ghP);
        txnfP.SetGlobalBuffer((__gm__ bfloat16_t*)p_xnfP);
        tEmb.SetGlobalBuffer((__gm__ bfloat16_t*)p_xEmb);
        tQkvOut.SetGlobalBuffer(gqkv_out); tScores.SetGlobalBuffer(gscores);
        tPartM.SetGlobalBuffer(gpartm);    tPartL.SetGlobalBuffer(gpartl);
        tPartAcc.SetGlobalBuffer(gpartAccP);
        toOut.SetGlobalBuffer(go_out);     tguOut.SetGlobalBuffer(ggu_out);
        tdOut.SetGlobalBuffer(gd_out);     tLogits.SetGlobalBuffer(glogits);
        tIdxTab.SetGlobalBuffer(gIdxTab);
        tPartMax.SetGlobalBuffer(gPartMax); tPartIdx.SetGlobalBuffer(gPartIdx);
        tCosT.SetGlobalBuffer(gCosT);      tSinT.SetGlobalBuffer(gSinT);
        gsync.SetGlobalBuffer((__gm__ int32_t*)p_syncws);
        gseq.SetGlobalBuffer((__gm__ int32_t*)p_syncws);
        volatile __gm__ int32_t* gseqp = (volatile __gm__ int32_t*)p_syncws;  // poll view

        DataCopy(xm, tEmb, H);
        SetFlag<HardEvent::MTE2_V>(eM2V); WaitFlag<HardEvent::MTE2_V>(eM2V);

        for (int li = 0; li < lstop; li++) {
            const int R = li * 8;
            // ---- stage 1: rms1 -> xn1P ----
            {
                Cast(xf, xm, RoundMode::CAST_NONE, H);
                {
                    GlobalTensor<bfloat16_t> tg; tg.SetGlobalBuffer(gRms1w + (int64_t)li * H);
                    DataCopy(pack, tg, H);   // gamma (bf16 view)
                }
                SetFlag<HardEvent::MTE2_V>(eM2V); WaitFlag<HardEvent::MTE2_V>(eM2V);
                rms_norm(xf, xs, pack, xn, H, t16);
                if (sub == 0) { pack_out(pack, xn, txn1P, H / 256, eV3, eM3V); }
            }
            if (sub == 0) { seq_write(gseq, SEQ_WANT(cid, R + 0), tag + R + 0, sq, sqWhich ^= 1, eV3, eM3V); }

            // ---- stage 2: qkv gate (poll 1 assigned AIC slot + plane
            // SyncAll: after the barrier ALL 24 AIC gemvs are confirmed) ----
            poll_seq(gseqp + SEQ_WANT(24 + (bid % 24), R + 0), tag + R + 0);
            SyncAll<false>(gsync, syncub, 48);
                SyncAll<false>(gsync, syncub, 48);

            // ---- stage 3: qk-norm, rope, appends, q2P ----
            {
                // NOTE: direct f32 loads per region -- Cast(f32->f32,
                // CAST_NONE) is BROKEN on dav-2201 (corrupts data; verified
                // by probe p13norm). f32->bf16 / bf16->f32 casts are fine.
                DataCopy(q0f, tQkvOut[(kv * 2) * D], D);
                DataCopy(q1f, tQkvOut[(kv * 2 + 1) * D], D);
                DataCopy(kf, tQkvOut[NQKV / 2 + kv * D], D);
                DataCopy(qkv, tQkvOut[3 * NQKV / 4 + kv * D], D);
                DataCopy(cosr, tCosT[(int64_t)sl * D], D);
                DataCopy(sinr, tSinT[(int64_t)sl * D], D);
                {
                    GlobalTensor<bfloat16_t> tg; tg.SetGlobalBuffer(gQnw + (int64_t)li * D);
                    DataCopy(pack, tg, D);                       // q-norm gamma
                }
                {
                    GlobalTensor<bfloat16_t> tg; tg.SetGlobalBuffer(gKnw + (int64_t)li * D);
                    DataCopy(pack[128], tg, D);                  // k-norm gamma
                }
                SetFlag<HardEvent::MTE2_V>(eM2V); WaitFlag<HardEvent::MTE2_V>(eM2V);
                Cast(vb, qkv[0], RoundMode::CAST_RINT, D);       // f32 -> bf16
                rms_norm(q0f, rot, pack, q0b, D, t16);
                rms_norm(q1f, rot, pack, q1b, D, t16);
                rms_norm(kf, rot, pack[128], kb, D, t16);
                rope_apply(q0f, cosr, sinr, rot);
                rope_apply(q1f, cosr, sinr, rot);
                rope_apply(kf, cosr, sinr, rot);
                Cast(q0b, q0f, RoundMode::CAST_RINT, D);
                Cast(q1b, q1f, RoundMode::CAST_RINT, D);
                Cast(kb, kf, RoundMode::CAST_RINT, D);
                if (sub == 0) {
                    strided_rows(q2, q0b, 8, 0);
                    strided_rows(q2, q1b, 8, 1);
                    SetFlag<HardEvent::V_MTE3>(eV3); WaitFlag<HardEvent::V_MTE3>(eV3);
                    DataCopy(tq2P[(int64_t)cid * 8 * 256], q2, 8 * 256);
                    SetFlag<HardEvent::MTE3_V>(eM3V); WaitFlag<HardEvent::MTE3_V>(eM3V);
                    if (sid == sidOwner) {
                        const int iBlk = sl / 16;
                        const int iRow = sl % 16;
                        const int64_t kbase = (int64_t)(li * KVH + kv) * 8 * (SPADG / 16) * 256;
                        const int64_t vbase = (int64_t)(li * KVH + kv) * (SPADG / 16) * 8 * 256;
                        PipeBarrier<PIPE_V>();               // vb reads below
                        for (int j = 0; j < 8; j++) {
                            DataCopy(tKcP[kbase + (int64_t)j * (SPADG / 16) * 256 +
                                          (int64_t)iBlk * 256 + iRow * 16],
                                     kb[j * 16], 16);
                        }
                        for (int j = 0; j < 8; j++) {
                            DataCopy(vfr, tVcP[vbase + (int64_t)iBlk * 8 * 256 + j * 256], 256);
                            SetFlag<HardEvent::MTE2_S>(eM2S); WaitFlag<HardEvent::MTE2_S>(eM2S);
                            for (int n = 0; n < 16; n++) {
                                vfr.SetValue(n * 16 + iRow, vb.GetValue(j * 16 + n));
                            }
                            event_t esApp = static_cast<event_t>(s3cnt++ & 3);  // one-shot ids
                            SetFlag<HardEvent::S_MTE3>(esApp); WaitFlag<HardEvent::S_MTE3>(esApp);
                            DataCopy(tVcP[vbase + (int64_t)iBlk * 8 * 256 + j * 256], vfr, 256);
                            SetFlag<HardEvent::MTE3_MTE2>(eM3M2); WaitFlag<HardEvent::MTE3_MTE2>(eM3M2);
                        }
                    }
                }
            }
            if (sub == 0) { seq_write(gseq, SEQ_WANT(cid, R + 2), tag + R + 2, sq, sqWhich ^= 1, eV3, eM3V); }

            {
                // ---- stage 4: two-pass softmax -> probs2P, part_m/l ----
                // (full gate: poll + SyncAll -- fixpipe output needs the
                // barrier delay to land in L2; a bare poll is NOT enough,
                // verified by probe p15vis)
                poll_seq(gseqp + SEQ_WANT(24 + (bid % 24), R + 2), tag + R + 2);
                SyncAll<false>(gsync, syncub, 48);
                SyncAll<false>(gsync, syncub, 48);
                float m0 = NEG_INF, m1 = NEG_INF, l0 = 0.0f, l1 = 0.0f;
                const float scale = 0.08838834764831845f;  // 1/sqrt(128)
                if (sblk > 0) {
                    for (int t = 0; t < nloc; t++) {       // pass 1: max
                        const int tail = sblk - t * SLT;
                        LocalTensor<float> s0v = (t & 1) ? sc0b : sc0;
                        LocalTensor<float> s1v = (t & 1) ? sc1b : sc1;
                        DataCopy(s0v, tScores[((int64_t)cid * 16) * STEP + (int64_t)(s0 + t * SLT)], 256);
                        DataCopy(s1v, tScores[((int64_t)cid * 16 + 1) * STEP + (int64_t)(s0 + t * SLT)], 256);
                        SetFlag<HardEvent::MTE2_V>(eM2V); WaitFlag<HardEvent::MTE2_V>(eM2V);
                        Muls(s0v, s0v, scale, 256);
                        Muls(s1v, s1v, scale, 256);
                        if (tail >= 256) {
                            float tm = tree_max(s0v, 256); if (tm > m0) m0 = tm;
                            tm = tree_max(s1v, 256); if (tm > m1) m1 = tm;
                        } else {
                            PipeBarrier<PIPE_V>();
                            for (int i = 0; i < tail; i++) {
                                float x = s0v.GetValue(i); if (x > m0) m0 = x;
                                x = s1v.GetValue(i); if (x > m1) m1 = x;
                            }
                        }
                    }
                    for (int t = 0; t < nloc; t++) {       // pass 2: exp/probs/l
                        const int tail = sblk - t * SLT;
                        LocalTensor<float> s0v = (t & 1) ? sc0b : sc0;
                        LocalTensor<float> s1v = (t & 1) ? sc1b : sc1;
                        DataCopy(s0v, tScores[((int64_t)cid * 16) * STEP + (int64_t)(s0 + t * SLT)], 256);
                        DataCopy(s1v, tScores[((int64_t)cid * 16 + 1) * STEP + (int64_t)(s0 + t * SLT)], 256);
                        SetFlag<HardEvent::MTE2_V>(eM2V); WaitFlag<HardEvent::MTE2_V>(eM2V);
                        Muls(s0v, s0v, scale, 256);
                        Muls(s1v, s1v, scale, 256);
                        Adds(s0v, s0v, -m0, 256);
                        Adds(s1v, s1v, -m1, 256);
                        Exp(ex0, s0v, 256);
                        Exp(ex1, s1v, 256);
                        // land probs BEFORE tree_sum (it destroys ex0 in place!)
                        Cast(hb[0], ex0, RoundMode::CAST_RINT, 256);    // head0 bf16
                        Cast(hb[256], ex1, RoundMode::CAST_RINT, 256);  // head1 bf16
                        if (tail >= 256) {
                            l0 += tree_sum(ex0, 256);
                            l1 += tree_sum(ex1, 256);
                        } else {
                            PipeBarrier<PIPE_V>();
                            for (int i = 0; i < tail; i++) { l0 += ex0.GetValue(i); }
                            for (int i = 0; i < tail; i++) { l1 += ex1.GetValue(i); }
                        }
                        if (sub == 0) {
                            strided_rows(p2, hb[0], 16, 0);
                            strided_rows(p2, hb[256], 16, 1);
                            SetFlag<HardEvent::V_MTE3>(eV3); WaitFlag<HardEvent::V_MTE3>(eV3);
                            DataCopy(tprobs2P[((int64_t)cid * NTM + t) * 16 * 256], p2, 16 * 256);
                            SetFlag<HardEvent::MTE3_V>(eM3V); WaitFlag<HardEvent::MTE3_V>(eM3V);
                        }
                    }
                }
                if (sub == 0) {
                    put_scalar16(tPartM, (kv * 2) * NSL + sid, m0, t16, eV3, eM3V);
                    put_scalar16(tPartM, (kv * 2 + 1) * NSL + sid, m1, t16, eV3, eM3V);
                    put_scalar16(tPartL, (kv * 2) * NSL + sid, l0, t16, eV3, eM3V);
                    put_scalar16(tPartL, (kv * 2 + 1) * NSL + sid, l1, t16, eV3, eM3V);
                }
                if (sub == 0) { seq_write(gseq, SEQ_WANT(cid, R + 3), tag + R + 3, sq, sqWhich ^= 1, eV3, eM3V); }

                // ---- stage 5: combine (full gate: poll + SyncAll, then
                // leaders read part_m/l/partAcc -- the partAcc fixpipe
                // output needs the barrier delay; part_m/l are AIV-written
                // and MTE3-ordered behind the R+3 polls transitively) ----
                poll_seq(gseqp + SEQ_WANT(24 + (bid % 24), R + 4), tag + R + 4);
                SyncAll<false>(gsync, syncub, 48);
                SyncAll<false>(gsync, syncub, 48);
                if (sub == 0 && leader) {
                    // combine heads 2kv, 2kv+1 over 3 slice partials
                    DataCopy(pml, tPartM[(int64_t)(kv * 2) * NSL * 16], 6 * 16);
                    SetFlag<HardEvent::MTE2_S>(eM2S); WaitFlag<HardEvent::MTE2_S>(eM2S);
                    float mm[2][NSL], ll[2][NSL];
                    for (int h = 0; h < 2; h++) {
                        for (int s = 0; s < NSL; s++) {
                            mm[h][s] = pml.GetValue((h * NSL + s) * 16);
                        }
                    }
                    DataCopy(pml, tPartL[(int64_t)(kv * 2) * NSL * 16], 6 * 16);
                    SetFlag<HardEvent::MTE2_S>(eM2S); WaitFlag<HardEvent::MTE2_S>(eM2S);
                    for (int h = 0; h < 2; h++) {
                        for (int s = 0; s < NSL; s++) {
                            ll[h][s] = pml.GetValue((h * NSL + s) * 16);
                        }
                    }
                    for (int h = 0; h < 2; h++) {
                        float mg = mm[h][0];
                        for (int s = 1; s < NSL; s++) { if (mm[h][s] > mg) mg = mm[h][s]; }
                        float w[NSL], den = 0.0f;
                        for (int s = 0; s < NSL; s++) {
                            w[s] = vexp_scalar(t16c, mm[h][s] - mg);
                            den += w[s] * ll[h][s];
                        }
                        Duplicate(num, 0.0f, D);
                        for (int s = 0; s < NSL; s++) {
                            if (w[s] == 0.0f) continue;
                            LocalTensor<float> av = (s & 1) ? accb : acc;
                            DataCopy(av, tPartAcc[((int64_t)(kv * NSL + s) * 16 + h) * D], D);
                            SetFlag<HardEvent::MTE2_V>(eM2V); WaitFlag<HardEvent::MTE2_V>(eM2V);
                            Muls(av, av, w[s], D);
                            Add(num, num, av, D);
                        }
                        Muls(num, num, 1.0f / den, D);
                        Cast(hb[h * 128], num, RoundMode::CAST_RINT, D);
                    }
                    bcast_pack(pack, hb[0]);
                    SetFlag<HardEvent::V_MTE3>(eV3); WaitFlag<HardEvent::V_MTE3>(eV3);
                    DataCopy(tattnP[(int64_t)(kv * 2 * D) * 16], pack, 4096);
                    SetFlag<HardEvent::MTE3_V>(eM3V); WaitFlag<HardEvent::MTE3_V>(eM3V);
                }
                SyncAll<false>(gsync, syncub, 48);
                if (sub == 0) { seq_write(gseq, SEQ_WANT(cid, R + 5), tag + R + 5, sq, sqWhich ^= 1, eV3, eM3V); }
            }

            // ---- stage 6: resid1 + rms2 -> xn2P ----
            {
                poll_seq(gseqp + SEQ_WANT(24 + (bid % 24), R + 5), tag + R + 5);  // o done
                SyncAll<false>(gsync, syncub, 48);
                SyncAll<false>(gsync, syncub, 48);
                DataCopy(xf, toOut, H);                      // o row 0 (f32)
                SetFlag<HardEvent::MTE2_V>(eM2V); WaitFlag<HardEvent::MTE2_V>(eM2V);
                Cast(xs, xm, RoundMode::CAST_NONE, H);
                Add(xs, xs, xf, H);
                Cast(xm, xs, RoundMode::CAST_RINT, H);
                Cast(xf, xm, RoundMode::CAST_NONE, H);
                {
                    GlobalTensor<bfloat16_t> tg; tg.SetGlobalBuffer(gRms2w + (int64_t)li * H);
                    DataCopy(pack, tg, H);
                }
                SetFlag<HardEvent::MTE2_V>(eM2V); WaitFlag<HardEvent::MTE2_V>(eM2V);
                rms_norm(xf, xs, pack, xn, H, t16);
                if (sub == 0) { pack_out(pack, xn, txn2P, H / 256, eV3, eM3V); }
            }
            if (sub == 0) { seq_write(gseq, SEQ_WANT(cid, R + 6), tag + R + 6, sq, sqWhich ^= 1, eV3, eM3V); }

            // ---- stage 7: silu -> hP ----
            {
                poll_seq(gseqp + SEQ_WANT(24 + (bid % 24), R + 6), tag + R + 6);  // gu done
                SyncAll<false>(gsync, syncub, 48);
                SyncAll<false>(gsync, syncub, 48);
                const int chunks = II / 256;
                for (int c = 0; c < chunks; c++) {
                    LocalTensor<float> gv = (c & 1) ? ggub : ggu;
                    LocalTensor<float> uv = (c & 1) ? uupb : uup;
                    DataCopy(gv, tguOut[c * 256], 256);
                    DataCopy(uv, tguOut[II + c * 256], 256);
                    SetFlag<HardEvent::MTE2_V>(eM2V); WaitFlag<HardEvent::MTE2_V>(eM2V);
                    Sigmoid(ex0, gv, sig, 256);
                    Mul(gv, gv, ex0, 256);                   // g * sig(g)
                    Mul(gv, gv, uv, 256);                    // silu(g) * u
                    Cast(hb[0], gv, RoundMode::CAST_RINT, 256);
                    if (sub == 0) {
                        bcast_pack(pack, hb[0]);
                        SetFlag<HardEvent::V_MTE3>(eV3); WaitFlag<HardEvent::V_MTE3>(eV3);
                        DataCopy(thP[(int64_t)c * 4096], pack, 4096);
                        SetFlag<HardEvent::MTE3_V>(eM3V); WaitFlag<HardEvent::MTE3_V>(eM3V);
                    }
                }
            }
            if (sub == 0) { seq_write(gseq, SEQ_WANT(cid, R + 7), tag + R + 7, sq, sqWhich ^= 1, eV3, eM3V); }

            // ---- stage 8: resid2 ----
            {
                poll_seq(gseqp + SEQ_WANT(24 + (bid % 24), R + 7), tag + R + 7);  // down done
                SyncAll<false>(gsync, syncub, 48);
                SyncAll<false>(gsync, syncub, 48);
                DataCopy(xf, tdOut, H);                      // down row 0 (f32)
                SetFlag<HardEvent::MTE2_V>(eM2V); WaitFlag<HardEvent::MTE2_V>(eM2V);
                Cast(xs, xm, RoundMode::CAST_NONE, H);
                Add(xs, xs, xf, H);
                Cast(xm, xs, RoundMode::CAST_RINT, H);
            }
        }

        // ---- final: rmsf -> xnfP, argmax ----
        {
            Cast(xf, xm, RoundMode::CAST_NONE, H);
            {
                GlobalTensor<bfloat16_t> tg; tg.SetGlobalBuffer(gRmsfw);
                DataCopy(pack, tg, H);
            }
            SetFlag<HardEvent::MTE2_V>(eM2V); WaitFlag<HardEvent::MTE2_V>(eM2V);
            rms_norm(xf, xs, pack, xn, H, t16);
            if (sub == 0) { pack_out(pack, xn, txnfP, H / 256, eV3, eM3V); }
        }
        if (sub == 0) { seq_write(gseq, SEQ_WANT(cid, 224), tag + 224, sq, sqWhich ^= 1, eV3, eM3V); }

        {
            // full gate: poll + SyncAll (lm fixpipe output must land first)
            poll_seq(gseqp + SEQ_WANT(24 + (bid % 24), 224), tag + 224);
            SyncAll<false>(gsync, syncub, 48);
                SyncAll<false>(gsync, syncub, 48);
            for (int ch = 0; ch < NCH; ch++) {          // argmax over own span
                // ping-pong buffers; NO reload (MTE2 issues can be hoisted
                // past PipeBarrier -- never reload a tensor V ops still read)
                LocalTensor<float> base = (ch & 1) ? lgcb : lgc;
                LocalTensor<float> work = (ch & 1) ? lgc : lgcb;
                LocalTensor<float> idv = (ch & 1) ? idxcb : idxc;
                DataCopy(base, tLogits[(int64_t)cid * SPANP + ch * CH], CH);
                DataCopy(idv, tIdxTab[ch * CH], CH);
                SetFlag<HardEvent::MTE2_V>(eM2V); WaitFlag<HardEvent::MTE2_V>(eM2V);
                Add(work, base, base, CH);               // 2x (exact)
                float cm = 0.5f * tree_max(work, CH);    // max(base), work destroyed
                Adds(base, base, -cm, CH);
                Muls(base, base, 1e30f, CH);
                Sigmoid(base, base, sig, CH);
                Add(base, base, base, CH);               // 2*sigmoid
                Mul(base, base, idv, CH);
                float ci = tree_max(base, CH);
                if (sub == 0) {
                    put_scalar16(tPartMax, cid * NCH + ch, cm, t16, eV3, eM3V);
                    put_scalar16(tPartIdx, cid * NCH + ch, ci, t16, eV3, eM3V);
                }
            }
        }
    }
}

extern "C" void run_megak(void* ffts, void* wqkv, void* wo, void* wgu, void* wd, void* wlm,
    void* rms1w, void* rms2w, void* rmsfw, void* qnw, void* knw, void* cosT, void* sinT,
    void* KcP, void* VcP, void* xEmb, void* xn1P, void* qkv_out, void* q2P, void* scores16,
    void* probs2P, void* partm, void* partl, void* partAccP, void* attnP, void* o_out,
    void* xn2P, void* gu_out, void* hP, void* d_out, void* logits, void* idxTab,
    void* partMax, void* partIdx, void* syncws, void* xnfP,
    int sl, int lstop, int tag, int grid, void* stream) {
    megak<<<grid, nullptr, stream>>>(static_cast<uint8_t*>(ffts), static_cast<uint8_t*>(wqkv),
        static_cast<uint8_t*>(wo), static_cast<uint8_t*>(wgu), static_cast<uint8_t*>(wd),
        static_cast<uint8_t*>(wlm), static_cast<uint8_t*>(rms1w), static_cast<uint8_t*>(rms2w),
        static_cast<uint8_t*>(rmsfw), static_cast<uint8_t*>(qnw), static_cast<uint8_t*>(knw),
        static_cast<uint8_t*>(cosT), static_cast<uint8_t*>(sinT), static_cast<uint8_t*>(KcP),
        static_cast<uint8_t*>(VcP), static_cast<uint8_t*>(xEmb), static_cast<uint8_t*>(xn1P),
        static_cast<uint8_t*>(qkv_out), static_cast<uint8_t*>(q2P), static_cast<uint8_t*>(scores16),
        static_cast<uint8_t*>(probs2P), static_cast<uint8_t*>(partm), static_cast<uint8_t*>(partl),
        static_cast<uint8_t*>(partAccP), static_cast<uint8_t*>(attnP), static_cast<uint8_t*>(o_out),
        static_cast<uint8_t*>(xn2P), static_cast<uint8_t*>(gu_out), static_cast<uint8_t*>(hP),
        static_cast<uint8_t*>(d_out), static_cast<uint8_t*>(logits), static_cast<uint8_t*>(idxTab),
        static_cast<uint8_t*>(partMax), static_cast<uint8_t*>(partIdx), static_cast<uint8_t*>(syncws),
        static_cast<uint8_t*>(xnfP), sl, lstop, tag);
}
