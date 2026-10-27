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
//   * GEMV A inputs: two compact ND rows per 256-element chunk; MTE2
//     expands rows 0/1 into the L1 fractal. Row 1 remains the sentinel row.
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
//     need MTE2_S; vector/scalar scratch reuse needs V_S and S_V events.
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
#ifndef MEGA_LM_TILE
#define MEGA_LM_TILE 96
#endif
#ifndef MEGA_LM_STREAM
#define MEGA_LM_STREAM 1
#endif
constexpr int PR_LM = 234;       // final-head chunk progress; tag + chunk + 1
#ifndef MEGA_COMPACT_A
#define MEGA_COMPACT_A 1
#endif
#ifndef MEGA_GU_INTERLEAVE
#define MEGA_GU_INTERLEAVE 1
#endif
constexpr int A_PACK_CHUNK = MEGA_COMPACT_A ? 512 : 4096;
constexpr int A_PACK_ROWS = MEGA_COMPACT_A ? 2 : 16;
constexpr int NC_LM = MEGA_LM_TILE;
static_assert(NC_LM > 0 && NC_LM <= 128 && NC_LM % 16 == 0,
              "LM tile must fit the 64KB weight ring slot");
constexpr int NRD_LM = (SPAN + NC_LM - 1) / NC_LM;
constexpr int SPANP = 8192;
constexpr int CH = 2048;
constexpr int NCH = SPAN / CH + 1;    // 4 (last chunk 192 valid)
constexpr int NTM = STEP / SLT;       // 54 max tiles per slice
constexpr float EPS = 1e-6f;
constexpr float NEG_INF = -1e30f;
// chunk-major weight pack: elements per layer per gemv
//   = NB * nrd * nk * (16 kfrac * maxncc nfrac * 256)
constexpr int QKV_LAYER = NB * 2 * (H / KTILE) * 16 * (NC_QKV / 16) * 256;   // 9437184
constexpr int O_LAYER = NB * 1 * (H / KTILE) * 16 * (NC_O / 16) * 256;       // = NO_PAD*H
constexpr int GU_LAYER = NB * 4 * (H / KTILE) * 16 * (NC_GU / 16) * 256;     // = NGU*H
constexpr int D_LAYER = NB * 1 * (II / KTILE) * 16 * (NC_D / 16) * 256;      // = ND_PAD*II
constexpr int LM_LAYER = NB * NRD_LM * (H / KTILE) * 16 * (NC_LM / 16) * 256;  // includes per-slab N-tile padding

// seq workspace layout (int32), carved out of p_syncws:
//   [0, 1536)                SyncAll GM ws (48 x 8) -- unused in v2
//   [4096 ...]               rows 0..23 AIV slots, 24..47 AIC slots;
//                           slot = 16 int32 at SEQ_WANT(row, r)
//                           = 4096 + (row*256 + r)*16  (max 196608)
constexpr int SEQ_OFF = 4096;
constexpr int SEQ_ROUNDS = 256;        // 8*28 + finals headroom
// Double-buffer the previous layer's head maxima. A current-layer
// slice must never overwrite the shift before another slice has read it.
constexpr int SHIFT_HISTORY = SEQ_OFF + 48 * SEQ_ROUNDS * 16 + 4096;
__aicore__ inline int SEQ_WANT(int cid, int r) { return SEQ_OFF + (cid * 256 + r) * 16; }

// ---- v7 fine-grained producer/consumer progress words (see OPTLOG R6) ----
// One dedicated seq round per channel.  The producer stores
//     value = tag + li*PROG_SPAN + count
// and the consumer tests `>= want`, so a consumer on a COARSE grain (e.g. a
// 1024-position softmax chunk = 4 score tiles) can proceed the instant that
// chunk's producer step lands, instead of waiting for the whole stage.  This
// turns a phase barrier into a streaming hand-off (Mirage-style grain
// interleave) and is what lets gemm2's PV stream overlap the softmax tail.
// PROG_SPAN=16 keeps li*PROG_SPAN+count < 512 = the per-launch tag step, so
// each word stays strictly monotonic across launches: a stale word from a
// previous token can never satisfy `>= want` (the old count is >512 behind).
constexpr int PROG_SPAN = 16;          // grains/layer (softmax chunks <= 14)
constexpr int PR_PROBS = 232;          // round idx: AIV->AIC probs chunks landed
constexpr int PR_SCORES = 233;         // round idx: AIC->AIV score groups landed
// (232/233 sit above the per-layer rounds 0..223 and the 224 finals round)
// R13 (2a): the scores->softmax hand-off is streamed per 4096-position group
// (16 score tiles) instead of one whole-slice barrier, so pass 1's max overlaps
// gemm1's K stream.  Numerics are unchanged: the slice max is the max of the
// same 4096-wide reductions, merged in the same order.

// ---- v2 sentinel gates (replace poll+2xSyncAll; see OPTLOG/README) ----
// AIC arms a DEAD-ROW word in its own fixpipe region with SENT at layer
// start, then (after the gate's fixpipes) self-polls it until the data lands
// (!= SENT), then publishes its seq word at SEQ_WANT(24+cid, r). Consumers
// poll the seq word(s) -- single word for 1:1 gates, the [c0,c1) range for
// all-24 gates. No plane SyncAll anywhere.
// R27 generality test hook: build with -DMEGA_SCORE_BUMP=<int> to add that
// constant to every scaled score.  Softmax is shift invariant, so the token
// must be unchanged -- but a shift-0 build with a bump > 88 MUST produce inf
// (that is the whole point: it proves the carried shift, not a lucky constant,
// is what keeps the exponent in range).  Off by default, zero cost.
#ifndef MEGA_SCORE_BUMP
#define MEGA_SCORE_BUMP 0
#endif
constexpr int SENT = 0x7FC00000;       // qNaN bit pattern: never a real f32 GEMV value

// Optional diagnostic timestamps. Each traced core owns complete 128B lines.
// Stored in the existing extra sync workspace; disabled builds emit no stamps.
#ifndef MEGA_TIMELINE
#define MEGA_TIMELINE 0
#endif
#if MEGA_TIMELINE
constexpr int TRACE_BASE = SEQ_OFF + 48 * SEQ_ROUNDS * 16;
__aicore__ inline void mega_stamp(GM_ADDR workspace, int plane, int layer, int event) {
    volatile __gm__ int32_t* p = reinterpret_cast<volatile __gm__ int32_t*>(workspace)
        + TRACE_BASE + (plane * (L + 1) + layer) * 32 + event;
    *p = static_cast<int32_t>(GetSystemCycle());
    __asm__ __volatile__("" ::: "memory");
    dcci(reinterpret_cast<__gm__ uint64_t*>((__gm__ int32_t*)p),
         cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
    __asm__ __volatile__("" ::: "memory");
}
#define TRACE_C(layer, event) do { if ((cid % 8) == 0) mega_stamp(p_syncws, cid / 8, layer, event); } while (0)
#define TRACE_V(layer, event) do { if (cid == 0 && sub == 0) mega_stamp(p_syncws, 3, layer, event); } while (0)
#else
#define TRACE_C(layer, event) do {} while (0)
#define TRACE_V(layer, event) do {} while (0)
#endif

// ---------------- cube helpers ----------------
struct CubeCtx {
    event_t ev0, ev1, ev2, ev3, ev4, ev5;
    event_t ev6, ev7;              // second MTE2_MTE1 / MTE1_MTE2 (ping-pong parity 1)
    LocalTensor<bfloat16_t> la1, lb1, la2, lb2;
    LocalTensor<bfloat16_t> la1b, lb1b;   // second L1 buffer set (parity 1)
    LocalTensor<float> lc1;
    // ---- v5 prefetch ring: static B-side weights streamed during gate waits ----
    event_t evA;                   // A-pack group load
    LocalTensor<bfloat16_t> abuf;  // A-pack group staging (8 k-chunks)
    event_t eM2R[6], eM1R[6];      // ring MTE2_MTE1 / MTE1_MTE2 (distinct channels)
    LocalTensor<bfloat16_t> ringb[6];
    int cid;
    int64_t prodCount, consCount;  // ring cursors (reset per gemv)
};

// Ring depth (chunks).  v10: 6 x 64KB = 384KB, aliasing the gemm1/gemm2 K/V
// buffers b1/b1b (idle while a weight gemv runs) so no extra L1 is needed;
// during a gate wait MTE2 can now keep 6 chunks (384KB) in flight instead of 4.
constexpr int RN = 6;

// Per-gemv framing (cursors live in CubeCtx so the ring can be filled before
// the gate opens, then consumed by gemv_ring).
struct GemvCtx {
    int nk, nrd, nrdLive, maxncc, nc, slab, nGlobal;
    int64_t chunkElems, total;
    int prodNt, prodKc;
};

// AIC: poll a GM seq word until it equals want (dcci + volatile read).
__aicore__ inline void poll_seq(volatile __gm__ int32_t* p, int32_t want) {
    while (*p != want) {
        dcci(reinterpret_cast<__gm__ uint64_t*>((__gm__ int32_t*)p),
             cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
        __asm__ __volatile__("" ::: "memory");
    }
}

// AIC: poll a progress word until it reaches at least `want` (>= test).  Used
// for the streaming probs hand-off: gemm2 only needs chunk c (4 tiles) to
// start, not the whole softmax stage.
__aicore__ inline void aic_wait_prog(volatile __gm__ int32_t* p, int32_t want) {
    while (*p < want) {
        dcci(reinterpret_cast<__gm__ uint64_t*>((__gm__ int32_t*)p),
             cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
        __asm__ __volatile__("" ::: "memory");
    }
    __asm__ __volatile__("" ::: "memory");
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

// AIC: arm a sentinel word in its own fixpipe region (dead row). Done at
// layer start, long before the fixpipe, so the store lands first.
__aicore__ inline void aic_arm(volatile __gm__ float* p) {
    *(volatile __gm__ int32_t*)p = SENT;
    __asm__ __volatile__("");
    dcci(reinterpret_cast<__gm__ uint64_t*>((__gm__ int32_t*)p),
         cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
    __asm__ __volatile__("" ::: "memory");
}

// AIC: after the gate's fixpipes, poll the armed word until data (!= SENT)
// lands in L2. Detection of the actual landing -- replaces the SyncAll delay.
__aicore__ inline void aic_wait_land(volatile __gm__ float* p) {
    // FIX_M only retires the L0C source. Scalar publication additionally
    // depends on completion of the outstanding FIX destination writes.
    SetFlag<HardEvent::FIX_S>((event_t)0);
    WaitFlag<HardEvent::FIX_S>((event_t)0);
    while (*(volatile __gm__ int32_t*)p == SENT) {
        dcci(reinterpret_cast<__gm__ uint64_t*>((__gm__ int32_t*)p),
             cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
        __asm__ __volatile__("" ::: "memory");
    }
}

// Ring push: stream one B chunk into slot (prodCount%RN); wait the slot's
// MTE1_MTE2 free-flag first if the slot is being recycled.
__aicore__ inline void ring_push(CubeCtx& c, GlobalTensor<bfloat16_t>& wBase,
    int64_t chunkOff, int64_t chunkElems) {
    const int b = (int)(c.prodCount % RN);
    if (c.prodCount >= RN) { WaitFlag<HardEvent::MTE1_MTE2>(c.eM1R[b]); }
    DataCopy(c.ringb[b], wBase[chunkOff], chunkElems);
    SetFlag<HardEvent::MTE2_MTE1>(c.eM2R[b]);
    c.prodCount++;
}

// Frame the gemv and reset the ring cursors (previous gemv drained its slots).
__aicore__ inline void gemv_begin(CubeCtx& c, GemvCtx& s, int nGlobal, int kTotal,
    int nc, int nrd, int slab) {
    s.nGlobal = nGlobal; s.nc = nc; s.nrd = nrd; s.slab = slab;
    s.nk = kTotal / KTILE; s.maxncc = nc / 16;
    s.chunkElems = (int64_t)(KTILE / 16) * s.maxncc * 256;
    s.nrdLive = 0;
    for (int nt = 0; nt < nrd; nt++) { if (c.cid * slab + nt * nc >= nGlobal) break; s.nrdLive++; }
    s.total = (int64_t)s.nrdLive * s.nk;
    s.prodNt = 0; s.prodKc = 0;
    c.prodCount = 0; c.consCount = 0;
}

// Top the ring up to RN chunks (call before the gate opens and after each
// consume). Only B is prefetched -- the A-pack is gate-dependent.
__aicore__ inline void ring_fill(CubeCtx& c, GemvCtx& s, GlobalTensor<bfloat16_t>& wBase) {
    while (c.prodCount - c.consCount < RN && c.prodCount < s.total) {
        const int64_t off = ((int64_t)c.cid * s.nrd + s.prodNt) * s.nk + s.prodKc;
        ring_push(c, wBase, off * s.chunkElems, s.chunkElems);
        if (++s.prodKc == s.nk) { s.prodKc = 0; s.prodNt++; }
    }
}

// ---- R16: probs A-side load ------------------------------------------------
// The gemm2 A fractal is 16 rows but only rows 0,1 are live (the two Q heads of
// this kv head); the fixpipe keeps rows 0..2 and row 2 only feeds the sentinel.
// So the AIV stores probs COMPACT -- [tile][head0 256][head1 256] bf16 = 1KB/tile
// -- and the cube's MTE2 expands them into the L1 fractal with Nd2Nz, leaving
// rows 2..15 stale (finite garbage from the previous tile/K data).  This cuts
// the probs round trip 8x: ~568MB -> ~72MB per 40k token.
// dst layout check: dstNzC0Stride is in C0_size (16 bf16) units, so 16 -> 256
// elements between consecutive k-blocks = exactly the A-fractal stride, and
// nValue=2 writes block rows 0,1 (= the 32B rows the old strided_rows produced).
__aicore__ inline void nd2nz_probs(LocalTensor<bfloat16_t> dst,
    GlobalTensor<bfloat16_t> src, int64_t off) {
    Nd2NzParams p;
    p.ndNum = 1;
    p.nValue = 2;                  // two live heads
    p.dValue = SLT;                // 256 positions per tile
    p.srcNdMatrixStride = 0;
    p.srcDValue = SLT;
    p.dstNzC0Stride = 16;
    p.dstNzNStride = 1;
    p.dstNzMatrixStride = 0;
    DataCopy(dst, src[off], p);
}

// Retain two live A rows: output row 0 and the sentinel row 1.
__aicore__ inline void load_gemv_a(LocalTensor<bfloat16_t> dst,
    GlobalTensor<bfloat16_t> src, int64_t chunk) {
#if MEGA_COMPACT_A
    Nd2NzParams p;
    p.ndNum = 8; p.nValue = 2; p.dValue = KTILE;
    p.srcNdMatrixStride = 512; p.srcDValue = KTILE;
    p.dstNzC0Stride = 16; p.dstNzNStride = 1;
    p.dstNzMatrixStride = 4096;
    DataCopy(dst, src[chunk * A_PACK_CHUNK], p);
#else
    DataCopy(dst, src[chunk * A_PACK_CHUNK], 8 * 16 * KTILE);
#endif
}

// GEMV over one weight slab set, consuming the B-side from the prefetch ring.
// W is CHUNK-MAJOR packed: [layer][cid][nt][kc][16 kfrac][maxncc nfrac][256],
// Ragged tiles are padded to maxncc; Fixpipe trims to nce. With compact A,
// xPacked stores two ND rows per chunk; load_gemv_a expands the L1 fractal.
// Output row 0 is live and row 1 provides the landing sentinel.
//
// v5: the static B-side stream is decoupled from the gate. The caller runs
// gemv_begin()+ring_fill() BEFORE its gate poll, so MTE2 streams the weights
// while the AIC would otherwise spin. gemv_ring then consumes the ring and
// tops it up inline. The A-pack (same for all nt) is staged in L1 in groups of
// 8 k-chunks; a group is loaded at gate-open and later groups reloaded at
// their boundary (nk>8 only: the down gemv).
//
// A-group channel 5 carries ready and buffer-free events in opposite
// directions. A free event is produced/consumed only between K groups.
// Weight ring slots use channels 1..4,6,7; their final free events are drained.
__aicore__ inline void gemv_ring(CubeCtx& c, GemvCtx& s, GlobalTensor<bfloat16_t> wBase,
    GlobalTensor<bfloat16_t> xPacked, GlobalTensor<float> outBase, int outStride, int outSlab,
    volatile __gm__ int32_t* inputSeq = nullptr, int inputRound = 0, int inputWant = 0,
    volatile __gm__ int32_t* outputSeq = nullptr,
    volatile __gm__ float* outputRow = nullptr, int outputTag = 0) {
    const int KCC = KTILE / 16;
    const int nk = s.nk;
    const int maxncc = s.maxncc;
    // A-pack group 0 (chunks 0..7) staged up-front (after the gate has opened)
    // Down consumes 8 K chunks at a time; wait only for that A group.
    if (inputSeq != nullptr) {
        for (int cc = 0; cc < 8; cc++)
            poll_seq(inputSeq + SEQ_WANT(cc, inputRound), inputWant);
    }
    load_gemv_a(c.abuf, xPacked, 0);
    SetFlag<HardEvent::MTE2_MTE1>(c.evA);
    WaitFlag<HardEvent::MTE2_MTE1>(c.evA);
    bool l0cBusy = false;
#if MEGA_LM_STREAM
    int publishedChunks = 0;
#endif
    for (int nt = 0; nt < s.nrdLive; nt++) {
        const int n0 = c.cid * s.slab + nt * s.nc;
        const int nlim = (c.cid * s.slab + s.slab < s.nGlobal) ? (c.cid * s.slab + s.slab) : s.nGlobal;
        const int ncEnd = (n0 + s.nc < nlim) ? (n0 + s.nc) : nlim;
        const int nce = ncEnd - n0;
        bool l0Busy = false;
        for (int i = 0; i < nk; i++) {
            if ((i & 7) == 0 && i > 0) {            // next A group (down: 3 groups)
                // The previous group's final LoadData must finish reading
                // abuf before MTE2 overwrites it with the next K group.
                WaitFlag<HardEvent::MTE1_MTE2>(c.evA);
                if (inputSeq != nullptr) {
                    for (int cc = i; cc < i + 8; cc++)
                        poll_seq(inputSeq + SEQ_WANT(cc, inputRound), inputWant);
                }
                load_gemv_a(c.abuf, xPacked, i);
                SetFlag<HardEvent::MTE2_MTE1>(c.evA);
                WaitFlag<HardEvent::MTE2_MTE1>(c.evA);
            }
            const int b = (int)(c.consCount % RN);
            WaitFlag<HardEvent::MTE2_MTE1>(c.eM2R[b]);                 // B chunk landed
            if (l0cBusy) { WaitFlag<HardEvent::FIX_M>(c.ev4); l0cBusy = false; }
            if (l0Busy)  { WaitFlag<HardEvent::M_MTE1>(c.ev3); l0Busy = false; }
            LoadData2DParams pa(0, (uint8_t)KCC, 1, 0, 0, false, 0);
            LoadData(c.la2, c.abuf[(i & 7) * (16 * KTILE)], pa);
            LoadData2DParams pb(0, (uint8_t)(KCC * maxncc), 1, 0, 0, false, 0);
            LoadData(c.lb2, c.ringb[b], pb);
            SetFlag<HardEvent::MTE1_MTE2>(c.eM1R[b]);                  // ring slot free
            if ((i & 7) == 7 && i + 1 < nk) {
                SetFlag<HardEvent::MTE1_MTE2>(c.evA);  // A-group buffer free
            }
            SetFlag<HardEvent::MTE1_M>(c.ev1); WaitFlag<HardEvent::MTE1_M>(c.ev1);
            MmadParams mp; mp.m = 16; mp.n = maxncc * 16; mp.k = KTILE; mp.cmatrixInitVal = (i == 0);
            Mmad(c.lc1, c.la2, c.lb2, mp);
            SetFlag<HardEvent::M_MTE1>(c.ev3);
            l0Busy = true;
            c.consCount++;
            ring_fill(c, s, wBase);                 // top the ring up
        }
        if (l0Busy)  { WaitFlag<HardEvent::M_MTE1>(c.ev3); l0Busy = false; }
        SetFlag<HardEvent::M_FIX>(c.ev2); WaitFlag<HardEvent::M_FIX>(c.ev2);
        FixpipeParamsV220 fp; fp.nSize = nce; fp.mSize = 2; fp.srcStride = 16; fp.dstStride = outStride;
        Fixpipe(outBase[(int64_t)c.cid * outSlab + nt * s.nc], c.lc1, fp);
        SetFlag<HardEvent::FIX_M>(c.ev4);
        l0cBusy = true;
#if MEGA_LM_STREAM
        if (outputSeq != nullptr) {
            const int live = s.nGlobal - c.cid * s.slab < s.slab
                ? s.nGlobal - c.cid * s.slab : s.slab;
            const int end = nt * s.nc + nce;
            const int boundary = (publishedChunks + 1) * CH < live
                ? (publishedChunks + 1) * CH : live;
            if (end >= boundary) {
                // Check actual data landing, as for the previous whole-span gate.
                aic_wait_land(outputRow + boundary - 1);
                aic_seq(outputSeq, SEQ_WANT(24 + c.cid, PR_LM),
                        outputTag + ++publishedChunks);
            }
        }
#endif
    }
    if (l0cBusy) { WaitFlag<HardEvent::FIX_M>(c.ev4); }
    // drain the ring's leftover slot-free flags (one per used slot; none when
    // this slab is empty, e.g. AICs 22/23 for o/down)
    const int used = (s.total < RN) ? (int)s.total : RN;
    for (int b = 0; b < used; b++) { WaitFlag<HardEvent::MTE1_MTE2>(c.eM1R[b]); }
}

// ---------------- AIV helpers ----------------
// GetValue executes on S. A vector-pipe barrier alone is not an explicit
// V -> S dependency. Reserve V_S event 7 for synchronous scalar readback.
__aicore__ inline void vector_scalar_fence() {
    SetFlag<HardEvent::V_S>((event_t)7);
    WaitFlag<HardEvent::V_S>((event_t)7);
}

// Complete scalar reads before the same scratch is written by V again.
__aicore__ inline void scalar_vector_fence() {
    SetFlag<HardEvent::S_V>((event_t)7);
    WaitFlag<HardEvent::S_V>((event_t)7);
}

// tree reduce over n (power-of-2 multiple of 8, n>=16); result in t[0..8)
__aicore__ inline float tree_sum(LocalTensor<float> t, int n) {
    for (int len = n / 2; len >= 8; len /= 2) { Add(t, t, t[len], len); }
    vector_scalar_fence();
    float s = t.GetValue(0);
    for (int i = 1; i < 8; i++) { s += t.GetValue(i); }
    scalar_vector_fence();
    return s;
}
__aicore__ inline float tree_max(LocalTensor<float> t, int n) {
    for (int len = n / 2; len >= 8; len /= 2) { Max(t, t, t[len], len); }
    vector_scalar_fence();
    float m = t.GetValue(0);
    for (int i = 1; i < 8; i++) { float x = t.GetValue(i); if (x > m) m = x; }
    scalar_vector_fence();
    return m;
}
// scalar exp via 16-el vector buffer
__aicore__ inline float vexp_scalar(LocalTensor<float> t16, float x) {
    Duplicate(t16, x, 16);
    Exp(t16, t16, 16);
    vector_scalar_fence();
    float result = t16.GetValue(0);
    scalar_vector_fence();
    return result;
}

// rmsnorm: t (n f32, destroyed), s (n f32 scratch), gb (n bf16 gamma),
// o (n bf16 out), t16 (16 f32 scratch). n multiple of 8, >= 16.
__aicore__ inline void rms_norm(LocalTensor<float> t, LocalTensor<float> s,
    LocalTensor<bfloat16_t> gb, LocalTensor<bfloat16_t> o, int n, LocalTensor<float> t16) {
    Mul(s, t, t, n);
    const float ms = tree_sum(s, n) / (float)n;
    Duplicate(t16, ms + EPS, 16);
    Rsqrt(t16, t16, 16);
    vector_scalar_fence();
    float r = t16.GetValue(0);
    // Two Newton steps recover FP32 rsqrt accuracy before BF16 rounding.
    r = r * (1.5f - 0.5f * (ms + EPS) * r * r);
    r = r * (1.5f - 0.5f * (ms + EPS) * r * r);
    scalar_vector_fence();
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

// Prepare one A chunk. Compact mode copies two contiguous ND rows; legacy
// mode broadcasts all 16 rows in the fractal (probe11).
__aicore__ inline void bcast_pack(LocalTensor<bfloat16_t> pack, LocalTensor<bfloat16_t> src) {
#if MEGA_COMPACT_A
    CopyRepeatParams cp;
    cp.dstStride = 1; cp.srcStride = 1;
    cp.dstRepeatSize = 8; cp.srcRepeatSize = 8;
    Copy(pack, src, (uint64_t)128, (uint8_t)2, cp);
    Copy(pack[256], src, (uint64_t)128, (uint8_t)2, cp);
#else
    CopyRepeatParams cp; cp.dstStride = 1; cp.srcStride = 0; cp.dstRepeatSize = 8; cp.srcRepeatSize = 0;
    for (int j = 0; j < 16; j++) {
        Copy(pack[j * 256], src[j * 16], (uint64_t)128, (uint8_t)2, cp);
    }
#endif
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

// Pack a bf16 vector into GM A chunks using idle argmax scratch. Compact
// mode emits eight chunks (8KB) in one transfer; legacy mode emits four
// broadcast chunks (32KB). Both preserve the bf16 input values exactly.
__aicore__ inline void pack_out(LocalTensor<bfloat16_t> pack, LocalTensor<bfloat16_t> v,
    GlobalTensor<bfloat16_t> gmPack, int nChunks, event_t eV3, event_t eM3V) {
    constexpr int batch = MEGA_COMPACT_A ? 8 : 4;
    for (int b = 0; b < nChunks; b += batch) {
        const int nc = (b + batch <= nChunks) ? batch : (nChunks - b);
        for (int c = 0; c < nc; c++) { bcast_pack(pack[c * A_PACK_CHUNK], v[(b + c) * 256]); }
        SetFlag<HardEvent::V_MTE3>(eV3); WaitFlag<HardEvent::V_MTE3>(eV3);
        DataCopy(gmPack[(int64_t)b * A_PACK_CHUNK], pack, nc * A_PACK_CHUNK);
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

// AIV: poll the AIC seq words (SEQ_WANT(24+c, r), far apart so the aic_seq
// store+dcci of different AICs never share a cache line) until ALL words in
// [c0,c1) equal want. v3: sweep split into (a) N independent volatile reads
// issued back-to-back -- their L2 latencies overlap in the memory pipeline
// (the old interleaved read+dcci serialized at ~200ns x N = ~5us per sweep,
// which the timeline showed as the gate detect latency), (b) dcci only for
// the words that missed. All words are re-read after any miss.
__aicore__ inline void aic_gate_poll(volatile __gm__ int32_t* gseqp, int r, int32_t want,
    int c0, int c1) {
    for (;;) {
        int32_t v[24];
        for (int c = c0; c < c1; c++) {
            v[c] = *(gseqp + SEQ_WANT(24 + c, r));
        }
        bool ok = true;
        for (int c = c0; c < c1; c++) {
            if (v[c] != want) {
                volatile __gm__ int32_t* p = gseqp + SEQ_WANT(24 + c, r);
                dcci(reinterpret_cast<__gm__ uint64_t*>((__gm__ int32_t*)p),
                     cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
                ok = false;
            }
        }
        if (ok) break;
        __asm__ __volatile__("" ::: "memory");
    }
    __asm__ __volatile__("" ::: "memory");
}

// AIV: poll ONE AIC seq word (1:1 gates).
__aicore__ inline void aic_gate_poll1(volatile __gm__ int32_t* gseqp, int r, int cid,
    int32_t want) {
    volatile __gm__ int32_t* p = gseqp + SEQ_WANT(24 + cid, r);
    while (*p != want) {
        dcci(reinterpret_cast<__gm__ uint64_t*>((__gm__ int32_t*)p),
             cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
        __asm__ __volatile__("" ::: "memory");
    }
    __asm__ __volatile__("" ::: "memory");
}

// AIV: poll an AIC PROGRESS word (row 24+cid, round r) until it reaches `want`
// (>= test, same monotonic-count scheme as the AIC's aic_wait_prog).  Used for
// the streamed score-group hand-off (R13/2a).
__aicore__ inline void aiv_wait_prog(volatile __gm__ int32_t* gseqp, int r, int cid,
    int32_t want) {
    volatile __gm__ int32_t* p = gseqp + SEQ_WANT(24 + cid, r);
    while (*p < want) {
        dcci(reinterpret_cast<__gm__ uint64_t*>((__gm__ int32_t*)p),
             cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
        __asm__ __volatile__("" ::: "memory");
    }
    __asm__ __volatile__("" ::: "memory");
}

// ---- v8 AIV-side L2 warmer (see OPTLOG R7) -------------------------------
// The AIV plane is ~97% MTE2-idle (msprof: `aiv_mte2_ratio` 3%), while the AIC
// plane is stalled ~34us/layer waiting on AIV stages.  During one of those
// windows -- the all-AIV wait for the 22 AIC "o" words -- the AIVs have an
// independent MTE2 and nothing to do, so they stream the gu gemv's weight slab
// (the very next bulk read) into L2 and discard it.  gu then reads from L2 and
// its HBM time drops.  Unlike the rejected R6c (same trick from the AIC, which
// self-blocked on its own MTE2 queue), this uses a *different* core's MTE2.
constexpr int WARM_SLOT = 8192;         // bf16 elements per UB scratch slot (16KB)
// Poll AIC words [c0,c1) for `want` and warm [off0,off1) of g meanwhile.
// `pending` comes back as the number of outstanding SetFlags for warm_drain.
__aicore__ inline void aiv_warm_gate(volatile __gm__ int32_t* gseqp, int r,
    int32_t want, int c0, int c1, GlobalTensor<bfloat16_t> g,
    int64_t off0, int64_t off1, LocalTensor<bfloat16_t> s0,
    LocalTensor<bfloat16_t> s1, int& pending) {
    const event_t e0 = (event_t)6, e1 = (event_t)7;   // unused MTE2_V ids
    int64_t off = off0;
    int issued = 0;
    for (;;) {
        int32_t v[24];
        for (int c = c0; c < c1; c++) { v[c] = *(gseqp + SEQ_WANT(24 + c, r)); }
        bool ok = true;
        for (int c = c0; c < c1; c++) {
            if (v[c] != want) {
                volatile __gm__ int32_t* p = gseqp + SEQ_WANT(24 + c, r);
                dcci(reinterpret_cast<__gm__ uint64_t*>((__gm__ int32_t*)p),
                     cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
                ok = false;
            }
        }
        if (ok) break;
        if (off + WARM_SLOT <= off1) {
            const int par = issued & 1;
            if (issued >= 2) { WaitFlag<HardEvent::MTE2_V>(par ? e1 : e0); }
            DataCopy(par ? s1 : s0, g[off], WARM_SLOT);
            SetFlag<HardEvent::MTE2_V>(par ? e1 : e0);
            off += WARM_SLOT;
            issued++;
        }
        __asm__ __volatile__("" ::: "memory");
    }
    pending = issued < 2 ? issued : 2;
}
// Drain the warmer's outstanding SetFlags. MUST be called before the scratch
// (pw/tPw) or the ids 6/7 are reused -- i.e. before the next layer's pass 1
// writes pw, and before kernel exit (a pending flag at exit hangs the core).
__aicore__ inline void aiv_warm_drain(int pending) {
    const event_t e0 = (event_t)6, e1 = (event_t)7;
    for (int k = 0; k < pending; k++) {
        WaitFlag<HardEvent::MTE2_V>((((pending - 1 - k) & 1) ? e1 : e0));
    }
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
        TRACE_C(L, 0);
        const int kv = cid % KVH;
        const int sid = cid / KVH;
        const int s0 = sid * thirds;
        const int e0 = (sl + 1 < s0 + thirds) ? (sl + 1) : (s0 + thirds);
        const int sblk = (e0 > s0) ? (e0 - s0) : 0;
        const int nloc = (sblk + SLT - 1) / SLT;
        const int ngrp = (sblk + 4095) / 4096;   // 16-tile score groups (R13/2a)

        TPipe pipe;
        TBuf<TPosition::A1> a1; TBuf<TPosition::B1> b1;
        TBuf<TPosition::A1> a1b; TBuf<TPosition::B1> b1b;
        TBuf<TPosition::A1> ab1;                     // A-pack group staging
        TBuf<TPosition::B1> rb0, rb1, rb2, rb3;      // prefetch ring (B chunks)
        TBuf<TPosition::A2> a2; TBuf<TPosition::B2> b2; TBuf<TPosition::CO1> c1;
        pipe.InitBuffer(a1, 16 * KTILE * sizeof(bfloat16_t));
        pipe.InitBuffer(b1, 64 * 1024);
        pipe.InitBuffer(a1b, 16 * KTILE * sizeof(bfloat16_t));
        pipe.InitBuffer(b1b, 64 * 1024);
        pipe.InitBuffer(ab1, 8 * 16 * KTILE * sizeof(bfloat16_t));
        pipe.InitBuffer(rb0, 64 * 1024);
        pipe.InitBuffer(rb1, 64 * 1024);
        pipe.InitBuffer(rb2, 64 * 1024);
        pipe.InitBuffer(rb3, 64 * 1024);
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
        C.ev6 = static_cast<event_t>(pipe.FetchEventID<HardEvent::MTE2_MTE1>());
        C.ev7 = static_cast<event_t>(pipe.FetchEventID<HardEvent::MTE1_MTE2>());
        // FetchEventID only peeks at an unallocated channel. The two
        // outstanding score/PV ping-pong transfers MUST have distinct IDs.
        // GEMV drains the ring before these stages, so channel 1 is free.
        C.ev0 = (event_t)0; C.ev6 = (event_t)1;
        C.ev5 = (event_t)0; C.ev7 = (event_t)1;
        // v5 ring: FetchEventID only PEEKS (always 0), so the 4 slot channels
        // are hardcoded (distinct ids 1..4 verified on dav-2201).
        C.eM2R[0] = (event_t)1; C.eM2R[1] = (event_t)2;
        C.eM2R[2] = (event_t)3; C.eM2R[3] = (event_t)4;
        C.eM2R[4] = (event_t)6; C.eM2R[5] = (event_t)7;
        C.eM1R[0] = (event_t)1; C.eM1R[1] = (event_t)2;
        C.eM1R[2] = (event_t)3; C.eM1R[3] = (event_t)4;
        C.eM1R[4] = (event_t)6; C.eM1R[5] = (event_t)7;
        C.evA = (event_t)5;
        C.cid = cid; C.prodCount = 0; C.consCount = 0;
        C.abuf = ab1.Get<bfloat16_t>();
        C.ringb[0] = rb0.Get<bfloat16_t>(); C.ringb[1] = rb1.Get<bfloat16_t>();
        C.ringb[2] = rb2.Get<bfloat16_t>(); C.ringb[3] = rb3.Get<bfloat16_t>();
        C.la1 = a1.Get<bfloat16_t>();
        C.lb1 = b1.Get<bfloat16_t>();
        C.la1b = a1b.Get<bfloat16_t>();
        C.lb1b = b1b.Get<bfloat16_t>();
        C.ringb[4] = C.lb1; C.ringb[5] = C.lb1b;   // alias gemm1/2 buffers
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

        volatile __gm__ float* gqkvF = (volatile __gm__ float*)gqkv_out;
        volatile __gm__ float* gscoresF = (volatile __gm__ float*)gscores;
        volatile __gm__ float* gpartAccF = (volatile __gm__ float*)gpartAccP;
        volatile __gm__ float* goF = (volatile __gm__ float*)go_out;
        volatile __gm__ float* gguF = (volatile __gm__ float*)ggu_out;
        volatile __gm__ float* gdF = (volatile __gm__ float*)gd_out;
        volatile __gm__ float* glogitsF = (volatile __gm__ float*)glogits;
        // sentinel column helpers (last-landed word of this AIC's region)
        const int qkvSC = ((cid + 1) * 176 < NQKV ? (cid + 1) * 176 : NQKV) - 1;
        const int odSC = ((cid + 1) * 96 < H ? (cid + 1) * 96 : H) - 1;
        const int guSC = cid * 512 + 511;
        const int lmLive = (VV - cid * SPAN < SPAN) ? VV - cid * SPAN : SPAN;
        const int lmSC = lmLive - 1;
        const bool odActive = (cid * 96 < H);   // AICs 22,23 skip o/d gemv
        // lm sentinel armed once per launch (logits region is host-prefilled)
#if MEGA_LM_STREAM
        for (int ch = 0; ch < NCH; ch++) {
            const int end = (ch + 1) * CH < lmLive ? (ch + 1) * CH : lmLive;
            aic_arm(glogitsF + cid * SPANP + end - 1);
        }
#else
        aic_arm(glogitsF + cid * SPANP + lmSC);
#endif

        for (int li = 0; li < lstop; li++) {
            const int R = li * 8;
            TRACE_C(li, 0);
            // ---- arm this layer's sentinel words (dead rows of the fixpipe
            // regions). Dead rows are never read by AIVs, so arming is safe
            // regardless of how far the AIVs still are in layer li-1. The
            // previous layer's fixpipe for each region is confirmed landed
            // (this AIC self-polled it) before we re-arm. ----
            const int sr = 1;                  // dead row 1 (weight gemvs write mSize=2)
            const int sr2 = 2;                 // dead row 2 (scores/PV write mSize=3)
            aic_arm(gqkvF + sr * NQKV_PAD + qkvSC);
            if (sblk > 0) {
                // one dead-row sentinel per 4096-position score group (R13/2a) so
                // the AIV can start the max pass as tiles land instead of after
                // the whole slice.  group g ends at tile min(16g+15, nloc-1); the
                // last group's word IS the old single all-scores sentinel.
                for (int g = 0; g < ngrp; g++) {
                    const int ti = ((g + 1) * 16 - 1 < nloc - 1) ? (g + 1) * 16 - 1 : nloc - 1;
                    aic_arm(gscoresF + (int64_t)(cid * 16 + sr2) * STEP + s0 +
                            (int64_t)ti * SLT + (SLT - 1));
                }
                aic_arm(gpartAccF + ((kv * NSL + sid) * 16 + sr2) * D + (D - 1));
            }
            if (odActive) {
                aic_arm(goF + sr * NO_PAD + odSC);
                aic_arm(gdF + sr * ND_PAD + odSC);
            }
            // Certify the live row read by SiLU, one marker per output tile.
            for (int nt = 0; nt < 4; nt++) aic_arm(gguF + cid * 512 + (nt + 1) * 128 - 1);
            // Retire scalar sentinel stores/cache cleans before DMA/Mmad
            // can reach the corresponding Fixpipe destination.
            SetFlag<HardEvent::S_MTE2>((event_t)0);
            WaitFlag<HardEvent::S_MTE2>((event_t)0);
            // ---- qkv gemv (after own aiv wrote xn1P) ----
            {
                GlobalTensor<bfloat16_t> tw; tw.SetGlobalBuffer(gWqkv + (int64_t)li * QKV_LAYER);
                GemvCtx sq; gemv_begin(C, sq, NQKV, H, NC_QKV, 2, 176);
                ring_fill(C, sq, tw);                            // prefetch during wait
                TRACE_C(li, 1);
                poll_seq(gseq + SEQ_WANT(cid, R + 0), tag + R + 0);
                TRACE_C(li, 2);      // rms1 done
                gemv_ring(C, sq, tw, txn1P, tqkvOut, NQKV_PAD, 176);
                TRACE_C(li, 3);
            }
            aic_wait_land(gqkvF + sr * NQKV_PAD + qkvSC);
            aic_seq(gseq, SEQ_WANT(24 + cid, R + 0), tag + R + 0);
            TRACE_C(li, 4);            // qkv done

            {
                // ---- gemm1: scores per 256-tile (MTE2-pipelined across
                // tiles: issue tile t+1 before waiting on t) ----
                poll_seq(gseq + SEQ_WANT(cid, R + 2), tag + R + 2);
                TRACE_C(li, 5);  // appends done
                int pubG = 0;                       // score groups published (R13/2a)
                if (sblk > 0) {
                    const int64_t khead = (int64_t)(li * KVH + kv) * 8 * (SPADG / 16) * 256;
                    const int i0 = s0 / 16;
                    {
                        // prologue: tile 0 into parity-0 buffers
                        DataCopy(C.la1, tq2P[(int64_t)cid * 8 * 256], 8 * 256);
                        DataCopy(C.lb1, tKcP[khead + (int64_t)(i0 / 16) * 32768], 32768);
                        SetFlag<HardEvent::MTE2_MTE1>(C.ev0);
                    }
                    bool l0Busy = false, l0cBusy = false;
                    for (int t = 0; t < nloc; t++) {
                        if (t + 1 < nloc) {      // prefetch tile t+1
                            const int p2 = (t + 1) & 1;
                            if (t >= 1) { WaitFlag<HardEvent::MTE1_MTE2>(p2 ? C.ev7 : C.ev5); }
                            LocalTensor<bfloat16_t> a1p = p2 ? C.la1b : C.la1;
                            LocalTensor<bfloat16_t> b1p = p2 ? C.lb1b : C.lb1;
                            DataCopy(a1p, tq2P[(int64_t)cid * 8 * 256], 8 * 256);
                            DataCopy(b1p, tKcP[khead + (int64_t)(i0 / 16 + (t + 1)) * 32768], 32768);
                            SetFlag<HardEvent::MTE2_MTE1>(p2 ? C.ev6 : C.ev0);
                        }
                        WaitFlag<HardEvent::MTE2_MTE1>((t & 1) ? C.ev6 : C.ev0);
                        if (l0cBusy) { WaitFlag<HardEvent::FIX_M>(C.ev4); l0cBusy = false; }
                        if (l0Busy)  { WaitFlag<HardEvent::M_MTE1>(C.ev3); l0Busy = false; }
                        LocalTensor<bfloat16_t> a1p = (t & 1) ? C.la1b : C.la1;
                        LocalTensor<bfloat16_t> b1p = (t & 1) ? C.lb1b : C.lb1;
                        LoadData2DParams pa(0, 8, 1, 0, 0, false, 0);
                        LoadData(C.la2, a1p, pa);
                        LoadData2DParams pb(0, 128, 1, 0, 0, false, 0);
                        LoadData(C.lb2, b1p, pb);
                        SetFlag<HardEvent::MTE1_MTE2>((t & 1) ? C.ev7 : C.ev5);
                        SetFlag<HardEvent::MTE1_M>(C.ev1); WaitFlag<HardEvent::MTE1_M>(C.ev1);
                        MmadParams mp; mp.m = 16; mp.n = 256; mp.k = D; mp.cmatrixInitVal = true;
                        Mmad(C.lc1, C.la2, C.lb2, mp);
                        SetFlag<HardEvent::M_MTE1>(C.ev3);
                        l0Busy = true;
                        SetFlag<HardEvent::M_FIX>(C.ev2); WaitFlag<HardEvent::M_FIX>(C.ev2);
                        FixpipeParamsV220 fp; fp.nSize = 256; fp.mSize = 3; fp.srcStride = 16;
                        fp.dstStride = STEP;
                        Fixpipe(tScores[((int64_t)cid * 16) * STEP + (int64_t)(s0 + t * SLT)],
                                C.lc1, fp);
                        SetFlag<HardEvent::FIX_M>(C.ev4);
                        l0cBusy = true;
                        // R13/2a: publish a 4096-position score group once its
                        // last tile has certainly reached L2.  Checking 4 tiles
                        // (~5us) after that tile's fixpipe keeps the land poll
                        // off the critical path while the AIV's max pass for the
                        // group overlaps the rest of gemm1.
                        if ((t & 3) == 3 && t >= 19) {
                            const int gd = (t - 19) >> 4;
                            if (gd == pubG) {
                                const int ti = ((gd + 1) * 16 - 1 < nloc - 1) ? (gd + 1) * 16 - 1 : nloc - 1;
                                aic_wait_land(gscoresF + (int64_t)(cid * 16 + sr2) * STEP + s0 +
                                              (int64_t)ti * SLT + (SLT - 1));
                                aic_seq(gseq, SEQ_WANT(24 + cid, PR_SCORES),
                                        tag + li * PROG_SPAN + gd + 1);
                                pubG = gd + 1;
                            }
                        }
                    }
                    if (nloc >= 2) { WaitFlag<HardEvent::MTE1_MTE2>(((nloc - 2) & 1) ? C.ev7 : C.ev5); }
                    WaitFlag<HardEvent::MTE1_MTE2>(((nloc - 1) & 1) ? C.ev7 : C.ev5);
                    if (l0Busy)  { WaitFlag<HardEvent::M_MTE1>(C.ev3); }
                    if (l0cBusy) { WaitFlag<HardEvent::FIX_M>(C.ev4); }
                }
                // R13/2a drain: publish the remaining score groups.  The last
                // poll is exactly the old single all-scores-landed wait, so the
                // whole-slice barrier is gone but the R+2 ordering is unchanged.
                for (int g = pubG; g < ngrp; g++) {
                    const int ti = ((g + 1) * 16 - 1 < nloc - 1) ? (g + 1) * 16 - 1 : nloc - 1;
                    aic_wait_land(gscoresF + (int64_t)(cid * 16 + sr2) * STEP + s0 +
                                  (int64_t)ti * SLT + (SLT - 1));
                    aic_seq(gseq, SEQ_WANT(24 + cid, PR_SCORES), tag + li * PROG_SPAN + g + 1);
                }
                TRACE_C(li, 6);
                // R20: the "scores done" word (24+cid, R+2) was superseded by
                // the per-group PR_SCORES publishes above and is polled by
                // nobody -- dropped (one scalar store + dcci / layer / AIC).
                // ---- gemm2: PV per tile (accumulates across tiles into ONE
                // CO1, single Fixpipe; MTE2-pipelined across tiles) ----
                if (sblk > 0) {
                    // v7 streaming hand-off: gemm2 starts as soon as softmax
                    // chunk 0 (4 tiles) lands and consumes each further chunk
                    // the instant it is published, so the PV stream overlaps
                    // the softmax tail instead of waiting for the whole stage.
                    const int64_t vhead = (int64_t)(li * KVH + kv) * (SPADG / 16) * 8 * 256;
                    const int i0 = s0 / 16;
                    const int32_t pbase = tag + li * PROG_SPAN;
                    {
                        // prologue: tile 0 into parity-0 buffers (chunk 0 ready).
                        aic_wait_prog(gseq + SEQ_WANT(cid, PR_PROBS), pbase + 1);
                        TRACE_C(li, 7);
                        nd2nz_probs(C.la1, tprobs2P, (int64_t)cid * NTM * 512);
                        // V layout is [s-blk][d-blk][16d x 16s]; a 256-tile is 16
                        // consecutive s-blocks => ONE contiguous 32768-el copy
                        // (was 16 x 4KB -- dominated the AIC's MTE2 instruction
                        // count with instruction-bound tiny copies).
                        DataCopy(C.lb1, tVcP[vhead + (int64_t)i0 * 8 * 256], 32768);
                        SetFlag<HardEvent::MTE2_MTE1>(C.ev0);
                    }
                    bool l0Busy = false;
                    for (int t = 0; t < nloc; t++) {
                        if (t + 1 < nloc) {      // prefetch tile t+1
                            const int p2 = (t + 1) & 1;
                            if (t >= 1) { WaitFlag<HardEvent::MTE1_MTE2>(p2 ? C.ev7 : C.ev5); }
                            LocalTensor<bfloat16_t> a1p = p2 ? C.la1b : C.la1;
                            LocalTensor<bfloat16_t> b1p = p2 ? C.lb1b : C.lb1;
                            // V first (no probs dependency), then wait for only
                            // the chunk that owns tile t+1 before its probs copy.
                            DataCopy(b1p, tVcP[vhead + (int64_t)(i0 + (t + 1) * 16) * 8 * 256], 32768);
                            aic_wait_prog(gseq + SEQ_WANT(cid, PR_PROBS), pbase + (t + 1) / 4 + 1);
                            nd2nz_probs(a1p, tprobs2P, ((int64_t)cid * NTM + t + 1) * 512);
                            SetFlag<HardEvent::MTE2_MTE1>(p2 ? C.ev6 : C.ev0);
                        }
                        WaitFlag<HardEvent::MTE2_MTE1>((t & 1) ? C.ev6 : C.ev0);
                        if (l0Busy) { WaitFlag<HardEvent::M_MTE1>(C.ev3); l0Busy = false; }
                        LocalTensor<bfloat16_t> a1p = (t & 1) ? C.la1b : C.la1;
                        LocalTensor<bfloat16_t> b1p = (t & 1) ? C.lb1b : C.lb1;
                        LoadData2DParams pa(0, 16, 1, 0, 0, false, 0);
                        LoadData(C.la2, a1p, pa);
                        LoadData2DParams pb(0, 128, 1, 0, 0, false, 0);
                        LoadData(C.lb2, b1p, pb);
                        SetFlag<HardEvent::MTE1_MTE2>((t & 1) ? C.ev7 : C.ev5);
                        SetFlag<HardEvent::MTE1_M>(C.ev1); WaitFlag<HardEvent::MTE1_M>(C.ev1);
                        MmadParams mp; mp.m = 16; mp.n = D; mp.k = SLT; mp.cmatrixInitVal = (t == 0);
                        Mmad(C.lc1, C.la2, C.lb2, mp);
                        SetFlag<HardEvent::M_MTE1>(C.ev3);
                        l0Busy = true;
                    }
                    if (nloc >= 2) { WaitFlag<HardEvent::MTE1_MTE2>(((nloc - 2) & 1) ? C.ev7 : C.ev5); }
                    WaitFlag<HardEvent::MTE1_MTE2>(((nloc - 1) & 1) ? C.ev7 : C.ev5);
                    if (l0Busy) { WaitFlag<HardEvent::M_MTE1>(C.ev3); }
                    SetFlag<HardEvent::M_FIX>(C.ev2); WaitFlag<HardEvent::M_FIX>(C.ev2);
                    FixpipeParamsV220 fp; fp.nSize = D; fp.mSize = 3; fp.srcStride = 16;
                    fp.dstStride = D;
                    TRACE_C(li, 8);
                    Fixpipe(tPartAcc[((int64_t)(kv * NSL + sid) * 16) * D], C.lc1, fp);
                    SetFlag<HardEvent::FIX_M>(C.ev4);
                    WaitFlag<HardEvent::FIX_M>(C.ev4);
                }
                if (sblk > 0) { aic_wait_land(gpartAccF + ((kv * NSL + sid) * 16 + sr2) * D + (D - 1)); }
                aic_seq(gseq, SEQ_WANT(24 + cid, R + 4), tag + R + 4);
                TRACE_C(li, 9);  // partials done
            }
            // ---- o gemv ----
            {
                GlobalTensor<bfloat16_t> tw; tw.SetGlobalBuffer(gWo + (int64_t)li * O_LAYER);
                GemvCtx so; gemv_begin(C, so, H, H, NC_O, 1, 96);
                ring_fill(C, so, tw);
                TRACE_C(li, 10);        // prefetch Wo during the B_ATT wait
                // B_ATT go: poll all 8 leader words with one overlapped
                // sweep (independent reads, dcci only on miss)
                {
                    int32_t want5 = tag + R + 5;
                    for (;;) {
                        int32_t v[8];
                        for (int c = 0; c < 8; c++) {
                            v[c] = *(gseq + SEQ_WANT(c, R + 5));
                        }
                        bool ok = true;
                        for (int c = 0; c < 8; c++) {
                            if (v[c] != want5) {
                                volatile __gm__ int32_t* p = gseq + SEQ_WANT(c, R + 5);
                                dcci(reinterpret_cast<__gm__ uint64_t*>((__gm__ int32_t*)p),
                                     cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
                                ok = false;
                            }
                        }
                        if (ok) break;
                        __asm__ __volatile__("" ::: "memory");
                    }
                }
                TRACE_C(li, 11);
                gemv_ring(C, so, tw, tattnP, toOut, NO_PAD, 96);
                TRACE_C(li, 12);
            }
            if (odActive) { aic_wait_land(goF + sr * NO_PAD + odSC); }
            aic_seq(gseq, SEQ_WANT(24 + cid, R + 5), tag + R + 5);
            TRACE_C(li, 13);      // o done
            // ---- gate_up gemv ----
            {
                GlobalTensor<bfloat16_t> tw; tw.SetGlobalBuffer(gWgu + (int64_t)li * GU_LAYER);
                GemvCtx s; gemv_begin(C, s, NGU, H, NC_GU, 4, 512);
                ring_fill(C, s, tw);
                TRACE_C(li, 14);
                poll_seq(gseq + SEQ_WANT(cid, R + 6), tag + R + 6);
                TRACE_C(li, 15);      // xn2 done
                gemv_ring(C, s, tw, txn2P, tguOut, NGU, 512);
                TRACE_C(li, 16);
            }
            for (int nt = 0; nt < 4; nt++) aic_wait_land(gguF + cid * 512 + (nt + 1) * 128 - 1);
            aic_seq(gseq, SEQ_WANT(24 + cid, R + 6), tag + R + 6);
            TRACE_C(li, 17);      // gu done
            // ---- down gemv ----
            // AIV c writes hP chunk c. Wait for each 8-chunk A group
            // immediately before loading it; later groups overlap the GEMV.
            {
                GlobalTensor<bfloat16_t> tw; tw.SetGlobalBuffer(gWd + (int64_t)li * D_LAYER);
                GemvCtx s; gemv_begin(C, s, H, II, NC_D, 1, 96);
                ring_fill(C, s, tw);
                TRACE_C(li, 18);
                gemv_ring(C, s, tw, thP, tdOut, ND_PAD, 96,
                          gseq, R + 7, tag + R + 7);
                TRACE_C(li, 19);
            }
            if (odActive) { aic_wait_land(gdF + sr * ND_PAD + odSC); }
            aic_seq(gseq, SEQ_WANT(24 + cid, R + 7), tag + R + 7);
            // Empty down slabs only consumed A group 0. Retire every SiLU
            // reader before these fast cores arm next layer's live GU markers.
            if (!odActive) {
                for (int c = 8; c < NB; c++) poll_seq(gseq + SEQ_WANT(c, R + 7), tag + R + 7);
            }
            TRACE_C(li, 20);      // down done
        }

        // ---- final: lm gemv + logits ----
        {
            GemvCtx s; gemv_begin(C, s, VV, H, NC_LM, NRD_LM, SPAN);
            ring_fill(C, s, tWlm);
            TRACE_C(L, 1);
            poll_seq(gseq + SEQ_WANT(cid, 224), tag + 224);
            TRACE_C(L, 2);          // rmsf done
            // lm head as a pipelined slab gemv; weights packed to NB*SPAN,
            // output spans padded to SPANP per cid (outSlab > slab).
#if MEGA_LM_STREAM
            gemv_ring(C, s, tWlm, txnfP, tLogits, NB * SPANP, SPANP,
                      nullptr, 0, 0, gseq, glogitsF + cid * SPANP, tag);
#else
            gemv_ring(C, s, tWlm, txnfP, tLogits, NB * SPANP, SPANP);
#endif
            TRACE_C(L, 3);
            aic_wait_land(glogitsF + cid * SPANP + lmSC);
            aic_seq(gseq, SEQ_WANT(24 + cid, 224), tag + 224);
            TRACE_C(L, 4);            // logits done
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
        // slice 1/2 live lengths (leaders poll those slices' AICs only if live)
        const int e1 = (sl + 1 < 2 * thirds) ? (sl + 1) : (2 * thirds);
        const int sblk1 = (e1 > thirds) ? (e1 - thirds) : 0;
        const int e2 = (sl + 1 < 3 * thirds) ? (sl + 1) : (3 * thirds);
        const int sblk2 = (e2 > 2 * thirds) ? (e2 - 2 * thirds) : 0;
        const int sidOwner = (sl / thirds < NSL) ? (sl / thirds) : (NSL - 1);
        const bool leader = (sid == 0);
        // R27: carried softmax shift (a rescale-free online softmax).
        // exp(x - sh) is exact for ANY sh -- it cancels in combine's num/den --
        // so the only thing sh must do is keep the exponent in range.  Layer 0
        // MEASURES its own slice max (one extra read, once per token); every
        // later layer reuses the previous layer's measured max, which turns the
        // safety condition from "max < 88" (a per-model constant) into "the max
        // cannot grow by >88 from one layer to the next" (model independent).
        // The in-pass max is still computed: it is what gets carried.
        float sh0 = NEG_INF, sh1 = NEG_INF;   // R27 shift: set per layer below

        TPipe pipe;
        TBuf<TPosition::VECCALC> tXm, tXf, tXn, tPack, tQkv, tHead, tHb, tSm, tP2, tQ2,
            tCmb, tAm, tSig, tSync, tFr, tSq, tW, tHbw, tPw;
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
        pipe.InitBuffer(tFr, 8 * 256 * sizeof(bfloat16_t));      // V append RMW (batched)
        pipe.InitBuffer(tSq, 2 * 16 * sizeof(int32_t));          // seq word (x2)
        // wide softmax: CW positions at once (CW/SLT tiles), packed output
        pipe.InitBuffer(tW, 2 * 1024 * sizeof(float));           // w0,w1 f32 row chunks
        pipe.InitBuffer(tHbw, 2048 * sizeof(bfloat16_t));        // wide exp bf16 (2 heads)
        pipe.InitBuffer(tPw, (1024 / 256) * 16 * 256 * sizeof(bfloat16_t));  // packed prob tiles
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
        LocalTensor<float> ws = tW.Get<float>();
        LocalTensor<float> w0 = ws[0], w1 = ws[1024];
        LocalTensor<bfloat16_t> hbw = tHbw.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> pw = tPw.Get<bfloat16_t>();
        LocalTensor<float> pwf = tPw.Get<float>();   // warm-gate scratch (unused by stage 4)
        // R24: diagnostic-max scratch (2x copy of a 1024-wide scaled row chunk)
        LocalTensor<float> mx0 = pwf[0], mx1 = pwf[1024];

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
        LocalTensor<bfloat16_t> packw = tAm.Get<bfloat16_t>();   // wide pack scratch (32KB, idle)

        event_t eM2V = static_cast<event_t>(pipe.FetchEventID<HardEvent::MTE2_V>());
        event_t eV2 = static_cast<event_t>(pipe.FetchEventID<HardEvent::V_MTE2>());
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
        GlobalTensor<bfloat16_t> tWguA;              // v8: gu weights (L2 warmer)
        tWguA.SetGlobalBuffer((__gm__ bfloat16_t*)p_WguP);
        tKcP.SetGlobalBuffer(gKcP);   tVcP.SetGlobalBuffer(gVcP);
        txn1P.SetGlobalBuffer(gxn1P); tq2P.SetGlobalBuffer(gq2P);
        tprobs2P.SetGlobalBuffer(gprobs2P); tattnP.SetGlobalBuffer(gattnP);
        txn2P.SetGlobalBuffer(gxn2P); thP.SetGlobalBuffer(ghP);
        txnfP.SetGlobalBuffer((__gm__ bfloat16_t*)p_xnfP);
        tEmb.SetGlobalBuffer((__gm__ bfloat16_t*)p_xEmb);
        tQkvOut.SetGlobalBuffer(gqkv_out); tScores.SetGlobalBuffer(gscores);
        tPartM.SetGlobalBuffer(gpartm);    tPartL.SetGlobalBuffer(gpartl);
        tPartAcc.SetGlobalBuffer(gpartAccP);
        GlobalTensor<float> tShiftHistory;
        tShiftHistory.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(p_syncws) + SHIFT_HISTORY);

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
            TRACE_V(li, 0);
            // ---- stage 1: rms1 -> xn1P ----
            {
                Cast(xf, xm, RoundMode::CAST_NONE, H);
                {
                    GlobalTensor<bfloat16_t> tg; tg.SetGlobalBuffer(gRms1w + (int64_t)li * H);
                    DataCopy(pack, tg, H);   // gamma (bf16 view)
                }
                SetFlag<HardEvent::MTE2_V>(eM2V); WaitFlag<HardEvent::MTE2_V>(eM2V);
                rms_norm(xf, xs, pack, xn, H, t16);
                if (sub == 0) { pack_out(packw, xn, txn1P, H / 256, eV3, eM3V); }
            }
            if (sub == 0) { seq_write(gseq, SEQ_WANT(cid, R + 0), tag + R + 0, sq, sqWhich ^= 1, eV3, eM3V); }

            TRACE_V(li, 1);
            // ---- stage 2: qkv gate. This AIV reads only its own head's
            // q/k/v output columns, which come from a few AIC slabs -- poll
            // only those AICs so we decouple from the slowest of all 24
            // (removes the cross-block qkv land-skew from the critical path).
            {
                const int qc0 = kv * 256;                 // q head 2kv,2kv+1 cols
                const int kc0 = NQKV / 2 + kv * D;
                const int vc0 = 3 * NQKV / 4 + kv * D;
                aic_gate_poll(gseqp, R + 0, tag + R + 0, qc0 / 176, (qc0 + 255) / 176 + 1);
                aic_gate_poll(gseqp, R + 0, tag + R + 0, kc0 / 176, (kc0 + D - 1) / 176 + 1);
                aic_gate_poll(gseqp, R + 0, tag + R + 0, vc0 / 176, (vc0 + D - 1) / 176 + 1);
            }

            TRACE_V(li, 2);
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
                        vector_scalar_fence();               // vb reads below
                        for (int j = 0; j < 8; j++) {
                            DataCopy(tKcP[kbase + (int64_t)(iBlk / 16) * 32768 +
                                          (int64_t)j * 4096 + (int64_t)(iBlk % 16) * 256 + iRow * 16],
                                     kb[j * 16], 16);
                        }
                        // V append: one 4KB RMW over all 8 d-blocks
                        // (replaces 8 serialized per-block RMWs, v3)
                        DataCopy(vfr, tVcP[vbase + (int64_t)iBlk * 8 * 256], 8 * 256);
                        SetFlag<HardEvent::MTE2_S>(eM2S); WaitFlag<HardEvent::MTE2_S>(eM2S);
                        for (int j = 0; j < 8; j++) {
                            for (int n = 0; n < 16; n++) {
                                vfr.SetValue(j * 256 + n * 16 + iRow, vb.GetValue(j * 16 + n));
                            }
                        }
                        event_t esApp = static_cast<event_t>(s3cnt++ & 3);  // one-shot ids
                        SetFlag<HardEvent::S_MTE3>(esApp); WaitFlag<HardEvent::S_MTE3>(esApp);
                        DataCopy(tVcP[vbase + (int64_t)iBlk * 8 * 256], vfr, 8 * 256);
                        SetFlag<HardEvent::MTE3_MTE2>(eM3M2); WaitFlag<HardEvent::MTE3_MTE2>(eM3M2);
                    }
                }
            }
            if (sub == 0) { seq_write(gseq, SEQ_WANT(cid, R + 2), tag + R + 2, sq, sqWhich ^= 1, eV3, eM3V); }

            {
                TRACE_V(li, 3);
                // ---- stage 4: SINGLE-PASS softmax -> probs2P, part_m/l ----
                // (1:1 gate: this AIV reads only its own AIC's score rows)
                // R13/2a: no whole-slice score barrier -- the pass takes its
                // 4096-position groups as the AIC publishes them (see the
                // PR_SCORES wait inside the loop), so it runs under gemm1's K
                // stream instead of after it.
                // R21: the max pass is GONE.  Softmax is shift invariant, so
                // exp(x) with a GLOBAL shift of 0 is exactly as correct as
                // exp(x - m_slice) as long as no exponent overflows f32
                // (needs x > 88) or underflows bf16 (x < -87); the measured
                // worst scaled score on the real model is 27.  Scores are now
                // read ONCE instead of twice (-2.6MB/layer of HBM) and combine
                // no longer rescales by exp(m_s - mg): every live slice's
                // partial carries the same shift, so it just sums them and
                // divides by the summed row totals.  part_m is still
                // published (from a cheap in-pass reduce) so the gate keeps
                // verifying the score values end to end.
                float m0 = NEG_INF, m1 = NEG_INF, l0 = 0.0f, l1 = 0.0f;
                const float scale = 0.08838834764831845f;  // 1/sqrt(128)
                // R18: sub1's softmax is dead work -- every output of stage 4
                // (probs2P, part_m/l, R+3, combine) is sub0-only, so the second
                // AIV of each group was re-reading the whole score slice twice
                // for nothing (~146MB/token of HBM reads).
                if (sub == 0 && sblk > 0) {
                    // v6: process CW=1024 positions per iteration (4 tiles) to
                    // collapse the per-tile MTE2 latency and batch the probs2P
                    // write; the tail (sblk%CW) falls back to the 256 loop.
                    const int64_t sbase0 = ((int64_t)cid * 16) * STEP + s0;
                    const int64_t sbase1 = ((int64_t)cid * 16 + 1) * STEP + s0;
                    // R27: every slice of a head uses ONE common shift, so the
                    // three partials stay commensurable and combine can keep
                    // summing them.  exp(x-sh) is exact for ANY sh (it cancels
                    // in num/den); sh only has to stay within ~88 of the max.
                    //   li > 0 : slice 0's max as published by the layer below.
                    //            The value lives in a separate parity bank until
                    //            every slice has finished the current layer.
                    //   li == 0: measure slice 0 once.  Every core reads the
                    //            SAME region, so all three slices get an
                    //            identical value; layer 0 is the only layer
                    //            with no history to carry.
                    if (li > 0) {
                        // Both heads are contiguous in the immutable history bank;
                        // current maxima are published to the other bank.
                        DataCopy(pml, tShiftHistory[((li - 1) & 1) * (KVH * 2 * 16) + kv * 2 * 16], 32);
                        SetFlag<HardEvent::MTE2_S>(eM2S); WaitFlag<HardEvent::MTE2_S>(eM2S);
                        sh0 = pml.GetValue(0);
                        sh1 = pml.GetValue(16);
                        // Sanity guard; layer 0 always bootstraps and publishes
                        // this launch's history before layer 1 reads it.
                        // The band only rejects NaN/inf/absurd values -- a legit
                        // shift can be large (any constant offset is valid).
                        if (!(sh0 > -1e5f && sh0 < 1e5f)) sh0 = 0.0f;
                        if (!(sh1 > -1e5f && sh1 < 1e5f)) sh1 = 0.0f;
                    } else {
                        const int64_t b0 = ((int64_t)kv * 16) * STEP;
                        const int sb0 = (sl + 1 < thirds) ? (sl + 1) : thirds;
                        for (int c = 0; c < sb0; c += 4096) {
                            aiv_wait_prog(gseqp, PR_SCORES, kv, tag + c / 4096 + 1);
                            const int cl = (c + 4096 <= sb0) ? 4096 : (sb0 - c);
                            if (cl == 4096) {
                                // Protect the reduction operand until its vector readers finish.
                                SetFlag<HardEvent::V_MTE2>(eV2); WaitFlag<HardEvent::V_MTE2>(eV2);
                                DataCopy(pwf[0], tScores[b0 + c], 4096);
                                DataCopy(pwf[4096], tScores[b0 + STEP + c], 4096);
                                SetFlag<HardEvent::MTE2_V>(eM2V); WaitFlag<HardEvent::MTE2_V>(eM2V);
                                Muls(pwf[0], pwf[0], scale, 4096);
                                Muls(pwf[4096], pwf[4096], scale, 4096);
#if MEGA_SCORE_BUMP != 0
                                Adds(pwf[0], pwf[0], (float)MEGA_SCORE_BUMP, 4096);
                                Adds(pwf[4096], pwf[4096], (float)MEGA_SCORE_BUMP, 4096);
#endif
                                float tm = tree_max(pwf[0], 4096); if (tm > sh0) sh0 = tm;
                                tm = tree_max(pwf[4096], 4096); if (tm > sh1) sh1 = tm;
                            } else {
                                for (int d = 0; d < cl; d += SLT) {
                                    const int tail = cl - d;
                                    // Protect the reduction operand until its vector readers finish.
                                    SetFlag<HardEvent::V_MTE2>(eV2); WaitFlag<HardEvent::V_MTE2>(eV2);
                                    DataCopy(sc0, tScores[b0 + c + d], SLT);
                                    DataCopy(sc1, tScores[b0 + STEP + c + d], SLT);
                                    SetFlag<HardEvent::MTE2_V>(eM2V); WaitFlag<HardEvent::MTE2_V>(eM2V);
                                    Muls(sc0, sc0, scale, SLT);
                                    Muls(sc1, sc1, scale, SLT);
#if MEGA_SCORE_BUMP != 0
                                    Adds(sc0, sc0, (float)MEGA_SCORE_BUMP, SLT);
                                    Adds(sc1, sc1, (float)MEGA_SCORE_BUMP, SLT);
#endif
                                    if (tail >= SLT) {
                                        Add(mx0, sc0, sc0, SLT);
                                        Add(mx1, sc1, sc1, SLT);
                                        float tm = 0.5f * tree_max(mx0, SLT); if (tm > sh0) sh0 = tm;
                                        tm = 0.5f * tree_max(mx1, SLT); if (tm > sh1) sh1 = tm;
                                    } else {
                                        vector_scalar_fence();
                                        for (int i = 0; i < tail; i++) {
                                            float v = sc0.GetValue(i); if (v > sh0) sh0 = v;
                                            v = sc1.GetValue(i); if (v > sh1) sh1 = v;
                                        }
                                        scalar_vector_fence();
                                    }
                                }
                            }
                        }
                    }
                    // R24: one pass over the scores.  probs are exp(x - sh) with
                    // a per-AIV carried shift (R27); the in-pass max feeds the
                    // NEXT layer's shift, so nothing here is model specific.
                    for (int c = 0; c < sblk; c += 1024) {          // exp/probs/l
                        // R13/2a: the AIC publishes one 4096-position group at
                        // a time -- wait for the group this chunk lives in.
                        if ((c & 4095) == 0) {
                            aiv_wait_prog(gseqp, PR_SCORES, cid,
                                          tag + li * PROG_SPAN + c / 4096 + 1);
                        }
                        const int cl = (c + 1024 <= sblk) ? 1024 : (sblk - c);
                        if (cl == 1024) {
                            // Protect the reduction operand until its vector readers finish.
                            SetFlag<HardEvent::V_MTE2>(eV2); WaitFlag<HardEvent::V_MTE2>(eV2);
                            DataCopy(w0, tScores[sbase0 + c], 1024);
                            DataCopy(w1, tScores[sbase1 + c], 1024);
                            SetFlag<HardEvent::MTE2_V>(eM2V); WaitFlag<HardEvent::MTE2_V>(eM2V);
                            Muls(w0, w0, scale, 1024);
                            Muls(w1, w1, scale, 1024);
#if MEGA_SCORE_BUMP != 0
                            Adds(w0, w0, (float)MEGA_SCORE_BUMP, 1024);
                            Adds(w1, w1, (float)MEGA_SCORE_BUMP, 1024);
#endif
                            // diagnostic slice max only (Add is an exact 2x
                            // copy and, unlike Muls by 1.0, is not folded away)
                            Add(mx0, w0, w0, 1024);
                            Add(mx1, w1, w1, 1024);
                            Adds(w0, w0, -sh0, 1024);
                            Adds(w1, w1, -sh1, 1024);
                            Exp(w0, w0, 1024);
                            Exp(w1, w1, 1024);
                            float tm = 0.5f * tree_max(mx0, 1024); if (tm > m0) m0 = tm;
                            tm = 0.5f * tree_max(mx1, 1024); if (tm > m1) m1 = tm;
                            // land probs BEFORE tree_sum (it destroys in place!)
                            // R16 compact: [tile][head0 256][head1 256] bf16
                            for (int q = 0; q < 4; q++) {
                                Cast(hbw[q * 512], w0[q * 256], RoundMode::CAST_RINT, 256);
                                Cast(hbw[q * 512 + 256], w1[q * 256], RoundMode::CAST_RINT, 256);
                            }
                            l0 += tree_sum(w0, 1024);
                            l1 += tree_sum(w1, 1024);
                            if (sub == 0) {
                                SetFlag<HardEvent::V_MTE3>(eV3); WaitFlag<HardEvent::V_MTE3>(eV3);
                                DataCopy(tprobs2P[((int64_t)cid * NTM + c / SLT) * 512], hbw, 4 * 512);
                                SetFlag<HardEvent::MTE3_V>(eM3V); WaitFlag<HardEvent::MTE3_V>(eM3V);
                                // v7: publish this 1024-chunk (4 tiles).  The
                                // seq MTE3 is ordered after the probs MTE3, so
                                // gemm2 sees the progress word only once the
                                // chunk is actually in GM.
                                seq_write(gseq, SEQ_WANT(cid, PR_PROBS),
                                          tag + li * PROG_SPAN + c / 1024 + 1,
                                          sq, sqWhich ^= 1, eV3, eM3V);
                            }
                        } else {
                            for (int d = 0; d < cl; d += SLT) {
                                const int tail = cl - d;
                                // Protect the reduction operand until its vector readers finish.
                                SetFlag<HardEvent::V_MTE2>(eV2); WaitFlag<HardEvent::V_MTE2>(eV2);
                                DataCopy(sc0, tScores[sbase0 + c + d], SLT);
                                DataCopy(sc1, tScores[sbase1 + c + d], SLT);
                                SetFlag<HardEvent::MTE2_V>(eM2V); WaitFlag<HardEvent::MTE2_V>(eM2V);
                                Muls(sc0, sc0, scale, SLT);
                                Muls(sc1, sc1, scale, SLT);
#if MEGA_SCORE_BUMP != 0
                                Adds(sc0, sc0, (float)MEGA_SCORE_BUMP, SLT);
                                Adds(sc1, sc1, (float)MEGA_SCORE_BUMP, SLT);
#endif
                                // diagnostic slice max (valid lanes only)
                                vector_scalar_fence();
                                for (int i = 0; i < tail && i < SLT; i++) {
                                    float v = sc0.GetValue(i); if (v > m0) m0 = v;
                                    v = sc1.GetValue(i); if (v > m1) m1 = v;
                                }
                                scalar_vector_fence();
                                Adds(sc0, sc0, -sh0, SLT);
                                Adds(sc1, sc1, -sh1, SLT);
                                Exp(ex0, sc0, SLT);
                                Exp(ex1, sc1, SLT);
                                Cast(hb[0], ex0, RoundMode::CAST_RINT, SLT);
                                Cast(hb[256], ex1, RoundMode::CAST_RINT, SLT);
                                if (tail >= SLT) {
                                    l0 += tree_sum(ex0, SLT);
                                    l1 += tree_sum(ex1, SLT);
                                } else {
                                    vector_scalar_fence();
                                    for (int i = 0; i < tail; i++) { l0 += ex0.GetValue(i); }
                                    for (int i = 0; i < tail; i++) { l1 += ex1.GetValue(i); }
                                    scalar_vector_fence();
                                }
                                if (sub == 0) {
                                    SetFlag<HardEvent::V_MTE3>(eV3); WaitFlag<HardEvent::V_MTE3>(eV3);
                                    DataCopy(tprobs2P[((int64_t)cid * NTM + (c + d) / SLT) * 512], hb, 512);
                                    SetFlag<HardEvent::MTE3_V>(eM3V); WaitFlag<HardEvent::MTE3_V>(eM3V);
                                }
                            }
                            // v7: tail chunk fully written -> publish its index
                            // (the whole <=1024 tail is one chunk).
                            if (sub == 0) {
                                seq_write(gseq, SEQ_WANT(cid, PR_PROBS),
                                          tag + li * PROG_SPAN + c / 1024 + 1,
                                          sq, sqWhich ^= 1, eV3, eM3V);
                            }
                        }
                    }
                }
                if (sub == 0) {
                    // R27: part_m normally publishes this layer's slice max --
                    // that is exactly what the next layer carries as its shift.
                    // On the LAST layer nobody carries it, so publish the shift
                    // that was actually used instead; that is the only way the
                    // gate can rebuild the exp reference exactly (a common
                    // shift is invisible end to end, but part_l/partAcc are not
                    // shift invariant).
                    if (sid == 0) {
                        put_scalar16(tShiftHistory, (li & 1) * (KVH * 2) + kv * 2, m0, t16, eV3, eM3V);
                        put_scalar16(tShiftHistory, (li & 1) * (KVH * 2) + kv * 2 + 1, m1, t16, eV3, eM3V);
                    }
                    const bool last = (li + 1 == lstop);
                    put_scalar16(tPartM, (kv * 2) * NSL + sid, last ? sh0 : m0, t16, eV3, eM3V);
                    put_scalar16(tPartM, (kv * 2 + 1) * NSL + sid, last ? sh1 : m1, t16, eV3, eM3V);
                    put_scalar16(tPartL, (kv * 2) * NSL + sid, l0, t16, eV3, eM3V);
                    put_scalar16(tPartL, (kv * 2 + 1) * NSL + sid, l1, t16, eV3, eM3V);
                    // Retirement alone is not a publication fence between GM
                    // addresses. Confirm both denominator values are visible
                    // before a consumer can observe the R+3 ready word.
                    SetFlag<HardEvent::MTE3_MTE2>(eM3M2); WaitFlag<HardEvent::MTE3_MTE2>(eM3M2);
                    for (;;) {
                        DataCopy(pml, tPartL[((int64_t)(kv * 2) * NSL + sid) * 16], 64);
                        SetFlag<HardEvent::MTE2_S>(eM2S); WaitFlag<HardEvent::MTE2_S>(eM2S);
                        if (pml.GetValue(0) == l0 && pml.GetValue(NSL * 16) == l1) break;
                    }
                    if (sid == 0) {
                        for (;;) {
                            DataCopy(pml, tShiftHistory[(li & 1) * (KVH * 2 * 16) + kv * 2 * 16], 32);
                            SetFlag<HardEvent::MTE2_S>(eM2S); WaitFlag<HardEvent::MTE2_S>(eM2S);
                            if (pml.GetValue(0) == m0 && pml.GetValue(16) == m1) break;
                        }
                    }

                }

                if (sub == 0) { seq_write(gseq, SEQ_WANT(cid, R + 3), tag + R + 3, sq, sqWhich ^= 1, eV3, eM3V); }

                // ---- stage 5: combine. Leaders gate on: (a) all 3 producer
                // groups' R+3 seqs -- covers part_m/l (AIV-written, MTE3,
                // ordered behind each group's own R+3 write) even for DEAD
                // slices; (b) live slices' AIC ASEQ(R+4) -- partAcc fixpipe
                // visibility (confirmed by each AIC's own sentinel wait). ----
                if (sub == 0 && leader) {
                    poll_seq(gseqp + SEQ_WANT(kv, R + 3), tag + R + 3);
                    poll_seq(gseqp + SEQ_WANT(kv + 8, R + 3), tag + R + 3);
                    poll_seq(gseqp + SEQ_WANT(kv + 16, R + 3), tag + R + 3);
                    aic_gate_poll1(gseqp, R + 4, kv, tag + R + 4);
                    if (sblk1 > 0) { aic_gate_poll1(gseqp, R + 4, kv + 8, tag + R + 4); }
                    if (sblk2 > 0) { aic_gate_poll1(gseqp, R + 4, kv + 16, tag + R + 4); }
                    // combine heads 2kv, 2kv+1 over 3 slice partials.
                    // R21: the probs in GM are exp(scaled score) with a GLOBAL
                    // shift of 0, so every live slice's partAcc/part_l already
                    // carries the same scale -- no exp(m_s - mg) rescale, just
                    // sum the live partials and divide by the summed totals.
                    // (also drops the two part_m MTE2 round-trips from the
                    // leader's critical path)
                    const bool live[NSL] = { sblk > 0, sblk1 > 0, sblk2 > 0 };
                    float ll[2][NSL];
                    for (int h = 0; h < 2; h++) {
                        Duplicate(num, 0.0f, D);
                        for (int s = 0; s < NSL; s++) {
                            if (!live[s]) continue;
                            LocalTensor<float> av = (s & 1) ? accb : acc;
                            // Ping-pong alone does not protect an operand from
                            // an MTE2 reload overtaking its pending vector Add.
                            SetFlag<HardEvent::V_MTE2>(eV2); WaitFlag<HardEvent::V_MTE2>(eV2);
                            DataCopy(av, tPartAcc[((int64_t)(kv * NSL + s) * 16 + h) * D], D);
                            SetFlag<HardEvent::MTE2_V>(eM2V); WaitFlag<HardEvent::MTE2_V>(eM2V);
                            Add(num, num, av, D);
                        }
                        if (h == 0) {
                            // read the totals only NOW: the partAcc MTE2s above
                            // are what give the producers' part_l MTE3 time to
                            // reach L2 (see R22 -- taking it straight off the
                            // R+3 gate was racy).
                            DataCopy(pml, tPartL[(int64_t)(kv * 2) * NSL * 16], 6 * 16);
                            SetFlag<HardEvent::MTE2_S>(eM2S); WaitFlag<HardEvent::MTE2_S>(eM2S);
                            for (int q = 0; q < 2; q++) {
                                for (int q2 = 0; q2 < NSL; q2++) {
                                    ll[q][q2] = pml.GetValue((q * NSL + q2) * 16);
                                }
                            }
                        }
                        float den = 0.0f;
                        for (int s = 0; s < NSL; s++) { if (live[s]) den += ll[h][s]; }
                        Muls(num, num, 1.0f / den, D);
                        Cast(hb[h * 128], num, RoundMode::CAST_RINT, D);
                    }
                    bcast_pack(pack, hb[0]);
                    SetFlag<HardEvent::V_MTE3>(eV3); WaitFlag<HardEvent::V_MTE3>(eV3);
                    DataCopy(tattnP[(int64_t)(kv * 2 * D) * A_PACK_ROWS], pack, A_PACK_CHUNK);
                    SetFlag<HardEvent::MTE3_V>(eM3V); WaitFlag<HardEvent::MTE3_V>(eM3V);
                    // B_ATT go: leaders only; AICs poll all 8 leader words
                    seq_write(gseq, SEQ_WANT(cid, R + 5), tag + R + 5, sq, sqWhich ^= 1, eV3, eM3V);
                }
            }

            TRACE_V(li, 4);
            // ---- stage 6: resid1 + rms2 -> xn2P ----
            int wpend = 0;
            {
                // v8: while waiting for all 22 AIC "o" words, warm the paired
                // AIC's gu weight slab (chunks 4..31; 0..3 are the AIC's ring)
                // into L2.  Each sub takes half, so both MTE2 engines work.
                const int64_t gub = (int64_t)li * GU_LAYER + (int64_t)cid * (32 * 32768);
                // R18b: the gu warm only pays when the KV stream is small enough
                // to leave gu resident in L2 (measured: +3% at ctx=1, -0.5% at
                // 40k where 168MB/layer of KV flushes it straight back out).
                if (sl < 4096) {
                    aiv_warm_gate(gseqp, R + 5, tag + R + 5, 0, 22, tWguA,
                                  gub + (int64_t)(sub ? 18 : 4) * 32768,
                                  gub + (int64_t)(sub ? 32 : 18) * 32768,
                                  pw, pw[WARM_SLOT], wpend);
                } else {
                    aic_gate_poll(gseqp, R + 5, tag + R + 5, 0, 22);
                }
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
                if (sub == 0) { pack_out(packw, xn, txn2P, H / 256, eV3, eM3V); }
            }
            if (sub == 0) { seq_write(gseq, SEQ_WANT(cid, R + 6), tag + R + 6, sq, sqWhich ^= 1, eV3, eM3V); }
            aiv_warm_drain(wpend);   // off the critical path: AIC's gu runs on xn2

            TRACE_V(li, 5);
            // ---- stage 7: sharded SiLU -> hP. Interleaved GU assigns
            // gate[256] and up[256] to the same producer AIC. Legacy mode
            // waits for the two original producer slabs. ----
            {
#if MEGA_GU_INTERLEAVE
                // The paired gate/up rows are produced by this AIC alone.
                aic_gate_poll1(gseqp, R + 6, cid, tag + R + 6);
                DataCopy(ggu, tguOut[cid * 512], 256);
                DataCopy(uup, tguOut[cid * 512 + 256], 256);
#else
                aic_gate_poll1(gseqp, R + 6, cid >> 1, tag + R + 6);        // gate cols
                aic_gate_poll1(gseqp, R + 6, 12 + (cid >> 1), tag + R + 6); // up cols
                DataCopy(ggu, tguOut[cid * 256], 256);
                DataCopy(uup, tguOut[II + cid * 256], 256);
#endif
                SetFlag<HardEvent::MTE2_V>(eM2V); WaitFlag<HardEvent::MTE2_V>(eM2V);
                Sigmoid(ex0, ggu, sig, 256);
                Mul(ggu, ggu, ex0, 256);                   // g * sig(g)
                Mul(ggu, ggu, uup, 256);                   // silu(g) * u
                Cast(hb[0], ggu, RoundMode::CAST_RINT, 256);
                if (sub == 0) {
                    bcast_pack(pack, hb[0]);
                    SetFlag<HardEvent::V_MTE3>(eV3); WaitFlag<HardEvent::V_MTE3>(eV3);
                    DataCopy(thP[(int64_t)cid * A_PACK_CHUNK], pack, A_PACK_CHUNK);
                    SetFlag<HardEvent::MTE3_V>(eM3V); WaitFlag<HardEvent::MTE3_V>(eM3V);
                }
            }
            if (sub == 0) { seq_write(gseq, SEQ_WANT(cid, R + 7), tag + R + 7, sq, sqWhich ^= 1, eV3, eM3V); }

            TRACE_V(li, 6);
            // ---- stage 8: resid2 ----
            {
                aic_gate_poll(gseqp, R + 7, tag + R + 7, 0, 22);  // down done
                DataCopy(xf, tdOut, H);                      // down row 0 (f32)
                SetFlag<HardEvent::MTE2_V>(eM2V); WaitFlag<HardEvent::MTE2_V>(eM2V);
                Cast(xs, xm, RoundMode::CAST_NONE, H);
                Add(xs, xs, xf, H);
                Cast(xm, xs, RoundMode::CAST_RINT, H);
            }
        }

        TRACE_V(L, 0);
        // ---- final: rmsf -> xnfP, argmax ----
        {
            Cast(xf, xm, RoundMode::CAST_NONE, H);
            {
                GlobalTensor<bfloat16_t> tg; tg.SetGlobalBuffer(gRmsfw);
                DataCopy(pack, tg, H);
            }
            SetFlag<HardEvent::MTE2_V>(eM2V); WaitFlag<HardEvent::MTE2_V>(eM2V);
            rms_norm(xf, xs, pack, xn, H, t16);
            if (sub == 0) { pack_out(packw, xn, txnfP, H / 256, eV3, eM3V); }
        }
        if (sub == 0) { seq_write(gseq, SEQ_WANT(cid, 224), tag + 224, sq, sqWhich ^= 1, eV3, eM3V); }

        {
            // 1:1 gate: this AIV's argmax reads only its own AIC's span
            TRACE_V(L, 1);
#if !MEGA_LM_STREAM
            aic_gate_poll1(gseqp, 224, cid, tag + 224);
#endif
            TRACE_V(L, 2);
            for (int ch = 0; ch < NCH; ch++) {          // argmax over own span
#if MEGA_LM_STREAM
                aiv_wait_prog(gseqp, PR_LM, cid, tag + ch + 1);
#endif
                // ping-pong buffers; NO reload (MTE2 issues can be hoisted
                // past PipeBarrier -- never reload a tensor V ops still read)
                LocalTensor<float> base = (ch & 1) ? lgcb : lgc;
                LocalTensor<float> work = (ch & 1) ? lgc : lgcb;
                LocalTensor<float> idv = (ch & 1) ? idxcb : idxc;
                SetFlag<HardEvent::V_MTE2>(eV2); WaitFlag<HardEvent::V_MTE2>(eV2);
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
        TRACE_V(L, 3);
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

// Host-only packing descriptor; loading a library must not require callers
// to guess the device's compile-time layouts. Legacy libraries lack this API.
extern "C" uint32_t megak_layout() {
    return static_cast<uint32_t>(NC_LM)
        | (static_cast<uint32_t>(MEGA_COMPACT_A) << 16)
        | (static_cast<uint32_t>(MEGA_GU_INTERLEAVE) << 17)
        | (static_cast<uint32_t>(MEGA_LM_STREAM) << 18);
}
