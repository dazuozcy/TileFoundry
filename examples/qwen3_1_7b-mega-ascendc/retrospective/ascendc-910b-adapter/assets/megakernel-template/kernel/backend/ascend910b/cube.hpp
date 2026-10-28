#pragma once
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
