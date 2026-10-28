#pragma once
// Prepare one A chunk: compact two ND rows, or legacy 16-row broadcast.
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
