// EDIT MODEL STAGE. Keep its published/read edges consistent; see PORTING.md.
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
