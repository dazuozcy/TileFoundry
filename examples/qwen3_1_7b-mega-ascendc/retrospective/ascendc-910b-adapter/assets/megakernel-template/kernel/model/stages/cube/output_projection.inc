// EDIT MODEL STAGE. Keep its published/read edges consistent; see PORTING.md.
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
