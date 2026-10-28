// EDIT MODEL STAGE. Keep its published/read edges consistent; see PORTING.md.
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
