// EDIT MODEL STAGE. Keep its published/read edges consistent; see PORTING.md.
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
