// EDIT MODEL STAGE. Keep its published/read edges consistent; see PORTING.md.
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
