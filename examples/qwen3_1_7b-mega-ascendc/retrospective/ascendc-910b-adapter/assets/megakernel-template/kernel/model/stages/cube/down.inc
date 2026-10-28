// EDIT MODEL STAGE. Keep its published/read edges consistent; see PORTING.md.
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
