// EDIT MODEL FINAL HEAD. Preserve live-logit landing and padding.
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
