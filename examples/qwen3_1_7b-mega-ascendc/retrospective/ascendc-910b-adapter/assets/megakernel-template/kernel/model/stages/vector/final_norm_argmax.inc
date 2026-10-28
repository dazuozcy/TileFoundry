// EDIT MODEL FINAL. Exclude padded vocab and retain reader fences.
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
