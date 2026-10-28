// EDIT MODEL STAGE. Keep its published/read edges consistent; see PORTING.md.
            TRACE_V(li, 5);
            // ---- stage 7: sharded SiLU -> hP. Interleaved GU assigns
            // gate[256] and up[256] to the same producer AIC. Legacy mode
            // waits for the two original producer slabs. ----
            {
#if MEGA_GU_INTERLEAVE
                // The paired gate/up rows are produced by this AIC alone.
                aic_gate_poll1(gseqp, R + 6, cid, tag + R + 6);
                DataCopy(ggu, tguOut[cid * 512], 256);
                DataCopy(uup, tguOut[cid * 512 + 256], 256);
#else
                aic_gate_poll1(gseqp, R + 6, cid >> 1, tag + R + 6);        // gate cols
                aic_gate_poll1(gseqp, R + 6, 12 + (cid >> 1), tag + R + 6); // up cols
                DataCopy(ggu, tguOut[cid * 256], 256);
                DataCopy(uup, tguOut[II + cid * 256], 256);
#endif
                SetFlag<HardEvent::MTE2_V>(eM2V); WaitFlag<HardEvent::MTE2_V>(eM2V);
                Sigmoid(ex0, ggu, sig, 256);
                Mul(ggu, ggu, ex0, 256);                   // g * sig(g)
                Mul(ggu, ggu, uup, 256);                   // silu(g) * u
                Cast(hb[0], ggu, RoundMode::CAST_RINT, 256);
                if (sub == 0) {
                    bcast_pack(pack, hb[0]);
                    SetFlag<HardEvent::V_MTE3>(eV3); WaitFlag<HardEvent::V_MTE3>(eV3);
                    DataCopy(thP[(int64_t)cid * A_PACK_CHUNK], pack, A_PACK_CHUNK);
                    SetFlag<HardEvent::MTE3_V>(eM3V); WaitFlag<HardEvent::MTE3_V>(eM3V);
                }
            }
            if (sub == 0) { seq_write(gseq, SEQ_WANT(cid, R + 7), tag + R + 7, sq, sqWhich ^= 1, eV3, eM3V); }
