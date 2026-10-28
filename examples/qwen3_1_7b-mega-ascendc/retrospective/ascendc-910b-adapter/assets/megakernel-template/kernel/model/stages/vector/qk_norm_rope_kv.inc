// EDIT MODEL STAGE. Keep its published/read edges consistent; see PORTING.md.
            TRACE_V(li, 1);
            // ---- stage 2: qkv gate. This AIV reads only its own head's
            // q/k/v output columns, which come from a few AIC slabs -- poll
            // only those AICs so we decouple from the slowest of all 24
            // (removes the cross-block qkv land-skew from the critical path).
            {
                const int qc0 = kv * 256;                 // q head 2kv,2kv+1 cols
                const int kc0 = NQKV / 2 + kv * D;
                const int vc0 = 3 * NQKV / 4 + kv * D;
                aic_gate_poll(gseqp, R + 0, tag + R + 0, qc0 / 176, (qc0 + 255) / 176 + 1);
                aic_gate_poll(gseqp, R + 0, tag + R + 0, kc0 / 176, (kc0 + D - 1) / 176 + 1);
                aic_gate_poll(gseqp, R + 0, tag + R + 0, vc0 / 176, (vc0 + D - 1) / 176 + 1);
            }

            TRACE_V(li, 2);
            // ---- stage 3: qk-norm, rope, appends, q2P ----
            {
                // NOTE: direct f32 loads per region -- Cast(f32->f32,
                // CAST_NONE) is BROKEN on dav-2201 (corrupts data; verified
                // by probe p13norm). f32->bf16 / bf16->f32 casts are fine.
                DataCopy(q0f, tQkvOut[(kv * 2) * D], D);
                DataCopy(q1f, tQkvOut[(kv * 2 + 1) * D], D);
                DataCopy(kf, tQkvOut[NQKV / 2 + kv * D], D);
                DataCopy(qkv, tQkvOut[3 * NQKV / 4 + kv * D], D);
                DataCopy(cosr, tCosT[(int64_t)sl * D], D);
                DataCopy(sinr, tSinT[(int64_t)sl * D], D);
                {
                    GlobalTensor<bfloat16_t> tg; tg.SetGlobalBuffer(gQnw + (int64_t)li * D);
                    DataCopy(pack, tg, D);                       // q-norm gamma
                }
                {
                    GlobalTensor<bfloat16_t> tg; tg.SetGlobalBuffer(gKnw + (int64_t)li * D);
                    DataCopy(pack[128], tg, D);                  // k-norm gamma
                }
                SetFlag<HardEvent::MTE2_V>(eM2V); WaitFlag<HardEvent::MTE2_V>(eM2V);
                Cast(vb, qkv[0], RoundMode::CAST_RINT, D);       // f32 -> bf16
                rms_norm(q0f, rot, pack, q0b, D, t16);
                rms_norm(q1f, rot, pack, q1b, D, t16);
                rms_norm(kf, rot, pack[128], kb, D, t16);
                rope_apply(q0f, cosr, sinr, rot);
                rope_apply(q1f, cosr, sinr, rot);
                rope_apply(kf, cosr, sinr, rot);
                Cast(q0b, q0f, RoundMode::CAST_RINT, D);
                Cast(q1b, q1f, RoundMode::CAST_RINT, D);
                Cast(kb, kf, RoundMode::CAST_RINT, D);
                if (sub == 0) {
                    strided_rows(q2, q0b, 8, 0);
                    strided_rows(q2, q1b, 8, 1);
                    SetFlag<HardEvent::V_MTE3>(eV3); WaitFlag<HardEvent::V_MTE3>(eV3);
                    DataCopy(tq2P[(int64_t)cid * 8 * 256], q2, 8 * 256);
                    SetFlag<HardEvent::MTE3_V>(eM3V); WaitFlag<HardEvent::MTE3_V>(eM3V);
                    if (sid == sidOwner) {
                        const int iBlk = sl / 16;
                        const int iRow = sl % 16;
                        const int64_t kbase = (int64_t)(li * KVH + kv) * 8 * (SPADG / 16) * 256;
                        const int64_t vbase = (int64_t)(li * KVH + kv) * (SPADG / 16) * 8 * 256;
                        vector_scalar_fence();               // vb reads below
                        for (int j = 0; j < 8; j++) {
                            DataCopy(tKcP[kbase + (int64_t)(iBlk / 16) * 32768 +
                                          (int64_t)j * 4096 + (int64_t)(iBlk % 16) * 256 + iRow * 16],
                                     kb[j * 16], 16);
                        }
                        // V append: one 4KB RMW over all 8 d-blocks
                        // (replaces 8 serialized per-block RMWs, v3)
                        DataCopy(vfr, tVcP[vbase + (int64_t)iBlk * 8 * 256], 8 * 256);
                        SetFlag<HardEvent::MTE2_S>(eM2S); WaitFlag<HardEvent::MTE2_S>(eM2S);
                        for (int j = 0; j < 8; j++) {
                            for (int n = 0; n < 16; n++) {
                                vfr.SetValue(j * 256 + n * 16 + iRow, vb.GetValue(j * 16 + n));
                            }
                        }
                        event_t esApp = static_cast<event_t>(s3cnt++ & 3);  // one-shot ids
                        SetFlag<HardEvent::S_MTE3>(esApp); WaitFlag<HardEvent::S_MTE3>(esApp);
                        DataCopy(tVcP[vbase + (int64_t)iBlk * 8 * 256], vfr, 8 * 256);
                        SetFlag<HardEvent::MTE3_MTE2>(eM3M2); WaitFlag<HardEvent::MTE3_MTE2>(eM3M2);
                    }
                }
            }
            if (sub == 0) { seq_write(gseq, SEQ_WANT(cid, R + 2), tag + R + 2, sq, sqWhich ^= 1, eV3, eM3V); }
