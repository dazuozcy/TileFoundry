// EDIT MODEL STAGE. Keep its published/read edges consistent; see PORTING.md.
            // ---- stage 1: rms1 -> xn1P ----
            {
                Cast(xf, xm, RoundMode::CAST_NONE, H);
                {
                    GlobalTensor<bfloat16_t> tg; tg.SetGlobalBuffer(gRms1w + (int64_t)li * H);
                    DataCopy(pack, tg, H);   // gamma (bf16 view)
                }
                SetFlag<HardEvent::MTE2_V>(eM2V); WaitFlag<HardEvent::MTE2_V>(eM2V);
                rms_norm(xf, xs, pack, xn, H, t16);
                if (sub == 0) { pack_out(packw, xn, txn1P, H / 256, eV3, eM3V); }
            }
            if (sub == 0) { seq_write(gseq, SEQ_WANT(cid, R + 0), tag + R + 0, sq, sqWhich ^= 1, eV3, eM3V); }
