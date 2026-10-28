// EDIT MODEL STAGE. Keep its published/read edges consistent; see PORTING.md.
            TRACE_V(li, 6);
            // ---- stage 8: resid2 ----
            {
                aic_gate_poll(gseqp, R + 7, tag + R + 7, 0, 22);  // down done
                DataCopy(xf, tdOut, H);                      // down row 0 (f32)
                SetFlag<HardEvent::MTE2_V>(eM2V); WaitFlag<HardEvent::MTE2_V>(eM2V);
                Cast(xs, xm, RoundMode::CAST_NONE, H);
                Add(xs, xs, xf, H);
                Cast(xm, xs, RoundMode::CAST_RINT, H);
            }
