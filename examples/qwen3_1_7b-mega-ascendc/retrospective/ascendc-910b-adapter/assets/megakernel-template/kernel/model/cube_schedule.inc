#include "backend/ascend910b/cube_resources.inc"
        for (int li = 0; li < lstop; li++) {
            const int R = li * 8;
            TRACE_C(li, 0);
            // ---- arm this layer's sentinel words (GU uses live rows; other
            // regions use dead rows). See Down empty-core ack for live GU reuse.
            // Dead rows are never read by AIVs, so their arming is safe
            // regardless of how far the AIVs still are in layer li-1. The
            // previous layer's fixpipe for each region is confirmed landed
            // (this AIC self-polled it) before we re-arm. ----
            const int sr = 1;                  // dead row 1 (weight gemvs write mSize=2)
            const int sr2 = 2;                 // dead row 2 (scores/PV write mSize=3)
            aic_arm(gqkvF + sr * NQKV_PAD + qkvSC);
            if (sblk > 0) {
                // one dead-row sentinel per 4096-position score group (R13/2a) so
                // the AIV can start the max pass as tiles land instead of after
                // the whole slice.  group g ends at tile min(16g+15, nloc-1); the
                // last group's word IS the old single all-scores sentinel.
                for (int g = 0; g < ngrp; g++) {
                    const int ti = ((g + 1) * 16 - 1 < nloc - 1) ? (g + 1) * 16 - 1 : nloc - 1;
                    aic_arm(gscoresF + (int64_t)(cid * 16 + sr2) * STEP + s0 +
                            (int64_t)ti * SLT + (SLT - 1));
                }
                aic_arm(gpartAccF + ((kv * NSL + sid) * 16 + sr2) * D + (D - 1));
            }
            if (odActive) {
                aic_arm(goF + sr * NO_PAD + odSC);
                aic_arm(gdF + sr * ND_PAD + odSC);
            }
            // Certify the live row read by SiLU, one marker per output tile.
            for (int nt = 0; nt < 4; nt++) aic_arm(gguF + cid * 512 + (nt + 1) * 128 - 1);
            // Retire scalar sentinel stores/cache cleans before DMA/Mmad
            // can reach the corresponding Fixpipe destination.
            SetFlag<HardEvent::S_MTE2>((event_t)0);
            WaitFlag<HardEvent::S_MTE2>((event_t)0);
#include "model/stages/cube/qkv.inc"
#include "model/stages/cube/attention_qk_pv.inc"
#include "model/stages/cube/output_projection.inc"
#include "model/stages/cube/gate_up.inc"
#include "model/stages/cube/down.inc"
            TRACE_C(li, 20);      // down done
        }

#include "model/stages/cube/lm_head.inc"
