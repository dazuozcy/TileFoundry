// EDIT MODEL STAGE. Keep its published/read edges consistent; see PORTING.md.
            {
                TRACE_V(li, 3);
                // ---- stage 4: SINGLE-PASS softmax -> probs2P, part_m/l ----
                // (1:1 gate: this AIV reads only its own AIC's score rows)
                // R13/2a: no whole-slice score barrier -- the pass takes its
                // 4096-position groups as the AIC publishes them (see the
                // PR_SCORES wait inside the loop), so it runs under gemm1's K
                // stream instead of after it.
                // R21: the max pass is GONE.  Softmax is shift invariant, so
                // exp(x) with a GLOBAL shift of 0 is exactly as correct as
                // exp(x - m_slice) as long as no exponent overflows f32
                // (needs x > 88) or underflows bf16 (x < -87); the measured
                // worst scaled score on the real model is 27.  Scores are now
                // read ONCE instead of twice (-2.6MB/layer of HBM) and combine
                // no longer rescales by exp(m_s - mg): every live slice's
                // partial carries the same shift, so it just sums them and
                // divides by the summed row totals.  part_m is still
                // published (from a cheap in-pass reduce) so the gate keeps
                // verifying the score values end to end.
                float m0 = NEG_INF, m1 = NEG_INF, l0 = 0.0f, l1 = 0.0f;
                const float scale = 0.08838834764831845f;  // 1/sqrt(128)
                // R18: sub1's softmax is dead work -- every output of stage 4
                // (probs2P, part_m/l, R+3, combine) is sub0-only, so the second
                // AIV of each group was re-reading the whole score slice twice
                // for nothing (~146MB/token of HBM reads).
                if (sub == 0 && sblk > 0) {
                    // v6: process CW=1024 positions per iteration (4 tiles) to
                    // collapse the per-tile MTE2 latency and batch the probs2P
                    // write; the tail (sblk%CW) falls back to the 256 loop.
                    const int64_t sbase0 = ((int64_t)cid * 16) * STEP + s0;
                    const int64_t sbase1 = ((int64_t)cid * 16 + 1) * STEP + s0;
                    // R27: every slice of a head uses ONE common shift, so the
                    // three partials stay commensurable and combine can keep
                    // summing them.  exp(x-sh) is exact for ANY sh (it cancels
                    // in num/den); sh only has to stay within ~88 of the max.
                    //   li > 0 : slice 0's max as published by the layer below.
                    //            The value lives in a separate parity bank until
                    //            every slice has finished the current layer.
                    //   li == 0: measure slice 0 once.  Every core reads the
                    //            SAME region, so all three slices get an
                    //            identical value; layer 0 is the only layer
                    //            with no history to carry.
                    if (li > 0) {
                        // Both heads are contiguous in the immutable history bank;
                        // current maxima are published to the other bank.
                        DataCopy(pml, tShiftHistory[((li - 1) & 1) * (KVH * 2 * 16) + kv * 2 * 16], 32);
                        SetFlag<HardEvent::MTE2_S>(eM2S); WaitFlag<HardEvent::MTE2_S>(eM2S);
                        sh0 = pml.GetValue(0);
                        sh1 = pml.GetValue(16);
                        // Sanity guard; layer 0 always bootstraps and publishes
                        // this launch's history before layer 1 reads it.
                        // The band only rejects NaN/inf/absurd values -- a legit
                        // shift can be large (any constant offset is valid).
                        if (!(sh0 > -1e5f && sh0 < 1e5f)) sh0 = 0.0f;
                        if (!(sh1 > -1e5f && sh1 < 1e5f)) sh1 = 0.0f;
                    } else {
                        const int64_t b0 = ((int64_t)kv * 16) * STEP;
                        const int sb0 = (sl + 1 < thirds) ? (sl + 1) : thirds;
                        for (int c = 0; c < sb0; c += 4096) {
                            aiv_wait_prog(gseqp, PR_SCORES, kv, tag + c / 4096 + 1);
                            const int cl = (c + 4096 <= sb0) ? 4096 : (sb0 - c);
                            if (cl == 4096) {
                                // Protect the reduction operand until its vector readers finish.
                                SetFlag<HardEvent::V_MTE2>(eV2); WaitFlag<HardEvent::V_MTE2>(eV2);
                                DataCopy(pwf[0], tScores[b0 + c], 4096);
                                DataCopy(pwf[4096], tScores[b0 + STEP + c], 4096);
                                SetFlag<HardEvent::MTE2_V>(eM2V); WaitFlag<HardEvent::MTE2_V>(eM2V);
                                Muls(pwf[0], pwf[0], scale, 4096);
                                Muls(pwf[4096], pwf[4096], scale, 4096);
#if MEGA_SCORE_BUMP != 0
                                Adds(pwf[0], pwf[0], (float)MEGA_SCORE_BUMP, 4096);
                                Adds(pwf[4096], pwf[4096], (float)MEGA_SCORE_BUMP, 4096);
#endif
                                float tm = tree_max(pwf[0], 4096); if (tm > sh0) sh0 = tm;
                                tm = tree_max(pwf[4096], 4096); if (tm > sh1) sh1 = tm;
                            } else {
                                for (int d = 0; d < cl; d += SLT) {
                                    const int tail = cl - d;
                                    // Protect the reduction operand until its vector readers finish.
                                    SetFlag<HardEvent::V_MTE2>(eV2); WaitFlag<HardEvent::V_MTE2>(eV2);
                                    DataCopy(sc0, tScores[b0 + c + d], SLT);
                                    DataCopy(sc1, tScores[b0 + STEP + c + d], SLT);
                                    SetFlag<HardEvent::MTE2_V>(eM2V); WaitFlag<HardEvent::MTE2_V>(eM2V);
                                    Muls(sc0, sc0, scale, SLT);
                                    Muls(sc1, sc1, scale, SLT);
#if MEGA_SCORE_BUMP != 0
                                    Adds(sc0, sc0, (float)MEGA_SCORE_BUMP, SLT);
                                    Adds(sc1, sc1, (float)MEGA_SCORE_BUMP, SLT);
#endif
                                    if (tail >= SLT) {
                                        Add(mx0, sc0, sc0, SLT);
                                        Add(mx1, sc1, sc1, SLT);
                                        float tm = 0.5f * tree_max(mx0, SLT); if (tm > sh0) sh0 = tm;
                                        tm = 0.5f * tree_max(mx1, SLT); if (tm > sh1) sh1 = tm;
                                    } else {
                                        vector_scalar_fence();
                                        for (int i = 0; i < tail; i++) {
                                            float v = sc0.GetValue(i); if (v > sh0) sh0 = v;
                                            v = sc1.GetValue(i); if (v > sh1) sh1 = v;
                                        }
                                        scalar_vector_fence();
                                    }
                                }
                            }
                        }
                    }
                    // R24: one pass over the scores.  probs are exp(x - sh) with
                    // a per-AIV carried shift (R27); the in-pass max feeds the
                    // NEXT layer's shift, so nothing here is model specific.
                    for (int c = 0; c < sblk; c += 1024) {          // exp/probs/l
                        // R13/2a: the AIC publishes one 4096-position group at
                        // a time -- wait for the group this chunk lives in.
                        if ((c & 4095) == 0) {
                            aiv_wait_prog(gseqp, PR_SCORES, cid,
                                          tag + li * PROG_SPAN + c / 4096 + 1);
                        }
                        const int cl = (c + 1024 <= sblk) ? 1024 : (sblk - c);
                        if (cl == 1024) {
                            // Protect the reduction operand until its vector readers finish.
                            SetFlag<HardEvent::V_MTE2>(eV2); WaitFlag<HardEvent::V_MTE2>(eV2);
                            DataCopy(w0, tScores[sbase0 + c], 1024);
                            DataCopy(w1, tScores[sbase1 + c], 1024);
                            SetFlag<HardEvent::MTE2_V>(eM2V); WaitFlag<HardEvent::MTE2_V>(eM2V);
                            Muls(w0, w0, scale, 1024);
                            Muls(w1, w1, scale, 1024);
#if MEGA_SCORE_BUMP != 0
                            Adds(w0, w0, (float)MEGA_SCORE_BUMP, 1024);
                            Adds(w1, w1, (float)MEGA_SCORE_BUMP, 1024);
#endif
                            // diagnostic slice max only (Add is an exact 2x
                            // copy and, unlike Muls by 1.0, is not folded away)
                            Add(mx0, w0, w0, 1024);
                            Add(mx1, w1, w1, 1024);
                            Adds(w0, w0, -sh0, 1024);
                            Adds(w1, w1, -sh1, 1024);
                            Exp(w0, w0, 1024);
                            Exp(w1, w1, 1024);
                            float tm = 0.5f * tree_max(mx0, 1024); if (tm > m0) m0 = tm;
                            tm = 0.5f * tree_max(mx1, 1024); if (tm > m1) m1 = tm;
                            // land probs BEFORE tree_sum (it destroys in place!)
                            // R16 compact: [tile][head0 256][head1 256] bf16
                            for (int q = 0; q < 4; q++) {
                                Cast(hbw[q * 512], w0[q * 256], RoundMode::CAST_RINT, 256);
                                Cast(hbw[q * 512 + 256], w1[q * 256], RoundMode::CAST_RINT, 256);
                            }
                            l0 += tree_sum(w0, 1024);
                            l1 += tree_sum(w1, 1024);
                            if (sub == 0) {
                                SetFlag<HardEvent::V_MTE3>(eV3); WaitFlag<HardEvent::V_MTE3>(eV3);
                                DataCopy(tprobs2P[((int64_t)cid * NTM + c / SLT) * 512], hbw, 4 * 512);
                                SetFlag<HardEvent::MTE3_V>(eM3V); WaitFlag<HardEvent::MTE3_V>(eM3V);
                                // v7: publish this 1024-chunk (4 tiles).  The
                                // seq MTE3 is ordered after the probs MTE3, so
                                // gemm2 sees the progress word only once the
                                // chunk is actually in GM.
                                seq_write(gseq, SEQ_WANT(cid, PR_PROBS),
                                          tag + li * PROG_SPAN + c / 1024 + 1,
                                          sq, sqWhich ^= 1, eV3, eM3V);
                            }
                        } else {
                            for (int d = 0; d < cl; d += SLT) {
                                const int tail = cl - d;
                                // Protect the reduction operand until its vector readers finish.
                                SetFlag<HardEvent::V_MTE2>(eV2); WaitFlag<HardEvent::V_MTE2>(eV2);
                                DataCopy(sc0, tScores[sbase0 + c + d], SLT);
                                DataCopy(sc1, tScores[sbase1 + c + d], SLT);
                                SetFlag<HardEvent::MTE2_V>(eM2V); WaitFlag<HardEvent::MTE2_V>(eM2V);
                                Muls(sc0, sc0, scale, SLT);
                                Muls(sc1, sc1, scale, SLT);
#if MEGA_SCORE_BUMP != 0
                                Adds(sc0, sc0, (float)MEGA_SCORE_BUMP, SLT);
                                Adds(sc1, sc1, (float)MEGA_SCORE_BUMP, SLT);
#endif
                                // diagnostic slice max (valid lanes only)
                                vector_scalar_fence();
                                for (int i = 0; i < tail && i < SLT; i++) {
                                    float v = sc0.GetValue(i); if (v > m0) m0 = v;
                                    v = sc1.GetValue(i); if (v > m1) m1 = v;
                                }
                                scalar_vector_fence();
                                Adds(sc0, sc0, -sh0, SLT);
                                Adds(sc1, sc1, -sh1, SLT);
                                Exp(ex0, sc0, SLT);
                                Exp(ex1, sc1, SLT);
                                Cast(hb[0], ex0, RoundMode::CAST_RINT, SLT);
                                Cast(hb[256], ex1, RoundMode::CAST_RINT, SLT);
                                if (tail >= SLT) {
                                    l0 += tree_sum(ex0, SLT);
                                    l1 += tree_sum(ex1, SLT);
                                } else {
                                    vector_scalar_fence();
                                    for (int i = 0; i < tail; i++) { l0 += ex0.GetValue(i); }
                                    for (int i = 0; i < tail; i++) { l1 += ex1.GetValue(i); }
                                    scalar_vector_fence();
                                }
                                if (sub == 0) {
                                    SetFlag<HardEvent::V_MTE3>(eV3); WaitFlag<HardEvent::V_MTE3>(eV3);
                                    DataCopy(tprobs2P[((int64_t)cid * NTM + (c + d) / SLT) * 512], hb, 512);
                                    SetFlag<HardEvent::MTE3_V>(eM3V); WaitFlag<HardEvent::MTE3_V>(eM3V);
                                }
                            }
                            // v7: tail chunk fully written -> publish its index
                            // (the whole <=1024 tail is one chunk).
                            if (sub == 0) {
                                seq_write(gseq, SEQ_WANT(cid, PR_PROBS),
                                          tag + li * PROG_SPAN + c / 1024 + 1,
                                          sq, sqWhich ^= 1, eV3, eM3V);
                            }
                        }
                    }
                }
                if (sub == 0) {
                    // R27: part_m normally publishes this layer's slice max --
                    // that is exactly what the next layer carries as its shift.
                    // On the LAST layer nobody carries it, so publish the shift
                    // that was actually used instead; that is the only way the
                    // gate can rebuild the exp reference exactly (a common
                    // shift is invisible end to end, but part_l/partAcc are not
                    // shift invariant).
                    if (sid == 0) {
                        put_scalar16(tShiftHistory, (li & 1) * (KVH * 2) + kv * 2, m0, t16, eV3, eM3V);
                        put_scalar16(tShiftHistory, (li & 1) * (KVH * 2) + kv * 2 + 1, m1, t16, eV3, eM3V);
                    }
                    const bool last = (li + 1 == lstop);
                    put_scalar16(tPartM, (kv * 2) * NSL + sid, last ? sh0 : m0, t16, eV3, eM3V);
                    put_scalar16(tPartM, (kv * 2 + 1) * NSL + sid, last ? sh1 : m1, t16, eV3, eM3V);
                    put_scalar16(tPartL, (kv * 2) * NSL + sid, l0, t16, eV3, eM3V);
                    put_scalar16(tPartL, (kv * 2 + 1) * NSL + sid, l1, t16, eV3, eM3V);
                    // Retirement alone is not a publication fence between GM
                    // addresses. Confirm both denominator values are visible
                    // before a consumer can observe the R+3 ready word.
                    SetFlag<HardEvent::MTE3_MTE2>(eM3M2); WaitFlag<HardEvent::MTE3_MTE2>(eM3M2);
                    for (;;) {
                        DataCopy(pml, tPartL[((int64_t)(kv * 2) * NSL + sid) * 16], 64);
                        SetFlag<HardEvent::MTE2_S>(eM2S); WaitFlag<HardEvent::MTE2_S>(eM2S);
                        if (pml.GetValue(0) == l0 && pml.GetValue(NSL * 16) == l1) break;
                    }
                    if (sid == 0) {
                        for (;;) {
                            DataCopy(pml, tShiftHistory[(li & 1) * (KVH * 2 * 16) + kv * 2 * 16], 32);
                            SetFlag<HardEvent::MTE2_S>(eM2S); WaitFlag<HardEvent::MTE2_S>(eM2S);
                            if (pml.GetValue(0) == m0 && pml.GetValue(16) == m1) break;
                        }
                    }

                }

                if (sub == 0) { seq_write(gseq, SEQ_WANT(cid, R + 3), tag + R + 3, sq, sqWhich ^= 1, eV3, eM3V); }

                // ---- stage 5: combine. Leaders gate on: (a) all 3 producer
                // groups' R+3 seqs -- covers part_m/l (AIV-written, MTE3,
                // ordered behind each group's own R+3 write) even for DEAD
                // slices; (b) live slices' AIC ASEQ(R+4) -- partAcc fixpipe
                // visibility (confirmed by each AIC's own sentinel wait). ----
                if (sub == 0 && leader) {
                    poll_seq(gseqp + SEQ_WANT(kv, R + 3), tag + R + 3);
                    poll_seq(gseqp + SEQ_WANT(kv + 8, R + 3), tag + R + 3);
                    poll_seq(gseqp + SEQ_WANT(kv + 16, R + 3), tag + R + 3);
                    aic_gate_poll1(gseqp, R + 4, kv, tag + R + 4);
                    if (sblk1 > 0) { aic_gate_poll1(gseqp, R + 4, kv + 8, tag + R + 4); }
                    if (sblk2 > 0) { aic_gate_poll1(gseqp, R + 4, kv + 16, tag + R + 4); }
                    // combine heads 2kv, 2kv+1 over 3 slice partials.
                    // R21: the probs in GM are exp(scaled score) with a GLOBAL
                    // shift of 0, so every live slice's partAcc/part_l already
                    // carries the same scale -- no exp(m_s - mg) rescale, just
                    // sum the live partials and divide by the summed totals.
                    // (also drops the two part_m MTE2 round-trips from the
                    // leader's critical path)
                    const bool live[NSL] = { sblk > 0, sblk1 > 0, sblk2 > 0 };
                    float ll[2][NSL];
                    for (int h = 0; h < 2; h++) {
                        Duplicate(num, 0.0f, D);
                        for (int s = 0; s < NSL; s++) {
                            if (!live[s]) continue;
                            LocalTensor<float> av = (s & 1) ? accb : acc;
                            // Ping-pong alone does not protect an operand from
                            // an MTE2 reload overtaking its pending vector Add.
                            SetFlag<HardEvent::V_MTE2>(eV2); WaitFlag<HardEvent::V_MTE2>(eV2);
                            DataCopy(av, tPartAcc[((int64_t)(kv * NSL + s) * 16 + h) * D], D);
                            SetFlag<HardEvent::MTE2_V>(eM2V); WaitFlag<HardEvent::MTE2_V>(eM2V);
                            Add(num, num, av, D);
                        }
                        if (h == 0) {
                            // read the totals only NOW: the partAcc MTE2s above
                            // are what give the producers' part_l MTE3 time to
                            // reach L2 (see R22 -- taking it straight off the
                            // R+3 gate was racy).
                            DataCopy(pml, tPartL[(int64_t)(kv * 2) * NSL * 16], 6 * 16);
                            SetFlag<HardEvent::MTE2_S>(eM2S); WaitFlag<HardEvent::MTE2_S>(eM2S);
                            for (int q = 0; q < 2; q++) {
                                for (int q2 = 0; q2 < NSL; q2++) {
                                    ll[q][q2] = pml.GetValue((q * NSL + q2) * 16);
                                }
                            }
                        }
                        float den = 0.0f;
                        for (int s = 0; s < NSL; s++) { if (live[s]) den += ll[h][s]; }
                        Muls(num, num, 1.0f / den, D);
                        Cast(hb[h * 128], num, RoundMode::CAST_RINT, D);
                    }
                    bcast_pack(pack, hb[0]);
                    SetFlag<HardEvent::V_MTE3>(eV3); WaitFlag<HardEvent::V_MTE3>(eV3);
                    DataCopy(tattnP[(int64_t)(kv * 2 * D) * A_PACK_ROWS], pack, A_PACK_CHUNK);
                    SetFlag<HardEvent::MTE3_V>(eM3V); WaitFlag<HardEvent::MTE3_V>(eM3V);
                    // B_ATT go: leaders only; AICs poll all 8 leader words
                    seq_write(gseq, SEQ_WANT(cid, R + 5), tag + R + 5, sq, sqWhich ^= 1, eV3, eM3V);
                }
            }
