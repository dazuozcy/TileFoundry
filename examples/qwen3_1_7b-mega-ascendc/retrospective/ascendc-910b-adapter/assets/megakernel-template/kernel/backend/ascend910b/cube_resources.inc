// ADAPT HARDWARE/PLACEMENT: buffers, aliases, channels and views.
        // ================= CUBE SIDE =================
        const int cid = GetBlockIdx();
        TRACE_C(L, 0);
        const int kv = cid % KVH;
        const int sid = cid / KVH;
        const int s0 = sid * thirds;
        const int e0 = (sl + 1 < s0 + thirds) ? (sl + 1) : (s0 + thirds);
        const int sblk = (e0 > s0) ? (e0 - s0) : 0;
        const int nloc = (sblk + SLT - 1) / SLT;
        const int ngrp = (sblk + 4095) / 4096;   // 16-tile score groups (R13/2a)

        TPipe pipe;
        TBuf<TPosition::A1> a1; TBuf<TPosition::B1> b1;
        TBuf<TPosition::A1> a1b; TBuf<TPosition::B1> b1b;
        TBuf<TPosition::A1> ab1;                     // A-pack group staging
        TBuf<TPosition::B1> rb0, rb1, rb2, rb3;      // prefetch ring (B chunks)
        TBuf<TPosition::A2> a2; TBuf<TPosition::B2> b2; TBuf<TPosition::CO1> c1;
        pipe.InitBuffer(a1, 16 * KTILE * sizeof(bfloat16_t));
        pipe.InitBuffer(b1, 64 * 1024);
        pipe.InitBuffer(a1b, 16 * KTILE * sizeof(bfloat16_t));
        pipe.InitBuffer(b1b, 64 * 1024);
        pipe.InitBuffer(ab1, 8 * 16 * KTILE * sizeof(bfloat16_t));
        pipe.InitBuffer(rb0, 64 * 1024);
        pipe.InitBuffer(rb1, 64 * 1024);
        pipe.InitBuffer(rb2, 64 * 1024);
        pipe.InitBuffer(rb3, 64 * 1024);
        pipe.InitBuffer(a2, 16 * KTILE * sizeof(bfloat16_t));
        pipe.InitBuffer(b2, 64 * 1024);
        pipe.InitBuffer(c1, 16 * 256 * sizeof(float));
        pipe.FetchEventID<HardEvent::MTE2_MTE1>();
        CubeCtx C;
        C.ev0 = static_cast<event_t>(pipe.FetchEventID<HardEvent::MTE2_MTE1>());
        pipe.FetchEventID<HardEvent::MTE1_M>();
        C.ev1 = static_cast<event_t>(pipe.FetchEventID<HardEvent::MTE1_M>());
        pipe.FetchEventID<HardEvent::M_FIX>();
        C.ev2 = static_cast<event_t>(pipe.FetchEventID<HardEvent::M_FIX>());
        pipe.FetchEventID<HardEvent::M_MTE1>();
        C.ev3 = static_cast<event_t>(pipe.FetchEventID<HardEvent::M_MTE1>());
        pipe.FetchEventID<HardEvent::MTE1_MTE2>();
        C.ev5 = static_cast<event_t>(pipe.FetchEventID<HardEvent::MTE1_MTE2>());
        pipe.FetchEventID<HardEvent::FIX_M>();
        C.ev4 = static_cast<event_t>(pipe.FetchEventID<HardEvent::FIX_M>());
        C.ev6 = static_cast<event_t>(pipe.FetchEventID<HardEvent::MTE2_MTE1>());
        C.ev7 = static_cast<event_t>(pipe.FetchEventID<HardEvent::MTE1_MTE2>());
        // FetchEventID only peeks at an unallocated channel. The two
        // outstanding score/PV ping-pong transfers MUST have distinct IDs.
        // GEMV drains the ring before these stages, so channel 1 is free.
        C.ev0 = (event_t)0; C.ev6 = (event_t)1;
        C.ev5 = (event_t)0; C.ev7 = (event_t)1;
        // v5 ring: FetchEventID only PEEKS (always 0), so the 4 slot channels
        // are hardcoded (distinct ids 1..4 verified on dav-2201).
        C.eM2R[0] = (event_t)1; C.eM2R[1] = (event_t)2;
        C.eM2R[2] = (event_t)3; C.eM2R[3] = (event_t)4;
        C.eM2R[4] = (event_t)6; C.eM2R[5] = (event_t)7;
        C.eM1R[0] = (event_t)1; C.eM1R[1] = (event_t)2;
        C.eM1R[2] = (event_t)3; C.eM1R[3] = (event_t)4;
        C.eM1R[4] = (event_t)6; C.eM1R[5] = (event_t)7;
        C.evA = (event_t)5;
        C.cid = cid; C.prodCount = 0; C.consCount = 0;
        C.abuf = ab1.Get<bfloat16_t>();
        C.ringb[0] = rb0.Get<bfloat16_t>(); C.ringb[1] = rb1.Get<bfloat16_t>();
        C.ringb[2] = rb2.Get<bfloat16_t>(); C.ringb[3] = rb3.Get<bfloat16_t>();
        C.la1 = a1.Get<bfloat16_t>();
        C.lb1 = b1.Get<bfloat16_t>();
        C.la1b = a1b.Get<bfloat16_t>();
        C.lb1b = b1b.Get<bfloat16_t>();
        C.ringb[4] = C.lb1; C.ringb[5] = C.lb1b;   // alias gemm1/2 buffers
        C.la2 = a2.Get<bfloat16_t>();
        C.lb2 = b2.Get<bfloat16_t>();
        C.lc1 = c1.Get<float>();

        __gm__ bfloat16_t* gWqkv = (__gm__ bfloat16_t*)p_WqkvP;
        __gm__ bfloat16_t* gWo = (__gm__ bfloat16_t*)p_WoP;
        __gm__ bfloat16_t* gWgu = (__gm__ bfloat16_t*)p_WguP;
        __gm__ bfloat16_t* gWd = (__gm__ bfloat16_t*)p_WdP;
        __gm__ bfloat16_t* gWlm = (__gm__ bfloat16_t*)p_WlmP;
        __gm__ bfloat16_t* gxn1P = (__gm__ bfloat16_t*)p_xn1P;
        __gm__ bfloat16_t* gq2P = (__gm__ bfloat16_t*)p_q2P;
        __gm__ bfloat16_t* gKcP = (__gm__ bfloat16_t*)p_KcP;
        __gm__ bfloat16_t* gVcP = (__gm__ bfloat16_t*)p_VcP;
        __gm__ bfloat16_t* gprobs2P = (__gm__ bfloat16_t*)p_probs2P;
        __gm__ float* gqkv_out = (__gm__ float*)p_qkv_out;
        __gm__ float* gscores = (__gm__ float*)p_scores16;
        __gm__ float* gpartAccP = (__gm__ float*)p_partAccP;
        __gm__ float* go_out = (__gm__ float*)p_o_out;
        __gm__ float* ggu_out = (__gm__ float*)p_gu_out;
        __gm__ float* gd_out = (__gm__ float*)p_d_out;
        __gm__ float* glogits = (__gm__ float*)p_logits;
        __gm__ bfloat16_t* gattnP = (__gm__ bfloat16_t*)p_attnP;
        __gm__ bfloat16_t* gxn2P = (__gm__ bfloat16_t*)p_xn2P;
        __gm__ bfloat16_t* ghP = (__gm__ bfloat16_t*)p_hP;
        __gm__ bfloat16_t* gxnfP = (__gm__ bfloat16_t*)p_xnfP;
        volatile __gm__ int32_t* gseq = (volatile __gm__ int32_t*)p_syncws;

        GlobalTensor<bfloat16_t> txn1P, tq2P, tKcP, tVcP, tprobs2P, tattnP, txn2P, thP, txnfP, tWlm;
        tWlm.SetGlobalBuffer(gWlm);
        GlobalTensor<float> tqkvOut, tScores, tPartAcc, toOut, tguOut, tdOut, tLogits;
        txn1P.SetGlobalBuffer(gxn1P);  tq2P.SetGlobalBuffer(gq2P);
        tKcP.SetGlobalBuffer(gKcP);    tVcP.SetGlobalBuffer(gVcP);
        tprobs2P.SetGlobalBuffer(gprobs2P);
        tattnP.SetGlobalBuffer(gattnP); txn2P.SetGlobalBuffer(gxn2P);
        thP.SetGlobalBuffer(ghP);      txnfP.SetGlobalBuffer(gxnfP);
        tqkvOut.SetGlobalBuffer(gqkv_out); tScores.SetGlobalBuffer(gscores);
        tPartAcc.SetGlobalBuffer(gpartAccP); toOut.SetGlobalBuffer(go_out);
        tguOut.SetGlobalBuffer(ggu_out);    tdOut.SetGlobalBuffer(gd_out);
        tLogits.SetGlobalBuffer(glogits);

        volatile __gm__ float* gqkvF = (volatile __gm__ float*)gqkv_out;
        volatile __gm__ float* gscoresF = (volatile __gm__ float*)gscores;
        volatile __gm__ float* gpartAccF = (volatile __gm__ float*)gpartAccP;
        volatile __gm__ float* goF = (volatile __gm__ float*)go_out;
        volatile __gm__ float* gguF = (volatile __gm__ float*)ggu_out;
        volatile __gm__ float* gdF = (volatile __gm__ float*)gd_out;
        volatile __gm__ float* glogitsF = (volatile __gm__ float*)glogits;
        // sentinel column helpers (last-landed word of this AIC's region)
        const int qkvSC = ((cid + 1) * 176 < NQKV ? (cid + 1) * 176 : NQKV) - 1;
        const int odSC = ((cid + 1) * 96 < H ? (cid + 1) * 96 : H) - 1;
        const int guSC = cid * 512 + 511;
        const int lmLive = (VV - cid * SPAN < SPAN) ? VV - cid * SPAN : SPAN;
        const int lmSC = lmLive - 1;
        const bool odActive = (cid * 96 < H);   // AICs 22,23 skip o/d gemv
        // lm sentinel armed once per launch (logits region is host-prefilled)
#if MEGA_LM_STREAM
        for (int ch = 0; ch < NCH; ch++) {
            const int end = (ch + 1) * CH < lmLive ? (ch + 1) * CH : lmLive;
            aic_arm(glogitsF + cid * SPANP + end - 1);
        }
#else
        aic_arm(glogitsF + cid * SPANP + lmSC);
#endif
