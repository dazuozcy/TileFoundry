// Current softmax uses a common carried shift; this is not a proof for arbitrary
// cross-layer score jumps in a new model. See PORTING.md before reusing it.
// ADAPT HARDWARE/PLACEMENT: buffers, aliases, channels and views.
        // ================= VECTOR SIDE =================
        const int bid = GetBlockIdx();
        const int sub = bid & 1;
        const int cid = bid >> 1;
        const int kv = cid % KVH;
        const int sid = cid / KVH;
        const int s0 = sid * thirds;
        const int e0 = (sl + 1 < s0 + thirds) ? (sl + 1) : (s0 + thirds);
        const int sblk = (e0 > s0) ? (e0 - s0) : 0;
        const int nloc = (sblk + SLT - 1) / SLT;
        // slice 1/2 live lengths (leaders poll those slices' AICs only if live)
        const int e1 = (sl + 1 < 2 * thirds) ? (sl + 1) : (2 * thirds);
        const int sblk1 = (e1 > thirds) ? (e1 - thirds) : 0;
        const int e2 = (sl + 1 < 3 * thirds) ? (sl + 1) : (3 * thirds);
        const int sblk2 = (e2 > 2 * thirds) ? (e2 - 2 * thirds) : 0;
        const int sidOwner = (sl / thirds < NSL) ? (sl / thirds) : (NSL - 1);
        const bool leader = (sid == 0);
        // R27: carried softmax shift (a rescale-free online softmax).
        // exp(x - sh) is exact for ANY sh -- it cancels in combine's num/den --
        // so the only thing sh must do is keep the exponent in range.  Layer 0
        // MEASURES its own slice max (one extra read, once per token); every
        // later layer reuses the previous layer's measured max, which turns the
        // safety condition from "max < 88" (a per-model constant) into "the max
        // cannot grow by >88 from one layer to the next" (model independent).
        // The in-pass max is still computed: it is what gets carried.
        float sh0 = NEG_INF, sh1 = NEG_INF;   // R27 shift: set per layer below

        TPipe pipe;
        TBuf<TPosition::VECCALC> tXm, tXf, tXn, tPack, tQkv, tHead, tHb, tSm, tP2, tQ2,
            tCmb, tAm, tSig, tSync, tFr, tSq, tW, tHbw, tPw;
        pipe.InitBuffer(tXm, H * sizeof(bfloat16_t));            // residual carrier
        pipe.InitBuffer(tXf, 2 * H * sizeof(float));             // f32 H (val + sq)
        pipe.InitBuffer(tXn, H * sizeof(bfloat16_t));            // bf16 H out
        pipe.InitBuffer(tPack, 4096 * sizeof(bfloat16_t));       // broadcast chunk
        pipe.InitBuffer(tQkv, NQKV_PAD * sizeof(float));         // qkv row 0
        pipe.InitBuffer(tHead, 16 * 128 * sizeof(float));        // head f32 scratch
        pipe.InitBuffer(tHb, 512 * sizeof(bfloat16_t));          // head bf16 scratch
        pipe.InitBuffer(tSm, 12 * 256 * sizeof(float));          // softmax/silu f32
        pipe.InitBuffer(tP2, 16 * 256 * sizeof(bfloat16_t));     // probs2P tile
        pipe.InitBuffer(tQ2, 8 * 256 * sizeof(bfloat16_t));      // q2P tile
        pipe.InitBuffer(tCmb, 4 * 128 * sizeof(float));          // combine
        pipe.InitBuffer(tAm, 4 * 2048 * sizeof(float));          // argmax
        pipe.InitBuffer(tSig, 8192);                             // sigmoid tmp
        pipe.InitBuffer(tSync, 8 * 48 * sizeof(int32_t));        // SyncAll ub ws
        pipe.InitBuffer(tFr, 8 * 256 * sizeof(bfloat16_t));      // V append RMW (batched)
        pipe.InitBuffer(tSq, 2 * 16 * sizeof(int32_t));          // seq word (x2)
        // wide softmax: CW positions at once (CW/SLT tiles), packed output
        pipe.InitBuffer(tW, 2 * 1024 * sizeof(float));           // w0,w1 f32 row chunks
        pipe.InitBuffer(tHbw, 2048 * sizeof(bfloat16_t));        // wide exp bf16 (2 heads)
        pipe.InitBuffer(tPw, (1024 / 256) * 16 * 256 * sizeof(bfloat16_t));  // packed prob tiles
        LocalTensor<bfloat16_t> xm = tXm.Get<bfloat16_t>();
        LocalTensor<float> xf = tXf.Get<float>();
        LocalTensor<float> xs = tXf.Get<float>()[H];
        LocalTensor<bfloat16_t> xn = tXn.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> pack = tPack.Get<bfloat16_t>();
        LocalTensor<float> qkv = tQkv.Get<float>();
        LocalTensor<float> hd = tHead.Get<float>();
        LocalTensor<bfloat16_t> hb = tHb.Get<bfloat16_t>();
        LocalTensor<float> sm = tSm.Get<float>();
        LocalTensor<bfloat16_t> p2 = tP2.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> q2 = tQ2.Get<bfloat16_t>();
        LocalTensor<float> cmb = tCmb.Get<float>();
        LocalTensor<float> am = tAm.Get<float>();
        LocalTensor<uint8_t> sig = tSig.Get<uint8_t>();
        LocalTensor<int32_t> syncub = tSync.Get<int32_t>();
        LocalTensor<bfloat16_t> vfr = tFr.Get<bfloat16_t>();
        LocalTensor<int32_t> sq = tSq.Get<int32_t>();
        LocalTensor<float> ws = tW.Get<float>();
        LocalTensor<float> w0 = ws[0], w1 = ws[1024];
        LocalTensor<bfloat16_t> hbw = tHbw.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> pw = tPw.Get<bfloat16_t>();
        LocalTensor<float> pwf = tPw.Get<float>();   // warm-gate scratch (unused by stage 4)
        // R24: diagnostic-max scratch (2x copy of a 1024-wide scaled row chunk)
        LocalTensor<float> mx0 = pwf[0], mx1 = pwf[1024];

        // named scratch views
        LocalTensor<float> q0f = hd[0], q1f = hd[128], kf = hd[256];
        LocalTensor<float> cosr = hd[640], sinr = hd[768], rot = hd[896];
        LocalTensor<float> t16 = hd[1408];
        LocalTensor<float> pml = hd[1664];                       // part_m/l loads
        LocalTensor<bfloat16_t> q0b = hb[0], q1b = hb[128], kb = hb[256], vb = hb[384];
        LocalTensor<float> sc0 = sm[0], sc1 = sm[256], ex0 = sm[512], ex1 = sm[768];
        // ping-pong reload buffers: the compiler/HW may hoist MTE2 issues past
        // PipeBarrier, so reload loops MUST alternate buffers (verified hard
        // way: silu chunks read one iteration ahead otherwise)
        LocalTensor<float> sc0b = sm[1536], sc1b = sm[1792];
        LocalTensor<float> ggu = sm[1024], uup = sm[1280];
        LocalTensor<float> ggub = sm[2048], uupb = sm[2304];
        LocalTensor<float> num = cmb[0], acc = cmb[128], t16c = cmb[256];
        LocalTensor<float> accb = cmb[384];   // ping-pong partner for acc
        LocalTensor<float> lgc = am[0], idxc = am[2048];
        LocalTensor<float> lgcb = am[4096], idxcb = am[6144];   // argmax ping-pong
        LocalTensor<bfloat16_t> packw = tAm.Get<bfloat16_t>();   // wide pack scratch (32KB, idle)

        event_t eM2V = static_cast<event_t>(pipe.FetchEventID<HardEvent::MTE2_V>());
        event_t eV2 = static_cast<event_t>(pipe.FetchEventID<HardEvent::V_MTE2>());
        event_t eV3 = static_cast<event_t>(pipe.FetchEventID<HardEvent::V_MTE3>());
        event_t eM3V = static_cast<event_t>(pipe.FetchEventID<HardEvent::MTE3_V>());
        event_t eM3M2 = static_cast<event_t>(pipe.FetchEventID<HardEvent::MTE3_MTE2>());
        event_t eM2S = static_cast<event_t>(pipe.FetchEventID<HardEvent::MTE2_S>());
        int sqWhich = 0;          // alternating seq-word buffer
        int s3cnt = 0;            // S_MTE3 id cycler (one-shot ids!)

        __gm__ bfloat16_t* gRms1w = (__gm__ bfloat16_t*)p_rms1w;
        __gm__ bfloat16_t* gRms2w = (__gm__ bfloat16_t*)p_rms2w;
        __gm__ bfloat16_t* gRmsfw = (__gm__ bfloat16_t*)p_rmsfw;
        __gm__ bfloat16_t* gQnw = (__gm__ bfloat16_t*)p_qnw;
        __gm__ bfloat16_t* gKnw = (__gm__ bfloat16_t*)p_knw;
        __gm__ float* gCosT = (__gm__ float*)p_cosT;
        __gm__ float* gSinT = (__gm__ float*)p_sinT;
        __gm__ bfloat16_t* gKcP = (__gm__ bfloat16_t*)p_KcP;
        __gm__ bfloat16_t* gVcP = (__gm__ bfloat16_t*)p_VcP;
        __gm__ bfloat16_t* gxn1P = (__gm__ bfloat16_t*)p_xn1P;
        __gm__ float* gqkv_out = (__gm__ float*)p_qkv_out;
        __gm__ bfloat16_t* gq2P = (__gm__ bfloat16_t*)p_q2P;
        __gm__ float* gscores = (__gm__ float*)p_scores16;
        __gm__ bfloat16_t* gprobs2P = (__gm__ bfloat16_t*)p_probs2P;
        __gm__ float* gpartm = (__gm__ float*)p_partm;
        __gm__ float* gpartl = (__gm__ float*)p_partl;
        __gm__ float* gpartAccP = (__gm__ float*)p_partAccP;
        __gm__ bfloat16_t* gattnP = (__gm__ bfloat16_t*)p_attnP;
        __gm__ float* go_out = (__gm__ float*)p_o_out;
        __gm__ bfloat16_t* gxn2P = (__gm__ bfloat16_t*)p_xn2P;
        __gm__ float* ggu_out = (__gm__ float*)p_gu_out;
        __gm__ bfloat16_t* ghP = (__gm__ bfloat16_t*)p_hP;
        __gm__ float* gd_out = (__gm__ float*)p_d_out;
        __gm__ float* glogits = (__gm__ float*)p_logits;
        __gm__ float* gIdxTab = (__gm__ float*)p_idxTab;
        __gm__ float* gPartMax = (__gm__ float*)p_partMax;
        __gm__ float* gPartIdx = (__gm__ float*)p_partIdx;
        GlobalTensor<bfloat16_t> tKcP, tVcP, txn1P, tq2P, tprobs2P, tattnP, txn2P, thP, txnfP;
        GlobalTensor<float> tQkvOut, tScores, tPartM, tPartL, tPartAcc, toOut, tguOut, tdOut,
            tLogits, tIdxTab, tPartMax, tPartIdx, tCosT, tSinT;
        GlobalTensor<bfloat16_t> tEmb;
        GlobalTensor<int32_t> gsync, gseq;
        GlobalTensor<bfloat16_t> tWguA;              // v8: gu weights (L2 warmer)
        tWguA.SetGlobalBuffer((__gm__ bfloat16_t*)p_WguP);
        tKcP.SetGlobalBuffer(gKcP);   tVcP.SetGlobalBuffer(gVcP);
        txn1P.SetGlobalBuffer(gxn1P); tq2P.SetGlobalBuffer(gq2P);
        tprobs2P.SetGlobalBuffer(gprobs2P); tattnP.SetGlobalBuffer(gattnP);
        txn2P.SetGlobalBuffer(gxn2P); thP.SetGlobalBuffer(ghP);
        txnfP.SetGlobalBuffer((__gm__ bfloat16_t*)p_xnfP);
        tEmb.SetGlobalBuffer((__gm__ bfloat16_t*)p_xEmb);
        tQkvOut.SetGlobalBuffer(gqkv_out); tScores.SetGlobalBuffer(gscores);
        tPartM.SetGlobalBuffer(gpartm);    tPartL.SetGlobalBuffer(gpartl);
        tPartAcc.SetGlobalBuffer(gpartAccP);
        GlobalTensor<float> tShiftHistory;
        tShiftHistory.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(p_syncws) + SHIFT_HISTORY);

        toOut.SetGlobalBuffer(go_out);     tguOut.SetGlobalBuffer(ggu_out);
        tdOut.SetGlobalBuffer(gd_out);     tLogits.SetGlobalBuffer(glogits);
        tIdxTab.SetGlobalBuffer(gIdxTab);
        tPartMax.SetGlobalBuffer(gPartMax); tPartIdx.SetGlobalBuffer(gPartIdx);
        tCosT.SetGlobalBuffer(gCosT);      tSinT.SetGlobalBuffer(gSinT);
        gsync.SetGlobalBuffer((__gm__ int32_t*)p_syncws);
        gseq.SetGlobalBuffer((__gm__ int32_t*)p_syncws);
        volatile __gm__ int32_t* gseqp = (volatile __gm__ int32_t*)p_syncws;  // poll view

        DataCopy(xm, tEmb, H);
        SetFlag<HardEvent::MTE2_V>(eM2V); WaitFlag<HardEvent::MTE2_V>(eM2V);
