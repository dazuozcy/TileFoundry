#pragma once
// EDIT MODEL: exact RMS rounding, epsilon and rotate-half semantics.
// rmsnorm: t (n f32, destroyed), s (n f32 scratch), gb (n bf16 gamma),
// o (n bf16 out), t16 (16 f32 scratch). n multiple of 8, >= 16.
__aicore__ inline void rms_norm(LocalTensor<float> t, LocalTensor<float> s,
    LocalTensor<bfloat16_t> gb, LocalTensor<bfloat16_t> o, int n, LocalTensor<float> t16) {
    Mul(s, t, t, n);
    const float ms = tree_sum(s, n) / (float)n;
    Duplicate(t16, ms + EPS, 16);
    Rsqrt(t16, t16, 16);
    vector_scalar_fence();
    float r = t16.GetValue(0);
    // Two Newton steps recover FP32 rsqrt accuracy before BF16 rounding.
    r = r * (1.5f - 0.5f * (ms + EPS) * r * r);
    r = r * (1.5f - 0.5f * (ms + EPS) * r * r);
    scalar_vector_fence();
    Muls(t, t, r, n);
    Cast(o, t, RoundMode::CAST_RINT, n);      // land bf16 (HF numerics)
    Cast(t, o, RoundMode::CAST_NONE, n);      // back to f32
    Cast(s, gb, RoundMode::CAST_NONE, n);     // gamma
    Mul(t, t, s, n);
    Cast(o, t, RoundMode::CAST_RINT, n);
}

// rope: t (128 f32, in/out), cos/sinT (128 f32, sinT first half
// pre-negated), r (128 f32 scratch). pairs (i, i+64).
__aicore__ inline void rope_apply(LocalTensor<float> t, LocalTensor<float> cosr,
    LocalTensor<float> sinr, LocalTensor<float> r) {
    CopyRepeatParams cp; cp.dstStride = 1; cp.srcStride = 1; cp.dstRepeatSize = 8; cp.srcRepeatSize = 8;
    Copy(r, t[64], (uint64_t)64, (uint8_t)1, cp);
    Copy(r[64], t[0], (uint64_t)64, (uint8_t)1, cp);
    Mul(t, t, cosr, 128);
    Mul(r, r, sinr, 128);
    Add(t, t, r, 128);
}
