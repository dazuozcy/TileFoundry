#pragma once
// ---------------- AIV helpers ----------------
// GetValue executes on S. A vector-pipe barrier alone is not an explicit
// V -> S dependency. Reserve V_S event 7 for synchronous scalar readback.
__aicore__ inline void vector_scalar_fence() {
    SetFlag<HardEvent::V_S>((event_t)7);
    WaitFlag<HardEvent::V_S>((event_t)7);
}

// Complete scalar reads before the same scratch is written by V again.
__aicore__ inline void scalar_vector_fence() {
    SetFlag<HardEvent::S_V>((event_t)7);
    WaitFlag<HardEvent::S_V>((event_t)7);
}

// tree reduce over n (power-of-2 multiple of 8, n>=16); result in t[0..8)
__aicore__ inline float tree_sum(LocalTensor<float> t, int n) {
    for (int len = n / 2; len >= 8; len /= 2) { Add(t, t, t[len], len); }
    vector_scalar_fence();
    float s = t.GetValue(0);
    for (int i = 1; i < 8; i++) { s += t.GetValue(i); }
    scalar_vector_fence();
    return s;
}
__aicore__ inline float tree_max(LocalTensor<float> t, int n) {
    for (int len = n / 2; len >= 8; len /= 2) { Max(t, t, t[len], len); }
    vector_scalar_fence();
    float m = t.GetValue(0);
    for (int i = 1; i < 8; i++) { float x = t.GetValue(i); if (x > m) m = x; }
    scalar_vector_fence();
    return m;
}
// scalar exp via 16-el vector buffer
__aicore__ inline float vexp_scalar(LocalTensor<float> t16, float x) {
    Duplicate(t16, x, 16);
    Exp(t16, t16, 16);
    vector_scalar_fence();
    float result = t16.GetValue(0);
    scalar_vector_fence();
    return result;
}
