extern "C" void run_megak(void* ffts, void* wqkv, void* wo, void* wgu, void* wd, void* wlm,
    void* rms1w, void* rms2w, void* rmsfw, void* qnw, void* knw, void* cosT, void* sinT,
    void* KcP, void* VcP, void* xEmb, void* xn1P, void* qkv_out, void* q2P, void* scores16,
    void* probs2P, void* partm, void* partl, void* partAccP, void* attnP, void* o_out,
    void* xn2P, void* gu_out, void* hP, void* d_out, void* logits, void* idxTab,
    void* partMax, void* partIdx, void* syncws, void* xnfP,
    int sl, int lstop, int tag, int grid, void* stream) {
    megak<<<grid, nullptr, stream>>>(static_cast<uint8_t*>(ffts), static_cast<uint8_t*>(wqkv),
        static_cast<uint8_t*>(wo), static_cast<uint8_t*>(wgu), static_cast<uint8_t*>(wd),
        static_cast<uint8_t*>(wlm), static_cast<uint8_t*>(rms1w), static_cast<uint8_t*>(rms2w),
        static_cast<uint8_t*>(rmsfw), static_cast<uint8_t*>(qnw), static_cast<uint8_t*>(knw),
        static_cast<uint8_t*>(cosT), static_cast<uint8_t*>(sinT), static_cast<uint8_t*>(KcP),
        static_cast<uint8_t*>(VcP), static_cast<uint8_t*>(xEmb), static_cast<uint8_t*>(xn1P),
        static_cast<uint8_t*>(qkv_out), static_cast<uint8_t*>(q2P), static_cast<uint8_t*>(scores16),
        static_cast<uint8_t*>(probs2P), static_cast<uint8_t*>(partm), static_cast<uint8_t*>(partl),
        static_cast<uint8_t*>(partAccP), static_cast<uint8_t*>(attnP), static_cast<uint8_t*>(o_out),
        static_cast<uint8_t*>(xn2P), static_cast<uint8_t*>(gu_out), static_cast<uint8_t*>(hP),
        static_cast<uint8_t*>(d_out), static_cast<uint8_t*>(logits), static_cast<uint8_t*>(idxTab),
        static_cast<uint8_t*>(partMax), static_cast<uint8_t*>(partIdx), static_cast<uint8_t*>(syncws),
        static_cast<uint8_t*>(xnfP), sl, lstop, tag);
}

// Host-only packing descriptor; loading a library must not require callers
// to guess the device's compile-time layouts. Legacy libraries lack this API.
extern "C" uint32_t megak_layout() {
    return static_cast<uint32_t>(NC_LM)
        | (static_cast<uint32_t>(MEGA_COMPACT_A) << 16)
        | (static_cast<uint32_t>(MEGA_GU_INTERLEAVE) << 17)
        | (static_cast<uint32_t>(MEGA_LM_STREAM) << 18);
}
