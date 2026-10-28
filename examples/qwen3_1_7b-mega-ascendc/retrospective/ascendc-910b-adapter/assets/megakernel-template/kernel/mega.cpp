// Compilable R30-derived template. Single launch; no dependency on parent project.
// Read PORTING.md before changing shapes or stage hand-offs.
#include <kernel_operator.h>
using namespace AscendC;
auto __enable_feature_for_compile_default = KERNEL_TYPE_MIX_AIC_1_2;
#include "model/model_config.hpp"
#include "backend/ascend910b/layout.hpp"
#include "backend/ascend910b/protocol.hpp"
#include "backend/ascend910b/cube.hpp"
#include "backend/ascend910b/vector_core.hpp"
#include "model/operators.hpp"
#include "backend/ascend910b/vector_io.hpp"
// ---------------- kernel ----------------
__global__ __aicore__ void megak(
    GM_ADDR p_ffts, GM_ADDR p_WqkvP, GM_ADDR p_WoP, GM_ADDR p_WguP, GM_ADDR p_WdP,
    GM_ADDR p_WlmP, GM_ADDR p_rms1w, GM_ADDR p_rms2w, GM_ADDR p_rmsfw, GM_ADDR p_qnw,
    GM_ADDR p_knw, GM_ADDR p_cosT, GM_ADDR p_sinT, GM_ADDR p_KcP, GM_ADDR p_VcP,
    GM_ADDR p_xEmb, GM_ADDR p_xn1P, GM_ADDR p_qkv_out, GM_ADDR p_q2P, GM_ADDR p_scores16,
    GM_ADDR p_probs2P, GM_ADDR p_partm, GM_ADDR p_partl, GM_ADDR p_partAccP, GM_ADDR p_attnP,
    GM_ADDR p_o_out, GM_ADDR p_xn2P, GM_ADDR p_gu_out, GM_ADDR p_hP, GM_ADDR p_d_out,
    GM_ADDR p_logits, GM_ADDR p_idxTab, GM_ADDR p_partMax, GM_ADDR p_partIdx,
    GM_ADDR p_syncws, GM_ADDR p_xnfP, int32_t sl, int32_t lstop, int32_t tag) {
    if ((uint64_t)p_ffts != 0) { SetSyncBaseAddr((uint64_t)p_ffts); }

    // dynamic slice split (SLT-aligned thirds)
    const int thirdsRaw = ((sl + 1) + NSL - 1) / NSL;
    const int thirds = ((thirdsRaw + SLT - 1) / SLT) * SLT;

    if ASCEND_IS_AIC {
#include "model/cube_schedule.inc"
    } else {
#include "model/vector_schedule.inc"
    }
}
#include "launch.inc"
