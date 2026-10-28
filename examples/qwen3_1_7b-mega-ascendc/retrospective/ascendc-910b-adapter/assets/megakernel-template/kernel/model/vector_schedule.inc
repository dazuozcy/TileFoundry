#include "backend/ascend910b/vector_resources.inc"
        for (int li = 0; li < lstop; li++) {
            const int R = li * 8;
            TRACE_V(li, 0);
#include "model/stages/vector/input_norm.inc"
#include "model/stages/vector/qk_norm_rope_kv.inc"
#include "model/stages/vector/softmax_combine.inc"
#include "model/stages/vector/residual_mlp_norm.inc"
#include "model/stages/vector/activation.inc"
#include "model/stages/vector/residual.inc"
        }

#include "model/stages/vector/final_norm_argmax.inc"
