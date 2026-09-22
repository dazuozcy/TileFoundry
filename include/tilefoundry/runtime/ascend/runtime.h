/// tilefoundry Ascend-target device runtime surface.
///
/// AscendC op implementations included in-context from the generated device
/// unit (compiled by bisheng ``-xasc``). Host-side code never includes this
/// header: the CPU host unit crosses into the device through the raw
/// ``extern "C"`` launch shims only.
#pragma once
#include "kernel_operator.h"
#include "utils/std/cmath.h"

namespace tilefoundry {
namespace ops {
using AscendC::DataCopy;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::TPipe;
using AscendC::TPosition;
using AscendC::TQue;

/// RMSNorm over a row-major (M, K) tile in GM: y = x * rsqrt(mean(x^2)+eps) * w.
///
/// The vector path stages the tile through UB queues when every row copy is
/// 32-byte aligned; otherwise a scalar GM fallback (GetValue/SetValue) keeps
/// every shape correct, just slower.
template <class T>
__aicore__ inline void rmsnorm(const GlobalTensor<T>& src, GlobalTensor<T>& dst,
                               const GlobalTensor<T>& weight, int64_t M, int64_t K,
                               float eps) {
    const int64_t total = M * K;
    const bool aligned = (K * int64_t(sizeof(T))) % 32 == 0;
    if (aligned) {
        TPipe pipe;
        TQue<TPosition::VECIN, 2> que_x;
        TQue<TPosition::VECIN, 2> que_w;
        TQue<TPosition::VECOUT, 2> que_y;
        pipe.InitBuffer(que_x, 2, total * int64_t(sizeof(T)));
        pipe.InitBuffer(que_w, 2, K * int64_t(sizeof(T)));
        pipe.InitBuffer(que_y, 2, total * int64_t(sizeof(T)));
        LocalTensor<T> local_x = que_x.AllocTensor<T>();
        LocalTensor<T> local_w = que_w.AllocTensor<T>();
        LocalTensor<T> local_y = que_y.AllocTensor<T>();
        DataCopy(local_x, src, total);
        DataCopy(local_w, weight, K);
        que_x.EnQue(local_x);
        que_w.EnQue(local_w);
        LocalTensor<T> x = que_x.DeQue<T>();
        LocalTensor<T> w = que_w.DeQue<T>();
        for (int64_t m = 0; m < M; ++m) {
            float sumsq = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                float v = static_cast<float>(x.GetValue(m * K + k));
                sumsq += v * v;
            }
            float rms = 1.0f / AscendC::Std::sqrt(sumsq / static_cast<float>(K) + eps);
            for (int64_t k = 0; k < K; ++k) {
                float v = static_cast<float>(x.GetValue(m * K + k));
                local_y.SetValue(m * K + k,
                                 static_cast<T>(v * rms * static_cast<float>(w.GetValue(k))));
            }
        }
        que_y.EnQue(local_y);
        LocalTensor<T> y = que_y.DeQue<T>();
        DataCopy(dst, y, total);
        que_x.FreeTensor(x);
        que_w.FreeTensor(w);
        que_y.FreeTensor(y);
    } else {
        for (int64_t m = 0; m < M; ++m) {
            float sumsq = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                float v = static_cast<float>(src.GetValue(m * K + k));
                sumsq += v * v;
            }
            float rms = 1.0f / AscendC::Std::sqrt(sumsq / static_cast<float>(K) + eps);
            for (int64_t k = 0; k < K; ++k) {
                float v = static_cast<float>(src.GetValue(m * K + k));
                dst.SetValue(m * K + k,
                             static_cast<T>(v * rms * static_cast<float>(weight.GetValue(k))));
            }
        }
    }
}

}  // namespace ops
}  // namespace tilefoundry
