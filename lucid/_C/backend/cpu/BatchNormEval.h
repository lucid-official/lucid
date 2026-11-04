// lucid/_C/backend/cpu/BatchNormEval.h
//
// Inference-mode BatchNorm kernels for the CPU backend: the per-channel
// running statistics are given, so the forward is one affine pass and the
// backward two per-channel sums.  Written once over the scalar type for
// float32 and float64.  The 16-bit floats have no Accelerate kernels and
// reach these through float32 — :class:`CpuBackend` widens them at the door
// as it does for the training kernel (LCD-487).
//
// Layout: x, y, grad and dx are (batch, channels, spatial) row-major; every
// per-channel buffer is (channels,).

#pragma once

#include <cmath>
#include <cstddef>

namespace lucid::backend::cpu {

// ``rstd = 1 / sqrt(var + eps)`` and ``y = gamma * (x - mean) * rstd + beta``.
// rstd is written out because the backward has no var to rebuild it from.
template <typename T>
void batch_norm_eval_forward_typed(const T* x,
                                   const T* mean,
                                   const T* var,
                                   const T* gamma,
                                   const T* beta,
                                   T* y,
                                   T* rstd,
                                   int batch,
                                   int channels,
                                   int spatial,
                                   double eps) {
    for (int c = 0; c < channels; ++c)
        rstd[c] = T{1} / static_cast<T>(std::sqrt(static_cast<double>(var[c]) + eps));
    for (int b = 0; b < batch; ++b)
        for (int c = 0; c < channels; ++c) {
            const T m = mean[c], r = rstd[c], g = gamma[c], bb = beta[c];
            const std::size_t row = (static_cast<std::size_t>(b) * channels + c) * spatial;
            for (int s = 0; s < spatial; ++s)
                y[row + s] = g * (x[row + s] - m) * r + bb;
        }
}

// The gradients of the forward above with rstd held fixed:
//   dx = gamma * rstd * g,  dgamma = rstd * sum((x - mean) * g),
//   dbeta = sum(g),  dmean = -gamma * rstd * sum(g),
//   dvar = -gamma * rstd^3 * sum((x - mean) * g) / 2.
template <typename T>
void batch_norm_eval_backward_typed(const T* x,
                                    const T* mean,
                                    const T* gamma,
                                    const T* rstd,
                                    const T* grad,
                                    T* dx,
                                    T* dmean,
                                    T* dvar,
                                    T* dgamma,
                                    T* dbeta,
                                    int batch,
                                    int channels,
                                    int spatial) {
    const bool need_xm = (dgamma != nullptr || dvar != nullptr);
    const bool need_g_sum = (dbeta != nullptr || dmean != nullptr);
    const bool need_gamma = (dx != nullptr || dmean != nullptr || dvar != nullptr);
    for (int c = 0; c < channels; ++c) {
        const T r = rstd[c];
        const T g = need_gamma ? gamma[c] : T{0};
        const T m = need_xm ? mean[c] : T{0};
        T sum_g = T{0}, sum_xm_g = T{0};
        for (int b = 0; b < batch; ++b) {
            const std::size_t row = (static_cast<std::size_t>(b) * channels + c) * spatial;
            for (int s = 0; s < spatial; ++s) {
                if (dx)
                    dx[row + s] = g * r * grad[row + s];
                if (need_g_sum)
                    sum_g += grad[row + s];
                if (need_xm)
                    sum_xm_g += (x[row + s] - m) * grad[row + s];
            }
        }
        if (dgamma)
            dgamma[c] = sum_xm_g * r;
        if (dbeta)
            dbeta[c] = sum_g;
        if (dmean)
            dmean[c] = -g * r * sum_g;
        if (dvar)
            dvar[c] = static_cast<T>(-0.5) * g * r * r * r * sum_xm_g;
    }
}

}  // namespace lucid::backend::cpu
