// lucid/_C/backend/cpu/Norm.cpp
//
// Implements the LayerNorm and RMSNorm forward and backward passes declared in
// Norm.h, parallelised across rows via cpu::parallel_for and vectorised with
// Apple Silicon NEON SIMD accumulators.

#include "Norm.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include <arm_neon.h>

#include "../GradMask.h"
#include "Parallel.h"

namespace lucid::backend::cpu {

namespace {

inline constexpr std::size_t kNormElemGrain = 65536;
inline constexpr std::size_t kNormMaxChunks = 8;

inline std::size_t norm_grain(std::size_t outer, std::size_t N) noexcept {
    const std::size_t min_grain =
        std::max<std::size_t>(1, kNormElemGrain / std::max<std::size_t>(N, 1));
    const std::size_t cap_grain = std::max<std::size_t>(1, outer / kNormMaxChunks);
    return std::max(min_grain, cap_grain);
}

template <typename T>
struct NormSimd;

template <>
struct NormSimd<float> {
    using Vec = float32x4_t;
    static constexpr std::size_t kWidth = 4;
    static Vec zero() noexcept { return vdupq_n_f32(0.0f); }
    static Vec dup(float v) noexcept { return vdupq_n_f32(v); }
    static Vec load(const float* p) noexcept { return vld1q_f32(p); }
    static void store(float* p, Vec v) noexcept { vst1q_f32(p, v); }
    static Vec add(Vec a, Vec b) noexcept { return vaddq_f32(a, b); }
    static Vec sub(Vec a, Vec b) noexcept { return vsubq_f32(a, b); }
    static Vec mul(Vec a, Vec b) noexcept { return vmulq_f32(a, b); }
    static Vec fma(Vec acc, Vec a, Vec b) noexcept { return vfmaq_f32(acc, a, b); }
    static float hsum(Vec v) noexcept { return vaddvq_f32(v); }
};

template <>
struct NormSimd<double> {
    using Vec = float64x2_t;
    static constexpr std::size_t kWidth = 2;
    static Vec zero() noexcept { return vdupq_n_f64(0.0); }
    static Vec dup(double v) noexcept { return vdupq_n_f64(v); }
    static Vec load(const double* p) noexcept { return vld1q_f64(p); }
    static void store(double* p, Vec v) noexcept { vst1q_f64(p, v); }
    static Vec add(Vec a, Vec b) noexcept { return vaddq_f64(a, b); }
    static Vec sub(Vec a, Vec b) noexcept { return vsubq_f64(a, b); }
    static Vec mul(Vec a, Vec b) noexcept { return vmulq_f64(a, b); }
    static Vec fma(Vec acc, Vec a, Vec b) noexcept { return vfmaq_f64(acc, a, b); }
    static double hsum(Vec v) noexcept { return vaddvq_f64(v); }
};

template <typename T>
inline T row_sum(const T* __restrict xb, std::size_t N) noexcept {
    using S = NormSimd<T>;
    constexpr std::size_t W = S::kWidth;
    auto acc0 = S::zero();
    auto acc1 = S::zero();
    std::size_t i = 0;
    for (; i + 2 * W <= N; i += 2 * W) {
        acc0 = S::add(acc0, S::load(xb + i));
        acc1 = S::add(acc1, S::load(xb + i + W));
    }
    T sum = S::hsum(S::add(acc0, acc1));
    for (; i < N; ++i)
        sum += xb[i];
    return sum;
}

template <typename T>
inline T row_sqdiff(const T* __restrict xb, T mean, std::size_t N) noexcept {
    using S = NormSimd<T>;
    constexpr std::size_t W = S::kWidth;
    const auto vm = S::dup(mean);
    auto acc0 = S::zero();
    auto acc1 = S::zero();
    std::size_t i = 0;
    for (; i + 2 * W <= N; i += 2 * W) {
        const auto d0 = S::sub(S::load(xb + i), vm);
        const auto d1 = S::sub(S::load(xb + i + W), vm);
        acc0 = S::fma(acc0, d0, d0);
        acc1 = S::fma(acc1, d1, d1);
    }
    T var_sum = S::hsum(S::add(acc0, acc1));
    for (; i < N; ++i) {
        const T d = xb[i] - mean;
        var_sum += d * d;
    }
    return var_sum;
}

template <typename T>
void layer_norm_forward(const T* x,
                        const T* gamma,
                        const T* beta,
                        T* y,
                        T* saved_mean,
                        T* saved_rstd,
                        std::size_t outer,
                        std::size_t N,
                        double eps) {
    if (outer == 0 || N == 0)
        return;
    const T inv_N = T{1} / static_cast<T>(N);
    const T eps_t = static_cast<T>(eps);
    parallel_for(outer, norm_grain(outer, N), [&](std::size_t lo, std::size_t hi) {
        const T* __restrict gm = gamma;
        const T* __restrict bt = beta;
        for (std::size_t o = lo; o < hi; ++o) {
            const T* __restrict xb = x + o * N;
            T* __restrict yb = y + o * N;
            const T mean = row_sum(xb, N) * inv_N;
            const T var = row_sqdiff(xb, mean, N) * inv_N;
            const T rstd = T{1} / std::sqrt(var + eps_t);
            for (std::size_t i = 0; i < N; ++i)
                yb[i] = gm[i] * ((xb[i] - mean) * rstd) + bt[i];
            saved_mean[o] = mean;
            saved_rstd[o] = rstd;
        }
    });
}

// The row only owes the bias gradient: no x, no gamma, no dx are read.
template <bool NeedDx, bool NeedDg, bool NeedDb>
inline constexpr bool kBiasGradOnly = !NeedDx && !NeedDg && NeedDb;

template <typename T, bool NeedDx, bool NeedDg, bool NeedDb>
inline void layer_norm_bwd_row(const T* __restrict xb,
                               const T* __restrict gb,
                               const T* __restrict gm,
                               T* __restrict dxb,
                               T* __restrict dg,
                               T* __restrict db,
                               T m,
                               T r,
                               T inv_N,
                               std::size_t N) noexcept {
    using S = NormSimd<T>;
    constexpr std::size_t W = S::kWidth;
    if constexpr (kBiasGradOnly<NeedDx, NeedDg, NeedDb>) {
        std::size_t i = 0;
        for (; i + W <= N; i += W)
            S::store(db + i, S::add(S::load(db + i), S::load(gb + i)));
        for (; i < N; ++i)
            db[i] += gb[i];
        return;
    }
    const auto vm = S::dup(m);
    const auto vr = S::dup(r);
    auto v_sdxn = S::zero();
    auto v_sdxn_xn = S::zero();
    std::size_t i = 0;
    for (; i + W <= N; i += W) {
        const auto vx = S::load(xb + i);
        const auto vg = S::load(gb + i);
        const auto vxn = S::mul(S::sub(vx, vm), vr);
        if constexpr (NeedDx) {
            const auto vgm = S::load(gm + i);
            const auto vdxn = S::mul(vgm, vg);
            v_sdxn = S::add(v_sdxn, vdxn);
            v_sdxn_xn = S::fma(v_sdxn_xn, vdxn, vxn);
        }
        if constexpr (NeedDg)
            S::store(dg + i, S::fma(S::load(dg + i), vg, vxn));
        if constexpr (NeedDb)
            S::store(db + i, S::add(S::load(db + i), vg));
    }
    // Both accumulators are still zero when dx is not wanted.
    T sum_dxn = S::hsum(v_sdxn);
    T sum_dxn_xn = S::hsum(v_sdxn_xn);
    for (; i < N; ++i) {
        const T xn_i = (xb[i] - m) * r;
        const T g_i = gb[i];
        if constexpr (NeedDx) {
            const T dxn_i = gm[i] * g_i;
            sum_dxn += dxn_i;
            sum_dxn_xn += dxn_i * xn_i;
        }
        if constexpr (NeedDg)
            dg[i] += g_i * xn_i;
        if constexpr (NeedDb)
            db[i] += g_i;
    }
    if constexpr (NeedDx) {
        const T c1 = sum_dxn * inv_N;
        const T c2 = sum_dxn_xn * inv_N;
        for (std::size_t k = 0; k < N; ++k) {
            const T xn_k = (xb[k] - m) * r;
            const T dxn_k = gm[k] * gb[k];
            dxb[k] = r * (dxn_k - c1 - xn_k * c2);
        }
    }
}

template <typename T, typename RowFn>
void reduce_norm_bwd_rows(
    T* param_grads, std::size_t param_stride, std::size_t outer, std::size_t N, RowFn&& row_fn) {
    std::memset(param_grads, 0, param_stride * sizeof(T));
    if (outer == 0 || N == 0)
        return;
    const std::size_t grain = norm_grain(outer, N);
    const std::size_t chunks = parallel_chunk_count(outer, grain);
    if (chunks <= 1) {
        for (std::size_t o = 0; o < outer; ++o)
            row_fn(o, param_grads);
        return;
    }
    std::vector<T> partials(chunks * param_stride, T{});
    parallel_for(outer, grain, [&](std::size_t c, std::size_t lo, std::size_t hi) {
        T* __restrict chunk_grads = partials.data() + c * param_stride;
        for (std::size_t o = lo; o < hi; ++o)
            row_fn(o, chunk_grads);
    });
    for (std::size_t c = 0; c < chunks; ++c) {
        const T* __restrict src = partials.data() + c * param_stride;
        for (std::size_t i = 0; i < param_stride; ++i)
            param_grads[i] += src[i];
    }
}

template <typename T>
void layer_norm_backward(const T* x,
                         const T* gamma,
                         const T* saved_mean,
                         const T* saved_rstd,
                         const T* g,
                         T* dx,
                         T* dgamma,
                         T* dbeta,
                         std::size_t outer,
                         std::size_t N) {
    if (N == 0 || none_set(dx, dgamma, dbeta))
        return;
    const T inv_N = T{1} / static_cast<T>(N);
    if (dx && none_set(dgamma, dbeta)) {
        if (outer == 0)
            return;
        parallel_for(outer, norm_grain(outer, N), [&](std::size_t lo, std::size_t hi) {
            for (std::size_t o = lo; o < hi; ++o) {
                layer_norm_bwd_row<T, true, false, false>(x + o * N, g + o * N, gamma, dx + o * N,
                                                          nullptr, nullptr, saved_mean[o],
                                                          saved_rstd[o], inv_N, N);
            }
        });
        return;
    }
    if (dgamma && !dbeta) {
        if (dx) {
            reduce_norm_bwd_rows(dgamma, N, outer, N, [&](std::size_t o, T* buf) {
                layer_norm_bwd_row<T, true, true, false>(x + o * N, g + o * N, gamma, dx + o * N,
                                                         buf, nullptr, saved_mean[o], saved_rstd[o],
                                                         inv_N, N);
            });
        } else {
            reduce_norm_bwd_rows(dgamma, N, outer, N, [&](std::size_t o, T* buf) {
                layer_norm_bwd_row<T, false, true, false>(x + o * N, g + o * N, nullptr, nullptr,
                                                          buf, nullptr, saved_mean[o],
                                                          saved_rstd[o], inv_N, N);
            });
        }
        return;
    }
    if (!dgamma && dbeta) {
        if (dx) {
            reduce_norm_bwd_rows(dbeta, N, outer, N, [&](std::size_t o, T* buf) {
                layer_norm_bwd_row<T, true, false, true>(x + o * N, g + o * N, gamma, dx + o * N,
                                                         nullptr, buf, saved_mean[o], saved_rstd[o],
                                                         inv_N, N);
            });
        } else {
            reduce_norm_bwd_rows(dbeta, N, outer, N, [&](std::size_t o, T* buf) {
                layer_norm_bwd_row<T, false, false, true>(nullptr, g + o * N, nullptr, nullptr,
                                                          nullptr, buf, T{0}, T{0}, inv_N, N);
            });
        }
        return;
    }
    std::vector<T> combined(2 * N, T{});
    if (dx) {
        reduce_norm_bwd_rows(combined.data(), 2 * N, outer, N, [&](std::size_t o, T* buf) {
            layer_norm_bwd_row<T, true, true, true>(x + o * N, g + o * N, gamma, dx + o * N, buf,
                                                    buf + N, saved_mean[o], saved_rstd[o], inv_N,
                                                    N);
        });
    } else {
        reduce_norm_bwd_rows(combined.data(), 2 * N, outer, N, [&](std::size_t o, T* buf) {
            layer_norm_bwd_row<T, false, true, true>(x + o * N, g + o * N, nullptr, nullptr, buf,
                                                     buf + N, saved_mean[o], saved_rstd[o], inv_N,
                                                     N);
        });
    }
    std::memcpy(dgamma, combined.data(), N * sizeof(T));
    std::memcpy(dbeta, combined.data() + N, N * sizeof(T));
}

template <typename T>
void rms_norm_forward(
    const T* x, const T* gamma, T* y, T* saved_rstd, std::size_t outer, std::size_t N, double eps) {
    if (outer == 0 || N == 0)
        return;
    const T inv_N = T{1} / static_cast<T>(N);
    const T eps_t = static_cast<T>(eps);
    parallel_for(outer, norm_grain(outer, N), [&](std::size_t lo, std::size_t hi) {
        const T* __restrict gm = gamma;
        for (std::size_t o = lo; o < hi; ++o) {
            const T* __restrict xb = x + o * N;
            T* __restrict yb = y + o * N;
            const T meansq = row_sqdiff(xb, T{0}, N) * inv_N;
            const T rstd = T{1} / std::sqrt(meansq + eps_t);
            for (std::size_t i = 0; i < N; ++i)
                yb[i] = gm[i] * xb[i] * rstd;
            saved_rstd[o] = rstd;
        }
    });
}

template <typename T, bool NeedDx, bool NeedDg>
inline void rms_norm_bwd_row(const T* __restrict xb,
                             const T* __restrict gb,
                             const T* __restrict gm,
                             T* __restrict dxb,
                             T* __restrict dg,
                             T r,
                             T inv_N,
                             std::size_t N) noexcept {
    using S = NormSimd<T>;
    constexpr std::size_t W = S::kWidth;
    const auto vr = S::dup(r);
    auto vcross = S::zero();
    std::size_t i = 0;
    for (; i + W <= N; i += W) {
        const auto vx = S::load(xb + i);
        const auto vg = S::load(gb + i);
        if constexpr (NeedDx) {
            const auto vgm = S::load(gm + i);
            vcross = S::fma(vcross, S::mul(vgm, vg), vx);
        }
        if constexpr (NeedDg)
            S::store(dg + i, S::fma(S::load(dg + i), S::mul(vg, vx), vr));
    }
    T cross = NeedDx ? S::hsum(vcross) : T{0};
    for (; i < N; ++i) {
        const T g_i = gb[i];
        const T x_i = xb[i];
        if constexpr (NeedDx)
            cross += (gm[i] * g_i) * x_i;
        if constexpr (NeedDg)
            dg[i] += g_i * x_i * r;
    }
    if constexpr (NeedDx) {
        const T scaled = inv_N * r * r * cross;
        for (std::size_t k = 0; k < N; ++k)
            dxb[k] = r * (gm[k] * gb[k] - xb[k] * scaled);
    }
}

template <typename T>
void rms_norm_backward(const T* x,
                       const T* gamma,
                       const T* saved_rstd,
                       const T* g,
                       T* dx,
                       T* dgamma,
                       std::size_t outer,
                       std::size_t N) {
    if (N == 0 || (!dx && !dgamma))
        return;
    const T inv_N = T{1} / static_cast<T>(N);
    if (dx && !dgamma) {
        if (outer == 0)
            return;
        parallel_for(outer, norm_grain(outer, N), [&](std::size_t lo, std::size_t hi) {
            for (std::size_t o = lo; o < hi; ++o) {
                rms_norm_bwd_row<T, true, false>(x + o * N, g + o * N, gamma, dx + o * N, nullptr,
                                                 saved_rstd[o], inv_N, N);
            }
        });
        return;
    }
    if (!dx && dgamma) {
        reduce_norm_bwd_rows(dgamma, N, outer, N, [&](std::size_t o, T* buf) {
            rms_norm_bwd_row<T, false, true>(x + o * N, g + o * N, nullptr, nullptr, buf,
                                             saved_rstd[o], inv_N, N);
        });
        return;
    }
    reduce_norm_bwd_rows(dgamma, N, outer, N, [&](std::size_t o, T* buf) {
        rms_norm_bwd_row<T, true, true>(x + o * N, g + o * N, gamma, dx + o * N, buf, saved_rstd[o],
                                        inv_N, N);
    });
}

}  // namespace

void layer_norm_forward_f32(const float* x,
                            const float* gamma,
                            const float* beta,
                            float* y,
                            float* saved_mean,
                            float* saved_rstd,
                            std::size_t outer,
                            std::size_t N,
                            double eps) {
    layer_norm_forward<float>(x, gamma, beta, y, saved_mean, saved_rstd, outer, N, eps);
}

void layer_norm_forward_f64(const double* x,
                            const double* gamma,
                            const double* beta,
                            double* y,
                            double* saved_mean,
                            double* saved_rstd,
                            std::size_t outer,
                            std::size_t N,
                            double eps) {
    layer_norm_forward<double>(x, gamma, beta, y, saved_mean, saved_rstd, outer, N, eps);
}

void layer_norm_backward_f32(const float* x,
                             const float* gamma,
                             const float* saved_mean,
                             const float* saved_rstd,
                             const float* g,
                             float* dx,
                             float* dgamma,
                             float* dbeta,
                             std::size_t outer,
                             std::size_t N) {
    layer_norm_backward<float>(x, gamma, saved_mean, saved_rstd, g, dx, dgamma, dbeta, outer, N);
}

void layer_norm_backward_f64(const double* x,
                             const double* gamma,
                             const double* saved_mean,
                             const double* saved_rstd,
                             const double* g,
                             double* dx,
                             double* dgamma,
                             double* dbeta,
                             std::size_t outer,
                             std::size_t N) {
    layer_norm_backward<double>(x, gamma, saved_mean, saved_rstd, g, dx, dgamma, dbeta, outer, N);
}

void rms_norm_forward_f32(const float* x,
                          const float* gamma,
                          float* y,
                          float* saved_rstd,
                          std::size_t outer,
                          std::size_t N,
                          double eps) {
    rms_norm_forward<float>(x, gamma, y, saved_rstd, outer, N, eps);
}

void rms_norm_forward_f64(const double* x,
                          const double* gamma,
                          double* y,
                          double* saved_rstd,
                          std::size_t outer,
                          std::size_t N,
                          double eps) {
    rms_norm_forward<double>(x, gamma, y, saved_rstd, outer, N, eps);
}

void rms_norm_backward_f32(const float* x,
                           const float* gamma,
                           const float* saved_rstd,
                           const float* g,
                           float* dx,
                           float* dgamma,
                           std::size_t outer,
                           std::size_t N) {
    rms_norm_backward<float>(x, gamma, saved_rstd, g, dx, dgamma, outer, N);
}

void rms_norm_backward_f64(const double* x,
                           const double* gamma,
                           const double* saved_rstd,
                           const double* g,
                           double* dx,
                           double* dgamma,
                           std::size_t outer,
                           std::size_t N) {
    rms_norm_backward<double>(x, gamma, saved_rstd, g, dx, dgamma, outer, N);
}

}  // namespace lucid::backend::cpu
