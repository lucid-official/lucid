// lucid/_C/backend/cpu/Norm.h
//
// CPU implementations of LayerNorm and RMSNorm forward and backward passes.
// Both norms iterate over `outer` independent rows of length N (the normalized
// size) and compute per-row statistics.
//
// LayerNorm: y = gamma * (x - mean) / sqrt(var + eps) + beta.
//   The f32 forward path uses vDSP (vmean_f32, vdotpr_f32, etc.) for SIMD
//   throughput; the f64 forward path uses a generic template loop.
//   The backward pass computes dx, dgamma, and dbeta using the standard
//   LayerNorm gradient formula: dx_i = r * (N * dxn_i - sum_dxn - xn_i * sum_dxn_xn) / N.
//
// RMSNorm: y = gamma * x / sqrt(mean(x^2) + eps).  No mean subtraction; only
//   the root-mean-square scale is normalised.  The backward pass returns dx
//   and dgamma.
//
// Both norms save per-row reciprocal standard deviation (rstd = 1/std) during
// the forward pass so the backward pass can avoid a redundant division.
//
// Lp normalisation (``lp_normalize``): ``y = x / max(||x||_p, eps)`` along one
// axis, written once over the scalar type at the end of this header.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "../../api.h"
#include "../../core/Shape.h"
#include "Parallel.h"
#include "Vforce.h"

namespace lucid::backend::cpu {

// Single-precision LayerNorm forward — normalises every row of length ``N``
// using its empirical mean and variance, then applies a per-feature affine
// transform $\gamma \hat{x} + \beta$ in the same pass.
//
// The f32 path is a vDSP fast path: it computes the row mean with
// ``vmean_f32``, centres the row into a scratch buffer with ``vsadd_f32``,
// derives the variance as a self-dot-product via ``vdotpr_f32``, then fuses
// the affine into a ``vmadd_f32`` call so the data is touched twice rather
// than three times.  Per-row ``mean`` and ``rstd = 1/sqrt(var + eps)`` are
// stashed so the backward kernel can skip the recomputation.
//
// Parameters
// ----------
// x : const float*
//     Input buffer of shape ``(outer, N)`` in row-major order.
// gamma, beta : const float*
//     Per-feature affine parameters of shape ``(N,)``.  Both required.
// y : float*
//     Output buffer of shape ``(outer, N)``; written densely.
// saved_mean, saved_rstd : float*
//     Per-row scratch outputs of shape ``(outer,)``.  Consumed by
//     :cpp:func:`layer_norm_backward_f32`.
// outer : std::size_t
//     Number of independent rows (product of all leading dimensions).
// N : std::size_t
//     Length of the normalised axis.
// eps : double
//     Variance stabiliser; cast to ``float`` before being added to ``var``.
//
// Math
// ----
// $$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \varepsilon}}, \qquad
//   y_i = \gamma_i \hat{x}_i + \beta_i,$$
// with $\mu, \sigma^2$ the row mean and variance of ``x``.
//
// References
// ----------
// Ba, Kiros & Hinton, "Layer Normalization" (arXiv:1607.06450, 2016).
LUCID_INTERNAL void layer_norm_forward_f32(const float* x,
                                           const float* gamma,
                                           const float* beta,
                                           float* y,
                                           float* saved_mean,
                                           float* saved_rstd,
                                           std::size_t outer,
                                           std::size_t N,
                                           double eps);

// Double-precision LayerNorm forward — generic template implementation that
// computes mean and variance with a two-pass loop (no vDSP equivalent exists
// for ``double`` at the same fusion level).
//
// Parameters
// ----------
// x : const double*
//     Input buffer of shape ``(outer, N)``.
// gamma, beta : const double*
//     Per-feature affine parameters of shape ``(N,)``.
// y : double*
//     Output buffer of shape ``(outer, N)``.
// saved_mean, saved_rstd : double*
//     Per-row mean and reciprocal standard deviation, both shape ``(outer,)``.
// outer, N : std::size_t
//     Number of rows and length of the normalised axis.
// eps : double
//     Variance stabiliser.
//
// Math
// ----
// $$y_i = \gamma_i \frac{x_i - \mu}{\sqrt{\sigma^2 + \varepsilon}} + \beta_i.$$
LUCID_INTERNAL void layer_norm_forward_f64(const double* x,
                                           const double* gamma,
                                           const double* beta,
                                           double* y,
                                           double* saved_mean,
                                           double* saved_rstd,
                                           std::size_t outer,
                                           std::size_t N,
                                           double eps);

// Single-precision LayerNorm backward — produces input, weight, and bias
// gradients from a saved-tensor forward pass.
//
// Uses the closed-form LayerNorm gradient that avoids recomputing the row
// statistics.  ``dgamma`` and ``dbeta`` are *accumulated* into the supplied
// buffers; callers reducing over multiple sub-batches must zero them once at
// the start.  ``dx`` is written fully (not accumulated).
//
// Parameters
// ----------
// x : const float*
//     Original forward input of shape ``(outer, N)``.
// gamma : const float*
//     Affine weight of shape ``(N,)``.
// saved_mean, saved_rstd : const float*
//     Per-row statistics produced by :cpp:func:`layer_norm_forward_f32`.
// g : const float*
//     Upstream gradient $\partial L / \partial y$ of shape ``(outer, N)``.
// dx : float*
//     Output gradient w.r.t. ``x``; shape ``(outer, N)``; *overwritten*.
// dgamma, dbeta : float*
//     Output gradients w.r.t. the affine parameters; shape ``(N,)``;
//     *accumulated* (caller must zero on first call).
// outer, N : std::size_t
//     Layout extents.
//
// Math
// ----
// Let $\hat{x}_i = (x_i - \mu)\,\text{rstd}$ and
// $\widetilde{g}_i = \gamma_i g_i$.  Then
// $$\frac{\partial L}{\partial x_i} =
//   \frac{\text{rstd}}{N} \!\left(N \widetilde{g}_i - \sum_j \widetilde{g}_j -
//     \hat{x}_i \sum_j \widetilde{g}_j \hat{x}_j\right),$$
// $$\frac{\partial L}{\partial \gamma_i} = \sum_{\text{rows}} g_i \hat{x}_i,
//   \qquad \frac{\partial L}{\partial \beta_i} = \sum_{\text{rows}} g_i.$$
LUCID_INTERNAL void layer_norm_backward_f32(const float* x,
                                            const float* gamma,
                                            const float* saved_mean,
                                            const float* saved_rstd,
                                            const float* g,
                                            float* dx,
                                            float* dgamma,
                                            float* dbeta,
                                            std::size_t outer,
                                            std::size_t N);

// Double-precision counterpart to :cpp:func:`layer_norm_backward_f32`.
//
// Parameters
// ----------
// x : const double*
//     Original forward input of shape ``(outer, N)``.
// gamma : const double*
//     Affine weight of shape ``(N,)``.
// saved_mean, saved_rstd : const double*
//     Per-row statistics from the matching forward call.
// g : const double*
//     Upstream gradient of shape ``(outer, N)``.
// dx : double*
//     Output gradient w.r.t. ``x``; *overwritten*.
// dgamma, dbeta : double*
//     Output gradients of shape ``(N,)``; *accumulated* — caller must zero
//     them before the first invocation.
// outer, N : std::size_t
//     Layout extents.
LUCID_INTERNAL void layer_norm_backward_f64(const double* x,
                                            const double* gamma,
                                            const double* saved_mean,
                                            const double* saved_rstd,
                                            const double* g,
                                            double* dx,
                                            double* dgamma,
                                            double* dbeta,
                                            std::size_t outer,
                                            std::size_t N);

// Single-precision RMSNorm forward — divides each row by its root-mean-square
// magnitude and scales by a per-feature gain.
//
// Unlike LayerNorm there is no mean subtraction and no bias parameter, so the
// kernel only stores ``rstd = 1/sqrt(mean(x^2) + eps)`` per row.  This makes
// RMSNorm faster and is the form used in many modern transformer variants.
//
// Parameters
// ----------
// x : const float*
//     Input buffer of shape ``(outer, N)``.
// gamma : const float*
//     Per-feature scale of shape ``(N,)``.
// y : float*
//     Output buffer of shape ``(outer, N)``.
// saved_rstd : float*
//     Per-row reciprocal RMS, shape ``(outer,)``; consumed by
//     :cpp:func:`rms_norm_backward_f32`.
// outer, N : std::size_t
//     Number of rows and normalised-axis length.
// eps : double
//     Stabiliser added to ``mean(x^2)`` before the square root; cast to
//     ``float``.
//
// Math
// ----
// $$y_i = \gamma_i \frac{x_i}{\sqrt{\frac{1}{N}\sum_j x_j^2 + \varepsilon}}.$$
//
// References
// ----------
// Zhang & Sennrich, "Root Mean Square Layer Normalization" (NeurIPS 2019).
LUCID_INTERNAL void rms_norm_forward_f32(const float* x,
                                         const float* gamma,
                                         float* y,
                                         float* saved_rstd,
                                         std::size_t outer,
                                         std::size_t N,
                                         double eps);

// Double-precision counterpart to :cpp:func:`rms_norm_forward_f32`.
//
// Parameters
// ----------
// x : const double*
//     Input buffer of shape ``(outer, N)``.
// gamma : const double*
//     Per-feature scale of shape ``(N,)``.
// y : double*
//     Output buffer of shape ``(outer, N)``.
// saved_rstd : double*
//     Per-row reciprocal RMS, shape ``(outer,)``.
// outer, N : std::size_t
//     Layout extents.
// eps : double
//     Variance stabiliser.
LUCID_INTERNAL void rms_norm_forward_f64(const double* x,
                                         const double* gamma,
                                         double* y,
                                         double* saved_rstd,
                                         std::size_t outer,
                                         std::size_t N,
                                         double eps);

// Single-precision RMSNorm backward — produces input and weight gradients.
//
// Reuses the saved per-row ``rstd`` to avoid recomputing the row's RMS.
// ``dgamma`` is *accumulated*; callers reducing over multiple sub-batches
// must zero it once at the start.  ``dx`` is written fully (not accumulated).
//
// Parameters
// ----------
// x : const float*
//     Original forward input of shape ``(outer, N)``.
// gamma : const float*
//     Affine scale of shape ``(N,)``.
// saved_rstd : const float*
//     Per-row reciprocal RMS from the matching forward call.
// g : const float*
//     Upstream gradient of shape ``(outer, N)``.
// dx : float*
//     Output gradient w.r.t. ``x``; shape ``(outer, N)``; *overwritten*.
// dgamma : float*
//     Output gradient w.r.t. ``gamma``; shape ``(N,)``; *accumulated*.
// outer, N : std::size_t
//     Layout extents.
//
// Math
// ----
// With $r = \text{rstd}$ and $c = \sum_j \gamma_j g_j x_j$,
// $$\frac{\partial L}{\partial x_i} = r \!\left(\gamma_i g_i - \frac{x_i r^2 c}{N}\right),
//   \qquad \frac{\partial L}{\partial \gamma_i} = \sum_{\text{rows}} g_i x_i r.$$
LUCID_INTERNAL void rms_norm_backward_f32(const float* x,
                                          const float* gamma,
                                          const float* saved_rstd,
                                          const float* g,
                                          float* dx,
                                          float* dgamma,
                                          std::size_t outer,
                                          std::size_t N);

// Double-precision counterpart to :cpp:func:`rms_norm_backward_f32`.
//
// Parameters
// ----------
// x : const double*
//     Original forward input of shape ``(outer, N)``.
// gamma : const double*
//     Affine scale of shape ``(N,)``.
// saved_rstd : const double*
//     Per-row reciprocal RMS from the matching forward call.
// g : const double*
//     Upstream gradient of shape ``(outer, N)``.
// dx : double*
//     Output gradient w.r.t. ``x``; *overwritten*.
// dgamma : double*
//     Output gradient w.r.t. ``gamma``; *accumulated* — caller must zero
//     before the first invocation.
// outer, N : std::size_t
//     Layout extents.
LUCID_INTERNAL void rms_norm_backward_f64(const double* x,
                                          const double* gamma,
                                          const double* saved_rstd,
                                          const double* g,
                                          double* dx,
                                          double* dgamma,
                                          std::size_t outer,
                                          std::size_t N);

// ---------------------------------------------------------------------------
// Lp normalisation.  x, y, grad and dx are (outer, axis_len, inner) row-major;
// the norm is (outer, inner).
//
// Everything is taken relative to the slice's largest magnitude or to the
// denominator itself, so no intermediate leaves the float range the answer is
// in.  ``sum(|x|^p)`` of a float32 slice near 1e25 is 1e50 and one near 1e-25
// is 1e-50 with p = 2, although the norm and the normalised row are ordinary
// floats; BLAS's ``nrm2`` rescales by the largest magnitude for the same
// reason.  The accumulators are double, as the powers are.
// ---------------------------------------------------------------------------

// ``x^e`` for a fixed exponent ``e``.  A small exponent that is a multiple of a
// half — 2, 3 and 1.5 above all, the usual orders of a norm — is a square root and
// a few multiplications by squaring, where ``std::pow`` is ~15 ns; any other
// exponent takes ``std::pow``.  ``x^0`` is 1 for every ``x``, NaN included, as
// ``std::pow``'s is.
struct LpPower {
    static constexpr int kMaxHalves = 64;
    double exponent;
    int halves;  // ``2 * exponent`` when that is a whole number in [0, kMaxHalves], else -1

    explicit LpPower(double e)
        : exponent(e),
          halves((e >= 0.0 && 2.0 * e <= kMaxHalves && 2.0 * e == std::floor(2.0 * e))
                     ? static_cast<int>(2.0 * e)
                     : -1) {}

    double operator()(double x) const {
        if (halves < 0)
            return std::pow(x, exponent);
        double result = (halves & 1) != 0 ? std::sqrt(x) : 1.0;
        double base = x;
        for (int n = halves >> 1; n != 0; n >>= 1) {
            if (n & 1)
                result *= base;
            base *= base;
        }
        return result;
    }
};

// The largest magnitude of the ``len`` entries ``x[a * stride]``, NaN when any entry is.
template <typename T>
double lp_slice_top(const T* x, std::size_t len, std::size_t stride) {
    double top = 0.0;
    bool nan_seen = false;
    for (std::size_t a = 0; a < len; ++a) {
        const double v = std::abs(static_cast<double>(x[a * stride]));
        nan_seen = nan_seen || v != v;
        top = v > top ? v : top;
    }
    return nan_seen ? std::numeric_limits<double>::quiet_NaN() : top;
}

// ``(sum |x_a|^p)^(1/p)`` of the ``len`` entries ``x[a * stride]``, ``power``
// being ``x^p``.  The sum is the plain one in double when no power can leave the
// double range: an order of at most 1 has no power larger than its base, and a
// float up to 3.4e38 raised to p <= 6 is below 1e231 (its smallest, 1.4e-45, is
// above 1e-270).  Otherwise it is
// ``top * (sum (|x_a| / top)^p)^(1/p)`` with ``top`` the largest magnitude, so
// every power is in [0, 1]: a NaN anywhere is the answer, an infinity makes it
// inf, an all-zero slice is 0.
template <typename T>
double
lp_slice_norm(const T* x, std::size_t len, std::size_t stride, double ord, const LpPower& power) {
    constexpr double kFloatSafeOrder = 6.0;
    if (!(ord > 1.0) || (sizeof(T) < sizeof(double) && ord <= kFloatSafeOrder)) {
        double acc = 0.0;
        for (std::size_t a = 0; a < len; ++a)
            acc += power(std::abs(static_cast<double>(x[a * stride])));
        return ord == 2.0 ? std::sqrt(acc) : std::pow(acc, 1.0 / ord);
    }
    const double top = lp_slice_top(x, len, stride);
    if (top != top || top == 0.0 || std::isinf(top))
        return top;
    double acc = 0.0;
    // A multiplication is cheaper than a division; ``1 / top`` overflows only for
    // a subnormal double ``top``, which divides instead.
    const double inv = 1.0 / top;
    const bool by_inverse = std::isfinite(inv);
    for (std::size_t a = 0; a < len; ++a) {
        const double v = std::abs(static_cast<double>(x[a * stride]));
        acc += power(by_inverse ? v * inv : v / top);
    }
    return top * (ord == 2.0 ? std::sqrt(acc) : std::pow(acc, 1.0 / ord));
}

// ``y = x / d`` with ``d = max(||x||_p, eps)``, and ``d`` into ``norm`` — negated
// for a slice whose norm is below eps.  Under the floor ``d`` is the constant eps,
// so the backward's second term (the dependence of ``d`` on ``x``) is absent; the
// sign is how the backward, which has no eps and only ``d``, tells ``||x|| < eps``
// from ``||x|| == eps`` (a tie keeps its derivative).  ``lp_normalize_backward_typed``
// reads it, as does ``GpuBackend::lp_normalize_backward``.
template <typename T>
void lp_normalize_forward_typed(
    const T* x, T* y, T* norm, int outer, int axis_len, int inner, double ord, double eps) {
    const T floor = static_cast<T>(eps);
    const LpPower power(ord);
    for (int o = 0; o < outer; ++o)
        for (int n = 0; n < inner; ++n) {
            const std::size_t base = static_cast<std::size_t>(o) * axis_len * inner + n;
            const T nm = static_cast<T>(lp_slice_norm(x + base, axis_len, inner, ord, power));
            // ``max(nm, eps)``, written so a NaN norm survives it: ``nm > eps`` is
            // false for NaN, and the ternary would otherwise divide the row by eps
            // (``std::fmax`` returns the non-NaN operand, the same swallowing).
            const T denom = (nm > floor || nm != nm) ? nm : floor;
            norm[static_cast<std::size_t>(o) * inner + n] = nm < floor ? -denom : denom;
            for (int a = 0; a < axis_len; ++a) {
                const std::size_t idx = base + static_cast<std::size_t>(a) * inner;
                y[idx] = x[idx] / denom;
            }
        }
}

// ``dx = (g - sgn(u) |u|^(p-1) <g, u>) / d`` with ``u = x / d`` — the usual
// ``g/d - sgn(x) |x|^(p-1) <g, x> / d^(p+1)`` after factoring out ``d``.  ``u`` is
// the forward output (every entry at most 1 in magnitude once ``d`` is the norm),
// so the form never raises a magnitude to a power of the norm, which leaves the
// float range for a norm outside roughly 1e-13 .. 1e13 (``d^3``).  For p = 2 the
// factor ``sgn(u) |u|`` is ``u``.  A slice the forward floored (``norm`` negative,
// see ``lp_normalize_forward_typed``) divides by the constant eps: no second term.
template <typename T>
void lp_normalize_backward_typed(
    const T* x, const T* norm, const T* g, T* dx, int outer, int axis_len, int inner, double ord) {
    const bool two = ord == 2.0;
    const LpPower power(ord - 1.0);
    auto weight = [&](double u) {
        if (two)
            return u;
        const double sgn = u > 0.0 ? 1.0 : (u < 0.0 ? -1.0 : 0.0);
        return sgn * power(std::abs(u));
    };
    for (int o = 0; o < outer; ++o)
        for (int n = 0; n < inner; ++n) {
            const std::size_t base = static_cast<std::size_t>(o) * axis_len * inner + n;
            const T saved = norm[static_cast<std::size_t>(o) * inner + n];
            const bool floored = saved < T{0};
            // ``d`` is at least eps, so its inverse is finite for any eps above ~1e-308.
            const double inv = 1.0 / static_cast<double>(floored ? -saved : saved);
            double proj = 0.0;
            for (int a = 0; a < axis_len && !floored; ++a) {
                const std::size_t idx = base + static_cast<std::size_t>(a) * inner;
                proj += static_cast<double>(g[idx]) * (static_cast<double>(x[idx]) * inv);
            }
            for (int a = 0; a < axis_len; ++a) {
                const std::size_t idx = base + static_cast<std::size_t>(a) * inner;
                const double u = static_cast<double>(x[idx]) * inv;
                dx[idx] = static_cast<T>((static_cast<double>(g[idx]) - weight(u) * proj) * inv);
            }
        }
}

// ---------------------------------------------------------------------------
// Entrywise p-norm over any set of axes (``CpuBackend::p_norm_forward``), p > 1.
//
// The same range-safe value as :func:`lp_slice_norm` — the sum in double when no float
// power can leave it, relative to the largest magnitude otherwise — computed to be as
// fast as the plain float graph it replaces:
//   * the orders a norm is taken at most (3/2, 2, 3, 4) are straight multiplications;
//   * a sum is four independent partial sums: one running sum is a chain of dependent
//     additions, each entry waiting four cycles for the one before it;
//   * any other order goes through vForce ``pow`` on blocks of ratios in [0, 1] (so a
//     float power cannot overflow, and an underflowing one is negligible next to the 1
//     the largest entry contributes), the sum still in double;
//   * the slices of a large reduction are chunks of one parallel loop, each slice one
//     chunk's, so the result is the same at any core count.
// ---------------------------------------------------------------------------

struct SquarePower {
    double operator()(double x) const { return x * x; }
};
struct CubePower {
    double operator()(double x) const { return x * x * x; }
};
struct QuarticPower {
    double operator()(double x) const {
        const double square = x * x;
        return square * square;
    }
};
struct ThreeHalvesPower {
    double operator()(double x) const { return x * std::sqrt(x); }
};

// ``f`` called with the cheapest functor for ``power``.
template <class F>
double with_fixed_power(const LpPower& power, F&& f) {
    switch (power.halves) {
    case 3:
        return f(ThreeHalvesPower{});
    case 4:
        return f(SquarePower{});
    case 6:
        return f(CubePower{});
    case 8:
        return f(QuarticPower{});
    default:
        return f(power);
    }
}

// ``sum power(entry(x[a * stride]))`` over ``len`` entries in four partial sums.
template <typename T, class Entry, class Power>
double lp_power_sum(const T* x, std::size_t len, std::size_t stride, Entry entry, Power power) {
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    std::size_t a = 0;
    for (; a + 4 <= len; a += 4) {
        s0 += power(entry(x[a * stride]));
        s1 += power(entry(x[(a + 1) * stride]));
        s2 += power(entry(x[(a + 2) * stride]));
        s3 += power(entry(x[(a + 3) * stride]));
    }
    for (; a < len; ++a)
        s0 += power(entry(x[a * stride]));
    return (s0 + s1) + (s2 + s3);
}

// ``sum (|x_a| / top)^p`` by vForce, for an order with no cheap functor.  The ratios and
// their powers are in the entries' own precision (a ratio is in [0, 1], so a float power cannot
// overflow, and one that underflows is negligible next to the 1 the largest entry
// contributes), the sum in double.  A float ratio is formed in float when the reciprocal of
// ``top`` is a normal float: one rounding more than the division, still within an ulp.
template <typename T>
double lp_ratio_power_sum(const T* x, std::size_t len, std::size_t stride, double top, double p) {
    constexpr std::size_t kBlock = 512;
    T ratio[kBlock];
    T expo[kBlock];
    T powered[kBlock];
    const double inv = 1.0 / top;
    const bool by_inverse = std::isfinite(inv);
    const T narrow_inv = static_cast<T>(inv);
    const bool narrow = std::isnormal(narrow_inv);
    std::fill(expo, expo + std::min(kBlock, len), static_cast<T>(p));
    double acc = 0.0;
    for (std::size_t off = 0; off < len; off += kBlock) {
        const std::size_t m = std::min(kBlock, len - off);
        const T* block = x + off * stride;
        if (narrow) {
            for (std::size_t j = 0; j < m; ++j)
                ratio[j] = std::abs(block[j * stride]) * narrow_inv;
        } else {
            for (std::size_t j = 0; j < m; ++j) {
                const double v = std::abs(static_cast<double>(block[j * stride]));
                ratio[j] = static_cast<T>(by_inverse ? v * inv : v / top);
            }
        }
        if constexpr (sizeof(T) == sizeof(float))
            vpow_f32(ratio, expo, powered, m);
        else
            vpow_f64(ratio, expo, powered, m);
        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
        std::size_t j = 0;
        for (; j + 4 <= m; j += 4) {
            s0 += powered[j];
            s1 += powered[j + 1];
            s2 += powered[j + 2];
            s3 += powered[j + 3];
        }
        for (; j < m; ++j)
            s0 += powered[j];
        acc += (s0 + s1) + (s2 + s3);
    }
    return acc;
}

// One order's ``(sum |x_a|^p)^(1/p)`` for p > 1, the value of :func:`lp_slice_norm` (NaN,
// infinity and all-zero slices included), summed over fixed blocks of ``kBlock`` entries
// that are added in order: a block may run on any core, the total does not change.
//
// ``plain``: a float entry raised to p <= 6 stays inside double, so the sum is the plain one.
// Otherwise it is relative to the slice's largest magnitude ``top``.
template <typename T>
class PNorm {
public:
    static constexpr std::size_t kBlock = std::size_t{1} << 14;

    explicit PNorm(double p)
        : p_(p), power_(p), plain_(sizeof(T) < sizeof(double) && p <= 6.0 && power_.halves >= 0) {}

    bool plain() const { return plain_; }

    // ``sum`` of one block's powers (relative to ``top`` unless the sum is plain).
    double block_sum(const T* x, std::size_t len, std::size_t stride, double top) const {
        const auto magnitude = [](T v) { return std::abs(static_cast<double>(v)); };
        if (plain_)
            return with_fixed_power(
                power_, [&](auto pw) { return lp_power_sum(x, len, stride, magnitude, pw); });
        if (power_.halves < 0)
            return lp_ratio_power_sum(x, len, stride, top, p_);
        const double inv = 1.0 / top;
        if (std::isfinite(inv)) {
            const auto ratio = [inv](T v) { return std::abs(static_cast<double>(v)) * inv; };
            return with_fixed_power(
                power_, [&](auto pw) { return lp_power_sum(x, len, stride, ratio, pw); });
        }
        const auto ratio = [top](T v) { return std::abs(static_cast<double>(v)) / top; };
        return with_fixed_power(power_,
                                [&](auto pw) { return lp_power_sum(x, len, stride, ratio, pw); });
    }

    // The norm from the blocks' total and the slice's ``top`` (unused when plain).
    double finish(double total, double top) const {
        const double root = p_ == 2.0 ? std::sqrt(total) : std::pow(total, 1.0 / p_);
        return plain_ ? root : top * root;
    }

    // Whether ``top`` is the answer by itself: NaN, an infinity, or an all-zero slice.
    static bool settled(double top) { return top != top || top == 0.0 || std::isinf(top); }

    // The whole slice on this thread.
    double serial(const T* x, std::size_t len, std::size_t stride) const {
        double top = 1.0;
        if (!plain_) {
            top = lp_slice_top(x, len, stride);
            if (settled(top))
                return top;
        }
        double total = 0.0;
        for (std::size_t at = 0; at < len; at += kBlock)
            total += block_sum(x + at * stride, std::min(kBlock, len - at), stride, top);
        return finish(total, top);
    }

    // The same value with the blocks spread over the cores, for a slice worth it.
    double blocked(const T* x, std::size_t len, std::size_t stride) const {
        const std::size_t blocks = (len + kBlock - 1) / kBlock;
        const auto extent = [&](std::size_t b) { return std::min(kBlock, len - b * kBlock); };
        std::vector<double> part(blocks);
        double top = 1.0;
        if (!plain_) {
            parallel_for(blocks, 1, [&](std::size_t lo, std::size_t hi) {
                for (std::size_t b = lo; b < hi; ++b)
                    part[b] = lp_slice_top(x + b * kBlock * stride, extent(b), stride);
            });
            top = 0.0;
            for (const double t : part) {
                if (t != t)
                    return t;
                top = t > top ? t : top;
            }
            if (settled(top))
                return top;
        }
        parallel_for(blocks, 1, [&](std::size_t lo, std::size_t hi) {
            for (std::size_t b = lo; b < hi; ++b)
                part[b] = block_sum(x + b * kBlock * stride, extent(b), stride, top);
        });
        double total = 0.0;
        for (const double s : part)
            total += s;
        return finish(total, top);
    }

private:
    double p_;
    LpPower power_;
    bool plain_;
};

// ``out[o * inner + n] = ||x[o, :, n]||_p`` for ``x`` of layout ``(outer, len, inner)``: the
// norm over the middle axis, the layout a reduction over one run of axes has.  Many slices are
// chunks of one parallel loop, a few long ones are split into blocks; either way each slice's
// value is the same.
template <typename T>
void p_norm_typed(
    const T* x, T* out, std::size_t outer, std::size_t len, std::size_t inner, double p) {
    const PNorm<T> norm(p);
    const std::size_t slices = outer * inner;
    const auto at = [&](std::size_t slice) {
        const std::size_t o = slice / inner;
        return x + o * len * inner + (slice - o * inner);
    };
    if (len > 2 * PNorm<T>::kBlock && slices < 8 * parallel_workers()) {
        for (std::size_t slice = 0; slice < slices; ++slice)
            out[slice] = static_cast<T>(norm.blocked(at(slice), len, inner));
        return;
    }
    parallel_for(slices, std::max<std::size_t>(1, PNorm<T>::kBlock / std::max<std::size_t>(len, 1)),
                 [&](std::size_t lo, std::size_t hi) {
                     for (std::size_t slice = lo; slice < hi; ++slice)
                         out[slice] = static_cast<T>(norm.serial(at(slice), len, inner));
                 });
}

// ``x`` of ``shape`` rearranged so the ``reduced`` axes are the trailing ones, row-major in the
// order they appear.  The reduction then runs over contiguous slices.
template <typename T>
std::vector<T>
p_norm_gather_reduced_last(const T* x, const Shape& shape, const std::vector<bool>& reduced) {
    const std::size_t rank = shape.size();
    const Stride stride = contiguous_stride(shape, 1);
    std::vector<std::size_t> order;
    for (std::size_t d = 0; d < rank; ++d)
        if (!reduced[d])
            order.push_back(d);
    for (std::size_t d = 0; d < rank; ++d)
        if (reduced[d])
            order.push_back(d);
    std::vector<std::int64_t> extent(rank), step(rank), coord(rank, 0);
    for (std::size_t i = 0; i < rank; ++i) {
        extent[i] = shape[order[i]];
        step[i] = stride[order[i]];
    }
    std::vector<T> out(shape_numel(shape));
    std::int64_t at = 0;
    for (T& slot : out) {
        slot = x[at];
        for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(rank) - 1; i >= 0; --i) {
            const auto k = static_cast<std::size_t>(i);
            at += step[k];
            if (++coord[k] < extent[k])
                break;
            at -= step[k] * extent[k];
            coord[k] = 0;
        }
    }
    return out;
}

// ``p_norm_typed`` over ``reduced`` axes of any layout: one contiguous run of axes is
// ``(outer, len, inner)`` as it stands, anything else is gathered to put them last first.
template <typename T>
void p_norm_over_axes(
    const T* x, T* out, const Shape& shape, const std::vector<bool>& reduced, double p) {
    const std::size_t rank = shape.size();
    std::size_t first = rank, last = 0, count = 0;
    for (std::size_t d = 0; d < rank; ++d)
        if (reduced[d]) {
            first = std::min(first, d);
            last = d;
            ++count;
        }
    if (count == 0) {
        p_norm_typed(x, out, shape_numel(shape), 1, 1, p);
        return;
    }
    auto span = [&](std::size_t lo, std::size_t hi) {
        std::size_t n = 1;
        for (std::size_t d = lo; d < hi; ++d)
            n *= static_cast<std::size_t>(shape[d]);
        return n;
    };
    if (last - first + 1 == count) {
        p_norm_typed(x, out, span(0, first), span(first, last + 1), span(last + 1, rank), p);
        return;
    }
    const std::vector<T> moved = p_norm_gather_reduced_last(x, shape, reduced);
    std::size_t len = 1;
    for (std::size_t d = 0; d < rank; ++d)
        if (reduced[d])
            len *= static_cast<std::size_t>(shape[d]);
    p_norm_typed(moved.data(), out, shape_numel(shape) / std::max<std::size_t>(len, 1), len, 1, p);
}

}  // namespace lucid::backend::cpu
