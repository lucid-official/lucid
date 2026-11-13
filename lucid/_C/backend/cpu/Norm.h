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

#include <cmath>
#include <cstddef>
#include <limits>

#include "../../api.h"

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

// ``(sum |x_a|^p)^(1/p)`` of the ``len`` entries ``x[a * stride]``, ``power``
// being ``x^p``.  The sum is the plain one in double when no power can leave the
// double range: an order of at most 1 has no power larger than its base, and a
// float up to 3.4e38 raised to p <= 6 is below 1e231 (its smallest, 1.4e-45, is
// above 1e-270).  Otherwise it is
// ``top * (sum (|x_a| / top)^p)^(1/p)`` with ``top`` the largest magnitude, so
// every power is in [0, 1]: a NaN anywhere is the answer, an infinity makes it
// inf, an all-zero slice is 0.
template <typename T>
double lp_slice_norm(const T* x, int len, std::size_t stride, double ord, const LpPower& power) {
    constexpr double kFloatSafeOrder = 6.0;
    if (!(ord > 1.0) || (sizeof(T) < sizeof(double) && ord <= kFloatSafeOrder)) {
        double acc = 0.0;
        for (int a = 0; a < len; ++a)
            acc += power(std::abs(static_cast<double>(x[a * stride])));
        return ord == 2.0 ? std::sqrt(acc) : std::pow(acc, 1.0 / ord);
    }
    double top = 0.0;
    bool nan_seen = false;
    for (int a = 0; a < len; ++a) {
        const double v = std::abs(static_cast<double>(x[a * stride]));
        nan_seen = nan_seen || v != v;
        top = v > top ? v : top;
    }
    if (nan_seen)
        return std::numeric_limits<double>::quiet_NaN();
    if (top == 0.0 || std::isinf(top))
        return top;
    double acc = 0.0;
    // A multiplication is cheaper than a division; ``1 / top`` overflows only for
    // a subnormal double ``top``, which divides instead.
    const double inv = 1.0 / top;
    const bool by_inverse = std::isfinite(inv);
    for (int a = 0; a < len; ++a) {
        const double v = std::abs(static_cast<double>(x[a * stride]));
        acc += power(by_inverse ? v * inv : v / top);
    }
    return top * (ord == 2.0 ? std::sqrt(acc) : std::pow(acc, 1.0 / ord));
}

// ``y = x / d`` with ``d = max(||x||_p, eps)``, and ``d`` itself into ``norm``.
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
            norm[static_cast<std::size_t>(o) * inner + n] = denom;
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
// factor ``sgn(u) |u|`` is ``u``.
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
            // ``d`` is at least eps, so its inverse is finite for any eps above ~1e-308.
            const double inv =
                1.0 / static_cast<double>(norm[static_cast<std::size_t>(o) * inner + n]);
            double proj = 0.0;
            for (int a = 0; a < axis_len; ++a) {
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

}  // namespace lucid::backend::cpu
