// lucid/_C/backend/cpu/Vforce.h
//
// Thin wrappers around Apple Accelerate vForce transcendental math routines.
// All functions apply the corresponding vv* vector math function element-wise
// to an array of n values.  vForce functions are vectorized for Apple Silicon
// NEON and are significantly faster than calling std:: math functions in a loop.
//
// Naming convention: v<op>_f32 wraps the vv*f (float) variant; v<op>_f64
// wraps the vv* (double) variant.  The element count n is passed as an int*
// to match the vForce API signature.

#pragma once

#include <cstddef>

#include "../../api.h"

namespace lucid::backend::cpu {

// Single-precision elementwise natural exponential.
//
// Computes $\text{out}_i = e^{\text{in}_i}$ via ``vvexpf``.
//
// Parameters
// ----------
// in : const float*
//     Input buffer.
// out : float*
//     Output buffer (may alias ``in``).
// n : std::size_t
//     Element count.
//
// Notes
// -----
// vForce trades roughly one ulp of accuracy at the extremes for much higher
// throughput than ``std::expf``.
//
// References
// ----------
// Accelerate.framework ``vvexpf``.
LUCID_INTERNAL void vexp_f32(const float* in, float* out, std::size_t n);

// Single-precision elementwise natural logarithm.
//
// Computes $\text{out}_i = \ln(\text{in}_i)$ via ``vvlogf``.  Negative or
// zero inputs follow IEEE rules (yield NaN / -Inf).
//
// Parameters
// ----------
// in : const float*
//     Input buffer.
// out : float*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vvlogf``.
LUCID_INTERNAL void vlog_f32(const float* in, float* out, std::size_t n);

// Single-precision elementwise square root, correctly rounded.  A plain
// loop: ``vvsqrtf`` is not correctly rounded (see Vforce.cpp).
//
// Parameters
// ----------
// in : const float*
//     Input buffer.
// out : float*
//     Output buffer.
// n : std::size_t
//     Element count.
LUCID_INTERNAL void vsqrt_f32(const float* in, float* out, std::size_t n);

// Single-precision elementwise hyperbolic tangent.
//
// Computes $\text{out}_i = \tanh(\text{in}_i)$ via ``vvtanhf``, with
// $|x| \le 3.52 \times 10^{-4}$ set back to x (see split_edges in Vforce.cpp).
//
// Parameters
// ----------
// in : const float*
//     Input buffer.
// out : float*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vvtanhf``.
LUCID_INTERNAL void vtanh_f32(const float* in, float* out, std::size_t n);

// Single-precision elementwise hyperbolic tangent via raw ``vvtanhf`` without
// the near-zero identity post-fix.  Used by GELU's tanh approximation, where
// ``1 + tanh(u)`` absorbs a 1-ulp difference on a tiny ``u`` (LCD-595).
//
// Parameters
// ----------
// in : const float*
//     Input buffer.
// out : float*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vvtanhf``.
LUCID_INTERNAL void vtanh_unfixed_f32(const float* in, float* out, std::size_t n);

// Single-precision elementwise sine.
//
// Computes $\text{out}_i = \sin(\text{in}_i)$ via ``vvsinf``, with
// $|x| \le 3.52 \times 10^{-4}$ set back to x.  Near odd multiples of $\pi/2$
// the result can be 1-2 ulp short of $\pm 1$.
//
// Parameters
// ----------
// in : const float*
//     Input buffer (radians).
// out : float*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vvsinf``.
LUCID_INTERNAL void vsin_f32(const float* in, float* out, std::size_t n);

// Single-precision elementwise cosine.
//
// Computes $\text{out}_i = \cos(\text{in}_i)$ via ``vvcosf``, with
// $|x| \le 2^{-12}$ set to 1.  Near other multiples of $\pi$ the result can be
// 1-2 ulp short of $\pm 1$.
//
// Parameters
// ----------
// in : const float*
//     Input buffer (radians).
// out : float*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vvcosf``.
LUCID_INTERNAL void vcos_f32(const float* in, float* out, std::size_t n);

// Single-precision elementwise tangent.
//
// Computes $\text{out}_i = \tan(\text{in}_i)$ via ``vvtanf``, with
// $|x| \le 3.52 \times 10^{-4}$ (and -0) set back to x.
//
// Parameters
// ----------
// in : const float*
//     Input buffer (radians).
// out : float*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vvtanf``.
LUCID_INTERNAL void vtan_f32(const float* in, float* out, std::size_t n);

// Double-precision elementwise exponential.  Uses ``vvexp``.
//
// Parameters
// ----------
// in : const double*
//     Input buffer.
// out : double*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vvexp``.
LUCID_INTERNAL void vexp_f64(const double* in, double* out, std::size_t n);

// Double-precision elementwise natural log.  Uses ``vvlog``.
//
// Parameters
// ----------
// in : const double*
//     Input buffer.
// out : double*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vvlog``.
LUCID_INTERNAL void vlog_f64(const double* in, double* out, std::size_t n);

// Double-precision elementwise square root.  Uses ``vvsqrt``.
//
// Parameters
// ----------
// in : const double*
//     Input buffer.
// out : double*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vvsqrt``.
LUCID_INTERNAL void vsqrt_f64(const double* in, double* out, std::size_t n);

// Double-precision elementwise tanh.  Uses ``vvtanh``, with
// $|x| \le 1.35 \times 10^{-8}$ set back to x.
//
// Parameters
// ----------
// in : const double*
//     Input buffer.
// out : double*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vvtanh``.
LUCID_INTERNAL void vtanh_f64(const double* in, double* out, std::size_t n);

// Double-precision elementwise tanh via raw ``vvtanh`` without the near-zero
// identity post-fix.  Used by GELU's tanh approximation (LCD-595).
//
// Parameters
// ----------
// in : const double*
//     Input buffer.
// out : double*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vvtanh``.
LUCID_INTERNAL void vtanh_unfixed_f64(const double* in, double* out, std::size_t n);

// Double-precision elementwise sine.  Uses ``vvsin``; near odd multiples of
// $\pi/2$ the result can be 1-3 ulp short of $\pm 1$.
//
// Parameters
// ----------
// in : const double*
//     Input buffer (radians).
// out : double*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vvsin``.
LUCID_INTERNAL void vsin_f64(const double* in, double* out, std::size_t n);

// Double-precision elementwise cosine.  Uses ``vvcos``, with $|x| < 2^{-26.5}$
// set to 1; near other multiples of $\pi$ the result can be 1-3 ulp short of
// $\pm 1$.
//
// Parameters
// ----------
// in : const double*
//     Input buffer (radians).
// out : double*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vvcos``.
LUCID_INTERNAL void vcos_f64(const double* in, double* out, std::size_t n);

// Double-precision elementwise tangent.  Uses ``vvtan``, with
// $|x| \le 1.35 \times 10^{-8}$ (and -0) set back to x.
//
// Parameters
// ----------
// in : const double*
//     Input buffer (radians).
// out : double*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vvtan``.
LUCID_INTERNAL void vtan_f64(const double* in, double* out, std::size_t n);

// Single-precision elementwise power.
//
// Computes $\text{out}_i = \text{base}_i^{\text{expo}_i}$ via ``vvpowf``.
// vForce's native argument order is ``(out, expo, base, &n)``; this wrapper
// reorders to the more natural ``(base, expo, out, n)`` for call sites.
//
// Parameters
// ----------
// base : const float*
//     Bases.
// expo : const float*
//     Exponents.
// out : float*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// Math
// ----
// $$ \text{out}_i = \text{base}_i^{\,\text{expo}_i} $$
//
// References
// ----------
// Accelerate.framework ``vvpowf``.
LUCID_INTERNAL void vpow_f32(const float* base, const float* expo, float* out, std::size_t n);

// Double-precision elementwise power.  Uses ``vvpow``.
//
// Parameters
// ----------
// base : const double*
//     Bases.
// expo : const double*
//     Exponents.
// out : double*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vvpow``.
LUCID_INTERNAL void vpow_f64(const double* base, const double* expo, double* out, std::size_t n);

// Single-precision elementwise arcsine.
//
// Computes $\text{out}_i = \arcsin(\text{in}_i)$ via ``vvasinf``.  Outputs
// in $[-\pi/2, \pi/2]$; inputs must be in $[-1, 1]$.
//
// Parameters
// ----------
// in : const float*
//     Input buffer in $[-1, 1]$.
// out : float*
//     Output buffer (radians).
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vvasinf``.
LUCID_INTERNAL void vasin_f32(const float* in, float* out, std::size_t n);

// Single-precision elementwise arccosine.
//
// Computes $\text{out}_i = \arccos(\text{in}_i)$ via ``vvacosf``.  Outputs
// in $[0, \pi]$.
//
// Parameters
// ----------
// in : const float*
//     Input buffer in $[-1, 1]$.
// out : float*
//     Output buffer (radians).
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vvacosf``.
LUCID_INTERNAL void vacos_f32(const float* in, float* out, std::size_t n);

// Single-precision elementwise arctangent.
//
// Computes $\text{out}_i = \arctan(\text{in}_i)$ via ``vvatanf``.  Outputs
// in $(-\pi/2, \pi/2)$.
//
// Parameters
// ----------
// in : const float*
//     Input buffer (any real).
// out : float*
//     Output buffer (radians).
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vvatanf``.
LUCID_INTERNAL void vatan_f32(const float* in, float* out, std::size_t n);

// Double-precision arcsine.  Uses ``vvasin``.
//
// Parameters
// ----------
// in : const double*
//     Input in $[-1, 1]$.
// out : double*
//     Output (radians).
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vvasin``.
LUCID_INTERNAL void vasin_f64(const double* in, double* out, std::size_t n);

// Double-precision arccosine.  Uses ``vvacos``.
//
// Parameters
// ----------
// in : const double*
//     Input in $[-1, 1]$.
// out : double*
//     Output (radians).
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vvacos``.
LUCID_INTERNAL void vacos_f64(const double* in, double* out, std::size_t n);

// Double-precision arctangent.  Uses ``vvatan``.
//
// Parameters
// ----------
// in : const double*
//     Input.
// out : double*
//     Output (radians).
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vvatan``.
LUCID_INTERNAL void vatan_f64(const double* in, double* out, std::size_t n);

// Single-precision hyperbolic sine.
//
// Computes $\text{out}_i = \sinh(\text{in}_i)$ via ``vvsinhf``.
//
// Parameters
// ----------
// in : const float*
//     Input buffer.
// out : float*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vvsinhf``.
LUCID_INTERNAL void vsinh_f32(const float* in, float* out, std::size_t n);

// Single-precision hyperbolic cosine.
//
// Computes $\text{out}_i = \cosh(\text{in}_i)$ via ``vvcoshf``.
//
// Parameters
// ----------
// in : const float*
//     Input buffer.
// out : float*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vvcoshf``.
LUCID_INTERNAL void vcosh_f32(const float* in, float* out, std::size_t n);

// Double-precision sinh.  Uses ``vvsinh``.
//
// Parameters
// ----------
// in : const double*
//     Input buffer.
// out : double*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vvsinh``.
LUCID_INTERNAL void vsinh_f64(const double* in, double* out, std::size_t n);

// Double-precision cosh.  Uses ``vvcosh``.
//
// Parameters
// ----------
// in : const double*
//     Input buffer.
// out : double*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vvcosh``.
LUCID_INTERNAL void vcosh_f64(const double* in, double* out, std::size_t n);

// Single-precision base-2 logarithm.
//
// Computes $\text{out}_i = \log_2(\text{in}_i)$ via ``vvlog2f``.
//
// Parameters
// ----------
// in : const float*
//     Input buffer.
// out : float*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vvlog2f``.
LUCID_INTERNAL void vlog2_f32(const float* in, float* out, std::size_t n);

// Double-precision base-2 log.  Uses ``vvlog2``.
//
// Parameters
// ----------
// in : const double*
//     Input buffer.
// out : double*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vvlog2``.
LUCID_INTERNAL void vlog2_f64(const double* in, double* out, std::size_t n);

// Single-precision Gauss error function.
//
// Computes $\text{out}_i = \mathrm{erf}(\text{in}_i)$ via ``vverff``, where
// $\mathrm{erf}(x) = \tfrac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} \, dt$.  Used
// by GELU activation among others.
//
// Parameters
// ----------
// in : const float*
//     Input buffer.
// out : float*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vverff``.
LUCID_INTERNAL void verf_f32(const float* in, float* out, std::size_t n);

// Double-precision error function.  Uses ``vverf``.
//
// Parameters
// ----------
// in : const double*
//     Input buffer.
// out : double*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vverf``.
LUCID_INTERNAL void verf_f64(const double* in, double* out, std::size_t n);

// Single-precision IEEE absolute value (vForce variant).
//
// Computes $\text{out}_i = |\text{in}_i|$ via ``vvfabsf``.  Equivalent to
// the vDSP ``vabs_f32`` for finite inputs; behaviour on NaNs differs from
// the bitwise-mask form because vvfabs preserves NaN payloads.
//
// Parameters
// ----------
// in : const float*
//     Input buffer.
// out : float*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vvfabsf``.
LUCID_INTERNAL void vfabs_f32(const float* in, float* out, std::size_t n);

// Double-precision IEEE absolute value.  Uses ``vvfabs``.
//
// Parameters
// ----------
// in : const double*
//     Input buffer.
// out : double*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vvfabs``.
LUCID_INTERNAL void vfabs_f64(const double* in, double* out, std::size_t n);

// Single-precision reciprocal $\text{out}_i = 1 / \text{in}_i$, correctly
// rounded.  A plain loop: ``vvrecf`` is not correctly rounded (see
// Vforce.cpp).
//
// Parameters
// ----------
// in : const float*
//     Input buffer.
// out : float*
//     Output buffer.
// n : std::size_t
//     Element count.
LUCID_INTERNAL void vrec_f32(const float* in, float* out, std::size_t n);

// Double-precision reciprocal.  Uses ``vvrec``.
//
// Parameters
// ----------
// in : const double*
//     Input buffer.
// out : double*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vvrec``.
LUCID_INTERNAL void vrec_f64(const double* in, double* out, std::size_t n);

// Single-precision floor.
//
// Computes $\text{out}_i = \lfloor \text{in}_i \rfloor$ via ``vvfloorf``
// (largest integer ≤ input, returned as float).
//
// Parameters
// ----------
// in : const float*
//     Input buffer.
// out : float*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vvfloorf``.
LUCID_INTERNAL void vfloor_f32(const float* in, float* out, std::size_t n);

// Double-precision floor.  Uses ``vvfloor``.
//
// Parameters
// ----------
// in : const double*
//     Input buffer.
// out : double*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vvfloor``.
LUCID_INTERNAL void vfloor_f64(const double* in, double* out, std::size_t n);

// Single-precision ceiling.
//
// Computes $\text{out}_i = \lceil \text{in}_i \rceil$ via ``vvceilf``.
//
// Parameters
// ----------
// in : const float*
//     Input buffer.
// out : float*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vvceilf``.
LUCID_INTERNAL void vceil_f32(const float* in, float* out, std::size_t n);

// Double-precision ceiling.  Uses ``vvceil``.
//
// Parameters
// ----------
// in : const double*
//     Input buffer.
// out : double*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vvceil``.
LUCID_INTERNAL void vceil_f64(const double* in, double* out, std::size_t n);

// Single-precision round-to-nearest-even.
//
// Computes $\text{out}_i = \mathrm{round}(\text{in}_i)$ via ``vvnintf``;
// ties are broken to the nearest even integer (banker's rounding) — same
// rule as IEEE 754 default rounding mode.
//
// Parameters
// ----------
// in : const float*
//     Input buffer.
// out : float*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vvnintf``.
LUCID_INTERNAL void vround_f32(const float* in, float* out, std::size_t n);

// Double-precision round-to-nearest-even.  Uses ``vvnint``.
//
// Parameters
// ----------
// in : const double*
//     Input buffer.
// out : double*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vvnint``.
LUCID_INTERNAL void vround_f64(const double* in, double* out, std::size_t n);

}  // namespace lucid::backend::cpu
