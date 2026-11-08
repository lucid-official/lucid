// lucid/_C/backend/cpu/Vdsp.h
//
// Thin wrappers around Apple Accelerate vDSP vector arithmetic primitives used
// by the CPU backend.  Each function dispatches to the corresponding vDSP_*
// intrinsic (stride-1 paths) so the compiler and hardware can use NEON SIMD.
// "f32" variants wrap single-precision vDSP routines; "f64" wraps their "D"
// double-precision counterparts.  Integer add variants (i32, i64) have no vDSP
// equivalent and fall back to scalar loops.

#pragma once

#include <cstddef>
#include <cstdint>

#include "../../api.h"

namespace lucid::backend::cpu {

// Single-precision elementwise vector addition.
//
// Computes $\text{out}_i = a_i + b_i$ via ``vDSP_vadd``.  Stride-1 contiguous
// inputs are required; broadcasting is handled at a higher layer.
//
// Parameters
// ----------
// a, b : const float*
//     Input buffers of length $n$.
// out : float*
//     Output buffer of length $n$ (may alias ``a`` or ``b``).
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vDSP_vadd``.
LUCID_INTERNAL void vadd_f32(const float* a, const float* b, float* out, std::size_t n);

// Single-precision elementwise vector subtraction.
//
// Computes $\text{out}_i = a_i - b_i$ via ``vDSP_vsub``.
//
// Parameters
// ----------
// a, b : const float*
//     Input buffers.
// out : float*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vDSP_vsub``.
LUCID_INTERNAL void vsub_f32(const float* a, const float* b, float* out, std::size_t n);

// Single-precision elementwise vector multiplication.
//
// Computes $\text{out}_i = a_i \cdot b_i$ via ``vDSP_vmul``.
//
// Parameters
// ----------
// a, b : const float*
//     Input buffers.
// out : float*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vDSP_vmul``.
LUCID_INTERNAL void vmul_f32(const float* a, const float* b, float* out, std::size_t n);

// Single-precision elementwise vector division, correctly rounded.  A plain
// loop: ``vDSP_vdiv`` is not correctly rounded on macOS 26 (see Vdsp.cpp).
//
// Parameters
// ----------
// a, b : const float*
//     Numerator and denominator buffers.
// out : float*
//     Output buffer.
// n : std::size_t
//     Element count.
LUCID_INTERNAL void vdiv_f32(const float* a, const float* b, float* out, std::size_t n);

// Double-precision elementwise addition.  Uses ``vDSP_vaddD``.
//
// Parameters
// ----------
// a, b : const double*
//     Input buffers.
// out : double*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vDSP_vaddD``.
LUCID_INTERNAL void vadd_f64(const double* a, const double* b, double* out, std::size_t n);

// Double-precision elementwise subtraction.  Uses ``vDSP_vsubD``.
//
// Parameters
// ----------
// a, b : const double*
//     Input buffers.
// out : double*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vDSP_vsubD``.
LUCID_INTERNAL void vsub_f64(const double* a, const double* b, double* out, std::size_t n);

// Double-precision elementwise multiplication.  Uses ``vDSP_vmulD``.
//
// Parameters
// ----------
// a, b : const double*
//     Input buffers.
// out : double*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vDSP_vmulD``.
LUCID_INTERNAL void vmul_f64(const double* a, const double* b, double* out, std::size_t n);

// Double-precision elementwise division, correctly rounded.  A plain loop:
// ``vDSP_vdivD`` is not correctly rounded (see Vdsp.cpp).
//
// Parameters
// ----------
// a, b : const double*
//     Numerator / denominator buffers.
// out : double*
//     Output buffer.
// n : std::size_t
//     Element count.
LUCID_INTERNAL void vdiv_f64(const double* a, const double* b, double* out, std::size_t n);

// Single-precision elementwise negation.
//
// Computes $\text{out}_i = -\text{in}_i$ via ``vDSP_vneg``.
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
// References
// ----------
// Accelerate.framework ``vDSP_vneg``.
LUCID_INTERNAL void vneg_f32(const float* in, float* out, std::size_t n);

// Single-precision elementwise absolute value.
//
// Computes $\text{out}_i = |\text{in}_i|$ via ``vDSP_vabs`` (sign-bit mask
// on the SIMD register; cheaper than a branch).
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
// Accelerate.framework ``vDSP_vabs``.
LUCID_INTERNAL void vabs_f32(const float* in, float* out, std::size_t n);

// Single-precision elementwise square.
//
// Computes $\text{out}_i = \text{in}_i^2$ via ``vDSP_vsq``.  Fused single
// multiply (no ``sqrt`` is taken).
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
// Accelerate.framework ``vDSP_vsq``.
LUCID_INTERNAL void vsq_f32(const float* in, float* out, std::size_t n);

// Double-precision elementwise negation.  Uses ``vDSP_vnegD``.
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
// Accelerate.framework ``vDSP_vnegD``.
LUCID_INTERNAL void vneg_f64(const double* in, double* out, std::size_t n);

// Double-precision elementwise absolute value.  Uses ``vDSP_vabsD``.
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
// Accelerate.framework ``vDSP_vabsD``.
LUCID_INTERNAL void vabs_f64(const double* in, double* out, std::size_t n);

// Double-precision elementwise square.  Uses ``vDSP_vsqD``.
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
// Accelerate.framework ``vDSP_vsqD``.
LUCID_INTERNAL void vsq_f64(const double* in, double* out, std::size_t n);

// Single-precision scalar-vector add broadcast.
//
// Computes $\text{out}_i = \text{in}_i + s$ via ``vDSP_vsadd``.
//
// Parameters
// ----------
// in : const float*
//     Input buffer.
// scalar : float
//     Scalar value added to every element.
// out : float*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vDSP_vsadd``.
LUCID_INTERNAL void vsadd_f32(const float* in, float scalar, float* out, std::size_t n);

// Single-precision scalar-vector multiply broadcast.
//
// Computes $\text{out}_i = \text{in}_i \cdot s$ via ``vDSP_vsmul``.
//
// Parameters
// ----------
// in : const float*
//     Input buffer.
// scalar : float
//     Scalar multiplier.
// out : float*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vDSP_vsmul``.
LUCID_INTERNAL void vsmul_f32(const float* in, float scalar, float* out, std::size_t n);

// Double-precision scalar add broadcast.  Uses ``vDSP_vsaddD``.
//
// Parameters
// ----------
// in : const double*
//     Input buffer.
// scalar : double
//     Scalar added elementwise.
// out : double*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vDSP_vsaddD``.
LUCID_INTERNAL void vsadd_f64(const double* in, double scalar, double* out, std::size_t n);

// Double-precision scalar multiply broadcast.  Uses ``vDSP_vsmulD``.
//
// Parameters
// ----------
// in : const double*
//     Input buffer.
// scalar : double
//     Scalar multiplier.
// out : double*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vDSP_vsmulD``.
LUCID_INTERNAL void vsmul_f64(const double* in, double scalar, double* out, std::size_t n);

// Widen a single-precision buffer to double precision.
//
// Lets a kernel that must accumulate in ``double`` still read float32 inputs
// through Accelerate rather than an element-at-a-time cast loop.
//
// Parameters
// ----------
// in : const float*
//     Input buffer.
// out : double*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vDSP_vspdp``.
LUCID_INTERNAL void widen_f32_f64(const float* in, double* out, std::size_t n);

// Single-precision scaled vector plus vector, $out = a \cdot s + b$.
//
// The three-buffer form of a scaled accumulation: unlike a BLAS ``axpy`` the
// addend and the destination are separate, so the first term of a linear
// combination can be written straight into a fresh output without copying the
// base into it first.
//
// Parameters
// ----------
// a : const float*
//     Vector to scale.
// scalar : float
//     Scale applied to ``a``.
// b : const float*
//     Vector added to the scaled result; may alias ``out``.
// out : float*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// Math
// ----
// $$ \text{out}_i = s \cdot a_i + b_i $$
//
// References
// ----------
// Accelerate.framework ``vDSP_vsma``.
LUCID_INTERNAL void
vsma_f32(const float* a, float scalar, const float* b, float* out, std::size_t n);

// Double-precision scaled vector plus vector, $out = a \cdot s + b$.
//
// Parameters
// ----------
// a : const double*
//     Vector to scale.
// scalar : double
//     Scale applied to ``a``.
// b : const double*
//     Vector added to the scaled result; may alias ``out``.
// out : double*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// Math
// ----
// $$ \text{out}_i = s \cdot a_i + b_i $$
//
// References
// ----------
// Accelerate.framework ``vDSP_vsmaD``.
LUCID_INTERNAL void
vsma_f64(const double* a, double scalar, const double* b, double* out, std::size_t n);

// Single-precision ReLU clamp.
//
// Computes $\text{out}_i = \max(0, \text{in}_i)$ via ``vDSP_vthres`` with
// threshold $0$ (values below zero are replaced by zero).
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
// Branchless ``(x < 0) ? 0 : x`` rather than ``vDSP_vthres``, which
// returns the threshold for NaN and so swallowed NaN on the CPU while
// Metal propagated it.  Also faster.
LUCID_INTERNAL void vrelu_f32(const float* in, float* out, std::size_t n);

// Double-precision ReLU clamp.  Branchless, for the same reason as the
// single-precision form: ``vDSP_vthresD`` does not propagate NaN.
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
// IEEE: ``NaN < 0`` is false, so NaN passes through.
LUCID_INTERNAL void vrelu_f64(const double* in, double* out, std::size_t n);

// Single-precision elementwise maximum.
//
// Computes $\text{out}_i = \max(a_i, b_i)$ via ``vDSP_vmax``.
//
// Parameters
// ----------
// a, b : const float*
//     Input buffers.
// out : float*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vDSP_vmax``.
LUCID_INTERNAL void vmax_f32(const float* a, const float* b, float* out, std::size_t n);

// Single-precision elementwise minimum.
//
// Computes $\text{out}_i = \min(a_i, b_i)$ via ``vDSP_vmin``.
//
// Parameters
// ----------
// a, b : const float*
//     Input buffers.
// out : float*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vDSP_vmin``.
LUCID_INTERNAL void vmin_f32(const float* a, const float* b, float* out, std::size_t n);

// Double-precision elementwise maximum.  Uses ``vDSP_vmaxD``.
//
// Parameters
// ----------
// a, b : const double*
//     Input buffers.
// out : double*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vDSP_vmaxD``.
LUCID_INTERNAL void vmax_f64(const double* a, const double* b, double* out, std::size_t n);

// Double-precision elementwise minimum.  Uses ``vDSP_vminD``.
//
// Parameters
// ----------
// a, b : const double*
//     Input buffers.
// out : double*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vDSP_vminD``.
LUCID_INTERNAL void vmin_f64(const double* a, const double* b, double* out, std::size_t n);

// Single-precision >= boolean mask (no vDSP equivalent).
//
// Computes $\text{out}_i = (a_i \ge b_i) \,?\, 1 : 0$ via a scalar loop;
// used by autograd for ReLU/Max gradient masks where the comparison result
// must be a float.
//
// Parameters
// ----------
// a, b : const float*
//     Input buffers.
// out : float*
//     Output buffer (0.0 / 1.0 mask).
// n : std::size_t
//     Element count.
LUCID_INTERNAL void vge_mask_f32(const float* a, const float* b, float* out, std::size_t n);

// Single-precision < boolean mask.
//
// Computes $\text{out}_i = (a_i < b_i) \,?\, 1 : 0$ via a scalar loop.
//
// Parameters
// ----------
// a, b : const float*
//     Input buffers.
// out : float*
//     Output buffer (0.0 / 1.0 mask).
// n : std::size_t
//     Element count.
LUCID_INTERNAL void vle_mask_f32(const float* a, const float* b, float* out, std::size_t n);

// Double-precision >= boolean mask.
//
// Parameters
// ----------
// a, b : const double*
//     Input buffers.
// out : double*
//     Output buffer (0.0 / 1.0 mask).
// n : std::size_t
//     Element count.
LUCID_INTERNAL void vge_mask_f64(const double* a, const double* b, double* out, std::size_t n);

// Double-precision < boolean mask.
//
// Parameters
// ----------
// a, b : const double*
//     Input buffers.
// out : double*
//     Output buffer (0.0 / 1.0 mask).
// n : std::size_t
//     Element count.
LUCID_INTERNAL void vle_mask_f64(const double* a, const double* b, double* out, std::size_t n);

// Single-precision array sum reduction.
//
// Computes $\sum_{i=0}^{n-1} \text{in}_i$ via ``vDSP_sve`` (single-pass,
// pairwise tree-style accumulation — numerically more stable than a serial
// loop).
//
// Parameters
// ----------
// in : const float*
//     Input buffer.
// n : std::size_t
//     Element count.
//
// Returns
// -------
// float
//     Sum of all $n$ input elements.
//
// References
// ----------
// Accelerate.framework ``vDSP_sve``.
LUCID_INTERNAL float vsum_f32(const float* in, std::size_t n);

// Double-precision array sum reduction.  Uses ``vDSP_sveD``.
//
// Parameters
// ----------
// in : const double*
//     Input buffer.
// n : std::size_t
//     Element count.
//
// Returns
// -------
// double
//     Sum of all $n$ input elements.
//
// References
// ----------
// Accelerate.framework ``vDSP_sveD``.
LUCID_INTERNAL double vsum_f64(const double* in, std::size_t n);

// Single-precision arithmetic mean reduction.
//
// Computes $\frac{1}{n} \sum_{i=0}^{n-1} \text{in}_i$ via ``vDSP_meanv``.
//
// Parameters
// ----------
// in : const float*
//     Input buffer.
// n : std::size_t
//     Element count.
//
// Returns
// -------
// float
//     Arithmetic mean of the input.
//
// References
// ----------
// Accelerate.framework ``vDSP_meanv``.
LUCID_INTERNAL float vmean_f32(const float* in, std::size_t n);

// Double-precision mean reduction.  Uses ``vDSP_meanvD``.
//
// Parameters
// ----------
// in : const double*
//     Input buffer.
// n : std::size_t
//     Element count.
//
// Returns
// -------
// double
//     Mean of the input.
//
// References
// ----------
// Accelerate.framework ``vDSP_meanvD``.
LUCID_INTERNAL double vmean_f64(const double* in, std::size_t n);

// Clamp to ``[lo, hi]``, ``min(max(x, lo), hi)`` with NaN passed through.  A zero
// that ties a zero bound takes the sign the extremum rule gives it (max of a
// signed-zero tie is +0, min is -0, as ``vmax`` / ``vmin``), so a +0 floor also
// takes -0 and a -0 cap also takes +0.
LUCID_INTERNAL void vclip_f32(const float* in, float* out, std::size_t n, float lo, float hi);
LUCID_INTERNAL void vclip_f64(const double* in, double* out, std::size_t n, double lo, double hi);

// Single-precision maximum-value reduction.
//
// Computes $\max_i \text{in}_i$ via ``vDSP_maxv``.
//
// Parameters
// ----------
// in : const float*
//     Input buffer.
// n : std::size_t
//     Element count.
//
// Returns
// -------
// float
//     Maximum element.
//
// References
// ----------
// Accelerate.framework ``vDSP_maxv``.
LUCID_INTERNAL float vmaxval_f32(const float* in, std::size_t n);

// Double-precision maximum-value reduction.  Uses ``vDSP_maxvD``.
//
// Parameters
// ----------
// in : const double*
//     Input buffer.
// n : std::size_t
//     Element count.
//
// Returns
// -------
// double
//     Maximum element.
//
// References
// ----------
// Accelerate.framework ``vDSP_maxvD``.
LUCID_INTERNAL double vmaxval_f64(const double* in, std::size_t n);

// Single-precision inner / dot product.
//
// Computes $\langle a, b \rangle = \sum_{i=0}^{n-1} a_i b_i$ via
// ``vDSP_dotpr``.
//
// Parameters
// ----------
// a, b : const float*
//     Input vectors.
// n : std::size_t
//     Element count.
//
// Returns
// -------
// float
//     Scalar inner product.
//
// References
// ----------
// Accelerate.framework ``vDSP_dotpr``.
LUCID_INTERNAL float vdotpr_f32(const float* a, const float* b, std::size_t n);

// Double-precision dot product.  Uses ``vDSP_dotprD``.
//
// Parameters
// ----------
// a, b : const double*
//     Input vectors.
// n : std::size_t
//     Element count.
//
// Returns
// -------
// double
//     Scalar inner product.
//
// References
// ----------
// Accelerate.framework ``vDSP_dotprD``.
LUCID_INTERNAL double vdotpr_f64(const double* a, const double* b, std::size_t n);

// Single-precision fused multiply-add.
//
// Computes $\text{out}_i = a_i \cdot b_i + c_i$ via ``vDSP_vma`` (single
// vectorised pass; cheaper and more accurate than separate ``vmul`` then
// ``vadd``).
//
// Parameters
// ----------
// a, b, c : const float*
//     Input buffers.
// out : float*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vDSP_vma``.
LUCID_INTERNAL void
vmadd_f32(const float* a, const float* b, const float* c, float* out, std::size_t n);

// Double-precision fused multiply-add.  Uses ``vDSP_vmaD``.
//
// Parameters
// ----------
// a, b, c : const double*
//     Input buffers.
// out : double*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// References
// ----------
// Accelerate.framework ``vDSP_vmaD``.
LUCID_INTERNAL void
vmadd_f64(const double* a, const double* b, const double* c, double* out, std::size_t n);

// Signed 32-bit integer elementwise addition.
//
// Computes $\text{out}_i = a_i + b_i$ via a plain scalar loop because vDSP
// has no integer primitives.  Only used by the integer-tensor add path.
//
// Parameters
// ----------
// a, b : const std::int32_t*
//     Input buffers.
// out : std::int32_t*
//     Output buffer.
// n : std::size_t
//     Element count.
//
// Notes
// -----
// Overflow wraps modulo $2^{32}$ following standard C++ semantics for signed
// integer addition on Apple Silicon.
LUCID_INTERNAL void
vadd_i32(const std::int32_t* a, const std::int32_t* b, std::int32_t* out, std::size_t n);

// Signed 64-bit integer elementwise addition.  Scalar loop fallback.
//
// Parameters
// ----------
// a, b : const std::int64_t*
//     Input buffers.
// out : std::int64_t*
//     Output buffer.
// n : std::size_t
//     Element count.
LUCID_INTERNAL void
vadd_i64(const std::int64_t* a, const std::int64_t* b, std::int64_t* out, std::size_t n);

// Complex C64 elementwise multiplication on interleaved storage.
//
// Multiplies two complex64 vectors stored as interleaved ``[re, im, re, im,
// ...]`` floats.  Internally rebinds the buffers as ``DSPSplitComplex`` views
// (via Accelerate's ``DSPComplex *`` cast) and delegates to ``vDSP_zvmul``
// with ``flag = 1`` (no conjugation).
//
// Parameters
// ----------
// a, b : const float*
//     Interleaved complex buffers, length $2n$ floats.
// out : float*
//     Interleaved complex output buffer, length $2n$ floats.
// n : std::size_t
//     Number of *complex* elements (NOT floats).
//
// Math
// ----
// $$ \text{out}_i = a_i \cdot b_i, \quad a_i, b_i \in \mathbb{C} $$
//
// References
// ----------
// Accelerate.framework ``vDSP_zvmul``.
LUCID_INTERNAL void vzmul_c64(const float* a, const float* b, float* out, std::size_t n);
// Interleaved double-precision complex lanes, without narrowing to float32.
LUCID_INTERNAL void vzdiv_c128(const double* a, const double* b, double* out, std::size_t n);
// Interleaved double-precision complex lanes, without narrowing to float32.
LUCID_INTERNAL void vzmul_c128(const double* a, const double* b, double* out, std::size_t n);
// Interleaved double-precision complex lanes, without narrowing to float32.
LUCID_INTERNAL void vzsub_c128(const double* a, const double* b, double* out, std::size_t n);
// Interleaved double-precision complex lanes, without narrowing to float32.
LUCID_INTERNAL void vzadd_c128(const double* a, const double* b, double* out, std::size_t n);

// Complex C64 add / subtract / divide on interleaved storage.
//
// The same layout and loop as :func:`vzmul_c64`, for the three arithmetic
// ops the CPU stream used to refuse for complex input ("dtype not
// supported") while Metal ran them.  Division is ``std::complex``'s own.
//
// Parameters
// ----------
// a, b : const float*
//     Interleaved complex buffers, length $2n$ floats.
// out : float*
//     Interleaved complex output buffer, length $2n$ floats.
// n : std::size_t
//     Number of *complex* elements (NOT floats).
LUCID_INTERNAL void vzadd_c64(const float* a, const float* b, float* out, std::size_t n);
LUCID_INTERNAL void vzsub_c64(const float* a, const float* b, float* out, std::size_t n);
LUCID_INTERNAL void vzdiv_c64(const float* a, const float* b, float* out, std::size_t n);

// Complex C64 conjugation on interleaved storage.
//
// Returns the complex conjugate of an interleaved complex64 vector by
// negating every imaginary component (every odd-indexed float) and leaving
// the real components untouched.  Implemented as a ``vDSP_vneg`` over the
// stride-2 imaginary view.
//
// Parameters
// ----------
// a : const float*
//     Interleaved complex input, length $2n$ floats.
// out : float*
//     Interleaved complex output, length $2n$ floats (may alias ``a``).
// n : std::size_t
//     Number of complex elements.
//
// Math
// ----
// $$ \text{out}_i = \overline{a_i} = \mathrm{Re}(a_i) - i \, \mathrm{Im}(a_i) $$
//
// References
// ----------
// Accelerate.framework ``vDSP_vneg``.
LUCID_INTERNAL void vzconj_c64(const float* a, float* out, std::size_t n);

}  // namespace lucid::backend::cpu
