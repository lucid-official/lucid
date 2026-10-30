// lucid/_C/ops/composite/Math.h
//
// Elementwise math operations whose forward pass is built from existing
// primitive ops in ``ufunc/`` and ``bfunc/``.  No new backward classes —
// gradients flow through the underlying primitives' autograd nodes.
//
// Membership criterion for this header: a single mathematical formula whose
// implementation requires composing two or more primitive ``*_op`` calls.
// Standalone primitives belong in ``ufunc/`` or ``bfunc/``.
//
// Listing:
//   log10  = log(x) / log(10)
//   log1p  = log(1 + x)
//   exp2   = exp(x · log(2))
//   (trunc, which frac uses, is a step function of its own — ufunc/Discrete.h)
//   frac   = x − trunc(x)
//   atan2  = quadrant-aware arctangent (where + arctan + ±π corrections)
//   hypot  = sqrt(a² + b²)
//   logaddexp = m + log(exp(a − m) + exp(b − m))        [stable; m = max(a,b)]
//
// ``fmod`` and ``remainder`` were here as ``a − trunc(a / b) · b`` and
// ``a − floor(a / b) · b``; the rounded quotient made them inexact, so they
// are kernels now (bfunc/Floordiv.h).

#pragma once

#include "../../api.h"
#include "../../core/fwd.h"

namespace lucid {

// ── unary compositions ──────────────────────────────────────────────────────

// Base-10 logarithm: $y = \log_{10}(x) = \ln x / \ln 10$.
//
// Composite over :func:`log_op` + :func:`div_op`.  Materialises $\ln 10$
// as a same-shape constant so the divide does not need scalar broadcasting.
// Gradient flows through ``LogBackward`` and ``DivBackward``.
//
// Math
// ----
// $$
//   y = \frac{\ln x}{\ln 10}, \qquad \frac{\partial y}{\partial x} = \frac{1}{x \ln 10}
// $$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.  Domain: $x > 0$.
//
// Returns
// -------
// TensorImplPtr
//     Tensor of the same shape and dtype as ``a``.
//
// Notes
// -----
// Inherits the precision of the underlying ``log_op``; no specialised
// base-10 kernel is used.
//
// See Also
// --------
// :func:`log_op`, :func:`log1p_op`.
LUCID_API TensorImplPtr log10_op(const TensorImplPtr& a);

// Natural log of 1 + x: $y = \ln(1 + x)$.
//
// Composite over :func:`add_op` + :func:`log_op`, with a rounding
// correction that keeps full relative precision for $|x| \ll 1$.  The
// gradient is ``LogBackward``'s $1 / (1 + x)$: $+\infty$ at $x = -1$ and
// finite below it.
//
// Math
// ----
// $$
//   y = \ln(1 + x), \qquad \frac{\partial y}{\partial x} = \frac{1}{1 + x}
// $$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.  $x = -1$ gives $-\infty$, $x < -1$ NaN, $+\infty$
//     gives $+\infty$.
//
// Returns
// -------
// TensorImplPtr
//     Tensor of the same shape and dtype as ``a``.
//
// See Also
// --------
// :func:`log_op`, :func:`log10_op`.
LUCID_API TensorImplPtr log1p_op(const TensorImplPtr& a);

// Base-2 exponential: $y = 2^{x} = \exp(x \ln 2)$.
//
// Composite over :func:`mul_op` + :func:`exp_op`.  Multiplying against a
// constant tensor keeps the gradient chain visible to ``MulBackward`` and
// ``ExpBackward``.
//
// Math
// ----
// $$
//   y = \exp(x \cdot \ln 2),
//   \qquad \frac{\partial y}{\partial x} = (\ln 2) \cdot 2^{x}
// $$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.
//
// Returns
// -------
// TensorImplPtr
//     Tensor of the same shape and dtype as ``a``.
//
// See Also
// --------
// :func:`exp_op`.
LUCID_API TensorImplPtr exp2_op(const TensorImplPtr& a);

// Fractional part: $y = x - \mathrm{trunc}(x)$.
//
// Composite over :func:`trunc_op` + :func:`sub_op`.  Because the ``trunc``
// branch contributes zero gradient, the gradient w.r.t. ``a`` is exactly
// the upstream gradient (identity).
//
// Math
// ----
// $$
//   y = x - \mathrm{trunc}(x),
//   \qquad \frac{\partial y}{\partial x} = 1
// $$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.
//
// Returns
// -------
// TensorImplPtr
//     Tensor of the same shape and dtype as ``a``, with $y \in (-1, 1)$
//     and sign matching $x$.
//
// See Also
// --------
// :func:`trunc_op`.
LUCID_API TensorImplPtr frac_op(const TensorImplPtr& a);

// ── binary compositions ─────────────────────────────────────────────────────

// Quadrant-aware arctangent: $y = \mathrm{atan2}(y_\text{in}, x_\text{in})$.
//
// Composite over :func:`arctan_op` plus :func:`where_op` branches that
// patch the result with $\pm \pi$ / $\pm \pi/2$ / $0$ corrections based on
// the signs of the operands.  The divide $y / x$ is guarded by
// substituting $1$ for $x = 0$ so the safe branch is always defined.
// Gradient flows through the underlying primitives.
//
// Math
// ----
// $$
//   \mathrm{atan2}(y, x) =
//   \begin{cases}
//     \arctan(y/x) & x > 0 \\
//     \arctan(y/x) + \pi & x < 0,\ y \ge 0 \\
//     \arctan(y/x) - \pi & x < 0,\ y < 0 \\
//     +\pi/2 & x = 0,\ y > 0 \\
//     -\pi/2 & x = 0,\ y < 0 \\
//     0 & x = 0,\ y = 0
//   \end{cases}
// $$
//
// Parameters
// ----------
// y : TensorImplPtr
//     Numerator (sine-like) operand.
// x : TensorImplPtr
//     Denominator (cosine-like) operand.  Broadcastable with ``y``.
//
// Returns
// -------
// TensorImplPtr
//     Angles in $[-\pi, \pi]$ on the broadcast shape of ``y`` and ``x``.
//
// See Also
// --------
// :func:`arctan_op`.
LUCID_API TensorImplPtr atan2_op(const TensorImplPtr& y, const TensorImplPtr& x);

// Pythagorean length: $y = \sqrt{a^2 + b^2}$.
//
// Composite over :func:`square_op` + :func:`add_op` + :func:`sqrt_op`.
// **Naive** formulation — does not perform IEEE-754 overflow-safe scaling
// (i.e. no $\max(|a|, |b|) \cdot \sqrt{1 + (\min / \max)^2}$ trick),
// matching the engine's ``square_op`` precision.
//
// Math
// ----
// $$
//   y = \sqrt{a^2 + b^2}
// $$
//
// Parameters
// ----------
// a, b : TensorImplPtr
//     Broadcastable operands.
//
// Returns
// -------
// TensorImplPtr
//     Tensor on the broadcast shape; values $\ge 0$.
//
// Notes
// -----
// Scaled by the larger magnitude, so neither a huge nor a subnormal operand
// overflows or loses its low bits.  An infinite operand gives $+\infty$ even
// beside a NaN (C99 Annex G).  The rule itself is :func:`hypot_lanes`.
LUCID_API TensorImplPtr hypot_op(const TensorImplPtr& a, const TensorImplPtr& b);

// $\sqrt{a^2 + b^2}$ of two real floating tensors of one dtype, with
// :func:`hypot_op`'s value rule but none of its AMP casting or device
// alignment.
//
// This is the one place the magnitude rule lives.  :func:`hypot_op` calls it
// on its two operands and the complex ``abs`` calls it on the real and
// imaginary lanes, so the two cannot disagree about NaN, infinity or the
// subnormal range.
LUCID_API TensorImplPtr hypot_lanes(const TensorImplPtr& a, const TensorImplPtr& b);

// Numerically-stable $\log(\exp(a) + \exp(b))$ via the max-shift trick.
//
// Composite over :func:`maximum_op` + :func:`sub_op` + :func:`exp_op` +
// :func:`add_op` + :func:`log_op`.  Factoring out the per-pair max keeps
// the exponent's argument $\le 0$ so the underflow side bears the rounding
// error rather than the overflow side.
//
// Math
// ----
// $$
//   y = m + \ln\!\left( e^{a - m} + e^{b - m} \right),
//   \qquad m = \max(a, b)
// $$
//
// Parameters
// ----------
// a, b : TensorImplPtr
//     Broadcastable operands.
//
// Returns
// -------
// TensorImplPtr
//     Tensor on the broadcast shape.
//
// See Also
// --------
// :func:`logsumexp_op` — generalised reduction over axes.
LUCID_API TensorImplPtr logaddexp_op(const TensorImplPtr& a, const TensorImplPtr& b);

// Elementwise tolerance check: $|a - b| \le \mathrm{atol} + \mathrm{rtol} \cdot |b|$.
//
// Composite over :func:`sub_op` + :func:`abs_op` + :func:`mul_op` +
// :func:`add_op` + :func:`less_equal_op` — no dedicated kernel required.
// ``rtol`` and ``atol`` are materialised as same-shape constant tensors so
// the multiply does not depend on any scalar-broadcast support in the
// binary ops.  Returns a bool tensor; not differentiable.
//
// Math
// ----
// $$
//   y_i = \mathbb{1}\!\left[ |a_i - b_i| \le \mathrm{atol} + \mathrm{rtol} \cdot |b_i| \right]
// $$
//
// Parameters
// ----------
// a, b : TensorImplPtr
//     Broadcastable operands.
// rtol : double
//     Relative tolerance (asymmetric in ``b``).
// atol : double
//     Absolute tolerance.
//
// Returns
// -------
// TensorImplPtr
//     Bool tensor on the broadcast shape of ``a`` and ``b``.
//
// Notes
// -----
// The asymmetry in ``b`` matches the reference framework's convention —
// swap operands if the symmetric formulation is needed.
LUCID_API TensorImplPtr isclose_op(const TensorImplPtr& a,
                                   const TensorImplPtr& b,
                                   double rtol,
                                   double atol);

}  // namespace lucid
