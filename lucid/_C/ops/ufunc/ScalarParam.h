// lucid/_C/ops/ufunc/ScalarParam.h
//
// Backward nodes for unary ops that carry a scalar hyper-parameter:
//   - PowScalarBackward  — x^exp  (exp is a floating-point scalar exponent)
//   - RPowScalarBackward — base^x (base is a floating-point scalar base)
//   - ClipBackward       — clamp(x, min, max)
//
// All three override the standard static forward() from UnaryKernel so they
// can capture the scalar on the backward node.  They do not use the generic
// dispatch() path.

#pragma once

#include <utility>

#include "../../api.h"
#include "../../autograd/Helpers.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/TensorImpl.h"
#include "../../core/fwd.h"
#include "_UnaryOp.h"

namespace lucid {

// Autograd node for scalar-exponent power: $y = x^{\text{exp}}$.
//
// The exponent is a Python ``float`` (here promoted to ``double``) captured
// on the node at forward time; the *input* tensor is saved so the backward
// pass can reconstruct $x^{\text{exp}-1}$ via the same backend kernel.
//
// Attributes
// ----------
// exp_ : double
//     Scalar exponent captured at forward; reused inside ``grad_formula``
//     to compute $x^{\text{exp}-1}$ and the leading multiplier.
// schema_v1 : OpSchema
//     Op name ``"pow_scalar"`` with ``AmpPolicy::ForceFP32`` (integer /
//     half precision are unsafe for arbitrary exponents).
//
// Math
// ----
// Forward
//
// $$y_i = x_i^{\text{exp}}$$
//
// Backward
//
// $$\frac{\partial L}{\partial x_i} = \text{exp} \cdot x_i^{\text{exp} - 1}
// \cdot \frac{\partial L}{\partial y_i}$$
//
// Notes
// -----
// Capturing only the scalar (rather than baking it into the op name) keeps
// the op table small.  ``pow_scalar`` is reused inside ``grad_formula`` to
// compute $x^{\text{exp}-1}$ — no separate "decrement-exponent" kernel.
//
// See Also
// --------
// :class:`RPowScalarBackward`, :func:`pow_scalar_op`.
class LUCID_API PowScalarBackward : public UnaryOp<PowScalarBackward> {
public:
    double exp_ = 0.0;
    static const OpSchema schema_v1;
    // Forward dispatch with scalar capture — computes $y = x^{\text{exp}}$
    // and wires the backward node manually so ``exp`` is persisted.
    //
    // Parameters
    // ----------
    // a : const TensorImplPtr&
    //     Input tensor (must be non-null; promoted to FP32 by the schema).
    // exp : double
    //     Scalar exponent.
    //
    // Returns
    // -------
    // TensorImplPtr
    //     Output tensor with the same shape and (post-promotion) dtype
    //     as ``a``; the backward node carries ``exp``.
    static TensorImplPtr forward(const TensorImplPtr& a, double exp);
    // Computes $\partial L/\partial x = \text{exp} \cdot x^{\text{exp}-1}
    // \cdot g$ from the saved input and the captured ``exp_``.
    //
    // Parameters
    // ----------
    // g : const Storage&
    //     Upstream gradient $\partial L/\partial y$.
    //
    // Returns
    // -------
    // Storage
    //     Gradient with respect to ``x`` — same shape, dtype, and device
    //     as the saved input.
    Storage grad_formula(const Storage& g);

    // Graph-mode equivalent of ``grad_formula`` used when higher-order
    // autograd is enabled — produces a fresh ``TensorImpl`` whose own
    // backward chain remains intact.
    //
    // Parameters
    // ----------
    // g : const TensorImplPtr&
    //     Upstream gradient tensor $\partial L/\partial y$.
    // a : const TensorImplPtr&
    //     Saved input tensor.
    // /* out */ : const TensorImplPtr&
    //     Saved output tensor (unused; preserved for signature parity
    //     with other ``grad_formula_impl`` overrides).
    //
    // Returns
    // -------
    // TensorImplPtr
    //     Tensor representing $\text{exp} \cdot x^{\text{exp}-1} \cdot g$
    //     with a live autograd graph.
    //
    // Math
    // ----
    // $$\frac{\partial}{\partial x}(x^e) = e \cdot x^{e-1}$$
    TensorImplPtr
    grad_formula_impl(const TensorImplPtr& g, const TensorImplPtr& a, const TensorImplPtr& out);
};

// Autograd node for scalar-base reverse power: $y = \text{base}^{x}$.
//
// The base is captured on the node; the *output* tensor (and the input
// when ``base == 0``) is saved because the gradient formula uses $y$
// directly and masks $x \geq 0$ when $\text{base} = 0$.
//
// Attributes
// ----------
// base_ : double
//     Scalar base captured at forward; ``std::log(base_)`` is computed
//     inside ``grad_formula``.
// kSavesInput : bool
//     ``false`` by default (input saved only when ``base_ == 0``).
// kSavesOutput : bool
//     ``true`` — the forward output $y = \text{base}^{x}$ is saved.
// schema_v1 : OpSchema
//     Op name ``"rpow_scalar"`` with ``AmpPolicy::ForceFP32``.
//
// Math
// ----
// Forward
//
// $$y_i = \text{base}^{x_i}$$
//
// Backward
//
// $$\frac{\partial L}{\partial x_i} = \ln(\text{base}) \cdot y_i \cdot
// \frac{\partial L}{\partial y_i}$$
//
// Notes
// -----
// Undefined for ``base <= 0`` (``ln`` would not be real).  Use
// ``pow_scalar_op`` if the exponent — not the base — is the scalar.
//
// See Also
// --------
// :class:`PowScalarBackward`, :func:`rpow_scalar_op`.
class LUCID_API RPowScalarBackward : public UnaryOp<RPowScalarBackward> {
public:
    double base_ = 0.0;

    static constexpr bool kSavesInput = false;
    static constexpr bool kSavesOutput = true;
    static const OpSchema schema_v1;
    // Forward dispatch with base capture — computes $y = \text{base}^{x}$,
    // saves the output on the backward node, and wires autograd manually
    // (``save_input=false``).
    //
    // Parameters
    // ----------
    // base : double
    //     Scalar base (must be positive for a real-valued gradient).
    // a : const TensorImplPtr&
    //     Input tensor — the exponent.
    //
    // Returns
    // -------
    // TensorImplPtr
    //     Output tensor with the same shape and (post-promotion) dtype
    //     as ``a``; the backward node carries ``base`` and the saved
    //     output.
    static TensorImplPtr forward(double base, const TensorImplPtr& a);
    // Computes $\partial L/\partial x = \ln(\text{base}) \cdot y \cdot g$
    // from the saved output and the captured ``base_``.
    //
    // Parameters
    // ----------
    // g : const Storage&
    //     Upstream gradient $\partial L/\partial y$.
    //
    // Returns
    // -------
    // Storage
    //     Gradient with respect to ``x``.
    Storage grad_formula(const Storage& g);
    // Graph-mode backward: $\ln(\text{base}) \cdot y \cdot g$ from the
    // tracked output, so the second derivative sees $y$'s dependence on $x$.
    TensorImplPtr
    grad_formula_impl(const TensorImplPtr& g, const TensorImplPtr&, const TensorImplPtr& out);
};

// Autograd node for element-wise clip (clamp):
// $y = \operatorname{clip}(x, \text{min}, \text{max})$.
//
// The gradient is the in-range indicator: gradient flows through where
// the input lies strictly between the bounds, and is zero on the
// saturated boundaries.
//
// Attributes
// ----------
// min_ : double
//     Lower clamp bound captured at forward.
// max_ : double
//     Upper clamp bound captured at forward.
// schema_v1 : OpSchema
//     Op name ``"clip"`` with ``AmpPolicy::KeepInput`` (clip is valid on
//     both integer and floating-point inputs).
//
// Math
// ----
// Forward
//
// $$y_i = \max\bigl(\text{min},\ \min(\text{max},\ x_i)\bigr)$$
//
// Backward
//
// $$\frac{\partial L}{\partial x_i} = \begin{cases}
//   \frac{\partial L}{\partial y_i} & \text{min} < x_i < \text{max} \\
//   0                                & \text{otherwise}
// \end{cases}$$
//
// Notes
// -----
// ``min_`` and ``max_`` are persisted so ``grad_formula`` can rebuild the
// in-range mask from the saved input rather than caching the mask itself.
//
// See Also
// --------
// :func:`clip_op`.
class LUCID_API ClipBackward : public UnaryOp<ClipBackward> {
public:
    TensorImplPtr
    grad_formula_impl(const TensorImplPtr& g, const TensorImplPtr& x, const TensorImplPtr& out);
    double min_ = 0.0;
    double max_ = 0.0;
    static const OpSchema schema_v1;
    // Forward dispatch with bounds capture — computes
    // $y = \operatorname{clip}(x, \text{min}, \text{max})$ and wires the
    // backward node manually with both bounds persisted.
    //
    // Parameters
    // ----------
    // a : const TensorImplPtr&
    //     Input tensor.
    // min_v : double
    //     Lower bound.
    // max_v : double
    //     Upper bound (must satisfy ``min_v <= max_v``).
    //
    // Returns
    // -------
    // TensorImplPtr
    //     Clamped output tensor with the same shape, dtype, and device
    //     as ``a``.
    static TensorImplPtr forward(const TensorImplPtr& a, double min_v, double max_v);
    // Computes the masked gradient $g \odot \mathbb{1}[\text{min} < x <
    // \text{max}]$ from the saved input and the persisted bounds.
    //
    // Parameters
    // ----------
    // g : const Storage&
    //     Upstream gradient $\partial L/\partial y$.
    //
    // Returns
    // -------
    // Storage
    //     Gradient with respect to ``x``; zero at every saturated slot.
    Storage grad_formula(const Storage& g);
};

// Autograd node for ``nan_to_num`` — finite values pass through, NaN and
// the infinities are replaced by the three scalars.
//
// Math
// ----
// $$\frac{\partial y}{\partial x} = \mathbb{1}[x \text{ finite}]$$
//
// The replaced slots are constants, so they carry no gradient; everywhere
// else the op is the identity.  This is the reference framework's rule, and
// it used to be missing: the op returned a detached tensor, so
// ``nan_to_num(x) + x`` differentiated to 1 instead of 2 — silently.
//
// Notes
// -----
// The three scalars are recorded on the trace (``scope.set_attr``) for the
// compile path; the gradient needs only the saved input.
//
// See Also
// --------
// :func:`nan_to_num_op`, :class:`ClipBackward`.
class LUCID_API NanToNumBackward : public UnaryOp<NanToNumBackward> {
public:
    static const OpSchema schema_v1;
    // Forward — replaces NaN / +inf / -inf and wires the backward node.
    //
    // Parameters
    // ----------
    // a : const TensorImplPtr&
    //     Input tensor.
    // nan_val, posinf_val, neginf_val : double
    //     Replacements for NaN, +inf and -inf.
    //
    // Returns
    // -------
    // TensorImplPtr
    //     Tensor of the input's shape, dtype and device.
    static TensorImplPtr
    forward(const TensorImplPtr& a, double nan_val, double posinf_val, double neginf_val);
    // $g \odot \mathbb{1}[x \text{ finite}]$ from the saved input.
    Storage grad_formula(const Storage& g);
    // Graph-mode equivalent — ``where(isfinite(x), g, 0)`` — so the
    // gradient itself stays differentiable.
    TensorImplPtr
    grad_formula_impl(const TensorImplPtr& g, const TensorImplPtr& x, const TensorImplPtr& out);
};

// Element-wise scalar-exponent power — returns $a^{\text{exp}}$ with a
// fully wired autograd node.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor (promoted to FP32 by the op schema).
// exp : double
//     Scalar exponent.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor with the same shape as ``a``; gradient is
//     $\text{exp} \cdot x^{\text{exp}-1} \cdot g$.
//
// Math
// ----
// $$y_i = x_i^{\text{exp}}, \qquad
// \frac{\partial y_i}{\partial x_i} = \text{exp} \cdot x_i^{\text{exp} - 1}$$
//
// Shape
// -----
// Output shape equals input shape (elementwise).
//
// See Also
// --------
// :func:`rpow_scalar_op`, :class:`PowScalarBackward`.
LUCID_API TensorImplPtr pow_scalar_op(const TensorImplPtr& a, double exp);

// Element-wise scalar-base reverse power — returns $\text{base}^{a}$ with
// a fully wired autograd node.
//
// Parameters
// ----------
// base : double
//     Scalar base (must be positive for a real-valued gradient).
// a : TensorImplPtr
//     Input tensor — the exponent (promoted to FP32 by the op schema).
//
// Returns
// -------
// TensorImplPtr
//     Output tensor with the same shape as ``a``; gradient is
//     $\ln(\text{base}) \cdot y \cdot g$.
//
// Math
// ----
// $$y_i = \text{base}^{x_i}, \qquad
// \frac{\partial y_i}{\partial x_i} = \ln(\text{base}) \cdot y_i$$
//
// Shape
// -----
// Output shape equals input shape (elementwise).
//
// Raises
// ------
// Gradient is undefined / non-real for ``base <= 0``; callers should
// guard the base if they intend to backpropagate.
//
// See Also
// --------
// :func:`pow_scalar_op`, :class:`RPowScalarBackward`.
LUCID_API TensorImplPtr rpow_scalar_op(double base, const TensorImplPtr& a);

// Element-wise clamp — returns ``a`` clipped to the range
// ``[min_v, max_v]`` with a fully wired autograd node.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor (integer or floating-point).
// min_v : double
//     Lower clamp bound.
// max_v : double
//     Upper clamp bound (must satisfy ``min_v <= max_v``).
//
// Returns
// -------
// TensorImplPtr
//     Clamped output tensor with the same shape, dtype, and device as
//     ``a``.  Gradient is ``g`` where ``min_v < a < max_v`` and zero
//     where ``a`` saturates either bound.
//
// Math
// ----
// $$y_i = \max\bigl(\text{min\_v},\ \min(\text{max\_v},\ x_i)\bigr)$$
//
// Shape
// -----
// Output shape equals input shape (elementwise).
//
// Notes
// -----
// Equivalent to reference framework's ``clamp`` (Lucid uses the name
// ``clip`` for the canonical op).  Saturated positions act as gradient
// stoppers — useful for hard-tanh-style activations.
//
// See Also
// --------
// :class:`ClipBackward`.
LUCID_API TensorImplPtr clip_op(const TensorImplPtr& a, double min_v, double max_v);

}  // namespace lucid
