// lucid/_C/ops/bfunc/Div.h
//
// Declares DivBackward, the autograd node for element-wise tensor division,
// and the public free function div_op.

#pragma once

#include <utility>

#include "../../api.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_BinaryOp.h"

namespace lucid {

// Autograd node for element-wise division $y = a / b$ with NumPy
// broadcasting.
//
// Saves both forward inputs because the quotient rule for $\partial L /
// \partial b$ requires both $a$ and $b$.  Broadcast-reduces each branch back
// to the original input shape via :func:`sum_to_shape` inside
// :func:`BinaryKernel::apply`.
//
// Math
// ----
// $$
//   y = \frac{a}{b}, \qquad
//   \frac{\partial L}{\partial a} = \frac{1}{b} \cdot \frac{\partial L}{\partial y}, \qquad
//   \frac{\partial L}{\partial b} = -\frac{a}{b^2} \cdot \frac{\partial L}{\partial y}
// $$
//
// Shape
// -----
// Inputs follow NumPy broadcasting; output has the broadcasted shape.  Each
// gradient is then reduced to the original input's shape.
//
// Attributes
// ----------
// kSavesInputs : bool
//     ``true`` (inherited default from BinaryKernel) — both forward inputs
//     are retained for the quotient-rule backward pass.
// schema_v1 : OpSchema
//     ``"div"``, schema version 1, ``AmpPolicy::Promote``, deterministic.
//
// Notes
// -----
// The CPU quotient is correctly rounded at F32 and F64: ``vDSP_vdiv`` for
// F32, and a plain loop for F64 because ``vDSP_vdivD`` is not exact
// (backend/cpu/Vdsp.cpp).  GPU dispatch uses MLX's broadcast-aware ``divide`` primitive.  Division
// by zero follows IEEE-754 semantics (yields $\pm\infty$ or NaN); no
// explicit guard is inserted.
//
// See Also
// --------
// AddBackward, SubBackward, MulBackward
class LUCID_API DivBackward : public BinaryOp<DivBackward> {
public:
    // Op registration metadata.
    //
    // Attributes
    // ----------
    // name : const char*
    //     ``"div"``.
    // version : int
    //     ``1``.
    // amp_policy : AmpPolicy
    //     ``Promote``.
    // deterministic : bool
    //     ``true``.
    static const OpSchema schema_v1;

    // ``d/da = 1/b`` reads ``b`` alone; ``d/db = -a/b^2`` reads both.
    static constexpr GradReads<2> kGradReads{{{false, true}, {true, true}}};

    // Forward dispatch hook called by BinaryKernel::forward.
    //
    // Parameters
    // ----------
    // be : backend::IBackend&
    //     Active backend (CPU Accelerate or GPU MLX).
    // a, b : const Storage&
    //     Operands already broadcast-expanded to ``shape``.
    // shape : const Shape&
    //     Broadcasted output shape.
    // dt : Dtype
    //     Promoted output dtype.
    //
    // Returns
    // -------
    // Storage
    //     Output storage holding $a / b$.
    static Storage dispatch(
        backend::IBackend& be, const Storage& a, const Storage& b, const Shape& shape, Dtype dt) {
        return be.div(a, b, shape, dt);
    }

    // Compute the storage-level gradients for both operands (quotient rule).
    //
    // Parameters
    // ----------
    // grad_out : const Storage&
    //     Upstream gradient at the broadcasted output shape.
    //
    // Returns
    // -------
    // std::pair<Storage, Storage>
    //     ``(dA, dB)`` where ``dA = grad_out / b_broadcast`` and
    //     ``dB = -(grad_out * a_broadcast) / b_broadcast^2``.  Both still
    //     at output shape; :func:`BinaryKernel::apply` performs the
    //     broadcast reduction.
    //
    // Math
    // ----
    // $$
    //   \mathrm{dA} = \frac{1}{b} \odot \frac{\partial L}{\partial y}, \qquad
    //   \mathrm{dB} = -\frac{a}{b^2} \odot \frac{\partial L}{\partial y}
    // $$
    //
    // Notes
    // -----
    // ``dB`` is evaluated as ``-(grad_out * (a / b)) / b``: $b^2$ leaves float32
    // outside $|b| \approx 10^{-19} \ldots 10^{19}$, which would zero or NaN a
    // gradient of ordinary size.  The implementation materialises intermediates
    // (``quotient``, ``g_times_q``, ``div_by_b``) explicitly so each transient
    // storage can be released as soon as it is no longer needed.
    //
    // Warns
    // -----
    // Division by zero in $b$ propagates IEEE-754 $\pm\infty$ / NaN into the
    // gradient; downstream optimizers should handle this case.
    std::pair<Storage, Storage> grad_formula(const Storage& grad_out);

    // Graph-mode (TensorImpl) variant of the gradient.
    //
    // Parameters
    // ----------
    // grad_out : const TensorImplPtr&
    //     Upstream gradient node.
    // a, b : const TensorImplPtr&
    //     Forward inputs, already broadcast-expanded to ``out_shape_`` by
    //     BinaryKernel.
    //
    // Returns
    // -------
    // std::pair<TensorImplPtr, TensorImplPtr>
    //     ``(grad_out / b, -(grad_out * (a / b)) / b)``.
    std::pair<TensorImplPtr, TensorImplPtr> grad_formula_impl(const TensorImplPtr& grad_out,
                                                              const TensorImplPtr& a,
                                                              const TensorImplPtr& b);
};

// Public entry point — element-wise division with autograd support.
//
// Parameters
// ----------
// a, b : const TensorImplPtr&
//     Operands of any rank.  Shapes must be NumPy-broadcast-compatible.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor of shape ``broadcast_shapes(a.shape, b.shape)`` and
//     promoted dtype.  Participates in autograd via :class:`DivBackward`.
//
// Math
// ----
// $$ y = \frac{a}{b} $$
//
// Raises
// ------
// LucidError
//     If ``a`` and ``b`` have incompatible shapes under NumPy broadcasting
//     rules, or if their devices differ.
//
// Warns
// -----
// No explicit zero-divisor guard.  Division by zero yields IEEE-754
// $\pm\infty$ / NaN in both the forward output and gradient.
//
// Examples
// --------
// >>> auto c = div_op(a, b);    // c = a / b
//
// See Also
// --------
// add_op, sub_op, mul_op
LUCID_API TensorImplPtr div_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
