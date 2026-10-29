// lucid/_C/ops/bfunc/Sub.h
//
// Declares SubBackward, the autograd node for element-wise tensor subtraction,
// and the public free function sub_op.

#pragma once

#include <utility>

#include "../../api.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_BinaryOp.h"

namespace lucid {

// Autograd node for element-wise subtraction $y = a - b$ with NumPy
// broadcasting.
//
// Inherits forward dispatch, broadcasting logic, and the apply() trampoline
// from BinaryOp<SubBackward>.  No forward inputs are saved because the
// gradient depends only on the upstream gradient: $a$ receives it unchanged
// and $b$ receives it negated.  Broadcast-reduces each branch back to the
// original input shape via :func:`sum_to_shape` inside
// :func:`BinaryKernel::apply`.
//
// Math
// ----
// $$
//   y = a - b, \qquad
//   \frac{\partial L}{\partial a} = \frac{\partial L}{\partial y}, \qquad
//   \frac{\partial L}{\partial b} = -\frac{\partial L}{\partial y}
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
//     ``false`` — gradient is value-independent.
// schema_v1 : OpSchema
//     ``"sub"``, schema version 1, ``AmpPolicy::Promote``, deterministic.
//
// Notes
// -----
// CPU dispatch uses Accelerate ``vDSP_vsub`` / ``vDSP_vsubD`` for F32 / F64.
// GPU dispatch uses MLX's broadcast-aware ``subtract`` primitive.
//
// See Also
// --------
// AddBackward, MulBackward, DivBackward
class LUCID_API SubBackward : public BinaryOp<SubBackward> {
public:
    // Disables saving of forward input Storages — the subtractive gradient is
    // independent of input values.
    static constexpr bool kSavesInputs = false;

    // The local derivatives are +1 and -1, so the formula is its own
    // conjugate and a complex backward skips the two conjugations around it.
    static constexpr ComplexGrad kComplexGrad = ComplexGrad::AsWritten;

    // Op registration metadata.
    //
    // Attributes
    // ----------
    // name : const char*
    //     ``"sub"``.
    // version : int
    //     ``1``.
    // amp_policy : AmpPolicy
    //     ``Promote``.
    // deterministic : bool
    //     ``true``.
    static const OpSchema schema_v1;

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
    //     Output storage holding $a - b$.
    static Storage dispatch(
        backend::IBackend& be, const Storage& a, const Storage& b, const Shape& shape, Dtype dt) {
        return be.sub(a, b, shape, dt);
    }

    // Compute the storage-level gradients for both operands.
    //
    // Parameters
    // ----------
    // grad_out : const Storage&
    //     Upstream gradient at the broadcasted output shape.
    //
    // Returns
    // -------
    // std::pair<Storage, Storage>
    //     ``(dA, dB)`` where ``dA`` is a clone of ``grad_out`` and ``dB`` is
    //     its element-wise negation.
    //
    // Math
    // ----
    // $$
    //   \mathrm{dA} = \frac{\partial L}{\partial y}, \qquad
    //   \mathrm{dB} = -\frac{\partial L}{\partial y}
    // $$
    std::pair<Storage, Storage> grad_formula(const Storage& grad_out);

    // Graph-mode (TensorImpl) variant of the gradient.
    //
    // Parameters
    // ----------
    // grad_out : const TensorImplPtr&
    //     Upstream gradient node.
    // a, b : const TensorImplPtr&
    //     Forward inputs (unused — values irrelevant to subtraction gradient).
    //
    // Returns
    // -------
    // std::pair<TensorImplPtr, TensorImplPtr>
    //     ``(grad_out, -grad_out)``.
    std::pair<TensorImplPtr, TensorImplPtr> grad_formula_impl(const TensorImplPtr& grad_out,
                                                              const TensorImplPtr& /*a*/,
                                                              const TensorImplPtr& /*b*/);
};

// Public entry point — element-wise subtraction with autograd support.
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
//     promoted dtype.  Participates in autograd via :class:`SubBackward`.
//
// Math
// ----
// $$ y = a - b $$
//
// Raises
// ------
// LucidError
//     If ``a`` and ``b`` have incompatible shapes under NumPy broadcasting
//     rules, or if their devices differ.
//
// Examples
// --------
// >>> auto c = sub_op(a, b);    // c = a - b
//
// See Also
// --------
// add_op, mul_op, div_op
LUCID_API TensorImplPtr sub_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
