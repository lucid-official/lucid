// lucid/_C/ops/bfunc/Add.h
//
// Declares AddBackward, the autograd node for element-wise tensor addition, and
// the public free function add_op that serves as the engine entry point.

#pragma once

#include <utility>

#include "../../api.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_BinaryOp.h"

namespace lucid {

// Autograd node for element-wise addition $y = a + b$ with NumPy broadcasting.
//
// Inherits forward dispatch, broadcasting logic, and the apply() trampoline
// from BinaryOp<AddBackward> (alias for BinaryKernel<AddBackward>); only the
// backend hook and the gradient formula are defined locally.  No forward
// inputs are saved because the gradient is a pass-through: the upstream
// gradient is delivered unchanged to each operand (modulo broadcast
// reduction).
//
// Math
// ----
// $$
//   y = a + b, \qquad
//   \frac{\partial L}{\partial a} = \frac{\partial L}{\partial y}, \qquad
//   \frac{\partial L}{\partial b} = \frac{\partial L}{\partial y}
// $$
//
// When $a$ or $b$ was broadcast against the other operand, the corresponding
// branch is reduced back to the original input shape via :func:`sum_to_shape`
// inside :func:`BinaryKernel::apply`.
//
// Shape
// -----
// Inputs follow NumPy broadcasting; output has the broadcasted shape.  Each
// gradient is then reduced to the original input's shape.
//
// Attributes
// ----------
// kSavesInputs : bool
//     ``false`` — the additive gradient is independent of input values, so
//     no Storage is retained.
// schema_v1 : OpSchema
//     ``"add"``, schema version 1, ``AmpPolicy::Promote`` (operands are
//     up-cast to the wider dtype), deterministic.
//
// Notes
// -----
// CPU dispatch routes to Accelerate ``vDSP_vadd`` / ``vDSP_vaddD`` for F32 /
// F64.  GPU dispatch uses MLX's broadcast-aware ``add`` primitive.
//
// See Also
// --------
// SubBackward, MulBackward, DivBackward
class LUCID_API AddBackward : public BinaryOp<AddBackward> {
public:
    // Disables saving of forward input Storages — the addition gradient does
    // not require them, so retaining them would waste memory.
    static constexpr bool kSavesInputs = false;

    // Both local derivatives are 1, so the formula is its own conjugate and
    // a complex backward skips the two conjugations around it.
    static constexpr ComplexGrad kComplexGrad = ComplexGrad::AsWritten;

    // Op registration metadata.
    //
    // Attributes
    // ----------
    // name : const char*
    //     ``"add"``.
    // version : int
    //     ``1``.
    // amp_policy : AmpPolicy
    //     ``Promote`` — both inputs cast to the wider floating dtype.
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
    //     Newly allocated output storage holding $a + b$.
    static Storage dispatch(
        backend::IBackend& be, const Storage& a, const Storage& b, const Shape& shape, Dtype dt) {
        return be.add(a, b, shape, dt);
    }

    static Storage dispatch_scalar(
        backend::IBackend& be, const Storage& a, const Shape& shape, Dtype dt, double s) {
        return be.add_scalar(a, shape, dt, s);
    }

    // Compute the storage-level gradients for both operands.
    //
    // Parameters
    // ----------
    // grad_out : const Storage&
    //     Upstream gradient $\partial L / \partial y$ at the broadcasted
    //     output shape.
    //
    // Returns
    // -------
    // std::pair<Storage, Storage>
    //     ``(dA, dB)`` where both equal ``grad_out`` (still at output shape;
    //     :func:`BinaryKernel::apply` performs the broadcast reduction).
    //
    // Math
    // ----
    // $$
    //   \mathrm{dA} = \frac{\partial L}{\partial y}, \qquad
    //   \mathrm{dB} = \frac{\partial L}{\partial y}
    // $$
    std::pair<Storage, Storage> grad_formula(const Storage& grad_out);

    // Graph-mode (TensorImpl) variant of the gradient.
    //
    // Used when autograd is built as a dataflow graph rather than executed
    // eagerly on storages.  Returns the upstream gradient unchanged for both
    // branches — broadcast reduction is handled by the caller.
    //
    // Parameters
    // ----------
    // grad_out : const TensorImplPtr&
    //     Upstream gradient node.
    // a, b : const TensorImplPtr&
    //     Forward inputs (unused for addition; retained for signature
    //     uniformity across binary ops).
    //
    // Returns
    // -------
    // std::pair<TensorImplPtr, TensorImplPtr>
    //     ``(grad_out, grad_out)`` — identity in graph form.
    std::pair<TensorImplPtr, TensorImplPtr> grad_formula_impl(const TensorImplPtr& grad_out,
                                                              const TensorImplPtr& /*a*/,
                                                              const TensorImplPtr& /*b*/) {
        return {grad_out, grad_out};
    }
};

// Public entry point — element-wise addition with autograd support.
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
//     promoted dtype.  Participates in autograd via :class:`AddBackward`.
//
// Math
// ----
// $$ y = a + b $$
//
// Raises
// ------
// LucidError
//     If ``a`` and ``b`` have incompatible shapes under NumPy broadcasting
//     rules, or if their devices differ.
//
// Examples
// --------
// >>> auto c = add_op(a, b);    // c = a + b, same as a + b in Python
//
// See Also
// --------
// sub_op, mul_op, div_op
LUCID_API TensorImplPtr add_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
