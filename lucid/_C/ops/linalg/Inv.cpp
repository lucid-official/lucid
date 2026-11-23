// lucid/_C/ops/linalg/Inv.cpp
//
// Implementation of the matrix inverse op and its autograd backward node.
//
// Forward: dispatches to IBackend::linalg_inv() via Dispatcher.
//   CPU path: LAPACK dgetrf (LU factorisation) + dgetri (explicit inversion).
//   GPU path: mlx::core::linalg::inv() on the CPU stream (see _Detail.h).
//
// Backward: implements ∂L/∂A = -Bᵀ G Bᵀ where B = A⁻¹ and G = ∂L/∂B.
// The formula is composed from matmul_op (which handles batched matrix
// multiply) and neg_op, so the entire backward sub-graph is itself
// differentiable, enabling second-order gradients for free.
//
// Memory note: saved_output_ retains a reference-counted copy of the forward
// output storage.  In a long forward pass this means A⁻¹ is kept alive until
// the backward is executed and the grad function is destroyed.  For large
// matrices in deep graphs, users may prefer to recompute A⁻¹ in backward
// (not currently supported here) to reduce peak memory.

#include "Inv.h"

#include <variant>

#include "../../autograd/FuncOp.h"
#include "../../backend/Dispatcher.h"
#include "../../compile/Tracer.h"
#include "../../core/GradMode.h"
#include "../../core/Helpers.h"
#include "../../core/OpRegistry.h"
#include "../../core/SchemaGuard.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../../kernel/NaryKernel.h"
#include "../../ops/bfunc/Matmul.h"
#include "../../ops/ufunc/Arith.h"
#include "../../ops/ufunc/Transpose.h"
#include "_Detail.h"

namespace lucid {

// schema_v1 associates this backward node with the op name "inv" for
// serialisation and the OpRegistry, and sets AmpPolicy::KeepInput so that
// mixed-precision training does not silently downcast the saved inverse.
const OpSchema InvBackward::schema_v1 =
    OpSchema{"inv", 1, AmpPolicy::KeepInput}.refusing(kSignedIntegers, DtypeRefusal::Guard);

// Backward pass for inv_op.
//
// B = A⁻¹ was saved as saved_output_ during the forward pass.
// G = grad_out = ∂L/∂B.
//
// Derivation:
//   Differentiate A A⁻¹ = I with respect to A:
//     (dA) A⁻¹ + A d(A⁻¹) = 0  =>  d(A⁻¹) = -A⁻¹ (dA) A⁻¹
//   Contracting with G:
//     ∂L/∂A = tr(Gᵀ d(A⁻¹)) = -tr(Gᵀ B dA B)
//           = -(Bᵀ G Bᵀ)   [using tr(Aᵀ B) = sum(A ⊙ B) rearrangement]
//
// NoGradGuard prevents the backward sub-graph from being tracked, which
// would otherwise accumulate unnecessary nodes for a second-order pass.
// (Second-order gradients still work because NoGradGuard is scoped to the
// backward execution; the ops themselves remain differentiable.)
std::vector<Storage> InvBackward::apply(Storage grad_out) {
    NoGradGuard ng;
    using ::lucid::helpers::fresh;

    // Reconstruct B (A⁻¹) and the upstream gradient G as tracked tensors.
    auto B = fresh(Storage{saved_output_}, out_shape_, dtype_, device_);
    auto dB = fresh(std::move(grad_out), out_shape_, dtype_, device_);

    // Bᵀ is the matrix transpose of B; mT_op performs the last-two-axes swap.
    auto Bt = mT_op(B);
    // Compute -Bᵀ G Bᵀ via two successive matrix multiplies, then negate.
    auto dA = neg_op(matmul_op(matmul_op(Bt, dB), Bt));
    return {dA->storage()};
}

std::vector<TensorImplPtr> InvBackward::apply_for_graph(const TensorImplPtr& grad_out) {
    const auto& a = saved_impl_inputs_[0];
    if (!a)
        ErrorBuilder("inv").fail("graph-mode backward is missing its saved input");
    auto Bt = mT_op(inv_op(a));
    return {neg_op(matmul_op(matmul_op(Bt, grad_out), Bt))};
}

// Register InvBackward with the global op registry so that the autograd
// engine can look it up by schema name when replaying a saved graph.
LUCID_REGISTER_OP(InvBackward)

// Compute A⁻¹.
//
// Validates inputs (must be square, at least 2-D, float-typed), calls the
// backend, wraps the result in a TensorImpl, and wires InvBackward.
// The wire_autograd call passes save_inputs=false because the backward only
// needs B = A⁻¹ (saved_output_), not the original input A.
TensorImplPtr inv_op(const TensorImplPtr& a_in) {
    Validator::input(a_in, "inv.a").non_null().square_2d();
    SchemaGuard sg{InvBackward::schema_v1, a_in->dtype(), a_in->device()};
    const TensorImplPtr a = astype_op(a_in, sg.effective_dtype());
    Validator::input(a, "inv.a").float_only();
    OpScopeFull scope{"inv", a->device(), a->dtype(), a->shape()};

    Storage out_storage = backend::Dispatcher::for_device(a->device())
                              .linalg_inv(a->storage(), a->shape(), a->dtype());
    auto out = linalg_detail::fresh(std::move(out_storage), a->shape(), a->dtype(), a->device());
    if (auto* trc = ::lucid::compile::current_tracer()) {
        trc->on_op_io({a}, out);
    }
    auto bwd = std::make_shared<InvBackward>();
    // Save the forward result A⁻¹; the backward uses it in place of recomputing.
    bwd->keep_output(out->storage());
    // save_inputs=false: InvBackward does not need the original A.
    kernel::NaryKernel<InvBackward, 1>::wire_autograd(std::move(bwd), {a}, out, false);
    return out;
}

}  // namespace lucid
