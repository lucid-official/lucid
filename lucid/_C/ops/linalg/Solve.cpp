// lucid/_C/ops/linalg/Solve.cpp
//
// Implementation of the linear system solve op and its autograd backward node.
//
// Forward: dispatches to IBackend::linalg_solve() via Dispatcher.
//   CPU path: LAPACK dgesv performs LU factorisation of A (in-place) followed
//             by the forward and backward substitution steps to solve AX = B.
//   GPU path: GpuBackend::linalg_solve (LU inverse on the MLX CPU stream).
//
// Shape: ``solve_rhs_contract`` (``_Detail.h``) reads B, broadcasts the
// batch and refuses any other shape before the backend is reached.  The
// backward node is wired on the aligned ``(batch, n, n)`` / ``(batch, n, k)``
// operands; the broadcast and the vector axis are differentiable views around
// it, so their gradients are reduced by their own nodes.
//
// Backward:
//   Given upstream gradient G = ∂L/∂X:
//     ∂L/∂B = solve(Aᵀ, G)
//     ∂L/∂A = -(∂L/∂B) Xᵀ
//   Both calls are composed from existing ops so that second-order gradients
//   flow automatically.
//
// Note on the backward solve call: the transpose solve solve(Aᵀ, G) does
// not reuse the LU factors from the forward pass; instead it factors Aᵀ
// independently.  A future optimisation could pass the LU factors through
// as a saved tensor to halve the backward factorisation cost.

#include "Solve.h"

#include <variant>

#include "../../backend/Dispatcher.h"
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
#include "../gfunc/Gfunc.h"
#include "_Detail.h"

namespace lucid {

// schema_v1: the OpSchema tag "solve" with one saved input slot.  The template
// parameter 2 on FuncOp means two gradient outputs (one per operand), but the
// schema input count of 1 refers to the number of saved inputs in the registry
// sense.  AmpPolicy::KeepInput prevents lossy dtype promotion before the solve.
const OpSchema SolveBackward::schema_v1 =
    OpSchema{"solve", 1, AmpPolicy::KeepInput}.refusing(kSignedIntegers, DtypeRefusal::Guard);

namespace {

// The factor-and-solve on operands ``solve_rhs_contract`` already aligned:
// ``a`` is ``(batch, n, n)`` and ``b`` is ``(batch, n, k)`` with the same
// batch, so the backward node only ever sees matrix right-hand sides and
// needs no broadcast bookkeeping of its own.
TensorImplPtr solve_aligned(const TensorImplPtr& a, const TensorImplPtr& b) {
    Storage out_storage =
        backend::Dispatcher::for_device(a->device())
            .linalg_solve(a->storage(), b->storage(), a->shape(), b->shape(), a->dtype());
    auto out = linalg_detail::fresh(std::move(out_storage), b->shape(), a->dtype(), a->device());
    auto bwd = std::make_shared<SolveBackward>();
    bwd->keep_output(out->storage());
    kernel::NaryKernel<SolveBackward, 2>::wire_autograd(std::move(bwd), {a, b}, out, true);
    return out;
}

}  // namespace

// dB = solve(Aᵀ, G) and dA = -dB Xᵀ, on the aligned operands.
//
// Differentiating A X = B: A dX = dB - dA X, so for an upstream G the
// adjoint is dB = A⁻ᵀ G and dA = -dB Xᵀ.  Gradients are returned in the
// input order [A, B].
std::vector<Storage> SolveBackward::apply(Storage grad_out) {
    NoGradGuard ng;
    using ::lucid::helpers::fresh;
    auto A = fresh(Storage{saved_inputs_[0]}, input_shapes_[0], dtype_, device_);
    auto dX = fresh(std::move(grad_out), out_shape_, dtype_, device_);
    auto X = fresh(Storage{saved_output_}, out_shape_, dtype_, device_);
    auto dB = solve_aligned(mT_op(A), dX);
    auto dA = neg_op(matmul_op(dB, mT_op(X)));
    return {dA->storage(), dB->storage()};
}

std::vector<TensorImplPtr> SolveBackward::apply_for_graph(const TensorImplPtr& grad_out) {
    const auto& a = saved_impl_inputs_[0];
    const auto& b = saved_impl_inputs_[1];
    if (!a || !b)
        ErrorBuilder("solve").fail("graph-mode backward is missing its saved inputs");
    auto dB = solve_aligned(mT_op(a), grad_out);
    auto dA = neg_op(matmul_op(dB, mT_op(solve_aligned(a, b))));
    return {dA, dB};
}

LUCID_REGISTER_OP(SolveBackward)

TensorImplPtr solve_op(const TensorImplPtr& a_in, const TensorImplPtr& b_in) {
    using namespace linalg_detail;
    Validator::input(a_in, "solve.a").non_null().square_2d();
    Validator::input(b_in, "solve.b").non_null();
    Validator::pair(a_in, b_in, "solve").same_device();
    SchemaGuard sg{SolveBackward::schema_v1, a_in->dtype(), a_in->device()};
    check_schema_dtype(SolveBackward::schema_v1, b_in->dtype());
    const Dtype eff_dt = sg.effective_dtype();
    const TensorImplPtr a = astype_op(a_in, eff_dt);
    const TensorImplPtr b = astype_op(b_in, eff_dt);
    Validator::input(a, "solve.a").float_only();
    Validator::pair(a, b, "solve").same_dtype();
    const SolveRhs rhs = solve_rhs_contract(a->shape(), b->shape(), VectorRhs::OrBatched, "solve");
    OpScopeFull scope{"solve", a->device(), a->dtype(), rhs.a_shape};

    // An empty system has an empty solution; LAPACK will not be the one to
    // say so — see ``empty_matrix``.
    if (shape_numel(rhs.a_shape) == 0 || shape_numel(rhs.b_shape) == 0)
        return zeros_op(rhs.out_shape, a->dtype(), a->device());

    return restore_rhs(solve_aligned(align_to(a, rhs.a_shape), align_rhs(b, rhs)), rhs);
}

}  // namespace lucid
