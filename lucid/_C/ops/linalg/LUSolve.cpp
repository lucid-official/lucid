// lucid/_C/ops/linalg/LUSolve.cpp
//
// lu_solve_op and its backward node.  The right-hand side is read by
// ``solve_rhs_contract``; the factor, its pivots and the right-hand side are
// broadcast to one batch before LAPACK sees them, and the node is wired on
// those aligned operands.
#include "LUSolve.h"

#include "../../backend/Dispatcher.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/Helpers.h"
#include "../../core/OpRegistry.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../../kernel/NaryKernel.h"
#include "../../ops/bfunc/Add.h"
#include "../../ops/bfunc/Matmul.h"
#include "../../ops/ufunc/Arith.h"
#include "../../ops/ufunc/Transpose.h"
#include "../gfunc/Gfunc.h"
#include "../utils/Tri.h"
#include "SolveTriangular.h"
#include "_Detail.h"
namespace lucid {

const OpSchema LUSolveBackward::schema_v1{"lu_solve", 1, AmpPolicy::KeepInput};

namespace {

TensorImplPtr
lu_solve_aligned(const TensorImplPtr& LU, const TensorImplPtr& pivots, const TensorImplPtr& b) {
    auto result = backend::Dispatcher::for_device(LU->device())
                      .linalg_lu_solve(LU->storage(), pivots->storage(), b->storage(), LU->shape(),
                                       b->shape(), LU->dtype());
    auto out = linalg_detail::fresh(std::move(result), b->shape(), LU->dtype(), LU->device());
    if (GradMode::is_enabled() && (LU->requires_grad() || b->requires_grad()))
        pivots->check_not_inference_saved("lu_solve");
    auto bwd = std::make_shared<LUSolveBackward>();
    bwd->pivots_ = pivots;
    bwd->pivots_version_ = pivots->version();
    bwd->keep_output(out->storage());
    kernel::NaryKernel<LUSolveBackward, 2>::wire_autograd(std::move(bwd), {LU, b}, out, true);
    return out;
}

// The dense permutation P of ``A = P L U``, read off the pivots by solving
// against an identity factor: getrs with L = U = I returns Pᵀ B, so B = I
// gives Pᵀ.  It stays on the device and needs no host read of the pivots.
TensorImplPtr permutation_of(const TensorImplPtr& LU, const TensorImplPtr& pivots) {
    NoGradGuard ng;
    const auto& sh = LU->shape();
    const std::int64_t n = sh.back();
    auto eye = linalg_detail::align_to(eye_op(n, n, 0, LU->dtype(), LU->device()), sh);
    return mT_op(lu_solve_aligned(eye, pivots, eye));
}

// dB = A⁻ᵀ G and dLU = triu(R) + tril(L⁻ᵀ R Uᵀ, -1) with R = -U⁻ᵀ G Xᵀ.
//
// The triangular solves read LUᵀ: its lower triangle is Uᵀ, its upper
// triangle with a unit diagonal is Lᵀ — the same packing LAPACK uses.
std::vector<TensorImplPtr> lu_solve_adjoint(const TensorImplPtr& LU,
                                            const TensorImplPtr& pivots,
                                            const TensorImplPtr& x,
                                            const TensorImplPtr& g) {
    auto LUt = mT_op(LU);
    auto Y = solve_triangular_op(LUt, g, /*upper=*/false, /*unitriangular=*/false);
    auto dB = matmul_op(permutation_of(LU, pivots),
                        solve_triangular_op(LUt, Y, /*upper=*/true, /*unitriangular=*/true));
    auto R = neg_op(matmul_op(Y, mT_op(x)));
    auto dL = tril_op(solve_triangular_op(LUt, matmul_op(R, mT_op(triu_op(LU, 0))),
                                          /*upper=*/true, /*unitriangular=*/true),
                      -1);
    return {add_op(triu_op(R, 0), dL), dB};
}

}  // namespace

std::vector<Storage> LUSolveBackward::apply(Storage grad_out) {
    NoGradGuard ng;
    using ::lucid::helpers::fresh;
    auto LU = fresh(Storage{saved_inputs_[0]}, input_shapes_[0], dtype_, device_);
    auto G = fresh(std::move(grad_out), out_shape_, dtype_, device_);
    auto X = fresh(Storage{saved_output_}, out_shape_, dtype_, device_);
    auto grads = lu_solve_adjoint(LU, pivots_, X, G);
    return {grads[0]->storage(), grads[1]->storage()};
}

std::vector<TensorImplPtr> LUSolveBackward::apply_for_graph(const TensorImplPtr& grad_out) {
    const auto& LU = saved_impl_inputs_[0];
    const auto& b = saved_impl_inputs_[1];
    if (!LU || !b || !pivots_)
        ErrorBuilder("lu_solve").fail("graph-mode backward is missing its saved inputs");
    // X recomputed so its own dependence on LU and B is recorded.
    auto x = lu_solve_aligned(LU, pivots_, b);
    return lu_solve_adjoint(LU, pivots_, x, grad_out);
}

void LUSolveBackward::validate_versions() {
    FuncOp<LUSolveBackward, 2>::validate_versions();
    check_version_match(pivots_, pivots_version_, schema_v1.name, 1);
}

LUCID_REGISTER_OP(LUSolveBackward)

TensorImplPtr
lu_solve_op(const TensorImplPtr& LU, const TensorImplPtr& pivots, const TensorImplPtr& b) {
    using namespace linalg_detail;
    Validator::input(LU, "lu_solve.LU").float_only().non_null();
    // LAPACK's ``ipiv`` is ``const int*``, so the buffer is read 32 bits at
    // a time whatever dtype it arrived as.  Only checking for null let
    // three separate failures through: an int8 or bool pivot vector was
    // read past its own allocation and took the process down with SIGBUS,
    // an int16 one produced 1e+133 instead of a solution, and an int64 one
    // returned a different — silently wrong — answer.  Only I32 is the
    // width this reads, so only I32 may be passed.
    Validator::input(pivots, "lu_solve.pivots").non_null().dtype_eq(Dtype::I32);
    Validator::input(b, "lu_solve.b").float_only().non_null();
    Validator::pair(LU, b, "lu_solve").same_dtype().same_device();
    Validator::pair(LU, pivots, "lu_solve").same_device();

    // The factor has to be square.  ``?getrs`` solves ``A X = B`` from an
    // LU of A, and only a square A has a solve; ``lu_factor`` accepts any
    // shape, so a rectangular factor can now reach this call.  It used to
    // be handed to LAPACK anyway, which read ``n`` rows out of a matrix
    // that had fewer and answered ``[nan, nan, -inf]``.
    const auto& lu_sh = LU->shape();
    require_square_matrix(lu_sh, "lu_solve.LU (only a square system has a solution)");
    // One pivot per row of each factor: getrs reads ``n`` of them per
    // matrix, so a shorter (or differently batched) vector is read past.
    const Shape piv_expected(lu_sh.begin(), lu_sh.end() - 1);
    if (pivots->shape() != piv_expected)
        throw ShapeMismatch(piv_expected, pivots->shape(),
                            "lu_solve: pivots must have shape LU.shape[:-1]");

    const SolveRhs rhs = solve_rhs_contract(lu_sh, b->shape(), VectorRhs::OneD, "lu_solve");
    OpScopeFull scope{"lu_solve", LU->device(), LU->dtype(), rhs.a_shape};

    // An empty system has an empty solution; LAPACK will not be the one to
    // say so — see ``empty_matrix``.
    if (shape_numel(rhs.a_shape) == 0 || shape_numel(rhs.b_shape) == 0)
        return zeros_op(rhs.out_shape, LU->dtype(), LU->device());

    const Shape piv_shape(rhs.a_shape.begin(), rhs.a_shape.end() - 1);
    return restore_rhs(
        lu_solve_aligned(align_to(LU, rhs.a_shape), align_to(pivots, piv_shape), align_rhs(b, rhs)),
        rhs);
}

}  // namespace lucid
