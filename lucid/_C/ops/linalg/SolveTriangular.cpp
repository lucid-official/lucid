// lucid/_C/ops/linalg/SolveTriangular.cpp
//
// Implements solve_triangular_op via IBackend::linalg_solve_triangular()
// → BLAS strsm/dtrsm/ctrsm/ztrsm (the GPU backend delegates to the CPU
// backend), and its backward node.  The right-hand side is read by
// ``solve_rhs_contract``; the node is wired on the aligned operands and the
// broadcast / vector axis are differentiable views around it.

#include "SolveTriangular.h"

#include "../../autograd/ComplexGrad.h"
#include "../../backend/Dispatcher.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/Helpers.h"
#include "../../core/OpRegistry.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../../kernel/NaryKernel.h"
#include "../../ops/bfunc/Matmul.h"
#include "../../ops/ufunc/Arith.h"
#include "../../ops/ufunc/Transpose.h"
#include "../gfunc/Gfunc.h"
#include "../utils/Tri.h"
#include "_Detail.h"

namespace lucid {

const OpSchema SolveTriangularBackward::schema_v1{"solve_triangular", 1, AmpPolicy::KeepInput};

namespace {

TensorImplPtr solve_triangular_aligned(const TensorImplPtr& a,
                                       const TensorImplPtr& b,
                                       bool upper,
                                       bool unitriangular) {
    OpScopeFull scope{"solve_triangular", a->device(), a->dtype(), b->shape()};
    scope.set_attr("upper", upper);
    scope.set_attr("unitriangular", unitriangular);
    auto out_storage = backend::Dispatcher::for_device(a->device())
                           .linalg_solve_triangular(a->storage(), b->storage(), a->shape(),
                                                    b->shape(), upper, unitriangular, a->dtype());
    auto out = linalg_detail::fresh(std::move(out_storage), b->shape(), b->dtype(), b->device());
    auto bwd = std::make_shared<SolveTriangularBackward>();
    bwd->upper_ = upper;
    bwd->unitriangular_ = unitriangular;
    bwd->keep_output(out->storage());
    kernel::NaryKernel<SolveTriangularBackward, 2>::wire_autograd(std::move(bwd), {a, b}, out,
                                                                  true);
    return out;
}

// The solve takes a complex operand (BLAS ``ctrsm`` / ``ztrsm``).  Any other
// dtype the substitution cannot take keeps ``float_only``'s refusal.
Validator& float_or_complex(Validator& v, const TensorImplPtr& t) {
    return (t && is_complex(t->dtype())) ? v.non_null() : v.float_only();
}

// dB = A⁻ᵀ G and dA = Π(-dB Xᵀ), with Π the triangle the forward read.  On a
// complex dtype this is the real-dtype formula, and the callers conjugate
// around it (``autograd/ComplexGrad.h``), which makes it A⁻ᴴ G and -dB Xᴴ.
std::vector<TensorImplPtr> solve_triangular_adjoint(
    const TensorImplPtr& a, const TensorImplPtr& x, const TensorImplPtr& g, bool upper, bool unit) {
    auto dB = solve_triangular_aligned(mT_op(a), g, !upper, unit);
    auto dA = neg_op(matmul_op(dB, mT_op(x)));
    dA = upper ? triu_op(dA, unit ? 1 : 0) : tril_op(dA, unit ? -1 : 0);
    return {dA, dB};
}

}  // namespace

std::vector<Storage> SolveTriangularBackward::apply(Storage grad_out) {
    NoGradGuard ng;
    using ::lucid::helpers::fresh;
    auto A = fresh(Storage{saved_inputs_[0]}, input_shapes_[0], dtype_, device_);
    auto G =
        fresh(conj_if_complex(grad_out, out_shape_, dtype_, device_), out_shape_, dtype_, device_);
    auto X = fresh(Storage{saved_output_}, out_shape_, dtype_, device_);
    auto grads = solve_triangular_adjoint(A, X, G, upper_, unitriangular_);
    return {conj_if_complex(grads[0]->storage(), input_shapes_[0], dtype_, device_),
            conj_if_complex(grads[1]->storage(), input_shapes_[1], dtype_, device_)};
}

std::vector<TensorImplPtr> SolveTriangularBackward::apply_for_graph(const TensorImplPtr& grad_out) {
    const auto& a = saved_impl_inputs_[0];
    const auto& b = saved_impl_inputs_[1];
    if (!a || !b)
        ErrorBuilder("solve_triangular").fail("graph-mode backward is missing its saved inputs");
    // X recomputed rather than read from the saved output, so its own
    // dependence on A and B is part of the recorded graph.
    auto x = solve_triangular_aligned(a, b, upper_, unitriangular_);
    auto grads = solve_triangular_adjoint(a, x, conj_if_complex(grad_out), upper_, unitriangular_);
    return {conj_if_complex(grads[0]), conj_if_complex(grads[1])};
}

LUCID_REGISTER_OP(SolveTriangularBackward)

TensorImplPtr solve_triangular_op(const TensorImplPtr& a,
                                  const TensorImplPtr& b,
                                  bool upper,
                                  bool unitriangular) {
    using namespace linalg_detail;
    auto check_a = Validator::input(a, "solve_triangular.a");
    float_or_complex(check_a, a).square_2d();
    auto check_b = Validator::input(b, "solve_triangular.b");
    float_or_complex(check_b, b);
    if (a->dtype() != b->dtype())
        ErrorBuilder("solve_triangular").fail("A and b must have the same dtype");
    if (a->device() != b->device())
        ErrorBuilder("solve_triangular").fail("A and b must be on the same device");
    const SolveRhs rhs =
        solve_rhs_contract(a->shape(), b->shape(), VectorRhs::OneD, "solve_triangular");
    ErrorContextGuard ctx{"solve_triangular"};

    // An empty system has an empty solution; BLAS will not be the one to say
    // so — see ``empty_matrix``.
    if (shape_numel(rhs.a_shape) == 0 || shape_numel(rhs.b_shape) == 0)
        return zeros_op(rhs.out_shape, b->dtype(), b->device());

    return restore_rhs(
        solve_triangular_aligned(align_to(a, rhs.a_shape), align_rhs(b, rhs), upper, unitriangular),
        rhs);
}

}  // namespace lucid
