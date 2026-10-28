// lucid/_C/ops/linalg/Pinv.cpp
//
// Implementation of the Moore-Penrose pseudoinverse forward op.
//
// The transposed output shape (..., n, m) is computed here and passed to
// fresh() so the TensorImpl carries the correct shape metadata.
//
// Forward:
//   CPU path: the backend calls LAPACK dgesdd to compute a full SVD
//             A = U S Vᵀ, inverts the non-negligible singular values to form
//             S⁺, and then computes A⁺ = V S⁺ Uᵀ.
//   GPU path: delegates to the CPU backend so gesdd owns the SVD policy.
//
// The rcond threshold (below which singular values are treated as zero) is
// chosen by the backend and typically follows the NumPy convention:
//   rcond = max(m, n) * max(S) * machine_epsilon
//
// Autograd is not wired here.  A future backward would use the pseudoinverse
// differential identity:
//   dA⁺ = -A⁺ (dA) A⁺ + A⁺ A⁺ᵀ (dAᵀ)(I - A A⁺) + (I - A⁺ A)(dAᵀ) A⁺ᵀ A⁺
// which reduces to the ordinary inverse gradient when A is square and full rank.

#include "Pinv.h"

#include <variant>

#include "../../backend/Dispatcher.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../gfunc/Gfunc.h"
#include "_Detail.h"

namespace lucid {

// Compute A⁺ (the Moore-Penrose pseudoinverse) of a.
//
// The output shape swaps the last two dimensions of the input because the
// pseudoinverse of an (m×n) matrix is (n×m).
TensorImplPtr pinv_op(const TensorImplPtr& a) {
    using namespace linalg_detail;
    Validator::input(a, "pinv.a").non_null();
    require_float(a->dtype(), "pinv");
    if (a->shape().size() < 2)
        ErrorBuilder("pinv").invalid_argument("input must be at least 2-D");
    OpScopeFull scope{"pinv", a->device(), a->dtype(), a->shape()};

    const auto& sh = a->shape();
    const int m = static_cast<int>(sh[sh.size() - 2]);
    const int n = static_cast<int>(sh[sh.size() - 1]);
    // The pseudoinverse transposes the matrix dims: output is (..., n, m).
    Shape out_shape(sh.begin(), sh.end() - 2);
    out_shape.push_back(n);
    out_shape.push_back(m);

    // A degenerate matrix has an answer; LAPACK just will not be the one
    // to compute it — see ``empty_matrix``.
    if (empty_matrix(sh))
        return zeros_op(out_shape, a->dtype(), a->device());

    Storage out =
        backend::Dispatcher::for_device(a->device()).linalg_pinv(a->storage(), sh, a->dtype());
    auto res = fresh(std::move(out), std::move(out_shape), a->dtype(), a->device());
    if (auto* trc = compile::current_tracer())
        trc->on_op_io({a}, res);
    return res;
}

}  // namespace lucid
