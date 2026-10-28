// lucid/_C/ops/linalg/Eigh.cpp
//
// Implementation of the symmetric eigendecomposition forward op.
//
// Forward:
//   CPU path: LAPACK ssyev (F32) or dsyev (F64) computes all eigenvalues and
//             optionally eigenvectors of a real symmetric matrix.  The routine
//             reads only the lower triangle of A (UPLO='L') and overwrites the
//             input buffer with the eigenvectors; eigenvalues are returned in
//             a separate vector in ascending order.
//   GPU path: mlx::core::linalg::eigh() on the CPU stream.
//
// Mirrors eig_op in structure but calls linalg_eigh() which exploits symmetry
// for about 2x better performance compared to the general dgeev path.
// Shape derivation and output wrapping follow the same pattern as Eig.cpp.
//
// The symmetry precondition is not checked at runtime; callers must ensure A
// is symmetric (or only fill the lower triangle).  Passing a non-symmetric A
// produces numerically meaningless results without an error.
//
// Autograd is not wired here.  The eigenvector gradient must handle sign
// ambiguity (each eigenvector can be negated without changing A V = V diag(w))
// and near-degenerate eigenvalue pairs; this is deferred to a future phase.

#include "Eigh.h"

#include <variant>
#include <vector>

#include "../../backend/Dispatcher.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../gfunc/Gfunc.h"
#include "_Detail.h"

namespace lucid {

// Compute eigenvalues and eigenvectors of symmetric matrix a.
//
// wsh drops the last dimension (n scalars per n×n matrix).
// vsh is the same as the input shape (the full n×n orthogonal eigenvector matrix).
// The backend guarantees eigenvalues in wsh are sorted ascending.
std::vector<TensorImplPtr> eigh_op(const TensorImplPtr& a_in, const std::string& uplo) {
    using namespace linalg_detail;
    Validator::input(a_in, "eigh.a").non_null();
    if (uplo != "L" && uplo != "U")
        ErrorBuilder("eigh").invalid_argument("UPLO must be 'L' or 'U', got " + uplo);
    require_square_matrix(a_in->shape(), "eigh.a");
    const AmpScope amp{"eigh", a_in->device()};
    const TensorImplPtr a = amp.cast(a_in);
    require_float(a->dtype(), "eigh");
    OpScopeFull scope{"eigh", a->device(), a->dtype(), a->shape()};
    scope.set_attr("upper", uplo == "U");

    const auto& sh = a->shape();
    // Eigenvalue shape: one value per row/column, so remove the last dim.
    Shape wsh(sh.begin(), sh.end() - 1);
    Shape vsh = sh;

    // A degenerate matrix has a decomposition; LAPACK just will not be
    // the one to compute it.  See ``empty_matrix`` for why dispatching
    // wrote to a stream Lucid does not own and then raised.
    if (empty_matrix(sh))
        return {zeros_op(wsh, a->dtype(), a->device()), zeros_op(vsh, a->dtype(), a->device())};

    auto out = backend::Dispatcher::for_device(a->device())
                   .linalg_eigh(a->storage(), sh, wsh, vsh, a->dtype(), uplo == "U");
    auto res_w = fresh(std::move(out.first), wsh, a->dtype(), a->device());
    auto res_v = fresh(std::move(out.second), vsh, a->dtype(), a->device());
    if (auto* trc = compile::current_tracer()) {
        trc->on_op_io({a}, res_w);
        trc->on_op_io({}, res_v);
    }
    return {res_w, res_v};
}

}  // namespace lucid
