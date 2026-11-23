// lucid/_C/ops/linalg/Eig.cpp
//
// Implementation of the general eigendecomposition forward op.
//
// Forward:
//   CPU path: LAPACK dgeev computes both the right eigenvectors and the
//             eigenvalues of a general (non-symmetric) real matrix.  For
//             matrices with complex eigenvalues dgeev returns the real and
//             imaginary parts in separate vectors; the current binding returns
//             only the real part, which is correct only for real-symmetric A.
//   GPU path: mlx::core::linalg::eig() on the CPU stream.
//
// Shapes for the two outputs are derived before dispatch:
//   w (eigenvalues):  (..., n)    — one eigenvalue per column of A
//   V (eigenvectors): (..., n, n) — same shape as A; columns are eigenvectors
//
// Autograd is not wired here.  The general eigenvector gradient is numerically
// sensitive near degenerate eigenvalues and requires complex arithmetic when
// eigenvalues are not real; this is deferred to a future phase.

#include "Eig.h"

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

// Compute eigenvalues and right eigenvectors of a.
//
// wsh drops the last dimension (the redundant n in n×n).
// vsh is identical to the input shape (the full n×n eigenvector matrix).
std::vector<TensorImplPtr> eig_op(const TensorImplPtr& a_in) {
    using namespace linalg_detail;
    Validator::input(a_in, "eig.a").non_null();
    require_square_matrix(a_in->shape(), "eig.a");
    const AmpScope amp{"eig", a_in->device()};
    const TensorImplPtr a = amp.cast(a_in);
    linalg_detail::require_solver_dtype(a->dtype(), "eig");
    OpScopeFull scope{"eig", a->device(), a->dtype(), a->shape()};

    const auto& sh = a->shape();
    // Eigenvalue shape: remove the last dim (one eigenvalue per row/column).
    Shape wsh(sh.begin(), sh.end() - 1);
    Shape vsh = sh;
    // A degenerate matrix has a decomposition; LAPACK just will not be
    // the one to compute it.  See ``empty_matrix`` for why dispatching
    // wrote to a stream Lucid does not own and then raised.
    // A real matrix's eigenvalues are complex; only the ones whose
    // characteristic polynomial happens to split over R are not.  The
    // output dtype follows from the input's lane width rather than being
    // the input's dtype — see ``unpack_eig`` in the CPU backend.
    const Dtype out_dt = complex_for(a->dtype());

    if (empty_matrix(sh))
        return {zeros_op(wsh, out_dt, a->device()), zeros_op(vsh, out_dt, a->device())};

    auto out = backend::Dispatcher::for_device(a->device())
                   .linalg_eig(a->storage(), sh, wsh, vsh, a->dtype());
    std::vector<TensorImplPtr> result;
    result.push_back(fresh(std::move(out.first), wsh, out_dt, a->device()));
    result.push_back(fresh(std::move(out.second), vsh, out_dt, a->device()));
    return result;
}

}  // namespace lucid
