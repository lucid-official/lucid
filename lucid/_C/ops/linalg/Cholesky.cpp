// lucid/_C/ops/linalg/Cholesky.cpp
//
// Implementation of the Cholesky decomposition forward op.
//
// Forward:
//   CPU path: LAPACK dpotrf reads only the lower (or upper) triangle of A,
//             writes the Cholesky factor in-place, and zeros the other half.
//             If LAPACK returns info > 0, the matrix is not positive-definite;
//             the backend raises LinAlgError (backend/LapackInfo.h).
//   GPU path: mlx::core::linalg::cholesky() on the CPU stream, its failed
//             pivot read off the factor's diagonal — the same LinAlgError.
//
// There is no backward node registered here; gradients through Cholesky are
// not yet supported.  Implementing the backward requires Iain Murray's formula
// (2016, "Differentiation of the Cholesky decomposition"):
//
//   Given G = ∂L/∂L (upstream gradient on L), compute ∂L/∂A:
//     S = L⁻ᵀ Phi(Lᵀ G) L⁻¹
//     ∂L/∂A = (S + Sᵀ) / 2
//   where Phi(X) zeroes the strictly upper-triangular part of X.
//
// This requires two triangular solves (L⁻ᵀ and L⁻¹) which are not yet
// exposed through the backend interface.
//
// Note on the output triangle: the backend fills only the requested triangle
// (lower when upper=false, upper when upper=true) and zeros the opposite half.
// Callers must not assume the discarded triangle is consistent with L or U;
// it is set to zero as a convention, not a mathematical requirement.

#include "Cholesky.h"

#include <variant>

#include "../../backend/Dispatcher.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../gfunc/Gfunc.h"
#include "_Detail.h"

namespace lucid {

// Compute the Cholesky factor of A.
//
// The upper flag is forwarded verbatim to the backend so the caller can
// choose which triangular factor is returned.  The output has the same shape
// as the input; only the requested triangle is filled; the other triangle is
// zeroed by the backend implementation.
//
// Autograd: not wired (no backward node is defined for this op).  The output
// TensorImpl therefore has requires_grad=false and is treated as a leaf in the
// gradient graph.  Attempting to call backward through cholesky_op will not
// propagate gradients to the input.
TensorImplPtr cholesky_op(const TensorImplPtr& a_in, bool upper) {
    Validator::input(a_in, "cholesky.a").non_null().square_2d();
    const AmpScope amp{"cholesky", a_in->device()};
    const TensorImplPtr a = amp.cast(a_in);
    Validator::input(a, "cholesky.a").float_only();
    OpScopeFull scope{"cholesky", a->device(), a->dtype(), a->shape()};
    scope.set_attr("upper", upper);

    // A degenerate matrix has a decomposition; LAPACK just will not be
    // the one to compute it.  See ``empty_matrix`` for why dispatching
    // wrote to a stream Lucid does not own and then raised.
    if (linalg_detail::empty_matrix(a->shape()))
        return zeros_op(a->shape(), a->dtype(), a->device());

    Storage out = backend::Dispatcher::for_device(a->device())
                      .linalg_cholesky(a->storage(), a->shape(), upper, a->dtype());
    auto res = linalg_detail::fresh(std::move(out), a->shape(), a->dtype(), a->device());
    if (auto* trc = compile::current_tracer())
        trc->on_op_io({a}, res);
    return res;
}

}  // namespace lucid
