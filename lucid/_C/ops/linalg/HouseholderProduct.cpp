// lucid/_C/ops/linalg/HouseholderProduct.cpp
#include "HouseholderProduct.h"

#include "../../backend/Dispatcher.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../gfunc/Gfunc.h"
#include "_Detail.h"
namespace lucid {

TensorImplPtr householder_product_op(const TensorImplPtr& H_in, const TensorImplPtr& tau_in) {
    using namespace linalg_detail;
    Validator::input(H_in, "householder_product.H").non_null();
    Validator::input(tau_in, "householder_product.tau").non_null();
    const AmpScope amp{"householder_product", H_in->device()};
    const TensorImplPtr H = amp.cast(H_in);
    const TensorImplPtr tau = amp.cast(tau_in);
    Validator::input(H, "householder_product.H").float_only();
    Validator::input(tau, "householder_product.tau").float_only();

    const auto& hs = H->shape();
    if (hs.size() < 2)
        ErrorBuilder("householder_product").invalid_argument("H must be at least 2-D");
    const int m = static_cast<int>(hs[hs.size() - 2]);
    const int n = static_cast<int>(hs[hs.size() - 1]);
    const int k = std::min(m, n);

    Shape q_shape(hs.begin(), hs.end() - 2);
    q_shape.push_back(static_cast<std::int64_t>(m));
    q_shape.push_back(static_cast<std::int64_t>(k));
    // A degenerate matrix has an answer; LAPACK just will not be the one
    // to compute it — see ``empty_matrix``.
    if (empty_matrix(hs))
        return zeros_op(q_shape, H->dtype(), H->device());

    OpScopeFull scope{"householder_product", H->device(), H->dtype(), q_shape};
    auto result = backend::Dispatcher::for_device(H->device())
                      .linalg_householder_product(H->storage(), tau->storage(), hs, H->dtype());

    auto q = fresh(std::move(result), q_shape, H->dtype(), H->device());
    // Unrecorded, Q reaches the consumer as an external feed, bound at trace
    // time; recorded with both inputs, and with no emitter, the signature
    // falls back to eager.  See ``lu_factor_op``.
    if (auto* trc = compile::current_tracer())
        trc->on_op_io({H, tau}, q);
    return q;
}

}  // namespace lucid
