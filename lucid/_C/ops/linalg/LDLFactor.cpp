// lucid/_C/ops/linalg/LDLFactor.cpp
#include "LDLFactor.h"

#include "../../backend/Dispatcher.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Helpers.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../gfunc/Gfunc.h"
#include "_Detail.h"
namespace lucid {

std::vector<TensorImplPtr> ldl_factor_op(const TensorImplPtr& a) {
    using namespace linalg_detail;
    Validator::input(a, "ldl_factor.a").float_only().square_2d();

    const auto& sh = a->shape();
    const int n = static_cast<int>(sh[sh.size() - 1]);

    Shape empty_piv(sh.begin(), sh.end() - 2);
    empty_piv.push_back(static_cast<std::int64_t>(n));
    // A degenerate matrix has an answer; LAPACK just will not be the one
    // to compute it — see ``empty_matrix``.
    // ``sytrf``'s own ``info`` per matrix, the batch dims: a singular D is
    // factored, and whether that is an error is the caller's call.
    const Shape info_shape(sh.begin(), sh.end() - 2);
    if (empty_matrix(sh))
        return {zeros_op(sh, a->dtype(), a->device()), zeros_op(empty_piv, Dtype::I32, a->device()),
                zeros_op(info_shape, Dtype::I32, a->device())};

    OpScopeFull scope{"ldl_factor", a->device(), a->dtype(), sh};
    auto out = backend::Dispatcher::for_device(a->device())
                   .linalg_ldl_factor(a->storage(), sh, a->dtype());

    auto ld = fresh(std::move(out[0]), sh, a->dtype(), a->device());
    auto piv = fresh(std::move(out[1]), empty_piv, Dtype::I32, a->device());
    auto info = fresh(std::move(out[2]), info_shape, Dtype::I32, a->device());
    // Unrecorded, the three outputs reach the consumer as external feeds,
    // bound at trace time; recorded with the input, and with no emitter, the
    // signature falls back to eager.  See ``lu_factor_op``.
    if (auto* trc = compile::current_tracer()) {
        trc->on_op_io({a}, ld);
        trc->on_op_io({}, piv);
        trc->on_op_io({}, info);
    }
    return {ld, piv, info};
}

}  // namespace lucid
