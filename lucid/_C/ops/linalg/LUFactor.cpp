// lucid/_C/ops/linalg/LUFactor.cpp
//
// Implements lu_factor_op: LU factorisation with partial pivoting via
// IBackend::linalg_lu_factor() → LAPACK sgetrf_/dgetrf_.

#include "LUFactor.h"

#include <algorithm>

#include "../../backend/Dispatcher.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/Helpers.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../gfunc/Gfunc.h"
#include "_Detail.h"

namespace lucid {

std::vector<TensorImplPtr> lu_factor_op(const TensorImplPtr& a) {
    using namespace linalg_detail;
    Validator::input(a, "lu_factor.a").float_only();
    if (a->shape().size() < 2)
        ErrorBuilder("lu_factor.a").invalid_argument("input must be at least 2-D");

    const auto& sh = a->shape();
    const int m = static_cast<int>(sh[sh.size() - 2]);
    const int n = static_cast<int>(sh[sh.size() - 1]);

    // LU with partial pivoting is defined for any shape — ``?getrf``
    // factorises an m-by-n matrix into ``P L U`` with L unit-lower
    // m-by-min(m,n) and U upper min(m,n)-by-n.  This op required a square
    // input, which was a restriction of the wrapper rather than of the
    // factorisation: the LAPACK call underneath was passing the same
    // extent for both dimensions.
    //
    // One pivot per elimination step, and there are only as many steps as
    // the shorter side.
    Shape pivot_shape(sh.begin(), sh.end() - 2);
    pivot_shape.push_back(std::min(m, n));

    // A degenerate matrix has a decomposition; LAPACK just will not be
    // the one to compute it.  See ``empty_matrix`` for why dispatching
    // wrote to a stream Lucid does not own and then raised.
    // ``getrf``'s own ``info`` per matrix, the batch dims: a singular
    // matrix is factored, and whether that is an error is the caller's call.
    const Shape info_shape(sh.begin(), sh.end() - 2);
    if (empty_matrix(sh))
        return {zeros_op(sh, a->dtype(), a->device()),
                zeros_op(pivot_shape, Dtype::I32, a->device()),
                zeros_op(info_shape, Dtype::I32, a->device())};

    OpScopeFull scope{"lu_factor", a->device(), a->dtype(), sh};
    auto out =
        backend::Dispatcher::for_device(a->device()).linalg_lu_factor(a->storage(), sh, a->dtype());

    auto lu = fresh(std::move(out[0]), sh, a->dtype(), a->device());
    auto pivots = fresh(std::move(out[1]), pivot_shape, Dtype::I32, a->device());
    auto info = fresh(std::move(out[2]), info_shape, Dtype::I32, a->device());
    // No emitter builds a factorisation, but an unrecorded one is worse than
    // an unbuildable one: its three outputs would reach the consumer as
    // external feeds, bound at trace time, and every later call would answer
    // the first.  Recorded with its input, the signature falls back to eager.
    if (auto* trc = compile::current_tracer()) {
        trc->on_op_io({a}, lu);
        trc->on_op_io({}, pivots);
        trc->on_op_io({}, info);
    }
    return {lu, pivots, info};
}

}  // namespace lucid
