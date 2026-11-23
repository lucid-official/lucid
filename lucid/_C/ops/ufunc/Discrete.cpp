// lucid/_C/ops/ufunc/Discrete.cpp
//
// round, floor, ceil and trunc are step functions: their gradient is zero wherever
// it exists, and they stay in the graph with that zero gradient, as the
// reference framework keeps them.  invert (bitwise not) takes integers and
// bools only, so it never carries a gradient.

#include "Discrete.h"

#include "../../autograd/Helpers.h"
#include "../../core/OpRegistry.h"
#include "../gfunc/Gfunc.h"

namespace lucid {

// round — KeepInput preserves integer types (round is a no-op on integers).
// The four step functions below share one refusal: the reference has no bool
// kernel for them (``NotImplementedError``) and answers every integer width.
const OpSchema RoundBackward::schema_v1 =
    OpSchema{"round", 1, AmpPolicy::KeepInput, true}.refusing_bool(DtypeRefusal::Kernel,
                                                                   CastHint::Number);

// Zero gradient: round is piecewise constant.
// A step function is flat wherever it is differentiable, so the gradient is
// zero — kept in the graph, as the reference framework keeps it.
Storage RoundBackward::grad_formula(const Storage&) {
    return make_zero_storage(out_shape_, dtype_, device_);
}

TensorImplPtr RoundBackward::grad_formula_impl(const TensorImplPtr& g,
                                               const TensorImplPtr&,
                                               const TensorImplPtr&) {
    return zeros_like_op(g);
}
TensorImplPtr round_op(const TensorImplPtr& a) {
    return RoundBackward::forward(a);
}
LUCID_REGISTER_OP(RoundBackward)

// floor — KeepInput.
const OpSchema FloorBackward::schema_v1 =
    OpSchema{"floor", 1, AmpPolicy::KeepInput, true}.refusing_bool(DtypeRefusal::Kernel,
                                                                   CastHint::Number);

// Zero gradient: floor is piecewise constant.
// A step function is flat wherever it is differentiable, so the gradient is
// zero — kept in the graph, as the reference framework keeps it.
Storage FloorBackward::grad_formula(const Storage&) {
    return make_zero_storage(out_shape_, dtype_, device_);
}

TensorImplPtr FloorBackward::grad_formula_impl(const TensorImplPtr& g,
                                               const TensorImplPtr&,
                                               const TensorImplPtr&) {
    return zeros_like_op(g);
}
TensorImplPtr floor_op(const TensorImplPtr& a) {
    return FloorBackward::forward(a);
}
LUCID_REGISTER_OP(FloorBackward)

// ceil — KeepInput.
const OpSchema CeilBackward::schema_v1 =
    OpSchema{"ceil", 1, AmpPolicy::KeepInput, true}.refusing_bool(DtypeRefusal::Kernel,
                                                                  CastHint::Number);

// Zero gradient: ceil is piecewise constant.
// A step function is flat wherever it is differentiable, so the gradient is
// zero — kept in the graph, as the reference framework keeps it.
Storage CeilBackward::grad_formula(const Storage&) {
    return make_zero_storage(out_shape_, dtype_, device_);
}

TensorImplPtr CeilBackward::grad_formula_impl(const TensorImplPtr& g,
                                              const TensorImplPtr&,
                                              const TensorImplPtr&) {
    return zeros_like_op(g);
}
TensorImplPtr ceil_op(const TensorImplPtr& a) {
    return CeilBackward::forward(a);
}
LUCID_REGISTER_OP(CeilBackward)

// trunc — KeepInput.
const OpSchema TruncBackward::schema_v1 =
    OpSchema{"trunc", 1, AmpPolicy::KeepInput, true}.refusing_bool(DtypeRefusal::Kernel,
                                                                   CastHint::Number);

Storage TruncBackward::dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
    if (!is_floating_point(dt))
        return be.floor(a, s, dt);  // already whole: floor is the identity here
    // ``-0.0 >= 0`` holds and ``floor(-0.0)`` is ``-0.0``, so the sign of zero
    // survives; NaN fails the test and ``ceil(NaN)`` is NaN.
    constexpr int kGreaterEqual = 3;
    const Storage non_negative = be.compare_binary(a, be.full(s, dt, 0.0), s, dt, kGreaterEqual);
    return be.where_op(non_negative, be.floor(a, s, dt), be.ceil(a, s, dt), s, dt);
}

// Zero gradient: trunc is piecewise constant.
Storage TruncBackward::grad_formula(const Storage&) {
    return make_zero_storage(out_shape_, dtype_, device_);
}

TensorImplPtr TruncBackward::grad_formula_impl(const TensorImplPtr& g,
                                               const TensorImplPtr&,
                                               const TensorImplPtr&) {
    return zeros_like_op(g);
}
TensorImplPtr trunc_op(const TensorImplPtr& a) {
    return TruncBackward::forward(a);
}
LUCID_REGISTER_OP(TruncBackward)

// invert — KeepInput; bitwise NOT is only defined for integer types.
const OpSchema InvertBackward::schema_v1{"invert", 1, AmpPolicy::KeepInput, true};

// Zero gradient: bitwise NOT has no floating-point derivative.
Storage InvertBackward::grad_formula(const Storage&) {
    return Storage{CpuStorage{}};
}
TensorImplPtr invert_op(const TensorImplPtr& a) {
    return InvertBackward::forward(a);
}
LUCID_REGISTER_OP(InvertBackward)

}  // namespace lucid
