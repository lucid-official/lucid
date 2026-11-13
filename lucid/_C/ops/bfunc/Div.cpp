// lucid/_C/ops/bfunc/Div.cpp
//
// Implements DivBackward::grad_formula and the div_op free function.

#include "Div.h"

#include <mlx/ops.h>

#include "../../autograd/Helpers.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/OpRegistry.h"
#include "../../ops/ufunc/Arith.h"
#include "Mul.h"
#include "_Opmath.h"

namespace lucid {

const OpSchema DivBackward::schema_v1{"div", 1, AmpPolicy::Promote, true};

// Gradient of element-wise division (quotient rule).
//
// Given c = a / b:
//   dA = grad_out / b
//   dB = -(grad_out * (a / b)) / b
//
// dB is the textbook -(grad_out * a) / b^2 with the square never formed.  b^2 leaves
// float32 for a |b| outside ~1e-19 .. 1e19, so the textbook form returns 0 or NaN
// there for a gradient that is an ordinary float whenever grad_out and a are of b's
// scale: x / ||x|| for a norm near 1e25, or any create_graph backward through a tracked
// divisor, whose second-order cotangent is of the divisor's magnitude.  Dividing by b
// twice never forms a square of b, so the range is that of grad_out * (a / b) rather
// than of b^2, and a / b is 0 for an a that is 0, which a product g * a / b^2 turns
// into 0 / 0 once b^2 underflows.
//
// The computation is broken into explicit storage operations so that each
// intermediate value can be freed individually.  Both saved inputs are first
// expanded to out_shape_ (via saved_input_broadcasted) so that all element-wise
// operations are shape-compatible.  BinaryKernel::apply conjugates around this
// formula on a complex dtype (grad_out / conj(b)).
std::pair<Storage, Storage> DivBackward::grad_formula(const Storage& grad_out) {
    const std::size_t n = shape_numel(out_shape_);
    const bool need_a = needs_input_grad(0);
    const bool need_b = needs_input_grad(1);
    if (!need_a && !need_b)
        return {Storage{}, Storage{}};

    auto b_b = saved_input_broadcasted(1);

    // dA = grad_out / b
    Storage dx = need_a ? divide_storages(grad_out, b_b, n, dtype_, device_) : Storage{};
    if (!need_b)
        return {std::move(dx), Storage{}};

    auto a_b = saved_input_broadcasted(0);

    // dB = -(grad_out * (a / b)) / b
    Storage quotient = divide_storages(a_b, b_b, n, dtype_, device_);
    Storage g_times_q = multiply_storages(grad_out, quotient, n, dtype_, device_);
    Storage div_by_b = divide_storages(g_times_q, b_b, n, dtype_, device_);
    Storage dy = negate_storage(div_by_b, n, dtype_, device_);
    return {std::move(dx), std::move(dy)};
}

std::pair<TensorImplPtr, TensorImplPtr> DivBackward::grad_formula_impl(
    const TensorImplPtr& grad_out, const TensorImplPtr& a, const TensorImplPtr& b) {
    // da = grad_out / b
    TensorImplPtr da = needs_input_grad(0) ? div_op(grad_out, b) : nullptr;
    if (!needs_input_grad(1))
        return {std::move(da), nullptr};
    // db = -(grad_out * (a / b)) / b; see grad_formula for why b^2 is never formed.
    // This graph is differentiated again, and the quotient rule of each division in it
    // forms the same expression: the divisor's square stays out of every order.
    auto db = neg_op(div_op(mul_op(grad_out, div_op(a, b)), b));
    return {std::move(da), std::move(db)};
}

TensorImplPtr div_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    if (auto r = opmath::try_scalar<opmath::Op::Div>(a, b, &div_op))
        return r;
    return DivBackward::forward(a, b);
}

LUCID_REGISTER_OP(DivBackward)

}  // namespace lucid
