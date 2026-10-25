// lucid/_C/ops/ufunc/ScalarParam.cpp
//
// Forward implementations and gradient formulas for the three scalar-parameter
// unary ops.  Each uses a custom forward() that captures the scalar on the
// backward node before calling NaryKernel::wire_autograd.

#include "ScalarParam.h"

#include <cmath>

#include "../../autograd/Engine.h"
#include "../../backend/Dispatcher.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Half.h"
#include "../../core/OpRegistry.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../../kernel/NaryKernel.h"
#include "../bfunc/Compare.h"
#include "../bfunc/Mul.h"
#include "../gfunc/Gfunc.h"
#include "../utils/Promote.h"
#include "../utils/Select.h"
#include "Astype.h"
#include "Predicate.h"

namespace lucid {

// pow_scalar — ForceFP32; general power is not safe in integer or half types.
// set_flops uses *11 as a rough estimate for the cost of a pow() call.
const OpSchema PowScalarBackward::schema_v1{"pow_scalar", 1, AmpPolicy::ForceFP32, true};

// dL/dx = exp * x^(exp-1) * dL/dy (0 everywhere when exp == 0 so x = 0 does
// not produce 0 * inf = NaN).
// Reuses the backend's pow_scalar to compute x^(exp-1) rather than
// implementing a separate "reduce-exponent-by-one" kernel.
Storage PowScalarBackward::grad_formula(const Storage& g) {
    if (exp_ == 0.0)
        return make_zero_storage(out_shape_, dtype_, device_);
    const std::size_t n = shape_numel(out_shape_);
    Storage x_pow_em1 = backend::Dispatcher::for_device(device_).pow_scalar(
        saved_inputs_[0], out_shape_, dtype_, exp_ - 1.0);
    Storage scaled = mul_scalar_storage(x_pow_em1, exp_, n, dtype_, device_);
    return multiply_storages(scaled, g, n, dtype_, device_);
}

TensorImplPtr PowScalarBackward::grad_formula_impl(const TensorImplPtr& g,
                                                   const TensorImplPtr& a,
                                                   const TensorImplPtr& /*out*/) {
    if (exp_ == 0.0)
        return zeros_like_op(g, false);
    // e * x^(e-1), kept on the graph.  Scaling the storage directly and
    // wrapping it in a fresh tensor cut x^(e-1) off from x, so the
    // second derivative lost its e (e - 1) x^(e-2) g term and answered
    // wrong — silently — for every exponent but 0 and 1.
    auto a_pow_em1 = pow_scalar_op(a, exp_ - 1.0);
    auto scaled = mul_op(a_pow_em1, full_like_op(a_pow_em1, exp_, false));
    return mul_op(scaled, g);
}

// Capture exp before wiring autograd so grad_formula has it at backward time.
TensorImplPtr PowScalarBackward::forward(const TensorImplPtr& a_in, double exp) {
    Validator::input(a_in, "pow_scalar.a").non_null();
    const TensorImplPtr a = promote_for_schema(schema_v1, a_in);
    if (a->dtype() == Dtype::F16) {
        exp = backend::detail::half_bits_to_float(
            backend::detail::float_to_half_bits(static_cast<float>(exp)));
    } else if (a->dtype() == Dtype::BF16) {
        exp = backend::detail::bfloat16_bits_to_float(
            backend::detail::float_to_bfloat16_bits(static_cast<float>(exp)));
    }

    OpScopeFull scope{schema_v1.name, a->device(), a->dtype(), a->shape()};
    scope.set_attr("exp", exp);
    Storage out_storage = backend::Dispatcher::for_device(a->device())
                              .pow_scalar(a->storage(), a->shape(), a->dtype(), exp);
    auto out = std::make_shared<TensorImpl>(std::move(out_storage), a->shape(), a->dtype(),
                                            a->device(), false);
    scope.set_flops(static_cast<std::int64_t>(out->numel()) * 11);

    auto bwd = std::make_shared<PowScalarBackward>();
    bwd->exp_ = exp;
    kernel::NaryKernel<PowScalarBackward, 1>::wire_autograd(std::move(bwd), {a}, out, exp != 0.0);
    return out;
}

TensorImplPtr pow_scalar_op(const TensorImplPtr& a, double exp) {
    return PowScalarBackward::forward(a, exp);
}
LUCID_REGISTER_OP(PowScalarBackward)

// rpow_scalar — ForceFP32; ln(base) requires floating-point.
const OpSchema RPowScalarBackward::schema_v1{"rpow_scalar", 1, AmpPolicy::ForceFP32, true};

// dL/dx = ln(base) * y * dL/dy  where y = saved_output_ = base^x.
// When base == 0 and x >= 0, masks the derivative to 0 (matching the reference
// so 0 ** 0 -> 1 * (-inf) = -inf and 0 ** (+x) -> 0 * (-inf) = NaN become 0).
Storage RPowScalarBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);

    const double ln_base = std::log(base_);
    Storage scaled_out = mul_scalar_storage(saved_output_, ln_base, n, dtype_, device_);
    if (base_ == 0.0) {
        auto& be = backend::Dispatcher::for_device(device_);
        Storage zero = make_zero_storage(out_shape_, dtype_, device_);
        Storage ge_zero = be.compare_binary(saved_inputs_[0], zero, out_shape_, dtype_, 3);
        scaled_out = be.where_op(ge_zero, zero, scaled_out, out_shape_, dtype_);
    }
    return multiply_storages(scaled_out, g, n, dtype_, device_);
}

TensorImplPtr RPowScalarBackward::grad_formula_impl(const TensorImplPtr& g,
                                                    const TensorImplPtr& x,
                                                    const TensorImplPtr& out) {
    auto slope = mul_op(out, full_like_op(out, std::log(base_)));
    if (base_ == 0.0)
        slope = where_op(greater_equal_op(x, zeros_like_op(x)), zeros_like_op(slope), slope);
    return mul_op(g, slope);
}

// Manually saves the output in bwd->saved_output_ because the standard
// wire_autograd helper does not do so; saves the input only when base == 0.0.
TensorImplPtr RPowScalarBackward::forward(double base, const TensorImplPtr& a_in) {
    Validator::input(a_in, "rpow_scalar.a").non_null();
    const TensorImplPtr a = promote_for_schema(schema_v1, a_in);

    OpScopeFull scope{schema_v1.name, a->device(), a->dtype(), a->shape()};
    scope.set_attr("base", base);
    Storage out_storage = backend::Dispatcher::for_device(a->device())
                              .rpow_scalar(a->storage(), a->shape(), a->dtype(), base);
    auto out = std::make_shared<TensorImpl>(std::move(out_storage), a->shape(), a->dtype(),
                                            a->device(), false);
    scope.set_flops(static_cast<std::int64_t>(out->numel()) * 11);

    auto bwd = std::make_shared<RPowScalarBackward>();
    bwd->keep_output(out->storage());
    bwd->base_ = base;
    kernel::NaryKernel<RPowScalarBackward, 1>::wire_autograd(std::move(bwd), {a}, out, base == 0.0);
    return out;
}

TensorImplPtr rpow_scalar_op(double base, const TensorImplPtr& a) {
    return RPowScalarBackward::forward(base, a);
}
LUCID_REGISTER_OP(RPowScalarBackward)

// clip — KeepInput: clamp is valid on integer inputs.
const OpSchema ClipBackward::schema_v1{"clip", 1, AmpPolicy::KeepInput, true};

// dL/dx = 1 where min_ <= x <= max_, else 0  (in-range boolean mask).
Storage ClipBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    // Inclusive: the reference's ``clamp`` backward passes the gradient at
    // the bound, and ``grad_formula_impl`` below matches it.
    Storage mask = in_range_mask_storage(saved_inputs_[0], min_, max_, n, dtype_, device_, true);
    return multiply_storages(g, mask, n, dtype_, device_);
}

// Capture both bounds on the backward node before wiring autograd.
TensorImplPtr ClipBackward::forward(const TensorImplPtr& a, double min_v, double max_v) {
    Validator::input(a, "clip.a").non_null();

    OpScopeFull scope{schema_v1.name, a->device(), a->dtype(), a->shape()};
    scope.set_attr("min", min_v);
    scope.set_attr("max", max_v);
    Storage out_storage = backend::Dispatcher::for_device(a->device())
                              .clip(a->storage(), a->shape(), a->dtype(), min_v, max_v);
    auto out = std::make_shared<TensorImpl>(std::move(out_storage), a->shape(), a->dtype(),
                                            a->device(), false);
    scope.set_flops(static_cast<std::int64_t>(out->numel()));

    auto bwd = std::make_shared<ClipBackward>();
    bwd->min_ = min_v;
    bwd->max_ = max_v;
    kernel::NaryKernel<ClipBackward, 1>::wire_autograd(std::move(bwd), {a}, out);
    return out;
}

// clip'(x) = 1 on the closed interval, 0 outside it.
//
// The bound itself carries the gradient: at x == min_ the clamp is the
// identity from the right, and the reference framework's ``clamp``
// backward masks with ``(x >= min) & (x <= max)``.  This used to be
// strict on both ends while ``grad_formula`` above used the inclusive
// mask, so the same gradient came out differently depending on whether
// it was asked for by ``backward()`` or ``grad(create_graph=True)`` --
// exactly 1.0 relative disagreement at a value sitting on a bound.
// ``two_hot`` puts one there whenever the encoded value lands on a bin,
// which for a symmetric grid includes zero.
//
// Note this is the opposite convention to ``relu6``/``hardtanh``, where
// the reference returns 0 at the bound.  They are different ops, not an
// inconsistency to be smoothed over.
//
// Same omission as ``maximum``: no ``grad_formula_impl``, so every
// composite over it — ``binary_cross_entropy``, ``hardtanh``,
// ``adjust_brightness``, ``ndtri`` — could not be differentiated twice.
TensorImplPtr ClipBackward::grad_formula_impl(const TensorImplPtr& g,
                                              const TensorImplPtr& x,
                                              const TensorImplPtr&) {
    auto above_low = greater_equal_op(x, full_like_op(x, min_));
    auto below_high = less_equal_op(x, full_like_op(x, max_));
    auto zero = zeros_like_op(x);
    auto slope = where_op(above_low, where_op(below_high, ones_like_op(x), zero), zero);
    // Kept in the graph with derivative zero: ``hardtanh`` is this op, and
    // the reference's hardtanh backward is an op of ``x``.
    return zero_derivative_through(mul_op(g, slope), x);
}

TensorImplPtr clip_op(const TensorImplPtr& a, double min_v, double max_v) {
    return ClipBackward::forward(a, min_v, max_v);
}
LUCID_REGISTER_OP(ClipBackward)

const OpSchema NanToNumBackward::schema_v1{"nan_to_num", 1, AmpPolicy::KeepInput, true};

TensorImplPtr NanToNumBackward::forward(const TensorImplPtr& a,
                                        double nan_val,
                                        double posinf_val,
                                        double neginf_val) {
    Validator::input(a, "nan_to_num.a").non_null();
    OpScopeFull scope{schema_v1.name, a->device(), a->dtype(), a->shape()};
    scope.set_attr("nan", nan_val);
    scope.set_attr("posinf", posinf_val);
    scope.set_attr("neginf", neginf_val);
    Storage out_storage =
        backend::Dispatcher::for_device(a->device())
            .nan_to_num(a->storage(), a->shape(), a->dtype(), nan_val, posinf_val, neginf_val);
    auto out = std::make_shared<TensorImpl>(std::move(out_storage), a->shape(), a->dtype(),
                                            a->device(), false);
    scope.set_flops(static_cast<std::int64_t>(out->numel()));
    // ``wire_autograd`` also records the op on an active trace.
    kernel::NaryKernel<NanToNumBackward, 1>::wire_autograd(std::make_shared<NanToNumBackward>(),
                                                           {a}, out);
    return out;
}

Storage NanToNumBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    auto& be = backend::Dispatcher::for_device(device_);
    Storage finite = be.isfinite(saved_inputs_[0], out_shape_, dtype_);
    Storage mask = be.astype(finite, out_shape_, Dtype::Bool, dtype_);
    return multiply_storages(g, mask, n, dtype_, device_);
}

TensorImplPtr NanToNumBackward::grad_formula_impl(const TensorImplPtr& g,
                                                  const TensorImplPtr& x,
                                                  const TensorImplPtr&) {
    return where_op(isfinite_op(x), g, zeros_like_op(g));
}

LUCID_REGISTER_OP(NanToNumBackward)

}  // namespace lucid
