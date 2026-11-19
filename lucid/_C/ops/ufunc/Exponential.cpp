// lucid/_C/ops/ufunc/Exponential.cpp
//
// Gradient formulas and entry points for the exponential / logarithm family.
// AmpPolicy::ForceFP32 is applied to exp/log/log2 to keep numerical precision
// consistent with vForce, which operates in single precision on CPU.

#include "Exponential.h"

#include "../../autograd/Helpers.h"
#include "../../backend/Dispatcher.h"
#include "../../core/OpRegistry.h"
#include "../bfunc/Add.h"
#include "../bfunc/Div.h"
#include "../bfunc/Mul.h"
#include "../bfunc/Sub.h"
#include "../gfunc/Gfunc.h"
#include "Arith.h"

namespace lucid {

// exp — ForceFP32 prevents half-precision underflow during exponentiation.
const OpSchema ExpBackward::schema_v1{"exp", 1, AmpPolicy::ForceFP32, true};

// dL/dx = dL/dy * y  (since d/dx e^x = e^x = y).
// Using saved_output_ avoids re-running the vForce exp kernel.
Storage ExpBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);

    return multiply_storages(g, saved_output_, n, dtype_, device_);
}

TensorImplPtr ExpBackward::grad_formula_impl(const TensorImplPtr& g,
                                             const TensorImplPtr&,
                                             const TensorImplPtr& out) {
    // dx = out * g  (out = exp(x) was saved as the forward output)
    return mul_op(g, out);
}

TensorImplPtr exp_op(const TensorImplPtr& a) {
    return ExpBackward::forward(a);
}
LUCID_REGISTER_OP(ExpBackward)

// log — ForceFP32 for consistent behaviour with vvlogf.
const OpSchema LogBackward::schema_v1{"log", 1, AmpPolicy::ForceFP32, true};

// dL/dx = dL/dy / x.
Storage LogBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    return divide_storages(g, saved_inputs_[0], n, dtype_, device_);
}

TensorImplPtr LogBackward::grad_formula_impl(const TensorImplPtr& g,
                                             const TensorImplPtr& x,
                                             const TensorImplPtr&) {
    return div_op(g, x);
}

TensorImplPtr log_op(const TensorImplPtr& a) {
    return LogBackward::forward(a);
}
LUCID_REGISTER_OP(LogBackward)

const OpSchema Log1pBackward::schema_v1{"log1p", 1, AmpPolicy::KeepInput, true};

Storage Log1pBackward::grad_formula(const Storage& g) {
    const auto n = shape_numel(out_shape_);
    auto denominator = add_scalar_storage(saved_inputs_[0], 1.0, n, dtype_, device_);
    return divide_storages(g, denominator, n, dtype_, device_);
}

TensorImplPtr Log1pBackward::grad_formula_impl(const TensorImplPtr& g,
                                               const TensorImplPtr& x,
                                               const TensorImplPtr&) {
    return div_op(g, add_op(x, ones_like_op(x)));
}

LUCID_REGISTER_OP(Log1pBackward)

// log2 — ForceFP32; gradient includes the chain-rule factor 1/ln(2).
const OpSchema Log2Backward::schema_v1{"log2", 1, AmpPolicy::ForceFP32, true};

// dL/dx = dL/dy / (x * ln(2)).
// kLn2 is specified to 50 significant digits so that it rounds correctly to
// both float32 and float64 without rounding compensation.
Storage Log2Backward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);

    constexpr double kLn2 = 0.69314718055994530941723212145817656807550013436;
    Storage x_ln2 = mul_scalar_storage(saved_inputs_[0], kLn2, n, dtype_, device_);
    return divide_storages(g, x_ln2, n, dtype_, device_);
}

TensorImplPtr log2_op(const TensorImplPtr& a) {
    return Log2Backward::forward(a);
}
TensorImplPtr Log2Backward::grad_formula_impl(const TensorImplPtr& g,
                                              const TensorImplPtr& x,
                                              const TensorImplPtr&) {
    // 1 / (x ln 2)
    constexpr double kLn2 = 0.6931471805599453;
    return div_op(g, mul_op(x, full_like_op(x, kLn2)));
}

LUCID_REGISTER_OP(Log2Backward)

// sqrt — AmpPolicy::Promote (not ForceFP32) so that float64 inputs remain f64.
const OpSchema SqrtBackward::schema_v1{"sqrt", 1, AmpPolicy::Promote, true, "", true};

// dL/dx = 0.5 * dL/dy / y  (since d/dx sqrt(x) = 1 / (2*sqrt(x)) = 1/(2*y)).
// Dividing by saved_output_ rather than recomputing sqrt(x) avoids a second
// vForce call and keeps the formula numerically stable near x = 0.
Storage SqrtBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);

    Storage half_g = mul_scalar_storage(g, 0.5, n, dtype_, device_);
    return divide_storages(half_g, saved_output_, n, dtype_, device_);
}

TensorImplPtr SqrtBackward::grad_formula_impl(const TensorImplPtr& g,
                                              const TensorImplPtr&,
                                              const TensorImplPtr& out) {
    // dx = g / (2*y) where y = sqrt(x) is the saved output
    return div_op(g, add_op(out, out));
}

TensorImplPtr sqrt_op(const TensorImplPtr& a) {
    return SqrtBackward::forward(a);
}
LUCID_REGISTER_OP(SqrtBackward)

// rsqrt — AmpPolicy::Promote so float64 inputs stay float64.
const OpSchema RsqrtBackward::schema_v1{"rsqrt", 1, AmpPolicy::Promote, true, "", true};

// dL/dx = dL/dy * (-0.5) * y^3  where y = rsqrt(x) = x^(-1/2).
// Uses saved_output_ (y) to compute y^3 = y * y * y without re-running rsqrt.
Storage RsqrtBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    const Storage& y = saved_output_;
    Storage y2 = multiply_storages(y, y, n, dtype_, device_);
    Storage y3 = multiply_storages(y2, y, n, dtype_, device_);
    Storage neg_half_y3 = mul_scalar_storage(y3, -0.5, n, dtype_, device_);
    return multiply_storages(g, neg_half_y3, n, dtype_, device_);
}

TensorImplPtr rsqrt_op(const TensorImplPtr& a) {
    return RsqrtBackward::forward(a);
}
TensorImplPtr RsqrtBackward::grad_formula_impl(const TensorImplPtr& g,
                                               const TensorImplPtr&,
                                               const TensorImplPtr& out) {
    // d/dx x^(-1/2) = -1/2 x^(-3/2) = -1/2 * out^3
    auto cubed = mul_op(out, mul_op(out, out));
    return mul_op(g, mul_op(full_like_op(out, -0.5), cubed));
}

LUCID_REGISTER_OP(RsqrtBackward)

// erf — AmpPolicy::Promote so float64 inputs remain float64.
const OpSchema ErfBackward::schema_v1{"erf", 1, AmpPolicy::Promote, true, "", true};

// dL/dx = (2/√π) * exp(-x²) * dL/dy.
// Saves the input x to compute exp(-x²) in the backward pass.
static constexpr double kTwoOverSqrtPi = 1.1283791670955126;  // 2/sqrt(pi)

Storage ErfBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    const Storage& x = saved_inputs_[0];
    // x^2
    Storage x2 = multiply_storages(x, x, n, dtype_, device_);
    // -x^2
    Storage neg_x2 = mul_scalar_storage(x2, -1.0, n, dtype_, device_);
    // exp(-x^2)
    Storage ex2 = backend::Dispatcher::for_device(device_).exp(neg_x2, out_shape_, dtype_);
    // (2/sqrt(pi)) * exp(-x^2)
    Storage coeff = mul_scalar_storage(ex2, kTwoOverSqrtPi, n, dtype_, device_);
    // multiply by upstream gradient
    return multiply_storages(g, coeff, n, dtype_, device_);
}

TensorImplPtr ErfBackward::grad_formula_impl(const TensorImplPtr& g,
                                             const TensorImplPtr& x,
                                             const TensorImplPtr&) {
    // dx = (2/sqrt(pi)) * exp(-x^2) * g
    auto neg_x2 = neg_op(mul_op(x, x));
    auto ex2 = exp_op(neg_x2);
    // Multiply ex2 by the 2/sqrt(pi) constant via full_like
    auto scale = full_like_op(ex2, kTwoOverSqrtPi, /*requires_grad=*/false);
    return mul_op(g, mul_op(scale, ex2));
}

TensorImplPtr erf_op(const TensorImplPtr& a) {
    return ErfBackward::forward(a);
}
LUCID_REGISTER_OP(ErfBackward)

// erfinv — AmpPolicy::Promote so float64 inputs remain float64.
const OpSchema ErfinvBackward::schema_v1{"erfinv", 1, AmpPolicy::Promote, true, "", true};

// dL/dx = (sqrt(π)/2) * exp(out²) * g   (out = erfinv(x) was saved as forward output).
static constexpr double kSqrtPiOver2 = 0.8862269254527580;  // sqrt(pi)/2

Storage ErfinvBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    const Storage& y = saved_output_;
    // y^2
    Storage y2 = multiply_storages(y, y, n, dtype_, device_);
    // exp(y^2)
    Storage ey2 = backend::Dispatcher::for_device(device_).exp(y2, out_shape_, dtype_);
    // sqrt(pi)/2 * exp(y^2)
    Storage coeff = mul_scalar_storage(ey2, kSqrtPiOver2, n, dtype_, device_);
    return multiply_storages(g, coeff, n, dtype_, device_);
}

TensorImplPtr ErfinvBackward::grad_formula_impl(const TensorImplPtr& g,
                                                const TensorImplPtr&,
                                                const TensorImplPtr& out) {
    // dx = sqrt(pi)/2 * exp(out^2) * g
    auto out2 = mul_op(out, out);
    auto ey2 = exp_op(out2);
    auto coeff = mul_op(full_like_op(ey2, kSqrtPiOver2), ey2);
    return mul_op(g, coeff);
}

TensorImplPtr erfinv_op(const TensorImplPtr& a) {
    return ErfinvBackward::forward(a);
}
LUCID_REGISTER_OP(ErfinvBackward)

}  // namespace lucid
