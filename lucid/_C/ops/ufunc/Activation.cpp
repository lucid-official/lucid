// lucid/_C/ops/ufunc/Activation.cpp
//
// Implementations of activation backward nodes and forward entry points.
// Activations with complex backward formulas (gelu, elu, selu, mish,
// hard_sigmoid, hard_swish) delegate to the backend dispatcher so that
// platform-specific implementations (Accelerate on CPU, MLX on GPU) can be
// used without duplicating the math here.

#include "Activation.h"

#include <cmath>

#include "../../autograd/Engine.h"
#include "../../autograd/Helpers.h"
#include "../../backend/Dispatcher.h"
#include "../../core/Allocator.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/OpRegistry.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../../kernel/NaryKernel.h"
#include "../bfunc/Add.h"
#include "../bfunc/Compare.h"
#include "../bfunc/Mul.h"
#include "../bfunc/Sub.h"
#include "../gfunc/Gfunc.h"
#include "../utils/Promote.h"
#include "../utils/Select.h"
#include "Arith.h"
#include "Exponential.h"
#include "Hyperbolic.h"

namespace lucid {

// relu — AmpPolicy::KeepInput: ReLU is valid on integer types.
//
// Every activation below that refuses a bool does so by its schema row.  relu
// and relu6 are the ones the reference refuses with a dtype guard
// (``RuntimeError``) and answers every integer width, so they cast to "a
// number"; the rest have no bool kernel (``NotImplementedError``) and no
// integer one either, so they cast to a floating dtype.
const OpSchema ReluBackward::schema_v1 =
    OpSchema{"relu", 1, AmpPolicy::KeepInput, true}.refusing_bool(DtypeRefusal::Guard,
                                                                  CastHint::Number);

// dL/dx = 0 where x <= 0, else dL/dy.  The reference tests ``x <= 0``, so
// a NaN input passes its gradient; testing ``x > 0`` dropped it.
//
// 3.4+ Phase A.1: dispatch to a single fused backend kernel rather than
// composing positive_mask + multiply at the Storage layer.  The prior path
// emitted three MLX ops (greater + astype + multiply) split across two
// Storage boundaries, which MLX did not fuse into a single kernel — leaving
// ~0.79 ms / step of overhead on ResNet-18 vs the reference framework.  The
// backend now exposes :func:`IBackend::relu_backward` which on GPU lowers
// to one ``where(less_equal(x, 0), 0, g)`` call and on CPU to one tight loop.
Storage ReluBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    Shape flat{static_cast<std::int64_t>(n)};
    return backend::Dispatcher::for_device(device_).relu_backward(g, saved_inputs_[0], flat,
                                                                  dtype_);
}

TensorImplPtr ReluBackward::grad_formula_impl(const TensorImplPtr& g,
                                              const TensorImplPtr& x,
                                              const TensorImplPtr&) {
    return zero_derivative_through(
        where_op(less_equal_op(x, zeros_like_op(x)), zeros_like_op(g), g), x);
}

// ── graph-mode derivatives ───────────────────────────────────────────────────
//
// Eight activations had an eager ``grad_formula`` and no
// ``grad_formula_impl``, so ``grad(create_graph=True)`` refused them
// with "Implement grad_formula_impl() to add support".  Each is built
// from ops that are themselves differentiable, which is what makes the
// gradient differentiable in turn.
//
// Each branch is chosen as the reference's vectorised kernels choose it,
// the path any real input takes, and the eager kernels match: a NaN fails
// every comparison, so the comparison decides where it lands.  relu passes
// it the gradient and leaky_relu the slope; relu6 gives it 0, and elu,
// selu and hard_swish take the side whose value is NaN.  At 0, elu and
// selu take the negative side.  A formula that reads ``x`` only through
// those comparisons keeps its result in the graph with derivative zero
// (zero_derivative_through), as the reference's own backward op does.

// leaky_relu'(x) = 1 where x > 0, slope elsewhere.
TensorImplPtr LeakyReluBackward::grad_formula_impl(const TensorImplPtr& g,
                                                   const TensorImplPtr& x,
                                                   const TensorImplPtr&) {
    auto positive = greater_op(x, zeros_like_op(x));
    auto slope = where_op(positive, ones_like_op(x), full_like_op(x, slope_));
    return zero_derivative_through(mul_op(g, slope), x);
}

// softplus'(x) = sigmoid(beta * x); the threshold branch is linear, and
// its derivative is 1, which is what sigmoid already tends to there.
TensorImplPtr SoftplusBackward::grad_formula_impl(const TensorImplPtr& g,
                                                  const TensorImplPtr& x,
                                                  const TensorImplPtr&) {
    return mul_op(g, sigmoid_op(x));
}

// elu'(x) = 1 for x > 0, and alpha*exp(x) elsewhere.
TensorImplPtr EluBackward::grad_formula_impl(const TensorImplPtr& g,
                                             const TensorImplPtr& x,
                                             const TensorImplPtr& out) {
    // Rebuilt from ``x`` rather than read from ``out``: this op does not
    // hand its output to the graph-mode formula, and reading it gave
    // "add: null input tensor".
    (void)out;
    auto positive = greater_op(x, zeros_like_op(x));
    auto negative_side = mul_op(full_like_op(x, alpha_), exp_op(x));
    return mul_op(g, where_op(positive, ones_like_op(x), negative_side));
}

// selu'(x) = scale for x > 0, and scale*alpha*exp(x) elsewhere.  The two
// constants are the paper's, and the forward holds them as literals.
TensorImplPtr SeluBackward::grad_formula_impl(const TensorImplPtr& g,
                                              const TensorImplPtr& x,
                                              const TensorImplPtr&) {
    constexpr double kAlpha = 1.6732632423543772;
    constexpr double kScale = 1.0507009873554805;
    auto positive = greater_op(x, zeros_like_op(x));
    auto upper = full_like_op(x, kScale);
    auto lower = mul_op(full_like_op(x, kScale * kAlpha), exp_op(x));
    return mul_op(g, where_op(positive, upper, lower));
}

// mish(x) = x * tanh(softplus(x)).  Product rule, with sigmoid the
// derivative of the inner softplus:
//   mish' = tanh(sp) + x * (1 - tanh(sp)^2) * sigmoid(x)
TensorImplPtr MishBackward::grad_formula_impl(const TensorImplPtr& g,
                                              const TensorImplPtr& x,
                                              const TensorImplPtr&) {
    auto t = tanh_op(softplus_op(x));
    auto sech2 = sub_op(ones_like_op(x), mul_op(t, t));
    auto second = mul_op(mul_op(x, sech2), sigmoid_op(x));
    return mul_op(g, add_op(t, second));
}

// hard_sigmoid is piecewise linear: 1/6 inside the band, 0 outside.
TensorImplPtr HardSigmoidBackward::grad_formula_impl(const TensorImplPtr& g,
                                                     const TensorImplPtr& x,
                                                     const TensorImplPtr&) {
    // ``where`` rather than multiplying by the comparison: a compare
    // returns bool and ``mul`` refuses to mix that with a float.
    auto above_low = greater_op(x, full_like_op(x, -3.0));
    auto below_high = less_op(x, full_like_op(x, 3.0));
    auto slope =
        where_op(above_low, where_op(below_high, full_like_op(x, 1.0 / 6.0), zeros_like_op(x)),
                 zeros_like_op(x));
    return zero_derivative_through(mul_op(g, slope), x);
}

// hard_swish(x) = x * hard_sigmoid(x), so inside the band the derivative
// is (2x + 3)/6; it is 0 at and below -3 and 1 from 3 up, as the eager
// kernel has it.  This one said ``x < -3`` and ``x > 3``, so the two
// paths disagreed at the bounds.
TensorImplPtr HardSwishBackward::grad_formula_impl(const TensorImplPtr& g,
                                                   const TensorImplPtr& x,
                                                   const TensorImplPtr&) {
    auto low = less_equal_op(x, full_like_op(x, -3.0));
    auto high = greater_equal_op(x, full_like_op(x, 3.0));
    auto inside_value = mul_op(add_op(mul_op(x, full_like_op(x, 2.0)), full_like_op(x, 3.0)),
                               full_like_op(x, 1.0 / 6.0));
    auto slope = where_op(low, zeros_like_op(x), where_op(high, ones_like_op(x), inside_value));
    return mul_op(g, slope);
}

// relu6'(x) = 1 strictly inside (0, 6), 0 elsewhere — NaN included.
TensorImplPtr Relu6Backward::grad_formula_impl(const TensorImplPtr& g,
                                               const TensorImplPtr& x,
                                               const TensorImplPtr&) {
    auto above_low = greater_op(x, zeros_like_op(x));
    auto below_high = less_op(x, full_like_op(x, 6.0));
    auto slope = where_op(above_low, where_op(below_high, ones_like_op(x), zeros_like_op(x)),
                          zeros_like_op(x));
    return zero_derivative_through(mul_op(g, slope), x);
}

TensorImplPtr relu_op(const TensorImplPtr& a) {
    return ReluBackward::forward(a);
}
LUCID_REGISTER_OP(ReluBackward)

// sigmoid — AmpPolicy::Promote: output requires float for stability.
const OpSchema SigmoidBackward::schema_v1{"sigmoid", 1, AmpPolicy::Promote, true, "", true};

// dL/dx = z*(1-z) * dL/dy  where z = saved_output_.
// Building (1-z) via mul_scalar(-1) + add_scalar(1) avoids a dedicated
// subtract kernel, reusing storage primitives already in the hot path.
Storage SigmoidBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);

    Storage neg_z = mul_scalar_storage(saved_output_, -1.0, n, dtype_, device_);
    Storage one_m_z = add_scalar_storage(neg_z, 1.0, n, dtype_, device_);
    Storage z_omz = multiply_storages(saved_output_, one_m_z, n, dtype_, device_);
    return multiply_storages(z_omz, g, n, dtype_, device_);
}

TensorImplPtr SigmoidBackward::grad_formula_impl(const TensorImplPtr& g,
                                                 const TensorImplPtr&,
                                                 const TensorImplPtr& out) {
    // dx = out*(1-out)*g
    auto one_minus_out = sub_op(ones_like_op(out), out);
    return mul_op(mul_op(out, one_minus_out), g);
}

TensorImplPtr sigmoid_op(const TensorImplPtr& a) {
    return SigmoidBackward::forward(a);
}
LUCID_REGISTER_OP(SigmoidBackward)

// silu — AmpPolicy::Promote; gradient derived analytically from y = x*σ(x).
const OpSchema SiluBackward::schema_v1 =
    OpSchema{"silu", 1, AmpPolicy::Promote, true, "", true}.refusing(kBoolAndSignedIntegers,
                                                                     DtypeRefusal::Kernel);

// dL/dx = σ(x) * (1 + x*(1 - σ(x))) * dL/dy.  Delegates to the backend so
// GPU can dispatch to a single fused kernel (MLX expression or MPSGraph)
// rather than the prior 7-op storage-primitive composition.
Storage SiluBackward::grad_formula(const Storage& g) {
    return backend::Dispatcher::for_device(device_).silu_backward(saved_inputs_[0], g, out_shape_,
                                                                  dtype_);
}

TensorImplPtr SiluBackward::grad_formula_impl(const TensorImplPtr& g,
                                              const TensorImplPtr& x,
                                              const TensorImplPtr&) {
    // dx = σ(x) * (1 + x*(1 - σ(x))) * g — the same expression the fused
    // kernel evaluates, written in ops so it keeps a graph of its own.
    auto s = sigmoid_op(x);
    auto one = ones_like_op(s);
    auto inner = add_op(one, mul_op(x, sub_op(one, s)));
    return mul_op(g, mul_op(s, inner));
}

TensorImplPtr silu_op(const TensorImplPtr& a) {
    return SiluBackward::forward(a);
}
LUCID_REGISTER_OP(SiluBackward)

// gelu — ForceFP32 because the tanh approximation used inside the backend is
// not numerically safe in half precision.
const OpSchema GeluBackward::schema_v1 = OpSchema{"gelu", 1, AmpPolicy::ForceFP32, true}.refusing(
    kBoolAndSignedIntegers, DtypeRefusal::Kernel);

// Delegate to the backend; the exact formula involves the Gaussian CDF
// 0.5*(1 + erf(x/sqrt(2))) and its derivative.
Storage GeluBackward::grad_formula(const Storage& g) {
    return backend::Dispatcher::for_device(device_).gelu_backward(saved_inputs_[0], g, out_shape_,
                                                                  dtype_);
}

TensorImplPtr GeluBackward::grad_formula_impl(const TensorImplPtr& g,
                                              const TensorImplPtr& x,
                                              const TensorImplPtr&) {
    // y = 0.5x(1 + tanh(u)) with u = c(x + a x^3), c = sqrt(2/pi).
    // dy/dx = 0.5(1 + tanh u) + 0.5x (1 - tanh^2 u) u',  u' = c(1 + 3a x^2).
    constexpr double kC = 0.7978845608028654;  // sqrt(2 / pi)
    constexpr double kA = 0.044715;

    auto x_sq = square_op(x);
    auto one = ones_like_op(x);
    auto u = mul_op(full_like_op(x, kC), add_op(x, mul_op(full_like_op(x, kA), mul_op(x_sq, x))));
    auto th = tanh_op(u);
    auto sech_sq = sub_op(ones_like_op(th), square_op(th));
    auto du = mul_op(full_like_op(x, kC), add_op(one, mul_op(full_like_op(x, 3.0 * kA), x_sq)));

    auto half = full_like_op(x, 0.5);
    auto derivative = add_op(mul_op(half, add_op(ones_like_op(th), th)),
                             mul_op(mul_op(half, x), mul_op(sech_sq, du)));
    return mul_op(g, derivative);
}

TensorImplPtr gelu_op(const TensorImplPtr& a) {
    return GeluBackward::forward(a);
}
LUCID_REGISTER_OP(GeluBackward)

// gelu_exact — Gaussian-CDF formulation (exact erf-based).  ForceFP32
// matches the tanh-approx variant for consistent AMP behaviour.
const OpSchema GeluExactBackward::schema_v1 =
    OpSchema{"gelu_exact", 1, AmpPolicy::ForceFP32, true}.refusing(kBoolAndSignedIntegers,
                                                                   DtypeRefusal::Kernel);

Storage GeluExactBackward::grad_formula(const Storage& g) {
    return backend::Dispatcher::for_device(device_).gelu_exact_backward(saved_inputs_[0], g,
                                                                        out_shape_, dtype_);
}

TensorImplPtr GeluExactBackward::grad_formula_impl(const TensorImplPtr& g,
                                                   const TensorImplPtr& x,
                                                   const TensorImplPtr&) {
    // dy/dx = Phi(x) + x * phi(x), with Phi the Gaussian CDF and phi its
    // density — both reachable from erf and exp, which carry graph-mode
    // formulas of their own.
    constexpr double kInvSqrt2 = 0.7071067811865476;
    constexpr double kInvSqrt2Pi = 0.3989422804014327;

    auto cdf = mul_op(full_like_op(x, 0.5),
                      add_op(ones_like_op(x), erf_op(mul_op(x, full_like_op(x, kInvSqrt2)))));
    auto pdf =
        mul_op(full_like_op(x, kInvSqrt2Pi), exp_op(mul_op(full_like_op(x, -0.5), square_op(x))));
    return mul_op(g, add_op(cdf, mul_op(x, pdf)));
}

TensorImplPtr gelu_exact_op(const TensorImplPtr& a) {
    return GeluExactBackward::forward(a);
}
LUCID_REGISTER_OP(GeluExactBackward)

// leaky_relu — KeepInput (valid for integer slopes).
const OpSchema LeakyReluBackward::schema_v1 =
    OpSchema{"leaky_relu", 1, AmpPolicy::KeepInput, true, "", true}.refusing(kBoolAndSignedIntegers,
                                                                             DtypeRefusal::Kernel);

// dL/dx = (x > 0 ? 1 : slope_) * dL/dy  (leaky mask).
Storage LeakyReluBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    Storage mask = leaky_mask_storage(saved_inputs_[0], slope_, n, dtype_, device_);
    return multiply_storages(g, mask, n, dtype_, device_);
}

// Custom forward: the slope parameter must be captured on the backward node,
// which is not possible through the standard dispatch() path.
TensorImplPtr LeakyReluBackward::forward(const TensorImplPtr& a, double slope) {
    Validator::input(a, "leaky_relu.a").non_null();

    // Assembles its own forward, so it asks for the schema dtype the
    // kernel templates would have applied.  See promote_for_schema.
    const TensorImplPtr x = promote_for_schema(schema_v1, a);

    OpScopeFull scope{schema_v1.name, x->device(), x->dtype(), x->shape()};
    scope.set_attr("slope", slope);
    Storage out_storage = backend::Dispatcher::for_device(x->device())
                              .leaky_relu(x->storage(), x->shape(), x->dtype(), slope);
    auto out = std::make_shared<TensorImpl>(std::move(out_storage), x->shape(), x->dtype(),
                                            x->device(), false);
    scope.set_flops(static_cast<std::int64_t>(x->numel()));

    auto bwd = std::make_shared<LeakyReluBackward>();
    bwd->slope_ = slope;
    kernel::NaryKernel<LeakyReluBackward, 1>::wire_autograd(std::move(bwd), {x}, out);
    return out;
}

TensorImplPtr leaky_relu_op(const TensorImplPtr& a, double slope) {
    return LeakyReluBackward::forward(a, slope);
}
LUCID_REGISTER_OP(LeakyReluBackward)

// softplus — ForceFP32 because log(1 + e^x) overflows in float16.
const OpSchema SoftplusBackward::schema_v1 =
    OpSchema{"softplus", 1, AmpPolicy::ForceFP32, true}.refusing(kSignedIntegers,
                                                                 DtypeRefusal::Kernel);

// dL/dx = sigmoid(x) * dL/dy  (derivative of log(1 + e^x) is sigmoid(x)).
Storage SoftplusBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    Storage sx = sigmoid_storage(saved_inputs_[0], n, dtype_, device_);
    return multiply_storages(sx, g, n, dtype_, device_);
}

TensorImplPtr softplus_op(const TensorImplPtr& a) {
    return SoftplusBackward::forward(a);
}
LUCID_REGISTER_OP(SoftplusBackward)

// elu — ForceFP32; backward depends on alpha_ captured in the node.
const OpSchema EluBackward::schema_v1 = OpSchema{"elu", 1, AmpPolicy::ForceFP32, true}.refusing(
    kBoolAndSignedIntegers, DtypeRefusal::Kernel);

// Delegate to the backend because the piecewise formula (1 if x>=0 else
// alpha*e^x) requires a conditional that the generic storage primitives do
// not express without an explicit branch kernel.
Storage EluBackward::grad_formula(const Storage& g) {
    return backend::Dispatcher::for_device(device_).elu_backward(saved_inputs_[0], g, out_shape_,
                                                                 dtype_, alpha_);
}

// Custom forward: alpha_ must be captured on the backward node.
TensorImplPtr EluBackward::forward(const TensorImplPtr& a, double alpha) {
    Validator::input(a, "elu.a").non_null();

    // Assembles its own forward, so it asks for the schema dtype the
    // kernel templates would have applied.  See promote_for_schema.
    const TensorImplPtr x = promote_for_schema(schema_v1, a);

    OpScopeFull scope{schema_v1.name, x->device(), x->dtype(), x->shape()};
    scope.set_attr("alpha", alpha);
    Storage out_storage = backend::Dispatcher::for_device(x->device())
                              .elu(x->storage(), x->shape(), x->dtype(), alpha);
    auto out = std::make_shared<TensorImpl>(std::move(out_storage), x->shape(), x->dtype(),
                                            x->device(), false);
    scope.set_flops(static_cast<std::int64_t>(x->numel()));

    auto bwd = std::make_shared<EluBackward>();
    bwd->alpha_ = alpha;
    kernel::NaryKernel<EluBackward, 1>::wire_autograd(std::move(bwd), {x}, out);
    return out;
}

TensorImplPtr elu_op(const TensorImplPtr& a, double alpha) {
    return EluBackward::forward(a, alpha);
}
LUCID_REGISTER_OP(EluBackward)

// selu — ForceFP32; fixed α/λ constants embedded in the backend.
const OpSchema SeluBackward::schema_v1 = OpSchema{"selu", 1, AmpPolicy::ForceFP32, true}.refusing(
    kBoolAndSignedIntegers, DtypeRefusal::Kernel);

// Delegate; the backend applies the standard SELU backward with its fixed
// self-normalising constants.
Storage SeluBackward::grad_formula(const Storage& g) {
    return backend::Dispatcher::for_device(device_).selu_backward(saved_inputs_[0], g, out_shape_,
                                                                  dtype_);
}

TensorImplPtr selu_op(const TensorImplPtr& a) {
    return SeluBackward::forward(a);
}
LUCID_REGISTER_OP(SeluBackward)

// mish — ForceFP32; composed function requires tanh and softplus in backward.
const OpSchema MishBackward::schema_v1 = OpSchema{"mish", 1, AmpPolicy::ForceFP32, true}.refusing(
    kBoolAndSignedIntegers, DtypeRefusal::Kernel);

// Delegate; the backend computes d/dx[x*tanh(log(1+e^x))] which involves the
// product rule applied to tanh(softplus(x)).
Storage MishBackward::grad_formula(const Storage& g) {
    return backend::Dispatcher::for_device(device_).mish_backward(saved_inputs_[0], g, out_shape_,
                                                                  dtype_);
}

TensorImplPtr mish_op(const TensorImplPtr& a) {
    return MishBackward::forward(a);
}
LUCID_REGISTER_OP(MishBackward)

// hard_sigmoid — KeepInput; piecewise linear, valid for float inputs.
const OpSchema HardSigmoidBackward::schema_v1 =
    OpSchema{"hard_sigmoid", 1, AmpPolicy::KeepInput, true, "", true}.refusing(
        kBoolAndSignedIntegers, DtypeRefusal::Kernel);

// Delegate; the backend returns 1/6 inside the active region, 0 outside.
Storage HardSigmoidBackward::grad_formula(const Storage& g) {
    return backend::Dispatcher::for_device(device_).hard_sigmoid_backward(saved_inputs_[0], g,
                                                                          out_shape_, dtype_);
}

TensorImplPtr hard_sigmoid_op(const TensorImplPtr& a) {
    return HardSigmoidBackward::forward(a);
}
LUCID_REGISTER_OP(HardSigmoidBackward)

// hard_swish — KeepInput; piecewise linear backward.
const OpSchema HardSwishBackward::schema_v1 =
    OpSchema{"hard_swish", 1, AmpPolicy::KeepInput, true, "", true}.refusing(kBoolAndSignedIntegers,
                                                                             DtypeRefusal::Kernel);

// Delegate; the backend handles the three-region piecewise formula.
Storage HardSwishBackward::grad_formula(const Storage& g) {
    return backend::Dispatcher::for_device(device_).hard_swish_backward(saved_inputs_[0], g,
                                                                        out_shape_, dtype_);
}

TensorImplPtr hard_swish_op(const TensorImplPtr& a) {
    return HardSwishBackward::forward(a);
}
LUCID_REGISTER_OP(HardSwishBackward)

// relu6 — KeepInput; gradient is non-zero only in the open interval (0, 6).
const OpSchema Relu6Backward::schema_v1 =
    OpSchema{"relu6", 1, AmpPolicy::KeepInput, true}.refusing_bool(DtypeRefusal::Guard,
                                                                   CastHint::Number);

// dL/dx = (0 < x < 6) * dL/dy  using an in-range boolean mask.
Storage Relu6Backward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    // Exclusive, unlike ``clip``: the reference's ``hardtanh`` backward
    // returns 0 at either bound, and ``grad_formula_impl`` above already
    // did.  This used the inclusive mask, so the same gradient came out
    // as 1 through ``backward()`` and 0 through ``grad(create_graph=True)``.
    Storage mask = in_range_mask_storage(saved_inputs_[0], 0.0, 6.0, n, dtype_, device_, false);
    return multiply_storages(g, mask, n, dtype_, device_);
}

TensorImplPtr relu6_op(const TensorImplPtr& a) {
    return Relu6Backward::forward(a);
}
LUCID_REGISTER_OP(Relu6Backward)

}  // namespace lucid
