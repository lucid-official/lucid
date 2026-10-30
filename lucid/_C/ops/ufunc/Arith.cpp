// lucid/_C/ops/ufunc/Arith.cpp
//
// Implementations of the six basic arithmetic unary backward nodes: neg, abs,
// sign, reciprocal, square, cube.  Each section defines the static OpSchema,
// implements grad_formula, provides the public entry-point wrapper, and
// registers the op in the global OpRegistry via LUCID_REGISTER_OP.

#include "Arith.h"

#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/OpRegistry.h"
#include "../../kernel/NaryKernel.h"
#include "../bfunc/Add.h"
#include "../bfunc/Compare.h"
#include "../bfunc/Div.h"
#include "../bfunc/Mul.h"
#include "../complex/Complex.h"
#include "../complex/Imag.h"
#include "../complex/Real.h"
#include "../composite/Math.h"
#include "../gfunc/Gfunc.h"
#include "../utils/Select.h"
#include "Astype.h"
#include "ScalarParam.h"

namespace lucid {

// neg — AmpPolicy::Promote promotes integer inputs to float before dispatch.
const OpSchema NegBackward::schema_v1{"neg", 1, AmpPolicy::Promote, true};

// dL/dx = -dL/dy: negate the upstream gradient in-place over the output shape.
Storage NegBackward::grad_formula(const Storage& g) {
    return negate_storage(g, shape_numel(out_shape_), dtype_, device_);
}

TensorImplPtr
NegBackward::grad_formula_impl(const TensorImplPtr& g, const TensorImplPtr&, const TensorImplPtr&) {
    return neg_op(g);
}

TensorImplPtr neg_op(const TensorImplPtr& a) {
    // A bool has no negation.  The CPU refused with "dtype not supported"
    // and Metal passed MLX's own ValueError through; both now say what to
    // use instead, as the reference framework does.
    if (a && a->dtype() == Dtype::Bool)
        ErrorBuilder("neg").fail("negation, the `-` operator, on a bool tensor is not supported; "
                                 "use `~` or logical_not() to invert a mask");
    return NegBackward::forward(a);
}
LUCID_REGISTER_OP(NegBackward)

// abs — saves input so grad_formula can recompute sign(x).
const OpSchema AbsBackward::schema_v1{"abs", 1, AmpPolicy::Promote, true};

// dL/dx = sign(x) * dL/dy.
Storage AbsBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    Storage s = sign_storage(saved_inputs_[0], n, dtype_, device_);
    return multiply_storages(g, s, n, dtype_, device_);
}

TensorImplPtr abs_op(const TensorImplPtr& a) {
    if (!a)
        ErrorBuilder("abs").fail("null input tensor");
    // Complex magnitude |z| = sqrt(Re(z)² + Im(z)²) is a complex→REAL map, so
    // it cannot go through UnaryKernel: that path tags the output with the
    // *input* dtype, which left the real result labelled C64 — the reader then
    // paired consecutive magnitudes into (re, im) and ran off the end of the
    // buffer.  Compose from the real/imag primitives instead, exactly as the
    // sibling `angle` composite does.  Like real/imag/angle (and unlike abs on
    // a real tensor, which keeps its full UnaryKernel gradient), this path is
    // forward-only — Lucid has no complex autograd.
    //
    // Cost: ~20 element-wise kernels (``hypot_lanes`` plus the two lane
    // views) instead of one native complex-abs.  Going native would need a
    // UnaryKernel hook for a complex->real output dtype *and* a hand-written
    // C64 branch in the CPU backend; reusing ops that are already
    // device-parity tested is the better trade until a profile says
    // otherwise.
    if (is_complex(a->dtype()))
        return complex_abs_forward(a);
    return AbsBackward::forward(a);
}
const OpSchema ComplexAbsBackward::schema_v1{"abs", 1, AmpPolicy::KeepInput, true};

// ``|z|`` with the composite kept for the value and a stated derivative.
//
// The composite runs under ``NoGradGuard`` so its intermediate divisions
// leave no nodes behind; the single node wired here owns the derivative.
TensorImplPtr complex_abs_forward(const TensorImplPtr& a) {
    TensorImplPtr magnitude;
    {
        NoGradGuard nograd;
        // |z| is the length of the (Re, Im) pair.  ``hypot_lanes`` owns the
        // rule — overflow-safe, subnormal-exact, +inf for an infinite lane
        // even beside a NaN — so this and ``hypot`` cannot disagree.
        magnitude = hypot_lanes(real_op(a), imag_op(a));
    }
    if (!GradMode::is_enabled() || !a->requires_grad())
        return magnitude;

    // ``z`` is saved by value: the backward reads its values, and holding the
    // tensor instead read whatever a later in-place write left in it.
    kernel::NaryKernel<ComplexAbsBackward, 1>::wire_autograd(std::make_shared<ComplexAbsBackward>(),
                                                             {a}, magnitude);
    return magnitude;
}

namespace {

// ``g * z / |z|``, lane by lane, with the quotient forced to zero where the
// magnitude is — the direction of a zero vector is not defined and the
// reference reports 0.  Built from ops so graph mode can record it.
TensorImplPtr complex_abs_grad(const TensorImplPtr& g, const TensorImplPtr& z) {
    auto re = real_op(z);
    auto im = imag_op(z);
    auto mag = complex_abs_forward(z);
    auto zero = zeros_like_op(mag);
    auto nonzero = greater_op(mag, zero);
    auto safe = where_op(nonzero, mag, ones_like_op(mag));
    auto scale = where_op(nonzero, div_op(g, safe), zero);
    return complex_op(mul_op(re, scale), mul_op(im, scale));
}

}  // namespace

std::vector<Storage> ComplexAbsBackward::apply(Storage grad_out) {
    NoGradGuard nograd;
    const auto z =
        std::make_shared<TensorImpl>(saved_inputs_[0], input_shapes_[0], dtype_, device_, false);
    auto g = std::make_shared<TensorImpl>(std::move(grad_out), z->shape(), real_lane_of(z->dtype()),
                                          z->device(), false);
    return {complex_abs_grad(g, z)->storage()};
}

std::vector<TensorImplPtr> ComplexAbsBackward::apply_for_graph(const TensorImplPtr& grad_out) {
    const auto& z = saved_impl_inputs_[0];
    if (!z)
        ErrorBuilder("abs").fail("graph-mode backward is missing its saved input");
    return {complex_abs_grad(grad_out, z)};
}

LUCID_REGISTER_OP(ComplexAbsBackward)

TensorImplPtr AbsBackward::grad_formula_impl(const TensorImplPtr& g,
                                             const TensorImplPtr& x,
                                             const TensorImplPtr&) {
    // |x|' = sign(x), which is 0 at the kink — the same
    // subgradient the eager path takes.
    return mul_op(g, sign_op(x));
}

LUCID_REGISTER_OP(AbsBackward)

// sign — AmpPolicy::KeepInput preserves the input dtype (sign is valid on
// integers).  kHasGradient = false means UnaryKernel never wires autograd, so
// grad_formula is only called if someone constructs a backward node manually;
// the returned empty CpuStorage acts as a zero-gradient sentinel.
const OpSchema SignBackward::schema_v1{"sign", 1, AmpPolicy::KeepInput, true};

// Gradient of sign is zero almost everywhere (discontinuous at 0).
Storage SignBackward::grad_formula(const Storage&) {
    return make_zero_storage(out_shape_, dtype_, device_);
}

TensorImplPtr sign_op(const TensorImplPtr& a) {
    return SignBackward::forward(a);
}
TensorImplPtr SignBackward::grad_formula_impl(const TensorImplPtr& g,
                                              const TensorImplPtr&,
                                              const TensorImplPtr&) {
    // A step function is flat wherever it is differentiable.
    return zeros_like_op(g);
}

LUCID_REGISTER_OP(SignBackward)

// reciprocal — saves input to compute x^2 in the backward pass.
const OpSchema ReciprocalBackward::schema_v1{"reciprocal", 1, AmpPolicy::Promote, true, "", true};

// dL/dx = -dL/dy / x^2.
Storage ReciprocalBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);

    Storage x_sq = square_storage(saved_inputs_[0], n, dtype_, device_);
    Storage g_div = divide_storages(g, x_sq, n, dtype_, device_);
    return negate_storage(g_div, n, dtype_, device_);
}

TensorImplPtr ReciprocalBackward::grad_formula_impl(const TensorImplPtr& g,
                                                    const TensorImplPtr& x,
                                                    const TensorImplPtr&) {
    // dx = -g / x^2
    auto x_sq = mul_op(x, x);
    return neg_op(div_op(g, x_sq));
}

TensorImplPtr reciprocal_op(const TensorImplPtr& a) {
    return ReciprocalBackward::forward(a);
}
LUCID_REGISTER_OP(ReciprocalBackward)

// square — saves input to compute 2*x in the backward pass.
const OpSchema SquareBackward::schema_v1{"square", 1, AmpPolicy::Promote, true};

// dL/dx = 2*x * dL/dy.
Storage SquareBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);

    Storage two_x = mul_scalar_storage(saved_inputs_[0], 2.0, n, dtype_, device_);
    return multiply_storages(two_x, g, n, dtype_, device_);
}

TensorImplPtr SquareBackward::grad_formula_impl(const TensorImplPtr& g,
                                                const TensorImplPtr& x,
                                                const TensorImplPtr&) {
    // dx = 2*x * g = (x+x) * g
    return mul_op(add_op(x, x), g);
}

namespace {

// ``square`` and ``cube`` are ``x ** 2`` and ``x ** 3``, and a power of a bool
// is an int64 in the reference — ``x * x`` is a logical and, a different op.
TensorImplPtr power_operand(const TensorImplPtr& a) {
    return a->dtype() == Dtype::Bool ? astype_op(a, Dtype::I64) : a;
}

}  // namespace

TensorImplPtr square_op(const TensorImplPtr& a) {
    return SquareBackward::forward(power_operand(a));
}
LUCID_REGISTER_OP(SquareBackward)

// cube — saves input to compute 3*x^2 in the backward pass.
const OpSchema CubeBackward::schema_v1{"cube", 1, AmpPolicy::Promote, true};

// dL/dx = 3*x^2 * dL/dy.
Storage CubeBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);

    Storage x_sq = square_storage(saved_inputs_[0], n, dtype_, device_);
    Storage three_xsq = mul_scalar_storage(x_sq, 3.0, n, dtype_, device_);
    return multiply_storages(three_xsq, g, n, dtype_, device_);
}

TensorImplPtr CubeBackward::grad_formula_impl(const TensorImplPtr& g,
                                              const TensorImplPtr& x,
                                              const TensorImplPtr&) {
    return mul_op(g, mul_op(full_like_op(x, 3.0), mul_op(x, x)));
}

TensorImplPtr cube_op(const TensorImplPtr& a) {
    return CubeBackward::forward(power_operand(a));
}
LUCID_REGISTER_OP(CubeBackward)

}  // namespace lucid
