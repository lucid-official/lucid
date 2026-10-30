// lucid/_C/ops/composite/Math.cpp
//
// Composition-based elementwise math ops.  Each entry function builds its
// forward graph from existing differentiable ``*_op`` calls; the chain rule
// is provided automatically by the primitives' backward nodes (LogBackward,
// ExpBackward, MulBackward, etc.) so no new schemas are registered here.
//
// Each opens with an :class:`AmpScope` under its own name, so autocast
// applies the composite's row once to its inputs and the primitives inside
// compute in that dtype.

#include "Math.h"

#include <cmath>
#include <limits>
#include <string>
#include <utility>

#include "../../core/Dtype.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/TensorImpl.h"
#include "../bfunc/Add.h"
#include "../bfunc/Compare.h"
#include "../bfunc/Div.h"
#include "../bfunc/Maximum.h"
#include "../bfunc/Minimum.h"
#include "../bfunc/Mul.h"
#include "../bfunc/Pow.h"
#include "../bfunc/Sub.h"
#include "../gfunc/Gfunc.h"
#include "../ufunc/Arith.h"
#include "../ufunc/Astype.h"
#include "../ufunc/Discrete.h"
#include "../ufunc/Exponential.h"
#include "../ufunc/Predicate.h"
#include "../ufunc/Trig.h"
#include "../utils/Nextafter.h"
#include "../utils/Promote.h"
#include "../utils/Select.h"
#include "Logical.h"

namespace lucid {

namespace {

// The operand of a unary composite, null-checked before its device is read.
const TensorImplPtr& unary_input(const TensorImplPtr& a, const char* op) {
    if (!a)
        ErrorBuilder(op).fail("null input");
    return a;
}

// The two operands of a binary composite, cast together by its scope.
std::pair<TensorImplPtr, TensorImplPtr>
cast_operands(const AmpScope& amp, const TensorImplPtr& a, const TensorImplPtr& b) {
    auto cast = amp.cast_widest({a, b});
    return {std::move(cast[0]), std::move(cast[1])};
}

}  // namespace

// ── unary compositions ──────────────────────────────────────────────────────

// ``log10(x) = log(x) / log(10)``.  Materialises ``log(10)`` as a same-shape
// constant so the divide kernel does not need scalar broadcasting.
TensorImplPtr log10_op(const TensorImplPtr& a_in) {
    const AmpScope amp{"log10", unary_input(a_in, "log10")->device()};
    const TensorImplPtr a = amp.cast(a_in);
    auto log_a = log_op(a);
    auto ln10 = full_like_op(log_a, std::log(10.0));
    return div_op(log_a, ln10);
}

// ``log1p(x) = log(1 + x)`` — but not by forming ``1 + x`` and taking the
// log of that, which is where the information goes.  Below the machine
// epsilon ``1 + x`` rounds to exactly 1 and the answer comes back 0 where
// it should be x: ``log1p(1e-30)`` was 0.0 against the reference's 1e-30,
// and everything under about 1e-8 had lost all of its digits.
//
// The standard repair takes the log at the rounded operand and rescales
// by the increment the rounding actually applied:
//
//     u = 1 + x;   log1p(x) = (u - 1 == 0) ? x : log(u) · x / (u - 1)
//
// ``u - 1`` recovers exactly what the addition put in — it is not
// algebraically ``x``, and that difference is the correction.
//
// The rescaling is a value correction only.  Differentiated, it is
// 0/0 or inf/inf wherever ``log(u)`` is not finite: the value at +inf was
// NaN (inf · inf / inf), and the gradient was NaN at x = -1 and below,
// where the derivative is 1 / (1 + x) — +inf at -1, finite beyond.  So
// the graph carries ``log(u)``, whose backward is exactly 1 / (1 + x),
// and the correction is added outside autograd, and only where
// ``log(u)`` is finite; elsewhere ``log(u)`` already is the answer
// (+inf, -inf at -1, NaN below -1 and at NaN).  ``x / (u - 1)`` is
// divided before it multiplies, since it is about 1 and ``log(u) · x``
// overflows float32 from x ≈ 4e36.
TensorImplPtr log1p_op(const TensorImplPtr& a_in) {
    const AmpScope amp{"log1p", unary_input(a_in, "log1p")->device()};
    const TensorImplPtr a = amp.cast(a_in);
    auto ones = full_like_op(a, 1.0);
    auto u = add_op(a, ones);
    auto plain = log_op(u);
    TensorImplPtr vanished;
    TensorImplPtr correction;
    {
        NoGradGuard no_grad;
        auto increment = sub_op(u, ones);
        vanished = equal_op(increment, full_like_op(increment, 0.0));
        auto safe = where_op(vanished, ones, increment);
        auto accurate = mul_op(plain, div_op(a, safe));
        correction =
            where_op(isfinite_op(plain), sub_op(accurate, plain), full_like_op(plain, 0.0));
    }
    // Where the increment vanished, x is already the answer to within one
    // ulp, its derivative 1 / (1 + x) rounds to 1, and taking x keeps
    // log1p(-0) = -0.
    return where_op(vanished, a, add_op(plain, correction));
}

// ``exp2(x) = 2**x``.  Routing through ``pow_op(2, a)`` reaches the exact
// base-2 power path on CPU and Metal (so integer inputs are exact powers of
// two) while keeping the gradient chain visible to ``PowBackward``.
TensorImplPtr exp2_op(const TensorImplPtr& a_in) {
    const AmpScope amp{"exp2", unary_input(a_in, "exp2")->device()};
    const TensorImplPtr a = amp.cast(a_in);
    return pow_op(full_op({}, 2.0, a->dtype(), a->device()), a);
}

// Fractional part — the ``trunc`` term contributes zero gradient, so the
// gradient w.r.t. ``a`` is just the upstream gradient.
TensorImplPtr frac_op(const TensorImplPtr& a_in) {
    const AmpScope amp{"frac", unary_input(a_in, "frac")->device()};
    const TensorImplPtr a = amp.cast(a_in);
    return sub_op(a, trunc_op(a));
}

// ── binary compositions ─────────────────────────────────────────────────────

namespace {

// IEEE-754 sign bit of a floating-point tensor ``t`` as a Bool mask (true for
// negative values, ``-0.0``, and ``-nan``).  Under an active compile trace,
// ``_bitcast`` is not traced, so the sign of zero is read via ``1 / t < 0``.
TensorImplPtr float_signbit_mask(const TensorImplPtr& t) {
    const auto zero = full_like_op(t, 0.0);
    if (compile::current_tracer() != nullptr || t->is_phantom()) {
        return logical_or_op(less_op(t, zero),
                             logical_and_op(equal_op(t, zero), less_op(reciprocal_op(t), zero)));
    }
    NoGradGuard no_grad;
    const Dtype idt = (t->dtype() == Dtype::F16 || t->dtype() == Dtype::BF16)
                          ? Dtype::I16
                          : (t->dtype() == Dtype::F32 ? Dtype::I32 : Dtype::I64);
    const auto bits = bitcast_op(t, idt);
    return less_op(bits, full_like_op(bits, 0.0));
}

}  // namespace

// Quadrant-aware arctangent.  Patches ``arctan(y / x)`` with ±π/±0/±π/2/±π/4/±3π/4
// branches via ``where`` based on the IEEE-754 sign bits of the operands; the
// divide is guarded by substituting ±1 for x = ±0 or ±inf and ±1 for y = ±inf
// so the safe branch is always finite.
TensorImplPtr atan2_op(const TensorImplPtr& y_in, const TensorImplPtr& x_in) {
    constexpr double pi = 3.14159265358979323846;
    if (!y_in || !x_in)
        ErrorBuilder("atan2").fail("null input");
    const auto [y_dev, x_dev] = detail::align_binary_devices(y_in, x_in, "atan2");
    const AmpScope amp{"atan2", y_dev->device()};
    const auto [y_amp, x_amp] = cast_operands(amp, y_dev, x_dev);
    auto to_float = [](const TensorImplPtr& t) {
        return (!is_floating_point(t->dtype()) && !is_complex(t->dtype()))
                   ? astype_op(t, Dtype::F32)
                   : t;
    };
    const TensorImplPtr y = to_float(y_amp);
    const TensorImplPtr x = to_float(x_amp);

    const auto zero_x = full_like_op(x, 0.0);
    const auto zero_y = full_like_op(y, 0.0);
    const auto ones_x = full_like_op(x, 1.0);
    const auto ones_y = full_like_op(y, 1.0);

    const auto x_neg = float_signbit_mask(x);
    const auto y_neg = float_signbit_mask(y);
    const auto x_is_zero = equal_op(x, zero_x);
    const auto y_is_zero = equal_op(y, zero_y);
    const auto x_is_inf = isinf_op(x);
    const auto y_is_inf = isinf_op(y);

    const auto safe_x =
        where_op(logical_or_op(x_is_zero, x_is_inf), where_op(x_neg, neg_op(ones_x), ones_x), x);
    const auto safe_y = where_op(y_is_inf, where_op(y_neg, neg_op(ones_y), ones_y), y);
    const auto base = arctan_op(div_op(safe_y, safe_x));
    const auto pos_zero = full_like_op(base, 0.0);
    const auto neg_zero = neg_op(pos_zero);
    // Hardware arctan(-0.0) on Metal (and half on CPU) returns +0.0; adding -base
    // detached inside NoGradGuard turns base's +0.0 into 0.0 - (+0.0) = 0.0,
    // whereas selecting neg_zero when grad is not tracked or adding a zero-grad
    // -0.0 shift is needed.  Even simpler: ``sub_op(base, base_detached)`` is +0.0
    // with derivative 1 * base', and ``sub_op(neg_zero, sub_op(base_detached, base))``
    // is ``-0.0 - 0.0 = -0.0`` with derivative ``+base'``!
    TensorImplPtr base_detached;
    {
        NoGradGuard no_grad;
        base_detached = add_op(base, pos_zero);
    }
    const auto neg_zero_with_grad = sub_op(neg_zero, sub_op(base_detached, base));
    const auto base_signed =
        where_op(logical_and_op(logical_and_op(y_is_zero, y_neg), logical_not_op(x_neg)),
                 neg_zero_with_grad, base);

    const auto pi_t = full_like_op(base, pi);
    const auto half_pi = full_like_op(base, 0.5 * pi);
    const auto quarter_pi = full_like_op(base, 0.25 * pi);
    const auto three_quarter_pi = full_like_op(base, 0.75 * pi);

    const auto signed_pi = where_op(y_neg, neg_op(pi_t), pi_t);
    const auto signed_half_pi = where_op(y_neg, neg_op(half_pi), half_pi);
    const auto signed_zero = where_op(y_neg, neg_zero, pos_zero);

    const auto x_neg_branch = where_op(y_neg, sub_op(base_signed, pi_t), add_op(base_signed, pi_t));
    const auto normal = where_op(x_neg, x_neg_branch, base_signed);

    const auto y0_branch = where_op(x_neg, signed_pi, signed_zero);
    const auto inf_inf_mag = where_op(x_neg, three_quarter_pi, quarter_pi);
    const auto inf_inf_branch = where_op(y_neg, neg_op(inf_inf_mag), inf_inf_mag);

    auto out = where_op(logical_and_op(x_is_inf, logical_not_op(y_is_inf)), y0_branch, normal);
    out = where_op(logical_or_op(logical_and_op(x_is_zero, logical_not_op(y_is_zero)),
                                 logical_and_op(y_is_inf, logical_not_op(x_is_inf))),
                   signed_half_pi, out);
    out = where_op(logical_and_op(x_is_inf, y_is_inf), inf_inf_branch, out);
    out = where_op(logical_and_op(x_is_zero, y_is_zero), y0_branch, out);
    return where_op(logical_or_op(isnan_op(x), isnan_op(y)), base, out);
}

namespace {

// ``hypot`` and ``logaddexp`` are defined on real floating values only, as
// in the reference framework.  An integer or bool input used to fail deep
// inside the composition — a ``DtypeMismatch`` from an internal ``mul`` or
// ``add`` against a float constant — which named neither the op nor the
// reason.
void require_floating(const char* op, const TensorImplPtr& a, const TensorImplPtr& b) {
    for (const auto* t : {&a, &b}) {
        if (*t && !is_floating_point((*t)->dtype()))
            ErrorBuilder(op).not_implemented(std::string("not implemented for ") +
                                             std::string(dtype_name((*t)->dtype())) +
                                             "; cast to a floating dtype first");
    }
}

}  // namespace

// Pythagorean length, scaled so it cannot overflow on the way.
//
// The naive ``sqrt(a² + b²)`` squares first, so ``hypot(1e200, 1e200)``
// overflowed to inf while the answer, 1.41e200, is an ordinary double.
// Avoiding exactly that is what distinguishes ``hypot`` from writing the
// formula out, and the previous comment here said so and left it.
//
// Factoring out the larger magnitude gives ``m · sqrt(1 + (n/m)²)`` where
// the ratio is in [0, 1], so nothing squares above 2.  ``logaddexp``
// immediately below already does the same trick for the same reason.
//
// ``m == 0`` means both operands are zero and the answer is zero; the
// ratio would be 0/0, so the divisor is 1 there and the multiply by ``m``
// puts the zero back.  The divisor is ``m`` itself everywhere else, down
// into the subnormal range: an earlier ``max(m, FLT_MIN)`` floor changed
// the divisor but not the multiplier, so ``hypot(1e-40, 1e-40)`` came back
// 1.00004e-40 instead of 1.4142e-40.
//
// An infinite operand makes the result +inf even when the other is NaN
// (C99 Annex G, and the reference), which ``maximum`` above would
// propagate as NaN.  That case is overridden last, so the arithmetic above
// may produce NaN for it (inf/inf) without consequence.
TensorImplPtr hypot_lanes(const TensorImplPtr& a, const TensorImplPtr& b) {
    auto abs_a = abs_op(a);
    auto abs_b = abs_op(b);
    auto big = maximum_op(abs_a, abs_b);
    auto small = minimum_op(abs_a, abs_b);
    auto one = ones_like_op(big);
    auto safe = where_op(equal_op(big, zeros_like_op(big)), one, big);
    auto ratio = div_op(small, safe);
    auto out = mul_op(big, sqrt_op(add_op(one, square_op(ratio))));
    out = where_op(isinf_op(a), abs_a, out);
    return where_op(isinf_op(b), abs_b, out);
}

TensorImplPtr hypot_op(const TensorImplPtr& a_in, const TensorImplPtr& b_in) {
    if (!a_in || !b_in)
        ErrorBuilder("hypot").fail("null input");
    require_floating("hypot", a_in, b_in);
    const auto [a_dev, b_dev] = detail::align_binary_devices(a_in, b_in, "hypot");
    const AmpScope amp{"hypot", a_dev->device()};
    const auto [a, b] = cast_operands(amp, a_dev, b_dev);
    return hypot_lanes(a, b);
}

// Numerically-stable ``log(exp(a) + exp(b))``.  Factor out the per-pair max
// so only the smaller-magnitude exponential sees rounding error.
TensorImplPtr logaddexp_op(const TensorImplPtr& a_in, const TensorImplPtr& b_in) {
    if (!a_in || !b_in)
        ErrorBuilder("logaddexp").fail("null input");
    require_floating("logaddexp", a_in, b_in);
    const auto [a_dev, b_dev] = detail::align_binary_devices(a_in, b_in, "logaddexp");
    const AmpScope amp{"logaddexp", a_dev->device()};
    const auto [a, b] = cast_operands(amp, a_dev, b_dev);
    auto m = maximum_op(a, b);
    auto safe_m = where_op(isinf_op(m), full_like_op(m, 0.0), m);
    auto exp_a = exp_op(sub_op(a, safe_m));
    auto exp_b = exp_op(sub_op(b, safe_m));
    return add_op(m, log_op(add_op(exp_a, exp_b)));
}

// Tolerance check via the standard formula.  We materialise the ``rtol``
// scalar as a same-shape constant tensor so the multiply doesn't depend on
// any scalar-broadcast support in the binary ops.
TensorImplPtr
isclose_op(const TensorImplPtr& a_in, const TensorImplPtr& b_in, double rtol, double atol) {
    if (!a_in || !b_in)
        ErrorBuilder("isclose").fail("null input");
    const auto [a_dev, b_dev] = detail::align_binary_devices(a_in, b_in, "isclose");
    const AmpScope amp{"isclose", a_dev->device()};
    const auto [a, b] = cast_operands(amp, a_dev, b_dev);
    TensorImplPtr diff;
    TensorImplPtr abs_b;
    if (!is_floating_point(a->dtype()) && !is_floating_point(b->dtype())) {
        // The tolerances are fractions.  Built in an integer dtype they
        // truncate to 0 — ``isclose`` quietly became equality — and in bool
        // ``full_like(…, 1e-8)`` is True, so every pair of bools was "close".
        // Take the difference exactly in an integer type, then compare in
        // float32 the way a floating input would.
        auto as_int = [](const TensorImplPtr& t) {
            return (t->dtype() == Dtype::Bool || t->dtype() == Dtype::I8 ||
                    t->dtype() == Dtype::I16)
                       ? astype_op(t, Dtype::I32)
                       : t;
        };
        const auto ai = as_int(a);
        const auto bi = as_int(b);
        diff = abs_op(astype_op(sub_op(ai, bi), Dtype::F32));
        abs_b = abs_op(astype_op(bi, Dtype::F32));
    } else {
        diff = abs_op(sub_op(a, b));
        abs_b = abs_op(b);
    }
    auto rtol_t = full_like_op(abs_b, rtol);
    auto scaled = mul_op(rtol_t, abs_b);
    auto atol_t = full_like_op(scaled, atol);
    auto tol = add_op(atol_t, scaled);
    auto within_tol = logical_and_op(isfinite_op(diff), less_equal_op(diff, tol));
    return logical_or_op(equal_op(a, b), within_tol);
}

}  // namespace lucid
