"""OpInfo catalog for unary, special-function, binary, and ternary math ops."""

from collections.abc import Callable, Sequence
import numpy as np

import lucid
import lucid.special as lspecial
from lucid.test.parity.opinfo._core import OpInfo, SampleInput, mask_arg, tensor_arg


def _unary_domain_samples(low: float, high: float) -> Callable[[], Sequence[SampleInput]]:
    """Build 1-D, 2-D, and 3-D sample inputs drawn deterministically in ``[low, high]``."""

    def _gen() -> Sequence[SampleInput]:
        rng = np.random.default_rng(42)
        x_2d = rng.uniform(low, high, size=(3, 4))
        x_3d = rng.uniform(low, high, size=(2, 3, 2))
        x_1d = np.linspace(low, high, num=5, dtype=np.float64)
        return (
            SampleInput("2d", (tensor_arg(x_2d),)),
            SampleInput("3d", (tensor_arg(x_3d),)),
            SampleInput("1d_linspace", (tensor_arg(x_1d),)),
        )

    return _gen


def _binary_domain_samples(
    a_low: float, a_high: float, b_low: float, b_high: float
) -> Callable[[], Sequence[SampleInput]]:
    """Build same-shape and broadcast multi-dimensional sample pairs."""

    def _gen() -> Sequence[SampleInput]:
        rng = np.random.default_rng(108)
        a_2d = rng.uniform(a_low, a_high, size=(3, 4))
        b_2d = rng.uniform(b_low, b_high, size=(3, 4))
        a_bcast = rng.uniform(a_low, a_high, size=(2, 1, 3))
        b_bcast = rng.uniform(b_low, b_high, size=(2, 3))
        a_vec = rng.uniform(a_low, a_high, size=(4,))
        b_row = rng.uniform(b_low, b_high, size=(2, 1))
        return (
            SampleInput("same_2d", (tensor_arg(a_2d), tensor_arg(b_2d))),
            SampleInput("bcast_3d_2d", (tensor_arg(a_bcast), tensor_arg(b_bcast))),
            SampleInput("bcast_outer", (tensor_arg(a_vec), tensor_arg(b_row))),
        )

    return _gen


def _clamp_samples() -> Sequence[SampleInput]:
    x = np.array([[-2.5, -0.8, 0.0, 0.7], [1.2, -1.5, 0.3, 2.2]], dtype=np.float64)
    return (
        SampleInput("both_bounds", (tensor_arg(x),), {"min": -1.0, "max": 1.0}),
        SampleInput("min_only", (tensor_arg(x),), {"min": -0.5}),
        SampleInput("max_only", (tensor_arg(x),), {"max": 0.5}),
    )


def _nan_to_num_samples() -> Sequence[SampleInput]:
    x = np.array([[0.0, -1.25, 2.5], [0.75, -0.5, 1.5]], dtype=np.float64)
    x_nonfinite = np.array([[np.nan, np.inf, -np.inf], [0.5, -1.0, 2.0]], dtype=np.float64)
    return (
        SampleInput("finite", (tensor_arg(x),), {"nan": 0.0, "posinf": 10.0, "neginf": -10.0}),
        SampleInput(
            "nonfinite",
            (tensor_arg(x_nonfinite, differentiable=False),),
            {"nan": 0.25, "posinf": 9.0, "neginf": -9.0},
            supports_backward=False,
            supports_gradcheck=False,
        ),
    )


def _where_samples() -> Sequence[SampleInput]:
    cond_2d = np.array([[True, False, True], [False, True, False]])
    x_2d = np.array([[1.2, -0.5, 2.3], [-1.1, 0.8, -2.0]], dtype=np.float64)
    y_2d = np.array([[-0.7, 1.5, -1.3], [2.1, -0.4, 0.9]], dtype=np.float64)
    cond_bcast = np.array([[True], [False]])
    x_row = np.array([[0.5, -1.2, 1.8]], dtype=np.float64)
    return (
        SampleInput("same_2d", (mask_arg(cond_2d), tensor_arg(x_2d), tensor_arg(y_2d))),
        SampleInput("bcast", (mask_arg(cond_bcast), tensor_arg(x_row), tensor_arg(y_2d))),
    )


def _lerp_samples() -> Sequence[SampleInput]:
    start = np.array([[0.5, -1.0, 2.0], [1.5, -0.5, 0.25]], dtype=np.float64)
    end = np.array([[2.5, 1.0, -1.0], [-0.5, 1.5, 2.25]], dtype=np.float64)
    w_tensor = np.array([[0.2, 0.5, 0.8], [0.1, 0.9, 0.4]], dtype=np.float64)
    w_bcast = np.array([[0.3], [0.7]], dtype=np.float64)
    return (
        SampleInput("scalar_weight", (tensor_arg(start), tensor_arg(end), 0.35)),
        SampleInput("tensor_weight", (tensor_arg(start), tensor_arg(end), tensor_arg(w_tensor))),
        SampleInput("bcast_weight", (tensor_arg(start), tensor_arg(end), tensor_arg(w_bcast))),
    )


def _addcmul_samples() -> Sequence[SampleInput]:
    inp = np.array([[1.0, -0.5, 0.8], [-1.2, 0.4, 1.5]], dtype=np.float64)
    t1 = np.array([[0.6, 1.2, -0.7], [0.9, -1.1, 0.5]], dtype=np.float64)
    t2 = np.array([[1.4, -0.8, 0.5], [-0.6, 0.7, 1.3]], dtype=np.float64)
    t2_bcast = np.array([[0.5, -1.0, 1.5]], dtype=np.float64)
    return (
        SampleInput("default_value", (tensor_arg(inp), tensor_arg(t1), tensor_arg(t2))),
        SampleInput(
            "custom_value_bcast",
            (tensor_arg(inp), tensor_arg(t1), tensor_arg(t2_bcast)),
            {"value": 0.5},
        ),
    )


def _addcdiv_samples() -> Sequence[SampleInput]:
    inp = np.array([[1.0, -0.5, 0.8], [-1.2, 0.4, 1.5]], dtype=np.float64)
    t1 = np.array([[0.6, 1.2, -0.7], [0.9, -1.1, 0.5]], dtype=np.float64)
    t2 = np.array([[1.4, 0.8, 1.5], [0.6, 1.1, 1.3]], dtype=np.float64)
    return (
        SampleInput("default_value", (tensor_arg(inp), tensor_arg(t1), tensor_arg(t2))),
        SampleInput(
            "custom_value", (tensor_arg(inp), tensor_arg(t1), tensor_arg(t2)), {"value": -0.5}
        ),
    )


_UNARY_SPECS: tuple[tuple[str, Callable[..., object], Callable[..., object], float, float], ...] = (
    ("abs", lucid.abs, lambda r, x: r.abs(x), -2.5, -0.25),
    ("neg", lucid.neg, lambda r, x: r.neg(x), -2.5, 2.5),
    ("exp", lucid.exp, lambda r, x: r.exp(x), -2.0, 2.0),
    ("expm1", lucid.expm1, lambda r, x: r.expm1(x), -1.5, 1.5),
    ("exp2", lucid.exp2, lambda r, x: r.exp2(x), -2.0, 2.0),
    ("log", lucid.log, lambda r, x: r.log(x), 0.2, 3.5),
    ("log1p", lucid.log1p, lambda r, x: r.log1p(x), -0.5, 2.5),
    ("log2", lucid.log2, lambda r, x: r.log2(x), 0.2, 3.5),
    ("log10", lucid.log10, lambda r, x: r.log10(x), 0.2, 3.5),
    ("sqrt", lucid.sqrt, lambda r, x: r.sqrt(x), 0.25, 4.0),
    ("rsqrt", lucid.rsqrt, lambda r, x: r.rsqrt(x), 0.35, 3.5),
    ("reciprocal", lucid.reciprocal, lambda r, x: r.reciprocal(x), 0.4, 3.0),
    ("square", lucid.square, lambda r, x: r.square(x), -2.5, 2.5),
    ("sin", lucid.sin, lambda r, x: r.sin(x), -2.5, 2.5),
    ("cos", lucid.cos, lambda r, x: r.cos(x), -2.5, 2.5),
    ("tan", lucid.tan, lambda r, x: r.tan(x), -1.1, 1.1),
    ("asin", lucid.asin, lambda r, x: r.asin(x), -0.8, 0.8),
    ("acos", lucid.acos, lambda r, x: r.acos(x), -0.8, 0.8),
    ("atan", lucid.atan, lambda r, x: r.atan(x), -2.5, 2.5),
    ("sinh", lucid.sinh, lambda r, x: r.sinh(x), -1.8, 1.8),
    ("cosh", lucid.cosh, lambda r, x: r.cosh(x), -1.8, 1.8),
    ("tanh", lucid.tanh, lambda r, x: r.tanh(x), -2.5, 2.5),
    ("asinh", lucid.asinh, lambda r, x: r.asinh(x), -2.5, 2.5),
    ("acosh", lucid.acosh, lambda r, x: r.acosh(x), 1.25, 3.5),
    ("atanh", lucid.atanh, lambda r, x: r.atanh(x), -0.8, 0.8),
    ("erf", lucid.erf, lambda r, x: r.erf(x), -2.0, 2.0),
    ("erfc", lucid.erfc, lambda r, x: r.erfc(x), -2.0, 2.0),
    ("erfinv", lucid.erfinv, lambda r, x: r.erfinv(x), -0.8, 0.8),
    ("sigmoid", lucid.sigmoid, lambda r, x: r.sigmoid(x), -3.0, 3.0),
    ("logit", lucid.logit, lambda r, x: r.logit(x), 0.15, 0.85),
    ("sinc", lucid.sinc, lambda r, x: r.sinc(x), -2.1, 2.3),
    ("digamma", lucid.digamma, lambda r, x: r.digamma(x), 0.6, 3.5),
    ("lgamma", lucid.lgamma, lambda r, x: r.lgamma(x), 0.5, 3.5),
    ("i0", lucid.i0, lambda r, x: r.i0(x), -2.0, 2.0),
    ("special.i0e", lspecial.i0e, lambda r, x: r.special.i0e(x), -2.0, 2.0),
    ("special.i1", lspecial.i1, lambda r, x: r.special.i1(x), -2.0, 2.0),
    ("special.i1e", lspecial.i1e, lambda r, x: r.special.i1e(x), -2.0, 2.0),
    (
        "special.erfcx",
        lspecial.erfcx,
        lambda r, x: r.special.erfcx(x.float()).to(x.dtype),
        -1.0,
        2.0,
    ),
    ("special.ndtr", lspecial.ndtr, lambda r, x: r.special.ndtr(x), -2.0, 2.0),
    (
        "special.log_ndtr",
        lspecial.log_ndtr,
        lambda r, x: r.special.log_ndtr(x.float()).to(x.dtype),
        -2.0,
        2.0,
    ),
    ("special.entr", lspecial.entr, lambda r, x: r.special.entr(x), 0.25, 2.5),
)

_BINARY_SPECS: tuple[
    tuple[str, Callable[..., object], Callable[..., object], float, float, float, float, bool], ...
] = (
    ("add", lucid.add, lambda r, a, b: r.add(a, b), -2.0, 2.0, -2.0, 2.0, True),
    ("sub", lucid.sub, lambda r, a, b: r.sub(a, b), -2.0, 2.0, -2.0, 2.0, True),
    ("mul", lucid.mul, lambda r, a, b: r.mul(a, b), -2.0, 2.0, -2.0, 2.0, True),
    ("div", lucid.div, lambda r, a, b: r.div(a, b), -2.0, 2.0, 0.5, 2.5, True),
    ("pow", lucid.pow, lambda r, a, b: r.pow(a, b), 0.5, 2.5, -1.2, 1.8, True),
    ("atan2", lucid.atan2, lambda r, a, b: r.atan2(a, b), -2.0, 2.0, 0.4, 2.2, True),
    ("hypot", lucid.hypot, lambda r, a, b: r.hypot(a, b), 0.3, 2.2, 0.4, 2.2, True),
    ("logaddexp", lucid.logaddexp, lambda r, a, b: r.logaddexp(a, b), -2.0, 2.0, -2.0, 2.0, True),
    ("fmod", lucid.fmod, lambda r, a, b: r.fmod(a, b), 0.35, 2.6, 0.8, 1.9, False),
    ("remainder", lucid.remainder, lambda r, a, b: r.remainder(a, b), 0.35, 2.6, 0.8, 1.9, False),
    ("maximum", lucid.maximum, lambda r, a, b: r.maximum(a, b), -2.0, -0.3, 0.3, 2.0, True),
    ("minimum", lucid.minimum, lambda r, a, b: r.minimum(a, b), -2.0, -0.3, 0.3, 2.0, True),
    ("fmax", lucid.fmax, lambda r, a, b: r.fmax(a, b), -2.0, -0.3, 0.3, 2.0, True),
    ("fmin", lucid.fmin, lambda r, a, b: r.fmin(a, b), -2.0, -0.3, 0.3, 2.0, True),
    ("copysign", lucid.copysign, lambda r, a, b: r.copysign(a, b), 0.4, 2.5, 0.4, 2.5, False),
    ("xlogy", lucid.xlogy, lambda r, a, b: r.xlogy(a, b), -1.5, 1.5, 0.4, 2.5, True),
    (
        "special.xlog1py",
        lspecial.xlog1py,
        lambda r, a, b: r.special.xlog1py(a, b),
        -1.5,
        1.5,
        -0.4,
        2.0,
        True,
    ),
    ("heaviside", lucid.heaviside, lambda r, a, b: r.heaviside(a, b), -1.5, 1.5, 0.2, 0.8, False),
)

MATH_OPINFOS: tuple[OpInfo, ...] = (
    *(
        OpInfo(
            name=name,
            family="unary_math",
            lucid_fn=l_fn,
            ref_fn=r_fn,
            sample_inputs_fn=_unary_domain_samples(lo, hi),
        )
        for name, l_fn, r_fn, lo, hi in _UNARY_SPECS
    ),
    OpInfo(
        name="clamp",
        family="unary_math",
        lucid_fn=lucid.clamp,
        ref_fn=lambda r, x, **kw: r.clamp(x, **kw),
        sample_inputs_fn=_clamp_samples,
    ),
    OpInfo(
        name="nan_to_num",
        family="unary_math",
        lucid_fn=lucid.nan_to_num,
        ref_fn=lambda r, x, **kw: r.nan_to_num(x, **kw),
        sample_inputs_fn=_nan_to_num_samples,
    ),
    *(
        OpInfo(
            name=name,
            family="binary_math",
            lucid_fn=l_fn,
            ref_fn=r_fn,
            sample_inputs_fn=_binary_domain_samples(a_lo, a_hi, b_lo, b_hi),
            supports_backward=has_bwd,
            supports_gradcheck=has_bwd,
        )
        for name, l_fn, r_fn, a_lo, a_hi, b_lo, b_hi, has_bwd in _BINARY_SPECS
    ),
    OpInfo(
        name="where",
        family="ternary_math",
        lucid_fn=lucid.where,
        ref_fn=lambda r, c, x, y: r.where(c, x, y),
        sample_inputs_fn=_where_samples,
    ),
    OpInfo(
        name="lerp",
        family="ternary_math",
        lucid_fn=lucid.lerp,
        ref_fn=lambda r, s, e, w: r.lerp(s, e, w),
        sample_inputs_fn=_lerp_samples,
    ),
    OpInfo(
        name="addcmul",
        family="ternary_math",
        lucid_fn=lucid.addcmul,
        ref_fn=lambda r, inp, t1, t2, **kw: r.addcmul(inp, t1, t2, **kw),
        sample_inputs_fn=_addcmul_samples,
    ),
    OpInfo(
        name="addcdiv",
        family="ternary_math",
        lucid_fn=lucid.addcdiv,
        ref_fn=lambda r, inp, t1, t2, **kw: r.addcdiv(inp, t1, t2, **kw),
        sample_inputs_fn=_addcdiv_samples,
    ),
)
