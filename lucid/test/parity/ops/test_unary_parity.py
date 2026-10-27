"""Reference-parity tests for unary ops.

For each op we draw the same NumPy-seeded input, push it through both
Lucid and the reference framework, and assert element-wise agreement
within tight atol.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import lucid
from lucid.test._helpers.compare import assert_close, assert_equal_int

# ── Value parity: (name, lucid_fn, ref_attr, (lo, hi), atol) ─────────────────
_UNARY_OPS: list[tuple[str, Callable, str, tuple[float, float], float]] = [
    # basic math
    ("abs", lambda t: t.abs(), "abs", (-2.0, 2.0), 1e-6),
    ("neg", lambda t: -t, "neg", (-2.0, 2.0), 1e-6),
    ("sign", lambda t: lucid.sign(t), "sign", (-2.0, 2.0), 0.0),
    ("square", lambda t: lucid.square(t), "square", (-2.0, 2.0), 1e-5),
    ("reciprocal", lambda t: lucid.reciprocal(t), "reciprocal", (0.2, 3.0), 1e-5),
    # exp / log family
    ("exp", lambda t: t.exp(), "exp", (-2.0, 2.0), 1e-5),
    ("exp2", lambda t: lucid.exp2(t), "exp2", (-2.0, 2.0), 1e-5),
    ("expm1", lambda t: lucid.expm1(t), "expm1", (-1.0, 1.0), 1e-5),
    ("log", lambda t: t.log(), "log", (0.1, 5.0), 1e-5),
    ("log2", lambda t: lucid.log2(t), "log2", (0.1, 5.0), 1e-5),
    ("log10", lambda t: lucid.log10(t), "log10", (0.1, 5.0), 1e-5),
    ("log1p", lambda t: lucid.log1p(t), "log1p", (-0.5, 5.0), 1e-5),
    # roots / power
    ("sqrt", lambda t: t.sqrt(), "sqrt", (0.1, 5.0), 1e-5),
    ("rsqrt", lambda t: lucid.rsqrt(t), "rsqrt", (0.1, 5.0), 1e-4),
    # rounding
    ("floor", lambda t: t.floor(), "floor", (-2.5, 2.5), 0.0),
    ("ceil", lambda t: t.ceil(), "ceil", (-2.5, 2.5), 0.0),
    ("round", lambda t: t.round(), "round", (-2.5, 2.5), 0.0),
    ("trunc", lambda t: lucid.trunc(t), "trunc", (-2.5, 2.5), 0.0),
    ("frac", lambda t: lucid.frac(t), "frac", (-2.5, 2.5), 1e-6),
    # trig
    ("sin", lambda t: t.sin(), "sin", (-3.0, 3.0), 1e-5),
    ("cos", lambda t: t.cos(), "cos", (-3.0, 3.0), 1e-5),
    ("tan", lambda t: lucid.tan(t), "tan", (-1.0, 1.0), 1e-4),
    ("asin", lambda t: lucid.asin(t), "asin", (-0.9, 0.9), 1e-5),
    ("acos", lambda t: lucid.acos(t), "acos", (-0.9, 0.9), 1e-5),
    ("atan", lambda t: lucid.atan(t), "atan", (-5.0, 5.0), 1e-5),
    # hyperbolic
    ("sinh", lambda t: lucid.sinh(t), "sinh", (-2.0, 2.0), 1e-5),
    ("cosh", lambda t: lucid.cosh(t), "cosh", (-2.0, 2.0), 1e-5),
    ("tanh", lambda t: t.tanh(), "tanh", (-2.0, 2.0), 1e-5),
    ("asinh", lambda t: lucid.asinh(t), "asinh", (-3.0, 3.0), 1e-5),
    ("acosh", lambda t: lucid.acosh(t), "acosh", (1.05, 4.0), 1e-4),
    ("atanh", lambda t: lucid.atanh(t), "atanh", (-0.9, 0.9), 1e-5),
    # activation
    ("sigmoid", lambda t: t.sigmoid(), "sigmoid", (-3.0, 3.0), 1e-5),
    ("relu", lambda t: lucid.relu(t), "relu", (-2.0, 2.0), 0.0),
    # special / error
    ("erf", lambda t: lucid.erf(t), "erf", (-2.0, 2.0), 1e-5),
    ("erfc", lambda t: lucid.erfc(t), "erfc", (-2.0, 2.0), 1e-5),
    ("erfinv", lambda t: lucid.erfinv(t), "erfinv", (-0.9, 0.9), 1e-4),
    ("lgamma", lambda t: lucid.lgamma(t), "lgamma", (0.5, 5.0), 1e-4),
    ("digamma", lambda t: lucid.digamma(t), "digamma", (0.5, 5.0), 1e-4),
    ("i0", lambda t: lucid.i0(t), "i0", (-2.0, 2.0), 1e-4),
    ("sinc", lambda t: lucid.sinc(t), "sinc", (-3.0, 3.0), 1e-5),
    ("logit", lambda t: lucid.logit(t), "logit", (0.05, 0.95), 1e-4),
    # nan/inf handling
    ("nan_to_num", lambda t: lucid.nan_to_num(t), "nan_to_num", (-2.0, 2.0), 0.0),
]

# ── Predicate ops (bool output — exact equality) ──────────────────────────────
_PRED_OPS: list[tuple[str, Callable, str, tuple[float, float]]] = [
    ("isfinite", lambda t: lucid.isfinite(t), "isfinite", (-2.0, 2.0)),
    ("isinf", lambda t: lucid.isinf(t), "isinf", (-2.0, 2.0)),
    ("isnan", lambda t: lucid.isnan(t), "isnan", (-2.0, 2.0)),
    ("signbit", lambda t: lucid.signbit(t), "signbit", (-2.0, 2.0)),
]

# ── Backward parity: d(op)/dx must agree ──────────────────────────────────────
_BACKWARD_OPS: list[tuple[str, Callable, str, tuple[float, float], float]] = [
    ("sin_bwd", lambda t: t.sin().sum(), "sin", (-2.0, 2.0), 1e-5),
    ("cos_bwd", lambda t: t.cos().sum(), "cos", (-2.0, 2.0), 1e-5),
    ("exp_bwd", lambda t: t.exp().sum(), "exp", (-1.0, 1.0), 1e-5),
    ("log_bwd", lambda t: t.log().sum(), "log", (0.5, 3.0), 1e-5),
    ("sqrt_bwd", lambda t: t.sqrt().sum(), "sqrt", (0.1, 3.0), 1e-4),
    ("tanh_bwd", lambda t: t.tanh().sum(), "tanh", (-2.0, 2.0), 1e-5),
    ("erf_bwd", lambda t: lucid.erf(t).sum(), "erf", (-1.5, 1.5), 1e-5),
    ("sigmoid_bwd", lambda t: t.sigmoid().sum(), "sigmoid", (-2.0, 2.0), 1e-5),
]


@pytest.mark.parity
@pytest.mark.parametrize(
    "name,lucid_fn,ref_attr,rng,atol", _UNARY_OPS, ids=[op[0] for op in _UNARY_OPS]
)
def test_unary_parity(
    name: str,
    lucid_fn: Callable,
    ref_attr: str,
    rng: tuple[float, float],
    atol: float,
    ref: Any,
) -> None:
    np.random.seed(0)
    x = np.random.uniform(rng[0], rng[1], size=(4, 5)).astype(np.float32)
    lt = lucid.tensor(x.copy())
    rt = ref.tensor(x.copy())
    l_out = lucid_fn(lt)
    r_out = getattr(ref, ref_attr)(rt)
    assert_close(l_out, r_out, atol=atol)


@pytest.mark.parity
@pytest.mark.parametrize("name,lucid_fn,ref_attr,rng", _PRED_OPS, ids=[op[0] for op in _PRED_OPS])
def test_unary_predicate_parity(
    name: str,
    lucid_fn: Callable,
    ref_attr: str,
    rng: tuple[float, float],
    ref: Any,
) -> None:
    np.random.seed(0)
    x = np.random.uniform(rng[0], rng[1], size=(4, 5)).astype(np.float32)
    lt = lucid.tensor(x.copy())
    rt = ref.tensor(x.copy())
    l_out = lucid_fn(lt)
    r_out = getattr(ref, ref_attr)(rt)
    assert_equal_int(l_out, r_out)


@pytest.mark.parity
@pytest.mark.parametrize(
    "name,lucid_fn,ref_attr,rng,atol",
    _BACKWARD_OPS,
    ids=[op[0] for op in _BACKWARD_OPS],
)
def test_unary_backward_parity(
    name: str,
    lucid_fn: Callable,
    ref_attr: str,
    rng: tuple[float, float],
    atol: float,
    ref: Any,
) -> None:
    np.random.seed(1)
    x = np.random.uniform(rng[0], rng[1], size=(3, 4)).astype(np.float32)

    xl = lucid.tensor(x.copy(), requires_grad=True)
    lucid_fn(xl).backward()

    xr = ref.tensor(x.copy(), requires_grad=True)
    getattr(ref, ref_attr)(xr).sum().backward()

    assert_close(xl.grad, xr.grad, atol=atol)
