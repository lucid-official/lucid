"""Element-wise unary ops — value, shape, dtype, device coverage.

Each op gets a single parametrized class that walks a representative
shape × device × dtype matrix and verifies element-wise agreement with
a numpy reference.  The (metal, float64) cells are never collected —
``device`` and ``float_dtype`` are swept as supported pairs (see
``lucid/test/_fixtures/devices.py``).
"""

from collections.abc import Callable

import numpy as np
import pytest

import lucid
from lucid.test._helpers.compare import assert_close

# ── (lucid_fn, numpy_fn, sample_range, atol) tuples ──────────────────────


_UNARY_OPS: list[tuple[str, Callable, Callable, tuple[float, float], float]] = [
    ("abs", lambda t: t.abs(), np.abs, (-2.0, 2.0), 1e-6),
    ("neg", lambda t: -t, np.negative, (-2.0, 2.0), 1e-6),
    ("sign", lambda t: t.sign(), np.sign, (-2.0, 2.0), 0.0),
    ("exp", lambda t: t.exp(), np.exp, (-2.0, 2.0), 1e-5),
    ("exp2", lambda t: lucid.exp2(t), np.exp2, (-2.0, 2.0), 1e-5),
    ("expm1", lambda t: lucid.expm1(t), np.expm1, (-1.0, 1.0), 1e-5),
    ("log", lambda t: t.log(), np.log, (0.1, 5.0), 1e-5),
    ("log2", lambda t: lucid.log2(t), np.log2, (0.1, 5.0), 1e-5),
    ("log10", lambda t: lucid.log10(t), np.log10, (0.1, 5.0), 1e-5),
    ("log1p", lambda t: lucid.log1p(t), np.log1p, (-0.5, 5.0), 1e-5),
    ("sqrt", lambda t: t.sqrt(), np.sqrt, (0.1, 5.0), 1e-5),
    ("rsqrt", lambda t: lucid.rsqrt(t), lambda x: 1.0 / np.sqrt(x), (0.1, 5.0), 1e-4),
    ("square", lambda t: lucid.square(t), np.square, (-2.0, 2.0), 1e-5),
    ("reciprocal", lambda t: lucid.reciprocal(t), np.reciprocal, (0.5, 2.0), 1e-4),
    ("floor", lambda t: t.floor(), np.floor, (-2.5, 2.5), 0.0),
    ("ceil", lambda t: t.ceil(), np.ceil, (-2.5, 2.5), 0.0),
    ("round", lambda t: t.round(), np.round, (-2.5, 2.5), 0.0),
    ("trunc", lambda t: t.trunc(), np.trunc, (-2.5, 2.5), 0.0),
    ("frac", lambda t: lucid.frac(t), lambda x: x - np.trunc(x), (-2.5, 2.5), 1e-6),
    ("sin", lambda t: t.sin(), np.sin, (-3.0, 3.0), 1e-5),
    ("cos", lambda t: t.cos(), np.cos, (-3.0, 3.0), 1e-5),
    ("tan", lambda t: t.tan(), np.tan, (-1.0, 1.0), 1e-4),
    ("sinh", lambda t: t.sinh(), np.sinh, (-1.5, 1.5), 1e-5),
    ("cosh", lambda t: t.cosh(), np.cosh, (-1.5, 1.5), 1e-5),
    ("tanh", lambda t: t.tanh(), np.tanh, (-2.0, 2.0), 1e-5),
    ("asin", lambda t: lucid.asin(t), np.arcsin, (-0.9, 0.9), 1e-5),
    ("acos", lambda t: lucid.acos(t), np.arccos, (-0.9, 0.9), 1e-5),
    ("atan", lambda t: lucid.atan(t), np.arctan, (-2.0, 2.0), 1e-5),
    (
        "sigmoid",
        lambda t: t.sigmoid(),
        lambda x: 1.0 / (1.0 + np.exp(-x)),
        (-3.0, 3.0),
        1e-5,
    ),
    (
        "erf",
        lambda t: lucid.erf(t),
        lambda x: np.vectorize(__import__("math").erf)(x),
        (-2.0, 2.0),
        1e-5,
    ),
    (
        "erfc",
        lambda t: lucid.erfc(t),
        lambda x: np.vectorize(__import__("math").erfc)(x),
        (-2.0, 2.0),
        1e-5,
    ),
]


@pytest.mark.parametrize(
    "name,lucid_fn,np_fn,rng,atol",
    [(name, lf, nf, r, a) for (name, lf, nf, r, a) in _UNARY_OPS if nf is not None],
    ids=[op[0] for op in _UNARY_OPS if op[2] is not None],
)
def test_unary_value_match(
    name: str,
    lucid_fn: Callable,
    np_fn: Callable,
    rng: tuple[float, float],
    atol: float,
    device: str,
    float_dtype: lucid.dtype,
) -> None:
    """Element-wise agreement with the numpy reference."""
    np.random.seed(0)
    x = np.random.uniform(rng[0], rng[1], size=(4, 5)).astype(np.float32)
    t = lucid.tensor(x.copy(), dtype=float_dtype, device=device)
    out = lucid_fn(t)
    expected = np_fn(x)
    assert_close(out, expected, atol=atol)


@pytest.mark.parametrize(
    "name,lucid_fn,np_fn,rng,atol", _UNARY_OPS, ids=[op[0] for op in _UNARY_OPS]
)
def test_unary_shape_preserved(
    name: str,
    lucid_fn: Callable,
    np_fn,  # type: ignore[no-untyped-def]
    rng: tuple[float, float],
    atol: float,
    device: str,
) -> None:
    np.random.seed(0)
    x = np.random.uniform(rng[0] if rng else -1.0, rng[1] if rng else 1.0, size=(2, 3)).astype(
        np.float32
    )
    t = lucid.tensor(x.copy(), device=device)
    out = lucid_fn(t)
    assert out.shape == t.shape


# ── stability: NaN / inf safety ──────────────────────────────────────────


@pytest.mark.stability
class TestStability:
    def test_isfinite(self, device: str) -> None:
        t = lucid.tensor([1.0, float("inf"), float("-inf"), float("nan")], device=device)
        out = lucid.isfinite(t).numpy()
        np.testing.assert_array_equal(out, [True, False, False, False])

    def test_isinf(self, device: str) -> None:
        t = lucid.tensor([1.0, float("inf"), float("-inf"), float("nan")], device=device)
        out = lucid.isinf(t).numpy()
        np.testing.assert_array_equal(out, [False, True, True, False])

    def test_isnan(self, device: str) -> None:
        t = lucid.tensor([1.0, float("inf"), float("-inf"), float("nan")], device=device)
        out = lucid.isnan(t).numpy()
        np.testing.assert_array_equal(out, [False, False, False, True])

    def test_nan_to_num_default(self, device: str) -> None:
        t = lucid.tensor([1.0, float("nan"), float("inf"), float("-inf")], device=device)
        out = lucid.nan_to_num(t).numpy()
        # ``nan`` → 0.0 by default; ``inf`` and ``-inf`` clamp to the
        # dtype's min/max.
        assert out[0] == 1.0
        assert out[1] == 0.0
        assert np.isfinite(out[2])
        assert np.isfinite(out[3])


# ── basic backward existence check ───────────────────────────────────────


class TestUnaryBackward:
    def test_exp_backward(self, device: str) -> None:
        x = lucid.tensor([1.0, 2.0, 3.0], device=device, requires_grad=True)
        y = x.exp().sum()
        y.backward()
        # d/dx [exp(x)] = exp(x).
        np.testing.assert_allclose(x.grad.numpy(), np.exp([1.0, 2.0, 3.0]), atol=1e-5)

    def test_log_backward(self, device: str) -> None:
        x = lucid.tensor([1.0, 2.0, 4.0], device=device, requires_grad=True)
        y = x.log().sum()
        y.backward()
        # d/dx [log(x)] = 1/x.
        np.testing.assert_allclose(x.grad.numpy(), [1.0, 0.5, 0.25], atol=1e-5)

    def test_sigmoid_backward(self, device: str) -> None:
        x = lucid.tensor([0.0], device=device, requires_grad=True)
        y = x.sigmoid().sum()
        y.backward()
        # σ'(0) = 0.25.
        np.testing.assert_allclose(x.grad.numpy(), [0.25], atol=1e-5)
