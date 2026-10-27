"""Element-wise binary ops + broadcasting."""

from collections.abc import Callable

import numpy as np
import pytest

import lucid
from lucid.test._helpers.compare import assert_close

# (name, lucid_fn, numpy_fn, low, high, atol)
_BINARY_OPS: list[tuple[str, Callable, Callable, float, float, float]] = [
    ("add", lambda a, b: a + b, np.add, -2.0, 2.0, 1e-6),
    ("sub", lambda a, b: a - b, np.subtract, -2.0, 2.0, 1e-6),
    ("mul", lambda a, b: a * b, np.multiply, -2.0, 2.0, 1e-6),
    ("div", lambda a, b: a / b, np.divide, 0.5, 2.0, 1e-5),
    ("pow", lambda a, b: a**b, np.power, 0.5, 2.0, 1e-4),
    ("maximum", lambda a, b: lucid.maximum(a, b), np.maximum, -2.0, 2.0, 0.0),
    ("minimum", lambda a, b: lucid.minimum(a, b), np.minimum, -2.0, 2.0, 0.0),
    ("atan2", lambda a, b: lucid.atan2(a, b), np.arctan2, -2.0, 2.0, 1e-5),
    ("hypot", lambda a, b: lucid.hypot(a, b), np.hypot, 0.1, 2.0, 1e-4),
    ("fmod", lambda a, b: lucid.fmod(a, b), np.fmod, 0.1, 2.0, 1e-5),
    ("remainder", lambda a, b: lucid.remainder(a, b), np.remainder, 0.1, 2.0, 1e-5),
    ("fmax", lambda a, b: lucid.fmax(a, b), np.fmax, -2.0, 2.0, 0.0),
    ("fmin", lambda a, b: lucid.fmin(a, b), np.fmin, -2.0, 2.0, 0.0),
    ("logaddexp", lambda a, b: lucid.logaddexp(a, b), np.logaddexp, -2.0, 2.0, 1e-5),
]


@pytest.mark.parametrize(
    "name,lucid_fn,np_fn,low,high,atol", _BINARY_OPS, ids=[op[0] for op in _BINARY_OPS]
)
def test_binary_value_match(
    name: str,
    lucid_fn: Callable,
    np_fn: Callable,
    low: float,
    high: float,
    atol: float,
    device: str,
    float_dtype: lucid.dtype,
) -> None:
    np.random.seed(0)
    shape = (4, 5)
    a = np.random.uniform(low, high, size=shape).astype(np.float32)
    b = np.random.uniform(low, high, size=shape).astype(np.float32)
    la = lucid.tensor(a.copy(), dtype=float_dtype, device=device)
    lb = lucid.tensor(b.copy(), dtype=float_dtype, device=device)
    out = lucid_fn(la, lb)
    expected = np_fn(a, b)
    assert_close(out, expected, atol=atol)


# ── broadcasting ─────────────────────────────────────────────────────────


_BCAST_PAIRS = [
    ((3,), (3,)),
    ((1,), (4,)),
    ((4, 1), (1, 5)),
    ((2, 3, 1), (1, 3, 5)),
    ((1, 3, 8, 8), (3, 1, 1)),
]


@pytest.mark.parametrize("shape_a,shape_b", _BCAST_PAIRS)
def test_add_broadcasts(shape_a: tuple[int, ...], shape_b: tuple[int, ...], device: str) -> None:
    np.random.seed(0)
    a = np.random.uniform(-1.0, 1.0, size=shape_a).astype(np.float32)
    b = np.random.uniform(-1.0, 1.0, size=shape_b).astype(np.float32)
    la = lucid.tensor(a.copy(), device=device)
    lb = lucid.tensor(b.copy(), device=device)
    out = la + lb
    assert_close(out, a + b, atol=1e-6)


# ── scalar mixing ────────────────────────────────────────────────────────


class TestScalarOps:
    def test_add_scalar(self, device: str) -> None:
        t = lucid.tensor([1.0, 2.0, 3.0], device=device)
        out = (t + 5.0).numpy()
        np.testing.assert_array_equal(out, [6.0, 7.0, 8.0])

    def test_radd_scalar(self, device: str) -> None:
        t = lucid.tensor([1.0, 2.0, 3.0], device=device)
        out = (10.0 + t).numpy()
        np.testing.assert_array_equal(out, [11.0, 12.0, 13.0])

    def test_mul_scalar(self, device: str) -> None:
        t = lucid.tensor([1.0, 2.0, 3.0], device=device)
        np.testing.assert_array_equal((t * 3.0).numpy(), [3.0, 6.0, 9.0])

    def test_div_scalar(self, device: str) -> None:
        t = lucid.tensor([2.0, 4.0, 8.0], device=device)
        np.testing.assert_array_equal((t / 2.0).numpy(), [1.0, 2.0, 4.0])


# ── matmul / mm / bmm ────────────────────────────────────────────────────


class TestMatmul:
    def test_2d_2d(self, device: str) -> None:
        a = lucid.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
        b = lucid.tensor([[5.0, 6.0], [7.0, 8.0]], device=device)
        out = (a @ b).numpy()
        np.testing.assert_array_equal(out, [[19.0, 22.0], [43.0, 50.0]])

    def test_mm_alias(self, device: str) -> None:
        a = lucid.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
        b = lucid.tensor([[5.0, 6.0], [7.0, 8.0]], device=device)
        out_mm = lucid.mm(a, b).numpy()
        out_at = (a @ b).numpy()
        np.testing.assert_array_equal(out_mm, out_at)

    def test_bmm(self, device: str) -> None:
        a = lucid.tensor([[[1.0, 2.0], [3.0, 4.0]]] * 2, device=device)
        b = lucid.tensor([[[5.0, 6.0], [7.0, 8.0]]] * 2, device=device)
        out = lucid.bmm(a, b).numpy()
        assert out.shape == (2, 2, 2)

    def test_empty_output_realize_no_crash(self, device: str) -> None:
        # Realizing an empty-output matmul (.numpy() / repr) used to SIGSEGV on
        # Metal — the download path ran contiguous()+eval() before the numel==0
        # short-circuit. Must produce a correctly-shaped empty array, not crash.
        a = lucid.zeros(0, 4, device=device)
        b = lucid.zeros(4, 3, device=device)
        out = (a @ b).numpy()
        assert out.shape == (0, 3)
        # empty inner dim too
        c = (lucid.zeros(2, 0, device=device) @ lucid.zeros(0, 3, device=device)).numpy()
        assert c.shape == (2, 3)


class TestIntegerRemainder:
    """`remainder` is the floored modulo (sign of the divisor) for integers too."""

    def test_floored_sign_and_invariant(self, device: str) -> None:
        a = lucid.tensor([-7, 7, -7, 7], dtype=lucid.int32, device=device)
        b = lucid.tensor([3, -3, -3, 3], dtype=lucid.int32, device=device)
        r = lucid.remainder(a, b)
        # floored modulo follows the sign of b (vs fmod which follows a).
        np.testing.assert_array_equal(r.numpy(), [2, -2, -1, 1])
        assert r.dtype == lucid.int32  # input dtype preserved
        # invariant: (a // b) * b + remainder == a
        np.testing.assert_array_equal(((a // b) * b + r).numpy(), a.numpy())

    def test_float_unchanged(self, device: str) -> None:
        a = lucid.tensor([-7.0, 7.0], device=device)
        b = lucid.tensor([3.0, -3.0], device=device)
        np.testing.assert_allclose(lucid.remainder(a, b).numpy(), [2.0, -2.0], atol=1e-6)


_TRUE_DIVISIONS = pytest.mark.parametrize(
    "fn", [lucid.divide, lucid.true_divide], ids=["divide", "true_divide"]
)


class TestTrueDivisionOfIntegers:
    """``divide`` / ``true_divide`` return a float for integer operands.

    The engine's ``div`` keeps the integer dtype (``[7, 8] / 2`` is
    ``[3, 4]``), so the two composites promote the numerator themselves —
    which is what both document, and what the reference framework does.
    """

    @_TRUE_DIVISIONS
    def test_int_by_int(self, fn: Callable, device: str) -> None:
        a = lucid.tensor([7, 8], dtype=lucid.int32, device=device)
        b = lucid.tensor([2, 2], dtype=lucid.int32, device=device)
        out = fn(a, b)
        assert out.dtype == lucid.float32
        np.testing.assert_array_equal(out.numpy(), [3.5, 4.0])

    @_TRUE_DIVISIONS
    def test_int_by_scalar(self, fn: Callable, device: str) -> None:
        a = lucid.tensor([7, 8], dtype=lucid.int32, device=device)
        out = fn(a, 2)
        assert out.dtype == lucid.float32
        np.testing.assert_array_equal(out.numpy(), [3.5, 4.0])

    @_TRUE_DIVISIONS
    def test_bool_by_bool(self, fn: Callable) -> None:
        a = lucid.tensor([True, False])
        b = lucid.tensor([True, True])
        out = fn(a, b)
        assert out.dtype == lucid.float32
        np.testing.assert_array_equal(out.numpy(), [1.0, 0.0])

    def test_wider_float_denominator_wins(self) -> None:
        a = lucid.tensor([7, 8], dtype=lucid.int64)
        b = lucid.tensor([2.0, 2.0], dtype=lucid.float64)
        out = lucid.true_divide(a, b)
        assert out.dtype == lucid.float64
        np.testing.assert_array_equal(out.numpy(), [3.5, 4.0])


# ── backward sanity ──────────────────────────────────────────────────────


class TestBinaryBackward:
    def test_add_backward(self, device: str) -> None:
        a = lucid.tensor([1.0, 2.0], device=device, requires_grad=True)
        b = lucid.tensor([3.0, 4.0], device=device, requires_grad=True)
        (a + b).sum().backward()
        np.testing.assert_array_equal(a.grad.numpy(), [1.0, 1.0])
        np.testing.assert_array_equal(b.grad.numpy(), [1.0, 1.0])

    def test_mul_backward(self, device: str) -> None:
        a = lucid.tensor([2.0, 3.0], device=device, requires_grad=True)
        b = lucid.tensor([5.0, 7.0], device=device, requires_grad=True)
        (a * b).sum().backward()
        np.testing.assert_array_equal(a.grad.numpy(), [5.0, 7.0])
        np.testing.assert_array_equal(b.grad.numpy(), [2.0, 3.0])


@pytest.mark.parametrize("op", ["maximum", "minimum"])
@pytest.mark.parametrize("bound", ["0-d", "one-element", "view"])
def test_minmax_gradient_against_a_broadcast_operand(op: str, bound: str, device: str) -> None:
    """The gradient mask compares the operands at the output's shape.

    It read them at their own shapes over the output's length, so a
    one-element operand — ``clamp`` against a 0-d bound, ``minimum(x,
    bins[-1])`` — was read past its buffer on the CPU and the mask came from
    whatever lay there.  ``F.two_hot`` clamps exactly that way.
    """
    grid = lucid.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device=device)
    b = {"0-d": grid[4].clone(), "one-element": grid[1:2].clone(), "view": grid[4]}[bound]
    x = lucid.tensor([-3.0, -1.5, 0.5, 1.5, 3.0], device=device).requires_grad_(True)
    getattr(lucid, op)(x, b).sum().backward()
    bv = float(b.reshape(-1)[0].item())
    xs = np.array([-3.0, -1.5, 0.5, 1.5, 3.0])
    want = (xs >= bv) if op == "maximum" else (xs <= bv)
    assert x.grad is not None
    assert x.grad.tolist() == want.astype(float).tolist()
