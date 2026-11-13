"""LCD-1004: a compiled division / reciprocal backward never forms the divisor's square.

``div``'s VJP wrote the quotient rule as ``-(go * a) / (b * b)`` and ``reciprocal``'s as
``-go / (x * x)``.  ``b * b`` leaves float32 for a ``|b|`` outside ~1e-19 .. 1e19, so a compiled
step returned 0 (``b * b`` = inf) or inf / NaN (``b * b`` flushed to 0) for a gradient that is an
ordinary float whenever the cotangent and the numerator are of ``b``'s scale: ``x / ||x||`` for
a norm near 1e25, a learned scale next to a numerator that tracks it.  The emitters now divide by
``b`` twice, ``-(go * (a / b)) / b``, as eager's ``DivBackward`` does (LCD-998).

Every case runs a compiled step and checks it against a float64 closed form computed from the
float32 values actually fed in, and against the eager step (range-safe since LCD-998).  ``LUCID_MANUAL_VJP_REQUIRE=1``
makes a node the emitters cannot lower raise instead of being differentiated by MPSGraph, so a
passing case is the emitter's own gradient.
"""

from dataclasses import dataclass

import numpy as np
import pytest

import lucid
import lucid.nn as nn
from lucid._C import engine as _C_engine
from lucid.compile import compiled_step
from lucid.compile._core.trace_context import compile_attempt, recording
from lucid.test._default_device import metal_available
from lucid.test.unit.compile._helpers import COMPILE_DEVICE

pytestmark = pytest.mark.skipif(not metal_available(), reason="Metal unavailable")


@pytest.fixture(autouse=True)
def _manual_vjp_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LUCID_MANUAL_VJP_REQUIRE", "1")


#: Float32 holds 1.2e-38 .. 3.4e38; a divisor squared leaves it outside |b| ~ 1e-19 .. 1e19.
MAGNITUDES = [1e-30, 1e-25, 1e-20, 1e-15, 1e-10, 1.0, 1e10, 1e15, 1e20, 1e25, 1e30]
DENOMINATOR = np.array([1.5, -2.0, 0.5, 3.0, -1.0], dtype=np.float32)
NUMERATOR = np.array([0.7, 1.0, -2.0, 0.25, 4.0], dtype=np.float32)
ROW_SCALE = np.array([[1.0], [-0.5], [2.0]], dtype=np.float32)
RTOL = 2e-6


@dataclass(frozen=True)
class Case:
    """Shapes of ``a / b`` and which of the two carry a gradient (``b`` always does)."""

    name: str
    a_shape: tuple[int, ...]
    b_shape: tuple[int, ...]
    a_tracked: bool
    zero_numerator: bool = False


CASES = [
    Case("both_tracked", (5,), (5,), a_tracked=True),
    Case("divisor_only", (5,), (5,), a_tracked=False),
    Case("divisor_broadcast", (3, 5), (5,), a_tracked=True),
    Case("numerator_broadcast", (5,), (3, 5), a_tracked=True),
    Case("scalar_divisor", (3, 5), (), a_tracked=True),
    Case("zero_numerator", (5,), (5,), a_tracked=True, zero_numerator=True),
]


def _values(case: Case, magnitude: float) -> tuple[np.ndarray, np.ndarray]:
    """The numerator and divisor, both of ``magnitude``'s scale, float32."""
    scale = np.float32(magnitude)

    def shaped(base: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        if shape == ():
            return base[:1].reshape(())
        return base * ROW_SCALE if len(shape) == 2 else base

    a = shaped(NUMERATOR, case.a_shape) * scale
    b = shaped(DENOMINATOR, case.b_shape) * scale
    return (np.zeros_like(a) if case.zero_numerator else a), b


class Quotient(nn.Module):
    """``x / b`` with ``b`` a parameter."""

    def __init__(self, b: lucid.Tensor) -> None:
        super().__init__()
        self.b = nn.Parameter(b)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return x / self.b


class Reciprocal(nn.Module):
    """``x * reciprocal(b)``: the reciprocal's own cotangent is ``x``, of ``b``'s scale."""

    def __init__(self, b: lucid.Tensor) -> None:
        super().__init__()
        self.b = nn.Parameter(b)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return x * lucid.reciprocal(self.b)


MODELS = {"div": Quotient, "reciprocal": Reciprocal}


def _sum(out: lucid.Tensor) -> lucid.Tensor:
    return out.sum()


def _sum_to(full: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """``full`` summed down to ``shape``, the way a broadcast operand's gradient is."""
    while full.ndim > len(shape):
        full = full.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and full.shape[axis] != 1:
            full = full.sum(axis=axis, keepdims=True)
    return full


def _closed_form(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``d sum(a / b)`` in float64: ``1 / b`` and ``-(a / b) / b``, never ``b ** 2``."""
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    out_shape = np.broadcast(a64, b64).shape
    da = np.broadcast_to(1.0 / b64, out_shape)
    db = np.broadcast_to(-(a64 / b64) / b64, out_shape)
    return _sum_to(np.array(da), a.shape), _sum_to(np.array(db), b.shape)


def _step(
    kind: str,
    case: Case,
    magnitude: float,
    *,
    compiled: bool,
    dtype: lucid.dtype = lucid.float32,
) -> tuple[np.ndarray, np.ndarray | None]:
    """The gradients of ``sum(model(x))`` with respect to ``b`` and (if tracked) ``x``."""
    a, b = _values(case, magnitude)
    model = MODELS[kind](lucid.tensor(b, dtype=dtype, device=COMPILE_DEVICE)).to(COMPILE_DEVICE)
    x = lucid.tensor(a, dtype=dtype, device=COMPILE_DEVICE, requires_grad=case.a_tracked)
    model.zero_grad()
    if compiled:
        compiled_step(model, x, _sum)
    else:
        _sum(model(x)).backward()
    grad_b = model.b.grad.to(lucid.float32).numpy()
    return grad_b, (x.grad.to(lucid.float32).numpy() if case.a_tracked else None)


def _agree(got: np.ndarray, want: np.ndarray, rtol: float) -> None:
    assert got.shape == want.shape
    assert np.all(np.isfinite(got)), got
    np.testing.assert_allclose(got, want, rtol=rtol, atol=0)


def _check_against_truth(
    kind: str, case: Case, magnitude: float, dtype: lucid.dtype, rtol: float
) -> None:
    a, b = _values(case, magnitude)
    # The truth is taken from the values the dtype actually holds.
    held_a = lucid.tensor(a, dtype=dtype).to(lucid.float32).numpy()
    held_b = lucid.tensor(b, dtype=dtype).to(lucid.float32).numpy()
    want_a, want_b = _closed_form(held_a, held_b)
    got_b, got_a = _step(kind, case, magnitude, compiled=True, dtype=dtype)
    _agree(got_b, want_b, rtol)
    if got_a is not None:
        _agree(got_a, want_a, rtol)


def _magnitude_id(value: float) -> str:
    return f"{value:g}"


def test_the_ops_have_vjp_emitters() -> None:
    for name in ("div", "reciprocal"):
        assert (
            _C_engine.compile.vjp_registration_status(name)
            == _C_engine.compile.VjpRegistration.Registered
        )


@pytest.mark.parametrize(
    ("kind", "expected"),
    [("div", "div"), ("reciprocal", "reciprocal")],
)
def test_a_trace_records_the_op_under_test(kind: str, expected: str) -> None:
    case = CASES[0]
    a, b = _values(case, 1.0)
    model = MODELS[kind](lucid.tensor(b, device=COMPILE_DEVICE)).to(COMPILE_DEVICE)
    x = lucid.tensor(a, device=COMPILE_DEVICE)
    with compile_attempt(), recording(watch=(x,)) as tracer:
        model(x)
    assert expected in {node.name for node in tracer.graph.ops}


# ── float32: a compiled step against the float64 closed form ─────────────────


@pytest.mark.parametrize("magnitude", MAGNITUDES, ids=_magnitude_id)
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
@pytest.mark.parametrize("kind", MODELS)
def test_a_compiled_step_matches_the_closed_form(kind: str, case: Case, magnitude: float) -> None:
    _check_against_truth(kind, case, magnitude, lucid.float32, RTOL)


@pytest.mark.parametrize("kind", MODELS)
def test_a_zero_numerator_has_a_zero_divisor_gradient_where_b_squared_underflows(
    kind: str,
) -> None:
    """``0 * g / 0`` was NaN once ``b * b`` flushed; ``(g * (0 / b)) / b`` is exactly 0."""
    case = next(c for c in CASES if c.zero_numerator)
    for magnitude in (1e-30, 1e-20, 1.0, 1e20, 1e30):
        grad_b, _ = _step(kind, case, magnitude, compiled=True)
        assert np.all(grad_b == 0.0), (magnitude, grad_b)


# ── float32: a compiled step against the eager one ───────────────────────────


@pytest.mark.parametrize("magnitude", MAGNITUDES, ids=_magnitude_id)
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
@pytest.mark.parametrize("kind", MODELS)
def test_a_compiled_step_matches_the_eager_one(kind: str, case: Case, magnitude: float) -> None:
    got_b, got_a = _step(kind, case, magnitude, compiled=True)
    want_b, want_a = _step(kind, case, magnitude, compiled=False)
    _agree(got_b, want_b, RTOL)
    if got_a is not None and want_a is not None:
        _agree(got_a, want_a, RTOL)


# ── half precision: the same ordering in the dtype the graph computes in ─────

#: bfloat16 has float32's range, so the old spelling failed at the same magnitudes.
BFLOAT16_MAGNITUDES = [1e-30, 1e-20, 1e20, 1e30]
#: float16 (6e-5 .. 65504) held up before: MPSGraph evaluates a fused elementwise chain in
#: float32, so ``b * b`` never rounded to half.  These guard the new ordering's precision, three
#: half roundings in place of three.
FLOAT16_MAGNITUDES = [1e-2, 1.0, 1e2, 1e3]
HALF_CASES = [c for c in CASES if c.name in {"both_tracked", "divisor_broadcast"}]


@pytest.mark.parametrize(
    ("dtype", "magnitude", "rtol"),
    [(lucid.bfloat16, m, 2e-2) for m in BFLOAT16_MAGNITUDES]
    + [(lucid.float16, m, 5e-3) for m in FLOAT16_MAGNITUDES],
    ids=lambda v: _magnitude_id(v) if isinstance(v, float) else str(v),
)
@pytest.mark.parametrize("case", HALF_CASES, ids=lambda c: c.name)
@pytest.mark.parametrize("kind", MODELS)
def test_a_half_compiled_step_matches_the_closed_form(
    kind: str, case: Case, dtype: lucid.dtype, magnitude: float, rtol: float
) -> None:
    _check_against_truth(kind, case, magnitude, dtype, rtol)
