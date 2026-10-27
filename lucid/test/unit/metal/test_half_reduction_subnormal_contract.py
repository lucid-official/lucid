"""Metal reductions flush subnormals by one rule, eager and compiled alike.

A float16 / bfloat16 reduction on Metal is the float32 reduction of its
widened input, rounded once (``backend/gpu/HalfAccumulation.h``).  The GPU
flushes float32 subnormals in arithmetic, so a bfloat16 subnormal result of
a sum over two or more elements, or of any mean or variance, is a signed
zero, exactly as for float32; a sum over one element is a copy and keeps
the value.  A float16 subnormal is a normal float32 and survives.  The
reference's GPU stream flushes the same float32 / bfloat16 results (a
one-element sum too) and keeps float16 subnormals.

``lucid.compile``'s graphs keep the rule (LCD-452), so every cell is
compared bit for bit between eager Metal, a ``CompiledModule`` and a
``make_step`` loss.  LCD-121: the half kernels returned a one-element mean
without the 1/1 scale that flushes it — a bfloat16 ``mean`` of 1e-40 kept
its bits eagerly and was 0 compiled — and a variance whose squares flushed
came out as -0.
"""

import struct
from collections.abc import Callable

import numpy as np
import pytest

import lucid
import lucid.nn as nn
from lucid._C import engine as _C_engine
from lucid.compile import make_step
from lucid.test._fixtures.devices import metal_available

pytestmark = pytest.mark.skipif(not metal_available(), reason="metal unavailable")

_DTYPES = [lucid.float32, lucid.bfloat16, lucid.float16]
#: A subnormal of each dtype.
_SUBNORMAL = {lucid.float32: 1e-40, lucid.bfloat16: 1e-40, lucid.float16: 1e-7}
#: A spread ``t`` whose variance ``t**2`` is subnormal in the dtype.
_SPREAD = {lucid.float32: 1e-20, lucid.bfloat16: 1e-20, lucid.float16: 3e-4}

_Reduce = Callable[[lucid.Tensor, object, bool], lucid.Tensor]
_OPS: dict[str, _Reduce] = {
    "sum": lambda t, d, k: t.sum(dim=d, keepdim=k),
    "mean": lambda t, d, k: t.mean(dim=d, keepdim=k),
    "nansum": lambda t, d, k: lucid.nansum(t, dim=d, keepdim=k),
    "nanmean": lambda t, d, k: lucid.nanmean(t, dim=d, keepdim=k),
    "var": lambda t, d, k: t.var(dim=d, keepdim=k, correction=0),
    "std": lambda t, d, k: t.std(dim=d, keepdim=k, correction=0),
}
_SPREAD_OPS = ("var", "std")

_Plan = tuple[str, tuple[int, ...], object, bool]
#: (id, shape, dim, keepdim) — one of each reduction plan.
_PLANS: list[_Plan] = [
    ("one-element", (1,), None, False),
    ("size1-axis", (3, 1), 1, False),
    ("size1-axis-keepdim", (3, 1), 1, True),
    ("size1-leading-axis", (1, 3), 0, False),
    ("size2-axis", (3, 2), 1, False),
    ("size2-axis-keepdim", (3, 2), 1, True),
    ("leading-axis", (2, 3), 0, False),
    ("all", (2, 2), None, False),
    ("all-keepdim", (2, 2), (0, 1), True),
    ("row-per-simdgroup", (300, 2), 1, False),
    ("split-and-merged", (4096,), None, False),
]
#: The plans with a scalar output, which a ``make_step`` loss can be.
_SCALAR_PLANS = [p for p in _PLANS if p[2] is None]


def _name(dtype: lucid.dtype) -> str:
    return str(dtype).split(".")[-1]


def _cells(plans: list[_Plan]) -> list[object]:
    """Every (dtype, op, plan, sign) cell."""
    cells: list[object] = []
    for dtype in _DTYPES:
        for op in _OPS:
            for plan in plans:
                for sign in (1.0, -1.0):
                    cells.append(
                        pytest.param(
                            dtype,
                            op,
                            plan,
                            sign,
                            id=f"{_name(dtype)}-{op}-{plan[0]}-{'neg' if sign < 0 else 'pos'}",
                        )
                    )
    return cells


def _bits(t: lucid.Tensor) -> list[int]:
    values = t.to("cpu").to(lucid.float32).numpy().reshape(-1)
    return [struct.unpack("<I", struct.pack("<f", float(v)))[0] for v in values]


def _input(op: str, dtype: lucid.dtype, shape: tuple[int, ...], sign: float) -> lucid.Tensor:
    """Every element the subnormal for a sum or mean; alternating ``±t``
    for a variance, so the variance is subnormal."""
    n = 1
    for extent in shape:
        n *= extent
    if op in _SPREAD_OPS:
        values = [sign * _SPREAD[dtype] * (1 if i % 2 == 0 else -1) for i in range(n)]
    else:
        values = [sign * _SUBNORMAL[dtype]] * n
    return lucid.tensor(values, dtype=dtype).to("metal").reshape(*shape)


class _ReduceModule(nn.Module):
    def __init__(self, fn: Callable[[lucid.Tensor], lucid.Tensor]) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fn(x)


class _Select(nn.Module):
    """``x`` itself through a select (arithmetic would flush it first),
    with a parameter for ``make_step`` to differentiate."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(lucid.zeros(1))

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return lucid.where(x == x, x, self.w.to(x.dtype))


@pytest.mark.parametrize(
    ("dtype", "op", "plan", "sign"),
    _cells(_PLANS),
)
def test_compiled_module_matches_eager_bits(
    dtype: lucid.dtype, op: str, plan: _Plan, sign: float
) -> None:
    _, shape, dim, keepdim = plan
    x = _input(op, dtype, shape, sign)
    fn = _OPS[op]

    def reduce(t: lucid.Tensor) -> lucid.Tensor:
        return fn(t, dim, keepdim)

    _C_engine.compile.session_cache_clear()
    got = lucid.compile(_ReduceModule(reduce).to("metal"))(x)
    assert _C_engine.compile.session_cache_size() >= 1, "fell back to eager"
    assert _bits(got) == _bits(reduce(x))


@pytest.mark.parametrize(
    ("dtype", "op", "plan", "sign"),
    _cells(_SCALAR_PLANS),
)
def test_make_step_loss_matches_eager_bits(
    dtype: lucid.dtype, op: str, plan: _Plan, sign: float
) -> None:
    _, shape, dim, keepdim = plan
    x = _input(op, dtype, shape, sign)
    fn = _OPS[op]

    def loss(y: lucid.Tensor) -> lucid.Tensor:
        return fn(y, dim, keepdim)

    model = _Select().to("metal")
    want = _bits(loss(model(x)))
    step = make_step(model, loss)
    got = step(x)
    assert not step.eager_only, "fell back to eager"
    assert _bits(got) == want


@pytest.mark.parametrize(
    ("dtype", "op", "plan", "sign"),
    _cells(_PLANS),
)
def test_eager_follows_the_flush_rule(
    dtype: lucid.dtype, op: str, plan: _Plan, sign: float
) -> None:
    """The rule itself, so eager and compiled cannot drift together."""
    _, shape, dim, keepdim = plan
    x = _input(op, dtype, shape, sign)
    got = _bits(_OPS[op](x, dim, keepdim))
    exact = _bits(_OPS[op](x.to("cpu").to(lucid.float64), dim, keepdim).to(dtype))
    copies = op == "sum" and len(got) == x.numel()
    if dtype == lucid.float16 or copies:
        # Float32 holds a float16 subnormal, its square and its mean; a sum
        # over one element is that element.
        assert got == exact
    elif op == "nansum" and len(got) == x.numel():
        # ``nansum`` selects through ``where(isnan(x), 0, x)`` before the
        # 1-element sum: M1's software bfloat16 select keeps the subnormal
        # (``exact``), while M3+/M4 native bfloat16 select flushes its
        # mantissa and keeps the sign bit.
        assert got in (exact, [b & 0x80000000 for b in exact])
    elif op in _SPREAD_OPS:
        assert got == [0] * len(got), "a flushed variance is +0"
    else:
        assert [b & 0x7FFFFFFF for b in got] == [0] * len(got)


@pytest.mark.parametrize("dtype", [lucid.bfloat16, lucid.float16], ids=_name)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_one_element_variance_of_a_non_finite_value_is_nan(
    dtype: lucid.dtype, value: float
) -> None:
    """It was 0: the one-element plan wrote zeros without reading ``x``."""
    x = lucid.tensor([[value]], dtype=dtype, device="metal")
    got = x.var(dim=1, correction=0).to("cpu").to(lucid.float32).item()
    assert got != got


def _every_value(dtype: lucid.dtype) -> lucid.Tensor:
    """Every bit pattern of a half dtype, as a (65536, 1) Metal tensor."""
    pats = np.arange(1 << 16, dtype=np.uint32)
    if dtype == lucid.float16:
        with np.errstate(invalid="ignore"):
            values = pats.astype(np.uint16).view(np.float16).astype(np.float32)
    else:
        values = (pats << 16).view(np.float32)
    return lucid.tensor(values.reshape(-1, 1)).to(dtype).to("metal")


def _bits_nan_as_one(t: lucid.Tensor) -> np.ndarray:
    values = t.to("cpu").to(lucid.float32).numpy().reshape(-1).astype(np.float32)
    bits = values.view(np.uint32).copy()
    bits[np.isnan(values)] = 0x7FC00000
    return bits


_ONE_ELEMENT_OPS: dict[str, Callable[[lucid.Tensor], lucid.Tensor]] = {
    "sum": lambda t: t.sum(dim=-1),
    "mean": lambda t: t.mean(dim=-1),
    "var": lambda t: t.var(dim=-1, correction=0),
}


@pytest.mark.parametrize("op", list(_ONE_ELEMENT_OPS))
@pytest.mark.parametrize("dtype", [lucid.bfloat16, lucid.float16], ids=_name)
def test_one_element_plan_is_the_widened_op_for_every_value(dtype: lucid.dtype, op: str) -> None:
    """The one-element plan computes in the half dtype, without the float32
    copy, and must still agree bit for bit (any NaN with any NaN) with the
    float32 op rounded once and with the compiled graph."""
    x = _every_value(dtype)
    reduce = _ONE_ELEMENT_OPS[op]
    got = _bits_nan_as_one(reduce(x))
    widened = _bits_nan_as_one(reduce(x.to(lucid.float32)).to("cpu").to(dtype))
    _C_engine.compile.session_cache_clear()
    compiled = _bits_nan_as_one(lucid.compile(_ReduceModule(reduce).to("metal"))(x))
    assert np.array_equal(got, widened), int((got != widened).sum())
    assert np.array_equal(got, compiled), int((got != compiled).sum())
