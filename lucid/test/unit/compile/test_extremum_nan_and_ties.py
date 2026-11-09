"""Compiled max/min-built ops answer as eager does on NaN and ties (LCD-873, LCD-874).

MPSGraph's plain ``maximum`` / ``minimum``, ``reLU`` and cumulative extremum *skip* a NaN:
``maximum(1, nan)`` was 1, ``relu(nan)`` 0 and ``cummax([1, nan, 2])`` carried the 2 past the
NaN, so a compiled model hid a layer that had gone bad where eager's loss is NaN.  The
compiled VJP of ``maximum`` / ``minimum`` gave a tie wholly to the first operand (eager
halves it) and a NaN pair no gradient at all (eager gives both operands all of it); the
``relu`` VJP dropped a NaN's gradient (eager passes it); and the ``cummax`` VJP matched on
equality, which a NaN never satisfies, sending the gradient to a neighbour.

Every op built on those extrema is tabled — ``maximum``, ``minimum`` (and ``hypot``, built
on both), ``relu``, ``relu6``, ``hardtanh``, ``hardsigmoid``, ``clip`` / ``clamp`` with one
or both bounds, ``normalize``, ``cummax``, ``cummin`` and ``max_pool`` in 1, 2 and 3 dims —
over a NaN in each position, ties (an input on a bound, equal operands, signed zeros, equal
infinities, a repeated window maximum) and ordinary values.  Forward values are held to
eager's element by element; the gradient through the three training entry points to
eager's, for two SGD steps, in float32, float16 and bfloat16.  Integer inputs keep the
plain operations and are controls.

A max pool carries a NaN as eager does (LCD-913): the stencil pool encodes NaN as ``+inf``
before it and decodes it after, inside the fused prologue / epilogue (no second pass).
"""

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid.compile import fused_step, make_step
from lucid.compile._core.fallback import CompileFallbackWarning
from lucid.compile._entry.function import compiled_step
from lucid.compile._entry.fused_step import _FusedStep

_NAN = math.nan
_INF = math.inf
_LR = 0.5
_STEPS = 2
#: Far under any value a wrong branch would give; MPSGraph's ``hardsigmoid`` and ``hypot`` differ
#: from eager's in the last place or two, which is not what is under test.
_TOL = {lucid.float32: 1e-5, lucid.float16: 4e-3, lucid.bfloat16: 3e-2}
_DTYPE_IDS = {lucid.float32: "f32", lucid.float16: "f16", lucid.bfloat16: "bf16"}


@dataclass(frozen=True)
class _Case:
    #: One list per operand; each becomes a parameter for the gradient test.
    inputs: tuple[list[float], ...]
    op: Callable[..., lucid.Tensor]
    #: Entries beyond ``make_step`` the gradient test also runs: maximum and minimum are
    #: held to all three, everything else to ``make_step`` alone.
    every_entry: bool = False


def _unary(op: Callable[[lucid.Tensor], lucid.Tensor], *rows: list[float]) -> list[_Case]:
    return [_Case((row,), op) for row in rows]


# A NaN in the first, a middle and the last position, then ordinary values and the
# boundaries themselves (an input that is a tie with the bound it is clamped to).
_NAN_FIRST = [_NAN, -1.0, 0.5, 2.0]
_NAN_MID = [-2.0, _NAN, 0.5, 8.0]
_NAN_LAST = [-0.5, 1.0, 3.0, _NAN]
_ORDINARY = [-2.0, -0.5, 0.5, 3.0]
_AT_BOUNDS = [0.0, -1.0, 1.0, 6.0]

_PAIRS: dict[str, tuple[list[float], list[float]]] = {
    "ordinary": ([1.0, 2.0, 3.0, -1.0], [0.0, 5.0, -1.0, -2.0]),
    "tie": ([1.0, 2.0, 3.0], [1.0, 1.0, 3.0]),
    "tie-all": ([2.0, 2.0], [2.0, 2.0]),
    "signed-zero": ([0.0, -0.0, 0.0, -0.0], [-0.0, 0.0, 0.0, -0.0]),
    "inf-tie": ([_INF, -_INF, _INF], [_INF, -_INF, 1.0]),
    "nan-a-first": ([_NAN, 1.0, 2.0], [0.5, 1.0, 3.0]),
    "nan-a-last": ([1.0, 2.0, _NAN], [0.5, 3.0, 3.0]),
    "nan-b-first": ([1.0, 2.0, 3.0], [_NAN, 1.0, 3.0]),
    "nan-b-last": ([1.0, 2.0, 3.0], [0.0, 1.0, _NAN]),
    "nan-pair": ([1.0, _NAN, _NAN], [_NAN, _NAN, 2.0]),
    "nan-and-tie": ([_NAN, 2.0, 5.0], [1.0, 2.0, _NAN]),
}


def _pair_cases(op: Callable[..., lucid.Tensor], every_entry: bool = True) -> dict[str, _Case]:
    return {name: _Case((a, b), op, every_entry) for name, (a, b) in _PAIRS.items()}


def _pooled(shape: tuple[int, ...], pool: Callable[..., lucid.Tensor], *args: object):
    return lambda a: pool(a.reshape(shape), *args)


_NAN_GRID = [1.0, _NAN, 2.0, 3.0, 4.0, 4.0, _NAN, 0.0, 5.0, 5.0, 1.0, 1.0, 2.0, 3.0, 3.0, _NAN]
_TIE_GRID = [2.0, 2.0, 1.0, 3.0, 3.0, 1.0, 2.0, 2.0, 0.0, 4.0, 4.0, 4.0, 1.0, 1.0, 1.0, 1.0]
_PLAIN_GRID = [float((7 * i) % 11) for i in range(16)]
_TWO_NANS = [_NAN, _NAN, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

_CASES: dict[str, _Case] = {}


def _register(prefix: str, cases: dict[str, _Case]) -> None:
    for name, case in cases.items():
        _CASES[f"{prefix}-{name}"] = case


def _register_rows(prefix: str, op: Callable[[lucid.Tensor], lucid.Tensor]) -> None:
    rows = {
        "nan-first": _NAN_FIRST,
        "nan-mid": _NAN_MID,
        "nan-last": _NAN_LAST,
        "ordinary": _ORDINARY,
        "at-bounds": _AT_BOUNDS,
    }
    _register(prefix, {name: _Case((row,), op) for name, row in rows.items()})


_register("maximum", _pair_cases(lambda a, b: lucid.maximum(a, b)))
_register("minimum", _pair_cases(lambda a, b: lucid.minimum(a, b)))
_register("hypot", _pair_cases(lambda a, b: lucid.hypot(a, b), every_entry=False))
_register_rows("relu", lambda a: F.relu(a))
_register_rows("relu6", lambda a: F.relu6(a))
_register_rows("hardtanh", lambda a: F.hardtanh(a))
_register_rows("hardtanh-0-6", lambda a: F.hardtanh(a, 0.0, 6.0))
_register_rows("hardsigmoid", lambda a: F.hardsigmoid(a))
_register_rows("clip", lambda a: lucid.clip(a, -1.0, 2.0))
_register_rows("clamp-min", lambda a: lucid.clamp(a, min=0.0))
_register_rows("clamp-max", lambda a: lucid.clamp(a, max=1.0))
_register(
    "cummax",
    {
        "nan-first": _Case(([_NAN, 1.0, 2.0, 3.0],), lambda a: lucid.cummax(a, 0)[0]),
        "nan-mid": _Case(([1.0, _NAN, 2.0, 3.0],), lambda a: lucid.cummax(a, 0)[0]),
        "nan-last": _Case(([1.0, 3.0, 2.0, _NAN],), lambda a: lucid.cummax(a, 0)[0]),
        "two-nans": _Case(([1.0, _NAN, 2.0, _NAN, 3.0],), lambda a: lucid.cummax(a, 0)[0]),
        "ties": _Case(([1.0, 1.0, 0.5, 1.0, 2.0, 2.0],), lambda a: lucid.cummax(a, 0)[0]),
        "ordinary": _Case(([1.0, 3.0, 2.0, 3.0, 5.0],), lambda a: lucid.cummax(a, 0)[0]),
        "2d-rows": _Case(
            ([[1.0, _NAN, 2.0], [3.0, 1.0, 2.0], [_NAN, 0.0, 0.0]],),
            lambda a: lucid.cummax(a, 1)[0],
        ),
    },
)
_register(
    "cummin",
    {
        "nan-first": _Case(([_NAN, 1.0, 2.0, 3.0],), lambda a: lucid.cummin(a, 0)[0]),
        "nan-mid": _Case(([3.0, _NAN, 2.0, 1.0],), lambda a: lucid.cummin(a, 0)[0]),
        "nan-last": _Case(([3.0, 1.0, 2.0, _NAN],), lambda a: lucid.cummin(a, 0)[0]),
        "ties": _Case(([3.0, 3.0, 4.0, 3.0, 1.0, 1.0],), lambda a: lucid.cummin(a, 0)[0]),
        "ordinary": _Case(([3.0, 1.0, 2.0, 1.0, 0.0],), lambda a: lucid.cummin(a, 0)[0]),
    },
)
_register(
    "max_pool1d",
    {
        "nan-window": _Case(([1.0, _NAN, 2.0, 3.0],), _pooled((1, 1, 4), F.max_pool1d, 2)),
        "nan-overlap": _Case(
            ([1.0, 2.0, _NAN, 3.0, 4.0, 0.0],), _pooled((1, 1, 6), F.max_pool1d, 3, 1)
        ),
        "all-nan": _Case(([_NAN, _NAN, 2.0, 3.0],), _pooled((1, 1, 4), F.max_pool1d, 2)),
        "nan-in-a-ceil-window": _Case(
            ([1.0, 2.0, 3.0, 4.0, _NAN],), _pooled((1, 1, 5), F.max_pool1d, 2, 2, 0, 1, True)
        ),
        "nan-beside-inf": _Case(
            ([_INF, _NAN, _NAN, _INF, 1.0, 2.0],), _pooled((1, 1, 6), F.max_pool1d, 2)
        ),
        "inf": _Case(([_INF, 1.0, _INF, _INF, -_INF, -_INF],), _pooled((1, 1, 6), F.max_pool1d, 2)),
        "tie": _Case(([2.0, 2.0, 3.0, 3.0],), _pooled((1, 1, 4), F.max_pool1d, 2)),
        "ordinary": _Case(([1.0, 2.0, 5.0, 3.0],), _pooled((1, 1, 4), F.max_pool1d, 2)),
    },
)
_register(
    "max_pool2d",
    {
        "nan-windows": _Case((_NAN_GRID,), _pooled((1, 1, 4, 4), F.max_pool2d, 2)),
        "two-nans-in-a-window": _Case((_TWO_NANS,), _pooled((1, 1, 4, 4), F.max_pool2d, 2)),
        "nan-padded": _Case(
            ([-1.0, _NAN, -2.0, -3.0] * 4,), _pooled((1, 1, 4, 4), F.max_pool2d, 3, 1, 1)
        ),
        "tie": _Case((_TIE_GRID,), _pooled((1, 1, 4, 4), F.max_pool2d, 2)),
        "ordinary": _Case((_PLAIN_GRID,), _pooled((1, 1, 4, 4), F.max_pool2d, 2)),
        "adaptive": _Case((_NAN_GRID,), _pooled((1, 1, 4, 4), F.adaptive_max_pool2d, 2)),
    },
)
_register(
    "max_pool3d",
    {
        "nan-windows": _Case((_NAN_GRID,), _pooled((1, 1, 2, 2, 4), F.max_pool3d, 2)),
        "tie": _Case((_TIE_GRID,), _pooled((1, 1, 2, 2, 4), F.max_pool3d, 2)),
    },
)
_register(
    "normalize",
    {
        "nan-row": _Case(
            ([1.0, 2.0, 2.0, _NAN, 3.0, 4.0],), lambda a: F.normalize(a.reshape(2, 3), dim=1)
        ),
        "ordinary": _Case(
            ([1.0, 2.0, 2.0, 0.0, 3.0, 4.0],), lambda a: F.normalize(a.reshape(2, 3), dim=1)
        ),
    },
)


# ── forward values, element by element ──────────────────────────────────────


class _Apply(nn.Module):
    def __init__(self, op: Callable[..., lucid.Tensor]) -> None:
        super().__init__()
        self.op = op

    def forward(self, *x: lucid.Tensor) -> lucid.Tensor:
        return self.op(*x)


def _assert_close(got: lucid.Tensor, want: lucid.Tensor, tol: float, why: str) -> None:
    assert tuple(got.shape) == tuple(want.shape), f"{why}: {got.shape} != {want.shape}"
    g = got.to(lucid.float32).reshape(-1).tolist()
    w = want.to(lucid.float32).reshape(-1).tolist()
    for p, q in zip(g, w, strict=True):
        same = math.isnan(p) if math.isnan(q) else math.isclose(p, q, rel_tol=tol, abs_tol=tol)
        assert same, f"{why}: compiled {g} != eager {w}"
        # ``isclose`` cannot see the sign of a zero; the rule (max of a tie is +0, min is -0)
        # is a property of the value, so it is held element by element (LCD-879).
        if not math.isnan(q) and q == 0.0 and p == 0.0:
            assert math.copysign(1.0, p) == math.copysign(
                1.0, q
            ), f"{why}: compiled {g} != eager {w} in the sign of a zero"


#: A compiled element-wise ``maximum`` / ``minimum`` hands a signed-zero tie to its second
#: operand (MPSGraph's plain op), where eager's answer is +0 / -0 whichever operand held it.
#: A select-free fix-up that keeps the NaN carrier is not known: both ops sit on the
#: autodiff-safe list, and MPSGraph's autodiff aborts on a select.  Documented in
#: ``op-compile-extremum-nan-ties``; the strict xfail turns the fix into an XPASS.
_ZERO_SIGN_UNFIXED = {"maximum-signed-zero", "minimum-signed-zero"}


def _forward_cases() -> list[object]:
    unfixed = pytest.mark.xfail(
        strict=True, reason="LCD-930: compiled maximum/minimum keep the second operand's zero"
    )
    return [
        pytest.param(
            name, dtype, id=f"{name}-{dt}", marks=[unfixed] if name in _ZERO_SIGN_UNFIXED else []
        )
        for name in _CASES
        for dtype, dt in _DTYPE_IDS.items()
    ]


@pytest.mark.parametrize(("name", "dtype"), _forward_cases())
def test_compiled_forward_matches_eager(
    name: str, dtype: lucid.dtype, device_gpu_only: str
) -> None:
    case = _CASES[name]
    xs = [lucid.tensor(v, dtype=dtype, device=device_gpu_only) for v in case.inputs]
    want = case.op(*xs)
    compiled = lucid.compile(_Apply(case.op))
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompileFallbackWarning)
        with lucid.no_grad():
            got = [compiled(*xs) for _ in range(2)]
    info = compiled.cache_info()
    assert info["entries"] == 1 and not info["eager_only"]
    for k, g in enumerate(got):
        _assert_close(g.cpu(), want.cpu(), _TOL[dtype], f"call {k}")


_INT_CONTROLS: dict[str, tuple[tuple[list[int], ...], Callable[..., lucid.Tensor]]] = {
    "maximum": (([3, 1, -2, 7], [3, 4, -5, 0]), lambda a, b: lucid.maximum(a, b)),
    "minimum": (([3, 1, -2, 7], [3, 4, -5, 0]), lambda a, b: lucid.minimum(a, b)),
    "relu": (([3, 0, -2, 7],), lambda a: F.relu(a)),
    "clip": (([3, 0, -2, 7],), lambda a: lucid.clip(a, -1, 4)),
    "cummax": (([1, 3, 2, 3, 5],), lambda a: lucid.cummax(a, 0)[0]),
    "cummin": (([3, 1, 2, 1, 0],), lambda a: lucid.cummin(a, 0)[0]),
}


@pytest.mark.parametrize("name", list(_INT_CONTROLS))
def test_integer_inputs_keep_the_plain_operations(name: str, device_gpu_only: str) -> None:
    values, op = _INT_CONTROLS[name]
    xs = [lucid.tensor(v, dtype=lucid.int32, device=device_gpu_only) for v in values]
    want = op(*xs)
    compiled = lucid.compile(_Apply(op))
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompileFallbackWarning)
        with lucid.no_grad():
            got = [compiled(*xs) for _ in range(2)]
    assert got[0].dtype == want.dtype
    for k, g in enumerate(got):
        _assert_close(g.cpu(), want.cpu(), 0.0, f"call {k}")


# ── gradient through the training entry points ──────────────────────────────


class _OfParams(nn.Module):
    """``op(*params)`` weighted element by element, so a gradient sent to the wrong place shows."""

    def __init__(self, case: _Case, dtype: lucid.dtype, device: str) -> None:
        super().__init__()
        self.ps = nn.ParameterList(
            [nn.Parameter(lucid.tensor(v, dtype=dtype, device=device)) for v in case.inputs]
        )
        self.op = case.op
        with lucid.no_grad():
            shape = tuple(case.op(*self.ps).shape)
        weight = lucid.tensor(
            [1.0 + k for k in range(math.prod(shape))], dtype=dtype, device=device
        )
        self.register_buffer("w", weight.reshape(shape))

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return (self.op(*self.ps) * self.w * x).sum().reshape(1)


def _loss(out: lucid.Tensor, target: lucid.Tensor) -> lucid.Tensor:
    return (out * target).sum()


@dataclass(frozen=True)
class _Step:
    loss: lucid.Tensor
    #: ``None`` for ``fused_step``: its backward runs inside the executable.
    grads: list[lucid.Tensor] | None
    params: list[lucid.Tensor]


def _snapshot(model: _OfParams, loss: lucid.Tensor, with_grads: bool) -> _Step:
    grads = None
    if with_grads:
        grads = []
        for p in model.ps:
            assert p.grad is not None
            grads.append(p.grad.detach().cpu())
    return _Step(loss.detach().cpu(), grads, [p.detach().cpu() for p in model.ps])


def _eager(model: _OfParams, opt: optim.Optimizer, x: lucid.Tensor, t: lucid.Tensor) -> list[_Step]:
    steps = []
    for _ in range(_STEPS):
        opt.zero_grad()
        loss = _loss(model(x), t)
        loss.backward()
        snap = _snapshot(model, loss, True)
        opt.step()
        steps.append(_Step(snap.loss, snap.grads, [p.detach().cpu() for p in model.ps]))
    return steps


def _compiled(
    entry: str, model: _OfParams, opt: optim.Optimizer, x: lucid.Tensor, t: lucid.Tensor
) -> list[_Step]:
    steps = []
    if entry == "fused_step":
        fused = fused_step(model, _loss, opt)
        assert isinstance(fused, _FusedStep)
        for _ in range(_STEPS):
            steps.append(_snapshot(model, fused(x, t), False))
        info = fused.cache_info()
        assert info["entries"] == 1 and not info["eager_only"]
        return steps
    step = make_step(model, _loss) if entry == "make_step" else None
    for _ in range(_STEPS):
        opt.zero_grad()
        if step is not None:
            loss = step(x, t)
            loss.backward()
        else:
            loss = compiled_step(model, x, lambda y: _loss(y, t))
        snap = _snapshot(model, loss, True)
        opt.step()
        steps.append(_Step(snap.loss, snap.grads, [p.detach().cpu() for p in model.ps]))
    assert step is None or not step.eager_only
    return steps


def _train_cases() -> list[object]:
    out = []
    for name, case in _CASES.items():
        entries = (
            ("make_step", "fused_step", "compiled_step") if case.every_entry else ("make_step",)
        )
        for entry in entries:
            for dtype, dt in _DTYPE_IDS.items():
                out.append(pytest.param(name, entry, dtype, id=f"{name}-{entry}-{dt}"))
    return out


@pytest.mark.parametrize(("name", "entry", "dtype"), _train_cases())
def test_compiled_step_matches_eager_gradient(
    name: str, entry: str, dtype: lucid.dtype, device_gpu_only: str
) -> None:
    case = _CASES[name]
    x = lucid.ones(1, dtype=dtype, device=device_gpu_only)
    t = lucid.ones(1, dtype=dtype, device=device_gpu_only)
    eager_model = _OfParams(case, dtype, device_gpu_only)
    want = _eager(eager_model, optim.SGD(eager_model.parameters(), lr=_LR), x, t)
    model = _OfParams(case, dtype, device_gpu_only)
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompileFallbackWarning)
        got = _compiled(entry, model, optim.SGD(model.parameters(), lr=_LR), x, t)
    tol = _TOL[dtype]
    for k, (g, w) in enumerate(zip(got, want, strict=True)):
        _assert_close(g.loss.reshape(-1), w.loss.reshape(-1), tol, f"step {k} loss")
        if g.grads is not None:
            assert w.grads is not None
            for i, (gg, wg) in enumerate(zip(g.grads, w.grads, strict=True)):
                _assert_close(gg, wg, tol, f"step {k} gradient of operand {i}")
        for i, (gp, wp) in enumerate(zip(g.params, w.params, strict=True)):
            _assert_close(gp, wp, tol, f"step {k} operand {i} after SGD")


def test_a_tie_splits_the_gradient_and_a_nan_pair_gives_all_of_it(device_gpu_only: str) -> None:
    """LCD-874's own numbers, independent of eager."""
    tie = _OfParams(_CASES["maximum-tie"], lucid.float32, device_gpu_only)
    step = make_step(tie, _loss)
    one = lucid.ones(1, device=device_gpu_only)
    step(one, one).backward()
    assert not step.eager_only
    a, b = (p.grad.cpu() for p in tie.ps if p.grad is not None)
    _assert_close(a, lucid.tensor([0.5, 2.0, 1.5]), 1e-6, "tie: a")
    _assert_close(b, lucid.tensor([0.5, 0.0, 1.5]), 1e-6, "tie: b")

    both = _OfParams(_CASES["maximum-nan-pair"], lucid.float32, device_gpu_only)
    step = make_step(both, _loss)
    step(one, one).backward()
    a, b = (p.grad.cpu() for p in both.ps if p.grad is not None)
    _assert_close(a, lucid.tensor([1.0, 2.0, 3.0]), 1e-6, "NaN pair: a")
    _assert_close(b, lucid.tensor([1.0, 2.0, 3.0]), 1e-6, "NaN pair: b")


@pytest.mark.parametrize("dtype", list(_DTYPE_IDS), ids=list(_DTYPE_IDS.values()))
def test_max_pool_top_code_is_shared_with_infinity(
    dtype: lucid.dtype, device_gpu_only: str
) -> None:
    """The one window that reads off, besides the second-lowest (LCD-913): a true maximum of
    exactly the dtype's largest finite value comes back ``+inf``, because NaN needed a code
    above ``+inf`` and ``+inf`` took the largest finite value's.  Pinned so a change is seen."""
    top = {lucid.float32: 3.4028234663852886e38, lucid.float16: 65504.0}.get(
        dtype, 3.3895313892515355e38
    )
    x = lucid.tensor([[[top, 1.0, 2.0, 3.0, _INF, 1.0]]], dtype=dtype, device=device_gpu_only)
    compiled = lucid.compile(_Apply(lambda a: F.max_pool1d(a, 2)))
    with lucid.no_grad():
        got = [compiled(x) for _ in range(2)][-1]
    assert got.to(lucid.float32).reshape(-1).tolist() == [_INF, 3.0, _INF]


# ── the sign of a zero extremum, and the tie share past a half dtype's exact range (LCD-879) ──

_NEG0, _POS0 = -0.0, 0.0
_ZERO_ROWS = {
    "neg-pos": [_NEG0, _POS0],
    "pos-neg": [_POS0, _NEG0],
    "all-neg": [_NEG0] * 5,
    "all-pos": [_POS0] * 5,
    "below": [-1.0, _NEG0, _POS0, -2.0, _POS0, _NEG0],
    "above": [1.0, _NEG0, _POS0, 2.0, _NEG0, _POS0],
    "wide": [(_NEG0, _POS0)[(i * 7 + i // 3) % 2] for i in range(64)],
}


def _ranked(values: list[float], is_max: bool) -> float:
    """max of a signed-zero tie is +0, min is -0; any other extremum is the value."""
    best = max(values) if is_max else min(values)
    if best != 0.0:
        return best
    signs = {math.copysign(1.0, v) for v in values if v == 0.0}
    return (0.0 if 1.0 in signs else -0.0) if is_max else (-0.0 if -1.0 in signs else 0.0)


def _signed_equal(got: float, want: float) -> bool:
    return got == want and math.copysign(1.0, got) == math.copysign(1.0, want)


_REDUCTION_ZERO_REASON = (
    "LCD-930: Metal and compiled max/min reductions keep the backend's zero order "
    "(exact form costs one more pass, +47-77%)"
)


@pytest.mark.parametrize("dtype", list(_DTYPE_IDS), ids=list(_DTYPE_IDS.values()))
@pytest.mark.parametrize("row", ["all-neg", "all-pos"])
@pytest.mark.parametrize("name", ["max", "min", "amax", "amin"])
def test_compiled_reduction_of_one_signed_zero_keeps_its_sign(
    dtype: lucid.dtype, row: str, name: str, device_gpu_only: str
) -> None:
    """No tie between the two zeros, so no order to follow: the sign is the input's."""
    v = _ZERO_ROWS[row]
    x = lucid.tensor(v, dtype=dtype, device=device_gpu_only)
    op: Callable[[lucid.Tensor], lucid.Tensor] = (
        (lambda t: getattr(t, name)()) if name in ("max", "min") else getattr(lucid, name)
    )
    compiled = lucid.compile(_Apply(op))
    with lucid.no_grad():
        got = [compiled(x) for _ in range(2)][-1].item()
    assert _signed_equal(got, v[0])


@pytest.mark.xfail(strict=True, reason=_REDUCTION_ZERO_REASON)
@pytest.mark.parametrize("dtype", list(_DTYPE_IDS), ids=list(_DTYPE_IDS.values()))
@pytest.mark.parametrize(
    ("name", "row", "is_max"),
    [
        ("max", "pos-neg", True),
        ("amax", "pos-neg", True),
        ("min", "neg-pos", False),
        ("amin", "neg-pos", False),
    ],
)
# (the compiled reduction hands a tie to its last element, eager Metal's to its first)
def test_compiled_reduction_zero_order_is_a_known_difference(
    dtype: lucid.dtype, name: str, row: str, is_max: bool, device_gpu_only: str
) -> None:
    v = _ZERO_ROWS[row]
    x = lucid.tensor(v, dtype=dtype, device=device_gpu_only)
    op: Callable[[lucid.Tensor], lucid.Tensor] = (
        (lambda t: getattr(t, name)()) if name in ("max", "min") else getattr(lucid, name)
    )
    compiled = lucid.compile(_Apply(op))
    with lucid.no_grad():
        got = [compiled(x) for _ in range(2)][-1].item()
    assert _signed_equal(got, _ranked(v, is_max))


@pytest.mark.xfail(
    strict=True, reason="LCD-930: compiled clip gives a zero tie to its bound (eager ranks it)"
)
@pytest.mark.parametrize(
    ("op", "bound", "is_max"),
    [
        pytest.param(lambda a: a.clamp(min=-0.0), -0.0, True, id="floor-minus-zero"),
        pytest.param(lambda a: a.clamp(max=0.0), 0.0, False, id="cap-plus-zero"),
    ],
)
def test_compiled_clamp_to_the_opposite_zero_is_a_known_difference(
    op: Callable[[lucid.Tensor], lucid.Tensor], bound: float, is_max: bool, device_gpu_only: str
) -> None:
    v = [_POS0, _POS0, _NEG0, _NEG0]
    x = lucid.tensor(v, device=device_gpu_only)
    want = [_ranked([e, bound], is_max) for e in v]
    assert all(map(_signed_equal, op(x).tolist(), want))  # eager holds the rule
    compiled = lucid.compile(_Apply(op))
    with lucid.no_grad():
        got = [compiled(x) for _ in range(2)][-1].tolist()
    assert all(map(_signed_equal, got, want)), f"compiled {got} != rule {want}"


@pytest.mark.parametrize("op_name", ["max", "min", "amax", "amin"])
@pytest.mark.parametrize(("dtype", "n"), [(lucid.bfloat16, 257), (lucid.float16, 2049)])
def test_compiled_tie_share_is_one_over_the_true_count(
    op_name: str, dtype: lucid.dtype, n: int, device_gpu_only: str
) -> None:
    """The compiled VJP already counted past the half dtype's exact range; eager now does too."""
    ops: dict[str, Callable[[lucid.Tensor], lucid.Tensor]] = {
        "max": lambda p: p.max(),
        "min": lambda p: p.min(),
        "amax": lambda p: lucid.amax(p),
        "amin": lambda p: lucid.amin(p),
    }

    class _Ties(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.p = nn.Parameter(lucid.ones(n, dtype=dtype, device=device_gpu_only))

        def forward(self, x: lucid.Tensor) -> lucid.Tensor:
            return (ops[op_name](self.p) * x).sum().reshape(1)

    one = lucid.ones(1, dtype=dtype, device=device_gpu_only)
    want = lucid.tensor([1.0 / n], dtype=dtype).to(lucid.float32).item()
    model = _Ties()
    step = make_step(model, _loss)
    step(one, one).backward()
    assert not step.eager_only
    assert model.p.grad is not None
    assert model.p.grad.to(lucid.float32).cpu().tolist() == [want] * n
