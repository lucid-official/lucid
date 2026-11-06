"""A compiled ``max`` / ``min`` reduction answers as eager does on ties and NaN (LCD-865).

The compiled VJP of the ``max`` / ``min`` reductions gave the whole gradient to
every element equal to the result, where eager's ``share_among_ties`` gives each
of a slice's ``k`` tied elements ``1 / k`` of it.  A step on a tie — a freshly
zero-initialised tensor, padded rows, a saturated activation, ``matrix_norm(a, 2)``
or ``cond`` on a repeated singular value — was ``k`` times too large.  The
compiled forward skipped NaN (MPSGraph's plain ``reductionMaximum``): ``[1, nan,
1]`` reduced to 1 where eager's is NaN, an all-NaN slice to ``-inf``, and the
gradient then went to the elements beside the NaN.

Every public reduction that reaches the engine's ``max`` / ``min`` is covered —
``max`` / ``min`` with no ``dim`` or a tuple, ``amax`` / ``amin``, ``aminmax``,
the ``±inf`` vector norms and the ``±1`` / ``±inf`` matrix norms,
``matrix_norm(a, 2)`` and ``cond`` — over every element and over an axis with
ties in some rows only, with and without ``keepdim``, through the three training
entry points, for two SGD steps (the second runs the cached executable on the
updated values), in float32, float16 and bfloat16.  ``x.max(dim)`` with an int
``dim`` gathers at its index and gives a tie's gradient to that element alone: a
control, as is a slice with no tie.  The forward values are held to eager's
element by element, an integer input included (it keeps the plain reduction).
"""

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import pytest

import lucid
import lucid.nn as nn
import lucid.optim as optim
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid.compile import fused_step, make_step
from lucid.compile._core.fallback import CompileFallbackWarning
from lucid.compile._entry.function import compiled_step
from lucid.compile._entry.fused_step import _FusedStep

_LA = lucid.linalg
_NAN = math.nan
_LR = 0.5
_STEPS = 2
#: Each far under the 0.25 that a whole gradient on a two-way tie adds to a step at lr 0.5.
_TOL = {lucid.float32: 1e-5, lucid.float16: 4e-3, lucid.bfloat16: 3e-2}
_DTYPE_IDS = {lucid.float32: "f32", lucid.float16: "f16", lucid.bfloat16: "bf16"}
_ENTRIES = ("make_step", "fused_step", "compiled_step")

_ROWS = [[1.0, 1.0, 0.0], [2.0, 3.0, 1.0]]  # a tie in the first row only
_COLS = [[1.0, 1.0, 0.0], [1.0, 3.0, 0.0]]  # ties in the first and last columns
_NAN_ROW = [[1.0, _NAN, 1.0], [2.0, 2.0, 0.0]]


@dataclass(frozen=True)
class _Case:
    values: object
    op: Callable[[lucid.Tensor], lucid.Tensor]
    #: Through ``svd``, a custom ``Function`` that ``compiled_step`` refuses; run in
    #: float32 only.
    spectral: bool = False


_CASES: dict[str, _Case] = {
    "max": _Case([1.0, 1.0, 1.0], lambda a: a.max()),
    "max-2d": _Case([[2.0, 0.0], [2.0, 2.0]], lambda a: a.max()),
    "max-zeros": _Case([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], lambda a: a.max()),
    "max-tuple": _Case([[2.0, 0.0], [2.0, 2.0]], lambda a: a.max(dim=(0, 1))),
    "max-tuple-keepdim": _Case(_ROWS, lambda a: a.max(dim=(1,), keepdim=True)),
    "min": _Case([[0.0, 1.0], [0.0, 0.0]], lambda a: a.min()),
    "amax": _Case([1.0, 1.0, 1.0], lambda a: lucid.amax(a)),
    "amax-dim1": _Case(_ROWS, lambda a: lucid.amax(a, dim=1)),
    "amax-dim1-keepdim": _Case(_ROWS, lambda a: lucid.amax(a, dim=1, keepdim=True)),
    "amax-dim-neg": _Case(_ROWS, lambda a: lucid.amax(a, dim=-1)),
    "amax-3d-dims02": _Case(
        [[[1.0, 2.0], [0.0, 1.0]], [[2.0, 0.0], [1.0, 1.0]]], lambda a: lucid.amax(a, dim=(0, 2))
    ),
    "amin": _Case([1.0, 1.0, 1.0], lambda a: lucid.amin(a)),
    "amin-dim0": _Case(_COLS, lambda a: lucid.amin(a, dim=0)),
    "amin-dim0-keepdim": _Case(_COLS, lambda a: lucid.amin(a, dim=0, keepdim=True)),
    "aminmax": _Case([2.0, 1.0, 2.0, 1.0], lambda a: lucid.stack(list(lucid.aminmax(a)))),
    "aminmax-dim1": _Case(
        [[1.0, 1.0, 0.0, 0.0], [2.0, 3.0, 1.0, 3.0]],
        lambda a: lucid.stack(list(lucid.aminmax(a, dim=1))),
    ),
    "vector_norm-inf": _Case([1.0, -1.0, 0.5], lambda a: _LA.vector_norm(a, math.inf)),
    "vector_norm-neg-inf": _Case([1.0, -1.0, 2.0], lambda a: _LA.vector_norm(a, -math.inf)),
    "vector_norm-inf-dim1": _Case(
        [[1.0, -1.0, 0.0], [2.0, 0.5, 0.0]], lambda a: _LA.vector_norm(a, math.inf, dim=1)
    ),
    "matrix_norm-inf": _Case([[1.0, -1.0], [2.0, 0.0]], lambda a: _LA.matrix_norm(a, math.inf)),
    "matrix_norm-neg-inf": _Case(
        [[1.0, -1.0], [2.0, 0.0]], lambda a: _LA.matrix_norm(a, -math.inf)
    ),
    "matrix_norm-1": _Case([[1.0, -1.0], [1.0, 1.0]], lambda a: _LA.matrix_norm(a, 1)),
    "matrix_norm-2-eye": _Case(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        lambda a: _LA.matrix_norm(a, 2),
        spectral=True,
    ),
    # The smallest singular value is tied: eager's diagonal is [1, -1.5, -1.5].
    "cond-diag-3-1-1": _Case(
        [[3.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], lambda a: _LA.cond(a), spectral=True
    ),
    # Controls: an int ``dim`` gathers at its index; a slice with no tie.
    "max-int-dim": _Case(_ROWS, lambda a: a.max(dim=1)[0]),
    "min-int-dim-keepdim": _Case(_COLS, lambda a: a.min(dim=0, keepdim=True)[0]),
    "amax-no-tie": _Case([1.0, 3.0, 2.0], lambda a: lucid.amax(a)),
    # A slice holding a NaN reduces to NaN and gives none of its elements a gradient.
    "max-nan": _Case([1.0, _NAN, 1.0], lambda a: a.max()),
    "min-nan": _Case([1.0, _NAN, 1.0], lambda a: a.min()),
    "max-all-nan": _Case([_NAN, _NAN], lambda a: a.max()),
    "amax-dim1-nan-row": _Case(_NAN_ROW, lambda a: lucid.amax(a, dim=1)),
    "amin-dim1-nan-row": _Case(_NAN_ROW, lambda a: lucid.amin(a, dim=1)),
    "vector_norm-inf-nan": _Case([1.0, _NAN, -1.0], lambda a: _LA.vector_norm(a, math.inf)),
}


class _OfParam(nn.Module):
    """``op(a)`` weighted element by element, so a gradient sent to the wrong slice shows."""

    def __init__(self, case: _Case, dtype: lucid.dtype, device: str) -> None:
        super().__init__()
        self.a = nn.Parameter(lucid.tensor(case.values, dtype=dtype, device=device))
        self.op = case.op
        with lucid.no_grad():
            shape = tuple(case.op(self.a).shape)
        n = math.prod(shape)
        weight = lucid.tensor([1.0 + k for k in range(n)], dtype=dtype, device=device)
        self.register_buffer("w", weight.reshape(shape))

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return (self.op(self.a) * self.w * x).sum().reshape(1)


def _loss(out: lucid.Tensor, target: lucid.Tensor) -> lucid.Tensor:
    return (out * target).sum()


@dataclass(frozen=True)
class _Step:
    loss: lucid.Tensor
    #: ``None`` for ``fused_step``: its backward runs inside the executable.
    grad: lucid.Tensor | None
    param: lucid.Tensor


def _snapshot(model: _OfParam, loss: lucid.Tensor, grad: lucid.Tensor | None) -> _Step:
    return _Step(loss.detach().cpu(), grad, model.a.detach().cpu())


def _grad_of(model: _OfParam) -> lucid.Tensor:
    grad = model.a.grad
    assert grad is not None
    return grad.detach().cpu()


def _eager(model: _OfParam, opt: optim.Optimizer, x: lucid.Tensor, t: lucid.Tensor) -> list[_Step]:
    steps = []
    for _ in range(_STEPS):
        opt.zero_grad()
        loss = _loss(model(x), t)
        loss.backward()
        grad = _grad_of(model)
        opt.step()
        steps.append(_snapshot(model, loss, grad))
    return steps


def _compiled(
    entry: str, model: _OfParam, opt: optim.Optimizer, x: lucid.Tensor, t: lucid.Tensor
) -> list[_Step]:
    steps = []
    if entry == "fused_step":
        fused = fused_step(model, _loss, opt)
        assert isinstance(fused, _FusedStep)
        for _ in range(_STEPS):
            steps.append(_snapshot(model, fused(x, t), None))
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
        grad = _grad_of(model)
        opt.step()
        steps.append(_snapshot(model, loss, grad))
    assert step is None or not step.eager_only
    return steps


def _assert_close(got: lucid.Tensor, want: lucid.Tensor, tol: float, why: str) -> None:
    assert tuple(got.shape) == tuple(want.shape), f"{why}: {got.shape} != {want.shape}"
    g = got.to(lucid.float32).reshape(-1).tolist()
    w = want.to(lucid.float32).reshape(-1).tolist()
    for p, q in zip(g, w, strict=True):
        same = math.isnan(p) if math.isnan(q) else math.isclose(p, q, rel_tol=tol, abs_tol=tol)
        assert same, f"{why}: compiled {g} != eager {w}"


def _train_cases() -> list[object]:
    out = []
    for name, case in _CASES.items():
        for entry in _ENTRIES:
            if case.spectral and entry == "compiled_step":
                continue
            for dtype, dt in _DTYPE_IDS.items():
                if case.spectral and dtype != lucid.float32:
                    continue
                out.append(pytest.param(name, entry, dtype, id=f"{name}-{entry}-{dt}"))
    return out


@pytest.mark.parametrize(("name", "entry", "dtype"), _train_cases())
def test_compiled_step_matches_eager_on_ties_and_nan(
    name: str, entry: str, dtype: lucid.dtype, device_gpu_only: str
) -> None:
    case = _CASES[name]
    x = lucid.ones(1, dtype=dtype, device=device_gpu_only)
    t = lucid.ones(1, dtype=dtype, device=device_gpu_only)
    eager_model = _OfParam(case, dtype, device_gpu_only)
    want = _eager(eager_model, optim.SGD(eager_model.parameters(), lr=_LR), x, t)
    model = _OfParam(case, dtype, device_gpu_only)
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompileFallbackWarning)
        got = _compiled(entry, model, optim.SGD(model.parameters(), lr=_LR), x, t)
    tol = _TOL[dtype]
    for k, (g, w) in enumerate(zip(got, want, strict=True)):
        _assert_close(g.loss.reshape(-1), w.loss.reshape(-1), tol, f"step {k} loss")
        if g.grad is not None:
            assert w.grad is not None
            _assert_close(g.grad, w.grad, tol, f"step {k} gradient")
        _assert_close(g.param, w.param, tol, f"step {k} parameter after SGD")


def test_a_tie_shares_the_gradient(device_gpu_only: str) -> None:
    """The defect's own numbers, independent of eager: a three-way tie gets thirds."""
    model = _OfParam(_CASES["max"], lucid.float32, device_gpu_only)
    step = make_step(model, _loss)
    one = lucid.ones(1, device=device_gpu_only)
    step(one, one).backward()
    assert not step.eager_only
    _assert_close(_grad_of(model), lucid.full((3,), 1.0 / 3.0), 1e-6, "max of [1, 1, 1]")


# ── forward values, element by element ──────────────────────────────────────


class _Apply(nn.Module):
    def __init__(self, op: Callable[[lucid.Tensor], lucid.Tensor]) -> None:
        super().__init__()
        self.op = op

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.op(x)


def _engine_norm(ord_: float, dims: list[int]) -> Callable[[lucid.Tensor], lucid.Tensor]:
    """The engine's own ``norm`` op; the public norms compose ``abs`` and ``max``."""
    return lambda x: _wrap(_C_engine.linalg.norm(_unwrap(x), ord_, dims, False))


_INT_ROWS = [[3, 1, 3], [0, 0, -2]]

_FORWARD: dict[str, tuple[object, lucid.dtype, Callable[[lucid.Tensor], lucid.Tensor]]] = {
    **{
        name: (case.values, lucid.float32, case.op)
        for name, case in _CASES.items()
        if not case.spectral
    },
    "max-nan-f16": ([1.0, _NAN, 1.0], lucid.float16, lambda a: a.max()),
    "amax-dim1-nan-row-f16": (_NAN_ROW, lucid.float16, lambda a: lucid.amax(a, dim=1)),
    "amin-dim1-nan-row-bf16": (_NAN_ROW, lucid.bfloat16, lambda a: lucid.amin(a, dim=1)),
    "min-all-nan": ([_NAN, _NAN], lucid.float32, lambda a: a.min()),
    "amax-inf-tie": ([math.inf, math.inf, 1.0], lucid.float32, lambda a: lucid.amax(a)),
    "int32-max": (_INT_ROWS, lucid.int32, lambda a: a.max()),
    "int32-amax-dim1": (_INT_ROWS, lucid.int32, lambda a: lucid.amax(a, dim=1)),
    "int64-amin-dim0": (_INT_ROWS, lucid.int64, lambda a: lucid.amin(a, dim=0)),
    "int64-min": (_INT_ROWS, lucid.int64, lambda a: a.min()),
    "engine-norm-inf-nan": ([1.0, _NAN, 3.0], lucid.float32, _engine_norm(math.inf, [])),
    "engine-norm-inf-all-nan": ([_NAN, _NAN], lucid.float32, _engine_norm(math.inf, [])),
    "engine-norm-inf-nan-row": (
        [[1.0, _NAN], [2.0, -3.0]],
        lucid.float32,
        _engine_norm(math.inf, [1]),
    ),
    "engine-norm-neg-inf-nan-row": (
        [[1.0, _NAN], [2.0, -3.0]],
        lucid.float32,
        _engine_norm(-math.inf, [1]),
    ),
}


@pytest.mark.parametrize("name", list(_FORWARD))
def test_compiled_forward_matches_eager(name: str, device_gpu_only: str) -> None:
    values, dtype, op = _FORWARD[name]
    x = lucid.tensor(values, dtype=dtype, device=device_gpu_only)
    want = op(x)
    compiled = lucid.compile(_Apply(op))
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompileFallbackWarning)
        with lucid.no_grad():
            got = [compiled(x) for _ in range(2)]
    info = compiled.cache_info()
    assert info["entries"] == 1 and not info["eager_only"]
    assert got[0].dtype == want.dtype
    for k, g in enumerate(got):
        _assert_close(g.cpu(), want.cpu(), 0.0, f"call {k}")
