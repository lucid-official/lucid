"""Every full reduction, at every rank, keeps one shape from eager to compiled (LCD-804).

A reduction over every axis binds a value whose shape the trace also records.  When the
two disagreed (a keepdim-shaped ``(1, ..., 1)`` in the graph against ``()`` in the trace)
the next broadcast promoted its result, a compiled step summed the wrong axis, and
``cat`` / ``stack`` aborted.  The invariant is now "a bound value has its trace's shape",
so this table walks the whole family instead of the one symptom:

* the eager result has the reference's shape, for ``keepdim`` False and True;
* a compiled function computes the eager values when the reduction feeds a broadcast;
* ``stack`` / ``cat`` of reduced scalars is the eager one;
* a compiled step's gradient through the reduction is the eager gradient.

Compile is Metal-only; the eager shape and gradient rows also run on the CPU stream.
"""

from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType

import pytest

import lucid
import lucid.nn as nn
from lucid.test._fixtures.devices import metal_available
from lucid.test.unit.compile._helpers import COMPILE_DEVICE

_SHAPES = {1: (5,), 2: (2, 3), 3: (2, 3, 2), 4: (2, 3, 2, 2)}
_RANKS = sorted(_SHAPES)


@dataclass(frozen=True)
class _Reduction:
    """One reduction and the spellings of "all axes" the reference accepts for it."""

    name: str
    call: Callable[[ModuleType, object, dict[str, object]], object]
    #: ``"none"`` (no ``dim``), ``"all"`` (``dim=`` every axis) or both.
    dims: tuple[str, ...] = ("none", "all")
    keepdim: bool = True
    #: ``keepdim=True`` without ``dim`` is a call the reference accepts.
    keep_none: bool = False
    differentiable: bool = True


def _method(name: str) -> Callable[[ModuleType, object, dict[str, object]], object]:
    return lambda m, x, kw: getattr(m, name)(x, **kw)


def _norm(m: ModuleType, x: object, kw: dict[str, object]) -> object:
    return m.linalg.norm(x, **kw)


_TABLE = [
    _Reduction("sum", _method("sum")),
    _Reduction("mean", _method("mean")),
    _Reduction("nansum", _method("nansum")),
    _Reduction("prod", _method("prod"), dims=("none",), keepdim=False),
    _Reduction("amax", _method("amax"), keep_none=True),
    _Reduction("amin", _method("amin"), keep_none=True),
    _Reduction("max", _method("max"), dims=("none",), keepdim=False),
    _Reduction("min", _method("min"), dims=("none",), keepdim=False),
    _Reduction("var", _method("var"), keep_none=True),
    _Reduction("std", _method("std"), keep_none=True),
    _Reduction("logsumexp", _method("logsumexp"), dims=("all",)),
    _Reduction("norm", _norm, dims=("none",), keep_none=True),
    _Reduction("all", _method("all"), differentiable=False),
    _Reduction("any", _method("any"), differentiable=False),
    _Reduction("argmax", _method("argmax"), dims=("none",), keep_none=True, differentiable=False),
    _Reduction("argmin", _method("argmin"), dims=("none",), keep_none=True, differentiable=False),
    _Reduction("count_nonzero", _method("count_nonzero"), keepdim=False, differentiable=False),
]
_BY_NAME = {r.name: r for r in _TABLE}


def _cases(*, differentiable_only: bool = False) -> list[tuple[str, str, bool, int]]:
    out = []
    for r in _TABLE:
        if differentiable_only and not r.differentiable:
            continue
        for dims in r.dims:
            for keepdim in (False, True) if r.keepdim else (False,):
                if keepdim and dims == "none" and not r.keep_none:
                    continue
                out.extend((r.name, dims, keepdim, rank) for rank in _RANKS)
    return out


def _ids(case: tuple[str, str, bool, int]) -> str:
    name, dims, keepdim, rank = case
    return f"{name}-{dims}-{'keep' if keepdim else 'drop'}-r{rank}"


def _kwargs(rank: int, dims: str, keepdim: bool) -> dict[str, object]:
    kw: dict[str, object] = {}
    if dims == "all":
        kw["dim"] = tuple(range(rank))
    if keepdim:
        kw["keepdim"] = True
    return kw


def _data(m: ModuleType, rank: int, **kw: object) -> object:
    shape = _SHAPES[rank]
    n = 1
    for d in shape:
        n *= d
    return m.linspace(0.5, 1.5, n, **kw).reshape(*shape)


def _reduce(m: ModuleType, case: tuple[str, str, bool, int], a: object) -> object:
    name, dims, keepdim, rank = case
    return _BY_NAME[name].call(m, a, _kwargs(rank, dims, keepdim))


def _floats(t: lucid.Tensor) -> lucid.Tensor:
    return t.to(lucid.float32)


def _flat(values: object) -> list[float]:
    if isinstance(values, list):
        return [leaf for item in values for leaf in _flat(item)]
    return [float(values)]


def _close(got: lucid.Tensor, want: lucid.Tensor, why: str) -> None:
    assert got.shape == want.shape, f"{why}: {got.shape} != {want.shape}"
    assert lucid.allclose(
        _floats(got), _floats(want), rtol=1e-4, atol=1e-4
    ), f"{why}: {got.tolist()} != {want.tolist()}"


# ── eager: the shape contract, against the reference ─────────────────────────


@pytest.mark.parametrize("case", _cases(), ids=_ids)
def test_the_eager_shape_and_value_match_the_reference(
    ref: ModuleType, device: str, case: tuple[str, str, bool, int]
) -> None:
    got = _reduce(lucid, case, _data(lucid, case[3], device=device))
    want = _reduce(ref, case, _data(ref, case[3]))
    assert got.shape == tuple(want.shape)
    assert _flat(got.tolist()) == pytest.approx(_flat(want.tolist()), rel=1e-4, abs=1e-5)


@pytest.mark.parametrize("case", _cases(differentiable_only=True), ids=_ids)
def test_the_eager_gradient_matches_the_reference(
    ref: ModuleType, device: str, case: tuple[str, str, bool, int]
) -> None:
    a = _data(lucid, case[3], device=device)
    a.requires_grad_(True)
    _reduce(lucid, case, a).sum().backward()
    b = _data(ref, case[3])
    b.requires_grad_(True)
    _reduce(ref, case, b).sum().backward()
    assert a.grad.shape == tuple(b.grad.shape)
    assert _flat(a.grad.tolist()) == pytest.approx(_flat(b.grad.tolist()), rel=1e-4, abs=1e-5)


# ── compiled ─────────────────────────────────────────────────────────────────

needs_metal = pytest.mark.skipif(not metal_available(), reason="compile needs Metal")


def _vector() -> lucid.Tensor:
    return lucid.tensor([0.0, 1.0, 2.0, 3.0], device=COMPILE_DEVICE)


@needs_metal
@pytest.mark.parametrize("case", _cases(), ids=_ids)
def test_a_compiled_broadcast_of_the_reduction_is_eager(case: tuple[str, str, bool, int]) -> None:
    def fn(x: lucid.Tensor, a: lucid.Tensor) -> lucid.Tensor:
        return (x + _floats(_reduce(lucid, case, a))).sum(dim=0)

    x, a = _vector(), _data(lucid, case[3], device=COMPILE_DEVICE)
    compiled = lucid.compile(fn)
    compiled(x, a)  # the first call traces; the second runs the executable
    _close(compiled(x, a), fn(x, a), _ids(case))


@needs_metal
@pytest.mark.parametrize("join", ["stack", "cat"])
@pytest.mark.parametrize("case", [c for c in _cases() if not c[2]], ids=_ids)
def test_joining_reduced_scalars_is_eager(case: tuple[str, str, bool, int], join: str) -> None:
    def fn(x: lucid.Tensor, a: lucid.Tensor) -> lucid.Tensor:
        red = _floats(_reduce(lucid, case, a))
        if join == "stack":
            return lucid.stack([red, red * 2.0, red + x[0]])
        return lucid.cat([x + red, x], dim=0)

    x, a = _vector(), _data(lucid, case[3], device=COMPILE_DEVICE)
    compiled = lucid.compile(fn)
    compiled(x, a)
    _close(compiled(x, a), fn(x, a), f"{join} {_ids(case)}")


class _Reduced(nn.Module):
    def __init__(self, case: tuple[str, str, bool, int]) -> None:
        super().__init__()
        self._case = case
        self.w = nn.Parameter(_data(lucid, case[3]) * 1.0)

    def forward(self, scale: lucid.Tensor) -> lucid.Tensor:
        return _reduce(lucid, self._case, self.w * scale).sum()


@needs_metal
@pytest.mark.filterwarnings(
    "ignore:lucid.compile:lucid.compile._core.fallback.CompileFallbackWarning"
)
@pytest.mark.parametrize("case", _cases(differentiable_only=True), ids=_ids)
def test_a_compiled_step_gradient_is_eager(case: tuple[str, str, bool, int]) -> None:
    model = _Reduced(case).to(COMPILE_DEVICE)
    scale = lucid.tensor(1.5, device=COMPILE_DEVICE)
    model(scale).backward()
    want = model.w.grad.clone()
    model.w.grad = None
    step = lucid.compile.make_step(model, lambda out: out)
    step(scale).backward()
    assert not step.eager_only, f"{_ids(case)} fell back to eager"
    assert model.w.grad is not None
    _close(model.w.grad, want, f"gradient {_ids(case)}")
