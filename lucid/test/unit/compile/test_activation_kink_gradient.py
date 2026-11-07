"""A compiled activation's gradient takes eager's side at the kinks (LCD-911).

The compiled VJPs of the piecewise activations tested ``x >= 0`` (``leaky_relu``,
``elu``, ``selu``) or ``x >= -3`` (``hardswish``) where eager's backward tests
``x > 0`` and ``x <= -3``: at exactly 0 the compiled gradient took the positive
slope (1 where eager gives the negative slope, ``alpha`` or ``lambda * alpha``),
and ``hardswish`` at ``-3`` / ``3`` gave the ramp's slope (-0.5 / 1.5) where
eager's outer pieces give 0 / 1.  ``hardswish`` also blended its three pieces by
0 / 1 masks, so a ``+-inf`` input met ``0 * inf`` and came out NaN where eager
gives 0 / 1.

The table runs every piecewise activation through its kinks, a point either side
of each, NaN and ``+-inf`` — the ones eager shares (``hardsigmoid``, ``relu6``,
``hardtanh``, ``threshold``, ``hardshrink`` / ``softshrink``, ``clamp``, ``celu``,
``relu``, ``abs``) stand as controls — through ``make_step``, ``fused_step`` and
``compiled_step``, in float32, float16 and bfloat16.  The probe points are all
exactly representable in the half formats, so a kink is hit exactly in each.
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
_LR = 1.0
_TOL = {lucid.float32: 1e-5, lucid.float16: 4e-3, lucid.bfloat16: 3e-2}
_DTYPE_IDS = {lucid.float32: "f32", lucid.float16: "f16", lucid.bfloat16: "bf16"}
_ENTRIES = ("make_step", "fused_step", "compiled_step")
_TAIL = [_NAN, _INF, -_INF]


@dataclass(frozen=True)
class _Case:
    op: Callable[[lucid.Tensor], lucid.Tensor]
    points: list[float]


def _around(kink: float, step: float = 0.25) -> list[float]:
    return [kink - step, kink, kink + step]


_CASES: dict[str, _Case] = {
    "relu": _Case(F.relu, _around(0.0) + _TAIL),
    "leaky_relu": _Case(lambda a: F.leaky_relu(a, 0.125), _around(0.0) + _TAIL),
    "leaky_relu-default": _Case(F.leaky_relu, _around(0.0) + _TAIL),
    "elu": _Case(lambda a: F.elu(a, 0.5), _around(0.0) + _TAIL),
    "elu-default": _Case(F.elu, _around(0.0) + _TAIL),
    "celu": _Case(lambda a: F.celu(a, 0.5), _around(0.0) + _TAIL),
    "selu": _Case(F.selu, _around(0.0) + _TAIL),
    "hardswish": _Case(F.hardswish, _around(-3.0) + _around(3.0) + _TAIL),
    "hardsigmoid": _Case(F.hardsigmoid, _around(-3.0) + _around(3.0) + _TAIL),
    "relu6": _Case(F.relu6, _around(0.0) + _around(6.0) + _TAIL),
    "hardtanh": _Case(F.hardtanh, _around(-1.0) + _around(1.0) + _TAIL),
    "hardtanh-asymmetric": _Case(
        lambda a: F.hardtanh(a, -2.0, 0.5), _around(-2.0) + _around(0.5) + _TAIL
    ),
    "threshold": _Case(lambda a: F.threshold(a, 0.5, 2.0), _around(0.5) + _TAIL),
    "hardshrink": _Case(lambda a: F.hardshrink(a, 0.5), _around(-0.5) + _around(0.5) + _TAIL),
    "softshrink": _Case(lambda a: F.softshrink(a, 0.5), _around(-0.5) + _around(0.5) + _TAIL),
    "clamp": _Case(lambda a: lucid.clamp(a, -1.0, 1.0), _around(-1.0) + _around(1.0) + _TAIL),
    "clamp-min": _Case(lambda a: lucid.clamp(a, min=0.5), _around(0.5) + _TAIL),
    "clamp-max": _Case(lambda a: lucid.clamp(a, max=0.5), _around(0.5) + _TAIL),
    "abs": _Case(lucid.abs, _around(0.0) + _TAIL),
}


class _OfParam(nn.Module):
    def __init__(self, case: _Case, dtype: lucid.dtype, device: str) -> None:
        super().__init__()
        self.a = nn.Parameter(lucid.tensor(case.points, dtype=dtype, device=device))
        self.op = case.op

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.op(self.a * x).sum().reshape(1)


def _loss(out: lucid.Tensor, target: lucid.Tensor) -> lucid.Tensor:
    return (out * target).sum()


def _grad_of(model: _OfParam) -> list[float]:
    grad = model.a.grad
    assert grad is not None
    return grad.detach().cpu().to(lucid.float32).tolist()


def _param_of(model: _OfParam) -> list[float]:
    return model.a.detach().cpu().to(lucid.float32).tolist()


def _eager(model: _OfParam, x: lucid.Tensor, t: lucid.Tensor) -> tuple[list[float], list[float]]:
    opt = optim.SGD(model.parameters(), lr=_LR)
    _loss(model(x), t).backward()
    grad = _grad_of(model)
    opt.step()
    return grad, _param_of(model)


def _compiled(
    entry: str, model: _OfParam, x: lucid.Tensor, t: lucid.Tensor
) -> tuple[list[float] | None, list[float]]:
    opt = optim.SGD(model.parameters(), lr=_LR)
    if entry == "fused_step":
        fused = fused_step(model, _loss, opt)
        assert isinstance(fused, _FusedStep)
        fused(x, t)
        assert not fused.cache_info()["eager_only"]
        return None, _param_of(model)
    if entry == "make_step":
        step = make_step(model, _loss)
        step(x, t).backward()
        assert not step.eager_only
    else:
        compiled_step(model, x, lambda y: _loss(y, t))
    grad = _grad_of(model)
    opt.step()
    return grad, _param_of(model)


def _assert_close(
    got: list[float], want: list[float], tol: float, points: list[float], why: str
) -> None:
    assert len(got) == len(want)
    for p, q, at in zip(got, want, points, strict=True):
        same = math.isnan(p) if math.isnan(q) else math.isclose(p, q, rel_tol=tol, abs_tol=tol)
        assert same, f"{why} at x={at}: compiled {got} != eager {want}"


def _cases() -> list[object]:
    return [
        pytest.param(name, entry, dtype, id=f"{name}-{entry}-{dt}")
        for name in _CASES
        for entry in _ENTRIES
        for dtype, dt in _DTYPE_IDS.items()
    ]


@pytest.mark.parametrize(("name", "entry", "dtype"), _cases())
def test_compiled_activation_gradient_matches_eager_at_the_kinks(
    name: str, entry: str, dtype: lucid.dtype, device_gpu_only: str
) -> None:
    case = _CASES[name]
    x = lucid.ones(1, dtype=dtype, device=device_gpu_only)
    t = lucid.ones(1, dtype=dtype, device=device_gpu_only)
    want_grad, want_param = _eager(_OfParam(case, dtype, device_gpu_only), x, t)
    model = _OfParam(case, dtype, device_gpu_only)
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompileFallbackWarning)
        got_grad, got_param = _compiled(entry, model, x, t)
    tol = _TOL[dtype]
    if got_grad is not None:
        _assert_close(got_grad, want_grad, tol, case.points, "gradient")
    _assert_close(got_param, want_param, tol, case.points, "parameter after SGD")


def test_the_defects_own_numbers(device_gpu_only: str) -> None:
    """The card's repro, independent of eager: the kink takes the outer / negative piece."""
    one = lucid.ones(1, device=device_gpu_only)
    leaky = _OfParam(
        _Case(lambda a: F.leaky_relu(a, 0.1), [0.0, 3.0, -3.0]), lucid.float32, device_gpu_only
    )
    make_step(leaky, _loss)(one, one).backward()
    _assert_close(_grad_of(leaky), [0.1, 1.0, 0.1], 1e-6, [0.0, 3.0, -3.0], "leaky_relu")
    swish = _OfParam(_Case(F.hardswish, [0.0, 3.0, -3.0]), lucid.float32, device_gpu_only)
    make_step(swish, _loss)(one, one).backward()
    _assert_close(_grad_of(swish), [0.5, 1.0, 0.0], 1e-6, [0.0, 3.0, -3.0], "hardswish")
