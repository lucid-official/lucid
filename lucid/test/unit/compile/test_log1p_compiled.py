"""A compiled ``log1p`` is eager's, and so is everything built on it (LCD-1033).

Eager's ``log1p`` takes the log at ``u = 1 + x`` and rescales by the increment the
addition applied, ``u - 1``.  The recording holds exactly that, and MPSGraph's
simplifier cancelled ``(1 + x) - 1`` to ``x``: below 6e-8 the answer came back 0
where eager gives ``x`` (``log1p(1e-8)`` was 0, ``log1p(1e-5)`` 1.00135e-5 against
9.99995e-6).  ``asinh`` / ``atanh`` / ``acosh`` run on ``log1p``, so a compiled
``asinh(1e-8)`` was 0 too, and ``acosh`` next to 1 was 7e-5 off.  The gradient of
``copysign(log1p(abs(t)), t)`` was 0 for the same reason: ``abs`` of a value that
should have been ``t`` and was 0 has slope 0.

The emitter of the sum now commits its rounding before the subtraction reads it
(``_CancellingSums.h``); the class-wide guard is ``test_cancelling_sums_compiled``.
"""

import math
import warnings
from collections.abc import Callable

import pytest

import lucid
import lucid.nn as nn
from lucid.compile import make_step
from lucid.compile._core.fallback import CompileFallbackWarning

#: A few float32 ulps: the compiled ``exp`` / ``log`` are MPSGraph's, eager's are MLX's.
_REL = 4e-7


def _decades(lo: int, hi: int, per_decade: int = 2) -> list[float]:
    return [10.0 ** (e / per_decade) for e in range(lo * per_decade, hi * per_decade + 1)]


def _signed(points: list[float]) -> list[float]:
    return points + [-p for p in points]


_ABOVE_MINUS_ONE = [p for p in _signed(_decades(-30, 30)) if p > -1.0]
_UNIT = [p for p in _signed(_decades(-30, -1)) if abs(p) < 1.0]
# 1 + 2^-k and 1 + 10^-k, then well above: ``acosh`` next to its branch point, and far from it.
_FROM_ONE = [1.0 + 2.0**-k for k in range(1, 24)] + [1.0 + 1e-3, 1.5, 2.0] + _decades(1, 30)


def _compiled_value(fn: Callable[[lucid.Tensor], lucid.Tensor], x: lucid.Tensor) -> list[float]:
    compiled = lucid.compile(fn)
    with lucid.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("error", CompileFallbackWarning)
        compiled(x)
        got = compiled(x)
    info = compiled.cache_info()
    assert info["entries"] == 1 and not info["eager_only"], info
    return got.tolist()


def _assert_close(got: list[float], want: list[float], points: list[float], why: str) -> None:
    assert len(got) == len(want) == len(points)
    for g, w, at in zip(got, want, points, strict=True):
        same = (math.isnan(g) and math.isnan(w)) or g == w or abs(g - w) <= _REL * abs(w)
        assert same, f"{why} at x={at!r}: compiled {g!r} != eager {w!r}"


def test_the_defects_own_numbers(device_gpu_only: str) -> None:
    x = lucid.tensor([1e-8, 1e-5], device=device_gpu_only)
    got = _compiled_value(lucid.log1p, x)
    assert got[0] == pytest.approx(1e-8, rel=1e-6, abs=0)
    assert got[1] == pytest.approx(9.99995e-6, rel=1e-6, abs=0)
    assert got == lucid.log1p(x).tolist()


_VALUE_OPS: dict[str, tuple[Callable[[lucid.Tensor], lucid.Tensor], list[float]]] = {
    "log1p": (lucid.log1p, _ABOVE_MINUS_ONE),
    "asinh": (lucid.asinh, _signed(_decades(-30, 30))),
    "atanh": (lucid.atanh, _UNIT),
    "acosh": (lucid.acosh, _FROM_ONE),
    "expm1": (lucid.expm1, _signed(_decades(-30, 1))),
    "logaddexp": (lambda t: lucid.logaddexp(t, t * 1e-8), _signed(_decades(-30, 30))),
}


@pytest.mark.parametrize("name", list(_VALUE_OPS))
def test_compiled_value_matches_eager_over_the_float_range(name: str, device_gpu_only: str) -> None:
    fn, points = _VALUE_OPS[name]
    x = lucid.tensor(points, device=device_gpu_only)
    got = _compiled_value(fn, x)
    _assert_close(got, fn(x).tolist(), points, name)


def test_compiled_log1p_matches_the_float64_answer(device_gpu_only: str) -> None:
    points = [p for p in _ABOVE_MINUS_ONE if p > -0.999]
    x = lucid.tensor(points, device=device_gpu_only)
    got = _compiled_value(lucid.log1p, x)
    # ``lucid.tensor`` rounds to float32: the answer is for the stored value.
    stored = x.tolist()
    for g, s in zip(got, stored, strict=True):
        assert g == pytest.approx(math.log1p(s), rel=_REL, abs=0), f"x={s!r}"


@pytest.mark.parametrize("x", [1e-8, 1e-5, 3e-8, 1e-30, -1e-8])
def test_the_small_arguments_the_inverse_hyperbolics_lose(x: float, device_gpu_only: str) -> None:
    """``asinh`` / ``atanh`` at 1e-8 are 1e-8, as eager's, not 0."""
    t = lucid.tensor([x], device=device_gpu_only)
    for fn in (lucid.asinh, lucid.atanh):
        got = _compiled_value(fn, t)
        assert got == fn(t).tolist()
        assert got[0] == pytest.approx(x, rel=1e-6, abs=0)


class _OfParam(nn.Module):
    def __init__(
        self, points: list[float], fn: Callable[[lucid.Tensor], lucid.Tensor], device: str
    ) -> None:
        super().__init__()
        self.a = nn.Parameter(lucid.tensor(points, device=device))
        self.fn = fn

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fn(self.a * x).sum().reshape(1)


def _loss(out: lucid.Tensor, target: lucid.Tensor) -> lucid.Tensor:
    return (out * target).sum()


def _grad(model: _OfParam) -> list[float]:
    grad = model.a.grad
    assert grad is not None
    return grad.detach().cpu().tolist()


_GRAD_OPS: dict[str, tuple[Callable[[lucid.Tensor], lucid.Tensor], list[float]]] = {
    "log1p": (lucid.log1p, _ABOVE_MINUS_ONE),
    "asinh": (lucid.asinh, _signed(_decades(-30, 30))),
    "atanh": (lucid.atanh, _UNIT),
    "acosh": (lucid.acosh, _FROM_ONE),
    "copysign-log1p-abs": (
        lambda t: lucid.copysign(lucid.log1p(t.abs()), t),
        _signed(_decades(-30, 30)),
    ),
    "logaddexp": (lambda t: lucid.logaddexp(t, t * 1e-8), _signed(_decades(-30, 30))),
    "expm1": (lucid.expm1, _signed(_decades(-30, 1))),
}


@pytest.mark.parametrize("scale", [1.0, 1.5, 3.0])
@pytest.mark.parametrize("name", list(_GRAD_OPS))
def test_compiled_gradient_matches_eager_over_the_float_range(
    name: str, scale: float, device_gpu_only: str
) -> None:
    fn, points = _GRAD_OPS[name]
    x = lucid.tensor([scale], device=device_gpu_only)
    target = lucid.ones(1, device=device_gpu_only)
    eager = _OfParam(points, fn, device_gpu_only)
    _loss(eager(x), target).backward()
    model = _OfParam(points, fn, device_gpu_only)
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompileFallbackWarning)
        step = make_step(model, _loss)
        step(x, target).backward()
    assert not step.eager_only
    _assert_close(_grad(model), _grad(eager), points, f"{name} gradient")


def test_the_copysign_gradient_the_card_names(device_gpu_only: str) -> None:
    """``abs`` of a ``log1p`` that came back 0 has slope 0; with the value right it is the slope of ``log1p``."""
    points = [1e-8, 1e-5, 1e-30, 3e-8, -1e-8]
    x = lucid.tensor([1.5], device=device_gpu_only)
    target = lucid.ones(1, device=device_gpu_only)
    model = _OfParam(points, lambda t: lucid.copysign(lucid.log1p(t.abs()), t), device_gpu_only)
    make_step(model, _loss)(x, target).backward()
    for got, a in zip(_grad(model), points, strict=True):
        assert got == pytest.approx(1.5 / (1.0 + 1.5 * abs(a)), rel=1e-6, abs=0)
