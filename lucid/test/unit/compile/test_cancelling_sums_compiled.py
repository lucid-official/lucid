"""A compiled graph rounds a sum before it subtracts a term of it, as eager does (LCD-1033).

MPSGraph's simplifier cancels ``(a + b) - a`` to ``b``, ``a - (a - b)`` to ``b``,
``(a - b) + b`` to ``a`` and ``(x + C) - C`` to ``x``.  In float32 the left side is
the rounding error of the sum, which is what error-compensating code is after, and
the right side is not: Kahan's compensation, Fast2Sum / TwoSum, the
``(x + 1.5 * 2**23) - 1.5 * 2**23`` round-to-integer trick, and the ``log1p`` that
eager builds from ``(1 + x) - 1`` all lost it under ``lucid.compile``.

The add / sub emitters commit the rounding of a sum that a later add / sub would
cancel against (``_CancellingSums.h``).  Each pattern here is compiled and compared
with eager bit for bit over 4096 float32 pairs spread across eight decades; before
the fix 1400 .. 4096 of them differed, by up to the whole error term.
"""

import warnings
from collections.abc import Callable

import pytest

import lucid
import lucid.nn as nn
from lucid.compile import make_step
from lucid.compile._core.fallback import CompileFallbackWarning

_N = 4096

_Fn = Callable[..., lucid.Tensor]


def _kahan_step(s: lucid.Tensor, y: lucid.Tensor) -> lucid.Tensor:
    t = s + y
    return (t - s) - y


def _two_sum_error(a: lucid.Tensor, b: lucid.Tensor) -> lucid.Tensor:
    s = a + b
    bb = s - a
    return (a - (s - bb)) + (b - bb)


def _fast_two_sum_error(a: lucid.Tensor, b: lucid.Tensor) -> lucid.Tensor:
    s = a + b
    return b - (s - a)


def _split(a: lucid.Tensor) -> lucid.Tensor:
    c = a * 4097.0
    return a - (c - (c - a))


_PATTERNS: dict[str, tuple[_Fn, int]] = {
    "(a+b)-a": (lambda a, b: (a + b) - a, 2),
    "(a+b)-b": (lambda a, b: (a + b) - b, 2),
    "(b+a)-a": (lambda a, b: (b + a) - a, 2),
    "a-(a-b)": (lambda a, b: a - (a - b), 2),
    "(a-b)+b": (lambda a, b: (a - b) + b, 2),
    "b+(a-b)": (lambda a, b: b + (a - b), 2),
    "(a+1)-1": (lambda a: (a + 1.0) - 1.0, 1),
    "(a+C)-C": (lambda a: (a + 3.5) - 3.5, 1),
    "(a-C)+C": (lambda a: (a - 1e6) + 1e6, 1),
    "(a+C)+(-C)": (lambda a: (a + 1e6) + (-1e6), 1),
    "(a+C)-C-shifted": (lambda a: (a + 1e6) - 2e6, 1),
    "(a+1)+2": (lambda a: (a + 1.0) + 2.0, 1),
    "round-trick": (lambda a: (a + 12582912.0) - 12582912.0, 1),
    "kahan-step": (_kahan_step, 2),
    "two-sum-error": (_two_sum_error, 2),
    "fast-two-sum-error": (_fast_two_sum_error, 2),
    "dekker-split": (_split, 1),
    "kahan-chain": (lambda a, b, c: _kahan_step(a + b, c), 3),
}


def _operands(device: str, count: int) -> list[lucid.Tensor]:
    lucid.manual_seed(0)
    decade = 10.0 ** (lucid.rand(_N, device=device) * 8.0 - 4.0)
    return [lucid.randn(_N, device=device) * decade for _ in range(count)]


def _compiled(fn: _Fn, args: list[lucid.Tensor]) -> lucid.Tensor:
    compiled = lucid.compile(fn)
    with lucid.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("error", CompileFallbackWarning)
        compiled(*args)
        got = compiled(*args)
    info = compiled.cache_info()
    assert info["entries"] == 1 and not info["eager_only"], info
    return got


def _differing(got: lucid.Tensor, want: lucid.Tensor) -> int:
    return sum(1 for g, w in zip(got.tolist(), want.tolist(), strict=True) if g != w)


@pytest.mark.parametrize("name", list(_PATTERNS))
def test_a_compiled_sum_keeps_its_rounding_for_the_term_taken_off(
    name: str, device_gpu_only: str
) -> None:
    fn, arity = _PATTERNS[name]
    args = _operands(device_gpu_only, arity)
    with lucid.no_grad():
        want = fn(*args)
    got = _compiled(fn, args)
    assert _differing(got, want) == 0, f"{name}: compiled differs from eager"


def test_the_gradient_through_a_cancelling_pair_is_eagers(device_gpu_only: str) -> None:
    """``d/da (a + b) - a`` is 0 either way; the committed sum must not change a gradient."""
    class _Pair(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = nn.Parameter(lucid.tensor([1e-8, 0.5, -3.0, 1e6], device=device_gpu_only))

        def forward(self, b: lucid.Tensor) -> lucid.Tensor:
            return (((self.a + b) - b) * (self.a - (self.a - b)) * 3.0).sum().reshape(1)

    def loss(out: lucid.Tensor, target: lucid.Tensor) -> lucid.Tensor:
        return (out * target).sum()

    b = lucid.tensor([1.0, 2.0, 0.25, 7.0], device=device_gpu_only)
    target = lucid.ones(1, device=device_gpu_only)
    eager = _Pair()
    loss(eager(b), target).backward()
    model = _Pair()
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompileFallbackWarning)
        step = make_step(model, loss)
        step(b, target).backward()
    assert not step.eager_only
    assert model.a.grad is not None and eager.a.grad is not None
    assert model.a.grad.tolist() == eager.a.grad.tolist()


@pytest.mark.xfail(
    strict=True,
    reason="LCD-1033: operands are matched by graph-tensor identity, so a term that is recomputed "
    "(two `a * 4097` nodes) rather than reused is not seen to cancel before MPSGraph merges them",
)
def test_a_term_recomputed_rather_than_reused_is_not_yet_seen(device_gpu_only: str) -> None:
    def fn(a: lucid.Tensor) -> lucid.Tensor:
        return (a * 4097.0) - ((a * 4097.0) - a)

    args = _operands(device_gpu_only, 1)
    with lucid.no_grad():
        want = fn(*args)
    assert _differing(_compiled(fn, args), want) == 0
