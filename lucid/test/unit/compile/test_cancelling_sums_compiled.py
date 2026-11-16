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

import platform
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
    "C-(C-a)": (lambda a: 1.0 - (1.0 - a), 1),
    "C-(C-a)-large": (lambda a: 1e6 - (1e6 - a), 1),
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


def _eager_and_compiled(fn: _Fn, args: list[lucid.Tensor]) -> tuple[lucid.Tensor, lucid.Tensor]:
    with lucid.no_grad():
        want = fn(*args)
    return want, _compiled(fn, args)


def _is_eager(fn: _Fn, args: list[lucid.Tensor]) -> bool:
    want, got = _eager_and_compiled(fn, args)
    return _differing(got.to(lucid.float32), want.to(lucid.float32)) == 0


# What the guard does not reach.  Each is a strict xfail: it passes the day the
# mechanism is closed, and the XPASS says so.  The identity-based cases depend on
# MPSGraph's simplifier, which folds them from macOS 27 on (the hosted runner's
# macOS 26 does not, so they pass there): strict only where the fold happens.
_SIMPLIFIER_FOLDS = int((platform.mac_ver()[0] or "0").split(".")[0]) >= 27


@pytest.mark.xfail(
    strict=_SIMPLIFIER_FOLDS,
    reason="LCD-1033 follow-up: operands are matched by graph-tensor identity, so a term that is "
    "recomputed (two `a * 4097` nodes) rather than reused is not seen to cancel before MPSGraph "
    "merges them",
)
def test_a_term_recomputed_rather_than_reused_is_not_yet_seen(device_gpu_only: str) -> None:
    def fn(a: lucid.Tensor) -> lucid.Tensor:
        return (a * 4097.0) - ((a * 4097.0) - a)

    assert _is_eager(fn, _operands(device_gpu_only, 1))


@pytest.mark.xfail(
    strict=_SIMPLIFIER_FOLDS,
    reason="LCD-1033 follow-up: a sum computed twice (two add nodes) is two tensors to the guard "
    "and one after MPSGraph's CSE, so `((a + b) + c) - (a + b)` is cancelled",
)
def test_a_sum_recomputed_rather_than_reused_is_not_yet_seen(device_gpu_only: str) -> None:
    def fn(a: lucid.Tensor, b: lucid.Tensor, c: lucid.Tensor) -> lucid.Tensor:
        return ((a + b) + c) - (a + b)

    assert _is_eager(fn, _operands(device_gpu_only, 3))


@pytest.mark.xfail(
    strict=_SIMPLIFIER_FOLDS,
    reason="LCD-1033 follow-up: a moves-only op (reshape, detach, a full slice) between the sum "
    "and the subtraction binds a new tensor, so the operands stop matching by identity while "
    "MPSGraph still sees through the view",
)
@pytest.mark.parametrize(
    "view",
    [
        pytest.param(lambda t: t.reshape(64, 64).reshape(-1), id="reshape-round-trip"),
        pytest.param(lambda t: t[:], id="full-slice"),
        pytest.param(lambda t: t.detach(), id="detach"),
    ],
)
def test_a_sum_seen_through_a_view_is_not_yet_guarded(
    view: Callable[[lucid.Tensor], lucid.Tensor], device_gpu_only: str
) -> None:
    def fn(a: lucid.Tensor, b: lucid.Tensor) -> lucid.Tensor:
        return view(a + b) - a

    assert _is_eager(fn, _operands(device_gpu_only, 2))


@pytest.mark.xfail(
    strict=_SIMPLIFIER_FOLDS,
    reason="LCD-1033 follow-up: the commit is emitted for float16 / bfloat16 but Lucid's half "
    "pipeline keeps a fused elementwise chain's intermediates at float32 and rounds once, so "
    "`(a + 1) - 1` compiled is `a` where eager rounds the sum to half",
)
@pytest.mark.parametrize("dtype", [lucid.float16, lucid.bfloat16], ids=["float16", "bfloat16"])
def test_a_half_precision_sum_is_not_yet_rounded_before_the_subtraction(
    dtype: lucid.dtype, device_gpu_only: str
) -> None:
    def fn(a: lucid.Tensor) -> lucid.Tensor:
        return (a + 1.0) - 1.0

    (a,) = _operands(device_gpu_only, 1)
    assert _is_eager(fn, [(a * 1e-3).to(dtype)])


@pytest.mark.xfail(
    strict=_SIMPLIFIER_FOLDS,
    reason="LCD-1033 follow-up: is_floating_point excludes complex64, so an add / sub of complex "
    "values is neither recorded nor committed and `(z + w) - z` is cancelled",
)
def test_a_complex_sum_is_not_yet_guarded(device_gpu_only: str) -> None:
    def fn(z: lucid.Tensor, w: lucid.Tensor) -> lucid.Tensor:
        return ((z + w) - z).real

    a, b, c = _operands(device_gpu_only, 3)
    assert _is_eager(fn, [lucid.complex(a, b), lucid.complex(b, c)])
