"""A compiled group_norm of a rank-4+ input runs at rank 2 and keeps eager's values (LCD-920d).

The emitter and its VJP used to reshape ``(N, C, H, W)`` to ``(N, G, C/G, H, W)`` for the
group statistics.  MPSGraph runs an op on a rank-5+ tensor as a slice loop over the dims
above the last four, so ``N = 256`` took ~9.8 ms against ~2.7 ms for the same elements at
rank 2.  A concrete shape now works in ``(N*G, C/G * prod(spatial))`` and applies the
per-channel affine on ``(N, C, prod(spatial))``; a symbolic batch keeps the grouped form.
The graph cannot be inspected from Python, so the guard is

* a value table: forward output, loss and every gradient of a compiled step equal eager's
  for ranks 2..5, group counts from 1 (LayerNorm-like) to C (InstanceNorm-like), affine on
  and off, and float32 / float16 / bfloat16;
* NaN and inf rows: a poisoned group stays poisoned in the same cells as eager;
* a dynamic-batch row, where the grouped form must still hold;
* a coarse timing bound against the rank-2 twin the layout is meant to reach.
"""

import time
from collections.abc import Callable

import pytest

import lucid
import lucid.nn as nn
from lucid.compile import make_step
from lucid.test._fixtures.devices import metal_available
from lucid.test.unit.compile._helpers import COMPILE_DEVICE, assert_cache_hit, to_metal

pytestmark = [
    pytest.mark.skipif(not metal_available(), reason="compile needs Metal"),
    # A fallback to eager would make every compiled == eager row below vacuous.
    pytest.mark.filterwarnings("error::lucid.compile._core.fallback.CompileFallbackWarning"),
]

_C = 8

#: ``(shape, num_groups)``: rank 2..5 inputs (C = 8), G = 1, 2, 4, 8.
_CASES: list[tuple[tuple[int, ...], int]] = [
    ((3, _C), 2),
    ((3, _C, 5), 1),
    ((3, _C, 5), 4),
    ((3, _C, 4, 3), 1),
    ((3, _C, 4, 3), 2),
    ((3, _C, 4, 3), 4),
    ((3, _C, 4, 3), 8),
    ((2, _C, 3, 2, 3), 2),
    ((2, _C, 3, 2, 3), 8),
    ((1, _C, 4, 4), 4),
]


class _Net(nn.Module):
    """``GroupNorm(x * w)`` with a full-shape ``w``, so the gradient reaches the input."""

    def __init__(self, shape: tuple[int, ...], groups: int, affine: bool) -> None:
        super().__init__()
        n = 1
        for s in shape:
            n *= s
        self.w = nn.Parameter(lucid.linspace(0.6, 1.4, n).reshape(*shape))
        self.gn = nn.GroupNorm(groups, shape[1], affine=affine)
        if affine:
            assert self.gn.weight is not None and self.gn.bias is not None
            self.gn.weight.data = lucid.linspace(0.5, 1.5, shape[1])
            self.gn.bias.data = lucid.linspace(-0.3, 0.3, shape[1])

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.gn(x * self.w)


def _input(
    shape: tuple[int, ...], dtype: lucid.dtype, *, poison: float | None = None
) -> lucid.Tensor:
    n = 1
    for s in shape:
        n *= s
    x = (lucid.sin(lucid.arange(n, dtype=lucid.float32) * 1.3) * 2.0 + 0.4).reshape(*shape)
    if poison is not None:
        flat = x.reshape(-1)
        flat[n // 3] = poison
        x = flat.reshape(*shape)
    return x.to(dtype).to(COMPILE_DEVICE)


def _module(shape: tuple[int, ...], groups: int, affine: bool, dtype: lucid.dtype) -> nn.Module:
    return to_metal(_Net(shape, groups, affine)).to(dtype)


def _weights(n: int, dtype: lucid.dtype) -> lucid.Tensor:
    return lucid.linspace(0.5, 1.5, n, device=COMPILE_DEVICE, dtype=dtype)


def _loss(y: lucid.Tensor) -> lucid.Tensor:
    return (y * _weights(y.numel(), y.dtype).reshape(*y.shape)).sum()


def _same(got: lucid.Tensor, want: lucid.Tensor, why: str, tol: float) -> None:
    g, w = got.float().reshape(-1).tolist(), want.float().reshape(-1).tolist()
    assert len(g) == len(w), why
    for i, (a, b) in enumerate(zip(g, w, strict=True)):
        if b != b:  # NaN must stay NaN
            assert a != a, f"{why}[{i}]: {a} != nan"
        elif b in (float("inf"), float("-inf")):
            assert a == b, f"{why}[{i}]: {a} != {b}"
        else:
            assert a == pytest.approx(b, rel=tol, abs=tol), f"{why}[{i}]: {a} != {b}"


def _grads(m: nn.Module) -> dict[str, lucid.Tensor]:
    out = {}
    for name, p in m.named_parameters():
        assert p.grad is not None, name
        out[name] = p.grad
    return out


def _check(
    shape: tuple[int, ...],
    groups: int,
    affine: bool,
    dtype: lucid.dtype,
    tol: float,
    *,
    poison: float | None = None,
) -> None:
    x = _input(shape, dtype, poison=poison)
    eager = _module(shape, groups, affine, dtype)
    e_out = eager(x)
    e_loss = _loss(e_out)
    e_loss.backward()
    comp = _module(shape, groups, affine, dtype)
    step = make_step(comp, _loss)
    c_loss = step(x)
    c_loss.backward()
    _same(c_loss, e_loss, "loss", tol)
    e_g, c_g = _grads(eager), _grads(comp)
    assert e_g.keys() == c_g.keys()
    for name in e_g:
        _same(c_g[name], e_g[name], f"grad {name}", tol)
    # The compiled forward alone (no gradient) takes the same canonical path.
    with lucid.no_grad():
        _same(lucid.compile(comp.eval())(x), eager.eval()(x), "forward", tol)


def _ids(case: tuple[tuple[int, ...], int]) -> str:
    shape, g = case
    return "x".join(map(str, shape)) + f"-g{g}"


@pytest.mark.parametrize("affine", [True, False], ids=["affine", "plain"])
@pytest.mark.parametrize("case", _CASES, ids=_ids)
def test_a_compiled_group_norm_matches_eager(
    case: tuple[tuple[int, ...], int], affine: bool
) -> None:
    shape, groups = case
    _check(shape, groups, affine, lucid.float32, 2e-4)


@pytest.mark.parametrize(
    "dtype,tol", [(lucid.float16, 4e-2), (lucid.bfloat16, 1e-1)], ids=["float16", "bfloat16"]
)
@pytest.mark.parametrize("case", _CASES[3:7], ids=_ids)
def test_a_half_precision_group_norm_matches_eager(
    case: tuple[tuple[int, ...], int], dtype: lucid.dtype, tol: float
) -> None:
    shape, groups = case
    _check(shape, groups, True, dtype, tol)


@pytest.mark.parametrize("poison", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("case", [_CASES[5], _CASES[7]], ids=_ids)
def test_a_poisoned_group_stays_poisoned_like_eager(
    case: tuple[tuple[int, ...], int], poison: float
) -> None:
    shape, groups = case
    _check(shape, groups, True, lucid.float32, 2e-4, poison=poison)


class _Plain(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gn = nn.GroupNorm(4, _C)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.gn(x) * 1.5


@pytest.mark.usefixtures("inference")
def test_a_symbolic_batch_keeps_the_grouped_form() -> None:
    m = to_metal(_Plain()).eval()
    cm = lucid.compile(m, dynamic=True)
    for batch in (2, 5, 3):
        x = _input((batch, _C, 4, 3), lucid.float32)
        want = m(x)
        got = cm(x)
        assert got.shape == want.shape
        _same(got, want, f"batch {batch}", 2e-4)
    assert_cache_hit(cm, 1)


def _step_ms(
    model: nn.Module, x: lucid.Tensor, fn: Callable[[lucid.Tensor], lucid.Tensor]
) -> float:
    step = make_step(model, fn)
    for _ in range(2):
        step(x).backward()
    best = float("inf")
    for _ in range(5):
        t0 = time.perf_counter()
        loss = step(x)
        loss.backward()
        float(loss.item())
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


class _Twin(nn.Module):
    """Layer-norm over the same ``(N*G, C/G * H*W)`` rows a group-norm normalises."""

    def __init__(self, shape: tuple[int, ...], groups: int) -> None:
        super().__init__()
        n, c, h, w = shape
        self.rows = (n * groups, c // groups * h * w)
        self.w = nn.Parameter(lucid.ones(*shape))

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        y = (x * self.w).reshape(*self.rows)
        mu = y.mean(dim=1, keepdim=True)
        v = y.var(dim=1, keepdim=True)
        return ((y - mu) * (v + 1e-5).rsqrt()).reshape(*x.shape)


def test_a_group_norm_is_not_slower_than_its_rank_2_twin() -> None:
    """Coarse on purpose: the grouped rank-5 form ran ~6x the twin at N = 256, the fix ~1.3x."""
    shape = (256, 64, 16, 16)
    x = _input(shape, lucid.float32)
    gn = _step_ms(_module(shape, 8, True, lucid.float32), x, _loss)
    twin = _step_ms(to_metal(_Twin(shape, 8)), x, _loss)
    assert gn < 2.0 * twin + 1.0, f"group_norm {gn:.1f} ms vs rank-2 twin {twin:.1f} ms"
