"""A compiled reduction of a rank-5+ tensor runs at lower rank and keeps eager's values (LCD-918).

MPSGraph runs an op on a rank-5+ tensor as a loop over the dims above the last four, one
dispatch per slice (~5 us each): ``x * w`` on a ``(6144, 16, 2, 16, 2)`` input took 28 ms
against 3 ms for ``(96, 64, 16, 2, 32)``.  A compiled ``reshape(6144, 16, 2, 16, 2).sum(2)``
was ~27x slower than eager.  The reduction emitters now fuse adjacent kept axes, adjacent
reduced axes and size-1 axes before emitting, so the graph holds no rank-5+ tensor from a
reduction.  The graph cannot be inspected from Python, so the guard is

* a value table: loss and parameter gradient of a compiled step equal eager's for every
  reduction kind across ranks 2..6 and axis patterns (adjacent, split, alternating, with
  a size-1 axis, full), with NaN rows for the extremum kinds (LCD-873 / LCD-878);
* a dynamic-batch row, where the fused leading block must stay symbolic;
* a coarse timing bound: the rank-5 reduction may not run more than a few times slower
  than the same elements reduced as a rank-3 view, both compiled (the bug was 30x).
"""

from collections.abc import Callable

import pytest

import lucid
import lucid.nn as nn
from lucid.compile import make_step
from lucid.test._fixtures.devices import metal_available
from lucid.test.unit.compile._helpers import (
    COMPILE_DEVICE,
    assert_cache_hit,
    assert_ratio_within,
    step_call,
    to_metal,
)

pytestmark = [
    pytest.mark.skipif(not metal_available(), reason="compile needs Metal"),
    # A fallback to eager would make every compiled == eager row below vacuous.
    pytest.mark.filterwarnings("error::lucid.compile._core.fallback.CompileFallbackWarning"),
]

#: ``(shape, dims)``: ranks 2..6, adjacent / split / alternating / size-1 / full patterns.
_CASES: list[tuple[tuple[int, ...], tuple[int, ...]]] = [
    ((3, 4), (1,)),
    ((3, 4, 2), (0, 2)),
    ((3, 4, 2, 3), (1, 3)),
    ((2, 3, 2, 3, 2), (2,)),
    ((2, 3, 2, 3, 2), (0, 1)),
    ((2, 3, 2, 3, 2), (1, 3)),
    ((2, 3, 2, 3, 2), (0, 2, 4)),
    ((2, 3, 2, 3, 2), (1, 2, 3, 4)),
    ((2, 3, 2, 3, 2), (0, 1, 2, 3, 4)),
    ((2, 1, 3, 2, 3), (1, 3)),
    ((3, 1, 2, 1, 3), (2,)),
    ((2, 3, 2, 3, 2, 2), (3, 5)),
    ((2, 3, 2, 3, 2, 2), (0, 2, 4)),
    ((2, 3, 2, 3, 2, 2), (1, 2)),
    ((2, 3, 2, 3, 2, 2), (4, 5)),
    ((2, 3, 2, 3, 2, 2), (2,)),
    ((2, 3, 2, 3, 2, 2), (0, 1, 2, 3, 4, 5)),
    # Fused layout still rank >= 5 (LCD-921): reduced one block at a time.
    ((2, 3, 2, 3, 2, 2), (1, 3, 5)),
    ((2, 3, 2, 3, 2, 2), (0, 2, 4)),
    ((2, 2, 3, 2, 3, 2), (0, 2, 4)),
    ((2, 2, 3, 2, 3, 2), (1, 3, 5)),
    ((2, 2, 2, 3, 2, 2, 3), (0, 2, 4, 6)),
    ((2, 2, 2, 3, 2, 2, 3), (1, 3, 5)),
]

_KINDS: dict[str, Callable[[lucid.Tensor, tuple[int, ...], bool], lucid.Tensor]] = {
    "sum": lambda t, d, k: t.sum(dim=d, keepdim=k),
    "mean": lambda t, d, k: t.mean(dim=d, keepdim=k),
    "amax": lambda t, d, k: t.amax(dim=d, keepdim=k),
    "amin": lambda t, d, k: t.amin(dim=d, keepdim=k),
    "prod": lambda t, d, k: t.prod(dim=d[0], keepdim=k),
    "var": lambda t, d, k: t.var(dim=d, keepdim=k),
    "std": lambda t, d, k: t.std(dim=d, keepdim=k),
    "logsumexp": lambda t, d, k: t.logsumexp(dim=d, keepdim=k),
}


class _Scaled(nn.Module):
    """``x * w`` with a full-shape parameter, so the gradient is per element."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        super().__init__()
        n = 1
        for s in shape:
            n *= s
        self.w = nn.Parameter(lucid.linspace(0.6, 1.4, n).reshape(*shape).to(COMPILE_DEVICE))

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return x * self.w


def _input(shape: tuple[int, ...], *, nan_at: int | None = None) -> lucid.Tensor:
    n = 1
    for s in shape:
        n *= s
    # A non-monotone pattern so every extremum has a unique position and ties are rare.
    x = (lucid.sin(lucid.arange(n, dtype=lucid.float32) * 1.7) + 1.5).reshape(*shape)
    if nan_at is not None:
        flat = x.reshape(-1)
        flat[nan_at] = float("nan")
        x = flat.reshape(*shape)
    return x.to(COMPILE_DEVICE)


def _loss_fn(
    kind: str, dims: tuple[int, ...], keepdim: bool
) -> Callable[[lucid.Tensor], lucid.Tensor]:
    def fn(y: lucid.Tensor) -> lucid.Tensor:
        r = _KINDS[kind](y, dims, keepdim)
        weights = lucid.linspace(0.5, 1.5, r.numel(), device=COMPILE_DEVICE).reshape(*r.shape)
        return (r * weights).sum()

    return fn


def _eager_and_compiled(
    shape: tuple[int, ...], loss_fn: Callable[[lucid.Tensor], lucid.Tensor], x: lucid.Tensor
) -> tuple[lucid.Tensor, lucid.Tensor, lucid.Tensor, lucid.Tensor]:
    eager_m = to_metal(_Scaled(shape))
    eager_loss = loss_fn(eager_m(x))
    eager_loss.backward()
    comp_m = to_metal(_Scaled(shape))
    step = make_step(comp_m, loss_fn)
    comp_loss = step(x)
    comp_loss.backward()
    assert eager_m.w.grad is not None and comp_m.w.grad is not None
    return eager_loss, comp_loss, eager_m.w.grad, comp_m.w.grad


def _same(got: lucid.Tensor, want: lucid.Tensor, why: str) -> None:
    g, w = got.reshape(-1).tolist(), want.reshape(-1).tolist()
    assert len(g) == len(w), why
    for i, (a, b) in enumerate(zip(g, w, strict=True)):
        if b != b:  # NaN must stay NaN
            assert a != a, f"{why}[{i}]: {a} != nan"
        else:
            assert a == pytest.approx(b, rel=2e-4, abs=2e-4), f"{why}[{i}]: {a} != {b}"


def _ids(case: tuple[tuple[int, ...], tuple[int, ...]]) -> str:
    shape, dims = case
    return "x".join(map(str, shape)) + "-d" + "".join(map(str, dims))


def _kind_cases() -> list[tuple[tuple[tuple[int, ...], tuple[int, ...]], str]]:
    """Every case under every kind; ``prod`` reduces one axis, so it takes the one-axis cases."""
    return [(c, k) for c in _CASES for k in sorted(_KINDS) if k != "prod" or len(c[1]) == 1]


@pytest.mark.parametrize(
    "case,kind", _kind_cases(), ids=lambda v: v if isinstance(v, str) else _ids(v)
)
def test_a_compiled_reduction_of_every_kind_matches_eager(
    case: tuple[tuple[int, ...], tuple[int, ...]], kind: str
) -> None:
    shape, dims = case
    keepdim = len(dims) % 2 == 0
    fn = _loss_fn(kind, dims, keepdim)
    el, cl, eg, cg = _eager_and_compiled(shape, fn, _input(shape))
    _same(cl, el, f"{kind} loss")
    _same(cg, eg, f"{kind} grad")


#: Rank-5+ cases: the first seven fuse below rank 5; the rest stay K R K R K (LCD-921) and
#: are reduced one block at a time.
_NAN_CASES: list[tuple[tuple[int, ...], tuple[int, ...]]] = [
    ((2, 3, 2, 3, 2), (0, 1)),
    ((2, 3, 2, 3, 2), (2,)),
    ((2, 3, 2, 3, 2), (0, 4)),
    ((2, 3, 2, 3, 2), (1, 2, 3, 4)),
    ((2, 3, 2, 3, 2, 2), (3, 5)),
    ((2, 3, 2, 3, 2, 2), (4, 5)),
    ((2, 3, 2, 3, 2, 2), (1, 2)),
    ((2, 3, 2, 3, 2), (0, 2, 4)),
    ((2, 3, 2, 3, 2), (1, 3)),
    ((2, 3, 2, 3, 2, 2), (1, 3, 5)),
    ((2, 3, 2, 3, 2, 2), (0, 2, 4)),
    ((2, 2, 2, 3, 2, 2, 3), (0, 2, 4, 6)),
]


def _nan_check(case: tuple[tuple[int, ...], tuple[int, ...]], kind: str) -> None:
    shape, dims = case
    n = 1
    for s in shape:
        n *= s
    fn = _loss_fn(kind, dims, False)
    el, cl, eg, cg = _eager_and_compiled(shape, fn, _input(shape, nan_at=n // 3))
    _same(cl, el, f"{kind} nan loss")
    _same(cg, eg, f"{kind} nan grad")


@pytest.mark.parametrize("kind", ["amax", "amin"])
@pytest.mark.parametrize("case", _NAN_CASES, ids=_ids)
def test_a_nan_in_a_fused_reduction_stays_nan(
    case: tuple[tuple[int, ...], tuple[int, ...]], kind: str
) -> None:
    _nan_check(case, kind)


@pytest.mark.parametrize("kind", ["amax", "amin"])
@pytest.mark.parametrize("case", _NAN_CASES[7:], ids=_ids)
def test_every_nan_and_every_tie_survive_a_blockwise_reduction(
    case: tuple[tuple[int, ...], tuple[int, ...]], kind: str
) -> None:
    """Several NaNs (one slice all NaN) and a tied extremum: the value and the gradient share."""
    shape, dims = case
    n = 1
    for s in shape:
        n *= s
    # Quantised values repeat, so a slice's extremum is shared by several elements.
    x = (lucid.sin(lucid.arange(n, dtype=lucid.float32) * 1.7) * 3.0).round().reshape(*shape)
    flat = x.reshape(-1)
    for i in (0, n // 4, n // 4 + 1, n - 1):
        flat[i] = float("nan")
    x = flat.reshape(*shape).to(COMPILE_DEVICE)
    fn = _loss_fn(kind, dims, False)
    el, cl, eg, cg = _eager_and_compiled(shape, fn, x)
    _same(cl, el, f"{kind} loss")
    _same(cg, eg, f"{kind} grad")


class _Reduce(nn.Module):
    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return x.sum(dim=(2, 4)).sum(dim=(1, 2)) + x.mean(dim=(1, 3)).sum(dim=(1, 2)) * 0.5


@pytest.mark.usefixtures("inference")
def test_a_symbolic_batch_survives_the_fused_reduction() -> None:
    m = to_metal(_Reduce()).eval()
    cm = lucid.compile(m, dynamic=True)
    for batch in (2, 5, 3):
        x = _input((batch, 3, 2, 4, 2))
        want = m(x)
        got = cm(x)
        assert got.shape == want.shape
        _same(got, want, f"batch {batch}")
    assert_cache_hit(cm, 1)


def _step_of(
    shape: tuple[int, ...], fn: Callable[[lucid.Tensor], lucid.Tensor]
) -> Callable[[], float]:
    return step_call(make_step(to_metal(_Scaled(shape)), fn), _input(shape))


def test_a_rank_5_reduction_is_not_slower_than_its_rank_3_twin() -> None:
    """Coarse on purpose: the unfixed graph was ~30x the twin's time, the fixed one ~1.3x."""
    shape = (96, 64, 32, 32)  # the view is made inside the loss; an input is rank 4 either way
    assert_ratio_within(
        lambda: {
            "rank-5": _step_of(shape, lambda y: y.reshape(6144, 16, 2, 16, 2).sum(dim=2).sum()),
            "rank-3": _step_of(shape, lambda y: y.reshape(6144 * 16, 2, 32).sum(dim=1).sum()),
        },
        "rank-5",
        "rank-3",
        bound=8.0,
        slack_ms=5.0,
    )
