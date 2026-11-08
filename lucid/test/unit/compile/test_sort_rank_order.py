"""A compiled sort over many rows is as fast as eager and keeps eager's answer (LCD-926).

MPSGraph's ``argSort`` costs per row, about 4 us a row at any row length: a compiled
``sort`` of ``(256, 8, 16, 16)`` took 140 ms against 1.3 ms eager, ``(131072, 4)`` took 556
ms against 4 ms, and at 2^25 elements the call did not finish in two minutes.  The order now
comes from ``OpEmitters/_SortOrder.h``'s ``graph_sort_order``:

* few rows        -> ``argSort``;
* a short axis    -> *ranked*: the position of an element is how many come before it, every
  row compared with itself at once, then inverted by one scatter;
* up to 128 rows  -> ``argSort``;
* anything else   -> the emitter declines and the function runs eager.

``test_sort_order_compiled.py`` holds the few-row path to eager; this file holds the ranked
and declined paths — ties, NaN, signed zeros, wide int64, descending, every axis, rank 5 —
to the same answer, their gradients, and the speed.
"""

import time
import warnings
from collections.abc import Callable

import pytest

import lucid
import lucid.nn as nn
from lucid._C import engine as _C_engine
from lucid.compile import make_step
from lucid.compile._core.fallback import CompileFallbackWarning
from lucid.test._fixtures.devices import metal_available
from lucid.test.unit.compile._helpers import COMPILE_DEVICE
from lucid.test.unit.compile.test_sort_order_compiled import (
    FLOATS,
    INF,
    INTS,
    ROWS,
    _same,
    _wide,
)

pytestmark = [
    pytest.mark.skipif(not metal_available(), reason="compile needs Metal"),
    # A fallback to eager would make every compiled == eager row below vacuous.
    pytest.mark.filterwarnings("error::lucid.compile._core.fallback.CompileFallbackWarning"),
]

DTYPES = {**FLOATS, **INTS}
ROW_COUNT, WIDTH = 40, 12  # 40 rows is past the few-row ``argSort`` bound (16)


def _grid(dtype: lucid.dtype, rows: int = ROW_COUNT, width: int = WIDTH) -> lucid.Tensor:
    """``rows x width`` of ties, NaN, infinities and signed zeros: ``ROWS`` cycled, each
    block of five rows rotated by one column."""
    src = [[ROWS[r % 5][(c + r // 5) % 6] for c in range(width)] for r in range(rows)]
    if dtype in INTS.values():
        src = [[0.0 if v != v or abs(v) == INF else v for v in row] for row in src]
    return lucid.tensor(src).to(dtype=dtype).to(COMPILE_DEVICE)


@pytest.mark.parametrize("dtype", list(DTYPES))
def test_sort_and_argsort_over_many_rows(dtype: str) -> None:
    x = _grid(DTYPES[dtype])
    _same(lambda t: tuple(o * 1 for o in lucid.sort(t)), x)
    _same(lambda t: tuple(o * 1 for o in lucid.sort(t, descending=True)), x)
    _same(lambda t: tuple(o * 1 for o in lucid.sort(t, stable=True)), x)
    _same(lambda t: lucid.argsort(t) * 1, x)
    _same(lambda t: lucid.argsort(t, descending=True) * 1, x)


@pytest.mark.parametrize("dtype", list(DTYPES))
def test_sort_along_the_leading_and_a_middle_axis(dtype: str) -> None:
    x = _grid(DTYPES[dtype])  # (40, 12)
    _same(lambda t: tuple(o * 1 for o in lucid.sort(t, dim=0)), x.T.contiguous())
    _same(lambda t: lucid.argsort(t, dim=1, descending=True) * 1, x.reshape(4, 10, WIDTH))
    _same(lambda t: tuple(o * 1 for o in lucid.sort(t, dim=1)), x.reshape(4, 10, WIDTH))


@pytest.mark.parametrize("dim", [0, 2, -1])
def test_sort_of_a_rank_5_tensor(dim: int) -> None:
    """Rank five and up is folded around the axis: MPSGraph loops over the dims above the
    last four (LCD-918)."""
    x = _grid(lucid.float32, 30, 12).reshape(3, 2, 3, 4, 5)
    _same(lambda t: tuple(o * 1 for o in lucid.sort(t, dim=dim)), x)
    _same(lambda t: lucid.argsort(t, dim=dim, descending=True) * 1, x)


@pytest.mark.parametrize("k", [1, 3, 8, 10, 12])
@pytest.mark.parametrize("dtype", ["f32", "f16", "i32"])
def test_topk_over_many_rows(dtype: str, k: int) -> None:
    """``k`` past eight sorts instead of selecting, so it ranks."""
    x = _grid(DTYPES[dtype])
    _same(lambda t: tuple(o * 1 for o in lucid.topk(t, k)), x)
    _same(lambda t: tuple(o * 1 for o in lucid.topk(t, k, largest=False)), x)


@pytest.mark.parametrize("shape", [(4096, 32), (8192, 64)])
def test_few_distinct_values_keep_index_order_on_a_large_grid(shape: tuple[int, int]) -> None:
    """Seven distinct values over thousands of rows: every row is mostly ties, and the order
    the sort keeps for them is eager's."""
    n = shape[0] * shape[1]
    x = (lucid.arange(n) * 7919 % 7).to(lucid.float32).reshape(*shape).to(COMPILE_DEVICE)
    _same(lambda t: lucid.argsort(t) * 1, x)
    _same(lambda t: lucid.argsort(t, descending=True) * 1, x)
    _same(lambda t: tuple(o * 1 for o in lucid.topk(t, 12)), x)


def test_ranked_wide_int64_compares_exactly() -> None:
    """2^40 + 3 must not tie with 2^40: the ranking compares the int64 itself."""
    x = lucid.cat([_wide().reshape(1, 8)] * ROW_COUNT, dim=0) * 1
    _same(lambda t: tuple(o * 1 for o in lucid.sort(t)), x)
    _same(lambda t: lucid.argsort(t, descending=True) * 1, x)


def test_ranked_int32_keeps_its_low_bits() -> None:
    base = 1 << 30
    row = [base + 3, base, base + 1, -base - 5, base + 2, -base, base + 3, 7]
    x = lucid.tensor([row] * ROW_COUNT, dtype=lucid.int32).to(COMPILE_DEVICE)
    _same(lambda t: tuple(o * 1 for o in lucid.sort(t)), x)
    _same(lambda t: lucid.argsort(t, descending=True) * 1, x)


def test_ranked_bool_sorts_false_before_true() -> None:
    x = (lucid.arange(ROW_COUNT * 9).reshape(ROW_COUNT, 9) % 3 == 0).to(COMPILE_DEVICE)
    _same(lambda t: lucid.argsort(t) * 1, x)
    _same(lambda t: lucid.argsort(t, descending=True) * 1, x)


@pytest.mark.parametrize("name", ["kthvalue", "median", "msort"])
def test_sorting_composites_over_many_rows(name: str) -> None:
    x = _grid(lucid.float32)
    if name == "kthvalue":
        _same(lambda t: tuple(o * 1 for o in lucid.kthvalue(t, 4, dim=-1)), x)
    elif name == "median":
        _same(lambda t: tuple(o * 1 for o in lucid.median(t, dim=-1)), x)
    else:
        _same(lambda t: lucid.msort(t) * 1, x.T.contiguous())


def test_a_symbolic_batch_still_sorts() -> None:
    """A dynamic dimension is unknown when the graph is built, so ``argSort`` orders it."""

    class _Sort(nn.Module):
        def forward(self, t: lucid.Tensor) -> lucid.Tensor:
            v, i = lucid.sort(t)
            return v * 1 + i.to(lucid.float32)

    model = _Sort()
    compiled = lucid.compile(model, dynamic=True)
    for rows in (20, 33):
        x = _grid(lucid.float32, rows, WIDTH)
        got, want = compiled(x).to("cpu"), model(x).to("cpu")
        assert (got.isnan() == want.isnan()).all().item()
        assert lucid.where(got.isnan(), lucid.zeros_like(got), got).tolist() == (
            lucid.where(want.isnan(), lucid.zeros_like(want), want).tolist()
        )


class _OrderLoss(nn.Module):
    """``sum(order(x * w) * probe)``: every position's gradient is its own probe, so the
    gradient names exactly where each value was sent."""

    def __init__(self, kind: str, shape: tuple[int, int]) -> None:
        super().__init__()
        self.kind = kind
        n = shape[0] * shape[1]
        self.w = nn.Parameter(lucid.linspace(0.8, 1.2, n).reshape(*shape).to(COMPILE_DEVICE))
        width = 10 if kind == "topk" else shape[1]
        self.probe = lucid.linspace(1.0, 2.0, shape[0] * width).reshape(shape[0], width)
        self.probe = self.probe.to(COMPILE_DEVICE)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        y = x * self.w
        out = lucid.topk(y, 10)[0] if self.kind == "topk" else lucid.sort(y)[0]
        return (out * self.probe).sum()


@pytest.mark.parametrize("kind", ["sort", "topk"])
def test_gradient_through_a_ranked_order(kind: str) -> None:
    shape = (ROW_COUNT, WIDTH)
    n = shape[0] * shape[1]
    x = (lucid.sin(lucid.arange(n, dtype=lucid.float32) * 0.91) + 1.5).reshape(*shape)
    x = x.to(COMPILE_DEVICE)  # distinct values: the gradient does not depend on tie order
    eager = _OrderLoss(kind, shape)
    eager(x).backward()
    model = _OrderLoss(kind, shape)
    step = make_step(model, lambda out: out)
    for _ in range(2):  # trace, then replay
        model.w.grad = None
        step(x).backward()
    assert len(step.eager_only) == 0  # type: ignore[attr-defined]
    assert eager.w.grad is not None and model.w.grad is not None
    got, want = (
        model.w.grad.to("cpu").reshape(-1).tolist(),
        eager.w.grad.to("cpu").reshape(-1).tolist(),
    )
    assert got == pytest.approx(want, rel=1e-5, abs=1e-6)


@pytest.mark.parametrize("shape", [(512, 1024), (2048, 256)])
def test_a_shape_neither_path_makes_fast_runs_eager(shape: tuple[int, int]) -> None:
    """Too many rows to ``argSort`` and too many compares to rank: the emitter declines, the
    function runs eager and is neither wrong nor slower."""
    n = shape[0] * shape[1]
    x = (lucid.sin(lucid.arange(n, dtype=lucid.float32) * 0.37) + 1.5).reshape(*shape)
    x = x.to(COMPILE_DEVICE)
    fn = lambda t: lucid.sort(t * 1.0)[0] * 1  # noqa: E731
    _C_engine.compile.session_cache_clear()
    compiled = lucid.compile(fn)
    with pytest.warns(CompileFallbackWarning):
        got = compiled(x)
    assert compiled.cache_info()["eager_only"]  # type: ignore[attr-defined]
    assert got.to("cpu").tolist() == fn(x).to("cpu").tolist()


# ---------------------------------------------------------------------------------------
# Speed.  The unfixed graph was 100x eager, the fixed one ~1x; the bound is the card's 3x.
# ---------------------------------------------------------------------------------------


def _fill(shape: tuple[int, ...]) -> lucid.Tensor:
    n = 1
    for s in shape:
        n *= s
    return (lucid.sin(lucid.arange(n, dtype=lucid.float32) * 0.37) + 1.5).reshape(*shape)


def _best_ms(call: Callable[[], lucid.Tensor], reps: int = 15) -> float:
    for _ in range(3):
        float(call().item())
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        float(call().item())
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


_OPS: dict[str, Callable[[lucid.Tensor], lucid.Tensor]] = {
    "sort": lambda t: lucid.sort(t * 1.0, dim=-1)[0].reshape(-1).sum(),
    "sort_descending": lambda t: lucid.sort(t * 1.0, dim=-1, descending=True)[0].reshape(-1).sum(),
    "sort_stable": lambda t: lucid.sort(t * 1.0, dim=-1, stable=True)[0].reshape(-1).sum(),
    "sort_dim1": lambda t: lucid.sort(t * 1.0, dim=1)[0].reshape(-1).sum(),
    "argsort": lambda t: lucid.argsort(t * 1.0, dim=-1).to(lucid.float32).reshape(-1).sum(),
    "topk_select": lambda t: lucid.topk(t * 1.0, 4, dim=-1)[0].reshape(-1).sum(),
    "topk_sort": lambda t: lucid.topk(t * 1.0, 10, dim=-1)[0].reshape(-1).sum(),
    "topk_smallest": lambda t: lucid.topk(t * 1.0, 3, dim=-1, largest=False)[0].reshape(-1).sum(),
    "kthvalue": lambda t: lucid.kthvalue(t * 1.0, 5, dim=-1)[0].reshape(-1).sum(),
}


@pytest.mark.parametrize("name", list(_OPS))
@pytest.mark.parametrize("shape", [(256, 8, 16, 16), (64, 4, 8, 16, 16)])
def test_a_compiled_sort_is_not_slower_than_eager(name: str, shape: tuple[int, ...]) -> None:
    fn = _OPS[name]
    x = _fill(shape).to(COMPILE_DEVICE)
    ratios: list[float] = []
    with lucid.no_grad():
        for _ in range(3):  # a loaded machine times eager and compiled unevenly: retry
            _C_engine.compile.session_cache_clear()
            compiled = lucid.compile(fn)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", CompileFallbackWarning)
                got = _best_ms(lambda: compiled(x))
            want = _best_ms(lambda: fn(x))
            ratios.append(got / want)
            if ratios[-1] <= 3.0:
                return
    pytest.fail(f"compiled {name} {shape} ran {min(ratios):.1f}x eager (best of {len(ratios)})")
