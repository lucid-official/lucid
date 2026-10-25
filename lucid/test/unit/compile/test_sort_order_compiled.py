"""Compiled ordering ops name the positions eager names (LCD-425).

Eager orders stably — equal values in index order, so ``topk`` names ties
lowest index first (LCD-407) — treats NaN as the largest value (and, for
``argmin``, the smallest), calls -0.0 and +0.0 equal and compares int64
exactly.  MPSGraph's ``topK`` did none of that: ``topk([3, 3, 3, 3], 2)``
named 3 and 1, NaN came back as 0.0, and 2^40 + 3 lost to 2^40.  The
emitters now order through ``OpEmitters/_SortOrder.h``; this matrix — ties,
NaN, signed zeros, infinities, wide int64, every axis, long axes — holds
``topk`` / ``sort`` / ``argsort`` / ``argmax`` / ``argmin`` to the eager
answer exactly.
"""

import pytest

import lucid
import lucid.nn as nn
from lucid._C import engine as _C_engine

from lucid.test.unit.compile._helpers import COMPILE_DEVICE

NAN, INF = float("nan"), float("inf")
ROWS = [
    [0.0, 5.0, 4.0, 4.0, 1.0, 4.0],
    [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
    [1.0, NAN, 2.0, NAN, -INF, INF],
    [-0.0, 0.0, -1.0, 0.0, -0.0, 7.0],
    [2.0, 2.0, -2.0, -2.0, 2.0, -2.0],
]
FLOATS = {"f32": lucid.float32, "f16": lucid.float16, "bf16": lucid.bfloat16}
INTS = {"i32": lucid.int32, "i64": lucid.int64}


def _rows(dtype: lucid.dtype) -> lucid.Tensor:
    if dtype in INTS.values():
        src = [[0.0 if v != v or abs(v) == INF else v for v in r] for r in ROWS]
        return lucid.tensor(src).to(dtype=dtype).to(COMPILE_DEVICE)
    return lucid.tensor(ROWS).to(dtype=dtype).to(COMPILE_DEVICE)


def _same(fn: object, x: lucid.Tensor) -> None:
    _C_engine.compile.session_cache_clear()
    compiled = lucid.compile(fn)  # type: ignore[arg-type]
    got = compiled(x)
    info = compiled.cache_info()  # type: ignore[attr-defined]
    assert info["entries"] == 1 and not info["eager_only"], info
    want = fn(x)  # type: ignore[operator]
    for g, w in zip(
        got if isinstance(got, tuple) else (got,),
        want if isinstance(want, tuple) else (want,),
        strict=True,
    ):
        g_cpu, w_cpu = g.to("cpu"), w.to("cpu")
        if g_cpu.is_floating_point():
            assert (g_cpu.isnan() == w_cpu.isnan()).all().item()
            g_cpu = lucid.where(g_cpu.isnan(), lucid.zeros_like(g_cpu), g_cpu)
            w_cpu = lucid.where(w_cpu.isnan(), lucid.zeros_like(w_cpu), w_cpu)
            # -0.0 keeps its sign: compare the bits' sign as well.
            assert ((1.0 / g_cpu.float()) == (1.0 / w_cpu.float())).all().item()
        assert g_cpu.tolist() == w_cpu.tolist()


DTYPES = {**FLOATS, **INTS}


@pytest.mark.parametrize("dim", [-1, 0])
@pytest.mark.parametrize("k", [1, 2, 3, 5])
@pytest.mark.parametrize("dtype", list(DTYPES))
def test_topk(dtype: str, k: int, dim: int) -> None:
    x = _rows(DTYPES[dtype])
    _same(lambda t: tuple(o * 1 for o in lucid.topk(t, k, dim=dim)), x)


@pytest.mark.parametrize("dim", [-1, 0])
@pytest.mark.parametrize("dtype", list(DTYPES))
def test_sort_and_argsort(dtype: str, dim: int) -> None:
    x = _rows(DTYPES[dtype])
    _same(lambda t: tuple(o * 1 for o in lucid.sort(t, dim=dim)), x)
    _same(lambda t: lucid.argsort(t, dim=dim) * 1, x)
    _same(lambda t: lucid.argsort(t, dim=dim, descending=True) * 1, x)


@pytest.mark.parametrize("op", ["argmax", "argmin"])
@pytest.mark.parametrize("dim", [-1, 0])
@pytest.mark.parametrize("dtype", list(DTYPES))
def test_argmax_and_argmin(dtype: str, dim: int, op: str) -> None:
    fn = getattr(lucid, op)
    _same(lambda t: fn(t, dim=dim) * 1, _rows(DTYPES[dtype]))


@pytest.mark.parametrize("dtype", list(FLOATS))
def test_compiling_the_order_prints_no_mps_warning(
    dtype: str, capfd: pytest.CaptureFixture[str]
) -> None:
    """The ±0 key once ran a float16 ``equal`` that MPSGraph's
    compare-to-zero pass rejected with a warning on every compile."""
    x = _rows(FLOATS[dtype])
    capfd.readouterr()
    _same(lambda t: lucid.argsort(t) * 1, x)
    _same(lambda t: tuple(o * 1 for o in lucid.sort(t)), x)
    _same(lambda t: tuple(o * 1 for o in lucid.topk(t, 3)), x)
    out, err = capfd.readouterr()
    assert "mps_equal" not in out + err
    assert "ConvertBinaryCompareToZero" not in out + err


def _wide() -> lucid.Tensor:
    base = 1 << 40
    vals = [base + 1, base + 3, base + 2, -base, 1 << 33, base + 3 + (1 << 34), 5, base]
    return lucid.tensor(vals, dtype=lucid.int64).to(COMPILE_DEVICE)


def test_wide_int64_ranks_exactly() -> None:
    _same(lambda t: lucid.topk(t, 8)[1] * 1, _wide())
    _same(lambda t: lucid.argsort(t, descending=True) * 1, _wide())


def test_wide_int64_values_come_back_whole() -> None:
    """The values come back through ``_DataMovement.h`` (LCD-435)."""
    _same(lambda t: lucid.topk(t, 8)[0] * 1, _wide())
    _same(lambda t: lucid.sort(t, descending=True).values * 1, _wide())


@pytest.mark.parametrize("n", [5387, 1 << 17])
def test_ties_on_a_long_axis_stay_in_index_order(n: int) -> None:
    """Few distinct values, many ties: the order the sort keeps for them."""
    x = (lucid.arange(n) * 7919 % 5).to(lucid.float32).reshape(1, n).to(COMPILE_DEVICE)
    _same(lambda t: tuple(o * 1 for o in lucid.topk(t, 64)), x)
    _same(lambda t: lucid.argsort(t) * 1, x)


# LCD-461: ``topk`` with a small ``k`` selects rather than sorts.  The
# selection once marked a taken position -inf, so a real -inf was taken twice
# and a row of nothing but -inf named position -1 — the masked logits of beam
# search.  It keyed int32 through float32, so 2^24 + 1 tied with 2^24, and
# +inf through FLT_MAX, so the two tied.  MPSGraph's arg reductions also name
# -1 for a slice that is all -inf, all +inf, or all an integer extreme.
FMAX = 3.4028234663852886e38
_W = 40
_SELECT_ROWS = [
    [-INF] * _W,
    [0.1, -INF, -INF, -INF, 0.3] + [-INF] * (_W - 5),
    [NAN, INF, -INF, 1.0, NAN, INF, -INF, 1.0] * (_W // 8),
    [3.0] * _W,
    [FMAX, INF, -FMAX, -INF, -0.0, 0.0, 1e-45, -1e-45] * (_W // 8),
    [float((i * 7) % 5) for i in range(_W)],
    [INF] * _W,
    [NAN] * _W,
]
_I32 = [
    [-(1 << 31)] * _W,
    [(1 << 24) + d for d in (0, 1, -1, 1, 0, -2)] * 6 + [5, 5, -7, -(1 << 31)],
    [(1 << 31) - 1, (1 << 30) + 3, (1 << 30), (1 << 30) + 3] * (_W // 4),
    [7] * _W,
]
_I64 = [
    [-(1 << 63)] * _W,
    [(1 << 62) + d for d in (0, 1, -1, 1, 0, 5)] * 6 + [5, 5, -7, (1 << 63) - 1],
    [(1 << 24) + d for d in (0, 1, -1, 1)] * (_W // 4),
    [-((1 << 62) + 1), -(1 << 62), 3, -((1 << 62) + 1)] * (_W // 4),
]
SELECT_KS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 32]


def _select_rows(dtype: str) -> lucid.Tensor:
    if dtype == "i32":
        return lucid.tensor(_I32, dtype=lucid.int32).to(COMPILE_DEVICE)
    if dtype == "i64":
        return lucid.tensor(_I64, dtype=lucid.int64).to(COMPILE_DEVICE)
    return lucid.tensor(_SELECT_ROWS).to(dtype=FLOATS[dtype]).to(COMPILE_DEVICE)


@pytest.mark.parametrize("k", SELECT_KS)
@pytest.mark.parametrize("dtype", list(DTYPES))
def test_topk_on_infinities_nan_ties_and_wide_integers(dtype: str, k: int) -> None:
    _same(lambda t: tuple(o * 1 for o in lucid.topk(t, k)), _select_rows(dtype))


@pytest.mark.parametrize("k", [1, 2, 4])
@pytest.mark.parametrize("dtype", list(DTYPES))
def test_topk_across_rows(dtype: str, k: int) -> None:
    """Along the short axis, where each column is one of the rows' patterns."""
    _same(lambda t: tuple(o * 1 for o in lucid.topk(t, k, dim=0)), _select_rows(dtype))


def test_topk_takes_the_next_minus_inf() -> None:
    x = lucid.tensor([[0.1, -INF, -INF, -INF, 0.3, -INF]]).to(COMPILE_DEVICE)
    _same(lambda t: tuple(o * 1 for o in lucid.topk(t, 3)), x)
    values, indices = lucid.compile(lambda t: lucid.topk(t, 3))(x)
    assert indices.to("cpu").tolist() == [[4, 0, 1]]
    assert values.to("cpu").tolist()[0][2] == -INF


@pytest.mark.parametrize("k", [1, 8])
def test_topk_selects_whole_axis(k: int) -> None:
    """``k`` equal to the axis: every position, each exactly once."""
    x = lucid.tensor([[-INF] * k, [2.0] * k]).to(COMPILE_DEVICE)
    _same(lambda t: tuple(o * 1 for o in lucid.topk(t, k)), x)


@pytest.mark.parametrize("op", ["argmax", "argmin"])
@pytest.mark.parametrize("dtype", list(DTYPES))
def test_arg_extreme_of_a_constant_extreme_slice(dtype: str, op: str) -> None:
    fn = getattr(lucid, op)
    _same(lambda t: fn(t, dim=-1) * 1, _select_rows(dtype))
    _same(lambda t: fn(t, dim=0) * 1, _select_rows(dtype))


@pytest.mark.parametrize("n", [5387, 1 << 17])
def test_selection_ties_on_a_long_axis(n: int) -> None:
    x = (lucid.arange(2 * n) * 7919 % 5).to(lucid.float32).reshape(2, n)
    x = lucid.where(x == 0.0, lucid.full_like(x, -INF), x).to(COMPILE_DEVICE)
    _same(lambda t: tuple(o * 1 for o in lucid.topk(t, 8)), x)
    _same(lambda t: lucid.argmax(t, dim=-1) * 1, x)


class _TopKLoss(nn.Module):
    """A loss through ``topk`` of rows that are mostly -inf.  The loss is
    -inf, its gradient is not: each selected position takes its slot's
    ``probe`` value, so the gradient names the positions selected."""

    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k
        self.w = nn.Parameter(lucid.zeros(2, _W).to(COMPILE_DEVICE))
        self.probe = lucid.arange(1, k + 1).to(lucid.float32).to(COMPILE_DEVICE)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return (lucid.topk(x + self.w, self.k)[0] * self.probe).sum()


@pytest.mark.parametrize("k", [3, 8])
def test_topk_gradient_reaches_the_selected_positions(k: int) -> None:
    from lucid.compile import make_step

    x = _select_rows("f32")[:2] * 1
    model = _TopKLoss(k)
    model(x).backward()
    assert model.w.grad is not None
    want = model.w.grad.to("cpu").tolist()
    step = make_step(model, lambda out: out)
    for _ in range(2):  # trace, then replay
        model.w.grad = None
        step(x).backward()
    assert len(step.eager_only) == 0  # type: ignore[attr-defined]
    assert model.w.grad is not None
    assert model.w.grad.to("cpu").tolist() == want


def test_topk_selection_under_a_symbolic_batch() -> None:
    class _Top(nn.Module):
        def forward(self, t: lucid.Tensor) -> lucid.Tensor:
            v, i = lucid.topk(t, 3)
            return v * 1 + i.to(lucid.float32)

    model = _Top()
    compiled = lucid.compile(model, dynamic=True)
    for bs in (2, 5):
        x = _select_rows("f32")[1:2].expand(bs, _W) * 1
        assert compiled(x).to("cpu").tolist() == model(x).to("cpu").tolist()
