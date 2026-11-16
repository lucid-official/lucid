"""An op that takes an index runs at rank <= 4 in a compiled graph (LCD-1014).

An index is checked against the axis it names before MPSGraph sees it (``OpEmitters/_IndexBounds.h``):
a run of elementwise compares and selects over the index, one select over the answer, and then the
gather or scatter itself.  Each of them reads or writes a tensor of the index's rank, so at rank 5+
the whole chain is a loop over the dims above the last four (~4.5 us a slice, M1 Pro, 1M elements):

* ``embedding`` of a ``(2048, 2, 2, 2, 8)`` index took 40 ms against 2 for the same rows flat, and
  of a rank-4 index (whose answer is rank 5) 11 ms; an int32 table with a rank-4 index took
  11.6 ms against 5.5 with ``LUCID_COMPILE_RANK_CANON_MIN_LEAD=-1``, because the wide-integer lane
  merge of LCD-997 only added a reshape round trip to an answer that is rank 5 by itself;
* a rank-5 / rank-6 ``gather`` (and so ``take_along_dim`` and ``index_select``, which record it)
  12 / 40 ms against 1.4, and a rank-5 ``scatter`` of any kind 17-58 ms against 1.2;
* a rank-5 ``searchsorted`` 640 ms and a rank-6 one 1850 ms against 5 ms flat: its binary search
  is a dozen elementwise ops per step;
* the first run of a ``one_hot`` of a ``(64, 4, 4, 8)`` index (a rank-5 answer) never finished: the
  watchdog gave it up after 120 s and the call ran eagerly.

The layout of an index op is free to change, so ``RankCanon`` owns it (``FlatIndexRule``,
``AlongAxisIndexRule``, ``SearchsortedRule``): the index of an ``embedding`` / ``one_hot`` is read
flat, a ``gather`` / ``scatter`` merges the axes before and after the one it names, ``searchsorted``
reads its values flat (one sequence) or by row (a batched one), and every answer is viewed back.  The
backward is built from the rewritten graph and follows.  Policy is the pass's lead gate.

Three guards, as for the other rules:

* the pass's IR, read through ``canonicalize_rank`` with no GPU: which nodes are rewritten, at which
  shapes, with which attribute, and what is left alone (below the gate, a symbolic batch on the named
  axis, a consumer that reads the answer wide);
* a value table on Metal: the rewritten graph (gate ``0``) equals eager and the native one (gate
  ``-1``) bit for bit at index ranks 1-6, on float and wide-integer tables, for the forward of every
  op above and the gradients of a compiled step; an index outside the table of an executable that was
  built from valid ones answers as the native graph does;
* the IR of each timed case against its twin (the same elements, the layout the rule reads them as):
  the same ops at the same shapes, none reading or writing rank 5+, and the same comparison with
  the pass off, which must fail.  The clock row for the same pairs is in the perf tier
  (``lucid/test/perf/compile/test_index_rank_twin.py``): a ~1 ms wall-clock ratio does not hold on
  the hosted runner, which read the int32 embedding at 2.04x for a graph that was its twin's
  (LCD-1039).
"""

import os
import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from unittest import mock

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._C import engine as _C_engine
from lucid.compile import make_step
from lucid.test.unit.compile._helpers import COMPILE_DEVICE, to_metal
from lucid.test.unit.compile._index_twins import CASES, Side, ramp as _ramp, rotation as _rotation
from lucid.test.unit.compile.test_rank_canonical_movement import (
    ENV,
    _canon,
    _filled,
    _named,
    _only,
    _shape,
    _trace,
    needs_metal,
)

pytestmark = pytest.mark.filterwarnings(
    "error::lucid.compile._core.fallback.CompileFallbackWarning"
)

Shape = tuple[int, ...]


@pytest.fixture(autouse=True)
def _default_gate() -> Iterator[None]:
    """Every test starts from the default lead threshold, whatever the shell exports."""
    with mock.patch.dict(os.environ):
        os.environ.pop(ENV, None)
        yield


def _numel(shape: Shape) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


# ── the IR ────────────────────────────────────────────────────────────────────────────

R4: Shape = (3, 4, 2, 3)
R5: Shape = (3, 4, 2, 3, 2)
R6: Shape = (3, 2, 2, 3, 2, 2)
ROWS, WIDTH = 11, 8


def _rows(shape: Shape, modulo: int = ROWS) -> lucid.Tensor:
    return (lucid.arange(_numel(shape)) % modulo).reshape(*shape)


def _table(dtype: lucid.dtype = lucid.float32) -> lucid.Tensor:
    return _filled((ROWS, WIDTH)).to(dtype)


@pytest.mark.parametrize("shape", [R4, R5, R6], ids=["rank_4", "rank_5", "rank_6"])
@pytest.mark.parametrize("dtype", [lucid.float32, lucid.int32], ids=str)
def test_an_embedding_reads_its_index_flat(shape: Shape, dtype: lucid.dtype) -> None:
    ops = _canon(lambda w, i: F.embedding(i, w), _table(dtype), _rows(shape))
    count = _numel(shape)
    assert _shape(_only(ops, "embedding")) == (count, WIDTH)
    assert [(n.name, _shape(n)) for n in ops[-2:]] == [
        ("embedding", (count, WIDTH)),
        ("reshape", (*shape, WIDTH)),
    ]
    assert (count,) in [_shape(n) for n in _named(ops, "reshape")]


@pytest.mark.parametrize("shape", [(24,), (6, 4), (3, 4, 2)], ids=["rank_1", "rank_2", "rank_3"])
def test_an_embedding_whose_answer_is_below_rank_5_is_left_alone(shape: Shape) -> None:
    ops = _canon(lambda w, i: F.embedding(i, w), _table(), _rows(shape))
    assert [n.name for n in ops] == ["embedding"] and _shape(ops[0]) == (*shape, WIDTH)


def test_an_embedding_below_the_lead_gate_is_left_alone() -> None:
    # the answer (3, 4, 2, 3, 8) has lead 3, under the default gate of 64
    ops = _canon(lambda w, i: F.embedding(i, w), _table(), _rows(R4), gate=None)
    assert _shape(_only(ops, "embedding")) == (*R4, WIDTH)


def test_an_embedding_under_a_symbolic_batch_is_left_alone() -> None:
    # the flat index would fold the batch axis in
    ops = _canon(lambda w, i: F.embedding(i, w), _table(), _rows(R5), dynamic=True)
    assert _shape(_only(ops, "embedding")) == (*R5, WIDTH)


@pytest.mark.parametrize("shape", [R4, R5, R6], ids=["rank_4", "rank_5", "rank_6"])
def test_a_one_hot_reads_its_index_flat(shape: Shape) -> None:
    ops = _canon(lambda i: F.one_hot(i, ROWS), _rows(shape))
    at = [n.name for n in ops].index("one_hot")
    assert _shape(ops[at]) == (_numel(shape), ROWS)
    # the cast to the result's dtype follows at the same layout; the view back is the answer
    assert ops[-1].name == "reshape" and _shape(ops[-1]) == (*shape, ROWS)


def test_a_one_hot_below_rank_5_is_left_alone() -> None:
    ops = _canon(lambda i: F.one_hot(i, ROWS), _rows((3, 4, 2)))
    assert "reshape" not in [n.name for n in ops] and _shape(_only(ops, "one_hot")) == (
        3,
        4,
        2,
        ROWS,
    )


def test_an_embedding_a_wide_consumer_reads_keeps_its_layout() -> None:
    """``topk`` has no rule and reads the answer at rank 5: the guard drops the rewrite, since the
    view would be materialised for it.  The value rows below reach the emitter this way."""
    ops = _canon(
        lambda w, i: lucid.topk(F.embedding(i, w), 2, dim=-1)[0], _table(lucid.int32), _rows(R4)
    )
    assert _shape(_only(ops, "embedding")) == (*R4, WIDTH)


def _ints_attr(node: _C_engine.compile.OpNode, key: str) -> int:
    return int(node.attrs[key])


_AXES: dict[str, tuple[Shape, int, Shape, int]] = {
    # name: (shape, dim, rewritten shape, rewritten dim)
    "last": (R5, -1, (72, 2), 1),
    "mid": (R5, 2, (12, 2, 6), 1),
    "second": (R5, 1, (3, 4, 12), 1),
    "first": (R5, 0, (3, 48), 0),
    "rank_6_mid": (R6, 3, (12, 3, 4), 1),
    "unit_axes": ((1, 6, 1, 1, 3), 1, (6, 3), 0),
}


@pytest.mark.parametrize("name", sorted(_AXES))
def test_a_gather_merges_the_axes_around_the_one_it_names(name: str) -> None:
    shape, dim, merged, at = _AXES[name]
    ops = _canon(lambda t, i: lucid.gather(t, dim, i), _filled(shape), _rows(shape, 2))
    node = _only(ops, "gather")
    assert _shape(node) == merged and _ints_attr(node, "axis") == at
    assert ops[-1].name == "reshape" and _shape(ops[-1]) == shape


def test_an_index_that_is_shorter_along_the_axis_is_merged_with_its_own_size() -> None:
    index = _rows((3, 4, 1, 3, 2), 2)
    ops = _canon(lambda t, i: lucid.gather(t, 2, i), _filled(R5), index)
    assert _shape(_only(ops, "gather")) == (12, 1, 6)


def test_take_along_dim_and_index_select_record_a_gather_and_follow() -> None:
    ops = _canon(lambda t, i: lucid.take_along_dim(t, i, 2), _filled(R5), _rows(R5, 2))
    assert _shape(_only(ops, "gather")) == (12, 2, 6)
    ops = _canon(lambda t: lucid.index_select(t, 2, lucid.tensor([1, 0])), _filled(R5))
    assert _shape(_only(ops, "gather")) == (12, 2, 6)


_SCATTERS: dict[str, tuple[Callable[..., lucid.Tensor], str]] = {
    "set": (lambda t, i, s: t.scatter(2, i, s), "scatter_set"),
    "add": (lambda t, i, s: t.scatter(2, i, s, reduce="add"), "scatter_add"),
    "multiply": (lambda t, i, s: t.scatter(2, i, s, reduce="multiply"), "scatter_prod"),
    "amax": (lambda t, i, s: lucid.scatter_reduce(t, 2, i, s, reduce="amax"), "scatter_amax"),
    "amin": (lambda t, i, s: lucid.scatter_reduce(t, 2, i, s, reduce="amin"), "scatter_amin"),
}


@pytest.mark.parametrize("name", sorted(_SCATTERS))
def test_a_scatter_merges_around_its_axis_in_the_base_the_index_and_the_source(name: str) -> None:
    fn, op = _SCATTERS[name]
    ops = _canon(fn, _filled(R5), _rows(R5, 2), _filled(R5, 1.0))
    node = _only(ops, op)
    assert _shape(node) == (12, 2, 6) and _ints_attr(node, "dim") == 1
    assert [_shape(n) for n in _named(ops, "reshape")[:2]] == [(12, 2, 6)] * 2
    assert _shape(ops[-1]) == R5


def test_a_scatter_of_a_shorter_index_merges_it_with_its_own_size() -> None:
    """The engine cuts the source to the index (``split_at``), so the node sees them alike."""
    ops = _canon(
        lambda t, i, s: t.scatter(2, i, s), _filled(R5), _rows((3, 4, 1, 3, 2), 2), _filled(R5)
    )
    node = _only(ops, "scatter_set")
    assert _shape(node) == (12, 2, 6)
    assert (12, 1, 6) in [_shape(n) for n in _named(ops, "reshape")]


def test_a_gather_keeps_the_batch_axis_of_a_symbolic_batch_apart() -> None:
    ops = _canon(lambda t, i: lucid.gather(t, -1, i), _filled(R5), _rows(R5, 2), dynamic=True)
    assert _shape(_only(ops, "gather")) == (3, 24, 2)
    # ... and cannot name it
    ops = _canon(lambda t, i: lucid.gather(t, 0, i), _filled(R5), _rows(R5, 2), dynamic=True)
    assert _shape(_only(ops, "gather")) == R5


def test_a_gather_below_the_lead_gate_is_left_alone() -> None:
    ops = _canon(lambda t, i: lucid.gather(t, -1, i), _filled(R5), _rows(R5, 2), gate=None)
    assert [n.name for n in ops] == ["gather"] and _shape(ops[0]) == R5


@pytest.mark.parametrize("shape", [R5, R6], ids=["rank_5", "rank_6"])
def test_a_search_of_one_sequence_reads_its_values_flat(shape: Shape) -> None:
    sequence = lucid.sort(_filled((16,)), dim=-1)[0]
    ops = _canon(lambda s, v: lucid.searchsorted(s, v), sequence, _filled(shape))
    assert _shape(_only(ops, "searchsorted")) == (_numel(shape),)
    assert _shape(ops[-1]) == shape


def test_a_search_with_a_sequence_per_row_merges_the_rows() -> None:
    sequence = lucid.sort(_filled((*R5[:-1], 6)), dim=-1)[0]
    ops = _canon(lambda s, v: lucid.searchsorted(s, v), sequence, _filled(R5))
    assert _shape(_only(ops, "searchsorted")) == (72, 2)
    assert [_shape(n) for n in _named(ops, "reshape")[:2]] == [(72, 6), (72, 2)]


def test_a_search_under_a_symbolic_batch_keeps_the_batch_axis() -> None:
    one = lucid.sort(_filled((16,)), dim=-1)[0]
    ops = _canon(lambda s, v: lucid.searchsorted(s, v), one, _filled(R5), dynamic=True)
    assert _shape(_only(ops, "searchsorted")) == R5  # one sequence: no axis to keep, left alone
    rows = lucid.sort(_filled((*R5[:-1], 6)), dim=-1)[0]
    ops = _canon(lambda s, v: lucid.searchsorted(s, v), rows, _filled(R5), dynamic=True)
    assert _shape(_only(ops, "searchsorted")) == (3, 24, 2)


# ── values ────────────────────────────────────────────────────────────────────────────

_RANKS: dict[int, Shape] = {
    1: (24,),
    2: (6, 4),
    3: (3, 4, 2),
    4: (2, 3, 2, 3),
    5: (3, 2, 2, 3, 2),
    6: (2, 2, 2, 3, 2, 2),
}

_DTYPES = {"float32": lucid.float32, "int32": lucid.int32, "int64": lucid.int64}
_BITS = {"int32": 32, "int64": 64}


def _wide(shape: Shape, dtype: str, seed: int) -> lucid.Tensor:
    """Every bit of the dtype in play (float32 keeps none of the low ones of a wide integer)."""
    rng = random.Random(seed)
    bits = _BITS[dtype]
    values = [rng.getrandbits(bits) - 2 ** (bits - 1) for _ in range(_numel(shape))]
    return lucid.tensor(values, dtype=_DTYPES[dtype]).reshape(*shape).to(COMPILE_DEVICE)


def _operand(shape: Shape, dtype: str, seed: int) -> lucid.Tensor:
    if dtype in _BITS:
        return _wide(shape, dtype, seed)
    return _filled(shape, 0.01 * seed).to(COMPILE_DEVICE)


def _positions(shape: Shape, extent: int, seed: int, spill: int = 0) -> lucid.Tensor:
    """Positions along an axis of ``extent`` -- negative ones (which wrap) and, with ``spill``,
    ones that name nothing -- as a tensor on the device."""
    rng = random.Random(seed)
    values = [rng.randrange(-extent - spill, extent + spill) for _ in range(_numel(shape))]
    return lucid.tensor(values).reshape(*shape).to(COMPILE_DEVICE)


def _row_ids(shape: Shape, seed: int, spill: int = 0) -> lucid.Tensor:
    """Row numbers of the table: in range, or (with ``spill``) outside it."""
    rng = random.Random(seed)
    values = [rng.randrange(-spill, ROWS + spill) for _ in range(_numel(shape))]
    return lucid.tensor(values).reshape(*shape).to(COMPILE_DEVICE)


def _flat(t: lucid.Tensor) -> list[float]:
    return t.to("cpu").reshape(-1).tolist()


def _alike(a: float, b: float) -> bool:
    return a == b or (a != a and b != b)


def _same(got: lucid.Tensor, want: lucid.Tensor, why: str) -> None:
    g, w = _flat(got), _flat(want)
    assert len(g) == len(w), why
    bad = [i for i, (a, b) in enumerate(zip(g, w, strict=True)) if not _alike(a, b)]
    assert (
        not bad
    ), f"{why}: {len(bad)} values differ, first at {bad[0]}: {g[bad[0]]} != {w[bad[0]]}"


def _run(
    fn: Callable[..., lucid.Tensor], gate: str, *calls: tuple[lucid.Tensor, ...]
) -> list[lucid.Tensor]:
    """``fn`` compiled under ``gate``, called with each of ``calls`` on one executable."""
    with mock.patch.dict(os.environ, {ENV: gate}):
        _C_engine.compile.session_cache_clear()
        compiled = lucid.compile(fn)
        out = [compiled(*args) for args in calls]
        info = compiled.cache_info()
    assert info["entries"] == 1 and not info["eager_only"], info
    return out


def _holds(fn: Callable[..., lucid.Tensor], *args: lucid.Tensor) -> None:
    """The rewritten graph, the native one and eager give the same values."""
    want = fn(*args)
    _same(_run(fn, "0", args)[0], want, "rewritten vs eager")
    _same(_run(fn, "-1", args)[0], want, "native vs eager")


def _embed(w: lucid.Tensor, i: lucid.Tensor) -> lucid.Tensor:
    return F.embedding(i, w) + 0


@needs_metal
@pytest.mark.parametrize("dtype", sorted(_DTYPES))
@pytest.mark.parametrize("rank", sorted(_RANKS))
def test_an_embedding_is_whole_at_every_index_rank(rank: int, dtype: str) -> None:
    table = _operand((ROWS, WIDTH), dtype, 1)
    _holds(_embed, table, _row_ids(_RANKS[rank], 2))


@needs_metal
@pytest.mark.parametrize("rank", sorted(_RANKS))
def test_a_one_hot_is_whole_at_every_index_rank(rank: int) -> None:
    _holds(lambda i: F.one_hot(i, ROWS) + 0, _row_ids(_RANKS[rank], 6))


@needs_metal
@pytest.mark.usefixtures("inference")
def test_a_one_hot_of_a_rank_4_index_finishes_its_first_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MPSGraph never finished the first run of this graph unflattened (``(64, 4, 4, 8)``; the
    watchdog gave it up after 120 s and the call ran eagerly, which the module's warning filter
    turns into a failure); the bound here keeps a regression to 15 s."""
    monkeypatch.setenv("LUCID_COMPILE_FIRST_RUN_TIMEOUT_S", "15")
    index = _ramp((64, 4, 4, 8), 16, lucid.int64)
    fn: Callable[..., lucid.Tensor] = lambda i: F.one_hot(i, 16)
    _same(_run(fn, "64", (index,))[0], fn(index), "compiled vs eager")


@needs_metal
@pytest.mark.parametrize("dtype", sorted(_DTYPES))
@pytest.mark.parametrize("rank", [4, 5, 6])
def test_a_row_outside_the_table_answers_as_the_native_graph_does(rank: int, dtype: str) -> None:
    """The graph is built from valid rows and then fed others: the check is in the graph, and it
    runs on the flat index now.  NaN for a float table, 0 for an integer one (policy B)."""
    table = _operand((ROWS, WIDTH), dtype, 3)
    valid, spilled = _row_ids(_RANKS[rank], 4), _row_ids(_RANKS[rank], 5, spill=ROWS)
    on = _run(_embed, "0", (table, valid), (table, spilled))
    off = _run(_embed, "-1", (table, valid), (table, spilled))
    _same(on[0], off[0], "valid rows, rewritten vs native")
    _same(on[1], off[1], "rows outside, rewritten vs native")
    assert any(r < 0 or r >= ROWS for r in _flat(spilled))


@dataclass(frozen=True)
class _Axis:
    shape: Shape
    dim: int
    #: the index's length along ``dim`` as ``extent + grow`` (an index may be shorter or longer).
    grow: int = 0


_GATHER_AXES: dict[str, _Axis] = {
    "last": _Axis(R5, -1),
    "last_longer": _Axis(R5, -1, 2),
    "mid": _Axis(R5, 2),
    "second": _Axis(R5, 1),
    "first_shorter": _Axis(R5, 0, -1),
    "unit_axes": _Axis((1, 6, 1, 1, 3), 1, 2),
    "rank_6_mid": _Axis(R6, 3),
    "rank_6_last_shorter": _Axis(R6, -1, -1),
}


def _along(axis: _Axis, seed: int, spill: int = 0) -> lucid.Tensor:
    shape = list(axis.shape)
    shape[axis.dim] += axis.grow
    return _positions(tuple(shape), axis.shape[axis.dim], seed, spill)


@needs_metal
@pytest.mark.parametrize("dtype", sorted(_DTYPES))
@pytest.mark.parametrize("name", sorted(_GATHER_AXES))
def test_a_gather_is_whole_at_every_axis(name: str, dtype: str) -> None:
    axis = _GATHER_AXES[name]
    data = _operand(axis.shape, dtype, 11)
    _holds(lambda t, i: lucid.gather(t, axis.dim, i) + 0, data, _along(axis, 12))


@needs_metal
@pytest.mark.parametrize("dtype", sorted(_DTYPES))
@pytest.mark.parametrize("name", ["last", "mid", "first_shorter", "rank_6_mid"])
def test_take_along_dim_and_index_select_are_whole(name: str, dtype: str) -> None:
    axis = _GATHER_AXES[name]
    data = _operand(axis.shape, dtype, 21)
    picks = _positions((5,), axis.shape[axis.dim], 22)
    _holds(lambda t, i: lucid.take_along_dim(t, i, axis.dim) + 0, data, _along(axis, 23))
    _holds(lambda t, p: lucid.index_select(t, axis.dim, p) + 0, data, picks)


@needs_metal
@pytest.mark.parametrize("name", ["last", "mid", "first_shorter"])
def test_a_position_outside_the_axis_answers_as_the_native_graph_does(name: str) -> None:
    axis = _GATHER_AXES[name]
    data = _operand(axis.shape, "float32", 31)
    fn: Callable[..., lucid.Tensor] = lambda t, i: lucid.gather(t, axis.dim, i) + 0
    valid, spilled = _along(axis, 32), _along(axis, 33, spill=axis.shape[axis.dim])
    on = _run(fn, "0", (data, valid), (data, spilled))
    off = _run(fn, "-1", (data, valid), (data, spilled))
    _same(on[1], off[1], "positions outside, rewritten vs native")


_REDUCTIONS: dict[str, Callable[[int], Callable[..., lucid.Tensor]]] = {
    "set": lambda d: lambda t, i, s: t.scatter(d, i, s) + 0,
    "add": lambda d: lambda t, i, s: t.scatter(d, i, s, reduce="add") + 0,
    "multiply": lambda d: lambda t, i, s: t.scatter(d, i, s, reduce="multiply") + 0,
    "amax": lambda d: lambda t, i, s: lucid.scatter_reduce(t, d, i, s, reduce="amax") + 0,
    "amin": lambda d: lambda t, i, s: lucid.scatter_reduce(t, d, i, s, reduce="amin") + 0,
}

#: an int64 reduction other than a set or a sum is eager (the emitter declines it).
_SCATTER_ROWS = [
    (reduce, dtype)
    for reduce in sorted(_REDUCTIONS)
    for dtype in sorted(_DTYPES)
    if dtype != "int64" or reduce in {"set", "add"}
]


@needs_metal
@pytest.mark.parametrize(("reduce", "dtype"), _SCATTER_ROWS)
@pytest.mark.parametrize("name", ["last", "mid", "first_shorter", "rank_6_mid"])
def test_a_scatter_is_whole_at_every_axis(name: str, reduce: str, dtype: str) -> None:
    axis = _GATHER_AXES[name]
    written = list(axis.shape)
    written[axis.dim] += axis.grow
    base = _operand(axis.shape, dtype, 41)
    if reduce == "set":
        index = _rotation(tuple(written), axis.dim, axis.shape[axis.dim])
    else:
        index = _positions(tuple(written), axis.shape[axis.dim], 42)
    # small whole numbers: a float sum, product and extremum of them are exact in any order
    src = (lucid.arange(_numel(tuple(written))) % 3 + 1).reshape(*written)
    src = src.to(_DTYPES[dtype]).to(COMPILE_DEVICE)
    if dtype in _BITS and reduce != "set":
        base = (lucid.arange(_numel(axis.shape)) % 5).reshape(*axis.shape).to(_DTYPES[dtype])
        base = base.to(COMPILE_DEVICE)
    _holds(_REDUCTIONS[reduce](axis.dim), base, index, src)


def _sequence(shape: Shape, dtype: str, seed: int) -> lucid.Tensor:
    return lucid.sort(_operand(shape, dtype, seed), dim=-1)[0]


@needs_metal
@pytest.mark.parametrize("dtype", sorted(_DTYPES))
@pytest.mark.parametrize("right", [False, True], ids=["left", "right"])
@pytest.mark.parametrize("shape", [R5, R6], ids=["rank_5", "rank_6"])
def test_a_search_of_one_sequence_is_whole(shape: Shape, right: bool, dtype: str) -> None:
    sequence = _sequence((16,), dtype, 51)
    values = _operand(shape, dtype, 52)
    # half the values are ones the sequence holds, so the two sides of a tie differ
    held = lucid.where(
        _rows(shape, 2).to(COMPILE_DEVICE) == 0, values, sequence[3].to(values.dtype)
    )
    _holds(lambda s, v: lucid.searchsorted(s, v, right=right), sequence, held)


@needs_metal
@pytest.mark.parametrize("dtype", sorted(_DTYPES))
@pytest.mark.parametrize("right", [False, True], ids=["left", "right"])
def test_a_search_with_a_sequence_per_row_is_whole(right: bool, dtype: str) -> None:
    sequence = _sequence((*R5[:-1], 6), dtype, 61)
    values = _operand(R5, dtype, 62)
    _holds(lambda s, v: lucid.searchsorted(s, v, right=right), sequence, values)


@needs_metal
@pytest.mark.parametrize("dtype", ["int32", "int64"])
@pytest.mark.parametrize("rank", [4, 5])
def test_an_embedding_the_pass_leaves_at_rank_5_is_whole(rank: int, dtype: str) -> None:
    """The rewrite is dropped when a reader (here ``topk``) would read the answer at rank 5+, so
    ``lane_take`` sees the wide answer itself: it merges around the table only when the answer is
    rank <= 4 uncarried (rank 4 index -> rank 5 here, and the merge was 1.4x-1.9x slower)."""
    table = _wide((ROWS, WIDTH), dtype, 71)
    fn: Callable[..., lucid.Tensor] = lambda w, i: lucid.topk(F.embedding(i, w), 2, dim=-1)[0]
    _holds(fn, table, _row_ids(_RANKS[rank], 72))


# ── the backward ──────────────────────────────────────────────────────────────────────


class _Lookup(nn.Module):
    """``embedding(i, w)`` scaled: ``w``'s gradient is the backward of the embedding."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(_filled((ROWS, WIDTH), 0.4).to(COMPILE_DEVICE))

    def forward(self, i: lucid.Tensor) -> lucid.Tensor:
        return F.embedding(i, self.w) * 1.5


class _Pick(nn.Module):
    """``gather`` of a parameter: its gradient is a scatter-add over the same merged layout."""

    def __init__(self, shape: Shape, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.x = nn.Parameter(_filled(shape, 0.6).to(COMPILE_DEVICE))

    def forward(self, i: lucid.Tensor) -> lucid.Tensor:
        return lucid.gather(self.x, self.dim, i) * 1.5


def _loss(y: lucid.Tensor) -> lucid.Tensor:
    return (y * y).sum()


def _grad_of(
    make: Callable[[], nn.Module], arg: lucid.Tensor, gate: str
) -> tuple[float, lucid.Tensor]:
    model = to_metal(make())
    with mock.patch.dict(os.environ, {ENV: gate}):
        _C_engine.compile.session_cache_clear()
        loss = make_step(model, _loss)(arg)
        loss.backward()
    (grad,) = [p.grad for p in model.parameters()]
    assert grad is not None
    return float(loss.item()), grad


@needs_metal
@pytest.mark.parametrize(
    ("make", "index"),
    [
        (_Lookup, _row_ids(R5, 81)),
        (_Lookup, _row_ids(R4, 82)),
        (lambda: _Pick(R5, 2), _positions(R5, R5[2], 83)),
        (lambda: _Pick(R5, -1), _positions(R5, R5[-1], 84)),
        (lambda: _Pick(R6, 0), _positions(R6, R6[0], 85)),
    ],
    ids=[
        "embedding_rank_5",
        "embedding_rank_4",
        "gather_mid",
        "gather_last",
        "gather_rank_6_first",
    ],
)
def test_the_backward_of_an_index_op_follows_the_rewritten_graph(
    make: Callable[[], nn.Module], index: lucid.Tensor
) -> None:
    eager = to_metal(make())
    want = _loss(eager(index))
    want.backward()
    (want_grad,) = [p.grad for p in eager.parameters()]
    assert want_grad is not None
    for gate in ("0", "-1"):
        loss, grad = _grad_of(make, index, gate)
        assert loss == pytest.approx(float(want.item()), rel=1e-5), gate
        scale = max(max(abs(v) for v in _flat(want_grad)), 1.0)
        got, ref = _flat(grad / scale), _flat(want_grad / scale)
        assert all(a == pytest.approx(b, abs=3e-5) for a, b in zip(got, ref, strict=True)), gate


# ── a symbolic batch ──────────────────────────────────────────────────────────────────


@needs_metal
@pytest.mark.usefixtures("inference")
def test_a_symbolic_batch_survives_the_index_rewrites() -> None:
    table = _filled((ROWS, WIDTH), 0.2).to(COMPILE_DEVICE)

    def fn(t: lucid.Tensor, i: lucid.Tensor, r: lucid.Tensor) -> lucid.Tensor:
        picked = lucid.gather(t, -1, i) + 0.0
        return picked + F.embedding(r, table)[..., 3]

    with mock.patch.dict(os.environ, {ENV: "0"}):
        _C_engine.compile.session_cache_clear()
        compiled = lucid.compile(fn, dynamic=True)
        for batch in (3, 5, 3):
            shape = (batch, *R5[1:])
            data = _filled(shape, 0.1 * batch).to(COMPILE_DEVICE)
            index, rows = _positions(shape, shape[-1], 90 + batch), _row_ids(shape, 95 + batch)
            _same(compiled(data, index, rows), fn(data, index, rows), f"batch {batch}")


# ── the twin: the same ops at the same shapes ─────────────────────────────────────────
#
# A twin holds the same elements as its case in the layout the rules read it as
# (``_index_twins.CASES``), so a graph the pass has rewritten computes what the twin computes:
# the same ops at the same shapes and attributes, only the views differ.  That is asserted on the
# IR, which is deterministic.  The clock row that used to sit here (rank-5 not slower than the
# twin, 1.5x + 0.5 ms) flaked on the hosted runner at 2.04x for a graph that was the twin's, so it
# runs in the recorded perf tier (``lucid/test/perf/compile/test_index_rank_twin.py``, LCD-1039).
# What the IR cannot see -- an emitter that is slow for a layout both sides share, the ~0.2 ms
# MPSGraph copies a rank-4 feed read through the pass's flat view -- is that tier's to report.

IrOps = list[_C_engine.compile.OpNode]


def _ir(side: Side, gate: str | None) -> tuple[IrOps, dict[int, Shape]]:
    """The pass's output for ``side`` and the shape of every tensor id in it (feeds included)."""
    graph, feeds = _trace(side.fn, *side.args)
    with mock.patch.dict(os.environ, {} if gate is None else {ENV: gate}):
        ops = list(_C_engine.compile.canonicalize_rank(graph, feeds, False).ops)
    shapes = {fid: tuple(feed.shape) for fid, feed in feeds.items()}
    shapes |= {out.id: tuple(out.shape) for node in ops for out in node.outputs}
    return ops, shapes


def _computes(ops: IrOps) -> list[tuple[str, Shape, dict[str, object]]]:
    """What the graph computes, views aside: each op but a reshape, its answer's shape and attrs."""
    return [(n.name, tuple(n.outputs[0].shape), dict(n.attrs)) for n in ops if n.name != "reshape"]


def _read_or_write_rank_5(ops: IrOps, shapes: dict[int, Shape]) -> list[str]:
    """The ops, a reshape (a free view) aside, that read or write a tensor of rank >= 5."""
    return [
        n.name
        for n in ops
        if n.name != "reshape"
        and any(len(shapes[i]) > 4 for i in (*n.inputs, *(out.id for out in n.outputs)))
    ]


@pytest.mark.parametrize("name", sorted(CASES))
def test_a_rank_5_index_op_computes_what_its_twin_computes(name: str) -> None:
    """The case and the twin share every op, shape and attribute, and nothing computes at rank 5+.

    Equal answers also say the twin holds the case's elements: a twin of a quarter of the index
    (the int32 embedding's, until LCD-1039) names a smaller embedding and fails here.
    """
    case = CASES[name]("cpu")
    wide, wide_shapes = _ir(case.wide, None)
    twin, twin_shapes = _ir(case.twin, None)
    assert case.op in [n.name for n in wide]
    assert _read_or_write_rank_5(wide, wide_shapes) == []
    assert _read_or_write_rank_5(twin, twin_shapes) == []
    assert _computes(wide) == _computes(twin)


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_twin_comparison_fails_for_the_unrewritten_graph(name: str) -> None:
    """The guard above can fail: with the pass off the op reads rank 5+ and is not the twin's."""
    case = CASES[name]("cpu")
    wide, wide_shapes = _ir(case.wide, "-1")
    twin, _ = _ir(case.twin, None)
    assert case.op in _read_or_write_rank_5(wide, wide_shapes)
    assert _computes(wide) != _computes(twin)


# ── no timing row for the wide-integer lane merge of an answer that is rank 5 by itself ──
#
# ``lane_take`` (``_AxisSplitLayout.h``) merged whenever the carried answer was rank 5+, and for an
# int32 table of ``(1000, 32)`` with an index of ``(512, 4, 4, 8)`` -- an answer that is rank 5 before
# any lanes are added -- that only added a reshape round trip: 11.2 ms against 8.2 native, 1.4x to
# 1.9x slower over three shapes (``argmax(embedding)`` below, the consumer that keeps the pass from
# flattening the index: 14.3 / 28.9 / 88.9 ms merged against 9.9 / 18.8 / 70.0 native).  It now merges
# only when the table's own carried shape is wide, or the answer is rank <= 4 uncarried (the merged
# rows of LCD-1002: 1.1x for an embedding of a rank-3 index, 1.3x to 1.4x for ``searchsorted`` of
# rank-4 values; kept).  After the rewrite above the pass flattens the index of every such embedding
# first, so the emitter sees a rank-1 index and the lane merge is reached only through a consumer the
# pass has no rule for (``test_an_embedding_the_pass_leaves_at_rank_5_is_whole``, held by value).
# The clock cannot hold it: the gap, 1.3x to 1.5x, is under the 1.5x floor a guard may state
# (``MIN_RATIO_BOUND``), and the consumer's own cost at rank 5 is in both sides.
