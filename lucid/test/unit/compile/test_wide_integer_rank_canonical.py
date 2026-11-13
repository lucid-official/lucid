"""A gather of a wide integer keeps its carried operand at rank <= 4 (LCD-997).

MPSGraph's gather rounds int32 and int64 through float32 (LCD-435), so ``_DataMovement.h`` carries
such an operand as int16 lanes on one more trailing axis.  A rank-4 operand is then a rank-5
intermediate, which MPSGraph runs as a loop over the dims above the last four (~4.5 us a slice):
``lucid.gather`` of an int32 ``(8192, 8, 16, 16)`` took 53 ms against 12 for the same elements at
rank 3 (M1 Pro), a float32 one 7.2 against 6.6.  The helper reads the operand and the index as
``(pre, d, post)`` around the gathered axis before the lanes are added and reads the answer back
(``OpEmitters/_AxisSplitLayout.h``), so every user of ``graph_gather`` /
``graph_gather_along_axis`` -- ``gather``, ``take_along_dim``, ``index_select``, ``embedding``,
``sort`` / ``topk`` values, the overwrite scatter that reads its winners back -- follows.  The
policy is the rank pass's lead gate: a small lead, a symbolic dim and
``LUCID_COMPILE_RANK_CANON_MIN_LEAD=-1`` keep the native spelling.

The graph cannot be inspected from Python, so the guards are

* a value table on Metal: with the gate at ``0`` (merge whenever the carried tensor is rank 5+)
  every case equals eager's and the native spelling's (gate ``-1``) bit for bit, over values that
  float32 would round, the axes that change where the merge falls (first, middle, last, bands of
  size 1), an index shorter and longer than the axis, and rank 5;
* a symbolic batch, which keeps the native spelling;
* a coarse timing bound against the rank-3 twin of each case (the same elements), and the same
  measurement with the gate off, which must exceed the bound (a bound the unfixed graph meets
  guards nothing).
"""

import os
import random
from collections.abc import Callable
from dataclasses import dataclass
from unittest import mock

import pytest

import lucid
import lucid.nn.functional as F
from lucid._C import engine as _C_engine
from lucid.test.unit.compile._helpers import (
    COMPILE_DEVICE,
    assert_ratio_within,
    compare_ms,
    compiled_call,
    with_env,
)
from lucid.test.unit.compile.test_rank_canonical_movement import ENV, needs_metal

pytestmark = pytest.mark.filterwarnings(
    "error::lucid.compile._core.fallback.CompileFallbackWarning"
)

Shape = tuple[int, ...]

_DTYPES = {"int32": lucid.int32, "int64": lucid.int64}
_BITS = {"int32": 32, "int64": 64}


def _numel(shape: Shape) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


def _wide(shape: Shape, dtype: str, seed: int) -> lucid.Tensor:
    """Every bit of the dtype in play: float32 keeps none of the low ones of these."""
    rng = random.Random(seed)
    bits = _BITS[dtype]
    values = [rng.getrandbits(bits) - 2 ** (bits - 1) for _ in range(_numel(shape))]
    return lucid.tensor(values, dtype=_DTYPES[dtype]).reshape(*shape).to(COMPILE_DEVICE)


def _index(shape: Shape, extent: int, seed: int) -> lucid.Tensor:
    """Positions along an axis of ``extent`` -- negative ones (which wrap) included."""
    rng = random.Random(seed)
    values = [rng.randrange(-extent, extent) for _ in range(_numel(shape))]
    return lucid.tensor(values).reshape(*shape).to(COMPILE_DEVICE)


def _rotation(shape: Shape, dim: int, extent: int) -> lucid.Tensor:
    """Positions along ``dim`` (an axis of ``extent``) that are distinct in every slice, so an
    overwrite scatter has one answer."""
    dim %= len(shape)
    values = []
    for flat in range(_numel(shape)):
        position = (flat // _numel(shape[dim + 1 :])) % shape[dim]
        values.append((position + 1) % extent)
    return lucid.tensor(values).reshape(*shape).to(COMPILE_DEVICE)


def _under(fn: Callable[..., lucid.Tensor], args: tuple[lucid.Tensor, ...], gate: str) -> object:
    with mock.patch.dict(os.environ, {ENV: gate}):
        _C_engine.compile.session_cache_clear()
        c = lucid.compile(fn)
        got = c(*args)
        info = c.cache_info()
    assert info["entries"] == 1 and not info["eager_only"], info
    return got.to("cpu").tolist()


def _holds(fn: Callable[..., lucid.Tensor], *args: lucid.Tensor) -> None:
    """The merged spelling, the native one and eager give the same integers."""
    want = fn(*args).to("cpu").tolist()
    assert _under(fn, args, "0") == want, "merged vs eager"
    assert _under(fn, args, "-1") == want, "native vs eager"


# ── values ────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Axis:
    shape: Shape
    dim: int
    #: the index's length along ``dim`` as ``extent + grow`` (an index may be shorter or longer).
    grow: int = 0


_AXES: dict[str, _Axis] = {
    "last": _Axis((3, 4, 2, 5), -1),
    "last_shorter": _Axis((3, 4, 2, 5), -1, -2),
    "last_longer": _Axis((3, 4, 2, 5), -1, 3),
    "mid": _Axis((3, 4, 2, 5), 1),
    "mid_longer": _Axis((3, 4, 2, 5), 2, 1),
    "first": _Axis((3, 4, 2, 5), 0),
    "first_shorter": _Axis((3, 4, 2, 5), 0, -1),
    "unit_lead": _Axis((1, 4, 3, 5), 1),
    "unit_bands": _Axis((1, 6, 1, 1), 1),
    "rank_5": _Axis((3, 2, 2, 3, 2), 2),
}


def _along_shape(axis: _Axis) -> Shape:
    """``axis.shape`` with the axis ``axis.grow`` longer: what an index into it may be."""
    shape = list(axis.shape)
    shape[axis.dim] += axis.grow
    return tuple(shape)


def _along(axis: _Axis, seed: int) -> lucid.Tensor:
    return _index(_along_shape(axis), axis.shape[axis.dim], seed)


@needs_metal
@pytest.mark.parametrize("dtype", sorted(_DTYPES))
@pytest.mark.parametrize("name", sorted(_AXES))
def test_a_carried_gather_is_whole_at_every_axis(name: str, dtype: str) -> None:
    axis = _AXES[name]
    data = _wide(axis.shape, dtype, 11)
    index = _along(axis, 12)
    _holds(lambda t, i: lucid.gather(t, axis.dim, i) + 0, data, index)


@needs_metal
@pytest.mark.parametrize("dtype", sorted(_DTYPES))
@pytest.mark.parametrize("name", ["last", "mid", "first"])
def test_take_along_dim_and_index_select_are_whole(name: str, dtype: str) -> None:
    axis = _AXES[name]
    data = _wide(axis.shape, dtype, 21)
    index = _along(axis, 22)
    picks = _index((7,), axis.shape[axis.dim], 23)
    _holds(lambda t, i: lucid.take_along_dim(t, i, axis.dim) + 0, data, index)
    _holds(lambda t, p: lucid.index_select(t, axis.dim, p) + 0, data, picks)


@needs_metal
@pytest.mark.parametrize("dtype", sorted(_DTYPES))
def test_a_table_read_by_a_rank_3_index_is_whole(dtype: str) -> None:
    """``embedding`` reads a table by ``graph_gather``: a rank-3 index makes a rank-4 answer."""
    table = _wide((6, 3), dtype, 31)
    rows = lucid.tensor([[[k % 6 for k in range(j, j + 2)] for j in range(4)]] * 3).to(
        COMPILE_DEVICE
    )
    _holds(lambda w, r: F.embedding(r, w) + 0, table, rows)


@needs_metal
@pytest.mark.parametrize("dtype", sorted(_DTYPES))
@pytest.mark.parametrize("dim", [-1, 1, 2])
def test_sorted_values_are_whole(dim: int, dtype: str) -> None:
    data = _wide((3, 4, 2, 5), dtype, 41)
    _holds(lambda t: lucid.sort(t, dim=dim)[0] + 0, data)
    _holds(lambda t: lucid.topk(t, 2, dim=dim)[0] + 0, data)


@needs_metal
@pytest.mark.parametrize("dtype", sorted(_DTYPES))
@pytest.mark.parametrize("rows", [(), (3, 4, 2)], ids=["one_sequence", "row_per_value"])
def test_a_search_of_rank_4_values_is_whole(rows: Shape, dtype: str) -> None:
    """``searchsorted`` reads its sequence by ``graph_gather`` (one sequence) or
    ``graph_gather_along_axis`` (one per row) at the values' shape, here rank 4."""
    sequence = lucid.sort(_wide((*rows, 16), dtype, 91), dim=-1)[0]
    values = _wide((3, 4, 2, 5), dtype, 92)
    _holds(lambda s, v: lucid.searchsorted(s, v), sequence, values)


@needs_metal
@pytest.mark.parametrize("where", ["last", "mid", "first", "last_shorter", "first_shorter"])
def test_an_int64_overwrite_scatter_is_whole(where: str) -> None:
    """Which update wins a slot is chosen once and read back whole (``scatter_set_whole``); the
    read-back is a carried gather over the update's shape."""
    axis = _AXES[where]
    written = _along_shape(axis)
    data, source = _wide(axis.shape, "int64", 51), _wide(written, "int64", 52)
    position = _rotation(written, axis.dim, axis.shape[axis.dim])
    _holds(lambda t, i, s: t.scatter(axis.dim, i, s) + 0, data, position, source)


@needs_metal
@pytest.mark.parametrize("dtype", sorted(_DTYPES))
def test_a_symbolic_batch_keeps_the_native_spelling(dtype: str) -> None:
    """A dim the batch makes unknown cannot be merged: the helper steps aside, the result holds."""
    axis = _AXES["last"]

    def fn(t: lucid.Tensor, i: lucid.Tensor) -> lucid.Tensor:
        return lucid.gather(t, -1, i) + 0

    with mock.patch.dict(os.environ, {ENV: "0"}):
        _C_engine.compile.session_cache_clear()
        compiled = lucid.compile(fn, dynamic=True)
        for batch in (3, 5, 3):
            shape = (batch, *axis.shape[1:])
            data, index = _wide(shape, dtype, 60 + batch), _index(shape, shape[-1], 70 + batch)
            assert compiled(data, index).to("cpu").tolist() == fn(data, index).to("cpu").tolist()


# ── time ──────────────────────────────────────────────────────────────────────────────

#: 1M elements at rank 4 and the same elements at rank 3 with the axes around the gathered one
#: merged, which is what the helper reads them as.  What the unfixed graph pays is the slice loop
#: over the lead (2048 slices), not the elements, so the inner dims are small: at this size every
#: unfixed graph is 4x to 8x its twin and the file stays well under 1 GB.
_W4: Shape = (2048, 8, 8, 8)
_TWINS: dict[str, tuple[int, Shape]] = {
    "last": (-1, (16384, 8, 8)),
    "mid": (1, (2048, 8, 64)),
    "first": (0, (2048, 512)),
}

_BOUND = 1.5
#: The unfixed graph is expected to miss the bound, and ``compare_ms`` only stops early on a hit,
#: so every attempt of an "exceeds the bound" guard runs: one is enough for a 4x miss.
_UNFIXED_ATTEMPTS = 1


def _gather(dim: int) -> Callable[..., lucid.Tensor]:
    return lambda t, i: lucid.gather(t, dim, i).reshape(-1).sum()


def _sorted(dim: int) -> Callable[..., lucid.Tensor]:
    return lambda t: lucid.sort(t, dim=dim)[0].reshape(-1).sum()


def _scattered(dim: int) -> Callable[..., lucid.Tensor]:
    return lambda t, i, s: t.scatter(dim, i, s).reshape(-1).sum()


@dataclass(frozen=True)
class _Case:
    make: Callable[[int], Callable[..., lucid.Tensor]]
    dtype: str
    wheres: tuple[str, ...]
    #: how many operands the function takes: the data, an index, an update.
    operands: int


#: ``sort`` along axis 0 runs eagerly (the emitter declines it), so it is timed on two axes.
_CASES: dict[str, _Case] = {
    "gather_int32": _Case(_gather, "int32", ("last", "mid", "first"), 2),
    "gather_int64": _Case(_gather, "int64", ("last", "mid", "first"), 2),
    "sort_int32": _Case(_sorted, "int32", ("last", "mid"), 1),
    "scatter_int64": _Case(_scattered, "int64", ("last", "mid", "first"), 3),
}

_ROWS = [(name, where) for name, case in sorted(_CASES.items()) for where in case.wheres]


def _ramp(shape: Shape, dtype: str, modulo: int) -> lucid.Tensor:
    """``0 .. modulo - 1`` repeating, made on the device (a timed operand has 1M elements)."""
    ramp = (lucid.arange(_numel(shape)) % modulo).reshape(*shape)
    return ramp.to(_DTYPES[dtype]).to(COMPILE_DEVICE)


def _operands(name: str, shape: Shape, dim: int) -> tuple[lucid.Tensor, ...]:
    case = _CASES[name]
    out = [_ramp(shape, case.dtype, 1000)]
    if case.operands > 1:
        out.append(_ramp(shape, "int64", shape[dim]))
    if case.operands > 2:
        out.append(_ramp(shape, case.dtype, 7))
    return tuple(out)


def _timed(name: str, where: str, gate: str | None, shape: Shape) -> Callable[[], float]:
    dim = _TWINS[where][0]
    call = compiled_call(_CASES[name].make(dim), *_operands(name, shape, dim))
    return with_env(call, ENV, gate)


@needs_metal
@pytest.mark.parametrize(("name", "where"), _ROWS)
def test_a_rank_4_carried_op_is_not_slower_than_its_rank_3_twin(name: str, where: str) -> None:
    """Held to 1.5x the twin; unfixed the rank-5 carried operand ran 4x to 8x it."""
    twin_shape = _TWINS[where][1]

    def build() -> dict[str, Callable[[], float]]:
        _C_engine.compile.session_cache_clear()
        return {
            "rank-4": _timed(name, where, None, _W4),
            "twin": _timed(name, where, None, twin_shape),
        }

    assert_ratio_within(build, "rank-4", "twin", bound=_BOUND, slack_ms=0.3, what=f"{name} {where}")


@needs_metal
@pytest.mark.parametrize(("name", "where"), _ROWS)
def test_the_carried_op_bound_is_exceeded_by_the_unfixed_graph(name: str, where: str) -> None:
    """The guard above can fail: with the gate off the rank-4 op is over its bound."""
    twin_shape = _TWINS[where][1]

    def build() -> dict[str, Callable[[], float]]:
        _C_engine.compile.session_cache_clear()
        return {
            "gate off": _timed(name, where, "-1", _W4),
            "twin": _timed(name, where, None, twin_shape),
        }

    unfixed = compare_ms(
        build, "gate off", "twin", bound=_BOUND, slack_ms=0.3, attempts=_UNFIXED_ATTEMPTS
    )
    assert not unfixed.within, (
        f"{name} {where}: the unfixed graph meets the guard's bound, so the guard cannot fail: "
        f"{unfixed.report}"
    )
