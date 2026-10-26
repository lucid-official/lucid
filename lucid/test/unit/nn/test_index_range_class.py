"""An integer index outside the axis it indexes is refused before any op
reads with it, and all such indices follow one rule (LCD-234).

Every ``nn.functional`` op that reads with a class or table index checks it
through ``lucid/nn/functional/_index_checks.py``. That covers a class-index
loss's targets, a ``one_hot`` class, an embedding row, ``max_unpool{1,2,3}d``'s
flat spatial indices, and ``ctc_loss``'s labels, ``blank`` and lengths. The ops
used to follow different policies:

- the class losses checked after their gather;
- the embedding table was read on the host on every device;
- ``max_unpool{1,2,3}d`` let ``scatter_add`` wrap negative indices into the
  output and caught positive ones only inside the CPU kernel;
- ``one_hot`` and ``ctc_loss`` did not check at all.

As a result, ``one_hot([0, 7], 3)`` gave a zero row for the 7, and
``ctc_loss`` with an ``input_lengths`` of 50 against 5 frames read past
the input and returned -0.379.

The rule this file holds every consumer to, in the table ``ROWS``:

- **CPU**: ``IndexError`` before the consumer's kernel runs. A refusal
  dispatches only the guard's reads of the index (``GUARD_READS``) and
  waits on no device.
- **Metal**: the index is not read back. The class losses' sample is NaN
  (``POISON``), ``one_hot`` gives a zero row (``ZERO_ROW``),
  ``max_unpool{1,2,3}d`` drops the out-of-range update (``DROP``), and
  ``embedding`` / ``embedding_bag`` give a NaN row or bag (``NAN_ROW``, or
  a zero row for an integer ``embedding`` table), with 0 host syncs on the
  table indices. ``ctc_loss`` and ``embedding_bag``'s 1-D ``offsets`` read
  on the host anyway (``READ``), because their kernels or backwards walk
  those boundaries on the CPU.

Under ``-m parity`` the reference is checked to refuse the same rows,
except for the two places where Lucid is deliberately stricter (marked
``stricter``).
"""

import ast
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._C import engine as _C_engine
from lucid.nn.functional import _index_checks
from lucid.nn.functional.sparse import check_embedding_indices
from lucid.test._fixtures.devices import metal_available

_needs_metal = pytest.mark.skipif(not metal_available(), reason="needs Metal")

N, C, T = 2, 4, 5

#: The only ops a refusal may dispatch, all of them reads of the index
#: itself:
#:
#: - the clamp and the comparison against it, and ``all`` over the result;
#: - the masks of the entries a consumer does not read as an index (an
#:   ``ignore_index`` sentinel, the pad after a label list, the padding of
#:   a ``ctc_loss`` row);
#: - the views and casts that shape them.
#:
#: Anything else is the consumer's own work, and must not have started.
GUARD_READS = frozenset(
    {
        "all",
        "arange",
        "astype",
        "bitwise_or",
        "broadcast_to",
        "clip",
        "cumprod",
        "equal",
        "full",
        "greater_equal",
        "invert",
        "less",
        "not_equal",
        "reshape",
        "unsqueeze",
    }
)

#: Metal policies.
POISON = "poison"  # the bad sample's loss is NaN, nothing read back
ZERO_ROW = "zero-row"  # one_hot's row is all zeros, nothing read back
NAN_ROW = "nan-row"  # embedding / embedding_bag's row or bag is NaN, index not read
DROP = "drop"  # max_unpool drops the out-of-range update, index not read
READ = "read"  # read on the host and refused, as on the CPU
REFUSE = "refuse"  # a bad parameter, refused on both devices, nothing read

Kwargs = dict[str, object]


def _scores(device: str) -> lucid.Tensor:
    return lucid.tensor([[0.1, 0.2, 0.4, 0.8], [0.9, -0.3, 0.5, 0.0]], device=device)


def _log_probs(device: str) -> lucid.Tensor:
    lucid.manual_seed(0)
    return F.log_softmax(lucid.randn(T, N, C), dim=2).to(device)


def _table(device: str) -> lucid.Tensor:
    return lucid.arange(12, dtype=lucid.float32).reshape(C, 3).to(device)


def _ids(values: object, device: str) -> lucid.Tensor:
    return lucid.tensor(values, dtype=lucid.int64, device=device)


@dataclass(frozen=True)
class Row:
    """One out-of-range index given to one consumer."""

    id: str
    consumer: str
    call: Callable[[str], Kwargs]
    match: str
    metal: str
    #: The consumer's own kernels; none of them may run in a refusal.
    kernels: frozenset[str]
    #: For POISON / ZERO_ROW / DROP: the sample or row the bad index is in.
    bad: int = 1
    #: Lucid refuses what the reference computes; documented in the owner.
    stricter: bool = False
    #: Extra kwargs that only the per-sample (Metal) check passes.
    per_sample: Kwargs = field(default_factory=dict)
    #: The refusal: an index outside its axis, or a malformed parameter.
    exc: type[Exception] = IndexError


def _class_rows(name: str, kernels: frozenset[str]) -> list[Row]:
    rows = []
    for bad in (C, 9, -1):
        rows.append(
            Row(
                id=f"{name}-target{bad}",
                consumer=name,
                call=lambda d, b=bad: {"input": _scores(d), "target": _ids([1, b], d)},
                match=f"{name}: Target {bad} is out of bounds",
                metal=POISON,
                kernels=kernels,
                per_sample={"reduction": "none"},
            )
        )
    return rows


def _ctc_row(id: str, match: str, *, stricter: bool = False, **override: object) -> Row:
    """One ``ctc_loss`` row: the valid call ``(T, N, C) = (5, 2, 4)`` with
    padded targets ``[[1, 2], [2, 3]]``, with ``override`` swapped in."""
    args: Kwargs = {
        "targets": [[1, 2], [2, 3]],
        "input_lengths": [T, T],
        "target_lengths": [2, 2],
        **override,
    }

    def call(d: str) -> Kwargs:
        return {**args, "log_probs": _log_probs(d), "targets": _ids(args["targets"], d)}

    return Row(
        id=id,
        consumer="ctc_loss",
        call=call,
        match=match,
        metal=READ,
        kernels=frozenset({"gather", "maximum", "div"}),
        stricter=stricter,
    )


def _unpool_rows(n: int) -> list[Row]:
    name = f"max_unpool{n}d"
    shape = (1, 1) + (2,) * n
    count = 2**n
    extent = 4**n
    kernels = frozenset({"scatter_add", "div"})
    rows = [
        Row(
            id=f"{name}-index{bad}",
            consumer=name,
            call=lambda d, b=bad: {
                "input": lucid.arange(
                    1, count + 1, dtype=lucid.float32, device=d
                ).reshape(*shape),
                "indices": _ids([*range(count - 1), b], d).reshape(*shape),
                "kernel_size": 2,
            },
            match=rf"{name}: index {bad} is out of range for output_size .* \({extent} elements\)",
            metal=DROP,
            kernels=kernels,
            bad=count - 1,
        )
        for bad in (extent, 999, -1, -5)
    ]
    rows.append(
        Row(
            id=f"{name}-output_size-index{3**n}",
            consumer=name,
            call=lambda d: {
                "input": lucid.arange(
                    1, count + 1, dtype=lucid.float32, device=d
                ).reshape(*shape),
                "indices": _ids([*range(count - 1), 3**n], d).reshape(*shape),
                "kernel_size": 2,
                "output_size": (3,) * n,
            },
            match=rf"{name}: index {3**n} is out of range for output_size .* \({3**n} elements\)",
            metal=DROP,
            kernels=kernels,
            bad=count - 1,
        )
    )
    return rows


ROWS: list[Row] = [
    *_class_rows("cross_entropy", frozenset({"log_softmax", "gather"})),
    *_class_rows("nll_loss", frozenset({"gather"})),
    *_class_rows("multi_margin_loss", frozenset({"gather", "relu"})),
    Row(
        id="cross_entropy-kd-target",
        consumer="cross_entropy",
        call=lambda d: {
            "input": _scores(d).reshape(N, C, 1),
            "target": _ids([[1], [7]], d),
        },
        match="cross_entropy: Target 7 is out of bounds",
        metal=POISON,
        kernels=frozenset({"log_softmax", "gather"}),
        per_sample={"reduction": "none"},
    ),
    Row(
        id="multilabel_margin_loss-listed",
        consumer="multilabel_margin_loss",
        call=lambda d: {
            "input": _scores(d),
            "target": _ids([[1, 0, -1, 0], [2, 9, -1, 7]], d),
        },
        match="multilabel_margin_loss: Target 9 is out of bounds",
        metal=POISON,
        kernels=frozenset({"scatter_add", "relu"}),
        per_sample={"reduction": "none"},
    ),
    *_unpool_rows(1),
    *_unpool_rows(2),
    *_unpool_rows(3),
    *[
        Row(
            id=f"one_hot-class{bad}",
            consumer="one_hot",
            call=lambda d, b=bad: {"tensor": _ids([0, b], d), "num_classes": 3},
            match=f"one_hot: class {bad} is out of range for num_classes=3",
            metal=ZERO_ROW,
            kernels=frozenset({"one_hot"}),
        )
        for bad in (3, 7, -1)
    ],
    *[
        Row(
            id=f"embedding-index{bad}",
            consumer="embedding",
            call=lambda d, b=bad: {"input": _ids([[0, b]], d), "weight": _table(d)},
            match=f"embedding: index {bad} is out of range for a table with 4",
            metal=NAN_ROW,
            kernels=frozenset({"embedding"}),
        )
        for bad in (C, -1, 10**6)
    ],
    *[
        Row(
            id=f"embedding_bag-{form}-index{bad}",
            consumer="embedding_bag",
            call=(
                (
                    lambda d, b=bad: {
                        "input": _ids([[0, 1], [0, b]], d),
                        "weight": _table(d),
                    }
                )
                if form == "rows"
                else (
                    lambda d, b=bad: {
                        "input": _ids([0, 1, b], d),
                        "weight": _table(d),
                        "offsets": _ids([0, 2], d),
                    }
                )
            ),
            match=f"embedding_bag: index {bad} is out of range for a table with 4",
            metal=NAN_ROW,
            kernels=frozenset({"embedding_bag", "embedding"}),
        )
        for bad in (C, -1)
        for form in ("rows", "offsets")
    ],
    # ctc_loss: (T, N, C) = (5, 2, 4).
    *[
        _ctc_row(
            f"ctc_loss-{form}-target{bad}",
            f"ctc_loss: target {bad} is out of range for 4 classes",
            stricter=True,
            targets=[[1, bad], [2, 3]] if form == "padded" else [1, bad, 2, 3],
        )
        for bad in (C, 9, -1)
        for form in ("padded", "concatenated")
    ],
    *[
        _ctc_row(
            f"ctc_loss-blank{bad}",
            f"ctc_loss: blank {bad} is out of range for 4 classes",
            blank=bad,
        )
        for bad in (C, 10, -1)
    ],
    *[
        _ctc_row(
            f"ctc_loss-input_lengths{bad}",
            rf"ctc_loss: each of input_lengths must lie in \[0, {T}\], got {bad}",
            input_lengths=[bad, T],
        )
        for bad in (T + 1, 50, -1)
    ],
    *[
        _ctc_row(
            f"ctc_loss-target_lengths{bad}",
            rf"ctc_loss: each of target_lengths must lie in \[0, 2\], got {bad}",
            target_lengths=[bad, 2],
        )
        for bad in (3, -1)
    ],
    _ctc_row(
        "ctc_loss-concatenated-target_lengths-1",
        "ctc_loss: each of target_lengths must be non-negative, got -1",
        targets=[1, 2, 2, 3],
        target_lengths=[-1, 5],
    ),
    # embedding_bag's bag boundaries index into its indices.
    *[
        Row(
            id=f"embedding_bag-offsets{off}{'-last' if last else ''}",
            consumer="embedding_bag",
            call=lambda d, o=off, la=last: {
                "input": _ids([0, 1, 2], d),
                "weight": _table(d),
                "offsets": _ids(o, d),
                "include_last_offset": la,
            },
            match=match,
            metal=READ,
            kernels=frozenset({"embedding_bag", "embedding"}),
            exc=exc,
        )
        for off, last, exc, match in (
            ([0, 5], False, IndexError, "offset 5 is out of range for 3 indices"),
            ([0, -1], False, IndexError, "offset -1 is out of range for 3"),
            ([0, 2, 4], True, IndexError, "offset 4 is out of range for 3"),
            ([0, 3, 1], False, ValueError, "offsets must not decrease"),
            ([1, 2], False, ValueError, r"offsets\[0\] must be 0"),
        )
    ],
    # Parameters naming a row or a class count: refused before anything is
    # read, on both devices.
    *[
        Row(
            id=f"{name}-padding_idx{bad}",
            consumer=name,
            call=lambda d, b=bad: {
                "input": _ids([[0, 1]], d),
                "weight": _table(d),
                "padding_idx": b,
            },
            match=rf"{name}: padding_idx must be within \[-4, 4\); got {bad}",
            metal=REFUSE,
            kernels=frozenset({"embedding_bag", "embedding"}),
            exc=ValueError,
        )
        for name in ("embedding", "embedding_bag")
        for bad in (C, -C - 1)
    ],
    Row(
        id="one_hot-num_classes0",
        consumer="one_hot",
        call=lambda d: {"tensor": _ids([0], d), "num_classes": 0},
        match="one_hot: num_classes must be positive",
        metal=REFUSE,
        kernels=frozenset({"one_hot"}),
        exc=ValueError,
    ),
]

#: Every op that reads with a class or table index, and so has a row.
CONSUMERS = frozenset(row.consumer for row in ROWS)


def _ids_of(rows: list[Row]) -> list[str]:
    return [row.id for row in rows]


def test_every_row_has_a_unique_id() -> None:
    assert len(set(_ids_of(ROWS))) == len(ROWS)


# ── CPU: refused before the consumer's kernel ──────────────────────────────


def _invoke(
    fn: Callable[..., lucid.Tensor], kwargs: Kwargs, **extra: object
) -> lucid.Tensor:
    kw = {**kwargs, **extra}
    args = [kw.pop("tensor")] if "tensor" in kw else []
    return fn(*args, **kw)


@pytest.mark.parametrize("row", ROWS, ids=_ids_of(ROWS))
def test_the_cpu_refuses_before_any_kernel(row: Row) -> None:
    kwargs = row.call("cpu")
    fn = getattr(F, row.consumer)
    syncs = _C_engine.host_sync_count()
    with lucid.profiler.profile() as prof:
        with pytest.raises(row.exc, match=row.match):
            _invoke(fn, kwargs)
    dispatched = {event.name for event in prof.events()}
    assert dispatched <= GUARD_READS, f"not a guard read: {dispatched - GUARD_READS}"
    assert not dispatched & row.kernels
    assert _C_engine.host_sync_count() == syncs


@pytest.mark.parametrize(
    ("consumer", "wide"),
    [
        ("cross_entropy", 2**40),
        ("nll_loss", 2**40),
        ("multi_margin_loss", 2**40),
        # 2**31 is -2**31 in int32: a mask read there ended the label list.
        ("multilabel_margin_loss", 2**31),
    ],
)
def test_a_cpu_int64_target_past_int32_is_refused_not_wrapped(
    consumer: str, wide: int
) -> None:
    # 2**40 cast to int32 is 0, and used to be scored as class 0.  A CPU
    # target is checked at its own width.  (Metal checks at the int32 its
    # gather reads with; the owner's docstring says why.)
    x = _scores("cpu")
    values = (
        [[wide, 0, -1, 0], [0, -1, 0, 0]]
        if consumer.startswith("multil")
        else [1, wide]
    )
    target = _ids(values, "cpu")
    with lucid.profiler.profile() as prof:
        with pytest.raises(IndexError, match=f"Target {wide} is out of bounds"):
            getattr(F, consumer)(x, target)
    assert {event.name for event in prof.events()} <= GUARD_READS


# ── Metal: one documented policy ───────────────────────────────────────────


def _flat(values: object) -> list[float]:
    if isinstance(values, list):
        return [v for item in values for v in _flat(item)]
    assert isinstance(values, int | float)
    return [float(values)]


@_needs_metal
@pytest.mark.parametrize("row", ROWS, ids=_ids_of(ROWS))
def test_metal_follows_the_documented_policy(row: Row) -> None:
    fn = getattr(F, row.consumer)
    if row.metal in (READ, REFUSE):
        kwargs = row.call("metal")
        syncs = _C_engine.host_sync_count()
        with pytest.raises(row.exc, match=row.match):
            _invoke(fn, kwargs)
        if row.metal == REFUSE:
            assert _C_engine.host_sync_count() == syncs, "read back on Metal"
        return
    kwargs, again = row.call("metal"), row.call("metal")
    # 1-D embedding_bag's offsets are read on the host (_check_offsets);
    # the table indices themselves are never read back.
    expected_syncs = 1 if "offsets" in kwargs else 0
    syncs = _C_engine.host_sync_count()
    out = _invoke(fn, kwargs, **row.per_sample)
    assert _C_engine.host_sync_count() - syncs == expected_syncs, "read back on Metal"
    if row.metal == POISON:
        values = _flat(out.tolist())
        assert math.isnan(values[row.bad])
        assert all(math.isfinite(v) for i, v in enumerate(values) if i != row.bad)
        syncs = _C_engine.host_sync_count()
        mean = _invoke(fn, again)
        assert _C_engine.host_sync_count() == syncs, "read back on Metal"
        assert math.isnan(float(mean.item()))
    elif row.metal == NAN_ROW:
        rows = out.reshape(-1, 3).tolist()
        assert isinstance(rows, list)
        assert all(isinstance(v, float) and math.isnan(v) for v in rows[row.bad])
        assert all(
            isinstance(v, float) and math.isfinite(v)
            for i, r in enumerate(rows)
            if i != row.bad
            for v in r
        )
    elif row.metal == DROP:
        values = _flat(out.tolist())
        assert values[: row.bad] == [float(i) for i in range(1, row.bad + 1)]
        assert all(v == 0.0 for v in values[row.bad :])
    else:
        rows = out.tolist()
        assert isinstance(rows, list)
        assert rows[row.bad] == [0, 0, 0]
        assert rows[0] == [1, 0, 0]


@_needs_metal
def test_metal_embedding_integer_table_reads_zero_out_of_range() -> None:
    table = (lucid.arange(12, dtype=lucid.int32).reshape(C, 3) + 10).to("metal")
    ids = _ids([[0, C]], "metal")
    syncs = _C_engine.host_sync_count()
    out = F.embedding(ids, table)
    assert _C_engine.host_sync_count() == syncs
    assert out.tolist() == [[[10, 11, 12], [0, 0, 0]]]


# ── what is not refused ────────────────────────────────────────────────────


def test_the_ignore_index_and_the_pad_are_not_indices(device: str) -> None:
    x = _scores(device)
    for fn in (F.cross_entropy, F.nll_loss):
        assert math.isfinite(float(fn(x, _ids([1, -100], device)).item()))
        assert math.isfinite(float(fn(x, _ids([1, 7], device), ignore_index=7).item()))
    # After a label list's first negative entry, nothing is read.
    t = _ids([[1, -1, 9, -7], [2, 3, -1, 99]], device)
    assert math.isfinite(float(F.multilabel_margin_loss(x, t).item()))
    # Past a ctc row's target length, nothing is read either.
    out = F.ctc_loss(
        _log_probs(device), _ids([[1, 99], [2, 3]], device), [T, T], [1, 2]
    )
    assert math.isfinite(float(out.item()))


def test_the_ends_of_every_axis_are_taken(device: str) -> None:
    x = _scores(device)
    for fn in (F.cross_entropy, F.nll_loss, F.multi_margin_loss):
        assert math.isfinite(float(fn(x, _ids([0, C - 1], device)).item()))
    assert F.one_hot(_ids([0, 2], device), 3).tolist() == [[1, 0, 0], [0, 0, 1]]
    assert F.embedding(_ids([0, C - 1], device), _table(device)).shape == (2, 3)
    assert F.embedding_bag(_ids([[0, C - 1]], device), _table(device)).shape == (
        1,
        3,
    )
    for n in (1, 2, 3):
        shape = (1, 1) + (2,) * n
        count, extent = 2**n, 4**n
        vals = lucid.arange(1, count + 1, dtype=lucid.float32, device=device).reshape(
            *shape
        )
        ends = _ids([0, *range(1, count - 1), extent - 1], device).reshape(*shape)
        unpooled = _flat(getattr(F, f"max_unpool{n}d")(vals, ends, 2).tolist())
        assert unpooled[0] == 1.0 and unpooled[-1] == float(count)
    full = F.ctc_loss(
        _log_probs(device),
        _ids([[1, C - 1], [2, 3]], device),
        [T, T],
        [2, 2],
        blank=0,
    )
    assert math.isfinite(float(full.item()))
    last_blank = F.ctc_loss(
        _log_probs(device), _ids([[1, 2], [2, 0]], device), [T, 3], [2, 2], blank=3
    )
    assert math.isfinite(float(last_blank.item()))


def test_an_empty_index_has_nothing_to_check(device: str) -> None:
    empty = lucid.zeros(0, dtype=lucid.int64, device=device)
    assert F.one_hot(empty, 3).shape == (0, 3)
    assert F.embedding(empty, _table(device)).shape == (0, 3)
    for n in (1, 2, 3):
        shape = (0, 1) + (2,) * n
        vals = lucid.zeros(*shape, dtype=lucid.float32, device=device)
        idx = lucid.zeros(*shape, dtype=lucid.int64, device=device)
        assert getattr(F, f"max_unpool{n}d")(vals, idx, 2).shape == (0, 1) + (4,) * n


@pytest.mark.parametrize("n", [1, 2, 3])
def test_max_unpool_modules_follow_the_same_rule(device: str, n: int) -> None:
    cls = getattr(nn, f"MaxUnpool{n}d")
    mod = cls(kernel_size=2)
    shape = (1, 1) + (2,) * n
    count, extent = 2**n, 4**n
    vals = lucid.arange(1, count + 1, dtype=lucid.float32, device=device).reshape(
        *shape
    )
    for bad in (-5, extent):
        idx = _ids([*range(count - 1), bad], device).reshape(*shape)
        if device == "cpu":
            with pytest.raises(
                IndexError, match=rf"max_unpool{n}d: index {bad} is out of range"
            ):
                mod(vals, idx)
        else:
            syncs = _C_engine.host_sync_count()
            out = mod(vals, idx)
            assert _C_engine.host_sync_count() == syncs
            flat_out = _flat(out.tolist())
            assert flat_out[: count - 1] == [float(i) for i in range(1, count)]
            assert all(v == 0.0 for v in flat_out[count - 1 :])


def test_check_embedding_indices_is_the_same_rule(device: str) -> None:
    # Kept under its released name; it is the owner's table check.
    table = _table(device)
    check_embedding_indices(_ids([0, 3], device), table, "lookup")
    if device == "cpu":
        with pytest.raises(IndexError, match="lookup: index 4 is out of range"):
            check_embedding_indices(_ids([4], device), table, "lookup")
    else:
        bad = _ids([4], device)
        syncs = _C_engine.host_sync_count()
        check_embedding_indices(bad, table, "lookup")
        assert _C_engine.host_sync_count() == syncs
    with pytest.raises(TypeError, match="lookup: indices must be an integer"):
        check_embedding_indices(lucid.tensor([0.5], device=device), table, "lookup")


# ── ctc_loss: shapes the kernel would read past ────────────────────────────


@pytest.mark.parametrize(
    ("targets", "input_lengths", "target_lengths", "match"),
    [
        ([[1, 2], [2, 3]], [T], [2, 2], "input_lengths must hold one length"),
        ([[1, 2], [2, 3]], [T, T], [2], "target_lengths must hold one length"),
        ([[1, 2]], [T, T], [2, 2], "padded targets must hold one row"),
        ([1, 2, 2, 3], [T, T], [2, 3], "concatenated targets hold 4 labels"),
        ([[[1, 2]], [[2, 3]]], [T, T], [2, 2], "targets must be padded"),
    ],
    ids=["input-count", "target-count", "rows", "concat-sum", "rank-3"],
)
def test_ctc_refuses_a_shape_it_would_read_past(
    device: str,
    targets: object,
    input_lengths: list[int],
    target_lengths: list[int],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        F.ctc_loss(
            _log_probs(device), _ids(targets, device), input_lengths, target_lengths
        )


def test_ctc_accepts_an_empty_input_sequence(device: str) -> None:
    losses = _flat(
        F.ctc_loss(
            _log_probs(device),
            _ids([[1, 2], [2, 3]], device),
            [0, T],
            [0, 2],
            reduction="none",
        ).tolist()
    )
    assert losses[0] == 0.0 and math.isfinite(losses[1])


# ── one owner ──────────────────────────────────────────────────────────────


_FUNCTIONAL = Path(F.__file__).parent
_OWNER = Path(_index_checks.__file__).name


def _calls_the_owner() -> set[str]:
    """Every public ``nn.functional`` function that calls into the owner."""
    owner_names = {
        name for name in vars(_index_checks) if name.startswith("_check")
    } | {"_class_targets", "check_embedding_indices"}
    found: set[str] = set()
    for path in sorted(_FUNCTIONAL.glob("*.py")):
        if path.name == _OWNER:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        callees_by_fn: dict[str, set[str]] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            callees_by_fn[node.name] = {
                call.func.id
                for call in ast.walk(node)
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
            }
        reaches_owner = set(owner_names)
        changed = True
        while changed:
            changed = False
            for fn_name, callees in callees_by_fn.items():
                if (
                    fn_name.startswith("_")
                    and fn_name not in reaches_owner
                    and callees & reaches_owner
                ):
                    reaches_owner.add(fn_name)
                    changed = True
        found.update(
            fn_name
            for fn_name, callees in callees_by_fn.items()
            if not fn_name.startswith("_") and callees & reaches_owner
        )
    return found


def test_every_consumer_calls_the_owner_and_has_rows() -> None:
    # A new op that reads with an index and calls the owner gets rows in
    # the table; one of the table's ops that stopped calling it is caught.
    assert _calls_the_owner() - {"check_embedding_indices"} == set(CONSUMERS)


def test_only_the_owner_raises_index_error_in_nn_functional() -> None:
    # One rule, one place: an IndexError raised by an op of its own is
    # another policy starting.
    for path in sorted(_FUNCTIONAL.glob("*.py")):
        if path.name == _OWNER:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Raise) and node.exc is not None:
                exc = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
                assert not (
                    isinstance(exc, ast.Name) and exc.id == "IndexError"
                ), f"{path.name}:{node.lineno} raises IndexError itself"


# ── the reference ──────────────────────────────────────────────────────────


def _ref_kwargs(R: ModuleType, row: Row) -> Kwargs:
    out: Kwargs = {}
    for key, value in row.call("cpu").items():
        if isinstance(value, lucid.Tensor):
            out[key] = R.tensor(value.tolist())
        else:
            out[key] = value
    return out


@pytest.mark.parity
@pytest.mark.parametrize("row", ROWS, ids=_ids_of(ROWS))
def test_the_reference_refuses_the_same_rows(ref: ModuleType, row: Row) -> None:
    R = ref
    kwargs = _ref_kwargs(R, row)
    args: list[object] = []
    if row.consumer in ("embedding", "embedding_bag"):
        args = [kwargs.pop("input"), kwargs.pop("weight")]
    elif "input" in kwargs:
        args = [kwargs.pop("input")]
    elif "tensor" in kwargs:
        args = [kwargs.pop("tensor")]
    fn = getattr(R.nn.functional, row.consumer)
    if row.stricter:
        # The reference reads past the class axis and answers; Lucid
        # refuses, as the owner documents.
        fn(*args, **kwargs)
        return
    with pytest.raises(Exception):  # noqa: B017 - its types vary by op
        fn(*args, **kwargs)


@pytest.mark.parity
@pytest.mark.parametrize("blank", [0, C - 1])
def test_ctc_at_the_ends_of_its_axes_matches_the_reference(
    ref: ModuleType, blank: int
) -> None:
    R = ref
    lp = _log_probs("cpu")
    targets = [[1, 2], [2, 1]] if blank == 0 else [[0, 2], [2, 1]]
    for il, tl in (([T, T], [2, 2]), ([T, 2], [2, 0]), ([3, T], [2, 1])):
        got = F.ctc_loss(
            lp, _ids(targets, "cpu"), il, tl, blank=blank, reduction="none"
        )
        want = R.nn.functional.ctc_loss(
            R.tensor(lp.tolist()),
            R.tensor(targets),
            il,
            tl,
            blank=blank,
            reduction="none",
        )
        for g, w in zip(_flat(got.tolist()), _flat(want.tolist()), strict=True):
            assert math.isclose(g, w, rel_tol=1e-4, abs_tol=1e-5) or (
                math.isinf(g) and math.isinf(w)
            )


@pytest.mark.parity
def test_ctc_takes_an_empty_input_sequence_as_the_reference_does(
    ref: ModuleType,
) -> None:
    R = ref
    lp = _log_probs("cpu")
    for zero_infinity in (False, True):
        got = F.ctc_loss(
            lp,
            _ids([[1, 2], [2, 3]], "cpu"),
            [0, 0],
            [0, 1],
            reduction="none",
            zero_infinity=zero_infinity,
        )
        want = R.nn.functional.ctc_loss(
            R.tensor(lp.tolist()),
            R.tensor([[1, 2], [2, 3]]),
            [0, 0],
            [0, 1],
            reduction="none",
            zero_infinity=zero_infinity,
        )
        assert got.tolist() == want.tolist()
