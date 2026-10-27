"""A bad index argument is refused at the public op, as the reference refuses it (LCD-586).

The ``index_*`` family left its index argument to the engine kernels it
runs on.  A 2-D index for ``index_copy`` or ``index_add`` was flattened
in silence when ``source`` had as many slices as the index had elements.
Otherwise the ``reshape`` that flattened it refused it with
``ShapeMismatch (reshape: …)``.  A float index, or an index or source on
another device, was refused by ``scatter_set`` / ``scatter_add``.  A 0-d
input indexed past its shape (``list index out of range``).  ``put``'s
refusals named ``scatter``.

The owner for the ops that flatten their index before a kernel sees it
is the index check in ``lucid._ops.composite.indexing`` (``_require_index``,
``_positions``, ``_require_source_slices``).  ``index_select`` and
``scatter_reduce`` hand the caller's index to the engine, whose doors own
their rules.

Each table records the class the reference raises, as measured, so the
guard runs without the reference installed.  ``-m parity`` checks the
record against the reference itself.
"""

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import ModuleType

import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid.test._fixtures.devices import metal_available

# A call of one op, given the framework module ``m``, then ``input``,
# ``dim``, ``index`` and ``source`` (unused by the ops without one).
_Call = Callable[[ModuleType, object, int, object, object], object]
# The arguments of a case, built on the CPU by the framework module ``m``.
_Case = Callable[[ModuleType], tuple[object, int, object, object]]

_I, _R = IndexError, RuntimeError
_DEVICES = ("cpu", "metal") if metal_available() else ("cpu",)
_BOTH = frozenset({"cpu", "metal"})
_CPU = frozenset({"cpu"})
_METAL = frozenset({"metal"})


@dataclass(frozen=True)
class _Bad:
    """A refused call: the class the reference raises, per public op."""

    make: _Case
    kinds: Mapping[str, type[Exception]]
    devices: frozenset[str] = _BOTH
    # Arguments left on the CPU while the rest move to the device under test.
    cpu_args: frozenset[int] = frozenset()
    # Ops whose reference accepts the call; Lucid refuses it as ``kinds`` says.
    stricter: frozenset[str] = frozenset()


def _same(kind: type[Exception], ops: frozenset[str] | set[str]) -> dict[str, type[Exception]]:
    return dict.fromkeys(ops, kind)


def _op(spelling: str) -> str:
    """The public op a spelling belongs to: ``Tensor.index_copy_`` → ``index_copy``."""
    return spelling.removeprefix("Tensor.").split("-")[0].removesuffix("_")


def _names(message: str, op: str) -> bool:
    """Whether ``message`` is a refusal by ``op``: ``op: …`` or ``Kind (op): …``."""
    return re.match(rf"(?:\w+ \()?{re.escape(op)}[:)]", message) is not None


def _build(bad: _Bad, m: ModuleType, device: str) -> tuple[object, ...]:
    args = bad.make(m)
    return tuple(
        a if i in bad.cpu_args or not hasattr(a, "to") else a.to(device) for i, a in enumerate(args)
    )


def _cells(table: Mapping[str, _Bad], spellings: Mapping[str, _Call], *lead: str) -> list[object]:
    """One cell per spelling, case and device; ``lead`` goes first in each."""
    cells: list[object] = []
    for case, bad in table.items():
        for spelling in spellings:
            op = _op(spelling)
            if op not in bad.kinds:
                continue
            for device in _DEVICES:
                if device in bad.devices:
                    cells.append(
                        pytest.param(
                            *lead,
                            spelling,
                            case,
                            device,
                            id=f"{spelling}-{case}-{device}",
                        )
                    )
    return cells


def _parity_cells(
    table: Mapping[str, _Bad], spellings: Mapping[str, _Call], *lead: str
) -> list[object]:
    return [
        pytest.param(*lead, spelling, case, id=f"{spelling}-{case}")
        for case, bad in table.items()
        for spelling in spellings
        if _op(spelling) in bad.kinds
    ]


def _builtin(exc: BaseException) -> type[Exception]:
    """The builtin class of a reference exception, most specific first."""
    for kind in (IndexError, TypeError, ValueError, NotImplementedError, RuntimeError):
        if isinstance(exc, kind):
            return kind
    raise AssertionError(f"unexpected reference exception {exc!r}")


def _ref_device(ref: ModuleType, bad: _Bad) -> str:
    if "cpu" in bad.devices:
        return "cpu"
    if not ref.backends.mps.is_available():
        pytest.skip("the reference has no GPU device here")
    return "mps"


# ── the index_* family ──────────────────────────────────────────────────────

_FAMILY: dict[str, _Call] = {
    "index_copy": lambda m, x, d, i, s: m.index_copy(x, d, i, s),
    "Tensor.index_copy": lambda m, x, d, i, s: x.index_copy(d, i, s),
    "Tensor.index_copy_": lambda m, x, d, i, s: x.clone().index_copy_(d, i, s),
    "index_add": lambda m, x, d, i, s: m.index_add(x, d, i, s),
    "Tensor.index_add": lambda m, x, d, i, s: x.index_add(d, i, s),
    "Tensor.index_add_": lambda m, x, d, i, s: x.clone().index_add_(d, i, s),
    "index_fill": lambda m, x, d, i, s: m.index_fill(x, d, i, -1.0),
    "Tensor.index_fill": lambda m, x, d, i, s: x.index_fill(d, i, -1.0),
    "Tensor.index_fill_": lambda m, x, d, i, s: x.clone().index_fill_(d, i, -1.0),
    "index_select": lambda m, x, d, i, s: m.index_select(x, d, i),
    "Tensor.index_select": lambda m, x, d, i, s: x.index_select(d, i),
}

_ALL = frozenset({"index_copy", "index_add", "index_fill", "index_select"})
_SOURCED = frozenset({"index_copy", "index_add"})
# The reference's classes for an index of the wrong rank, and of the wrong dtype.
_RANK = {"index_copy": _I, "index_add": _I, "index_fill": _R, "index_select": _I}
_DTYPE = {"index_copy": _R, "index_add": _R, "index_fill": _I, "index_select": _R}

_BAD: dict[str, _Bad] = {
    # The LCD-586 repro: as many slices as the index has elements.
    "2d-index": _Bad(
        lambda m: (m.zeros(5, 3), 0, m.tensor([[0, 1], [2, 3]]), m.ones(4, 3)),
        _RANK,
    ),
    "2d-index-short-source": _Bad(
        lambda m: (m.zeros(5, 3), 0, m.tensor([[0, 1], [2, 3]]), m.ones(2, 3)),
        _RANK,
    ),
    "empty-2d-index": _Bad(
        lambda m: (m.zeros(5, 3), 0, m.zeros(0, 1, dtype=m.int64), m.ones(0, 3)),
        _RANK,
    ),
    "float-index": _Bad(lambda m: (m.zeros(5, 3), 0, m.tensor([0.0, 1.0]), m.ones(2, 3)), _DTYPE),
    "bool-index": _Bad(lambda m: (m.zeros(5, 3), 0, m.tensor([True, False]), m.ones(2, 3)), _DTYPE),
    "dim-past-the-last": _Bad(
        lambda m: (m.zeros(5, 3), 2, m.tensor([0, 2]), m.ones(2, 3)), _same(_I, _ALL)
    ),
    "dim-before-the-first": _Bad(
        lambda m: (m.zeros(5, 3), -3, m.tensor([0, 2]), m.ones(2, 3)), _same(_I, _ALL)
    ),
    "0d-input-past-dim": _Bad(
        lambda m: (m.tensor(3.0), 1, m.tensor([0]), m.tensor([5.0])), _same(_I, _ALL)
    ),
    "source-length": _Bad(
        lambda m: (m.zeros(5, 3), 0, m.tensor([0, 2]), m.ones(3, 3)),
        {"index_copy": _I, "index_add": _R},
    ),
    "source-other-dims": _Bad(
        lambda m: (m.zeros(5, 3), 0, m.tensor([0, 2]), m.ones(2, 4)),
        _same(_R, _SOURCED),
    ),
    "source-other-dims-dim1": _Bad(
        lambda m: (m.zeros(2, 3), 1, m.tensor([2, 0]), m.ones(3, 2)),
        _same(_R, _SOURCED),
    ),
    "source-more-dims": _Bad(
        lambda m: (m.zeros(5, 3), 0, m.tensor([0, 2]), m.ones(2, 3, 1)),
        {"index_copy": _I, "index_add": _R},
    ),
    "0d-index-1d-source": _Bad(
        lambda m: (m.zeros(5, 3), 0, m.tensor(1), m.ones(3)),
        {"index_copy": _I, "index_add": _R},
    ),
    "0d-source-into-a-matrix": _Bad(
        lambda m: (m.zeros(5, 3), 0, m.tensor(1), m.tensor(7.0)), _same(_R, _SOURCED)
    ),
    # The reference's index_add pairs a 0-d source with a 0-d input only;
    # its index_copy reads a 0-d source as one slice (valid cases below).
    "0d-source-into-a-vector": _Bad(
        lambda m: (m.zeros(5), 0, m.tensor(1), m.tensor(7.0)), {"index_add": _R}
    ),
    "0d-input-1d-source": _Bad(
        lambda m: (m.tensor(3.0), 0, m.tensor([0]), m.tensor([5.0])), {"index_add": _R}
    ),
    "0d-input-two-indices": _Bad(
        lambda m: (m.tensor(3.0), 0, m.tensor([0, 0]), m.tensor(5.0)), {"index_add": _I}
    ),
    # Any index into an axis of size 0 is out of range.  Lucid's contract
    # is that an index outside [-size, size) is an IndexError, and main
    # refused this call before LCD-586 too.  `stricter` keeps index_add
    # refusing it on purpose, so do not "fix" it toward the reference.  The
    # reference has no consistent rule here:
    #   * its CPU index_add skips the bounds check when input.numel() == 0
    #     and writes nothing, so it accepts index 99 just as well;
    #   * it refuses the same index into a 1-D empty input with IndexError;
    #   * its MPS backend fails an INTERNAL ASSERT on it.
    "index-into-an-empty-axis": _Bad(
        lambda m: (m.zeros(0, 3), 0, m.tensor([0]), m.zeros(1, 3)),
        {**_same(_R, {"index_copy", "index_fill", "index_select"}), "index_add": _I},
        stricter=frozenset({"index_add"}),
    ),
    # Index values past the axis: refused on the CPU, dropped on Metal.
    "index-value-past-the-axis": _Bad(
        lambda m: (m.zeros(5, 3), 0, m.tensor([7]), m.ones(1, 3)),
        {"index_copy": _I, "index_add": _R, "index_fill": _I, "index_select": _I},
        devices=_CPU,
    ),
    "index-on-the-cpu": _Bad(
        lambda m: (m.zeros(5, 3), 0, m.tensor([0, 2]), m.ones(2, 3)),
        _same(_R, _ALL),
        devices=_METAL,
        cpu_args=frozenset({2}),
    ),
    "input-on-the-cpu": _Bad(
        lambda m: (m.zeros(5, 3), 0, m.tensor([0, 2]), m.ones(2, 3)),
        _same(_R, _ALL),
        devices=_METAL,
        cpu_args=frozenset({0, 3}),
    ),
    "source-on-the-cpu": _Bad(
        lambda m: (m.zeros(5, 3), 0, m.tensor([0, 2]), m.ones(2, 3)),
        _same(_R, _SOURCED),
        devices=_METAL,
        cpu_args=frozenset({3}),
    ),
}


@pytest.mark.parametrize(("spelling", "case", "device"), _cells(_BAD, _FAMILY))
def test_a_bad_index_argument_raises_the_reference_class(
    spelling: str, case: str, device: str
) -> None:
    bad = _BAD[case]
    with pytest.raises(bad.kinds[_op(spelling)]):
        _FAMILY[spelling](lucid, *_build(bad, lucid, device))


@pytest.mark.parametrize(("spelling", "case", "device"), _cells(_BAD, _FAMILY))
def test_a_bad_index_argument_is_refused_by_the_public_op(
    spelling: str, case: str, device: str
) -> None:
    bad = _BAD[case]
    with pytest.raises((IndexError, TypeError, ValueError, RuntimeError)) as info:
        _FAMILY[spelling](lucid, *_build(bad, lucid, device))
    assert _names(str(info.value), _op(spelling)), str(info.value)


@pytest.mark.parity
@pytest.mark.parametrize(("spelling", "case"), _parity_cells(_BAD, _FAMILY))
def test_the_recorded_family_classes_are_the_reference_s(
    ref: ModuleType, spelling: str, case: str
) -> None:
    bad = _BAD[case]
    op = _op(spelling)
    args = _build(bad, ref, _ref_device(ref, bad))
    if op in bad.stricter:
        _FAMILY[spelling](ref, *args)
        return
    with pytest.raises((IndexError, TypeError, ValueError, RuntimeError)) as theirs:
        _FAMILY[spelling](ref, *args)
    assert _builtin(theirs.value) is bad.kinds[op], theirs.value


@pytest.mark.parametrize("name", ["index_copy_", "index_add_", "index_fill_"])
def test_a_refused_in_place_write_leaves_its_input_alone(device: str, name: str) -> None:
    x = lucid.zeros(5, 3, device=device)
    index = lucid.tensor([[0, 1], [2, 3]], device=device)
    args = (2.0,) if name == "index_fill_" else (lucid.ones(4, 3, device=device),)
    with pytest.raises(_C_engine.IndexError, match=name.removesuffix("_")):
        getattr(x, name)(0, index, *args)
    assert x.tolist() == [[0.0] * 3] * 5


def test_an_index_that_is_not_a_tensor_is_a_type_error() -> None:
    with pytest.raises(TypeError, match="index_copy: index must be a Tensor"):
        lucid.index_copy(lucid.zeros(3), 0, [0, 1], lucid.ones(2))


# ── the valid cases answer as the reference does ───────────────────────────


def _matrix(m: ModuleType) -> object:
    return m.arange(6.0).reshape(3, 2)


def _rows(m: ModuleType) -> object:
    return m.tensor([[10.0, 20.0], [30.0, 40.0]])


@dataclass(frozen=True)
class _Valid:
    """A valid call and the reference's answer, ``(shape, values)``, per public op."""

    make: _Case
    want: Mapping[str, tuple[tuple[int, ...], object]]


_VALID: dict[str, _Valid] = {
    "1d-index": _Valid(
        lambda m: (_matrix(m), 0, m.tensor([2, 0]), _rows(m)),
        {
            "index_copy": ((3, 2), [[30.0, 40.0], [2.0, 3.0], [10.0, 20.0]]),
            "index_add": ((3, 2), [[30.0, 41.0], [2.0, 3.0], [14.0, 25.0]]),
            "index_fill": ((3, 2), [[-1.0, -1.0], [2.0, 3.0], [-1.0, -1.0]]),
            "index_select": ((2, 2), [[4.0, 5.0], [0.0, 1.0]]),
        },
    ),
    # index_copy of a repeated index keeps an unspecified one of its writes.
    "repeated-index": _Valid(
        lambda m: (_matrix(m), 0, m.tensor([1, 1]), _rows(m)),
        {
            "index_add": ((3, 2), [[0.0, 1.0], [42.0, 63.0], [4.0, 5.0]]),
            "index_fill": ((3, 2), [[0.0, 1.0], [-1.0, -1.0], [4.0, 5.0]]),
            "index_select": ((2, 2), [[2.0, 3.0], [2.0, 3.0]]),
        },
    ),
    "negative-dim": _Valid(
        lambda m: (_matrix(m), -1, m.tensor([1]), m.tensor([[10.0], [20.0], [30.0]])),
        {
            "index_copy": ((3, 2), [[0.0, 10.0], [2.0, 20.0], [4.0, 30.0]]),
            "index_add": ((3, 2), [[0.0, 11.0], [2.0, 23.0], [4.0, 35.0]]),
            "index_fill": ((3, 2), [[0.0, -1.0], [2.0, -1.0], [4.0, -1.0]]),
            "index_select": ((3, 1), [[1.0], [3.0], [5.0]]),
        },
    ),
    # The reference counts a negative index from the end for index_fill
    # alone; Lucid does for the whole family.
    "negative-index": _Valid(
        lambda m: (_matrix(m), 0, m.tensor([-1]), m.tensor([[10.0, 20.0]])),
        {"index_fill": ((3, 2), [[0.0, 1.0], [2.0, 3.0], [-1.0, -1.0]])},
    ),
    "0d-index": _Valid(
        lambda m: (_matrix(m), 0, m.tensor(1), m.tensor([[10.0, 20.0]])),
        {
            "index_copy": ((3, 2), [[0.0, 1.0], [10.0, 20.0], [4.0, 5.0]]),
            "index_add": ((3, 2), [[0.0, 1.0], [12.0, 23.0], [4.0, 5.0]]),
            "index_fill": ((3, 2), [[0.0, 1.0], [-1.0, -1.0], [4.0, 5.0]]),
            "index_select": ((1, 2), [[2.0, 3.0]]),
        },
    ),
    "empty-index": _Valid(
        lambda m: (_matrix(m), 0, m.zeros(0, dtype=m.int64), m.zeros(0, 2)),
        {
            **dict.fromkeys(
                _SOURCED | {"index_fill"},
                ((3, 2), [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]),
            ),
            "index_select": ((0, 2), []),
        },
    ),
    "empty-index-into-an-empty-axis": _Valid(
        lambda m: (m.zeros(0, 2), 0, m.zeros(0, dtype=m.int64), m.zeros(0, 2)),
        dict.fromkeys(_ALL, ((0, 2), [])),
    ),
    "0d-input": _Valid(
        lambda m: (m.tensor(3.0), 0, m.tensor(0), m.tensor(5.0)),
        {
            "index_copy": ((), 5.0),
            "index_add": ((), 8.0),
            "index_fill": ((), -1.0),
            "index_select": ((), 3.0),
        },
    ),
    "0d-input-1d-index": _Valid(
        lambda m: (m.tensor(3.0), -1, m.tensor([0]), m.tensor(5.0)),
        {
            "index_copy": ((), 5.0),
            "index_add": ((), 8.0),
            "index_fill": ((), -1.0),
            "index_select": ((), 3.0),
        },
    ),
    "0d-input-1d-source": _Valid(
        lambda m: (m.tensor(3.0), 0, m.tensor(0), m.tensor([5.0])),
        {"index_copy": ((), 5.0)},
    ),
    "0d-source-into-a-vector": _Valid(
        lambda m: (m.zeros(3), 0, m.tensor(1), m.tensor(7.0)),
        {"index_copy": ((3,), [0.0, 7.0, 0.0])},
    ),
}


def _valid_cells(with_device: bool) -> list[object]:
    cells: list[object] = []
    for case, valid in _VALID.items():
        for spelling in _FAMILY:
            if _op(spelling) not in valid.want:
                continue
            if not with_device:
                cells.append(pytest.param(spelling, case, id=f"{spelling}-{case}"))
                continue
            for device in _DEVICES:
                cells.append(pytest.param(spelling, case, device, id=f"{spelling}-{case}-{device}"))
    return cells


@pytest.mark.parametrize(("spelling", "case", "device"), _valid_cells(with_device=True))
def test_a_valid_index_answers_as_the_reference(spelling: str, case: str, device: str) -> None:
    valid = _VALID[case]
    shape, values = valid.want[_op(spelling)]
    args = tuple(a.to(device) if hasattr(a, "to") else a for a in valid.make(lucid))
    ours = _FAMILY[spelling](lucid, *args)
    assert isinstance(ours, lucid.Tensor)
    assert ours.device.type == device
    assert tuple(ours.shape) == shape
    assert ours.tolist() == values


@pytest.mark.parity
@pytest.mark.parametrize(("spelling", "case"), _valid_cells(with_device=False))
def test_the_recorded_answers_are_the_reference_s(
    ref: ModuleType, spelling: str, case: str
) -> None:
    valid = _VALID[case]
    theirs = _FAMILY[spelling](ref, *valid.make(ref))
    assert (tuple(theirs.shape), theirs.tolist()) == valid.want[_op(spelling)]


@pytest.mark.parametrize("spelling", list(_FAMILY))
def test_an_int32_index_answers_as_an_int64_one(device: str, spelling: str) -> None:
    # Lucid takes an int32 index throughout; the reference's index_copy
    # and index_fill take int64 only.
    x = lucid.arange(15.0, device=device).reshape(5, 3)
    src = lucid.ones(2, 3, device=device)
    wide = lucid.tensor([4, 0], dtype=lucid.int64, device=device)
    got = _FAMILY[spelling](lucid, x, 0, wide.to(lucid.int32), src)
    want = _FAMILY[spelling](lucid, x, 0, wide, src)
    assert isinstance(got, lucid.Tensor) and isinstance(want, lucid.Tensor)
    assert got.tolist() == want.tolist()


# ── siblings: put (same owner) and scatter_reduce (the engine's door) ──────

_PUTS: dict[str, _Call] = {
    "put": lambda m, x, d, i, s: m.put(x, i, s),
    "Tensor.put_": lambda m, x, d, i, s: x.clone().put_(i, s),
}

_PUT = frozenset({"put"})

_PUT_BAD: dict[str, _Bad] = {
    "float-index": _Bad(lambda m: (m.zeros(4, 3), 0, m.tensor([0.0, 1.0]), m.ones(2)), {"put": _R}),
    "bool-index": _Bad(
        lambda m: (m.zeros(4, 3), 0, m.tensor([True, False]), m.ones(2)), {"put": _R}
    ),
    "source-size": _Bad(lambda m: (m.zeros(4, 3), 0, m.tensor([0, 1]), m.ones(3)), {"put": _I}),
    "empty-input": _Bad(lambda m: (m.zeros(0), 0, m.tensor([0]), m.ones(1)), {"put": _I}),
    "index-value-past-the-end": _Bad(
        lambda m: (m.zeros(4, 3), 0, m.tensor([12]), m.ones(1)),
        {"put": _I},
        devices=_CPU,
    ),
    # The reference's GPU device has no put; the class is its CPU one's.
    "index-on-the-cpu": _Bad(
        lambda m: (m.zeros(4, 3), 0, m.tensor([0, 1]), m.ones(2)),
        {"put": _R},
        devices=_METAL,
        cpu_args=frozenset({2}),
    ),
    "source-on-the-cpu": _Bad(
        lambda m: (m.zeros(4, 3), 0, m.tensor([0, 1]), m.ones(2)),
        {"put": _R},
        devices=_METAL,
        cpu_args=frozenset({3}),
    ),
}

_SCATTERS: dict[str, _Call] = {
    "scatter_reduce-sum": lambda m, x, d, i, s: m.scatter_reduce(x, d, i, s, "sum"),
    "scatter_reduce-mean": lambda m, x, d, i, s: m.scatter_reduce(x, d, i, s, "mean"),
    "scatter_reduce-amax-noself": lambda m, x, d, i, s: m.scatter_reduce(
        x, d, i, s, "amax", include_self=False
    ),
    "Tensor.scatter_reduce-prod": lambda m, x, d, i, s: x.scatter_reduce(d, i, s, "prod"),
    "Tensor.scatter_reduce_-amin": lambda m, x, d, i, s: x.clone().scatter_reduce_(d, i, s, "amin"),
}

_SCATTER_BAD: dict[str, _Bad] = {
    "index-fewer-dims": _Bad(
        lambda m: (m.zeros(4, 3), 0, m.tensor([0, 1]), m.ones(4, 3)),
        {"scatter_reduce": _R},
    ),
    "index-more-dims": _Bad(
        lambda m: (m.zeros(4, 3), 0, m.zeros(2, 3, 1, dtype=m.int64), m.ones(4, 3)),
        {"scatter_reduce": _R},
    ),
    "src-more-dims": _Bad(
        lambda m: (m.zeros(2, 3), 0, m.tensor([[0, 1]]), m.ones(2, 3, 1)),
        {"scatter_reduce": _R},
    ),
    "float-index": _Bad(
        lambda m: (m.zeros(4, 3), 0, m.tensor([[0.0, 1.0, 2.0]]), m.ones(4, 3)),
        {"scatter_reduce": _R},
    ),
    "index-wider-than-input": _Bad(
        lambda m: (m.zeros(4, 3), 0, m.zeros(2, 4, dtype=m.int64), m.ones(4, 4)),
        {"scatter_reduce": _R},
    ),
    "index-larger-than-src": _Bad(
        lambda m: (m.zeros(4, 3), 0, m.zeros(2, 3, dtype=m.int64), m.ones(1, 3)),
        {"scatter_reduce": _R},
    ),
    "dim-past-the-last": _Bad(
        lambda m: (m.zeros(4, 3), 2, m.zeros(2, 3, dtype=m.int64), m.ones(4, 3)),
        {"scatter_reduce": _I},
    ),
    "index-into-an-empty-axis": _Bad(
        lambda m: (m.zeros(0, 3), 0, m.zeros(1, 3, dtype=m.int64), m.ones(1, 3)),
        {"scatter_reduce": _R},
    ),
    "index-value-past-the-axis": _Bad(
        lambda m: (m.zeros(4, 3), 0, m.tensor([[7, 0, 0]]), m.ones(1, 3)),
        {"scatter_reduce": _R},
        devices=_CPU,
    ),
    "index-on-the-cpu": _Bad(
        lambda m: (m.zeros(4, 3), 0, m.tensor([[0, 1, 2]]), m.ones(1, 3)),
        {"scatter_reduce": _R},
        devices=_METAL,
        cpu_args=frozenset({2}),
    ),
}

_SIBLINGS: dict[str, tuple[Mapping[str, _Call], Mapping[str, _Bad]]] = {
    "put": (_PUTS, _PUT_BAD),
    "scatter_reduce": (_SCATTERS, _SCATTER_BAD),
}


def _sibling_cells() -> list[object]:
    return [
        cell
        for family, (spellings, table) in _SIBLINGS.items()
        for cell in _cells(table, spellings, family)
    ]


@pytest.mark.parametrize(("family", "spelling", "case", "device"), _sibling_cells())
def test_a_sibling_raises_the_reference_class(
    family: str, spelling: str, case: str, device: str
) -> None:
    spellings, table = _SIBLINGS[family]
    bad = table[case]
    with pytest.raises(bad.kinds[_op(spelling)]):
        spellings[spelling](lucid, *_build(bad, lucid, device))


@pytest.mark.parametrize(("family", "spelling", "case", "device"), _sibling_cells())
def test_a_sibling_is_refused_by_the_public_op(
    family: str, spelling: str, case: str, device: str
) -> None:
    spellings, table = _SIBLINGS[family]
    with pytest.raises((IndexError, TypeError, ValueError, RuntimeError)) as info:
        spellings[spelling](lucid, *_build(table[case], lucid, device))
    assert _names(str(info.value), _op(spelling)), str(info.value)


@pytest.mark.parity
@pytest.mark.parametrize(
    ("family", "spelling", "case"),
    [
        cell
        for family, (spellings, table) in _SIBLINGS.items()
        for cell in _parity_cells(table, spellings, family)
    ],
)
def test_the_recorded_sibling_classes_are_the_reference_s(
    ref: ModuleType, family: str, spelling: str, case: str
) -> None:
    spellings, table = _SIBLINGS[family]
    bad = table[case]
    if family == "put" and "cpu" not in bad.devices:
        pytest.skip("the reference's GPU device has no put")
    with pytest.raises((IndexError, TypeError, ValueError, RuntimeError)) as theirs:
        spellings[spelling](ref, *_build(bad, ref, _ref_device(ref, bad)))
    assert _builtin(theirs.value) is bad.kinds[_op(spelling)], theirs.value


# ── empty-index scatter family (LCD-609) ────────────────────────────────────

_EMPTY_SCATTER_OPS: dict[str, Callable[[ModuleType, object, int, object, object], object]] = {
    "scatter": lambda m, b, d, i, s: m.scatter(b, d, i, s),
    "scatter_add": lambda m, b, d, i, s: m.scatter_add(b, d, i, s),
    "scatter_reduce-sum": lambda m, b, d, i, s: m.scatter_reduce(b, d, i, s, "sum"),
    "scatter_reduce-mean": lambda m, b, d, i, s: m.scatter_reduce(b, d, i, s, "mean"),
    "scatter_reduce-amax": lambda m, b, d, i, s: m.scatter_reduce(b, d, i, s, "amax"),
    "scatter_reduce-amin-noself": lambda m, b, d, i, s: m.scatter_reduce(
        b, d, i, s, "amin", include_self=False
    ),
    "scatter_reduce-prod": lambda m, b, d, i, s: m.scatter_reduce(b, d, i, s, "prod"),
}

_EMPTY_INDEX_SHAPES: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = (
    ((0,), (2, 3)),
    ((0, 0), (2, 3)),
    ((0,), (0,)),
    ((0,), ()),
    ((2, 0), (2, 3)),
)


@pytest.mark.parametrize(("idx_shape", "src_shape"), _EMPTY_INDEX_SHAPES)
@pytest.mark.parametrize("op", list(_EMPTY_SCATTER_OPS))
def test_empty_index_scatter_returns_input_and_propagates_grad(
    device: str,
    op: str,
    idx_shape: tuple[int, ...],
    src_shape: tuple[int, ...],
) -> None:
    base = lucid.arange(1.0, 7.0, device=device).reshape(2, 3).requires_grad_(True)
    idx = lucid.zeros(idx_shape, dtype=lucid.int64, device=device)
    src = lucid.ones(src_shape, device=device).requires_grad_(True)
    out = _EMPTY_SCATTER_OPS[op](lucid, base, 0, idx, src)
    assert isinstance(out, lucid.Tensor)
    assert out.shape == (2, 3)
    assert out.tolist() == base.tolist()
    out.sum().backward()
    assert base.grad is not None and base.grad.tolist() == [[1.0, 1.0, 1.0]] * 2
    assert src.grad is not None and src.grad.shape == src_shape
    assert src.grad.reshape(-1).tolist() == [0.0] * src.numel()


@pytest.mark.parity
@pytest.mark.parametrize(("idx_shape", "src_shape"), _EMPTY_INDEX_SHAPES)
@pytest.mark.parametrize("op", list(_EMPTY_SCATTER_OPS))
def test_empty_index_scatter_matches_reference(
    ref: ModuleType,
    op: str,
    idx_shape: tuple[int, ...],
    src_shape: tuple[int, ...],
) -> None:
    base = lucid.arange(1.0, 7.0).reshape(2, 3)
    idx = lucid.zeros(idx_shape, dtype=lucid.int64)
    src = lucid.ones(src_shape)
    got = _EMPTY_SCATTER_OPS[op](lucid, base, 0, idx, src)
    r_base = ref.arange(1.0, 7.0).reshape(2, 3)
    r_idx = ref.zeros(idx_shape, dtype=ref.int64)
    r_src = ref.ones(src_shape)
    want = _EMPTY_SCATTER_OPS[op](ref, r_base, 0, r_idx, r_src)
    assert got.tolist() == want.tolist()
