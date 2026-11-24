"""Operands of two dtypes, and positions of the wrong one: refused as the reference refuses them (LCD-1075).

Three classes of refusal share one owner, ``lucid._ops._bool_refusal``:

* an op whose two operands the reference will not promote -- ``polar``, ``vecdot``, ``cdist`` and
  the two-operand solvers, then ``heaviside`` and ``complex`` -- is a row of ``PAIRS`` and calls
  ``refuse_unlike``.  A float16 beside a float32 is refused, and so is a 0-d operand beside a
  vector: the reference does not rank the operands before it looks at their dtypes;
* an index tensor of a dtype the reference does not read (``take_along_dim`` takes int64 beside a
  dim, ``take`` takes int64, ``gather`` takes int32 and int64) is a row of ``ROWS`` keyed by its
  role;
* ``trapezoid`` and ``cumulative_trapezoid`` read a 0-d ``x`` as one coordinate along ``dim``.

Each table below is the reference's answer first (``-m parity`` replays it on the reference) and
Lucid's second, on every device.
"""

from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING

import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid._ops import _bool_refusal as _table
from lucid.test._fixtures.devices import device_supports

if TYPE_CHECKING:
    from _pytest.mark import ParameterSet

pytestmark = pytest.mark.filterwarnings("ignore")

# ``fw`` is the framework under test, ``a`` and ``b`` are the two operands.
Call = Callable[[ModuleType, object, object], object]


def _tensor(fw: ModuleType, data: object, dtype: str, device: str) -> object:
    """``data`` as a tensor of ``dtype`` in either framework (``device`` is Lucid's alone)."""
    if fw is lucid:
        return lucid.tensor(data, dtype=getattr(lucid, dtype), device=device)
    return fw.tensor(data, dtype=getattr(fw, dtype))


@dataclass(frozen=True)
class Layout:
    """The two operands of a call, as nested lists."""

    name: str
    first: object
    second: object


@dataclass(frozen=True)
class PairCase:
    """An op whose operands must share a dtype, its call, and the shapes it is tried at."""

    key: str
    call: Call
    layouts: tuple[Layout, ...]


_VECTOR = [1.0, 2.0, 3.0]
_SCALAR = 2.0

#: The four ways a pair of operands can be shaped for an op that broadcasts them.
_BROADCAST = (
    Layout("vector-vector", _VECTOR, _VECTOR),
    Layout("vector-0d", _VECTOR, _SCALAR),
    Layout("0d-vector", _SCALAR, _VECTOR),
    Layout("0d-0d", _SCALAR, _SCALAR),
)

PAIR_CASES: tuple[PairCase, ...] = (
    PairCase("polar", lambda fw, a, b: fw.polar(a, b), _BROADCAST),
    # The engine's ``mul`` reads a wide 0-d beside a half operand as a scalar and answers in the half
    # dtype; the reference refuses the pair, so the 0-d layouts are the ones that matter here.
    PairCase("linalg.vecdot", lambda fw, a, b: fw.linalg.vecdot(a, b), _BROADCAST),
    PairCase(
        "cdist",
        lambda fw, a, b: fw.cdist(a, b),
        (Layout("matrices", [[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 5.0]]),),
    ),
    PairCase(
        "linalg.solve",
        lambda fw, a, b: fw.linalg.solve(a, b),
        (Layout("matrix-column", [[2.0, 0.0], [0.0, 2.0]], [[1.0], [2.0]]),),
    ),
    PairCase(
        "linalg.cross",
        lambda fw, a, b: fw.linalg.cross(a, b),
        (Layout("vectors", _VECTOR, [3.0, 2.0, 1.0]),),
    ),
    PairCase(
        "linalg.householder_product",
        lambda fw, a, b: fw.linalg.householder_product(a, b),
        (Layout("reflectors", [[1.0, 0.0], [0.5, 1.0], [0.25, 0.5]], [0.5, 0.5]),),
    ),
    PairCase(
        "grid_sample",
        lambda fw, a, b: fw.nn.functional.grid_sample(a, b, align_corners=False),
        (
            Layout(
                "image-grid",
                [[[[1.0, 2.0], [3.0, 4.0]]]],
                [[[[0.0, 0.0], [0.5, 0.5]], [[-0.5, 0.0], [0.0, 0.0]]]],
            ),
        ),
    ),
    # The bicubic and the 5-D paths are composites of Python's: the pair is judged by the owner.
    PairCase(
        "grid_sample",
        lambda fw, a, b: fw.nn.functional.grid_sample(a, b, mode="bicubic", align_corners=False),
        (
            Layout(
                "bicubic-image-grid",
                [[[[1.0, 2.0], [3.0, 4.0]]]],
                [[[[0.0, 0.0], [0.5, 0.5]], [[-0.5, 0.0], [0.0, 0.0]]]],
            ),
        ),
    ),
    PairCase(
        "grid_sample",
        lambda fw, a, b: fw.nn.functional.grid_sample(a, b, align_corners=False),
        (
            Layout(
                "volume-grid",
                [[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]],
                [[[[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]]]],
            ),
        ),
    ),
    PairCase("heaviside", lambda fw, a, b: fw.heaviside(a, b), _BROADCAST),
    PairCase("complex", lambda fw, a, b: fw.complex(a, b), _BROADCAST),
)

#: The dtypes tried against each other.  The reference answers a pair only where the two are the
#: same and the op has a kernel for it: float32 and float64 (a half pair is a kernel some ops lack).
_FLOATS = ("float16", "bfloat16", "float32", "float64")
_UNLIKE = [(a, b) for a in (*_FLOATS, "int32") for b in (*_FLOATS, "int32") if a != b]
_LIKE = [("float32", "float32"), ("float64", "float64")]


def _cells(pairs: list[tuple[str, str]]) -> list[ParameterSet]:
    """``(case, layout, first, second)`` for every case, layout and dtype pair."""
    return [
        pytest.param(case, layout, first, second, id=f"{case.key}-{layout.name}-{first}-{second}")
        for case in PAIR_CASES
        for layout in case.layouts
        for first, second in pairs
    ]


def _verdict(
    fw: ModuleType, case: PairCase, layout: Layout, first: str, second: str, device: str
) -> str:
    """``"answers"``, or the class of the refusal: ``"guard"`` or ``"kernel"`` or the rest."""
    a = _tensor(fw, layout.first, first, device)
    b = _tensor(fw, layout.second, second, device)
    try:
        case.call(fw, a, b)
    except NotImplementedError:
        return "kernel"
    except RuntimeError:
        return "guard"
    return "answers"


def _supported(device: str, *names: str) -> bool:
    return all(device_supports(device, getattr(lucid, name)) for name in names)


@pytest.mark.parametrize(("case", "layout", "first", "second"), _cells(_UNLIKE))
def test_lucid_refuses_operands_of_two_dtypes(
    device: str, case: PairCase, layout: Layout, first: str, second: str
) -> None:
    if not _supported(device, first, second):
        pytest.skip(f"{device} cannot hold {first} / {second}")
    # A pair of floats is the pair guard's; an integer beside a float may meet its op's own refusal of
    # the integer first (a kernel the op has no instantiation of), which is as much a refusal.
    allowed = ("guard", "kernel") if "int32" in (first, second) else ("guard",)
    assert _verdict(lucid, case, layout, first, second, device) in allowed


@pytest.mark.parametrize(("case", "layout", "first", "second"), _cells(_LIKE))
def test_lucid_answers_operands_of_one_dtype(
    device: str, case: PairCase, layout: Layout, first: str, second: str
) -> None:
    if not _supported(device, first, second):
        pytest.skip(f"{device} cannot hold {first} / {second}")
    assert _verdict(lucid, case, layout, first, second, device) == "answers"


@pytest.mark.parity
@pytest.mark.parametrize(
    ("case", "layout", "first", "second"),
    [
        pytest.param(c.values[0], c.values[1], c.values[2], c.values[3], id=c.id)
        for c in _cells(_UNLIKE)
    ],
)
def test_the_reference_refuses_operands_of_two_dtypes(
    ref: ModuleType, case: PairCase, layout: Layout, first: str, second: str
) -> None:
    # A kernel the reference lacks for a half dtype (``cdist``) raises before it looks at the pair.
    assert _verdict(ref, case, layout, first, second, "cpu") in ("guard", "kernel")


@pytest.mark.parity
@pytest.mark.parametrize(
    ("case", "layout", "first", "second"),
    [
        pytest.param(c.values[0], c.values[1], c.values[2], c.values[3], id=c.id)
        for c in _cells(_LIKE)
    ],
)
def test_the_reference_answers_operands_of_one_dtype(
    ref: ModuleType, case: PairCase, layout: Layout, first: str, second: str
) -> None:
    assert _verdict(ref, case, layout, first, second, "cpu") == "answers"


def test_the_message_names_both_operands_and_both_dtypes() -> None:
    abs_ = lucid.tensor([1.0, 2.0], dtype=lucid.float32)
    angle = lucid.tensor([0.0, 1.0], dtype=lucid.float16)
    with pytest.raises(
        _C_engine.DtypeMismatch, match=r"polar: expected abs and angle .*same dtype"
    ):
        lucid.polar(abs_, angle)
    with pytest.raises(_C_engine.DtypeMismatch, match=r"float32 and float16"):
        lucid.polar(abs_, angle)


def test_a_pair_error_is_a_runtime_and_a_type_error() -> None:
    """The reference's class is a ``RuntimeError``; the engine's dtype guard is a ``TypeError`` too."""
    x = lucid.tensor([1.0], dtype=lucid.float32)
    y = lucid.tensor([1.0], dtype=lucid.float16)
    with pytest.raises(RuntimeError):
        lucid.linalg.vecdot(x, y)
    with pytest.raises(TypeError):
        lucid.linalg.vecdot(x, y)


def test_refuse_unlike_reads_dtypes_only_from_tensors() -> None:
    t = lucid.tensor([1.0], dtype=lucid.float32)
    for other in (2.0, 1, True, None, [1.0], "x"):
        _table.refuse_unlike("polar", t, other)
        _table.refuse_unlike("polar", other, t)
    _table.refuse_unlike("polar", t, lucid.tensor([2.0], dtype=lucid.float32))


def test_an_in_place_twin_takes_the_pair_of_its_out_of_place_op() -> None:
    x = lucid.tensor([1.0], dtype=lucid.float32)
    y = lucid.tensor([1.0], dtype=lucid.float16)
    with pytest.raises(_C_engine.DtypeMismatch, match=r"^polar_: expected abs and angle"):
        _table.refuse_unlike("polar_", x, y)


def test_a_pair_the_table_does_not_hold_is_a_key_error_not_a_silent_pass() -> None:
    x = lucid.tensor([1.0], dtype=lucid.float32)
    y = lucid.tensor([1.0], dtype=lucid.float16)
    with pytest.raises(KeyError, match="no pair row"):
        _table.refuse_unlike("an_op_with_no_row", x, y)
    _table.refuse_unlike("an_op_with_no_row", x, x)


def test_every_pair_row_has_a_case_and_every_case_a_row() -> None:
    cases = {case.key for case in PAIR_CASES}
    assert cases == set(_table.PAIRS), cases ^ set(_table.PAIRS)


def test_a_pair_is_declared_once() -> None:
    with pytest.raises(ValueError, match="declared twice"):
        _table._pair_table((_table.Pair("heaviside"), _table.Pair("heaviside")))


# ── index dtypes ────────────────────────────────────────────────────────────────────────────────

_INDEX_DTYPES = ("bool", "int8", "int16", "int32", "int64")


@dataclass(frozen=True)
class IndexCall:
    """A call that reads an index tensor, and the index dtypes the reference answers."""

    call: Callable[[ModuleType, object, object], object]
    #: The dtypes of an index with elements.
    answered: frozenset[str]
    #: The dtypes of an index with none: ``gather`` reads no element of it, so answers any.
    answered_empty: frozenset[str]
    #: Whether the index is read flat (1-d) rather than shaped like the result.
    flat: bool


_ALL = frozenset(_INDEX_DTYPES)
_INT64 = frozenset({"int64"})

#: ``take_along_dim`` reads int64 beside a dim and is a ``gather`` (int32 and int64) flattened;
#: ``take`` reads int64.
_INDEX_CALLS: dict[str, IndexCall] = {
    "take_along_dim": IndexCall(lambda fw, x, i: fw.take_along_dim(x, i, 1), _INT64, _INT64, False),
    "take_along_dim-flat": IndexCall(
        lambda fw, x, i: fw.take_along_dim(x, i), frozenset({"int32", "int64"}), _ALL, True
    ),
    "take": IndexCall(lambda fw, x, i: fw.take(x, i), _INT64, _INT64, True),
    "gather": IndexCall(
        lambda fw, x, i: fw.gather(x, 1, i), frozenset({"int32", "int64"}), _ALL, False
    ),
}

#: The empty cells the reference answers and Lucid's engine still refuses: a bool index is an engine
#: error whether or not it has elements.
_EMPTY_OPEN = frozenset({("gather", "bool"), ("take_along_dim-flat", "bool")})
_EMPTY_OPEN_REASON = "the engine's gather refuses an empty bool index, as it refuses a full one"

_X = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
_I = [[0, 2], [1, 0]]


def _index_cells(*, empty: bool) -> list[ParameterSet]:
    return [
        pytest.param(
            name,
            dtype,
            id=f"{name}-{dtype}",
            marks=(
                [pytest.mark.xfail(strict=True, reason=_EMPTY_OPEN_REASON)]
                if empty and (name, dtype) in _EMPTY_OPEN
                else []
            ),
        )
        for name in _INDEX_CALLS
        for dtype in _INDEX_DTYPES
    ]


def _index_verdict(fw: ModuleType, name: str, dtype: str, device: str, *, empty: bool) -> str:
    entry = _INDEX_CALLS[name]
    x = _tensor(fw, _X, "float32", device)
    if empty:
        index = _tensor(fw, [], dtype, device)
        if not entry.flat:
            index = index.reshape(2, 0)
    else:
        index = _tensor(fw, [0, 2, 1] if entry.flat else _I, dtype, device)
    try:
        entry.call(fw, x, index)
    except RuntimeError:
        return "refuses"
    except IndexError:
        return "refuses"
    return "answers"


def _index_want(name: str, dtype: str, *, empty: bool) -> str:
    entry = _INDEX_CALLS[name]
    return "answers" if dtype in (entry.answered_empty if empty else entry.answered) else "refuses"


@pytest.mark.parametrize(("name", "dtype"), _index_cells(empty=False))
def test_lucid_reads_the_index_dtypes_the_reference_reads(
    device: str, name: str, dtype: str
) -> None:
    got = _index_verdict(lucid, name, dtype, device, empty=False)
    assert got == _index_want(name, dtype, empty=False)


@pytest.mark.parametrize(("name", "dtype"), _index_cells(empty=True))
def test_lucid_reads_the_dtype_of_an_empty_index_as_the_reference_does(
    device: str, name: str, dtype: str
) -> None:
    """``gather`` of an index with no element is an empty result whatever its dtype; ``take`` and
    ``take_along_dim`` beside a dim still want int64."""
    got = _index_verdict(lucid, name, dtype, device, empty=True)
    assert got == _index_want(name, dtype, empty=True)


@pytest.mark.parity
@pytest.mark.parametrize("empty", [False, True], ids=["full", "empty"])
@pytest.mark.parametrize(("name", "dtype"), [(n, d) for n in _INDEX_CALLS for d in _INDEX_DTYPES])
def test_the_reference_reads_the_index_dtypes_the_table_says(
    ref: ModuleType, name: str, dtype: str, empty: bool
) -> None:
    got = _index_verdict(ref, name, dtype, "cpu", empty=empty)
    assert got == _index_want(name, dtype, empty=empty)


@pytest.mark.parametrize("dtype", ["bool", "int8", "int16", "int32"])
def test_take_along_dim_beside_a_dim_refuses_as_a_dtype_guard(device: str, dtype: str) -> None:
    x = lucid.tensor(_X, device=device)
    index = lucid.tensor(_I, dtype=getattr(lucid, dtype), device=device)
    with pytest.raises(
        _C_engine.DtypeMismatch, match=r"take_along_dim does not accept .* index tensor"
    ):
        lucid.take_along_dim(x, index, 1)
    with pytest.raises(_C_engine.DtypeMismatch, match="int64"):
        x.take_along_dim(index, 1)


@pytest.mark.parametrize("dtype", ["int8", "int16"])
def test_an_empty_narrow_gather_index_is_an_empty_result(device: str, dtype: str) -> None:
    src = lucid.ones(3, 2, device=device)
    empty = lucid.zeros(0, 2, dtype=getattr(lucid, dtype), device=device)
    assert tuple(lucid.gather(src, 0, empty).shape) == (0, 2)
    assert tuple(src.gather(0, empty).shape) == (0, 2)
    assert tuple(lucid.take_along_dim(src.reshape(-1), empty.reshape(-1)).shape) == (0,)


def test_a_narrow_gather_index_with_elements_is_still_refused(device: str) -> None:
    src = lucid.ones(3, 2, device=device)
    index = lucid.zeros(1, 2, dtype=lucid.int8, device=device)
    with pytest.raises(_C_engine.DtypeMismatch, match=r"gather does not accept an int8 index"):
        lucid.gather(src, 0, index)


# ── the exponent of ldexp ───────────────────────────────────────────────────────────────────────

_LDEXP_BASES = (
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "float16",
    "bfloat16",
    "float32",
    "float64",
    "complex64",
    "complex128",
)
_LDEXP_EXPONENTS = ("bool", "int32")
_REAL_FLOATS = frozenset({"float16", "bfloat16", "float32", "float64"})


def _ldexp_verdict(fw: ModuleType, base: str, exponent: str, device: str) -> str:
    x = _tensor(fw, [1, 2], base, device)
    e = _tensor(fw, [True, False] if exponent == "bool" else [1, 2], exponent, device)
    try:
        fw.ldexp(x, e)
    except NotImplementedError:
        return "kernel"
    except RuntimeError:
        return "guard"
    return "answers"


def _ldexp_want(base: str, exponent: str) -> str:
    """The reference has no kernel for a bool exponent beside a real float; every other cell answers."""
    return "kernel" if exponent == "bool" and base in _REAL_FLOATS else "answers"


@pytest.mark.parametrize("exponent", _LDEXP_EXPONENTS)
@pytest.mark.parametrize("base", _LDEXP_BASES)
def test_lucid_refuses_a_bool_exponent_beside_a_real_float(
    device: str, base: str, exponent: str
) -> None:
    if not _supported(device, base):
        pytest.skip(f"{device} cannot hold {base}")
    assert _ldexp_verdict(lucid, base, exponent, device) == _ldexp_want(base, exponent)


@pytest.mark.parity
@pytest.mark.parametrize("exponent", _LDEXP_EXPONENTS)
@pytest.mark.parametrize("base", _LDEXP_BASES)
def test_the_reference_refuses_a_bool_exponent_beside_a_real_float(
    ref: ModuleType, base: str, exponent: str
) -> None:
    assert _ldexp_verdict(ref, base, exponent, "cpu") == _ldexp_want(base, exponent)


def test_a_zero_d_bool_exponent_is_refused_too() -> None:
    x = lucid.tensor([1.0, 2.0])
    with pytest.raises(
        _C_engine.NotImplementedError, match=r"ldexp does not accept a bool exponent"
    ):
        lucid.ldexp(x, lucid.tensor(True))
    with pytest.raises(_C_engine.NotImplementedError, match="ldexp"):
        x.ldexp(lucid.tensor([True, False]))


# ── a 0-d x of trapezoid ────────────────────────────────────────────────────────────────────────

_TRAPEZOIDS = ("trapezoid", "cumulative_trapezoid")
_Y_SHAPES = [(0,), (1,), (2,), (3,), (2, 3), (2, 1), (2, 2), (3, 2), (0, 2), (2, 0)]


#: The ``(y, x)`` dtypes tried: the 0-d ``x`` ranks as a dimensioned operand, so a wider one gives
#: its dtype to the answer (a float64 ``x`` beside a float32 ``y`` is float64) and a narrower one
#: leaves ``y``'s.
_TRAPEZOID_DTYPES = [
    ("float32", "float32"),
    ("float32", "float64"),
    ("float64", "float32"),
    ("float16", "float32"),
    ("float32", "float16"),
    ("bfloat16", "float32"),
]


def _integral(
    fw: ModuleType,
    name: str,
    shape: tuple[int, ...],
    dim: int,
    device: str,
    dtypes: tuple[str, str],
) -> object:
    n = 1
    for extent in shape:
        n *= extent
    y = _tensor(fw, [float(k + 1) for k in range(n)], dtypes[0], device).reshape(*shape)
    x = _tensor(fw, 2.0, dtypes[1], device)
    return getattr(fw, name)(y, x, dim=dim)


def _outcome(
    fw: ModuleType,
    name: str,
    shape: tuple[int, ...],
    dim: int,
    device: str,
    dtypes: tuple[str, str] = ("float32", "float32"),
) -> tuple[str, tuple[int, ...], str, list[object]]:
    try:
        out = _integral(fw, name, shape, dim, device, dtypes)
    except RuntimeError:
        return ("refuses", (), "", [])
    values = out.to("cpu").tolist() if fw is lucid else out.tolist()
    return ("answers", tuple(out.shape), str(out.dtype).rsplit(".", 1)[-1], values)


@pytest.mark.parity
@pytest.mark.parametrize("dtypes", _TRAPEZOID_DTYPES, ids="-".join)
@pytest.mark.parametrize("dim", [-1, 0])
@pytest.mark.parametrize("shape", _Y_SHAPES, ids=str)
@pytest.mark.parametrize("name", _TRAPEZOIDS)
def test_a_zero_d_x_is_one_coordinate_along_dim(
    ref: ModuleType,
    device: str,
    name: str,
    shape: tuple[int, ...],
    dim: int,
    dtypes: tuple[str, str],
) -> None:
    """``y`` of ``n`` samples has ``n - 1`` areas and the one coordinate no spacing: they meet where
    ``n - 1`` is 0 or 1 (an empty or all-zero answer) and are a shape mismatch beyond it.  The
    answer's dtype is the pair's, the 0-d ``x`` ranked as the dimensioned operand it is viewed as."""
    if not _supported(device, *dtypes):
        pytest.skip(f"{device} cannot hold {dtypes}")
    want = _outcome(ref, name, shape, dim, "cpu", dtypes)
    assert _outcome(lucid, name, shape, dim, device, dtypes) == want


@pytest.mark.parity
@pytest.mark.parametrize("x_shape", [(), (0,)], ids=["0d", "vector"])
@pytest.mark.parametrize("name", _TRAPEZOIDS)
def test_zero_samples_keep_the_dtype_of_y_in_trapezoid_only(
    ref: ModuleType, device: str, name: str, x_shape: tuple[int, ...]
) -> None:
    """The reference answers a ``y`` with no sample before it ranks ``y`` with ``x``; the running
    integral ranks them as it does for any other length."""
    if not _supported(device, "float64"):
        pytest.skip(f"{device} cannot hold float64")
    x = 1.0 if not x_shape else []
    got = getattr(lucid, name)(
        _tensor(lucid, [], "float32", device), _tensor(lucid, x, "float64", device)
    )
    want = getattr(ref, name)(_tensor(ref, [], "float32", "cpu"), _tensor(ref, x, "float64", "cpu"))
    assert str(got.dtype).rsplit(".", 1)[-1] == str(want.dtype).rsplit(".", 1)[-1]
    assert tuple(got.shape) == tuple(want.shape)


@pytest.mark.parametrize("name", _TRAPEZOIDS)
def test_a_zero_d_x_beside_three_samples_is_a_shape_mismatch(device: str, name: str) -> None:
    y = lucid.tensor([1.0, 2.0, 3.0], device=device)
    with pytest.raises(RuntimeError):
        getattr(lucid, name)(y, lucid.tensor(2.0, device=device))


def test_a_vector_x_still_gives_the_trapezoidal_areas(device: str) -> None:
    y = lucid.tensor([1.0, 2.0, 4.0], device=device)
    x = lucid.tensor([0.0, 1.0, 3.0], device=device)
    assert lucid.trapezoid(y, x).item() == pytest.approx(1.5 + 6.0)
    assert lucid.cumulative_trapezoid(y, x).tolist() == pytest.approx([1.5, 7.5])
