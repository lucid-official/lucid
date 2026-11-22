"""A Python number or list beside a tensor meets the engine on that tensor's device (LCD-1021).

``isin`` built a number or a list with ``lucid.tensor(x)``, a tensor on the
default device, so ``isin(metal_tensor, 2)`` raised ``DeviceMismatch``;
``searchsorted`` and ``bucketize`` took no Python number at all and raised
``DtypeMismatch`` for two operands of different dtypes.  The defect class is "a
Python operand turned into a tensor without the device (and the dtype) of the
tensor it meets".  Its owner is :func:`lucid._dispatch._unwrap_or_scalar`, which
every adapter reaches a number through; ``isin`` is a composite that also takes
a list, so it goes through :func:`lucid._ops.composite._shared._operands_together`,
the same rule for a composite.

This file is the guard of the whole class:

* every entry that takes a Python number beside a tensor is replayed on both
  devices against the reference (``-m parity``): the sweep that found the two
  defects above found no other entry broken, and this keeps it so;
* a Python list is Lucid's extension -- the reference's ``isin`` and
  ``searchsorted`` take a number or a tensor, never a list -- so a list is
  checked against the reference's answer for a tensor of the same values;
* a 0-d CPU tensor follows the other operand's device where the reference
  allows it (the arithmetic and comparison operators, ``where``, ``masked_fill``
  ...) and is refused where it does not (``isin``); the reference's own verdict
  is read on its accelerator when the host has one;
* a static test fails when an op builds a tensor with no ``device=``.
"""

import ast
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Literal

import pytest

import lucid
import lucid._ops

# ``(data, dtype=None) -> tensor`` of one framework on one device.
_Make = Callable[..., object]
# ``(framework, make, operand) -> result``: one call, the operand in the slot under test.
_Call = Callable[[ModuleType, _Make, object], object]

_X = [1.5, -2.0, 3.0]
_I = [1, 2, 3]
_M = [True, False, True]
_IDX = [0, 2]

#: What the reference does with a 0-d CPU tensor in the operand's slot when the other
#: operands are on its accelerator: answers, refuses, or not asked.
_ZeroDim = Literal["answers", "refused", "unchecked"]


def _maker(fw: ModuleType, device: str) -> _Make:
    def make(data: object, dtype: str | None = None) -> object:
        extra = {"dtype": getattr(fw, dtype)} if dtype is not None else {}
        return fw.tensor(data, device=device, **extra)

    return make


def _name(dtype: object) -> str:
    return str(dtype).rsplit(".", 1)[-1]


def _same(got: object, want: object) -> None:
    """``got`` (a Lucid tensor) is ``want`` (the reference's): dtype, shape and values."""
    assert _name(got.dtype) == _name(want.dtype), (got.dtype, want.dtype)
    assert tuple(got.shape) == tuple(want.shape), (got.shape, want.shape)
    g, w = got.reshape(-1).tolist(), want.reshape(-1).tolist()
    if _name(want.dtype).startswith(("float", "bfloat")):
        assert g == pytest.approx(w, rel=1e-5, abs=1e-6, nan_ok=True)
    else:
        assert g == w


@dataclass(frozen=True)
class _Entry:
    """One call with a Python operand in one slot.

    ``scalar`` is the operand as a Python number; ``sequence`` the same values
    as a list, for the calls that take one (Lucid's extension); ``zero_dim``
    what the reference does with ``tensor(scalar)`` on the CPU in that slot
    when the other operands are on its accelerator.
    """

    call: _Call
    scalar: object
    sequence: list[object] | None = None
    zero_dim: _ZeroDim = "unchecked"


def _masked_fill_inplace(fw: ModuleType, mk: _Make, s: object) -> object:
    x = mk(_X)
    x.masked_fill_(mk(_M), s)
    return x


def _setitem(key: str) -> _Call:
    def call(fw: ModuleType, mk: _Make, s: object) -> object:
        x = mk(_X)
        x[{"mask": mk(_M), "slice": slice(0, 2), "list": _IDX}[key]] = s
        return x

    return call


_ENTRIES: dict[str, _Entry] = {
    # -- isin: the number or list was a CPU tensor beside a Metal one -------------------------
    "isin-test-scalar": _Entry(lambda fw, mk, s: fw.isin(mk(_I), s), 2, [2, 3], "refused"),
    "isin-elements-scalar": _Entry(lambda fw, mk, s: fw.isin(s, mk(_I)), 2, [2, 3, 7], "refused"),
    "isin-invert": _Entry(lambda fw, mk, s: fw.isin(mk(_I), s, invert=True), 2, [2, 3], "refused"),
    "isin-float-elements-int-scalar": _Entry(lambda fw, mk, s: fw.isin(mk(_X), s), -2, [-2, 3]),
    "isin-int-elements-float-scalar": _Entry(lambda fw, mk, s: fw.isin(mk(_I), s), 2.0, [2.0]),
    "isin-int-elements-fractional-scalar": _Entry(lambda fw, mk, s: fw.isin(mk(_I), s), 2.5),
    "isin-2d-elements": _Entry(lambda fw, mk, s: fw.isin(mk([[1, 2], [3, 4]]), s), 3, [3, 9]),
    # A number is weak beside a tensor: it takes the tensor's dtype, wrapping or rounding to it.
    "isin-weak-scalar-int8": _Entry(lambda fw, mk, s: fw.isin(mk([44], "int8"), s), 300),
    "isin-weak-scalar-float16": _Entry(
        lambda fw, mk, s: fw.isin(mk([1.0], "float16"), s), 1.0000001
    ),
    # -- searchsorted / bucketize: a number as the values was a TypeError ---------------------
    "searchsorted-int-scalar": _Entry(lambda fw, mk, s: fw.searchsorted(mk(_I), s), 2),
    "searchsorted-fractional-scalar": _Entry(lambda fw, mk, s: fw.searchsorted(mk(_I), s), 2.5),
    "searchsorted-bool-scalar-beside-int": _Entry(
        lambda fw, mk, s: fw.searchsorted(mk(_I), s), True
    ),
    "searchsorted-float-sequence": _Entry(
        lambda fw, mk, s: fw.searchsorted(mk([-2.0, 1.5, 3.0]), s), 2
    ),
    "searchsorted-right": _Entry(lambda fw, mk, s: fw.searchsorted(mk(_I), s, right=True), 2),
    "searchsorted-out-int32": _Entry(
        lambda fw, mk, s: fw.searchsorted(mk(_I), s, out_int32=True), 2.5
    ),
    "bucketize-int-scalar": _Entry(lambda fw, mk, s: fw.bucketize(s, mk(_I)), 2),
    "bucketize-fractional-scalar": _Entry(lambda fw, mk, s: fw.bucketize(s, mk(_I)), 2.5),
    "bucketize-right-out-int32": _Entry(
        lambda fw, mk, s: fw.bucketize(s, mk(_I), right=True, out_int32=True), 2
    ),
    # -- where / clamp / clip: the bound or branch is a number ---------------------------------
    "where-scalar-input": _Entry(
        lambda fw, mk, s: fw.where(mk(_M), s, mk(_X)), 1.0, None, "answers"
    ),
    "where-scalar-other": _Entry(
        lambda fw, mk, s: fw.where(mk(_M), mk(_X), s), 0.5, None, "answers"
    ),
    "where-two-scalars": _Entry(lambda fw, mk, s: fw.where(mk(_M), s, 0.0), 1.0),
    "where-int-tensor-float-scalar": _Entry(lambda fw, mk, s: fw.where(mk(_M), mk(_I), s), 1.5),
    "clamp-min": _Entry(lambda fw, mk, s: fw.clamp(mk(_X), min=s), 0.0),
    "clamp-max-method": _Entry(lambda fw, mk, s: mk(_X).clamp(max=s), 1.0),
    "clamp-int-tensor-int-bounds": _Entry(lambda fw, mk, s: fw.clamp(mk(_I), min=1, max=s), 2),
    "clip-both": _Entry(lambda fw, mk, s: fw.clip(mk(_X), min=-1.0, max=s), 2.0),
    # -- fill-style arguments --------------------------------------------------------------------
    "masked_fill": _Entry(lambda fw, mk, s: mk(_X).masked_fill(mk(_M), s), 0.5, None, "answers"),
    "masked_fill_inplace": _Entry(_masked_fill_inplace, 0.5, None, "answers"),
    "fill_": _Entry(lambda fw, mk, s: mk(_X).fill_(s), 2, None, "answers"),
    "full_like": _Entry(lambda fw, mk, s: fw.full_like(mk(_X), s), 2),
    "new_full": _Entry(lambda fw, mk, s: mk(_X).new_full((2,), s), 3),
    "index_fill": _Entry(lambda fw, mk, s: mk(_X).index_fill(0, mk(_IDX), s), 9.0, None, "answers"),
    "scatter-value": _Entry(lambda fw, mk, s: mk(_X).scatter(0, mk(_IDX), s), 9.0),
    "setitem-mask": _Entry(_setitem("mask"), 9.0, None, "answers"),
    "setitem-slice": _Entry(_setitem("slice"), 9.0, None, "answers"),
    "setitem-index-list": _Entry(_setitem("list"), 9.0),
    "lerp-weight": _Entry(
        lambda fw, mk, s: fw.lerp(mk(_X), mk([3.0, 3.0, 3.0]), s), 0.5, None, "answers"
    ),
}

#: Binary ops the reference takes as ``op(tensor, number)``, and the ones it also takes as
#: ``op(number, tensor)`` (it refuses a number first for the others).
_NUMBER_SECOND = ["add", "sub", "mul", "div", "floor_divide", "remainder", "fmod", "pow"]
_NUMBER_SECOND += ["eq", "ne", "lt", "le", "gt", "ge"]
_NUMBER_FIRST = ["add", "sub", "mul", "div", "floor_divide", "remainder", "pow"]
#: A float tensor beside an int number, and an int tensor beside a float one.
_OPERANDS = [("float-int", _X, 2), ("int-float", _I, 2.5)]


def _binary(op: str, number_first: bool, data: list[object]) -> _Call:
    def call(fw: ModuleType, mk: _Make, s: object) -> object:
        fn = getattr(fw, op)
        return fn(s, mk(data)) if number_first else fn(mk(data), s)

    return call


for _op in _NUMBER_SECOND:
    for _tag, _data, _number in _OPERANDS:
        _ENTRIES[f"{_op}-tensor-number-{_tag}"] = _Entry(
            _binary(_op, False, _data), _number, None, "answers"
        )
for _op in _NUMBER_FIRST:
    for _tag, _data, _number in _OPERANDS:
        _ENTRIES[f"{_op}-number-tensor-{_tag}"] = _Entry(
            _binary(_op, True, _data), _number, None, "answers"
        )

_NAMES = list(_ENTRIES)
_WITH_SEQUENCE = [n for n in _NAMES if _ENTRIES[n].sequence is not None]
_ANSWERS = [n for n in _NAMES if _ENTRIES[n].zero_dim == "answers"]
_REFUSED = [n for n in _NAMES if _ENTRIES[n].zero_dim == "refused"]


def _zero_dim(fw: ModuleType, scalar: object) -> object:
    return fw.tensor(scalar, device="cpu")


@pytest.mark.parity
@pytest.mark.parametrize("name", _NAMES)
def test_a_python_number_beside_a_tensor_answers_as_the_reference_does(
    ref: ModuleType, device: str, name: str
) -> None:
    entry = _ENTRIES[name]
    got = entry.call(lucid, _maker(lucid, device), entry.scalar)
    _same(got, entry.call(ref, _maker(ref, "cpu"), entry.scalar))
    assert got.device.type == device


@pytest.mark.parity
@pytest.mark.parametrize("name", _WITH_SEQUENCE)
def test_a_python_list_is_the_tensor_of_its_values(ref: ModuleType, device: str, name: str) -> None:
    # The reference takes no list, so it is asked the question with a tensor of the same values.
    entry = _ENTRIES[name]
    got = entry.call(lucid, _maker(lucid, device), entry.sequence)
    _same(got, entry.call(ref, _maker(ref, "cpu"), ref.tensor(entry.sequence)))
    assert got.device.type == device


@pytest.mark.parity
@pytest.mark.parametrize("name", _ANSWERS)
def test_a_zero_dim_cpu_tensor_follows_the_device_of_the_other_operand(
    ref: ModuleType, device: str, name: str
) -> None:
    entry = _ENTRIES[name]
    got = entry.call(lucid, _maker(lucid, device), _zero_dim(lucid, entry.scalar))
    _same(got, entry.call(ref, _maker(ref, "cpu"), _zero_dim(ref, entry.scalar)))
    assert got.device.type == device


@pytest.mark.parity
@pytest.mark.parametrize("name", _REFUSED)
def test_a_zero_dim_cpu_tensor_is_refused_off_the_cpu_as_the_reference_refuses_it(
    ref: ModuleType, device: str, name: str
) -> None:
    entry = _ENTRIES[name]
    zero = _zero_dim(lucid, entry.scalar)
    if device == "cpu":
        got = entry.call(lucid, _maker(lucid, device), zero)
        _same(got, entry.call(ref, _maker(ref, "cpu"), _zero_dim(ref, entry.scalar)))
    else:
        with pytest.raises(RuntimeError):
            entry.call(lucid, _maker(lucid, device), zero)


def _accelerator(ref: ModuleType) -> str:
    if not ref.backends.mps.is_available():
        pytest.skip("the reference has no accelerator on this host")
    return "mps"


@pytest.mark.parity
@pytest.mark.parametrize("name", _ANSWERS)
def test_the_reference_answers_a_zero_dim_cpu_tensor_on_its_accelerator(
    ref: ModuleType, name: str
) -> None:
    entry = _ENTRIES[name]
    entry.call(ref, _maker(ref, _accelerator(ref)), _zero_dim(ref, entry.scalar))


@pytest.mark.parity
@pytest.mark.parametrize("name", _REFUSED)
def test_the_reference_refuses_a_zero_dim_cpu_tensor_on_its_accelerator(
    ref: ModuleType, name: str
) -> None:
    entry = _ENTRIES[name]
    with pytest.raises(RuntimeError):
        entry.call(ref, _maker(ref, _accelerator(ref)), _zero_dim(ref, entry.scalar))


# ── the operand rules, without the reference ────────────────────────────────────────────────


def test_with_no_tensor_among_them_the_operands_are_built_on_the_default_device() -> None:
    got = lucid.isin([1, 2, 3], 2)
    assert got.tolist() == [False, True, False]
    assert got.device.type == "cpu"
    scalar_elements = lucid.isin(2, [2, 3])
    assert scalar_elements.shape == ()
    assert scalar_elements.item() is True


@pytest.mark.parametrize("operand", [True, [True]], ids=["scalar", "list"])
@pytest.mark.parametrize("side", ["elements", "test_elements"])
def test_isin_refuses_a_python_bool_as_it_refuses_a_bool_tensor(
    device: str, side: str, operand: object
) -> None:
    # A Python bool beside an integer tensor is not weak: the op sees the bool, as the table says.
    ints = lucid.tensor(_I, device=device)
    args = (ints, operand) if side == "test_elements" else (operand, ints)
    with pytest.raises(RuntimeError, match="isin does not accept bool"):
        lucid.isin(*args)


@pytest.mark.parity
def test_the_reference_refuses_a_python_bool_in_isin(ref: ModuleType) -> None:
    with pytest.raises(RuntimeError):
        ref.isin(ref.tensor(_I), True)
    with pytest.raises(RuntimeError):
        ref.isin(True, ref.tensor(_I))


def test_a_python_bool_beside_a_bool_sequence_is_refused_and_a_python_int_is_not(
    device: str,
) -> None:
    sequence = lucid.tensor([False, True], device=device)
    with pytest.raises(RuntimeError, match="searchsorted does not accept bool"):
        lucid.searchsorted(sequence, True)
    with pytest.raises(RuntimeError, match="bucketize does not accept bool"):
        lucid.bucketize(True, sequence)
    assert lucid.searchsorted(sequence, 1).item() == 1
    assert lucid.bucketize(1, sequence).item() == 1


@pytest.mark.parity
def test_the_reference_refuses_a_python_bool_beside_a_bool_sequence_only(ref: ModuleType) -> None:
    sequence = ref.tensor([False, True])
    with pytest.raises(NotImplementedError):
        ref.searchsorted(sequence, True)
    with pytest.raises(NotImplementedError):
        ref.bucketize(True, sequence)
    assert ref.searchsorted(sequence, 1).item() == 1


@pytest.mark.parametrize("name", ["searchsorted", "bucketize"])
def test_the_sorted_sequence_is_a_tensor(name: str) -> None:
    call = lucid.searchsorted if name == "searchsorted" else lambda s, v: lucid.bucketize(v, s)
    with pytest.raises(TypeError):
        call(2, lucid.tensor(_I))


# ── searchsorted / bucketize: the two operands are promoted to a common dtype ───────────────

_SEQUENCE: dict[str, list[object]] = {
    "bool": [False, True],
    "int8": [0, 1, 2, 3],
    "int32": [0, 1, 2, 3],
    "int64": [0, 1, 2, 3],
    "float16": [0.0, 1.0, 2.0, 3.0],
    "float32": [0.0, 1.0, 2.0, 3.0],
}
_VALUES: dict[str, list[object]] = {
    "bool": [False, True],
    "int8": [-1, 0, 1, 2, 3, 4],
    "int32": [-1, 0, 1, 2, 3, 4],
    "int64": [-1, 0, 1, 2, 3, 4],
    "float16": [-0.5, 0.0, 0.5, 1.0, 2.5, 3.0, 3.5],
    "float32": [-0.5, 0.0, 0.5, 1.0, 2.5, 3.0, 3.5],
}
#: ``(sequence dtype, values dtype)``: a bool beside each category, and each way two
#: integers or two floats of different widths meet; (bool, bool) is the refusal of LCD-1013.
_PAIRS = [
    ("int64", "bool"),
    ("bool", "int64"),
    ("bool", "float32"),
    ("float32", "bool"),
    ("int32", "int64"),
    ("int64", "int32"),
    ("int8", "int64"),
    ("int64", "float32"),
    ("float32", "int64"),
    ("float16", "int64"),
    ("int64", "float16"),
    ("float32", "float16"),
    ("float16", "float32"),
]


def _unsorted(dtype: str) -> tuple[list[object], list[int]]:
    """The sequence of ``dtype`` out of order, and the sorter that puts it right."""
    ordered = _SEQUENCE[dtype]
    order = [1, 0] if len(ordered) == 2 else [2, 0, 3, 1]
    shuffled = [ordered[i] for i in order]
    return shuffled, sorted(range(len(shuffled)), key=lambda k: shuffled[k])


_SEARCHES: dict[str, Callable[..., object]] = {
    "searchsorted": lambda fw, s, v: fw.searchsorted(s, v),
    "searchsorted-right": lambda fw, s, v: fw.searchsorted(s, v, right=True),
    "searchsorted-side-right": lambda fw, s, v: fw.searchsorted(s, v, side="right"),
    "searchsorted-out-int32": lambda fw, s, v: fw.searchsorted(s, v, out_int32=True),
    "bucketize": lambda fw, s, v: fw.bucketize(v, s),
    "bucketize-right-out-int32": lambda fw, s, v: fw.bucketize(v, s, right=True, out_int32=True),
}


@pytest.mark.parity
@pytest.mark.parametrize("search", list(_SEARCHES))
@pytest.mark.parametrize(("seq_dtype", "values_dtype"), _PAIRS)
def test_two_operands_of_different_dtypes_are_searched_at_their_common_dtype(
    ref: ModuleType, device: str, search: str, seq_dtype: str, values_dtype: str
) -> None:
    def run(fw: ModuleType, where: str) -> object:
        mk = _maker(fw, where)
        s, v = mk(_SEQUENCE[seq_dtype], seq_dtype), mk(_VALUES[values_dtype], values_dtype)
        return _SEARCHES[search](fw, s, v)

    got = run(lucid, device)
    _same(got, run(ref, "cpu"))
    assert got.device.type == device


@pytest.mark.parity
@pytest.mark.parametrize(("seq_dtype", "values_dtype"), _PAIRS)
def test_a_sorter_is_applied_beside_a_mixed_dtype_pair(
    ref: ModuleType, device: str, seq_dtype: str, values_dtype: str
) -> None:
    shuffled, sorter = _unsorted(seq_dtype)

    def run(fw: ModuleType, where: str) -> object:
        mk = _maker(fw, where)
        values = mk(_VALUES[values_dtype], values_dtype)
        return fw.searchsorted(mk(shuffled, seq_dtype), values, sorter=mk(sorter, "int64"))

    _same(run(lucid, device), run(ref, "cpu"))


@pytest.mark.parity
def test_a_batch_of_sequences_beside_values_of_another_dtype(ref: ModuleType, device: str) -> None:
    def run(fw: ModuleType, where: str) -> object:
        mk = _maker(fw, where)
        seq = mk([[1, 3, 5], [2, 4, 6]], "int64")
        return fw.searchsorted(seq, mk([[1.5, 3.5], [2.0, 7.0]], "float32"))

    _same(run(lucid, device), run(ref, "cpu"))


@pytest.mark.parity
@pytest.mark.parametrize("dims", [0, 1])
def test_a_dimensioned_operand_outranks_a_zero_dim_one_in_the_common_dtype(
    ref: ModuleType, device_cpu_only: str, dims: int
) -> None:
    # The reference's tiers: a 0-d float64 value beside a float32 sequence is searched at float32,
    # so 2.5000000001 is 2.5 and the leftmost slot is 1; a 1-d float64 one is searched at
    # float64 and lands past 2.5, in slot 2.
    value = 2.5000000001 if dims == 0 else [2.5000000001]

    def run(fw: ModuleType) -> object:
        mk = _maker(fw, device_cpu_only)
        return fw.searchsorted(mk([1.5, 2.5], "float32"), mk(value, "float64"))

    _same(run(lucid), run(ref))
    assert run(lucid).reshape(-1).tolist() == [1 if dims == 0 else 2]


def test_the_search_options_keep_their_checks(device: str) -> None:
    seq = lucid.tensor(_I, device=device)
    values = lucid.tensor([False, True], device=device)
    with pytest.raises(RuntimeError, match="side can only be"):
        lucid.searchsorted(seq, values, side="middle")
    with pytest.raises(RuntimeError, match="opposites"):
        lucid.searchsorted(seq, values, side="left", right=True)
    with pytest.raises(RuntimeError, match="sorter must be int64"):
        lucid.searchsorted(seq, values, sorter=lucid.tensor([0.0, 1.0, 2.0], device=device))
    with pytest.raises(RuntimeError, match="sorter must be int64"):
        lucid.searchsorted(seq, values, sorter=lucid.tensor([0, 1], device=device))


# ── the guard of the class, static ──────────────────────────────────────────────────────────

_BUILDERS = frozenset(
    {"tensor", "as_tensor", "asarray", "scalar_tensor", "full", "zeros", "ones", "empty", "arange"}
)
#: ``(file under lucid/_ops, function)`` that build a tensor with no operand to follow: the
#: empty result of ``block_diag()`` has no tensor among its arguments, and
#: ``_operand_beside`` is the one place that says so (``like`` is ``None``).
_NO_PARTNER = {("composite/blas.py", "block_diag"), ("composite/_shared.py", "_operand_beside")}


def _deviceless_builders(source: str) -> list[tuple[str, int]]:
    """``(enclosing function, line)`` of each ``lucid.<builder>(...)`` with no ``device=``."""
    found: list[tuple[str, int]] = []

    def visit(node: ast.AST, scope: str) -> None:
        for child in ast.iter_child_nodes(node):
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and isinstance(child.func.value, ast.Name)
                and child.func.value.id == "lucid"
                and child.func.attr in _BUILDERS
                and not any(k.arg in ("device", None) for k in child.keywords)
            ):
                found.append((scope, child.lineno))
            visit(child, child.name if isinstance(child, ast.FunctionDef) else scope)

    visit(ast.parse(source), "<module>")
    return found


def test_the_scan_finds_a_builder_with_no_device() -> None:
    source = (
        "def f(x):\n    return lucid.tensor(x)\n"
        "def g(x, d):\n    return lucid.tensor(x, device=d)\n"
        "def h(x, **kw):\n    return lucid.full((1,), x, **kw)\n"
    )
    assert _deviceless_builders(source) == [("f", 2)]


def test_no_op_builds_a_tensor_without_following_a_device() -> None:
    # ``lucid.tensor(x)`` is on the default device: beside a Metal tensor it is a CPU one and the
    # engine refuses the pair.  An op builds its constants with ``device=`` or through
    # ``_unwrap_or_scalar`` / ``_operands_together`` -- a case with no tensor to follow is named.
    root = Path(lucid._ops.__file__).parent
    offenders = [
        f"{path.relative_to(root)}:{line} in {scope}"
        for path in sorted(root.rglob("*.py"))
        for scope, line in _deviceless_builders(path.read_text())
        if (path.relative_to(root).as_posix(), scope) not in _NO_PARTNER
    ]
    assert not offenders, "tensor built with no device= in lucid/_ops: " + ", ".join(offenders)
