"""A 0-d tensor as a scale factor, read as the reference reads it (LCD-1088).

``alpha``, ``beta``, ``value`` and ``index_add``'s ``alpha`` are ``Number`` arguments of the
reference, and its argument parser takes a 0-d tensor for one: it reads the number the tensor holds
(``item()``, a bool, int, float or complex of the tensor's own dtype) and from there on the call is
the call with that Python number.  ``add(int32, int32, alpha=tensor(2.5))`` is refused because
``alpha=2.5`` is, ``addcmul(int32, ..., value=tensor(2.0))`` stays ``int32``, and
``index_add(int32, ..., alpha=tensor(2.0))`` answers.  Lucid multiplied by the tensor as by an
operand, which promoted, ignored the by-type rules and mismatched dtypes.

``lucid/_scalar_cast.py`` owns the reading (``scale_factor``, applied at the entry of every op that
takes a scale factor), and this sweep holds every op x spelling x tensor dtype x scalar dtype x
device to the reference, class, message, dtype and value, on the op table of
``_scale_factor_sweep.py`` that ``test_scale_factor_reading.py`` uses for Python numbers.  Beside
the 0-d tensors:

* a tensor with a dimension, or one that requires grad, is not a number: ``TypeError`` before the
  op checks anything else (``add(): argument 'alpha' must be Number, not Tensor``).  The reference
  names an in-place twin it writes (``addmm_()``); Lucid names the op it runs (``addmm()``), so
  the sweep compares the name without its trailing underscore.  Where the reference lists every
  signature (``addmm``, ``baddbmm``, ``subtract``, ``rsub``) the sweep holds the class and the
  short sentence;
* ``lerp``'s ``weight`` is an operand in the reference too, so a 0-d tensor there is promoted and
  never read as a number: the sweep holds it to the reference as well, to show the line.
"""

import math
import re
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import lucid
from lucid.amp import autocast
from lucid.test._fixtures.devices import devices_supporting
from lucid.test.unit.ops._scale_factor_sweep import (
    COMPLEX,
    HALF,
    HALF_RTOL,
    OPS,
    Side,
    outcome,
    same_values,
)

pytestmark = pytest.mark.parity

#: ``(dtype, value)`` of the 0-d tensor passed as the scale factor.  Each dtype family, a value the
#: element types of the operand tensor hold and one they refuse, the by-type refusals (a float for
#: an integer tensor, a bool, a complex number) and the non-finite values.
_SCALARS: tuple[tuple[str, complex], ...] = (
    ("bool", True),
    ("bool", False),
    ("int8", 2),
    ("int8", -128),
    ("int32", 300),
    ("int64", 2**40),
    ("float16", 2.5),
    ("float32", 2.5),
    ("float32", 2.0),
    ("float32", math.inf),
    ("float32", math.nan),
    ("float64", 300.7),
    ("float64", 1e30),
    ("float64", 3.5e38),
    ("complex64", 1 + 2j),
    ("complex64", 2 + 0j),
    ("complex128", 0.5 + 0.25j),
)


def _id(dtype: str, value: complex) -> str:
    return f"{dtype}({value})"


def _cells() -> list[Any]:
    return [
        pytest.param(name, dtype, device, scalar, id=f"{name}-{dtype}-{device}-{_id(*scalar)}")
        for name, op in OPS.items()
        for dtype in op.dtypes
        for device in devices_supporting(getattr(lucid, dtype))
        for scalar in _SCALARS
        # complex arithmetic with an infinite scalar is its own gap (see test_scale_factor_reading)
        if not (dtype in COMPLEX and math.isinf(complex(scalar[1]).real))
        # a complex ``lerp`` weight is an operand that promotes the result (the reference has no
        # complex32 for a half pair, and ``lerp_`` spells its dtype refusal with other names)
        if not (name.startswith("lerp") and scalar[0] in COMPLEX)
    ]


@pytest.mark.parametrize(("name", "dtype", "device", "scalar"), _cells())
def test_a_0d_tensor_scale_factor_is_read_as_the_number_it_holds(
    name: str, dtype: str, device: str, scalar: tuple[str, complex], ref: ModuleType
) -> None:
    op = OPS[name]
    ours, theirs = Side(lucid, dtype, device), Side(ref, dtype, device)
    got = outcome(lambda: op.call(ours, ours.scalar(*scalar)))
    want = outcome(lambda: op.call(theirs, theirs.scalar(*scalar)))
    assert got[0] == want[0], f"{got} vs {want}"
    if got[0] == "raised":
        assert got[1] is want[1]
        assert got[2] == want[2], f"{got[2]!r} vs {want[2]!r}"
        return
    assert got[1] == want[1]
    rtol = HALF_RTOL[dtype] if op.op_math and dtype in HALF else 0.0
    assert same_values(got[2], want[2], rtol), f"{got[2]} vs {want[2]}"


@pytest.mark.parametrize(
    "name",
    [
        "add",
        "sub",
        "rsub",
        "add-method",
        "add_",
        "addmm-alpha",
        "addmv-beta",
        "addr-alpha",
        "baddbmm-beta",
        "addcmul",
        "addcdiv_",
        "index_add",
    ],
)
@pytest.mark.parametrize("device", devices_supporting(lucid.float32))
def test_a_0d_tensor_on_the_cpu_is_read_for_operands_on_another_device(
    name: str, device: str, ref: ModuleType
) -> None:
    """The reading is a host read of one number, so where the tensor lives does not matter."""
    op = OPS[name]
    ours, theirs = Side(lucid, "float32", device), Side(ref, "float32", device)
    cpu = lucid.tensor(2.5, device="cpu")
    got = outcome(lambda: op.call(ours, cpu))
    want = outcome(lambda: op.call(theirs, theirs.scalar("float32", 2.5)))
    assert got[0] == want[0] == "answered", f"{got} vs {want}"
    assert got[1:] == want[1:]


# ── a tensor that is not a number ─────────────────────────────────────────────────────────────

#: Every op whose scale factor must be a number, except ``lerp`` (an operand when it is a tensor).
_NUMBER_OPS = tuple(name for name in OPS if not name.startswith("lerp"))
#: ``(label, builder)``: a tensor the reference does not take for a ``Number``.
_NOT_A_NUMBER: dict[str, Callable[[Side], Any]] = {
    "1-element": lambda side: side.tensor([2.0], "float32"),
    "2-elements": lambda side: side.tensor([2.0, 3.0], "float32"),
    "2d": lambda side: side.tensor([[2.0]], "float32"),
    "empty": lambda side: side.tensor([], "float32"),
    "requires-grad": lambda side: side.scalar("float32", 2.0, requires_grad=True),
}


def _op_name_without_underscore(message: str) -> str:
    """``addmm_(): argument ...`` is ``addmm(): argument ...``: the op, whichever spelling wrote it."""
    return re.sub(r"^(\w+?)_?\(\): ", r"\1(): ", message)


@pytest.mark.parametrize("kind", _NOT_A_NUMBER)
@pytest.mark.parametrize("dtype", ["int32", "float32"])
@pytest.mark.parametrize("name", _NUMBER_OPS)
@pytest.mark.parametrize("device", devices_supporting(lucid.float32))
def test_a_tensor_that_is_not_a_number_is_a_type_error(
    name: str, device: str, dtype: str, kind: str, ref: ModuleType
) -> None:
    op = OPS[name]
    if dtype not in op.dtypes:
        pytest.skip(f"{name} takes no {dtype} tensor")
    ours, theirs = Side(lucid, dtype, device), Side(ref, dtype, device)
    got = outcome(lambda: op.call(ours, _NOT_A_NUMBER[kind](ours)))
    want = outcome(lambda: op.call(theirs, _NOT_A_NUMBER[kind](theirs)))
    assert got[0] == want[0] == "raised", f"{got} vs {want}"
    assert got[1] is want[1] is TypeError
    if "invalid combination" in want[2]:
        # several signatures: the reference lists them all, and ``subtract`` is its own name there
        assert re.search(r"^\w+\(\): argument '\w+' must be Number, not Tensor$", got[2])
    else:
        assert _op_name_without_underscore(got[2]) == _op_name_without_underscore(want[2])


@pytest.mark.parametrize("device", devices_supporting(lucid.float32))
def test_a_refused_scale_factor_leaves_an_in_place_target_alone(device: str) -> None:
    x = lucid.tensor([1.0, 2.0], device=device)
    ones = lucid.ones(2, device=device)
    with pytest.raises(TypeError):
        x.add_(ones, alpha=lucid.tensor([2.0, 3.0], device=device))
    with pytest.raises(TypeError):
        x.addcmul_(ones, ones, value=lucid.tensor(2.0, device=device, requires_grad=True))
    with pytest.raises(RuntimeError):
        x.add_(ones, alpha=lucid.tensor(1 + 2j))
    assert x.tolist() == [1.0, 2.0]


# ── lerp's weight is an operand, never a scale factor ────────────────────────────────────────


@pytest.mark.parametrize("weight", ["bool", "int32", "float16", "float64", "complex64"])
@pytest.mark.parametrize(
    ("device", "dtype"),
    [(d, t) for t in ("float32", "float16", "float64") for d in devices_supporting(getattr(lucid, t))],
)
def test_a_0d_lerp_weight_is_promoted_not_read_as_a_number(
    device: str, dtype: str, weight: str, ref: ModuleType
) -> None:
    if dtype == "float16" and weight == "complex64":
        pytest.skip("the reference has no complex32 kernel for the promoted pair; Lucid has no complex32")
    ours, theirs = Side(lucid, dtype, device), Side(ref, dtype, device)
    value = 1 if weight in ("bool", "int32") else 0.5
    got = outcome(lambda: lucid.lerp(ours.ones(2), ours.ones(2), ours.scalar(weight, value)))
    want = outcome(lambda: ref.lerp(theirs.ones(2), theirs.ones(2), theirs.scalar(weight, value)))
    assert got[0] == want[0], f"{got} vs {want}"
    if got[0] == "raised":
        assert got[1] is want[1]
        return
    assert got[1] == want[1]
    assert same_values(got[2], want[2], HALF_RTOL.get(dtype, 0.0))


# ── what the card reported, pinned without the reference ─────────────────────────────────────


@pytest.mark.parametrize("device", devices_supporting(lucid.int32))
def test_a_float_tensor_alpha_is_refused_for_an_integer_result(device: str) -> None:
    x = lucid.tensor([1, 1], dtype=lucid.int32, device=device)
    for alpha in (lucid.tensor(2.5), lucid.tensor(2.0), lucid.tensor(1 + 2j)):
        with pytest.raises(RuntimeError, match="must not be a floating point number"):
            lucid.add(x, x, alpha=alpha)
    assert lucid.add(x, x, alpha=lucid.tensor(2)).tolist() == [3, 3]
    assert lucid.add(x, x, alpha=lucid.tensor(2, dtype=lucid.int8)).tolist() == [3, 3]


@pytest.mark.parametrize("device", devices_supporting(lucid.int32))
def test_addcmul_with_a_float_tensor_value_keeps_the_integer_dtype(device: str) -> None:
    x = lucid.tensor([1, 1], dtype=lucid.int32, device=device)
    out = lucid.addcmul(x, x, x, value=lucid.tensor(2.0))
    assert out.dtype is lucid.int32
    assert out.tolist() == [3, 3]


@pytest.mark.parametrize("device", devices_supporting(lucid.int32))
def test_index_add_with_a_float_tensor_alpha_answers_in_the_integer_dtype(device: str) -> None:
    base = lucid.zeros(3, dtype=lucid.int32, device=device)
    source = lucid.ones(2, dtype=lucid.int32, device=device)
    out = lucid.index_add(base, 0, lucid.tensor([0, 1], device=device), source, alpha=lucid.tensor(2.0))
    assert out.dtype is lucid.int32
    assert out.tolist() == [2, 2, 0]


@pytest.mark.parametrize("device", devices_supporting(lucid.float32))
def test_a_tensor_scale_factor_does_not_promote_the_result(device: str) -> None:
    """A float64 tensor is a number that scales a float32 result; it is not an operand."""
    x = lucid.ones(2, device=device)
    assert lucid.add(x, x, alpha=lucid.tensor(2.0, dtype=lucid.float64)).dtype is lucid.float32
    assert lucid.addcmul(x, x, x, value=lucid.tensor(2.0, dtype=lucid.float64)).dtype is lucid.float32


@pytest.mark.parametrize("device", devices_supporting(lucid.float16))
def test_autocast_does_not_cast_a_tensor_scale_factor(device: str) -> None:
    """The reference's autocast casts the tensors an op is handed, and a scale factor is a number to
    it: ``0.1`` stays ``0.1`` (a float32), not the half-rounded ``0.09998``."""
    half = lucid.float16 if device == "metal" else lucid.bfloat16
    m = lucid.ones((2, 2), device=device)
    alpha = lucid.tensor(0.1, device=device)
    with autocast(device_type=device, dtype=half):
        got = lucid.addmm(m, m, m, alpha=alpha)
        want = lucid.addmm(m, m, m, alpha=alpha.item())
    assert got.dtype is want.dtype is half
    assert got.tolist() == want.tolist()
