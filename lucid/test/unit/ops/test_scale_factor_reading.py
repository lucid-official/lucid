"""A Python scale factor, read as the reference reads it (LCD-1076, LCD-1078).

``alpha`` and ``beta`` of the ``add`` family and the scaled products, ``value`` of ``addcmul`` /
``addcdiv``, the ``weight`` of ``lerp`` and the exponent of an in-place ``pow`` are Python numbers
a kernel converts to the tensor's element type.  Lucid read them its own way: a complex number lost
its imaginary part, a bool was refused where the reference counts it as 0 or 1, and a number the
element type cannot hold either wrapped (``300.7`` on an ``int8`` tensor was 46) or leaked Python's
own ``OverflowError`` / ``ValueError`` (infinity, NaN).  ``lucid/_scalar_cast.py`` now owns the two
rules the reference has, and this sweep holds every spelling of every op to them:

* ``add``, ``sub``, ``rsub`` (and ``addr`` on an integral tensor) judge the scalar by its *type*:
  a bool is refused unless the tensor is bool, a float unless it is floating, a complex number
  unless it is complex (``0j`` included).  What passes must still fit the element type, and
  ``sub`` / ``rsub`` check the *negated* scalar, so ``128`` is an ``int8`` alpha there and
  ``-128`` is not;
* every other scaled op converts the scalar (``Scalar::to<T>``): a bool is 0 or 1, a float
  becomes an integer on an integer tensor, a complex number with no imaginary part is its real
  part, and a number the type cannot hold is refused as ``value cannot be converted to type <T>
  without overflow``.  A bool tensor reads anything as ``value != 0``.

The sweep is op x tensor dtype x scalar x device and is checked against the reference, not a
table: a cell either is refused by both, with one exception class and one message, or is answered
by both, with one dtype and one value.  A cell is left out only where the reference answers
differently for a reason that is not this one:

* a Python int outside ``[-2**63, 2**64)``: the reference refuses it while parsing the argument,
  for every op that takes a scalar (``test_a_python_int_past_64_bits_is_refused``);
* a bool tensor of a product (``addmm`` ...), ``addcdiv`` / ``lerp`` of an integer tensor, a
  bool ``sub``: the two name the unsupported dtype differently; ``lerp`` of a complex tensor,
  which has no kernel yet;
* the int ``-2**63`` as the ``alpha`` of ``sub`` on a floating tensor: the reference negates it as
  an int64, which wraps, and answers a wrong sign (it is not among the scalars below);
* a complex tensor with an infinite scalar: complex arithmetic with infinity is its own gap
  (Metal's ``add(alpha=inf)`` is NaN in both parts, the CPU's ``addcmul`` has an infinite real
  part), whatever the scalar's reading.

The reference's float32 accumulating ops (``addmm``, ``addbmm``, ``baddbmm``, ``addcmul``,
``addcdiv``, ``lerp``) read a half tensor's scalar as a float and round once, at the end, and so do
these (LCD-1087; ``test_half_opmath_accumulating.py`` holds them to the reference on values).  A
cell of one of them is compared to within one half ulp, the one place the two may differ: the
reference's ``addbmm`` rounds after each batch, and a product may be summed in another order.
"""

import math
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import lucid
from lucid.test._fixtures.devices import device_dtype_params, devices_supporting
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

_SCALARS: dict[str, complex] = {
    "1j": 1j,
    "0j": 0j,
    "1+2j": 1 + 2j,
    "1+0j": 1 + 0j,
    "300+0j": 300 + 0j,
    "nan+0j": complex(math.nan, 0),
    "True": True,
    "False": False,
    "2": 2,
    "127": 127,
    "128": 128,
    "-128": -128,
    "300": 300,
    "32768": 32768,
    "70000": 70000,
    "2**31": 2**31,
    "2**63-1": 2**63 - 1,
    "2**63": 2**63,
    "2.5": 2.5,
    "300.7": 300.7,
    "127.5": 127.5,
    "2**63f": float(2**63),
    "inf": math.inf,
    "-inf": -math.inf,
    "nan": math.nan,
    "1e30": 1e30,
    "3.5e38": 3.5e38,
}
_ALL = tuple(_SCALARS)


def _cells() -> list[Any]:
    return [
        pytest.param(name, dtype, device, scalar, id=f"{name}-{dtype}-{device}-{scalar}")
        for name, op in OPS.items()
        for dtype in op.dtypes
        for device in devices_supporting(getattr(lucid, dtype))
        for scalar in _ALL
        if not (dtype in COMPLEX and math.isinf(complex(_SCALARS[scalar]).real))
    ]




@pytest.mark.parametrize(("name", "dtype", "device", "scalar"), _cells())
def test_scalar_is_read_as_the_reference_reads_it(
    name: str, dtype: str, device: str, scalar: str, ref: ModuleType
) -> None:
    value = _SCALARS[scalar]
    op = OPS[name]
    got = outcome(lambda: op.call(Side(lucid, dtype, device), value))
    want = outcome(lambda: op.call(Side(ref, dtype, device), value))
    assert got[0] == want[0], f"{got} vs {want}"
    if got[0] == "raised":
        assert got[1] is want[1]
        assert got[2] == want[2], f"{got[2]!r} vs {want[2]!r}"
        return
    assert got[1] == want[1]
    rtol = HALF_RTOL[dtype] if op.op_math and dtype in HALF else 0.0
    assert same_values(got[2], want[2], rtol), f"{got[2]} vs {want[2]}"


# ── what the reference raises, pinned without the reference ──────────────────────────────────


@pytest.mark.parametrize("device", devices_supporting(lucid.float32))
def test_add_with_a_complex_alpha_on_a_float_tensor_is_refused(device: str) -> None:
    x = lucid.tensor([1.0, 2.0], device=device)
    with pytest.raises(
        RuntimeError,
        match="For non-complex input tensors, argument alpha must not be a complex number.",
    ):
        lucid.add(x, x, alpha=1j)


@pytest.mark.parametrize("device", devices_supporting(lucid.int32))
def test_add_with_a_complex_alpha_on_an_integer_tensor_is_refused(device: str) -> None:
    x = lucid.tensor([1, 2], device=device)
    with pytest.raises(
        RuntimeError,
        match="For integral input tensors, argument alpha must not be a floating point number.",
    ):
        lucid.add(x, x, alpha=1j)


@pytest.mark.parametrize("device", devices_supporting(lucid.float32))
def test_addmm_with_a_complex_beta_on_a_float_tensor_is_refused(device: str) -> None:
    m = lucid.ones((2, 2), device=device)
    with pytest.raises(
        RuntimeError, match="value cannot be converted to type float without overflow"
    ):
        lucid.addmm(m, m, m, beta=1j)


@pytest.mark.parametrize("device", devices_supporting(lucid.float32))
def test_a_refused_in_place_write_leaves_the_tensor_alone(device: str) -> None:
    x = lucid.tensor([1.0, 2.0], device=device)
    with pytest.raises(RuntimeError):
        x.add_(lucid.ones(2, device=device), alpha=1j)
    with pytest.raises(RuntimeError):
        x.addcmul_(lucid.ones(2, device=device), lucid.ones(2, device=device), value=1j)
    assert x.tolist() == [1.0, 2.0]


def test_a_complex_scalar_with_no_imaginary_part_keeps_a_real_result_real() -> None:
    """``addcmul``'s ``value`` is converted, not promoted: a float tensor stays float."""
    out = lucid.addcmul(lucid.ones(2), lucid.ones(2), lucid.ones(2), value=2 + 0j)
    assert out.dtype is lucid.float32
    assert out.tolist() == [3.0, 3.0]


# A scalar the element type cannot hold is refused by name, not wrapped or leaked as a Python error.

_CONVERTED = [
    ("int8", 300.7, "int8_t"),
    ("int8", 300, "int8_t"),
    ("int8", -129, "int8_t"),
    ("int8", 127.9, "int8_t"),
    ("int8", 1e30, "int8_t"),
    ("int16", 32768, "int16_t"),
    ("int16", math.nan, "int16_t"),
    ("int32", 2**31, "int"),
    ("int32", 2.0**31, "int"),
    ("int32", math.inf, "int"),
    ("int64", 2**63, "int64_t"),
    ("int64", -math.inf, "int64_t"),
    ("float32", 3.5e38, "float"),
    ("float32", -1e39, "float"),
]
#: ``add`` judges a float on an integer tensor by its type first, so only what it may hold.
_BY_TYPE = [
    ("int8", 300, "int8_t"),
    ("int16", 32768, "int16_t"),
    ("int32", 2**31, "int"),
    ("int64", 2**63, "int64_t"),
    ("float32", 3.5e38, "float"),
    ("float32", -1e39, "float"),
]
_DEVICES = devices_supporting(lucid.float32)


def _scaled_calls(m: Any, scalar: float) -> dict[str, Callable[[], Any]]:
    return {
        "addmm": lambda: lucid.addmm(m, m, m, alpha=scalar),
        "addcmul": lambda: lucid.addcmul(m, m, m, value=scalar),
        "index_add": lambda: lucid.index_add(
            m, 0, lucid.tensor([0], device=m.device), m[:1], alpha=scalar
        ),
        "add": lambda: lucid.add(m, m, alpha=scalar),
    }


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("fn", ["addmm", "addcmul", "index_add"])
@pytest.mark.parametrize(("dtype", "scalar", "named"), _CONVERTED)
def test_a_converted_scalar_the_type_cannot_hold_is_refused_by_name(
    fn: str, dtype: str, scalar: float, named: str, device: str
) -> None:
    m = lucid.ones((2, 2), dtype=getattr(lucid, dtype), device=device)
    with pytest.raises(
        RuntimeError, match=f"^value cannot be converted to type {named} without overflow$"
    ):
        _scaled_calls(m, scalar)[fn]()


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize(("dtype", "scalar", "named"), _BY_TYPE)
def test_an_add_alpha_the_type_cannot_hold_is_refused_by_name(
    dtype: str, scalar: float, named: str, device: str
) -> None:
    m = lucid.ones((2, 2), dtype=getattr(lucid, dtype), device=device)
    with pytest.raises(
        RuntimeError, match=f"^value cannot be converted to type {named} without overflow$"
    ):
        _scaled_calls(m, scalar)["add"]()


@pytest.mark.parametrize("device", devices_supporting(lucid.float16))
def test_a_half_scalar_is_judged_by_the_type_the_op_reads(device: str) -> None:
    """``add`` and ``addmv`` read it as ``c10::Half``; ``addmm`` and ``addcmul`` as ``float``."""
    v = lucid.ones(2, dtype=lucid.float16, device=device)
    m = lucid.ones((2, 2), dtype=lucid.float16, device=device)
    with pytest.raises(RuntimeError, match="type c10::Half without overflow"):
        lucid.add(v, v, alpha=70000)
    with pytest.raises(RuntimeError, match="type c10::Half without overflow"):
        lucid.addmv(v, m, v, alpha=70000.5)
    with pytest.raises(RuntimeError, match="type float without overflow"):
        lucid.addmm(m, m, m, alpha=3.5e38)
    assert lucid.addmm(m, m, m, alpha=70000).tolist() == [[math.inf] * 2] * 2
    assert lucid.add(v, v, alpha=math.inf).tolist() == [math.inf] * 2


@pytest.mark.parametrize("device", devices_supporting(lucid.int8))
def test_subtracting_an_int8_alpha_checks_the_negated_scalar(device: str) -> None:
    """The reference subtracts by adding ``-alpha``: ``128`` fits an int8 there, ``-128`` does not."""
    x = lucid.ones(2, dtype=lucid.int8, device=device)
    assert lucid.sub(x, x, alpha=128).tolist() == [-127, -127]
    assert lucid.rsub(x, x, alpha=128).tolist() == [-127, -127]
    assert lucid.add(x, x, alpha=-128).tolist() == [-127, -127]
    with pytest.raises(RuntimeError, match="type int8_t without overflow"):
        lucid.sub(x, x, alpha=-128)
    with pytest.raises(RuntimeError, match="type int8_t without overflow"):
        lucid.add(x, x, alpha=128)


# A bool counts as 0 or 1 for a converted scalar, and is refused by name for the by-type ones.


@pytest.mark.parametrize(
    ("device", "dtype"),
    device_dtype_params([lucid.int8, lucid.int32, lucid.float16, lucid.float32]),
)
def test_a_bool_scale_factor_is_one_or_zero_for_the_scaled_products(
    device: str, dtype: lucid.dtype
) -> None:
    m = lucid.ones((2, 2), dtype=dtype, device=device)
    assert lucid.addmm(m, m, m, alpha=True).tolist() == [[3, 3], [3, 3]]
    assert lucid.addmm(m, m, m, alpha=False).tolist() == [[1, 1], [1, 1]]
    assert lucid.addmm(m, m, m, beta=True).tolist() == [[3, 3], [3, 3]]
    assert lucid.addmm(m, m, m, beta=False).tolist() == [[2, 2], [2, 2]]
    assert lucid.baddbmm(m[None], m[None], m[None], alpha=True).tolist() == [[[3, 3], [3, 3]]]
    assert lucid.addcmul(m, m, m, value=True).tolist() == [[2, 2], [2, 2]]


@pytest.mark.parametrize("device", _DEVICES)
def test_a_bool_alpha_is_still_refused_where_the_reference_judges_the_type(device: str) -> None:
    x = lucid.ones(2, device=device)
    with pytest.raises(RuntimeError, match="Boolean alpha only supported for Boolean results."):
        lucid.add(x, x, alpha=True)
    with pytest.raises(RuntimeError, match="Boolean beta only supported for Boolean results."):
        lucid.addr(lucid.ones((2, 2), device=device), x, x, beta=False)


@pytest.mark.parametrize("device", devices_supporting(lucid.bool_))
def test_a_bool_tensor_reads_any_scalar_as_nonzero(device: str) -> None:
    """``index_add`` on a bool tensor takes a float or a complex ``alpha``: it is ``alpha != 0``."""
    zeros = lucid.zeros(3, dtype=lucid.bool_, device=device)
    source = lucid.ones(2, dtype=lucid.bool_, device=device)
    index = lucid.tensor([0, 1], device=device)
    for alpha, kept in [(1.5, True), (0.0, False), (math.nan, True), (1j, True), (0j, False)]:
        out = lucid.index_add(zeros, 0, index, source, alpha=alpha)
        assert out.dtype is lucid.bool_
        assert out.tolist() == [kept, kept, False]


_POWER_OVERFLOWS = {
    "int8": ("int8_t", 8),
    "int16": ("int16_t", 16),
    "int32": ("int", 32),
    "int64": ("int64_t", 64),
}


@pytest.mark.parametrize("dtype", _POWER_OVERFLOWS)
def test_an_in_place_power_names_the_type_the_exponent_overflows(dtype: str) -> None:
    named, bits = _POWER_OVERFLOWS[dtype]
    x = lucid.ones(2, dtype=getattr(lucid, dtype))
    with pytest.raises(RuntimeError, match=f"^value cannot be converted to type {named} without"):
        x.pow_(1 << (bits - 1))
    # one past an int64 is an int64 refusal whatever the base
    with pytest.raises(RuntimeError, match="^value cannot be converted to type int64_t without"):
        x.pow_(2**63)


@pytest.mark.parametrize("device", devices_supporting(lucid.float16))
def test_a_half_power_names_the_storage_type(device: str) -> None:
    half = lucid.ones(2, dtype=lucid.float16, device=device)
    brain = lucid.ones(2, dtype=lucid.bfloat16, device=device)
    with pytest.raises(RuntimeError, match="type c10::Half without overflow"):
        half.pow_(70000)
    with pytest.raises(RuntimeError, match="type c10::BFloat16 without overflow"):
        brain.pow_(3.5e38)


@pytest.mark.parametrize("scalar", ["2", None, [2]], ids=["str", "none", "list"])
@pytest.mark.parametrize("fn", ["add", "addmm", "addcmul", "index_add"])
def test_a_scale_factor_that_is_not_a_number_is_refused_as_a_type_error(
    fn: str, scalar: object, ref: ModuleType
) -> None:
    """``add(): argument 'alpha' must be Number, not str``: the reference refuses it where it parses
    the arguments.  ``addmm`` has several signatures there and lists them all; its class is held."""

    def calls(lib: Any) -> dict[str, Callable[[], Any]]:
        m = lib.ones((2, 2))
        return {
            "add": lambda: lib.add(m, m, alpha=scalar),
            "addmm": lambda: lib.addmm(m, m, m, beta=scalar),
            "addcmul": lambda: lib.addcmul(m, m, m, value=scalar),
            "index_add": lambda: lib.index_add(m, 0, lib.tensor([0]), m[:1], alpha=scalar),
        }

    got = outcome(calls(lucid)[fn])
    want = outcome(calls(ref)[fn])
    assert got[0] == want[0] == "raised"
    assert got[1] is want[1] is TypeError
    if "invalid combination" in want[2]:
        assert got[2].startswith(f"{fn}(): argument ")
    else:
        assert got[2] == want[2]


# ── what still works: the scalar a tensor can hold ──────────────────────────────────────────


@pytest.mark.parametrize("device", devices_supporting(lucid.complex64))
def test_a_complex_alpha_on_a_complex_tensor_scales_it(device: str) -> None:
    c = lucid.tensor([1 + 1j, 2 - 1j], dtype=lucid.complex64, device=device)
    d = lucid.tensor([0.5, 3j], dtype=lucid.complex64, device=device)
    assert lucid.add(c, d, alpha=1j).tolist() == [1 + 1.5j, -1 - 1j]
    assert lucid.sub(c, d, alpha=1j).tolist() == [1 + 0.5j, 5 - 1j]
    assert lucid.rsub(c, d, alpha=1j).tolist() == [1.5 - 1j, -1 + 1j]
    out = lucid.addcmul(c, d, d, value=1j)
    assert out.tolist() == [1 + 1.25j, 2 - 10j]


@pytest.mark.parametrize("device", devices_supporting(lucid.complex128))
def test_complex_alpha_on_complex128_keeps_double_precision(device: str, ref: ModuleType) -> None:
    values = [1 + 1j, 2 - 1j]
    other = [0.5, 3j]
    got = lucid.add(
        lucid.tensor(values, dtype=lucid.complex128, device=device),
        lucid.tensor(other, dtype=lucid.complex128, device=device),
        alpha=0.1 + 0.2j,
    )
    want = ref.add(
        ref.tensor(values, dtype=ref.complex128),
        ref.tensor(other, dtype=ref.complex128),
        alpha=0.1 + 0.2j,
    )
    assert got.tolist() == want.tolist()


@pytest.mark.parametrize("device", devices_supporting(lucid.float32))
def test_a_real_alpha_on_a_real_tensor_is_unchanged(device: str) -> None:
    x = lucid.tensor([1.0, 2.0], device=device)
    assert lucid.add(x, x, alpha=2).tolist() == [3.0, 6.0]
    assert lucid.sub(x, x, alpha=0.5).tolist() == [0.5, 1.0]
    assert lucid.rsub(x, 10.0, alpha=2).tolist() == [8.0, 6.0]
    assert lucid.rsub(x, 10).tolist() == [9.0, 8.0]


@pytest.mark.parametrize("device", devices_supporting(lucid.float16))
@pytest.mark.parametrize(
    "case",
    [
        lambda lib, kw: lib.addmm(*[lib.ones((2, 2), dtype=lib.bfloat16, **kw)] * 3, alpha=300.7),
        lambda lib, kw: lib.lerp(*[lib.ones(2, dtype=lib.float16, **kw)] * 2, 70000),
    ],
    ids=["addmm-bfloat16-300.7", "lerp-float16-70000"],
)
def test_a_half_scalar_is_read_as_float_by_the_accumulating_ops(
    case: Callable[[Any, dict[str, str]], Any], device: str, ref: ModuleType
) -> None:
    """``addmm`` is 604 for 300.7 (a half-rounded 300 gave 600) and ``lerp`` answers 1 for 70000
    (a half-rounded infinity gave NaN): the scalar is a float until the answer is stored."""
    assert case(lucid, {"device": device}).tolist() == case(ref, {}).tolist()


# ── known gaps, pinned so the day they close is noticed ──────────────────────────────────────


@pytest.mark.xfail(
    strict=True,
    reason=(
        "LCD-1086: a Python int outside [-2**63, 2**64) is refused by the reference's "
        "argument parser for every scalar op (OverflowError); Lucid takes it as a float"
    ),
)
@pytest.mark.parametrize("scalar", [2**64, -(2**63) - 1])
def test_a_python_int_past_64_bits_is_refused(scalar: int) -> None:
    with pytest.raises(OverflowError):
        lucid.mul(lucid.ones(2), scalar)
