"""A Python number beside a half tensor, through every spelling.

``float16`` / ``bfloat16`` times, over or floor-divided by a Python number
— or by a 0-d float32 tensor that does not require grad — reads the number
at float32 and rounds the result once (the engine's op-math, ENG-21), so
``ones(f16) / 1e5`` is not 0 and ``full(1e-3, f16) * 1e5`` is not inf.  Every
other op rounds the number to the half dtype first, as the reference does.
The operator, ``lucid.<op>`` and ``Tensor.<op>`` must all reach the same
values.

The reference's CPU backend applies op-math only with the number on the
right; with the number on the left (``s / t``) and for floor division the
expected values come from the reference's own float32 arithmetic rounded
once, which is what Lucid does.  For floor division that is the exact
float32 floor (LCD-159), so f16 ``1.0 // 0.1`` is 9.

The last group pins integer and complex numbers: an integer past 2**53
reaches an int64 tensor exactly, and a complex number reaches a complex
tensor as itself.
"""

import math
import operator
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import lucid

pytestmark = pytest.mark.parity

_HALF = {"f16": lucid.float16, "bf16": lucid.bfloat16}
_SCALARS: list[float | int] = [0.1, 1e5, 1e-5, 0.3, 1.1, 100000]
_VALUES = [1.0, -1.0, 3.09375, 0.3, -2.5, 7.0, 1e-3, -6e-5, 300.0, 12.5, 5.5]

#: name -> (operator, lucid function, Tensor method or None, number on the left)
_OPS: dict[str, tuple[Callable[[Any, Any], Any], Any, str | None, bool]] = {
    "*": (operator.mul, lucid.mul, "mul", False),
    "/": (operator.truediv, lucid.div, "div", False),
    "rdiv": (operator.truediv, lucid.div, None, True),
    "//": (operator.floordiv, lucid.floor_divide, "floor_divide", False),
    "+": (operator.add, lucid.add, "add", False),
    "-": (operator.sub, lucid.sub, "sub", False),
    "**": (operator.pow, lucid.pow, "pow", False),
}


def _ref_half(ref: ModuleType, name: str) -> Any:
    return ref.float16 if name == "f16" else ref.bfloat16


def _same(got: list[float], want: list[float]) -> bool:
    """Elementwise equality in which NaN matches NaN (``(-1) ** 0.1``)."""
    return len(got) == len(want) and all(
        g == w or (math.isnan(g) and math.isnan(w)) for g, w in zip(got, want, strict=True)
    )


def _expected(ref: ModuleType, half: str, op: str, s: float | int) -> list[float] | None:
    rt = ref.tensor(_VALUES, dtype=_ref_half(ref, half))
    if op == "rdiv":
        out = (ref.tensor(float(s), dtype=ref.float32) / rt.float()).to(rt.dtype)
    elif op == "//":
        out = rt.float() // ref.tensor(float(s), dtype=ref.float32)
        out = out.to(rt.dtype)
    else:
        try:
            out = _OPS[op][0](rt, s)
        except RuntimeError:
            if op == "**":
                return None
            # The reference refuses a number its half dtype cannot hold
            # (float16 + 100000); Lucid rounds it to inf.  Not a promotion
            # question, so there is nothing to pin here.
            pytest.skip("the reference refuses this number in the half dtype")
        assert out.dtype == rt.dtype
    return [float(v) for v in out.float().tolist()]


def _spellings(op: str, t: lucid.Tensor, s: Any) -> dict[str, Callable[[], Any]]:
    dunder, fn, method, left = _OPS[op]
    if left:
        return {"operator": lambda: dunder(s, t), "lucid.fn": lambda: fn(s, t)}
    calls: dict[str, Callable[[], Any]] = {
        "operator": lambda: dunder(t, s),
        "lucid.fn": lambda: fn(t, s),
    }
    if method is not None:
        calls["Tensor.method"] = lambda: getattr(t, method)(s)
    return calls


@pytest.mark.parametrize("half", list(_HALF))
@pytest.mark.parametrize("op", list(_OPS))
@pytest.mark.parametrize("s", _SCALARS, ids=[repr(s) for s in _SCALARS])
def test_python_number_matches_the_reference(
    half: str, op: str, s: float | int, device: str, ref: ModuleType
) -> None:
    expected = _expected(ref, half, op, s)
    t = lucid.tensor(_VALUES, dtype=lucid.float32, device=device).to(_HALF[half])
    for spelling, call in _spellings(op, t, s).items():
        if expected is None:
            with pytest.raises(RuntimeError):
                call()
        else:
            out = call()
            assert out.dtype == _HALF[half], spelling
            assert _same(out.to(lucid.float32).tolist(), expected), spelling


@pytest.mark.parametrize("half", list(_HALF))
@pytest.mark.parametrize("op", ["*", "/", "//"])
@pytest.mark.parametrize("s", [0.1, 1e5, 1e-5])
def test_a_0d_float32_tensor_takes_the_same_path(
    half: str, op: str, s: float, device: str, ref: ModuleType
) -> None:
    expected = _expected(ref, half, op, s)
    t = lucid.tensor(_VALUES, dtype=lucid.float32, device=device).to(_HALF[half])
    z = lucid.tensor(s, dtype=lucid.float32, device=device)
    for spelling, call in _spellings(op, t, z).items():
        out = call()
        assert out.dtype == _HALF[half], spelling
        assert _same(out.to(lucid.float32).tolist(), expected), spelling


@pytest.mark.parametrize("half", list(_HALF))
@pytest.mark.parametrize("op", ["*", "/", "rdiv"])
def test_a_0d_operand_that_requires_grad_keeps_op_math_and_gets_f32_grad(
    half: str, op: str, device: str, ref: ModuleType
) -> None:
    expected = _expected(ref, half, op, 0.1)
    assert expected is not None
    t = lucid.tensor(_VALUES, dtype=lucid.float32, device=device, requires_grad=True).to(
        _HALF[half]
    )
    t.retain_grad()
    z = lucid.tensor(0.1, dtype=lucid.float32, device=device, requires_grad=True)
    out = t * z if op == "*" else (t / z if op == "/" else z / t)
    out.sum().backward()
    rt = ref.tensor(_VALUES, dtype=_ref_half(ref, half)).float()
    rz = ref.tensor(0.1, dtype=ref.float32, requires_grad=True)
    rout_w = rt * rz if op == "*" else (rt / rz if op == "/" else rz / rt)
    rout_w.sum().backward()
    assert out.dtype == _HALF[half]
    assert _same(out.detach().to(lucid.float32).tolist(), expected)
    assert z.grad is not None and rz.grad is not None
    assert z.grad.dtype == lucid.float32
    assert z.grad.shape == ()
    assert z.grad.item() == pytest.approx(rz.grad.item(), rel=1e-6, abs=1e-6)
    assert t.grad is not None
    assert t.grad.dtype == _HALF[half]


@pytest.mark.parametrize("half", list(_HALF))
@pytest.mark.parametrize("op,s", [("*", 0.1), ("*", 1e5), ("/", 1e5), ("/", 0.3)], ids=str)
def test_in_place_forms_take_the_same_path(
    half: str, op: str, s: float, device: str, ref: ModuleType
) -> None:
    expected = _expected(ref, half, op, s)
    name = "mul_" if op == "*" else "div_"
    augmented = operator.imul if op == "*" else operator.itruediv
    base = lucid.tensor(_VALUES, dtype=lucid.float32, device=device).to(_HALF[half])
    by_method = base.clone()
    getattr(by_method, name)(s)
    by_operator = augmented(base.clone(), s)
    for out in (by_method, by_operator):
        assert out.dtype == _HALF[half]
        assert _same(out.to(lucid.float32).tolist(), expected)


def test_the_reference_values_the_rounding_got_wrong(device: str) -> None:
    ones = lucid.ones(2, dtype=lucid.float16, device=device)
    assert (ones / 1e5).tolist() == [1.0013580322265625e-05] * 2
    big = lucid.full((2,), 1e-3, dtype=lucid.float16, device=device)
    assert (big * 1e5).tolist() == [100.0625] * 2
    b = lucid.tensor([3.09375], dtype=lucid.float32, device=device)
    assert (b.to(lucid.bfloat16) * 0.1).tolist() == [0.30859375]


_EXACT: list[tuple[str, lucid.dtype, Callable[[Any, Any], Any], Any]] = [
    ("int64 + (2**60 + 1)", lucid.int64, operator.add, 2**60 + 1),
    ("int64 * (2**53 + 1)", lucid.int64, operator.mul, 2**53 + 1),
    ("int64 - (2**62 + 3)", lucid.int64, operator.sub, 2**62 + 3),
    ("int8 + 300", lucid.int8, operator.add, 300),
    ("complex64 + (2+3j)", lucid.complex64, operator.add, 2 + 3j),
    ("complex64 * (2+3j)", lucid.complex64, operator.mul, 2 + 3j),
]


@pytest.mark.parametrize("dt,op,s", [c[1:] for c in _EXACT], ids=[c[0] for c in _EXACT])
def test_integer_and_complex_numbers_arrive_exactly(
    dt: lucid.dtype,
    op: Callable[[Any, Any], Any],
    s: Any,
    device: str,
    ref: ModuleType,
) -> None:
    base = [1, -2] if dt is not lucid.complex64 else [1 + 1j, -2 + 0.5j]
    rt = ref.tensor(base, dtype=getattr(ref, str(dt).removeprefix("lucid.")))
    want = op(rt, s)
    t = lucid.tensor(base, dtype=dt, device=device)
    fn = {operator.add: lucid.add, operator.mul: lucid.mul, operator.sub: lucid.sub}[op]
    for out in (op(t, s), fn(t, s), op(s, t) if op is not operator.sub else op(t, s)):
        assert out.dtype == dt
        assert out.tolist() == want.tolist()
