"""``div``'s ``rounding_mode``, through every spelling (LCD-485).

``lucid.div``, ``Tensor.div``, ``Tensor.div_`` and ``lucid.divide`` took no
``rounding_mode``, so ``div(t, 0.5, rounding_mode="floor")`` raised
``TypeError``.  ``"floor"`` is ``floor_divide``'s path and ``"trunc"``
rounds toward zero; the exact values, for every dtype and both devices,
are held in :mod:`.test_divmod_exact`.  This holds what is not a value:
the dtype each mode promotes to, the refusals, the in-place form and the
gradient, each against the reference.
"""

from collections.abc import Callable
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import lucid
from lucid._C import engine as _C_engine

pytestmark = pytest.mark.parity

_MODES = [None, "trunc", "floor"]

#: name -> one call, for ``lib`` either library.
_SPELLINGS: dict[str, Callable[[Any, Any, Any, str | None], Any]] = {
    "div": lambda lib, a, b, m: lib.div(a, b, rounding_mode=m),
    "divide": lambda lib, a, b, m: lib.divide(a, b, rounding_mode=m),
    "Tensor.div": lambda lib, a, b, m: a.div(b, rounding_mode=m),
    "Tensor.divide": lambda lib, a, b, m: a.divide(b, rounding_mode=m),
}

#: Operand pairs whose result dtype differs by mode: ``(a, a_dtype, b, b_dtype)``
#: with ``None`` for a Python number.
_PAIRS: dict[str, tuple[list[Any], str, Any, str | None]] = {
    "int/int": ([7, -7, 9, -9, 0], "int64", [2, 2, -4, -4, 3], "int64"),
    "int8/int64": ([7, -7, 100, -128], "int8", [2, -3, 7, -1], "int64"),
    "int/number": ([7, -7, 9, -9], "int64", 2, None),
    "int/float": ([7, -7, 9, -9], "int64", 2.5, None),
    "bool/int": ([True, False, True], "bool", [2, 3, -1], "int32"),
    "float/float": ([7.5, -7.5, 1.0, -0.5], "float32", [2.0, 2.0, 0.1, 3.0], "float32"),
    "half/number": ([7.5, -7.5, 1000.0, 0.1], "float16", 0.1, None),
}


def _operand(lib: ModuleType, values: Any, dtype: str | None, device: str) -> Any:
    if dtype is None:
        return values
    if lib is lucid:
        return lucid.tensor(values, dtype=getattr(lucid, dtype), device=device)
    return lib.tensor(values, dtype=getattr(lib, dtype))


@pytest.mark.parametrize("pair", list(_PAIRS))
@pytest.mark.parametrize("mode", _MODES, ids=str)
@pytest.mark.parametrize("spelling", list(_SPELLINGS))
def test_each_mode_promotes_as_the_reference(
    ref: ModuleType, device: str, spelling: str, mode: str | None, pair: str
) -> None:
    av, adt, bv, bdt = _PAIRS[pair]
    call = _SPELLINGS[spelling]
    got = call(lucid, _operand(lucid, av, adt, device), _operand(lucid, bv, bdt, device), mode)
    want = call(ref, _operand(ref, av, adt, device), _operand(ref, bv, bdt, device), mode)
    assert str(got.dtype).split(".")[-1] == str(want.dtype).split(".")[-1]
    np.testing.assert_array_equal(got.numpy(), want.numpy())


def test_rounding_mode_is_keyword_only(ref: ModuleType) -> None:
    for lib in (lucid, ref):
        with pytest.raises(TypeError):
            lib.div(lib.tensor([7]), 2, "floor")


@pytest.mark.parametrize("spelling", list(_SPELLINGS))
def test_an_unknown_mode_is_refused(ref: ModuleType, device: str, spelling: str) -> None:
    call = _SPELLINGS[spelling]
    with pytest.raises(RuntimeError, match="rounding_mode"):
        call(ref, ref.tensor([7.0]), 2, "round")
    # A value error too, which is what an unknown option is.
    with pytest.raises(_C_engine.InvalidArgument, match="found 'round'"):
        call(lucid, lucid.tensor([7.0], device=device), 2, "round")


@pytest.mark.parametrize("mode", ["trunc", "floor"])
@pytest.mark.parametrize("values", [[True, False], [1 + 1j, 2 - 1j]], ids=["bool", "complex"])
def test_bool_and_complex_have_no_rounded_quotient(
    ref: ModuleType, device: str, mode: str, values: list[Any]
) -> None:
    with pytest.raises(NotImplementedError):
        ref.div(ref.tensor(values), ref.tensor(values), rounding_mode=mode)
    a = lucid.tensor(values, device=device)
    with pytest.raises(NotImplementedError):
        lucid.div(a, a, rounding_mode=mode)


# ── in place ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", ["trunc", "floor"])
@pytest.mark.parametrize("divisor", [2, -3], ids=str)
def test_an_integer_tensor_rounds_in_place(
    ref: ModuleType, device: str, mode: str, divisor: int
) -> None:
    values = [7, -7, 9, -9, 0]
    x = lucid.tensor(values, device=device)
    same = x
    out = x.div_(divisor, rounding_mode=mode)
    want = ref.tensor(values).div_(divisor, rounding_mode=mode)
    assert out is same and x.dtype == lucid.int64
    assert x.tolist() == want.tolist()


@pytest.mark.parametrize("mode", _MODES, ids=str)
def test_an_integer_tensor_cannot_hold_a_float_quotient(
    ref: ModuleType, device: str, mode: str | None
) -> None:
    divisor = 2 if mode is None else 2.5
    with pytest.raises(RuntimeError, match="can't be cast"):
        ref.tensor([7, -7]).div_(divisor, rounding_mode=mode)
    x = lucid.tensor([7, -7], device=device)
    with pytest.raises(RuntimeError, match="can't be cast"):
        x.div_(divisor, rounding_mode=mode)
    assert x.tolist() == [7, -7]


# ── gradients ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", ["trunc", "floor"])
def test_a_rounded_quotient_has_a_zero_gradient(ref: ModuleType, device: str, mode: str) -> None:
    av, bv = [7.5, -7.5, 0.5, 3.0], [2.0]
    a = lucid.tensor(av, device=device, requires_grad=True)
    b = lucid.tensor(bv, device=device, requires_grad=True)
    ra, rb = ref.tensor(av, requires_grad=True), ref.tensor(bv, requires_grad=True)
    out = lucid.div(a, b, rounding_mode=mode)
    rout = ref.div(ra, rb, rounding_mode=mode)
    assert out.requires_grad and rout.requires_grad
    out.sum().backward()
    rout.sum().backward()
    assert a.grad is not None and b.grad is not None
    np.testing.assert_array_equal(np.abs(a.grad.numpy()), ra.grad.numpy())
    np.testing.assert_array_equal(np.abs(b.grad.numpy()), rb.grad.numpy())


@pytest.mark.parametrize("mode", ["trunc", "floor"])
def test_a_rounded_quotient_saves_neither_operand(device: str, mode: str) -> None:
    """Its gradient reads neither, so a later in-place write is allowed.

    ``trunc`` of a recorded ``div`` saved both operands, and ``y.mul_``
    after it was refused, where the reference accepts it.
    """
    w = lucid.tensor([1.5, -2.5], device=device, requires_grad=True)
    y = w * 1.0
    q = lucid.div(y, 2.0, rounding_mode=mode)
    y.mul_(3.0)
    (q.sum() + y.sum()).backward()
    assert w.grad is not None and w.grad.tolist() == [3.0, 3.0]


@pytest.mark.parametrize("mode", ["trunc", "floor"])
def test_rounding_in_place_beside_a_live_view(ref: ModuleType, mode: str) -> None:
    w = lucid.tensor([1.5, -2.5], requires_grad=True)
    rw = ref.tensor([1.5, -2.5], requires_grad=True)
    y, ry = w * 1.0, rw * 1.0
    view, rview = y[0:1], ry[0:1]
    y.div_(lucid.tensor([0.5, 2.0]), rounding_mode=mode)
    ry.div_(ref.tensor([0.5, 2.0]), rounding_mode=mode)
    assert y.tolist() == ry.tolist() and view.tolist() == rview.tolist()
    y.sum().backward()
    ry.sum().backward()
    assert w.grad is not None and w.grad.tolist() == rw.grad.tolist()


def test_tensor_divide_and_true_divide_methods(ref: ModuleType, device: str) -> None:
    """LCD-507: ``Tensor.divide``, ``Tensor.divide_``, and ``Tensor.true_divide`` match reference."""
    a = lucid.tensor([7, -7, 8], dtype=lucid.int64, device=device)
    ra = ref.tensor([7, -7, 8], dtype=ref.int64)
    got_td = a.true_divide(2)
    want_td = ra.true_divide(2)
    assert str(got_td.dtype).split(".")[-1] == str(want_td.dtype).split(".")[-1]
    np.testing.assert_allclose(got_td.numpy(), want_td.numpy())

    b = a.clone()
    rb = ra.clone()
    b.divide_(2, rounding_mode="trunc")
    rb.divide_(2, rounding_mode="trunc")
    assert b.tolist() == rb.tolist()
