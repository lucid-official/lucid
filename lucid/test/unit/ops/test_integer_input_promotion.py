"""Float-natured ops on integer tensors never truncate: they answer in float or are refused.

``mean`` of ``[1, 2, 3, 4]`` was 2, ``int @ float`` cast the float operand to
int, and an in-place float op on an integer tensor (``exp_``) kept the
integer's 8-byte strides over the new 4-byte floats, reading every other
element — all without an error.  That in-place op is refused now, as the
reference refuses it: its float result cannot be cast to the integer target,
and an in-place op does not change its target's dtype (LCD-474).  ``mean`` of
an integer is refused as well, as the reference refuses it (LCD-1026): the call
site casts, and the float answer is the mean it meant.
"""

import math

import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid.test._helpers.dtype_refusal import engine_refusal

_DEVICES = [
    "cpu",
    pytest.param(
        "metal", marks=pytest.mark.skipif(not lucid.metal.is_available(), reason="no Metal device")
    ),
]


def _ints(*values: int, device: str = "cpu") -> lucid.Tensor:
    return lucid.tensor(list(values), dtype=lucid.int64, device=device)


@pytest.mark.parametrize("device", _DEVICES)
def test_mean_of_integers_is_refused(device: str) -> None:
    refusal = engine_refusal("mean", lucid.int64)
    assert refusal is not None
    with refusal.raises():
        _ints(1, 2, 3, 4, device=device).mean()


@pytest.mark.parametrize("device", _DEVICES)
def test_mean_of_the_cast_integers_is_floating(device: str) -> None:
    # The spelling the refusal offers: cast, then average.
    out = _ints(1, 2, 3, 4, device=device).float().mean()
    assert out.dtype == lucid.float32
    assert out.item() == 2.5


@pytest.mark.parametrize("device", _DEVICES)
def test_norm_of_integers_is_refused(device: str) -> None:
    # Not promoted: the reference refuses an integer norm, and so does
    # Lucid now (LCD-384), for every order.
    with pytest.raises(_C_engine.DtypeMismatch):
        lucid.linalg.norm(_ints(1, 1, device=device))


@pytest.mark.parametrize("device", _DEVICES)
def test_matmul_refuses_an_integer_with_a_float_operand(device: str) -> None:
    a = lucid.tensor([[1, 2]], dtype=lucid.int64, device=device)
    b = lucid.tensor([[0.5], [0.25]], device=device)
    with pytest.raises(TypeError, match="DtypeMismatch"):
        _ = a @ b
    with pytest.raises(TypeError, match="DtypeMismatch"):
        _ = lucid.matmul(a, b)
    with pytest.raises(TypeError, match="DtypeMismatch"):
        _ = b.reshape(1, 2) @ a.reshape(2, 1)


@pytest.mark.parametrize("device", _DEVICES)
def test_in_place_float_op_on_integers_is_refused_and_reads_back_unchanged(device: str) -> None:
    x = _ints(1, 2, 3, 4, 5, device=device)
    with pytest.raises(RuntimeError, match="can't be cast to the desired output type"):
        x.exp_()
    assert x.dtype == lucid.int64
    assert x.tolist() == [1, 2, 3, 4, 5]
    assert x.exp().tolist() == pytest.approx([math.exp(v) for v in range(1, 6)], rel=1e-6)
