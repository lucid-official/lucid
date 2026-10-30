"""Bit-shift amounts outside ``[0, width)`` follow the reference on both streams (LCD-666).

Metal's native shift masks the amount modulo the bit width, so ``1 << 32`` was 1
and ``1 << -1`` was ``1 << 31``; the CPU right shift answered 0 for a negative
amount where the reference answers the sign fill.  The reference: an amount below
0 or at least the width shifts every bit out, leaving 0 for a left shift and the
sign fill (0 or -1) for a right shift.
"""

import operator
from collections.abc import Callable
from types import ModuleType

import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid.test._fixtures.devices import device_dtype_params, metal_available

_INT_DTYPES = [lucid.int8, lucid.int16, lucid.int32, lucid.int64]
_BITS = {lucid.int8: 8, lucid.int16: 16, lucid.int32: 32, lucid.int64: 64}


def _ref_dtype(ref: ModuleType, dtype: lucid.dtype) -> object:
    return getattr(ref, str(dtype).split(".")[-1])


def _values(bits: int) -> list[int]:
    lo = -(1 << (bits - 1))
    hi = (1 << (bits - 1)) - 1
    return [0, 1, 2, 3, 5, -1, -2, -5, lo, lo + 1, hi]


def _amounts(bits: int) -> list[int]:
    return list(range(-bits - 2, 2 * bits + 3))


_OPS: list[tuple[str, Callable[[object, object], object], Callable[[object, object], object]]] = [
    ("lshift", operator.lshift, lucid.bitwise_left_shift),
    ("rshift", operator.rshift, lucid.bitwise_right_shift),
]


@pytest.mark.parametrize(("device", "dtype"), device_dtype_params(_INT_DTYPES))
@pytest.mark.parametrize(("name", "op", "fn"), _OPS, ids=[o[0] for o in _OPS])
def test_tensor_shift_amounts_match_reference(
    ref: ModuleType,
    device: str,
    dtype: lucid.dtype,
    name: str,
    op: Callable[[object, object], object],
    fn: Callable[[object, object], object],
) -> None:
    """Every value against every amount in ``[-w - 2, 2w + 2]``, broadcast, per dtype."""
    bits = _BITS[dtype]
    rdt = _ref_dtype(ref, dtype)
    vals, amts = _values(bits), _amounts(bits)

    x = lucid.tensor(vals, dtype=dtype, device=device).reshape(-1, 1)
    y = lucid.tensor(amts, dtype=dtype, device=device).reshape(1, -1)
    rx = ref.tensor(vals, dtype=rdt).reshape(-1, 1)
    ry = ref.tensor(amts, dtype=rdt).reshape(1, -1)
    want = op(rx, ry).tolist()

    for got in (op(x, y), fn(x, y)):
        assert got.dtype == dtype
        assert got.tolist() == want, name

    full_x = x.expand(len(vals), len(amts)).contiguous()
    full_y = y.expand(len(vals), len(amts)).contiguous()
    assert op(full_x, full_y).tolist() == want


@pytest.mark.parametrize(("device", "dtype"), device_dtype_params(_INT_DTYPES))
@pytest.mark.parametrize(("name", "op", "fn"), _OPS, ids=[o[0] for o in _OPS])
def test_scalar_shift_amounts_match_reference(
    ref: ModuleType,
    device: str,
    dtype: lucid.dtype,
    name: str,
    op: Callable[[object, object], object],
    fn: Callable[[object, object], object],
) -> None:
    bits = _BITS[dtype]
    rdt = _ref_dtype(ref, dtype)
    vals = _values(bits)
    x = lucid.tensor(vals, dtype=dtype, device=device)
    rx = ref.tensor(vals, dtype=rdt)
    for amount in (-bits, -1, 0, 1, bits - 1, bits, bits + 1, 2 * bits):
        got = op(x, amount)
        assert got.dtype == dtype
        assert got.tolist() == op(rx, amount).tolist(), (name, amount)


@pytest.mark.parametrize(("device", "dtype"), device_dtype_params(_INT_DTYPES))
def test_inplace_shift_amounts_match_reference(
    ref: ModuleType, device: str, dtype: lucid.dtype
) -> None:
    bits = _BITS[dtype]
    rdt = _ref_dtype(ref, dtype)
    vals = _values(bits) * 3
    amts = ([-1, bits, 2 * bits] * len(_values(bits)))[: len(vals)]
    for method in ("bitwise_left_shift_", "bitwise_right_shift_"):
        x = lucid.tensor(vals, dtype=dtype, device=device)
        y = lucid.tensor(amts, dtype=dtype, device=device)
        rx = ref.tensor(vals, dtype=rdt)
        getattr(rx, method)(ref.tensor(amts, dtype=rdt))
        out = getattr(x, method)(y)
        assert out.tolist() == rx.tolist(), method
        assert x.tolist() == rx.tolist(), method


@pytest.mark.parametrize("dtype", _INT_DTYPES)
def test_shift_with_out_of_range_amounts_on_metal_is_unsynced(
    ref: ModuleType, dtype: lucid.dtype
) -> None:
    """The amounts are replaced in the graph, not on the host."""
    if not metal_available():
        pytest.skip("Metal unavailable")
    bits = _BITS[dtype]
    rdt = _ref_dtype(ref, dtype)
    vals, amts = [1, -5, 5, 2], [bits - 1, bits, -1, 2 * bits]
    x = lucid.tensor(vals, dtype=dtype, device="metal")
    y = lucid.tensor(amts, dtype=dtype, device="metal")
    lucid.eval(x, y)
    before = _C_engine.host_sync_count()
    left, right = x << y, x >> y
    lucid.eval(left, right)
    assert _C_engine.host_sync_count() == before
    rx, ry = ref.tensor(vals, dtype=rdt), ref.tensor(amts, dtype=rdt)
    assert left.tolist() == (rx << ry).tolist()
    assert right.tolist() == (rx >> ry).tolist()
