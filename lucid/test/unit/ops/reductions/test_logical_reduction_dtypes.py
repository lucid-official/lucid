"""any / all / count_nonzero judge each element's truth, whatever its dtype (LCD-726).

``any`` of an int8 tensor on Metal went to MLX's Or-reduction with the int8
elements themselves, and that kernel answered False for every even value:
``any(int8 [2, 2])`` and ``any(int8 [2, 0])`` were both False, a flake the
boundary hypothesis test caught one run in eight.  The logical reductions now
reduce a bool array of truths on both streams.

The cells below cross every dtype the device holds with values that make a
narrow accumulator lie: even values, a sum that wraps to zero, the type's
extremes, NaN, negative zero.  The first test states the answer from the
definition (an element is true when it is not zero), so it needs no reference
framework; the second compares shapes, dims and ``keepdim`` with it.
"""

import math
from types import ModuleType

import pytest

import lucid
from lucid.test._fixtures.devices import device_dtype_params

_INT_BITS = {lucid.int8: 8, lucid.int16: 16, lucid.int32: 32, lucid.int64: 64}
_FLOATS = [lucid.float16, lucid.bfloat16, lucid.float32, lucid.float64]
_COMPLEX = [lucid.complex64, lucid.complex128]
_DTYPES = [lucid.bool, *_INT_BITS, *_FLOATS, *_COMPLEX]

#: The smallest positive subnormal of each float format that has a Metal type.
_TINY = {lucid.float16: 2.0**-24, lucid.bfloat16: 2.0**-133, lucid.float32: 2.0**-149}


def _name(dtype: lucid.dtype) -> str:
    return str(dtype).split(".")[-1]


def _ref_dtype(ref: ModuleType, dtype: lucid.dtype) -> object:
    return getattr(ref, _name(dtype))


def _nonzeros(dtype: lucid.dtype) -> list[object]:
    """Six nonzero values of ``dtype``, the awkward ones first."""
    if dtype == lucid.bool:
        return [True] * 6
    if dtype in _INT_BITS:
        bits = _INT_BITS[dtype]
        lo, hi, half = -(1 << (bits - 1)), (1 << (bits - 1)) - 1, 1 << (bits - 2)
        return [2, half, lo, hi, -1, 1]
    if dtype in _FLOATS:
        return [2.0, math.nan, -3.0, 64.0, math.inf, -1.0]
    return [2j, complex(math.nan, 0.0), 2 + 0j, complex(0.0, -4.0), 1j, -1 + 0j]


def _zero(dtype: lucid.dtype) -> object:
    return False if dtype == lucid.bool else (0j if dtype in _COMPLEX else 0)


def _rows(dtype: lucid.dtype) -> dict[str, list[object]]:
    """1-D inputs whose truth is known from the definition."""
    nz, zero = _nonzeros(dtype), _zero(dtype)
    rows: dict[str, list[object]] = {
        "single-true": [nz[0]],
        "single-false": [zero],
        "all-true": nz[:3],
        "all-false": [zero] * 5,
        "true-then-false": [nz[0], zero],
        "false-then-true": [zero, nz[0]],
        "long-all-true": [nz[0]] * 1000,
        "long-one-late": [zero] * 999 + [nz[0]],
        "long-one-early": [nz[0]] + [zero] * 999,
        "every-nonzero-kind": nz,
    }
    if dtype in _INT_BITS:
        half = nz[1]
        rows["sum-wraps-to-zero"] = [half] * 4
        rows["opposites"] = [-1, 1]
        rows["extremes"] = [nz[2], nz[3]]
        rows["only-min"] = [nz[2], zero]
    if dtype in _FLOATS:
        rows["negative-zero"] = [-0.0, 0.0]
        rows["nan-and-zero"] = [math.nan, 0.0]
    return rows


def _cells() -> list[object]:
    return [
        pytest.param(dev, dt, row, id=f"{dev}-{_name(dt)}-{row}")
        for dev, dt in (p.values for p in device_dtype_params(_DTYPES))
        for row in _rows(dt)
    ]


@pytest.mark.parametrize(("device", "dtype", "row"), _cells())
def test_logical_reductions_follow_the_definition(
    device: str, dtype: lucid.dtype, row: str
) -> None:
    """True means "not zero"; NaN is not zero; the answer never depends on the dtype's width."""
    x = lucid.tensor(_rows(dtype)[row], dtype=dtype, device=device)
    truth = [v != 0 for v in x.tolist()]
    some, every, count = any(truth), all(truth), sum(truth)

    for got, want in (
        (lucid.any(x), some),
        (x.any(), some),
        (lucid.any(x, dim=0), some),
        (lucid.any(x, dim=0, keepdim=True), some),
        (lucid.all(x), every),
        (x.all(), every),
        (lucid.all(x, dim=0), every),
        (lucid.all(x, dim=0, keepdim=True), every),
    ):
        assert got.dtype == lucid.bool
        assert got.tolist() == ([want] if got.ndim else want)
    for got in (lucid.count_nonzero(x), lucid.count_nonzero(x, 0)):
        assert got.dtype == lucid.int64
        assert got.item() == count


def _grid(dtype: lucid.dtype, mask: str) -> list[list[object]]:
    """A 2x3 input: the nonzero values where ``mask`` has a 1, zero elsewhere."""
    nz, zero = _nonzeros(dtype), _zero(dtype)
    flat = [nz[i] if bit == "1" else zero for i, bit in enumerate(mask)]
    return [flat[:3], flat[3:]]


_MASKS = ["111111", "000000", "101101", "000111", "100010", "010000"]
_DIMS = [None, 0, 1, -1, (0, 1), (1, 0), (-1, -2)]


@pytest.mark.parametrize(("device", "dtype"), device_dtype_params(_DTYPES))
@pytest.mark.parametrize("mask", _MASKS)
def test_dims_and_keepdim_match_reference(
    ref: ModuleType, device: str, dtype: lucid.dtype, mask: str
) -> None:
    values = _grid(dtype, mask)
    x = lucid.tensor(values, dtype=dtype, device=device)
    r = ref.tensor(x.tolist(), dtype=_ref_dtype(ref, dtype))
    for name in ("any", "all"):
        for dim in _DIMS:
            for keepdim in (False, True):
                got = getattr(lucid, name)(x, dim=dim, keepdim=keepdim)
                want = getattr(ref, name)(r, dim=dim, keepdim=keepdim)
                assert got.dtype == lucid.bool and want.dtype == ref.bool, (name, dim, keepdim)
                assert got.shape == tuple(want.shape), (name, dim, keepdim)
                assert got.tolist() == want.tolist(), (name, dim, keepdim)
    for dim in _DIMS:
        got = lucid.count_nonzero(x, dim)
        want = ref.count_nonzero(r, dim=dim)
        assert got.dtype == lucid.int64 and want.dtype == ref.int64
        assert got.tolist() == want.tolist(), dim


_EMPTY_SHAPES = [(0,), (0, 3), (3, 0)]


@pytest.mark.parametrize(("device", "dtype"), device_dtype_params(_DTYPES))
@pytest.mark.parametrize("shape", _EMPTY_SHAPES, ids=str)
def test_empty_inputs_match_reference(
    ref: ModuleType, device: str, dtype: lucid.dtype, shape: tuple[int, ...]
) -> None:
    """An empty reduction is the identity: ``any`` False, ``all`` True, a count of 0."""
    x = lucid.zeros(shape, dtype=dtype, device=device)
    r = ref.zeros(shape, dtype=_ref_dtype(ref, dtype))
    for name in ("any", "all"):
        for dim in [None, *range(len(shape))]:
            for keepdim in (False, True):
                got = getattr(lucid, name)(x, dim=dim, keepdim=keepdim)
                want = getattr(ref, name)(r, dim=dim, keepdim=keepdim)
                assert got.shape == tuple(want.shape), (name, dim, keepdim)
                assert got.tolist() == want.tolist(), (name, dim, keepdim)
    for dim in [None, *range(len(shape))]:
        got = lucid.count_nonzero(x, dim)
        want = ref.count_nonzero(r, dim=dim)
        assert got.shape == tuple(want.shape) and got.tolist() == want.tolist(), dim


@pytest.mark.parametrize(("device", "dtype"), device_dtype_params(_DTYPES))
@pytest.mark.parametrize("which", ["nonzero", "zero"])
def test_zero_dim_matches_reference(
    ref: ModuleType, device: str, dtype: lucid.dtype, which: str
) -> None:
    value = _nonzeros(dtype)[0] if which == "nonzero" else _zero(dtype)
    x = lucid.tensor(value, dtype=dtype, device=device)
    r = ref.tensor(x.tolist(), dtype=_ref_dtype(ref, dtype))
    assert x.ndim == 0
    for name in ("any", "all"):
        got, want = getattr(lucid, name)(x), getattr(ref, name)(r)
        assert got.shape == () and got.tolist() == want.tolist(), name
        got, want = getattr(x, name)(), getattr(r, name)()
        assert got.tolist() == want.tolist(), name
    got, want = lucid.count_nonzero(x), ref.count_nonzero(r)
    assert got.shape == () and got.tolist() == want.tolist()


_SUBNORMAL_VALUES = {
    lucid.float16: [_TINY[lucid.float16]],
    lucid.bfloat16: [_TINY[lucid.bfloat16]],
    lucid.float32: [_TINY[lucid.float32]],
    lucid.complex64: [complex(_TINY[lucid.float32], 0.0), complex(0.0, -_TINY[lucid.float32])],
}


@pytest.mark.parametrize(("device", "dtype"), device_dtype_params(list(_SUBNORMAL_VALUES)))
def test_a_subnormal_element_is_true(device: str, dtype: lucid.dtype) -> None:
    """Metal flushes a subnormal in ``x != 0``; the truth of an element is not that comparison."""
    for tiny in _SUBNORMAL_VALUES[dtype]:
        zero = _zero(dtype)
        # Built on the host: a bfloat16 subnormal is flushed by the Metal conversion that
        # ``lucid.tensor(..., device="metal")`` runs, before any reduction sees it.
        only = lucid.tensor([tiny, zero], dtype=dtype).to(device)
        both = lucid.tensor([tiny, tiny], dtype=dtype).to(device)
        assert only.tolist()[0] == tiny
        assert lucid.any(only).item() and not lucid.all(only).item(), tiny
        assert lucid.all(both).item(), tiny
        assert lucid.any(only, dim=0).item() and not lucid.all(only, dim=0).item(), tiny
        assert lucid.all(both, dim=0, keepdim=True).tolist() == [True], tiny
