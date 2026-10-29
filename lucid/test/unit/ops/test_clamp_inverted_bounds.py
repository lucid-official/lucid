"""``clamp`` with ``min > max`` sends every element to ``max`` (LCD-670).

The reference defines an inverted range as "all elements are set to
``max``".  The tensor-bound path always did; the scalar-bound path handed the
inverted pair to the engine's ``clip`` kernel, whose CPU form kept whichever
bound an element crossed first (``clamp(tensor([0, 1.5, 3]), 2, 1)`` came
back ``[2, 2, 1]``) while Metal answered ``max``.  The two devices, the free
function, the method, the ``clip`` alias and the in-place twins now agree
with the reference for scalar and tensor bounds, floating and integer.
"""

from collections.abc import Callable
from types import ModuleType

import pytest

import lucid

_NAN = float("nan")

#: (min, max) — inverted, tied and ordered pairs, ints and floats.
_BOUNDS = [(2, 1), (2.0, 1.0), (0.5, -0.5), (1, 1), (-1, 2), (1, -3), (2, 2.5)]

_FLOAT_VALUES = [0.0, 1.5, 3.0, -1.0, 0.25, _NAN]
_INT_VALUES = [0, 1, 3, -1, 2, -7]

_DTYPES = ["float32", "int32", "int64"]


def _values(dtype: str) -> list[float]:
    return _FLOAT_VALUES if dtype == "float32" else _INT_VALUES


def _name(dtype: object) -> str:
    return str(dtype).rsplit(".", 1)[-1]


def _same(got: lucid.Tensor, want: object) -> None:
    assert _name(got.dtype) == _name(want.dtype), (got.dtype, want.dtype)
    g, w = got.tolist(), want.tolist()
    assert len(g) == len(w)
    for gi, wi in zip(g, w, strict=True):
        assert (gi != gi and wi != wi) or gi == wi, (g, w)


_SPELLINGS: dict[str, Callable[..., object]] = {
    "free-clamp": lambda lib, x, lo, hi: lib.clamp(x, min=lo, max=hi),
    "free-clip": lambda lib, x, lo, hi: lib.clip(x, min=lo, max=hi),
    "method-clamp": lambda lib, x, lo, hi: x.clamp(lo, hi),
    "method-clip": lambda lib, x, lo, hi: x.clip(lo, hi),
}


@pytest.mark.parity
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("spelling", list(_SPELLINGS))
@pytest.mark.parametrize(("lo", "hi"), _BOUNDS, ids=str)
def test_scalar_bounds_clamp_as_the_reference_clamps(
    ref: ModuleType, device: str, dtype: str, spelling: str, lo: float, hi: float
) -> None:
    call = _SPELLINGS[spelling]
    values = _values(dtype)
    x = lucid.tensor(values, dtype=getattr(lucid, dtype), device=device)
    r = ref.tensor(values, dtype=getattr(ref, dtype))
    _same(call(lucid, x, lo, hi), call(ref, r, lo, hi))


@pytest.mark.parity
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("name", ["clamp_", "clip_"])
@pytest.mark.parametrize(("lo", "hi"), _BOUNDS, ids=str)
def test_in_place_clamp_agrees_with_the_reference(
    ref: ModuleType, device: str, dtype: str, name: str, lo: float, hi: float
) -> None:
    values = _values(dtype)
    x = lucid.tensor(values, dtype=getattr(lucid, dtype), device=device)
    r = ref.tensor(values, dtype=getattr(ref, dtype))
    # An int tensor refuses bounds that promote it to float, in both libraries.
    try:
        want = getattr(r, name)(lo, hi)
    except RuntimeError:
        with pytest.raises(RuntimeError):
            getattr(x, name)(lo, hi)
        return
    _same(getattr(x, name)(lo, hi), want)


@pytest.mark.parity
@pytest.mark.parametrize("dtype", ["float32", "int32"])
@pytest.mark.parametrize("spelling", ["free-clamp", "method-clamp"])
def test_tensor_bounds_clamp_as_the_reference_clamps(
    ref: ModuleType, device: str, dtype: str, spelling: str
) -> None:
    call = _SPELLINGS[spelling]
    values = _values(dtype)
    # Element by element: inverted, tied and ordered bounds in one tensor.
    los = [2, 1, 1, -1, 5, 0]
    his = [1, 1, 4, -3, 2, 0]
    mine_dt, theirs_dt = getattr(lucid, dtype), getattr(ref, dtype)
    x = lucid.tensor(values, dtype=mine_dt, device=device)
    r = ref.tensor(values, dtype=theirs_dt)
    lo = lucid.tensor(los, dtype=mine_dt, device=device)
    hi = lucid.tensor(his, dtype=mine_dt, device=device)
    _same(
        call(lucid, x, lo, hi),
        call(ref, r, ref.tensor(los, dtype=theirs_dt), ref.tensor(his, dtype=theirs_dt)),
    )


@pytest.mark.parametrize("dtype", ["float32", "int32"])
def test_scalar_and_tensor_bounds_agree_on_an_inverted_range(device: str, dtype: str) -> None:
    mine_dt = getattr(lucid, dtype)
    x = lucid.tensor(_values(dtype), dtype=mine_dt, device=device)
    scalar = lucid.clamp(x, min=2, max=1)
    tensor = lucid.clamp(
        x,
        min=lucid.tensor(2, dtype=mine_dt, device=device),
        max=lucid.tensor(1, dtype=mine_dt, device=device),
    )
    _same(scalar, tensor)
