"""Which dtypes Metal holds, and what it does with the ones it does not.

The GPU stream is MLX, and MLX has no double-precision type.  The
engine's ``to_mlx_dtype`` therefore refuses ``float64`` and
``complex128`` at every way onto Metal — a factory, a transfer, a cast —
with a message that names the dtype and says what to do instead.  Every
other dtype maps to a native MLX one, bfloat16 and complex64 included
(the suite used to list both as unsupported).

The device × dtype sweeps elsewhere never generate the refused pairs:
``METAL_UNSUPPORTED_DTYPES`` in ``lucid/test/_fixtures/devices.py`` leaves
them out at collection.  This file covers those pairs instead, and checks
that set against the engine in both directions — every dtype in it is
refused, every dtype outside it is held — so the filter cannot drift from
what Metal actually does.

The refusal is the engine's ``NotImplementedError``, which subclasses
``LucidError`` -> ``RuntimeError`` and not the builtin of that name, so
``except NotImplementedError`` does not catch it.  That trap is pinned in
``lucid/test/unit/ops/test_shape_composites.py``.
"""

from collections.abc import Callable

import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid.test._fixtures.devices import METAL_UNSUPPORTED_DTYPES

# Every dtype Lucid exposes, whatever alias it also goes by.
_ALL_DTYPES = sorted({v for v in vars(lucid).values() if isinstance(v, lucid.dtype)}, key=str)
_REFUSED = [d for d in _ALL_DTYPES if d in METAL_UNSUPPORTED_DTYPES]
_HELD = [d for d in _ALL_DTYPES if d not in METAL_UNSUPPORTED_DTYPES]

# The way out each refusal offers.
_ADVICE = {
    lucid.float64: "Cast to float32 first, or keep the tensor on CPU.",
    lucid.complex128: "Cast to complex64 first, or keep the tensor on CPU.",
}

# Every way a tensor of a given dtype can arrive on Metal.
_WAYS_ONTO_METAL: dict[str, Callable[[lucid.dtype], lucid.Tensor]] = {
    "zeros": lambda dt: lucid.zeros(2, 3, dtype=dt, device="metal"),
    "ones": lambda dt: lucid.ones(2, 3, dtype=dt, device="metal"),
    "empty": lambda dt: lucid.empty(2, 3, dtype=dt, device="metal"),
    "tensor": lambda dt: lucid.tensor([1, 0], dtype=dt, device="metal"),
    "to-metal": lambda dt: lucid.ones(2, dtype=dt).to("metal"),
    "cast-on-metal": lambda dt: lucid.ones(2, device="metal").to(dt),
}


def test_the_filter_names_only_real_dtypes() -> None:
    # A stale entry would drop out of _REFUSED below and go unchecked.
    assert METAL_UNSUPPORTED_DTYPES <= set(_ALL_DTYPES)


@pytest.mark.parametrize("way", list(_WAYS_ONTO_METAL))
@pytest.mark.parametrize("dtype", _REFUSED, ids=str)
def test_a_refused_dtype_is_refused_at_every_way_in(dtype: lucid.dtype, way: str) -> None:
    with pytest.raises(_C_engine.NotImplementedError) as info:
        _WAYS_ONTO_METAL[way](dtype)
    message = str(info.value)
    assert f"{str(dtype).removeprefix('lucid.')} is not supported on GPU" in message
    assert _ADVICE[dtype] in message


@pytest.mark.parametrize("way", list(_WAYS_ONTO_METAL))
@pytest.mark.parametrize("dtype", _HELD, ids=str)
def test_a_held_dtype_lands_on_metal_as_itself(dtype: lucid.dtype, way: str) -> None:
    t = _WAYS_ONTO_METAL[way](dtype)
    assert t.device.type == "metal"
    assert t.dtype == dtype


@pytest.mark.parametrize("dtype", _HELD, ids=str)
def test_a_held_dtype_computes_on_metal(dtype: lucid.dtype) -> None:
    host = lucid.tensor([1, 0, 1], dtype=dtype)
    on_metal = host.to("metal")
    out = on_metal + on_metal
    assert out.device.type == "metal"
    assert out.dtype == dtype
    assert out.to("cpu").tolist() == (host + host).tolist()
