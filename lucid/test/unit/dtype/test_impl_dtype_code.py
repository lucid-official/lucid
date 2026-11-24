"""``TensorImpl.dtype_code`` is ``TensorImpl.dtype.value``, read without the enum object (LCD-1099).

A Python fast path that only compares an operand's dtype used to read
``impl.dtype.value``: pybind builds a ``Dtype`` object for ``impl.dtype`` (about 0.23 us)
and ``.value`` is a second lookup on top (0.29 us together).  ``dtype_code`` hands back the
same integer straight from the impl (about 0.07 us), so the dtype refusals that run on every
``gather`` / ``take`` / ``complex`` call stop costing a third of the call.

What must hold is that the two reads never disagree, for every engine dtype, on both
streams, and through every shape of tensor the fast paths see (0-d, empty, a view, a
phantom).  A dtype added to the engine without a code that matches its enum value fails the
sweep here.
"""

import contextlib
from collections.abc import Iterator

import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid._dtype import _ENGINE_TO_DTYPE
from lucid.test._fixtures.devices import device_dtype_params

_DTYPES = list(_ENGINE_TO_DTYPE.values())


@contextlib.contextmanager
def _phantom() -> Iterator[None]:
    previous = _C_engine.phantom_mode()
    _C_engine.set_phantom_mode(True)
    try:
        yield
    finally:
        _C_engine.set_phantom_mode(previous)


def test_sweep_covers_every_engine_dtype() -> None:
    """The cases below name every member of the engine's ``Dtype``, none left out."""
    assert set(_ENGINE_TO_DTYPE) == set(_C_engine.Dtype.__members__.values())


@pytest.mark.parametrize(("device", "dtype"), device_dtype_params(_DTYPES))
def test_code_is_the_enum_value(device: str, dtype: lucid.dtype) -> None:
    impl = lucid.zeros(3, dtype=dtype, device=device)._impl
    assert impl.dtype_code == impl.dtype.value
    assert _C_engine.Dtype(impl.dtype_code) == impl.dtype


@pytest.mark.parametrize(("device", "dtype"), device_dtype_params(_DTYPES))
def test_code_is_a_plain_int(device: str, dtype: lucid.dtype) -> None:
    code = lucid.zeros(3, dtype=dtype, device=device)._impl.dtype_code
    assert type(code) is int


@pytest.mark.parametrize(("device", "dtype"), device_dtype_params(_DTYPES))
@pytest.mark.parametrize("shape", [(), (0,), (2, 0, 3), (4, 3)], ids=str)
def test_zero_d_and_empty_tensors(device: str, dtype: lucid.dtype, shape: tuple[int, ...]) -> None:
    impl = lucid.zeros(shape, dtype=dtype, device=device)._impl
    assert impl.dtype_code == impl.dtype.value


@pytest.mark.parametrize(("device", "dtype"), device_dtype_params(_DTYPES))
def test_view_reads_its_own_dtype(device: str, dtype: lucid.dtype) -> None:
    """A non-contiguous view is its own impl and answers with the dtype of its data."""
    base = lucid.zeros(4, 6, dtype=dtype, device=device)
    view = base.transpose(0, 1)[::2]
    assert not view.is_contiguous()
    assert view._impl.dtype_code == view._impl.dtype.value
    assert view._impl.dtype_code == base._impl.dtype_code


def test_code_follows_a_cast() -> None:
    """The code is read off the impl each time: a cast tensor answers with its new dtype."""
    x = lucid.zeros(3, dtype=lucid.float32)
    y = x.to(lucid.int64)
    assert x._impl.dtype_code == _C_engine.Dtype.F32.value
    assert y._impl.dtype_code == _C_engine.Dtype.I64.value


def test_phantom_tensor_has_a_code() -> None:
    """A phantom tensor has no storage but still has a dtype to read."""
    with _phantom():
        impl = lucid.zeros(2, 3, dtype=lucid.float16)._impl
    assert impl.is_phantom
    assert impl.dtype_code == _C_engine.Dtype.F16.value
