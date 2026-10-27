"""Every ingest path keeps the source's rank (LCD-399).

``np.ascontiguousarray`` returns at least one dimension, and the
converter ran an ndarray through it whenever ``device=`` or ``dtype=``
was given: ``lucid.tensor(np.asarray(0.5), device="cpu")`` came back as
shape ``(1,)`` while the same call without ``device`` stayed 0-d.  The
shape of a converted source must not depend on which keyword was passed,
so each source kind is swept over every device and dtype spelling and
held to the reference's shape.
"""

import numpy as np
import pytest

import lucid
from lucid.test._fixtures.devices import metal_available

_SOURCES = {
    "python_scalar": lambda: 0.5,
    "ndarray_0d": lambda: np.asarray(0.5, dtype=np.float32),
    "ndarray_1elem": lambda: np.asarray([0.5], dtype=np.float32),
    "ndarray_0d_uint8": lambda: np.asarray(3, dtype=np.uint8),
    "ndarray_noncontiguous": lambda: np.arange(6, dtype=np.float32).reshape(2, 3).T,
    "numpy_scalar": lambda: np.float32(0.5),
}

_DEVICES = [None, "cpu", "metal"]
_DTYPES = [None, "float32", "float64", "float16", "int64"]


def _skip_unheld(device, dtype):
    if device == "metal" and not metal_available():
        pytest.skip("Metal device not available on this host")
    if device == "metal" and dtype == "float64":
        pytest.skip("Metal holds no float64")


def _kwargs(lib, device, dtype):
    kw = {}
    if dtype is not None:
        kw["dtype"] = getattr(lib, dtype)
    if device is not None and lib is lucid:
        kw["device"] = device
    return kw


@pytest.mark.parametrize("dtype", _DTYPES, ids=str)
@pytest.mark.parametrize("device", _DEVICES, ids=str)
@pytest.mark.parametrize("source", list(_SOURCES))
@pytest.mark.parametrize("entry", ["tensor", "Tensor", "as_tensor"])
def test_rank_matches_reference(ref, entry, source, device, dtype):
    _skip_unheld(device, dtype)
    # The reference refuses some casts at ingest (a float numpy scalar into
    # int64) that it takes as a conversion; the shape is the same either way.
    want = ref.as_tensor(_SOURCES[source]())
    if dtype is not None:
        want = want.to(getattr(ref, dtype))
    got = getattr(lucid, entry)(_SOURCES[source](), **_kwargs(lucid, device, dtype))
    assert tuple(got.shape) == tuple(want.shape)
    np.testing.assert_allclose(
        got.to("cpu").numpy().astype(np.float64), want.numpy().astype(np.float64)
    )


@pytest.mark.parametrize("source", ["ndarray_0d", "ndarray_1elem", "ndarray_0d_uint8"])
def test_from_numpy_keeps_the_rank(ref, source):
    arr = _SOURCES[source]()
    assert tuple(lucid.from_numpy(arr).shape) == tuple(ref.from_numpy(arr).shape)


@pytest.mark.parametrize("dtype", _DTYPES, ids=str)
@pytest.mark.parametrize("device", _DEVICES, ids=str)
def test_a_0d_dlpack_source_keeps_its_rank(ref, device, dtype):
    _skip_unheld(device, dtype)
    imported = lucid.from_dlpack(np.asarray(0.5, dtype=np.float32))
    want = ref.from_dlpack(np.asarray(0.5, dtype=np.float32))
    assert tuple(imported.shape) == tuple(want.shape) == ()
    # And on through a Tensor source, which converts rather than re-ingests.
    converted = lucid.Tensor(imported, **_kwargs(lucid, device, dtype))
    assert tuple(converted.shape) == ()


@pytest.mark.parametrize("dtype", _DTYPES, ids=str)
@pytest.mark.parametrize("device", _DEVICES, ids=str)
def test_a_list_of_0d_arrays_stacks_them(device, dtype):
    """The reference refuses this source; NumPy's stacking is the answer."""
    _skip_unheld(device, dtype)
    source = [np.asarray(0.5, dtype=np.float32)] * 2
    assert tuple(lucid.tensor(source, **_kwargs(lucid, device, dtype)).shape) == (2,)
