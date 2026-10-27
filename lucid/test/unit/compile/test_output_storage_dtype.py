"""A compiled output's storage has to carry its dtype, not only its tensor.

``run_executable`` wrapped every output MTLBuffer in a ``GpuStorage`` whose
``dtype`` and ``nbytes`` were left at their defaults, ``F32`` and ``0``.
The tensor around it said ``int64``; the storage said ``float32``; every
reader that asks the storage took its word.  ``.numpy()`` is one — an
int64 ``x * 2 + 1`` came back as a float32 array of integer bit patterns
(15 read as ``2.1e-44``).  ``.to("cpu")`` and any eager op on the result
went through MLX, which knows the real dtype, so every test that compared
after one of those passed.
"""

import numpy as np
import pytest

import lucid

from lucid.test.unit.compile._helpers import COMPILE_DEVICE


def _metal_ok() -> bool:
    try:
        lucid.zeros(1).to(COMPILE_DEVICE)
    except Exception:  # noqa: BLE001 — any failure means no Metal here
        return False
    return True


pytestmark = pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")


@pytest.mark.parametrize(
    ("dtype", "np_dtype"),
    [(lucid.int64, np.int64), (lucid.int32, np.int32), (lucid.float32, np.float32)],
)
def test_numpy_of_a_compiled_output_keeps_its_dtype(dtype: object, np_dtype: type) -> None:
    lucid.manual_seed(0)
    x = lucid.randint(-9, 9, (4, 6)).to(dtype).to(COMPILE_DEVICE)  # type: ignore[arg-type]
    fn = lambda t: t * 2 + 1  # noqa: E731

    got = lucid.compile(fn)(x).numpy()
    want = fn(x.to("cpu")).numpy()
    assert got.dtype == np_dtype
    assert np.array_equal(got, want)


def test_numpy_of_a_compiled_bool_output() -> None:
    lucid.manual_seed(0)
    x = lucid.randn(4, 6).to(COMPILE_DEVICE)
    fn = lambda t: t > 0  # noqa: E731

    got = lucid.compile(fn)(x).numpy()
    assert got.dtype == np.bool_
    assert np.array_equal(got, fn(x.to("cpu")).numpy())
