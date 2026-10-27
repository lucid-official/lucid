"""Packing a strided CPU view: every view shape and dtype matches NumPy's reading.

``contiguous()``, every op that reads a view, and a write through a view
copy a view's elements to or from a dense buffer.  That copy walked
element by element and took 4.6 ms for a 1024 x 1024 transpose.  It now
merges axes, moves contiguous runs with memcpy, fills broadcast runs and
tiles 2-D transposes (``core/StridedCopy.h``).  Each path is held to
NumPy, which reads the same strides independently through ``.numpy()``.
"""

from collections.abc import Callable

import numpy as np
import pytest

import lucid

DTYPES = [
    np.float32,
    np.float64,
    np.float16,
    np.int64,
    np.int32,
    np.int16,
    np.int8,
    np.bool_,
    np.complex64,
    np.complex128,
]
VIEWS: dict[str, Callable[[lucid.Tensor], lucid.Tensor]] = {
    "transpose": lambda x: x.transpose(0, x.ndim - 1),
    "permute": lambda x: x.permute(*reversed(range(x.ndim))),
    "inner slice": lambda x: x[..., 1:-1],
    "steps": lambda x: x[::2, ..., ::3],
    "expand": lambda x: x[:1].expand(4, *x.shape[1:]),
    "row run": lambda x: x[1:3],
    "unfold": lambda x: x.reshape(-1).unfold(0, 4, 3),
    "transposed slice": lambda x: x[:, 1:].transpose(0, 1),
}


def _array(dtype: type, shape: tuple[int, ...]) -> np.ndarray:
    base = np.random.default_rng(0).standard_normal(shape) * 10
    if np.issubdtype(dtype, np.complexfloating):
        base = base + 1j * base[::-1]
    return base.astype(dtype)


@pytest.mark.parametrize("shape", [(7, 9), (5, 6, 7)], ids=["2-D", "3-D"])
@pytest.mark.parametrize("view", list(VIEWS))
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: np.dtype(d).name)
def test_a_packed_view_matches_numpy(dtype: type, view: str, shape: tuple[int, ...]) -> None:
    v = VIEWS[view](lucid.tensor(_array(dtype, shape)))
    want = np.asarray(v.numpy())
    np.testing.assert_array_equal(v.contiguous().numpy(), want)
    np.testing.assert_array_equal(v.clone().numpy(), want)  # an op reading the view


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int16, np.complex64])
def test_a_write_through_a_strided_view_lands_on_its_elements(dtype: type) -> None:
    ref = np.arange(42).reshape(6, 7).astype(dtype)
    x = lucid.tensor(ref.copy())
    x.t()[1:4].mul_(2)
    ref.T[1:4] *= 2
    x[::2, 1::3].fill_(-1)
    ref[::2, 1::3] = -1
    np.testing.assert_array_equal(x.numpy(), ref)
