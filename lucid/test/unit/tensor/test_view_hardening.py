"""Paths that must read a view correctly, whichever op made it.

Written before any op made a view, when every tensor started at the first
byte of its own buffer.  On the CPU reshapes, slices, transposes,
``expand``, ``diagonal``, ``unfold`` and ``as_strided`` now all do; these
tests build views directly (``TensorImpl._make_view``), at offsets and
strides no single op has to produce, and hold each path to reading the
view's elements rather than the buffer's first ones.
"""

import numpy as np
import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid._dispatch import _wrap


def _view(base: lucid.Tensor, shape: list[int], stride: list[int], offset: int) -> lucid.Tensor:
    return _wrap(_C_engine.TensorImpl._make_view(base._impl, shape, stride, offset))


@pytest.fixture
def base() -> lucid.Tensor:
    return lucid.arange(6).float()  # [0, 1, 2, 3, 4, 5]


def test_a_view_at_an_offset_is_contiguous_but_not_dense(base) -> None:
    v = _view(base, [3], [1], 2)
    assert v._impl.is_contiguous() and not v._impl.is_dense()
    assert base._impl.is_dense()
    assert v._impl.storage_offset_bytes() == 2 * 4


def test_item_reads_the_element_the_view_starts_at(base) -> None:
    # offset_ is in bytes; item() scaled it by the element size again.
    assert _view(base, [], [], 4).item() == 4.0


def test_numpy_and_tolist_read_the_view(base) -> None:
    v = _view(base, [3], [1], 2)
    assert v.tolist() == [2.0, 3.0, 4.0]
    np.testing.assert_array_equal(v.numpy(), [2.0, 3.0, 4.0])


def test_bfloat16_numpy_reads_only_the_view() -> None:
    b = lucid.tensor([1.0, 2.0, 3.0, 4.0], dtype=lucid.bfloat16)
    np.testing.assert_array_equal(_view(b, [2], [1], 1).numpy(), [2.0, 3.0])


@pytest.mark.parametrize(
    ("shape", "stride", "offset", "values"),
    [([3], [1], 2, [2.0, 3.0, 4.0]), ([3], [2], 0, [0.0, 2.0, 4.0])],
    ids=["offset", "strided"],
)
def test_ops_compute_on_the_view(base, shape, stride, offset, values) -> None:
    v = _view(base, shape, stride, offset)
    want = lucid.tensor(values)
    assert (v + 1.0).tolist() == (want + 1.0).tolist()  # binary kernel
    assert v.exp().tolist() == want.exp().tolist()  # unary kernel
    assert v.sum().item() == sum(values)  # reduce kernel
    assert v.reshape(3, 1).tolist() == [[x] for x in values]  # view op


def test_data_ptr_sits_at_the_view_s_first_element(base) -> None:
    v = _view(base, [3], [1], 2)
    assert v._impl.data_ptr() == base._impl.data_ptr() + 2 * 4


def test_copying_from_a_view_takes_the_view_s_values(base) -> None:
    dst = lucid.zeros(3)
    dst.copy_(_view(base, [3], [1], 2))
    assert dst.tolist() == [2.0, 3.0, 4.0]


def test_copying_into_a_view_lands_in_the_view_s_elements(base) -> None:
    # A copy into a view once wrote at the buffer's first byte, and was then
    # refused; it now follows the view's offset.
    _view(base, [3], [1], 2).copy_(lucid.ones(3))
    assert base.tolist() == [0.0, 1.0, 1.0, 1.0, 1.0, 5.0]


def test_a_view_past_the_end_of_its_buffer_is_refused(base) -> None:
    with pytest.raises(ValueError, match="past the end"):
        _view(base, [3], [1], 4)
