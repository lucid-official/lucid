"""Unit tests for CPU block-view storage offsets, allocation-frame as_strided,
and Metal shared-tensor sibling writes (LCD-517 E1).
"""

import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid._dispatch import _wrap
from lucid.test._fixtures.devices import metal_available


def test_cpu_block_view_reports_allocation_offset_and_stays_dense() -> None:
    x = lucid.arange(18.0).reshape(3, 6)
    row = x[1]
    assert row._impl.is_dense()
    assert row.storage_offset() == 6
    assert row.data_ptr() - x.data_ptr() == 6 * row.element_size()

    sub = row[2:5]
    assert sub._impl.is_dense()
    assert sub.storage_offset() == 8
    assert sub.data_ptr() - x.data_ptr() == 8 * sub.element_size()


def test_cpu_block_view_as_strided_reads_and_writes_root_allocation() -> None:
    x = lucid.arange(6.0).reshape(2, 3)
    row1 = x[1]
    assert row1.as_strided((2,), (1,), 0).tolist() == [0.0, 1.0]
    assert x[0].as_strided((6,), (1,)).tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    win = row1.as_strided((2,), (1,), 0)
    win.fill_(-5.0)
    assert x.tolist() == [[-5.0, -5.0, 2.0], [3.0, 4.0, 5.0]]


def test_cpu_block_view_inplace_after_dropping_root_preserves_allocation_frame() -> None:
    x = lucid.arange(6.0).reshape(2, 3)
    row1 = x[1]
    del x
    row1.add_(10.0)
    assert row1.tolist() == [13.0, 14.0, 15.0]
    assert row1.storage_offset() == 3
    assert row1._impl.storage_nbytes() == 24


@pytest.mark.skipif(not metal_available(), reason="Metal not available")
def test_metal_shared_tensor_block_view_settles_pending_gpu_readers() -> None:
    buf = lucid.metal.shared_tensor((2, 3))
    g = buf.to("metal")
    p = g * 1.0
    buf[1].fill_(5.0)
    assert p.tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


@pytest.mark.skipif(not metal_available(), reason="Metal not available")
def test_metal_engine_view_family_member_as_strided_reads_root() -> None:
    x = lucid.arange(6.0, device="metal")
    v = _wrap(_C_engine.TensorImpl._make_view(x._impl, [3], [1], 3))
    assert v.storage_offset() == 3
    assert v.as_strided((2,), (1,), 0).tolist() == [0.0, 1.0]
    assert v.as_strided((2,), (1,)).tolist() == [3.0, 4.0]
