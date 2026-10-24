"""Slices along any axis and ``expand`` are CPU views, as the reference's are.

Only dim-0 slices shared their tensor's buffer; a column slice, a ``split``
along columns and ``expand`` were copies, so a write through one never
reached the tensor it came from.  Every op now reads a strided CPU view
correctly and a write through one follows its strides, so these are views
too.  On Metal a slice is a member of its base's view family in the engine,
as on the CPU, so a write through it ends the same way.
"""

import pytest

import lucid


def _grid() -> lucid.Tensor:
    return lucid.arange(12).float().reshape(3, 4)


def _flat(t: lucid.Tensor) -> list[float]:
    return [float(v) for v in t.reshape(-1).tolist()]


def test_a_column_slice_is_a_view() -> None:
    x = _grid()
    c = x[:, 1:3]
    assert not c.is_contiguous()
    c.mul_(10.0)
    assert x.tolist() == [
        [0.0, 10.0, 20.0, 3.0],
        [4.0, 50.0, 60.0, 7.0],
        [8.0, 90.0, 100.0, 11.0],
    ]
    x.add_(1.0)
    assert c.tolist() == [[11.0, 21.0], [51.0, 61.0], [91.0, 101.0]]


def test_an_indexed_column_is_a_view() -> None:
    x = _grid()
    col = x[:, 2]
    col.fill_(-1.0)
    assert x.tolist()[0] == [0.0, 1.0, -1.0, 3.0]
    assert x[:, 2].tolist() == [-1.0, -1.0, -1.0]


def test_split_and_unbind_along_columns_are_views() -> None:
    x = _grid()
    _, right = lucid.split(x, 2, dim=1)
    right.zero_()
    assert x.tolist()[1] == [4.0, 5.0, 0.0, 0.0]
    cols = lucid.unbind(x, dim=1)
    cols[0].fill_(9.0)
    assert [row[0] for row in x.tolist()] == [9.0, 9.0, 9.0]


def test_a_recorded_write_through_a_column_slice_reaches_the_gradient() -> None:
    w = lucid.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    h = w * 1.0
    col = h[:, 1:2]
    col.mul_(3.0)
    assert _flat(h) == [1.0, 6.0, 3.0, 4.0, 15.0, 6.0]
    h.sum().backward()
    assert w.grad is not None
    assert _flat(w.grad) == [1.0, 3.0, 1.0, 1.0, 3.0, 1.0]


def test_expand_is_a_read_only_view() -> None:
    x = lucid.tensor([1.0, 2.0, 3.0])
    e = x.expand(2, 3)
    assert e.tolist() == [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    x.mul_(2.0)
    assert e.tolist() == [[2.0, 4.0, 6.0], [2.0, 4.0, 6.0]]
    with pytest.raises(Exception, match="overlap"):
        e.add_(1.0)


def test_expand_s_gradient_sums_over_the_broadcast_axis() -> None:
    w = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True)
    (w.expand(4, 3) * 2.0).sum().backward()
    assert w.grad is not None
    assert _flat(w.grad) == [8.0, 8.0, 8.0]


@pytest.mark.skipif(not lucid.metal.is_available(), reason="no Metal device")
def test_a_write_through_a_metal_slice_reaches_its_base() -> None:
    x = _grid().to("metal")
    c = x[:, 1:3]
    c.mul_(10.0)
    want = _grid()
    want[:, 1:3].mul_(10.0)
    assert x.to("cpu").tolist() == want.tolist()
