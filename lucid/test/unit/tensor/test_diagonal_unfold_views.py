"""``diagonal`` and ``unfold`` are CPU views, as the reference's are.

Both copied: a write through a diagonal or a window never reached the
tensor it came from.  They now share its buffer.  Windows that overlap
(``step < size``) repeat elements, so a write through them is refused, and
the overlap test behind that refusal is exact: strides that interleave
without landing on the same byte are not refused.  On Metal both are their
own arrays; a write through them is carried back to the base, and windows
that overlap are refused there too.
"""

import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid._dispatch import _wrap


def _flat(t: lucid.Tensor) -> list[float]:
    return [float(v) for v in t.reshape(-1).tolist()]


def _grid() -> lucid.Tensor:
    return lucid.arange(9).float().reshape(3, 3)


def _strided(base: lucid.Tensor, shape: list[int], stride: list[int]) -> lucid.Tensor:
    return _wrap(_C_engine.TensorImpl._make_view(base._impl, shape, stride, 0))


def test_a_diagonal_is_a_view() -> None:
    x = _grid()
    d = x.diagonal()
    d.fill_(-1.0)
    assert x.tolist() == [[-1.0, 1.0, 2.0], [3.0, -1.0, 5.0], [6.0, 7.0, -1.0]]
    x.add_(10.0)
    assert d.tolist() == [9.0, 9.0, 9.0]


@pytest.mark.parametrize("offset", [-3, -2, -1, 0, 1, 2, 4])
def test_an_offset_diagonal_reads_the_right_elements(offset: int) -> None:
    x = lucid.arange(12).float().reshape(3, 4)
    want = [float(r * 4 + r + offset) for r in range(3) if 0 <= r + offset < 4]
    assert _flat(x.diagonal(offset=offset)) == want


def test_swapped_axes_give_the_diagonal_at_the_negated_offset() -> None:
    x = lucid.arange(12).float().reshape(3, 4)
    assert _flat(x.diagonal(offset=1, dim1=1, dim2=0)) == [4.0, 9.0]
    assert _flat(x.diagonal(offset=-1, dim1=0, dim2=1)) == [4.0, 9.0]


def test_the_default_axes_are_the_last_two() -> None:
    # Lucid's diagonal is the trailing pair at any rank; the compile and
    # Core ML emitters are built on that.
    x = lucid.arange(24).float().reshape(2, 3, 4)
    d = x.diagonal()
    assert tuple(d.shape) == (2, 3)
    assert d.tolist() == x.diagonal(dim1=-2, dim2=-1).tolist()


def test_a_diagonal_of_a_transposed_view_reads_through_both() -> None:
    t = lucid.arange(12).float().reshape(3, 4).T
    assert _flat(t.diagonal(offset=1)) == [4.0, 9.0]
    assert t.diagonal(offset=1).tolist() == t.contiguous().diagonal(offset=1).tolist()


def test_a_recorded_write_through_a_diagonal_reaches_the_gradient() -> None:
    w = lucid.ones(3, 3, requires_grad=True)
    h = w * 1.0
    h.diagonal().mul_(3.0)
    h.sum().backward()
    assert w.grad is not None
    assert w.grad.tolist() == [[3.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 3.0]]


def test_non_overlapping_windows_are_a_writable_view() -> None:
    x = lucid.arange(10).float()
    u = x.unfold(0, 2, 2)
    assert tuple(u.shape) == (5, 2)
    u[1].fill_(0.0)
    assert _flat(x)[2:4] == [0.0, 0.0]


def test_overlapping_windows_read_but_refuse_a_write() -> None:
    x = lucid.arange(6).float()
    u = x.unfold(0, 3, 1)
    assert u.tolist() == [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]]
    with pytest.raises(Exception, match="overlap"):
        u.add_(1.0)


def test_unfold_gradient_counts_each_elements_windows() -> None:
    x = lucid.ones(6, requires_grad=True)
    x.unfold(0, 3, 1).sum().backward()
    assert x.grad is not None
    assert _flat(x.grad) == [1.0, 2.0, 3.0, 3.0, 2.0, 1.0]


def test_unfold_of_a_transposed_view_matches_the_packed_input() -> None:
    t = lucid.arange(12).float().reshape(3, 4).T
    assert t.unfold(1, 2, 1).tolist() == t.contiguous().unfold(1, 2, 1).tolist()


def test_interleaving_strides_that_never_collide_stay_writable() -> None:
    base = lucid.zeros(20)
    # addresses 2i + 3j for i, j in 0..2 — all nine distinct
    _strided(base, [3, 3], [2, 3]).fill_(1.0)
    assert sum(_flat(base)) == 9.0


def test_interleaving_strides_that_collide_refuse_a_write() -> None:
    base = lucid.zeros(20)
    # i = 3, j = 0 and i = 0, j = 2 both land on address 6
    with pytest.raises(Exception, match="overlap"):
        _strided(base, [4, 3], [2, 3]).add_(1.0)


def test_make_view_bounds_count_the_bases_own_offset() -> None:
    x = lucid.arange(12).float().reshape(3, 4)
    column = x[:, 3]  # starts three elements into the buffer
    with pytest.raises(ValueError, match="past the end"):
        _C_engine.TensorImpl._make_view(column._impl, [1], [1], 9)


@pytest.mark.skipif(not lucid.metal.is_available(), reason="no Metal device")
def test_a_write_through_metal_diagonal_and_unfold_reaches_the_base() -> None:
    x = _grid().to("metal")
    x.diagonal().fill_(-1.0)
    assert x.to("cpu").diagonal().tolist() == [-1.0, -1.0, -1.0]
    x.reshape(-1).unfold(0, 3, 3)[1].fill_(-2.0)  # windows that do not overlap
    assert x.to("cpu").tolist()[1] == [-2.0, -2.0, -2.0]
    with pytest.raises(RuntimeError, match="overlap"):
        x.reshape(-1).unfold(0, 3, 2).add_(1.0)
