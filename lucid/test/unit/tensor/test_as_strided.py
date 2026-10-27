"""``as_strided`` views a tensor's storage through chosen sizes and strides.

Element units, as the reference's: ``storage_offset`` counts from the start
of the storage and defaults to the tensor's own.  On the CPU the result is a
view sharing the buffer; on metal it is a copy of the elements it names, and
a write through it is carried back to the tensor.  A write through elements
that overlap is refused, and a gradient scatters back to the elements read.
"""

import pytest

import lucid


def _flat(t: lucid.Tensor) -> list[float]:
    return [float(v) for v in t.reshape(-1).tolist()]


def test_it_reads_the_elements_its_strides_name() -> None:
    x = lucid.arange(12).float()
    assert lucid.as_strided(x, (2, 3), (4, 1)).tolist() == [[0.0, 1.0, 2.0], [4.0, 5.0, 6.0]]


def test_the_method_form_with_an_explicit_offset() -> None:
    x = lucid.arange(12).float()
    assert x.as_strided((2,), (5,), 1).tolist() == [1.0, 6.0]


def test_the_offset_counts_from_the_storage_not_the_view() -> None:
    x = lucid.arange(12).float().reshape(3, 4)
    col = x[:, 1]  # starts one element into the buffer
    assert col.as_strided((2,), (4,)).tolist() == [1.0, 5.0]
    assert col.as_strided((2,), (1,), 0).tolist() == [0.0, 1.0]


def test_it_is_a_view_on_the_cpu() -> None:
    x = lucid.arange(6).float()
    x.as_strided((3,), (2,)).fill_(-1.0)
    assert _flat(x) == [-1.0, 1.0, -1.0, 3.0, -1.0, 5.0]


def test_overlapping_elements_refuse_a_write() -> None:
    x = lucid.arange(6).float()
    v = x.as_strided((2, 2), (1, 1))
    with pytest.raises(Exception, match="overlap"):
        v.add_(1.0)


def test_it_refuses_a_view_past_the_storage() -> None:
    x = lucid.arange(6).float()
    with pytest.raises(Exception, match="past the end"):
        x.as_strided((3,), (3,))


def test_the_gradient_collects_at_the_elements_read() -> None:
    w = lucid.ones(6, requires_grad=True)
    lucid.as_strided(w, (2, 2), (1, 1)).sum().backward()  # reads 0, 1, 1, 2
    assert w.grad is not None
    assert _flat(w.grad) == [1.0, 2.0, 1.0, 0.0, 0.0, 0.0]


@pytest.mark.skipif(not lucid.metal.is_available(), reason="no Metal device")
def test_metal_gives_the_same_values_and_writes_through_like_cpu() -> None:
    """A write through ``as_strided`` reaches the tensor on Metal, as on the CPU.

    This test used to assert the Metal write was lost — it pinned the defect
    of debug-metal-view-writes-silent (CHA-73) as if it were the intent.
    """
    x = lucid.arange(12).float()
    m = x.to("metal")
    want = lucid.as_strided(x, (2, 3), (4, 1)).tolist()
    assert lucid.as_strided(m, (2, 3), (4, 1)).to("cpu").tolist() == want
    lucid.as_strided(x, (2,), (1,)).fill_(-1.0)
    lucid.as_strided(m, (2,), (1,)).fill_(-1.0)
    assert x.tolist()[:3] == [-1.0, -1.0, 2.0]
    assert m.to("cpu").tolist() == x.tolist()
