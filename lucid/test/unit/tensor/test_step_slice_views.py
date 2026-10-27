"""Positive-step slices are CPU views, as the reference's are.

``x[::2]`` copied — it gathered the elements it named — so a write through
it never reached ``x``.  On the CPU a positive step is now the run it covers,
one-element windows every ``step`` along the axis, and the window axis
squeezed away: split_at, unfold_dim and squeeze each make a view.  A negative
step, which the reference does not have, still copies.  On Metal the slice is
its own array and a write through it is carried back to its base.
"""

import pytest

import lucid


def _flat(t: lucid.Tensor) -> list[float]:
    return [float(v) for v in t.reshape(-1).tolist()]


@pytest.mark.parametrize(
    ("start", "stop", "step"),
    [
        (None, None, 2),
        (1, None, 2),
        (0, 9, 3),
        (2, 8, 4),
        (5, None, 7),
        (0, 1, 5),
        (None, 7, 2),
        (-4, None, 3),
    ],
)
def test_a_positive_step_reads_what_python_would(
    start: int | None, stop: int | None, step: int
) -> None:
    x = lucid.arange(10).float()
    want = [float(v) for v in list(range(10))[start:stop:step]]
    assert _flat(x[start:stop:step]) == want


def test_a_step_slice_is_a_view() -> None:
    x = lucid.arange(10).float()
    v = x[::2]
    v.fill_(-1.0)
    assert _flat(x) == [-1.0, 1.0, -1.0, 3.0, -1.0, 5.0, -1.0, 7.0, -1.0, 9.0]
    x.add_(10.0)
    assert _flat(v) == [9.0] * 5


def test_a_step_along_a_later_axis_is_a_view() -> None:
    x = lucid.arange(12).float().reshape(3, 4)
    x[:, 1::2].zero_()
    assert x.tolist() == [[0.0, 0.0, 2.0, 0.0], [4.0, 0.0, 6.0, 0.0], [8.0, 0.0, 10.0, 0.0]]


def test_a_step_slice_of_a_transposed_view_reads_through_both() -> None:
    t = lucid.arange(12).float().reshape(3, 4).T
    assert t[::2].tolist() == t.contiguous()[::2].tolist()


def test_the_gradient_lands_on_the_stepped_elements() -> None:
    w = lucid.ones(6, requires_grad=True)
    (w[1::2] * 2.0).sum().backward()
    assert w.grad is not None
    assert _flat(w.grad) == [0.0, 2.0, 0.0, 2.0, 0.0, 2.0]


def test_a_recorded_write_through_a_step_slice_reaches_the_gradient() -> None:
    w = lucid.ones(6, requires_grad=True)
    h = w * 1.0
    h[::2].mul_(3.0)
    h.sum().backward()
    assert w.grad is not None
    assert _flat(w.grad) == [3.0, 1.0, 3.0, 1.0, 3.0, 1.0]


def test_assignment_through_a_step_still_writes() -> None:
    x = lucid.zeros(6)
    x[::3] = 5.0
    assert _flat(x) == [5.0, 0.0, 0.0, 5.0, 0.0, 0.0]


def test_a_negative_step_reads_right_and_stays_a_copy() -> None:
    x = lucid.arange(6).float()
    r = x[::-2]
    assert _flat(r) == [5.0, 3.0, 1.0]
    r.fill_(0.0)
    assert _flat(x) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


@pytest.mark.skipif(not lucid.metal.is_available(), reason="no Metal device")
def test_a_write_through_a_metal_step_slice_reaches_its_base() -> None:
    x = lucid.arange(6).float().to("metal")
    x[::2].fill_(-1.0)
    assert _flat(x.to("cpu")) == [-1.0, 1.0, -1.0, 3.0, -1.0, 5.0]
