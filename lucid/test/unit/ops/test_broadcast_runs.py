"""Broadcasting on the CPU, walked in runs rather than elements.

The CPU backend used to materialise a broadcast one element at a time,
recomputing an N-d coordinate for each — 2.3 ns an element, so ``x * 2.0``
on half a million floats took fifty times a same-shape multiply, and every
bias add and scalar product in a model paid it.  It now copies or fills
whole runs, and the gradient sums them back the same way.

A copy is exact, so the forward is held bitwise against numpy.  The
backward sums, and a sum's bits depend on its order: each source element
must still receive its contributions in increasing output order, the order
the element walk used, so it is held bitwise against a sequential float32
sum in that order — a vectorised or pairwise sum would pass a tolerance and
fail this.
"""

import numpy as np
import pytest

import lucid

#: (source, destination): a scalar, a row, a column, a middle axis, several
#: axes at once, a leading axis the source lacks, trailing size-1 axes on
#: both sides, and an empty destination.
CASES = [
    ((), (4, 5, 6)),
    ((1,), (7,)),
    ((6,), (4, 5, 6)),
    ((4, 5, 1), (4, 5, 6)),
    ((4, 1, 6), (4, 5, 6)),
    ((1, 5, 1), (4, 5, 6)),
    ((5, 1), (3, 4, 5, 2)),
    ((1, 1, 1), (2, 3, 4)),
    ((3, 1), (3, 1)),
    ((2, 1, 3, 1), (2, 4, 3, 5)),
    ((6,), (0, 6)),
]


def _sequential_reduce(grad: np.ndarray, src_shape: tuple[int, ...]) -> np.ndarray:
    """Sum ``grad`` back onto ``src_shape``, one output element at a time."""
    out = np.zeros(src_shape, dtype=np.float32)
    padded = (1,) * (grad.ndim - len(src_shape)) + tuple(src_shape)
    flat = out.reshape(padded)
    for index in np.ndindex(*grad.shape):
        target = tuple(0 if size == 1 else i for i, size in zip(index, padded))
        flat[target] = np.float32(flat[target] + grad[index])
    return out


@pytest.mark.parametrize("src,dst", CASES)
def test_broadcast_copies_every_element_where_numpy_does(src, dst) -> None:
    rng = np.random.default_rng(0)
    source = rng.standard_normal(src).astype(np.float32)
    got = lucid.broadcast_to(lucid.tensor(source), dst).numpy()
    assert np.array_equal(got, np.broadcast_to(source, dst))


@pytest.mark.parametrize("src,dst", CASES)
def test_the_gradient_sums_in_output_order(src, dst) -> None:
    rng = np.random.default_rng(1)
    source = lucid.tensor(rng.standard_normal(src).astype(np.float32), requires_grad=True)
    upstream = rng.standard_normal(dst).astype(np.float32)
    lucid.broadcast_to(source, dst).backward(lucid.tensor(upstream))
    want = _sequential_reduce(upstream, src)
    assert source.grad is not None
    assert np.array_equal(source.grad.numpy().reshape(src), want)


@pytest.mark.parametrize("dtype", [lucid.float64, lucid.int32, lucid.int64, lucid.bool_])
def test_broadcast_keeps_every_dtype(dtype) -> None:
    source = lucid.tensor([[1], [0], [1]]).to(dtype)
    got = lucid.broadcast_to(source, (2, 3, 4)).numpy()
    assert np.array_equal(got, np.broadcast_to(source.numpy(), (2, 3, 4)))


@pytest.mark.parametrize("src,dst", [c for c in CASES if 0 not in c[1]])
def test_a_broadcasting_binary_op_sums_its_gradient_in_output_order(src, dst) -> None:
    rng = np.random.default_rng(2)
    small = lucid.tensor(rng.standard_normal(src).astype(np.float32), requires_grad=True)
    big = lucid.tensor(rng.standard_normal(dst).astype(np.float32))
    upstream = rng.standard_normal(dst).astype(np.float32)
    (small + big).backward(lucid.tensor(upstream))
    assert small.grad is not None
    assert np.array_equal(small.grad.numpy().reshape(src), _sequential_reduce(upstream, src))


@pytest.mark.parametrize(
    "shape,axes", [((4, 5, 6), (1,)), ((4, 5, 6), (0, 2)), ((4, 5, 6), (2,)), ((7,), (0,))]
)
@pytest.mark.parametrize("keepdim", [False, True])
def test_a_reductions_gradient_is_the_upstream_broadcast(shape, axes, keepdim) -> None:
    rng = np.random.default_rng(3)
    x = lucid.tensor(rng.standard_normal(shape).astype(np.float32), requires_grad=True)
    y = x.sum(dim=axes, keepdim=keepdim)
    upstream = rng.standard_normal(tuple(y.shape)).astype(np.float32)
    y.backward(lucid.tensor(upstream))
    kept = tuple(1 if d in axes else n for d, n in enumerate(shape))
    assert x.grad is not None
    assert np.array_equal(x.grad.numpy(), np.broadcast_to(upstream.reshape(kept), shape))
