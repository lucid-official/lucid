"""An empty selection: topk with k = 0, kthvalue on an empty axis (LCD-378).

``topk(x, 0)`` raised "k must be in (0, axis_size]".  The reference takes
k in [0, axis_size]: k = 0 gives empty values and indices, the values
still carry a grad_fn, and their backward is a zero gradient.

``kthvalue`` of an empty axis has no k-th value whatever k is.  The
reference calls that an index error, as it does for argmax of an empty
axis; Lucid raised InvalidArgument, the class of a bad k.
"""

import numpy as np
import pytest

import lucid

_CASES = [
    ((4,), -1),
    ((0,), 0),
    ((2, 3), -1),
    ((2, 3), 0),
    ((2, 0, 3), 1),
    ((2, 0, 3), 2),
]


@pytest.mark.parametrize("largest", [True, False])
@pytest.mark.parametrize(("shape", "dim"), _CASES, ids=str)
def test_topk_k0_matches_reference(ref, device, shape, dim, largest):
    values = np.random.default_rng(378).standard_normal(shape).astype(np.float32)
    x = lucid.tensor(values, device=device, requires_grad=True)
    xr = ref.tensor(values, requires_grad=True)
    got_v, got_i = lucid.topk(x, 0, dim=dim, largest=largest)
    want_v, want_i = ref.topk(xr, 0, dim=dim, largest=largest)
    assert tuple(got_v.shape) == tuple(want_v.shape)
    assert tuple(got_i.shape) == tuple(want_i.shape)
    assert got_i.dtype == lucid.int64
    assert got_v.requires_grad and want_v.requires_grad
    got_v.sum().backward()
    want_v.sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), xr.grad.numpy())


@pytest.mark.parametrize("k", [-1, 4])
def test_topk_k_outside_the_axis_is_refused(ref, device, k):
    with pytest.raises(RuntimeError):
        ref.topk(ref.zeros(3), k)
    with pytest.raises(RuntimeError, match="k out of range"):
        lucid.topk(lucid.zeros(3, device=device), k)


@pytest.mark.parametrize(("shape", "dim"), [((0,), 0), ((2, 0), 1), ((0, 3), 0)], ids=str)
def test_kthvalue_of_an_empty_axis_is_an_index_error(ref, device, shape, dim):
    with pytest.raises(IndexError):
        ref.kthvalue(ref.zeros(shape), 1, dim=dim)
    with pytest.raises(IndexError, match="non-zero size"):
        lucid.kthvalue(lucid.zeros(shape, device=device), 1, dim=dim)


@pytest.mark.parametrize("k", [0, 4])
def test_kthvalue_k_outside_the_axis_is_still_a_bad_k(ref, device, k):
    with pytest.raises(RuntimeError):
        ref.kthvalue(ref.zeros(3), k)
    with pytest.raises(ValueError, match="k out of range"):
        lucid.kthvalue(lucid.zeros(3, device=device), k)
