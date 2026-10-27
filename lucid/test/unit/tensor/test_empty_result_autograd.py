"""An empty result stays in the graph (LCD-361).

A selection that selects nothing is still a function of its input.  The
reference keeps such a result in the graph, and backward hands the input a
gradient of zeros of its own shape.  ``_select_slice`` used to answer an
empty slice with a fresh ``zeros``.  The result had no grad_fn, so a module
that slices its input (GRUCell's gates at batch 0) lost its weights'
gradients, and backward through it raised.

Every selection form below has to keep the edge: basic and step slices, a
slice of an empty source, narrow, a split or chunk piece, index_select or
gather with an empty index, and a selection by an all-false mask.
"""

from collections.abc import Callable
from types import ModuleType

import numpy as np
import pytest

import lucid

# A selection receives the framework module, the leaf and its device, and
# returns the empty result.
_Select = Callable[[ModuleType, object, str], object]


def _empty_index(fw: ModuleType, device: str, *shape: int) -> object:
    return fw.zeros(*shape, 0, dtype=fw.int64, device=device)


def _all_false(x: object) -> object:
    return x.detach() > 1e9


# name → (leaf shape, selection, the reference has the form)
_CASES: dict[str, tuple[tuple[int, ...], _Select, bool]] = {
    "slice_past_end": ((2, 4), lambda fw, x, d: x[:, 4:], True),
    "slice_backwards": ((2, 4), lambda fw, x, d: x[:, 3:1], True),
    "slice_of_empty_source": ((0, 4), lambda fw, x, d: x[:, :3], True),
    "step_slice": ((2, 4), lambda fw, x, d: x[:, 4::2], True),
    # The reference has no negative step; the edge is Lucid's own contract.
    "negative_step_slice": ((2, 4), lambda fw, x, d: x[:, 1:3:-1], False),
    "narrow": ((2, 4), lambda fw, x, d: fw.narrow(x, 1, 2, 0), True),
    "split_piece": ((2, 4), lambda fw, x, d: fw.split(x, [4, 0], 1)[1], True),
    "chunk_piece": ((0, 4), lambda fw, x, d: fw.chunk(x, 2, 0)[0], True),
    "index_select_empty": (
        (2, 4),
        lambda fw, x, d: fw.index_select(x, 1, _empty_index(fw, d)),
        True,
    ),
    "gather_empty": (
        (2, 4),
        lambda fw, x, d: fw.gather(x, 1, _empty_index(fw, d, 2)),
        True,
    ),
    "masked_select_all_false": (
        (2, 4),
        lambda fw, x, d: fw.masked_select(x, _all_false(x)),
        True,
    ),
    "getitem_empty_index": ((2, 4), lambda fw, x, d: x[_empty_index(fw, d)], True),
    "getitem_all_false_mask": ((2, 4), lambda fw, x, d: x[_all_false(x)], True),
}
_REF_CASES = [name for name, (_, _, in_ref) in _CASES.items() if in_ref]


def _run(fw: ModuleType, name: str, device: str) -> tuple[object, object]:
    """The empty result of case ``name`` and the leaf's gradient after
    backpropagating its sum."""
    shape, select, _ = _CASES[name]
    x = fw.randn(*shape, device=device).requires_grad_(True)
    y = select(fw, x, device)
    assert y.numel() == 0
    assert y.requires_grad, f"{name}: the empty result left the graph"
    y.sum().backward()
    return y, x.grad


@pytest.mark.parametrize("name", list(_CASES))
def test_empty_result_keeps_autograd_edge(name: str, device: str) -> None:
    _, grad = _run(lucid, name, device)
    shape = _CASES[name][0]
    assert isinstance(grad, lucid.Tensor), f"{name}: backward did not reach the leaf"
    assert tuple(grad.shape) == shape
    np.testing.assert_array_equal(grad.numpy(), np.zeros(shape, dtype=np.float32))


@pytest.mark.parametrize("name", _REF_CASES)
def test_empty_result_matches_reference(ref: ModuleType, name: str, device: str) -> None:
    y, grad = _run(lucid, name, device)
    want_y, want_grad = _run(ref, name, "cpu")
    assert tuple(y.shape) == tuple(want_y.shape)
    assert isinstance(grad, lucid.Tensor)
    np.testing.assert_array_equal(grad.numpy(), want_grad.numpy())


# A whole-tensor nanmedian with no number to take (every entry NaN, or none
# at all) answered with a fresh NaN constant: the same cut edge, reached
# through a reduction.  The reference spreads the gradient evenly.
@pytest.mark.parametrize("size", [3, 0], ids=["all_nan", "empty"])
def test_nanmedian_without_a_number_matches_reference(
    ref: ModuleType, size: int, device: str
) -> None:
    x = lucid.full((size,), float("nan"), device=device).requires_grad_(True)
    want_x = ref.full((size,), float("nan"), requires_grad=True)
    y, want_y = lucid.nanmedian(x), ref.nanmedian(want_x)
    assert y.requires_grad
    assert np.isnan(y.item()) and np.isnan(want_y.item())
    y.backward()
    want_y.backward()
    assert isinstance(x.grad, lucid.Tensor)
    np.testing.assert_allclose(x.grad.numpy(), want_x.grad.numpy())
