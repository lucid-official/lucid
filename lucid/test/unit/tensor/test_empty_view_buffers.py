"""An empty view's storage offset against a buffer that is null (LCD-1047).

A CPU tensor with no elements has no buffer (``allocate_aligned_bytes(0)`` is a null pointer), and
a view of one still carries the offset its slice or ``as_strided`` gave it: ``empty(2, 0, 3)[:, :, 1:]``
starts one element in, ``as_strided((0,), (1,), 17)`` seventeen.  ``nullptr + offset`` is undefined
behaviour for any non-zero offset, read or not.  Every route that turns a view's buffer and offset
into a pointer -- NumPy and DLPack (which goes through NumPy), ``repr``, ``tolist``, pickling, a
copy, a ``.grad`` assignment, an in-place write -- has to leave an empty view's pointer alone.
``core/PointerOffset.h`` is where that is decided; the nightly sanitizer job runs this directory
under UBSan, and what is asserted here is the answer each route must still give.
"""

import pickle
from collections.abc import Callable

import numpy as np
import pytest

import lucid

#: Empty views at a non-zero offset.  The first four are views of an empty tensor, whose buffer is
#: null; the last is a control, an empty view into a buffer that exists.
VIEWS: dict[str, Callable[[], lucid.Tensor]] = {
    "slice-from-1": lambda: lucid.empty(2, 0, 3)[:, :, 1:],
    "slice-from-2": lambda: lucid.empty(2, 0, 3)[:, :, 2:],
    "narrow-wide": lambda: lucid.empty(4, 0, 3).narrow(2, 1, 2),
    "transposed-slice": lambda: lucid.empty(0, 5).mT[2:],
    "as_strided-17": lambda: lucid.empty(0).as_strided((0,), (1,), 17),
    "as_strided-5": lambda: lucid.empty(2, 0, 3).as_strided((0, 3), (3, 1), 5),
    "control-live-buffer": lambda: lucid.arange(12.0).reshape(3, 4)[1:2, 3:3],
}


def _nested_empty(shape: tuple[int, ...]) -> list[object]:
    """``tolist`` of an empty tensor of ``shape``: lists down to the first axis of length 0."""
    if shape[0] == 0:
        return []
    return [_nested_empty(shape[1:]) for _ in range(shape[0])]


@pytest.fixture(params=list(VIEWS))
def empty_view(request: pytest.FixtureRequest, device_cpu_only: str) -> lucid.Tensor:
    view = VIEWS[request.param]()
    assert view.numel() == 0
    assert view.storage_offset() > 0, "the case no longer exercises an offset into nothing"
    return view


def test_numpy_of_an_empty_view_is_an_empty_array(empty_view: lucid.Tensor) -> None:
    got = empty_view.numpy()
    assert got.shape == tuple(empty_view.shape)
    assert got.dtype == np.float32
    assert got.size == 0


def test_dlpack_of_an_empty_view_reaches_the_consumer(empty_view: lucid.Tensor) -> None:
    capsule = empty_view.__dlpack__()
    assert type(capsule).__name__ == "PyCapsule"
    got = np.from_dlpack(empty_view)
    assert got.shape == tuple(empty_view.shape)
    assert got.size == 0


def test_repr_str_and_tolist_of_an_empty_view(empty_view: lucid.Tensor) -> None:
    shape = tuple(empty_view.shape)
    assert repr(empty_view).startswith("tensor(")
    assert str(empty_view).startswith("tensor(")
    assert empty_view.tolist() == _nested_empty(shape)


def test_a_copy_of_an_empty_view_keeps_its_shape_and_dtype(empty_view: lucid.Tensor) -> None:
    shape = tuple(empty_view.shape)
    for copied in (
        empty_view.clone(),
        empty_view.contiguous(),
        empty_view.detach(),
        empty_view.to(lucid.float64),
        pickle.loads(pickle.dumps(empty_view)),
    ):
        assert tuple(copied.shape) == shape
        assert copied.numel() == 0
    assert empty_view.to(lucid.float64).dtype == lucid.float64


def test_an_empty_view_can_be_written_in_place(empty_view: lucid.Tensor) -> None:
    shape = tuple(empty_view.shape)
    empty_view.fill_(1.0)
    empty_view.zero_()
    empty_view.copy_(lucid.zeros(*shape))
    empty_view.add_(1.0)
    empty_view.mul_(2.0)
    empty_view[...] = 0.0
    assert tuple(empty_view.shape) == shape
    assert empty_view.numel() == 0


def test_an_empty_view_can_be_assigned_as_a_gradient(empty_view: lucid.Tensor) -> None:
    """``Tensor.grad = view`` keeps a contiguous view's own bytes: a window at its offset."""
    shape = tuple(empty_view.shape)
    param = lucid.zeros(*shape, requires_grad=True)
    param.grad = empty_view
    assert param.grad is not None
    assert tuple(param.grad.shape) == shape


def test_an_op_on_an_empty_view_answers_as_on_a_dense_empty(empty_view: lucid.Tensor) -> None:
    shape = tuple(empty_view.shape)
    assert float(empty_view.sum().item()) == 0.0
    assert tuple((empty_view + 1.0).shape) == shape
    assert tuple(empty_view.reshape(-1).shape) == (0,)
    assert tuple(empty_view.unsqueeze(0).shape) == (1, *shape)
