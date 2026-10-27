"""Reference parity and contract tests for ``lucid.broadcast_shapes`` and
``lucid.broadcast_tensors`` (LCD-339).
"""

from types import ModuleType

import numpy as np
import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid._tensor._indexing import _broadcast_shape as _idx_broadcast_shape
from lucid.distributions._util import _broadcast_shapes as _dist_broadcast_shapes
from lucid.linalg import _broadcast_batch
from lucid.nn.functional._loss_checks import _broadcast_shape as _loss_broadcast_shape

pytestmark = pytest.mark.parity

_VALID_SHAPE_GROUPS: tuple[tuple[tuple[int, ...] | list[int], ...], ...] = (
    # 0 shapes
    (),
    # 1 shape (0-D, 1-D, N-D, empty dim 0)
    ((),),
    ((5,),),
    ([2, 3],),
    ((2, 0, 4),),
    # 2 shapes
    ((), ()),
    ((), (3, 4)),
    ((3, 1), (1, 4)),
    ([2, 1, 3], (4, 1)),
    ((1, 0), (2, 1)),
    ((1, 0), (2, 0)),
    ((0,), (1,)),
    # 3 shapes
    ((), (1, 4), (3, 1)),
    ((2, 1, 1), (1, 3, 1), (1, 1, 4)),
    ([5, 1, 1, 3], (1, 2, 1, 1), [1, 4, 3]),
    ((1, 0, 3), (2, 1, 1), (1, 0, 1)),
)

_INCOMPATIBLE_SHAPE_GROUPS: tuple[tuple[tuple[int, ...], ...], ...] = (
    ((2,), (3,)),
    ((2, 3), (2, 4)),
    ((2, 1, 3), (2, 4, 2)),
    ((1, 0), (2, 3)),
    ((0,), (2,)),
    ((2, 1), (1, 3), (4, 3)),
)

_NEGATIVE_SHAPE_GROUPS: tuple[tuple[tuple[int, ...], ...], ...] = (
    ((-1,),),
    ((2, -1),),
    ((2, 1), (1, -3)),
    ((1, 2), (2, 2), (-2, 1, 2)),
)


@pytest.mark.parametrize("shapes", _VALID_SHAPE_GROUPS)
def test_broadcast_shapes_matches_reference(
    ref: ModuleType, shapes: tuple[tuple[int, ...] | list[int], ...]
) -> None:
    got = lucid.broadcast_shapes(*shapes)
    want = tuple(ref.broadcast_shapes(*shapes))
    assert isinstance(got, tuple)
    assert got == want


@pytest.mark.parametrize("shapes", _INCOMPATIBLE_SHAPE_GROUPS)
def test_broadcast_shapes_incompatible_raises_like_reference(
    ref: ModuleType, shapes: tuple[tuple[int, ...], ...]
) -> None:
    with pytest.raises(RuntimeError):
        ref.broadcast_shapes(*shapes)
    with pytest.raises((_C_engine.ShapeMismatch, RuntimeError, ValueError)):
        lucid.broadcast_shapes(*shapes)


@pytest.mark.parametrize("shapes", _NEGATIVE_SHAPE_GROUPS)
def test_broadcast_shapes_negative_dim_raises_value_error(
    ref: ModuleType, shapes: tuple[tuple[int, ...], ...]
) -> None:
    with pytest.raises(ValueError):
        ref.broadcast_shapes(*shapes)
    with pytest.raises(ValueError):
        lucid.broadcast_shapes(*shapes)


def test_broadcast_tensors_empty_matches_reference(ref: ModuleType) -> None:
    assert lucid.broadcast_tensors() == tuple(ref.broadcast_tensors()) == ()


@pytest.mark.parametrize(
    "shapes",
    [
        ((),),
        ((3,),),
        ((2, 1), (1, 3)),
        ((), (2, 3)),
        ((2, 1, 1), (1, 3, 1), (1, 1, 4)),
        ((1, 0), (2, 1)),
        ((1, 0), (2, 0)),
    ],
)
def test_broadcast_tensors_forward_and_backward_match_reference(
    ref: ModuleType, device: str, shapes: tuple[tuple[int, ...], ...]
) -> None:
    rng = np.random.default_rng(42)
    arrays = [rng.standard_normal(s).astype(np.float32) for s in shapes]
    l_in = [lucid.tensor(a, device=device, requires_grad=True) for a in arrays]
    r_in = [ref.tensor(a, requires_grad=True) for a in arrays]

    l_out = lucid.broadcast_tensors(*l_in)
    r_out = ref.broadcast_tensors(*r_in)

    assert isinstance(l_out, tuple)
    assert len(l_out) == len(r_out)
    for lt, rt in zip(l_out, r_out, strict=True):
        assert lt.device.type == device
        assert tuple(lt.shape) == tuple(rt.shape)
        np.testing.assert_allclose(
            np.asarray(lt.detach().cpu().numpy()),
            np.asarray(rt.detach().cpu().numpy()),
            rtol=1e-5,
            atol=1e-6,
        )

    weights = [rng.standard_normal(tuple(r_out[0].shape)).astype(np.float32) for _ in shapes]
    l_loss = sum(
        (lt * lucid.tensor(w, device=device) for lt, w in zip(l_out, weights, strict=True)),
        start=lucid.zeros((), device=device),
    ).sum()
    r_loss = sum(
        (rt * ref.tensor(w) for rt, w in zip(r_out, weights, strict=True)), start=ref.zeros(())
    ).sum()
    l_loss.backward()
    r_loss.backward()

    for lt, rt in zip(l_in, r_in, strict=True):
        assert lt.grad is not None
        assert rt.grad is not None
        assert tuple(lt.grad.shape) == tuple(rt.grad.shape)
        np.testing.assert_allclose(
            np.asarray(lt.grad.detach().cpu().numpy()),
            np.asarray(rt.grad.detach().cpu().numpy()),
            rtol=1e-5,
            atol=1e-6,
        )


@pytest.mark.parametrize("shapes", _INCOMPATIBLE_SHAPE_GROUPS)
def test_broadcast_tensors_incompatible_raises_like_reference(
    ref: ModuleType, device: str, shapes: tuple[tuple[int, ...], ...]
) -> None:
    l_in = [lucid.zeros(s, device=device) for s in shapes]
    r_in = [ref.zeros(s) for s in shapes]
    with pytest.raises(RuntimeError):
        ref.broadcast_tensors(*r_in)
    with pytest.raises((_C_engine.ShapeMismatch, RuntimeError, ValueError)):
        lucid.broadcast_tensors(*l_in)


def test_broadcast_tensors_rejects_non_tensor(ref: ModuleType) -> None:
    with pytest.raises(TypeError):
        ref.broadcast_tensors(1)
    with pytest.raises(TypeError):
        getattr(lucid, "broadcast_tensors")(1)


def test_deduplicated_private_callers_delegate_to_broadcast_shapes() -> None:
    assert _idx_broadcast_shape([[2, 1], [1, 3]]) == [2, 3]
    with pytest.raises(IndexError):
        _idx_broadcast_shape([[2, 3], [2, 4]])

    assert _loss_broadcast_shape((2, 1), (1, 3)) == (2, 3)
    assert _loss_broadcast_shape((2, 3), (2, 4)) is None

    assert _dist_broadcast_shapes((2, 1), (1, 3)) == (2, 3)
    with pytest.raises(ValueError):
        _dist_broadcast_shapes((2, 3), (2, 4))

    assert _broadcast_batch((2, 1), (1, 3)) == (2, 3)
    with pytest.raises(ValueError):
        _broadcast_batch((2, 3), (2, 4))
