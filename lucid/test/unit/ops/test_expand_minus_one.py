"""``-1`` in a target size keeps that dim, in ``expand`` and ``broadcast_to`` (LCD-547).

``broadcast_to(x, (-1, 6))`` raised ``InvalidArgument`` where the reference
keeps the size, and ``expand`` read a ``-1`` in a dim the input does not
have as a 1 where the reference refuses it.  Both read the size through one
helper now.
"""

from types import ModuleType

import pytest

import lucid

_KEEPS = [
    ((3, 1), (-1, 6)),
    ((3, 1), (2, -1, 6)),
    ((3, 1), (-1, -1)),
    ((2, 1, 4), (5, 2, -1, 4)),
    ((), (2, 3)),
]


@pytest.mark.parametrize("shape,size", _KEEPS, ids=str)
def test_minus_one_keeps_the_aligned_size(
    device: str, shape: tuple[int, ...], size: tuple[int, ...]
) -> None:
    x = (
        lucid.zeros(*shape, device=device)
        if shape
        else lucid.tensor(1.0, device=device)
    )
    lead = len(size) - len(shape)
    want = tuple(shape[i - lead] if s == -1 else s for i, s in enumerate(size))
    assert lucid.broadcast_to(x, size).shape == want
    assert x.broadcast_to(size).shape == want
    assert x.expand(*size).shape == want


@pytest.mark.parametrize("op", ["broadcast_to", "expand"])
def test_minus_one_in_a_new_leading_dim_is_refused(device: str, op: str) -> None:
    x = lucid.zeros(3, device=device)
    with pytest.raises(RuntimeError, match="leading, non-existing dimension 0"):
        getattr(x, op)((-1, 3))


@pytest.mark.parity
@pytest.mark.parametrize("shape,size", _KEEPS, ids=str)
def test_the_sizes_are_the_references(
    ref: ModuleType, shape: tuple[int, ...], size: tuple[int, ...]
) -> None:
    ours = lucid.broadcast_to(lucid.zeros(*shape) if shape else lucid.tensor(1.0), size)
    theirs = ref.broadcast_to(ref.zeros(*shape) if shape else ref.tensor(1.0), size)
    assert ours.shape == tuple(theirs.shape)
    with pytest.raises(RuntimeError):
        ref.zeros(3).expand(-1, 3)
