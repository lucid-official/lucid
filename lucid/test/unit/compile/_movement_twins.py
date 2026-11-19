"""Movement operations in their logical and canonical layouts."""

from collections.abc import Callable
import lucid
import lucid.nn.functional as F
from lucid.test.unit.compile._rank_twins import RankTwin, Shape, Side, _filled

Op = Callable[[lucid.Tensor], lucid.Tensor]
_WIDE = (2048, 4, 8, 4, 4)
_AROUND_AXIS = (8192, 8, 16)


def _expand_twin(x: lucid.Tensor) -> lucid.Tensor:
    # The second split still runs at its logical rank on the single selected outer row.
    row = x.reshape(2048, 512)[:1].reshape(1, 4, 8, 4, 4)
    return x + row[:, :, :1].reshape(1, 4, 1, 16).expand(*x.shape)


def _masked_fill(t: lucid.Tensor) -> lucid.Tensor:
    return (t * 1.0).masked_fill(t > 1.5, 0.0)


def _tril(t: lucid.Tensor) -> lucid.Tensor:
    return lucid.tril(t * 1.0)


def _pad(t: lucid.Tensor) -> lucid.Tensor:
    return F.pad(t * 1.0, (1, 1, 1, 1))


def _repeat(t: lucid.Tensor) -> lucid.Tensor:
    return lucid.repeat(t * 1.0, 2, dim=-1)


MOVEMENT_CASES: dict[str, tuple[Shape, Op, Op]] = {
    "masked_fill": ((1048576,), _masked_fill, _masked_fill),
    "flip": (
        _AROUND_AXIS,
        lambda t: lucid.flip(t * 1.0, (2,)),
        lambda t: lucid.flip(t * 1.0, (1,)),
    ),
    "tril": ((65536, 4, 4), _tril, _tril),
    "cat": (
        _AROUND_AXIS,
        lambda t: lucid.cat([t * 1.0, t * 2.0], dim=2),
        lambda t: lucid.cat([t * 1.0, t * 2.0], dim=1),
    ),
    "pad": ((65536, 4, 4), _pad, _pad),
    "tile": (
        _AROUND_AXIS,
        lambda t: lucid.tile(t * 1.0, (1, 1, 2, 1, 1)),
        lambda t: lucid.tile(t * 1.0, (1, 2, 1)),
    ),
    "repeat": ((262144, 4), _repeat, _repeat),
    "cumsum": (
        _AROUND_AXIS,
        lambda t: lucid.cumsum(t * 1.0, dim=2),
        lambda t: lucid.cumsum(t * 1.0, dim=1),
    ),
    "expand": ((2048, 4, 8, 16), lambda t: t + t[:1, :, :1].expand(*_WIDE), _expand_twin),
    "slice": (_AROUND_AXIS, lambda t: (t * 1.0)[:, :, 2:5], lambda t: (t * 1.0)[:, 2:5]),
    "stack": (
        (8192, 128),
        lambda t: lucid.stack([t * 1.0, t * 2.0], dim=2),
        lambda t: lucid.stack([t * 1.0, t * 2.0], dim=1),
    ),
}


def movement_twin(name: str, device: str) -> RankTwin:
    shape, wide, twin = MOVEMENT_CASES[name]

    def side(fn: Op, shape: Shape) -> Side:
        return Side(lambda t: fn(t).reshape(-1).sum(), (_filled(shape).to(device),))

    return RankTwin(name, side(wide, _WIDE), side(twin, shape))


def _space_to_depth(x: lucid.Tensor, patch: int = 2) -> lucid.Tensor:
    b, c, h, w = x.shape
    v = x.reshape(b, c, h // patch, patch, w // patch, patch).permute(0, 3, 5, 1, 2, 4)
    return v.reshape(b, patch * patch * c, h // patch, w // patch)


def space_to_depth_side(device: str) -> Side:
    return Side(lambda t: _space_to_depth(t).sum(), (_filled((96, 64, 32, 64)).to(device),))
