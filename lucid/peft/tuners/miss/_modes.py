r"""The three shapes a MiSS adapter takes: how each lays its block out and writes it into a weight.

An adapter holds one block, ``miss_block``, whose shape and meaning depend on the
variant the config's ``init_weights`` names.  A ``_Mode`` is the strategy for one
variant — the block's shape, what it needs of the layer, how it rewrites a weight and
takes the rewrite back, and, where the update is a plain addition to the weight, what it
adds to a layer's output — registered under its key as it is defined, so the layer
branches on nothing and a new variant is one subclass.

* ``balance`` — an ``(r, d_out)`` block; ``\Delta W_{o,c}`` is its entry ``(c mod r, o)``.
* ``mini`` — an ``(r, m)`` block tiled ``d_out / m`` times along the output axis, then as ``balance``.
* ``bat`` — one ``(r, r)`` block per run of ``r`` output rows, ``W' = W (I + S) + S`` blockwise.
"""

from abc import ABC, abstractmethod
from typing import ClassVar, Literal, override

import lucid
import lucid.linalg as linalg
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor

_Key = Literal["balance", "mini", "bat"]


class _Mode(ABC):
    """How one variant of MiSS shapes its block and applies it."""

    __slots__ = ()

    _modes: ClassVar[dict[str, _Mode]] = {}

    @override
    def __init_subclass__(cls, *, key: _Key | None = None) -> None:
        super().__init_subclass__()
        if key is not None:
            _Mode._modes[key] = cls()

    @staticmethod
    def named(key: str) -> _Mode:
        """The variant registered under ``key``."""
        return _Mode._modes[key]

    @staticmethod
    def of(init_weights: bool | str) -> _Key:
        """The key of the variant ``init_weights`` names (``True`` and ``False`` are both ``balance``)."""
        if init_weights == "bat":
            return "bat"
        return "mini" if init_weights == "mini" else "balance"

    #: Whether the update is a weight-independent addition (so it can be applied to an input).
    additive: ClassVar[bool] = True

    def check(self, in_features: int, out_features: int, r: int, mini_r: int) -> None:
        """Raise ``ValueError`` if a layer of these sizes cannot take a block of rank ``r``."""
        if r > in_features:
            raise ValueError(f"`r` ({r}) must be less than or equal to in_features ({in_features})")
        self._check(in_features, out_features, r, mini_r)

    def _check(self, in_features: int, out_features: int, r: int, mini_r: int) -> None:
        del in_features, out_features, r, mini_r  # most variants ask nothing more

    @abstractmethod
    def block_shape(self, out_features: int, r: int, mini_r: int) -> tuple[int, ...]:
        """The shape of the block of a layer with ``out_features`` outputs."""

    @abstractmethod
    def rewritten(self, weight: Tensor, block: Tensor) -> Tensor:
        """``weight`` with the block applied, in the block's dtype."""

    @abstractmethod
    def reverted(self, weight: Tensor, block: Tensor) -> Tensor:
        """``weight`` with the block taken back out."""

    def delta(self, weight: Tensor, block: Tensor) -> Tensor:
        """The change the block makes to ``weight``."""
        return self.rewritten(weight, block) - weight.to(block.dtype)

    def contribution(self, x: Tensor, block: Tensor, out_features: int) -> Tensor:
        """The addition to the layer's output for the input ``x`` (additive variants only)."""
        raise NotImplementedError(
            f"{type(self).__name__} rewrites the weight; it adds nothing to an output"
        )


class _Shards(_Mode):
    """The variants whose update adds the same ``r``-vector to every run of ``r`` columns of a row."""

    __slots__ = ()

    @abstractmethod
    def full(self, block: Tensor, out_features: int) -> Tensor:
        """The ``(r, d_out)`` matrix the block stands for."""

    @override
    def delta(self, weight: Tensor, block: Tensor) -> Tensor:
        """The ``(d_out, d_in)`` change, whatever the weight: entry ``(o, c)`` is ``full[c mod r, o]``."""
        out_features, in_features = weight.shape
        r = block.shape[0]
        columns = self.full(block, out_features).mT.repeat(1, -(-in_features // r))
        return columns[:, :in_features]

    @override
    def rewritten(self, weight: Tensor, block: Tensor) -> Tensor:
        return weight.to(block.dtype) + self.delta(weight, block)

    @override
    def reverted(self, weight: Tensor, block: Tensor) -> Tensor:
        return weight.to(block.dtype) - self.delta(weight, block)

    @override
    def contribution(self, x: Tensor, block: Tensor, out_features: int) -> Tensor:
        r = block.shape[0]
        pad = (-x.shape[-1]) % r
        x = F.pad(x, (0, pad)) if pad else x
        x = x.to(block.dtype)
        shards = x.reshape(*x.shape[:-1], x.shape[-1] // r, r).sum(-2)
        return shards @ self.full(block, out_features)


class _Balance(_Shards, key="balance"):
    @override
    def block_shape(self, out_features: int, r: int, mini_r: int) -> tuple[int, ...]:
        del mini_r  # the block spans every output
        return (r, out_features)

    @override
    def full(self, block: Tensor, out_features: int) -> Tensor:
        del out_features  # the block already spans every output
        return block


class _Mini(_Shards, key="mini"):
    @override
    def _check(self, in_features: int, out_features: int, r: int, mini_r: int) -> None:
        if out_features % mini_r != 0:
            raise ValueError(
                "mini_r is divided along the out_features dimension. For optimal performance and "
                "implementation simplicity, it is recommended that out_features be divisible by "
                f"mini_r. Error: {out_features} % mini_r != 0"
            )

    @override
    def block_shape(self, out_features: int, r: int, mini_r: int) -> tuple[int, ...]:
        del out_features  # the block is tiled across the outputs
        return (r, mini_r)

    @override
    def full(self, block: Tensor, out_features: int) -> Tensor:
        return block.repeat(1, out_features // block.shape[1])


class _Bat(_Mode, key="bat"):
    additive: ClassVar[bool] = False

    @override
    def _check(self, in_features: int, out_features: int, r: int, mini_r: int) -> None:
        if in_features % r != 0 or out_features % r != 0:
            raise ValueError("The weight matrix must be fully divisible into [r, r] blocks.")

    @override
    def block_shape(self, out_features: int, r: int, mini_r: int) -> tuple[int, ...]:
        del mini_r  # one square block per run of r output rows
        return (out_features // r, r, r)

    @staticmethod
    def _tiles(weight: Tensor, r: int) -> Tensor:
        """``weight`` as ``(d_in / r, d_out / r, r, r)`` tiles: entry ``[k, l]`` is rows ``l r..`` by columns ``k r..``."""
        out_features, in_features = weight.shape
        return weight.reshape(out_features // r, r, in_features // r, r).permute(2, 0, 1, 3)

    @staticmethod
    def _untiled(tiles: Tensor) -> Tensor:
        in_blocks, out_blocks, r, _ = tiles.shape
        return tiles.permute(1, 2, 0, 3).reshape(out_blocks * r, in_blocks * r)

    @override
    def rewritten(self, weight: Tensor, block: Tensor) -> Tensor:
        tiles = self._tiles(weight.to(block.dtype), block.shape[-1])
        return self._untiled(tiles + (tiles @ block + block))

    @override
    def reverted(self, weight: Tensor, block: Tensor) -> Tensor:
        r = block.shape[-1]
        eye = lucid.eye(r, device=block.device, dtype=block.dtype)
        tiles = self._tiles(weight.to(block.dtype), r)
        return self._untiled((tiles - block) @ linalg.inv(eye + block))


# The variants register themselves as they are defined; naming them here is how the dead-code
# gate sees that they are used.
__all__ = ["_Balance", "_Bat", "_Mini", "_Mode"]
