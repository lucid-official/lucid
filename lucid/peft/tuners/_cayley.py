r"""The orthogonal-matrix pieces OFT and BOFT share.

Both learn skew-symmetric blocks :math:`Q` and turn them into orthogonal ones
with the Cayley transform :math:`R = (I - Q)(I + Q)^{-1}`; OFT lays its blocks
along a diagonal, BOFT lays a stack of them in butterfly order.  The parts that
do not depend on the method are here once: the map between a vector of free
entries and the skew-symmetric matrix it stands for, the exact and the
Neumann-series transform, the block-diagonal assembly, and the dropout that
swaps blocks for the identity.
"""

import functools
from typing import NamedTuple

import lucid
import lucid.linalg as linalg
from lucid._tensor.tensor import Tensor
from lucid._types import DeviceLike


class _SkewMap(NamedTuple):
    """Index tables between a block's strict upper triangle (a vector) and its flat ``n * n`` matrix."""

    #: For each of the ``n * n`` matrix entries, the vector entry it reads (``n_el`` for the zero diagonal).
    source: Tensor
    #: ``+1`` above the diagonal, ``-1`` below, ``0`` on it.
    sign: Tensor
    #: For each vector entry, the flat matrix position it was read from.
    upper: Tensor


@functools.cache
def _skew_map(block_size: int, device: DeviceLike) -> _SkewMap:
    """The index tables of ``block_size`` blocks on ``device`` (read-only, built once)."""
    n = block_size
    above = [(i, j) for i in range(n) for j in range(i + 1, n)]
    entry = {pair: k for k, pair in enumerate(above)}
    count = len(above)
    source = [
        entry[(i, j)] if i < j else entry[(j, i)] if i > j else count
        for i in range(n)
        for j in range(n)
    ]
    sign = [1.0 if i < j else -1.0 if i > j else 0.0 for i in range(n) for j in range(n)]
    flat = [i * n + j for i, j in above]
    return _SkewMap(
        lucid.tensor(source, dtype=lucid.int64, device=device),
        lucid.tensor(sign, device=device),
        lucid.tensor(flat, dtype=lucid.int64, device=device),
    )


class _Cayley:
    r"""Skew-symmetric blocks and their Cayley transform, batched over the blocks.

    Every method is static: a block's free entries are the caller's
    parameters, and nothing here holds state.
    """

    @staticmethod
    def skew(vector: Tensor, block_size: int) -> Tensor:
        """The ``(b, n, n)`` skew-symmetric matrices whose strict upper triangles are the rows of ``vector`` (``(b, n(n-1)/2)``)."""
        maps = _skew_map(block_size, vector.device)
        padded = lucid.cat([vector, lucid.zeros(vector.shape[0], 1, dtype=vector.dtype, device=vector.device)], dim=1)
        entries = padded[:, maps.source] * maps.sign.to(vector.dtype)
        return entries.reshape(vector.shape[0], block_size, block_size)

    @staticmethod
    def upper(skew: Tensor, block_size: int) -> Tensor:
        """The strict upper triangles of ``(b, n, n)`` matrices, as ``(b, n(n-1)/2)`` rows (inverse of :meth:`skew`)."""
        maps = _skew_map(block_size, skew.device)
        return skew.reshape(skew.shape[0], block_size * block_size)[:, maps.upper]

    @staticmethod
    def exact(skew: Tensor) -> Tensor:
        r""":math:`(I - Q)(I + Q)^{-1}` of each ``(b, n, n)`` matrix: orthogonal, exact, one batched solve.

        The solve runs in float32 and the result returns to ``skew``'s dtype.
        """
        wide = skew if skew.dtype == lucid.float32 else skew.to(lucid.float32)
        eye = lucid.eye(wide.shape[-1], dtype=wide.dtype, device=wide.device)
        return linalg.solve(eye + wide, eye - wide, left=False).to(skew.dtype)

    @staticmethod
    def neumann(skew: Tensor, terms: int) -> Tensor:
        r"""The Cayley transform as a truncated Neumann series, :math:`I + 2Q + 2Q^2 + \dots + Q^{terms-1}`, last term halved.

        Cheap (matrix products only) and orthogonal only up to the truncation.
        With ``terms = 1`` the identity; ``terms = 2``, :math:`I + 2Q`.
        """
        eye = lucid.eye(skew.shape[-1], dtype=skew.dtype, device=skew.device)
        result = eye.expand(*skew.shape)
        if terms > 1:
            result = result + 2.0 * skew
            if terms > 2:
                power = skew @ skew
                result = result + 2.0 * power
                for _ in range(3, terms - 1):
                    power = power @ skew
                    result = result + 2.0 * power
                power = power @ skew
                result = result + power
        return result


def _block_diagonal(blocks: Tensor, copies: int = 1) -> Tensor:
    """The block-diagonal matrix of ``blocks`` (``(D, n, n)``), or of the single block repeated ``copies`` times."""
    if blocks.shape[0] == 1:
        return lucid.block_diag(*[blocks[0]] * copies)
    return lucid.block_diag(*blocks.unbind(0))


class _IdentityDropout:
    """Training-time dropout that puts the identity in place of whole orthogonal blocks.

    ``int(p * D)`` of the ``D`` blocks, chosen at random, become the identity
    (the rotation of that block is skipped).  Stateless: the probability is the caller's.
    """

    @staticmethod
    def _chosen(count: int, probability: float, like: Tensor) -> Tensor:
        """A 0/1 vector of ``count`` entries with ``int(p * count)`` ones, in random places."""
        replaced = int(probability * count)
        flags = lucid.cat(
            [
                lucid.ones(replaced, dtype=like.dtype, device=like.device),
                lucid.zeros(count - replaced, dtype=like.dtype, device=like.device),
            ]
        )
        return flags[lucid.rand(count, device=like.device).argsort()]

    @classmethod
    def blocks(cls, blocks: Tensor, probability: float) -> Tensor:
        """``(D, n, n)`` blocks with some replaced by the identity; one block (shared) is left as it is."""
        count = blocks.shape[0]
        if count == 1:
            return blocks
        mask = cls._chosen(count, probability, blocks).reshape(count, 1, 1)
        eye = lucid.eye(blocks.shape[-1], dtype=blocks.dtype, device=blocks.device)
        return (1 - mask) * blocks + mask * eye

    @classmethod
    def butterfly(cls, stack: Tensor, probability: float) -> Tensor:
        """``(N, D, n, n)`` blocks with some of one randomly chosen factor's replaced by the identity."""
        factors, count = stack.shape[0], stack.shape[1]
        factor = int(lucid.randint(0, factors, (1,)).item())
        mask = cls._chosen(count, probability, stack).reshape(1, count, 1, 1)
        row = (lucid.arange(factors, device=stack.device) == factor).to(stack.dtype)
        full = row.reshape(factors, 1, 1, 1) * mask
        eye = lucid.eye(stack.shape[-1], dtype=stack.dtype, device=stack.device)
        return (1 - full) * stack + full * eye
