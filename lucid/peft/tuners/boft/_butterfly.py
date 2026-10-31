r"""The butterfly permutations of BOFT.

Factor :math:`i` of the butterfly product conjugates its block-diagonal matrix
:math:`B_i` by a fixed permutation :math:`P_i`, :math:`M_i = P_i B_i P_i^\top`,
which as an index map reads :math:`M_i[a, b] = B_i[\pi_i(a), \pi_i(b)]`.  The
permutations depend on sizes alone, so they are computed once per shape.
"""

import functools
import math

import lucid
from lucid._tensor.tensor import Tensor
from lucid._types import DeviceLike


class _Butterfly:
    """Block butterfly permutations; stateless."""

    @staticmethod
    def _interleaved(size: int, base: int) -> list[int]:
        """The order that lists a block's runs of ``base`` units evens-first, then odds."""
        step = size / base
        runs = [*range(0, math.ceil(step), 2), *range(1, math.ceil(step), 2)]
        return [unit for run in runs for unit in range(run * base, run * base + base)]

    @classmethod
    def permutation(cls, n: int, blocks: int, base: int, factors: int) -> list[int]:
        """The permutation of ``n`` units that groups them for ``blocks`` blocks, runs of ``base`` units interleaved.

        Parameters
        ----------
        n : int
            Number of units (the layer's input size).
        blocks : int
            Number of blocks this factor will have.
        base : int
            Run length, half the block size.
        factors : int
            The butterfly depth minus one; ``0`` is the identity (plain OFT).

        Raises
        ------
        ValueError
            ``blocks`` and ``base`` do not fit in ``n``.
        """
        if factors == 0:
            return list(range(n))
        if blocks * base * 2 > n:
            raise ValueError("Invalid number of blocks!")
        size = n // blocks
        order = cls._interleaved(size, base)
        indices = list(range(n))
        for start in range(0, n, size):
            chunk = indices[start : start + size]
            indices[start : start + size] = [chunk[j] for j in order]
        return indices


@functools.cache
def _permutation_rows(
    n: int, num_blocks: int, block_size: int, factors: int, device: DeviceLike
) -> Tensor:
    """The ``(factors + 1, n)`` index rows of all the factors' permutations, on ``device`` (built once per shape)."""
    rows = [
        _Butterfly.permutation(n, num_blocks // 2**i, block_size // 2, factors)
        for i in range(factors + 1)
    ]
    return lucid.tensor(rows, dtype=lucid.int64, device=device)
