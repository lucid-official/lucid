r"""Where FourierFT's coefficients sit in the spectrum, as the reference framework's CPU generator picks them.

An adapter file holds the coefficients only; the ``n`` locations in the
:math:`d_{out} \times d_{in}` spectrum are not stored, they are the first ``n``
entries of a random permutation of ``d_{out} d_{in}`` positions seeded by
``random_loc_seed``.  A file moves between the two libraries only if both
pick the same positions, so the permutation is reproduced bit for bit: the
generator is the 32-bit Mersenne Twister the reference framework seeds with
``init_genrand``, and the permutation is its CPU algorithm, a Fisher-Yates
pass drawing ``r = random() mod (m - i)``.  Only the first ``n`` steps are run
(later steps cannot change the entries already placed), over a sparse overlay
instead of the whole array, so ``n = 1000`` costs a thousand draws whatever the layer's size.
"""

import functools
import random
from typing import Final

import lucid
from lucid._tensor.tensor import Tensor
from lucid._types import DeviceLike

_STATE_SIZE: Final = 624
_INIT_MULTIPLIER: Final = 1812433253
_MASK32: Final = 0xFFFFFFFF
#: ``random.Random.setstate`` version of the Mersenne Twister state tuple.
_STATE_VERSION: Final = 3
#: Above this many positions the reference framework draws 64-bit words; that path is not reproduced.
_MAX_POSITIONS: Final = (2**32 - 1) // 20


def _initial_state(seed: int) -> tuple[int, ...]:
    """The 624-word Mersenne Twister state ``init_genrand(seed)`` makes, then the index that triggers the first refill."""
    words = [seed & _MASK32]
    for position in range(1, _STATE_SIZE):
        previous = words[-1]
        words.append((_INIT_MULTIPLIER * (previous ^ (previous >> 30)) + position) & _MASK32)
    return (*words, _STATE_SIZE)


@functools.cache
def _spectrum_positions(size: int, count: int, seed: int) -> tuple[int, ...]:
    """The first ``count`` entries of the seeded random permutation of ``range(size)``.

    Raises
    ------
    NotImplementedError
        ``size`` is so large that the reference framework switches to 64-bit draws.
    """
    if size >= _MAX_POSITIONS:
        raise NotImplementedError(
            f"FourierFT locations over {size} spectrum positions need 64-bit draws, "
            f"which are reproduced for fewer than {_MAX_POSITIONS} positions only"
        )
    generator = random.Random()
    generator.setstate((_STATE_VERSION, _initial_state(seed), None))
    moved: dict[int, int] = {}  # positions the shuffle has overwritten so far; the rest hold their own index
    picked: list[int] = []
    for i in range(count):
        j = i + generator.getrandbits(32) % (size - i)
        picked.append(moved.get(j, j))
        moved[j] = moved.get(i, i)
    return tuple(picked)


@functools.cache
def _entry_indices(rows: int, columns: int, count: int, seed: int, device: DeviceLike) -> Tensor:
    """The ``(2, count)`` ``(row, column)`` indices of the coefficients in a ``rows x columns`` spectrum, on ``device``."""
    flat = _spectrum_positions(rows * columns, count, seed)
    return lucid.tensor(
        [[p // columns for p in flat], [p % columns for p in flat]],
        dtype=lucid.int64,
        device=device,
    )
