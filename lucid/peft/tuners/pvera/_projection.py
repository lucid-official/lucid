r"""The random projections PVeRA shares: their shapes and how they are drawn.

One pair :math:`(A, B)` serves every layer of an adapter, sized for the largest layer.  :math:`A` is
``(2r, d_in)``: its ``2r`` rows give the mean and the log-variance of the rank-``r`` latent
(Fillioux et al., 2025, Sec. 3.2).  :math:`B` is ``(d_out, r)``.  A layer slices them to its size.
Both are drawn on the CPU from a generator seeded with ``projection_prng_key`` and then moved, so
the same key gives the same projections on every device — which is what lets an adapter saved
without them (``save_projection=False``) be restored — Kaiming-uniform with a gain of
:math:`\sqrt 2`, :math:`A` first.
"""

import math
from typing import Final, NamedTuple

import lucid
from lucid._tensor.tensor import Tensor
from lucid._types import DeviceLike, DTypeLike

#: The gain Kaiming initialisation uses for a ReLU network.
_GAIN: Final = math.sqrt(2.0)


class _Projection(NamedTuple):
    """The shared pair: ``pvera_A`` is ``(2r, in)`` and ``pvera_B`` is ``(out, r)``."""

    pvera_A: Tensor
    pvera_B: Tensor

    @staticmethod
    def draw(
        key: int, r: int, out_features: int, in_features: int, device: DeviceLike, dtype: DTypeLike
    ) -> _Projection:
        """The pair for a largest layer of ``(out_features, in_features)`` and rank ``r``, drawn from ``key``."""
        generator = lucid.Generator(key, device="cpu")
        pair = []
        for rows, columns in ((2 * r, in_features), (out_features, r)):
            tensor = lucid.empty(rows, columns, device="cpu")
            bound = math.sqrt(3.0) * _GAIN / math.sqrt(columns)
            tensor.uniform_(-bound, bound, generator=generator)
            pair.append(tensor.to(device=device, dtype=dtype))
        return _Projection(pair[0], pair[1])
