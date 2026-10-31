r"""The random projections VeRA shares: how they are drawn, and the container that holds them.

One pair :math:`(A, B)` serves every layer of an adapter, sized for the
largest layer and sliced for the others.  It is drawn from a generator
seeded with ``projection_prng_key`` on the CPU and then moved, so the same
key gives the same projections on every device — which is what lets an
adapter saved without them (``save_projection=False``) be restored.
"""

import math
from typing import Final, NamedTuple

import lucid
from lucid._tensor.tensor import Tensor
from lucid._types import DeviceLike, DTypeLike
from lucid.peft.tuners._shared import (
    _BufferDict as _BufferDict,
)  # also read from here by the layer and tuner

#: The gain Kaiming initialisation uses for a ReLU network, as the reference does for the projections.
_GAIN: Final = math.sqrt(2.0)


class _Shared:
    """The two containers a tuner shares between its layers, and whether a layer holds them in its tree.

    Exactly one layer *owns* them — registers them as its submodules, so they are in the
    module tree once (``.to()``, ``state_dict``, ``named_buffers`` see each tensor once) —
    and the other layers read them through a plain reference.
    """

    __slots__ = ("owned", "vera_a", "vera_b")

    def __init__(self, persistent: bool) -> None:
        self.vera_a = _BufferDict(persistent)
        self.vera_b = _BufferDict(persistent)
        self.owned = False


class _Projection(NamedTuple):
    """The shared pair: ``vera_a`` is ``(r, in)`` and ``vera_b`` is ``(out, r)``."""

    vera_a: Tensor
    vera_b: Tensor

    @staticmethod
    def draw(
        key: int, r: int, out_features: int, in_features: int, device: DeviceLike, dtype: DTypeLike
    ) -> _Projection:
        """The pair for a largest layer of ``(out_features, in_features)``, drawn from ``key``.

        ``A`` then ``B`` come from one generator, Kaiming-uniform with ``a = 0``
        and a gain of :math:`\\sqrt 2` (``bound = sqrt(3) * gain / sqrt(fan_in)``),
        so the draw is reproducible and independent of the global random state.
        """
        generator = lucid.Generator(key, device="cpu")
        pair = []
        for rows, columns in ((r, in_features), (out_features, r)):
            tensor = lucid.empty(rows, columns, device="cpu")
            bound = math.sqrt(3.0) * _GAIN / math.sqrt(columns)
            tensor.uniform_(-bound, bound, generator=generator)
            pair.append(tensor.to(device=device, dtype=dtype))
        return _Projection(pair[0], pair[1])
