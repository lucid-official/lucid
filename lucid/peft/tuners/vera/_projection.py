r"""The random projections VeRA shares: how they are drawn, and the container that holds them.

One pair :math:`(A, B)` serves every layer of an adapter, sized for the
largest layer and sliced for the others.  It is drawn from a generator
seeded with ``projection_prng_key`` on the CPU and then moved, so the same
key gives the same projections on every device — which is what lets an
adapter saved without them (``save_projection=False``) be restored.
"""

import math
from collections.abc import Iterator
from typing import Final, NamedTuple

import lucid
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid._types import DeviceLike, DTypeLike

#: The gain Kaiming initialisation uses for a ReLU network, as the reference does for the projections.
_GAIN: Final = math.sqrt(2.0)


def _take_entries(
    container: _BufferDict, state_dict: dict[str, Tensor], prefix: str, *rest: object
) -> None:
    """Load pre-hook: a container holds whatever entries it is given, so a fresh one makes a buffer per name in a checkpoint.

    As ``ModuleDict`` takes the keys of its children's: without this a layer's own checkpoint
    (which lists the projections it holds) would not load into a fresh instance of the layer.
    """
    del rest  # missing / unexpected keys and the rest of the hook's arguments: the default loader's
    if container.persistent:
        for key, value in state_dict.items():
            name = key[len(prefix) :]
            if key.startswith(prefix) and "." not in name and name not in container:
                container[name] = lucid.zeros_like(value)


class _BufferDict(nn.Module):
    """Buffers by adapter name: tensors a layer reads and does not train, as ``ParameterDict`` holds parameters.

    A buffer saved in ``state_dict`` as ``<name>`` (so ``vera_A.default``) unless the
    container is not ``persistent``, in which case it is never saved — the
    projections are then restored from the PRNG key.
    """

    def __init__(self, persistent: bool) -> None:
        super().__init__()
        self._persistent = persistent
        self.register_load_state_dict_pre_hook(_take_entries)

    @property
    def persistent(self) -> bool:
        """Whether the buffers are saved in ``state_dict``."""
        return self._persistent

    def __setitem__(self, name: str, tensor: Tensor) -> None:
        self.register_buffer(name, tensor, persistent=self._persistent)

    def __getitem__(self, name: str) -> Tensor:
        buffer = self._buffers.get(name)
        if buffer is None:
            raise KeyError(name)
        return buffer

    def __delitem__(self, name: str) -> None:
        if name not in self._buffers:
            raise KeyError(name)
        # ``delattr`` also drops the name from the persistence bookkeeping.
        delattr(self, name)

    def __contains__(self, name: object) -> bool:
        return name in self._buffers

    def __iter__(self) -> Iterator[str]:
        return iter(self._buffers)

    def __len__(self) -> int:
        return len(self._buffers)


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
