"""The projections FRoD's layers share, held the way the reference holds them: per category, per adapter.

The reference keeps three tables on the tuner, ``frod_V``, ``frod_s_indices`` and ``frod_s_size``, each a
container per category with one tensor per adapter.  A layer of that category reads its adapter's entry.
Here the tables are held in the module tree by one layer (the first bound, as VeRA's are) so that ``.to()``,
``state_dict`` and an adapter injected into a bare model all see them once; the others read them through
this object.  The adapter file holds each table once per category, as ``base_model.<table>.<category>``.
"""

from typing import Final

import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.peft.tuners._shared import _BufferDict, _Shared

#: The tables, as the reference names them and as the file's keys end them.
NAMES: Final = ("frod_V", "frod_s_indices", "frod_s_size")


class _Projections:
    """The three tables, and how to put an adapter's entry in them, read it and take it out."""

    def __init__(self) -> None:
        self.shared = _Shared({name: nn.ModuleDict() for name in NAMES})

    def _containers(self, name: str) -> nn.ModuleDict[nn.Module]:
        table = self.shared.tables[name]
        assert isinstance(table, nn.ModuleDict)
        return table

    def _buffers(self, name: str, category: str) -> _BufferDict:
        container = self._containers(name)[category]
        assert isinstance(container, _BufferDict)
        return container

    def categories(self) -> list[str]:
        """The categories that hold an entry for some adapter."""
        return list(self._containers(NAMES[0]))

    def has(self, category: str, adapter: str) -> bool:
        """Whether ``adapter`` has an entry in ``category``."""
        return category in self._containers(NAMES[0]) and adapter in self._buffers(
            NAMES[0], category
        )

    def persistent(self, category: str) -> bool:
        """Whether the tables write ``category`` into ``state_dict`` (they do if the adapter saves its projections)."""
        return self._buffers(NAMES[0], category).persistent

    def put(
        self, category: str, adapter: str, entry: tuple[Tensor, Tensor, Tensor], persistent: bool
    ) -> None:
        """Set ``adapter``'s ``(V, indices, size)`` in ``category``; a category's containers are made with the first."""
        for name, tensor in zip(NAMES, entry, strict=True):
            containers = self._containers(name)
            if category not in containers:
                containers[category] = _BufferDict(persistent=persistent)
            self._buffers(name, category)[adapter] = tensor

    def entry(self, category: str, adapter: str) -> tuple[Tensor, Tensor, Tensor]:
        """``adapter``'s ``(V, indices, size)`` in ``category``."""
        v, indices, size = (self._buffers(name, category)[adapter] for name in NAMES)
        return v, indices, size

    def drop(self, adapter: str) -> None:
        """Take every entry of ``adapter`` out of every category, and the categories that hold none after."""
        for name in NAMES:
            containers = self._containers(name)
            for category in list(containers):
                buffers = self._buffers(name, category)
                if adapter in buffers:
                    del buffers[adapter]
                if not buffers:
                    del containers[category]
