r"""What the low-rank delta layers share: where an adapter's tensors are allocated and a dropout of the input per adapter.

DeLoRA, GraLoRA, HiRA, GLoRA, DEFT and PEANuT each put a learned update beside a frozen base
module and differ in the formula of that update.  They agree on three smaller things: an
adapter's tensors are allocated where the base weight is, in the dtype ``_adapter_dtype`` gives
it; most methods drop out their input before the update reads it, one dropout per adapter name;
and a layer built from the module, a name and the method's config alone (``_ConfigLayer``) is how
every one of them is created.  This module holds those, so each method's ``_layer.py`` is its
formula.
"""

from typing import ClassVar, NamedTuple, override

import lucid
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid._types import DeviceLike, DTypeLike
from lucid.peft._config import _TargetedConfig
from lucid.peft.tuners._support import _adapter_dtype, _ConfigLayer


class _Home(NamedTuple):
    """Where and in what dtype an adapter's tensors are allocated: the base weight's device, its dtype widened to float32 if half."""

    device: DeviceLike
    dtype: DTypeLike

    def zeros(self, *shape: int) -> Tensor:
        """A zero tensor of ``shape`` here."""
        return lucid.zeros(*shape, device=self.device, dtype=self.dtype)

    def full(self, shape: tuple[int, ...], value: float) -> Tensor:
        """A tensor of ``shape`` filled with ``value`` here."""
        return lucid.full(shape, value, device=self.device, dtype=self.dtype)

    def linear(self, n_in: int, n_out: int) -> nn.Linear:
        """A bias-free ``nn.Linear`` here."""
        return nn.Linear(n_in, n_out, bias=False, device=self.device, dtype=self.dtype)


class _DeltaLayer[C: _TargetedConfig = _TargetedConfig](_ConfigLayer[C]):
    """A layer of a low-rank delta method: its tensors are allocated by ``_home`` and it may hold a dropout of its input per adapter.

    A subclass names the attribute its dropouts are kept in (``_dropout_table``, ``"hira_dropout"``)
    or leaves it empty, enters one with ``_enter_dropout`` while it builds an adapter and applies it
    with ``_dropped``.  Deleting an adapter takes its dropout with it.

    Parameters
    ----------
    base_layer, adapter_name, config
        As in ``_ConfigLayer``.
    """

    #: The attribute the per-adapter dropouts are kept in (a ``ModuleDict``), or ``""`` for none.
    _dropout_table: ClassVar[str] = ""

    def _home(self) -> _Home:
        """Where the adapter's tensors go: beside the base weight, in float32 if that is half precision."""
        anchor = self.weight
        return _Home(anchor.device, _adapter_dtype(anchor))

    @override
    def _prepare(self, config: C) -> None:
        super()._prepare(config)
        if self._dropout_table:
            setattr(self, self._dropout_table, nn.ModuleDict())

    @property
    def _dropouts(self) -> nn.ModuleDict[nn.Module]:
        table = getattr(self, self._dropout_table)
        assert isinstance(table, nn.ModuleDict)
        return table

    def _enter_dropout(self, adapter_name: str, probability: float) -> None:
        """Keep a dropout of ``probability`` for ``adapter_name`` (the identity when it is zero)."""
        self._dropouts[adapter_name] = nn.Dropout(p=probability) if probability > 0.0 else nn.Identity()

    def _dropped(self, adapter: str, x: Tensor) -> Tensor:
        """``x`` through ``adapter``'s dropout."""
        dropped: Tensor = self._dropouts[adapter](x)
        return dropped

    @override
    def delete_adapter(self, adapter_name: str) -> None:
        """Remove ``adapter_name``: its tensors, its recorded settings and its dropout."""
        super().delete_adapter(adapter_name)
        if self._dropout_table and adapter_name in self._dropouts:
            del self._dropouts[adapter_name]
