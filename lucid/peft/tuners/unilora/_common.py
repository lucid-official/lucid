"""What the shared-vector methods add to ``tuners._shared``: shared tables that train, and the bias refusal.

Uni-LoRA and TinyLoRA read every adapted layer's update out of a vector (or a group of vectors) the
tuner holds once for the whole model.  That vector is the adapter's only trainable tensor, so it
trains and freezes with the adapter of each layer that reads it, though no layer owns it.  VB-LoRA's
bank is the same kind of table; its layer carries a copy of the two overrides below.
"""

from collections.abc import Sequence
from typing import override

import lucid.nn as nn
from lucid.nn.parameter import Parameter
from lucid.peft._slots import _as_names
from lucid.peft.tuners._layer import _entry_parameters
from lucid.peft.tuners._shared import _SharingLayer


def _refuse_bias(method: str, bias: str) -> None:
    """Raise ``NotImplementedError`` unless ``bias`` is ``"none"``: these adapters train their own tensors only."""
    if bias != "none":
        raise NotImplementedError(
            f"bias={bias!r}: a {method} adapter in lucid.peft trains its own tensors only "
            "(bias='none')"
        )


class _TrainedSharedLayer(_SharingLayer):
    """A sharing layer whose shared tables hold the adapter's trainable parameters.

    ``_shared_names`` names the tables; each maps an adapter name to a ``Parameter`` or to a module of
    them.  A layer that holds an adapter reports the parameters its shared tables hold for it, and
    switches their ``requires_grad`` with the adapter's.
    """

    def _shared_parameters(self, adapter_name: str) -> list[Parameter]:
        """The parameters the shared tables hold for ``adapter_name``."""
        found: list[Parameter] = []
        for name in self._shared_names:
            table = self._table(name)
            if isinstance(table, (nn.ParameterDict, nn.ModuleDict)) and adapter_name in table:
                found.extend(_entry_parameters(table[adapter_name]))
        return found

    @override
    def _adapter_parameters(self, adapter_name: str) -> list[Parameter]:
        own = super()._adapter_parameters(adapter_name)
        if adapter_name not in self._adapter_names():
            return own
        return [*own, *self._shared_parameters(adapter_name)]

    @override
    def set_requires_grad(
        self, adapter_names: str | Sequence[str], requires_grad: bool = True
    ) -> None:
        """Set ``requires_grad`` on the named adapters' own tensors and on the shared parameters they read."""
        super().set_requires_grad(adapter_names, requires_grad)
        held = self._adapter_names()
        for name in _as_names(adapter_names):
            if name in held:
                for parameter in self._shared_parameters(name):
                    parameter.requires_grad_(requires_grad)
