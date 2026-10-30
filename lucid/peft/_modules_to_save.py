"""``modules_to_save``: a module trained in full beside the adapter.

The original stays in the tree, frozen, and a trainable deep copy is
what runs — so the base model is still there to fall back on (under
``disable_adapter()``, after ``unload()``) and the trained copy is what
the adapter file stores, under the original module's own key names.
"""

import copy
from typing import override

import lucid.nn as nn
from lucid.peft._forwarding import _ForwardsAttributes


class ModulesToSaveWrapper(_ForwardsAttributes):
    """Hold ``module`` beside a copy that runs in its place.

    Building one changes nothing about ``module`` — ``LoraModel`` builds
    them while it is still planning, and a plan that fails must leave the
    model as it was.  Which of the two trains is ``LoraModel``'s freezing
    step: the copy, not the original.

    Parameters
    ----------
    module_to_save : nn.Module
        The module trained in full (a classification head, an embedding).
    adapter_name : str
        Key of the copy in ``modules_to_save``.
    """

    def __init__(self, module_to_save: nn.Module, adapter_name: str) -> None:
        super().__init__()
        self.original_module = module_to_save
        trained = copy.deepcopy(module_to_save)
        self.modules_to_save: nn.ModuleDict[nn.Module] = nn.ModuleDict({adapter_name: trained})
        self._active_adapter = adapter_name
        self._disable_adapters = False

    @property
    def active_adapter(self) -> str:
        """Name of the copy that runs."""
        return self._active_adapter

    @property
    def disable_adapters(self) -> bool:
        """Whether the original module runs instead of the copy."""
        return self._disable_adapters

    def enable_adapters(self, enabled: bool) -> None:
        """Run the trained copy (``True``) or the original module (``False``)."""
        self._disable_adapters = not enabled

    def _active_module(self) -> nn.Module:
        if self._disable_adapters:
            return self.original_module
        return self.modules_to_save[self._active_adapter]

    @override
    def _wrapped(self) -> nn.Module | None:
        # ``_disable_adapters`` is the last thing ``__init__`` sets.
        return self._active_module() if "_disable_adapters" in self.__dict__ else None

    @override
    def forward(self, *args: object, **kwargs: object) -> object:
        return self._active_module()(*args, **kwargs)
