"""``modules_to_save``: a module trained in full beside the adapter.

The original stays in the tree, frozen, and a trainable deep copy — one
per adapter that asks for this module — is what runs, so the base model is
still there to fall back on (under ``disable_adapter()``, after
``unload()``, while an adapter that did not ask for the module is active)
and the trained copy is what the adapter file stores, under the original
module's own key names.
"""

import copy
from collections.abc import Sequence
from typing import override

import lucid.nn as nn
from lucid.nn.parameter import Parameter
from lucid.peft._forwarding import _ForwardsAttributes
from lucid.peft._slots import _as_names, _SlotState


class ModulesToSaveWrapper(_ForwardsAttributes):
    """Hold ``module`` beside a trained copy per adapter; the active adapter's copy runs.

    Building one changes nothing about ``module`` — the tuner builds them
    while it is still planning, and a plan that fails must leave the model
    as it was.  Which of the copies train is the tuner's freezing step.  It
    answers the tuner's ``_AdapterSlot`` protocol, as a tuner layer does.

    At most one adapter is active here: two copies of one module would have
    no rule for whose weights the forward pass uses.  ``active_adapter`` is
    the active adapter that has a copy, or ``""`` when none does, in which
    case the original runs.

    Parameters
    ----------
    module_to_save : nn.Module
        The module trained in full (a classification head, an embedding).
    adapter_name : str
        Key of the first copy in ``modules_to_save``.
    """

    def __init__(self, module_to_save: nn.Module, adapter_name: str) -> None:
        super().__init__()
        self.original_module = module_to_save
        self.modules_to_save: nn.ModuleDict[nn.Module] = nn.ModuleDict()
        self.update(adapter_name)
        # Public scalar attributes, which ``lucid.compile`` keys on (LCD-703):
        # name of the copy that runs, and whether the original runs instead.
        self.active_adapter = adapter_name
        self.disable_adapters = False

    def update(self, adapter_name: str) -> None:
        """Give ``adapter_name`` a trained copy of the original, if it has none yet."""
        if adapter_name not in self.modules_to_save:
            self.modules_to_save[adapter_name] = copy.deepcopy(self.original_module)

    def get_base_layer(self) -> nn.Module:
        """Return the original module, the one the base model had."""
        return self.original_module

    def enable_adapters(self, enabled: bool) -> None:
        """Run the trained copy (``True``) or the original module (``False``)."""
        self.disable_adapters = not enabled

    @override
    def extra_repr(self) -> str:
        return f"active_adapter={self.active_adapter!r}, disabled={self.disable_adapters}"

    # ── the tuner's ``_AdapterSlot`` protocol ──────────────────────────────

    def _held(self, adapter_names: str | Sequence[str]) -> list[str]:
        return [name for name in _as_names(adapter_names) if name in self.modules_to_save]

    def _check_activation(self, adapter_names: list[str]) -> None:
        held = self._held(adapter_names)
        if len(held) > 1:
            raise ValueError(
                f"Only one adapter can be set at a time for modules_to_save, got {len(held)}: {held}"
            )

    def _activate(self, adapter_names: str | list[str]) -> None:
        self._check_activation(_as_names(adapter_names))
        held = self._held(adapter_names)
        self.active_adapter = held[0] if held else ""

    def set_adapter(self, adapter_names: str | list[str], inference_mode: bool = False) -> None:
        """Run the named adapter's copy, and train it unless ``inference_mode``.

        The names this module has no copy for are skipped: it runs the
        original while only such adapters are active.

        Parameters
        ----------
        adapter_names : str or list of str
            The adapter, or adapters, to apply.
        inference_mode : bool, default=False
            Freeze the copy that runs as well.

        Raises
        ------
        ValueError
            More than one of the names has a copy here.
        """
        self._activate(adapter_names)
        active = self.active_adapter
        for held in self._adapter_names():
            self.set_requires_grad(held, held == active and not inference_mode)

    def set_requires_grad(
        self, adapter_names: str | Sequence[str], requires_grad: bool = True
    ) -> None:
        """Set ``requires_grad`` on the copies of the named adapters."""
        for name in self._held(adapter_names):
            self.modules_to_save[name].requires_grad_(requires_grad)

    def delete_adapter(self, adapter_name: str) -> None:
        """Drop ``adapter_name``'s copy; if it was the active one the original runs."""
        if adapter_name in self.modules_to_save:
            del self.modules_to_save[adapter_name]
        if self.active_adapter == adapter_name:
            self.active_adapter = ""

    def _unwrap(self, keep_trained: bool) -> nn.Module:
        if keep_trained and self.active_adapter in self.modules_to_save:
            return self.modules_to_save[self.active_adapter]
        return self.original_module

    def _adapter_names(self) -> list[str]:
        return list(self.modules_to_save)

    def _state(self) -> _SlotState:
        return _SlotState(
            not self.disable_adapters, [self.active_adapter] if self.active_adapter else [], []
        )

    def _adapter_parameters(self, adapter_name: str) -> list[Parameter]:
        if adapter_name not in self.modules_to_save:
            return []
        return list(self.modules_to_save[adapter_name].parameters())

    def _adapter_keys(self, adapter_name: str) -> dict[str, str]:
        """The copy's keys, filed under the original module's own names."""
        if adapter_name not in self.modules_to_save:
            return {}
        return {
            f"modules_to_save.{adapter_name}.{leaf}": leaf
            for leaf in self.modules_to_save[adapter_name].state_dict()
        }

    def _active_module(self) -> nn.Module:
        if self.disable_adapters or self.active_adapter not in self.modules_to_save:
            return self.original_module
        return self.modules_to_save[self.active_adapter]

    @override
    def _wrapped(self) -> nn.Module | None:
        # ``disable_adapters`` is the last thing ``__init__`` sets.
        return self._active_module() if "disable_adapters" in self.__dict__ else None

    @override
    def forward(self, *args: object, **kwargs: object) -> object:
        return self._active_module()(*args, **kwargs)
