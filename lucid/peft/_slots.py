"""What stands in for a base module, and how to find those stand-ins.

A tuner layer (``lora.Linear``) and a ``modules_to_save`` copy both sit in
the tree where a base module was, hold one set of parameters per adapter
and answer the same questions.  ``_AdapterSlot`` is that set of questions,
so the generic code — the tuner's switching, deleting, freezing and
unloading, the state-dict helpers — walks slots without knowing their
classes.
"""

from collections.abc import Iterator, Sequence
from typing import Protocol, runtime_checkable

import lucid.nn as nn
from lucid.nn.parameter import Parameter


def _as_names(adapter_names: str | Sequence[str]) -> list[str]:
    """One adapter name, or several, as a list."""
    return [adapter_names] if isinstance(adapter_names, str) else list(adapter_names)


@runtime_checkable
class _AdapterSlot(Protocol):
    """A module standing in for a base module, holding a set of parameters per adapter.

    What the tuner needs of each: switch it, list what trains, map its
    keys to the file's, drop an adapter from it, and put the base back.
    """

    def get_base_layer(self) -> nn.Module:
        """The module this one stands in for."""
        ...

    def enable_adapters(self, enabled: bool) -> None:
        """Take part (``True``) or let the base module run alone (``False``)."""
        ...

    def set_adapter(self, adapter_names: str | list[str], inference_mode: bool = False) -> None:
        """Make ``adapter_names`` the active ones and train exactly those, unless ``inference_mode``."""
        ...

    def delete_adapter(self, adapter_name: str) -> None:
        """Drop ``adapter_name``'s parameters; nothing happens if the slot has none."""
        ...

    def set_requires_grad(
        self, adapter_names: str | Sequence[str], requires_grad: bool = True
    ) -> None:
        """Set ``requires_grad`` on the parameters of the named adapters this slot holds."""
        ...

    def _activate(self, adapter_names: str | list[str]) -> None:
        """Make ``adapter_names`` the active ones, leaving every ``requires_grad`` as it is."""
        ...

    def _check_activation(self, adapter_names: list[str]) -> None:
        """Raise ``ValueError`` if ``adapter_names`` cannot all be active in this slot."""
        ...

    def _unwrap(self, keep_trained: bool) -> nn.Module:
        """What goes back in this slot's place: the trained result, or the base as it was."""
        ...

    def _adapter_names(self) -> list[str]:
        """The adapters this slot holds parameters for."""
        ...

    def _adapter_parameters(self, adapter_name: str) -> list[Parameter]:
        """The parameters that train for ``adapter_name``."""
        ...

    def _adapter_keys(self, adapter_name: str) -> dict[str, str]:
        """Own state-dict key → the adapter file's key, for ``adapter_name``'s tensors."""
        ...


def _adapter_slots(model: nn.Module) -> list[tuple[str, _AdapterSlot]]:
    return [(name, m) for name, m in model.named_modules() if isinstance(m, _AdapterSlot)]


def _outside_slots(model: nn.Module) -> Iterator[tuple[str, nn.Module]]:
    """``model.named_modules()`` without the insides of the adapter slots.

    A slot is yielded, its children (``q.base_layer``, ``q.lora_A.default``,
    a ``modules_to_save`` copy) are not: they are the adapter's own, and
    selecting a module for an adapter must not reach into one.
    """
    inside: str | None = None
    for name, module in model.named_modules():
        if inside is not None:
            if name.startswith(inside):
                continue
            inside = None
        yield name, module
        if name and isinstance(module, _AdapterSlot):
            inside = f"{name}."
