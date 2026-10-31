"""Putting other tensors in a module's parameter and buffer slots, for a while.

``functional_call`` runs a module with parameters and buffers that are not
the ones it holds.  :class:`TensorSwap` is that substitution as a context
manager: it works out, before it touches anything, which slots the given
names mean (and which others they are tied to), puts the tensors in, and on
the way out — normally or by an exception — puts back the very objects that
were there.

The slots are the module's own registries: the module's forward reads
``self.weight`` and finds what was swapped in, with its autograd history
intact.  A parameter slot is shown an alias of the given tensor (the same
engine tensor wrapped as a :class:`~lucid.nn.Parameter`, which is what the
registry declares); a buffer slot holds the tensor itself.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from types import TracebackType
from typing import Self, final, override

from lucid._tensor.tensor import Tensor
from lucid.nn.module import Module, _tensors_replaced
from lucid.nn.parameter import Parameter

__all__ = ["TensorSwap"]


class _Slot(ABC):
    """One parameter or buffer entry of one module: a registry and a key."""

    __slots__ = ("owner", "key")

    def __init__(self, owner: Module, key: str) -> None:
        self.owner = owner
        self.key = key

    @property
    def identity(self) -> tuple[type, int, str]:
        """What the slot is, whatever path it was reached by — a module
        used under two names has the same slots under both."""
        return type(self), id(self.owner), self.key

    @abstractmethod
    def held(self) -> Tensor | None:
        """The tensor in the slot now."""

    @abstractmethod
    def put(self, tensor: Tensor) -> Callable[[], None]:
        """Stand ``tensor`` in the slot; the call that puts back what was there."""


@final
class _ParameterSlot(_Slot):
    """An entry of ``_parameters``, which declares ``Parameter``s: the module
    is shown an alias of the tensor — the same engine tensor, so the same
    values and autograd history, never a copy."""

    __slots__ = ()

    @override
    def held(self) -> Tensor | None:
        return self.owner._parameters[self.key]

    @override
    def put(self, tensor: Tensor) -> Callable[[], None]:
        registry = self.owner._parameters
        original = registry[self.key]
        registry[self.key] = Parameter.__new_from_impl__(tensor._impl)

        def restore() -> None:
            registry[self.key] = original

        return restore


@final
class _BufferSlot(_Slot):
    """An entry of ``_buffers``: the tensor itself stands there."""

    __slots__ = ()

    @override
    def held(self) -> Tensor | None:
        return self.owner._buffers[self.key]

    @override
    def put(self, tensor: Tensor) -> Callable[[], None]:
        registry = self.owner._buffers
        original = registry[self.key]
        registry[self.key] = tensor

        def restore() -> None:
            registry[self.key] = original

        return restore


def _slots(module: Module) -> dict[str, _Slot]:
    """Every parameter and buffer slot under ``module`` by its dotted name —
    a tensor shared by two names has a slot under each."""
    found: dict[str, _Slot] = {}
    for path, owner in module.named_modules(remove_duplicate=False):
        for registry, slot in ((owner._parameters, _ParameterSlot), (owner._buffers, _BufferSlot)):
            for key, tensor in registry.items():
                if tensor is not None:
                    found[f"{path}.{key}" if path else key] = slot(owner, key)
    return found


def _tied(slots: Mapping[str, _Slot]) -> list[list[str]]:
    """The groups of names that hold one and the same tensor."""
    by_object: dict[int, list[str]] = {}
    for name, slot in slots.items():
        held = slot.held()
        assert held is not None  # ``_slots`` lists occupied slots only
        by_object.setdefault(id(held), []).append(name)
    return [names for names in by_object.values() if len(names) > 1]


def _keys(names: list[str]) -> str:
    return ", ".join(f"'{name}'" for name in names)


@final
class TensorSwap:
    """Context manager: ``tensors`` stand in the slots of ``module`` named by
    their keys, and the original objects are put back on exit.

    Parameters
    ----------
    module : Module
        The module whose parameters and buffers are replaced.
    tensors : mapping of str to Tensor
        Dotted parameter / buffer name → the tensor to stand in its place.
        A name that is not a parameter or buffer of ``module`` is ignored
        unless ``strict``.
    tie_weights : bool, optional
        Whether a tensor given for one name also stands for every name that
        holds the same object (tied weights).  Two *different* tensors for
        names that are tied is a ``ValueError``.  Default ``True``.
    strict : bool, optional
        Require ``tensors`` to name exactly the parameters and buffers of
        ``module`` (tied names count as named when ``tie_weights``).
        Default ``False``.

    Raises
    ------
    ValueError
        Different tensors were given for tied names.
    RuntimeError
        ``strict`` and the names do not match the module's.
    """

    __slots__ = ("_plan", "_undo")

    def __init__(
        self,
        module: Module,
        tensors: Mapping[str, Tensor],
        *,
        tie_weights: bool = True,
        strict: bool = False,
    ) -> None:
        slots = _slots(module)
        plan = {name: tensor for name, tensor in tensors.items() if name in slots}
        if tie_weights:
            self._share_tied(plan, slots)
        if strict:
            self._require_exact(module, plan, [n for n in tensors if n not in slots], slots)
        self._plan: list[tuple[_Slot, Tensor]] = self._by_slot(plan, slots)
        self._undo: list[Callable[[], None]] = []

    @staticmethod
    def _share_tied(plan: dict[str, Tensor], slots: Mapping[str, _Slot]) -> None:
        """Extend ``plan`` to every name tied to a name in it."""
        for group in _tied(slots):
            given = [name for name in group if name in plan]
            if len({id(plan[name]) for name in given}) > 1:
                raise ValueError(
                    f"functional_call got multiple values for keys {sorted(given)}, "
                    "which are tied. Consider using tie_weights=False"
                )
            if given:
                for name in group:
                    plan[name] = plan[given[0]]

    @staticmethod
    def _require_exact(
        module: Module,
        plan: Mapping[str, Tensor],
        unexpected: list[str],
        slots: Mapping[str, _Slot],
    ) -> None:
        missing = [name for name in slots if name not in plan]
        problems = [
            f"{label} key(s): {_keys(names)}."
            for label, names in (("Unexpected", unexpected), ("Missing", missing))
            if names
        ]
        if problems:
            raise RuntimeError(
                f"Error(s) in reparametrizing for {type(module).__name__}:\n\t"
                + "\n\t".join(problems)
            )

    @staticmethod
    def _by_slot(
        plan: Mapping[str, Tensor], slots: Mapping[str, _Slot]
    ) -> list[tuple[_Slot, Tensor]]:
        """``plan`` once per slot: a module used under two names has one."""
        by_identity = {slots[name].identity: (slots[name], tensor) for name, tensor in plan.items()}
        return list(by_identity.values())

    def __enter__(self) -> Self:
        try:
            for slot, tensor in self._plan:
                self._undo.append(slot.put(tensor))
        except BaseException:
            self._restore()  # a half-swapped module is worse than either state
            raise
        self._announce()
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        self._restore()

    def _restore(self) -> None:
        while self._undo:
            self._undo.pop()()
        self._announce()

    def _announce(self) -> None:
        """Tell compiled wrappers their module's tensors changed."""
        for owner in {id(slot.owner): slot.owner for slot, _ in self._plan}.values():
            _tensors_replaced(owner)
