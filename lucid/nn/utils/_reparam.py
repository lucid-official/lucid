"""The one owner of the forward-pre-hooks the legacy reparametrisations install.

``weight_norm``, ``spectral_norm`` and the pruning methods each replace a
parameter by a tensor recomputed before every forward.  They do it with a
forward pre-hook that *is* the method object — a plain class instance, so a
module holding one pickles and deep-copies, which a closure over the
registration arguments did not.  The three families share what is not about
the maths: finding the hook that serves a parameter name, installing it,
removing it, and telling the compile tracer which pre-hooks are theirs.  That
lives here, once.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import ClassVar, Self, override

from lucid._tensor.tensor import Tensor
from lucid.nn.module import Module


class _ReparamHook(ABC):
    """Base of every pre-hook that derives ``module.<target>`` from other tensors.

    A subclass names the parameter it serves (:attr:`target`), recomputes it in
    ``__call__`` and undoes itself in :meth:`remove`.  Installing, finding and
    uninstalling are shared.
    """

    #: What the reparametrisation is called in an error message.
    _LABEL: ClassVar[str]

    @property
    @abstractmethod
    def target(self) -> str:
        """Name of the module attribute this hook recomputes."""

    @abstractmethod
    def refresh(self, module: Module) -> None:
        """Recompute ``module.<target>`` from the tensors it is derived from."""

    @abstractmethod
    def remove(self, module: Module) -> None:
        """Bake the current value into ``module`` and drop the helper state."""

    def __call__(self, module: object, inputs: tuple[Tensor, ...]) -> None:
        """The forward pre-hook: refresh the target before ``module`` runs."""
        del inputs  # the derivation reads the module's tensors, never the call's
        if not isinstance(module, Module):
            raise TypeError(f"{type(self).__name__} hooks a Module, got {type(module).__name__}")
        self.refresh(module)

    def _publish(self, module: Module, value: Tensor) -> None:
        """Set the derived tensor ``module.<target>`` without touching any registry.

        A plain attribute write: the derived tensor is recomputed on every forward
        and must not move the module's tree version each time, which would
        invalidate every compiled call that reads the tree.
        """
        object.__setattr__(module, self.target, value)

    def _vacate(self, module: Module) -> None:
        """Drop the derived tensor cached as ``module.<target>`` so a Parameter may take the name."""
        module.__dict__.pop(self.target, None)

    @staticmethod
    def installed(module: Module) -> Iterator[tuple[int, _ReparamHook]]:
        """``(hook id, hook)`` for every reparametrisation hook on ``module``, in call order.

        The compile tracer reads the ids to leave these hooks out of the hook
        contract — it traces what they compute in-graph instead.
        """
        for key, hook in list(module._forward_pre_hooks.items()):
            if isinstance(hook, _ReparamHook):
                yield key, hook

    @classmethod
    def _entry(cls, module: Module, name: str) -> tuple[int, Self] | None:
        """The ``(id, hook)`` of this class serving ``module.<name>``, or ``None``."""
        for key, hook in cls.installed(module):
            if isinstance(hook, cls) and hook.target == name:
                return key, hook
        return None

    @classmethod
    def find(cls, module: Module, name: str) -> Self | None:
        """The hook of this class serving ``module.<name>``, or ``None``."""
        entry = cls._entry(module, name)
        return None if entry is None else entry[1]

    @staticmethod
    def _register(module: Module, hook: _ReparamHook) -> None:
        """Put ``hook`` on the module's forward pre-hooks (the hook stays reachable there)."""
        module.register_forward_pre_hook(hook)

    @classmethod
    def uninstall(cls, module: Module, name: str) -> Module:
        """Remove the hook of this class serving ``module.<name>``.

        The hook bakes its value into the module (:meth:`remove`) and leaves
        the pre-hook registry, in that order, so a failure leaves it installed.

        Raises
        ------
        ValueError
            If no such hook is installed.
        """
        entry = cls._entry(module, name)
        if entry is None:
            raise ValueError(f"{cls._LABEL} of '{name}' not found in {type(module).__name__}")
        key, hook = entry
        hook.remove(module)
        del module._forward_pre_hooks[key]
        return module


class _NamedHook(_ReparamHook):
    """A reparametrisation of one parameter the caller names: ``weight_norm``, ``spectral_norm``.

    Pruning keeps the reference's own ``_tensor_name`` and does not use this.
    """

    def __init__(self, name: str) -> None:
        self.name: str = name

    @property
    @override
    def target(self) -> str:
        """Name of the reparametrised parameter."""
        return self.name

    @classmethod
    def _require_free(cls, module: Module, name: str) -> None:
        """A parameter takes one hook of a kind: refuse a second.

        Raises
        ------
        RuntimeError
            If ``name`` already carries a hook of this class.
        """
        if cls.find(module, name) is not None:
            raise RuntimeError(
                f"Cannot register two {cls._LABEL} hooks on the same parameter {name}"
            )
