"""One version number for the state of every module tree.

``lucid.compile`` keys an executable on a walk of the model's tree — each
submodule's ``training`` flag, public scalar attributes and hooks.  The walk
cost about a millisecond per call on a 200-module model, every call, so the
compile side caches it against :func:`_tree_version`.

The version moves after any change the walk can see, anywhere: an attribute
set or deleted through :meth:`Module.__setattr__` / ``__delattr__``, and
every write to a module's parameter, buffer, submodule or hook registry.
Those registries are :class:`_TrackedDict`\\ s because they are written
directly — by ``register_parameter`` / ``register_buffer`` / ``add_module``,
an ``assign=True`` load, the containers, the reparametrisations,
quantization and :meth:`~lucid.nn.hooks.RemovableHandle.remove` — rather
than through ``__setattr__``.  ``lucid.compile`` reads the tensor registries
too: an executable is fed the tensors it was traced with, so one put in
their place has to be seen.  A write that bypasses both
(``module.__dict__[name] = v``) is not seen, nor is a tensor given new
storage under the same object (``p.data = v``).
"""

import itertools
from collections import OrderedDict
from typing import overload, override

_versions = itertools.count(1)
_version = 0
_NO_DEFAULT = object()


def _tree_version() -> int:
    """The current version; it differs from every earlier one after a change."""
    return _version


def _tree_changed() -> None:
    """Move the version.  Call it after the change, never before: a reader
    that walks in between then caches the old version, not the new one."""
    global _version
    _version = next(_versions)


class _TrackedDict[K, V](OrderedDict[K, V]):
    """An ``OrderedDict`` that moves the tree version on every write.

    ``pop``, ``popitem``, ``clear`` and ``move_to_end`` do not go through
    ``__setitem__`` / ``__delitem__`` on a subclass, so each is wrapped too;
    ``update``, ``setdefault`` and ``|=`` do.
    """

    @override
    def __setitem__(self, key: K, value: V) -> None:
        super().__setitem__(key, value)
        _tree_changed()

    @override
    def __delitem__(self, key: K) -> None:
        super().__delitem__(key)
        _tree_changed()

    @overload
    def pop(self, key: K) -> V: ...
    @overload
    def pop(self, key: K, default: V) -> V: ...
    @overload
    def pop[T](self, key: K, default: T) -> V | T: ...
    @override
    def pop(self, key: K, default: object = _NO_DEFAULT) -> object:
        out = super().pop(key) if default is _NO_DEFAULT else super().pop(key, default)
        _tree_changed()
        return out

    @override
    def popitem(self, last: bool = True) -> tuple[K, V]:
        item = super().popitem(last)
        _tree_changed()
        return item

    @override
    def clear(self) -> None:
        super().clear()
        _tree_changed()

    @override
    def move_to_end(self, key: K, last: bool = True) -> None:
        super().move_to_end(key, last)
        _tree_changed()

    @override
    def __reduce__(self) -> tuple[type[OrderedDict[K, V]], tuple[list[tuple[K, V]]]]:
        # Pickled and copied as a plain ``OrderedDict``, so a checkpoint never
        # names this private type; ``Module.__setstate__`` tracks it again.
        return (OrderedDict, (list(self.items()),))
