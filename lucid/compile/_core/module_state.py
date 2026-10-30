"""
lucid.compile._core.module_state — the module state a compiled call is keyed on.

A forward branches in Python on what its module holds, and a trace bakes the
branch taken into the graph.  :class:`StateContract` says which of a module's
attributes are such state and how each enters the key
(:attr:`~lucid.compile._core.signature.CacheKey.module_state`); every entry
point keys on it through the one key builder,
:meth:`~lucid.compile._core.signature.ParamFingerprint.key_of`.
"""

import struct
from collections.abc import Iterable, Sequence
from enum import Enum
from functools import cache
from typing import ClassVar, final

from lucid._dtype import dtype as _Dtype

__all__ = ["HeldState", "StateContract", "atom_sig"]

# A mutable container a module of the tree holds: the module's position in the
# walk, the attribute's name, and the container itself.
type HeldState = tuple[int, str, object]


def atom_sig(value: object) -> object:
    """Key a non-tensor value by its type and, for a float, its bits.

    The trace bakes the value in, so two calls share an executable only
    when they would trace the same graph.  Plain ``==`` is too coarse:
    ``1 == 1.0 == True`` although each promotes an integer tensor
    differently, ``0.0 == -0.0`` although they add differently, and a NaN
    never equals another NaN, so it recompiled on every call (LCD-446).
    Anything unhashable still raises on hash, which sends the call eager.
    """
    kind = f"{type(value).__module__}.{type(value).__qualname__}"
    if isinstance(value, float):
        return (kind, struct.pack("<d", value))
    return (kind, value)


@final
class StateContract:
    """Which attributes of a module are its compile state, and how each is keyed.

    A forward branches in Python on what its module holds — ``if self.training``,
    ``if self._disable_adapters``, ``if name in self.merged_adapters`` — and a
    trace bakes the branch taken into the graph.  So an executable is the same
    only for calls made while every module of the tree held the same such
    values; each one is part of the key.

    The state is every attribute in a module's ``__dict__``, public or private
    (LCD-704: a private flag or a list of names went unseen and a compiled call
    replayed the branch of a state its model had left), whose value is an
    immutable atom or a short container of them.  A tensor, a module or a
    callable is not: tensors are fed and followed by identity, modules are
    walked.  Left out are the dunder names, and the bookkeeping every
    :class:`~lucid.nn.Module` keeps — its registries and hook sets, which the
    walk reads for itself.

    An assignment moves the tree version, so what a module holds directly is
    read once per walk (:meth:`fixed`).  A list, set or dict changed in place
    moves nothing; those containers are held by the walk and read again on
    every call (:meth:`held`, :meth:`read`).

    The forward itself is never watched: the tracer records ops, not the
    Python attribute reads that decide which ops run, so what a branch reads
    is not known — every value it could read is keyed instead.  Measured over
    the zoo (79 families, eval and train), no module's state moved from one
    forward to the next; two moved on their first forward only.
    """

    #: Values keyed by their type and value (a float by its bits, :func:`atom_sig`).
    ATOMS: ClassVar[tuple[type, ...]] = (
        type(None),
        bool,
        int,
        float,
        complex,
        str,
        bytes,
        Enum,
        _Dtype,
    )
    #: Containers keyed item by item, tagged by kind: a set's items unordered,
    #: a dict's by their ``str`` keys.
    CONTAINERS: ClassVar[dict[type, str]] = {
        tuple: "tuple",
        list: "list",
        frozenset: "frozenset",
        set: "set",
        dict: "dict",
    }
    #: The containers that change without an assignment.
    MUTABLE: ClassVar[tuple[type, ...]] = (list, set, dict)
    #: A container keyed holds at most this many items, nested this deep —
    #: enough for a list of adapter names or a tuple of kernel sizes; a larger
    #: one is data, not a switch, and keying it would cost its size every call.
    MAX_ITEMS: ClassVar[int] = 64
    MAX_DEPTH: ClassVar[int] = 3

    @classmethod
    def value_sig(cls, value: object, depth: int = 0) -> object | None:
        """The hashable signature of ``value``, or ``None`` if it is not state."""
        if isinstance(value, cls.ATOMS):
            return atom_sig(value)
        if depth >= cls.MAX_DEPTH:
            return None
        if isinstance(value, dict):
            return cls._dict_sig(value, depth)
        if not isinstance(value, (tuple, list, frozenset, set)) or len(value) > cls.MAX_ITEMS:
            return None
        items: list[object] = []
        for item in value:
            sig = cls.value_sig(item, depth + 1)
            if sig is None:
                return None
            items.append(sig)
        kind = next(name for kind, name in cls.CONTAINERS.items() if isinstance(value, kind))
        return (kind, frozenset(items) if isinstance(value, (set, frozenset)) else tuple(items))

    @classmethod
    def _dict_sig(cls, value: dict[object, object], depth: int) -> object | None:
        if len(value) > cls.MAX_ITEMS:
            return None
        pairs: list[tuple[str, object]] = []
        for key, item in value.items():
            sig = cls.value_sig(item, depth + 1)
            if not isinstance(key, str) or sig is None:
                return None
            pairs.append((key, sig))
        return ("dict", tuple(sorted(pairs, key=lambda pair: pair[0])))

    @staticmethod
    def _candidates(obj: object) -> list[tuple[str, object]]:
        """``obj``'s ``__dict__`` entries that may be state, by name."""
        d = getattr(obj, "__dict__", None)
        if not isinstance(d, dict):
            return []
        skip = _module_bookkeeping()
        found = [
            (name, value)
            for name, value in d.items()
            if isinstance(name, str) and name not in skip and not _is_dunder(name)
        ]
        found.sort(key=lambda pair: pair[0])
        return found

    @classmethod
    def fixed(cls, obj: object) -> tuple[tuple[str, object], ...]:
        """``(name, signature)`` of each state attribute of ``obj`` that only
        an assignment changes."""
        out: list[tuple[str, object]] = []
        for name, value in cls._candidates(obj):
            if not isinstance(value, cls.MUTABLE) and (sig := cls.value_sig(value)) is not None:
                out.append((name, sig))
        return tuple(out)

    @classmethod
    def held(cls, subs: Sequence[object]) -> tuple[HeldState, ...]:
        """Every mutable container the modules ``subs`` hold, to :meth:`read`
        on every call — one that is not state now may become it in place."""
        return tuple(
            (pos, name, value)
            for pos, sub in enumerate(subs)
            for name, value in cls._candidates(sub)
            if isinstance(value, cls.MUTABLE)
        )

    @classmethod
    def read(cls, held: Iterable[HeldState]) -> tuple[tuple[int, str, object], ...]:
        """What each container :meth:`held` keeps now, as its signature."""
        return tuple((pos, name, cls.value_sig(value)) for pos, name, value in held)

    @classmethod
    def attrs(cls, obj: object) -> tuple[tuple[str, object], ...]:
        """``(name, signature)`` of every state attribute of ``obj``, read now."""
        out: list[tuple[str, object]] = []
        for name, value in cls._candidates(obj):
            if (sig := cls.value_sig(value)) is not None:
                out.append((name, sig))
        return tuple(out)


def _is_dunder(name: str) -> bool:
    return len(name) > 4 and name.startswith("__") and name.endswith("__")


@cache
def _module_bookkeeping() -> frozenset[str]:
    """What every module keeps in its ``__dict__`` from ``Module.__init__``:
    the registries and hook sets, and ``training``, which is keyed apart."""
    from lucid.nn.module import Module

    return frozenset(vars(Module()))
