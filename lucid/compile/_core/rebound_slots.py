"""
lucid.compile._core.rebound_slots — a tensor slot the traced call rebinds.

A trace runs the forward's Python once.  What an executable is fed is the tensor
each slot held then (a module's attribute, an item of a dict or list it keeps, a
closure cell or a global), and a replay runs no Python: nothing rebinds the slot
and nothing re-reads it.  So a call that rebinds a slot, ``self.h = y.detach()``,
is wrong in a compiled run in one of two ways, and this module tells which:

* **read, then rebound** — the trace read the old tensor as a feed, and every
  later call of the signature is fed it again.  A recurrent state, an EMA kept as
  an attribute and the previous step's feature all answered from the tensor of
  the call that was traced;
* **stored where nothing was** — the slot held ``None`` (or nothing) and the call
  stores a tensor.  The signature is keyed on that ``None``, and a replay of its
  executable stores nothing, so a model that starts again from ``None`` takes
  its first call's branch for good.  Not when what it stores is a constant
  (``Tracer.is_constant``): a mask built once and kept holds the same values
  however often a replay builds it.

A slot the call rebinds without reading it, over a tensor, is neither: a mask
rebuilt before it is used, a reparametrised weight, a tensor kept for someone
outside.  Writing into the tensor a slot holds is not a rebinding either; those
writes are carried (:mod:`~lucid.compile._core.buffer_writes`).

:class:`SlotWatch` is taken before a trace and asked after it.  The one place that
asks is :func:`~lucid.compile._core.buffer_writes.held_writes`, which every entry
that compiles a call already calls: the call stands as the call, as one that wrote
a tensor it made and keeps does, and its signature runs eagerly.

The same watch is also how a refused trace is undone.  The trace-time call ran the
forward's Python, and an eager call that follows it runs that Python again: if the
first left a stamp, a counter, a cache or a rebound tensor behind, the second
finds the work done and skips it (LCD-990).  :meth:`SlotWatch.restore` puts every
slot back as it was before the trace, scalars and containers as much as tensors, so
the eager call is the call.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import ClassVar, final, override

from lucid._C import engine as _C_engine
from lucid._tensor.tensor import Tensor
from lucid.compile._core.callable_slots import callable_slots
from lucid.compile._core.module_state import StateContract
from lucid.nn.module import Module

__all__ = ["REBOUND_MARK", "Rebinding", "SlotWatch"]

#: Words the verdict starts with, for the notice's category.
REBOUND_MARK = "the traced call rebinds"

_ABSENT: object = object()

# A slot: its root's position, its name there, and the keys of the containers
# opened down to it.
type _Path = tuple[int, str, tuple[object, ...]]


class _Root:
    """Where named slots live: a module's ``__dict__`` or a callable's cells and globals."""

    def names(self) -> dict[str, object]:
        """Each slot's name and what it holds now."""
        raise NotImplementedError

    def put(self, name: str, value: object) -> None:
        """Hold ``value`` in slot ``name``; ``_ABSENT`` removes it."""
        raise NotImplementedError

    def label(self, name: str) -> str:
        raise NotImplementedError

    def buffers(self) -> dict[str, object]:
        """The tensors the root keeps in a registry, not as attributes: what
        :meth:`SlotWatch.restore` puts back, and no verdict is read from."""
        return {}

    def rebind_buffer(self, name: str, value: object) -> None:
        """Hold ``value`` in registered buffer ``name``."""
        raise NotImplementedError


class _Attributes(_Root):
    """A module's plain attributes, but not the registries and hook sets it keeps."""

    def __init__(self, module: Module) -> None:
        self._module = module

    @override
    def names(self) -> dict[str, object]:
        return dict(StateContract.members(self._module))

    @override
    def put(self, name: str, value: object) -> None:
        if value is _ABSENT:
            vars(self._module).pop(name, None)
        else:
            vars(self._module)[name] = value

    @override
    def label(self, name: str) -> str:
        return f"attribute {name!r} of {type(self._module).__name__}"

    @override
    def buffers(self) -> dict[str, object]:
        return dict(self._module._buffers)

    @override
    def rebind_buffer(self, name: str, value: object) -> None:
        self._module._buffers[name] = value


class _Callable(_Root):
    """The closure cells and globals a wrapped callable reads."""

    def __init__(self, fn: object) -> None:
        self._fn = fn

    @override
    def names(self) -> dict[str, object]:
        return {f"{kind}:{key}": value for (kind, key), value in callable_slots(self._fn)}

    @override
    def put(self, name: str, value: object) -> None:
        kind, _, key = name.partition(":")
        if kind == "cell":
            cells = getattr(self._fn, "__closure__", None) or ()
            if value is not _ABSENT and int(key) < len(cells):
                cells[int(key)].cell_contents = value
            return
        space = getattr(self._fn, "__globals__", None)
        if isinstance(space, dict):
            if value is _ABSENT:
                space.pop(key, None)
            else:
                space[key] = value

    @override
    def label(self, name: str) -> str:
        kind, _, key = name.partition(":")
        return f"{'closure cell' if kind == 'cell' else 'global'} {key!r}"


@dataclass(frozen=True, slots=True)
class Rebinding:
    """What a finished trace found rebound.

    Attributes
    ----------
    why : str
        The verdict, for the notice and the trace's unsupported reason.
    """

    why: str


@dataclass(frozen=True, slots=True)
class _State:
    """Every watched slot at one moment."""

    #: What each root holds, by name, in the roots' order.
    names: tuple[dict[str, object], ...]
    #: What each root keeps in its buffer registry.
    buffers: tuple[dict[str, object], ...]
    #: The slots by path, containers opened.
    leaves: dict[_Path, object]
    #: What puts each container opened back as it was.
    containers: tuple[Callable[[], None], ...]


@final
class SlotWatch:
    """The tensor slots of a model and its wrapped callable, as they were.

    A slot is a module's attribute, an item of a ``dict`` or ``list`` held in one
    (to :attr:`MAX_DEPTH` levels, at most :attr:`MAX_ITEMS` items each), or a
    closure cell or global of the callable.  Only the built-in ``dict``, ``list``
    and ``tuple`` are opened: a subclass may override what reading or copying it
    does, and that would run in the user's call.  Larger containers are data, not
    state, and are taken as one value.

    Each :meth:`rebound` compares the slots with how they were at the last one
    (or at construction): a segmented step traces a segment's forward and then its
    backward, which runs the forward again, and each is asked for what it rebound.
    :meth:`restore` goes back to construction, whatever was asked since.

    Parameters
    ----------
    modules : sequence of Module
        The modules whose attributes are watched.
    fn : object or None
        The wrapped callable, whose cells and globals are watched.
    """

    MAX_ITEMS: ClassVar[int] = 256
    MAX_DEPTH: ClassVar[int] = 2

    def __init__(self, modules: Sequence[Module], fn: object | None) -> None:
        self._roots: list[_Root] = [_Attributes(m) for m in modules]
        if fn is not None:
            self._roots.append(_Callable(fn))
        self._start = self._base = self._take()

    def _take(self) -> _State:
        names = tuple(root.names() for root in self._roots)
        leaves: dict[_Path, object] = {}
        containers: list[Callable[[], None]] = []
        for index, held in enumerate(names):
            for name, value in held.items():
                self._open((index, name, ()), value, leaves, containers)
        buffers = tuple(root.buffers() for root in self._roots)
        return _State(names, buffers, leaves, tuple(containers))

    def _open(
        self,
        path: _Path,
        value: object,
        leaves: dict[_Path, object],
        containers: list[Callable[[], None]],
    ) -> None:
        index, name, keys = path
        opened = _opened(value, self.MAX_ITEMS) if len(keys) < self.MAX_DEPTH else None
        if opened is None:
            leaves[path] = value
            return
        items, refill = opened
        if refill is not None:
            containers.append(refill)
        for key, item in items:
            self._open((index, name, (*keys, key)), item, leaves, containers)

    def rebound(self, tracer: _C_engine.compile.Tracer) -> Rebinding | None:
        """What the traced call rebound that a compiled run would get wrong.

        Parameters
        ----------
        tracer : Tracer
            The finished trace, which says which tensors it read as feeds.

        Returns
        -------
        Rebinding or None
            The verdict, ``None`` when no slot is rebound that way.
        """
        base, now = self._base, self._take()
        self._base = now
        fed = {id(impl) for impl in tracer.external_feeds.values()}
        why: list[str] = []
        for path in {*base.leaves, *now.leaves}:
            was, is_ = base.leaves.get(path, _ABSENT), now.leaves.get(path, _ABSENT)
            if was is is_ or _same_tensor(was, is_):
                continue
            if isinstance(was, Tensor) and id(was._impl) in fed:
                why.append(f"{self._describe(path)}, which the call reads")
            elif (
                isinstance(is_, Tensor)
                and not isinstance(was, Tensor)
                and not tracer.is_constant(is_._impl)
            ):
                why.append(f"{self._describe(path)}, where it held no tensor")
        if not why:
            return None
        listed = "; ".join(sorted(why)[:3])
        return Rebinding(
            f"{REBOUND_MARK} {listed}: a compiled call runs no Python, so it would not rebind "
            "the slot and would be fed the tensor the trace saw there"
        )

    def restore(self) -> None:
        """Put every slot back as it was when the watch was taken.

        For a refused trace: the eager call that follows runs the forward's Python
        again, and must find what the first run left (a stamp, a counter, a cache,
        a rebound tensor) as it was before it, or it skips the work the first run
        already did and reads what that work was to replace.  A slot the call
        added is removed, and a container held in one is refilled in place.  A
        registered buffer the call rebound (a cache grown by concatenation) is put
        back, one it wrote in place is not a slot: the tracer puts that back.
        """
        start, now = self._start, self._take()
        for root, before, after in zip(self._roots, start.names, now.names, strict=True):
            for name in {*before, *after}:
                if before.get(name, _ABSENT) is not after.get(name, _ABSENT):
                    root.put(name, before.get(name, _ABSENT))
        for root, before, after in zip(self._roots, start.buffers, now.buffers, strict=True):
            for name, value in before.items():
                if value is not after.get(name, _ABSENT):
                    root.rebind_buffer(name, value)
        for refill in start.containers:
            refill()

    def _describe(self, path: _Path) -> str:
        index, name, keys = path
        return self._roots[index].label(name) + "".join(f" item {key!r}" for key in keys)


def _opened(
    value: object, limit: int
) -> tuple[list[tuple[object, object]], Callable[[], None] | None] | None:
    """``value``'s ``(key, item)`` pairs and what puts it back as it is (none for a
    tuple), when it is exactly a ``dict``, ``list``, ``tuple`` or ``set`` of at most
    ``limit`` items.  A set has no positions: its items are not slots, only its copy."""
    if type(value) is dict and len(value) <= limit:
        return list(value.items()), _refiller(_refill_dict, value, value.copy())
    if type(value) is list and len(value) <= limit:
        return list(enumerate(value)), _refiller(_refill_list, value, value.copy())
    if type(value) is tuple and len(value) <= limit:
        return list(enumerate(value)), None
    if type(value) is set and len(value) <= limit:
        return [], _refiller(_refill_set, value, value.copy())
    return None


def _same_tensor(a: object, b: object) -> bool:
    return isinstance(a, Tensor) and isinstance(b, Tensor) and a._impl is b._impl


def _refiller[C](refill: Callable[[C, C], None], target: C, copy: C) -> Callable[[], None]:
    return lambda: refill(target, copy)


def _refill_dict(target: dict[object, object], copy: dict[object, object]) -> None:
    target.clear()
    target.update(copy)


def _refill_set(target: set[object], copy: set[object]) -> None:
    target.clear()
    target.update(copy)


def _refill_list(target: list[object], copy: list[object]) -> None:
    target[:] = copy
