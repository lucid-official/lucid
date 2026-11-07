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
a tensor it made and keeps does, and its signature runs eagerly.  Where it cannot
stand, :meth:`Rebinding.restore` puts the slots back for the eager call that
follows.
"""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import ClassVar, final

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


class _Attributes(_Root):
    """A module's plain attributes, but not the registries and hook sets it keeps."""

    def __init__(self, module: Module) -> None:
        self._module = module

    def names(self) -> dict[str, object]:
        return dict(StateContract.members(self._module))

    def put(self, name: str, value: object) -> None:
        if value is _ABSENT:
            vars(self._module).pop(name, None)
        else:
            vars(self._module)[name] = value

    def label(self, name: str) -> str:
        return f"attribute {name!r} of {type(self._module).__name__}"


class _Callable(_Root):
    """The closure cells and globals a wrapped callable reads."""

    def __init__(self, fn: object) -> None:
        self._fn = fn

    def names(self) -> dict[str, object]:
        return {f"{kind}:{key}": value for (kind, key), value in callable_slots(self._fn)}

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

    def label(self, name: str) -> str:
        kind, _, key = name.partition(":")
        return f"{'closure cell' if kind == 'cell' else 'global'} {key!r}"


@dataclass(frozen=True, slots=True)
class Rebinding:
    """What a finished trace found rebound, and how to put it back.

    Attributes
    ----------
    why : str
        The verdict, for the notice and the trace's unsupported reason.
    undo : tuple of callable
        Each puts one slot or container back as it was before the trace.
    """

    why: str
    undo: tuple[Callable[[], None], ...]

    def restore(self) -> None:
        """Put every slot the call rebound back, for the eager call that follows."""
        for put_back in self.undo:
            put_back()


@final
class SlotWatch:
    """The tensor slots of a model and its wrapped callable, as they were.

    A slot is a module's attribute, an item of a ``dict`` or ``list`` held in one
    (to :attr:`MAX_DEPTH` levels, at most :attr:`MAX_ITEMS` items each), or a
    closure cell or global of the callable.  Larger containers are data, not
    state, and are taken as one value.

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
        self._containers: list[tuple[object, object]] = []
        self._before = self._leaves(keep=self._containers)
        self._values = [root.names() for root in self._roots]

    def _leaves(self, keep: list[tuple[object, object]] | None = None) -> dict[_Path, object]:
        """Every watched slot by path, containers opened.  ``keep`` collects each
        opened container with a copy of what it holds."""
        out: dict[_Path, object] = {}
        for index, root in enumerate(self._roots):
            for name, value in root.names().items():
                self._open((index, name, ()), value, out, keep)
        return out

    def _open(
        self,
        path: _Path,
        value: object,
        out: dict[_Path, object],
        keep: list[tuple[object, object]] | None,
    ) -> None:
        index, name, keys = path
        if (
            len(keys) < self.MAX_DEPTH
            and isinstance(value, (dict, list, tuple))
            and len(value) <= self.MAX_ITEMS
        ):
            if keep is not None and not isinstance(value, tuple):
                keep.append((value, value.copy()))
            for key, item in value.items() if isinstance(value, dict) else enumerate(value):
                self._open((index, name, (*keys, key)), item, out, keep)
        else:
            out[path] = value

    def rebound(self, tracer: _C_engine.compile.Tracer) -> Rebinding | None:
        """What the traced call rebound that a compiled run would get wrong.

        Parameters
        ----------
        tracer : Tracer
            The finished trace, which says which tensors it read as feeds.

        Returns
        -------
        Rebinding or None
            The verdict and the means to put the slots back, ``None`` when no
            slot is rebound that way.
        """
        now = self._leaves()
        fed = {id(impl) for impl in tracer.external_feeds.values()}
        why: list[str] = []
        for path in {*self._before, *now}:
            was, is_ = self._before.get(path, _ABSENT), now.get(path, _ABSENT)
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
            "the slot and would be fed the tensor the trace saw there",
            tuple(self._undo()),
        )

    def _describe(self, path: _Path) -> str:
        index, name, keys = path
        return self._roots[index].label(name) + "".join(f" item {key!r}" for key in keys)

    def _undo(self) -> Iterator[Callable[[], None]]:
        for root, before in zip(self._roots, self._values, strict=True):
            now = root.names()
            for name in {*before, *now}:
                if before.get(name, _ABSENT) is not now.get(name, _ABSENT):
                    yield _put(root, name, before.get(name, _ABSENT))
        for container, copy in self._containers:
            yield _refill(container, copy)


def _same_tensor(a: object, b: object) -> bool:
    return isinstance(a, Tensor) and isinstance(b, Tensor) and a._impl is b._impl


def _put(root: _Root, name: str, value: object) -> Callable[[], None]:
    return lambda: root.put(name, value)


def _refill(container: object, copy: object) -> Callable[[], None]:
    def refill() -> None:
        if isinstance(container, dict) and isinstance(copy, dict):
            container.clear()
            container.update(copy)
        elif isinstance(container, list) and isinstance(copy, list):
            container[:] = copy

    return refill
