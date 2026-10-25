"""
lucid.compile._core.signature — Phase 1.4 CacheKey.

A :class:`CacheKey` is the hashable identity of a compiled executable
under the user-facing :class:`CompiledModule`.  Two calls land on the
same executable iff their keys compare equal:

  * Same input *positional* and *keyword* tensor shapes / dtypes.
  * Same device for every input tensor.
  * Same ``training`` flag on the wrapped model (so a ``train()`` ↔
    ``eval()`` flip forces a recompile — BN / Dropout differ).
  * Same ``dynamic`` flag at compile time (the symbolic-batch knob,
    Phase 1.6).
  * Same parameter dtype (so ``.half()`` / ``.float()`` triggers a
    recompile).
  * Same binding of the call's tensors: which of them are one tensor, and
    which are a tensor the model holds (a parameter passed as the input).

The key intentionally does NOT capture:

  * Parameter *values* — those flow in via the cached executable's
    feeds at call time; values mutate between optimizer steps but the
    graph stays valid.
  * Input data values — only the shape / dtype / device of the input
    matters to the captured graph.

Non-tensor positional / keyword arguments must be hashable themselves;
the key embeds each with its type (and a float by its bits) so a Python
``int`` / ``bool`` / ``float`` / ``str`` distinguishes signatures (useful
for ``forward(self, x, training=True)`` style guard arguments).
"""

import struct
import weakref
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from types import CodeType, MethodType
from typing import TYPE_CHECKING, final, override

from lucid._C import engine as _C_engine
from lucid.nn.module import _ModuleWrapper, _watch_tensors

if TYPE_CHECKING:
    from lucid._C.engine.compile import Tracer
    from lucid._tensor.tensor import Tensor
    from lucid.nn.module import Module

__all__ = ["TensorSig", "CacheKey", "signature_of"]

# Where a tensor sits that a compiled call reads: the position of its module
# in the tree walk, the registry or attribute it is under, and its name.  The
# wrapped callable's own closure cells and globals sit at position -1.
type _Slot = tuple[int, str, str]


@dataclass(frozen=True, slots=True)
class TensorSig:
    """One tensor's hashable description.

    Captures only what the graph capture path actually depends on:
    rank, per-axis size, dtype name, device type, and a flag whether
    the tensor required grad at trace time (so a leaf vs non-leaf
    input never collides).

    Attributes
    ----------
    shape : tuple of int
        Per-axis sizes.  When ``dynamic_batch`` was active during
        construction, ``shape[0]`` is ``-1`` to wildcard the batch
        dim.
    dtype : str
        Stringified dtype (``"lucid.float32"``, ``"lucid.int64"``, …).
    device : str
        Device kind (``"cpu"`` / ``"metal"``).  Index is intentionally
        dropped — Lucid currently runs a single device per kind.
    requires_grad : bool
        Whether the tensor was a graph leaf at trace time.

    Examples
    --------
    >>> import lucid
    >>> from lucid.compile._core.signature import TensorSig
    >>> sig = TensorSig.of(lucid.randn(4, 8))
    >>> sig.shape, sig.dtype, sig.device
    ((4, 8), 'lucid.float32', 'cpu')

    See Also
    --------
    CacheKey : the hashable tuple that embeds many ``TensorSig``\\ s.
    signature_of : the function that builds a key from a call site.
    """

    shape: tuple[int, ...]
    dtype: str
    device: str
    requires_grad: bool

    @classmethod
    def of(cls, t: Tensor, *, dynamic_batch: bool = False) -> TensorSig:
        """Build a :class:`TensorSig` summary of ``t`` for cache keying.

        Parameters
        ----------
        t : Tensor
            Input whose shape / dtype / device / ``requires_grad``
            are read.  The values themselves are never captured.
        dynamic_batch : bool, optional
            When ``True`` wildcard the leading axis to ``-1`` so two
            calls that differ only in batch size share a cache
            entry.  Phase 1.6 opt-in; currently always called with
            the default ``False``.

        Returns
        -------
        TensorSig
            Hashable description ready to embed in a
            :class:`CacheKey`.
        """
        # ``device.type`` (e.g. ``'metal'`` / ``'cpu'``) is hashable;
        # full device repr would include an index slot we don't need.
        # When ``dynamic_batch`` is set we coerce the leading dim of
        # any ≥1-D tensor to ``-1`` so different batch sizes share
        # the same cache entry (Phase 1.6 symbolic-batch opt-in).
        shape = tuple(t.shape)
        if dynamic_batch and len(shape) >= 1:
            shape = (-1, *shape[1:])
        return cls(
            shape=shape,
            dtype=str(t.dtype),
            device=str(t.device.type),
            requires_grad=bool(t.requires_grad),
        )


@dataclass(frozen=True, slots=True)
class CacheKey:
    """Hashable identity of a CompiledModule signature.

    Two calls share an executable iff their CacheKeys compare equal.
    The dataclass is ``@dataclass(frozen=True)`` so it can sit in a :class:`set` /
    :class:`dict`; equality / hash both fall out of the field tuple.

    Attributes
    ----------
    args : tuple
        Per-argument signature (``TensorSig`` for tensors, typed atom
        for primitives, nested tuple for containers).  Position-
        sensitive.
    kwargs : tuple of (str, object)
        Same shape as ``args`` but keyed by argument name and sorted
        — insertion order can't shift the key.
    training : bool
        Snapshot of ``model.training`` at trace time.  Switching
        ``train()`` / ``eval()`` forces a separate cache entry
        because BatchNorm / Dropout dispatch differently.
    dynamic : bool
        Phase 1.6 symbolic-batch flag.  ``True`` means every tensor
        sig has its leading dim wildcarded.
    param_fingerprint : tuple of (str, str)
        Sorted ``(dtype, device)`` pairs across the model's parameters
        — flips when the user calls ``.half()`` / ``.to('cpu')``.
    alias : tuple of int
        For each argument (positional, then keyword by name), the index
        of the first argument that is the same tensor, or ``-1`` for a
        non-tensor.  ``f(x, x)`` traces to one feed read twice and
        ``f(a, b)`` to two, at identical shapes — without this they
        shared an entry and ``f(a, b)`` ran the one-feed executable.
    tree_alias : tuple of (int, slot)
        For each of the call's tensors that is one the model holds — a
        parameter, a buffer, a tensor attribute a trace read — its position
        among the call's tensors and the slot it sits in (module position,
        registry, name).  Empty when no argument is.  ``m(m.weight)``
        traces to one feed the weight and the input share, so another
        tensor of the same signature ran with the weight replaced by it
        (LCD-629); ``alias`` cannot see this, since it compares the
        arguments only with each other.
    grad_mode : bool
        Snapshot of ``grad_enabled()`` at the call site.  A frozen model
        with an ``is_grad_enabled()`` branch traces different graphs under
        ``enable_grad`` and ``no_grad``, so each mode gets its own entry.
    grad : bool
        Whether the call needs a differentiable result: grad mode was on
        and a tensor it reads requires grad.  Such a call runs eagerly, so
        its key never names an executable; it is its own key so that the
        verdict never falls on the same call made under ``no_grad``.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.compile._core.signature import signature_of
    >>> model = nn.Linear(8, 4).to("metal")
    >>> x = lucid.randn(2, 8, device="metal")
    >>> compiled = lucid.compile(model)
    >>> with lucid.no_grad():
    ...     key = signature_of(model, (x,), {}, dynamic=False)
    ...     key in compiled._cache           # is this signature compiled yet?
    ...     _ = compiled(x)
    ...     key in compiled._cache
    False
    True

    See Also
    --------
    TensorSig : per-tensor descriptor that fills ``args`` / ``kwargs``.
    signature_of : the canonical builder for a key.
    """

    args: tuple[object, ...]
    kwargs: tuple[tuple[str, object], ...]
    training: bool
    dynamic: bool
    # A coarse fingerprint of parameter dtypes / devices — enough that
    # ``.half()`` / ``.to('cpu')`` invalidate the key without us having
    # to walk every parameter on every call.  Captured as a sorted
    # tuple of ``(dtype, device)`` pairs (no name, no shape).
    param_fingerprint: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    # Which arguments are the same tensor; see ``alias`` above.
    alias: tuple[int, ...] = field(default_factory=tuple)
    # Which arguments are a tensor the model holds; see ``tree_alias`` above.
    tree_alias: tuple[tuple[int, _Slot], ...] = field(default_factory=tuple)
    # Per-submodule ``(training, scalar_attrs)`` snapshot across ``model.modules()``.
    module_state: tuple[object, ...] = field(default_factory=tuple)
    # Snapshot of ``grad_enabled()`` at the call site; see ``grad_mode`` above.
    grad_mode: bool = False
    # The call needs gradients and runs eagerly; see ``grad`` above.
    grad: bool = False


def _wrapped_callable(model: object) -> object | None:
    """The function a ``lucid.compile(fn)`` wrapper runs; ``None`` for a module.

    Read from ``__dict__``: on a module, ``getattr`` of a missing name goes
    through ``Module.__getattr__`` on every call.
    """
    attrs = getattr(model, "__dict__", None)
    return attrs.get("_fn") if isinstance(attrs, dict) else None


# By the code object's address: hashing a code object hashes its bytecode and
# constants, ~2 µs a call for a large function.  An entry leaves with its code.
_GLOBAL_NAMES: dict[int, tuple[weakref.ref[CodeType], tuple[str, ...]]] = {}


def _global_names(code: CodeType) -> tuple[str, ...]:
    """The names ``code`` and the code nested in it may read as globals, sorted.

    ``co_names`` holds attribute names too, so this over-approximates: a
    global that only shares a name with an attribute is watched for nothing
    worse than an extra trace.
    """
    key = id(code)
    hit = _GLOBAL_NAMES.get(key)
    if hit is not None:
        return hit[1]
    found = set(code.co_names)
    for const in code.co_consts:
        if isinstance(const, CodeType):
            found.update(_global_names(const))
    names = tuple(sorted(found))
    gone = partial(_GLOBAL_NAMES.pop, key, None)
    _GLOBAL_NAMES[key] = (weakref.ref(code, lambda _: gone()), names)
    return names


def _cell_value(cell: object) -> object:
    """A closure cell's contents; ``None`` while its variable is unbound."""
    try:
        return getattr(cell, "cell_contents", None)
    except ValueError:  # an empty cell: the variable is assigned later
        return None


def _read_slot(fn: object, key: tuple[str, str]) -> object:
    """What ``fn``'s closure cell or global ``key`` holds now."""
    kind, name = key
    if kind == "cell":
        cells = getattr(fn, "__closure__", None) or ()
        return _cell_value(cells[int(name)]) if int(name) < len(cells) else None
    space = getattr(fn, "__globals__", None)
    return space.get(name) if isinstance(space, dict) else None


def _callable_slots(fn: object) -> list[tuple[tuple[str, str], object]]:
    """What ``fn`` reads that no argument hands it: its closure cells and globals.

    A global counts when ``fn``'s code names it (LCD-599): a module-level
    model is as much the function's state as one it closes over.  What a
    function ``fn`` calls reads in its own globals is not followed.
    """
    out: list[tuple[tuple[str, str], object]] = [
        (("cell", str(i)), _cell_value(cell))
        for i, cell in enumerate(getattr(fn, "__closure__", None) or ())
    ]
    code = getattr(fn, "__code__", None)
    space = getattr(fn, "__globals__", None)
    if isinstance(code, CodeType) and isinstance(space, dict):
        out.extend(
            (("global", name), space[name])
            for name in _global_names(code)
            if name in space
        )
    return out


# A hop in a path from the wrapped callable to something it reaches
# indirectly: ("cell"/"global", name) relative to a callable, ("item", key)
# into a list/tuple (by stringified index) or a str-keyed dict, or ("attr",
# name) on a plain object's ``__dict__``.
type _Hop = tuple[str, str]
type _Path = tuple[_Hop, ...]

# Bounds for the indirect walk below (LCD-642): a container or an object's
# attributes cost the same regardless of their size, and a chain of helper
# functions is followed only this many hops deep — enough for the shapes
# seen in practice (one container, or one helper calling the model) with a
# little headroom, not a search of the whole reachable heap.
_MAX_PATH_DEPTH = 3
_MAX_PATH_ITEMS = 64


def _sequence_children(value: Sequence[object]) -> Iterator[tuple[_Hop, object]]:
    """``("item", index)`` for each of ``value``'s first ``_MAX_PATH_ITEMS``."""
    for i, item in enumerate(value):
        if i >= _MAX_PATH_ITEMS:
            return
        yield ("item", str(i)), item


def _dict_children(value: dict[object, object]) -> Iterator[tuple[_Hop, object]]:
    """``("item", key)`` for each str key among ``value``'s first
    ``_MAX_PATH_ITEMS``."""
    for i, (key, item) in enumerate(value.items()):
        if i >= _MAX_PATH_ITEMS:
            return
        if isinstance(key, str):
            yield ("item", key), item


def _function_children(
    value: object, seen_funcs: set[int]
) -> Iterator[tuple[_Hop, object]]:
    """``value``'s own cells and globals, once per function (cycle guard)."""
    if id(value) in seen_funcs:
        return
    seen_funcs.add(id(value))
    yield from _callable_slots(value)


def _object_children(value: object) -> Iterator[tuple[_Hop, object]]:
    """``("attr", name)`` for each public attribute among ``value``'s first
    ``_MAX_PATH_ITEMS``, or nothing if ``value`` has no ``__dict__``."""
    try:
        attrs = vars(value)
    except TypeError:  # no __dict__ (a plain scalar, a slotted object, …)
        return
    for i, (name, item) in enumerate(attrs.items()):
        if i >= _MAX_PATH_ITEMS:
            return
        if not name.startswith("_"):
            yield ("attr", name), item


def _path_children(
    value: object, seen_funcs: set[int]
) -> Iterator[tuple[_Hop, object]]:
    """One further hop's worth of values reachable from ``value``: the items
    of a list/tuple/dict, the cells and globals of a function it is, or the
    public attributes of a plain object — each capped at
    ``_MAX_PATH_ITEMS`` so a large one costs the same as a small one.
    """
    from lucid._tensor.tensor import Tensor
    from lucid.nn.module import Module

    if isinstance(value, (list, tuple)):
        yield from _sequence_children(value)
    elif isinstance(value, dict):
        yield from _dict_children(value)
    elif callable(value) and hasattr(value, "__code__"):
        yield from _function_children(value, seen_funcs)
    elif not isinstance(value, (Module, Tensor)):
        yield from _object_children(value)


def _indirect_module_paths(fn: object) -> list[tuple[_Path, Module]]:
    """Every module ``fn`` reaches beyond its own direct cells and globals.

    A dict/list global, a helper function ``fn`` calls that itself closes
    over or names a global module, and a plain attribute of an object one
    of those holds, are each walked — bounded by ``_MAX_PATH_DEPTH`` hops
    and ``_MAX_PATH_ITEMS`` per container, function or object so the cost
    never scales with how large the container is (LCD-642: a module
    reached only this way used to go unwatched, so a parameter swap or a
    rebind after compile answered silently wrong).
    """
    from lucid.nn.module import Module

    if fn is None:
        return []
    found: list[tuple[_Path, Module]] = []
    seen_funcs: set[int] = {id(fn)}
    seen_values: set[int] = set()
    frontier: list[tuple[_Path, object]] = [
        ((hop,), value) for hop, value in _callable_slots(fn)
    ]
    for _ in range(_MAX_PATH_DEPTH):
        if not frontier:
            break
        next_frontier: list[tuple[_Path, object]] = []
        for path, value in frontier:
            if isinstance(value, Module):
                if len(path) > 1:
                    found.append((path, value))
                continue
            vid = id(value)
            if vid in seen_values:
                continue
            seen_values.add(vid)
            for hop, child in _path_children(value, seen_funcs):
                next_frontier.append((path + (hop,), child))
        frontier = next_frontier
    return found


def _resolve_path(fn: object, path: _Path) -> object | None:
    """What following ``path`` from ``fn`` reaches now, or ``None`` if a hop
    is gone — the cheap, targeted re-check :meth:`_TreeTensors.reads_the_same`
    runs every call, as opposed to :func:`_indirect_module_paths`'s search
    for one."""
    current: object = fn
    for kind, key in path:
        if kind in ("cell", "global"):
            current = _read_slot(current, (kind, key))
        elif kind == "item":
            if isinstance(current, dict):
                current = current.get(key)
            elif isinstance(current, (list, tuple)):
                idx = int(key)
                current = current[idx] if 0 <= idx < len(current) else None
            else:
                return None
        elif kind == "attr":
            current = getattr(current, key, None)
        else:
            return None
    return current


def _closure_modules(model: object) -> list[Module]:
    """The modules ``model._fn`` is bound to, closes over, reads as a global,
    or reaches through a container, a helper function or an object's
    attribute (:func:`_indirect_module_paths`, LCD-642)."""
    from lucid.nn.module import Module

    fn = _wrapped_callable(model)
    if fn is None:
        return []
    # A bound method — ``lucid.compile(model.forward)`` — runs its module's
    # tree as a closure would.  Read on every call (``_tree_walk``), so the
    # values only, not ``_callable_slots``' keys.
    held = [getattr(fn, "__self__", None)]
    held += map(_cell_value, getattr(fn, "__closure__", None) or ())
    code = getattr(fn, "__code__", None)
    space = getattr(fn, "__globals__", None)
    if isinstance(code, CodeType) and isinstance(space, dict):
        held += map(space.get, _global_names(code))
    direct = [v for v in held if isinstance(v, Module)]
    indirect = [m for _, m in _indirect_module_paths(fn)]
    seen: set[int] = set()
    out: list[Module] = []
    for m in (*direct, *indirect):
        if id(m) not in seen:
            seen.add(id(m))
            out.append(m)
    return out


def _root_modules(model: object) -> list[Module]:
    """``model`` if it is a module, then the modules its ``_fn`` closes over."""
    from lucid.nn.module import Module

    roots: list[Module] = [model] if isinstance(model, Module) else []
    roots.extend(_closure_modules(model))
    return roots


def _iter_modules(roots: Sequence[Module]) -> list[Module]:
    """Every module reachable from ``roots``, each once, in deterministic order.

    A registered parametrization's :class:`~lucid.nn.utils.parametrize.ParametrizationList`
    sits in ``parametrizations`` as a plain submodule (LCD-624), so ``root.modules()``
    already walks into it — nothing extra is needed to reach the originals it holds.
    """
    seen: set[int] = set()
    out: list[Module] = []
    for root in roots:
        for sub in root.modules():
            if id(sub) not in seen:
                seen.add(id(sub))
                out.append(sub)
    return out


def _scalar_attr_sig(value: object) -> object | None:
    """Hashable signature for a scalar (or scalar-tuple) attribute, else ``None``."""
    if isinstance(value, (int, float, bool, str, bytes, type(None))):
        return _atom_sig(value)
    if isinstance(value, tuple):
        items: list[object] = []
        for item in value:
            sig = _scalar_attr_sig(item)
            if sig is None:
                return None
            items.append(sig)
        return tuple(items)
    return None


def _dict_scalar_attrs(obj: object) -> tuple[tuple[str, object], ...]:
    """Sorted ``(name, sig)`` pairs for public scalar attributes in ``obj.__dict__``."""
    d = getattr(obj, "__dict__", None)
    if not isinstance(d, dict):
        return ()
    attrs: list[tuple[str, object]] = []
    for k, v in d.items():
        if isinstance(k, str) and not k.startswith("_"):
            sig = _scalar_attr_sig(v)
            if sig is not None:
                attrs.append((k, sig))
    attrs.sort(key=lambda item: item[0])
    return tuple(attrs)


def _module_state(model: object) -> tuple[object, ...]:
    """Snapshot every submodule's ``training`` flag and public scalar attributes."""
    if model is None:
        return ()
    states = _tree_walk(model).states
    fn = getattr(model, "_fn", None)
    if fn is not None:
        fn_attrs = _dict_scalar_attrs(fn)
        if fn_attrs:
            return (*states, ("fn", fn_attrs))
    return states


def _handle_key(handle: object) -> int | None:
    """Return integer hook handle key if ``handle`` is a hook handle."""
    key = getattr(handle, "_key", None)
    return key if isinstance(key, int) else None


def _internal_pre_hook_keys(mod: Module) -> set[int]:
    """Hook ids installed by ``nn.utils`` reparametrisations (traced in-graph).

    ``nn.utils.parametrize`` installs no forward pre-hook since LCD-624 — the
    reparametrised tensor is read through a property on a generated subclass,
    not recomputed by a hook — so it contributes nothing here.
    """
    handles: list[object] = []
    for attr in ("_weight_norm_hooks", "_spectral_norm_hooks"):
        d = getattr(mod, attr, None)
        if isinstance(d, dict):
            handles.extend(e[0] for e in d.values() if isinstance(e, tuple) and e)
    prune_d = getattr(mod, "_prune_hooks", None)
    if isinstance(prune_d, dict):
        handles.extend(prune_d.values())
    return {k for h in handles if (k := _handle_key(h)) is not None}


def _sub_has_active_hooks(sub: Module) -> bool:
    """Return ``True`` if single module ``sub`` has user forward, backward, or parameter hooks."""
    from lucid.autograd._hooks import _has_active_tensor_hooks

    if (
        getattr(sub, "_forward_hooks", None)
        or getattr(sub, "_backward_pre_hooks", None)
        or getattr(sub, "_backward_hooks", None)
    ):
        return True
    pre_hooks = getattr(sub, "_forward_pre_hooks", None)
    if pre_hooks and (set(pre_hooks) - _internal_pre_hook_keys(sub)):
        return True
    params = getattr(sub, "_parameters", None)
    if isinstance(params, dict):
        for p in params.values():
            if p is not None and _has_active_tensor_hooks(p):
                return True
    return False


@dataclass(frozen=True, slots=True)
class _TreeWalk:
    """What one walk of a model's module tree found, and when it was taken."""

    version: int
    # The roots walked from, held weakly: a closure that now names another
    # module is another tree, and a root that has died cannot be mistaken
    # for a new module at its address.
    roots: tuple[weakref.ref[Module], ...]
    states: tuple[object, ...]
    hooked: bool
    # Every parameter and buffer of the tree.  Their ``requires_grad`` moves
    # no version, so it is read from these each time it is asked.
    tensors: tuple[Tensor, ...]
    # The slot each of them sits in, by ``TensorImpl`` identity.
    slot_of: Mapping[int, _Slot]

    def walked_from(self, roots: Sequence[Module]) -> bool:
        return _same_roots(self.roots, roots)


def _same_roots(refs: Sequence[weakref.ref[Module]], roots: Sequence[Module]) -> bool:
    """Whether ``roots`` are the modules ``refs`` were taken of, in order."""
    return len(roots) == len(refs) and all(
        ref() is root for ref, root in zip(refs, roots, strict=True)
    )


def _registry_slots(subs: Sequence[Module]) -> Iterator[tuple[_Slot, Tensor | None]]:
    """Every parameter and buffer slot of ``subs``, with what it holds."""
    for pos, sub in enumerate(subs):
        for kind, registry in (
            ("parameter", sub._parameters),
            ("buffer", sub._buffers),
        ):
            for name, tensor in registry.items():
                yield (pos, kind, name), tensor


def _slot_map(slots: Iterable[tuple[_Slot, Tensor | None]]) -> dict[int, _Slot]:
    """The first slot each tensor sits in, by ``TensorImpl`` identity — the
    identity the feed binding merges a trace's reads by."""
    out: dict[int, _Slot] = {}
    for slot, tensor in slots:
        if tensor is not None:
            out.setdefault(id(tensor._impl), slot)
    return out


_WALKS: weakref.WeakKeyDictionary[Module, _TreeWalk] = weakref.WeakKeyDictionary()


def _tree_walk(model: object) -> _TreeWalk:
    """The ``training`` flags, scalar attributes and hooks of ``model``'s tree.

    The walk is O(submodules) — a millisecond a call on a 200-module model —
    so it is kept per model and taken again only when the tree version
    (``lucid/nn/_tree_version.py``) has moved or the closure names other
    modules.  The version is read before the walk: a change made while it
    runs moves it past the version stored, and the next call walks again.
    """
    from lucid.autograd._hooks import _tensor_hook_version
    from lucid.nn._tree_version import _tree_version
    from lucid.nn.module import Module

    version = _tree_version() + _tensor_hook_version()
    roots = _root_modules(model)
    cached = _WALKS.get(model) if isinstance(model, Module) else None
    if cached is not None and cached.version == version and cached.walked_from(roots):
        return cached
    subs = _iter_modules(roots)
    slots = tuple(_registry_slots(subs))
    walk = _TreeWalk(
        version=version,
        roots=tuple(weakref.ref(root) for root in roots),
        states=tuple(
            (bool(getattr(sub, "training", False)), _dict_scalar_attrs(sub))
            for sub in subs
        ),
        hooked=any(_sub_has_active_hooks(sub) for sub in subs),
        tensors=tuple(t for _, t in slots if t is not None),
        slot_of=_slot_map(slots),
    )
    if isinstance(model, Module):
        _WALKS[model] = walk
    return walk


def tree_tensors(model: object) -> tuple[Tensor, ...]:
    """Every parameter and buffer of ``model``'s tree, from the cached walk.

    The tree takes in the modules a wrapped function is bound to, closes over
    or reads as a global, and the containers of a reparametrisation.
    """
    return _tree_walk(model).tensors


def tree_requires_grad(model: object) -> bool:
    """Whether a parameter or buffer of ``model``'s tree requires grad.

    The modules a wrapped callable is bound to or closes over are part of
    the tree.  The walk is the one kept per tree version; the flags are read
    from it on every call, as freezing a parameter moves no version.
    """
    return any(t.requires_grad for t in tree_tensors(model))


def model_has_active_hooks(model: object) -> bool:
    """Return ``True`` if ``model`` or any submodule has active forward/backward hooks."""
    from lucid.nn.hooks import (
        _GLOBAL_BACKWARD_HOOKS,
        _GLOBAL_BACKWARD_PRE_HOOKS,
        _GLOBAL_FORWARD_HOOKS,
        _GLOBAL_FORWARD_PRE_HOOKS,
    )

    if (
        _GLOBAL_FORWARD_PRE_HOOKS
        or _GLOBAL_FORWARD_HOOKS
        or _GLOBAL_BACKWARD_PRE_HOOKS
        or _GLOBAL_BACKWARD_HOOKS
    ):
        return True
    return _tree_walk(model).hooked


def tensors_have_active_hooks(tensors: Iterable[Tensor]) -> bool:
    """Return ``True`` if any tensor in ``tensors`` carries an active tensor gradient hook (``Tensor.register_hook``)."""
    from lucid.autograd._hooks import _has_active_tensor_hooks

    return any(_has_active_tensor_hooks(t) for t in tensors)


def leaf_tensors(args: Iterable[object], kwargs: dict[str, object]) -> list[Tensor]:
    """Every tensor in the call, nested ones included, in one fixed order.

    Positional arguments first, then keyword arguments by name; lists and
    tuples in order, dicts by key — the order :func:`_arg_sig` normalises
    them in, so two calls with the same :class:`CacheKey` list their
    tensors in the same order.  Feeds bind by position in this list: a
    tensor inside a list used to be bound only at the top level, so it
    was pinned at its trace-time value and every later call silently
    reused it.

    Parameters
    ----------
    args : iterable of object
        The call's positional arguments, walked in order.
    kwargs : dict of str to object
        The call's keyword arguments, walked by name.

    Returns
    -------
    list of Tensor
        Every tensor found, in the order described above; the same tensor
        appears once per position it occupies.
    """
    from lucid._tensor.tensor import Tensor

    out: list[Tensor] = []

    def walk(value: object) -> None:
        if isinstance(value, Tensor):
            out.append(value)
        elif isinstance(value, (list, tuple)):
            for v in value:
                walk(v)
        elif isinstance(value, dict):
            for k in sorted(value, key=str):
                walk(value[k])

    for a in args:
        walk(a)
    for k in sorted(kwargs):
        walk(kwargs[k])
    return out


def _alias_pattern(leaves: Sequence[Tensor]) -> tuple[int, ...]:
    """First-occurrence index of each of the call's tensors (:func:`leaf_tensors`).

    Identity is the ``TensorImpl`` — the test the feed binding uses to
    map a trace's external feeds back to the call's tensors, so two keys
    differ exactly when the binding would.
    """
    seen: dict[int, int] = {}
    return tuple(seen.setdefault(id(t._impl), i) for i, t in enumerate(leaves))


def _tree_alias(
    leaves: Sequence[Tensor], slot_of: Mapping[int, _Slot]
) -> tuple[tuple[int, _Slot], ...]:
    """The call's tensors that are tensors the model holds, with their slots.

    A trace reads such a tensor once, through one feed the argument and the
    model share: the binding sends that feed to the argument, so the
    executable is the same only for calls that pass the same slot there.
    """
    if not slot_of:
        return ()
    return tuple(
        (i, slot)
        for i, t in enumerate(leaves)
        if (slot := slot_of.get(id(t._impl))) is not None
    )


def outside_feeds(tracer: Tracer) -> tuple[_C_engine.TensorImpl, ...]:
    """Every tensor a finished trace was fed from outside — none it made itself.

    A constant the trace built and folded is a feed too, but no slot of the
    model held it before the call.
    """
    return tuple(
        impl for impl in tracer.external_feeds.values() if not tracer.made_here(impl)
    )


@final
@dataclass(frozen=True, slots=True)
class TraceReads:
    """What a traced executable reads that its call does not hand it.

    Under grad mode, a call that reads a tensor requiring grad runs eagerly:
    its result must require grad (LCD-582).  The arguments are checked on each
    call; this covers the rest — the tensors the executable is fed from its
    own record (parameters, buffers, a closure's or a global's tensors), and
    one the call makes requiring grad itself.  And what the trace was fed from
    outside goes to :meth:`ParamFingerprint.settle` (LCD-606).

    Attributes
    ----------
    pinned : tuple of TensorImpl
        The feeds no argument supplies.
    grad_leaf : bool
        The trace made a tensor that requires grad from no feed — a factory
        given ``requires_grad=True`` inside the forward, which eager hands to
        autograd as a leaf.  Every call of the signature makes it again, so
        this is decided once.
    fed : tuple of TensorImpl
        Every feed from outside the trace (:func:`outside_feeds`).
    """

    pinned: tuple[_C_engine.TensorImpl, ...] = ()
    grad_leaf: bool = False
    fed: tuple[_C_engine.TensorImpl, ...] = ()

    @classmethod
    def of(
        cls, tracer: Tracer, input_ids: Iterable[int], sources: Iterable[object]
    ) -> TraceReads:
        """Read them off a finished trace.

        Parameters
        ----------
        tracer : Tracer
            The trace the executable was lowered from.
        input_ids : iterable of int
            The executable's inputs, by trace id.
        sources : iterable of object
            For each of ``input_ids``, where a call takes it from — ``None``
            for a feed the executable's own record supplies.

        Returns
        -------
        TraceReads
            What the executable reads besides the call.
        """
        feeds = tracer.external_feeds
        ids = {id(impl) for impl in feeds.values()}
        held = tracer.retained_values.values()
        pairs = zip(input_ids, sources, strict=True)
        return cls(
            pinned=tuple(feeds[tid] for tid, src in pairs if src is None),
            grad_leaf=any(
                impl.requires_grad and impl.is_leaf and id(impl) not in ids
                for impl in held
            ),
            fed=outside_feeds(tracer),
        )

    def require_grad(self) -> bool:
        """Whether one of them requires grad — read now, as freezing one moves
        no version."""
        return self.grad_leaf or any(impl.requires_grad for impl in self.pinned)


def _arg_sig(value: object, *, dynamic_batch: bool = False) -> object:
    """Normalise one call argument into a hashable signature leaf.

    Tensors become :class:`TensorSig`; lists / tuples / dicts are
    recursively normalised; primitives (int / float / bool / str /
    None) are keyed by :func:`_atom_sig`.  Anything still unhashable forces
    the caller to fall back to eager (the cache lookup raises and
    :class:`CompiledModule.__call__` catches it).
    """
    # Lucid Tensor → TensorSig; everything else passes through if
    # hashable.  Iterables (list / tuple) are normalised to a tuple of
    # element sigs so a user-side ``forward(self, [x, y])`` is captured
    # too.  ``dynamic_batch`` propagates into the Tensor case so the
    # leading dim is wildcarded.
    from lucid._tensor.tensor import Tensor

    if isinstance(value, Tensor):
        return TensorSig.of(value, dynamic_batch=dynamic_batch)
    if isinstance(value, (list, tuple)):
        return tuple(_arg_sig(v, dynamic_batch=dynamic_batch) for v in value)
    if isinstance(value, dict):
        return tuple(
            sorted(
                (str(k), _arg_sig(v, dynamic_batch=dynamic_batch))
                for k, v in value.items()
            )
        )
    return _atom_sig(value)


def _atom_sig(value: object) -> object:
    """Key a non-tensor argument by its type and, for a float, its bits.

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


def _param_fingerprint(model: Module) -> tuple[tuple[str, str], ...]:
    """Summarise a model's parameter set as a sorted ``(dtype, device)`` tuple.

    Used inside :class:`CacheKey` to invalidate the cache when a user
    flips precision (``.half()`` / ``.float()``) or moves the model
    across devices.  The fingerprint is intentionally coarse — we
    don't track per-parameter shapes (those are already pinned by
    the trace itself) — so it stays cheap to compute and compare.

    Returns
    -------
    tuple of (dtype_str, device_str)
        Sorted, deduplicated pairs across every ``model.parameters()``
        slot.  Empty tuple when the model has no parameters.
    """
    seen: dict[tuple[str, str], int] = {}
    for p in model.parameters():
        key = (str(p.dtype), str(p.device.type))
        seen[key] = seen.get(key, 0) + 1
    return tuple(sorted(seen.keys()))


def _reached_modules(
    values: dict[tuple[str, str], object], fn: object
) -> tuple[tuple[_Path, weakref.ref[Module]], ...]:
    """Every module ``fn`` reaches, direct cells/globals and indirect alike,
    each with the path :meth:`_TreeTensors.reads_the_same` re-follows later.

    A module reached only through a container, a helper function or an
    object's attribute (:func:`_indirect_module_paths`, LCD-642) is keyed by
    the longer path found there instead of a bare ``(key,)``.
    """
    from lucid.nn.module import Module

    reached: dict[_Path, Module] = {
        (key,): v for key, v in values.items() if isinstance(v, Module)
    }
    reached.update(_indirect_module_paths(fn) if fn is not None else ())
    return tuple((path, weakref.ref(v)) for path, v in reached.items())


@final
class _TreeTensors:
    """Which modules a tree holds, and which tensor sits in each of their slots.

    An executable is fed the parameters and buffers it was traced with and
    runs the forwards of the modules that were in the tree then.  A slot
    given another tensor — ``m.w = Parameter(...)``, ``register_buffer``, a
    ``ParameterList`` item, weight tying — or a submodule put in another's
    place leaves the key as it was: same shapes, dtypes and devices.  So the
    tree is compared with this record of it, by identity (LCD-560).

    The record is the addresses, with the tensors behind them held, so no
    new tensor can be given one; the modules are held weakly, and one that
    died has left the tree.

    Besides the registries it records the slots a trace was seen to read
    (:meth:`ParamFingerprint.settle`): a module's plain tensor attributes,
    by name, and the wrapped callable's closure cells and globals (LCD-606).
    Rebinding a cell or a global moves no tree version, so what the callable
    reaches is checked on every call (:meth:`reads_the_same`).
    """

    __slots__ = ("_record", "_modules", "_reached", "_read", "slot_of")

    def __init__(
        self,
        model: object,
        attributes: Mapping[Module, tuple[str, ...]] | None = None,
        callable_slots: Sequence[tuple[str, str]] = (),
    ) -> None:
        subs = _iter_modules(_root_modules(model))
        slots = list(_registry_slots(subs))
        if attributes:
            slots.extend(
                ((pos, "attribute", name), _as_tensor(sub.__dict__.get(name)))
                for pos, sub in enumerate(subs)
                for name in attributes.get(sub, ())
            )
        fn = _wrapped_callable(model)
        values = dict(_callable_slots(fn)) if fn is not None else {}
        reads = [(key, _as_tensor(values.get(key))) for key in callable_slots]
        slots.extend(((-1, *key), tensor) for key, tensor in reads)
        ids: list[object] = [id(sub) for sub in subs]
        held: list[object] = []
        for slot, tensor in slots:
            impl = None if tensor is None else tensor._impl
            ids += (slot, id(tensor), id(impl))
            held += (tensor, impl)
        self._record = (tuple(ids), tuple(held))
        self._modules = tuple(weakref.ref(sub) for sub in subs)
        # The callable's cells and globals this record depends on: the modules
        # they name (held weakly, as the tree's are) and the tensors read.
        self._reached = _reached_modules(values, fn)
        self._read = tuple((key, values.get(key)) for key in callable_slots)
        # Where each recorded tensor sits, for the key (``CacheKey.tree_alias``).
        self.slot_of = _slot_map(slots)

    def same_as(self, now: _TreeTensors) -> bool:
        """Whether ``now`` holds the modules and tensors this record does."""
        return self._record[0] == now._record[0] and all(
            ref() is not None for ref in self._modules
        )

    def reads_the_same(self, model: object) -> bool:
        """Whether the wrapped callable reaches the modules and recorded tensors
        it did — a module's tree moves the version when it changes, a rebound
        closure cell or global does not, and neither does one swapped inside
        a container or a helper function's own globals (LCD-642): each
        ``_reached`` path is re-followed from ``fn``, not rescanned, so this
        stays cheap regardless of a container's size."""
        if not self._reached and not self._read:
            return True
        fn = _wrapped_callable(model)
        return all(
            _resolve_path(fn, path) is ref() for path, ref in self._reached
        ) and all(_read_slot(fn, key) is value for key, value in self._read)


def _as_tensor(value: object) -> Tensor | None:
    """``value`` if it is a tensor, else ``None``."""
    from lucid._tensor.tensor import Tensor

    return value if isinstance(value, Tensor) else None


# A tensor held outside the registries: ``(id(module), attribute name, id of
# its TensorImpl)``, or ``(-1, (kind, key), ...)`` for a closure cell or a global.
type _Held = tuple[int, object, int]


def _outside_tensors(
    model: Module,
) -> dict[_Held, tuple[Module | None, str | tuple[str, str]]]:
    """Every tensor the tree's modules hold as plain attributes, and the
    wrapped callable's cells and globals hold, with where it sits."""
    from lucid._tensor.tensor import Tensor

    out: dict[_Held, tuple[Module | None, str | tuple[str, str]]] = {}
    for sub in _iter_modules(_root_modules(model)):
        for name, value in vars(sub).items():
            if isinstance(value, Tensor):
                out[(id(sub), name, id(value._impl))] = (sub, name)
    fn = _wrapped_callable(model)
    for key, value in _callable_slots(fn) if fn is not None else ():
        if isinstance(value, Tensor):
            out[(-1, key, id(value._impl))] = (None, key)
    return out


@final
class ParamFingerprint(_ModuleWrapper):
    """A model's :func:`_param_fingerprint`, walked again only when it can change.

    The walk over every parameter cost 0.7–0.8 ms a call on bert_base, and
    a compiled step keyed each call on it (LCD-489).  It changes when a
    parameter is replaced.  ``.to()`` / ``.half()`` and an ``assign=True``
    load put new tensors under the ``Parameter`` objects without a
    ``setattr``, so the tree version cannot see them; they tell the
    module's watchers (``_tensors_replaced``), and this watches every module
    of the tree, a cast of one submodule included.  A write to a module's
    parameter, buffer or submodule registry moves the tree version, which is
    checked too.  A tensor given new storage by hand (``p.data = ...``) is
    not seen.

    The same news reaches its owner: an executable is fed the tensors it was
    traced with, so one traced before a replacement goes on training the old
    ones (LCD-505).  A tensor or submodule put in another's place leaves the
    fingerprint as it was, so the tree is recorded at every point the owner
    takes it — when this is built (``make_step`` takes the parameters, the
    segmented step its children), after each of its traces (:meth:`settle`)
    and after it is told of a replacement, when it takes the tree again — and
    when the version moves the tree is compared with what was recorded
    (:class:`_TreeTensors`): any difference is a replacement (LCD-560).
    Settling after the trace, not before, takes in what a forward does to its
    own state while traced — a lazy layer materialising, a statistic kept by
    assignment — which would otherwise read as a replacement on every call.
    A wrapped callable whose closure cell or global names another module, or
    another tensor where a trace read one, is a replacement too; that moves
    no version, so it is checked on every call.

    Parameters
    ----------
    model : Module
        The model whose parameters are fingerprinted.  Held weakly: the
        watcher lives in its owner, and must not keep the model alive.
    on_replaced : callable, optional
        Called after a module of the tree had its tensors replaced, once per
        module told: the owner drops what it traced.  A bound method is held
        weakly, so an owner that holds this watcher is not kept alive by it
        — with its executables — until a cycle collection.
    """

    def __init__(
        self, model: Module, on_replaced: Callable[[], object] | None = None
    ) -> None:
        self._model: weakref.ref[Module] = weakref.ref(model)
        self._value: tuple[tuple[str, str], ...] | None = None
        self._version = -1
        self._on_replaced: Callable[[], Callable[[], object] | None] = (
            weakref.WeakMethod(on_replaced)
            if isinstance(on_replaced, MethodType)
            else lambda: on_replaced
        )
        self._watch(model)
        # The tensors outside the registries that a trace was fed (LCD-606):
        # a module's plain tensor attribute (``self.mask = ...`` lives in
        # ``__dict__``), a closure cell's or a global's.  Only those a trace
        # read are recorded — weight_norm, prune and spectral_norm rewrite
        # ``module.weight`` as a plain attribute on every eager forward, and
        # recording every attribute would trace again after each.  They only
        # grow, so a record and the one compared with it name the same slots.
        self._attributes: weakref.WeakKeyDictionary[Module, tuple[str, ...]] = (
            weakref.WeakKeyDictionary()
        )
        self._callable_slots: tuple[tuple[str, str], ...] = ()
        # The tree as the owner last took it.  ``None`` after the owner was
        # told of a replacement: it took the tree again at ``_retaken_at``.
        self._settled: _TreeTensors | None = None
        # Where each tensor of the last record sits, for the key.
        self._slot_of: Mapping[int, _Slot] = {}
        self._keep(_TreeTensors(model))
        self._retaken_at = -1

    @override
    def __reduce__(
        self,
    ) -> tuple[type[ParamFingerprint], tuple[Module, Callable[[], object] | None]]:
        # A weak reference does not pickle; a copy of its owner (a
        # CompiledModule) fingerprints the copy of the model, walked afresh.
        return (ParamFingerprint, (self.model, self._on_replaced()))

    def _watch(self, model: Module) -> None:
        # The modules a wrapped callable closes over too: their ``.half()``
        # replaces the tensors its executables read.
        for sub in _iter_modules(_root_modules(model)):
            _watch_tensors(sub, self)

    @property
    @override
    def model(self) -> Module:
        """The fingerprinted model."""
        model = self._model()
        if model is None:
            raise ReferenceError("ParamFingerprint: the model has been deleted")
        return model

    def get(self) -> tuple[tuple[str, str], ...]:
        """The fingerprint, walking the parameters only after a change.

        Returns
        -------
        tuple of (dtype_str, device_str)
            What :func:`_param_fingerprint` gives for the model now.
        """
        from lucid.nn._tree_version import _tree_version

        version = _tree_version()
        value = self._value
        settled = self._settled
        if (
            value is None
            or version != self._version
            or (settled is not None and not settled.reads_the_same(self._model()))
        ):
            model = self.model
            # A submodule added since the last walk is watched from now on.
            self._watch(model)
            # Read before the walk, as ``_tree_walk`` does: a change made
            # while it runs moves the version past the one stored.
            self._version = version
            self._compare_tree(model, version)
            value = self._value = _param_fingerprint(model)
        return value

    def _record_of(self, model: Module) -> _TreeTensors:
        return _TreeTensors(model, self._attributes, self._callable_slots)

    def _keep(self, record: _TreeTensors) -> None:
        self._settled = record
        self._slot_of = record.slot_of

    def _compare_tree(self, model: Module, version: int) -> None:
        """Tell the owner of a replacement since it took the tree; record it."""
        now = self._record_of(model)
        settled = self._settled
        if settled is not None and settled.same_as(now):
            return
        # Told of a replacement, the owner took the tree again then; anything
        # that has moved the version since may be another one.  Recorded here,
        # not in the notice: ``.to()`` tells every module of the tree in turn,
        # and a walk per notice is a walk per module.
        if settled is not None or version != self._retaken_at:
            self._tensors_replaced()
        self._keep(now)

    def outside_slots(self) -> frozenset[_Held]:
        """The tensors the tree holds outside its registries, taken before a trace.

        A module's plain tensor attributes and the wrapped callable's closure
        cells and globals, each with the tensor in it now.  Hand it to
        :meth:`settle`: only a slot that held a fed tensor *before* the trace
        is the executable's state.  One the forward assigned while traced —
        ``self.last = x``, ``nonlocal last``, a global — holds the call's
        argument then, and recorded it traced again on every new input.

        Returns
        -------
        frozenset
            ``(holder, name, TensorImpl id)`` for each such slot.
        """
        return frozenset(_outside_tensors(self.model))

    def settle(
        self,
        fed: Iterable[_C_engine.TensorImpl] = (),
        before: frozenset[_Held] = frozenset(),
    ) -> None:
        """Record the tree as the owner's executables read it: it just traced.

        Call it after every trace, whether or not an executable came of it:
        the trace ran the forward, and what that did to the module's own
        state is not a replacement.

        Parameters
        ----------
        fed : iterable of TensorImpl, optional
            What the executable the trace became was fed from outside it
            (:func:`outside_feeds`); empty when none came of it.
        before : frozenset, optional
            :meth:`outside_slots` taken before the trace.  A module's plain
            tensor attribute, a closure cell or a global that held one of
            ``fed`` then, and holds it still, is recorded from now on: put
            another tensor there and the executable is a replacement's.
        """
        model = self.model
        self._learn(model, {id(impl) for impl in fed}, before)
        self._keep(self._record_of(model))

    def _learn(self, model: Module, fed: set[int], before: frozenset[_Held]) -> None:
        """Add the slots outside the registries that held one of ``fed``
        before the trace and hold it still."""
        if not fed or not before:
            return
        for held, (sub, name) in _outside_tensors(model).items():
            if held[2] not in fed or held not in before:
                continue
            if isinstance(name, tuple):
                self._callable_slots = tuple(sorted({*self._callable_slots, name}))
            elif sub is not None:
                self._attributes[sub] = tuple(
                    sorted({*self._attributes.get(sub, ()), name})
                )

    def tree_alias(
        self, args: Iterable[object], kwargs: dict[str, object]
    ) -> tuple[tuple[int, _Slot], ...]:
        """:attr:`CacheKey.tree_alias` of a call, from the tree as last
        recorded — by :meth:`get`, or by :meth:`settle` after a trace.

        Parameters
        ----------
        args : iterable
            The call's positional arguments.
        kwargs : dict
            The call's keyword arguments.

        Returns
        -------
        tuple of (int, slot)
            Each of the call's tensors that the model holds, with its slot.
        """
        return _tree_alias(leaf_tensors(args, kwargs), self._slot_of)

    def binding(
        self, args: Iterable[object], kwargs: dict[str, object]
    ) -> tuple[tuple[int, ...], tuple[tuple[int, _Slot], ...]]:
        """How a call's tensors bind to a trace's feeds, for an owner with a
        key of its own.

        Parameters
        ----------
        args : iterable
            The call's positional arguments.
        kwargs : dict
            The call's keyword arguments.

        Returns
        -------
        tuple
            :attr:`CacheKey.alias` and :attr:`CacheKey.tree_alias` of the call.
        """
        self.get()
        leaves = leaf_tensors(args, kwargs)
        return _alias_pattern(leaves), _tree_alias(leaves, self._slot_of)

    def rekey(
        self, key: CacheKey, args: Iterable[object], kwargs: dict[str, object]
    ) -> CacheKey:
        """``key`` with the slots :meth:`settle` just learnt the trace reads.

        An argument that is a tensor attribute the trace read shares that
        attribute's feed, and only the trace shows the attribute is read.
        The rest of the key stays as the call keyed it before the trace.

        Parameters
        ----------
        key : CacheKey
            The call's key, taken before the trace.
        args : iterable
            The call's positional arguments.
        kwargs : dict
            The call's keyword arguments.

        Returns
        -------
        CacheKey
            ``key`` with :attr:`CacheKey.tree_alias` read again.
        """
        return replace(key, tree_alias=self.tree_alias(args, kwargs))

    def key_of(
        self, args: Iterable[object], kwargs: dict[str, object], *, dynamic: bool
    ) -> CacheKey:
        """:func:`signature_of` for a call of the model, with this fingerprint.

        Parameters
        ----------
        args : iterable
            The call's positional arguments, as :func:`signature_of` takes
            them.
        kwargs : dict
            The call's keyword arguments.
        dynamic : bool
            Whether the batch axis is wildcarded.

        Returns
        -------
        CacheKey
            The call's key.
        """
        fingerprint = self.get()
        return signature_of(
            self.model,
            args,
            kwargs,
            dynamic=dynamic,
            param_fingerprint=fingerprint,
            slots=self._slot_of,
        )

    def invalidate(self) -> None:
        """Walk the parameters again on the next :meth:`get`."""
        self._value = None

    @override
    def _tensors_replaced(self) -> None:
        """A module of the tree had its parameters or buffers replaced."""
        from lucid.nn._tree_version import _tree_version

        self.invalidate()
        on_replaced = self._on_replaced()
        if on_replaced is not None:
            on_replaced()
        # The owner dropped what it traced and took the tree again — make_step
        # its parameters, the segmented step its children: the next ``get``
        # records it, or tells the owner again if the tree moved meanwhile.
        self._settled = None
        self._retaken_at = _tree_version()


def signature_of(
    model: Module,
    args: Iterable[object],
    kwargs: dict[str, object],
    *,
    dynamic: bool,
    param_fingerprint: tuple[tuple[str, str], ...] | None = None,
    slots: Mapping[int, _Slot] | None = None,
) -> CacheKey:
    """Compute the :class:`CacheKey` for this ``model(*args, **kwargs)`` call.

    ``dynamic=True`` opts into Phase 1.6's symbolic batch axis: the
    leading dim of every input tensor is wildcarded to ``-1`` in the
    captured :class:`TensorSig`, so a call sequence like
    ``compiled(x_bs1); compiled(x_bs32)`` shares a single cache entry.
    ``dynamic`` is itself recorded in :attr:`CacheKey.dynamic` so the
    same model can co-exist as static + dynamic.

    Parameters
    ----------
    model : Module
        Owner of the compiled graph; its :func:`id` is part of the key
        so two different model instances never share a cache slot even
        when their input signatures coincide.
    args : iterable
        Positional call arguments.  Each is summarised via
        :func:`_arg_sig` (tensor → shape/dtype/device, scalar → typed value).
    kwargs : dict
        Keyword call arguments; summarised the same way as ``args`` and
        sorted by key so insertion order doesn't affect the cache key.
    dynamic : bool
        When ``True`` wildcard the leading axis of every tensor sig
        (dynamic batching).  Recorded in :attr:`CacheKey.dynamic`.
    param_fingerprint : tuple of (dtype_str, device_str) pairs, optional
        Pre-computed parameter fingerprint (dtype × device pairs across
        every parameter, deduplicated).  Defaults to the result of
        :func:`_param_fingerprint(model)`.  Callers that walk parameters
        once per call pass it in to avoid the second walk inside this
        function.
    slots : mapping of int to slot, optional
        Where each tensor the model holds sits, by ``TensorImpl`` identity
        (:attr:`CacheKey.tree_alias`); by default its parameters and buffers.

    Returns
    -------
    CacheKey
        Hashable tuple identifying the (model, signature, dynamic)
        triple — used as the dict key in the trace cache.

    Examples
    --------
    >>> # ``CompiledModule.__call__`` uses this to pick the cache slot.
    >>> import lucid
    >>> import lucid.nn as nn
    >>> model = nn.Linear(8, 4)
    >>> key = signature_of(model, (lucid.randn(2, 8),), {}, dynamic=False)
    >>> key == signature_of(model, (lucid.randn(2, 8),), {}, dynamic=False)
    True
    >>> key == signature_of(model, (lucid.randn(5, 8),), {}, dynamic=False)
    False

    With ``dynamic=True`` the batch axis is wildcarded, so both calls
    share one slot:

    >>> a = signature_of(model, (lucid.randn(2, 8),), {}, dynamic=True)
    >>> a == signature_of(model, (lucid.randn(5, 8),), {}, dynamic=True)
    True

    See Also
    --------
    CacheKey : returned hashable type.
    TensorSig.of : per-tensor descriptor used to build ``args``.
    lucid.compile._entry.module.CompiledModule : the caller.
    """

    arg_list = tuple(args)
    leaves = leaf_tensors(arg_list, kwargs)
    arg_sigs = tuple(_arg_sig(a, dynamic_batch=dynamic) for a in arg_list)
    kwarg_sigs = tuple(
        sorted((str(k), _arg_sig(v, dynamic_batch=dynamic)) for k, v in kwargs.items())
    )
    fp = (
        param_fingerprint
        if param_fingerprint is not None
        else _param_fingerprint(model)
    )
    module_state = _module_state(model)
    return CacheKey(
        args=arg_sigs,
        kwargs=kwarg_sigs,
        training=bool(getattr(model, "training", False)),
        dynamic=bool(dynamic),
        param_fingerprint=fp,
        alias=_alias_pattern(leaves),
        tree_alias=_tree_alias(
            leaves, _tree_walk(model).slot_of if slots is None else slots
        ),
        module_state=module_state,
        grad_mode=bool(_C_engine.grad_enabled()),
    )
