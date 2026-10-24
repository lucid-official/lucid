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
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from types import MethodType
from typing import TYPE_CHECKING, final, override

from lucid.nn.module import _ModuleWrapper, _watch_tensors

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor
    from lucid.nn.module import Module

__all__ = ["TensorSig", "CacheKey", "signature_of"]


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

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.compile._core.signature import signature_of
    >>> model = nn.Linear(8, 4).to("metal")
    >>> x = lucid.randn(2, 8, device="metal")
    >>> compiled = lucid.compile(model)
    >>> key = signature_of(model, (x,), {}, dynamic=False)
    >>> key in compiled._cache               # is this signature compiled yet?
    False
    >>> _ = compiled(x)
    >>> key in compiled._cache
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
    # Per-submodule ``(training, scalar_attrs)`` snapshot across ``model.modules()``.
    module_state: tuple[object, ...] = field(default_factory=tuple)


def _closure_modules(model: object) -> list[Module]:
    """The :class:`Module` objects ``model._fn`` is bound to or closes over."""
    from lucid.nn.module import Module

    fn = getattr(model, "_fn", None)
    closure = getattr(fn, "__closure__", None) or ()
    out: list[Module] = []
    # A bound method — ``lucid.compile(model.forward)`` — runs its module's
    # tree as a closure would.
    owner = getattr(fn, "__self__", None)
    if isinstance(owner, Module):
        out.append(owner)
    for cell in closure:
        contents = getattr(cell, "cell_contents", None)
        if isinstance(contents, Module):
            out.append(contents)
    return out


def _parametrization_modules(mod: Module) -> list[Module]:
    """Return any :class:`Module` containers stored in ``mod.parametrizations``."""
    from lucid.nn.module import Module

    param_d = getattr(mod, "parametrizations", None)
    if not isinstance(param_d, dict):
        return []
    return [c for c in param_d.values() if isinstance(c, Module)]


def _root_modules(model: object) -> list[Module]:
    """``model`` if it is a module, then the modules its ``_fn`` closes over."""
    from lucid.nn.module import Module

    roots: list[Module] = [model] if isinstance(model, Module) else []
    roots.extend(_closure_modules(model))
    return roots


def _iter_modules(roots: Sequence[Module]) -> list[Module]:
    """Every module reachable from ``roots``, each once, in deterministic order."""
    seen: set[int] = set()
    out: list[Module] = []
    for root in roots:
        for sub in root.modules():
            if id(sub) not in seen:
                seen.add(id(sub))
                out.append(sub)
            for pmod in _parametrization_modules(sub):
                for psub in pmod.modules():
                    if id(psub) not in seen:
                        seen.add(id(psub))
                        out.append(psub)
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
    """Hook ids installed by ``nn.utils`` reparametrisations (traced in-graph)."""
    handles: list[object] = []
    for attr in ("_weight_norm_hooks", "_spectral_norm_hooks"):
        d = getattr(mod, attr, None)
        if isinstance(d, dict):
            handles.extend(e[0] for e in d.values() if isinstance(e, tuple) and e)
    prune_d = getattr(mod, "_prune_hooks", None)
    if isinstance(prune_d, dict):
        handles.extend(prune_d.values())
    param_d = getattr(mod, "parametrizations", None)
    if isinstance(param_d, dict):
        handles.extend(getattr(c, "_hook_handle", None) for c in param_d.values())
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

    def walked_from(self, roots: Sequence[Module]) -> bool:
        return len(roots) == len(self.roots) and all(
            ref() is root for ref, root in zip(self.roots, roots, strict=True)
        )


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
    walk = _TreeWalk(
        version=version,
        roots=tuple(weakref.ref(root) for root in roots),
        states=tuple(
            (bool(getattr(sub, "training", False)), _dict_scalar_attrs(sub))
            for sub in subs
        ),
        hooked=any(_sub_has_active_hooks(sub) for sub in subs),
    )
    if isinstance(model, Module):
        _WALKS[model] = walk
    return walk


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


def _alias_pattern(
    args: tuple[object, ...], kwargs: dict[str, object]
) -> tuple[int, ...]:
    """First-occurrence index of each tensor in :func:`leaf_tensors` order.

    Identity is the ``TensorImpl`` — the test the feed binding uses to
    map a trace's external feeds back to the call's tensors, so two keys
    differ exactly when the binding would.
    """
    seen: dict[int, int] = {}
    return tuple(
        seen.setdefault(id(t._impl), i)
        for i, t in enumerate(leaf_tensors(args, kwargs))
    )


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
    """

    __slots__ = ("_record", "_modules")

    def __init__(self, model: object) -> None:
        subs = _iter_modules(_root_modules(model))
        ids: list[object] = []
        held: list[object] = []
        for sub in subs:
            ids.append(id(sub))
            for registry in (sub._parameters, sub._buffers):
                ids.append(len(registry))
                for name, tensor in registry.items():
                    impl = None if tensor is None else tensor._impl
                    ids += (name, id(tensor), id(impl))
                    held += (tensor, impl)
        self._record = (tuple(ids), tuple(held))
        self._modules = tuple(weakref.ref(sub) for sub in subs)

    def same_as(self, now: _TreeTensors) -> bool:
        """Whether ``now`` holds the modules and tensors this record does."""
        return self._record[0] == now._record[0] and all(
            ref() is not None for ref in self._modules
        )


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
    fingerprint as it was, so after the owner traces it calls :meth:`settle`,
    and when the version moves the tree is compared with what was settled
    (:class:`_TreeTensors`): any difference is a replacement (LCD-560).
    Settling after the trace, not before, takes in what a forward does to its
    own state while traced — a lazy layer materialising, a statistic kept by
    assignment — which would otherwise read as a replacement on every call.

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
        self._model: Callable[[], Module | None] = weakref.ref(model)
        self._value: tuple[tuple[str, str], ...] | None = None
        self._version = -1
        # The tree the owner's executables were traced against; ``None`` while
        # it has none.
        self._settled: _TreeTensors | None = None
        self._on_replaced: Callable[[], Callable[[], object] | None] = (
            weakref.WeakMethod(on_replaced)
            if isinstance(on_replaced, MethodType)
            else lambda: on_replaced
        )
        self._watch(model)

    @classmethod
    def of_callable(
        cls, fn: Callable[..., object], on_replaced: Callable[[], object]
    ) -> ParamFingerprint:
        """The fingerprint of the modules ``fn`` is bound to or closes over.

        It holds the adapter that names them, which nothing else does.

        Parameters
        ----------
        fn : callable
            A forward that is not a module — ``compiled_decode_step``'s.
        on_replaced : callable
            As for the constructor: drops what was traced from ``fn``.

        Returns
        -------
        ParamFingerprint
            The watcher, with the adapter held for as long as it lives.
        """
        from lucid.compile import _CallableModule

        owner = _CallableModule(fn)
        fingerprint = cls(owner, on_replaced)
        fingerprint._model = lambda: owner
        return fingerprint

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
        if value is None or version != self._version:
            model = self.model
            # A submodule added since the last walk is watched from now on.
            self._watch(model)
            # Read before the walk, as ``_tree_walk`` does: a change made
            # while it runs moves the version past the one stored.
            self._version = version
            settled = self._settled
            if settled is not None and not settled.same_as(_TreeTensors(model)):
                self._tensors_replaced()
            value = self._value = _param_fingerprint(model)
        return value

    def settle(self) -> None:
        """Record the tree as the owner's executables read it: it just traced.

        Call it after every trace, whether or not an executable came of it:
        the trace ran the forward, and what that did to the module's own
        state is not a replacement.
        """
        self._settled = _TreeTensors(self.model)

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
        return signature_of(
            self.model, args, kwargs, dynamic=dynamic, param_fingerprint=self.get()
        )

    def invalidate(self) -> None:
        """Walk the parameters again on the next :meth:`get`."""
        self._value = None

    @override
    def _tensors_replaced(self) -> None:
        """A module of the tree had its parameters or buffers replaced."""
        self.invalidate()
        # The owner drops what it traced, so nothing is settled until it
        # traces again.
        self._settled = None
        on_replaced = self._on_replaced()
        if on_replaced is not None:
            on_replaced()


def signature_of(
    model: Module,
    args: Iterable[object],
    kwargs: dict[str, object],
    *,
    dynamic: bool,
    param_fingerprint: tuple[tuple[str, str], ...] | None = None,
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
        alias=_alias_pattern(arg_list, kwargs),
        module_state=module_state,
    )
