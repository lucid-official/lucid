"""
lucid.compile._entry.module — Phase 1.4 CompiledModule.

Wraps a regular :class:`nn.Module` so that subsequent calls with the
same input signature reuse a single :class:`MPSGraphExecutable` rather
than re-dispatching through eager.  The wrapper is not itself an
:class:`nn.Module`: it re-exposes the inner model's parameters /
state_dict / training mode, and assigned onto a parent module it
registers the inner model in the parent's tree while the attribute
keeps answering with the wrapper (``lucid.nn.module._ModuleWrapper``).

The cached executable captures the forward pass only and records no
autograd graph.  A call that needs one — grad mode on and a tensor it
reads requiring grad — therefore runs eagerly, so ``backward`` reaches
what it would reach without ``lucid.compile``; inference compiles under
``no_grad`` / ``inference_mode`` or when nothing requires grad.  Compiled
training — one executable for forward and backward — is
:func:`lucid.compile.make_step` / :func:`lucid.compile.fused_step`.

Acceptance gate (Plan §1.4):

  * 100 calls same signature → 1 compile + 99 cache hits.
  * Shape change → recompile.
  * Unsupported op → eager fallback with correct result + signature
    remembered in :class:`EagerFallbackSet` so we don't re-attempt.
"""

import time
from dataclasses import dataclass, field, replace
from typing import (
    TYPE_CHECKING,
    Callable,
    Iterator,
    Mapping,
    Protocol,
    Self,
    cast,
    final,
    override,
)


# Engine-side objects exposed via pybind11; the binding doesn't carry
# stubs, so we name the surface we actually use here.
class _TracerLike(Protocol):
    graph: object
    external_feeds: dict[int, _C_engine.TensorImpl]
    retained_values: dict[int, _C_engine.TensorImpl]

    def lookup_id(self, impl: object) -> int | None: ...

    def made_here(self, impl: object) -> bool: ...


class _ExecutableLike(Protocol):
    input_ids: list[int]
    output_ids: list[int]
    num_inputs: int


from lucid._C import engine as _C_engine

from lucid.compile._core.fallback import (
    EagerFallbackSet,
    _fallback_notice,
    _narrate,
    run_eager,
)
from lucid.compile._core.signature import (
    CacheKey,
    ParamFingerprint,
    TraceReads,
    leaf_tensors,
    model_has_active_hooks,
    tensors_have_active_hooks,
    tree_requires_grad,
)
from lucid.nn.module import Module, _ModuleWrapper
from lucid.nn.parameter import Parameter

if TYPE_CHECKING:
    from lucid._C.engine.compile import Tracer
    from lucid._tensor.tensor import Tensor
    from lucid.nn.module import Module


# Return-spec descriptors: small tags + nested children so that the
# executable's flat output list can be re-packed into the structure
# the user's forward returned.  ``("tensor", slot_index)`` is a leaf —
# replace with the executable output at ``slot_index``.  Other tags
# are container nodes whose children are themselves return-spec
# descriptors.
#
# Examples:
#   * ``return out`` (Tensor)               →  ("tensor", 0)
#   * ``return out, h_n`` (tuple)           →  ("tuple", [("tensor", 0), ("tensor", 1)])
#   * ``return {"loss": l, "logits": x}``  →  ("dict", [("loss", ("tensor", 0)),
#                                                       ("logits", ("tensor", 1))])


def _extract_return_impls(value: object, tracer: _TracerLike) -> tuple[object, list[int]]:
    """Walk ``value`` and replace every Tensor with a slot descriptor.

    Returns ``(return_spec, explicit_outputs)`` where:

    * ``return_spec`` is the same structure as ``value`` but with each
      Tensor swapped for a ``("tensor", slot_index)`` leaf;
    * ``explicit_outputs`` is the flat ordered list of trace TensorIds
      that the slot indices reference.

    A Tensor whose impl wasn't observed during the trace is left in
    place (its impl appears as an external feed) — we still allocate a
    slot for it so the executable's flat output list lines up with the
    spec.  The Python compile harness handles that case by falling
    back to eager when the tensor isn't reachable in the graph.
    """
    from lucid._dispatch import _unwrap
    from lucid._tensor.tensor import Tensor

    flat_ids: list[int] = []

    def visit(v: object) -> object:
        """Recursive walk producing the spec tree + side-effecting ``flat_ids``.

        Tensors become ``("tensor", slot)`` leaves whose ``slot`` is
        an index into ``flat_ids`` (the per-output trace id); every
        other container type (tuple / list / dict / dataclass) becomes
        a tagged node carrying its children; scalars passthrough.
        """
        if isinstance(v, Tensor):
            impl = _unwrap(v)
            tid = tracer.lookup_id(impl)
            if tid is None and tracer.made_here(impl):
                # A constant the trace made and folded: eager answers a new
                # tensor on every call, and so does the compiled call.
                return ("constant", v)
            if tid is None:
                # Tensor that flows out untouched (external feed
                # returned as-is, or a host-precomputed factory the
                # builder skips).  Mark with a sentinel so the run
                # path knows to passthrough.
                return ("passthrough", v)
            slot = len(flat_ids)
            flat_ids.append(int(tid))
            return ("tensor", slot)
        if isinstance(v, tuple):
            return ("tuple", [visit(x) for x in v])
        if isinstance(v, list):
            return ("list", [visit(x) for x in v])
        if isinstance(v, Mapping):
            return ("dict", [(k, visit(x)) for k, x in v.items()])
        # Dataclass or BaseModelOutput-style record: walk its
        # __dict__-equivalent in declaration order.  We require
        # ``__dataclass_fields__`` to avoid blindly mutating arbitrary
        # objects.
        fields = getattr(v, "__dataclass_fields__", None)
        if fields is not None:
            kvs = []
            for fname in fields:
                kvs.append((fname, visit(getattr(v, fname))))
            return ("dataclass", type(v), kvs)
        # Anything else (None, scalar, str, …) passes through as-is.
        return ("scalar", v)

    spec = visit(value)
    return spec, flat_ids


def _repack_outputs(spec: object, outs_wrapped: list[object]) -> object:
    """Inverse of :func:`_extract_return_impls`.

    ``outs_wrapped`` is the flat list of wrapped Tensors returned by
    :func:`_C_engine.compile.run_executable` followed by :func:`_wrap`.
    Walks ``spec`` and substitutes each ``("tensor", slot)`` leaf with
    ``outs_wrapped[slot]``.  Reassembles tuples / lists / dicts /
    dataclasses in the original order.
    """
    tag = spec[0]
    if tag == "tensor":
        return outs_wrapped[spec[1]]
    if tag == "passthrough":
        return spec[1]
    if tag == "constant":
        from lucid._tensor.tensor import Tensor

        constant = spec[1]
        return constant.clone() if isinstance(constant, Tensor) else constant
    if tag == "scalar":
        return spec[1]
    if tag == "tuple":
        return tuple(_repack_outputs(child, outs_wrapped) for child in spec[1])
    if tag == "list":
        return [_repack_outputs(child, outs_wrapped) for child in spec[1]]
    if tag == "dict":
        return {k: _repack_outputs(child, outs_wrapped) for (k, child) in spec[1]}
    if tag == "dataclass":
        cls = spec[1]
        kvs = spec[2]
        kwargs = {k: _repack_outputs(child, outs_wrapped) for (k, child) in kvs}
        return cls(**kwargs)
    raise RuntimeError(f"_repack_outputs: unknown spec tag {tag!r}")


#: What a call that needs gradients is told, once per process.
_AUTOGRAD_FALLBACK = (
    "the call needs gradients (grad mode is on and a tensor it reads requires "
    "grad), and a compiled forward records no autograd graph. Wrap inference in "
    "lucid.no_grad() (or lucid.inference_mode()) to use the compiled path; use "
    "lucid.compile.make_step or lucid.compile.fused_step for compiled training"
)


__all__ = ["CompiledModule"]


@final
@dataclass(slots=True)
class _CacheEntry:
    """One executable + the I/O plan needed to invoke it.

    The compiled MPSGraphExecutable is shape-fixed (placeholders are
    sized from the trace), so we record:
      * the ordered list of feed :type:`TensorId`s that the executable
        expects (``exe.input_ids``);
      * a mapping from those ids back to (a) external-feed Python
        objects pinned in the trace, or (b) the slot in ``args`` /
        ``kwargs`` that supplies the feed at call time.
    """

    exe: _ExecutableLike  # _C_engine.compile.PyCompiledExecutable
    external_feeds: dict[int, _C_engine.TensorImpl]  # TensorId -> TensorImpl
    # For each ``exe.input_ids[i]`` slot: an index into the call's tensors
    # in :func:`leaf_tensors` order — positional, keyword and nested alike
    # — OR ``None`` meaning "use the pinned external_feeds entry"
    # (parameters, constants).
    input_source: tuple[int | None, ...]
    # Structure descriptor used to re-pack the executable's flat output
    # list back into the shape the user's forward returned.  Each entry
    # is the path (sequence of keys / indices) to insert that output at
    # within a nested Python structure of tuples/lists/dicts.  When
    # ``return_spec`` is None, the executable returns a single Tensor.
    # See :func:`_extract_return_impls` for the format.
    return_spec: object = None
    # JSON-serialised TraceGraph used to build this executable —
    # cheap to compute at compile time and useful for ``graph_dump``
    # observability after the fact.  Kept as a string (not as a live
    # TraceGraph) so the entry can outlive the original C++ object.
    graph_json: str = ""
    n_hits: int = 0
    compile_ms: float = 0.0
    last_run_ms: float = 0.0
    # Tensors from outside the trace that the forward writes in place (a
    # buffer's EMA, a counter).  Their new values trail the returned ones;
    # each run copies them back.  See ``_core/buffer_writes``.
    written: list[object] = field(default_factory=list)
    # What the executable reads besides the call's tensors — parameters,
    # buffers, a closure's or a global's tensors — and whether the call makes
    # a grad leaf itself.  Under grad mode one requiring grad makes it eager.
    reads: TraceReads = TraceReads()

    def reads_grad(self) -> bool:
        """Whether, besides its arguments, the call reads or makes a tensor
        that requires grad — read now, as freezing one moves no version."""
        return self.reads.require_grad()


class CompiledModule[**P, R](_ModuleWrapper):
    # ── Class-level annotations ──────────────────────────────────
    # ``CompiledModule[**P, R]`` (PEP 695) captures the wrapped model's /
    # callable's call signature so ``compiled(*args)`` keeps the original
    # ``forward`` parameter list + return type at the type level instead of
    # collapsing to ``object``.  ``P``/``R`` are bound by ``lucid.compile``
    # from the ``Callable[P, R]`` it receives.
    # All instance attrs are written via ``object.__setattr__`` to
    # bypass ``Module.__setattr__`` (we are NOT an nn.Module
    # subclass — we just delegate to one).  Declaring them at the
    # class level gives mypy a real type for each, so the delegation
    # methods below (parameters / state_dict / to / train / …) stay
    # checkable instead of collapsing to ``object``.
    _model: Module
    _dynamic: bool
    _symbolic: bool
    _symbolic_resolved: bool | None
    _cache: dict[CacheKey, _CacheEntry]
    _eager_only: EagerFallbackSet
    _call_counter: dict[CacheKey, int]
    _param_fp: ParamFingerprint
    _step_callables: dict[int, Callable[..., Tensor]]

    r"""User-facing wrapper returned by :func:`lucid.compile`.

    Wraps an :class:`nn.Module` (or any callable, via the internal
    :class:`_CallableModule` adapter) so the model's ``__call__`` is
    routed through a per-signature cache of compiled
    :class:`MPSGraphExecutable` objects.  Behaves like the underlying
    model for every non-call attribute (``parameters`` / ``state_dict``
    / ``train`` / ``eval`` / ``to`` / submodule attribute access), so
    optimizers, schedulers and checkpointing pipelines see the
    compiled module as a drop-in replacement.

    Lifecycle
    ---------
    On the first call with a given input signature
    ``(shape × dtype × device × training × param_fingerprint)`` the
    wrapper:

    1. Traces ``model(*args)`` under :func:`no_grad` while the
       :class:`Tracer` records every op into a :class:`TraceGraph`.
    2. Lowers the graph to a single :class:`MPSGraphExecutable` via
       ``compile_or_cached``.
    3. Builds an *input plan* mapping each placeholder in
       ``exe.input_ids`` to either a positional :class:`Tensor` slot
       from the call site or a pinned external feed (parameter /
       constant).
    4. Records a ``return_spec`` so the executable's flat output list
       can be re-packed into the user's original return structure
       (tuple, dataclass, dict, ...).

    On subsequent calls with the same signature step (1)–(3) are
    skipped and the cached executable runs directly.  Cache lookup
    cost is one ``__hash__`` + ``__eq__`` on the :class:`CacheKey`
    plus an integer-slot resolution per feed — typically O(μs).

    Two cache layers cooperate here:

    * **Python-side ``self._cache``** — unbounded ``dict`` keyed by
      :class:`CacheKey`, holds the :class:`_CacheEntry` + return spec
      + timing.  Dropped on :meth:`train` / :meth:`eval` / :meth:`to`
      / :meth:`load_state_dict` / :meth:`clear_cache`.
    * **C++ ``ExecutableCache::session()``** — process-global LRU
      bounded by ``LUCID_COMPILE_MAX_CACHE`` (default 32).  Holds
      one ``CompiledExecutable`` per *trace structure* so two
      :class:`CompiledModule` instances tracing the same op DAG
      share the underlying Metal kernels.  Cache key includes per-op
      attributes (since 2026-05-24), so ``dropout(p=0)`` and
      ``dropout(p=0.5)`` no longer collide.

    Attributes
    ----------
    model : nn.Module
        The wrapped model (read via the ``.model`` property).
    dynamic : bool
        ``True`` if the wrapper was constructed with the symbolic
        batch axis opt-in (Phase 1.6).  Forced to ``False`` until
        the MPSGraph SDK stabilises dynamic-shape lowering.
    training : bool
        Mirrors ``self._model.training``; flipping invalidates the
        cache (BN / Dropout codepaths diverge between modes).

    Notes
    -----
    * **A call that needs gradients runs eagerly.**  The executable is a
      forward pass with no autograd graph.  Under grad mode, a call that
      reads a tensor requiring grad — an argument, a parameter or buffer,
      a tensor a closure holds — runs the model eagerly instead, so its
      result requires grad and ``backward`` reaches what it would reach
      uncompiled, in eval mode too.  A one-time ``UserWarning`` says so.
      Wrap inference in :class:`lucid.no_grad` (or
      :func:`lucid.inference_mode`) to use the compiled path; train with
      :func:`make_step` / :func:`fused_step` (or :meth:`step`), which
      compile the backward too.  A call made under ``no_grad`` keeps its
      own cache entry, so alternating the two retraces nothing.
    * **Metal-only.**  Compile is meaningful only when both the model
      and its inputs live on the ``metal`` device.  On CPU every call
      silently routes through eager — the cache stays empty and
      ``eager_only`` grows.
    * **kwargs compile too.**  Keyword-passed tensors are traced like
      positional ones; each executable feed is bound back to either an
      ``args`` index or a ``kwargs`` name at call time (see
      ``_CacheEntry.input_source``).  Non-tensor kwargs are baked into
      the trace structure and captured by the :class:`CacheKey`, so a
      different scalar kwarg re-keys (and recompiles) correctly.
    * **Random draws are drawn again on every call.**  A draw from the
      default generator inside the traced function — ``randn`` noise, a
      ``randint`` timestep, a training-mode dropout mask — is a feed the
      run path makes afresh through the same eager op, so under the same
      seed a compiled call draws exactly what eager draws
      (:file:`lucid/_C/compile/RngFeeds.h`).  A draw from a generator the
      caller passes explicitly runs eager.

    Examples
    --------
    Inference path::

        cm = lucid.compile(model.to('metal').eval())
        with lucid.no_grad():
            for batch in loader:
                y = cm(batch.to('metal'))

    A plain training loop works unchanged — it needs gradients, so each
    call runs eagerly (correct, not faster)::

        cm = lucid.compile(model.to('metal'))
        opt = lucid.optim.Adam(cm.parameters(), lr=1e-3)
        for batch, target in loader:
            opt.zero_grad()
            loss = F.cross_entropy(cm(batch), target)
            loss.backward()
            opt.step()

    Fully fused (forward + backward + update in one executable)::

        cm = lucid.compile(model.to('metal'))
        opt = lucid.optim.Adam(cm.parameters(), lr=1e-3)
        for batch, target in loader:
            loss = cm.step(batch, target, loss_fn=F.cross_entropy)
            loss.backward()
            opt.step()

    See Also
    --------
    :func:`lucid.compile` : the user-facing constructor.
    :func:`fused_step` : single-executable fwd+bwd+update.
    :func:`compile_optimizer` : compile only the update step.
    """

    def __init__(self, model: Module, *, dynamic: bool = False) -> None:
        """Wrap ``model`` for cached graph-capture execution.

        Parameters
        ----------
        model : nn.Module or _CallableModule
            The compute unit to compile.  Stored unwrapped so
            attribute access falls through to it.
        dynamic : bool, optional
            Declare that this module will be called with **varying batch
            sizes**.  Default ``False``.  Variable batch is always handled by
            **per-shape static caching** — each distinct input shape compiles its
            own executable, cached and reused — which works for every model and
            never recompiles for a shape already seen.  A *single* executable
            shared across batch sizes (a symbolic batch axis) is **not** offered
            automatically: Apple's MPSGraph aborts (uncatchably) the moment a
            graph materialises a constant carrying the symbolic axis — scalar
            arithmetic (``x * 0.5``), ``reshape`` / ``flatten``, reductions,
            convolutions and attention all trip it, i.e. essentially every real
            model.  The narrow class that *is* safe (pure ``linear`` /
            activation / ``softmax`` / ``layer_norm`` / ``embedding`` graphs) can
            opt in to the experimental symbolic path with the
            ``LUCID_COMPILE_DYNAMIC=1`` environment variable; without it
            ``dynamic=True`` is exactly the safe per-shape static behaviour.
        """
        # CRITICAL: bypass nn.Module's __setattr__ tracking so the
        # inner model is NOT registered as a submodule (otherwise
        # parameters / state_dict double-count).
        import os as _os

        # ``dynamic=True`` ATTEMPTS the symbolic batch axis (one shared
        # executable).  Safety is enforced on the first trace by a gate
        # (``graph_symbolic_safe``): graphs that bake the batch into a constant
        # MPSGraph can't infer — broadcast / batch-axis join / batch-shaped
        # factory — fall back to robust per-shape static caching instead.  So
        # the attempt never crashes a real model; it just shares one executable
        # where provably safe and recompiles per shape otherwise.  Escape hatch:
        # ``LUCID_COMPILE_DYNAMIC=0`` forces pure static (no symbolic attempt).
        _symbolic = bool(dynamic) and _os.environ.get("LUCID_COMPILE_DYNAMIC", "1") != "0"
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_dynamic", bool(dynamic))  # requested (intent / repr)
        object.__setattr__(self, "_symbolic", _symbolic)  # attempt symbolic?
        # Resolved on the first trace by the gate + actual lowering: ``True`` →
        # keep the symbolic executable, ``False`` → per-shape static.  ``None``
        # until then (and forever for non-dynamic modules).
        object.__setattr__(self, "_symbolic_resolved", None)
        object.__setattr__(self, "_cache", {})
        object.__setattr__(self, "_eager_only", EagerFallbackSet())
        object.__setattr__(self, "_call_counter", {})
        # The key's parameter fingerprint, walked again only when a parameter
        # can have changed (and on clear_cache).  Without it signature_of
        # would walk model.parameters() on every __call__, which dominates the
        # Python overhead for deep models.  Executables pin the tensors they
        # were traced with, so it also tells _tensors_replaced when a ``.to()``
        # or an ``assign=True`` load replaces them, on any module of the tree
        # — through a parent module, or on the model itself — and a cached
        # call would go on answering with the old weights.
        object.__setattr__(
            self,
            "_param_fp",
            ParamFingerprint(model, on_replaced=self._tensors_replaced),
        )
        # Lazy training-step callables, keyed by ``id(loss_fn)``.  The
        # first ``cm.step(..., loss_fn=fn)`` call creates a make_step
        # callable and stashes it here; subsequent calls with the same
        # loss_fn re-use the cached fwd+bwd executable inside.  We
        # intentionally key by object identity, not value equality —
        # a fresh closure each call would defeat the cache, which the
        # docstring warns about.
        object.__setattr__(self, "_step_callables", {})

    @override
    def _tensors_replaced(self) -> None:
        """Recompile on the next call: the model's tensors were replaced.

        Every executable read the old ones, so all are dropped.  A signature
        that runs eagerly stays eager — what kept it from compiling is in its
        graph, not in which tensors fill it — and a forward that replaces its
        own state on every eager call would otherwise be traced again on
        every call.
        """
        self._cache.clear()
        self._step_callables.clear()

    # ── Delegation surface ────────────────────────────────────────

    @property
    @override
    def model(self) -> Module:
        """The wrapped :class:`nn.Module` (or duck-typed callable wrapper)."""

        return self._model

    @property
    def dynamic(self) -> bool:
        """Whether ``dynamic=True`` (variable batch) was requested.

        This is the *requested* flag.  By default it selects robust per-shape
        static caching; a single symbolic-batch executable is used only when the
        experimental ``LUCID_COMPILE_DYNAMIC=1`` opt-in is also set.
        """

        return self._dynamic

    @property
    def training(self) -> bool:
        """Mirrors the inner model's ``training`` flag (cache-invalidating)."""
        return self._model.training

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """Delegate to ``model.parameters`` so optimizers see the live tensors."""
        return self._model.parameters(recurse=recurse)

    def named_parameters(self, recurse: bool = True) -> Iterator[tuple[str, Parameter]]:
        """Delegate to ``model.named_parameters`` (no recompile, read-only)."""
        return self._model.named_parameters(recurse=recurse)

    def buffers(self, recurse: bool = True) -> Iterator[Tensor]:
        """Delegate to ``model.buffers``; buffers are pinned trace constants."""
        return self._model.buffers(recurse=recurse)

    def state_dict(
        self,
        destination: dict[str, Tensor] | None = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> dict[str, Tensor]:
        """Delegate to ``model.state_dict`` — the cache holds no separate state."""
        return self._model.state_dict(destination, prefix, keep_vars)

    def load_state_dict(
        self, state_dict: dict[str, Tensor], strict: bool = True, assign: bool = False
    ) -> object:
        """Forward to ``model.load_state_dict`` and **drop the cache**.

        Loading new weights doesn't change the graph topology, but
        captured tensor *identities* may shift (e.g. when the loader
        replaces parameter storage in-place); clearing the cache is
        the safe default so the next call traces against the new
        state.

        Parameters
        ----------
        state_dict : dict[str, Tensor]
            Mapping from parameter / buffer name to weight tensor —
            passed straight through to the wrapped model's
            ``load_state_dict``.
        strict : bool, optional, default=True
            When ``True``, mismatched keys raise; when ``False``,
            missing / unexpected entries are silently ignored
            (matches the underlying module's contract).
        assign : bool, optional, default=False
            When ``True``, replace parameter storage by reference
            instead of copying into the existing tensors.  Useful for
            checkpoint loading where the new tensors are already on
            the correct device.

        Returns
        -------
        object
            The wrapped model's ``load_state_dict`` return value
            (typically a ``NamedTuple`` of missing / unexpected
            keys).
        """
        # Loading new weights doesn't invalidate the graph itself —
        # only its captured constants would — but the safe default is
        # to drop the cache so the user never sees stale graph state.
        result = self._model.load_state_dict(state_dict, strict, assign)
        self.clear_cache()
        return result

    def train(self, mode: bool = True) -> Self:
        """Flip training mode; **clears the cache** since BN / Dropout diverge."""
        # train ↔ eval flips force different BN / Dropout codepaths
        # → recompile.  Clearing on every flip is the safe default.
        self._model.train(mode)
        self.clear_cache()
        return self

    def eval(self) -> Self:
        """Shorthand for ``self.train(False)``; also clears the cache."""
        return self.train(False)

    def to(self, *args: object, **kwargs: object) -> Self:
        """Forward to ``model.to`` and **drop the cache**.

        Placeholders inside the cached MPSGraph executables are
        device-locked, so any device or dtype move invalidates every
        entry.
        """
        # Device / dtype move ⇒ every cached executable is invalid
        # (placeholders are device-locked).  Drop the cache.
        self._model.to(*args, **kwargs)
        self.clear_cache()
        return self

    def __getattr__(self, name: str) -> object:
        """Forward unknown attribute lookups to the wrapped model.

        Lets ``compiled.fc1`` reach a regular submodule and
        ``compiled.named_modules()`` return the inner model's tree.
        The construction-time guard on ``_model`` prevents infinite
        recursion before ``__init__`` finishes.
        """
        # Any attribute we don't override is forwarded to the inner
        # model (so e.g. ``compiled.fc1`` reaches a regular submodule).
        # Guard against recursion during construction (``_model`` not
        # yet set).
        if name == "_model":
            raise AttributeError(name)
        inner = object.__getattribute__(self, "_model")
        return getattr(inner, name)

    # ── Cache introspection ──────────────────────────────────────

    def cache_info(self) -> dict[str, object]:
        """Snapshot the cache state for tests + telemetry.

        Examples
        --------
        >>> import lucid.nn as nn
        >>> model = nn.Linear(8, 4).to("metal")
        >>> compiled = lucid.compile(model)
        >>> x1 = lucid.randn(2, 8, device="metal")
        >>> x2 = lucid.randn(5, 8, device="metal")
        >>> with lucid.no_grad():
        ...     _ = compiled(x1); _ = compiled(x2)
        >>> info = compiled.cache_info()
        >>> info["entries"]                 # one slot per unique signature
        2

        See Also
        --------
        timing : per-signature wall-time breakdown.
        clear_cache : drop every cached entry.

        Returns
        -------
        dict
            Four fields:

            * ``entries`` (int) — number of cached executables.
            * ``keys`` (tuple[CacheKey]) — every signature that
              currently has a compiled executable.
            * ``eager_only`` (tuple[CacheKey]) — signatures that
              failed to compile and were blacklisted; future calls
              with these signatures route straight to eager.  A call
              that needed gradients is here too, under its key with
              ``grad=True``.
            * ``n_calls`` (dict[CacheKey, int]) — per-signature call
              counter, useful for spotting hot signatures.
        """

        return {
            "entries": len(self._cache),
            "keys": tuple(self._cache.keys()),
            "eager_only": self._eager_only.snapshot(),
            "n_calls": dict(self._call_counter),
        }

    def timing(self) -> list[dict[str, object]]:
        """Per-signature compile + run cost breakdown.

        Returns
        -------
        list[dict]
            One entry per cached signature with fields ``key``
            (CacheKey), ``compile_ms`` (float — first-call compile
            cost), ``last_run_ms`` (float — most recent execution
            time), and ``n_hits`` (int — total runs of this entry).
            Useful to verify the cache amortises compile cost over
            subsequent calls.
        """

        return [
            {
                "key": key,
                "compile_ms": entry.compile_ms,
                "last_run_ms": entry.last_run_ms,
                "n_hits": entry.n_hits,
            }
            for key, entry in self._cache.items()
        ]

    def clear_cache(self) -> None:
        """Drop every cached executable + retry blacklist + call counter.

        Invalidates the param-fingerprint cache and the lazy training
        step callables too, so the next call re-fingerprints the
        model.  Called automatically by :meth:`train` / :meth:`eval`
        / :meth:`to` / :meth:`load_state_dict` — direct user calls
        are useful only when forcing a recompile after manual
        parameter buffer surgery.
        """

        self._cache.clear()
        self._eager_only.clear()
        self._call_counter.clear()
        # Invalidate the param fingerprint cache so the next call
        # recomputes it (param dtype/device may have changed).
        self._param_fp.invalidate()
        # Drop the lazy training-step callables too — their internal
        # caches reference parameter impls that may have moved.
        self._step_callables.clear()

    # ── Training-step compile surface ────────────────────────────

    def step(
        self,
        *args: object,
        loss_fn: Callable[..., Tensor] | None,
    ) -> object:
        """Run one compiled training step.

        Equivalent to::

            step_fn = make_step(self.model, loss_fn)
            loss = step_fn(*args)

        but the underlying ``make_step`` callable is cached on this
        :class:`CompiledModule` so repeated training-step calls with
        the same loss function share the fwd+bwd executable.  The
        returned scalar loss tensor carries a ``grad_fn`` — calling
        ``loss.backward()`` populates ``.grad`` on every parameter
        from the cached gradient outputs.

        Parameters
        ----------
        *args : Tensor
            ``(model_input, *extra_loss_inputs)`` — the same positional
            tuple ``make_step`` expects.  Typically ``(x, target)``.
        loss_fn : callable
            ``loss_fn(model_output, *extra_inputs) -> Tensor`` scalar
            loss.  Keyed by object identity for caching; pass the same
            function object across iterations to hit the cache.

        Returns
        -------
        Tensor
            Scalar loss with a working ``grad_fn``.

        Examples
        --------
        >>> import lucid.nn as nn
        >>> import lucid.nn.functional as F
        >>> cm = lucid.compile(nn.Linear(8, 3).to("metal"))
        >>> opt = lucid.optim.SGD(cm.parameters(), lr=0.1)
        >>> loader = [
        ...     (lucid.randn(4, 8, device="metal"), lucid.randint(0, 3, (4,), device="metal"))
        ...     for _ in range(2)
        ... ]
        >>> for x, t in loader:
        ...     opt.zero_grad()
        ...     loss = cm.step(x, t, loss_fn=F.cross_entropy)
        ...     loss.backward()
        ...     opt.step()
        """

        if loss_fn is None:
            raise TypeError(
                "CompiledModule.step(): loss_fn keyword argument is "
                "required.  Pass the scalar-returning callable that "
                "produces the training loss, e.g. "
                "``cm.step(x, target, loss_fn=F.cross_entropy)``."
            )

        key = id(loss_fn)
        step_callable = self._step_callables.get(key)
        if step_callable is None:
            from lucid.compile._entry.step import make_step

            step_callable = make_step(self._model, loss_fn, dynamic=self._dynamic)
            self._step_callables[key] = step_callable

        return step_callable(*args)

    def graph_dump(self, key: CacheKey | None = None) -> str:
        """Return a JSON view of the captured trace(s).

        Without ``key``: a JSON list of ``{sig, graph}`` pairs covering
        every cached signature.  With ``key``: just the graph string
        for that one signature.

        Useful for offline debugging — pipe the result through
        :file:`tools/inspect_trace.py` (Phase 1.6 deliverable) or
        diff two consecutive compiles to see what changed.

        Parameters
        ----------
        key : CacheKey, optional
            When supplied, dump only the trace for that signature.
            When ``None`` (default), dump every cached signature in
            one combined JSON array.

        Returns
        -------
        str
            JSON-encoded trace dump.  Empty string when ``key`` was
            provided but no entry matches.

        See Also
        --------
        cache_info : structural snapshot of every cached signature.
        lucid.compile._debug.trace_dump.trace_to_json : the underlying
            serialiser called per cache entry.
        """

        if key is not None:
            entry = self._cache.get(key)
            return entry.graph_json if entry is not None else ""

        import json as _json

        items = [
            {"sig_hash": hash(k), "graph": _json.loads(e.graph_json)}
            for k, e in self._cache.items()
        ]
        return _json.dumps(items, indent=2)

    # ── Call surface ─────────────────────────────────────────────

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Route the call through the per-signature executable cache.

        On cache hit: bind the input tensors to the cached
        executable's feed slots, run, and re-pack the flat output
        list into the user's return structure (Tensor / tuple / dict
        / dataclass).  On miss: trace, compile, store, run — and on
        compile failure mark the signature ``eager_only`` so future
        calls skip the recompile attempt.  Under grad mode, a call
        that reads a tensor requiring grad runs eagerly, so its result
        is differentiable (see the class Notes).

        Typing
        ------
        ``CompiledModule[**P, R]`` carries the wrapped callable's
        signature, so this returns ``R`` (the model's ``forward``
        return type) instead of ``object``.  The runtime is
        signature-agnostic; the static surface mirrors what
        :func:`lucid.compile` captured from its ``Callable[P, R]``
        argument.

        Parameters
        ----------
        *args, **kwargs
            Whatever the wrapped model's ``forward`` accepts.  Only
            positional :class:`Tensor` arguments become executable
            feeds; non-tensor positionals (and any keyword arguments)
            force the call to eager (today's compile pipeline doesn't
            bind kwargs by name).

        Returns
        -------
        R
            Same structure the underlying ``model(*args, **kwargs)``
            produces — Tensor, tuple, dataclass, etc.
        """
        # ``P.args``/``P.kwargs`` may only be forwarded, not inspected
        # as concrete tuple/dict (PEP 612), so cast at the boundary into
        # the signature-agnostic dispatcher and re-tag the result as R.
        return cast(
            R,
            self._dispatch_call(
                cast(tuple[object, ...], args),
                cast(dict[str, object], kwargs),
            ),
        )

    def _dispatch_call(
        self,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> object:
        """Signature-agnostic cache lookup + execute (body of :meth:`__call__`)."""
        from lucid.compile._core.trace_context import recording_active

        if recording_active():
            # Called while someone else's trace is recording (LCD-626):
            # a cache hit would run this executable directly, handing the
            # outer trace a tensor no recorded op produced — pinned at
            # this call's value forever.  A miss would open a second
            # Tracer, which detaches the outer one on exit.  Neither is
            # safe here, so run the wrapped model inline: its ops dispatch
            # through the ALREADY-installed (outer) tracer, exactly as if
            # the caller had written them out instead of nesting a second
            # ``lucid.compile``.
            return run_eager(self._model, args, kwargs)

        # Hashable signature.  If the signature itself is un-hashable
        # (e.g. an exotic input that we can't fingerprint), fall back
        # to eager without poisoning the cache.
        #
        # Perf: the param fingerprint is cached (``ParamFingerprint``) so we
        # don't walk ``model.parameters()`` on every call.
        try:
            # A symbolic (shared) executable wildcards the batch axis; static
            # keys on the concrete shape.  ``_symbolic_resolved`` is the
            # per-instance decision (None until the first trace → tentatively
            # ``_symbolic``).
            eff = self._symbolic if self._symbolic_resolved is None else self._symbolic_resolved
            key = self._param_fp.key_of(args, kwargs, dynamic=eff)
        except (TypeError, AttributeError):  # fmt: skip
            return run_eager(self._model, args, kwargs)

        # An eager call's result is differentiable already: the gradient
        # check, which can walk the tree, is for calls that would compile.
        leaves = leaf_tensors(args, kwargs)
        if (
            key in self._eager_only
            or model_has_active_hooks(self._model)
            or tensors_have_active_hooks(leaves)
        ):
            self._count(key)
            return run_eager(self._model, args, kwargs)

        grad_mode = _C_engine.grad_enabled()
        entry = self._cache.get(key)
        if grad_mode and self._needs_gradients(leaves, entry):
            return self._run_differentiable(key, args, kwargs)
        if entry is None:
            key, entry = self._compile_and_cache(key, eff, args, kwargs)
            # A tensor no module holds — a closure's, a global, one a factory
            # makes — is seen only once the trace has read it.  The executable
            # stays cached for the calls made under ``no_grad``.
            if entry is not None and grad_mode and entry.reads_grad():
                return self._run_differentiable(key, args, kwargs)
        self._count(key)
        return (
            run_eager(self._model, args, kwargs)
            if entry is None
            else self._execute(key, entry, args, kwargs)
        )

    def _compile_and_cache(
        self,
        key: CacheKey,
        symbolic: bool,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> tuple[CacheKey, _CacheEntry | None]:
        """Trace and compile a signature not cached yet; ``None`` if it cannot be.

        Returns the key the entry is cached under: the first trace may have
        downgraded symbolic → static (the gate rejected the graph, or the
        symbolic lowering failed), and the wildcarded key is then wrong.
        A signature that does not compile is remembered as eager-only.
        """
        from lucid.compile._core.trace_context import compile_attempt

        with compile_attempt():
            before = self._param_fp.outside_slots()
            entry = self._compile_for(args, kwargs)
            self._param_fp.settle(() if entry is None else entry.reads.fed, before)
        resolved = self._symbolic_resolved
        if resolved is not None and resolved != symbolic:
            key = self._param_fp.key_of(args, kwargs, dynamic=resolved)
        # An argument that is a tensor attribute the trace read shares its feed.
        key = self._param_fp.rekey(key, args, kwargs)
        if entry is None:
            self._eager_only.add(key)
        else:
            self._cache[key] = entry
        return key, entry

    def _count(self, key: CacheKey) -> None:
        """Count one call that ran as ``key``."""
        self._call_counter[key] = self._call_counter.get(key, 0) + 1

    def _needs_gradients(self, leaves: list[Tensor], entry: _CacheEntry | None) -> bool:
        """Whether, grad mode on, the eager call would record an autograd graph.

        It does when a tensor it reads or makes requires grad: an argument,
        or one the executable is fed or builds besides them.  Before the
        first trace those are not known yet, and the model tree's parameters
        and buffers stand in for them — a call that trains its model is then
        never traced; a closure tensor or a factory's is found once the trace
        has read it.
        """
        if any(t.requires_grad for t in leaves):
            return True
        if entry is not None:
            return entry.reads_grad()
        return tree_requires_grad(self._model)

    def _run_differentiable(
        self, key: CacheKey, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> object:
        """Run a call that needs gradients eagerly, under a key of its own.

        A compiled forward records no autograd graph, so its result would
        not require grad and ``backward`` would stop at it.  The verdict is
        recorded as eager-only under ``key`` with ``grad=True``: the same
        call under ``no_grad`` keeps its own key and its executable, so a
        model alternating between the two is neither traced again nor sent
        eager when it need not be.
        """
        grad_key = replace(key, grad=True)
        self._count(grad_key)
        if grad_key not in self._eager_only:
            self._eager_only.add(grad_key)
            _fallback_notice("autograd", _AUTOGRAD_FALLBACK)
        return run_eager(self._model, args, kwargs)

    def _execute(
        self,
        key: CacheKey,
        entry: _CacheEntry,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> object:
        """Run ``entry``; on its first run, answer eagerly if MPSGraph fails.

        MPSGraph resolves some kernels only on the first run and has gaps
        there that compiling does not reveal — an int64 ``x * x`` is
        simplified into an int64 ``square`` it has no kernel for.  A first
        run that fails that way is the same verdict as a compile that
        declines: nothing was written yet, so the call is answered eagerly
        and the signature remembered.
        """
        if entry.n_hits:
            return self._run(entry, args, kwargs)
        try:
            return self._run(entry, args, kwargs)
        except RuntimeError as why:
            if "MPSGraph failed" not in str(why):
                raise
            _fallback_notice("first-run", str(why))
        self._cache.pop(key, None)
        self._eager_only.add(key)
        return run_eager(self._model, args, kwargs)

    # ── Internals ────────────────────────────────────────────────

    def _want_symbolic(
        self,
        graph: object,
        args: tuple[object, ...],
        tracer: _TracerLike | None = None,
        kwargs: dict[str, object] | None = None,
    ) -> bool:
        """Whether to attempt a symbolic-batch lowering for this traced graph.

        Returns the cached ``_symbolic_resolved`` once resolved; on the first
        trace, ``True`` only if ``dynamic=True`` was requested AND the safety
        gate (:func:`graph_symbolic_safe`) clears the graph of batch-baking ops
        (broadcast / batch-axis join / batch-shaped constant).  Otherwise the
        compile uses per-shape static caching.
        """
        if self._symbolic_resolved is not None:
            return bool(self._symbolic_resolved)
        if not self._symbolic:
            return False
        from lucid._tensor.tensor import Tensor
        from lucid.compile._core.symbolic_gate import graph_symbolic_safe

        first = args[0] if args else None
        if not isinstance(first, Tensor) or not first.shape:
            return False
        # The tensors whose leading axis is the batch, by trace id, so the
        # gate's axis rules follow the batch rather than every weight.
        batch_ids: set[int] | None = None
        lookup = getattr(tracer, "lookup_id", None)
        if lookup is not None:
            batch_ids = set()
            for a in args:
                if isinstance(a, Tensor) and a.shape and a.shape[0] == first.shape[0]:
                    tid = lookup(a._impl)
                    if tid is not None:
                        batch_ids.add(int(tid))
        constants = self._built_in_call(tracer, args, kwargs or {})
        return graph_symbolic_safe(graph, int(first.shape[0]), batch_ids, constants)

    def _built_in_call(
        self,
        tracer: _TracerLike | None,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> list[tuple[int, ...]]:
        """Shapes of the feeds the call built itself — a folded constant, a
        literal — rather than took as an argument or the model's state.  One
        sized from the batch holds the trace-time batch's value."""
        if tracer is None:
            return []
        state = (*self._model.parameters(), *self._model.buffers())
        given = {id(t._impl) for t in (*leaf_tensors(args, kwargs), *state)}
        return [
            tuple(impl.shape) for impl in tracer.external_feeds.values() if id(impl) not in given
        ]

    def _emit(
        self,
        graph: object,
        ext: dict[int, object],
        use_dynamic: bool,
        explicit_outputs: list[int],
    ) -> _ExecutableLike | None:
        """Lower ``graph`` to an executable (symbolic or static).

        For the symbolic path, the model's parameters are flagged so the builder
        keeps their first dim static (only the per-call feeds get the symbolic
        leading axis).  Returns the executable, or ``None`` on a graceful
        lowering failure (unsupported op / off-dim-0 view / backend rejection),
        which the caller turns into a static retry or eager fallback.
        """
        from lucid._dispatch import _unwrap

        param_ids: list[int] = []
        if use_dynamic:
            p_impls = [_unwrap(p) for p in self._model.parameters()]
            id_to_tid = {id(impl): tid for tid, impl in ext.items()}
            for p_impl in p_impls:
                tid = id_to_tid.get(id(p_impl))
                if tid is not None:
                    param_ids.append(tid)
        try:
            exe = _C_engine.compile.compile_or_cached(
                graph, ext, use_dynamic, param_ids, explicit_outputs
            )
        except RuntimeError as why:
            # The engine says why it could not build the graph — an
            # emitter that declined the variant, an unresolved input, a
            # mixed-device trace — and dropping that here is what makes a
            # fallback silent. A fallback is correct and a warning on
            # every call would be noise, so it warns once per kind of
            # reason; ``LUCID_COMPILE_VERBOSE=1``, which already narrates
            # the build, prints every one.  A symbolic lowering is not the
            # last attempt — the caller retries per-shape — so its failure
            # is narrated, not announced as a fallback it may not become.
            if use_dynamic:
                _narrate(f"symbolic lowering declined, retrying per shape: {why}")
            else:
                _fallback_notice("lowering", str(why))
            return None
        if exe is None:
            if not use_dynamic:
                _fallback_notice("lowering", "backend returned no executable")
            return None
        return cast(_ExecutableLike, exe)

    def _trace_forward(
        self,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> tuple[Tracer, object] | None:
        """Trace ``self._model(*args, **kwargs)`` in the caller's grad mode."""
        from lucid._tensor.tensor import Tensor
        from lucid.compile._core.trace_context import (
            notice_if_tensor_hook_error,
            recording,
        )

        try:
            with recording(watch=(args, kwargs), module=self._model, fold_constants=True) as tracer:
                return_value = self._model(*(cast(Tensor, a) for a in args), **kwargs)
        except RuntimeError as why:
            if not notice_if_tensor_hook_error(why):
                raise
            return None
        if not tracer.graph.ops:
            _fallback_notice("empty-trace", "traced call recorded no operations")
            return None
        return tracer, return_value

    def _compile_for(
        self,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> _CacheEntry | None:
        """Trace + compile + wrap one signature into a :class:`_CacheEntry`.

        Parameters
        ----------
        args, kwargs : tuple, dict
            The user's call arguments to feed through the trace.

        Returns
        -------
        _CacheEntry or None
            ``None`` on compile failure or any abort condition
            (kwarg-bound tensors, empty trace, unresolvable feed,
            backend rejection); the caller marks the signature as
            ``eager_only`` and routes to eager.
        """
        # Local imports keep ``lucid.compile`` import-cheap.
        from lucid._dispatch import _unwrap
        from lucid.compile._core.bn_runstats import model_has_tracking_bn
        from lucid.compile._core.buffer_writes import outside_writes

        # 3.5 BatchNorm running-stats: this is the forward-only compile path
        # (no backward, hence no write-back hook).  In training mode a
        # running-stats-tracking BN must advance its EMA every call, which the
        # eager kernel now skips under a tracer — so fall back to eager to keep
        # the stats correct (use make_step for compiled training).  In eval mode
        # BN dispatches the distinct ``batch_norm_eval`` op (no running-stats
        # update), so it compiles normally; track_running_stats=False BN keeps no
        # stats and is likewise unaffected.
        if model_has_tracking_bn(self._model) and any(
            getattr(m, "training", False) and getattr(m, "track_running_stats", False)
            for m in self._model.modules()
        ):
            _fallback_notice(
                "batch-norm-training",
                "forward-only compile cannot advance BatchNorm running stats in training mode "
                "(use make_step or fused_step for compiled training)",
            )
            return None

        t_compile_start = time.perf_counter()
        traced = self._trace_forward(args, kwargs)
        if traced is None:
            return None
        tracer, return_value = traced
        graph = tracer.graph

        # If this graph has attention, make sure the fused-attention workaround
        # capability flag is resolved before the emitters run (probes once).
        from lucid.compile._core.attention_probe import maybe_probe_for_graph

        maybe_probe_for_graph(graph)

        ext = tracer.external_feeds
        # The trace-time call applied its in-place buffer writes; put the
        # buffers back so the executable (or an eager fallback) applies them.
        arguments = {id(_unwrap(t)) for t in leaf_tensors(args, kwargs)}
        writes = outside_writes(tracer, arguments)
        if isinstance(writes, str):
            _fallback_notice("argument-write", writes)
            return None
        write_ids, written = writes

        # Long attention is faster eagerly (CHA-14); the signature is then
        # remembered as eager-only, so only this first call paid the trace.
        from lucid.compile._core.attention_cost import long_attention

        slow_attention = long_attention(tracer)
        if slow_attention is not None:
            _fallback_notice("long-attention", slow_attention)
            return None

        # Extract the user's return value structure into:
        #   * return_spec: a tree that mirrors ``return_value`` but
        #     with every Tensor swapped for a slot index;
        #   * explicit_outputs: ordered list of trace ids the executable
        #     must expose so the slot indices line up.
        # Falling back to eager is safer than guessing if return_value
        # contains an un-traceable type, but the helper tolerates
        # arbitrary scalars / strings / Nones via the "scalar" leaf.
        return_spec, explicit_outputs = _extract_return_impls(return_value, tracer)
        explicit_outputs = explicit_outputs + write_ids

        # ── Symbolic-batch resolution (per-instance, decided on first trace) ──
        # ``dynamic=True`` ATTEMPTS symbolic, but only for graphs the gate proves
        # safe (``_want_symbolic``).  A graph that passes the gate yet still can't
        # lower symbolically — e.g. an off-dim-0 view, which the emitter rejects
        # so ``compile_or_cached`` returns None — falls back to per-shape static.
        # The decision is cached in ``_symbolic_resolved`` so later input shapes
        # key consistently (the model's symbolic-safety is graph-structural, the
        # same across batch sizes).
        use_dynamic = self._want_symbolic(graph, args, tracer, kwargs)
        exe = self._emit(graph, ext, use_dynamic, explicit_outputs)
        if exe is None and use_dynamic:
            use_dynamic = False  # symbolic lowering failed → per-shape static
            exe = self._emit(graph, ext, False, explicit_outputs)
        if exe is None:
            return None
        if self._symbolic and self._symbolic_resolved is None:
            object.__setattr__(self, "_symbolic_resolved", use_dynamic)

        # Figure out, for each feed in exe.input_ids, where it comes from at
        # call time: a positional arg (int index), a keyword arg (str name), or
        # a pinned parameter / constant (None).  Index by ``TensorImpl is``
        # identity.  A tensor passed BOTH positionally and by keyword resolves
        # to the positional slot (args win — they're checked first).
        # Nested tensors too: a tensor inside a list argument used to fall
        # through to ``None`` — pinned at its trace-time value, so every
        # later call reused it and returned a wrong answer without a word.
        feed_source: dict[int, int] = {}
        for i, t in enumerate(leaf_tensors(args, kwargs)):
            feed_source.setdefault(id(_unwrap(t)), i)

        input_source: list[int | None] = []
        for tid in exe.input_ids:
            impl = ext.get(tid)
            if impl is None:
                # Should not happen — the builder only emits placeholder
                # ids that are in the external_feeds set.  Treat as a
                # compile failure.
                _fallback_notice(
                    "unresolvable-feed",
                    f"executable input id {tid} is not in traced external_feeds",
                )
                return None
            input_source.append(feed_source.get(id(impl)))

        from lucid.compile._debug.trace_dump import trace_to_json

        return _CacheEntry(
            exe=exe,
            external_feeds=dict(ext),
            input_source=tuple(input_source),
            return_spec=return_spec,
            graph_json=trace_to_json(graph),
            compile_ms=(time.perf_counter() - t_compile_start) * 1000.0,
            written=written,
            reads=TraceReads.of(tracer, exe.input_ids, input_source),
        )

    def _run(
        self,
        entry: _CacheEntry,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> object:
        """Execute a cached entry and re-pack the output into the user's structure.

        Parameters
        ----------
        entry : _CacheEntry
            The cached executable + feed plan returned by
            :meth:`_compile_for`.
        args, kwargs : tuple, dict
            The user's call arguments; positional :class:`Tensor`
            slots become the per-call feeds, pinned external feeds
            are bound from ``entry.external_feeds``.

        Returns
        -------
        object
            The user's return structure, with each Tensor leaf
            substituted by the matching output of the executable.
            Falls back to :func:`run_eager` if any positional slot
            isn't a Tensor at this call site.
        """
        from lucid._dispatch import _unwrap, _wrap

        # Build the feed list in exe.input_ids order.  ``src`` is an index
        # into the call's tensors (``leaf_tensors`` order) or None (a pinned
        # parameter / constant).
        feed_impls: list[object] = []
        exe = entry.exe
        leaves = leaf_tensors(args, kwargs)
        for tid, src in zip(exe.input_ids, entry.input_source, strict=True):
            if src is None:
                # Pinned parameter / constant — use the trace-time impl.
                feed_impls.append(entry.external_feeds[tid])
                continue
            # Fresh input tensor at this call site, found where the trace
            # found it.  The key pins the structure, so a missing one means
            # the call no longer matches — run it eager.
            if src >= len(leaves):
                return run_eager(self._model, args, kwargs)
            feed_impls.append(_unwrap(leaves[src]))

        t0 = time.perf_counter()
        outs = _C_engine.compile.run_executable(entry.exe, feed_impls)
        entry.last_run_ms = (time.perf_counter() - t0) * 1000.0
        entry.n_hits += 1
        if entry.written:
            from lucid.compile._core.buffer_writes import write_back

            n_writes = len(entry.written)
            write_back(entry.written, list(outs[-n_writes:]))
            outs = outs[:-n_writes]

        # If the trace recorded the user's return structure, re-pack the
        # flat outs list into that shape (single Tensor, tuple, dict,
        # dataclass, …).  Falls back to the legacy single/tuple
        # heuristic if the spec is missing (older cached entries).
        outs_wrapped = [_wrap(o) for o in outs]
        if entry.return_spec is not None:
            return _repack_outputs(entry.return_spec, cast(list[object], outs_wrapped))

        # Legacy: single forward output → plain Tensor.  Multi-output →
        # tuple.  No grad_fn — the compiled forward graph is its own
        # ground truth.
        if not outs_wrapped:
            return None
        if len(outs_wrapped) == 1:
            return outs_wrapped[0]
        return tuple(outs_wrapped)
