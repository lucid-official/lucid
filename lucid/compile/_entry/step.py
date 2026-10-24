"""
lucid.compile._entry.step — Phase 1.5 step 5.1.

``make_step(model, loss_fn)`` returns a callable that traces +
compiles ``model(x); loss_fn(out, target)`` into a single
:class:`MPSGraphExecutable` (forward + backward) and wraps the result
as a Lucid :class:`autograd.Function`.  The user can then call
``loss.backward()`` and the cached fwd+bwd executable supplies the
parameter gradients exactly the way an eager step would — but the
Lucid autograd engine is responsible for wiring those gradients into
the parameter ``.grad`` slots.

This is the autograd-integrated successor to Phase 1.3's
:func:`compiled_step`.  Both surfaces co-exist for the duration of
Phase 1.5: ``compiled_step`` remains the direct call helper for
benchmarks / smoke tests, while ``make_step`` is the production entry
that plays well with eager Lucid models that already populate the
forward call site with `lucid.Tensor` graph state.

Acceptance:

- For a single step, ``loss.backward(); opt.step()`` with ``make_step``
  yields the same parameter trajectory as the eager ``model(x);
  loss_fn(out, target); loss.backward(); opt.step()`` baseline within
  fp32 round-off.
- Cache hit on identical signature avoids re-tracing.
- A graph-incompatible op (e.g. fused ``F.mse_loss``) falls through to
  eager + records the signature as ``eager-only``.
"""

import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Protocol, cast, final, override

from lucid._C import engine as _C_engine
from lucid.autograd.function import Function, FunctionCtx


class _ExecutableLike(Protocol):
    """Surface of pybind11 ``CompiledExecutable`` consumed by _step.py."""

    input_ids: list[int]
    output_ids: list[int]
    grad_output_ids: list[int]
    num_inputs: int


if TYPE_CHECKING:
    from lucid._C.engine.compile import Tracer
    from lucid._tensor.tensor import Tensor
    from lucid.nn.module import Module
    from lucid.nn.parameter import Parameter

__all__ = ["make_step"]


@final
@dataclass(slots=True)
class _StepEntry:
    """One compiled fwd+bwd executable + the I/O plan to invoke it."""

    exe: _ExecutableLike  # _C_engine.compile.PyCompiledExecutable
    external_feeds: dict[int, object]
    # Per ``exe.input_ids[i]`` slot: ``None`` → pinned param/constant in
    # ``external_feeds``; integer → index into the ``step`` call's tensors
    # in ``leaf_tensors`` order (nested ones included).
    input_source: tuple[int | None, ...]
    # JSON-serialised TraceGraph for observability (graph_dump).
    graph_json: str = ""
    n_hits: int = 0
    compile_ms: float = 0.0
    last_run_ms: float = 0.0
    # 3.5 BatchNorm running-stats write-back.  Fused-momentum BN traces as a
    # 5-input/3-output node whose 2 extra outputs are the EMA-updated
    # running_mean / running_var.  They are surfaced as extra executable
    # outputs (trailing [loss, *grads]) and, after each run, copy_'d back into
    # these live module buffers so ``model.eval()`` reads fresh stats instead
    # of the frozen trace-time value.  Empty when the model has no such BN.
    bn_stat_buffers: list[Tensor] = field(default_factory=list)
    bn_stat_out_count: int = 0
    bn_counters: list[tuple[Module, int]] = field(default_factory=list)
    # Tensors from outside the trace that the step writes in place (a
    # buffer's EMA, a counter), in the order their new values trail the BN
    # statistics; copied back after each run.  See ``_core/buffer_writes``.
    written: list[object] = field(default_factory=list)
    # Per parameter, in ``params`` order: did the trace reach it?  A
    # parameter the loss never touches (an auxiliary head unused in eval)
    # has no gradient, exactly as in eager; the executable produces grads
    # for the reached ones only.  Empty = every parameter was reached.
    observed: tuple[bool, ...] = ()
    # Per reached parameter: reached, but the loss does not depend on it
    # (a pooler a token classifier never reads).  The executable fills its
    # slot with zeros and marks its grad id -1; eager leaves such a gradient
    # None, and so does this.  Empty = every reached parameter has one.
    absent: tuple[bool, ...] = ()
    # Per gradient the executable returns after the parameters' — one for
    # each call tensor that requires grad (an eager trunk's output): its
    # position among the call's tensors, or None where the loss does not
    # depend on it.  Without these a compiled head trained no trunk (LCD-617).
    input_grads: tuple[int | None, ...] = ()
    # Every tensor the trace was fed from outside, for ``ParamFingerprint.settle``.
    fed: tuple[_C_engine.TensorImpl, ...] = ()


def _match_param_ids(
    params: list[Parameter],
    ext: dict[int, object],
) -> tuple[list[int], list[bool]]:
    """Match each parameter's ``TensorImpl`` identity against ``ext``."""
    from lucid._dispatch import _unwrap

    param_ids: list[int] = []
    observed: list[bool] = []
    for p in params:
        p_impl = _unwrap(p)
        found = next((tid for tid, impl in ext.items() if impl is p_impl), None)
        # A parameter the trace never reached — GoogLeNet's auxiliary
        # heads in eval, say — used to send the whole step to eager
        # without a word.  Eager leaves its gradient None; so does this.
        observed.append(found is not None)
        if found is not None:
            param_ids.append(found)
    return param_ids, observed


def _input_grad_targets(
    x_args: tuple[Tensor, ...],
    ext: dict[int, object],
    taken: list[int],
) -> tuple[list[int], list[int]]:
    """The feeds of the call's tensors that require grad, and their positions.

    A tensor fed twice, or one that is also a parameter (``taken``), is asked
    for once: its single gradient is the whole of it, and the autograd node
    adds the slots it reaches back together.  A tensor the trace never read
    has no gradient, as in eager.
    """
    from lucid._dispatch import _unwrap
    from lucid.compile._core.signature import leaf_tensors

    feed_of = {id(impl): tid for tid, impl in ext.items()}
    seen = set(taken)
    ids: list[int] = []
    positions: list[int] = []
    for pos, t in enumerate(leaf_tensors(x_args, {})):
        tid = feed_of.get(id(_unwrap(t))) if t.requires_grad else None
        if tid is None or tid in seen:
            continue
        seen.add(tid)
        ids.append(tid)
        positions.append(pos)
    return ids, positions


def _grad_slots(
    exe: _ExecutableLike, n_params: int, input_positions: list[int]
) -> tuple[tuple[bool, ...], tuple[int | None, ...]]:
    """Read the gradients ``exe`` returns: ``_StepEntry.absent`` and ``.input_grads``.

    A gradient the loss does not depend on comes back as zeros with grad id
    -1: absent for a parameter, ``None`` in place of an input's position.
    """
    grad_ids = [int(g) for g in exe.grad_output_ids]
    absent = tuple(g == -1 for g in grad_ids[:n_params])
    inputs = grad_ids[n_params : n_params + len(input_positions)]
    return (
        absent if any(absent) else (),
        tuple(
            None if g == -1 else pos
            for pos, g in zip(input_positions, inputs, strict=True)
        ),
    )


def _build_input_source(
    exe: _ExecutableLike,
    ext: dict[int, object],
    x_args: tuple[Tensor, ...],
) -> tuple[int | None, ...] | None:
    """Build the positional-feed plan for ``exe.input_ids``."""
    from lucid._dispatch import _unwrap
    from lucid.compile._core.signature import leaf_tensors

    positional_impls: dict[int, int] = {}
    for i, a in enumerate(leaf_tensors(x_args, {})):
        positional_impls.setdefault(id(_unwrap(a)), i)

    input_source: list[int | None] = []
    for tid in exe.input_ids:
        impl = ext.get(tid)
        if impl is None:
            return None
        input_source.append(positional_impls.get(id(impl)))
    return tuple(input_source)


def _lower_step_executable(
    tracer: Tracer,
    ext: dict[int, object],
    loss_id: int,
    param_ids: list[int],
    extra_output_ids: list[int],
    *,
    use_dynamic: bool = False,
) -> _ExecutableLike | None:
    """Lower ``tracer``'s recording with backward into an executable, or ``None``."""
    from lucid.compile._core.custom_backward import (
        CustomBackwardInTrace,
        refuse_custom_backward,
    )
    from lucid.compile._core.fallback import _fallback_notice

    try:
        refuse_custom_backward(tracer, loss_id, param_ids)
        exe = _C_engine.compile.compile_trace_with_backward(
            tracer.graph,
            ext,
            loss_id,
            param_ids,
            use_dynamic,
            extra_output_ids=extra_output_ids,
        )
    except CustomBackwardInTrace as why:
        _fallback_notice("custom-backward", str(why))
        return None
    except RuntimeError as why:
        _fallback_notice("lowering", str(why))
        return None
    return None if exe is None else cast(_ExecutableLike, exe)


def _compile_step_entry(
    model: Module,
    loss_fn: Callable[..., Tensor],
    params: list[Parameter],
    x_args: tuple[Tensor, ...],
    *,
    use_dynamic: bool = False,
) -> _StepEntry | None:
    """Trace ``model`` + ``loss_fn`` once for ``x_args`` and compile."""
    from lucid._dispatch import _unwrap, _wrap
    from lucid._tensor.tensor import Tensor
    from lucid.compile._core.attention_probe import maybe_probe_for_graph
    from lucid.compile._core.bn_runstats import (
        bn_counter_targets,
        bn_writeback_targets,
        model_has_cumulative_bn,
    )
    from lucid.compile._core.buffer_writes import outside_writes
    from lucid.compile._core.fallback import _fallback_notice
    from lucid.compile._core.signature import leaf_tensors, outside_feeds
    from lucid.compile._debug.trace_dump import trace_to_json

    t0 = time.perf_counter()
    traced = _trace_step(model, loss_fn, x_args)
    if traced is None:
        return None
    tracer, loss_t = traced
    graph = tracer.graph
    maybe_probe_for_graph(graph)

    ext = tracer.external_feeds
    arguments = {id(_unwrap(t)) for t in leaf_tensors(x_args, {})}
    writes = outside_writes(tracer, arguments)
    if isinstance(writes, str):
        _fallback_notice("argument-write", writes)
        return None
    write_ids, written = writes

    found = tracer.lookup_id(_unwrap(loss_t)) if isinstance(loss_t, Tensor) else None
    if found is None or model_has_cumulative_bn(model):
        return None
    loss_id = int(found)

    bn_targets = bn_writeback_targets(graph, ext)
    bn_stat_out_ids = [out_id for _, out_id, _ in bn_targets]
    bn_stat_buffers = [_wrap(impl) for _, _, impl in bn_targets]

    param_ids, observed = _match_param_ids(params, ext)
    if not param_ids:
        return None
    input_ids, input_positions = _input_grad_targets(x_args, ext, param_ids)

    exe_like = _lower_step_executable(
        tracer,
        ext,
        loss_id,
        param_ids + input_ids,
        bn_stat_out_ids + write_ids,
        use_dynamic=use_dynamic,
    )
    if exe_like is None:
        return None

    input_source = _build_input_source(exe_like, ext, x_args)
    if input_source is None:
        return None

    absent, input_grads = _grad_slots(exe_like, len(param_ids), input_positions)
    return _StepEntry(
        exe=exe_like,
        external_feeds=dict(ext),
        input_source=input_source,
        observed=() if all(observed) else tuple(observed),
        absent=absent,
        input_grads=input_grads,
        graph_json=trace_to_json(graph),
        compile_ms=(time.perf_counter() - t0) * 1000.0,
        bn_stat_buffers=bn_stat_buffers,
        bn_stat_out_count=len(bn_stat_out_ids),
        written=written,
        bn_counters=bn_counter_targets(model, graph, ext),
        fed=outside_feeds(tracer),
    )


_Grads = list[_C_engine.TensorImpl | None]


def _run_step_entry(
    entry: _StepEntry,
    params: list[Parameter],
    x_args: tuple[Tensor, ...],
) -> tuple[_C_engine.TensorImpl, _Grads, _Grads]:
    """Execute one cached step entry.

    Returns the loss, a gradient per parameter, and a gradient per tensor in
    ``x_args`` (``None`` for those that need none).
    """
    from lucid._dispatch import _unwrap, _wrap
    from lucid.compile._core.bn_runstats import advance_bn_counters
    from lucid.compile._core.buffer_writes import write_back

    exe = entry.exe
    feed_impls: list[object] = [
        entry.external_feeds[tid] if src is None else _unwrap(x_args[src])
        for tid, src in zip(exe.input_ids, entry.input_source, strict=True)
    ]
    t0 = time.perf_counter()
    outs = _C_engine.compile.run_executable(entry.exe, feed_impls)
    entry.last_run_ms = (time.perf_counter() - t0) * 1000.0
    entry.n_hits += 1

    n_grads = sum(entry.observed) if entry.observed else len(params)
    n_inputs = len(entry.input_grads)
    n_loss_grad = 1 + n_grads + n_inputs
    n_writes = len(entry.written)
    if len(outs) != n_loss_grad + entry.bn_stat_out_count + n_writes:
        raise RuntimeError(
            f"make_step: run_executable returned {len(outs)} outputs, "
            f"expected {n_loss_grad + entry.bn_stat_out_count + n_writes} "
            f"(1 loss + {n_grads} grads + {n_inputs} input grads + "
            f"{entry.bn_stat_out_count} BN stats + {n_writes} buffer writes)"
        )
    for i, buf in enumerate(entry.bn_stat_buffers):
        buf.copy_(_wrap(cast(_C_engine.TensorImpl, outs[n_loss_grad + i])))
    first = n_loss_grad + entry.bn_stat_out_count
    write_back(entry.written, list(outs[first : first + n_writes]))
    advance_bn_counters(entry.bn_counters)

    grads: list[_C_engine.TensorImpl] = list(outs[1:n_loss_grad])
    input_grads: _Grads = [None] * len(x_args)
    for pos, g in zip(entry.input_grads, grads[n_grads:], strict=True):
        if pos is not None:
            input_grads[pos] = g
    reached: _Grads = [
        None if entry.absent and entry.absent[i] else g
        for i, g in enumerate(grads[:n_grads])
    ]
    loss_impl = cast(_C_engine.TensorImpl, outs[0])
    if not entry.observed:
        return loss_impl, reached, input_grads
    it = iter(reached)
    return (
        loss_impl,
        [next(it) if seen else None for seen in entry.observed],
        input_grads,
    )


@final
class _CompiledStepFunction(Function):
    """Autograd Function wrapping the compiled forward+backward step."""

    _once_differentiable = True  # second derivatives through it are unverified

    @override
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        entry_holder: tuple[_StepEntry, list[Parameter]],
        *step_args: Tensor,
    ) -> Tensor:
        """Run the cached executable; save grads for backward."""
        from lucid._dispatch import _wrap

        entry, params = entry_holder
        x_args = step_args[: len(step_args) - len(params)]

        loss_impl, grad_impls, input_grad_impls = _run_step_entry(entry, params, x_args)
        ctx.saved_grad_impls = input_grad_impls + grad_impls
        return _wrap(loss_impl)

    @override
    @staticmethod
    def backward(ctx: FunctionCtx, grad_loss: Tensor) -> tuple[Tensor | None, ...]:
        """Chain ``grad_loss`` into each saved input and parameter gradient."""
        from lucid._dispatch import _wrap

        saved_grad_impls = cast(_Grads, ctx.saved_grad_impls)
        return tuple(
            None if g is None else grad_loss * _wrap(g) for g in saved_grad_impls
        )


def _make_monolithic_step(
    model: Module,
    loss_fn: Callable[..., Tensor],
) -> Callable[..., Tensor]:
    """Build the monolithic (single-segment) compiled step callable."""
    from lucid._tensor.tensor import Tensor
    from lucid.compile._core.fallback import EagerFallbackSet
    from lucid.compile._core.signature import (
        ParamFingerprint,
        leaf_tensors,
        model_has_active_hooks,
        tensors_have_active_hooks,
    )

    use_dynamic = False
    cache: dict[object, _StepEntry] = {}
    eager_only = EagerFallbackSet()

    params = list(model.parameters())
    if not params:
        raise ValueError("make_step: model has no trainable parameters")
    retrace = _retrace_on_replace(model, cache.clear, params)
    fingerprint = ParamFingerprint(model, on_replaced=retrace)

    def step(*x_args: Tensor) -> Tensor:
        """Run one fwd+bwd compiled training step (or eager fallback)."""
        try:
            key = fingerprint.key_of(x_args, {}, dynamic=use_dynamic)
        except (TypeError, AttributeError):  # fmt: skip
            key = None

        leaves = leaf_tensors(x_args, {})
        if (
            (key in eager_only if key is not None else False)
            or model_has_active_hooks(model)
            or tensors_have_active_hooks(leaves)
        ):
            return loss_fn(model(*x_args[:1]), *x_args[1:])

        entry = cache.get(key) if key is not None else None
        if entry is None:
            entry = _compile_step_entry(
                model, loss_fn, params, x_args, use_dynamic=use_dynamic
            )
            fingerprint.settle(() if entry is None else entry.fed)
            if key is not None:
                key = fingerprint.rekey(key, x_args, {})
            if entry is None:
                if key is not None:
                    eager_only.add(key)
                return loss_fn(model(*x_args[:1]), *x_args[1:])
            if key is not None:
                cache[key] = entry

        return cast(
            Tensor, _CompiledStepFunction.apply((entry, params), *leaves, *params)
        )

    def clear_cache() -> None:
        """Drop every cached executable + eager-fallback blacklist for this step callable."""
        cache.clear()
        eager_only.clear()

    def graph_dump() -> str:
        return _graph_dump(cache.items())

    step.cache = cache  # type: ignore[attr-defined]
    step.eager_only = eager_only  # type: ignore[attr-defined]
    step.clear_cache = clear_cache  # type: ignore[attr-defined]
    step.graph_dump = graph_dump  # type: ignore[attr-defined]
    return step


def make_step(
    model: Module,
    loss_fn: Callable[..., Tensor],
    *,
    dynamic: bool = False,
    segments: int | str = 1,
) -> Callable[..., Tensor]:
    """Return a callable that runs one compiled training step.

    The returned ``step(x, *extra_inputs)`` callable:

    1. Calls ``loss_fn(model(x), *extra_inputs)`` once under
       :func:`_tracing` to capture the op DAG (forward + loss).
    2. Compiles a single MPSGraph executable that produces the loss
       and the gradient of the loss w.r.t. every model parameter and
       every input that requires grad, via
       :func:`compile_trace_with_backward`.
    3. Runs the executable, populates a Lucid :class:`autograd.Function`
       with the resulting (loss, grads) so the returned scalar loss has
       a working ``grad_fn`` and ``loss.backward()`` drives ``.grad`` on
       every parameter — and on through an input computed by eager ops,
       so a compiled head trains the eager trunk that feeds it.
    4. Caches the executable so a second call with the same input /
       parameter signature is a pure run.  Which inputs require grad is
       part of the signature: a step whose inputs need no gradient
       computes none.

    Parameters
    ----------
    model : nn.Module
        Model whose forward will be traced + compiled.
    loss_fn : callable
        ``loss_fn(model_output, *extra_inputs) -> Tensor`` scalar loss.
    dynamic : bool, optional
        Accepted for API symmetry with :func:`lucid.compile`, but the compiled
        *training* step is always per-shape static: any batch size works (one
        fwd+bwd executable per distinct shape), it just does not share a single
        symbolic-batch executable across sizes.
    segments : int or {"auto"}, optional
        Number of executable segments (default ``1`` — the monolithic single
        executable described above).  When ``> 1`` and ``model`` is an
        :class:`nn.Sequential`, the model is split into that many contiguous
        groups of children, each compiled into its own forward + backward
        executable and stitched with eager autograd
        (:func:`make_segmented_step`).  ``"auto"`` probes a few candidate counts
        on the first batch and keeps the fastest (:func:`make_autoseg_step`).

    Returns
    -------
    callable
        ``step(x, *extra_inputs) -> Tensor`` — the scalar loss.
        ``loss.backward()`` works the same way as the eager baseline.

    Notes
    -----
    A step whose gradient passes through a custom
    :class:`~lucid.autograd.Function` runs eagerly, with a one-time
    ``CompileFallbackWarning``: the executable can only differentiate the
    ops ``forward`` recorded, and the ``backward`` the ``Function`` defines
    — a straight-through estimator, a scaled or reversed gradient — would
    not run.

    Examples
    --------
    >>> import lucid, lucid.nn as nn, lucid.nn.functional as F
    >>> import lucid.optim as optim
    >>> from lucid.compile import make_step
    >>> model = nn.Linear(8, 4).to('metal')
    >>> step = make_step(model, lambda y, t: F.cross_entropy(y, t))
    >>> opt = optim.SGD(model.parameters(), lr=0.1)
    >>> for batch in batches:                          # doctest: +SKIP
    ...     opt.zero_grad()
    ...     loss = step(batch.x, batch.target)
    ...     loss.backward()
    ...     opt.step()

    See Also
    --------
    lucid.compile.compile : module-level compile entrypoint.
    lucid.compile.compile_optimizer : even tighter form that fuses
        the optimizer update into the same MPSGraph executable.
    lucid.compile._entry.module.CompiledModule.step : per-instance
        cached version of the same dispatch.
    make_segmented_step : the ``segments > 1`` implementation (executable
        splitting for memory-bounded deep training).
    """
    if isinstance(segments, str):
        if segments == "auto":
            from lucid.compile._entry.segmented_step import make_autoseg_step

            return make_autoseg_step(model, loss_fn, dynamic=dynamic)
        raise ValueError(
            f"make_step: segments must be an int or 'auto', got {segments!r}"
        )
    if segments > 1:
        from lucid.compile._entry.segmented_step import make_segmented_step

        return make_segmented_step(model, loss_fn, segments=segments, dynamic=dynamic)
    return _make_monolithic_step(model, loss_fn)


def _trace_step(
    model: Module,
    loss_fn: Callable[..., Tensor],
    x_args: tuple[Tensor, ...],
) -> tuple[Tracer, Tensor] | None:
    """Trace ``model`` + ``loss_fn`` under :func:`no_grad`, declining on tensor hooks."""
    from lucid.autograd._grad_mode import no_grad
    from lucid.compile import _tracing
    from lucid.compile._core.fallback import _fallback_notice

    try:
        with no_grad(), _tracing(watch=x_args, module=model) as tracer:
            x_out = model(*x_args[:1])
            loss_t = loss_fn(x_out, *x_args[1:])
    except RuntimeError as why:
        if "register_hook" not in str(why):
            raise
        _fallback_notice("tensor-hook", str(why))
        return None
    if not tracer.graph.ops:
        return None
    return tracer, loss_t


def _retrace_on_replace(
    model: Module, drop_executables: Callable[[], None], params: list[Parameter]
) -> Callable[[], None]:
    """What a :func:`make_step` does when its model's tensors are replaced.

    Each executable is fed the tensors it was traced with: after an
    ``assign=True`` load, or a ``.half()`` / ``.float()`` round trip that
    keys back to the first executable, the step went on training the old
    tensors — a loss from the old weights, gradients for tensors the model no
    longer holds (LCD-505).  So every executable is dropped, and the
    parameters are taken again: an ``assign=True`` load puts new
    ``Parameter`` objects in the tree, as does one set in another's place
    (LCD-560) — the gradients go to the parameters the model holds now.
    """

    def retrace() -> None:
        drop_executables()
        params[:] = model.parameters()

    return retrace


def _graph_dump(entries: Iterable[tuple[object, _StepEntry]]) -> str:
    """Return a JSON view of every cached signature's :class:`TraceGraph`.

    ``step.graph_dump()`` of a :func:`make_step` callable.  Used by
    `tools/inspect_trace.py` and the test harness to diff consecutive
    compiles and pinpoint divergent ops.
    """
    import json as _json

    items = [
        {"sig_hash": hash(k), "graph": _json.loads(e.graph_json)} for k, e in entries
    ]
    return _json.dumps(items, indent=2)
