"""
lucid.compile._entry.segmented_step — memory-bounded compiled training via
executable splitting.

``make_step(model, loss_fn, segments=K)`` routes here when ``K > 1``.

Why
---
A monolithic compiled training step (:func:`make_step` with ``segments=1``)
lowers the whole ``model(x); loss`` forward + backward into ONE
:class:`MPSGraphExecutable`.  That executable **holds every forward
activation alive until its backward consumes it** — peak memory grows
linearly with depth (measured: a bottleneck ``×8`` stack peaks at 8.1 GB vs
the reference framework's incrementally-freed 3.4 GB).  In-executable
gradient checkpointing does NOT help: MPSGraph CSE-merges the recompute back
onto the saved activations, so peak is unchanged (validated dead).

Splitting the model into ``K`` segments, each its OWN forward + backward
executable, escapes the penalty: MPSGraph can neither CSE nor hold memory across
executable boundaries, so each segment's backward holds only its own
activations.  This is gradient checkpointing realised as separate executables —
a memory↔compute trade (each segment's forward is recomputed in its backward),
the same trade as the reference framework's ``checkpoint``.

Architecture (every executable is one segment — pool-friendly)
--------------------------------------------------------------
Each segment ``i`` gets two compiled executables, both built lazily on the first
call for a given input signature:

* **forward exe** — ``x_{i-1} -> x_i`` (forward only).  Run during the forward
  chain.  Its output is the segment boundary; only boundaries stay alive between
  forward and backward (segment internals are freed).
* **backward exe** — ``compile_trace_with_backward`` of ``loss_seg =
  sum(x_i * grad_out)`` with the segment INPUT added to ``param_ids``.  Run
  during backprop with ``grad_out`` (the upstream cotangent) fed as a runtime
  input; returns ``(grad_input, *grad_params)``.  The forward is recomputed
  inside this executable (one segment deep — cheap).

Keeping every executable one-segment-sized is deliberate: the Metal allocator
pool retains buffers across executables, and same-shape small exes reuse the
pool, holding peak near "boundaries + one segment".  Two rejected alternatives
both ballooned peak — a single *fused* forward exe for the whole chain
(seg8 ~9.9 GB; many spread outputs defeat MPSGraph liveness) and an *eager*
forward chain (seg8 ~11 GB; MLX transients pool poorly).  The per-segment
compiled forward is the memory optimum; its only cost is ``K`` forward
dispatches.

The segments are stitched with **eager autograd**: each is an
:class:`autograd.Function` whose ``forward`` runs the forward exe and whose
``backward`` runs the backward exe.  ``loss_fn`` is applied eagerly to the final
segment's output, so ``loss.backward()`` flows the loss cotangent into the last
segment and the chain unwinds segment-by-segment, each backward exe running
sequentially with only its own activations resident.

Correctness: token-identical parameter gradients to the monolithic
``segments=1`` path (same emitters, same forward recompute, exact chain rule).
"""

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Sequence, cast, final, override

from lucid._C import engine as _C_engine
from lucid._tensor.tensor import Tensor
from lucid.autograd.function import Function, FunctionCtx
from lucid.compile._core.signature import outside_feeds, tree_tensors

if TYPE_CHECKING:
    from lucid._C.engine.compile import CompiledExecutable, Tracer
    from lucid.compile._core.signature import ParamFingerprint, _Slot
    from lucid.nn.module import Module
    from lucid.nn.modules.container import Sequential

__all__ = ["make_segmented_step", "make_autoseg_step"]


class _SegmentCompileError(Exception):
    """Internal signal: a segment failed to compile → fall back to eager."""

    def __init__(self, index: int) -> None:
        super().__init__(f"segment {index} failed to compile")
        self.index = index


# Feed-plan tags: how to source each executable input slot at run time.
#   ("X", None)    -> the segment's runtime input tensor
#   ("GO", None)   -> the upstream cotangent (backward exe only)
#   ("P", idx)     -> live impl of ``segment.params[idx]`` (re-read each call,
#                     so optimizer param rebinds are picked up)
#   ("C", impl)    -> a frozen graph constant captured at compile time
_FeedTag = tuple[str, object]


@final
@dataclass(slots=True)
class _SegExe:
    """One compiled executable + the per-input-slot feed plan to invoke it."""

    exe: object  # _C_engine.compile.PyCompiledExecutable
    feed_plan: tuple[_FeedTag, ...]


@final
@dataclass(slots=True)
class _SegCompiled:
    """The forward + backward executables compiled for one input signature."""

    fwd: _SegExe
    bwd: _SegExe
    # ``produce_input_grad`` ⇒ the backward exe's first grad output is
    # ``grad_input`` (the cotangent to feed the previous segment, or the eager
    # trunk before the first); the rest are per-parameter grads.  Set when
    # the segment's input requires grad — a first segment fed a data batch
    # skips it — and is not one of its parameters (``_grad_targets``).
    produce_input_grad: bool
    n_params: int
    # Every tensor the traces were fed from outside, for ``ParamFingerprint.settle``.
    fed: tuple[_C_engine.TensorImpl, ...] = ()


type _SegKey = tuple[tuple[int, ...], str, bool, tuple[tuple[int, _Slot], ...]]


def _sig(x: Tensor, fingerprint: ParamFingerprint) -> _SegKey:
    """Compile-cache key for a segment: its input shape, dtype and grad need.

    Batch size is fixed across a training run, so one entry is compiled per
    segment.  Parameters never change shape during training, so they are not
    part of the key.  Whether the input requires grad is: the first segment
    returned no gradient to an input that did — an eager trunk before a
    compiled model trained nothing (LCD-617).  So is whether the input is a
    tensor the model holds: a parameter passed as the input shares that
    parameter's feed, and another input ran in its place (LCD-629).
    """
    held = fingerprint.tree_alias((x,), {})
    return (tuple(x.shape), str(x.dtype), bool(x.requires_grad), held)


def _match_param_ids(ext: dict[int, object], param_impls: Sequence[object]) -> list[int] | None:
    """Map each parameter impl to its trace feed id (by pointer identity).

    Returns ``None`` if any parameter was not observed in the trace (segment
    forward did not use it) → the step falls back to eager.
    """
    param_ids: list[int] = []
    for p_impl in param_impls:
        found: int | None = None
        for tid, impl in ext.items():
            if impl is p_impl:
                found = tid
                break
        if found is None:
            return None
        param_ids.append(found)
    return param_ids


def _build_feed_plan(
    input_ids: Sequence[int],
    ext: dict[int, object],
    x_id: int,
    go_id: int | None,
    param_id_to_index: dict[int, int],
) -> tuple[_FeedTag, ...] | None:
    """Classify every executable input slot into a runtime feed source."""
    plan: list[_FeedTag] = []
    for tid in input_ids:
        if tid == x_id:
            plan.append(("X", None))
        elif go_id is not None and tid == go_id:
            plan.append(("GO", None))
        elif tid in param_id_to_index:
            plan.append(("P", param_id_to_index[tid]))
        else:
            impl = ext.get(tid)
            if impl is None:
                return None
            plan.append(("C", impl))
    return tuple(plan)


def _feed(
    plan: tuple[_FeedTag, ...],
    x_impl: object,
    go_impl: object | None,
    param_impls: Sequence[object],
) -> list[object]:
    """Materialise the run_executable feed list from a feed plan."""
    feeds: list[object] = []
    for tag, payload in plan:
        if tag == "X":
            feeds.append(x_impl)
        elif tag == "GO":
            assert go_impl is not None, "GO feed requested outside backward"
            feeds.append(go_impl)
        elif tag == "P":
            feeds.append(param_impls[cast(int, payload)])
        else:  # "C"
            feeds.append(payload)
    return feeds


@final
@dataclass(slots=True)
class _Segment:
    """One model segment: its sub-forward, parameters, and per-signature cache."""

    index: int
    forward_fn: Callable[..., Tensor]  # x -> y (traced, not run eagerly)
    params: list[Tensor]
    # Its modules' parameters and buffers: a trace that reads one on the host
    # is unsupported, or the value read would be baked in.
    state: tuple[Tensor, ...] = ()
    cache: dict[_SegKey, _SegCompiled] = field(default_factory=dict)


def _trace_segment_fn(
    watch: tuple[object, ...], fn: Callable[[], Tensor]
) -> tuple[Tracer, Tensor] | None:
    """Trace *fn* in the caller's grad mode, returning ``None`` on a tensor hook or empty graph."""
    from lucid.compile._core.attention_probe import maybe_probe_for_graph
    from lucid.compile._core.buffer_writes import refuse
    from lucid.compile._core.trace_context import notice_if_tensor_hook_error, recording

    try:
        with recording(watch=watch) as tracer:
            out = fn()
    except RuntimeError as why:
        if not notice_if_tensor_hook_error(why):
            raise
        return None
    if not tracer.graph.ops:
        return None
    maybe_probe_for_graph(tracer.graph)
    refuse(tracer)
    return tracer, out


def _grad_targets(x: Tensor, x_id: int, param_ids: list[int]) -> list[int]:
    """The ids a segment's backward returns gradients for: [input?] + params.

    The input's makes the executable also return the cotangent fed to whatever
    computed the input, exactly when eager would compute one.  An input that
    is one of the segment's parameters is that parameter's feed: its gradient
    is the whole of it, and returned twice it was added twice (LCD-629).
    """
    if x.requires_grad and x_id not in param_ids:
        return [x_id, *param_ids]
    return param_ids


def _lower_segment_forward(
    tracer: Tracer, y_id: int, grad_target_ids: list[int]
) -> CompiledExecutable | None:
    """A segment's forward executable, or ``None`` when it cannot be compiled."""
    from lucid.compile._core.custom_backward import CustomBackwardInTrace, refuse_custom_backward
    from lucid.compile._core.fallback import _fallback_notice

    try:
        refuse_custom_backward(tracer, y_id, grad_target_ids)
        return _C_engine.compile.compile_or_cached(
            tracer.graph, dict(tracer.external_feeds), False, [], [y_id]
        )
    except CustomBackwardInTrace as why:
        _fallback_notice("custom-backward", str(why))
        return None
    except RuntimeError as why:
        _fallback_notice("lowering", str(why))
        return None


def _lower_segment_backward(
    tracer: Tracer, loss_id: int, grad_target_ids: list[int]
) -> CompiledExecutable | None:
    """A segment's backward executable, or ``None`` when it cannot be compiled."""
    from lucid.compile._core.custom_backward import CustomBackwardInTrace, refuse_custom_backward
    from lucid.compile._core.fallback import _fallback_notice

    try:
        refuse_custom_backward(tracer, loss_id, grad_target_ids)
        return _C_engine.compile.compile_trace_with_backward(
            tracer.graph, dict(tracer.external_feeds), loss_id, grad_target_ids, False
        )
    except CustomBackwardInTrace as why:
        _fallback_notice("custom-backward", str(why))
        return None
    except RuntimeError as why:
        _fallback_notice("lowering", str(why))
        return None


def _compile_segment(seg: _Segment, x: Tensor) -> _SegCompiled | None:
    """Trace + compile the forward and backward executables for ``seg`` at ``x``.

    Two traces are taken (forward-only, then forward + ``loss_seg``) so the
    forward executable does not carry the cotangent-product tail.  Returns
    ``None`` on any uncompilable condition → the caller falls back to eager.
    """
    import lucid
    from lucid._dispatch import _unwrap

    param_impls = [_unwrap(p) for p in seg.params]
    x_impl = _unwrap(x)

    # --- Trace 1: forward only (x -> y) -----------------------------------
    traced_fwd = _trace_segment_fn((x, seg.state), lambda: seg.forward_fn(x))
    if traced_fwd is None:
        return None
    t_fwd, y = traced_fwd
    ext_fwd = t_fwd.external_feeds
    # The segment's output, not whatever op it recorded last.
    found = t_fwd.lookup_id(_unwrap(y))
    if found is None:
        return None
    y_id = int(found)

    x_id_fwd: int | None = None
    for tid, impl in ext_fwd.items():
        if impl is x_impl:
            x_id_fwd = tid
            break
    if x_id_fwd is None:
        return None

    fwd_param_ids = _match_param_ids(ext_fwd, param_impls)
    if fwd_param_ids is None:
        return None

    fwd_exe = _lower_segment_forward(t_fwd, y_id, _grad_targets(x, x_id_fwd, fwd_param_ids))
    if fwd_exe is None:
        return None

    fwd_pid_to_idx = {pid: i for i, pid in enumerate(fwd_param_ids)}
    fwd_plan = _build_feed_plan(list(fwd_exe.input_ids), ext_fwd, x_id_fwd, None, fwd_pid_to_idx)
    if fwd_plan is None:
        return None

    # --- Trace 2: forward + loss_seg = sum(y * grad_out) -------------------
    # ``grad_out`` is a fresh leaf feed shaped like ``y``; the backward feeds the
    # upstream cotangent into this slot.
    go = lucid.zeros(y.shape, dtype=y.dtype, device=y.device)
    traced_bwd = _trace_segment_fn((x, seg.state), lambda: (seg.forward_fn(x) * go).sum())
    if traced_bwd is None:
        return None
    t_bwd, _ = traced_bwd
    g_bwd = t_bwd.graph
    ext_bwd = t_bwd.external_feeds
    loss_id = g_bwd.ops[-1].outputs[0].id

    x_id_bwd: int | None = None
    go_id_bwd: int | None = None
    go_impl = _unwrap(go)
    for tid, impl in ext_bwd.items():
        if impl is x_impl:
            x_id_bwd = tid
        elif impl is go_impl:
            go_id_bwd = tid
    if x_id_bwd is None or go_id_bwd is None:
        return None

    bwd_param_ids = _match_param_ids(ext_bwd, param_impls)
    if bwd_param_ids is None:
        return None

    grad_target_ids = _grad_targets(x, x_id_bwd, bwd_param_ids)
    bwd_exe = _lower_segment_backward(t_bwd, int(loss_id), grad_target_ids)
    if bwd_exe is None:
        return None

    bwd_pid_to_idx = {pid: i for i, pid in enumerate(bwd_param_ids)}
    bwd_plan = _build_feed_plan(
        list(bwd_exe.input_ids), ext_bwd, x_id_bwd, go_id_bwd, bwd_pid_to_idx
    )
    if bwd_plan is None:
        return None

    return _SegCompiled(
        fwd=_SegExe(exe=fwd_exe, feed_plan=fwd_plan),
        bwd=_SegExe(exe=bwd_exe, feed_plan=bwd_plan),
        produce_input_grad=len(grad_target_ids) > len(bwd_param_ids),
        n_params=len(seg.params),
        fed=outside_feeds(t_fwd) + outside_feeds(t_bwd),
    )


def _compile_into(seg: _Segment, x: Tensor, fingerprint: ParamFingerprint) -> _SegCompiled:
    """Compile ``seg`` at ``x`` into its cache, telling ``fingerprint`` what the
    traces read; raise :class:`_SegmentCompileError` if it does not compile."""
    from lucid.compile._core.trace_context import compile_attempt

    with compile_attempt():
        before = fingerprint.outside_slots()
        compiled = _compile_segment(seg, x)
        fingerprint.settle(() if compiled is None else compiled.fed, before)
    if compiled is None:
        raise _SegmentCompileError(seg.index)
    seg.cache[_sig(x, fingerprint)] = compiled
    return compiled


def _forward_segments(
    seg_objs: Sequence[_Segment], cur: Tensor, fingerprint: ParamFingerprint
) -> Tensor:
    """Run (and lazily compile) each segment in ``seg_objs`` on ``cur``."""
    for seg in seg_objs:
        compiled = seg.cache.get(_sig(cur, fingerprint))
        if compiled is None:
            compiled = _compile_into(seg, cur, fingerprint)
        holder: list[object] = [compiled]
        cur = cast(Tensor, _SegmentFunction.apply(holder, cur, *seg.params))
    return cur


@final
class _SegmentFunction(Function):
    """autograd.Function wrapping one compiled segment.

    ``apply(holder, x, *params)`` — ``holder`` is a 1-element list owning the
    compiled :class:`_SegCompiled` (a non-Tensor positional, given no autograd
    edge, like ``make_step``'s ``_StepEntry`` holder).  ``x`` + ``params`` are
    the Tensor inputs; backward returns one cotangent per Tensor input.
    """

    _once_differentiable = True  # second derivatives through it are unverified

    # forward's 1st positional is a non-Tensor holder list, like
    # make_step._CompiledStepFunction's.
    @override
    @staticmethod
    def forward(ctx: FunctionCtx, holder: list[object], x: Tensor, *params: Tensor) -> Tensor:
        """Run the segment's forward executable; stash state for backward."""
        from lucid._dispatch import _unwrap, _wrap

        compiled = cast(_SegCompiled, holder[0])
        param_impls = [_unwrap(p) for p in params]
        feeds = _feed(compiled.fwd.feed_plan, _unwrap(x), None, param_impls)
        outs = _C_engine.compile.run_executable(compiled.fwd.exe, feeds)
        y_impl = cast(_C_engine.TensorImpl, outs[0])

        ctx.compiled = compiled
        ctx.x_impl = _unwrap(x)
        ctx.param_impls = param_impls
        ctx.n_params = len(params)
        return _wrap(y_impl)

    # backward returns tuple[Tensor | None, ...], one per Tensor input (x then
    # params); the base returns Tensor | tuple.
    @override
    @staticmethod
    def backward(ctx: FunctionCtx, grad_out: Tensor) -> tuple[Tensor | None, ...]:
        """Run the segment's backward executable; return (grad_x, *grad_params)."""
        from lucid._dispatch import _unwrap, _wrap

        compiled = cast(_SegCompiled, ctx.compiled)
        x_impl = cast(_C_engine.TensorImpl, ctx.x_impl)
        n_params = cast(int, ctx.n_params)
        param_impls = cast(list[object], ctx.param_impls)

        feeds = _feed(compiled.bwd.feed_plan, x_impl, _unwrap(grad_out), param_impls)
        outs = _C_engine.compile.run_executable(compiled.bwd.exe, feeds)

        # Output layout: [loss_seg, (grad_x?), *grad_params].
        idx = 1  # skip loss_seg
        if compiled.produce_input_grad:
            grad_x: Tensor | None = _wrap(cast(_C_engine.TensorImpl, outs[idx]))
            idx += 1
        else:
            grad_x = None
        grad_params = [_wrap(cast(_C_engine.TensorImpl, outs[idx + i])) for i in range(n_params)]
        return tuple([grad_x, *grad_params])


def _group_fn(mods: list[Module]) -> Callable[[Tensor], Tensor]:
    """The forward of one segment: its modules, one after another."""
    from lucid._tensor.tensor import Tensor

    def fn(x: Tensor) -> Tensor:
        for m in mods:
            out = m(x)
            if not isinstance(out, Tensor):
                raise TypeError(
                    "make_step(segments>1): a segment module returned a "
                    f"non-Tensor ({type(out).__name__}); segmented compile "
                    "needs single-Tensor stage outputs"
                )
            x = out
        return x

    return fn


def _segments_of(model: Sequential, segments: int) -> list[_Segment]:
    """Split ``model``'s children, as it holds them now, into contiguous segments."""
    children = list(model)
    n_children = len(children)
    if n_children == 0:
        raise ValueError("make_step(segments>1): model has no child modules")
    k = max(1, min(segments, n_children))

    # Balanced contiguous partition of children into k groups.
    seg_objs: list[_Segment] = []
    base, extra = divmod(n_children, k)
    start = 0
    for gi in range(k):
        mods = children[start : start + base + (1 if gi < extra else 0)]
        start += len(mods)
        seg_objs.append(
            _Segment(
                index=gi,
                forward_fn=_group_fn(mods),
                params=[p for m in mods for p in m.parameters()],
                state=tuple(t for m in mods for t in tree_tensors(m)),
            )
        )
    return seg_objs


def make_segmented_step(
    model: Module, loss_fn: Callable[..., Tensor], *, segments: int, dynamic: bool = False
) -> Callable[..., Tensor]:
    """Return a segmented compiled training step (executable splitting).

    The model is split into ``segments`` contiguous groups of its
    :class:`nn.Sequential` children; each group is compiled into its own forward
    + backward executable and the groups are stitched with eager autograd.  Peak
    memory drops on deep models because no single executable holds the full
    activation stack (see module docstring); the recompute it trades for is the
    standard checkpointing cost.

    Parameters
    ----------
    model : nn.Module
        Must be an :class:`nn.Sequential` so the split into contiguous segments
        is well-defined.  Non-sequential models raise — use
        ``make_step(..., segments=1)`` for those.
    loss_fn : callable
        ``loss_fn(final_output, *extra_inputs) -> Tensor`` scalar loss, applied
        eagerly to the last segment's output.
    segments : int
        Number of segments ``K`` (``>= 2``; clamped to the child count).
    dynamic : bool, optional
        Accepted for API symmetry; the segmented step is always per-shape static.

    Returns
    -------
    callable
        ``step(x, *extra_inputs) -> Tensor`` scalar loss whose ``.backward()``
        drives ``.grad`` on every parameter, token-identical to the monolithic
        path.
    """
    import lucid.nn as nn
    from lucid.compile._core.signature import (
        ParamFingerprint,
        leaf_tensors,
        model_has_active_hooks,
        tensors_have_active_hooks,
    )

    del dynamic  # API symmetry with make_step; segmented steps are per-shape static
    if not isinstance(model, nn.Sequential):
        raise TypeError(
            "make_step(segments>1) requires an nn.Sequential model so the "
            "split into contiguous segments is well-defined; got "
            f"{type(model).__name__}. Use segments=1 for arbitrary models, or "
            "wrap the sequential trunk in nn.Sequential."
        )

    # Rebuilt when the model's tensors or children are replaced: each segment
    # holds its modules and parameters, and its executables read them (LCD-560).
    seq = model
    seg_objs = _segments_of(seq, segments)
    caches = [seg.cache for seg in seg_objs]

    def rebuild() -> None:
        seg_objs[:] = _segments_of(seq, segments)
        caches[:] = [seg.cache for seg in seg_objs]

    fingerprint = ParamFingerprint(model, on_replaced=rebuild)

    all_params = list(model.parameters())
    if not all_params:
        raise ValueError("make_step(segments>1): model has no trainable parameters")

    eager_only = {"flag": False}

    def _run_eager(x_args: tuple[Tensor, ...]) -> Tensor:
        out = model(*x_args[:1])
        return loss_fn(out, *x_args[1:])

    def step(*x_args: Tensor) -> Tensor:
        """Run one segmented compiled training step (or eager fallback)."""
        fingerprint.get()  # rebuilds the segments if the model's were replaced
        if (
            eager_only["flag"]
            or model_has_active_hooks(model)
            or tensors_have_active_hooks(leaf_tensors(x_args, {}))
        ):
            return _run_eager(x_args)

        # Compile every segment for this input chain on the first call.  Forward
        # exes must run to know the next segment's input shape, so compile +
        # forward are interleaved.  Each segment is an autograd node;
        # ``loss.backward()`` later unwinds them in reverse.
        from lucid._factories._generator import _restore_stream_states, _stream_states

        rng_before = _stream_states()
        try:
            cur = _forward_segments(seg_objs, x_args[0], fingerprint)
        except _SegmentCompileError:
            _restore_stream_states(rng_before)
            eager_only["flag"] = True
            return _run_eager(x_args)

        # Eager loss on the final segment output.
        return loss_fn(cur, *x_args[1:])

    step.cache = caches  # type: ignore[attr-defined]
    step.eager_only = eager_only  # type: ignore[attr-defined]
    step.segments = len(seg_objs)  # type: ignore[attr-defined]

    def clear_cache() -> None:
        """Drop every cached executable for this segmented step callable."""
        for seg in seg_objs:
            seg.cache.clear()
        eager_only["flag"] = False

    step.clear_cache = clear_cache  # type: ignore[attr-defined]
    return step


def _auto_candidates(n_children: int) -> list[int]:
    """Candidate segment counts to probe for ``segments="auto"``.

    A small geometric set in ``[2, n_children]`` plus ``n_children`` itself,
    capped at four so the one-time probe stays cheap.  ``segments=1`` (mono) is
    intentionally excluded — auto is opt-in to splitting; a user who wants the
    monolithic step does not pass ``"auto"``.
    """
    cands: set[int] = set()
    k = 2
    while k < n_children:
        cands.add(k)
        k *= 2
    if n_children >= 2:
        cands.add(n_children)
    ordered = sorted(c for c in cands if 2 <= c <= n_children)
    if len(ordered) > 4:
        # Keep the extremes + two spread interior points.
        lo, hi = ordered[0], ordered[-1]
        mids = ordered[1:-1]
        picks = {lo, hi, mids[len(mids) // 3], mids[2 * len(mids) // 3]}
        ordered = sorted(picks)
    return ordered


def _stand_ins(x_args: tuple[object, ...]) -> tuple[object, ...]:
    """Fresh leaves with ``x_args``' values and grad need, for the timing runs.

    The probe runs a step and its backward several times.  Fed the call's own
    tensors, those backwards would run on into the graph behind an input that
    eager ops computed — a trunk's — and the first would free it under the
    caller.  A stand-in that requires grad times the same executables (its
    gradient included) and ends the backward there.
    """
    from lucid._tensor.tensor import Tensor

    return tuple(
        a.detach().requires_grad_(a.requires_grad) if isinstance(a, Tensor) else a for a in x_args
    )


def _probe_one(model: Module, seg_step: Callable[..., Tensor], x_args: tuple[object, ...]) -> float:
    """Median fwd+bwd wall-time (ms) of ``seg_step`` over a few warm runs.

    The backward is taken with :func:`lucid.autograd.grad`, which hands the
    gradients back instead of adding them into ``.grad``: the timing runs
    leave the caller's gradients as they were (they used to be set to None).
    """
    from lucid._tensor.tensor import Tensor
    from lucid.autograd import grad

    wrt: list[Tensor] = [p for p in model.parameters() if p.requires_grad]
    wrt += [a for a in x_args if isinstance(a, Tensor) and a.requires_grad]

    def one() -> None:
        grads = grad(seg_step(*x_args), wrt, allow_unused=True)
        # Force the backward graph to materialise so the timing is real
        # (MLX dispatch is async): touch one gradient.
        for g in grads:
            if g is not None:
                float(g.reshape(-1)[0].item())
                break

    for _ in range(2):
        one()
    samples: list[float] = []
    for _ in range(3):
        t0 = time.perf_counter()
        one()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


def make_autoseg_step(
    model: Module,
    loss_fn: Callable[..., Tensor],
    *,
    dynamic: bool = False,
    candidates: list[int] | None = None,
) -> Callable[..., Tensor]:
    """Return a step that auto-selects the segment count on the first batch.

    On the first call, builds a segmented step for each candidate ``K`` (see
    :func:`_auto_candidates`), times a few real fwd+bwd steps of each, and keeps
    the **fastest**; every later call delegates to it.

    Why time, not memory: the segmented-step lever's sweet spot (fastest split)
    is also a good memory point, and step *time* is the only metric reliably
    measurable in-process — the Metal allocator pool never shrinks, so
    per-``K`` peak-memory readings within one process are cumulative and cannot
    be compared.  For a strict memory ceiling, pass an explicit
    ``segments=K`` (more segments ⇒ less peak) instead of ``"auto"``.

    The probe clears each candidate's executables before trying the next, so its
    transient peak stays near the heaviest single candidate (not their sum).

    Parameters
    ----------
    model : nn.Module
        Must be an :class:`nn.Sequential` (same constraint as
        :func:`make_segmented_step`).
    loss_fn : callable
        ``loss_fn(final_output, *extra_inputs) -> Tensor`` scalar loss.
    dynamic : bool, optional
        Forwarded to each candidate's :func:`make_segmented_step`.
    candidates : list of int, optional
        Override the probed segment counts.  Defaults to
        :func:`_auto_candidates`.

    Returns
    -------
    callable
        ``step(x, *extra_inputs) -> Tensor`` — identical interface to
        :func:`make_segmented_step`'s, with ``.chosen_segments()`` and
        ``.probe_results()`` introspection helpers attached.
    """
    import lucid.nn as nn
    from lucid._tensor.tensor import Tensor

    if not isinstance(model, nn.Sequential):
        raise TypeError(
            'make_step(segments="auto") requires an nn.Sequential model; got '
            f"{type(model).__name__}."
        )
    n_children = len(list(model))
    if n_children < 2:
        raise ValueError(
            'make_step(segments="auto"): model has fewer than 2 child modules ' "to split"
        )
    cand_list = candidates if candidates is not None else _auto_candidates(n_children)

    state: dict[str, object] = {}

    def _select(x_args: tuple[Tensor, ...]) -> Callable[..., Tensor]:
        results: list[tuple[int, float]] = []
        probe_args = _stand_ins(x_args)
        for k in cand_list:
            seg_step = make_segmented_step(model, loss_fn, segments=k, dynamic=dynamic)
            results.append((k, _probe_one(model, seg_step, probe_args)))
            # Drop this candidate's executables before probing the next so the
            # probe's transient peak stays near one candidate, not their sum.
            if k != cand_list[-1]:
                seg_step.clear_cache()  # type: ignore[attr-defined]
                _C_engine.compile.session_cache_clear()
        best_k = min(results, key=lambda r: r[1])[0]
        # Rebuild the winner fresh (its cache may have been cleared above).
        chosen = make_segmented_step(model, loss_fn, segments=best_k, dynamic=dynamic)
        state["chosen_segments"] = best_k
        state["probe_results"] = results
        return chosen

    def step(*x_args: Tensor) -> Tensor:
        """Run one training step, selecting the segment count on first call."""
        from lucid.compile._core.trace_context import compile_attempt

        sel = state.get("step")
        if sel is None:
            with compile_attempt():
                sel = _select(x_args)
            state["step"] = sel
        return cast(Callable[..., Tensor], sel)(*x_args)

    step.chosen_segments = lambda: state.get("chosen_segments")  # type: ignore[attr-defined]
    step.probe_results = lambda: state.get("probe_results")  # type: ignore[attr-defined]
    return step
