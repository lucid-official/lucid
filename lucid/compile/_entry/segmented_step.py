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
A segment whose forward draws random values (a dropout mask, a sampler) is
recomputed from the stream position its forward started at, so the backward
differentiates the draws the forward applied and the stream ends where
eager's does (``_core/rng_replay``, LCD-896).
"""

import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, NoReturn, Sequence, cast, final, override

from lucid._C import engine as _C_engine
from lucid._factories._generator import _restore_stream_states, _stream_states
from lucid._tensor.tensor import Tensor
from lucid.autograd.function import Function, FunctionCtx
from lucid.compile._core.bn_runstats import BnWriteback
from lucid.compile._core.buffer_writes import TraceAnswer, eager_rerun
from lucid.compile._core.call_values import CallValues
from lucid.compile._core.fallback import EagerFallbackSet, run_or_answer_eagerly
from lucid.compile._core.rng_replay import advance_since, replayed
from lucid.compile._core.signature import CacheKey, outside_feeds, tree_tensors

if TYPE_CHECKING:
    from lucid._C.engine.compile import CompiledExecutable, Tracer
    from lucid._factories._generator import _StreamState
    from lucid.compile._core.rebound_slots import SlotWatch
    from lucid.compile._core.signature import ParamFingerprint
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
    # Per gradient the backward executable returns (the input's, if
    # ``produce_input_grad``, then each parameter's): does the segment's loss
    # depend on its target?  A gradient it does not reach comes back as zeros,
    # and eager leaves that one ``None`` (LCD-899).
    reached: tuple[bool, ...]
    # The BatchNorm statistics the forward executable returns after its
    # output, and the counters, written back after each forward run.  The
    # backward recomputes the forward and writes nothing: one step, one update.
    bn: BnWriteback = field(default_factory=BnWriteback)
    # Every tensor the traces were fed from outside, for ``ParamFingerprint.settle``.
    fed: tuple[_C_engine.TensorImpl, ...] = ()
    # Whether the segment draws random values.  Both executables draw them anew
    # on every run, so the backward replays the forward's draws (LCD-896).
    draws: bool = False
    # Whether the forward executable has run to the end once: MPSGraph specializes it
    # in that first run, and a failure there is a refusal (:func:`_refuse_first_run`).
    warm: bool = False
    # The same for the backward executable, whose first run happens inside the
    # caller's ``loss.backward()`` (:func:`_SegmentFunction.backward`).
    bwd_warm: bool = False


@final
@dataclass(frozen=True, slots=True)
class _SegCall:
    """What one segment's autograd node needs beyond its tensors.

    ``mark_eager`` makes the call's signature eager-only: a backward executable
    that fails in its first run is the call's verdict, and it is found out
    after the forward chain has run, so the step that raised could not take it
    back.  ``owner`` is the model, whose buffers an eager recompute puts back, and
    ``slot_watch`` takes the watch that puts back the rest of what its Python does
    (:meth:`ParamFingerprint.slot_watch`).
    """

    compiled: _SegCompiled
    seg: _Segment
    owner: Module
    mark_eager: Callable[[], None]
    slot_watch: Callable[[Iterable[Tensor]], SlotWatch]


def _segment_key(x: Tensor, fingerprint: ParamFingerprint) -> CacheKey:
    """Compile-cache key for a segment fed ``x``: every entry point's key.

    That is the input's shape, dtype and grad need — the first segment
    returned no gradient to an input that did, and an eager trunk before a
    compiled model trained nothing (LCD-617) — whether the input is a tensor
    the model holds (LCD-629), and the state of every module of the model
    and of the call site: keyed on the input alone, a segment traced in
    training replayed after ``model.eval()`` (LCD-720).
    """
    return fingerprint.key_of((x,), {}, dynamic=False)


def _feed_id(ext: dict[int, object], impl: object) -> int | None:
    """The trace feed id of ``impl`` (by pointer identity), or ``None``."""
    return next((tid for tid, held in ext.items() if held is impl), None)


def _match_param_ids(ext: dict[int, object], param_impls: Sequence[object]) -> list[int] | None:
    """Map each parameter impl to its trace feed id (by pointer identity).

    Returns ``None`` if any parameter was not observed in the trace (segment
    forward did not use it) → the step falls back to eager.
    """
    found = [_feed_id(ext, p_impl) for p_impl in param_impls]
    return None if None in found else [tid for tid in found if tid is not None]


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
    cache: dict[CacheKey, _SegCompiled] = field(default_factory=dict)


def _trace_segment_fn(
    watch: tuple[object, ...], fn: Callable[[], Tensor], owner: Module | None = None
) -> tuple[Tracer, Tensor] | None:
    """Trace *fn* in the caller's grad mode, returning ``None`` on a tensor hook or empty graph.

    The caller decides what becomes of the writes the trace made
    (``_core/buffer_writes``).
    """
    from lucid.compile._core.attention_probe import maybe_probe_for_graph
    from lucid.compile._core.buffer_writes import completes_on_raise
    from lucid.compile._core.trace_context import notice_if_tensor_hook_error, recording

    try:
        with recording(watch=watch) as tracer, completes_on_raise(tracer, owner):
            out = fn()
    except RuntimeError as why:
        if not notice_if_tensor_hook_error(why):
            raise
        return None
    if not tracer.graph.ops:
        return None
    maybe_probe_for_graph(tracer.graph)
    return tracer, out


def _trace_segment_forward(
    seg: _Segment, x: Tensor, owner: Module
) -> tuple[Tracer, Tensor] | TraceAnswer[Tensor] | None:
    """Trace ``seg``'s forward at ``x``; the trace-time call stands as the
    segment's call when it wrote a tensor it made and Python keeps.

    Such a tensor, built once and kept or built anew every call and kept, looks
    the same from inside one trace (``held_writes``).  Compiled, a buffer filled
    from the first batch would be filled again from every batch; put back for
    an eager rerun, one the rerun finds built would never be written again
    (LCD-876).  The trace ran the forward once in the caller's grad mode, so
    its result, with its autograd graph, is the segment's: the step runs the
    rest of this call and every later call eagerly.
    """
    from lucid._dispatch import _unwrap, _wrap
    from lucid.compile._core.buffer_writes import held_writes, refuse, stood_as_the_call

    traced = _trace_segment_fn((x, seg.state), lambda: seg.forward_fn(x), owner)
    if traced is None:
        return None
    tracer, y = traced
    found = tracer.read_id(_unwrap(y))
    # Held in C++ while ``held_writes`` asks who else holds a tensor the call
    # made: the output is made anew on every run.
    tracer.keep_answer([] if found is None else [_unwrap(y)])
    del traced, y
    held = held_writes(tracer, stands=found is not None)
    if held and found is not None:
        return stood_as_the_call(tracer, owner, _wrap(tracer.take_answer()[0]))
    refuse(tracer, held)
    answer = tracer.take_answer()
    return (tracer, _wrap(answer[0])) if answer else None


def _trace_segment_backward(
    seg: _Segment, x: Tensor, go: Tensor, drew: dict[_C_engine.Generator, int]
) -> Tracer | None:
    """Trace ``sum(seg.forward_fn(x) * go)``, the forward again: whatever it
    writes was the forward trace's to stand for, so a write here refuses it.

    ``drew`` is what the forward trace drew from each stream.  The backward
    replays the forward's draws from where it started, which is the forward's
    only if it draws as much: otherwise the step refuses.
    """
    from lucid.compile._core.buffer_writes import held_writes, refuse
    from lucid.compile._core.fallback import _fallback_notice

    before = _stream_states()
    traced = _trace_segment_fn((x, seg.state), lambda: (seg.forward_fn(x) * go).sum())
    if traced is None:
        return None
    if advance_since(before) != drew:
        _fallback_notice(
            "rng-draws",
            f"segment {seg.index} draws a different number of random values in its backward",
        )
        return None
    refuse(traced[0], held_writes(traced[0], stands=False))
    return traced[0]


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
    tracer: Tracer, y_id: int, grad_target_ids: list[int], stats: Sequence[int]
) -> CompiledExecutable | None:
    """A segment's forward executable, returning its output and then the
    BatchNorm ``stats``; ``None`` when it cannot be compiled."""
    from lucid.compile._core.custom_backward import CustomBackwardInTrace, refuse_custom_backward
    from lucid.compile._core.fallback import _fallback_notice

    try:
        refuse_custom_backward(tracer, y_id, grad_target_ids)
        return _C_engine.compile.compile_or_cached(
            tracer.graph, dict(tracer.external_feeds), False, [], [y_id, *stats]
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


def _read_another_segments_value(values: CallValues, seg: _Segment, tracer: Tracer) -> bool:
    """Whether ``seg``'s trace read a value an earlier segment made in this call,
    outside its arguments; the step then runs eagerly (LCD-902).

    The value is a constant of the executable, and whatever stored it — a
    stash, an attribute, a closure cell — ran only while its segment was
    traced: every compiled call would read the first call's.  A segment's own
    values go on to its backward trace, and a value from before the call is the
    caller's to rebind (LCD-903).
    """
    from lucid.compile._core.fallback import _fallback_notice

    maker = values.foreign_read(seg.index, tracer)
    if maker is None:
        return False
    _fallback_notice(
        "segment-values",
        f"segment {seg.index} reads a value segment {maker} computed in this call, "
        "outside the segment's arguments",
    )
    return True


@final
@dataclass(frozen=True, slots=True)
class _SegBackward:
    """A segment's backward executable, whether it returns the input's
    cotangent, and what its trace was fed from outside."""

    exe: _SegExe
    produce_input_grad: bool
    reached: tuple[bool, ...]
    fed: tuple[_C_engine.TensorImpl, ...]


def _compile_segment_backward(
    seg: _Segment, x: Tensor, y: Tensor, drew: dict[_C_engine.Generator, int], values: CallValues
) -> _SegBackward | None:
    """Trace ``loss_seg = sum(y * grad_out)`` for ``seg`` and compile it, or
    ``None`` when it cannot be compiled.

    ``grad_out`` is a fresh leaf feed shaped like ``y``; the backward feeds the
    upstream cotangent into this slot.
    """
    import lucid
    from lucid._dispatch import _unwrap

    go = lucid.zeros(y.shape, dtype=y.dtype, device=y.device)
    t_bwd = _trace_segment_backward(seg, x, go, drew)
    if t_bwd is None or _read_another_segments_value(values, seg, t_bwd):
        return None
    values.made_by(seg.index, t_bwd)
    ext_bwd = t_bwd.external_feeds
    loss_id = t_bwd.graph.ops[-1].outputs[0].id

    x_id = _feed_id(ext_bwd, _unwrap(x))
    go_id = _feed_id(ext_bwd, _unwrap(go))
    if x_id is None or go_id is None:
        return None

    param_ids = _match_param_ids(ext_bwd, [_unwrap(p) for p in seg.params])
    if param_ids is None:
        return None

    grad_target_ids = _grad_targets(x, x_id, param_ids)
    exe = _lower_segment_backward(t_bwd, int(loss_id), grad_target_ids)
    if exe is None:
        return None

    plan = _build_feed_plan(
        list(exe.input_ids), ext_bwd, x_id, go_id, {pid: i for i, pid in enumerate(param_ids)}
    )
    if plan is None:
        return None
    return _SegBackward(
        exe=_SegExe(exe=exe, feed_plan=plan),
        produce_input_grad=len(grad_target_ids) > len(param_ids),
        reached=tuple(int(g) != -1 for g in exe.grad_output_ids),
        fed=outside_feeds(t_bwd),
    )


def _compile_segment(
    seg: _Segment, x: Tensor, owner: Module, values: CallValues
) -> _SegCompiled | TraceAnswer[Tensor] | None:
    """Trace + compile the forward and backward executables for ``seg`` at ``x``.

    Two traces are taken (forward-only, then forward + ``loss_seg``) so the
    forward executable does not carry the cotangent-product tail.  The second
    runs the forward's Python again, from where the call began: the module is
    put back between them, so what the Python does to it happens once.  ``owner``
    is the model the segment is part of, whose BatchNorm counters it advances.
    Returns ``None`` on any uncompilable condition → the caller falls back to
    eager; the forward trace's output when it stands as the segment's call.
    """
    from lucid._dispatch import _unwrap
    from lucid.compile._core.buffer_writes import restore_slots

    param_impls = [_unwrap(p) for p in seg.params]
    x_impl = _unwrap(x)

    # --- Trace 1: forward only (x -> y) -----------------------------------
    rng_before = _stream_states()
    traced_fwd = _trace_segment_forward(seg, x, owner)
    drew = advance_since(rng_before)
    if not isinstance(traced_fwd, tuple):
        return traced_fwd
    t_fwd, y = traced_fwd
    if _read_another_segments_value(values, seg, t_fwd):
        return None
    values.made_by(seg.index, t_fwd)
    ext_fwd = t_fwd.external_feeds
    # The segment's output, not whatever op it recorded last.
    found = t_fwd.read_id(_unwrap(y))
    if found is None:
        return None
    y_id = int(found)

    x_id_fwd = _feed_id(ext_fwd, x_impl)
    if x_id_fwd is None:
        return None

    fwd_param_ids = _match_param_ids(ext_fwd, param_impls)
    if fwd_param_ids is None:
        return None

    bn = BnWriteback.of(owner, t_fwd)
    fwd_exe = _lower_segment_forward(
        t_fwd, y_id, _grad_targets(x, x_id_fwd, fwd_param_ids), bn.out_ids
    )
    if fwd_exe is None:
        return None

    fwd_pid_to_idx = {pid: i for i, pid in enumerate(fwd_param_ids)}
    fwd_plan = _build_feed_plan(list(fwd_exe.input_ids), ext_fwd, x_id_fwd, None, fwd_pid_to_idx)
    if fwd_plan is None:
        return None

    # The backward trace runs the forward's Python again.  From where the call began:
    # both traces then take the same path, and the first run's effects (a counter, a
    # stamp) stand once, as the one call's.  Left standing, the second run found the
    # work done and moved them twice (LCD-992).
    restore_slots(grads=False)
    backward = _compile_segment_backward(seg, x, y, drew, values)
    if backward is None:
        return None
    return _SegCompiled(
        fwd=_SegExe(exe=fwd_exe, feed_plan=fwd_plan),
        bwd=backward.exe,
        produce_input_grad=backward.produce_input_grad,
        n_params=len(seg.params),
        reached=backward.reached,
        bn=bn,
        fed=outside_feeds(t_fwd) + backward.fed,
        draws=bool(drew),
    )


def _compile_into(
    seg: _Segment, x: Tensor, fingerprint: ParamFingerprint, key: CacheKey, values: CallValues
) -> _SegCompiled | TraceAnswer[Tensor]:
    """Compile ``seg`` at ``x`` into its cache, telling ``fingerprint`` what the
    traces read; raise :class:`_SegmentCompileError` if it does not compile.

    ``key`` is the segment's key taken before the traces: what a forward does
    to its own state while traced does not move it.  A model with a
    cumulative-average BatchNorm runs eagerly, as the monolithic step does:
    that update reads the counter on the host.  A forward trace that stands
    as the segment's call is answered, not cached; its draws were the call's.
    ``values`` is what the call's earlier segments made, and ``x`` is handed to
    ``seg`` unless the model holds it: then it is the model's own state.
    """
    from lucid._dispatch import _unwrap
    from lucid.compile._core.bn_runstats import model_has_cumulative_bn
    from lucid.compile._core.trace_context import compile_attempt

    owner = fingerprint.model
    if not fingerprint.tree_alias((x,), {}):
        values.hand(seg.index, (_unwrap(x),))
    with compile_attempt() as attempt:
        before = fingerprint.outside_slots((x,))
        compiled = (
            None if model_has_cumulative_bn(owner) else _compile_segment(seg, x, owner, values)
        )
        attempt.concluded(compiled)
        fingerprint.settle(compiled.fed if isinstance(compiled, _SegCompiled) else (), before)
    if compiled is None:
        raise _SegmentCompileError(seg.index)
    if isinstance(compiled, _SegCompiled):
        seg.cache[fingerprint.rekey(key, (x,), {})] = compiled
    return compiled


def _refuse_first_run(
    seg: _Segment, compiled: _SegCompiled, why: RuntimeError, undoable: bool
) -> NoReturn:
    """Turn a failed first run of ``seg``'s forward into a refusal, or re-raise ``why``.

    Nothing of ``seg`` was written (its BatchNorm write-back follows a run that
    succeeded), but the segments before it have written theirs.  The eager step
    that answers instead writes them again, so it is only taken when they can be
    put back: ``undoable`` — the call kept the buffers, or ``seg`` is the first.
    Otherwise the failure raises: a silent double update is worse than an error.

    Raises
    ------
    _SegmentCompileError
        ``why`` is a first-run failure and the call can be undone; the executable
        is dropped so it is not found again.
    RuntimeError
        ``why`` itself, otherwise.
    """
    from lucid.compile._core.fallback import _fallback_notice, is_first_run_failure

    if compiled.warm or not undoable or not is_first_run_failure(why):
        raise why
    _fallback_notice("first-run", str(why))
    _forget(seg, compiled)
    raise _SegmentCompileError(seg.index) from why


def _forget(seg: _Segment, compiled: _SegCompiled) -> None:
    """Drop ``compiled`` from ``seg``'s cache, so no later call finds it."""
    for key in [k for k, v in seg.cache.items() if v is compiled]:
        del seg.cache[key]


def _forward_segments(
    seg_objs: Sequence[_Segment],
    cur: Tensor,
    fingerprint: ParamFingerprint,
    key: CacheKey,
    mark_eager: Callable[[], None],
) -> tuple[Tensor, bool]:
    """Run (and lazily compile) each segment in ``seg_objs`` on ``cur``.

    ``key`` is ``cur``'s segment key, which the caller took to look the call up;
    ``mark_eager`` is what each segment's backward calls if its executable is
    refused (:class:`_SegCall`).

    Each segment run writes its BatchNorm state back, so a later segment that
    does not compile would leave the step half-applied, and the eager step
    that replaces it applies it again — statistics and counters moved twice.
    A call compiles when its first segment has no executable for its input
    (every later input follows from that one and the module state, both in
    the key), so that is when the buffers and the module's slots are kept, to
    be put back: the Python each segment's trace ran goes back with them.

    Returns the last segment's output, and whether a segment's forward trace
    stood as its call (:func:`_trace_segment_forward`).  That ends the
    compiled part of the step: the segments after it run eagerly on the
    trace's output, every segment ran once and nothing is put back, and the
    caller marks the call's key eager, as every entry point does after a
    trace stood.
    """
    compiling = key not in seg_objs[0].cache
    restore = _keep_buffers(fingerprint.model) if compiling else None
    # Past every segment's own attempt: what each ran in Python while traced goes back too.
    slots = fingerprint.slot_watch((cur,)) if compiling else None
    values = CallValues()
    try:
        for i, seg in enumerate(seg_objs):
            if i:
                key = _segment_key(cur, fingerprint)
            compiled = seg.cache.get(key)
            if compiled is None:
                built = _compile_into(seg, cur, fingerprint, key, values)
                if isinstance(built, TraceAnswer):
                    return _run_eagerly(seg_objs[i + 1 :], built.value), True
                compiled = built
            holder: list[object] = [
                _SegCall(compiled, seg, fingerprint.model, mark_eager, fingerprint.slot_watch)
            ]
            try:
                cur = cast(Tensor, _SegmentFunction.apply(holder, cur, *seg.params))
            except RuntimeError as why:
                _refuse_first_run(seg, compiled, why, restore is not None or i == 0)
    except _SegmentCompileError:
        if restore is not None:
            restore()
        if slots is not None:
            slots.restore()
        raise
    return cur, False


def _forward_or_refuse(
    seg_objs: Sequence[_Segment],
    x: Tensor,
    fingerprint: ParamFingerprint,
    key: CacheKey,
    eager_only: EagerFallbackSet,
) -> Tensor | None:
    """The last segment's output for ``x``, or ``None`` when the call must run eagerly.

    Every segment of the input chain compiles on the first call: the forward
    executables must run to know the next segment's input shape, so compile and
    forward are interleaved, each segment an autograd node that ``loss.backward()``
    unwinds in reverse.  A refused compile, or a forward trace that stood as its
    segment's call, marks ``key`` eager: the key as the traces left it, which a
    call that changes what the model reads (a slot it stored into) no longer has.
    """
    rng_before = _stream_states()

    def mark_eager() -> None:
        eager_only.add(fingerprint.rekey(key, (x,), {}))

    try:
        cur, stood = _forward_segments(seg_objs, x, fingerprint, key, mark_eager)
    except _SegmentCompileError:
        _restore_stream_states(rng_before)
        eager_only.add(fingerprint.rekey(key, (x,), {}))
        return None
    if stood:
        eager_only.add(fingerprint.rekey(key, (x,), {}))
    return cur


def _run_eagerly(seg_objs: Sequence[_Segment], cur: Tensor) -> Tensor:
    """``cur`` through each segment of ``seg_objs``, eagerly."""
    for seg in seg_objs:
        cur = seg.forward_fn(cur)
    return cur


def _eager_segment_grads(
    call: _SegCall,
    x: Tensor,
    param_impls: Sequence[object],
    grad_out: Tensor,
    rng: tuple[_StreamState, ...] | None,
) -> tuple[Tensor | None, ...]:
    """``call``'s segment differentiated eagerly: ``(grad_x, *grad_params)``.

    The forward ran compiled and wrote its BatchNorm statistics once, so the
    eager recompute puts the buffers back as it found them; its draws are the
    forward's (``rng``).  The input is a detached copy, so the recompute never
    runs on into whatever computed it: an earlier segment's own backward does
    that.  An input that is one of the segment's parameters is that parameter,
    whose gradient then counts both uses, as in the executable (LCD-629).  The forward's
    Python runs once more here, and what it does to the module (a counter, a stamp, a
    cache) goes back with the buffers: the compiled forward was the call's run (LCD-992).
    """
    from lucid._dispatch import _unwrap
    from lucid.autograd import enable_grad, grad

    seg = call.seg
    restore = _keep_buffers(call.owner)
    slots = call.slot_watch((x,))
    x_impl = _unwrap(x)
    as_param = next(
        (p for p, impl in zip(seg.params, param_impls, strict=True) if impl is x_impl), None
    )
    wants_x = call.compiled.produce_input_grad
    found: dict[int, Tensor | None] = {}
    try:
        with replayed(rng), enable_grad():
            x_in = as_param if as_param is not None else x.detach().requires_grad_(wants_x)
            out = seg.forward_fn(x_in)
            wrt = [p for p in seg.params if p.requires_grad]
            if wants_x:
                wrt.insert(0, x_in)
            if wrt:
                found = {
                    id(t): g
                    for t, g in zip(
                        wrt, grad(out, wrt, grad_outputs=grad_out, allow_unused=True), strict=True
                    )
                }
    finally:
        restore()
        slots.restore(grads=False)  # ``grad`` is functional: nothing reached ``.grad``
    return (found.get(id(x_in)) if wants_x else None, *(found.get(id(p)) for p in seg.params))


@final
class _SegmentFunction(Function):
    """autograd.Function wrapping one compiled segment.

    ``apply(holder, x, *params)`` — ``holder`` is a 1-element list owning the
    segment's :class:`_SegCall` (a non-Tensor positional, given no autograd
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

        call = cast(_SegCall, holder[0])
        compiled = call.compiled
        param_impls = [_unwrap(p) for p in params]
        feeds = _feed(compiled.fwd.feed_plan, _unwrap(x), None, param_impls)
        # Where the streams stand before the forward draws, for the backward's replay.
        ctx.rng = _stream_states() if compiled.draws else None
        outs = _C_engine.compile.run_executable(compiled.fwd.exe, feeds)
        y_impl = cast(_C_engine.TensorImpl, outs[0])
        compiled.bn.apply(outs[1:])
        compiled.warm = True

        ctx.call = call
        ctx.x_impl = _unwrap(x)
        ctx.param_impls = param_impls
        return _wrap(y_impl)

    # backward returns tuple[Tensor | None, ...], one per Tensor input (x then
    # params); the base returns Tensor | tuple.
    @override
    @staticmethod
    def backward(ctx: FunctionCtx, grad_out: Tensor) -> tuple[Tensor | None, ...]:
        """Run the segment's backward executable; return (grad_x, *grad_params)."""
        from lucid._dispatch import _unwrap, _wrap

        call = cast(_SegCall, ctx.call)
        compiled = call.compiled
        x_impl = cast(_C_engine.TensorImpl, ctx.x_impl)
        param_impls = cast(list[object], ctx.param_impls)
        # The recompute draws the values the forward applied, and the streams
        # end where they stand: eager's backward draws nothing.
        forward_rng = ctx.rng
        rng = forward_rng if isinstance(forward_rng, tuple) else None

        def run() -> tuple[Tensor | None, ...]:
            feeds = _feed(compiled.bwd.feed_plan, x_impl, _unwrap(grad_out), param_impls)
            with replayed(rng):
                outs = _C_engine.compile.run_executable(compiled.bwd.exe, feeds)
            compiled.bwd_warm = True
            # Output layout: [loss_seg, (grad_x?), *grad_params]; a gradient the
            # loss does not reach is a zero slot, which eager leaves None.
            grads: list[Tensor | None] = [
                _wrap(cast(_C_engine.TensorImpl, out)) if reached else None
                for out, reached in zip(outs[1:], compiled.reached, strict=True)
            ]
            if not compiled.produce_input_grad:
                grads.insert(0, None)
            return tuple(grads)

        def drop() -> None:
            _forget(call.seg, compiled)
            call.mark_eager()

        # The backward executable's first run happens here, in the caller's
        # ``loss.backward()``, and writes nothing: a refusal is answered with
        # this segment's gradients taken eagerly, from the same inputs.
        return run_or_answer_eagerly(
            run,
            warm=compiled.bwd_warm,
            drop=drop,
            eagerly=lambda: _eager_segment_grads(call, _wrap(x_impl), param_impls, grad_out, rng),
        )


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
        hooks_decline,
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

    # The signatures that run eagerly.  A refusal belongs to the call's key, which
    # moves when the call changes what the model reads (a slot it stored into now
    # holds a tensor), as the monolithic step's does: the step is not eager for good.
    eager_only = EagerFallbackSet()

    def _run_eager(x_args: tuple[Tensor, ...]) -> Tensor:
        with eager_rerun():
            return loss_fn(model(*x_args[:1]), *x_args[1:])

    def step(*x_args: Tensor) -> Tensor:
        """Run one segmented compiled training step (or eager fallback)."""
        # Taking the key rebuilds the segments if the model's were replaced.
        key = _segment_key(x_args[0], fingerprint)
        if (
            key in eager_only
            or hooks_decline(model, "make_step(segments>1)")
            or tensors_have_active_hooks(leaf_tensors(x_args, {}))
        ):
            return _run_eager(x_args)

        cur = _forward_or_refuse(seg_objs, x_args[0], fingerprint, key, eager_only)
        if cur is None:
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
        eager_only.clear()

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


def _keep_buffers(model: Module) -> Callable[[], None]:
    """What puts ``model``'s buffers back as they are now, in place.

    For runs that are not the step the caller asked for: the segments a
    call ran before one failed to compile (:func:`_forward_segments`), and
    the timing runs of the ``segments="auto"`` probe — each a training step
    that moved the BatchNorm statistics and counters.
    """
    from lucid.autograd._grad_mode import no_grad

    kept = [(b, b.detach().clone()) for b in model.buffers()]

    def restore() -> None:
        with no_grad():
            for buffer, value in kept:
                buffer.copy_(value)

    return restore


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
            'make_step(segments="auto"): model has fewer than 2 child modules to split'
        )
    cand_list = candidates if candidates is not None else _auto_candidates(n_children)

    state: dict[str, object] = {}

    def _select(x_args: tuple[Tensor, ...]) -> Callable[..., Tensor]:
        results: list[tuple[int, float]] = []
        probe_args = _stand_ins(x_args)
        restore = _keep_buffers(model)
        try:
            for k in cand_list:
                seg_step = make_segmented_step(model, loss_fn, segments=k, dynamic=dynamic)
                results.append((k, _probe_one(model, seg_step, probe_args)))
                # Drop this candidate's executables before probing the next so the
                # probe's transient peak stays near one candidate, not their sum.
                if k != cand_list[-1]:
                    seg_step.clear_cache()  # type: ignore[attr-defined]
                    _C_engine.compile.session_cache_clear()
        finally:
            restore()
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
