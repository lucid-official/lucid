"""Custom ``Function`` calls a compiled backward cannot run.

A compiled training step traces its model, where ``Function.apply`` runs
``forward`` under ``no_grad``: the recording holds ``forward``'s own ops, and
the compiled backward differentiates those.  The gradient ``backward`` defines
never runs.  For a straight-through estimator, a reversed or scaled gradient,
or a clipped one, that is a different gradient — ``fake_quantize`` on a QAT
weight differentiated its ``round`` and handed the weight a compiled gradient
of exactly zero.

Whether a ``backward`` is the derivative of its ``forward`` cannot be told in
general, so an unmarked call is never differentiated through.  Two opt-in
mechanisms allow a ``Function`` to participate in compiled training steps:

1. **Exact-derivative marker** (:attr:`~lucid.autograd.Function._exact_derivative`):
   A trusted Lucid class whose ``backward`` equals the derivative of its
   recorded ``forward`` (for example :func:`~lucid.autograd.checkpoint`'s
   ``_CheckpointFn``) is kept when eager would also run its ``backward`` — that
   is, when one of its inputs requires grad or descends from a differentiated
   source.  When none of its inputs does and only a tensor ``forward`` closed
   over descends from a source, eager leaves the node off the graph while a
   compiled backward would differentiate through the closed-over tensor, so that
   call is refused as well.
2. **Traced custom VJP** (:attr:`~lucid.autograd.Function._traced_backward`):
   When ``_traced_backward = True`` is declared on the subclass, ``apply``
   traces ``backward(ctx, *grad_outputs)`` immediately after ``forward`` into a
   :class:`VjpRegion` on the ``TraceGraph`` (covering linear-algebra
   decompositions ``qr``, ``svd``, ``eigh``, ``cholesky``, ``pinv``,
   ``slogdet``, implicit-reparameterisation ``_StandardGamma``, QAT
   ``_FakeQuantizeAffine``, and user ``Function`` subclasses).  During backward
   emission the compiler skips automatic differentiation of the forward op range
   and emits the recorded ``VjpRegion`` directly into the single
   :class:`MPSGraphExecutable`.

``apply`` reports every call made while a tracer is installed
(:func:`note_applied`); each entry point that differentiates a recording calls
:func:`refuse_custom_backward` first, and a gradient that would pass through a
refused call makes the step run eagerly.  A forward-only compile and a Core ML
export differentiate nothing and never ask.
"""

import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, final

import lucid
from lucid._C import engine as _C_engine
from lucid._engine_scope import untraced
from lucid._tensor.tensor import Tensor
from lucid.autograd._python_node import _InputMeta, _per_input, _validate
from lucid.autograd.function import Function, FunctionCtx, _is_exact_derivative

if TYPE_CHECKING:
    from lucid._C.engine.compile import TraceGraph, Tracer, VjpRegion

__all__ = [
    "GRADIENT_DTYPES",
    "CustomBackwardInTrace",
    "note_applied",
    "refuse_custom_backward",
    "step_region_ancestors",
    "vjp_region_index",
]

#: The dtypes a gradient can flow through — eager never tracks an integer or
#: bool value.  The backward walker's ``carries_gradient`` is the same rule.
GRADIENT_DTYPES = frozenset(
    {
        _C_engine.Dtype.F16,
        _C_engine.Dtype.BF16,
        _C_engine.Dtype.F32,
        _C_engine.Dtype.F64,
        _C_engine.Dtype.C64,
        _C_engine.Dtype.C128,
    }
)


class CustomBackwardInTrace(NotImplementedError):
    """A gradient a lowering would compute passes through a custom ``Function``.

    A ``NotImplementedError``, and so a ``RuntimeError``: an entry point
    that runs eagerly when its lowering fails catches it with the rest,
    and ``compiled_step``, which has no eager step, lets it through.
    """


@final
@dataclass(frozen=True, slots=True)
class _Applied:
    """One ``Function.apply`` made while a recording was being made."""

    name: str
    # The tensors ``forward`` was given and the floating ones it returned,
    # with the ids the tracer gave them right then.  The impls are looked up
    # again at the end as well: a tensor no op has consumed yet — an input
    # handed back unchanged — has no id until one does.
    inputs: tuple[_C_engine.TensorImpl, ...]
    outputs: tuple[_C_engine.TensorImpl, ...]
    input_ids: frozenset[int]
    output_ids: frozenset[int]
    # Whether the class is a trusted Lucid ``Function`` whose ``backward`` is
    # the exact derivative of its recorded ``forward``.
    exact: bool
    # Whether any positional input required grad when ``apply`` ran — a leaf
    # parameter ``forward`` ignored, an activation in a grad-mode recording,
    # or ``checkpoint(..., use_reentrant=False)``'s anchor.
    tracked: bool
    # Whether ``backward`` was traced into ``tracer.graph.vjp_regions``.
    traced_vjp: bool = False


#: Every call noted, per tracer — gone with the tracer.
_APPLIED: weakref.WeakKeyDictionary[Tracer, list[_Applied]] = weakref.WeakKeyDictionary()


def _impls(values: tuple[object, ...], *, floating: bool) -> tuple[_C_engine.TensorImpl, ...]:
    """The impls of the tensors among ``values`` — only floating ones if asked."""
    return tuple(
        v._impl
        for v in values
        if isinstance(v, Tensor) and (not floating or v._impl.dtype in GRADIENT_DTYPES)
    )


def _ids(tracer: Tracer, impls: tuple[_C_engine.TensorImpl, ...]) -> frozenset[int]:
    """The trace ids ``tracer`` has for ``impls`` now; one it has none for is left out."""
    return frozenset(int(tid) for tid in map(tracer.lookup_id, impls) if tid is not None)


@final
@dataclass(frozen=True, slots=True)
class _VjpTraceSpec:
    """Validated inputs/outputs for tracing a custom ``Function.backward``."""

    tracer: Tracer
    function: type[Function]
    ctx: FunctionCtx
    raw_saved: tuple[Tensor, ...]
    args: tuple[object, ...]
    tensor_inputs: list[Tensor]
    tensor_outputs: list[Tensor]
    input_ids: list[int]
    diff_output_ids: list[int]
    diff_output_impls: list[_C_engine.TensorImpl]
    cotangent_ids: list[int]
    cotangents: list[Tensor | None]
    fwd_begin: int
    fwd_end: int


def _ensure_ctx_tensors(
    tracer: Tracer, raw_saved: tuple[Tensor, ...], extra: dict[str, object]
) -> bool:
    """Ensure every tensor saved on ``ctx`` resides on GPU and has a ``TensorId``."""
    items: list[object] = list(raw_saved)
    for val in extra.values():
        if isinstance(val, (tuple, list)):
            items.extend(val)
        else:
            items.append(val)
    for item in items:
        if isinstance(item, Tensor):
            if item._impl.device != _C_engine.Device.GPU:
                return False
            tracer.ensure_id(item._impl)
    return True


def _build_cotangents(
    tracer: Tracer, tensor_outputs: list[Tensor], marked: list[bool]
) -> tuple[list[int], list[_C_engine.TensorImpl], list[int], list[Tensor | None]] | None:
    """Mint symbolic cotangent ids for each differentiable output tensor."""
    diff_ids: list[int] = []
    diff_impls: list[_C_engine.TensorImpl] = []
    cot_ids: list[int] = []
    cotangents: list[Tensor | None] = []
    for t, is_marked in zip(tensor_outputs, marked, strict=True):
        if is_marked or t._impl.dtype not in GRADIENT_DTYPES:
            cotangents.append(None)
            continue
        tid = tracer.lookup_id(t._impl)
        if tid is None or int(tid) < 0:
            return None
        with untraced():
            cot = lucid.zeros(tuple(t.shape), dtype=t.dtype, device=t.device)
        diff_ids.append(int(tid))
        diff_impls.append(t._impl)
        cot_ids.append(int(tracer.register_cotangent(cot._impl)))
        cotangents.append(cot)
    return (diff_ids, diff_impls, cot_ids, cotangents) if diff_ids else None


def _resolve_input_ids(tracer: Tracer, tensor_inputs: list[Tensor]) -> list[int] | None:
    """Resolve traced input IDs once all inputs are on GPU with at least one grad dtype."""
    has_grad_dtype = False
    input_ids: list[int] = []
    for t in tensor_inputs:
        if t._impl.device != _C_engine.Device.GPU:
            return None
        if t._impl.dtype in GRADIENT_DTYPES:
            has_grad_dtype = True
        tid = tracer.lookup_id(t._impl)
        if tid is None or int(tid) < 0:
            return None
        input_ids.append(int(tid))
    return input_ids if has_grad_dtype else None


def _check_vjp_io(
    tracer: Tracer, args: tuple[object, ...], output: object, marked: list[bool]
) -> tuple[list[Tensor], list[Tensor], list[int]] | None:
    """Extract GPU tensor inputs/outputs and input trace IDs for custom VJP tracing."""
    settled_seq = output if isinstance(output, tuple) else (output,)
    tensor_outputs = [v for v in settled_seq if isinstance(v, Tensor)]
    tensor_inputs = [a for a in args if isinstance(a, Tensor)]
    if len(tensor_outputs) != len(marked) or not tensor_outputs or not tensor_inputs:
        return None
    for out_t in tensor_outputs:
        if out_t._impl.device != _C_engine.Device.GPU:
            return None
    input_ids = _resolve_input_ids(tracer, tensor_inputs)
    if input_ids is None:
        return None
    return tensor_inputs, tensor_outputs, input_ids


def _prepare_vjp_spec(
    tracer: Tracer,
    function: type[Function],
    args: tuple[object, ...],
    output: object,
    vjp_call: tuple[FunctionCtx, tuple[Tensor, ...], list[bool], int],
) -> _VjpTraceSpec | None:
    """Validate GPU/id invariants and build a :class:`_VjpTraceSpec`."""
    ctx, raw_saved, marked, fwd_begin = vjp_call
    fwd_end = int(tracer.op_count())
    if fwd_end <= fwd_begin or tracer.graph.unsupported:
        return None
    if not _ensure_ctx_tensors(tracer, raw_saved, ctx._extra):
        return None
    io = _check_vjp_io(tracer, args, output, marked)
    if io is None:
        return None
    tensor_inputs, tensor_outputs, input_ids = io
    cots = _build_cotangents(tracer, tensor_outputs, marked)
    if cots is None:
        return None
    return _VjpTraceSpec(
        tracer,
        function,
        ctx,
        raw_saved,
        args,
        tensor_inputs,
        tensor_outputs,
        input_ids,
        cots[0],
        cots[1],
        cots[2],
        cots[3],
        fwd_begin,
        fwd_end,
    )


def _run_vjp_backward(spec: _VjpTraceSpec) -> list[int] | None:
    """Invoke ``spec.function.backward`` under ``no_grad`` and resolve input grad ids."""
    cotangents: list[Tensor | None] = list(spec.cotangents)
    if getattr(spec.function, "_materialize_grads", True):
        for idx, (t, cot) in enumerate(zip(spec.tensor_outputs, cotangents, strict=True)):
            if cot is None:
                cotangents[idx] = lucid.zeros(tuple(t.shape), dtype=t.dtype, device=t.device)
    grads = spec.function.backward(spec.ctx, *cotangents)
    returned: list[object] = list(grads) if isinstance(grads, (list, tuple)) else [grads]
    bwd_name = f"{spec.function.__name__}Backward"
    metas = [
        _InputMeta(
            tuple(t.shape), t.dtype, t.is_complex(), t.device, t._impl.dtype in GRADIENT_DTYPES
        )
        for t in spec.tensor_inputs
    ]
    is_tensor_arg = tuple(isinstance(a, Tensor) for a in spec.args)
    per_in = _per_input(bwd_name, returned, is_tensor_arg, len(metas))
    input_grad_ids: list[int] = []
    for idx, (g, meta) in enumerate(zip(per_in, metas, strict=True)):
        checked = _validate(bwd_name, idx, g, meta)
        if checked is None:
            input_grad_ids.append(-1)
            continue
        gtid = int(spec.tracer.ensure_id(checked._impl))
        if gtid < 0:
            return None
        input_grad_ids.append(gtid)
    return input_grad_ids


_HOST_CONSTANT_OPS: frozenset[str] = frozenset(
    {
        "arange",
        "empty",
        "eye",
        "full",
        "linspace",
        "logspace",
        "ones",
        "zeros",
        "einsum",
        "meshgrid",
        "tri",
        "tril",
        "triu",
    }
)


def _valid_vjp_ops(tracer: Tracer, bwd_begin: int, writes_before: int) -> bool:
    """Whether every op recorded in ``[bwd_begin, ...)`` can be emitted on GPU."""
    if tracer.graph.unsupported or int(tracer.outside_write_count()) != writes_before:
        return False
    for node in tracer.graph.ops[bwd_begin:]:
        if not node.outputs or node.outputs[0].device != _C_engine.Device.GPU:
            return False
        if any(int(iid) < 0 for iid in node.inputs):
            return False
        if not _C_engine.compile.emitter_registered(node.name):
            if not node.inputs and node.name in _HOST_CONSTANT_OPS:
                continue
            return False
    return True


def _try_trace_custom_vjp(spec: _VjpTraceSpec) -> bool:
    """Trace ``spec.function.backward`` into ``spec.tracer.graph.vjp_regions``."""
    tracer = spec.tracer
    ctx = spec.ctx
    bwd_begin = int(tracer.op_count())
    writes_before = int(tracer.outside_write_count())
    prev_state = (
        ctx.needs_input_grad,
        list(ctx._saved_tensors),
        ctx._saved_record,
        ctx._carrier,
        bool(tracer.fold_constants),
    )
    try:
        if not any(ctx.needs_input_grad):
            ctx.needs_input_grad = tuple(
                isinstance(a, Tensor) and a._impl.dtype in GRADIENT_DTYPES for a in spec.args
            )
        ctx._saved_tensors = list(spec.raw_saved)
        ctx._saved_record = None
        ctx._carrier = None
        tracer.fold_constants = False
        with _C_engine.NoGradGuard():
            input_grad_ids = _run_vjp_backward(spec)
    except (RuntimeError, ValueError, TypeError, LookupError, ArithmeticError, AttributeError):
        tracer.discard_ops_from(bwd_begin)
        return False
    finally:
        (
            ctx.needs_input_grad,
            ctx._saved_tensors,
            ctx._saved_record,
            ctx._carrier,
            tracer.fold_constants,
        ) = prev_state

    if input_grad_ids is None or not _valid_vjp_ops(tracer, bwd_begin, writes_before):
        tracer.discard_ops_from(bwd_begin)
        return False
    tracer.commit_vjp_region(
        spec.fwd_begin,
        spec.fwd_end,
        spec.input_ids,
        spec.diff_output_ids,
        spec.diff_output_impls,
        spec.cotangent_ids,
        input_grad_ids,
        bwd_begin,
    )
    return True


def note_applied(
    tracer: Tracer,
    function: type,
    args: tuple[object, ...],
    output: object,
    *,
    vjp_call: tuple[FunctionCtx, tuple[Tensor, ...], list[bool], int] | None = None,
) -> None:
    """Record that ``function``'s ``forward`` ran while ``tracer`` recorded.

    Called by ``Function.apply`` as soon as ``forward`` returns, whatever the
    grad mode: under a compiled step's ``no_grad`` it is the only trace the
    call leaves.

    Parameters
    ----------
    tracer : Tracer
        The installed tracer.
    function : type
        The ``Function`` subclass, named in the fallback reason.
    args : tuple of object
        ``forward``'s positional arguments, the ones ``backward`` answers for.
    output : object
        What ``forward`` returned — a tensor, or a tuple that may hold
        non-tensor entries.
    vjp_call : tuple, optional
        ``(ctx, raw_saved, marked, fwd_begin)`` captured when
        ``function._traced_backward`` is enabled.
    """
    outputs = _impls(output if isinstance(output, tuple) else (output,), floating=True)
    if not outputs:
        return
    inputs = _impls(args, floating=False)
    exact = issubclass(function, Function) and _is_exact_derivative(function)
    tracked = any(isinstance(a, Tensor) and a.requires_grad for a in args)
    traced_vjp = False
    if vjp_call is not None and issubclass(function, Function):
        spec = _prepare_vjp_spec(tracer, function, args, output, vjp_call)
        if spec is not None:
            traced_vjp = _try_trace_custom_vjp(spec)
    _APPLIED.setdefault(tracer, []).append(
        _Applied(
            function.__name__,
            inputs,
            outputs,
            _ids(tracer, inputs),
            _ids(tracer, outputs),
            exact,
            tracked,
            traced_vjp,
        )
    )


def vjp_region_index(graph: TraceGraph) -> tuple[set[int], dict[int, VjpRegion]]:
    """Return ``(in_region_op_indices, region_by_last_op_index)`` for traced VJP regions in ``graph``."""
    in_region: set[int] = set()
    region_at_end: dict[int, VjpRegion] = {}
    for r in graph.vjp_regions:
        b, e = int(r.fwd_begin), int(r.fwd_end)
        if e > b:
            in_region.update(range(b, e))
            region_at_end[e - 1] = r
    return in_region, region_at_end


def step_region_ancestors(r: VjpRegion | None, needed: set[int]) -> None:
    """Propagate ``needed`` backwards across ``r`` from outputs to differentiable inputs."""
    if r is not None and any(int(om.id) in needed for om in r.output_metas):
        needed.update(
            int(iid)
            for iid, gid in zip(r.input_ids, r.input_grad_ids, strict=False)
            if int(iid) >= 0 and int(gid) >= 0
        )


# ``_ancestors`` follows every recorded op: which ops a gradient stops at on
# the way down from the loss is the differentiating backend's to decide, and a
# path kept here that no gradient takes only sends to eager a step that could
# have compiled.  ``_descendants``, however, decides whether a call's inputs
# carry a gradient from a source — which allows an ``exact`` call to stay
# compiled — so it stops at gradient sinks (``detach``, comparisons, ``argmax``)
# and non-floating outputs, matching the backward walker (``VjpEmitter.mm``).


def _ancestors(graph: TraceGraph, sink_id: int) -> set[int]:
    """Every trace id ``sink_id`` was computed from, itself included."""
    _, region_at_end = vjp_region_index(graph)
    reached = {sink_id}
    for idx in range(len(graph.ops) - 1, -1, -1):
        step_region_ancestors(region_at_end.get(idx), reached)
        node = graph.ops[idx]
        if any(int(meta.id) in reached for meta in node.outputs):
            reached.update(int(i) for i in node.inputs if int(i) >= 0)
    return reached


def _descendants(graph: TraceGraph, source_ids: list[int]) -> set[int]:
    """Every gradient-carrying trace id computed from ``source_ids``."""
    sink = _C_engine.compile.VjpRegistration.GradSink
    status = _C_engine.compile.vjp_registration_status
    in_region, region_at_end = vjp_region_index(graph)
    reached = set(source_ids)
    for idx, node in enumerate(graph.ops):
        if idx in in_region:
            r = region_at_end.get(idx)
            if r is not None and any(
                int(iid) in reached and int(gid) >= 0
                for iid, gid in zip(r.input_ids, r.input_grad_ids, strict=False)
            ):
                reached.update(int(om.id) for om in r.output_metas if om.dtype in GRADIENT_DTYPES)
            continue
        if status(node.name) == sink:
            continue
        if any(int(i) in reached for i in node.inputs):
            reached.update(int(meta.id) for meta in node.outputs if meta.dtype in GRADIENT_DTYPES)
    return reached


def refuse_custom_backward(tracer: Tracer, sink_id: int, source_ids: list[int]) -> None:
    """Refuse a lowering whose gradient would disagree with a noted call's.

    When one of a call's outputs leads to ``sink_id``:

    * If one of its inputs requires grad (``tracked``) or descends from a
      source, eager runs ``backward`` for the call.  An ``exact`` call's
      recorded ``forward`` has the same derivative as its ``backward`` (and a
      ``traced_vjp`` call has its ``backward`` traced into a ``VjpRegion``),
      so it is kept; any other call is refused.
    * Otherwise, if one of its outputs descends from a source — ``forward``
      closed over a parameter while its positional inputs carried no gradient,
      as in ``checkpoint(fn, x, use_reentrant=True)`` on an ungrad ``x`` —
      eager does not run ``backward`` at all while a compiled backward would
      differentiate the closed-over tensor, so even an ``exact`` call is
      refused.

    Parameters
    ----------
    tracer : Tracer
        A finished recording.
    sink_id : int
        Trace id of the value the lowering differentiates — the loss.
    source_ids : list of int
        Trace ids of what it differentiates with respect to — the parameters,
        and a segment's input.

    Raises
    ------
    CustomBackwardInTrace
        Naming the first such ``Function``.
    """
    applied = _APPLIED.get(tracer)
    if not applied:
        return
    ancestors = _ancestors(tracer.graph, sink_id)
    descendants = _descendants(tracer.graph, source_ids)
    for call in applied:
        if call.traced_vjp:
            continue
        outputs = call.output_ids | _ids(tracer, call.outputs)
        if ancestors.isdisjoint(outputs):
            continue
        inputs = call.input_ids | _ids(tracer, call.inputs)
        if call.tracked or not descendants.isdisjoint(inputs):
            if call.exact:
                continue
            raise CustomBackwardInTrace(
                f"autograd.Function {call.name} defines its own backward, which a "
                f"compiled backward cannot run: it would differentiate "
                f"{call.name}.forward instead"
            )
        if not descendants.isdisjoint(outputs):
            if call.exact:
                raise CustomBackwardInTrace(
                    f"autograd.Function {call.name} receives no input that requires "
                    f"grad, so eager does not run {call.name}.backward while a "
                    f"compiled backward would differentiate the tensors "
                    f"{call.name}.forward closed over"
                )
            raise CustomBackwardInTrace(
                f"autograd.Function {call.name} defines its own backward, which a "
                f"compiled backward cannot run: it would differentiate "
                f"{call.name}.forward instead"
            )
