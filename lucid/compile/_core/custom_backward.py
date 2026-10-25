"""Custom ``Function`` calls a compiled backward cannot run.

A compiled training step traces its model, where ``Function.apply`` runs
``forward`` under ``no_grad``: the recording holds ``forward``'s own ops, and
the compiled backward differentiates those.  The gradient ``backward`` defines
never runs.  For a straight-through estimator, a reversed or scaled gradient,
or a clipped one, that is a different gradient — ``fake_quantize`` on a QAT
weight differentiated its ``round`` and handed the weight a compiled gradient
of exactly zero.

Whether a ``backward`` is the derivative of its ``forward`` cannot be told in
general, so an unmarked call is never differentiated through.  A Lucid class
whose ``backward`` is the exact derivative of its recorded ``forward`` may set
:attr:`~lucid.autograd.Function._exact_derivative` (for example
:func:`~lucid.autograd.checkpoint`'s ``_CheckpointFn``), and a call to it is
kept when eager would also run its ``backward`` — that is, when one of its
inputs requires grad or descends from a differentiated source.  When none of
its inputs does and only a tensor ``forward`` closed over descends from a
source, eager leaves the node off the graph while a compiled backward would
differentiate through the closed-over tensor, so that call is refused as well.

``apply`` reports every call made while a tracer is installed
(:func:`note_applied`); each entry point that differentiates a recording calls
:func:`refuse_custom_backward` first, and a gradient that would pass through a
refused call makes the step run eagerly.  A forward-only compile and a Core ML
export differentiate nothing and never ask.
"""

import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, final

from lucid._C import engine as _C_engine
from lucid._tensor.tensor import Tensor
from lucid.autograd.function import Function, _is_exact_derivative

if TYPE_CHECKING:
    from lucid._C.engine.compile import TraceGraph, Tracer

__all__ = [
    "GRADIENT_DTYPES",
    "CustomBackwardInTrace",
    "refuse_custom_backward",
    "note_applied",
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


#: Every call noted, per tracer — gone with the tracer.
_APPLIED: weakref.WeakKeyDictionary[Tracer, list[_Applied]] = (
    weakref.WeakKeyDictionary()
)


def _impls(
    values: tuple[object, ...], *, floating: bool
) -> tuple[_C_engine.TensorImpl, ...]:
    """The impls of the tensors among ``values`` — only floating ones if asked."""
    return tuple(
        v._impl
        for v in values
        if isinstance(v, Tensor) and (not floating or v._impl.dtype in GRADIENT_DTYPES)
    )


def _ids(tracer: Tracer, impls: tuple[_C_engine.TensorImpl, ...]) -> frozenset[int]:
    """The trace ids ``tracer`` has for ``impls`` now; one it has none for is left out."""
    return frozenset(
        int(tid) for tid in map(tracer.lookup_id, impls) if tid is not None
    )


def note_applied(
    tracer: Tracer, function: type, args: tuple[object, ...], output: object
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
    """
    outputs = _impls(output if isinstance(output, tuple) else (output,), floating=True)
    if not outputs:
        return
    inputs = _impls(args, floating=False)
    exact = issubclass(function, Function) and _is_exact_derivative(function)
    tracked = any(isinstance(a, Tensor) and a.requires_grad for a in args)
    _APPLIED.setdefault(tracer, []).append(
        _Applied(
            function.__name__,
            inputs,
            outputs,
            _ids(tracer, inputs),
            _ids(tracer, outputs),
            exact,
            tracked,
        )
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
    reached = {sink_id}
    for node in reversed(graph.ops):
        if any(int(meta.id) in reached for meta in node.outputs):
            reached.update(int(i) for i in node.inputs if int(i) >= 0)
    return reached


def _descendants(graph: TraceGraph, source_ids: list[int]) -> set[int]:
    """Every gradient-carrying trace id computed from ``source_ids``."""
    sink = _C_engine.compile.VjpRegistration.GradSink
    status = _C_engine.compile.vjp_registration_status
    reached = set(source_ids)
    for node in graph.ops:
        if status(node.name) == sink:
            continue
        if any(int(i) in reached for i in node.inputs):
            reached.update(
                int(meta.id) for meta in node.outputs if meta.dtype in GRADIENT_DTYPES
            )
    return reached


def refuse_custom_backward(tracer: Tracer, sink_id: int, source_ids: list[int]) -> None:
    """Refuse a lowering whose gradient would disagree with a noted call's.

    When one of a call's outputs leads to ``sink_id``:

    * If one of its inputs requires grad (``tracked``) or descends from a
      source, eager runs ``backward`` for the call.  An ``exact`` call's
      recorded ``forward`` has the same derivative as its ``backward``, so it
      is kept; any other call is refused.
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
