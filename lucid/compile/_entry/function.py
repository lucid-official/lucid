"""
lucid.compile._entry.function — single-step compile+run helper for Phase 1.3.

The Plan's `CompiledStepFunction(autograd.Function)` design routes the
compiled fwd+bwd through Lucid's autograd graph so a ``loss.backward()``
call replays our cached executable.  That full integration lives with
Phase 1.4's ``CompiledModule`` (cache + ownership semantics + recompile
triggers all in one place).

For Phase 1.3's acceptance gate (ResNet-18 + CE + SGD, 5 steps via
compile, param drift < 5e-3) the minimal surface is a direct
``compiled_step(model, x, loss_fn)`` function that:

  1. Runs ``model(x); loss_fn(out)`` once under ``_tracing()`` to record
     the op DAG.
  2. Calls ``compile_trace_with_backward`` to get a single executable
     that produces ``loss`` + the gradients of the parameters and of
     ``x`` (when it requires grad) in one shot.
  3. Runs that executable.
  4. Hands those gradients to the autograd engine as seeds, which adds
     them into ``.grad`` exactly as ``loss.backward()`` would, so
     downstream ``optimizer.step()`` works like an eager training step.

The loss returned carries no graph: the backward has already run.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

from lucid._C import engine as _C_engine
from lucid._tensor.tensor import Tensor
from lucid.compile._core.custom_backward import refuse_custom_backward
from lucid.compile._entry.step import (
    _ExecutableLike,
    _grad_slots,
    _input_grad_targets,
    _match_param_ids,
)

if TYPE_CHECKING:
    from lucid.nn.module import Module
    from lucid.nn.parameter import Parameter

__all__ = ["compiled_step"]


def _accumulate_grads(
    exe: _ExecutableLike,
    grads: list[_C_engine.TensorImpl],
    params: list[Parameter],
    leaves: list[Tensor],
    input_positions: list[int],
) -> None:
    """Add ``exe``'s gradients into ``.grad`` the way ``loss.backward()`` does.

    Each gradient seeds the autograd engine at its tensor, so the engine's
    rule holds: a gradient already there is added to in place (a held
    ``p.grad`` sees the sum); a parameter that does not require grad, or
    that the loss does not depend on, gets none; and an input computed by
    eager ops passes its gradient on to the graph behind it.  Assigning
    ``p.grad`` instead dropped every earlier gradient and never reached
    the input (LCD-617).
    """
    from lucid._dispatch import _wrap
    from lucid.autograd import backward

    absent, input_grads = _grad_slots(exe, len(params), input_positions)
    roots: list[Tensor] = []
    seeds: list[Tensor] = []
    for i, (p, g) in enumerate(zip(params, grads[: len(params)], strict=True)):
        if p.requires_grad and not (absent and absent[i]):
            roots.append(p)
            seeds.append(_wrap(g))
    for pos, g in zip(input_grads, grads[len(params) :], strict=True):
        if pos is not None:
            roots.append(leaves[pos])
            seeds.append(_wrap(g))
    if roots:
        backward(roots, seeds)


def compiled_step(
    model: Module,
    x: Tensor,
    loss_fn: Callable[[Tensor], Tensor],
) -> Tensor:
    """Run one compiled training step: forward + backward + grad accumulation.

    Parameters
    ----------
    model : nn.Module
        Module whose ``forward(x)`` produces the model output.
    x : Tensor
        Input tensor (single forward pass).
    loss_fn : callable
        Maps ``model(x)`` → scalar loss tensor.  Typical: ``lambda y: F.cross_entropy(y, target)``.

    Returns
    -------
    Tensor
        The loss (compiled result, with no graph: its backward has run).
        Its gradients are added into ``.grad`` as ``loss.backward()`` adds
        them — of each parameter that requires grad, and of ``x`` and the
        graph behind it when ``x`` requires grad.  Then run the optimizer.

    Raises
    ------
    NotImplementedError
        A gradient passes through a custom ``Function``, whose ``backward``
        the executable cannot run (:func:`make_step` runs such a step eagerly).

    Notes
    -----
    The trace is captured in the caller's grad mode and fresh on every call;
    cache integration (compile-or-cached) lives in Phase 1.4's
    ``CompiledModule``.

    Phase 1.3 acceptance: ``ResNet-18 + CE + SGD, 5 steps``, param
    drift < 5e-3 vs eager.

    Examples
    --------
    >>> import lucid, lucid.nn as nn, lucid.nn.functional as F
    >>> import lucid.optim as optim
    >>> from lucid.compile._entry.function import compiled_step
    >>> model = nn.Linear(8, 4).to('metal')
    >>> opt = optim.SGD(model.parameters(), lr=0.1)
    >>> for batch in batches:                       # doctest: +SKIP
    ...     opt.zero_grad()
    ...     loss = compiled_step(model, batch.x,
    ...                          lambda y: F.cross_entropy(y, batch.target))
    ...     opt.step()

    See Also
    --------
    lucid.compile.make_step : the cached entry point preferred for
        repeated training loops — wraps this in a signature-keyed
        cache so the compile cost amortises.
    lucid.compile.compile : module-level entrypoint.
    """
    # Local imports to avoid circulars during package init.
    from lucid._dispatch import _unwrap, _wrap
    from lucid.compile._core.attention_probe import maybe_probe_for_graph
    from lucid.compile._core.bn_runstats import (
        advance_bn_counters,
        bn_counter_targets,
        bn_writeback_targets,
        model_has_cumulative_bn,
    )
    from lucid.compile._core.buffer_writes import refuse
    from lucid.compile._core.trace_context import compile_attempt, recording

    params = list(model.parameters())
    if not params:
        raise ValueError("compiled_step: model has no trainable parameters")

    with compile_attempt():
        with recording(watch=(x,), module=model) as tracer:
            out = model(x)
            # Module.__call__ widens to Tensor|tuple; loss_fn expects Tensor.
            if not isinstance(out, Tensor):
                raise TypeError(
                    "compiled_step: model must return a Tensor for the loss_fn "
                    f"to consume, got {type(out).__name__}"
                )
            loss_t = loss_fn(out)

        g = tracer.graph
        ext = tracer.external_feeds
        # No write-back on this path: a traced in-place buffer write sends it eager.
        refuse(tracer)
        if not g.ops:
            raise RuntimeError("compiled_step: empty trace (model produced no ops)")
        maybe_probe_for_graph(g)
        # The tensor ``loss_fn`` returned, not the last op traced — see make_step.
        found = (
            tracer.lookup_id(_unwrap(loss_t)) if isinstance(loss_t, Tensor) else None
        )
        if found is None:
            raise RuntimeError("compiled_step: loss_fn's result is not a traced tensor")
        loss_id = int(found)

        param_ids, observed = _match_param_ids(params, ext)
        if not all(observed):
            raise RuntimeError(
                "compiled_step: a model parameter was not observed in the trace "
                "(model.forward did not use it, or the parameter was wrapped/copied)"
            )
        input_ids, input_positions = _input_grad_targets((x,), ext, param_ids)
        grad_ids = param_ids + input_ids
        refuse_custom_backward(tracer, loss_id, grad_ids)

        # 3.5 BatchNorm running-stats. compiled_step has no eager fallback (it raises
        # on any uncompilable trace), so a cumulative-MA BN (momentum=None) is a hard
        # error. A track_running_stats=False BN keeps no buffers and compiles
        # unchanged. A fused-momentum BN traces 5-input; surface its EMA outputs as
        # extra executable outputs and copy_ them into the live module buffers after
        # the run, else the running stats freeze at trace time and eval() reads stale
        # stats (the eager EMA is skipped under a tracer).
        if model_has_cumulative_bn(model):
            raise RuntimeError(
                "compiled_step: BatchNorm with momentum=None (cumulative moving "
                "average) can't have its running-stats lowered into the compiled "
                "graph, and compiled_step has no eager fallback. Use "
                "lucid.compile.make_step or a momentum-based BatchNorm."
            )
        bn_targets = bn_writeback_targets(g, ext)
        counters = bn_counter_targets(model, g, ext)
        bn_stat_out_ids = [out_id for _, out_id, _ in bn_targets]
        bn_stat_buffers = [_wrap(impl) for _, _, impl in bn_targets]

        exe = _C_engine.compile.compile_trace_with_backward(
            g, ext, loss_id, grad_ids, extra_output_ids=bn_stat_out_ids
        )
        if exe is None:
            raise RuntimeError(
                "compiled_step: compile_trace_with_backward returned None — "
                "the trace contains an op without an emitter (eager fallback)"
            )

    feed_impls = [ext[tid] for tid in exe.input_ids]
    outs = _C_engine.compile.run_executable(exe, feed_impls)

    # Output order is [loss, *param grads, *input grads, *bn_running_stats].
    n_loss_grad = 1 + len(grad_ids)
    if len(outs) != n_loss_grad + len(bn_stat_out_ids):
        raise RuntimeError(
            f"compiled_step: executable produced {len(outs)} outputs but expected "
            f"{n_loss_grad + len(bn_stat_out_ids)} (loss + {len(grad_ids)} grads + "
            f"{len(bn_stat_out_ids)} BN running-stats)"
        )
    # Write BN running stats back into the live module buffers.
    for _i, _buf in enumerate(bn_stat_buffers):
        _buf.copy_(_wrap(outs[n_loss_grad + _i]))
    advance_bn_counters(counters)

    _accumulate_grads(exe, list(outs[1:n_loss_grad]), params, [x], input_positions)
    return _wrap(outs[0])
