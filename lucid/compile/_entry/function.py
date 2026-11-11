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

The first run of the executable is timed (``LUCID_COMPILE_FIRST_RUN_TIMEOUT_S``,
default 120 s, ``0`` turns the guard off) because MPSGraph can fail to finish
preparing a graph there.  Every other compile entry answers a first run the
engine gave up on eagerly; ``compiled_step`` has no eager path and raises the
engine's ``FirstRunTimeout`` (a ``RuntimeError``).  Each call traces a new
executable, so each call that meets such a graph spends the budget and leaves
MPSGraph's thread behind; after four, no further first run is attempted in the
process.  :func:`make_step` answers such a call eagerly instead.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

from lucid._C import engine as _C_engine
from lucid._tensor.tensor import Tensor
from lucid.compile._core.custom_backward import refuse_custom_backward
from lucid.compile._entry.step import _ExecutableLike, _GradTargets, _grad_slots, _grad_targets

if TYPE_CHECKING:
    from lucid._C.engine.compile import Tracer
    from lucid.compile._core.buffer_writes import CarriedWrites
    from lucid.nn.module import Module
    from lucid.nn.parameter import Parameter

__all__ = ["compiled_step"]


def _accumulate_grads(
    exe: _ExecutableLike,
    grads: list[_C_engine.TensorImpl],
    params: list[Parameter],
    leaves: list[Tensor],
    targets: _GradTargets,
) -> None:
    """Add ``exe``'s gradients into ``.grad`` the way ``loss.backward()`` does.

    Each gradient seeds the autograd engine at its tensor, so the engine's
    rule holds: a gradient already there is added to in place (a held
    ``p.grad`` sees the sum); a parameter that does not require grad, or
    that the loss does not depend on, gets none; and an input computed by
    eager ops passes its gradient on to the graph behind it.  Assigning
    ``p.grad`` instead dropped every earlier gradient and never reached
    the input (LCD-617).  ``loose`` are the tensors the loss itself owns (a
    learned loss weight): they get their gradients the same way (LCD-721).
    """
    from lucid._dispatch import _wrap
    from lucid.autograd import backward

    observed_params = [p for p, seen in zip(params, targets.observed, strict=True) if seen]
    n_params = len(observed_params)
    loose = targets.loose
    absent, input_grads, loose_present = _grad_slots(
        exe, n_params, targets.input_positions, len(loose)
    )
    roots: list[Tensor] = []
    seeds: list[Tensor] = []
    for i, (p, g) in enumerate(zip(observed_params, grads[:n_params], strict=True)):
        if p.requires_grad and not (absent and absent[i]):
            roots.append(p)
            seeds.append(_wrap(g))
    n_in = len(targets.input_positions)
    for pos, g in zip(input_grads, grads[n_params : n_params + n_in], strict=True):
        if pos is not None:
            roots.append(leaves[pos])
            seeds.append(_wrap(g))
    for impl, present, g in zip(loose, loose_present, grads[n_params + n_in :], strict=True):
        if present:
            roots.append(_wrap(impl))
            seeds.append(_wrap(g))
    if not roots:
        raise RuntimeError("element 0 of tensors does not require grad and does not have a grad_fn")
    backward(roots, seeds)


def _forward_loss(model: Module, x: Tensor, loss_fn: Callable[[Tensor], Tensor]) -> Tensor:
    """``loss_fn(model(x))`` — the one forward the traced and the eager step both run."""
    out = model(x)
    # Module.__call__ widens to Tensor|tuple; loss_fn expects Tensor.
    if not isinstance(out, Tensor):
        raise TypeError(
            "compiled_step: model must return a Tensor for the loss_fn "
            f"to consume, got {type(out).__name__}"
        )
    loss = loss_fn(out)
    if not isinstance(loss, Tensor):
        raise TypeError(f"compiled_step: loss_fn must return a Tensor, got {type(loss).__name__}")
    if loss.numel() != 1:
        raise RuntimeError("compiled_step: grad can be implicitly created only for scalar outputs")
    if not loss.requires_grad:
        raise RuntimeError("element 0 of tensors does not require grad and does not have a grad_fn")
    return loss


def _eager_step(model: Module, x: Tensor, loss_fn: Callable[[Tensor], Tensor], why: str) -> Tensor:
    """What :func:`compiled_step` does where nothing can be compiled: the eager step.

    Forward, loss and ``backward()`` add the gradients into ``.grad`` as the
    compiled step does, and the loss returned carries no graph, as its does.
    """
    from lucid.autograd._grad_mode import enable_grad
    from lucid.compile._core.fallback import _fallback_notice

    _fallback_notice("cpu", f"compiled_step: {why}")
    with enable_grad():
        loss = _forward_loss(model, x, loss_fn)
        loss.backward()
    return loss.detach()


def _trace_loss(
    model: Module, x: Tensor, loss_fn: Callable[[Tensor], Tensor]
) -> tuple[Tracer, Tensor]:
    """Trace the forward and loss in grad mode; a forward that raises completes the
    call first (:func:`~lucid.compile._core.buffer_writes.completes_on_raise`)."""
    from lucid.compile._core.buffer_writes import completes_on_raise
    from lucid.compile._core.trace_context import recording

    with (
        recording(watch=(x,), module=model, grad_mode=True) as tracer,
        completes_on_raise(tracer, model),
    ):
        loss_t = _forward_loss(model, x, loss_fn)
    return tracer, loss_t


def _refuse_cumulative_norm(model: Module) -> None:
    """Raise for a cumulative-average norm in training, before anything is traced.

    ``momentum=None`` reads the batch count on the host, so the layer traces without
    its buffers, and ``compiled_step`` has no eager path to hand the update to: it
    must not drop it silently.  An evaluating one reads its buffers and updates
    nothing.
    """
    from lucid.compile._core.bn_runstats import model_has_cumulative_bn

    if model_has_cumulative_bn(model, training_only=True):
        raise RuntimeError(
            "compiled_step: BatchNorm with momentum=None (cumulative moving "
            "average) in training mode can't have its running-stats lowered into "
            "the compiled graph, and compiled_step has no eager fallback. Use "
            "lucid.compile.make_step or a momentum-based BatchNorm."
        )


def _carried_writes(tracer: Tracer, model: Module, x: Tensor) -> CarriedWrites:
    """What the executable writes back after a run: statistics, counters, in-place writes.

    The same plan the other entries carry (:func:`carried_writes`).  Nothing is held
    over from an earlier trace: every call here traces afresh, so the trace's writes
    are the call's.  A write with nowhere to go raises, there being no eager call to
    run instead.
    """
    from lucid._dispatch import _unwrap
    from lucid.compile._core.buffer_writes import HeldWrites, carried_writes

    carried = carried_writes(tracer, model, [_unwrap(x)], HeldWrites())
    if not isinstance(carried, str):
        return carried
    raise RuntimeError(
        "compiled_step: running statistics and in-place writes to tensors "
        f"from outside the call can't all be carried here ({carried}); "
        "compiled_step has no eager fallback. Use lucid.compile.make_step."
    )


def compiled_step(model: Module, x: Tensor, loss_fn: Callable[[Tensor], Tensor]) -> Tensor:
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
        them — of each parameter that requires grad, of a loss weight
        ``loss_fn`` reads, and of ``x`` and the graph behind it when ``x``
        requires grad.  Then run the optimizer.

    Raises
    ------
    NotImplementedError
        A gradient passes through a custom ``Function``, whose ``backward``
        the executable cannot run (:func:`make_step` runs such a step eagerly).
    RuntimeError
        A BatchNorm with ``momentum=None`` is in training mode (its cumulative
        average reads the batch count on the host), or the call's running
        statistics or in-place writes have no tensor for the executable to write
        back to.  There is no eager path to run instead.

    Notes
    -----
    Parameters on the CPU have nothing to compile (Metal only): the step runs
    eagerly, with a one-time ``CompileFallbackWarning``.

    A norm's tracked running statistics (BatchNorm's, an InstanceNorm's) and its
    ``num_batches_tracked`` advance once per call, as in an eager step: the
    executable returns them and they are written back into the buffers after the
    run, as in :func:`make_step`.  So is any in-place write to a tensor from
    outside the call.

    A trace it cannot lower raises, there being no eager path, as does one reading
    a value computed on another thread (the tracer is per thread).

    The trace is captured in the caller's grad mode and fresh on every call;
    cache integration (compile-or-cached) lives in Phase 1.4's
    ``CompiledModule``.

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
    from lucid.compile._core.fallback import off_gpu_detail
    from lucid.compile._core.trace_context import compile_attempt

    params = list(model.parameters())
    if not params:
        raise ValueError("compiled_step: model has no trainable parameters")
    if (off_gpu := off_gpu_detail(params)) is not None:
        return _eager_step(model, x, loss_fn, off_gpu)
    _refuse_cumulative_norm(model)

    with compile_attempt():
        tracer, loss_t = _trace_loss(model, x, loss_fn)
        g = tracer.graph
        ext = tracer.external_feeds
        carried = _carried_writes(tracer, model, x)
        if not g.ops:
            raise RuntimeError("compiled_step: empty trace (model produced no ops)")
        maybe_probe_for_graph(g)
        found = tracer.read_id(_unwrap(loss_t))
        if found is None:
            raise RuntimeError("compiled_step: loss_fn's result is not a traced tensor")
        loss_id = int(found)

        targets = _grad_targets(tracer, params, (x,))
        grad_ids = targets.ids
        refuse_custom_backward(tracer, loss_id, grad_ids)

        exe = _C_engine.compile.compile_trace_with_backward(
            g, ext, loss_id, grad_ids, extra_output_ids=carried.out_ids
        )
        if exe is None:
            raise RuntimeError(
                "compiled_step: compile_trace_with_backward returned None — "
                "the trace contains an op without an emitter (eager fallback)"
            )

    feed_impls = [ext[tid] for tid in exe.input_ids]
    outs = _C_engine.compile.run_executable(exe, feed_impls)

    # Output order is [loss, *param grads, *input grads, *carried writes].
    n_loss_grad = 1 + len(grad_ids)
    if len(outs) != n_loss_grad + len(carried):
        raise RuntimeError(
            f"compiled_step: executable produced {len(outs)} outputs but expected "
            f"{n_loss_grad + len(carried)} (loss + {len(grad_ids)} grads + "
            f"{len(carried)} carried writes)"
        )
    carried.apply(outs[n_loss_grad:])

    _accumulate_grads(exe, list(outs[1:n_loss_grad]), params, [x], targets)
    return _wrap(outs[0])
