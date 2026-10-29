"""
lucid.compile._entry.fused_step — Phase 1.8 generic fused training step.

``fused_step(model, loss_fn, optimizer)`` returns a callable that runs
the entire training step (forward + loss + auto-derived gradients +
optimizer.step) inside a single MPSGraph executable.  One ``run`` call
per training iteration, all writes to parameters / optimizer state
happen in-place via ``run_executable_inplace``.

Architecture
------------
The optimizer update math lives in the corresponding
:mod:`lucid.compile._optim.compiler` subclass (``_CompiledSGD``,
``_CompiledAdam``, ..., ``_CompiledNAdam``).  On the first call:

1. Open a single :class:`Tracer`.  Run ``model(x); loss_fn(out, t)``
   (the forward + loss path).
2. Find the parameters the loss depends on (its ancestors in the trace)
   that also require grad — the parameters eager ``backward()`` would
   give a gradient.  Only those take part: MPSGraph's autodiff aborts
   the process when asked for the gradient of a tensor that does not
   precede the loss, and eager leaves such a parameter untouched.
3. Emit the optimizer update of those parameters with **ghost grad
   placeholders** (zero tensors that take the slot of "param_i.grad").
4. Call ``compile_generic_fused_step`` with the ghost grad ids.  C++
   derives gradients via MPSGraph autograd (or the manual VJPs) after
   emitting the forward, binds each ghost grad id to its derived
   gradient, then emits the update ops.
5. Each ``step(x, t)`` call: pack this step's hyper-parameters and
   per-parameter scalars (read from ``param_groups`` — LR schedulers
   take effect), then ``run_executable_inplace`` writes new params /
   new state directly into the corresponding tensors.

Freezing or unfreezing a parameter (``requires_grad_``) or flipping a
structural hyper-parameter (weight decay on/off …) builds another
executable on the next call; every executable built is kept, so
switching back is free.

Supported optimizers
--------------------
Every optimizer that :func:`compile_optimizer` accepts — all 13 of
Lucid's eager optimizers, any number of parameter groups.

Usage
-----
::

    model = MyModel().to('metal')
    opt   = optim.Adam(model.parameters(), lr=0.001)
    step  = lucid.compile.fused_step(model, F.mse_loss, opt)

    for x, t in loader:
        loss = step(x, t)        # forward + bwd + opt.step in one go
        # No backward(), no opt.step() — already done.
        if want_logging:
            print(float(loss.item()))

Limitations
-----------
* No dynamic batch — shape-locked to the first call's input signature.
* The loss tensor returned has no ``grad_fn`` (the backward has
  already run inside the executable).  ``loss.backward()`` is a no-op,
  and an input that requires grad is refused: no gradient can reach it.
* A step whose gradient passes through a custom ``autograd.Function``
  runs eagerly: the executable cannot run the ``Function``'s ``backward``.
"""

import threading
from dataclasses import dataclass, field as dataclass_field
from typing import TYPE_CHECKING, Callable, Iterator, final

from contextlib import contextmanager

from lucid._C import engine as _C_engine
from lucid._device import device as _device_cls
from lucid._dtype import dtype as _dtype_cls
from lucid.compile._core.bn_runstats import advance_bn_counters, bn_counter_targets
from lucid.compile._core.custom_backward import (
    GRADIENT_DTYPES,
    CustomBackwardInTrace,
    refuse_custom_backward,
    step_region_ancestors,
    vjp_region_index,
)
from lucid.compile._core.signature import ParamFingerprint, outside_feeds

# Thread-local flag flipped on while ``_FusedStep._build_plan``
# is actively tracing.  ``lucid.nn.functional.dropout`` checks it to
# decide whether to route training-mode dispatch through the
# ``dropout_stateful`` engine op (which only works when the compile
# path uses ``compile_generic_fused_step_with_vars`` to promote the
# state buffer to an MPSGraph variable).  The forward-only
# :class:`CompiledModule` path doesn't set this flag — its compile
# entry (``compile_trace``) has no variable-promotion hook, so
# training-mode dropout there continues to fall back to eager.
_tls = threading.local()


def _is_fused_step_tracing() -> bool:
    """Return True while inside ``_FusedStep._build_plan``'s trace."""
    return bool(getattr(_tls, "active", False))


# Hot-path imports — hoisted out of ``_FusedStep._run`` so the
# per-call ``from ... import _unwrap`` / ``import lucid as _lucid``
# bookkeeping (≈ 30-50 μs on M-series) doesn't run every training
# step.  Phase 1.10 per-call overhead reduction.
from lucid._dispatch import _unwrap as _unwrap_hot
import lucid as _lucid_hot

if TYPE_CHECKING:
    from lucid._C.engine.compile import CompiledExecutable, TraceGraph, Tracer
    from lucid._tensor.tensor import Tensor
    from lucid.amp.grad_scaler import GradScaler
    from lucid.compile._optim.compiler import _CompiledStepBase, _Flags, _ScalarLayout
    from lucid.nn.module import Module
    from lucid.optim.optimizer import Optimizer

__all__ = ["fused_step"]

# Executables a single fused step keeps (oldest dropped first).
_MAX_PLANS: int = 8


@final
@dataclass(slots=True)
class _FusedTrace:
    """Raw outputs captured while tracing forward + loss + optimizer update."""

    tracer: Tracer
    graph: TraceGraph
    ext: dict[int, _C_engine.TensorImpl]
    loss: Tensor
    loss_for_bwd: Tensor
    members: tuple[int, ...]
    ghost_grads: dict[int, Tensor]
    layout: _ScalarLayout
    vectors: dict[_C_engine.Dtype, Tensor]
    opt_outputs: list[Tensor]
    slots: list[tuple[str, int, str]]
    found_inf_f32: Tensor | None
    scaler_enabled: bool
    loss_id: int = 0
    unscaled_loss_tid: int | None = None
    param_ids: list[int] = dataclass_field(default_factory=list)
    ghost_grad_ids: list[int] = dataclass_field(default_factory=list)
    output_target_ids: list[int] = dataclass_field(default_factory=list)
    dropout_pairs: list[tuple[int, int]] = dataclass_field(default_factory=list)
    bn_pairs: list[tuple[int, int]] = dataclass_field(default_factory=list)


def fused_step(
    model: Module,
    loss_fn: Callable[..., Tensor],
    optimizer: Optimizer,
    *,
    grad_scaler: GradScaler | None = None,
) -> Callable[..., Tensor]:
    """Return a callable that runs one fused training step.

    The first call traces ``model(*x); loss_fn(out, *targets)`` and the
    optimizer update once under a single :class:`Tracer`, plumbs the
    ghost-grad placeholders, and compiles the resulting graph into one
    :class:`MPSGraphExecutable` that runs forward + loss + backward (via
    MPSGraph autodiff) + optimizer update in a single submission.
    Subsequent calls reuse the cached executable.

    The step matches an eager ``zero_grad(); loss.backward();
    optimizer.step()`` loop: a parameter the loss does not depend on, or
    one with ``requires_grad=False``, gets no gradient and is left
    untouched (state and step count included), and every hyper-parameter
    is read from ``optimizer.param_groups`` on each call, so LR
    schedulers work.  ``optimizer.state_dict()`` /
    ``load_state_dict()`` checkpoint the state this step updates (the
    compiled optimizer owns it from ``fused_step(...)`` on — see
    :func:`compile_optimizer`).

    Parameters
    ----------
    model : Module
        The trainable :class:`nn.Module`.  Parameters from
        ``optimizer.param_groups`` must reference these tensors.
    loss_fn : Callable
        Scalar-loss-returning callable invoked as
        ``loss_fn(model_output, *targets)``.  Captured by identity in
        the trace, so passing a fresh closure each call defeats the
        cache.
    optimizer : Optimizer
        Any optimizer :func:`compile_optimizer` supports, with any
        number of parameter groups.
    grad_scaler : GradScaler, optional
        When provided and enabled, the fused step replicates the
        eager :class:`lucid.amp.GradScaler` contract inside the
        compiled executable (scales loss before backward, unscales
        gradients in F32, guards updates with ``where(found_inf, old, new)``,
        and reads ``found_inf`` back to drive ``scaler.update()``).

    Returns
    -------
    Callable[..., Tensor]
        A callable ``step(*args)`` where ``args`` is
        ``(model_input, *loss_targets)``.  Returns the scalar loss
        :class:`Tensor` from the just-completed step.

    Raises
    ------
    RuntimeError
        On the first call, when no parameter both requires grad and
        reaches the loss (there would be nothing to step).  On any call
        given a tensor that requires grad — the output of an eager trunk,
        say: the step cannot return a gradient to it.  Detach it, or use
        :func:`make_step`, whose loss carries the input gradients.

    Notes
    -----
    Parameters on the CPU leave nothing to compile (the executables run on
    Metal only): the call runs the eager step — ``zero_grad()``, forward,
    ``backward()``, ``optimizer.step()`` of the optimizer passed in, which keeps
    its own state — and returns the eager loss, with a one-time
    ``CompileFallbackWarning``.

    Examples
    --------
    >>> import lucid, lucid.nn as nn, lucid.nn.functional as F
    >>> import lucid.optim as optim
    >>> from lucid.compile import fused_step
    >>> model = nn.Linear(8, 4).to('metal')
    >>> opt = optim.Adam(model.parameters(), lr=1e-3)
    >>> step = fused_step(model, F.cross_entropy, opt)
    >>> for batch in batches:                        # doctest: +SKIP
    ...     loss = step(batch.x, batch.target)       # one executable submission

    See Also
    --------
    lucid.compile.make_step : forward+backward only, no optimizer fusion.
    lucid.compile.compile_optimizer : the underlying optimizer-side
        translator that produces the update graph.
    """
    return _FusedStep(model, loss_fn, optimizer, grad_scaler=grad_scaler)


def _refuse_input_grads(tensors: list[Tensor]) -> None:
    """Refuse a call tensor that requires grad: the step cannot hand one back.

    The backward runs inside the executable and the loss returned has no
    ``grad_fn``, so the gradient of an input computed by eager ops never
    reached them — a compiled head on an eager trunk trained no trunk, and
    only the eager fallback (which calls ``backward`` itself) did (LCD-617).
    Refused on every path, before any state moves, so the answer does not
    depend on whether the step compiled.
    """
    for k, t in enumerate(tensors):
        if t.requires_grad:
            raise RuntimeError(
                f"fused_step: input {k} requires grad, which fused_step cannot "
                "produce: its backward runs inside the executable and the loss "
                "it returns has no grad_fn. Pass the input detached, or train a "
                "compiled head on an eager trunk with make_step"
            )


def _prod_shape(shape: tuple[int, ...]) -> int:
    """Product of the dims in ``shape`` (treating 0-D as 1).

    Used by the GradScaler integration: the ``found_inf`` reduction
    sums ``isfinite(g)`` to a scalar and compares against the
    expected total (the param's numel).  Computed at trace time
    from the trace-recorded shape, not from a graph reduction, so
    the comparison constant is a pure Python int / float baked into
    the trace via ``lucid.tensor(expected_total)``.
    """
    n = 1
    for d in shape:
        n *= int(d)
    return n


def _zeros_like(t: Tensor) -> Tensor:
    """Allocate a same-shape, same-dtype, same-device zero-filled Tensor.

    Used to materialise the *ghost-grad placeholders* fed into the
    trace: the executable will overwrite each placeholder with the
    actual gradient computed by MPSGraph's
    ``gradientForPrimaryTensor:`` at compile time, so the runtime
    value doesn't matter — only the shape + dtype + device that
    pin the trace placeholder's identity.
    """
    import lucid as _lucid

    return _lucid.zeros(*t.shape, dtype=t.dtype, device=t.device)


@contextmanager
def _untraced(tracer: object) -> Iterator[None]:
    """Detach ``tracer`` for the block, then put it back.

    Optimizer state and placeholders are allocated in the middle of the
    trace (only once the forward shows which parameters take part); a
    factory call recorded there would turn a feed into a traced op.
    """
    _C_engine.compile.set_current_tracer(None)
    try:
        yield
    finally:
        _C_engine.compile.set_current_tracer(tracer)


def _fallback_category(why: Exception) -> str:
    """The kind of reason a fused plan was declined for, as announced once."""
    if isinstance(why, CustomBackwardInTrace):
        return "custom-backward"
    if isinstance(why, NotImplementedError):
        return "cumulative-bn"
    return "first-run" if "MPSGraph failed" in str(why) else "lowering"


def _loss_ancestors(graph: TraceGraph, loss_id: int) -> set[int]:
    """Every trace id a gradient of ``loss_id`` flows back to (itself included).

    One reverse walk over the ops recorded so far — the forward and the
    loss, since the optimizer update has not been traced yet.  The walk
    stops where the backward walker (``VjpEmitter.mm``) does: at a gradient
    sink (``detach``, a comparison, a factory) and at an op with no
    floating result (a cast to an integer), while stepping across any
    traced custom VJP regions (``graph.vjp_regions``) from region outputs
    to region inputs.  A parameter seen only through a sink gets no
    gradient, so eager leaves its ``grad`` None and the optimizer skips it.
    """
    sink = _C_engine.compile.VjpRegistration.GradSink
    status = _C_engine.compile.vjp_registration_status
    in_region, region_at_end = vjp_region_index(graph)
    needed = {loss_id}
    for idx in range(len(graph.ops) - 1, -1, -1):
        if idx in in_region:
            step_region_ancestors(region_at_end.get(idx), needed)
            continue
        node = graph.ops[idx]
        if not any(int(meta.id) in needed for meta in node.outputs):
            continue
        if status(node.name) == sink or not any(
            meta.dtype in GRADIENT_DTYPES for meta in node.outputs
        ):
            continue
        needed.update(int(i) for i in node.inputs if int(i) >= 0)
    return needed


@final
@dataclass
class _FusedPlan:
    """One compiled fused step: its executable and how to feed / drain it."""

    exe: CompiledExecutable
    members: tuple[int, ...]
    layout: _ScalarLayout
    resolvers: list[Callable[[], _C_engine.TensorImpl]]
    target_impls: tuple[_C_engine.TensorImpl, ...]
    bn_counters: list[tuple[Module, int]]
    loss_shape: tuple[int, ...]
    loss_dtype: _dtype_cls
    loss_device: _device_cls
    scaled_loss_sink: _C_engine.TensorImpl | None = None


def _verify_target_shapes(
    graph: TraceGraph, checked_ids: list[int], output_targets: list[Tensor]
) -> None:
    """Guard against misordering between ``checked_ids`` and ``output_targets``."""
    id_to_shape: dict[int, tuple[int, ...]] = {}
    for node in graph.ops:
        for meta in node.outputs:
            id_to_shape[int(meta.id)] = tuple(meta.shape)
    for slot, (oid, tgt) in enumerate(zip(checked_ids, output_targets, strict=True)):
        exp = id_to_shape.get(int(oid))
        if exp is not None and tuple(tgt.shape) != exp:
            raise RuntimeError(
                f"fused_step: output target slot {slot} (id {oid}) shape "
                f"{tuple(tgt.shape)} != trace meta {exp} — "
                "output_target_ids / output_targets ordering drift"
            )


def _require_tid(tracer: Tracer, t: Tensor, msg: str) -> int:
    """Return the trace ID for ``t``, raising ``RuntimeError(msg)`` if absent."""
    tid = tracer.lookup_id(_unwrap_hot(t))
    if tid is None:
        raise RuntimeError(msg)
    return int(tid)


def _pinned_resolver(impl: _C_engine.TensorImpl) -> Callable[[], _C_engine.TensorImpl]:
    """Return a zero-argument callable returning ``impl``."""
    return lambda: impl


def _param_resolver(params: list[Tensor], i: int) -> Callable[[], _C_engine.TensorImpl]:
    return lambda: _unwrap_hot(params[i])


def _state_resolver(
    state: list[dict[str, Tensor]], i: int, name: str
) -> Callable[[], _C_engine.TensorImpl]:
    return lambda: _unwrap_hot(state[i][name])


@final
class _FusedStep:
    r"""Driver behind :func:`fused_step` — one executable, whole training step.

    Composes :mod:`lucid.compile._optim.compiler` (which already knows how to
    trace each optimizer's update arithmetic and identify the in-
    place output targets) with a trace of the model's forward pass +
    loss function.  The result is one :class:`MPSGraphExecutable`
    whose op DAG covers:

    .. code-block:: text

        x ─▶ model(x) ─▶ loss_fn(out, *targets) ─▶ loss
                              │
                              ▼
                ∂loss/∂param  (auto-derived inside C++ via
                              gradientForPrimaryTensor:withTensors:)
                              │
                              ▼
                opt_update(param, grad, state, scalars) ─▶ new_param, new_state
                              │
                              ▼
                          (in-place write back)

    Ghost-grad mechanism
    --------------------
    The challenge is that the optimizer update reads
    ``param.grad``, but the trace runs under :func:`no_grad` so no
    eager gradient exists yet.  We sidestep this by:

    1. Allocating one **ghost-grad placeholder** Tensor per
       participating parameter (a same-shape, same-dtype zero tensor).
    2. Feeding the placeholders into the optimizer's update emitter
       so the optimizer math captures them as graph inputs.
    3. Calling ``compile_generic_fused_step`` with the
       ``ghost_grad_ids``.  The C++ builder, before emitting the
       optimizer ops, derives the gradient of the loss with respect to
       each participating parameter and binds each ``ghost_grad_id`` to
       it.

    Which parameters take part
    --------------------------
    Those the loss depends on that also require grad — exactly the ones
    an eager ``backward()`` leaves a gradient on.  The set is part of
    the executable's key: ``requires_grad_`` between calls selects (or
    builds) another executable.

    Attributes
    ----------
    _model : nn.Module
        The compute unit being traced.
    _loss_fn : Callable
        Scalar-returning callable invoked as
        ``loss_fn(model_output, *targets)``.
    _copt : _CompiledStepBase
        The compiled optimizer driving the update math and owning the
        optimizer state (exposed as :attr:`optimizer`).
    _plans : dict
        Compiled executables keyed by (participating parameters,
        structural flags of every group, per-parameter scalar classes).
    _reachable : frozenset[int] or None
        Flat indices of the parameters the loss depends on; found on the
        first trace.

    See Also
    --------
    :func:`fused_step` : user-facing constructor.
    :func:`compile_optimizer` : the underlying optimizer compile
        path whose update emitter is reused here.
    :class:`CompiledModule` : forward-only compile (no
        backward+update fusion).
    """

    def __init__(
        self,
        model: Module,
        loss_fn: Callable[..., Tensor],
        optimizer: Optimizer,
        *,
        grad_scaler: GradScaler | None = None,
    ) -> None:
        """Initialise the driver; the executable itself is lazy.

        Parameters
        ----------
        model, loss_fn, optimizer
            See :func:`fused_step` for semantics.
        grad_scaler : GradScaler, optional
            See :func:`fused_step`.

        Raises
        ------
        ValueError
            If ``optimizer`` exposes no trainable parameters (every
            param_group is empty).
        """
        from lucid.compile._core.fallback import off_gpu_detail

        self._model = model
        self._loss_fn = loss_fn
        self._optimizer = optimizer
        # The compiled optimizer owns the optimizer math + state buffers
        # + per-step scalars.  Any optimizer it supports (or rejects)
        # propagates automatically here.  It is built when the parameters are
        # on Metal — here, or on the first call after the model moves there —
        # so a CPU-resident ``optimizer`` keeps its own state and steps eagerly.
        self._compiled: _CompiledStepBase | None = None
        self._params_seen: list[Tensor] = []
        if not any(group["params"] for group in optimizer.param_groups):
            raise ValueError("fused_step: optimizer has no trainable parameters")
        if off_gpu_detail(model.parameters()) is None:
            self._compile_optimizer()

        # GradScaler integration (X4.3).  None ⇒ unscaled path.
        self._grad_scaler: GradScaler | None = grad_scaler
        # Stable 0-D scalar holders refreshed each step via ``copy_``
        # so the cached executable keeps its TensorImpl identity.
        # Allocated lazily at the first build.
        self._scale_holder: Tensor | None = None
        self._inv_scale_holder: Tensor | None = None
        # Persistent F32 0-D holder for the found_inf output.  Read
        # back after each run to drive ``scaler.update()``.
        self._found_inf_target: Tensor | None = None

        self._plans: dict[tuple[object, ...], _FusedPlan] = {}
        self._eager_only: set[tuple[object, ...]] = set()
        self._reachable: frozenset[int] | None = None
        # Part of the plan key, as make_step's; and a plan feeds and writes the
        # tensors it was traced with: after a ``.half()`` / ``.float()`` round
        # trip or an ``assign=True`` load on any module of the tree, it went on
        # training the old ones (LCD-505).
        self._fingerprint = ParamFingerprint(model, on_replaced=self._plans.clear)
        # Per-call args stash so positional resolvers can pick them up.
        self._current_args: tuple[Tensor, ...] = ()
        # This call's packed scalar vectors (read by their resolvers).
        self._step_vectors: list[_C_engine.TensorImpl] = []

    # ── Public API ──────────────────────────────────────────────

    @property
    def optimizer(self) -> object:
        """The compiled optimizer this step updates — checkpoint through it.

        Its ``state_dict()`` / ``load_state_dict()`` carry the state the
        fused executable reads and writes, in the eager format.  Where the
        step runs eagerly because the model is on the CPU, it is the
        optimizer passed in, which keeps its own state.
        """
        return self._optimizer if self._compiled is None else self._compiled

    def cache_info(self) -> dict[str, object]:
        """Snapshot the plan cache and eager-fallback blacklist for tests and telemetry."""
        return {
            "entries": len(self._plans),
            "keys": tuple(self._plans.keys()),
            "eager_only": tuple(self._eager_only),
        }

    def __call__(self, *args: Tensor) -> Tensor:
        """Run one fused training step; lazy-compile on first call.

        Parameters
        ----------
        *args : Tensor
            ``(model_input, *loss_targets)`` — same positional tuple
            that ``loss_fn(model(input), *targets)`` would consume.

        Returns
        -------
        Tensor
            Scalar loss tensor freshly allocated for this step.  All
            parameter and optimizer-state buffers have already been
            updated **in-place** before this returns.

        Notes
        -----
        Each call counts as an optimizer step for the LR schedulers built on
        the optimizer or on ``self.optimizer``, as an eager
        ``optimizer.step()`` does.  It counts even when ``grad_scaler``
        skips the update for a gradient that overflowed.  An eager
        ``GradScaler.step`` that skips does not call ``optimizer.step``, so
        there the skipped step does not count.
        """
        from lucid.compile._core.fallback import _fallback_notice, off_gpu_detail
        from lucid.compile._core.signature import (
            leaf_tensors,
            model_has_active_hooks,
            tensors_have_active_hooks,
        )

        leaves = leaf_tensors(args, {})
        _refuse_input_grads(leaves)
        # Once the compiled optimizer owns the state there is no CPU step to
        # fall back to (a model moved back to CPU): its own step refuses.
        if (
            self._compiled is None
            and (off_gpu := off_gpu_detail(self._model.parameters())) is not None
        ):
            _fallback_notice("cpu", f"fused_step: {off_gpu}")
            return self._run_eager(args)
        copt = self._copt
        copt._count_step()  # the fused update never calls ``optimizer.step``
        copt._sync_params()
        if copt._params is not self._params_seen:
            # ``add_param_group`` re-flattened the parameters.
            self._params_seen = copt._params
            self.recompile()
        if model_has_active_hooks(self._model) or tensors_have_active_hooks(leaves):
            return self._run_eager(args)
        flags = copt._flags_now()
        key = self._key(self._members() if self._reachable is not None else (), flags, args)
        if key in self._eager_only:
            return self._run_eager(args)
        plan = self._plans.get(key) if self._reachable is not None else None
        if plan is None:
            return self._compile_and_run_first(args, flags)
        # GradScaler scalar refresh — write the current ``scaler._scale``
        # + ``1/scale`` into the persistent 0-D holders before run.
        self._refresh_scaler_scalars()
        return self._run(plan, args, flags)

    def _compile_and_run_first(self, args: tuple[Tensor, ...], flags: tuple[_Flags, ...]) -> Tensor:
        """Build a new fused plan and execute its first run, or fall back to eager."""
        from lucid.compile._core.fallback import _fallback_notice
        from lucid.compile._core.trace_context import compile_attempt

        copt = self._copt
        prev_init = list(copt._initialized)
        prev_pstate = [dict(d) for d in copt._pstate]
        try:
            with compile_attempt():
                plan = self._build_plan(args, flags)
            self._refresh_scaler_scalars()
            result = self._run(plan, args, flags)
        except (RuntimeError, NotImplementedError) as why:
            msg = str(why)
            if (
                "LUCID_MANUAL_VJP_REQUIRE=1" in msg
                or "no optimizer parameter both requires grad" in msg
            ):
                raise
            copt._initialized[:] = prev_init
            copt._pstate[:] = prev_pstate
            _fallback_notice(_fallback_category(why), msg)
            fallback_key = self._key(
                self._members() if self._reachable is not None else (), flags, args
            )
            self._eager_only.add(fallback_key)
            return self._run_eager(args)
        # Keyed after the build + first run: it initialised the new members' state.
        self._plans[self._key(plan.members, flags, args)] = plan
        while len(self._plans) > _MAX_PLANS:
            self._plans.pop(next(iter(self._plans)))
        return result

    def _run_eager(self, args: tuple[Tensor, ...]) -> Tensor:
        """Run one training step eagerly when the fused plan declines or fails."""
        from lucid._tensor.tensor import Tensor
        from lucid.autograd._grad_mode import enable_grad

        # Whoever owns the optimizer state steps it: the compiled wrapper once
        # it has the state, else the optimizer itself.
        stepper = self._optimizer if self._compiled is None else self._compiled
        stepper.zero_grad()
        with enable_grad():
            out = self._model(*args[:1])
            loss = self._loss_fn(out, *args[1:])
            if self._grad_scaler is not None and self._grad_scaler._enabled:
                scaled = self._grad_scaler.scale(loss)
                assert isinstance(scaled, Tensor)
                scaled.backward()
                self._grad_scaler.step(self._optimizer)
                self._grad_scaler.update()
            else:
                loss.backward()
                stepper.step()
        return loss.detach()

    def _refresh_scaler_scalars(self) -> None:
        """Copy ``scaler._scale`` + ``1/scale`` into the persistent feeds.

        ``copy_`` writes through to the holder's existing buffer so
        the cached executable keeps hitting the same input slot.
        """
        if self._grad_scaler is None or not self._grad_scaler._enabled:
            return
        if self._scale_holder is None or self._inv_scale_holder is None:
            return

        import lucid as _lucid

        scale = float(self._grad_scaler.get_scale())
        inv = 1.0 / scale
        dt = self._scale_holder.dtype
        dev = self._scale_holder.device
        self._scale_holder.copy_(_lucid.tensor(scale, dtype=dt, device=dev))
        self._inv_scale_holder.copy_(_lucid.tensor(inv, dtype=dt, device=dev))

    def recompile(self) -> None:
        """Drop every cached executable so the next call retraces from scratch.

        Useful after manual surgery on the model (e.g. resizing a
        parameter buffer) where the captured tensor identities no longer
        match the live ones.  Normal training loops never need to call
        this — freezing / unfreezing parameters, LR schedules, the
        tensors a ``.to()`` / ``.half()`` or an ``assign=True`` load
        replaces, and a parameter, buffer or submodule set in another's
        place are picked up on their own.
        """
        self._plans.clear()
        self._eager_only.clear()
        self._reachable = None

    # ── Internals ───────────────────────────────────────────────

    @property
    def _copt(self) -> _CompiledStepBase:
        """The compiled optimizer, built — and handed the state — on first need."""
        return self._compiled if self._compiled is not None else self._compile_optimizer()

    def _compile_optimizer(self) -> _CompiledStepBase:
        """Hand the optimizer's state to a compiled wrapper (once)."""
        from lucid.compile._optim.compiler import compile_optimizer

        copt = self._compiled = compile_optimizer(self._optimizer)
        self._params_seen = copt._params
        return copt

    def _key(
        self, members: tuple[int, ...], flags: tuple[_Flags, ...], args: tuple[Tensor, ...] = ()
    ) -> tuple[object, ...]:
        """Plan key: participants, structural flags, per-parameter classes, inputs, params.

        The inputs by how they bind to the trace's feeds too: ``step(x, x)``
        reads one feed where ``step(a, b)`` reads two, and a buffer passed as
        the input shares the buffer's (``ParamFingerprint.binding``).
        """
        arg_sig = tuple((tuple(a.shape), a.dtype, a.device) for a in args)
        amp_active = _C_engine.amp_is_active()
        amp_sig = (amp_active, _C_engine.amp_active_dtype() if amp_active else None)
        partition = self._copt._partition(members)
        binding = self._fingerprint.binding(args, {})
        return (members, flags, partition, arg_sig, amp_sig, self._fingerprint.get(), binding)

    def _members(self) -> tuple[int, ...]:
        """Parameters that step: reach the loss and require grad (eager's rule)."""
        assert self._reachable is not None
        params = self._copt._params
        return tuple(i for i in sorted(self._reachable) if params[i].requires_grad)

    def _ensure_scaler_holders(self) -> bool:
        """Allocate and refresh GradScaler 0-D scalar holders when enabled."""
        scaler_enabled = self._grad_scaler is not None and self._grad_scaler._enabled
        if not scaler_enabled:
            return False
        if self._scale_holder is None:
            import lucid as _lucid

            p0 = self._copt._params[0]
            self._scale_holder = _lucid.zeros((), dtype=_lucid.float32, device=p0.device)
            self._inv_scale_holder = _lucid.zeros((), dtype=_lucid.float32, device=p0.device)
            self._found_inf_target = _lucid.zeros((), dtype=_lucid.float32, device=p0.device)
        self._refresh_scaler_scalars()
        return True

    def _resolve_members(self, tracer: Tracer, loss_for_bwd: Tensor) -> tuple[int, ...]:
        """Find participating parameters that require grad and reach the loss."""
        from lucid._dispatch import _unwrap

        params = self._copt._params
        loss_tid = tracer.lookup_id(_unwrap(loss_for_bwd))
        if loss_tid is None:
            raise RuntimeError("fused_step: loss missing from trace")
        ancestors = _loss_ancestors(tracer.graph, int(loss_tid))
        reachable = {
            i
            for i, p in enumerate(params)
            if (tid := tracer.lookup_id(_unwrap(p))) is not None and int(tid) in ancestors
        }
        self._reachable = frozenset(reachable)
        members = self._members()
        if not members:
            raise RuntimeError(
                "fused_step: no optimizer parameter both requires "
                "grad and reaches the loss — there is nothing to step"
            )
        return members

    def _emit_scaled_update(
        self,
        ghost_grads: dict[int, Tensor],
        layout: _ScalarLayout,
        vectors: dict[_C_engine.Dtype, Tensor],
        flags: tuple[_Flags, ...],
        loss_for_bwd: Tensor,
    ) -> tuple[list[Tensor], list[tuple[str, int, str]], Tensor]:
        """Emit unscaled gradients, ``found_inf``, and ``where``-guarded optimizer updates."""
        import lucid as _lucid

        copt = self._copt
        params = copt._params
        members = tuple(ghost_grads.keys())
        assert self._inv_scale_holder is not None
        unscaled: dict[int, Tensor] = {
            i: (g if g.dtype == _lucid.float32 else g.to(_lucid.float32)) * self._inv_scale_holder
            for i, g in ghost_grads.items()
        }
        finite_counts: list[Tensor] = [
            _lucid.isfinite(g).to(_lucid.float32).sum() for g in unscaled.values()
        ]
        expected_total = sum(float(int(_prod_shape(g.shape))) for g in unscaled.values())
        if loss_for_bwd.dtype == _lucid.float16:
            assert self._scale_holder is not None
            finite_counts.append(_lucid.isfinite(loss_for_bwd).to(_lucid.float32).sum())
            expected_total += float(int(_prod_shape(loss_for_bwd.shape)))
            finite_counts.append((self._scale_holder <= 65504.0).to(_lucid.float32).sum())
            expected_total += 1.0
        total_finite = finite_counts[0]
        for nf in finite_counts[1:]:
            total_finite = total_finite + nf
        expected_t = _lucid.tensor(expected_total, dtype=_lucid.float32, device=params[0].device)
        found_inf_bool = total_finite < expected_t
        found_inf_f32 = found_inf_bool.to(_lucid.float32)

        opt_outputs, slots = copt._emit(members, unscaled, layout, vectors, flags, False)
        final_outputs: list[Tensor] = []
        for slot, new in zip(slots, opt_outputs, strict=True):
            old = copt._slot_tensor(slot)
            if old.dtype != new.dtype:
                old = old.to(new.dtype)
            final_outputs.append(_lucid.where(found_inf_bool, old, new))
        return final_outputs, slots, found_inf_f32

    def _trace_fused_graph(
        self, args: tuple[Tensor, ...], flags: tuple[_Flags, ...], *, scaler_enabled: bool
    ) -> _FusedTrace:
        """Trace forward + loss in the caller's grad mode and optimizer update under ``no_grad()``."""
        from lucid.autograd._grad_mode import no_grad
        from lucid.compile._core.buffer_writes import refuse
        from lucid.compile._core.trace_context import recording

        copt = self._copt
        params = copt._params
        groups = copt._opt.param_groups

        _tls.active = True
        try:
            with recording(watch=args, module=self._model) as tracer:
                try:
                    out = self._model(*args[:1])
                    loss = self._loss_fn(out, *args[1:])
                    if scaler_enabled:
                        assert self._scale_holder is not None
                        loss_for_bwd = loss * self._scale_holder.to(loss.dtype)
                    else:
                        loss_for_bwd = loss

                    members = self._resolve_members(tracer, loss_for_bwd)
                    with no_grad():
                        with _untraced(tracer):
                            for i in members:
                                copt._activate(i, flags, groups)
                            ghost_grads = {i: _zeros_like(params[i]) for i in members}
                            layout = copt._new_layout(members, False, flags)
                            vectors = copt._trace_vectors(layout)

                        found_inf_f32: Tensor | None = None
                        if scaler_enabled:
                            opt_outputs, slots, found_inf_f32 = self._emit_scaled_update(
                                ghost_grads, layout, vectors, flags, loss_for_bwd
                            )
                        else:
                            opt_outputs, slots = copt._emit(
                                members, ghost_grads, layout, vectors, flags, False
                            )
                finally:
                    refuse(tracer)
        finally:
            _tls.active = False

        return _FusedTrace(
            tracer=tracer,
            graph=tracer.graph,
            ext=dict(tracer.external_feeds),
            loss=loss,
            loss_for_bwd=loss_for_bwd,
            members=members,
            ghost_grads=ghost_grads,
            layout=layout,
            vectors=vectors,
            opt_outputs=opt_outputs,
            slots=slots,
            found_inf_f32=found_inf_f32,
            scaler_enabled=scaler_enabled,
        )

    def _lookup_trace_ids(self, traced: _FusedTrace) -> None:
        """Resolve trace IDs for loss, parameters, ghost grads, and optimizer outputs."""
        tracer = traced.tracer
        params = self._copt._params
        traced.loss_id = _require_tid(
            tracer, traced.loss_for_bwd, "fused_step: loss missing from trace"
        )
        if traced.scaler_enabled:
            traced.unscaled_loss_tid = _require_tid(
                tracer, traced.loss, "fused_step: unscaled loss missing from trace"
            )
        for i in traced.members:
            traced.param_ids.append(
                _require_tid(tracer, params[i], "fused_step: a parameter vanished from the trace")
            )
            traced.ghost_grad_ids.append(
                _require_tid(
                    tracer,
                    traced.ghost_grads[i],
                    "fused_step: ghost grad placeholder missing from trace",
                )
            )
        for o in traced.opt_outputs:
            traced.output_target_ids.append(
                _require_tid(tracer, o, "fused_step: optimizer output tensor missing from trace")
            )
        if traced.scaler_enabled and traced.found_inf_f32 is not None:
            traced.output_target_ids.append(
                _require_tid(
                    tracer, traced.found_inf_f32, "fused_step: found_inf scalar missing from trace"
                )
            )

    def _collect_stateful_targets(self, traced: _FusedTrace) -> None:
        """Collect dropout state pairs and BatchNorm running-stat target pairs."""
        from lucid.compile._core.bn_runstats import bn_writeback_targets, model_has_cumulative_bn

        for node in traced.graph.ops:
            if node.name == "dropout_stateful" and len(node.inputs) >= 2 and len(node.outputs) >= 2:
                sin = int(node.inputs[1])
                sout = int(node.outputs[1].id)
                traced.output_target_ids.append(sout)
                traced.dropout_pairs.append((sin, sout))

        if model_has_cumulative_bn(self._model):
            raise NotImplementedError(
                "BatchNorm with momentum=None (cumulative moving average) "
                "cannot lower its running-stats update into the compiled graph"
            )
        traced.bn_pairs = [
            (feed_id, out_id)
            for feed_id, out_id, _ in bn_writeback_targets(traced.graph, traced.ext)
        ]
        for _bn_feed_id, bn_new_id in traced.bn_pairs:
            traced.output_target_ids.append(bn_new_id)
        if traced.unscaled_loss_tid is not None:
            traced.output_target_ids.append(traced.unscaled_loss_tid)

    def _compile_fused_executable(self, traced: _FusedTrace) -> CompiledExecutable:
        """Lower the combined forward+backward+optimizer trace into an executable."""
        import os as _os

        params_as_vars = _os.environ.get("LUCID_COMPILE_VARS", "0") in ("1", "true", "True")
        if params_as_vars or traced.dropout_pairs:
            variable_pairs: list[tuple[int, int]] = []
            if params_as_vars:
                for i, pid in enumerate(traced.param_ids):
                    variable_pairs.append((pid, traced.output_target_ids[i]))
            variable_pairs.extend(traced.dropout_pairs)
            exe = _C_engine.compile.compile_generic_fused_step_with_vars(
                traced.graph,
                traced.ext,
                traced.loss_id,
                traced.param_ids,
                traced.ghost_grad_ids,
                traced.output_target_ids,
                variable_pairs,
            )
        else:
            exe = _C_engine.compile.compile_generic_fused_step(
                traced.graph,
                traced.ext,
                traced.loss_id,
                traced.param_ids,
                traced.ghost_grad_ids,
                traced.output_target_ids,
            )
        if exe is None:
            raise RuntimeError(
                "fused_step: compile_generic_fused_step returned None — "
                "an op in the combined trace has no emitter, or the "
                "trace is otherwise incompatible with the fused path."
            )
        return exe

    def _vector_resolver(self, k: int) -> Callable[[], _C_engine.TensorImpl]:
        return lambda: self._step_vectors[k]

    def _arg_resolver(self, slot: int) -> Callable[[], _C_engine.TensorImpl]:
        return lambda: _unwrap_hot(self._current_args[slot])

    def _build_resolvers(
        self, exe: CompiledExecutable, traced: _FusedTrace, args: tuple[Tensor, ...]
    ) -> list[Callable[[], _C_engine.TensorImpl]]:
        """Build per-input resolvers mapping ``exe.input_ids`` to live ``TensorImpl``s."""
        from lucid._tensor.tensor import Tensor

        copt = self._copt
        params = copt._params
        ext = traced.ext
        impl_to_resolver: dict[int, Callable[[], _C_engine.TensorImpl]] = {
            id(_unwrap_hot(p)): _param_resolver(params, i) for i, p in enumerate(params)
        }
        for i in traced.members:
            for name, buf in copt._state[i].items():
                impl_to_resolver[id(_unwrap_hot(buf))] = _state_resolver(copt._state, i, name)
        for k, dt in enumerate(traced.layout.dtypes):
            impl_to_resolver[id(_unwrap_hot(traced.vectors[dt]))] = self._vector_resolver(k)
        for holder in (self._scale_holder, self._inv_scale_holder):
            if holder is not None:
                h_impl = _unwrap_hot(holder)
                impl_to_resolver[id(h_impl)] = _pinned_resolver(h_impl)
        for pos, a in enumerate(args):
            if isinstance(a, Tensor):
                impl_to_resolver[id(_unwrap_hot(a))] = self._arg_resolver(pos)
        for bn_feed_id, _bn_new_id in traced.bn_pairs:
            bn_impl = ext.get(bn_feed_id)
            if bn_impl is not None and id(bn_impl) not in impl_to_resolver:
                impl_to_resolver[id(bn_impl)] = _pinned_resolver(bn_impl)

        resolvers: list[Callable[[], _C_engine.TensorImpl]] = []
        for tid in exe.input_ids:
            impl = ext.get(tid)
            if impl is None:
                raise RuntimeError(f"fused_step: input id {tid} not in external_feeds")
            r = impl_to_resolver.get(id(impl))
            resolvers.append(r if r is not None else _pinned_resolver(impl))
        return resolvers

    def _build_output_targets(self, traced: _FusedTrace) -> tuple[_C_engine.TensorImpl, ...]:
        """Build and validate in-place output target ``TensorImpl``s."""
        from lucid._dispatch import _unwrap, _wrap

        copt = self._copt
        ext = traced.ext
        output_targets: list[Tensor] = [copt._slot_tensor(s) for s in traced.slots]
        if traced.scaler_enabled and self._found_inf_target is not None:
            output_targets.append(self._found_inf_target)
        for state_in_tid, _state_out_tid in traced.dropout_pairs:
            impl = ext.get(state_in_tid)
            if impl is None:
                raise RuntimeError(
                    f"fused_step: dropout state feed id {state_in_tid} not in external_feeds"
                )
            output_targets.append(_wrap(impl))
        for bn_feed_id, _bn_new_id in traced.bn_pairs:
            bn_impl = ext.get(bn_feed_id)
            if bn_impl is None:
                raise RuntimeError(
                    f"fused_step: BN running-stat feed id {bn_feed_id} not in external_feeds"
                )
            output_targets.append(_wrap(bn_impl))

        checked_ids = (
            traced.output_target_ids[:-1]
            if traced.unscaled_loss_tid is not None
            else traced.output_target_ids
        )
        if len(output_targets) != len(checked_ids):
            raise RuntimeError(
                f"fused_step: output_targets count ({len(output_targets)}) doesn't match "
                f"output_target_ids count ({len(checked_ids)})"
            )
        _verify_target_shapes(traced.graph, checked_ids, output_targets)
        return tuple(_unwrap(t) for t in output_targets)

    def _build_plan(self, args: tuple[Tensor, ...], flags: tuple[_Flags, ...]) -> _FusedPlan:
        """Trace + compile the fused step for the current participants."""
        from lucid.compile._core.attention_probe import maybe_probe_for_graph

        scaler_enabled = self._ensure_scaler_holders()
        before = self._fingerprint.outside_slots()
        try:
            traced = self._trace_fused_graph(args, flags, scaler_enabled=scaler_enabled)
        except BaseException:
            self._fingerprint.settle((), before)
            raise
        self._fingerprint.settle(outside_feeds(traced.tracer), before)
        bn_counters = bn_counter_targets(self._model, traced.graph, traced.ext)
        if not traced.graph.ops:
            raise RuntimeError("fused_step: empty trace")
        maybe_probe_for_graph(traced.graph)

        self._lookup_trace_ids(traced)
        refuse_custom_backward(traced.tracer, traced.loss_id, traced.param_ids)
        self._collect_stateful_targets(traced)
        exe = self._compile_fused_executable(traced)
        resolvers = self._build_resolvers(exe, traced, args)
        target_impls = self._build_output_targets(traced)
        scaled_loss_sink: _C_engine.TensorImpl | None = None
        if traced.unscaled_loss_tid is not None:
            loss_bwd = traced.loss_for_bwd
            scaled_loss_sink = _unwrap_hot(
                _lucid_hot.zeros(
                    *loss_bwd.shape if loss_bwd.shape else (),
                    dtype=loss_bwd.dtype,
                    device=loss_bwd.device,
                )
            )
        return _FusedPlan(
            exe=exe,
            members=traced.members,
            layout=traced.layout,
            resolvers=resolvers,
            target_impls=target_impls,
            bn_counters=bn_counters,
            loss_shape=tuple(traced.loss.shape),
            loss_dtype=traced.loss.dtype,
            loss_device=traced.loss.device,
            scaled_loss_sink=scaled_loss_sink,
        )

    def _run(self, plan: _FusedPlan, args: tuple[Tensor, ...], flags: tuple[_Flags, ...]) -> Tensor:
        """Bind feeds + targets and invoke the cached executable in-place.

        Packs this step's scalars (hyper-parameters read from the live
        ``param_groups`` and per-parameter step-dependent values), runs
        the executable, and advances the step counts of the parameters
        that stepped — unless the GradScaler found an overflow, in which
        case nothing stepped.

        Parameters
        ----------
        plan : _FusedPlan
            The executable for this step's participants.
        args : tuple of Tensor
            Same per-step arguments forwarded from :meth:`__call__`.
        flags : tuple
            Every group's structural flags (the plan was built for them).

        Returns
        -------
        Tensor
            Scalar loss for this step.
        """
        copt = self._copt
        self._current_args = args
        self._step_vectors = copt._scalar_vectors(plan.layout, flags, None)
        feeds = [r() for r in plan.resolvers]

        # Fresh loss tensor (single-element output that flows back to
        # Python; not in-place since callers want a clean handle).
        loss_tensor = _lucid_hot.zeros(
            *plan.loss_shape if plan.loss_shape else (),
            dtype=plan.loss_dtype,
            device=plan.loss_device,
        )
        if plan.scaled_loss_sink is None:
            output_targets = [_unwrap_hot(loss_tensor), *plan.target_impls]
        else:
            output_targets = [plan.scaled_loss_sink, *plan.target_impls, _unwrap_hot(loss_tensor)]
        _C_engine.compile.run_executable_inplace(plan.exe, feeds, output_targets)
        advance_bn_counters(plan.bn_counters)

        skipped = False
        if (
            self._grad_scaler is not None
            and self._grad_scaler._enabled
            and self._found_inf_target is not None
        ):
            # ``item()`` forces a CPU sync — unavoidable since the
            # scaler's update logic needs the bool.
            skipped = bool(float(self._found_inf_target.item()))
            self._grad_scaler._found_inf = skipped
            self._grad_scaler.update()
        if not skipped:
            copt._commit(plan.members)
        return loss_tensor
