"""
lucid.compile._optim.compiler — compiled optimizer wrappers (in-place output path).

Wraps an eager :class:`lucid.optim.Optimizer` so ``step()`` becomes a
single MPSGraph executable that fuses the per-parameter update math AND
writes its outputs directly back into the parameter / state buffers via
``run_executable_inplace``.

Why in-place
------------
A first revision routed the update through ``lucid.compile(update_fn)``
and ``param.copy_(new_param)`` per parameter.  That version compiled
correctly (bit-exact parity) but ran ~50 % SLOWER than the eager C++
optim, because the per-output ``copy_`` calls each forced an
``mlx::copy`` sync.  The current implementation runs one executable whose
outputs replace the parameter and state buffers directly.

What the executable reads
-------------------------
Nothing the user can change between steps is baked into the trace:

* **Hyper-parameters** (``lr``, ``weight_decay``, ``momentum``, betas,
  ``eps`` …) are read from ``param_groups`` on every step, so an LR
  scheduler — or any hand edit of a group — takes effect on the next
  step without a retrace.
* **Per-parameter step counts** (bias corrections, NAdam's momentum
  product, ASGD's averaging schedule, Adagrad's decayed rate) are kept
  per parameter, as the eager engine keeps them, so a parameter that
  joins training late starts its bias correction at step 1.

All of those numbers travel as one packed 1-D vector per parameter dtype
— a single host-to-device upload per step — that the trace splits with
one ``unbind``.  Only *structural* choices (whether weight decay or
momentum is on, AMSGrad, centred RMSprop, which parameters take part)
shape the trace; when one of those changes the next step builds (or
reuses from a small cache) another executable.

Parameters whose gradient is ``None`` are skipped exactly as the eager
optimizers skip them: no update, no state change, no step advance.
"""

import math
import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Sequence, cast, final, override

from lucid._C import engine as _C_engine
from lucid._dtype import _float32_or_wider
from lucid._tensor.tensor import Tensor
from lucid.compile._optim.spec import _hp
from lucid.optim._step_counter import _counts_as_step, _record_step

if TYPE_CHECKING:
    from lucid.optim.optimizer import Optimizer

__all__ = ["compile_optimizer"]


def compile_optimizer(opt: Optimizer) -> _CompiledStepBase:
    r"""Wrap ``opt`` so ``opt.step()`` runs as a single MPSGraph executable.

    Dispatches on the concrete optimizer subclass and returns the
    matching :class:`_CompiledStepBase` subclass instance.  The
    returned object exposes the same eager-optimizer API surface
    (``step`` / ``zero_grad`` / ``param_groups`` / ``state_dict`` /
    ``load_state_dict``) so existing training loops slot it in
    without code changes — the only difference is that ``step()``
    runs as one compiled GPU kernel instead of N sequential per-
    parameter element-wise updates.

    Supported optimizers (13):
        * :class:`~lucid.optim.SGD` — classical, with optional
          momentum / Nesterov / weight-decay branches.
        * :class:`~lucid.optim.Adam`, :class:`~lucid.optim.AdamW` —
          bias-corrected first + second moments (AMSGrad too); coupled
          vs decoupled weight decay respectively.
        * :class:`~lucid.optim.RMSprop` — exponentially-smoothed
          squared gradient; ``centered=True`` keeps the running
          gradient mean as one more state buffer.
        * :class:`~lucid.optim.Adagrad`, :class:`~lucid.optim.Adadelta`,
          :class:`~lucid.optim.Adamax`, :class:`~lucid.optim.NAdam`,
          :class:`~lucid.optim.RAdam`, :class:`~lucid.optim.ASGD`,
          :class:`~lucid.optim.Rprop`, :class:`~lucid.optim.SparseAdam`
          — the eager update rule, with every step-dependent factor fed
          as a per-step scalar.
        * :class:`~lucid.optim.LBFGS` — a closure-less single-step
          subset (per-element Barzilai-Borwein direction); not the eager
          line-search algorithm.

    Every parameter group compiles into the same executable: each
    parameter reads its own group's hyper-parameters.

    **One owner of the optimizer state at a time.**  Compiling hands the
    state over to the wrapper: whatever ``opt`` built up in eager steps
    so far is adopted, and from then on ``opt.step()``,
    ``opt.state_dict()`` and ``opt.load_state_dict()`` all go to the
    wrapper — whichever handle the training loop holds (a loop driven by
    :func:`fused_step` only holds ``opt``), it steps, saves and restores
    the one state.  The checkpoint format is the eager one, so
    checkpoints move freely between eager and compiled runs.  Compiling
    the same optimizer again returns the same wrapper.

    The exception is :class:`~lucid.optim.LBFGS`, whose compiled step is
    a different algorithm: ``opt`` keeps its own eager state and methods,
    and the wrapper's state is its own.

    Parameters
    ----------
    opt : Optimizer
        Concrete optimizer instance whose ``step()`` is being
        lifted.  Held by reference; LR-scheduler callbacks and
        other edits of ``opt.param_groups`` take effect on the next
        step because the compiled step reads the groups every time.

    Returns
    -------
    _CompiledStepBase
        Drop-in optimizer replacement.  The underlying class is one
        of the ``_Compiled<Name>`` subclasses listed above.

    Raises
    ------
    TypeError
        When ``opt`` is not one of the supported optimizer classes —
        the message lists the supported set.
    ValueError
        When ``opt`` has no parameters.

    Examples
    --------
    Drop-in replacement::

        opt = lucid.optim.Adam(model.parameters(), lr=1e-3)
        sched = lucid.optim.lr_scheduler.StepLR(opt, step_size=10)
        copt = compile_optimizer(opt)

        for batch, target in loader:
            copt.zero_grad()
            loss = F.cross_entropy(model(batch), target)
            loss.backward()
            copt.step()             # one MPSGraph executable
            sched.step()            # seen by the next copt.step()

    See Also
    --------
    :func:`fused_step` : single-executable forward + loss + backward
        + update via the ghost-grad mechanism.  Use this when the
        whole step is the unit of work and the model's forward
        compiles cleanly.
    :func:`make_step` : autograd-graph-aware fwd+bwd compile that
        still lets ``loss.backward()`` run a regular eager
        optimizer step.
    """
    from lucid.optim.sgd import SGD
    from lucid.optim.adam import Adam, AdamW
    from lucid.optim.others import (
        RMSprop,
        Adagrad,
        Adadelta,
        Adamax,
        NAdam,
        ASGD,
        RAdam,
        Rprop,
        SparseAdam,
    )
    from lucid.optim.lbfgs import LBFGS

    # One optimizer, one state: compiling it again (a second fused_step
    # over the same optimizer, say) hands back the wrapper that owns it.
    existing = opt.__dict__.get(_OWNER_ATTR)
    if isinstance(existing, _CompiledStepBase):
        return existing

    if isinstance(opt, SGD):
        return _CompiledSGD(opt)
    if isinstance(opt, AdamW):
        return _CompiledAdamW(opt)
    if isinstance(opt, Adam):
        return _CompiledAdam(opt)
    if isinstance(opt, RMSprop):
        return _CompiledRMSprop(opt)
    if isinstance(opt, Adagrad):
        return _CompiledAdagrad(opt)
    if isinstance(opt, Adadelta):
        return _CompiledAdadelta(opt)
    if isinstance(opt, Adamax):
        return _CompiledAdamax(opt)
    if isinstance(opt, NAdam):
        return _CompiledNAdam(opt)
    if isinstance(opt, SparseAdam):
        from lucid.compile._optim.sparse_adam import _CompiledSparseAdam

        return _CompiledSparseAdam(opt)
    if isinstance(opt, Rprop):
        return _CompiledRprop(opt)
    if isinstance(opt, ASGD):
        return _CompiledASGD(opt)
    if isinstance(opt, RAdam):
        return _CompiledRAdam(opt)
    if isinstance(opt, LBFGS):
        # LBFGS compile path supports the closure-less single-step
        # subset only — full closure-driven line search is genuinely
        # incompatible with a fixed MPSGraph executable.
        return _CompiledLBFGS(opt)

    raise TypeError(
        f"compile_optimizer: unsupported optimizer class "
        f"{type(opt).__name__!r}.  Supported: SGD, Adam, AdamW, "
        f"RMSprop, Adagrad, Adadelta, Adamax, NAdam, SparseAdam, "
        f"Rprop, ASGD, RAdam, LBFGS (closure-less single-step subset)."
    )


# ── Common helpers ──────────────────────────────────────────────────

# Attribute on the eager optimizer naming the compiled wrapper that owns
# its state.
_OWNER_ATTR = "_compiled_step"

# Every method of an eager optimizer that reads or advances its state —
# all of them answer for the compiled wrapper once it owns the state.
_STATE_ENTRY_POINTS: tuple[str, ...] = ("step", "state_dict", "load_state_dict")

# Structural choices of one parameter group (weight decay on, momentum
# on, AMSGrad …) — the part of the hyper-parameters that shapes the trace.
_Flags = tuple[object, ...]

# One number in the packed per-step scalar vector:
#   ("g", name, group)  — a group hyper-parameter (or a value derived
#                         from group hyper-parameters only),
#   ("c", name, class)  — a per-parameter value (bias corrections …),
#                         shared by the parameters of one *class*: same
#                         group, step count and scalar state,
#   ("a", "",   param)  — 1.0 when the parameter steps this time, else
#                         0.0 (masked executables only).
_Entry = tuple[str, str, int]

# One executable input or output:
#   ("param", i, "") / ("grad", i, "") / ("state", i, name) /
#   ("scalars", k, "") — the k-th packed scalar vector.
_Slot = tuple[str, int, str]

# A trace-time scalar lookup: name → 0-D tensor in the parameter's dtype.
_ScalarFn = Callable[[str], Tensor]

# One parameter's state buffers by name, and what an ``_update`` hook
# returns: the new parameter and the new state.
_State = dict[str, Tensor]
_Update = tuple[Tensor, _State]

_STRUCT_FORMAT: dict[_C_engine.Dtype, str] = {
    _C_engine.Dtype.F16: "e",
    _C_engine.Dtype.F32: "f",
    _C_engine.Dtype.F64: "d",
}


def _round_state_scalar(value: float, dt: _C_engine.Dtype) -> float:
    """Round ``value`` to the precision the eager engine keeps scalar state at.

    The engine stores NAdam's ``mu_product`` and ASGD's ``eta`` / ``mu``
    as float32 (float64 for a float64 parameter) and rounds them on
    every step — mirrored here so the two paths take the same steps and
    a checkpoint round trip is exact.
    """
    if dt == _C_engine.Dtype.F64:
        return value
    return float(struct.unpack("f", struct.pack("f", value))[0])


def _pack_impl(
    values: Sequence[float], dt: _C_engine.Dtype, dev: _C_engine.Device
) -> _C_engine.TensorImpl:
    """One 1-D tensor of ``values`` in ``dt`` on ``dev`` — a single upload.

    ``struct.pack`` + ``TensorImpl.from_bytes`` costs ~10 µs for a few
    hundred values, against ~2 ms through ``lucid.tensor(list)``'s
    per-element path.
    """
    fmt = _STRUCT_FORMAT.get(dt)
    n = len(values)
    if fmt is None:
        f32 = _C_engine.TensorImpl.from_bytes(
            struct.pack(f"={n}f", *values), [n], _C_engine.Dtype.F32, dev, False
        )
        return _C_engine.astype(f32, dt)
    return _C_engine.TensorImpl.from_bytes(struct.pack(f"={n}{fmt}", *values), [n], dt, dev, False)


def _wrap(impl: _C_engine.TensorImpl) -> Tensor:
    """Wrap an engine impl as a Tensor (no copy)."""
    from lucid._dispatch import _wrap as _dispatch_wrap

    return _dispatch_wrap(impl)


def _zeros_like(t: Tensor) -> Tensor:
    """Allocate a same-shape zero-filled tensor on the same device/dtype."""
    import lucid as _lucid

    return _lucid.zeros(*t.shape, dtype=t.dtype, device=t.device)


def _full_like(t: Tensor, value: float) -> Tensor:
    """Allocate a same-shape tensor filled with ``value``."""
    import lucid as _lucid

    return _lucid.full(tuple(t.shape), value, dtype=t.dtype, device=t.device)


@final
class _ScalarLayout:
    """Where each per-step scalar sits in the packed vectors of one plan.

    Every parameter dtype that takes part gets its own vector holding
    the same entries (the eager engine rounds each scalar to the
    parameter's dtype, so one vector per dtype keeps that).

    Per-parameter scalars are stored once per *class* — parameters of one
    group whose step count and scalar state agree, which in a normal run
    is every parameter of the group.  Each entry costs a slice in the
    executable, so one entry per parameter would make a large model's
    step measurably slower.  A masked plan gives every parameter its own
    class (its active set, and so the classes, change from step to step).
    """

    def __init__(
        self,
        entries: list[_Entry],
        dtypes: list[_C_engine.Dtype],
        device: _C_engine.Device,
        param_class: dict[int, int],
        class_reps: list[int],
    ) -> None:
        self.entries = entries
        self.index: dict[_Entry, int] = {e: k for k, e in enumerate(entries)}
        self.dtypes = dtypes
        self.device = device
        # Member → class index, and class index → a member representing it.
        self.param_class = param_class
        self.class_reps = class_reps


@final
class _TraceScalars:
    """Trace-time view of the packed scalar vectors — one ``unbind`` each."""

    def __init__(self, layout: _ScalarLayout, vectors: dict[_C_engine.Dtype, Tensor]) -> None:
        self._layout = layout
        self._pieces: dict[_C_engine.Dtype, Sequence[Tensor]] = {}
        if layout.entries:
            for dt, vec in vectors.items():
                self._pieces[dt] = vec.unbind(0)

    def get(self, entry: _Entry, dt: _C_engine.Dtype) -> Tensor:
        """The 0-D tensor for ``entry`` in dtype ``dt``."""
        return self._pieces[dt][self._layout.index[entry]]


@final
@dataclass
class _Plan:
    """One compiled update: which parameters, and how it is wired."""

    exe: object
    members: tuple[int, ...]
    masked: bool
    layout: _ScalarLayout
    feeds: list[_Slot]
    targets: list[_Slot]


class _CompiledStepBase:
    r"""Abstract base for the compiled-optimizer wrappers.

    Each concrete subclass (one per supported optimizer family)
    describes its update rule through a handful of hooks; this class
    owns the per-parameter state, the per-step scalar feed, the plan
    cache and the run loop.

    Lifecycle of one :meth:`step`
    -----------------------------
    1. The parameters whose ``grad`` is not ``None`` take part (the
       eager rule).  A parameter taking part for the first time has its
       state initialised now, as the eager engine does.
    2. The executable for *(those parameters, each group's structural
       flags)* is looked up or traced + compiled.  After
       :attr:`_EXACT_PLAN_LIMIT` distinct sets, a single *masked* plan
       over every parameter is used for any new set instead (a per-
       parameter flag selects the old or new value), so a model whose
       active set changes every step does not recompile every step.
    3. Hyper-parameters and per-parameter scalars are packed into one
       vector per dtype and the executable writes new parameters and
       state in place.
    4. The step count (and NAdam / ASGD scalar state) of each parameter
       that took part advances.

    Subclass hooks
    --------------
    :meth:`_flags(group)`
        Structural choices of a group — anything that changes the
        trace.  Values that only change numbers stay out.
    :meth:`_state_names(flags)`
        Ordered names of the per-parameter state buffers, under the
        reference framework's names (``exp_avg``, ``momentum_buffer`` …).
    :meth:`_group_scalar_names(flags)` / :meth:`_param_scalar_names(flags)`
        Names of the per-group and per-parameter scalars the update reads.
    :meth:`_group_values(group, flags)`
        This step's group numbers (both the fed scalars and the raw
        hyper-parameters :meth:`_param_values` needs).
    :meth:`_param_values(i, step, gv, flags)`
        This step's per-parameter numbers for parameter ``i`` taking its
        ``step``-th step.
    :meth:`_update(p, g, state, flags, gs, ps)`
        The update math, emitted under the active tracer.
    :meth:`_class_key(i)`, :meth:`_exports_state(i, name)`,
    :meth:`_loaded_state(i, name)`
        For per-parameter state that is not a step count: what the
        per-parameter scalars depend on, and which buffers a checkpoint
        carries (SGD's momentum buffer only once it has started).

    Attributes
    ----------
    _opt : Optimizer
        The wrapped eager optimizer.  Its ``param_groups`` are the live
        source of every hyper-parameter.
    _params : list[Tensor]
        Flat parameter list across every group, in ``param_groups``
        order (the same flat index the eager ``state_dict`` uses).
    _steps : list[int]
        Per-parameter step count.
    _state : list[dict[str, Tensor]]
        Per-parameter state buffers.  A buffer keeps its Tensor identity
        for life — loads and (re)initialisation write into it.
    """

    # Distinct exact plans built before switching to the masked plan.
    _EXACT_PLAN_LIMIT: int = 4
    # Hard bound on cached executables (oldest dropped first).
    _MAX_PLANS: int = 16
    # Whether the eager optimizer's state carries a per-parameter "step".
    _EXPORTS_STEP: bool = True
    # Per-parameter Python scalar state exported next to "step".
    _PSTATE_NAMES: tuple[str, ...] = ()
    # Whether the wrapper takes over the eager optimizer's state — adopts
    # it, steps for it and checkpoints it.  False only where the compiled
    # algorithm is not the eager one (LBFGS).
    _OWNS_EAGER_STATE: bool = True
    # State buffers a parameter may hold whatever its group's flags are now:
    # one kept after its feature was switched off (AMSGrad's running
    # maximum, a momentum buffer), which the eager state keeps, exports and
    # restores as the reference framework's does.
    _OPTIONAL_STATE: tuple[str, ...] = ()

    def __init__(self, opt: Optimizer) -> None:
        """Set up the shared compile-step plumbing on top of ``opt``.

        Parameters
        ----------
        opt : Optimizer
            The eager optimizer instance whose math is being lifted
            into a compiled executable.  Held by reference so
            scheduler callbacks (LR updates, etc.) still flow
            through naturally.

        Raises
        ------
        ValueError
            If ``opt`` has no parameters.
        """
        self._opt = opt
        self._params: list[Tensor] = []
        self._group_of: list[int] = []
        self._steps: list[int] = []
        self._initialized: list[bool] = []
        self._state: list[dict[str, Tensor]] = []
        self._pstate: list[dict[str, float]] = []
        self._zero_grads: dict[int, Tensor] = {}
        self._plans: dict[tuple[object, ...], _Plan] = {}
        self._exact_plans_built: int = 0
        self._group_ids: tuple[tuple[int, ...], ...] = ()
        # Group values of the step being taken — read by ``_commit``.
        self._step_gv: list[dict[str, float]] = []
        # Whether this optimizer reads per-parameter (step-dependent) scalars.
        self._class_scalars = (
            type(self)._param_scalar_names is not _CompiledStepBase._param_scalar_names
        )
        self._sync_params()
        if not self._params:
            raise ValueError("compile_optimizer: optimizer has no trainable parameters")
        self._take_over(opt)

    def _take_over(self, opt: Optimizer) -> None:
        """The one place the optimizer state changes owner (``compile_optimizer``).

        One owner at a time: the eager engines and this wrapper must never
        both advance, or a checkpoint taken through one handle misses the
        steps taken through the other.  So the wrapper adopts whatever
        eager steps built so far, then every entry point that reads or
        advances the state on ``opt`` (``_STATE_ENTRY_POINTS``)
        answers for the wrapper.  A class whose compiled algorithm is not
        the eager one (``_OWNS_EAGER_STATE = False``) leaves ``opt`` the
        owner of its own state.
        """
        opt.__dict__[_OWNER_ATTR] = self
        if not self._OWNS_EAGER_STATE:
            return
        from lucid.optim.optimizer import Optimizer as _Optimizer

        # Keyed on the state itself, not on how the class keeps it: an
        # optimizer with Python-side state (SparseAdam) never builds engines.
        eager = _Optimizer.state_dict(opt)
        if eager["state"]:
            self.load_state_dict(eager)
        for name in _STATE_ENTRY_POINTS:
            opt.__dict__[name] = getattr(self, name)

    @override
    def __getstate__(self) -> dict[str, object]:
        """Pickle without the executables — they are rebuilt on the next step."""
        state = dict(self.__dict__)
        state["_plans"] = {}
        state["_exact_plans_built"] = 0
        state["_zero_grads"] = {}
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore a pickled wrapper; executables are rebuilt lazily."""
        self.__dict__.update(state)

    # ── Drop-in API surface ──────────────────────────────────────

    @property
    def param_groups(self) -> list[dict[str, object]]:
        """Delegate to the wrapped optimizer's parameter groups."""
        return self._opt.param_groups

    @property
    def state(self) -> dict[int, dict[str, object]]:
        """The state snapshot of the last :meth:`state_dict` / :meth:`load_state_dict`."""
        return self._opt.state

    @property
    def defaults(self) -> dict[str, object]:
        """Delegate to the wrapped optimizer's hyperparameter defaults."""
        return self._opt.defaults

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Forward to the wrapped optimizer's ``zero_grad`` — drop-in API."""
        self._opt.zero_grad(set_to_none=set_to_none)

    def add_param_group(self, group: dict[str, object]) -> None:
        """Add a parameter group to the wrapped optimizer; the next step sees it."""
        self._opt.add_param_group(group)
        self._sync_params()

    def _sync_hyperparams(self) -> None:
        """LR-scheduler hook — the compiled step reads ``param_groups`` itself.

        Forwarded to the wrapped optimizer so its own engines (if it was
        ever stepped eagerly) stay in step too.
        """
        self._opt._sync_hyperparams()

    def state_dict(self) -> dict[str, object]:
        """Checkpoint in the eager optimizer's format.

        ``state`` is keyed by flat parameter index and holds, for every
        parameter that has stepped, its ``step`` (0-d int64), any scalar
        state (NAdam ``mu_product``, ASGD ``eta`` / ``mu``) and its
        buffers under the reference framework's names — the same layout
        the eager optimizer of this class writes, so either can load the
        other's checkpoint.  ``param_groups`` mirrors the live groups with
        ``params`` replaced by flat indices.
        """
        import lucid as _lucid

        self._sync_params()
        id_map = {id(p): k for k, p in enumerate(self._params)}
        groups_out: list[dict[str, object]] = []
        for group in self._opt.param_groups:
            g: dict[str, object] = {k: v for k, v in group.items() if k != "params"}
            g["params"] = [id_map[id(p)] for p in cast(list[Tensor], group["params"])]
            groups_out.append(g)

        # Checks an edited group, as the eager ``state_dict`` does.
        self._flags_now()
        state: dict[int, dict[str, object]] = {}
        for i, p in enumerate(self._params):
            if not self._initialized[i]:
                continue
            entry: dict[str, object] = {}
            if self._EXPORTS_STEP:
                entry["step"] = self._step_entry(i)
            scalar_dt = _float32_or_wider(p.dtype)
            for name in self._PSTATE_NAMES:
                if (value := self._pstate[i].get(name)) is not None:
                    entry[name] = _lucid.tensor(value, dtype=scalar_dt).numpy()
            # Every buffer the parameter holds, including one kept after its
            # feature was switched off — the eager state keeps it too.
            for name, buf in self._state[i].items():
                if self._exports_state(i, name):
                    entry[name] = buf.detach().numpy().copy()
            if entry:
                state[i] = entry
        self._opt.state = state
        return {"state": state, "param_groups": groups_out}

    def load_state_dict(self, sd: dict[str, object]) -> None:
        """Restore a checkpoint written by this class or by its eager optimizer.

        Group hyper-parameters go back into the live ``param_groups``;
        each parameter's ``step``, scalar state and buffers are written
        into the compiled state (in place — the executables stay valid).
        A parameter without an entry keeps its current state, as in the
        eager loader.

        Parameters
        ----------
        sd : dict
            An ``optimizer.state_dict()`` payload.

        Raises
        ------
        ValueError
            When the group count differs, or a saved buffer's shape does
            not match its parameter.
        """
        from lucid.autograd._grad_mode import no_grad
        from lucid.optim.optimizer import _from_host

        loaded_groups = cast(list[dict[str, object]], sd["param_groups"])
        if len(loaded_groups) != len(self._opt.param_groups):
            raise ValueError(
                f"loaded state_dict has {len(loaded_groups)} param_groups but "
                f"optimizer has {len(self._opt.param_groups)}"
            )
        for g_new, g_old in zip(self._opt.param_groups, loaded_groups, strict=True):
            for k, v in g_old.items():
                if k != "params":
                    g_new[k] = v
        self._sync_params()
        loaded = cast(dict[int, dict[str, object]], sd.get("state", {}))
        self._opt.state = loaded
        flags = self._flags_now()
        groups = self._opt.param_groups
        for raw_idx, entry in loaded.items():
            i = int(raw_idx)
            if i < 0 or i >= len(self._params) or not entry:
                continue
            p = self._params[i]
            gi = self._group_of[i]
            self._ensure_state(i, flags[gi], groups[gi])
            if not self._initialized[i]:
                self._init_pstate(i, groups[gi])
                self._initialized[i] = True
            for name, value in entry.items():
                if name == "step":
                    self._steps[i] = int(cast(int, value))
                elif name in self._PSTATE_NAMES:
                    self._pstate[i][name] = _round_state_scalar(
                        float(cast(float, value)), p._impl.dtype
                    )
                elif name in self._state[i] or name in self._OPTIONAL_STATE:
                    # A kept buffer comes back whatever the flags are now,
                    # as the eager loader restores it.
                    buf = self._state[i].get(name)
                    if buf is None:
                        buf = self._state[i][name] = self._init_state(i, name, groups[gi])
                    # The eager loader's host-array door: it keeps a 0-d
                    # entry 0-d.
                    host = _from_host(value, buf._impl.device)
                    src = _wrap(_C_engine.astype(host, buf._impl.dtype))
                    if tuple(src.shape) != tuple(buf.shape):
                        raise ValueError(
                            f"load_state_dict: state {name!r} of parameter {i} has "
                            f"shape {tuple(src.shape)}, expected {tuple(buf.shape)}"
                        )
                    with no_grad():
                        buf.copy_(src)
                    self._loaded_state(i, name)
        self._opt._sync_hyperparams()

    # ── Public step() ────────────────────────────────────────────

    def _count_step(self) -> None:
        """Count a step that skips ``opt.step``, for the schedulers on either handle."""
        _record_step(self)
        _record_step(self._opt)

    @_counts_as_step  # so ``opt.step = self.step`` is no override to a scheduler
    def step(self, closure: Callable[..., Tensor] | None = None) -> Tensor | None:
        """Run one compiled update step, in place; returns ``closure()``'s loss or ``None``.

        Parameters
        ----------
        closure : callable, optional
            Called once before the update, as the eager optimizers call it.
        """
        from lucid._dispatch import _unwrap

        self._count_step()
        loss: Tensor | None = closure() if closure is not None else None
        self._sync_params()
        active = tuple(i for i, p in enumerate(self._params) if p.grad is not None)
        if not active:
            return loss
        flags = self._flags_now()
        groups = self._opt.param_groups
        for i in active:
            self._activate(i, flags, groups)
        plan = self._plan_for(active, flags)
        vectors = self._scalar_vectors(plan.layout, flags, set(active) if plan.masked else None)
        feeds = [self._feed(slot, vectors) for slot in plan.feeds]
        targets = [_unwrap(self._slot_tensor(slot)) for slot in plan.targets]
        _C_engine.compile.run_executable_inplace(plan.exe, feeds, targets)
        self._commit(active)
        return loss

    # ── Parameter bookkeeping ────────────────────────────────────

    def _sync_params(self) -> None:
        """Re-flatten the groups when they hold other parameters, by identity.

        A parameter put in another's place keeps the sizes (LCD-560).  State
        and step counts follow each parameter; every executable is dropped.
        """
        groups = [cast(list[Tensor], g["params"]) for g in self._opt.param_groups]
        layout = tuple(tuple(map(id, ps)) for ps in groups)  # _params holds each
        if layout == self._group_ids:
            return
        old = {id(p): k for k, p in enumerate(self._params)}
        params: list[Tensor] = []
        group_of: list[int] = []
        for gi, ps in enumerate(groups):
            for p in ps:
                params.append(p)
                group_of.append(gi)

        def carry[T](values: list[T], default: Callable[[], T]) -> list[T]:
            out: list[T] = []
            for p in params:
                k = old.get(id(p))
                out.append(values[k] if k is not None else default())
            return out

        self._steps = carry(self._steps, lambda: 0)
        self._initialized = carry(self._initialized, lambda: False)
        self._state = carry(self._state, dict)
        self._pstate = carry(self._pstate, dict)
        self._params = params
        self._group_of = group_of
        self._group_ids = layout
        self._zero_grads = {}
        self._plans = {}
        self._exact_plans_built = 0

    def _flags_now(self) -> tuple[_Flags, ...]:
        """Every group's structural flags, read from the live groups.

        Each step (``step`` and ``fused_step``) and each checkpoint reads the
        groups here first, so an edited group is checked here: through the
        eager optimizer, whose hyper-parameter table and rules are the one
        statement of them, once per change.  A value the eager step would
        refuse raises ``InvalidArgument`` before anything is initialised,
        packed or run.
        """
        self._opt._sync_group_hyperparams()
        return tuple(self._flags(g) for g in self._opt.param_groups)

    def _ensure_state(self, i: int, flags: _Flags, group: dict[str, object]) -> None:
        """Allocate any state buffer parameter ``i`` needs and lacks."""
        state = self._state[i]
        for name in self._state_names(flags):
            if name not in state:
                state[name] = self._init_state(i, name, group)

    def _activate(self, i: int, flags: tuple[_Flags, ...], groups: list[dict[str, object]]) -> None:
        """Make parameter ``i`` ready to step; first time = eager's slot init."""
        gi = self._group_of[i]
        group = groups[gi]
        if self._initialized[i]:
            self._ensure_state(i, flags[gi], group)
            return
        from lucid.autograd._grad_mode import no_grad

        state = self._state[i]
        for name in self._state_names(flags[gi]):
            fresh = self._init_state(i, name, group)
            buf = state.get(name)
            if buf is None:
                state[name] = fresh
            else:
                # A placeholder a masked plan allocated — keep its identity.
                with no_grad():
                    buf.copy_(fresh)
        self._init_pstate(i, group)
        self._initialized[i] = True

    def _commit(self, members: Sequence[int]) -> None:
        """Advance the step count (and scalar state) of each parameter that stepped."""
        gv = self._step_gv
        for i in members:
            self._steps[i] += 1
            self._advance(i, self._steps[i], gv[self._group_of[i]])

    # ── Plans ────────────────────────────────────────────────────

    def _plan_for(self, active: tuple[int, ...], flags: tuple[_Flags, ...]) -> _Plan:
        """The executable for this step's active set (see the class docstring)."""
        key: tuple[object, ...] = (active, False, flags, self._partition(active))
        plan = self._plans.get(key)
        if plan is not None:
            return plan
        if self._exact_plans_built < self._EXACT_PLAN_LIMIT:
            self._exact_plans_built += 1
            plan = self._build_plan(active, False, flags)
            self._remember(key, plan)
            return plan
        everyone = tuple(range(len(self._params)))
        mkey: tuple[object, ...] = (everyone, True, flags)
        plan = self._plans.get(mkey)
        if plan is None:
            plan = self._build_plan(everyone, True, flags)
            self._remember(mkey, plan)
        return plan

    def _remember(self, key: tuple[object, ...], plan: _Plan) -> None:
        """Cache ``plan``, dropping the oldest beyond :attr:`_MAX_PLANS`."""
        self._plans[key] = plan
        while len(self._plans) > self._MAX_PLANS:
            self._plans.pop(next(iter(self._plans)))

    def _build_plan(
        self, members: tuple[int, ...], masked: bool, flags: tuple[_Flags, ...]
    ) -> _Plan:
        """Trace + compile the update of ``members``."""
        from lucid._dispatch import _unwrap
        from lucid.autograd._grad_mode import no_grad
        from lucid.compile import _tracing

        groups = self._opt.param_groups
        for i in members:
            # A masked plan reads every member's state, stepped or not.
            self._ensure_state(i, flags[self._group_of[i]], groups[self._group_of[i]])
        registry: dict[int, _Slot] = {}
        for i in members:
            registry[id(_unwrap(self._params[i]))] = ("param", i, "")
            for name, buf in self._state[i].items():
                registry[id(_unwrap(buf))] = ("state", i, name)
        grads: dict[int, Tensor] = {}
        for i in members:
            grads[i] = _zeros_like(self._params[i])
            registry[id(_unwrap(grads[i]))] = ("grad", i, "")
        layout = self._new_layout(members, masked, flags)
        vectors = self._trace_vectors(layout)
        for k, dt in enumerate(layout.dtypes):
            registry[id(_unwrap(vectors[dt]))] = ("scalars", k, "")

        with no_grad():
            with _tracing() as tracer:
                outputs, targets = self._emit(members, grads, layout, vectors, flags, masked)

        graph = tracer.graph
        if not graph.ops:
            raise RuntimeError(
                "compile_optimizer: empty trace — update function emitted no ops (unexpected)."
            )
        ext = tracer.external_feeds
        explicit_outputs: list[int] = []
        for out_t in outputs:
            tid = tracer.lookup_id(_unwrap(out_t))
            if tid is None:
                raise RuntimeError(
                    "compile_optimizer: trace output tensor has no id — "
                    "the update function produced a tensor that wasn't "
                    "captured by the tracer (bug)."
                )
            explicit_outputs.append(int(tid))
        try:
            exe = _C_engine.compile.compile_or_cached(graph, ext, False, [], explicit_outputs)
        except RuntimeError as e:
            raise RuntimeError(f"compile_optimizer: compile_or_cached failed: {e}")
        if exe is None:
            raise RuntimeError(
                "compile_optimizer: compile_or_cached returned None — "
                "an op in the update graph has no emitter."
            )
        feeds: list[_Slot] = []
        for tid in exe.input_ids:
            impl = ext.get(tid)
            if impl is None:
                raise RuntimeError(f"compile_optimizer: input id {tid} not in external_feeds")
            slot = registry.get(id(impl))
            if slot is None:
                raise RuntimeError(
                    f"compile_optimizer: input id {tid} not in our "
                    "registry — trace captured an unexpected tensor"
                )
            feeds.append(slot)
        if len(targets) != len(exe.output_ids):
            raise RuntimeError(
                f"compile_optimizer: target count {len(targets)} != "
                f"executable output count {len(exe.output_ids)}"
            )
        return _Plan(exe, members, masked, layout, feeds, targets)

    def _feed(self, slot: _Slot, vectors: Sequence[_C_engine.TensorImpl]) -> _C_engine.TensorImpl:
        """The live engine impl bound to one executable input."""
        from lucid._dispatch import _unwrap

        kind, i, name = slot
        if kind == "param":
            return _unwrap(self._params[i])
        if kind == "state":
            return _unwrap(self._state[i][name])
        if kind == "scalars":
            return vectors[i]
        # "grad" — a parameter that does not step (masked plan) reads zeros.
        p = self._params[i]
        g = p.grad
        if g is None:
            z = self._zero_grads.get(i)
            if z is None:
                z = self._zero_grads[i] = _zeros_like(p)
            return _unwrap(z)
        if g.dtype != p.dtype:
            g = g.to(p.dtype)
        return _unwrap(g)

    def _slot_tensor(self, slot: _Slot) -> Tensor:
        """The live parameter or state tensor named by an output slot."""
        kind, i, name = slot
        if kind == "param":
            return self._params[i]
        return self._state[i][name]

    # ── Scalars ──────────────────────────────────────────────────

    def _partition(self, members: Sequence[int]) -> tuple[int, ...]:
        """Class index of each member: same group, step count and scalar state.

        Part of an exact plan's key.  Classes only split when parameters
        stop stepping together (a freeze, a load) and then stay split, so
        a steady run keeps one key.
        """
        if not self._class_scalars:
            # No per-parameter scalars: nothing to tell the classes apart by.
            return (0,) * len(members)
        seen: dict[tuple[object, ...], int] = {}
        out: list[int] = []
        for i in members:
            key = (self._group_of[i], *self._class_key(i))
            out.append(seen.setdefault(key, len(seen)))
        return tuple(out)

    def _new_layout(
        self, members: Sequence[int], masked: bool, flags: tuple[_Flags, ...]
    ) -> _ScalarLayout:
        """Every scalar the update of ``members`` reads, in one fixed order."""
        from lucid._dispatch import _unwrap

        classes = tuple(range(len(members))) if masked else self._partition(members)
        param_class = dict(zip(members, classes, strict=True))
        class_reps: list[int] = []
        for i, c in zip(members, classes, strict=True):
            if c == len(class_reps):
                class_reps.append(i)
        entries: list[_Entry] = []
        seen_groups: set[int] = set()
        for i in members:
            gi = self._group_of[i]
            if gi not in seen_groups:
                seen_groups.add(gi)
                entries.extend(("g", n, gi) for n in self._group_scalar_names(flags[gi]))
        for c, rep in enumerate(class_reps):
            names = self._param_scalar_names(flags[self._group_of[rep]])
            entries.extend(("c", n, c) for n in names)
        if masked:
            entries.extend(("a", "", i) for i in members)
        dtypes: list[_C_engine.Dtype] = []
        for i in members:
            dt = _unwrap(self._params[i]).dtype
            if dt not in dtypes:
                dtypes.append(dt)
        device = _unwrap(self._params[members[0]]).device
        return _ScalarLayout(entries, dtypes, device, param_class, class_reps)

    def _trace_vectors(self, layout: _ScalarLayout) -> dict[_C_engine.Dtype, Tensor]:
        """Trace-time stand-ins for the packed vectors (values do not matter)."""
        n = len(layout.entries)
        return {dt: _wrap(_pack_impl([0.0] * n, dt, layout.device)) for dt in layout.dtypes}

    def _scalar_vectors(
        self, layout: _ScalarLayout, flags: tuple[_Flags, ...], active: set[int] | None
    ) -> list[_C_engine.TensorImpl]:
        """This step's packed scalar vectors, one per dtype of ``layout``."""
        groups = self._opt.param_groups
        gv = [self._group_values(g, flags[k]) for k, g in enumerate(groups)]
        self._step_gv = gv
        if not layout.entries:
            return []
        memo: dict[tuple[object, ...], dict[str, float]] = {}
        cv: list[dict[str, float]] = []
        for i in layout.class_reps:
            gi = self._group_of[i]
            step = self._steps[i] + 1
            mkey = (gi, *self._class_key(i))
            vals = memo.get(mkey)
            if vals is None:
                vals = memo[mkey] = self._param_values(i, step, gv[gi], flags[gi])
            cv.append(vals)
        values: list[float] = []
        for kind, name, idx in layout.entries:
            if kind == "g":
                values.append(gv[idx][name])
            elif kind == "c":
                values.append(cv[idx][name])
            else:
                values.append(1.0 if active is not None and idx in active else 0.0)
        return [_pack_impl(values, dt, layout.device) for dt in layout.dtypes]

    def _pstate_key(self, i: int) -> tuple[float, ...]:
        """Parameter ``i``'s scalar state as a hashable key."""
        st = self._pstate[i]
        return tuple(st.get(n, 0.0) for n in self._PSTATE_NAMES)

    def _class_key(self, i: int) -> tuple[object, ...]:
        """What parameter ``i``'s per-parameter scalars depend on, within its group.

        Parameters of one group with equal keys share their scalars (one
        *class*): by default the step count and the scalar state.
        """
        return (self._steps[i], *self._pstate_key(i))

    # ── Trace ────────────────────────────────────────────────────

    def _emit(
        self,
        members: Sequence[int],
        grads: dict[int, Tensor],
        layout: _ScalarLayout,
        vectors: dict[_C_engine.Dtype, Tensor],
        flags: tuple[_Flags, ...],
        masked: bool,
    ) -> tuple[list[Tensor], list[_Slot]]:
        """Emit the update of every member under the active tracer.

        Returns the new tensors — every member's parameter first (in
        ``members`` order), then their state buffers — and the slot each
        one is written to.  ``fused_step`` relies on the parameters
        coming first.
        """
        import lucid as _lucid
        from lucid._dispatch import _unwrap

        sc = _TraceScalars(layout, vectors)
        new_params: list[Tensor] = []
        new_states: list[Tensor] = []
        p_slots: list[_Slot] = []
        s_slots: list[_Slot] = []
        for i in members:
            p = self._params[i]
            gi = self._group_of[i]
            fl = flags[gi]
            dt = _unwrap(p).dtype
            names = self._state_names(fl)
            st = {n: self._state[i][n] for n in names}

            def gs(name: str, gi: int = gi, dt: _C_engine.Dtype = dt) -> Tensor:
                return sc.get(("g", name, gi), dt)

            def ps(name: str, c: int = layout.param_class[i], dt: _C_engine.Dtype = dt) -> Tensor:
                return sc.get(("c", name, c), dt)

            new_p, new_st = self._update(p, grads[i], st, fl, gs, ps)
            if masked:
                on = sc.get(("a", "", i), dt) > 0.5
                new_p = _lucid.where(on, new_p, p)
                new_st = {n: _lucid.where(on, new_st[n], st[n]) for n in names}
            new_params.append(new_p)
            p_slots.append(("param", i, ""))
            for n in names:
                new_states.append(new_st[n])
                s_slots.append(("state", i, n))
        return new_params + new_states, p_slots + s_slots

    # ── Subclass hooks ───────────────────────────────────────────

    def _flags(self, group: dict[str, object]) -> _Flags:
        """Structural choices of ``group`` — part of the executable's key."""
        del group
        return ()

    def _state_names(self, flags: _Flags) -> tuple[str, ...]:
        """Ordered state-buffer names for a group with ``flags``."""
        raise NotImplementedError

    def _group_scalar_names(self, flags: _Flags) -> tuple[str, ...]:
        """Per-group scalars the update reads (keys of :meth:`_group_values`)."""
        del flags
        return ()

    def _param_scalar_names(self, flags: _Flags) -> tuple[str, ...]:
        """Per-parameter scalars the update reads (keys of :meth:`_param_values`)."""
        del flags
        return ()

    def _group_values(self, group: dict[str, object], flags: _Flags) -> dict[str, float]:
        """This step's numbers for ``group``."""
        del group, flags
        return {}

    def _param_values(
        self, i: int, step: int, gv: dict[str, float], flags: _Flags
    ) -> dict[str, float]:
        """Parameter ``i``'s numbers for its ``step``-th step."""
        del i, step, gv, flags
        return {}

    def _init_state(self, i: int, name: str, group: dict[str, object]) -> Tensor:
        """A fresh state buffer for parameter ``i`` — zeros unless overridden."""
        del name, group
        return _zeros_like(self._params[i])

    def _init_pstate(self, i: int, group: dict[str, object]) -> None:
        """Initialise parameter ``i``'s Python scalar state (first step)."""
        del i, group

    def _advance(self, i: int, step: int, gv: dict[str, float]) -> None:
        """Update parameter ``i``'s scalar state after its ``step``-th step."""
        del i, step, gv

    def _step_entry(self, i: int) -> object:
        """Parameter ``i``'s ``step`` as the eager checkpoint holds it: 0-d int64."""
        import lucid as _lucid

        return _lucid.tensor(self._steps[i], dtype=_lucid.int64).numpy()

    def _exports_state(self, i: int, name: str) -> bool:
        """Whether parameter ``i``'s buffer ``name`` goes into a checkpoint."""
        del i, name
        return True

    def _loaded_state(self, i: int, name: str) -> None:
        """Note that a checkpoint wrote parameter ``i``'s buffer ``name``."""
        del i, name

    def _update(
        self, p: Tensor, g: Tensor, state: _State, flags: _Flags, gs: _ScalarFn, ps: _ScalarFn
    ) -> _Update:
        """Emit one parameter's update; returns the new parameter and state."""
        raise NotImplementedError


def _wd_on(group: dict[str, object], default: float = 0.0) -> bool:
    """Whether ``group`` applies weight decay."""
    return _hp(group, "weight_decay", default) != 0.0


# SGD's per-parameter record of whether the momentum buffer has started —
# kept with the scalar state, but not part of the eager checkpoint (the
# buffer's presence there says the same thing).
_MOMENTUM_STARTED = "momentum_started"


# ── SGD ─────────────────────────────────────────────────────────────


@final
class _CompiledSGD(_CompiledStepBase):
    r"""Compiled :class:`~lucid.optim.SGD` (momentum / Nesterov / weight decay).

    .. math::

        g_t       &\leftarrow g_t + \lambda \theta_t \\
        v_{t+1}   &= \mu v_t + (1 - \tau) g_t \\
        \theta_{t+1} &= \theta_t - \eta \,(g_t + \mu v_{t+1}
                         \text{ (Nesterov) or } v_{t+1})

    A parameter's first momentum step starts the buffer at the gradient,
    :math:`v_1 = g_1`, undamped, as the eager engine and the reference
    framework do — also when momentum is switched on part-way through.
    Whether a parameter's buffer has started is a per-parameter scalar fed
    to the executable, so it costs no retrace.  A buffer stays after
    momentum is switched off, and resumes when it is switched back on.  The
    eager optimizer's state holds no ``step`` for SGD, and neither does
    this one's.

    See Also
    --------
    :class:`lucid.optim.SGD` : eager counterpart.
    """

    _EXPORTS_STEP = False
    _OPTIONAL_STATE = ("momentum_buffer",)

    @override
    def _flags(self, group: dict[str, object]) -> _Flags:
        return (
            _wd_on(group),
            _hp(group, "momentum", 0.0) != 0.0,
            bool(group.get("nesterov", False)),
        )

    @override
    def _state_names(self, flags: _Flags) -> tuple[str, ...]:
        return ("momentum_buffer",) if flags[1] else ()

    @override
    def _group_scalar_names(self, flags: _Flags) -> tuple[str, ...]:
        names = ["lr"]
        if flags[0]:
            names.append("weight_decay")
        if flags[1]:
            names += ["momentum", "one_minus_dampening"]
        return tuple(names)

    @override
    def _param_scalar_names(self, flags: _Flags) -> tuple[str, ...]:
        return ("first_momentum_step",) if flags[1] else ()

    @override
    def _param_values(
        self, i: int, step: int, gv: dict[str, float], flags: _Flags
    ) -> dict[str, float]:
        return {"first_momentum_step": 0.0 if self._momentum_started(i) else 1.0}

    @override
    def _class_key(self, i: int) -> tuple[object, ...]:
        # The step count does not enter SGD's update; only whether the
        # momentum buffer has started does.
        return (self._momentum_started(i),)

    @override
    def _init_pstate(self, i: int, group: dict[str, object]) -> None:
        self._pstate[i][_MOMENTUM_STARTED] = 0.0

    @override
    def _advance(self, i: int, step: int, gv: dict[str, float]) -> None:
        if gv["momentum"] != 0.0:
            self._pstate[i][_MOMENTUM_STARTED] = 1.0

    @override
    def _exports_state(self, i: int, name: str) -> bool:
        # The eager state has no momentum buffer before its first step.
        return name != "momentum_buffer" or self._momentum_started(i)

    @override
    def _loaded_state(self, i: int, name: str) -> None:
        if name == "momentum_buffer":
            self._pstate[i][_MOMENTUM_STARTED] = 1.0

    def _momentum_started(self, i: int) -> bool:
        """Whether parameter ``i``'s momentum buffer has taken its first step."""
        return self._pstate[i].get(_MOMENTUM_STARTED, 0.0) != 0.0

    @override
    def _group_values(self, group: dict[str, object], flags: _Flags) -> dict[str, float]:
        return {
            "lr": _hp(group, "lr", 0.0),
            "weight_decay": _hp(group, "weight_decay", 0.0),
            "momentum": _hp(group, "momentum", 0.0),
            "one_minus_dampening": 1.0 - _hp(group, "dampening", 0.0),
        }

    @override
    def _update(
        self, p: Tensor, g: Tensor, state: _State, flags: _Flags, gs: _ScalarFn, ps: _ScalarFn
    ) -> _Update:
        import lucid as _lucid

        wd_on, mom_on, nesterov = flags
        if wd_on:
            g = g + gs("weight_decay") * p
        new_state: dict[str, Tensor] = {}
        if mom_on:
            mu = gs("momentum")
            buf = _lucid.where(
                ps("first_momentum_step") > 0.5,
                g,
                mu * state["momentum_buffer"] + gs("one_minus_dampening") * g,
            )
            new_state["momentum_buffer"] = buf
            eff = g + mu * buf if nesterov else buf
        else:
            eff = g
        return p - gs("lr") * eff, new_state


# ── Adam / AdamW ────────────────────────────────────────────────────


class _CompiledAdam(_CompiledStepBase):
    r"""Compiled :class:`~lucid.optim.Adam` (AMSGrad included).

    Follows the eager GPU kernel's arrangement, which folds both bias
    corrections into two per-parameter scalars:

    .. math::

        m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad
        v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
        \theta_t &= \theta_{t-1} - \eta_{\text{eff}} \,
                    m_t / (\sqrt{v_t} + \varepsilon_{\text{eff}}),
        \quad \eta_{\text{eff}} = \eta \sqrt{1-\beta_2^t} / (1-\beta_1^t),
        \ \varepsilon_{\text{eff}} = \varepsilon \sqrt{1-\beta_2^t}

    Weight decay is folded into the gradient (coupled L2).  With
    ``amsgrad=True`` the denominator uses the running maximum of ``v``.

    See Also
    --------
    :class:`lucid.optim.Adam` : eager counterpart.
    :class:`_CompiledAdamW` : decoupled-weight-decay variant.
    """

    _DECOUPLED: bool = False
    _DEFAULT_WD: float = 0.0
    _OPTIONAL_STATE = ("max_exp_avg_sq",)

    @override
    def _flags(self, group: dict[str, object]) -> _Flags:
        return (_wd_on(group, self._DEFAULT_WD), bool(group.get("amsgrad", False)))

    @override
    def _state_names(self, flags: _Flags) -> tuple[str, ...]:
        if flags[1]:
            return ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")
        return ("exp_avg", "exp_avg_sq")

    @override
    def _group_scalar_names(self, flags: _Flags) -> tuple[str, ...]:
        names = ["beta1", "one_minus_beta1", "beta2", "one_minus_beta2"]
        if flags[0]:
            names.append("wd_factor" if self._DECOUPLED else "weight_decay")
        return tuple(names)

    @override
    def _param_scalar_names(self, flags: _Flags) -> tuple[str, ...]:
        return ("lr_eff", "eps_eff")

    @override
    def _group_values(self, group: dict[str, object], flags: _Flags) -> dict[str, float]:
        lr = _hp(group, "lr", 1e-3)
        b1 = _hp(group, "beta1", 0.9)
        b2 = _hp(group, "beta2", 0.999)
        wd = _hp(group, "weight_decay", self._DEFAULT_WD)
        return {
            "lr": lr,
            "eps": _hp(group, "eps", 1e-8),
            "beta1": b1,
            "one_minus_beta1": 1.0 - b1,
            "beta2": b2,
            "one_minus_beta2": 1.0 - b2,
            "weight_decay": wd,
            "wd_factor": 1.0 - lr * wd,
        }

    @override
    def _param_values(
        self, i: int, step: int, gv: dict[str, float], flags: _Flags
    ) -> dict[str, float]:
        bc1 = 1.0 - gv["beta1"] ** float(step)
        bc2 = 1.0 - gv["beta2"] ** float(step)
        sqrt_bc2 = math.sqrt(bc2)
        return {"lr_eff": gv["lr"] * sqrt_bc2 / bc1, "eps_eff": gv["eps"] * sqrt_bc2}

    @override
    def _update(
        self, p: Tensor, g: Tensor, state: _State, flags: _Flags, gs: _ScalarFn, ps: _ScalarFn
    ) -> _Update:
        import lucid as _lucid

        wd_on, amsgrad = flags
        base = p
        if wd_on:
            if self._DECOUPLED:
                base = p * gs("wd_factor")
            else:
                g = g + gs("weight_decay") * p
        m = gs("beta1") * state["exp_avg"] + gs("one_minus_beta1") * g
        v = gs("beta2") * state["exp_avg_sq"] + gs("one_minus_beta2") * (g * g)
        new_state = {"exp_avg": m, "exp_avg_sq": v}
        v_used = v
        if amsgrad:
            v_used = _lucid.maximum(state["max_exp_avg_sq"], v)
            new_state["max_exp_avg_sq"] = v_used
        denom = v_used.sqrt() + ps("eps_eff")
        return base - ps("lr_eff") * (m / denom), new_state


@final
class _CompiledAdamW(_CompiledAdam):
    r"""Compiled :class:`~lucid.optim.AdamW` — Adam with decoupled weight decay.

    The parameter is first scaled by :math:`1 - \eta\lambda` (as the
    eager kernel does), then takes the Adam step; the decay never enters
    the moments.

    See Also
    --------
    :class:`lucid.optim.AdamW` : eager counterpart.
    """

    _DECOUPLED = True
    _DEFAULT_WD = 1e-2


# ── RMSprop ─────────────────────────────────────────────────────────


@final
class _CompiledRMSprop(_CompiledStepBase):
    r"""Compiled :class:`~lucid.optim.RMSprop` (momentum / centred / weight decay).

    .. math::

        s_t &= \alpha s_{t-1} + (1-\alpha) g_t^2, \quad
        \bar g_t = \operatorname{lerp}(\bar g_{t-1}, g_t, 1-\alpha)
        \ \text{(centred)} \\
        \theta_t &= \theta_{t-1} - \eta \, (b_t \text{ or }
                    g_t / (\sqrt{s_t - \bar g_t^2} + \varepsilon))

    The lerp picks its form by the weight, as the eager engine does, so
    that choice is part of the structural flags.

    See Also
    --------
    :class:`lucid.optim.RMSprop` : eager counterpart.
    """

    _OPTIONAL_STATE = ("momentum_buffer", "grad_avg")

    @override
    def _flags(self, group: dict[str, object]) -> _Flags:
        return (
            _wd_on(group),
            _hp(group, "momentum", 0.0) != 0.0,
            bool(group.get("centered", False)),
            abs(1.0 - _hp(group, "alpha", 0.99)) < 0.5,
        )

    @override
    def _state_names(self, flags: _Flags) -> tuple[str, ...]:
        names = ["square_avg"]
        if flags[1]:
            names.append("momentum_buffer")
        if flags[2]:
            names.append("grad_avg")
        return tuple(names)

    @override
    def _group_scalar_names(self, flags: _Flags) -> tuple[str, ...]:
        names = ["lr", "alpha", "one_minus_alpha", "eps"]
        if flags[0]:
            names.append("weight_decay")
        if flags[1]:
            names.append("momentum")
        if flags[2] and not flags[3]:
            names.append("lerp_complement")
        return tuple(names)

    @override
    def _group_values(self, group: dict[str, object], flags: _Flags) -> dict[str, float]:
        alpha = _hp(group, "alpha", 0.99)
        w = 1.0 - alpha
        return {
            "lr": _hp(group, "lr", 1e-2),
            "alpha": alpha,
            "one_minus_alpha": w,
            "eps": _hp(group, "eps", 1e-8),
            "weight_decay": _hp(group, "weight_decay", 0.0),
            "momentum": _hp(group, "momentum", 0.0),
            "lerp_complement": 1.0 - w,
        }

    @override
    def _update(
        self, p: Tensor, g: Tensor, state: _State, flags: _Flags, gs: _ScalarFn, ps: _ScalarFn
    ) -> _Update:
        wd_on, mom_on, centered, small_w = flags
        if wd_on:
            g = g + gs("weight_decay") * p
        sq = gs("alpha") * state["square_avg"] + gs("one_minus_alpha") * (g * g)
        new_state = {"square_avg": sq}
        avg = sq
        if centered:
            ga = state["grad_avg"]
            diff = g - ga
            if small_w:
                new_ga = ga + gs("one_minus_alpha") * diff
            else:
                new_ga = g - diff * gs("lerp_complement")
            new_state["grad_avg"] = new_ga
            avg = sq - new_ga * new_ga
        denom = avg.sqrt() + gs("eps")
        if mom_on:
            buf = gs("momentum") * state["momentum_buffer"] + g / denom
            new_state["momentum_buffer"] = buf
            return p - gs("lr") * buf, new_state
        return p - (gs("lr") * g) / denom, new_state


# ── Adagrad ─────────────────────────────────────────────────────────


@final
class _CompiledAdagrad(_CompiledStepBase):
    r"""Compiled :class:`~lucid.optim.Adagrad`.

    .. math::

        s_t = s_{t-1} + g_t^2, \quad
        \theta_t = \theta_{t-1} - \eta_t g_t / (\sqrt{s_t} + \varepsilon),
        \quad \eta_t = \eta / (1 + (t-1)\gamma)

    :math:`\eta_t` is a per-parameter scalar (it follows the parameter's
    own step count).

    See Also
    --------
    :class:`lucid.optim.Adagrad` : eager counterpart.
    """

    @override
    def _flags(self, group: dict[str, object]) -> _Flags:
        return (_wd_on(group),)

    @override
    def _state_names(self, flags: _Flags) -> tuple[str, ...]:
        return ("sum",)

    @override
    def _group_scalar_names(self, flags: _Flags) -> tuple[str, ...]:
        return ("eps", "weight_decay") if flags[0] else ("eps",)

    @override
    def _param_scalar_names(self, flags: _Flags) -> tuple[str, ...]:
        return ("clr",)

    @override
    def _group_values(self, group: dict[str, object], flags: _Flags) -> dict[str, float]:
        return {
            "lr": _hp(group, "lr", 1e-2),
            "lr_decay": _hp(group, "lr_decay", 0.0),
            "eps": _hp(group, "eps", 1e-10),
            "weight_decay": _hp(group, "weight_decay", 0.0),
        }

    @override
    def _param_values(
        self, i: int, step: int, gv: dict[str, float], flags: _Flags
    ) -> dict[str, float]:
        return {"clr": gv["lr"] / (1.0 + float(step - 1) * gv["lr_decay"])}

    @override
    def _init_state(self, i: int, name: str, group: dict[str, object]) -> Tensor:
        init = _hp(group, "initial_accumulator_value", 0.0)
        return _full_like(self._params[i], init)

    @override
    def _update(
        self, p: Tensor, g: Tensor, state: _State, flags: _Flags, gs: _ScalarFn, ps: _ScalarFn
    ) -> _Update:
        if flags[0]:
            g = g + gs("weight_decay") * p
        s = state["sum"] + g * g
        denom = s.sqrt() + gs("eps")
        return p - (ps("clr") * g) / denom, {"sum": s}


# ── Adadelta ────────────────────────────────────────────────────────


@final
class _CompiledAdadelta(_CompiledStepBase):
    r"""Compiled :class:`~lucid.optim.Adadelta`.

    .. math::

        v_t &= \rho v_{t-1} + (1-\rho) g_t^2, \quad
        \Delta_t = \frac{\sqrt{u_{t-1} + \varepsilon}}{\sqrt{v_t + \varepsilon}} g_t \\
        u_t &= \rho u_{t-1} + (1-\rho) \Delta_t^2, \quad
        \theta_t = \theta_{t-1} - \eta \Delta_t

    See Also
    --------
    :class:`lucid.optim.Adadelta` : eager counterpart.
    """

    @override
    def _flags(self, group: dict[str, object]) -> _Flags:
        return (_wd_on(group),)

    @override
    def _state_names(self, flags: _Flags) -> tuple[str, ...]:
        return ("square_avg", "acc_delta")

    @override
    def _group_scalar_names(self, flags: _Flags) -> tuple[str, ...]:
        names = ("lr", "rho", "one_minus_rho", "eps")
        return names + ("weight_decay",) if flags[0] else names

    @override
    def _group_values(self, group: dict[str, object], flags: _Flags) -> dict[str, float]:
        rho = _hp(group, "rho", 0.9)
        return {
            "lr": _hp(group, "lr", 1.0),
            "rho": rho,
            "one_minus_rho": 1.0 - rho,
            "eps": _hp(group, "eps", 1e-6),
            "weight_decay": _hp(group, "weight_decay", 0.0),
        }

    @override
    def _update(
        self, p: Tensor, g: Tensor, state: _State, flags: _Flags, gs: _ScalarFn, ps: _ScalarFn
    ) -> _Update:
        if flags[0]:
            g = g + gs("weight_decay") * p
        eps = gs("eps")
        acc = state["acc_delta"]
        sq = gs("rho") * state["square_avg"] + gs("one_minus_rho") * (g * g)
        delta = ((acc + eps).sqrt() / (sq + eps).sqrt()) * g
        new_acc = gs("rho") * acc + gs("one_minus_rho") * (delta * delta)
        return p - gs("lr") * delta, {"square_avg": sq, "acc_delta": new_acc}


# ── Adamax ──────────────────────────────────────────────────────────


@final
class _CompiledAdamax(_CompiledStepBase):
    r"""Compiled :class:`~lucid.optim.Adamax` — Adam with an L∞ second moment.

    .. math::

        m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad
        u_t = \max(\beta_2 u_{t-1}, |g_t| + \varepsilon) \\
        \theta_t &= \theta_{t-1} - \frac{\eta}{1-\beta_1^t}
                    \cdot \frac{m_t}{u_t}

    See Also
    --------
    :class:`lucid.optim.Adamax` : eager counterpart.
    """

    @override
    def _flags(self, group: dict[str, object]) -> _Flags:
        return (_wd_on(group),)

    @override
    def _state_names(self, flags: _Flags) -> tuple[str, ...]:
        return ("exp_avg", "exp_inf")

    @override
    def _group_scalar_names(self, flags: _Flags) -> tuple[str, ...]:
        names = ("beta1", "one_minus_beta1", "beta2", "eps")
        return names + ("weight_decay",) if flags[0] else names

    @override
    def _param_scalar_names(self, flags: _Flags) -> tuple[str, ...]:
        return ("step_size",)

    @override
    def _group_values(self, group: dict[str, object], flags: _Flags) -> dict[str, float]:
        b1 = _hp(group, "beta1", 0.9)
        return {
            "lr": _hp(group, "lr", 2e-3),
            "beta1": b1,
            "one_minus_beta1": 1.0 - b1,
            "beta2": _hp(group, "beta2", 0.999),
            "eps": _hp(group, "eps", 1e-8),
            "weight_decay": _hp(group, "weight_decay", 0.0),
        }

    @override
    def _param_values(
        self, i: int, step: int, gv: dict[str, float], flags: _Flags
    ) -> dict[str, float]:
        return {"step_size": gv["lr"] / (1.0 - gv["beta1"] ** float(step))}

    @override
    def _update(
        self, p: Tensor, g: Tensor, state: _State, flags: _Flags, gs: _ScalarFn, ps: _ScalarFn
    ) -> _Update:
        import lucid as _lucid

        if flags[0]:
            g = g + gs("weight_decay") * p
        m = gs("beta1") * state["exp_avg"] + gs("one_minus_beta1") * g
        u = _lucid.maximum(gs("beta2") * state["exp_inf"], g.abs() + gs("eps"))
        new_p = p - ps("step_size") * (m / u)
        return new_p, {"exp_avg": m, "exp_inf": u}


# ── NAdam ───────────────────────────────────────────────────────────


@final
class _CompiledNAdam(_CompiledStepBase):
    r"""Compiled :class:`~lucid.optim.NAdam` — Adam with Nesterov lookahead.

    With :math:`\mu_t = \beta_1(1 - \tfrac12 \cdot 0.96^{t\psi})`
    (:math:`\psi = 0.004`, the engine's fixed momentum decay) and the
    per-parameter running product :math:`\Pi_t = \prod_{k\le t}\mu_k`:

    .. math::

        \theta_t = \theta_{t-1}
            - \frac{\eta(1-\mu_t)}{1-\Pi_t} \frac{g_t}{d_t}
            - \frac{\eta\mu_{t+1}}{1-\Pi_t\mu_{t+1}} \frac{m_t}{d_t},
        \quad d_t = \sqrt{v_t / (1-\beta_2^t)} + \varepsilon

    :math:`\Pi_t` is per-parameter scalar state rounded to float32 every
    step, as the engine keeps it (``mu_product`` in the state dict).

    See Also
    --------
    :class:`lucid.optim.NAdam` : eager counterpart.
    """

    _MOMENTUM_DECAY: float = 0.004
    _PSTATE_NAMES = ("mu_product",)

    @override
    def _flags(self, group: dict[str, object]) -> _Flags:
        return (_wd_on(group), bool(group.get("decoupled_weight_decay", False)))

    @override
    def _state_names(self, flags: _Flags) -> tuple[str, ...]:
        return ("exp_avg", "exp_avg_sq")

    @override
    def _group_scalar_names(self, flags: _Flags) -> tuple[str, ...]:
        names = ["beta1", "one_minus_beta1", "beta2", "one_minus_beta2", "eps"]
        if flags[0]:
            names.append("wd_factor" if flags[1] else "weight_decay")
        return tuple(names)

    @override
    def _param_scalar_names(self, flags: _Flags) -> tuple[str, ...]:
        return ("inv_bc2", "c1", "c2")

    @override
    def _group_values(self, group: dict[str, object], flags: _Flags) -> dict[str, float]:
        lr = _hp(group, "lr", 2e-3)
        b1 = _hp(group, "beta1", 0.9)
        b2 = _hp(group, "beta2", 0.999)
        wd = _hp(group, "weight_decay", 0.0)
        return {
            "lr": lr,
            "beta1": b1,
            "one_minus_beta1": 1.0 - b1,
            "beta2": b2,
            "one_minus_beta2": 1.0 - b2,
            "eps": _hp(group, "eps", 1e-8),
            "weight_decay": wd,
            "wd_factor": 1.0 - lr * wd,
            "momentum_decay": _hp(group, "momentum_decay", self._MOMENTUM_DECAY),
        }

    def _mu(self, beta1: float, step: int, momentum_decay: float = _MOMENTUM_DECAY) -> float:
        return beta1 * (1.0 - 0.5 * float(0.96 ** (float(step) * momentum_decay)))

    def _next_mu_product(
        self, i: int, step: int, beta1: float, momentum_decay: float = _MOMENTUM_DECAY
    ) -> float:
        prod = self._pstate[i].get("mu_product", 1.0)
        return _round_state_scalar(
            prod * self._mu(beta1, step, momentum_decay), self._params[i]._impl.dtype
        )

    @override
    def _param_values(
        self, i: int, step: int, gv: dict[str, float], flags: _Flags
    ) -> dict[str, float]:
        b1 = gv["beta1"]
        md = gv["momentum_decay"]
        mu = self._mu(b1, step, md)
        mu_next = self._mu(b1, step + 1, md)
        mu_prod = self._next_mu_product(i, step, b1, md)
        mu_prod_next = mu_prod * mu_next
        bc2 = 1.0 - gv["beta2"] ** float(step)
        lr = gv["lr"]
        return {
            "inv_bc2": 1.0 / bc2,
            "c1": lr * (1.0 - mu) / (1.0 - mu_prod),
            "c2": lr * mu_next / (1.0 - mu_prod_next),
        }

    @override
    def _init_pstate(self, i: int, group: dict[str, object]) -> None:
        self._pstate[i] = {"mu_product": 1.0}

    @override
    def _advance(self, i: int, step: int, gv: dict[str, float]) -> None:
        self._pstate[i]["mu_product"] = self._next_mu_product(
            i, step, gv["beta1"], gv["momentum_decay"]
        )

    @override
    def _update(
        self, p: Tensor, g: Tensor, state: _State, flags: _Flags, gs: _ScalarFn, ps: _ScalarFn
    ) -> _Update:
        wd_on, decoupled = flags
        base = p
        if wd_on:
            if decoupled:
                base = p * gs("wd_factor")
            else:
                g = g + gs("weight_decay") * p
        m = gs("beta1") * state["exp_avg"] + gs("one_minus_beta1") * g
        v = gs("beta2") * state["exp_avg_sq"] + gs("one_minus_beta2") * (g * g)
        denom = (ps("inv_bc2") * v).sqrt() + gs("eps")
        new_p = (base - ps("c1") * (g / denom)) - ps("c2") * (m / denom)
        return new_p, {"exp_avg": m, "exp_avg_sq": v}


# ── RAdam ───────────────────────────────────────────────────────────


@final
class _CompiledRAdam(_CompiledStepBase):
    r"""Compiled :class:`~lucid.optim.RAdam` — Rectified Adam.

    The rectification term :math:`r_t` and the "variance tractable"
    test :math:`\rho_t > 5` depend only on the parameter's step, so both
    are per-parameter scalars and the branch is a ``where`` on a 0-D
    flag:

    .. math::

        \hat m_t &= m_t / (1-\beta_1^t) \\
        \theta_t &= \theta_{t-1} - \begin{cases}
            \eta r_t \hat m_t \sqrt{1-\beta_2^t} / (\sqrt{v_t} + \varepsilon)
                & \rho_t > 5 \\
            \eta \hat m_t & \text{otherwise}
        \end{cases}

    See Also
    --------
    :class:`lucid.optim.RAdam` : eager counterpart.
    """

    @override
    def _flags(self, group: dict[str, object]) -> _Flags:
        return (_wd_on(group),)

    @override
    def _state_names(self, flags: _Flags) -> tuple[str, ...]:
        return ("exp_avg", "exp_avg_sq")

    @override
    def _group_scalar_names(self, flags: _Flags) -> tuple[str, ...]:
        names = ("lr", "beta1", "one_minus_beta1", "beta2", "one_minus_beta2", "eps")
        return names + ("weight_decay",) if flags[0] else names

    @override
    def _param_scalar_names(self, flags: _Flags) -> tuple[str, ...]:
        return ("inv_bc1", "bc2_sqrt", "lr_rt", "use_rect")

    @override
    def _group_values(self, group: dict[str, object], flags: _Flags) -> dict[str, float]:
        b1 = _hp(group, "beta1", 0.9)
        b2 = _hp(group, "beta2", 0.999)
        return {
            "lr": _hp(group, "lr", 1e-3),
            "beta1": b1,
            "one_minus_beta1": 1.0 - b1,
            "beta2": b2,
            "one_minus_beta2": 1.0 - b2,
            "eps": _hp(group, "eps", 1e-8),
            "weight_decay": _hp(group, "weight_decay", 0.0),
        }

    @override
    def _param_values(
        self, i: int, step: int, gv: dict[str, float], flags: _Flags
    ) -> dict[str, float]:
        b1, b2 = gv["beta1"], gv["beta2"]
        bc1 = 1.0 - b1 ** float(step)
        bc2 = 1.0 - b2 ** float(step)
        rho_inf = 2.0 / (1.0 - b2) - 1.0
        rho_t = rho_inf - 2.0 * step * b2 ** float(step) / bc2
        use_rect = rho_t > 5.0
        r_t = 0.0
        if use_rect:
            r_t = math.sqrt(
                (rho_t - 4.0)
                * (rho_t - 2.0)
                * rho_inf
                / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t)
            )
        return {
            "inv_bc1": 1.0 / bc1,
            "bc2_sqrt": math.sqrt(bc2),
            "lr_rt": gv["lr"] * r_t,
            "use_rect": 1.0 if use_rect else 0.0,
        }

    @override
    def _update(
        self, p: Tensor, g: Tensor, state: _State, flags: _Flags, gs: _ScalarFn, ps: _ScalarFn
    ) -> _Update:
        import lucid as _lucid

        if flags[0]:
            g = g + gs("weight_decay") * p
        m = gs("beta1") * state["exp_avg"] + gs("one_minus_beta1") * g
        v = gs("beta2") * state["exp_avg_sq"] + gs("one_minus_beta2") * (g * g)
        m_hat = ps("inv_bc1") * m
        adaptive = ps("bc2_sqrt") / (v.sqrt() + gs("eps"))
        p_rect = p - ps("lr_rt") * (m_hat * adaptive)
        p_sgd = p - gs("lr") * m_hat
        new_p = _lucid.where(ps("use_rect") > 0.5, p_rect, p_sgd)
        return new_p, {"exp_avg": m, "exp_avg_sq": v}


# ── ASGD ────────────────────────────────────────────────────────────


@final
class _CompiledASGD(_CompiledStepBase):
    r"""Compiled :class:`~lucid.optim.ASGD` — averaged SGD.

    Each parameter carries ``eta`` and ``mu`` (rounded to the state
    precision, as the engine keeps them); a step uses the values left by
    the previous one and then advances them:

    .. math::

        \eta_{t+1} = \frac{\eta_0}{(1 + \lambda \eta_0 t)^\alpha}, \quad
        \mu_{t+1} = \frac{1}{\max(1, t - t_0)}

    Update: :math:`\theta \leftarrow \theta(1-\lambda\eta) - \eta g`; the
    average copies :math:`\theta` while :math:`\mu = 1` and then moves
    toward it by :math:`\mu`.

    See Also
    --------
    :class:`lucid.optim.ASGD` : eager counterpart.
    """

    _PSTATE_NAMES = ("eta", "mu")

    @override
    def _flags(self, group: dict[str, object]) -> _Flags:
        return (_wd_on(group),)

    @override
    def _state_names(self, flags: _Flags) -> tuple[str, ...]:
        return ("ax",)

    @override
    def _group_scalar_names(self, flags: _Flags) -> tuple[str, ...]:
        return ("weight_decay",) if flags[0] else ()

    @override
    def _param_scalar_names(self, flags: _Flags) -> tuple[str, ...]:
        return ("decay", "neg_eta", "mu", "keep")

    @override
    def _group_values(self, group: dict[str, object], flags: _Flags) -> dict[str, float]:
        return {
            "lr": _hp(group, "lr", 1e-2),
            "lambd": _hp(group, "lambd", 1e-4),
            "alpha": _hp(group, "alpha", 0.75),
            "t0": _hp(group, "t0", 1e6),
            "weight_decay": _hp(group, "weight_decay", 0.0),
        }

    @override
    def _param_values(
        self, i: int, step: int, gv: dict[str, float], flags: _Flags
    ) -> dict[str, float]:
        st = self._pstate[i]
        eta = st.get("eta", _round_state_scalar(gv["lr"], self._params[i]._impl.dtype))
        mu = st.get("mu", 1.0)
        return {
            "decay": 1.0 - gv["lambd"] * eta,
            "neg_eta": -eta,
            "mu": mu,
            "keep": 0.0 if mu == 1.0 else 1.0,
        }

    @override
    def _init_pstate(self, i: int, group: dict[str, object]) -> None:
        lr = _hp(group, "lr", 1e-2)
        self._pstate[i] = {"eta": _round_state_scalar(lr, self._params[i]._impl.dtype), "mu": 1.0}

    @override
    def _advance(self, i: int, step: int, gv: dict[str, float]) -> None:
        dt = self._params[i]._impl.dtype
        lr = gv["lr"]
        t = float(step)
        self._pstate[i]["eta"] = _round_state_scalar(
            lr / ((1.0 + gv["lambd"] * lr * t) ** gv["alpha"]), dt
        )
        self._pstate[i]["mu"] = _round_state_scalar(1.0 / max(1.0, t - gv["t0"]), dt)

    @override
    def _update(
        self, p: Tensor, g: Tensor, state: _State, flags: _Flags, gs: _ScalarFn, ps: _ScalarFn
    ) -> _Update:
        if flags[0]:
            g = g + gs("weight_decay") * p
        new_p = p * ps("decay") + ps("neg_eta") * g
        base = state["ax"] * ps("keep")
        new_ax = base + (new_p - base) * ps("mu")
        return new_p, {"ax": new_ax}


# ── Rprop ───────────────────────────────────────────────────────────


@final
class _CompiledRprop(_CompiledStepBase):
    r"""Compiled :class:`~lucid.optim.Rprop` — sign-based step adaptation.

    The eager rule, element by element: the step size grows by
    :math:`\eta^+` where the gradient kept its sign and shrinks by
    :math:`\eta^-` where it flipped (then clamps to
    ``[step_min, step_max]``); a flipped element takes no step and
    stores a zero previous gradient, so the next step sees no sign
    agreement.  ``step_size`` starts at ``lr``.

    See Also
    --------
    :class:`lucid.optim.Rprop` : eager counterpart.
    """

    @override
    def _state_names(self, flags: _Flags) -> tuple[str, ...]:
        return ("prev", "step_size")

    @override
    def _group_scalar_names(self, flags: _Flags) -> tuple[str, ...]:
        return ("eta_plus", "eta_minus", "step_min", "step_max")

    @override
    def _group_values(self, group: dict[str, object], flags: _Flags) -> dict[str, float]:
        return {
            "eta_plus": _hp(group, "eta_plus", 1.2),
            "eta_minus": _hp(group, "eta_minus", 0.5),
            "step_min": _hp(group, "step_min", 1e-6),
            "step_max": _hp(group, "step_max", 50.0),
        }

    @override
    def _init_state(self, i: int, name: str, group: dict[str, object]) -> Tensor:
        if name == "step_size":
            return _full_like(self._params[i], _hp(group, "lr", 1e-2))
        return _zeros_like(self._params[i])

    @override
    def _update(
        self, p: Tensor, g: Tensor, state: _State, flags: _Flags, gs: _ScalarFn, ps: _ScalarFn
    ) -> _Update:
        import lucid as _lucid

        ss = state["step_size"]
        agree = g * state["prev"]
        grow = agree > 0.0
        flip = agree < 0.0
        new_ss = _lucid.where(grow, gs("eta_plus") * ss, ss)
        new_ss = _lucid.where(flip, gs("eta_minus") * new_ss, new_ss)
        new_ss = _lucid.minimum(_lucid.maximum(new_ss, gs("step_min")), gs("step_max"))
        eff_g = _lucid.where(flip, 0.0, g)
        new_p = p - _lucid.sign(eff_g) * new_ss
        return new_p, {"prev": eff_g, "step_size": new_ss}


# ── LBFGS ───────────────────────────────────────────────────────────


@final
class _CompiledLBFGS(_CompiledStepBase):
    r"""Compiled :class:`~lucid.optim.LBFGS` — single-iteration, no-line-search variant.

    L-BFGS in general needs a closure-driven line search whose iteration
    count depends on the loss value — incompatible with a fixed MPSGraph
    executable.  This path runs a restricted subset: one step per call,
    no closure, ``line_search_fn`` / ``max_iter`` / tolerances ignored,
    and a **per-element** Barzilai-Borwein direction from the last
    curvature pair instead of the flat-vector two-loop recursion:

    ::

        s     = param - prev_param
        y     = grad  - prev_grad
        alpha = clamp(|s| / (|y| + 1e-10), 0, 10)
        d     = -alpha * g        (steepest descent -g on a parameter's first two steps)
        param = param + lr * d

    It is not the eager algorithm, so its checkpoint (``step``,
    ``prev_param``, ``prev_grad``) is its own, read through the wrapper;
    the eager optimizer keeps its own state, ``step`` and checkpoint
    methods (it neither hands its state over nor answers for this one).

    See Also
    --------
    :class:`lucid.optim.LBFGS` : eager counterpart (full closure-driven
        line search).
    """

    _EPS_HISTORY: float = 1e-10
    # A different algorithm from the eager LBFGS: ``opt`` keeps its state.
    _OWNS_EAGER_STATE = False

    @override
    def _state_names(self, flags: _Flags) -> tuple[str, ...]:
        return ("prev_param", "prev_grad")

    @override
    def _group_scalar_names(self, flags: _Flags) -> tuple[str, ...]:
        return ("lr",)

    @override
    def _param_scalar_names(self, flags: _Flags) -> tuple[str, ...]:
        return ("use_history",)

    @override
    def _group_values(self, group: dict[str, object], flags: _Flags) -> dict[str, float]:
        return {"lr": _hp(group, "lr", 1.0)}

    @override
    def _param_values(
        self, i: int, step: int, gv: dict[str, float], flags: _Flags
    ) -> dict[str, float]:
        return {"use_history": 1.0 if step >= 2 else 0.0}

    @override
    def _update(
        self, p: Tensor, g: Tensor, state: _State, flags: _Flags, gs: _ScalarFn, ps: _ScalarFn
    ) -> _Update:
        import lucid as _lucid

        s = p - state["prev_param"]
        y = g - state["prev_grad"]
        alpha = (_lucid.abs(s) / (_lucid.abs(y) + self._EPS_HISTORY)).clip(0.0, 10.0)
        d = _lucid.where(ps("use_history") > 0.5, -alpha * g, -g)
        new_p = p + gs("lr") * d
        return new_p, {"prev_param": p, "prev_grad": g}
