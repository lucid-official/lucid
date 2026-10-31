"""
The learning-rate scheduler base classes and the rules the schedulers share.
"""

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import ClassVar
import warnings

from lucid.optim._step_counter import _STEP_MARK, _STEPPED, _watch_steps
from lucid.optim.optimizer import Optimizer, _group_float

_STEPPED_FIRST = (
    "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
    "You should call them in the opposite order: `optimizer.step()` before "
    "`lr_scheduler.step()`.  Failure to do this will result in skipping the "
    "first value of the learning rate schedule."
)

_STEP_REPLACED = (
    "Seems like `optimizer.step()` has been overridden after learning rate "
    "scheduler initialization. Please, make sure to call `optimizer.step()` "
    "before `lr_scheduler.step()`."
)


EPOCH_DEPRECATION_WARNING = (
    "The epoch parameter in `scheduler.step()` was not necessary and is being "
    "deprecated where possible. Please use `scheduler.step()` to step the "
    "scheduler. During the deprecation, if epoch is different from None, the "
    "closed form is used instead of the new chainable form, where available. "
    "Please open an issue if you are unable to replicate your use case: "
    "https://github.com/lucid-official/lucid/issues/new/choose."
)


def _check_step_order(optimizer: Optimizer) -> None:
    """Warn when a scheduler's first step comes before an optimizer step.

    Stepping the scheduler first skips the schedule's first value.  The
    warning points at the caller of the scheduler's ``step()``.
    """
    if not hasattr(optimizer.step, _STEP_MARK):
        warnings.warn(_STEP_REPLACED, UserWarning, stacklevel=3)
    elif not getattr(optimizer, _STEPPED, False):
        warnings.warn(_STEPPED_FIRST, UserWarning, stacklevel=3)


def _require_optimizer(optimizer: object) -> None:
    """Refuse what is not an optimizer, before it is read for param groups.

    An optimizer is whatever holds ``param_groups``: a compiled optimizer
    stands in for the :class:`Optimizer` it was built from and is scheduled
    like it, without being an instance of it.
    """
    if not hasattr(optimizer, "param_groups"):
        raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")


def _require_initial_lr(optimizer: Optimizer) -> None:
    """Refuse to resume (``last_epoch >= 0``) over groups that never had a schedule."""
    for i, group in enumerate(optimizer.param_groups):
        if "initial_lr" not in group:
            raise KeyError(
                f"param 'initial_lr' is not specified in param_groups[{i}] when resuming "
                "scheduler with last_epoch >= 0.\n"
                "This typically happens when:\n"
                "1. You're trying to resume training from a checkpoint but haven't properly "
                "loaded the optimizer state\n"
                "2. You're using last_epoch >= 0 for a fresh training run (not recommended)"
            )


def _per_group_lambdas(
    optimizer: Optimizer, lr_lambda: Callable[[int], float] | list[Callable[[int], float]]
) -> list[Callable[[int], float]]:
    """One function per param group: a list or tuple as given, a single one repeated."""
    _require_optimizer(optimizer)
    groups = len(optimizer.param_groups)
    if isinstance(lr_lambda, (list, tuple)):
        if len(lr_lambda) != groups:
            raise ValueError(f"Expected {groups} lr_lambdas, but got {len(lr_lambda)}")
        fns = list(lr_lambda)
    else:
        fns = [lr_lambda] * groups
    for fn in fns:
        if not callable(fn):
            raise TypeError(f"lr_lambda should be a function, but got {type(fn).__name__}")
    return fns


def _holds_a_callable(value: object) -> bool:
    """Whether ``value`` carries a Python function into a checkpoint.

    ``not callable(v)`` was the whole test, and ``LambdaLR`` stores its
    lambdas as ``lr_lambdas`` — a *list* — which is not itself callable.
    So the one scheduler the exclusion was written for was the one it
    missed, and ``lucid.save`` on a checkpoint holding a ``LambdaLR``
    raised out of ``pickle``.  A container is disqualified by any
    callable in it, not by all of them: a list with one lambda in it is
    exactly as unpicklable as a list of nothing else.
    """
    if callable(value):
        return True
    if isinstance(value, (list, tuple)):
        return any(callable(item) for item in value)
    return False


class _CheckpointState:
    """A scheduler's checkpoint: its attributes, less the optimizer and callables."""

    def state_dict(self) -> dict[str, object]:
        """Serialisable scheduler state, for checkpoint / resume.

        Mirrors the optimizer's contract: everything needed to continue the
        schedule exactly, minus the ``optimizer`` itself (the caller already
        holds it) and minus any callable hyper-parameter.

        Callables — ``LambdaLR``'s and ``MultiplicativeLR``'s ``lr_lambda`` —
        are **not** captured: a Python function is not reliably serialisable,
        and silently pickling one would make a checkpoint that only reloads in
        the process that wrote it.  Reconstruct the scheduler with the same
        lambda, then :meth:`load_state_dict` the rest.  This holds whether the
        callables are stored singly (``CyclicLR.scale_fn``) or in a
        list (``LambdaLR.lr_lambdas``, one per param group).

        Returns
        -------
        dict
            Attribute snapshot; feed it back to :meth:`load_state_dict`.
        """
        return {
            k: v for k, v in self.__dict__.items() if k != "optimizer" and not _holds_a_callable(v)
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Restore state produced by :meth:`state_dict`.

        Parameters
        ----------
        state_dict : dict
            A snapshot from :meth:`state_dict` of a scheduler of the same type.

        Notes
        -----
        Without this the schedule silently restarted on resume while the
        optimizer state restored correctly — the two went out of sync.
        """
        self.__dict__.update(state_dict)


def _floored(rates: list[float]) -> list[float]:
    """Each rate, or ``0.0`` where a schedule's formula took it below zero.

    The optimizer refuses a negative rate, and a schedule's own formula can
    leave ``[0, inf)`` without anyone having asked for it: a linear
    ``OneCycleLR`` extrapolates past ``total_steps - 1`` on its last step, and
    a cosine from a base below ``eta_min`` lands a few ulps under zero at the
    bottom of a cycle.  The reference writes those and lets the optimizer take
    them; here the write would raise out of a step that only ran the schedule
    to its end, so the schedule's value is held at the nearest one the
    optimizer takes.  ``nan`` stays ``nan``.
    """
    return [0.0 if lr <= 0.0 else lr for lr in rates]


class LRScheduler(_CheckpointState):
    """Base class for all learning rate schedulers.

    Subclasses must implement :meth:`get_lr`, which returns a list of new
    learning rates — one per optimizer parameter group — given the current
    ``last_epoch`` counter.

    The scheduler tracks ``base_lrs``: the initial learning rates captured
    from the optimizer at construction time.  Every call to :meth:`step`
    increments ``last_epoch`` by one, recomputes the learning rates via
    :meth:`get_lr`, and writes them back into the optimizer's param groups.

    Most schedules are *chainable*: each step scales every group's current
    rate, so two of them on one optimizer compose (:class:`ChainedScheduler`)
    and a group added later with ``add_param_group`` follows the schedule
    from the rate it was added with.  The schedules that hold something per
    group from construction — a base rate, a lambda, a cycle's bounds:
    :class:`LambdaLR`, :class:`MultiplicativeLR`, :class:`CyclicLR`,
    :class:`OneCycleLR`, :class:`CosineAnnealingWarmRestarts` — have nothing
    for such a group and refuse the step with ``ValueError``.

    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer whose learning rate will be scheduled.
    last_epoch : int, optional
        The index of the last epoch.  Pass ``-1`` (default) to start from
        epoch 0 on the first :meth:`step` call; pass ``N`` to resume from
        an existing checkpoint where ``N`` steps have already been taken.

    Attributes
    ----------
    optimizer : Optimizer
        The optimizer being scheduled.
    base_lrs : list of float
        The initial learning rates of each param group captured at
        construction time.
    last_epoch : int
        Number of :meth:`step` calls completed so far.

    Notes
    -----
    Schedulers should be stepped **after** the optimizer:

    .. code-block:: python

        for batch in dataloader:
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()
            scheduler.step()

    Stepping the scheduler first skips the schedule's first value.  So the
    scheduler's first :meth:`step` issues a ``UserWarning`` when the
    optimizer has not stepped since the first scheduler on it was built,
    as the reference does.  To see the optimizer step, the constructor
    replaces ``optimizer.step`` with a counter that forwards every call.
    If ``optimizer.step`` is replaced again after that, the first
    :meth:`step` warns that it can no longer tell.

    Examples
    --------
    This class is not instantiated directly.  Use a concrete subclass such
    as :class:`StepLR` or :class:`CosineAnnealingLR`.
    """

    #: ``True`` only while the constructor writes step 0's rates.  A
    #: chainable schedule then leaves the rates as they are — a second
    #: scheduler on the optimizer must not apply its own step 0 on top of the
    #: first's, and a resumed one (``last_epoch >= 0``) not a step twice.
    _is_initial: bool = False

    #: Whether the first :meth:`step` checks that the optimizer stepped
    #: before it.  :class:`CosineAnnealingWarmRestarts` does not check,
    #: because the reference's does not.
    _checks_step_order: ClassVar[bool] = True

    #: Whether a negative rate is the schedule's own formula's artefact (held
    #: at zero) rather than the caller's; see :class:`_PerGroupLambda`.
    _floors_rates: ClassVar[bool] = True

    def __init__(self, optimizer: Optimizer, last_epoch: int = -1) -> None:
        _require_optimizer(optimizer)
        self.optimizer = optimizer
        _watch_steps(optimizer)
        # After N step() calls, last_epoch == N (the constructor's initial
        # step takes it from -1 to 0).
        self.last_epoch = last_epoch
        # The rate before any scheduler touched it, kept on the group: the
        # schedule below writes step 0's rate straight away, and a second
        # scheduler on the same optimizer — a SequentialLR's cosine after
        # its warmup — must still start from the true base, not from that.
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault("initial_lr", group["lr"])
        else:
            _require_initial_lr(optimizer)
        self.base_lrs: list[float] = [_group_float(g, "initial_lr") for g in optimizer.param_groups]
        self._last_lr: list[float] = []
        # Step 0's rate takes effect now, before the first optimizer step.
        # Only step() used to write a rate, so the first update ran at the
        # base rate whatever the schedule said: LambdaLR and LinearLR warmups
        # at full rate, ConstantLR unscaled, CyclicLR at max_lr, OneCycleLR at
        # max_lr rather than max_lr / div_factor — 25x its first step.
        self._initial_step()

    def _initial_step(self) -> None:
        """Write step 0's rates, in force before the first ``step()``."""
        # This step is counted, as the reference counts it, so the first
        # ``step()`` finds 1 and checks the order.  A checkpoint carries the
        # count, so a resumed scheduler does not check again.
        self._step_count = 1
        self._is_initial = True
        try:
            self._update_lr()
        finally:
            self._is_initial = False

    def _group_lrs(self) -> list[float]:
        """Each param group's current rate — what a chainable step scales."""
        return [_group_float(g) for g in self.optimizer.param_groups]

    def _unrated_group(self, rated: int) -> ValueError:
        """The error for a step that has rates for only the first ``rated`` groups.

        A schedule that computes from what it captured at construction —
        ``base_lrs``, one lambda or one bound per group — has nothing for a
        group added afterwards.  The rates used to be written by a
        non-strict ``zip``, which left that group at the rate it was added
        with: unscheduled, and silently so.  The reference refuses the step.
        """
        return ValueError(
            f"{type(self).__name__}: param group {rated} was added after the "
            f"scheduler was built, and this schedule rates only the {rated} "
            "group(s) it was built over; build the scheduler after "
            "add_param_group"
        )

    def _update_lr(self, epoch: int | None = None) -> None:
        """Move to the next epoch — or to ``epoch``, by the closed form — and write it.

        Every group gets a rate or none does: a refused step leaves the
        rates and ``last_epoch`` as they were.
        """
        previous = self.last_epoch
        self.last_epoch = previous + 1 if epoch is None else epoch
        try:
            values = self.get_lr() if epoch is None else self._get_closed_form_lr()
            if len(values) < len(self.optimizer.param_groups):
                raise self._unrated_group(len(values))
            if self._floors_rates:
                values = _floored(values)
        except ValueError:
            self.last_epoch = previous
            raise
        for group, lr in zip(self.optimizer.param_groups, values, strict=True):
            group["lr"] = lr
        self._last_lr = list(values)
        self.optimizer._sync_hyperparams()

    def _get_closed_form_lr(self) -> list[float]:
        """The rates at ``last_epoch`` from ``base_lrs`` alone, for a jump to it.

        :class:`SequentialLR` starts each child after the first at epoch 0
        this way.  A schedule with no closed form takes the step instead.
        """
        return self.get_lr()

    def _restart(self) -> None:
        """Start over at epoch 0, from the base rates — a SequentialLR milestone."""
        self._update_lr(0)

    def _undo_initial_step(self) -> None:
        """Take back the step 0 the constructor took, as SequentialLR does."""
        self.last_epoch -= 1

    def step(self, epoch: int | None = None) -> None:
        """Advance the scheduler by one epoch and update optimizer learning rates.

        Increments ``last_epoch`` by one, calls :meth:`get_lr` to obtain
        the new per-group learning rates, and writes them into the optimizer's
        param groups.  Also calls ``optimizer._sync_hyperparams()`` so that
        any engine-level optimizer mirrors the updated value.

        Should be called **once per epoch**, after ``optimizer.step()``.

        Parameters
        ----------
        epoch : int, optional
            Deprecated.  Sets ``last_epoch`` to ``epoch`` and computes the
            rates from the schedule's closed form where it has one, instead
            of stepping from the current rates.  Not every schedule has one.

        Warns
        -----
        UserWarning
            On the first call, when the optimizer has not stepped since the
            first scheduler on it was built, or when ``optimizer.step`` was
            replaced after that; and whenever ``epoch`` is given.

        Examples
        --------
        >>> import lucid
        >>> import lucid.nn as nn
        >>> import lucid.optim as optim
        >>> model = nn.Linear(4, 2)
        >>> optimizer = optim.SGD(model.parameters(), lr=0.1)
        >>> scheduler = lucid.optim.StepLR(optimizer, step_size=10)
        >>> for epoch in range(50):
        ...     optimizer.step()
        ...     scheduler.step()
        """
        # Checked before the update, so a refused first step still warns, as
        # the reference's does.  The count moves only with a step that is
        # taken: a refused step changes nothing.
        if self._step_count == 1 and self._checks_step_order:
            _check_step_order(self.optimizer)
        if epoch is not None:
            warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning, stacklevel=2)
        self._update_lr(epoch)
        self._step_count += 1

    def get_lr(self) -> list[float]:
        """Compute new learning rates for all param groups.

        Returns
        -------
        list of float
            One learning rate per optimizer param group, from each group's
            current rate (a chainable schedule) or from ``base_lrs`` and the
            current ``last_epoch`` (a closed-form one).

        Raises
        ------
        NotImplementedError
            Concrete subclasses must override this method.
        """
        raise NotImplementedError

    def get_last_lr(self) -> list[float]:
        """Return the last learning rate computed by the scheduler.

        Returns
        -------
        list of float
            The current learning rate of each optimizer param group, read
            directly from the optimizer (not a cached copy).

        Examples
        --------
        """
        return [_group_float(g) for g in self.optimizer.param_groups]


_LRScheduler = LRScheduler  # the name the reference keeps for the same base


class _PerGroupLambda(LRScheduler):
    """The schedules whose rate comes from the caller's function, one per param group.

    ``LambdaLR`` and ``MultiplicativeLR`` write what the function returns: a
    negative result is the caller's to see, from the optimizer, and not a
    formula's artefact for the write path to hold at zero.
    """

    _floors_rates: ClassVar[bool] = False

    def __init__(
        self,
        optimizer: Optimizer,
        lr_lambda: Callable[[int], float] | list[Callable[[int], float]],
        last_epoch: int = -1,
    ) -> None:
        self.lr_lambdas = _per_group_lambdas(optimizer, lr_lambda)
        super().__init__(optimizer, last_epoch)


class _CompositeSchedulerMixin:
    """What the composite schedulers share: checkpoints and taking back step 0."""

    optimizer: Optimizer
    schedulers: Sequence[LRScheduler | _CompositeSchedulerMixin]

    def state_dict(self) -> dict[str, object]:
        """Serialisable state, including each wrapped scheduler's own state."""
        own = {
            k: v
            for k, v in self.__dict__.items()
            if k not in ("optimizer", "schedulers") and not _holds_a_callable(v)
        }
        own["schedulers"] = [sch.state_dict() for sch in self.schedulers]
        return own

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Restore state produced by :meth:`state_dict`."""
        incoming = dict(state_dict)
        raw_children = incoming.pop("schedulers", None)
        self.__dict__.update(incoming)
        if raw_children is not None:
            if not isinstance(raw_children, list):
                raise TypeError("load_state_dict: 'schedulers' must be a list")
            child_states: list[dict[str, object]] = raw_children
            if len(child_states) != len(self.schedulers):
                raise ValueError(
                    f"{type(self).__name__}.load_state_dict: got {len(child_states)} child states "
                    f"for {len(self.schedulers)} schedulers"
                )
            for sch, child in zip(self.schedulers, child_states, strict=True):
                sch.load_state_dict(child)

    def _undo_initial_step(self) -> None:
        """Take back each wrapped scheduler's step 0, as the reference recurses."""
        for sched in self.schedulers:
            sched._undo_initial_step()


def _check_schedulers(
    owner: str,
    schedulers: Sequence[LRScheduler | _CompositeSchedulerMixin],
    optimizer: Optimizer | None,
    accepted: tuple[type, ...],
) -> Optimizer:
    """The optimizer a composite's children share, or why it cannot wrap them.

    Checked before the composite changes anything.  SequentialLR used to
    put every rate back to its base and take back its children's step 0
    first, then fail on a child it could not drive with an untyped
    ``AttributeError`` — leaving the optimizer and the children altered.
    A scheduler that steps on a metric is a scheduler, just not one a
    composite can step: ``ValueError``, as the reference raises.
    """
    if not schedulers:
        raise ValueError(f"{owner} expects at least one scheduler")
    for i, sched in enumerate(schedulers):
        if getattr(sched, "_steps_on_a_metric", False):
            raise ValueError(
                f"{owner} at index {i}: {type(sched).__name__} steps on a metric, "
                f"which {owner}.step() does not take"
            )
        if not isinstance(sched, accepted):
            kinds = {
                LRScheduler: "an LR scheduler",
                _CompositeSchedulerMixin: "SequentialLR or ChainedScheduler",
            }
            names = " or ".join(kinds.get(t, t.__name__) for t in accepted)
            raise TypeError(f"{owner} at index {i}: expected {names}, got {type(sched).__name__}")
    shared = optimizer or schedulers[0].optimizer
    for i, sched in enumerate(schedulers):
        if sched.optimizer is not shared:
            raise ValueError(
                f"{owner} expects all schedulers to belong to the same optimizer; "
                f"the one at index {i} does not"
            )
    return shared


def _require_group_keys(owner: str, optimizer: Optimizer, keys: tuple[str, ...]) -> None:
    """Refuse a step whose schedule reads ``keys`` off a group that lacks them.

    The schedule sets them on the groups it was built over, so a group added
    afterwards has none unless it brought its own.
    """
    for i, group in enumerate(optimizer.param_groups):
        missing = [k for k in keys if k not in group]
        if missing:
            raise ValueError(
                f"{owner}: param group {i} has no {', '.join(missing)}, which the "
                "schedule reads off each group and sets only on the groups it was "
                "built over; build the scheduler after add_param_group, or give "
                f"the group {', '.join(keys)}"
            )


def _reset_to_base(optimizer: Optimizer) -> None:
    """Put each group back at the rate it had before any scheduler touched it."""
    for group in optimizer.param_groups:
        group["lr"] = _group_float(group, "initial_lr" if "initial_lr" in group else "lr")


def _scheduler_tree(
    schedulers: Sequence[LRScheduler | _CompositeSchedulerMixin],
) -> Iterator[LRScheduler | _CompositeSchedulerMixin]:
    """Every scheduler under ``schedulers``, composites and what they wrap."""
    for sched in schedulers:
        yield sched
        if isinstance(sched, _CompositeSchedulerMixin):
            yield from _scheduler_tree(sched.schedulers)


@contextmanager
def _all_or_nothing(composite: _CompositeSchedulerMixin) -> Iterator[None]:
    """Undo everything a composite's step did when one of its children refuses.

    A chain steps its children one after another, so a refusal by the
    second left the first stepped and the rates half-written.  The rates
    and the state of every scheduler under the composite go back as they
    were.  Each step replaces the values it changes rather than mutating
    them, so shallow copies are enough.
    """
    groups = [dict(g) for g in composite.optimizer.param_groups]
    states = [(s, dict(vars(s))) for s in _scheduler_tree(composite.schedulers)]
    try:
        yield
    except ValueError:
        for group, saved in zip(composite.optimizer.param_groups, groups, strict=True):
            group.clear()
            group.update(saved)
        for sched, state in states:
            vars(sched).clear()
            vars(sched).update(state)
        composite.optimizer._sync_hyperparams()
        raise
