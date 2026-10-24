"""
Optimizer base class.
"""

import math
import operator
import warnings
from collections.abc import Callable, Iterable, Sequence
from typing import Any, ClassVar, Protocol, SupportsFloat, SupportsIndex, override

import lucid as _lucid
from lucid._tensor.tensor import Tensor
from lucid._types import _OptimizerClosure
from lucid.autograd._grad_inplace import zero_grads_
from lucid.nn.parameter import Parameter
from lucid.optim._step_counter import _CountedStep

from lucid._C import engine as _C_engine

_INTEGER_DTYPES = frozenset(
    {
        _C_engine.Dtype.I8,
        _C_engine.Dtype.I16,
        _C_engine.Dtype.I32,
        _C_engine.Dtype.I64,
        _C_engine.Dtype.Bool,
    }
)


class _EngineRules(Protocol):
    """An engine optimizer class, as the base class reads its rules."""

    @staticmethod
    def check_hyperparams(values: dict[str, float]) -> None: ...


class _EngineOptimizer(Protocol):
    """An engine optimizer bound to one parameter group."""

    step_count: int

    def step(self) -> None: ...

    def set_hyperparams(self, values: dict[str, Any]) -> None: ...

    def state_buffers(
        self,
    ) -> Sequence[tuple[str, Sequence[_C_engine.TensorImpl | None]]]: ...

    def load_state_buffers(self, bufs: Sequence[tuple[str, Sequence[Any]]]) -> None: ...


class _GroupReader(Protocol):
    """Reads a class's ``_HYPERPARAMS`` out of a param group, as one tuple.

    A callable object rather than a function, so a reader stored on the
    class is not bound to the instances that call it.
    """

    def __call__(self, group: dict[str, object], /) -> Any: ...


class _OneKey:
    """Reads one param-group key as a 1-tuple.

    ``operator.itemgetter`` with a single key returns the bare value; this
    keeps the tuple every hyper-parameter reader returns.
    """

    __slots__ = ("_key",)

    def __init__(self, key: str) -> None:
        self._key = key

    def __call__(self, group: dict[str, object]) -> tuple[object, ...]:
        return (group[self._key],)


def _hyperparam_reader(
    keys: tuple[str, ...],
) -> _GroupReader | None:
    """Return a reader of ``keys`` out of a param group, as one tuple.

    ``operator.itemgetter`` does the reading in C, so comparing a group with
    the values last handed over costs one call and one tuple comparison.
    ``None`` when there are no keys to read.  Neither reader is a function,
    so stored on a class it is not bound to the instances that read it.
    """
    if not keys:
        return None
    if len(keys) == 1:
        return _OneKey(keys[0])
    return operator.itemgetter(*keys)


def _follow(
    src: _C_engine.TensorImpl, like: _C_engine.TensorImpl
) -> _C_engine.TensorImpl:
    """``src`` in ``like``'s dtype and on its device; ``src`` itself when it already is.

    The one rule by which optimizer state follows its parameter through a
    conversion (``module.to()``, ``.double()``, ``.half()``).  Engine
    optimizers apply it when their engine is rebuilt (:func:`_state_like`);
    optimizers that keep their state in Python apply it before they read
    that state in ``step``.
    """
    out = src
    if out.dtype != like.dtype:
        out = _C_engine.astype(out, like.dtype)
    if out.device != like.device:
        out = out.transfer_to_device(like.device, False)
    return out


def _state_like(
    src: _C_engine.TensorImpl, dst: _C_engine.TensorImpl
) -> _C_engine.TensorImpl | None:
    """Return optimizer state entry ``src`` ready for parameter impl ``dst``.

    A buffer shaped like the parameter (``exp_avg``, ``momentum_buffer``
    ...) follows it (:func:`_follow`).  A per-parameter scalar — a
    0-d entry of a parameter that is not 0-d, such as ``step`` or NAdam's
    ``mu_product`` — and any integer entry (a count) pass unchanged: the
    engine reads them as numbers wherever they live, and ``step`` stays an
    integer.  ``None`` when a buffer's shape no longer matches: it belonged
    to a differently shaped parameter and cannot be reused.
    """
    if src.dtype in _INTEGER_DTYPES or (not src.shape and dst.shape):
        return src
    if list(src.shape) != list(dst.shape):
        return None
    return _follow(src, dst)


def _host_copy(src: _C_engine.TensorImpl) -> object:
    """``src`` as a host array of its own, as a checkpoint stores state.

    A copy, so the checkpoint shares no buffer with the live state and
    pickles without engine objects.
    """
    import numpy as _np

    return _np.asarray(src.data_as_python()).copy()


def _from_host(value: object, device: _C_engine.Device) -> _C_engine.TensorImpl:
    """A checkpointed state entry — a host array or a tensor — on ``device``."""
    if isinstance(value, Tensor):
        return value._impl.transfer_to_device(device, False)
    return _C_engine.TensorImpl(value, device, False)


def _group_params(group: dict[str, object]) -> list[Parameter]:
    """The parameters of one param group (``add_param_group`` stores a list)."""
    params = group["params"]
    if not isinstance(params, list):
        raise TypeError(
            f"a param group's 'params' must be a list, got {type(params).__name__}"
        )
    return params


def _group_float(group: dict[str, object], key: str = "lr") -> float:
    """Read a numeric entry ``key`` from ``group`` as a ``float``."""
    value = group[key]
    if not isinstance(value, (SupportsFloat, SupportsIndex)):
        raise TypeError(
            f"a param group's {key!r} must be a number, got {type(value).__name__}"
        )
    return float(value)


def _format_param(
    name: str, optimizer: Optimizer, param: float | list[float]
) -> list[float]:
    """Broadcast a scalar or validate a per-group list for ``optimizer``."""
    n = len(optimizer.param_groups)
    if isinstance(param, (list, tuple)):
        if len(param) != n:
            raise ValueError(f"expected {n} values for {name}, but got {len(param)}")
        return [float(x) for x in param]
    return [float(param)] * n


def _set_group_momentum(
    use_beta1: bool, group: dict[str, object], momentum: float
) -> None:
    """Write ``momentum`` into ``group['betas'][0]`` or ``group['momentum']``."""
    if use_beta1:
        betas = group["betas"]
        assert isinstance(betas, tuple)
        group["betas"] = (momentum, *betas[1:])
    else:
        group["momentum"] = momentum


def _init_cycle_momentum(
    optimizer: Optimizer,
    cycle_momentum: bool,
    base_momentum: float | list[float],
    max_momentum: float | list[float],
    last_epoch: int,
) -> tuple[bool, list[float], list[float]]:
    """Validate and initialise per-group momentum cycling for CyclicLR / OneCycleLR."""
    if not cycle_momentum:
        return False, [], []
    if "momentum" not in optimizer.defaults and "betas" not in optimizer.defaults:
        raise ValueError(
            "optimizer must support momentum or beta1 with `cycle_momentum` option enabled"
        )
    use_beta1 = "betas" in optimizer.defaults
    base_moms = _format_param("base_momentum", optimizer, base_momentum)
    max_moms = _format_param("max_momentum", optimizer, max_momentum)
    if last_epoch == -1:
        for m_mom, b_mom, group in zip(
            max_moms, base_moms, optimizer.param_groups, strict=True
        ):
            _set_group_momentum(use_beta1, group, m_mom)
            group["max_momentum"] = m_mom
            group["base_momentum"] = b_mom
    return use_beta1, base_moms, max_moms


def _check_positive_int(name: str, val: int | None) -> int:
    if isinstance(val, bool) or not isinstance(val, int) or val <= 0:
        raise ValueError(f"Expected positive integer {name}, but got {val}")
    return val


type _OneCycleSteps = tuple[int | None, int | None, int | None]


def _resolve_one_cycle_steps(step_cfg: _OneCycleSteps) -> int:
    """Validate and compute ``total_steps`` for :class:`OneCycleLR`."""
    total_steps, epochs, steps_per_epoch = step_cfg
    if total_steps is not None:
        return _check_positive_int("total_steps", total_steps)
    if epochs is not None and steps_per_epoch is not None:
        return _check_positive_int("epochs", epochs) * _check_positive_int(
            "steps_per_epoch", steps_per_epoch
        )
    raise ValueError(
        "You must define either total_steps OR (epochs AND steps_per_epoch)"
    )


def _one_cycle_phases(
    step_cfg: _OneCycleSteps,
    pct_start: float,
    anneal_strategy: str,
    three_phase: bool,
) -> tuple[int, list[tuple[float, str, str, str, str]]]:
    """Resolve ``total_steps`` and the ``(end_step, start_lr, end_lr, start_mom, end_mom)`` phases."""
    steps = _resolve_one_cycle_steps(step_cfg)
    if pct_start < 0 or pct_start > 1 or not isinstance(pct_start, float):
        raise ValueError(
            f"Expected float between 0 and 1 pct_start, but got {pct_start}"
        )
    if anneal_strategy not in ("cos", "linear"):
        raise ValueError(
            f"anneal_strategy must be one of 'cos' or 'linear', instead got {anneal_strategy}"
        )
    p1 = (
        float(pct_start * steps) - 1.0,
        "initial_lr",
        "max_lr",
        "max_momentum",
        "base_momentum",
    )
    if three_phase:
        phases = [
            p1,
            (
                float(2.0 * pct_start * steps) - 2.0,
                "max_lr",
                "initial_lr",
                "base_momentum",
                "max_momentum",
            ),
            (
                float(steps - 1),
                "initial_lr",
                "min_lr",
                "max_momentum",
                "max_momentum",
            ),
        ]
    else:
        phases = [
            p1,
            (
                float(steps - 1),
                "max_lr",
                "min_lr",
                "base_momentum",
                "max_momentum",
            ),
        ]
    return steps, phases


def _cyclic_scale(sched: _CyclicLike, cycle: int) -> float:
    """Evaluate the CyclicLR amplitude scale factor."""
    if sched._scale_fn_custom is not None:
        arg = cycle if sched.scale_mode == "cycle" else sched.last_epoch
        return float(sched._scale_fn_custom(arg))
    if sched.mode == "triangular2":
        return float(1.0 / (2.0 ** (cycle - 1)))
    if sched.mode == "exp_range":
        return float(sched.gamma**sched.last_epoch)
    return 1.0


class _CyclicLike:
    optimizer: Optimizer
    last_epoch: int
    base_lrs: list[float]
    max_lrs: list[float]
    step_size_up: int
    total_size: float
    step_ratio: float
    mode: str
    gamma: float
    _scale_fn_custom: Callable[[float], float] | None
    scale_mode: str
    cycle_momentum: bool
    use_beta1: bool
    base_momentums: list[float]
    max_momentums: list[float]


class _OneCycleLike:
    optimizer: Optimizer
    last_epoch: int
    base_lrs: list[float]
    total_steps: int
    _schedule_phases: list[tuple[float, str, str, str, str]]
    max_lr: float | list[float]
    pct_start: float
    anneal_strategy: str
    div_factor: float
    final_div_factor: float
    three_phase: bool
    cycle_momentum: bool
    use_beta1: bool

    def _annealing_cos(self, start: float, end: float, pct: float) -> float:
        return end + (start - end) / 2 * (math.cos(math.pi * pct) + 1)

    def _annealing_linear(self, start: float, end: float, pct: float) -> float:
        return start + pct * (end - start)


type _CyclicArgs = tuple[
    float | list[float],
    float | list[float],
    int,
    int | None,
    str,
    float,
    Callable[[float], float] | None,
    str,
    bool,
    float | list[float],
    float | list[float],
    int,
]

type _OneCycleArgs = tuple[
    float | list[float],
    int | None,
    int | None,
    int | None,
    float,
    str,
    bool,
    float | list[float],
    float | list[float],
    float,
    float,
    bool,
    int,
]


def _init_cyclic(
    sched: _CyclicLike, optimizer: Optimizer, args: _CyclicArgs
) -> list[float]:
    """Initialise CyclicLR attributes on ``sched`` and return ``base_lrs``."""
    (
        base_lr,
        max_lr,
        step_size_up,
        step_size_down,
        mode,
        gamma,
        scale_fn,
        scale_mode,
        cycle_momentum,
        base_momentum,
        max_momentum,
        last_epoch,
    ) = args
    if mode not in ("triangular", "triangular2", "exp_range") and scale_fn is None:
        raise ValueError("mode is invalid and scale_fn is None")
    sched.optimizer = optimizer
    base_lrs = _format_param("base_lr", optimizer, base_lr)
    if last_epoch == -1:
        for lr, group in zip(base_lrs, optimizer.param_groups, strict=True):
            group["lr"] = lr
            group["initial_lr"] = lr
    sched.max_lrs = _format_param("max_lr", optimizer, max_lr)
    sched.step_size_up = step_size_up
    up = float(step_size_up)
    down = float(step_size_down) if step_size_down is not None else up
    sched.total_size = up + down
    sched.step_ratio = up / sched.total_size
    sched.mode = mode
    sched.gamma = gamma
    sched._scale_fn_custom = scale_fn
    sched.scale_mode = (
        scale_mode
        if scale_fn is not None
        else ("iterations" if mode == "exp_range" else "cycle")
    )
    sched.cycle_momentum = cycle_momentum
    sched.use_beta1, sched.base_momentums, sched.max_momentums = _init_cycle_momentum(
        optimizer, cycle_momentum, base_momentum, max_momentum, last_epoch
    )
    return base_lrs


def _cyclic_get_lr(sched: _CyclicLike, cycle: int) -> list[float]:
    """Compute per-group learning rates (and update momentum) for CyclicLR."""
    x = 1.0 + sched.last_epoch / sched.total_size - cycle
    sf = (
        x / sched.step_ratio
        if x <= sched.step_ratio
        else (x - 1) / (sched.step_ratio - 1)
    )
    s = _cyclic_scale(sched, cycle)
    if sched.cycle_momentum:
        for group, base_m, max_m in zip(
            sched.optimizer.param_groups,
            sched.base_momentums,
            sched.max_momentums,
            strict=False,
        ):
            _set_group_momentum(
                sched.use_beta1, group, max_m - (max_m - base_m) * sf * s
            )
    return [
        b_lr + (m_lr - b_lr) * sf * s
        for b_lr, m_lr in zip(sched.base_lrs, sched.max_lrs, strict=True)
    ]


def _init_one_cycle(
    sched: _OneCycleLike, optimizer: Optimizer, args: _OneCycleArgs
) -> None:
    """Initialise OneCycleLR attributes on ``sched``."""
    (
        max_lr,
        total_steps,
        epochs,
        steps_per_epoch,
        pct_start,
        anneal_strategy,
        cycle_momentum,
        base_momentum,
        max_momentum,
        div_factor,
        final_div_factor,
        three_phase,
        last_epoch,
    ) = args
    sched.optimizer = optimizer
    sched.total_steps, sched._schedule_phases = _one_cycle_phases(
        (total_steps, epochs, steps_per_epoch), pct_start, anneal_strategy, three_phase
    )
    sched.max_lr = max_lr
    sched.pct_start = pct_start
    sched.anneal_strategy = anneal_strategy
    sched.div_factor = div_factor
    sched.final_div_factor = final_div_factor
    sched.three_phase = three_phase
    max_lrs = _format_param("max_lr", optimizer, max_lr)
    if last_epoch == -1:
        for idx, group in enumerate(optimizer.param_groups):
            init_lr = max_lrs[idx] / div_factor
            group["initial_lr"] = init_lr
            group["max_lr"] = max_lrs[idx]
            group["min_lr"] = init_lr / final_div_factor
    sched.cycle_momentum = cycle_momentum
    sched.use_beta1, _, _ = _init_cycle_momentum(
        optimizer, cycle_momentum, base_momentum, max_momentum, last_epoch
    )


def _one_cycle_get_lr(sched: _OneCycleLike) -> list[float]:
    """Evaluate OneCycleLR per-group learning rates (and update momentum) at ``sched.last_epoch``."""
    step_num = sched.last_epoch
    if step_num > sched.total_steps:
        raise ValueError(
            f"Tried to step {step_num} times. The specified number of total steps is {sched.total_steps}"
        )
    anneal = (
        sched._annealing_cos
        if sched.anneal_strategy == "cos"
        else sched._annealing_linear
    )
    lrs: list[float] = []
    for group in sched.optimizer.param_groups:
        start_step = 0.0
        for i, (end_step, s_lr, e_lr, s_mom, e_mom) in enumerate(
            sched._schedule_phases
        ):
            if step_num <= end_step or i == len(sched._schedule_phases) - 1:
                denom = end_step - start_step
                pct = (step_num - start_step) / denom if denom != 0.0 else 0.0
                lrs.append(
                    anneal(_group_float(group, s_lr), _group_float(group, e_lr), pct)
                )
                if sched.cycle_momentum:
                    mom = anneal(
                        _group_float(group, s_mom), _group_float(group, e_mom), pct
                    )
                    _set_group_momentum(sched.use_beta1, group, mom)
                break
            start_step = end_step
    return lrs


type _AdamCore = tuple[float, tuple[float, float], float, float]
type _AdamFlags = tuple[bool | None, bool, bool, bool]


class Optimizer:
    """Base class for all optimizers.

    Every concrete optimizer must subclass ``Optimizer`` and implement
    :meth:`step`.  The base class manages parameter groups, provides
    :meth:`zero_grad`, and handles :meth:`state_dict` / :meth:`load_state_dict`
    checkpointing.

    Parameters
    ----------
    params : iterable of Parameter or iterable of dict
        Either a flat iterable of :class:`~lucid.nn.Parameter` objects, or
        a list of *parameter-group dicts*.  Each group dict must contain at
        least a ``"params"`` key; any other keys override the corresponding
        entry in ``defaults`` for that group.
    defaults : dict
        Default hyperparameter values (e.g. ``{"lr": 1e-3, "weight_decay": 0}``).
        These are copied into each param group that does not override them.

    Attributes
    ----------
    param_groups : list of dict
        List of parameter groups.  Each entry is a dict with at least
        ``"params": list[Parameter]`` plus all hyperparameter keys from
        ``defaults`` (possibly overridden per group).
    defaults : dict
        The default hyperparameters passed at construction.
    state : dict
        Per-parameter optimizer state indexed by flat parameter index.
        Populated by :meth:`state_dict` and consumed by
        :meth:`load_state_dict`; an optimizer that keeps its state in
        Python (:class:`~lucid.optim.SparseAdam`) holds its live state here.

    Notes
    -----
    Every concrete ``step()`` override is automatically wrapped at class
    definition time (via :meth:`__init_subclass__`) to flush all parameter
    tensors to the Metal GPU after each update.  This ensures that
    parameter values are materialised before the next forward pass.

    Each step reads the hyper-parameters from ``param_groups``, so a value
    assigned there takes effect at the next :meth:`step`, the way a
    hand-written warm-up or fine-tuning schedule expects:

    .. code-block:: python

        for group in optimizer.param_groups:
            group["lr"] = 1e-4

    Each group is checked against the optimizer's rules when it is added and
    again when an edited group is next used.  A value the constructor would
    reject (a negative ``lr``, a beta outside ``[0, 1)``) raises
    ``InvalidArgument``, a ``ValueError``, and nothing of that group is
    applied.

    Parameter groups allow different hyperparameters per group:

    .. code-block:: python

        optimizer = optim.SGD(
            [
                {"params": model.backbone.parameters(), "lr": 1e-3},
                {"params": model.head.parameters(),     "lr": 1e-2},
            ],
            lr=1e-4,   # default, overridden per group above
        )

    Examples
    --------
    ``Optimizer`` is not used directly.  Use a concrete subclass:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> model = nn.Linear(4, 2)
    >>> loss = model(lucid.randn(1, 4)).sum()
    >>> import lucid.optim as optim
    >>> optimizer = optim.Adam(model.parameters(), lr=1e-3)
    >>> optimizer.zero_grad()
    >>> loss.backward()
    >>> optimizer.step()
    """

    # Class-level flag controlling whether ``step()`` forces an eval of every
    # parameter tensor after the update.  Historical default was True — meant
    # as a "safety flush" for Metal params — but profiling on M4 Max
    # (May 2026) showed it costs ~37% of training throughput at small batch
    # sizes because it shatters the MLX lazy-graph pipeline that would
    # otherwise chain forward → backward → step into one submission.
    #
    # The reference framework's optimizers do NOT force-eval after step();
    # they rely on the next ``.item()`` / ``.cpu()`` / numpy bridge call
    # to materialise.
    # We mirror that default — set this class attribute to True (per-instance
    # via ``optimizer.AUTO_EVAL_AFTER_STEP = True`` or globally on a
    # subclass) only when you need step() to act as a synchronisation point.
    AUTO_EVAL_AFTER_STEP: ClassVar[bool] = False

    # The hyper-parameter table: the ``param_groups`` keys this optimizer's
    # rules and engine take.  It drives both the checks of a group (when it
    # is added, and when it is next used after an edit) and the hand-over of
    # edited values to the group's engine.  Empty for an optimizer with
    # nothing to check.
    _HYPERPARAMS: ClassVar[tuple[str, ...]] = ()

    # The engine optimizer class whose ``check_hyperparams`` states the
    # rules — the same checks its constructor and ``set_hyperparams`` run.
    # ``None`` for an optimizer that runs in Python; that one overrides
    # :meth:`_check_hyperparams`.
    _ENGINE_RULES: ClassVar[_EngineRules | None] = None

    # ``True`` for an optimizer whose live per-parameter state *is*
    # :attr:`state` (SparseAdam).  Checkpointing then snapshots and rebuilds
    # that state through ``_save_state`` / ``_load_state`` instead of
    # replacing it with the host-array snapshot, which ``step`` cannot read.
    _LIVE_STATE: ClassVar[bool] = False

    # Pair hyper-parameters exposed in ``param_groups`` as a 2-tuple (e.g.
    # ``"betas": ("beta1", "beta2")``, ``"etas": ("eta_minus", "eta_plus")``)
    # alongside the split keys the C++ engine bindings take.
    _PAIR_KEYS: ClassVar[dict[str, tuple[str, str]]] = {}

    # ``True`` for an engine optimizer whose ``decoupled_weight_decay`` is
    # applied on the parameter buffer before ``optim.step()`` (RAdam, NAdam).
    _PYTHON_DECOUPLED_WD: ClassVar[bool] = False

    # The reader of ``_HYPERPARAMS`` out of a group, derived once per class
    # by :meth:`__init_subclass__`.  On the class rather than the instance,
    # so an optimizer pickles without it.
    _read_hparams: ClassVar[_GroupReader | None] = None

    @override
    def __init_subclass__(cls, **kwargs: object) -> None:
        """Wrap every concrete ``step()`` *only when* ``AUTO_EVAL_AFTER_STEP``
        is set on the subclass.

        3.2.2 change: the default (``AUTO_EVAL_AFTER_STEP = False``) keeps
        the user's overridden ``step()`` as-is — no closure layer, no
        per-call ``type(self).AUTO_EVAL_AFTER_STEP`` branch.  The wrapper
        existed historically to support the optional eager flush, but
        when the flag is False (default since 3.0.3) the wrapper just
        adds dispatch overhead with no semantic effect.

        To opt back into the synchronous flush, set
        ``AUTO_EVAL_AFTER_STEP = True`` on the subclass *before* it is
        first declared; the wrapper is then installed.  Toggling the
        flag at runtime is still possible — see ``step()``'s post-hoc
        wrapping in subclasses that need it.

        Also derives the class's ``_HYPERPARAMS`` reader.
        """
        super().__init_subclass__(**kwargs)
        cls._read_hparams = _hyperparam_reader(cls._HYPERPARAMS)
        if not cls.AUTO_EVAL_AFTER_STEP:
            # Default path: no wrapper, user's step() runs raw.
            return
        if "step" in cls.__dict__:
            _orig: Callable[[Optimizer, _OptimizerClosure], Tensor | None] = (
                cls.__dict__["step"]
            )

            def _step_with_eval(
                self: Optimizer, closure: _OptimizerClosure = None
            ) -> Tensor | None:
                result = _orig(self, closure)
                if type(self).AUTO_EVAL_AFTER_STEP:
                    self._metal_eval_params()
                return result

            _step_with_eval.__name__ = "step"
            _step_with_eval.__doc__ = _orig.__doc__
            cls.step = _step_with_eval  # type: ignore[method-assign]

    def _metal_eval_params(self) -> None:
        """Flush all parameter tensors via C++ eval_tensors() — no mlx import.

        Called automatically after every optimizer step when
        ``AUTO_EVAL_AFTER_STEP`` is True.  GPU tensors are flushed in one
        C++ call; CPU tensors are ignored.  Most training loops never need
        this — the lazy graph evaluates implicitly on the next
        ``loss.item()`` / ``.cpu()`` / numpy bridge call.
        """

        impls: list[_C_engine.TensorImpl] = [
            p._impl for group in self.param_groups for p in _group_params(group)
        ]
        if impls:
            _C_engine.eval_tensors(impls)

    def __init__(
        self,
        params: Iterable[Parameter] | Iterable[dict[str, object]],
        defaults: dict[str, object],
    ) -> None:
        """Initialise the Optimizer.  See the class docstring for parameter semantics."""
        if (
            isinstance(params, (list, tuple))
            and len(params) > 0
            and isinstance(params[0], dict)
        ):
            param_groups: list[dict[str, object]] = list(params)
        else:
            param_groups = [{"params": list(params)}]

        # An optimiser over nothing steps nothing, and the loss still goes
        # down because the rest of the model is learning — so the usual way
        # to find out is a model that trains worse than it should for
        # reasons nobody can name.  The commonest cause is a lazy layer
        # whose parameters did not exist yet when ``parameters()`` was read;
        # ``Module.parameters`` warns about that case separately, and this
        # catches it when the whole model was lazy.
        if not any(group.get("params") for group in param_groups):
            raise ValueError(
                "optimizer got an empty parameter list.  If the model has "
                "lazy layers, run one forward pass before building the "
                "optimizer so their parameters exist."
            )

        self.param_groups: list[dict[str, object]] = []
        self._engines: list[_EngineOptimizer] = []
        # Parallel to ``_engines``: the parameters each engine optimizer was
        # built over, and the impls they held at that moment.
        self._engine_binds: list[tuple[list[Parameter], list[_C_engine.TensorImpl]]] = (
            []
        )
        self._engine_state_reset_warned: bool = False
        self._engines_built: bool = False
        # Parallel to ``param_groups``: each group's hyper-parameters as they
        # were last checked or handed to its engine.
        self._group_hparams: list[tuple[object, ...]] = []
        self.state: dict[int, dict[str, object]] = {}
        self._normalize_group_pairs(defaults, defaults)
        self.defaults: dict[str, object] = defaults

        for group in param_groups:
            self.add_param_group(group)

    def _init_adam_family(
        self,
        params: Iterable[Parameter] | Iterable[dict[str, object]],
        core: _AdamCore,
        flags: _AdamFlags,
        **extra: object,
    ) -> None:
        """Initialise an Adam-family optimizer with ``(lr, betas, eps, weight_decay)`` and common flags."""
        lr, betas, eps, weight_decay = core
        foreach, maximize, capturable, differentiable = flags
        Optimizer.__init__(
            self,
            params,
            {
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
                "foreach": foreach,
                "maximize": maximize,
                "capturable": capturable,
                "differentiable": differentiable,
                **extra,
            },
        )

    @property
    def _engine_optims(self) -> list[_EngineOptimizer]:
        """The engine optimizers, built on first use rather than in ``__init__``.

        Each engine optimizer captures ``TensorImpl`` pointers when it is
        constructed and steps those objects directly.  Building them in
        ``__init__`` therefore freezes whatever impls the parameters
        happened to hold at that moment — and a lazy layer replaces its
        impls at the first forward, which is *after* the optimizer is
        usually built.  The engine kept stepping the old ones, so the
        layer never moved and nothing said why.

        Deferring to first use costs nothing (the first ``step`` is after
        the first forward by construction) and fixes the general case,
        not only lazy layers: any parameter whose impl is replaced
        between construction and the first step is now picked up.

        Impls are also replaced *after* the first step — ``module.to()``,
        ``.double()``, ``.half()`` and any other conversion through
        ``Module._apply`` give each parameter a new buffer under the same
        ``Parameter`` object.  An engine built before that kept stepping
        the old buffer, so the parameter silently stopped training.  Every
        use therefore re-checks each group's impls against the ones its
        engine holds and rebuilds a group's engine when they differ,
        carrying its state across (see :meth:`_rebuild_engine_optim`).
        """
        if not self._engines_built:
            # Every group is checked before anything is built, so a group
            # edited to a rejected value fails the step with no engine half
            # made.  The engines are then built from these same values.
            resolved = [
                self._resolve_group_hparams(idx, group)
                for idx, group in enumerate(self.param_groups)
            ]
            for values, _ in resolved:
                self._check_hyperparams(self._hyperparam_dict(values))
            for group, (_, upd) in zip(self.param_groups, resolved, strict=True):
                if upd:
                    group.update(upd)
            hparams = [values for values, _ in resolved]
            # Set first: ``_append_engine_optim`` appends through this
            # same property, and a re-entrant build would recurse.
            self._engines_built = True
            for group in self.param_groups:
                self._bind_engine_optim(group)
            self._group_hparams = hparams
            if not self._LIVE_STATE and self.state:
                self._load_engine_state(self.state)
        else:
            self._rebind_replaced_params()
            self._sync_group_hyperparams()
        return self._engines

    def add_param_group(self, group: dict[str, object]) -> None:
        """Add a parameter group, creating one new engine optimizer for it.

        The group's hyper-parameters are checked first, against the rules
        the optimizer's constructor applies; a group that breaks one raises
        ``InvalidArgument`` and is not added.
        """
        merged: dict[str, object] = {**self.defaults, **group}
        raw_params = merged["params"]
        if not isinstance(raw_params, Iterable):
            raise TypeError(
                f"a param group's 'params' must be an iterable, got {type(raw_params).__name__}"
            )
        merged["params"] = list(raw_params)
        self._normalize_group_pairs(group, merged)
        self._validate_group_flags(merged)
        values: tuple[object, ...] = self._hyperparams_of(merged)
        self._check_hyperparams(self._hyperparam_dict(values))
        self.param_groups.append(merged)
        self._group_hparams.append(values)
        # Before the build, the group is simply on the list the build reads;
        # after it, the new group needs an engine of its own right now.
        if self._engines_built:
            self._bind_engine_optim(merged)

    def _normalize_group_pairs(
        self, group: dict[str, object], merged: dict[str, object]
    ) -> None:
        """Synchronize 2-tuple pair keys and their split scalar entries in ``merged``."""
        for pair_key, (k0, k1) in self._PAIR_KEYS.items():
            if k0 in group or k1 in group:
                merged[pair_key] = (merged[k0], merged[k1])
                continue
            if pair_key not in merged:
                continue
            raw_pair = merged[pair_key]
            if not isinstance(raw_pair, (tuple, list)) or len(raw_pair) != 2:
                raise _C_engine.InvalidArgument(
                    f"{type(self).__name__}: {pair_key} must be a 2-tuple"
                )
            merged[pair_key] = (raw_pair[0], raw_pair[1])
            merged[k0] = raw_pair[0]
            merged[k1] = raw_pair[1]

    def _validate_group_flags(self, merged: dict[str, object]) -> None:
        """Validate ``differentiable``, ``maximize``, and ``decoupled_weight_decay``."""
        if merged.get("differentiable", False):
            raise _C_engine.NotImplementedError(
                f"{type(self).__name__}: differentiable=True is not supported"
            )
        for key in ("maximize", "decoupled_weight_decay"):
            if key in merged and not isinstance(merged[key], bool):
                raise TypeError(
                    f"{type(self).__name__}: {key} must be a bool, "
                    f"got {type(merged[key]).__name__}"
                )

    # ── hyper-parameters ──────────────────────────────────────────────────────
    #
    # ``_HYPERPARAMS`` names the group keys; the engine class (or a Python
    # optimizer's override of ``_check_hyperparams``) holds the rules.  A
    # group is checked when it is added, and every use of the engines
    # compares each group with the values last handed over — one tuple
    # comparison per group — and hands over only a group that changed.

    def _hyperparams_of(self, group: dict[str, object]) -> tuple[object, ...]:
        """Return ``group``'s hyper-parameters in ``_HYPERPARAMS`` order."""
        read = self._read_hparams
        return () if read is None else tuple(read(group))

    def _reconcile_pair(
        self,
        pair_key: str,
        keys: tuple[str, str],
        group: dict[str, object],
        old_pair: tuple[object, object],
    ) -> tuple[object, object, dict[str, object]]:
        """Reconcile one ``(pair_key, (k0, k1))`` entry against ``old_pair``."""
        k0, k1 = keys
        cur_split = (group[k0], group[k1])
        raw_pair = group.get(pair_key, cur_split)
        if not isinstance(raw_pair, (tuple, list)) or len(raw_pair) != 2:
            raise _C_engine.InvalidArgument(
                f"{type(self).__name__}: {pair_key} must be a 2-tuple"
            )
        cur_pair = (raw_pair[0], raw_pair[1])
        if cur_split != old_pair and cur_pair == old_pair:
            return cur_split[0], cur_split[1], {pair_key: cur_split}
        if cur_pair != cur_split:
            return (
                cur_pair[0],
                cur_pair[1],
                {pair_key: cur_pair, k0: cur_pair[0], k1: cur_pair[1]},
            )
        upd: dict[str, object] = (
            {} if isinstance(raw_pair, tuple) else {pair_key: cur_pair}
        )
        return cur_pair[0], cur_pair[1], upd

    def _resolve_group_hparams(
        self, idx: int, group: dict[str, object]
    ) -> tuple[tuple[object, ...], dict[str, object]]:
        """Resolve ``group``'s hyper-parameters, reconciling any edited pair keys."""
        raw = list(self._hyperparams_of(group))
        if not self._PAIR_KEYS or idx >= len(self._group_hparams):
            return tuple(raw), {}
        last = self._group_hparams[idx]
        slot_of = {k: i for i, k in enumerate(self._HYPERPARAMS)}
        updates: dict[str, object] = {}
        for pair_key, (k0, k1) in self._PAIR_KEYS.items():
            i0, i1 = slot_of[k0], slot_of[k1]
            v0, v1, upd = self._reconcile_pair(
                pair_key, (k0, k1), group, (last[i0], last[i1])
            )
            raw[i0], raw[i1] = v0, v1
            updates.update(upd)
        return tuple(raw), updates

    def _hyperparam_dict(self, values: tuple[object, ...]) -> dict[str, Any]:
        """Name ``values`` by ``_HYPERPARAMS``, as the engine bindings take them."""
        return dict(zip(self._HYPERPARAMS, values, strict=True))

    @classmethod
    def _check_hyperparams(cls, values: dict[str, Any]) -> None:
        """Raise ``InvalidArgument`` when a group's hyper-parameters break a rule.

        The engine class states the rules, so a group is held to exactly the
        checks its engine's constructor and ``set_hyperparams`` run.  An
        optimizer that runs in Python overrides this with its own.

        Parameters
        ----------
        values : dict
            The group's ``_HYPERPARAMS`` entries, by name.  Empty for an
            optimizer without a table, which has nothing to check.
        """
        if values and cls._ENGINE_RULES is not None:
            cls._ENGINE_RULES.check_hyperparams(values)

    def _sync_group_hyperparams(self) -> None:
        """Hand every edited group's hyper-parameters to where they apply.

        An engine optimizer receives them through ``set_hyperparams``, which
        checks the whole set before applying any of it; an optimizer without
        an engine, which reads its groups as it steps, has them checked.  A
        group equal to what was last handed over is skipped, so a step after
        no edit makes no binding call.  A rejected group raises and stays
        marked as changed, so every later step raises until it is fixed.
        """
        read = self._read_hparams
        if read is None:
            return
        seen = self._group_hparams
        for idx, (group, last) in enumerate(zip(self.param_groups, seen, strict=True)):
            values, upd = self._resolve_group_hparams(idx, group)
            if values == last and not upd:
                continue
            if values != last:
                self._apply_hyperparams(idx, self._hyperparam_dict(values))
            if upd:
                group.update(upd)
            seen[idx] = values

    def _apply_hyperparams(self, idx: int, values: dict[str, Any]) -> None:
        """Hand group ``idx``'s edited hyper-parameters to its engine.

        A group without an engine (not built yet, or an optimizer that runs
        in Python) has them checked instead.
        """
        self._check_hyperparams(values)
        engines = self._engines
        engine = engines[idx] if idx < len(engines) else None
        if engine is not None:
            hp = (
                {**values, "weight_decay": 0.0}
                if self._PYTHON_DECOUPLED_WD
                and self.param_groups[idx].get("decoupled_weight_decay", False)
                else values
            )
            engine.set_hyperparams(hp)

    @staticmethod
    def _group_binding(
        group: dict[str, object],
    ) -> tuple[list[Parameter], list[_C_engine.TensorImpl]]:
        """Snapshot a group's parameters and the impls they hold right now."""
        params: list[Parameter] = list(_group_params(group))
        return params, [p._impl for p in params]

    def _bind_engine_optim(self, group: dict[str, object]) -> None:
        """Build a group's engine optimizer and remember what it was built over."""
        built: int = len(self._engines)
        self._append_engine_optim(group)
        if len(self._engines) > built:
            self._engine_binds.append(self._group_binding(group))

    def _build_engine_optim(self, group: dict[str, object]) -> _EngineOptimizer | None:
        """Build one engine optimizer for ``group`` without appending it.

        ``_append_engine_optim`` is the per-optimizer hook and appends to
        ``_engines``; it runs against a scratch list here so the result can
        replace a group's existing engine in place.
        """
        live: list[_EngineOptimizer] = self._engines
        self._engines = []
        try:
            self._append_engine_optim(group)
            built: list[_EngineOptimizer] = self._engines
        finally:
            self._engines = live
        return built[-1] if built else None

    def _rebind_replaced_params(self) -> None:
        """Rebuild the engine of every group whose parameter impls changed.

        One identity comparison per parameter, so the common case — nothing
        replaced — stays a cheap scan.  An optimizer without engines (LBFGS,
        SparseAdam) has no bindings and is skipped.
        """
        binds = self._engine_binds
        if len(binds) != len(self.param_groups):
            return
        impl_of = operator.attrgetter("_impl")
        for idx, group in enumerate(self.param_groups):
            params: list[Parameter] = _group_params(group)
            impls: list[_C_engine.TensorImpl] = binds[idx][1]
            if len(params) == len(impls) and all(
                map(operator.is_, map(impl_of, params), impls)
            ):
                continue
            self._rebuild_engine_optim(idx, group)

    def _rebuild_engine_optim(self, idx: int, group: dict[str, object]) -> None:
        """Replace group ``idx``'s engine with one over its current impls.

        The engine optimizer binds ``TensorImpl`` pointers at construction
        and has no way to re-point them, so a new one is built from the
        group (whose hyperparameters are the live ones — schedulers write
        there) and the old one's state is carried over through
        ``state_buffers`` / ``load_state_buffers``: entries are matched by
        ``Parameter`` identity, buffers are cast to the new impl's dtype and
        moved to its device, and per-parameter scalars (``step``,
        ``mu_product``) pass as they are (see :func:`_state_like`).

        Moving the state is a deliberate superset of the reference
        framework, which leaves state where it was after ``module.to()``
        and then fails at the next step for most optimizers.  State
        following the parameter is its own ``load_state_dict`` policy too.
        A buffer whose shape no longer matches its parameter is dropped.
        """
        values, upd = self._resolve_group_hparams(idx, group)
        self._check_hyperparams(self._hyperparam_dict(values))
        if upd:
            group.update(upd)
        old: _EngineOptimizer = self._engines[idx]
        old_params: list[Parameter] = self._engine_binds[idx][0]
        fresh: _EngineOptimizer | None = self._build_engine_optim(group)
        self._engine_binds[idx] = self._group_binding(group)
        # Built from the group as it is now, edits included.
        self._group_hparams[idx] = values
        if fresh is None:
            return
        self._engines[idx] = fresh

        buffers = old.state_buffers()
        if not buffers:
            # Nothing exported: no state yet, or an engine that keeps state
            # it cannot hand over — which only this optimizer can tell.
            if self._engine_holds_state(group):
                self._warn_engine_state_reset()
            return

        slot_of: dict[int, int] = {id(p): j for j, p in enumerate(old_params)}
        params: list[Parameter] = _group_params(group)
        carried: list[tuple[str, list[_C_engine.TensorImpl | None]]] = []
        for name, tensors in buffers:
            moved: list[_C_engine.TensorImpl | None] = []
            for p in params:
                j: int | None = slot_of.get(id(p))
                state = tensors[j] if j is not None and j < len(tensors) else None
                moved.append(None if state is None else _state_like(state, p._impl))
            carried.append((name, moved))
        if any(t is not None for _, moved in carried for t in moved):
            fresh.load_state_buffers(carried)

    def _engine_holds_state(self, group: dict[str, object]) -> bool:
        """Whether ``group``'s engine may hold state while exporting none.

        Read only when a rebuilt engine's ``state_buffers()`` came back
        empty, to tell "nothing to carry" from "state that could not be
        carried".  Every Lucid engine but SGD names its buffers even before
        it has any, so an empty export from one of them would be an engine
        without the hooks; the default assumes that.  SGD overrides it: its
        engine exports nothing until a momentum buffer exists.
        """
        del group
        return True

    def _warn_engine_state_reset(self) -> None:
        """Say once that a rebuilt engine started over from empty state."""
        if self._engine_state_reset_warned:
            return
        self._engine_state_reset_warned = True
        warnings.warn(
            f"{type(self).__name__}: a parameter's buffer was replaced after "
            "the optimizer had stepped (module.to(), .double(), .half() or "
            "another conversion), and this optimizer cannot carry its "
            "per-parameter state over, so that state restarted from empty.  "
            "Convert the model before building the optimizer to keep it.",
            RuntimeWarning,
            # warn ← rebuild ← rebind ← _engine_optims ← step() ← caller
            stacklevel=6,
        )

    def _build_engine(
        self, cls: Callable[..., _EngineOptimizer], group: dict[str, object]
    ) -> _EngineOptimizer:
        """Construct one engine optimizer ``cls`` for ``group``."""
        hp = self._hyperparam_dict(self._hyperparams_of(group))
        if self._PYTHON_DECOUPLED_WD and group.get("decoupled_weight_decay", False):
            hp["weight_decay"] = 0.0
        return cls([p._impl for p in _group_params(group)], **hp)

    def _apply_decoupled_wd(
        self, group: dict[str, object], params: list[Parameter]
    ) -> None:
        """Apply decoupled weight decay in Python for optimizers whose engine lacks it."""
        if not (
            self._PYTHON_DECOUPLED_WD and group.get("decoupled_weight_decay", False)
        ):
            return
        wd = _group_float(group, "weight_decay")
        if wd == 0.0:
            return
        factor = 1.0 - _group_float(group, "lr") * wd
        with _lucid.no_grad():
            for p in params:
                if p.grad is not None and p.grad.numel() > 0:
                    p._impl.copy_from((p * factor)._impl)

    def _step_one_group(
        self, group: dict[str, object], optim: _EngineOptimizer
    ) -> None:
        """Step a single parameter group's engine optimizer."""
        if group.get("differentiable", False):
            raise _C_engine.NotImplementedError(
                f"{type(self).__name__}: differentiable=True is not supported"
            )
        params = _group_params(group)
        saved_grads: list[tuple[Parameter, Tensor]] = []
        if group.get("maximize", False):
            with _lucid.no_grad():
                for p in params:
                    if p.grad is not None and p.grad.numel() > 0:
                        saved_grads.append((p, p.grad))
                        p.grad = -p.grad
        try:
            self._apply_decoupled_wd(group, params)
            optim.step()
        finally:
            for p, g in saved_grads:
                p.grad = g

    def _refuse_phantom_step(self) -> None:
        """Refuse an optimizer step under phantom mode or over a phantom parameter/gradient."""
        if _C_engine.phantom_mode():
            raise _C_engine.LucidError(
                "phantom tensor has no values: optimizer step under phantom mode"
            )
        for group in self.param_groups:
            for p in _group_params(group):
                g_impl = p._impl.grad_as_impl()
                if p._impl.is_phantom or (g_impl is not None and g_impl.is_phantom):
                    raise _C_engine.LucidError(
                        "phantom tensor has no values: optimizer step of a phantom parameter"
                    )

    def _step_engines(self, closure: _OptimizerClosure = None) -> Tensor | None:
        """Evaluate ``closure`` (if given) and step each group's engine optimizer."""
        self._refuse_phantom_step()
        loss: Tensor | None = None
        if closure is not None:
            with _lucid.enable_grad():
                loss = closure()
            self._refuse_phantom_step()
        for group, optim in zip(self.param_groups, self._engine_optims, strict=True):
            self._step_one_group(group, optim)
        return loss

    def _append_engine_optim(self, group: dict[str, object]) -> None:
        """Create and append one engine optimizer for a single param group.

        Override in subclasses. Base is a no-op so Optimizer can be subclassed
        without a C++ backend.
        """
        pass

    def _sync_hyperparams(self) -> None:
        """Hand edited ``param_groups`` hyper-parameters over now.

        Preserves all accumulated optimizer state (e.g. Adam first/second
        moments).  Every step does this anyway (see
        :meth:`_sync_group_hyperparams`); LR schedulers call it so the rate
        they write is in the engine as soon as they return, and an optimizer
        that runs in Python calls it at the top of its ``step``.

        A no-op for an engine optimizer before its engines exist — a
        scheduler constructed alongside the optimizer must not be what
        forces them into being, or the deferral above buys nothing.  The
        build reads ``param_groups``, which is where the new values already
        are.
        """
        if self._engines_built or self._ENGINE_RULES is None:
            self._sync_group_hyperparams()

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Reset the gradients of all parameters.

        ``set_to_none=True`` (the default) drops each gradient.  ``False``
        zeroes it in place, out of the graph, so a ``g = p.grad`` read
        earlier reads zeros too (see ``lucid.autograd._grad_inplace``).

        >>> import lucid
        >>> p = lucid.nn.Parameter(lucid.ones(3))
        >>> opt = lucid.optim.SGD([p], lr=0.1)
        >>> (p * 2).sum().backward()
        >>> g = p.grad
        >>> opt.zero_grad(set_to_none=False)
        >>> g.tolist()
        [0.0, 0.0, 0.0]
        """
        zero_grads_(
            (p for group in self.param_groups for p in _group_params(group)),
            set_to_none=set_to_none,
        )

    def step(self, closure: _OptimizerClosure = None) -> Tensor | None:
        """Perform a single optimization step.

        Subclasses must override this to update parameters from their
        current gradients.  Optionally accepts a *closure* that re-evaluates
        the model and returns the loss (required by some optimizers, e.g.
        LBFGS; ignored by most first-order methods).
        """
        raise NotImplementedError(
            f"{type(self).__name__}.step() is not implemented. "
            "Subclasses of Optimizer must override step()."
        )

    # ── state_dict round-trip ─────────────────────────────────────────────────
    #
    # Format follows reference framework: ``param_groups`` mirrors the live
    # groups but each ``params`` entry is replaced with a list of integer
    # parameter ids; ``state`` is keyed by those same ids. Parameter tensors
    # themselves live on the model — saving them inside the optimizer would
    # double-checkpoint the weights and break partial restores.
    #
    # Engine optimizers (Adam, AdamW, SGD) expose their per-parameter mutable
    # state via the C++ ``state_buffers``/``load_state_buffers`` hooks; the base
    # class harvests those automatically below. Subclasses that own additional
    # Python-side state (e.g. LBFGS history) should override _save_state /
    # _load_state to round-trip it.

    def _save_state(self) -> dict[int, dict[str, object]]:
        """Snapshot per-parameter state. Default: pull from engine optimizers."""
        return self._save_engine_state()

    def _load_state(self, state: dict[int, dict[str, object]]) -> None:
        """Restore per-parameter state. Default: push back to engine optimizers."""
        self._load_engine_state(state)

    def _save_engine_state(self) -> dict[int, dict[str, object]]:
        """Snapshot every engine optimizer's per-parameter state.

        Output is keyed by flat parameter index.  Each entry holds one
        numpy array per entry the engine exports for that parameter, keyed
        by the reference framework's names: buffers shaped like the
        parameter (``exp_avg``, ``exp_avg_sq``, ``momentum_buffer`` ...)
        and per-parameter scalars as 0-d arrays — ``step``, the parameter's
        own step count (int64), and e.g. NAdam's ``mu_product``.  A
        parameter that has never been updated has no entry.

        Before the engines exist there is no state to save, and building
        them here would bind whatever impls the parameters hold now — a
        model converted afterwards then needs a rebuild at the first step.
        """
        if not self._engines_built:
            return self.state
        out: dict[int, dict[str, object]] = {}
        flat_idx: int = 0
        for group, eng in zip(self.param_groups, self._engine_optims, strict=True):
            params: list[Parameter] = _group_params(group)
            buffers = eng.state_buffers()
            step_count: int = int(getattr(eng, "step_count", 0) or 0)
            for slot, _ in enumerate(params):
                snapshot: dict[str, object] = {}
                for name, tensors in buffers:
                    buf = tensors[slot] if slot < len(tensors) else None
                    if buf is not None:
                        snapshot[name] = _host_copy(buf)
                if step_count != 0 and "step" not in snapshot:
                    # An engine with one count per group, not per parameter.
                    snapshot["step"] = step_count
                if snapshot:
                    out[flat_idx + slot] = snapshot
            flat_idx += len(params)
        return out

    def _load_engine_state(self, state: dict[int, dict[str, object]]) -> None:
        """Push numpy-backed state entries back into each engine optimizer.

        Every entry, ``step`` included, goes back to its own parameter
        through ``load_state_buffers``.  A ``step`` that is a plain number
        is the older layout, one count for the whole group, and is set as
        that group's count.
        """
        if not state:
            return
        flat_idx: int = 0
        for group, eng in zip(self.param_groups, self._engine_optims, strict=True):
            params: list[Parameter] = _group_params(group)
            # Collect per-buffer-name lists running parallel to params.
            by_name: dict[str, list[_C_engine.TensorImpl | None]] = {}
            step_count: int = 0
            for slot, p in enumerate(params):
                snapshot: dict[str, object] = state.get(flat_idx + slot, {})
                for k, v in snapshot.items():
                    if k == "step" and isinstance(v, (int, float)):
                        # Folding per-parameter arrays to their maximum here
                        # gave every parameter the furthest one's count.
                        step_count = max(step_count, int(v))
                        continue
                    by_name.setdefault(k, [None] * len(params))
                    loaded_impl = _from_host(v, p._impl.device)
                    if (
                        p._impl.dtype == _C_engine.Dtype.BF16
                        and loaded_impl.dtype == _C_engine.Dtype.F32
                        and k not in ("step", "eta", "mu", "mu_product")
                    ):
                        loaded_impl = _C_engine.astype(
                            loaded_impl, _C_engine.Dtype.BF16
                        )
                    by_name[k][slot] = loaded_impl
            if by_name:
                eng.load_state_buffers(list(by_name.items()))
            if step_count and hasattr(eng, "step_count"):
                eng.step_count = step_count
            flat_idx += len(params)

    def _param_id_map(self) -> dict[int, int]:
        """Map ``id(param)`` → flat integer index across all param groups."""
        out: dict[int, int] = {}
        idx: int = 0
        for group in self.param_groups:
            for p in _group_params(group):
                out[id(p)] = idx
                idx += 1
        return out

    def state_dict(self) -> dict[str, object]:
        """Return a checkpointable snapshot of the optimizer.

        Mirrors reference framework's optimizer state_dict layout:

        - ``state``: ``{param_index: {key: value, ...}}`` — each parameter's
          optimizer state under the reference framework's keys: the engine's
          buffers and per-parameter scalars (``exp_avg``,
          ``momentum_buffer``, ``step`` ...) as numpy arrays, or the
          Python-side state of an optimizer that keeps its own (LBFGS).
        - ``param_groups``: list of group dicts; each group's ``params`` is a
          list of integer indices into the flat parameter list.
        """
        id_map: dict[int, int] = self._param_id_map()
        groups_out: list[dict[str, object]] = []
        for group in self.param_groups:
            g: dict[str, object] = {k: v for k, v in group.items() if k != "params"}
            g["params"] = [id_map[id(p)] for p in _group_params(group)]
            groups_out.append(g)
        state = self._save_state()
        if not self._LIVE_STATE:
            self.state = state
        return {"state": state, "param_groups": groups_out}

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Restore from a state_dict produced by :meth:`state_dict`.

        Hyperparameters in ``param_groups`` are restored, and each
        parameter's state (returned from :meth:`_save_state`) is restored
        via :meth:`_load_state` — for the engine optimizers, every buffer
        and per-parameter ``step`` goes back into the engine.
        """
        raw_groups = state_dict["param_groups"]
        if not isinstance(raw_groups, list):
            raise TypeError("load_state_dict: 'param_groups' must be a list")
        loaded_groups: list[dict[str, object]] = raw_groups
        if len(loaded_groups) != len(self.param_groups):
            raise ValueError(
                f"loaded state_dict has {len(loaded_groups)} param_groups but "
                f"optimizer has {len(self.param_groups)}"
            )
        for g_new, g_old in zip(self.param_groups, loaded_groups, strict=True):
            for k, v in g_old.items():
                if k == "params":
                    continue
                g_new[k] = v
        raw_state = state_dict.get("state", {})
        if not isinstance(raw_state, dict):
            raise TypeError("load_state_dict: 'state' must be a dict")
        loaded_state: dict[int, dict[str, object]] = raw_state
        self._load_state(loaded_state)
        if not self._LIVE_STATE:
            self.state = loaded_state
        self._sync_hyperparams()

    @override
    def __getstate__(self) -> dict[str, object]:
        """Snapshot engine state into ``self.state`` and drop live engine handles.

        C++ engine optimizers hold raw ``TensorImpl`` pointers and cannot be
        pickled directly.  Synchronising ``self.state`` first and clearing
        ``_engines`` / ``_engine_binds`` / ``_engines_built`` in the returned
        dict lets ``pickle`` and ``copy.deepcopy`` serialise only Python state
        while leaving the source optimizer's live engines untouched; the next
        :meth:`step` on the restored instance rebuilds its engines over the
        restored parameters and reloads ``self.state`` into them.
        """
        if not self._LIVE_STATE and self._engines_built:
            self.state = self._save_engine_state()
        state = dict(self.__dict__)
        state["_engines"] = []
        state["_engine_binds"] = []
        state["_engines_built"] = False
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore pickled or deep-copied optimizer state; engines rebuild lazily."""
        self.__dict__.update(state)
        # The scheduler's step counter counts one optimizer: every copy —
        # shallow, deep or unpickled — gets one of its own.
        step = state.get("step")
        if isinstance(step, _CountedStep):
            self.__dict__["step"] = step.on(self)
