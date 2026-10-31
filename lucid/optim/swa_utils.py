"""
Weight averaging: stochastic weight averaging (SWA) and exponential moving average (EMA).

``AveragedModel`` keeps a running average of another model's parameters (and,
optionally, buffers); ``SWALR`` anneals the learning rate to the constant SWA
phase; ``update_bn`` re-estimates BatchNorm statistics for an averaged model.
The ``get_*_avg_fn`` factories return the averaging rules.
"""

from collections.abc import Callable, Iterable, Sequence
import copy
import math
from typing import Literal, override

import lucid
from lucid._tensor.tensor import Tensor
from lucid._types import DeviceLike
from lucid.nn.module import Module
from lucid.nn.modules.normalization import _BatchNormBase
from lucid.optim._scheduler_base import _LRScheduler
from lucid.optim.optimizer import Optimizer, _format_param, _group_float

type AvgFn = Callable[[Tensor, Tensor, Tensor | int], Tensor]
type MultiAvgFn = Callable[[Sequence[Tensor], Sequence[Tensor], Tensor | int], None]

__all__ = [
    "AveragedModel",
    "update_bn",
    "SWALR",
    "get_ema_multi_avg_fn",
    "get_swa_multi_avg_fn",
    "get_ema_avg_fn",
    "get_swa_avg_fn",
]


def _check_decay(decay: float) -> None:
    if decay < 0.0 or decay > 1.0:
        raise ValueError(
            f"Invalid decay value {decay} provided. Please provide a value in [0,1] range."
        )


def _require_same_count(
    what: str, model_side: Sequence[Tensor], averaged_side: Sequence[Tensor]
) -> None:
    if len(model_side) != len(averaged_side):
        raise ValueError(
            f"model has {len(model_side)} {what}, but the averaged copy has "
            f"{len(averaged_side)}; it must have the same architecture"
        )


def _is_epoch_count(value: object) -> bool:
    return isinstance(value, int) and value >= 0


def get_ema_avg_fn(decay: float = 0.999) -> AvgFn:
    r"""Get the rule that folds one parameter into its exponential moving average.

    .. math::

        W_0^{\text{EMA}} = W_0^{\text{model}}, \qquad
        W_{t+1}^{\text{EMA}} = \text{decay} \cdot W_t^{\text{EMA}}
            + (1 - \text{decay}) \cdot W_{t+1}^{\text{model}}

    Parameters
    ----------
    decay : float, optional
        Decay rate, in ``[0, 1]`` (default: ``0.999``).

    Returns
    -------
    Callable
        ``f(ema_param, current_param, num_averaged)`` returning the new
        average of one parameter.  The count is accepted and ignored: the
        weights of an EMA do not depend on it.

    Raises
    ------
    ValueError
        If ``decay`` is outside ``[0, 1]``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.optim.swa_utils import get_ema_avg_fn
    >>> ema = get_ema_avg_fn(0.5)
    >>> ema(lucid.tensor([2.0]), lucid.tensor([4.0]), 3).tolist()
    [3.0]
    """
    _check_decay(decay)

    def ema_update(ema_param: Tensor, current_param: Tensor, num_averaged: Tensor | int) -> Tensor:
        del num_averaged  # an EMA weights the past by decay alone
        with lucid.no_grad():
            return decay * ema_param + (1 - decay) * current_param

    return ema_update


def get_swa_avg_fn() -> AvgFn:
    r"""Get the rule that folds one parameter into its equally weighted average.

    .. math::

        W^{\text{SWA}}_{n+1} = W^{\text{SWA}}_n
            + \frac{W^{\text{model}} - W^{\text{SWA}}_n}{n + 1}

    Returns
    -------
    Callable
        ``f(averaged_param, current_param, num_averaged)`` returning the new
        average of one parameter.

    Examples
    --------
    >>> import lucid
    >>> from lucid.optim.swa_utils import get_swa_avg_fn
    >>> swa = get_swa_avg_fn()
    >>> swa(lucid.tensor([2.0]), lucid.tensor([5.0]), 2).tolist()
    [3.0]
    """

    def swa_update(
        averaged_param: Tensor, current_param: Tensor, num_averaged: Tensor | int
    ) -> Tensor:
        with lucid.no_grad():
            return averaged_param + (current_param - averaged_param) / (num_averaged + 1)

    return swa_update


def _in_place(avg_fn: AvgFn) -> MultiAvgFn:
    """Lift a per-tensor averaging rule to one that updates a list of tensors in place.

    The one place the list form and the per-tensor form meet, so a rule is
    written once.  Writing into the average through ``copy_`` casts the
    result to its dtype, which is how an integer buffer (``num_batches_tracked``)
    is averaged.
    """

    def update(
        averaged_params: Sequence[Tensor],
        current_params: Sequence[Tensor],
        num_averaged: Tensor | int,
    ) -> None:
        with lucid.no_grad():
            for averaged, current in zip(averaged_params, current_params, strict=True):
                averaged.copy_(avg_fn(averaged, current, num_averaged))

    return update


def get_ema_multi_avg_fn(decay: float = 0.999) -> MultiAvgFn:
    r"""Get the in-place list form of :func:`get_ema_avg_fn`.

    Parameters
    ----------
    decay : float, optional
        Passed to :func:`get_ema_avg_fn`, which also refuses a value outside
        ``[0, 1]`` (default: ``0.999``).

    Returns
    -------
    Callable
        ``f(ema_param_list, current_param_list, num_averaged)``, which writes
        the averages into ``ema_param_list``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.optim.swa_utils import get_ema_multi_avg_fn
    >>> ema = [lucid.tensor([2.0])]
    >>> get_ema_multi_avg_fn(0.5)(ema, [lucid.tensor([4.0])], 3)
    >>> ema[0].tolist()
    [3.0]
    """
    return _in_place(get_ema_avg_fn(decay))


def get_swa_multi_avg_fn() -> MultiAvgFn:
    r"""Get the rule that folds a list of parameters into their equally weighted averages.

    The list form of :func:`get_swa_avg_fn`: it updates the averages in place.

    Returns
    -------
    Callable
        ``f(averaged_param_list, current_param_list, num_averaged)`` updating
        ``averaged_param_list`` in place.

    Examples
    --------
    >>> import lucid
    >>> from lucid.optim.swa_utils import get_swa_multi_avg_fn
    >>> avg = [lucid.tensor([2.0])]
    >>> get_swa_multi_avg_fn()(avg, [lucid.tensor([5.0])], 2)
    >>> avg[0].tolist()
    [3.0]
    """
    return _in_place(get_swa_avg_fn())


class AveragedModel(Module):
    r"""A running average of another model's weights, for SWA and EMA.

    Stochastic Weight Averaging (Izmailov et al., 2018) averages the weights
    visited late in training with equal weight; an exponential moving average
    weights recent ones more.  ``AveragedModel`` holds a copy of ``model`` and
    :meth:`update_parameters` folds ``model``'s current weights into it.

    The first :meth:`update_parameters` copies the weights; each later call
    applies ``avg_fn`` / ``multi_avg_fn``, or the equally weighted average
    when neither is given.  ``avg_fn`` and ``multi_avg_fn`` are not part of
    the ``state_dict``.

    Parameters
    ----------
    model : Module
        The model to average.  It is deep-copied.
    device : DeviceLike, optional
        Where the averaged copy lives (default: wherever ``model`` is).
    avg_fn : Callable, optional
        ``f(averaged_param, model_param, num_averaged)`` returning the new
        value of one averaged parameter.
    multi_avg_fn : Callable, optional
        ``f(averaged_params, model_params, num_averaged)`` updating a list of
        averaged parameters in place; the lists share a device and dtype.
        At most one of ``avg_fn`` and ``multi_avg_fn`` may be given.
    use_buffers : bool, optional
        Average the buffers as well as the parameters.  Otherwise the buffers
        are copied from ``model`` on every update (default: ``False``).

    Attributes
    ----------
    module : Module
        The averaged copy of ``model``.
    n_averaged : Tensor
        A ``int64`` scalar buffer: the number of models averaged so far.

    Raises
    ------
    ValueError
        If both ``avg_fn`` and ``multi_avg_fn`` are given.

    Notes
    -----
    A model with BatchNorm needs its running statistics re-estimated for the
    averaged weights, with :func:`update_bn` or ``use_buffers=True``.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
    >>> model = nn.Linear(2, 1)
    >>> swa = AveragedModel(model)
    >>> swa.update_parameters(model)
    >>> int(swa.n_averaged.item())
    1
    >>> ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.9))
    """

    n_averaged: Tensor

    def __init__(
        self,
        model: Module,
        device: DeviceLike = None,
        avg_fn: AvgFn | None = None,
        multi_avg_fn: MultiAvgFn | None = None,
        use_buffers: bool = False,
    ) -> None:
        super().__init__()
        if avg_fn is not None and multi_avg_fn is not None:
            raise ValueError("Only one of avg_fn and multi_avg_fn should be provided")
        self.module = copy.deepcopy(model)
        if device is not None:
            self.module = self.module.to(device)
        self.register_buffer("n_averaged", lucid.tensor(0, dtype=lucid.int64, device=device))
        self.avg_fn = avg_fn
        self.multi_avg_fn = multi_avg_fn
        self.use_buffers = use_buffers

    @override
    def forward(self, *args: Tensor, **kwargs: Tensor) -> Tensor:
        """Run the averaged copy of the model."""
        out: Tensor = self.module(*args, **kwargs)
        return out

    def _averaged_state(self) -> list[Tensor]:
        """The tensors that are averaged: the parameters, and the buffers if asked."""
        tensors: list[Tensor] = list(self.module.parameters())
        if self.use_buffers:
            tensors.extend(self.module.buffers())
        return tensors

    def _averaging_rule(self) -> MultiAvgFn:
        """The list rule an update applies: ``multi_avg_fn``, else ``avg_fn`` lifted, else SWA."""
        if self.multi_avg_fn is not None:
            return self.multi_avg_fn
        if self.avg_fn is not None:
            return _in_place(self.avg_fn)
        return get_swa_multi_avg_fn()

    def update_parameters(self, model: Module) -> None:
        """Fold ``model``'s current weights into the average.

        Parameters
        ----------
        model : Module
            A model of the same architecture as the one averaged.  The first
            call copies its weights; later ones apply the averaging rule.

        Raises
        ------
        ValueError
            If ``model`` does not have the same number of averaged tensors
            as the averaged copy; nothing is changed then.
        """
        averaged = self._averaged_state()
        source: list[Tensor] = list(model.parameters())
        if self.use_buffers:
            source.extend(model.buffers())
        buffers = list(self.module.buffers())
        model_buffers = list(model.buffers())
        _require_same_count("tensors to average", source, averaged)
        _require_same_count("buffers", model_buffers, buffers)

        with lucid.no_grad():
            current = [s.detach().to(a.device) for a, s in zip(averaged, source, strict=True)]
            averaged = [a.detach() for a in averaged]
            if int(self.n_averaged.item()) == 0:
                for a, c in zip(averaged, current, strict=True):
                    a.copy_(c)
            else:
                self._average(averaged, current)
            if not self.use_buffers:
                # Buffers that are not averaged stay in step with the source model.
                for b, mb in zip(buffers, model_buffers, strict=True):
                    b.detach().copy_(mb.detach().to(b.device))
            self.n_averaged.add_(1)

    def _average(self, averaged: list[Tensor], current: list[Tensor]) -> None:
        """Apply the averaging rule to each device-and-dtype group of tensors."""
        groups: dict[tuple[object, object], tuple[list[Tensor], list[Tensor]]] = {}
        for a, c in zip(averaged, current, strict=True):
            lhs, rhs = groups.setdefault((a.device, a.dtype), ([], []))
            lhs.append(a)
            rhs.append(c)
        rule = self._averaging_rule()
        for lhs, rhs in groups.values():
            rule(lhs, rhs, self.n_averaged.to(lhs[0].device))


def update_bn(
    loader: Iterable[Tensor | Sequence[Tensor]],
    model: Module,
    device: DeviceLike = None,
) -> None:
    """Re-estimate the BatchNorm running statistics of ``model`` from one pass over data.

    Every BatchNorm layer is reset and run with ``momentum=None`` — a cumulative
    average, so each batch weighs equally — while ``loader`` is consumed once.
    Each layer's momentum and the model's train/eval mode are restored afterwards.
    A model without BatchNorm is left alone.

    Parameters
    ----------
    loader : Iterable
        Batches: a tensor, or a list or tuple whose first element is the
        tensor to run the model on.
    model : Module
        The model whose BatchNorm statistics are updated.
    device : DeviceLike, optional
        If given, each batch is moved there before it is run.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.optim.swa_utils import update_bn
    >>> net = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3))
    >>> update_bn([lucid.randn(8, 3) for _ in range(4)], net)
    >>> int(net[1].num_batches_tracked.item())
    4
    """
    momenta = [
        (module, module.momentum)
        for module in model.modules()
        if isinstance(module, _BatchNormBase)
    ]
    for module, _ in momenta:
        module.reset_running_stats()
    if not momenta:
        return

    was_training = model.training
    model.train()
    for module, _ in momenta:
        module.momentum = None
    try:
        with lucid.no_grad():
            for batch in loader:
                x = batch if isinstance(batch, Tensor) else batch[0]
                if device is not None:
                    x = x.to(device)
                model(x)
    finally:
        for module, momentum in momenta:
            module.momentum = momentum
        model.train(was_training)


class SWALR(_LRScheduler):
    r"""Anneal each group's learning rate to a fixed SWA rate, then hold it there.

    Meant for the SWA phase of training (see :class:`AveragedModel`): over
    ``anneal_epochs`` epochs the rate moves from where it was to ``swa_lr``,
    by a cosine or a linear path, and stays at ``swa_lr`` after.

    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer.
    swa_lr : float or list of float
        The final rate, for all groups or one per group.
    anneal_epochs : int, optional
        Epochs in the annealing phase (default: ``10``).
    anneal_strategy : {"cos", "linear"}, optional
        Shape of the path (default: ``"cos"``).
    last_epoch : int, optional
        The index of the last epoch (default: ``-1``).

    Raises
    ------
    ValueError
        If ``anneal_strategy`` is not ``"cos"`` or ``"linear"``, if
        ``anneal_epochs`` is not an int ``>= 0``, or if a list ``swa_lr``
        does not have one value per group.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> import lucid.optim as optim
    >>> from lucid.optim.swa_utils import SWALR
    >>> opt = optim.SGD(nn.Linear(2, 1).parameters(), lr=0.1)
    >>> sched = SWALR(opt, swa_lr=0.05, anneal_epochs=2, anneal_strategy="linear")
    >>> for _ in range(3):
    ...     opt.step()
    ...     sched.step()
    >>> round(opt.param_groups[0]["lr"], 4)
    0.05
    """

    anneal_func: Callable[[float], float]

    def __init__(
        self,
        optimizer: Optimizer,
        swa_lr: float | list[float],
        anneal_epochs: int = 10,
        anneal_strategy: Literal["cos", "linear"] = "cos",
        last_epoch: int = -1,
    ) -> None:
        swa_lrs = _format_param("swa_lr", optimizer, swa_lr)
        self._set_anneal_func(anneal_strategy)
        if not _is_epoch_count(anneal_epochs):
            raise ValueError(f"anneal_epochs must be equal or greater than 0, got {anneal_epochs}")
        for lr, group in zip(swa_lrs, optimizer.param_groups, strict=True):
            group["swa_lr"] = lr
        self.anneal_epochs = anneal_epochs
        super().__init__(optimizer, last_epoch)

    @staticmethod
    def _linear_anneal(t: float) -> float:
        return t

    @staticmethod
    def _cosine_anneal(t: float) -> float:
        return (1 - math.cos(math.pi * t)) / 2

    @staticmethod
    def _get_initial_lr(lr: float, swa_lr: float, alpha: float) -> float:
        """The rate the annealing started from, recovered from the current rate ``lr``."""
        if alpha == 1:
            return swa_lr
        return (lr - alpha * swa_lr) / (1 - alpha)

    def _set_anneal_func(self, anneal_strategy: Literal["cos", "linear"]) -> None:
        """Choose the annealing path; the one place a strategy name is validated."""
        paths = {"cos": self._cosine_anneal, "linear": self._linear_anneal}
        if anneal_strategy not in paths:
            raise ValueError(
                f"anneal_strategy must be one of 'cos' or 'linear', instead got {anneal_strategy}"
            )
        self._anneal_strategy = anneal_strategy
        self.anneal_func = paths[anneal_strategy]

    def _epoch(self) -> int:
        """The scheduler's step count: 0 at construction, 1 after the first ``step()``.

        It counts steps taken, not ``last_epoch``, so a scheduler resumed with
        ``last_epoch`` set anneals from its own beginning, as the reference does.
        """
        return self._step_count - (1 if self._is_initial else 0)

    @override
    def get_lr(self) -> list[float]:
        """Interpolate each group's rate toward its ``swa_lr`` over ``anneal_epochs``.

        The path starts where the group's rate was when the annealing began —
        recovered from the current rate — so the schedule composes with the
        one that ran before it.
        """
        step = self._epoch()
        if self.anneal_epochs == 0:
            step = max(1, step)
        prev_t = max(0.0, min(1.0, (step - 1) / max(1, self.anneal_epochs)))
        prev_alpha = self.anneal_func(prev_t)
        prev_lrs = [
            self._get_initial_lr(_group_float(g), _group_float(g, "swa_lr"), prev_alpha)
            for g in self.optimizer.param_groups
        ]
        t = max(0.0, min(1.0, step / max(1, self.anneal_epochs)))
        alpha = self.anneal_func(t)
        return [
            _group_float(g, "swa_lr") * alpha + lr * (1 - alpha)
            for g, lr in zip(self.optimizer.param_groups, prev_lrs, strict=True)
        ]

    @override
    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Restore state produced by :meth:`state_dict`, and the annealing path it names."""
        super().load_state_dict(state_dict)
        self._set_anneal_func(self._anneal_strategy)
