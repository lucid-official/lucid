"""
Reduce the learning rate when a monitored metric stops improving.
"""

from collections.abc import Mapping
import math
from typing import ClassVar, Literal, SupportsFloat, override
import warnings

from lucid._tensor.tensor import Tensor
from lucid.optim._scheduler_base import EPOCH_DEPRECATION_WARNING, _CheckpointState
from lucid.optim.optimizer import Optimizer, _group_float

#: Keys a checkpoint written before the attributes were public carries.
_LEGACY_KEYS = {
    "_best": "best",
    "_num_bad_epochs": "num_bad_epochs",
    "_cooldown_counter": "cooldown_counter",
}


def _as_scalar(metrics: SupportsFloat) -> float:
    """The metric as a plain ``float``, which holds no tensor and no graph.

    The scheduler keeps the best value it has seen, and a tensor kept there
    was the loss itself: its autograd graph stayed alive between epochs, an
    in-place change to it moved the stored best, and a checkpoint carried
    it.  The value is read once, here.
    """
    if isinstance(metrics, Tensor) and metrics.numel() != 1:
        raise ValueError(
            f"ReduceLROnPlateau.step() takes a scalar metric, got a tensor of shape "
            f"{tuple(metrics.shape)}"
        )
    return float(metrics)


class ReduceLROnPlateau(_CheckpointState):
    r"""Reduce the learning rate when a monitored metric stops improving.

    Once the metric fails to improve by more than ``threshold`` for
    ``patience`` consecutive epochs, the learning rate is multiplied by
    ``factor``.  This is useful when progress stalls and a smaller step
    size may help escape a plateau:

    .. math::

        \eta \leftarrow \max(\eta \cdot \text{factor},\; \eta_{\min})

    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer.
    mode : {"min", "max"}, optional
        In ``"min"`` mode the LR is reduced when the metric has stopped
        decreasing; in ``"max"`` mode when it has stopped increasing
        (default: ``"min"``).
    factor : float, optional
        Multiplicative factor by which the learning rate is reduced
        (default: ``0.1``).  Must be less than 1.
    patience : int, optional
        Number of epochs with no improvement after which the LR is
        reduced (default: ``10``).
    threshold : float, optional
        Minimum change to qualify as an improvement (default: ``1e-4``).
    threshold_mode : {"rel", "abs"}, optional
        Whether ``threshold`` is relative to the best value or absolute
        (default: ``"rel"``).
    cooldown : int, optional
        Epochs to wait after a reduction before counting bad epochs again
        (default: ``0``).
    min_lr : float or list of float, optional
        A lower bound on the learning rate, for every param group or one
        per group (default: ``0``).
    eps : float, optional
        A reduction smaller than this is not applied (default: ``1e-8``).

    Attributes
    ----------
    optimizer : Optimizer
        The optimizer being scheduled.
    best : float
        The best metric seen so far.
    num_bad_epochs : int
        Epochs since the metric last improved (reset by a reduction).
    cooldown_counter : int
        Epochs left of the cooldown after the last reduction.
    last_epoch : int
        Number of :meth:`step` calls so far.
    min_lrs : list of float
        The lower bound for each param group.

    Raises
    ------
    ValueError
        If ``factor >= 1``, ``mode`` or ``threshold_mode`` is unknown, or
        ``min_lr`` is a list whose length is not the number of param groups.
    TypeError
        If ``optimizer`` is not an :class:`~lucid.optim.Optimizer`.

    Notes
    -----
    Unlike the epoch-based schedulers, ``ReduceLROnPlateau`` needs a metric
    at each :meth:`step` call rather than advancing by a fixed epoch count,
    so a :class:`SequentialLR` or :class:`ChainedScheduler` cannot hold one.

    The metric is stored as a Python ``float``: a tensor passed in is read
    once and neither it nor its graph is kept.

    A typical use-case is to monitor validation loss:

    .. code-block:: python

        scheduler = optim.ReduceLROnPlateau(optimizer, mode="min", patience=5)
        for epoch in range(epochs):
            train(...)
            val_loss = validate(...)
            scheduler.step(val_loss)

    Examples
    --------
    >>> import lucid
    >>> weight = lucid.nn.Parameter(lucid.ones(3))
    >>> optimizer = lucid.optim.SGD([weight], lr=0.1)
    >>> scheduler = lucid.optim.ReduceLROnPlateau(optimizer, factor=0.5, patience=1)
    >>> for loss in (1.0, 1.0, 1.0, 1.0):  # no improvement: reduced at the third
    ...     optimizer.step()
    ...     scheduler.step(loss)
    >>> scheduler.get_last_lr()
    [0.05]
    """

    # so the composites, which step without one, refuse it
    _steps_on_a_metric: ClassVar[bool] = True

    def __init__(
        self,
        optimizer: Optimizer,
        mode: Literal["min", "max"] = "min",
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: Literal["rel", "abs"] = "rel",
        cooldown: int = 0,
        min_lr: list[float] | float = 0,
        eps: float = 1e-8,
    ) -> None:
        """Initialise the ReduceLROnPlateau.  See the class docstring for parameter semantics."""
        if factor >= 1.0:
            raise ValueError("Factor should be < 1.0.")
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")
        self.factor = factor
        self.optimizer = optimizer
        groups = len(optimizer.param_groups)
        self.default_min_lr: float | None
        if isinstance(min_lr, (list, tuple)):
            if len(min_lr) != groups:
                raise ValueError(f"expected {groups} min_lrs, got {len(min_lr)}")
            self.default_min_lr = None
            self.min_lrs = [float(lr) for lr in min_lr]
        else:
            self.default_min_lr = float(min_lr)
            self.min_lrs = [self.default_min_lr] * groups
        self.patience = patience
        self.cooldown = cooldown
        self.eps = eps
        self.last_epoch = 0
        self._last_lr = self._current_lrs()
        self._init_is_better(mode, threshold, threshold_mode)
        self._reset()

    def _current_lrs(self) -> list[float]:
        return [_group_float(g) for g in self.optimizer.param_groups]

    def _reset(self) -> None:
        """Forget the metric history and any cooldown."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def _init_is_better(self, mode: str, threshold: float, threshold_mode: str) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode {mode} is unknown!")
        if threshold_mode not in ("rel", "abs"):
            raise ValueError(f"threshold mode {threshold_mode} is unknown!")
        self.mode_worse = math.inf if mode == "min" else -math.inf
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

    @property
    def in_cooldown(self) -> bool:
        """Whether a reduction's cooldown is still running."""
        return self.cooldown_counter > 0

    def _is_better(self, value: float, best: float) -> bool:
        rel = self.threshold_mode == "rel"
        if self.mode == "min":
            return value < (best * (1.0 - self.threshold) if rel else best - self.threshold)
        return value > (best * (1.0 + self.threshold) if rel else best + self.threshold)

    def _min_lrs_for_groups(self) -> list[float]:
        """The lower bounds, one per group the optimizer has now."""
        groups = len(self.optimizer.param_groups)
        if groups == len(self.min_lrs):
            return self.min_lrs
        if self.default_min_lr is None:
            raise RuntimeError(
                f"The number of param groups in the `optimizer` ({groups}) differs from "
                f"when `ReduceLROnPlateau` was initialized ({len(self.min_lrs)}), usually "
                "due to a new param group being added to the optimizer. Please modify "
                "the `min_lrs` field to match the length of the `optimizer` param groups."
            )
        return [self.default_min_lr] * groups

    def _reduce_lr(self, min_lrs: list[float]) -> None:
        for group, floor in zip(self.optimizer.param_groups, min_lrs, strict=True):
            old_lr = _group_float(group)
            new_lr = max(old_lr * self.factor, floor)
            if old_lr - new_lr > self.eps:
                group["lr"] = new_lr
        self.optimizer._sync_hyperparams()

    def step(self, metrics: SupportsFloat, epoch: int | None = None) -> None:
        """Update the scheduler with the latest monitored metric value.

        Parameters
        ----------
        metrics : float or Tensor
            The monitored value; a tensor must hold exactly one element.
        epoch : int, optional
            Deprecated.  Sets :attr:`last_epoch` instead of counting up.

        Raises
        ------
        ValueError
            If ``metrics`` is a tensor with more than one element.
        RuntimeError
            If a reduction is due and param groups were added after the
            scheduler was built with a per-group ``min_lr``.  The scheduler
            is left as it was.
        """
        current = _as_scalar(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        else:
            warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning, stacklevel=2)
        better = self._is_better(current, self.best)
        best = current if better else self.best
        bad = 0 if better else self.num_bad_epochs + 1
        cooldown = self.cooldown_counter
        if self.in_cooldown:
            cooldown -= 1
            bad = 0  # bad epochs inside a cooldown do not count
        if bad > self.patience:
            self._reduce_lr(self._min_lrs_for_groups())
            cooldown = self.cooldown
            bad = 0
        self.last_epoch = epoch
        self.best, self.num_bad_epochs, self.cooldown_counter = best, bad, cooldown
        self._last_lr = self._current_lrs()

    def get_last_lr(self) -> list[float]:
        """The learning rate of each param group as of the last :meth:`step`.

        Returns
        -------
        list of float
            One rate per param group, a copy.
        """
        return list(self._last_lr)

    @override
    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore state produced by :meth:`state_dict`.

        A checkpoint from before the metric history had public names
        (``_best``, ``_num_bad_epochs``, ``_cooldown_counter``, a scalar
        ``min_lr``) loads too.
        """
        incoming = {
            _LEGACY_KEYS.get(key, key): value
            for key, value in state_dict.items()
            if key not in ("verbose", "min_lr")
        }
        legacy_min = state_dict.get("min_lr")
        if isinstance(legacy_min, (int, float)):
            incoming.setdefault("default_min_lr", float(legacy_min))
            incoming.setdefault("min_lrs", [float(legacy_min)] * len(self.optimizer.param_groups))
        best = incoming.get("best")
        if isinstance(best, Tensor):
            incoming["best"] = float(best)
        super().load_state_dict(incoming)
        self._init_is_better(self.mode, self.threshold, self.threshold_mode)
