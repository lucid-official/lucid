"""
Learning rate schedulers.
"""

from bisect import bisect_right
from collections.abc import Callable, Sequence
import math
from typing import override
from lucid.optim._reduce_on_plateau import ReduceLROnPlateau as ReduceLROnPlateau
from lucid.optim._scheduler_base import (
    _CallerRated,
    _CompositeSchedulerMixin,
    _LRScheduler,
    _all_or_nothing,
    _check_schedulers,
    _require_group_keys,
    _reset_to_base,
)
from lucid.optim.optimizer import (
    Optimizer,
    _CyclicLike,
    _OneCycleLike,
    _cyclic_get_lr,
    _group_float,
    _init_cyclic,
    _init_one_cycle,
    _one_cycle_get_lr,
)


class StepLR(_LRScheduler):
    r"""Decay the learning rate by a fixed multiplicative factor every fixed number of epochs.

    At each epoch that is a multiple of ``step_size``, every param-group
    learning rate is multiplied by ``gamma``:

    .. math::

        \eta_t =
        \begin{cases}
            \eta_{t-1} \times \gamma & \text{if } t \bmod \text{step\_size} = 0 \\
            \eta_{t-1} & \text{otherwise}
        \end{cases}

    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer.
    step_size : int
        Period (in epochs) between learning rate decays.
    gamma : float, optional
        Multiplicative factor applied at each decay step (default: ``0.1``).
    last_epoch : int, optional
        The index of the last epoch (default: ``-1``).
    verbose : bool, optional
        Print the updated LR after each step if ``True`` (default: ``False``).

    Attributes
    ----------
    step_size : int
        Decay period in epochs.
    gamma : float
        Multiplicative decay factor.

    Notes
    -----
    After ``k`` full decay steps the effective learning rate is:

    .. math::

        \eta_k = \eta_0 \cdot \gamma^{\lfloor t / \text{step\_size} \rfloor}

    A common choice is ``step_size=30, gamma=0.1`` for image classification
    tasks trained for 90 epochs.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> model = nn.Linear(4, 2)
    >>> import lucid.optim as optim
    >>> optimizer = optim.SGD(model.parameters(), lr=0.1)
    >>> scheduler = optim.StepLR(optimizer, step_size=30, gamma=0.1)
    >>> for epoch in range(90):
    ...     optimizer.step()
    ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int,
        gamma: float = 0.1,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        """Initialise the StepLR.  See the class docstring for parameter semantics."""
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)

    @override
    def get_lr(self) -> list[float]:
        """Compute the learning rate for each parameter group at the current step.

        Returns
        -------
        list[float]
            One learning rate per param group, derived from the schedule formula
            documented in the class docstring.
        """
        if self.last_epoch == 0 or self.last_epoch % self.step_size != 0:
            return self._group_lrs()
        return [lr * self.gamma for lr in self._group_lrs()]

    @override
    def _get_closed_form_lr(self) -> list[float]:
        """The rates at ``last_epoch`` from ``base_lrs``."""
        return [
            base_lr * self.gamma ** (self.last_epoch // self.step_size) for base_lr in self.base_lrs
        ]


class ExponentialLR(_LRScheduler):
    r"""Decay the learning rate by a constant factor every epoch.

    The learning rate is multiplied by ``gamma`` at every epoch, producing
    an exponential decay over time:

    .. math::

        \eta_t = \eta_0 \cdot \gamma^{t}

    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer.
    gamma : float
        Multiplicative factor applied each epoch.  Values in ``(0, 1)``
        produce decay; values greater than ``1`` produce growth (rarely
        useful).
    last_epoch : int, optional
        The index of the last epoch (default: ``-1``).
    verbose : bool, optional
        Print the updated LR after each step if ``True`` (default: ``False``).

    Attributes
    ----------
    gamma : float
        Multiplicative factor applied each epoch.

    Notes
    -----
    ``ExponentialLR`` applies the decay at every epoch without any
    plateau detection.  For a coarser schedule that decays only at
    certain milestones, prefer :class:`StepLR` or :class:`MultiStepLR`.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> model = nn.Linear(4, 2)
    >>> import lucid.optim as optim
    >>> optimizer = optim.SGD(model.parameters(), lr=0.1)
    >>> scheduler = optim.ExponentialLR(optimizer, gamma=0.95)
    >>> for epoch in range(100):
    ...     optimizer.step()
    ...     scheduler.step()
    """

    def __init__(
        self, optimizer: Optimizer, gamma: float, last_epoch: int = -1, verbose: bool = False
    ) -> None:
        """Initialise the ExponentialLR.  See the class docstring for parameter semantics."""
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)

    @override
    def get_lr(self) -> list[float]:
        """Compute the learning rate for each parameter group at the current step.

        Returns
        -------
        list[float]
            One learning rate per param group, derived from the schedule formula
            documented in the class docstring.
        """
        if self._is_initial:
            return self._group_lrs()
        return [lr * self.gamma for lr in self._group_lrs()]

    @override
    def _get_closed_form_lr(self) -> list[float]:
        """The rates at ``last_epoch`` from ``base_lrs``."""
        return [base_lr * self.gamma**self.last_epoch for base_lr in self.base_lrs]


class MultiStepLR(_LRScheduler):
    r"""Decay the learning rate by a fixed factor at a list of epoch milestones.

    Unlike :class:`StepLR`, the decay epochs need not be evenly spaced.
    At each epoch listed in ``milestones`` the learning rate is multiplied
    by ``gamma``:

    .. math::

        \eta_t = \eta_0 \cdot \gamma^{\sum_{m \in \text{milestones}} \mathbf{1}[t \ge m]}

    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer.
    milestones : list of int
        Sorted list of epoch indices at which the learning rate is decayed.
        If unsorted, the list is sorted internally.
    gamma : float, optional
        Multiplicative factor applied at each milestone (default: ``0.1``).
    last_epoch : int, optional
        The index of the last epoch (default: ``-1``).
    verbose : bool, optional
        Print the updated LR after each step if ``True`` (default: ``False``).

    Attributes
    ----------
    milestones : list of int
        Sorted epoch indices where decay is applied.
    gamma : float
        Multiplicative decay factor.

    Notes
    -----
    Useful when natural training phases occur at irregular intervals (e.g.,
    decay at epochs 50, 100, and 150 in a 200-epoch run).

    Examples
    --------
    >>> import lucid.nn as nn
    >>> model = nn.Linear(4, 2)
    >>> import lucid.optim as optim
    >>> optimizer = optim.SGD(model.parameters(), lr=0.1)
    >>> scheduler = optim.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.1)
    >>> for epoch in range(200):
    ...     optimizer.step()
    ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        milestones: list[int],
        gamma: float = 0.1,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        """Initialise the MultiStepLR.  See the class docstring for parameter semantics."""
        self.milestones = sorted(milestones)
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)

    @override
    def get_lr(self) -> list[float]:
        """Compute the learning rate for each parameter group at the current step.

        Returns
        -------
        list[float]
            One learning rate per param group, derived from the schedule formula
            documented in the class docstring.
        """
        if self.last_epoch not in self.milestones:
            return self._group_lrs()
        # A milestone listed twice decays twice.
        decay = self.gamma ** self.milestones.count(self.last_epoch)
        return [lr * decay for lr in self._group_lrs()]

    @override
    def _get_closed_form_lr(self) -> list[float]:
        """The rates at ``last_epoch`` from ``base_lrs``."""
        passed = bisect_right(self.milestones, self.last_epoch)
        return [base_lr * self.gamma**passed for base_lr in self.base_lrs]


class CosineAnnealingLR(_LRScheduler):
    r"""Anneal the learning rate following a cosine curve over ``T_max`` epochs.

    The learning rate at epoch :math:`t` is:

    .. math::

        \eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})
                  \left(1 + \cos\!\left(\frac{\pi\, t}{T_{\max}}\right)\right)

    where :math:`\eta_{\max}` is the initial learning rate captured from the
    optimizer (``base_lr``) and :math:`\eta_{\min}` is the floor set by
    ``eta_min``.

    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer.
    T_max : int
        Half-period of the cosine cycle in epochs.  After ``T_max`` epochs
        the learning rate reaches ``eta_min``.
    eta_min : float, optional
        Minimum learning rate (default: ``0``).
    last_epoch : int, optional
        The index of the last epoch (default: ``-1``).
    verbose : bool, optional
        Print the updated LR after each step if ``True`` (default: ``False``).

    Attributes
    ----------
    T_max : int
        Decay period in epochs.
    eta_min : float
        Lower bound on the learning rate.

    Notes
    -----
    Cosine annealing produces a smooth, monotonically decreasing schedule
    that starts fast and slows near the minimum.  It is widely used for
    training deep networks and pairs naturally with warm-restarts
    (see :class:`CosineAnnealingWarmRestarts`).

    Each step scales a group's distance from ``eta_min`` by the ratio of
    consecutive cosine values.  For a group present from the start that is
    the formula above; it is also what composes with another schedule and
    carries a group added later from the rate it was added with.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> model = nn.Linear(4, 2)
    >>> import lucid.optim as optim
    >>> optimizer = optim.SGD(model.parameters(), lr=0.1)
    >>> scheduler = optim.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
    >>> for epoch in range(100):
    ...     optimizer.step()
    ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        eta_min: float = 0,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        """Initialise the CosineAnnealingLR.  See the class docstring for parameter semantics."""
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch, verbose)

    @override
    def get_lr(self) -> list[float]:
        """Compute the learning rate for each parameter group at the current step.

        Returns
        -------
        list[float]
            One learning rate per param group, derived from the schedule formula
            documented in the class docstring.
        """
        t = self.last_epoch
        if self._is_initial:
            return self._group_lrs()
        if (t - 1 - self.T_max) % (2 * self.T_max) == 0:
            # The step off the trough, where the ratio below divides by
            # zero: the rise is a fixed share of the base, so only the
            # groups the schedule was built over have one.
            rise = (1 - math.cos(math.pi / self.T_max)) / 2
            return [
                lr + (base_lr - self.eta_min) * rise
                for base_lr, lr in zip(self.base_lrs, self._group_lrs(), strict=False)
            ]
        ratio = (1 + math.cos(math.pi * t / self.T_max)) / (
            1 + math.cos(math.pi * (t - 1) / self.T_max)
        )
        return [self.eta_min + (lr - self.eta_min) * ratio for lr in self._group_lrs()]

    @override
    def _get_closed_form_lr(self) -> list[float]:
        """The rates at ``last_epoch`` from ``base_lrs``."""
        return [
            self.eta_min
            + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
            for base_lr in self.base_lrs
        ]


class LambdaLR(_CallerRated, _LRScheduler):
    r"""Set the learning rate using a user-supplied multiplicative factor function.

    At each epoch :math:`t`, the learning rate for param group :math:`i` is:

    .. math::

        \eta_t^{(i)} = \eta_0^{(i)} \cdot \lambda_i(t)

    where :math:`\eta_0^{(i)}` is the initial learning rate of group
    :math:`i` and :math:`\lambda_i` is the corresponding callable.

    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer.
    lr_lambda : callable or list of callable
        A function ``f(epoch: int) -> float`` returning the multiplicative
        factor for the learning rate.  If a single callable is given it is
        applied to every param group; if a list is given it must have the
        same length as ``optimizer.param_groups``.
    last_epoch : int, optional
        The index of the last epoch (default: ``-1``).
    verbose : bool, optional
        Print the updated LR after each step if ``True`` (default: ``False``).

    Attributes
    ----------
    lr_lambdas : list of callable
        One factor function per param group.

    Notes
    -----
    ``LambdaLR`` provides maximum flexibility: any monotone or cyclic
    schedule can be encoded as a Python function.  For a linearly increasing
    warmup followed by constant LR:

    .. code-block:: python

        warmup = 5
        fn = lambda t: min(1.0, t / warmup)
        scheduler = optim.LambdaLR(optimizer, lr_lambda=fn)

    Examples
    --------
    >>> import lucid.nn as nn
    >>> model = nn.Linear(4, 2)
    >>> import lucid.optim as optim
    >>> optimizer = optim.Adam(model.parameters(), lr=1e-3)
    >>> scheduler = optim.LambdaLR(optimizer, lr_lambda=lambda t: 0.95 ** t)
    >>> for epoch in range(50):
    ...     optimizer.step()
    ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        lr_lambda: Callable[[int], float] | list[Callable[[int], float]],
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        """Initialise the LambdaLR.  See the class docstring for parameter semantics."""
        if callable(lr_lambda):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            self.lr_lambdas = list(lr_lambda)
            if len(self.lr_lambdas) != len(optimizer.param_groups):
                raise ValueError(
                    f"LambdaLR: expected {len(optimizer.param_groups)} lr_lambdas, "
                    f"one per param group, got {len(self.lr_lambdas)}"
                )
        super().__init__(optimizer, last_epoch, verbose)

    @override
    def get_lr(self) -> list[float]:
        """Compute the learning rate for each parameter group at the current step.

        Returns
        -------
        list[float]
            One learning rate per param group, derived from the schedule formula
            documented in the class docstring.
        """
        return [
            base_lr * fn(self.last_epoch)
            for base_lr, fn in zip(self.base_lrs, self.lr_lambdas, strict=True)
        ]


class CyclicLR(_LRScheduler, _CyclicLike):
    r"""Cycle the learning rate between ``base_lr`` and ``max_lr``.

    Implements the triangular, triangular2, and exp_range cyclic policies.
    Within each cycle of length :math:`2 \times \text{step\_size\_up}` the
    learning rate rises linearly from ``base_lr`` to ``max_lr`` and then
    falls back:

    .. math::

        \text{cycle} &= \left\lfloor 1 + \frac{t}{2 \cdot s} \right\rfloor \\
        x &= \left|\frac{t}{s} - 2 \cdot \text{cycle} + 1\right| \\
        \text{scale} &= \max(0,\; 1 - x) \\
        \eta_t &= \eta_{\min} + (\eta_{\max} - \eta_{\min}) \cdot \text{scale}

    where :math:`s = \text{step\_size\_up}`.

    For ``mode="triangular2"`` the amplitude halves each cycle:

    .. math::

        \text{scale} \mathrel{/}= 2^{\text{cycle}-1}

    For ``mode="exp_range"`` the amplitude decays exponentially each step:

    .. math::

        \text{scale} \mathrel{\times}= \gamma^{t}

    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer.
    base_lr : float
        Lower boundary of the learning rate cycle.
    max_lr : float
        Upper boundary of the learning rate cycle.
    step_size_up : int, optional
        Number of steps in the increasing half of the cycle (default: ``2000``).
    mode : str, optional
        One of ``"triangular"`` (constant amplitude), ``"triangular2"``
        (amplitude halves each cycle), or ``"exp_range"`` (amplitude
        decays by :math:`\gamma^t` each step).  Default: ``"triangular"``.
    gamma : float, optional
        Decay factor used only in ``"exp_range"`` mode (default: ``1.0``).
    last_epoch : int, optional
        The index of the last epoch (default: ``-1``).

    Attributes
    ----------
    base_lrs : list of float
        Lower bounds of the cycle per parameter group.
    max_lrs : list of float
        Upper bounds of the cycle per parameter group.
    step_size_up : int
        Half-cycle length in steps.
    mode : str
        Scaling policy name.
    gamma : float
        Decay factor for ``"exp_range"`` mode.

    Notes
    -----
    Cyclic learning rates can reduce the need for careful manual tuning by
    automatically exploring a range of rates.  Use ``step_size_up`` between
    2 and 10 times the number of iterations per epoch.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> model = nn.Linear(4, 2)
    >>> import lucid.optim as optim
    >>> optimizer = optim.SGD(model.parameters(), lr=0.01)
    >>> scheduler = optim.CyclicLR(
    ...     optimizer, base_lr=1e-4, max_lr=1e-2, step_size_up=500
    ... )
    """

    def __init__(
        self,
        optimizer: Optimizer,
        base_lr: float | list[float],
        max_lr: float | list[float],
        step_size_up: int = 2000,
        step_size_down: int | None = None,
        mode: str = "triangular",
        gamma: float = 1.0,
        scale_fn: Callable[[float], float] | None = None,
        scale_mode: str = "cycle",
        cycle_momentum: bool = True,
        base_momentum: float | list[float] = 0.8,
        max_momentum: float | list[float] = 0.9,
        last_epoch: int = -1,
    ) -> None:
        """Initialise the CyclicLR.  See the class docstring for parameter semantics."""
        cfg = (
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
        )
        base_lrs = _init_cyclic(self, optimizer, cfg)
        super().__init__(optimizer, last_epoch)
        self.base_lrs = base_lrs

    @override
    def get_lr(self) -> list[float]:
        """Compute the learning rate for each parameter group at the current step."""
        # Refused before the momentum of the groups it does rate is touched.
        if len(self.optimizer.param_groups) > len(self.base_lrs):
            raise self._unrated_group(len(self.base_lrs))
        return _cyclic_get_lr(self, math.floor(1 + self.last_epoch / self.total_size))


class NoamScheduler(_LRScheduler):
    r"""Noam learning rate schedule from the original Transformer paper.

    The learning rate increases linearly during a warmup phase and then
    decays proportionally to the inverse square root of the step number:

    .. math::

        \eta_t = d_{\text{model}}^{-0.5} \cdot
                  \min\!\left(t^{-0.5},\;
                                 t \cdot w^{-1.5}\right)

    where :math:`d_{\text{model}}` is the model dimensionality and
    :math:`w` is the number of warmup steps.

    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer.  The ``lr`` in each param group is set to the
        Noam value directly (the base learning rate is not used as a
        multiplicative factor).
    d_model : int
        Dimensionality of the model (e.g. 512 for the base Transformer).
        Larger models have smaller peak learning rates.
    warmup_steps : int
        Number of warmup steps during which the LR increases linearly.
        A typical value is 4000.
    last_epoch : int, optional
        The index of the last epoch (default: ``-1``).
    verbose : bool, optional
        Print the updated LR after each step if ``True`` (default: ``False``).

    Attributes
    ----------
    d_model : int
        Model dimension used in the scaling formula.
    warmup_steps : int
        Warmup period length.

    Notes
    -----
    The Noam schedule is designed to be called **once per training step**
    (i.e. per batch), not once per epoch.  The learning rate peaks at step
    :math:`t^* = w` and then decreases as :math:`t^{-0.5}`.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> model = nn.Linear(4, 2)
    >>> import lucid.optim as optim
    >>> optimizer = optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98))
    >>> scheduler = optim.NoamScheduler(optimizer, d_model=512, warmup_steps=4000)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        d_model: int,
        warmup_steps: int,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        """Initialise the NoamScheduler.  See the class docstring for parameter semantics."""
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch, verbose)

    @override
    def get_lr(self) -> list[float]:
        """Compute the learning rate for each parameter group at the current step.

        Returns
        -------
        list[float]
            One learning rate per param group, derived from the schedule formula
            documented in the class docstring.
        """
        step = max(1, self.last_epoch)
        scale = self.d_model ** (-0.5) * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
        return [scale] * len(self.optimizer.param_groups)


class MultiplicativeLR(_CallerRated, _LRScheduler):
    r"""Multiply the learning rate by a factor returned by a function each epoch.

    Unlike :class:`LambdaLR`, which computes the learning rate as
    ``base_lr * fn(t)``, ``MultiplicativeLR`` **accumulates** the factor:
    each epoch the current learning rate is multiplied by ``lr_lambda(t)``:

    .. math::

        \eta_t = \eta_{t-1} \cdot \lambda(t)

    This means the schedule depends on the entire history of ``lr_lambda``
    values, not only the current epoch index.

    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer.
    lr_lambda : callable
        A function ``f(epoch: int) -> float`` returning the multiplicative
        factor to apply at that epoch.
    last_epoch : int, optional
        The index of the last epoch (default: ``-1``).
    verbose : bool, optional
        Print the updated LR after each step if ``True`` (default: ``False``).

    Attributes
    ----------
    lr_lambda : callable
        The factor function applied each epoch.

    Notes
    -----
    At epoch 0 the learning rate is left unchanged (the factor is not
    applied on the very first step so that the initial LR from the
    optimizer is honored).

    Examples
    --------
    >>> import lucid.nn as nn
    >>> model = nn.Linear(4, 2)
    >>> import lucid.optim as optim
    >>> optimizer = optim.SGD(model.parameters(), lr=0.1)
    >>> scheduler = optim.MultiplicativeLR(optimizer, lr_lambda=lambda t: 0.95)
    >>> for epoch in range(50):
    ...     optimizer.step()
    ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        lr_lambda: Callable[[int], float],
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        """Initialise the MultiplicativeLR.  See the class docstring for parameter semantics."""
        self.lr_lambda = lr_lambda
        super().__init__(optimizer, last_epoch, verbose)

    @override
    def get_lr(self) -> list[float]:
        """Compute the learning rate for each parameter group at the current step.

        Returns
        -------
        list[float]
            One learning rate per param group, derived from the schedule formula
            documented in the class docstring.
        """
        if self._is_initial:
            return self._group_lrs()
        factor = self.lr_lambda(self.last_epoch)
        # One factor per group built over, as in the reference: a group added
        # later has none and the step is refused.
        return [lr * factor for lr in self._group_lrs()[: len(self.base_lrs)]]


class LinearLR(_LRScheduler):
    r"""Linearly interpolate the learning rate from a start factor to an end factor.

    The learning rate is scaled by a factor that changes linearly from
    ``start_factor`` to ``end_factor`` over ``total_iters`` steps:

    .. math::

        \text{factor}(t) = \text{start\_factor}
            + \frac{(\text{end\_factor} - \text{start\_factor})
                     \cdot \min(t,\, T)}{T}

    .. math::

        \eta_t = \eta_0 \cdot \text{factor}(t)

    where :math:`T = \text{total\_iters}`.

    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer.
    start_factor : float, optional
        The multiplier applied to the base LR at epoch 0, in ``(0, 1]``
        (default: ``1/3``).
    end_factor : float, optional
        The multiplier applied to the base LR at epoch ``total_iters``, in
        ``[0, 1]`` (default: ``1.0``).
    total_iters : int, optional
        Number of steps over which the interpolation runs (default: ``5``).
    last_epoch : int, optional
        The index of the last epoch (default: ``-1``).
    verbose : bool, optional
        Print the updated LR after each step if ``True`` (default: ``False``).

    Attributes
    ----------
    start_factor : float
        Initial LR multiplier.
    end_factor : float
        Final LR multiplier.
    total_iters : int
        Interpolation length in epochs.

    Notes
    -----
    ``LinearLR`` is commonly used for **linear warmup**: set
    ``start_factor`` to a small value (e.g. ``1/total_iters``) and
    ``end_factor=1.0`` so the learning rate ramps up to its nominal value
    over the first few epochs.

    Each step scales a group's current rate by the factor's growth since
    the step before, so a group added later is carried from the rate it was
    added with through the rest of the ramp.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> model = nn.Linear(4, 2)
    >>> import lucid.optim as optim
    >>> optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    >>> scheduler = optim.LinearLR(
    ...     optimizer, start_factor=1/3, end_factor=1.0, total_iters=5
    ... )
    >>> for epoch in range(50):
    ...     optimizer.step()
    ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        start_factor: float = 1.0 / 3,
        end_factor: float = 1.0,
        total_iters: int = 5,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        """Initialise the LinearLR.  See the class docstring for parameter semantics."""
        # Each step scales the current rate by the factor's growth since the
        # last one, which a start of zero has no ratio to.
        if not 0.0 < start_factor <= 1.0:
            raise ValueError(f"LinearLR: start_factor must be in (0, 1], got {start_factor}")
        if not 0.0 <= end_factor <= 1.0:
            raise ValueError(f"LinearLR: end_factor must be in [0, 1], got {end_factor}")
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch, verbose)

    @override
    def get_lr(self) -> list[float]:
        """Compute the learning rate for each parameter group at the current step.

        Returns
        -------
        list[float]
            One learning rate per param group, derived from the schedule formula
            documented in the class docstring.
        """
        t = self.last_epoch
        if t == 0:
            return [lr * self.start_factor for lr in self._group_lrs()]
        if self._is_initial or t > self.total_iters:
            return self._group_lrs()
        span = self.end_factor - self.start_factor
        growth = 1.0 + span / (self.total_iters * self.start_factor + (t - 1) * span)
        return [lr * growth for lr in self._group_lrs()]

    @override
    def _get_closed_form_lr(self) -> list[float]:
        """The rates at ``last_epoch`` from ``base_lrs``."""
        t = min(self.last_epoch, self.total_iters)
        factor = self.start_factor + (self.end_factor - self.start_factor) * t / self.total_iters
        return [base_lr * factor for base_lr in self.base_lrs]


class ConstantLR(_LRScheduler):
    r"""Hold the learning rate at a constant scaled value, then restore the base LR.

    For the first ``total_iters`` epochs the learning rate is scaled by
    ``factor``; after that it is restored to the original base learning rate:

    .. math::

        \eta_t =
        \begin{cases}
            \eta_0 \cdot \text{factor} & t < T \\
            \eta_0                        & t \ge T
        \end{cases}

    where :math:`T = \text{total\_iters}`.

    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer.
    factor : float, optional
        Multiplicative factor applied to the base LR during the constant
        phase, in ``[0, 1]`` (default: ``1/3``).  A factor of ``0`` has
        nothing to scale back up, so the step that reaches ``total_iters``
        raises ``ValueError``.
    total_iters : int, optional
        Number of epochs to hold the scaled LR before restoring the base
        value (default: ``5``).
    last_epoch : int, optional
        The index of the last epoch (default: ``-1``).
    verbose : bool, optional
        Print the updated LR after each step if ``True`` (default: ``False``).

    Attributes
    ----------
    factor : float
        LR scaling factor during the constant phase.
    total_iters : int
        Duration of the constant phase in epochs.

    Notes
    -----
    ``ConstantLR`` is useful as the first stage in a :class:`SequentialLR`
    chain, e.g. to hold a reduced LR during an initial burn-in period
    before switching to a main decay schedule.

    The rate is scaled, not set: by ``factor`` at epoch 0 and by
    ``1 / factor`` at ``total_iters``.  A group added in between was never
    scaled down, and is scaled up from the rate it was added with.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> model = nn.Linear(4, 2)
    >>> import lucid.optim as optim
    >>> optimizer = optim.SGD(model.parameters(), lr=0.1)
    >>> scheduler = optim.ConstantLR(optimizer, factor=1/3, total_iters=5)
    >>> for epoch in range(50):
    ...     optimizer.step()
    ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        factor: float = 1.0 / 3,
        total_iters: int = 5,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        """Initialise the ConstantLR.  See the class docstring for parameter semantics."""
        if not 0.0 <= factor <= 1.0:
            raise ValueError(f"ConstantLR: factor must be in [0, 1], got {factor}")
        self.factor = factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch, verbose)

    @override
    def get_lr(self) -> list[float]:
        """Compute the learning rate for each parameter group at the current step.

        Returns
        -------
        list[float]
            One learning rate per param group, derived from the schedule formula
            documented in the class docstring.
        """
        if self.last_epoch == 0:
            return [lr * self.factor for lr in self._group_lrs()]
        if self.last_epoch != self.total_iters:
            return self._group_lrs()
        if self.factor == 0.0:
            # Where the reference divides by zero: the step refuses instead.
            raise ValueError(
                "ConstantLR: factor=0 cannot be restored at total_iters; use a "
                "larger total_iters or SequentialLR"
            )
        return [lr * (1.0 / self.factor) for lr in self._group_lrs()]

    @override
    def _get_closed_form_lr(self) -> list[float]:
        """The rates at ``last_epoch`` from ``base_lrs``."""
        scale = 1.0 if self.last_epoch >= self.total_iters else self.factor
        return [base_lr * scale for base_lr in self.base_lrs]


class PolynomialLR(_LRScheduler):
    r"""Decay the learning rate using a polynomial function over a fixed number of steps.

    The learning rate follows a polynomial decay from the base LR down to
    ``eta_min`` over ``total_iters`` epochs:

    .. math::

        \eta_t =
        \begin{cases}
            (\eta_0 - \eta_{\min})
            \left(1 - \dfrac{t}{T}\right)^{p}
            + \eta_{\min}
            & t < T \\
            \eta_{\min} & t \ge T
        \end{cases}

    where :math:`T = \text{total\_iters}` and :math:`p = \text{power}`.

    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer.
    total_iters : int, optional
        Number of epochs over which the decay runs (default: ``5``).
    power : float, optional
        Exponent of the polynomial.  ``power=1`` gives linear decay;
        ``power=2`` gives quadratic decay (default: ``1.0``).
    last_epoch : int, optional
        The index of the last epoch (default: ``-1``).
    verbose : bool, optional
        Print the updated LR after each step if ``True`` (default: ``False``).
    eta_min : float, optional
        Minimum learning rate reached at ``total_iters`` (default: ``0.0``).

    Attributes
    ----------
    total_iters : int
        Decay period in epochs.
    power : float
        Polynomial exponent.
    eta_min : float
        Lower bound on the learning rate.

    Notes
    -----
    ``power=1`` (linear) is equivalent to :class:`LinearLR` with
    ``start_factor=1`` and ``end_factor=0``.  Higher powers produce a
    schedule that decays slowly at first and more steeply near the end.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> model = nn.Linear(4, 2)
    >>> import lucid.optim as optim
    >>> optimizer = optim.SGD(model.parameters(), lr=0.1)
    >>> scheduler = optim.PolynomialLR(
    ...     optimizer, total_iters=100, power=2.0, eta_min=1e-5
    ... )
    >>> for epoch in range(100):
    ...     optimizer.step()
    ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_iters: int = 5,
        power: float = 1.0,
        last_epoch: int = -1,
        verbose: bool = False,
        eta_min: float = 0.0,
    ) -> None:
        """Initialise the PolynomialLR.  See the class docstring for parameter semantics."""
        self.total_iters = total_iters
        self.power = power
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch, verbose)

    @override
    def get_lr(self) -> list[float]:
        """Compute the learning rate for each parameter group at the current step.

        Returns
        -------
        list[float]
            One learning rate per param group, derived from the schedule formula
            documented in the class docstring.
        """
        t = self.last_epoch
        T = self.total_iters
        if self._is_initial or t > T:
            return self._group_lrs()
        decay = ((1.0 - t / T) / (1.0 - (t - 1) / T)) ** self.power
        return [self.eta_min + (lr - self.eta_min) * decay for lr in self._group_lrs()]

    @override
    def _get_closed_form_lr(self) -> list[float]:
        """The rates at ``last_epoch`` from ``base_lrs``."""
        t = min(self.last_epoch, self.total_iters)
        return [
            (base_lr - self.eta_min) * ((1 - t / self.total_iters) ** self.power) + self.eta_min
            for base_lr in self.base_lrs
        ]


class CosineAnnealingWarmRestarts(_LRScheduler):
    r"""Cosine annealing with periodic warm restarts (SGDR).

    Implements the Stochastic Gradient Descent with Warm Restarts (SGDR)
    schedule.  Within each restart cycle of length :math:`T_i` epochs the
    learning rate follows a cosine curve from the base LR down to
    ``eta_min``, then restarts:

    .. math::

        \eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})
                  \left(1 + \cos\!\left(
                      \frac{\pi\, T_{\text{cur}}}{T_i}
                  \right)\right)

    where :math:`T_{\text{cur}}` is the step count within the current
    cycle and :math:`T_i` is the current cycle length.  After each full
    cycle the cycle length is multiplied by ``T_mult``:

    .. math::

        T_{i+1} = T_i \cdot T_{\text{mult}}

    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer.
    T_0 : int
        Length (in epochs) of the first restart cycle.
    T_mult : int, optional
        Factor by which the cycle length is multiplied after each restart
        (default: ``1``, i.e. all cycles have the same length).
    eta_min : float, optional
        Minimum learning rate at the bottom of each cosine curve
        (default: ``0.0``).
    last_epoch : int, optional
        The index of the last epoch (default: ``-1``).
    verbose : bool, optional
        Print the updated LR after each step if ``True`` (default: ``False``).

    Attributes
    ----------
    T_0 : int
        Initial cycle length.
    T_mult : int
        Cycle-length multiplier applied after each restart.
    eta_min : float
        Lower bound on the learning rate.

    Notes
    -----
    With ``T_mult=1`` every cycle has the same length ``T_0``.  With
    ``T_mult=2`` cycle lengths double after each restart: ``T_0``,
    ``2*T_0``, ``4*T_0``, …  Longer later cycles are useful because the
    model can refine a good basin found in earlier cycles.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> model = nn.Linear(4, 2)
    >>> import lucid.optim as optim
    >>> optimizer = optim.SGD(model.parameters(), lr=0.1)
    >>> scheduler = optim.CosineAnnealingWarmRestarts(
    ...     optimizer, T_0=10, T_mult=2, eta_min=1e-5
    ... )
    >>> for epoch in range(80):
    ...     optimizer.step()
    ...     scheduler.step()
    """

    # The reference's step for this schedule skips the step-order check.
    _checks_step_order = False

    def __init__(
        self,
        optimizer: Optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0.0,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        """Initialise the CosineAnnealingWarmRestarts.  See the class docstring for parameter semantics."""
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self._T_cur = last_epoch  # a resumed one is that far in, not at the start
        self._T_i = T_0
        super().__init__(optimizer, last_epoch, verbose)

    def _advance_cycle(self) -> None:
        """Move a step through the cycle — to its start, before step 0 — as the reference."""
        if self.last_epoch < 0:
            self._T_cur, self._T_i = 0, self.T_0
            return
        self._T_cur += 1
        if self._T_cur >= self._T_i:
            self._T_cur %= self._T_i
            self._T_i *= self.T_mult

    @override
    def _initial_step(self) -> None:
        """Write step 0's rates, at the cycle position ``last_epoch`` puts it."""
        self._advance_cycle()
        super()._initial_step()

    @override
    def step(self) -> None:
        """Advance the scheduler by one step and update the optimizer learning rates.

        Notes
        -----
        Should be called after the optimizer's ``.step()`` at the end of each
        epoch (or each iteration, depending on the schedule).
        """
        cycle = (self._T_cur, self._T_i)
        self._advance_cycle()
        try:
            super().step()
        except ValueError:
            # A refused step leaves the cycle where it was, too.
            self._T_cur, self._T_i = cycle
            raise

    @override
    def get_lr(self) -> list[float]:
        """Compute the learning rate for each parameter group at the current step.

        Returns
        -------
        list[float]
            One learning rate per param group, derived from the schedule formula
            documented in the class docstring.
        """
        return [
            self.eta_min
            + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self._T_cur / self._T_i)) / 2
            for base_lr in self.base_lrs
        ]


class OneCycleLR(_LRScheduler, _OneCycleLike):
    r"""1Cycle learning rate policy.

    Implements the 1cycle policy: the learning rate first rises from an
    initial value to ``max_lr`` over a warmup phase, then anneals down to
    a minimum value over the remaining steps.

    Three special learning rates are derived from ``max_lr``:

    .. math::

        \eta_{\text{init}} &= \frac{\eta_{\max}}{\text{div\_factor}} \\
        \eta_{\min}  &= \frac{\eta_{\text{init}}}{\text{final\_div\_factor}}

    The schedule has two phases:

    1. **Warmup** (first ``pct_start * total_steps`` steps): anneal from
       :math:`\eta_{\text{init}}` up to :math:`\eta_{\max}`.
    2. **Cooldown** (remaining steps): anneal from :math:`\eta_{\max}` down
       to :math:`\eta_{\min}`.

    Each phase uses either cosine or linear annealing depending on
    ``anneal_strategy``.

    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer.
    max_lr : float
        Peak learning rate reached at the end of the warmup phase.
    total_steps : int
        Total number of steps (batches) in the training run.
    pct_start : float, optional
        Fraction of ``total_steps`` devoted to the warmup phase
        (default: ``0.3``).
    anneal_strategy : str, optional
        Annealing function: ``"cos"`` for cosine annealing or
        ``"linear"`` for linear annealing (default: ``"cos"``).
    div_factor : float, optional
        Determines the initial LR as ``max_lr / div_factor``
        (default: ``25.0``).
    final_div_factor : float, optional
        Determines the minimum LR as ``initial_lr / final_div_factor``
        (default: ``1e4``).
    last_epoch : int, optional
        The index of the last epoch (default: ``-1``).

    Attributes
    ----------
    max_lr : float
        Peak learning rate.
    total_steps : int
        Total training steps.
    pct_start : float
        Warmup fraction.
    anneal_strategy : str
        ``"cos"`` or ``"linear"``.
    div_factor : float
        Initial LR divisor.
    final_div_factor : float
        Minimum LR divisor relative to initial LR.

    Notes
    -----
    The 1cycle policy should be called **once per training step** (per
    batch), not once per epoch.  It is designed for super-convergence
    and often allows training with much larger learning rates than
    standard schedules.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> model = nn.Linear(4, 2)
    >>> import lucid.optim as optim
    >>> optimizer = optim.SGD(model.parameters(), lr=0.1)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: float | list[float],
        total_steps: int | None = None,
        epochs: int | None = None,
        steps_per_epoch: int | None = None,
        pct_start: float = 0.3,
        anneal_strategy: str = "cos",
        cycle_momentum: bool = True,
        base_momentum: float | list[float] = 0.85,
        max_momentum: float | list[float] = 0.95,
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        three_phase: bool = False,
        last_epoch: int = -1,
    ) -> None:
        """Initialise the OneCycleLR.  See the class docstring for parameter semantics."""
        cfg = (
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
        )
        _init_one_cycle(self, optimizer, cfg)
        super().__init__(optimizer, last_epoch)

    @override
    def get_lr(self) -> list[float]:
        """Compute the learning rate for each parameter group at the current step."""
        # The phases read their bounds off each group, so a group added
        # later is scheduled if it carries them; checked before any
        # group's momentum is touched.
        keys = ("initial_lr", "max_lr", "min_lr")
        momentum = ("max_momentum", "base_momentum") if self.cycle_momentum else ()
        _require_group_keys("OneCycleLR", self.optimizer, keys + momentum)
        return _one_cycle_get_lr(self)


class SequentialLR(_CompositeSchedulerMixin):
    """Chain multiple schedulers and switch between them at epoch milestones.

    ``SequentialLR`` runs its list of schedulers one at a time, activating
    the next scheduler whenever the epoch counter crosses a milestone.
    Before the first milestone, ``schedulers[0]`` is active; between
    milestone ``k-1`` and ``k``, ``schedulers[k]`` is active.  At its
    milestone a scheduler starts over at epoch 0, from the base rates.  An
    LR scheduler does so by its closed form, which has no rate for a param
    group added before then: the switch raises ``ValueError``.  A
    ``ChainedScheduler`` starts its chain over as one that runs first does
    (a Lucid extension: the reference cannot restart a chain).

    Parameters
    ----------
    optimizer : Optimizer
        The optimizer shared by all schedulers.
    schedulers : list of _LRScheduler or ChainedScheduler
        Schedulers to activate in sequence, on ``optimizer``.  Must have
        ``len(schedulers) == len(milestones) + 1``.
    milestones : list of int
        Sorted list of epoch indices at which to switch to the next
        scheduler.
    last_epoch : int, optional
        The index of the last epoch (default: ``-1``).

    Attributes
    ----------
    optimizer : Optimizer
        The shared optimizer.
    schedulers : list of _LRScheduler or ChainedScheduler
        All schedulers in activation order.
    milestones : list of int
        Epoch indices marking scheduler transitions.
    last_epoch : int
        Number of :meth:`step` calls completed.

    Notes
    -----
    A common pattern is to combine a short warmup with a long cosine decay:

    .. code-block:: python

        warmup   = optim.LinearLR(optimizer, start_factor=0.1, total_iters=5)
        cosine   = optim.CosineAnnealingLR(optimizer, T_max=95)
        combined = optim.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[5],
        )

    Examples
    --------
    >>> import lucid.nn as nn
    >>> model = nn.Linear(4, 2)
    >>> import lucid.optim as optim
    >>> optimizer = optim.SGD(model.parameters(), lr=0.1)
    >>> s1 = optim.ConstantLR(optimizer, factor=0.1, total_iters=5)
    >>> s2 = optim.ExponentialLR(optimizer, gamma=0.9)
    >>> scheduler = optim.SequentialLR(optimizer, schedulers=[s1, s2], milestones=[5])
    >>> for epoch in range(50):
    ...     optimizer.step()
    ...     scheduler.step()
    """

    schedulers: list[_LRScheduler | ChainedScheduler]

    def __init__(
        self,
        optimizer: Optimizer,
        schedulers: Sequence[_LRScheduler | ChainedScheduler],
        milestones: list[int],
        last_epoch: int = -1,
    ) -> None:
        """Initialise the SequentialLR.  See the class docstring for parameter semantics."""
        _check_schedulers("SequentialLR", schedulers, optimizer, (_LRScheduler, ChainedScheduler))
        if len(milestones) != len(schedulers) - 1:
            raise ValueError(
                f"SequentialLR expects one milestone fewer than schedulers, got "
                f"{len(milestones)} milestones for {len(schedulers)} schedulers"
            )
        self.optimizer = optimizer
        self.schedulers = list(schedulers)
        self.milestones = milestones
        self.last_epoch = last_epoch + 1
        # Each child took its own step 0 as it was built, one on top of the
        # other.  Start from the base again, take those steps back, and give
        # step 0 to the child that runs first only.
        _reset_to_base(optimizer)
        self._undo_initial_step()
        schedulers[bisect_right(milestones, 0)]._initial_step()

    @override
    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Restore state produced by :meth:`state_dict`, from this or an older Lucid.

        An older Lucid's (it holds ``_idx``) ``last_epoch`` is one behind.
        """
        incoming = dict(state_dict)
        last = incoming.get("last_epoch")
        if incoming.pop("_idx", None) is not None and isinstance(last, int):
            incoming["last_epoch"] = last + 1
        super().load_state_dict(incoming)

    def step(self) -> None:
        """Advance the active scheduler by one epoch, switching if a milestone is reached.

        Increments the internal epoch counter, checks whether any
        milestone has been crossed, and delegates to the currently active
        scheduler's :meth:`~_LRScheduler.step`.

        Examples
        --------
        >>> import lucid.nn as nn
        >>> import lucid.optim as optim
        >>> model = nn.Linear(4, 2)
        >>> optimizer = optim.SGD(model.parameters(), lr=0.1)
        >>> scheduler = optim.SequentialLR(
        ...     optimizer,
        ...     schedulers=[optim.ConstantLR(optimizer, factor=0.1),
        ...                 optim.ExponentialLR(optimizer, gamma=0.9)],
        ...     milestones=[5],
        ... )
        >>> for epoch in range(100):
        ...     optimizer.step()
        ...     scheduler.step()
        """
        self.last_epoch += 1
        idx = bisect_right(self.milestones, self.last_epoch)
        sched = self.schedulers[idx]
        try:
            if idx > 0 and self.milestones[idx - 1] == self.last_epoch:
                # The child taking over starts its own schedule, at epoch 0
                # and from the base: stepping it instead switched a step
                # late and carried the outgoing child's rate into it.
                sched._restart()
            else:
                sched.step()
        except ValueError:
            self.last_epoch -= 1
            raise

    def get_last_lr(self) -> list[float]:
        """Return the last learning rates from the currently active scheduler.

        Returns
        -------
        list of float
            Current learning rate of each optimizer param group, as
            reported by the active scheduler.
        """
        active = self.schedulers[bisect_right(self.milestones, self.last_epoch)]
        return active.get_last_lr()


class ChainedScheduler(_CompositeSchedulerMixin):
    r"""Apply multiple schedulers simultaneously, compounding their LR changes.

    At every :meth:`step` call each scheduler in the list is stepped in
    order.  Because each scheduler writes its own learning rates back into
    the shared optimizer, the net effect is the **composition** of all
    individual schedule factors:

    .. math::

        \eta_t = \eta_0 \cdot \prod_{k} f_k(t)

    where :math:`f_k(t)` is the multiplicative change applied by scheduler
    :math:`k` at epoch :math:`t`.

    Parameters
    ----------
    schedulers : list of _LRScheduler, SequentialLR or ChainedScheduler
        Schedulers to step together, all on the same optimizer.  A
        ``ReduceLROnPlateau`` steps on a metric and is refused.

    Attributes
    ----------
    schedulers : list of _LRScheduler, SequentialLR or ChainedScheduler
        The chained schedulers stepped in order.

    Notes
    -----
    ``ChainedScheduler`` differs from :class:`SequentialLR` in that **all**
    schedulers are active at every step rather than one at a time.  This is
    useful when you want, for example, a warmup schedule and an exponential
    decay to both apply simultaneously.

    The product holds for chainable children — :class:`StepLR`,
    :class:`MultiStepLR`, :class:`ExponentialLR`, :class:`ConstantLR`,
    :class:`LinearLR`, :class:`CosineAnnealingLR`, :class:`PolynomialLR`,
    :class:`MultiplicativeLR` — which scale the rate they find.  A child that
    computes its rate from its base alone (:class:`LambdaLR`,
    :class:`CyclicLR`, :class:`OneCycleLR`,
    :class:`CosineAnnealingWarmRestarts`) overwrites what the children
    before it wrote.

    :meth:`get_last_lr` returns the composed rate, which is also what is
    stored in the optimizer.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> model = nn.Linear(4, 2)
    >>> import lucid.optim as optim
    >>> optimizer = optim.SGD(model.parameters(), lr=0.1)
    >>> warmup = optim.LinearLR(optimizer, start_factor=0.1, total_iters=5)
    >>> decay  = optim.ExponentialLR(optimizer, gamma=0.95)
    >>> scheduler = optim.ChainedScheduler([warmup, decay])
    >>> for epoch in range(50):
    ...     optimizer.step()
    ...     scheduler.step()
    """

    schedulers: list[_LRScheduler | SequentialLR | ChainedScheduler]

    def __init__(
        self,
        schedulers: Sequence[_LRScheduler | SequentialLR | ChainedScheduler],
        optimizer: Optimizer | None = None,
    ) -> None:
        """Initialise the ChainedScheduler.  See the class docstring for parameter semantics."""
        # Nothing to write: each child took its step 0 on the last one's rate.
        self.optimizer = _check_schedulers(
            "ChainedScheduler", schedulers, optimizer, (_LRScheduler, _CompositeSchedulerMixin)
        )
        self.schedulers = list(schedulers)

    def _initial_step(self) -> None:
        """Step 0 of a chain that runs first in a SequentialLR: each child steps once."""
        for sched in self.schedulers:
            sched.step()

    def _restart(self) -> None:
        """Start over at a SequentialLR milestone, as a chain that runs first does."""
        with _all_or_nothing(self):
            _reset_to_base(self.optimizer)
            self._initial_step()

    def step(self) -> None:
        """Step all chained schedulers simultaneously.

        Calls :meth:`~_LRScheduler.step` on every scheduler in order.
        Each scheduler scales the rate the one before it wrote, so the
        factors multiply — for every param group, including one added
        after the chain was built.

        Examples
        --------
        >>> import lucid.nn as nn
        >>> import lucid.optim as optim
        >>> model = nn.Linear(4, 2)
        >>> optimizer = optim.SGD(model.parameters(), lr=0.1)
        """
        with _all_or_nothing(self):
            for sched in self.schedulers:
                sched.step()

    def get_last_lr(self) -> list[float]:
        """Return the composed learning rates, as the optimizer holds them."""
        return [_group_float(g) for g in self.optimizer.param_groups]
