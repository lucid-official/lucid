"""
Additional optimizers: RMSprop, Adagrad, Adadelta, Adamax, RAdam, NAdam, ASGD, Rprop, SparseAdam.
"""

import math
import operator
from collections.abc import Iterable
from typing import Any, SupportsFloat, SupportsIndex, override

import lucid
from lucid._tensor.tensor import Tensor
from lucid._types import _OptimizerClosure
from lucid._C import engine as _C_engine
from lucid._dispatch import _wrap
from lucid.nn.parameter import Parameter
from lucid.optim._state_load import _follow
from lucid.optim.optimizer import Optimizer, _from_host, _group_params, _host_copy


class RMSprop(Optimizer):
    r"""Root Mean Square Propagation optimizer.

    RMSprop maintains a running average of the squared gradient for each
    parameter and divides the gradient by the square root of that average,
    which normalises the effective learning rate per parameter:

    .. math::

        v_t &= \alpha \, v_{t-1} + (1 - \alpha) \, g_t^2 \\
        \theta_t &= \theta_{t-1}
            - \frac{\eta}{\sqrt{v_t} + \epsilon} \, g_t

    With ``centered=True`` a running mean of the gradient :math:`\bar g`
    is kept as well, and the denominator uses the estimated variance in
    place of the raw second moment:

    .. math::

        \bar g_t &= \alpha \, \bar g_{t-1} + (1 - \alpha) \, g_t \\
        \tilde v_t &= v_t - \bar g_t^2

    With momentum an additional velocity buffer :math:`b` is maintained:

    .. math::

        b_t &= \mu \, b_{t-1} + \frac{g_t}{\sqrt{v_t} + \epsilon} \\
        \theta_t &= \theta_{t-1} - \eta \, b_t

    Parameters
    ----------
    params : iterable of Parameter or iterable of dict
        Parameters to optimise, or a list of parameter-group dicts.
    lr : float, optional
        Learning rate :math:`\eta` (default: ``1e-2``).
    alpha : float, optional
        Smoothing constant :math:`\alpha` for the running squared-gradient
        average (default: ``0.99``).
    eps : float, optional
        Term :math:`\epsilon` added to the denominator for numerical
        stability (default: ``1e-8``).
    weight_decay : float, optional
        L2 regularisation coefficient (default: ``0``).
    momentum : float, optional
        Momentum factor :math:`\mu` (default: ``0``).
    centered : bool, optional
        If ``True``, normalise by the estimated variance rather than the
        raw second moment (default: ``False``).

    Attributes
    ----------
    param_groups : list of dict
        Parameter groups with keys ``"params"``, ``"lr"``, ``"alpha"``,
        ``"eps"``, ``"weight_decay"``, ``"momentum"``, and ``"centered"``.
    defaults : dict
        Default hyperparameter values.

    Notes
    -----
    RMSprop was proposed as an unpublished improvement for non-stationary
    objectives and recurrent networks.  It works well with a moderate
    learning rate and is often used for reinforcement learning tasks.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> import lucid.optim as optim
    >>> model = nn.Linear(4, 2)
    >>> loss = model(lucid.randn(1, 4)).sum()
    >>> import lucid.optim as optim
    >>> optimizer = optim.RMSprop(model.parameters(), lr=1e-3, alpha=0.99)
    >>> optimizer.zero_grad()
    >>> loss.backward()
    >>> optimizer.step()
    """

    _HYPERPARAMS = ("lr", "alpha", "eps", "weight_decay", "momentum", "centered")
    _ENGINE_RULES = _C_engine.RMSprop

    def __init__(
        self,
        params: Iterable[Parameter] | Iterable[dict[str, object]],
        lr: float = 1e-2,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0,
        momentum: float = 0,
        centered: bool = False,
        capturable: bool = False,
        foreach: bool | None = None,
        maximize: bool = False,
        differentiable: bool = False,
    ) -> None:
        """Initialise the RMSprop.  See the class docstring for parameter semantics."""
        defaults: dict[str, object] = dict(
            lr=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=centered,
            capturable=capturable,
            foreach=foreach,
            maximize=maximize,
            differentiable=differentiable,
        )
        super().__init__(params, defaults)

    @override
    def _append_engine_optim(self, group: dict[str, object]) -> None:
        self._engines.append(self._build_engine(_C_engine.RMSprop, group))

    @override
    def step(self, closure: _OptimizerClosure = None) -> Tensor | None:
        """Perform a single RMSprop step."""
        return self._step_engines(closure)


class Adagrad(Optimizer):
    r"""Adaptive Gradient optimizer.

    Adagrad adapts the learning rate for each parameter based on the sum
    of all past squared gradients.  Parameters that receive large or
    frequent gradient updates get smaller effective learning rates:

    .. math::

        G_t &= G_{t-1} + g_t^2 \\
        \theta_t &= \theta_{t-1}
            - \frac{\eta_t}{\sqrt{G_t} + \epsilon} \, g_t

    where the effective learning rate decays with time as:

    .. math::

        \eta_t = \frac{\eta_0}{1 + (t - 1) \cdot \lambda}

    and :math:`\lambda` is the ``lr_decay`` parameter.

    Parameters
    ----------
    params : iterable of Parameter or iterable of dict
        Parameters to optimise, or a list of parameter-group dicts.
    lr : float, optional
        Initial learning rate :math:`\eta_0` (default: ``1e-2``).
    lr_decay : float, optional
        Learning-rate decay applied to the effective LR at each step
        :math:`\lambda` (default: ``0``).
    weight_decay : float, optional
        L2 regularisation coefficient (default: ``0``).
    initial_accumulator_value : float, optional
        Value every entry of :math:`G` starts from (default: ``0``).
    eps : float, optional
        Term :math:`\epsilon` added to the denominator for numerical
        stability (default: ``1e-10``).

    Attributes
    ----------
    param_groups : list of dict
        Parameter groups with keys ``"params"``, ``"lr"``, ``"lr_decay"``,
        ``"weight_decay"``, ``"initial_accumulator_value"``, and ``"eps"``.
    defaults : dict
        Default hyperparameter values.

    Notes
    -----
    Adagrad is particularly well-suited for sparse data (e.g. NLP with
    word embeddings) because infrequently updated parameters retain a
    larger effective learning rate.  The main drawback is that the
    accumulated squared-gradient sum :math:`G_t` only grows, so the
    effective learning rate can become vanishingly small over long training
    runs.  ``initial_accumulator_value`` precedes ``eps`` in the constructor
    signature, matching the reference framework.

    Examples
    --------
    The sparse-feature case it suits — an embedding whose rows are hit
    unevenly:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> import lucid.optim as optim
    >>> table = nn.Embedding(10, 3)
    >>> optimizer = optim.Adagrad(table.parameters(), lr=1e-2)
    >>> optimizer.zero_grad()
    >>> (table(lucid.tensor([1, 4, 4, 7])) ** 2).sum().backward()
    >>> optimizer.step()
    """

    _HYPERPARAMS = ("lr", "lr_decay", "weight_decay", "initial_accumulator_value", "eps")
    _ENGINE_RULES = _C_engine.Adagrad

    def __init__(
        self,
        params: Iterable[Parameter] | Iterable[dict[str, object]],
        lr: float = 1e-2,
        lr_decay: float = 0,
        weight_decay: float = 0,
        initial_accumulator_value: float = 0,
        eps: float = 1e-10,
        foreach: bool | None = None,
        *,
        maximize: bool = False,
        differentiable: bool = False,
        fused: bool | None = None,
    ) -> None:
        """Initialise the Adagrad.  See the class docstring for parameter semantics."""
        defaults: dict[str, object] = dict(
            lr=lr,
            lr_decay=lr_decay,
            weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value,
            eps=eps,
            foreach=foreach,
            maximize=maximize,
            differentiable=differentiable,
            fused=fused,
        )
        super().__init__(params, defaults)

    @override
    def _append_engine_optim(self, group: dict[str, object]) -> None:
        self._engines.append(self._build_engine(_C_engine.Adagrad, group))

    @override
    def step(self, closure: _OptimizerClosure = None) -> Tensor | None:
        """Perform a single Adagrad step."""
        return self._step_engines(closure)


class Adadelta(Optimizer):
    r"""Adadelta optimizer — an adaptive learning rate method with no global LR.

    Adadelta addresses Adagrad's aggressive, monotonically decreasing learning
    rate by limiting the accumulated past gradients to a fixed-size window via
    exponential moving averages.  No global learning rate is required in the
    canonical form:

    .. math::

        E[g^2]_t &= \rho \, E[g^2]_{t-1} + (1 - \rho) \, g_t^2 \\
        \Delta\theta_t &=
            -\frac{\sqrt{E[\Delta\theta^2]_{t-1} + \epsilon}}
                   {\sqrt{E[g^2]_t + \epsilon}} \, g_t \\
        E[\Delta\theta^2]_t &= \rho \, E[\Delta\theta^2]_{t-1}
            + (1 - \rho) \, \Delta\theta_t^2 \\
        \theta_t &= \theta_{t-1} + \eta \, \Delta\theta_t

    where :math:`\eta` is ``lr`` (defaults to ``1.0`` in the original
    formulation) and :math:`\rho` controls the decay window size.

    Parameters
    ----------
    params : iterable of Parameter or iterable of dict
        Parameters to optimise, or a list of parameter-group dicts.
    lr : float, optional
        Scaling factor :math:`\eta` applied to the update (default: ``1.0``).
    rho : float, optional
        Coefficient for the running averages of squared gradients and
        squared updates :math:`\rho` (default: ``0.9``).
    eps : float, optional
        Term :math:`\epsilon` added to the denominator for numerical
        stability (default: ``1e-6``).
    weight_decay : float, optional
        L2 regularisation coefficient (default: ``0``).

    Attributes
    ----------
    param_groups : list of dict
        Parameter groups with keys ``"params"``, ``"lr"``, ``"rho"``,
        ``"eps"``, and ``"weight_decay"``.
    defaults : dict
        Default hyperparameter values.

    Notes
    -----
    Because Adadelta automatically adapts its learning rate based on a
    window of recent gradient magnitudes, it is relatively robust to
    hyperparameter choices and does not require manual LR tuning.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> import lucid.optim as optim
    >>> model = nn.Linear(4, 2)
    >>> loss = model(lucid.randn(1, 4)).sum()
    >>> import lucid.optim as optim
    >>> optimizer = optim.Adadelta(model.parameters(), rho=0.9, eps=1e-6)
    >>> optimizer.zero_grad()
    >>> loss.backward()
    >>> optimizer.step()
    """

    _HYPERPARAMS = ("lr", "rho", "eps", "weight_decay")
    _ENGINE_RULES = _C_engine.Adadelta

    def __init__(
        self,
        params: Iterable[Parameter] | Iterable[dict[str, object]],
        lr: float = 1.0,
        rho: float = 0.9,
        eps: float = 1e-6,
        weight_decay: float = 0,
        foreach: bool | None = None,
        *,
        capturable: bool = False,
        maximize: bool = False,
        differentiable: bool = False,
    ) -> None:
        """Initialise the Adadelta.  See the class docstring for parameter semantics."""
        defaults: dict[str, object] = dict(
            lr=lr,
            rho=rho,
            eps=eps,
            weight_decay=weight_decay,
            foreach=foreach,
            capturable=capturable,
            maximize=maximize,
            differentiable=differentiable,
        )
        super().__init__(params, defaults)

    @override
    def _append_engine_optim(self, group: dict[str, object]) -> None:
        self._engines.append(self._build_engine(_C_engine.Adadelta, group))

    @override
    def step(self, closure: _OptimizerClosure = None) -> Tensor | None:
        """Perform a single Adadelta step."""
        return self._step_engines(closure)


class Adamax(Optimizer):
    r"""Adamax optimizer — a variant of Adam based on the infinity norm.

    Adamax generalises Adam by using the :math:`\ell_\infty` norm instead
    of the :math:`\ell_2` norm for the second-moment estimate.  The update
    rule replaces :math:`v_t` with the element-wise maximum of past
    absolute gradients scaled by :math:`\beta_2`:

    .. math::

        m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
        u_t &= \max\!\left(\beta_2 \, u_{t-1},\; |g_t| + \epsilon\right) \\
        \theta_t &= \theta_{t-1}
            - \frac{\eta}{(1 - \beta_1^t)} \cdot \frac{m_t}{u_t}

    Because :math:`u_t` is bounded by :math:`\max_k \beta_2^k (|g_{t-k}| + \epsilon)`,
    the effective step size is naturally bounded.

    Parameters
    ----------
    params : iterable of Parameter or iterable of dict
        Parameters to optimise, or a list of parameter-group dicts.
    lr : float, optional
        Learning rate :math:`\eta` (default: ``2e-3``).
    betas : tuple of float, optional
        Coefficients :math:`(\beta_1, \beta_2)` for the first-moment
        estimate and the :math:`\ell_\infty` norm decay
        (default: ``(0.9, 0.999)``).
    eps : float, optional
        Term :math:`\epsilon` added to the infinity-norm update for numerical
        stability (default: ``1e-8``).
    weight_decay : float, optional
        L2 regularisation coefficient (default: ``0``).

    Attributes
    ----------
    param_groups : list of dict
        Parameter groups with keys ``"params"``, ``"lr"``, ``"beta1"``,
        ``"beta2"``, ``"eps"``, and ``"weight_decay"``.
    defaults : dict
        Default hyperparameter values.

    Notes
    -----
    Adamax can be more stable than Adam on problems where gradients are
    sparse or have large outliers, because the infinity norm is less
    sensitive to large individual gradient magnitudes than the L2 norm.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> import lucid.optim as optim
    >>> model = nn.Linear(4, 2)
    >>> loss = model(lucid.randn(1, 4)).sum()
    >>> import lucid.optim as optim
    >>> optimizer = optim.Adamax(model.parameters(), lr=2e-3)
    >>> optimizer.zero_grad()
    >>> loss.backward()
    >>> optimizer.step()
    """

    _HYPERPARAMS = ("lr", "beta1", "beta2", "eps", "weight_decay")
    _PAIR_KEYS = {"betas": ("beta1", "beta2")}
    _ENGINE_RULES = _C_engine.Adamax

    def __init__(
        self,
        params: Iterable[Parameter] | Iterable[dict[str, object]],
        lr: float = 2e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        foreach: bool | None = None,
        *,
        maximize: bool = False,
        differentiable: bool = False,
        capturable: bool = False,
    ) -> None:
        """Initialise the Adamax.  See the class docstring for parameter semantics."""
        self._init_adam_family(
            params, (lr, betas, eps, weight_decay), (foreach, maximize, capturable, differentiable)
        )

    @override
    def _append_engine_optim(self, group: dict[str, object]) -> None:
        self._engines.append(self._build_engine(_C_engine.Adamax, group))

    @override
    def step(self, closure: _OptimizerClosure = None) -> Tensor | None:
        """Perform a single Adamax step."""
        return self._step_engines(closure)


class RAdam(Optimizer):
    r"""Rectified Adam optimizer with variance-adapted step size.

    RAdam addresses the large variance in Adam's effective learning rate
    during the early training steps (when the moving averages are poorly
    initialised) by computing a rectification term that smoothly transitions
    between SGD and Adam.

    The maximum length of the approximated SMA is:

    .. math::

        \rho_\infty = \frac{2}{1 - \beta_2} - 1

    At step :math:`t` the current SMA length estimate is:

    .. math::

        \rho_t = \rho_\infty
            - \frac{2 t \, \beta_2^t}{1 - \beta_2^t}

    When :math:`\rho_t > 4` the variance is tractable and a rectified
    adaptive step is used:

    .. math::

        r_t &= \sqrt{
            \frac{(\rho_t - 4)(\rho_t - 2)\rho_\infty}
                 {(\rho_\infty - 4)(\rho_\infty - 2)\rho_t}} \\
        \theta_t &= \theta_{t-1}
            - \alpha \, r_t
              \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}

    Otherwise the update falls back to SGD with bias-corrected momentum.

    Parameters
    ----------
    params : iterable of Parameter or iterable of dict
        Parameters to optimise, or a list of parameter-group dicts.
    lr : float, optional
        Learning rate :math:`\alpha` (default: ``1e-3``).
    betas : tuple of float, optional
        Coefficients :math:`(\beta_1, \beta_2)` for the first- and
        second-moment estimates (default: ``(0.9, 0.999)``).
    eps : float, optional
        Term :math:`\epsilon` for numerical stability (default: ``1e-8``).
    weight_decay : float, optional
        L2 regularisation coefficient (default: ``0``).

    Attributes
    ----------
    param_groups : list of dict
        Parameter groups with keys ``"params"``, ``"lr"``, ``"betas"``,
        ``"beta1"``, ``"beta2"``, ``"eps"``, and ``"weight_decay"``.
    defaults : dict
        Default hyperparameter values.

    Notes
    -----
    RAdam removes the need for a warmup schedule by automatically
    stabilising the adaptive learning rate in the early stages of training.
    It is a drop-in replacement for Adam that is less sensitive to the
    choice of learning rate.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> import lucid.optim as optim
    >>> model = nn.Linear(4, 2)
    >>> loss = model(lucid.randn(1, 4)).sum()
    >>> import lucid.optim as optim
    >>> optimizer = optim.RAdam(model.parameters(), lr=1e-3)
    >>> optimizer.zero_grad()
    >>> loss.backward()
    >>> optimizer.step()
    """

    _HYPERPARAMS = ("lr", "beta1", "beta2", "eps", "weight_decay")
    _PAIR_KEYS = {"betas": ("beta1", "beta2")}
    _ENGINE_RULES = _C_engine.RAdam
    _PYTHON_DECOUPLED_WD = True

    def __init__(
        self,
        params: Iterable[Parameter] | Iterable[dict[str, object]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        decoupled_weight_decay: bool = False,
        *,
        foreach: bool | None = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
    ) -> None:
        """Initialise the RAdam.  See the class docstring for parameter semantics."""
        self._init_adam_family(
            params,
            (lr, betas, eps, weight_decay),
            (foreach, maximize, capturable, differentiable),
            decoupled_weight_decay=decoupled_weight_decay,
        )

    @override
    def _append_engine_optim(self, group: dict[str, object]) -> None:
        self._engines.append(self._build_engine(_C_engine.RAdam, group))

    @override
    def step(self, closure: _OptimizerClosure = None) -> Tensor | None:
        """Perform a single RAdam step."""
        return self._step_engines(closure)


class NAdam(Optimizer):
    r"""Nesterov-accelerated Adaptive Moment Estimation optimizer.

    NAdam incorporates Nesterov momentum into Adam by replacing the
    standard first-moment estimate :math:`\hat{m}_t` in the denominator
    with a one-step lookahead estimate.  The update rule is:

    .. math::

        m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
        v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
        \hat{m}_t^{\text{Nesterov}}
            &= \frac{\beta_1 m_t}{1 - \beta_1^{t+1}}
             + \frac{(1 - \beta_1) g_t}{1 - \beta_1^t} \\
        \theta_t &= \theta_{t-1}
            - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon}
              \hat{m}_t^{\text{Nesterov}}

    Parameters
    ----------
    params : iterable of Parameter or iterable of dict
        Parameters to optimise, or a list of parameter-group dicts.
    lr : float, optional
        Learning rate :math:`\alpha` (default: ``2e-3``).
    betas : tuple of float, optional
        Coefficients :math:`(\beta_1, \beta_2)` for the first- and
        second-moment estimates (default: ``(0.9, 0.999)``).
    eps : float, optional
        Term :math:`\epsilon` for numerical stability (default: ``1e-8``).
    weight_decay : float, optional
        L2 regularisation coefficient (default: ``0``).
    momentum_decay : float, optional
        Momentum schedule decay factor :math:`\psi` (default: ``4e-3``).
    decoupled_weight_decay : bool, optional
        Whether to decouple the weight decay as in AdamW (default: ``False``).

    Attributes
    ----------
    param_groups : list of dict
        Parameter groups with keys ``"params"``, ``"lr"``, ``"betas"``,
        ``"beta1"``, ``"beta2"``, ``"eps"``, ``"weight_decay"``,
        ``"momentum_decay"``, and ``"decoupled_weight_decay"``.
    defaults : dict
        Default hyperparameter values.

    Notes
    -----
    NAdam often converges faster than vanilla Adam because the Nesterov
    lookahead provides a more accurate gradient direction.  It is
    particularly effective on recurrent networks and tasks with noisy
    gradients.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> import lucid.optim as optim
    >>> model = nn.Linear(4, 2)
    >>> loss = model(lucid.randn(1, 4)).sum()
    >>> import lucid.optim as optim
    >>> optimizer = optim.NAdam(model.parameters(), lr=2e-3)
    >>> optimizer.zero_grad()
    >>> loss.backward()
    >>> optimizer.step()
    """

    _HYPERPARAMS = (
        "lr",
        "beta1",
        "beta2",
        "eps",
        "weight_decay",
        "momentum_decay",
        "decoupled_weight_decay",
    )
    _PAIR_KEYS = {"betas": ("beta1", "beta2")}
    _ENGINE_RULES = _C_engine.NAdam
    _PYTHON_DECOUPLED_WD = True

    def __init__(
        self,
        params: Iterable[Parameter] | Iterable[dict[str, object]],
        lr: float = 2e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        momentum_decay: float = 4e-3,
        decoupled_weight_decay: bool = False,
        *,
        foreach: bool | None = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
    ) -> None:
        """Initialise the NAdam.  See the class docstring for parameter semantics."""
        self._init_adam_family(
            params,
            (lr, betas, eps, weight_decay),
            (foreach, maximize, capturable, differentiable),
            momentum_decay=momentum_decay,
            decoupled_weight_decay=decoupled_weight_decay,
        )

    @override
    @classmethod
    def _check_hyperparams(cls, values: dict[str, Any]) -> None:
        _C_engine.NAdam.check_hyperparams(
            {k: values[k] for k in ("lr", "beta1", "beta2", "eps", "weight_decay")}
        )
        mom_decay = values["momentum_decay"]
        if isinstance(mom_decay, bool) or not isinstance(mom_decay, (SupportsFloat, SupportsIndex)):
            raise TypeError(
                f"NAdam: momentum_decay must be a number, got {type(mom_decay).__name__}"
            )
        if not float(mom_decay) >= 0.0:
            raise _C_engine.InvalidArgument("NAdam: momentum_decay must be >= 0")
        dwd = values["decoupled_weight_decay"]
        if not isinstance(dwd, bool):
            raise TypeError(
                f"NAdam: decoupled_weight_decay must be a bool, got {type(dwd).__name__}"
            )

    @override
    def _append_engine_optim(self, group: dict[str, object]) -> None:
        hp = self._hyperparam_dict(self._hyperparams_of(group))
        wd = 0.0 if hp["decoupled_weight_decay"] else float(hp["weight_decay"])
        self._engines.append(
            _C_engine.NAdam(
                [p._impl for p in _group_params(group)],
                lr=float(hp["lr"]),
                beta1=float(hp["beta1"]),
                beta2=float(hp["beta2"]),
                eps=float(hp["eps"]),
                weight_decay=wd,
                momentum_decay=float(hp["momentum_decay"]),
            )
        )

    @override
    def _apply_hyperparams(self, idx: int, values: dict[str, Any]) -> None:
        self._check_hyperparams(values)
        engines = self._engines
        engine = engines[idx] if idx < len(engines) else None
        if engine is None:
            return
        old_mom_decay = self._group_hparams[idx][5]
        if values["momentum_decay"] != old_mom_decay:
            self.param_groups[idx]["momentum_decay"] = values["momentum_decay"]
            self._rebuild_engine_optim(idx, self.param_groups[idx])
            return
        wd = 0.0 if values["decoupled_weight_decay"] else float(values["weight_decay"])
        engine.set_hyperparams(
            {
                "lr": values["lr"],
                "beta1": values["beta1"],
                "beta2": values["beta2"],
                "eps": values["eps"],
                "weight_decay": wd,
            }
        )

    @override
    def step(self, closure: _OptimizerClosure = None) -> Tensor | None:
        """Perform a single NAdam step."""
        return self._step_engines(closure)


class ASGD(Optimizer):
    r"""Averaged Stochastic Gradient Descent optimizer.

    ASGD performs standard SGD updates but maintains a running average of
    the iterate sequence, which serves as the final parameter estimate.
    The averaging improves convergence in the presence of noise and is
    particularly effective near the end of training.

    Each step decays the parameter and takes an SGD step with the current
    step size :math:`\eta_t` (:math:`g_t` includes any ``weight_decay``
    term):

    .. math::

        \theta_t = \theta_{t-1} \bigl(1 - \lambda \, \eta_t\bigr)
            - \eta_t \, g_t

    then folds the result into the running average ``ax`` with weight
    :math:`\mu_t` (``ax`` simply copies :math:`\theta_t` while
    :math:`\mu_t = 1`):

    .. math::

        \bar{\theta}_t = \bar{\theta}_{t-1}
            + \mu_t \bigl(\theta_t - \bar{\theta}_{t-1}\bigr)

    and sets the step size and weight for the next step, starting from
    :math:`\eta_1 = \eta_0` and :math:`\mu_1 = 1`:

    .. math::

        \eta_{t+1} = \frac{\eta_0}{(1 + \lambda \, \eta_0 \, t)^\alpha},
        \qquad
        \mu_{t+1} = \frac{1}{\max(1, \; t - t_0)}

    Until :math:`t_0` the average therefore tracks the parameter; after it,
    the average is the mean of the iterates since :math:`t_0`.  The state
    of each parameter is ``step``, ``eta``, ``mu`` and ``ax``.

    Parameters
    ----------
    params : iterable of Parameter or iterable of dict
        Parameters to optimise, or a list of parameter-group dicts.
    lr : float, optional
        Initial learning rate :math:`\eta_0` (default: ``1e-2``).
    lambd : float, optional
        Decay term :math:`\lambda` (default: ``1e-4``).
    alpha : float, optional
        Power for LR decay :math:`\alpha` (default: ``0.75``).
    t0 : float, optional
        Step at which averaging begins (default: ``1e6``).
    weight_decay : float, optional
        L2 regularisation coefficient (default: ``0``).

    Attributes
    ----------
    param_groups : list of dict
        Parameter groups with keys ``"params"``, ``"lr"``, ``"lambd"``,
        ``"alpha"``, ``"t0"``, and ``"weight_decay"``.
    defaults : dict
        Default hyperparameter values.

    Notes
    -----
    ASGD can match or exceed the convergence rate of SGD with careful
    learning-rate tuning, and the averaging step provides additional
    regularisation.  The default ``t0=1e6`` delays averaging until very
    late in training; reduce it to start averaging earlier.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> import lucid.optim as optim
    >>> model = nn.Linear(4, 2)
    >>> loss = model(lucid.randn(1, 4)).sum()
    >>> import lucid.optim as optim
    >>> optimizer = optim.ASGD(model.parameters(), lr=1e-2, t0=1e5)
    >>> optimizer.zero_grad()
    >>> loss.backward()
    >>> optimizer.step()
    """

    _HYPERPARAMS = ("lr", "lambd", "alpha", "t0", "weight_decay")
    _ENGINE_RULES = _C_engine.ASGD

    def __init__(
        self,
        params: Iterable[Parameter] | Iterable[dict[str, object]],
        lr: float = 1e-2,
        lambd: float = 1e-4,
        alpha: float = 0.75,
        t0: float = 1e6,
        weight_decay: float = 0,
        foreach: bool | None = None,
        maximize: bool = False,
        differentiable: bool = False,
        capturable: bool = False,
    ) -> None:
        """Initialise the ASGD.  See the class docstring for parameter semantics."""
        defaults: dict[str, object] = dict(
            lr=lr,
            lambd=lambd,
            alpha=alpha,
            t0=t0,
            weight_decay=weight_decay,
            foreach=foreach,
            maximize=maximize,
            differentiable=differentiable,
            capturable=capturable,
        )
        super().__init__(params, defaults)

    @override
    def _append_engine_optim(self, group: dict[str, object]) -> None:
        self._engines.append(self._build_engine(_C_engine.ASGD, group))

    @override
    def step(self, closure: _OptimizerClosure = None) -> Tensor | None:
        """Perform a single ASGD step."""
        return self._step_engines(closure)


class Rprop(Optimizer):
    r"""Resilient Backpropagation optimizer.

    Rprop ignores gradient magnitudes and adapts a per-parameter step size
    based only on the **sign** of the gradient.  If the sign of the
    gradient does not change between steps the step size is increased; if
    the sign reverses the step size is decreased:

    .. math::

        \Delta_{i,t} =
        \begin{cases}
            \min(\Delta_{i,t-1} \cdot \eta^+,\; \Delta_{\max})
                & \text{if } g_{i,t} \cdot g_{i,t-1} > 0 \\
            \max(\Delta_{i,t-1} \cdot \eta^-,\; \Delta_{\min})
                & \text{if } g_{i,t} \cdot g_{i,t-1} < 0 \\
            \Delta_{i,t-1} & \text{otherwise}
        \end{cases}

    The parameter update is then:

    .. math::

        \theta_{i,t} = \theta_{i,t-1}
            - \operatorname{sign}(g_{i,t}) \cdot \Delta_{i,t}

    Parameters
    ----------
    params : iterable of Parameter or iterable of dict
        Parameters to optimise, or a list of parameter-group dicts.
    lr : float, optional
        Initial step size for each parameter (default: ``1e-2``).
    etas : tuple of float, optional
        Multiplicative decrease and increase factors
        :math:`(\eta^-, \eta^+)` (default: ``(0.5, 1.2)``).
    step_sizes : tuple of float, optional
        Minimum and maximum allowed step sizes
        :math:`(\Delta_{\min}, \Delta_{\max})` (default: ``(1e-6, 50)``).

    Attributes
    ----------
    param_groups : list of dict
        Parameter groups with keys ``"params"``, ``"lr"``, ``"eta_minus"``,
        ``"eta_plus"``, ``"step_min"``, and ``"step_max"``.
    defaults : dict
        Default hyperparameter values.

    Notes
    -----
    Rprop is particularly effective for full-batch training because the
    sign-based update is well-defined when gradients are deterministic.
    For stochastic mini-batch training RMSprop or Adam are generally
    preferred.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> import lucid.optim as optim
    >>> model = nn.Linear(4, 2)
    >>> loss = model(lucid.randn(1, 4)).sum()
    >>> import lucid.optim as optim
    >>> optimizer = optim.Rprop(model.parameters(), lr=1e-2, etas=(0.5, 1.2))
    >>> optimizer.zero_grad()
    >>> loss.backward()
    >>> optimizer.step()
    """

    _HYPERPARAMS = ("lr", "eta_minus", "eta_plus", "step_min", "step_max")
    _PAIR_KEYS = {"etas": ("eta_minus", "eta_plus"), "step_sizes": ("step_min", "step_max")}
    _ENGINE_RULES = _C_engine.Rprop

    def __init__(
        self,
        params: Iterable[Parameter] | Iterable[dict[str, object]],
        lr: float = 1e-2,
        etas: tuple[float, float] = (0.5, 1.2),
        step_sizes: tuple[float, float] = (1e-6, 50),
        *,
        capturable: bool = False,
        foreach: bool | None = None,
        maximize: bool = False,
        differentiable: bool = False,
    ) -> None:
        """Initialise the Rprop.  See the class docstring for parameter semantics."""
        super().__init__(
            params,
            {
                "lr": lr,
                "etas": (float(etas[0]), float(etas[1])),
                "step_sizes": (float(step_sizes[0]), float(step_sizes[1])),
                "capturable": capturable,
                "foreach": foreach,
                "maximize": maximize,
                "differentiable": differentiable,
            },
        )

    @override
    def _append_engine_optim(self, group: dict[str, object]) -> None:
        self._engines.append(self._build_engine(_C_engine.Rprop, group))

    @override
    def step(self, closure: _OptimizerClosure = None) -> Tensor | None:
        """Perform a single Rprop step."""
        return self._step_engines(closure)


def _moment_like(src: _C_engine.TensorImpl, param: Parameter) -> Tensor | None:
    """Moment ``src`` ready for ``param``'s buffer, or ``None`` when shaped for another.

    Follows the parameter through a conversion by the rule every
    optimizer's state follows (:func:`~lucid.optim._state_load._follow`).
    """
    if list(src.shape) != list(param._impl.shape):
        return None
    return _wrap(_follow(src, param._impl))


def _moment(entry: dict[str, object], key: str) -> Tensor:
    """State entry ``key`` of one parameter, which must be a tensor."""
    value = entry[key]
    if not isinstance(value, Tensor):
        raise TypeError(
            f"SparseAdam: state entry {key!r} must be a Tensor, got {type(value).__name__}"
        )
    return value


def _step_count(value: object) -> int:
    """A parameter's ``step`` entry as an int.

    Raises
    ------
    TypeError
        ``value`` is not an integer (a Python or a 0-d host-array integer).
    """
    if not isinstance(value, SupportsIndex):
        raise TypeError(f"SparseAdam: 'step' must be an integer, got {type(value).__name__}")
    return operator.index(value)


def _sparse_adam_rows(
    param: Parameter,
    grad: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    step: int,
    hparams: tuple[float, float, float, float],
    *,
    maximize: bool = False,
) -> tuple[Tensor, Tensor]:
    """One SparseAdam update of the rows of ``param`` its gradient touches.

    A row is a slice along dim 0 (the whole tensor for a 0-d parameter)
    holding at least one nonzero gradient value.  Only those rows of the
    moments and of the parameter change; the arithmetic is the
    reference's, ``m += (1 - beta1)(g - m)``.  Returns the new moments.
    """
    lr, beta1, beta2, eps = hparams
    shape = param.shape
    n = shape[0] if shape else 1
    g = grad.reshape(n, -1)
    rows = (g != 0).any(dim=1).nonzero().reshape(-1)
    if rows.shape[0] == 0:
        return exp_avg, exp_avg_sq
    g = g.index_select(0, rows)
    if maximize:
        g = -g
    m_all = exp_avg.reshape(n, -1)
    v_all = exp_avg_sq.reshape(n, -1)
    m = m_all.index_select(0, rows)
    v = v_all.index_select(0, rows)
    m = m + (g - m) * (1.0 - beta1)
    v = v + (g * g - v) * (1.0 - beta2)
    step_size = lr * math.sqrt(1.0 - beta2**step) / (1.0 - beta1**step)
    p_all = param.reshape(n, -1)
    p = p_all.index_select(0, rows) + (m / (v.sqrt() + eps)) * -step_size
    param._impl.copy_from(p_all.index_copy(0, rows, p).reshape(shape)._impl)
    return (
        m_all.index_copy(0, rows, m).reshape(shape),
        v_all.index_copy(0, rows, v).reshape(shape),
    )


class SparseAdam(Optimizer):
    r"""Lazy Adam for gradients that touch a few rows of a parameter.

    The variant of Adam meant for embedding tables: each step updates the
    moments and the values of only the rows the gradient touches — a row
    being a slice along the first dimension.  A row absent from a batch
    keeps its moments and its value; it does not keep moving on its old
    momentum the way it would under :class:`~lucid.optim.Adam`.

    For each touched row, with the parameter's step count :math:`t` (one
    count per parameter, advanced by every step in which it has a
    gradient):

    .. math::

        m &\leftarrow m + (1 - \beta_1)(g - m) \\
        v &\leftarrow v + (1 - \beta_2)(g^2 - v) \\
        \theta &\leftarrow \theta
            - \frac{\alpha \sqrt{1 - \beta_2^t}}{1 - \beta_1^t}
              \cdot \frac{m}{\sqrt{v} + \epsilon}

    Parameters
    ----------
    params : iterable of Parameter or iterable of dict
        Parameters to optimise, or a list of parameter-group dicts.
    lr : float, optional
        Learning rate :math:`\alpha` (default: ``1e-3``).
    betas : tuple of float, optional
        Coefficients :math:`(\beta_1, \beta_2)` for the first- and
        second-moment estimates (default: ``(0.9, 0.999)``).
    eps : float, optional
        Term :math:`\epsilon` for numerical stability (default: ``1e-8``).
    maximize : bool, optional
        If ``True``, maximise the objective with respect to ``params`` by
        negating the gradient on the touched rows before updating the
        moments and parameter values (default: ``False``).

    Attributes
    ----------
    param_groups : list of dict
        Parameter groups with keys ``"params"``, ``"lr"``, ``"betas"``,
        ``"eps"``, and ``"maximize"``.
    defaults : dict
        Default hyperparameter values.
    state : dict
        The live state of each parameter that has stepped, by flat
        parameter index: ``"step"`` (int), ``"exp_avg"`` and
        ``"exp_avg_sq"`` (tensors shaped like the parameter).  It is what
        :meth:`state_dict` saves and :meth:`load_state_dict` restores.

    Notes
    -----
    Lucid's gradients are dense — ``nn.Embedding(sparse=True)`` is not
    supported — so a row counts as touched when its gradient holds a
    nonzero value.  The reference framework takes only a sparse gradient
    and updates every row it lists, so the two differ for a row that was
    looked up but whose gradient is exactly zero: the reference decays its
    moments, Lucid leaves it alone.  Within a touched row every element is
    updated, zero or not, as the reference's embedding gradient lists whole
    rows.

    A parameter whose gradient is ``None`` is skipped and its step count
    does not advance; one whose gradient is all zero advances its count
    and changes nothing else, as in the reference.

    The moments follow their parameter through a conversion
    (``module.to()``, ``.double()``, ``.half()``) between steps.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> import lucid.optim as optim
    >>> embedding = nn.Embedding(4, 2)
    >>> optimizer = optim.SparseAdam(embedding.parameters(), lr=0.1)
    >>> before = embedding.weight.detach().clone()
    >>> embedding(lucid.tensor([0, 2])).sum().backward()
    >>> optimizer.step()
    >>> (embedding.weight != before).any(dim=1).tolist()
    [True, False, True, False]
    >>> optimizer.state[0]["step"]
    1
    """

    _HYPERPARAMS = ("lr", "betas", "eps", "maximize")
    _LIVE_STATE = True

    def __init__(
        self,
        params: Iterable[Parameter] | Iterable[dict[str, object]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        maximize: bool = False,
    ) -> None:
        """Initialise the SparseAdam.  See the class docstring for parameter semantics."""
        defaults: dict[str, object] = dict(lr=lr, betas=betas, eps=eps, maximize=maximize)
        super().__init__(params, defaults)

    @override
    @classmethod
    def _check_hyperparams(cls, values: dict[str, Any]) -> None:
        """Hold a group to the reference framework's SparseAdam rules.

        ``lr`` and ``eps`` must be positive and each beta in ``[0, 1)``.
        Written so a NaN fails, as the reference's ``not 0.0 < lr`` does.
        """
        if not 0.0 < values["lr"]:
            raise _C_engine.InvalidArgument("SparseAdam: lr must be > 0")
        if not 0.0 < values["eps"]:
            raise _C_engine.InvalidArgument("SparseAdam: eps must be > 0")
        beta1, beta2 = values["betas"]
        if not 0.0 <= beta1 < 1.0:
            raise _C_engine.InvalidArgument("SparseAdam: betas[0] must be in [0, 1)")
        if not 0.0 <= beta2 < 1.0:
            raise _C_engine.InvalidArgument("SparseAdam: betas[1] must be in [0, 1)")
        if not isinstance(values["maximize"], bool):
            raise TypeError(
                f"SparseAdam: maximize must be a bool, got {type(values['maximize']).__name__}"
            )

    def _flat_params(self) -> list[Parameter]:
        """Every parameter, in the flat order ``state`` is indexed by."""
        return [p for group in self.param_groups for p in _group_params(group)]

    def _state_for(self, idx: int, param: Parameter) -> dict[str, object]:
        """Parameter ``idx``'s state, its moments ready for ``param``'s buffer.

        A parameter with no state yet, or whose shape no longer matches its
        moments, starts from zero moments and a zero step count.
        """
        entry = self.state.get(idx)
        if entry:
            m = _moment_like(_moment(entry, "exp_avg")._impl, param)
            v = _moment_like(_moment(entry, "exp_avg_sq")._impl, param)
            if m is not None and v is not None:
                entry["exp_avg"] = m
                entry["exp_avg_sq"] = v
                return entry
        entry = {
            "step": 0,
            "exp_avg": lucid.zeros_like(param),
            "exp_avg_sq": lucid.zeros_like(param),
        }
        self.state[idx] = entry
        return entry

    @override
    def step(self, closure: _OptimizerClosure = None) -> Tensor | None:
        """Perform a single SparseAdam optimisation step.

        Each parameter with a gradient advances its step count and updates
        the rows its gradient touches (see the class notes); parameters
        whose gradient is ``None`` are skipped.

        Parameters
        ----------
        closure : callable, optional
            A closure that re-evaluates the model and returns the loss.
            If provided, it is called **before** the parameter updates.

        Returns
        -------
        Tensor or None
            The loss returned by ``closure``, or ``None`` if no closure
            was provided.

        Examples
        --------
        >>> import lucid
        >>> import lucid.nn as nn
        >>> import lucid.optim as optim
        >>> embedding = nn.Embedding(100, 8)
        >>> optimizer = optim.SparseAdam(embedding.parameters(), lr=1e-3)
        >>> optimizer.zero_grad()
        >>> loss = embedding(lucid.tensor([[3, 17, 3]])).sum()
        >>> loss.backward()
        >>> optimizer.step()
        """
        self._refuse_phantom_step()
        loss = closure() if closure is not None else None
        if closure is not None:
            self._refuse_phantom_step()
        # The groups are read below as they are; an edited one is checked
        # first.
        self._sync_hyperparams()
        idx = 0
        with lucid.no_grad():
            for group in self.param_groups:
                hp = self._hyperparam_dict(self._hyperparams_of(group))
                beta1, beta2 = hp["betas"]
                hparams = (hp["lr"], beta1, beta2, hp["eps"])
                maximize = bool(hp["maximize"])
                for p in _group_params(group):
                    grad = p.grad
                    if grad is not None:
                        self._step_param(idx, p, grad, hparams, maximize=maximize)
                    idx += 1
        return loss

    def _step_param(
        self,
        idx: int,
        param: Parameter,
        grad: Tensor,
        hparams: tuple[float, float, float, float],
        *,
        maximize: bool = False,
    ) -> None:
        """Advance parameter ``idx``'s step count and update its touched rows."""
        entry = self._state_for(idx, param)
        step = _step_count(entry["step"]) + 1
        entry["step"] = step
        if grad.numel() == 0:
            return
        entry["exp_avg"], entry["exp_avg_sq"] = _sparse_adam_rows(
            param,
            grad,
            _moment(entry, "exp_avg"),
            _moment(entry, "exp_avg_sq"),
            step,
            hparams,
            maximize=maximize,
        )

    # ── state_dict round-trip ────────────────────────────────────────────────
    #
    # The live state is ``self.state`` itself (``_LIVE_STATE``); a
    # checkpoint holds host-array copies under the reference's keys.

    @override
    def _save_state(self) -> dict[int, dict[str, object]]:
        """Each stepped parameter's step count and moments, as host arrays."""
        return {
            idx: {
                "step": entry["step"],
                "exp_avg": _host_copy(_moment(entry, "exp_avg")._impl),
                "exp_avg_sq": _host_copy(_moment(entry, "exp_avg_sq")._impl),
            }
            for idx, entry in self.state.items()
        }

    @override
    def _load_state(self, state: dict[int, dict[str, object]]) -> None:
        """Replace the live state with ``state``'s, every entry checked first.

        Each moment is placed on its parameter's device and cast to its
        dtype.  A parameter without an entry starts afresh, as in the
        reference.  Nothing is replaced when an entry is refused.

        Raises
        ------
        IndexError
            An entry for a parameter index the optimizer does not have.
        TypeError
            A ``step`` that is not an integer.
        lucid._C.engine.ShapeMismatch
            A moment not shaped like its parameter.
        """
        params = self._flat_params()
        loaded: dict[int, dict[str, object]] = {}
        for idx, entry in state.items():
            if not 0 <= idx < len(params):
                raise IndexError(
                    f"SparseAdam.load_state_dict: state for parameter {idx}, "
                    f"but the optimizer has {len(params)} parameters"
                )
            param = params[idx]
            restored: dict[str, object] = {"step": _step_count(entry["step"])}
            for key in ("exp_avg", "exp_avg_sq"):
                src = _from_host(entry[key], param._impl.device)
                moment = _moment_like(src, param)
                if moment is None:
                    raise _C_engine.ShapeMismatch(
                        f"SparseAdam.load_state_dict: {key} of parameter {idx} "
                        f"has shape {list(src.shape)}, the parameter "
                        f"{list(param._impl.shape)}"
                    )
                restored[key] = moment
            loaded[idx] = restored
        self.state = loaded
