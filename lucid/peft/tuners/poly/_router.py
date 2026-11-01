"""The routers of Poly: from the task of each batch row to the weights of the skills (private)."""

from abc import ABC, abstractmethod
from typing import ClassVar, override

import lucid
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter
from lucid.peft.tuners.poly._config import PolyConfig

#: Added to the normaliser so a row of vanishing weights does not divide by zero.
_EPS = 1e-12
#: How close to 0 and 1 the uniform noise of the relaxed sample is kept (float32 resolution).
_NOISE_EPS = 1.1920929e-07


class _Router(nn.Module, ABC):
    """Maps the task of each batch row to ``(batch, n_splits, n_skills)`` skill weights; one per ``poly_type``."""

    _routers: ClassVar[dict[str, type[_Router]]] = {}

    def __init__(self, config: PolyConfig) -> None:
        """Make a router for ``config``'s tasks and skills."""
        del config  # a subclass sizes its parameters from it
        super().__init__()

    @override
    def __init_subclass__(cls, *, poly_type: str | None = None, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if poly_type is not None:
            _Router._routers[poly_type] = cls

    @staticmethod
    def of(config: PolyConfig) -> _Router:
        """The router ``config.poly_type`` names, built for ``config``."""
        router = _Router._routers.get(config.poly_type)
        if router is None:
            raise ValueError(
                f"Unsupported poly_type: {config.poly_type}. "
                f"Currently, only the following types are supported: {', '.join(_Router._routers)}."
            )
        return router(config)

    @abstractmethod
    def reset(self) -> None:
        """Put the router's parameters at their starting values."""

    @abstractmethod
    @override
    def forward(self, task_ids: Tensor | None) -> Tensor:
        """The skill weights of each row, given the task of each row."""


class _PolyRouter(_Router, poly_type="poly"):
    r"""One learned logit vector per task; a row's skill weights are its task's, squashed and normalised.

    The weights are :math:`\sigma(z)` normalised over the skills of each split — in training a
    relaxed Bernoulli sample (temperature one) of them, a logistic noise added to the logits
    before the sigmoid, so routing is stochastic and differentiable.

    Parameters
    ----------
    config : PolyConfig
        ``n_tasks``, ``n_skills`` and ``n_splits`` size the logits.

    Attributes
    ----------
    module_logits : Parameter
        ``(n_tasks, n_splits * n_skills)``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.peft import PolyConfig
    >>> from lucid.peft.tuners.poly._router import _PolyRouter
    >>> router = _PolyRouter(PolyConfig(n_tasks=3, n_skills=2, n_splits=2, target_modules=["x"])).eval()
    >>> router(lucid.tensor([0, 2])).shape
    (2, 2, 2)
    """

    def __init__(self, config: PolyConfig) -> None:
        """Allocate the logits (uninitialised until ``reset``)."""
        super().__init__(config)
        self.n_tasks = config.n_tasks
        self.n_skills = config.n_skills
        self.n_splits = config.n_splits
        self.module_logits = Parameter(lucid.empty(self.n_tasks, self.n_splits * self.n_skills))

    @override
    def reset(self) -> None:
        """Draw every logit from ``U(-1e-3, 1e-3)``: near-uniform routing."""
        nn.init.uniform_(self.module_logits, -1e-3, 1e-3)

    @override
    def forward(self, task_ids: Tensor | None) -> Tensor:
        """The ``(batch, n_splits, n_skills)`` weights of the tasks ``task_ids`` (``(batch,)`` integers).

        Raises
        ------
        ValueError
            ``task_ids`` is missing, or names a task the router has no logits for.
        TypeError
            ``task_ids`` is not an integer tensor.
        """
        if task_ids is None:
            raise ValueError("task_ids should not be None.")
        if task_ids.dtype not in (lucid.int32, lucid.int64):
            raise TypeError(f"task_ids must be an integer tensor, got {task_ids.dtype}")
        highest, lowest = int(task_ids.max().item()), int(task_ids.min().item())
        if highest >= self.n_tasks:
            raise ValueError(f"Only {self.n_tasks} tasks available. Found task id = {highest}")
        if lowest < 0:
            raise ValueError(f"task ids must not be negative, got {lowest}")
        logits = self.module_logits.index_select(0, task_ids.to(self.module_logits.device))
        logits = logits.reshape(-1, self.n_splits, self.n_skills)
        if self.training:
            noise = lucid.rand(*logits.shape, device=logits.device).clamp(
                _NOISE_EPS, 1.0 - _NOISE_EPS
            )
            weights = lucid.sigmoid(logits + lucid.log(noise) - lucid.log1p(-noise))
        else:
            weights = lucid.sigmoid(logits)
        return weights / (weights.sum(dim=-1, keepdim=True) + _EPS)
