r"""``SubspaceAllocator``: which subspaces of an AdaMSS model keep training (adaptive subspace allocation, ASA).

AdaMSS (Zheng et al., 2025, NeurIPS, section 4) trains every subspace at first and then freezes the
less important ones, so that the model ends up updating only a small subset of them.  Training
reports each step how sensitive the loss is to the entries of each subspace's two factors
(:math:`|w \cdot \partial L/\partial w|`); the allocator smooths that with a moving average, weighs
it by a second average of its uncertainty (the shared ``_Sensitivity``), and scores a subspace by the
sum of the mean score of its :math:`A_k` and of its :math:`B_k`.  On the steps the schedule marks it
freezes the subspaces that do not rank among the first ``target`` of the whole model, and starts the
averages again.  A frozen subspace keeps its value and keeps adding it to the output; it stops training.

It follows the reference library's ``update_and_allocate`` step for step, including two things the
paper does not say: a subspace is kept only if its score is *strictly* greater than the ``target``-th
highest, so a model asked to keep ``target`` subspaces keeps one fewer (more on ties), and a frozen
subspace, having no gradient, is never ranked again.
"""

from typing import NamedTuple

import lucid
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter
from lucid.peft.tuners._importance import _decayed_budget, _Sensitivity
from lucid.peft.tuners._layer import _tuner_layers
from lucid.peft.tuners.adamss._config import AdamssConfig
from lucid.peft.tuners.adamss._layer import AdamssLayer, _parameters


class _Subspace(NamedTuple):
    """One trainable subspace: its two factors and the running importance of each."""

    path: str
    position: int
    left: Parameter
    right: Parameter


class SubspaceAllocator:
    """Decides, on a schedule, which subspaces of an AdaMSS model stay trainable.

    Call :meth:`update_and_allocate` once per training step, after ``loss.backward()`` and before the
    gradients are cleared; ``AdamssModel.update_and_allocate`` does so for the adapters that asked
    for ASA.  Importances are gathered every step from ``init_warmup`` to ``final_warmup``; every
    ``mask_interval`` steps in that window the subspaces are ranked across the whole model, all but
    the best frozen, and the importances restarted.  The number to keep falls from the model's total
    number of subspaces to ``asa_target_subspaces`` as
    :math:`\\text{target} + (\\text{total} - \\text{target})\\,(1 - p)^{\\rho}` truncated, with
    :math:`p` the progress through the window and :math:`\\rho` ``asa_schedule_exponent``.

    Parameters
    ----------
    model : nn.Module
        The model AdaMSS was applied to.
    config : AdamssConfig
        The adapter's config: the window, the interval, the averages' weights and the target.
    adapter_name : str
        The adapter that trains.

    Raises
    ------
    ValueError
        No AdaMSS layer of ``model`` holds ``adapter_name``.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.peft import AdamssConfig, AdamssModel
    >>> net = nn.Sequential(nn.Linear(6, 6), nn.Linear(6, 2))
    >>> config = AdamssConfig(r=4, num_subspaces=3, use_asa=True, asa_target_subspaces=2,
    ...                       init_warmup=1, final_warmup=5, mask_interval=1, target_modules=["0", "1"])
    >>> allocator = AdamssModel(net, config, "default").allocator("default")
    >>> allocator.total, [allocator.target(step) for step in range(1, 6)]
    (5, [5, 3, 2, 2, 2])
    """

    def __init__(self, model: nn.Module, config: AdamssConfig, adapter_name: str) -> None:
        """Find the adapter's layers in ``model``; ``config`` sets the schedule."""
        self.model = model
        self.config = config
        self.adapter_name = adapter_name
        self._sensitivity: dict[tuple[str, int, str], _Sensitivity] = {}
        self._subspaces()  # raises if there is nothing to allocate

    # ── what there is to allocate ─────────────────────────────────────────

    def _layers(self) -> dict[str, AdamssLayer]:
        """The AdaMSS layers holding the adapter, by their path in the model."""
        return {
            path: layer
            for path, layer in _tuner_layers(self.model)
            if isinstance(layer, AdamssLayer) and self.adapter_name in layer._specs
        }

    def _subspaces(self) -> list[_Subspace]:
        layers = self._layers()
        if not layers:
            raise ValueError(
                f"no AdaMSS layer of the model holds the adapter {self.adapter_name!r}"
            )
        return [
            _Subspace(path, position, left, right)
            for path, layer in layers.items()
            for position, (left, right) in enumerate(
                zip(
                    _parameters(layer.adamss_A, self.adapter_name),
                    _parameters(layer.adamss_B, self.adapter_name),
                    strict=True,
                )
            )
        ]

    @property
    def total(self) -> int:
        """The subspaces of the whole model, whether or not they train."""
        return len(self._subspaces())

    # ── the schedule ──────────────────────────────────────────────────────

    def target(self, step: int) -> int | None:
        """How many subspaces ``step`` allows to stay trainable, or ``None`` before the window opens.

        Parameters
        ----------
        step : int
            The training step.

        Returns
        -------
        int or None
            The truncated schedule value; ``asa_target_subspaces`` from ``final_warmup`` on.
        """
        config = self.config
        if step < config.init_warmup:
            return None
        if step > config.final_warmup:
            return config.asa_target_subspaces
        progress = (step - config.init_warmup) / (config.final_warmup - config.init_warmup)
        remaining = max(0.0, min(1.0, 1.0 - progress))
        return _decayed_budget(
            self.total, config.asa_target_subspaces, remaining, config.asa_schedule_exponent
        )

    # ── recording and masking ─────────────────────────────────────────────

    def update_importance(self) -> None:
        """Record the importance of every factor that has a gradient, at its current value."""
        config = self.config
        for space in self._subspaces():
            for side, factor in (("A", space.left), ("B", space.right)):
                if factor.grad is not None:
                    state = self._sensitivity.setdefault(
                        (space.path, space.position, side), _Sensitivity()
                    )
                    state.update(
                        factor,
                        config.asa_importance_beta,
                        config.asa_uncertainty_beta,
                        from_previous_mean=True,
                    )

    def reset_importance(self) -> None:
        """Forget the recorded importances."""
        self._sensitivity.clear()

    def _scores(self) -> list[tuple[_Subspace, Tensor]]:
        """Each subspace both of whose factors have an importance, with its score: the two factors' mean scores added."""
        scored: list[tuple[_Subspace, Tensor]] = []
        for space in self._subspaces():
            left = self._sensitivity.get((space.path, space.position, "A"))
            right = self._sensitivity.get((space.path, space.position, "B"))
            if left is not None and right is not None:
                scored.append((space, left.score.mean() + right.score.mean()))
        return scored

    def mask_to_target(self, target: int) -> None:
        """Freeze every recorded subspace whose score is not strictly above the ``target``-th highest.

        Nothing is frozen while ``target`` is at least the number of subspaces recorded.  A frozen
        factor loses its gradient and its ``requires_grad``; a kept one is trainable.

        Parameters
        ----------
        target : int
            The number of subspaces to keep, in the whole model.
        """
        scored = self._scores()
        if not scored:
            return
        listed = lucid.stack([score for _, score in scored]).tolist()
        assert isinstance(listed, list)
        values = [float(value) for value in listed if isinstance(value, (int, float))]
        threshold = (
            float("-inf") if target >= len(values) else sorted(values, reverse=True)[target - 1]
        )
        for (space, _), value in zip(scored, values, strict=True):
            kept = value > threshold
            for factor in (space.left, space.right):
                factor.requires_grad_(kept)
                if not kept:
                    factor.grad = None

    def update_and_allocate(self, global_step: int) -> None:
        """One training step's worth of allocation: record importances, then freeze if the schedule says so.

        Nothing happens outside the window ``init_warmup <= step <= final_warmup``.  Inside it the
        importances are recorded every step; on a step that is a multiple of ``mask_interval`` the
        subspaces are masked to the schedule's target and the importances restart.

        Parameters
        ----------
        global_step : int
            The training step.
        """
        config = self.config
        if not config.init_warmup <= global_step <= config.final_warmup:
            return
        self.update_importance()
        if global_step % config.mask_interval == 0:
            target = self.target(global_step)
            if target is not None:
                self.mask_to_target(target)
            self.reset_importance()
