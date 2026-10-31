r"""``RankAllocator``: which singular triplets of an AdaLoRA model are worth their budget.

Training reports each step how sensitive the loss is to every entry of
``lora_A`` / ``lora_E`` / ``lora_B`` (:math:`|w \cdot \partial L/\partial w|`);
the allocator smooths that with a moving average, weighs it by a second
average of its uncertainty, and scores a triplet by the sum of what its
singular value, its row of :math:`Q` and its column of :math:`P` score.
On the budget schedule it zeroes the singular values of every triplet at or
below the score that leaves the remaining budget, and records which survived
(the *rank pattern*).
"""

from collections.abc import Mapping
from typing import NamedTuple

import lucid
from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter
import lucid.nn as nn
from lucid.peft.tuners._base import _parameter_at, _tuner_layers
from lucid.peft.tuners.adalora._checks import _check_rank_pattern
from lucid.peft.tuners.adalora._config import AdaLoraConfig
from lucid.peft.tuners.adalora._layer import AdaLoraLayer


class _Budget(NamedTuple):
    """What the schedule says for one step: triplets the model may keep, and whether to mask down to it."""

    triplets: int
    mask: bool


class _Sensitivity:
    """The moving averages of one parameter's sensitivity :math:`|w\\,g|` and of its uncertainty.

    ``score`` — their product — is what ranks the entries.  The state is
    allocated on first use and again if the parameter's shape changes
    (a layer was pruned or resized under it).
    """

    __slots__ = ("_mean", "_spread")

    def __init__(self) -> None:
        self._mean: Tensor | None = None
        self._spread: Tensor | None = None

    def update(self, parameter: Parameter, beta1: float, beta2: float) -> None:
        """Fold in the sensitivity of ``parameter`` at its current value and gradient."""
        grad = parameter.grad
        if grad is None:
            raise RuntimeError(
                "update_and_allocate needs the gradients of the adapter: call it after "
                "loss.backward() and before the gradients are cleared"
            )
        with lucid.no_grad():
            sensitivity = (parameter * grad).abs()
            mean, spread = self._mean, self._spread
            if mean is None or spread is None or mean.shape != sensitivity.shape:
                mean, spread = lucid.zeros_like(sensitivity), lucid.zeros_like(sensitivity)
            self._mean = mean = beta1 * mean + (1 - beta1) * sensitivity
            self._spread = beta2 * spread + (1 - beta2) * (sensitivity - mean).abs()

    @property
    def score(self) -> Tensor:
        """Smoothed sensitivity times smoothed uncertainty, shaped like the parameter."""
        if self._mean is None or self._spread is None:
            raise RuntimeError("no sensitivity has been recorded yet")
        return self._mean * self._spread


class RankAllocator:
    """Decides, on a schedule, how many singular triplets each AdaLoRA layer keeps.

    Reference: Zhang et al., 2023 (AdaLoRA).  The allocator reads the
    sensitivity of the adapter's tensors after every backward pass
    (:meth:`update_ipt`), and on the steps the budget schedule marks, masks
    the least important triplets of the whole model down to the budget
    (:meth:`mask_to_budget`).  :meth:`update_and_allocate` is the one call a
    training step makes, and ``AdaLoraModel.update_and_allocate`` makes it for
    the model.

    The budget starts at the model's total number of triplets, stays there
    for ``tinit`` steps, falls on a cubic schedule to ``target_r`` triplets
    per layer at step ``total_step - tfinal`` and stays there.

    Parameters
    ----------
    model : nn.Module
        The model AdaLoRA was applied to.
    peft_config : AdaLoraConfig
        The adapter's config: the schedule and the moving-average weights.
    adapter_name : str
        The adapter that trains.

    Raises
    ------
    ValueError
        ``adapter_name`` is not held by any AdaLoRA layer of ``model``.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.peft import AdaLoraConfig, AdaLoraModel
    >>> net = nn.Sequential(nn.Linear(6, 6), nn.Linear(6, 2))
    >>> config = AdaLoraConfig(init_r=4, target_r=2, total_step=20, tfinal=5, target_modules=["0", "1"])
    >>> allocator = AdaLoraModel(net, config).rankallocator
    >>> allocator.budget_schedule(0), allocator.budget_schedule(19)
    ((8, False), (4, True))
    """

    def __init__(self, model: nn.Module, peft_config: AdaLoraConfig, adapter_name: str) -> None:
        self.peft_config = peft_config
        self.adapter_name = adapter_name
        self.beta1 = peft_config.beta1
        self.beta2 = peft_config.beta2
        self._sensitivity: dict[tuple[str, str], _Sensitivity] = {}
        #: Triplets the model holds before any is masked, and the total ``target_r`` per layer asks for.
        self.init_bgt = 0
        self.target_bgt = 0
        self._measure(model)

    # ── what there is to allocate ─────────────────────────────────────────

    def _layers(self, model: nn.Module) -> dict[str, AdaLoraLayer]:
        """The AdaLoRA layers holding the adapter, by their path in ``model``."""
        return {
            path: layer
            for path, layer in _tuner_layers(model)
            if isinstance(layer, AdaLoraLayer) and self.adapter_name in layer._specs
        }

    def _tables(self, layer: AdaLoraLayer) -> dict[str, nn.ParameterDict]:
        return {"lora_A": layer.lora_A, "lora_E": layer.lora_E, "lora_B": layer.lora_B}

    def _measure(self, model: nn.Module) -> None:
        """Set the budgets from the layers as they are now (a layer pruned or resized since changes them)."""
        layers = self._layers(model)
        if not layers:
            raise ValueError(
                f"no AdaLoRA layer of the model holds the adapter {self.adapter_name!r}"
            )
        self.init_bgt = sum(layer._specs[self.adapter_name].r for layer in layers.values())
        self.target_bgt = self.peft_config.target_r * len(layers)

    @property
    def _total_step(self) -> int:
        total = self.peft_config.total_step
        if total is None:
            raise ValueError("AdaLoRA does not work when `total_step` is None, supply a value > 0.")
        return total

    @property
    def final_step(self) -> int:
        """The step the budget reaches ``target_r`` and the final fine-tuning begins: ``total_step - tfinal``."""
        return self._total_step - self.peft_config.tfinal

    def set_total_step(self, total_step: int) -> None:
        """Change the number of training steps the schedule spreads the budget over."""
        self.peft_config.total_step = total_step

    def reset_ipt(self) -> None:
        """Forget the recorded sensitivities."""
        self._sensitivity.clear()

    # ── the schedule ──────────────────────────────────────────────────────

    def budget_schedule(self, step: int) -> _Budget:
        """The triplet budget at ``step`` and whether this step masks down to it.

        Parameters
        ----------
        step : int
            The training step.

        Returns
        -------
        tuple
            ``(triplets, mask)``: the budget, and whether to mask down to it now.
        """
        config = self.peft_config
        tinit, final = config.tinit, self.final_step
        if step <= tinit:
            return _Budget(self.init_bgt, False)
        if step > final:
            return _Budget(self.target_bgt, True)
        remaining = 1 - (step - tinit) / (final - tinit)
        budget = int((self.init_bgt - self.target_bgt) * remaining**3 + self.target_bgt)
        return _Budget(budget, step % config.deltaT == 0)

    # ── recording and masking ─────────────────────────────────────────────

    def update_ipt(self, model: nn.Module) -> None:
        """Record the sensitivity of every adapter tensor at its current value and gradient."""
        for path, layer in self._layers(model).items():
            for name, table in self._tables(layer).items():
                state = self._sensitivity.setdefault((path, name), _Sensitivity())
                state.update(_parameter_at(table, self.adapter_name), self.beta1, self.beta2)

    def _score(self, path: str, table: str) -> Tensor:
        state = self._sensitivity.get((path, table))
        if state is None:
            raise RuntimeError(
                "no importance has been recorded: update_and_allocate scores the triplets it "
                "saw on earlier steps, so call it on every step from the first"
            )
        return state.score

    def _triplet_scores(self, model: nn.Module) -> dict[str, Tensor]:
        """Per layer, one score per triplet: its value's, its row's and its column's, added."""
        scores: dict[str, Tensor] = {}
        for path in self._layers(model):
            value = self._score(path, "lora_E").reshape(-1)
            row = self._score(path, "lora_A").mean(dim=1)
            column = self._score(path, "lora_B").mean(dim=0)
            scores[path] = value + row + column
        return scores

    def mask_to_budget(self, model: nn.Module, budget: int) -> dict[str, list[bool]]:
        """Zero the singular values of the least important triplets so that ``budget`` are left.

        Every triplet scoring at or below the ``(initial - budget)``-th lowest
        score is masked, so ties can leave fewer than ``budget``.

        Parameters
        ----------
        model : nn.Module
            The model whose layers are masked.
        budget : int
            Triplets to keep, in the whole model.

        Returns
        -------
        dict of str to list of bool
            The rank pattern: ``"<module>.lora_E"`` to one flag per triplet, ``True`` where it was kept.
        """
        scores = self._triplet_scores(model)
        everything = lucid.cat(list(scores.values()))
        drop = self.init_bgt - budget
        # Nothing to drop (or a budget below zero) is a threshold nothing falls under.
        threshold = (
            float(lucid.kthvalue(everything, k=min(drop, everything.shape[0]))[0].item())
            if drop > 0
            else float("-inf")
        )
        pattern: dict[str, list[bool]] = {}
        with lucid.no_grad():
            for path, layer in self._layers(model).items():
                kept = scores[path] > threshold
                singular = _parameter_at(layer.lora_E, self.adapter_name)
                singular.copy_(singular * kept.reshape(-1, 1).to(singular.dtype))
                flags = kept.tolist()
                assert isinstance(flags, list)
                pattern[f"{path}.lora_E"] = [bool(flag) for flag in flags]
        return pattern

    def update_and_allocate(
        self, model: nn.Module, global_step: int, force_mask: bool = False
    ) -> tuple[int, dict[str, list[bool]] | None]:
        """One training step's worth of allocation: record sensitivities, then mask if the schedule says so.

        Parameters
        ----------
        model : nn.Module
            The model whose adapter trains.
        global_step : int
            The training step.
        force_mask : bool, default=False
            Mask down to the budget even on a step the schedule skips.

        Returns
        -------
        tuple
            The budget of this step and the rank pattern, ``None`` if nothing was masked.
        """
        self._measure(model)
        if global_step < self.final_step:
            self.update_ipt(model)
        budget = self.budget_schedule(global_step)
        pattern = self.mask_to_budget(model, budget.triplets) if budget.mask or force_mask else None
        return budget.triplets, pattern

    def mask_using_rank_pattern(
        self, model: nn.Module, rank_pattern: Mapping[str, list[bool]]
    ) -> None:
        """Zero the singular values the rank pattern dropped (the final phase, after the budget is reached).

        A layer whose rank no longer matches its pattern was pruned already
        and is left alone.

        Parameters
        ----------
        model : nn.Module
            The model whose layers are masked.
        rank_pattern : dict of str to list of bool
            ``"<module>.lora_E"`` (the adapter's name may follow) to one flag per triplet.
        """
        pattern = _check_rank_pattern("rank_pattern", dict(rank_pattern)) or {}
        with lucid.no_grad():
            for path, layer in self._layers(model).items():
                kept = pattern.get(f"{path}.lora_E")
                singular = _parameter_at(layer.lora_E, self.adapter_name)
                if kept is None or len(kept) != singular.shape[0]:
                    continue
                flags = lucid.tensor(kept, dtype=singular.dtype, device=singular.device)
                singular.copy_(singular * flags.reshape(-1, 1))
