"""``AdaLoraConfig``: the AdaLoRA hyper-parameters on top of the shared targeting fields."""

import warnings
from dataclasses import dataclass
from typing import ClassVar, Literal, Self, override

from lucid.peft._config import (
    PeftType,
    _Check,
    _check_bool,
    _check_dropout,
    _check_number,
    _check_rank,
    _choice,
    _TargetedConfig,
)
from lucid.peft.tuners.adalora._checks import (
    _check_ema,
    _check_non_negative,
    _check_rank_pattern,
    _check_steps,
    _check_total_step,
)


@dataclass(kw_only=True)
class AdaLoraConfig(_TargetedConfig, peft_type=PeftType.ADALORA):
    r"""How :func:`lucid.peft.get_peft_model` adapts a model with AdaLoRA.

    AdaLoRA (Zhang et al., 2023, arXiv:2303.10512) writes the update of a
    targeted ``nn.Linear`` (or ``nn.Conv2d``) as a truncated SVD,

    .. math:: \Delta W = \frac{\alpha}{\mathrm{ranknum}}\, P\,\mathrm{diag}(\lambda)\,Q,
              \qquad P \in \mathbb{R}^{d_{out} \times r},\;
              \lambda \in \mathbb{R}^{r},\;
              Q \in \mathbb{R}^{r \times d_{in}},

    with :math:`P, Q` kept near-orthogonal by a penalty on the loss, and
    learns how many of the :math:`r` singular triplets each layer deserves:
    the importance of each triplet is tracked during training and the least
    important are masked away on a schedule until the model holds
    ``target_r`` triplets per layer on average
    (:meth:`~lucid.peft.tuners.adalora.AdaLoraModel.update_and_allocate`).

    Training has three phases, set by ``tinit``, ``tfinal`` and ``total_step``:
    the first ``tinit`` steps train every triplet; then the budget falls on a
    cubic schedule until step ``total_step - tfinal``, where it reaches
    ``target_r`` per layer; the last ``tfinal`` steps fine-tune what is left.
    With ``tinit = 10``, ``tfinal = 20`` and ``total_step = 100``, 10 steps
    warm up, 70 steps shrink the budget and 20 steps fine-tune.

    Field names, defaults and ``adapter_config.json`` are the reference
    adapter library's.  Every field is keyword-only.  The reference's
    ``LoraConfig`` fields that AdaLoRA does not use (``use_rslora``,
    ``use_dora``, ``lora_bias``, ``alpha_pattern`` …) are not fields here;
    an adapter file that carries them at their "off" value is read, and one
    that turns them on is refused.

    Parameters
    ----------
    target_r : int, default=8
        The target average rank of the update: the budget the allocator
        shrinks to is ``target_r`` triplets per adapted layer.
    init_r : int, default=12
        The rank each layer starts with (its number of singular triplets).
    tinit : int, default=0
        Steps of initial fine-tuning before the budget starts to fall.
    tfinal : int, default=0
        Steps of final fine-tuning, after the budget has reached ``target_r``.
    deltaT : int, default=1
        Steps between two budget allocations.
    beta1 : float, default=0.85
        Weight of the previous value in the moving average that smooths a
        triplet's sensitivity; in ``(0, 1)``.
    beta2 : float, default=0.85
        Weight of the previous value in the moving average that tracks the
        uncertainty of the sensitivity; in ``(0, 1)``.
    orth_reg_weight : float, default=0.5
        Coefficient of the orthogonality penalty added to the loss.
    total_step : int
        The number of training steps; required, and must exceed
        ``tinit + tfinal``.
    rank_pattern : dict of str to list of bool, optional
        The triplets the allocator kept, per layer — ``"<module>.lora_E"``
        to one flag per initial triplet.  Filled in by training and saved with
        the adapter; the adapter's tensors are saved without the dropped ones.
    r : int, default=8
        Not used by AdaLoRA (it warns when set): the initial rank is ``init_r``.
    lora_alpha : float, default=8
        The numerator of the scale; the update is multiplied by
        ``lora_alpha / ranknum`` (``ranknum`` being the initial rank).  A
        value of 0 or below takes the rank itself, which makes the scale 1.
    lora_dropout : float, default=0.0
        Dropout on the update's input, in ``[0, 1)``.
    fan_in_fan_out : bool, default=False
        For bases that store their weight ``(in, out)``.  ``nn.Linear`` does
        not, so ``True`` warns and is reset to ``False``.
    bias : {"none"}, default="none"
        The reference also offers ``"all"`` and ``"lora_only"``; here the
        adapter trains its own tensors only, and those raise ``NotImplementedError``.
    init_lora_weights : bool, default=True
        ``True``: the singular values start at zero and the vectors are
        :math:`\mathcal{N}(0, 0.02^2)`, so the adapted model starts as the
        base.  ``False``: everything is :math:`\mathcal{N}(0, 1)` (for tests).
    target_modules : str or list of str, optional
        The ``nn.Linear`` / ``nn.Conv2d`` modules to adapt, matched as
        ``LoraConfig.target_modules`` is.  ``None`` takes the default for the
        base model's ``model_type`` — the LoRA table, as the reference does.
    exclude_modules, layers_to_transform, layers_pattern, modules_to_save
        As in ``LoraConfig``.

    Raises
    ------
    TypeError
        A field holds a value of the wrong type.
    ValueError
        A field is out of range, ``total_step`` is missing, or the schedule
        leaves no room for the budget to fall (``tinit >= total_step - tfinal``).
    NotImplementedError
        ``bias`` is not ``"none"``.

    Examples
    --------
    >>> from lucid.peft import AdaLoraConfig
    >>> config = AdaLoraConfig(init_r=12, target_r=4, total_step=100, target_modules=["q", "v"])
    >>> sorted(config.target_modules), config.peft_type
    (['q', 'v'], <PeftType.ADALORA: 'ADALORA'>)
    >>> AdaLoraConfig.from_dict(config.to_dict()) == config
    True
    """

    _field_checks: ClassVar[dict[str, _Check]] = {
        "target_r": _check_rank,
        "init_r": _check_rank,
        "tinit": _check_steps,
        "tfinal": _check_steps,
        "deltaT": _check_rank,
        "beta1": _check_ema,
        "beta2": _check_ema,
        "orth_reg_weight": _check_non_negative,
        "total_step": _check_total_step,
        "rank_pattern": _check_rank_pattern,
        "r": _check_rank,
        "lora_alpha": _check_number,
        "lora_dropout": _check_dropout,
        "fan_in_fan_out": _check_bool,
        "bias": _choice(("none", "all", "lora_only")),
        "init_lora_weights": _check_bool,
    }

    target_r: int = 8
    init_r: int = 12
    tinit: int = 0
    tfinal: int = 0
    deltaT: int = 1
    beta1: float = 0.85
    beta2: float = 0.85
    orth_reg_weight: float = 0.5
    total_step: int | None = None
    rank_pattern: dict[str, list[bool]] | None = None
    r: int = 8
    lora_alpha: float = 8
    lora_dropout: float = 0.0
    fan_in_fan_out: bool = False
    bias: Literal["none", "all", "lora_only"] = "none"
    init_lora_weights: bool = True

    @override
    @classmethod
    def _blank(cls) -> Self:
        return cls(
            total_step=1
        )  # the one field with no usable default; ``from_dict`` sets the real one

    @override
    def _check_fields(self) -> None:
        super()._check_fields()
        if self.bias != "none":
            raise NotImplementedError(
                f"bias={self.bias!r}: an AdaLoRA adapter in lucid.peft trains its own tensors "
                "only (bias='none')"
            )
        if self.r != 8:
            warnings.warn(
                "Note that `r` is not used in AdaLora and will be ignored."
                "If you intended to set the initial rank, use `init_r` instead.",
                stacklevel=2,
            )
        if self.total_step is None or self.total_step <= 0:
            raise ValueError("AdaLoRA does not work when `total_step` is None, supply a value > 0.")
        if self.tinit >= self.total_step - self.tfinal:
            raise ValueError(
                "The supplied schedule values don't allow for a budgeting phase. Decrease "
                "`tfinal`/`tinit` or increase `total_step`."
            )
