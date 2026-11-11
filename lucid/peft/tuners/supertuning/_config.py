"""``SupertuningConfig``: the Super-Tuning hyper-parameters on top of the shared targeting fields."""

import warnings
from dataclasses import dataclass
from typing import ClassVar, override

from lucid.peft._config import (
    PeftType,
    _Check,
    _check_bool,
    _check_dropout,
    _check_number,
    _check_rank,
    _nullable,
    _TargetedConfig,
)


@dataclass
class SupertuningConfig(_TargetedConfig, peft_type=PeftType.SUPERTUNING):
    r"""How :func:`lucid.peft.get_peft_model` adapts a model with Super-Tuning.

    Super-Tuning (Ilin et al., 2026, arXiv:2607.09287 — From Activation-Aware Pruning to Sparse
    Fine-Tuning) freezes the base weight and trains a sparse support of its entries: the ``1 - sparsity``
    fraction with the largest magnitude (``select_top``), as an additive update that starts at zero.  With ``r``
    set, a LoRA pair is added beside it (the paper's Supra hybrid).

    Parameters
    ----------
    task_type, peft_type, auto_mapping, peft_version, base_model_name_or_path, revision, inference_mode
        As in :class:`~lucid.peft.PeftConfig`.
    sparsity : float, default=0.99
        Fraction of the weight's entries left frozen, in ``[0, 1)``; ``0.99`` trains one
        percent.
    select_top : bool, default=True
        Train the largest-magnitude entries (``True``) or the smallest.
    r : int, optional
        Rank of the LoRA part of the Supra hybrid; ``None`` trains the sparse support alone.
    lora_alpha : float, optional
        Scale of that LoRA part; ``2 * r`` when omitted.  Needs ``r``.
    lora_dropout : float, default=0.0
        Dropout on the LoRA part, in ``[0, 1)``.
    init_weights : bool, default=True
        ``True``: the values and :math:`B` start at zero (identity).  ``False``: random (for
        tests).
    save_precomputed_indices : bool, default=True
        Save the support's indices with the adapter; ``False`` rebuilds them from the base
        weight on load.
    target_modules : str or list of str, optional
        The modules Super-Tuning adapts, matched as ``LoraConfig.target_modules`` is.
    modules_to_save
        As in :class:`~lucid.peft.LoraConfig`.

    Examples
    --------
    >>> from lucid.peft import SupertuningConfig
    >>> config = SupertuningConfig(sparsity=0.9, r=4, target_modules=['q_proj', 'v_proj'])
    >>> sorted(config.target_modules), config.peft_type
    (['q_proj', 'v_proj'], <PeftType.SUPERTUNING: 'SUPERTUNING'>)
    >>> config.lora_alpha
    8
    >>> SupertuningConfig.from_dict(config.to_dict()) == config
    True

    Notes
    -----
    Every field of the reference library's config is here, in its order and with its defaults, and the
    config is read and written as ``adapter_config.json``.  A field of the wrong type raises
    ``TypeError``.  ``ValueError`` is raised when ``sparsity`` is outside ``[0, 1)``, or ``lora_alpha``
    is given without ``r``.  ``save_precomputed_indices=False`` warns: the positions are then chosen
    again from the base weight when the adapter is loaded, which assumes the weight is the one the
    adapter was trained on.  A weight with few entries needs ``sparsity`` low enough that
    ``floor(entries * (1 - sparsity))`` is at least one; the layer raises ``ValueError`` otherwise.
    """

    _field_checks: ClassVar[dict[str, _Check]] = {
        "sparsity": _check_number,
        "select_top": _check_bool,
        "r": _nullable(_check_rank),
        "lora_alpha": _nullable(_check_number),
        "lora_dropout": _check_dropout,
        "init_weights": _check_bool,
        "save_precomputed_indices": _check_bool,
    }

    @override
    def _check_fields(self) -> None:
        super()._check_fields()
        if not 0.0 <= self.sparsity < 1.0:
            raise ValueError(f"sparsity must be in [0.0, 1.0), got {self.sparsity}")
        if self.r is None:
            if self.lora_alpha is not None:
                raise ValueError(
                    "lora_alpha is set but r is None — Supra mode requires an explicit r."
                )
        elif self.lora_alpha is None:
            self.lora_alpha = 2 * self.r
        if not self.save_precomputed_indices:
            warnings.warn(
                "save_precomputed_indices=False: sparse-support indices will be recomputed from base "
                "weight magnitudes at load time. This assumes the base model is identical to training "
                "time; numerical drift can cause topk tie-breaks to differ.",
                stacklevel=3,
            )

    target_modules: str | list[str] | set[str] | None = None
    modules_to_save: list[str] | None = None
    sparsity: float = 0.99
    select_top: bool = True
    r: int | None = None
    lora_alpha: float | None = None
    lora_dropout: float = 0.0
    init_weights: bool = True
    save_precomputed_indices: bool = True
