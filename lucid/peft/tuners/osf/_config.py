"""``OSFConfig``: the OSF hyper-parameters on top of the shared targeting fields."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.peft._config import (
    PeftType,
    _Check,
    _check_bool,
    _check_number,
    _check_rank,
    _keyed,
    _nullable,
    _TargetedConfig,
)


@dataclass
class OSFConfig(_TargetedConfig, peft_type=PeftType.OSF):
    r"""How :func:`lucid.peft.get_peft_model` adapts a model with OSF.

    OSF (Orthogonal Subspace Fine-tuning; Nayak et al., 2025, arXiv:2504.07097 — Sculpting Subspaces, section 3)
    is constrained full fine-tuning for continual learning.  It takes the SVD of each weight, freezes the top
    ``effective_rank`` singular directions, which carry what the model already knows, and trains the rest, with
    gradients projected orthogonally to the frozen directions so that a new task does not overwrite an old one.
    The trainable rank is ``min(weight.shape) - effective_rank``; no parameter is added to the model.

    Parameters
    ----------
    task_type, peft_type, auto_mapping, peft_version, base_model_name_or_path, revision, inference_mode
        As in :class:`~lucid.peft.PeftConfig`.
    effective_rank : int or float, optional
        Rank of the frozen high subspace.  An integer is a count; a float in ``(0, 1]`` is a
        fraction of the smaller weight dimension; ``None`` takes half of it.
    rank_pattern : dict of str to int or float, optional
        Per-module overrides of ``effective_rank``, keyed by regex; values are counts or
        fractions.
    init_weights : bool, optional
        ``True``: the trainable components start from the weight's own decomposition, so the
        adapter is the identity.  ``False``: they start random (for tests).
    target_svd_config : dict of str to int, optional
        Per-parameter SVD ranks, by parameter name; accepted for compatibility and not used.
    target_modules : str or list of str, optional
        The modules OSF adapts, matched as ``LoraConfig.target_modules`` is.
    modules_to_save
        As in :class:`~lucid.peft.LoraConfig`.

    Examples
    --------
    >>> from lucid.peft import OSFConfig
    >>> config = OSFConfig(effective_rank=0.5, target_modules=['q_proj', 'v_proj'])
    >>> sorted(config.target_modules), config.peft_type
    (['q_proj', 'v_proj'], <PeftType.OSF: 'OSF'>)
    >>> OSFConfig.from_dict(config.to_dict()) == config
    True

    Notes
    -----
    Field names, their order, defaults and ``adapter_config.json`` are the reference library's.
    ``TypeError`` is raised when a field holds a value of the wrong type.  A model type with no
    default row (``gpt2`` has one) adapts every linear layer when ``target_modules`` is left unset.
    """

    _field_checks: ClassVar[dict[str, _Check]] = {
        "effective_rank": _nullable(_check_number),
        "rank_pattern": _nullable(_keyed(_check_number)),
        "init_weights": _nullable(_check_bool),
        "target_svd_config": _nullable(_keyed(_check_rank)),
    }

    effective_rank: int | float | None = None
    target_modules: str | list[str] | set[str] | None = None
    rank_pattern: dict[str, int | float] | None = None
    init_weights: bool | None = True
    modules_to_save: list[str] | None = None
    target_svd_config: dict[str, int] | None = None
