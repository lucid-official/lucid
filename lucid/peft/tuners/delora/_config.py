"""``DeloraConfig``: the DeLoRA hyper-parameters on top of the shared targeting fields."""

from dataclasses import dataclass, field
from typing import ClassVar, Literal

from lucid.peft._config import (
    PeftType,
    _Check,
    _check_bool,
    _check_dropout,
    _check_number,
    _check_rank,
    _choice,
    _keyed,
    _nullable,
    _TargetedConfig,
)


@dataclass
class DeloraConfig(_TargetedConfig, peft_type=PeftType.DELORA):
    r"""How :func:`lucid.peft.get_peft_model` adapts a model with DeLoRA.

    DeLoRA (Bini et al., 2025, arXiv:2503.18225 — Decoupled Low-rank Adaptation) normalises each
    rank-one component of the LoRA update and scales the sum by a learned boundary :math:`\lambda`,

    .. math:: \Delta W = \frac{\lambda}{r} \sum_{i=1}^{r} \frac{b_i a_i^\top}{\lVert b_i \rVert\,\lVert a_i \rVert} .

    Every component then has unit Frobenius norm, so the direction of the update is learned apart
    from its strength and :math:`\lVert \Delta W \rVert_F \le \lambda`, which keeps the fine-tuned
    model from drifting far from the pretrained one.

    Parameters
    ----------
    task_type, peft_type, auto_mapping, peft_version, base_model_name_or_path, revision, inference_mode
        As in :class:`~lucid.peft.PeftConfig`.
    r : int, default=8
        Rank of the update.
    delora_lambda : float, default=15
        Initial boundary :math:`\lambda` on the Frobenius norm of the weight change.
    module_dropout : float, default=0.0
        Probability of disabling the module during training, in ``[0, 1)``.
    bias : {"none", "all", "delora_only"}, default="none"
        The reference trains none, all, or only the adapted layers' biases.  A DeLoRA adapter
        here trains its own tensors only, so ``"all"`` and ``"delora_only"`` raise
        ``NotImplementedError``.
    init_weights : bool, default=True
        ``True``: the update is zero at the start.  ``False``: both factors start random (for
        tests).
    rank_pattern : dict of str to int, optional
        Per-module rank overrides, keyed by name or regex suffix.
    lambda_pattern : dict of str to float, optional
        Per-module overrides of ``delora_lambda``, keyed the same way.
    target_modules : str or list of str, optional
        The modules DeLoRA adapts, matched as ``LoraConfig.target_modules`` is.
    exclude_modules : str or list of str, optional
        Modules DeLoRA never adapts, matched like ``target_modules``.
    layers_to_transform, layers_pattern, modules_to_save
        As in :class:`~lucid.peft.LoraConfig`.

    Examples
    --------
    >>> from lucid.peft import DeloraConfig
    >>> config = DeloraConfig(r=4, target_modules=['q_proj', 'v_proj'])
    >>> sorted(config.target_modules), config.peft_type
    (['q_proj', 'v_proj'], <PeftType.DELORA: 'DELORA'>)
    >>> DeloraConfig.from_dict(config.to_dict()) == config
    True

    Notes
    -----
    Every field of the reference library's config is here, in its order and with its defaults, read
    and written as ``adapter_config.json``.  A field of the wrong type raises ``TypeError``.
    ``ValueError`` is raised when ``r`` is not positive, a field is out of range or two fields
    conflict.  ``bias`` other than ``"none"`` raises ``NotImplementedError``.  The tuner is
    :class:`~lucid.peft.DeloraModel`; it adapts ``nn.Linear`` modules.
    """

    _unsupported: ClassVar[dict[str, str]] = {
        "bias": "a DeLoRA adapter trains its own tensors only (bias='none')"
    }

    _field_checks: ClassVar[dict[str, _Check]] = {
        "r": _check_rank,
        "delora_lambda": _check_number,
        "module_dropout": _check_dropout,
        "bias": _choice(("none", "all", "delora_only")),
        "init_weights": _check_bool,
        "rank_pattern": _nullable(_keyed(_check_rank)),
        "lambda_pattern": _nullable(_keyed(_check_number)),
    }

    r: int = 8
    delora_lambda: float = 15
    module_dropout: float = 0.0
    target_modules: str | list[str] | set[str] | None = None
    exclude_modules: str | list[str] | set[str] | None = None
    bias: Literal["none", "all", "delora_only"] = "none"
    init_weights: bool = True
    layers_to_transform: int | list[int] | None = None
    layers_pattern: str | list[str] | None = None
    rank_pattern: dict[str, int] | None = field(default_factory=dict)
    lambda_pattern: dict[str, float] | None = field(default_factory=dict)
    modules_to_save: list[str] | None = None
