"""``GraloraConfig``: the GraLoRA hyper-parameters on top of the shared targeting fields."""

from dataclasses import dataclass
from typing import ClassVar, Literal, override

from lucid.peft._config import (
    PeftType,
    _Check,
    _check_bool,
    _check_count,
    _check_dropout,
    _check_number,
    _check_rank,
    _choice,
    _TargetedConfig,
)


@dataclass
class GraloraConfig(_TargetedConfig, peft_type=PeftType.GRALORA):
    r"""How :func:`lucid.peft.get_peft_model` adapts a model with GraLoRA.

    GraLoRA (Jung et al., 2025, arXiv:2505.20355 — Granular Low-Rank Adaptation) cuts a weight into a
    ``gralora_k`` by ``gralora_k`` grid of sub-blocks and gives every block its own low-rank pair of rank
    ``r / gralora_k``, with information exchanged between the blocks.  The parameter count is that of a
    rank-``r`` LoRA, while the update can reach about ``gralora_k`` times its effective rank.  With
    ``hybrid_r > 0`` a plain LoRA of that rank is added (Hybrid GraLoRA); the update is scaled by
    :math:`\alpha / (r + \text{hybrid\_r})`.

    Parameters
    ----------
    task_type, peft_type, auto_mapping, peft_version, base_model_name_or_path, revision, inference_mode
        As in :class:`~lucid.peft.PeftConfig`.
    r : int, default=32
        Rank of the block-wise part; must be divisible by ``gralora_k``.
    hybrid_r : int, default=0
        Rank of the plain LoRA added beside it; ``0`` adds none.
    alpha : float, default=64
        Scaling factor :math:`\alpha`; the scale is ``alpha / (r + hybrid_r)``.
    gralora_dropout : float, default=0.0
        Dropout on the update's input, in ``[0, 1)``.
    gralora_k : int, default=2
        Number of sub-blocks along each side of the grid; positive.
    bias : {"none", "all", "gralora_only"}, default="none"
        Which biases also train with GraLoRA: none, all of them, or only those of the adapted
        layers.
    init_weights : bool, default=True
        Initialise the factors with their default scheme.
    target_modules : str or list of str, optional
        The modules GraLoRA adapts, matched as ``LoraConfig.target_modules`` is.
    fan_in_fan_out : bool, default=False
        Set for a base that stores its weight as ``(in, out)``; a GraLoRA layer follows that
        layout.
    modules_to_save, layers_to_transform, layers_pattern
        As in :class:`~lucid.peft.LoraConfig`.

    Examples
    --------
    >>> from lucid.peft import GraloraConfig
    >>> config = GraloraConfig(r=8, gralora_k=2, target_modules=['q_proj', 'v_proj'])
    >>> sorted(config.target_modules), config.peft_type
    (['q_proj', 'v_proj'], <PeftType.GRALORA: 'GRALORA'>)
    >>> GraloraConfig.from_dict(config.to_dict()) == config
    True

    Notes
    -----
    GraLoRA's config is complete: every field of the reference library's, in its order and with its
    defaults, read and written as ``adapter_config.json``.  A field of the wrong type raises
    ``TypeError``.  ``ValueError`` is raised when ``r`` is not divisible by ``gralora_k``, or a
    field is out of range.  No tuner applies it yet: ``get_peft_model`` raises
    ``NotImplementedError`` naming LCD-823, which tracks it.
    """

    _field_checks: ClassVar[dict[str, _Check]] = {
        "r": _check_rank,
        "hybrid_r": _check_count,
        "alpha": _check_number,
        "gralora_dropout": _check_dropout,
        "gralora_k": _check_rank,
        "fan_in_fan_out": _check_bool,
        "bias": _choice(("none", "all", "gralora_only")),
        "init_weights": _check_bool,
    }

    @override
    def _check_fields(self) -> None:
        super()._check_fields()
        if self.r % self.gralora_k != 0:
            raise ValueError(
                f"r should be divisible by gralora_k, but got {self.r} and {self.gralora_k}"
            )

    r: int = 32
    hybrid_r: int = 0
    target_modules: str | list[str] | set[str] | None = None
    alpha: float = 64
    gralora_dropout: float = 0.0
    gralora_k: int = 2
    fan_in_fan_out: bool = False
    bias: Literal["none", "all", "gralora_only"] = "none"
    modules_to_save: list[str] | None = None
    init_weights: bool = True
    layers_to_transform: int | list[int] | None = None
    layers_pattern: str | list[str] | None = None
