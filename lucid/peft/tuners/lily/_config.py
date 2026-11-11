"""``LilyConfig``: the Lily hyper-parameters on top of the shared targeting fields."""

from dataclasses import dataclass
from typing import ClassVar, override

from lucid.peft._config import (
    PeftType,
    _Check,
    _check_bool,
    _check_count,
    _check_number,
    _check_rank,
    _TargetedConfig,
)


@dataclass
class LilyConfig(_TargetedConfig, peft_type=PeftType.LILY):
    r"""How :func:`lucid.peft.get_peft_model` adapts a model with Lily.

    Lily (Zhong et al., 2024, arXiv:2407.09946 — Low-Rank Interconnected Adaptation across Layers)
    breaks LoRA's pairing of a down- and an up-projection to one layer.  Every ``stride_A`` consecutive layers
    share one low-rank projection :math:`A`; ``num_B`` projections :math:`B` are shared by all layers, and a
    learned router mixes them with softmax weights into one combined :math:`B` per layer.  The layer adds
    :math:`\text{scaling} \cdot A\,B_{\text{combined}}` to its output, so a layer's update draws on adapters
    other layers train too.

    Parameters
    ----------
    task_type, peft_type, auto_mapping, peft_version, base_model_name_or_path, revision, inference_mode
        As in :class:`~lucid.peft.PeftConfig`.
    r : int, default=32
        Inner dimension of every adapter; a larger value than a LoRA rank, since there are fewer
        adapters.
    stride_A : int, default=1
        Consecutive layers that share one :math:`A`; at least 1.
    num_B : int, default=2
        Number of :math:`B` adapters shared by all layers; at least 2, so the router has
        something to train.
    scaling : float, default=1.0
        Direct multiplier on the adapter's output (there is no ``alpha / r``).
    init_weights : bool, default=True
        ``True``: :math:`B` starts at zero, so the update does.  ``False``: random (for tests).
    target_modules : str or list of str, optional
        The modules Lily adapts, matched as ``LoraConfig.target_modules`` is.
    exclude_modules : str or list of str, optional
        Modules Lily never adapts, matched like ``target_modules``.
    modules_to_save, layers_to_transform, layers_pattern
        As in :class:`~lucid.peft.LoraConfig`.

    Examples
    --------
    >>> from lucid.peft import LilyConfig
    >>> config = LilyConfig(r=16, stride_A=2, num_B=2, target_modules=['q_proj', 'v_proj'])
    >>> sorted(config.target_modules), config.peft_type
    (['q_proj', 'v_proj'], <PeftType.LILY: 'LILY'>)
    >>> LilyConfig.from_dict(config.to_dict()) == config
    True

    Notes
    -----
    Lily's config is complete: every field of the reference library's, in its order and with its
    defaults, read and written as ``adapter_config.json``.  A field of the wrong type raises
    ``TypeError``.  ``ValueError`` is raised when ``r`` or ``stride_A`` is not positive, or
    ``num_B`` is below 2.  No tuner applies it yet: ``get_peft_model`` raises
    ``NotImplementedError`` naming LCD-824, which tracks it.
    """

    _field_checks: ClassVar[dict[str, _Check]] = {
        "r": _check_rank,
        "stride_A": _check_rank,
        "num_B": _check_count,
        "scaling": _check_number,
        "init_weights": _check_bool,
    }

    @override
    def _check_fields(self) -> None:
        super()._check_fields()
        if self.num_B < 2:
            raise ValueError("`num_B` must be at least 2 for the router to be trained.")

    r: int = 32
    stride_A: int = 1
    num_B: int = 2
    scaling: float = 1.0
    target_modules: str | list[str] | set[str] | None = None
    exclude_modules: str | list[str] | set[str] | None = None
    modules_to_save: list[str] | None = None
    layers_to_transform: int | list[int] | None = None
    layers_pattern: str | list[str] | None = None
    init_weights: bool = True
