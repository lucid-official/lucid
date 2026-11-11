"""``ShadowConfig``: the ShadowPEFT hyper-parameters on top of the shared targeting fields."""

from dataclasses import dataclass
from typing import ClassVar, override

from lucid.peft._config import (
    PeftType,
    _Check,
    _check_bool,
    _check_dropout,
    _check_number,
    _check_rank,
    _check_str,
    _nullable,
    _TargetedConfig,
)


@dataclass
class ShadowConfig(_TargetedConfig, peft_type=PeftType.SHADOW):
    r"""How :func:`lucid.peft.get_peft_model` adapts a model with ShadowPEFT.

    ShadowPEFT (Li et al., 2026, arXiv:2604.19254 — Shadow Network for Parameter-Efficient Fine-Tuning)
    augments a frozen decoder with a small trainable parallel *shadow* network.  A shadow backbone produces a
    shadow state that rides the base model's decoder loop: at every wrapped block the discrepancy between the base
    hidden states and the shadow state is injected into the block's input through a rank-``r`` bottleneck, and the
    shadow state is advanced by a gated residual update from the block's output.  The adaptation is an
    input-dependent trajectory in layer space, not a weight change, so it cannot be merged into the weights.

    The config is read and written here so a file of this method round-trips, but ``get_peft_model`` refuses it:
    the method needs the base model's decoder loop and key-value cache, and a second backbone built beside it.

    Parameters
    ----------
    task_type, peft_type, auto_mapping, peft_version, base_model_name_or_path, revision, inference_mode
        As in :class:`~lucid.peft.PeftConfig`.
    r : int, default=8
        Rank of the injection bottleneck.
    shadow_alpha : float, default=0.1
        Strength of the correction injected into the block input.
    shadow_dropout : float, default=0.2
        Dropout on the discrepancy signal before the bottleneck, in ``[0, 1)``.
    init_weights : bool, default=True
        Zero-initialise the up-projection, so the injection starts as a no-op.
    shadow_model : str, default="mirror"
        How the shadow backbone is built: ``"mirror"`` copies the base architecture; any other
        string names a model to load.
    shadow_num_hidden_layers : int, optional
        Depth of the built backbone; ``None`` is 1.
    shadow_hidden_size : int, optional
        Hidden size of the backbone; ``None`` is the base's.
    shadow_num_attention_heads : int, optional
        Attention heads of the backbone; ``None`` is the base's.
    shadow_intermediate_size : int, optional
        Feed-forward width of the backbone; ``None`` follows the hidden size.
    share_embeddings : bool, default=True
        Reuse the frozen base input embeddings in the shadow backbone.
    update_hidden_size : int, optional
        Hidden width of the update and gate networks; ``None`` is ``r``.
    auxiliary_loss_weight : float, default=0.05
        Weight of the auxiliary loss; ``0`` disables it.
    target_modules : str or list of str, optional
        The modules ShadowPEFT adapts, matched as ``LoraConfig.target_modules`` is.
    exclude_modules : str or list of str, optional
        Modules ShadowPEFT never adapts, matched like ``target_modules``.
    layers_to_transform, layers_pattern, modules_to_save
        As in :class:`~lucid.peft.LoraConfig`.

    Examples
    --------
    >>> from lucid.peft import ShadowConfig
    >>> config = ShadowConfig(r=4, target_modules=['layers.0'])
    >>> sorted(config.target_modules), config.peft_type
    (['layers.0'], <PeftType.SHADOW: 'SHADOW'>)
    >>> ShadowConfig.from_dict(config.to_dict()) == config
    True

    Notes
    -----
    ShadowPEFT's config is complete: every field of the reference library's, in its order and with
    its defaults, read and written as ``adapter_config.json``.  A field of the wrong type raises
    ``TypeError``.  ``ValueError`` is raised when a named ``shadow_model`` is combined with a
    mirror-only size override.  Its tuner is refused for good: ``get_peft_model`` raises
    ``NotImplementedError`` saying why.
    """

    _field_checks: ClassVar[dict[str, _Check]] = {
        "r": _check_rank,
        "shadow_alpha": _check_number,
        "shadow_dropout": _check_dropout,
        "init_weights": _check_bool,
        "shadow_model": _check_str,
        "shadow_num_hidden_layers": _nullable(_check_rank),
        "shadow_hidden_size": _nullable(_check_rank),
        "shadow_num_attention_heads": _nullable(_check_rank),
        "shadow_intermediate_size": _nullable(_check_rank),
        "share_embeddings": _check_bool,
        "update_hidden_size": _nullable(_check_rank),
        "auxiliary_loss_weight": _check_number,
    }

    @override
    def _check_fields(self) -> None:
        super()._check_fields()
        if self.shadow_model != "mirror":
            overrides = {
                "shadow_num_hidden_layers": self.shadow_num_hidden_layers,
                "shadow_hidden_size": self.shadow_hidden_size,
                "shadow_num_attention_heads": self.shadow_num_attention_heads,
                "shadow_intermediate_size": self.shadow_intermediate_size,
            }
            named = [name for name, value in overrides.items() if value is not None]
            if named:
                raise ValueError(
                    "A named `shadow_model` defines its own architecture; unset these mirror-only "
                    f"overrides: {', '.join(named)}."
                )

    target_modules: str | list[str] | set[str] | None = None
    exclude_modules: str | list[str] | set[str] | None = None
    r: int = 8
    shadow_alpha: float = 0.1
    shadow_dropout: float = 0.2
    init_weights: bool = True
    shadow_model: str = "mirror"
    shadow_num_hidden_layers: int | None = None
    shadow_hidden_size: int | None = None
    shadow_num_attention_heads: int | None = None
    shadow_intermediate_size: int | None = None
    share_embeddings: bool = True
    update_hidden_size: int | None = None
    auxiliary_loss_weight: float = 0.05
    layers_to_transform: int | list[int] | None = None
    layers_pattern: str | list[str] | None = None
    modules_to_save: list[str] | None = None
