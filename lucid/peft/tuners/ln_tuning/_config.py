"""``LNTuningConfig``: which normalisation layers LN tuning makes trainable."""

from dataclasses import dataclass, field
from typing import ClassVar

from lucid.peft._config import PeftType, _TargetedConfig


@dataclass
class LNTuningConfig(_TargetedConfig, peft_type=PeftType.LN_TUNING):
    r"""How :func:`lucid.peft.get_peft_model` adapts a model with LN tuning.

    LN tuning (Zhao et al., 2023, arXiv:2312.11420 — "Tuning LayerNorm in
    Attention") trains the normalisation layers of a model and nothing else:
    each targeted layer gets a copy of itself that trains while the original
    stays frozen.  The method adds no tensor of its own, so it is a few
    :math:`d` parameters per layer; "merging" makes the trained copy the layer.

    Field names, their order, defaults and ``adapter_config.json`` are the reference
    adapter library's.  Unlike the reference's, this
    config also takes ``layers_to_transform`` / ``layers_pattern``, which select layers
    the way ``LoraConfig``'s do; ``adapter_config.json`` holds them only when they are
    set, and ``from_pretrained`` reads files with or without them.

    Parameters
    ----------
    task_type, peft_type, auto_mapping, peft_version, base_model_name_or_path, revision, inference_mode
        As in :class:`~lucid.peft.PeftConfig`.
    target_modules : str or list of str, optional
        The modules to tune (any module, typically ``LayerNorm`` / ``RMSNorm``),
        matched as ``LoraConfig.target_modules`` is.  ``None`` takes the default
        for the base model's ``model_type`` (``gpt2`` and ``t5``;
        ``LN_TUNING_DEFAULT_TARGET_MODULES`` in ``lucid.peft.tuners.ln_tuning._defaults``);
        another model type raises ``ValueError``.
    exclude_modules : str or list of str, optional
        Modules never adapted, matched like ``target_modules``.
    modules_to_save : list of str, optional
        Modules trained in full beside the tuned layers (a classifier head).
    layers_to_transform : int or list of int, optional
        As in :class:`~lucid.peft.LoraConfig`.
    layers_pattern : str or list of str, optional
        As in :class:`~lucid.peft.LoraConfig`.

    Raises
    ------
    TypeError
        A field holds a value of the wrong type.
    ValueError
        A field is out of range or two fields conflict.

    Examples
    --------
    >>> from lucid.peft import LNTuningConfig
    >>> config = LNTuningConfig(target_modules=["ln_1", "ln_2"])
    >>> sorted(config.target_modules), config.peft_type
    (['ln_1', 'ln_2'], <PeftType.LN_TUNING: 'LN_TUNING'>)
    >>> LNTuningConfig.from_dict(config.to_dict()) == config
    True
    """

    _extension_fields: ClassVar[frozenset[str]] = frozenset(
        {"layers_to_transform", "layers_pattern"}
    )

    target_modules: str | list[str] | set[str] | None = None
    exclude_modules: str | list[str] | set[str] | None = None
    modules_to_save: list[str] | None = None
    layers_to_transform: int | list[int] | None = field(default=None, kw_only=True)
    layers_pattern: str | list[str] | None = field(default=None, kw_only=True)
