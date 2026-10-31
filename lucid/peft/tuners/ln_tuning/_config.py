"""``LNTuningConfig``: which normalisation layers LN tuning makes trainable."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.peft._config import PeftType, _TargetedConfig


@dataclass(kw_only=True)
class LNTuningConfig(_TargetedConfig, peft_type=PeftType.LN_TUNING):
    r"""How :func:`lucid.peft.get_peft_model` adapts a model with LN tuning.

    LN tuning (Zhao et al., 2023, arXiv:2312.11420 — "Tuning LayerNorm in
    Attention") trains the normalisation layers of a model and nothing else:
    each targeted layer gets a copy of itself that trains while the original
    stays frozen.  The method adds no tensor of its own, so it is a few
    :math:`d` parameters per layer; "merging" makes the trained copy the layer.

    Field names, defaults and ``adapter_config.json`` are the reference
    adapter library's.  Every field is keyword-only.  Unlike the reference's, this
    config also takes ``layers_to_transform`` / ``layers_pattern``, which select layers
    the way ``LoraConfig``'s do; ``adapter_config.json`` holds them only when they are
    set, and ``from_pretrained`` reads files with or without them.

    Parameters
    ----------
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
