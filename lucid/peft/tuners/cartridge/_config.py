"""``CartridgeConfig``: the Cartridge hyper-parameters on top of the shared prompt fields."""

from dataclasses import dataclass
from typing import ClassVar, override

from lucid.peft._config import PeftType, _Check, _check_count, PromptLearningConfig


@dataclass
class CartridgeConfig(PromptLearningConfig, peft_type=PeftType.CARTRIDGE):
    r"""How :func:`lucid.peft.get_peft_model` adapts a model with Cartridge.

    A Cartridge (Eyuboglu et al., 2025, arXiv:2506.06266 — Cartridges: Lightweight and general-purpose long
    context representations via self-study) is a trained key-value cache.  ``num_virtual_tokens`` learned
    key/value pairs per attention layer are put in front of the model's cache, in place of a long document the
    model would otherwise read at every query.  The first ``num_frozen_tokens`` of them stay fixed; the paper
    recommends freezing one as an attention sink.  The cache is laid out as in prefix tuning, and starting it from
    text or from a past cache is a separate step.

    Parameters
    ----------
    task_type, peft_type, auto_mapping, peft_version, base_model_name_or_path, revision, inference_mode
        As in :class:`~lucid.peft.PeftConfig`.
    num_virtual_tokens, token_dim, num_transformer_submodules, num_attention_heads, num_layers, modules_to_save
        As in :class:`~lucid.peft.PromptLearningConfig`.
    num_frozen_tokens : int, default=1
        Initial virtual tokens that receive no gradient; not more than ``num_virtual_tokens``.

    Examples
    --------
    >>> from lucid.peft import CartridgeConfig
    >>> config = CartridgeConfig(num_virtual_tokens=16, num_frozen_tokens=2, task_type='CAUSAL_LM')
    >>> config.num_virtual_tokens, config.num_frozen_tokens, config.peft_type
    (16, 2, <PeftType.CARTRIDGE: 'CARTRIDGE'>)
    >>> CartridgeConfig.from_dict(config.to_dict()) == config
    True

    Notes
    -----
    Cartridge's config is complete: every field of the reference library's, in its order and with
    its defaults, read and written as ``adapter_config.json``.  A field of the wrong type raises
    ``TypeError``.  ``ValueError`` is raised when ``num_frozen_tokens`` is negative or exceeds
    ``num_virtual_tokens``.  No tuner applies it yet: ``get_peft_model`` raises
    ``NotImplementedError`` naming LCD-826, which tracks it.
    """

    _field_checks: ClassVar[dict[str, _Check]] = {"num_frozen_tokens": _check_count}

    @override
    def _check_fields(self) -> None:
        super()._check_fields()
        if self.num_virtual_tokens is not None and self.num_frozen_tokens > self.num_virtual_tokens:
            raise ValueError(
                "`num_frozen_tokens` must be <= `num_virtual_tokens`, got "
                f"num_frozen_tokens={self.num_frozen_tokens} and num_virtual_tokens={self.num_virtual_tokens}."
            )

    num_frozen_tokens: int = 1
