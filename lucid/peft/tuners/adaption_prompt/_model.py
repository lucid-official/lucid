"""``AdaptionPromptModel``: the LLaMA-Adapter tuner — which attention layers are adapted; the rest is the generic tuner's."""

import copy
from collections.abc import Mapping
from typing import ClassVar, override

import lucid.nn as nn
from lucid.peft._config import PeftType, _TargetedConfig
from lucid.peft._slots import _outside_slots
from lucid.peft.tuners._base import _NO_PARAMETERS, _Target
from lucid.peft.tuners._support import _ConfigTuner
from lucid.peft.tuners.adaption_prompt._attention import _AttentionKind
from lucid.peft.tuners.adaption_prompt._config import AdaptionPromptConfig
from lucid.peft.tuners.adaption_prompt._layer import AdaptedAttention


def _as_adaption(config: _TargetedConfig) -> AdaptionPromptConfig:
    # ``_BaseTuner`` has checked it against ``_config_class``.
    assert isinstance(config, AdaptionPromptConfig)
    return config


class AdaptionPromptModel(_ConfigTuner, peft_type=PeftType.ADAPTION_PROMPT):
    r"""A model whose top attention layers also attend to learned prompts, and the tools to take them out again.

    Construction wraps the attention module of each of the top ``adapter_layers`` decoder blocks
    in an :class:`~lucid.peft.tuners.adaption_prompt.AdaptedAttention` and freezes every other
    parameter.  The model is changed in place and kept as ``model``; a public name this wrapper
    lacks is read from it.  ``peft_config`` holds a copy of ``config`` with ``target_modules``
    filled in.

    Any number of adapters live side by side under their own names (``add_adapter``,
    ``set_adapter``, ``delete_adapter``), but one applies at a time, and none can be merged: an
    attention over learned keys is not a change of the weights.

    Parameters
    ----------
    model : nn.Module
        The decoder to adapt; ``model.config.model_type`` must be one the method supports.
    config : AdaptionPromptConfig
        What to adapt.
    adapter_name : str, default="default"
        Name of the adapter, the key in ``adaption_prompt`` and ``adaption_gate``.

    Raises
    ------
    ValueError
        The model type is not supported, ``adapter_len`` or ``adapter_layers`` is unset, or the
        config asks for more layers than the model has.
    NotImplementedError
        The model already holds adapter layers of another method.

    Examples
    --------
    >>> from lucid.models.text.gpt2 import GPT2Config, GPT2Model
    >>> from lucid.peft import AdaptionPromptConfig, AdaptionPromptModel
    >>> gpt = GPT2Model(GPT2Config(num_hidden_layers=3, hidden_size=16, num_attention_heads=2,
    ...                            intermediate_size=32, vocab_size=20))
    >>> tuned = AdaptionPromptModel(gpt, AdaptionPromptConfig(adapter_len=4, adapter_layers=2))
    >>> tuned.targeted_module_names
    ['h.1.attn', 'h.2.attn']
    >>> tuned.unload() is gpt, type(gpt.h[2].attn).__name__
    (True, '_GPT2SelfAttention')
    """

    _layer_types = AdaptedAttention._layer_types
    _config_class: ClassVar[type[_TargetedConfig]] = AdaptionPromptConfig
    _default_parameters: ClassVar[Mapping[str, frozenset[str]]] = _NO_PARAMETERS

    def __init__(
        self, model: nn.Module, config: AdaptionPromptConfig, adapter_name: str = "default"
    ) -> None:
        """Adapt ``model`` in place, building the prompts and gates of ``adapter_name``."""
        super().__init__(model, config, adapter_name)

    @override
    @classmethod
    def _prepare_config(cls, config: _TargetedConfig, model: nn.Module) -> _TargetedConfig:
        stored = _as_adaption(copy.deepcopy(config))
        kind = _AttentionKind.for_model(model)
        if stored.target_modules is None:
            stored.target_modules = kind.target_module
        for name in ("adapter_len", "adapter_layers"):
            if getattr(stored, name) is None:
                raise ValueError(f"`{name}` must be set to build an adaption prompt")
        return stored

    @override
    @classmethod
    def _plan_targets(cls, model: nn.Module, config: _TargetedConfig) -> list[_Target]:
        adaption = _as_adaption(config)
        name, wanted = adaption.target_modules, adaption.adapter_layers
        assert name is not None and wanted is not None  # ``_prepare_config`` filled both
        # The blocks' order in the module tree is their order in the stack; the top ones are last.
        found = [(key, module) for key, module in _outside_slots(model) if key.endswith(f".{name}")]
        if len(found) < wanted:
            raise ValueError(
                f"Config specifies more adapter layers '{wanted}' than the model has "
                f"'{len(found)}'."
            )
        return [cls._target_for(key, module) for key, module in found[len(found) - wanted :]]
