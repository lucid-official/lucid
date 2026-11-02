"""``AdaptionPromptConfig``: the LLaMA-Adapter hyper-parameters on top of the shared targeting fields."""

from dataclasses import dataclass, field
from typing import ClassVar

from lucid.peft._config import PeftType, _Check, _optional_count, _optional_str, _TargetedConfig


@dataclass
class AdaptionPromptConfig(_TargetedConfig, peft_type=PeftType.ADAPTION_PROMPT):
    r"""How :func:`lucid.peft.get_peft_model` adapts a decoder with adaption prompts (LLaMA-Adapter).

    LLaMA-Adapter (Zhang et al., 2023, arXiv:2303.16199) freezes the model and gives each of the
    top ``adapter_layers`` attention layers ``adapter_len`` learned prompt vectors.  The layer's
    queries attend to keys and values projected from the prompt, and that attention output is
    added to the layer's own, scaled by a gate that starts at zero, so the adapted model starts as
    the base.

    Field names, their order, defaults and ``adapter_config.json`` are the reference adapter
    library's.  Unlike the reference's, this config also takes ``modules_to_save`` as a trailing
    keyword-only field (modules trained in full beside the prompts, a classification head);
    ``adapter_config.json`` holds it only when it is set.

    Parameters
    ----------
    task_type, peft_type, auto_mapping, peft_version, base_model_name_or_path, revision, inference_mode
        As in :class:`~lucid.peft.PeftConfig`.
    target_modules : str, optional
        The name of the attention module inside each decoder block (``"attn"``); ``None`` takes the
        base model type's.
    adapter_len : int
        How many prompt vectors each adapted layer learns; required by the time the adapter is built.
    adapter_layers : int
        How many attention layers, counted from the top of the stack, are adapted; required then.
    modules_to_save : list of str, optional
        Modules trained in full beside the adapter.

    Raises
    ------
    TypeError
        A field holds a value of the wrong type.
    ValueError
        A field is not positive, or a layer-selection field the method has no use for is set.

    Examples
    --------
    >>> from lucid.peft import AdaptionPromptConfig
    >>> config = AdaptionPromptConfig(adapter_len=10, adapter_layers=2, task_type="CAUSAL_LM")
    >>> config.peft_type, config.adapter_len
    (<PeftType.ADAPTION_PROMPT: 'ADAPTION_PROMPT'>, 10)
    >>> AdaptionPromptConfig.from_dict(config.to_dict()) == config
    True
    """

    _extension_fields: ClassVar[frozenset[str]] = frozenset({"modules_to_save"})
    _field_checks: ClassVar[dict[str, _Check]] = {
        "target_modules": _optional_str,
        "adapter_len": _optional_count,
        "adapter_layers": _optional_count,
    }

    target_modules: str | None = None
    adapter_len: int | None = None
    adapter_layers: int | None = None
    modules_to_save: list[str] | None = field(default=None, kw_only=True)
