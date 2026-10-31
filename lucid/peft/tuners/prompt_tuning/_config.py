"""``PromptTuningConfig``: how the learned prompt is initialised, on top of the shared prompt fields."""

import enum
from dataclasses import dataclass
from typing import ClassVar, override

from lucid.peft._config import PeftType, PromptLearningConfig, _Check, _choice, _optional_str


class PromptTuningInit(enum.StrEnum):
    """How the prompt's rows start: from ``TEXT``, ``SAMPLE_VOCAB`` tokens, or ``RANDOM``."""

    #: From the embeddings of ``prompt_tuning_init_text``'s tokens.
    TEXT = "TEXT"
    #: From the embeddings of tokens drawn at random from the vocabulary.
    SAMPLE_VOCAB = "SAMPLE_VOCAB"
    #: From the embedding table's own random initialisation.
    RANDOM = "RANDOM"


def _optional_dict(name: str, value: object) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError(f"`{name}` must be a dict or None, got {value!r}")
    return {str(k): v for k, v in value.items()}


@dataclass(kw_only=True)
class PromptTuningConfig(PromptLearningConfig, peft_type=PeftType.PROMPT_TUNING):
    r"""How :func:`lucid.peft.get_peft_model` adapts a model with prompt tuning.

    Prompt tuning (Lester et al., 2021, arXiv:2104.08691) freezes the model and
    learns ``num_virtual_tokens`` embedding vectors that are put in front of
    the input embeddings: the only trained parameters are
    ``num_virtual_tokens * token_dim`` numbers, and a task is a file of that size.
    An encoder-decoder model gets them in front of the encoder's input.

    Field names, defaults and ``adapter_config.json`` are the reference adapter
    library's.  Every field is keyword-only.

    Parameters
    ----------
    num_virtual_tokens : int
        How many learned embeddings go in front of the input.
    prompt_tuning_init : PromptTuningInit or str, default="RANDOM"
        ``"RANDOM"`` keeps the embedding table's initialisation; ``"SAMPLE_VOCAB"``
        starts from the embeddings of random vocabulary tokens; ``"TEXT"``
        starts from those of ``prompt_tuning_init_text``'s tokens (cut or
        repeated to ``num_virtual_tokens``).
    prompt_tuning_init_text : str, optional
        The text ``"TEXT"`` starts from; required then.
    tokenizer_name_or_path : str, optional
        Names the tokenizer that turns that text into ids; required for ``"TEXT"``.
    tokenizer_kwargs : dict, optional
        Keyword arguments for loading that tokenizer; only with ``"TEXT"``.

    Raises
    ------
    TypeError
        A field holds a value of the wrong type.
    ValueError
        ``"TEXT"`` without its text or tokenizer, or ``tokenizer_kwargs`` without ``"TEXT"``.

    Examples
    --------
    >>> from lucid.peft import PromptTuningConfig
    >>> config = PromptTuningConfig(num_virtual_tokens=8, task_type="CAUSAL_LM")
    >>> config.peft_type, config.prompt_tuning_init
    (<PeftType.PROMPT_TUNING: 'PROMPT_TUNING'>, <PromptTuningInit.RANDOM: 'RANDOM'>)
    >>> PromptTuningConfig.from_dict(config.to_dict()) == config
    True
    """

    _field_checks: ClassVar[dict[str, _Check]] = {
        "prompt_tuning_init": _choice(tuple(PromptTuningInit)),
        "prompt_tuning_init_text": _optional_str,
        "tokenizer_name_or_path": _optional_str,
        "tokenizer_kwargs": _optional_dict,
    }

    prompt_tuning_init: PromptTuningInit | str = PromptTuningInit.RANDOM
    prompt_tuning_init_text: str | None = None
    tokenizer_name_or_path: str | None = None
    tokenizer_kwargs: dict[str, object] | None = None

    @override
    def _check_fields(self) -> None:
        super()._check_fields()
        text = self.prompt_tuning_init == PromptTuningInit.TEXT
        if text and not self.tokenizer_name_or_path:
            raise ValueError(
                f"When prompt_tuning_init='{PromptTuningInit.TEXT.value}', "
                f"tokenizer_name_or_path can't be {self.tokenizer_name_or_path}."
            )
        if text and self.prompt_tuning_init_text is None:
            raise ValueError(
                f"When prompt_tuning_init='{PromptTuningInit.TEXT.value}', "
                f"prompt_tuning_init_text can't be {self.prompt_tuning_init_text}."
            )
        if self.tokenizer_kwargs and not text:
            raise ValueError(
                "tokenizer_kwargs only valid when using "
                f"prompt_tuning_init='{PromptTuningInit.TEXT.value}'."
            )
