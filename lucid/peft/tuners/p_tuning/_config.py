"""``PromptEncoderConfig``: the shared prompt fields plus the encoder that reparameterises the prompt."""

import enum
from dataclasses import dataclass
from typing import ClassVar

from lucid.peft._config import (
    PeftType,
    PromptLearningConfig,
    _Check,
    _check_dropout,
    _check_rank,
    _choice,
    _optional_count,
)


class PromptEncoderReparameterizationType(enum.StrEnum):
    """What turns the virtual tokens' embeddings into the prompt: an ``MLP`` or an ``LSTM``."""

    MLP = "MLP"
    LSTM = "LSTM"


@dataclass(kw_only=True)
class PromptEncoderConfig(PromptLearningConfig, peft_type=PeftType.P_TUNING):
    r"""How :func:`lucid.peft.get_peft_model` adapts a model with p-tuning.

    P-tuning (Liu et al., 2021, arXiv:2103.10385) puts ``num_virtual_tokens``
    learned embeddings in front of the input like prompt tuning, but produces
    them through an encoder — an MLP, or a bidirectional LSTM followed by one —
    so the tokens depend on each other.  The encoder is only used while
    training: the file stores the prompt it produced.

    Field names, defaults and ``adapter_config.json`` are the reference adapter
    library's.  Every field is keyword-only.

    Parameters
    ----------
    num_virtual_tokens : int
        How many learned embeddings go in front of the input.
    encoder_reparameterization_type : {"MLP", "LSTM"}, default="MLP"
        The encoder.
    encoder_hidden_size : int, optional
        Its hidden width; ``token_dim`` when omitted.
    encoder_num_layers : int, default=2
        LSTM layers (an MLP always has two hidden layers; a different value warns).
    encoder_dropout : float, default=0.0
        LSTM dropout between layers.

    Raises
    ------
    TypeError
        A field holds a value of the wrong type.

    Examples
    --------
    >>> from lucid.peft import PromptEncoderConfig
    >>> config = PromptEncoderConfig(num_virtual_tokens=4, encoder_reparameterization_type="LSTM")
    >>> config.peft_type, config.encoder_reparameterization_type
    (<PeftType.P_TUNING: 'P_TUNING'>, <PromptEncoderReparameterizationType.LSTM: 'LSTM'>)
    """

    _field_checks: ClassVar[dict[str, _Check]] = {
        "encoder_reparameterization_type": _choice(tuple(PromptEncoderReparameterizationType)),
        "encoder_hidden_size": _optional_count,
        "encoder_num_layers": _check_rank,
        "encoder_dropout": _check_dropout,
    }

    encoder_reparameterization_type: PromptEncoderReparameterizationType | str = (
        PromptEncoderReparameterizationType.MLP
    )
    encoder_hidden_size: int | None = None
    encoder_num_layers: int = 2
    encoder_dropout: float = 0.0
