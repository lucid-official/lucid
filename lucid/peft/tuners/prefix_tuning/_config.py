"""``PrefixTuningConfig``: the shared prompt fields plus the optional projection."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.peft._config import PeftType, PromptLearningConfig, _Check, _check_bool, _optional_count


def _zero_or_none(name: str, value: object) -> str | None:
    if value is None or value == "zero":
        return value
    raise ValueError(f"`{name}` must be 'zero' or None, got {value!r}")


@dataclass(kw_only=True)
class PrefixTuningConfig(PromptLearningConfig, peft_type=PeftType.PREFIX_TUNING):
    r"""How :func:`lucid.peft.get_peft_model` adapts a model with prefix tuning.

    Prefix tuning (Li and Liang, 2021, arXiv:2101.00190) learns
    ``num_virtual_tokens`` key and value vectors for **every** attention layer
    and puts them in front of that layer's own, so the whole network can attend
    to them: ``num_virtual_tokens * 2 * num_layers * token_dim`` numbers.  With
    ``prefix_projection`` the vectors are produced from a smaller embedding by a
    two-layer MLP while training, which the paper found steadier; the file
    still stores the finished vectors.  An encoder-decoder model gets the prefix
    on the decoder's self-attention.

    Field names, defaults and ``adapter_config.json`` are the reference adapter
    library's.  Every field is keyword-only.

    Parameters
    ----------
    num_virtual_tokens : int
        How many positions the prefix has.
    encoder_hidden_size : int, optional
        Width of the projection MLP's hidden layer; ``token_dim`` when omitted.
    prefix_projection : bool, default=False
        Produce the prefix through the projection MLP while training.
    init_weights : {"zero"} or None, default=None
        ``"zero"`` starts the prefix at zeros (the output layer of the
        projection, when there is one) so the keys and values it adds are null.

    Raises
    ------
    TypeError
        A field holds a value of the wrong type.

    Examples
    --------
    >>> from lucid.peft import PrefixTuningConfig
    >>> config = PrefixTuningConfig(num_virtual_tokens=4, prefix_projection=True)
    >>> config.peft_type, config.prefix_projection
    (<PeftType.PREFIX_TUNING: 'PREFIX_TUNING'>, True)
    """

    _field_checks: ClassVar[dict[str, _Check]] = {
        "encoder_hidden_size": _optional_count,
        "prefix_projection": _check_bool,
        "init_weights": _zero_or_none,
    }

    encoder_hidden_size: int | None = None
    prefix_projection: bool = False
    init_weights: str | None = None
