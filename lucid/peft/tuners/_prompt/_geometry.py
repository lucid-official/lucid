"""The sizes a prompt is built to, read off the model's config.

Every family names its sizes its own way (``hidden_size`` / ``n_embd`` / ``d_model``), so the
names are listed once here.  A prompt in front of the input embeddings only needs the
hidden size.  A key/value prefix has to match the attention it is cached for — heads times head
width, which a model whose head width is not ``hidden / heads`` (T5-3B attends at four times
its model width) does not have the hidden size for.
"""

from collections.abc import Mapping
from typing import Final

import lucid.nn as nn
from lucid.peft._config import PromptLearningConfig

#: Where a model's config keeps each size, by the names the families use.
_SIZE_NAMES: Final[Mapping[str, tuple[str, ...]]] = {
    "token_dim": ("hidden_size", "n_embd", "d_model"),
    "num_layers": ("num_hidden_layers", "num_layers", "n_layer"),
    "num_attention_heads": (
        "num_attention_heads",
        "n_head",
        "num_heads",
        "encoder_attention_heads",
    ),
}

#: What a config says about its attention when the keys and values are not ``hidden / heads`` wide.
_KEY_VALUE_HEADS: Final = "num_key_value_heads"
_HEAD_DIM: Final = "head_dim"
#: How an encoder-decoder's config counts the decoder's layers, which the prefix goes into.
_DECODER_LAYERS: Final = "num_decoder_layers"


def _first_int(config: object, names: tuple[str, ...]) -> int | None:
    """The first of ``names`` that ``config`` holds as an ``int``, or ``None``."""
    for name in names:
        value = getattr(config, name, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _model_size(model: nn.Module, field: str) -> int:
    """The model's size of kind ``field`` (a key of ``_SIZE_NAMES``); ``ValueError`` if its config has none."""
    value = _first_int(getattr(model, "config", None), _SIZE_NAMES[field])
    if value is None:
        raise ValueError(f"Please specify `{field}` in `peft_config`")
    return value


def _fill_sizes(model: nn.Module, config: PromptLearningConfig, *, key_values: bool) -> None:
    """Fill in ``config``'s sizes the caller left out, from ``model``'s config.

    ``key_values`` is whether the method's prompt is a key/value prefix (``_Injection.KEY_VALUES``),
    which is sized to the attention, not to the hidden size, and has a layer for every layer of the
    stack it is cached for: an encoder-decoder's prefix goes to the decoder, whose layer count its
    config declares separately (``num_decoder_layers``), as T5's does.
    """
    if key_values:
        if config.num_layers is None:
            decoder_layers = _first_int(getattr(model, "config", None), (_DECODER_LAYERS,))
            config.num_layers = (
                decoder_layers if decoder_layers is not None else _model_size(model, "num_layers")
            )
        _fit_key_values(model, config)
        return
    for field in _SIZE_NAMES:
        if getattr(config, field) is None:
            setattr(config, field, _model_size(model, field))


def _fit_key_values(model: nn.Module, config: PromptLearningConfig) -> None:
    """Size a key/value prefix to the attention it is cached for, filling ``token_dim`` and ``num_attention_heads``.

    The prefix holds ``num_attention_heads`` heads of ``token_dim // num_attention_heads`` numbers
    for every key and value, so those two sizes must be the model's attention, not just its hidden
    size.  A model whose config declares its head width (``head_dim``) or its key/value head count
    (``num_key_value_heads``) is held to it: a ``token_dim`` or ``num_attention_heads`` the caller set
    that disagrees raises, naming both widths, instead of failing as a shape error in the first
    forward.  A config that declares neither has ``hidden / heads`` wide heads and its caller's
    numbers are taken as they are.

    Raises
    ------
    ValueError
        The caller's sizes disagree with a declared attention shape, or the hidden size does not
        split into the heads.
    """
    model_config = getattr(model, "config", None)
    declared_heads = _first_int(model_config, (_KEY_VALUE_HEADS,))
    declared_dim = _first_int(model_config, (_HEAD_DIM,))
    if declared_heads is None and declared_dim is None:
        heads = config.num_attention_heads or _model_size(model, "num_attention_heads")
        if config.token_dim is None:
            hidden = _model_size(model, "token_dim")
            if hidden % heads:
                raise ValueError(
                    f"the hidden size {hidden} does not split into {heads} attention heads; "
                    "set `token_dim` (heads x head width) and `num_attention_heads` in `peft_config`"
                )
            config.token_dim = hidden
        config.num_attention_heads = heads
        return
    heads = declared_heads or _model_size(model, "num_attention_heads")
    head_dim = declared_dim
    if head_dim is None:  # grouped heads: each is as wide as a full head
        head_dim = _model_size(model, "token_dim") // _model_size(model, "num_attention_heads")
    width = heads * head_dim
    given = (config.num_attention_heads or heads, config.token_dim or width)
    if given != (heads, width):
        raise ValueError(
            f"a key/value prefix must match the attention it is cached for: the model has "
            f"{heads} heads of {head_dim} ({width} wide), but `peft_config` has "
            f"num_attention_heads={given[0]}, token_dim={given[1]}; leave them unset"
        )
    config.num_attention_heads, config.token_dim = heads, width
