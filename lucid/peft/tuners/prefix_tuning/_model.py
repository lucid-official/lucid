"""``PrefixEncoder``: prefix tuning's learned part, and ``PrefixTuningModel``, its tuner."""

from typing import ClassVar, Self, override

import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.peft._config import PeftType, PromptLearningConfig
from lucid.peft.tuners._prompt._encoder import _Injection, _PromptEncoder
from lucid.peft.tuners._prompt._tuner import _PromptLearningTuner
from lucid.peft.tuners.prefix_tuning._config import PrefixTuningConfig


class PrefixEncoder(_PromptEncoder):
    r"""The prefix of prefix tuning: key and value vectors for every layer, one row per virtual token.

    A row holds ``2 * num_layers * token_dim`` numbers — the key and the value
    of each layer.  With ``prefix_projection`` (and not ``inference_mode``) the
    rows come from a ``token_dim``-wide embedding through a two-layer MLP;
    otherwise ``embedding`` holds them directly.

    Parameters
    ----------
    config : PrefixTuningConfig
        Sizes (``num_virtual_tokens``, ``num_layers``, ``token_dim``,
        ``encoder_hidden_size`` set) and the projection.

    Attributes
    ----------
    embedding : nn.Embedding
        The rows (or the projection's input).
    transform : nn.Module
        The projection MLP, or ``nn.Identity``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.peft import PrefixTuningConfig
    >>> from lucid.peft.tuners.prefix_tuning import PrefixEncoder
    >>> config = PrefixTuningConfig(num_virtual_tokens=3, token_dim=8, num_layers=2,
    ...                             encoder_hidden_size=8)
    >>> PrefixEncoder(config)(lucid.arange(3).long().reshape(1, 3)).shape
    (1, 3, 32)
    """

    injection: ClassVar[_Injection] = _Injection.KEY_VALUES

    def __init__(self, config: PrefixTuningConfig) -> None:
        super().__init__()
        assert config.num_virtual_tokens is not None and config.num_layers is not None
        assert config.token_dim is not None  # the tuner resolves the sizes
        width = config.num_layers * 2 * config.token_dim
        if config.prefix_projection and not config.inference_mode:
            hidden = config.encoder_hidden_size or config.token_dim
            self.embedding = nn.Embedding(config.num_virtual_tokens, config.token_dim)
            out = nn.Linear(hidden, width)
            self.transform: nn.Module = nn.Sequential(
                nn.Linear(config.token_dim, hidden), nn.Tanh(), out
            )
            if config.init_weights == "zero":
                nn.init.zeros_(out.weight)
                if out.bias is not None:
                    nn.init.zeros_(out.bias)
        else:
            self.embedding = nn.Embedding(config.num_virtual_tokens, width)
            self.transform = nn.Identity()
            if config.init_weights == "zero":
                nn.init.zeros_(self.embedding.weight)

    @classmethod
    @override
    def from_config(cls, config: PromptLearningConfig, word_embeddings: nn.Module | None) -> Self:
        del word_embeddings  # the prefix is not made of token embeddings
        assert isinstance(config, PrefixTuningConfig)
        return cls(config)

    @property
    @override
    def stores_prompt(self) -> bool:
        return isinstance(self.transform, nn.Identity)

    @override
    def forward(self, prefix: Tensor) -> Tensor:
        """Key/value rows of the virtual-token ids: ``(B, V)`` ids to ``(B, V, 2 * num_layers * token_dim)``."""
        out: Tensor = self.transform(self.embedding(prefix))
        return out


class PrefixTuningModel(_PromptLearningTuner, peft_type=PeftType.PREFIX_TUNING):
    """The tuner of prefix tuning: a ``PrefixEncoder`` per adapter, read as every layer's cached keys and values."""

    _config_class: ClassVar[type[PromptLearningConfig]] = PrefixTuningConfig
    _encoder_class: ClassVar[type[_PromptEncoder]] = PrefixEncoder
