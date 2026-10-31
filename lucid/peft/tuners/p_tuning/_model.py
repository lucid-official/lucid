"""``PromptEncoder``: p-tuning's learned part, and ``PromptEncoderModel``, its tuner."""

import warnings
from typing import ClassVar, Self, override

import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.peft._config import PeftType, PromptLearningConfig
from lucid.peft.tuners._prompt._encoder import _Injection, _PromptEncoder
from lucid.peft.tuners._prompt._tuner import _PromptLearningTuner
from lucid.peft.tuners.p_tuning._config import (
    PromptEncoderConfig,
    PromptEncoderReparameterizationType,
)

_MLP_HIDDEN_LAYERS = 2


class PromptEncoder(_PromptEncoder):
    r"""The prompt of p-tuning: virtual-token embeddings through an MLP, or a bidirectional LSTM and an MLP.

    Built for a trainable adapter it holds the whole encoder; built with
    ``inference_mode`` it holds ``embedding`` alone, whose rows are then the
    prompt (as an adapter file stores it).

    Parameters
    ----------
    config : PromptEncoderConfig
        Sizes (``num_virtual_tokens``, ``num_transformer_submodules``,
        ``token_dim``, ``encoder_hidden_size`` set) and the encoder's type.

    Attributes
    ----------
    embedding : nn.Embedding
        ``(num_virtual_tokens * num_transformer_submodules, token_dim)``.
    mlp_head : nn.Module
        The MLP, or ``nn.Identity``.
    lstm_head : nn.LSTM or None
        The LSTM of the ``"LSTM"`` type.

    Examples
    --------
    >>> import lucid
    >>> from lucid.peft import PromptEncoderConfig
    >>> from lucid.peft.tuners.p_tuning import PromptEncoder
    >>> config = PromptEncoderConfig(num_virtual_tokens=3, token_dim=8, encoder_hidden_size=16,
    ...                              num_transformer_submodules=1)
    >>> PromptEncoder(config)(lucid.arange(3).long().reshape(1, 3)).shape
    (1, 3, 8)
    """

    injection: ClassVar[_Injection] = _Injection.EMBEDDINGS

    def __init__(self, config: PromptEncoderConfig) -> None:
        super().__init__()
        assert config.num_virtual_tokens is not None and config.token_dim is not None
        dim = config.token_dim
        hidden = config.encoder_hidden_size or dim
        kind = config.encoder_reparameterization_type
        total = config.num_virtual_tokens * (config.num_transformer_submodules or 1)
        self.embedding = nn.Embedding(total, dim)
        self.lstm_head: nn.LSTM | None = None
        self.mlp_head: nn.Module = nn.Identity()
        if config.inference_mode:
            return
        if kind == PromptEncoderReparameterizationType.LSTM:
            self.lstm_head = nn.LSTM(
                input_size=dim,
                hidden_size=hidden,
                num_layers=config.encoder_num_layers,
                dropout=config.encoder_dropout,
                bidirectional=True,
                batch_first=True,
            )
            self.mlp_head = nn.Sequential(
                nn.Linear(hidden * 2, hidden * 2), nn.ReLU(), nn.Linear(hidden * 2, dim)
            )
        else:
            if config.encoder_num_layers != _MLP_HIDDEN_LAYERS:
                warnings.warn(
                    f"for {kind}, the argument `encoder_num_layers` is ignored. "
                    f"Exactly {_MLP_HIDDEN_LAYERS} MLP layers are used.",
                    stacklevel=3,
                )
            self.mlp_head = nn.Sequential(
                nn.Linear(dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, dim),
            )

    @classmethod
    @override
    def from_config(cls, config: PromptLearningConfig, word_embeddings: nn.Module | None) -> Self:
        del word_embeddings  # the encoder makes its own embeddings
        assert isinstance(config, PromptEncoderConfig)
        return cls(config)

    @property
    @override
    def stores_prompt(self) -> bool:
        return isinstance(self.mlp_head, nn.Identity)

    @override
    def forward(self, indices: Tensor) -> Tensor:
        """Prompt rows of the virtual-token ids: ``(B, total)`` ids to ``(B, total, token_dim)``."""
        rows: Tensor = self.embedding(indices)
        if self.lstm_head is not None:
            sequence = self.lstm_head(rows)[0]
            assert isinstance(sequence, Tensor)  # a padded batch in, a padded batch out
            rows = sequence
        out: Tensor = self.mlp_head(rows)
        return out


class PromptEncoderModel(_PromptLearningTuner, peft_type=PeftType.P_TUNING):
    """The tuner of p-tuning: a ``PromptEncoder`` per adapter, whose prompt goes in front of the input."""

    _config_class: ClassVar[type[PromptLearningConfig]] = PromptEncoderConfig
    _encoder_class: ClassVar[type[_PromptEncoder]] = PromptEncoder
