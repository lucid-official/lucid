"""``PromptEmbedding``: prompt tuning's learned part, and ``PromptTuningModel``, its tuner."""

import math
from typing import ClassVar, Protocol, Self, override

import lucid
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter
from lucid.peft._config import PeftType, PromptLearningConfig
from lucid.peft.tuners._prompt._encoder import _Injection, _PromptEncoder
from lucid.peft.tuners._prompt._tuner import _PromptLearningTuner
from lucid.peft.tuners.prompt_tuning._config import PromptTuningConfig, PromptTuningInit


class _TextEncoder(Protocol):
    """What ``PromptEmbedding`` asks of a tokenizer: text to ids."""

    def encode(self, text: str) -> list[int]: ...


class PromptEmbedding(_PromptEncoder):
    r"""The prompt of prompt tuning: one learned embedding per virtual token.

    Parameters
    ----------
    config : PromptTuningConfig
        Sizes (``num_virtual_tokens``, ``num_transformer_submodules`` and
        ``token_dim`` set) and the initialisation.
    word_embeddings : nn.Module
        The model's token table, which ``"SAMPLE_VOCAB"`` and ``"TEXT"``
        copy rows of.
    tokenizer : object with ``encode(text) -> list[int]``, optional
        Turns ``prompt_tuning_init_text`` into ids for ``"TEXT"``.

    Attributes
    ----------
    embedding : nn.Embedding
        ``(num_virtual_tokens * num_transformer_submodules, token_dim)``.

    Raises
    ------
    ValueError
        ``"TEXT"`` initialisation without a tokenizer.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft import PromptTuningConfig
    >>> from lucid.peft.tuners.prompt_tuning import PromptEmbedding
    >>> config = PromptTuningConfig(num_virtual_tokens=3, token_dim=8, num_transformer_submodules=1)
    >>> PromptEmbedding(config, nn.Embedding(10, 8))(lucid.arange(3).long()).shape
    (3, 8)
    """

    injection: ClassVar[_Injection] = _Injection.EMBEDDINGS

    def __init__(
        self,
        config: PromptTuningConfig,
        word_embeddings: nn.Module | None,
        tokenizer: _TextEncoder | None = None,
    ) -> None:
        super().__init__()
        assert config.num_virtual_tokens is not None and config.token_dim is not None
        total = config.num_virtual_tokens * (config.num_transformer_submodules or 1)
        self.embedding = nn.Embedding(total, config.token_dim)
        init = config.prompt_tuning_init
        if config.inference_mode or init == PromptTuningInit.RANDOM:
            return
        if not isinstance(word_embeddings, nn.Embedding):
            raise TypeError(f"{init} initialisation reads the model's nn.Embedding token table")
        device = word_embeddings.weight.device.type
        ids = self._init_ids(config, word_embeddings, tokenizer, total).to(device)
        with lucid.no_grad():
            rows = word_embeddings(ids)
        self.embedding.weight = Parameter(rows.detach().clone().float())

    @staticmethod
    def _init_ids(
        config: PromptTuningConfig, table: nn.Embedding, tokenizer: _TextEncoder | None, total: int
    ) -> Tensor:
        if config.prompt_tuning_init == PromptTuningInit.SAMPLE_VOCAB:
            return lucid.randint(0, table.num_embeddings, (total,)).long()
        if tokenizer is None:
            raise ValueError(
                "prompt_tuning_init='TEXT' needs a tokenizer to turn the text into ids; "
                f"lucid has no loader for tokenizer_name_or_path={config.tokenizer_name_or_path!r}, "
                "so build PromptEmbedding(config, word_embeddings, tokenizer=...) yourself"
            )
        assert config.prompt_tuning_init_text is not None  # the config checked it
        ids = tokenizer.encode(config.prompt_tuning_init_text)
        if not ids:
            raise ValueError("prompt_tuning_init_text tokenizes to nothing")
        return lucid.tensor((ids * math.ceil(total / len(ids)))[:total], dtype=lucid.int64)

    @classmethod
    @override
    def from_config(cls, config: PromptLearningConfig, word_embeddings: nn.Module | None) -> Self:
        assert isinstance(config, PromptTuningConfig)
        return cls(config, word_embeddings)

    @property
    @override
    def stores_prompt(self) -> bool:
        return True

    @override
    def forward(self, indices: Tensor) -> Tensor:
        """Embeddings of the virtual-token ids: ``(*, ...)`` ids to ``(*, ..., token_dim)``."""
        out: Tensor = self.embedding(indices)
        return out


class PromptTuningModel(_PromptLearningTuner, peft_type=PeftType.PROMPT_TUNING):
    """The tuner of prompt tuning: a ``PromptEmbedding`` per adapter, put in front of the input."""

    _config_class: ClassVar[type[PromptLearningConfig]] = PromptTuningConfig
    _encoder_class: ClassVar[type[_PromptEncoder]] = PromptEmbedding
