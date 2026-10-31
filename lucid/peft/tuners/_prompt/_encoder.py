"""What a prompt-learning method learns: the module that turns virtual-token ids into a prompt."""

import enum
from abc import ABC, abstractmethod
from typing import ClassVar, Self

import lucid
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.peft._config import PromptLearningConfig


class _Injection(enum.StrEnum):
    """Where a method's prompt enters the model."""

    #: As rows in front of the input embeddings (prompt tuning, p-tuning).
    EMBEDDINGS = "embeddings"
    #: As keys and values in front of every attention layer's (prefix tuning).
    KEY_VALUES = "key_values"


class _PromptEncoder(nn.Module, ABC):
    """The learned part of one prompt-learning adapter.

    A subclass owns an ``embedding`` of one row per virtual token and
    whatever maps its rows to the prompt (nothing, a projection, an MLP or
    LSTM).  Built for a trainable adapter it holds the whole map; built for
    an adapter that is only used (``inference_mode``) it holds ``embedding``
    alone, whose rows *are* the prompt — which is also what an adapter file
    stores, so a file written from a trainable adapter holds the mapped
    prompt and loads into the inference form.
    """

    #: Where this method's prompt enters the model.
    injection: ClassVar[_Injection]

    embedding: nn.Embedding

    @classmethod
    @abstractmethod
    def from_config(cls, config: PromptLearningConfig, word_embeddings: nn.Module | None) -> Self:
        """The encoder for ``config``; ``word_embeddings`` is the model's token table, when the method reads it."""

    @property
    @abstractmethod
    def stores_prompt(self) -> bool:
        """Whether ``embedding.weight`` is the prompt itself (nothing maps it)."""

    @property
    def rows(self) -> int:
        """How many virtual tokens the encoder produces."""
        return int(self.embedding.num_embeddings)

    def prompt_embeddings(self) -> Tensor:
        """The prompt as an adapter file stores it: ``(rows, width)``, one row per virtual token."""
        ids = lucid.arange(self.rows, device=self.embedding.weight.device.type).long()
        prompt: Tensor = self(ids.reshape(1, self.rows))[0]
        return prompt.detach()
