"""``CPTEmbedding``: the prompt of context-aware prompt tuning — token embeddings plus a bounded trainable delta."""

import math
from typing import ClassVar, Self, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter
from lucid.peft._config import PromptLearningConfig
from lucid.peft.tuners._prompt._encoder import _Injection, _PromptEncoder
from lucid.peft.tuners.cpt._config import CPTConfig

#: The radius of a row that must not move: the projection leaves it at (almost) zero.
_MIN_EPSILON = 1e-10
#: Token types (modulo 4) whose delta trains: a template, an input, an output template.
_TRAINABLE_TYPES = (1, 2, 3)
_TEMPLATE_TYPES = (1, 3)
_INPUT_TYPE = 2
#: The width the epsilons are quoted for; they scale with the square root of the model's.
_REFERENCE_WIDTH = 2048


class CPTEmbedding(_PromptEncoder):
    r"""The prompt of CPT: frozen token embeddings plus a trainable delta kept near zero.

    Built to train it holds ``embedding`` (the model's embeddings of ``cpt_token_ids``, frozen)
    and ``delta_embedding`` (zeros).  Each call first projects every delta row onto the ball of
    radius :math:`\epsilon_t` — the input or template epsilon of its token type, times
    :math:`\sqrt{d/2048}` — writing the result back into the parameter (the optimiser steps from
    the projected value, as the reference's does), then returns ``embedding + delta``.  Only the
    deltas of types 1 to 3 (templates and inputs) receive a gradient.  Built for
    ``inference_mode`` it holds ``embedding`` alone, whose rows are the finished prompt.

    Parameters
    ----------
    config : CPTConfig
        The token ids and types, the epsilons, and ``token_dim`` set.
    word_embeddings : nn.Module
        The model's token table; its rows for ``cpt_token_ids`` start the prompt.

    Attributes
    ----------
    embedding : nn.Embedding
        ``(num_virtual_tokens, token_dim)``, never trained.
    delta_embedding : nn.Embedding or None
        The trainable offset; ``None`` for an inference adapter.

    Raises
    ------
    TypeError
        A trainable prompt is built from a table that is not an ``nn.Embedding``.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft import CPTConfig
    >>> from lucid.peft.tuners.cpt import CPTEmbedding
    >>> config = CPTConfig(cpt_token_ids=[1, 2], cpt_tokens_type_mask=[2, 4], token_dim=8,
    ...                    task_type="CAUSAL_LM")
    >>> prompt = CPTEmbedding(config, nn.Embedding(5, 8))
    >>> prompt(lucid.arange(2).long()).shape
    (2, 8)
    >>> [p.requires_grad for p in prompt.parameters()]
    [False, True]
    """

    injection: ClassVar[_Injection] = _Injection.EMBEDDINGS

    delta_embedding: nn.Embedding | None
    update_mask: Tensor
    epsilon: Tensor

    def __init__(self, config: CPTConfig, word_embeddings: nn.Module | None) -> None:
        """Build the frozen embeddings (from ``word_embeddings``) and, to train, the zero delta."""
        super().__init__()
        assert config.cpt_token_ids is not None and config.cpt_tokens_type_mask is not None
        assert config.token_dim is not None
        count = len(config.cpt_token_ids)
        self.embedding = nn.Embedding(count, config.token_dim)
        self.embedding.requires_grad_(False)
        self.delta_embedding = None
        if not config.inference_mode:
            if not isinstance(word_embeddings, nn.Embedding):
                raise TypeError("a CPT prompt starts from the model's nn.Embedding token table")
            ids = lucid.tensor(
                config.cpt_token_ids, dtype=lucid.int64, device=word_embeddings.weight.device.type
            )
            with lucid.no_grad():
                rows = word_embeddings(ids)
            self.embedding.weight = Parameter(rows.detach().clone().float(), requires_grad=False)
            self.delta_embedding = nn.Embedding(count, config.token_dim)
            with lucid.no_grad():
                self.delta_embedding.weight.copy_(lucid.zeros_like(self.delta_embedding.weight))
        types = config.cpt_tokens_type_mask
        update = [float(t > 0 and t % 4 in _TRAINABLE_TYPES) for t in types]
        self.register_buffer(
            "update_mask",
            lucid.tensor(update, dtype=lucid.float32).reshape(-1, 1),
            persistent=False,
        )
        self.register_buffer("epsilon", self._epsilons(config), persistent=False)

    @staticmethod
    def _epsilons(config: CPTConfig) -> Tensor:
        """The projection radius of each prompt token, ``(count, 1)``."""
        assert config.token_dim is not None and config.cpt_tokens_type_mask is not None
        scale = math.sqrt(config.token_dim / _REFERENCE_WIDTH)
        radii: list[float] = []
        for kind in config.cpt_tokens_type_mask:
            if kind > 0 and kind % 4 in _TEMPLATE_TYPES:
                radii.append(config.opt_projection_format_epsilon * scale)
            elif kind > 0 and kind % 4 == _INPUT_TYPE:
                radii.append(config.opt_projection_epsilon * scale)
            else:
                radii.append(_MIN_EPSILON)
        return lucid.tensor(radii, dtype=lucid.float32).reshape(-1, 1)

    @override
    def requires_grad_(self, requires_grad: bool = True) -> Self:
        """Train (or freeze) the delta; the embeddings it offsets stay frozen."""
        super().requires_grad_(requires_grad)
        self.embedding.requires_grad_(False)
        return self

    @classmethod
    @override
    def from_config(cls, config: PromptLearningConfig, word_embeddings: nn.Module | None) -> Self:
        """The prompt for ``config``, started from ``word_embeddings``."""
        assert isinstance(config, CPTConfig)
        return cls(config, word_embeddings)

    @property
    @override
    def stores_prompt(self) -> bool:
        """``True`` for an inference adapter, whose ``embedding`` rows are the prompt."""
        return self.delta_embedding is None

    def project(self) -> None:
        """Pull every delta row back inside its ball and write that into the parameter."""
        delta = self.delta_embedding
        if delta is None:
            return
        with lucid.no_grad():
            weight = delta.weight
            norm = (weight * weight).sum(dim=1, keepdim=True).sqrt()
            # A row inside its ball keeps its value; one outside is rescaled to the ball's radius.
            weight.copy_(weight * (self.epsilon / lucid.maximum(norm, self.epsilon)))

    @override
    def forward(self, indices: Tensor) -> Tensor:
        """The prompt rows ``embedding + delta`` of the virtual-token ids ``indices``."""
        base: Tensor = self.embedding(indices)
        delta = self.delta_embedding
        if delta is None:
            return base
        self.project()
        masked = delta.weight * self.update_mask
        return base + F.embedding(indices, masked)
