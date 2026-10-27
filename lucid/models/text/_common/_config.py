"""Shared base config for ``lucid.models.text`` families.

``LanguageModelConfig`` sits between :class:`lucid.models._base.ModelConfig`
(persistence + ``model_type`` registry hook) and the family-specific configs
(``BERTConfig``, ``GPT2Config``, ``RoFormerConfig``, …).  It collects the
fields that every text model needs so the family configs only have to add
their unique knobs.

Fields documented here are the **lowest common denominator** of HuggingFace
Transformers config classes.  Adding a new field here should be a deliberate
decision — if only one family uses it, keep it in that family's config.
"""

from dataclasses import dataclass
from typing import ClassVar, Literal

from lucid.models._base import ModelConfig

# Activation alias accepted by every text family — narrower than the full
# nn.functional surface so misspellings raise a Literal error early.
TextActivation = Literal["gelu", "gelu_new", "relu", "silu", "swish"]


@dataclass(frozen=True)
class LanguageModelConfig(ModelConfig):
    """Shared base config for transformer-based text models.

    Args:
        vocab_size: Tokenizer vocabulary size.  The token / output embedding
            matrices both have this many rows.
        hidden_size: Transformer hidden dimension (``d_model``).
        num_hidden_layers: Number of stacked transformer blocks.
        num_attention_heads: Multi-head attention head count; ``hidden_size``
            must be divisible by this.
        intermediate_size: FFN inner dimension (4× hidden_size by paper
            convention, but configurable per family).
        hidden_act: FFN activation; one of ``gelu / gelu_new / relu / silu /
            swish``.
        max_position_embeddings: Maximum sequence length supported by the
            absolute / learned positional embedding lookup.  Models that use
            rotary or relative position embeddings still respect this as a
            soft upper bound on training-time sequence length.
        pad_token_id: Token id used for padding, or ``None`` if the model
            does not pad (typical for GPT-style decoders).  ``forward``
            implementations should mask attention away from pad positions.
        bos_token_id: Begin-of-sequence token id, used by
            :class:`CausalLMMixin` to seed generation.  ``None`` when the
            tokenizer does not define one.
        eos_token_id: End-of-sequence token id, which stops generation.
            ``None`` when the tokenizer does not define one.
        hidden_dropout: Dropout applied after embeddings and inside each
            transformer block's residual path.
        attention_dropout: Dropout applied to the attention probabilities.
        initializer_range: Std-dev of the truncated normal initialiser for
            ``Linear``/``Embedding`` weights (HF convention is 0.02).
        layer_norm_eps: Numerical stabiliser for :class:`nn.LayerNorm` —
            every text family in HF uses 1e-12 by default.
        tie_word_embeddings: If True, the LM-head decoder weight is *the
            same parameter* as the input token embedding (BERT MLM head,
            GPT-2 LM head).  If False, the two are independent (GPT-1).
        use_cache: Decoder models can cache past key/value tensors for
            ``generate()``.  This flag toggles cache emission in
            ``forward``; runtime can still override it per-call.

    Examples
    --------
    >>> from lucid.models.text._common._config import LanguageModelConfig
    >>> cfg = LanguageModelConfig()
    >>> cfg.model_type
    'language_model'
    >>> cfg.hidden_size, cfg.vocab_size
    (768, 30522)
    """

    model_type: ClassVar[str] = "language_model"

    # Vocabulary / dimensions
    vocab_size: int = 30_522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3_072
    hidden_act: TextActivation = "gelu"

    # Positional / token ids
    max_position_embeddings: int = 512
    pad_token_id: int | None = 0
    bos_token_id: int | None = None
    eos_token_id: int | None = None

    # Regularisation
    hidden_dropout: float = 0.1
    attention_dropout: float = 0.1

    # Init / numerics
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12

    # Embedding / cache behaviour
    tie_word_embeddings: bool = True
    use_cache: bool = True

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.num_hidden_layers <= 0:
            raise ValueError(f"num_hidden_layers must be positive, got {self.num_hidden_layers}")
        if not 0.0 <= self.hidden_dropout < 1.0:
            raise ValueError(f"hidden_dropout must be in [0, 1), got {self.hidden_dropout}")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError(f"attention_dropout must be in [0, 1), got {self.attention_dropout}")
        if self.layer_norm_eps <= 0.0:
            raise ValueError(f"layer_norm_eps must be positive, got {self.layer_norm_eps}")


# Common config field aliases — keeps the import surface small.
__all__ = ["LanguageModelConfig", "TextActivation"]
