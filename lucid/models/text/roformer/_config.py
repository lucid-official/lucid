"""RoFormer configuration (Su et al., 2021).

Paper: "RoFormer: Enhanced Transformer with Rotary Position Embedding".
RoFormer is BERT with absolute position embeddings replaced by rotary
position embedding (RoPE) inside every attention layer.  The config
therefore mirrors BERT's, minus the absolute ``position_embedding_type``
and plus the RoPE-specific ``rotary_base``.
"""

from dataclasses import dataclass
from typing import ClassVar, Literal, override

from lucid.models._meta import model_family_meta
from lucid.models.text._common._config import LanguageModelConfig


@model_family_meta(
    canonical_name="RoFormer",
    citation=(
        'Su, Jianlin, et al. "RoFormer: Enhanced Transformer with '
        'Rotary Position Embedding." Neurocomputing, vol. 568, 2024, '
        "article 127063."
    ),
    theory=r"""
    RoFormer replaces BERT's absolute (learned) position embedding with
    **Rotary Position Embedding (RoPE)** — a position-encoding scheme
    that injects relative-position information **multiplicatively**
    inside attention rather than additively at the input.

    Consider a query/key vector :math:`x \in \mathbb{R}^{d}` at position
    :math:`m`.  RoPE groups the :math:`d` features into :math:`d/2`
    complex coordinates and rotates the :math:`i`-th pair
    :math:`(x_{2i}, x_{2i+1})` by angle :math:`m \theta_i`, where

    .. math::

        \theta_i = \text{base}^{-2 i / d},
        \qquad i = 0, 1, \dots, d/2 - 1,

    and ``base`` (default :math:`10\,000`) controls the spectrum of
    rotational frequencies.  In complex form
    :math:`z_i = x_{2i} + j\, x_{2i+1}`, the rotation is just
    :math:`z_i \mapsto z_i\, e^{j m \theta_i}`.

    The crucial property is that the **attention dot product becomes a
    function of the relative offset** :math:`m - n` only:

    .. math::

        \langle R_m\, q,\; R_n\, k \rangle
            = \operatorname{Re}\!\sum_i z_i^{(q)} \overline{z_i^{(k)}}\,
              e^{j (m - n) \theta_i}
            = f_{\text{rel}}(q, k, m - n),

    so RoPE encodes relative positions without any extra parameters and
    without an additive bias term.  This grants three practical
    benefits: (i) it generalises to **longer sequences than seen during
    training** (with mild quality decay), (ii) it composes naturally
    with linear attention variants because it acts on :math:`Q, K`
    before the softmax, and (iii) it requires the per-head dimension
    :math:`d_{\text{head}} = d_{\text{model}} / h` to be even — enforced
    by ``__post_init__`` above.  RoFormer otherwise mirrors BERT-base
    (:math:`L=12`, :math:`H=768`, :math:`A=12`) and is the de-facto
    position-encoding choice for modern LLaMA-/Mistral-style LLMs.
    """,
)
@dataclass(frozen=True)
class RoFormerConfig(LanguageModelConfig):
    """Configuration for every RoFormer variant.

    Args:
        embedding_size: Width of the word and token-type tables when the
            checkpoint factorises the embedding.  ``None`` — the default —
            ties them to ``hidden_size``; when the two differ the tables
            and their LayerNorm are sized here and an
            ``embeddings_project`` linear maps up to ``hidden_size``.
        rotary_base: Frequency base ``θ_0`` for the rotary embedding
            (``θ_i = base ** (-2 i / d_head)``).  10000.0 per the paper.
        rotary_value: Also rotate ``V``, not only ``Q`` and ``K``.  Off by
            default because the paper's Eq. 16 rotates the two, and every
            released checkpoint was trained that way; the switch exists
            because the reference implementation carries it.
        type_vocab_size: Segment-id vocabulary (kept for BERT-parity even
            though RoFormer fine-tunes typically feed a single segment).
        position_embedding_type: Always ``"rotary"`` here — kept as a literal
            for forward compat with future variants (NTK-aware scaling, etc.).
        num_labels: Number of classes the sequence-classification head
            projects to.
        classifier_dropout: Dropout applied before that head.

    Examples
    --------
    >>> from lucid.models.text.roformer._config import RoFormerConfig
    >>> cfg = RoFormerConfig()
    >>> cfg.model_type
    'roformer'
    >>> cfg.hidden_size, cfg.vocab_size
    (768, 50000)
    """

    model_type: ClassVar[str] = "roformer"

    # RoFormer-base in the paper: L=12, H=768, A=12, intermediate=3072.
    vocab_size: int = 50_000
    hidden_size: int = 768
    # RoFormer checkpoints may factorise the embedding: the reference takes
    # ``embedding_size=None`` (falling back to hidden_size) and, when the two
    # differ, sizes the word/token-type tables and their LayerNorm at
    # ``embedding_size`` and adds an ``embeddings_project`` linear up to
    # ``hidden_size``.  Without the field such a checkpoint has no
    # representation here at all.
    embedding_size: int | None = None
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3_072
    max_position_embeddings: int = 1_536  # paper extends BERT's 512
    pad_token_id: int | None = 0

    # RoPE-specific.
    rotary_base: float = 10_000.0
    # The reference implementation carries a ``rotary_value`` switch that also
    # rotates V.  The paper's Eq. 16 rotates Q and K only, and every released
    # checkpoint was trained that way, so this defaults off.
    rotary_value: bool = False
    position_embedding_type: Literal["rotary"] = "rotary"

    # Inherited-from-BERT extras.
    type_vocab_size: int = 2
    num_labels: int = 2
    classifier_dropout: float | None = None

    @override
    def __post_init__(self) -> None:
        super().__post_init__()
        head_dim = self.hidden_size // self.num_attention_heads
        if head_dim % 2 != 0:
            raise ValueError(
                f"RoFormer requires an even head_dim (hidden_size / num_attention_heads); "
                f"got head_dim={head_dim}."
            )
        if self.rotary_base <= 1.0:
            raise ValueError(
                f"rotary_base must be > 1 for RoPE to make sense, got {self.rotary_base}"
            )
        if self.type_vocab_size <= 0:
            raise ValueError(f"type_vocab_size must be positive, got {self.type_vocab_size}")
        if self.num_labels <= 0:
            raise ValueError(f"num_labels must be positive, got {self.num_labels}")
        if self.classifier_dropout is not None and not (0.0 <= self.classifier_dropout < 1.0):
            raise ValueError(
                "classifier_dropout must be in [0, 1) or None, got " f"{self.classifier_dropout}"
            )
