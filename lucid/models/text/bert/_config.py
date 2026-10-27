"""BERT configuration (Devlin et al., 2018).

Paper: "BERT: Pre-training of Deep Bidirectional Transformers for Language
Understanding".  Field set mirrors the HuggingFace ``BERTConfig`` so existing
tokenizers / checkpoints map 1-to-1 against Lucid weights.
"""

from dataclasses import dataclass
from typing import ClassVar, Literal, override

from lucid.models._meta import model_family_meta
from lucid.models.text._common._config import LanguageModelConfig


@model_family_meta(
    canonical_name="BERT",
    citation=(
        'Devlin, Jacob, et al. "BERT: Pre-training of Deep '
        'Bidirectional Transformers for Language Understanding." '
        "Proceedings of NAACL-HLT, 2019, pp. 4171–4186."
    ),
    theory=r"""
    BERT — *Bidirectional Encoder Representations from Transformers* —
    is a transformer **encoder** pre-trained on two unsupervised
    objectives over raw text:

    1. **Masked Language Modelling (MLM).**  A random 15% of input
       tokens are replaced with a ``[MASK]`` placeholder; the model
       must predict the original token from its bidirectional
       context.  Formally, for masked positions :math:`\mathcal{M}`,

       .. math::

           \mathcal{L}_{\text{MLM}}
               = -\,\mathbb{E}_{x}\!\left[\;
                   \sum_{i \in \mathcal{M}}
                       \log p_\theta\!\bigl(x_i \mid x_{\setminus \mathcal{M}}\bigr)
               \right].

       Unlike the left-to-right LMs that preceded it, every token's
       prediction conditions on **both** left and right context — hence
       *bidirectional*.

    2. **Next-Sentence Prediction (NSP).**  Two segments :math:`A` and
       :math:`B` are concatenated with separator tokens; the model
       reads the pooled ``[CLS]`` representation and predicts whether
       :math:`B` was the actual next sentence in the corpus or a
       random other sentence.  This objective gives the model a
       coarse notion of inter-sentence coherence used by downstream
       sentence-pair tasks (QA, NLI).

    After pre-training on BookCorpus + English Wikipedia, the same
    encoder is fine-tuned on individual downstream tasks by adding a
    thin task-specific head on top of either the ``[CLS]`` token
    representation (classification) or all per-token representations
    (tagging, span prediction).  This "pre-train once, fine-tune
    everywhere" recipe defined the post-2018 NLP landscape and set
    new state-of-the-art on the GLUE, SQuAD, and SWAG benchmarks at
    publication time.
    """,
)
@dataclass(frozen=True)
class BERTConfig(LanguageModelConfig):
    r"""Configuration dataclass for every BERT variant.

    Frozen dataclass inheriting all common text-model fields from
    :class:`LanguageModelConfig` (``hidden_act``, ``hidden_dropout``,
    ``attention_dropout``, ``layer_norm_eps``, ``initializer_range``,
    ``tie_word_embeddings``, ...) and adding the two knobs unique to the BERT
    family.  Field names and defaults mirror the reference-framework
    ``BERTConfig`` so existing tokenizers and checkpoints map one-to-one onto
    Lucid weights.  Defaults match the original ``bert-base-uncased`` release
    (L=12, H=768, A=12, 30,522 WordPiece vocab) from Devlin et al., 2018.

    Parameters
    ----------
    vocab_size : int, default=30522
        Size of the WordPiece token vocabulary.  Both BERT-Base and BERT-Large
        use 30,522 cased/uncased WordPieces from Devlin et al. 2018.
    hidden_size : int, default=768
        Dimensionality :math:`H` of every transformer hidden state.  Must be
        divisible by ``num_attention_heads``.
    num_hidden_layers : int, default=12
        Number :math:`L` of transformer encoder layers stacked in the trunk.
    num_attention_heads : int, default=12
        Number :math:`A` of parallel attention heads per layer.  Per-head
        dimension is ``hidden_size // num_attention_heads``.
    intermediate_size : int, default=3072
        Width of the position-wise feed-forward inner projection (typically
        ``4 * hidden_size``).
    max_position_embeddings : int, default=512
        Maximum sequence length supported by the absolute position-embedding
        table.  Inputs longer than this raise a shape error at lookup.
    pad_token_id : int or None, default=0
        Token id treated as padding.  When set, the ``Embedding`` layer zeroes
        out that row (no gradient).
    type_vocab_size : int, default=2
        Size of the segment-id vocabulary used by the next-sentence objective.
        ``2`` corresponds to the canonical BERT (segment A / B) setup;
        single-segment fine-tunes still use the full table but only ever feed
        zeros.
    position_embedding_type : {"absolute"}, default="absolute"
        Position encoding flavour.  ``"absolute"`` is the only one shipped
        today — relative-position variants (T5-style, RoFormer-style) live in
        their own families.
    num_labels : int, default=2
        Number of output classes for downstream classification heads
        (``BERTForSequenceClassification`` / ``BERTForTokenClassification``).
    classifier_dropout : float or None, default=None
        Optional dropout probability inserted before the task-specific
        classifier.  When ``None``, the head falls back to ``hidden_dropout``.

    Attributes
    ----------
    model_type : str
        Class-level identifier ``"bert"`` used by the model registry to look
        up the matching factory at load time.

    Notes
    -----
    Reference: Devlin, Chang, Lee, and Toutanova, *"BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding"*, NAACL 2019
    (arXiv:1810.04805).

    Validation runs in ``__post_init__`` and raises ``ValueError`` for
    ``type_vocab_size <= 0``, ``num_labels <= 0``, or
    ``classifier_dropout`` outside :math:`[0, 1)`.

    Examples
    --------
    >>> from lucid.models.text.bert import BERTConfig
    >>> cfg = BERTConfig()          # BERT-Base defaults
    >>> cfg.hidden_size, cfg.num_hidden_layers, cfg.num_attention_heads
    (768, 12, 12)
    >>> large = BERTConfig(
    ...     hidden_size=1024,
    ...     num_hidden_layers=24,
    ...     num_attention_heads=16,
    ...     intermediate_size=4096,
    ... )
    >>> large.vocab_size
    30522
    """

    model_type: ClassVar[str] = "bert"

    # BERT defaults match ``bert-base-uncased`` for vocab + layer counts.
    vocab_size: int = 30_522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3_072
    max_position_embeddings: int = 512
    pad_token_id: int | None = 0

    # BERT-specific
    type_vocab_size: int = 2
    position_embedding_type: Literal["absolute"] = "absolute"

    # Downstream task knobs (used by the ``For*`` heads).
    num_labels: int = 2
    classifier_dropout: float | None = None  # falls back to ``hidden_dropout``

    @override
    def __post_init__(self) -> None:
        super().__post_init__()
        if self.type_vocab_size <= 0:
            raise ValueError(f"type_vocab_size must be positive, got {self.type_vocab_size}")
        if self.num_labels <= 0:
            raise ValueError(f"num_labels must be positive, got {self.num_labels}")
        if self.classifier_dropout is not None and not (0.0 <= self.classifier_dropout < 1.0):
            raise ValueError(
                "classifier_dropout must be in [0, 1) or None, got " f"{self.classifier_dropout}"
            )
