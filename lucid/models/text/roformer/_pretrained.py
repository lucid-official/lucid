"""Registry factories for RoFormer (Su et al., 2021).

The paper specifies a single architecture (``L=12, H=768, A=12`` — same
shape as BERT-base, with absolute PE replaced by RoPE).  We therefore
expose just ``roformer`` (plus task heads).  Test code that needs a
small variant should override config fields via
``create_model("roformer", hidden_size=..., num_hidden_layers=..., ...)``.
"""

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models.text.roformer._config import RoFormerConfig
from lucid.models.text.roformer._model import (
    RoFormerForMaskedLM,
    RoFormerForMultipleChoice,
    RoFormerForQuestionAnswering,
    RoFormerForSequenceClassification,
    RoFormerForTokenClassification,
    RoFormerModel,
)
from lucid.models.text.roformer._weights import RoFormerMLMWeights, RoFormerWeights
from lucid.models._utils._common import apply_overrides

_CFG_BASE = RoFormerConfig()  # paper default — the only published size


# ── Backbone ──────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="roformer",
    model_type="roformer",
    model_class=RoFormerModel,
    default_config=_CFG_BASE,
)
def roformer(
    pretrained: bool | str = False, *, weights: RoFormerWeights | None = None, **overrides: object
) -> RoFormerModel:
    r"""Construct a RoFormer encoder trunk.

    Canonical RoFormer architecture from Su, Lu, Pan, Murtadha, Wen, and
    Liu, 2024: :math:`L=12` transformer layers, :math:`H=768` hidden,
    :math:`A=12` attention heads, intermediate width 3072 — same shape as
    BERT-base, but with absolute learned position embeddings replaced by
    rotary position embedding (RoPE) applied to :math:`Q` and :math:`K`
    inside every self-attention layer.  Max sequence length is extended
    from 512 to 1536 to exercise RoPE's length-extrapolation property.

    Parameters
    ----------
    pretrained : bool, default=False
        If ``True``, resolves and loads the ``DEFAULT`` pretrained
        checkpoint; a tag string selects a specific one.  (This was
        documented as a no-op while the body already downloaded and
        loaded weights.)
    weights : RoFormerWeights, optional, keyword-only
        Explicit weights enum member; takes precedence over ``pretrained``.
    **overrides : object
        Optional :class:`RoFormerConfig` field overrides (e.g.
        ``vocab_size=...``, ``rotary_base=...``,
        ``max_position_embeddings=...``) forwarded into the underlying
        config.

    Returns
    -------
    RoFormerModel
        Encoder trunk configured with the paper defaults plus any
        overrides.

    Notes
    -----
    Reference: Su, Lu, Pan, Murtadha, Wen, and Liu, *"RoFormer: Enhanced
    Transformer with Rotary Position Embedding"*, Neurocomputing,
    vol. 568, 2024, article 127063 (arXiv:2104.09864).

    Rotary rotation matrix per :math:`(\cos\theta_i, \sin\theta_i)` pair:

    .. math::

        R_\theta = \begin{pmatrix}
            \cos\theta & -\sin\theta \\
            \sin\theta & \cos\theta
        \end{pmatrix}.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.roformer import roformer
    >>> model = roformer().eval()
    >>> input_ids = lucid.tensor([[101, 7592, 2088, 102]])
    >>> out = model(input_ids)
    >>> out.last_hidden_state.shape, out.pooler_output.shape
    ((1, 4, 768), (1, 768))
    """
    entry = weights_mod.resolve_weights(RoFormerWeights, pretrained, weights)
    model = RoFormerModel(apply_overrides(_CFG_BASE, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="roformer")
    return model


# ── Task heads ────────────────────────────────────────────────────────────────


@register_model(
    task="language-modeling",
    family="roformer",
    model_type="roformer",
    model_class=RoFormerForMaskedLM,
    default_config=_CFG_BASE,
)
def roformer_mlm(
    pretrained: bool | str = False,
    *,
    weights: RoFormerMLMWeights | None = None,
    **overrides: object,
) -> RoFormerForMaskedLM:
    r"""Construct a RoFormer model with a tied masked-LM head.

    Same trunk as :func:`roformer` (L=12, H=768, A=12), augmented with the
    BERT-style two-layer projection mapping each hidden state to vocabulary
    logits via a decoder whose weight matrix is tied to the input
    ``word_embeddings`` table when ``config.tie_word_embeddings`` is True.
    Use for MLM pre-training when RoPE relative-position semantics are
    preferred to absolute position embeddings.

    Parameters
    ----------
    pretrained : bool, default=False
        If ``True``, resolves and loads the ``DEFAULT`` pretrained
        checkpoint; a tag string selects a specific one.  (This was
        documented as a no-op while the body already downloaded and
        loaded weights.)
    weights : RoFormerMLMWeights, optional, keyword-only
        Explicit weights enum member; takes precedence over ``pretrained``.
    **overrides : object
        Optional :class:`RoFormerConfig` field overrides forwarded into the
        underlying config.

    Returns
    -------
    RoFormerForMaskedLM
        RoFormer trunk wrapped with the tied MLM head.

    Notes
    -----
    Reference: Su et al., *"RoFormer: Enhanced Transformer with Rotary
    Position Embedding"*, Neurocomputing, vol. 568, 2024 (arXiv:2104.09864).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.roformer import roformer_mlm
    >>> model = roformer_mlm().eval()
    >>> input_ids = lucid.tensor([[101, 7592, 103, 102]])   # [CLS] hello [MASK] [SEP]
    >>> out = model(input_ids)
    >>> out.logits.shape   # (1, 4, vocab=50000)
    (1, 4, 50000)
    """
    entry = weights_mod.resolve_weights(RoFormerMLMWeights, pretrained, weights)
    model = RoFormerForMaskedLM(apply_overrides(_CFG_BASE, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="roformer_mlm")
    return model


@register_model(
    task="sequence-classification",
    family="roformer",
    model_type="roformer",
    model_class=RoFormerForSequenceClassification,
    default_config=_CFG_BASE,
)
def roformer_cls(
    pretrained: bool | str = False, *, weights: RoFormerWeights | None = None, **overrides: object
) -> RoFormerForSequenceClassification:
    r"""Construct a RoFormer model with a sequence-classification head.

    Same trunk as :func:`roformer` (L=12, H=768, A=12), augmented with a
    dropout-regularised linear classifier on the pooled first-token
    embedding.  Use for GLUE-style fine-tunes when the RoPE relative
    position bias is preferred to absolute position embeddings.

    Parameters
    ----------
    pretrained : bool or str, default=False
        Encoder-weight selector.  ``False`` → fully random init; ``True``
        → loads the pretrained :func:`roformer` encoder
        (:attr:`RoFormerWeights.DEFAULT`) into the ``.roformer`` trunk; a
        tag string selects a specific encoder checkpoint.  **The classifier
        head is always randomly initialised** (fine-tuning starting point —
        no fine-tuned head ships).
    weights : RoFormerWeights, optional, keyword-only
        Explicit encoder-weights enum member; takes precedence over
        ``pretrained``.
    **overrides : object
        Optional :class:`RoFormerConfig` field overrides forwarded into the
        underlying config.  Pass ``num_labels=N`` to set the number of
        classes (default 2).  Overrides that change the encoder shape are
        incompatible with loading pretrained encoder weights.

    Returns
    -------
    RoFormerForSequenceClassification
        RoFormer trunk wrapped with the pooled classifier head (encoder
        pretrained when requested; head random).

    Notes
    -----
    Reference: Su et al., *"RoFormer: Enhanced Transformer with Rotary
    Position Embedding"*, Neurocomputing, vol. 568, 2024 (arXiv:2104.09864).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.roformer import roformer_cls
    >>> model = roformer_cls(num_labels=3).eval()
    >>> input_ids = lucid.tensor([[101, 7592, 102]])
    >>> out = model(input_ids)
    >>> out.logits.shape   # (1, num_labels=3)
    (1, 3)
    """
    entry = weights_mod.resolve_weights(RoFormerWeights, pretrained, weights)
    model = RoFormerForSequenceClassification(apply_overrides(_CFG_BASE, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model.roformer, entry, name="roformer")
    return model


@register_model(
    task="token-classification",
    family="roformer",
    model_type="roformer",
    model_class=RoFormerForTokenClassification,
    default_config=_CFG_BASE,
)
def roformer_token_cls(
    pretrained: bool | str = False, *, weights: RoFormerWeights | None = None, **overrides: object
) -> RoFormerForTokenClassification:
    r"""Construct a RoFormer model with a per-token classification head.

    Same trunk as :func:`roformer` (L=12, H=768, A=12), augmented with a
    dropout-regularised per-position linear classifier of output width
    ``config.num_labels``.  Use for sequence-labelling tasks (NER, POS,
    chunking) when a RoPE-trained encoder is preferred.

    Parameters
    ----------
    pretrained : bool or str, default=False
        Encoder-weight selector.  ``False`` → fully random init; ``True``
        → loads the pretrained :func:`roformer` encoder
        (:attr:`RoFormerWeights.DEFAULT`) into the ``.roformer`` trunk; a
        tag string selects a specific encoder checkpoint.  **The per-token
        classifier head is always randomly initialised** (fine-tuning
        starting point — no NER/POS-fine-tuned head ships).
    weights : RoFormerWeights, optional, keyword-only
        Explicit encoder-weights enum member; takes precedence over
        ``pretrained``.
    **overrides : object
        Optional :class:`RoFormerConfig` field overrides forwarded into the
        underlying config.  Pass ``num_labels=N`` to set the tag set size.
        Overrides that change the encoder shape are incompatible with
        loading pretrained encoder weights.

    Returns
    -------
    RoFormerForTokenClassification
        RoFormer trunk wrapped with the per-token classifier head (encoder
        pretrained when requested; head random).

    Notes
    -----
    Reference: Su et al., *"RoFormer: Enhanced Transformer with Rotary
    Position Embedding"*, Neurocomputing, vol. 568, 2024 (arXiv:2104.09864).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.roformer import roformer_token_cls
    >>> model = roformer_token_cls(num_labels=9).eval()
    >>> input_ids = lucid.tensor([[101, 2198, 7592, 102]])
    >>> out = model(input_ids)
    >>> out.logits.shape   # (1, T=4, num_labels=9)
    (1, 4, 9)
    """
    entry = weights_mod.resolve_weights(RoFormerWeights, pretrained, weights)
    model = RoFormerForTokenClassification(apply_overrides(_CFG_BASE, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model.roformer, entry, name="roformer")
    return model


@register_model(
    task="sequence-classification",
    family="roformer",
    model_type="roformer",
    model_class=RoFormerForMultipleChoice,
    default_config=_CFG_BASE,
)
def roformer_multiple_choice(
    pretrained: bool | str = False, *, weights: RoFormerWeights | None = None, **overrides: object
) -> RoFormerForMultipleChoice:
    r"""Construct a RoFormer model with a multiple-choice head.

    Scores each candidate independently and softmaxes across them, so the
    input is ``(N, C, L)`` — batch, choices, tokens — rather than the
    ``(N, L)`` every other head here takes.  The choices are flattened
    into the batch, run through the shared trunk, and reshaped back to
    ``(N, C)`` logits.

    Parameters
    ----------
    pretrained : bool or str, default=False
        Encoder-weight selector.  ``False`` → fully random init; ``True``
        → loads the :func:`roformer` encoder
        (:attr:`RoFormerWeights.DEFAULT`) into the ``.roformer`` trunk.
        **The scoring head is always randomly initialised.**
    weights : RoFormerWeights, optional, keyword-only
        Explicit encoder-weights enum member; takes precedence over
        ``pretrained``.
    **overrides : object
        Optional :class:`RoFormerConfig` field overrides.  ``num_labels``
        does not apply — the head is a single scalar per choice, and the
        choice count comes from the input's second axis.

    Returns
    -------
    RoFormerForMultipleChoice
        RoFormer trunk wrapped with the per-choice scorer.

    Notes
    -----
    Reference: Su et al., *"RoFormer: Enhanced Transformer with Rotary
    Position Embedding"*, Neurocomputing, vol. 568, 2024 (arXiv:2104.09864).

    The head reads the **last** token's hidden state, not a pooled ``[CLS]``
    — see :class:`RoFormerForMultipleChoice` for why.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.roformer import roformer_multiple_choice
    >>> model = roformer_multiple_choice(vocab_size=99, hidden_size=32,
    ...                                  num_hidden_layers=2,
    ...                                  num_attention_heads=2,
    ...                                  intermediate_size=64).eval()
    >>> ids = lucid.tensor([[[1, 2, 3, 4], [1, 2, 3, 5]]])   # (N=1, C=2, L=4)
    >>> out = model(ids)
    >>> out.logits.shape   # (N=1, C=2)
    (1, 2)
    """
    entry = weights_mod.resolve_weights(RoFormerWeights, pretrained, weights)
    model = RoFormerForMultipleChoice(apply_overrides(_CFG_BASE, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model.roformer, entry, name="roformer")
    return model


@register_model(
    task="sequence-classification",
    family="roformer",
    model_type="roformer",
    model_class=RoFormerForQuestionAnswering,
    default_config=_CFG_BASE,
)
def roformer_qa(
    pretrained: bool | str = False, *, weights: RoFormerWeights | None = None, **overrides: object
) -> RoFormerForQuestionAnswering:
    r"""Construct a RoFormer model with an extractive span head.

    A two-way linear over each position, scoring it as the start and the
    end of the answer span.

    Parameters
    ----------
    pretrained : bool or str, default=False
        Encoder-weight selector.  ``False`` → fully random init; ``True``
        → loads the :func:`roformer` encoder
        (:attr:`RoFormerWeights.DEFAULT`) into the ``.roformer`` trunk.
        **The span head is always randomly initialised** — no
        SQuAD-fine-tuned RoFormer checkpoint ships here.
    weights : RoFormerWeights, optional, keyword-only
        Explicit encoder-weights enum member; takes precedence over
        ``pretrained``.
    **overrides : object
        Optional :class:`RoFormerConfig` field overrides.

    Returns
    -------
    RoFormerForQuestionAnswering
        RoFormer trunk wrapped with the span head.

    Notes
    -----
    Reference: Su et al., *"RoFormer: Enhanced Transformer with Rotary
    Position Embedding"*, Neurocomputing, vol. 568, 2024 (arXiv:2104.09864).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.roformer import roformer_qa
    >>> model = roformer_qa(vocab_size=99, hidden_size=32,
    ...                     num_hidden_layers=2, num_attention_heads=2,
    ...                     intermediate_size=64).eval()
    >>> out = model(lucid.tensor([[1, 2, 3, 4]]))
    >>> out.start_logits.shape, out.end_logits.shape
    ((1, 4), (1, 4))
    """
    entry = weights_mod.resolve_weights(RoFormerWeights, pretrained, weights)
    model = RoFormerForQuestionAnswering(apply_overrides(_CFG_BASE, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model.roformer, entry, name="roformer")
    return model
