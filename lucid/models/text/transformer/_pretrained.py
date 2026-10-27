"""Registry factories for the original Vaswani Transformer."""

from lucid.models._registry import register_model
from lucid.models.text.transformer._config import TransformerConfig
from lucid.models.text.transformer._model import (
    TransformerForSeq2SeqLM,
    TransformerForSequenceClassification,
    TransformerForTokenClassification,
    TransformerModel,
)
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained

# Vaswani et al. (2017) §6.2 — Table 3 "base" and "big" (the only two
# sizes the paper specifies).
_CFG_BASE = TransformerConfig()
_CFG_LARGE = TransformerConfig(
    hidden_size=1024,
    num_attention_heads=16,
    intermediate_size=4096,
    hidden_dropout=0.3,
    attention_dropout=0.3,
)


# ── Backbones ─────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="transformer",
    model_type="transformer",
    model_class=TransformerModel,
    default_config=_CFG_BASE,
    params=63084544,
)
def transformer_base(pretrained: bool = False, **overrides: object) -> TransformerModel:
    r"""Construct a Vaswani-style Transformer "base" encoder-decoder.

    Canonical seq2seq architecture from Vaswani, Shazeer, Parmar, Uszkoreit,
    Jones, Gomez, Kaiser, and Polosukhin, 2017, Table 3 ("base" row):
    :math:`N=6` encoder + :math:`N=6` decoder layers, :math:`d_{\text{model}}
    = 512`, :math:`h = 8` heads, :math:`d_{\text{ff}} = 2048`, dropout 0.1.
    Roughly 65M parameters — the model that achieved SOTA on WMT 2014
    En-De / En-Fr at a fraction of the training cost of prior recurrent
    seq2seq systems.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``transformer_base``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        Optional :class:`TransformerConfig` field overrides (e.g.
        ``vocab_size=...``, ``decoder_vocab_size=...``,
        ``share_embeddings=True``) forwarded into the underlying config.

    Returns
    -------
    TransformerModel
        Encoder-decoder trunk configured with the "base" size and any
        overrides.

    Notes
    -----
    Reference: Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser,
    and Polosukhin, *"Attention Is All You Need"*, NeurIPS, 2017
    (arXiv:1706.03762), Table 3.

    Multi-head attention:

    .. math::

        \mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(
            \frac{Q K^{\top}}{\sqrt{d_k}}
        \right) V.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.transformer import transformer_base
    >>> model = transformer_base().eval()
    >>> src = lucid.tensor([[1, 234, 567, 2]])
    >>> tgt = lucid.tensor([[1, 100, 200]])
    >>> out = model(src, decoder_input_ids=tgt)
    >>> out.last_hidden_state.shape   # (B=1, T_tgt=3, d_model=512)
    (1, 3, 512)
    """
    if pretrained:
        reject_unavailable_pretrained("transformer_base")
    return TransformerModel(apply_overrides(_CFG_BASE, overrides))


@register_model(
    task="base",
    family="transformer",
    model_type="transformer",
    model_class=TransformerModel,
    default_config=_CFG_LARGE,
    params=214249472,
)
def transformer_large(pretrained: bool = False, **overrides: object) -> TransformerModel:
    r"""Construct a Vaswani-style Transformer "big" encoder-decoder.

    Larger of the two original variants from Vaswani et al., 2017, Table 3
    ("big" row): :math:`N=6` encoder + :math:`N=6` decoder layers,
    :math:`d_{\text{model}} = 1024`, :math:`h = 16` heads,
    :math:`d_{\text{ff}} = 4096`, dropout 0.3.  Roughly 213M parameters —
    pushed BLEU on WMT 2014 En-De to 28.4 / En-Fr to 41.8 at release time.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``transformer_large``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        Optional :class:`TransformerConfig` field overrides forwarded into
        the underlying config.

    Returns
    -------
    TransformerModel
        Encoder-decoder trunk configured with the "big" size and any
        overrides.

    Notes
    -----
    Reference: Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser,
    and Polosukhin, *"Attention Is All You Need"*, NeurIPS, 2017
    (arXiv:1706.03762), Table 3.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.transformer import transformer_large
    >>> model = transformer_large().eval()
    >>> src = lucid.tensor([[1, 234, 567, 2]])
    >>> tgt = lucid.tensor([[1, 100, 200]])
    >>> out = model(src, decoder_input_ids=tgt)
    >>> out.last_hidden_state.shape   # (B=1, T_tgt=3, d_model=1024)
    (1, 3, 1024)
    """
    if pretrained:
        reject_unavailable_pretrained("transformer_large")
    return TransformerModel(apply_overrides(_CFG_LARGE, overrides))


# ── Seq2SeqLM heads ───────────────────────────────────────────────────────────


@register_model(
    task="language-modeling",
    family="transformer",
    model_type="transformer",
    model_class=TransformerForSeq2SeqLM,
    default_config=_CFG_BASE,
    params=63084544,
)
def transformer_base_seq2seq(
    pretrained: bool = False, **overrides: object
) -> TransformerForSeq2SeqLM:
    r"""Construct a Transformer "base" with a tied seq2seq-LM head.

    Same trunk as :func:`transformer_base` (N=6, d_model=512, h=8),
    augmented with a linear LM head whose weight is tied to the target-side
    embedding when ``config.tie_word_embeddings`` is ``True``.  Provides a
    ``.generate(input_ids)`` method for greedy seq2seq decoding (translation
    / summarisation inference).

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``transformer_base_seq2seq``, so ``True``
        raises ``NotImplementedError`` rather than return randomly initialised
        weights.
    **overrides : object
        Optional :class:`TransformerConfig` field overrides forwarded into
        the underlying config.

    Returns
    -------
    TransformerForSeq2SeqLM
        Encoder-decoder trunk wrapped with the tied LM head.

    Notes
    -----
    Reference: Vaswani et al., *"Attention Is All You Need"*, NeurIPS, 2017.

    Training loss when ``labels`` is supplied is the standard seq2seq
    cross-entropy:

    .. math::

        \mathcal{L}_{\mathrm{S2S}}
            = -\frac{1}{BT} \sum_{b, t}
              \log p_\theta(y_{b, t} \mid x_b, y_{b, < t}),

    with positions labelled ``-100`` excluded.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.transformer import transformer_base_seq2seq
    >>> model = transformer_base_seq2seq().eval()
    >>> src = lucid.tensor([[1, 234, 567, 2]])
    >>> tgt = lucid.tensor([[1, 100, 200]])
    >>> out = model(src, decoder_input_ids=tgt)
    >>> out.logits.shape   # (B=1, T_tgt=3, vocab=37000)
    (1, 3, 37000)
    """
    if pretrained:
        reject_unavailable_pretrained("transformer_base_seq2seq")
    return TransformerForSeq2SeqLM(apply_overrides(_CFG_BASE, overrides))


@register_model(
    task="language-modeling",
    family="transformer",
    model_type="transformer",
    model_class=TransformerForSeq2SeqLM,
    default_config=_CFG_LARGE,
    params=214249472,
)
def transformer_large_seq2seq(
    pretrained: bool = False, **overrides: object
) -> TransformerForSeq2SeqLM:
    r"""Construct a Transformer "big" with a tied seq2seq-LM head.

    Same trunk as :func:`transformer_large` (N=6, d_model=1024, h=16),
    augmented with a linear LM head whose weight is tied to the target-side
    embedding when ``config.tie_word_embeddings`` is ``True``.  Use for
    high-accuracy translation / summarisation fine-tunes when compute
    permits.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``transformer_large_seq2seq``, so ``True``
        raises ``NotImplementedError`` rather than return randomly initialised
        weights.
    **overrides : object
        Optional :class:`TransformerConfig` field overrides forwarded into
        the underlying config.

    Returns
    -------
    TransformerForSeq2SeqLM
        Encoder-decoder trunk wrapped with the tied LM head.

    Notes
    -----
    Reference: Vaswani et al., *"Attention Is All You Need"*, NeurIPS, 2017.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.transformer import transformer_large_seq2seq
    >>> model = transformer_large_seq2seq().eval()
    >>> src = lucid.tensor([[1, 234, 567, 2]])
    >>> tgt = lucid.tensor([[1, 100, 200]])
    >>> out = model(src, decoder_input_ids=tgt)
    >>> out.logits.shape   # (B=1, T_tgt=3, vocab=37000)
    (1, 3, 37000)
    """
    if pretrained:
        reject_unavailable_pretrained("transformer_large_seq2seq")
    return TransformerForSeq2SeqLM(apply_overrides(_CFG_LARGE, overrides))


# ── Encoder-only downstream heads ─────────────────────────────────────────────


@register_model(
    task="sequence-classification",
    family="transformer",
    model_type="transformer",
    model_class=TransformerForSequenceClassification,
    default_config=_CFG_BASE,
    params=37860354,
)
def transformer_base_cls(
    pretrained: bool = False, **overrides: object
) -> TransformerForSequenceClassification:
    r"""Construct a Transformer "base" with a sequence-classification head.

    Same trunk as :func:`transformer_base` (N=6, d_model=512, h=8); only the
    encoder half is used in ``forward``.  The first source-side token's
    encoder hidden state is dropout-regularised and projected through a
    linear of output width ``config.num_labels``.  Pattern mirrors
    :class:`BERTForSequenceClassification` and is appropriate for
    GLUE-style fine-tunes on the encoder-only half of the Vaswani trunk.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``transformer_base_cls``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        Optional :class:`TransformerConfig` field overrides forwarded into
        the underlying config.  Pass ``num_labels=N`` to set the number of
        classes (default 2).

    Returns
    -------
    TransformerForSequenceClassification
        Encoder-only Transformer trunk wrapped with a pooled classifier
        head.

    Notes
    -----
    Reference: Vaswani et al., *"Attention Is All You Need"*, NeurIPS, 2017.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.transformer import transformer_base_cls
    >>> model = transformer_base_cls(num_labels=3).eval()
    >>> input_ids = lucid.tensor([[1, 234, 567, 2]])
    >>> out = model(input_ids)
    >>> out.logits.shape   # (1, num_labels=3)
    (1, 3)
    """
    if pretrained:
        reject_unavailable_pretrained("transformer_base_cls")
    return TransformerForSequenceClassification(apply_overrides(_CFG_BASE, overrides))


@register_model(
    task="token-classification",
    family="transformer",
    model_type="transformer",
    model_class=TransformerForTokenClassification,
    default_config=_CFG_BASE,
    params=37860354,
)
def transformer_base_token_cls(
    pretrained: bool = False, **overrides: object
) -> TransformerForTokenClassification:
    r"""Construct a Transformer "base" with a per-token classification head.

    Same trunk as :func:`transformer_base` (N=6, d_model=512, h=8); only the
    encoder half is used in ``forward``.  Every source-side token's hidden
    state is dropout-regularised and projected through a linear of output
    width ``config.num_labels``, with the standard masked cross-entropy
    inherited from :class:`MaskedLMMixin`.  Use for sequence-labelling
    tasks (NER, POS) when an encoder-only Vaswani trunk is preferred to
    BERT.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``transformer_base_token_cls``, so ``True``
        raises ``NotImplementedError`` rather than return randomly initialised
        weights.
    **overrides : object
        Optional :class:`TransformerConfig` field overrides forwarded into
        the underlying config.  Pass ``num_labels=N`` to set the tag set
        size.

    Returns
    -------
    TransformerForTokenClassification
        Encoder-only Transformer trunk wrapped with a per-token classifier
        head.

    Notes
    -----
    Reference: Vaswani et al., *"Attention Is All You Need"*, NeurIPS, 2017.

    Loss (when ``labels`` is provided) is the masked cross-entropy over
    positions whose label is not ``-100``:

    .. math::

        \mathcal{L} = -\frac{1}{|V|} \sum_{(b, t) \in V}
            \log p_\theta(y_{b, t} \mid x_b).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.transformer import transformer_base_token_cls
    >>> model = transformer_base_token_cls(num_labels=9).eval()
    >>> input_ids = lucid.tensor([[1, 234, 567, 2]])
    >>> out = model(input_ids)
    >>> out.logits.shape   # (1, T=4, num_labels=9)
    (1, 4, 9)
    """
    if pretrained:
        reject_unavailable_pretrained("transformer_base_token_cls")
    return TransformerForTokenClassification(apply_overrides(_CFG_BASE, overrides))
