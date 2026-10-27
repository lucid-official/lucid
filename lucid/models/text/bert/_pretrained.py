"""Registry factories for BERT variants.

Sizes follow Devlin et al. (base / large) and Turc et al. 2019
("Well-Read Students Learn Better") for the four pre-distilled sizes
(tiny / mini / small / medium).  All six base encoders and the two
masked-LM heads ship Wikipedia + BookCorpus pretrained weights through
:mod:`lucid.weights` (per-factory ``*Weights`` enums + ``weights=`` /
``pretrained=``, mirroring the ResNet wiring).

The downstream-task heads follow a "best available canonical weights" rule:

* **QA** (``bert_base_qa`` / ``bert_large_qa``) and **NER**
  (``bert_base_token_cls``) ship *full fine-tuned* checkpoints on canonical
  benchmarks — SQuAD v1.1 (``csarron`` base / official Google large-wwm) and
  CoNLL-2003 (``dslim`` cased) — so ``pretrained=True`` is inference-ready.
* **Sequence classification** (``*_cls``) has no clean canonical+permissive
  GLUE checkpoint, so ``pretrained=`` loads the pretrained **encoder** into
  the ``.bert`` submodule and leaves the classifier random (the standard
  fine-tuning start, mirroring ``AutoModelForX.from_pretrained(<encoder>)``).
"""

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models.text.bert._config import BERTConfig
from lucid.models.text.bert._model import (
    BERTForCausalLM,
    BERTForMaskedLM,
    BERTForNextSentencePrediction,
    BERTForPreTraining,
    BERTForQuestionAnswering,
    BERTForSequenceClassification,
    BERTForTokenClassification,
    BERTModel,
)
from lucid.models.text.bert._weights import (
    BERTBaseMLMWeights,
    BERTBaseNERWeights,
    BERTBaseQAWeights,
    BERTBaseWeights,
    BERTLargeMLMWeights,
    BERTLargeQAWeights,
    BERTLargeWeights,
    BERTMediumWeights,
    BERTMiniWeights,
    BERTSmallWeights,
    BERTTinyWeights,
)
from lucid.models._utils._common import apply_overrides

# Devlin et al. 2018 + Turc et al. 2019 size table.
_CFG_TINY = BERTConfig(
    hidden_size=128, num_hidden_layers=2, num_attention_heads=2, intermediate_size=512
)
_CFG_MINI = BERTConfig(
    hidden_size=256, num_hidden_layers=4, num_attention_heads=4, intermediate_size=1024
)
_CFG_SMALL = BERTConfig(
    hidden_size=512, num_hidden_layers=4, num_attention_heads=8, intermediate_size=2048
)
_CFG_MEDIUM = BERTConfig(
    hidden_size=512, num_hidden_layers=8, num_attention_heads=8, intermediate_size=2048
)
_CFG_BASE = BERTConfig(
    hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072
)
_CFG_LARGE = BERTConfig(
    hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, intermediate_size=4096
)

# dslim/bert-base-NER is a *cased* BERT-Base (vocab 28 996) with a 9-way BIO
# tag head — the only field deltas from _CFG_BASE are vocab_size + num_labels.
_CFG_NER_CONLL = BERTConfig(
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    vocab_size=28_996,
    num_labels=9,
)


# ── Backbones ─────────────────────────────────────────────────────────────────


@register_model(
    task="base", family="bert", model_type="bert", model_class=BERTModel, default_config=_CFG_TINY
)
def bert_tiny(
    pretrained: bool | str = False, *, weights: BERTTinyWeights | None = None, **overrides: object
) -> BERTModel:
    r"""Construct a BERT-Tiny encoder.

    Smallest of the four distillation-targeted variants from Turc, Chang,
    Lee, and Toutanova, *"Well-Read Students Learn Better"*, 2019
    (arXiv:1908.08962): :math:`L=2` transformer layers, :math:`H=128` hidden,
    :math:`A=2` heads, intermediate width 512.  Roughly 4M parameters — use
    for CPU/edge inference, latency-bound demos, or unit tests of the BERT
    pipeline.

    Parameters
    ----------
    pretrained : bool or str, default=False
        ``True`` loads the ``DEFAULT`` checkpoint of :class:`BERTTinyWeights`; a
        string names a specific tag.  ``False`` returns a randomly initialised
        model.
    weights : BERTTinyWeights, optional, keyword-only
        Explicit weights enum member; takes precedence over ``pretrained``.
    **overrides : object
        Optional :class:`BERTConfig` field overrides (e.g. ``vocab_size=...``,
        ``num_labels=...``).  Forwarded into a fresh ``BERTConfig`` whose
        defaults match the Tiny size table above.

    Returns
    -------
    BERTModel
        Encoder trunk configured with the BERT-Tiny size and any overrides.

    Notes
    -----
    Reference: Turc, Chang, Lee, and Toutanova, *"Well-Read Students Learn
    Better: On the Importance of Pre-training Compact Models"*, 2019
    (arXiv:1908.08962).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.bert import bert_tiny
    >>> model = bert_tiny().eval()
    >>> out = model(lucid.tensor([[101, 7592, 102]]))
    >>> out.last_hidden_state.shape   # (1, 3, 128)
    (1, 3, 128)
    """
    entry = weights_mod.resolve_weights(BERTTinyWeights, pretrained, weights)
    model = BERTModel(apply_overrides(_CFG_TINY, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="bert_tiny")
    return model


@register_model(
    task="base", family="bert", model_type="bert", model_class=BERTModel, default_config=_CFG_MINI
)
def bert_mini(
    pretrained: bool | str = False, *, weights: BERTMiniWeights | None = None, **overrides: object
) -> BERTModel:
    r"""Construct a BERT-Mini encoder.

    Second-smallest of the Turc et al., 2019 distillation sizes:
    :math:`L=4` layers, :math:`H=256` hidden, :math:`A=4` heads, intermediate
    width 1024.  Roughly 11M parameters.

    Parameters
    ----------
    pretrained : bool or str, default=False
        ``True`` loads the ``DEFAULT`` checkpoint of :class:`BERTMiniWeights`; a
        string names a specific tag.  ``False`` returns a randomly initialised
        model.
    weights : BERTMiniWeights, optional, keyword-only
        Explicit weights enum member; takes precedence over ``pretrained``.
    **overrides : object
        Optional :class:`BERTConfig` field overrides forwarded into the
        underlying config.

    Returns
    -------
    BERTModel
        Encoder trunk configured with the BERT-Mini size and any overrides.

    Notes
    -----
    Reference: Turc, Chang, Lee, and Toutanova, *"Well-Read Students Learn
    Better"*, 2019 (arXiv:1908.08962).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.bert import bert_mini
    >>> model = bert_mini().eval()
    >>> out = model(lucid.tensor([[101, 7592, 102]]))
    >>> out.last_hidden_state.shape   # (1, 3, 256)
    (1, 3, 256)
    """
    entry = weights_mod.resolve_weights(BERTMiniWeights, pretrained, weights)
    model = BERTModel(apply_overrides(_CFG_MINI, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="bert_mini")
    return model


@register_model(
    task="base", family="bert", model_type="bert", model_class=BERTModel, default_config=_CFG_SMALL
)
def bert_small(
    pretrained: bool | str = False, *, weights: BERTSmallWeights | None = None, **overrides: object
) -> BERTModel:
    r"""Construct a BERT-Small encoder.

    Distillation-targeted variant from Turc et al., 2019: :math:`L=4` layers,
    :math:`H=512` hidden, :math:`A=8` heads, intermediate width 2048.
    Roughly 29M parameters — a useful mid-point between Mini and Base.

    Parameters
    ----------
    pretrained : bool or str, default=False
        ``True`` loads the ``DEFAULT`` checkpoint of :class:`BERTSmallWeights`; a
        string names a specific tag.  ``False`` returns a randomly initialised
        model.
    weights : BERTSmallWeights, optional, keyword-only
        Explicit weights enum member; takes precedence over ``pretrained``.
    **overrides : object
        Optional :class:`BERTConfig` field overrides forwarded into the
        underlying config.

    Returns
    -------
    BERTModel
        Encoder trunk configured with the BERT-Small size and any overrides.

    Notes
    -----
    Reference: Turc, Chang, Lee, and Toutanova, *"Well-Read Students Learn
    Better"*, 2019 (arXiv:1908.08962).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.bert import bert_small
    >>> model = bert_small().eval()
    >>> out = model(lucid.tensor([[101, 7592, 102]]))
    >>> out.last_hidden_state.shape   # (1, 3, 512)
    (1, 3, 512)
    """
    entry = weights_mod.resolve_weights(BERTSmallWeights, pretrained, weights)
    model = BERTModel(apply_overrides(_CFG_SMALL, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="bert_small")
    return model


@register_model(
    task="base", family="bert", model_type="bert", model_class=BERTModel, default_config=_CFG_MEDIUM
)
def bert_medium(
    pretrained: bool | str = False, *, weights: BERTMediumWeights | None = None, **overrides: object
) -> BERTModel:
    r"""Construct a BERT-Medium encoder.

    Largest of the Turc et al., 2019 distillation sizes: :math:`L=8` layers,
    :math:`H=512` hidden, :math:`A=8` heads, intermediate width 2048.
    Roughly 41M parameters — closest in depth to Base while keeping the
    Small width.

    Parameters
    ----------
    pretrained : bool or str, default=False
        ``True`` loads the ``DEFAULT`` checkpoint of :class:`BERTMediumWeights`; a
        string names a specific tag.  ``False`` returns a randomly initialised
        model.
    weights : BERTMediumWeights, optional, keyword-only
        Explicit weights enum member; takes precedence over ``pretrained``.
    **overrides : object
        Optional :class:`BERTConfig` field overrides forwarded into the
        underlying config.

    Returns
    -------
    BERTModel
        Encoder trunk configured with the BERT-Medium size and any overrides.

    Notes
    -----
    Reference: Turc, Chang, Lee, and Toutanova, *"Well-Read Students Learn
    Better"*, 2019 (arXiv:1908.08962).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.bert import bert_medium
    >>> model = bert_medium().eval()
    >>> out = model(lucid.tensor([[101, 7592, 102]]))
    >>> out.last_hidden_state.shape   # (1, 3, 512)
    (1, 3, 512)
    """
    entry = weights_mod.resolve_weights(BERTMediumWeights, pretrained, weights)
    model = BERTModel(apply_overrides(_CFG_MEDIUM, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="bert_medium")
    return model


@register_model(
    task="base", family="bert", model_type="bert", model_class=BERTModel, default_config=_CFG_BASE
)
def bert_base(
    pretrained: bool | str = False, *, weights: BERTBaseWeights | None = None, **overrides: object
) -> BERTModel:
    r"""Construct a BERT-Base encoder.

    Canonical mid-size variant from Devlin et al., 2018 Table 1:
    :math:`L=12` transformer layers, :math:`H=768` hidden, :math:`A=12`
    attention heads, intermediate width 3072 — roughly 110M parameters.
    Uses the 30,522-piece WordPiece vocabulary and a maximum sequence length
    of 512.  This is the default starting point for most BERT fine-tuning
    work.

    Parameters
    ----------
    pretrained : bool or str, default=False
        ``True`` loads the ``DEFAULT`` checkpoint of :class:`BERTBaseWeights`; a
        string names a specific tag.  ``False`` returns a randomly initialised
        model.
    weights : BERTBaseWeights, optional, keyword-only
        Explicit weights enum member; takes precedence over ``pretrained``.
    **overrides : object
        Optional :class:`BERTConfig` field overrides (e.g. ``vocab_size=...``,
        ``max_position_embeddings=...``) forwarded into the underlying
        config.

    Returns
    -------
    BERTModel
        Encoder trunk configured with the BERT-Base size and any overrides.

    Notes
    -----
    Reference: Devlin, Chang, Lee, and Toutanova, *"BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding"*, NAACL 2019
    (arXiv:1810.04805), Table 1 (BERT\ :sub:`BASE`).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.bert import bert_base
    >>> model = bert_base().eval()
    >>> input_ids = lucid.tensor([[101, 7592, 2088, 102]])   # [CLS] hello world [SEP]
    >>> out = model(input_ids)
    >>> out.last_hidden_state.shape, out.pooler_output.shape
    ((1, 4, 768), (1, 768))
    """
    entry = weights_mod.resolve_weights(BERTBaseWeights, pretrained, weights)
    model = BERTModel(apply_overrides(_CFG_BASE, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="bert_base")
    return model


@register_model(
    task="base", family="bert", model_type="bert", model_class=BERTModel, default_config=_CFG_LARGE
)
def bert_large(
    pretrained: bool | str = False, *, weights: BERTLargeWeights | None = None, **overrides: object
) -> BERTModel:
    r"""Construct a BERT-Large encoder.

    Larger of the two original variants from Devlin et al., 2018 Table 1:
    :math:`L=24` transformer layers, :math:`H=1024` hidden, :math:`A=16`
    attention heads, intermediate width 4096 — roughly 340M parameters.
    Uses the 30,522-piece WordPiece vocabulary and a maximum sequence length
    of 512.  Achieved SOTA on every GLUE task and SQuAD v1.1/v2.0 at release.

    Parameters
    ----------
    pretrained : bool or str, default=False
        ``True`` loads the ``DEFAULT`` checkpoint of :class:`BERTLargeWeights`; a
        string names a specific tag.  ``False`` returns a randomly initialised
        model.
    weights : BERTLargeWeights, optional, keyword-only
        Explicit weights enum member; takes precedence over ``pretrained``.
    **overrides : object
        Optional :class:`BERTConfig` field overrides forwarded into the
        underlying config.

    Returns
    -------
    BERTModel
        Encoder trunk configured with the BERT-Large size and any overrides.

    Notes
    -----
    Reference: Devlin, Chang, Lee, and Toutanova, *"BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding"*, NAACL 2019
    (arXiv:1810.04805), Table 1 (BERT\ :sub:`LARGE`).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.bert import bert_large
    >>> model = bert_large().eval()
    >>> input_ids = lucid.tensor([[101, 7592, 2088, 102]])   # [CLS] hello world [SEP]
    >>> out = model(input_ids)
    >>> out.last_hidden_state.shape, out.pooler_output.shape
    ((1, 4, 1024), (1, 1024))
    """
    entry = weights_mod.resolve_weights(BERTLargeWeights, pretrained, weights)
    model = BERTModel(apply_overrides(_CFG_LARGE, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="bert_large")
    return model


# ── Masked-LM heads ───────────────────────────────────────────────────────────


@register_model(
    task="language-modeling",
    family="bert",
    model_type="bert",
    model_class=BERTForMaskedLM,
    default_config=_CFG_BASE,
)
def bert_base_mlm(
    pretrained: bool | str = False,
    *,
    weights: BERTBaseMLMWeights | None = None,
    **overrides: object,
) -> BERTForMaskedLM:
    r"""Construct a BERT-Base model with a tied masked-LM head.

    Same trunk as :func:`bert_base` (L=12, H=768, A=12, ~110M parameters),
    augmented with the masked-language-modeling prediction head used at
    pre-training time.  Suitable for continued pre-training on domain
    corpora, mask-filling inference, and reproducing the MLM half of the
    Devlin et al. (2018) pre-training recipe.

    Parameters
    ----------
    pretrained : bool or str, default=False
        ``True`` loads the ``DEFAULT`` checkpoint of :class:`BERTBaseMLMWeights`; a
        string names a specific tag.  ``False`` returns a randomly initialised
        model.
    weights : BERTBaseMLMWeights, optional, keyword-only
        Explicit weights enum member; takes precedence over ``pretrained``.
    **overrides : object
        Optional :class:`BERTConfig` field overrides forwarded into the
        underlying config.

    Returns
    -------
    BERTForMaskedLM
        BERT-Base wrapped with the tied MLM head.

    Notes
    -----
    Reference: Devlin, Chang, Lee, and Toutanova, *"BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding"*, NAACL 2019
    (arXiv:1810.04805) §3.1 (task #1).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.bert import bert_base_mlm
    >>> model = bert_base_mlm().eval()
    >>> input_ids = lucid.tensor([[101, 7592, 103, 102]])   # [CLS] hello [MASK] [SEP]
    >>> out = model(input_ids)
    >>> out.logits.shape   # (1, 4, vocab_size=30522)
    (1, 4, 30522)
    """
    entry = weights_mod.resolve_weights(BERTBaseMLMWeights, pretrained, weights)
    model = BERTForMaskedLM(apply_overrides(_CFG_BASE, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="bert_base_mlm")
    return model


@register_model(
    task="language-modeling",
    family="bert",
    model_type="bert",
    model_class=BERTForMaskedLM,
    default_config=_CFG_LARGE,
)
def bert_large_mlm(
    pretrained: bool | str = False,
    *,
    weights: BERTLargeMLMWeights | None = None,
    **overrides: object,
) -> BERTForMaskedLM:
    r"""Construct a BERT-Large model with a tied masked-LM head.

    Same trunk as :func:`bert_large` (L=24, H=1024, A=16, ~340M parameters),
    augmented with the masked-language-modeling prediction head used at
    pre-training time.

    Parameters
    ----------
    pretrained : bool or str, default=False
        ``True`` loads the ``DEFAULT`` checkpoint of :class:`BERTLargeMLMWeights`; a
        string names a specific tag.  ``False`` returns a randomly initialised
        model.
    weights : BERTLargeMLMWeights, optional, keyword-only
        Explicit weights enum member; takes precedence over ``pretrained``.
    **overrides : object
        Optional :class:`BERTConfig` field overrides forwarded into the
        underlying config.

    Returns
    -------
    BERTForMaskedLM
        BERT-Large wrapped with the tied MLM head.

    Notes
    -----
    Reference: Devlin, Chang, Lee, and Toutanova, *"BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding"*, NAACL 2019
    (arXiv:1810.04805) §3.1 (task #1).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.bert import bert_large_mlm
    >>> model = bert_large_mlm().eval()
    >>> input_ids = lucid.tensor([[101, 7592, 103, 102]])
    >>> out = model(input_ids)
    >>> out.logits.shape   # (1, 4, 30522)
    (1, 4, 30522)
    """
    entry = weights_mod.resolve_weights(BERTLargeMLMWeights, pretrained, weights)
    model = BERTForMaskedLM(apply_overrides(_CFG_LARGE, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="bert_large_mlm")
    return model


# ── Sequence / token / QA classification heads ────────────────────────────────


@register_model(
    task="sequence-classification",
    family="bert",
    model_type="bert",
    model_class=BERTForSequenceClassification,
    default_config=_CFG_BASE,
)
def bert_base_cls(
    pretrained: bool | str = False, *, weights: BERTBaseWeights | None = None, **overrides: object
) -> BERTForSequenceClassification:
    r"""Construct a BERT-Base model with a sequence-classification head.

    Same trunk as :func:`bert_base` (L=12, H=768, A=12), augmented with a
    pooled-CLS linear classifier of output width ``config.num_labels``.  The
    canonical fine-tuning recipe for GLUE-style single-sentence and
    sentence-pair tasks (SST-2, MNLI, QQP, RTE, MRPC, STS-B).

    Parameters
    ----------
    pretrained : bool or str, default=False
        Encoder-weight selector.  ``False`` → fully random init; ``True``
        → loads the pretrained :func:`bert_base` encoder
        (:attr:`BERTBaseWeights.DEFAULT`) into the ``.bert`` trunk; a tag
        string selects a specific encoder checkpoint.  **The classification
        head is always randomly initialised** — this is the standard
        fine-tuning starting point (no GLUE-fine-tuned head ships), so
        train on a downstream task before inference.
    weights : BERTBaseWeights, optional, keyword-only
        Explicit encoder-weights enum member; takes precedence over
        ``pretrained``.
    **overrides : object
        Optional :class:`BERTConfig` field overrides forwarded into the
        underlying config.  Pass ``num_labels=N`` to set the number of
        classes (default 2).  Overrides that change the encoder shape are
        incompatible with loading pretrained encoder weights.

    Returns
    -------
    BERTForSequenceClassification
        BERT-Base wrapped with the pooled classifier head (encoder
        pretrained when requested; head random).

    Notes
    -----
    Reference: Devlin, Chang, Lee, and Toutanova, *"BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding"*, NAACL 2019
    (arXiv:1810.04805) §4.1.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.bert import bert_base_cls
    >>> model = bert_base_cls(num_labels=3).eval()
    >>> input_ids = lucid.tensor([[101, 7592, 102]])
    >>> out = model(input_ids)
    >>> out.logits.shape   # (1, num_labels=3)
    (1, 3)
    """
    entry = weights_mod.resolve_weights(BERTBaseWeights, pretrained, weights)
    model = BERTForSequenceClassification(apply_overrides(_CFG_BASE, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model.bert, entry, name="bert_base")
    return model


@register_model(
    task="sequence-classification",
    family="bert",
    model_type="bert",
    model_class=BERTForSequenceClassification,
    default_config=_CFG_LARGE,
)
def bert_large_cls(
    pretrained: bool | str = False, *, weights: BERTLargeWeights | None = None, **overrides: object
) -> BERTForSequenceClassification:
    r"""Construct a BERT-Large model with a sequence-classification head.

    Same trunk as :func:`bert_large` (L=24, H=1024, A=16, ~340M parameters),
    augmented with a pooled-CLS linear classifier of output width
    ``config.num_labels``.  Use for high-accuracy GLUE-style fine-tunes when
    compute permits.

    Parameters
    ----------
    pretrained : bool or str, default=False
        Encoder-weight selector.  ``False`` → fully random init; ``True``
        → loads the pretrained :func:`bert_large` encoder
        (:attr:`BERTLargeWeights.DEFAULT`) into the ``.bert`` trunk; a tag
        string selects a specific encoder checkpoint.  **The classification
        head is always randomly initialised** (fine-tuning starting point).
    weights : BERTLargeWeights, optional, keyword-only
        Explicit encoder-weights enum member; takes precedence over
        ``pretrained``.
    **overrides : object
        Optional :class:`BERTConfig` field overrides forwarded into the
        underlying config.  Pass ``num_labels=N`` to set the number of
        classes (default 2).  Overrides that change the encoder shape are
        incompatible with loading pretrained encoder weights.

    Returns
    -------
    BERTForSequenceClassification
        BERT-Large wrapped with the pooled classifier head (encoder
        pretrained when requested; head random).

    Notes
    -----
    Reference: Devlin, Chang, Lee, and Toutanova, *"BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding"*, NAACL 2019
    (arXiv:1810.04805) §4.1.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.bert import bert_large_cls
    >>> model = bert_large_cls(num_labels=2).eval()
    >>> input_ids = lucid.tensor([[101, 7592, 102]])
    >>> out = model(input_ids)
    >>> out.logits.shape   # (1, 2)
    (1, 2)
    """
    entry = weights_mod.resolve_weights(BERTLargeWeights, pretrained, weights)
    model = BERTForSequenceClassification(apply_overrides(_CFG_LARGE, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model.bert, entry, name="bert_large")
    return model


@register_model(
    task="token-classification",
    family="bert",
    model_type="bert",
    model_class=BERTForTokenClassification,
    default_config=_CFG_BASE,
)
def bert_base_token_cls(
    pretrained: bool | str = False,
    *,
    weights: BERTBaseNERWeights | None = None,
    **overrides: object,
) -> BERTForTokenClassification:
    r"""Construct a BERT-Base per-token classification model (CoNLL-2003 NER).

    A per-position linear classifier over the encoder sequence output, for
    sequence-labelling tasks: named-entity recognition, POS tagging, chunking.

    Parameters
    ----------
    pretrained : bool or str, default=False
        Weight selector.  ``False`` → random init (uncased ``_CFG_BASE``,
        ``num_labels=2``); ``True`` → the CoNLL-2003 NER checkpoint
        (:attr:`BERTBaseNERWeights.CONLL2003`, from ``dslim/bert-base-NER``)
        — an inference-ready 9-way BIO tagger (O, B/I-PER, B/I-ORG, B/I-LOC,
        B/I-MISC; test F1 ≈ 91.3).  **This is a cased BERT-Base** (vocab
        28 996); when ``pretrained`` is requested the model is built with the
        matching cased config, so tokenize with the cased
        :class:`BERTTokenizer` (``do_lower_case=False``).
    weights : BERTBaseNERWeights, optional, keyword-only
        Explicit weights enum member; takes precedence over ``pretrained``.
    **overrides : object
        Optional :class:`BERTConfig` field overrides.  When given, they
        override the (random-init) ``_CFG_BASE`` and disable pretrained
        loading's cased-config switch — pass your own ``vocab_size`` /
        ``num_labels`` for custom fine-tuning.

    Returns
    -------
    BERTForTokenClassification
        BERT-Base wrapped with the per-token classifier head (fully
        fine-tuned on CoNLL-2003 when ``pretrained`` is requested).

    Notes
    -----
    Reference: Devlin, Chang, Lee, and Toutanova, *"BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding"*, NAACL 2019
    (arXiv:1810.04805) §4.3.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.bert import bert_base_token_cls
    >>> model = bert_base_token_cls(pretrained=True).eval()   # CoNLL-2003 NER (9 tags)
    >>> input_ids = lucid.tensor([[101, 2198, 7592, 102]])
    >>> out = model(input_ids)
    >>> out.logits.shape   # (1, T=4, num_labels=9)
    (1, 4, 9)
    """
    entry = weights_mod.resolve_weights(BERTBaseNERWeights, pretrained, weights)
    # The canonical CoNLL checkpoint is cased (vocab 28 996, 9 BIO labels);
    # build the matching config when loading it (unless overrides intervene).
    if entry is not None and not overrides:
        cfg = _CFG_NER_CONLL
    else:
        cfg = apply_overrides(_CFG_BASE, overrides)
    model = BERTForTokenClassification(cfg)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="bert_base_token_cls")
    return model


@register_model(
    task="sequence-classification",
    family="bert",
    model_type="bert",
    model_class=BERTForQuestionAnswering,
    default_config=_CFG_BASE,
)
def bert_base_qa(
    pretrained: bool | str = False, *, weights: BERTBaseQAWeights | None = None, **overrides: object
) -> BERTForQuestionAnswering:
    r"""Construct a BERT-Base extractive-QA model (SQuAD v1.1).

    Same trunk as :func:`bert_base` (L=12, H=768, A=12), augmented with a
    2-way linear head producing start- and end-position logits over each
    input token.

    Parameters
    ----------
    pretrained : bool or str, default=False
        Weight selector.  ``False`` → random init; ``True`` → the SQuAD v1.1
        fine-tuned checkpoint (:attr:`BERTBaseQAWeights.SQUAD_V1`, from
        ``csarron/bert-base-uncased-squad-v1``) — an inference-ready
        extractive-QA model (EM ≈ 80.9 / F1 ≈ 88.1).  Tokenize with the
        uncased :class:`BERTTokenizer`.
    weights : BERTBaseQAWeights, optional, keyword-only
        Explicit weights enum member; takes precedence over ``pretrained``.
    **overrides : object
        Optional :class:`BERTConfig` field overrides forwarded into the
        underlying config.  ``num_labels`` is ignored by this head.
        Overrides that change the encoder shape are incompatible with the
        pretrained checkpoint.

    Returns
    -------
    BERTForQuestionAnswering
        BERT-Base wrapped with the span-prediction head (fully fine-tuned
        on SQuAD when ``pretrained`` is requested).

    Notes
    -----
    Reference: Devlin, Chang, Lee, and Toutanova, *"BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding"*, NAACL 2019
    (arXiv:1810.04805) §4.2.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.bert import bert_base_qa
    >>> model = bert_base_qa().eval()
    >>> # question + context (CLS q [SEP] c [SEP]) ids
    >>> input_ids = lucid.tensor([[101, 2040, 102, 1045, 2572, 102]])
    >>> out = model(input_ids)
    >>> out.start_logits.shape, out.end_logits.shape
    ((1, 6), (1, 6))
    """
    entry = weights_mod.resolve_weights(BERTBaseQAWeights, pretrained, weights)
    model = BERTForQuestionAnswering(apply_overrides(_CFG_BASE, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="bert_base_qa")
    return model


@register_model(
    task="sequence-classification",
    family="bert",
    model_type="bert",
    model_class=BERTForQuestionAnswering,
    default_config=_CFG_LARGE,
)
def bert_large_qa(
    pretrained: bool | str = False,
    *,
    weights: BERTLargeQAWeights | None = None,
    **overrides: object,
) -> BERTForQuestionAnswering:
    r"""Construct a BERT-Large extractive-QA model (whole-word-masking, SQuAD v1.1).

    Same trunk as :func:`bert_large` (L=24, H=1024, A=16, ~335M parameters),
    augmented with a 2-way span head.

    Parameters
    ----------
    pretrained : bool or str, default=False
        Weight selector.  ``False`` → random init; ``True`` → the official
        Google whole-word-masking SQuAD v1.1 checkpoint
        (:attr:`BERTLargeQAWeights.SQUAD_V1`, from
        ``google-bert/bert-large-uncased-whole-word-masking-finetuned-squad``)
        — an inference-ready extractive-QA model (EM ≈ 86.9 / F1 ≈ 93.2),
        the strongest BERT SQuAD result.  Tokenize with the uncased
        :class:`BERTTokenizer`.
    weights : BERTLargeQAWeights, optional, keyword-only
        Explicit weights enum member; takes precedence over ``pretrained``.
    **overrides : object
        Optional :class:`BERTConfig` field overrides forwarded into the
        underlying config.  ``num_labels`` is ignored by this head.
        Overrides that change the encoder shape are incompatible with the
        pretrained checkpoint.

    Returns
    -------
    BERTForQuestionAnswering
        BERT-Large wrapped with the span-prediction head (fully fine-tuned
        on SQuAD when ``pretrained`` is requested).

    Notes
    -----
    Reference: Devlin, Chang, Lee, and Toutanova, *"BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding"*, NAACL 2019
    (arXiv:1810.04805) §4.2.  Whole-word masking is the pre-training tweak
    from the BERT repository's later release.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.bert import bert_large_qa
    >>> model = bert_large_qa().eval()
    >>> input_ids = lucid.tensor([[101, 2040, 102, 1045, 2572, 102]])
    >>> out = model(input_ids)
    >>> out.start_logits.shape, out.end_logits.shape
    ((1, 6), (1, 6))
    """
    entry = weights_mod.resolve_weights(BERTLargeQAWeights, pretrained, weights)
    model = BERTForQuestionAnswering(apply_overrides(_CFG_LARGE, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="bert_large_qa")
    return model


@register_model(
    task="language-modeling",
    family="bert",
    model_type="bert",
    model_class=BERTForCausalLM,
    default_config=_CFG_BASE,
)
def bert_base_causal_lm(
    pretrained: bool | str = False, *, weights: BERTBaseWeights | None = None, **overrides: object
) -> BERTForCausalLM:
    r"""Construct a BERT-Base trunk with a left-to-right LM head.

    The trunk of :func:`bert_base` under a causal attention mask, with the
    tied vocabulary head predicting the *next* token rather than a masked
    one.  BERT was not pre-trained this way — the checkpoints saw
    bidirectional context — so the released weights are a starting point
    for causal fine-tuning, not a generative model as they stand.

    Parameters
    ----------
    pretrained : bool or str, default=False
        Encoder-weight selector.  ``False`` → fully random init; ``True``
        → loads the :func:`bert_base` encoder
        (:attr:`BERTBaseWeights.DEFAULT`) into the ``.bert`` trunk.
    weights : BERTBaseWeights, optional, keyword-only
        Explicit encoder-weights enum member; takes precedence over
        ``pretrained``.
    **overrides : object
        Optional :class:`BERTConfig` field overrides.

    Returns
    -------
    BERTForCausalLM
        BERT-Base wrapped with the causal LM head.

    Notes
    -----
    Reference: Devlin, Chang, Lee, and Toutanova, *"BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding"*, NAACL 2019
    (arXiv:1810.04805).  The paper's §3.1 objective is masked, not causal;
    this wrapper exists because the architecture supports the mask and
    downstream code asks for it, and :func:`bert_base_mlm` is the head the
    checkpoints were trained with.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.bert import bert_base_causal_lm
    >>> model = bert_base_causal_lm(vocab_size=99, hidden_size=32,
    ...                             num_hidden_layers=2,
    ...                             num_attention_heads=2,
    ...                             intermediate_size=64).eval()
    >>> out = model(lucid.tensor([[1, 2, 3, 4]]))
    >>> out.logits.shape   # (1, T=4, vocab_size=99)
    (1, 4, 99)
    """
    entry = weights_mod.resolve_weights(BERTBaseWeights, pretrained, weights)
    model = BERTForCausalLM(apply_overrides(_CFG_BASE, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model.bert, entry, name="bert_base")
    return model


@register_model(
    task="sequence-classification",
    family="bert",
    model_type="bert",
    model_class=BERTForNextSentencePrediction,
    default_config=_CFG_BASE,
)
def bert_base_nsp(
    pretrained: bool | str = False, *, weights: BERTBaseWeights | None = None, **overrides: object
) -> BERTForNextSentencePrediction:
    r"""Construct BERT-Base with the next-sentence-prediction head.

    The binary head from pre-training task #2: given a sentence pair
    packed with ``token_type_ids``, score whether the second follows the
    first.  Unlike the fine-tuning heads, this one *was* trained — it is
    half of the original objective — so the released encoder and this head
    belong together.

    Parameters
    ----------
    pretrained : bool or str, default=False
        Encoder-weight selector.  ``False`` → fully random init; ``True``
        → loads the :func:`bert_base` encoder
        (:attr:`BERTBaseWeights.DEFAULT`) into the ``.bert`` trunk.  **The
        pooler and the two-way classifier are randomly initialised** — the
        distributed checkpoint drops the NSP head.
    weights : BERTBaseWeights, optional, keyword-only
        Explicit encoder-weights enum member; takes precedence over
        ``pretrained``.
    **overrides : object
        Optional :class:`BERTConfig` field overrides.

    Returns
    -------
    BERTForNextSentencePrediction
        BERT-Base wrapped with the two-way sentence-pair head.

    Notes
    -----
    Reference: Devlin, Chang, Lee, and Toutanova, *"BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding"*, NAACL 2019
    (arXiv:1810.04805) §3.1 (task #2).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.bert import bert_base_nsp
    >>> model = bert_base_nsp(vocab_size=99, hidden_size=32,
    ...                       num_hidden_layers=2, num_attention_heads=2,
    ...                       intermediate_size=64).eval()
    >>> out = model(lucid.tensor([[1, 2, 3, 4]]))
    >>> out.logits.shape   # (N=1, 2) — IsNext / NotNext
    (1, 2)
    """
    entry = weights_mod.resolve_weights(BERTBaseWeights, pretrained, weights)
    model = BERTForNextSentencePrediction(apply_overrides(_CFG_BASE, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model.bert, entry, name="bert_base")
    return model


@register_model(
    task="sequence-classification",
    family="bert",
    model_type="bert",
    model_class=BERTForPreTraining,
    default_config=_CFG_BASE,
)
def bert_base_pretrain(
    pretrained: bool | str = False, *, weights: BERTBaseWeights | None = None, **overrides: object
) -> BERTForPreTraining:
    r"""Construct BERT-Base with both pre-training heads attached.

    The masked-LM head and the next-sentence head at once, which is the
    configuration the paper trains: the two losses are summed, and the
    forward returns each separately so a caller can weight or inspect
    them.  This is the factory for continued pre-training on a domain
    corpus; :func:`bert_base_mlm` is the one for mask-filling alone.

    Parameters
    ----------
    pretrained : bool or str, default=False
        Encoder-weight selector.  ``False`` → fully random init; ``True``
        → loads the :func:`bert_base` encoder
        (:attr:`BERTBaseWeights.DEFAULT`) into the ``.bert`` trunk.  Both
        heads are randomly initialised.
    weights : BERTBaseWeights, optional, keyword-only
        Explicit encoder-weights enum member; takes precedence over
        ``pretrained``.
    **overrides : object
        Optional :class:`BERTConfig` field overrides.

    Returns
    -------
    BERTForPreTraining
        BERT-Base wrapped with the masked-LM and next-sentence heads.

    Notes
    -----
    Reference: Devlin, Chang, Lee, and Toutanova, *"BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding"*, NAACL 2019
    (arXiv:1810.04805) §3.1.

    The output carries ``prediction_logits`` and
    ``seq_relationship_logits`` rather than a single ``logits`` — two heads
    produce two predictions, and ``mlm_loss`` / ``nsp_loss`` are reported
    beside the summed ``loss`` so the balance between them is visible.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.bert import bert_base_pretrain
    >>> model = bert_base_pretrain(vocab_size=99, hidden_size=32,
    ...                            num_hidden_layers=2,
    ...                            num_attention_heads=2,
    ...                            intermediate_size=64).eval()
    >>> out = model(lucid.tensor([[1, 2, 3, 4]]))
    >>> out.prediction_logits.shape, out.seq_relationship_logits.shape
    ((1, 4, 99), (1, 2))
    """
    entry = weights_mod.resolve_weights(BERTBaseWeights, pretrained, weights)
    model = BERTForPreTraining(apply_overrides(_CFG_BASE, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model.bert, entry, name="bert_base")
    return model
