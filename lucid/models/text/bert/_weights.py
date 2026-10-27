"""Pretrained-weight declarations for the BERT family.

Eight checkpoints ship through the :mod:`lucid.weights` system, all
converted from the upstream Hugging Face checkpoints (a pure identity
parameter map — Lucid mirrors the ``BERTModel`` naming one-for-one):

* **Base encoders** (:class:`BERTTinyWeights` … :class:`BERTLargeWeights`)
  — the bare encoder trunk.  The four miniatures are the Turc et al. 2019
  "Well-Read Students" pre-distilled sizes; ``base`` / ``large`` are the
  original Devlin et al. 2018 ``bert-*-uncased`` checkpoints.
* **Masked-LM heads** (:class:`BERTBaseMLMWeights`,
  :class:`BERTLargeMLMWeights`) — encoder + the ``cls.predictions`` head
  used for the pre-training objective, re-tied to the input embedding.

All checkpoints were pre-trained on the Wikipedia + BookCorpus corpus
(uncased WordPiece, 30 522-token vocabulary).  Text models consume token
ids directly, so the entries carry a no-op preprocessing transform; tokenize
inputs with the matching :class:`lucid.models.text.bert.BERTTokenizer`.
"""

from lucid.utils.transforms import Compose
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights

# Text preprocessing is tokenization (a lucid.utils.tokenizer concern), not a
# tensor transform — entries carry an explicit no-op so `.transforms()` is safe.
_NOOP = Compose([])

_TAG = "WIKIPEDIA_BOOKSCORPUS"
_LICENSE = "apache-2.0"


def _url(slug: str) -> str:
    return f"{HUB_BASE}/{slug}/resolve/main/{_TAG}/model.safetensors"


def _url_tag(slug: str, tag: str) -> str:
    return f"{HUB_BASE}/{slug}/resolve/main/{tag}/model.safetensors"


@register_weights("bert_tiny")
class BERTTinyWeights(WeightsEnum):
    r"""Pretrained weight tags for :func:`lucid.models.bert_tiny`.

    Turc et al. 2019 ``L=2, H=128, A=2`` distilled encoder (4.4 M params).

    Attributes
    ----------
    WIKIPEDIA_BOOKSCORPUS : WeightEntry
        Wikipedia + BookCorpus pre-training checkpoint sourced from
        ``transformers/google/bert_uncased_L-2_H-128_A-2``.
    DEFAULT : WeightEntry
        Alias for :attr:`WIKIPEDIA_BOOKSCORPUS`.

    Notes
    -----
    Reference: Turc, Chang, Lee, Toutanova, *"Well-Read Students Learn
    Better"*, 2019 (arXiv:1908.08962).

    Examples
    --------
    >>> from lucid.models import bert_tiny
    >>> model = bert_tiny(pretrained=True).eval()
    """

    WIKIPEDIA_BOOKSCORPUS = WeightEntry(
        url=_url("bert-tiny"),
        sha256="3572b8818d3e4e9e46507b327224348f32d838bd7e75bff140dc5728d4ff6cc2",
        num_classes=128,
        transforms=_NOOP,
        meta={
            "tag": _TAG,
            "source": "transformers/google/bert_uncased_L-2_H-128_A-2",
            "license": _LICENSE,
            "num_params": 4_385_920,
            "file_size_mb": 16.74,
        },
    )
    DEFAULT = WIKIPEDIA_BOOKSCORPUS


@register_weights("bert_mini")
class BERTMiniWeights(WeightsEnum):
    r"""Pretrained weight tags for :func:`lucid.models.bert_mini`.

    Turc et al. 2019 ``L=4, H=256, A=4`` distilled encoder (11.2 M params).

    Attributes
    ----------
    WIKIPEDIA_BOOKSCORPUS : WeightEntry
        Wikipedia + BookCorpus pre-training checkpoint sourced from
        ``transformers/google/bert_uncased_L-4_H-256_A-4``.
    DEFAULT : WeightEntry
        Alias for :attr:`WIKIPEDIA_BOOKSCORPUS`.

    Notes
    -----
    Reference: Turc, Chang, Lee, Toutanova, *"Well-Read Students Learn
    Better"*, 2019 (arXiv:1908.08962).

    Examples
    --------
    >>> from lucid.models import bert_mini
    >>> model = bert_mini(pretrained=True).eval()
    """

    WIKIPEDIA_BOOKSCORPUS = WeightEntry(
        url=_url("bert-mini"),
        sha256="c644db652fbc91ae2e2d55a9efe8fb0b5f582448420093ba5c65cff7bcb7b5d9",
        num_classes=256,
        transforms=_NOOP,
        meta={
            "tag": _TAG,
            "source": "transformers/google/bert_uncased_L-4_H-256_A-4",
            "license": _LICENSE,
            "num_params": 11_170_560,
            "file_size_mb": 42.62,
        },
    )
    DEFAULT = WIKIPEDIA_BOOKSCORPUS


@register_weights("bert_small")
class BERTSmallWeights(WeightsEnum):
    r"""Pretrained weight tags for :func:`lucid.models.bert_small`.

    Turc et al. 2019 ``L=4, H=512, A=8`` distilled encoder (28.8 M params).

    Attributes
    ----------
    WIKIPEDIA_BOOKSCORPUS : WeightEntry
        Wikipedia + BookCorpus pre-training checkpoint sourced from
        ``transformers/google/bert_uncased_L-4_H-512_A-8``.
    DEFAULT : WeightEntry
        Alias for :attr:`WIKIPEDIA_BOOKSCORPUS`.

    Notes
    -----
    Reference: Turc, Chang, Lee, Toutanova, *"Well-Read Students Learn
    Better"*, 2019 (arXiv:1908.08962).

    Examples
    --------
    >>> from lucid.models import bert_small
    >>> model = bert_small(pretrained=True).eval()
    """

    WIKIPEDIA_BOOKSCORPUS = WeightEntry(
        url=_url("bert-small"),
        sha256="ff63fc91d94ae3cb1240f7ad147db7d6f45958c544ee00c1c9e264cb45675979",
        num_classes=512,
        transforms=_NOOP,
        meta={
            "tag": _TAG,
            "source": "transformers/google/bert_uncased_L-4_H-512_A-8",
            "license": _LICENSE,
            "num_params": 28_763_648,
            "file_size_mb": 109.73,
        },
    )
    DEFAULT = WIKIPEDIA_BOOKSCORPUS


@register_weights("bert_medium")
class BERTMediumWeights(WeightsEnum):
    r"""Pretrained weight tags for :func:`lucid.models.bert_medium`.

    Turc et al. 2019 ``L=8, H=512, A=8`` distilled encoder (41.4 M params).

    Attributes
    ----------
    WIKIPEDIA_BOOKSCORPUS : WeightEntry
        Wikipedia + BookCorpus pre-training checkpoint sourced from
        ``transformers/google/bert_uncased_L-8_H-512_A-8``.
    DEFAULT : WeightEntry
        Alias for :attr:`WIKIPEDIA_BOOKSCORPUS`.

    Notes
    -----
    Reference: Turc, Chang, Lee, Toutanova, *"Well-Read Students Learn
    Better"*, 2019 (arXiv:1908.08962).

    Examples
    --------
    >>> from lucid.models import bert_medium
    >>> model = bert_medium(pretrained=True).eval()
    """

    WIKIPEDIA_BOOKSCORPUS = WeightEntry(
        url=_url("bert-medium"),
        sha256="2edd0b9805b731de1c625308449f6e8cb8487c1982bbff22cf2f0a07abbdf886",
        num_classes=512,
        transforms=_NOOP,
        meta={
            "tag": _TAG,
            "source": "transformers/google/bert_uncased_L-8_H-512_A-8",
            "license": _LICENSE,
            "num_params": 41_373_184,
            "file_size_mb": 157.84,
        },
    )
    DEFAULT = WIKIPEDIA_BOOKSCORPUS


@register_weights("bert_base")
class BERTBaseWeights(WeightsEnum):
    r"""Pretrained weight tags for :func:`lucid.models.bert_base`.

    Devlin et al. 2018 ``L=12, H=768, A=12`` uncased encoder (109.5 M params).

    Attributes
    ----------
    WIKIPEDIA_BOOKSCORPUS : WeightEntry
        Wikipedia + BookCorpus pre-training checkpoint sourced from
        ``transformers/google-bert/bert-base-uncased``.
    DEFAULT : WeightEntry
        Alias for :attr:`WIKIPEDIA_BOOKSCORPUS`.

    Notes
    -----
    Reference: Devlin, Chang, Lee, Toutanova, *"BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding"*, NAACL
    2019 (arXiv:1810.04805).

    Examples
    --------
    >>> from lucid.models import bert_base
    >>> model = bert_base(pretrained=True).eval()
    """

    WIKIPEDIA_BOOKSCORPUS = WeightEntry(
        url=_url("bert-base"),
        sha256="a4faff5f2aab76df5bc433bef9b3f4e7e8628855885bf54256614095fa814a53",
        num_classes=768,
        transforms=_NOOP,
        meta={
            "tag": _TAG,
            "source": "transformers/google-bert/bert-base-uncased",
            "license": _LICENSE,
            "num_params": 109_482_240,
            "file_size_mb": 417.66,
        },
    )
    DEFAULT = WIKIPEDIA_BOOKSCORPUS


@register_weights("bert_large")
class BERTLargeWeights(WeightsEnum):
    r"""Pretrained weight tags for :func:`lucid.models.bert_large`.

    Devlin et al. 2018 ``L=24, H=1024, A=16`` uncased encoder (335.1 M params).

    Attributes
    ----------
    WIKIPEDIA_BOOKSCORPUS : WeightEntry
        Wikipedia + BookCorpus pre-training checkpoint sourced from
        ``transformers/google-bert/bert-large-uncased``.
    DEFAULT : WeightEntry
        Alias for :attr:`WIKIPEDIA_BOOKSCORPUS`.

    Notes
    -----
    Reference: Devlin, Chang, Lee, Toutanova, *"BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding"*, NAACL
    2019 (arXiv:1810.04805).

    Examples
    --------
    >>> from lucid.models import bert_large
    >>> model = bert_large(pretrained=True).eval()
    """

    WIKIPEDIA_BOOKSCORPUS = WeightEntry(
        url=_url("bert-large"),
        sha256="e15f6d8cc7bb0babc43228c83cde9a3e41ccdb3c066e78b8b071b1a16dacebb2",
        num_classes=1024,
        transforms=_NOOP,
        meta={
            "tag": _TAG,
            "source": "transformers/google-bert/bert-large-uncased",
            "license": _LICENSE,
            "num_params": 335_141_888,
            "file_size_mb": 1278.51,
        },
    )
    DEFAULT = WIKIPEDIA_BOOKSCORPUS


@register_weights("bert_base_mlm")
class BERTBaseMLMWeights(WeightsEnum):
    r"""Pretrained weight tags for :func:`lucid.models.bert_base_mlm`.

    BERT-Base encoder + tied MLM head over the 30 522 WordPiece vocab.

    Attributes
    ----------
    WIKIPEDIA_BOOKSCORPUS : WeightEntry
        Wikipedia + BookCorpus pre-training checkpoint (encoder + MLM
        head) sourced from ``transformers/google-bert/bert-base-uncased``.
    DEFAULT : WeightEntry
        Alias for :attr:`WIKIPEDIA_BOOKSCORPUS`.

    Notes
    -----
    Reference: Devlin, Chang, Lee, Toutanova, *"BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding"*, NAACL
    2019 (arXiv:1810.04805).

    Examples
    --------
    >>> from lucid.models import bert_base_mlm
    >>> model = bert_base_mlm(pretrained=True).eval()
    """

    WIKIPEDIA_BOOKSCORPUS = WeightEntry(
        url=_url("bert-base-mlm"),
        sha256="76c4910e80b5c31bee8cab879dc647a3976052ebe32b592cd522103a88b7b282",
        num_classes=30_522,
        transforms=_NOOP,
        meta={
            "tag": _TAG,
            "source": "transformers/google-bert/bert-base-uncased",
            "license": _LICENSE,
            "num_params": 110_104_890,
            "file_size_mb": 509.46,
        },
    )
    DEFAULT = WIKIPEDIA_BOOKSCORPUS


@register_weights("bert_large_mlm")
class BERTLargeMLMWeights(WeightsEnum):
    r"""Pretrained weight tags for :func:`lucid.models.bert_large_mlm`.

    BERT-Large encoder + tied MLM head over the 30 522 WordPiece vocab.

    Attributes
    ----------
    WIKIPEDIA_BOOKSCORPUS : WeightEntry
        Wikipedia + BookCorpus pre-training checkpoint (encoder + MLM
        head) sourced from ``transformers/google-bert/bert-large-uncased``.
    DEFAULT : WeightEntry
        Alias for :attr:`WIKIPEDIA_BOOKSCORPUS`.

    Notes
    -----
    Reference: Devlin, Chang, Lee, Toutanova, *"BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding"*, NAACL
    2019 (arXiv:1810.04805).

    Examples
    --------
    >>> from lucid.models import bert_large_mlm
    >>> model = bert_large_mlm(pretrained=True).eval()
    """

    WIKIPEDIA_BOOKSCORPUS = WeightEntry(
        url=_url("bert-large-mlm"),
        sha256="22b8affd371ec095d2b93fb25b4b16380fa7cd6395a7033aa07ca485df7d5a78",
        num_classes=30_522,
        transforms=_NOOP,
        meta={
            "tag": _TAG,
            "source": "transformers/google-bert/bert-large-uncased",
            "license": _LICENSE,
            "num_params": 336_224_058,
            "file_size_mb": 1401.87,
        },
    )
    DEFAULT = WIKIPEDIA_BOOKSCORPUS


# ── Fine-tuned task heads (canonical downstream-benchmark checkpoints) ─────────
#
# These are *full* fine-tuned models (encoder + task head), unlike the encoder
# tags above.  `pretrained=True` on the matching factory loads an inference-ready
# checkpoint.  The pooler is the checkpoint's own (untrained during fine-tuning,
# unused by the QA / token-cls head, kept so the Lucid ``BERTModel`` slot fills).

_SQUAD_TAG = "SQUAD_V1"
_CONLL_TAG = "CONLL2003"


@register_weights("bert_base_qa")
class BERTBaseQAWeights(WeightsEnum):
    r"""Fine-tuned SQuAD v1.1 weights for :func:`lucid.models.bert_base_qa`.

    BERT-Base encoder + 2-way span head; F1 88.1 / EM 80.9.

    Attributes
    ----------
    SQUAD_V1 : WeightEntry
        SQuAD v1.1 fine-tuned checkpoint (F1 88.1 / EM 80.9), sourced
        from ``transformers/csarron/bert-base-uncased-squad-v1``.
    DEFAULT : WeightEntry
        Alias for :attr:`SQUAD_V1`.

    Notes
    -----
    Reference: Devlin, Chang, Lee, Toutanova, *"BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding"*, NAACL
    2019 (arXiv:1810.04805).  Fine-tuned on SQuAD v1.1.

    Examples
    --------
    >>> from lucid.models import bert_base_qa
    >>> model = bert_base_qa(pretrained=True).eval()
    """

    SQUAD_V1 = WeightEntry(
        url=_url_tag("bert-base-squad", _SQUAD_TAG),
        sha256="00c936fe41ec21b9a8868c036e4866836ae17905362f0e9dcbf1d989d2b7728e",
        num_classes=2,
        transforms=_NOOP,
        meta={
            "tag": _SQUAD_TAG,
            "source": "transformers/csarron/bert-base-uncased-squad-v1",
            "license": "mit",
            "num_params": 109_483_778,
            "metrics": {"squad": {"f1": 88.1, "exact_match": 80.9}},
        },
    )
    DEFAULT = SQUAD_V1


@register_weights("bert_large_qa")
class BERTLargeQAWeights(WeightsEnum):
    r"""Fine-tuned SQuAD v1.1 weights for :func:`lucid.models.bert_large_qa`.

    The official Google whole-word-masking BERT-Large SQuAD checkpoint.

    Attributes
    ----------
    SQUAD_V1 : WeightEntry
        SQuAD v1.1 fine-tuned checkpoint (F1 93.2 / EM 86.9), sourced
        from ``transformers/google-bert/bert-large-uncased-whole-word-
        masking-finetuned-squad``.
    DEFAULT : WeightEntry
        Alias for :attr:`SQUAD_V1`.

    Notes
    -----
    Reference: Devlin, Chang, Lee, Toutanova, *"BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding"*, NAACL
    2019 (arXiv:1810.04805).  Fine-tuned on SQuAD v1.1.

    Examples
    --------
    >>> from lucid.models import bert_large_qa
    >>> model = bert_large_qa(pretrained=True).eval()
    """

    SQUAD_V1 = WeightEntry(
        url=_url_tag("bert-large-squad", _SQUAD_TAG),
        sha256="8f4b8d0abfff79465413ff2f99738fac29c42d4eca483bec571938f564dfc011",
        num_classes=2,
        transforms=_NOOP,
        meta={
            "tag": _SQUAD_TAG,
            "source": (
                "transformers/google-bert/" "bert-large-uncased-whole-word-masking-finetuned-squad"
            ),
            "license": "apache-2.0",
            "num_params": 335_143_938,
            "metrics": {"squad": {"f1": 93.2, "exact_match": 86.9}},
        },
    )
    DEFAULT = SQUAD_V1


@register_weights("bert_base_token_cls")
class BERTBaseNERWeights(WeightsEnum):
    r"""Fine-tuned CoNLL-2003 NER weights for :func:`lucid.models.bert_base_token_cls`.

    ``dslim/bert-base-NER`` — a **cased** BERT-Base (vocab 28 996) with a
    9-way BIO tag head (O, B/I-PER, B/I-ORG, B/I-LOC, B/I-MISC).  Tokenize
    with the cased :class:`BERTTokenizer` (``do_lower_case=False``).

    Attributes
    ----------
    CONLL2003 : WeightEntry
        CoNLL-2003 NER fine-tuned checkpoint (F1 91.3), sourced from
        ``transformers/dslim/bert-base-NER``.
    DEFAULT : WeightEntry
        Alias for :attr:`CONLL2003`.

    Notes
    -----
    Reference: Devlin, Chang, Lee, Toutanova, *"BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding"*, NAACL
    2019 (arXiv:1810.04805).  Fine-tuned on CoNLL-2003 NER
    (dslim/bert-base-NER).

    Examples
    --------
    >>> from lucid.models import bert_base_token_cls
    >>> model = bert_base_token_cls(pretrained=True).eval()
    """

    CONLL2003 = WeightEntry(
        url=_url_tag("bert-base-ner", _CONLL_TAG),
        sha256="9c40b1e767a636302bb5aa2cb1a9ef659bfc3804c468d10de0117962ca50b329",
        num_classes=9,
        transforms=_NOOP,
        meta={
            "tag": _CONLL_TAG,
            "source": "transformers/dslim/bert-base-NER",
            "license": "mit",
            "num_params": 109_483_778,
            "metrics": {"conll2003": {"f1": 91.3}},
        },
    )
    DEFAULT = CONLL2003
