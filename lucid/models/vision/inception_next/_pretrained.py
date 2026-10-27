"""Registry factories for InceptionNeXt variants."""

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models.vision.inception_next._config import InceptionNeXtConfig
from lucid.models.vision.inception_next._model import (
    InceptionNeXt,
    InceptionNeXtForImageClassification,
)
from lucid.models.vision.inception_next._weights import (
    InceptionNeXtBaseWeights,
    InceptionNeXtSmallWeights,
    InceptionNeXtTinyWeights,
)
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained

_CFG_T = InceptionNeXtConfig(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), band_kernel=11)
_CFG_S = InceptionNeXtConfig(depths=(3, 3, 27, 3), dims=(96, 192, 384, 768), band_kernel=11)
_CFG_B = InceptionNeXtConfig(depths=(3, 3, 27, 3), dims=(128, 256, 512, 1024), band_kernel=11)


# ── Backbones ─────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="inception_next",
    model_type="inception_next",
    model_class=InceptionNeXt,
    default_config=_CFG_T,
    params=23974296,
)
def inception_next_tiny(pretrained: bool = False, **overrides: object) -> InceptionNeXt:
    r"""InceptionNeXt-Tiny backbone (Yu et al., 2024).

    Builds the canonical *InceptionNeXt-T* configuration:
    ``depths=(3, 3, 9, 3)``, ``dims=(96, 192, 384, 768)``,
    ``band_kernel=11``.  Approximately **28M parameters** —
    ConvNeXt-T compatible, but with each :math:`7 \times 7` depthwise
    conv replaced by the four-branch InceptionDWConv mixer.

    Parameters
    ----------
    pretrained : bool, optional
        No checkpoint is published for ``inception_next_tiny``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``inception_next_tiny_cls`` is the checkpointed variant.
    **overrides : object
        Keyword overrides applied on top of the canonical
        InceptionNeXt-T config.  Each override must match a field of
        :class:`InceptionNeXtConfig`.

    Returns
    -------
    InceptionNeXt
        An :class:`InceptionNeXt` backbone returning a flat
        :math:`(B, 768)` feature.

    Notes
    -----
    InceptionNeXt-T matches ConvNeXt-T accuracy
    (**82.3% top-1 on ImageNet-1k**, Yu et al., 2024, Table 2) at
    noticeably lower latency.  See
    `arXiv:2303.16900 <https://arxiv.org/abs/2303.16900>`_.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.inception_next import inception_next_tiny
    >>> model = inception_next_tiny()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> feat = model.forward_features(x)
    >>> feat.shape
    (1, 768)
    """
    if pretrained:
        reject_unavailable_pretrained("inception_next_tiny", alternative="inception_next_tiny_cls")
    return InceptionNeXt(apply_overrides(_CFG_T, overrides))


# ── Classifiers ───────────────────────────────────────────────────────────────


@register_model(
    task="image-classification",
    family="inception_next",
    model_type="inception_next",
    model_class=InceptionNeXtForImageClassification,
    default_config=_CFG_T,
    params=28055680,
)
def inception_next_tiny_cls(
    pretrained: bool | str = False,
    *,
    weights: InceptionNeXtTinyWeights | None = None,
    **overrides: object,
) -> InceptionNeXtForImageClassification:
    r"""InceptionNeXt-Tiny image classifier (Yu et al., 2024).

    Combines the :func:`inception_next_tiny` backbone with the
    reference MLP classifier head (``Linear → GELU → LayerNorm →
    Linear``).  Default output is ``num_classes=1000`` (ImageNet-1k).
    Approximately **28M parameters**.

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`InceptionNeXtTinyWeights.SAIL_IN1K`);
        a tag string (e.g. ``"SAIL_IN1K"``) → that specific checkpoint.
        Mutually exclusive with ``weights`` (which wins if both given).
    weights : InceptionNeXtTinyWeights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides : object
        Keyword overrides on top of the canonical InceptionNeXt-T config.

    Returns
    -------
    InceptionNeXtForImageClassification
        Classifier returning :class:`ImageClassificationOutput` whose
        ``logits`` has shape ``(B, num_classes)``.

    Notes
    -----
    InceptionNeXt-T reaches **82.3% top-1 on ImageNet-1k** (Yu et al.,
    2024, Table 2).  Pretrained weights are converted from timm's
    ``inception_next_tiny.sail_in1k`` and hosted on the Hugging Face Hub
    under ``lucid-dl/inception-next-tiny``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.inception_next import inception_next_tiny_cls
    >>> model = inception_next_tiny_cls(num_classes=1000)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model(x).logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(InceptionNeXtTinyWeights, pretrained, weights)
    model = InceptionNeXtForImageClassification(apply_overrides(_CFG_T, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="inception_next_tiny_cls")
    return model


@register_model(
    task="base",
    family="inception_next",
    model_type="inception_next",
    model_class=InceptionNeXt,
    default_config=_CFG_S,
    params=45292632,
)
def inception_next_small(pretrained: bool = False, **overrides: object) -> InceptionNeXt:
    r"""InceptionNeXt-Small backbone (Yu et al., 2024).

    Builds the canonical *InceptionNeXt-S* configuration:
    ``depths=(3, 3, 27, 3)``, ``dims=(96, 192, 384, 768)``,
    ``band_kernel=11``.  Approximately **49M parameters** —
    InceptionNeXt-T widened in the third stage (9 → 27 blocks).

    Parameters
    ----------
    pretrained : bool, optional
        No checkpoint is published for ``inception_next_small``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``inception_next_small_cls`` is the checkpointed variant.
    **overrides : object
        Keyword overrides applied on top of the canonical
        InceptionNeXt-S config.

    Returns
    -------
    InceptionNeXt
        An :class:`InceptionNeXt` backbone returning a flat
        :math:`(B, 768)` feature.

    Notes
    -----
    InceptionNeXt-S reaches **83.5% top-1 on ImageNet-1k** (Yu et al.,
    2024, Table 2).  See
    `arXiv:2303.16900 <https://arxiv.org/abs/2303.16900>`_.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.inception_next import inception_next_small
    >>> model = inception_next_small()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> feat = model.forward_features(x)
    >>> feat.shape
    (1, 768)
    """
    if pretrained:
        reject_unavailable_pretrained(
            "inception_next_small", alternative="inception_next_small_cls"
        )
    return InceptionNeXt(apply_overrides(_CFG_S, overrides))


@register_model(
    task="image-classification",
    family="inception_next",
    model_type="inception_next",
    model_class=InceptionNeXtForImageClassification,
    default_config=_CFG_S,
    params=49374016,
)
def inception_next_small_cls(
    pretrained: bool | str = False,
    *,
    weights: InceptionNeXtSmallWeights | None = None,
    **overrides: object,
) -> InceptionNeXtForImageClassification:
    r"""InceptionNeXt-Small image classifier (Yu et al., 2024).

    Combines the :func:`inception_next_small` backbone with the
    reference MLP classifier head (``Linear → GELU → LayerNorm →
    Linear``).  Default output is ``num_classes=1000`` (ImageNet-1k).
    Approximately **49M parameters**.

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`InceptionNeXtSmallWeights.SAIL_IN1K`);
        a tag string → that specific checkpoint.  Mutually exclusive
        with ``weights`` (which wins if both given).
    weights : InceptionNeXtSmallWeights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides : object
        Keyword overrides on top of the canonical InceptionNeXt-S config.

    Returns
    -------
    InceptionNeXtForImageClassification
        Classifier returning :class:`ImageClassificationOutput` whose
        ``logits`` has shape ``(B, num_classes)``.

    Notes
    -----
    InceptionNeXt-S reaches **83.5% top-1 on ImageNet-1k** (Yu et al.,
    2024, Table 2).  Pretrained weights are converted from timm's
    ``inception_next_small.sail_in1k`` and hosted on the Hugging Face
    Hub under ``lucid-dl/inception-next-small``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.inception_next import inception_next_small_cls
    >>> model = inception_next_small_cls(num_classes=1000)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model(x).logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(InceptionNeXtSmallWeights, pretrained, weights)
    model = InceptionNeXtForImageClassification(apply_overrides(_CFG_S, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="inception_next_small_cls")
    return model


@register_model(
    task="base",
    family="inception_next",
    model_type="inception_next",
    model_class=InceptionNeXt,
    default_config=_CFG_B,
    params=80444192,
)
def inception_next_base(pretrained: bool = False, **overrides: object) -> InceptionNeXt:
    r"""InceptionNeXt-Base backbone (Yu et al., 2024).

    Builds the canonical *InceptionNeXt-B* configuration:
    ``depths=(3, 3, 27, 3)``, ``dims=(128, 256, 512, 1024)``,
    ``band_kernel=11``.  Approximately **87M parameters** — the widest
    of the paper's three main variants.

    Parameters
    ----------
    pretrained : bool, optional
        No checkpoint is published for ``inception_next_base``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``inception_next_base_cls`` is the checkpointed variant.
    **overrides : object
        Keyword overrides applied on top of the canonical
        InceptionNeXt-B config.

    Returns
    -------
    InceptionNeXt
        An :class:`InceptionNeXt` backbone returning a flat
        :math:`(B, 1024)` feature.

    Notes
    -----
    InceptionNeXt-B reaches **84.0% top-1 on ImageNet-1k** (Yu et al.,
    2024, Table 2).  See
    `arXiv:2303.16900 <https://arxiv.org/abs/2303.16900>`_.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.inception_next import inception_next_base
    >>> model = inception_next_base()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> feat = model.forward_features(x)
    >>> feat.shape
    (1, 1024)
    """
    if pretrained:
        reject_unavailable_pretrained("inception_next_base", alternative="inception_next_base_cls")
    return InceptionNeXt(apply_overrides(_CFG_B, overrides))


@register_model(
    task="image-classification",
    family="inception_next",
    model_type="inception_next",
    model_class=InceptionNeXtForImageClassification,
    default_config=_CFG_B,
    params=86672136,
)
def inception_next_base_cls(
    pretrained: bool | str = False,
    *,
    weights: InceptionNeXtBaseWeights | None = None,
    **overrides: object,
) -> InceptionNeXtForImageClassification:
    r"""InceptionNeXt-Base image classifier (Yu et al., 2024).

    Combines the :func:`inception_next_base` backbone with the
    reference MLP classifier head (``Linear → GELU → LayerNorm →
    Linear``).  Default output is ``num_classes=1000`` (ImageNet-1k).
    Approximately **87M parameters**.

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`InceptionNeXtBaseWeights.SAIL_IN1K`);
        a tag string → that specific checkpoint.  Mutually exclusive
        with ``weights`` (which wins if both given).
    weights : InceptionNeXtBaseWeights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides : object
        Keyword overrides on top of the canonical InceptionNeXt-B config.

    Returns
    -------
    InceptionNeXtForImageClassification
        Classifier returning :class:`ImageClassificationOutput` whose
        ``logits`` has shape ``(B, num_classes)``.

    Notes
    -----
    InceptionNeXt-B reaches **84.0% top-1 on ImageNet-1k** (Yu et al.,
    2024, Table 2).  Pretrained weights are converted from timm's
    ``inception_next_base.sail_in1k`` and hosted on the Hugging Face Hub
    under ``lucid-dl/inception-next-base``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.inception_next import inception_next_base_cls
    >>> model = inception_next_base_cls(num_classes=1000)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model(x).logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(InceptionNeXtBaseWeights, pretrained, weights)
    model = InceptionNeXtForImageClassification(apply_overrides(_CFG_B, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="inception_next_base_cls")
    return model
