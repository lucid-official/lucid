"""Registry factories for ConvNeXt variants."""

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models.vision.convnext._config import ConvNeXtConfig
from lucid.models.vision.convnext._model import ConvNeXt, ConvNeXtForImageClassification
from lucid.models.vision.convnext._weights import (
    ConvNeXtBaseWeights,
    ConvNeXtLargeWeights,
    ConvNeXtSmallWeights,
    ConvNeXtTinyWeights,
    ConvNeXtXLargeWeights,
)
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained

_CFG_T = ConvNeXtConfig(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768))
_CFG_S = ConvNeXtConfig(depths=(3, 3, 27, 3), dims=(96, 192, 384, 768))
_CFG_B = ConvNeXtConfig(depths=(3, 3, 27, 3), dims=(128, 256, 512, 1024))
_CFG_L = ConvNeXtConfig(depths=(3, 3, 27, 3), dims=(192, 384, 768, 1536))
_CFG_XL = ConvNeXtConfig(depths=(3, 3, 27, 3), dims=(256, 512, 1024, 2048))


# ── Backbones ─────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="convnext",
    model_type="convnext",
    model_class=ConvNeXt,
    default_config=_CFG_T,
    params=28_589_128,
)
def convnext_tiny(pretrained: bool = False, **overrides: object) -> ConvNeXt:
    r"""ConvNeXt-Tiny backbone (Liu et al., 2022).

    Builds the canonical *ConvNeXt-T* configuration: ``depths=(3, 3, 9, 3)``,
    ``dims=(96, 192, 384, 768)``.  The final stage produces a
    :math:`(B, 768, H/32, W/32)` feature map; ``forward_features``
    globally averages and LayerNorms to a :math:`(B, 768)` vector.
    Approximately **29M parameters** — Swin-T equivalent.

    Parameters
    ----------
    pretrained : bool, optional
        No checkpoint is published for ``convnext_tiny``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``convnext_tiny_cls`` is the checkpointed variant.
    **overrides : object
        Keyword overrides on top of the canonical ConvNeXt-T config.
        Each override must match a field of :class:`ConvNeXtConfig`.

    Returns
    -------
    ConvNeXt
        A :class:`ConvNeXt` backbone returning a flat
        :math:`(B, 768)` feature.

    Notes
    -----
    ConvNeXt-T reaches **82.1% top-1 on ImageNet-1k** (Liu et al., 2022,
    Table 1).  See `arXiv:2201.03545 <https://arxiv.org/abs/2201.03545>`_.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.convnext import convnext_tiny
    >>> model = convnext_tiny()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> feat = model.forward_features(x)
    >>> feat.shape
    (1, 768)
    """
    if pretrained:
        reject_unavailable_pretrained("convnext_tiny", alternative="convnext_tiny_cls")
    return ConvNeXt(apply_overrides(_CFG_T, overrides))


@register_model(
    task="base",
    family="convnext",
    model_type="convnext",
    model_class=ConvNeXt,
    default_config=_CFG_S,
    params=50_223_688,
)
def convnext_small(pretrained: bool = False, **overrides: object) -> ConvNeXt:
    r"""ConvNeXt-Small backbone (Liu et al., 2022).

    Builds the canonical *ConvNeXt-S* configuration: ``depths=(3, 3, 27, 3)``
    (3x deeper stage 3 than ConvNeXt-T), ``dims=(96, 192, 384, 768)``.
    Approximately **50M parameters** — Swin-S equivalent.

    Parameters
    ----------
    pretrained : bool, optional
        No checkpoint is published for ``convnext_small``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``convnext_small_cls`` is the checkpointed variant.
    **overrides : object
        Keyword overrides on top of the canonical ConvNeXt-S config.

    Returns
    -------
    ConvNeXt
        A :class:`ConvNeXt` backbone returning a flat :math:`(B, 768)`
        feature.

    Notes
    -----
    ConvNeXt-S reaches **83.1% top-1 on ImageNet-1k** (Liu et al., 2022,
    Table 1).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.convnext import convnext_small
    >>> model = convnext_small()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model.forward_features(x).shape
    (1, 768)
    """
    if pretrained:
        reject_unavailable_pretrained("convnext_small", alternative="convnext_small_cls")
    return ConvNeXt(apply_overrides(_CFG_S, overrides))


@register_model(
    task="base",
    family="convnext",
    model_type="convnext",
    model_class=ConvNeXt,
    default_config=_CFG_B,
    params=88_591_464,
)
def convnext_base(pretrained: bool = False, **overrides: object) -> ConvNeXt:
    r"""ConvNeXt-Base backbone (Liu et al., 2022).

    Builds the canonical *ConvNeXt-B* configuration:
    ``depths=(3, 3, 27, 3)``, ``dims=(128, 256, 512, 1024)``.  Final
    feature width is 1024.  Approximately **89M parameters** — Swin-B
    equivalent.

    Parameters
    ----------
    pretrained : bool, optional
        No checkpoint is published for ``convnext_base``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``convnext_base_cls`` is the checkpointed variant.
    **overrides : object
        Keyword overrides on top of the canonical ConvNeXt-B config.

    Returns
    -------
    ConvNeXt
        A :class:`ConvNeXt` backbone returning a flat :math:`(B, 1024)`
        feature.

    Notes
    -----
    ConvNeXt-B reaches **83.8% top-1 on ImageNet-1k** (224x224).  With
    ImageNet-22k pretraining it reaches **85.8%** at 224x224 and
    **86.8%** at 384x384 (Liu et al., 2022, Table 1 lower block; the
    per-resolution breakdown is Table 12 — Table 11 is "Detailed results
    for modernizing a ResNet-200").

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.convnext import convnext_base
    >>> model = convnext_base()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model.forward_features(x).shape
    (1, 1024)
    """
    if pretrained:
        reject_unavailable_pretrained("convnext_base", alternative="convnext_base_cls")
    return ConvNeXt(apply_overrides(_CFG_B, overrides))


@register_model(
    task="base",
    family="convnext",
    model_type="convnext",
    model_class=ConvNeXt,
    default_config=_CFG_L,
    params=197_767_336,
)
def convnext_large(pretrained: bool = False, **overrides: object) -> ConvNeXt:
    r"""ConvNeXt-Large backbone (Liu et al., 2022).

    Builds the canonical *ConvNeXt-L* configuration:
    ``depths=(3, 3, 27, 3)``, ``dims=(192, 384, 768, 1536)``.  Final
    feature width is 1536.  Approximately **198M parameters** — Swin-L
    equivalent.

    Parameters
    ----------
    pretrained : bool, optional
        No checkpoint is published for ``convnext_large``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``convnext_large_cls`` is the checkpointed variant.
    **overrides : object
        Keyword overrides on top of the canonical ConvNeXt-L config.

    Returns
    -------
    ConvNeXt
        A :class:`ConvNeXt` backbone returning a flat :math:`(B, 1536)`
        feature.

    Notes
    -----
    ConvNeXt-L reaches **84.3% top-1 on ImageNet-1k** (224x224) and
    **86.6% top-1** at 224x224 — and **87.5%** at 384x384 — after
    ImageNet-22k pretraining (Liu et al., 2022, Table 1 lower block;
    per-resolution detail in Table 12).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.convnext import convnext_large
    >>> model = convnext_large()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model.forward_features(x).shape
    (1, 1536)
    """
    if pretrained:
        reject_unavailable_pretrained("convnext_large", alternative="convnext_large_cls")
    return ConvNeXt(apply_overrides(_CFG_L, overrides))


@register_model(
    task="base",
    family="convnext",
    model_type="convnext",
    model_class=ConvNeXt,
    default_config=_CFG_XL,
    params=350_196_968,
)
def convnext_xlarge(pretrained: bool = False, **overrides: object) -> ConvNeXt:
    r"""ConvNeXt-XLarge backbone (Liu et al., 2022).

    Builds the canonical *ConvNeXt-XL* configuration:
    ``depths=(3, 3, 27, 3)``, ``dims=(256, 512, 1024, 2048)``.  Final
    feature width is 2048 — twice the ConvNeXt-B width.  Approximately
    **350M parameters** — the largest variant in the paper.

    Parameters
    ----------
    pretrained : bool, optional
        No checkpoint is published for ``convnext_xlarge``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``convnext_xlarge_cls`` is the checkpointed variant.
    **overrides : object
        Keyword overrides on top of the canonical ConvNeXt-XL config.

    Returns
    -------
    ConvNeXt
        A :class:`ConvNeXt` backbone returning a flat :math:`(B, 2048)`
        feature.

    Notes
    -----
    ConvNeXt-XL reaches **87.0% top-1 on ImageNet-1k** at 224x224 —
    and **87.8%** at 384x384 — after ImageNet-22k pretraining (Liu
    et al., 2022, Table 1 lower block; per-resolution detail in
    Table 12).  87.0 is the 224 number, which is what this family's
    weight entry preprocesses for.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.convnext import convnext_xlarge
    >>> model = convnext_xlarge()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model.forward_features(x).shape
    (1, 2048)
    """
    if pretrained:
        reject_unavailable_pretrained("convnext_xlarge", alternative="convnext_xlarge_cls")
    return ConvNeXt(apply_overrides(_CFG_XL, overrides))


# ── Classifiers ───────────────────────────────────────────────────────────────


@register_model(
    task="image-classification",
    family="convnext",
    model_type="convnext",
    model_class=ConvNeXtForImageClassification,
    default_config=_CFG_T,
    params=28_589_128,
)
def convnext_tiny_cls(
    pretrained: bool | str = False,
    *,
    weights: ConvNeXtTinyWeights | None = None,
    **overrides: object,
) -> ConvNeXtForImageClassification:
    r"""ConvNeXt-Tiny image classifier (Liu et al., 2022).

    Combines the :func:`convnext_tiny` backbone with a global average
    pool + LayerNorm + single :class:`nn.Linear` classification head.
    Default output is ``num_classes=1000`` (ImageNet-1k).  ~29M
    parameters.

    Parameters
    ----------
    pretrained : bool, optional
        If ``True``, loads ImageNet-1k pretrained weights when
        available.  Defaults to ``False``.
    weights : ConvNeXtTinyWeights, optional, keyword-only
        Explicit weights enum member; takes precedence over ``pretrained``.
    **overrides : object
        Keyword overrides on top of the canonical ConvNeXt-T config.

    Returns
    -------
    ConvNeXtForImageClassification
        Classifier returning :class:`ImageClassificationOutput` whose
        ``logits`` has shape ``(B, num_classes)``.

    Notes
    -----
    ConvNeXt-T reaches **82.1% top-1 on ImageNet-1k** (Liu et al.,
    2022, Table 1).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.convnext import convnext_tiny_cls
    >>> model = convnext_tiny_cls(num_classes=1000)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model(x).logits.shape
    (1, 1000)

    Load ImageNet-pretrained weights:

    >>> model = convnext_tiny_cls(pretrained=True)            # DEFAULT tag
    >>> from lucid.models.weights import ConvNeXtTinyWeights
    >>> model = convnext_tiny_cls(weights=ConvNeXtTinyWeights.IMAGENET1K_V1)
    """
    entry = weights_mod.resolve_weights(ConvNeXtTinyWeights, pretrained, weights)
    model = ConvNeXtForImageClassification(apply_overrides(_CFG_T, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="convnext_tiny_cls")
    return model


@register_model(
    task="image-classification",
    family="convnext",
    model_type="convnext",
    model_class=ConvNeXtForImageClassification,
    default_config=_CFG_S,
    params=50_223_688,
)
def convnext_small_cls(
    pretrained: bool | str = False,
    *,
    weights: ConvNeXtSmallWeights | None = None,
    **overrides: object,
) -> ConvNeXtForImageClassification:
    r"""ConvNeXt-Small image classifier (Liu et al., 2022).

    Combines the :func:`convnext_small` backbone (``depths=(3, 3, 27, 3)``,
    ``dims=(96, 192, 384, 768)``) with a global average pool +
    LayerNorm + linear classification head.  ~50M parameters.

    Parameters
    ----------
    pretrained : bool, optional
        If ``True``, loads ImageNet-1k pretrained weights when
        available.  Defaults to ``False``.
    weights : ConvNeXtSmallWeights, optional, keyword-only
        Explicit weights enum member; takes precedence over ``pretrained``.
    **overrides : object
        Keyword overrides on top of the canonical ConvNeXt-S config.

    Returns
    -------
    ConvNeXtForImageClassification
        Classifier whose ``logits`` has shape ``(B, num_classes)``.

    Notes
    -----
    ConvNeXt-S reaches **83.1% top-1 on ImageNet-1k** (Liu et al.,
    2022, Table 1).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.convnext import convnext_small_cls
    >>> model = convnext_small_cls(num_classes=1000)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model(x).logits.shape
    (1, 1000)

    Load ImageNet-pretrained weights:

    >>> model = convnext_small_cls(pretrained=True)
    >>> from lucid.models.weights import ConvNeXtSmallWeights
    >>> model = convnext_small_cls(weights=ConvNeXtSmallWeights.IMAGENET1K_V1)
    """
    entry = weights_mod.resolve_weights(ConvNeXtSmallWeights, pretrained, weights)
    model = ConvNeXtForImageClassification(apply_overrides(_CFG_S, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="convnext_small_cls")
    return model


@register_model(
    task="image-classification",
    family="convnext",
    model_type="convnext",
    model_class=ConvNeXtForImageClassification,
    default_config=_CFG_B,
    params=88_591_464,
)
def convnext_base_cls(
    pretrained: bool | str = False,
    *,
    weights: ConvNeXtBaseWeights | None = None,
    **overrides: object,
) -> ConvNeXtForImageClassification:
    r"""ConvNeXt-Base image classifier (Liu et al., 2022).

    Combines the :func:`convnext_base` backbone (``depths=(3, 3, 27, 3)``,
    ``dims=(128, 256, 512, 1024)``) with a global average pool +
    LayerNorm + linear classification head.  ~89M parameters.

    Parameters
    ----------
    pretrained : bool, optional
        If ``True``, loads ImageNet-1k or ImageNet-22k pretrained
        weights when available.  Defaults to ``False``.
    weights : ConvNeXtBaseWeights, optional, keyword-only
        Explicit weights enum member; takes precedence over ``pretrained``.
    **overrides : object
        Keyword overrides on top of the canonical ConvNeXt-B config.

    Returns
    -------
    ConvNeXtForImageClassification
        Classifier whose ``logits`` has shape ``(B, num_classes)``.

    Notes
    -----
    ConvNeXt-B reaches **83.8% top-1 on ImageNet-1k** (224x224).  With
    ImageNet-22k pretraining it reaches **85.8%** at 224x224 and
    **86.8%** at 384x384 (Liu et al., 2022, Table 1 lower block; the
    per-resolution breakdown is Table 12 — Table 11 is "Detailed results
    for modernizing a ResNet-200").

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.convnext import convnext_base_cls
    >>> model = convnext_base_cls(num_classes=1000)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model(x).logits.shape
    (1, 1000)

    Load ImageNet-pretrained weights:

    >>> model = convnext_base_cls(pretrained=True)
    >>> from lucid.models.weights import ConvNeXtBaseWeights
    >>> model = convnext_base_cls(weights=ConvNeXtBaseWeights.IMAGENET1K_V1)
    """
    entry = weights_mod.resolve_weights(ConvNeXtBaseWeights, pretrained, weights)
    model = ConvNeXtForImageClassification(apply_overrides(_CFG_B, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="convnext_base_cls")
    return model


@register_model(
    task="image-classification",
    family="convnext",
    model_type="convnext",
    model_class=ConvNeXtForImageClassification,
    default_config=_CFG_L,
    params=197_767_336,
)
def convnext_large_cls(
    pretrained: bool | str = False,
    *,
    weights: ConvNeXtLargeWeights | None = None,
    **overrides: object,
) -> ConvNeXtForImageClassification:
    r"""ConvNeXt-Large image classifier (Liu et al., 2022).

    Combines the :func:`convnext_large` backbone (``depths=(3, 3, 27, 3)``,
    ``dims=(192, 384, 768, 1536)``) with a global average pool +
    LayerNorm + linear classification head.  ~198M parameters.

    Parameters
    ----------
    pretrained : bool, optional
        If ``True``, loads ImageNet-22k pretrained weights when
        available.  Defaults to ``False``.
    weights : ConvNeXtLargeWeights, optional, keyword-only
        Explicit weights enum member; takes precedence over ``pretrained``.
    **overrides : object
        Keyword overrides on top of the canonical ConvNeXt-L config.

    Returns
    -------
    ConvNeXtForImageClassification
        Classifier whose ``logits`` has shape ``(B, num_classes)``.

    Notes
    -----
    ConvNeXt-L reaches **84.3% top-1 on ImageNet-1k** (224x224) and
    **86.6% top-1** at 224x224 — and **87.5%** at 384x384 — after
    ImageNet-22k pretraining (Liu et al., 2022, Table 1 lower block;
    per-resolution detail in Table 12).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.convnext import convnext_large_cls
    >>> model = convnext_large_cls(num_classes=1000)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model(x).logits.shape
    (1, 1000)

    Load ImageNet-pretrained weights:

    >>> model = convnext_large_cls(pretrained=True)
    >>> from lucid.models.weights import ConvNeXtLargeWeights
    >>> model = convnext_large_cls(weights=ConvNeXtLargeWeights.IMAGENET1K_V1)
    """
    entry = weights_mod.resolve_weights(ConvNeXtLargeWeights, pretrained, weights)
    model = ConvNeXtForImageClassification(apply_overrides(_CFG_L, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="convnext_large_cls")
    return model


@register_model(
    task="image-classification",
    family="convnext",
    model_type="convnext",
    model_class=ConvNeXtForImageClassification,
    default_config=_CFG_XL,
    params=350_196_968,
)
def convnext_xlarge_cls(
    pretrained: bool | str = False,
    *,
    weights: ConvNeXtXLargeWeights | None = None,
    **overrides: object,
) -> ConvNeXtForImageClassification:
    r"""ConvNeXt-XLarge image classifier (Liu et al., 2022).

    Combines the :func:`convnext_xlarge` backbone (``depths=(3, 3, 27, 3)``,
    ``dims=(256, 512, 1024, 2048)``) with a global average pool +
    LayerNorm + linear classification head.  ~350M parameters — the
    largest ConvNeXt variant in the paper.

    Parameters
    ----------
    pretrained : bool, optional
        If ``True``, loads ImageNet-22k pretrained weights when
        available.  Defaults to ``False``.
    weights : ConvNeXtXLargeWeights, optional, keyword-only
        Explicit weights enum member; takes precedence over ``pretrained``.
    **overrides : object
        Keyword overrides on top of the canonical ConvNeXt-XL config.

    Returns
    -------
    ConvNeXtForImageClassification
        Classifier whose ``logits`` has shape ``(B, num_classes)``.

    Notes
    -----
    ConvNeXt-XL reaches **87.0% top-1 on ImageNet-1k** at 224x224 —
    and **87.8%** at 384x384 — after ImageNet-22k pretraining (Liu
    et al., 2022, Table 1 lower block; per-resolution detail in
    Table 12).  87.0 is the 224 number, which is what this family's
    weight entry preprocesses for.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.convnext import convnext_xlarge_cls
    >>> model = convnext_xlarge_cls(num_classes=1000)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model(x).logits.shape
    (1, 1000)

    Load ImageNet-22k-pretrained-then-1k-finetuned weights:

    >>> model = convnext_xlarge_cls(pretrained=True)
    >>> from lucid.models.weights import ConvNeXtXLargeWeights
    >>> model = convnext_xlarge_cls(weights=ConvNeXtXLargeWeights.FB_IN22K_FT_IN1K)
    """
    entry = weights_mod.resolve_weights(ConvNeXtXLargeWeights, pretrained, weights)
    model = ConvNeXtForImageClassification(apply_overrides(_CFG_XL, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="convnext_xlarge_cls")
    return model
