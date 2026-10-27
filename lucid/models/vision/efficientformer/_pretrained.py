"""Registry factories for EfficientFormer variants."""

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models.vision.efficientformer._config import EfficientFormerConfig
from lucid.models.vision.efficientformer._model import (
    EfficientFormer,
    EfficientFormerForImageClassification,
)
from lucid.models.vision.efficientformer._weights import (
    EfficientFormerL1Weights,
    EfficientFormerL3Weights,
    EfficientFormerL7Weights,
)
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained

# Stochastic depth is 0.0 for every variant: both reference implementations
# leave ``drop_path_rate`` at its 0.0 default, snap-research's training entry
# point has no --drop-path flag at all, and the paper never mentions it.  The
# graded 0.0/0.1/0.2 schedule previously baked in here was attributed to the
# paper but appears in neither it nor either reference.  LayerScale init is
# 1e-5 across all variants.
_CFG_L1 = EfficientFormerConfig(
    depths=(3, 2, 6, 4),
    embed_dims=(48, 96, 224, 448),
    mlp_ratios=(4.0, 4.0, 4.0, 4.0),
    num_vit=1,
    drop_path_rate=0.0,
)

# Two documented divergences from Table 6, both resolved in favour of the
# reference because the paper's own Table 1 parameter counts confirm it:
#
#   * L3's last stage is 2 MB4D + 4 MB3D (``num_vit=4``); Table 6 prints
#     3 + 3.  ``num_vit=4`` builds 31,406,000 params, matching Table 1's
#     31.3M; the Table 6 split does not.
#   * L7's stage 3 is 18 blocks; Table 6 prints 8.  ``depths[2]=18``
#     builds 82,229,328, matching Table 1's 82.1M.
#
# So Table 6 is internally inconsistent with Table 1, and the reference
# implements the arrangement Table 1 was measured from.
_CFG_L3 = EfficientFormerConfig(
    depths=(4, 4, 12, 6), embed_dims=(64, 128, 320, 512), mlp_ratios=(4.0, 4.0, 4.0, 4.0), num_vit=4
)

_CFG_L7 = EfficientFormerConfig(
    depths=(6, 6, 18, 8), embed_dims=(96, 192, 384, 768), mlp_ratios=(4.0, 4.0, 4.0, 4.0), num_vit=8
)


# ── Backbones ─────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="efficientformer",
    model_type="efficientformer",
    model_class=EfficientFormer,
    default_config=_CFG_L1,
    params=12300000,
    summary="auto",
)
def efficientformer_l1(pretrained: bool = False, **overrides: object) -> EfficientFormer:
    r"""EfficientFormer-L1 backbone (Li et al., 2022).

    Builds the canonical *EfficientFormer-L1* configuration:
    ``depths=(3, 2, 6, 4)``, ``embed_dims=(48, 96, 224, 448)``,
    ``mlp_ratios=(4.0, 4.0, 4.0, 4.0)``, ``drop_path_rate=0.0``.
    **11.4M parameters** headless (12.3M with the classification
    heads) — the smallest, lowest-latency
    variant in the paper.

    Parameters
    ----------
    pretrained : bool, optional
        No checkpoint is published for ``efficientformer_l1``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``efficientformer_l1_cls`` is the checkpointed variant.
    **overrides : object
        Keyword overrides on top of the canonical L1 config.  Each
        override must match a field of :class:`EfficientFormerConfig`.

    Returns
    -------
    EfficientFormer
        An :class:`EfficientFormer` backbone returning a flat
        :math:`(B, 448)` feature.

    Notes
    -----
    EfficientFormer-L1 reaches **79.2% top-1 on ImageNet-1k** at
    MobileNetV2-class on-device latency (Li et al., 2022, Table 4).
    See `arXiv:2206.01191 <https://arxiv.org/abs/2206.01191>`_.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.efficientformer import efficientformer_l1
    >>> model = efficientformer_l1()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> feat = model.forward_features(x)
    >>> feat.shape
    (1, 448)
    """
    if pretrained:
        reject_unavailable_pretrained("efficientformer_l1", alternative="efficientformer_l1_cls")
    return EfficientFormer(apply_overrides(_CFG_L1, overrides))


@register_model(
    task="base",
    family="efficientformer",
    model_type="efficientformer",
    model_class=EfficientFormer,
    default_config=_CFG_L3,
    params=31300000,
    summary="auto",
)
def efficientformer_l3(pretrained: bool = False, **overrides: object) -> EfficientFormer:
    r"""EfficientFormer-L3 backbone (Li et al., 2022).

    Builds the canonical *EfficientFormer-L3* configuration:
    ``depths=(4, 4, 12, 6)``, ``embed_dims=(64, 128, 320, 512)``,
    ``drop_path_rate=0.1``.  **30.4M parameters** headless (31.4M
    with the classification heads).

    Parameters
    ----------
    pretrained : bool, optional
        No checkpoint is published for ``efficientformer_l3``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``efficientformer_l3_cls`` is the checkpointed variant.
    **overrides : object
        Keyword overrides on top of the canonical L3 config.

    Returns
    -------
    EfficientFormer
        An :class:`EfficientFormer` backbone returning a flat
        :math:`(B, 512)` feature.

    Notes
    -----
    EfficientFormer-L3 reaches **82.4% top-1 on ImageNet-1k** (Li
    et al., 2022, Table 4).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.efficientformer import efficientformer_l3
    >>> model = efficientformer_l3()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model.forward_features(x).shape
    (1, 512)
    """
    if pretrained:
        reject_unavailable_pretrained("efficientformer_l3", alternative="efficientformer_l3_cls")
    return EfficientFormer(apply_overrides(_CFG_L3, overrides))


@register_model(
    task="base",
    family="efficientformer",
    model_type="efficientformer",
    model_class=EfficientFormer,
    default_config=_CFG_L7,
    params=82100000,
    summary="auto",
)
def efficientformer_l7(pretrained: bool = False, **overrides: object) -> EfficientFormer:
    r"""EfficientFormer-L7 backbone (Li et al., 2022).

    Builds the canonical *EfficientFormer-L7* configuration:
    ``depths=(6, 6, 18, 8)``, ``embed_dims=(96, 192, 384, 768)``,
    ``drop_path_rate=0.2``.  **80.7M parameters** headless (82.2M
    with the classification heads) — the
    largest variant in the paper.

    Parameters
    ----------
    pretrained : bool, optional
        No checkpoint is published for ``efficientformer_l7``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``efficientformer_l7_cls`` is the checkpointed variant.
    **overrides : object
        Keyword overrides on top of the canonical L7 config.

    Returns
    -------
    EfficientFormer
        An :class:`EfficientFormer` backbone returning a flat
        :math:`(B, 768)` feature.

    Notes
    -----
    EfficientFormer-L7 reaches **83.3% top-1 on ImageNet-1k** (Li
    et al., 2022, Table 4) — the headline result of the paper.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.efficientformer import efficientformer_l7
    >>> model = efficientformer_l7()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model.forward_features(x).shape
    (1, 768)
    """
    if pretrained:
        reject_unavailable_pretrained("efficientformer_l7", alternative="efficientformer_l7_cls")
    return EfficientFormer(apply_overrides(_CFG_L7, overrides))


# ── Classifiers ───────────────────────────────────────────────────────────────


@register_model(
    task="image-classification",
    family="efficientformer",
    model_type="efficientformer",
    model_class=EfficientFormerForImageClassification,
    default_config=_CFG_L1,
    params=12300000,
    summary="auto",
)
def efficientformer_l1_cls(
    pretrained: bool | str = False,
    *,
    weights: EfficientFormerL1Weights | None = None,
    **overrides: object,
) -> EfficientFormerForImageClassification:
    r"""EfficientFormer-L1 image classifier (Li et al., 2022).

    Combines the :func:`efficientformer_l1` backbone with a final
    LayerNorm, a mean pool over tokens, and a distilled dual head
    (``head`` + ``head_dist``) averaged at inference.  Default output is
    ``num_classes=1000`` (ImageNet-1k).  ~12.3M parameters.

    Parameters
    ----------
    pretrained : bool or str, optional
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag
        (:attr:`EfficientFormerL1Weights.SNAP_DIST_IN1K`); a tag string
        → that specific checkpoint.  Mutually exclusive with ``weights``
        (which wins if both are given).
    weights : EfficientFormerL1Weights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides : object
        Keyword overrides on top of the canonical L1 config.

    Returns
    -------
    EfficientFormerForImageClassification
        Classifier returning :class:`ImageClassificationOutput` whose
        ``logits`` has shape ``(B, num_classes)``.

    Notes
    -----
    EfficientFormer-L1 reaches **79.2% top-1 on ImageNet-1k** at
    MobileNetV2-class latency (Li et al., 2022, Table 4).  Pretrained
    weights are converted from timm's
    ``efficientformer_l1.snap_dist_in1k`` and hosted under
    ``lucid-dl/efficientformer-l1``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.efficientformer import efficientformer_l1_cls
    >>> model = efficientformer_l1_cls(num_classes=1000)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model(x).logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(EfficientFormerL1Weights, pretrained, weights)
    model = EfficientFormerForImageClassification(apply_overrides(_CFG_L1, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="efficientformer_l1_cls")
    return model


@register_model(
    task="image-classification",
    family="efficientformer",
    model_type="efficientformer",
    model_class=EfficientFormerForImageClassification,
    default_config=_CFG_L3,
    params=31300000,
    summary="auto",
)
def efficientformer_l3_cls(
    pretrained: bool | str = False,
    *,
    weights: EfficientFormerL3Weights | None = None,
    **overrides: object,
) -> EfficientFormerForImageClassification:
    r"""EfficientFormer-L3 image classifier (Li et al., 2022).

    Combines the :func:`efficientformer_l3` backbone (``depths=
    (4, 4, 12, 6)``, ``embed_dims=(64, 128, 320, 512)``) with a final
    LayerNorm, mean pool, and a distilled dual head (``head`` +
    ``head_dist``) averaged at inference.  ~31.4M parameters.

    Parameters
    ----------
    pretrained : bool or str, optional
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag
        (:attr:`EfficientFormerL3Weights.SNAP_DIST_IN1K`); a tag string
        → that specific checkpoint.  Mutually exclusive with ``weights``
        (which wins if both are given).
    weights : EfficientFormerL3Weights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides : object
        Keyword overrides on top of the canonical L3 config.

    Returns
    -------
    EfficientFormerForImageClassification
        Classifier whose ``logits`` has shape ``(B, num_classes)``.

    Notes
    -----
    EfficientFormer-L3 reaches **82.4% top-1 on ImageNet-1k** (Li
    et al., 2022, Table 4).  Pretrained weights are converted from
    timm's ``efficientformer_l3.snap_dist_in1k`` and hosted under
    ``lucid-dl/efficientformer-l3``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.efficientformer import efficientformer_l3_cls
    >>> model = efficientformer_l3_cls(num_classes=1000)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model(x).logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(EfficientFormerL3Weights, pretrained, weights)
    model = EfficientFormerForImageClassification(apply_overrides(_CFG_L3, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="efficientformer_l3_cls")
    return model


@register_model(
    task="image-classification",
    family="efficientformer",
    model_type="efficientformer",
    model_class=EfficientFormerForImageClassification,
    default_config=_CFG_L7,
    params=82100000,
    summary="auto",
)
def efficientformer_l7_cls(
    pretrained: bool | str = False,
    *,
    weights: EfficientFormerL7Weights | None = None,
    **overrides: object,
) -> EfficientFormerForImageClassification:
    r"""EfficientFormer-L7 image classifier (Li et al., 2022).

    Combines the :func:`efficientformer_l7` backbone (``depths=
    (6, 6, 18, 8)``, ``embed_dims=(96, 192, 384, 768)``) with a final
    LayerNorm, mean pool, and a distilled dual head (``head`` +
    ``head_dist``) averaged at inference.  ~82.2M parameters — the
    largest EfficientFormer variant.

    Parameters
    ----------
    pretrained : bool or str, optional
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag
        (:attr:`EfficientFormerL7Weights.SNAP_DIST_IN1K`); a tag string
        → that specific checkpoint.  Mutually exclusive with ``weights``
        (which wins if both are given).
    weights : EfficientFormerL7Weights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides : object
        Keyword overrides on top of the canonical L7 config.

    Returns
    -------
    EfficientFormerForImageClassification
        Classifier whose ``logits`` has shape ``(B, num_classes)``.

    Notes
    -----
    EfficientFormer-L7 reaches **83.3% top-1 on ImageNet-1k** (Li
    et al., 2022, Table 4) — the headline result of the paper.
    Pretrained weights are converted from timm's
    ``efficientformer_l7.snap_dist_in1k`` and hosted under
    ``lucid-dl/efficientformer-l7``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.efficientformer import efficientformer_l7_cls
    >>> model = efficientformer_l7_cls(num_classes=1000)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model(x).logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(EfficientFormerL7Weights, pretrained, weights)
    model = EfficientFormerForImageClassification(apply_overrides(_CFG_L7, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="efficientformer_l7_cls")
    return model
