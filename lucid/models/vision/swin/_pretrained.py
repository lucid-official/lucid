"""Registry factories for Swin Transformer variants."""

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models.vision.swin._config import SwinConfig
from lucid.models.vision.swin._model import SwinTransformer, SwinTransformerForImageClassification
from lucid.models.vision.swin._weights import (
    SwinBaseWeights,
    SwinLargeWeights,
    SwinSmallWeights,
    SwinTinyWeights,
)
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained

_CFG_T = SwinConfig(embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24))
_CFG_S = SwinConfig(embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24))
_CFG_B = SwinConfig(embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))
_CFG_L = SwinConfig(embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))


# ── Backbones ─────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="swin",
    model_type="swin",
    model_class=SwinTransformer,
    default_config=_CFG_T,
)
def swin_tiny(pretrained: bool = False, **overrides: object) -> SwinTransformer:
    r"""Swin-Tiny backbone (Liu et al., 2021).

    Builds the canonical *Swin-T* configuration: ``embed_dim=96``,
    ``depths=(2, 2, 6, 2)``, ``num_heads=(3, 6, 12, 24)``,
    ``window_size=7``.  With the default 224x224 input the trunk
    produces a :math:`(B, 768)` feature after the final stage's
    :math:`7 \times 7` map is globally pooled.  Approximately
    **28M parameters**.

    Parameters
    ----------
    pretrained : bool, optional
        No checkpoint is published for ``swin_tiny``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``swin_tiny_cls`` is the checkpointed variant.
    **overrides : object
        Keyword overrides applied on top of the canonical Swin-T config.
        Common examples: ``image_size=384``, ``in_channels=1``,
        ``window_size=12``.  Each override must match a field of
        :class:`SwinConfig`.

    Returns
    -------
    SwinTransformer
        A :class:`SwinTransformer` backbone returning a flat
        :math:`(B, 768)` feature.

    Notes
    -----
    Swin-T reaches **81.3% top-1 / 95.5% top-5 on ImageNet-1k**
    (Liu et al., 2021, Table 1).  See
    `arXiv:2103.14030 <https://arxiv.org/abs/2103.14030>`_.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.swin import swin_tiny
    >>> model = swin_tiny()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> feat = model.forward_features(x)
    >>> feat.shape
    (1, 768)
    """
    if pretrained:
        reject_unavailable_pretrained("swin_tiny", alternative="swin_tiny_cls")
    return SwinTransformer(apply_overrides(_CFG_T, overrides))


@register_model(
    task="base",
    family="swin",
    model_type="swin",
    model_class=SwinTransformer,
    default_config=_CFG_S,
)
def swin_small(pretrained: bool = False, **overrides: object) -> SwinTransformer:
    r"""Swin-Small backbone (Liu et al., 2021).

    Builds the canonical *Swin-S* configuration: ``embed_dim=96``,
    ``depths=(2, 2, 18, 2)``, ``num_heads=(3, 6, 12, 24)``.  Same width
    as Swin-T but with 18 blocks in stage 3 instead of 6, roughly
    doubling the compute.  Approximately **50M parameters**.

    Parameters
    ----------
    pretrained : bool, optional
        No checkpoint is published for ``swin_small``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``swin_tiny_cls`` is the checkpointed variant.
    **overrides : object
        Keyword overrides on top of the canonical Swin-S config.

    Returns
    -------
    SwinTransformer
        A :class:`SwinTransformer` backbone returning a flat
        :math:`(B, 768)` feature.

    Notes
    -----
    Swin-S reaches **83.0% top-1 on ImageNet-1k** (Liu et al., 2021,
    Table 1).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.swin import swin_small
    >>> model = swin_small()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model.forward_features(x).shape
    (1, 768)
    """
    if pretrained:
        reject_unavailable_pretrained("swin_small", alternative="swin_tiny_cls")
    return SwinTransformer(apply_overrides(_CFG_S, overrides))


@register_model(
    task="base",
    family="swin",
    model_type="swin",
    model_class=SwinTransformer,
    default_config=_CFG_B,
)
def swin_base(pretrained: bool = False, **overrides: object) -> SwinTransformer:
    r"""Swin-Base backbone (Liu et al., 2021).

    Builds the canonical *Swin-B* configuration: ``embed_dim=128``,
    ``depths=(2, 2, 18, 2)``, ``num_heads=(4, 8, 16, 32)``.  Same depth
    as Swin-S but ~33% wider per stage; final feature dimension is
    :math:`8 \cdot 128 = 1024`.  Approximately **88M parameters**.

    Parameters
    ----------
    pretrained : bool, optional
        No checkpoint is published for ``swin_base``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``swin_tiny_cls`` is the checkpointed variant.
    **overrides : object
        Keyword overrides on top of the canonical Swin-B config.

    Returns
    -------
    SwinTransformer
        A :class:`SwinTransformer` backbone returning a flat
        :math:`(B, 1024)` feature.

    Notes
    -----
    Swin-B reaches **83.5% top-1 on ImageNet-1k** (224x224) and
    **86.4% top-1** when fine-tuned at 384x384 after ImageNet-22k
    pretraining (Liu et al., 2021, Tables 1 and 2).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.swin import swin_base
    >>> model = swin_base()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model.forward_features(x).shape
    (1, 1024)
    """
    if pretrained:
        reject_unavailable_pretrained("swin_base", alternative="swin_tiny_cls")
    return SwinTransformer(apply_overrides(_CFG_B, overrides))


@register_model(
    task="base",
    family="swin",
    model_type="swin",
    model_class=SwinTransformer,
    default_config=_CFG_L,
)
def swin_large(pretrained: bool = False, **overrides: object) -> SwinTransformer:
    r"""Swin-Large backbone (Liu et al., 2021).

    Builds the canonical *Swin-L* configuration: ``embed_dim=192``,
    ``depths=(2, 2, 18, 2)``, ``num_heads=(6, 12, 24, 48)``.  Final
    feature dimension is :math:`8 \cdot 192 = 1536`.  Approximately
    **197M parameters**.  Pretrained variants are typically released
    only with ImageNet-22k pretraining due to the model's capacity.

    Parameters
    ----------
    pretrained : bool, optional
        No checkpoint is published for ``swin_large``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``swin_tiny_cls`` is the checkpointed variant.
    **overrides : object
        Keyword overrides on top of the canonical Swin-L config.

    Returns
    -------
    SwinTransformer
        A :class:`SwinTransformer` backbone returning a flat
        :math:`(B, 1536)` feature.

    Notes
    -----
    Swin-L reaches **87.3% top-1 on ImageNet-1k** (384x384) after
    ImageNet-22k pretraining (Liu et al., 2021 — the classification
    results; Table 2 is COCO detection).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.swin import swin_large
    >>> model = swin_large()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model.forward_features(x).shape
    (1, 1536)
    """
    if pretrained:
        reject_unavailable_pretrained("swin_large", alternative="swin_tiny_cls")
    return SwinTransformer(apply_overrides(_CFG_L, overrides))


# ── Classifiers ───────────────────────────────────────────────────────────────


@register_model(
    task="image-classification",
    family="swin",
    model_type="swin",
    model_class=SwinTransformerForImageClassification,
    default_config=_CFG_T,
)
def swin_tiny_cls(
    pretrained: bool | str = False, *, weights: SwinTinyWeights | None = None, **overrides: object
) -> SwinTransformerForImageClassification:
    r"""Swin-Tiny image classifier (Liu et al., 2021).

    Combines the :func:`swin_tiny` backbone with a global average pool
    + single :class:`nn.Linear` classification head.  Default output is
    ``num_classes=1000`` (ImageNet-1k).  Approximately **28M parameters**
    for the 1000-class head.

    Parameters
    ----------
    pretrained : bool or str, optional
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`SwinTinyWeights.IMAGENET1K_V1`);
        a tag string → that specific checkpoint.  Mutually exclusive
        with ``weights`` (which wins if both are given).
    weights : SwinTinyWeights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides : object
        Keyword overrides on top of the canonical Swin-T config.
        Commonly used: ``num_classes`` to switch label space,
        ``image_size`` to fine-tune at a different resolution.

    Returns
    -------
    SwinTransformerForImageClassification
        Classifier returning :class:`ImageClassificationOutput` whose
        ``logits`` has shape ``(B, num_classes)``.

    Notes
    -----
    Swin-T reaches **81.3% top-1 on ImageNet-1k** (Liu et al., 2021).
    Pretrained weights are converted from reference_vision's
    ``Swin_T_Weights.IMAGENET1K_V1`` and hosted under
    ``lucid-dl/swin-tiny``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.swin import swin_tiny_cls
    >>> model = swin_tiny_cls(num_classes=1000)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(SwinTinyWeights, pretrained, weights)
    model = SwinTransformerForImageClassification(apply_overrides(_CFG_T, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="swin_tiny_cls")
    return model


@register_model(
    task="image-classification",
    family="swin",
    model_type="swin",
    model_class=SwinTransformerForImageClassification,
    default_config=_CFG_S,
)
def swin_small_cls(
    pretrained: bool | str = False, *, weights: SwinSmallWeights | None = None, **overrides: object
) -> SwinTransformerForImageClassification:
    r"""Swin-Small image classifier (Liu et al., 2021).

    Combines the :func:`swin_small` backbone (``embed_dim=96``,
    ``depths=(2, 2, 18, 2)``) with a global average pool plus a linear
    classification head.  ~50M parameters.

    Parameters
    ----------
    pretrained : bool or str, optional
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`SwinSmallWeights.IMAGENET1K_V1`);
        a tag string → that specific checkpoint.  Mutually exclusive
        with ``weights`` (which wins if both are given).
    weights : SwinSmallWeights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides : object
        Keyword overrides on top of the canonical Swin-S config.

    Returns
    -------
    SwinTransformerForImageClassification
        Classifier whose ``logits`` has shape ``(B, num_classes)``.

    Notes
    -----
    Swin-S reaches **83.0% top-1 on ImageNet-1k** (Liu et al., 2021,
    Table 1).  Pretrained weights are converted from reference_vision's
    ``Swin_S_Weights.IMAGENET1K_V1`` and hosted under
    ``lucid-dl/swin-small``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.swin import swin_small_cls
    >>> model = swin_small_cls(num_classes=1000)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model(x).logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(SwinSmallWeights, pretrained, weights)
    model = SwinTransformerForImageClassification(apply_overrides(_CFG_S, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="swin_small_cls")
    return model


@register_model(
    task="image-classification",
    family="swin",
    model_type="swin",
    model_class=SwinTransformerForImageClassification,
    default_config=_CFG_B,
)
def swin_base_cls(
    pretrained: bool | str = False, *, weights: SwinBaseWeights | None = None, **overrides: object
) -> SwinTransformerForImageClassification:
    r"""Swin-Base image classifier (Liu et al., 2021).

    Combines the :func:`swin_base` backbone (``embed_dim=128``,
    ``depths=(2, 2, 18, 2)``) with a global average pool plus a linear
    classification head.  Final feature width is :math:`8 \cdot 128 =
    1024`.  ~88M parameters.

    Parameters
    ----------
    pretrained : bool or str, optional
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`SwinBaseWeights.IMAGENET1K_V1`);
        a tag string → that specific checkpoint.  Mutually exclusive
        with ``weights`` (which wins if both are given).
    weights : SwinBaseWeights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides : object
        Keyword overrides on top of the canonical Swin-B config.

    Returns
    -------
    SwinTransformerForImageClassification
        Classifier whose ``logits`` has shape ``(B, num_classes)``.

    Notes
    -----
    Swin-B reaches **83.5% top-1 on ImageNet-1k** (224x224) and
    **86.4% top-1** at 384x384 after ImageNet-22k pretraining (Liu
    et al., 2021, Tables 1 and 2).  The shipped checkpoint is converted
    from reference_vision's ``Swin_B_Weights.IMAGENET1K_V1`` and hosted under
    ``lucid-dl/swin-base``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.swin import swin_base_cls
    >>> model = swin_base_cls(num_classes=1000)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model(x).logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(SwinBaseWeights, pretrained, weights)
    model = SwinTransformerForImageClassification(apply_overrides(_CFG_B, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="swin_base_cls")
    return model


@register_model(
    task="image-classification",
    family="swin",
    model_type="swin",
    model_class=SwinTransformerForImageClassification,
    default_config=_CFG_L,
)
def swin_large_cls(
    pretrained: bool | str = False, *, weights: SwinLargeWeights | None = None, **overrides: object
) -> SwinTransformerForImageClassification:
    r"""Swin-Large image classifier (Liu et al., 2021).

    Combines the :func:`swin_large` backbone (``embed_dim=192``,
    ``depths=(2, 2, 18, 2)``) with a global average pool plus a linear
    classification head.  Final feature width is :math:`8 \cdot 192 =
    1536`.  ~197M parameters.

    Parameters
    ----------
    pretrained : bool or str, optional
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag
        (:attr:`SwinLargeWeights.MS_IN22K_FT_IN1K`); a tag string →
        that specific checkpoint.  Mutually exclusive with ``weights``
        (which wins if both are given).
    weights : SwinLargeWeights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides : object
        Keyword overrides on top of the canonical Swin-L config.

    Returns
    -------
    SwinTransformerForImageClassification
        Classifier whose ``logits`` has shape ``(B, num_classes)``.

    Notes
    -----
    Swin-L reaches **87.3% top-1 on ImageNet-1k** at 384x384
    fine-tune resolution after ImageNet-22k pretraining (Liu
    et al., 2021 — the classification results; Table 2 is COCO
    detection).  The shipped checkpoint is the 224x224
    ImageNet-22k → ImageNet-1k finetune (86.3% top-1), converted from
    timm's ``swin_large_patch4_window7_224.ms_in22k_ft_in1k`` and
    hosted under ``lucid-dl/swin-large``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.swin import swin_large_cls
    >>> model = swin_large_cls(num_classes=1000)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model(x).logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(SwinLargeWeights, pretrained, weights)
    model = SwinTransformerForImageClassification(apply_overrides(_CFG_L, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="swin_large_cls")
    return model
