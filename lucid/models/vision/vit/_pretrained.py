"""Registry factories for ViT variants."""

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models.vision.vit._config import ViTConfig
from lucid.models.vision.vit._model import ViT, ViTForImageClassification
from lucid.models.vision.vit._weights import (
    ViTBase16Weights,
    ViTBase32Weights,
    ViTLarge16Weights,
    ViTLarge32Weights,
)
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained

_CFG_B16 = ViTConfig(patch_size=16, dim=768, depth=12, num_heads=12)
_CFG_B32 = ViTConfig(patch_size=32, dim=768, depth=12, num_heads=12)
_CFG_L16 = ViTConfig(patch_size=16, dim=1024, depth=24, num_heads=16)
_CFG_L32 = ViTConfig(patch_size=32, dim=1024, depth=24, num_heads=16)
_CFG_H14 = ViTConfig(patch_size=14, dim=1280, depth=32, num_heads=16)


# ── Backbones ─────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="vit",
    model_type="vit",
    model_class=ViT,
    default_config=_CFG_B16,
)
def vit_base_16(pretrained: bool = False, **overrides: object) -> ViT:
    r"""ViT-Base/16 backbone (Dosovitskiy et al., 2020).

    Builds the canonical *ViT-Base* configuration with patch size 16:
    ``dim=768``, ``depth=12``, ``num_heads=12``, ``mlp_ratio=4.0``.  With
    the default 224x224 input this yields :math:`(224/16)^2 = 196` patch
    tokens plus one CLS token, for an input sequence length of ``197`` to
    the transformer encoder.  Approximately **86M parameters**.

    Parameters
    ----------
    pretrained : bool, optional
        No checkpoint is published for ``vit_base_16``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``vit_base_16_cls`` is the checkpointed variant.
    **overrides : object
        Keyword overrides applied on top of the canonical ViT-Base/16
        config — e.g. ``image_size=384``, ``in_channels=1``, or
        ``dropout=0.1``.  Each override must match a field of
        :class:`ViTConfig`.

    Returns
    -------
    ViT
        A :class:`ViT` backbone instance returning a flat ``(B, 768)`` CLS
        feature.

    Notes
    -----
    ViT-Base/16 was pretrained on JFT-300M / ImageNet-21k and reported
    **84.15% top-1 / 97.20% top-5 on ImageNet-1k** after fine-tuning at
    384x384 in Dosovitskiy et al. (2020, Table 5).  See
    `arXiv:2010.11929 <https://arxiv.org/abs/2010.11929>`_.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.vit import vit_base_16
    >>> model = vit_base_16()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> feat = model.forward_features(x)
    >>> feat.shape
    (1, 768)
    """
    if pretrained:
        reject_unavailable_pretrained("vit_base_16", alternative="vit_base_16_cls")
    return ViT(apply_overrides(_CFG_B16, overrides))


@register_model(
    task="base",
    family="vit",
    model_type="vit",
    model_class=ViT,
    default_config=_CFG_B32,
)
def vit_base_32(pretrained: bool = False, **overrides: object) -> ViT:
    r"""ViT-Base/32 backbone (Dosovitskiy et al., 2020).

    Same ViT-Base trunk as :func:`vit_base_16` (``dim=768``, ``depth=12``,
    ``num_heads=12``) but with patch size 32, which reduces the sequence
    length to :math:`(224/32)^2 = 49` patch tokens (+1 CLS = 50).  The
    larger patches trade fine-grained spatial detail for ~4x cheaper
    self-attention and ~88M parameters in total.

    Parameters
    ----------
    pretrained : bool, optional
        No checkpoint is published for ``vit_base_32``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``vit_base_16_cls`` is the checkpointed variant.
    **overrides : object
        Keyword overrides on top of the canonical ViT-Base/32 config.

    Returns
    -------
    ViT
        A :class:`ViT` backbone returning a flat ``(B, 768)`` CLS feature.

    Notes
    -----
    Reference: Dosovitskiy et al. (2020), Table 1.  The ``/32`` variant
    is most useful when latency / FLOPs dominate over absolute accuracy.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.vit import vit_base_32
    >>> model = vit_base_32()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model.forward_features(x).shape
    (1, 768)
    """
    if pretrained:
        reject_unavailable_pretrained("vit_base_32", alternative="vit_base_16_cls")
    return ViT(apply_overrides(_CFG_B32, overrides))


@register_model(
    task="base",
    family="vit",
    model_type="vit",
    model_class=ViT,
    default_config=_CFG_L16,
)
def vit_large_16(pretrained: bool = False, **overrides: object) -> ViT:
    r"""ViT-Large/16 backbone (Dosovitskiy et al., 2020).

    The canonical *ViT-Large* configuration with patch size 16:
    ``dim=1024``, ``depth=24``, ``num_heads=16``, ``mlp_ratio=4.0``.  With
    a 224x224 input it produces 197 tokens (196 patches + CLS) feeding 24
    pre-norm transformer blocks of width 1024.  Approximately
    **307M parameters**.

    Parameters
    ----------
    pretrained : bool, optional
        No checkpoint is published for ``vit_large_16``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``vit_base_16_cls`` is the checkpointed variant.
    **overrides : object
        Keyword overrides on top of the canonical ViT-Large/16 config.

    Returns
    -------
    ViT
        A :class:`ViT` backbone returning a flat ``(B, 1024)`` CLS feature.

    Notes
    -----
    ViT-Large/16 reaches **85.30% top-1 on ImageNet-1k** when fine-tuned
    at 384x384 after ImageNet-21k pretraining (Dosovitskiy et al., 2020,
    Table 5).  Best accuracy/cost trade-off among the Large variants.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.vit import vit_large_16
    >>> model = vit_large_16()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model.forward_features(x).shape
    (1, 1024)
    """
    if pretrained:
        reject_unavailable_pretrained("vit_large_16", alternative="vit_base_16_cls")
    return ViT(apply_overrides(_CFG_L16, overrides))


@register_model(
    task="base",
    family="vit",
    model_type="vit",
    model_class=ViT,
    default_config=_CFG_L32,
)
def vit_large_32(pretrained: bool = False, **overrides: object) -> ViT:
    r"""ViT-Large/32 backbone (Dosovitskiy et al., 2020).

    Same ViT-Large trunk as :func:`vit_large_16` (``dim=1024``,
    ``depth=24``, ``num_heads=16``) but with patch size 32, producing only
    :math:`(224/32)^2 = 49` patch tokens (+1 CLS).  Approximately
    **307M parameters** with substantially reduced attention cost.

    Parameters
    ----------
    pretrained : bool, optional
        No checkpoint is published for ``vit_large_32``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``vit_base_16_cls`` is the checkpointed variant.
    **overrides : object
        Keyword overrides on top of the canonical ViT-Large/32 config.

    Returns
    -------
    ViT
        A :class:`ViT` backbone returning a flat ``(B, 1024)`` CLS feature.

    Notes
    -----
    Reference: Dosovitskiy et al. (2020), Table 1.  Useful when Large
    capacity is desired but the ``/16`` token count is too expensive.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.vit import vit_large_32
    >>> model = vit_large_32()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model.forward_features(x).shape
    (1, 1024)
    """
    if pretrained:
        reject_unavailable_pretrained("vit_large_32", alternative="vit_base_16_cls")
    return ViT(apply_overrides(_CFG_L32, overrides))


@register_model(
    task="base",
    family="vit",
    model_type="vit",
    model_class=ViT,
    default_config=_CFG_H14,
)
def vit_huge_14(pretrained: bool = False, **overrides: object) -> ViT:
    r"""ViT-Huge/14 backbone (Dosovitskiy et al., 2020).

    The canonical *ViT-Huge* configuration with patch size 14:
    ``dim=1280``, ``depth=32``, ``num_heads=16``, ``mlp_ratio=4.0``.  With
    a 224x224 input it produces :math:`(224/14)^2 = 256` patch tokens
    (+1 CLS = 257 total) feeding 32 transformer blocks of width 1280.
    Approximately **632M parameters** — the largest variant in the
    paper.

    Parameters
    ----------
    pretrained : bool, optional
        No checkpoint is published for ``vit_huge_14``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``vit_base_16_cls`` is the checkpointed variant.
    **overrides : object
        Keyword overrides on top of the canonical ViT-Huge/14 config.

    Returns
    -------
    ViT
        A :class:`ViT` backbone returning a flat ``(B, 1280)`` CLS feature.

    Notes
    -----
    ViT-Huge/14 achieves **88.55% top-1 on ImageNet-1k** when fine-tuned
    at 518x518 after JFT-300M pretraining (Dosovitskiy et al., 2020,
    Table 2) — the state-of-the-art result reported in the paper at
    release.  Training from scratch on ImageNet-1k alone is impractical
    at this scale; pretrained weights are strongly recommended.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.vit import vit_huge_14
    >>> model = vit_huge_14()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model.forward_features(x).shape
    (1, 1280)
    """
    if pretrained:
        reject_unavailable_pretrained("vit_huge_14", alternative="vit_base_16_cls")
    return ViT(apply_overrides(_CFG_H14, overrides))


# ── Classifiers ───────────────────────────────────────────────────────────────


@register_model(
    task="image-classification",
    family="vit",
    model_type="vit",
    model_class=ViTForImageClassification,
    default_config=_CFG_B16,
)
def vit_base_16_cls(
    pretrained: bool | str = False,
    *,
    weights: ViTBase16Weights | None = None,
    **overrides: object,
) -> ViTForImageClassification:
    r"""ViT-Base/16 image classifier (Dosovitskiy et al., 2020).

    Combines the :func:`vit_base_16` backbone with a single
    :class:`nn.Linear` classification head on the CLS token.  Default
    output is ``num_classes=1000`` (ImageNet-1k); override via
    ``num_classes=...``.  Approximately **86M parameters** for the
    1000-class head.

    Parameters
    ----------
    pretrained : bool or str, optional
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`ViTBase16Weights.IMAGENET1K_V1`); a
        tag string → that specific checkpoint.  Mutually exclusive with
        ``weights`` (which wins if both are given).
    weights : ViTBase16Weights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides : object
        Keyword overrides on top of the canonical ViT-Base/16 config.
        Commonly used: ``num_classes`` to switch label space,
        ``image_size`` to fine-tune at higher resolution.

    Returns
    -------
    ViTForImageClassification
        Classifier returning :class:`ImageClassificationOutput` whose
        ``logits`` has shape ``(B, num_classes)``.

    Notes
    -----
    Standard ImageNet recipe: backbone returns the CLS token after the
    final LayerNorm, head applies a single affine map to produce class
    logits.  Pretrained weights are converted from reference_vision's
    ``ViT_B_16_Weights.IMAGENET1K_V1`` (81.072% top-1) and hosted under
    ``lucid-dl/vit-base-16``.  See
    `arXiv:2010.11929 <https://arxiv.org/abs/2010.11929>`_.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.vit import vit_base_16_cls
    >>> model = vit_base_16_cls(num_classes=1000)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(ViTBase16Weights, pretrained, weights)
    model = ViTForImageClassification(apply_overrides(_CFG_B16, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="vit_base_16_cls")
    return model


@register_model(
    task="image-classification",
    family="vit",
    model_type="vit",
    model_class=ViTForImageClassification,
    default_config=_CFG_B32,
)
def vit_base_32_cls(
    pretrained: bool | str = False,
    *,
    weights: ViTBase32Weights | None = None,
    **overrides: object,
) -> ViTForImageClassification:
    r"""ViT-Base/32 image classifier (Dosovitskiy et al., 2020).

    Combines the :func:`vit_base_32` backbone (``dim=768``, ``depth=12``,
    ``num_heads=12``, patch size 32) with a linear classification head on
    the CLS token.  The ``/32`` patching yields a 49-token sequence,
    making this the cheapest ViT-Base classifier.

    Parameters
    ----------
    pretrained : bool or str, optional
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`ViTBase32Weights.IMAGENET1K_V1`); a
        tag string → that specific checkpoint.  Mutually exclusive with
        ``weights`` (which wins if both are given).
    weights : ViTBase32Weights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides : object
        Keyword overrides on top of the canonical ViT-Base/32 config.

    Returns
    -------
    ViTForImageClassification
        Classifier whose ``logits`` has shape ``(B, num_classes)``.

    Notes
    -----
    Reference: Dosovitskiy et al. (2020), Table 1.  Pretrained weights are
    converted from reference_vision's ``ViT_B_32_Weights.IMAGENET1K_V1``
    (75.912% top-1) and hosted under ``lucid-dl/vit-base-32``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.vit import vit_base_32_cls
    >>> model = vit_base_32_cls(num_classes=10)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model(x).logits.shape
    (1, 10)
    """
    entry = weights_mod.resolve_weights(ViTBase32Weights, pretrained, weights)
    model = ViTForImageClassification(apply_overrides(_CFG_B32, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="vit_base_32_cls")
    return model


@register_model(
    task="image-classification",
    family="vit",
    model_type="vit",
    model_class=ViTForImageClassification,
    default_config=_CFG_L16,
)
def vit_large_16_cls(
    pretrained: bool | str = False,
    *,
    weights: ViTLarge16Weights | None = None,
    **overrides: object,
) -> ViTForImageClassification:
    r"""ViT-Large/16 image classifier (Dosovitskiy et al., 2020).

    Combines the :func:`vit_large_16` backbone (``dim=1024``,
    ``depth=24``, ``num_heads=16``, patch size 16) with a linear
    classification head on the CLS token.  Approximately
    **307M parameters**.

    Parameters
    ----------
    pretrained : bool or str, optional
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`ViTLarge16Weights.IMAGENET1K_V1`); a
        tag string → that specific checkpoint.  Mutually exclusive with
        ``weights`` (which wins if both are given).
    weights : ViTLarge16Weights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides : object
        Keyword overrides on top of the canonical ViT-Large/16 config.

    Returns
    -------
    ViTForImageClassification
        Classifier whose ``logits`` has shape ``(B, num_classes)``.

    Notes
    -----
    Reference: Dosovitskiy et al. (2020), Table 5.  Pretrained weights are
    converted from reference_vision's ``ViT_L_16_Weights.IMAGENET1K_V1``
    (79.662% top-1, 242-pixel resize) and hosted under
    ``lucid-dl/vit-large-16``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.vit import vit_large_16_cls
    >>> model = vit_large_16_cls(num_classes=1000)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model(x).logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(ViTLarge16Weights, pretrained, weights)
    model = ViTForImageClassification(apply_overrides(_CFG_L16, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="vit_large_16_cls")
    return model


@register_model(
    task="image-classification",
    family="vit",
    model_type="vit",
    model_class=ViTForImageClassification,
    default_config=_CFG_L32,
)
def vit_large_32_cls(
    pretrained: bool | str = False,
    *,
    weights: ViTLarge32Weights | None = None,
    **overrides: object,
) -> ViTForImageClassification:
    r"""ViT-Large/32 image classifier (Dosovitskiy et al., 2020).

    Combines the :func:`vit_large_32` backbone (``dim=1024``,
    ``depth=24``, ``num_heads=16``, patch size 32) with a linear
    classification head on the CLS token.  The ``/32`` patching gives a
    49-token sequence and substantially lowers self-attention cost.

    Parameters
    ----------
    pretrained : bool or str, optional
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`ViTLarge32Weights.IMAGENET1K_V1`); a
        tag string → that specific checkpoint.  Mutually exclusive with
        ``weights`` (which wins if both are given).
    weights : ViTLarge32Weights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides : object
        Keyword overrides on top of the canonical ViT-Large/32 config.

    Returns
    -------
    ViTForImageClassification
        Classifier whose ``logits`` has shape ``(B, num_classes)``.

    Notes
    -----
    Reference: Dosovitskiy et al. (2020), Table 1.  Pretrained weights are
    converted from reference_vision's ``ViT_L_32_Weights.IMAGENET1K_V1``
    (76.972% top-1) and hosted under ``lucid-dl/vit-large-32``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.vit import vit_large_32_cls
    >>> model = vit_large_32_cls(num_classes=1000)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model(x).logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(ViTLarge32Weights, pretrained, weights)
    model = ViTForImageClassification(apply_overrides(_CFG_L32, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="vit_large_32_cls")
    return model


@register_model(
    task="image-classification",
    family="vit",
    model_type="vit",
    model_class=ViTForImageClassification,
    default_config=_CFG_H14,
)
def vit_huge_14_cls(pretrained: bool = False, **overrides: object) -> ViTForImageClassification:
    r"""ViT-Huge/14 image classifier (Dosovitskiy et al., 2020).

    Combines the :func:`vit_huge_14` backbone (``dim=1280``, ``depth=32``,
    ``num_heads=16``, patch size 14) with a linear classification head on
    the CLS token.  Approximately **632M parameters** — the largest ViT
    variant in the paper.

    Parameters
    ----------
    pretrained : bool, optional
        No checkpoint is published for ``vit_huge_14_cls``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``vit_base_16_cls`` is the checkpointed variant.
    **overrides : object
        Keyword overrides on top of the canonical ViT-Huge/14 config.

    Returns
    -------
    ViTForImageClassification
        Classifier whose ``logits`` has shape ``(B, num_classes)``.

    Notes
    -----
    Achieves **88.55% top-1 on ImageNet-1k** at 518x518 fine-tune
    resolution after JFT-300M pretraining (Dosovitskiy et al., 2020,
    Table 2) — the headline result of the paper.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.vit import vit_huge_14_cls
    >>> model = vit_huge_14_cls(num_classes=1000)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model(x).logits.shape
    (1, 1000)
    """
    if pretrained:
        reject_unavailable_pretrained("vit_huge_14_cls", alternative="vit_base_16_cls")
    return ViTForImageClassification(apply_overrides(_CFG_H14, overrides))
