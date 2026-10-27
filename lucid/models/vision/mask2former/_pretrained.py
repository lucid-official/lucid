"""Registry factories for Mask2Former Swin variants."""

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models.vision.mask2former._config import Mask2FormerConfig
from lucid.models.vision.mask2former._model import Mask2FormerForSemanticSegmentation
from lucid.models.vision.mask2former._weights import (
    Mask2FormerSwinBaseWeights,
    Mask2FormerSwinLargeWeights,
    Mask2FormerSwinSmallWeights,
    Mask2FormerSwinTinyWeights,
)
from lucid.models._utils._common import apply_overrides


def _swin_cfg(
    embed_dim: int,
    depths: tuple[int, int, int, int],
    num_heads: tuple[int, int, int, int],
    window_size: int = 7,
) -> Mask2FormerConfig:
    return Mask2FormerConfig(
        num_classes=150,
        in_channels=3,
        swin_embed_dim=embed_dim,
        swin_depths=depths,
        swin_num_heads=num_heads,
        swin_window_size=window_size,
        d_model=256,
        mask_feature_size=256,
        n_head=8,
        num_encoder_layers=6,
        encoder_feedforward_dim=1024,
        num_decoder_layers=10,
        dim_feedforward=2048,
        dropout=0.0,
        num_queries=100,
        num_feature_levels=3,
    )


_CFG_SWIN_TINY = _swin_cfg(96, (2, 2, 6, 2), (3, 6, 12, 24))
_CFG_SWIN_SMALL = _swin_cfg(96, (2, 2, 18, 2), (3, 6, 12, 24))
# base/large use the 384-pretrained Swin with window 12 (tiny/small use 7).
_CFG_SWIN_BASE = _swin_cfg(128, (2, 2, 18, 2), (4, 8, 16, 32), window_size=12)
_CFG_SWIN_LARGE = _swin_cfg(192, (2, 2, 18, 2), (6, 12, 24, 48), window_size=12)


@register_model(
    task="semantic-segmentation",
    family="mask2former",
    model_type="mask2former",
    model_class=Mask2FormerForSemanticSegmentation,
    default_config=_CFG_SWIN_TINY,
)
def mask2former_swin_tiny(
    pretrained: bool | str = False,
    *,
    weights: Mask2FormerSwinTinyWeights | None = None,
    **overrides: object,
) -> Mask2FormerForSemanticSegmentation:
    r"""Mask2Former with Swin-Tiny backbone (Cheng et al., CVPR 2022).

    Builds a :class:`Mask2FormerForSemanticSegmentation` with the Swin-Tiny
    backbone (``embed_dim = 96``, ``depths = (2, 2, 6, 2)``,
    ``num_heads = (3, 6, 12, 24)``, ``window_size = 7``), a 6-layer
    multi-scale deformable pixel decoder, and a 9-layer masked-attention
    transformer decoder with 100 queries.  Default targets ADE20K
    (150 classes); reaches ADE20K validation mIoU of 47.7%.

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`Mask2FormerSwinTinyWeights.ADE20K`);
        a tag string → that specific checkpoint.  Mutually exclusive with
        ``weights`` (which wins if both are given).
    weights : Mask2FormerSwinTinyWeights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`Mask2FormerConfig`.

    Returns
    -------
    Mask2FormerForSemanticSegmentation
        Segmentation model with the Mask2Former-Swin-T configuration applied
        (or with ``overrides`` merged on top of it).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mask2former import mask2former_swin_tiny
    >>> model = mask2former_swin_tiny()
    >>> x = lucid.randn(1, 3, 384, 384)
    >>> out = model(x)
    >>> out.logits.shape[1]
    150
    """
    entry = weights_mod.resolve_weights(Mask2FormerSwinTinyWeights, pretrained, weights)
    model = Mask2FormerForSemanticSegmentation(apply_overrides(_CFG_SWIN_TINY, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="mask2former_swin_tiny")
    return model


@register_model(
    task="semantic-segmentation",
    family="mask2former",
    model_type="mask2former",
    model_class=Mask2FormerForSemanticSegmentation,
    default_config=_CFG_SWIN_SMALL,
)
def mask2former_swin_small(
    pretrained: bool | str = False,
    *,
    weights: Mask2FormerSwinSmallWeights | None = None,
    **overrides: object,
) -> Mask2FormerForSemanticSegmentation:
    r"""Mask2Former with Swin-Small backbone (Cheng et al., CVPR 2022).

    Builds a :class:`Mask2FormerForSemanticSegmentation` with the Swin-Small
    backbone (``embed_dim = 96``, ``depths = (2, 2, 18, 2)``,
    ``num_heads = (3, 6, 12, 24)``).  Same transformer head as
    :func:`mask2former_swin_tiny`; ADE20K validation mIoU of 51.3%.

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag; a tag string → that specific checkpoint.
    weights : Mask2FormerSwinSmallWeights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`Mask2FormerConfig`.

    Returns
    -------
    Mask2FormerForSemanticSegmentation
        Segmentation model with the Mask2Former-Swin-S configuration applied
        (or with ``overrides`` merged on top of it).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mask2former import mask2former_swin_small
    >>> model = mask2former_swin_small()
    >>> x = lucid.randn(1, 3, 384, 384)
    >>> out = model(x)
    >>> out.logits.shape[1]
    150
    """
    entry = weights_mod.resolve_weights(Mask2FormerSwinSmallWeights, pretrained, weights)
    model = Mask2FormerForSemanticSegmentation(apply_overrides(_CFG_SWIN_SMALL, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="mask2former_swin_small")
    return model


@register_model(
    task="semantic-segmentation",
    family="mask2former",
    model_type="mask2former",
    model_class=Mask2FormerForSemanticSegmentation,
    default_config=_CFG_SWIN_BASE,
)
def mask2former_swin_base(
    pretrained: bool | str = False,
    *,
    weights: Mask2FormerSwinBaseWeights | None = None,
    **overrides: object,
) -> Mask2FormerForSemanticSegmentation:
    r"""Mask2Former with Swin-Base backbone (Cheng et al., CVPR 2022).

    Builds a :class:`Mask2FormerForSemanticSegmentation` with the Swin-Base
    backbone (``embed_dim = 128``, ``depths = (2, 2, 18, 2)``,
    ``num_heads = (4, 8, 16, 32)``).  ADE20K validation mIoU of 53.9%.

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag; a tag string → that specific checkpoint.
    weights : Mask2FormerSwinBaseWeights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`Mask2FormerConfig`.

    Returns
    -------
    Mask2FormerForSemanticSegmentation
        Segmentation model with the Mask2Former-Swin-B configuration applied
        (or with ``overrides`` merged on top of it).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mask2former import mask2former_swin_base
    >>> model = mask2former_swin_base()
    >>> x = lucid.randn(1, 3, 384, 384)
    >>> out = model(x)
    >>> out.logits.shape[1]
    150
    """
    entry = weights_mod.resolve_weights(Mask2FormerSwinBaseWeights, pretrained, weights)
    model = Mask2FormerForSemanticSegmentation(apply_overrides(_CFG_SWIN_BASE, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="mask2former_swin_base")
    return model


@register_model(
    task="semantic-segmentation",
    family="mask2former",
    model_type="mask2former",
    model_class=Mask2FormerForSemanticSegmentation,
    default_config=_CFG_SWIN_LARGE,
)
def mask2former_swin_large(
    pretrained: bool | str = False,
    *,
    weights: Mask2FormerSwinLargeWeights | None = None,
    **overrides: object,
) -> Mask2FormerForSemanticSegmentation:
    r"""Mask2Former with Swin-Large backbone (Cheng et al., CVPR 2022).

    Builds a :class:`Mask2FormerForSemanticSegmentation` with the largest
    Swin backbone (``embed_dim = 192``, ``depths = (2, 2, 18, 2)``,
    ``num_heads = (6, 12, 24, 48)``).  Reaches ADE20K validation mIoU of
    56.1% — the headline result of the paper.

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag; a tag string → that specific checkpoint.
    weights : Mask2FormerSwinLargeWeights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`Mask2FormerConfig`.

    Returns
    -------
    Mask2FormerForSemanticSegmentation
        Segmentation model with the Mask2Former-Swin-L configuration applied
        (or with ``overrides`` merged on top of it).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mask2former import mask2former_swin_large
    >>> model = mask2former_swin_large()
    >>> x = lucid.randn(1, 3, 384, 384)
    >>> out = model(x)
    >>> out.logits.shape[1]
    150
    """
    entry = weights_mod.resolve_weights(Mask2FormerSwinLargeWeights, pretrained, weights)
    model = Mask2FormerForSemanticSegmentation(apply_overrides(_CFG_SWIN_LARGE, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="mask2former_swin_large")
    return model
