"""Registry factories for U-Net variants."""

from lucid.models._registry import register_model
from lucid.models.vision.unet._config import UNetConfig
from lucid.models.vision.unet._model import UNetForSemanticSegmentation
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained

_CFG_BASE = UNetConfig(
    num_classes=2, in_channels=1, base_channels=64, depth=4, bilinear=False, dropout=0.0
)

# Residual variants (ResUNet)
_CFG_RES_2D = UNetConfig(
    num_classes=2,
    in_channels=1,
    base_channels=64,
    depth=4,
    bilinear=False,
    dropout=0.0,
    block="res",
)

# 3-D variants (volumetric segmentation)
_CFG_3D = UNetConfig(
    num_classes=2,
    in_channels=1,
    # Cicek's analysis path is 32 -> 64 -> 128 -> 256 with the second conv of
    # each level carrying the doubling, so the *level outputs* are 64/128/256
    # and the bottom of the U is 512.
    base_channels=64,
    depth=3,
    bilinear=False,
    dropout=0.0,
    dim=3,
    double_before_pool=True,
)

_CFG_RES_3D = UNetConfig(
    num_classes=2,
    in_channels=1,
    base_channels=32,
    depth=3,
    bilinear=False,
    dropout=0.0,
    dim=3,
    block="res",
)


@register_model(
    task="semantic-segmentation",
    family="unet",
    model_type="unet",
    model_class=UNetForSemanticSegmentation,
    default_config=_CFG_BASE,
)
def unet(pretrained: bool = False, **overrides: object) -> UNetForSemanticSegmentation:
    r"""U-Net (Ronneberger et al., MICCAI 2015).

    Builds the canonical 2-D U-Net: 4-level encoder / decoder with
    ``base_channels = 64`` (channel schedule 64 -> 128 -> 256 -> 512 ->
    1024), ``in_channels = 1`` (single-channel biomedical default), and
    ``num_classes = 2``.  ConvTranspose2d upsampling in the decoder
    (set ``bilinear=True`` to use bilinear interpolation instead).
    Approximately 31M parameters.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``unet``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides
        Keyword overrides forwarded into :class:`UNetConfig` —
        ``num_classes``, ``in_channels`` (e.g. 3 for RGB), ``base_channels``,
        ``depth``, ``bilinear``, ``dropout``.

    Returns
    -------
    UNetForSemanticSegmentation
        Segmentation model with the U-Net configuration applied (or
        with ``overrides`` merged on top of it).

    Notes
    -----
    See Ronneberger et al., "U-Net: Convolutional Networks for Biomedical
    Image Segmentation", MICCAI 2015 (arXiv:1505.04597).  The skip
    connections that copy the encoder feature maps onto the decoder are
    the architectural key idea.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.unet import unet
    >>> model = unet(num_classes=4, in_channels=3)
    >>> x = lucid.randn(1, 3, 256, 256)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 4, 256, 256)
    """
    if pretrained:
        reject_unavailable_pretrained("unet")
    return UNetForSemanticSegmentation(apply_overrides(_CFG_BASE, overrides))


@register_model(
    task="semantic-segmentation",
    family="unet",
    model_type="unet",
    model_class=UNetForSemanticSegmentation,
    default_config=_CFG_RES_2D,
)
def res_unet_2d(pretrained: bool = False, **overrides: object) -> UNetForSemanticSegmentation:
    r"""ResUNet — U-Net with residual DoubleConv blocks, 2-D.

    Builds a 2-D U-Net with the standard channel / depth schedule but
    every DoubleConv block wraps an identity shortcut around its two
    3x3 convs.  This residual variant trains more reliably at greater
    depth and is a common drop-in upgrade for medical-imaging U-Nets.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``res_unet_2d``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides
        Keyword overrides forwarded into :class:`UNetConfig` (``num_classes``,
        ``in_channels``, ``base_channels``, ``depth``, ...).

    Returns
    -------
    UNetForSemanticSegmentation
        Segmentation model with the 2-D ResUNet configuration applied
        (or with ``overrides`` merged on top of it).

    Notes
    -----
    This is a residual U-Net, **not Deep ResUNet**.  The DoubleConv
    update with residual shortcut is

    .. math::

        y = \mathrm{ReLU}(F(x; W) + W_s x),

    the He et al. residual formulation applied at every encoder /
    decoder stage rather than only inside a backbone.

    Zhang, Liu, Wang, "Road Extraction by Deep Residual U-Net" (IEEE
    Geosci. Remote Sens. Lett. 2018, arXiv:1711.10684) describes a
    different network and is deliberately *not* cited as this model's
    source: it uses full pre-activation units (BN → ReLU → Conv twice,
    with no activation after the addition), replaces every pooling layer
    with a stride-2 convolution, and has three encoding levels plus a
    bridge — 15 convolutions and 7.8 M parameters against this model's
    post-activation blocks, max-pooling, four levels and 32.4 M
    parameters.  A caller wanting the paper's architecture wants a
    separate implementation, not this factory.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.unet import res_unet_2d
    >>> model = res_unet_2d()
    >>> x = lucid.randn(1, 1, 256, 256)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 2, 256, 256)
    """
    if pretrained:
        reject_unavailable_pretrained("res_unet_2d")
    return UNetForSemanticSegmentation(apply_overrides(_CFG_RES_2D, overrides))


@register_model(
    task="semantic-segmentation",
    family="unet",
    model_type="unet",
    model_class=UNetForSemanticSegmentation,
    default_config=_CFG_3D,
)
def unet_3d(pretrained: bool = False, **overrides: object) -> UNetForSemanticSegmentation:
    r"""3-D U-Net for volumetric segmentation (Cicek et al., MICCAI 2016).

    Builds the volumetric U-Net variant: all Conv2d / BatchNorm2d /
    MaxPool2d ops are replaced with their 3-D counterparts; "bilinear"
    upsampling becomes "trilinear".  Default ``base_channels = 32`` and
    ``depth = 3`` to keep memory tractable on volumetric inputs.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``unet_3d``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides
        Keyword overrides forwarded into :class:`UNetConfig` (``num_classes``,
        ``in_channels``, ``base_channels``, ``depth``, ``bilinear``, ...).

    Returns
    -------
    UNetForSemanticSegmentation
        Segmentation model with the 3-D U-Net configuration applied (or
        with ``overrides`` merged on top of it).

    Notes
    -----
    See Cicek et al., "3D U-Net: Learning Dense Volumetric Segmentation
    from Sparse Annotation", MICCAI 2016 (arXiv:1606.06650).  Volumetric
    convolutions add a depth axis, so memory cost scales linearly with
    the input depth — keep depth and base_channels modest.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.unet import unet_3d
    >>> model = unet_3d()
    >>> x = lucid.randn(1, 1, 64, 64, 64)   # (B, C, D, H, W)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 2, 64, 64, 64)
    """
    if pretrained:
        reject_unavailable_pretrained("unet_3d")
    return UNetForSemanticSegmentation(apply_overrides(_CFG_3D, overrides))


@register_model(
    task="semantic-segmentation",
    family="unet",
    model_type="unet",
    model_class=UNetForSemanticSegmentation,
    default_config=_CFG_RES_3D,
)
def res_unet_3d(pretrained: bool = False, **overrides: object) -> UNetForSemanticSegmentation:
    r"""3-D ResUNet — volumetric residual U-Net.

    H11 marker: **not a paper variant.**  No single paper defines a 3-D
    residual U-Net with this depth and channel schedule; it is the
    volumetric counterpart of :func:`res_unet_2d`, assembled from the
    same building blocks rather than reproduced from a publication.  It
    ships no weights and no accuracy claim.  Treat it as a starting
    point to configure, not as a reproduction of any published result.

    Builds a 3-D U-Net with residual DoubleConv blocks: the volumetric
    counterpart to :func:`res_unet_2d`.  Useful for deeper variants on
    biomedical volumes where the residual shortcut materially helps
    optimisation stability.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``res_unet_3d``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides
        Keyword overrides forwarded into :class:`UNetConfig`.

    Returns
    -------
    UNetForSemanticSegmentation
        Segmentation model with the 3-D ResUNet configuration applied
        (or with ``overrides`` merged on top of it).

    Notes
    -----
    Combines the 3-D U-Net topology of Cicek et al. (MICCAI 2016) with
    the residual DoubleConv blocks of Zhang et al. (2018).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.unet import res_unet_3d
    >>> model = res_unet_3d()
    >>> x = lucid.randn(1, 1, 64, 64, 64)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 2, 64, 64, 64)
    """
    if pretrained:
        reject_unavailable_pretrained("res_unet_3d")
    return UNetForSemanticSegmentation(apply_overrides(_CFG_RES_3D, overrides))
