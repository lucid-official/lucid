"""Registry factories for Attention U-Net variants."""

from dataclasses import replace

from lucid.models._registry import register_model
from lucid.models.vision.attention_unet._config import AttentionUNetConfig
from lucid.models.vision.attention_unet._model import AttentionUNetForSemanticSegmentation
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained

_CFG_BASE = AttentionUNetConfig(
    num_classes=2, in_channels=1, base_channels=64, depth=4, bilinear=False
)


_CFG_3D = replace(_CFG_BASE, spatial_dims=3)


@register_model(
    task="semantic-segmentation",
    family="attention_unet",
    model_type="attention_unet",
    model_class=AttentionUNetForSemanticSegmentation,
    default_config=_CFG_BASE,
)
def attention_unet(
    pretrained: bool = False, **overrides: object
) -> AttentionUNetForSemanticSegmentation:
    r"""Attention U-Net (Oktay et al., MIDL 2018).

    Builds an :class:`AttentionUNetForSemanticSegmentation` with the
    standard configuration: 4-level encoder / decoder, ``base_channels =
    64`` (channel schedule 64 -> 128 -> 256 -> 512 -> 1024),
    ``in_channels = 1`` (medical imaging default), and ``num_classes = 2``.
    Soft attention gates on every skip connection suppress irrelevant
    encoder activations.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``attention_unet``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides
        Keyword overrides forwarded into :class:`AttentionUNetConfig`
        (``num_classes``, ``in_channels`` for RGB inputs,
        ``base_channels``, ``depth``, ``bilinear``).

    Returns
    -------
    AttentionUNetForSemanticSegmentation
        Segmentation model with the standard Attention U-Net
        configuration applied (or with ``overrides`` merged on top of it).

    Notes
    -----
    See Oktay et al., "Attention U-Net: Learning Where to Look for the
    Pancreas", MIDL 2018 (arXiv:1804.03999).  The defining attention-gate
    update is

    .. math::

        \hat{x}^\ell = \sigma\!\bigl(\psi^\top
            \tanh(W_x x^\ell + W_g g^\ell)\bigr) \odot x^\ell,

    where :math:`x^\ell` is the encoder feature at level :math:`\ell`
    and :math:`g^\ell` is the up-sampled decoder feature serving as the
    gating signal.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.attention_unet import attention_unet
    >>> model = attention_unet(num_classes=4, in_channels=3)
    >>> x = lucid.randn(1, 3, 256, 256)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 4, 256, 256)
    """
    if pretrained:
        reject_unavailable_pretrained("attention_unet")
    return AttentionUNetForSemanticSegmentation(apply_overrides(_CFG_BASE, overrides))


@register_model(
    task="semantic-segmentation",
    family="attention_unet",
    model_type="attention_unet",
    model_class=AttentionUNetForSemanticSegmentation,
    default_config=_CFG_3D,
    params=91_866_693,
)
def attention_unet_3d(
    pretrained: bool = False, **overrides: object
) -> AttentionUNetForSemanticSegmentation:
    r"""Attention U-Net, 3-D (Oktay et al., MIDL 2018).

    The rank the paper actually specifies.  Its Implementation Details
    state "in contrast to the state-of-the-art CNN segmentation frameworks
    ... we propose a 3D-model to capture sufficient semantic context", and
    every released network is ``Conv3d`` / ``BatchNorm3d`` with trilinear
    resampling — it was built for CT volumes.

    Identical to :func:`attention_unet` in every other respect; only the
    convolution rank differs, so the attention gate that is the paper's
    contribution is the same mechanism in both.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``attention_unet_3d``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides
        Keyword overrides forwarded into :class:`AttentionUNetConfig`.

    Returns
    -------
    AttentionUNetForSemanticSegmentation
        Volumetric model expecting ``(B, C, D, H, W)`` input.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.attention_unet import attention_unet_3d
    >>> model = attention_unet_3d(base_channels=8, depth=2).eval()
    >>> out = model(lucid.randn(1, 1, 16, 32, 32))
    >>> out.logits.shape[2:]
    (16, 32, 32)
    """
    if pretrained:
        reject_unavailable_pretrained("attention_unet_3d")
    return AttentionUNetForSemanticSegmentation(apply_overrides(_CFG_3D, overrides))
