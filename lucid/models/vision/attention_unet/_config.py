"""Attention U-Net configuration (Oktay et al., MIDL 2018)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta
from lucid.models._utils._common import check_choice, check_positive


@model_family_meta(
    canonical_name="Attention U-Net",
    citation=(
        'Oktay, Ozan, et al. "Attention U-Net: Learning Where to Look '
        'for the Pancreas." Medical Imaging with Deep Learning, 2018.'
    ),
    theory=r"""
    Attention U-Net augments the standard U-Net skip pathways with
    **soft spatial attention gates**.  In a plain U-Net every
    encoder feature is concatenated unconditionally into the decoder,
    so irrelevant background activations still propagate to the
    output.  For small structures like the pancreas — where the
    target occupies a tiny fraction of the field of view — this
    leaks gradient signal into the wrong regions and hurts both
    convergence and final Dice.

    Each Attention Gate (AG) at decoder level :math:`\ell` takes
    two inputs: the skip feature :math:`x^{\ell} \in \mathbb{R}^{C
    \times H \times W}` from the encoder and a *gating signal*
    :math:`g^{\ell+1} \in \mathbb{R}^{C_g \times H/2 \times W/2}`
    from the deeper decoder stage.  Both are projected to a common
    intermediate space with :math:`1 \times 1` convolutions and
    additively combined:

    .. math::

        q^{\ell} = \psi^{\top}\!\bigl(
            \sigma_1\!\bigl(
                W_x^{\top} x^{\ell} + W_g^{\top} g^{\ell+1} + b_g
            \bigr)
        \bigr) + b_\psi,
        \qquad
        \alpha^{\ell} = \sigma_2(q^{\ell}),

    where :math:`\sigma_1` is ReLU and :math:`\sigma_2` is sigmoid,
    so :math:`\alpha^{\ell}(h, w) \in [0, 1]` is a per-pixel
    attention coefficient.  The gated skip
    :math:`\tilde{x}^{\ell} = \alpha^{\ell} \odot x^{\ell}` then
    replaces the raw skip in the U-Net concatenation.

    Because the gate is differentiable and trained jointly with the
    rest of the network, no extra supervision (bounding boxes,
    landmarks, …) is needed: the model **learns where to look**
    purely from the segmentation loss.  The added cost is small
    (two :math:`1 \times 1` convs per gate) but consistently
    improves Dice on small-organ targets and tightens the
    response near tissue boundaries — a now-standard add-on in
    medical-imaging U-Net variants.
    """,
)
@dataclass(frozen=True)
class AttentionUNetConfig(ModelConfig):
    """Configuration for Attention U-Net.

    Extends the standard U-Net architecture (Ronneberger et al., 2015) by
    adding soft attention gates on skip connections.  Each gate computes a
    spatial attention map from the skip feature and the gating signal from
    the decoder, suppressing irrelevant activations before concatenation.

    Architecture overview:
      Encoder: depth × (2×Conv3x3-BN-ReLU + MaxPool2x2)
      Bottleneck: 2×Conv3x3-BN-ReLU
      Decoder: depth × (Upsample/ConvTranspose + AttentionGate + Cat + 2×Conv3x3-BN-ReLU)
      Head: Conv1x1 → num_classes

    Args:
        num_classes:   Number of output segmentation classes.
        in_channels:   Number of input image channels.
        base_channels: Feature channels at the first encoder stage.
                       Doubles at each depth level.
        depth:         Number of encoder/decoder stages (excluding bottleneck).
        bilinear:      If True, use bilinear upsampling; otherwise ConvTranspose2d.
        deep_supervision: Attach a 1x1 classifier to *every* decoder level,
            resample each to the input resolution, and fuse the stack with a
            final 1x1 convolution — the ``dsv`` in the reference's
            ``unet_CT_multi_att_dsv`` model.  Despite the name this is not a
            training-only auxiliary loss: the prediction itself comes out of
            the fusion, so switching it on changes inference too.  Defaults
            off so existing checkpoints keep loading.

    Examples
    --------
    >>> from lucid.models.vision.attention_unet._config import AttentionUNetConfig
    >>> cfg = AttentionUNetConfig()
    >>> cfg.model_type
    'attention_unet'
    >>> cfg.num_classes, cfg.in_channels
    (2, 1)
    """

    model_type: ClassVar[str] = "attention_unet"

    num_classes: int = 2
    # 2 for images, 3 for volumes.  The paper's Implementation Details
    # state "we propose a 3D-model to capture sufficient semantic context"
    # and every released network is Conv3d — but the 2-D form is what most
    # callers reach for, so it stays the default and 3-D is opt-in.
    spatial_dims: int = 2
    in_channels: int = 1
    base_channels: int = 64
    depth: int = 4
    bilinear: bool = False
    deep_supervision: bool = False

    def __post_init__(self) -> None:
        check_positive(self, "num_classes", "in_channels", "base_channels", "depth")
        check_choice(self, "spatial_dims", (2, 3))
