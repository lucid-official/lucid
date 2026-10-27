"""FCN configuration (Long et al., CVPR 2015)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta


@model_family_meta(
    canonical_name="FCN",
    citation=(
        'Long, Jonathan, et al. "Fully Convolutional Networks for '
        'Semantic Segmentation." Proceedings of the IEEE Conference on '
        "Computer Vision and Pattern Recognition, 2015, pp. 3431–3440."
    ),
    theory=r"""
    The Fully Convolutional Network is the first deep model trained
    end-to-end, pixels-to-pixels, for **semantic segmentation**.  Its
    core idea is to **convert a classification CNN into a dense
    predictor** by reinterpreting its fully-connected layers as
    :math:`1 \times 1` convolutions and learning to up-sample the
    coarse spatial output back to input resolution.

    A classifier such as VGG-16 ends with FC layers that collapse the
    feature map to a single :math:`1 \times 1 \times K` vector.  By
    rewriting each FC as a convolution with kernel equal to the input
    feature size, the network becomes input-agnostic and produces a
    *heatmap* :math:`\mathbb{R}^{K \times H/s \times W/s}` at the
    backbone's coarse stride :math:`s` (32 for VGG-16, 32 for most
    ResNets).

    To recover full resolution the paper introduces **skip
    architectures** — FCN-32s / FCN-16s / FCN-8s — that combine the
    coarse final layer prediction with the higher-resolution
    intermediate pool-4 and pool-3 predictions through learned
    deconvolutions:

    .. math::

        \hat{y} = \mathrm{upsample}_{2}\!\bigl(
            \mathrm{upsample}_{2}(p_5) + p_4
        \bigr) + p_3, \quad
        \text{(FCN-8s sketch)},

    where :math:`p_\ell` is the per-class score map from level
    :math:`\ell`.  This multi-resolution fusion gives the model
    *coarse* semantic information and *fine* spatial detail without
    needing extra encoder–decoder parameters.

    Training uses a per-pixel softmax cross-entropy loss summed over
    all output positions; the entire model is jointly fine-tuned
    from ImageNet-pretrained classification weights.  FCN
    demonstrated for the first time that segmentation could be a
    "natural" output mode of a CNN, paving the way for U-Net,
    DeepLab, and every subsequent dense-prediction architecture.
    """,
)
@dataclass(frozen=True)
class FCNConfig(ModelConfig):
    """Configuration for Fully Convolutional Network (FCN).

    FCN adapts a classification CNN backbone (ResNet) into a dense
    predictor by replacing fully-connected layers with convolutions and
    adding upsampling to restore spatial resolution.

    Architecture overview:
      ResNet backbone with dilated convolutions (layer3 dilation=2,
      layer4 dilation=4) → FCN head (3×3 conv + BN + ReLU + 1×1 conv)
      → bilinear upsample to input resolution.

      An auxiliary head on layer3 output provides additional supervision
      during training.

    Args:
        num_classes:         Number of output segmentation classes.
        in_channels:         Number of input image channels.
        backbone:            Backbone descriptor label ("resnet50" or "resnet101").
        variant:             FCN variant string ("fcn32s", "fcn16s", "fcn8s").
        classifier_hidden_channels:
                             Hidden channels in the FCN segmentation head.
        aux_hidden_channels: Hidden channels in the auxiliary head.
        dropout:             Dropout probability in the heads.

    Examples
    --------
    >>> from lucid.models.vision.fcn import FCNConfig
    >>> cfg = FCNConfig()
    >>> cfg.num_classes
    21
    >>> cfg.model_type
    'fcn'
    """

    model_type: ClassVar[str] = "fcn"

    num_classes: int = 21
    in_channels: int = 3
    backbone: str = "resnet50"
    variant: str = "fcn32s"
    classifier_hidden_channels: int = 512
    aux_hidden_channels: int = 256
    dropout: float = 0.1
    # Void / unlabelled marker in the VOC + COCO-with-VOC label set the
    # 21-class default targets.  These pixels must contribute no loss and no
    # gradient; the segmentation recipe that produced the shipped checkpoints
    # passes ``ignore_index=255`` to both the main and the auxiliary term.
    ignore_index: int = 255

    def __post_init__(self) -> None:
        # ``variant`` advertised fcn32s / fcn16s / fcn8s but drove nothing:
        # every value built the same single-scale dilated model, so a caller
        # asking for the skip architectures got the coarse one and a config
        # that claimed otherwise.  Only the single-scale head exists here, so
        # the other two now refuse rather than silently substituting it.
        if self.variant not in ("fcn32s", "fcn16s", "fcn8s"):
            raise ValueError(
                f"variant must be one of 'fcn32s', 'fcn16s', 'fcn8s', got " f"{self.variant!r}"
            )
        if self.variant != "fcn32s":
            raise NotImplementedError(
                f"variant={self.variant!r} needs the paper's skip fusion "
                f"(a scored pool4, and pool3 for fcn8s), which this module "
                f"does not build; only the single-scale head is implemented."
            )
