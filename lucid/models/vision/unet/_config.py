"""U-Net configuration (Ronneberger et al., MICCAI 2015)."""

from dataclasses import dataclass
from typing import ClassVar, Literal, get_args

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta
from lucid.models._utils._common import check_choice, check_positive

# The accepted values live here once: the field annotations and the
# ``__post_init__`` refusal both read them.
_Dim = Literal[2, 3]
_Block = Literal["basic", "res"]


@model_family_meta(
    canonical_name="U-Net",
    citation=(
        'Ronneberger, Olaf, et al. "U-Net: Convolutional Networks for '
        'Biomedical Image Segmentation." Medical Image Computing and '
        "Computer-Assisted Intervention, 2015, pp. 234–241."
    ),
    theory=r"""
    U-Net is a symmetric **encoder–decoder** architecture with dense
    skip connections, designed for biomedical segmentation where
    training data is scarce and every pixel matters.

    The encoder ("contracting path") follows the FCN philosophy of a
    classification CNN: pairs of :math:`3 \times 3` convolutions
    (with ReLU) followed by :math:`2 \times 2` max-pooling, doubling
    channel depth at every spatial halving:

    .. math::

        C_\ell = 2 \cdot C_{\ell - 1}, \quad
        (H, W)_\ell = (H, W)_{\ell - 1} / 2.

    The decoder ("expansive path") mirrors this structure: each
    stage performs a :math:`2 \times 2` *up-convolution* (transposed
    conv) that halves the channel count and doubles spatial size,
    **concatenates** the corresponding encoder feature map along the
    channel axis, and applies two :math:`3 \times 3` convs:

    .. math::

        x_\ell^{\mathrm{dec}} = \phi\!\bigl(
            [\, \mathrm{upconv}(x_{\ell+1}^{\mathrm{dec}}); \,
            x_\ell^{\mathrm{enc}} \,]
        \bigr).

    The crucial design choice is the **skip connection by
    concatenation** (not addition): it forwards the encoder's
    high-resolution but semantically shallow features directly into
    the decoder, letting the convs combine them with the deeper,
    semantically rich up-sampled features.  This is what allows the
    network to produce crisp pixel-level boundaries with very few
    training images (the paper trains on 30 cell images via heavy
    elastic augmentation).

    The final layer is a :math:`1 \times 1` convolution mapping to
    :math:`K` classes (softmax + per-pixel cross-entropy).
    Weighted-pixel loss is recommended in the paper to emphasise
    thin separating boundaries between touching cells — a small but
    practically important detail for cell segmentation.  U-Net's
    skip-connection pattern has since become the default starting
    point for almost every dense-prediction task, from natural
    images to volumetric medical scans (3-D U-Net, V-Net, …).
    """,
)
@dataclass(frozen=True)
class UNetConfig(ModelConfig):
    """Configuration for U-Net.

    U-Net is a fully convolutional encoder-decoder architecture with skip
    connections between corresponding encoder and decoder stages.  The
    architecture was originally proposed for biomedical image segmentation.

    Architecture overview:
      Encoder: depth × (DoubleConv + MaxPool2d) — halves spatial resolution.
      Bottleneck: DoubleConv at the deepest level.
      Decoder: depth × (Upsample + skip-cat + DoubleConv).
      Head: Conv2d(base_channels, num_classes, 1).

    Channel schedule (base_channels=64, depth=4):
      Encoder: 64 → 128 → 256 → 512
      Bottleneck: 1024
      Decoder: 512 → 256 → 128 → 64

    Args:
        num_classes:   Number of output segmentation classes.
        in_channels:   Number of input image channels.
        base_channels: Feature channels at the first encoder stage.
                       Doubles at each depth level.
        depth:         Number of encoder/decoder stages (excluding bottleneck).
        bilinear:      If True, use bilinear upsampling + Conv2d;
                       otherwise use ConvTranspose2d for learned upsampling.
        dropout:       Dropout probability applied in DoubleConv blocks.

    Examples
    --------
    >>> from lucid.models.vision.unet._config import UNetConfig
    >>> cfg = UNetConfig()
    >>> cfg.model_type
    'unet'
    >>> cfg.num_classes, cfg.in_channels
    (2, 1)
    """

    model_type: ClassVar[str] = "unet"

    num_classes: int = 2
    in_channels: int = 1
    base_channels: int = 64
    depth: int = 4
    bilinear: bool = False
    dropout: float = 0.0
    # Spatial dimensionality (2 → standard images, 3 → volumetric data).
    dim: _Dim = 2
    # Block style: "basic" = paper DoubleConv, "res" = residual DoubleConv.
    block: _Block = "basic"
    # 3D U-Net (Cicek 2016): "we avoid bottlenecks by doubling the number
    # of channels already before max pooling. We also adopt this scheme in
    # the synthesis path."  With it, the first conv of each level emits
    # half the level's width and the second emits all of it.  The 2D U-Net
    # keeps both convs at the full width, so this defaults off.
    double_before_pool: bool = False

    def __post_init__(self) -> None:
        owner = type(self).__name__
        check_positive(owner, num_classes=self.num_classes, in_channels=self.in_channels)
        check_positive(owner, base_channels=self.base_channels, depth=self.depth)
        check_choice(owner, "dim", self.dim, get_args(_Dim))
        check_choice(owner, "block", self.block, get_args(_Block))
