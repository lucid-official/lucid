"""FCN — Fully Convolutional Network (Long et al., CVPR 2015).

Paper: "Fully Convolutional Networks for Semantic Segmentation"

Key innovation
--------------
Replace all fully-connected layers in a classification CNN with
convolutions so the network can accept arbitrary input sizes and produce
a dense per-pixel prediction.  Skip connections from shallower stages
fuse coarse semantic information with fine spatial detail.

Architecture (ResNet variant)
------------------------------
  ResNet backbone with dilated convolutions:
    - layer3 uses dilation=2 (stride replaced) → 1/8 spatial resolution
    - layer4 uses dilation=4 (stride replaced) → still 1/8

  FCN head (on layer4 output, 2048 channels):
    Conv2d(2048, 512, 3, pad=1) → BN → ReLU → Dropout → Conv2d(512, K, 1)
    Bilinear upsample ×8 → (B, K, H, W)

  Auxiliary head (on layer3 output, 1024 channels, training only):
    Conv2d(1024, 256, 3, pad=1) → BN → ReLU → Dropout → Conv2d(256, K, 1)
    Bilinear upsample ×8 → (B, K, H, W)

Losses (training)
-----------------
  primary loss:   cross-entropy on main head output vs. targets
  auxiliary loss: cross-entropy on aux head output vs. targets
  total loss:     primary + 0.4 × auxiliary
"""

from dataclasses import dataclass
from typing import ClassVar, final, override

import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._tasks import SemanticSegmentationModel
from lucid.models._output import SemanticSegmentationOutput
from lucid.models.vision.fcn._config import FCNConfig

# ---------------------------------------------------------------------------
# ResNet backbone (dilated, self-contained)
# ---------------------------------------------------------------------------


_AUX_LOSS_WEIGHT: float = 0.5


class _Bottleneck(nn.Module):
    """1×1 → 3×3 (dilated) → 1×1 bottleneck block."""

    expansion: ClassVar[int] = 4

    def __init__(
        self,
        in_ch: int,
        mid_ch: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: nn.Sequential | None = None,
    ) -> None:
        super().__init__()
        out_ch = mid_ch * self.expansion
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(
            mid_ch,
            mid_ch,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample

    @override
    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out: Tensor = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return F.relu(out + identity)


def _make_layer(
    in_ch: int,
    mid_ch: int,
    num_blocks: int,
    stride: int = 1,
    dilation: int = 1,
    first_dilation: int | None = None,
) -> tuple[nn.Sequential, int]:
    """Build one ResNet stage with optional dilation.

    The reference dilated ResNet (reference_vision ``replace_stride_with_dilation``)
    applies the *previous* stage's dilation to the first block of a dilated
    stage and the new dilation to the remaining blocks — e.g. layer3's first
    block uses dilation 1 then 2, layer4's first uses 2 then 4.
    ``first_dilation`` carries that previous value (defaults to ``dilation``
    for non-dilated stages, where they coincide).
    """
    out_ch = mid_ch * 4
    if first_dilation is None:
        first_dilation = dilation
    ds: nn.Sequential | None = None
    if stride != 1 or in_ch != out_ch:
        ds = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        )
    blocks: list[nn.Module] = [
        _Bottleneck(in_ch, mid_ch, stride=stride, dilation=first_dilation, downsample=ds)
    ]
    for _ in range(1, num_blocks):
        blocks.append(_Bottleneck(out_ch, mid_ch, dilation=dilation))
    return nn.Sequential(*blocks), out_ch


@final
class _DilatedResNet(nn.Module):
    """ResNet backbone with dilated convolutions for dense prediction.

    Replaces stride in layer3 and layer4 with dilation so the output
    feature map retains 1/8 the input spatial resolution rather than 1/32.

    Args:
        in_channels: Number of input image channels.
        layers:      Per-stage repetition counts, e.g. (3, 4, 6, 3).

    Attributes:
        c4_channels: Channel count of layer3 output (for auxiliary head).
        c5_channels: Channel count of layer4 output (for FCN head).
    """

    def __init__(
        self,
        in_channels: int,
        layers: tuple[int, int, int, int],
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1, c2 = _make_layer(64, 64, layers[0], stride=1)
        self.layer2, c3 = _make_layer(c2, 128, layers[1], stride=2)
        # layer3: stride→dilation 2; first block keeps the previous dilation 1
        self.layer3, c4 = _make_layer(c3, 256, layers[2], stride=1, dilation=2, first_dilation=1)
        # layer4: stride→dilation 4; first block keeps the previous dilation 2
        self.layer4, c5 = _make_layer(c4, 512, layers[3], stride=1, dilation=4, first_dilation=2)

        self.c4_channels: int = c4
        self.c5_channels: int = c5

    @override
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return (c4, c5) feature maps.

        Returns:
            c4: (B, c4_channels, H/8, W/8) — layer3 output.
            c5: (B, c5_channels, H/8, W/8) — layer4 output.
        """
        x = self.stem(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        c4: Tensor = self.layer3(x)
        c5: Tensor = self.layer4(c4)
        return c4, c5


# ---------------------------------------------------------------------------
# FCN segmentation head
# ---------------------------------------------------------------------------


@final
class _FCNHead(nn.Sequential):
    """FCN segmentation head: Conv3×3-BN-ReLU-Dropout-Conv1×1.

    Args:
        in_channels:  Input feature channel count.
        hidden_channels: Intermediate channel count (default 512 for main head).
        num_classes:  Number of output segmentation classes.
        dropout:      Dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Conv2d(hidden_channels, num_classes, 1),
        )


# ---------------------------------------------------------------------------
# FCN model
# ---------------------------------------------------------------------------


@dataclass
class FCNOutput(SemanticSegmentationOutput):
    r"""Segmentation output that also carries the auxiliary head.

    The reference runs its aux classifier whenever one exists — not only
    when targets are supplied — and returns both predictions, each already
    upsampled to the input size.  Computing it solely inside the loss
    branch made the aux prediction unreachable: a caller wanting to weight
    the aux term itself, or simply to inspect it, had no way to get at it.

    Attributes
    ----------
    aux_logits : Tensor or None, optional
        Auxiliary head logits at input resolution, or ``None`` when the
        model was built without an auxiliary classifier.

    Examples
    --------
    The value a segmentation forward returns. It is not exported from the
    package — you receive one rather than construct one.

    >>> import lucid
    >>> from lucid.models.vision.fcn import fcn_resnet50
    >>> model = fcn_resnet50(num_classes=21).eval()
    >>> out = model(lucid.randn(1, 3, 224, 224))
    >>> type(out).__name__
    'FCNOutput'
    >>> out.logits.shape          # (B, K, H, W) — full input resolution
    (1, 21, 224, 224)

    ``aux_logits`` is the auxiliary head's map at the same resolution. It
    exists to be added to the loss during training and is ignored at
    inference, so a prediction reads ``logits`` alone.

    >>> out.aux_logits.shape
    (1, 21, 224, 224)
    >>> out.loss is None          # no targets were passed
    True
    """

    aux_logits: Tensor | None = None


class FCNForSemanticSegmentation(SemanticSegmentationModel):
    r"""Fully Convolutional Network for semantic segmentation (Long et al., CVPR 2015).

    The original deep-learning semantic segmentation architecture: a
    classification backbone is repurposed as a *fully convolutional*
    network by replacing fully-connected layers with 1x1 convolutions
    and adding a final upsampling stage to recover input resolution.
    In Lucid's implementation the backbone is a dilated ResNet
    (``layer3`` dilation = 2, ``layer4`` dilation = 4) so the deepest
    feature map stays at stride 8 instead of stride 32, giving denser
    spatial sampling without sacrificing the receptive field.

    A primary FCN head produces ``num_classes`` logits from the C5
    feature, and an auxiliary head on C4 stabilises optimisation
    (used at training time only).  Bilinear interpolation lifts both
    heads to the input resolution.

    Parameters
    ----------
    config : FCNConfig
        Frozen architecture spec.  Use :func:`fcn_resnet50` /
        :func:`fcn_resnet101` for the standard variants.

    Attributes
    ----------
    config : FCNConfig
        Stored copy of the config that built this model.
    backbone : _DilatedResNet
        Dilated ResNet trunk producing C4 (stride 8, used by aux head)
        and C5 (stride 8, used by main head) — both at the same
        resolution thanks to atrous convolution in layer4.
    classifier : _FCNHead
        3x3 conv -> BatchNorm -> ReLU -> Dropout -> 1x1 conv producing
        ``num_classes`` channels from the C5 feature.
    aux_classifier : _FCNHead
        Auxiliary head with the same structure applied to C4; its loss
        is weighted by 0.4 and added to the main loss only during
        training.

    Notes
    -----
    See Long et al., "Fully Convolutional Networks for Semantic
    Segmentation", CVPR 2015 (arXiv:1411.4038).  The bilinear
    upsampling at the end is equivalent to a learnable transpose
    convolution initialised to bilinear weights — but kept fixed in
    this modernised dilated-ResNet variant.  Training loss is

    .. math::

        \mathcal{L} = \mathcal{L}_\mathrm{CE}(\mathrm{main}, y)
                    + 0.4\, \mathcal{L}_\mathrm{CE}(\mathrm{aux}, y),

    with categorical cross-entropy **averaged** over the non-ignored
    pixels.  The paper (§4.1) defines the objective as an unnormalised
    spatial *sum*, and its quoted learning rates are calibrated to that
    scale; the reference recipe that produced the shipped checkpoints
    uses the mean instead, and this implementation follows the
    reference so those weights reproduce.  Scale the learning rate
    accordingly if switching to the paper's formulation.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.fcn import fcn_resnet50
    >>> model = fcn_resnet50()
    >>> x = lucid.randn(1, 3, 512, 512)
    >>> out = model(x)
    >>> out.logits.shape    # (B, num_classes, H, W)
    (1, 21, 512, 512)
    """

    config_class: ClassVar[type[FCNConfig]] = FCNConfig
    base_model_prefix: ClassVar[str] = "fcn"

    def __init__(self, config: FCNConfig) -> None:
        super().__init__(config)
        self._cfg = config

        # Backbone layer counts by name
        _layer_counts: dict[str, tuple[int, int, int, int]] = {
            "resnet50": (3, 4, 6, 3),
            "resnet101": (3, 4, 23, 3),
        }
        layers = _layer_counts.get(config.backbone, (3, 4, 6, 3))

        self.backbone = _DilatedResNet(config.in_channels, layers)

        self.classifier = _FCNHead(
            self.backbone.c5_channels,
            config.classifier_hidden_channels,
            config.num_classes,
            config.dropout,
        )
        self.aux_classifier = _FCNHead(
            self.backbone.c4_channels,
            config.aux_hidden_channels,
            config.num_classes,
            config.dropout,
        )

    @override
    def forward(
        self,
        x: Tensor,
        targets: Tensor | None = None,
    ) -> SemanticSegmentationOutput:
        """Run FCN forward pass.

        Args:
            x:       (B, C, H, W) image batch.
            targets: Optional (B, H, W) integer ground-truth masks.

        Returns:
            ``SemanticSegmentationOutput`` with logits (B, num_classes, H, W)
            and optional cross-entropy loss.
        """
        iH = int(x.shape[2])
        iW = int(x.shape[3])

        # Backbone outputs — (B, c4, H/8, W/8) and (B, c5, H/8, W/8)
        c4, c5 = self.backbone(x)

        # Main head
        main_feat: Tensor = self.classifier(c5)
        logits: Tensor = F.interpolate(
            main_feat, size=(iH, iW), mode="bilinear", align_corners=False
        )

        # Run the aux head whenever one exists, as the reference does —
        # its output is part of the model's prediction, not a detail of
        # the loss.
        aux_feat: Tensor = self.aux_classifier(c4)
        aux_logits: Tensor = F.interpolate(
            aux_feat, size=(iH, iW), mode="bilinear", align_corners=False
        )

        loss: Tensor | None = None
        if targets is not None:
            ignore_index = self._cfg.ignore_index
            main_loss: Tensor = F.cross_entropy(logits, targets, ignore_index=ignore_index)

            aux_loss: Tensor = F.cross_entropy(aux_logits, targets, ignore_index=ignore_index)
            # The paper has no auxiliary head at all, so the recipe that
            # trained these checkpoints is the only authority: it weights
            # the aux term at 0.5.  0.4 is the PSPNet / mmsegmentation
            # convention and belongs to different weights.
            loss = main_loss + _AUX_LOSS_WEIGHT * aux_loss

        return FCNOutput(logits=logits, loss=loss, aux_logits=aux_logits)
