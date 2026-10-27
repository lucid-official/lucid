"""ResNeSt backbone and classifier (Zhang et al., 2020).

Paper: "ResNeSt: Split-Attention Networks" — https://arxiv.org/abs/2004.08955

Architecture overview for ResNeSt-50 (radix=2, groups=1, avd=True, avg_down=True):

  Deep Stem : Conv(3→32, k=3, s=2) → BN → ReLU
              Conv(32→32, k=3, s=1) → BN → ReLU
              Conv(32→64, k=3, s=1) → BN → ReLU
  MaxPool   : 3×3, stride=2
  Stage 1–4 : ResNeStBottleneck blocks with Split-Attention conv
  Head      : AdaptiveAvgPool(1×1) → Flatten → Linear

Split-Attention convolution (SplitAttn):
  - A single grouped conv produces ``radix`` branches at once
  - Global average pool over the branch sum produces an attention vector
  - Branches are recombined via softmax attention weights
"""

from typing import ClassVar, final, override

import math

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._tasks import ImageClassificationModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models._utils._common import zero_init_last_bn, make_divisible as _make_divisible
from lucid.models.vision.resnest._config import ResNeStConfig

# ---------------------------------------------------------------------------
# RadixSoftmax
# ---------------------------------------------------------------------------


@final
class _RadixSoftmax(nn.Module):
    """Per-radix softmax (or sigmoid when radix==1)."""

    def __init__(self, radix: int, groups: int) -> None:
        super().__init__()
        self._radix = radix
        self._groups = groups

    @override
    def forward(self, x: Tensor) -> Tensor:
        batch = x.shape[0]
        if self._radix > 1:
            # (B, groups, radix, C) → permute → (B, radix, groups, C)
            x = x.reshape(batch, self._groups, self._radix, -1).permute(0, 2, 1, 3)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = F.sigmoid(x)
        return x


# ---------------------------------------------------------------------------
# Split-Attention convolution
# ---------------------------------------------------------------------------


@final
class _SplitAttn(nn.Module):
    """Split-Attention (Splat) convolution.

    Parameters
    ----------
    in_channels:    Input channels.
    out_channels:   Output channels (= input channels in ResNeSt bottleneck).
    kernel_size:    Convolution kernel size.
    stride:         Spatial stride (applied to the main grouped conv).
    padding:        Spatial padding.
    dilation:       Dilation for the main conv.
    groups:         Cardinality groups.
    radix:          Number of split branches.
    rd_ratio:       Attention channel reduction ratio.
    rd_divisor:     Channel divisibility constraint.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        radix: int = 2,
        rd_ratio: float = 0.25,
        rd_divisor: int = 8,
    ) -> None:
        super().__init__()
        self._radix = radix
        mid_chs = out_channels * radix
        attn_chs = _make_divisible(in_channels * radix * rd_ratio, divisor=rd_divisor, min_value=32)

        # Single grouped conv producing all radix branches simultaneously
        self.conv = nn.Conv2d(
            in_channels,
            mid_chs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups * radix,
            bias=False,
        )
        self.bn0 = nn.BatchNorm2d(mid_chs)
        self.act0 = nn.ReLU(inplace=True)

        # Attention path — 1×1 convs with default bias=True
        self.fc1 = nn.Conv2d(out_channels, attn_chs, 1, groups=groups)
        self.bn1 = nn.BatchNorm2d(attn_chs)
        self.act1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(attn_chs, mid_chs, 1, groups=groups)
        self.rsoftmax = _RadixSoftmax(radix, groups)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn0(x)
        x = self.act0(x)

        b, rc, h, w = x.shape
        r = self._radix

        if r > 1:
            # Shape — (B, radix, C, H, W)
            x = x.reshape(b, r, rc // r, h, w)
            x_gap = x.sum(dim=1)  # (B, C, H, W)
        else:
            x_gap = x

        # Global average pool → (B, C, 1, 1)
        x_gap = x_gap.mean((2, 3), keepdim=True)
        x_gap = self.fc1(x_gap)
        x_gap = self.bn1(x_gap)
        x_gap = self.act1(x_gap)
        x_attn = self.fc2(x_gap)

        x_attn = self.rsoftmax(x_attn).reshape(b, -1, 1, 1)

        if r > 1:
            # x: (B, radix, C, H, W) ; x_attn: (B, radix*C, 1, 1) → (B, radix, C, 1, 1)
            out = (x * x_attn.reshape(b, r, rc // r, 1, 1)).sum(dim=1)
        else:
            out = x * x_attn

        return out


# ---------------------------------------------------------------------------
# ResNeSt Bottleneck
# ---------------------------------------------------------------------------


@final
class _DropBlock2d(nn.Module):
    r"""Structured spatial dropout (Ghiasi et al., 2018).

    Section 4.2 applies this to the last two stages: "As a structured
    variant of dropout, DropBlock randomly masks out local block regions,
    and is more effective than dropout for specifically regularizing
    convolutional layers."  Dropping isolated activations barely dents a
    convolutional feature map — the neighbours carry the same information —
    so the mask has to remove contiguous ``block_size x block_size``
    regions to actually withhold anything.

    Identity at inference and at ``drop_prob == 0``.
    """

    def __init__(self, drop_prob: float = 0.0, block_size: int = 3) -> None:
        super().__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    @override
    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        _B, _C, H, W = (int(v) for v in x.shape)
        bs = min(self.block_size, H, W)
        if bs < 1:
            return x
        # gamma is set so the *expected* fraction of zeroed units matches
        # drop_prob once each seed has been dilated to a bs x bs block.
        valid = float((H - bs + 1) * (W - bs + 1))
        gamma = self.drop_prob * H * W / (bs * bs * valid)
        seeds = (lucid.rand(*x.shape, device=x.device.type) < gamma).float()
        # Dilate each seed into its block: max-pool with a bs window.
        pad = bs // 2
        mask = F.max_pool2d(seeds, bs, stride=1, padding=pad)
        if int(mask.shape[2]) != H or int(mask.shape[3]) != W:
            mask = mask[:, :, :H, :W]
        keep = 1.0 - mask
        # Renormalise so the expected activation magnitude is unchanged.
        n_elems = float(math.prod(int(v) for v in keep.shape))
        scale = n_elems / keep.sum().clip(min=1.0)
        return x * keep * scale


@final
class _ResNeStBottleneck(nn.Module):
    """ResNeSt Bottleneck block.

    Parameters
    ----------
    inplanes:   Input channels.
    planes:     Base width (output = planes * 4 due to expansion=4).
    stride:     Spatial stride for the block.
    downsample: Optional downsampling module for the skip connection.
    radix:      Number of split branches in SplitAttn.
    groups:     Cardinality.
    avd:        Use averaged downsampling (AvgPool before/after SplitAttn).
    avd_first:  Place the AvgPool before (True) or after (False) SplitAttn.
    is_first:   True for the first block of the first stage (triggers avd even
                at stride=1 to match the reference implementation).
    dilation:   Dilation for the SplitAttn conv.
    """

    expansion: ClassVar[int] = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Sequential | None = None,
        radix: int = 2,
        groups: int = 1,
        avd: bool = True,
        avd_first: bool = False,
        is_first: bool = False,
        dilation: int = 1,
        bottleneck_width: int = 64,
        drop_block_rate: float = 0.0,
        drop_block_size: int = 3,
    ) -> None:
        super().__init__()
        self.downsample = downsample
        # Both references: ``int(planes * (bottleneck_width / 64.)) * groups``.
        group_width = int(planes * (bottleneck_width / 64.0)) * groups

        # Determine avd pooling stride
        if avd and (stride > 1 or is_first):
            avd_stride = stride
            conv_stride = 1
        else:
            avd_stride = 0
            conv_stride = stride

        # 1×1 expand
        self.conv1 = nn.Conv2d(inplanes, group_width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)
        self.act1 = nn.ReLU(inplace=True)

        # AVD pool before SplitAttn (avd_first=True)
        self.avd_first: nn.AvgPool2d | nn.Identity = (
            nn.AvgPool2d(3, avd_stride, padding=1)
            if avd_stride > 0 and avd_first
            else nn.Identity()
        )

        # Split-Attention 3×3 conv (always uses conv_stride=1 when avd active)
        self.conv2 = _SplitAttn(
            group_width,
            group_width,
            kernel_size=3,
            stride=conv_stride,
            padding=dilation,
            dilation=dilation,
            groups=groups,
            radix=radix,
        )
        # bn2 is Identity — SplitAttn already contains bn0 after its conv.
        # Both references reserve this slot for DropBlock.
        self.bn2: _DropBlock2d = _DropBlock2d(drop_block_rate, drop_block_size)
        self.act2: nn.Identity = nn.Identity()

        # AVD pool after SplitAttn (avd_first=False, default)
        self.avd_last: nn.AvgPool2d | nn.Identity = (
            nn.AvgPool2d(3, avd_stride, padding=1)
            if avd_stride > 0 and not avd_first
            else nn.Identity()
        )

        # 1×1 project
        self.conv3 = nn.Conv2d(group_width, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.act3 = nn.ReLU(inplace=True)

    @override
    def forward(self, x: Tensor) -> Tensor:
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.avd_first(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)

        out = self.avd_last(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = out + shortcut
        return self.act3(out)


# ---------------------------------------------------------------------------
# Downsampling helpers
# ---------------------------------------------------------------------------


def _make_avg_downsample(in_ch: int, out_ch: int, stride: int) -> nn.Sequential:
    """avg_down=True: AvgPool2d + 1×1 Conv (no stride in conv) + BN."""
    return nn.Sequential(
        nn.AvgPool2d(stride, stride=stride, ceil_mode=True),
        nn.Conv2d(in_ch, out_ch, 1, bias=False),
        nn.BatchNorm2d(out_ch),
    )


# ---------------------------------------------------------------------------
# Stage builder
# ---------------------------------------------------------------------------


def _make_layer(
    inplanes: int,
    planes: int,
    num_blocks: int,
    stride: int,
    radix: int,
    groups: int,
    avd: bool,
    avd_first: bool,
    avg_down: bool,
    dilation: int = 1,
    is_first_stage: bool = False,
    bottleneck_width: int = 64,
    drop_block_rate: float = 0.0,
    drop_block_size: int = 3,
) -> tuple[nn.Sequential, int]:
    """Build one stage of ResNeSt blocks.

    Returns ``(stage_module, out_channels)``.
    """
    expansion = _ResNeStBottleneck.expansion
    out_ch = planes * expansion

    # Downsample for first block
    downsample: nn.Sequential | None = None
    if stride != 1 or inplanes != out_ch:
        if avg_down:
            downsample = _make_avg_downsample(inplanes, out_ch, stride)
        else:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, out_ch, 1, stride=stride, bias=False), nn.BatchNorm2d(out_ch)
            )

    blocks: list[nn.Module] = [
        _ResNeStBottleneck(
            inplanes,
            planes,
            stride=stride,
            downsample=downsample,
            radix=radix,
            groups=groups,
            avd=avd,
            avd_first=avd_first,
            # AVD pooling fires only on blocks that actually downsample
            # (stride > 1); stage 1 keeps stride 1 and therefore gets no AVD,
            # matching the reference ResNeSt.  ``is_first`` stays available as
            # a per-block override but no stage enables it by default.
            is_first=is_first_stage,
            dilation=dilation,
            bottleneck_width=bottleneck_width,
            drop_block_rate=drop_block_rate,
            drop_block_size=drop_block_size,
        )
    ]
    for _ in range(1, num_blocks):
        blocks.append(
            _ResNeStBottleneck(
                out_ch,
                planes,
                stride=1,
                radix=radix,
                groups=groups,
                avd=avd,
                avd_first=avd_first,
                dilation=dilation,
                bottleneck_width=bottleneck_width,
                drop_block_rate=drop_block_rate,
                drop_block_size=drop_block_size,
            )
        )

    return nn.Sequential(*blocks), out_ch


# ---------------------------------------------------------------------------
# Deep stem
# ---------------------------------------------------------------------------


def _make_stem(
    in_channels: int, stem_width: int, deep_stem: bool
) -> tuple[nn.Sequential, nn.BatchNorm2d, int]:
    """Build the ResNeSt stem components to match timm key layout.

    Returns ``(conv1, bn1, stem_out_channels)`` where:
      - ``conv1``: Sequential whose indexed sub-modules match timm positions
      - ``bn1``: final BN (top-level on the model, not inside conv1)
      - ``stem_out_channels``: channels exiting the stem

    timm deep-stem layout (resnest50d)::

        conv1.0  Conv(in→sw, k=3, s=2)
        conv1.1  BN(sw)
        conv1.3  Conv(sw→sw, k=3, s=1)
        conv1.4  BN(sw)
        conv1.6  Conv(sw→2sw, k=3, s=1)
        bn1      BN(2sw)          ← top-level model attribute

    timm flat-stem layout (fixed 64 channels, independent of ``sw``)::

        conv1.0  Conv(in→64, k=7, s=2)
        bn1      BN(64)           ← top-level model attribute
    """
    # ``stem_width`` is a *deep*-stem parameter: the official repo's flat
    # branch is literally ``conv_layer(3, 64, kernel_size=7, ...)`` and timm
    # only computes ``stem_width * 2`` when the stem type contains 'deep'.
    # Deriving the flat width from stem_width made a non-default stem_width
    # silently resize the whole trunk's entry.
    out = stem_width * 2 if deep_stem else 64
    if deep_stem:
        conv1 = nn.Sequential(
            nn.Conv2d(in_channels, stem_width, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(stem_width, stem_width, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(stem_width, out, 3, stride=1, padding=1, bias=False),
        )
    else:
        conv1 = nn.Sequential(nn.Conv2d(in_channels, out, 7, stride=2, padding=3, bias=False))
    bn1 = nn.BatchNorm2d(out)
    return conv1, bn1, out


# ---------------------------------------------------------------------------
# Shared trunk builder
# ---------------------------------------------------------------------------


def _build_resnest_body(
    config: ResNeStConfig,
) -> tuple[
    nn.MaxPool2d,  # maxpool
    nn.Sequential,  # layer1
    nn.Sequential,  # layer2
    nn.Sequential,  # layer3
    nn.Sequential,  # layer4
    list[FeatureInfo],
    int,  # out_channels after layer4
]:
    """Build ResNeSt stages (excluding stem) to share between backbone and classifier."""
    stem_out = config.stem_width * 2

    pool = nn.MaxPool2d(3, stride=2, padding=1)

    widths = (64, 128, 256, 512)
    strides = (1, 2, 2, 2)
    layers_cfg = config.layers
    radix = config.radix
    groups = config.groups
    bottleneck_width = config.bottleneck_width
    # Section 4.2 applies DropBlock to the last two stages only.
    db_rate = config.drop_block_rate
    db_size = config.drop_block_size
    avd = config.avd
    avd_first = config.avd_first
    avg_down = config.avg_down

    cur = stem_out
    fi: list[FeatureInfo] = []

    layer1, cur = _make_layer(
        cur,
        widths[0],
        layers_cfg[0],
        strides[0],
        radix,
        groups,
        avd,
        avd_first,
        avg_down,
        bottleneck_width=bottleneck_width,
    )
    fi.append(FeatureInfo(stage=1, num_channels=widths[0] * 4, reduction=4))
    layer2, cur = _make_layer(
        cur,
        widths[1],
        layers_cfg[1],
        strides[1],
        radix,
        groups,
        avd,
        avd_first,
        avg_down,
        bottleneck_width=bottleneck_width,
    )
    fi.append(FeatureInfo(stage=2, num_channels=widths[1] * 4, reduction=8))
    layer3, cur = _make_layer(
        cur,
        widths[2],
        layers_cfg[2],
        strides[2],
        radix,
        groups,
        avd,
        avd_first,
        avg_down,
        bottleneck_width=bottleneck_width,
        drop_block_rate=db_rate,
        drop_block_size=db_size,
    )
    fi.append(FeatureInfo(stage=3, num_channels=widths[2] * 4, reduction=16))
    layer4, cur = _make_layer(
        cur,
        widths[3],
        layers_cfg[3],
        strides[3],
        radix,
        groups,
        avd,
        avd_first,
        avg_down,
        bottleneck_width=bottleneck_width,
        drop_block_rate=db_rate,
        drop_block_size=db_size,
    )
    fi.append(FeatureInfo(stage=4, num_channels=widths[3] * 4, reduction=32))

    return pool, layer1, layer2, layer3, layer4, fi, cur


# ---------------------------------------------------------------------------
# ResNeSt backbone  (task="base")
# ---------------------------------------------------------------------------


class ResNeSt(PretrainedModel, BackboneMixin):
    r"""ResNeSt feature-extracting backbone (no classification head).

    Implements the Split-Attention augmentation of the ResNet
    topology from Zhang et al., "ResNeSt: Split-Attention
    Networks", CVPR Workshops 2022 (arXiv:2004.08955).  Each
    bottleneck's :math:`3 \times 3` is replaced by a *Split-Attention*
    block that fuses ResNeXt cardinality, SKNet kernel selection,
    and SENet channel attention into a single unified primitive.
    The block splits the feature map into :math:`K` cardinal
    groups, then further splits each group into :math:`R` parallel
    *radix* branches, and computes per-radix softmax attention
    weights

    .. math::

        a_{r, c} = \frac{\exp(z_{r, c})}{\sum_{r'=1}^{R} \exp(z_{r', c})},
        \qquad V_c = \sum_{r=1}^{R} a_{r, c} \cdot U_{r, c}.

    Two further refinements complete the architecture: a *deep
    stem* (three 3×3 convolutions replacing the original 7×7
    stem) and an *average-pool downsampling* path that replaces
    strided convolutions in residual shortcuts and inside the
    Split-Attention block itself.

    Parameters
    ----------
    config : ResNeStConfig
        Frozen architecture spec.  Use the factory functions
        (:func:`resnest_50`, :func:`resnest_101`, …) for
        paper-cited variants.

    Attributes
    ----------
    config : ResNeStConfig
        Stored copy of the config that built this model.
    conv1 : nn.Module
        Deep stem (three 3×3 convolutions) or single 7×7 conv,
        depending on ``config.deep_stem``.
    bn1 : nn.BatchNorm2d
        BatchNorm paired with the stem output.
    act1 : nn.ReLU
        Post-stem ReLU activation.
    maxpool : nn.MaxPool2d
        3×3 max-pool at stride 2.
    layer1, layer2, layer3, layer4 : nn.Sequential
        The four Split-Attention residual stages.  Output strides
        are 4, 8, 16, and 32 relative to the input.
    feature_info : list[FeatureInfo]
        Per-stage descriptor (channels + reduction factor) exposed
        via :class:`BackboneMixin`.

    Notes
    -----
    When :math:`R = 1` the Split-Attention block degenerates to
    ResNeXt; at :math:`R \geq 2` it generalises SKNet to arbitrary
    radix counts with a softmax (rather than two-way sigmoid)
    gate.  ResNeSt-50 outperforms plain ResNet-50 by roughly 3
    ImageNet top-1 percentage points at the same parameter count,
    and serves as a strong backbone for downstream detection /
    segmentation pipelines.

    Examples
    --------
    Build a ResNeSt-50 backbone and run a forward pass:

    >>> import lucid
    >>> from lucid.models.vision.resnest import resnest_50
    >>> backbone = resnest_50()
    >>> x = lucid.randn(2, 3, 224, 224)
    >>> out = backbone(x)
    >>> out.last_hidden_state.shape
    (2, 2048, 7, 7)
    """

    config_class: ClassVar[type[ResNeStConfig]] = ResNeStConfig
    base_model_prefix: ClassVar[str] = "resnest"

    def __init__(self, config: ResNeStConfig) -> None:
        super().__init__(config)
        # Stem stored as top-level conv1/bn1 to match timm key layout
        self.conv1, self.bn1, _ = _make_stem(
            config.in_channels, config.stem_width, config.deep_stem
        )
        self.act1 = nn.ReLU(inplace=True)
        pool, l1, l2, l3, l4, fi, _ = _build_resnest_body(config)
        self.maxpool = pool
        self.layer1 = l1
        self.layer2 = l2
        self.layer3 = l3
        self.layer4 = l4
        self._feature_info = fi

        if config.zero_init_residual:
            # "Zero-initialize the last BN in each residual branch, so the
            # block starts as an identity" — the paper's training recipe and
            # what this config field has always advertised.  It was declared
            # and then never read.
            zero_init_last_bn(self, (_ResNeStBottleneck,), attr="bn3")

    @override
    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    @override
    def forward_features(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.act1(self.bn1(x))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    @override
    def forward(self, x: Tensor) -> BaseModelOutput:
        return BaseModelOutput(last_hidden_state=self.forward_features(x))


# ---------------------------------------------------------------------------
# ResNeSt for image classification  (task="image-classification")
# ---------------------------------------------------------------------------


class ResNeStForImageClassification(ImageClassificationModel, ClassificationHeadMixin):
    r"""ResNeSt with global-average-pooled linear classification head.

    Combines a :class:`ResNeSt` backbone with global average
    pooling, an optional :class:`~lucid.nn.Dropout` (probability
    ``config.dropout``), and a linear projection to
    ``config.num_classes`` logits.  When ``labels`` are supplied
    to :meth:`forward`, a cross-entropy loss is computed and
    returned alongside the logits.

    Parameters
    ----------
    config : ResNeStConfig
        Architecture spec.  Use the ``*_cls`` factory functions
        (:func:`resnest_50_cls`, :func:`resnest_101_cls`, …) for
        paper-cited configurations.

    Attributes
    ----------
    config : ResNeStConfig
        Stored copy of the config that built this model.
    conv1, bn1, act1, maxpool, layer1, layer2, layer3, layer4
        Same backbone components as on :class:`ResNeSt`; see that
        class for shape semantics.
    avgpool : nn.AdaptiveAvgPool2d
        Global average pool collapsing the final feature map to
        ``1 × 1``.
    classifier : nn.Module
        Built by :meth:`ClassificationHeadMixin._build_classifier` —
        a :class:`~lucid.nn.Linear` (or :class:`~lucid.nn.Sequential`
        with dropout) projecting from 2048 channels to
        ``config.num_classes``.

    Notes
    -----
    The classification flow is

    .. math::

        \text{logits} = W \,\operatorname{Drop}\!\left(
            \operatorname{GAP}(\,\mathrm{backbone}(x)\,)
        \right) + b.

    Examples
    --------
    Run inference on a batch of 224×224 RGB images:

    >>> import lucid
    >>> from lucid.models.vision.resnest import resnest_50_cls
    >>> model = resnest_50_cls()
    >>> x = lucid.randn(4, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (4, 1000)
    """

    config_class: ClassVar[type[ResNeStConfig]] = ResNeStConfig
    base_model_prefix: ClassVar[str] = "resnest"

    def __init__(self, config: ResNeStConfig) -> None:
        super().__init__(config)
        # Stem stored as top-level conv1/bn1 to match timm key layout
        self.conv1, self.bn1, _ = _make_stem(
            config.in_channels, config.stem_width, config.deep_stem
        )
        self.act1 = nn.ReLU(inplace=True)
        pool, l1, l2, l3, l4, _, out_ch = _build_resnest_body(config)
        self.maxpool = pool
        self.layer1 = l1
        self.layer2 = l2
        self.layer3 = l3
        self.layer4 = l4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._build_classifier(out_ch, config.num_classes, dropout=config.dropout)

    @override
    def forward(self, x: Tensor, labels: Tensor | None = None) -> ImageClassificationOutput:
        x = self.conv1(x)
        x = self.act1(self.bn1(x))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        logits = self.classifier(x)

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ImageClassificationOutput(logits=logits, loss=loss)
