"""ResNet backbone and classification head."""

from typing import ClassVar, final, override

import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.models._utils._common import init_cnn_fan_out
from lucid.models._base import PretrainedModel
from lucid.models._tasks import ImageClassificationModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models.vision.resnet._config import ResNetConfig

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


@final
class _BasicBlock(nn.Module):
    r"""Two-convolution residual block used by ResNet-18 and ResNet-34.

    The block stacks two 3×3 convolutions (each followed by BatchNorm
    and ReLU) and adds a learned residual / identity shortcut.  Letting
    :math:`x` denote the block input and :math:`F` the two-conv stack,
    the forward computation is

    .. math::

        y = \sigma\!\left(F(x) + \text{shortcut}(x)\right),

    where :math:`\sigma` is ReLU.  The shortcut is the identity when
    ``stride == 1`` and ``in_channels == out_channels``; otherwise it
    is a 1×1 strided projection passed in via ``downsample``.

    Parameters
    ----------
    in_channels : int
        Input channel count :math:`C_{\text{in}}`.
    out_channels : int
        Output channel count :math:`C_{\text{out}}` — also the inner
        channel count, since BasicBlock has ``expansion == 1``.
    stride : int, optional, default=1
        Stride of the first 3×3 convolution.  Set to ``2`` at the first
        block of stage-2 / stage-3 / stage-4 to halve the spatial size.
    downsample : nn.Module or None, optional, default=None
        Optional projection applied to the shortcut branch when the
        input shape does not match the residual shape.  Built by
        :func:`_make_layer` as ``Conv2d(1×1, stride) → BatchNorm2d``.

    Attributes
    ----------
    expansion : ClassVar[int]
        ``1`` — the block does not multiply the output channel count.
    conv1, conv2 : nn.Conv2d
        The two 3×3 convolutions; ``conv1`` carries the stride,
        ``conv2`` always has stride 1.
    bn1, bn2 : nn.BatchNorm2d
        BatchNorm layers paired with each conv.
    relu : nn.ReLU
        In-place ReLU shared between the two activations and the final
        post-add activation.

    Notes
    -----
    From He et al., "Deep Residual Learning for Image Recognition",
    CVPR 2016 (arXiv:1512.03385), §3.3 and Figure 5 (left).  The
    "basic" block is preferred for shallower networks where the lower
    parameter count per block is acceptable; deeper ResNets switch to
    the bottleneck variant to keep FLOPs manageable.
    """

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Sequential | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    @override
    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        return self.relu(out)


class _Bottleneck(nn.Module):
    r"""Three-convolution bottleneck residual block used by ResNet-50/101/152.

    The block applies a 1×1 → 3×3 → 1×1 convolutional sequence (each
    followed by BatchNorm; ReLU between the first two stages) and adds
    a learned residual / identity shortcut.  The 1×1 layers compress
    and expand the channel dimension around the spatial 3×3 layer,
    reducing FLOPs at the cost of one extra layer per block.  Writing
    :math:`F` for the three-conv stack and :math:`x` for the input,

    .. math::

        y = \sigma\!\left(F(x) + \text{shortcut}(x)\right),

    with :math:`\sigma = \text{ReLU}`.  The output channel count is
    ``out_channels * expansion`` (four times the inner bottleneck
    width) — the "expansion" factor that gives the block its name.

    Parameters
    ----------
    in_channels : int
        Input channel count :math:`C_{\text{in}}`.
    out_channels : int
        Base bottleneck width.  The inner 3×3 conv runs at
        ``out_channels * width_mult`` channels; the final output width
        is ``out_channels * expansion``.
    stride : int, optional, default=1
        Stride of the middle 3×3 convolution.  Set to ``2`` at the
        first block of stage-2 / stage-3 / stage-4 to halve the
        spatial size.
    downsample : nn.Module or None, optional, default=None
        Projection applied to the shortcut branch when the input shape
        does not match the residual shape.  Built by
        :func:`_make_layer` as ``Conv2d(1×1, stride) → BatchNorm2d``.
    width_mult : int, optional, default=1
        Inner-width multiplier for the 3×3 convolution.  ``1`` for the
        canonical ResNet; ``2`` for Wide ResNet-50-2 / 101-2 (the
        output channel count stays the same).

    Attributes
    ----------
    expansion : ClassVar[int]
        ``4`` — the output channel count is four times ``out_channels``.
    conv1, conv2, conv3 : nn.Conv2d
        The 1×1 → 3×3 → 1×1 convolutional chain.  ``conv2`` carries
        the optional stride.
    bn1, bn2, bn3 : nn.BatchNorm2d
        BatchNorm layers paired with each conv.
    relu : nn.ReLU
        In-place ReLU shared across both intermediate activations and
        the final post-add activation.

    Notes
    -----
    From He et al., "Deep Residual Learning for Image Recognition",
    CVPR 2016 (arXiv:1512.03385), §4.1 and Figure 5 (right).  The
    width-multiplier extension comes from Zagoruyko & Komodakis, "Wide
    Residual Networks", BMVC 2016 (arXiv:1605.07146).
    """

    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Sequential | None = None,
        width_mult: int = 1,
    ) -> None:
        super().__init__()
        # out_channels is the bottleneck base width; inner width is scaled by
        # width_mult; final output width is out_channels * expansion.
        inner = out_channels * width_mult
        final = out_channels * self.expansion
        self.conv1 = nn.Conv2d(in_channels, inner, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(inner)
        self.conv2 = nn.Conv2d(inner, inner, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inner)
        self.conv3 = nn.Conv2d(inner, final, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(final)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    @override
    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        return self.relu(out)


# ---------------------------------------------------------------------------
# Shared ResNet stem + body builder
# ---------------------------------------------------------------------------


def _make_layer(
    block_cls: type[_BasicBlock] | type[_Bottleneck],
    in_channels: int,
    out_channels: int,
    num_blocks: int,
    stride: int = 1,
    width_mult: int = 1,
) -> tuple[nn.Sequential, int]:
    """Build one ResNet stage. Returns (layer, new_in_channels)."""
    expansion = block_cls.expansion
    final_channels = out_channels * expansion

    downsample: nn.Sequential | None = None
    if stride != 1 or in_channels != final_channels:
        downsample = nn.Sequential(
            nn.Conv2d(in_channels, final_channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(final_channels),
        )

    def _make_block(inc: int, outc: int, s: int = 1, ds: nn.Sequential | None = None) -> nn.Module:
        if block_cls is _Bottleneck:
            return _Bottleneck(inc, outc, stride=s, downsample=ds, width_mult=width_mult)
        return _BasicBlock(inc, outc, stride=s, downsample=ds)

    layers: list[nn.Module] = [
        _make_block(in_channels, out_channels, stride, downsample),
        *(_make_block(final_channels, out_channels) for _ in range(1, num_blocks)),
    ]

    return nn.Sequential(*layers), final_channels


def _build_body(
    config: ResNetConfig,
) -> tuple[
    nn.Sequential,
    nn.MaxPool2d,
    nn.Sequential,
    nn.Sequential,
    nn.Sequential,
    nn.Sequential,
    list[FeatureInfo],
]:
    block_cls: type[_BasicBlock] | type[_Bottleneck] = (
        _BasicBlock if config.block_type == "basic" else _Bottleneck
    )
    sc = config.stem_channels
    hs = config.hidden_sizes

    stem = nn.Sequential(
        nn.Conv2d(config.in_channels, sc, 7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(sc),
        nn.ReLU(inplace=True),
    )
    pool = nn.MaxPool2d(3, stride=2, padding=1)

    wm = config.bottleneck_width_mult
    cur = sc
    layer1, cur = _make_layer(block_cls, cur, hs[0], config.layers[0], width_mult=wm)
    layer2, cur = _make_layer(block_cls, cur, hs[1], config.layers[1], stride=2, width_mult=wm)
    layer3, cur = _make_layer(block_cls, cur, hs[2], config.layers[2], stride=2, width_mult=wm)
    layer4, cur = _make_layer(block_cls, cur, hs[3], config.layers[3], stride=2, width_mult=wm)

    exp = block_cls.expansion
    feature_info = [
        FeatureInfo(stage=1, num_channels=hs[0] * exp, reduction=4),
        FeatureInfo(stage=2, num_channels=hs[1] * exp, reduction=8),
        FeatureInfo(stage=3, num_channels=hs[2] * exp, reduction=16),
        FeatureInfo(stage=4, num_channels=hs[3] * exp, reduction=32),
    ]
    return stem, pool, layer1, layer2, layer3, layer4, feature_info


def _zero_init_residual(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, _Bottleneck):
            if m.bn3.weight is not None:
                nn.init.constant_(m.bn3.weight, 0.0)
        elif isinstance(m, _BasicBlock) and m.bn2.weight is not None:
            nn.init.constant_(m.bn2.weight, 0.0)


# ---------------------------------------------------------------------------
# ResNet backbone (task="base")
# ---------------------------------------------------------------------------


class ResNet(PretrainedModel, BackboneMixin):
    r"""ResNet feature-extracting backbone (no classification head).

    Implements the ResNet topology from He et al. in its *v1.5* form:
    the downsampling stride sits on the bottleneck's 3x3 convolution.
    The paper says only that "downsampling is performed by conv3_1,
    conv4_1, and conv5_1 with a stride of 2" without naming which conv
    carries it; the original MSRA release put it on the first 1x1 (v1).
    v1.5 is what every modern reference builds and what the shipped
    checkpoints were trained as., "Deep
    Residual Learning for Image Recognition", CVPR 2016
    (arXiv:1512.03385), restricted to the convolutional trunk: a 7×7
    stem at stride 2, a 3×3 max-pool at stride 2, then four stages of
    residual blocks producing feature maps at strides 4, 8, 16, and 32
    relative to the input.  No global pooling, no flatten, and no
    linear classifier — those live on :class:`ResNetForImageClassification`.

    The block topology is selected by ``config.block_type``: ResNet-18
    and ResNet-34 use :class:`_BasicBlock`, while the deeper ResNet-50
    / 101 / 152 / 200 / 269 and Wide ResNet variants use
    :class:`_Bottleneck`.  This class is the canonical feature
    extractor consumed by detection / segmentation heads (FPN,
    Faster R-CNN, Mask R-CNN, …) and by self-supervised pretraining
    pipelines.

    Parameters
    ----------
    config : ResNetConfig
        Frozen architecture spec.  Use the factory functions
        (:func:`resnet_18`, :func:`resnet_50`, …) for paper-cited
        variants — they instantiate the correct config for you.

    Attributes
    ----------
    config : ResNetConfig
        Stored copy of the config that built this model.
    stem : nn.Sequential
        7×7 conv (stride 2) → BatchNorm → ReLU — reduces input from
        :math:`H \times W` to :math:`H/2 \times W/2`.
    maxpool : nn.MaxPool2d
        3×3 max-pool with stride 2 and padding 1 — further reduces
        spatial size to :math:`H/4 \times W/4`.
    layer1, layer2, layer3, layer4 : nn.Sequential
        The four residual stages.  ``layer1`` keeps the input spatial
        size; ``layer2``/``layer3``/``layer4`` each halve it.  Output
        channel counts are ``hidden_sizes[i] * block.expansion``.
    feature_info : list[FeatureInfo]
        Per-stage descriptor (channels + reduction factor) exposed via
        :class:`BackboneMixin` for downstream FPN / decoder modules.

    Notes
    -----
    Each residual stage applies the standard skip-connection update

    .. math::

        y_{l+1} = \sigma\!\left(F(y_l, W_l) + W_s\, y_l\right),

    where :math:`F` is the in-block transformation, :math:`\sigma` is
    ReLU, and :math:`W_s` is either the identity (matched dimensions)
    or a 1×1 strided projection (dimension change at stage boundaries).
    The residual formulation lets gradients flow directly through the
    identity branch, making it practical to train networks with over
    a hundred layers — the central contribution of the original paper.

    The output of ``forward`` is a :class:`BaseModelOutput` whose
    ``last_hidden_state`` has shape ``(B, C_4, H/32, W/32)`` with
    :math:`C_4 = \text{hidden\_sizes}[3] \times \text{expansion}` —
    e.g. ``(B, 512, H/32, W/32)`` for ResNet-18 and
    ``(B, 2048, H/32, W/32)`` for ResNet-50.

    Examples
    --------
    Build a ResNet-50 backbone and run a single forward pass:

    >>> import lucid
    >>> from lucid.models.vision.resnet import resnet_50
    >>> backbone = resnet_50()
    >>> x = lucid.randn(2, 3, 224, 224)   # (batch, channels, H, W)
    >>> out = backbone(x)
    >>> out.last_hidden_state.shape
    (2, 2048, 7, 7)

    Inspect the per-stage feature map descriptors for FPN integration:

    >>> info = backbone.feature_info
    >>> [(fi.stage, fi.num_channels, fi.reduction) for fi in info]
    [(1, 256, 4), (2, 512, 8), (3, 1024, 16), (4, 2048, 32)]
    """

    config_class: ClassVar[type[ResNetConfig]] = ResNetConfig
    base_model_prefix: ClassVar[str] = "resnet"

    def __init__(self, config: ResNetConfig) -> None:
        super().__init__(config)
        stem, pool, l1, l2, l3, l4, fi = _build_body(config)
        self.stem = stem
        self.maxpool = pool
        self.layer1 = l1
        self.layer2 = l2
        self.layer3 = l3
        self.layer4 = l4
        self._feature_info = fi

        # Reference initialisation (He/MSRA fan-out convs, unit norms).
        # Without this the family starts from Lucid's generic
        # kaiming_uniform(a=sqrt(5)), a different distribution and gain
        # than the one the paper's schedule is tuned for.
        init_cnn_fan_out(self)

        if config.zero_init_residual:
            _zero_init_residual(self)

    @override
    def forward_features(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)

    @override
    def forward(self, x: Tensor) -> BaseModelOutput:
        return BaseModelOutput(last_hidden_state=self.forward_features(x))


# ---------------------------------------------------------------------------
# ResNet for image classification (task="image-classification")
# ---------------------------------------------------------------------------


class ResNetForImageClassification(ImageClassificationModel, ClassificationHeadMixin):
    r"""ResNet with a global-average-pooled linear classification head.

    Combines a :class:`ResNet` backbone with the standard ImageNet
    classification head: an :class:`~lucid.nn.AdaptiveAvgPool2d` to
    pool every spatial location into a single feature vector, an
    optional :class:`~lucid.nn.Dropout` (controlled by
    ``config.dropout``), and a :class:`~lucid.nn.Linear` projection to
    ``config.num_classes`` logits.  When ``labels`` are supplied to
    :meth:`forward`, a cross-entropy loss is computed and returned
    alongside the logits.

    Parameters
    ----------
    config : ResNetConfig
        Architecture spec.  Use the ``*_cls`` factory functions
        (:func:`resnet_18_cls`, :func:`resnet_50_cls`, …) to obtain a
        paper-cited configuration; pass a custom config to change
        ``num_classes`` or enable ``dropout`` / ``zero_init_residual``.

    Attributes
    ----------
    config : ResNetConfig
        Stored copy of the config that built this model.
    stem, maxpool, layer1, layer2, layer3, layer4
        Same backbone components as on :class:`ResNet`; see that
        class for shape semantics.
    avgpool : nn.AdaptiveAvgPool2d
        Global average pool collapsing the ``H/32 × W/32`` feature map
        to ``1 × 1``.
    classifier : nn.Module
        Built by :meth:`ClassificationHeadMixin._build_classifier` —
        either a bare :class:`~lucid.nn.Linear`
        (``dropout == 0.0``) or :class:`~lucid.nn.Sequential` wrapping
        :class:`~lucid.nn.Dropout` and :class:`~lucid.nn.Linear`.

    Notes
    -----
    The classification flow is

    .. math::

        \text{logits} = W\,\operatorname{Drop}\!
            \left(\operatorname{GAP}(\,\mathrm{backbone}(x)\,)\right) + b,

    where :math:`\operatorname{GAP}` is the global average pool and
    :math:`W \in \mathbb{R}^{C \times D}` with
    :math:`D = \text{hidden\_sizes}[3] \times \text{expansion}`.
    Loss is the standard categorical cross-entropy

    .. math::

        \mathcal{L} = -\frac{1}{N} \sum_{n=1}^{N}
            \log \operatorname{softmax}(\text{logits}_n)_{\,y_n},

    computed only when ``labels`` is not ``None`` — useful for the
    common pattern of running inference without a label tensor.

    Examples
    --------
    Run inference on a batch of 224x224 RGB images:

    >>> import lucid
    >>> from lucid.models.vision.resnet import resnet_50_cls
    >>> model = resnet_50_cls()
    >>> x = lucid.randn(4, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (4, 1000)
    >>> out.loss is None
    True

    Compute a loss given labels (e.g. during training):

    >>> labels = lucid.tensor([0, 1, 2, 3], dtype=lucid.int64)
    >>> out = model(x, labels=labels)
    >>> out.loss.shape
    ()
    """

    config_class: ClassVar[type[ResNetConfig]] = ResNetConfig
    base_model_prefix: ClassVar[str] = "resnet"

    def __init__(self, config: ResNetConfig) -> None:
        super().__init__(config)
        stem, pool, l1, l2, l3, l4, _ = _build_body(config)
        self.stem = stem
        self.maxpool = pool
        self.layer1 = l1
        self.layer2 = l2
        self.layer3 = l3
        self.layer4 = l4

        block_cls: type[_BasicBlock] | type[_Bottleneck] = (
            _BasicBlock if config.block_type == "basic" else _Bottleneck
        )
        final_channels = config.hidden_sizes[-1] * block_cls.expansion
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._build_classifier(final_channels, config.num_classes, dropout=config.dropout)

        init_cnn_fan_out(self)
        if config.zero_init_residual:
            _zero_init_residual(self)

    @override
    def forward(self, x: Tensor, labels: Tensor | None = None) -> ImageClassificationOutput:
        x = self.stem(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self._classify(x, labels)
