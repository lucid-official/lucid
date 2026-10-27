"""ResNeXt backbone and classification head (Xie et al., 2017).

Paper: "Aggregated Residual Transformations for Deep Neural Networks"
ResNeXt replaces the 3×3 conv in the ResNet bottleneck with a grouped
convolution.  The intermediate width is determined by:
    width = (width_per_group * out_channels // base_width) * cardinality
where base_width = 64 (the channel count at the first stage).
"""

from typing import ClassVar, final, override

import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._utils._common import init_cnn_fan_out
from lucid.models._base import PretrainedModel
from lucid.models._tasks import ImageClassificationModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models.vision.resnext._config import ResNeXtConfig

# ---------------------------------------------------------------------------
# ResNeXt bottleneck block
# ---------------------------------------------------------------------------

_BASE_WIDTH: int = 64


@final
class _ResNeXtBottleneck(nn.Module):
    """1×1 → grouped 3×3 → 1×1 bottleneck with cardinality."""

    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Sequential | None = None,
        cardinality: int = 32,
        width_per_group: int = 4,
    ) -> None:
        super().__init__()
        # Compute grouped-conv width
        width = (width_per_group * out_channels // _BASE_WIDTH) * cardinality

        self.conv1 = nn.Conv2d(in_channels, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(
            width,
            width,
            3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
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
# Stage builder
# ---------------------------------------------------------------------------


def _zero_init_residual(model: nn.Module) -> None:
    """Zero the final BN gamma of every bottleneck (He et al., *Bag of Tricks*)."""
    for m in model.modules():
        if isinstance(m, _ResNeXtBottleneck) and m.bn3.weight is not None:
            nn.init.constant_(m.bn3.weight, 0.0)


def _make_layer(
    in_channels: int,
    out_channels: int,
    num_blocks: int,
    stride: int,
    cardinality: int,
    width_per_group: int,
) -> tuple[nn.Sequential, int]:
    """Build one ResNeXt stage. Returns (layer, new_in_channels)."""
    final_channels = out_channels * _ResNeXtBottleneck.expansion

    downsample: nn.Sequential | None = None
    if stride != 1 or in_channels != final_channels:
        downsample = nn.Sequential(
            nn.Conv2d(in_channels, final_channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(final_channels),
        )

    blocks: list[nn.Module] = [
        _ResNeXtBottleneck(
            in_channels,
            out_channels,
            stride=stride,
            downsample=downsample,
            cardinality=cardinality,
            width_per_group=width_per_group,
        )
    ]
    for _ in range(1, num_blocks):
        blocks.append(
            _ResNeXtBottleneck(
                final_channels,
                out_channels,
                cardinality=cardinality,
                width_per_group=width_per_group,
            )
        )

    return nn.Sequential(*blocks), final_channels


def _build_body(
    config: ResNeXtConfig,
) -> tuple[
    nn.Sequential,
    nn.MaxPool2d,
    nn.Sequential,
    nn.Sequential,
    nn.Sequential,
    nn.Sequential,
    list[FeatureInfo],
]:
    stem_channels = 64
    hidden_sizes = (64, 128, 256, 512)

    stem = nn.Sequential(
        nn.Conv2d(config.in_channels, stem_channels, 7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(stem_channels),
        nn.ReLU(inplace=True),
    )
    pool = nn.MaxPool2d(3, stride=2, padding=1)

    cur = stem_channels
    layer1, cur = _make_layer(
        cur,
        hidden_sizes[0],
        config.layers[0],
        1,
        config.cardinality,
        config.width_per_group,
    )
    layer2, cur = _make_layer(
        cur,
        hidden_sizes[1],
        config.layers[1],
        2,
        config.cardinality,
        config.width_per_group,
    )
    layer3, cur = _make_layer(
        cur,
        hidden_sizes[2],
        config.layers[2],
        2,
        config.cardinality,
        config.width_per_group,
    )
    layer4, cur = _make_layer(
        cur,
        hidden_sizes[3],
        config.layers[3],
        2,
        config.cardinality,
        config.width_per_group,
    )

    exp = _ResNeXtBottleneck.expansion
    feature_info = [
        FeatureInfo(stage=1, num_channels=hidden_sizes[0] * exp, reduction=4),
        FeatureInfo(stage=2, num_channels=hidden_sizes[1] * exp, reduction=8),
        FeatureInfo(stage=3, num_channels=hidden_sizes[2] * exp, reduction=16),
        FeatureInfo(stage=4, num_channels=hidden_sizes[3] * exp, reduction=32),
    ]
    return stem, pool, layer1, layer2, layer3, layer4, feature_info


# ---------------------------------------------------------------------------
# ResNeXt backbone (task="base")
# ---------------------------------------------------------------------------


class ResNeXt(PretrainedModel, BackboneMixin):
    r"""ResNeXt feature-extracting backbone (no classification head).

    Implements the grouped-convolution residual topology from Xie et
    al., "Aggregated Residual Transformations for Deep Neural
    Networks", CVPR 2017.  Macroscopically identical to ResNet-50+ — a
    :math:`7\times7` stride-2 stem, a :math:`3\times3` stride-2
    max-pool, then four stages of :class:`_ResNeXtBottleneck` blocks
    producing feature maps at strides 4, 8, 16, 32.  Microscopically
    different: the :math:`3\times3` convolution inside each bottleneck
    is *grouped* into ``config.cardinality`` groups of
    ``config.width_per_group`` channels each, which is mathematically
    equivalent to summing the outputs of ``cardinality`` parallel
    low-dimensional residual branches.

    Parameters
    ----------
    config : ResNeXtConfig
        Frozen architecture spec.  Use the factory functions
        (:func:`resnext_50_32x4d`, :func:`resnext_101_32x8d`, …) for
        paper-cited configurations.

    Attributes
    ----------
    config : ResNeXtConfig
        Stored copy of the config that built this model.
    stem : nn.Sequential
        Conv :math:`7\times7` stride-2 → BatchNorm → ReLU.
    maxpool : nn.MaxPool2d
        :math:`3\times3` stride-2 max-pool (further reduces spatial
        size to :math:`H/4 \times W/4`).
    layer1, layer2, layer3, layer4 : nn.Sequential
        The four residual stages of :class:`_ResNeXtBottleneck` blocks.
        ``layer1`` keeps the input spatial size; ``layer2`` / ``layer3``
        / ``layer4`` each halve it.  All four end with 2048 channels
        (same as ResNet-50+ — only the *internal* width changes).
    feature_info : list[FeatureInfo]
        Per-stage descriptor (channels + reduction factor) exposed via
        :class:`BackboneMixin`.

    Notes
    -----
    A ResNeXt block of cardinality :math:`C` computes

    .. math::

        y = x + \sum_{i=1}^{C} \mathcal{T}_i(x),

    where each :math:`\mathcal{T}_i` is a low-dimensional
    :math:`1\times1 \to 3\times3 \to 1\times1` bottleneck.  This
    "split-transform-merge" pattern is implemented as a single
    bottleneck with a ``groups=C`` :math:`3\times3` convolution — a
    one-line change versus ResNet but a substantial accuracy gain at
    matched FLOPs.  ResNeXt-50 (32x4d) outperforms ResNet-50 by roughly
    one percentage point on ImageNet top-1 at the same compute budget.

    Examples
    --------
    Build a ResNeXt-50 (32x4d) backbone:

    >>> import lucid
    >>> from lucid.models.vision.resnext import resnext_50_32x4d
    >>> backbone = resnext_50_32x4d()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = backbone(x)
    >>> out.last_hidden_state.shape   # (B, 2048, 7, 7)
    (1, 2048, 7, 7)
    """

    config_class: ClassVar[type[ResNeXtConfig]] = ResNeXtConfig
    base_model_prefix: ClassVar[str] = "resnext"

    def __init__(self, config: ResNeXtConfig) -> None:
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
    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    @override
    def forward_features(self, x: Tensor) -> Tensor:
        x = self.stem(x)
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
# ResNeXt for image classification (task="image-classification")
# ---------------------------------------------------------------------------


class ResNeXtForImageClassification(ImageClassificationModel, ClassificationHeadMixin):
    r"""ResNeXt image classifier (backbone + GAP + linear head).

    Combines a :class:`ResNeXt` backbone with the standard ImageNet
    head: :class:`~lucid.nn.AdaptiveAvgPool2d` to pool every spatial
    location into a single 2048-dim feature, an optional
    :class:`~lucid.nn.Dropout` (controlled by ``config.dropout``), and
    a :class:`~lucid.nn.Linear` projection to ``config.num_classes``.
    When ``labels`` are supplied to :meth:`forward`, a cross-entropy
    loss is returned alongside the logits.

    Parameters
    ----------
    config : ResNeXtConfig
        Architecture spec.  Use the ``*_cls`` factory functions
        (:func:`resnext_50_32x4d_cls`, :func:`resnext_101_32x8d_cls`,
        …) for paper-cited configurations.

    Attributes
    ----------
    config : ResNeXtConfig
        Stored copy of the config that built this model.
    stem, maxpool, layer1, layer2, layer3, layer4
        Same backbone components as on :class:`ResNeXt`.
    avgpool : nn.AdaptiveAvgPool2d
        Global average pool to :math:`1\times1`.
    classifier : nn.Module
        Final classifier built by
        :meth:`ClassificationHeadMixin._build_classifier`; bare linear
        if ``dropout == 0.0``, otherwise a
        :class:`~lucid.nn.Sequential` wrapping
        :class:`~lucid.nn.Dropout` and :class:`~lucid.nn.Linear`.

    Notes
    -----
    From Xie et al., "Aggregated Residual Transformations for Deep
    Neural Networks", CVPR 2017.  Loss is the standard categorical
    cross-entropy

    .. math::

        \mathcal{L} = -\frac{1}{N} \sum_{n=1}^{N}
            \log \operatorname{softmax}(\text{logits}_n)_{\,y_n}.

    The headline ImageNet result is ResNeXt-101 (64x4d) reaching
    20.4% top-1 error — meaningfully better than ResNet-152 at the same
    FLOP budget.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnext import resnext_50_32x4d_cls
    >>> model = resnext_50_32x4d_cls()
    >>> x = lucid.randn(2, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (2, 1000)
    """

    config_class: ClassVar[type[ResNeXtConfig]] = ResNeXtConfig
    base_model_prefix: ClassVar[str] = "resnext"

    def __init__(self, config: ResNeXtConfig) -> None:
        super().__init__(config)
        stem, pool, l1, l2, l3, l4, _ = _build_body(config)
        self.stem = stem
        self.maxpool = pool
        self.layer1 = l1
        self.layer2 = l2
        self.layer3 = l3
        self.layer4 = l4

        final_channels = 512 * _ResNeXtBottleneck.expansion  # 2048
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._build_classifier(final_channels, config.num_classes, dropout=config.dropout)

        # Reference initialisation: He/MSRA fan-out normal on convs, unit
        # norms — Lucid's Conv2d default is a different distribution and gain.
        init_cnn_fan_out(self)
        if config.zero_init_residual:
            _zero_init_residual(self)

    @override
    def forward(
        self,
        x: Tensor,
        labels: Tensor | None = None,
    ) -> ImageClassificationOutput:
        x = self.stem(x)
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
