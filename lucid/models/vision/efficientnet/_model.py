"""EfficientNet backbone and classifier (Tan & Le, 2019).

Paper: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"

Key ideas:
  1. MBConv (Mobile Inverted Bottleneck) blocks inherited from MobileNet v2.
  2. Squeeze-and-Excitation (SE) channel attention in each block.
  3. Compound scaling: width, depth, and resolution scaled simultaneously
     via a single compound coefficient φ.

B0 baseline block specs (expand_ratio, in_ch, out_ch, n_layers, stride, kernel):
  Stage 1: (1,  32,  16, 1, 1, 3)
  Stage 2: (6,  16,  24, 2, 2, 3)
  Stage 3: (6,  24,  40, 2, 2, 5)
  Stage 4: (6,  40,  80, 3, 2, 3)
  Stage 5: (6,  80, 112, 3, 1, 5)
  Stage 6: (6, 112, 192, 4, 2, 5)
  Stage 7: (6, 192, 320, 1, 1, 3)
  Head: Conv1×1(→1280) → AdaptiveAvgPool → Dropout → FC

Each MBConv block:
  1×1 expand conv (if expand_ratio > 1) → BN+SiLU
  k×k DW conv                            → BN+SiLU
  SE: squeeze → excite (sigmoid)
  1×1 project conv                       → BN  (no activation)
  + residual (if stride=1 & in==out)
"""

import math
from typing import ClassVar, final, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._tasks import ImageClassificationModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models._utils._common import make_divisible as _make_divisible
from lucid.models.vision.efficientnet._config import EfficientNetConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _round_channels(c: int, width_mult: float) -> int:
    return _make_divisible(c * width_mult)


def _round_layers(n: int, depth_mult: float) -> int:
    return max(1, math.ceil(n * depth_mult))


# ---------------------------------------------------------------------------
# Squeeze-and-Excitation
# ---------------------------------------------------------------------------


class _SEBlock(nn.Module):
    def __init__(self, in_channels: int, se_channels: int) -> None:
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, se_channels, 1)
        self.fc2 = nn.Conv2d(se_channels, in_channels, 1)

    @override
    def forward(self, x: Tensor) -> Tensor:
        scale = self.squeeze(x)
        scale = F.silu(self.fc1(scale))
        scale = F.sigmoid(self.fc2(scale))
        return x * scale


# ---------------------------------------------------------------------------
# MBConv block
# ---------------------------------------------------------------------------


@final
class _MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Convolution with SE."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expand_ratio: int,
        se_ratio: float,
        drop_connect_rate: float,
        # Section 5.2 / the official TPU builder use TF's batch-norm
        # constants: decay 0.99 (momentum 0.01 in this convention) and
        # epsilon 1e-3.  The B0-B4 checkpoints Lucid ships were converted
        # from the reference port, which kept the framework defaults, so
        # those are the defaults here -- ``bn_eps`` / ``bn_momentum`` on the
        # config let a caller ask for the paper's constants instead.
        bn_eps: float = 1e-5,
        bn_momentum: float = 0.1,
    ) -> None:
        super().__init__()
        self._has_residual = (stride == 1) and (in_channels == out_channels)
        self._drop_connect_rate = drop_connect_rate

        expanded = in_channels * expand_ratio

        layers: list[nn.Module] = []
        # Expansion phase (skip if ratio == 1)
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_channels, expanded, 1, bias=False),
                nn.BatchNorm2d(expanded, eps=bn_eps, momentum=bn_momentum),
                nn.SiLU(inplace=True),
            ]
        # Depthwise
        layers += [
            nn.Conv2d(
                expanded,
                expanded,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=expanded,
                bias=False,
            ),
            nn.BatchNorm2d(expanded, eps=bn_eps, momentum=bn_momentum),
            nn.SiLU(inplace=True),
        ]
        self.conv = nn.Sequential(*layers)

        # SE — the reference gates the block on ``0 < se_ratio <= 1`` and
        # omits it entirely otherwise.  ``max(1, ...)`` floored the reduction
        # width at 1, so ``se_ratio=0`` still built an SE that gated every
        # activation through a single scalar-per-image sigmoid — the opposite
        # of disabling it.
        self.se: _SEBlock | None
        if 0.0 < se_ratio <= 1.0:
            se_channels = max(1, int(in_channels * se_ratio))
            self.se = _SEBlock(expanded, se_channels)
        else:
            self.se = None

        # Projection
        self.project_conv = nn.Conv2d(expanded, out_channels, 1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels, eps=bn_eps, momentum=bn_momentum)

    @override
    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        if self.se is not None:
            out = self.se(out)
        out = self.project_bn(self.project_conv(out))
        if self._has_residual:
            # Stochastic depth (Tan & Le 2019 §3.3) — applied only when the
            # block has a real residual (in_ch == out_ch and stride == 1) and
            # only at training time.
            if self.training and self._drop_connect_rate > 0.0:
                keep_prob = 1.0 - self._drop_connect_rate
                # Per-sample binary mask: shape (B, 1, 1, 1) so it broadcasts.
                B = int(out.shape[0])
                mask_shape = (B,) + (1,) * (out.ndim - 1)
                random_tensor = lucid.rand(mask_shape, device=out.device.type)
                binary_mask = (random_tensor < keep_prob).float()
                # Inverted scaling so expectation matches at inference time.
                out = out * binary_mask / keep_prob
            out = out + x
        return out


# ---------------------------------------------------------------------------
# Block spec table and builder
# ---------------------------------------------------------------------------

# Tuple layout — (expand_ratio, in_ch, out_ch, n_layers, stride, kernel_size)
_BASE_SPECS: list[tuple[int, int, int, int, int, int]] = [
    (1, 32, 16, 1, 1, 3),
    (6, 16, 24, 2, 2, 3),
    (6, 24, 40, 2, 2, 5),
    (6, 40, 80, 3, 2, 3),
    (6, 80, 112, 3, 1, 5),
    (6, 112, 192, 4, 2, 5),
    (6, 192, 320, 1, 1, 3),
]


def _build_features(cfg: EfficientNetConfig) -> tuple[nn.Sequential, int]:
    """Return (feature_extractor, num_final_channels)."""
    w = cfg.width_mult
    d = cfg.depth_mult

    stem_ch = _round_channels(32, w)
    stem = [
        # Divergence, deliberate: the official TPU implementation builds
        # every conv with TF 'SAME' padding, which for a stride-2 even input
        # pads one more row/column on the bottom-right than the top-left and
        # so shifts the sampling grid by a pixel.  The reference reproduces
        # that only under its ``tf_efficientnet_*`` entrypoints; the
        # checkpoints this family loads come from the symmetric-padding
        # entrypoints, so symmetric padding is what makes them correct here.
        nn.Conv2d(cfg.in_channels, stem_ch, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(stem_ch, eps=cfg.bn_eps, momentum=cfg.bn_momentum),
        nn.SiLU(inplace=True),
    ]

    # Count total blocks for linear drop_connect schedule
    total_blocks = sum(_round_layers(s[3], d) for s in _BASE_SPECS)
    block_idx = 0

    layers: list[nn.Module] = list(stem)
    in_ch = stem_ch

    for expand, _base_in, base_out, base_n, stride, kernel in _BASE_SPECS:
        out_ch = _round_channels(base_out, w)
        n = _round_layers(base_n, d)
        for i in range(n):
            # Every reference divides by ``total_blocks``, not
            # ``total_blocks - 1``: the schedule is meant to *approach* the
            # configured rate over the trunk, not reach it at the last block.
            # Dividing by N-1 made every non-zero block ~6.7% over on B0.
            dc = cfg.drop_connect_rate * block_idx / max(1, total_blocks)
            layers.append(
                _MBConvBlock(
                    in_ch,
                    out_ch,
                    kernel_size=kernel,
                    stride=stride if i == 0 else 1,
                    expand_ratio=expand,
                    se_ratio=cfg.se_ratio,
                    drop_connect_rate=dc,
                    bn_eps=cfg.bn_eps,
                    bn_momentum=cfg.bn_momentum,
                )
            )
            in_ch = out_ch
            block_idx += 1

    head_ch = _round_channels(1280, w)
    layers += [
        nn.Conv2d(in_ch, head_ch, 1, bias=False),
        nn.BatchNorm2d(head_ch, eps=cfg.bn_eps, momentum=cfg.bn_momentum),
        nn.SiLU(inplace=True),
    ]
    return nn.Sequential(*layers), head_ch


# ---------------------------------------------------------------------------
# EfficientNet backbone  (task="base")
# ---------------------------------------------------------------------------


class EfficientNet(PretrainedModel, BackboneMixin):
    r"""EfficientNet feature-extracting backbone (no classification head).

    Implements the compound-scaled topology from Tan & Le,
    "EfficientNet: Rethinking Model Scaling for Convolutional Neural
    Networks", ICML 2019 (arXiv:1905.11946).  The baseline (B0) is a
    NAS-designed network of MBConv blocks — inverted-residual
    bottlenecks (MobileNet-v2 style) augmented with a
    squeeze-and-excitation module and the swish (SiLU) activation.
    Successive B-variants (B0 through B7) apply the paper's
    compound-scaling rule

    .. math::

        d = \alpha^\phi, \quad w = \beta^\phi, \quad r = \gamma^\phi,
        \qquad \alpha \beta^2 \gamma^2 \approx 2,

    to scale depth, width, and resolution jointly along a single
    coefficient :math:`\phi`.

    The body is a 3×3 stem (stride 2) followed by seven MBConv
    stages and a final 1×1 head expansion to ``round(1280 ·
    width_mult)`` channels.  Stochastic depth (``drop_connect_rate``)
    is applied linearly across blocks as a strong regulariser at
    large depth.

    Parameters
    ----------
    config : EfficientNetConfig
        Frozen architecture spec.  Use the factory functions
        (:func:`efficientnet_b0` through :func:`efficientnet_b7`)
        for paper-cited variants.

    Attributes
    ----------
    config : EfficientNetConfig
        Stored copy of the config that built this model.
    features : nn.Sequential
        Stem + seven MBConv stages + 1×1 head.  Channel and depth
        counts scale with ``config.width_mult`` and
        ``config.depth_mult``.
    avgpool : nn.AdaptiveAvgPool2d
        Global average pool collapsing the final feature map to
        ``(B, C, 1, 1)``.
    feature_info : list[FeatureInfo]
        Per-stage descriptor (channels + reduction factor) exposed
        via :class:`BackboneMixin` for downstream FPN / decoder
        modules.

    Notes
    -----
    Each MBConv block applies

    .. math::

        x \to \mathrm{Expand}_{1\times1} \to \mathrm{DW}_{k\times k}
            \to \mathrm{SE} \to \mathrm{Proj}_{1\times1},

    with a residual when stride is 1 and channel counts match.
    During training the residual branch is randomly dropped with
    probability ``drop_connect_rate * block_idx / (total - 1)`` —
    deeper blocks get a higher drop probability, matching the
    "deep network regularisation" recipe of stochastic depth.

    Examples
    --------
    Build an EfficientNet-B0 backbone and run a forward pass:

    >>> import lucid
    >>> from lucid.models.vision.efficientnet import efficientnet_b0
    >>> backbone = efficientnet_b0()
    >>> x = lucid.randn(2, 3, 224, 224)
    >>> out = backbone(x)
    >>> out.last_hidden_state.shape
    (2, 1280, 1, 1)
    """

    config_class: ClassVar[type[EfficientNetConfig]] = EfficientNetConfig
    base_model_prefix: ClassVar[str] = "efficientnet"

    def __init__(self, config: EfficientNetConfig) -> None:
        super().__init__(config)
        features, num_features = _build_features(config)
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._num_features = num_features

        w = config.width_mult
        # Stem is a stride-2 conv — count it, or every reduction comes out a
        # factor of two too small.
        cumulative = 2
        fi: list[FeatureInfo] = []
        for i, (_, _, base_out, _, stride, _) in enumerate(_BASE_SPECS):
            cumulative *= stride
            fi.append(
                FeatureInfo(
                    stage=i + 1,
                    num_channels=_round_channels(base_out, w),
                    reduction=cumulative,
                )
            )
        self._feature_info = fi

    @override
    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    @override
    def forward_features(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return self.avgpool(x)

    @override
    def forward(self, x: Tensor) -> BaseModelOutput:
        return BaseModelOutput(last_hidden_state=self.forward_features(x))


# ---------------------------------------------------------------------------
# EfficientNet for image classification  (task="image-classification")
# ---------------------------------------------------------------------------


class EfficientNetForImageClassification(ImageClassificationModel, ClassificationHeadMixin):
    r"""EfficientNet with global-average-pooled linear classification head.

    Combines an :class:`EfficientNet` backbone with the standard
    ImageNet classification head: an :class:`~lucid.nn.AdaptiveAvgPool2d`
    to pool every spatial location into a single feature vector,
    a :class:`~lucid.nn.Dropout` (probability ``config.dropout``,
    scaled per B-variant from 0.2 at B0 up to 0.5 at B7), and a
    :class:`~lucid.nn.Linear` projection to ``config.num_classes``
    logits.  When ``labels`` are supplied to :meth:`forward`, a
    cross-entropy loss is computed and returned alongside the
    logits.

    Parameters
    ----------
    config : EfficientNetConfig
        Architecture spec.  Use the ``*_cls`` factory functions
        (:func:`efficientnet_b0_cls` through
        :func:`efficientnet_b7_cls`) for paper-cited configurations.

    Attributes
    ----------
    config : EfficientNetConfig
        Stored copy of the config that built this model.
    features : nn.Sequential
        Same MBConv stack as on :class:`EfficientNet`.
    avgpool : nn.AdaptiveAvgPool2d
        Global average pool collapsing the final feature map to
        ``1 × 1``.
    drop : nn.Dropout
        Dropout layer (probability ``config.dropout``) applied
        before the linear classifier.
    classifier : nn.Linear
        Linear projection from ``round(1280 · width_mult)`` to
        ``config.num_classes``.

    Notes
    -----
    The classification flow is

    .. math::

        \text{logits} = W \,\operatorname{Drop}\!\left(
            \operatorname{GAP}(\,\mathrm{backbone}(x)\,)
        \right) + b,

    with cross-entropy loss computed when ``labels`` is provided.

    Examples
    --------
    Run inference on a batch of 224×224 RGB images:

    >>> import lucid
    >>> from lucid.models.vision.efficientnet import efficientnet_b0_cls
    >>> model = efficientnet_b0_cls()
    >>> x = lucid.randn(4, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (4, 1000)

    Retarget B3 to CIFAR-10:

    >>> from lucid.models.vision.efficientnet import efficientnet_b3_cls
    >>> model = efficientnet_b3_cls(num_classes=10)
    >>> model.config.num_classes
    10
    """

    config_class: ClassVar[type[EfficientNetConfig]] = EfficientNetConfig
    base_model_prefix: ClassVar[str] = "efficientnet"

    def __init__(self, config: EfficientNetConfig) -> None:
        super().__init__(config)
        features, num_features = _build_features(config)
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(p=config.dropout)
        self._build_classifier(num_features, config.num_classes)

    @override
    def forward(
        self,
        x: Tensor,
        labels: Tensor | None = None,
    ) -> ImageClassificationOutput:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.drop(x.flatten(1))
        logits = self.classifier(x)

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ImageClassificationOutput(logits=logits, loss=loss)
