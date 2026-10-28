"""MobileNet v2 backbone and classifier (Sandler et al., 2018).

Paper: "MobileNetV2: Inverted Residuals and Linear Bottlenecks"

Key idea: Inverted residual blocks — expand channels with 1×1 PW, apply
depthwise conv, then project back to a smaller channel count with a linear
(no-activation) 1×1 PW.  A residual shortcut is added only when stride==1
and input channels match output channels.
"""

from typing import ClassVar, override

import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._tasks import ImageClassificationModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models._utils._common import init_cnn_fan_out, make_divisible as _make_divisible
from lucid.models.vision.mobilenet_v2._config import MobileNetV2Config

# ---------------------------------------------------------------------------
# Inverted Residual block
# ---------------------------------------------------------------------------


class _InvertedResidual(nn.Module):
    """MobileNet v2 Inverted Residual (linear bottleneck) block."""

    def __init__(self, in_ch: int, out_ch: int, stride: int, expand_ratio: int) -> None:
        super().__init__()
        self._use_residual = stride == 1 and in_ch == out_ch
        hidden = in_ch * expand_ratio
        layers: list[nn.Module] = []

        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_ch, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU6(inplace=True),
            ]

        layers += [
            # Depthwise
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            # Pointwise linear (no activation)
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.conv = nn.Sequential(*layers)

    @override
    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        if self._use_residual:
            out = out + x
        return out


# ---------------------------------------------------------------------------
# Architecture spec  (t=expand_ratio, c=out_ch, n=repeat, s=stride)
# ---------------------------------------------------------------------------

# Tuple layout — (expand_ratio, out_channels, repeat, stride)
_INVERTED_RESIDUAL_SETTINGS: list[tuple[int, int, int, int]] = [
    (1, 16, 1, 1),
    (6, 24, 2, 2),
    (6, 32, 3, 2),
    (6, 64, 4, 2),
    (6, 96, 3, 1),
    (6, 160, 3, 2),
    (6, 320, 1, 1),
]


def _build_features(cfg: MobileNetV2Config) -> tuple[nn.Sequential, int]:
    """Return (features Sequential, num_out_channels)."""
    w = cfg.width_mult

    def _ch(c: int) -> int:
        return _make_divisible(c * w)

    # Stem
    stem_ch = _ch(32)
    layers: list[nn.Module] = [
        nn.Conv2d(cfg.in_channels, stem_ch, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(stem_ch),
        nn.ReLU6(inplace=True),
    ]

    in_ch = stem_ch
    for t, c, n, s in _INVERTED_RESIDUAL_SETTINGS:
        out_ch = _ch(c)
        for i in range(n):
            layers.append(
                _InvertedResidual(in_ch, out_ch, stride=s if i == 0 else 1, expand_ratio=t)
            )
            in_ch = out_ch

    # Head conv — the paper's own rule, stated in §6.1: scale 1280 by
    # ``max(1.0, width_mult)`` so wider models still get a wider head, but
    # narrow variants don't shrink the head below 1280.  (§3.4 describes
    # the inverted-residual block, not this.)
    last_ch = _make_divisible(1280 * max(1.0, cfg.width_mult))
    layers += [
        nn.Conv2d(in_ch, last_ch, 1, bias=False),
        nn.BatchNorm2d(last_ch),
        nn.ReLU6(inplace=True),
    ]
    return nn.Sequential(*layers), last_ch


# ---------------------------------------------------------------------------
# MobileNet v2 backbone  (task="base")
# ---------------------------------------------------------------------------


class MobileNetV2(PretrainedModel, BackboneMixin):
    r"""MobileNet v2 feature-extracting backbone (no classification head).

    Implements the inverted-residual / linear-bottleneck topology
    from Sandler et al., "MobileNetV2: Inverted Residuals and Linear
    Bottlenecks", CVPR 2018 (arXiv:1801.04381).  Each block expands
    a low-dimensional input up by an integer ``expand_ratio``, runs
    a depthwise :math:`3 \times 3` convolution in the wider interior
    space, then projects back to a narrow bottleneck — *without* a
    final non-linearity (the "linear bottleneck").  When stride and
    channel counts match, an identity shortcut is added around the
    whole block:

    .. math::

        y = x + \mathrm{Proj}\big(
            \mathrm{DW}\big( \mathrm{Expand}(x) \big)
        \big).

    The body is a 3×3 stem (stride 2) followed by seven inverted-residual
    stages and a final 1×1 head expansion to 1280 channels.  A global
    average pool collapses the final feature map to a single 1×1
    descriptor.

    Parameters
    ----------
    config : MobileNetV2Config
        Frozen architecture spec.  Use the factory functions
        (:func:`mobilenet_v2`, :func:`mobilenet_v2_075`) for
        paper-cited width-multiplier variants.

    Attributes
    ----------
    config : MobileNetV2Config
        Stored copy of the config that built this model.
    features : nn.Sequential
        Stem + seven inverted-residual stages + 1×1 head expansion.
        Channel counts scale with ``config.width_mult``.
    avgpool : nn.AdaptiveAvgPool2d
        Global average pool collapsing the final feature map to
        ``(B, C, 1, 1)``.
    feature_info : list[FeatureInfo]
        Per-stage descriptor (channels + reduction factor) exposed
        via :class:`BackboneMixin` for downstream FPN / decoder
        modules.

    Notes
    -----
    The inverted-residual block is the *inverse* of the classical
    bottleneck shape: instead of wide → narrow → wide, it does
    narrow → wide → narrow, with the residual connection on the
    *narrow* (low-dimensional) tensor.  Combined with the linear
    bottleneck (no ReLU on the projection), this preserves the
    representational capacity of the narrow channels while letting
    the wide interior use ReLU6 freely.  The result is roughly
    :math:`30\%` fewer parameters and FLOPs than MobileNet-v1 at
    comparable accuracy.

    Examples
    --------
    Build a MobileNet-v2 backbone and run a forward pass:

    >>> import lucid
    >>> from lucid.models.vision.mobilenet_v2 import mobilenet_v2
    >>> backbone = mobilenet_v2()
    >>> x = lucid.randn(2, 3, 224, 224)
    >>> out = backbone(x)
    >>> out.last_hidden_state.shape
    (2, 1280, 1, 1)
    """

    config_class: ClassVar[type[MobileNetV2Config]] = MobileNetV2Config
    base_model_prefix: ClassVar[str] = "mobilenet_v2"

    def __init__(self, config: MobileNetV2Config) -> None:
        super().__init__(config)
        features, num_features = _build_features(config)
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._num_features = num_features

        w = config.width_mult

        def _ch(c: int) -> int:
            return _make_divisible(c * w)

        # The stem is a stride-2 conv, so the pyramid starts at /2.  Seeding
        # the accumulator at 1 reported half the true downsampling factor for
        # every stage, which would place each level one octave too fine for any
        # detector consuming this backbone.
        cumulative = 2
        fi: list[FeatureInfo] = []
        for stage, (_, c, _, s) in enumerate(_INVERTED_RESIDUAL_SETTINGS):
            cumulative *= s
            fi.append(FeatureInfo(stage=stage + 1, num_channels=_ch(c), reduction=cumulative))
        self._feature_info = fi

        # Reference initialisation: He/MSRA fan-out normal on convs, unit
        # norms.  Lucid's Conv2d default is kaiming_uniform(a=sqrt(5)) — a
        # different distribution *and* gain — so a from-scratch run would
        # otherwise start somewhere the paper's schedule was never tuned for.
        init_cnn_fan_out(self, linear_std=0.01)

    @override
    def forward_features(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return self.avgpool(x)

    @override
    def forward(self, x: Tensor) -> BaseModelOutput:
        return BaseModelOutput(last_hidden_state=self.forward_features(x))


# ---------------------------------------------------------------------------
# MobileNet v2 for image classification  (task="image-classification")
# ---------------------------------------------------------------------------


class MobileNetV2ForImageClassification(ImageClassificationModel, ClassificationHeadMixin):
    r"""MobileNet v2 with global-average-pooled linear classification head.

    Combines a :class:`MobileNetV2` backbone with the standard
    ImageNet classification head: an :class:`~lucid.nn.AdaptiveAvgPool2d`
    to pool every spatial location into a single feature vector,
    a :class:`~lucid.nn.Dropout` (probability ``config.dropout``,
    default 0.2), and a :class:`~lucid.nn.Linear` projection to
    ``config.num_classes`` logits.  When ``labels`` are supplied to
    :meth:`forward`, a cross-entropy loss is computed and returned
    alongside the logits.

    Parameters
    ----------
    config : MobileNetV2Config
        Architecture spec.  Use the ``*_cls`` factory functions
        (:func:`mobilenet_v2_cls`, :func:`mobilenet_v2_075_cls`) for
        paper-cited configurations.

    Attributes
    ----------
    config : MobileNetV2Config
        Stored copy of the config that built this model.
    features : nn.Sequential
        Same inverted-residual stack as on :class:`MobileNetV2`.
    avgpool : nn.AdaptiveAvgPool2d
        Global average pool collapsing the final feature map to
        ``1 × 1``.
    drop : nn.Dropout
        Dropout layer (probability ``config.dropout``) applied
        before the linear classifier.
    classifier : nn.Linear
        Linear projection from the 1280-ch head (scaled by
        ``max(1.0, width_mult)``) to ``config.num_classes``.

    Notes
    -----
    The classification flow is

    .. math::

        \text{logits} = W \,\operatorname{Drop}\!\left(
            \operatorname{GAP}(\,\mathrm{backbone}(x)\,)
        \right) + b,

    where :math:`\operatorname{GAP}` is the global average pool.
    Loss is the standard categorical cross-entropy, computed only
    when ``labels`` is not ``None``.

    Examples
    --------
    Run inference on a batch of 224×224 RGB images:

    >>> import lucid
    >>> from lucid.models.vision.mobilenet_v2 import mobilenet_v2_cls
    >>> model = mobilenet_v2_cls()
    >>> x = lucid.randn(4, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (4, 1000)

    Retarget to CIFAR-10:

    >>> model = mobilenet_v2_cls(num_classes=10)
    >>> model.config.num_classes
    10
    """

    config_class: ClassVar[type[MobileNetV2Config]] = MobileNetV2Config
    base_model_prefix: ClassVar[str] = "mobilenet_v2"

    def __init__(self, config: MobileNetV2Config) -> None:
        super().__init__(config)
        features, num_features = _build_features(config)
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(p=config.dropout)
        self._build_classifier(num_features, config.num_classes)

        # Reference initialisation: He/MSRA fan-out normal on convs, unit
        # norms.  Lucid's Conv2d default is kaiming_uniform(a=sqrt(5)) — a
        # different distribution *and* gain — so a from-scratch run would
        # otherwise start somewhere the paper's schedule was never tuned for.
        init_cnn_fan_out(self, linear_std=0.01)

    @override
    def forward(self, x: Tensor, labels: Tensor | None = None) -> ImageClassificationOutput:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.drop(x.flatten(1))
        return self._classify(x, labels)
