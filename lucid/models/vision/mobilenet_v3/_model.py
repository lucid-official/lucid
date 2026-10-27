"""MobileNet v3 backbone and classifier (Howard et al., 2019).

Paper: "Searching for MobileNetV3"

Key innovations:
  - Hard-swish and Hard-sigmoid activations for mobile efficiency.
  - Squeeze-and-Excitation (SE) blocks with Hard-sigmoid gating.
  - NAS-designed architecture for Large and Small variants.
"""

from typing import ClassVar, override

import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._tasks import ImageClassificationModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models._utils._common import init_cnn_fan_out, make_divisible as _make_divisible
from lucid.models.vision.mobilenet_v3._config import MobileNetV3Config

# MobileNet v3 overrides the default batch-norm hyperparameters: the NAS
# reference implementation trains every block's normalisation with
# ``eps=0.001`` and ``momentum=0.01`` (a slower running-stat update than
# the conventional 0.1).  The eps in particular is load-bearing for
# inference — the published checkpoints fold their running statistics
# against ``eps=0.001``, so a 1e-5 default shifts every activation and
# breaks numerical parity with the source weights.
_BN_EPS = 0.001
_BN_MOMENTUM = 0.01

# ---------------------------------------------------------------------------
# Squeeze-and-Excitation block (Hard-sigmoid gating)
# ---------------------------------------------------------------------------


class _SEBlock(nn.Module):
    def __init__(self, in_channels: int, se_channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, se_channels, 1)
        self.fc2 = nn.Conv2d(se_channels, in_channels, 1)

    @override
    def forward(self, x: Tensor) -> Tensor:
        scale = self.pool(x)
        scale = F.relu(self.fc1(scale), inplace=True)
        scale = F.hardsigmoid(self.fc2(scale))
        return x * scale


# ---------------------------------------------------------------------------
# Bottleneck block (InvertedResidual + optional SE + configurable activation)
# ---------------------------------------------------------------------------


class _InvertedResidual(nn.Module):
    """MobileNet v3 bottleneck with optional SE and activation choice."""

    def __init__(
        self,
        in_ch: int,
        exp_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int,
        use_se: bool,
        use_hs: bool,
    ) -> None:
        super().__init__()
        self._use_residual = stride == 1 and in_ch == out_ch

        act: nn.Module = nn.Hardswish() if use_hs else nn.ReLU(inplace=True)
        layers: list[nn.Module] = []

        if exp_ch != in_ch:
            layers += [
                nn.Conv2d(in_ch, exp_ch, 1, bias=False),
                nn.BatchNorm2d(exp_ch, eps=_BN_EPS, momentum=_BN_MOMENTUM),
                act,
            ]
        pad = (kernel_size - 1) // 2
        act2: nn.Module = nn.Hardswish() if use_hs else nn.ReLU(inplace=True)
        layers += [
            nn.Conv2d(
                exp_ch, exp_ch, kernel_size, stride=stride, padding=pad, groups=exp_ch, bias=False
            ),
            nn.BatchNorm2d(exp_ch, eps=_BN_EPS, momentum=_BN_MOMENTUM),
            act2,
        ]

        if use_se:
            se_ch = _make_divisible(exp_ch * 0.25)
            layers.append(_SEBlock(exp_ch, se_ch))

        layers += [
            nn.Conv2d(exp_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch, eps=_BN_EPS, momentum=_BN_MOMENTUM),
        ]
        self.block = nn.Sequential(*layers)

    @override
    def forward(self, x: Tensor) -> Tensor:
        out = self.block(x)
        if self._use_residual:
            out = out + x
        return out


# ---------------------------------------------------------------------------
# Architecture specs
# Tuple layout — (in_ch, exp_ch, out_ch, kernel_size, use_se, use_hs, stride)
# ---------------------------------------------------------------------------

_LARGE_SPECS: list[tuple[int, int, int, int, bool, bool, int]] = [
    (16, 16, 16, 3, False, False, 1),
    (16, 64, 24, 3, False, False, 2),
    (24, 72, 24, 3, False, False, 1),
    (24, 72, 40, 5, True, False, 2),
    (40, 120, 40, 5, True, False, 1),
    (40, 120, 40, 5, True, False, 1),
    (40, 240, 80, 3, False, True, 2),
    (80, 200, 80, 3, False, True, 1),
    (80, 184, 80, 3, False, True, 1),
    (80, 184, 80, 3, False, True, 1),
    (80, 480, 112, 3, True, True, 1),
    (112, 672, 112, 3, True, True, 1),
    (112, 672, 160, 5, True, True, 2),
    (160, 960, 160, 5, True, True, 1),
    (160, 960, 160, 5, True, True, 1),
]

_SMALL_SPECS: list[tuple[int, int, int, int, bool, bool, int]] = [
    (16, 16, 16, 3, True, False, 2),
    (16, 72, 24, 3, False, False, 2),
    (24, 88, 24, 3, False, False, 1),
    (24, 96, 40, 5, True, True, 2),
    (40, 240, 40, 5, True, True, 1),
    (40, 240, 40, 5, True, True, 1),
    (40, 120, 48, 5, True, True, 1),
    (48, 144, 48, 5, True, True, 1),
    (48, 288, 96, 5, True, True, 2),
    (96, 576, 96, 5, True, True, 1),
    (96, 576, 96, 5, True, True, 1),
]


def _apply_width(
    specs: list[tuple[int, int, int, int, bool, bool, int]], w: float
) -> list[tuple[int, int, int, int, bool, bool, int]]:
    """Scale channel counts by width_mult."""
    out: list[tuple[int, int, int, int, bool, bool, int]] = []
    for in_ch, exp_ch, o_ch, k, se, hs, s in specs:
        out.append(
            (
                _make_divisible(in_ch * w),
                _make_divisible(exp_ch * w),
                _make_divisible(o_ch * w),
                k,
                se,
                hs,
                s,
            )
        )
    return out


def _build_features(cfg: MobileNetV3Config) -> tuple[nn.Sequential, int, int]:
    """Return (features, last_block_ch, head_out_ch)."""
    w = cfg.width_mult
    specs = _LARGE_SPECS if cfg.variant == "large" else _SMALL_SPECS

    stem_ch = _make_divisible(16 * w)
    layers: list[nn.Module] = [
        nn.Conv2d(cfg.in_channels, stem_ch, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(stem_ch, eps=_BN_EPS, momentum=_BN_MOMENTUM),
        nn.Hardswish(),
    ]

    scaled = _apply_width(specs, w)
    in_ch = stem_ch
    for _spec_in, exp_ch, out_ch, k, se, hs, stride in scaled:
        # Clamp first in_ch to stem_ch; subsequent blocks chain naturally
        layers.append(_InvertedResidual(in_ch, exp_ch, out_ch, k, stride, se, hs))
        in_ch = out_ch

    # Head conv
    last_block_ch = in_ch
    # The reference derives this from the *already-rounded* last block width:
    # ``6 * lastconv_input_channels``.  Composing the other way —
    # ``make_divisible(576 * w)`` — is a different function of ``w``; the two
    # agree at width_mult=1.0 (6*96=576, 6*160=960) and diverge below it.
    penultimate_ch = 6 * last_block_ch
    layers += [
        # Table 2 marks SE on MobileNetV3-Small's final 1x1 to 576, and there
        # is no SE module here.  That follows the reference: neither the
        # canonical vision library's ``mobilenet_v3_small`` nor timm's
        # ``mobilenetv3_small_100`` places SE on that layer, and the shipped
        # checkpoints have no tensors for one.
        nn.Conv2d(last_block_ch, penultimate_ch, 1, bias=False),
        nn.BatchNorm2d(penultimate_ch, eps=_BN_EPS, momentum=_BN_MOMENTUM),
        nn.Hardswish(),
    ]
    # Large → 1280, Small → 1024 (paper Table 2 / the reference).
    #
    # The final NBN layer does NOT shrink below width 1 — the MobileNetV2
    # convention the reference keeps.  Scaling it made the 0.75 variants
    # miss Table 3's counts (V3-Large 0.75 -> 4.0M, V3-Small 0.75 -> 2.0M),
    # which are only reproducible with the head fixed at 1280 / 1024.
    base_head = 1280 if cfg.variant == "large" else 1024
    head_ch = _make_divisible(base_head * w) if w > 1.0 else base_head
    return nn.Sequential(*layers), penultimate_ch, head_ch


def _build_classifier_head(
    penultimate_ch: int, head_ch: int, num_classes: int, dropout: float
) -> nn.Sequential:
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(penultimate_ch, head_ch, 1),
        nn.Hardswish(),
        nn.Flatten(),
        nn.Dropout(p=dropout),
        nn.Linear(head_ch, num_classes),
    )


# ---------------------------------------------------------------------------
# MobileNet v3 backbone  (task="base")
# ---------------------------------------------------------------------------


class MobileNetV3(PretrainedModel, BackboneMixin):
    r"""MobileNet v3 feature-extracting backbone (no classification head).

    Implements the NAS-designed topology from Howard et al.,
    "Searching for MobileNetV3", ICCV 2019 (arXiv:1905.02244).  The
    architecture combines MobileNet-v2's inverted-residual block
    with three new ingredients selected by a platform-aware
    architecture search: a lightweight squeeze-and-excitation
    module in selected blocks, the **hard-swish** activation
    (:math:`x \cdot \mathrm{ReLU6}(x+3) / 6`) in the deeper half
    of the network, and a redesigned head that moves the expensive
    1×1 expansion *after* the global average pool.

    Two hand-tuned variants are exposed: ``"large"`` (15 blocks,
    targeting higher-accuracy mobile deployments) and ``"small"``
    (11 blocks, targeting tight latency budgets).  Channel counts
    scale uniformly with ``config.width_mult``.

    Parameters
    ----------
    config : MobileNetV3Config
        Frozen architecture spec.  Use the factory functions
        (:func:`mobilenet_v3_large`, :func:`mobilenet_v3_small`) for
        paper-cited variants.

    Attributes
    ----------
    config : MobileNetV3Config
        Stored copy of the config that built this model.
    features : nn.Sequential
        Stem + 11 (Small) or 15 (Large) inverted-residual SE blocks
        + 1×1 penultimate expansion.  Channel counts scale with
        ``config.width_mult``.
    avgpool : nn.AdaptiveAvgPool2d
        Global average pool collapsing the final feature map to
        ``(B, C, 1, 1)``.
    feature_info : list[FeatureInfo]
        Per-stage descriptor (channels + reduction factor) exposed
        via :class:`BackboneMixin` for downstream FPN / decoder
        modules.

    Notes
    -----
    The hard-swish activation,

    .. math::

        \mathrm{h\text{-}swish}(x) = x \cdot \frac{\mathrm{ReLU6}(x + 3)}{6},

    closely approximates swish/SiLU using only piecewise-linear
    primitives, making it dramatically cheaper to evaluate on
    mobile-friendly fixed-point hardware.  Combined with the
    redesigned tail (moving the 1×1 expansion to operate on a
    single spatial location after global average pooling), the
    overall network achieves :math:`3.2\%` higher ImageNet top-1
    accuracy than MobileNet-v2 at :math:`20\%` lower latency on a
    Pixel-1 CPU.

    Examples
    --------
    Build a MobileNet-v3-Large backbone and run a forward pass:

    >>> import lucid
    >>> from lucid.models.vision.mobilenet_v3 import mobilenet_v3_large
    >>> backbone = mobilenet_v3_large()
    >>> x = lucid.randn(2, 3, 224, 224)
    >>> out = backbone(x)
    >>> out.last_hidden_state.shape
    (2, 960, 7, 7)
    """

    config_class: ClassVar[type[MobileNetV3Config]] = MobileNetV3Config
    base_model_prefix: ClassVar[str] = "mobilenet_v3"

    def __init__(self, config: MobileNetV3Config) -> None:
        super().__init__(config)
        features, num_features, _ = _build_features(config)
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._num_features = num_features

        w = config.width_mult
        specs = _LARGE_SPECS if config.variant == "large" else _SMALL_SPECS
        scaled = _apply_width(specs, w)

        # One entry per *resolution* stage, not per block.  Enumerating every
        # inverted residual gave 11 (small) / 15 (large) entries at repeated
        # reductions, and the last one described the final block (96 / 160 ch)
        # rather than what ``forward_features`` returns — the post-conv map at
        # 576 / 960 channels.  A pyramid consumer reading the last entry got
        # the wrong channel count.
        cumulative = 2  # stem stride=2
        per_reduction: dict[int, int] = {}
        order: list[int] = []
        for _, _, o_ch, _, _, _, s in scaled:
            cumulative *= s
            if cumulative not in per_reduction:
                order.append(cumulative)
            per_reduction[cumulative] = o_ch
        # The trunk ends with a 1x1 conv that widens the last stage, and that
        # is the tensor the backbone hands back.
        per_reduction[cumulative] = num_features
        self._feature_info = [
            FeatureInfo(stage=i + 1, num_channels=per_reduction[r], reduction=r)
            for i, r in enumerate(order)
        ]

        # Reference initialisation: He/MSRA fan-out normal on convs, unit
        # norms.  Lucid's Conv2d default is kaiming_uniform(a=sqrt(5)) — a
        # different distribution *and* gain — so a from-scratch run would
        # otherwise start somewhere the paper's schedule was never tuned for.
        init_cnn_fan_out(self, linear_std=0.01)

    @override
    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    @override
    def forward_features(self, x: Tensor) -> Tensor:
        x = self.features(x)
        # ``feature_info`` describes the pre-pool pyramid, and the mixin
        # documents this as returning the deepest stage's *feature map*.
        # Pooling here collapsed it to 1x1, so the declared reduction was
        # wrong by the input size and no consumer could reach a spatial map.
        # The classifier applies its own pooling.
        return x

    @override
    def forward(self, x: Tensor) -> BaseModelOutput:
        return BaseModelOutput(last_hidden_state=self.forward_features(x))


# ---------------------------------------------------------------------------
# MobileNet v3 for image classification  (task="image-classification")
# ---------------------------------------------------------------------------


class MobileNetV3ForImageClassification(ImageClassificationModel, ClassificationHeadMixin):
    r"""MobileNet v3 image classifier with the redesigned inverted head.

    Combines a :class:`MobileNetV3` feature stack with the paper's
    *redesigned* classification head, which is the central
    efficiency innovation of v3.  Instead of the conventional
    ``GAP → Linear`` head, v3 places a :math:`1 \times 1`
    convolution *after* the global average pool — when the spatial
    map has already collapsed to ``1 × 1`` — saving roughly
    :math:`10\%` of total latency compared to v2's pre-pool
    expansion.  The full head is

    .. math::

        \operatorname{AvgPool} \to \operatorname{Conv}_{1\times1}
            \to \mathrm{h\text{-}swish} \to \operatorname{Flatten}
            \to \operatorname{Dropout} \to \operatorname{Linear}.

    Parameters
    ----------
    config : MobileNetV3Config
        Architecture spec.  Use the ``*_cls`` factory functions
        (:func:`mobilenet_v3_large_cls`, :func:`mobilenet_v3_small_cls`)
        for paper-cited variants.

    Attributes
    ----------
    config : MobileNetV3Config
        Stored copy of the config that built this model.
    features : nn.Sequential
        Same NAS-designed feature stack as on :class:`MobileNetV3`.
    classifier : nn.Sequential
        The redesigned head described above, projecting from the
        penultimate channel count (960 for Large, 576 for Small) to
        ``config.num_classes`` via a 1280-ch (Large) or 1024-ch
        (Small) intermediate width.

    Notes
    -----
    When ``labels`` are supplied to :meth:`forward`, a categorical
    cross-entropy loss is computed against the logits and returned
    alongside them.

    Examples
    --------
    Run inference on a batch of 224×224 RGB images:

    >>> import lucid
    >>> from lucid.models.vision.mobilenet_v3 import mobilenet_v3_large_cls
    >>> model = mobilenet_v3_large_cls()
    >>> x = lucid.randn(4, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (4, 1000)

    Build the smaller variant for latency-critical deployment:

    >>> from lucid.models.vision.mobilenet_v3 import mobilenet_v3_small_cls
    >>> small = mobilenet_v3_small_cls(num_classes=10)
    >>> small.config.variant
    'small'
    """

    config_class: ClassVar[type[MobileNetV3Config]] = MobileNetV3Config
    base_model_prefix: ClassVar[str] = "mobilenet_v3"

    def __init__(self, config: MobileNetV3Config) -> None:
        super().__init__(config)
        features, penultimate_ch, head_ch = _build_features(config)
        self.features = features
        self.classifier = _build_classifier_head(
            penultimate_ch, head_ch, config.num_classes, config.dropout
        )

        # Reference initialisation: He/MSRA fan-out normal on convs, unit
        # norms.  Lucid's Conv2d default is kaiming_uniform(a=sqrt(5)) — a
        # different distribution *and* gain — so a from-scratch run would
        # otherwise start somewhere the paper's schedule was never tuned for.
        init_cnn_fan_out(self, linear_std=0.01)

    @override
    def forward(self, x: Tensor, labels: Tensor | None = None) -> ImageClassificationOutput:
        x = self.features(x)
        logits = self.classifier(x)

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ImageClassificationOutput(logits=logits, loss=loss)
