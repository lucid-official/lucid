"""Inception-ResNet v2 backbone and classifier (Szegedy et al., 2016).

Paper: "Inception-v4, Inception-ResNet and the Impact of Residual Connections
        on Learning"

Architecture overview (299×299 input):
    Stem:        299×299×3  → 35×35×192  (5 named ConvNormAct)
    Mixed_5b:    35×35×192  → 35×35×320  (Inception-A pre-block)
    Block35 ×10  (residual Inception-A): 35×35×320,   scale=0.17
    Mixed_6a:    35×35×320  → 17×17×1088 (Reduction-A)
    Block17 ×20  (residual Inception-B): 17×17×1088,  scale=0.10
    Mixed_7a:    17×17×1088 →  8×8×2080  (Reduction-B)
    Block8  ×9   (residual Inception-C):  8×8×2080,   scale=0.20
    Block8  ×1   (no ReLU):               8×8×2080
    conv2d_7b:    8×8×2080  →  8×8×1536  (final projection)
    Head: AdaptiveAvgPool(1×1) → Dropout → classif(Linear)

State-dict naming matches timm inception_resnet_v2 exactly:
  conv2d_1a.conv/bn.*        ← stem Conv1
  conv2d_2a.conv/bn.*        ← stem Conv2
  conv2d_2b.conv/bn.*        ← stem Conv3
  conv2d_3b.conv/bn.*        ← stem Conv4
  conv2d_4a.conv/bn.*        ← stem Conv5
  mixed_5b.branch0.conv/bn.*
  mixed_5b.branch1.N.conv/bn.*      N = 0,1
  mixed_5b.branch2.N.conv/bn.*      N = 0,1,2
  mixed_5b.branch3.N.conv/bn.*      N = 1 (0 is AvgPool, no params)
  repeat.N.branch0.conv/bn.*
  repeat.N.branch1.M.conv/bn.*      M = 0,1
  repeat.N.branch2.M.conv/bn.*      M = 0,1,2
  repeat.N.conv2d.weight/bias       ← projection (has bias)
  mixed_6a.*
  repeat_1.N.*
  mixed_7a.branch0.M.*     M = 0,1
  mixed_7a.branch1.M.*     M = 0,1
  mixed_7a.branch2.M.*     M = 0,1,2
  repeat_2.N.*
  block8.*                  ← single final block (no relu)
  conv2d_7b.conv/bn.*
  classif.weight/bias       ← Linear head
"""

from dataclasses import dataclass
from typing import ClassVar, TypedDict, final, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._tasks import ImageClassificationModel
from lucid.models._utils._classification import BasicConvBnReLU2d as _ConvBnReLU
from lucid.models._mixins import BackboneMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models.vision.inception_resnet._config import InceptionResNetConfig

# ---------------------------------------------------------------------------
# ConvBnReLU helper — .conv and .bn sub-modules (matches timm ConvNormAct)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# AvgPool with count_include_pad=False (TF-Slim Inception semantics)
# ---------------------------------------------------------------------------


@final
class _AvgPoolExclPad(nn.Module):
    r"""3×3 stride-1 padded average pool that excludes padding cells.

    The canonical TensorFlow-Slim / timm Inception-ResNet v2 averages
    each ``3×3`` window over only its *real* (non-padded) members — i.e.
    ``count_include_pad=False``.  Computed without relying on that flag:
    a standard include-pad average gives ``sum / k^2``; multiplying by
    ``k^2`` recovers the window sum, and dividing by the per-position
    real-cell count (obtained by pooling a ones tensor the same way)
    yields the exclude-pad mean exactly.
    """

    def __init__(self, kernel_size: int, *, stride: int = 1, padding: int = 1) -> None:
        super().__init__()
        self._pool = nn.AvgPool2d(kernel_size, stride=stride, padding=padding)
        self._area = float(kernel_size * kernel_size)

    @override
    def forward(self, x: Tensor) -> Tensor:
        window_sum = self._pool(x) * self._area
        valid = self._pool(lucid.ones_like(x)) * self._area
        return window_sum / valid


# ---------------------------------------------------------------------------
# Mixed_5b: 35×35×192 → 35×35×320  (Inception-A pre-block, no residual)
# ---------------------------------------------------------------------------


@final
class _Mixed5b(nn.Module):
    """Initial Inception-style block: 192 → 96+64+96+64 = 320 channels.

    timm branch naming:
      branch0  : single ConvBnReLU
      branch1  : Sequential([0],[1])
      branch2  : Sequential([0],[1],[2])
      branch3  : Sequential([0]=AvgPool, [1]=ConvBnReLU)
    """

    def __init__(self) -> None:
        super().__init__()
        # branch0: 1×1 → 96
        self.branch0 = _ConvBnReLU(192, 96, 1)
        # branch1: 1×1(48) → 5×5(64)
        self.branch1 = nn.Sequential(_ConvBnReLU(192, 48, 1), _ConvBnReLU(48, 64, 5, padding=2))
        # branch2: 1×1(64) → 3×3(96) → 3×3(96)
        self.branch2 = nn.Sequential(
            _ConvBnReLU(192, 64, 1),
            _ConvBnReLU(64, 96, 3, padding=1),
            _ConvBnReLU(96, 96, 3, padding=1),
        )
        # branch3: AvgPool(3×3 s1 p1) → 1×1(64).  Excludes padding cells
        # from the divisor to match the canonical timm / TF-Slim
        # Inception-ResNet v2 (``count_include_pad=False``).
        self.branch3 = nn.Sequential(
            _AvgPoolExclPad(3, stride=1, padding=1), _ConvBnReLU(192, 64, 1)
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        b0 = self.branch0(x)
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        return lucid.cat([b0, b1, b2, b3], dim=1)  # 96+64+96+64 = 320


# ---------------------------------------------------------------------------
# Block35: Residual Inception-A (×10, 35×35×320, scale=0.17)
# ---------------------------------------------------------------------------


@final
class _Block35(nn.Module):
    """Residual Inception-A: 35×35×320 → 35×35×320.

    timm naming: branch0 (single), branch1 (Sequential[0,1]),
                 branch2 (Sequential[0,1,2]), conv2d (projection with bias).
    """

    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale
        self.branch0 = _ConvBnReLU(320, 32, 1)
        self.branch1 = nn.Sequential(_ConvBnReLU(320, 32, 1), _ConvBnReLU(32, 32, 3, padding=1))
        self.branch2 = nn.Sequential(
            _ConvBnReLU(320, 32, 1),
            _ConvBnReLU(32, 48, 3, padding=1),
            _ConvBnReLU(48, 64, 3, padding=1),
        )
        # timm uses `conv2d` (not `proj`) with bias=True
        # Figures 16/17/19 give the closing linear projections as
        # 384 / 1154 / 2048; TF-Slim uses 320 / 1088 / 2080 and the
        # checkpoint's tensors are shaped accordingly.
        self.conv2d = nn.Conv2d(128, 320, 1)

    @override
    def forward(self, x: Tensor) -> Tensor:
        b0 = self.branch0(x)
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        mixed = lucid.cat([b0, b1, b2], dim=1)
        out = self.conv2d(mixed)
        return F.relu(x + self.scale * out)


# ---------------------------------------------------------------------------
# Mixed_6a (Reduction-A): 35×35×320 → 17×17×1088
# ---------------------------------------------------------------------------


@final
class _Mixed6a(nn.Module):
    """Reduction-A: 35×35×320 → 17×17×1088.

    branch0: 3×3 s2              → 384
    branch1: Sequential[0,1,2]   → 384
    branch2: MaxPool 3×3 s2      → 320 (pass-through, no params)
    total: 384 + 384 + 320 = 1088
    """

    def __init__(self) -> None:
        super().__init__()
        self.branch0 = _ConvBnReLU(320, 384, 3, stride=2)
        self.branch1 = nn.Sequential(
            _ConvBnReLU(320, 256, 1),
            _ConvBnReLU(256, 256, 3, padding=1),
            _ConvBnReLU(256, 384, 3, stride=2),
        )
        self.branch2 = nn.MaxPool2d(3, stride=2)

    @override
    def forward(self, x: Tensor) -> Tensor:
        b0 = self.branch0(x)
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        return lucid.cat([b0, b1, b2], dim=1)  # 384+384+320 = 1088


# ---------------------------------------------------------------------------
# Block17: Residual Inception-B (×20, 17×17×1088, scale=0.10)
# ---------------------------------------------------------------------------


@final
class _Block17(nn.Module):
    """Residual Inception-B: 17×17×1088 → 17×17×1088.

    timm: branch0 (single), branch1 (Sequential[0,1,2]), conv2d.
    """

    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale
        self.branch0 = _ConvBnReLU(1088, 192, 1)
        self.branch1 = nn.Sequential(
            _ConvBnReLU(1088, 128, 1),
            _ConvBnReLU(128, 160, (1, 7), padding=(0, 3)),
            _ConvBnReLU(160, 192, (7, 1), padding=(3, 0)),
        )
        self.conv2d = nn.Conv2d(384, 1088, 1)

    @override
    def forward(self, x: Tensor) -> Tensor:
        b0 = self.branch0(x)
        b1 = self.branch1(x)
        mixed = lucid.cat([b0, b1], dim=1)
        out = self.conv2d(mixed)
        return F.relu(x + self.scale * out)


# ---------------------------------------------------------------------------
# Mixed_7a (Reduction-B): 17×17×1088 → 8×8×2080
# ---------------------------------------------------------------------------


@final
class _Mixed7a(nn.Module):
    """Reduction-B: 17×17×1088 → 8×8×2080.

    branch0: Sequential[0,1]     → 384
    branch1: Sequential[0,1]     → 288
    branch2: Sequential[0,1,2]   → 320
    branch3: MaxPool 3×3 s2      → 1088 (pass-through)
    total: 384 + 288 + 320 + 1088 = 2080
    """

    def __init__(self) -> None:
        super().__init__()
        self.branch0 = nn.Sequential(_ConvBnReLU(1088, 256, 1), _ConvBnReLU(256, 384, 3, stride=2))
        self.branch1 = nn.Sequential(_ConvBnReLU(1088, 256, 1), _ConvBnReLU(256, 288, 3, stride=2))
        self.branch2 = nn.Sequential(
            _ConvBnReLU(1088, 256, 1),
            _ConvBnReLU(256, 288, 3, padding=1),
            _ConvBnReLU(288, 320, 3, stride=2),
        )
        self.branch3 = nn.MaxPool2d(3, stride=2)

    @override
    def forward(self, x: Tensor) -> Tensor:
        b0 = self.branch0(x)
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        return lucid.cat([b0, b1, b2, b3], dim=1)  # 384+288+320+1088 = 2080


# ---------------------------------------------------------------------------
# Block8: Residual Inception-C (×9 + ×1 no_relu, 8×8×2080, scale=0.20)
# ---------------------------------------------------------------------------


@final
class _Block8(nn.Module):
    """Residual Inception-C: 8×8×2080 → 8×8×2080.

    timm: branch0 (single), branch1 (Sequential[0,1,2]), conv2d.
    The `no_relu` variant (final block) omits the activation on the output sum.
    """

    def __init__(self, scale: float, no_relu: bool = False) -> None:
        super().__init__()
        self.scale = scale
        self.no_relu = no_relu
        self.branch0 = _ConvBnReLU(2080, 192, 1)
        self.branch1 = nn.Sequential(
            _ConvBnReLU(2080, 192, 1),
            _ConvBnReLU(192, 224, (1, 3), padding=(0, 1)),
            _ConvBnReLU(224, 256, (3, 1), padding=(1, 0)),
        )
        self.conv2d = nn.Conv2d(448, 2080, 1)

    @override
    def forward(self, x: Tensor) -> Tensor:
        b0 = self.branch0(x)
        b1 = self.branch1(x)
        mixed = lucid.cat([b0, b1], dim=1)
        out = self.conv2d(mixed)
        result = x + self.scale * out
        if self.no_relu:
            return result
        return F.relu(result)


# ---------------------------------------------------------------------------
# Inception-ResNet v2 output dataclass
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class InceptionResNetOutput(ImageClassificationOutput):
    r"""Structured forward output for :class:`InceptionResNetV2ForImageClassification`.

    Inception-ResNet v2 does not use auxiliary classifiers (unlike
    Inception v3 / GoogLeNet) — the residual shortcuts already mitigate
    the vanishing-gradient problem that auxiliary heads were originally
    introduced to address — so this dataclass only carries the main
    logits and an optional loss tensor.

    Parameters
    ----------
    logits : Tensor
        Main classifier output of shape ``(B, num_classes)``.
    loss : Tensor or None, optional, default=None
        Cross-entropy loss when labels were passed to :meth:`forward`;
        ``None`` otherwise.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.inception_resnet import inception_resnet_v2_cls
    >>> model = inception_resnet_v2_cls().eval()
    >>> x = lucid.randn(1, 3, 299, 299)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    >>> out.loss is None
    True
    """


# ---------------------------------------------------------------------------
# Shared body builder (all stages as named attributes)
# ---------------------------------------------------------------------------


class _BodyModules(TypedDict):
    conv2d_1a: _ConvBnReLU
    conv2d_2a: _ConvBnReLU
    conv2d_2b: _ConvBnReLU
    conv2d_3b: _ConvBnReLU
    conv2d_4a: _ConvBnReLU
    mixed_5b: _Mixed5b
    repeat: nn.Sequential
    mixed_6a: _Mixed6a
    repeat_1: nn.Sequential
    mixed_7a: _Mixed7a
    repeat_2: nn.Sequential
    block8: _Block8
    conv2d_7b: _ConvBnReLU


def _build_body(config: InceptionResNetConfig) -> _BodyModules:
    """Return the ordered modules that make up the Inception-ResNet v2 body."""
    s_a = config.scale_a
    s_b = config.scale_b
    s_c = config.scale_c

    return {
        # Stem — 5 flat named ConvBnReLU (no MaxPool in state dict)
        "conv2d_1a": _ConvBnReLU(config.in_channels, 32, 3, stride=2),
        "conv2d_2a": _ConvBnReLU(32, 32, 3),
        "conv2d_2b": _ConvBnReLU(32, 64, 3, padding=1),
        # MaxPool 3×3 s2 — no params, stored separately in model
        "conv2d_3b": _ConvBnReLU(64, 80, 1),
        "conv2d_4a": _ConvBnReLU(80, 192, 3),
        # MaxPool 3×3 s2 — no params, stored separately in model
        # Inception body
        "mixed_5b": _Mixed5b(),
        # Figure 15 of the paper shows 5 / 10 / 5 residual blocks; TF-Slim --
        # and therefore the checkpoint whose tensors these modules must match
        # -- builds 10 / 20 / 9+1.  The counts are fixed by the weights.
        "repeat": nn.Sequential(*[_Block35(scale=s_a) for _ in range(10)]),
        "mixed_6a": _Mixed6a(),
        "repeat_1": nn.Sequential(*[_Block17(scale=s_b) for _ in range(20)]),
        "mixed_7a": _Mixed7a(),
        "repeat_2": nn.Sequential(*[_Block8(scale=s_c) for _ in range(9)]),
        "block8": _Block8(scale=1.0, no_relu=True),
        # A fourth divergence from the paper: Figure 15 pools the 8x8
        # grid directly into the classifier, with no 1x1 projection
        # between.  TF-Slim inserts this 2080 -> 1536 conv, and the
        # checkpoint carries its weights, so it is not optional here.
        "conv2d_7b": _ConvBnReLU(2080, 1536, 1),
    }


# ---------------------------------------------------------------------------
# InceptionResNetV2 backbone (task="base")
# ---------------------------------------------------------------------------


class InceptionResNetV2(PretrainedModel, BackboneMixin):
    r"""Inception-ResNet v2 feature-extracting backbone.

    Implements the residual-Inception topology from Szegedy et al.,
    "Inception-v4, Inception-ResNet and the Impact of Residual
    Connections on Learning", AAAI 2017.  The body consists of three
    residual Inception block families — :class:`_Block35` (×10),
    :class:`_Block17` (×20), and :class:`_Block8` (×9 + 1 no-ReLU
    block) — separated by Reduction-A (:class:`_Mixed6a`) and
    Reduction-B (:class:`_Mixed7a`) blocks, with a leading Mixed_5b
    pre-block (:class:`_Mixed5b`) and a final :math:`1\times1`
    projection (``conv2d_7b``) to 1536 channels.  Designed for
    :math:`299\times299` RGB inputs.  State-dict naming matches the
    canonical timm / TensorFlow-Slim implementation so weight transfer
    is straightforward.

    Parameters
    ----------
    config : InceptionResNetConfig
        Frozen architecture spec.  Use :func:`inception_resnet_v2` for
        the paper-cited configuration with the recommended residual
        scale factors (0.17 / 0.10 / 0.20).

    Attributes
    ----------
    config : InceptionResNetConfig
        Stored copy of the config that built this model.
    conv2d_1a, conv2d_2a, conv2d_2b, conv2d_3b, conv2d_4a : nn.Module
        Stem ConvBnReLU layers (TensorFlow-Slim naming preserved).
    mixed_5b : _Mixed5b
        Pre-Block35 Inception block (192 → 320 channels) at
        :math:`35\times35`.
    repeat : nn.Sequential
        Stack of 10 :class:`_Block35` residual Inception-A blocks.
    mixed_6a : _Mixed6a
        Reduction-A block, :math:`35\times35 \to 17\times17`, widening
        320 → 1088 channels.
    repeat_1 : nn.Sequential
        Stack of 20 :class:`_Block17` residual Inception-B blocks.
    mixed_7a : _Mixed7a
        Reduction-B block, :math:`17\times17 \to 8\times8`, widening
        1088 → 2080 channels.
    repeat_2 : nn.Sequential
        Stack of 9 :class:`_Block8` residual Inception-C blocks.
    block8 : _Block8
        Final no-ReLU residual block (scale 1.0).
    conv2d_7b : _ConvBnReLU
        :math:`1\times1` projection 2080 → 1536.
    avgpool : nn.AdaptiveAvgPool2d
        Global average pool to :math:`1\times1`.
    feature_info : list[FeatureInfo]
        Per-stage descriptor (channels + reduction factor) exposed via
        :class:`BackboneMixin`.

    Notes
    -----
    Each residual Inception block computes

    .. math::

        y = \sigma\!\left(x + \alpha \cdot \mathcal{F}_{\text{inception}}(x)\right),

    with :math:`\alpha` a small fixed scale factor (0.10–0.20).  The
    residual-scale trick was introduced specifically for
    Inception-ResNet because unscaled residuals caused training to
    diverge on networks with many filters per Inception module.
    Approximately 55.8 M parameters; achieves a top-5 ImageNet
    validation error of 3.08% in the original paper.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.inception_resnet import inception_resnet_v2
    >>> backbone = inception_resnet_v2()
    >>> x = lucid.randn(1, 3, 299, 299)
    >>> out = backbone(x)
    >>> out.last_hidden_state.shape   # (B, 1536, 1, 1)
    (1, 1536, 8, 8)
    """

    config_class: ClassVar[type[InceptionResNetConfig]] = InceptionResNetConfig
    base_model_prefix: ClassVar[str] = "inception_resnet_v2"

    def __init__(self, config: InceptionResNetConfig) -> None:
        super().__init__(config)
        body = _build_body(config)
        # Stem conv layers
        self.conv2d_1a = body["conv2d_1a"]
        self.conv2d_2a = body["conv2d_2a"]
        self.conv2d_2b = body["conv2d_2b"]
        self._pool1 = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = body["conv2d_3b"]
        self.conv2d_4a = body["conv2d_4a"]
        self._pool2 = nn.MaxPool2d(3, stride=2)
        # Inception body
        self.mixed_5b = body["mixed_5b"]
        self.repeat = body["repeat"]
        self.mixed_6a = body["mixed_6a"]
        self.repeat_1 = body["repeat_1"]
        self.mixed_7a = body["mixed_7a"]
        self.repeat_2 = body["repeat_2"]
        self.block8 = body["block8"]
        self.conv2d_7b = body["conv2d_7b"]

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self._feature_info = [
            FeatureInfo(stage=1, num_channels=320, reduction=8),
            FeatureInfo(stage=2, num_channels=1088, reduction=16),
            FeatureInfo(stage=3, num_channels=1536, reduction=32),
        ]

    def _forward_stem(self, x: Tensor) -> Tensor:
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self._pool1(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        return self._pool2(x)

    @override
    def forward_features(self, x: Tensor) -> Tensor:
        x = self._forward_stem(x)
        x = self.mixed_5b(x)
        x = self.repeat(x)
        x = self.mixed_6a(x)
        x = self.repeat_1(x)
        x = self.mixed_7a(x)
        x = self.repeat_2(x)
        x = self.block8(x)
        return self.conv2d_7b(x)
        # ``feature_info`` describes the pre-pool pyramid, and the mixin
        # documents this as returning the deepest stage's *feature map*.
        # Pooling here collapsed it to 1x1, so the declared reduction was
        # wrong by the input size and no consumer could reach a spatial map.
        # The classifier applies its own pooling.

    @override
    def forward(self, x: Tensor) -> BaseModelOutput:
        return BaseModelOutput(last_hidden_state=self.forward_features(x))


# ---------------------------------------------------------------------------
# InceptionResNetV2 for image classification (task="image-classification")
# ---------------------------------------------------------------------------


class InceptionResNetV2ForImageClassification(ImageClassificationModel):
    r"""Inception-ResNet v2 image classifier (backbone + GAP + dropout + linear).

    Combines an :class:`InceptionResNetV2` backbone with a
    global-average-pool, optional :class:`~lucid.nn.Dropout`
    (``p=config.dropout``, default 0.2), and a single
    :class:`~lucid.nn.Linear` projection producing
    ``config.num_classes`` logits.  No auxiliary classifier — the
    residual shortcuts already provide effective gradient flow
    throughout the network.  The classifier attribute is named
    ``classif`` (not ``classifier``) to match the canonical timm /
    TensorFlow-Slim state-dict key.

    Parameters
    ----------
    config : InceptionResNetConfig
        Architecture spec.  Use :func:`inception_resnet_v2_cls` for the
        paper-cited configuration.

    Attributes
    ----------
    config : InceptionResNetConfig
        Stored copy of the config that built this model.
    conv2d_1a … conv2d_7b, mixed_5b, repeat, mixed_6a, repeat_1,
        mixed_7a, repeat_2, block8
        Same backbone components as on :class:`InceptionResNetV2`.
    avgpool : nn.AdaptiveAvgPool2d
        Final global average pool to :math:`1\times1`.
    dropout : nn.Module
        :class:`~lucid.nn.Dropout` when ``config.dropout > 0`` (default
        0.2), otherwise :class:`~lucid.nn.Identity`.
    classif : nn.Linear
        Final classifier projecting 1536 → ``num_classes`` — named
        ``classif`` for state-dict compatibility.

    Notes
    -----
    From Szegedy et al., "Inception-v4, Inception-ResNet and the
    Impact of Residual Connections on Learning", AAAI 2017.  Loss is
    the standard categorical cross-entropy

    .. math::

        \mathcal{L} = -\frac{1}{N} \sum_{n=1}^{N}
            \log \operatorname{softmax}(\text{logits}_n)_{\,y_n},

    computed only when ``labels`` is not ``None``.  Approximately
    55.8 M parameters; top-5 ImageNet validation error 3.08%.  The key
    empirical message of the paper is that residual connections do not
    raise the final accuracy ceiling beyond a comparably-sized
    non-residual Inception v4, but they *dramatically* accelerate
    training convergence.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.inception_resnet import inception_resnet_v2_cls
    >>> model = inception_resnet_v2_cls()
    >>> x = lucid.randn(2, 3, 299, 299)
    >>> out = model(x)
    >>> out.logits.shape
    (2, 1000)
    """

    config_class: ClassVar[type[InceptionResNetConfig]] = InceptionResNetConfig
    base_model_prefix: ClassVar[str] = "inception_resnet_v2"

    def __init__(self, config: InceptionResNetConfig) -> None:
        super().__init__(config)
        body = _build_body(config)
        # Stem
        self.conv2d_1a = body["conv2d_1a"]
        self.conv2d_2a = body["conv2d_2a"]
        self.conv2d_2b = body["conv2d_2b"]
        self._pool1 = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = body["conv2d_3b"]
        self.conv2d_4a = body["conv2d_4a"]
        self._pool2 = nn.MaxPool2d(3, stride=2)
        # Inception body
        self.mixed_5b = body["mixed_5b"]
        self.repeat = body["repeat"]
        self.mixed_6a = body["mixed_6a"]
        self.repeat_1 = body["repeat_1"]
        self.mixed_7a = body["mixed_7a"]
        self.repeat_2 = body["repeat_2"]
        self.block8 = body["block8"]
        self.conv2d_7b = body["conv2d_7b"]

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if config.dropout > 0.0:
            self.dropout: nn.Dropout | nn.Identity = nn.Dropout(config.dropout)
        else:
            self.dropout = nn.Identity()
        # Named `classif` to match timm
        self.classif = nn.Linear(1536, config.num_classes)

    def _forward_stem(self, x: Tensor) -> Tensor:
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self._pool1(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        return self._pool2(x)

    def reset_classifier(self, num_classes: int) -> None:
        """Replace the classification head with a freshly initialised one.

        The head keeps the reference checkpoint's ``classif`` attribute name,
        so this cannot come from ``ClassificationHeadMixin`` (which looks for
        ``self.classifier``).  The class docstring documented the method
        without anything implementing it.
        """
        self.classif = nn.Linear(self.classif.in_features, num_classes)

    @override
    def forward(self, x: Tensor, labels: Tensor | None = None) -> InceptionResNetOutput:
        x = self._forward_stem(x)
        x = self.mixed_5b(x)
        x = self.repeat(x)
        x = self.mixed_6a(x)
        x = self.repeat_1(x)
        x = self.mixed_7a(x)
        x = self.repeat_2(x)
        x = self.block8(x)
        x = self.conv2d_7b(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        logits = self.classif(x)

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return InceptionResNetOutput(logits=logits, loss=loss)
