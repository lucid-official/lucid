"""Inception-v4 backbone and classifier (Szegedy et al., 2017).

Paper: "Inception-v4, Inception-ResNet and the Impact of Residual Connections
        on Learning" (arXiv:1602.07261), Figures 3–9.

Architecture overview (299×299 input):
    Stem (Figure 3):       299×299×3  → 35×35×384
        features.0–2       3 ConvBnReLU (3×3/2 V, 3×3 V, 3×3 same)
        features.3         Mixed_3a: MaxPool 3×3/2 ‖ Conv 3×3/2 (96)   → 160
        features.4         Mixed_4a: 1×1→3×3 ‖ 1×1→1×7→7×1→3×3 (96 each) → 192
        features.5         Mixed_5a: Conv 3×3/2 (192) ‖ MaxPool 3×3/2    → 384
    features.6–9           4× Inception-A          35×35×384
    features.10            Reduction-A             35×35×384  → 17×17×1024
    features.11–17         7× Inception-B          17×17×1024
    features.18            Reduction-B             17×17×1024 →  8×8×1536
    features.19–21         3× Inception-C           8×8×1536
    Head: AdaptiveAvgPool(1×1) → Dropout(0.2) → last_linear(Linear)

State-dict naming mirrors the TF-Slim port used by the reference model zoo
(``features.N`` for the 22-module trunk, ``last_linear`` for the head), so
published weights transfer key-for-key with no remapping.
"""

from dataclasses import dataclass
from typing import ClassVar, final, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models._tasks import ImageClassificationModel
from lucid.models.vision.inception_v4._config import InceptionV4Config

# BatchNorm epsilon used by every conv block.  TF-Slim's default is 1e-3,
# and the released checkpoint's running statistics were accumulated under
# it; the usual 1e-5 would shift every normalised activation slightly.
_BN_EPS = 1e-3

# Width of the trunk's final feature map (Inception-C output).
_NUM_FEATURES = 1536

# ---------------------------------------------------------------------------
# Conv → BN → ReLU helper (``.conv`` / ``.bn`` sub-modules)
# ---------------------------------------------------------------------------


@final
class _ConvBnReLU(nn.Module):
    """Bias-free Conv2d → BatchNorm2d(eps=1e-3) → ReLU.

    The sub-modules are named ``.conv`` and ``.bn`` so the parameter keys
    match the reference zoo's ``ConvNormAct`` layout.  Padding defaults to
    zero — the paper's "V" (valid) convolutions — and is passed explicitly
    wherever a figure marks a convolution as same-padded.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        *,
        stride: int = 1,
        padding: int | tuple[int, int] = 0,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=_BN_EPS)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.bn(self.conv(x)))


def _avg_pool_branch(in_channels: int, out_channels: int) -> nn.Sequential:
    """3×3 stride-1 average pool (padding excluded) followed by a 1×1 conv.

    The pool averages each window over its *real* cells only
    (``count_include_pad=False``), as TF-Slim's ``SAME`` average pooling
    does; including the zero padding would darken every border position.
    """
    return nn.Sequential(
        nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
        _ConvBnReLU(in_channels, out_channels, 1),
    )


# ---------------------------------------------------------------------------
# Stem (Figure 3)
# ---------------------------------------------------------------------------


@final
class _Mixed3a(nn.Module):
    """First stem junction: 147×147×64 → 73×73×160.

    ``MaxPool 3×3/2 V`` (64 channels) concatenated with ``Conv 3×3/2 V``
    (96 channels).
    """

    def __init__(self) -> None:
        super().__init__()
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.conv = _ConvBnReLU(64, 96, 3, stride=2)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x0 = self.maxpool(x)
        x1 = self.conv(x)
        return lucid.cat([x0, x1], dim=1)  # 64 + 96 = 160


@final
class _Mixed4a(nn.Module):
    """Second stem junction: 73×73×160 → 71×71×192.

    branch0: 1×1 (64) → 3×3 V (96)
    branch1: 1×1 (64) → 1×7 (64) → 7×1 (64) → 3×3 V (96)
    """

    def __init__(self) -> None:
        super().__init__()
        self.branch0 = nn.Sequential(_ConvBnReLU(160, 64, 1), _ConvBnReLU(64, 96, 3))
        self.branch1 = nn.Sequential(
            _ConvBnReLU(160, 64, 1),
            _ConvBnReLU(64, 64, (1, 7), padding=(0, 3)),
            _ConvBnReLU(64, 64, (7, 1), padding=(3, 0)),
            _ConvBnReLU(64, 96, 3),
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        return lucid.cat([x0, x1], dim=1)  # 96 + 96 = 192


@final
class _Mixed5a(nn.Module):
    """Third stem junction: 71×71×192 → 35×35×384.

    ``Conv 3×3/2 V`` (192 channels) concatenated with ``MaxPool 3×3/2 V``
    (192 channels).
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = _ConvBnReLU(192, 192, 3, stride=2)
        self.maxpool = nn.MaxPool2d(3, stride=2)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x0 = self.conv(x)
        x1 = self.maxpool(x)
        return lucid.cat([x0, x1], dim=1)  # 192 + 192 = 384


# ---------------------------------------------------------------------------
# Inception-A (Figure 4) — 35×35×384, ×4
# ---------------------------------------------------------------------------


@final
class _InceptionA(nn.Module):
    """Inception-A: 35×35×384 → 35×35×384.

    branch0: 1×1 (96)
    branch1: 1×1 (64) → 3×3 (96)
    branch2: 1×1 (64) → 3×3 (96) → 3×3 (96)
    branch3: AvgPool 3×3 → 1×1 (96)
    """

    def __init__(self) -> None:
        super().__init__()
        self.branch0 = _ConvBnReLU(384, 96, 1)
        self.branch1 = nn.Sequential(_ConvBnReLU(384, 64, 1), _ConvBnReLU(64, 96, 3, padding=1))
        self.branch2 = nn.Sequential(
            _ConvBnReLU(384, 64, 1),
            _ConvBnReLU(64, 96, 3, padding=1),
            _ConvBnReLU(96, 96, 3, padding=1),
        )
        self.branch3 = _avg_pool_branch(384, 96)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        return lucid.cat([x0, x1, x2, x3], dim=1)  # 4 × 96 = 384


# ---------------------------------------------------------------------------
# Reduction-A (Figure 7, Table 1: k=192, l=224, m=256, n=384)
# ---------------------------------------------------------------------------


@final
class _ReductionA(nn.Module):
    """Reduction-A: 35×35×384 → 17×17×1024.

    branch0: 3×3/2 V (n=384)
    branch1: 1×1 (k=192) → 3×3 (l=224) → 3×3/2 V (m=256)
    branch2: MaxPool 3×3/2 V (384, pass-through)
    """

    def __init__(self) -> None:
        super().__init__()
        self.branch0 = _ConvBnReLU(384, 384, 3, stride=2)
        self.branch1 = nn.Sequential(
            _ConvBnReLU(384, 192, 1),
            _ConvBnReLU(192, 224, 3, padding=1),
            _ConvBnReLU(224, 256, 3, stride=2),
        )
        self.branch2 = nn.MaxPool2d(3, stride=2)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        return lucid.cat([x0, x1, x2], dim=1)  # 384 + 256 + 384 = 1024


# ---------------------------------------------------------------------------
# Inception-B (Figure 5) — 17×17×1024, ×7
# ---------------------------------------------------------------------------


@final
class _InceptionB(nn.Module):
    """Inception-B: 17×17×1024 → 17×17×1024.

    branch0: 1×1 (384)
    branch1: 1×1 (192) → 1×7 (224) → 7×1 (256)
    branch2: 1×1 (192) → 7×1 (192) → 1×7 (224) → 7×1 (224) → 1×7 (256)
    branch3: AvgPool 3×3 → 1×1 (128)

    ``branch2`` alternates its factorised pair starting with ``7×1``, as
    TF-Slim builds it; the checkpoint's kernel shapes fix that order.
    """

    def __init__(self) -> None:
        super().__init__()
        self.branch0 = _ConvBnReLU(1024, 384, 1)
        self.branch1 = nn.Sequential(
            _ConvBnReLU(1024, 192, 1),
            _ConvBnReLU(192, 224, (1, 7), padding=(0, 3)),
            _ConvBnReLU(224, 256, (7, 1), padding=(3, 0)),
        )
        self.branch2 = nn.Sequential(
            _ConvBnReLU(1024, 192, 1),
            _ConvBnReLU(192, 192, (7, 1), padding=(3, 0)),
            _ConvBnReLU(192, 224, (1, 7), padding=(0, 3)),
            _ConvBnReLU(224, 224, (7, 1), padding=(3, 0)),
            _ConvBnReLU(224, 256, (1, 7), padding=(0, 3)),
        )
        self.branch3 = _avg_pool_branch(1024, 128)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        return lucid.cat([x0, x1, x2, x3], dim=1)  # 384 + 256 + 256 + 128


# ---------------------------------------------------------------------------
# Reduction-B (Figure 8)
# ---------------------------------------------------------------------------


@final
class _ReductionB(nn.Module):
    """Reduction-B: 17×17×1024 → 8×8×1536.

    branch0: 1×1 (192) → 3×3/2 V (192)
    branch1: 1×1 (256) → 1×7 (256) → 7×1 (320) → 3×3/2 V (320)
    branch2: MaxPool 3×3/2 V (1024, pass-through)
    """

    def __init__(self) -> None:
        super().__init__()
        self.branch0 = nn.Sequential(_ConvBnReLU(1024, 192, 1), _ConvBnReLU(192, 192, 3, stride=2))
        self.branch1 = nn.Sequential(
            _ConvBnReLU(1024, 256, 1),
            _ConvBnReLU(256, 256, (1, 7), padding=(0, 3)),
            _ConvBnReLU(256, 320, (7, 1), padding=(3, 0)),
            _ConvBnReLU(320, 320, 3, stride=2),
        )
        self.branch2 = nn.MaxPool2d(3, stride=2)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        return lucid.cat([x0, x1, x2], dim=1)  # 192 + 320 + 1024 = 1536


# ---------------------------------------------------------------------------
# Inception-C (Figure 6) — 8×8×1536, ×3
# ---------------------------------------------------------------------------


@final
class _InceptionC(nn.Module):
    """Inception-C: 8×8×1536 → 8×8×1536.

    branch0:  1×1 (256)
    branch1:  1×1 (384) → {1×3 (256) ‖ 3×1 (256)}
    branch2:  1×1 (384) → 3×1 (448) → 1×3 (512) → {1×3 (256) ‖ 3×1 (256)}
    branch3:  AvgPool 3×3 → 1×1 (256)

    The two split points fan one tensor out into a ``1×3`` and a ``3×1``
    convolution whose outputs are concatenated, so the modules are flat
    named attributes (``branch1_1a`` / ``branch1_1b`` …) rather than
    ``Sequential`` chains.
    """

    def __init__(self) -> None:
        super().__init__()
        self.branch0 = _ConvBnReLU(1536, 256, 1)

        self.branch1_0 = _ConvBnReLU(1536, 384, 1)
        self.branch1_1a = _ConvBnReLU(384, 256, (1, 3), padding=(0, 1))
        self.branch1_1b = _ConvBnReLU(384, 256, (3, 1), padding=(1, 0))

        self.branch2_0 = _ConvBnReLU(1536, 384, 1)
        self.branch2_1 = _ConvBnReLU(384, 448, (3, 1), padding=(1, 0))
        self.branch2_2 = _ConvBnReLU(448, 512, (1, 3), padding=(0, 1))
        self.branch2_3a = _ConvBnReLU(512, 256, (1, 3), padding=(0, 1))
        self.branch2_3b = _ConvBnReLU(512, 256, (3, 1), padding=(1, 0))

        self.branch3 = _avg_pool_branch(1536, 256)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x0 = self.branch0(x)

        x1_0 = self.branch1_0(x)
        x1 = lucid.cat([self.branch1_1a(x1_0), self.branch1_1b(x1_0)], dim=1)

        x2_0 = self.branch2_0(x)
        x2_2 = self.branch2_2(self.branch2_1(x2_0))
        x2 = lucid.cat([self.branch2_3a(x2_2), self.branch2_3b(x2_2)], dim=1)

        x3 = self.branch3(x)
        return lucid.cat([x0, x1, x2, x3], dim=1)  # 256 + 512 + 512 + 256


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class InceptionV4Output(ImageClassificationOutput):
    r"""Structured forward output for :class:`InceptionV4ForImageClassification`.

    Inception-v4 trains with a single softmax head — the paper drops the
    auxiliary classifiers of GoogLeNet / Inception v3 — so this dataclass
    carries only the logits and an optional loss.

    Parameters
    ----------
    logits : Tensor
        Classifier output of shape ``(B, num_classes)``.
    loss : Tensor or None, optional, default=None
        Mean cross-entropy against ``labels`` when they were passed to
        :meth:`InceptionV4ForImageClassification.forward`; ``None``
        otherwise.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.inception_v4 import inception_v4_cls
    >>> model = inception_v4_cls().eval()
    >>> out = model(lucid.randn(1, 3, 299, 299))
    >>> out.logits.shape
    (1, 1000)
    >>> out.loss is None
    True
    """


# ---------------------------------------------------------------------------
# Trunk builder
# ---------------------------------------------------------------------------


def _build_features(config: InceptionV4Config) -> nn.Sequential:
    """Return the 22-module Inception-v4 trunk (stem through Inception-C)."""
    modules: list[nn.Module] = [
        _ConvBnReLU(config.in_channels, 32, 3, stride=2),  # 149×149×32
        _ConvBnReLU(32, 32, 3),  # 147×147×32
        _ConvBnReLU(32, 64, 3, padding=1),  # 147×147×64
        _Mixed3a(),  # 73×73×160
        _Mixed4a(),  # 71×71×192
        _Mixed5a(),  # 35×35×384
    ]
    modules += [_InceptionA() for _ in range(4)]
    modules.append(_ReductionA())
    modules += [_InceptionB() for _ in range(7)]
    modules.append(_ReductionB())
    modules += [_InceptionC() for _ in range(3)]
    return nn.Sequential(*modules)


# ---------------------------------------------------------------------------
# InceptionV4 backbone (task="base")
# ---------------------------------------------------------------------------


class InceptionV4(PretrainedModel, BackboneMixin):
    r"""Inception-v4 feature-extracting backbone.

    Implements the non-residual network of Szegedy et al., "Inception-v4,
    Inception-ResNet and the Impact of Residual Connections on Learning",
    AAAI 2017 (Figure 9): the multi-branch stem of Figure 3, four
    Inception-A modules, Reduction-A, seven Inception-B modules,
    Reduction-B, and three Inception-C modules.  Designed for
    :math:`299\times299` RGB inputs, which it maps to an
    :math:`8\times8\times1536` feature map.

    Parameters
    ----------
    config : InceptionV4Config
        Frozen architecture spec.  Use :func:`inception_v4` for the
        paper configuration.

    Attributes
    ----------
    config : InceptionV4Config
        Stored copy of the config that built this model.
    features : nn.Sequential
        The 22-module trunk: ``features.0``–``features.5`` are the stem,
        ``6``–``9`` Inception-A, ``10`` Reduction-A, ``11``–``17``
        Inception-B, ``18`` Reduction-B and ``19``–``21`` Inception-C.
    feature_info : list[FeatureInfo]
        One entry per stage boundary — channels 64 / 160 / 384 / 1024 /
        1536 at reductions 2 / 4 / 8 / 16 / 32.

    Notes
    -----
    Every module is a filter concatenation of parallel branches,

    .. math::

        y = \operatorname{concat}\big(\mathcal{B}_1(x), \dots,
            \mathcal{B}_k(x)\big),

    with no residual shortcut anywhere — that is what separates
    Inception-v4 from its sibling Inception-ResNet.  The trunk holds
    41.1 M parameters; the paper reports 20.0% top-1 / 5.0% top-5
    single-crop error on the ImageNet validation set for the full
    classifier.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.inception_v4 import inception_v4
    >>> backbone = inception_v4().eval()
    >>> x = lucid.randn(1, 3, 299, 299)
    >>> backbone.forward_features(x).shape
    (1, 1536, 8, 8)
    >>> [f.num_channels for f in backbone.feature_info]
    [64, 160, 384, 1024, 1536]
    """

    config_class: ClassVar[type[InceptionV4Config]] = InceptionV4Config
    base_model_prefix: ClassVar[str] = "inception_v4"

    def __init__(self, config: InceptionV4Config) -> None:
        super().__init__(config)
        self.features = _build_features(config)
        self._feature_info = [
            FeatureInfo(stage=0, num_channels=64, reduction=2),
            FeatureInfo(stage=1, num_channels=160, reduction=4),
            FeatureInfo(stage=2, num_channels=384, reduction=8),
            FeatureInfo(stage=3, num_channels=1024, reduction=16),
            FeatureInfo(stage=4, num_channels=_NUM_FEATURES, reduction=32),
        ]

    @override
    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    @override
    def forward_features(self, x: Tensor) -> Tensor:
        r"""Run the trunk and return the final :math:`8\times8\times1536` map.

        Parameters
        ----------
        x : Tensor
            Input images of shape ``(B, in_channels, H, W)``.

        Returns
        -------
        Tensor
            Unpooled Inception-C output of shape ``(B, 1536, H', W')``
            (``8×8`` at the native ``299×299``).
        """
        return self.features(x)

    @override
    def forward(self, x: Tensor) -> BaseModelOutput:
        r"""Return the trunk's feature map as ``last_hidden_state``.

        Parameters
        ----------
        x : Tensor
            Input images of shape ``(B, in_channels, H, W)``.

        Returns
        -------
        BaseModelOutput
            ``last_hidden_state`` of shape ``(B, 1536, H', W')``.
        """
        return BaseModelOutput(last_hidden_state=self.forward_features(x))


# ---------------------------------------------------------------------------
# InceptionV4 for image classification (task="image-classification")
# ---------------------------------------------------------------------------


class InceptionV4ForImageClassification(ImageClassificationModel):
    r"""Inception-v4 image classifier (trunk + GAP + dropout + linear).

    Adds the paper's head to the :class:`InceptionV4` trunk: global
    average pooling over the :math:`8\times8` grid, dropout with keep
    probability 0.8 (``config.dropout = 0.2``), and a single
    :class:`~lucid.nn.Linear` layer producing ``config.num_classes``
    logits.  The trunk is held directly as ``features`` and the head as
    ``last_linear`` so the state dict matches the released checkpoint's
    keys one-for-one.

    Parameters
    ----------
    config : InceptionV4Config
        Architecture spec.  Use :func:`inception_v4_cls` for the paper
        configuration.

    Attributes
    ----------
    config : InceptionV4Config
        Stored copy of the config that built this model.
    features : nn.Sequential
        Same 22-module trunk as :attr:`InceptionV4.features`.
    global_pool : nn.AdaptiveAvgPool2d
        Global average pool to :math:`1\times1`.
    head_drop : nn.Module
        :class:`~lucid.nn.Dropout` when ``config.dropout > 0``, otherwise
        :class:`~lucid.nn.Identity`.
    last_linear : nn.Linear
        Classifier projecting 1536 → ``num_classes``.

    Notes
    -----
    The loss, computed only when ``labels`` is given, is the mean
    categorical cross-entropy

    .. math::

        \mathcal{L} = -\frac{1}{N} \sum_{n=1}^{N}
            \log \operatorname{softmax}(\text{logits}_n)_{\,y_n}.

    42.7 M parameters in total.  The paper reports 20.0% top-1 / 5.0%
    top-5 single-crop error on the ImageNet validation set (Table 2).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.inception_v4 import inception_v4_cls
    >>> model = inception_v4_cls().eval()
    >>> out = model(lucid.randn(2, 3, 299, 299))
    >>> out.logits.shape
    (2, 1000)
    """

    config_class: ClassVar[type[InceptionV4Config]] = InceptionV4Config
    base_model_prefix: ClassVar[str] = "inception_v4"

    def __init__(self, config: InceptionV4Config) -> None:
        super().__init__(config)
        self.features = _build_features(config)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        if config.dropout > 0.0:
            self.head_drop: nn.Dropout | nn.Identity = nn.Dropout(config.dropout)
        else:
            self.head_drop = nn.Identity()
        self.last_linear = nn.Linear(_NUM_FEATURES, config.num_classes)

    def reset_classifier(self, num_classes: int) -> None:
        """Replace the classification head with a freshly initialised one.

        The head keeps the checkpoint's ``last_linear`` attribute name, so
        this cannot come from ``ClassificationHeadMixin`` (which looks for
        ``self.classifier``).

        Parameters
        ----------
        num_classes : int
            Width of the new head.
        """
        self.last_linear = nn.Linear(self.last_linear.in_features, num_classes)

    @override
    def forward(self, x: Tensor, labels: Tensor | None = None) -> InceptionV4Output:
        r"""Classify a batch of images.

        Parameters
        ----------
        x : Tensor
            Input images of shape ``(B, in_channels, H, W)``.
        labels : Tensor or None, optional, default=None
            Integer class targets of shape ``(B,)``.  When given, the
            cross-entropy loss is returned alongside the logits.

        Returns
        -------
        InceptionV4Output
            ``logits`` of shape ``(B, num_classes)`` and ``loss`` (or
            ``None``).
        """
        feats = self.features(x)
        pooled = self.global_pool(feats).flatten(1)
        logits = self.last_linear(self.head_drop(pooled))

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return InceptionV4Output(logits=logits, loss=loss)
