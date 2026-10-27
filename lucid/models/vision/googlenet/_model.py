"""GoogLeNet (Inception v1) backbone and classifier (Szegedy et al., 2014).

Paper: "Going Deeper with Convolutions"

Architecture overview:
    Stem   : Conv7×7-s2 → MaxPool3×3-s2 → Conv1×1 → Conv3×3 → MaxPool3×3-s2
    Stage 3: Inception(3a) → Inception(3b) → MaxPool3×3-s2
    Stage 4: Inception(4a)[→aux1] → (4b) → (4c) → (4d)[→aux2] → (4e)
             → MaxPool2×2-s2
    Stage 5: Inception(5a) → Inception(5b)
    Head   : AdaptiveAvgPool(1×1) → Dropout(0.4) → FC(1024, num_classes)

Each Inception module is a four-branch parallel block:
    branch1 : 1×1 conv
    branch2 : 1×1 conv → 3×3 conv
    branch3 : 1×1 conv → 3×3 conv
    branch4 : 3×3 max pool → 1×1 conv
    → channel-wise concat

This implementation mirrors the canonical batch-normalised reference
implementation: every convolution is wrapped in a
:class:`_BasicConv2d` (Conv → BatchNorm → ReLU), the third Inception
branch uses a :math:`3\times3` convolution rather than the
paper's :math:`5\times5`, and every max-pool uses ``ceil_mode=True``.
This makes the architecture state-dict-compatible with the widely
distributed ImageNet-1k checkpoint.

The two auxiliary classifiers (training only) attach at Inception 4a
and 4d:
    AdaptiveAvgPool(4×4) → Conv1×1(128) → FC(1024) → Dropout(0.7)
    → FC(num_classes)
"""

from dataclasses import dataclass
from typing import ClassVar, final, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._utils._common import transform_input_imagenet_to_tf
from lucid.models._base import PretrainedModel
from lucid.models._tasks import ImageClassificationModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models.vision.googlenet._config import GoogLeNetConfig

# Large finite stand-in for ``-inf`` used to pad the extra ceil-mode
# window so it never wins a max against real activations.

# ---------------------------------------------------------------------------
# Ceil-mode max pool (stateless)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Basic Conv → BatchNorm → ReLU block
# ---------------------------------------------------------------------------


@final
class _BasicConv2d(nn.Module):
    """Conv (no bias) → BatchNorm → ReLU, the GoogLeNet conv primitive."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.bn(self.conv(x)))


# ---------------------------------------------------------------------------
# Inception module
# ---------------------------------------------------------------------------


@final
class _InceptionModule(nn.Module):
    """Single Inception block with four parallel branches."""

    def __init__(
        self,
        in_channels: int,
        out_1x1: int,
        out_3x3_reduce: int,
        out_3x3: int,
        out_5x5_reduce: int,
        out_5x5: int,
        out_pool_proj: int,
    ) -> None:
        super().__init__()
        # Branch 1: 1×1 (named directly, not wrapped in a Sequential)
        self.branch1 = _BasicConv2d(in_channels, out_1x1, kernel_size=1)
        # Branch 2: 1×1 → 3×3
        self.branch2 = nn.Sequential(
            _BasicConv2d(in_channels, out_3x3_reduce, kernel_size=1),
            _BasicConv2d(out_3x3_reduce, out_3x3, kernel_size=3, padding=1),
        )
        # Branch 3: 1×1 → 3×3 (the reference 5×5-branch deviation)
        self.branch3 = nn.Sequential(
            _BasicConv2d(in_channels, out_5x5_reduce, kernel_size=1),
            _BasicConv2d(out_5x5_reduce, out_5x5, kernel_size=3, padding=1),
        )
        # Branch 4: 3×3 max pool → 1×1
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True),
            _BasicConv2d(in_channels, out_pool_proj, kernel_size=1),
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return lucid.cat([b1, b2, b3, b4], dim=1)


# ---------------------------------------------------------------------------
# Auxiliary classifier (training only)
# ---------------------------------------------------------------------------


@final
class _AuxClassifier(nn.Module):
    """Auxiliary classifier attached at Inception 4a and 4d."""

    def __init__(self, in_channels: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.conv = _BasicConv2d(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = F.adaptive_avg_pool2d(x, (4, 4))
        x = self.conv(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# ---------------------------------------------------------------------------
# GoogLeNet output dataclass
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class GoogLeNetOutput(ImageClassificationOutput):
    r"""Structured forward output for :class:`GoogLeNetForImageClassification`.

    Carries the main classifier logits plus the optional auxiliary
    classifier logits emitted during training.  Auxiliary heads attach
    at Inception 4a and 4d in the original paper and are evaluated
    *only* in training mode when ``config.aux_logits=True``; at
    inference both auxiliary fields are ``None``.

    Parameters
    ----------
    logits : Tensor
        Main classifier output of shape ``(B, num_classes)``.
    aux_logits1 : Tensor or None, optional, default=None
        Logits from the first auxiliary classifier (attached at
        Inception 4a) of shape ``(B, num_classes)``; ``None`` at
        inference or when ``aux_logits=False``.
    aux_logits2 : Tensor or None, optional, default=None
        Logits from the second auxiliary classifier (attached at
        Inception 4d) of shape ``(B, num_classes)``; ``None`` at
        inference or when ``aux_logits=False``.
    loss : Tensor or None, optional, default=None
        Cross-entropy loss including auxiliary terms when labels were
        passed to :meth:`GoogLeNetForImageClassification.forward`;
        ``None`` otherwise.

    Notes
    -----
    From Szegedy et al., "Going Deeper with Convolutions", CVPR 2015,
    §5.  When auxiliary heads are active the training loss combines
    the main and auxiliary cross-entropies as

    .. math::

        \mathcal{L} = \mathcal{L}_{\text{main}}
            + 0.3 \cdot \mathcal{L}_{\text{aux1}}
            + 0.3 \cdot \mathcal{L}_{\text{aux2}},

    a weighting taken directly from the paper that compensates for the
    auxiliary heads' shallower position in the network.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.googlenet import googlenet_cls
    >>> model = googlenet_cls().eval()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    >>> out.aux_logits1 is None
    True
    """

    aux_logits1: Tensor | None = None
    aux_logits2: Tensor | None = None


# Inception module specs: (in_ch, 1x1, 3x3r, 3x3, 5x5r, 5x5, pool)
_INCEPTION_SPECS: list[tuple[int, int, int, int, int, int, int]] = [
    (192, 64, 96, 128, 16, 32, 32),  # 3a → 256
    (256, 128, 128, 192, 32, 96, 64),  # 3b → 480
    # after maxpool(3×3 s2)
    (480, 192, 96, 208, 16, 48, 64),  # 4a → 512  [aux1 here]
    (512, 160, 112, 224, 24, 64, 64),  # 4b → 512
    (512, 128, 128, 256, 24, 64, 64),  # 4c → 512
    (512, 112, 144, 288, 32, 64, 64),  # 4d → 528  [aux2 here]
    (528, 256, 160, 320, 32, 128, 128),  # 4e → 832
    # after maxpool(3×3 s2)
    (832, 256, 160, 320, 32, 128, 128),  # 5a → 832
    (832, 384, 192, 384, 48, 128, 128),  # 5b → 1024
]


def _make_inception(spec: tuple[int, int, int, int, int, int, int]) -> _InceptionModule:
    in_ch, o1, o3r, o3, o5r, o5, op = spec
    return _InceptionModule(in_ch, o1, o3r, o3, o5r, o5, op)


# ---------------------------------------------------------------------------
# GoogLeNet backbone  (task="base")
# ---------------------------------------------------------------------------


def _init_reference_weights(model: nn.Module) -> None:
    r"""Truncated-normal init at std 0.01, as the reference builds this net.

    The reference initialises every conv and linear with
    ``trunc_normal_(std=0.01, a=-2, b=2)`` and sets BatchNorm to unit
    scale / zero shift.  The unusually small std is deliberate for a
    22-layer network with no residual connections — Lucid's default is a
    kaiming-uniform roughly 7x wider, and uniform rather than truncated
    normal.
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01, a=-2.0, b=2.0)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class GoogLeNet(PretrainedModel, BackboneMixin):
    r"""GoogLeNet (Inception v1) feature-extracting backbone.

    Implements the 22-layer Inception v1 network from Szegedy et al.,
    "Going Deeper with Convolutions", CVPR 2015: a five-block stem
    (Conv :math:`7\times7` stride-2 → MaxPool → :math:`1\times1` →
    :math:`3\times3` → MaxPool) followed by nine
    :class:`_InceptionModule` blocks grouped into three stages
    (Inception 3a-3b → MaxPool → 4a-4e → MaxPool → 5a-5b) and a final
    :class:`~lucid.nn.AdaptiveAvgPool2d` to a :math:`1\times1` spatial
    map.  Every convolution is batch-normalised, matching the canonical
    distributed ImageNet checkpoint.  The two auxiliary classifiers
    attached at Inception 4a and 4d in the original paper are part of
    the classifier variant only, not the backbone.

    Parameters
    ----------
    config : GoogLeNetConfig
        Frozen architecture spec.  Use :func:`googlenet` for the
        paper-cited configuration; pass a custom config to switch input
        channel count.  ``aux_logits`` is irrelevant for the backbone.

    Attributes
    ----------
    config : GoogLeNetConfig
        Stored copy of the config that built this model.
    conv1, conv2, conv3 : _BasicConv2d
        The pre-Inception conv stack.
    maxpool1, maxpool2 : nn.MaxPool2d
        Stem :math:`3\times3` stride-2 ceil-mode max-pools.
    inception3a, inception3b : _InceptionModule
        First Inception stage, producing a :math:`28\times28` spatial
        map with 256 and 480 channels respectively.
    inception4a … inception4e : _InceptionModule
        Second Inception stage at :math:`14\times14` resolution; widths
        ramp 512 → 512 → 512 → 528 → 832.
    inception5a, inception5b : _InceptionModule
        Third Inception stage at :math:`7\times7` resolution, producing
        the final 1024-channel feature map.
    maxpool3, maxpool4 : nn.MaxPool2d
        Ceil-mode max-pools between Inception stages
        (:math:`3\times3` stride-2 and :math:`2\times2` stride-2).
    avgpool : nn.AdaptiveAvgPool2d
        Final global average pool collapsing the :math:`7\times7`
        feature map to :math:`1\times1`.
    feature_info : list[FeatureInfo]
        Per-stage descriptor (channels + reduction factor) exposed via
        :class:`BackboneMixin`.

    Notes
    -----
    Each :class:`_InceptionModule` computes four parallel branches in a
    single layer:

    .. math::

        y = \mathrm{concat}\bigl(
            f_{1\times1}(x),\; f_{3\times3}(g_{1\times1}(x)),\;
            f_{3\times3}(h_{1\times1}(x)),\;
            p_{1\times1}(\mathrm{pool}(x))
        \bigr).

    The :math:`1\times1` bottlenecks before the larger spatial
    convolutions keep the parameter count manageable: GoogLeNet has
    only ≈6.8 M parameters compared with ≈60 M for AlexNet, despite
    being substantially deeper and reaching a top-5 ImageNet validation
    error of 6.67%.  This was the ILSVRC-2014 classification winner.

    Examples
    --------
    Build a GoogLeNet backbone and run a single forward pass:

    >>> import lucid
    >>> from lucid.models.vision.googlenet import googlenet
    >>> backbone = googlenet()
    >>> x = lucid.randn(2, 3, 224, 224)
    >>> out = backbone(x)
    >>> out.last_hidden_state.shape
    (2, 1024, 7, 7)
    """

    config_class: ClassVar[type[GoogLeNetConfig]] = GoogLeNetConfig
    base_model_prefix: ClassVar[str] = "googlenet"

    def __init__(self, config: GoogLeNetConfig) -> None:
        super().__init__(config)
        self.conv1 = _BasicConv2d(config.in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = _BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = _BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        specs = _INCEPTION_SPECS
        self.inception3a = _make_inception(specs[0])
        self.inception3b = _make_inception(specs[1])
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = _make_inception(specs[2])
        self.inception4b = _make_inception(specs[3])
        self.inception4c = _make_inception(specs[4])
        self.inception4d = _make_inception(specs[5])
        self.inception4e = _make_inception(specs[6])
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = _make_inception(specs[7])
        self.inception5b = _make_inception(specs[8])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self._feature_info = [
            FeatureInfo(stage=3, num_channels=480, reduction=8),
            FeatureInfo(stage=4, num_channels=832, reduction=16),
            FeatureInfo(stage=5, num_channels=1024, reduction=32),
        ]

        _init_reference_weights(self)

    @override
    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    @override
    def forward_features(self, x: Tensor) -> Tensor:
        if getattr(self.config, "transform_input", False):
            x = transform_input_imagenet_to_tf(x)
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
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
# GoogLeNet for image classification  (task="image-classification")
# ---------------------------------------------------------------------------


class GoogLeNetForImageClassification(ImageClassificationModel, ClassificationHeadMixin):
    r"""GoogLeNet image classifier with optional auxiliary classifiers.

    Combines the GoogLeNet backbone with a global-average-pool +
    :class:`~lucid.nn.Dropout` (``config.dropout``, default 0.4) +
    :class:`~lucid.nn.Linear` head producing ``config.num_classes``
    logits, plus two optional auxiliary classifiers attached at
    Inception 4a (512-channel feature map) and Inception 4d
    (528-channel feature map).  Auxiliary heads are evaluated only in
    training mode and only when ``config.aux_logits=True``; their
    contributions to the loss are weighted 0.3 each, following the
    original paper.

    Parameters
    ----------
    config : GoogLeNetConfig
        Architecture spec.  Use :func:`googlenet_cls` for the
        paper-cited configuration (auxiliary heads enabled).  Pass
        ``aux_logits=False`` for an inference-only variant.

    Attributes
    ----------
    config : GoogLeNetConfig
        Stored copy of the config that built this model.
    conv1, conv2, conv3, maxpool1, maxpool2, inception3a, ...,
    inception5b, maxpool3, maxpool4
        Same backbone components as on :class:`GoogLeNet`.
    avgpool : nn.AdaptiveAvgPool2d
        Final global average pool to :math:`1\times1`.
    drop : nn.Dropout
        Dropout applied before the main classifier
        (``p=config.dropout``, 0.4 in the paper).
    classifier : nn.Module
        Final linear projection 1024 → ``num_classes``, built by
        :meth:`ClassificationHeadMixin._build_classifier`.
    aux1, aux2 : nn.Module
        Auxiliary classifiers (instances of :class:`_AuxClassifier`)
        when ``config.aux_logits=True``; otherwise
        :class:`~lucid.nn.Identity` placeholders.

    Notes
    -----
    From Szegedy et al., "Going Deeper with Convolutions", CVPR 2015,
    §5.  The full training loss with auxiliary heads is

    .. math::

        \mathcal{L} = \mathcal{L}_{\text{main}}
            + 0.3 \cdot \mathcal{L}_{\text{aux1}}
            + 0.3 \cdot \mathcal{L}_{\text{aux2}},

    each term being the standard categorical cross-entropy
    :math:`-\frac{1}{N} \sum_n \log \operatorname{softmax}(\cdot)_{y_n}`.
    The :math:`0.3` weighting compensates for the auxiliary heads being
    much shallower than the main classifier; at inference only the main
    branch is evaluated.

    Examples
    --------
    Inference path (no labels, auxiliary heads inactive):

    >>> import lucid
    >>> from lucid.models.vision.googlenet import googlenet_cls
    >>> model = googlenet_cls().eval()
    >>> x = lucid.randn(4, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (4, 1000)
    >>> out.aux_logits1 is None
    True

    Training path (auxiliary heads active, labels provided):

    >>> model = model.train()
    >>> labels = lucid.tensor([0, 1, 2, 3], dtype=lucid.int64)
    >>> out = model(x, labels=labels)
    >>> out.aux_logits1.shape   # available during training
    (4, 1000)
    """

    config_class: ClassVar[type[GoogLeNetConfig]] = GoogLeNetConfig
    base_model_prefix: ClassVar[str] = "googlenet"

    def __init__(self, config: GoogLeNetConfig) -> None:
        super().__init__(config)
        self.conv1 = _BasicConv2d(config.in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = _BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = _BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        specs = _INCEPTION_SPECS
        self.inception3a = _make_inception(specs[0])
        self.inception3b = _make_inception(specs[1])
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = _make_inception(specs[2])
        self.inception4b = _make_inception(specs[3])
        self.inception4c = _make_inception(specs[4])
        self.inception4d = _make_inception(specs[5])
        self.inception4e = _make_inception(specs[6])
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = _make_inception(specs[7])
        self.inception5b = _make_inception(specs[8])

        # Auxiliary classifiers (attached at 4a → 512ch, 4d → 528ch)
        if config.aux_logits:
            self.aux1: nn.Module = _AuxClassifier(512, config.num_classes, config.aux_dropout)
            self.aux2: nn.Module = _AuxClassifier(528, config.num_classes, config.aux_dropout)
        else:
            self.aux1 = nn.Identity()
            self.aux2 = nn.Identity()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(p=config.dropout)
        self._build_classifier(1024, config.num_classes)

        _init_reference_weights(self)

    @override
    def forward(self, x: Tensor, labels: Tensor | None = None) -> GoogLeNetOutput:
        cfg = self.config
        assert isinstance(cfg, GoogLeNetConfig)
        use_aux = cfg.aux_logits and self.training

        if getattr(self.config, "transform_input", False):
            x = transform_input_imagenet_to_tf(x)
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        aux1: Tensor | None = None
        if use_aux and isinstance(self.aux1, _AuxClassifier):
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        aux2: Tensor | None = None
        if use_aux and isinstance(self.aux2, _AuxClassifier):
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = self.drop(x.flatten(1))
        logits = self.classifier(x)

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            if aux1 is not None:
                loss = loss + 0.3 * F.cross_entropy(aux1, labels)
            if aux2 is not None:
                loss = loss + 0.3 * F.cross_entropy(aux2, labels)

        return GoogLeNetOutput(logits=logits, aux_logits1=aux1, aux_logits2=aux2, loss=loss)
