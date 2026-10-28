"""VGG backbone and classifier (Simonyan & Zisserman, 2014).

Paper: "Very Deep Convolutional Networks for Large-Scale Image Recognition"
All variants share the same macro structure:
  5 blocks of (N × Conv3×3 [→ BN] → ReLU) → MaxPool
  → AdaptiveAvgPool(7×7)
  → FC(512*7*7, 4096) → ReLU → Dropout
  → FC(4096, 4096)   → ReLU → Dropout
  → FC(4096, num_classes)

Channel widths: [64, 128, 256, 512, 512] — fixed across all VGG variants;
only the per-block conv count changes.
"""

from typing import ClassVar, override

import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._utils._common import init_cnn_fan_out
from lucid.models._base import PretrainedModel
from lucid.models._tasks import ImageClassificationModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models.vision.vgg._config import VGGConfig

_CHANNELS = (64, 128, 256, 512, 512)


def _make_block(in_ch: int, out_ch: int, num_convs: int, batch_norm: bool) -> list[nn.Module]:
    layers: list[nn.Module] = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_ch, out_ch, 3, padding=1))
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        in_ch = out_ch
    layers.append(nn.MaxPool2d(2, stride=2))
    return layers


def _build_features(cfg: VGGConfig) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_ch = cfg.in_channels
    for out_ch, n in zip(_CHANNELS, cfg.arch, strict=True):
        layers.extend(_make_block(in_ch, out_ch, n, cfg.batch_norm))
        in_ch = out_ch
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# VGG backbone  (task="base")
# ---------------------------------------------------------------------------


class VGG(PretrainedModel, BackboneMixin):
    r"""VGG feature-extracting backbone (no fully-connected head).

    Implements the unified VGG topology from Simonyan & Zisserman,
    "Very Deep Convolutional Networks for Large-Scale Image
    Recognition", ICLR 2015.  All variants (VGG-11/13/16/19, with or
    without :class:`~lucid.nn.BatchNorm2d`) share the same macro
    structure: five blocks of stacked :math:`3\times3` convolutions
    interleaved with :math:`2\times2` max-pools, with channel widths
    fixed at ``(64, 128, 256, 512, 512)``.  Only the number of
    convolutions per block changes between variants.  A final
    :class:`~lucid.nn.AdaptiveAvgPool2d` collapses the feature map to
    :math:`7\times7` so the backbone output is fixed-size regardless of
    input resolution.

    Parameters
    ----------
    config : VGGConfig
        Frozen architecture spec.  Use the factory functions
        (:func:`vgg_11`, :func:`vgg_13`, :func:`vgg_16`, :func:`vgg_19`
        and their ``*_bn`` variants) for paper-cited configurations.

    Attributes
    ----------
    config : VGGConfig
        Stored copy of the config that built this model.
    features : nn.Sequential
        The five conv blocks; structure depends on ``config.arch`` and
        ``config.batch_norm`` — see :func:`_build_features`.
    avgpool : nn.AdaptiveAvgPool2d
        Global pool down to :math:`7\times7` so the backbone produces a
        fixed-size feature regardless of input resolution.
    feature_info : list[FeatureInfo]
        Per-stage descriptor (channels + reduction factor) exposed via
        :class:`BackboneMixin` for downstream decoder modules.

    Notes
    -----
    From Simonyan & Zisserman, ICLR 2015, Table 1.  The motivating
    insight is that two stacked :math:`3\times3` convolutions cover the
    same receptive field as a single :math:`5\times5` convolution but
    use only :math:`2 \cdot 9 C^2 = 18 C^2` parameters versus
    :math:`25 C^2`, while inserting an extra nonlinearity between them.
    Three stacked :math:`3\times3` layers match a :math:`7\times7`
    receptive field at :math:`27 C^2` versus :math:`49 C^2` parameters
    and *two* extra nonlinearities.  Depth at a fixed receptive field
    therefore buys representational power essentially for free.  VGG
    backbones became the standard feature extractor for downstream
    tasks (Faster R-CNN, neural style transfer) for several years
    after publication.

    Examples
    --------
    Build a VGG-16 backbone and run a single forward pass:

    >>> import lucid
    >>> from lucid.models.vision.vgg import vgg_16
    >>> backbone = vgg_16()
    >>> x = lucid.randn(2, 3, 224, 224)
    >>> out = backbone(x)
    >>> out.last_hidden_state.shape   # (B, 512, 7, 7)
    (2, 512, 7, 7)
    """

    config_class: ClassVar[type[VGGConfig]] = VGGConfig
    base_model_prefix: ClassVar[str] = "vgg"

    def __init__(self, config: VGGConfig) -> None:
        super().__init__(config)
        self.features = _build_features(config)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self._feature_info = [
            FeatureInfo(stage=i + 1, num_channels=ch, reduction=2 ** (i + 1))
            for i, ch in enumerate(_CHANNELS)
        ]

        # Reference initialisation (He/MSRA fan-out convs, unit norms).
        # Without this the family starts from Lucid's generic
        # kaiming_uniform(a=sqrt(5)), a different distribution and gain
        # than the one the paper's schedule is tuned for.
        init_cnn_fan_out(self)

    @override
    def forward_features(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return self.avgpool(x)

    @override
    def forward(self, x: Tensor) -> BaseModelOutput:
        return BaseModelOutput(last_hidden_state=self.forward_features(x))


# ---------------------------------------------------------------------------
# VGG for image classification  (task="image-classification")
# ---------------------------------------------------------------------------


class VGGForImageClassification(ImageClassificationModel, ClassificationHeadMixin):
    r"""VGG with two 4096-dim fully-connected layers and a linear classifier.

    Combines a :class:`VGG` convolutional backbone with the standard
    paper-cited classifier head: FC6 (512·7·7 = 25088 → 4096), FC7
    (4096 → 4096), and a final linear projection to
    ``config.num_classes`` (default 1000 for ImageNet-1k).
    :class:`~lucid.nn.Dropout` with ``config.dropout`` (0.5 in the
    paper) is applied after both ReLU activations.  When ``labels`` are
    supplied to :meth:`forward`, cross-entropy loss is returned
    alongside the logits.

    Parameters
    ----------
    config : VGGConfig
        Architecture spec.  Use the ``*_cls`` factory functions
        (:func:`vgg_11_cls`, :func:`vgg_16_cls`, …) for paper-cited
        configurations.

    Attributes
    ----------
    config : VGGConfig
        Stored copy of the config that built this model.
    features, avgpool
        Same backbone components as on :class:`VGG`.
    fc6, fc7 : nn.Linear
        The two 4096-dim hidden fully-connected layers.
    drop6, drop7 : nn.Dropout
        Dropout layers applied after each ReLU in the hidden FC stack,
        controlled by ``config.dropout``.
    classifier : nn.Module
        Final linear projection 4096 → ``num_classes``.

    Notes
    -----
    From Simonyan & Zisserman, ICLR 2015.  The two 4096-dim FC layers
    dominate the parameter count: VGG-16 has 138 M parameters total, of
    which roughly 124 M sit in FC6 + FC7.  This is a recognised
    weakness — almost all later architectures (ResNet, DenseNet, …)
    replace the giant FC head with a single :class:`~lucid.nn.Linear`
    after :class:`~lucid.nn.AdaptiveAvgPool2d` to slash the parameter
    cost.  VGG-16 reaches a top-5 ImageNet validation error of 7.3% and
    VGG-19 reaches 7.5%.

    Examples
    --------
    Run inference on a 224x224 RGB batch:

    >>> import lucid
    >>> from lucid.models.vision.vgg import vgg_16_cls
    >>> model = vgg_16_cls()
    >>> x = lucid.randn(2, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (2, 1000)
    """

    config_class: ClassVar[type[VGGConfig]] = VGGConfig
    base_model_prefix: ClassVar[str] = "vgg"

    def __init__(self, config: VGGConfig) -> None:
        super().__init__(config)
        self.features = _build_features(config)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.drop6 = nn.Dropout(p=config.dropout)
        self.drop7 = nn.Dropout(p=config.dropout)
        self._build_classifier(4096, config.num_classes)
        init_cnn_fan_out(self)

    @override
    def forward(self, x: Tensor, labels: Tensor | None = None) -> ImageClassificationOutput:
        feat = self.avgpool(self.features(x)).flatten(1)
        h = self.drop6(F.relu(self.fc6(feat)))
        h = self.drop7(F.relu(self.fc7(h)))
        return self._classify(h, labels)
