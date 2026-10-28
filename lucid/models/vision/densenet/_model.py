"""DenseNet backbone and classifier (Huang et al., 2016).

Paper: "Densely Connected Convolutional Networks"

Key idea: each layer receives feature maps from ALL preceding layers in the block
(dense connection), then contributes its own k feature maps.  This encourages
feature reuse and keeps the model compact.

Architecture:
    Stem     : Conv7×7-s2(→num_init_features) → BN → ReLU → MaxPool3×3-s2
    Block 1  : DenseBlock(layers[0]) → Transition (½ channels, AvgPool-s2)
    Block 2  : DenseBlock(layers[1]) → Transition
    Block 3  : DenseBlock(layers[2]) → Transition
    Block 4  : DenseBlock(layers[3])
    Head     : BN → ReLU → AdaptiveAvgPool(1×1) → FC

Each DenseLayer (bottleneck variant):
    BN → ReLU → Conv1×1(bn_size×k) → BN → ReLU → Conv3×3(k) [→ Dropout]

Transition block:
    BN → ReLU → Conv1×1(½ in_channels) → AvgPool2×2-s2

State-dict key structure mirrors timm densenet121:
    features.conv0.*            ← stem Conv2d
    features.norm0.*            ← stem BN
    features.denseblock1.denselayer1.norm1/conv1/norm2/conv2.*
    features.transition1.norm/conv.*
    features.norm5.*            ← final BN
    classifier.weight/bias
"""

from typing import ClassVar, final, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._tasks import ImageClassificationModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models.vision.densenet._config import DenseNetConfig

# ---------------------------------------------------------------------------
# Dense layer (bottleneck)
# ---------------------------------------------------------------------------


@final
class _DenseLayer(nn.Module):
    """Single dense layer: BN-ReLU-Conv1×1-BN-ReLU-Conv3×3."""

    def __init__(
        self, num_input_features: int, growth_rate: int, bn_size: int, dropout_rate: float
    ) -> None:
        super().__init__()
        inter_features = bn_size * growth_rate
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.conv1 = nn.Conv2d(num_input_features, inter_features, 1, bias=False)
        self.norm2 = nn.BatchNorm2d(inter_features)
        self.conv2 = nn.Conv2d(inter_features, growth_rate, 3, padding=1, bias=False)
        # Section 3 reads literally: "we add a dropout layer after each
        # convolutional layer (except the first one)" — which would put
        # dropout after the 1x1 bottleneck conv and inside the transition
        # layers too, as the authors' original reference-framework code
        # does.  Only the post-3x3 site is built here, matching the
        # reference the shipped checkpoints come from.  Training-time
        # only, so inference is unaffected either way.
        self.drop = nn.Dropout(p=dropout_rate) if dropout_rate > 0.0 else None

    @override
    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(F.relu(self.norm1(x)))
        out = self.conv2(F.relu(self.norm2(out)))
        if self.drop is not None:
            out = self.drop(out)
        # Dense connection: concat input with new features along channel dim
        return lucid.cat([x, out], dim=1)


# ---------------------------------------------------------------------------
# Dense block  — children named  denselayer1, denselayer2, …  (1-indexed)
# ---------------------------------------------------------------------------


@final
class _DenseBlock(nn.Module):
    """N stacked _DenseLayers; each layer receives all previous outputs.

    Children are registered as ``denselayer1``, ``denselayer2``, … so that
    the state-dict mirrors timm's key scheme.
    """

    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        growth_rate: int,
        bn_size: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self._layers: list[_DenseLayer] = []
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate, growth_rate, bn_size, dropout_rate
            )
            self.add_module(f"denselayer{i + 1}", layer)
            self._layers.append(layer)

    @override
    def forward(self, x: Tensor) -> Tensor:
        for module in self._layers:
            x = module(x)
        return x


# ---------------------------------------------------------------------------
# Transition block  — attributes: norm, conv, pool
# ---------------------------------------------------------------------------


@final
class _Transition(nn.Module):
    """BN-ReLU-Conv1×1-AvgPool — halves channels and spatial resolution."""

    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.conv = nn.Conv2d(num_input_features, num_output_features, 1, bias=False)
        self.pool = nn.AvgPool2d(2, stride=2)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(F.relu(self.norm(x)))
        return self.pool(x)


# ---------------------------------------------------------------------------
# Features container — mirrors timm's ``features`` attribute
#
#   features.conv0        ← stem Conv2d (no bias)
#   features.norm0        ← stem BN
#   features.relu0        ← stem ReLU   (no params, but part of stem)
#   features.pool0        ← stem MaxPool (no params)
#   features.denseblock1  ← block 1
#   features.transition1  ← transition after block 1
#   features.denseblock2
#   features.transition2
#   features.denseblock3
#   features.transition3
#   features.denseblock4  ← last block (no transition)
#   features.norm5        ← final BN
# ---------------------------------------------------------------------------


@final
class _Features(nn.Module):
    """Flat-attribute container whose state-dict key prefix is ``features``."""

    def __init__(self, cfg: DenseNetConfig) -> None:
        super().__init__()
        # --- stem ---
        self.conv0 = nn.Conv2d(
            cfg.in_channels, cfg.num_init_features, 7, stride=2, padding=3, bias=False
        )
        self.norm0 = nn.BatchNorm2d(cfg.num_init_features)
        self.relu0 = nn.ReLU(inplace=True)
        self.pool0 = nn.MaxPool2d(3, stride=2, padding=1)

        # --- dense blocks + transitions ---
        self._stages: list[_DenseBlock | _Transition] = []
        num_features = cfg.num_init_features
        num_blocks = len(cfg.block_config)
        for idx, num_layers in enumerate(cfg.block_config):
            block = _DenseBlock(
                num_layers, num_features, cfg.growth_rate, cfg.bn_size, cfg.dropout_rate
            )
            self.add_module(f"denseblock{idx + 1}", block)
            self._stages.append(block)
            num_features += num_layers * cfg.growth_rate

            if idx < num_blocks - 1:
                out_features = num_features // 2
                transition = _Transition(num_features, out_features)
                self.add_module(f"transition{idx + 1}", transition)
                self._stages.append(transition)
                num_features = out_features

        # --- final BN ---
        self.norm5 = nn.BatchNorm2d(num_features)
        self._num_features = num_features

    @property
    def num_features(self) -> int:
        return self._num_features

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = self.pool0(self.relu0(self.norm0(self.conv0(x))))
        # iterate dense blocks and transitions in registration order
        for module in self._stages:
            x = module(x)
        # final BN + ReLU (caller applies ReLU after pool)
        return self.norm5(x)


# ---------------------------------------------------------------------------
# DenseNet backbone  (task="base")
# ---------------------------------------------------------------------------


def _init_reference(model: nn.Module) -> None:
    """Reference DenseNet initialisation — ``kaiming_normal_`` at its defaults.

    The reference's ``_initialize_weights`` is a bare
    ``nn.init.kaiming_normal_(m.weight)`` on convolutions, which means
    ``mode="fan_in", nonlinearity="leaky_relu"`` — not the explicit
    ``fan_out`` / ``relu`` pair the ResNet family passes.  Norm layers get
    unit weight and zero bias; the classifier bias is zeroed.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)


class DenseNet(PretrainedModel, BackboneMixin):
    r"""DenseNet feature-extracting backbone (no classification head).

    Implements the densely-connected topology from Huang et al.,
    "Densely Connected Convolutional Networks", CVPR 2017: a Conv-BN-
    ReLU-MaxPool stem followed by four :class:`_DenseBlock` stages
    interleaved with three :class:`_Transition` blocks that halve both
    the channel and spatial dimensions.  Inside each dense block every
    layer receives the concatenated feature maps of *all* preceding
    layers in the block.  A final BatchNorm + ReLU + global average
    pool produces a fixed-size feature regardless of input resolution.

    Parameters
    ----------
    config : DenseNetConfig
        Frozen architecture spec.  Use the factory functions
        (:func:`densenet_121`, :func:`densenet_169`, …) for the
        paper-cited variants.

    Attributes
    ----------
    config : DenseNetConfig
        Stored copy of the config that built this model.
    features : _Features
        Flat container holding the stem (``conv0``, ``norm0``,
        ``relu0``, ``pool0``), the four dense blocks
        (``denseblock1`` … ``denseblock4``), the three transitions
        (``transition1`` … ``transition3``), and the final BatchNorm
        (``norm5``).  State-dict key prefix is ``features.*``.
    avgpool : nn.AdaptiveAvgPool2d
        Global average pool collapsing the final feature map to
        :math:`1\times1`.
    feature_info : list[FeatureInfo]
        Per-stage descriptor (channels + reduction factor) exposed via
        :class:`BackboneMixin`.

    Notes
    -----
    From Huang et al., CVPR 2017.  Each layer in a dense block computes

    .. math::

        x_\ell = H_\ell\bigl(
            [x_0,\; x_1,\; \dots,\; x_{\ell-1}]
        \bigr),

    where :math:`[\cdot]` is channel-wise concatenation and
    :math:`H_\ell` is a BN → ReLU → :math:`1\times1` Conv → BN → ReLU
    → :math:`3\times3` Conv composite (the bottleneck variant used in
    all ImageNet DenseNets).  Each :math:`H_\ell` contributes
    ``growth_rate`` new channels — typically :math:`k = 32` — so the
    input width of layer :math:`\ell` grows linearly as :math:`k_0 +
    (\ell - 1)\,k`.  Transition layers compress the channel count by
    a factor of 2 between blocks.  DenseNet-121 has only 8 M parameters
    yet matches the accuracy of ResNet-50 (25 M parameters) on
    ImageNet.

    Examples
    --------
    Build a DenseNet-121 backbone and run a single forward pass:

    >>> import lucid
    >>> from lucid.models.vision.densenet import densenet_121
    >>> backbone = densenet_121()
    >>> x = lucid.randn(2, 3, 224, 224)
    >>> out = backbone(x)
    >>> out.last_hidden_state.shape   # (B, 1024, 1, 1)
    (2, 1024, 1, 1)
    """

    config_class: ClassVar[type[DenseNetConfig]] = DenseNetConfig
    base_model_prefix: ClassVar[str] = "densenet"

    def __init__(self, config: DenseNetConfig) -> None:
        super().__init__(config)
        self.features = _Features(config)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # feature_info: one entry per block, reporting the dense block's own
        # output (before the following transition).
        #
        # The running width has to carry the transition's theta=0.5 compression
        # forward — the previous form halved ``nf`` only for display and fed the
        # *uncompressed* total into the next block, so stages 2+ drifted
        # (121 reported 128/320/704/1920 where the network emits
        # 256/512/1024/1024).  The reduction likewise belongs to the block, not
        # to the transition that follows it, so it advances after the append.
        fi: list[FeatureInfo] = []
        nf = config.num_init_features
        reduction = 4
        last = len(config.block_config) - 1
        for i, nl in enumerate(config.block_config):
            nf += nl * config.growth_rate
            fi.append(FeatureInfo(stage=i + 1, num_channels=nf, reduction=reduction))
            if i < last:
                nf //= 2  # transition compression theta = 0.5
                reduction *= 2
        self._feature_info = fi

        # Reference initialisation.  DenseNet is NOT the fan-out/relu variant
        # the ResNet line uses: its reference calls ``kaiming_normal_(weight)``
        # with the defaults, i.e. **fan_in** and the leaky-relu gain.  Using
        # fan_out here inflates every layer's weights, and because DenseNet
        # concatenates rather than adds, that compounds across 121 layers into
        # activations around 1e5 before BN is trained.
        _init_reference(self)

    @override
    def forward_features(self, x: Tensor) -> Tensor:
        x = F.relu(self.features(x))
        return self.avgpool(x)

    @override
    def forward(self, x: Tensor) -> BaseModelOutput:
        return BaseModelOutput(last_hidden_state=self.forward_features(x))


# ---------------------------------------------------------------------------
# DenseNet for image classification  (task="image-classification")
# ---------------------------------------------------------------------------


class DenseNetForImageClassification(ImageClassificationModel, ClassificationHeadMixin):
    r"""DenseNet image classifier (backbone + GAP + linear head).

    Combines a :class:`DenseNet` backbone with a global-average-pool
    and a single :class:`~lucid.nn.Linear` classifier producing
    ``config.num_classes`` logits.  When ``labels`` are supplied to
    :meth:`forward`, a categorical cross-entropy loss is returned
    alongside the logits.

    Parameters
    ----------
    config : DenseNetConfig
        Architecture spec.  Use the ``*_cls`` factory functions
        (:func:`densenet_121_cls`, :func:`densenet_169_cls`, …) for the
        paper-cited variants.

    Attributes
    ----------
    config : DenseNetConfig
        Stored copy of the config that built this model.
    features : _Features
        Same dense feature stack as on :class:`DenseNet`.
    avgpool : nn.AdaptiveAvgPool2d
        Global average pool to :math:`1\times1`.
    classifier : nn.Linear
        Final linear projection ``features.num_features`` →
        ``num_classes``.

    Notes
    -----
    From Huang et al., "Densely Connected Convolutional Networks",
    CVPR 2017.  Loss is the standard categorical cross-entropy

    .. math::

        \mathcal{L} = -\frac{1}{N} \sum_{n=1}^{N}
            \log \operatorname{softmax}(\text{logits}_n)_{\,y_n}.

    DenseNet's parameter efficiency makes it particularly attractive
    for embedded / on-device deployment: DenseNet-121 reaches the same
    ImageNet top-1 accuracy as ResNet-50 with roughly one-third the
    parameter budget.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.densenet import densenet_121_cls
    >>> model = densenet_121_cls()
    >>> x = lucid.randn(2, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (2, 1000)
    """

    config_class: ClassVar[type[DenseNetConfig]] = DenseNetConfig
    base_model_prefix: ClassVar[str] = "densenet"

    def __init__(self, config: DenseNetConfig) -> None:
        super().__init__(config)
        self.features = _Features(config)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._build_classifier(self.features.num_features, config.num_classes)

        # Reference initialisation.  DenseNet is NOT the fan-out/relu variant
        # the ResNet line uses: its reference calls ``kaiming_normal_(weight)``
        # with the defaults, i.e. **fan_in** and the leaky-relu gain.  Using
        # fan_out here inflates every layer's weights, and because DenseNet
        # concatenates rather than adds, that compounds across 121 layers into
        # activations around 1e5 before BN is trained.
        _init_reference(self)

    @override
    def forward(self, x: Tensor, labels: Tensor | None = None) -> ImageClassificationOutput:
        x = F.relu(self.features(x))
        x = self.avgpool(x)
        x = x.flatten(1)
        return self._classify(x, labels)
