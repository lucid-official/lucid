"""Xception backbone and classifier (Chollet, 2017).

Paper: "Xception: Deep Learning with Depthwise Separable Convolutions"

Architecture overview (299×299 input):
    Stem:
        conv1 + bn1 → ReLU  (Conv 3×3-s2, 32 ch)
        conv2 + bn2 → ReLU  (Conv 3×3, 64 ch)
    Entry flow (3 blocks — block1, block2, block3):
        blockN.rep  — Sequential of ReLU/SepConv/BN ops
        blockN.skip — 1×1 Conv (channel projection)
        blockN.skipbn — BN on skip
    Middle flow (8 blocks — block4…block11):
        blockN.rep  — Sequential of 3× (ReLU+SepConv+BN)
    Exit flow (block12 + conv3/bn3 + conv4/bn4):
        block12.rep  — Sequential (SepConv(728)+SepConv(1024)+MaxPool)
        block12.skip / skipbn
        conv3 + bn3  → ReLU  (SepConv 1536)
        conv4 + bn4  → ReLU  (SepConv 2048)
    Head:
        AdaptiveAvgPool(1×1) → Dropout → fc

SepConv sub-module attribute names (timm layout):
    conv1  — depthwise Conv2d (groups=in_ch)
    pointwise — pointwise Conv2d (1×1)
    (BN stored separately in the parent Sequential, not inside _SepConv)
"""

from dataclasses import dataclass
from typing import ClassVar, final, override

import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._utils._common import init_cnn_fan_out
from lucid.models._base import PretrainedModel
from lucid.models._tasks import ImageClassificationModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models.vision.xception._config import XceptionConfig

# ---------------------------------------------------------------------------
# Low-level SepConv primitive matching timm attribute names
# ---------------------------------------------------------------------------


@final
class _SepConvOp(nn.Module):
    """Depthwise + pointwise conv (no BN, no activation).

    Attribute names match timm legacy_xception layout:
      self.conv1      — depthwise Conv2d (groups=in_channels)
      self.pointwise  — pointwise Conv2d (1×1)
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        return self.pointwise(x)


# ---------------------------------------------------------------------------
# Entry flow block (block1 / block2 / block3)
# ---------------------------------------------------------------------------


@final
class _EntryBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, activate_first: bool) -> None:
        super().__init__()
        ops: list[nn.Module] = []
        if activate_first:
            # Not in place: the skip reads the block's input (timm).
            ops.append(nn.ReLU(inplace=False))
        # SepConv 1
        ops.append(_SepConvOp(in_channels, out_channels))
        ops.append(nn.BatchNorm2d(out_channels))
        ops.append(nn.ReLU(inplace=True))
        # SepConv 2
        ops.append(_SepConvOp(out_channels, out_channels))
        ops.append(nn.BatchNorm2d(out_channels))
        ops.append(nn.MaxPool2d(3, stride=2, padding=1))
        self.rep = nn.Sequential(*ops)
        self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=2, bias=False)
        self.skipbn = nn.BatchNorm2d(out_channels)

    @override
    def forward(self, x: Tensor) -> Tensor:
        residual = self.skipbn(self.skip(x))
        return self.rep(x) + residual


def _make_entry_block(in_channels: int, out_channels: int, *, activate_first: bool) -> _EntryBlock:
    """Return a block matching timm entry-flow key layout.

    timm key layout for block1 (activate_first=False):
      rep.0.conv1/pointwise  — SepConv1 depthwise/pointwise
      rep.1.*                — BN after SepConv1
      rep.2                  — ReLU  (no params, index 2)
      rep.3.conv1/pointwise  — SepConv2
      rep.4.*                — BN after SepConv2
      rep.5                  — MaxPool2d
      skip                   — 1×1 Conv (channel projection)
      skipbn                 — BN on skip

    timm key layout for block2/block3 (activate_first=True):
      rep.0                  — ReLU  (no params, index 0)
      rep.1.conv1/pointwise  — SepConv1
      rep.2.*                — BN
      rep.3                  — ReLU
      rep.4.conv1/pointwise  — SepConv2
      rep.5.*                — BN
      rep.6                  — MaxPool2d
    """
    return _EntryBlock(in_channels, out_channels, activate_first=activate_first)


# ---------------------------------------------------------------------------
# Middle flow block (block4 … block11)
# ---------------------------------------------------------------------------


@final
class _MiddleBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.rep = nn.Sequential(
            # Not in place: ``forward`` adds the block's input back.
            nn.ReLU(inplace=False),
            _SepConvOp(channels, channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            _SepConvOp(channels, channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            _SepConvOp(channels, channels),
            nn.BatchNorm2d(channels),
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.rep(x) + x


def _make_middle_block(channels: int) -> _MiddleBlock:
    """Return a block matching timm middle-flow key layout.

    timm key layout (example: block4):
      block4.rep.1.conv1  block4.rep.1.pointwise  block4.rep.2.*  — SepConv1
      block4.rep.4.conv1  block4.rep.4.pointwise  block4.rep.5.*  — SepConv2
      block4.rep.7.conv1  block4.rep.7.pointwise  block4.rep.8.*  — SepConv3

    The `rep` Sequential indices:
      0 — ReLU
      1 — _SepConvOp
      2 — BN
      3 — ReLU
      4 — _SepConvOp
      5 — BN
      6 — ReLU
      7 — _SepConvOp
      8 — BN
    """
    return _MiddleBlock(channels)


# ---------------------------------------------------------------------------
# Exit flow block (block12)
# ---------------------------------------------------------------------------


@final
class _ExitBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rep = nn.Sequential(
            # Not in place: the skip reads the block's input (timm).
            nn.ReLU(inplace=False),
            _SepConvOp(728, 728),
            nn.BatchNorm2d(728),
            nn.ReLU(inplace=True),
            _SepConvOp(728, 1024),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.skip = nn.Conv2d(728, 1024, 1, stride=2, bias=False)
        self.skipbn = nn.BatchNorm2d(1024)

    @override
    def forward(self, x: Tensor) -> Tensor:
        residual = self.skipbn(self.skip(x))
        return self.rep(x) + residual


def _make_exit_block() -> _ExitBlock:
    """Return a block matching timm exit-flow key layout.

    timm key layout (block12):
      block12.rep.1.conv1  block12.rep.1.pointwise  block12.rep.2.*  — SepConv(728)
      block12.rep.4.conv1  block12.rep.4.pointwise  block12.rep.5.*  — SepConv(1024)
      block12.skip         — 1×1 Conv 728→1024, stride=2
      block12.skipbn       — BN(1024)

    The `rep` Sequential indices:
      0 — ReLU
      1 — _SepConvOp(728, 728)
      2 — BN(728)
      3 — ReLU
      4 — _SepConvOp(728, 1024)
      5 — BN(1024)
      6 — MaxPool2d(3, stride=2, padding=1)
    """
    return _ExitBlock()


# ---------------------------------------------------------------------------
# SepConv used in exit conv3/conv4 (wraps _SepConvOp + BN; called separately)
# ---------------------------------------------------------------------------


@final
class _ExitSepConv(nn.Module):
    """Exit-flow final SepConv (conv3/conv4): dw+pw, no activation.

    BN stored separately as bn3/bn4 on the parent model (timm layout).
    Attribute names inside: conv1 (dw), pointwise (pw).
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        return self.pointwise(x)


# ---------------------------------------------------------------------------
# Xception output dataclass
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class XceptionOutput(ImageClassificationOutput):
    r"""Forward-return dataclass for :class:`XceptionForImageClassification`.

    Carries the per-class logits and the optional supervised loss
    computed when training labels are supplied to ``forward``.

    Attributes
    ----------
    logits : Tensor
        Pre-softmax classification scores of shape
        ``(batch_size, num_classes)``.  Apply :func:`lucid.softmax` or
        :func:`lucid.argmax` to convert to probabilities or hard
        predictions.
    loss : Tensor or None, optional, default=None
        Scalar cross-entropy loss; present only when ``labels`` were
        passed to the forward call.  ``None`` during inference.

    Notes
    -----
    Returned by :meth:`XceptionForImageClassification.forward` so the
    public API stays decoupled from raw tensor tuples — callers can
    address fields by name (``out.logits``) and ignore irrelevant
    entries.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.xception import xception_cls
    >>> model = xception_cls()
    >>> x = lucid.randn(2, 3, 299, 299)
    >>> out = model(x)
    >>> out.logits.shape
    (2, 1000)
    >>> out.loss is None
    True
    """


# ---------------------------------------------------------------------------
# Shared forward helper
# ---------------------------------------------------------------------------


class _XceptionBody:
    conv1: nn.Conv2d
    bn1: nn.BatchNorm2d
    conv2: nn.Conv2d
    bn2: nn.BatchNorm2d
    block1: _EntryBlock
    block2: _EntryBlock
    block3: _EntryBlock
    block4: _MiddleBlock
    block5: _MiddleBlock
    block6: _MiddleBlock
    block7: _MiddleBlock
    block8: _MiddleBlock
    block9: _MiddleBlock
    block10: _MiddleBlock
    block11: _MiddleBlock
    block12: _ExitBlock
    conv3: _ExitSepConv
    bn3: nn.BatchNorm2d
    conv4: _ExitSepConv
    bn4: nn.BatchNorm2d
    avgpool: nn.AdaptiveAvgPool2d


def _xception_forward_features(model: _XceptionBody, x: Tensor) -> Tensor:
    """Common feature-extraction path for Xception backbone and classifier."""
    # Stem
    x = F.relu(model.bn1(model.conv1(x)))
    x = F.relu(model.bn2(model.conv2(x)))

    # Entry flow
    x = model.block1(x)
    x = model.block2(x)
    x = model.block3(x)

    # Middle flow
    x = model.block4(x)
    x = model.block5(x)
    x = model.block6(x)
    x = model.block7(x)
    x = model.block8(x)
    x = model.block9(x)
    x = model.block10(x)
    x = model.block11(x)

    # Exit flow
    x = model.block12(x)
    x = F.relu(model.bn3(model.conv3(x)))
    x = F.relu(model.bn4(model.conv4(x)))

    # ``feature_info`` describes the pre-pool pyramid, and the mixin
    # documents this as returning the deepest stage's *feature map*.
    # Pooling here collapsed it to 1x1, so the declared reduction was
    # wrong by the input size and no consumer could reach a spatial map.
    # The classifier applies its own pooling.
    return x


def _build_xception_body(ic: int) -> dict[str, nn.Module]:
    """Return a dict of named sub-modules matching timm legacy_xception layout."""
    return {
        # Stem (timm: conv1.weight, bn1.*, conv2.weight, bn2.*).
        # Both stem convs are *valid* (padding=0) in the original design —
        # the 299×299 input is reduced to 149×149 (conv1, s2) then
        # 147×147 (conv2) — matching Inception-v3's stem schedule.
        "conv1": nn.Conv2d(ic, 32, 3, stride=2, bias=False),
        "bn1": nn.BatchNorm2d(32),
        "conv2": nn.Conv2d(32, 64, 3, bias=False),
        "bn2": nn.BatchNorm2d(64),
        # Entry flow
        "block1": _make_entry_block(64, 128, activate_first=False),
        "block2": _make_entry_block(128, 256, activate_first=True),
        "block3": _make_entry_block(256, 728, activate_first=True),
        # Middle flow (8 blocks: block4–block11)
        "block4": _make_middle_block(728),
        "block5": _make_middle_block(728),
        "block6": _make_middle_block(728),
        "block7": _make_middle_block(728),
        "block8": _make_middle_block(728),
        "block9": _make_middle_block(728),
        "block10": _make_middle_block(728),
        "block11": _make_middle_block(728),
        # Exit flow block
        "block12": _make_exit_block(),
        # Exit SepConvs (named conv3/conv4 with bn3/bn4 at model level)
        "conv3": _ExitSepConv(1024, 1536),
        "bn3": nn.BatchNorm2d(1536),
        "conv4": _ExitSepConv(1536, 2048),
        "bn4": nn.BatchNorm2d(2048),
        # Pooling
        "avgpool": nn.AdaptiveAvgPool2d((1, 1)),
    }


# ---------------------------------------------------------------------------
# Xception backbone (task="base")
# ---------------------------------------------------------------------------


class Xception(PretrainedModel, BackboneMixin, _XceptionBody):
    r"""Xception feature-extracting backbone (no classification head).

    Implements the "Extreme Inception" topology from Chollet,
    "Xception: Deep Learning with Depthwise Separable Convolutions",
    CVPR 2017 (arXiv:1610.02357).  The architecture replaces every
    Inception module of Inception-v3 with a depthwise separable
    convolution — the limiting case of an Inception block in which
    every output channel of the pointwise convolution receives its
    own independent spatial filter.  This makes the cross-channel
    and spatial correlations *fully decoupled*, dramatically
    reducing parameters and FLOPs:

    .. math::

        D_K^2 \cdot M \cdot N \;\longrightarrow\;
            M \cdot N \;+\; D_K^2 \cdot M.

    The body is organised into three phases — *entry flow* (3
    blocks, with 1×1 strided residuals and channel doubling from
    64 → 728), *middle flow* (8 identical 728-ch blocks with
    additive identity shortcuts), and *exit flow* (1 strided
    transition block plus two final separable convolutions
    expanding to 1536 → 2048 channels).  Designed for a 299×299
    input — the same crop size as Inception-v3 — so the
    spatial-reduction schedule isolates the depthwise-separable
    factorisation as the sole source of accuracy improvement.

    Parameters
    ----------
    config : XceptionConfig
        Frozen architecture spec.  Use the :func:`xception` factory
        for the paper-cited configuration.

    Attributes
    ----------
    config : XceptionConfig
        Stored copy of the config that built this model.
    conv1, conv2 : nn.Conv2d
        Two 3×3 stem convolutions; ``conv1`` carries stride 2.
    bn1, bn2 : nn.BatchNorm2d
        BatchNorm layers paired with the stem convs.
    block1, block2, block3 : nn.Module
        Entry-flow blocks — strided 1×1 residual shortcuts with
        channel projections 64→128, 128→256, 256→728.
    block4, …, block11 : nn.Module
        Eight identical middle-flow blocks with additive identity
        shortcuts at 728 channels.
    block12 : nn.Module
        Exit-flow strided transition block (728 → 1024).
    conv3, conv4 : _ExitSepConv
        Final two separable convolutions expanding to 1536 and
        2048 channels.
    bn3, bn4 : nn.BatchNorm2d
        BatchNorm layers paired with ``conv3`` and ``conv4``.
    avgpool : nn.AdaptiveAvgPool2d
        Global average pool collapsing the 2048-ch feature map to
        ``(B, 2048, 1, 1)``.
    feature_info : list[FeatureInfo]
        Per-stage descriptor exposed via :class:`BackboneMixin`.

    Notes
    -----
    Sub-module attribute names match timm's ``legacy_xception``
    layout so that state-dict round-trips are key-compatible with
    standard pretrained weight files.

    Examples
    --------
    Build an Xception backbone and run a forward pass at the
    native 299×299 resolution:

    >>> import lucid
    >>> from lucid.models.vision.xception import xception
    >>> backbone = xception()
    >>> x = lucid.randn(2, 3, 299, 299)
    >>> out = backbone(x)
    >>> out.last_hidden_state.shape   # 299 / 32, rounded up — not pooled
    (2, 2048, 10, 10)
    """

    config_class: ClassVar[type[XceptionConfig]] = XceptionConfig
    base_model_prefix: ClassVar[str] = "xception"

    def __init__(self, config: XceptionConfig) -> None:
        super().__init__(config)
        for name, module in _build_xception_body(config.in_channels).items():
            setattr(self, name, module)

        self._feature_info = [
            FeatureInfo(stage=1, num_channels=128, reduction=4),
            FeatureInfo(stage=2, num_channels=256, reduction=8),
            FeatureInfo(stage=3, num_channels=728, reduction=16),
            FeatureInfo(stage=4, num_channels=2048, reduction=32),
        ]

        # Reference initialisation: He/MSRA fan-out normal on convs, unit
        # norms.  Lucid's Conv2d default is kaiming_uniform(a=sqrt(5)) — a
        # different distribution *and* gain — so a from-scratch run would
        # otherwise start somewhere the paper's schedule was never tuned for.
        init_cnn_fan_out(self)

    @override
    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    @override
    def forward_features(self, x: Tensor) -> Tensor:
        return _xception_forward_features(self, x)

    @override
    def forward(self, x: Tensor) -> BaseModelOutput:
        return BaseModelOutput(last_hidden_state=self.forward_features(x))


# ---------------------------------------------------------------------------
# Xception for image classification (task="image-classification")
# ---------------------------------------------------------------------------


class XceptionForImageClassification(
    ImageClassificationModel, ClassificationHeadMixin, _XceptionBody
):
    r"""Xception with dropout + linear classification head.

    Combines an :class:`Xception` backbone with a dropout layer
    (probability ``config.dropout``, default 0.5 per the paper) and
    a linear projection to ``config.num_classes`` logits.  When
    ``labels`` are supplied to :meth:`forward`, a cross-entropy
    loss is computed and returned alongside the logits inside an
    :class:`XceptionOutput` dataclass.

    Parameters
    ----------
    config : XceptionConfig
        Architecture spec.  Use the :func:`xception_cls` factory for
        the paper-cited configuration.

    Attributes
    ----------
    config : XceptionConfig
        Stored copy of the config that built this model.
    conv1, conv2, bn1, bn2, block1, …, block12, conv3, conv4, bn3, bn4, avgpool
        Same backbone components as on :class:`Xception`; see that
        class for shape semantics.
    classifier : nn.Module
        Built by :meth:`ClassificationHeadMixin._build_classifier` —
        a :class:`~lucid.nn.Sequential` wrapping
        :class:`~lucid.nn.Dropout` and :class:`~lucid.nn.Linear`.

    Notes
    -----
    The classification flow is

    .. math::

        \text{logits} = W \,\operatorname{Drop}\!\left(
            \operatorname{GAP}(\,\mathrm{backbone}(x)\,)
        \right) + b.

    Chollet (2017) reports 79.0% top-1 ImageNet validation
    accuracy with this configuration at the 299×299 input
    resolution, edging out Inception-v3's 78.2% at the same
    parameter count (~22.9M).

    Examples
    --------
    Run inference at the native 299×299 input resolution:

    >>> import lucid
    >>> from lucid.models.vision.xception import xception_cls
    >>> model = xception_cls()
    >>> x = lucid.randn(4, 3, 299, 299)
    >>> out = model(x)
    >>> out.logits.shape
    (4, 1000)
    """

    config_class: ClassVar[type[XceptionConfig]] = XceptionConfig
    base_model_prefix: ClassVar[str] = "xception"

    def __init__(self, config: XceptionConfig) -> None:
        super().__init__(config)
        for name, module in _build_xception_body(config.in_channels).items():
            setattr(self, name, module)
        self._build_classifier(2048, config.num_classes, dropout=config.dropout)

        # Reference initialisation: He/MSRA fan-out normal on convs, unit
        # norms.  Lucid's Conv2d default is kaiming_uniform(a=sqrt(5)) — a
        # different distribution *and* gain — so a from-scratch run would
        # otherwise start somewhere the paper's schedule was never tuned for.
        init_cnn_fan_out(self)

    @override
    def forward(self, x: Tensor, labels: Tensor | None = None) -> XceptionOutput:
        pooled = self.avgpool(_xception_forward_features(self, x)).flatten(1)
        logits = self.classifier(pooled)
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return XceptionOutput(logits=logits, loss=loss)
