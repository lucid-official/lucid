"""CSPNet backbone + classifier — Wang et al., CVPRW 2020.

Paper: "CSPNet: A New Backbone that can Enhance Learning Capability of
CNN" (arXiv:1911.11929).

Three paper-cited variants ship from this module via
:mod:`lucid.models.vision.cspnet._pretrained`:

* CSPResNet-50  — ResNet-50 base wrapped in ``CrossStage``
* CSPResNeXt-50 — ResNeXt-50 base (groups=32) wrapped in ``CrossStage``
* CSPDarknet-53 — Darknet-53 base wrapped in ``CrossStage`` with the
  Darknet block (paper-cited Wang 2020 baseline used by YOLOv4)

Module + state-dict naming mirrors ``timm.models.cspnet`` exactly, so a
single converter handles all three variants with no per-arch
``map_key`` branches.
"""

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from typing import ClassVar, final, override

from lucid._tensor.tensor import Tensor
from lucid.models._utils._common import init_cnn_fan_out
from lucid.models._base import PretrainedModel
from lucid.models._tasks import ImageClassificationModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models.vision.cspnet._config import CSPNetConfig

# ---------------------------------------------------------------------------
# Conv-BN-Act primitive (mirrors timm's ``ConvNormAct``)
# ---------------------------------------------------------------------------


# darknet writes ``activation=leaky`` for slope 0.1, and this family was
# built to that spelling.  Every checkpoint it ships was trained somewhere
# else: all three come from timm, whose ``act_layer='leaky_relu'`` resolves
# to the framework default of 0.01.  Loading is unaffected — an activation
# has no weights — so the mismatch was invisible until the models were run
# side by side, where cspdarknet_53 agreed to 1.3e-02 at 0.1 and to 6.5e-09
# at 0.01.  The published weights decide it.
#
# A from-scratch darknet reproduction would want 0.1 back; that is the
# point at which this earns a config field rather than a constant.
_LEAKY_SLOPE: float = 0.01


class _ConvBnAct(nn.Module):
    """``Conv2d`` + ``BatchNorm2d`` + optional ``ReLU``.

    State-dict naming: ``conv.weight`` / ``bn.{weight, bias,
    running_mean, running_var, num_batches_tracked}`` — identical to
    timm.
    """

    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        apply_act: bool = True,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_chs, out_chs, kernel_size, stride=stride, padding=padding, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_chs)
        self.apply_act = apply_act

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        if self.apply_act:
            x = F.leaky_relu(x, negative_slope=_LEAKY_SLOPE)
        return x


# ---------------------------------------------------------------------------
# Residual bottleneck block — used by CSPResNet-50 / CSPResNeXt-50
# ---------------------------------------------------------------------------


@final
class _BottleneckBlock(nn.Module):
    """1×1 → 3×3 (grouped) → 1×1 with a residual.

    timm-faithful: ``conv1 → conv2 → attn2 → conv3 → attn3 →
    drop_path → + shortcut → act3``.  Paper-cited CSPNet variants do
    not use any attention layer, so ``attn2`` / ``attn3`` are
    ``Identity`` and ``drop_path`` likewise (kept as attributes only
    to preserve state-dict layout).
    """

    def __init__(
        self, in_chs: int, out_chs: int, bottle_ratio: float = 0.5, groups: int = 1
    ) -> None:
        super().__init__()
        mid = int(round(out_chs * bottle_ratio))
        self.conv1 = _ConvBnAct(in_chs, mid, kernel_size=1)
        self.conv2 = _ConvBnAct(mid, mid, kernel_size=3, groups=groups)
        self.attn2 = nn.Identity()
        self.conv3 = _ConvBnAct(mid, out_chs, kernel_size=1, apply_act=False)
        self.attn3 = nn.Identity()
        self.drop_path = nn.Identity()

    @override
    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attn2(x)
        x = self.conv3(x)
        x = self.attn3(x)
        x = self.drop_path(x) + shortcut
        return F.leaky_relu(x, negative_slope=_LEAKY_SLOPE)


# ---------------------------------------------------------------------------
# Darknet block — used by CSPDarknet-53
# ---------------------------------------------------------------------------


@final
class _DarkBlock(nn.Module):
    """1×1 → 3×3 with a residual (no final ReLU after addition).

    timm-faithful: ``conv1 → attn → conv2 → drop_path → + shortcut``.
    """

    def __init__(
        self, in_chs: int, out_chs: int, bottle_ratio: float = 0.5, groups: int = 1
    ) -> None:
        super().__init__()
        mid = int(round(out_chs * bottle_ratio))
        self.conv1 = _ConvBnAct(in_chs, mid, kernel_size=1)
        self.attn = nn.Identity()
        self.conv2 = _ConvBnAct(mid, out_chs, kernel_size=3, groups=groups)
        self.drop_path = nn.Identity()

    @override
    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.conv1(x)
        x = self.attn(x)
        x = self.conv2(x)
        return self.drop_path(x) + shortcut


# ---------------------------------------------------------------------------
# CrossStage — CSP-wrapped residual stage (paper §3.1)
# ---------------------------------------------------------------------------


def _zero_init_last(model: nn.Module) -> None:
    """Zero the last block BN gamma — ``conv3`` on bottlenecks, ``conv2`` on dark blocks."""
    for m in model.modules():
        if isinstance(m, _BottleneckBlock):
            bn = getattr(m.conv3, "bn", None)
            if bn is not None and bn.weight is not None:
                nn.init.constant_(bn.weight, 0.0)
        elif isinstance(m, _DarkBlock):
            bn = getattr(m.conv2, "bn", None)
            if bn is not None and bn.weight is not None:
                nn.init.constant_(bn.weight, 0.0)


@final
class _CrossStage(nn.Module):
    """CSP-wrapped stage: split / process / concat / project.

    timm forward (verbatim):

    .. code-block:: python

        x = conv_down(x)
        x = conv_exp(x)
        xs, xb = x.split(expand_chs // 2, dim=1)
        xb = blocks(xb)
        xb = conv_transition_b(xb)
        out = conv_transition(cat([xs, xb], dim=1))
        return out
    """

    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        stride: int,
        depth: int,
        block_type: str = "bottle",
        expand_ratio: float = 2.0,
        block_ratio: float = 1.0,
        bottle_ratio: float = 0.5,
        groups: int = 1,
        cross_linear: bool = True,
        down_growth: bool = False,
    ) -> None:
        super().__init__()
        # Channels at various points inside the stage.
        # ``down_growth=True`` (CSPDarknet) means ``conv_down`` is the
        # primary channel-growth step in this stage (in_chs → out_chs).
        # ``down_growth=False`` (CSPResNet / CSPResNeXt) means
        # ``conv_down`` preserves the input width — the expansion to
        # ``out_chs`` happens inside ``conv_exp``.
        down_chs = out_chs if down_growth else in_chs
        exp_chs = int(round(out_chs * expand_ratio))
        block_out_chs = int(round(out_chs * block_ratio))

        # ── conv_down ──
        # First stage in CSPResNet has stride=1 ⇒ ``conv_down`` collapses
        # to ``nn.Identity`` (matching timm's state-dict layout where
        # stages.0.conv_down has no parameters).  ``groups`` carries
        # through here so CSPResNeXt's 32-way group convolution is
        # applied at the downsample step as well as inside the blocks.
        self.conv_down: _ConvBnAct | nn.Identity
        if stride > 1:
            self.conv_down = _ConvBnAct(
                in_chs, down_chs, kernel_size=3, stride=stride, groups=groups
            )
        else:
            self.conv_down = nn.Identity()

        # ── conv_exp ──  (expanded input — split in half along channels)
        #
        # Divergence, deliberate: the authors' cfgs use an **asymmetric
        # split** in stage 1.  ``csresnet50.cfg`` applies two separate 1x1
        # convs to the same stem output at unequal widths (128 for the
        # bypass, 64 for the block branch), and the first block then widens
        # 64 -> 128, so darknet's shortcut adds across a channel-truncated
        # residual.  Here the split is even and the residual is full.
        # Following the weights rather than the cfg: the checkpoints this
        # family serves come from the reference implementation, which is
        # even-split throughout, and they load key-for-key only against
        # this shape.  A caller reproducing darknet's own trained weights
        # needs the asymmetric form, which is a different model.
        self.conv_exp = _ConvBnAct(down_chs, exp_chs, kernel_size=1, apply_act=not cross_linear)
        self.expand_chs = exp_chs

        # ── blocks ──  (only the "process" half goes through these)
        #
        # The reference widens on the *first* block (exp_chs//2 ->
        # block_out_chs) and keeps block_out_chs thereafter, then narrows
        # back in conv_transition_b.  Wiring every block at exp_chs//2 and
        # widening in the transition instead is the same network only while
        # ``block_ratio`` leaves block_out_chs == exp_chs//2 — which every
        # shipped config does, so this was invisible until someone overrode
        # ``block_ratio`` and got a differently-shaped stage with no error.
        block_in = exp_chs // 2
        blocks: list[nn.Module] = []
        for _ in range(depth):
            if block_type == "bottle":
                blocks.append(
                    _BottleneckBlock(
                        block_in, block_out_chs, bottle_ratio=bottle_ratio, groups=groups
                    )
                )
            else:
                blocks.append(
                    _DarkBlock(block_in, block_out_chs, bottle_ratio=bottle_ratio, groups=groups)
                )
            block_in = block_out_chs
        self.blocks = nn.Sequential(*blocks)

        # ── transitions ──
        self.conv_transition_b = _ConvBnAct(block_out_chs, exp_chs // 2, kernel_size=1)
        self.conv_transition = _ConvBnAct(exp_chs, out_chs, kernel_size=1)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_down(x)
        x = self.conv_exp(x)
        # Channel split — equal halves.
        half = self.expand_chs // 2
        xs = x[:, :half]
        xb = x[:, half:]
        xb = self.blocks(xb)
        xb = self.conv_transition_b(xb)
        return self.conv_transition(lucid.cat([xs, xb], dim=1))


# ---------------------------------------------------------------------------
# Stem
# ---------------------------------------------------------------------------


class _Stem(nn.Module):
    """Single-conv stem matching timm's ``stem.conv1`` + optional max-pool."""

    def __init__(
        self, in_channels: int, out_chs: int, kernel_size: int, stride: int, pool: str = "max"
    ) -> None:
        super().__init__()
        self.conv1 = _ConvBnAct(in_channels, out_chs, kernel_size, stride=stride)
        self.pool: nn.MaxPool2d | nn.Identity
        if pool == "max":
            # Divergence, deliberate: csresnet50.cfg / csresnext50.cfg specify
            # ``[maxpool] size=2 stride=2``, but the checkpoints this family
            # ships (``timm/cspresnet50.ra_in1k`` and siblings) were trained
            # with a 3x3/stride-2/pad-1 pool.  Both give the same output size
            # (224 -> 112 -> 56), so the weights load either way — but the
            # 2x2 pool sees a different receptive field and would silently
            # cost accuracy on every shipped checkpoint.  Following the
            # weights, not the cfg.
            self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        else:
            self.pool = nn.Identity()

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        return self.pool(x)


# ---------------------------------------------------------------------------
# Shared trunk builder
# ---------------------------------------------------------------------------


def _build_trunk(
    cfg: CSPNetConfig,
) -> tuple[_Stem, nn.ModuleList[_CrossStage], list[FeatureInfo], int]:
    """Returns ``(stem, stages, feature_info, final_chs)``."""
    stem = _Stem(
        cfg.in_channels,
        cfg.stem_out_chs,
        kernel_size=cfg.stem_kernel,
        stride=cfg.stem_stride,
        pool=cfg.stem_pool,
    )

    stages: list[_CrossStage] = []
    fi: list[FeatureInfo] = []

    # Reduction factor tracking — stem may have stride>1 and an optional pool.
    reduction = cfg.stem_stride * (2 if cfg.stem_pool == "max" else 1)

    in_chs = cfg.stem_out_chs
    for i in range(len(cfg.depths)):
        depth = cfg.depths[i]
        out_chs = cfg.out_chs[i]
        stride = cfg.strides[i]
        # All paper-cited CSPNet variants use ``CrossStage`` topology;
        # only ``block_type`` (``"bottle"`` vs ``"dark"``) differs.
        stage = _CrossStage(
            in_chs,
            out_chs,
            stride=stride,
            depth=depth,
            block_type=cfg.block_type[i],
            expand_ratio=cfg.expand_ratio[i],
            block_ratio=cfg.block_ratio[i],
            bottle_ratio=cfg.bottle_ratio[i],
            groups=cfg.groups[i],
            cross_linear=cfg.cross_linear[i],
            down_growth=cfg.down_growth[i],
        )
        stages.append(stage)
        reduction *= stride
        fi.append(FeatureInfo(stage=i + 1, num_channels=out_chs, reduction=reduction))
        in_chs = out_chs

    return stem, nn.ModuleList(stages), fi, in_chs


# ---------------------------------------------------------------------------
# CSPNet backbone  (task="base")
# ---------------------------------------------------------------------------


class CSPNet(PretrainedModel, BackboneMixin):
    r"""CSPNet feature-extracting backbone (Wang et al., CVPRW 2020).

    Builds a four-stage CSP-wrapped (or plain Darknet) trunk according to
    :class:`CSPNetConfig`.  Returns the final feature map without
    pooling.

    Examples
    --------
    The backbone, which returns a feature map rather than logits.

    >>> import lucid
    >>> from lucid.models.vision.cspnet import CSPNet, CSPNetConfig
    >>> model = CSPNet(CSPNetConfig()).eval()
    >>> out = model(lucid.randn(1, 3, 224, 224))
    >>> out.last_hidden_state.shape   # (B, C, H/32, W/32)
    (1, 1024, 7, 7)
    """

    config_class: ClassVar[type[CSPNetConfig]] = CSPNetConfig
    base_model_prefix: ClassVar[str] = "cspnet"

    def __init__(self, config: CSPNetConfig) -> None:
        super().__init__(config)
        stem, stages, fi, _final_chs = _build_trunk(config)
        self.stem = stem
        self.stages = stages
        self._feature_info = fi

        # Reference initialisation: He/MSRA fan-out normal on convs, unit
        # norms.  Lucid's Conv2d default is kaiming_uniform(a=sqrt(5)) — a
        # different distribution *and* gain — so a from-scratch run would
        # otherwise start somewhere the paper's schedule was never tuned for.
        init_cnn_fan_out(self, linear_std=0.01)
        if config.zero_init_last:
            _zero_init_last(self)

    @override
    def forward_features(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x

    @override
    def forward(self, x: Tensor) -> BaseModelOutput:
        return BaseModelOutput(last_hidden_state=self.forward_features(x))


# ---------------------------------------------------------------------------
# CSPNet for image classification  (task="image-classification")
# ---------------------------------------------------------------------------


class CSPNetForImageClassification(ImageClassificationModel, ClassificationHeadMixin):
    r"""CSPNet image classifier — backbone + GAP + dropout + linear head.

    Wraps a :class:`CSPNet` trunk and appends the timm-style
    ``ClassifierHead`` (global-average-pool, dropout, ``Linear``) so the
    whole pipeline returns ``(B, num_classes)`` logits.  Use this entry
    point when training or fine-tuning on classification datasets.

    Parameters
    ----------
    config : CSPNetConfig
        Hyperparameters including ``num_classes`` and head ``dropout``.

    Attributes
    ----------
    stem : nn.Module
        Conv-BN-act stem assembled from the config ``stem`` block.
    stages : nn.ModuleList
        The four CSP / Darknet stages emitting the ``C2..C5`` feature
        pyramid.
    avgpool : nn.AdaptiveAvgPool2d
        Global average pool over the final feature map.
    classifier : nn.Linear
        Final linear projection to ``num_classes`` (named ``head.fc`` to
        match the upstream timm layout for direct checkpoint loading).

    Notes
    -----
    Reference: Wang et al., *"CSPNet: A New Backbone that can Enhance
    Learning Capability of CNN"*, CVPRW 2020 (arXiv:1911.11929).

    Examples
    --------
    >>> from lucid.models import cspresnet_50_cls
    >>> model = cspresnet_50_cls()
    >>> import lucid
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """

    config_class: ClassVar[type[CSPNetConfig]] = CSPNetConfig
    base_model_prefix: ClassVar[str] = "cspnet"

    def __init__(self, config: CSPNetConfig) -> None:
        super().__init__(config)
        stem, stages, fi, final_chs = _build_trunk(config)
        self.stem = stem
        self.stages = stages
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Match timm naming: ``head.fc`` (Linear).
        self._build_classifier(final_chs, config.num_classes, dropout=config.dropout)

        # Reference initialisation: He/MSRA fan-out normal on convs, unit
        # norms.  Lucid's Conv2d default is kaiming_uniform(a=sqrt(5)) — a
        # different distribution *and* gain — so a from-scratch run would
        # otherwise start somewhere the paper's schedule was never tuned for.
        init_cnn_fan_out(self, linear_std=0.01)
        if config.zero_init_last:
            _zero_init_last(self)

    @override
    def forward(self, x: Tensor, labels: Tensor | None = None) -> ImageClassificationOutput:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.avgpool(x).flatten(1)
        return self._classify(x, labels)
