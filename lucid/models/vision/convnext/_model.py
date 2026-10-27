"""ConvNeXt backbone and classifier (Liu et al., 2022).

Paper: "A ConvNet for the 2020s"

Design recipe: systematically modernise ResNet by borrowing ideas from Swin:
  • Patchify stem (4×4 stride-4 conv, like ViT) instead of 7×7 stride-2
  • Stage ratios (3,3,9,3) matching Swin-T block counts
  • Inverted bottleneck MLP (4× expansion) borrowed from ViT
  • Depthwise 7×7 conv (large kernel, like shifted-window attention)
  • LayerNorm instead of BatchNorm; GELU instead of ReLU
  • Fewer normalisations (one LN per block, not BN-ReLU pairs)
  • Layer scale (per-channel γ initialised to 1e-6) for stable training

Architecture (ConvNeXt-T, dims=(96,192,384,768)):
  Stem     : Conv2d(4×4, stride=4) → LN                  → (56×56, 96)
  Stage 1  : 3 × ConvNeXtBlock(96)   → LN-Downsample(2×) → (28×28, 192)
  Stage 2  : 3 × ConvNeXtBlock(192)  → LN-Downsample(2×) → (14×14, 384)
  Stage 3  : 9 × ConvNeXtBlock(384)  → LN-Downsample(2×) → (7×7,  768)
  Stage 4  : 3 × ConvNeXtBlock(768)                       → (7×7,  768)
  Head     : AdaptiveAvgPool(1×1) → LN → FC

ConvNeXt block (all ops on (B, C, H, W)):
  DWConv(7×7, groups=C) → permute to (B,H,W,C)
  LN → Linear(C → 4C) → GELU → Linear(4C → C) → γ-scale
  permute back → + residual
"""

from typing import ClassVar, final, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._utils._common import init_transformer_trunc_normal
from lucid.models._base import PretrainedModel
from lucid.models._tasks import ImageClassificationModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models._utils._classification import DropPath
from lucid.models.vision.convnext._config import ConvNeXtConfig

# ---------------------------------------------------------------------------
# ConvNeXt block
# ---------------------------------------------------------------------------


@final
class _ConvNeXtBlock(nn.Module):
    """Depthwise 7×7 + inverted-bottleneck MLP + layer scale + DropPath."""

    def __init__(
        self,
        dim: int,
        layer_scale_init: float,
        drop_path_rate: float = 0.0,
        layer_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)
        # Layer scale is *optional*: the reference allocates gamma only when
        # the init value is positive and skips the multiply otherwise (the
        # paper disables it for the isotropic variants).  Allocating it
        # unconditionally turned ``layer_scale_init=0.0`` into a residual
        # branch multiplied by exactly zero — a permanently dead block, not a
        # disabled layer scale.
        self.gamma: nn.Parameter | None = (
            nn.Parameter(lucid.full((dim,), layer_scale_init)) if layer_scale_init > 0.0 else None
        )
        self.drop_path = DropPath(drop_path_rate)

    @override
    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.dwconv(x)  # (B, C, H, W)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm(x)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        if self.gamma is not None:
            x = x * self.gamma  # layer scale
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        return shortcut + self.drop_path(x)


# ---------------------------------------------------------------------------
# Downsampling between stages
# ---------------------------------------------------------------------------


class _Downsample(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, layer_norm_eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_dim, eps=layer_norm_eps)
        self.conv = nn.Conv2d(in_dim, out_dim, 2, stride=2)

    @override
    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, H, W) → norm in channel-last → back → strided conv
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Shared trunk builder
# ---------------------------------------------------------------------------


def _build_convnext(
    cfg: ConvNeXtConfig,
) -> tuple[
    _StemWithNorm,  # stem
    nn.ModuleList[nn.Sequential],  # stages
    nn.ModuleList[_Downsample],  # downsamplers (len = num_stages - 1)
    nn.LayerNorm,  # head norm
    list[FeatureInfo],
    int,  # final channels
]:
    # Patchify stem
    stem = nn.Sequential(
        nn.Conv2d(cfg.in_channels, cfg.dims[0], 4, stride=4),
        # LN applied in channel-last: use a wrapper
    )
    stem_norm = nn.LayerNorm(cfg.dims[0], eps=cfg.layer_norm_eps)

    stages: list[nn.Sequential] = []
    downsamplers: list[_Downsample] = []
    fi: list[FeatureInfo] = []
    reduction = 4

    # Linear stochastic-depth schedule across the entire trunk (Liu 2022 §3).
    total_blocks = sum(cfg.depths)
    if total_blocks > 1 and cfg.dropout > 0.0:
        dp_rates = [cfg.dropout * i / (total_blocks - 1) for i in range(total_blocks)]
    else:
        dp_rates = [cfg.dropout] * total_blocks
    block_cursor = 0

    for i, (depth, dim) in enumerate(zip(cfg.depths, cfg.dims, strict=True)):
        stage_blocks: list[nn.Module] = []
        for _ in range(depth):
            stage_blocks.append(
                _ConvNeXtBlock(
                    dim,
                    cfg.layer_scale_init,
                    dp_rates[block_cursor],
                    layer_norm_eps=cfg.layer_norm_eps,
                )
            )
            block_cursor += 1
        stage = nn.Sequential(*stage_blocks)
        stages.append(stage)
        fi.append(FeatureInfo(stage=i + 1, num_channels=dim, reduction=reduction))

        if i < len(cfg.depths) - 1:
            next_dim = cfg.dims[i + 1]
            downsamplers.append(_Downsample(dim, next_dim, layer_norm_eps=cfg.layer_norm_eps))
            reduction *= 2

    head_norm = nn.LayerNorm(cfg.dims[-1], eps=cfg.layer_norm_eps)

    # Wrap stem + stem_norm together
    full_stem = _StemWithNorm(stem, stem_norm)

    return (
        full_stem,
        nn.ModuleList(stages),
        nn.ModuleList(downsamplers),
        head_norm,
        fi,
        cfg.dims[-1],
    )


@final
class _StemWithNorm(nn.Module):
    def __init__(self, conv: nn.Sequential, norm: nn.LayerNorm) -> None:
        super().__init__()
        self.conv = conv
        self.norm = norm

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)  # (B, C, H, W)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)  # (B, C, H, W)


# ---------------------------------------------------------------------------
# ConvNeXt backbone  (task="base")
# ---------------------------------------------------------------------------


class ConvNeXt(PretrainedModel, BackboneMixin):
    r"""ConvNeXt backbone (Liu et al., 2022).

    ConvNeXt is a *pure-convolutional* backbone built by systematically
    modernising a ResNet using design choices borrowed from the Swin
    Transformer.  The trunk is a four-stage pyramid that starts with a
    non-overlapping :math:`4 \times 4` patchify stem and processes the
    feature map through stacks of identical *ConvNeXt blocks*:

    .. math::

        x \leftarrow x + \gamma \odot
            \mathrm{Linear}_{4d \to d}\!\bigl(
            \mathrm{GELU}\!\bigl(\mathrm{Linear}_{d \to 4d}\!\bigl(
            \mathrm{LN}(\mathrm{DWConv}_{7 \times 7}(x))
            \bigr)\bigr)\bigr),

    where :math:`\gamma \in \mathbb{R}^d` is a per-channel *layer-scale*
    parameter initialised to :math:`10^{-6}` (Touvron et al., 2021).
    Between stages a :math:`2 \times 2` stride-2 LN + Conv2d downsampler
    halves the resolution and doubles the channel width.

    :meth:`forward_features` returns a global-average-pooled
    :math:`(B, C_{\text{out}})` feature where
    :math:`C_{\text{out}} = \texttt{dims[-1]}`.  Use this backbone when
    you want features for transfer learning or dense prediction; for
    end-to-end classification use
    :class:`ConvNeXtForImageClassification`.

    Parameters
    ----------
    config : ConvNeXtConfig
        Frozen dataclass specifying ``depths``, ``dims``,
        ``layer_scale_init``, ``dropout``, and ``in_channels``.  See
        :class:`ConvNeXtConfig`.

    Attributes
    ----------
    stem : _StemWithNorm
        Patchify stem: :math:`4 \times 4` stride-4 Conv2d followed by a
        channel-last LayerNorm.
    stages : nn.ModuleList
        Four stages of stacked :class:`_ConvNeXtBlock` modules.
    downsamplers : nn.ModuleList
        Three between-stage LN + stride-2 Conv2d downsamplers.
    head_norm : nn.LayerNorm
        Final LayerNorm applied (channel-last) to the pooled feature.
    avgpool : nn.AdaptiveAvgPool2d
        :math:`1 \times 1` adaptive average pool over spatial dims.
    feature_info : list[FeatureInfo]
        Four-stage feature description with reductions
        :math:`(4, 8, 16, 32)` and channel counts from ``config.dims``.

    Notes
    -----
    Reference: Zhuang Liu *et al.*, *"A ConvNet for the 2020s"*,
    CVPR 2022, `arXiv:2201.03545 <https://arxiv.org/abs/2201.03545>`_.

    Examples
    --------
    Build a ConvNeXt-T backbone and run a forward pass:

    >>> import lucid
    >>> from lucid.models.vision.convnext import ConvNeXt, ConvNeXtConfig
    >>> model = ConvNeXt(ConvNeXtConfig())
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> feat = model.forward_features(x)
    >>> feat.shape                       # (B, dims[-1])
    (1, 768)
    >>> out = model(x)
    >>> out.last_hidden_state.shape      # (B, 1, dims[-1])
    (1, 1, 768)
    """

    config_class: ClassVar[type[ConvNeXtConfig]] = ConvNeXtConfig
    base_model_prefix: ClassVar[str] = "convnext"

    stem: _StemWithNorm

    def __init__(self, config: ConvNeXtConfig) -> None:
        super().__init__(config)
        stem, stages, downs, hn, fi, out_dim = _build_convnext(config)
        self.stem = stem
        self.stages = stages
        self.downsamplers = downs
        self.head_norm = hn
        self._feature_info = fi

        # Reference initialisation: trunc_normal_(0.02) on Linear / patch-embed
        # convs with zero bias, unit LayerNorms, and the same draw for the
        # class token and positional table (bare Parameters the framework
        # default never touches — they were sitting at exact zeros).
        init_transformer_trunc_normal(self)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # Per-stage maps from the most recent ``forward_features``.  The
        # mixin documents this as the way a backbone exposes the stages its
        # ``feature_info`` advertises; ConvNeXt pools before returning, so
        # without it the four entries at reductions 4/8/16/32 named nothing
        # a caller could actually obtain.
        self._stage_features: list[Tensor] = []

    @override
    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    @override
    def forward_features(self, x: Tensor) -> Tensor:
        """Pooled ``(B, C)`` feature vector, post head-norm.

        The per-stage maps this backbone's :attr:`feature_info` describes
        are left on :attr:`stage_features` for pyramid consumers.
        """
        x = self.stem(x)
        stages: list[Tensor] = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            stages.append(x)
            if i < len(self.downsamplers):
                x = self.downsamplers[i](x)
        self._stage_features = stages
        x = self.avgpool(x).flatten(1)  # (B, C)
        x = x.unsqueeze(-1).unsqueeze(-1)  # keep (B,C,1,1) for consistency
        # Apply head norm in channel-last
        x = x.permute(0, 2, 3, 1)
        x = self.head_norm(x)
        return x.permute(0, 3, 1, 2).flatten(1)  # (B, C)

    @property
    def stage_features(self) -> list[Tensor]:
        """The four stage outputs from the last :meth:`forward_features`.

        Ordered stem-to-deepest, so entry ``i`` corresponds to
        ``feature_info[i]`` — reductions 4, 8, 16 and 32.  Empty until a
        forward pass has run.
        """
        return list(self._stage_features)

    @override
    def forward(self, x: Tensor) -> BaseModelOutput:
        feat = self.forward_features(x)
        return BaseModelOutput(last_hidden_state=feat.unsqueeze(1))


# ---------------------------------------------------------------------------
# ConvNeXt for image classification  (task="image-classification")
# ---------------------------------------------------------------------------


class ConvNeXtForImageClassification(ImageClassificationModel, ClassificationHeadMixin):
    r"""ConvNeXt with a linear classification head (Liu et al., 2022).

    Wraps the same four-stage trunk as :class:`ConvNeXt` (patchify stem
    → four stages of ConvNeXt blocks with between-stage downsamplers)
    and adds a global average pool, a final channel-last LayerNorm, and
    a single :class:`nn.Linear` classification head:

    .. math::

        \text{logits} = W_{\text{head}}\,
            \mathrm{LN}(\mathrm{GAP}(z^{L})) + b_{\text{head}},
        \qquad W_{\text{head}} \in
            \mathbb{R}^{C_{\text{out}} \times d_{L}}.

    Pass ``labels`` to :meth:`forward` to compute the cross-entropy
    loss in the same pass.

    Parameters
    ----------
    config : ConvNeXtConfig
        Architecture specification.  Must set ``num_classes`` to the
        desired number of output categories (default 1000 for
        ImageNet-1k).  See :class:`ConvNeXtConfig`.

    Attributes
    ----------
    stem : _StemWithNorm
        Patchify stem: :math:`4 \times 4` stride-4 Conv2d + LayerNorm.
    stages : nn.ModuleList
        Four stages of stacked ConvNeXt blocks.
    downsamplers : nn.ModuleList
        Three between-stage LN + stride-2 Conv2d downsamplers.
    head_norm : nn.LayerNorm
        Final LayerNorm applied to the pooled feature.
    avgpool : nn.AdaptiveAvgPool2d
        :math:`1 \times 1` adaptive average pool.
    classifier : nn.Linear
        Final linear projection of width ``(num_classes, dims[-1])``.

    Notes
    -----
    Reference: Zhuang Liu *et al.*, *"A ConvNet for the 2020s"*,
    CVPR 2022.  ConvNeXt-T / S / B / L reach **82.1 / 83.1 / 83.8 /
    84.3 % top-1 on ImageNet-1k** (Table 1 upper block, 224x224) —
    matching or exceeding Swin at equal FLOPs.  There is no
    ImageNet-1k-only XL row: XL appears only among the 22k-pretrained
    models (87.0 at 224x224).  The 85.5 previously attributed to XL is
    ConvNeXt-L at 384x384.

    Examples
    --------
    End-to-end inference with the default ConvNeXt-T classifier:

    >>> import lucid
    >>> from lucid.models.vision.convnext import (
    ...     ConvNeXtConfig, ConvNeXtForImageClassification,
    ... )
    >>> model = ConvNeXtForImageClassification(ConvNeXtConfig())
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """

    config_class: ClassVar[type[ConvNeXtConfig]] = ConvNeXtConfig
    base_model_prefix: ClassVar[str] = "convnext"

    stem: _StemWithNorm

    def __init__(self, config: ConvNeXtConfig) -> None:
        super().__init__(config)
        stem, stages, downs, hn, _, out_dim = _build_convnext(config)
        self.stem = stem
        self.stages = stages
        self.downsamplers = downs
        self.head_norm = hn
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self._build_classifier(out_dim, config.num_classes)
        init_transformer_trunc_normal(self)

    @override
    def forward(
        self,
        x: Tensor,
        labels: Tensor | None = None,
    ) -> ImageClassificationOutput:
        x = self.stem(x)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.downsamplers):
                x = self.downsamplers[i](x)
        x = self.avgpool(x).flatten(1)
        # head norm in channel-last
        x = x.unsqueeze(-1).unsqueeze(-1).permute(0, 2, 3, 1)
        x = self.head_norm(x)
        x = x.flatten(1)
        logits = self.classifier(x)

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ImageClassificationOutput(logits=logits, loss=loss)
