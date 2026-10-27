"""MobileNet v4 backbone and classifier (Qin et al., 2024).

Paper: "MobileNetV4: Universal Models for the Mobile Ecosystem"

Key ideas:
  - Universal Inverted Bottleneck (UIB): an inverted bottleneck with two
    optional depthwise convolutions, which recovers IB, ConvNeXt-like, FFN
    and the new ExtraDW block as special cases.
  - Fused inverted bottleneck in the stem (dense 3x3 conv -> 1x1 conv).
  - Mobile MQA (hybrid variants): multi-query attention whose single key /
    value head can be spatially reduced by a stride-2 depthwise conv.
  - MobileNet-v3 style head: global pool before the widest 1x1 layer.

The module tree mirrors the reference implementation's layout, so the
published checkpoints load with an identity key map:

  conv_stem, bn1                      stem conv + norm
  blocks.{stage}.{index}.*            five stages of searched blocks
      FusedIB  : conv_exp, bn1, conv_pwl, bn2
      UIB      : dw_start, pw_exp, dw_mid, pw_proj (each .conv / .bn),
                 layer_scale.gamma (hybrids)
      MQA      : norm, attn.{query,key,value,output}.*, layer_scale.gamma
      Conv     : conv, bn1
  conv_head, norm_head, classifier    post-pool head (classifier only)
"""

from collections.abc import Callable
from dataclasses import dataclass
from math import sqrt
from typing import ClassVar, Literal, override

import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models._tasks import ImageClassificationModel
from lucid.models._utils._classification import DropPath, LayerScale
from lucid.models._utils._common import make_divisible
from lucid.models.vision.mobilenet_v4._config import MobileNetV4Config

# ---------------------------------------------------------------------------
# Architecture specification
# ---------------------------------------------------------------------------

_BlockKind = Literal["conv", "fused", "uib", "mqa"]


@dataclass(frozen=True)
class _Spec:
    """One block of the searched sequence.

    ``kernel`` is the dense kernel of a ``conv`` / ``fused`` block;
    ``start`` / ``mid`` are the UIB's optional depthwise kernels (0 = off);
    ``heads`` / ``key_dim`` / ``kv_stride`` describe a Mobile MQA block.
    """

    kind: _BlockKind
    out: int
    stride: int = 1
    kernel: int = 3
    exp: float = 1.0
    start: int = 0
    mid: int = 0
    heads: int = 0
    key_dim: int = 0
    kv_stride: int = 1


def _conv(out: int, kernel: int, stride: int = 1) -> list[_Spec]:
    return [_Spec("conv", out, stride=stride, kernel=kernel)]


def _fused(out: int, stride: int, exp: float) -> list[_Spec]:
    return [_Spec("fused", out, stride=stride, kernel=3, exp=exp)]


def _uib(out: int, start: int, mid: int, stride: int, exp: float, repeat: int = 1) -> list[_Spec]:
    # A repeated group strides only on its first block.
    return [
        _Spec("uib", out, stride=stride if i == 0 else 1, exp=exp, start=start, mid=mid)
        for i in range(repeat)
    ]


def _mqa(out: int, heads: int, key_dim: int, kv_stride: int) -> list[_Spec]:
    return [_Spec("mqa", out, heads=heads, key_dim=key_dim, kv_stride=kv_stride)]


def _stage(*groups: list[_Spec]) -> tuple[_Spec, ...]:
    return tuple(spec for group in groups for spec in group)


@dataclass(frozen=True)
class _Arch:
    stem: int
    act: Literal["relu", "gelu"]
    layer_scale: float | None
    stages: tuple[tuple[_Spec, ...], ...]


# The block tables below are the paper's Appendix D (Tables 11-15) as the
# authors released them.  UIB instantiations: ``start`` and ``mid`` both
# set = ExtraDW, ``mid`` only = IB, ``start`` only = ConvNeXt-like, neither
# = FFN.  Conv-Small's two stem "FusedIB" rows are released as a pair of
# conv-norm-act layers (the 1x1 carries an activation and no residual),
# which is what the checkpoints hold.

_CONV_SMALL = _Arch(
    stem=32,
    act="relu",
    layer_scale=None,
    stages=(
        _stage(_conv(32, 3, 2), _conv(32, 1)),
        _stage(_conv(96, 3, 2), _conv(64, 1)),
        _stage(
            _uib(96, 5, 5, 2, 3.0),
            _uib(96, 0, 3, 1, 2.0, repeat=4),
            _uib(96, 3, 0, 1, 4.0),
        ),
        _stage(
            _uib(128, 3, 3, 2, 6.0),
            _uib(128, 5, 5, 1, 4.0),
            _uib(128, 0, 5, 1, 4.0),
            _uib(128, 0, 5, 1, 3.0),
            _uib(128, 0, 3, 1, 4.0, repeat=2),
        ),
        _stage(_conv(960, 1)),
    ),
)

_CONV_MEDIUM = _Arch(
    stem=32,
    act="relu",
    layer_scale=None,
    stages=(
        _stage(_fused(48, 2, 4.0)),
        _stage(_uib(80, 3, 5, 2, 4.0), _uib(80, 3, 3, 1, 2.0)),
        _stage(
            _uib(160, 3, 5, 2, 6.0),
            _uib(160, 3, 3, 1, 4.0, repeat=2),
            _uib(160, 3, 5, 1, 4.0),
            _uib(160, 3, 3, 1, 4.0),
            _uib(160, 3, 0, 1, 4.0),
            _uib(160, 0, 0, 1, 2.0),
            _uib(160, 3, 0, 1, 4.0),
        ),
        _stage(
            _uib(256, 5, 5, 2, 6.0),
            _uib(256, 5, 5, 1, 4.0),
            _uib(256, 3, 5, 1, 4.0, repeat=2),
            _uib(256, 0, 0, 1, 4.0),
            _uib(256, 3, 0, 1, 4.0),
            _uib(256, 3, 5, 1, 2.0),
            _uib(256, 5, 5, 1, 4.0),
            _uib(256, 0, 0, 1, 4.0, repeat=2),
            _uib(256, 5, 0, 1, 2.0),
        ),
        _stage(_conv(960, 1)),
    ),
)

_CONV_LARGE = _Arch(
    stem=24,
    act="relu",
    layer_scale=None,
    stages=(
        _stage(_fused(48, 2, 4.0)),
        _stage(_uib(96, 3, 5, 2, 4.0), _uib(96, 3, 3, 1, 4.0)),
        _stage(
            _uib(192, 3, 5, 2, 4.0),
            _uib(192, 3, 3, 1, 4.0, repeat=3),
            _uib(192, 3, 5, 1, 4.0),
            _uib(192, 5, 3, 1, 4.0, repeat=5),
            _uib(192, 3, 0, 1, 4.0),
        ),
        _stage(
            _uib(512, 5, 5, 2, 4.0, repeat=4),
            _uib(512, 5, 0, 1, 4.0),
            _uib(512, 5, 3, 1, 4.0),
            _uib(512, 5, 0, 1, 4.0, repeat=2),
            _uib(512, 5, 3, 1, 4.0),
            _uib(512, 5, 5, 1, 4.0),
            _uib(512, 5, 0, 1, 4.0, repeat=3),
        ),
        _stage(_conv(960, 1)),
    ),
)

_HYBRID_MEDIUM = _Arch(
    stem=32,
    act="relu",
    layer_scale=1e-5,
    stages=(
        _stage(_fused(48, 2, 4.0)),
        _stage(_uib(80, 3, 5, 2, 4.0), _uib(80, 3, 3, 1, 2.0)),
        _stage(
            _uib(160, 3, 5, 2, 6.0),
            _uib(160, 0, 0, 1, 2.0),
            _uib(160, 3, 3, 1, 4.0),
            _uib(160, 3, 5, 1, 4.0),
            _mqa(160, 4, 64, 2),
            _uib(160, 3, 3, 1, 4.0),
            _mqa(160, 4, 64, 2),
            _uib(160, 3, 0, 1, 4.0),
            _mqa(160, 4, 64, 2),
            _uib(160, 3, 3, 1, 4.0),
            _mqa(160, 4, 64, 2),
            _uib(160, 3, 0, 1, 4.0),
        ),
        _stage(
            _uib(256, 5, 5, 2, 6.0),
            _uib(256, 5, 5, 1, 4.0),
            _uib(256, 3, 5, 1, 4.0, repeat=2),
            _uib(256, 0, 0, 1, 2.0),
            _uib(256, 3, 5, 1, 2.0),
            _uib(256, 0, 0, 1, 2.0),
            _uib(256, 0, 0, 1, 4.0),
            _mqa(256, 4, 64, 1),
            _uib(256, 3, 0, 1, 4.0),
            _mqa(256, 4, 64, 1),
            _uib(256, 5, 5, 1, 4.0),
            _mqa(256, 4, 64, 1),
            _uib(256, 5, 0, 1, 4.0),
            _mqa(256, 4, 64, 1),
            _uib(256, 5, 0, 1, 4.0),
        ),
        _stage(_conv(960, 1)),
    ),
)

_HYBRID_LARGE = _Arch(
    stem=24,
    act="gelu",
    layer_scale=1e-5,
    stages=(
        _stage(_fused(48, 2, 4.0)),
        _stage(_uib(96, 3, 5, 2, 4.0), _uib(96, 3, 3, 1, 4.0)),
        _stage(
            _uib(192, 3, 5, 2, 4.0),
            _uib(192, 3, 3, 1, 4.0, repeat=3),
            _uib(192, 3, 5, 1, 4.0),
            _uib(192, 5, 3, 1, 4.0, repeat=2),
            _mqa(192, 8, 48, 2),
            _uib(192, 5, 3, 1, 4.0),
            _mqa(192, 8, 48, 2),
            _uib(192, 5, 3, 1, 4.0),
            _mqa(192, 8, 48, 2),
            _uib(192, 5, 3, 1, 4.0),
            _mqa(192, 8, 48, 2),
            _uib(192, 3, 0, 1, 4.0),
        ),
        _stage(
            _uib(512, 5, 5, 2, 4.0, repeat=4),
            _uib(512, 5, 0, 1, 4.0),
            _uib(512, 5, 3, 1, 4.0),
            _uib(512, 5, 0, 1, 4.0, repeat=2),
            _uib(512, 5, 3, 1, 4.0),
            _uib(512, 5, 5, 1, 4.0),
            _mqa(512, 8, 64, 1),
            _uib(512, 5, 0, 1, 4.0),
            _mqa(512, 8, 64, 1),
            _uib(512, 5, 0, 1, 4.0),
            _mqa(512, 8, 64, 1),
            _uib(512, 5, 0, 1, 4.0),
            _mqa(512, 8, 64, 1),
            _uib(512, 5, 0, 1, 4.0),
        ),
        _stage(_conv(960, 1)),
    ),
)

_ARCHS: dict[str, _Arch] = {
    "conv_small": _CONV_SMALL,
    "conv_medium": _CONV_MEDIUM,
    "conv_large": _CONV_LARGE,
    "hybrid_medium": _HYBRID_MEDIUM,
    "hybrid_large": _HYBRID_LARGE,
}

#: Width of the post-pool 1x1 layer, shared by all five variants.
_HEAD_WIDTH = 1280


def _chain(x: Tensor, *layers: Callable[[Tensor], Tensor]) -> Tensor:
    """Apply ``layers`` in order (each returns a single tensor)."""
    for layer in layers:
        x = layer(x)
    return x


def _make_act(kind: Literal["relu", "gelu"]) -> nn.ReLU | nn.GELU:
    return nn.GELU() if kind == "gelu" else nn.ReLU()


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _ConvNorm(nn.Module):
    """Conv -> BatchNorm -> optional activation (``conv`` / ``bn``)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        *,
        stride: int = 1,
        groups: int = 1,
        act: nn.ReLU | nn.GELU | None = None,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel,
            stride=stride,
            padding=kernel // 2,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = act if act is not None else nn.Identity()

    @override
    def forward(self, x: Tensor) -> Tensor:
        return _chain(x, self.conv, self.bn, self.act)


class _ConvBlock(nn.Module):
    """Plain conv-norm-act layer, no residual (the ``Conv2D`` table rows)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        stride: int,
        act: Literal["relu", "gelu"],
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act = _make_act(act)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return _chain(x, self.conv, self.bn1, self.act)


class _FusedIB(nn.Module):
    """Fused inverted bottleneck: dense k x k expansion -> linear 1x1."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        stride: int,
        exp: float,
        act: Literal["relu", "gelu"],
        drop_path: float,
    ) -> None:
        super().__init__()
        mid = make_divisible(in_ch * exp)
        self._has_skip = stride == 1 and in_ch == out_ch
        self.conv_exp = nn.Conv2d(
            in_ch, mid, kernel, stride=stride, padding=kernel // 2, bias=False
        )
        self.bn1 = nn.BatchNorm2d(mid)
        self.act = _make_act(act)
        self.conv_pwl = nn.Conv2d(mid, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.drop_path = DropPath(drop_path)

    @override
    def forward(self, x: Tensor) -> Tensor:
        y = _chain(x, self.conv_exp, self.bn1, self.act, self.conv_pwl, self.bn2)
        if self._has_skip:
            y = self.drop_path(y) + x
        return y


class _UIB(nn.Module):
    """Universal Inverted Bottleneck (Section 4, Fig. 4).

    ``start`` / ``mid`` are the kernels of the optional depthwise layers
    before and after the expansion (0 disables one).  The stride sits on
    the middle depthwise when present, otherwise on the starting one.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        start: int,
        mid: int,
        stride: int,
        exp: float,
        act: Literal["relu", "gelu"],
        layer_scale: float | None,
        drop_path: float,
    ) -> None:
        super().__init__()
        if stride > 1 and not (start or mid):
            raise ValueError("a strided UIB needs at least one depthwise conv")
        self._has_skip = stride == 1 and in_ch == out_ch
        mid_ch = make_divisible(in_ch * exp)

        self.dw_start: _ConvNorm | nn.Identity = (
            _ConvNorm(in_ch, in_ch, start, stride=1 if mid else stride, groups=in_ch)
            if start
            else nn.Identity()
        )
        self.pw_exp = _ConvNorm(in_ch, mid_ch, 1, act=_make_act(act))
        self.dw_mid: _ConvNorm | nn.Identity = (
            _ConvNorm(mid_ch, mid_ch, mid, stride=stride, groups=mid_ch, act=_make_act(act))
            if mid
            else nn.Identity()
        )
        self.pw_proj = _ConvNorm(mid_ch, out_ch, 1)
        self.layer_scale: LayerScale | nn.Identity = (
            LayerScale(out_ch, layer_scale) if layer_scale is not None else nn.Identity()
        )
        self.drop_path = DropPath(drop_path)

    @override
    def forward(self, x: Tensor) -> Tensor:
        y = _chain(x, self.dw_start, self.pw_exp, self.dw_mid, self.pw_proj, self.layer_scale)
        if self._has_skip:
            y = self.drop_path(y) + x
        return y


class _KVProjection(nn.Module):
    """Key or value path of Mobile MQA: optional spatial reduction -> 1x1."""

    def __init__(self, dim: int, out_dim: int, kv_stride: int) -> None:
        super().__init__()
        if kv_stride > 1:
            # SR(X): a stride-2 3x3 depthwise conv in place of average
            # pooling (Section 5, Eq. 2).
            self.down_conv: nn.Conv2d | nn.Identity = nn.Conv2d(
                dim, dim, 3, stride=kv_stride, padding=1, groups=dim, bias=False
            )
            self.norm: nn.BatchNorm2d | nn.Identity = nn.BatchNorm2d(dim)
        else:
            self.down_conv = nn.Identity()
            self.norm = nn.Identity()
        self.proj = nn.Conv2d(dim, out_dim, 1, bias=False)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return _chain(x, self.down_conv, self.norm, self.proj)


class _Projection(nn.Module):
    """A bias-free 1x1 projection held under the name ``proj``."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_dim, 1, bias=False)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class _MultiQueryAttention(nn.Module):
    """Multi-query attention over a 2-D feature map (Eq. 2).

    ``heads`` query heads share one key head and one value head of width
    ``key_dim``.  Because the key/value head is shared, the attention of
    all heads is one batched product: the queries are laid out as
    ``(B, heads * HW, key_dim)`` against keys of ``(B, M, key_dim)``.
    """

    def __init__(self, dim: int, out_dim: int, heads: int, key_dim: int, kv_stride: int) -> None:
        super().__init__()
        self.heads = heads
        self.key_dim = key_dim
        self.scale = key_dim**-0.5
        self.query = _Projection(dim, heads * key_dim)
        self.key = _KVProjection(dim, key_dim, kv_stride)
        self.value = _KVProjection(dim, key_dim, kv_stride)
        self.output = _Projection(heads * key_dim, out_dim)

    @override
    def forward(self, x: Tensor) -> Tensor:
        b, _, h, w = x.shape
        n = h * w
        heads, d = self.heads, self.key_dim

        # (B, heads*d, H, W) -> (B, heads, HW, d) -> (B, heads*HW, d)
        q = self.query(x).reshape(b, heads, d, n)
        q = q.permute(0, 1, 3, 2).reshape(b, heads * n, d)
        k = self.key(x)
        k = k.reshape(b, d, -1).permute(0, 2, 1)  # (B, M, d)
        v = self.value(x)
        v = v.reshape(b, d, -1).permute(0, 2, 1)  # (B, M, d)

        o = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        # (B, heads*HW, d) -> (B, HW, heads, d) -> (B, heads*d, H, W)
        o = o.reshape(b, heads, n, d).permute(0, 2, 1, 3)
        o = o.reshape(b, h, w, heads * d).permute(0, 3, 1, 2)
        return self.output(o)


class _MobileMQA(nn.Module):
    """Mobile MQA block: BatchNorm -> multi-query attention -> residual."""

    def __init__(
        self,
        dim: int,
        heads: int,
        key_dim: int,
        kv_stride: int,
        layer_scale: float | None,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(dim)
        self.attn = _MultiQueryAttention(dim, dim, heads, key_dim, kv_stride)
        self.layer_scale: LayerScale | nn.Identity = (
            LayerScale(dim, layer_scale) if layer_scale is not None else nn.Identity()
        )
        self.drop_path = DropPath(drop_path)

    @override
    def forward(self, x: Tensor) -> Tensor:
        y = _chain(x, self.norm, self.attn, self.layer_scale)
        return self.drop_path(y) + x


# ---------------------------------------------------------------------------
# Trunk construction and initialisation
# ---------------------------------------------------------------------------


def _build_trunk(
    cfg: MobileNetV4Config,
) -> tuple[
    nn.Conv2d,
    nn.BatchNorm2d,
    nn.ReLU | nn.GELU,
    nn.Sequential,
    list[FeatureInfo],
]:
    """Return ``(conv_stem, bn1, act, blocks, feature_info)`` for ``cfg``."""
    arch = _ARCHS[cfg.variant]
    conv_stem = nn.Conv2d(cfg.in_channels, arch.stem, 3, stride=2, padding=1, bias=False)
    bn1 = nn.BatchNorm2d(arch.stem)
    act = _make_act(arch.act)

    total = sum(len(stage) for stage in arch.stages)
    index = 0
    in_ch = arch.stem
    reduction = 2
    stages: list[nn.Module] = []
    # One entry per resolution; the last stage (a 1x1 widening conv) does
    # not change resolution and overwrites the stride-32 entry, since that
    # widened map is what ``forward_features`` returns.
    per_reduction: dict[int, int] = {}
    for stage in arch.stages:
        blocks: list[nn.Module] = []
        for spec in stage:
            dpr = cfg.drop_path_rate * index / total
            block: nn.Module
            if spec.kind == "conv":
                block = _ConvBlock(in_ch, spec.out, spec.kernel, spec.stride, arch.act)
            elif spec.kind == "fused":
                block = _FusedIB(in_ch, spec.out, spec.kernel, spec.stride, spec.exp, arch.act, dpr)
            elif spec.kind == "uib":
                block = _UIB(
                    in_ch,
                    spec.out,
                    spec.start,
                    spec.mid,
                    spec.stride,
                    spec.exp,
                    arch.act,
                    arch.layer_scale,
                    dpr,
                )
            else:
                block = _MobileMQA(
                    in_ch,
                    spec.heads,
                    spec.key_dim,
                    spec.kv_stride,
                    arch.layer_scale,
                    dpr,
                )
            blocks.append(block)
            reduction *= spec.stride
            in_ch = spec.out
            index += 1
        per_reduction[reduction] = in_ch
        stages.append(nn.Sequential(*blocks))

    feature_info = [
        FeatureInfo(stage=i, num_channels=ch, reduction=r)
        for i, (r, ch) in enumerate(sorted(per_reduction.items()))
    ]
    return conv_stem, bn1, act, nn.Sequential(*stages), feature_info


def _init_weights(model: nn.Module) -> None:
    r"""Reference initialisation of the released implementation.

    Convolutions draw from :math:`\mathcal{N}(0, 2 / \text{fan-out})` with
    the fan-out divided by ``groups`` — so a depthwise ``k x k`` kernel has
    fan-out ``k^2``, not ``k^2 C`` — which is the TPU MobileNet convention
    and wider than the framework's He fan-out for depthwise layers.  Norms
    start at identity, the classifier at
    :math:`\mathcal{U}(\pm 1/\sqrt{\text{out}})`, and layer scales keep
    their :math:`10^{-5}`.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            kh, kw = m.kernel_size
            fan_out = kh * kw * m.out_channels // m.groups
            nn.init.normal_(m.weight, mean=0.0, std=sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            bound = 1.0 / sqrt(m.out_features)
            nn.init.uniform_(m.weight, -bound, bound)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# MobileNet v4 backbone  (task="base")
# ---------------------------------------------------------------------------


class MobileNetV4(PretrainedModel, BackboneMixin):
    r"""MobileNet-v4 feature-extracting backbone (no classification head).

    Implements the searched architectures of Qin et al., "MobileNetV4:
    Universal Models for the Mobile Ecosystem", ECCV 2024
    (arXiv:2404.10518).  The trunk is a stride-2 stem followed by five
    stages built from the **Universal Inverted Bottleneck** (UIB), whose
    two optional depthwise convolutions let the architecture search pick
    an inverted bottleneck, a ConvNeXt-like block, an FFN or the new
    ExtraDW block at every position; a fused inverted bottleneck opens the
    medium and large models, and the hybrid variants interleave **Mobile
    MQA** attention blocks into the two deepest stages.  The last stage is
    a :math:`1 \times 1` convolution widening the stride-32 map to 960
    channels.

    Parameters
    ----------
    config : MobileNetV4Config
        Architecture spec.  Use the factory functions
        (:func:`mobilenet_v4_conv_small`, :func:`mobilenet_v4_conv_medium`,
        :func:`mobilenet_v4_conv_large`, :func:`mobilenet_v4_hybrid_medium`,
        :func:`mobilenet_v4_hybrid_large`) for the paper's variants.

    Attributes
    ----------
    config : MobileNetV4Config
        The config that built this model.
    conv_stem : nn.Conv2d
        Stride-2 :math:`3 \times 3` stem convolution (24 or 32 channels).
    bn1 : nn.BatchNorm2d
        Stem normalisation.
    act : nn.Module
        Stem activation (ReLU, or GELU for Hybrid-Large).
    blocks : nn.Sequential
        Five stage ``Sequential`` containers; stage :math:`i` ends at stride
        :math:`2^{i+2}` for :math:`i \le 3`, and stage 4 is the 960-channel
        :math:`1 \times 1` widening layer.
    feature_info : list[FeatureInfo]
        One entry per output resolution (strides 4, 8, 16 and 32); the
        stride-32 entry reports the 960 channels that
        :meth:`forward_features` returns.

    Notes
    -----
    The UIB computes

    .. math::

        y = x + \gamma \odot
            P\bigl(D_{\mathrm{mid}}(E(D_{\mathrm{start}}(x)))\bigr),

    with :math:`E` the pointwise expansion (norm + activation), :math:`P`
    the linear pointwise projection, each :math:`D` an optional depthwise
    convolution and :math:`\gamma` a per-channel layer scale present only
    in the hybrid variants.  The residual is used whenever the block keeps
    both resolution and width.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mobilenet_v4 import mobilenet_v4_conv_small
    >>> backbone = mobilenet_v4_conv_small().eval()
    >>> out = backbone(lucid.randn(1, 3, 224, 224))
    >>> out.last_hidden_state.shape
    (1, 960, 7, 7)
    >>> [f.reduction for f in backbone.feature_info]
    [4, 8, 16, 32]
    """

    config_class: ClassVar[type[MobileNetV4Config]] = MobileNetV4Config
    base_model_prefix: ClassVar[str] = "mobilenet_v4"

    def __init__(self, config: MobileNetV4Config) -> None:
        super().__init__(config)
        conv_stem, bn1, act, blocks, feature_info = _build_trunk(config)
        self.conv_stem = conv_stem
        self.bn1 = bn1
        self.act = act
        self.blocks = blocks
        self._feature_info = feature_info
        _init_weights(self)

    @override
    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    @override
    def forward_features(self, x: Tensor) -> Tensor:
        return _chain(x, self.conv_stem, self.bn1, self.act, self.blocks)

    @override
    def forward(self, x: Tensor) -> BaseModelOutput:
        return BaseModelOutput(last_hidden_state=self.forward_features(x))


# ---------------------------------------------------------------------------
# MobileNet v4 for image classification  (task="image-classification")
# ---------------------------------------------------------------------------


class MobileNetV4ForImageClassification(ImageClassificationModel, ClassificationHeadMixin):
    r"""MobileNet-v4 image classifier with the post-pool head.

    Runs the :class:`MobileNetV4` trunk and the head of the paper's tables:
    the 960-channel map is globally average-pooled *first*, then widened by
    a :math:`1 \times 1` convolution to 1280 channels with normalisation
    and activation, and finally projected to the classes,

    .. math::

        \operatorname{AvgPool} \to \operatorname{Conv}_{1\times1}^{1280}
            \to \operatorname{BN} \to \sigma \to \operatorname{Dropout}
            \to \operatorname{Linear}.

    Pooling before the widest layer is the MobileNet-v3 head redesign; the
    normalisation after that layer is new in v4.

    Parameters
    ----------
    config : MobileNetV4Config
        Architecture spec.  Use the ``*_cls`` factories
        (:func:`mobilenet_v4_conv_small_cls` and siblings) for the paper's
        variants.

    Attributes
    ----------
    config : MobileNetV4Config
        The config that built this model.
    conv_stem, bn1, act, blocks : nn.Module
        The same trunk as :class:`MobileNetV4`, held directly on the
        classifier so the parameter names match the released checkpoints.
    global_pool : nn.AdaptiveAvgPool2d
        Global average pool to :math:`1 \times 1`.
    conv_head : nn.Conv2d
        Bias-free :math:`1 \times 1` convolution, 960 → 1280 channels.
    norm_head : nn.BatchNorm2d
        Normalisation of the 1280-channel head features.
    act_head : nn.Module
        Head activation (the variant's activation).
    head_drop : nn.Dropout
        Dropout with probability ``config.dropout``.
    classifier : nn.Linear
        Final 1280 → ``config.num_classes`` projection.

    Notes
    -----
    When ``labels`` are passed to :meth:`forward`, the categorical
    cross-entropy against the logits is returned as ``loss``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mobilenet_v4 import mobilenet_v4_conv_small_cls
    >>> model = mobilenet_v4_conv_small_cls().eval()
    >>> out = model(lucid.randn(2, 3, 224, 224))
    >>> out.logits.shape
    (2, 1000)

    Retarget the head for a 10-class task:

    >>> model = mobilenet_v4_conv_small_cls(num_classes=10).eval()
    >>> model(lucid.randn(1, 3, 224, 224)).logits.shape
    (1, 10)
    """

    config_class: ClassVar[type[MobileNetV4Config]] = MobileNetV4Config
    base_model_prefix: ClassVar[str] = "mobilenet_v4"

    def __init__(self, config: MobileNetV4Config) -> None:
        super().__init__(config)
        conv_stem, bn1, act, blocks, feature_info = _build_trunk(config)
        self.conv_stem = conv_stem
        self.bn1 = bn1
        self.act = act
        self.blocks = blocks
        trunk_ch = feature_info[-1].num_channels

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_head = nn.Conv2d(trunk_ch, _HEAD_WIDTH, 1, bias=False)
        self.norm_head = nn.BatchNorm2d(_HEAD_WIDTH)
        self.act_head = _make_act(_ARCHS[config.variant].act)
        self.head_drop = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(_HEAD_WIDTH, config.num_classes)
        _init_weights(self)

    @override
    def forward(
        self,
        x: Tensor,
        labels: Tensor | None = None,
    ) -> ImageClassificationOutput:
        x = _chain(
            x,
            self.conv_stem,
            self.bn1,
            self.act,
            self.blocks,
            self.global_pool,
            self.conv_head,
            self.norm_head,
            self.act_head,
        )
        x = self.head_drop(x.reshape(x.shape[0], -1))
        logits = self.classifier(x)

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return ImageClassificationOutput(logits=logits, loss=loss)
