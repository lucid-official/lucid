"""EfficientDet — scalable compound-scaled detector (Tan et al., CVPR 2020).

Paper: "EfficientDet: Scalable and Efficient Object Detection"

Key contributions
-----------------
1. **BiFPN** — Bidirectional Feature Pyramid Network with fast-normalised
   weighted feature fusion.  Unlike top-down-only FPN, BiFPN allows both
   top-down and bottom-up paths and learns per-input, per-level fusion weights:
       out_i = ReLU(Σ_j w_j · x_j) / (ε + Σ_j w_j),  ε = 1e-4
   The weights are forced positive via ReLU.

2. **Compound scaling** — A single coefficient φ jointly scales the backbone
   (EfficientNet-Bφ), BiFPN width (W_bifpn) / depth (D_bifpn), and
   prediction head depth (D_head) / image resolution.

3. **Shared prediction heads** — Class and box heads share depthwise-separable
   convolution weights across all five BiFPN output levels (P3–P7), with
   separate batch-norm per level.

Architecture
------------
  Image → EfficientNet-Bφ → (P3, P4, P5) at strides (8, 16, 32)
    ↓  + P6 = MaxPool(P5), P7 = MaxPool(P6) for two additional coarse levels
  [P3,P4,P5,P6,P7] → BiFPN × D_bifpn
    ├─ Class head: D_head × SepConv(W_bifpn) → Conv(num_classes × A)
    └─ Box head:  D_head × SepConv(W_bifpn) → Conv(4 × A)

  where A = len(anchor_scales) × len(anchor_ratios) = 9 (3 scales × 3 ratios).

BiFPN single pass (for L levels, finest = 0):
  Intermediate top-down:
    P_td[L-1] = P_in[L-1]
    P_td[i]   = Conv(w1·P_in[i] + w2·Resize(P_td[i+1]))  for i = L-2 … 0
  Bottom-up:
    P_out[0]  = Conv(w1·P_in[0] + w2·P_td[0])
    P_out[i]  = Conv(w1·P_in[i] + w2·P_td[i] + w3·MaxPool(P_out[i-1]))  for i > 0

Faithfulness notes
------------------
* BiFPN uses depthwise-separable conv (DWConv + PWConv) per the paper.
* Learnable fusion weights ReLU-initialised to 1.0.
* Backbone: simplified EfficientNet-B0 (MBConv blocks) for default φ=0.
* Anchors: 9 per cell (3 scales × 3 ratios), at 5 levels (P3–P7).
* No NMS-free inference (unlike DETR) — uses per-class NMS.
* Loss: focal loss for classification + smooth-L1 for regression (paper §4.2).
"""

from typing import ClassVar, cast, final, override

import math

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._utils._common import (
    host_floats,
    host_ints,
    host_rows,
    make_divisible,
)
from lucid.models._tasks import ObjectDetectionModel
from lucid.models._output import ObjectDetectionOutput
from lucid.models._utils._detection import (
    AnchorGenerator,
    batched_nms,
    clip_boxes_to_image,
    decode_boxes,
    encode_boxes,
)
from lucid.models._utils._detection_results import (
    check_per_image,
    merge_class_detections,
)
from lucid.models.vision.efficientdet._config import EfficientDetConfig

# ---------------------------------------------------------------------------
# Depthwise-separable convolution (BiFPN building block)
# ---------------------------------------------------------------------------


@final
class _SepConv(nn.Module):
    """Depthwise-separable conv: DWConv(k,k,groups=C) + PWConv(1,1) + BN + ReLU."""

    def __init__(self, channels: int, kernel_size: int = 3, padding: int = 1) -> None:
        super().__init__()
        self.dw = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            groups=channels,
            bias=False,
        )
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    @override
    def forward(self, x: Tensor) -> Tensor:
        # Reference order is act -> dwconv -> pwconv -> BN, with no activation
        # after the norm: the fused sum is activated on the way *in*, and the
        # node's output stays linear.
        h = F.silu(x)
        return self.bn(self.pw(self.dw(h)))


# ---------------------------------------------------------------------------
# BiFPN level
# ---------------------------------------------------------------------------

_EPS = 1e-4


@final
class _BiFPNLayer(nn.Module):
    """One BiFPN repetition (bidirectional top-down + bottom-up fusion)."""

    def __init__(self, num_channels: int, num_levels: int = 5) -> None:
        super().__init__()
        self.num_levels = num_levels
        L = num_levels

        # Top-down intermediate weights (L-1 intermediate nodes, each fusing 2 inputs)
        self.td_weights: nn.ParameterList = nn.ParameterList(
            [nn.Parameter(lucid.ones((2,))) for _ in range(L - 1)]
        )
        # Bottom-up output weights, for levels 1 … L-1 only.
        # Paper Fig. 2(d): a BiFPN cell over P3…P7 has exactly EIGHT nodes —
        # four top-down (levels 6,5,4,3) and four bottom-up (levels 4,5,6,7).
        # The finest level owns a single node: the top-down node at level 3 IS
        # P3_out, never re-fused with P3_in.  The coarsest bottom-up node takes
        # two inputs (P7_in and the down-sampled P6_out), because level 7 has no
        # top-down intermediate — fusing ``td[L-1]`` there would count P7_in
        # twice, since the top-down pass passes it straight through.
        self.out_weights: nn.ParameterList = nn.ParameterList(
            [nn.Parameter(lucid.ones((3,))) for _ in range(L - 2)]
            + [nn.Parameter(lucid.ones((2,)))]
        )

        # One conv per top-down node (L-1) and one per bottom-up node (L-1).
        self.td_convs: nn.ModuleList[_SepConv] = nn.ModuleList(
            [_SepConv(num_channels) for _ in range(L - 1)]
        )
        self.out_convs: nn.ModuleList[_SepConv] = nn.ModuleList(
            [_SepConv(num_channels) for _ in range(L - 1)]
        )
        # Registered down-sampler (avoids allocating a new module each forward call)
        self.down = nn.MaxPool2d(2, stride=2)

    @override
    def forward(self, features: list[Tensor]) -> list[Tensor]:
        """
        Args:
            features: [P3, P4, P5, P6, P7] (finest → coarsest).

        Returns:
            Fused [P3_out, P4_out, P5_out, P6_out, P7_out].
        """
        L = self.num_levels
        assert len(features) == L

        # --- Top-down intermediate ---
        # Tensor-level arithmetic only — gradients must flow back to
        # ``self.td_weights`` / ``self.out_weights`` (paper's key contribution).
        td: list[Tensor] = [features[-1]]  # coarsest passes through
        for i in range(L - 2, -1, -1):  # from coarsest-1 down to finest
            # Eq. (3): the epsilon guards the *denominator* only.  Adding it to
            # the weight vector puts +eps in every numerator and N*eps in the
            # sum, which changes the fusion ratios rather than just avoiding a
            # division by zero.
            w: Tensor = F.relu(self.td_weights[L - 2 - i])
            wsum = w.sum() + _EPS
            up = F.interpolate(td[0], scale_factor=2.0, mode="nearest")
            fused: Tensor = (w[0] / wsum) * features[i] + (w[1] / wsum) * up
            node = self.td_convs[L - 2 - i](fused)
            td.insert(0, node)  # prepend so td[0] = finest

        # --- Bottom-up output ---
        # The finest level's top-down node is already P3_out.
        out: list[Tensor] = [td[0]]

        for i in range(1, L):
            wl: Tensor = F.relu(self.out_weights[i - 1])
            wlsum = wl.sum() + _EPS
            down = self.down(out[-1])
            if i < L - 1:
                fused_l: Tensor = (
                    (wl[0] / wlsum) * features[i]
                    + (wl[1] / wlsum) * td[i]
                    + (wl[2] / wlsum) * down
                )
            else:
                # Coarsest level: no top-down intermediate exists here.
                fused_l = (wl[0] / wlsum) * features[i] + (wl[1] / wlsum) * down
            out.append(self.out_convs[i - 1](fused_l))

        return out


# ---------------------------------------------------------------------------
# EfficientNet-B0 backbone (simplified, P3/P4/P5 outputs)
# ---------------------------------------------------------------------------


class _MBConv(nn.Module):
    """Mobile Inverted Bottleneck (MBConv) with optional skip connection."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        expand_ratio: int = 6,
        stride: int = 1,
        kernel_size: int = 3,
        se_ratio: float = 0.25,
    ) -> None:
        super().__init__()
        mid_ch = in_ch * expand_ratio
        padding = (kernel_size - 1) // 2

        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_ch, mid_ch, 1, bias=False),
                nn.BatchNorm2d(mid_ch),
                nn.SiLU(inplace=True),
            ]
        layers += [
            nn.Conv2d(
                mid_ch,
                mid_ch,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=mid_ch,
                bias=False,
            ),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(inplace=True),
        ]
        self.block = nn.Sequential(*layers)

        # EfficientNet's MBConv carries squeeze-and-excitation at ratio 0.25;
        # omitting it drops the channel-gating the backbone was designed
        # around and quietly changes the parameter count.
        self.se: nn.Sequential | None
        if 0.0 < se_ratio <= 1.0:
            se_ch = max(1, int(in_ch * se_ratio))
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(mid_ch, se_ch, 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(se_ch, mid_ch, 1),
                nn.Sigmoid(),
            )
        else:
            self.se = None

        self.project = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.use_skip = stride == 1 and in_ch == out_ch

    @override
    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.block(x)
        if self.se is not None:
            out = out * self.se(out)
        out = self.project(out)
        return out + x if self.use_skip else out


def _make_mbconv_stage(
    in_ch: int, out_ch: int, n: int, stride: int = 1, expand: int = 6, k: int = 3
) -> nn.Sequential:
    blocks: list[nn.Module] = [_MBConv(in_ch, out_ch, expand, stride, k)]
    for _ in range(1, n):
        blocks.append(_MBConv(out_ch, out_ch, expand, 1, k))
    return nn.Sequential(*blocks)


@final
class _EfficientNetBackbone(nn.Module):
    """Simplified EfficientNet-B0 backbone.

    Returns (P3, P4, P5) feature maps at strides (8, 16, 32).
    Channel widths:
      After stage 2 (stride 8):  P3 = 40ch
      After stage 4 (stride 16): P4 = 112ch
      After stage 6 (stride 32): P5 = 320ch
    """

    def __init__(
        self,
        in_channels: int,
        width_coeff: float = 1.0,
        depth_coeff: float = 1.0,
    ) -> None:
        super().__init__()

        # EfficientNet compound scaling: widths rounded to a multiple of 8 with
        # the "never lose more than 10%" guard, depths rounded up.  Without
        # this the backbone stayed at B0 for every variant while the channel
        # projections were built from the scaled table, so D2 and up could be
        # constructed but not run — the first projection saw 40 channels where
        # it expected 48.
        def w(c: int) -> int:
            return make_divisible(c * width_coeff, 8)

        def d(n: int) -> int:
            return int(math.ceil(n * depth_coeff))

        stem_ch = w(32)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_ch),
            nn.SiLU(inplace=True),
        )
        # MBConv stages (EfficientNet-B0 base settings, compound-scaled)
        c0, c1, c2 = w(16), w(24), w(40)
        c3, c4, c5, c6 = w(80), w(112), w(192), w(320)
        self.stage0 = _make_mbconv_stage(stem_ch, c0, n=d(1), stride=1, expand=1, k=3)
        self.stage1 = _make_mbconv_stage(c0, c1, n=d(2), stride=2, expand=6, k=3)
        self.stage2 = _make_mbconv_stage(c1, c2, n=d(2), stride=2, expand=6, k=5)
        self.stage3 = _make_mbconv_stage(c2, c3, n=d(3), stride=2, expand=6, k=3)
        self.stage4 = _make_mbconv_stage(c3, c4, n=d(3), stride=1, expand=6, k=5)
        self.stage5 = _make_mbconv_stage(c4, c5, n=d(4), stride=2, expand=6, k=5)
        self.stage6 = _make_mbconv_stage(c5, c6, n=d(1), stride=1, expand=6, k=3)

    @override
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x = self.stem(x)
        x = self.stage0(x)
        x = self.stage1(x)
        p3: Tensor = self.stage2(x)  # stride 8
        x = self.stage3(p3)
        p4: Tensor = self.stage4(x)  # stride 16
        x = self.stage5(p4)
        p5: Tensor = self.stage6(x)  # stride 32
        return p3, p4, p5


# ---------------------------------------------------------------------------
# Prediction head (class or box)
# ---------------------------------------------------------------------------


@final
class _PredictionHead(nn.Module):
    """Shared-weight prediction head for all BiFPN levels.

    Uses depth-wise separable convolutions with separate batch-norm per level.
    """

    def __init__(
        self,
        in_channels: int,
        num_outputs: int,  # num_classes or 4 * num_anchors
        num_repeats: int,
        num_levels: int = 5,
    ) -> None:
        super().__init__()
        self.num_levels = num_levels
        self.num_repeats = num_repeats

        # Shared DWConv weights (one per repeat depth)
        self.dw_convs: nn.ModuleList[nn.Conv2d] = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    3,
                    padding=1,
                    groups=in_channels,
                    bias=False,
                )
                for _ in range(num_repeats)
            ]
        )
        self.pw_convs: nn.ModuleList[nn.Conv2d] = nn.ModuleList(
            [
                nn.Conv2d(in_channels, in_channels, 1, bias=False)
                for _ in range(num_repeats)
            ]
        )
        # Separate BN per level per depth
        self.bns: nn.ModuleList[nn.ModuleList[nn.BatchNorm2d]] = nn.ModuleList(
            [
                nn.ModuleList([nn.BatchNorm2d(in_channels) for _ in range(num_levels)])
                for _ in range(num_repeats)
            ]
        )
        # The reference predictor is a *separable 3x3* conv, not a dense 1x1:
        # depthwise 3x3 then a pointwise projection, no norm and no activation.
        self.predictor_dw = nn.Conv2d(
            in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False
        )
        self.predictor = nn.Conv2d(in_channels, num_outputs, 1)

    @override
    def forward(self, features: list[Tensor]) -> list[Tensor]:
        """
        Args:
            features: List of num_levels feature maps (finest → coarsest).

        Returns:
            List of prediction maps, one per level.
        """
        outs: list[Tensor] = []
        for lvl, feat in enumerate(features):
            x = feat
            for depth in range(self.num_repeats):
                dw: Tensor = self.dw_convs[depth](x)
                pw: Tensor = self.pw_convs[depth](dw)
                bn_list = self.bns[depth]
                x = F.silu(bn_list[lvl](pw))
            outs.append(self.predictor(self.predictor_dw(x)))
        return outs


def _init_head_priors(
    cls_head: _PredictionHead,
    box_head: _PredictionHead,
    prior_prob: float = 0.01,
) -> None:
    """Apply the focal-loss bias prior to the class predictor.

    The class head's output bias is set to :math:`-\\log((1 - \\pi) / \\pi)`
    so that a freshly built detector predicts :math:`p \\approx \\pi` for every
    anchor and class.  Without it the initial sigmoid sits at 0.5 across
    ~10^5 mostly-background slots and the focal loss starts two orders of
    magnitude too large — the instability RetinaNet and EfficientDet both
    document.  The box predictor's bias starts at zero.

    Args:
        cls_head:   Classification head whose ``predictor`` bias is set.
        box_head:   Box head whose ``predictor`` bias is zeroed.
        prior_prob: Target foreground probability at initialisation.
    """
    bias_value = -math.log((1.0 - prior_prob) / prior_prob)
    cls_bias = cls_head.predictor.bias
    if cls_bias is not None:
        nn.init.constant_(cls_bias, bias_value)
    box_bias = box_head.predictor.bias
    if box_bias is not None:
        nn.init.zeros_(box_bias)


# ---------------------------------------------------------------------------
# Smooth-L1 and focal loss helpers
# ---------------------------------------------------------------------------


def _smooth_l1(x: Tensor, beta: float = 0.1) -> Tensor:
    """Huber loss with the reference's scaling.

    The reference is ``0.5 x^2`` for ``|x| < delta`` and
    ``delta |x| - 0.5 delta^2`` beyond — *not* divided by delta.  Dividing
    made the box term 1/delta times larger (10x at the default 0.1), which
    silently re-weights it against the classification term.
    """
    abs_x: Tensor = lucid.abs(x)
    cond: Tensor = abs_x < beta
    return lucid.where(cond, 0.5 * x * x, beta * abs_x - 0.5 * beta * beta)


def _focal_loss(
    logits: Tensor,
    targets: Tensor,
    alpha: float = 0.25,
    gamma: float = 1.5,
    mask: Tensor | None = None,
) -> Tensor:
    """Binary focal loss for multi-label classification (sigmoid).

    Returns the SUM over all anchor x class slots.  The RetinaNet /
    EfficientDet convention is to normalise the summed class and box losses by
    ``num_positives + 1`` — the count of foreground anchors — not by the number
    of logits.  Averaging over A*K instead divides by ~10^5 slots that are
    almost all background, which shrinks the term by orders of magnitude and
    makes it drift with the anchor count rather than the object count.

    ``gamma`` defaults to the paper's 1.5, not RetinaNet's 2.0.

    Args:
        logits:  (N,) raw logits.
        targets: (N,) binary targets {0.0, 1.0}.
        mask:    Optional (N,) multiplier in {0.0, 1.0}.  Slots set to 0 are
                 excluded from the loss — this is how the reference drops the
                 "ignore" IoU band, which otherwise trains as background.
    """
    p: Tensor = F.sigmoid(logits)
    ce: Tensor = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t: Tensor = targets * p + (1.0 - targets) * (1.0 - p)
    alpha_t = targets * alpha + (1.0 - targets) * (1.0 - alpha)
    focal_weight = alpha_t * (1.0 - p_t) ** gamma
    per_slot: Tensor = focal_weight * ce
    if mask is not None:
        per_slot = per_slot * mask
    return per_slot.sum()


# ---------------------------------------------------------------------------
# EfficientDet
# ---------------------------------------------------------------------------


class EfficientDetForObjectDetection(ObjectDetectionModel):
    r"""EfficientDet object detector (Tan et al., CVPR 2020).

    A family of compound-scaled single-stage detectors combining an
    EfficientNet backbone, a **BiFPN** (bidirectional weighted feature
    pyramid) neck, and shared classification + box regression heads
    applied to 5 feature levels (P3-P7).  The family is parameterised
    by a compound coefficient :math:`\varphi \in \{0, 1, \dots, 7\}`
    that simultaneously scales backbone depth / width, BiFPN width and
    repeat count, head depth, and input resolution — yielding D0-D7
    variants that span ~4M to ~52M parameters and trade speed for
    accuracy along a Pareto-optimal curve.

    Parameters
    ----------
    config : EfficientDetConfig
        Frozen architecture spec.  Use the per-:math:`\varphi` factories
        (:func:`efficientdet_d0` through :func:`efficientdet_d7`) for the
        paper-cited compound-scaled variants.

    Attributes
    ----------
    config : EfficientDetConfig
        Stored copy of the config that built this model.
    backbone : _EfficientNetBackbone
        EfficientNet trunk producing C3 / C4 / C5 features at strides
        8 / 16 / 32.
    p3_proj, p4_proj, p5_proj : nn.Sequential
        1x1 conv + BatchNorm channel projections from backbone widths
        to ``config.fpn_channels``.
    p6_pool, p7_pool : nn.MaxPool2d
        2x downsamples producing P6 and P7 at strides 64 and 128.
    bifpn : nn.ModuleList
        ``config.fpn_repeats`` :class:`_BiFPNLayer` blocks performing
        weighted bidirectional top-down + bottom-up feature fusion.
    cls_head, box_head : _PredictionHead
        Shared per-level prediction heads with ``config.head_repeats``
        3x3 separable convolutions; outputs ``K * num_anchors`` and
        ``4 * num_anchors`` channels respectively.
    _anchor_gen : AnchorGenerator
        Anchor generator covering 5 levels with
        :math:`|\mathrm{scales}| \times |\mathrm{ratios}|` anchors per cell.

    Notes
    -----
    See Tan et al., "EfficientDet: Scalable and Efficient Object
    Detection", CVPR 2020 (arXiv:1911.09070).  The BiFPN's defining
    feature is weighted feature fusion at each node:

    .. math::

        O = \sum_{i} \frac{w_i}{\epsilon + \sum_j w_j} \cdot I_i,

    with non-negative weights :math:`w_i` learned end-to-end (the paper
    calls this "fast normalized fusion").  Compound scaling follows

    .. math::

        \begin{aligned}
            W_\mathrm{BiFPN} &= 64 \cdot 1.35^\varphi, &
            D_\mathrm{BiFPN} &= 3 + \varphi, \\
            D_\mathrm{head}  &= 3 + \lfloor \varphi / 3 \rfloor, &
            R_\mathrm{input} &= 512 + 128\varphi,
        \end{aligned}

    so D0 (:math:`\varphi = 0`) trains on 512x512 inputs while D7
    (:math:`\varphi = 7`) uses 1536x1536.  Training uses focal loss
    :math:`\mathrm{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log p_t` on
    the K-channel class output plus smooth-:math:`L_1` on box deltas.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.efficientdet import efficientdet_d0
    >>> model = efficientdet_d0()
    >>> x = lucid.randn(1, 3, 512, 512)
    >>> out = model(x)
    >>> out.logits.shape[-1], out.pred_boxes.shape[-1]
    (80, 4)
    """

    config_class: ClassVar[type[EfficientDetConfig]] = EfficientDetConfig
    base_model_prefix: ClassVar[str] = "efficientdet"

    def __init__(self, config: EfficientDetConfig) -> None:
        super().__init__(config)
        self._cfg = config
        W = config.fpn_channels
        K = config.num_classes
        num_levels = 5  # P3–P7
        num_anchors = len(config.anchor_scales) * len(config.anchor_ratios)

        # Backbone
        self.backbone = _EfficientNetBackbone(
            config.in_channels,
            width_coeff=config.backbone_width_coeff,
            depth_coeff=config.backbone_depth_coeff,
        )

        # Channel projection: P3/P4/P5 → fpn_channels
        bb_ch = config.backbone_in_channels
        self.p3_proj = nn.Sequential(nn.Conv2d(bb_ch[0], W, 1), nn.BatchNorm2d(W))
        self.p4_proj = nn.Sequential(nn.Conv2d(bb_ch[1], W, 1), nn.BatchNorm2d(W))
        self.p5_proj = nn.Sequential(nn.Conv2d(bb_ch[2], W, 1), nn.BatchNorm2d(W))

        # Each coarse level gets its own ResampleFeatureMap in the reference:
        # a 1x1 conv + BN (no activation) resampling from the *backbone's* C5,
        # then a max-pool with ``kernel_size = stride + 1 = 3``.  Reusing P5's
        # already-projected tensor and pooling it with a kernel-2 window is a
        # different operator with a different receptive field.
        self.p6_proj = nn.Sequential(nn.Conv2d(bb_ch[2], W, 1), nn.BatchNorm2d(W))
        self.p6_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.p7_pool = nn.MaxPool2d(3, stride=2, padding=1)

        # BiFPN stack
        self.bifpn = nn.ModuleList(
            [_BiFPNLayer(W, num_levels=num_levels) for _ in range(config.fpn_repeats)]
        )

        # Prediction heads
        self.cls_head = _PredictionHead(
            W, K * num_anchors, config.head_repeats, num_levels
        )
        self.box_head = _PredictionHead(
            W, 4 * num_anchors, config.head_repeats, num_levels
        )
        # Focal-loss prior on the class predictor's bias: at initialisation
        # every anchor should predict p ~= prior (0.01), not 0.5, or the first
        # steps are swamped by ~10^5 background slots.  The box predictor's
        # bias starts at zero, as in the reference.
        _init_head_priors(
            self.cls_head, self.box_head, prior_prob=config.focal_prior_prob
        )

        # Anchor generator (5 levels; one base size per level)
        tuple(
            (int(s * r),)
            for s, r in [
                (
                    config.anchor_base_sizes[i],
                    config.anchor_scales[0],
                )  # base size only; scales handled by anchor_scales
                for i in range(num_levels)
            ]
        )
        # Build anchors with all scales × ratios
        all_sizes: tuple[tuple[int, ...], ...] = tuple(
            tuple(
                int(config.anchor_base_sizes[lvl] * sc) for sc in config.anchor_scales
            )
            for lvl in range(num_levels)
        )
        self._anchor_gen = AnchorGenerator(
            sizes=all_sizes,
            aspect_ratios=(tuple(config.anchor_ratios),) * num_levels,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _project_backbone(self, p3: Tensor, p4: Tensor, p5: Tensor) -> list[Tensor]:
        """Project P3/P4/P5 to FPN width and build P6/P7."""
        # The reference's ResampleFeatureMap is conv + BN with ``act_layer=None``
        # — no activation on the lateral projections.
        fp3: Tensor = self.p3_proj(p3)
        fp4: Tensor = self.p4_proj(p4)
        fp5: Tensor = self.p5_proj(p5)
        fp6: Tensor = self.p6_pool(self.p6_proj(p5))
        fp7: Tensor = self.p7_pool(fp6)
        return [fp3, fp4, fp5, fp6, fp7]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @override
    def forward(
        self,
        x: Tensor,
        targets: list[dict[str, Tensor]] | None = None,
    ) -> ObjectDetectionOutput:
        """Run EfficientDet.

        Args:
            x:       (B, C, H, W) image batch.
            targets: Optional training targets.

        Returns:
            ``ObjectDetectionOutput``:
              ``logits``    : (B, A, K) per-class sigmoid logits (A = total anchors).
              ``pred_boxes``: (B, A, 4) decoded xyxy boxes.
              ``loss``      : focal + smooth-L1 when targets provided.
        """
        B = int(x.shape[0])
        iH = int(x.shape[2])
        iW = int(x.shape[3])
        check_per_image(type(self).__name__, B, targets=targets)

        # 1. Backbone → (P3, P4, P5)
        p3, p4, p5 = self.backbone(x)

        # 2. Project → [P3, P4, P5, P6, P7] in FPN-width channels
        fpn_feats = self._project_backbone(p3, p4, p5)

        # 3. BiFPN stack
        for bifpn_layer in self.bifpn:
            fpn_feats = bifpn_layer(fpn_feats)

        # 4. Strides for 5 levels (P3=8, P4=16, P5=32, P6=64, P7=128)
        strides: list[tuple[int, int]] = [
            (8, 8),
            (16, 16),
            (32, 32),
            (64, 64),
            (128, 128),
        ]

        # 5. Generate anchors → list of (A_l, 4) per level
        anchors_per_level = self._anchor_gen(fpn_feats, (iH, iW), strides)

        # 6. Prediction heads → per-level class / box maps
        cls_maps = self.cls_head(fpn_feats)
        box_maps = self.box_head(fpn_feats)

        # 7. Reshape and concatenate → (B, A_total, K) and (B, A_total, 4)
        num_anchors = len(self._cfg.anchor_scales) * len(self._cfg.anchor_ratios)
        K = self._cfg.num_classes

        cls_all_parts: list[Tensor] = []
        box_all_parts: list[Tensor] = []
        anchors_flat_parts: list[Tensor] = []

        for lvl, (cm, bm, anc) in enumerate(
            zip(cls_maps, box_maps, anchors_per_level, strict=True)
        ):
            fH = int(cm.shape[2])
            fW = int(cm.shape[3])
            # cm: (B, K*A, H, W) → (B, H*W*A, K)
            cm_r = cm.reshape(B, K, num_anchors, fH, fW)
            cm_r = cm_r.permute(0, 3, 4, 2, 1).reshape(B, -1, K)
            # bm: (B, 4*A, H, W) → (B, H*W*A, 4)
            bm_r = bm.reshape(B, 4, num_anchors, fH, fW)
            bm_r = bm_r.permute(0, 3, 4, 2, 1).reshape(B, -1, 4)

            cls_all_parts.append(cm_r)
            box_all_parts.append(bm_r)
            anchors_flat_parts.append(anc)  # (A_l, 4)

        all_logits: Tensor = lucid.cat(cls_all_parts, dim=1)  # (B, A, K)
        all_deltas: Tensor = lucid.cat(box_all_parts, dim=1)  # (B, A, 4)
        all_anchors: Tensor = lucid.cat(anchors_flat_parts, dim=0)  # (A, 4)

        # 8. Decode boxes
        int(all_deltas.shape[1])
        all_boxes_parts: list[Tensor] = []
        for b in range(B):
            boxes_b = decode_boxes(all_deltas[b], all_anchors)  # (A, 4)
            boxes_b = clip_boxes_to_image(boxes_b, (iH, iW))
            all_boxes_parts.append(boxes_b.unsqueeze(0))
        all_boxes: Tensor = lucid.cat(all_boxes_parts, dim=0)  # (B, A, 4)

        # 9. Loss
        loss: Tensor | None = None
        if targets is not None:
            loss = self._compute_loss(all_logits, all_deltas, all_anchors, targets)

        return ObjectDetectionOutput(
            logits=all_logits,
            pred_boxes=all_boxes,
            loss=loss,
        )

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------

    def _compute_loss(
        self,
        all_logits: Tensor,  # (B, A, K)
        all_deltas: Tensor,  # (B, A, 4)
        all_anchors: Tensor,  # (A, 4)
        targets: list[dict[str, Tensor]],
    ) -> Tensor:
        B = len(targets)
        K = self._cfg.num_classes
        A = int(all_anchors.shape[0])
        dev, dt = all_logits.device.type, all_logits.dtype  # targets in both

        cls_losses: list[Tensor] = []
        reg_losses: list[Tensor] = []

        for b in range(B):
            gt_boxes = targets[b]["boxes"]  # (M, 4) xyxy
            gt_labels = targets[b]["labels"]  # (M,)
            M = int(gt_boxes.shape[0])

            lg_b = all_logits[b]  # (A, K)

            if M == 0:
                # All anchors → background; normaliser is the +1 floor.
                tgt_cls = lucid.zeros((A, K), dtype=dt, device=dev)
                cls_losses.append(
                    _focal_loss(
                        lg_b.reshape(-1),
                        tgt_cls.reshape(-1),
                        alpha=self._cfg.focal_alpha,
                        gamma=self._cfg.focal_gamma,
                    )
                    / 1.0
                )
                continue

            # Compute pairwise IoU between anchors and GT boxes
            # Build manually to avoid importing box_iou with its O(NxM) loop
            from lucid.models._utils._detection import box_iou as _box_iou

            iou_mat = _box_iou(all_anchors, gt_boxes)  # (A, M)

            # Assign each anchor: best GT, then label
            tgt_cls_data = [[0.0] * K for _ in range(A)]
            # Slot mask: 1 everywhere except the ignore band, which the
            # reference removes from the class loss with
            # ``cls_loss * (cls_targets_at_level != -2)``.
            cls_mask_data = [[1.0] * K for _ in range(A)]
            pos_idx: list[int] = []
            pos_gt: list[int] = []

            # Vectorised per-anchor reductions.
            best_m_t = lucid.argmax(iou_mat, dim=1)
            best_v_t = iou_mat.max(dim=1)
            best_v_list = host_floats(best_v_t)
            best_m_list = host_ints(best_m_t)
            labels_list = host_ints(gt_labels)

            fg_thr = self._cfg.iou_fg_thresh
            bg_thr = self._cfg.iou_bg_thresh
            num_ignored = 0
            for a in range(A):
                best_v = best_v_list[a]
                best_m = best_m_list[a]
                if best_v >= fg_thr:
                    c = labels_list[best_m] - 1  # 0-indexed
                    if 0 <= c < K:
                        tgt_cls_data[a][c] = 1.0
                    pos_idx.append(a)
                    pos_gt.append(best_m)
                elif best_v < bg_thr:
                    pass  # background — all zeros (already set)
                else:
                    # Ignore band: ambiguous supervision, so the anchor is
                    # dropped from the class loss rather than trained as
                    # background.
                    cls_mask_data[a] = [0.0] * K
                    num_ignored += 1

            # Force-match: every GT row takes its best anchor regardless of IoU
            # (``force_match_for_each_row=True``).  Anchor-major assignment
            # alone leaves an object with no positive at all whenever its best
            # overlap falls under the foreground threshold — common for small
            # or oddly-shaped boxes — so it contributes only background loss.
            best_a_per_gt = host_ints(lucid.argmax(iou_mat, dim=0))
            for m_i in range(M):
                a_forced = best_a_per_gt[m_i]
                if a_forced in pos_idx:
                    continue
                c_f = labels_list[m_i] - 1
                if 0 <= c_f < K:
                    tgt_cls_data[a_forced][c_f] = 1.0
                cls_mask_data[a_forced] = [1.0] * K
                pos_idx.append(a_forced)
                pos_gt.append(m_i)

            tgt_cls = lucid.tensor(tgt_cls_data, dtype=dt, device=dev)  # (A, K)
            cls_mask: Tensor | None = None
            if num_ignored:
                cls_mask = lucid.tensor(cls_mask_data, dtype=dt, device=dev)
            # ``num_positives_sum = num_positives.sum() + 1.0`` in the reference
            # loss; both the class and box terms divide by it.
            num_pos = float(len(pos_idx)) + 1.0
            cls_losses.append(
                _focal_loss(
                    lg_b.reshape(-1),
                    tgt_cls.reshape(-1),
                    alpha=self._cfg.focal_alpha,
                    gamma=self._cfg.focal_gamma,
                    mask=None if cls_mask is None else cls_mask.reshape(-1),
                )
                / num_pos
            )

            if pos_idx:
                pos_t = lucid.tensor(pos_idx, device=dev).long()
                gt_rows = host_rows(gt_boxes[:, :4])
                gt_boxes_pos = lucid.tensor(gt_rows, dtype=dt, device=dev)[pos_gt]
                anc_pos = all_anchors[pos_t]  # (P, 4)
                tgt_d = encode_boxes(gt_boxes_pos, anc_pos)
                pred_d = all_deltas[b][pos_t]  # (P, 4)
                reg_losses.append(_smooth_l1(pred_d - tgt_d).sum() / num_pos)

        cls_l = (
            lucid.cat([l.reshape(1) for l in cls_losses]).mean()
            if cls_losses
            else lucid.zeros((1,), dtype=dt, device=dev)
        )
        reg_l = (
            lucid.cat([l.reshape(1) for l in reg_losses]).mean()
            if reg_losses
            else lucid.zeros((1,), dtype=dt, device=dev)
        )
        # ``total_loss = cls_loss + box_loss_weight * box_loss`` — the box term
        # is boosted 50x precisely because both terms share the same
        # positive-anchor normaliser.
        return cls_l + self._cfg.box_loss_weight * reg_l

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def postprocess(
        self,
        output: ObjectDetectionOutput,
    ) -> list[dict[str, Tensor]]:
        """Per-class NMS on raw sigmoid predictions.

        Returns list of per-image result dicts with "boxes", "scores", "labels".
        """
        B = int(output.logits.shape[0])
        K = self._cfg.num_classes
        results: list[dict[str, Tensor]] = []
        dev = output.logits.device.type

        for b in range(B):
            lg_b = output.logits[b]  # (A, K)
            bx_b = output.pred_boxes[b]  # (A, 4)
            sc_b = F.sigmoid(lg_b)  # (A, K) — per-class probabilities

            keep_boxes: list[Tensor] = []
            keep_scores: list[Tensor] = []
            keep_labels: list[Tensor] = []

            # One host transfer for the whole (A, K) score matrix.  Reading
            # it element-by-element cost a device sync per (anchor, class):
            # at D0's 512px input that is 49104 anchors x 90 classes =
            # 4.4M syncs per image, which is not a runnable postprocess.
            A = int(sc_b.shape[0])
            thr = self._cfg.score_thresh
            flat = cast(list[float], sc_b.reshape(-1).tolist())

            for c in range(K):
                mask: list[int] = [a for a in range(A) if flat[a * K + c] >= thr]
                if not mask:
                    continue
                sc_c = sc_b[:, c]
                mask_t = lucid.tensor(mask, device=dev).long()
                sc_sel = sc_c[mask_t]
                bx_sel = bx_b[mask_t]
                keep = batched_nms(
                    bx_sel,
                    sc_sel,
                    lucid.zeros(int(sc_sel.shape[0]), device=dev),
                    self._cfg.nms_thresh,
                )
                # No per-class cap here — ``max_detections`` is a *per image*
                # limit in the reference, applied once after a global score
                # sort.  Capping inside the class loop let K classes each keep
                # ``max_detections``, so the returned count scaled with the
                # class count and low-scoring boxes from a sparse class
                # outranked nothing.
                keep_boxes.append(bx_sel[keep])
                keep_scores.append(sc_sel[keep])
                keep_labels.append(
                    lucid.full(
                        (int(keep.shape[0]),), c + 1, dtype=lucid.int64, device=dev
                    )
                )

            results.append(
                merge_class_detections(
                    keep_boxes, keep_scores, keep_labels, dev, self._cfg.max_detections
                )
            )
        return results
