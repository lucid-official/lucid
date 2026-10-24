"""YOLOv4 — You Only Look Once v4 (Bochkovskiy et al., arXiv 2020).

Paper: "YOLOv4: Optimal Speed and Accuracy of Object Detection"
https://arxiv.org/abs/2004.10934

Key improvements over YOLOv3
-----------------------------
1. **CSPDarknet-53 backbone** — replaces standard Darknet-53 residual blocks
   with Cross Stage Partial (CSP) blocks that split the feature map, apply
   residuals to one branch, and concatenate, reducing gradient duplication and
   improving learning capacity.
2. **SPP module** (Spatial Pyramid Pooling) at P5: MaxPool(k=5), MaxPool(k=9),
   MaxPool(k=13) + original feature → concat, quadrupling the receptive field
   at essentially no extra inference cost.
3. **PANet neck** (Path Aggregation Network): augments the FPN-style top-down
   pathway with a bottom-up pathway, so each detection scale receives both
   fine-grained detail (from bottom-up) and semantic context (from top-down).
4. **CIoU loss** for bounding-box regression — Complete IoU accounts for
   overlap area, centre distance, and aspect-ratio consistency simultaneously,
   giving faster convergence and better localisation than MSE.

Architecture overview
---------------------
  Image (B, 3, H, W)
    ↓  CSPDarknet-53 backbone → P3 (256ch), P4 (512ch), P5 (1024ch)
    ↓  SPP at P5: MaxPool(5,9,13) concat → 2048ch → compress → 512ch
    ↓  PANet neck
       Top-down:
         P5(512) → upsample → concat P4(512) → CSP compress → P4'(256)
         P4'(256) → upsample → concat P3(256) → CSP compress → P3'(128)
       Bottom-up:
         P3'(128) → stride-2 conv → concat P4'(256) → CSP compress → P4''(256)
         P4''(256) → stride-2 conv → concat P5(512) → CSP compress → P5''(512)
    ↓  Detection heads at P3'', P4'', P5''
       Each: Conv → nA*(5+C) predictions
    ↓  Decode + CIoU loss / NMS

Loss (training)
---------------
  Requires ``targets`` — list of B dicts:
    "boxes"  : (M, 4) xyxy pixel coordinates
    "labels" : (M,)   integer class ids (0-indexed)

  Anchor assignment: same as YOLOv3 (best-IoU per GT at matching grid cell).
  L = L_ciou(box)   [positive anchors only]
    + L_bce(obj)    [positive=1, negative=lambda_noobj scaled]
    + L_bce(cls)    [positive anchors only]
"""

import math
from dataclasses import dataclass
from typing import ClassVar, final, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import ModelConfig
from lucid.models._tasks import ObjectDetectionModel
from lucid.models._meta import model_family_meta
from lucid.models._output import ObjectDetectionOutput
from lucid.models._registry import register_model
from lucid.models.vision.yolo._weights import YOLOV4Weights
import lucid.weights as weights_mod

# v4 reuses v3's anchor-ignore rule verbatim; only the threshold differs
# (yolov4.cfg says .7, yolov3's paper text says .5).  Its detection scoring
# and NMS are v3's too.
from lucid.models.vision.yolo._v3 import (
    _gt_rows,
    _ignore_mask,
    _postprocess_detections,
    _scale_priors,
)
from lucid.models._utils._common import apply_overrides
from lucid.models._utils._detection_results import check_per_image

_IGNORE_IOU_THRESH = 0.7

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@model_family_meta(
    canonical_name="YOLO",
    citation=(
        'Redmon, Joseph, et al. "You Only Look Once: Unified, Real-Time '
        'Object Detection." Proceedings of the IEEE Conference on '
        "Computer Vision and Pattern Recognition, 2016, pp. 779–788."
    ),
    theory=r"""
    YOLOv4 (Bochkovskiy et al., 2020) is a systematic engineering
    redesign rather than a new theoretical departure: the authors
    enumerate "bag-of-freebies" (training-time tricks with no
    inference cost) and "bag-of-specials" (small architectural
    additions with marginal inference cost) and select the
    combination that maximises AP at real-time throughput on a
    single consumer GPU.

    **CSPDarknet-53 backbone.**  Each residual stage is wrapped in a
    *Cross-Stage Partial* block that splits the feature map along the
    channel dimension, processes one half with the residual stack,
    and concatenates the result back:

    .. math::

        y = \mathrm{conv}\bigl(
            [x_1, \, \mathrm{Stack}(x_2)]
        \bigr),
        \qquad x = [x_1, x_2].

    CSP cuts FLOPs while preserving (or improving) accuracy by
    reducing gradient duplication across the stack.  The backbone
    also swaps LeakyReLU for **Mish** activation
    :math:`x \cdot \tanh(\mathrm{softplus}(x))` for smoother
    gradient flow.

    **SPP + PANet neck.**  An SPP block pools the deepest feature
    with kernels :math:`\{5, 9, 13\}` for an enlarged receptive
    field, then a **Path Aggregation Network** adds bottom-up
    information flow to the standard FPN top-down path, shortening
    the path length between low-level localisation features and the
    deepest prediction head.

    **Improved heads + losses.**  The same three-scale anchored
    head as v3, trained with CIoU regression loss.  The result is the
    new Pareto frontier on COCO at the time of release (≈43 AP at
    65 fps on a V100) and a template that the v5/v6/v7/v8 lines
    extend further.

    Of the paper's bag-of-freebies, only **CIoU** lives in this
    module — it is a loss term, so it belongs to the model.  Mosaic
    augmentation, self-adversarial training, DropBlock, label
    smoothing, cosine LR and cross-mini-batch normalisation are
    properties of a *training recipe*, not of the architecture, and
    none of them is implemented here.  A caller reproducing the
    paper's AP supplies them from the data pipeline and optimiser.
    """,
)
@dataclass(frozen=True, slots=True)
class YOLOV4Config(ModelConfig):
    """Configuration for YOLOv4.

    YOLOv4 (Bochkovskiy et al., 2020) uses CSPDarknet-53 as backbone,
    SPP + PANet as neck, and three detection scales at strides 8/16/32.

    Args:
        num_classes:  Number of foreground classes (COCO default = 80).
        in_channels:  Input image channels.
        anchors:      9 (width, height) anchor pairs in pixels, 3 per scale.
                      Order: P3 (small), P4 (medium), P5 (large).
        strides:      Feature-map strides for P3, P4, P5.
        score_thresh: Minimum class score to keep a detection.
        nms_thresh:   IoU threshold for per-class NMS.
        lambda_noobj: Objectness loss weight for negative anchors.

    Examples
    --------
    >>> from lucid.models.vision.yolo import YOLOV4Config
    >>> cfg = YOLOV4Config()
    >>> cfg.num_classes
    80
    >>> cfg.model_type
    'yolo_v4'
    """

    model_type: ClassVar[str] = "yolo_v4"

    num_classes: int = 80
    in_channels: int = 3

    # Re-clustered for v4's 608x608 training resolution — these are not
    # YOLOv3's priors, which is what this field previously held.
    anchors: tuple[tuple[float, float], ...] = (
        # P3 (stride 8) — small objects
        (12.0, 16.0),
        (19.0, 36.0),
        (40.0, 28.0),
        # P4 (stride 16) — medium objects
        (36.0, 75.0),
        (76.0, 55.0),
        (72.0, 146.0),
        # P5 (stride 32) — large objects
        (142.0, 110.0),
        (192.0, 243.0),
        (459.0, 401.0),
    )

    strides: tuple[int, int, int] = (8, 16, 32)
    score_thresh: float = 0.5
    nms_thresh: float = 0.5
    lambda_noobj: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "anchors", tuple(tuple(a) for a in self.anchors))
        object.__setattr__(self, "strides", tuple(self.strides))
        # v4 matches each GT against the priors of one scale at a time, so a
        # prior outside the three masks would never be used — the count must
        # be exactly nine, not merely enough.
        if len(self.anchors) != 9 or len(self.strides) != 3:
            raise ValueError(
                f"YOLOV4Config needs 9 anchors and 3 strides — three priors at "
                f"each of P3, P4, P5 — but got {len(self.anchors)} anchors and "
                f"{len(self.strides)} strides {self.strides}"
            )


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------


class _ConvBnLeaky(nn.Module):
    """Conv2d(bias=False) → BatchNorm2d → LeakyReLU(0.1).

    Used by the YOLOv4 neck (SPP, PANet) and the prediction heads.  The
    backbone (CSPDarknet-53) uses ``_ConvBnMish`` instead — paper §3.4.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        stride: int = 1,
        padding: int = -1,
    ) -> None:
        super().__init__()
        pad = padding if padding >= 0 else (kernel - 1) // 2
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return F.leaky_relu(self.bn(self.conv(x)), negative_slope=0.1)


@final
class _ConvBnMish(nn.Module):
    """Conv2d(bias=False) → BatchNorm2d → Mish.

    YOLOv4 backbone activation per paper §3.4: ``Mish(x) = x · tanh(softplus(x))``.
    Empirically smoother gradient flow than LeakyReLU in deep CSP residual stacks.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        stride: int = 1,
        padding: int = -1,
    ) -> None:
        super().__init__()
        pad = padding if padding >= 0 else (kernel - 1) // 2
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.Mish()

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


# ---------------------------------------------------------------------------
# CSP Block — Cross Stage Partial
# ---------------------------------------------------------------------------


@final
class _CSPBottleneck(nn.Module):
    """One CSP bottleneck unit (1×1 → 3×3) used inside _CSPBlock.

    Lives inside the CSPDarknet-53 backbone → uses Mish activation per paper §3.4.
    """

    def __init__(self, ch: int, act: str = "mish", mid: int | None = None) -> None:
        super().__init__()
        # The cfg's residual unit is ``conv 1x1 (branch width)`` ->
        # ``conv 3x3 (branch width)``: the 1x1 does NOT halve.  Stage 1 is the
        # one exception (64-wide branch, 32-wide bottleneck), which is what
        # ``mid`` is for.
        inner = ch if mid is None else mid
        # The comment below used to claim the bottleneck follows the
        # surrounding block; it did not — both convs were hardcoded to Mish,
        # so a neck block built with act="leaky" still ran Mish inside.
        Conv = _ConvBnMish if act == "mish" else _ConvBnLeaky
        self.conv1 = Conv(ch, inner, 1)
        self.conv2 = Conv(inner, ch, 3)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return x + self.conv2(self.conv1(x))


@final
class _CSPBlock(nn.Module):
    """CSP block: split input into two routes, apply residuals to one, concat.

    Route 1 (skip): Conv(in_ch, branch, 1) — direct pass-through.
    Route 2 (main): Conv(in_ch, branch, 1) → n_repeats × _CSPBottleneck
                    → Conv(branch, branch, 1) transition.
    Merge: concat(route1, route2) → Conv(2*branch, in_ch, 1).

    ``branch`` is ``in_ch // 2`` except in stage 1, where the cfg keeps both
    routes at the full width.

    All convs inside the backbone use Mish; the same primitive is reused by
    the neck (PANet, SPP) where it's instantiated with ``act="leaky"``.

    Args:
        in_ch:      Number of input channels.
        n_repeats:  Number of residual bottleneck repeats in route 2.
        act:        ``"mish"`` for the backbone, ``"leaky"`` for the neck.
    """

    def __init__(
        self,
        in_ch: int,
        n_repeats: int,
        act: str = "mish",
        first_stage: bool = False,
    ) -> None:
        super().__init__()
        # Stage 1 of yolov4.cfg keeps both routes at the *full* 64 channels
        # and bottlenecks to 32; stages 2-5 halve each route and keep the
        # bottleneck at the branch width.
        branch = in_ch if first_stage else in_ch // 2
        bmid = in_ch // 2 if first_stage else branch
        Conv = _ConvBnMish if act == "mish" else _ConvBnLeaky
        self.route1 = Conv(in_ch, branch, 1)  # skip branch
        self.route2 = Conv(in_ch, branch, 1)  # main branch
        # Bottlenecks follow the surrounding block's activation.
        self.bottlenecks = nn.Sequential(
            *[_CSPBottleneck(branch, act=act, mid=bmid) for _ in range(n_repeats)]
        )
        # Every cfg stage closes its main branch with a 1x1 transition before
        # the route concat (layers 8 / 18 / 21 / 31).
        self.transition = Conv(branch, branch, 1)
        self.merge = Conv(branch * 2, in_ch, 1)  # after concat

    @override
    def forward(self, x: Tensor) -> Tensor:
        r1 = self.route1(x)
        r2 = self.bottlenecks(self.route2(x))
        r2 = self.transition(r2)
        merged = lucid.cat([r1, r2], dim=1)
        return self.merge(merged)


# ---------------------------------------------------------------------------
# CSPDarknet-53 backbone
# ---------------------------------------------------------------------------


@final
class _CSPDarknet53(nn.Module):
    """CSPDarknet-53 backbone.

    Returns three feature maps:
      P3 : (B, 256,  H/8,  W/8)
      P4 : (B, 512,  H/16, W/16)
      P5 : (B, 1024, H/32, W/32)
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        # Stem + every stride-2 down conv + every CSP block use Mish (paper §3.4).
        self.stem = _ConvBnMish(in_channels, 32, 3)

        # Stage 1: stride-2 → 64ch, 1×CSP
        self.down1 = _ConvBnMish(32, 64, 3, stride=2)
        self.csp1 = _CSPBlock(64, 1, act="mish", first_stage=True)

        # Stage 2: stride-2 → 128ch, 2×CSP
        self.down2 = _ConvBnMish(64, 128, 3, stride=2)
        self.csp2 = _CSPBlock(128, 2, act="mish")

        # Stage 3: stride-2 → 256ch, 8×CSP  [P3]
        self.down3 = _ConvBnMish(128, 256, 3, stride=2)
        self.csp3 = _CSPBlock(256, 8, act="mish")

        # Stage 4: stride-2 → 512ch, 8×CSP  [P4]
        self.down4 = _ConvBnMish(256, 512, 3, stride=2)
        self.csp4 = _CSPBlock(512, 8, act="mish")

        # Stage 5: stride-2 → 1024ch, 4×CSP  [P5]
        self.down5 = _ConvBnMish(512, 1024, 3, stride=2)
        self.csp5 = _CSPBlock(1024, 4, act="mish")

    @override
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x = self.stem(x)
        x = self.csp1(self.down1(x))
        x = self.csp2(self.down2(x))
        x = self.csp3(self.down3(x))
        p3 = x  # (B, 256,  H/8,  W/8)
        x = self.csp4(self.down4(x))
        p4 = x  # (B, 512,  H/16, W/16)
        x = self.csp5(self.down5(x))
        p5 = x  # (B, 1024, H/32, W/32)
        return p3, p4, p5


# ---------------------------------------------------------------------------
# SPP module (Spatial Pyramid Pooling)
# ---------------------------------------------------------------------------


@final
class _SPP(nn.Module):
    """Spatial Pyramid Pooling module used at P5.

    Pools input with MaxPool(k=5), MaxPool(k=9), MaxPool(k=13), then
    concatenates with the original feature map.  This 4× channel expansion
    is followed by a compress Conv to restore the original channel count.

    Args:
        in_ch:  Input channels (1024 for P5).
        out_ch: Output channels (512 — half of in_ch).
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        # Pre-SPP compress: in_ch → in_ch//2
        half = in_ch // 2
        self.pre = nn.Sequential(
            _ConvBnLeaky(in_ch, half, 1),
            _ConvBnLeaky(half, in_ch, 3),
            _ConvBnLeaky(in_ch, half, 1),
        )
        # MaxPool at 3 kernel sizes (same-padding to preserve spatial dims)
        self.pool5 = nn.MaxPool2d(5, stride=1, padding=2)
        self.pool9 = nn.MaxPool2d(9, stride=1, padding=4)
        self.pool13 = nn.MaxPool2d(13, stride=1, padding=6)
        # Post-SPP: the cfg's layers 114-116 are 512(1x1) / 1024(3x3) /
        # 512(1x1) — the 1x1 compresses the 2048-wide concat straight to
        # ``out_ch`` and the 3x3 expands to ``2*out_ch``, rather than routing
        # the whole thing through ``in_ch`` first.
        self.post = nn.Sequential(
            _ConvBnLeaky(half * 4, out_ch, 1),
            _ConvBnLeaky(out_ch, out_ch * 2, 3),
            _ConvBnLeaky(out_ch * 2, out_ch, 1),
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = self.pre(x)  # (B, half, H, W)
        p5 = self.pool5(x)
        p9 = self.pool9(x)
        p13 = self.pool13(x)
        concat = lucid.cat([x, p5, p9, p13], dim=1)  # (B, 4*half, H, W)
        return self.post(concat)  # (B, out_ch, H, W)


def _five_conv(in_ch: int, mid: int) -> nn.Sequential:
    """The cfg's alternating 1x1/3x3 five-convolution set.

    ``yolov4.cfg``'s neck is built entirely from these; it contains no CSP
    blocks at all.  Each set compresses to ``mid``, expands to ``2*mid`` and
    back, twice, ending at ``mid``.
    """
    return nn.Sequential(
        _ConvBnLeaky(in_ch, mid, 1),
        _ConvBnLeaky(mid, mid * 2, 3),
        _ConvBnLeaky(mid * 2, mid, 1),
        _ConvBnLeaky(mid, mid * 2, 3),
        _ConvBnLeaky(mid * 2, mid, 1),
    )


# ---------------------------------------------------------------------------
# PANet neck
# ---------------------------------------------------------------------------


@final
class _PANetNeck(nn.Module):
    """Path Aggregation Network neck.

    Connects the CSPDarknet-53 backbone outputs (P3, P4, P5) through:
      1. SPP at P5.
      2. Top-down pathway: P5→P4'→P3' (FPN-style).
      3. Bottom-up pathway: P3'→P4''→P5'' (PAN-style).

    Every stage is one of the cfg's alternating five-convolution sets; the
    reference neck contains no CSP blocks.

    Output channels:
      P3'' : 128ch
      P4'' : 256ch
      P5'' : 512ch
    """

    def __init__(self) -> None:
        super().__init__()
        # SPP at P5 (1024→512ch) — cfg layer 116.
        self.spp = _SPP(1024, 512)

        # Top-down: SPP(512) -1x1-> 256, upsample, concat P4 lateral(256)
        self.p5_lateral = _ConvBnLeaky(512, 256, 1)
        self.p4_lateral = _ConvBnLeaky(512, 256, 1)
        self.p4_td = _five_conv(512, 256)  # cfg layer 126, 256ch out
        self.p4_td_lat = _ConvBnLeaky(256, 128, 1)

        # Top-down: P4'(256) -1x1-> 128, upsample, concat P3 lateral(128)
        self.p3_lateral = _ConvBnLeaky(256, 128, 1)
        self.p3_td = _five_conv(256, 128)  # cfg layer 136, 128ch out

        # Bottom-up.  The cfg's routes are ``-1,-16`` and ``-1,-37``: the P4
        # concat takes the *top-down P4 output* (layer 126) and the P5 concat
        # takes the *SPP output* (layer 116).  Aggregating the compressed
        # laterals instead re-used tensors that had already been narrowed for
        # upsampling, so the bottom-up path never saw the top-down result it
        # is supposed to refine.
        self.p3_down = _ConvBnLeaky(128, 256, 3, stride=2)
        self.p4_bu = _five_conv(512, 256)  # cfg layer 147, 256ch out

        self.p4_down = _ConvBnLeaky(256, 512, 3, stride=2)
        self.p5_bu = _five_conv(1024, 512)  # cfg layer 158, 512ch out

    @override
    def forward(
        self,
        p3: Tensor,
        p4: Tensor,
        p5: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run PANet.

        Args:
            p3: (B, 256,  H/8,  W/8)
            p4: (B, 512,  H/16, W/16)
            p5: (B, 1024, H/32, W/32)

        Returns:
            (p3_out, p4_out, p5_out) — feature maps for detection heads.
        """
        # SPP at P5 — cfg layer 116, the tensor the bottom-up P5 route reads.
        p5_spp = self.spp(p5)  # (B, 512, H/32, W/32)

        # Top-down P5→P4'
        p5_lat = self.p5_lateral(p5_spp)  # (B, 256, H/32, W/32)
        fH4 = int(p4.shape[2])
        fW4 = int(p4.shape[3])
        p5_up = F.interpolate(
            p5_lat, size=(fH4, fW4), mode="nearest"
        )  # (B, 256, H/16, W/16)
        p4_comp = self.p4_lateral(p4)  # (B, 256, H/16, W/16)
        p4_cat = lucid.cat([p5_up, p4_comp], dim=1)  # (B, 512, H/16, W/16)
        p4_td_out = self.p4_td(p4_cat)  # (B, 256, H/16, W/16)

        # Top-down P4'→P3'
        p4_lat = self.p4_td_lat(p4_td_out)  # (B, 128, H/16, W/16)
        fH3 = int(p3.shape[2])
        fW3 = int(p3.shape[3])
        p4_up = F.interpolate(
            p4_lat, size=(fH3, fW3), mode="nearest"
        )  # (B, 128, H/8, W/8)
        p3_lat = self.p3_lateral(p3)  # (B, 128, H/8, W/8)
        p3_cat = lucid.cat([p4_up, p3_lat], dim=1)  # (B, 256, H/8, W/8)
        p3_td_out = self.p3_td(p3_cat)  # (B, 128, H/8, W/8)

        # Bottom-up P3'→P4'': route ``-1,-16`` joins the top-down P4 output.
        p3_down_feat = self.p3_down(p3_td_out)  # (B, 256, H/16, W/16)
        p4_bu_cat = lucid.cat([p3_down_feat, p4_td_out], dim=1)  # (B, 512, ...)
        p4_bu_out = self.p4_bu(p4_bu_cat)  # (B, 256, H/16, W/16)

        # Bottom-up P4''→P5'': route ``-1,-37`` joins the SPP output.
        p4_down_feat = self.p4_down(p4_bu_out)  # (B, 512, H/32, W/32)
        p5_bu_cat = lucid.cat([p4_down_feat, p5_spp], dim=1)  # (B, 1024, ...)
        p5_bu_out = self.p5_bu(p5_bu_cat)  # (B, 512, H/32, W/32)

        return p3_td_out, p4_bu_out, p5_bu_out


# ---------------------------------------------------------------------------
# Box decoding helper (shared with YOLOv3 style)
# ---------------------------------------------------------------------------


# yolov4.cfg's per-scale grid-sensitivity factors, keyed by stride.  The BoF
# item "eliminate grid sensitivity" scales the sigmoid by >1 so a box centre
# can actually reach a cell boundary — with a plain sigmoid it needs an
# infinite logit to land exactly on the grid line.
_SCALE_X_Y: dict[int, float] = {8: 1.2, 16: 1.1, 32: 1.05}


def _decode_predictions(
    raw: Tensor,
    anchors_wh: list[tuple[float, float]],
    stride: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Decode a single-scale raw detection tensor.

    Args:
        raw:         (B, nA*(5+C), H, W)
        anchors_wh:  3 anchor (w, h) pairs for this scale.
        stride:      Feature-map stride.

    Returns:
        (logits, pred_boxes, conf):
          logits    : (B, H*W*nA, C)  raw class logits (pre-sigmoid)
          pred_boxes: (B, H*W*nA, 4)  decoded xyxy pixel boxes
          conf      : (B, H*W*nA)     sigmoid objectness confidence
    """
    B = int(raw.shape[0])
    fH = int(raw.shape[2])
    fW = int(raw.shape[3])
    nA = len(anchors_wh)
    C = int(raw.shape[1]) // nA - 5
    # Grids and priors take the head's device and dtype.
    device, dt = raw.device, raw.dtype

    raw = raw.reshape(B, nA, 5 + C, fH, fW).permute(0, 1, 3, 4, 2)

    tx = raw[..., 0]
    ty = raw[..., 1]
    tw = raw[..., 2]
    th = raw[..., 3]
    tc = raw[..., 4]
    cls_logits = raw[..., 5:]

    col_data: list[list[float]] = [[float(c) for c in range(fW)] for _ in range(fH)]
    row_data: list[list[float]] = [[float(r)] * fW for r in range(fH)]
    col_t = lucid.tensor(col_data, dtype=dt, device=device)
    row_t = lucid.tensor(row_data, dtype=dt, device=device)

    # bx — scale * sigmoid(tx) - 0.5 * (scale - 1) + cx
    scale = _SCALE_X_Y.get(stride, 1.0)
    bias = 0.5 * (scale - 1.0)
    px = (scale * F.sigmoid(tx) - bias + col_t) * float(stride)
    py = (scale * F.sigmoid(ty) - bias + row_t) * float(stride)

    aw_data: list[list[list[list[float]]]] = []
    ah_data: list[list[list[list[float]]]] = []
    for _ in range(B):
        b_aw: list[list[list[float]]] = []
        b_ah: list[list[list[float]]] = []
        for a_idx in range(nA):
            aw_val = anchors_wh[a_idx][0]
            ah_val = anchors_wh[a_idx][1]
            b_aw.append([[aw_val] * fW for _ in range(fH)])
            b_ah.append([[ah_val] * fW for _ in range(fH)])
        aw_data.append(b_aw)
        ah_data.append(b_ah)

    aw_t = lucid.tensor(aw_data, dtype=dt, device=device)
    ah_t = lucid.tensor(ah_data, dtype=dt, device=device)

    pw = lucid.exp(tw) * aw_t
    ph = lucid.exp(th) * ah_t

    x1 = px - pw / 2.0
    y1 = py - ph / 2.0
    x2 = px + pw / 2.0
    y2 = py + ph / 2.0

    boxes = lucid.stack([x1, y1, x2, y2], dim=-1).reshape(B, nA * fH * fW, 4)
    cls_logits = cls_logits.reshape(B, nA * fH * fW, C)
    conf = F.sigmoid(tc.reshape(B, nA * fH * fW))

    return cls_logits, boxes, conf


# ---------------------------------------------------------------------------
# CIoU loss helper
# ---------------------------------------------------------------------------


def _ciou_loss(pred_boxes: Tensor, gt_boxes: Tensor) -> Tensor:
    """Complete IoU loss between paired predicted and GT boxes (xyxy format).

    CIoU = 1 - IoU + d²/c² + α·v
    where:
      d²  = squared Euclidean distance between box centres
      c²  = squared diagonal of the smallest enclosing box
      v   = (4/π²) * (arctan(w_gt/h_gt) - arctan(w_pred/h_pred))²
      α   = v / (1 - IoU + v)

    Args:
        pred_boxes: (N, 4) xyxy predicted boxes.
        gt_boxes:   (N, 4) xyxy ground-truth boxes.

    Returns:
        Scalar mean CIoU loss.
    """
    N = int(pred_boxes.shape[0])
    if N == 0:
        return pred_boxes.new_zeros(1)

    # Vectorised and differentiable.  The previous per-box Python-float form
    # rebuilt each term with ``lucid.tensor``, so the box-regression channels
    # received exactly zero gradient while the loss value still looked right.
    px1, py1, px2, py2 = (pred_boxes[:, i] for i in range(4))
    gx1, gy1, gx2, gy2 = (gt_boxes[:, i] for i in range(4))

    pw, ph = px2 - px1, py2 - py1
    gw, gh = gx2 - gx1, gy2 - gy1
    pcx, pcy = (px1 + px2) * 0.5, (py1 + py2) * 0.5
    gcx, gcy = (gx1 + gx2) * 0.5, (gy1 + gy2) * 0.5

    inter_w = lucid.clip(lucid.minimum(px2, gx2) - lucid.maximum(px1, gx1), 0.0, None)
    inter_h = lucid.clip(lucid.minimum(py2, gy2) - lucid.maximum(py1, gy1), 0.0, None)
    inter = inter_w * inter_h
    iou = inter / (pw * ph + gw * gh - inter + 1e-9)

    enc_w = lucid.maximum(px2, gx2) - lucid.minimum(px1, gx1)
    enc_h = lucid.maximum(py2, gy2) - lucid.minimum(py1, gy1)
    c_sq = enc_w * enc_w + enc_h * enc_h + 1e-9
    d_sq = (pcx - gcx) ** 2 + (pcy - gcy) ** 2

    v = (4.0 / (math.pi**2)) * (
        lucid.arctan(gw / (gh + 1e-9)) - lucid.arctan(pw / (ph + 1e-9))
    ) ** 2
    # α is a weighting coefficient, not a path to differentiate through —
    # detached exactly as in the reference CIoU implementation.
    alpha = (v / (1.0 - iou + v + 1e-9)).detach()

    return (1.0 - (iou - d_sq / c_sq - alpha * v)).mean()


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


# yolov4.cfg's ``mask =`` per [yolo] layer, in ``strides`` order (P3 first).
_V4_ANCHOR_MASKS: tuple[tuple[int, ...], ...] = ((0, 1, 2), (3, 4, 5), (6, 7, 8))


def _v4_scales(
    config: YOLOV4Config,
) -> list[tuple[tuple[int, ...], list[tuple[float, float]], int]]:
    """The config's priors and strides per head, coarse to fine (P5 first)."""
    return _scale_priors(config.anchors, _V4_ANCHOR_MASKS, config.strides)


def _yolov4_loss(
    raw_preds: list[Tensor],
    targets: list[dict[str, Tensor]],
    config: YOLOV4Config,
) -> Tensor:
    """YOLOv4 multi-scale detection loss with CIoU box regression.

    Args:
        raw_preds: 3 raw tensors (P5, P4, P3) each (B, nA*(5+C), H_l, W_l).
        targets:   List of B target dicts.
        config:    Model configuration.

    Returns:
        Scalar loss tensor.
    """
    B = int(raw_preds[0].shape[0])
    C = config.num_classes
    nA = 3

    total_loss: list[Tensor] = []

    # Three heads, and the config holds exactly three strides — same length
    # by construction.
    for raw, (_, anchors_wh, stride) in zip(raw_preds, _v4_scales(config), strict=True):
        fH = int(raw.shape[2])
        fW = int(raw.shape[3])

        raw_r = raw.reshape(B, nA, 5 + C, fH, fW).permute(0, 1, 3, 4, 2)
        # The ignore mask compares the boxes this head actually predicts —
        # scale_x_y included — with the GTs; a selection, so no gradient.
        _, pred_boxes, _ = _decode_predictions(raw.detach(), anchors_wh, stride)

        for b in range(B):
            gt_boxes = targets[b]["boxes"]
            M = int(gt_boxes.shape[0])

            # Build target arrays
            obj_arr: list[list[list[float]]] = [
                [[0.0] * fW for _ in range(fH)] for _ in range(nA)
            ]
            cls_arr: list[list[list[list[float]]]] = [
                [[[0.0] * C for _ in range(fW)] for _ in range(fH)] for _ in range(nA)
            ]
            mask_arr: list[list[list[float]]] = [
                [[0.0] * fW for _ in range(fH)] for _ in range(nA)
            ]

            # Collect positive assignments for CIoU.  Only *indices* and the
            # matched anchor sizes are recorded here — the predicted box itself
            # is decoded from ``pred_b`` with tensor ops after the loop, so the
            # box-regression channels stay connected to the loss.
            pos_cells: list[tuple[int, int, int]] = []
            pos_anchor_wh: list[tuple[float, float]] = []
            pos_gt_boxes: list[list[float]] = []

            pred_b = raw_r[b]  # (nA, fH, fW, 5+C)

            if M > 0:
                for (x1g, y1g, x2g, y2g), cls_id in _gt_rows(targets[b]):
                    wg = x2g - x1g
                    hg = y2g - y1g
                    cxg = (x1g + x2g) / 2.0
                    cyg = (y1g + y2g) / 2.0

                    col_idx = max(0, min(int(cxg / stride), fW - 1))
                    row_idx = max(0, min(int(cyg / stride), fH - 1))

                    best_iou = -1.0
                    best_a = 0
                    for a_i, (aw, ah) in enumerate(anchors_wh):
                        inter_w = min(wg, aw)
                        inter_h = min(hg, ah)
                        inter = inter_w * inter_h
                        union = wg * hg + aw * ah - inter
                        iou_val = inter / (union + 1e-6)
                        if iou_val > best_iou:
                            best_iou = iou_val
                            best_a = a_i

                    aw_best, ah_best = anchors_wh[best_a]

                    obj_arr[best_a][row_idx][col_idx] = 1.0
                    mask_arr[best_a][row_idx][col_idx] = 1.0
                    if 0 <= cls_id < C:
                        cls_arr[best_a][row_idx][col_idx][cls_id] = 1.0

                    pos_cells.append((best_a, row_idx, col_idx))
                    pos_anchor_wh.append((aw_best, ah_best))
                    pos_gt_boxes.append([x1g, y1g, x2g, y2g])

            dt, dev = raw.dtype, raw.device  # targets in the head's dtype
            tgt_obj = lucid.tensor(obj_arr, dtype=dt, device=dev)
            tgt_cls = lucid.tensor(cls_arr, dtype=dt, device=dev)
            obj_mask = lucid.tensor(mask_arr, dtype=dt, device=dev)

            pred_tc = pred_b[..., 4]  # (nA, fH, fW)
            pred_cls_logits = pred_b[..., 5:]  # (nA, fH, fW, C)

            # Objectness BCE
            obj_bce = F.binary_cross_entropy_with_logits(
                pred_tc, tgt_obj, reduction="none"
            )
            # yolov4.cfg sets ``ignore_thresh = .7`` on every [yolo] layer:
            # a non-assigned anchor that already overlaps a GT well is excluded
            # from the background term rather than pushed toward zero.
            ignore = _ignore_mask(pred_boxes[b], gt_boxes, _IGNORE_IOU_THRESH).reshape(
                nA, fH, fW
            )
            noobj_mask = (1.0 - obj_mask) * (1.0 - ignore)
            obj_loss = (
                obj_bce * obj_mask + obj_bce * noobj_mask * config.lambda_noobj
            ).sum()

            # Class BCE
            cls_bce = F.binary_cross_entropy_with_logits(
                pred_cls_logits, tgt_cls, reduction="none"
            )
            cls_loss = (cls_bce * obj_mask[..., None]).sum()

            # CIoU loss for positive anchors.  Decode σ(t_x)/σ(t_y)/exp(t_w)/
            # exp(t_h) with tensor ops so the gradient reaches the box channels.
            if pos_cells:

                tx, ty, tw, th = [
                    lucid.cat(
                        [pred_b[a, r, c, j].reshape(1) for (a, r, c) in pos_cells]
                    )
                    for j in range(4)
                ]
                cells = lucid.tensor(pos_cells, dtype=dt, device=dev)  # (a, r, c)
                rows, cols = cells[:, 1], cells[:, 2]
                anc = lucid.tensor(pos_anchor_wh, dtype=dt, device=dev)
                anc_w, anc_h = anc[:, 0], anc[:, 1]

                _sc = _SCALE_X_Y.get(stride, 1.0)
                _bias = 0.5 * (_sc - 1.0)
                pcx = (_sc * lucid.sigmoid(tx) - _bias + cols) * float(stride)
                pcy = (_sc * lucid.sigmoid(ty) - _bias + rows) * float(stride)
                p_w = lucid.exp(tw) * anc_w
                p_h = lucid.exp(th) * anc_h

                p_boxes = lucid.stack(
                    [
                        pcx - p_w / 2.0,
                        pcy - p_h / 2.0,
                        pcx + p_w / 2.0,
                        pcy + p_h / 2.0,
                    ],
                    dim=1,
                )  # (P, 4)
                g_boxes = lucid.tensor(pos_gt_boxes, dtype=dt, device=dev)  # (P, 4)
                ciou_l = _ciou_loss(p_boxes, g_boxes)
            else:
                ciou_l = raw.new_zeros(1)

            scale_loss = ciou_l.sum() + obj_loss + cls_loss
            total_loss.append(scale_loss.reshape(1))

    if not total_loss:
        return raw_preds[0].new_zeros(1)
    return lucid.cat(total_loss).sum()


# ---------------------------------------------------------------------------
# YOLOv4 model
# ---------------------------------------------------------------------------


class YOLOV4ForObjectDetection(ObjectDetectionModel):
    r"""YOLOv4 multi-scale object detector (Bochkovskiy et al., 2020).

    A heavily engineered iteration over YOLOv3 that combines several
    independently-published improvements ("bag of freebies" and "bag
    of specials" in the paper's terminology) into a single
    high-throughput detector.  The core architectural changes are:

    - **CSPDarknet-53** backbone — replaces residual blocks with
      Cross-Stage-Partial blocks that split and re-merge the feature
      stream, reducing FLOPs without hurting accuracy.
    - **SPP** (Spatial Pyramid Pooling) module on the final backbone
      stage — fuses three max-pooled receptive fields plus the identity,
      widening the effective receptive field.
    - **PANet** (Path Aggregation Network) neck — adds a bottom-up path
      on top of the FPN top-down path, giving each prediction level
      access to both fine and coarse features.

    Heads remain YOLOv3-style (three scales, 3 anchors / scale), but
    training switches the box-regression loss to **CIoU** which couples
    centre distance, IoU, and aspect-ratio consistency in a single
    differentiable objective.  COCO test-dev AP of 43.5% at 65 fps on a
    Tesla V100 (paper Table 8).

    Parameters
    ----------
    config : YOLOV4Config
        Frozen architecture spec.  Use :func:`yolo_v4` for the standard
        full-size model.

    Attributes
    ----------
    config : YOLOV4Config
        Stored copy of the config that built this model.
    backbone : _CSPDarknet53
        Cross-Stage-Partial Darknet-53 producing P3 / P4 / P5 features.
    neck : _PANetNeck
        SPP + PANet (top-down FPN + bottom-up path) producing the three
        head input features.
    p3_head, p4_head, p5_head : nn.Sequential
        Three-scale prediction heads, each producing :math:`3 (5 + C)`
        channels.

    Notes
    -----
    See Bochkovskiy et al., "YOLOv4: Optimal Speed and Accuracy of
    Object Detection", arXiv 2020 (arXiv:2004.10934).  Complete-IoU
    (CIoU) loss is defined as

    .. math::

        \mathcal{L}_\mathrm{CIoU} =
            1 - \mathrm{IoU} + \frac{\rho^2(b, b^\mathrm{gt})}{c^2}
            + \alpha v,
        \qquad
        v = \frac{4}{\pi^2}
            \Bigl(\arctan\frac{w^\mathrm{gt}}{h^\mathrm{gt}} -
                  \arctan\frac{w}{h}\Bigr)^2,

    where :math:`\rho` is the Euclidean distance between box centres,
    :math:`c` is the diagonal of the smallest enclosing box, and
    :math:`\alpha` is a balancing trade-off term.  CIoU converges faster
    than IoU / GIoU and is the standard regression objective from
    YOLOv4 onwards.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.yolo._v4 import yolo_v4
    >>> model = yolo_v4()
    >>> x = lucid.randn(1, 3, 608, 608)
    >>> out = model(x)
    >>> out.logits.shape[0]
    1
    """

    config_class: ClassVar[type[YOLOV4Config]] = YOLOV4Config
    base_model_prefix: ClassVar[str] = "yolo_v4"

    def __init__(self, config: YOLOV4Config) -> None:
        super().__init__(config)
        self._cfg = config
        C = config.num_classes
        nA = 3

        # CSPDarknet-53 backbone
        self.backbone = _CSPDarknet53(config.in_channels)

        # PANet neck
        self.neck = _PANetNeck()

        # Detection heads.  Each scale's five-conv set already did the
        # compressing, so the cfg's head is just ``conv 3x3`` then the linear
        # ``conv 1x1`` predictor.
        #   P3'' = 128ch, P4'' = 256ch, P5'' = 512ch
        self.p3_head = nn.Sequential(
            _ConvBnLeaky(128, 256, 3),
            nn.Conv2d(256, nA * (5 + C), 1, bias=True),
        )
        self.p4_head = nn.Sequential(
            _ConvBnLeaky(256, 512, 3),
            nn.Conv2d(512, nA * (5 + C), 1, bias=True),
        )
        self.p5_head = nn.Sequential(
            _ConvBnLeaky(512, 1024, 3),
            nn.Conv2d(1024, nA * (5 + C), 1, bias=True),
        )

    @override
    def forward(
        self,
        x: Tensor,
        targets: list[dict[str, Tensor]] | None = None,
    ) -> ObjectDetectionOutput:
        """Run YOLOv4.

        Args:
            x:       (B, C, H, W) image batch.
            targets: Optional list of target dicts per image.

        Returns:
            ``ObjectDetectionOutput``:
              ``logits``    : (B, total_anchors, C) raw class logits.
              ``pred_boxes``: (B, total_anchors, 4) xyxy decoded boxes.
              ``loss``      : loss scalar when targets provided.
        """
        cfg = self._cfg
        check_per_image(type(self).__name__, int(x.shape[0]), targets=targets)

        # Backbone
        p3, p4, p5 = self.backbone(x)

        # PANet neck
        p3_out, p4_out, p5_out = self.neck(p3, p4, p5)

        # Detection heads (P5→large, P4→medium, P3→small)
        p5_raw = self.p5_head(p5_out)  # (B, nA*(5+C), H/32, W/32)
        p4_raw = self.p4_head(p4_out)  # (B, nA*(5+C), H/16, W/16)
        p3_raw = self.p3_head(p3_out)  # (B, nA*(5+C), H/8,  W/8)

        # raw_preds order: P5, P4, P3 (coarse→fine)
        raw_preds = [p5_raw, p4_raw, p3_raw]

        all_logits: list[Tensor] = []
        all_boxes: list[Tensor] = []
        all_conf: list[Tensor] = []

        for raw_pred, (_, anch_wh, stride) in zip(
            raw_preds, _v4_scales(cfg), strict=True
        ):
            logits_s, boxes_s, conf_s = _decode_predictions(raw_pred, anch_wh, stride)
            all_logits.append(logits_s)
            all_boxes.append(boxes_s)
            all_conf.append(conf_s)

        logits = lucid.cat(all_logits, dim=1)
        pred_boxes = lucid.cat(all_boxes, dim=1)
        objectness = lucid.cat(all_conf, dim=1)

        loss: Tensor | None = None
        if targets is not None:
            loss = _yolov4_loss(raw_preds, targets, cfg)

        return ObjectDetectionOutput(
            logits=logits,
            pred_boxes=pred_boxes,
            loss=loss,
            objectness=objectness,
        )

    def postprocess(
        self,
        output: ObjectDetectionOutput,
        image_sizes: list[tuple[int, int]],
    ) -> list[dict[str, Tensor]]:
        """Filter by score, clip boxes, apply per-class NMS.

        Args:
            output:      Forward pass output.
            image_sizes: List of (H, W) per image.

        Returns:
            Per-image list of dicts with "boxes", "scores", "labels", on
            the model's device.
        """
        # score = sigmoid(objectness) * sigmoid(class), as darknet's
        # ``get_yolo_detections`` does — objectness is the only head in v4
        # that receives box-quality supervision.  The scoring, threshold and
        # NMS are YOLOv3's, shared so the device handling and the single
        # read-back of the score matrix cannot drift between versions.
        return _postprocess_detections(
            output, image_sizes, self._cfg.score_thresh, self._cfg.nms_thresh
        )


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


_CFG_V4 = YOLOV4Config()


@register_model(
    task="object-detection",
    family="yolo",
    model_type="yolo_v4",
    model_class=YOLOV4ForObjectDetection,
    default_config=_CFG_V4,
    params=64363101,
)
def yolo_v4(
    pretrained: bool | str = False,
    *,
    weights: YOLOV4Weights | None = None,
    **overrides: object,
) -> YOLOV4ForObjectDetection:
    r"""YOLOv4 — CSPDarknet-53 + SPP + PANet (Bochkovskiy et al., 2020).

    Builds the paper-cited full-size YOLOv4 detector: CSPDarknet-53
    backbone, SPP module on the final stage, PANet (top-down +
    bottom-up) neck, and 3-scale detection at strides 8 / 16 / 32 with
    3 anchors / scale.  Default 80 COCO classes; reaches COCO test-dev
    AP of 43.5% at 65 fps on Tesla V100 (paper Table 8).

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True`` →
        the ``DEFAULT`` tag (:attr:`YOLOV4Weights.COCO_2017`, converted
        from the AlexeyAB darknet release's ``yolov4.weights``); a tag
        string → that specific checkpoint.  Mutually exclusive with
        ``weights`` (which wins if both are given).
    weights : YOLOV4Weights, optional, keyword-only
        Explicit weights enum member, e.g. ``YOLOV4Weights.COCO_2017``.
        Takes precedence over ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`YOLOV4Config`.

    Returns
    -------
    YOLOV4ForObjectDetection
        Detector with the standard YOLOv4 configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See Bochkovskiy et al., "YOLOv4: Optimal Speed and Accuracy of
    Object Detection", arXiv 2020 (arXiv:2004.10934).  The paper's
    "bag of specials" includes Mish activation, CIoU loss, DropBlock
    regularisation, Mosaic augmentation, and CmBN — all motivated by
    independent prior work that YOLOv4 aggregates and tunes together.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.yolo._v4 import yolo_v4
    >>> model = yolo_v4()
    >>> x = lucid.randn(1, 3, 608, 608)
    >>> out = model(x)
    >>> out.logits.shape[0]
    1
    """
    config = apply_overrides(_CFG_V4, overrides)
    entry = weights_mod.resolve_weights(YOLOV4Weights, pretrained, weights)
    model = YOLOV4ForObjectDetection(config)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="yolo_v4")
    return model
