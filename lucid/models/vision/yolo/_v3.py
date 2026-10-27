"""YOLOv3 — You Only Look Once v3 (Redmon & Farhadi, arXiv 2018).

Paper: "YOLOv3: An Incremental Improvement"
https://arxiv.org/abs/1804.02767

Key design
----------
YOLOv3 extends YOLO by:
  1. Darknet-53 backbone — 53-layer convolutional network with residual
     connections and no pooling (stride-2 convolutions for downsampling).
  2. Multi-scale detection at 3 granularities (P3/P4/P5 — strides 8/16/32),
     each with 3 anchors, enabling small/medium/large object detection.
  3. Independent sigmoid activations per class (multi-label) rather than
     softmax, making class predictions class-independent.

Architecture overview
---------------------
  Image (B, 3, H, W)
    ↓  Darknet-53 backbone (stages 1–5)
       - Stem:  Conv(3→32, k3) → Conv(32→64, k3, stride=2)
       - Stage1: 1×dark53_block(64→64)    → stride-2 conv → 128ch
       - Stage2: 2×dark53_block(128→128)  → stride-2 conv → 256ch  [P3_raw]
       - Stage3: 8×dark53_block(256→256)  → stride-2 conv → 512ch  [P4_raw]
       - Stage4: 8×dark53_block(512→512)  → stride-2 conv → 1024ch
       - Stage5: 4×dark53_block(1024→1024) [P5]
    ↓  Detection heads at P5, P4, P3
       - P5:  5×Conv(1×1/3×3 alternating) → 1×Conv → 3*(5+C) predictions
       - P4:  P5_feature upsample+concat(P4_raw) → same head sequence
       - P3:  P4_feature upsample+concat(P3_raw) → same head sequence
    ↓  Decode each scale: sigmoid(tx,ty)+offset, exp(tw,th)*anchor, sigmoid(obj), sigmoid(cls)
    ↓  ObjectDetectionOutput(logits, pred_boxes, loss)

Loss (training)
---------------
  Requires ``targets`` — list of B dicts:
    "boxes"  : (M, 4) xyxy pixel coordinates
    "labels" : (M,)   integer class ids (0-indexed)

  Assignment: for each GT box, the anchor with highest IoU at the matching
  grid cell is made a positive. All other anchors are negatives.
  L = lambda_coord * L_mse(box)   [positive anchors only]
    + L_bce(obj)                  [positive=1, negative=lambda_noobj scaled]
    + L_bce(cls)                  [positive anchors only]
"""

import math
from dataclasses import dataclass
from typing import ClassVar, cast, final, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta
from lucid.models._output import ObjectDetectionOutput
from lucid.models._registry import register_model
from lucid.models._tasks import ObjectDetectionModel
from lucid.models._utils._detection import batched_nms, clip_boxes_to_image, box_iou
from lucid.models._utils._detection_results import (
    check_per_image,
    detection_result,
    empty_detection_result,
)
from lucid.models.vision.yolo._weights import YOLOV3TinyWeights, YOLOV3Weights
from lucid.models._utils._common import apply_overrides
import lucid.weights as weights_mod

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
    YOLOv3 scales up the one-stage YOLO recipe with a deeper backbone
    and **multi-scale prediction**, eliminating the original family's
    main blind spots on small objects.

    **Darknet-53 backbone.**  53 conv layers organised into residual
    stacks (1× / 2× / 8× / 8× / 4×) with stride-2 downsampling between
    stages.  Throughput is comparable to ResNet-101 at ResNet-152
    accuracy.

    **Multi-scale FPN-style head.**  Predictions are made at three
    feature levels :math:`(P_3, P_4, P_5)` with strides 8, 16, 32.
    Each level uses 3 anchors (9 in total, clustered from the dataset
    by k-means) so small / medium / large objects are detected on
    high- / mid- / low-resolution maps.  Top-down upsampling +
    concatenation with the corresponding backbone tap mirrors the
    FPN idea.

    **Independent class predictors.**  v3 drops the softmax in
    favour of independent **logistic classifiers** per class, using
    binary cross-entropy.  This handles overlapping labels (e.g.
    "woman" and "person") and prepares the model for multi-label
    datasets like Open Images.

    Each cell still predicts :math:`(t_x, t_y, t_w, t_h, t_o)` decoded
    against its anchor exactly as in v2:

    .. math::

        b_x = \sigma(t_x) + c_x, \quad b_y = \sigma(t_y) + c_y, \quad
        b_w = p_w e^{t_w}, \quad b_h = p_h e^{t_h}.

    The combination of multi-scale prediction + better backbone makes
    v3 substantially better on small objects, the chronic weakness
    of v1/v2.
    """,
)
@dataclass(frozen=True, slots=True)
class YOLOV3Config(ModelConfig):
    """Configuration for YOLOv3.

    YOLOv3 (Redmon & Farhadi, 2018) uses a Darknet-53 backbone with three
    detection scales at strides 8, 16, and 32 (P3/P4/P5).  Each scale has 3
    hand-picked COCO anchors for small, medium, and large objects respectively.
    YOLOv3-Tiny shares this config with its own table — six anchors, strides
    16 and 32 — which :func:`yolo_v3_tiny` supplies.

    Args:
        num_classes:   Number of foreground classes (COCO default = 80).
        in_channels:   Input image channels.
        anchors:       (width, height) anchor priors in pixels — every prior a
                       ground-truth box is matched against.
        strides:       Feature-map stride of each detection scale, fine to
                       coarse (P3, P4, P5 for the full model).
        score_thresh:  Minimum objectness × class score to keep a detection.
        nms_thresh:    IoU threshold for per-class NMS.
        lambda_coord:  Box regression loss weight.
        lambda_noobj:  Objectness loss weight for negative anchors.
        anchor_masks:  Indices into ``anchors`` predicted at each scale, in
                       ``strides`` order — the cfg's ``mask =`` per ``[yolo]``
                       layer.  Must hold one mask per stride.

    Examples
    --------
    >>> from lucid.models.vision.yolo import YOLOV3Config
    >>> cfg = YOLOV3Config()
    >>> cfg.num_classes
    80
    >>> cfg.model_type
    'yolo_v3'
    """

    model_type: ClassVar[str] = "yolo_v3"

    num_classes: int = 80
    in_channels: int = 3

    # 9 COCO anchor (w, h) pairs — 3 per detection scale
    anchors: tuple[tuple[float, float], ...] = (
        # P3 (stride 8) — small objects
        (10.0, 13.0),
        (16.0, 30.0),
        (33.0, 23.0),
        # P4 (stride 16) — medium objects
        (30.0, 61.0),
        (62.0, 45.0),
        (59.0, 119.0),
        # P5 (stride 32) — large objects
        (116.0, 90.0),
        (156.0, 198.0),
        (373.0, 326.0),
    )

    strides: tuple[int, ...] = (8, 16, 32)
    score_thresh: float = 0.5
    nms_thresh: float = 0.5
    lambda_coord: float = 5.0
    lambda_noobj: float = 0.5
    # Appended last: the released constructor binds the fields above by
    # position.
    anchor_masks: tuple[tuple[int, ...], ...] = ((0, 1, 2), (3, 4, 5), (6, 7, 8))

    def __post_init__(self) -> None:
        object.__setattr__(self, "anchors", tuple(tuple(a) for a in self.anchors))
        object.__setattr__(self, "strides", tuple(self.strides))
        object.__setattr__(self, "anchor_masks", tuple(tuple(m) for m in self.anchor_masks))
        _check_scale_table(type(self).__name__, self.anchors, self.strides, self.anchor_masks)


def _check_scale_table(
    owner: str,
    anchors: tuple[tuple[float, float], ...],
    strides: tuple[int, ...],
    anchor_masks: tuple[tuple[int, ...], ...],
) -> None:
    """Reject a per-scale prior table whose parts disagree.

    The heads, the decode and the loss all walk this table scale by scale;
    a stride without a mask, or a mask naming a prior that does not exist,
    would otherwise surface far from the config as a ``zip`` length error
    or an ``IndexError`` mid-loss.
    """
    if len(anchor_masks) != len(strides):
        raise ValueError(
            f"{owner}: {len(strides)} strides {strides} but {len(anchor_masks)} "
            f"anchor masks {anchor_masks}; each detection scale needs one stride "
            "and one anchor mask"
        )
    for mask in anchor_masks:
        if not mask:
            raise ValueError(f"{owner}: anchor mask is empty in {anchor_masks}")
        if any(not 0 <= i < len(anchors) for i in mask):
            raise ValueError(
                f"{owner}: anchor mask {mask} indexes outside the "
                f"{len(anchors)} anchors (valid indices 0..{len(anchors) - 1})"
            )


def _scale_priors(
    anchors: tuple[tuple[float, float], ...],
    anchor_masks: tuple[tuple[int, ...], ...],
    strides: tuple[int, ...],
) -> list[tuple[tuple[int, ...], list[tuple[float, float]], int]]:
    """``(mask, anchor (w, h) list, stride)`` per scale, coarse to fine.

    The config lists scales fine to coarse (P3 first); the heads emit
    coarse to fine (P5 first), as darknet's ``[yolo]`` layers do.  Every
    consumer reads the table through here so the two orders cannot drift.
    """
    table = zip(anchor_masks, strides, strict=True)
    return [(m, [anchors[i] for i in m], s) for m, s in reversed(list(table))]


def _require_scales(model: str, n_heads: int, strides: tuple[int, ...]) -> None:
    """Refuse a config whose scale count is not this network's head count."""
    if len(strides) != n_heads:
        raise ValueError(
            f"{model} predicts at {n_heads} scales but its config describes "
            f"{len(strides)} (strides={strides}); pass one stride and one "
            "anchor mask per head"
        )


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


# Pad fill for the stride-1 max-pool.  Must lose every max against a real
# activation; leaky-ReLU outputs go negative, so 0 will not do.
_NEG_FILL: float = -1e30

# YOLOv3-Tiny is clustered separately from the full model — six priors, not
# nine, and not a subset of them (yolov3-tiny.cfg:
# ``anchors = 10,14, 23,27, 37,58, 81,82, 135,169, 344,319``).  It predicts
# at strides 16 and 32 only; the stride-16 head takes mask 0,1,2 and the
# stride-32 head mask 3,4,5.
_TINY_ANCHORS: tuple[tuple[float, float], ...] = (
    (10.0, 14.0),
    (23.0, 27.0),
    (37.0, 58.0),
    (81.0, 82.0),
    (135.0, 169.0),
    (344.0, 319.0),
)
_TINY_STRIDES: tuple[int, ...] = (16, 32)
_TINY_ANCHOR_MASKS: tuple[tuple[int, ...], ...] = ((0, 1, 2), (3, 4, 5))


class _ConvBnLeaky(nn.Module):
    """Conv2d(bias=False) → BatchNorm2d → LeakyReLU(0.1)."""

    def __init__(
        self, in_ch: int, out_ch: int, kernel: int, stride: int = 1, padding: int = -1
    ) -> None:
        super().__init__()
        pad = padding if padding >= 0 else (kernel - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return F.leaky_relu(self.bn(self.conv(x)), negative_slope=0.1)


@final
class _Dark53Block(nn.Module):
    """Darknet-53 residual unit: Conv(1×1) → Conv(3×3) + skip."""

    def __init__(self, in_ch: int) -> None:
        super().__init__()
        mid_ch = in_ch // 2
        self.conv1 = _ConvBnLeaky(in_ch, mid_ch, 1)
        self.conv2 = _ConvBnLeaky(mid_ch, in_ch, 3)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return x + self.conv2(self.conv1(x))


def _make_dark53_stage(
    in_ch: int, out_ch: int, n_blocks: int
) -> tuple[nn.Sequential, _ConvBnLeaky]:
    """Build one Darknet-53 stage.

    Returns:
        (blocks, downsample_conv) — residual blocks and the stride-2 conv that
        follows them to double channels and halve spatial resolution.
    """
    blocks: list[nn.Module] = [_Dark53Block(in_ch) for _ in range(n_blocks)]
    stage = nn.Sequential(*blocks)
    down = _ConvBnLeaky(in_ch, out_ch, 3, stride=2)
    return stage, down


# ---------------------------------------------------------------------------
# Darknet-53 backbone
# ---------------------------------------------------------------------------


@final
class _Darknet53(nn.Module):
    """Darknet-53 backbone.

    Returns three feature maps for multi-scale detection:
      P3_raw : (B, 256, H/8,  W/8)
      P4_raw : (B, 512, H/16, W/16)
      P5     : (B, 1024, H/32, W/32)
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        # Stem
        self.conv0 = _ConvBnLeaky(in_channels, 32, 3)
        self.conv1 = _ConvBnLeaky(32, 64, 3, stride=2)

        # Stage 1: 1×block(64), then stride-2 → 128ch
        self.stage1_blocks, self.down1 = _make_dark53_stage(64, 128, 1)

        # Stage 2: 2×block(128), then stride-2 → 256ch  [P3_raw]
        self.stage2_blocks, self.down2 = _make_dark53_stage(128, 256, 2)

        # Stage 3: 8×block(256), then stride-2 → 512ch  [P4_raw]
        self.stage3_blocks, self.down3 = _make_dark53_stage(256, 512, 8)

        # Stage 4: 8×block(512), then stride-2 → 1024ch
        self.stage4_blocks, self.down4 = _make_dark53_stage(512, 1024, 8)

        # Stage 5: 4×block(1024) [P5 final]
        self.stage5_blocks = nn.Sequential(*[_Dark53Block(1024) for _ in range(4)])

    @override
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x = self.conv1(self.conv0(x))

        x = self.stage1_blocks(x)
        x = self.down1(x)

        # The detection taps come *after* their residual stack, not before the
        # preceding downsample: Darknet-53 predicts from the 8×256 stage
        # (stride 8) and the 8×512 stage (stride 16).  Tapping a stage early
        # yields stride-4/8 maps that the decoder then scales by 8/16, doubling
        # every predicted centre.
        x = self.stage2_blocks(x)
        x = self.down2(x)

        x = self.stage3_blocks(x)
        p3_raw = x  # (B, 256, H/8, W/8)
        x = self.down3(x)

        x = self.stage4_blocks(x)
        p4_raw = x  # (B, 512, H/16, W/16)
        x = self.down4(x)

        p5 = self.stage5_blocks(x)  # (B, 1024, H/32, W/32)

        return p3_raw, p4_raw, p5


# ---------------------------------------------------------------------------
# Tiny backbone (YOLOv3-Tiny)
# ---------------------------------------------------------------------------


@final
class _Darknet53Tiny(nn.Module):
    """Lightweight Darknet backbone used by YOLOv3-Tiny.

    6 conv stages with MaxPool (no residuals), returns two feature maps:
      P4_raw : (B, 256, H/16, W/16)
      P5     : (B, 1024, H/32, W/32)

    Only 2 detection scales are used in the Tiny variant.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv1 = _ConvBnLeaky(in_channels, 16, 3)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = _ConvBnLeaky(16, 32, 3)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = _ConvBnLeaky(32, 64, 3)
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.conv4 = _ConvBnLeaky(64, 128, 3)
        self.pool4 = nn.MaxPool2d(2, stride=2)
        self.conv5 = _ConvBnLeaky(128, 256, 3)  # P4_raw
        self.pool5 = nn.MaxPool2d(2, stride=2)
        self.conv6 = _ConvBnLeaky(256, 512, 3)
        # Stride-1 MaxPool that must *preserve* the grid.  darknet pads this
        # layer asymmetrically (0 left/top, 1 right/bottom), so 13x13 stays
        # 13x13; symmetric padding=1 gives 14x14 and desynchronises the P5
        # head from its stride-32 decode grid.  The pad value has to be very
        # negative rather than 0 — these activations come from a leaky ReLU
        # and are frequently negative, so a 0 pad would win the max.
        self.pool6 = nn.MaxPool2d(2, stride=1, padding=0)
        self.conv7 = _ConvBnLeaky(512, 1024, 3)
        self.conv8 = _ConvBnLeaky(1024, 256, 1)  # bottleneck before P5 head

    @override
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.pool4(self.conv4(x))
        x = self.conv5(x)
        p4_raw = x  # (B, 256, H/16, W/16)
        x = self.pool5(x)
        x = self.conv6(x)
        x = F.pad(x, (0, 1, 0, 1), mode="constant", value=_NEG_FILL)
        x = self.pool6(x)
        x = self.conv7(x)
        p5 = self.conv8(x)  # (B, 256, H/32, W/32)
        return p4_raw, p5


# ---------------------------------------------------------------------------
# Detection head
# ---------------------------------------------------------------------------


def _make_detection_head(
    in_ch: int, mid_ch: int, num_anchors: int, num_classes: int
) -> tuple[nn.Sequential, nn.Sequential]:
    """Build a YOLOv3 detection head.

    5 alternating 1×1/3×3 convolutions for feature compression, then a 3×3
    convolution back up to ``2 * mid_ch`` and a final 1×1 conv that outputs
    ``num_anchors*(5+num_classes)`` channels.

    The compress stack and the predictor are returned separately because
    ``yolov3.cfg`` routes the *compressed* feature — not the expanded one —
    into the top-down upsample path (``route -4`` back to the last 1×1).
    Predictions come off the wider 3×3, so the two consumers need different
    tensors.

    Returns:
        (compress_convs, predict_convs)
    """
    c = [
        _ConvBnLeaky(in_ch, mid_ch, 1),
        _ConvBnLeaky(mid_ch, mid_ch * 2, 3),
        _ConvBnLeaky(mid_ch * 2, mid_ch, 1),
        _ConvBnLeaky(mid_ch, mid_ch * 2, 3),
        _ConvBnLeaky(mid_ch * 2, mid_ch, 1),
    ]
    compress = nn.Sequential(*c)
    predict = nn.Sequential(
        _ConvBnLeaky(mid_ch, mid_ch * 2, 3),
        nn.Conv2d(mid_ch * 2, num_anchors * (5 + num_classes), 1, bias=True),
    )
    return compress, predict


def _make_tiny_detection_head(
    in_ch: int, mid_ch: int, num_anchors: int, num_classes: int
) -> tuple[nn.Sequential, nn.Conv2d]:
    """Build a YOLOv3-*Tiny* detection head.

    ``yolov3-tiny.cfg`` uses a two-convolution head, not the full model's
    alternating five: a single 3x3 expansion followed by the 1x1 predictor.
    Borrowing the 5-conv head roughly triples the head's parameters and
    changes the receptive field the predictions are made from, which is the
    opposite of what the "tiny" variant exists for.

    Returns:
        (compress_convs, predict_conv)
    """
    compress = nn.Sequential(_ConvBnLeaky(in_ch, mid_ch, 3))
    predict = nn.Conv2d(mid_ch, num_anchors * (5 + num_classes), 1, bias=True)
    return compress, predict


# ---------------------------------------------------------------------------
# Box decoding helper
# ---------------------------------------------------------------------------


def _decode_predictions(
    raw: Tensor, anchors_wh: list[tuple[float, float]], stride: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Decode a single-scale raw detection tensor.

    Args:
        raw:         (B, num_anchors*(5+C), H, W) raw network output.
        anchors_wh:  3 anchor (w, h) pairs in pixels for this scale.
        stride:      Stride of this feature map.

    Returns:
        (logits, pred_boxes, conf):
          logits    : (B, H*W*3, C)   raw (pre-sigmoid) class logits
          pred_boxes: (B, H*W*3, 4)   decoded xyxy pixel boxes
          conf      : (B, H*W*3)      objectness confidence (sigmoid)
    """
    B = int(raw.shape[0])
    fH = int(raw.shape[2])
    fW = int(raw.shape[3])
    nA = len(anchors_wh)
    C = int(raw.shape[1]) // nA - 5

    # Reshape → (B, nA, 5+C, H, W) → (B, nA, H, W, 5+C)
    raw = raw.reshape(B, nA, 5 + C, fH, fW)
    raw = raw.permute(0, 1, 3, 4, 2)  # (B, nA, fH, fW, 5+C)

    tx = raw[..., 0]  # (B, nA, fH, fW)
    ty = raw[..., 1]
    tw = raw[..., 2]
    th = raw[..., 3]
    tc = raw[..., 4]  # objectness logit
    cls_logits = raw[..., 5:]  # (B, nA, fH, fW, C)

    # Build cell offset grids as Python-level lists → tensor
    col_data: list[list[float]] = [[float(c) for c in range(fW)] for _ in range(fH)]
    row_data: list[list[float]] = [[float(r)] * fW for r in range(fH)]

    # The grids and priors take the head's device and dtype.
    col_t = lucid.tensor(col_data, dtype=raw.dtype, device=raw.device)  # (fH, fW)
    row_t = lucid.tensor(row_data, dtype=raw.dtype, device=raw.device)  # (fH, fW)

    # Decode box centres
    px = (F.sigmoid(tx) + col_t) * float(stride)  # (B, nA, fH, fW)
    py = (F.sigmoid(ty) + row_t) * float(stride)

    # Build anchor arrays as Python lists → tensors
    aw_data: list[list[list[list[float]]]] = []
    ah_data: list[list[list[list[float]]]] = []
    for _ in range(B):
        b_aw: list[list[list[float]]] = []
        b_ah: list[list[list[float]]] = []
        for a_idx in range(nA):
            aw_val = anchors_wh[a_idx][0]
            ah_val = anchors_wh[a_idx][1]
            row_aw: list[list[float]] = [[aw_val] * fW for _ in range(fH)]
            row_ah: list[list[float]] = [[ah_val] * fW for _ in range(fH)]
            b_aw.append(row_aw)
            b_ah.append(row_ah)
        aw_data.append(b_aw)
        ah_data.append(b_ah)

    aw_t = lucid.tensor(aw_data, dtype=raw.dtype, device=raw.device)  # (B, nA, fH, fW)
    ah_t = lucid.tensor(ah_data, dtype=raw.dtype, device=raw.device)

    pw = lucid.exp(tw) * aw_t  # (B, nA, fH, fW)
    ph = lucid.exp(th) * ah_t

    # xyxy boxes in pixel space
    x1 = px - pw / 2.0
    y1 = py - ph / 2.0
    x2 = px + pw / 2.0
    y2 = py + ph / 2.0

    boxes = lucid.stack([x1, y1, x2, y2], dim=-1)  # (B, nA, fH, fW, 4)

    N = nA * fH * fW
    boxes = boxes.reshape(B, N, 4)
    cls_logits = cls_logits.reshape(B, N, C)
    conf = F.sigmoid(tc.reshape(B, N))

    return cls_logits, boxes, conf


# ---------------------------------------------------------------------------
# Loss helper
# ---------------------------------------------------------------------------


# Paper §2.1 uses 0.5; darknet's yolov3.cfg spells it ``ignore_thresh = .7``
# for the [yolo] layers.  The paper value is the one the text commits to.
_IGNORE_IOU_THRESH = 0.5


def _ignore_mask(pred_boxes: Tensor, gt_boxes: Tensor, thresh: float) -> Tensor:
    """1.0 where a predicted box overlaps some GT by more than ``thresh``.

    ``pred_boxes`` are one image's ``(N, 4)`` decoded *predicted* boxes, as
    the paper specifies, decoded without gradient — the mask is a selection,
    not a target.  ``gt_boxes`` are ``(M, 4)`` xyxy pixels.  One pairwise
    IoU on the device replaces a per-anchor scan that read four values back
    to the host for each of the ~10k anchors at 416 px.
    """
    n_pred = int(pred_boxes.shape[0])
    if int(gt_boxes.shape[0]) == 0:
        return pred_boxes.new_zeros(n_pred)
    gt = gt_boxes.to(device=pred_boxes.device, dtype=pred_boxes.dtype)
    return (box_iou(pred_boxes, gt).max(dim=1).values > thresh).to(pred_boxes.dtype)


def _gt_rows(target: dict[str, Tensor]) -> list[tuple[tuple[float, float, float, float], int]]:
    """One image's GT ``(x1, y1, x2, y2)`` boxes and labels, read in one go.

    Target assignment indexes Python-side arrays, so the values are needed
    on the host; reading each coordinate with ``.item()`` cost five device
    round-trips per object.  Raises ``ValueError`` when the box and label
    counts disagree, rather than pairing them up short.
    """
    flat = target["boxes"].float().reshape(-1).tolist()
    ids = target["labels"].long().reshape(-1).tolist()
    coords = [float(v) for v in flat if isinstance(v, float)] if isinstance(flat, list) else []
    labels = [int(v) for v in ids if isinstance(v, int)] if isinstance(ids, list) else []
    if len(coords) != 4 * len(labels):
        raise ValueError(
            f"target has {len(coords) / 4:g} boxes but {len(labels)} labels; "
            "each box needs exactly one label"
        )
    return [
        ((coords[4 * i], coords[4 * i + 1], coords[4 * i + 2], coords[4 * i + 3]), c)
        for i, c in enumerate(labels)
    ]


def _yolov3_loss(
    raw_preds: list[Tensor], targets: list[dict[str, Tensor]], config: YOLOV3Config
) -> Tensor:
    """Compute YOLOv3 multi-scale detection loss.

    Args:
        raw_preds: One raw tensor per scale, coarse to fine (P5, P4, P3 for
                   the full model), each (B, nA*(5+C), H_l, W_l).
        targets:   List of B target dicts (boxes xyxy pixels, labels ints).
        config:    Model configuration — its anchors, masks and strides are
                   the table the heads were built and decoded from.

    Returns:
        Scalar loss tensor.
    """
    B = int(raw_preds[0].shape[0])
    n_pos = 0.0
    n_cells = 0.0
    C = config.num_classes
    anchors_all = config.anchors
    scales = _scale_priors(config.anchors, config.anchor_masks, config.strides)

    localisation: list[Tensor] = []
    objectness: list[Tensor] = []

    # The model refused a config whose scale count is not its head count, so
    # the two lists are the same length by construction.
    for raw, (anchor_idx_triple, anchors_wh, stride) in zip(raw_preds, scales, strict=True):
        fH = int(raw.shape[2])
        fW = int(raw.shape[3])
        nA = len(anchor_idx_triple)

        # Reshape raw → (B, nA, 5+C, fH, fW) → (B, nA, fH, fW, 5+C)
        raw_r = raw.reshape(B, nA, 5 + C, fH, fW).permute(0, 1, 3, 4, 2)

        # The ignore mask compares *decoded* predictions with the GTs; it is
        # a selection, so the boxes carry no gradient.
        _, pred_boxes, _ = _decode_predictions(raw.detach(), anchors_wh, stride)

        for b in range(B):
            gt_boxes = targets[b]["boxes"]  # (M, 4) xyxy pixels
            M = int(gt_boxes.shape[0])
            positives: set[tuple[int, int, int]] = set()

            # Target tensors for this image and scale, in the head's dtype
            tgt_tx = raw.new_zeros(nA, fH, fW)
            tgt_ty = raw.new_zeros(nA, fH, fW)
            tgt_tw = raw.new_zeros(nA, fH, fW)
            tgt_th = raw.new_zeros(nA, fH, fW)
            tgt_obj = raw.new_zeros(nA, fH, fW)
            tgt_cls = raw.new_zeros(nA, fH, fW, C)
            obj_mask = raw.new_zeros(nA, fH, fW)  # 1 for positives
            coord_scale = raw.new_ones(nA, fH, fW)

            if M > 0:
                # Rebuild entire target arrays from scratch using Python lists.
                # This avoids any in-place mutation on Lucid tensors.
                tx_arr: list[list[list[float]]] = [
                    [[0.0] * fW for _ in range(fH)] for _ in range(nA)
                ]
                ty_arr: list[list[list[float]]] = [
                    [[0.0] * fW for _ in range(fH)] for _ in range(nA)
                ]
                tw_arr: list[list[list[float]]] = [
                    [[0.0] * fW for _ in range(fH)] for _ in range(nA)
                ]
                th_arr: list[list[list[float]]] = [
                    [[0.0] * fW for _ in range(fH)] for _ in range(nA)
                ]
                obj_arr: list[list[list[float]]] = [
                    [[0.0] * fW for _ in range(fH)] for _ in range(nA)
                ]
                cls_arr: list[list[list[list[float]]]] = [
                    [[[0.0] * C for _ in range(fW)] for _ in range(fH)] for _ in range(nA)
                ]
                mask_arr: list[list[list[float]]] = [
                    [[0.0] * fW for _ in range(fH)] for _ in range(nA)
                ]
                coord_arr: list[list[list[float]]] = [
                    [[1.0] * fW for _ in range(fH)] for _ in range(nA)
                ]

                for (x1g, y1g, x2g, y2g), cls_id in _gt_rows(targets[b]):
                    wg = x2g - x1g
                    hg = y2g - y1g
                    cxg = (x1g + x2g) / 2.0
                    cyg = (y1g + y2g) / 2.0

                    col_idx = max(0, min(int(cxg / stride), fW - 1))
                    row_idx = max(0, min(int(cyg / stride), fH - 1))

                    # Paper §2.1: "our system only assigns one bounding box
                    # prior for each ground truth object".  The argmax runs
                    # over ALL priors (nine; six for Tiny), so a GT becomes a
                    # positive only at the one scale that owns its best prior —
                    # picking a best anchor per scale instead would force a
                    # large object to be regressed by a 10x13 prior on the
                    # stride-8 map, whose log-space target is pure noise.
                    best_iou = -1.0
                    best_global = 0
                    for a_i, (aw, ah) in enumerate(anchors_all):
                        inter_w = min(wg, aw)
                        inter_h = min(hg, ah)
                        inter = inter_w * inter_h
                        union = wg * hg + aw * ah - inter
                        iou_val = inter / (union + 1e-6)
                        if iou_val > best_iou:
                            best_iou = iou_val
                            best_global = a_i

                    if best_global not in anchor_idx_triple:
                        continue
                    best_a = anchor_idx_triple.index(best_global)
                    aw_best, ah_best = anchors_wh[best_a]
                    tx_arr[best_a][row_idx][col_idx] = cxg / stride - float(col_idx)
                    ty_arr[best_a][row_idx][col_idx] = cyg / stride - float(row_idx)
                    # t_w = log(w_gt / p_w) exactly inverts the decode.  Flooring
                    # the GT side at 1 px shifted the target for every box smaller
                    # than a pixel at this stride, and adding the epsilon to the
                    # *ratio* rather than guarding the argument perturbed all of
                    # them.  Guard the argument instead.
                    tw_arr[best_a][row_idx][col_idx] = math.log(max(wg / aw_best, 1e-9))
                    th_arr[best_a][row_idx][col_idx] = math.log(max(hg / ah_best, 1e-9))
                    obj_arr[best_a][row_idx][col_idx] = 1.0
                    mask_arr[best_a][row_idx][col_idx] = 1.0
                    positives.add((best_a, row_idx, col_idx))
                    # darknet scales the coordinate delta by
                    # ``2 - w_gt*h_gt/(W*H)`` so a small box is not
                    # out-weighted by a large one sharing the loss.
                    coord_arr[best_a][row_idx][col_idx] = 2.0 - (wg * hg) / max(
                        float(fW * stride) * float(fH * stride), 1e-6
                    )
                    if 0 <= cls_id < C:
                        cls_arr[best_a][row_idx][col_idx][cls_id] = 1.0

                dt, dev = raw.dtype, raw.device
                tgt_tx = lucid.tensor(tx_arr, dtype=dt, device=dev)
                tgt_ty = lucid.tensor(ty_arr, dtype=dt, device=dev)
                tgt_tw = lucid.tensor(tw_arr, dtype=dt, device=dev)
                tgt_th = lucid.tensor(th_arr, dtype=dt, device=dev)
                tgt_obj = lucid.tensor(obj_arr, dtype=dt, device=dev)
                tgt_cls = lucid.tensor(cls_arr, dtype=dt, device=dev)
                obj_mask = lucid.tensor(mask_arr, dtype=dt, device=dev)
                coord_scale = lucid.tensor(coord_arr, dtype=dt, device=dev)

            # Predicted values for this image
            pred_b = raw_r[b]  # (nA, fH, fW, 5+C)
            pred_tx = pred_b[..., 0]
            pred_ty = pred_b[..., 1]
            pred_tw = pred_b[..., 2]
            pred_th = pred_b[..., 3]
            pred_tc = pred_b[..., 4]
            pred_cls_logits = pred_b[..., 5:]  # (nA, fH, fW, C)

            # Box MSE loss (positive anchors only)
            _cw = obj_mask * coord_scale
            box_loss_sx = ((F.sigmoid(pred_tx) - tgt_tx) ** 2) * _cw
            box_loss_sy = ((F.sigmoid(pred_ty) - tgt_ty) ** 2) * _cw
            box_loss_w = ((pred_tw - tgt_tw) ** 2) * _cw
            box_loss_h = ((pred_th - tgt_th) ** 2) * _cw
            box_loss = (box_loss_sx + box_loss_sy + box_loss_w + box_loss_h).sum()

            # Objectness BCE
            obj_bce = F.binary_cross_entropy_with_logits(pred_tc, tgt_obj, reduction="none")
            # Paper §2.1: "If the bounding box prior is not the best but does
            # overlap a ground truth object by more than some threshold we
            # ignore the prediction ... We use the threshold of .5."  Without
            # this, a near-perfect box that merely lost the argmax is punished
            # as background, and with ~10k anchors that term dominates.
            ignore = _ignore_mask(pred_boxes[b], gt_boxes, _IGNORE_IOU_THRESH).reshape(nA, fH, fW)
            noobj_mask = (1.0 - obj_mask) * (1.0 - ignore)
            # v3 drops v1's lambda weighting — "we use sum of squared error
            # loss" — and lets the ignore mask do the negative balancing.
            obj_loss = (obj_bce * obj_mask + obj_bce * noobj_mask).sum()

            # Class BCE (positive anchors only)
            cls_bce = F.binary_cross_entropy_with_logits(pred_cls_logits, tgt_cls, reduction="none")
            cls_loss = (cls_bce * obj_mask[..., None]).sum()

            # The two families of term scale with different things, so they
            # need different denominators.  Box and class regress only at
            # positive anchors, and objectness is evaluated at *every* cell —
            # summing both raw made the loss grow with resolution, batch size
            # and object count at once, so one learning rate could not serve a
            # 416 crop and a 608 one.
            localisation.append((box_loss + cls_loss).reshape(1))
            objectness.append(obj_loss.reshape(1))
            n_pos += float(len(positives))
            n_cells += float(B * nA * fH * fW)

    if not localisation:
        return raw_preds[0].new_zeros(1)
    loc = lucid.cat(localisation).sum() / max(1.0, n_pos)
    obj = lucid.cat(objectness).sum() / max(1.0, n_cells)
    return loc + obj


def _yolov3_output(
    raw_preds: list[Tensor], targets: list[dict[str, Tensor]] | None, config: YOLOV3Config
) -> ObjectDetectionOutput:
    """Decode every scale's raw head output, and score it when given targets.

    Shared by YOLOv3 and YOLOv3-Tiny: the two differ only in their heads and
    in the per-scale table their config carries, which this reads for both
    the decode and the loss.
    """
    all_logits: list[Tensor] = []
    all_boxes: list[Tensor] = []
    all_conf: list[Tensor] = []
    scales = _scale_priors(config.anchors, config.anchor_masks, config.strides)
    # One raw tensor per head; the model checked its head count against the
    # config's scale count when it was built.
    for raw_pred, (_, anch_wh, stride) in zip(raw_preds, scales, strict=True):
        logits_s, boxes_s, conf_s = _decode_predictions(raw_pred, anch_wh, stride)
        all_logits.append(logits_s)
        all_boxes.append(boxes_s)
        all_conf.append(conf_s)

    loss: Tensor | None = None
    if targets is not None:
        loss = _yolov3_loss(raw_preds, targets, config)

    return ObjectDetectionOutput(
        logits=lucid.cat(all_logits, dim=1),  # (B, total_anchors, C)
        pred_boxes=lucid.cat(all_boxes, dim=1),  # (B, total_anchors, 4)
        loss=loss,
        objectness=lucid.cat(all_conf, dim=1),  # (B, total_anchors)
    )


# ---------------------------------------------------------------------------
# Postprocess helper (shared by YOLOv3, YOLOv3-Tiny and YOLOv4)
# ---------------------------------------------------------------------------


def _postprocess_detections(
    output: ObjectDetectionOutput,
    image_sizes: list[tuple[int, int]],
    score_thresh: float,
    nms_thresh: float,
) -> list[dict[str, Tensor]]:
    """Score, threshold, clip and per-class NMS for sigmoid-class YOLO heads.

    The score is Pr(object)·Pr(class|object), as darknet's
    ``get_yolo_detections`` computes it.  Every tensor built here —
    selection indices, scores, class ids, the empty result — is placed on
    the predictions' device; built on the CPU they made ``batched_nms``
    raise on a Metal model.
    """
    B = int(output.logits.shape[0])
    dev = output.logits.device
    results: list[dict[str, Tensor]] = []

    for b in range(B):
        cls_logits = output.logits[b]  # (total_anchors, C)
        boxes = output.pred_boxes[b]  # (total_anchors, 4)
        iH, iW = image_sizes[b]

        # YOLO scores a detection by Pr(object)·Pr(class|object); dropping
        # the objectness factor emits every background anchor whose class
        # sigmoid happens to clear the threshold, and ranks NMS wrongly.
        cls_probs = F.sigmoid(cls_logits)  # (total_anchors, C)
        if output.objectness is not None:
            cls_probs = cls_probs * output.objectness[b][:, None]
        C = int(cls_probs.shape[1])

        # One .tolist() instead of N_anc * C device syncs.  At 416px
        # with 80 classes that scan was 10,647 * 80 = 851,760 scalar
        # round-trips per image; darknet and every reference thresholds
        # the (N, C) score matrix in one go.
        flat = cast(list[float], cls_probs.reshape(-1).tolist())
        anchor_ids: list[int] = []
        scores: list[float] = []
        labels: list[int] = []
        for idx, sc in enumerate(flat):
            if sc >= score_thresh:
                a, c = divmod(idx, C)
                anchor_ids.append(a)
                scores.append(sc)
                labels.append(c)

        if not anchor_ids:
            results.append(empty_detection_result(dev))
            continue

        # One gather rather than a (1, 4) slice and two scalar tensors per
        # detection, which at a low threshold meant thousands of tiny device
        # allocations followed by a concatenation.
        det_boxes = boxes[lucid.tensor(anchor_ids, device=dev).long()]  # (K, 4)
        det_scores = lucid.tensor(scores, device=dev)
        det_labels = lucid.tensor(labels, dtype=lucid.int64, device=dev)

        det_boxes = clip_boxes_to_image(det_boxes, (iH, iW))

        keep_idx = batched_nms(det_boxes, det_scores, det_labels, nms_thresh)
        if int(keep_idx.shape[0]) == 0:
            results.append(empty_detection_result(dev))
            continue

        results.append(
            detection_result(det_boxes[keep_idx], det_scores[keep_idx], det_labels[keep_idx])
        )

    return results


# ---------------------------------------------------------------------------
# YOLOv3 model
# ---------------------------------------------------------------------------


class YOLOV3ForObjectDetection(ObjectDetectionModel):
    r"""YOLOv3 multi-scale object detector (Redmon & Farhadi, 2018).

    Adds a 53-layer **Darknet-53** backbone (residual connections, more
    capacity than Darknet-19) and an FPN-style multi-scale detection
    head that predicts on three feature levels — :math:`P_3` (stride 8),
    :math:`P_4` (stride 16), :math:`P_5` (stride 32) — with three
    anchors per cell per level (9 anchors total).  This multi-scale
    detection dramatically improves small-object recall over YOLOv2,
    bringing COCO AP up to 33.0% (paper Table 3).  Per-class predictions
    use independent sigmoids rather than softmax — supporting
    multi-label classification.

    Parameters
    ----------
    config : YOLOV3Config
        Frozen architecture spec.  Use :func:`yolo_v3` for the full
        Darknet-53 model or :func:`yolo_v3_tiny` for the lightweight
        2-scale variant.

    Attributes
    ----------
    config : YOLOV3Config
        Stored copy of the config that built this model.
    backbone : _Darknet53
        Residual backbone producing P3 / P4 / P5 features at strides
        8 / 16 / 32 (128 / 256 / 1024 channels).
    p5_compress, p5_predict : nn.Sequential
        Stride-32 detection head: five-conv compress, then a 3x3 + 1x1
        prediction pair producing :math:`3(5 + C)` channels.
    p4_compress, p4_predict, p4_to_p3_conv : nn.Sequential
        Stride-16 head with FPN-style top-down upsample + concatenation
        with the P4 backbone feature.
    p3_compress, p3_predict, p5_to_p4_conv : nn.Sequential
        Stride-8 head with another top-down upsample + concatenation
        with the P3 backbone feature.

    Notes
    -----
    See Redmon & Farhadi, "YOLOv3: An Incremental Improvement", 2018
    (arXiv:1804.02767).  Per-anchor box decoding reuses the YOLOv2
    direct-location parameterisation; the multi-class objective swaps
    YOLOv2's softmax for per-class binary cross-entropy:

    .. math::

        \mathcal{L}_\mathrm{cls} = -\sum_{c = 1}^{C}
            \bigl[\,y_c \log \sigma(s_c) + (1 - y_c) \log(1 - \sigma(s_c))\bigr],

    enabling multi-label predictions where a single object can belong
    to multiple categories (e.g. "Woman" + "Person").

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.yolo._v3 import yolo_v3
    >>> model = yolo_v3()
    >>> x = lucid.randn(1, 3, 416, 416)
    >>> out = model(x)
    >>> out.logits.shape[0]
    1
    """

    config_class: ClassVar[type[YOLOV3Config]] = YOLOV3Config
    base_model_prefix: ClassVar[str] = "yolo_v3"

    def __init__(self, config: YOLOV3Config) -> None:
        super().__init__(config)
        _require_scales("YOLOv3", 3, config.strides)
        self._cfg = config
        C = config.num_classes
        # Anchors per head, P3/P4/P5 — three each in the paper's table.
        nA3, nA4, nA5 = (len(m) for m in config.anchor_masks)

        # Backbone
        self.backbone = _Darknet53(config.in_channels)

        # Backbone output channels (Darknet-53, paper Table 1):
        #   p3_raw : 256ch  (stride-8  — after the 8× 256 residual stage)
        #   p4_raw : 512ch  (stride-16 — after the 8× 512 residual stage)
        #   p5     : 1024ch (stride-32 — after the 4× 1024 residual stage)

        # P5 head (1024ch input): compress mid_ch=512
        self.p5_compress, self.p5_predict = _make_detection_head(1024, 512, nA5, C)
        # p5_compress output: 512ch

        # Upsample + concat: p5_up (256ch) + p4_raw (512ch) = 768ch, matching
        # darknet's route layer that joins the upsample with layer 61.
        self.p5_to_p4_conv = _ConvBnLeaky(512, 256, 1)  # 512→256
        self.p4_compress, self.p4_predict = _make_detection_head(256 + 512, 256, nA4, C)
        # p4_compress output: 256ch

        # Upsample + concat: p4_up (128ch) + p3_raw (256ch) = 384ch (route with
        # layer 36).
        self.p4_to_p3_conv = _ConvBnLeaky(256, 128, 1)  # 256→128
        self.p3_compress, self.p3_predict = _make_detection_head(128 + 256, 128, nA3, C)

    @override
    def forward(
        self, x: Tensor, targets: list[dict[str, Tensor]] | None = None
    ) -> ObjectDetectionOutput:
        """Run YOLOv3.

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
        p3_raw, p4_raw, p5 = self.backbone(x)

        # P5 branch
        p5_feat = self.p5_compress(p5)  # (B, 512, H/32, W/32)
        p5_raw = self.p5_predict(p5_feat)  # (B, nA*(5+C), H/32, W/32)

        # P4 branch: upsample P5 → concat with P4_raw
        p5_up = self.p5_to_p4_conv(p5_feat)  # (B, 256, H/32, W/32)
        fH4 = int(p4_raw.shape[2])
        fW4 = int(p4_raw.shape[3])
        p5_up = F.interpolate(p5_up, size=(fH4, fW4), mode="nearest")
        p4_cat = lucid.cat([p5_up, p4_raw], dim=1)  # (B, 256+512, H/16, W/16)
        p4_feat = self.p4_compress(p4_cat)  # (B, 256, H/16, W/16)
        p4_raw_pred = self.p4_predict(p4_feat)  # (B, nA*(5+C), H/16, W/16)

        # P3 branch: upsample P4 → concat with P3_raw
        p4_up = self.p4_to_p3_conv(p4_feat)  # (B, 128, H/16, W/16)
        fH3 = int(p3_raw.shape[2])
        fW3 = int(p3_raw.shape[3])
        p4_up = F.interpolate(p4_up, size=(fH3, fW3), mode="nearest")
        p3_cat = lucid.cat([p4_up, p3_raw], dim=1)  # (B, 128+256, H/8, W/8)
        p3_feat = self.p3_compress(p3_cat)  # (B, 128, H/8, W/8)
        p3_raw_pred = self.p3_predict(p3_feat)  # (B, nA*(5+C), H/8, W/8)

        # raw_preds order: P5, P4, P3 (from coarse to fine)
        raw_preds = [p5_raw, p4_raw_pred, p3_raw_pred]
        return _yolov3_output(raw_preds, targets, cfg)

    def postprocess(
        self, output: ObjectDetectionOutput, image_sizes: list[tuple[int, int]]
    ) -> list[dict[str, Tensor]]:
        """Filter by score, clip boxes, apply per-class NMS.

        Args:
            output:      Forward pass output.
            image_sizes: List of (H, W) per image.

        Returns:
            Per-image list of dicts with "boxes", "scores", "labels", on
            the model's device.
        """
        return _postprocess_detections(
            output, image_sizes, self._cfg.score_thresh, self._cfg.nms_thresh
        )


# ---------------------------------------------------------------------------
# YOLOv3-Tiny model
# ---------------------------------------------------------------------------


class YOLOV3TinyForObjectDetection(ObjectDetectionModel):
    r"""YOLOv3-Tiny — the 2-scale lightweight YOLOv3 (Redmon & Farhadi, 2018).

    A shallow Darknet backbone (six conv + max-pool stages, no residual
    connections) feeding two detection scales — stride 32 and stride 16 —
    each through ``yolov3-tiny.cfg``'s two-convolution head rather than the
    full model's five.  Boxes, objectness and per-class sigmoids are decoded
    exactly as in :class:`YOLOV3ForObjectDetection`, so the two share their
    loss and :meth:`postprocess`.

    Parameters
    ----------
    config : YOLOV3Config
        Frozen architecture spec describing exactly two scales.
        :func:`yolo_v3_tiny` supplies ``yolov3-tiny.cfg``'s table: six
        anchors, strides ``(16, 32)``, masks ``((0, 1, 2), (3, 4, 5))``.
        The default :class:`YOLOV3Config` describes the full model's three
        scales and is refused with a ``ValueError``.

    Attributes
    ----------
    config : YOLOV3Config
        Stored copy of the config that built this model.
    backbone : _Darknet53Tiny
        Plain conv/max-pool backbone producing the stride-16 (256ch) and
        stride-32 (256ch bottleneck) features.
    p5_compress : nn.Sequential
        Stride-32 ``conv 512 3x3`` expansion.
    p5_predict : nn.Conv2d
        Stride-32 1x1 predictor emitting :math:`3(5 + C)` channels.
    p5_to_p4_conv : _ConvBnLeaky
        ``conv 128 1x1`` on the stride-32 bottleneck before upsampling.
    p4_compress : nn.Sequential
        Stride-16 ``conv 256 3x3`` over the upsampled-and-concatenated
        384 channels.
    p4_predict : nn.Conv2d
        Stride-16 1x1 predictor emitting :math:`3(5 + C)` channels.

    Notes
    -----
    See Redmon & Farhadi, "YOLOv3: An Incremental Improvement", 2018
    (arXiv:1804.02767).  Tiny is the variant released alongside the full
    model for real-time and embedded inference.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.yolo import yolo_v3_tiny
    >>> model = yolo_v3_tiny().eval()
    >>> x = lucid.randn(1, 3, 416, 416)
    >>> out = model(x)
    >>> dets = model.postprocess(out, image_sizes=[(416, 416)])
    >>> sorted(dets[0])
    ['boxes', 'labels', 'scores']
    """

    config_class: ClassVar[type[YOLOV3Config]] = YOLOV3Config
    base_model_prefix: ClassVar[str] = "yolo_v3_tiny"

    def __init__(self, config: YOLOV3Config) -> None:
        super().__init__(config)
        # The full model's default table is three scales; this network has
        # two heads, and its decode and loss walk the config's table — a
        # mismatch is refused here rather than as a zip error mid-loss.
        _require_scales("YOLOv3-Tiny", 2, config.strides)
        self._cfg = config
        C = config.num_classes
        # Anchors per head, stride 16 then stride 32.
        nA4, nA5 = (len(m) for m in config.anchor_masks)

        self.backbone = _Darknet53Tiny(config.in_channels)

        # P5 head.  The backbone already ends with the cfg's ``conv 256 1x1``
        # bottleneck, so this is the remaining ``conv 512 3x3`` -> ``conv 255
        # 1x1`` pair.
        self.p5_compress, self.p5_predict = _make_tiny_detection_head(256, 512, nA5, C)

        # P4 branch: the cfg routes off the *256ch bottleneck*, not the 512ch
        # expansion — ``conv 128 1x1`` -> upsample -> route(concat layer 8,
        # 256ch) = 384ch -> ``conv 256 3x3`` -> ``conv 255 1x1``.
        self.p5_to_p4_conv = _ConvBnLeaky(256, 128, 1)
        self.p4_compress, self.p4_predict = _make_tiny_detection_head(256 + 128, 256, nA4, C)

    @override
    def forward(
        self, x: Tensor, targets: list[dict[str, Tensor]] | None = None
    ) -> ObjectDetectionOutput:
        check_per_image(type(self).__name__, int(x.shape[0]), targets=targets)
        p4_raw, p5 = self.backbone(x)

        p5_feat = self.p5_compress(p5)
        p5_raw_pred = self.p5_predict(p5_feat)

        # Route off the bottleneck ``p5``, not the expanded ``p5_feat``.
        p5_up = self.p5_to_p4_conv(p5)
        fH4 = int(p4_raw.shape[2])
        fW4 = int(p4_raw.shape[3])
        p5_up = F.interpolate(p5_up, size=(fH4, fW4), mode="nearest")
        p4_cat = lucid.cat([p5_up, p4_raw], dim=1)
        p4_feat = self.p4_compress(p4_cat)
        p4_raw_pred = self.p4_predict(p4_feat)

        # Coarse to fine, as the config's table is walked.
        return _yolov3_output([p5_raw_pred, p4_raw_pred], targets, self._cfg)

    def postprocess(
        self, output: ObjectDetectionOutput, image_sizes: list[tuple[int, int]]
    ) -> list[dict[str, Tensor]]:
        """Filter by score, clip boxes, apply per-class NMS.

        Args:
            output:      Forward pass output.
            image_sizes: List of (H, W) per image.

        Returns:
            Per-image list of dicts with "boxes", "scores", "labels", on
            the model's device.
        """
        return _postprocess_detections(
            output, image_sizes, self._cfg.score_thresh, self._cfg.nms_thresh
        )


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

_CFG_V3 = YOLOV3Config()
_CFG_V3_TINY = YOLOV3Config(
    anchors=_TINY_ANCHORS, strides=_TINY_STRIDES, anchor_masks=_TINY_ANCHOR_MASKS
)


@register_model(
    task="object-detection",
    family="yolo",
    model_type="yolo_v3",
    model_class=YOLOV3ForObjectDetection,
    default_config=_CFG_V3,
    params=61949149,
)
def yolo_v3(
    pretrained: bool | str = False, *, weights: YOLOV3Weights | None = None, **overrides: object
) -> YOLOV3ForObjectDetection:
    r"""YOLOv3 with Darknet-53 backbone, 3-scale detection (Redmon & Farhadi, 2018).

    Builds the paper-cited YOLOv3 detector: 53-layer residual Darknet
    backbone + FPN-style top-down head producing predictions at
    strides 8 / 16 / 32 with 3 anchors per cell per scale (9 anchors
    total, COCO k-means clustered).  Default 80 COCO classes; reaches
    COCO test-dev AP of 33.0% at 416x416 input (paper Table 3).

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True`` →
        the ``DEFAULT`` tag (:attr:`YOLOV3Weights.COCO_2014`, converted
        from darknet's ``yolov3.weights``); a tag string → that specific
        checkpoint.  Mutually exclusive with ``weights`` (which wins if
        both are given).
    weights : YOLOV3Weights, optional, keyword-only
        Explicit weights enum member, e.g. ``YOLOV3Weights.COCO_2014``.
        Takes precedence over ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`YOLOV3Config`.

    Returns
    -------
    YOLOV3ForObjectDetection
        Detector with the standard YOLOv3 configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See Redmon & Farhadi, "YOLOv3: An Incremental Improvement", 2018
    (arXiv:1804.02767).  The multi-scale head is the principal upgrade
    over YOLOv2 and the source of its small-object accuracy gain.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.yolo._v3 import yolo_v3
    >>> model = yolo_v3()
    >>> x = lucid.randn(1, 3, 416, 416)
    >>> out = model(x)
    >>> out.logits.shape[0]
    1
    """
    config = apply_overrides(_CFG_V3, overrides)
    entry = weights_mod.resolve_weights(YOLOV3Weights, pretrained, weights)
    model = YOLOV3ForObjectDetection(config)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="yolo_v3")
    return model


@register_model(
    task="object-detection",
    family="yolo",
    model_type="yolo_v3_tiny",
    model_class=YOLOV3TinyForObjectDetection,
    default_config=_CFG_V3_TINY,
    params=8852366,
)
def yolo_v3_tiny(
    pretrained: bool | str = False, *, weights: YOLOV3TinyWeights | None = None, **overrides: object
) -> YOLOV3TinyForObjectDetection:
    r"""YOLOv3-Tiny — 2-scale lightweight variant (Redmon & Farhadi, 2018).

    Builds the YOLOv3-Tiny detector explicitly defined in the paper's
    accompanying release: shallower Darknet-Tiny backbone with only two
    detection scales (strides 16 and 32) and 3 anchors per scale.
    Approximately 8.7M parameters; targets edge / real-time deployment.

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True`` →
        the ``DEFAULT`` tag (:attr:`YOLOV3TinyWeights.COCO_2014`,
        converted from darknet's ``yolov3-tiny.weights``); a tag string →
        that specific checkpoint.  Mutually exclusive with ``weights``
        (which wins if both are given).
    weights : YOLOV3TinyWeights, optional, keyword-only
        Explicit weights enum member, e.g.
        ``YOLOV3TinyWeights.COCO_2014``.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`YOLOV3Config`.

    Returns
    -------
    YOLOV3TinyForObjectDetection
        Detector with the YOLOv3-Tiny configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See Redmon & Farhadi, "YOLOv3: An Incremental Improvement", 2018
    (arXiv:1804.02767).  Drops one detection scale and uses a much
    smaller backbone — the standard "tiny" variant ships alongside the
    full release for embedded inference.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.yolo._v3 import yolo_v3_tiny
    >>> model = yolo_v3_tiny()
    >>> x = lucid.randn(1, 3, 416, 416)
    >>> out = model(x)
    >>> out.logits.shape[0]
    1
    """
    config = apply_overrides(_CFG_V3_TINY, overrides)
    entry = weights_mod.resolve_weights(YOLOV3TinyWeights, pretrained, weights)
    model = YOLOV3TinyForObjectDetection(config)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="yolo_v3_tiny")
    return model
