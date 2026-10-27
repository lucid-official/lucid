"""YOLOv2 / YOLO9000 — Redmon & Farhadi, CVPR 2017.

Paper: "YOLO9000: Better, Faster, Stronger"

Key innovations over YOLOv1
----------------------------
- **Anchor boxes**: replaces per-cell bounding box predictions with
  dimension-cluster anchors (5 per cell, default COCO clusters).
- **Darknet-19 backbone**: 19 conv layers with batch normalisation and
  no fully connected layers; stride-32 feature map.
- **Passthrough layer**: concatenates the 26×26 (stride-16) feature map
  with the 13×13 (stride-32) map via space-to-depth (2×) to preserve
  fine-grained spatial detail.
- **Direct location prediction**: tx, ty are passed through sigmoid so
  they stay in the cell [0,1]; tw, th remain raw (log-space anchor offsets).
- **Multi-scale training**: the network is fully convolutional, so it runs
  at any multiple of 32 without rebuilding.  ``MultiScaleResolution``
  supplies the paper's schedule — a fresh size drawn from
  {320, 352, …, 608} every 10 batches — for a training loop to apply.

Architecture (Darknet-19)
-------------------------
  3→32 (3×3) + pool
  32→64 (3×3) + pool
  64→128 (3×3), 128→64 (1×1), 64→128 (3×3) + pool
  128→256 (3×3), 256→128 (1×1), 128→256 (3×3) + pool
  256→512 (3×3), 512→256 (1×1), 256→512 (3×3),
                 512→256 (1×1), 256→512 (3×3)   ← passthrough hook (26×26, 512ch)
  + pool
  512→1024 (3×3), 1024→512 (1×1), 512→1024 (3×3),
                  1024→512 (1×1), 512→1024 (3×3)

Detection head (darknet yolov2.cfg layer stack)
-----------------------------------------------
  1024→1024 (3×3) × 2
  Route reduce: 512→64 (1×1, BN+leaky)
  Passthrough: space_to_depth(26×26×64, s=2) → 256ch; cat → 1280ch
  1280→1024 (3×3)
  1024 → num_anchors × (5 + num_classes) (1×1, no BN/act)

  NB: the paper's prose describes reorganising the full 512-channel route
  into 2048 channels and concatenating that (→3072) instead.  The cfg —
  which is what the released weights were trained with — inserts the
  64-channel reduction first, and that is what is implemented here.

Output: (B, A*(5+C), H, W) → (B, H*W*A, 5+C)
  Per anchor: [tx, ty, tw, th, conf, cls_0, …, cls_C-1]
  Decoded:    [sigmoid(tx)+c, sigmoid(ty)+r, pw·exp(tw), ph·exp(th)]
              conf = sigmoid(conf_raw), classes = softmax(cls_raw)
              (softmax at inference *and* in the class loss — they must
              agree, see ``_compute_loss``)

Losses (training)
-----------------
Requires ``targets`` — list of B dicts with:
  ``"boxes"``  : (M_i, 4) xyxy, normalised to [0, 1].
  ``"labels"`` : (M_i,)  integer foreground class ids (0-indexed).

  L = λ_coord · L_xy  +  λ_coord · L_wh  +  L_conf_obj
    +  λ_noobj · L_conf_noobj  +  L_cls

where the responsible anchor is the one with highest IoU vs GT.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, final, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import ModelConfig
from lucid.models._tasks import ObjectDetectionModel
from lucid.models._meta import model_family_meta
from lucid.models._output import ObjectDetectionOutput
from lucid.models._registry import register_model
from lucid.models._utils._detection import batched_nms, box_iou
from lucid.models._utils._detection_results import (
    check_per_image,
    detection_result,
    empty_detection_result,
)
from lucid.models.vision.yolo._v1 import _threshold_by_class
from lucid.models.vision.yolo._weights import YOLOV2Weights
from lucid.models._utils._common import apply_overrides
import lucid.weights as weights_mod

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# yolov2.cfg [region] thresh: an unmatched predictor whose best IoU against a
# ground-truth box exceeds this is excluded from the no-object loss, so a
# near-duplicate good box is not trained toward zero confidence.
_NOOBJ_IOU_THRESH: float = 0.6

# Dimension-cluster anchors, in units of 32-pixel cells.
#
# These are per-dataset: the k-means clustering runs on the training set's
# boxes, so VOC and COCO produce different priors.  The default head here is
# 80-class (COCO), so it takes the COCO clusters — the VOC set below was
# previously used under a "COCO" label, pairing 20-class priors with an
# 80-class head.
_COCO_ANCHORS: tuple[tuple[float, float], ...] = (
    (0.57273, 0.677385),
    (1.87446, 2.06253),
    (3.33843, 5.47434),
    (7.88282, 3.52778),
    (9.77052, 9.16828),
)

# The 20-class clustering, for a VOC-configured head.
_VOC_ANCHORS: tuple[tuple[float, float], ...] = (
    (1.3221, 1.73145),
    (3.19275, 4.00944),
    (5.05587, 8.09892),
    (9.47112, 4.84053),
    (11.2364, 10.0071),
)


@model_family_meta(
    canonical_name="YOLO",
    citation=(
        'Redmon, Joseph, et al. "You Only Look Once: Unified, Real-Time '
        'Object Detection." Proceedings of the IEEE Conference on '
        "Computer Vision and Pattern Recognition, 2016, pp. 779–788."
    ),
    theory=r"""
    YOLOv2 (a.k.a. YOLO9000) keeps the one-stage, full-image-in-one-pass
    spirit of YOLOv1 and addresses its main weaknesses with a number of
    careful engineering changes.

    **Anchor boxes from data.**  Instead of regressing absolute box
    coordinates from each grid cell, the model predicts offsets relative
    to :math:`k` *anchor priors* per cell.  The priors are obtained by
    running k-means on ground-truth boxes (with a :math:`1 - \mathrm{IoU}`
    distance) so they match the dataset's true aspect-ratio distribution.

    **Constrained location prediction.**  For each anchor the network
    outputs :math:`(t_x, t_y, t_w, t_h, t_o)` decoded as

    .. math::

        b_x = \sigma(t_x) + c_x, \quad b_y = \sigma(t_y) + c_y, \quad
        b_w = p_w e^{t_w}, \quad b_h = p_h e^{t_h},

    where :math:`(c_x, c_y)` is the cell offset and :math:`(p_w, p_h)`
    the prior size.  The :math:`\sigma` on :math:`(t_x, t_y)` keeps the
    centre inside its cell, stabilising early training.

    **Better backbone and tricks.**  Darknet-19, BatchNorm on every
    conv, high-resolution classifier fine-tuning, a passthrough/route
    layer that fuses stride-16 features into the stride-32 head,
    multi-scale training (random :math:`\{320, \dots, 608\}`), and
    joint COCO + ImageNet training (the "9000-class" variant) all
    contribute to a Pareto improvement in both speed and accuracy
    over v1.
    """,
)
@dataclass(frozen=True)
class YOLOV2Config(ModelConfig):
    """Configuration for YOLOv2 / YOLO9000 (Redmon & Farhadi, CVPR 2017).

    Args:
        num_classes:  Number of object classes C (COCO default: 80).
        in_channels:  Input image channels (default 3).
        num_anchors:  Number of anchor boxes per cell A (default 5).
        anchors:      Anchor (width, height) pairs relative to a 32-pixel cell.
                      Must have exactly ``num_anchors`` entries.
        score_thresh: Minimum objectness × class-prob score for inference.
        nms_thresh:   IoU threshold for NMS at inference.
        lambda_coord: Up-weighting for box regression loss.
        lambda_noobj: Down-weighting for no-object confidence loss.
        darknet_reorg: Reproduce the original flat-index passthrough layout.
            Defaults to False for legacy user configurations; published
            pretrained weights select True automatically.

    Examples
    --------
    >>> from lucid.models.vision.yolo import YOLOV2Config
    >>> cfg = YOLOV2Config()
    >>> cfg.num_classes
    80
    >>> cfg.model_type
    'yolo_v2'
    """

    model_type: ClassVar[str] = "yolo_v2"

    num_classes: int = 80
    in_channels: int = 3
    num_anchors: int = 5
    anchors: tuple[tuple[float, float], ...] = field(default=_COCO_ANCHORS)
    score_thresh: float = 0.5
    nms_thresh: float = 0.5
    lambda_coord: float = 5.0
    lambda_noobj: float = 0.5
    # False preserves configurations saved before original-weight parity was
    # established. The published-checkpoint factory explicitly enables this.
    darknet_reorg: bool = False

    def __post_init__(self) -> None:
        if len(self.anchors) != self.num_anchors:
            raise ValueError(
                f"anchors length ({len(self.anchors)}) must equal "
                f"num_anchors ({self.num_anchors})"
            )
        object.__setattr__(self, "anchors", tuple((float(a), float(b)) for a, b in self.anchors))


# ---------------------------------------------------------------------------
# Darknet-19 backbone building blocks
# ---------------------------------------------------------------------------


def _conv_bn_lrelu(
    in_ch: int, out_ch: int, kernel: int, stride: int = 1, padding: int = 0
) -> nn.Sequential:
    """Conv2d → BatchNorm2d → LeakyReLU(0.1)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(negative_slope=0.1),
    )


@final
class _Darknet19(nn.Module):
    """Darknet-19 backbone.

    Returns two feature maps:
      - ``route``: (B, 512, H/16, W/16)  — for passthrough at stride 16
      - ``out``:   (B, 1024, H/32, W/32) — main features at stride 32
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        # Stage 1: 3→32, pool
        self.stage1 = nn.Sequential(
            _conv_bn_lrelu(in_channels, 32, 3, padding=1),
            nn.MaxPool2d(2, stride=2),
        )
        # Stage 2: 32→64, pool
        self.stage2 = nn.Sequential(
            _conv_bn_lrelu(32, 64, 3, padding=1),
            nn.MaxPool2d(2, stride=2),
        )
        # Stage 3: 64→128→64→128, pool
        self.stage3 = nn.Sequential(
            _conv_bn_lrelu(64, 128, 3, padding=1),
            _conv_bn_lrelu(128, 64, 1),
            _conv_bn_lrelu(64, 128, 3, padding=1),
            nn.MaxPool2d(2, stride=2),
        )
        # Stage 4: 128→256→128→256, pool
        self.stage4 = nn.Sequential(
            _conv_bn_lrelu(128, 256, 3, padding=1),
            _conv_bn_lrelu(256, 128, 1),
            _conv_bn_lrelu(128, 256, 3, padding=1),
            nn.MaxPool2d(2, stride=2),
        )
        # Stage 5: 256→512→256→512→256→512 (passthrough hook at stride-16)
        self.stage5 = nn.Sequential(
            _conv_bn_lrelu(256, 512, 3, padding=1),
            _conv_bn_lrelu(512, 256, 1),
            _conv_bn_lrelu(256, 512, 3, padding=1),
            _conv_bn_lrelu(512, 256, 1),
            _conv_bn_lrelu(256, 512, 3, padding=1),
        )
        # Pool between stage5 and stage6
        self.pool56 = nn.MaxPool2d(2, stride=2)
        # Stage 6: 512→1024→512→1024→512→1024 (output at stride-32)
        self.stage6 = nn.Sequential(
            _conv_bn_lrelu(512, 1024, 3, padding=1),
            _conv_bn_lrelu(1024, 512, 1),
            _conv_bn_lrelu(512, 1024, 3, padding=1),
            _conv_bn_lrelu(1024, 512, 1),
            _conv_bn_lrelu(512, 1024, 3, padding=1),
        )

    @override
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Run Darknet-19 and return (route, out).

        Returns:
            (route, out):
                route: (B, 512, H/16, W/16)
                out:   (B, 1024, H/32, W/32)
        """
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        route = self.stage5(x)  # (B, 512, H/16, W/16)
        x = self.pool56(route)
        out = self.stage6(x)  # (B, 1024, H/32, W/32)
        return route, out


def _space_to_depth(x: Tensor, block_size: int) -> Tensor:
    """Rearrange spatial blocks into the channel dimension (passthrough).

    Equivalent to ``nn.PixelUnshuffle`` or ``tf.nn.space_to_depth``.

    Args:
        x:          (B, C, H, W) — H and W must be divisible by block_size.
        block_size: Downscaling factor (typically 2).

    Returns:
        (B, C*block_size², H//block_size, W//block_size)
    """
    B = int(x.shape[0])
    C = int(x.shape[1])
    H = int(x.shape[2])
    W = int(x.shape[3])
    bs = block_size
    assert H % bs == 0 and W % bs == 0, "H and W must be divisible by block_size"

    new_h = H // bs
    new_w = W // bs
    new_c = C * bs * bs

    # Shape — (B, C, H, W)
    # → (B, C, new_h, bs, new_w, bs)
    # Divergence, deliberate: this is the textbook pixel-unshuffle
    # (tf.nn.space_to_depth).  darknet's reorg_layer is *not* — its
    # ``reorg_cpu`` branch indexes as if the input were a
    # (w*stride, h*stride, c/stride^2) tensor, which scrambles channels
    # against spatial position.  The two agree on shape and on the
    # multiset of values, so training from scratch is unaffected, but a
    # port of darknet weights would have to reproduce the quirk exactly
    # — retained for existing user checkpoints; released darknet weights
    # select _darknet_reorg instead.
    x = x.reshape(B, C, new_h, bs, new_w, bs)
    # → (B, new_h, new_w, C, bs, bs)
    x = x.permute(0, 2, 4, 1, 3, 5)
    # → (B, new_h, new_w, new_c)
    x = x.reshape(B, new_h, new_w, new_c)
    # → (B, new_c, new_h, new_w)
    return x.permute(0, 3, 1, 2)


def _darknet_reorg(x: Tensor, block_size: int) -> Tensor:
    """Match original reorg_cpu(forward=0), including its flat-index layout.

    This is not pixel-unshuffle: the original interprets the input buffer
    as fewer channels with larger spatial dimensions before gathering.
    """
    batch, channels, height, width = x.shape
    bs = block_size
    if bs <= 0 or channels % (bs * bs) or height % bs or width % bs:
        raise ValueError("darknet reorg requires divisible channels and spatial dimensions")
    return (
        x.reshape(batch, channels // (bs * bs), height, bs, width, bs)
        .permute(0, 3, 5, 1, 2, 4)
        .reshape(batch, channels * bs * bs, height // bs, width // bs)
    )


# ---------------------------------------------------------------------------
# YOLOv2 model
# ---------------------------------------------------------------------------


class YOLOV2ForObjectDetection(ObjectDetectionModel):
    r"""YOLOv2 / YOLO9000 object detector (Redmon & Farhadi, CVPR 2017).

    The successor to YOLOv1 with several decisive upgrades: a deeper
    **Darknet-19** backbone with batch normalisation, **anchor boxes**
    (5 k-means-clustered priors on COCO instead of YOLOv1's
    direct-regression "anchor-free" approach), and a **passthrough**
    layer that injects fine-grained stride-16 features into the
    stride-32 detection head via space-to-depth reorganisation.
    Anchors give per-cell predictions an inductive bias on shape that
    YOLOv1 lacks, and the passthrough recovers spatial detail lost in
    the final downsample — together they bump VOC 2007 mAP from 63.4%
    (YOLOv1) to 78.6% (paper Table 4).

    The output tensor :math:`(B, A \cdot (5 + C), fH, fW)` encodes
    :math:`A` anchors per cell with :math:`(t_x, t_y, t_w, t_h,
    \mathrm{conf})` deltas plus :math:`C` class scores.

    Parameters
    ----------
    config : YOLOV2Config
        Frozen architecture spec.  Use :func:`yolo_v2`.

    Attributes
    ----------
    config : YOLOV2Config
        Stored copy of the config that built this model.
    backbone : _Darknet19
        Darknet-19 trunk producing a 512-channel stride-16 route feature
        and a 1024-channel stride-32 trunk feature.
    det1, det2, det3 : _conv_bn_lrelu
        Stride-32 head convolutions; ``det3`` consumes the concatenation
        of the 2048-channel space-to-depth passthrough and the 1024-channel
        ``det2`` output.
    pred : nn.Conv2d
        Final 1x1 conv producing :math:`A \cdot (5 + C)` channels (no
        BatchNorm, no activation).

    Notes
    -----
    See Redmon & Farhadi, "YOLO9000: Better, Faster, Stronger",
    CVPR 2017 (arXiv:1612.08242).  Per-anchor box decoding uses the
    "direct location prediction" parameterisation:

    .. math::

        \begin{aligned}
            b_x &= \sigma(t_x) + c_x, & b_w &= p_w \exp(t_w), \\
            b_y &= \sigma(t_y) + c_y, & b_h &= p_h \exp(t_h),
        \end{aligned}

    where :math:`(c_x, c_y)` is the cell top-left in feature coords and
    :math:`(p_w, p_h)` is the anchor's prior size.  The
    sigmoid-on-:math:`t_{x,y}` bounds the centre to lie inside the
    responsible cell — a stability win over YOLOv1's unbounded
    regression.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.yolo._v2 import yolo_v2
    >>> model = yolo_v2()
    >>> x = lucid.randn(1, 3, 416, 416)
    >>> out = model(x)
    >>> out.logits.shape[0]
    1
    """

    config_class: ClassVar[type[YOLOV2Config]] = YOLOV2Config
    base_model_prefix: ClassVar[str] = "backbone"

    config: YOLOV2Config

    def __init__(self, config: YOLOV2Config) -> None:
        super().__init__(config)
        A = config.num_anchors
        C = config.num_classes

        # Backbone
        self.backbone = _Darknet19(config.in_channels)
        route_ch = 512

        # Detection head conv layers (applied to stride-32 feature map)
        self.det1 = _conv_bn_lrelu(1024, 1024, 3, padding=1)
        self.det2 = _conv_bn_lrelu(1024, 1024, 3, padding=1)

        # Passthrough, per darknet's yolov2.cfg:
        #     [route] -9  →  conv 64 1x1 (BN+leaky)  →  [reorg] stride 2 (→256)
        #     [route] -1,-4  →  1280 ch  →  conv 1024 3x3
        # The paper's prose instead describes reorganising the full 26x26x512
        # route straight into 13x13x2048 and concatenating that.  The two
        # disagree; the cfg is what every released YOLOv2 weight file was
        # trained with, so the cfg wins here and the divergence is called out
        # in the module docstring.
        self.route_reduce = _conv_bn_lrelu(route_ch, 64, 1)
        pt_ch = 64 * 4  # space-to-depth s=2 on the reduced route → 256
        merged_ch = pt_ch + 1024  # 1280

        self.det3 = _conv_bn_lrelu(merged_ch, 1024, 3, padding=1)
        # Final prediction conv: no BN, no activation
        self.pred = nn.Conv2d(1024, A * (5 + C), 1)

    def _forward_raw(self, x: Tensor) -> Tensor:
        """Run backbone + detection head → raw prediction tensor.

        Returns:
            (B, A*(5+C), H/32, W/32)
        """
        route, feat = self.backbone(x)  # (B,512,H/16,W/16), (B,1024,H/32,W/32)
        feat = self.det1(feat)  # (B, 1024, H/32, W/32)
        feat = self.det2(feat)

        # Passthrough: 1x1 reduce to 64 ch, then space_to_depth → (B, 256, ...)
        route = self.route_reduce(route)
        passthrough = (
            _darknet_reorg(route, 2) if self.config.darknet_reorg else _space_to_depth(route, 2)
        )

        # Concatenate along channel axis
        feat = lucid.cat([passthrough, feat], dim=1)  # (B, 1280, H/32, W/32)
        feat = self.det3(feat)  # (B, 1024, H/32, W/32)
        return self.pred(feat)  # (B, A*(5+C), H/32, W/32)

    def _decode_boxes(self, raw_r: Tensor, image_size: tuple[int, int]) -> Tensor:
        """Unclipped xyxy pixel boxes for every predictor.

        Args:
            raw_r:      (B, fH, fW, A, 5+C) predictions.
            image_size: (H, W) of the input image in pixels.

        Returns:
            (B, fH, fW, A, 4) boxes.  Postprocess clips them to the image;
            the loss matches against them as decoded, as darknet does.
        """
        anchors = self.config.anchors
        H, W = image_size
        fH = int(raw_r.shape[1])
        fW = int(raw_r.shape[2])
        A = int(raw_r.shape[3])
        device, dt = raw_r.device, raw_r.dtype
        stride_h = H / fH
        stride_w = W / fW

        # Offsets and priors in the head's dtype: a half model decodes in half.
        col_offsets = lucid.arange(fW, dtype=dt, device=device).reshape(1, 1, fW, 1)
        row_offsets = lucid.arange(fH, dtype=dt, device=device).reshape(1, fH, 1, 1)
        priors = lucid.tensor(anchors, dtype=dt, device=device)  # (A, 2)
        aw_t = priors[:, 0].reshape(1, 1, 1, A)
        ah_t = priors[:, 1].reshape(1, 1, 1, A)

        # bx — (sigmoid(tx) + col) * stride_w; bw — anchor_w * exp(tw) * stride_w
        bx = (F.sigmoid(raw_r[..., 0]) + col_offsets) * stride_w
        by = (F.sigmoid(raw_r[..., 1]) + row_offsets) * stride_h
        bw = aw_t * lucid.exp(raw_r[..., 2].clamp(min=-10.0, max=10.0)) * stride_w
        bh = ah_t * lucid.exp(raw_r[..., 3].clamp(min=-10.0, max=10.0)) * stride_h
        return lucid.stack([bx - bw / 2.0, by - bh / 2.0, bx + bw / 2.0, by + bh / 2.0], dim=-1)

    def _decode_predictions(
        self,
        raw: Tensor,
        image_size: tuple[int, int],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Decode raw anchor predictions to flat predictions.

        Args:
            raw:        (B, A*(5+C), fH, fW) from the prediction conv.
            image_size: (H, W) of the input image in pixels.

        Returns:
            (logits, pred_boxes):
                logits:     (B, fH*fW*A, C) raw class scores.
                pred_boxes: (B, fH*fW*A, 4) decoded xyxy pixel boxes.
        """
        cfg = self.config
        A = cfg.num_anchors
        C = cfg.num_classes
        H, W = image_size

        B_batch = int(raw.shape[0])
        fH = int(raw.shape[2])
        fW = int(raw.shape[3])

        # (B, A*(5+C), fH, fW) → (B, A, 5+C, fH, fW) → (B, fH, fW, A, 5+C)
        raw = raw.reshape(B_batch, A, 5 + C, fH, fW)
        raw = raw.permute(0, 3, 4, 1, 2)  # (B, fH, fW, A, 5+C)

        boxes = self._decode_boxes(raw, image_size)
        boxes = lucid.stack(
            [
                boxes[..., 0].clamp(min=0.0, max=float(W)),
                boxes[..., 1].clamp(min=0.0, max=float(H)),
                boxes[..., 2].clamp(min=0.0, max=float(W)),
                boxes[..., 3].clamp(min=0.0, max=float(H)),
            ],
            dim=-1,
        )

        # Flatten spatial and anchor dims: (B, fH*fW*A, 4)
        pred_boxes = boxes.reshape(B_batch, fH * fW * A, 4)

        # Class logits: (B, fH, fW, A, C) → (B, fH*fW*A, C)
        logits = raw[..., 5:].reshape(B_batch, fH * fW * A, C)

        # Objectness σ(t_o) — Pr(object)·IOU.  darknet's region layer scores a
        # detection as σ(t_o)·softmax(class); dropping it lets every empty-cell
        # anchor compete on class probability alone.
        conf = F.sigmoid(raw[..., 4]).reshape(B_batch, fH * fW * A)

        return logits, pred_boxes, conf

    def _compute_loss(
        self,
        raw: Tensor,
        targets: list[dict[str, Tensor]],
        image_size: tuple[int, int],
    ) -> Tensor:
        """Compute YOLOv2 multi-part MSE/CE loss.

        Args:
            raw:        (B, A*(5+C), fH, fW) raw network output.
            targets:    List of B dicts with "boxes" (M,4) xyxy [0,1]
                        and "labels" (M,) int.
            image_size: (H, W) in pixels.

        Returns:
            Scalar loss tensor.
        """
        cfg = self.config
        A = cfg.num_anchors
        C = cfg.num_classes
        B_batch = int(raw.shape[0])
        fH = int(raw.shape[2])
        fW = int(raw.shape[3])

        # (B, fH, fW, A, 5+C), flattened so predictor p = (row*fW + col)*A + a.
        raw_r = raw.reshape(B_batch, A, 5 + C, fH, fW).permute(0, 3, 4, 1, 2)
        n_pred = fH * fW * A
        flat = raw_r.reshape(B_batch, n_pred, 5 + C)
        # The matching IoU is a target, so it is decoded without gradient.
        boxes = self._decode_boxes(raw_r.detach(), image_size)
        boxes = boxes.reshape(B_batch, n_pred, 4)

        loss_parts = [
            self._image_loss(flat[bi], boxes[bi], targets[bi], image_size, (fH, fW))
            for bi in range(B_batch)
        ]
        return lucid.cat(loss_parts).mean()

    def _image_loss(
        self,
        pred: Tensor,
        pred_boxes: Tensor,
        target: dict[str, Tensor],
        image_size: tuple[int, int],
        grid: tuple[int, int],
    ) -> Tensor:
        """One image's loss over every predictor at once.

        ``pred`` is ``(P, 5+C)`` and ``pred_boxes`` its ``(P, 4)`` decoded
        boxes, both in ``(row, col, anchor)`` order.  Assignment, IoU
        targets and masks are tensor ops, so nothing is read to the host.
        """
        cfg = self.config
        C = cfg.num_classes
        A = cfg.num_anchors
        H, W = image_size
        fH, fW = grid
        stride_h = H / fH
        stride_w = W / fW
        dev, dt = pred.device, pred.dtype
        n_pred = int(pred.shape[0])

        scale = lucid.tensor([W, H, W, H], dtype=dt, device=dev)
        gt_xyxy = target["boxes"].to(device=dev, dtype=dt) * scale
        M = int(gt_xyxy.shape[0])

        # Per-predictor best IoU against every GT in this image.  darknet's
        # region layer needs it twice: the objectness target is the IoU
        # itself (cfg ``rescore=1``), and an unmatched predictor whose IoU
        # already exceeds ``thresh`` is *skipped* rather than pushed to zero
        # — otherwise a near-duplicate good box is trained as background.
        sig_conf = F.sigmoid(pred[:, 4])
        if M == 0:
            background_loss = (sig_conf**2).sum()
            return (cfg.lambda_noobj * background_loss / float(n_pred)).reshape(1)
        best_iou = box_iou(pred_boxes, gt_xyxy).max(dim=1).values  # (P,)

        gt_cx = (gt_xyxy[:, 0] + gt_xyxy[:, 2]) / 2.0
        gt_cy = (gt_xyxy[:, 1] + gt_xyxy[:, 3]) / 2.0
        gt_w = gt_xyxy[:, 2] - gt_xyxy[:, 0]
        gt_h = gt_xyxy[:, 3] - gt_xyxy[:, 1]

        # Responsible cell holds the centre; responsible anchor is the one
        # whose shape best matches the GT's (IoU of centred boxes).
        col = (gt_cx / stride_w).floor().clamp(0.0, float(fW - 1))
        row = (gt_cy / stride_h).floor().clamp(0.0, float(fH - 1))
        aw = lucid.tensor([a[0] * stride_w for a in cfg.anchors], dtype=dt, device=dev)
        ah = lucid.tensor([a[1] * stride_h for a in cfg.anchors], dtype=dt, device=dev)
        inter = lucid.minimum(gt_w[:, None], aw[None, :]) * lucid.minimum(
            gt_h[:, None], ah[None, :]
        )
        union = (gt_w * gt_h)[:, None] + (aw * ah)[None, :] - inter
        best_a = (inter / union.clamp(min=1e-6)).argmax(dim=1)  # (M,)
        idx = ((row * float(fW) + col) * float(A)).long() + best_a  # (M,)

        # Every object that claims a predictor contributes; a dict keyed by
        # (row, col, anchor) once erased all but the last claimant, and
        # darknet accumulates deltas for all of them.
        sel = pred[idx]  # (M, 5+C)

        # tx, ty targets: fractional cell offset (sigmoid space)
        tgt_tx = (gt_cx / stride_w - col).clamp(1e-6, 1.0 - 1e-6)
        tgt_ty = (gt_cy / stride_h - row).clamp(1e-6, 1.0 - 1e-6)
        loss_xy = (
            (F.sigmoid(sel[:, 0]) - tgt_tx) ** 2 + (F.sigmoid(sel[:, 1]) - tgt_ty) ** 2
        ).sum()

        # tw, th targets: log(gt / anchor)
        tgt_tw = lucid.log(gt_w.clamp(min=1e-6) / aw[best_a].clamp(min=1e-6))
        tgt_th = lucid.log(gt_h.clamp(min=1e-6) / ah[best_a].clamp(min=1e-6))
        loss_wh = ((sel[:, 2] - tgt_tw) ** 2 + (sel[:, 3] - tgt_th) ** 2).sum()

        # Objectness regresses Pr(object) x IoU, which is what the paper
        # defines sigma(t_o) to be and what ``rescore=1`` selects in darknet.
        # A constant 1.0 target makes confidence uninformative for ranking.
        loss_conf_o = ((sig_conf[idx] - best_iou[idx]) ** 2).sum()

        # Class loss: MSE vs one-hot on the SOFTMAX probabilities — the same
        # quantity postprocess thresholds.  Training the raw logits instead
        # drives them to a one-hot *logit* row, whose softmax peak is only
        # 1/(C-1+e) ≈ 0.03 for C=80, so a perfectly fitted model would emit
        # no detections at all.  Darknet's region layer likewise takes the
        # delta against the softmax.
        one_hot = F.one_hot(target["labels"].to(dev).long(), C).to(dt)
        loss_cls = ((F.softmax(sel[:, 5:], dim=-1) - one_hot) ** 2).sum()

        # cfg [region] thresh = .6 — an unmatched predictor that already
        # overlaps a GT well is left alone.
        assigned = F.one_hot(idx, n_pred).to(dt).sum(dim=0).clamp(0.0, 1.0)
        background = (1.0 - assigned) * (best_iou <= _NOOBJ_IOU_THRESH).to(dt)
        loss_conf_n = (sig_conf**2 * background).sum()

        # darknet's region layer is a SUM of squared errors over all cells and
        # predictors, not a per-group mean: averaging each group separately
        # renormalises the ~S²·B background terms against the handful of
        # object terms, which is exactly what lambda_noobj exists to control.
        #
        # Two denominators, not five.  Giving each group its own divisor would
        # set ``conf_noobj``'s to ~S²·A and ``conf_obj``'s to the object
        # count, and the ratio between those two is precisely what
        # ``lambda_noobj`` exists to set.  Dividing the pair by the *same*
        # count leaves that ratio alone — as it does for ``lambda_coord``
        # across the localisation terms — while removing the grid size from
        # the total.  Without this the loss tracks the cell count almost
        # exactly (69.9 at 320², 239.3 at 608² on a fixed batch), so no single
        # learning rate serves the multi-scale schedule the paper trains with.
        # This is what YOLOv3 does; v2 predates it.
        localisation = cfg.lambda_coord * (loss_xy + loss_wh) + loss_cls
        objectness = loss_conf_o + cfg.lambda_noobj * loss_conf_n
        img_loss: Tensor = localisation / float(M) + objectness / float(n_pred)
        return img_loss.reshape(1)

    @override
    def forward(
        self,
        x: Tensor,
        targets: list[dict[str, Tensor]] | None = None,
        image_size: tuple[int, int] | None = None,
    ) -> ObjectDetectionOutput:
        """Run YOLOv2 forward pass.

        Args:
            x:          (B, C, H, W) input image tensor.
            targets:    Optional list of B dicts for training loss.
                        Each dict has:
                        - ``"boxes"``:  (M_i, 4) xyxy boxes normalised to [0,1].
                        - ``"labels"``: (M_i,) integer class indices.
            image_size: (H, W) of the input image in pixels.  Defaults to
                        ``x``'s own spatial size; pass it only to assert
                        the resolution you expect.  Boxes are decoded in
                        pixels of this size, so a value that disagrees
                        with ``x`` is rejected rather than silently
                        rescaling every box.

        Returns:
            :class:`~lucid.models._output.ObjectDetectionOutput` with:
            - ``logits``:     (B, fH*fW*A, num_classes) raw class scores
            - ``pred_boxes``: (B, fH*fW*A, 4) decoded xyxy pixel boxes
            - ``loss``:       scalar tensor when targets are provided, else None
        """
        check_per_image(type(self).__name__, int(x.shape[0]), targets=targets)
        actual = (int(x.shape[2]), int(x.shape[3]))
        if image_size is not None and tuple(image_size) != actual:
            raise ValueError(
                f"image_size={tuple(image_size)} does not match the input, "
                f"which is {actual}.  Decoded boxes are in pixels of "
                "image_size, so a mismatch would rescale every box by "
                f"{actual[0] / image_size[0]:.3g}x vertically and "
                f"{actual[1] / image_size[1]:.3g}x horizontally.  Omit "
                "image_size to use the input's own resolution."
            )
        image_size = actual

        raw = self._forward_raw(x)
        logits, pred_boxes, objectness = self._decode_predictions(raw, image_size)

        loss: Tensor | None = None
        if targets is not None:
            loss = self._compute_loss(raw, targets, image_size)

        return ObjectDetectionOutput(
            logits=logits,
            pred_boxes=pred_boxes,
            loss=loss,
            objectness=objectness,
        )

    def postprocess(
        self,
        output: ObjectDetectionOutput,
        image_size: tuple[int, int] = (416, 416),
    ) -> list[dict[str, Tensor]]:
        """Per-class NMS on raw predictions.

        Args:
            output:     Output from :meth:`forward`.
            image_size: (H, W) of the input image.

        Returns:
            List of per-image result dicts with ``"boxes"``, ``"scores"``,
            ``"labels"``.
        """
        del image_size  # pred_boxes are already in pixel coordinates
        cfg = self.config
        B_batch = int(output.logits.shape[0])
        C = cfg.num_classes
        # Everything built here lives with the predictions; on the CPU it
        # made NMS raise on a Metal model.
        dev = output.logits.device
        results: list[dict[str, Tensor]] = []

        for b in range(B_batch):
            lg_b = output.logits[b]  # (fH*fW*A, C)
            bx_b = output.pred_boxes[b]  # (fH*fW*A, 4)

            # Class probabilities via softmax
            sc_b = F.softmax(lg_b, dim=-1)  # (N, C)
            if output.objectness is not None:
                sc_b = sc_b * output.objectness[b][:, None]

            per_class = _threshold_by_class(sc_b, cfg.score_thresh)

            keep_boxes: list[Tensor] = []
            keep_scores: list[Tensor] = []
            keep_labels: list[Tensor] = []

            for c in range(C):
                mask = per_class[c]
                if not mask:
                    continue
                mask_t = lucid.tensor(mask, device=dev).long()
                sc_sel = sc_b[:, c][mask_t]
                bx_sel = bx_b[mask_t]
                keep = batched_nms(
                    bx_sel,
                    sc_sel,
                    lucid.zeros(len(mask), device=dev),
                    cfg.nms_thresh,
                )
                keep_boxes.append(bx_sel[keep])
                keep_scores.append(sc_sel[keep])
                keep_labels.append(
                    lucid.full((int(keep.shape[0]),), c, dtype=lucid.int64, device=dev)
                )

            if keep_boxes:
                results.append(
                    detection_result(
                        lucid.cat(keep_boxes, dim=0),
                        lucid.cat(keep_scores, dim=0),
                        lucid.cat(keep_labels, dim=0),
                    )
                )
            else:
                results.append(empty_detection_result(dev))

        return results


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

_CFG_V2 = YOLOV2Config(
    num_classes=80,
    in_channels=3,
    num_anchors=5,
    anchors=_COCO_ANCHORS,
    score_thresh=0.5,
    nms_thresh=0.5,
    lambda_coord=5.0,
    lambda_noobj=0.5,
)


@register_model(
    task="object-detection",
    family="yolo",
    model_type="yolo_v2",
    model_class=YOLOV2ForObjectDetection,
    default_config=_CFG_V2,
    params=50962889,
)
def yolo_v2(
    pretrained: bool | str = False,
    *,
    weights: YOLOV2Weights | None = None,
    **overrides: object,
) -> YOLOV2ForObjectDetection:
    r"""YOLOv2 / YOLO9000 with Darknet-19 backbone (Redmon & Farhadi, CVPR 2017).

    Builds the paper-cited YOLOv2 detector: 19-layer Darknet-19 backbone,
    5 anchor priors clustered by k-means on the training set, passthrough
    space-to-depth layer, and 80 COCO classes by default.  At the
    standard 416x416 input the output grid is 13x13 (stride 32).
    Approximately 50M parameters; COCO test-dev mAP@.5 of 44.0%
    (paper Table 4).

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True`` →
        the ``DEFAULT`` tag (:attr:`YOLOV2Weights.COCO_2014`, converted
        from darknet's ``yolov2.weights``); a tag string → that specific
        checkpoint.  Mutually exclusive with ``weights`` (which wins if
        both are given).
    weights : YOLOV2Weights, optional, keyword-only
        Explicit weights enum member, e.g. ``YOLOV2Weights.COCO_2014``.
        Takes precedence over ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`YOLOV2Config` —
        ``num_classes``, ``num_anchors``, anchor priors, ``score_thresh``,
        ``nms_thresh``, etc.

    Returns
    -------
    YOLOV2ForObjectDetection
        Detector with the standard YOLOv2 configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See Redmon & Farhadi, "YOLO9000: Better, Faster, Stronger",
    CVPR 2017 (arXiv:1612.08242).  The five default anchors come from
    k-means clustering of VOC / COCO ground-truth boxes (paper Table 1).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.yolo._v2 import yolo_v2
    >>> model = yolo_v2()
    >>> x = lucid.randn(1, 3, 416, 416)
    >>> out = model(x)
    >>> out.logits.shape[0]
    1
    """
    entry = weights_mod.resolve_weights(YOLOV2Weights, pretrained, weights)
    if entry is not None:
        if overrides.get("darknet_reorg") is False:
            raise ValueError("published YOLOv2 weights require darknet_reorg=True")
        overrides = {**overrides, "darknet_reorg": True}
    model = YOLOV2ForObjectDetection(apply_overrides(_CFG_V2, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="yolo_v2")
    return model
