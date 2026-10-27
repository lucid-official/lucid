"""YOLOv1 — You Only Look Once (Redmon et al., CVPR 2016).

Paper: "You Only Look Once: Unified, Real-Time Object Detection"

Key ideas
----------
- Divide the image into an S×S grid; each cell predicts B bounding boxes
  and C class probabilities simultaneously in a single forward pass.
- Each box prediction consists of (cx, cy, w, h, conf) where cx/cy are
  offsets relative to the cell, and w/h are relative to the image size.
- At inference, class-specific confidence scores filter predictions via NMS.
- Training loss is a multi-part MSE loss with λ_coord=5 (up-weights box
  regression) and λ_noobj=0.5 (down-weights no-object confidence).

Architecture (Darknet backbone from the paper)
----------------------------------------------
  Image → Darknet conv layers → AdaptiveAvgPool2d(S,S) → flatten
        → FC(1024·S², 4096) → LeakyReLU(0.1)
        → FC(4096, S²·(B·5+C))

Each conv block uses Conv2d → BatchNorm2d → LeakyReLU(0.1).
"M" entries become MaxPool2d(2,2); list entries are repeated conv groups.

Losses (training)
-----------------
Requires ``targets`` — list of B dicts with:
  ``"boxes"``  : (M_i, 4) xyxy, normalised to [0, 1] by image H/W.
  ``"labels"`` : (M_i,)  integer foreground class ids (0-indexed).

Multi-part MSE loss matching the paper §2.2:
  L = λ_coord · L_xy  +  λ_coord · L_wh  +  L_conf_obj
    +  λ_noobj · L_conf_noobj  +  L_cls
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, cast, override

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
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained

if TYPE_CHECKING:
    pass


def _threshold_by_class(scores: Tensor, thresh: float) -> list[list[int]]:
    """Per class, the ascending row indices whose score clears ``thresh``.

    ``scores`` is one image's ``(N, C)`` score matrix.  It is read back
    with a single ``.tolist()``: the per-element ``.item()`` scan this
    replaces was ``N * C`` device round-trips (67,600 for YOLOv2 at
    416px with 80 classes) and cost several forward passes.
    """
    C = int(scores.shape[1])
    flat = cast(list[float], scores.reshape(-1).tolist())
    per_class: list[list[int]] = [[] for _ in range(C)]
    for idx, sc in enumerate(flat):
        if sc >= thresh:
            row, c = divmod(idx, C)
            per_class[c].append(row)
    return per_class


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Darknet backbone architecture spec used by the full model
_ARCH_FULL: list[object] = [
    (64, 7, 2, 3),
    "M",
    (192, 3, 1, 1),
    "M",
    (128, 1, 1, 0),
    (256, 3, 1, 1),
    (256, 1, 1, 0),
    (512, 3, 1, 1),
    "M",
    [(256, 1, 1, 0), (512, 3, 1, 1), 4],
    (512, 1, 1, 0),
    (1024, 3, 1, 1),
    "M",
    [(512, 1, 1, 0), (1024, 3, 1, 1), 2],
    (1024, 3, 1, 1),
    (1024, 3, 2, 1),
    (1024, 3, 1, 1),
    (1024, 3, 1, 1),
]

# Tiny Darknet architecture spec.  yolov1-tiny.cfg tapers 1024 -> 256 right
# before the connected layer, precisely so the FC head stays small: 7*7*256
# -> 1470 is ~18M weights.  Keeping three 1024-channel convs made the head
# 7*7*1024 -> 1470 and the whole model ~236M params, five times the size the
# "fast" variant exists to avoid.
_ARCH_TINY: list[object] = [
    (16, 3, 1, 1),
    "M",
    (32, 3, 1, 1),
    "M",
    (64, 3, 1, 1),
    "M",
    (128, 3, 1, 1),
    "M",
    (256, 3, 1, 1),
    "M",
    (512, 3, 1, 1),
    "M",
    (1024, 3, 1, 1),
    (256, 3, 1, 1),
]


@model_family_meta(
    canonical_name="YOLO",
    citation=(
        'Redmon, Joseph, et al. "You Only Look Once: Unified, Real-Time '
        'Object Detection." Proceedings of the IEEE Conference on '
        "Computer Vision and Pattern Recognition, 2016, pp. 779–788."
    ),
    theory=r"""
    YOLOv1 reframes detection as a **single regression problem**.
    The input image is divided into an :math:`S \times S` grid; each
    cell is responsible for objects whose centre falls inside it.
    For each cell the network predicts :math:`B` candidate boxes
    :math:`(x, y, w, h, \mathrm{conf})` plus one shared class
    distribution over :math:`C` categories, so the head output has
    shape :math:`S \times S \times (B \cdot 5 + C)`.

    A single CNN (the Darknet trunk in the paper) processes the full
    image once and emits all detections in one forward pass.  The
    confidence target is :math:`\Pr(\mathrm{obj}) \cdot
    \mathrm{IoU}^{\mathrm{truth}}_{\mathrm{pred}}` so that confidence
    encodes both objectness and localisation quality.  Training uses
    a weighted sum-squared-error loss

    .. math::

        \mathcal{L} =
            \lambda_{\mathrm{coord}} \sum_{i,j}
              \mathbb{1}^{\mathrm{obj}}_{ij}
              \bigl[(x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2\bigr]
          + \lambda_{\mathrm{coord}} \sum_{i,j}
              \mathbb{1}^{\mathrm{obj}}_{ij}
              \bigl[(\sqrt{w_i} - \sqrt{\hat{w}_i})^2
                  + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2\bigr]
          + \sum_{i,j} \mathbb{1}^{\mathrm{obj}}_{ij}
              (C_i - \hat{C}_i)^2
          + \lambda_{\mathrm{noobj}} \sum_{i,j}
              \mathbb{1}^{\mathrm{noobj}}_{ij}
              (C_i - \hat{C}_i)^2
          + \sum_{i} \mathbb{1}^{\mathrm{obj}}_{i}
              \sum_{c}(p_i(c) - \hat{p}_i(c))^2,

    where :math:`\sqrt{w}, \sqrt{h}` make the regression less
    sensitive to absolute scale and :math:`\lambda_{\mathrm{coord}},
    \lambda_{\mathrm{noobj}}` rebalance the dense background term.

    Because the entire pipeline is a single CNN, YOLOv1 runs at
    45 fps (155 fps for the *tiny* variant) — orders of magnitude
    faster than the two-stage R-CNN family — establishing the
    one-stage detection paradigm.  Trade-off: each cell predicts a
    fixed small :math:`B`, limiting recall on clustered small
    objects, which is exactly what YOLOv2/v3/v4 progressively fix
    with anchors, multi-scale heads, and a stronger backbone/neck.
    """,
)
@dataclass(frozen=True)
class YOLOConfig(ModelConfig):
    """Configuration for YOLOv1 (Redmon et al., CVPR 2016).

    Args:
        num_classes:  Number of object classes C (COCO default: 80).
        in_channels:  Input image channels.
        split_size:   Grid cells per side S (default 7 → 7×7 grid).
        num_boxes:    Bounding boxes predicted per cell B (default 2).
        lambda_coord: Up-weighting factor for box coordinate loss.
        lambda_noobj: Down-weighting factor for no-object confidence loss.
        score_thresh: Minimum class-confidence score at inference.
        nms_thresh:   IoU threshold for NMS.
        tiny:         Use the tiny Darknet backbone instead of the full one.

    Examples
    --------
    >>> from lucid.models.vision.yolo import YOLOConfig
    >>> cfg = YOLOConfig()
    >>> cfg.num_classes
    80
    >>> cfg.model_type
    'yolo'
    """

    model_type: ClassVar[str] = "yolo"

    num_classes: int = 80
    in_channels: int = 3
    split_size: int = 7
    num_boxes: int = 2
    lambda_coord: float = 5.0
    lambda_noobj: float = 0.5
    score_thresh: float = 0.5
    nms_thresh: float = 0.5
    tiny: bool = False


# ---------------------------------------------------------------------------
# Darknet backbone
# ---------------------------------------------------------------------------


def _conv_block(in_ch: int, out_ch: int, kernel: int, stride: int, padding: int) -> nn.Sequential:
    """Conv2d → BatchNorm2d → LeakyReLU(0.1) block.

    Divergence, deliberate: the CVPR-2016 paper describes plain
    conv + leaky-ReLU, and batch normalisation is introduced only in
    YOLO9000, where it heads the improvements table at +2% mAP.  But
    pjreddie's shipped ``cfg/yolov1.cfg`` carries ``batch_normalize=1``
    on all 24 convs — the released cfg is a later retrain, so paper and
    reference genuinely disagree.  Following the cfg, which is what any
    future weight port would need.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(negative_slope=0.1),
    )


def _build_darknet(
    in_channels: int,
    arch: list[object],
) -> tuple[nn.Sequential, int]:
    """Build the Darknet convolutional backbone from an architecture spec.

    Returns:
        (sequential, out_channels): The backbone module and its output channels.
    """
    layers: list[nn.Module] = []
    ch = in_channels

    for entry in arch:
        if entry == "M":
            layers.append(nn.MaxPool2d(2, stride=2))
        elif isinstance(entry, tuple):
            out_ch, k, s, p = entry
            layers.append(_conv_block(ch, out_ch, k, s, p))
            ch = out_ch
        elif isinstance(entry, list):
            # Format — [conv1_spec, conv2_spec, repeats]
            # The list ends with an int (repeat count); preceding elements are specs
            repeats = int(entry[-1])
            specs: list[tuple[int, int, int, int]] = [
                cast(tuple[int, int, int, int], e) for e in entry[:-1]
            ]
            for _ in range(repeats):
                for spec in specs:
                    out_ch, k, s, p = spec
                    layers.append(_conv_block(ch, out_ch, k, s, p))
                    ch = out_ch
        else:
            raise ValueError(f"Unknown arch entry: {entry!r}")

    return nn.Sequential(*layers), ch


# ---------------------------------------------------------------------------
# YOLOv1 model
# ---------------------------------------------------------------------------


class YOLOForObjectDetection(ObjectDetectionModel):
    r"""YOLOv1 single-shot object detector (Redmon et al., CVPR 2016).

    The original "You Only Look Once" detector — recasts object detection
    as a single regression problem.  An AlexNet-derived Darknet backbone
    processes the whole image, then an adaptive-average-pool + two FC
    layers produce an :math:`S \times S \times (B \cdot 5 + C)` tensor:
    for each of :math:`S^2` grid cells, :math:`B` candidate boxes
    :math:`(x, y, w, h, \mathrm{conf})` and :math:`C` class probabilities.
    A box is responsible for an object iff the object's centre falls
    inside that cell.  No anchors, no region proposals — a single
    forward pass yields all detections at 45 fps on a Titan X.

    Parameters
    ----------
    config : YOLOConfig
        Frozen architecture spec.  Use :func:`yolo` for the full
        Darknet backbone or :func:`yolo_tiny` for the lightweight
        Tiny variant.  Both use the paper's ``S = 7``, ``B = 2`` grid but
        default to ``C = 80`` (COCO), the repo-wide convention for
        detection configs — the paper's own model is ``C = 20`` (VOC),
        which you get by passing ``num_classes=20``.

    Attributes
    ----------
    config : YOLOConfig
        Stored copy of the config that built this model.
    darknet : nn.Sequential
        Backbone convolutional stack (full or tiny depending on
        ``config.tiny``).
    pool : nn.AdaptiveAvgPool2d
        Pools backbone output to a fixed :math:`S \times S` map.
    fc1, fc2 : nn.Linear
        Two-layer FC head producing the
        :math:`S \cdot S \cdot (B \cdot 5 + C)` flat prediction tensor.

    Notes
    -----
    See Redmon et al., "You Only Look Once: Unified, Real-Time Object
    Detection", CVPR 2016 (arXiv:1506.02640).  The defining multi-part
    loss is

    .. math::

        \begin{aligned}
            \mathcal{L} =\;
                & \lambda_\mathrm{coord} \sum_{i, j}
                    \mathbb{1}_{ij}^\mathrm{obj}
                    \bigl[(x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2\bigr] \\
                & + \lambda_\mathrm{coord} \sum_{i, j}
                    \mathbb{1}_{ij}^\mathrm{obj}
                    \bigl[(\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2\bigr] \\
                & + \sum_{i, j} \mathbb{1}_{ij}^\mathrm{obj} (C_i - \hat{C}_i)^2
                + \lambda_\mathrm{noobj}
                    \sum_{i, j} \mathbb{1}_{ij}^\mathrm{noobj}
                    (C_i - \hat{C}_i)^2 \\
                & + \sum_i \mathbb{1}_i^\mathrm{obj}
                    \sum_{c \in \mathcal{C}} (p_i(c) - \hat{p}_i(c))^2,
        \end{aligned}

    with :math:`\lambda_\mathrm{coord} = 5`,
    :math:`\lambda_\mathrm{noobj} = 0.5` to balance localisation vs.
    background regression.  The square-root parameterisation on width
    / height down-weights errors in large boxes.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.yolo._v1 import yolo
    >>> model = yolo()
    >>> x = lucid.randn(1, 3, 448, 448)   # YOLOv1 trained at 448
    >>> out = model(x)
    >>> out.logits.shape[-1] >= 20
    True
    """

    config_class: ClassVar[type[YOLOConfig]] = YOLOConfig
    base_model_prefix: ClassVar[str] = "darknet"

    config: YOLOConfig

    def __init__(self, config: YOLOConfig) -> None:
        super().__init__(config)
        S = config.split_size
        B = config.num_boxes
        C = config.num_classes

        arch: list[object] = _ARCH_TINY if config.tiny else _ARCH_FULL
        self.darknet, backbone_ch = _build_darknet(config.in_channels, arch)

        self.pool = nn.AdaptiveAvgPool2d((S, S))
        flat = backbone_ch * S * S
        self.fc1 = nn.Linear(flat, 4096)
        self.act1 = nn.LeakyReLU(negative_slope=0.1)
        self.drop = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(4096, S * S * (B * 5 + C))

    def _forward_features(self, x: Tensor) -> Tensor:
        """Run backbone + pool + FC → raw grid predictions.

        Returns:
            (B_batch, S, S, num_boxes*5 + num_classes) raw predictions.
        """
        cfg = self.config
        S = cfg.split_size
        B_boxes = cfg.num_boxes
        C = cfg.num_classes
        batch = int(x.shape[0])

        feat = self.darknet(x)  # (N, ch, H', W')
        feat = self.pool(feat)  # (N, ch, S, S)
        feat = feat.flatten(1)  # (N, ch*S*S)
        feat = self.act1(self.fc1(feat))
        feat = self.drop(feat)
        feat = self.fc2(feat)  # (N, S*S*(B*5+C))
        return feat.reshape(batch, S, S, B_boxes * 5 + C)

    def _decode_boxes(self, box_preds: Tensor, image_size: tuple[int, int]) -> Tensor:
        """Unclipped xyxy pixel boxes for every predictor.

        Args:
            box_preds:  (batch, S, S, num_boxes, 5) per-box predictions.
            image_size: (H, W) of the input in pixels.

        Returns:
            (batch, S, S, num_boxes, 4) boxes.  Postprocess clips them to
            the image; the loss matches against them as decoded.
        """
        S = self.config.split_size
        H, W = image_size
        device, dt = box_preds.device, box_preds.dtype
        cols = lucid.arange(S, dtype=dt, device=device).reshape(1, 1, S, 1)
        rows = lucid.arange(S, dtype=dt, device=device).reshape(1, S, 1, 1)

        # ``yolov1.cfg`` ends with ``[connected] activation=linear`` and
        # ``[detection] sqrt=1``: x/y are *raw* cell-relative offsets and the
        # w/h channels are literally sqrt(w), sqrt(h), which
        # ``detection_layer.c`` squares at decode.  The paper only requires the
        # values to land in [0, 1] and prescribes no squashing function; a
        # sigmoid is YOLOv2's anchor parameterisation, and it also makes w = 0
        # and w = 1 unreachable except in the limit.  The sqrt
        # parameterisation is the whole point of §2.2's "we predict the square
        # root of the bounding box width and height", which equalises the
        # loss's sensitivity across box scales.
        cx = (box_preds[..., 0] + cols) * (W / S)
        cy = (box_preds[..., 1] + rows) * (H / S)
        w = (box_preds[..., 2] ** 2) * float(W)
        h = (box_preds[..., 3] ** 2) * float(H)
        return lucid.stack([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dim=-1)

    def _decode_predictions(
        self,
        raw: Tensor,
        image_size: tuple[int, int],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Decode raw grid output to flat predictions.

        Args:
            raw:        (B_batch, S, S, num_boxes*5+C) raw network output.
            image_size: (H, W) of the input image in pixels.

        Returns:
            (logits, pred_boxes):
                logits:     (B_batch, S*S*num_boxes, C) raw class scores.
                pred_boxes: (B_batch, S*S*num_boxes, 4) decoded xyxy pixel boxes.
        """
        cfg = self.config
        S = cfg.split_size
        B_boxes = cfg.num_boxes
        C = cfg.num_classes
        H, W = image_size
        batch = int(raw.shape[0])
        n_pred = S * S * B_boxes

        # layout: [box0_cx, box0_cy, box0_w, box0_h, box0_conf, box1_..., cls...]
        box_preds = raw[..., : B_boxes * 5].reshape(batch, S, S, B_boxes, 5)
        cls_preds = raw[..., B_boxes * 5 :]  # (batch, S, S, C)

        boxes = self._decode_boxes(box_preds, image_size)
        pred_boxes = lucid.stack(
            [
                boxes[..., 0].clamp(min=0.0, max=float(W)),
                boxes[..., 1].clamp(min=0.0, max=float(H)),
                boxes[..., 2].clamp(min=0.0, max=float(W)),
                boxes[..., 3].clamp(min=0.0, max=float(H)),
            ],
            dim=-1,
        ).reshape(batch, n_pred, 4)

        # Every box of a cell shares the cell's class scores.
        logits = cls_preds.unsqueeze(3).expand(batch, S, S, B_boxes, C)
        logits = logits.reshape(batch, n_pred, C)

        # Paper §2.1 Eq. 2: the class-specific confidence score is
        # Pr(class|object)·Pr(object)·IOU, so the per-box confidence channel
        # must multiply the shared per-cell class term.  Without it the two
        # boxes of a cell are indistinguishable and background cells are
        # never suppressed.  Confidence is a linear output too; clamping to
        # [0, 1] keeps the score-threshold and NMS comparisons meaningful
        # without imposing a squashing function the network was never
        # trained through.
        conf = box_preds[..., 4].clamp(min=0.0, max=1.0).reshape(batch, n_pred)

        return logits, pred_boxes, conf

    def _compute_loss(
        self,
        raw: Tensor,
        targets: list[dict[str, Tensor]],
        image_size: tuple[int, int],
    ) -> Tensor:
        """Compute the YOLOv1 multi-part MSE loss.

        Args:
            raw:        (B_batch, S, S, B*5+C) raw network output.
            targets:    List of B dicts with "boxes" (M,4) xyxy in [0,1]
                        and "labels" (M,) int.
            image_size: (H, W) in pixels.

        Returns:
            Scalar loss tensor.
        """
        cfg = self.config
        S = cfg.split_size
        B_boxes = cfg.num_boxes
        batch_size = int(raw.shape[0])

        box_preds = raw[..., : B_boxes * 5].reshape(batch_size, S, S, B_boxes, 5)
        # Matching IoUs are targets, so the boxes are decoded without gradient
        # — and with the same parameterisation the decoder and the loss use.
        boxes = self._decode_boxes(box_preds.detach(), image_size)

        loss_parts = [
            self._image_loss(raw[bi], boxes[bi], targets[bi], image_size)
            for bi in range(batch_size)
        ]
        return lucid.cat(loss_parts).mean()

    def _image_loss(
        self,
        raw: Tensor,
        boxes: Tensor,
        target: dict[str, Tensor],
        image_size: tuple[int, int],
    ) -> Tensor:
        """One image's loss over every predictor at once.

        ``raw`` is the image's ``(S, S, B*5+C)`` output and ``boxes`` its
        ``(S, S, B, 4)`` decoded boxes.  Assignment, IoU targets and masks
        are tensor ops, so nothing is read to the host.
        """
        cfg = self.config
        S = cfg.split_size
        B_boxes = cfg.num_boxes
        C = cfg.num_classes
        H, W = image_size
        dev, dt = raw.device, raw.dtype
        cell_w = W / S
        cell_h = H / S

        box_preds = raw[..., : B_boxes * 5].reshape(S * S * B_boxes, 5)
        cls_preds = raw[..., B_boxes * 5 :].reshape(S * S, C)
        pred_conf = box_preds[:, 4]

        scale = lucid.tensor([W, H, W, H], dtype=dt, device=dev)
        gt_xyxy = target["boxes"].to(device=dev, dtype=dt) * scale
        M = int(gt_xyxy.shape[0])
        if M == 0:
            return (cfg.lambda_noobj * (pred_conf**2).sum()).reshape(1)

        gt_cx = (gt_xyxy[:, 0] + gt_xyxy[:, 2]) / 2.0
        gt_cy = (gt_xyxy[:, 1] + gt_xyxy[:, 3]) / 2.0
        gt_w = gt_xyxy[:, 2] - gt_xyxy[:, 0]
        gt_h = gt_xyxy[:, 3] - gt_xyxy[:, 1]

        # Responsible cell holds the centre; within it, the predictor whose
        # current box overlaps the GT most is responsible (paper §2.2).
        col = (gt_cx / cell_w).floor().clamp(0.0, float(S - 1))
        row = (gt_cy / cell_h).floor().clamp(0.0, float(S - 1))
        cell = (row * float(S) + col).long()  # (M,)
        cell_boxes = boxes.reshape(S * S, B_boxes, 4)[cell]  # (M, B, 4)
        ious = lucid.stack(
            [box_iou(cell_boxes[:, b], gt_xyxy).diagonal() for b in range(B_boxes)],
            dim=1,
        )  # (M, B)
        best_iou, best_b = ious.max(dim=1)
        idx = cell * B_boxes + best_b  # (M,) predictor index

        # One object per predictor: when two GTs claim the same one, the later
        # wins — the slot is overwritten, as darknet's truth grid is.
        same = (idx[:, None] == idx[None, :]).to(dt)
        later = lucid.triu(lucid.ones((M, M), dtype=dt, device=dev), 1)
        won = ((same * later).sum(dim=1) == 0.0).to(dt)  # (M,)

        sel = box_preds[idx]  # (M, 5)

        # The head is linear, so the loss compares raw outputs against
        # cell-relative targets — the same parameterisation the decoder
        # inverts.  Squashing here and squaring there would train the network
        # for one convention and read it in another.
        tgt_cx = gt_cx / cell_w - col
        tgt_cy = gt_cy / cell_h - row
        loss_xy = (((sel[:, 0] - tgt_cx) ** 2 + (sel[:, 1] - tgt_cy) ** 2) * won).sum()

        # Paper Eq.3 is MSE on sqrt(w), sqrt(h) — and with ``[detection]
        # sqrt=1`` the network output *is* sqrt(w), so the prediction enters
        # the term directly rather than being rooted first.
        tgt_sw = (gt_w / float(W)).clamp(0.0, 1.0).sqrt()
        tgt_sh = (gt_h / float(H)).clamp(0.0, 1.0).sqrt()
        loss_wh = (((sel[:, 2] - tgt_sw) ** 2 + (sel[:, 3] - tgt_sh) ** 2) * won).sum()

        # Confidence loss (obj): MSE vs the matched IoU, linear.
        loss_conf_o = (((sel[:, 4] - best_iou) ** 2) * won).sum()

        # Class loss: MSE on class scores vs one-hot
        one_hot = F.one_hot(target["labels"].to(dev).long(), C).to(dt)
        cls_err = ((cls_preds[cell] - one_hot) ** 2).sum(dim=1)
        loss_cls = (cls_err * won).sum()

        # No-object: confidence should be 0 everywhere nothing is assigned.
        n_pred = S * S * B_boxes
        assigned = F.one_hot(idx, n_pred).to(dt).sum(dim=0).clamp(0.0, 1.0)
        loss_conf_n = ((pred_conf**2) * (1.0 - assigned)).sum()

        # YOLOv1 Eq. 3 is a SUM of squared errors over all cells and
        # predictors, not a per-group mean.  Averaging each group separately
        # renormalises the ~S²·B background confidence terms against the
        # handful of object terms, which is exactly what lambda_noobj exists
        # to control — so the mean form silently cancels it and makes the
        # balance drift with grid size and object count.
        img_loss: Tensor = (
            cfg.lambda_coord * (loss_xy + loss_wh)
            + loss_conf_o
            + cfg.lambda_noobj * loss_conf_n
            + loss_cls
        )
        return img_loss.reshape(1)

    @override
    def forward(
        self,
        x: Tensor,
        targets: list[dict[str, Tensor]] | None = None,
        image_size: tuple[int, int] = (448, 448),
    ) -> ObjectDetectionOutput:
        """Run YOLOv1 forward pass.

        Args:
            x:          (B, C, H, W) input image tensor.
            targets:    Optional list of B dicts for training loss computation.
                        Each dict has:
                        - ``"boxes"``:  (M_i, 4) xyxy boxes normalised to [0, 1].
                        - ``"labels"``: (M_i,) integer class indices.
            image_size: (H, W) of the input in pixels (used for box decoding).
                        Defaults to 448×448 (the original paper resolution).

        Returns:
            :class:`~lucid.models._output.ObjectDetectionOutput` with:
            - ``logits``:     (B, S·S·num_boxes, num_classes)
            - ``pred_boxes``: (B, S·S·num_boxes, 4) xyxy pixel boxes
            - ``loss``:       scalar tensor when targets are provided, else None
        """
        check_per_image(type(self).__name__, int(x.shape[0]), targets=targets)
        raw = self._forward_features(x)
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
        image_size: tuple[int, int] = (448, 448),
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
        # Results, NMS class ids and selection indices all live with the
        # predictions; built on the CPU they made NMS raise on a Metal model.
        dev = output.logits.device
        results: list[dict[str, Tensor]] = []

        for b in range(B_batch):
            lg_b = output.logits[b]  # (S*S*B, C)
            bx_b = output.pred_boxes[b]  # (S*S*B, 4)
            # Confidence scores: max class prob (no sigmoid — raw logits)
            # darknet's yolov1.cfg ends in a *linear* [connected] layer with
            # softmax=0: the class channels ARE the probabilities, fitted to
            # the one-hot by MSE and used directly at test time.  Squashing
            # them through a sigmoid here made every background class sit at
            # 0.5 and clear the threshold.  Multiply in the per-box confidence
            # (Eq. 2) so the score is Pr(class|object)·Pr(object)·IOU.
            sc_b = lg_b  # (S*S*B, C)
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

_CFG_V1 = YOLOConfig(
    num_classes=80,
    in_channels=3,
    split_size=7,
    num_boxes=2,
    lambda_coord=5.0,
    lambda_noobj=0.5,
    score_thresh=0.5,
    nms_thresh=0.5,
    tiny=False,
)

_CFG_V1_TINY = YOLOConfig(
    num_classes=80,
    in_channels=3,
    split_size=7,
    num_boxes=2,
    lambda_coord=5.0,
    lambda_noobj=0.5,
    score_thresh=0.5,
    nms_thresh=0.5,
    tiny=True,
)


@register_model(
    task="object-detection",
    family="yolo",
    model_type="yolo",
    model_class=YOLOForObjectDetection,
    default_config=_CFG_V1,
)
def yolo(
    pretrained: bool = False,
    **overrides: object,
) -> YOLOForObjectDetection:
    r"""YOLOv1 with full Darknet backbone (Redmon et al., CVPR 2016).

    Builds the paper-cited full YOLOv1 detector: 24-layer Darknet
    backbone, 7x7 grid, 2 boxes per cell, and 80 COCO classes by
    default (pass ``num_classes=20`` for the paper's PASCAL VOC model,
    whose 7x7x30 output tensor the paper describes).  ~270M parameters
    (most in the two 4096-dim FC layers) and 63.4% mAP on VOC 2007
    test (paper Table 1) at 45 fps.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No YOLOv1 checkpoint ships; ``True`` raises rather than
        returning a randomly initialised model.
    **overrides
        Keyword overrides forwarded into :class:`YOLOConfig` —
        ``num_classes``, ``split_size`` (S), ``num_boxes`` (B),
        ``score_thresh``, ``nms_thresh``, and loss weights.

    Returns
    -------
    YOLOForObjectDetection
        Detector with the full YOLOv1 configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See Redmon et al., "You Only Look Once: Unified, Real-Time Object
    Detection", CVPR 2016 (arXiv:1506.02640).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.yolo._v1 import yolo
    >>> model = yolo()
    >>> x = lucid.randn(1, 3, 448, 448)
    >>> out = model(x)
    >>> out.logits.shape[0]
    1
    """
    if pretrained:
        reject_unavailable_pretrained("yolo")
    return YOLOForObjectDetection(apply_overrides(_CFG_V1, overrides))


@register_model(
    task="object-detection",
    family="yolo",
    model_type="yolo",
    model_class=YOLOForObjectDetection,
    default_config=_CFG_V1_TINY,
)
def yolo_tiny(
    pretrained: bool = False,
    **overrides: object,
) -> YOLOForObjectDetection:
    r"""YOLOv1-Tiny with reduced Darknet backbone (Redmon et al., CVPR 2016).

    Builds the lightweight variant described in the YOLOv1 paper alongside
    the full model: 9-layer Darknet-Tiny backbone with the same 7x7 grid
    and 2 boxes per cell.  52.7% mAP on VOC 2007 at 155 fps (paper Table 1)
    — trades accuracy for speed.

    The backbone tapers to 256 channels before the head, as
    ``yolov1-tiny.cfg`` does; the paper gives no parameter count for Fast
    YOLO, saying only that it uses "fewer filters in those layers".

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Fast YOLO has no checkpoint here either, so ``True`` is refused
        instead of quietly building an untrained detector.
    **overrides
        Keyword overrides forwarded into :class:`YOLOConfig`.

    Returns
    -------
    YOLOForObjectDetection
        Detector with the YOLOv1-Tiny configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See Redmon et al., "You Only Look Once: Unified, Real-Time Object
    Detection", CVPR 2016 (arXiv:1506.02640).  The tiny variant uses
    fewer / shallower conv stages but keeps the same FC head topology.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.yolo._v1 import yolo_tiny
    >>> model = yolo_tiny()
    >>> x = lucid.randn(1, 3, 448, 448)
    >>> out = model(x)
    >>> out.logits.shape[0]
    1
    """
    if pretrained:
        reject_unavailable_pretrained("yolo_tiny")
    return YOLOForObjectDetection(apply_overrides(_CFG_V1_TINY, overrides))
