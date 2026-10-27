"""Detection-task utilities: box ops, NMS, anchors, RoI ops, and shared modules.

Structure
---------
§1  Box operations   — pure functions on (N, 4) xyxy Tensors
§2  NMS              — greedy non-maximum suppression
§3  Anchor generator — multi-scale, multi-ratio anchor boxes
§4  RoI operations   — RoI Align and RoI Pool
§5  Shared nn.Module — FPN, RPN, RoI head (reused across RCNN family)
"""

import math
from typing import cast, final, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._utils._common import host_floats, host_ints, host_rows

# ---------------------------------------------------------------------------
# §1  Box operations
# ---------------------------------------------------------------------------


def box_area(boxes: Tensor) -> Tensor:
    """Area of axis-aligned boxes given in xyxy format.

    Parameters
    ----------
    boxes : Tensor
        Shape ``(N, 4)``; each row is ``[x1, y1, x2, y2]``.

    Returns
    -------
    Tensor
        Shape ``(N,)``.  Element ``i`` is
        ``(x2_i - x1_i) * (y2_i - y1_i)``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._detection import box_area
    >>> boxes = lucid.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0]])
    >>> box_area(boxes).tolist()
    [100.0, 100.0]
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Pairwise IoU matrix between two sets of boxes (xyxy format).

    Parameters
    ----------
    boxes1 : Tensor
        Shape ``(N, 4)`` xyxy.
    boxes2 : Tensor
        Shape ``(M, 4)`` xyxy.

    Returns
    -------
    Tensor
        Shape ``(N, M)``.  Entry ``(i, j)`` is the intersection-over-union
        of ``boxes1[i]`` and ``boxes2[j]``, in ``[0, 1]``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._detection import box_iou
    >>> boxes = lucid.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0]])
    >>> [[round(v, 4) for v in row] for row in box_iou(boxes, boxes).tolist()]
    [[1.0, 0.1429], [0.1429, 1.0]]

    The two overlap on a 5x5 corner against a union of 175, which is the
    0.1429 off the diagonal.
    """
    area1 = box_area(boxes1)  # (N,)
    area2 = box_area(boxes2)  # (M,)

    # Broadcast for pairwise intersection: (N, 1, 4) vs (1, M, 4)
    b1 = boxes1[:, None, :]  # (N, 1, 4)
    b2 = boxes2[None, :, :]  # (1, M, 4)

    # Element-wise max/min to get intersection corners
    inter_x1: Tensor = lucid.maximum(b1[..., 0], b2[..., 0])  # (N, M)
    inter_y1: Tensor = lucid.maximum(b1[..., 1], b2[..., 1])
    inter_x2: Tensor = lucid.minimum(b1[..., 2], b2[..., 2])
    inter_y2: Tensor = lucid.minimum(b1[..., 3], b2[..., 3])

    inter_w = (inter_x2 - inter_x1).clamp(0.0, 1e9)
    inter_h = (inter_y2 - inter_y1).clamp(0.0, 1e9)
    inter_area = inter_w * inter_h  # (N, M)

    union = area1[:, None] + area2[None, :] - inter_area
    return inter_area / union.clamp(1e-6, 1e9)


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    r"""Pairwise Generalised IoU (GIoU) — same shape convention as ``box_iou``.

    .. math::

        \text{GIoU}(A, B) = \text{IoU}(A, B)
            - \frac{|C \setminus (A \cup B)|}{|C|},

    where :math:`C` is the smallest axis-aligned box enclosing both
    :math:`A` and :math:`B`.  Unlike IoU, GIoU is informative even when
    the two boxes do not overlap.

    Parameters
    ----------
    boxes1 : Tensor
        Shape ``(N, 4)`` xyxy.
    boxes2 : Tensor
        Shape ``(M, 4)`` xyxy.

    Returns
    -------
    Tensor
        Shape ``(N, M)`` GIoU matrix in :math:`[-1, 1]`.

    References
    ----------
    .. [1] Rezatofighi et al., *Generalized Intersection over Union: A
       Metric and A Loss for Bounding Box Regression*, CVPR 2019.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._detection import generalized_box_iou
    >>> boxes = lucid.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0]])
    >>> [[round(v, 4) for v in row] for row in generalized_box_iou(boxes, boxes).tolist()]
    [[1.0, -0.0794], [-0.0794, 1.0]]

    Unlike IoU this goes negative: the penalty for the empty space in the
    enclosing box is what gives a gradient to boxes that do not overlap
    at all, where plain IoU is flat at zero.
    """
    area1 = box_area(boxes1)  # (N,)
    area2 = box_area(boxes2)  # (M,)

    b1 = boxes1[:, None, :]  # (N, 1, 4)
    b2 = boxes2[None, :, :]  # (1, M, 4)

    # Intersection
    inter_x1: Tensor = lucid.maximum(b1[..., 0], b2[..., 0])
    inter_y1: Tensor = lucid.maximum(b1[..., 1], b2[..., 1])
    inter_x2: Tensor = lucid.minimum(b1[..., 2], b2[..., 2])
    inter_y2: Tensor = lucid.minimum(b1[..., 3], b2[..., 3])

    inter_w = (inter_x2 - inter_x1).clamp(0.0, 1e9)
    inter_h = (inter_y2 - inter_y1).clamp(0.0, 1e9)
    inter_area = inter_w * inter_h  # (N, M)

    union = area1[:, None] + area2[None, :] - inter_area

    # Enclosing box
    enc_x1: Tensor = lucid.minimum(b1[..., 0], b2[..., 0])
    enc_y1: Tensor = lucid.minimum(b1[..., 1], b2[..., 1])
    enc_x2: Tensor = lucid.maximum(b1[..., 2], b2[..., 2])
    enc_y2: Tensor = lucid.maximum(b1[..., 3], b2[..., 3])

    enc_area = (enc_x2 - enc_x1).clamp(0.0, 1e9) * (enc_y2 - enc_y1).clamp(0.0, 1e9)

    iou = inter_area / union.clamp(1e-6, 1e9)
    return iou - (enc_area - union) / enc_area.clamp(1e-6, 1e9)


def box_xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    """Convert ``(x1, y1, x2, y2)`` → ``(cx, cy, w, h)``.

    Parameters
    ----------
    boxes : Tensor
        Shape ``(..., 4)`` in xyxy format.

    Returns
    -------
    Tensor
        Shape ``(..., 4)`` in cxcywh format (centre / width / height).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._detection import box_xyxy_to_cxcywh
    >>> boxes = lucid.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0]])
    >>> box_xyxy_to_cxcywh(boxes).tolist()
    [[5.0, 5.0, 10.0, 10.0], [10.0, 10.0, 10.0, 10.0]]
    """
    x1 = boxes[..., 0:1]
    y1 = boxes[..., 1:2]
    x2 = boxes[..., 2:3]
    y2 = boxes[..., 3:4]
    return lucid.cat([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1], dim=-1)


def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    """Convert ``(cx, cy, w, h)`` → ``(x1, y1, x2, y2)``.

    Parameters
    ----------
    boxes : Tensor
        Shape ``(..., 4)`` in cxcywh format.

    Returns
    -------
    Tensor
        Shape ``(..., 4)`` in xyxy format.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._detection import (
    ...     box_cxcywh_to_xyxy,
    ...     box_xyxy_to_cxcywh,
    ... )
    >>> boxes = lucid.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0]])
    >>> box_cxcywh_to_xyxy(box_xyxy_to_cxcywh(boxes)).tolist()
    [[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0]]

    The round trip is exact, which is what lets a model predict in centre
    form and be scored in corner form.
    """
    cx = boxes[..., 0:1]
    cy = boxes[..., 1:2]
    w = boxes[..., 2:3]
    h = boxes[..., 3:4]
    return lucid.cat([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dim=-1)


def clip_boxes_to_image(boxes: Tensor, size: tuple[int, int]) -> Tensor:
    """Clip boxes to image boundaries.

    Parameters
    ----------
    boxes : Tensor
        Shape ``(N, 4)`` xyxy in pixel coordinates.
    size : tuple[int, int]
        ``(height, width)`` of the image.

    Returns
    -------
    Tensor
        Shape ``(N, 4)`` with every coordinate clamped into the image
        rectangle ``[0, width] × [0, height]``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._detection import clip_boxes_to_image
    >>> boxes = lucid.tensor([[-5.0, -5.0, 20.0, 20.0]])
    >>> clip_boxes_to_image(boxes, (10, 10)).tolist()
    [[0.0, 0.0, 10.0, 10.0]]
    """
    h, w = size
    x1 = boxes[:, 0:1].clamp(0.0, float(w))
    y1 = boxes[:, 1:2].clamp(0.0, float(h))
    x2 = boxes[:, 2:3].clamp(0.0, float(w))
    y2 = boxes[:, 3:4].clamp(0.0, float(h))
    return lucid.cat([x1, y1, x2, y2], dim=1)


def remove_small_boxes(boxes: Tensor, min_size: float) -> Tensor:
    """Return indices of boxes whose width AND height are ``>= min_size``.

    Parameters
    ----------
    boxes : Tensor
        Shape ``(N, 4)`` xyxy.
    min_size : float
        Minimum side length in pixels.

    Returns
    -------
    Tensor
        1-D ``int64`` index tensor of surviving box positions; empty when
        every box is below threshold.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._detection import remove_small_boxes
    >>> boxes = lucid.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 10.0, 10.0]])
    >>> remove_small_boxes(boxes, 2.0).tolist()
    [1]

    Indices of what survives, not the boxes themselves, so the caller can
    take the same rows out of the scores and labels beside them.
    """
    # One host read for every box's (w, h), not two per box.
    sizes = host_rows(boxes[:, 2:4] - boxes[:, 0:2])
    keep: list[int] = [i for i, (w, h) in enumerate(sizes) if w >= min_size and h >= min_size]
    if not keep:
        return lucid.zeros((0,), device=boxes.device.type).long()
    return lucid.tensor(keep, device=boxes.device.type).long()


def encode_boxes(
    reference_boxes: Tensor,
    proposals: Tensor,
    weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> Tensor:
    """Encode box regression targets ``(dx, dy, dw, dh)``.

    Inverse of :func:`decode_boxes`.  Used to turn ground-truth boxes
    into the per-anchor regression targets consumed by the RPN /
    detection head loss.

    Parameters
    ----------
    reference_boxes : Tensor
        Shape ``(N, 4)`` xyxy ground-truth boxes.
    proposals : Tensor
        Shape ``(N, 4)`` xyxy anchor / proposal boxes paired one-to-one
        with ``reference_boxes``.
    weights : tuple[float, float, float, float], optional
        Per-component scaling ``(wx, wy, ww, wh)``.  Default
        ``(1.0, 1.0, 1.0, 1.0)``; Faster R-CNN canonically uses
        ``(10, 10, 5, 5)`` to reweight centre vs size terms.

    Returns
    -------
    Tensor
        Shape ``(N, 4)`` regression targets.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._detection import encode_boxes
    >>> target = lucid.tensor([[0.0, 0.0, 10.0, 10.0]])
    >>> proposal = lucid.tensor([[1.0, 1.0, 11.0, 11.0]])
    >>> [round(v, 4) for v in encode_boxes(target, proposal).tolist()[0]]
    [-0.1, -0.1, 0.0, 0.0]

    The centre moved one tenth of a width and the size did not change, so
    the last two deltas are zero: they are log ratios, and a ratio of one
    is a log of nothing.
    """
    wx, wy, ww, wh = weights

    ref = box_xyxy_to_cxcywh(reference_boxes)
    pro = box_xyxy_to_cxcywh(proposals)

    dx = wx * (ref[:, 0] - pro[:, 0]) / pro[:, 2].clamp(1e-6, 1e9)
    dy = wy * (ref[:, 1] - pro[:, 1]) / pro[:, 3].clamp(1e-6, 1e9)
    dw = ww * lucid.log(ref[:, 2] / pro[:, 2].clamp(1e-6, 1e9))
    dh = wh * lucid.log(ref[:, 3] / pro[:, 3].clamp(1e-6, 1e9))

    return lucid.stack([dx, dy, dw, dh], dim=1)


def decode_boxes(
    deltas: Tensor,
    anchors: Tensor,
    weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    bbox_xform_clip: float = math.log(1000.0 / 16),
) -> Tensor:
    """Decode box regression deltas back to xyxy format.

    Inverse of :func:`encode_boxes`.

    Parameters
    ----------
    deltas : Tensor
        Shape ``(N, 4)`` regression outputs.
    anchors : Tensor
        Shape ``(N, 4)`` xyxy reference boxes.
    weights : tuple[float, float, float, float], optional
        Per-component scaling; must match the value used in
        :func:`encode_boxes`.  Default ``(1.0, 1.0, 1.0, 1.0)``.
    bbox_xform_clip : float, optional
        Clamps ``dw`` / ``dh`` to prevent ``exp`` overflow on extreme
        deltas.  Default ``log(1000 / 16)`` (Faster R-CNN canonical
        value).

    Returns
    -------
    Tensor
        Shape ``(N, 4)`` decoded boxes in xyxy format.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._detection import (
    ...     decode_boxes,
    ...     encode_boxes,
    ... )
    >>> target = lucid.tensor([[0.0, 0.0, 10.0, 10.0]])
    >>> proposal = lucid.tensor([[1.0, 1.0, 11.0, 11.0]])
    >>> deltas = encode_boxes(target, proposal)
    >>> [round(v, 3) for v in decode_boxes(deltas, proposal).tolist()[0]]
    [0.0, 0.0, 10.0, 10.0]

    The inverse of :func:`encode_boxes` against the same anchors, which
    is what turns a head's regression back into pixels.
    """
    wx, wy, ww, wh = weights

    anc = box_xyxy_to_cxcywh(anchors)
    acx = anc[:, 0]
    acy = anc[:, 1]
    aw = anc[:, 2]
    ah = anc[:, 3]

    dx = deltas[:, 0] / wx
    dy = deltas[:, 1] / wy
    dw = (deltas[:, 2] / ww).clamp(-1e9, bbox_xform_clip)
    dh = (deltas[:, 3] / wh).clamp(-1e9, bbox_xform_clip)

    pred_cx = dx * aw + acx
    pred_cy = dy * ah + acy
    pred_w = lucid.exp(dw) * aw
    pred_h = lucid.exp(dh) * ah

    x1 = pred_cx - pred_w / 2.0
    y1 = pred_cy - pred_h / 2.0
    x2 = pred_cx + pred_w / 2.0
    y2 = pred_cy + pred_h / 2.0

    return lucid.stack([x1, y1, x2, y2], dim=1)


# ---------------------------------------------------------------------------
# §2  Non-maximum suppression
# ---------------------------------------------------------------------------


def nms(
    boxes: Tensor,
    scores: Tensor,
    iou_threshold: float,
) -> Tensor:
    """Greedy NMS — returns indices of surviving boxes, sorted by score desc.

    Algorithm:
      1. Sort boxes by descending score (``argsort``).
      2. Threshold the whole IoU matrix of the sorted boxes on the device and
         pack each row into 32-bit words, so one transfer brings back
         ``N * N / 32`` integers.
      3. Walk the sorted boxes on the host with Python integers as bit sets:
         a box not yet suppressed is kept, and its row suppresses every box
         it overlaps too much.

    The greedy order, and so the result, is the classic one.  It used to
    fetch each kept box's IoU row one ``item()`` at a time — half a million
    host syncs for one Faster R-CNN forward at 64x64, 25 of its 26 seconds.

    Args:
        boxes:         (N, 4) xyxy float Tensor.
        scores:        (N,) confidence scores.
        iou_threshold: Suppress if IoU > this value.

    Returns:
        1-D int Tensor of surviving box indices (descending score order).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._detection import nms
    >>> boxes = lucid.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0]])
    >>> scores = lucid.tensor([0.9, 0.8])
    >>> nms(boxes, scores, 0.3).tolist()
    [0, 1]

    Both survive: they overlap at 0.14, under the 0.3 threshold. Raising
    the threshold keeps more, not fewer — it is the overlap a box is
    allowed to have with a better-scoring one before it is dropped.
    """
    N: int = int(boxes.shape[0])
    dev = boxes.device.type
    if N == 0:
        return lucid.zeros((0,), device=dev).long()

    order_t = lucid.argsort(-scores)  # (N,) int
    order = cast(list[int], order_t.tolist())
    ranked = boxes[order_t]
    over = (box_iou(ranked, ranked) > iou_threshold).long()  # (N, N)

    words = (N + 31) // 32
    if words * 32 != N:
        over = F.pad(over, (0, words * 32 - N))
    bits = lucid.tensor([1 << b for b in range(32)], device=dev).long()
    packed = cast(list[list[int]], (over.reshape(N, words, 32) * bits).sum(dim=-1).tolist())

    suppressed = 0
    keep: list[int] = []
    for i in range(N):
        if (suppressed >> i) & 1:
            continue
        keep.append(order[i])
        row = packed[i]
        # Only boxes after this one can still be suppressed.
        for w in range(i // 32, words):
            if row[w]:
                suppressed |= row[w] << (32 * w)

    return lucid.tensor(keep, device=dev).long()


def batched_nms(
    boxes: Tensor,
    scores: Tensor,
    idxs: Tensor,
    iou_threshold: float,
) -> Tensor:
    """NMS applied independently per class via the class-offset trick.

    Boxes from different classes are offset by a large class-dependent
    value, preventing cross-class suppression in a single NMS pass.

    Args:
        boxes:         (N, 4) xyxy.
        scores:        (N,) confidence scores.
        idxs:          (N,) integer class index per box.
        iou_threshold: IoU threshold.

    Returns:
        Surviving box indices (sorted by score, descending).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._detection import batched_nms
    >>> boxes = lucid.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0]])
    >>> scores = lucid.tensor([0.9, 0.8])
    >>> batched_nms(boxes, scores, lucid.tensor([0, 1]), 0.3).tolist()
    [0, 1]

    The class indices keep the suppression inside each class, so a person
    and a car that overlap never suppress one another.
    """
    max_coord = float(boxes.max().item())
    offsets = idxs.float() * (max_coord + 1.0)
    boxes_for_nms = boxes + offsets[:, None]
    return nms(boxes_for_nms, scores, iou_threshold)


# ---------------------------------------------------------------------------
# §3  Anchor generator
# ---------------------------------------------------------------------------


class AnchorGenerator(nn.Module):
    """Generate multi-scale, multi-ratio anchors for each FPN level.

    For each (feature map, stride) pair the generator produces anchors
    centred at each spatial cell.

    Args:
        sizes:         Anchor sizes (sqrt of area) per FPN level,
                       e.g. ``((32,), (64,), (128,), (256,), (512,))``.
        aspect_ratios: Width/height ratios per FPN level,
                       e.g. ``((0.5, 1.0, 2.0),) * 5``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._detection import AnchorGenerator
    >>> generator = AnchorGenerator(sizes=((32,),), aspect_ratios=((1.0,),))
    >>> anchors = generator([lucid.randn(1, 4, 8, 8)], (64, 64), [(8, 8)])
    >>> anchors[0].shape
    (64, 4)

    One anchor per cell of an 8x8 map is 64 of them, in corner form and
    in image coordinates — the stride is what carries them there, which
    is why it is passed per level rather than inferred.
    """

    def __init__(
        self,
        sizes: tuple[tuple[int, ...], ...] = ((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios: tuple[tuple[float, ...], ...] = ((0.5, 1.0, 2.0),) * 5,
    ) -> None:
        super().__init__()
        assert len(sizes) == len(
            aspect_ratios
        ), "sizes and aspect_ratios must have the same number of levels"
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        # Plain numbers, not tensors: each forward builds them on the feature
        # maps' device.  Held as CPU tensors and moved there, they made every
        # compiled detector step a mixed-device trace, which runs eager.
        self._cell_anchors: list[list[list[float]]] = self._compute_cell_anchors()

    def _compute_cell_anchors(self) -> list[list[list[float]]]:
        """Base anchors (centred at origin) for every FPN level, as xyxy rows."""
        all_anchors: list[list[list[float]]] = []
        for level_sizes, level_ratios in zip(self.sizes, self.aspect_ratios, strict=True):
            anchors: list[list[float]] = []
            for size in level_sizes:
                area = float(size * size)
                for ratio in level_ratios:
                    w = math.sqrt(area / ratio)
                    h = w * ratio
                    anchors.append([-w / 2.0, -h / 2.0, w / 2.0, h / 2.0])
            all_anchors.append(anchors)
        return all_anchors

    def _grid_anchors(
        self,
        feature_map_size: tuple[int, int],
        stride: tuple[int, int],
        base_anchors: list[list[float]],
        device: str = "cpu",
    ) -> Tensor:
        """Tile base_anchors across a feature map grid.

        Args:
            feature_map_size: (H, W) of the feature map.
            stride:           (stride_h, stride_w) pixels per cell.
            base_anchors:     A rows of xyxy base anchors centred at origin.
            device:           Device for the generated anchor tensor.

        Returns:
            (H × W × A, 4) anchors in xyxy image-pixel coordinates.
        """
        fH, fW = feature_map_size
        sH, sW = stride

        # Build shift table: (fH*fW, 4) — (cx, cy, cx, cy)
        shifts: list[list[float]] = [
            [(c + 0.5) * sW, (r + 0.5) * sH, (c + 0.5) * sW, (r + 0.5) * sH]
            for r in range(fH)
            for c in range(fW)
        ]
        shifts_t = lucid.tensor(shifts, device=device)  # (G, 4)
        base_on_dev = lucid.tensor(base_anchors, device=device)  # (A, 4)

        G = fH * fW
        A = len(base_anchors)

        # (G, 1, 4) + (1, A, 4) → (G, A, 4) → (G*A, 4)
        grid = shifts_t[:, None, :] + base_on_dev[None, :, :]
        return grid.reshape(G * A, 4)

    @override
    def forward(
        self,
        feature_maps: list[Tensor],
        image_size: tuple[int, int],
        strides: list[tuple[int, int]],
    ) -> list[Tensor]:
        """Generate anchors for all FPN levels.

        Args:
            feature_maps: One (B, C, H, W) tensor per FPN level.
            image_size:   (H, W) of the input image (unused here, kept for
                          API symmetry with clip_boxes callers).
            strides:      (stride_h, stride_w) per level.

        Returns:
            List of (H_l × W_l × A_l, 4) anchor tensors, one per level.
        """
        assert len(feature_maps) == len(self._cell_anchors)
        # Device and dtype follow the feature maps, so anchors stay co-located
        # with — and as wide as — the predictions they are matched against.
        device = feature_maps[0].device.type if feature_maps else "cpu"
        all_anchors: list[Tensor] = []
        for feat, base, stride in zip(feature_maps, self._cell_anchors, strides, strict=True):
            fH = int(feat.shape[2])
            fW = int(feat.shape[3])
            all_anchors.append(
                self._grid_anchors((fH, fW), stride, base, device=device).to(feat.dtype)
            )
        return all_anchors


# ---------------------------------------------------------------------------
# §4  RoI operations
# ---------------------------------------------------------------------------


def roi_align(
    input: Tensor,
    boxes: list[Tensor],
    output_size: int | tuple[int, int],
    spatial_scale: float = 1.0,
    sampling_ratio: int = -1,
    aligned: bool = True,
) -> Tensor:
    """RoI Align — bilinear sub-pixel sampling into fixed-size crops.

    Reproduces the reference RoIAlign exactly, including its
    ``bilinear_interpolate`` boundary rule: a sample whose coordinate falls
    in ``[-1, 0]`` or ``[side-1, side]`` interpolates against an implicit
    zero outside the map (rather than snapping to the edge pixel), and a
    sample outside ``[-1, side]`` contributes zero.  When ``aligned`` is
    False the per-RoI side is additionally floored at 1 pixel — the
    reference legacy behaviour used by ``MultiScaleRoIAlign``.

    Sampling uses a vectorised gather of the four bilinear corners per
    sample point (no :func:`grid_sample`, whose ``align_corners`` /
    normalised mapping does not match the reference high-boundary rule).

    Args:
        input:        (B, C, H, W) feature map.
        boxes:        List of B tensors, each (N_i, 4) xyxy in *image*
                      pixel coordinates.
        output_size:  Height and width of each RoI crop.
        spatial_scale: Ratio of feature-map size to input image size
                       (e.g. 1/32 for a stride-32 backbone level).
        sampling_ratio: Sub-bin samples per side (fixed when > 0, else
                       ``ceil(roi_side / out_side)`` per box).
        aligned:      When True, apply the 0.5-pixel alignment offset.

    Returns:
        (sum(N_i), C, out_h, out_w) stacked crops.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._detection import roi_align
    >>> features = lucid.randn(1, 4, 16, 16)
    >>> boxes = [lucid.tensor([[0.0, 0.0, 8.0, 8.0]])]
    >>> roi_align(features, boxes, output_size=(2, 2), spatial_scale=1.0).shape
    (1, 4, 2, 2)

    One row per box across the whole list, not per image, so a batch with
    differing box counts still comes back as a single tensor.
    """
    if isinstance(output_size, int):
        out_h = out_w = output_size
    else:
        out_h, out_w = output_size

    feat_H = int(input.shape[2])
    feat_W = int(input.shape[3])
    C = int(input.shape[1])
    dev, dt = input.device.type, input.dtype

    offset = 0.5 if aligned else 0.0
    results: list[Tensor] = []

    for b_idx, roi_boxes in enumerate(boxes):
        N = int(roi_boxes.shape[0])
        if N == 0:
            continue

        feat_flat: Tensor = input[b_idx].reshape(C, feat_H * feat_W)  # (C, H*W)

        # Sample positions are host data: every box's corners in one read.
        corners = host_rows(roi_boxes[:, :4] * spatial_scale - offset)

        for rx1, ry1, rx2, ry2 in corners:
            roi_w = rx2 - rx1
            roi_h = ry2 - ry1
            if not aligned:
                roi_w = max(roi_w, 1.0)
                roi_h = max(roi_h, 1.0)
            bw = roi_w / out_w
            bh = roi_h / out_h

            if sampling_ratio > 0:
                ry = rx = sampling_ratio
            else:
                ry = max(1, math.ceil(roi_h / out_h))
                rx = max(1, math.ceil(roi_w / out_w))

            # Sub-bin sample centres (out_h*ry rows, out_w*rx cols).
            xs = [rx1 + j * bw + (ix + 0.5) * bw / rx for j in range(out_w) for ix in range(rx)]
            ys = [ry1 + i * bh + (iy + 0.5) * bh / ry for i in range(out_h) for iy in range(ry)]
            yl, yh, fy, zy = _bilinear_prep(ys, feat_H)
            xl, xh, fx, zx = _bilinear_prep(xs, feat_W)
            ny = len(ys)
            nx = len(xs)

            # Gather the four bilinear corners (broadcast row/col indices).
            zeros_ny_nx: Tensor = lucid.zeros((ny, nx), device=dev).long()
            yl_b = lucid.tensor([[v] for v in yl], device=dev).long() + zeros_ny_nx
            yh_b = lucid.tensor([[v] for v in yh], device=dev).long() + zeros_ny_nx
            xl_b = lucid.tensor([xl], device=dev).long() + zeros_ny_nx
            xh_b = lucid.tensor([xh], device=dev).long() + zeros_ny_nx

            v1 = _gather_corner(feat_flat, yl_b, xl_b, C, ny, nx, feat_W)
            v2 = _gather_corner(feat_flat, yl_b, xh_b, C, ny, nx, feat_W)
            v3 = _gather_corner(feat_flat, yh_b, xl_b, C, ny, nx, feat_W)
            v4 = _gather_corner(feat_flat, yh_b, xh_b, C, ny, nx, feat_W)

            fy_t = lucid.tensor([[v] for v in fy], dtype=dt, device=dev)  # (ny, 1)
            fx_t = lucid.tensor([fx], dtype=dt, device=dev)  # (1, nx)
            hy = 1.0 - fy_t
            hx = 1.0 - fx_t
            w1 = hy * hx
            w2 = hy * fx_t
            w3 = fy_t * hx
            w4 = fy_t * fx_t
            zy_t = lucid.tensor([[v] for v in zy], dtype=dt, device=dev)
            zmask = zy_t * lucid.tensor([zx], dtype=dt, device=dev)  # 0 off [-1, side]

            samp = (v1 * w1 + v2 * w2 + v3 * w3 + v4 * w4) * zmask  # (C, ny, nx)
            pooled = samp.reshape(C, out_h, ry, out_w, rx).mean(dim=(2, 4))
            results.append(pooled.unsqueeze(0))

    if not results:
        return input.new_zeros(0, C, out_h, out_w)
    return lucid.cat(results, dim=0)


def _bilinear_prep(
    coords: list[float], size: int
) -> tuple[list[int], list[int], list[float], list[float]]:
    """Reference ``bilinear_interpolate`` index / weight prep for one axis.

    For each continuous coordinate returns ``(low_idx, high_idx, frac,
    zero)`` where ``frac`` is the fractional weight toward the high
    neighbour and ``zero`` is 0 when the sample lies outside ``[-1, size]``
    (the reference zeroes those) else 1.  Coordinates in ``[-1, 0]`` clamp
    to 0; coordinates in ``[size-1, size]`` snap the neighbour pair to the
    last pixel so they interpolate toward the implicit zero via ``zero``.
    """
    low: list[int] = []
    high: list[int] = []
    frac: list[float] = []
    zero: list[float] = []
    for c in coords:
        if c < -1.0 or c > float(size):
            low.append(0)
            high.append(0)
            frac.append(0.0)
            zero.append(0.0)
            continue
        cc = 0.0 if c <= 0.0 else c
        lo = int(cc)
        if lo >= size - 1:
            lo = size - 1
            hi = size - 1
            cc = float(lo)
        else:
            hi = lo + 1
        low.append(lo)
        high.append(hi)
        frac.append(cc - lo)
        zero.append(1.0)
    return low, high, frac, zero


def _gather_corner(
    feat_flat: Tensor, yi: Tensor, xi: Tensor, C: int, ny: int, nx: int, fw: int
) -> Tensor:
    """Gather feature values at integer ``(yi, xi)`` → ``(C, ny, nx)``."""
    flat = (yi * fw + xi).reshape(-1)  # (ny*nx,)
    return feat_flat[:, flat].reshape(C, ny, nx)


def paste_masks_in_image(
    masks: Tensor,
    boxes: Tensor,
    image_size: tuple[int, int],
    threshold: float = 0.5,
) -> Tensor:
    """Resize each RoI mask onto its box and paste it into a full-image canvas.

    A mask head predicts on a fixed ``m x m`` grid in *RoI* coordinates.  That
    grid means nothing to a consumer until it is stretched back onto the box
    it came from and placed in the image — which is what the reference's
    ``paste_masks_in_image`` does, and what makes the configured binarisation
    threshold meaningful.

    Args:
        masks:      ``(D, 1, m, m)`` sigmoid probabilities in RoI space.
        boxes:      ``(D, 4)`` xyxy boxes in image coordinates.
        image_size: ``(H, W)`` of the image to paste into.
        threshold:  Probabilities strictly above this become 1.

    Returns:
        ``(D, 1, H, W)`` binary masks.  A degenerate box yields an all-zero
        plane rather than an error.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._detection import paste_masks_in_image
    >>> masks = lucid.zeros(1, 1, 28, 28)
    >>> boxes = lucid.tensor([[0.0, 0.0, 10.0, 10.0]])
    >>> paste_masks_in_image(masks, boxes, (32, 32)).shape
    (1, 1, 32, 32)

    A mask head predicts at a fixed small resolution; this puts each one
    back where its box was, at the image's size.
    """
    H, W = image_size
    D = int(masks.shape[0])
    if D == 0:
        return lucid.zeros((0, 1, H, W), device=masks.device.type)

    planes: list[Tensor] = []
    for i, (bx1, by1, bx2, by2) in enumerate(host_rows(boxes[:, :4])):
        x1, y1 = int(math.floor(bx1)), int(math.floor(by1))
        x2, y2 = int(math.ceil(bx2)), int(math.ceil(by2))
        # Clamp to the canvas; the reference keeps a minimum extent of 1.
        x1, y1 = max(0, min(x1, W - 1)), max(0, min(y1, H - 1))
        x2, y2 = max(x1 + 1, min(x2, W)), max(y1 + 1, min(y2, H))

        resized: Tensor = F.interpolate(
            masks[i : i + 1],
            size=(y2 - y1, x2 - x1),
            mode="bilinear",
            align_corners=False,
        )
        binary: Tensor = (resized > threshold).float()  # (1, 1, bh, bw)
        planes.append(F.pad(binary, (x1, W - x2, y1, H - y2)))

    return lucid.cat(planes, dim=0)


def roi_pool(
    input: Tensor,
    boxes: list[Tensor],
    output_size: int | tuple[int, int],
    spatial_scale: float = 1.0,
) -> Tensor:
    """RoI Pool (max-pool variant used in R-CNN / Fast R-CNN).

    Quantises RoI boundaries to integer pixels then adaptively
    **max**-pools each bin to ``output_size``.  Max — not average — is
    the operation Fast R-CNN §2.1 defines: "RoI max pooling works by
    dividing the h x w RoI window into an H x W grid of sub-windows ...
    and max-pooling the values in each sub-window into the corresponding
    output grid cell".

    Args:
        input:        (B, C, H, W) feature map.
        boxes:        List of B tensors, each (N_i, 4) xyxy image coords.
        output_size:  (out_h, out_w) of each crop.
        spatial_scale: Feature-map to image scale ratio.

    Returns:
        (sum(N_i), C, out_h, out_w).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._detection import roi_pool
    >>> features = lucid.randn(1, 4, 16, 16)
    >>> boxes = [lucid.tensor([[0.0, 0.0, 8.0, 8.0]])]
    >>> roi_pool(features, boxes, output_size=(2, 2), spatial_scale=1.0).shape
    (1, 4, 2, 2)

    The same shape :func:`roi_align` gives, reached by max-pooling
    quantised bins instead of sampling — cheaper, and it loses the
    sub-pixel alignment that align exists for.
    """
    if isinstance(output_size, int):
        out_h = out_w = output_size
    else:
        out_h, out_w = output_size

    feat_H = int(input.shape[2])
    feat_W = int(input.shape[3])
    C = int(input.shape[1])

    pool = nn.AdaptiveMaxPool2d((out_h, out_w))
    results: list[Tensor] = []

    for b_idx, roi_boxes in enumerate(boxes):
        N = int(roi_boxes.shape[0])
        if N == 0:
            continue
        feat: Tensor = input[b_idx]  # (C, H, W)

        for bx1, by1, bx2, by2 in host_rows(roi_boxes[:, :4]):
            x1 = int(round(bx1 * spatial_scale))
            y1 = int(round(by1 * spatial_scale))
            x2 = int(round(bx2 * spatial_scale))
            y2 = int(round(by2 * spatial_scale))

            # Both the Caffe original and reference_vision treat the rounded end
            # coordinate as *inclusive*:
            #     roi_width := max(roi_end_w - roi_start_w + 1, 1)
            # Python slicing is exclusive, so the window was one column and one
            # row short — a RoI rounding to columns 2..5 pooled 2,3,4 instead of
            # 2,3,4,5.  Add the +1 back when converting to a slice bound.
            x1 = max(0, min(x1, feat_W - 1))
            y1 = max(0, min(y1, feat_H - 1))
            x2 = max(x1 + 1, min(x2 + 1, feat_W))
            y2 = max(y1 + 1, min(y2 + 1, feat_H))

            crop: Tensor = feat[:, y1:y2, x1:x2]  # (C, rH, rW)
            pooled = pool(crop.unsqueeze(0)).squeeze(0)
            results.append(pooled.unsqueeze(0))

    if not results:
        return input.new_zeros(0, C, out_h, out_w)
    return lucid.cat(results, dim=0)


def multi_scale_deformable_attention(
    value: Tensor,
    value_spatial_shapes: list[tuple[int, int]],
    sampling_locations: Tensor,
    attention_weights: Tensor,
) -> Tensor:
    """Multi-scale deformable attention (Deformable DETR / Mask2Former).

    A composite over :func:`~lucid.nn.functional.grid_sample` (bilinear,
    ``align_corners=False``): each query head samples ``num_levels x
    num_points`` learned locations across the multi-scale feature maps and
    aggregates them with the predicted attention weights.  Reproduces the
    reference pure-tensor implementation exactly (no custom kernel needed).

    Parameters
    ----------
    value : Tensor
        ``(bs, sum(H_l * W_l), num_heads, head_dim)`` flattened multi-scale
        features.
    value_spatial_shapes : list of (int, int)
        ``(H_l, W_l)`` per level, in the same order ``value`` is concatenated.
    sampling_locations : Tensor
        ``(bs, num_queries, num_heads, num_levels, num_points, 2)`` in ``[0, 1]``
        normalised ``(x, y)`` coordinates.
    attention_weights : Tensor
        ``(bs, num_queries, num_heads, num_levels, num_points)`` — softmaxed
        over the flattened ``num_levels * num_points`` axis upstream.

    Returns
    -------
    Tensor
        ``(bs, num_queries, num_heads * head_dim)`` attended features.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._detection import (
    ...     multi_scale_deformable_attention,
    ... )
    >>> value = lucid.zeros(1, 64, 2, 8)
    >>> locations = lucid.zeros(1, 4, 2, 1, 4, 2)
    >>> weights = lucid.zeros(1, 4, 2, 1, 4)
    >>> multi_scale_deformable_attention(
    ...     value, [(8, 8)], locations, weights
    ... ).shape
    (1, 4, 16)

    Four queries attend to four sampled points each rather than to all 64
    positions — which is what makes deformable attention affordable
    across a feature pyramid, and why the locations carry a trailing 2
    for (x, y).
    """
    bs = int(value.shape[0])
    num_heads = int(value.shape[2])
    head_dim = int(value.shape[3])
    num_queries = int(sampling_locations.shape[1])
    num_levels = int(sampling_locations.shape[3])
    num_points = int(sampling_locations.shape[4])

    # grid_sample wants [-1, 1] coords; the reference uses align_corners=False.
    sampling_grids = 2.0 * sampling_locations - 1.0
    sampled: list[Tensor] = []
    offset = 0
    for level, (h, w) in enumerate(value_spatial_shapes):
        # (bs, H*W, num_heads, head_dim) → (bs*num_heads, head_dim, H, W)
        value_l = (
            value[:, offset : offset + h * w]
            .reshape(bs, h * w, num_heads * head_dim)
            .permute(0, 2, 1)
            .reshape(bs * num_heads, head_dim, h, w)
        )
        offset += h * w
        # (bs, num_queries, num_heads, num_points, 2) → (bs*num_heads, num_queries, num_points, 2)
        grid_l = (
            sampling_grids[:, :, :, level]
            .permute(0, 2, 1, 3, 4)
            .reshape(bs * num_heads, num_queries, num_points, 2)
        )
        sampled.append(
            F.grid_sample(
                value_l,
                grid_l,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
        )  # (bs*num_heads, head_dim, num_queries, num_points)

    # Shape — (bs*num_heads, head_dim, num_queries, num_levels*num_points)
    stacked = lucid.stack(sampled, dim=-2).reshape(
        bs * num_heads, head_dim, num_queries, num_levels * num_points
    )
    weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    out = (stacked * weights).sum(dim=-1)  # (bs*num_heads, head_dim, num_queries)
    return out.reshape(bs, num_heads * head_dim, num_queries).permute(0, 2, 1)


# ---------------------------------------------------------------------------
# §5  Shared nn.Module components
# ---------------------------------------------------------------------------


class FPN(nn.Module):
    """Feature Pyramid Network (Lin et al., 2017).

    Merges multi-scale backbone feature maps into a unified pyramid of
    semantically rich, spatially precise levels.

    Architecture per level:
      lateral  : Conv2d(in_ch, out_ch, 1)
      output   : Conv2d(out_ch, out_ch, 3, padding=1)
      top-down : upsample(2×, nearest) + element-wise add

    Args:
        in_channels:  Channel counts of each bottom-up map, finest first
                      (e.g. [256, 512, 1024, 2048] for ResNet C2–C5).
        out_channels: Unified channel count for all pyramid levels.
        extra_blocks: Additional coarser levels appended via 3×3 stride-2
                      conv on the coarsest FPN output (default: 1 → P6).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._detection import FPN
    >>> neck = FPN([4, 8], 16).eval()
    >>> features = [lucid.randn(1, 4, 16, 16), lucid.randn(1, 8, 8, 8)]
    >>> [tuple(t.shape) for t in neck(features)]
    [(1, 16, 16, 16), (1, 16, 8, 8), (1, 16, 4, 4)]

    Two levels in and three out: ``extra_blocks`` adds a coarser level on
    top by striding, which is where the largest objects get detected.
    Every output carries the same channel count, so one head can be
    shared across all of them.
    """

    def __init__(
        self,
        in_channels: list[int],
        out_channels: int,
        extra_blocks: int = 1,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        n = len(in_channels)

        self.lateral_convs = nn.ModuleList([nn.Conv2d(ic, out_channels, 1) for ic in in_channels])
        self.output_convs = nn.ModuleList(
            [nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in range(n)]
        )
        self.extra_convs = nn.ModuleList(
            [
                nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
                for _ in range(extra_blocks)
            ]
        )

    @override
    def forward(self, features: list[Tensor]) -> list[Tensor]:
        """
        Args:
            features: Bottom-up maps, finest → coarsest (e.g. C2, C3, C4, C5).

        Returns:
            FPN outputs, finest → coarsest (e.g. P2, P3, P4, P5, P6).
        """
        # Lateral projections
        laterals: list[Tensor] = [
            lat(f) for lat, f in zip(self.lateral_convs, features, strict=True)
        ]

        # Top-down: merge from coarsest to finest
        n = len(laterals)
        for i in range(n - 2, -1, -1):
            up = F.interpolate(laterals[i + 1], scale_factor=2.0, mode="nearest")
            laterals[i] = laterals[i] + up

        # 3×3 output convolutions (anti-aliasing)
        outs: list[Tensor] = [
            conv(lat) for conv, lat in zip(self.output_convs, laterals, strict=True)
        ]

        # Extra coarser levels
        extra_in = outs[-1]
        for conv in self.extra_convs:
            extra_in = F.relu(conv(extra_in))
            outs.append(extra_in)

        return outs


class RPN(nn.Module):
    """Region Proposal Network (Ren et al., 2015).

    Slides a 3×3 conv over each FPN level to predict per-anchor:
      - objectness score  (foreground vs background)
      - box delta         (dx, dy, dw, dh)

    Proposals from all levels are merged, clipped, filtered and NMS'd
    per image to produce the final region proposals.

    Args:
        in_channels:    FPN output channels.
        num_anchors:    Anchors per spatial cell (len(sizes) × len(ratios)).
        pre_nms_top_n:  Proposals kept per level before NMS.
        post_nms_top_n: Proposals kept per image after NMS.
        nms_threshold:  IoU threshold for NMS.
        min_size:       Minimum proposal side length (pixels).
        score_thresh:   Minimum objectness score (post-sigmoid).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._detection import RPN
    >>> rpn = RPN(16, 3).eval()
    >>> features = [lucid.randn(1, 16, 8, 8)]
    >>> anchors = [lucid.randn(192, 4)]
    >>> proposals, scores = rpn(features, anchors, (64, 64))
    >>> len(proposals), len(scores)
    (1, 1)

    Proposals per image, not per level: the head runs on every pyramid
    level and the results are merged, ranked and cut down by NMS before
    they come back, which is why the count does not follow the features.
    """

    def __init__(
        self,
        in_channels: int,
        num_anchors: int,
        pre_nms_top_n: int = 2000,
        post_nms_top_n: int = 1000,
        nms_threshold: float = 0.7,
        min_size: float = 1.0,
        score_thresh: float = 0.0,
    ) -> None:
        super().__init__()
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_threshold = nms_threshold
        self.min_size = min_size
        self.score_thresh = score_thresh

        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, 1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, 1)

    @override
    def forward(
        self,
        features: list[Tensor],
        anchors: list[Tensor],
        image_size: tuple[int, int],
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Run RPN over all FPN levels and return proposals per image.

        Args:
            features:   FPN outputs, each (B, C, H_l, W_l).
            anchors:    Per-level anchor tensors from AnchorGenerator.
            image_size: (H, W) of the input image.

        Returns:
            (proposals, scores):
                proposals[b]: (K_b, 4) xyxy proposals for image b.
                scores[b]:    (K_b,)   objectness probabilities.
        """
        B = int(features[0].shape[0])
        all_proposals: list[list[Tensor]] = [[] for _ in range(B)]
        all_scores: list[list[Tensor]] = [[] for _ in range(B)]

        for feat, level_anchors in zip(features, anchors, strict=True):
            t = F.relu(self.conv(feat))
            logits = self.cls_logits(t)  # (B, A, H, W)
            deltas = self.bbox_pred(t)  # (B, 4A, H, W)

            A = int(logits.shape[1])
            fH = int(logits.shape[2])
            fW = int(logits.shape[3])

            # Spatial-major flatten to match AnchorGenerator ordering (G*A, 4)
            # logits: (B, A, H, W) → permute(0,2,3,1) → (B,H,W,A) → (B, H*W*A)
            # deltas: (B,4A,H,W) → reshape(B,A,4,H,W) → permute(0,3,4,1,2)
            #         → (B,H,W,A,4) → (B, H*W*A, 4)
            scores_flat = F.sigmoid(logits.permute(0, 2, 3, 1).reshape(B, -1))
            deltas_flat = (
                deltas.reshape(B, A, 4, fH, fW).permute(0, 3, 4, 1, 2).reshape(B, fH * fW * A, 4)
            )

            for b in range(B):
                sc = scores_flat[b]  # (N_anc,)
                dl = deltas_flat[b]  # (N_anc, 4)

                K = min(self.pre_nms_top_n, int(sc.shape[0]))
                # argsort ascending on negated scores → top-K indices
                topk_idx = lucid.argsort(-sc)[:K]

                topk_sc = sc[topk_idx]
                topk_dl = dl[topk_idx]
                topk_anc = level_anchors[topk_idx]

                props = decode_boxes(topk_dl, topk_anc)
                props = clip_boxes_to_image(props, image_size)

                keep_small = remove_small_boxes(props, self.min_size)
                if int(keep_small.shape[0]) == 0:
                    continue

                props = props[keep_small]
                topk_sc = topk_sc[keep_small]

                score_mask: list[int] = [
                    i for i, score in enumerate(host_floats(topk_sc)) if score >= self.score_thresh
                ]
                if not score_mask:
                    continue

                mask_t = lucid.tensor(score_mask, device=props.device.type).long()
                props = props[mask_t]
                topk_sc = topk_sc[mask_t]

                all_proposals[b].append(props)
                all_scores[b].append(topk_sc)

        final_proposals: list[Tensor] = []
        final_scores: list[Tensor] = []

        for b in range(B):
            if not all_proposals[b]:
                final_proposals.append(features[0].new_zeros(0, 4))
                final_scores.append(features[0].new_zeros(0))
                continue

            props_b = lucid.cat(all_proposals[b], dim=0)
            sc_b = lucid.cat(all_scores[b], dim=0)

            keep = nms(props_b, sc_b, self.nms_threshold)
            K2 = min(self.post_nms_top_n, int(keep.shape[0]))
            keep = keep[:K2]

            final_proposals.append(props_b[keep])
            final_scores.append(sc_b[keep])

        return final_proposals, final_scores


class RoIHead(nn.Module):
    """Two-FC RoI head shared by Fast R-CNN, Faster R-CNN and Mask R-CNN.

    Takes RoI-aligned crops and predicts class logits and box deltas.

    Args:
        in_channels:         Channels of each RoI crop.
        roi_size:            (H, W) of the RoI Align output crop.
        num_classes:         Foreground classes (background adds +1).
        representation_size: Hidden size of the two FC layers.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._detection import RoIHead
    >>> head = RoIHead(16, (2, 2), 5).eval()
    >>> logits, deltas = head(lucid.randn(2, 16, 2, 2))
    >>> logits.shape, deltas.shape
    ((2, 6), (2, 20))

    Six classes for five plus background, and twenty deltas because the
    box is regressed per class — four coordinates for each of the five,
    with background getting none.
    """

    def __init__(
        self,
        in_channels: int,
        roi_size: int | tuple[int, int],
        num_classes: int,
        representation_size: int = 1024,
    ) -> None:
        super().__init__()
        if isinstance(roi_size, int):
            roi_h = roi_w = roi_size
        else:
            roi_h, roi_w = roi_size

        flat_size = in_channels * roi_h * roi_w

        self.fc6 = nn.Linear(flat_size, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)
        self.cls_score = nn.Linear(representation_size, num_classes + 1)
        self.bbox_pred = nn.Linear(representation_size, num_classes * 4)

    @override
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: (N_rois, C, roi_h, roi_w) RoI-aligned feature crops.

        Returns:
            (class_logits, box_deltas):
                class_logits: (N_rois, num_classes + 1)
                box_deltas:   (N_rois, num_classes * 4)
        """
        x = x.flatten(1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        class_logits = self.cls_score(x)
        box_deltas = self.bbox_pred(x)
        return class_logits, box_deltas


# ---------------------------------------------------------------------------
# Bipartite assignment (Hungarian / Kuhn-Munkres)
# ---------------------------------------------------------------------------


def solve_assignment(
    cost: list[list[float]],
) -> tuple[list[int], list[int]]:
    """Min-cost bipartite assignment for a rectangular ``(n_rows × n_cols)``
    cost matrix with ``n_rows ≤ n_cols``.

    Standard Jonker-Volgenant / Kuhn-Munkres algorithm — O(n_rows² · n_cols).
    Verified against ``scipy.optimize.linear_sum_assignment`` (drop-in
    replacement for the cases used by DETR / MaskFormer / Mask2Former).

    Args:
        cost: ``n_rows × n_cols`` matrix where ``cost[i][j]`` is the cost of
            assigning row ``i`` to column ``j``.  Rows must not outnumber
            columns; transpose the matrix outside this function if needed.

    Returns:
        ``(row_ind, col_ind)`` — both length ``n_rows``, with ``row_ind`` =
        ``[0, 1, ..., n_rows - 1]`` and ``col_ind[i]`` the column matched to
        row ``i``.  Total cost ``sum(cost[i][col_ind[i]])`` is minimal.

    Notes:
        Pure Python so it works inside autograd-free preprocessing on every
        backend.  For very large matrices (~1000+) consider a vectorised
        solver — none of Lucid's detection models hit that regime today.

    Examples
    --------
    >>> from lucid.models._utils._detection import solve_assignment
    >>> solve_assignment([[1.0, 4.0], [3.0, 2.0]])
    ([0, 1], [0, 1])

    Rows and columns of the cheapest one-to-one pairing: row 0 takes
    column 0 at cost 1 and row 1 takes column 1 at cost 2, for 3 rather
    than the 7 the other pairing would cost.
    """
    nr = len(cost)
    if nr == 0:
        return [], []
    nc = len(cost[0])
    if nr > nc:
        # The reference matcher hands scipy either orientation and gets back
        # ``min(n_rows, n_cols)`` pairs.  A bare assert here turned "more
        # ground-truth objects than queries" — a legitimate image, and what
        # ``num_queries=20`` produces on a crowded scene — into a crash.
        # Solve the transpose and swap the result back.
        transposed = [[cost[r][c] for r in range(nr)] for c in range(nc)]
        cols, rows = solve_assignment(transposed)
        return rows, cols

    INF = float("inf")
    u = [0.0] * (nr + 1)
    v = [0.0] * (nc + 1)
    # p[j] = row assigned to column j (1-indexed; 0 = free).
    p = [0] * (nc + 1)
    way = [0] * (nc + 1)

    for i in range(1, nr + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (nc + 1)
        used = [False] * (nc + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, nc + 1):
                if not used[j]:
                    c = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if c < minv[j]:
                        minv[j] = c
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(nc + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        # Augment along the way back to column 0.
        while j0:
            j2 = way[j0]
            p[j0] = p[j2]
            j0 = j2

    row_ind: list[int] = [0] * nr
    for j in range(1, nc + 1):
        if p[j] != 0:
            row_ind[p[j] - 1] = j - 1
    return list(range(nr)), row_ind


# ---------------------------------------------------------------------------
# §5.5  Training-time label assignment and sampling
# ---------------------------------------------------------------------------
#
# Every two-stage detector in the zoo needs the same two steps before it can
# compute a loss: decide which predictions correspond to which ground-truth
# object, and then choose a class-balanced subset of them to train on.  Both
# are pure bookkeeping over an IoU matrix — no learned parameters — so they
# live here rather than being reimplemented per family.


@final
class Matcher:
    r"""Assign each prediction to a ground-truth box, or reject it.

    Given the IoU between every ground-truth box and every prediction, each
    prediction is labelled with the index of its best-overlapping ground
    truth, or with one of two sentinels:

    * ``BELOW_LOW_THRESHOLD`` (-1) — a negative.  Its best overlap is under
      ``low_threshold``, so it is confidently background.
    * ``BETWEEN_THRESHOLDS`` (-2) — ignored.  Its overlap falls in the
      ambiguous band, so it contributes to no loss term at all.

    The two thresholds are what distinguishes the callers.  Faster R-CNN's
    RPN uses ``(0.7, 0.3)``: over 0.7 is an object, under 0.3 is background,
    and the band between is discarded (§3.1.2).  Its box head uses
    ``(0.5, 0.5)``, which leaves the band empty so nothing is ignored.  Fast
    R-CNN uses ``(0.5, 0.1)`` because §2.3 draws its negatives from
    ``[0.1, 0.5)`` specifically — for that caller "between" is the useful
    class and "below" is what gets discarded.

    Parameters
    ----------
    high_threshold : float
        Minimum IoU for a prediction to be matched to a ground truth.
    low_threshold : float
        IoU below which a prediction is a confident negative.  Must not
        exceed ``high_threshold``.
    allow_low_quality_matches : bool, optional, default=False
        Also force-match, for every ground truth, whichever prediction(s)
        overlap it most — even below ``high_threshold``.  Without this a
        small or oddly-shaped object whose best anchor still falls short of
        0.7 would train nothing, which is why §3.1.2 defines positives as
        "the anchor with the highest IoU *or* any anchor over 0.7".

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._detection import Matcher
    >>> iou = lucid.tensor([[0.9, 0.5, 0.05]])   # 1 ground truth, 3 anchors
    >>> Matcher(0.7, 0.3)(iou).tolist()
    [0, -2, -1]

    Matched to ground truth 0, ignored, background — the three outcomes
    in order. The middle anchor needs an overlap *between* the two
    thresholds to land in the ignore band; below the low one it is
    background like any other miss.
    """

    BELOW_LOW_THRESHOLD: int = -1
    BETWEEN_THRESHOLDS: int = -2

    def __init__(
        self,
        high_threshold: float,
        low_threshold: float,
        allow_low_quality_matches: bool = False,
    ) -> None:
        if low_threshold > high_threshold:
            raise ValueError(
                f"low_threshold ({low_threshold}) must not exceed "
                f"high_threshold ({high_threshold}); the band between them is "
                "the ignore region, and a negative-width band would silently "
                "make every prediction either matched or background."
            )
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, iou: Tensor) -> Tensor:
        """Label every prediction.

        Args:
            iou: ``(M, N)`` IoU between ``M`` ground truths and ``N``
                predictions, as produced by :func:`box_iou`.

        Returns:
            ``(N,)`` int tensor.  Entry ``n`` is the ground-truth index
            matched to prediction ``n``, or one of the two sentinels.

        Raises:
            ValueError: If there are no ground-truth rows.  With nothing to
                match against, every prediction is background — but the
                caller has to say so explicitly, because silently returning
                all-negative hides an empty-target bug.
        """
        if int(iou.shape[0]) == 0:
            raise ValueError(
                "Matcher got an IoU matrix with no ground-truth rows.  An "
                "image with no objects is legitimate, but the caller must "
                "handle it: every prediction is background, and no "
                "regression term is defined."
            )

        best_gt_iou, matches = iou.max(dim=0)  # (N,) best overlap and GT index

        below = best_gt_iou < self.low_threshold
        between = (best_gt_iou >= self.low_threshold) & (best_gt_iou < self.high_threshold)
        result = lucid.where(below, self.BELOW_LOW_THRESHOLD, matches)
        result = lucid.where(between, self.BETWEEN_THRESHOLDS, result)

        if self.allow_low_quality_matches:
            # For each ground truth, every prediction tied for its highest
            # overlap is restored to a match.  Comparing against the row max
            # rather than taking an argmax keeps all of the tied predictions,
            # which is what the reference does.
            best_per_gt = iou.max(dim=1, keepdim=True).values  # (M, 1)
            forced = (iou == best_per_gt).sum(dim=0) > 0  # (N,)
            result = lucid.where(forced, matches, result)

        return result


@final
class BalancedPositiveNegativeSampler:
    r"""Draw a class-balanced subset of labelled predictions to train on.

    A detector proposes far more background than foreground — an RPN sees
    tens of thousands of anchors of which a handful are objects — so
    training on all of them makes the classification loss almost entirely
    background and the model learns to predict nothing.  Both papers fix
    this the same way: sample a fixed-size minibatch per image at a target
    foreground ratio, and take everything else as background.

    When there are fewer positives than the ratio asks for, the shortfall is
    filled with negatives rather than left empty, so the minibatch size is
    held constant (as in Faster R-CNN §3.1.3, "if there are fewer than 128
    positive samples in an image, we pad the mini-batch with negative ones").

    Parameters
    ----------
    batch_size_per_image : int
        Total number of predictions sampled per image.  256 for Faster
        R-CNN's RPN, 64 for Fast R-CNN's box head.
    positive_fraction : float
        Target share of foreground, in ``[0, 1]``.  0.5 for the RPN, 0.25
        for the box head.

    Notes
    -----
    Selection uses :func:`lucid.randperm`, so :func:`lucid.manual_seed`
    makes it reproducible along with everything else in a training run.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._detection import (
    ...     BalancedPositiveNegativeSampler,
    ... )
    >>> labels = lucid.tensor([1, 1, 0, 0, 0, 0, -1]).long()
    >>> sampler = BalancedPositiveNegativeSampler(4, 0.5)
    >>> pos, neg = sampler(labels)
    >>> len(pos.tolist()), len(neg.tolist())
    (2, 2)
    """

    def __init__(self, batch_size_per_image: int, positive_fraction: float) -> None:
        if batch_size_per_image <= 0:
            raise ValueError(
                "batch_size_per_image is the number of predictions sampled "
                f"per image, so it must be positive; got {batch_size_per_image}."
            )
        if not 0.0 <= positive_fraction <= 1.0:
            raise ValueError(
                "positive_fraction is a share of the minibatch, so it must "
                f"lie in [0, 1]; got {positive_fraction}."
            )
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, labels: Tensor) -> tuple[Tensor, Tensor]:
        """Sample foreground and background indices for one image.

        Args:
            labels: ``(N,)`` per-prediction labels — ``1`` foreground, ``0``
                background, anything negative ignored.

        Returns:
            ``(positive_indices, negative_indices)``, both int tensors
            indexing into ``labels``.  Their combined length is at most
            ``batch_size_per_image`` and is smaller only when the image does
            not contain that many usable predictions.
        """
        flat = cast(list[int], labels.reshape(-1).tolist())
        pos = [i for i, v in enumerate(flat) if v >= 1]
        neg = [i for i, v in enumerate(flat) if v == 0]

        num_pos = min(len(pos), int(self.batch_size_per_image * self.positive_fraction))
        # Backfill with negatives so the minibatch keeps its size when the
        # image is object-poor; this is the padding rule from §3.1.3.
        num_neg = min(len(neg), self.batch_size_per_image - num_pos)

        pos_sel = _sample_without_replacement(pos, num_pos)
        neg_sel = _sample_without_replacement(neg, num_neg)
        dev = labels.device.type
        return (
            lucid.tensor(pos_sel, device=dev).long(),
            lucid.tensor(neg_sel, device=dev).long(),
        )


def _sample_without_replacement(pool: list[int], k: int) -> list[int]:
    """Take ``k`` distinct entries of ``pool`` uniformly at random.

    Args:
        pool: Candidate indices.
        k:    How many to take; ``k >= len(pool)`` returns the whole pool.

    Returns:
        The chosen indices, in ascending order so downstream gathers stay
        deterministic given the same draw.
    """
    if k <= 0:
        return []
    if k >= len(pool):
        return list(pool)
    perm = cast(list[int], lucid.randperm(len(pool)).tolist())
    return sorted(pool[i] for i in perm[:k])


def assign_anchors_to_targets(
    anchors: Tensor,
    gt_boxes: Tensor,
    matcher: Matcher,
    image_size: tuple[int, int],
    *,
    ignore_cross_boundary: bool = False,
) -> tuple[Tensor, Tensor]:
    """Label every anchor 1 / 0 / -1 and record which ground truth it hit.

    Args:
        anchors:    ``(N, 4)`` xyxy anchors, concatenated across levels.
        gt_boxes:   ``(M, 4)`` xyxy ground-truth boxes.
        matcher:    Configured :class:`Matcher`.
        image_size: ``(H, W)``, used only for cross-boundary removal.
        ignore_cross_boundary: Apply Faster R-CNN 3.1.2's "ignore all
            cross-boundary anchors" clause.

    Returns:
        ``(labels, matched)`` — ``labels`` is ``1`` foreground, ``0``
        background, ``-1`` ignored; ``matched`` gives the ground-truth index
        per anchor and is meaningful only where ``labels == 1``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._detection import (
    ...     Matcher,
    ...     assign_anchors_to_targets,
    ... )
    >>> anchors = lucid.tensor(
    ...     [[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0]]
    ... )
    >>> gt = lucid.tensor([[0.0, 0.0, 10.0, 10.0]])
    >>> labels, matched = assign_anchors_to_targets(
    ...     anchors, gt, Matcher(0.7, 0.3), (64, 64)
    ... )
    >>> labels.tolist()
    [1, 0]
    >>> matched.tolist()
    [0, -1]

    The first anchor is foreground and points at ground truth 0; the
    second overlaps nothing and is background, with -1 standing for "no
    target" rather than an index.
    """
    n = int(anchors.shape[0])
    dev = anchors.device.type
    if int(gt_boxes.shape[0]) == 0:
        # No objects: every anchor is a negative and nothing regresses.
        zeros = lucid.zeros((n,), device=dev).long()
        return zeros, zeros

    matched = matcher(box_iou(gt_boxes, anchors))
    labels = lucid.where(
        matched >= 0,
        lucid.ones_like(matched),
        lucid.where(
            matched == Matcher.BETWEEN_THRESHOLDS,
            lucid.full_like(matched, -1),
            lucid.zeros_like(matched),
        ),
    )
    if ignore_cross_boundary:
        iH, iW = image_size
        inside = (
            (anchors[:, 0] >= 0.0)
            & (anchors[:, 1] >= 0.0)
            & (anchors[:, 2] <= float(iW))
            & (anchors[:, 3] <= float(iH))
        )
        labels = lucid.where(inside, labels, lucid.full_like(labels, -1))
    return labels, matched


def rpn_loss(
    logits: list[Tensor],
    deltas: list[Tensor],
    anchors: list[Tensor],
    targets: list[dict[str, Tensor]],
    matcher: Matcher,
    sampler: BalancedPositiveNegativeSampler,
    image_size: tuple[int, int],
    *,
    ignore_cross_boundary: bool = False,
) -> tuple[Tensor, Tensor]:
    """RPN objectness + box-regression loss (Faster R-CNN 3.1.2 / 3.1.3).

    Both terms are divided by the number of *sampled* anchors rather than by
    the positive count.  Dividing the regression term by positives would let
    it grow on images with few objects, which is the opposite of what the
    normalisation is for.

    Args:
        logits:   Per-level objectness maps, ``(B, A, H, W)``.
        deltas:   Per-level box deltas, ``(B, A * 4, H, W)``.
        anchors:  Per-level anchor tensors, each ``(N_l, 4)``.
        targets:  Per-image ``{"boxes", "labels"}``.
        matcher:  Anchor-labelling rule.
        sampler:  Minibatch sampler.
        image_size: ``(H, W)`` of the input.
        ignore_cross_boundary: Forwarded to
            :func:`assign_anchors_to_targets`.

    Returns:
        ``(objectness_loss, regression_loss)``, both scalars.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._detection import (
    ...     BalancedPositiveNegativeSampler,
    ...     Matcher,
    ...     rpn_loss,
    ... )
    >>> logits = [lucid.zeros(1, 3, 4, 4)]
    >>> deltas = [lucid.zeros(1, 12, 4, 4)]
    >>> anchors = [lucid.tensor([[0.0, 0.0, 10.0, 10.0]] * 48)]
    >>> targets = [{
    ...     "boxes": lucid.tensor([[0.0, 0.0, 10.0, 10.0]]),
    ...     "labels": lucid.tensor([1]),
    ... }]
    >>> objectness, box = rpn_loss(
    ...     logits,
    ...     deltas,
    ...     anchors,
    ...     targets,
    ...     Matcher(0.7, 0.3),
    ...     BalancedPositiveNegativeSampler(64, 0.5),
    ...     (64, 64),
    ... )
    >>> round(float(objectness.item()), 6)
    0.693147

    log 2 — a logit of zero is exactly undecided about whether an anchor
    holds an object.

    >>> round(float(box.item()), 6)
    0.0

    The sampler is not optional. Anchors are overwhelmingly background,
    and training on all of them teaches the head to answer "nothing"
    every time; it draws a fixed batch at a fixed positive fraction
    instead.
    """
    flat_scores, flat_deltas = flatten_rpn_outputs(logits, deltas)
    objectness = lucid.cat(flat_scores, dim=1)  # (B, N)
    pred_deltas = lucid.cat(flat_deltas, dim=1)  # (B, N, 4)
    anchors_cat = lucid.cat(anchors, dim=0)  # (N, 4)

    obj_parts: list[Tensor] = []
    reg_parts: list[Tensor] = []
    n_sampled = 0

    for b, tgt in enumerate(targets):
        gt_boxes = tgt["boxes"]
        labels, matched = assign_anchors_to_targets(
            anchors_cat,
            gt_boxes,
            matcher,
            image_size,
            ignore_cross_boundary=ignore_cross_boundary,
        )
        pos, neg = sampler(labels)
        n_pos = int(pos.shape[0])
        n_this = n_pos + int(neg.shape[0])
        if n_this == 0:
            continue
        n_sampled += n_this

        sampled = lucid.cat([pos, neg], dim=0)
        obj_target = labels[sampled].to(objectness.dtype)
        obj_parts.append(
            F.binary_cross_entropy_with_logits(objectness[b][sampled], obj_target, reduction="sum")
        )
        if n_pos > 0:
            matched_gt = gt_boxes[matched[pos].long()]
            reg_targets = encode_boxes(matched_gt, anchors_cat[pos], (1.0, 1.0, 1.0, 1.0))
            reg_parts.append(
                F.smooth_l1_loss(
                    pred_deltas[b][pos],
                    reg_targets,
                    beta=1.0 / 9.0,
                    reduction="sum",
                )
            )

    if n_sampled == 0:
        zero = objectness.new_zeros()
        return zero, zero

    denom = float(n_sampled)
    obj = lucid.cat([t.reshape(1) for t in obj_parts]).sum() / denom
    reg = (
        lucid.cat([t.reshape(1) for t in reg_parts]).sum() / denom
        if reg_parts
        else objectness.new_zeros()
    )
    return obj, reg


def select_training_samples(
    proposals: list[Tensor],
    targets: list[dict[str, Tensor]],
    matcher: Matcher,
    sampler: BalancedPositiveNegativeSampler,
    bbox_reg_weights: tuple[float, float, float, float],
) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]]:
    """Sample the box head's minibatch and build its targets.

    Ground-truth boxes are appended to the proposal set first, as the
    reference does: early in training the RPN proposes nothing useful, and a
    minibatch with no positives teaches the box head only what background
    looks like.

    Args:
        proposals: Per-image proposal tensors, each ``(P, 4)`` xyxy.
        targets:   Per-image ``{"boxes", "labels"}``; labels are 1-based,
            with 0 reserved for background.
        matcher:   Proposal-labelling rule.
        sampler:   Minibatch sampler.
        bbox_reg_weights: Per-component scale applied when encoding the
            regression targets.

    Returns:
        ``(proposals, labels, regression_targets, matched_gt_indices)``,
        each per image and already restricted to the sampled RoIs.  The
        matched indices are what a mask branch needs to find each RoI's
        ground-truth mask.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._detection import (
    ...     BalancedPositiveNegativeSampler,
    ...     Matcher,
    ...     select_training_samples,
    ... )
    >>> proposals = [lucid.tensor(
    ...     [[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0]]
    ... )]
    >>> targets = [{
    ...     "boxes": lucid.tensor([[0.0, 0.0, 10.0, 10.0]]),
    ...     "labels": lucid.tensor([1]),
    ... }]
    >>> sampled, labels, regression, _ = select_training_samples(
    ...     proposals,
    ...     targets,
    ...     Matcher(0.5, 0.5),
    ...     BalancedPositiveNegativeSampler(64, 0.25),
    ...     (10.0, 10.0, 5.0, 5.0),
    ... )
    >>> sampled[0].shape, labels[0].shape
    ((3, 4), (3,))

    Three out of two: the ground-truth boxes are appended to the
    proposals before matching. Early in training the region proposals are
    noise, and without that the second stage would have almost no
    positive to learn from.
    """
    out_props: list[Tensor] = []
    out_labels: list[Tensor] = []
    out_reg: list[Tensor] = []
    out_matched: list[Tensor] = []

    for props, tgt in zip(proposals, targets, strict=True):
        gt_boxes = tgt["boxes"]
        gt_labels = tgt["labels"]
        dev = props.device.type

        if int(gt_boxes.shape[0]) > 0:
            props = lucid.cat([props, gt_boxes], dim=0)

        n = int(props.shape[0])
        if int(gt_boxes.shape[0]) == 0 or n == 0:
            out_props.append(props)
            out_labels.append(lucid.zeros((n,), device=dev).long())
            out_reg.append(props.new_zeros(n, 4))
            out_matched.append(lucid.zeros((n,), device=dev).long())
            continue

        matched = matcher(box_iou(gt_boxes, props))
        clamped = matched.clip(min=0).long()
        fg = matched >= 0
        labels = lucid.where(fg, gt_labels[clamped], lucid.zeros_like(gt_labels[clamped]))
        binary = lucid.where(fg, lucid.ones_like(labels), lucid.zeros_like(labels))

        pos, neg = sampler(binary)
        keep = lucid.tensor(
            sorted([*cast(list[int], pos.tolist()), *cast(list[int], neg.tolist())]),
            device=dev,
        ).long()

        props_k = props[keep]
        out_props.append(props_k)
        out_labels.append(labels[keep])
        out_reg.append(encode_boxes(gt_boxes[clamped[keep]], props_k, bbox_reg_weights))
        out_matched.append(clamped[keep])

    return out_props, out_labels, out_reg, out_matched


def fastrcnn_loss(
    class_logits: Tensor,
    box_deltas: Tensor,
    labels: list[Tensor],
    reg_targets: list[Tensor],
) -> tuple[Tensor, Tensor]:
    """Box-head classification + regression loss.

    The regression term is defined only on the ground-truth class of each
    positive RoI — the other ``K - 1`` box predictions for that RoI receive
    no gradient at all, which is what makes the head's outputs
    class-specific.

    Args:
        class_logits: ``(N, K)`` class logits over the sampled RoIs.
        box_deltas:   ``(N, K * 4)`` per-class box deltas.
        labels:       Per-image class ids for those RoIs.
        reg_targets:  Per-image encoded regression targets, ``(P, 4)``.

    Returns:
        ``(classification_loss, regression_loss)``, both scalars.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._detection import fastrcnn_loss
    >>> class_logits = lucid.zeros(4, 3)
    >>> box_deltas = lucid.zeros(4, 8)
    >>> labels = [lucid.tensor([0, 1, 2, 1])]
    >>> regression = [lucid.zeros(4, 4)]
    >>> classification, box = fastrcnn_loss(
    ...     class_logits, box_deltas, labels, regression
    ... )
    >>> round(float(classification.item()), 6)
    1.098612

    log 3 across three classes, again the undecided cost.

    >>> round(float(box.item()), 6)
    0.0

    The box term counts only the foreground proposals, so a batch whose
    deltas are already correct contributes nothing — background
    proposals have no box to regress toward at all.
    """
    dev = class_logits.device.type
    labels_cat = lucid.cat(labels, dim=0)
    n_total = int(labels_cat.shape[0])
    if n_total == 0:
        zero = class_logits.new_zeros()
        return zero, zero

    cls_loss = F.cross_entropy(class_logits, labels_cat.long())

    flat = cast(list[int], labels_cat.tolist())
    pos_idx = [i for i, v in enumerate(flat) if v > 0]
    if not pos_idx:
        return cls_loss, class_logits.new_zeros()

    pos = lucid.tensor(pos_idx, device=dev).long()
    pos_labels = labels_cat[pos].long()
    k = int(box_deltas.shape[1]) // 4
    per_class = box_deltas.reshape(n_total, k, 4)
    # Gather (row, its own class) without forming the (P, K, 4) block.
    chosen = per_class[pos].reshape(len(pos_idx) * k, 4)
    offsets = lucid.tensor(
        [i * k + int(c) for i, c in enumerate(cast(list[int], pos_labels.tolist()))],
        device=dev,
    ).long()
    pred = chosen[offsets]

    reg_cat = lucid.cat(reg_targets, dim=0)
    reg_loss = F.smooth_l1_loss(pred, reg_cat[pos], beta=1.0 / 9.0, reduction="sum") / float(
        n_total
    )
    return cls_loss, reg_loss


def project_masks_on_boxes(
    gt_masks: Tensor,
    proposals: Tensor,
    matched_idxs: Tensor,
    mask_size: int,
) -> Tensor:
    """Crop each proposal's ground-truth mask to the head's output grid.

    Mask R-CNN 3.1: "The mask target is the intersection between an RoI and
    its associated ground-truth mask."  Concretely, the ground-truth mask is
    RoI-aligned with the proposal's own box, so the target lands in the same
    ``m x m`` frame the head predicts in — which is why the head can be a
    small fixed-size FCN at all.

    Args:
        gt_masks:     ``(M, H, W)`` binary ground-truth masks for the image.
        proposals:    ``(P, 4)`` xyxy proposal boxes.
        matched_idxs: ``(P,)`` ground-truth index per proposal.
        mask_size:    Output side length ``m``.

    Returns:
        ``(P, mask_size, mask_size)`` float targets in ``[0, 1]``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._detection import project_masks_on_boxes
    >>> gt_masks = lucid.zeros(1, 32, 32)
    >>> proposals = lucid.tensor([[0.0, 0.0, 16.0, 16.0]])
    >>> matched = lucid.tensor([0])
    >>> project_masks_on_boxes(gt_masks, proposals, matched, 14).shape
    (1, 14, 14)

    The ground-truth mask cropped to each proposal and resampled to the
    head's resolution. Training targets have to live in the proposal's
    frame, not the image's, because that is where the head predicts.
    """
    if int(proposals.shape[0]) == 0:
        return proposals.new_zeros(0, mask_size, mask_size)

    # Each proposal crops its *own* ground-truth mask, so the batch axis is
    # the proposal axis: (P, 1, H, W) with exactly one RoI per entry.
    p_count = int(proposals.shape[0])
    picked = gt_masks[matched_idxs.long()].unsqueeze(1).to(proposals.dtype)
    rois = [proposals[i : i + 1] for i in range(p_count)]
    cropped = roi_align(
        picked,
        rois,
        output_size=mask_size,
        spatial_scale=1.0,
        sampling_ratio=1,
        aligned=True,
    )
    return cropped[:, 0]


def maskrcnn_loss(
    mask_logits: Tensor,
    labels: Tensor,
    mask_targets: Tensor,
) -> Tensor:
    r"""``L_mask`` — per-pixel sigmoid BCE on the ground-truth class only.

    3: "The mask branch has a :math:`Km^2`-dimensional output for each RoI,
    which encodes :math:`K` binary masks of resolution :math:`m \times m`,
    one for each of the :math:`K` classes.  To this we apply a per-pixel
    sigmoid, and define :math:`L_{\text{mask}}` as the average binary
    cross-entropy loss.  For an RoI associated with ground-truth class
    :math:`k`, :math:`L_{\text{mask}}` is only defined on the :math:`k`-th
    mask (other mask outputs do not contribute to the loss)."

    That last clause is the whole point: without it the classes compete
    per-pixel, and the paper's ablation shows the decoupled form is worth
    several AP.

    Args:
        mask_logits:  ``(P, K, m, m)`` raw logits for the sampled RoIs.
        labels:       ``(P,)`` class id per RoI; only ``> 0`` contribute.
        mask_targets: ``(P, m, m)`` binary targets from
            :func:`project_masks_on_boxes`.

    Returns:
        Scalar loss.  Zero when the minibatch contains no positive RoI —
        an image can legitimately sample none, and a NaN there would poison
        the whole step.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._detection import maskrcnn_loss
    >>> mask_logits = lucid.zeros(2, 5, 28, 28)
    >>> labels = lucid.tensor([1, 2])
    >>> mask_targets = lucid.zeros(2, 28, 28)
    >>> round(float(maskrcnn_loss(mask_logits, labels, mask_targets).item()), 6)
    0.693147

    log 2, the cost of an undecided logit. Only the channel matching each
    proposal's label contributes — the head predicts a mask per class and
    the loss reads one of them, which is what keeps classes from
    competing inside a single mask.
    """
    dev = mask_logits.device.type
    flat = cast(list[int], labels.reshape(-1).tolist())
    pos_idx = [i for i, v in enumerate(flat) if v > 0]
    if not pos_idx or int(mask_logits.shape[0]) == 0:
        return mask_logits.new_zeros()

    pos = lucid.tensor(pos_idx, device=dev).long()
    k = int(mask_logits.shape[1])
    m = int(mask_logits.shape[-1])

    # Select the ground-truth class channel of each positive RoI.
    chosen = mask_logits[pos].reshape(len(pos_idx) * k, m, m)
    pos_labels = cast(list[int], labels[pos].long().tolist())
    offsets = lucid.tensor([i * k + int(c) for i, c in enumerate(pos_labels)], device=dev).long()
    pred = chosen[offsets]  # (P_pos, m, m)

    return F.binary_cross_entropy_with_logits(pred, mask_targets[pos])


def flatten_rpn_outputs(
    logits: list[Tensor], deltas: list[Tensor]
) -> tuple[list[Tensor], list[Tensor]]:
    """Flatten per-level RPN maps to ``(B, H*W*A)`` / ``(B, H*W*A, 4)``.

    Spatial-major, anchor-minor — the order the anchor grid is laid out in,
    so entry ``n`` of the flattened prediction and entry ``n`` of the anchor
    tensor describe the same box.

    Shared by proposal generation and the training loss on purpose: the two
    must index anchors identically, and a second spelling of this reshape is
    a silent mis-assignment waiting to happen.

    Args:
        logits: Per-level objectness maps, each ``(B, A, H, W)``.
        deltas: Per-level box deltas, each ``(B, A * 4, H, W)``.

    Returns:
        ``(scores, deltas)`` — one entry per level, flattened to
        ``(B, H*W*A)`` and ``(B, H*W*A, 4)``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._detection import flatten_rpn_outputs
    >>> logits = [lucid.randn(1, 3, 4, 4)]
    >>> deltas = [lucid.randn(1, 12, 4, 4)]
    >>> flat_logits, flat_deltas = flatten_rpn_outputs(logits, deltas)
    >>> flat_logits[0].shape, flat_deltas[0].shape
    ((1, 48), (1, 48, 4))

    Three anchors across a 4x4 map is 48 predictions, and the deltas
    carry four numbers each — which is the layout the loss wants and the
    convolution does not produce.
    """
    per_level_scores: list[Tensor] = []
    per_level_deltas: list[Tensor] = []
    for lg, dl in zip(logits, deltas, strict=True):
        B = int(lg.shape[0])
        A = int(lg.shape[1])
        fH = int(lg.shape[2])
        fW = int(lg.shape[3])
        per_level_scores.append(lg.permute(0, 2, 3, 1).reshape(B, fH * fW * A))
        per_level_deltas.append(
            dl.reshape(B, A, 4, fH, fW).permute(0, 3, 4, 1, 2).reshape(B, fH * fW * A, 4)
        )
    return per_level_scores, per_level_deltas


# ---------------------------------------------------------------------------
# §6  Reference ResNet-50-FPN building blocks (reference_vision-exact key layout)
# ---------------------------------------------------------------------------
#
# The blocks below mirror the reference ``fasterrcnn_resnet50_fpn`` submodule
# names verbatim (``backbone.body.*`` / ``backbone.fpn.inner_blocks.{i}.0`` /
# ``backbone.fpn.layer_blocks.{i}.0`` / ``rpn.head.conv.0.0`` / ...), so the
# weight converter for the COCO checkpoint is a near-identity prefix map.  They
# are intentionally separate from the legacy ``FPN`` / ``RPN`` / ``RoIHead`` /
# ``AnchorGenerator`` above (used by Mask R-CNN / EfficientDet), whose key
# layout differs.


class _FrozenBatchNorm2d(nn.Module):
    r"""BatchNorm2d with frozen affine params + running stats (eval-only).

    Holds exactly four persistent buffers — ``weight``, ``bias``,
    ``running_mean``, ``running_var`` — with **no** ``num_batches_tracked``,
    matching the reference ``FrozenBatchNorm2d`` key-set.  The forward applies

    .. math::

        y = (x - \mathrm{running\_mean})
            \cdot \mathrm{rsqrt}(\mathrm{running\_var} + \varepsilon)
            \cdot \mathrm{weight} + \mathrm{bias}

    regardless of train / eval mode.  ``eps`` is configurable because the
    reference uses :math:`\varepsilon = 0` inside detection backbones but
    :math:`10^{-5}` elsewhere.
    """

    weight: Tensor
    bias: Tensor
    running_mean: Tensor
    running_var: Tensor

    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer("weight", lucid.ones(num_features))
        self.register_buffer("bias", lucid.zeros(num_features))
        self.register_buffer("running_mean", lucid.zeros(num_features))
        self.register_buffer("running_var", lucid.ones(num_features))

    @override
    def forward(self, x: Tensor) -> Tensor:
        w, b = self.weight.reshape(1, -1, 1, 1), self.bias.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        return x * scale + (b - rm * scale)


class _ResNetBottleneck(nn.Module):
    """ResNet bottleneck block (frozen BN) with reference key names."""

    expansion: int = 4

    def __init__(
        self,
        in_ch: int,
        mid_ch: int,
        stride: int = 1,
        downsample: nn.Sequential | None = None,
        bn_eps: float = 0.0,
    ) -> None:
        super().__init__()
        out_ch = mid_ch * self.expansion
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.bn1 = _FrozenBatchNorm2d(mid_ch, eps=bn_eps)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = _FrozenBatchNorm2d(mid_ch, eps=bn_eps)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn3 = _FrozenBatchNorm2d(out_ch, eps=bn_eps)
        self.downsample = downsample

    @override
    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out: Tensor = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return F.relu(out + identity)


def _make_resnet_layer(
    in_ch: int, mid_ch: int, num_blocks: int, stride: int, bn_eps: float
) -> tuple[nn.Sequential, int]:
    out_ch = mid_ch * 4
    ds: nn.Sequential | None = None
    if stride != 1 or in_ch != out_ch:
        ds = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            _FrozenBatchNorm2d(out_ch, eps=bn_eps),
        )
    blocks: list[nn.Module] = [
        _ResNetBottleneck(in_ch, mid_ch, stride=stride, downsample=ds, bn_eps=bn_eps)
    ]
    for _ in range(1, num_blocks):
        blocks.append(_ResNetBottleneck(out_ch, mid_ch, bn_eps=bn_eps))
    return nn.Sequential(*blocks), out_ch


@final
class _ResNetBody(nn.Module):
    """ResNet trunk exposing C2-C5, frozen BN, reference key layout.

    Submodule names (``conv1`` / ``bn1`` / ``layer1``..``layer4``) mirror the
    reference ResNet body so ``backbone.body.<rest>`` maps key-for-key.
    """

    def __init__(
        self,
        in_channels: int,
        layers: tuple[int, int, int, int],
        bn_eps: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = _FrozenBatchNorm2d(64, eps=bn_eps)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1, c2 = _make_resnet_layer(64, 64, layers[0], 1, bn_eps)
        self.layer2, c3 = _make_resnet_layer(c2, 128, layers[1], 2, bn_eps)
        self.layer3, c4 = _make_resnet_layer(c3, 256, layers[2], 2, bn_eps)
        self.layer4, c5 = _make_resnet_layer(c4, 512, layers[3], 2, bn_eps)
        self.out_channels_list: list[int] = [c2, c3, c4, c5]

    @override
    def forward(self, x: Tensor) -> list[Tensor]:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c2, c3, c4, c5]


@final
class _FeaturePyramidNetwork(nn.Module):
    """Reference FPN: ``inner_blocks`` (1x1 lateral) + ``layer_blocks`` (3x3).

    Matches the reference module names exactly — each block is a
    ``Sequential`` of a single ``Conv2d`` so the state-dict keys read
    ``inner_blocks.{i}.0.weight`` / ``layer_blocks.{i}.0.weight``.  A
    parameter-free ``LastLevelMaxPool`` appends the extra ``pool`` level
    (kernel 1, stride 2) on top of the coarsest output.
    """

    def __init__(self, in_channels_list: list[int], out_channels: int) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.inner_blocks = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(ic, out_channels, 1)) for ic in in_channels_list]
        )
        self.layer_blocks = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1))
                for _ in in_channels_list
            ]
        )

    @override
    def forward(self, features: list[Tensor]) -> list[Tensor]:
        """Top-down FPN over bottom-up maps (finest first) + a pool level.

        Returns ``len(features) + 1`` maps: the FPN levels followed by the
        ``LastLevelMaxPool`` output.
        """
        n = len(features)
        last_inner = self.inner_blocks[n - 1](features[n - 1])
        results: list[Tensor] = [self.layer_blocks[n - 1](last_inner)]
        for idx in range(n - 2, -1, -1):
            inner_lateral = self.inner_blocks[idx](features[idx])
            fh, fw = int(inner_lateral.shape[2]), int(inner_lateral.shape[3])
            inner_top_down = F.interpolate(last_inner, size=(fh, fw), mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.layer_blocks[idx](last_inner))
        # LastLevelMaxPool: stride-2 subsample of the coarsest FPN output.
        pool = F.max_pool2d(results[-1], kernel_size=1, stride=2, padding=0)
        results.append(pool)
        return results


@final
class _ReferenceAnchorGenerator(nn.Module):
    """Per-level anchors matching the reference ``AnchorGenerator`` exactly.

    Base anchors: ``ws = (1/sqrt(ratio)) * size``, ``hs = sqrt(ratio) * size``,
    stacked as ``[-ws, -hs, ws, hs] / 2`` then **rounded**.  Grid shifts use
    ``arange(0, dim) * stride`` (no half-cell offset); the ordering is
    spatial-major / anchor-minor (``shifts[:, None] + base[None, :]``).
    """

    def __init__(
        self,
        sizes: tuple[tuple[int, ...], ...],
        aspect_ratios: tuple[tuple[float, ...], ...],
    ) -> None:
        super().__init__()
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self._cell_anchors: list[Tensor] = [
            self._gen_cell_anchors(s, r) for s, r in zip(sizes, aspect_ratios, strict=True)
        ]

    @staticmethod
    def _gen_cell_anchors(scales: tuple[int, ...], ratios: tuple[float, ...]) -> Tensor:
        rows: list[list[float]] = []
        # Reference order: ratio outer, scale inner (w_ratios[:,None]*scales).
        for ratio in ratios:
            h_ratio = math.sqrt(ratio)
            w_ratio = 1.0 / h_ratio
            for scale in scales:
                ws = w_ratio * float(scale)
                hs = h_ratio * float(scale)
                rows.append(
                    [
                        round(-ws / 2.0),
                        round(-hs / 2.0),
                        round(ws / 2.0),
                        round(hs / 2.0),
                    ]
                )
        return lucid.tensor(rows)

    def _grid_anchors(
        self, grid_size: tuple[int, int], stride: int, base: Tensor, like: Tensor
    ) -> Tensor:
        gh, gw = grid_size
        shifts: list[list[float]] = [
            [float(c * stride), float(r * stride), float(c * stride), float(r * stride)]
            for r in range(gh)
            for c in range(gw)
        ]
        # (G, 4), in ``like``'s device and dtype: the reference casts so.
        shifts_t = lucid.tensor(shifts, dtype=like.dtype, device=like.device)
        base_dev = base.to(device=like.device, dtype=like.dtype)
        grid = shifts_t[:, None, :] + base_dev[None, :, :]
        return grid.reshape(-1, 4)

    @override
    def forward(self, feature_maps: list[Tensor], strides: list[int]) -> list[Tensor]:
        out: list[Tensor] = []
        for feat, base, stride in zip(feature_maps, self._cell_anchors, strides, strict=True):
            gh = int(feat.shape[2])
            gw = int(feat.shape[3])
            out.append(self._grid_anchors((gh, gw), stride, base, feat))
        return out


def _fpn_level_for_boxes(
    boxes: Tensor, k_min: int, k_max: int, canonical_scale: int, canonical_level: int
) -> Tensor:
    """Canonical FPN level assignment (eq. 1) — returns 0-based level index.

    ``k = floor(canonical_level + log2(sqrt(wh) / canonical_scale) + eps)``,
    clamped to ``[k_min, k_max]`` then shifted to a 0-based level index.
    """
    eps = 1e-6
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    # ``eps`` belongs *outside* the log, as Eq. 1 and the docstring above
    # both write it and as the reference's ``LevelMapper`` applies it: its
    # job is to stop a box that lands exactly on a level boundary from
    # floor()-ing down, not to regularise the logarithm.  Inside the log it
    # instead shifts every box's scale slightly, which changes the boundary
    # itself.  The lower clamp keeps a degenerate (zero-area) box out of
    # log2(0) now that eps no longer covers for it.
    s = area.clamp(0.0, 1e18).sqrt().clamp(min=1e-12)
    lvl = lucid.floor(float(canonical_level) + lucid.log2(s / float(canonical_scale)) + eps)
    lvl = lvl.clamp(float(k_min), float(k_max))
    return (lvl - float(k_min)).long()


def multiscale_roi_align(
    features: list[Tensor],
    boxes: list[Tensor],
    output_size: int,
    spatial_scales: list[float],
    sampling_ratio: int,
    canonical_scale: int = 224,
    canonical_level: int = 4,
) -> Tensor:
    r"""Pool per-image proposals from their assigned FPN level (RoI Align).

    Mirrors the reference ``MultiScaleRoIAlign``: each proposal is routed
    to one FPN level by :func:`_fpn_level_for_boxes`, RoI-aligned there
    with the level's spatial scale, and the per-level crops are scattered
    back into a single ``(sum(N_i), C, output_size, output_size)`` stack
    preserving the original proposal order (image-major, then per-image
    proposal order).

    Parameters
    ----------
    features : list[Tensor]
        FPN feature maps ``[P2, P3, ...]``, each of shape ``(B, C, H, W)``
        in coarsening order (highest spatial resolution first).
    boxes : list[Tensor]
        Per-image proposal boxes; ``boxes[i]`` has shape ``(N_i, 4)`` in
        image-pixel coordinates ``(x1, y1, x2, y2)``.
    output_size : int
        Spatial size of each pooled crop (square output of side
        ``output_size``).
    spatial_scales : list[float]
        Per-level scale factor mapping image pixels to feature-map units
        (e.g. ``1/4, 1/8, 1/16, 1/32`` for an FPN with 4 levels).
    sampling_ratio : int
        Number of bilinear samples per output bin (``0`` = adaptive,
        matching the reference RoI-Align default).
    canonical_scale : int, default 224
        Reference object scale used by the FPN level-assignment heuristic
        (Lin 2017 Eq. 1); larger boxes go to coarser levels.
    canonical_level : int, default 4
        FPN level a ``canonical_scale``-sized box is routed to.

    Returns
    -------
    Tensor
        Pooled features of shape
        ``(sum(N_i), C, output_size, output_size)``, image-major and
        preserving per-image proposal order.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._detection import multiscale_roi_align
    >>> features = [lucid.randn(1, 4, 16, 16)]
    >>> boxes = [lucid.tensor([[0.0, 0.0, 8.0, 8.0]])]
    >>> multiscale_roi_align(features, boxes, (2, 2), [1.0], 2).shape
    (1, 4, 2, 2)

    With a pyramid of several levels each box is routed to the one whose
    resolution suits its size, so a small box is not cropped from a map
    that has already thrown its detail away.
    """
    num_levels = len(features)
    C = int(features[0].shape[1])
    dev = features[0].device.type

    # Image-major proposal order (matches reference _convert_to_roi_format).
    flat_boxes = [b for b in boxes if int(b.shape[0]) > 0]
    if not flat_boxes:
        return features[0].new_zeros(0, C, output_size, output_size)
    all_boxes = lucid.cat(flat_boxes, dim=0)  # (R, 4)
    R = int(all_boxes.shape[0])

    # Source image of every row of ``all_boxes``.  ``roi_align`` selects the
    # feature map by *list position*, so a flattened box list would silently
    # pool every RoI from image 0; the batch index has to survive the flatten.
    img_of: list[int] = []
    for b_idx, b in enumerate(boxes):
        img_of.extend([b_idx] * int(b.shape[0]))

    # NB: the reference ``MultiScaleRoIAlign`` calls ``roi_align`` with the
    # default ``aligned=False`` (no half-pixel offset) — NOT ``aligned=True``.
    if num_levels == 1:
        return roi_align(
            features[0],
            list(boxes),
            output_size=output_size,
            spatial_scale=spatial_scales[0],
            sampling_ratio=sampling_ratio,
            aligned=False,
        )

    # Per-box level assignment.  lvl_min/lvl_max derived from the scales.
    lvl_min = int(round(-math.log2(spatial_scales[0])))
    lvl_max = int(round(-math.log2(spatial_scales[-1])))
    levels = _fpn_level_for_boxes(
        all_boxes, lvl_min, lvl_max, canonical_scale, canonical_level
    )  # (R,) 0-based
    lvl_list = host_ints(levels)

    n_img = len(boxes)
    out_rows: list[Tensor | None] = [None] * R
    for level in range(num_levels):
        idx = [i for i in range(R) if lvl_list[i] == level]
        if not idx:
            continue
        # Regroup this level's rows per source image so ``roi_align`` reads
        # each RoI from the feature map it actually came from.  ``roi_align``
        # emits image-major and skips empty entries, so ``order`` — built the
        # same way — maps its rows back to their original flat positions.
        per_img: list[Tensor] = []
        order: list[int] = []
        for b_idx in range(n_img):
            sel = [i for i in idx if img_of[i] == b_idx]
            if sel:
                sel_t = lucid.tensor(sel, device=dev).long()
                per_img.append(all_boxes[sel_t])
            else:
                per_img.append(all_boxes.new_zeros(0, 4))
            order.extend(sel)
        pooled = roi_align(
            features[level],
            per_img,
            output_size=output_size,
            spatial_scale=spatial_scales[level],
            sampling_ratio=sampling_ratio,
            aligned=False,
        )  # (len(idx), C, o, o)
        for j, orig in enumerate(order):
            out_rows[orig] = pooled[j : j + 1]

    parts: list[Tensor] = [r for r in out_rows if r is not None]
    return lucid.cat(parts, dim=0)


# ---------------------------------------------------------------------------
# §5.6  Multi-scale training schedule
# ---------------------------------------------------------------------------


@final
class MultiScaleResolution:
    r"""The input-size schedule of YOLOv2's multi-scale training (§2).

    The paper does not train at one resolution: *"every 10 batches our
    network randomly chooses a new image dimension size ... from the
    multiples of 32: {320, 352, ..., 608}"*.  The same weights then have to
    work from 320x320 to 608x608, which is what lets one trained model
    trade accuracy for speed at test time by simply being fed a smaller
    image.

    Nothing about the network needs to change to support this — it is fully
    convolutional, and the anchors are expressed relative to a 32-pixel
    cell — so all that was missing is the schedule itself.  This class is
    it: a training loop asks for the size before each batch and resizes its
    own images.

    Args:
        sizes:  Candidate side lengths.  Defaults to the paper's
            ``range(320, 609, 32)``.  Every entry must be a positive
            multiple of ``stride``, or the stride-32 feature map — and the
            passthrough layer that halves it again — would not come out
            integral.
        period: How many batches to hold a size before redrawing (paper: 10).
        stride: The network's total downsampling factor, used to validate
            ``sizes``.

    Notes
    -----
    The YOLOv2 loss used to make this schedule unusable on its own: the
    objectness term is evaluated at every cell and was summed raw, so the
    total tracked the cell count almost exactly — 69.9 at 320x320 against
    239.3 at 608x608 on a fixed batch, a 3.4x swing that no single learning
    rate can serve.  ``_compute_loss`` now divides the objectness pair by
    the cell count and the localisation terms by the positive count, which
    leaves the obj/noobj ratio ``lambda_noobj`` sets untouched and takes the
    grid out of the total.  Measured as ``loss / cells``, the spread across
    the schedule went from 1.05x — near-constant, which is what proportional
    means — to 5.9x.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn.functional as F
    >>> from lucid.models.vision.yolo import MultiScaleResolution
    >>> schedule = MultiScaleResolution()
    >>> images = lucid.randn(2, 3, 416, 416)
    >>> for batch in range(3):                      # doctest: +SKIP
    ...     side = schedule.size_for(batch)
    ...     batch_images = F.interpolate(
    ...         images, size=(side, side), mode="bilinear", align_corners=False
    ...     )
    ...     out = model(batch_images, targets=targets)
    >>> schedule.size_for(0) == schedule.size_for(9)   # same 10-batch window
    True
    """

    def __init__(
        self,
        sizes: tuple[int, ...] = tuple(range(320, 609, 32)),
        *,
        period: int = 10,
        stride: int = 32,
    ) -> None:
        if not sizes:
            raise ValueError("sizes must contain at least one resolution")
        if period <= 0:
            raise ValueError(f"period must be positive, got {period}")
        bad = [s for s in sizes if s <= 0 or s % stride != 0]
        if bad:
            raise ValueError(f"every size must be a positive multiple of {stride}, got {bad}")
        self.sizes = sizes
        self.period = period
        self.stride = stride
        self._window = -1
        self._size = sizes[0]

    def size_for(self, batch_index: int) -> int:
        """Side length to train batch ``batch_index`` at.

        Stable within a window of ``period`` batches and redrawn when the
        window rolls over — asking twice for the same batch gives the same
        answer, so a caller may query it for images and targets separately.
        """
        window = batch_index // self.period
        if window != self._window:
            self._window = window
            self._size = self._draw()
        return self._size

    def _draw(self) -> int:
        if len(self.sizes) == 1:
            return self.sizes[0]
        pick = cast(list[int], lucid.randperm(len(self.sizes)).tolist())
        return self.sizes[pick[0]]
