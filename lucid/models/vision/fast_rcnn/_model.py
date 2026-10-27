"""Fast R-CNN backbone and object detector (Girshick, 2015).

Paper: "Fast R-CNN" (ICCV 2015)

Key advance over R-CNN
----------------------
R-CNN applies the CNN once *per proposal* (very slow).
Fast R-CNN applies the CNN *once to the whole image*, projects proposals onto
the shared feature map, and uses RoI Pooling to extract fixed-size features.

Architecture
------------
1. Full image → VGG16 conv layers (conv1_1 … conv5_3) → feature map (stride 16).
2. External proposals are projected onto the feature map with RoI Pool (7×7).
3. Flattened features (7 × 7 × 512 = 25 088) → fc6 (4 096) → fc7 (4 096).
4. Two sibling output layers:
     a. cls_score  : (num_classes + 1) softmax — one score per class incl. bg
     b. bbox_pred  : (num_classes × 4) linear — class-specific (dx,dy,dw,dh)
5. Multi-task loss at training time:
     L = L_cls + λ · L_loc
   where L_cls is log-loss and L_loc is smooth-L1 on positive samples only.

Faithfulness notes
------------------
* Backbone is VGG16 conv1_1 … conv5_3 (pool5 replaced by RoI Pool).
* Spatial scale 1/16 — four max-pool halving layers before pool5.
* RoI Pool (not RoI Align) to match the original paper.
* bbox_reg_weights encode the target normalisation described in §3.1:
  tx* = wx * (Gx - Px) / Pw, etc.  Default (10, 10, 5, 5) matches the
  empirical mean/std used in the Fast R-CNN Caffe reference implementation.
* Smooth-L1 loss with σ = 1 per §3 (matches paper eq. 3).
"""

from typing import ClassVar, cast, final, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._tasks import ObjectDetectionModel
from lucid.models._output import ObjectDetectionOutput
from lucid.models._utils._detection import (
    batched_nms,
    clip_boxes_to_image,
    decode_boxes,
    encode_boxes,
    roi_pool,
)
from lucid.models._utils._detection_results import check_per_image, merge_class_detections
from lucid.models.vision.fast_rcnn._config import FastRCNNConfig
from lucid.models._utils._common import host_floats, host_ints, host_rows

# ---------------------------------------------------------------------------
# VGG16 convolutional backbone  (pool5 removed — replaced by RoI Pool)
# ---------------------------------------------------------------------------


def _vgg16_block(in_ch: int, out_ch: int, n: int) -> list[nn.Module]:
    """Build one VGG conv block (n × Conv-BN-ReLU, no pooling here)."""
    layers: list[nn.Module] = []
    for i in range(n):
        layers += [
            nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        ]
    return layers


@final
class _VGG16Features(nn.Module):
    """VGG16 conv layers conv1_1 … conv5_3 (pool5 omitted).

    Input  : (B, C, H, W)
    Output : (B, 512, H/16, W/16)   — stride 16 from four max-pool layers.

    Architecture:
      Block 1 : Conv(64)×2  → MaxPool → H/2
      Block 2 : Conv(128)×2 → MaxPool → H/4
      Block 3 : Conv(256)×3 → MaxPool → H/8
      Block 4 : Conv(512)×3 → MaxPool → H/16
      Block 5 : Conv(512)×3           → H/16  (pool5 REMOVED)
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            *_vgg16_block(in_channels, 64, 2),
            nn.MaxPool2d(2, stride=2),
            *_vgg16_block(64, 128, 2),
            nn.MaxPool2d(2, stride=2),
            *_vgg16_block(128, 256, 3),
            nn.MaxPool2d(2, stride=2),
            *_vgg16_block(256, 512, 3),
            nn.MaxPool2d(2, stride=2),
            *_vgg16_block(512, 512, 3),
            # pool5 intentionally omitted: RoI Pool takes its place
        )
        self.out_channels: int = 512

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.features(x)


# ---------------------------------------------------------------------------
# RoI head  (RoI Pool → FC → dual prediction heads)
# ---------------------------------------------------------------------------


@final
class _FastRCNNHead(nn.Module):
    """RoI-level feature processing and prediction heads.

    Input  : (N_rois, C, roi_size, roi_size) — RoI-pooled crops
    Output : (class_logits, bbox_deltas)
               class_logits : (N_rois, num_classes + 1)
               bbox_deltas  : (N_rois, num_classes × 4)

    Architecture:
      Flatten : roi_size² × C → flat (25 088 for 7×7×512)
      fc6     : flat → 4 096, ReLU, Dropout
      fc7     : 4 096 → 4 096, ReLU, Dropout
      cls     : 4 096 → num_classes + 1  (linear)
      bbox    : 4 096 → num_classes × 4  (linear)
    """

    def __init__(self, in_channels: int, roi_size: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        flat = in_channels * roi_size * roi_size  # 25 088 for VGG16 + 7×7

        # §3.1's truncated-SVD compression of fc6/fc7 is available as
        # ``lucid.models._utils._common.truncated_svd_linear``.  It is a
        # *post-training* transformation — it approximates weights that
        # already exist and changes the parameter layout, so it is applied
        # to a trained head rather than built in here.
        self.fc6 = nn.Linear(flat, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.drop = nn.Dropout(p=dropout)

        self.cls_score = nn.Linear(4096, num_classes + 1)
        self.bbox_pred = nn.Linear(4096, num_classes * 4)

    @override
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = x.flatten(1)
        x = self.drop(F.relu(self.fc6(x)))
        x = self.drop(F.relu(self.fc7(x)))
        return self.cls_score(x), self.bbox_pred(x)


# ---------------------------------------------------------------------------
# Smooth-L1 loss  (σ = 1, matching Fast R-CNN eq. 3)
# ---------------------------------------------------------------------------


def _smooth_l1(x: Tensor, sigma: float = 1.0) -> Tensor:
    """Element-wise smooth-L1 (Huber loss with transition at 1/σ²)."""
    sigma2 = sigma * sigma
    abs_x = lucid.abs(x)
    # |x| < 1/σ² → 0.5 σ² x²;  else  |x| - 0.5/σ²
    cond: Tensor = abs_x < (1.0 / sigma2)
    return lucid.where(cond, 0.5 * sigma2 * x * x, abs_x - 0.5 / sigma2)


# ---------------------------------------------------------------------------
# Fast R-CNN for Object Detection
# ---------------------------------------------------------------------------


class FastRCNNForObjectDetection(ObjectDetectionModel):
    r"""Fast R-CNN object detector (Girshick, ICCV 2015).

    The successor to R-CNN that fixes its main bottleneck: rather than running
    the CNN once *per proposal*, the backbone is applied **once** to the whole
    image, and per-proposal features are extracted via :func:`roi_pool` over
    the shared feature map (7x7 at stride 16 for the VGG16 default).  The
    pooled tensor is flattened, passed through two FC layers
    (``fc6``, ``fc7``), and split into sibling class (``cls_score``) and
    class-specific bounding-box (``bbox_pred``) heads.  At training time the
    model computes the paper's multi-task loss

    .. math::

        L = L_{\mathrm{cls}} + \lambda\, L_{\mathrm{loc}},

    with :math:`L_{\mathrm{cls}}` the categorical cross-entropy and
    :math:`L_{\mathrm{loc}}` the smooth-:math:`L_1` regression loss applied
    only to foreground proposals.

    Parameters
    ----------
    config : FastRCNNConfig
        Frozen architecture spec.  Use the :func:`fast_rcnn` factory for
        the paper-cited VGG16 configuration (RoI 7x7, stride-16 feature
        map, 80 COCO classes).

    Attributes
    ----------
    config : FastRCNNConfig
        Stored copy of the config that built this model.
    backbone : _VGG16Features
        VGG16 conv1_1 .. conv5_3 trunk (pool5 omitted) producing a
        :math:`(B, 512, H/16, W/16)` feature map.
    roi_head : _FastRCNNHead
        RoI feature processor: flatten -> ``fc6`` (4096) -> ``fc7`` (4096)
        -> sibling ``cls_score`` (K + 1 logits) and ``bbox_pred``
        (4K class-specific deltas).

    Notes
    -----
    See Girshick, "Fast R-CNN", ICCV 2015 (arXiv:1504.08083).  Bounding-box
    targets follow the paper's parameterisation

    .. math::

        t_x = w_x \frac{G_x - P_x}{P_w},\quad
        t_y = w_y \frac{G_y - P_y}{P_h},\quad
        t_w = w_w \log\!\frac{G_w}{P_w},\quad
        t_h = w_h \log\!\frac{G_h}{P_h},

    with the default normalisation weights :math:`(w_x, w_y, w_w, w_h) =
    (10, 10, 5, 5)` matching the Fast R-CNN Caffe reference.  RoI Pool (not
    RoI Align) is used to preserve faithfulness — see :class:`MaskRCNNForObjectDetection`
    for the RoI Align successor.  Per-class boxes are decoded for all classes
    and per-class NMS is applied by :meth:`postprocess`.

    Examples
    --------
    Inference with externally-supplied proposals:

    >>> import lucid
    >>> from lucid.models.vision.fast_rcnn import fast_rcnn
    >>> model = fast_rcnn(num_classes=20)
    >>> x = lucid.randn(1, 3, 600, 800)
    >>> proposals = [lucid.tensor(
    ...     [[10.0, 10.0, 200.0, 200.0],
    ...      [50.0, 60.0, 300.0, 280.0]])]
    >>> out = model(x, proposals)
    >>> out.logits.shape
    (2, 21)
    >>> out.loss is None
    True

    Training with ground-truth targets to compute the multi-task loss:

    >>> targets = [{
    ...     "boxes":  lucid.tensor([[20.0, 20.0, 180.0, 180.0]]),
    ...     "labels": lucid.tensor([3], dtype=lucid.int64),
    ... }]
    >>> out = model(x, proposals, targets=targets)
    >>> out.loss.shape
    ()
    """

    config_class: ClassVar[type[FastRCNNConfig]] = FastRCNNConfig
    base_model_prefix: ClassVar[str] = "fast_rcnn"

    def __init__(self, config: FastRCNNConfig) -> None:
        super().__init__(config)
        self._num_classes = config.num_classes
        self._spatial_scale = config.spatial_scale
        self._roi_size = config.roi_size
        self._score_thresh = config.score_thresh
        self._nms_thresh = config.nms_thresh
        self._max_det = config.max_detections
        self._bbox_weights = config.bbox_reg_weights

        self.backbone = _VGG16Features(config.in_channels)
        self.roi_head = _FastRCNNHead(
            in_channels=self.backbone.out_channels,
            roi_size=config.roi_size,
            num_classes=config.num_classes,
            dropout=config.dropout,
        )

    # ------------------------------------------------------------------
    # Training loss helpers
    # ------------------------------------------------------------------

    def _assign_proposals(
        self, proposals: Tensor, gt_boxes: Tensor, gt_labels: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Assign each proposal a GT class label and regression target.

        Rules (§2.3):
          IoU >= fg_iou_thresh              -> foreground, assigned to argmax GT
          bg_iou_thresh_lo <= IoU < fg      -> background (class 0)
          IoU < bg_iou_thresh_lo            -> ignored (label -1)

        The last rule is the paper's, not a convenience: §2.3 draws negatives
        from ``[0.1, 0.5)`` specifically, so a proposal overlapping nothing
        is *not* a training example — it is too easy to be informative.

        Args:
            proposals:  (N, 4) xyxy
            gt_boxes:   (M, 4) xyxy ground-truth boxes
            gt_labels:  (M,) foreground class ids (1-based)

        Returns:
            assigned_labels: (N,) int labels; -1 = ignored
            assigned_boxes:  (N, 4) matched GT box per proposal
        """
        from lucid.models._utils._detection import Matcher, box_iou

        N = int(proposals.shape[0])
        M = int(gt_boxes.shape[0])
        dev = proposals.device.type

        if M == 0:
            # An image with no objects: every proposal is background, and no
            # regression target is defined.  ``Matcher`` refuses this case on
            # purpose, so it is handled here where "no objects" is meaningful.
            return lucid.zeros((N,), device=dev), proposals.clone()

        cfg = cast(FastRCNNConfig, self.config)
        # (M, N) — Matcher takes ground truths as rows.
        iou = box_iou(gt_boxes, proposals)
        matcher = Matcher(cfg.fg_iou_thresh, cfg.bg_iou_thresh_lo)
        matched = matcher(iou)  # (N,) >=0 fg, -1 below band, -2 in band

        matched_idx = matched.clip(min=0)
        assigned_boxes = gt_boxes[matched_idx.long()]

        fg = matched >= 0
        # Matcher's "between" is Fast R-CNN's background; its "below" is the
        # ignore class.  That inversion is the whole reason the thresholds
        # are (0.5, 0.1) rather than (0.5, 0.5).
        ignored = matched == Matcher.BELOW_LOW_THRESHOLD
        fg_labels = gt_labels[matched_idx.long()]
        assigned_labels = lucid.where(
            fg, fg_labels, lucid.where(ignored, -1, lucid.zeros_like(fg_labels))
        )
        return assigned_labels, assigned_boxes

    def _sample_proposals(self, labels: Tensor) -> Tensor:
        """Draw §2.3's 64-RoI, 25%-foreground minibatch from one image.

        Training on every proposal makes the classification loss almost
        entirely background — the foreground share becomes whatever the
        proposal generator happened to produce — so the paper fixes the
        count and the ratio instead.

        Args:
            labels: ``(N,)`` from :meth:`_assign_proposals`; ``>0``
                foreground, ``0`` background, ``-1`` ignored.

        Returns:
            ``(S,)`` int tensor of the sampled proposal indices, ascending.
            ``S <= batch_size_per_image``, smaller only when the image does
            not have that many usable proposals.
        """
        from lucid.models._utils._detection import BalancedPositiveNegativeSampler

        cfg = cast(FastRCNNConfig, self.config)
        sampler = BalancedPositiveNegativeSampler(cfg.batch_size_per_image, cfg.positive_fraction)
        # The sampler speaks 1/0/-1; class ids carry more than that.
        binary = lucid.where(labels > 0, 1, lucid.where(labels == 0, lucid.zeros_like(labels), -1))
        pos, neg = sampler(binary)
        both = [*cast(list[int], pos.tolist()), *cast(list[int], neg.tolist())]
        return lucid.tensor(sorted(both), device=labels.device.type).long()

    def _compute_loss(
        self,
        proposals: list[Tensor],
        all_logits: Tensor,
        all_deltas: Tensor,
        targets: list[dict[str, Tensor]],
    ) -> Tensor:
        """Multi-task loss L = L_cls + λ * L_loc  (λ = 1, paper §3).

        Args:
            proposals:   Per-image proposal lists.
            all_logits:  (Σ N_i, K+1) raw class logits.
            all_deltas:  (Σ N_i, K*4) bbox regression output.
            targets:     Per-image dict with "boxes" and "labels".

        Returns:
            Scalar total loss.

        Note:
            Only the RoIs drawn by :meth:`_sample_proposals` contribute —
            §2.3's 64 per image at 25% foreground.  Both denominators below
            count the *sampled* RoIs, which is what makes the two terms
            comparable at lambda = 1.
        """
        all_cls_labels: list[Tensor] = []
        all_bbox_targets: list[Tensor] = []
        all_bbox_weights: list[Tensor] = []
        dev = all_logits.device.type

        sampled_logits: list[Tensor] = []
        sampled_deltas: list[Tensor] = []

        offset = 0
        for props, tgt in zip(proposals, targets, strict=True):
            N_i = int(props.shape[0])
            gt_boxes = tgt["boxes"]
            gt_labels = tgt["labels"]

            labels_i, matched_boxes_i = self._assign_proposals(props, gt_boxes, gt_labels)

            # §2.3's minibatch.  Everything downstream sees only these RoIs.
            keep = self._sample_proposals(labels_i)
            sampled_index = keep + offset
            labels_i = labels_i[keep]
            props_s = props[keep]
            matched_boxes_i = matched_boxes_i[keep]

            sampled_logits.append(all_logits[sampled_index])
            sampled_deltas.append(all_deltas[sampled_index])

            # Regression targets only for foreground (label > 0)
            reg_tgt_i = encode_boxes(matched_boxes_i, props_s, self._bbox_weights)
            weight_i = lucid.where(
                labels_i > 0, lucid.ones_like(labels_i).float(), lucid.zeros_like(labels_i).float()
            )

            all_cls_labels.append(labels_i)
            all_bbox_targets.append(reg_tgt_i)
            all_bbox_weights.append(weight_i)

            offset += N_i

        all_logits = lucid.cat(sampled_logits, dim=0)
        all_deltas = lucid.cat(sampled_deltas, dim=0)

        cls_labels = lucid.cat(all_cls_labels, dim=0)  # (Σ N_i,)
        bbox_targets = lucid.cat(all_bbox_targets, dim=0)  # (Σ N_i, 4)
        bbox_weights = lucid.cat(all_bbox_weights, dim=0)  # (Σ N_i,)

        # --- Classification loss (cross-entropy, skip ignored=-1) ---
        labels_list = host_ints(cls_labels)
        valid_mask = [n for n, label in enumerate(labels_list) if label >= 0]
        if not valid_mask:
            cls_loss: Tensor = all_logits.new_zeros(1)
        else:
            valid_t = lucid.tensor(valid_mask, device=dev).long()
            cls_loss = F.cross_entropy(all_logits[valid_t], cls_labels[valid_t])

        # --- Bbox regression loss (smooth-L1, foreground only) ---
        N_total = int(all_deltas.shape[0])
        K = self._num_classes

        # Select predicted delta for each proposal's assigned class
        # Expand bbox_targets to (N, K, 4) format for indexing
        pred_deltas = all_deltas.reshape(N_total, K, 4)

        reg_loss_parts: list[Tensor] = []
        weights_list = host_floats(bbox_weights)
        for n in range(N_total):
            if weights_list[n] == 0.0:
                continue
            cls_n = max(0, labels_list[n] - 1)
            cls_n = min(cls_n, K - 1)
            pred_d = pred_deltas[n, cls_n]  # (4,)
            tgt_d = bbox_targets[n]  # (4,)
            # Eq. (2) sums over the four coordinates; the division is by
            # *all* sampled RoIs (background included), not by the
            # foreground count and not by 4.
            reg_loss_parts.append(_smooth_l1(pred_d - tgt_d).sum())

        if reg_loss_parts:
            reg_loss = lucid.cat([l.reshape(1) for l in reg_loss_parts]).sum() / float(
                max(N_total, 1)
            )
        else:
            # Scalar in both branches — a foreground-free batch used to
            # return shape (1,), so the total loss changed rank.
            reg_loss = all_logits.new_zeros()

        return (cls_loss + reg_loss).reshape(())

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @override
    def forward(
        self,
        x: Tensor,
        proposals: list[Tensor] | None = None,
        targets: list[dict[str, Tensor]] | None = None,
    ) -> ObjectDetectionOutput:
        """Run Fast R-CNN on a batch of images.

        Args:
            x:         (B, C, H, W) image batch.
            proposals: list of B tensors, each (N_i, 4) xyxy proposals.
            targets:   Optional training targets (list of dicts with
                       "boxes" and "labels").

        Returns:
            ``ObjectDetectionOutput``:
              ``logits``     : (Σ N_i, num_classes + 1) raw class logits.
              ``pred_boxes`` : (Σ N_i, 4) decoded xyxy top-class boxes.
              ``loss``       : scalar multi-task loss (only with targets).
        """
        B = int(x.shape[0])
        iH = int(x.shape[2])
        iW = int(x.shape[3])

        if proposals is None:
            proposals = [x.new_zeros(0, 4) for _ in range(B)]
        check_per_image(type(self).__name__, B, proposals=proposals, targets=targets)

        # 1. Shared feature extraction (one forward pass for the whole batch)
        feat_map = self.backbone(x)  # (B, 512, H/16, W/16)

        # 2. RoI Pool on the shared feature map
        roi_crops = roi_pool(
            feat_map, proposals, output_size=self._roi_size, spatial_scale=self._spatial_scale
        )  # (Σ N_i, 512, roi_size, roi_size)

        # 3. FC head
        all_logits, all_deltas = self.roi_head(roi_crops)
        # all_logits : (Σ N_i, K+1)
        # all_deltas : (Σ N_i, K*4)

        # 4. Decode top-class boxes
        all_boxes = self._decode_all_boxes(proposals, all_deltas, (iH, iW))

        # 5. Training loss
        loss: Tensor | None = None
        if targets is not None:
            loss = self._compute_loss(proposals, all_logits, all_deltas, targets)

        return ObjectDetectionOutput(
            logits=all_logits, pred_boxes=all_boxes, proposals=tuple(proposals), loss=loss
        )

    def _decode_all_boxes(
        self, proposals: list[Tensor], all_deltas: Tensor, image_size: tuple[int, int]
    ) -> Tensor:
        """Decode bbox deltas per class, returning ``(N_total, K, 4)``.

        Paper §3.2 specifies class-specific bounding-box regression — at
        inference, NMS for class ``c`` must use the boxes decoded with
        class ``c``'s deltas, not a single canonical box across classes.
        """
        K = self._num_classes
        N_total = int(all_deltas.shape[0])

        if any(int(p.shape[0]) > 0 for p in proposals):
            flat_props = lucid.cat([p for p in proposals if int(p.shape[0]) > 0], dim=0)
        else:
            return all_deltas.new_zeros(0, K, 4)

        # all_deltas: (N_total, K*4) → (N_total, K, 4)
        deltas_3d = all_deltas.reshape(N_total, K, 4)

        # Decode every class's delta independently against the same proposals,
        # producing one (N_total, 4) box per class then stacking back.
        per_class: list[Tensor] = []
        for c in range(K):
            boxes_c = decode_boxes(deltas_3d[:, c, :], flat_props, self._bbox_weights)
            per_class.append(clip_boxes_to_image(boxes_c, image_size))
        return lucid.stack(per_class, dim=1)  # (N_total, K, 4)

    # ------------------------------------------------------------------
    # Post-processing (score threshold + per-class NMS)
    # ------------------------------------------------------------------

    def postprocess(
        self, output: ObjectDetectionOutput, proposals: list[Tensor]
    ) -> list[dict[str, Tensor]]:
        """Apply per-class NMS to raw Fast R-CNN output.

        Args:
            output:    ``ObjectDetectionOutput`` from ``forward()``.
            proposals: Proposal list passed to ``forward()`` (for shape info).

        Returns:
            Per-image list of result dicts:
              ``"boxes"``  : (K_det, 4)  kept xyxy detections
              ``"scores"`` : (K_det,)    class confidence scores
              ``"labels"`` : (K_det,)    class indices (1-based)
        """
        logits = output.logits  # (Σ N_i, K+1)
        pred_boxes = output.pred_boxes  # (Σ N_i, K, 4) — class-specific decoded

        results: list[dict[str, Tensor]] = []
        offset = 0
        dev = logits.device.type

        for props in proposals:
            N_i = int(props.shape[0])
            lg_i = logits[offset : offset + N_i]  # (N_i, K+1)
            bx_i = pred_boxes[offset : offset + N_i]  # (N_i, K, 4)
            offset += N_i

            scores_i = F.softmax(lg_i, dim=-1)
            host_scores = host_rows(scores_i)

            keep_boxes: list[Tensor] = []
            keep_scores: list[Tensor] = []
            keep_labels: list[Tensor] = []

            for c in range(1, self._num_classes + 1):
                cls_scores = scores_i[:, c]
                # Class-specific decoded boxes for class c (1-based label →
                # 0-based delta index).
                bx_class = bx_i[:, c - 1, :]  # (N_i, 4)

                mask: list[int] = [
                    i for i, row in enumerate(host_scores) if row[c] >= self._score_thresh
                ]
                if not mask:
                    continue

                mask_t = lucid.tensor(mask, device=dev).long()
                sc_c = cls_scores[mask_t]
                bx_c = bx_class[mask_t]

                keep = batched_nms(
                    bx_c, sc_c, lucid.zeros(int(sc_c.shape[0]), device=dev), self._nms_thresh
                )
                # ``max_detections`` is a *per image* limit applied once after a
                # global score sort.  Capping inside the class loop let each of
                # the K classes keep its own full quota, so the returned count
                # scaled with the class count.

                keep_boxes.append(bx_c[keep])
                keep_scores.append(sc_c[keep])
                keep_labels.append(
                    lucid.full((int(keep.shape[0]),), c, dtype=lucid.int64, device=dev)
                )

            results.append(
                merge_class_detections(keep_boxes, keep_scores, keep_labels, dev, self._max_det)
            )

        return results
