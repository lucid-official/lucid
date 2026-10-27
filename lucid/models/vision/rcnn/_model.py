"""R-CNN backbone and object detector (Girshick et al., 2014).

Paper: "Rich feature hierarchies for accurate object detection
        and semantic segmentation" (CVPR 2014)

Key idea:
  1. Generate ~2 000 category-independent region proposals (selective search,
     external — proposals are given as model input).
  2. Warp each proposal to a fixed roi_size × roi_size crop.
  3. Forward each crop independently through an AlexNet-style CNN to get a
     fixed-length feature vector (pool5 → 9 216-dim for 227 × 227 input).
  4. Classify with a linear softmax head; refine box coordinates with a
     separate linear regression head.

Faithfulness notes
------------------
* The original paper trains per-class one-vs-rest linear SVMs on top of
  frozen CNN features, and so does the authors' release — ``rcnn_train.m``
  is entirely an SVM trainer and ``rcnn_test.m`` scores with the stored
  SVM weights.  Softmax appears there only in the comparison utility
  ``rcnn_test_softmax.m``, which the paper itself reports as ~3.3 mAP
  worse (§4.2 / Appendix B).  This module implements the softmax head
  because it is what modern reproductions use and what an end-to-end
  training path can actually optimise — not because the reference does.
* ``proposals`` must be supplied externally (e.g. from selective search or
  any other proposal method).  This module covers the CNN + head portion.
* Bounding-box regression targets are *class-specific* (num_classes × 4
  deltas), matching the paper's formulation.
"""

from typing import ClassVar, cast, final, override

import math

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._tasks import ObjectDetectionModel
from lucid.models._output import ObjectDetectionOutput
from lucid.models._utils._detection import (
    encode_boxes,
    box_iou,
    batched_nms,
    clip_boxes_to_image,
    decode_boxes,
)
from lucid.models._utils._detection_results import (
    check_per_image,
    merge_class_detections,
)
from lucid.models.vision.rcnn._config import RCNNConfig
from lucid.models._utils._common import host_rows

# ---------------------------------------------------------------------------
# CNN backbone (AlexNet-style, applied per RoI crop)
# ---------------------------------------------------------------------------


@final
class _ConvFeatures(nn.Module):
    """AlexNet convolutional trunk applied to warped RoI crops.

    Input  : (N, C, roi_size, roi_size)  — N warped proposal crops
    Output : (N, 9216)                   — flattened pool5 features
                                           (for roi_size = 227)

    Architecture (CaffeNet, the network the paper fine-tunes):
      Conv1  : 11×11, s=4, p=0 → 96ch  → 55×55
      Pool1  : 3×3, s=2         →        27×27
      Conv2  : 5×5,  p=2 → 256ch →       27×27
      Pool2  : 3×3, s=2         →        13×13
      Conv3  : 3×3,  p=1 → 384ch →       13×13
      Conv4  : 3×3,  p=1 → 384ch →       13×13
      Conv5  : 3×3,  p=1 → 256ch →       13×13
      Pool5  : 3×3, s=2         →         6× 6
      Flatten:               → 256*6*6 = 9 216

    Two documented divergences from CaffeNet, both simplifications rather
    than accidents — this trunk is randomly initialised, so nothing depends
    on matching CaffeNet's tensor layout:

    * **Grouped convolutions.**  CaffeNet splits conv2/conv4/conv5 into two
      groups (the 2012 two-GPU partition).  Here they are dense, which is a
      strictly larger model at the same channel widths.
    * **Local response normalisation.**  CaffeNet has LRN after conv1 and
      conv2; it is omitted, as it is in every post-2014 re-derivation.
    """

    def __init__(self, in_channels: int, roi_size: int = 227) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1.  padding=0: CaffeNet takes a 227px crop straight into
            # an 11x11 stride-4 conv, giving 55x55.  padding=2 makes it 56x56,
            # which shifts every downstream receptive field by half a stride.
            nn.Conv2d(in_channels, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Block 2
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Block 3
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Block 4
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Block 5
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # ``roi_size`` is an advertised override, so the flattened width has to
        # follow it rather than assume the 6x6 pool5 of a 227px crop.  One
        # zero-input trace at construction settles it exactly.
        with lucid.no_grad():
            probe = lucid.zeros((1, in_channels, roi_size, roi_size))
            self.out_dim: int = int(self.features(probe).flatten(1).shape[1])

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return x.flatten(1)  # (N, 9216)


# ---------------------------------------------------------------------------
# FC head (classification + bbox regression)
# ---------------------------------------------------------------------------


@final
class _FCHead(nn.Module):
    """Two-layer FC trunk followed by dual prediction heads.

    Input  : (N, feat_dim)   — pool5 feature per RoI
    Output : (class_logits, bbox_deltas)
               class_logits : (N, num_classes + 1)
               bbox_deltas  : (N, num_classes * 4)  — class-specific deltas

    Architecture:
      fc6   : feat_dim → 4 096, ReLU, Dropout
      fc7   : 4 096    → 4 096, ReLU, Dropout
      cls   : 4 096    → num_classes + 1   (linear, on fc7)
      bbox  : feat_dim → num_classes × 4   (linear, on **pool5**)

    Appendix C is explicit that the regressor's input is pool5: "we use the
    features computed by the CNN ... :math:`P^i_5` denotes the pool_5
    features", and the released code trains its ridge regressors on the
    pool5 layer.  Reading fc7 instead puts two dropout-regularised
    non-linearities between the spatial map and a head whose whole job is to
    recover geometry.
    """

    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.fc6 = nn.Linear(feat_dim, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.drop = nn.Dropout(p=dropout)

        self.cls_head = nn.Linear(4096, num_classes + 1)
        self.bbox_head = nn.Linear(feat_dim, num_classes * 4)

    @override
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        pool5 = x
        x = F.relu(self.fc6(x))
        x = self.drop(x)
        x = F.relu(self.fc7(x))
        x = self.drop(x)
        return self.cls_head(x), self.bbox_head(pool5)


# ---------------------------------------------------------------------------
# R-CNN for Object Detection
# ---------------------------------------------------------------------------


class RCNNForObjectDetection(ObjectDetectionModel):
    r"""R-CNN object detector (Girshick et al., CVPR 2014).

    The first deep-learning detector to demonstrate that ImageNet-pretrained
    CNN features transfer to localisation.  At inference the pipeline runs an
    externally-supplied region-proposal set (typically from selective search)
    through an AlexNet-style convolutional trunk *per proposal*, then feeds
    the flattened pool5 features into a two-layer FC trunk with sibling
    classification (``cls_head``) and bounding-box regression (``bbox_head``)
    outputs.  This module covers the CNN + heads — proposals must be supplied
    by the caller; ``postprocess()`` applies score thresholding and per-class
    NMS to recover final detections.

    Parameters
    ----------
    config : RCNNConfig
        Frozen architecture spec.  Use the :func:`rcnn` factory for the
        paper-cited AlexNet configuration (227x227 RoIs, 80 COCO classes);
        override individual fields via ``**kwargs`` on the factory.

    Attributes
    ----------
    config : RCNNConfig
        Stored copy of the config that built this model.
    conv_features : _ConvFeatures
        AlexNet conv1-conv5 trunk applied to each warped RoI crop.  Output
        is a 9216-dim vector for the default ``roi_size=227``.
    fc_head : _FCHead
        Two-layer FC trunk (``fc6``, ``fc7``) plus sibling
        :class:`~lucid.nn.Linear` heads: ``cls_head`` reads fc7 and produces
        :math:`(N, K+1)` class logits (background + ``num_classes``
        foreground classes), while ``bbox_head`` reads the pool5 features
        directly (Appendix C) and produces :math:`(N, 4K)` class-specific
        bounding-box deltas.

    Notes
    -----
    See Girshick et al., "Rich Feature Hierarchies for Accurate Object
    Detection and Semantic Segmentation", CVPR 2014 (arXiv:1311.2524).
    For each proposal :math:`P` with centre :math:`(P_x, P_y)` and size
    :math:`(P_w, P_h)`, the bounding-box regressor predicts a
    parameterised offset :math:`(t_x, t_y, t_w, t_h)` so that the
    refined box centre and size are

    .. math::

        \begin{aligned}
            \hat{G}_x &= P_w t_x + P_x, & \hat{G}_y &= P_h t_y + P_y, \\
            \hat{G}_w &= P_w e^{t_w},   & \hat{G}_h &= P_h e^{t_h}.
        \end{aligned}

    The original paper trains per-class linear SVMs on top of frozen CNN
    features; this implementation follows the more common modern
    softmax-end-to-end form (matching the Caffe reference code).  Cost
    is dominated by the redundant per-proposal CNN forward — the main
    motivation for the Fast R-CNN successor's shared feature map.

    Examples
    --------
    Run inference on a single image with three external proposals:

    >>> import lucid
    >>> from lucid.models.vision.rcnn import rcnn
    >>> model = rcnn(num_classes=80)
    >>> x = lucid.randn(1, 3, 600, 600)
    >>> proposals = [lucid.tensor(
    ...     [[10.0,  10.0, 200.0, 200.0],
    ...      [50.0,  60.0, 300.0, 280.0],
    ...      [100.0, 80.0, 450.0, 400.0]])]
    >>> out = model(x, proposals)
    >>> out.logits.shape, out.pred_boxes.shape
    ((3, 81), (3, 80, 4))

    Apply post-processing (score threshold + per-class NMS) on the raw
    output to obtain final per-image detections:

    >>> detections = model.postprocess(out, proposals)
    >>> set(detections[0].keys()) == {"boxes", "scores", "labels"}
    True
    """

    config_class: ClassVar[type[RCNNConfig]] = RCNNConfig
    base_model_prefix: ClassVar[str] = "rcnn"

    def __init__(self, config: RCNNConfig) -> None:
        super().__init__(config)
        self._num_classes = config.num_classes
        self._roi_size = config.roi_size
        self._context_pad = config.context_pad
        self._score_thresh = config.score_thresh
        self._nms_thresh = config.nms_thresh
        self._max_det = config.max_detections

        self.conv_features = _ConvFeatures(config.in_channels, config.roi_size)
        self.fc_head = _FCHead(
            feat_dim=self.conv_features.out_dim,
            num_classes=config.num_classes,
            dropout=config.dropout,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _warp_proposals(
        self,
        img: Tensor,
        proposals: Tensor,
    ) -> Tensor:
        """Warp all proposals from one image into roi_size × roi_size crops.

        Args:
            img:       (1, C, H, W) single image.
            proposals: (N, 4) xyxy proposals in pixel coordinates.

        Returns:
            (N, C, roi_size, roi_size) warped crops.
        """
        H = int(img.shape[2])
        W = int(img.shape[3])
        S = self._roi_size
        pad = self._context_pad
        # Appendix A warps "with context": the box is dilated in *source* space
        # by however much puts exactly ``pad`` warped pixels of surroundings on
        # each side, so the box itself lands on the central S - 2*pad square.
        ctx = pad / (S - 2 * pad) if pad > 0 and S > 2 * pad else 0.0
        crops: list[Tensor] = []

        for bx1, by1, bx2, by2 in host_rows(proposals[:, :4]):
            if ctx > 0.0:
                dw, dh = (bx2 - bx1) * ctx, (by2 - by1) * ctx
                bx1, bx2 = bx1 - dw, bx2 + dw
                by1, by2 = by1 - dh, by2 + dh

            x1, y1 = int(math.floor(bx1)), int(math.floor(by1))
            x2, y2 = int(math.ceil(bx2)), int(math.ceil(by2))
            x2, y2 = max(x1 + 1, x2), max(y1 + 1, y2)

            # Clamp on *both* sides.  x1 was floored at 0 but never capped, so
            # a proposal lying entirely off the right/bottom edge gave a slice
            # start past the tensor and an empty crop, which then blew up in
            # the interpolate below.  The reference clips the box and rebuilds
            # a full crop window, zero-filling whatever falls outside — after
            # mean subtraction the reference's mean-pixel fill *is* zero.
            cx1 = min(max(0, x1), max(W - 1, 0))
            cy1 = min(max(0, y1), max(H - 1, 0))
            cx2 = max(cx1 + 1, min(W, x2))
            cy2 = max(cy1 + 1, min(H, y2))

            crop: Tensor = img[:, :, cy1:cy2, cx1:cx2]  # (1, C, h, w)
            left, top = max(0, cx1 - x1), max(0, cy1 - y1)
            right, bottom = max(0, x2 - cx2), max(0, y2 - cy2)
            if left or right or top or bottom:
                crop = F.pad(crop, (left, right, top, bottom))
            warped = F.interpolate(crop, size=(S, S), mode="bilinear")
            crops.append(warped)

        if not crops:
            C = int(img.shape[1])
            return lucid.zeros((0, C, S, S), device=img.device.type)
        return lucid.cat(crops, dim=0)

    def _decode_all_class_boxes(
        self,
        proposals: Tensor,
        bbox_deltas: Tensor,
        image_size: tuple[int, int],
    ) -> Tensor:
        """Decode every class's bbox deltas for each proposal.

        Args:
            proposals:   (N, 4) xyxy input proposals.
            bbox_deltas: (N, num_classes * 4) raw regression output.
            image_size:  (H, W) used for clipping.

        Returns:
            (N, K, 4) decoded xyxy boxes, one per foreground class.
        """
        N = int(proposals.shape[0])
        K = self._num_classes
        dev = proposals.device.type
        if N == 0:
            return lucid.zeros((0, K, 4), device=dev)

        # bbox_deltas: (N, K*4) → (N, K, 4)
        deltas = bbox_deltas.reshape(N, K, 4)

        # Paper Appendix C trains a SEPARATE regressor per class, so class c's
        # detections must use class c's four deltas.  Decoding only the argmax
        # class and reusing that one box for every class made the 4K-way head
        # dead weight for K-1 of the K classes — every class reported byte-
        # identical boxes.  Mirrors ``FastRCNN._decode_all_boxes``.
        per_class: list[Tensor] = []
        for c in range(K):
            boxes_c = decode_boxes(deltas[:, c, :], proposals)
            per_class.append(clip_boxes_to_image(boxes_c, image_size))
        return lucid.stack(per_class, dim=1)  # (N, K, 4)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _stage_losses(
        self,
        logits: Tensor,
        deltas: Tensor,
        proposals: Tensor,
        target: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Stage-1 and stage-3 losses for one image.

        Stage 1 (§2.3) is softmax cross-entropy over every proposal, with a
        proposal counted as its best-overlapping class at IoU >=
        ``fg_iou_thresh`` and as background otherwise.

        Stage 3 (Appendix C) trains the class-specific regressors, and only
        on proposals that already overlap a ground-truth box at IoU >
        ``bbox_reg_iou_thresh`` — the paper learns "from a proposal P only
        if it is near at least one ground-truth box", because a regressor
        asked to fix a box that shares nothing with the object has no
        signal to learn from.

        Returns:
            ``(classification_loss, regression_loss)``, both scalars.
        """
        cfg = cast(RCNNConfig, self.config)
        dev = logits.device.type
        gt_boxes = target["boxes"]
        gt_labels = target["labels"]
        n = int(proposals.shape[0])

        if int(gt_boxes.shape[0]) == 0 or n == 0:
            # No objects: every proposal is background and nothing regresses.
            zeros = lucid.zeros((n,), device=dev).long()
            return F.cross_entropy(logits, zeros), lucid.zeros((), device=dev)

        iou = box_iou(gt_boxes, proposals)  # (M, N)
        best_iou, best_gt = iou.max(dim=0)  # (N,)

        # Stage 1: class ids are 1-based, 0 is background.
        fg = best_iou >= cfg.fg_iou_thresh
        labels = lucid.where(
            fg, gt_labels[best_gt.long()], lucid.zeros_like(gt_labels[best_gt.long()])
        )
        cls_loss = F.cross_entropy(logits, labels.long())

        # Stage 3: a tighter threshold, and only the assigned class's four
        # deltas are trained.
        near = cast(list[float], (best_iou > cfg.bbox_reg_iou_thresh).float().tolist())
        keep = [i for i, v in enumerate(near) if v > 0.0]
        if not keep:
            return cls_loss, lucid.zeros((), device=dev)

        idx = lucid.tensor(keep, device=dev).long()
        props_k = proposals[idx]
        matched = gt_boxes[best_gt[idx].long()]
        reg_targets = encode_boxes(matched, props_k)

        k_classes = self._num_classes
        per_class = deltas[idx].reshape(len(keep), k_classes, 4)
        # Foreground labels are 1-based; class c's regressor is row c - 1.
        rows = cast(list[int], labels[idx].long().tolist())
        offsets = lucid.tensor(
            [i * k_classes + max(0, c - 1) for i, c in enumerate(rows)], device=dev
        ).long()
        pred = per_class.reshape(len(keep) * k_classes, 4)[offsets]
        reg_loss = F.smooth_l1_loss(pred, reg_targets, reduction="sum") / float(len(keep))
        return cls_loss, reg_loss

    @override
    def forward(
        self,
        x: Tensor,
        proposals: list[Tensor] | None = None,
        targets: list[dict[str, Tensor]] | None = None,
    ) -> ObjectDetectionOutput:
        """Run R-CNN on a batch of images.

        Args:
            x:         (B, C, H, W) image batch.
            proposals: list of B tensors, each (N_i, 4) xyxy region proposals.
                       Pass ``None`` to return empty detections.
            targets:   Optional per-image ``{"boxes": (M, 4) xyxy, "labels":
                       (M,) 1-based class ids}``.  When given, the stage-1
                       and stage-3 losses are computed; the stage-2 SVMs are
                       an offline pass and are not part of this model — see
                       :class:`RCNNConfig`.

        Returns:
            ``ObjectDetectionOutput`` with:
              ``logits``     : (Σ N_i, num_classes + 1) raw class logits.
              ``pred_boxes`` : (Σ N_i, 4) decoded xyxy boxes (top class).
              ``loss``       : scalar sum of the two terms, or ``None``.

        Raises:
            ValueError: If ``targets`` is given without ``proposals``, since
                R-CNN does not generate its own regions and there would be
                nothing to assign them to.
        """
        B = int(x.shape[0])
        H = int(x.shape[2])
        W = int(x.shape[3])
        dev = x.device.type

        if proposals is None:
            if targets is not None:
                raise ValueError(
                    "targets were supplied without proposals.  R-CNN does not "
                    "generate its own regions — the paper's pipeline feeds it "
                    "selective-search boxes — so there is nothing to assign "
                    "the targets to and the loss would be computed over an "
                    "empty set."
                )
            proposals = [lucid.zeros((0, 4), device=dev) for _ in range(B)]
        check_per_image(type(self).__name__, B, proposals=proposals, targets=targets)

        all_logits: list[Tensor] = []
        all_boxes: list[Tensor] = []
        loss_parts: list[Tensor] = []

        for b in range(B):
            props = proposals[b]  # (N_i, 4)
            N_i = int(props.shape[0])

            if N_i == 0:
                all_logits.append(lucid.zeros((0, self._num_classes + 1), device=dev))
                all_boxes.append(lucid.zeros((0, 4), device=dev))
                continue

            img = x[b : b + 1]  # (1, C, H, W)

            # 1. Warp proposals → fixed-size crops
            crops = self._warp_proposals(img, props)  # (N_i, C, S, S)

            # 2. Extract CNN features
            feats = self.conv_features(crops)  # (N_i, 9216)

            # 3. FC head → class logits + bbox deltas
            logits_i, deltas_i = self.fc_head(feats)
            # logits_i — (N_i, num_classes + 1)
            # deltas_i — (N_i, num_classes * 4)

            # 4. Decode one box per class (class-specific regression)
            boxes_i = self._decode_all_class_boxes(props, deltas_i, (H, W))

            all_logits.append(logits_i)
            all_boxes.append(boxes_i)

            if targets is not None:
                cls_loss, reg_loss = self._stage_losses(logits_i, deltas_i, props, targets[b])
                loss_parts.append((cls_loss + reg_loss).reshape(1))

        # Concatenate across batch (flat)
        if all_logits:
            final_logits = lucid.cat(all_logits, dim=0)
            final_boxes = lucid.cat(all_boxes, dim=0)
        else:
            final_logits = lucid.zeros((0, self._num_classes + 1), device=dev)
            final_boxes = lucid.zeros((0, 4), device=dev)

        loss: Tensor | None = None
        if targets is not None:
            loss = (
                lucid.cat(loss_parts).sum() / float(B)
                if loss_parts
                else lucid.zeros((), device=dev)
            )

        return ObjectDetectionOutput(
            logits=final_logits,
            pred_boxes=final_boxes,
            loss=loss,
            proposals=tuple(proposals),
        )

    # ------------------------------------------------------------------
    # Post-processing helper (call separately after forward)
    # ------------------------------------------------------------------

    def postprocess(
        self,
        output: ObjectDetectionOutput,
        proposals: list[Tensor] | None = None,
    ) -> list[dict[str, Tensor]]:
        """Apply score threshold + per-class NMS to raw R-CNN output.

        Args:
            output:    ``ObjectDetectionOutput`` from ``forward()``.
            proposals: Same proposal list passed to ``forward()``.  When
                       omitted, falls back to ``output.proposals``.

        Returns:
            List of per-image result dicts, each with:
              ``"boxes"``  : (K, 4) xyxy kept detections
              ``"scores"`` : (K,)   class confidence scores
              ``"labels"`` : (K,)   integer class indices (1-based, 0=bg)
        """
        if proposals is None:
            if output.proposals is None:
                raise ValueError(
                    "postprocess() needs proposals — pass them explicitly or "
                    "use the output produced by a forward() call."
                )
            proposals = list(output.proposals)
        logits = output.logits  # (Σ N_i, num_classes + 1)
        pred_boxes = output.pred_boxes  # (Σ N_i, 4)

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

            # Per-class NMS (skip background class 0)
            for c in range(1, self._num_classes + 1):
                cls_scores = scores_i[:, c]  # (N_i,)

                # Score threshold
                mask: list[int] = [
                    i for i, row in enumerate(host_scores) if row[c] >= self._score_thresh
                ]
                if not mask:
                    continue

                mask_t = lucid.tensor(mask, device=dev).long()
                sc_c = cls_scores[mask_t]
                bx_c = bx_i[mask_t][:, c - 1, :]  # class-c regressor

                # NMS
                keep = batched_nms(
                    bx_c,
                    sc_c,
                    lucid.zeros(int(sc_c.shape[0]), device=dev),  # single class → no offset
                    self._nms_thresh,
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
