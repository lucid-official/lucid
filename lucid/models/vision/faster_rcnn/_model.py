"""Faster R-CNN object detector (Ren et al., NeurIPS 2015).

Paper: "Faster R-CNN: Towards Real-Time Object Detection with
        Region Proposal Networks"

This module implements the **ResNet-50-FPN** two-stage detector — the
modern reference configuration shipped with the COCO ``box AP 37.0``
checkpoint — rather than the original VGG16 single-scale design.  The
submodule layout mirrors the reference detector verbatim so the COCO
checkpoint loads strict:

  Image (B, C, H, W)
    ↓  ResNet-50 backbone (frozen BN, eps=0) → C2, C3, C4, C5
    ↓  FPN: 1×1 lateral (``inner_blocks``) + top-down nearest add +
           3×3 output (``layer_blocks``) + LastLevelMaxPool ("pool")
  [P2, P3, P4, P5, pool]
    ├─ RPN head (3×3 conv + ReLU → cls_logits + bbox_pred), 3 anchors/cell
    │    sizes ((32,),(64,),(128,),(256,),(512,)), ratios (0.5,1,2)
    │    → per-level top-k → decode → clip → NMS 0.7 → top-1000 proposals
    │
    └─ MultiScale RoI Align (7×7, sampling_ratio=2, aligned=False) over P2-P5
         ↓  FPN level-assignment k = floor(4 + log2(sqrt(wh)/224)), k∈[2,5]
       TwoMLPHead: fc6 (256·7·7 → 1024) → fc7 (1024 → 1024)
         ↓
       FastRCNNPredictor: cls_score (1024 → 91) + bbox_pred (1024 → 91·4)
         ↓  softmax, per-class decode (10,10,5,5), clip, NMS 0.5, top-100

Faithfulness notes
------------------
* Backbone batch-norm is **frozen** (eval-only affine + running-stat math,
  no ``num_batches_tracked``) with ``eps = 0`` matching the reference
  detection ``FrozenBatchNorm2d``.
* FPN adds a parameter-free ``LastLevelMaxPool`` 5th level fed only to RPN.
* RPN box-coder weights are ``(1, 1, 1, 1)``; RoI box-coder weights are
  ``(10, 10, 5, 5)``.
* The detector accepts an already-resized + normalised image batch (the
  reference ``GeneralizedRCNNTransform`` normalisation / resize is a
  :class:`~lucid.utils.transforms.Detection` preset that runs outside the
  model).  ``forward`` returns raw RoI-head outputs and ``postprocess``
  produces the final per-image detections.
"""

from typing import ClassVar, final, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._tasks import ObjectDetectionModel
from lucid.models._output import ObjectDetectionOutput
from lucid.models._utils._detection import (
    BalancedPositiveNegativeSampler,
    Matcher,
    _FeaturePyramidNetwork,
    _ReferenceAnchorGenerator,
    _ResNetBody,
    assign_anchors_to_targets,
    batched_nms,
    clip_boxes_to_image,
    decode_boxes,
    fastrcnn_loss,
    flatten_rpn_outputs,
    multiscale_roi_align,
    nms,
    remove_small_boxes,
    rpn_loss,
    select_training_samples,
)
from lucid.models._utils._detection_results import (
    check_per_image,
    detection_result,
    empty_detection_result,
)
from lucid.models.vision.faster_rcnn._config import FasterRCNNConfig
from lucid.models._utils._common import host_floats, host_rows


def _level_strides(image_height: int, features: list[Tensor]) -> list[int]:
    """Stride of each pyramid level as ``image_height // level_height``.

    The reference anchor generator spaces its grid the same way.  The
    quotients are the real strides for P2 to P5 on a canvas that is a
    multiple of 32, and for the pool level too on a multiple of 64; on
    1333 they fall short (3, 7, 15, 31 and 63), which is why the R-CNN
    weights' :class:`~lucid.utils.transforms.Detection` presets round their
    canvas up to 1344.  Mask R-CNN imports this, so the RPN and box branch
    the two detectors share derive identical strides.
    """
    return [max(1, image_height // int(f.shape[2])) for f in features]


# ---------------------------------------------------------------------------
# Backbone with FPN  (key prefix: ``backbone.body.*`` / ``backbone.fpn.*``)
# ---------------------------------------------------------------------------


@final
class _BackboneWithFPN(nn.Module):
    """ResNet trunk (``body``) feeding a Feature Pyramid Network (``fpn``).

    Submodule names mirror the reference so the converter is a pure
    identity map for every backbone key.
    """

    def __init__(
        self,
        in_channels: int,
        layers: tuple[int, int, int, int],
        fpn_out_channels: int,
        bn_eps: float,
    ) -> None:
        super().__init__()
        self.body = _ResNetBody(in_channels, layers, bn_eps=bn_eps)
        self.fpn = _FeaturePyramidNetwork(self.body.out_channels_list, fpn_out_channels)
        self.out_channels = fpn_out_channels

    @override
    def forward(self, x: Tensor) -> list[Tensor]:
        c_feats = self.body(x)  # [C2, C3, C4, C5]
        return self.fpn.forward(c_feats)  # [P2, P3, P4, P5, pool]


# ---------------------------------------------------------------------------
# RPN  (key prefix: ``rpn.head.conv.0.0`` / ``rpn.head.cls_logits`` / ...)
# ---------------------------------------------------------------------------


@final
class _RPNHead(nn.Module):
    """RPN head: shared 3×3 conv + ReLU, then cls / bbox 1×1 sibling convs.

    The conv is wrapped in a ``Sequential`` of a ``Sequential`` so the
    state-dict keys read ``conv.0.0.weight`` (reference
    ``Conv2dNormActivation`` with no norm), matching exactly.
    """

    def __init__(self, in_channels: int, num_anchors: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1)))
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, 1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, 1)

    @override
    def forward(self, features: list[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        logits: list[Tensor] = []
        bbox: list[Tensor] = []
        for feat in features:
            t = F.relu(self.conv(feat))
            logits.append(self.cls_logits(t))
            bbox.append(self.bbox_pred(t))
        return logits, bbox


@final
class _RegionProposalNetwork(nn.Module):
    """Region Proposal Network — ``head`` + anchor-based proposal layer.

    Holds only the parametric ``head``; the anchor generator and proposal
    filtering are stateless helpers driven by the parent config.
    """

    def __init__(self, in_channels: int, num_anchors: int) -> None:
        super().__init__()
        self.head = _RPNHead(in_channels, num_anchors)


# ---------------------------------------------------------------------------
# RoI heads  (key prefix: ``roi_heads.box_head.*`` / ``roi_heads.box_predictor.*``)
# ---------------------------------------------------------------------------


@final
class _TwoMLPHead(nn.Module):
    """Reference ``TwoMLPHead``: flatten → fc6 → ReLU → fc7 → ReLU."""

    def __init__(self, in_channels: int, representation_size: int) -> None:
        super().__init__()
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(1)
        x = F.relu(self.fc6(x))
        return F.relu(self.fc7(x))


@final
class _FastRCNNPredictor(nn.Module):
    """Reference ``FastRCNNPredictor``: sibling cls_score + bbox_pred."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    @override
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.cls_score(x), self.bbox_pred(x)


@final
class _RoIHeads(nn.Module):
    """RoI heads container — ``box_head`` (TwoMLPHead) + ``box_predictor``."""

    def __init__(
        self, in_channels: int, roi_size: int, representation_size: int, num_classes: int
    ) -> None:
        super().__init__()
        self.box_head = _TwoMLPHead(in_channels * roi_size * roi_size, representation_size)
        self.box_predictor = _FastRCNNPredictor(representation_size, num_classes)

    @override
    def forward(self, roi_feats: Tensor) -> tuple[Tensor, Tensor]:
        feats = self.box_head(roi_feats)
        return self.box_predictor(feats)


# ---------------------------------------------------------------------------
# Faster R-CNN
# ---------------------------------------------------------------------------


class FasterRCNNForObjectDetection(ObjectDetectionModel):
    r"""Faster R-CNN with a ResNet-50-FPN backbone (Ren et al., NeurIPS 2015).

    The two-stage anchor-based detector in its modern reference
    configuration: a ResNet-50 trunk with frozen batch-norm feeds a
    Feature Pyramid Network, a Region Proposal Network emits per-image
    proposals from five pyramid levels, and a Fast R-CNN-style RoI head
    classifies + refines RoI-aligned crops.  The submodule layout mirrors
    the reference detector so the COCO ``box AP 37.0`` checkpoint loads
    strict and reproduces inference.

    Parameters
    ----------
    config : FasterRCNNConfig
        Frozen architecture spec.  Use the :func:`faster_rcnn_resnet50_fpn`
        factory for the COCO-pretrained configuration (``num_classes = 91``).

    Attributes
    ----------
    config : FasterRCNNConfig
        Stored copy of the config that built this model.
    backbone : _BackboneWithFPN
        ResNet-50 ``body`` + ``fpn`` producing five feature maps
        ``[P2, P3, P4, P5, pool]`` at strides ``4/8/16/32/64``.
    rpn : _RegionProposalNetwork
        Proposal head shared across all pyramid levels.
    roi_heads : _RoIHeads
        ``box_head`` (TwoMLPHead) + ``box_predictor`` (FastRCNNPredictor).

    Notes
    -----
    See Ren et al., "Faster R-CNN: Towards Real-Time Object Detection with
    Region Proposal Networks", NeurIPS 2015 (arXiv:1506.01497) and Lin et
    al., "Feature Pyramid Networks for Object Detection", CVPR 2017.  The
    model expects an already resized + normalised image batch; final
    detections come from :meth:`postprocess`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.faster_rcnn import faster_rcnn_resnet50_fpn
    >>> model = faster_rcnn_resnet50_fpn().eval()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape[-1]   # num_classes
    91
    >>> dets = model.postprocess(out, image_sizes=[(224, 224)])
    >>> sorted(dets[0].keys())
    ['boxes', 'labels', 'scores']
    """

    config_class: ClassVar[type[FasterRCNNConfig]] = FasterRCNNConfig
    base_model_prefix: ClassVar[str] = "faster_rcnn"

    # Nominal FPN level strides (P2..P5 then the pool level) for a size that
    # divides evenly.  ``forward`` derives the real ones from the feature maps
    # rather than trusting these.
    _strides: ClassVar[tuple[int, ...]] = (4, 8, 16, 32, 64)

    def __init__(self, config: FasterRCNNConfig) -> None:
        super().__init__(config)
        self._cfg = config

        self.backbone = _BackboneWithFPN(
            in_channels=config.in_channels,
            layers=config.backbone_layers,
            fpn_out_channels=config.fpn_out_channels,
            bn_eps=config.backbone_bn_eps,
        )
        C = self.backbone.out_channels

        num_anchors = len(config.rpn_anchor_ratios)  # 1 scale per level x ratios
        self.rpn = _RegionProposalNetwork(C, num_anchors)

        # One anchor size per FPN level (5 levels including pool).
        sizes: tuple[tuple[int, ...], ...] = tuple((s,) for s in config.rpn_anchor_sizes)
        ratios: tuple[tuple[float, ...], ...] = tuple(
            tuple(config.rpn_anchor_ratios) for _ in config.rpn_anchor_sizes
        )
        self._anchor_gen = _ReferenceAnchorGenerator(sizes, ratios)

        self.roi_heads = _RoIHeads(
            in_channels=C,
            roi_size=config.roi_size,
            representation_size=config.roi_representation_size,
            num_classes=config.num_classes,
        )

    # ------------------------------------------------------------------
    # RPN proposal generation (inference)
    # ------------------------------------------------------------------

    def _rpn_proposals(
        self,
        logits: list[Tensor],
        deltas: list[Tensor],
        anchors: list[Tensor],
        image_size: tuple[int, int],
    ) -> list[Tensor]:
        """Decode + filter RPN predictions into per-image proposals.

        Mirrors the reference ``filter_proposals``: per-level top-k pre-NMS
        selection, decode, clip, small-box removal, score threshold, then
        per-level (class-offset) NMS and a post-NMS top-k.
        """
        B = int(logits[0].shape[0])
        cfg = self._cfg
        dev = logits[0].device.type

        per_level_scores, per_level_deltas = flatten_rpn_outputs(logits, deltas)

        proposals: list[Tensor] = []
        for b in range(B):
            boxes_parts: list[Tensor] = []
            scores_parts: list[Tensor] = []
            levels_parts: list[Tensor] = []
            for lvl in range(len(logits)):
                sc_b = per_level_scores[lvl][b]  # (N_l,)
                dl_b = per_level_deltas[lvl][b]  # (N_l, 4)
                anc = anchors[lvl]  # (N_l, 4)

                K = min(cfg.rpn_pre_nms_top_n, int(sc_b.shape[0]))
                top_idx = lucid.argsort(-sc_b)[:K]
                sc_t = sc_b[top_idx]
                dl_t = dl_b[top_idx]
                anc_t = anc[top_idx]

                # Proposals are *inputs* to the second stage, not a
                # differentiable path back into the RPN: the reference
                # detaches them so the RoI head's loss trains the RoI head
                # alone.  Without this the box head's gradient reaches
                # rpn.bbox_pred through the coordinates it was handed.
                props = decode_boxes(dl_t.detach(), anc_t, (1.0, 1.0, 1.0, 1.0))
                props = clip_boxes_to_image(props, image_size)
                boxes_parts.append(props)
                scores_parts.append(F.sigmoid(sc_t))
                levels_parts.append(lucid.full((int(sc_t.shape[0]),), float(lvl), device=dev))

            boxes_b = lucid.cat(boxes_parts, dim=0)
            scores_b = lucid.cat(scores_parts, dim=0)
            lvls_b = lucid.cat(levels_parts, dim=0)

            keep_small = remove_small_boxes(boxes_b, cfg.rpn_min_size)
            if int(keep_small.shape[0]) == 0:
                proposals.append(lucid.zeros((0, 4), device=dev))
                continue
            boxes_b = boxes_b[keep_small]
            scores_b = scores_b[keep_small]
            lvls_b = lvls_b[keep_small]

            if cfg.rpn_score_thresh > 0.0:
                mask = [
                    i
                    for i, score in enumerate(host_floats(scores_b))
                    if score >= cfg.rpn_score_thresh
                ]
                if not mask:
                    proposals.append(lucid.zeros((0, 4), device=dev))
                    continue
                m_t = lucid.tensor(mask, device=dev).long()
                boxes_b = boxes_b[m_t]
                scores_b = scores_b[m_t]
                lvls_b = lvls_b[m_t]

            keep = batched_nms(boxes_b, scores_b, lvls_b, cfg.rpn_nms_thresh)
            keep = keep[: cfg.rpn_post_nms_top_n]
            proposals.append(boxes_b[keep])

        return proposals

    # ------------------------------------------------------------------
    # Training (§3.1.2 assignment, §3.1.3 sampling, four-term loss)
    # ------------------------------------------------------------------

    def _matchers(self) -> tuple[Matcher, Matcher]:
        """``(rpn_matcher, roi_matcher)`` built from this model's config."""
        cfg = self._cfg
        return (
            Matcher(cfg.rpn_fg_iou_thresh, cfg.rpn_bg_iou_thresh, allow_low_quality_matches=True),
            Matcher(cfg.roi_fg_iou_thresh, cfg.roi_bg_iou_thresh),
        )

    def _assign_anchors(
        self, anchors: Tensor, gt_boxes: Tensor, matcher: Matcher, image_size: tuple[int, int]
    ) -> tuple[Tensor, Tensor]:
        return assign_anchors_to_targets(
            anchors,
            gt_boxes,
            matcher,
            image_size,
            ignore_cross_boundary=self._cfg.rpn_ignore_cross_boundary,
        )

    def _rpn_loss(
        self,
        logits: list[Tensor],
        deltas: list[Tensor],
        anchors: list[Tensor],
        targets: list[dict[str, Tensor]],
        image_size: tuple[int, int],
    ) -> tuple[Tensor, Tensor]:
        cfg = self._cfg
        rpn_matcher, _ = self._matchers()
        return rpn_loss(
            logits,
            deltas,
            anchors,
            targets,
            rpn_matcher,
            BalancedPositiveNegativeSampler(
                cfg.rpn_batch_size_per_image, cfg.rpn_positive_fraction
            ),
            image_size,
            ignore_cross_boundary=cfg.rpn_ignore_cross_boundary,
        )

    def _select_training_samples(
        self, proposals: list[Tensor], targets: list[dict[str, Tensor]]
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]]:
        cfg = self._cfg
        _, roi_matcher = self._matchers()
        return select_training_samples(
            proposals,
            targets,
            roi_matcher,
            BalancedPositiveNegativeSampler(
                cfg.roi_batch_size_per_image, cfg.roi_positive_fraction
            ),
            cfg.bbox_reg_weights,
        )

    def _roi_loss(
        self,
        class_logits: Tensor,
        box_deltas: Tensor,
        labels: list[Tensor],
        reg_targets: list[Tensor],
    ) -> tuple[Tensor, Tensor]:
        return fastrcnn_loss(class_logits, box_deltas, labels, reg_targets)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @override
    def forward(
        self,
        x: Tensor,
        targets: list[dict[str, Tensor]] | None = None,
        proposals: list[Tensor] | None = None,
    ) -> ObjectDetectionOutput:
        """Run Faster R-CNN on a (pre-processed) image batch.

        Args:
            x:         (B, C, H, W) resized + normalised image batch.
            targets:   Optional per-image ``{"boxes": (M, 4) xyxy pixel
                       boxes, "labels": (M,) class ids in 1..K-1}``.  When
                       given, the four-term training loss is computed and
                       the box head runs on a sampled minibatch rather than
                       on every proposal.
            proposals: Optional precomputed per-image proposals.  When
                       ``None`` the RPN generates them.

        Returns:
            ``ObjectDetectionOutput`` with raw RoI-head outputs:
              ``logits``     : (Σ proposals, num_classes) class logits.
              ``pred_boxes`` : (Σ proposals, num_classes, 4) per-class boxes.
              ``proposals``  : per-image proposal tensors.
              ``loss``       : scalar sum of the four terms, or ``None``.

        Raises:
            ValueError: If ``targets`` is given but the RPN did not run,
                since the RPN half of the loss would then be undefined.
        """
        B, _, iH, iW = (int(d) for d in x.shape)
        dev = x.device.type
        check_per_image(type(self).__name__, B, proposals=proposals, targets=targets)

        # 1. Backbone + FPN → [P2, P3, P4, P5, pool]
        features = self.backbone(x)

        # Derive the level strides from what the backbone actually produced.
        # The class constant (4, 8, 16, 32, 64) only holds for input sizes
        # divisible all the way down; anywhere else the real ratio differs and
        # both the anchor grid and the RoI-Align scales silently drift off the
        # feature map they are addressing.
        strides = _level_strides(iH, features)

        # 2. RPN → per-image proposals (when not supplied)
        rpn_obj_loss: Tensor | None = None
        rpn_reg_loss: Tensor | None = None
        if proposals is None:
            logits, deltas = self.rpn.head.forward(features)
            anchors = self._anchor_gen.forward(features, strides)
            proposals = self._rpn_proposals(logits, deltas, anchors, (iH, iW))
            if targets is not None:
                rpn_obj_loss, rpn_reg_loss = self._rpn_loss(
                    logits, deltas, anchors, targets, (iH, iW)
                )
        elif targets is not None:
            raise ValueError(
                "targets were supplied together with precomputed proposals, "
                "but the RPN half of the loss is only defined when the RPN "
                "actually ran.  Returning just the box-head terms would "
                "silently train half the detector.  Omit `proposals` to "
                "train end to end, or omit `targets` to run inference on "
                "your own proposals."
            )

        # 3. Training: sample the box head's minibatch *before* RoI Align, so
        #    the expensive crop runs only on the RoIs that reach the loss.
        roi_labels: list[Tensor] | None = None
        roi_reg_targets: list[Tensor] | None = None
        if targets is not None:
            proposals, roi_labels, roi_reg_targets, _ = self._select_training_samples(
                proposals, targets
            )

        # 4. MultiScale RoI Align over the four FPN detection levels (P2-P5).
        det_feats = features[:4]
        det_scales = [1.0 / float(s) for s in strides[:4]]
        roi_feats = multiscale_roi_align(
            det_feats,
            proposals,
            output_size=self._cfg.roi_size,
            spatial_scales=det_scales,
            sampling_ratio=self._cfg.roi_sampling_ratio,
            canonical_scale=self._cfg.canonical_scale,
            canonical_level=self._cfg.canonical_level,
        )

        # 5. RoI head → class logits + per-class box deltas
        K = self._cfg.num_classes
        if int(roi_feats.shape[0]) > 0:
            class_logits, box_deltas = self.roi_heads(roi_feats)
        else:
            class_logits = lucid.zeros((0, K), device=dev)
            box_deltas = lucid.zeros((0, K * 4), device=dev)

        # 6. Decode per-class boxes → (N, K, 4)
        pred_boxes = self._decode_per_class(proposals, box_deltas, (iH, iW))

        loss: Tensor | None = None
        if targets is not None:
            assert roi_labels is not None and roi_reg_targets is not None
            roi_cls_loss, roi_reg_loss = self._roi_loss(
                class_logits, box_deltas, roi_labels, roi_reg_targets
            )
            assert rpn_obj_loss is not None and rpn_reg_loss is not None
            loss = rpn_obj_loss + rpn_reg_loss + roi_cls_loss + roi_reg_loss

        return ObjectDetectionOutput(
            logits=class_logits, pred_boxes=pred_boxes, loss=loss, proposals=tuple(proposals)
        )

    def _decode_per_class(
        self, proposals: list[Tensor], box_deltas: Tensor, image_size: tuple[int, int]
    ) -> Tensor:
        """Decode ``(N, K*4)`` deltas against proposals → ``(N, K, 4)`` boxes."""
        dev = box_deltas.device.type
        K = self._cfg.num_classes
        N = int(box_deltas.shape[0])
        if N == 0:
            return lucid.zeros((0, K, 4), device=dev)
        flat_props = lucid.cat([p for p in proposals if int(p.shape[0]) > 0], dim=0)
        deltas_3d = box_deltas.reshape(N, K, 4)
        per_class: list[Tensor] = []
        for c in range(K):
            boxes_c = decode_boxes(deltas_3d[:, c, :], flat_props, self._cfg.bbox_reg_weights)
            per_class.append(clip_boxes_to_image(boxes_c, image_size))
        return lucid.stack(per_class, dim=1)

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def postprocess(
        self,
        output: ObjectDetectionOutput,
        image_sizes: list[tuple[int, int]] | None = None,
        proposals: list[Tensor] | None = None,
    ) -> list[dict[str, Tensor]]:
        """Softmax → per-class score filter → per-class NMS → top-k.

        Mirrors the reference ``postprocess_detections``: drops the
        background class (slot 0), score-thresholds, removes empty boxes,
        runs per-class NMS, and keeps the top ``max_detections`` scores.

        Parameters
        ----------
        output : ObjectDetectionOutput
            Raw RoI-head outputs from :meth:`forward`.
        image_sizes : list of (H, W), optional
            Per-image extent.  A batch of differently-sized images is padded
            to one canvas, and decoding clips to *that* canvas — so without
            this a detection could legally extend into another image's
            padding.  When given, image ``i``'s boxes are re-clipped to
            ``image_sizes[i]``.
        proposals : list of Tensor, optional
            Per-image proposals; falls back to ``output.proposals``.

        Returns
        -------
        list of dict
            One dict per image with ``"boxes"`` ``(D, 4)``, ``"scores"``
            ``(D,)``, ``"labels"`` ``(D,)`` int64.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models import create_model
        >>> lucid.manual_seed(0)
        >>> model = create_model(
        ...     "faster_rcnn_resnet50_fpn", num_classes=3,
        ...     backbone_layers=(1, 1, 1, 1), fpn_out_channels=32,
        ...     roi_representation_size=64, rpn_post_nms_top_n=10,
        ... ).eval()
        >>> out = model(lucid.randn((1, 3, 64, 64)))
        >>> det = model.postprocess(out)[0]
        >>> sorted(det)
        ['boxes', 'labels', 'scores']

        The background slot never comes back, and detections are ranked by
        score.

        >>> bool((det["labels"] >= 1).all())
        True
        >>> scores = det["scores"]
        >>> bool((scores[:-1] >= scores[1:]).all())
        True

        Decoding clips to the 64-pixel canvas the batch ran on.  Say the
        real image is 48 tall and 40 wide — as it would be inside a padded
        batch — and boxes that reached into the padding are pulled back.

        >>> xs = det["boxes"][:, 0::2]  # x1 and x2 of each xyxy box
        >>> bool((xs <= 40).all())
        False
        >>> clipped = model.postprocess(out, image_sizes=[(48, 40)])[0]["boxes"]
        >>> xs, ys = clipped[:, 0::2], clipped[:, 1::2]
        >>> bool((xs <= 40).all()), bool((ys <= 48).all())
        (True, True)
        """
        if proposals is None:
            if output.proposals is None:
                raise ValueError(
                    "postprocess() needs proposals — pass them explicitly or "
                    "call forward() first so the output carries them."
                )
            proposals = list(output.proposals)

        logits = output.logits
        pred_boxes = output.pred_boxes  # (N, K, 4)
        cfg = self._cfg
        dev = logits.device.type
        results: list[dict[str, Tensor]] = []
        offset = 0

        for _img_idx, props in enumerate(proposals):
            N_i = int(props.shape[0])
            lg_i = logits[offset : offset + N_i]
            bx_i = pred_boxes[offset : offset + N_i]
            offset += N_i

            if N_i == 0:
                results.append(empty_detection_result(dev))
                continue

            scores_i = F.softmax(lg_i, dim=-1)  # (N_i, K)
            host_scores = host_rows(scores_i)
            keep_boxes: list[Tensor] = []
            keep_scores: list[Tensor] = []
            keep_labels: list[Tensor] = []

            for c in range(1, cfg.num_classes):  # skip background slot 0
                bx_class = bx_i[:, c, :]  # (N_i, 4)
                mask = [i for i, row in enumerate(host_scores) if row[c] > cfg.score_thresh]
                if not mask:
                    continue
                mask_t = lucid.tensor(mask, device=dev).long()
                sc_c = scores_i[:, c][mask_t]
                bx_c = bx_class[mask_t]
                # The reference drops degenerate boxes before NMS
                # (``remove_small_boxes(min_size=1e-2)``).  A zero-area box
                # has zero IoU with everything, so NMS cannot suppress it and
                # it survives into the final detections.
                keep_sz = remove_small_boxes(bx_c, 1e-2)
                if int(keep_sz.shape[0]) == 0:
                    continue
                bx_c = bx_c[keep_sz]
                sc_c = sc_c[keep_sz]
                keep = nms(bx_c, sc_c, cfg.nms_thresh)
                keep_boxes.append(bx_c[keep])
                keep_scores.append(sc_c[keep])
                keep_labels.append(
                    lucid.full((int(keep.shape[0]),), c, dtype=lucid.int64, device=dev)
                )

            if not keep_boxes:
                results.append(empty_detection_result(dev))
                continue

            all_b = lucid.cat(keep_boxes, dim=0)
            all_s = lucid.cat(keep_scores, dim=0)
            all_l = lucid.cat(keep_labels, dim=0)
            order = lucid.argsort(-all_s)[: cfg.max_detections]
            kept_boxes = all_b[order]
            if image_sizes is not None and _img_idx < len(image_sizes):
                kept_boxes = clip_boxes_to_image(kept_boxes, image_sizes[_img_idx])
            results.append(detection_result(kept_boxes, all_s[order], all_l[order]))
        return results
