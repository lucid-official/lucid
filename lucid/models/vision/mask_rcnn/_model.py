"""Mask R-CNN instance segmentation model (He et al., ICCV 2017).

Paper: "Mask R-CNN"

This module implements the **ResNet-50-FPN** instance-segmentation detector
— the modern reference configuration shipped with the COCO ``box AP 37.9 /
mask AP 34.6`` checkpoint.  Mask R-CNN is Faster R-CNN plus a parallel mask
branch, so the entire backbone + FPN + RPN + box-head stack is **reused
verbatim** from :mod:`lucid.models.vision.faster_rcnn`; only the mask branch
on ``roi_heads`` is new.  The submodule layout mirrors the reference detector
so the COCO checkpoint loads strict (307 keys = Faster R-CNN's 295 + 12
mask-branch keys):

  Image (B, C, H, W)
    ↓  ResNet-50 backbone (frozen BN, eps=0) → C2, C3, C4, C5
    ↓  FPN: 1×1 lateral + top-down nearest add + 3×3 output + LastLevelMaxPool
  [P2, P3, P4, P5, pool]
    ├─ RPN head → per-level top-k → decode → clip → NMS 0.7 → 1000 proposals
    │
    ├─ MultiScale RoI Align (7×7) → TwoMLPHead → FastRCNNPredictor
    │    → softmax, per-class decode, clip, NMS 0.5, top-100 detections
    │
    └─ (on the kept detections)
       MultiScale RoI Align (14×14) over P2-P5 (same FPN level assignment)
         ↓  MaskRCNNHeads: 4 × (Conv3×3(256→256, pad 1) + ReLU)
         ↓  MaskRCNNPredictor: ConvTranspose2d 2×2 s2 (256→256) + ReLU
         ↓                     → Conv1×1 (256→num_classes)
       per-detection mask logits (N, num_classes, 28, 28) — gather the
       predicted class channel and sigmoid for the final per-instance mask.

Faithfulness notes
------------------
* Backbone / FPN / RPN / box-head are byte-identical to the shipped
  Faster R-CNN — the shared 295 keys map the same way.
* The mask branch's RoI Align uses ``output_size = 14`` (vs 7 for the box
  head), ``sampling_ratio = 2``, ``aligned = False``, and the **same**
  canonical FPN level assignment as the box head.
* ``mask_head`` is a ``MaskRCNNHeads`` — four blocks, each a
  ``Sequential(Conv2d 3×3, ReLU)`` so the keys read
  ``roi_heads.mask_head.{i}.0.weight``.
* ``mask_predictor`` is a ``MaskRCNNPredictor`` — ``conv5_mask``
  (ConvTranspose2d 2×2 stride 2) + ReLU + ``mask_fcn_logits`` (Conv1×1).
* The detector accepts an already-resized + normalised image batch (the
  reference ``GeneralizedRCNNTransform`` normalisation / resize is a
  :class:`~lucid.utils.transforms.Detection` preset that runs outside the
  model).
"""

from typing import ClassVar, cast, final, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._tasks import ObjectDetectionModel
from lucid.models._output import InstanceSegmentationOutput
from lucid.models._utils._detection import (
    BalancedPositiveNegativeSampler,
    Matcher,
    paste_masks_in_image,
    remove_small_boxes,
    _ReferenceAnchorGenerator,
    clip_boxes_to_image,
    fastrcnn_loss,
    maskrcnn_loss,
    multiscale_roi_align,
    nms,
    project_masks_on_boxes,
    rpn_loss,
    select_training_samples,
)
from lucid.models._utils._detection_results import (
    check_per_image,
    detection_result,
    empty_detection_result,
)
from lucid.models.vision.faster_rcnn._model import (
    _BackboneWithFPN,
    _FastRCNNPredictor,
    _RegionProposalNetwork,
    _TwoMLPHead,
    _level_strides,
)
from lucid.models.vision.faster_rcnn._model import (
    FasterRCNNForObjectDetection as _FasterRCNNForObjectDetection,
)
from lucid.models.vision.mask_rcnn._config import MaskRCNNConfig
from lucid.models._utils._common import host_rows

# ---------------------------------------------------------------------------
# Mask branch building blocks
# ---------------------------------------------------------------------------
# Key prefixes mirror the reference detector verbatim:
#   roi_heads.mask_head.{i}.0.{weight,bias}        (MaskRCNNHeads)
#   roi_heads.mask_predictor.conv5_mask.{weight,bias}
#   roi_heads.mask_predictor.mask_fcn_logits.{weight,bias}


@final
class _MaskRCNNHeads(nn.Sequential):
    """Reference ``MaskRCNNHeads``: four ``Conv3×3 + ReLU`` blocks.

    Subclasses ``Sequential`` (like the reference) so each block is a
    direct integer child; every block is itself a
    ``Sequential(Conv2d, ReLU)`` (the reference ``Conv2dNormActivation``
    with no norm), so the conv is index ``.0`` and the state-dict keys
    read ``mask_head.{i}.0.weight`` / ``mask_head.{i}.0.bias`` (the ReLU
    at ``.1`` is parameter-free and contributes no key).
    """

    def __init__(self, in_channels: int, hidden_channels: int, num_convs: int) -> None:
        blocks: list[nn.Module] = []
        ch_in = in_channels
        for _ in range(num_convs):
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(ch_in, hidden_channels, 3, padding=1), nn.ReLU(inplace=True)
                )
            )
            ch_in = hidden_channels
        super().__init__(*blocks)

        # The reference ends both mask modules with a kaiming *normal*
        # fan-out sweep over every weight (zero bias), matching the paper's
        # FCN-style mask branch.  Lucid's conv default is kaiming *uniform*
        # with a=sqrt(5) — a different distribution and gain.
        for _name, _param in self.named_parameters():
            if "weight" in _name:
                nn.init.kaiming_normal_(_param, mode="fan_out", nonlinearity="relu")
            else:
                nn.init.zeros_(_param)


@final
class _MaskRCNNPredictor(nn.Module):
    """Reference ``MaskRCNNPredictor``: deconv-upsample then 1×1 logits.

    ``conv5_mask`` upsamples ``14×14 → 28×28`` (ConvTranspose2d 2×2
    stride 2), a parameter-free ReLU follows, then ``mask_fcn_logits``
    (Conv1×1) emits one logit map per class.
    """

    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int) -> None:
        super().__init__()
        self.conv5_mask = nn.ConvTranspose2d(in_channels, hidden_channels, 2, stride=2)
        self.mask_fcn_logits = nn.Conv2d(hidden_channels, num_classes, 1)

        # The reference ends both mask modules with a kaiming *normal*
        # fan-out sweep over every weight (zero bias), matching the paper's
        # FCN-style mask branch.  Lucid's conv default is kaiming *uniform*
        # with a=sqrt(5) — a different distribution and gain.
        for _name, _param in self.named_parameters():
            if "weight" in _name:
                nn.init.kaiming_normal_(_param, mode="fan_out", nonlinearity="relu")
            else:
                nn.init.zeros_(_param)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.conv5_mask(x))
        return self.mask_fcn_logits(x)


# ---------------------------------------------------------------------------
# RoI heads — box branch (reused) + mask branch (new)
# ---------------------------------------------------------------------------


@final
class _MaskRoIHeads(nn.Module):
    """RoI heads container: ``box_head`` + ``box_predictor`` (reused from
    Faster R-CNN) plus the new ``mask_head`` + ``mask_predictor``.

    The box-branch submodule names (``box_head`` = TwoMLPHead,
    ``box_predictor`` = FastRCNNPredictor) are byte-identical to
    :class:`~lucid.models.vision.faster_rcnn._model._RoIHeads`, so the 295
    shared keys map the same way.  The mask branch adds the 12 keys
    described in the module docstring.
    """

    def __init__(
        self,
        in_channels: int,
        roi_size: int,
        representation_size: int,
        num_classes: int,
        mask_hidden_channels: int,
        mask_num_convs: int,
        mask_predictor_hidden: int,
    ) -> None:
        super().__init__()
        # Box branch — identical to the Faster R-CNN RoI heads.
        self.box_head = _TwoMLPHead(in_channels * roi_size * roi_size, representation_size)
        self.box_predictor = _FastRCNNPredictor(representation_size, num_classes)
        # Mask branch.
        self.mask_head = _MaskRCNNHeads(in_channels, mask_hidden_channels, mask_num_convs)
        self.mask_predictor = _MaskRCNNPredictor(
            mask_hidden_channels, mask_predictor_hidden, num_classes
        )

    @override
    def forward(self, roi_feats: Tensor) -> tuple[Tensor, Tensor]:
        feats = self.box_head(roi_feats)
        return self.box_predictor(feats)

    def predict_masks(self, mask_feats: Tensor) -> Tensor:
        """Run the mask branch on RoI-aligned crops → ``(N, K, 28, 28)``."""
        x = self.mask_head(mask_feats)
        return self.mask_predictor(x)


# ---------------------------------------------------------------------------
# Mask R-CNN
# ---------------------------------------------------------------------------


class MaskRCNNForObjectDetection(ObjectDetectionModel):
    r"""Mask R-CNN with a ResNet-50-FPN backbone (He et al., ICCV 2017).

    The two-stage instance-segmentation detector in its modern reference
    configuration: Faster R-CNN's ResNet-50-FPN backbone, RPN, and Fast
    R-CNN box head, plus a parallel FCN mask branch on the RoI heads.  The
    submodule layout mirrors the reference detector so the COCO ``box AP
    37.9 / mask AP 34.6`` checkpoint loads strict (307 keys) and reproduces
    inference.

    Parameters
    ----------
    config : MaskRCNNConfig
        Frozen architecture spec.  Use the
        :func:`mask_rcnn_resnet50_fpn` factory for the COCO-pretrained
        configuration (``num_classes = 91``).

    Attributes
    ----------
    config : MaskRCNNConfig
        Stored copy of the config that built this model.
    backbone : _BackboneWithFPN
        ResNet-50 ``body`` + ``fpn`` producing five feature maps
        ``[P2, P3, P4, P5, pool]`` at strides ``4/8/16/32/64`` (reused
        from Faster R-CNN).
    rpn : _RegionProposalNetwork
        Proposal head shared across all pyramid levels (reused).
    roi_heads : _MaskRoIHeads
        ``box_head`` + ``box_predictor`` (reused) plus ``mask_head``
        (MaskRCNNHeads) + ``mask_predictor`` (MaskRCNNPredictor).

    Notes
    -----
    See He et al., "Mask R-CNN", ICCV 2017 (arXiv:1703.06870), Ren et al.,
    "Faster R-CNN", NeurIPS 2015, and Lin et al., "Feature Pyramid Networks
    for Object Detection", CVPR 2017.  The model expects an already resized
    + normalised image batch; final per-instance detections + masks come
    from :meth:`postprocess`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mask_rcnn import mask_rcnn_resnet50_fpn
    >>> model = mask_rcnn_resnet50_fpn().eval()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape[-1]   # num_classes
    91
    >>> out.pred_masks.shape[-2:]
    (28, 28)
    """

    config_class: ClassVar[type[MaskRCNNConfig]] = MaskRCNNConfig
    base_model_prefix: ClassVar[str] = "mask_rcnn"

    # Nominal FPN level strides (P2..P5 then the pool level) for a size that
    # divides evenly.  ``forward`` derives the real ones from the feature maps
    # rather than trusting these.
    _strides: ClassVar[tuple[int, ...]] = (4, 8, 16, 32, 64)

    def __init__(self, config: MaskRCNNConfig) -> None:
        super().__init__(config)
        self._cfg = config

        self.backbone = _BackboneWithFPN(
            in_channels=config.in_channels,
            layers=config.backbone_layers,
            fpn_out_channels=config.fpn_out_channels,
            bn_eps=config.backbone_bn_eps,
        )
        C = self.backbone.out_channels

        num_anchors = len(config.rpn_anchor_ratios)
        self.rpn = _RegionProposalNetwork(C, num_anchors)

        sizes: tuple[tuple[int, ...], ...] = tuple((s,) for s in config.rpn_anchor_sizes)
        ratios: tuple[tuple[float, ...], ...] = tuple(
            tuple(config.rpn_anchor_ratios) for _ in config.rpn_anchor_sizes
        )
        # Reuse the Faster R-CNN anchor generator verbatim.
        self._anchor_gen = _ReferenceAnchorGenerator(sizes, ratios)

        self.roi_heads = _MaskRoIHeads(
            in_channels=C,
            roi_size=config.roi_det_size,
            representation_size=config.roi_representation,
            num_classes=config.num_classes,
            mask_hidden_channels=config.mask_hidden_channels,
            mask_num_convs=config.mask_num_convs,
            mask_predictor_hidden=config.mask_predictor_hidden,
        )

    # ------------------------------------------------------------------
    # RPN proposal generation (inference) — delegate to the shared impl
    # ------------------------------------------------------------------

    def _rpn_proposals(
        self,
        logits: list[Tensor],
        deltas: list[Tensor],
        anchors: list[Tensor],
        image_size: tuple[int, int],
    ) -> list[Tensor]:
        """Decode + filter RPN predictions into per-image proposals.

        Reuses Faster R-CNN's proposal layer verbatim — the box-branch +
        RPN behaviour must be byte-identical so the shared keys produce
        identical proposals.
        """
        return _FasterRCNNForObjectDetection._rpn_proposals(
            cast(_FasterRCNNForObjectDetection, self), logits, deltas, anchors, image_size
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Training (shared assignment / sampling + the mask branch's L_mask)
    # ------------------------------------------------------------------

    def _rpn_loss(
        self,
        logits: list[Tensor],
        deltas: list[Tensor],
        anchors: list[Tensor],
        targets: list[dict[str, Tensor]],
        image_size: tuple[int, int],
    ) -> tuple[Tensor, Tensor]:
        cfg = self._cfg
        return rpn_loss(
            logits,
            deltas,
            anchors,
            targets,
            Matcher(cfg.rpn_fg_iou_thresh, cfg.rpn_bg_iou_thresh, allow_low_quality_matches=True),
            BalancedPositiveNegativeSampler(
                cfg.rpn_batch_size_per_image, cfg.rpn_positive_fraction
            ),
            image_size,
        )

    def _select_training_samples(
        self, proposals: list[Tensor], targets: list[dict[str, Tensor]]
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]]:
        cfg = self._cfg
        return select_training_samples(
            proposals,
            targets,
            Matcher(cfg.roi_fg_iou_thresh, cfg.roi_bg_iou_thresh),
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

    def _mask_loss(
        self,
        mask_logits: Tensor,
        proposals: list[Tensor],
        labels: list[Tensor],
        matched: list[Tensor],
        targets: list[dict[str, Tensor]],
    ) -> Tensor:
        """``L_mask`` over the sampled RoIs of every image in the batch.

        Returns zero when no target carries ``"masks"`` — training the
        detector without mask supervision is a legitimate configuration, and
        it should not be mistaken for a mask branch that has converged.
        """
        dev = mask_logits.device.type
        if not any("masks" in t for t in targets):
            return lucid.zeros((), device=dev)

        m = int(mask_logits.shape[-1])
        target_parts: list[Tensor] = []
        offset = 0
        index_parts: list[int] = []
        for props, lab, mt, tgt in zip(proposals, labels, matched, targets):
            n = int(props.shape[0])
            if "masks" in tgt and n > 0:
                target_parts.append(project_masks_on_boxes(tgt["masks"], props, mt, m))
                index_parts.extend(range(offset, offset + n))
            offset += n

        if not target_parts:
            return lucid.zeros((), device=dev)

        keep = lucid.tensor(index_parts, device=dev).long()
        labels_cat = lucid.cat(labels, dim=0)[keep]
        return maskrcnn_loss(mask_logits[keep], labels_cat, lucid.cat(target_parts, dim=0))

    @override
    def forward(
        self,
        x: Tensor,
        targets: list[dict[str, Tensor]] | None = None,
        proposals: list[Tensor] | None = None,
    ) -> InstanceSegmentationOutput:
        """Run Mask R-CNN on a (pre-processed) image batch.

        Args:
            x:         (B, C, H, W) resized + normalised image batch.
            targets:   Optional per-image ``{"boxes": (M, 4) xyxy, "labels":
                       (M,), "masks": (M, H, W) binary}``.  When given, the
                       five-term loss ``L_rpn_obj + L_rpn_reg + L_cls +
                       L_box + L_mask`` is computed.  ``"masks"`` may be
                       omitted, in which case ``L_mask`` is zero and only
                       the detector trains.
            proposals: Optional precomputed per-image proposals.  When
                       ``None`` the RPN generates them.

        Returns:
            ``InstanceSegmentationOutput`` with raw RoI-head outputs:
              ``logits``     : (Σ proposals, num_classes) class logits.
              ``pred_boxes`` : (Σ proposals, num_classes, 4) per-class boxes.
              ``pred_masks`` : (Σ proposals, num_classes, 28, 28) mask logits.
              ``loss``       : scalar sum of the five terms, or ``None``.

        Raises:
            ValueError: If ``targets`` is given but the RPN did not run.
        """
        B, _, iH, iW = (int(d) for d in x.shape)
        dev = x.device.type
        check_per_image(type(self).__name__, B, targets=targets, proposals=proposals)

        # 1. Backbone + FPN → [P2, P3, P4, P5, pool]
        features = self.backbone(x)

        # Level strides come from the maps the backbone produced, as in Faster
        # R-CNN.  The nominal ``_strides`` hold only for sizes divisible all
        # the way down: at 96 the pool level is 2 wide, a stride of 48.
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
                "actually ran.  Returning just the head terms would silently "
                "train part of the detector.  Omit `proposals` to train end "
                "to end, or omit `targets` to run inference on your own "
                "proposals."
            )

        # 3. Training: sample the head minibatch before either RoI Align, so
        #    both crops run only on the RoIs that reach the loss.
        roi_labels: list[Tensor] | None = None
        roi_reg_targets: list[Tensor] | None = None
        roi_matched: list[Tensor] | None = None
        if targets is not None:
            proposals, roi_labels, roi_reg_targets, roi_matched = self._select_training_samples(
                proposals, targets
            )

        # 3. MultiScale RoI Align over the four FPN detection levels (P2-P5).
        det_feats = features[:4]
        det_scales = [1.0 / float(s) for s in strides[:4]]
        roi_feats = multiscale_roi_align(
            det_feats,
            proposals,
            output_size=self._cfg.roi_det_size,
            spatial_scales=det_scales,
            sampling_ratio=self._cfg.roi_sampling_ratio,
            canonical_scale=self._cfg.canonical_scale,
            canonical_level=self._cfg.canonical_level,
        )

        # 4. Box RoI head → class logits + per-class box deltas
        K = self._cfg.num_classes
        if int(roi_feats.shape[0]) > 0:
            class_logits, box_deltas = self.roi_heads(roi_feats)
        else:
            class_logits = lucid.zeros((0, K), device=dev)
            box_deltas = lucid.zeros((0, K * 4), device=dev)

        # 5. Decode per-class boxes → (N, K, 4)
        pred_boxes = self._decode_per_class(proposals, box_deltas, (iH, iW))

        # 6. Mask RoI Align (14×14) over the SAME proposals + level assignment.
        mask_feats = multiscale_roi_align(
            det_feats,
            proposals,
            output_size=self._cfg.roi_mask_size,
            spatial_scales=det_scales,
            sampling_ratio=self._cfg.roi_sampling_ratio,
            canonical_scale=self._cfg.canonical_scale,
            canonical_level=self._cfg.canonical_level,
        )
        mH = self._cfg.roi_mask_size * 2  # deconv stride-2 upsample (14 → 28)
        if int(mask_feats.shape[0]) > 0:
            mask_logits = self.roi_heads.predict_masks(mask_feats)
        else:
            mask_logits = lucid.zeros((0, K, mH, mH), device=dev)

        loss: Tensor | None = None
        if targets is not None:
            assert roi_labels is not None
            assert roi_reg_targets is not None
            assert roi_matched is not None
            assert rpn_obj_loss is not None and rpn_reg_loss is not None
            cls_loss, box_loss = self._roi_loss(
                class_logits, box_deltas, roi_labels, roi_reg_targets
            )
            mask_loss = self._mask_loss(mask_logits, proposals, roi_labels, roi_matched, targets)
            loss = rpn_obj_loss + rpn_reg_loss + cls_loss + box_loss + mask_loss

        return InstanceSegmentationOutput(
            logits=class_logits,
            pred_boxes=pred_boxes,
            pred_masks=mask_logits,
            loss=loss,
            proposals=tuple(proposals),
            hidden_states=tuple(det_feats),
            input_size=(iH, iW),
        )

    def _decode_per_class(
        self, proposals: list[Tensor], box_deltas: Tensor, image_size: tuple[int, int]
    ) -> Tensor:
        """Decode ``(N, K*4)`` deltas against proposals → ``(N, K, 4)`` boxes."""
        return _FasterRCNNForObjectDetection._decode_per_class(
            cast(_FasterRCNNForObjectDetection, self), proposals, box_deltas, image_size
        )

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def postprocess(
        self,
        output: InstanceSegmentationOutput,
        image_sizes: list[tuple[int, int]] | None = None,
        proposals: list[Tensor] | None = None,
        features: list[Tensor] | None = None,
    ) -> list[dict[str, Tensor]]:
        """Box post-process → per-detection mask gather.

        Mirrors the reference inference flow: run the box branch
        post-processing (softmax → per-class score filter → per-class NMS →
        top-``max_detections``), then sigmoid the per-RoI mask logits and
        gather the channel for each detection's predicted class.

        Parameters
        ----------
        output : InstanceSegmentationOutput
            Raw RoI-head outputs from :meth:`forward`.
        image_sizes : list of (H, W), optional
            Per-image extent.  When given, image ``i``'s boxes are re-clipped
            to ``image_sizes[i]``, as in Faster R-CNN (decoding clips only to
            the padded batch canvas), and each detection's mask is pasted
            onto its clipped box in an ``H x W`` canvas and binarised at
            ``config.mask_thresh``.  When omitted, masks stay
            ``2 * roi_mask_size``-square (28 x 28 by default) probabilities
            in RoI coordinates.
        proposals : list of Tensor, optional
            Per-image proposals the RoI features were sampled from; falls
            back to ``output.proposals``, which :meth:`forward` fills in.
        features : list of Tensor, optional
            Multi-scale FPN feature maps; accepted for API symmetry with
            the detector-stage post-processor but not consumed here — the
            re-alignment on the final boxes reads ``output.hidden_states``.

        Returns
        -------
        list of dict
            One dict per image with ``"boxes"`` ``(D, 4)``, ``"scores"``
            ``(D,)``, ``"labels"`` ``(D,)`` int64, and ``"masks"`` — the
            channel of each detection's predicted class, as
            ``(D, 1, 28, 28)`` sigmoid probabilities or, with
            ``image_sizes``, as ``(D, 1, H, W)`` binary masks.  An image
            with no detections has ``D = 0`` and the same mask shape.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models import create_model
        >>> lucid.manual_seed(0)
        >>> model = create_model(
        ...     "mask_rcnn_resnet50_fpn", num_classes=3,
        ...     backbone_layers=(1, 1, 1, 1), fpn_out_channels=32,
        ...     roi_representation=64, mask_hidden_channels=16,
        ...     mask_num_convs=1, mask_predictor_hidden=16,
        ...     rpn_post_nms_top_n=10,
        ... ).eval()
        >>> out = model(lucid.randn((1, 3, 64, 64)))

        Without ``image_sizes`` there is one mask per box, each a 28 x 28
        grid of probabilities in that box's own frame.

        >>> det = model.postprocess(out)[0]
        >>> det["masks"].shape[0] == det["boxes"].shape[0]
        True
        >>> det["masks"].shape[1:]
        (1, 28, 28)

        With it, each grid is stretched onto its box inside an image-sized
        canvas and thresholded, so it can be laid over the picture as is.

        >>> masks = model.postprocess(out, image_sizes=[(64, 64)])[0]["masks"]
        >>> masks.shape[1:]
        (1, 64, 64)
        >>> bool(((masks == 0) | (masks == 1)).all())
        True

        Inside a padded batch an image's own extent is smaller than the
        canvas.  For one 48 tall and 40 wide the boxes are clipped to it,
        and the masks come back on its 48 x 40 frame.

        >>> det = model.postprocess(out, image_sizes=[(48, 40)])[0]
        >>> xs, ys = det["boxes"][:, 0::2], det["boxes"][:, 1::2]
        >>> bool((xs <= 40).all()), bool((ys <= 48).all())
        (True, True)
        >>> det["masks"].shape[1:]
        (1, 48, 40)
        """
        # ``forward`` carries its own proposals out, mirroring Faster R-CNN,
        # so ``model.postprocess(model(x))`` works without re-running the RPN.
        del features
        if proposals is None:
            proposals = list(output.proposals) if output.proposals is not None else None
        if proposals is None:
            raise ValueError("postprocess() needs the per-image proposals used in forward().")

        logits = output.logits
        pred_boxes = output.pred_boxes  # (N, K, 4)
        pred_masks = output.pred_masks  # (N, K, 28, 28)
        cfg = self._cfg
        dev = logits.device.type
        # Mask side in RoI space: the deconv doubles ``roi_mask_size``.
        roi_hw = (2 * cfg.roi_mask_size, 2 * cfg.roi_mask_size)
        results: list[dict[str, Tensor]] = []
        offset = 0

        for idx, props in enumerate(proposals):
            N_i = int(props.shape[0])
            lg_i = logits[offset : offset + N_i]
            bx_i = pred_boxes[offset : offset + N_i]
            mk_i = pred_masks[offset : offset + N_i]  # (N_i, K, 28, 28)
            offset += N_i
            # As in Faster R-CNN, an image past the end of ``image_sizes``
            # keeps the canvas it was decoded on.
            size_i = (
                image_sizes[idx] if image_sizes is not None and idx < len(image_sizes) else None
            )
            # An empty result carries the mask shape a non-empty one would.
            mask_hw = size_i if size_i is not None else roi_hw

            if N_i == 0:
                results.append(empty_detection_result(dev, mask_hw))
                continue

            scores_i = F.softmax(lg_i, dim=-1)  # (N_i, K)
            host_scores = host_rows(scores_i)
            keep_boxes: list[Tensor] = []
            keep_scores: list[Tensor] = []
            keep_labels: list[Tensor] = []
            keep_masks: list[Tensor] = []

            for c in range(1, cfg.num_classes):  # skip background slot 0
                bx_class = bx_i[:, c, :]  # (N_i, 4)
                mask = [i for i, row in enumerate(host_scores) if row[c] > cfg.score_thresh]
                if not mask:
                    continue
                mask_t = lucid.tensor(mask, device=dev).long()
                sc_c = scores_i[:, c][mask_t]
                bx_c = bx_class[mask_t]
                # Mask channel for class c (sigmoid → probability).
                mk_c = F.sigmoid(mk_i[mask_t][:, c, :, :])  # (k, 28, 28)
                # The reference drops degenerate boxes before NMS.  A
                # zero-area box has zero IoU with everything, so NMS cannot
                # suppress it and it survives into the final detections.
                keep_sz = remove_small_boxes(bx_c, 1e-2)
                if int(keep_sz.shape[0]) == 0:
                    continue
                bx_c = bx_c[keep_sz]
                sc_c = sc_c[keep_sz]
                mk_c = mk_c[keep_sz]
                keep = nms(bx_c, sc_c, cfg.nms_thresh)
                keep_boxes.append(bx_c[keep])
                keep_scores.append(sc_c[keep])
                keep_labels.append(
                    lucid.full((int(keep.shape[0]),), c, dtype=lucid.int64, device=dev)
                )
                keep_masks.append(mk_c[keep].unsqueeze(1))  # (k, 1, 28, 28)

            if not keep_boxes:
                results.append(empty_detection_result(dev, mask_hw))
                continue

            all_b = lucid.cat(keep_boxes, dim=0)
            all_s = lucid.cat(keep_scores, dim=0)
            all_l = lucid.cat(keep_labels, dim=0)
            all_m = lucid.cat(keep_masks, dim=0)
            order = lucid.argsort(-all_s)[: cfg.max_detections]
            det_b = all_b[order]
            det_s = all_s[order]
            det_l = all_l[order].long()
            det_m = all_m[order]
            if size_i is not None:
                # Decoding clipped to the padded batch canvas; the image's own
                # extent can be smaller.  Clipping before the masks are
                # re-aligned and pasted keeps each mask on the box returned.
                det_b = clip_boxes_to_image(det_b, size_i)

            # Paper §3.1 (Inference): "The mask branch is then applied to the
            # highest scoring 100 detection boxes ... it speeds up inference and
            # improves accuracy (due to the use of fewer, more accurate RoIs)."
            # The masks gathered above were RoI-Aligned on the *un-refined* RPN
            # proposals, so their 28x28 grid is in the proposal's coordinate
            # frame while ``boxes`` are the regressed ones — a consumer pasting
            # the mask into the box would place it wrong.  Re-align on the final
            # boxes when the feature levels came along with the output.
            feats = output.hidden_states
            if feats is not None and int(det_b.shape[0]) > 0:
                # Re-align with the strides ``forward`` derived, from the
                # input height it recorded; the feature maps alone do not pin
                # that down.  An output built without it falls back to four
                # times the P2 height (the stem halves the input twice,
                # rounding up), which is the input height whenever that is a
                # multiple of four.
                levels = list(feats)
                if output.input_size is not None:
                    canvas_h = output.input_size[0]
                else:
                    canvas_h = 4 * int(levels[0].shape[2])
                strides = _level_strides(canvas_h, levels)
                scales = [1.0 / float(s) for s in strides]
                m_feats = multiscale_roi_align(
                    levels,
                    [det_b],
                    output_size=cfg.roi_mask_size,
                    spatial_scales=scales,
                    sampling_ratio=cfg.roi_sampling_ratio,
                    canonical_scale=cfg.canonical_scale,
                    canonical_level=cfg.canonical_level,
                )
                m_logits = self.roi_heads.predict_masks(m_feats)  # (D, K, mh, mw)
                # Each detection's own class channel, gathered on device.
                rows = lucid.arange(int(det_b.shape[0]), device=dev).long()
                det_m = F.sigmoid(m_logits[rows, det_l]).unsqueeze(1)  # (D, 1, mh, mw)

            # Paste onto the boxes and binarise.  A raw mh x mw probability
            # grid is in RoI coordinates, so it says nothing about where in
            # the image the instance is -- and ``mask_thresh`` had nothing to
            # act on.  ``image_sizes`` is what makes the canvas known; without
            # it the RoI-space probabilities are returned unchanged.
            if size_i is not None and int(det_b.shape[0]) > 0:
                det_m = paste_masks_in_image(det_m, det_b, size_i, threshold=cfg.mask_thresh)

            results.append(detection_result(det_b, det_s, det_l, det_m))
        return results
