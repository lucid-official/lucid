"""Mask R-CNN configuration (He et al., 2017)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta
from lucid.models._utils._common import check_positive, check_residual_layers

# P2-P5 from the FPN plus its max-pooled P6.
_FPN_LEVELS = 5


@model_family_meta(
    canonical_name="Mask R-CNN",
    citation=(
        'He, Kaiming, et al. "Mask R-CNN." Proceedings of the IEEE '
        "International Conference on Computer Vision, 2017, pp. 2961–2969."
    ),
    theory=r"""
    Mask R-CNN extends Faster R-CNN with a **parallel mask branch**
    that predicts a per-RoI binary segmentation, turning the
    two-stage detector into a unified instance-segmentation model
    with negligible runtime overhead.

    Three design changes are critical:

    1. **Feature Pyramid Network backbone.**  Lateral :math:`1 \times 1`
       convolutions fuse top-down semantically strong features with
       bottom-up spatially precise ones, yielding multi-scale outputs
       :math:`\{P_2, \dots, P_6\}` all at the same channel depth (256).
       Each RoI is routed to the level

       .. math::

           k = \lfloor k_0 + \log_2(\sqrt{wh}/224) \rfloor,

       so small objects use high-resolution maps and large ones use
       deeper, semantically richer features.

    2. **RoI Align.**  RoI Pool quantises coordinates twice
       (proposal → feature grid, then feature grid → output cells),
       which misaligns mask predictions by several pixels.  RoI Align
       removes both quantisations and bilinearly interpolates four
       regularly-sampled points per cell.  This single change adds
       :math:`\sim 10\%` AP on the mask task.

    3. **Decoupled mask head.**  Unlike FCN-style approaches that
       compete classes via per-pixel softmax, Mask R-CNN emits a
       :math:`K \times m \times m` tensor of *per-class* binary masks
       and applies sigmoid + binary cross-entropy.  The classification
       branch alone chooses *which* class's mask to use at inference,
       avoiding inter-class competition during training.

    The total loss is a sum of the Faster R-CNN classification +
    regression terms and an additional mean BCE on the predicted
    mask channel of the ground-truth class, :math:`L = L_{\mathrm{cls}}
    + L_{\mathrm{box}} + L_{\mathrm{mask}}`.
    """,
)
@dataclass(frozen=True)
class MaskRCNNConfig(ModelConfig):
    """Configuration for Mask R-CNN.

    Mask R-CNN (He et al., ICCV 2017) extends Faster R-CNN with:
      1. Feature Pyramid Network (FPN) backbone replacing single-scale VGG16.
      2. RoI Align replacing RoI Pool (eliminating quantisation misalignment).
      3. A parallel mask branch predicting K binary segmentation masks per RoI.

    Architecture overview:
      Image → ResNet-50 (C2–C5, strides 4/8/16/32)
        ↓ FPN (lateral convs + top-down merging) → [P2, P3, P4, P5, P6] (256ch)
      P2–P6 → RPN → proposals
      P2–P5 + proposals → FPN-level assignment → RoI Align (7×7) → 2-FC head
                                                → RoI Align (14×14) → Mask head

    Args:
        num_classes:  Foreground categories (background = class 0).
        in_channels:  Input image channels.

        -- Backbone/FPN --
        backbone_layers:   ResNet layer counts (default ResNet-50 = 3,4,6,3).
        backbone_bn_eps:   FrozenBatchNorm2d epsilon (reference uses 0).
        fpn_out_channels:  Channel width of every FPN output level.

        -- RPN hyper-parameters --
        rpn_anchor_sizes:   One anchor size per FPN level (P2–P6).
        rpn_anchor_ratios:  Aspect ratios shared across all levels.
        rpn_pre_nms_top_n:  Proposals kept per level before NMS.
        rpn_post_nms_top_n: Proposals kept per image after NMS.
        rpn_nms_thresh:     RPN NMS IoU threshold.
        rpn_min_size:       Minimum proposal side length (feature pixels).
        rpn_score_thresh:   Minimum objectness score (post-sigmoid).
        rpn_fg_iou_thresh:  Anchor→GT IoU for foreground label.
        rpn_bg_iou_thresh:  Anchor→GT IoU upper bound for background.

        -- Detection head --
        roi_det_size:         RoI Align output for detection (7 → 7×7).
        roi_sampling_ratio:   RoI Align sub-bin sampling ratio (2).
        roi_representation:   Hidden size of the 2-FC detection head.
        roi_fg_iou_thresh:    Proposal→GT IoU for fg assignment.
        roi_bg_iou_thresh:    Proposal→GT IoU upper bound for bg.
        bbox_reg_weights:     Per-component bbox delta scale.
        canonical_scale:      FPN level-assignment canonical scale (224).
        canonical_level:      FPN level-assignment canonical level (4).

        -- Mask head --
        roi_mask_size:        RoI Align output for the mask branch (14 → 14×14).
        mask_hidden_channels: Channel width inside the mask FCN.
        mask_num_convs:       Number of 3×3 convs in the mask head (4).
        mask_predictor_hidden: Channel width of the deconv upsampler (256).

        -- Inference --
        score_thresh:    Minimum final class score.
        nms_thresh:      Per-class NMS IoU threshold.
        max_detections:  Maximum detections returned per image.
        mask_thresh:     Binarisation threshold applied to mask sigmoid output.

    .. note::

       **Training.**  Pass ``targets`` to :meth:`forward` for §3's
       multi-task objective ``L = L_cls + L_box + L_mask`` on top of the
       inherited RPN terms.  ``L_mask`` is the per-pixel sigmoid BCE on the
       ground-truth class's mask channel only, with targets from
       RoI-aligning the ground-truth masks onto the sampled proposals.
       ``"masks"`` may be omitted from a target to train the detector
       alone.

    Examples
    --------
    >>> from lucid.models.vision.mask_rcnn import MaskRCNNConfig
    >>> cfg = MaskRCNNConfig()
    >>> cfg.num_classes
    91
    >>> cfg.model_type
    'mask_rcnn'
    """

    model_type: ClassVar[str] = "mask_rcnn"

    num_classes: int = 91
    in_channels: int = 3

    # Backbone — ResNet-50-FPN
    backbone_layers: tuple[int, int, int, int] = (3, 4, 6, 3)  # ResNet-50
    backbone_bn_eps: float = 0.0  # reference FrozenBatchNorm2d uses eps=0
    fpn_out_channels: int = 256

    # RPN — one anchor scale per FPN level x 3 ratios = 3 anchors/location
    rpn_anchor_sizes: tuple[int, ...] = (32, 64, 128, 256, 512)
    rpn_anchor_ratios: tuple[float, ...] = (0.5, 1.0, 2.0)
    rpn_pre_nms_top_n: int = 1000
    rpn_post_nms_top_n: int = 1000
    rpn_nms_thresh: float = 0.7
    rpn_min_size: float = 1e-3
    rpn_score_thresh: float = 0.0
    # Faster R-CNN 3.1.2's label assignment, inherited unchanged.
    rpn_fg_iou_thresh: float = 0.7
    rpn_bg_iou_thresh: float = 0.3
    rpn_batch_size_per_image: int = 256
    rpn_positive_fraction: float = 0.5

    # Detection head
    roi_det_size: int = 7
    roi_sampling_ratio: int = 2
    roi_representation: int = 1024
    roi_fg_iou_thresh: float = 0.5
    roi_bg_iou_thresh: float = 0.5
    roi_batch_size_per_image: int = 512
    roi_positive_fraction: float = 0.25
    bbox_reg_weights: tuple[float, float, float, float] = (10.0, 10.0, 5.0, 5.0)
    canonical_scale: int = 224
    canonical_level: int = 4

    # Mask head
    roi_mask_size: int = 14
    mask_hidden_channels: int = 256
    mask_num_convs: int = 4
    mask_predictor_hidden: int = 256

    # Inference
    score_thresh: float = 0.05
    nms_thresh: float = 0.5
    max_detections: int = 100
    mask_thresh: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "backbone_layers", tuple(self.backbone_layers))
        object.__setattr__(self, "rpn_anchor_sizes", tuple(self.rpn_anchor_sizes))
        object.__setattr__(self, "rpn_anchor_ratios", tuple(self.rpn_anchor_ratios))
        object.__setattr__(self, "bbox_reg_weights", tuple(self.bbox_reg_weights))
        # The anchor generator pairs these with the FPN outputs level by
        # level; any other count fails mid-forward as a zip length error.
        if len(self.rpn_anchor_sizes) != _FPN_LEVELS:
            raise ValueError(
                f"{type(self).__name__}: rpn_anchor_sizes holds one anchor size "
                f"per FPN level and the backbone emits {_FPN_LEVELS} (P2-P6); "
                f"got {len(self.rpn_anchor_sizes)}: {self.rpn_anchor_sizes}"
            )

        for name in ("rpn", "roi"):
            fg = getattr(self, f"{name}_fg_iou_thresh")
            bg = getattr(self, f"{name}_bg_iou_thresh")
            if bg > fg:
                raise ValueError(
                    f"{name}_bg_iou_thresh ({bg}) must not exceed "
                    f"{name}_fg_iou_thresh ({fg}); the band between them is "
                    "the ignore region, and inverting them would leave no "
                    "ambiguous zone at all."
                )
        owner = type(self).__name__
        check_positive(owner, num_classes=self.num_classes, in_channels=self.in_channels)
        check_positive(owner, fpn_out_channels=self.fpn_out_channels)
        check_residual_layers(owner, backbone_layers=self.backbone_layers)
