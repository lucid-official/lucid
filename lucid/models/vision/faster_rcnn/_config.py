"""Faster R-CNN configuration (Ren et al., 2015)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta
from lucid.models._utils._common import check_positive, check_residual_layers

# P2-P5 from the FPN plus its max-pooled P6.
_FPN_LEVELS = 5


@model_family_meta(
    canonical_name="Faster R-CNN",
    citation=(
        'Ren, Shaoqing, et al. "Faster R-CNN: Towards Real-Time Object '
        'Detection with Region Proposal Networks." Advances in Neural '
        "Information Processing Systems, 2015."
    ),
    theory=r"""
    Faster R-CNN removes the last hand-engineered component of the
    R-CNN family — the external selective-search proposal step — by
    introducing a **Region Proposal Network (RPN)** that *shares
    convolutional features* with the detection head.

    The RPN is a small fully-convolutional network that slides over
    the backbone feature map.  At every spatial position it evaluates
    :math:`k` reference boxes called **anchors** (the paper uses
    :math:`k = 9` from 3 scales × 3 aspect ratios) and emits two
    heads:

    - **Objectness** — a :math:`2k`-dim softmax classifying each
      anchor as foreground or background.
    - **Box regression** — a :math:`4k`-dim parameterised offset
      :math:`t = (t_x, t_y, t_w, t_h)` refining anchors toward
      ground-truth boxes.

    Anchors with IoU :math:`\ge 0.7` against any GT are positive;
    :math:`\le 0.3` are negative.  The multi-task loss is

    .. math::

        L(\{p_i\}, \{t_i\}) =
            \tfrac{1}{N_{\mathrm{cls}}} \sum_i L_{\mathrm{cls}}(p_i, p_i^*)
            + \lambda \, \tfrac{1}{N_{\mathrm{reg}}} \sum_i p_i^*
              \, L_{\mathrm{reg}}(t_i, t_i^*).

    Top-scoring proposals (after NMS) replace selective search
    completely.  The **same backbone** then feeds the Fast R-CNN
    head, so the entire detector is end-to-end trainable.  With a
    VGG-16 trunk the whole pipeline runs at ≈5 fps while matching
    or exceeding selective-search Fast R-CNN accuracy — a milestone
    that established two-stage anchor-based detection as the
    dominant paradigm and seeded all later FPN / Mask R-CNN / Cascade
    R-CNN extensions.
    """,
)
@dataclass(frozen=True)
class FasterRCNNConfig(ModelConfig):
    """Configuration for Faster R-CNN.

    Faster R-CNN (Ren et al., NeurIPS 2015) replaces external proposal methods
    (selective search) with a learned Region Proposal Network (RPN) that shares
    convolutional features with the detection head.  This makes the entire
    pipeline end-to-end trainable and near real-time.

    Architecture overview (ResNet-50-FPN variant):
      Image → ResNet-50 backbone (frozen BN, C2-C5) → FPN (P2-P5 + pool)
      FPN levels → RPN head → region proposals (per-level top-k + NMS)
      FPN levels + proposals → MultiScale RoI Align (7×7) → TwoMLPHead
        (fc6/fc7) → FastRCNNPredictor (cls + per-class bbox)

    Args:
        num_classes:      Object classes incl. background (91 for COCO).
        in_channels:      Input image channels.

        -- Backbone --
        backbone_layers:  ResNet bottleneck counts (default 3,4,6,3 = R50).
        backbone_bn_eps:  FrozenBatchNorm2d epsilon (reference uses 0).
        fpn_out_channels: Unified FPN channel count (256).

        -- RPN hyper-parameters --
        rpn_anchor_sizes:    One anchor scale per FPN level (5 levels).
        rpn_anchor_ratios:   Aspect ratios for anchor generation.
        rpn_pre_nms_top_n:   Proposals kept per level before NMS (test 1000).
        rpn_post_nms_top_n:  Proposals kept per image after NMS (test 1000).
        rpn_nms_thresh:      RPN NMS IoU threshold.
        rpn_min_size:        Minimum proposal side length (pixels).
        rpn_score_thresh:    Minimum RPN objectness score.
        rpn_fg_iou_thresh:   Anchor→GT IoU for foreground assignment.
                             **Inert** — see the note on the field.
        rpn_bg_iou_thresh:   Anchor→GT IoU upper bound for background.
                             **Inert** — see the note on the field.

        -- RoI head hyper-parameters --
        roi_size:               RoI Align output spatial size (7 → 7×7).
        roi_sampling_ratio:     RoI Align sub-bin sampling ratio (2).
        roi_representation_size: TwoMLPHead hidden width (1024).
        roi_fg_iou_thresh:      Proposal→GT IoU for fg assignment.
                                **Inert** — see the note on the field.
        roi_bg_iou_thresh:      Proposal→GT IoU upper bound for bg.
                                **Inert** — see the note on the field.
        bbox_reg_weights:       Per-component bbox delta scale (10,10,5,5).
        canonical_scale:        FPN level-assignment canonical scale (224).
        canonical_level:        FPN level-assignment canonical level (4).

        -- Inference hyper-parameters --
        score_thresh:     Minimum final class score (0.05).
        nms_thresh:       Per-class NMS IoU threshold (0.5).
        max_detections:   Maximum detections returned per image (100).

    .. note::

       **Training.**  Pass ``targets`` to :meth:`forward` to get the
       four-term loss — RPN objectness and box regression, plus the box
       head's classification and regression.  Label assignment follows
       §3.1.2 (highest-IoU anchor per ground truth, or IoU over 0.7, as
       positive; under 0.3 as negative; the band between ignored) and the
       samplers follow §3.1.3.  The one clause not on by default is
       cross-boundary anchor removal — see ``rpn_ignore_cross_boundary``.

    Examples
    --------
    >>> from lucid.models.vision.faster_rcnn import FasterRCNNConfig
    >>> cfg = FasterRCNNConfig()
    >>> cfg.num_classes
    91
    >>> cfg.model_type
    'faster_rcnn'
    """

    model_type: ClassVar[str] = "faster_rcnn"

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
    # §3.1.2's label assignment: over 0.7 is an object, under 0.3 is
    # background, the band between contributes to no loss term.
    rpn_fg_iou_thresh: float = 0.7
    rpn_bg_iou_thresh: float = 0.3
    # §3.1.3: 256 anchors per image at up to 1:1 positive:negative.
    rpn_batch_size_per_image: int = 256
    rpn_positive_fraction: float = 0.5
    # §3.1.2 also says "we ignore all cross-boundary anchors so they do not
    # contribute to the loss".  The reference implementation these weights
    # come from clips anchors instead of dropping them, and its published
    # accuracies were trained that way, so this defaults off.
    rpn_ignore_cross_boundary: bool = False

    # RoI head
    roi_size: int = 7
    roi_sampling_ratio: int = 2
    roi_representation_size: int = 1024
    roi_fg_iou_thresh: float = 0.5
    roi_bg_iou_thresh: float = 0.5
    # The box head's own minibatch.  Fast R-CNN 2.3 specifies 64 at 25%;
    # the reference implementation raised the count to 512 at the same
    # ratio and that is what its published detectors were trained with.
    roi_batch_size_per_image: int = 512
    roi_positive_fraction: float = 0.25
    bbox_reg_weights: tuple[float, float, float, float] = (10.0, 10.0, 5.0, 5.0)
    canonical_scale: int = 224
    canonical_level: int = 4

    # Inference
    score_thresh: float = 0.05
    nms_thresh: float = 0.5
    max_detections: int = 100

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
                    "the ignore region, and inverting them would make every "
                    "prediction either a positive or a negative with no "
                    "ambiguous zone."
                )
        owner = type(self).__name__
        check_positive(owner, num_classes=self.num_classes, in_channels=self.in_channels)
        check_positive(owner, fpn_out_channels=self.fpn_out_channels)
        check_residual_layers(owner, backbone_layers=self.backbone_layers)
