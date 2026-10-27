"""EfficientDet configuration (Tan et al., 2020)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta


@model_family_meta(
    canonical_name="EfficientDet",
    citation=(
        'Tan, Mingxing, et al. "EfficientDet: Scalable and Efficient '
        'Object Detection." Proceedings of the IEEE/CVF Conference on '
        "Computer Vision and Pattern Recognition, 2020, pp. 10781–10790."
    ),
    theory=r"""
    EfficientDet jointly applies the **compound scaling** idea of
    EfficientNet to the entire detection pipeline — backbone, feature
    network, and prediction heads — instead of scaling any single
    axis (width, depth, resolution) in isolation.  A single compound
    coefficient :math:`\varphi` controls all three:

    .. math::

        D_{\mathrm{bifpn}} = 3 + \varphi, \quad
        W_{\mathrm{bifpn}} = 64 \cdot 1.35^{\varphi}, \quad
        D_{\mathrm{head}} = 3 + \lfloor \varphi / 3 \rfloor, \quad
        R_{\mathrm{input}} = 512 + 128 \varphi,

    so EfficientDet-D0 through D7 form a family with predictable
    accuracy–latency trade-offs without manual hyper-tuning.

    The feature network is a **BiFPN** (Bi-directional Feature
    Pyramid Network).  Unlike a one-way FPN, BiFPN adds bottom-up
    paths back into the pyramid and learns **fast normalised
    fusion** weights so each merged level is a convex combination
    of its inputs:

    .. math::

        O = \sum_{i} \frac{w_i}{\varepsilon + \sum_j w_j} \cdot I_i,
        \qquad w_i \ge 0,

    where the ReLU on :math:`w_i` (here written as a non-negativity
    constraint) keeps fusion stable without an explicit softmax,
    and the small :math:`\varepsilon = 10^{-4}` prevents division
    by zero.  BiFPN is then **stacked** :math:`D_{\mathrm{bifpn}}`
    times.

    The detection heads (classification + box) share weights across
    levels and use sigmoid + **focal loss** for classification, the
    same anchor-based formulation as RetinaNet.  Combined with the
    EfficientNet-B\ :math:`_{\varphi}` backbone, the result is the
    Pareto-optimal one-stage detector family of its time —
    EfficientDet-D7 reaches 55.1 AP on COCO at far fewer FLOPs than
    contemporaries.
    """,
)
@dataclass(frozen=True)
class EfficientDetConfig(ModelConfig):
    """Configuration for EfficientDet.

    EfficientDet (Tan et al., CVPR 2020) applies compound scaling to object
    detection.  Each compound coefficient φ jointly scales the backbone
    (EfficientNet-Bφ), the BiFPN width/depth, and the prediction head
    depth/resolution.

    Architecture overview:
      Image → EfficientNet backbone (P3–P7 feature maps)
        → BiFPN (D_bifpn repeats, W_bifpn channels, fast normalised fusion)
      BiFPN outputs (P3–P7) → Class prediction head (shared conv)
                             → Box prediction head  (shared conv)

    BiFPN fast-normalised fusion:
      weight_i / (ε + Σ weight_j) — learnable positive weights (ε=1e-4)

    Args:
        num_classes:    Foreground classes (background not counted separately
                        — EfficientDet uses sigmoid + focal loss).
        in_channels:    Input image channels.

        -- Compound scaling (φ = 0–7) --
        phi:          Compound coefficient index.  Default 0 (EfficientDet-D0).

        -- Backbone (EfficientNet-B0–B7 widths/depths) --
        backbone_width_coeff:  Width multiplier (EfficientNet channel scaling).
        backbone_depth_coeff:  Depth multiplier (EfficientNet block repeat).
        backbone_in_channels:  Channel widths of the three backbone stages
            the BiFPN taps (C3, C4, C5), before its lateral 1x1s.

        -- BiFPN --
        fpn_channels:   BiFPN channel width (W_bifpn).
        fpn_repeats:    Number of BiFPN stacking repetitions (D_bifpn).

        -- Prediction heads --
        head_repeats:   Depth of class/box conv heads (D_head).

        -- Anchors --
        anchor_scales:     Size multipliers applied at every level.  The
            default is the octave subdivision the paper uses,
            ``2 ** (0/3), 2 ** (1/3), 2 ** (2/3)`` — the subdivisions are
            these values, not a separate count.
        anchor_ratios:     Anchor aspect ratios.
        anchor_base_sizes: Base anchor edge length per pyramid level, in
            input pixels.  Scaled by ``anchor_scales`` to give the
            anchors of that level.

        image_size:     Input resolution the anchors and strides are
            defined against; the compound scaling ties it to ``phi``.

        -- Loss --
        focal_alpha:      Focal-loss class balance (paper: 0.25).
        focal_gamma:      Focal-loss focusing exponent (paper: 1.5).
        box_loss_weight:  Multiplier on the box term of the total loss (50).
        focal_prior_prob: Class-head bias prior probability (0.01).
        iou_fg_thresh:    IoU at or above which an anchor is foreground.
        iou_bg_thresh:    IoU below which an anchor is background; the
                          [bg, fg) band is ignored by the class loss.

        -- Inference --
        score_thresh:   Minimum sigmoid class score.
        nms_thresh:     Per-class NMS threshold.
        max_detections: Maximum detections per image.

    Examples
    --------
    >>> from lucid.models.vision.efficientdet._config import EfficientDetConfig
    >>> cfg = EfficientDetConfig()
    >>> cfg.model_type
    'efficientdet'
    >>> cfg.num_classes, cfg.image_size
    (80, 512)
    """

    model_type: ClassVar[str] = "efficientdet"

    num_classes: int = 80
    in_channels: int = 3

    # Compound coefficient
    phi: int = 0

    # Backbone feature channel counts at P3/P4/P5 (extracted from EfficientNet)
    backbone_in_channels: tuple[int, int, int] = (40, 112, 320)
    # EfficientNet compound coefficients for the backbone.  D{phi} uses
    # B{phi}: (1.0,1.0) (1.0,1.1) (1.1,1.2) (1.2,1.4) (1.4,1.8) (1.6,2.2)
    # (1.8,2.6) (2.0,3.1).  These have to agree with
    # ``backbone_in_channels`` — the projections are built from the latter
    # and fed by the former.
    backbone_width_coeff: float = 1.0
    backbone_depth_coeff: float = 1.0

    # BiFPN
    fpn_channels: int = 64
    fpn_repeats: int = 3

    # Prediction head depth
    head_repeats: int = 3

    # Anchors: 3 scales × 3 ratios = 9 per cell
    anchor_scales: tuple[float, ...] = (1.0, 2.0 ** (1.0 / 3.0), 2.0 ** (2.0 / 3.0))
    anchor_ratios: tuple[float, ...] = (0.5, 1.0, 2.0)
    anchor_base_sizes: tuple[int, ...] = (32, 64, 128, 256, 512)

    # Loss (paper §6: "focal loss with alpha = 0.25 and gamma = 1.5"; the
    # reference additionally weights the box term by 50 because both terms are
    # normalised by the same positive-anchor count).
    focal_alpha: float = 0.25
    focal_gamma: float = 1.5
    box_loss_weight: float = 50.0
    # Prior probability the class head's bias is initialised to, so that at
    # step 0 every anchor predicts p ~= 0.01 rather than 0.5.
    focal_prior_prob: float = 0.01
    # Anchors below ``iou_bg_thresh`` are background, at or above
    # ``iou_fg_thresh`` are foreground, and the band between the two is
    # *ignored* — excluded from the classification loss entirely.
    iou_fg_thresh: float = 0.5
    iou_bg_thresh: float = 0.4

    # Input resolution, the third scaled axis (paper eq. R = 512 + 128*phi).
    # Table 1: 512/640/768/896/1024/1280/1280/1536.
    image_size: int = 512

    # Inference
    score_thresh: float = 0.05
    nms_thresh: float = 0.5
    max_detections: int = 100

    def __post_init__(self) -> None:
        object.__setattr__(self, "backbone_in_channels", tuple(self.backbone_in_channels))
        object.__setattr__(self, "anchor_scales", tuple(self.anchor_scales))
        object.__setattr__(self, "anchor_ratios", tuple(self.anchor_ratios))
        object.__setattr__(self, "anchor_base_sizes", tuple(self.anchor_base_sizes))


# ---------------------------------------------------------------------------
# Compound-scaling lookup table (paper Table 1)
# ---------------------------------------------------------------------------

# Each entry: (phi, fpn_channels, fpn_repeats, head_repeats,
#              backbone_in_channels-P3/P4/P5, image_size)
# (fpn_channels, fpn_repeats, head_repeats, backbone_in_channels,
#  backbone_width, backbone_depth, image_size, anchor_scale)
_COMPOUND_PARAMS: dict[
    int, tuple[int, int, int, tuple[int, int, int], float, float, int, float]
] = {
    0: (64, 3, 3, (40, 112, 320), 1.0, 1.0, 512, 4.0),
    1: (88, 4, 3, (40, 112, 320), 1.0, 1.1, 640, 4.0),
    2: (112, 5, 3, (48, 120, 352), 1.1, 1.2, 768, 4.0),
    3: (160, 6, 4, (48, 136, 384), 1.2, 1.4, 896, 4.0),
    4: (224, 7, 4, (56, 160, 448), 1.4, 1.8, 1024, 4.0),
    5: (288, 7, 4, (64, 176, 512), 1.6, 2.2, 1280, 4.0),
    6: (384, 8, 5, (72, 200, 576), 1.8, 2.6, 1280, 4.0),
    # D7 is the one variant the reference overrides: anchor_scale 5.0, so the
    # base sizes become 40/80/160/320/640 rather than 4*stride.
    7: (384, 8, 5, (80, 224, 640), 2.0, 3.1, 1536, 5.0),
}


def _efficientdet_config(phi: int = 0, num_classes: int = 80) -> EfficientDetConfig:
    """Build the canonical EfficientDet-D{phi} config from the compound table.

    EfficientDet's compound scaling (Tan et al., 2020) maps a single
    scalar :math:`\\phi \\in \\{0, \\ldots, 7\\}` to the BiFPN channel
    count, BiFPN repeat count, head repeat count, and backbone output
    channel counts in lock-step.  This helper looks the row up and
    materialises an :class:`EfficientDetConfig`.

    Parameters
    ----------
    phi : int, optional
        Compound-scaling coefficient.  Values ``0`` through ``7``
        select EfficientDet-D0 (smallest, default) through
        EfficientDet-D7 (largest) per the paper's Table 1.
    num_classes : int, optional
        Foreground class count.  Default ``80`` (COCO).

    Returns
    -------
    EfficientDetConfig
        Frozen config ready to feed into the model factory.

    References
    ----------
    .. [1] Tan, Pang & Le, *EfficientDet: Scalable and Efficient Object
       Detection*, CVPR 2020.
    """
    fpn_ch, fpn_rep, head_rep, bb_ch, bb_w, bb_d, img, anc_scale = _COMPOUND_PARAMS[phi]
    strides = (8, 16, 32, 64, 128)
    return EfficientDetConfig(
        image_size=img,
        anchor_base_sizes=tuple(int(anc_scale * st) for st in strides),
        num_classes=num_classes,
        phi=phi,
        backbone_in_channels=bb_ch,
        backbone_width_coeff=bb_w,
        backbone_depth_coeff=bb_d,
        fpn_channels=fpn_ch,
        fpn_repeats=fpn_rep,
        head_repeats=head_rep,
    )
