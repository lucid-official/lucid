"""Fast R-CNN configuration (Girshick, 2015)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta
from lucid.models._utils._common import check_positive


@model_family_meta(
    canonical_name="Fast R-CNN",
    citation=(
        'Girshick, Ross. "Fast R-CNN." Proceedings of the IEEE '
        "International Conference on Computer Vision, 2015, pp. 1440–1448."
    ),
    theory=r"""
    Fast R-CNN closes the dominant performance gap of R-CNN — the
    one-CNN-forward-per-proposal cost — by sharing convolutional
    computation across all proposals in an image.

    The CNN is run **once** on the full image, producing a feature
    map at stride :math:`s` (e.g. :math:`s = 16` for VGG-16).  For
    each region proposal, **RoI Pooling** extracts a fixed-size
    :math:`H \times W` (paper: :math:`7 \times 7`) sub-window from
    the shared feature map by quantising the proposal coordinates to
    the feature grid and max-pooling adaptively inside each cell.

    The pooled features feed two FC layers and **two sibling output
    heads**:

    .. math::

        L(p, u, t^u, v) = L_{\mathrm{cls}}(p, u)
            + \lambda \, [u \ge 1] \, L_{\mathrm{loc}}(t^u, v),

    where :math:`L_{\mathrm{cls}}` is softmax log-loss over
    :math:`K+1` classes and :math:`L_{\mathrm{loc}}` is the smooth
    :math:`\ell_1` regression loss on the parameterised offsets
    :math:`t^u` for the predicted class :math:`u`.  The Iverson
    bracket :math:`[u \ge 1]` disables regression on background RoIs.

    Beyond the speedup (≈9× training, ≈213× inference over R-CNN),
    Fast R-CNN replaces the SVM + post-hoc regressor with a single
    multi-task network, which **improves accuracy** because gradients
    from both heads jointly tune the shared trunk.
    """,
)
@dataclass(frozen=True)
class FastRCNNConfig(ModelConfig):
    """Configuration for Fast R-CNN.

    Fast R-CNN (Girshick, ICCV 2015) runs the CNN on the **full image once**,
    then extracts per-proposal features with RoI Pooling on the shared feature
    map.  This is the key advance over R-CNN (one forward pass vs one per
    proposal).

    Default backbone: VGG16 conv layers (conv1_1 … conv5_3, pool5 removed).
    The feature map stride is 16 (four max-pool layers before pool5).

    Args:
        num_classes:     Foreground object classes.  Background is class 0,
                         giving (num_classes + 1) output logits.
        in_channels:     Input image channels (3 for RGB).
        roi_size:        RoI Pool output spatial size (paper: 7 → 7×7).
        spatial_scale:   Ratio of feature-map size to input image size.
                         For VGG16 with pool5 removed: 1/16.
        dropout:         Dropout probability after fc6 and fc7.
        bbox_reg_weights: Per-component weight scaling for bbox regression
                          targets (tx, ty, tw, th).  The paper states no
                          target normalisation; the (0, 0, 0, 0) mean /
                          (0.1, 0.1, 0.2, 0.2) std these weights invert
                          comes from the authors' released code
                          (``BBOX_NORMALIZE_STDS``), which every later
                          two-stage detector inherited.
        score_thresh:    Minimum class score at inference time.
        nms_thresh:      Per-class NMS IoU threshold.
        max_detections:  Maximum detections returned per image.

        -- Training-time RoI sampling (§2.3) --
        batch_size_per_image: RoIs sampled per image for the loss (64).
        positive_fraction:    Target foreground share of that minibatch
                              (0.25).  A shortfall of foreground is
                              backfilled with background so the minibatch
                              keeps its size.
        fg_iou_thresh:        IoU at or above which a proposal is
                              foreground.
        bg_iou_thresh_lo:     Lower edge of the hard-negative band;
                              proposals below it are ignored entirely
                              rather than treated as background.

    Examples
    --------
    >>> from lucid.models.vision.fast_rcnn._config import FastRCNNConfig
    >>> cfg = FastRCNNConfig()
    >>> cfg.model_type
    'fast_rcnn'
    >>> cfg.num_classes, cfg.in_channels
    (80, 3)
    """

    model_type: ClassVar[str] = "fast_rcnn"

    num_classes: int = 80
    in_channels: int = 3
    roi_size: int = 7
    spatial_scale: float = 1.0 / 16.0
    dropout: float = 0.5
    bbox_reg_weights: tuple[float, float, float, float] = (10.0, 10.0, 5.0, 5.0)
    score_thresh: float = 0.05
    # NMS IoU threshold: paper section 3 defers to R-CNN's settings, and
    # fast-rcnn's config.py sets __C.TEST.NMS = 0.3.
    nms_thresh: float = 0.3
    max_detections: int = 300

    # §2.3's minibatch: "mini-batches of size R = 128, sampling 64 RoIs from
    # each image ... 25% of the RoIs from object proposals that have IoU
    # overlap with a ground-truth bounding box of at least 0.5 ... The
    # remaining RoIs are sampled from object proposals that have a maximum
    # IoU with ground truth in the interval [0.1, 0.5)."  The reference
    # config spells the same thing as BATCH_SIZE=128 / FG_FRACTION=0.25 with
    # two images per batch.
    batch_size_per_image: int = 64
    positive_fraction: float = 0.25
    fg_iou_thresh: float = 0.5
    bg_iou_thresh_lo: float = 0.1

    def __post_init__(self) -> None:
        object.__setattr__(self, "bbox_reg_weights", tuple(self.bbox_reg_weights))
        owner = type(self).__name__
        check_positive(
            owner,
            num_classes=self.num_classes,
            in_channels=self.in_channels,
            roi_size=self.roi_size,
            spatial_scale=self.spatial_scale,
            max_detections=self.max_detections,
            batch_size_per_image=self.batch_size_per_image,
        )
