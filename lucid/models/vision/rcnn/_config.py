"""R-CNN configuration (Girshick et al., 2014)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta
from lucid.models._utils._common import check_positive


@model_family_meta(
    canonical_name="R-CNN",
    citation=(
        'Girshick, Ross, et al. "Rich Feature Hierarchies for Accurate '
        'Object Detection and Semantic Segmentation." Proceedings of the '
        "IEEE Conference on Computer Vision and Pattern Recognition, "
        "2014, pp. 580–587."
    ),
    theory=r"""
    R-CNN (Regions with CNN features) is the first deep-learning
    detector to demonstrate that ImageNet-pretrained CNN features
    transfer to localisation.  The pipeline is deliberately modular:

    1. **Region proposals.**  An external method (selective search in
       the original paper) yields :math:`\sim 2000` class-agnostic
       candidate windows per image.
    2. **CNN feature extraction.**  Each proposal is *warped* to a
       fixed :math:`227 \times 227` patch and forwarded through an
       ImageNet-pretrained CNN (AlexNet).  The ``pool5`` activations
       (effectively :math:`6 \times 6 \times 256 = 9216` features) are
       flattened and passed through two fully-connected layers.
    3. **Per-class linear SVMs** score each region; a class-specific
       bounding-box regressor refines the proposal coordinates with
       a parameterised :math:`(t_x, t_y, t_w, t_h)` offset.

    The cost is dominated by **redundant CNN forward passes** — one
    per proposal — because no feature sharing across overlapping
    regions exists.  Despite that inefficiency, R-CNN jumped mean
    average precision on PASCAL VOC 2012 from :math:`\sim 35\%`
    (DPM) to :math:`53.3\%`, making CNNs the dominant detection
    paradigm and motivating every successor in the family
    (Fast R-CNN, Faster R-CNN, Mask R-CNN).
    """,
)
@dataclass(frozen=True)
class RCNNConfig(ModelConfig):
    """Configuration for R-CNN.

    R-CNN (Girshick et al., CVPR 2014) applies a CNN independently to each
    region proposal (warped to a fixed ``roi_size × roi_size`` crop), then
    feeds the flattened pool5 features through two FC layers before the
    classification and bounding-box regression heads.

    Original paper uses AlexNet (5 conv + 3 FC) as the backbone.  The
    effective output of pool5 for a 227 × 227 input is 6 × 6 × 256 = 9 216.

    Args:
        num_classes:     Number of foreground object classes.
                         Background is automatically added as class 0,
                         giving (num_classes + 1) output logits.
        in_channels:     Number of input image channels (3 for RGB).
        context_pad:     Pixels of image context kept around each proposal
                         in the *warped* frame (Appendix A's :math:`p`).
                         ``0`` warps the tight box.
        roi_size:        Each region proposal is warped to this square size
                         before being forwarded through the backbone.
                         Original paper: 227.
        dropout:         Dropout probability applied after fc6 and fc7.
        score_thresh:    Minimum class-score threshold applied during
                         post-processing (after softmax).  Boxes whose
                         max class score is below this value are discarded.
        nms_thresh:      IoU threshold for per-class NMS at inference time.
        max_detections:  Maximum number of detections returned per image
                         after NMS.

    .. note::

       **Training — two of the paper's three stages.**  Passing
       ``targets`` to :meth:`forward` trains stage 1 (CNN fine-tuning,
       softmax cross-entropy over proposals at IoU >= ``fg_iou_thresh``)
       and stage 3 (the class-specific bounding-box regressors on pool5
       features, restricted to proposals at IoU > ``bbox_reg_iou_thresh``
       per Appendix C).

       Stage 2 — the per-class linear SVMs that *replace* the softmax at
       test time — is **not** part of the model.  It is an offline pass
       over features cached from a frozen network, with positives drawn
       only from ground-truth boxes, negatives at IoU < 0.3, and hard
       negative mining across the whole dataset; the mining is what makes
       it work, and it is a pipeline rather than a layer.
       :func:`lucid.models._utils._common.mine_hard_negatives` implements
       that loop over cached features — score the pool, keep the
       highest-scoring mistakes, refit, repeat — on top of
       :func:`~lucid.models._utils._common.fit_linear_svm`.  Extracting the
       features from a frozen network over a whole dataset, and scoring
       with the resulting weights instead of the softmax, remain the
       caller's pipeline.

    Examples
    --------
    >>> from lucid.models.vision.rcnn._config import RCNNConfig
    >>> cfg = RCNNConfig()
    >>> cfg.model_type
    'rcnn'
    >>> cfg.num_classes, cfg.in_channels
    (80, 3)
    """

    model_type: ClassVar[str] = "rcnn"

    num_classes: int = 80
    in_channels: int = 3
    roi_size: int = 227
    # Appendix A: the tight box is dilated so the warped crop carries
    # exactly p pixels of context on each side.  The released code uses
    # crop_padding of 16.
    context_pad: int = 16
    dropout: float = 0.5
    score_thresh: float = 0.05
    # NMS IoU threshold: the author's release hardcodes 0.3 (rcnn_test_bbox_regressor.m);
    # 0.5 is the Faster R-CNN convention, not R-CNN's.
    nms_thresh: float = 0.3
    max_detections: int = 300

    # §2.3's fine-tuning assignment: a proposal at IoU >= 0.5 with any
    # ground-truth box is a positive for that box's class, everything else
    # is background.
    fg_iou_thresh: float = 0.5
    # Appendix C trains the regressors only on proposals that already
    # overlap well — "we only learn from a proposal P if it is *near* at
    # least one ground-truth box" — with the released code using 0.6.
    # A looser threshold trains the regressor on boxes it cannot fix.
    bbox_reg_iou_thresh: float = 0.6

    def __post_init__(self) -> None:
        owner = type(self).__name__
        check_positive(owner, num_classes=self.num_classes, in_channels=self.in_channels)
        check_positive(owner, roi_size=self.roi_size)
