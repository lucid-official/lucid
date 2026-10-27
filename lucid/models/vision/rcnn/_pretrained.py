"""Registry factories for R-CNN variants."""

from lucid.models._registry import register_model
from lucid.models.vision.rcnn._config import RCNNConfig
from lucid.models.vision.rcnn._model import RCNNForObjectDetection
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained

# ---------------------------------------------------------------------------
# Base configurations (matching original paper)
# ---------------------------------------------------------------------------

# AlexNet backbone — original paper (CVPR 2014)
_CFG_ALEXNET = RCNNConfig(
    num_classes=80,
    in_channels=3,
    roi_size=227,
    dropout=0.5,
    score_thresh=0.05,
    # The released code uses 0.3; 0.5 is the Faster R-CNN convention.
    nms_thresh=0.3,
    max_detections=300,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Object-detection factories
# ---------------------------------------------------------------------------


@register_model(
    task="object-detection",
    family="rcnn",
    model_type="rcnn",
    model_class=RCNNForObjectDetection,
    default_config=_CFG_ALEXNET,
)
def rcnn(pretrained: bool = False, **overrides: object) -> RCNNForObjectDetection:
    r"""R-CNN with AlexNet backbone (Girshick et al., CVPR 2014).

    Builds a :class:`RCNNForObjectDetection` with the paper-cited AlexNet
    topology: five-conv trunk applied to each region proposal warped to
    :math:`227 \times 227`, followed by two FC layers and sibling
    classification (``num_classes + 1``) and class-specific bounding-box
    regression (``4 \cdot num_classes``) heads.  Original PASCAL VOC 2010
    test mAP of 53.3% (paper Table 9) — a ~20-point jump over the prior
    DPM state of the art.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``rcnn``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides
        Keyword overrides forwarded into :class:`RCNNConfig` (e.g.
        ``num_classes=20`` for PASCAL VOC, ``score_thresh=0.3`` for a
        tighter inference threshold, ``roi_size=224`` to match a
        non-default backbone receptive field).

    Returns
    -------
    RCNNForObjectDetection
        Detector with the AlexNet R-CNN configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See Girshick et al., "Rich Feature Hierarchies for Accurate Object
    Detection and Semantic Segmentation", CVPR 2014 (arXiv:1311.2524).
    Region proposals must be supplied externally to :meth:`forward` —
    the original paper uses selective search (~2000 proposals per image),
    but any class-agnostic proposal method works.  At inference, classes
    are scored via softmax over the :math:`K + 1` logits and bounding
    boxes are refined per top-class delta:

    .. math::

        \hat{G}_w = P_w \exp(t_w),\quad
        \hat{G}_x = P_w t_x + P_x.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.rcnn import rcnn
    >>> model = rcnn(num_classes=20)   # PASCAL VOC
    >>> x = lucid.randn(1, 3, 600, 600)
    >>> proposals = [lucid.tensor(
    ...     [[10.0, 10.0, 200.0, 200.0],
    ...      [50.0, 60.0, 300.0, 280.0]])]
    >>> out = model(x, proposals)
    >>> out.logits.shape
    (2, 21)
    """
    if pretrained:
        reject_unavailable_pretrained("rcnn")
    return RCNNForObjectDetection(apply_overrides(_CFG_ALEXNET, overrides))
