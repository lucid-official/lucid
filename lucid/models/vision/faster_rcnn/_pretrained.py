"""Registry factories for Faster R-CNN variants."""

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models.vision.faster_rcnn._config import FasterRCNNConfig
from lucid.models.vision.faster_rcnn._model import FasterRCNNForObjectDetection
from lucid.models.vision.faster_rcnn._weights import FasterRCNNResNet50FPNWeights
from lucid.models._utils._common import apply_overrides

# COCO config — the pretrained checkpoint uses num_classes=91 (90 COCO
# categories + background slot 0).
_CFG_R50_FPN = FasterRCNNConfig(num_classes=91)


@register_model(
    task="object-detection",
    family="faster_rcnn",
    model_type="faster_rcnn",
    model_class=FasterRCNNForObjectDetection,
    default_config=_CFG_R50_FPN,
)
def faster_rcnn(
    pretrained: bool | str = False,
    *,
    weights: FasterRCNNResNet50FPNWeights | None = None,
    **overrides: object,
) -> FasterRCNNForObjectDetection:
    r"""Faster R-CNN with a ResNet-50-FPN backbone (Ren et al., NeurIPS 2015).

    Alias of :func:`faster_rcnn_resnet50_fpn` — the canonical reference
    configuration: a ResNet-50 trunk (frozen BN) feeding a Feature Pyramid
    Network, a Region Proposal Network over five pyramid levels, and a Fast
    R-CNN RoI head.  The COCO config uses ``num_classes = 91``.  Reaches
    COCO box AP of 37.0 (reference ``fasterrcnn_resnet50_fpn``).

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True`` →
        the ``DEFAULT`` tag (:attr:`FasterRCNNResNet50FPNWeights.COCO_V1`);
        a tag string → that specific checkpoint.  Mutually exclusive with
        ``weights`` (which wins if both are given).
    weights : FasterRCNNResNet50FPNWeights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`FasterRCNNConfig`
        (``num_classes``, ``rpn_pre_nms_top_n``, ``score_thresh``, ...).

    Returns
    -------
    FasterRCNNForObjectDetection
        Detector with the ResNet-50-FPN configuration applied (or with
        ``overrides`` merged on top of it).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.faster_rcnn import faster_rcnn
    >>> model = faster_rcnn(num_classes=91).eval()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape[-1]
    91
    """
    entry = weights_mod.resolve_weights(FasterRCNNResNet50FPNWeights, pretrained, weights)
    model = FasterRCNNForObjectDetection(apply_overrides(_CFG_R50_FPN, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="faster_rcnn")
    return model


@register_model(
    task="object-detection",
    family="faster_rcnn",
    model_type="faster_rcnn",
    model_class=FasterRCNNForObjectDetection,
    default_config=_CFG_R50_FPN,
)
def faster_rcnn_resnet50_fpn(
    pretrained: bool | str = False,
    *,
    weights: FasterRCNNResNet50FPNWeights | None = None,
    **overrides: object,
) -> FasterRCNNForObjectDetection:
    r"""Faster R-CNN with a ResNet-50-FPN backbone (COCO-pretrained).

    Builds the reference ``fasterrcnn_resnet50_fpn`` detector: ResNet-50
    backbone with frozen batch-norm, a 5-level Feature Pyramid Network
    (P2-P5 + a parameter-free pool level for RPN), anchors at sizes
    ``((32,),(64,),(128,),(256,),(512,))`` and ratios ``(0.5, 1.0, 2.0)``,
    and a Fast R-CNN RoI head (TwoMLPHead + FastRCNNPredictor) over
    MultiScale RoI Align.  COCO box AP 37.0, ~41.8M parameters.

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True`` →
        the ``DEFAULT`` tag (:attr:`FasterRCNNResNet50FPNWeights.COCO_V1`);
        a tag string → that specific checkpoint.  Mutually exclusive with
        ``weights`` (which wins if both are given).
    weights : FasterRCNNResNet50FPNWeights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`FasterRCNNConfig`.

    Returns
    -------
    FasterRCNNForObjectDetection
        Detector with the ResNet-50-FPN configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    Pretrained weights are converted from the reference
    ``FasterRCNN_ResNet50_FPN_Weights.COCO_V1`` checkpoint and hosted under
    ``lucid-dl/faster-rcnn-resnet-50-fpn``.  The detector expects an
    already resized + normalised image batch (the
    :class:`~lucid.utils.transforms.Detection` preset).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.faster_rcnn import faster_rcnn_resnet50_fpn
    >>> model = faster_rcnn_resnet50_fpn().eval()
    >>> out = model(lucid.randn(1, 3, 224, 224))
    >>> out.logits.shape[-1]
    91
    """
    entry = weights_mod.resolve_weights(FasterRCNNResNet50FPNWeights, pretrained, weights)
    model = FasterRCNNForObjectDetection(apply_overrides(_CFG_R50_FPN, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="faster_rcnn_resnet50_fpn")
    return model
