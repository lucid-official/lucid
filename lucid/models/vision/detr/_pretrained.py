"""Registry factories for DETR variants."""

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models.vision.detr._config import DETRConfig
from lucid.models.vision.detr._model import DETRForObjectDetection
from lucid.models.vision.detr._weights import DETRResNet50Weights, DETRResNet101Weights
from lucid.models._utils._common import apply_overrides

# COCO 2017 configs — the pretrained checkpoints use num_classes=91
# (91 foreground + 1 no-object slot).
_CFG_R50 = DETRConfig(
    num_classes=91,
    in_channels=3,
    backbone_layers=(3, 4, 6, 3),
    d_model=256,
    n_head=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout=0.1,
    num_queries=100,
    num_bbox_layers=3,
    bbox_hidden_dim=256,
    score_thresh=0.7,
)

_CFG_R101 = DETRConfig(
    num_classes=91,
    in_channels=3,
    backbone_layers=(3, 4, 23, 3),  # ResNet-101
    d_model=256,
    n_head=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout=0.1,
    num_queries=100,
    num_bbox_layers=3,
    bbox_hidden_dim=256,
    score_thresh=0.7,
)


@register_model(
    task="object-detection",
    family="detr",
    model_type="detr",
    model_class=DETRForObjectDetection,
    default_config=_CFG_R50,
)
def detr_resnet50(
    pretrained: bool | str = False,
    *,
    weights: DETRResNet50Weights | None = None,
    **overrides: object,
) -> DETRForObjectDetection:
    r"""DETR with ResNet-50 backbone (Carion et al., ECCV 2020).

    Builds a :class:`DETRForObjectDetection` with the paper-cited
    ResNet-50 + transformer configuration: 100 object queries, a 6-layer
    encoder + 6-layer decoder with ``d_model = 256``, 8 attention heads,
    and ``dim_feedforward = 2048``.  The COCO config uses
    ``num_classes = 91``.  Reaches COCO box mAP of 42.0 (paper Table 1).
    Approximately 41.5M parameters with a notable *no-anchor / no-NMS*
    design.

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`DETRResNet50Weights.COCO_2017`); a
        tag string → that specific checkpoint.  Mutually exclusive with
        ``weights`` (which wins if both are given).
    weights : DETRResNet50Weights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`DETRConfig`
        (``num_queries``, ``num_classes``, ``dropout``,
        ``num_encoder_layers``, ``num_decoder_layers``, ``d_model``, ...).

    Returns
    -------
    DETRForObjectDetection
        Detector with the DETR ResNet-50 configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See Carion et al., "End-to-End Object Detection with Transformers",
    ECCV 2020 (arXiv:2005.12872).  Pretrained weights are converted from
    the original Facebook DETR ``detr_resnet50`` checkpoint and hosted
    under ``lucid-dl/detr-resnet-50``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.detr import detr_resnet50
    >>> model = detr_resnet50()
    >>> x = lucid.randn(1, 3, 800, 800)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 100, 92)
    """
    entry = weights_mod.resolve_weights(DETRResNet50Weights, pretrained, weights)
    model = DETRForObjectDetection(apply_overrides(_CFG_R50, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="detr_resnet50")
    return model


@register_model(
    task="object-detection",
    family="detr",
    model_type="detr",
    model_class=DETRForObjectDetection,
    default_config=_CFG_R101,
)
def detr_resnet101(
    pretrained: bool | str = False,
    *,
    weights: DETRResNet101Weights | None = None,
    **overrides: object,
) -> DETRForObjectDetection:
    r"""DETR with ResNet-101 backbone (Carion et al., ECCV 2020).

    Builds a :class:`DETRForObjectDetection` with the same transformer
    head as :func:`detr_resnet50` but a deeper ResNet-101 backbone
    (``[3, 4, 23, 3]`` bottleneck blocks).  The COCO config uses
    ``num_classes = 91``.  Approximately 60.5M parameters and COCO box
    mAP of 43.5 (paper Table 1, DETR-R101 row).

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`DETRResNet101Weights.COCO_2017`); a
        tag string → that specific checkpoint.  Mutually exclusive with
        ``weights`` (which wins if both are given).
    weights : DETRResNet101Weights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`DETRConfig`.

    Returns
    -------
    DETRForObjectDetection
        Detector with the DETR ResNet-101 configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See Carion et al., "End-to-End Object Detection with Transformers",
    ECCV 2020 (arXiv:2005.12872).  Switching to ResNet-101 buys ~1.5
    points of AP at the cost of ~50% more parameters in the backbone;
    the transformer head is unchanged.  Pretrained weights are converted
    from the original Facebook DETR ``detr_resnet101`` checkpoint and
    hosted under ``lucid-dl/detr-resnet-101``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.detr import detr_resnet101
    >>> model = detr_resnet101()
    >>> x = lucid.randn(1, 3, 800, 800)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 100, 92)
    """
    entry = weights_mod.resolve_weights(DETRResNet101Weights, pretrained, weights)
    model = DETRForObjectDetection(apply_overrides(_CFG_R101, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="detr_resnet101")
    return model
