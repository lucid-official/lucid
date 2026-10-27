"""Registry factories for FCN variants."""

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models.vision.fcn._config import FCNConfig
from lucid.models.vision.fcn._model import FCNForSemanticSegmentation
from lucid.models.vision.fcn._weights import FCNResNet50Weights, FCNResNet101Weights
from lucid.models._utils._common import apply_overrides

_CFG_RESNET50 = FCNConfig(
    num_classes=21,
    in_channels=3,
    backbone="resnet50",
    variant="fcn32s",
    classifier_hidden_channels=512,
    aux_hidden_channels=256,
    dropout=0.1,
)

_CFG_RESNET101 = FCNConfig(
    num_classes=21,
    in_channels=3,
    backbone="resnet101",
    variant="fcn32s",
    classifier_hidden_channels=512,
    aux_hidden_channels=256,
    dropout=0.1,
)


@register_model(
    task="semantic-segmentation",
    family="fcn",
    model_type="fcn",
    model_class=FCNForSemanticSegmentation,
    default_config=_CFG_RESNET50,
)
def fcn_resnet50(
    pretrained: bool | str = False,
    *,
    weights: FCNResNet50Weights | None = None,
    **overrides: object,
) -> FCNForSemanticSegmentation:
    r"""FCN with ResNet-50 backbone (Long et al., CVPR 2015).

    Builds an :class:`FCNForSemanticSegmentation` with a dilated
    ResNet-50 backbone (``layer3`` dilation = 2, ``layer4`` dilation = 4),
    21 output classes (Pascal VOC default), a 512-channel main FCN head,
    and a 256-channel auxiliary head.  The dilation trick keeps the
    final feature map at stride 8 instead of stride 32, retaining 16x
    more spatial sampling for dense prediction.

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True`` →
        the ``DEFAULT`` tag (:attr:`FCNResNet50Weights.COCO_WITH_VOC_LABELS_V1`);
        a tag string → that specific checkpoint.  Mutually exclusive with
        ``weights`` (which wins if both are given).
    weights : FCNResNet50Weights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`FCNConfig` (``num_classes``,
        ``in_channels``, ``classifier_hidden_channels``, ``dropout``, ...).

    Returns
    -------
    FCNForSemanticSegmentation
        Segmentation model with the FCN-R50 configuration applied (or
        with ``overrides`` merged on top of it).

    Notes
    -----
    See Long et al., "Fully Convolutional Networks for Semantic
    Segmentation", CVPR 2015 (arXiv:1411.4038).  The dilated variant
    follows the DeepLab v1 trick (Chen et al., 2015) of replacing
    stride-2 downsampling in layer3 / layer4 with atrous convolution.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.fcn import fcn_resnet50
    >>> model = fcn_resnet50()
    >>> x = lucid.randn(1, 3, 512, 512)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 21, 512, 512)
    """
    entry = weights_mod.resolve_weights(FCNResNet50Weights, pretrained, weights)
    model = FCNForSemanticSegmentation(apply_overrides(_CFG_RESNET50, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="fcn_resnet50")
    return model


@register_model(
    task="semantic-segmentation",
    family="fcn",
    model_type="fcn",
    model_class=FCNForSemanticSegmentation,
    default_config=_CFG_RESNET101,
)
def fcn_resnet101(
    pretrained: bool | str = False,
    *,
    weights: FCNResNet101Weights | None = None,
    **overrides: object,
) -> FCNForSemanticSegmentation:
    r"""FCN with ResNet-101 backbone (Long et al., CVPR 2015).

    Builds an :class:`FCNForSemanticSegmentation` with the same head
    configuration as :func:`fcn_resnet50` but a deeper dilated ResNet-101
    backbone (23 blocks in layer3 vs 6 for ResNet-50).  Typically gains
    1-2 mIoU points on Pascal VOC at the cost of ~70% more FLOPs.

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True`` →
        the ``DEFAULT`` tag (:attr:`FCNResNet101Weights.COCO_WITH_VOC_LABELS_V1`);
        a tag string → that specific checkpoint.  Mutually exclusive with
        ``weights`` (which wins if both are given).
    weights : FCNResNet101Weights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`FCNConfig`.

    Returns
    -------
    FCNForSemanticSegmentation
        Segmentation model with the FCN-R101 configuration applied (or
        with ``overrides`` merged on top of it).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.fcn import fcn_resnet101
    >>> model = fcn_resnet101()
    >>> x = lucid.randn(1, 3, 512, 512)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 21, 512, 512)
    """
    entry = weights_mod.resolve_weights(FCNResNet101Weights, pretrained, weights)
    model = FCNForSemanticSegmentation(apply_overrides(_CFG_RESNET101, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="fcn_resnet101")
    return model
