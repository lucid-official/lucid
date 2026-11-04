"""Registry factories for the three paper-cited CSPNet variants.

Wang et al., CVPRW 2020.  Three architectures ship: ``cspresnet_50``,
``cspresnext_50``, ``cspdarknet_53``.  Hyperparameters lifted from
``timm.models.cspnet`` to keep state-dict compatibility (one key map
converts all three variants).
"""

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models.vision.cspnet._config import CSPNetConfig
from lucid.models.vision.cspnet._model import CSPNet, CSPNetForImageClassification
from lucid.models.vision.cspnet._weights import (
    CSPDarknet53Weights,
    CSPResNet50Weights,
    CSPResNeXt50Weights,
)
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained

# ---------------------------------------------------------------------------
# Per-variant configs (timm ``model_cfgs`` values, paper-faithful).
# ---------------------------------------------------------------------------

_CFG_CSPRESNET_50 = CSPNetConfig(
    stem_out_chs=64,
    stem_kernel=7,
    stem_stride=2,
    stem_pool="max",
    depths=(3, 3, 5, 2),
    out_chs=(128, 256, 512, 1024),
    strides=(1, 2, 2, 2),
    groups=(1, 1, 1, 1),
    expand_ratio=(2.0, 2.0, 2.0, 2.0),
    bottle_ratio=(0.5, 0.5, 0.5, 0.5),
    block_ratio=(1.0, 1.0, 1.0, 1.0),
    # Linear in every stage, as the released weights are — darknet's
    # csresnet50.cfg / csresnext50.cfg make stages 1 and 4 leaky instead.
    cross_linear=(True, True, True, True),
    down_growth=(False, False, False, False),
    block_type=("bottle", "bottle", "bottle", "bottle"),
)

_CFG_CSPRESNEXT_50 = CSPNetConfig(
    stem_out_chs=64,
    stem_kernel=7,
    stem_stride=2,
    stem_pool="max",
    depths=(3, 3, 5, 2),
    out_chs=(256, 512, 1024, 2048),
    strides=(1, 2, 2, 2),
    groups=(32, 32, 32, 32),
    expand_ratio=(1.0, 1.0, 1.0, 1.0),
    bottle_ratio=(1.0, 1.0, 1.0, 1.0),
    block_ratio=(0.5, 0.5, 0.5, 0.5),
    # Linear in every stage, as the released weights are — darknet's
    # csresnet50.cfg / csresnext50.cfg make stages 1 and 4 leaky instead.
    cross_linear=(True, True, True, True),
    down_growth=(False, False, False, False),
    block_type=("bottle", "bottle", "bottle", "bottle"),
)

_CFG_CSPDARKNET_53 = CSPNetConfig(
    stem_out_chs=32,
    stem_kernel=3,
    stem_stride=1,
    stem_pool="",
    depths=(1, 2, 8, 8, 4),
    out_chs=(64, 128, 256, 512, 1024),
    strides=(2, 2, 2, 2, 2),
    groups=(1, 1, 1, 1, 1),
    expand_ratio=(2.0, 1.0, 1.0, 1.0, 1.0),
    bottle_ratio=(0.5, 1.0, 1.0, 1.0, 1.0),
    block_ratio=(1.0, 0.5, 0.5, 0.5, 0.5),
    cross_linear=(False, False, False, False, False),
    down_growth=(True, True, True, True, True),
    block_type=("dark", "dark", "dark", "dark", "dark"),
)


# ---------------------------------------------------------------------------
# Backbones — task="base"
# ---------------------------------------------------------------------------


@register_model(
    task="base",
    family="cspnet",
    model_type="cspnet",
    model_class=CSPNet,
    default_config=_CFG_CSPRESNET_50,
    params=21_620_000,
)
def cspresnet_50(pretrained: bool = False, **overrides: object) -> CSPNet:
    r"""CSPResNet-50 backbone — ResNet-50 with CSP-wrapped stages.

    The CSPNet recipe applied to the ResNet-50 trunk (Wang et al.,
    CVPRW 2020).  Returns the final feature map only; pair with a
    classifier head via :func:`cspresnet_50_cls`.

    Reference: Wang et al., *"CSPNet: A New Backbone that can Enhance
    Learning Capability of CNN"*, CVPRW 2020 (arXiv:1911.11929).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``cspresnet_50``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``cspresnet_50_cls`` is the checkpointed variant.
    **overrides : object
        Fields of :class:`CSPNetConfig` to change from the CSPResNet-50
        defaults (e.g. ``in_channels=1``).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.cspnet import cspresnet_50
    >>> model = cspresnet_50()
    >>> out = model(lucid.randn(1, 3, 224, 224))
    >>> out.last_hidden_state.shape
    (1, 1024, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained("cspresnet_50", alternative="cspresnet_50_cls")
    return CSPNet(apply_overrides(_CFG_CSPRESNET_50, overrides))


@register_model(
    task="base",
    family="cspnet",
    model_type="cspnet",
    model_class=CSPNet,
    default_config=_CFG_CSPRESNEXT_50,
    params=20_570_000,
)
def cspresnext_50(pretrained: bool = False, **overrides: object) -> CSPNet:
    r"""CSPResNeXt-50 backbone — ResNeXt-50 (32x4d) with CSP wrap.

    The CSPNet recipe applied to the grouped-conv ResNeXt-50 trunk
    (Wang et al., CVPRW 2020).  Returns the final feature map only;
    pair with a classifier head via :func:`cspresnext_50_cls`.

    Reference: Wang et al., *"CSPNet: A New Backbone that can Enhance
    Learning Capability of CNN"*, CVPRW 2020 (arXiv:1911.11929).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``cspresnext_50``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``cspresnext_50_cls`` is the checkpointed variant.
    **overrides : object
        Fields of :class:`CSPNetConfig` to change from the CSPResNeXt-50
        defaults (e.g. ``in_channels=1``).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.cspnet import cspresnext_50
    >>> model = cspresnext_50()
    >>> out = model(lucid.randn(1, 3, 224, 224))
    >>> out.last_hidden_state.shape
    (1, 2048, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained("cspresnext_50", alternative="cspresnext_50_cls")
    return CSPNet(apply_overrides(_CFG_CSPRESNEXT_50, overrides))


@register_model(
    task="base",
    family="cspnet",
    model_type="cspnet",
    model_class=CSPNet,
    default_config=_CFG_CSPDARKNET_53,
    params=27_610_000,
)
def cspdarknet_53(pretrained: bool = False, **overrides: object) -> CSPNet:
    r"""CSPDarknet-53 backbone — Darknet-53 (YOLOv3 base) with CSP wrap.

    The CSPNet recipe applied to the Darknet-53 trunk (Wang et al.,
    CVPRW 2020); same architecture later adopted by YOLOv4.  Returns
    the final feature map only; pair with a classifier head via
    :func:`cspdarknet_53_cls`.

    Reference: Wang et al., *"CSPNet: A New Backbone that can Enhance
    Learning Capability of CNN"*, CVPRW 2020 (arXiv:1911.11929).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``cspdarknet_53``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``cspdarknet_53_cls`` is the checkpointed variant.
    **overrides : object
        Fields of :class:`CSPNetConfig` to change from the CSPDarknet-53
        defaults (e.g. ``in_channels=1``).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.cspnet import cspdarknet_53
    >>> model = cspdarknet_53()
    >>> out = model(lucid.randn(1, 3, 224, 224))
    >>> out.last_hidden_state.shape
    (1, 1024, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained("cspdarknet_53", alternative="cspdarknet_53_cls")
    return CSPNet(apply_overrides(_CFG_CSPDARKNET_53, overrides))


# ---------------------------------------------------------------------------
# Classifiers — task="image-classification"
# ---------------------------------------------------------------------------


@register_model(
    task="image-classification",
    family="cspnet",
    model_type="cspnet",
    model_class=CSPNetForImageClassification,
    default_config=_CFG_CSPRESNET_50,
    params=21_620_000,
)
def cspresnet_50_cls(
    pretrained: bool | str = False,
    *,
    weights: CSPResNet50Weights | None = None,
    **overrides: object,
) -> CSPNetForImageClassification:
    r"""CSPResNet-50 image classifier — paper Table 1, **21.6M** params,
    ImageNet-1k acc@1 ~76.2% (timm ``ra_in1k`` recipe).

    Parameters
    ----------
    pretrained : bool or str, default ``False``
        ``False`` returns randomly-initialised weights; ``True`` loads
        the default entry of :class:`CSPResNet50Weights`; a string
        selects a named entry from that enum.
    weights : CSPResNet50Weights, optional
        Explicit weight-enum member overriding ``pretrained``.
    **overrides : object
        Per-field overrides forwarded to :class:`CSPNetConfig` (e.g.
        ``num_classes=10`` to retarget the head).

    Returns
    -------
    CSPNetForImageClassification
        The configured classifier.

    Notes
    -----
    Reference: Wang et al., *"CSPNet: A New Backbone that can Enhance
    Learning Capability of CNN"*, CVPRW 2020 (arXiv:1911.11929).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.cspnet import cspresnet_50_cls
    >>> model = cspresnet_50_cls()
    >>> out = model(lucid.randn(1, 3, 224, 224))
    >>> out.logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(CSPResNet50Weights, pretrained, weights)
    model = CSPNetForImageClassification(apply_overrides(_CFG_CSPRESNET_50, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="cspresnet_50_cls")
    return model


@register_model(
    task="image-classification",
    family="cspnet",
    model_type="cspnet",
    model_class=CSPNetForImageClassification,
    default_config=_CFG_CSPRESNEXT_50,
    params=20_570_000,
)
def cspresnext_50_cls(
    pretrained: bool | str = False,
    *,
    weights: CSPResNeXt50Weights | None = None,
    **overrides: object,
) -> CSPNetForImageClassification:
    r"""CSPResNeXt-50 image classifier — paper Table 1, **20.6M**
    params, ImageNet-1k acc@1 ~80.0% (timm ``ra_in1k`` recipe).

    Parameters
    ----------
    pretrained : bool or str, default ``False``
        ``False`` returns randomly-initialised weights; ``True`` loads
        the default entry of :class:`CSPResNeXt50Weights`; a string
        selects a named entry from that enum.
    weights : CSPResNeXt50Weights, optional
        Explicit weight-enum member overriding ``pretrained``.
    **overrides : object
        Per-field overrides forwarded to :class:`CSPNetConfig`.

    Returns
    -------
    CSPNetForImageClassification
        The configured classifier.

    Notes
    -----
    Reference: Wang et al., *"CSPNet: A New Backbone that can Enhance
    Learning Capability of CNN"*, CVPRW 2020 (arXiv:1911.11929).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.cspnet import cspresnext_50_cls
    >>> model = cspresnext_50_cls()
    >>> out = model(lucid.randn(1, 3, 224, 224))
    >>> out.logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(CSPResNeXt50Weights, pretrained, weights)
    model = CSPNetForImageClassification(apply_overrides(_CFG_CSPRESNEXT_50, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="cspresnext_50_cls")
    return model


@register_model(
    task="image-classification",
    family="cspnet",
    model_type="cspnet",
    model_class=CSPNetForImageClassification,
    default_config=_CFG_CSPDARKNET_53,
    params=27_610_000,
)
def cspdarknet_53_cls(
    pretrained: bool | str = False,
    *,
    weights: CSPDarknet53Weights | None = None,
    **overrides: object,
) -> CSPNetForImageClassification:
    r"""CSPDarknet-53 image classifier — paper Table 1, **27.6M**
    params, ImageNet-1k acc@1 ~80.1% (timm ``ra_in1k`` recipe).

    Parameters
    ----------
    pretrained : bool or str, default ``False``
        ``False`` returns randomly-initialised weights; ``True`` loads
        the default entry of :class:`CSPDarknet53Weights`; a string
        selects a named entry from that enum.
    weights : CSPDarknet53Weights, optional
        Explicit weight-enum member overriding ``pretrained``.
    **overrides : object
        Per-field overrides forwarded to :class:`CSPNetConfig`.

    Returns
    -------
    CSPNetForImageClassification
        The configured classifier.

    Notes
    -----
    Reference: Wang et al., *"CSPNet: A New Backbone that can Enhance
    Learning Capability of CNN"*, CVPRW 2020 (arXiv:1911.11929).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.cspnet import cspdarknet_53_cls
    >>> model = cspdarknet_53_cls()
    >>> out = model(lucid.randn(1, 3, 224, 224))
    >>> out.logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(CSPDarknet53Weights, pretrained, weights)
    model = CSPNetForImageClassification(apply_overrides(_CFG_CSPDARKNET_53, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="cspdarknet_53_cls")
    return model
