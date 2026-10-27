"""Registry factories for EfficientDet variants."""

from lucid.models._registry import register_model
from lucid.models.vision.efficientdet._config import _efficientdet_config
from lucid.models.vision.efficientdet._model import EfficientDetForObjectDetection
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained


@register_model(
    task="object-detection",
    family="efficientdet",
    model_type="efficientdet",
    model_class=EfficientDetForObjectDetection,
    default_config=_efficientdet_config(phi=0),
    params=3900000,
    summary="auto",
)
def efficientdet_d0(
    pretrained: bool = False, **overrides: object
) -> EfficientDetForObjectDetection:
    r"""EfficientDet-D0 (:math:`\varphi = 0`).

    Builds the smallest compound-scaled EfficientDet variant: EfficientNet-B0
    backbone, BiFPN with 64 channels and 3 repeats, head depth 3, and a
    512x512 input resolution.  Approximately 3.9M parameters; COCO test-dev
    mAP of 33.8% (paper Table 2) at 2.5 BFLOPs.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``efficientdet_d0``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides
        Keyword overrides forwarded into :class:`EfficientDetConfig`.

    Returns
    -------
    EfficientDetForObjectDetection
        Detector with the D0 configuration applied (or with ``overrides``
        merged on top of it).

    Notes
    -----
    See Tan et al., "EfficientDet: Scalable and Efficient Object Detection",
    CVPR 2020 (arXiv:1911.09070).  D0 is the smallest of the eight
    compound-scaled variants and the recommended baseline for
    mobile / edge deployment.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.efficientdet import efficientdet_d0
    >>> model = efficientdet_d0()
    >>> x = lucid.randn(1, 3, 512, 512)
    >>> out = model(x)
    >>> out.pred_boxes.shape[-1]
    4
    """
    if pretrained:
        reject_unavailable_pretrained("efficientdet_d0")
    return EfficientDetForObjectDetection(apply_overrides(_efficientdet_config(phi=0), overrides))


@register_model(
    task="object-detection",
    family="efficientdet",
    model_type="efficientdet",
    model_class=EfficientDetForObjectDetection,
    default_config=_efficientdet_config(phi=1),
    params=6600000,
    summary="auto",
)
def efficientdet_d1(
    pretrained: bool = False, **overrides: object
) -> EfficientDetForObjectDetection:
    r"""EfficientDet-D1 (:math:`\varphi = 1`).

    Compound-scaled variant: EfficientNet-B1 backbone, BiFPN 88 channels
    with 4 repeats, head depth 3, 640x640 input.  Approximately 6.6M
    parameters; COCO test-dev mAP of 39.6% (paper Table 2).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``efficientdet_d1``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides
        Keyword overrides forwarded into :class:`EfficientDetConfig`.

    Returns
    -------
    EfficientDetForObjectDetection
        Detector with the D1 configuration applied.

    Notes
    -----
    See Tan et al., 2020 (arXiv:1911.09070), Table 2.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.efficientdet import efficientdet_d1
    >>> model = efficientdet_d1()
    >>> x = lucid.randn(1, 3, 640, 640)
    >>> out = model(x)
    >>> out.pred_boxes.shape[-1]
    4
    """
    if pretrained:
        reject_unavailable_pretrained("efficientdet_d1")
    return EfficientDetForObjectDetection(apply_overrides(_efficientdet_config(phi=1), overrides))


@register_model(
    task="object-detection",
    family="efficientdet",
    model_type="efficientdet",
    model_class=EfficientDetForObjectDetection,
    default_config=_efficientdet_config(phi=2),
    params=8100000,
    summary="auto",
)
def efficientdet_d2(
    pretrained: bool = False, **overrides: object
) -> EfficientDetForObjectDetection:
    r"""EfficientDet-D2 (:math:`\varphi = 2`).

    Compound-scaled variant: EfficientNet-B2 backbone, BiFPN 112 channels
    with 5 repeats, head depth 3, 768x768 input.  Approximately 8.1M
    parameters; COCO test-dev mAP of 43.0% (paper Table 2).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``efficientdet_d2``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides
        Keyword overrides forwarded into :class:`EfficientDetConfig`.

    Returns
    -------
    EfficientDetForObjectDetection
        Detector with the D2 configuration applied.

    Notes
    -----
    See Tan et al., 2020 (arXiv:1911.09070), Table 2.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.efficientdet import efficientdet_d2
    >>> model = efficientdet_d2()
    >>> x = lucid.randn(1, 3, 768, 768)
    >>> out = model(x)
    >>> out.pred_boxes.shape[-1]
    4
    """
    if pretrained:
        reject_unavailable_pretrained("efficientdet_d2")
    return EfficientDetForObjectDetection(apply_overrides(_efficientdet_config(phi=2), overrides))


@register_model(
    task="object-detection",
    family="efficientdet",
    model_type="efficientdet",
    model_class=EfficientDetForObjectDetection,
    default_config=_efficientdet_config(phi=3),
    params=12000000,
    summary="auto",
)
def efficientdet_d3(
    pretrained: bool = False, **overrides: object
) -> EfficientDetForObjectDetection:
    r"""EfficientDet-D3 (:math:`\varphi = 3`).

    Compound-scaled variant: EfficientNet-B3 backbone, BiFPN 160 channels
    with 6 repeats, head depth 4, 896x896 input.  Approximately 12.0M
    parameters; COCO test-dev mAP of 45.8% (paper Table 2).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``efficientdet_d3``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides
        Keyword overrides forwarded into :class:`EfficientDetConfig`.

    Returns
    -------
    EfficientDetForObjectDetection
        Detector with the D3 configuration applied.

    Notes
    -----
    See Tan et al., 2020 (arXiv:1911.09070), Table 2.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.efficientdet import efficientdet_d3
    >>> model = efficientdet_d3()
    >>> x = lucid.randn(1, 3, 896, 896)
    >>> out = model(x)
    >>> out.pred_boxes.shape[-1]
    4
    """
    if pretrained:
        reject_unavailable_pretrained("efficientdet_d3")
    return EfficientDetForObjectDetection(apply_overrides(_efficientdet_config(phi=3), overrides))


@register_model(
    task="object-detection",
    family="efficientdet",
    model_type="efficientdet",
    model_class=EfficientDetForObjectDetection,
    default_config=_efficientdet_config(phi=4),
    params=20700000,
    summary="auto",
)
def efficientdet_d4(
    pretrained: bool = False, **overrides: object
) -> EfficientDetForObjectDetection:
    r"""EfficientDet-D4 (:math:`\varphi = 4`).

    Compound-scaled variant: EfficientNet-B4 backbone, BiFPN 224 channels
    with 7 repeats, head depth 4, 1024x1024 input.  Approximately 20.7M
    parameters; COCO test-dev mAP of 49.4% (paper Table 2).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``efficientdet_d4``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides
        Keyword overrides forwarded into :class:`EfficientDetConfig`.

    Returns
    -------
    EfficientDetForObjectDetection
        Detector with the D4 configuration applied.

    Notes
    -----
    See Tan et al., 2020 (arXiv:1911.09070), Table 2.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.efficientdet import efficientdet_d4
    >>> model = efficientdet_d4()
    >>> x = lucid.randn(1, 3, 1024, 1024)
    >>> out = model(x)
    >>> out.pred_boxes.shape[-1]
    4
    """
    if pretrained:
        reject_unavailable_pretrained("efficientdet_d4")
    return EfficientDetForObjectDetection(apply_overrides(_efficientdet_config(phi=4), overrides))


@register_model(
    task="object-detection",
    family="efficientdet",
    model_type="efficientdet",
    model_class=EfficientDetForObjectDetection,
    default_config=_efficientdet_config(phi=5),
    params=33700000,
    summary="auto",
)
def efficientdet_d5(
    pretrained: bool = False, **overrides: object
) -> EfficientDetForObjectDetection:
    r"""EfficientDet-D5 (:math:`\varphi = 5`).

    Compound-scaled variant: EfficientNet-B5 backbone, BiFPN 288 channels
    with 7 repeats, head depth 4, 1280x1280 input.  Approximately 33.7M
    parameters; COCO test-dev mAP of 50.7% (paper Table 2).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``efficientdet_d5``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides
        Keyword overrides forwarded into :class:`EfficientDetConfig`.

    Returns
    -------
    EfficientDetForObjectDetection
        Detector with the D5 configuration applied.

    Notes
    -----
    See Tan et al., 2020 (arXiv:1911.09070), Table 2.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.efficientdet import efficientdet_d5
    >>> model = efficientdet_d5()
    >>> x = lucid.randn(1, 3, 1280, 1280)
    >>> out = model(x)
    >>> out.pred_boxes.shape[-1]
    4
    """
    if pretrained:
        reject_unavailable_pretrained("efficientdet_d5")
    return EfficientDetForObjectDetection(apply_overrides(_efficientdet_config(phi=5), overrides))


@register_model(
    task="object-detection",
    family="efficientdet",
    model_type="efficientdet",
    model_class=EfficientDetForObjectDetection,
    default_config=_efficientdet_config(phi=6),
    params=51900000,
    summary="auto",
)
def efficientdet_d6(
    pretrained: bool = False, **overrides: object
) -> EfficientDetForObjectDetection:
    r"""EfficientDet-D6 (:math:`\varphi = 6`).

    Compound-scaled variant: EfficientNet-B6 backbone, BiFPN 384 channels
    with 8 repeats, head depth 5, 1280x1280 input.  Approximately 51.9M
    parameters; COCO test-dev mAP of 51.7% (paper Table 2).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``efficientdet_d6``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides
        Keyword overrides forwarded into :class:`EfficientDetConfig`.

    Returns
    -------
    EfficientDetForObjectDetection
        Detector with the D6 configuration applied.

    Notes
    -----
    See Tan et al., 2020 (arXiv:1911.09070), Table 2.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.efficientdet import efficientdet_d6
    >>> model = efficientdet_d6()
    >>> x = lucid.randn(1, 3, 1280, 1280)
    >>> out = model(x)
    >>> out.pred_boxes.shape[-1]
    4
    """
    if pretrained:
        reject_unavailable_pretrained("efficientdet_d6")
    return EfficientDetForObjectDetection(apply_overrides(_efficientdet_config(phi=6), overrides))


@register_model(
    task="object-detection",
    family="efficientdet",
    model_type="efficientdet",
    model_class=EfficientDetForObjectDetection,
    default_config=_efficientdet_config(phi=7),
    params=51900000,
    summary="auto",
)
def efficientdet_d7(
    pretrained: bool = False, **overrides: object
) -> EfficientDetForObjectDetection:
    r"""EfficientDet-D7 (:math:`\varphi = 7`).

    The largest compound-scaled variant: EfficientNet-B6 backbone (shared
    with D6), BiFPN 384 channels with 8 repeats, head depth 5, and a
    1536x1536 input.  Approximately 51.9M parameters; COCO test-dev mAP
    of 53.7% (paper Table 2) — the headline accuracy result.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``efficientdet_d7``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides
        Keyword overrides forwarded into :class:`EfficientDetConfig`.

    Returns
    -------
    EfficientDetForObjectDetection
        Detector with the D7 configuration applied.

    Notes
    -----
    See Tan et al., 2020 (arXiv:1911.09070), Table 2.  The only
    architectural difference from D6 is the larger input resolution;
    backbone / BiFPN / head shapes are shared.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.efficientdet import efficientdet_d7
    >>> model = efficientdet_d7()
    >>> x = lucid.randn(1, 3, 1536, 1536)
    >>> out = model(x)
    >>> out.pred_boxes.shape[-1]
    4
    """
    if pretrained:
        reject_unavailable_pretrained("efficientdet_d7")
    return EfficientDetForObjectDetection(apply_overrides(_efficientdet_config(phi=7), overrides))
