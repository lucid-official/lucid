"""Registry factories for MobileNet v1."""

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models.vision.mobilenet._config import MobileNetConfig
from lucid.models.vision.mobilenet._model import MobileNet, MobileNetForImageClassification
from lucid.models.vision.mobilenet._weights import MobileNetWeights
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained

_CFG_100 = MobileNetConfig(width_mult=1.0)
_CFG_075 = MobileNetConfig(width_mult=0.75)
_CFG_050 = MobileNetConfig(width_mult=0.5)
_CFG_025 = MobileNetConfig(width_mult=0.25)


# ── Backbones ─────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="mobilenet",
    model_type="mobilenet",
    model_class=MobileNet,
    default_config=_CFG_100,
    params=3_206_976,
)
def mobilenet(pretrained: bool = False, **overrides: object) -> MobileNet:
    r"""MobileNet-v1 backbone at width multiplier :math:`\alpha = 1.0`.

    Builds a :class:`MobileNet` with the canonical paper topology:
    a 3×3 stem (stride 2) followed by 13 depthwise+pointwise blocks,
    yielding approximately 4.2M parameters.  Howard et al., 2017
    report a 70.6% ImageNet-1k top-1 validation accuracy with this
    configuration (Table 6).  The default choice when the full
    accuracy budget is available.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``mobilenet``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``mobilenet_cls`` is the checkpointed variant.
    **overrides
        Keyword overrides forwarded into :class:`MobileNetConfig`
        (e.g. ``in_channels=1`` for grayscale input).

    Returns
    -------
    MobileNet
        Backbone with the MobileNet-v1 (:math:`\alpha = 1.0`)
        configuration applied (or with ``overrides`` merged on top
        of it).

    Notes
    -----
    See Howard et al., "MobileNets: Efficient Convolutional Neural
    Networks for Mobile Vision Applications", arXiv:1704.04861, 2017,
    Table 1 and Table 6.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mobilenet import mobilenet
    >>> model = mobilenet()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 1024, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained("mobilenet", alternative="mobilenet_cls")
    return MobileNet(apply_overrides(_CFG_100, overrides))


@register_model(
    task="base",
    family="mobilenet",
    model_type="mobilenet",
    model_class=MobileNet,
    default_config=_CFG_075,
    params=1_816_560,
)
def mobilenet_075(pretrained: bool = False, **overrides: object) -> MobileNet:
    r"""MobileNet-v1 backbone at width multiplier :math:`\alpha = 0.75`.

    Builds a :class:`MobileNet` with every channel count multiplied
    by 0.75 — approximately 2.6M parameters.  Howard et al., 2017
    report 68.4% ImageNet-1k top-1 accuracy with this configuration
    (Table 6), at roughly 60% of the FLOPs of the full-width model.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``mobilenet_075``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``mobilenet_cls`` is the checkpointed variant.
    **overrides
        Keyword overrides forwarded into :class:`MobileNetConfig`.

    Returns
    -------
    MobileNet
        Backbone with the MobileNet-v1 (:math:`\alpha = 0.75`)
        configuration applied (or with ``overrides`` merged on top
        of it).

    Notes
    -----
    See Howard et al., "MobileNets: Efficient Convolutional Neural
    Networks for Mobile Vision Applications", arXiv:1704.04861, 2017,
    Table 6.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mobilenet import mobilenet_075
    >>> model = mobilenet_075()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 768, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained("mobilenet_075", alternative="mobilenet_cls")
    return MobileNet(apply_overrides(_CFG_075, overrides))


@register_model(
    task="base",
    family="mobilenet",
    model_type="mobilenet",
    model_class=MobileNet,
    default_config=_CFG_050,
    params=818_592,
)
def mobilenet_050(pretrained: bool = False, **overrides: object) -> MobileNet:
    r"""MobileNet-v1 backbone at width multiplier :math:`\alpha = 0.5`.

    Builds a :class:`MobileNet` with every channel count multiplied
    by 0.5 — approximately 1.3M parameters.  Howard et al., 2017
    report 63.7% ImageNet-1k top-1 accuracy with this configuration
    (Table 6), at roughly 27% of the FLOPs of the full-width model.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``mobilenet_050``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``mobilenet_cls`` is the checkpointed variant.
    **overrides
        Keyword overrides forwarded into :class:`MobileNetConfig`.

    Returns
    -------
    MobileNet
        Backbone with the MobileNet-v1 (:math:`\alpha = 0.5`)
        configuration applied (or with ``overrides`` merged on top
        of it).

    Notes
    -----
    See Howard et al., "MobileNets: Efficient Convolutional Neural
    Networks for Mobile Vision Applications", arXiv:1704.04861, 2017,
    Table 6.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mobilenet import mobilenet_050
    >>> model = mobilenet_050()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 512, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained("mobilenet_050", alternative="mobilenet_cls")
    return MobileNet(apply_overrides(_CFG_050, overrides))


@register_model(
    task="base",
    family="mobilenet",
    model_type="mobilenet",
    model_class=MobileNet,
    default_config=_CFG_025,
    params=213_072,
)
def mobilenet_025(pretrained: bool = False, **overrides: object) -> MobileNet:
    r"""MobileNet-v1 backbone at width multiplier :math:`\alpha = 0.25`.

    Builds a :class:`MobileNet` with every channel count multiplied
    by 0.25 — approximately 0.5M parameters.  Howard et al., 2017
    report 50.6% ImageNet-1k top-1 accuracy with this configuration
    (Table 6) — the smallest MobileNet-v1 variant, targeted at
    extreme edge deployments where parameter count is the binding
    constraint.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``mobilenet_025``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``mobilenet_cls`` is the checkpointed variant.
    **overrides
        Keyword overrides forwarded into :class:`MobileNetConfig`.

    Returns
    -------
    MobileNet
        Backbone with the MobileNet-v1 (:math:`\alpha = 0.25`)
        configuration applied (or with ``overrides`` merged on top
        of it).

    Notes
    -----
    See Howard et al., "MobileNets: Efficient Convolutional Neural
    Networks for Mobile Vision Applications", arXiv:1704.04861, 2017,
    Table 6.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mobilenet import mobilenet_025
    >>> model = mobilenet_025()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 256, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained("mobilenet_025", alternative="mobilenet_cls")
    return MobileNet(apply_overrides(_CFG_025, overrides))


# ── Classifiers ───────────────────────────────────────────────────────────────


@register_model(
    task="image-classification",
    family="mobilenet",
    model_type="mobilenet",
    model_class=MobileNetForImageClassification,
    default_config=_CFG_100,
    params=4_231_976,
)
def mobilenet_cls(
    pretrained: bool | str = False, *, weights: MobileNetWeights | None = None, **overrides: object
) -> MobileNetForImageClassification:
    r"""MobileNet-v1 image classifier at width multiplier :math:`\alpha = 1.0`.

    Builds a :class:`MobileNetForImageClassification` with the
    canonical paper topology (13 depthwise+pointwise blocks) followed
    by global average pooling and a linear projection to
    ``config.num_classes`` (default 1000 for ImageNet-1k).
    Approximately 4.2M parameters and 70.6% ImageNet-1k top-1 in
    Howard et al., 2017 (Table 6).

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`MobileNetWeights.RA4_E3600_R224_IN1K`);
        a tag string → that specific checkpoint.  Mutually exclusive with
        ``weights`` (which wins if both are given).
    weights : MobileNetWeights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`MobileNetConfig`
        (typically ``num_classes`` to retarget the classifier).

    Returns
    -------
    MobileNetForImageClassification
        Classifier with the MobileNet-v1 (:math:`\alpha = 1.0`)
        configuration applied (or with ``overrides`` merged on top
        of it).

    Notes
    -----
    See Howard et al., "MobileNets: Efficient Convolutional Neural
    Networks for Mobile Vision Applications", arXiv:1704.04861, 2017,
    Table 6.  Pretrained weights are converted from timm's
    ``mobilenetv1_100.ra4_e3600_r224_in1k`` (75.4% top-1 at 224x224
    under the RA4 recipe) and hosted under ``lucid-dl/mobilenet-v1``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mobilenet import mobilenet_cls
    >>> model = mobilenet_cls(num_classes=10)
    >>> x = lucid.randn(2, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (2, 10)
    """
    entry = weights_mod.resolve_weights(MobileNetWeights, pretrained, weights)
    model = MobileNetForImageClassification(apply_overrides(_CFG_100, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="mobilenet_cls")
    return model


@register_model(
    task="image-classification",
    family="mobilenet",
    model_type="mobilenet",
    model_class=MobileNetForImageClassification,
    default_config=_CFG_075,
    params=2_585_560,
)
def mobilenet_075_cls(
    pretrained: bool = False, **overrides: object
) -> MobileNetForImageClassification:
    r"""MobileNet-v1 image classifier at width multiplier :math:`\alpha = 0.75`.

    Builds a :class:`MobileNetForImageClassification` with the
    paper-cited 0.75-width topology — approximately 2.6M parameters
    and 68.4% ImageNet-1k top-1 in Howard et al., 2017 (Table 6).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``mobilenet_075_cls``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``mobilenet_cls`` is the checkpointed variant.
    **overrides
        Keyword overrides forwarded into :class:`MobileNetConfig`.

    Returns
    -------
    MobileNetForImageClassification
        Classifier with the MobileNet-v1 (:math:`\alpha = 0.75`)
        configuration applied (or with ``overrides`` merged on top
        of it).

    Notes
    -----
    See Howard et al., "MobileNets: Efficient Convolutional Neural
    Networks for Mobile Vision Applications", arXiv:1704.04861, 2017,
    Table 6.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mobilenet import mobilenet_075_cls
    >>> model = mobilenet_075_cls()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    if pretrained:
        reject_unavailable_pretrained("mobilenet_075_cls", alternative="mobilenet_cls")
    return MobileNetForImageClassification(apply_overrides(_CFG_075, overrides))


@register_model(
    task="image-classification",
    family="mobilenet",
    model_type="mobilenet",
    model_class=MobileNetForImageClassification,
    default_config=_CFG_050,
    params=1_331_592,
)
def mobilenet_050_cls(
    pretrained: bool = False, **overrides: object
) -> MobileNetForImageClassification:
    r"""MobileNet-v1 image classifier at width multiplier :math:`\alpha = 0.5`.

    Builds a :class:`MobileNetForImageClassification` with the
    paper-cited 0.5-width topology — approximately 1.3M parameters
    and 63.7% ImageNet-1k top-1 in Howard et al., 2017 (Table 6).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``mobilenet_050_cls``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``mobilenet_cls`` is the checkpointed variant.
    **overrides
        Keyword overrides forwarded into :class:`MobileNetConfig`.

    Returns
    -------
    MobileNetForImageClassification
        Classifier with the MobileNet-v1 (:math:`\alpha = 0.5`)
        configuration applied (or with ``overrides`` merged on top
        of it).

    Notes
    -----
    See Howard et al., "MobileNets: Efficient Convolutional Neural
    Networks for Mobile Vision Applications", arXiv:1704.04861, 2017,
    Table 6.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mobilenet import mobilenet_050_cls
    >>> model = mobilenet_050_cls()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    if pretrained:
        reject_unavailable_pretrained("mobilenet_050_cls", alternative="mobilenet_cls")
    return MobileNetForImageClassification(apply_overrides(_CFG_050, overrides))


@register_model(
    task="image-classification",
    family="mobilenet",
    model_type="mobilenet",
    model_class=MobileNetForImageClassification,
    default_config=_CFG_025,
    params=470_072,
)
def mobilenet_025_cls(
    pretrained: bool = False, **overrides: object
) -> MobileNetForImageClassification:
    r"""MobileNet-v1 image classifier at width multiplier :math:`\alpha = 0.25`.

    Builds a :class:`MobileNetForImageClassification` with the
    paper-cited 0.25-width topology — approximately 0.5M parameters
    and 50.6% ImageNet-1k top-1 in Howard et al., 2017 (Table 6).
    The smallest variant in the family, targeted at extreme edge
    deployments.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``mobilenet_025_cls``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``mobilenet_cls`` is the checkpointed variant.
    **overrides
        Keyword overrides forwarded into :class:`MobileNetConfig`.

    Returns
    -------
    MobileNetForImageClassification
        Classifier with the MobileNet-v1 (:math:`\alpha = 0.25`)
        configuration applied (or with ``overrides`` merged on top
        of it).

    Notes
    -----
    See Howard et al., "MobileNets: Efficient Convolutional Neural
    Networks for Mobile Vision Applications", arXiv:1704.04861, 2017,
    Table 6.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mobilenet import mobilenet_025_cls
    >>> model = mobilenet_025_cls()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    if pretrained:
        reject_unavailable_pretrained("mobilenet_025_cls", alternative="mobilenet_cls")
    return MobileNetForImageClassification(apply_overrides(_CFG_025, overrides))
