"""Registry factories for EfficientNet B0–B7."""

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models.vision.efficientnet._config import EfficientNetConfig
from lucid.models.vision.efficientnet._model import (
    EfficientNet,
    EfficientNetForImageClassification,
)
from lucid.models.vision.efficientnet._weights import (
    EfficientNetB0Weights,
    EfficientNetB1Weights,
    EfficientNetB2Weights,
    EfficientNetB3Weights,
    EfficientNetB4Weights,
    EfficientNetB5Weights,
    EfficientNetB6Weights,
    EfficientNetB7Weights,
)
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained

# Compound scaling: (width_mult, depth_mult, dropout)
_CFGS = {
    "b0": EfficientNetConfig(width_mult=1.0, depth_mult=1.0, dropout=0.2),
    "b1": EfficientNetConfig(width_mult=1.0, depth_mult=1.1, dropout=0.2),
    "b2": EfficientNetConfig(width_mult=1.1, depth_mult=1.2, dropout=0.3),
    "b3": EfficientNetConfig(width_mult=1.2, depth_mult=1.4, dropout=0.3),
    "b4": EfficientNetConfig(width_mult=1.4, depth_mult=1.8, dropout=0.4),
    # B5-B7's shipped checkpoints come from a zoo that builds exactly these
    # three with eps=1e-3 / momentum=0.01; B0-B4 use the framework defaults.
    "b5": EfficientNetConfig(
        width_mult=1.6, depth_mult=2.2, dropout=0.4, bn_eps=1e-3, bn_momentum=0.01
    ),
    "b6": EfficientNetConfig(
        width_mult=1.8, depth_mult=2.6, dropout=0.5, bn_eps=1e-3, bn_momentum=0.01
    ),
    "b7": EfficientNetConfig(
        width_mult=2.0, depth_mult=3.1, dropout=0.5, bn_eps=1e-3, bn_momentum=0.01
    ),
}


# ── Backbones ─────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="efficientnet",
    model_type="efficientnet",
    model_class=EfficientNet,
    default_config=_CFGS["b0"],
)
def efficientnet_b0(pretrained: bool = False, **overrides: object) -> EfficientNet:
    r"""EfficientNet-B0 feature-extracting backbone (compound coefficient :math:`\phi = 0`).

    Builds an :class:`EfficientNet` with the NAS-designed baseline
    topology from Tan & Le, 2019 (Table 1): seven MBConv stages
    with block repeats ``(1, 2, 2, 3, 3, 4, 1)``, channel
    progression ``16 → 24 → 40 → 80 → 112 → 192 → 320``, and a
    1×1 head expansion to 1280 channels.  Approximately 5.3M
    parameters and 77.1% ImageNet-1k top-1 accuracy at the
    224×224 native resolution (Table 2).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``efficientnet_b0``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``efficientnet_b0_cls`` is the checkpointed variant.
    **overrides
        Keyword overrides forwarded into :class:`EfficientNetConfig`.

    Returns
    -------
    EfficientNet
        Backbone with the B0 configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See Tan & Le, "EfficientNet: Rethinking Model Scaling for
    Convolutional Neural Networks", ICML 2019 (arXiv:1905.11946),
    Table 1 and Table 2.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.efficientnet import efficientnet_b0
    >>> model = efficientnet_b0()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 1280, 1, 1)
    """
    if pretrained:
        reject_unavailable_pretrained("efficientnet_b0", alternative="efficientnet_b0_cls")
    return EfficientNet(apply_overrides(_CFGS["b0"], overrides))


@register_model(
    task="base",
    family="efficientnet",
    model_type="efficientnet",
    model_class=EfficientNet,
    default_config=_CFGS["b1"],
)
def efficientnet_b1(pretrained: bool = False, **overrides: object) -> EfficientNet:
    r"""EfficientNet-B1 feature-extracting backbone (compound coefficient :math:`\phi = 0.5`).

    Builds an :class:`EfficientNet` with depth multiplier 1.1 and
    width multiplier 1.0 — same channel counts as B0 but with
    block repeats scaled up by ``ceil(n · 1.1)``.  Approximately
    7.8M parameters and 79.1% ImageNet-1k top-1 accuracy at the
    240×240 native resolution (Tan & Le, 2019, Table 2).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``efficientnet_b1``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``efficientnet_b1_cls`` is the checkpointed variant.
    **overrides
        Keyword overrides forwarded into :class:`EfficientNetConfig`.

    Returns
    -------
    EfficientNet
        Backbone with the B1 configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See Tan & Le, "EfficientNet: Rethinking Model Scaling for
    Convolutional Neural Networks", ICML 2019 (arXiv:1905.11946),
    Table 2.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.efficientnet import efficientnet_b1
    >>> model = efficientnet_b1()
    >>> x = lucid.randn(1, 3, 240, 240)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 1280, 1, 1)
    """
    if pretrained:
        reject_unavailable_pretrained("efficientnet_b1", alternative="efficientnet_b1_cls")
    return EfficientNet(apply_overrides(_CFGS["b1"], overrides))


@register_model(
    task="base",
    family="efficientnet",
    model_type="efficientnet",
    model_class=EfficientNet,
    default_config=_CFGS["b2"],
)
def efficientnet_b2(pretrained: bool = False, **overrides: object) -> EfficientNet:
    r"""EfficientNet-B2 feature-extracting backbone (compound coefficient :math:`\phi = 1`).

    Builds an :class:`EfficientNet` with width multiplier 1.1 and
    depth multiplier 1.2.  Approximately 9.2M parameters and
    80.1% ImageNet-1k top-1 accuracy at the 260×260 native
    resolution (Tan & Le, 2019, Table 2).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``efficientnet_b2``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``efficientnet_b2_cls`` is the checkpointed variant.
    **overrides
        Keyword overrides forwarded into :class:`EfficientNetConfig`.

    Returns
    -------
    EfficientNet
        Backbone with the B2 configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See Tan & Le, "EfficientNet: Rethinking Model Scaling for
    Convolutional Neural Networks", ICML 2019 (arXiv:1905.11946),
    Table 2.  Head expansion: ``round(1280 · 1.1) = 1408``
    channels.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.efficientnet import efficientnet_b2
    >>> model = efficientnet_b2()
    >>> x = lucid.randn(1, 3, 260, 260)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 1408, 1, 1)
    """
    if pretrained:
        reject_unavailable_pretrained("efficientnet_b2", alternative="efficientnet_b2_cls")
    return EfficientNet(apply_overrides(_CFGS["b2"], overrides))


@register_model(
    task="base",
    family="efficientnet",
    model_type="efficientnet",
    model_class=EfficientNet,
    default_config=_CFGS["b3"],
)
def efficientnet_b3(pretrained: bool = False, **overrides: object) -> EfficientNet:
    r"""EfficientNet-B3 feature-extracting backbone (compound coefficient :math:`\phi = 2`).

    Builds an :class:`EfficientNet` with width multiplier 1.2 and
    depth multiplier 1.4.  Approximately 12M parameters and
    81.6% ImageNet-1k top-1 accuracy at the 300×300 native
    resolution (Tan & Le, 2019, Table 2).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``efficientnet_b3``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``efficientnet_b3_cls`` is the checkpointed variant.
    **overrides
        Keyword overrides forwarded into :class:`EfficientNetConfig`.

    Returns
    -------
    EfficientNet
        Backbone with the B3 configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See Tan & Le, "EfficientNet: Rethinking Model Scaling for
    Convolutional Neural Networks", ICML 2019 (arXiv:1905.11946),
    Table 2.  Head expansion: ``round(1280 · 1.2) = 1536``
    channels.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.efficientnet import efficientnet_b3
    >>> model = efficientnet_b3()
    >>> x = lucid.randn(1, 3, 300, 300)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 1536, 1, 1)
    """
    if pretrained:
        reject_unavailable_pretrained("efficientnet_b3", alternative="efficientnet_b3_cls")
    return EfficientNet(apply_overrides(_CFGS["b3"], overrides))


@register_model(
    task="base",
    family="efficientnet",
    model_type="efficientnet",
    model_class=EfficientNet,
    default_config=_CFGS["b4"],
)
def efficientnet_b4(pretrained: bool = False, **overrides: object) -> EfficientNet:
    r"""EfficientNet-B4 feature-extracting backbone (compound coefficient :math:`\phi = 3`).

    Builds an :class:`EfficientNet` with width multiplier 1.4 and
    depth multiplier 1.8.  Approximately 19M parameters and
    82.9% ImageNet-1k top-1 accuracy at the 380×380 native
    resolution (Tan & Le, 2019, Table 2).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``efficientnet_b4``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``efficientnet_b4_cls`` is the checkpointed variant.
    **overrides
        Keyword overrides forwarded into :class:`EfficientNetConfig`.

    Returns
    -------
    EfficientNet
        Backbone with the B4 configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See Tan & Le, "EfficientNet: Rethinking Model Scaling for
    Convolutional Neural Networks", ICML 2019 (arXiv:1905.11946),
    Table 2.  Head expansion: ``round(1280 · 1.4) = 1792``
    channels.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.efficientnet import efficientnet_b4
    >>> model = efficientnet_b4()
    >>> x = lucid.randn(1, 3, 380, 380)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 1792, 1, 1)
    """
    if pretrained:
        reject_unavailable_pretrained("efficientnet_b4", alternative="efficientnet_b4_cls")
    return EfficientNet(apply_overrides(_CFGS["b4"], overrides))


@register_model(
    task="base",
    family="efficientnet",
    model_type="efficientnet",
    model_class=EfficientNet,
    default_config=_CFGS["b5"],
)
def efficientnet_b5(pretrained: bool = False, **overrides: object) -> EfficientNet:
    r"""EfficientNet-B5 feature-extracting backbone (compound coefficient :math:`\phi = 4`).

    Builds an :class:`EfficientNet` with width multiplier 1.6 and
    depth multiplier 2.2.  Approximately 30M parameters and
    83.6% ImageNet-1k top-1 accuracy at the 456×456 native
    resolution (Tan & Le, 2019, Table 2).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``efficientnet_b5``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``efficientnet_b5_cls`` is the checkpointed variant.
    **overrides
        Keyword overrides forwarded into :class:`EfficientNetConfig`.

    Returns
    -------
    EfficientNet
        Backbone with the B5 configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See Tan & Le, "EfficientNet: Rethinking Model Scaling for
    Convolutional Neural Networks", ICML 2019 (arXiv:1905.11946),
    Table 2.  Head expansion: ``round(1280 · 1.6) = 2048``
    channels.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.efficientnet import efficientnet_b5
    >>> model = efficientnet_b5()
    >>> x = lucid.randn(1, 3, 456, 456)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 2048, 1, 1)
    """
    if pretrained:
        reject_unavailable_pretrained("efficientnet_b5", alternative="efficientnet_b5_cls")
    return EfficientNet(apply_overrides(_CFGS["b5"], overrides))


@register_model(
    task="base",
    family="efficientnet",
    model_type="efficientnet",
    model_class=EfficientNet,
    default_config=_CFGS["b6"],
)
def efficientnet_b6(pretrained: bool = False, **overrides: object) -> EfficientNet:
    r"""EfficientNet-B6 feature-extracting backbone (compound coefficient :math:`\phi = 5`).

    Builds an :class:`EfficientNet` with width multiplier 1.8 and
    depth multiplier 2.6.  Approximately 43M parameters and
    84.0% ImageNet-1k top-1 accuracy at the 528×528 native
    resolution (Tan & Le, 2019, Table 2).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``efficientnet_b6``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``efficientnet_b6_cls`` is the checkpointed variant.
    **overrides
        Keyword overrides forwarded into :class:`EfficientNetConfig`.

    Returns
    -------
    EfficientNet
        Backbone with the B6 configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See Tan & Le, "EfficientNet: Rethinking Model Scaling for
    Convolutional Neural Networks", ICML 2019 (arXiv:1905.11946),
    Table 2.  Head expansion: ``round(1280 · 1.8) = 2304``
    channels.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.efficientnet import efficientnet_b6
    >>> model = efficientnet_b6()
    >>> x = lucid.randn(1, 3, 528, 528)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 2304, 1, 1)
    """
    if pretrained:
        reject_unavailable_pretrained("efficientnet_b6", alternative="efficientnet_b6_cls")
    return EfficientNet(apply_overrides(_CFGS["b6"], overrides))


@register_model(
    task="base",
    family="efficientnet",
    model_type="efficientnet",
    model_class=EfficientNet,
    default_config=_CFGS["b7"],
)
def efficientnet_b7(pretrained: bool = False, **overrides: object) -> EfficientNet:
    r"""EfficientNet-B7 feature-extracting backbone (compound coefficient :math:`\phi = 6`).

    Builds an :class:`EfficientNet` with width multiplier 2.0 and
    depth multiplier 3.1 — the largest variant in the original
    paper.  Approximately 66M parameters and 84.4% ImageNet-1k
    top-1 accuracy at the 600×600 native resolution (Tan & Le,
    2019, Table 2), state-of-the-art at the time of publication.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``efficientnet_b7``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``efficientnet_b7_cls`` is the checkpointed variant.
    **overrides
        Keyword overrides forwarded into :class:`EfficientNetConfig`.

    Returns
    -------
    EfficientNet
        Backbone with the B7 configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See Tan & Le, "EfficientNet: Rethinking Model Scaling for
    Convolutional Neural Networks", ICML 2019 (arXiv:1905.11946),
    Table 2.  Head expansion: ``round(1280 · 2.0) = 2560``
    channels.  Memory footprint at 600×600 is substantial — plan
    accordingly for training.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.efficientnet import efficientnet_b7
    >>> model = efficientnet_b7()
    >>> x = lucid.randn(1, 3, 600, 600)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 2560, 1, 1)
    """
    if pretrained:
        reject_unavailable_pretrained("efficientnet_b7", alternative="efficientnet_b7_cls")
    return EfficientNet(apply_overrides(_CFGS["b7"], overrides))


# ── Classifiers ───────────────────────────────────────────────────────────────


@register_model(
    task="image-classification",
    family="efficientnet",
    model_type="efficientnet",
    model_class=EfficientNetForImageClassification,
    default_config=_CFGS["b0"],
)
def efficientnet_b0_cls(
    pretrained: bool | str = False,
    *,
    weights: EfficientNetB0Weights | None = None,
    **overrides: object,
) -> EfficientNetForImageClassification:
    r"""EfficientNet-B0 image classifier (backbone + GAP + linear head).

    Builds an :class:`EfficientNetForImageClassification` with the
    NAS-designed B0 topology followed by global average pooling,
    dropout (:math:`p = 0.2`), and a linear projection to
    ``config.num_classes`` (default 1000 for ImageNet-1k).
    Approximately 5.3M parameters and 77.1% ImageNet-1k top-1
    accuracy at 224×224 (Tan & Le, 2019, Table 2).

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag
        (:attr:`EfficientNetB0Weights.IMAGENET1K_V1`); a tag string
        (e.g. ``"IMAGENET1K_V1"``) → that specific checkpoint.
        Mutually exclusive with ``weights`` (which wins if both given).
    weights : EfficientNetB0Weights, optional, keyword-only
        Explicit weights enum member, e.g.
        ``EfficientNetB0Weights.IMAGENET1K_V1``.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`EfficientNetConfig`
        (typically ``num_classes`` to retarget the classifier).
        Overriding ``num_classes`` away from the checkpoint's class
        count makes pretrained loading fail the strict key/shape check.

    Returns
    -------
    EfficientNetForImageClassification
        Classifier with the B0 configuration applied (or with
        ``overrides`` merged on top of it), optionally initialised from
        pretrained weights.

    Notes
    -----
    See Tan & Le, "EfficientNet: Rethinking Model Scaling for
    Convolutional Neural Networks", ICML 2019 (arXiv:1905.11946),
    Table 2.  Pretrained weights are converted from the
    reference-framework ``EfficientNet_B0_Weights`` and hosted on the
    Hugging Face Hub under ``lucid-dl/efficientnet-b0``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.efficientnet import efficientnet_b0_cls
    >>> model = efficientnet_b0_cls(num_classes=10)
    >>> x = lucid.randn(2, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (2, 10)

    Load ImageNet-pretrained weights:

    >>> model = efficientnet_b0_cls(pretrained=True)            # DEFAULT tag
    """
    entry = weights_mod.resolve_weights(EfficientNetB0Weights, pretrained, weights)
    model = EfficientNetForImageClassification(apply_overrides(_CFGS["b0"], overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="efficientnet_b0_cls")
    return model


@register_model(
    task="image-classification",
    family="efficientnet",
    model_type="efficientnet",
    model_class=EfficientNetForImageClassification,
    default_config=_CFGS["b1"],
)
def efficientnet_b1_cls(
    pretrained: bool | str = False,
    *,
    weights: EfficientNetB1Weights | None = None,
    **overrides: object,
) -> EfficientNetForImageClassification:
    r"""EfficientNet-B1 image classifier (backbone + GAP + linear head).

    Builds an :class:`EfficientNetForImageClassification` with the
    B1 compound-scaling configuration (depth × 1.1, width × 1.0)
    followed by dropout (:math:`p = 0.2`) and a linear classifier.
    Approximately 7.8M parameters and 79.1% ImageNet-1k top-1
    accuracy at 240×240 (Tan & Le, 2019, Table 2).

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag
        (:attr:`EfficientNetB1Weights.IMAGENET1K_V1`); a tag string →
        that specific checkpoint.  Mutually exclusive with ``weights``.
    weights : EfficientNetB1Weights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`EfficientNetConfig`.

    Returns
    -------
    EfficientNetForImageClassification
        Classifier with the B1 configuration applied (or with
        ``overrides`` merged on top of it), optionally initialised from
        pretrained weights.

    Notes
    -----
    See Tan & Le, "EfficientNet: Rethinking Model Scaling for
    Convolutional Neural Networks", ICML 2019 (arXiv:1905.11946),
    Table 2.  Pretrained weights are converted from the
    reference-framework ``EfficientNet_B1_Weights`` and hosted under
    ``lucid-dl/efficientnet-b1``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.efficientnet import efficientnet_b1_cls
    >>> model = efficientnet_b1_cls(pretrained=True)
    >>> x = lucid.randn(1, 3, 240, 240)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(EfficientNetB1Weights, pretrained, weights)
    model = EfficientNetForImageClassification(apply_overrides(_CFGS["b1"], overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="efficientnet_b1_cls")
    return model


@register_model(
    task="image-classification",
    family="efficientnet",
    model_type="efficientnet",
    model_class=EfficientNetForImageClassification,
    default_config=_CFGS["b2"],
)
def efficientnet_b2_cls(
    pretrained: bool | str = False,
    *,
    weights: EfficientNetB2Weights | None = None,
    **overrides: object,
) -> EfficientNetForImageClassification:
    r"""EfficientNet-B2 image classifier (backbone + GAP + linear head).

    Builds an :class:`EfficientNetForImageClassification` with the
    B2 compound-scaling configuration (width × 1.1, depth × 1.2)
    followed by dropout (:math:`p = 0.3`) and a linear classifier.
    Approximately 9.2M parameters and 80.1% ImageNet-1k top-1
    accuracy at 260×260 (Tan & Le, 2019, Table 2).

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag
        (:attr:`EfficientNetB2Weights.IMAGENET1K_V1`); a tag string →
        that specific checkpoint.  Mutually exclusive with ``weights``.
    weights : EfficientNetB2Weights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`EfficientNetConfig`.

    Returns
    -------
    EfficientNetForImageClassification
        Classifier with the B2 configuration applied (or with
        ``overrides`` merged on top of it), optionally initialised from
        pretrained weights.

    Notes
    -----
    See Tan & Le, "EfficientNet: Rethinking Model Scaling for
    Convolutional Neural Networks", ICML 2019 (arXiv:1905.11946),
    Table 2.  Pretrained weights are converted from the
    reference-framework ``EfficientNet_B2_Weights`` and hosted under
    ``lucid-dl/efficientnet-b2``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.efficientnet import efficientnet_b2_cls
    >>> model = efficientnet_b2_cls(pretrained=True)
    >>> x = lucid.randn(1, 3, 260, 260)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(EfficientNetB2Weights, pretrained, weights)
    model = EfficientNetForImageClassification(apply_overrides(_CFGS["b2"], overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="efficientnet_b2_cls")
    return model


@register_model(
    task="image-classification",
    family="efficientnet",
    model_type="efficientnet",
    model_class=EfficientNetForImageClassification,
    default_config=_CFGS["b3"],
)
def efficientnet_b3_cls(
    pretrained: bool | str = False,
    *,
    weights: EfficientNetB3Weights | None = None,
    **overrides: object,
) -> EfficientNetForImageClassification:
    r"""EfficientNet-B3 image classifier (backbone + GAP + linear head).

    Builds an :class:`EfficientNetForImageClassification` with the
    B3 compound-scaling configuration (width × 1.2, depth × 1.4)
    followed by dropout (:math:`p = 0.3`) and a linear classifier.
    Approximately 12M parameters and 81.6% ImageNet-1k top-1
    accuracy at 300×300 (Tan & Le, 2019, Table 2).

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag
        (:attr:`EfficientNetB3Weights.IMAGENET1K_V1`); a tag string →
        that specific checkpoint.  Mutually exclusive with ``weights``.
    weights : EfficientNetB3Weights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`EfficientNetConfig`.

    Returns
    -------
    EfficientNetForImageClassification
        Classifier with the B3 configuration applied (or with
        ``overrides`` merged on top of it), optionally initialised from
        pretrained weights.

    Notes
    -----
    See Tan & Le, "EfficientNet: Rethinking Model Scaling for
    Convolutional Neural Networks", ICML 2019 (arXiv:1905.11946),
    Table 2.  Pretrained weights are converted from the
    reference-framework ``EfficientNet_B3_Weights`` and hosted under
    ``lucid-dl/efficientnet-b3``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.efficientnet import efficientnet_b3_cls
    >>> model = efficientnet_b3_cls(pretrained=True)
    >>> x = lucid.randn(1, 3, 300, 300)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(EfficientNetB3Weights, pretrained, weights)
    model = EfficientNetForImageClassification(apply_overrides(_CFGS["b3"], overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="efficientnet_b3_cls")
    return model


@register_model(
    task="image-classification",
    family="efficientnet",
    model_type="efficientnet",
    model_class=EfficientNetForImageClassification,
    default_config=_CFGS["b4"],
)
def efficientnet_b4_cls(
    pretrained: bool | str = False,
    *,
    weights: EfficientNetB4Weights | None = None,
    **overrides: object,
) -> EfficientNetForImageClassification:
    r"""EfficientNet-B4 image classifier (backbone + GAP + linear head).

    Builds an :class:`EfficientNetForImageClassification` with the
    B4 compound-scaling configuration (width × 1.4, depth × 1.8)
    followed by dropout (:math:`p = 0.4`) and a linear classifier.
    Approximately 19M parameters and 82.9% ImageNet-1k top-1
    accuracy at 380×380 (Tan & Le, 2019, Table 2).

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag
        (:attr:`EfficientNetB4Weights.IMAGENET1K_V1`); a tag string →
        that specific checkpoint.  Mutually exclusive with ``weights``.
    weights : EfficientNetB4Weights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`EfficientNetConfig`.

    Returns
    -------
    EfficientNetForImageClassification
        Classifier with the B4 configuration applied (or with
        ``overrides`` merged on top of it), optionally initialised from
        pretrained weights.

    Notes
    -----
    See Tan & Le, "EfficientNet: Rethinking Model Scaling for
    Convolutional Neural Networks", ICML 2019 (arXiv:1905.11946),
    Table 2.  Pretrained weights are converted from the
    reference-framework ``EfficientNet_B4_Weights`` and hosted under
    ``lucid-dl/efficientnet-b4``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.efficientnet import efficientnet_b4_cls
    >>> model = efficientnet_b4_cls(pretrained=True)
    >>> x = lucid.randn(1, 3, 380, 380)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(EfficientNetB4Weights, pretrained, weights)
    model = EfficientNetForImageClassification(apply_overrides(_CFGS["b4"], overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="efficientnet_b4_cls")
    return model


@register_model(
    task="image-classification",
    family="efficientnet",
    model_type="efficientnet",
    model_class=EfficientNetForImageClassification,
    default_config=_CFGS["b5"],
)
def efficientnet_b5_cls(
    pretrained: bool | str = False,
    *,
    weights: EfficientNetB5Weights | None = None,
    **overrides: object,
) -> EfficientNetForImageClassification:
    r"""EfficientNet-B5 image classifier (backbone + GAP + linear head).

    Builds an :class:`EfficientNetForImageClassification` with the
    B5 compound-scaling configuration (width × 1.6, depth × 2.2)
    followed by dropout (:math:`p = 0.4`) and a linear classifier.
    Approximately 30M parameters and 83.6% ImageNet-1k top-1
    accuracy at 456×456 (Tan & Le, 2019, Table 2).

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag
        (:attr:`EfficientNetB5Weights.IMAGENET1K_V1`); a tag string →
        that specific checkpoint.  Mutually exclusive with ``weights``.
    weights : EfficientNetB5Weights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`EfficientNetConfig`.

    Returns
    -------
    EfficientNetForImageClassification
        Classifier with the B5 configuration applied (or with
        ``overrides`` merged on top of it), optionally initialised from
        pretrained weights.

    Notes
    -----
    See Tan & Le, "EfficientNet: Rethinking Model Scaling for
    Convolutional Neural Networks", ICML 2019 (arXiv:1905.11946),
    Table 2.  Pretrained weights are converted from the
    reference-framework ``EfficientNet_B5_Weights`` and hosted under
    ``lucid-dl/efficientnet-b5``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.efficientnet import efficientnet_b5_cls
    >>> model = efficientnet_b5_cls(pretrained=True)
    >>> x = lucid.randn(1, 3, 456, 456)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(EfficientNetB5Weights, pretrained, weights)
    model = EfficientNetForImageClassification(apply_overrides(_CFGS["b5"], overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="efficientnet_b5_cls")
    return model


@register_model(
    task="image-classification",
    family="efficientnet",
    model_type="efficientnet",
    model_class=EfficientNetForImageClassification,
    default_config=_CFGS["b6"],
)
def efficientnet_b6_cls(
    pretrained: bool | str = False,
    *,
    weights: EfficientNetB6Weights | None = None,
    **overrides: object,
) -> EfficientNetForImageClassification:
    r"""EfficientNet-B6 image classifier (backbone + GAP + linear head).

    Builds an :class:`EfficientNetForImageClassification` with the
    B6 compound-scaling configuration (width × 1.8, depth × 2.6)
    followed by dropout (:math:`p = 0.5`) and a linear classifier.
    Approximately 43M parameters and 84.0% ImageNet-1k top-1
    accuracy at 528×528 (Tan & Le, 2019, Table 2).

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag
        (:attr:`EfficientNetB6Weights.IMAGENET1K_V1`); a tag string →
        that specific checkpoint.  Mutually exclusive with ``weights``.
    weights : EfficientNetB6Weights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`EfficientNetConfig`.

    Returns
    -------
    EfficientNetForImageClassification
        Classifier with the B6 configuration applied (or with
        ``overrides`` merged on top of it), optionally initialised from
        pretrained weights.

    Notes
    -----
    See Tan & Le, "EfficientNet: Rethinking Model Scaling for
    Convolutional Neural Networks", ICML 2019 (arXiv:1905.11946),
    Table 2.  Pretrained weights are converted from the
    reference-framework ``EfficientNet_B6_Weights`` and hosted under
    ``lucid-dl/efficientnet-b6``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.efficientnet import efficientnet_b6_cls
    >>> model = efficientnet_b6_cls(pretrained=True)
    >>> x = lucid.randn(1, 3, 528, 528)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(EfficientNetB6Weights, pretrained, weights)
    model = EfficientNetForImageClassification(apply_overrides(_CFGS["b6"], overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="efficientnet_b6_cls")
    return model


@register_model(
    task="image-classification",
    family="efficientnet",
    model_type="efficientnet",
    model_class=EfficientNetForImageClassification,
    default_config=_CFGS["b7"],
)
def efficientnet_b7_cls(
    pretrained: bool | str = False,
    *,
    weights: EfficientNetB7Weights | None = None,
    **overrides: object,
) -> EfficientNetForImageClassification:
    r"""EfficientNet-B7 image classifier (backbone + GAP + linear head).

    Builds an :class:`EfficientNetForImageClassification` with the
    B7 compound-scaling configuration (width × 2.0, depth × 3.1)
    followed by dropout (:math:`p = 0.5`) and a linear classifier —
    the largest variant in the original paper.  Approximately
    66M parameters and 84.4% ImageNet-1k top-1 accuracy at
    600×600 (Tan & Le, 2019, Table 2), state-of-the-art at the
    time of publication.

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag
        (:attr:`EfficientNetB7Weights.IMAGENET1K_V1`); a tag string →
        that specific checkpoint.  Mutually exclusive with ``weights``.
    weights : EfficientNetB7Weights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`EfficientNetConfig`.

    Returns
    -------
    EfficientNetForImageClassification
        Classifier with the B7 configuration applied (or with
        ``overrides`` merged on top of it), optionally initialised from
        pretrained weights.

    Notes
    -----
    See Tan & Le, "EfficientNet: Rethinking Model Scaling for
    Convolutional Neural Networks", ICML 2019 (arXiv:1905.11946),
    Table 2.  At 600×600 input resolution memory usage is
    substantial — plan inference accordingly.  Pretrained weights are
    converted from the reference-framework ``EfficientNet_B7_Weights``
    and hosted under ``lucid-dl/efficientnet-b7``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.efficientnet import efficientnet_b7_cls
    >>> model = efficientnet_b7_cls(pretrained=True)
    >>> x = lucid.randn(1, 3, 600, 600)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(EfficientNetB7Weights, pretrained, weights)
    model = EfficientNetForImageClassification(apply_overrides(_CFGS["b7"], overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="efficientnet_b7_cls")
    return model
