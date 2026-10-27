"""Registry factories for VGG variants."""

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models.vision.vgg._config import VGGConfig
from lucid.models.vision.vgg._model import VGG, VGGForImageClassification
from lucid.models.vision.vgg._weights import (
    VGG11BNWeights,
    VGG11Weights,
    VGG13BNWeights,
    VGG13Weights,
    VGG16BNWeights,
    VGG16Weights,
    VGG19BNWeights,
    VGG19Weights,
)
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained

_CFG_11 = VGGConfig(arch=(1, 1, 2, 2, 2))
_CFG_13 = VGGConfig(arch=(2, 2, 2, 2, 2))
_CFG_16 = VGGConfig(arch=(2, 2, 3, 3, 3))
_CFG_19 = VGGConfig(arch=(2, 2, 4, 4, 4))
_CFG_11_BN = VGGConfig(arch=(1, 1, 2, 2, 2), batch_norm=True)
_CFG_13_BN = VGGConfig(arch=(2, 2, 2, 2, 2), batch_norm=True)
_CFG_16_BN = VGGConfig(arch=(2, 2, 3, 3, 3), batch_norm=True)
_CFG_19_BN = VGGConfig(arch=(2, 2, 4, 4, 4), batch_norm=True)


# ── Backbones ─────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="vgg",
    model_type="vgg",
    model_class=VGG,
    default_config=_CFG_11,
    params=9220480,
)
def vgg_11(pretrained: bool = False, **overrides: object) -> VGG:
    r"""VGG-11 feature-extracting backbone (config A, no BatchNorm).

    Builds a :class:`VGG` with the paper-cited VGG-11 topology:
    per-block convolution counts ``(1, 1, 2, 2, 2)`` across the five
    blocks of widths ``(64, 128, 256, 512, 512)``.  Approximately
    132.9 M parameters in the full classifier variant (≈9.2 M in the
    convolutional trunk alone).  Reaches a top-5 ImageNet validation
    error of 10.4% in Simonyan & Zisserman, ICLR 2015 (Table 3,
    configuration A, single test scale S = Q = 256).  9.4% belongs to
    configuration C, not A.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``vgg_11``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``vgg_11_cls`` is the checkpointed variant.
    **overrides
        Keyword overrides forwarded into :class:`VGGConfig` to
        customise individual fields (``in_channels``, ``num_classes``,
        ``dropout``).

    Returns
    -------
    VGG
        Backbone with the VGG-11 configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See Simonyan & Zisserman, "Very Deep Convolutional Networks for
    Large-Scale Image Recognition", ICLR 2015, Table 1 (configuration A).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.vgg import vgg_11
    >>> model = vgg_11()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape   # (B, 512, 7, 7)
    (1, 512, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained("vgg_11", alternative="vgg_11_cls")
    return VGG(apply_overrides(_CFG_11, overrides))


@register_model(
    task="base",
    family="vgg",
    model_type="vgg",
    model_class=VGG,
    default_config=_CFG_13,
    params=9404992,
)
def vgg_13(pretrained: bool = False, **overrides: object) -> VGG:
    r"""VGG-13 feature-extracting backbone (config B, no BatchNorm).

    Builds a :class:`VGG` with the paper-cited VGG-13 topology:
    per-block convolution counts ``(2, 2, 2, 2, 2)``.  Approximately
    133.0 M parameters in the full classifier variant.  Reaches a top-5
    ImageNet validation error of 9.9% (Simonyan & Zisserman, ICLR 2015,
    Table 3, configuration B, single test scale S = Q = 256).  8.8% is
    configuration D.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``vgg_13``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``vgg_13_cls`` is the checkpointed variant.
    **overrides
        Keyword overrides forwarded into :class:`VGGConfig`.

    Returns
    -------
    VGG
        Backbone with the VGG-13 configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See Simonyan & Zisserman, ICLR 2015, Table 1 (configuration B).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.vgg import vgg_13
    >>> model = vgg_13()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape   # (B, 512, 7, 7)
    (1, 512, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained("vgg_13", alternative="vgg_13_cls")
    return VGG(apply_overrides(_CFG_13, overrides))


@register_model(
    task="base",
    family="vgg",
    model_type="vgg",
    model_class=VGG,
    default_config=_CFG_16,
    params=14714688,
)
def vgg_16(pretrained: bool = False, **overrides: object) -> VGG:
    r"""VGG-16 feature-extracting backbone (config D, no BatchNorm).

    Builds a :class:`VGG` with the paper-cited VGG-16 topology:
    per-block convolution counts ``(2, 2, 3, 3, 3)`` — adding a third
    :math:`3\times3` convolution to each of the deeper blocks 3/4/5.
    Approximately 138.4 M parameters in the full classifier variant
    (≈14.7 M in the convolutional trunk alone).  Reaches a top-5
    ImageNet validation error of 8.8% (Simonyan & Zisserman, ICLR 2015,
    Table 3, configuration D, single test scale S = Q = 256) — VGG-16
    was the most widely deployed variant of the family.  The often-quoted
    7.3% is the ILSVRC-2014 submission *test* error of an ensemble of
    seven nets (Table 6), not a single-model number.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``vgg_16``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``vgg_16_cls`` is the checkpointed variant.
    **overrides
        Keyword overrides forwarded into :class:`VGGConfig`.

    Returns
    -------
    VGG
        Backbone with the VGG-16 configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See Simonyan & Zisserman, ICLR 2015, Table 1 (configuration D).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.vgg import vgg_16
    >>> model = vgg_16()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape   # (B, 512, 7, 7)
    (1, 512, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained("vgg_16", alternative="vgg_16_cls")
    return VGG(apply_overrides(_CFG_16, overrides))


@register_model(
    task="base",
    family="vgg",
    model_type="vgg",
    model_class=VGG,
    default_config=_CFG_19,
    params=20024384,
)
def vgg_19(pretrained: bool = False, **overrides: object) -> VGG:
    r"""VGG-19 feature-extracting backbone (config E, no BatchNorm).

    Builds a :class:`VGG` with the paper-cited VGG-19 topology:
    per-block convolution counts ``(2, 2, 4, 4, 4)`` — four
    :math:`3\times3` convolutions in each of the deeper blocks.
    Approximately 143.7 M parameters in the full classifier variant.
    Reaches a top-5 ImageNet validation error of 9.0% (Simonyan &
    Zisserman, ICLR 2015, Table 3, configuration E, single test scale
    S = Q = 256).  7.5% is Table 4's multi-scale figure, which D and E
    share.  E is not worse than D: at multiple scales they tie, and the
    paper's best single model is E at 7.1% (Table 5, multi-crop and
    dense evaluation).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``vgg_19``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``vgg_19_cls`` is the checkpointed variant.
    **overrides
        Keyword overrides forwarded into :class:`VGGConfig`.

    Returns
    -------
    VGG
        Backbone with the VGG-19 configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See Simonyan & Zisserman, ICLR 2015, Table 1 (configuration E).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.vgg import vgg_19
    >>> model = vgg_19()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape   # (B, 512, 7, 7)
    (1, 512, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained("vgg_19", alternative="vgg_19_cls")
    return VGG(apply_overrides(_CFG_19, overrides))


@register_model(
    task="base",
    family="vgg",
    model_type="vgg",
    model_class=VGG,
    default_config=_CFG_11_BN,
    params=9225984,
)
def vgg_11_bn(pretrained: bool = False, **overrides: object) -> VGG:
    r"""VGG-11 with BatchNorm — feature-extracting backbone.

    Builds a :class:`VGG` with the VGG-11 topology
    ``arch=(1, 1, 2, 2, 2)`` and a :class:`~lucid.nn.BatchNorm2d` layer
    between each Conv and its ReLU.  BatchNorm was added in the
    timm / reference-framework reimplementations of VGG (not in the
    original paper) and substantially improves convergence speed and
    final accuracy; the topology is otherwise identical to
    :func:`vgg_11`.  9.22 M parameters without the FC head (:func:`vgg_11_cls` adds it for 132.9 M).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``vgg_11_bn``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``vgg_11_bn_cls`` is the checkpointed variant.
    **overrides
        Keyword overrides forwarded into :class:`VGGConfig`.

    Returns
    -------
    VGG
        Backbone with the VGG-11-BN configuration applied (or with
        ``overrides`` merged on top of it).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.vgg import vgg_11_bn
    >>> model = vgg_11_bn()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 512, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained("vgg_11_bn", alternative="vgg_11_bn_cls")
    return VGG(apply_overrides(_CFG_11_BN, overrides))


@register_model(
    task="base",
    family="vgg",
    model_type="vgg",
    model_class=VGG,
    default_config=_CFG_13_BN,
    params=9410880,
)
def vgg_13_bn(pretrained: bool = False, **overrides: object) -> VGG:
    r"""VGG-13 with BatchNorm — feature-extracting backbone.

    Builds a :class:`VGG` with the VGG-13 topology
    ``arch=(2, 2, 2, 2, 2)`` and :class:`~lucid.nn.BatchNorm2d` between
    each Conv and its ReLU.  9.41 M parameters without the FC head (:func:`vgg_13_bn_cls` adds it for 133.1 M).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``vgg_13_bn``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``vgg_13_bn_cls`` is the checkpointed variant.
    **overrides
        Keyword overrides forwarded into :class:`VGGConfig`.

    Returns
    -------
    VGG
        Backbone with the VGG-13-BN configuration applied (or with
        ``overrides`` merged on top of it).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.vgg import vgg_13_bn
    >>> model = vgg_13_bn()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 512, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained("vgg_13_bn", alternative="vgg_13_bn_cls")
    return VGG(apply_overrides(_CFG_13_BN, overrides))


@register_model(
    task="base",
    family="vgg",
    model_type="vgg",
    model_class=VGG,
    default_config=_CFG_16_BN,
    params=14723136,
)
def vgg_16_bn(pretrained: bool = False, **overrides: object) -> VGG:
    r"""VGG-16 with BatchNorm — feature-extracting backbone.

    Builds a :class:`VGG` with the VGG-16 topology
    ``arch=(2, 2, 3, 3, 3)`` and :class:`~lucid.nn.BatchNorm2d` between
    each Conv and its ReLU.  14.72 M parameters without the FC head (:func:`vgg_16_bn_cls` adds it for 138.4 M).  Often
    preferred over plain :func:`vgg_16` for downstream fine-tuning
    because BatchNorm stabilises gradient statistics in the early conv
    stack.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``vgg_16_bn``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``vgg_16_bn_cls`` is the checkpointed variant.
    **overrides
        Keyword overrides forwarded into :class:`VGGConfig`.

    Returns
    -------
    VGG
        Backbone with the VGG-16-BN configuration applied (or with
        ``overrides`` merged on top of it).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.vgg import vgg_16_bn
    >>> model = vgg_16_bn()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 512, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained("vgg_16_bn", alternative="vgg_16_bn_cls")
    return VGG(apply_overrides(_CFG_16_BN, overrides))


@register_model(
    task="base",
    family="vgg",
    model_type="vgg",
    model_class=VGG,
    default_config=_CFG_19_BN,
    params=20035392,
)
def vgg_19_bn(pretrained: bool = False, **overrides: object) -> VGG:
    r"""VGG-19 with BatchNorm — feature-extracting backbone.

    Builds a :class:`VGG` with the VGG-19 topology
    ``arch=(2, 2, 4, 4, 4)`` and :class:`~lucid.nn.BatchNorm2d` between
    each Conv and its ReLU.  20.04 M parameters without the FC head (:func:`vgg_19_bn_cls` adds it for 143.7 M).  The
    deepest VGG variant; BatchNorm is particularly helpful at this
    depth.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``vgg_19_bn``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``vgg_19_bn_cls`` is the checkpointed variant.
    **overrides
        Keyword overrides forwarded into :class:`VGGConfig`.

    Returns
    -------
    VGG
        Backbone with the VGG-19-BN configuration applied (or with
        ``overrides`` merged on top of it).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.vgg import vgg_19_bn
    >>> model = vgg_19_bn()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 512, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained("vgg_19_bn", alternative="vgg_19_bn_cls")
    return VGG(apply_overrides(_CFG_19_BN, overrides))


# ── Classifiers ───────────────────────────────────────────────────────────────


@register_model(
    task="image-classification",
    family="vgg",
    model_type="vgg",
    model_class=VGGForImageClassification,
    default_config=_CFG_11,
    params=132863336,
)
def vgg_11_cls(
    pretrained: bool | str = False, *, weights: VGG11Weights | None = None, **overrides: object
) -> VGGForImageClassification:
    r"""VGG-11 image classifier (config A, no BatchNorm).

    Builds a :class:`VGGForImageClassification` with the paper-cited
    VGG-11 topology (``arch=(1, 1, 2, 2, 2)``) followed by the two
    4096-dim fully-connected layers and a final linear projection to
    ``config.num_classes``.  Approximately 132.9 M parameters total
    (≈124 M of which sit in the two large FC layers).  Reaches a top-1
    ImageNet validation accuracy of 69.02% (reference_vision eval).

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`VGG11Weights.IMAGENET1K_V1`); a
        tag string → that specific checkpoint.  Mutually exclusive with
        ``weights`` (which wins if both are given).
    weights : VGG11Weights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`VGGConfig` — use
        ``num_classes=N`` to retarget the head.

    Returns
    -------
    VGGForImageClassification
        Classifier with the VGG-11 configuration applied (or with
        ``overrides`` merged on top of it), optionally initialised from
        pretrained weights.

    Notes
    -----
    Pretrained weights are converted from reference_vision's ``VGG11_Weights``
    and hosted on the Hugging Face Hub under ``lucid-dl/vgg-11``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.vgg import vgg_11_cls
    >>> model = vgg_11_cls(pretrained=True)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(VGG11Weights, pretrained, weights)
    model = VGGForImageClassification(apply_overrides(_CFG_11, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="vgg_11_cls")
    return model


@register_model(
    task="image-classification",
    family="vgg",
    model_type="vgg",
    model_class=VGGForImageClassification,
    default_config=_CFG_13,
    params=133047848,
)
def vgg_13_cls(
    pretrained: bool | str = False, *, weights: VGG13Weights | None = None, **overrides: object
) -> VGGForImageClassification:
    r"""VGG-13 image classifier (config B, no BatchNorm).

    Builds a :class:`VGGForImageClassification` with the paper-cited
    VGG-13 topology (``arch=(2, 2, 2, 2, 2)``) followed by the standard
    4096 → 4096 → ``num_classes`` FC head.  Approximately 133.0 M
    parameters total.  Reaches a top-1 ImageNet validation accuracy of
    69.93% (reference_vision eval).

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`VGG13Weights.IMAGENET1K_V1`); a
        tag string → that specific checkpoint.  Mutually exclusive with
        ``weights`` (which wins if both are given).
    weights : VGG13Weights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`VGGConfig`.

    Returns
    -------
    VGGForImageClassification
        Classifier with the VGG-13 configuration applied (or with
        ``overrides`` merged on top of it), optionally initialised from
        pretrained weights.

    Notes
    -----
    Pretrained weights are converted from reference_vision's ``VGG13_Weights``
    and hosted on the Hugging Face Hub under ``lucid-dl/vgg-13``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.vgg import vgg_13_cls
    >>> model = vgg_13_cls(pretrained=True)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(VGG13Weights, pretrained, weights)
    model = VGGForImageClassification(apply_overrides(_CFG_13, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="vgg_13_cls")
    return model


@register_model(
    task="image-classification",
    family="vgg",
    model_type="vgg",
    model_class=VGGForImageClassification,
    default_config=_CFG_16,
    params=138357544,
)
def vgg_16_cls(
    pretrained: bool | str = False, *, weights: VGG16Weights | None = None, **overrides: object
) -> VGGForImageClassification:
    r"""VGG-16 image classifier (config D, no BatchNorm).

    Builds a :class:`VGGForImageClassification` with the paper-cited
    VGG-16 topology (``arch=(2, 2, 3, 3, 3)``) followed by the standard
    4096 → 4096 → ``num_classes`` FC head.  Approximately 138.4 M
    parameters total and a top-1 ImageNet validation accuracy of 71.59%
    (reference_vision eval) — the most widely used VGG variant.

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`VGG16Weights.IMAGENET1K_V1`); a
        tag string → that specific checkpoint.  Mutually exclusive with
        ``weights`` (which wins if both are given).
    weights : VGG16Weights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`VGGConfig`.

    Returns
    -------
    VGGForImageClassification
        Classifier with the VGG-16 configuration applied (or with
        ``overrides`` merged on top of it), optionally initialised from
        pretrained weights.

    Notes
    -----
    Pretrained weights are converted from reference_vision's ``VGG16_Weights``
    and hosted on the Hugging Face Hub under ``lucid-dl/vgg-16``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.vgg import vgg_16_cls
    >>> model = vgg_16_cls(pretrained=True)
    >>> x = lucid.randn(2, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (2, 1000)
    """
    entry = weights_mod.resolve_weights(VGG16Weights, pretrained, weights)
    model = VGGForImageClassification(apply_overrides(_CFG_16, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="vgg_16_cls")
    return model


@register_model(
    task="image-classification",
    family="vgg",
    model_type="vgg",
    model_class=VGGForImageClassification,
    default_config=_CFG_19,
    params=143667240,
)
def vgg_19_cls(
    pretrained: bool | str = False, *, weights: VGG19Weights | None = None, **overrides: object
) -> VGGForImageClassification:
    r"""VGG-19 image classifier (config E, no BatchNorm).

    Builds a :class:`VGGForImageClassification` with the paper-cited
    VGG-19 topology (``arch=(2, 2, 4, 4, 4)``) followed by the standard
    4096 → 4096 → ``num_classes`` FC head.  Approximately 143.7 M
    parameters total.  Reaches a top-1 ImageNet validation accuracy of
    72.38% (reference_vision eval).

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`VGG19Weights.IMAGENET1K_V1`); a
        tag string → that specific checkpoint.  Mutually exclusive with
        ``weights`` (which wins if both are given).
    weights : VGG19Weights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`VGGConfig`.

    Returns
    -------
    VGGForImageClassification
        Classifier with the VGG-19 configuration applied (or with
        ``overrides`` merged on top of it), optionally initialised from
        pretrained weights.

    Notes
    -----
    Pretrained weights are converted from reference_vision's ``VGG19_Weights``
    and hosted on the Hugging Face Hub under ``lucid-dl/vgg-19``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.vgg import vgg_19_cls
    >>> model = vgg_19_cls(pretrained=True)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(VGG19Weights, pretrained, weights)
    model = VGGForImageClassification(apply_overrides(_CFG_19, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="vgg_19_cls")
    return model


@register_model(
    task="image-classification",
    family="vgg",
    model_type="vgg",
    model_class=VGGForImageClassification,
    default_config=_CFG_11_BN,
    params=132868840,
)
def vgg_11_bn_cls(
    pretrained: bool | str = False, *, weights: VGG11BNWeights | None = None, **overrides: object
) -> VGGForImageClassification:
    r"""VGG-11 with BatchNorm — image classifier.

    Same topology as :func:`vgg_11_cls` but with
    :class:`~lucid.nn.BatchNorm2d` between each Conv and its ReLU.
    BatchNorm was added in the timm / reference-framework
    reimplementations of VGG (not in the original paper) and
    substantially improves convergence speed and final accuracy.
    Approximately 132.9 M parameters; top-1 ImageNet validation
    accuracy of 70.37% (reference_vision eval).

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`VGG11BNWeights.IMAGENET1K_V1`); a
        tag string → that specific checkpoint.  Mutually exclusive with
        ``weights`` (which wins if both are given).
    weights : VGG11BNWeights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`VGGConfig`.

    Returns
    -------
    VGGForImageClassification
        Classifier with the VGG-11-BN configuration applied (or with
        ``overrides`` merged on top of it), optionally initialised from
        pretrained weights.

    Notes
    -----
    Pretrained weights are converted from reference_vision's
    ``VGG11_BN_Weights`` and hosted on the Hugging Face Hub under
    ``lucid-dl/vgg-11-bn``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.vgg import vgg_11_bn_cls
    >>> model = vgg_11_bn_cls(pretrained=True)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(VGG11BNWeights, pretrained, weights)
    model = VGGForImageClassification(apply_overrides(_CFG_11_BN, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="vgg_11_bn_cls")
    return model


@register_model(
    task="image-classification",
    family="vgg",
    model_type="vgg",
    model_class=VGGForImageClassification,
    default_config=_CFG_13_BN,
    params=133053736,
)
def vgg_13_bn_cls(
    pretrained: bool | str = False, *, weights: VGG13BNWeights | None = None, **overrides: object
) -> VGGForImageClassification:
    r"""VGG-13 with BatchNorm — image classifier.

    Same topology as :func:`vgg_13_cls` but with
    :class:`~lucid.nn.BatchNorm2d` between each Conv and its ReLU.
    Approximately 133.0 M parameters; top-1 ImageNet validation
    accuracy of 71.59% (reference_vision eval).

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`VGG13BNWeights.IMAGENET1K_V1`); a
        tag string → that specific checkpoint.  Mutually exclusive with
        ``weights`` (which wins if both are given).
    weights : VGG13BNWeights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`VGGConfig`.

    Returns
    -------
    VGGForImageClassification
        Classifier with the VGG-13-BN configuration applied (or with
        ``overrides`` merged on top of it), optionally initialised from
        pretrained weights.

    Notes
    -----
    Pretrained weights are converted from reference_vision's
    ``VGG13_BN_Weights`` and hosted on the Hugging Face Hub under
    ``lucid-dl/vgg-13-bn``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.vgg import vgg_13_bn_cls
    >>> model = vgg_13_bn_cls(pretrained=True)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(VGG13BNWeights, pretrained, weights)
    model = VGGForImageClassification(apply_overrides(_CFG_13_BN, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="vgg_13_bn_cls")
    return model


@register_model(
    task="image-classification",
    family="vgg",
    model_type="vgg",
    model_class=VGGForImageClassification,
    default_config=_CFG_16_BN,
    params=138365992,
)
def vgg_16_bn_cls(
    pretrained: bool | str = False, *, weights: VGG16BNWeights | None = None, **overrides: object
) -> VGGForImageClassification:
    r"""VGG-16 with BatchNorm — image classifier.

    Same topology as :func:`vgg_16_cls` but with
    :class:`~lucid.nn.BatchNorm2d` between each Conv and its ReLU.
    Approximately 138.4 M parameters; top-1 ImageNet validation
    accuracy of 73.36% (reference_vision eval).  Often preferred over plain
    :func:`vgg_16_cls` for downstream fine-tuning.

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`VGG16BNWeights.IMAGENET1K_V1`); a
        tag string → that specific checkpoint.  Mutually exclusive with
        ``weights`` (which wins if both are given).
    weights : VGG16BNWeights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`VGGConfig`.

    Returns
    -------
    VGGForImageClassification
        Classifier with the VGG-16-BN configuration applied (or with
        ``overrides`` merged on top of it), optionally initialised from
        pretrained weights.

    Notes
    -----
    Pretrained weights are converted from reference_vision's
    ``VGG16_BN_Weights`` and hosted on the Hugging Face Hub under
    ``lucid-dl/vgg-16-bn``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.vgg import vgg_16_bn_cls
    >>> model = vgg_16_bn_cls(pretrained=True)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(VGG16BNWeights, pretrained, weights)
    model = VGGForImageClassification(apply_overrides(_CFG_16_BN, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="vgg_16_bn_cls")
    return model


@register_model(
    task="image-classification",
    family="vgg",
    model_type="vgg",
    model_class=VGGForImageClassification,
    default_config=_CFG_19_BN,
    params=143678248,
)
def vgg_19_bn_cls(
    pretrained: bool | str = False, *, weights: VGG19BNWeights | None = None, **overrides: object
) -> VGGForImageClassification:
    r"""VGG-19 with BatchNorm — image classifier.

    Same topology as :func:`vgg_19_cls` but with
    :class:`~lucid.nn.BatchNorm2d` between each Conv and its ReLU.
    Approximately 143.7 M parameters; top-1 ImageNet validation
    accuracy of 74.22% (reference_vision eval).  The deepest VGG variant —
    BatchNorm is especially valuable at this depth.

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`VGG19BNWeights.IMAGENET1K_V1`); a
        tag string → that specific checkpoint.  Mutually exclusive with
        ``weights`` (which wins if both are given).
    weights : VGG19BNWeights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`VGGConfig`.

    Returns
    -------
    VGGForImageClassification
        Classifier with the VGG-19-BN configuration applied (or with
        ``overrides`` merged on top of it), optionally initialised from
        pretrained weights.

    Notes
    -----
    Pretrained weights are converted from reference_vision's
    ``VGG19_BN_Weights`` and hosted on the Hugging Face Hub under
    ``lucid-dl/vgg-19-bn``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.vgg import vgg_19_bn_cls
    >>> model = vgg_19_bn_cls(pretrained=True)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(VGG19BNWeights, pretrained, weights)
    model = VGGForImageClassification(apply_overrides(_CFG_19_BN, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="vgg_19_bn_cls")
    return model
