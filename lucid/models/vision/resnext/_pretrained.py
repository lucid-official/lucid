"""Registry factories for all ResNeXt variants."""

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models.vision.resnext._config import ResNeXtConfig
from lucid.models.vision.resnext._model import ResNeXt, ResNeXtForImageClassification
from lucid.models.vision.resnext._weights import (
    ResNeXt50_32x4dWeights,
    ResNeXt101_32x4dWeights,
    ResNeXt101_32x8dWeights,
)
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained

# ---------------------------------------------------------------------------
# Canonical configs
# ---------------------------------------------------------------------------

_CFG_50_32x4d = ResNeXtConfig(layers=(3, 4, 6, 3), cardinality=32, width_per_group=4)
_CFG_101_32x4d = ResNeXtConfig(layers=(3, 4, 23, 3), cardinality=32, width_per_group=4)
_CFG_101_32x8d = ResNeXtConfig(layers=(3, 4, 23, 3), cardinality=32, width_per_group=8)
# Table 4's 2x-complexity model and the paper's headline result (20.4% top-1
# error); the basis of the 2nd-place ILSVRC-2016 submission.
_CFG_101_64x4d = ResNeXtConfig(layers=(3, 4, 23, 3), cardinality=64, width_per_group=4)


# ---------------------------------------------------------------------------
# Backbone registrations (task="base")
# ---------------------------------------------------------------------------


@register_model(
    task="base",
    family="resnext",
    model_type="resnext",
    model_class=ResNeXt,
    default_config=_CFG_50_32x4d,
)
def resnext_50_32x4d(pretrained: bool = False, **overrides: object) -> ResNeXt:
    r"""ResNeXt-50 (32x4d) feature-extracting backbone.

    Builds a :class:`ResNeXt` with the paper-cited ResNeXt-50 (32x4d)
    topology: per-stage block counts ``(3, 4, 6, 3)`` (same as
    ResNet-50), cardinality :math:`C = 32`, width per group
    :math:`d = 4`.  23.0 M parameters without a classifier head
    (:func:`resnext_50_32x4d_cls` adds 2.05 M more for the 1000-way
    head, reaching the 25.0 M the paper quotes) — within roughly 1% of
    ResNet-50's budget while achieving ≈1pp higher ImageNet top-1
    accuracy (77.8%, i.e. the 22.2% top-1 error Xie et al., 2017
    report for this configuration).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``resnext_50_32x4d``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``resnext_50_32x4d_cls`` is the checkpointed variant.
    **overrides
        Keyword overrides forwarded into :class:`ResNeXtConfig` to
        customise ``in_channels``, ``num_classes``, ``cardinality``,
        ``width_per_group``, or ``dropout``.

    Returns
    -------
    ResNeXt
        Backbone with the ResNeXt-50 (32x4d) configuration applied
        (or with ``overrides`` merged on top of it).

    Notes
    -----
    See Xie et al., "Aggregated Residual Transformations for Deep
    Neural Networks", CVPR 2017, Table 1.  The ``32x4d`` shorthand
    encodes :math:`C \times d` — cardinality times width-per-group.
    Final-stage output is 2048 channels.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnext import resnext_50_32x4d
    >>> model = resnext_50_32x4d()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape   # (B, 2048, 7, 7)
    (1, 2048, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained("resnext_50_32x4d", alternative="resnext_50_32x4d_cls")
    cfg = apply_overrides(_CFG_50_32x4d, overrides)
    return ResNeXt(cfg)


@register_model(
    task="base",
    family="resnext",
    model_type="resnext",
    model_class=ResNeXt,
    default_config=_CFG_101_32x4d,
)
def resnext_101_32x4d(pretrained: bool = False, **overrides: object) -> ResNeXt:
    r"""ResNeXt-101 (32x4d) feature-extracting backbone.

    Builds a :class:`ResNeXt` with the paper-cited ResNeXt-101 (32x4d)
    topology: per-stage block counts ``(3, 4, 23, 3)`` (same as
    ResNet-101), cardinality :math:`C = 32`, width per group
    :math:`d = 4`.  42.1 M parameters without a classifier head
    (:func:`resnext_101_32x4d_cls` adds 2.05 M more, reaching the
    44.2 M the paper quotes); reaches 78.8% ImageNet top-1 in
    Xie et al., 2017.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``resnext_101_32x4d``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``resnext_101_32x4d_cls`` is the checkpointed variant.
    **overrides
        Keyword overrides forwarded into :class:`ResNeXtConfig`.

    Returns
    -------
    ResNeXt
        Backbone with the ResNeXt-101 (32x4d) configuration applied
        (or with ``overrides`` merged on top of it).

    Notes
    -----
    See Xie et al., CVPR 2017, Table 1.  Same depth as ResNet-101 with
    the :math:`3\times3` middle convolution split into 32 groups.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnext import resnext_101_32x4d
    >>> model = resnext_101_32x4d()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape   # (B, 2048, 7, 7)
    (1, 2048, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained("resnext_101_32x4d", alternative="resnext_101_32x4d_cls")
    cfg = apply_overrides(_CFG_101_32x4d, overrides)
    return ResNeXt(cfg)


@register_model(
    task="base",
    family="resnext",
    model_type="resnext",
    model_class=ResNeXt,
    default_config=_CFG_101_32x8d,
)
def resnext_101_32x8d(pretrained: bool = False, **overrides: object) -> ResNeXt:
    r"""ResNeXt-101 (32x8d) feature-extracting backbone.

    Builds a :class:`ResNeXt` with the higher-capacity ResNeXt-101
    (32x8d) topology: per-stage block counts ``(3, 4, 23, 3)``,
    cardinality :math:`C = 32`, width per group :math:`d = 8` (double
    the standard ResNeXt-101).  86.7 M parameters without a
    classifier head (:func:`resnext_101_32x8d_cls` adds 2.05 M more,
    reaching 88.8 M) — the widest of the canonical ResNeXt variants, widely used as the
    backbone for ImageNet-pretrained downstream models (e.g. Facebook's
    Instagram-pretrained ``ig_resnext_101_32x8d``).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``resnext_101_32x8d``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``resnext_101_32x8d_cls`` is the checkpointed variant.
    **overrides
        Keyword overrides forwarded into :class:`ResNeXtConfig`.

    Returns
    -------
    ResNeXt
        Backbone with the ResNeXt-101 (32x8d) configuration applied
        (or with ``overrides`` merged on top of it).

    Notes
    -----
    Not a variant of Xie et al., CVPR 2017 — 32x8d appears nowhere in
    the paper.  It comes from the reference model zoo, where it is kept
    for its transfer-learning performance (and is the topology
    Facebook's Instagram-pretrained ``ig_resnext_101_32x8d`` uses).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnext import resnext_101_32x8d
    >>> model = resnext_101_32x8d()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape   # (B, 2048, 7, 7)
    (1, 2048, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained("resnext_101_32x8d", alternative="resnext_101_32x8d_cls")
    cfg = apply_overrides(_CFG_101_32x8d, overrides)
    return ResNeXt(cfg)


@register_model(
    task="base",
    family="resnext",
    model_type="resnext",
    model_class=ResNeXt,
    default_config=_CFG_101_64x4d,
)
def resnext_101_64x4d(pretrained: bool = False, **overrides: object) -> ResNeXt:
    r"""ResNeXt-101 (64x4d) feature-extracting backbone — the paper's headline model.

    Builds a :class:`ResNeXt` with per-stage block counts
    ``(3, 4, 23, 3)``, cardinality :math:`C = 64` and width per group
    :math:`d = 4`.  This is Table 4's 2x-complexity configuration, the
    one §5.1's "reduces the top-1 error to 20.4%" refers to, and the
    basis of the 2nd-place ILSVRC-2016 submission.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``resnext_101_64x4d``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides
        Keyword overrides forwarded into :class:`ResNeXtConfig`.

    Returns
    -------
    ResNeXt
        Backbone with the ResNeXt-101 (64x4d) configuration applied
        (or with ``overrides`` merged on top of it).

    Notes
    -----
    Xie et al., CVPR 2017, Table 4: 20.4% top-1 / 5.3% top-5 error at
    224x224.  The paper also reports 19.1 / 4.4 for this configuration
    at 320x320, in its comparison against state-of-the-art models.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnext import resnext_101_64x4d
    >>> model = resnext_101_64x4d()
    >>> model(lucid.randn(1, 3, 224, 224)).last_hidden_state.shape
    (1, 2048, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained("resnext_101_64x4d")
    cfg = apply_overrides(_CFG_101_64x4d, overrides)
    return ResNeXt(cfg)


# ---------------------------------------------------------------------------
# Classification head registrations (task="image-classification")
# ---------------------------------------------------------------------------


@register_model(
    task="image-classification",
    family="resnext",
    model_type="resnext",
    model_class=ResNeXtForImageClassification,
    default_config=_CFG_50_32x4d,
)
def resnext_50_32x4d_cls(
    pretrained: bool | str = False,
    *,
    weights: ResNeXt50_32x4dWeights | None = None,
    **overrides: object,
) -> ResNeXtForImageClassification:
    r"""ResNeXt-50 (32x4d) image classifier (backbone + GAP + linear head).

    Builds a :class:`ResNeXtForImageClassification` with the
    paper-cited ResNeXt-50 (32x4d) topology and a
    :class:`~lucid.nn.Linear` classifier projecting 2048 →
    ``config.num_classes``.  Approximately 25.0 M parameters; the
    distributed ``IMAGENET1K_V2`` checkpoint reaches 81.198% ImageNet-1k
    top-1 with the improved training recipe.

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag
        (:attr:`ResNeXt50_32x4dWeights.IMAGENET1K_V2`); a tag string
        (e.g. ``"IMAGENET1K_V2"``) → that specific checkpoint.  Mutually
        exclusive with ``weights`` (which wins if both are given).
    weights : ResNeXt50_32x4dWeights, optional, keyword-only
        Explicit weights enum member, e.g.
        ``ResNeXt50_32x4dWeights.IMAGENET1K_V2``.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`ResNeXtConfig`.  Note:
        overriding ``num_classes`` away from the checkpoint's class
        count makes pretrained loading fail the strict key/shape check.

    Returns
    -------
    ResNeXtForImageClassification
        Classifier with the ResNeXt-50 (32x4d) configuration applied
        (or with ``overrides`` merged on top of it), optionally
        initialised from pretrained weights.

    Notes
    -----
    Pretrained weights are converted from reference_vision's
    ``ResNeXt50_32X4D_Weights.IMAGENET1K_V2`` and hosted on the Hugging
    Face Hub under ``lucid-dl/resnext-50-32x4d``.  The V2 preset uses a
    232 resize ahead of the 224 center crop.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnext import resnext_50_32x4d_cls
    >>> model = resnext_50_32x4d_cls(num_classes=100)
    >>> x = lucid.randn(2, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (2, 100)

    Load ImageNet-pretrained weights:

    >>> model = resnext_50_32x4d_cls(pretrained=True)
    >>> from lucid.models.vision.resnext import ResNeXt50_32x4dWeights
    >>> model = resnext_50_32x4d_cls(weights=ResNeXt50_32x4dWeights.IMAGENET1K_V2)
    """
    entry = weights_mod.resolve_weights(ResNeXt50_32x4dWeights, pretrained, weights)
    cfg = apply_overrides(_CFG_50_32x4d, overrides)
    model = ResNeXtForImageClassification(cfg)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="resnext_50_32x4d_cls")
    return model


@register_model(
    task="image-classification",
    family="resnext",
    model_type="resnext",
    model_class=ResNeXtForImageClassification,
    default_config=_CFG_101_32x4d,
)
def resnext_101_32x4d_cls(
    pretrained: bool | str = False,
    *,
    weights: ResNeXt101_32x4dWeights | None = None,
    **overrides: object,
) -> ResNeXtForImageClassification:
    r"""ResNeXt-101 (32x4d) image classifier (backbone + GAP + linear head).

    Builds a :class:`ResNeXtForImageClassification` with the
    paper-cited ResNeXt-101 (32x4d) topology and a
    :class:`~lucid.nn.Linear` classifier projecting 2048 →
    ``config.num_classes``.  Approximately 44.2 M parameters; the
    distributed Gluon ``GLUON_IN1K`` checkpoint reaches 80.342%
    ImageNet-1k top-1.

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag
        (:attr:`ResNeXt101_32x4dWeights.GLUON_IN1K`); a tag string
        (e.g. ``"GLUON_IN1K"``) → that specific checkpoint.  Mutually
        exclusive with ``weights`` (which wins if both are given).
    weights : ResNeXt101_32x4dWeights, optional, keyword-only
        Explicit weights enum member, e.g.
        ``ResNeXt101_32x4dWeights.GLUON_IN1K``.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`ResNeXtConfig`.  Note:
        overriding ``num_classes`` away from the checkpoint's class
        count makes pretrained loading fail the strict key/shape check.

    Returns
    -------
    ResNeXtForImageClassification
        Classifier with the ResNeXt-101 (32x4d) configuration applied
        (or with ``overrides`` merged on top of it), optionally
        initialised from pretrained weights.

    Notes
    -----
    Pretrained weights are converted from the timm Gluon checkpoint
    ``resnext101_32x4d.gluon_in1k`` and hosted on the Hugging Face Hub
    under ``lucid-dl/resnext-101-32x4d``.  The Gluon preset uses bicubic
    interpolation with a 0.875 crop_pct (256 resize → 224 crop).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnext import resnext_101_32x4d_cls
    >>> model = resnext_101_32x4d_cls()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)

    Load ImageNet-pretrained weights:

    >>> model = resnext_101_32x4d_cls(pretrained=True)
    >>> from lucid.models.vision.resnext import ResNeXt101_32x4dWeights
    >>> model = resnext_101_32x4d_cls(weights=ResNeXt101_32x4dWeights.GLUON_IN1K)
    """
    entry = weights_mod.resolve_weights(ResNeXt101_32x4dWeights, pretrained, weights)
    cfg = apply_overrides(_CFG_101_32x4d, overrides)
    model = ResNeXtForImageClassification(cfg)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="resnext_101_32x4d_cls")
    return model


@register_model(
    task="image-classification",
    family="resnext",
    model_type="resnext",
    model_class=ResNeXtForImageClassification,
    default_config=_CFG_101_32x8d,
)
def resnext_101_32x8d_cls(
    pretrained: bool | str = False,
    *,
    weights: ResNeXt101_32x8dWeights | None = None,
    **overrides: object,
) -> ResNeXtForImageClassification:
    r"""ResNeXt-101 (32x8d) image classifier (backbone + GAP + linear head).

    Builds a :class:`ResNeXtForImageClassification` with the
    high-capacity ResNeXt-101 (32x8d) topology and a
    :class:`~lucid.nn.Linear` classifier projecting 2048 →
    ``config.num_classes``.  Approximately 88.8 M parameters; the
    distributed ``IMAGENET1K_V2`` checkpoint reaches 82.834%
    ImageNet-1k top-1 with the improved training recipe.

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag
        (:attr:`ResNeXt101_32x8dWeights.IMAGENET1K_V2`); a tag string
        (e.g. ``"IMAGENET1K_V2"``) → that specific checkpoint.  Mutually
        exclusive with ``weights`` (which wins if both are given).
    weights : ResNeXt101_32x8dWeights, optional, keyword-only
        Explicit weights enum member, e.g.
        ``ResNeXt101_32x8dWeights.IMAGENET1K_V2``.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`ResNeXtConfig`.  Note:
        overriding ``num_classes`` away from the checkpoint's class
        count makes pretrained loading fail the strict key/shape check.

    Returns
    -------
    ResNeXtForImageClassification
        Classifier with the ResNeXt-101 (32x8d) configuration applied
        (or with ``overrides`` merged on top of it), optionally
        initialised from pretrained weights.

    Notes
    -----
    See Xie et al., "Aggregated Residual Transformations for Deep
    Neural Networks", CVPR 2017.  Pretrained weights are converted from
    reference_vision's ``ResNeXt101_32X8D_Weights.IMAGENET1K_V2`` and hosted
    on the Hugging Face Hub under ``lucid-dl/resnext-101-32x8d``.  The V2
    preset uses a 232 resize ahead of the 224 center crop.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnext import resnext_101_32x8d_cls
    >>> model = resnext_101_32x8d_cls()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)

    Load ImageNet-pretrained weights:

    >>> model = resnext_101_32x8d_cls(pretrained=True)
    >>> from lucid.models.vision.resnext import ResNeXt101_32x8dWeights
    >>> model = resnext_101_32x8d_cls(weights=ResNeXt101_32x8dWeights.IMAGENET1K_V2)
    """
    entry = weights_mod.resolve_weights(ResNeXt101_32x8dWeights, pretrained, weights)
    cfg = apply_overrides(_CFG_101_32x8d, overrides)
    model = ResNeXtForImageClassification(cfg)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="resnext_101_32x8d_cls")
    return model


@register_model(
    task="image-classification",
    family="resnext",
    model_type="resnext",
    model_class=ResNeXtForImageClassification,
    default_config=_CFG_101_64x4d,
)
def resnext_101_64x4d_cls(
    pretrained: bool = False, **overrides: object
) -> ResNeXtForImageClassification:
    r"""ResNeXt-101 (64x4d) image classifier — the paper's headline model.

    Builds a :class:`ResNeXtForImageClassification` with cardinality
    :math:`C = 64` and width per group :math:`d = 4`, plus a
    :class:`~lucid.nn.Linear` classifier projecting 2048 →
    ``config.num_classes``.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``resnext_101_64x4d_cls``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides
        Keyword overrides forwarded into :class:`ResNeXtConfig`.

    Returns
    -------
    ResNeXtForImageClassification
        Classifier with the ResNeXt-101 (64x4d) configuration applied
        (or with ``overrides`` merged on top of it).

    Notes
    -----
    Xie et al., CVPR 2017, Table 4: 20.4% top-1 / 5.3% top-5 error at
    224x224 — the result §5.1's headline refers to, and the basis of
    the 2nd-place ILSVRC-2016 submission.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnext import resnext_101_64x4d_cls
    >>> model = resnext_101_64x4d_cls(num_classes=10)
    >>> model(lucid.randn(1, 3, 224, 224)).logits.shape
    (1, 10)
    """
    if pretrained:
        reject_unavailable_pretrained("resnext_101_64x4d_cls")
    cfg = apply_overrides(_CFG_101_64x4d, overrides)
    return ResNeXtForImageClassification(cfg)
