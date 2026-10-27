"""Registry factories for MobileNet v2."""

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models.vision.mobilenet_v2._config import MobileNetV2Config
from lucid.models.vision.mobilenet_v2._model import MobileNetV2, MobileNetV2ForImageClassification
from lucid.models.vision.mobilenet_v2._weights import MobileNetV2Weights
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained

_CFG_100 = MobileNetV2Config(width_mult=1.0)
_CFG_075 = MobileNetV2Config(width_mult=0.75)


# ── Backbones ─────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="mobilenet_v2",
    model_type="mobilenet_v2",
    model_class=MobileNetV2,
    default_config=_CFG_100,
    params=2223872,
)
def mobilenet_v2(pretrained: bool = False, **overrides: object) -> MobileNetV2:
    r"""MobileNet-v2 backbone at width multiplier :math:`\alpha = 1.0`.

    Builds a :class:`MobileNetV2` with the canonical paper topology
    from Sandler et al., 2018 (Table 2): seven inverted-residual
    stages with block repeats ``(1, 2, 3, 4, 3, 3, 1)`` and channel
    progression ``16 → 24 → 32 → 64 → 96 → 160 → 320``, followed
    by a 1×1 head expansion to 1280 channels.  Approximately
    3.5M parameters and 72.0% ImageNet-1k top-1 accuracy (Table 4).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``mobilenet_v2``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``mobilenet_v2_cls`` is the checkpointed variant.
    **overrides
        Keyword overrides forwarded into :class:`MobileNetV2Config`
        (e.g. ``in_channels=1`` for grayscale input).

    Returns
    -------
    MobileNetV2
        Backbone with the MobileNet-v2 (:math:`\alpha = 1.0`)
        configuration applied (or with ``overrides`` merged on top
        of it).

    Notes
    -----
    See Sandler et al., "MobileNetV2: Inverted Residuals and Linear
    Bottlenecks", CVPR 2018 (arXiv:1801.04381), Table 2 and Table 4.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mobilenet_v2 import mobilenet_v2
    >>> model = mobilenet_v2()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 1280, 1, 1)
    """
    if pretrained:
        reject_unavailable_pretrained("mobilenet_v2", alternative="mobilenet_v2_cls")
    cfg = apply_overrides(_CFG_100, overrides)
    return MobileNetV2(cfg)


@register_model(
    task="base",
    family="mobilenet_v2",
    model_type="mobilenet_v2",
    model_class=MobileNetV2,
    default_config=_CFG_075,
    params=1355424,
)
def mobilenet_v2_075(pretrained: bool = False, **overrides: object) -> MobileNetV2:
    r"""MobileNet-v2 backbone at width multiplier :math:`\alpha = 0.75`.

    Builds a :class:`MobileNetV2` with every channel count multiplied
    by 0.75 — 1.36M parameters headless (2.64M with the classifier
    head).

    Sandler et al., 2018 has no :math:`\alpha = 0.75` row: Table 4
    tabulates only 1.0 and 1.4, and 0.75 appears solely in Figure 5's
    caption as a point on a curve.  The often-quoted 69.8% top-1 is
    the TF model-zoo figure for ``mobilenet_v2_0.75_224``, not a paper
    result, and the paper's 300M MAdds is for :math:`\alpha = 1.0`
    only — it supports no FLOP ratio for this width.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``mobilenet_v2_075``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides
        Keyword overrides forwarded into :class:`MobileNetV2Config`.

    Returns
    -------
    MobileNetV2
        Backbone with the MobileNet-v2 (:math:`\alpha = 0.75`)
        configuration applied (or with ``overrides`` merged on top
        of it).

    Notes
    -----
    See Sandler et al., "MobileNetV2: Inverted Residuals and Linear
    Bottlenecks", CVPR 2018 (arXiv:1801.04381), Table 4.  The head
    expansion remains at 1280 channels — narrow variants do *not*
    shrink the head below 1280.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mobilenet_v2 import mobilenet_v2_075
    >>> model = mobilenet_v2_075()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 1280, 1, 1)
    """
    if pretrained:
        reject_unavailable_pretrained("mobilenet_v2_075")
    cfg = apply_overrides(_CFG_075, overrides)
    return MobileNetV2(cfg)


# ── Classifiers ───────────────────────────────────────────────────────────────


@register_model(
    task="image-classification",
    family="mobilenet_v2",
    model_type="mobilenet_v2",
    model_class=MobileNetV2ForImageClassification,
    default_config=_CFG_100,
    params=3504872,
)
def mobilenet_v2_cls(
    pretrained: bool | str = False,
    *,
    weights: MobileNetV2Weights | None = None,
    **overrides: object,
) -> MobileNetV2ForImageClassification:
    r"""MobileNet-v2 image classifier at width multiplier :math:`\alpha = 1.0`.

    Builds a :class:`MobileNetV2ForImageClassification` with the
    canonical paper topology (Sandler et al., 2018) followed by
    global average pooling and a linear projection to
    ``config.num_classes`` (default 1000 for ImageNet-1k).
    Approximately 3.5M parameters and 72.0% ImageNet-1k top-1
    accuracy.

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag
        (:attr:`MobileNetV2Weights.IMAGENET1K_V1`); a tag string (e.g.
        ``"IMAGENET1K_V1"``) → that specific checkpoint.  Mutually
        exclusive with ``weights`` (which wins if both are given).
    weights : MobileNetV2Weights, optional, keyword-only
        Explicit weights enum member, e.g.
        ``MobileNetV2Weights.IMAGENET1K_V1``.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`MobileNetV2Config`
        (typically ``num_classes`` to retarget the classifier).
        Note: overriding ``num_classes`` away from the checkpoint's
        class count makes pretrained loading fail the strict key/shape
        check — load with a matching head, then call
        :meth:`reset_classifier`.

    Returns
    -------
    MobileNetV2ForImageClassification
        Classifier with the MobileNet-v2 (:math:`\alpha = 1.0`)
        configuration applied (or with ``overrides`` merged on top
        of it), optionally initialised from pretrained weights.

    Notes
    -----
    See Sandler et al., "MobileNetV2: Inverted Residuals and Linear
    Bottlenecks", CVPR 2018 (arXiv:1801.04381), Table 4.  Pretrained
    weights are converted from reference_vision's ``MobileNet_V2_Weights``
    and hosted on the Hugging Face Hub under ``lucid-dl/mobilenet-v2``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mobilenet_v2 import mobilenet_v2_cls
    >>> model = mobilenet_v2_cls(num_classes=10)
    >>> x = lucid.randn(2, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (2, 10)

    Load ImageNet-pretrained weights:

    >>> model = mobilenet_v2_cls(pretrained=True)            # DEFAULT tag
    >>> from lucid.models.vision.mobilenet_v2 import MobileNetV2Weights
    >>> model = mobilenet_v2_cls(weights=MobileNetV2Weights.IMAGENET1K_V1)
    """
    entry = weights_mod.resolve_weights(MobileNetV2Weights, pretrained, weights)
    cfg = apply_overrides(_CFG_100, overrides)
    model = MobileNetV2ForImageClassification(cfg)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="mobilenet_v2_cls")
    return model


@register_model(
    task="image-classification",
    family="mobilenet_v2",
    model_type="mobilenet_v2",
    model_class=MobileNetV2ForImageClassification,
    default_config=_CFG_075,
    params=2636424,
)
def mobilenet_v2_075_cls(
    pretrained: bool = False, **overrides: object
) -> MobileNetV2ForImageClassification:
    r"""MobileNet-v2 image classifier at width multiplier :math:`\alpha = 0.75`.

    Builds a :class:`MobileNetV2ForImageClassification` with the
    0.75-width topology — 2.64M parameters.  The 69.8% top-1 usually
    quoted for this width comes from the TF model zoo, not from
    Sandler et al., which tabulates only 1.0 and 1.4.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``mobilenet_v2_075_cls``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides
        Keyword overrides forwarded into :class:`MobileNetV2Config`.

    Returns
    -------
    MobileNetV2ForImageClassification
        Classifier with the MobileNet-v2 (:math:`\alpha = 0.75`)
        configuration applied (or with ``overrides`` merged on top
        of it).

    Notes
    -----
    See Sandler et al., "MobileNetV2: Inverted Residuals and Linear
    Bottlenecks", CVPR 2018 (arXiv:1801.04381), Table 4.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mobilenet_v2 import mobilenet_v2_075_cls
    >>> model = mobilenet_v2_075_cls()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    if pretrained:
        reject_unavailable_pretrained("mobilenet_v2_075_cls")
    cfg = apply_overrides(_CFG_075, overrides)
    return MobileNetV2ForImageClassification(cfg)
