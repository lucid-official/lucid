"""Registry factories for all ResNet variants."""

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models.vision.resnet._config import ResNetConfig
from lucid.models.vision.resnet._model import ResNet, ResNetForImageClassification
from lucid.models.vision.resnet._weights import (
    ResNet18Weights,
    ResNet34Weights,
    ResNet50Weights,
    ResNet101Weights,
    ResNet152Weights,
    WideResNet50Weights,
    WideResNet101Weights,
)
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained

# ---------------------------------------------------------------------------
# Canonical configs
# ---------------------------------------------------------------------------

_CFG_18 = ResNetConfig(block_type="basic", layers=(2, 2, 2, 2))
_CFG_34 = ResNetConfig(block_type="basic", layers=(3, 4, 6, 3))
_CFG_50 = ResNetConfig(block_type="bottleneck", layers=(3, 4, 6, 3))
_CFG_101 = ResNetConfig(block_type="bottleneck", layers=(3, 4, 23, 3))
_CFG_152 = ResNetConfig(block_type="bottleneck", layers=(3, 8, 36, 3))

# Wide ResNet-50-2 / 101-2: 2× width multiplier on bottleneck inner channels.
# Stage output channels remain the same as standard ResNet (256/512/1024/2048).
_CFG_WIDE50 = ResNetConfig(block_type="bottleneck", layers=(3, 4, 6, 3), bottleneck_width_mult=2)
_CFG_WIDE101 = ResNetConfig(block_type="bottleneck", layers=(3, 4, 23, 3), bottleneck_width_mult=2)

# ResNet-200 / ResNet-269: deeper bottleneck variants
_CFG_200 = ResNetConfig(block_type="bottleneck", layers=(3, 24, 36, 3))
_CFG_269 = ResNetConfig(block_type="bottleneck", layers=(3, 30, 48, 8))


# ---------------------------------------------------------------------------
# Backbone registrations (task="base")
# ---------------------------------------------------------------------------


@register_model(
    task="base",
    family="resnet",
    model_type="resnet",
    model_class=ResNet,
    default_config=_CFG_18,
    params=11176512,
)
def resnet_18(pretrained: bool = False, **overrides: object) -> ResNet:
    r"""ResNet-18 feature-extracting backbone (no classification head).

    Builds a :class:`ResNet` with the paper-cited ResNet-18 topology:
    :class:`_BasicBlock` blocks stacked ``[2, 2, 2, 2]`` over four
    stages, yielding approximately 11.7M parameters.  The original
    paper (He et al., 2015) reports a 69.76% ImageNet-1k top-1
    validation accuracy with this configuration (Table 4).  Used as a
    lightweight backbone where parameter / FLOP budget is tight.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``resnet_18``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``resnet_18_cls`` is the checkpointed variant.
    **overrides
        Keyword overrides forwarded into :class:`ResNetConfig` to
        customise individual fields (e.g. ``in_channels=1`` for
        grayscale input, ``zero_init_residual=True`` for large-batch
        training).

    Returns
    -------
    ResNet
        Backbone with the ResNet-18 configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See He et al., "Deep Residual Learning for Image Recognition",
    CVPR 2016 (arXiv:1512.03385), Table 1.  The key insight is the
    identity shortcut

    .. math::

        y_{l+1} = \sigma\!\left(F(y_l, W_l) + y_l\right),

    which lets gradients propagate through a clean identity branch
    even in very deep networks.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnet import resnet_18
    >>> model = resnet_18()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape   # (B, 512, H/32, W/32)
    (1, 512, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained("resnet_18", alternative="resnet_18_cls")
    cfg = apply_overrides(_CFG_18, overrides)
    return ResNet(cfg)


@register_model(
    task="base",
    family="resnet",
    model_type="resnet",
    model_class=ResNet,
    default_config=_CFG_34,
    params=21284672,
)
def resnet_34(pretrained: bool = False, **overrides: object) -> ResNet:
    r"""ResNet-34 feature-extracting backbone (no classification head).

    Builds a :class:`ResNet` with the paper-cited ResNet-34 topology:
    :class:`_BasicBlock` blocks stacked ``[3, 4, 6, 3]`` over four
    stages, yielding approximately 21.8M parameters.  Reaches 73.31%
    ImageNet-1k top-1 in He et al., 2015 (Table 4).  A good middle
    ground when ResNet-18 is too small but ResNet-50's bottleneck
    overhead is unwanted.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``resnet_34``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``resnet_34_cls`` is the checkpointed variant.
    **overrides
        Keyword overrides forwarded into :class:`ResNetConfig` to
        customise individual fields without writing a config by hand.

    Returns
    -------
    ResNet
        Backbone with the ResNet-34 configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See He et al., "Deep Residual Learning for Image Recognition",
    CVPR 2016 (arXiv:1512.03385), Table 1.  Like ResNet-18 this
    variant uses two-layer residual blocks, so the final-stage output
    is 512 channels (not 2048 as in the bottleneck variants).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnet import resnet_34
    >>> model = resnet_34()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape   # (B, 512, H/32, W/32)
    (1, 512, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained("resnet_34", alternative="resnet_34_cls")
    cfg = apply_overrides(_CFG_34, overrides)
    return ResNet(cfg)


@register_model(
    task="base",
    family="resnet",
    model_type="resnet",
    model_class=ResNet,
    default_config=_CFG_50,
    params=23508032,
)
def resnet_50(pretrained: bool = False, **overrides: object) -> ResNet:
    r"""ResNet-50 feature-extracting backbone (no classification head).

    Builds a :class:`ResNet` with the paper-cited ResNet-50 topology:
    :class:`_Bottleneck` blocks stacked ``[3, 4, 6, 3]`` over four
    stages, yielding approximately 25.6M parameters.  Reaches 76.13%
    ImageNet-1k top-1 in He et al., 2015 (Table 4).  By far the most
    common ResNet variant in production — used as the default backbone
    for Faster R-CNN, Mask R-CNN, DeepLab, and many other downstream
    tasks.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``resnet_50``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``resnet_50_cls`` is the checkpointed variant.
    **overrides
        Keyword overrides forwarded into :class:`ResNetConfig` to
        customise individual fields (``in_channels``, ``num_classes``,
        ``zero_init_residual``, …) without writing a config by hand.

    Returns
    -------
    ResNet
        Backbone with the ResNet-50 configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See He et al., "Deep Residual Learning for Image Recognition",
    CVPR 2016 (arXiv:1512.03385), Table 1.  The bottleneck block

    .. math::

        F(x) = W_3 \,\sigma\!\left(W_2 \,\sigma(W_1 x)\right)

    compresses the channel count via the leading 1×1 convolution and
    re-expands it via the trailing 1×1, reducing FLOPs at depth.  The
    final-stage output is 2048 channels (``hidden_sizes[3] *
    expansion``).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnet import resnet_50
    >>> model = resnet_50()
    >>> x = lucid.randn(2, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape   # (B, 2048, H/32, W/32)
    (2, 2048, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained("resnet_50", alternative="resnet_50_cls")
    cfg = apply_overrides(_CFG_50, overrides)
    return ResNet(cfg)


@register_model(
    task="base",
    family="resnet",
    model_type="resnet",
    model_class=ResNet,
    default_config=_CFG_101,
    params=42500160,
)
def resnet_101(pretrained: bool = False, **overrides: object) -> ResNet:
    r"""ResNet-101 feature-extracting backbone (no classification head).

    Builds a :class:`ResNet` with the paper-cited ResNet-101 topology:
    :class:`_Bottleneck` blocks stacked ``[3, 4, 23, 3]`` over four
    stages, yielding approximately 44.5M parameters.  Reaches 77.37%
    ImageNet-1k top-1 in He et al., 2015 (Table 4).  Higher-capacity
    drop-in replacement for ResNet-50 when accuracy matters more than
    inference latency.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``resnet_101``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``resnet_101_cls`` is the checkpointed variant.
    **overrides
        Keyword overrides forwarded into :class:`ResNetConfig` to
        customise individual fields without writing a config by hand.

    Returns
    -------
    ResNet
        Backbone with the ResNet-101 configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See He et al., "Deep Residual Learning for Image Recognition",
    CVPR 2016 (arXiv:1512.03385), Table 1.  The bulk of the depth lives
    in stage-3 (23 bottleneck blocks); stages 1, 2, and 4 are identical
    to ResNet-50.  Final-stage output is 2048 channels.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnet import resnet_101
    >>> model = resnet_101()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape   # (B, 2048, H/32, W/32)
    (1, 2048, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained("resnet_101", alternative="resnet_101_cls")
    cfg = apply_overrides(_CFG_101, overrides)
    return ResNet(cfg)


@register_model(
    task="base",
    family="resnet",
    model_type="resnet",
    model_class=ResNet,
    default_config=_CFG_152,
    params=58143808,
)
def resnet_152(pretrained: bool = False, **overrides: object) -> ResNet:
    r"""ResNet-152 feature-extracting backbone (no classification head).

    Builds a :class:`ResNet` with the paper-cited ResNet-152 topology:
    :class:`_Bottleneck` blocks stacked ``[3, 8, 36, 3]`` over four
    stages, yielding approximately 60.2M parameters.  Reaches 78.31%
    ImageNet-1k top-1 in He et al., 2015 (Table 4) — the highest of
    the canonical ResNet variants from the original paper.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``resnet_152``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``resnet_152_cls`` is the checkpointed variant.
    **overrides
        Keyword overrides forwarded into :class:`ResNetConfig` to
        customise individual fields without writing a config by hand.

    Returns
    -------
    ResNet
        Backbone with the ResNet-152 configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See He et al., "Deep Residual Learning for Image Recognition",
    CVPR 2016 (arXiv:1512.03385), Table 1.  The 36-block stage-3 is
    the defining feature; stages 1 and 4 are unchanged from
    ResNet-50/101, and stage-2 widens from 4 to 8 blocks.  Final-stage
    output is 2048 channels.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnet import resnet_152
    >>> model = resnet_152()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape   # (B, 2048, H/32, W/32)
    (1, 2048, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained("resnet_152", alternative="resnet_152_cls")
    cfg = apply_overrides(_CFG_152, overrides)
    return ResNet(cfg)


# ---------------------------------------------------------------------------
# Classification head registrations (task="image-classification")
# ---------------------------------------------------------------------------


@register_model(
    task="image-classification",
    family="resnet",
    model_type="resnet",
    model_class=ResNetForImageClassification,
    default_config=_CFG_18,
    params=11689512,
)
def resnet_18_cls(
    pretrained: bool | str = False,
    *,
    weights: ResNet18Weights | None = None,
    **overrides: object,
) -> ResNetForImageClassification:
    r"""ResNet-18 image classifier (backbone + GAP + linear head).

    Builds a :class:`ResNetForImageClassification` with the paper-cited
    ResNet-18 topology: :class:`_BasicBlock` blocks stacked
    ``[2, 2, 2, 2]`` over four stages, followed by global average
    pooling and a linear projection to ``config.num_classes``
    (default 1000 for ImageNet-1k).  Approximately 11.7M parameters
    and 69.76% ImageNet-1k top-1 in He et al., 2015 (Table 4).

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`ResNet18Weights.IMAGENET1K_V1`);
        a tag string (e.g. ``"IMAGENET1K_V1"``) → that specific
        checkpoint.  Mutually exclusive with ``weights`` (which wins if
        both are given).
    weights : ResNet18Weights, optional, keyword-only
        Explicit weights enum member, e.g.
        ``ResNet18Weights.IMAGENET1K_V1``.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`ResNetConfig`
        (typically ``num_classes`` to retarget the classifier or
        ``dropout`` to inject regularisation before the linear head).
        Note: overriding ``num_classes`` away from the checkpoint's
        class count makes pretrained loading fail the strict key/shape
        check — load with a matching head, then call
        :meth:`reset_classifier`.

    Returns
    -------
    ResNetForImageClassification
        Classifier with the ResNet-18 configuration applied (or with
        ``overrides`` merged on top of it), optionally initialised from
        pretrained weights.

    Notes
    -----
    See He et al., "Deep Residual Learning for Image Recognition",
    CVPR 2016 (arXiv:1512.03385), Table 1.  Pretrained weights are
    converted from reference_vision's ``ResNet18_Weights`` and hosted on the
    Hugging Face Hub under ``lucid-dl/resnet-18``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnet import resnet_18_cls
    >>> model = resnet_18_cls(num_classes=10)
    >>> x = lucid.randn(2, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (2, 10)

    Load ImageNet-pretrained weights:

    >>> model = resnet_18_cls(pretrained=True)            # DEFAULT tag
    >>> from lucid.models.vision.resnet import ResNet18Weights
    >>> model = resnet_18_cls(weights=ResNet18Weights.IMAGENET1K_V1)
    """
    entry = weights_mod.resolve_weights(ResNet18Weights, pretrained, weights)
    cfg = apply_overrides(_CFG_18, overrides)
    model = ResNetForImageClassification(cfg)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="resnet_18_cls")
    return model


@register_model(
    task="image-classification",
    family="resnet",
    model_type="resnet",
    model_class=ResNetForImageClassification,
    default_config=_CFG_34,
    params=21797672,
)
def resnet_34_cls(
    pretrained: bool | str = False,
    *,
    weights: ResNet34Weights | None = None,
    **overrides: object,
) -> ResNetForImageClassification:
    r"""ResNet-34 image classifier (backbone + GAP + linear head).

    Builds a :class:`ResNetForImageClassification` with the paper-cited
    ResNet-34 topology: :class:`_BasicBlock` blocks stacked
    ``[3, 4, 6, 3]`` over four stages, followed by global average
    pooling and a linear projection to ``config.num_classes``.
    Approximately 21.8M parameters and 73.31% ImageNet-1k top-1 in
    He et al., 2015 (Table 4).

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`ResNet34Weights.IMAGENET1K_V1`);
        a tag string → that specific checkpoint.  Mutually exclusive
        with ``weights`` (which wins if both are given).
    weights : ResNet34Weights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`ResNetConfig`.

    Returns
    -------
    ResNetForImageClassification
        Classifier with the ResNet-34 configuration applied (or with
        ``overrides`` merged on top of it), optionally initialised from
        pretrained weights.

    Notes
    -----
    See He et al., "Deep Residual Learning for Image Recognition",
    CVPR 2016 (arXiv:1512.03385), Table 1.  Pretrained weights are
    converted from reference_vision's ``ResNet34_Weights`` and hosted on the
    Hugging Face Hub under ``lucid-dl/resnet-34``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnet import resnet_34_cls
    >>> model = resnet_34_cls()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(ResNet34Weights, pretrained, weights)
    cfg = apply_overrides(_CFG_34, overrides)
    model = ResNetForImageClassification(cfg)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="resnet_34_cls")
    return model


@register_model(
    task="image-classification",
    family="resnet",
    model_type="resnet",
    model_class=ResNetForImageClassification,
    default_config=_CFG_50,
    params=25557032,
)
def resnet_50_cls(
    pretrained: bool | str = False,
    *,
    weights: ResNet50Weights | None = None,
    **overrides: object,
) -> ResNetForImageClassification:
    r"""ResNet-50 image classifier (backbone + GAP + linear head).

    Builds a :class:`ResNetForImageClassification` with the paper-cited
    ResNet-50 topology: :class:`_Bottleneck` blocks stacked
    ``[3, 4, 6, 3]`` over four stages, followed by global average
    pooling and a linear projection to ``config.num_classes``.
    Approximately 25.6M parameters and 76.13% ImageNet-1k top-1 in
    He et al., 2015 (Table 4) — the canonical default for ImageNet
    classification benchmarks.

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`ResNet50Weights.IMAGENET1K_V1`);
        a tag string → that specific checkpoint.  Mutually exclusive
        with ``weights`` (which wins if both are given).
    weights : ResNet50Weights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`ResNetConfig`.  Use
        ``num_classes=N`` to retarget the head (e.g. ``num_classes=10``
        for CIFAR-10 fine-tuning) and ``dropout=p`` to inject
        regularisation before the final linear layer.

    Returns
    -------
    ResNetForImageClassification
        Classifier with the ResNet-50 configuration applied (or with
        ``overrides`` merged on top of it), optionally initialised from
        pretrained weights.

    Notes
    -----
    See He et al., "Deep Residual Learning for Image Recognition",
    CVPR 2016 (arXiv:1512.03385), Table 1.  Final pre-classifier
    feature has 2048 channels (``hidden_sizes[3] * expansion``).
    Pretrained weights are converted from reference_vision's
    ``ResNet50_Weights`` and hosted on the Hugging Face Hub under
    ``lucid-dl/resnet-50``.

    Examples
    --------
    Run a forward pass without labels:

    >>> import lucid
    >>> from lucid.models.vision.resnet import resnet_50_cls
    >>> model = resnet_50_cls()
    >>> x = lucid.randn(4, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (4, 1000)

    Retarget to CIFAR-10 and add dropout:

    >>> model = resnet_50_cls(num_classes=10, dropout=0.1)
    >>> model.config.num_classes
    10
    """
    entry = weights_mod.resolve_weights(ResNet50Weights, pretrained, weights)
    cfg = apply_overrides(_CFG_50, overrides)
    model = ResNetForImageClassification(cfg)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="resnet_50_cls")
    return model


@register_model(
    task="image-classification",
    family="resnet",
    model_type="resnet",
    model_class=ResNetForImageClassification,
    default_config=_CFG_101,
    params=44549160,
)
def resnet_101_cls(
    pretrained: bool | str = False,
    *,
    weights: ResNet101Weights | None = None,
    **overrides: object,
) -> ResNetForImageClassification:
    r"""ResNet-101 image classifier (backbone + GAP + linear head).

    Builds a :class:`ResNetForImageClassification` with the paper-cited
    ResNet-101 topology: :class:`_Bottleneck` blocks stacked
    ``[3, 4, 23, 3]`` over four stages, followed by global average
    pooling and a linear projection to ``config.num_classes``.
    Approximately 44.5M parameters and 77.37% ImageNet-1k top-1 in
    He et al., 2015 (Table 4).

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`ResNet101Weights.IMAGENET1K_V1`);
        a tag string → that specific checkpoint.  Mutually exclusive
        with ``weights`` (which wins if both are given).
    weights : ResNet101Weights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`ResNetConfig`.

    Returns
    -------
    ResNetForImageClassification
        Classifier with the ResNet-101 configuration applied (or with
        ``overrides`` merged on top of it), optionally initialised from
        pretrained weights.

    Notes
    -----
    See He et al., "Deep Residual Learning for Image Recognition",
    CVPR 2016 (arXiv:1512.03385), Table 1.  Pretrained weights are
    converted from reference_vision's ``ResNet101_Weights`` and hosted on the
    Hugging Face Hub under ``lucid-dl/resnet-101``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnet import resnet_101_cls
    >>> model = resnet_101_cls()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(ResNet101Weights, pretrained, weights)
    cfg = apply_overrides(_CFG_101, overrides)
    model = ResNetForImageClassification(cfg)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="resnet_101_cls")
    return model


@register_model(
    task="image-classification",
    family="resnet",
    model_type="resnet",
    model_class=ResNetForImageClassification,
    default_config=_CFG_152,
    params=60192808,
)
def resnet_152_cls(
    pretrained: bool | str = False,
    *,
    weights: ResNet152Weights | None = None,
    **overrides: object,
) -> ResNetForImageClassification:
    r"""ResNet-152 image classifier (backbone + GAP + linear head).

    Builds a :class:`ResNetForImageClassification` with the paper-cited
    ResNet-152 topology: :class:`_Bottleneck` blocks stacked
    ``[3, 8, 36, 3]`` over four stages, followed by global average
    pooling and a linear projection to ``config.num_classes``.
    Approximately 60.2M parameters and 78.31% ImageNet-1k top-1 in
    He et al., 2015 (Table 4) — the deepest and most accurate of the
    original ResNet variants.

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`ResNet152Weights.IMAGENET1K_V1`);
        a tag string → that specific checkpoint.  Mutually exclusive
        with ``weights`` (which wins if both are given).
    weights : ResNet152Weights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`ResNetConfig`.

    Returns
    -------
    ResNetForImageClassification
        Classifier with the ResNet-152 configuration applied (or with
        ``overrides`` merged on top of it), optionally initialised from
        pretrained weights.

    Notes
    -----
    See He et al., "Deep Residual Learning for Image Recognition",
    CVPR 2016 (arXiv:1512.03385), Table 1.  Pretrained weights are
    converted from reference_vision's ``ResNet152_Weights`` and hosted on the
    Hugging Face Hub under ``lucid-dl/resnet-152``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnet import resnet_152_cls
    >>> model = resnet_152_cls()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(ResNet152Weights, pretrained, weights)
    cfg = apply_overrides(_CFG_152, overrides)
    model = ResNetForImageClassification(cfg)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="resnet_152_cls")
    return model


# ---------------------------------------------------------------------------
# Wide ResNet-50-2
# ---------------------------------------------------------------------------


@register_model(
    task="base",
    family="resnet",
    model_type="resnet",
    model_class=ResNet,
    default_config=_CFG_WIDE50,
    params=66834240,
)
def wide_resnet_50(pretrained: bool = False, **overrides: object) -> ResNet:
    r"""Wide ResNet-50-2 feature-extracting backbone (no classification head).

    Builds a :class:`ResNet` with ResNet-50's ``[3, 4, 6, 3]``
    :class:`_Bottleneck` layout but with a ``bottleneck_width_mult=2``
    multiplier on the inner 3×3 convolutions.  The per-stage output
    channel count is unchanged from ResNet-50 (256 / 512 / 1024 /
    2048), but the inner bottleneck width doubles — yielding roughly
    68.9M parameters.  Defined in Zagoruyko & Komodakis, "Wide
    Residual Networks", BMVC 2016 (arXiv:1605.07146).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``wide_resnet_50``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``wide_resnet_50_cls`` is the checkpointed variant.
    **overrides
        Keyword overrides forwarded into :class:`ResNetConfig`.

    Returns
    -------
    ResNet
        Backbone with the Wide ResNet-50-2 configuration applied (or
        with ``overrides`` merged on top of it).

    Notes
    -----
    Widening the inner channels lets shallower networks recover (and
    surpass) the accuracy of much deeper canonical ResNets while being
    easier to parallelise on modern accelerators because the FLOPs
    are concentrated in a smaller number of wider layers.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnet import wide_resnet_50
    >>> model = wide_resnet_50()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape   # (B, 2048, H/32, W/32)
    (1, 2048, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained("wide_resnet_50", alternative="wide_resnet_50_cls")
    cfg = apply_overrides(_CFG_WIDE50, overrides)
    return ResNet(cfg)


@register_model(
    task="image-classification",
    family="resnet",
    model_type="resnet",
    model_class=ResNetForImageClassification,
    default_config=_CFG_WIDE50,
    params=68883240,
)
def wide_resnet_50_cls(
    pretrained: bool | str = False,
    *,
    weights: WideResNet50Weights | None = None,
    **overrides: object,
) -> ResNetForImageClassification:
    r"""Wide ResNet-50-2 image classifier (backbone + GAP + linear head).

    Builds a :class:`ResNetForImageClassification` with the Wide
    ResNet-50-2 configuration: ResNet-50's ``[3, 4, 6, 3]`` bottleneck
    layout with a 2x inner-channel multiplier.  Approximately 68.9M
    parameters; from Zagoruyko & Komodakis, "Wide Residual Networks",
    BMVC 2016 (arXiv:1605.07146).

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag
        (:attr:`WideResNet50Weights.IMAGENET1K_V1`); a tag string →
        that specific checkpoint.  Mutually exclusive with ``weights``
        (which wins if both are given).
    weights : WideResNet50Weights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`ResNetConfig`.

    Returns
    -------
    ResNetForImageClassification
        Classifier with the Wide ResNet-50-2 configuration applied (or
        with ``overrides`` merged on top of it), optionally initialised
        from pretrained weights.

    Notes
    -----
    Pretrained weights are converted from reference_vision's
    ``Wide_ResNet50_2_Weights`` and hosted on the Hugging Face Hub under
    ``lucid-dl/wide-resnet-50-2``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnet import wide_resnet_50_cls
    >>> model = wide_resnet_50_cls()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(WideResNet50Weights, pretrained, weights)
    cfg = apply_overrides(_CFG_WIDE50, overrides)
    model = ResNetForImageClassification(cfg)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="wide_resnet_50_cls")
    return model


# ---------------------------------------------------------------------------
# Wide ResNet-101-2
# ---------------------------------------------------------------------------


@register_model(
    task="base",
    family="resnet",
    model_type="resnet",
    model_class=ResNet,
    default_config=_CFG_WIDE101,
    params=124837696,
)
def wide_resnet_101(pretrained: bool = False, **overrides: object) -> ResNet:
    r"""Wide ResNet-101-2 feature-extracting backbone (no classification head).

    Builds a :class:`ResNet` with ResNet-101's ``[3, 4, 23, 3]``
    :class:`_Bottleneck` layout and a ``bottleneck_width_mult=2``
    multiplier on the inner 3×3 convolutions.  Output channel counts
    match standard ResNet-101 (256 / 512 / 1024 / 2048); approximately
    126.9M parameters.  From Zagoruyko & Komodakis, "Wide Residual
    Networks", BMVC 2016 (arXiv:1605.07146).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``wide_resnet_101``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``wide_resnet_101_cls`` is the checkpointed variant.
    **overrides
        Keyword overrides forwarded into :class:`ResNetConfig`.

    Returns
    -------
    ResNet
        Backbone with the Wide ResNet-101-2 configuration applied (or
        with ``overrides`` merged on top of it).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnet import wide_resnet_101
    >>> model = wide_resnet_101()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape   # (B, 2048, H/32, W/32)
    (1, 2048, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained("wide_resnet_101", alternative="wide_resnet_101_cls")
    cfg = apply_overrides(_CFG_WIDE101, overrides)
    return ResNet(cfg)


@register_model(
    task="image-classification",
    family="resnet",
    model_type="resnet",
    model_class=ResNetForImageClassification,
    default_config=_CFG_WIDE101,
    params=126886696,
)
def wide_resnet_101_cls(
    pretrained: bool | str = False,
    *,
    weights: WideResNet101Weights | None = None,
    **overrides: object,
) -> ResNetForImageClassification:
    r"""Wide ResNet-101-2 image classifier (backbone + GAP + linear head).

    Builds a :class:`ResNetForImageClassification` with the Wide
    ResNet-101-2 configuration: ResNet-101's ``[3, 4, 23, 3]``
    bottleneck layout with a 2x inner-channel multiplier.
    Approximately 126.9M parameters; from Zagoruyko & Komodakis,
    "Wide Residual Networks", BMVC 2016 (arXiv:1605.07146).

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag
        (:attr:`WideResNet101Weights.IMAGENET1K_V1`); a tag string →
        that specific checkpoint.  Mutually exclusive with ``weights``
        (which wins if both are given).
    weights : WideResNet101Weights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`ResNetConfig`.

    Returns
    -------
    ResNetForImageClassification
        Classifier with the Wide ResNet-101-2 configuration applied
        (or with ``overrides`` merged on top of it), optionally
        initialised from pretrained weights.

    Notes
    -----
    Pretrained weights are converted from reference_vision's
    ``Wide_ResNet101_2_Weights`` and hosted on the Hugging Face Hub
    under ``lucid-dl/wide-resnet-101-2``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnet import wide_resnet_101_cls
    >>> model = wide_resnet_101_cls()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(WideResNet101Weights, pretrained, weights)
    cfg = apply_overrides(_CFG_WIDE101, overrides)
    model = ResNetForImageClassification(cfg)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="wide_resnet_101_cls")
    return model


# ---------------------------------------------------------------------------
# ResNet-200
# ---------------------------------------------------------------------------


@register_model(
    task="base",
    family="resnet",
    model_type="resnet",
    model_class=ResNet,
    default_config=_CFG_200,
    params=62624832,
)
def resnet_200(pretrained: bool = False, **overrides: object) -> ResNet:
    r"""ResNet-200 feature-extracting backbone (no classification head).

    Builds a :class:`ResNet` with the deep bottleneck topology
    introduced in He et al., "Identity Mappings in Deep Residual
    Networks", ECCV 2016 (arXiv:1603.05027): :class:`_Bottleneck`
    blocks stacked ``[3, 24, 36, 3]`` over four stages — approximately
    64.7M parameters.  Used as a high-capacity backbone for ImageNet
    pre-training in large-scale vision pipelines.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``resnet_200``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides
        Keyword overrides forwarded into :class:`ResNetConfig`.

    Returns
    -------
    ResNet
        Backbone with the ResNet-200 configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    Final-stage output is 2048 channels.  Memory cost scales with the
    24-block stage-2 — keep ``zero_init_residual=True`` for stable
    convergence at large batch sizes.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnet import resnet_200
    >>> model = resnet_200()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape   # (B, 2048, H/32, W/32)
    (1, 2048, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained("resnet_200")
    cfg = apply_overrides(_CFG_200, overrides)
    return ResNet(cfg)


@register_model(
    task="image-classification",
    family="resnet",
    model_type="resnet",
    model_class=ResNetForImageClassification,
    default_config=_CFG_200,
    params=64673832,
)
def resnet_200_cls(pretrained: bool = False, **overrides: object) -> ResNetForImageClassification:
    r"""ResNet-200 image classifier (backbone + GAP + linear head).

    Builds a :class:`ResNetForImageClassification` with the deep
    bottleneck topology ``[3, 24, 36, 3]`` from He et al., "Identity
    Mappings in Deep Residual Networks", ECCV 2016 (arXiv:1603.05027).
    Approximately 64.7M parameters.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``resnet_200_cls``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides
        Keyword overrides forwarded into :class:`ResNetConfig`.

    Returns
    -------
    ResNetForImageClassification
        Classifier with the ResNet-200 configuration applied (or with
        ``overrides`` merged on top of it).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnet import resnet_200_cls
    >>> model = resnet_200_cls()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    if pretrained:
        reject_unavailable_pretrained("resnet_200_cls")
    cfg = apply_overrides(_CFG_200, overrides)
    return ResNetForImageClassification(cfg)


# ---------------------------------------------------------------------------
# ResNet-269
# ---------------------------------------------------------------------------


@register_model(
    task="base",
    family="resnet",
    model_type="resnet",
    model_class=ResNet,
    default_config=_CFG_269,
    params=100024384,
)
def resnet_269(pretrained: bool = False, **overrides: object) -> ResNet:
    r"""ResNet-269 feature-extracting backbone (no classification head).

    Builds a :class:`ResNet` with the very deep bottleneck topology
    ``[3, 30, 48, 8]``, an extension of the canonical ResNet design
    studied in subsequent deep-residual literature (notably as the
    teacher network in MXNet-style training pipelines and as an
    inception-resnet baseline).  Approximately 102.1M parameters.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``resnet_269``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides
        Keyword overrides forwarded into :class:`ResNetConfig`.

    Returns
    -------
    ResNet
        Backbone with the ResNet-269 configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    Final-stage output is 2048 channels.  Training such a deep network
    benefits significantly from ``zero_init_residual=True`` and a long
    warm-up schedule.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnet import resnet_269
    >>> model = resnet_269()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape   # (B, 2048, H/32, W/32)
    (1, 2048, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained("resnet_269")
    cfg = apply_overrides(_CFG_269, overrides)
    return ResNet(cfg)


@register_model(
    task="image-classification",
    family="resnet",
    model_type="resnet",
    model_class=ResNetForImageClassification,
    default_config=_CFG_269,
    params=102073384,
)
def resnet_269_cls(pretrained: bool = False, **overrides: object) -> ResNetForImageClassification:
    r"""ResNet-269 image classifier (backbone + GAP + linear head).

    Builds a :class:`ResNetForImageClassification` with the very deep
    bottleneck topology ``[3, 30, 48, 8]``.  Approximately 102.1M
    parameters — primarily useful as a high-capacity teacher in
    knowledge distillation pipelines.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``resnet_269_cls``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides
        Keyword overrides forwarded into :class:`ResNetConfig`.

    Returns
    -------
    ResNetForImageClassification
        Classifier with the ResNet-269 configuration applied (or with
        ``overrides`` merged on top of it).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnet import resnet_269_cls
    >>> model = resnet_269_cls()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    if pretrained:
        reject_unavailable_pretrained("resnet_269_cls")
    cfg = apply_overrides(_CFG_269, overrides)
    return ResNetForImageClassification(cfg)
