"""Registry factories for all SKNet variants."""

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models.vision.sknet._config import SKNetConfig
from lucid.models.vision.sknet._model import SKNet, SKNetForImageClassification
from lucid.models.vision.sknet._weights import SKResNet18Weights, SKResNet34Weights
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained

# ---------------------------------------------------------------------------
# Canonical configs
# ---------------------------------------------------------------------------

# sk_resnet_18 / sk_resnet_34:
#   basic block (expansion=1); the first 3×3 conv per block is a
#   SelectiveKernel unit (two parallel 3×3 branches + channel attention),
#   the second is a plain 3×3 conv.  Matches the reference ``skresnet18`` /
#   ``skresnet34`` recipe: ``split_input=True`` (each branch receives half
#   the input channels), ``rd_ratio=1/8`` with ``rd_divisor=16`` for the
#   attention bottleneck.  sk_resnet_18: ~11.96M params;
#   sk_resnet_34: ~22.28M params.
_CFG_SK18 = SKNetConfig(
    layers=(2, 2, 2, 2),
    block_type="basic",
    cardinality=1,
    base_width=64,
    split_input=True,
    rd_ratio=1.0 / 8,
    rd_divisor=16,
    min_attn_channels=16,
)
_CFG_SK34 = SKNetConfig(
    layers=(3, 4, 6, 3),
    block_type="basic",
    cardinality=1,
    base_width=64,
    split_input=True,
    rd_ratio=1.0 / 8,
    rd_divisor=16,
    min_attn_channels=16,
)

# sk_resnet_50 / sk_resnet_101:
#   cardinality=1, base_width=64, split_input=True  →  timm ``skresnet50``
#   25,803,160 parameters for sk_resnet_50_cls (1000-class head)
_CFG_SK50 = SKNetConfig(layers=(3, 4, 6, 3), cardinality=1, base_width=64, split_input=True)
# Section 3.2: "SKNet-101, which has {3,4,23,3} SK units".  Table 2 pins it
# at 48.9M params / 8.46 GFLOPs, which is the ResNeXt-101 32x4d base -- not
# the cardinality-1, width-64 ResNet base this used to build.
_CFG_SK101 = SKNetConfig(layers=(3, 4, 23, 3), cardinality=32, base_width=4, split_input=True)

# sk_resnext_50_32x4d:
#   cardinality=32, base_width=4, split_input=False, rd_ratio=1/16, rd_divisor=32
#   Equivalent to the SKNet-50 entry in the original paper.  27,479,784 parameters.
_CFG_SK_RX50 = SKNetConfig(
    layers=(3, 4, 6, 3),
    cardinality=32,
    base_width=4,
    split_input=False,
    rd_ratio=1.0 / 16,
    rd_divisor=32,
)


# ---------------------------------------------------------------------------
# Backbone registrations (task="base")
# ---------------------------------------------------------------------------


@register_model(
    task="base",
    family="sknet",
    model_type="sknet",
    model_class=SKNet,
    default_config=_CFG_SK18,
    params=11463616,
)
def sk_resnet_18(pretrained: bool = False, **overrides: object) -> SKNet:
    r"""SK-ResNet-18 feature-extracting backbone (no classification head).

    Builds an :class:`SKNet` with ResNet-18 topology
    (:class:`_SelectiveKernelBasic` blocks stacked ``[2, 2, 2, 2]``).
    Both :math:`3 \times 3` convolutions inside every block are
    replaced by Selective Kernel units, giving full SK treatment
    of the basic-block design.  Approximately 24.7M parameters.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``sk_resnet_18``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``sk_resnet_18_cls`` is the checkpointed variant.
    **overrides
        Keyword overrides forwarded into :class:`SKNetConfig`.

    Returns
    -------
    SKNet
        Backbone with the SK-ResNet-18 configuration applied (or
        with ``overrides`` merged on top of it).

    Notes
    -----
    See Li et al., "Selective Kernel Networks", CVPR 2019
    (arXiv:1903.06586).  ``_CFG_SK18`` uses ``rd_ratio = 1/8``,
    ``rd_divisor = 16`` and ``split_input = True``, so each branch
    receives half the input.  Only the *first* :math:`3 \times 3` of
    each basic block is a Selective Kernel unit — the second stays a
    plain conv-BN-act (see ``_SelectiveKernelBasic``).  ~11.5M
    parameters.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.sknet import sk_resnet_18
    >>> model = sk_resnet_18()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 512, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained("sk_resnet_18", alternative="sk_resnet_18_cls")
    cfg = apply_overrides(_CFG_SK18, overrides)
    return SKNet(cfg)


@register_model(
    task="base",
    family="sknet",
    model_type="sknet",
    model_class=SKNet,
    default_config=_CFG_SK34,
    params=21803392,
)
def sk_resnet_34(pretrained: bool = False, **overrides: object) -> SKNet:
    r"""SK-ResNet-34 feature-extracting backbone (no classification head).

    Builds an :class:`SKNet` with ResNet-34 topology
    (:class:`_SelectiveKernelBasic` blocks stacked ``[3, 4, 6, 3]``).
    Only the *first* :math:`3 \times 3` of each basic block is a
    Selective Kernel unit; the second stays a plain conv-BN-act.
    Approximately 21.8M parameters.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``sk_resnet_34``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``sk_resnet_34_cls`` is the checkpointed variant.
    **overrides
        Keyword overrides forwarded into :class:`SKNetConfig`.

    Returns
    -------
    SKNet
        Backbone with the SK-ResNet-34 configuration applied (or
        with ``overrides`` merged on top of it).

    Notes
    -----
    See Li et al., "Selective Kernel Networks", CVPR 2019
    (arXiv:1903.06586).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.sknet import sk_resnet_34
    >>> model = sk_resnet_34()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 512, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained("sk_resnet_34", alternative="sk_resnet_34_cls")
    cfg = apply_overrides(_CFG_SK34, overrides)
    return SKNet(cfg)


@register_model(
    task="base",
    family="sknet",
    model_type="sknet",
    model_class=SKNet,
    default_config=_CFG_SK50,
    params=23879104,
)
def sk_resnet_50(pretrained: bool = False, **overrides: object) -> SKNet:
    r"""SK-ResNet-50 feature-extracting backbone (no classification head).

    Builds an :class:`SKNet` with ResNet-50 bottleneck topology
    (:class:`_SelectiveKernelBottleneck` blocks stacked
    ``[3, 4, 6, 3]``).  The central :math:`3 \times 3` of every
    bottleneck is replaced by a two-branch Selective Kernel unit
    with ``split_input=True`` (each branch receives half the
    channels — matching timm's ``skresnet50`` layout).
    Approximately 25.8M parameters.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``sk_resnet_50``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides
        Keyword overrides forwarded into :class:`SKNetConfig`.

    Returns
    -------
    SKNet
        Backbone with the SK-ResNet-50 configuration applied (or
        with ``overrides`` merged on top of it).

    Notes
    -----
    See Li et al., "Selective Kernel Networks", CVPR 2019
    (arXiv:1903.06586), Table 1.  Final-stage output is 2048
    channels.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.sknet import sk_resnet_50
    >>> model = sk_resnet_50()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 2048, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained("sk_resnet_50")
    cfg = apply_overrides(_CFG_SK50, overrides)
    return SKNet(cfg)


@register_model(
    task="base",
    family="sknet",
    model_type="sknet",
    model_class=SKNet,
    default_config=_CFG_SK101,
    params=44019008,
)
def sk_resnet_101(pretrained: bool = False, **overrides: object) -> SKNet:
    r"""SK-ResNet-101 feature-extracting backbone (no classification head).

    Builds an :class:`SKNet` with ResNet-101 bottleneck topology
    (:class:`_SelectiveKernelBottleneck` blocks stacked
    ``[3, 4, 23, 3]``).  Approximately 45M parameters.  Deeper
    variant of SK-ResNet-50 for higher-accuracy budgets.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``sk_resnet_101``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides
        Keyword overrides forwarded into :class:`SKNetConfig`.

    Returns
    -------
    SKNet
        Backbone with the SK-ResNet-101 configuration applied (or
        with ``overrides`` merged on top of it).

    Notes
    -----
    See Li et al., "Selective Kernel Networks", CVPR 2019
    (arXiv:1903.06586).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.sknet import sk_resnet_101
    >>> model = sk_resnet_101()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 2048, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained("sk_resnet_101")
    cfg = apply_overrides(_CFG_SK101, overrides)
    return SKNet(cfg)


@register_model(
    task="base",
    family="sknet",
    model_type="sknet",
    model_class=SKNet,
    default_config=_CFG_SK_RX50,
    params=25430784,
)
def sk_resnext_50_32x4d(pretrained: bool = False, **overrides: object) -> SKNet:
    r"""SK-ResNeXt-50 32×4d feature-extracting backbone (the paper's SKNet-50).

    Builds an :class:`SKNet` with ResNet-50 bottleneck topology
    and ResNeXt-style grouped widening: ``cardinality = 32``,
    ``base_width = 4``, ``split_input = False``.  The bottleneck
    width per stage follows the ResNeXt formula

    .. math::

        \text{width} = \lfloor \mathrm{planes} \cdot
            \tfrac{\text{base\_width}}{64} \rfloor \cdot
            \text{cardinality},

    matching the ``SKNet-50`` entry in Li et al., 2019.  Table 2 gives
    that entry 27.5M parameters, 4.47 GFLOPs and top-1 *error* 20.79 at
    a 224 centre crop — i.e. 79.21% accuracy, not the 77.5% previously
    quoted here (the paper reports error, not accuracy).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``sk_resnext_50_32x4d``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides
        Keyword overrides forwarded into :class:`SKNetConfig`.

    Returns
    -------
    SKNet
        Backbone with the SK-ResNeXt-50-32×4d configuration
        applied (or with ``overrides`` merged on top of it).

    Notes
    -----
    See Li et al., "Selective Kernel Networks", CVPR 2019
    (arXiv:1903.06586), Table 1 (SKNet-50 row).  Combines the
    cardinality of ResNeXt with the data-dependent receptive-field
    selection of SK.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.sknet import sk_resnext_50_32x4d
    >>> model = sk_resnext_50_32x4d()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 2048, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained("sk_resnext_50_32x4d")
    cfg = apply_overrides(_CFG_SK_RX50, overrides)
    return SKNet(cfg)


# ---------------------------------------------------------------------------
# Classification head registrations (task="image-classification")
# ---------------------------------------------------------------------------


@register_model(
    task="image-classification",
    family="sknet",
    model_type="sknet",
    model_class=SKNetForImageClassification,
    default_config=_CFG_SK18,
    params=11976616,
)
def sk_resnet_18_cls(
    pretrained: bool | str = False,
    *,
    weights: SKResNet18Weights | None = None,
    **overrides: object,
) -> SKNetForImageClassification:
    r"""SK-ResNet-18 image classifier (backbone + GAP + linear head).

    Builds an :class:`SKNetForImageClassification` with the
    SK-ResNet-18 backbone (basic blocks stacked ``[2, 2, 2, 2]``;
    the first :math:`3 \times 3` of each block is a Selective Kernel
    unit, the second a plain conv) followed by global average pooling
    and a linear projection to ``config.num_classes``.  Approximately
    11.96M parameters.

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`SKResNet18Weights.RA_IN1K`); a
        tag string (e.g. ``"RA_IN1K"``) → that specific checkpoint.
        Mutually exclusive with ``weights`` (which wins if both are
        given).
    weights : SKResNet18Weights, optional, keyword-only
        Explicit weights enum member, e.g.
        ``SKResNet18Weights.RA_IN1K``.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`SKNetConfig`.

    Returns
    -------
    SKNetForImageClassification
        Classifier with the SK-ResNet-18 configuration applied
        (or with ``overrides`` merged on top of it), optionally
        initialised from pretrained weights.

    Notes
    -----
    See Li et al., "Selective Kernel Networks", CVPR 2019
    (arXiv:1903.06586).  Pretrained weights are converted from
    ``timm``'s ``skresnet18.ra_in1k`` and hosted on the Hugging Face
    Hub under ``lucid-dl/sk-resnet-18``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.sknet import sk_resnet_18_cls
    >>> model = sk_resnet_18_cls(num_classes=10)
    >>> x = lucid.randn(2, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (2, 10)

    Load ImageNet-pretrained weights:

    >>> model = sk_resnet_18_cls(pretrained=True)            # DEFAULT tag
    >>> from lucid.models.vision.sknet import SKResNet18Weights
    >>> model = sk_resnet_18_cls(weights=SKResNet18Weights.RA_IN1K)
    """
    entry = weights_mod.resolve_weights(SKResNet18Weights, pretrained, weights)
    cfg = apply_overrides(_CFG_SK18, overrides)
    model = SKNetForImageClassification(cfg)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="sk_resnet_18_cls")
    return model


@register_model(
    task="image-classification",
    family="sknet",
    model_type="sknet",
    model_class=SKNetForImageClassification,
    default_config=_CFG_SK34,
    params=22316392,
)
def sk_resnet_34_cls(
    pretrained: bool | str = False,
    *,
    weights: SKResNet34Weights | None = None,
    **overrides: object,
) -> SKNetForImageClassification:
    r"""SK-ResNet-34 image classifier (backbone + GAP + linear head).

    Builds an :class:`SKNetForImageClassification` with the
    SK-ResNet-34 backbone (basic blocks stacked ``[3, 4, 6, 3]``;
    the first :math:`3 \times 3` of each block is a Selective Kernel
    unit, the second a plain conv) followed by global average pooling
    and a linear projection.  Approximately 22.28M parameters.

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`SKResNet34Weights.RA_IN1K`); a
        tag string → that specific checkpoint.  Mutually exclusive with
        ``weights`` (which wins if both are given).
    weights : SKResNet34Weights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`SKNetConfig`.

    Returns
    -------
    SKNetForImageClassification
        Classifier with the SK-ResNet-34 configuration applied
        (or with ``overrides`` merged on top of it), optionally
        initialised from pretrained weights.

    Notes
    -----
    See Li et al., "Selective Kernel Networks", CVPR 2019
    (arXiv:1903.06586).  Pretrained weights are converted from
    ``timm``'s ``skresnet34.ra_in1k`` and hosted on the Hugging Face
    Hub under ``lucid-dl/sk-resnet-34``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.sknet import sk_resnet_34_cls
    >>> model = sk_resnet_34_cls()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(SKResNet34Weights, pretrained, weights)
    cfg = apply_overrides(_CFG_SK34, overrides)
    model = SKNetForImageClassification(cfg)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="sk_resnet_34_cls")
    return model


@register_model(
    task="image-classification",
    family="sknet",
    model_type="sknet",
    model_class=SKNetForImageClassification,
    default_config=_CFG_SK50,
    params=25928104,
)
def sk_resnet_50_cls(pretrained: bool = False, **overrides: object) -> SKNetForImageClassification:
    r"""SK-ResNet-50 image classifier (backbone + GAP + linear head).

    Builds an :class:`SKNetForImageClassification` with the
    SK-ResNet-50 backbone (bottleneck blocks stacked
    ``[3, 4, 6, 3]``, one SK unit per bottleneck) followed by
    global average pooling and a linear projection.  Approximately
    25.8M parameters.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``sk_resnet_50_cls``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides
        Keyword overrides forwarded into :class:`SKNetConfig`.

    Returns
    -------
    SKNetForImageClassification
        Classifier with the SK-ResNet-50 configuration applied
        (or with ``overrides`` merged on top of it).

    Notes
    -----
    See Li et al., "Selective Kernel Networks", CVPR 2019
    (arXiv:1903.06586), Table 1.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.sknet import sk_resnet_50_cls
    >>> model = sk_resnet_50_cls(num_classes=10)
    >>> x = lucid.randn(2, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (2, 10)
    """
    if pretrained:
        reject_unavailable_pretrained("sk_resnet_50_cls")
    cfg = apply_overrides(_CFG_SK50, overrides)
    return SKNetForImageClassification(cfg)


@register_model(
    task="image-classification",
    family="sknet",
    model_type="sknet",
    model_class=SKNetForImageClassification,
    default_config=_CFG_SK101,
    params=46068008,
)
def sk_resnet_101_cls(pretrained: bool = False, **overrides: object) -> SKNetForImageClassification:
    r"""SK-ResNet-101 image classifier (backbone + GAP + linear head).

    Builds an :class:`SKNetForImageClassification` with the
    SK-ResNet-101 backbone (bottleneck blocks stacked
    ``[3, 4, 23, 3]``) followed by global average pooling and a
    linear classifier.  Approximately 45M parameters.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``sk_resnet_101_cls``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides
        Keyword overrides forwarded into :class:`SKNetConfig`.

    Returns
    -------
    SKNetForImageClassification
        Classifier with the SK-ResNet-101 configuration applied
        (or with ``overrides`` merged on top of it).

    Notes
    -----
    See Li et al., "Selective Kernel Networks", CVPR 2019
    (arXiv:1903.06586).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.sknet import sk_resnet_101_cls
    >>> model = sk_resnet_101_cls()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    if pretrained:
        reject_unavailable_pretrained("sk_resnet_101_cls")
    cfg = apply_overrides(_CFG_SK101, overrides)
    return SKNetForImageClassification(cfg)


@register_model(
    task="image-classification",
    family="sknet",
    model_type="sknet",
    model_class=SKNetForImageClassification,
    default_config=_CFG_SK_RX50,
    params=27479784,
)
def sk_resnext_50_32x4d_cls(
    pretrained: bool = False, **overrides: object
) -> SKNetForImageClassification:
    r"""SK-ResNeXt-50 32×4d image classifier — the paper's SKNet-50.

    Builds an :class:`SKNetForImageClassification` with the
    ResNeXt-style SK backbone (``cardinality = 32``,
    ``base_width = 4``) followed by global average pooling and a
    linear projection to ``config.num_classes``.  Approximately
    27.5M parameters and 79.21% ImageNet-1k top-1 accuracy
    (Table 2 reports 20.79 top-1 *error*) in
    Li et al., 2019 (Table 1, SKNet-50 row).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``sk_resnext_50_32x4d_cls``, so ``True``
        raises ``NotImplementedError`` rather than return randomly initialised
        weights.
    **overrides
        Keyword overrides forwarded into :class:`SKNetConfig`.

    Returns
    -------
    SKNetForImageClassification
        Classifier with the SK-ResNeXt-50-32×4d configuration
        applied (or with ``overrides`` merged on top of it).

    Notes
    -----
    See Li et al., "Selective Kernel Networks", CVPR 2019
    (arXiv:1903.06586), Table 1 (SKNet-50 row).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.sknet import sk_resnext_50_32x4d_cls
    >>> model = sk_resnext_50_32x4d_cls()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    if pretrained:
        reject_unavailable_pretrained("sk_resnext_50_32x4d_cls")
    cfg = apply_overrides(_CFG_SK_RX50, overrides)
    return SKNetForImageClassification(cfg)
