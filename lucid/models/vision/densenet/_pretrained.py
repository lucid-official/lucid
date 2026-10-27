"""Registry factories for DenseNet variants."""

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models.vision.densenet._config import DenseNetConfig
from lucid.models.vision.densenet._model import DenseNet, DenseNetForImageClassification
from lucid.models.vision.densenet._weights import (
    DenseNet121Weights,
    DenseNet161Weights,
    DenseNet169Weights,
    DenseNet201Weights,
)
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained

_CFG_121 = DenseNetConfig(block_config=(6, 12, 24, 16), growth_rate=32, num_init_features=64)
# DenseNet-161: k=48 with a 96-channel stem.  NOT from the paper — Table 1
# lists exactly four ImageNet architectures (121/169/201/264) and states
# "Growth rate k=32 for all networks"; the strings "161" and "k=48" appear
# nowhere in it.  The provenance is the authors' own released
# reference-framework models (liuzhuang13/DenseNet: "DenseNet-161 (k=48) |
# 22.2"), which the reference vision library mirrors and which the shipped
# checkpoint comes from.
_CFG_161 = DenseNetConfig(block_config=(6, 12, 36, 24), growth_rate=48, num_init_features=96)
_CFG_169 = DenseNetConfig(block_config=(6, 12, 32, 32), growth_rate=32, num_init_features=64)
_CFG_201 = DenseNetConfig(block_config=(6, 12, 48, 32), growth_rate=32, num_init_features=64)
_CFG_264 = DenseNetConfig(block_config=(6, 12, 64, 48), growth_rate=32, num_init_features=64)


# ── Backbones ─────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="densenet",
    model_type="densenet",
    model_class=DenseNet,
    default_config=_CFG_121,
    params=6_953_856,
)
def densenet_121(pretrained: bool = False, **overrides: object) -> DenseNet:
    r"""DenseNet-121 feature-extracting backbone (no classification head).

    Builds a :class:`DenseNet` with the paper-cited DenseNet-121
    topology: per-block dense-layer counts ``(6, 12, 24, 16)``,
    growth rate :math:`k = 32`, initial conv channels 64.
    Approximately 8.0 M parameters total — by far the most
    parameter-efficient of the ImageNet DenseNets, reaching a top-1
    error of 25.0% (Huang et al., 2017, Table 3).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``densenet_121``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``densenet_121_cls`` is the checkpointed variant.
    **overrides
        Keyword overrides forwarded into :class:`DenseNetConfig`
        (``growth_rate``, ``dropout_rate``, ``in_channels``, etc.).

    Returns
    -------
    DenseNet
        Backbone with the DenseNet-121 configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See Huang et al., "Densely Connected Convolutional Networks",
    CVPR 2017, Table 1.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.densenet import densenet_121
    >>> model = densenet_121()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape   # (B, 1024, 1, 1)
    (1, 1024, 1, 1)
    """
    if pretrained:
        reject_unavailable_pretrained("densenet_121", alternative="densenet_121_cls")
    return DenseNet(apply_overrides(_CFG_121, overrides))


@register_model(
    task="base",
    family="densenet",
    model_type="densenet",
    model_class=DenseNet,
    default_config=_CFG_169,
    params=12_484_480,
)
def densenet_169(pretrained: bool = False, **overrides: object) -> DenseNet:
    r"""DenseNet-169 feature-extracting backbone (no classification head).

    Builds a :class:`DenseNet` with the paper-cited DenseNet-169
    topology: per-block dense-layer counts ``(6, 12, 32, 32)``, growth
    rate :math:`k = 32`, initial conv channels 64.  Approximately
    14.3 M parameters total.  Reaches a top-1 ImageNet error of 23.8%
    (Huang et al., 2017, Table 3).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``densenet_169``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``densenet_169_cls`` is the checkpointed variant.
    **overrides
        Keyword overrides forwarded into :class:`DenseNetConfig`.

    Returns
    -------
    DenseNet
        Backbone with the DenseNet-169 configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See Huang et al., CVPR 2017, Table 1.  Final pre-classifier feature
    width is 1664 channels.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.densenet import densenet_169
    >>> model = densenet_169()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape   # (B, 1664, 1, 1)
    (1, 1664, 1, 1)
    """
    if pretrained:
        reject_unavailable_pretrained("densenet_169", alternative="densenet_169_cls")
    return DenseNet(apply_overrides(_CFG_169, overrides))


@register_model(
    task="base",
    family="densenet",
    model_type="densenet",
    model_class=DenseNet,
    default_config=_CFG_201,
    params=18_092_928,
)
def densenet_201(pretrained: bool = False, **overrides: object) -> DenseNet:
    r"""DenseNet-201 feature-extracting backbone (no classification head).

    Builds a :class:`DenseNet` with the paper-cited DenseNet-201
    topology: per-block dense-layer counts ``(6, 12, 48, 32)``, growth
    rate :math:`k = 32`, initial conv channels 64.  Approximately
    20.0 M parameters total.  Reaches a top-1 ImageNet error of 22.6%
    (Huang et al., 2017, Table 3).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``densenet_201``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``densenet_201_cls`` is the checkpointed variant.
    **overrides
        Keyword overrides forwarded into :class:`DenseNetConfig`.

    Returns
    -------
    DenseNet
        Backbone with the DenseNet-201 configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See Huang et al., CVPR 2017, Table 1.  Final pre-classifier feature
    width is 1920 channels.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.densenet import densenet_201
    >>> model = densenet_201()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape   # (B, 1920, 1, 1)
    (1, 1920, 1, 1)
    """
    if pretrained:
        reject_unavailable_pretrained("densenet_201", alternative="densenet_201_cls")
    return DenseNet(apply_overrides(_CFG_201, overrides))


@register_model(
    task="base",
    family="densenet",
    model_type="densenet",
    model_class=DenseNet,
    default_config=_CFG_264,
    params=30_648_704,
)
def densenet_264(pretrained: bool = False, **overrides: object) -> DenseNet:
    r"""DenseNet-264 feature-extracting backbone (no classification head).

    Builds a :class:`DenseNet` with the paper-cited DenseNet-264
    topology: per-block dense-layer counts ``(6, 12, 64, 48)``, growth
    rate :math:`k = 32`, initial conv channels 64.  Approximately
    33.3 M parameters total — the deepest DenseNet evaluated in the
    paper, reaching a top-1 ImageNet error of 22.2% (Huang et al.,
    2017, Table 3).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``densenet_264``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides
        Keyword overrides forwarded into :class:`DenseNetConfig`.

    Returns
    -------
    DenseNet
        Backbone with the DenseNet-264 configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See Huang et al., CVPR 2017, Table 1.  Final pre-classifier feature
    width is 2688 channels.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.densenet import densenet_264
    >>> model = densenet_264()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape   # (B, 2688, 1, 1)
    (1, 2688, 1, 1)
    """
    if pretrained:
        reject_unavailable_pretrained("densenet_264")
    return DenseNet(apply_overrides(_CFG_264, overrides))


# ── Classifiers ───────────────────────────────────────────────────────────────


@register_model(
    task="image-classification",
    family="densenet",
    model_type="densenet",
    model_class=DenseNetForImageClassification,
    default_config=_CFG_121,
    params=7_978_856,
)
def densenet_121_cls(
    pretrained: bool | str = False,
    *,
    weights: DenseNet121Weights | None = None,
    **overrides: object,
) -> DenseNetForImageClassification:
    r"""DenseNet-121 image classifier (backbone + GAP + linear head).

    Builds a :class:`DenseNetForImageClassification` with the
    paper-cited DenseNet-121 topology and a single
    :class:`~lucid.nn.Linear` classifier projecting 1024 →
    ``config.num_classes``.  Approximately 8.0 M parameters; top-1
    ImageNet accuracy 74.43% (reference_vision weights).

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        ``False`` → random init; ``True`` → ``DEFAULT`` tag; a tag
        string → that checkpoint.
    weights : DenseNet121Weights, optional, keyword-only
        Explicit enum member; takes precedence over ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`DenseNetConfig`.

    Returns
    -------
    DenseNetForImageClassification
        Classifier with the DenseNet-121 configuration applied (or with
        ``overrides`` merged on top of it), optionally initialised from
        pretrained weights.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.densenet import densenet_121_cls
    >>> model = densenet_121_cls(num_classes=100)
    >>> x = lucid.randn(2, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (2, 100)

    Load ImageNet-pretrained weights:

    >>> model = densenet_121_cls(pretrained=True)
    >>> from lucid.models.weights import DenseNet121Weights
    >>> model = densenet_121_cls(weights=DenseNet121Weights.IMAGENET1K_V1)
    """
    entry = weights_mod.resolve_weights(DenseNet121Weights, pretrained, weights)
    model = DenseNetForImageClassification(apply_overrides(_CFG_121, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="densenet_121_cls")
    return model


@register_model(
    task="image-classification",
    family="densenet",
    model_type="densenet",
    model_class=DenseNetForImageClassification,
    default_config=_CFG_169,
    params=14_149_480,
)
def densenet_169_cls(
    pretrained: bool | str = False,
    *,
    weights: DenseNet169Weights | None = None,
    **overrides: object,
) -> DenseNetForImageClassification:
    r"""DenseNet-169 image classifier (backbone + GAP + linear head).

    Builds a :class:`DenseNetForImageClassification` with the
    paper-cited DenseNet-169 topology and a :class:`~lucid.nn.Linear`
    classifier projecting 1664 → ``config.num_classes``.
    Approximately 14.3 M parameters; top-1 ImageNet accuracy 75.60%
    (reference_vision weights).

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        ``False`` → random init; ``True`` → ``DEFAULT`` tag; a tag
        string → that checkpoint.
    weights : DenseNet169Weights, optional, keyword-only
        Explicit enum member; takes precedence over ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`DenseNetConfig`.

    Returns
    -------
    DenseNetForImageClassification
        Classifier with the DenseNet-169 configuration applied (or with
        ``overrides`` merged on top of it).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.densenet import densenet_169_cls
    >>> model = densenet_169_cls(pretrained=True)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(DenseNet169Weights, pretrained, weights)
    model = DenseNetForImageClassification(apply_overrides(_CFG_169, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="densenet_169_cls")
    return model


@register_model(
    task="image-classification",
    family="densenet",
    model_type="densenet",
    model_class=DenseNetForImageClassification,
    default_config=_CFG_201,
    params=20_013_928,
)
def densenet_201_cls(
    pretrained: bool | str = False,
    *,
    weights: DenseNet201Weights | None = None,
    **overrides: object,
) -> DenseNetForImageClassification:
    r"""DenseNet-201 image classifier (backbone + GAP + linear head).

    Builds a :class:`DenseNetForImageClassification` with the
    paper-cited DenseNet-201 topology and a :class:`~lucid.nn.Linear`
    classifier projecting 1920 → ``config.num_classes``.
    Approximately 20.0 M parameters; top-1 ImageNet accuracy 76.90%
    (reference_vision weights).

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        ``False`` → random init; ``True`` → ``DEFAULT`` tag; a tag
        string → that checkpoint.
    weights : DenseNet201Weights, optional, keyword-only
        Explicit enum member; takes precedence over ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`DenseNetConfig`.

    Returns
    -------
    DenseNetForImageClassification
        Classifier with the DenseNet-201 configuration applied (or with
        ``overrides`` merged on top of it).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.densenet import densenet_201_cls
    >>> model = densenet_201_cls(pretrained=True)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(DenseNet201Weights, pretrained, weights)
    model = DenseNetForImageClassification(apply_overrides(_CFG_201, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="densenet_201_cls")
    return model


@register_model(
    task="image-classification",
    family="densenet",
    model_type="densenet",
    model_class=DenseNetForImageClassification,
    default_config=_CFG_264,
    params=33_337_704,
)
def densenet_264_cls(
    pretrained: bool = False, **overrides: object
) -> DenseNetForImageClassification:
    r"""DenseNet-264 image classifier (backbone + GAP + linear head).

    Builds a :class:`DenseNetForImageClassification` with the
    paper-cited DenseNet-264 topology and a :class:`~lucid.nn.Linear`
    classifier projecting 2688 → ``config.num_classes``.
    Approximately 33.3 M parameters; the deepest DenseNet evaluated in
    the paper, with a top-1 ImageNet error of 22.2% (Huang et al.,
    2017, Table 3).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``densenet_264_cls``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides
        Keyword overrides forwarded into :class:`DenseNetConfig`.

    Returns
    -------
    DenseNetForImageClassification
        Classifier with the DenseNet-264 configuration applied (or with
        ``overrides`` merged on top of it).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.densenet import densenet_264_cls
    >>> model = densenet_264_cls()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    if pretrained:
        reject_unavailable_pretrained("densenet_264_cls")
    return DenseNetForImageClassification(apply_overrides(_CFG_264, overrides))


# ── DenseNet-161 (k=48, 96-channel stem) ────────────────────────────────────────


@register_model(
    task="base",
    family="densenet",
    model_type="densenet",
    model_class=DenseNet,
    default_config=_CFG_161,
    params=26_472_000,
)
def densenet_161(pretrained: bool = False, **overrides: object) -> DenseNet:
    r"""DenseNet-161 feature-extracting backbone (no classification head).

    Builds a :class:`DenseNet` with the DenseNet-161 topology: per-block
    dense-layer counts ``(6, 12, 36, 24)``, growth rate :math:`k = 48`,
    initial conv channels 96.  Approximately 28.7 M parameters — the
    widest of the ImageNet DenseNets.  Reaches a top-1 ImageNet accuracy
    of 77.14%.

    This variant is *not* in Huang et al., CVPR 2017 — that paper's
    Table 1 lists 121/169/201/264 at :math:`k = 32` only.  It comes from
    the authors' own released reference-framework models, which the
    reference vision library mirrors.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``densenet_161``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``densenet_161_cls`` is the checkpointed variant.
    **overrides
        Keyword overrides forwarded into :class:`DenseNetConfig`.

    Returns
    -------
    DenseNet
        Backbone with the DenseNet-161 configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See Huang et al., "Densely Connected Convolutional Networks",
    CVPR 2017, Table 1.  Unlike the k=32 / 64-stem siblings,
    DenseNet-161 widens the growth rate to 48 and the stem to 96
    channels, giving the highest accuracy of the four canonical
    ImageNet variants at the cost of ~3.6× the parameters of
    DenseNet-121.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.densenet import densenet_161
    >>> model = densenet_161()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape   # (B, 2208, 1, 1)
    (1, 2208, 1, 1)
    """
    if pretrained:
        reject_unavailable_pretrained("densenet_161", alternative="densenet_161_cls")
    return DenseNet(apply_overrides(_CFG_161, overrides))


@register_model(
    task="image-classification",
    family="densenet",
    model_type="densenet",
    model_class=DenseNetForImageClassification,
    default_config=_CFG_161,
    params=28_681_000,
)
def densenet_161_cls(
    pretrained: bool | str = False,
    *,
    weights: DenseNet161Weights | None = None,
    **overrides: object,
) -> DenseNetForImageClassification:
    r"""DenseNet-161 image classifier (backbone + GAP + linear head).

    Builds a :class:`DenseNetForImageClassification` with the
    DenseNet-161 topology (``k=48``, 96-channel stem) and a
    :class:`~lucid.nn.Linear` classifier projecting 2208 →
    ``config.num_classes``.  Approximately 28.7 M parameters; top-1
    ImageNet accuracy 77.14% — the most accurate DenseNet shipped here.

    The topology is *not* in Huang et al., CVPR 2017 (Table 1 lists
    121/169/201/264 at :math:`k = 32`); it comes from the authors' own
    released reference-framework models.

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        ``False`` → random init; ``True`` → ``DEFAULT`` tag; a tag
        string → that checkpoint.
    weights : DenseNet161Weights, optional, keyword-only
        Explicit enum member; takes precedence over ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`DenseNetConfig`.

    Returns
    -------
    DenseNetForImageClassification
        Classifier with the DenseNet-161 configuration applied (or with
        ``overrides`` merged on top of it).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.densenet import densenet_161_cls
    >>> model = densenet_161_cls(pretrained=True)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(DenseNet161Weights, pretrained, weights)
    model = DenseNetForImageClassification(apply_overrides(_CFG_161, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="densenet_161_cls")
    return model
