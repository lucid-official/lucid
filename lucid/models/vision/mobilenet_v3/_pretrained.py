"""Registry factories for MobileNet v3."""

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models.vision.mobilenet_v3._config import MobileNetV3Config
from lucid.models.vision.mobilenet_v3._model import (
    MobileNetV3,
    MobileNetV3ForImageClassification,
)
from lucid.models.vision.mobilenet_v3._weights import (
    MobileNetV3LargeWeights,
    MobileNetV3SmallWeights,
)
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained

_CFG_LARGE = MobileNetV3Config(variant="large")
_CFG_SMALL = MobileNetV3Config(variant="small")


# ── Backbones ─────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="mobilenet_v3",
    model_type="mobilenet_v3",
    model_class=MobileNetV3,
    default_config=_CFG_LARGE,
    params=2971952,
)
def mobilenet_v3_large(pretrained: bool = False, **overrides: object) -> MobileNetV3:
    r"""MobileNet-v3-Large feature-extracting backbone.

    Builds a :class:`MobileNetV3` with the NAS-designed Large
    topology from Howard et al., 2019 (Table 1): 15 inverted-residual
    bottleneck blocks with selective SE attention and hard-swish in
    the deeper half, followed by a 1×1 expansion to 960 channels.
    Approximately 5.4M parameters and 75.2% ImageNet-1k top-1
    accuracy (Table 3) — the higher-accuracy variant of the family.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``mobilenet_v3_large``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``mobilenet_v3_large_cls`` is the checkpointed variant.
    **overrides
        Keyword overrides forwarded into :class:`MobileNetV3Config`
        (e.g. ``width_mult=0.75`` for a slimmer variant).

    Returns
    -------
    MobileNetV3
        Backbone with the MobileNet-v3-Large configuration applied
        (or with ``overrides`` merged on top of it).

    Notes
    -----
    See Howard et al., "Searching for MobileNetV3", ICCV 2019
    (arXiv:1905.02244), Table 1 (V3-Large spec) and Table 3
    (ImageNet accuracy).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mobilenet_v3 import mobilenet_v3_large
    >>> model = mobilenet_v3_large()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 960, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained("mobilenet_v3_large", alternative="mobilenet_v3_large_cls")
    cfg = apply_overrides(_CFG_LARGE, overrides)
    return MobileNetV3(cfg)


@register_model(
    task="base",
    family="mobilenet_v3",
    model_type="mobilenet_v3",
    model_class=MobileNetV3,
    default_config=_CFG_SMALL,
    params=927008,
)
def mobilenet_v3_small(pretrained: bool = False, **overrides: object) -> MobileNetV3:
    r"""MobileNet-v3-Small feature-extracting backbone.

    Builds a :class:`MobileNetV3` with the NAS-designed Small
    topology from Howard et al., 2019 (Table 2): 11 inverted-residual
    bottleneck blocks with selective SE attention and aggressive
    hard-swish usage from the very first stage, followed by a 1×1
    expansion to 576 channels.  927k parameters headless (2.5M with
    the classification head, which is Table 3's figure) and 67.4%
    ImageNet-1k top-1 accuracy (Table 3) — the lower-latency
    variant of the family, targeting tight mobile budgets.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``mobilenet_v3_small``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``mobilenet_v3_small_cls`` is the checkpointed variant.
    **overrides
        Keyword overrides forwarded into :class:`MobileNetV3Config`.

    Returns
    -------
    MobileNetV3
        Backbone with the MobileNet-v3-Small configuration applied
        (or with ``overrides`` merged on top of it).

    Notes
    -----
    See Howard et al., "Searching for MobileNetV3", ICCV 2019
    (arXiv:1905.02244), Table 2 (V3-Small spec) and Table 3
    (ImageNet accuracy).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mobilenet_v3 import mobilenet_v3_small
    >>> model = mobilenet_v3_small()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 576, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained("mobilenet_v3_small", alternative="mobilenet_v3_small_cls")
    cfg = apply_overrides(_CFG_SMALL, overrides)
    return MobileNetV3(cfg)


# ── Classifiers ───────────────────────────────────────────────────────────────


@register_model(
    task="image-classification",
    family="mobilenet_v3",
    model_type="mobilenet_v3",
    model_class=MobileNetV3ForImageClassification,
    default_config=_CFG_LARGE,
    params=5483032,
)
def mobilenet_v3_large_cls(
    pretrained: bool | str = False,
    *,
    weights: MobileNetV3LargeWeights | None = None,
    **overrides: object,
) -> MobileNetV3ForImageClassification:
    r"""MobileNet-v3-Large image classifier with the redesigned head.

    Builds a :class:`MobileNetV3ForImageClassification` with the
    Large topology (Howard et al., 2019) plus the redesigned
    inverted classification head (``GAP → 1×1 Conv → h-swish →
    Flatten → Dropout → Linear``).  Approximately 5.4M parameters
    and 75.2% ImageNet-1k top-1 accuracy (Table 3).

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag
        (:attr:`MobileNetV3LargeWeights.IMAGENET1K_V1`); a tag string →
        that specific checkpoint.  Mutually exclusive with ``weights``
        (which wins if both are given).
    weights : MobileNetV3LargeWeights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`MobileNetV3Config`
        (typically ``num_classes`` to retarget the classifier).

    Returns
    -------
    MobileNetV3ForImageClassification
        Classifier with the MobileNet-v3-Large configuration applied
        (or with ``overrides`` merged on top of it), optionally
        initialised from pretrained weights.

    Notes
    -----
    See Howard et al., "Searching for MobileNetV3", ICCV 2019
    (arXiv:1905.02244), Table 1 and Table 3.  Pretrained weights are
    converted from reference_vision's ``MobileNet_V3_Large_Weights`` and
    hosted on the Hugging Face Hub under ``lucid-dl/mobilenet-v3-large``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mobilenet_v3 import mobilenet_v3_large_cls
    >>> model = mobilenet_v3_large_cls(num_classes=10)
    >>> x = lucid.randn(2, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (2, 10)

    Load ImageNet-pretrained weights:

    >>> model = mobilenet_v3_large_cls(pretrained=True)
    >>> from lucid.models.weights import MobileNetV3LargeWeights
    >>> model = mobilenet_v3_large_cls(weights=MobileNetV3LargeWeights.IMAGENET1K_V1)
    """
    entry = weights_mod.resolve_weights(MobileNetV3LargeWeights, pretrained, weights)
    cfg = apply_overrides(_CFG_LARGE, overrides)
    model = MobileNetV3ForImageClassification(cfg)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="mobilenet_v3_large_cls")
    return model


@register_model(
    task="image-classification",
    family="mobilenet_v3",
    model_type="mobilenet_v3",
    model_class=MobileNetV3ForImageClassification,
    default_config=_CFG_SMALL,
    params=2542856,
)
def mobilenet_v3_small_cls(
    pretrained: bool | str = False,
    *,
    weights: MobileNetV3SmallWeights | None = None,
    **overrides: object,
) -> MobileNetV3ForImageClassification:
    r"""MobileNet-v3-Small image classifier with the redesigned head.

    Builds a :class:`MobileNetV3ForImageClassification` with the
    Small topology (Howard et al., 2019) plus the redesigned
    inverted classification head.  2.5M parameters, matching
    Table 3, and 67.4% ImageNet-1k top-1 accuracy — the
    latency-focused variant of the family.

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag
        (:attr:`MobileNetV3SmallWeights.IMAGENET1K_V1`); a tag string →
        that specific checkpoint.  Mutually exclusive with ``weights``
        (which wins if both are given).
    weights : MobileNetV3SmallWeights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`MobileNetV3Config`.

    Returns
    -------
    MobileNetV3ForImageClassification
        Classifier with the MobileNet-v3-Small configuration applied
        (or with ``overrides`` merged on top of it), optionally
        initialised from pretrained weights.

    Notes
    -----
    See Howard et al., "Searching for MobileNetV3", ICCV 2019
    (arXiv:1905.02244), Table 2 and Table 3.  Pretrained weights are
    converted from reference_vision's ``MobileNet_V3_Small_Weights`` and
    hosted on the Hugging Face Hub under ``lucid-dl/mobilenet-v3-small``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mobilenet_v3 import mobilenet_v3_small_cls
    >>> model = mobilenet_v3_small_cls()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)

    Load ImageNet-pretrained weights:

    >>> model = mobilenet_v3_small_cls(pretrained=True)
    >>> from lucid.models.weights import MobileNetV3SmallWeights
    >>> model = mobilenet_v3_small_cls(weights=MobileNetV3SmallWeights.IMAGENET1K_V1)
    """
    entry = weights_mod.resolve_weights(MobileNetV3SmallWeights, pretrained, weights)
    cfg = apply_overrides(_CFG_SMALL, overrides)
    model = MobileNetV3ForImageClassification(cfg)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="mobilenet_v3_small_cls")
    return model
