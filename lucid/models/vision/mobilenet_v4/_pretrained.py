"""Registry factories for MobileNet v4 (Qin et al., 2024).

The five variants are the paper's Appendix D architectures (Tables 11-15);
their ImageNet-1k numbers are Table 6.  ``params`` on each registration is
the exact parameter count of the model built here, which equals the
authors' released implementation; the paper rounds the same counts.

The ``*_cls`` factories load ImageNet-1k checkpoints converted from timm
(see ``_weights.py`` for the tag chosen per variant and its preset); the
headless backbones have no checkpoint of their own and refuse
``pretrained=True``.
"""

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained
from lucid.models.vision.mobilenet_v4._config import MobileNetV4Config
from lucid.models.vision.mobilenet_v4._model import (
    MobileNetV4,
    MobileNetV4ForImageClassification,
)
from lucid.models.vision.mobilenet_v4._weights import (
    MobileNetV4ConvLargeWeights,
    MobileNetV4ConvMediumWeights,
    MobileNetV4ConvSmallWeights,
    MobileNetV4HybridLargeWeights,
    MobileNetV4HybridMediumWeights,
)

_CFG_CONV_SMALL = MobileNetV4Config(variant="conv_small", dropout=0.3)
_CFG_CONV_MEDIUM = MobileNetV4Config(variant="conv_medium", dropout=0.2)
_CFG_CONV_LARGE = MobileNetV4Config(variant="conv_large", dropout=0.2)
_CFG_HYBRID_MEDIUM = MobileNetV4Config(variant="hybrid_medium", dropout=0.2)
_CFG_HYBRID_LARGE = MobileNetV4Config(variant="hybrid_large", dropout=0.2)


# ── Backbones ─────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="mobilenet_v4",
    model_type="mobilenet_v4",
    model_class=MobileNetV4,
    default_config=_CFG_CONV_SMALL,
    params=1_261_664,
    summary="auto",
)
def mobilenet_v4_conv_small(pretrained: bool = False, **overrides: object) -> MobileNetV4:
    r"""MobileNet-v4-Conv-Small feature-extracting backbone.

    Builds a :class:`MobileNetV4` with the Conv-Small architecture of
    Qin et al., 2024 (Appendix D, Table 11): the two stem stages are dense conv pairs, the next two are built from ExtraDW, IB and ConvNeXt-like UIBs, closed by a
    :math:`1 \times 1` widening to 960 channels.  1.26 M
    parameters without the classification head.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Must stay ``False`` — no checkpoint is published for the headless
        backbone, and ``True`` raises rather than returning random
        weights.  The ImageNet-1k checkpoint belongs to the matching
        ``*_cls`` classifier, whose error message this names.
    **overrides
        Keyword overrides merged into :class:`MobileNetV4Config`
        (e.g. ``drop_path_rate=0.1``).

    Returns
    -------
    MobileNetV4
        Backbone returning the stride-32, 960-channel feature map.

    Raises
    ------
    NotImplementedError
        If ``pretrained=True``.

    Notes
    -----
    Qin et al., "MobileNetV4: Universal Models for the Mobile Ecosystem",
    ECCV 2024 (arXiv:2404.10518).  The paper trains and evaluates this
    variant at 224x224; the network is fully convolutional up to the
    pool, so any input whose sides are multiples of 32 works.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mobilenet_v4 import mobilenet_v4_conv_small
    >>> model = mobilenet_v4_conv_small().eval()
    >>> model(lucid.randn(1, 3, 224, 224)).last_hidden_state.shape
    (1, 960, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained(
            "mobilenet_v4_conv_small", alternative="mobilenet_v4_conv_small_cls"
        )
    return MobileNetV4(apply_overrides(_CFG_CONV_SMALL, overrides))


@register_model(
    task="base",
    family="mobilenet_v4",
    model_type="mobilenet_v4",
    model_class=MobileNetV4,
    default_config=_CFG_CONV_MEDIUM,
    params=7_203_152,
    summary="auto",
)
def mobilenet_v4_conv_medium(pretrained: bool = False, **overrides: object) -> MobileNetV4:
    r"""MobileNet-v4-Conv-Medium feature-extracting backbone.

    Builds a :class:`MobileNetV4` with the Conv-Medium architecture of
    Qin et al., 2024 (Appendix D, Table 12): a fused-IB stem stage followed by ExtraDW-led stages that mix in ConvNeXt-like and FFN blocks, closed by a
    :math:`1 \times 1` widening to 960 channels.  7.20 M
    parameters without the classification head.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Must stay ``False`` — no checkpoint is published for the headless
        backbone, and ``True`` raises rather than returning random
        weights.  The ImageNet-1k checkpoint belongs to the matching
        ``*_cls`` classifier, whose error message this names.
    **overrides
        Keyword overrides merged into :class:`MobileNetV4Config`
        (e.g. ``drop_path_rate=0.1``).

    Returns
    -------
    MobileNetV4
        Backbone returning the stride-32, 960-channel feature map.

    Raises
    ------
    NotImplementedError
        If ``pretrained=True``.

    Notes
    -----
    Qin et al., "MobileNetV4: Universal Models for the Mobile Ecosystem",
    ECCV 2024 (arXiv:2404.10518).  The paper trains and evaluates this
    variant at 256x256; the network is fully convolutional up to the
    pool, so any input whose sides are multiples of 32 works.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mobilenet_v4 import mobilenet_v4_conv_medium
    >>> model = mobilenet_v4_conv_medium().eval()
    >>> model(lucid.randn(1, 3, 224, 224)).last_hidden_state.shape
    (1, 960, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained(
            "mobilenet_v4_conv_medium", alternative="mobilenet_v4_conv_medium_cls"
        )
    return MobileNetV4(apply_overrides(_CFG_CONV_MEDIUM, overrides))


@register_model(
    task="base",
    family="mobilenet_v4",
    model_type="mobilenet_v4",
    model_class=MobileNetV4,
    default_config=_CFG_CONV_LARGE,
    params=30_078_504,
    summary="auto",
)
def mobilenet_v4_conv_large(pretrained: bool = False, **overrides: object) -> MobileNetV4:
    r"""MobileNet-v4-Conv-Large feature-extracting backbone.

    Builds a :class:`MobileNetV4` with the Conv-Large architecture of
    Qin et al., 2024 (Appendix D, Table 14): a 24-channel stem, a fused-IB stage, and two deep stages of ExtraDW and ConvNeXt-like blocks up to 512 channels, closed by a
    :math:`1 \times 1` widening to 960 channels.  30.08 M
    parameters without the classification head.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Must stay ``False`` — no checkpoint is published for the headless
        backbone, and ``True`` raises rather than returning random
        weights.  The ImageNet-1k checkpoint belongs to the matching
        ``*_cls`` classifier, whose error message this names.
    **overrides
        Keyword overrides merged into :class:`MobileNetV4Config`
        (e.g. ``drop_path_rate=0.1``).

    Returns
    -------
    MobileNetV4
        Backbone returning the stride-32, 960-channel feature map.

    Raises
    ------
    NotImplementedError
        If ``pretrained=True``.

    Notes
    -----
    Qin et al., "MobileNetV4: Universal Models for the Mobile Ecosystem",
    ECCV 2024 (arXiv:2404.10518).  The paper trains and evaluates this
    variant at 384x384; the network is fully convolutional up to the
    pool, so any input whose sides are multiples of 32 works.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mobilenet_v4 import mobilenet_v4_conv_large
    >>> model = mobilenet_v4_conv_large().eval()
    >>> model(lucid.randn(1, 3, 224, 224)).last_hidden_state.shape
    (1, 960, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained(
            "mobilenet_v4_conv_large", alternative="mobilenet_v4_conv_large_cls"
        )
    return MobileNetV4(apply_overrides(_CFG_CONV_LARGE, overrides))


@register_model(
    task="base",
    family="mobilenet_v4",
    model_type="mobilenet_v4",
    model_class=MobileNetV4,
    default_config=_CFG_HYBRID_MEDIUM,
    params=8_562_288,
    summary="auto",
)
def mobilenet_v4_hybrid_medium(pretrained: bool = False, **overrides: object) -> MobileNetV4:
    r"""MobileNet-v4-Hybrid-Medium feature-extracting backbone.

    Builds a :class:`MobileNetV4` with the Hybrid-Medium architecture of
    Qin et al., 2024 (Appendix D, Table 13): a layout close to Conv-Medium's with four Mobile MQA blocks (4 heads of width 64) interleaved into each of the two deepest stages, keys and values reduced by a stride-2 depthwise conv in the first of them, closed by a
    :math:`1 \times 1` widening to 960 channels.  8.56 M
    parameters without the classification head.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Must stay ``False`` — no checkpoint is published for the headless
        backbone, and ``True`` raises rather than returning random
        weights.  The ImageNet-1k checkpoint belongs to the matching
        ``*_cls`` classifier, whose error message this names.
    **overrides
        Keyword overrides merged into :class:`MobileNetV4Config`
        (e.g. ``drop_path_rate=0.1``).

    Returns
    -------
    MobileNetV4
        Backbone returning the stride-32, 960-channel feature map.

    Raises
    ------
    NotImplementedError
        If ``pretrained=True``.

    Notes
    -----
    Qin et al., "MobileNetV4: Universal Models for the Mobile Ecosystem",
    ECCV 2024 (arXiv:2404.10518).  The paper trains and evaluates this
    variant at 256x256; the network is fully convolutional up to the
    pool, so any input whose sides are multiples of 32 works.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mobilenet_v4 import mobilenet_v4_hybrid_medium
    >>> model = mobilenet_v4_hybrid_medium().eval()
    >>> model(lucid.randn(1, 3, 224, 224)).last_hidden_state.shape
    (1, 960, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained(
            "mobilenet_v4_hybrid_medium", alternative="mobilenet_v4_hybrid_medium_cls"
        )
    return MobileNetV4(apply_overrides(_CFG_HYBRID_MEDIUM, overrides))


@register_model(
    task="base",
    family="mobilenet_v4",
    model_type="mobilenet_v4",
    model_class=MobileNetV4,
    default_config=_CFG_HYBRID_LARGE,
    params=35_252_264,
    summary="auto",
)
def mobilenet_v4_hybrid_large(pretrained: bool = False, **overrides: object) -> MobileNetV4:
    r"""MobileNet-v4-Hybrid-Large feature-extracting backbone.

    Builds a :class:`MobileNetV4` with the Hybrid-Large architecture of
    Qin et al., 2024 (Appendix D, Table 15): Conv-Large's layout with four Mobile MQA blocks (8 heads) interleaved into each of the two deepest stages and GELU activations throughout, closed by a
    :math:`1 \times 1` widening to 960 channels.  35.25 M
    parameters without the classification head.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Must stay ``False`` — no checkpoint is published for the headless
        backbone, and ``True`` raises rather than returning random
        weights.  The ImageNet-1k checkpoint belongs to the matching
        ``*_cls`` classifier, whose error message this names.
    **overrides
        Keyword overrides merged into :class:`MobileNetV4Config`
        (e.g. ``drop_path_rate=0.1``).

    Returns
    -------
    MobileNetV4
        Backbone returning the stride-32, 960-channel feature map.

    Raises
    ------
    NotImplementedError
        If ``pretrained=True``.

    Notes
    -----
    Qin et al., "MobileNetV4: Universal Models for the Mobile Ecosystem",
    ECCV 2024 (arXiv:2404.10518).  The paper trains and evaluates this
    variant at 384x384; the network is fully convolutional up to the
    pool, so any input whose sides are multiples of 32 works.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mobilenet_v4 import mobilenet_v4_hybrid_large
    >>> model = mobilenet_v4_hybrid_large().eval()
    >>> model(lucid.randn(1, 3, 224, 224)).last_hidden_state.shape
    (1, 960, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained(
            "mobilenet_v4_hybrid_large", alternative="mobilenet_v4_hybrid_large_cls"
        )
    return MobileNetV4(apply_overrides(_CFG_HYBRID_LARGE, overrides))


# ── Classifiers ───────────────────────────────────────────────────────────────


@register_model(
    task="image-classification",
    family="mobilenet_v4",
    model_type="mobilenet_v4",
    model_class=MobileNetV4ForImageClassification,
    default_config=_CFG_CONV_SMALL,
    params=3_774_024,
    summary="auto",
)
def mobilenet_v4_conv_small_cls(
    pretrained: bool | str = False,
    *,
    weights: MobileNetV4ConvSmallWeights | None = None,
    **overrides: object,
) -> MobileNetV4ForImageClassification:
    r"""MobileNet-v4-Conv-Small image classifier.

    Builds a :class:`MobileNetV4ForImageClassification` with the Conv-Small
    architecture (Qin et al., 2024, Appendix D, Table 11) and the
    post-pool head (960 → 1280 → classes).  3.77 M parameters;
    Table 6 reports 73.8% ImageNet-1k top-1 at 3.8 M parameters and
    0.2 G MACs.  Classifier dropout defaults to the paper's 0.3
    (Table 10).

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True`` → the
        ``DEFAULT`` tag
        (:attr:`MobileNetV4ConvSmallWeights.E2400_R224_IN1K`); a tag string
        (e.g. ``"E2400_R224_IN1K"``) → that specific checkpoint. Mutually
        exclusive with ``weights`` (which wins if both are given).
    weights : MobileNetV4ConvSmallWeights, optional, keyword-only
        Explicit weights enum member, e.g.
        ``MobileNetV4ConvSmallWeights.E2400_R224_IN1K``.  Takes precedence
        over ``pretrained``.
    **overrides
        Keyword overrides merged into :class:`MobileNetV4Config`
        (typically ``num_classes`` to retarget the head).  Overriding
        ``num_classes`` away from the checkpoint's 1000 makes pretrained
        loading fail the strict key/shape check — load with the
        matching head, then call :meth:`reset_classifier`.

    Returns
    -------
    MobileNetV4ForImageClassification
        Classifier with the Conv-Small configuration (plus ``overrides``),
        optionally initialised from pretrained weights.

    Notes
    -----
    Qin et al., "MobileNetV4: Universal Models for the Mobile Ecosystem",
    ECCV 2024 (arXiv:2404.10518).  Parameter names match timm's
    implementation, so its ImageNet-1k checkpoints load with an identity
    key map.  The paper evaluates this variant at 224x224.

    Pretrained weights are converted from timm's
    ``mobilenetv4_conv_small.e2400_r224_in1k`` — timm's own ImageNet-1k
    training of this architecture, trained at 224x224; the authors released
    none — and hosted on the Hugging Face Hub under
    ``lucid-dl/mobilenet-v4-conv-small``.  timm reports 73.756% top-1 /
    91.430% top-5 for it at 224x224 with the preset that
    :meth:`MobileNetV4ConvSmallWeights.E2400_R224_IN1K.transforms`
    reproduces (224 crop, 256 resize, bicubic, ImageNet mean/std).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mobilenet_v4 import mobilenet_v4_conv_small_cls
    >>> model = mobilenet_v4_conv_small_cls(num_classes=10).eval()
    >>> model(lucid.randn(2, 3, 224, 224)).logits.shape
    (2, 10)

    Load ImageNet-pretrained weights:

    >>> model = mobilenet_v4_conv_small_cls(pretrained=True)  # doctest: +SKIP
    >>> from lucid.models.weights import MobileNetV4ConvSmallWeights
    >>> model = mobilenet_v4_conv_small_cls(  # doctest: +SKIP
    ...     weights=MobileNetV4ConvSmallWeights.E2400_R224_IN1K
    ... )
    """
    entry = weights_mod.resolve_weights(MobileNetV4ConvSmallWeights, pretrained, weights)
    model = MobileNetV4ForImageClassification(apply_overrides(_CFG_CONV_SMALL, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="mobilenet_v4_conv_small_cls")
    return model


@register_model(
    task="image-classification",
    family="mobilenet_v4",
    model_type="mobilenet_v4",
    model_class=MobileNetV4ForImageClassification,
    default_config=_CFG_CONV_MEDIUM,
    params=9_715_512,
    summary="auto",
)
def mobilenet_v4_conv_medium_cls(
    pretrained: bool | str = False,
    *,
    weights: MobileNetV4ConvMediumWeights | None = None,
    **overrides: object,
) -> MobileNetV4ForImageClassification:
    r"""MobileNet-v4-Conv-Medium image classifier.

    Builds a :class:`MobileNetV4ForImageClassification` with the Conv-Medium
    architecture (Qin et al., 2024, Appendix D, Table 12) and the
    post-pool head (960 → 1280 → classes).  9.72 M parameters;
    Table 6 reports 79.9% ImageNet-1k top-1 at 9.2 M parameters and
    1.0 G MACs.  Classifier dropout defaults to the paper's 0.2
    (Table 10).

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True`` → the
        ``DEFAULT`` tag
        (:attr:`MobileNetV4ConvMediumWeights.E500_R256_IN1K`); a tag string
        (e.g. ``"E500_R256_IN1K"``) → that specific checkpoint. Mutually
        exclusive with ``weights`` (which wins if both are given).
    weights : MobileNetV4ConvMediumWeights, optional, keyword-only
        Explicit weights enum member, e.g.
        ``MobileNetV4ConvMediumWeights.E500_R256_IN1K``.  Takes precedence
        over ``pretrained``.
    **overrides
        Keyword overrides merged into :class:`MobileNetV4Config`
        (typically ``num_classes`` to retarget the head).  Overriding
        ``num_classes`` away from the checkpoint's 1000 makes pretrained
        loading fail the strict key/shape check — load with the
        matching head, then call :meth:`reset_classifier`.

    Returns
    -------
    MobileNetV4ForImageClassification
        Classifier with the Conv-Medium configuration (plus ``overrides``),
        optionally initialised from pretrained weights.

    Notes
    -----
    Qin et al., "MobileNetV4: Universal Models for the Mobile Ecosystem",
    ECCV 2024 (arXiv:2404.10518).  Parameter names match timm's
    implementation, so its ImageNet-1k checkpoints load with an identity
    key map.  The paper evaluates this variant at 256x256.

    Pretrained weights are converted from timm's
    ``mobilenetv4_conv_medium.e500_r256_in1k`` — timm's own ImageNet-1k
    training of this architecture, trained at 256x256; the authors released
    none — and hosted on the Hugging Face Hub under
    ``lucid-dl/mobilenet-v4-conv-medium``.  timm reports 79.916% top-1 /
    95.188% top-5 for it at 256x256 with the preset that
    :meth:`MobileNetV4ConvMediumWeights.E500_R256_IN1K.transforms`
    reproduces (256 crop, 269 resize, bicubic, ImageNet mean/std).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mobilenet_v4 import mobilenet_v4_conv_medium_cls
    >>> model = mobilenet_v4_conv_medium_cls(num_classes=10).eval()
    >>> model(lucid.randn(2, 3, 224, 224)).logits.shape
    (2, 10)

    Load ImageNet-pretrained weights:

    >>> model = mobilenet_v4_conv_medium_cls(pretrained=True)  # doctest: +SKIP
    >>> from lucid.models.weights import MobileNetV4ConvMediumWeights
    >>> model = mobilenet_v4_conv_medium_cls(  # doctest: +SKIP
    ...     weights=MobileNetV4ConvMediumWeights.E500_R256_IN1K
    ... )
    """
    entry = weights_mod.resolve_weights(MobileNetV4ConvMediumWeights, pretrained, weights)
    model = MobileNetV4ForImageClassification(apply_overrides(_CFG_CONV_MEDIUM, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="mobilenet_v4_conv_medium_cls")
    return model


@register_model(
    task="image-classification",
    family="mobilenet_v4",
    model_type="mobilenet_v4",
    model_class=MobileNetV4ForImageClassification,
    default_config=_CFG_CONV_LARGE,
    params=32_590_864,
    summary="auto",
)
def mobilenet_v4_conv_large_cls(
    pretrained: bool | str = False,
    *,
    weights: MobileNetV4ConvLargeWeights | None = None,
    **overrides: object,
) -> MobileNetV4ForImageClassification:
    r"""MobileNet-v4-Conv-Large image classifier.

    Builds a :class:`MobileNetV4ForImageClassification` with the Conv-Large
    architecture (Qin et al., 2024, Appendix D, Table 14) and the
    post-pool head (960 → 1280 → classes).  32.59 M parameters;
    Table 6 reports 82.9% ImageNet-1k top-1 at 31 M parameters and
    5.9 G MACs.  Classifier dropout defaults to the paper's 0.2
    (Table 10).

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True`` → the
        ``DEFAULT`` tag
        (:attr:`MobileNetV4ConvLargeWeights.E600_R384_IN1K`); a tag string
        (e.g. ``"E600_R384_IN1K"``) → that specific checkpoint. Mutually
        exclusive with ``weights`` (which wins if both are given).
    weights : MobileNetV4ConvLargeWeights, optional, keyword-only
        Explicit weights enum member, e.g.
        ``MobileNetV4ConvLargeWeights.E600_R384_IN1K``.  Takes precedence
        over ``pretrained``.
    **overrides
        Keyword overrides merged into :class:`MobileNetV4Config`
        (typically ``num_classes`` to retarget the head).  Overriding
        ``num_classes`` away from the checkpoint's 1000 makes pretrained
        loading fail the strict key/shape check — load with the
        matching head, then call :meth:`reset_classifier`.

    Returns
    -------
    MobileNetV4ForImageClassification
        Classifier with the Conv-Large configuration (plus ``overrides``),
        optionally initialised from pretrained weights.

    Notes
    -----
    Qin et al., "MobileNetV4: Universal Models for the Mobile Ecosystem",
    ECCV 2024 (arXiv:2404.10518).  Parameter names match timm's
    implementation, so its ImageNet-1k checkpoints load with an identity
    key map.  The paper evaluates this variant at 384x384.

    Pretrained weights are converted from timm's
    ``mobilenetv4_conv_large.e600_r384_in1k`` — timm's own ImageNet-1k
    training of this architecture, trained at 384x384; the authors released
    none — and hosted on the Hugging Face Hub under
    ``lucid-dl/mobilenet-v4-conv-large``.  timm reports 82.974% top-1 /
    96.244% top-5 for it at 384x384 with the preset that
    :meth:`MobileNetV4ConvLargeWeights.E600_R384_IN1K.transforms` reproduces
    (384 crop, 404 resize, bicubic, ImageNet mean/std).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mobilenet_v4 import mobilenet_v4_conv_large_cls
    >>> model = mobilenet_v4_conv_large_cls(num_classes=10).eval()
    >>> model(lucid.randn(2, 3, 224, 224)).logits.shape
    (2, 10)

    Load ImageNet-pretrained weights:

    >>> model = mobilenet_v4_conv_large_cls(pretrained=True)  # doctest: +SKIP
    >>> from lucid.models.weights import MobileNetV4ConvLargeWeights
    >>> model = mobilenet_v4_conv_large_cls(  # doctest: +SKIP
    ...     weights=MobileNetV4ConvLargeWeights.E600_R384_IN1K
    ... )
    """
    entry = weights_mod.resolve_weights(MobileNetV4ConvLargeWeights, pretrained, weights)
    model = MobileNetV4ForImageClassification(apply_overrides(_CFG_CONV_LARGE, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="mobilenet_v4_conv_large_cls")
    return model


@register_model(
    task="image-classification",
    family="mobilenet_v4",
    model_type="mobilenet_v4",
    model_class=MobileNetV4ForImageClassification,
    default_config=_CFG_HYBRID_MEDIUM,
    params=11_074_648,
    summary="auto",
)
def mobilenet_v4_hybrid_medium_cls(
    pretrained: bool | str = False,
    *,
    weights: MobileNetV4HybridMediumWeights | None = None,
    **overrides: object,
) -> MobileNetV4ForImageClassification:
    r"""MobileNet-v4-Hybrid-Medium image classifier.

    Builds a :class:`MobileNetV4ForImageClassification` with the Hybrid-Medium
    architecture (Qin et al., 2024, Appendix D, Table 13) and the
    post-pool head (960 → 1280 → classes).  11.07 M parameters;
    Table 6 reports 80.7% ImageNet-1k top-1 at 10.5 M parameters and
    1.2 G MACs.  Classifier dropout defaults to the paper's 0.2
    (Table 10).

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True`` → the
        ``DEFAULT`` tag
        (:attr:`MobileNetV4HybridMediumWeights.IX_E550_R256_IN1K`); a tag
        string (e.g. ``"IX_E550_R256_IN1K"``) → that specific checkpoint.
        Mutually exclusive with ``weights`` (which wins if both are given).
    weights : MobileNetV4HybridMediumWeights, optional, keyword-only
        Explicit weights enum member, e.g.
        ``MobileNetV4HybridMediumWeights.IX_E550_R256_IN1K``.  Takes
        precedence over ``pretrained``.
    **overrides
        Keyword overrides merged into :class:`MobileNetV4Config`
        (typically ``num_classes`` to retarget the head).  Overriding
        ``num_classes`` away from the checkpoint's 1000 makes pretrained
        loading fail the strict key/shape check — load with the
        matching head, then call :meth:`reset_classifier`.

    Returns
    -------
    MobileNetV4ForImageClassification
        Classifier with the Hybrid-Medium configuration (plus ``overrides``),
        optionally initialised from pretrained weights.

    Notes
    -----
    Qin et al., "MobileNetV4: Universal Models for the Mobile Ecosystem",
    ECCV 2024 (arXiv:2404.10518).  Parameter names match timm's
    implementation, so its ImageNet-1k checkpoints load with an identity
    key map.  The paper evaluates this variant at 256x256.

    Pretrained weights are converted from timm's
    ``mobilenetv4_hybrid_medium.ix_e550_r256_in1k`` — timm's own ImageNet-1k
    training of this architecture, trained at 256x256; the authors released
    none — and hosted on the Hugging Face Hub under
    ``lucid-dl/mobilenet-v4-hybrid-medium``.  timm reports 81.478% top-1 /
    95.692% top-5 for it at 256x256 with the preset that
    :meth:`MobileNetV4HybridMediumWeights.IX_E550_R256_IN1K.transforms`
    reproduces (256 crop, 269 resize, bicubic, ImageNet mean/std).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mobilenet_v4 import mobilenet_v4_hybrid_medium_cls
    >>> model = mobilenet_v4_hybrid_medium_cls(num_classes=10).eval()
    >>> model(lucid.randn(2, 3, 224, 224)).logits.shape
    (2, 10)

    Load ImageNet-pretrained weights:

    >>> model = mobilenet_v4_hybrid_medium_cls(pretrained=True)  # doctest: +SKIP
    >>> from lucid.models.weights import MobileNetV4HybridMediumWeights
    >>> model = mobilenet_v4_hybrid_medium_cls(  # doctest: +SKIP
    ...     weights=MobileNetV4HybridMediumWeights.IX_E550_R256_IN1K
    ... )
    """
    entry = weights_mod.resolve_weights(MobileNetV4HybridMediumWeights, pretrained, weights)
    model = MobileNetV4ForImageClassification(apply_overrides(_CFG_HYBRID_MEDIUM, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="mobilenet_v4_hybrid_medium_cls")
    return model


@register_model(
    task="image-classification",
    family="mobilenet_v4",
    model_type="mobilenet_v4",
    model_class=MobileNetV4ForImageClassification,
    default_config=_CFG_HYBRID_LARGE,
    params=37_764_624,
    summary="auto",
)
def mobilenet_v4_hybrid_large_cls(
    pretrained: bool | str = False,
    *,
    weights: MobileNetV4HybridLargeWeights | None = None,
    **overrides: object,
) -> MobileNetV4ForImageClassification:
    r"""MobileNet-v4-Hybrid-Large image classifier.

    Builds a :class:`MobileNetV4ForImageClassification` with the Hybrid-Large
    architecture (Qin et al., 2024, Appendix D, Table 15) and the
    post-pool head (960 → 1280 → classes).  37.76 M parameters;
    Table 6 reports 83.4% ImageNet-1k top-1 at 35.9 M parameters and
    7.2 G MACs.  Classifier dropout defaults to the paper's 0.2
    (Table 10).

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True`` → the
        ``DEFAULT`` tag
        (:attr:`MobileNetV4HybridLargeWeights.IX_E600_R384_IN1K`); a tag
        string (e.g. ``"IX_E600_R384_IN1K"``) → that specific checkpoint.
        Mutually exclusive with ``weights`` (which wins if both are given).
    weights : MobileNetV4HybridLargeWeights, optional, keyword-only
        Explicit weights enum member, e.g.
        ``MobileNetV4HybridLargeWeights.IX_E600_R384_IN1K``.  Takes
        precedence over ``pretrained``.
    **overrides
        Keyword overrides merged into :class:`MobileNetV4Config`
        (typically ``num_classes`` to retarget the head).  Overriding
        ``num_classes`` away from the checkpoint's 1000 makes pretrained
        loading fail the strict key/shape check — load with the
        matching head, then call :meth:`reset_classifier`.

    Returns
    -------
    MobileNetV4ForImageClassification
        Classifier with the Hybrid-Large configuration (plus ``overrides``),
        optionally initialised from pretrained weights.

    Notes
    -----
    Qin et al., "MobileNetV4: Universal Models for the Mobile Ecosystem",
    ECCV 2024 (arXiv:2404.10518).  Parameter names match timm's
    implementation, so its ImageNet-1k checkpoints load with an identity
    key map.  The paper evaluates this variant at 384x384.

    Pretrained weights are converted from timm's
    ``mobilenetv4_hybrid_large.ix_e600_r384_in1k`` — timm's own ImageNet-1k
    training of this architecture, trained at 384x384; the authors released
    none — and hosted on the Hugging Face Hub under
    ``lucid-dl/mobilenet-v4-hybrid-large``.  timm reports 83.996% top-1 /
    96.714% top-5 for it at 384x384 with the preset that
    :meth:`MobileNetV4HybridLargeWeights.IX_E600_R384_IN1K.transforms`
    reproduces (384 crop, 404 resize, bicubic, ImageNet mean/std).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mobilenet_v4 import mobilenet_v4_hybrid_large_cls
    >>> model = mobilenet_v4_hybrid_large_cls(num_classes=10).eval()
    >>> model(lucid.randn(2, 3, 224, 224)).logits.shape
    (2, 10)

    Load ImageNet-pretrained weights:

    >>> model = mobilenet_v4_hybrid_large_cls(pretrained=True)  # doctest: +SKIP
    >>> from lucid.models.weights import MobileNetV4HybridLargeWeights
    >>> model = mobilenet_v4_hybrid_large_cls(  # doctest: +SKIP
    ...     weights=MobileNetV4HybridLargeWeights.IX_E600_R384_IN1K
    ... )
    """
    entry = weights_mod.resolve_weights(MobileNetV4HybridLargeWeights, pretrained, weights)
    model = MobileNetV4ForImageClassification(apply_overrides(_CFG_HYBRID_LARGE, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="mobilenet_v4_hybrid_large_cls")
    return model
