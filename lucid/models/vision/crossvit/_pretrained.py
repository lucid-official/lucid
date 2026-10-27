"""Registry factories for the paper-cited CrossViT variants.

All six configs from Chen et al., ICCV 2021, Table 2.  Per-variant
hyperparameters are sourced from the paper directly and cross-checked
against ``timm.models.crossvit`` so the converted timm checkpoints
load with a direct ``blocks → stages`` key rename.
"""

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models.vision.crossvit._config import CrossViTConfig
from lucid.models.vision.crossvit._model import CrossViT, CrossViTForImageClassification
from lucid.models.vision.crossvit._weights import (
    CrossViT9Weights,
    CrossViT15Weights,
    CrossViT18Weights,
    CrossViTBaseWeights,
    CrossViTSmallWeights,
    CrossViTTinyWeights,
)
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained

# ---------------------------------------------------------------------------
# Paper Table 2 configurations.
#
# Every variant shares the same skeleton (K=3 MultiScaleBlocks, patch sizes
# (12, 16), dual-input scale (240, 224)); only ``embed_dims`` / ``depths`` /
# ``num_heads`` / ``mlp_ratio`` change.
# ---------------------------------------------------------------------------

_DEPTH_TINY = ((1, 4, 0), (1, 4, 0), (1, 4, 0))
_DEPTH_SMALL = ((1, 4, 0), (1, 4, 0), (1, 4, 0))
_DEPTH_BASE = ((1, 4, 0), (1, 4, 0), (1, 4, 0))
_DEPTH_9 = ((1, 3, 0), (1, 3, 0), (1, 3, 0))
_DEPTH_15 = ((1, 5, 0), (1, 5, 0), (1, 5, 0))
_DEPTH_18 = ((1, 6, 0), (1, 6, 0), (1, 6, 0))

_MLP_BCT = (4.0, 4.0, 1.0)  # tiny / small / base
_MLP_9_15_18 = (3.0, 3.0, 1.0)  # 9 / 15 / 18

_CFG_TINY = CrossViTConfig(
    embed_dims=(96, 192), depths=_DEPTH_TINY, num_heads=(3, 3), mlp_ratio=_MLP_BCT
)
_CFG_SMALL = CrossViTConfig(
    embed_dims=(192, 384), depths=_DEPTH_SMALL, num_heads=(6, 6), mlp_ratio=_MLP_BCT
)
_CFG_BASE = CrossViTConfig(
    embed_dims=(384, 768), depths=_DEPTH_BASE, num_heads=(12, 12), mlp_ratio=_MLP_BCT
)
_CFG_9 = CrossViTConfig(
    embed_dims=(128, 256), depths=_DEPTH_9, num_heads=(4, 4), mlp_ratio=_MLP_9_15_18
)
_CFG_15 = CrossViTConfig(
    embed_dims=(192, 384), depths=_DEPTH_15, num_heads=(6, 6), mlp_ratio=_MLP_9_15_18
)
_CFG_18 = CrossViTConfig(
    embed_dims=(224, 448), depths=_DEPTH_18, num_heads=(7, 7), mlp_ratio=_MLP_9_15_18
)


# ---------------------------------------------------------------------------
# Backbone factories (task="base")
# ---------------------------------------------------------------------------


@register_model(
    task="base",
    family="crossvit",
    model_type="crossvit",
    model_class=CrossViT,
    default_config=_CFG_TINY,
    params=6_724_800,
)
def crossvit_tiny(pretrained: bool = False, **overrides: object) -> CrossViT:
    r"""CrossViT-Ti backbone — ``embed_dims=(96, 192)``, depths
    ``((1, 4, 0))×3``, 3 heads.  ~7M params (paper Table 2).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``crossvit_tiny``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``crossvit_tiny_cls`` is the checkpointed variant.
    **overrides : object
        Fields of :class:`CrossViTConfig` to change from the CrossViT-Ti
        defaults (e.g. ``in_channels=1``).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.crossvit import crossvit_tiny
    >>> model = crossvit_tiny()
    >>> out = model(lucid.randn(1, 3, 240, 240))
    >>> out.last_hidden_state.shape
    (1, 288)
    """
    if pretrained:
        reject_unavailable_pretrained("crossvit_tiny", alternative="crossvit_tiny_cls")
    return CrossViT(apply_overrides(_CFG_TINY, overrides))


@register_model(
    task="base",
    family="crossvit",
    model_type="crossvit",
    model_class=CrossViT,
    default_config=_CFG_SMALL,
    params=26_278_272,
)
def crossvit_small(pretrained: bool = False, **overrides: object) -> CrossViT:
    r"""CrossViT-S backbone — ``embed_dims=(192, 384)``, depths
    ``((1, 4, 0))×3``, 6 heads.  ~26.7M params (paper Table 2).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``crossvit_small``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``crossvit_small_cls`` is the checkpointed variant.
    **overrides : object
        Fields of :class:`CrossViTConfig` to change from the CrossViT-S
        defaults (e.g. ``in_channels=1``).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.crossvit import crossvit_small
    >>> model = crossvit_small()
    >>> out = model(lucid.randn(1, 3, 240, 240))
    >>> out.last_hidden_state.shape
    (1, 576)
    """
    if pretrained:
        reject_unavailable_pretrained("crossvit_small", alternative="crossvit_small_cls")
    return CrossViT(apply_overrides(_CFG_SMALL, overrides))


@register_model(
    task="base",
    family="crossvit",
    model_type="crossvit",
    model_class=CrossViT,
    default_config=_CFG_BASE,
    params=103_871_232,
)
def crossvit_base(pretrained: bool = False, **overrides: object) -> CrossViT:
    r"""CrossViT-B backbone — ``embed_dims=(384, 768)``, depths
    ``((1, 4, 0))×3``, 12 heads.  ~105M params (paper Table 2).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``crossvit_base``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``crossvit_base_cls`` is the checkpointed variant.
    **overrides : object
        Fields of :class:`CrossViTConfig` to change from the CrossViT-B
        defaults (e.g. ``in_channels=1``).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.crossvit import crossvit_base
    >>> model = crossvit_base()
    >>> out = model(lucid.randn(1, 3, 240, 240))
    >>> out.last_hidden_state.shape
    (1, 1152)
    """
    if pretrained:
        reject_unavailable_pretrained("crossvit_base", alternative="crossvit_base_cls")
    return CrossViT(apply_overrides(_CFG_BASE, overrides))


@register_model(
    task="base",
    family="crossvit",
    model_type="crossvit",
    model_class=CrossViT,
    default_config=_CFG_9,
    params=8_167_296,
)
def crossvit_9(pretrained: bool = False, **overrides: object) -> CrossViT:
    r"""CrossViT-9 backbone — ``embed_dims=(128, 256)``, depths
    ``((1, 3, 0))×3``, 4 heads.  ~8.6M params (paper Table 2).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``crossvit_9``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``crossvit_9_cls`` is the checkpointed variant.
    **overrides : object
        Fields of :class:`CrossViTConfig` to change from the CrossViT-9
        defaults (e.g. ``in_channels=1``).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.crossvit import crossvit_9
    >>> model = crossvit_9()
    >>> out = model(lucid.randn(1, 3, 240, 240))
    >>> out.last_hidden_state.shape
    (1, 384)

    CrossViT works at 240 pixels rather than 224: the two branches
    use 12- and 16-pixel patches, and 240 is what both divide.
    """
    if pretrained:
        reject_unavailable_pretrained("crossvit_9", alternative="crossvit_9_cls")
    return CrossViT(apply_overrides(_CFG_9, overrides))


@register_model(
    task="base",
    family="crossvit",
    model_type="crossvit",
    model_class=CrossViT,
    default_config=_CFG_15,
    params=26_950_464,
)
def crossvit_15(pretrained: bool = False, **overrides: object) -> CrossViT:
    r"""CrossViT-15 backbone — ``embed_dims=(192, 384)``, depths
    ``((1, 5, 0))×3``, 6 heads.  ~27.4M params (paper Table 2).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``crossvit_15``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``crossvit_15_cls`` is the checkpointed variant.
    **overrides : object
        Fields of :class:`CrossViTConfig` to change from the CrossViT-15
        defaults (e.g. ``in_channels=1``).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.crossvit import crossvit_15
    >>> model = crossvit_15()
    >>> out = model(lucid.randn(1, 3, 240, 240))
    >>> out.last_hidden_state.shape
    (1, 576)
    """
    if pretrained:
        reject_unavailable_pretrained("crossvit_15", alternative="crossvit_15_cls")
    return CrossViT(apply_overrides(_CFG_15, overrides))


@register_model(
    task="base",
    family="crossvit",
    model_type="crossvit",
    model_class=CrossViT,
    default_config=_CFG_18,
    params=42_597_408,
)
def crossvit_18(pretrained: bool = False, **overrides: object) -> CrossViT:
    r"""CrossViT-18 backbone — ``embed_dims=(224, 448)``, depths
    ``((1, 6, 0))×3``, 7 heads.  ~43.3M params (paper Table 2).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``crossvit_18``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``crossvit_18_cls`` is the checkpointed variant.
    **overrides : object
        Fields of :class:`CrossViTConfig` to change from the CrossViT-18
        defaults (e.g. ``in_channels=1``).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.crossvit import crossvit_18
    >>> model = crossvit_18()
    >>> out = model(lucid.randn(1, 3, 240, 240))
    >>> out.last_hidden_state.shape
    (1, 672)
    """
    if pretrained:
        reject_unavailable_pretrained("crossvit_18", alternative="crossvit_18_cls")
    return CrossViT(apply_overrides(_CFG_18, overrides))


# ---------------------------------------------------------------------------
# Classification factories with pretrained-weight wiring (task="image-classification")
# ---------------------------------------------------------------------------


@register_model(
    task="image-classification",
    family="crossvit",
    model_type="crossvit",
    model_class=CrossViTForImageClassification,
    default_config=_CFG_TINY,
    params=7_014_800,
)
def crossvit_tiny_cls(
    pretrained: bool | str = False,
    *,
    weights: CrossViTTinyWeights | None = None,
    **overrides: object,
) -> CrossViTForImageClassification:
    r"""CrossViT-Ti image classifier (Chen et al., ICCV 2021).

    Two-branch ViT with embedding dims ``(96, 192)``, per-stage depths
    ``((1, 4, 0)) x 3``, and 3 attention heads per branch.  ~7M
    parameters; paper Table 2 reports **73.4%** ImageNet-1k top-1 at
    240x240.

    Parameters
    ----------
    pretrained : bool or str, default ``False``
        ``False`` returns randomly-initialised weights; ``True`` loads
        the default entry of :class:`CrossViTTinyWeights`; a string
        selects a named entry from that enum.
    weights : CrossViTTinyWeights, optional
        Explicit weight-enum member.  Mutually exclusive with a string
        ``pretrained``; raises if both resolve to different entries.
    **overrides : object
        Per-field overrides forwarded to :class:`CrossViTConfig` (e.g.
        ``num_classes=10`` to retarget the head).

    Returns
    -------
    CrossViTForImageClassification
        The configured classifier, ready for ``model(x)`` inference or
        fine-tuning.

    Notes
    -----
    Reference: Chen, Fan, and Panda, *"CrossViT: Cross-Attention
    Multi-Scale Vision Transformer for Image Classification"*, ICCV
    2021 (arXiv:2103.14899).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.crossvit import crossvit_tiny_cls
    >>> model = crossvit_tiny_cls()
    >>> out = model(lucid.randn(1, 3, 240, 240))
    >>> out.logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(CrossViTTinyWeights, pretrained, weights)
    model = CrossViTForImageClassification(apply_overrides(_CFG_TINY, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="crossvit_tiny_cls")
    return model


@register_model(
    task="image-classification",
    family="crossvit",
    model_type="crossvit",
    model_class=CrossViTForImageClassification,
    default_config=_CFG_SMALL,
    params=26_856_272,
)
def crossvit_small_cls(
    pretrained: bool | str = False,
    *,
    weights: CrossViTSmallWeights | None = None,
    **overrides: object,
) -> CrossViTForImageClassification:
    r"""CrossViT-S image classifier — ``embed_dims=(192, 384)``, depths
    ``((1, 4, 0)) x 3``, 6 heads per branch.  ~26.7M params; paper
    Table 2 reports **81.0%** ImageNet-1k top-1 at 240x240.

    Parameters
    ----------
    pretrained : bool or str, default ``False``
        ``False`` returns randomly-initialised weights; ``True`` loads
        the default entry of :class:`CrossViTSmallWeights`; a string
        selects a named entry from that enum.
    weights : CrossViTSmallWeights, optional
        Explicit weight-enum member overriding ``pretrained``.
    **overrides : object
        Per-field overrides forwarded to :class:`CrossViTConfig`.

    Returns
    -------
    CrossViTForImageClassification
        The configured classifier.

    Notes
    -----
    Reference: Chen, Fan, and Panda, *"CrossViT: Cross-Attention
    Multi-Scale Vision Transformer for Image Classification"*, ICCV
    2021 (arXiv:2103.14899).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.crossvit import crossvit_small_cls
    >>> model = crossvit_small_cls()
    >>> out = model(lucid.randn(1, 3, 240, 240))
    >>> out.logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(CrossViTSmallWeights, pretrained, weights)
    model = CrossViTForImageClassification(apply_overrides(_CFG_SMALL, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="crossvit_small_cls")
    return model


@register_model(
    task="image-classification",
    family="crossvit",
    model_type="crossvit",
    model_class=CrossViTForImageClassification,
    default_config=_CFG_BASE,
    params=105_025_232,
)
def crossvit_base_cls(
    pretrained: bool | str = False,
    *,
    weights: CrossViTBaseWeights | None = None,
    **overrides: object,
) -> CrossViTForImageClassification:
    r"""CrossViT-B image classifier — ``embed_dims=(384, 768)``, depths
    ``((1, 4, 0)) x 3``, 12 heads per branch.  ~105M params; paper
    Table 2 reports **82.2%** ImageNet-1k top-1 at 240x240.

    Parameters
    ----------
    pretrained : bool or str, default ``False``
        ``False`` returns randomly-initialised weights; ``True`` loads
        the default entry of :class:`CrossViTBaseWeights`; a string
        selects a named entry from that enum.
    weights : CrossViTBaseWeights, optional
        Explicit weight-enum member overriding ``pretrained``.
    **overrides : object
        Per-field overrides forwarded to :class:`CrossViTConfig`.

    Returns
    -------
    CrossViTForImageClassification
        The configured classifier.

    Notes
    -----
    Reference: Chen, Fan, and Panda, *"CrossViT: Cross-Attention
    Multi-Scale Vision Transformer for Image Classification"*, ICCV
    2021 (arXiv:2103.14899).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.crossvit import crossvit_base_cls
    >>> model = crossvit_base_cls()
    >>> out = model(lucid.randn(1, 3, 240, 240))
    >>> out.logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(CrossViTBaseWeights, pretrained, weights)
    model = CrossViTForImageClassification(apply_overrides(_CFG_BASE, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="crossvit_base_cls")
    return model


@register_model(
    task="image-classification",
    family="crossvit",
    model_type="crossvit",
    model_class=CrossViTForImageClassification,
    default_config=_CFG_9,
    params=8_553_296,
)
def crossvit_9_cls(
    pretrained: bool | str = False, *, weights: CrossViT9Weights | None = None, **overrides: object
) -> CrossViTForImageClassification:
    r"""CrossViT-9 image classifier — ``embed_dims=(128, 256)``, depths
    ``((1, 3, 0)) x 3``, 4 heads per branch.  ~8.6M params; paper
    Table 2 reports **73.9%** ImageNet-1k top-1 at 240x240.

    Parameters
    ----------
    pretrained : bool or str, default ``False``
        ``False`` returns randomly-initialised weights; ``True`` loads
        the default entry of :class:`CrossViT9Weights`; a string
        selects a named entry from that enum.
    weights : CrossViT9Weights, optional
        Explicit weight-enum member overriding ``pretrained``.
    **overrides : object
        Per-field overrides forwarded to :class:`CrossViTConfig`.

    Returns
    -------
    CrossViTForImageClassification
        The configured classifier.

    Notes
    -----
    Reference: Chen, Fan, and Panda, *"CrossViT: Cross-Attention
    Multi-Scale Vision Transformer for Image Classification"*, ICCV
    2021 (arXiv:2103.14899).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.crossvit import crossvit_9_cls
    >>> model = crossvit_9_cls()
    >>> out = model(lucid.randn(1, 3, 240, 240))
    >>> out.logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(CrossViT9Weights, pretrained, weights)
    model = CrossViTForImageClassification(apply_overrides(_CFG_9, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="crossvit_9_cls")
    return model


@register_model(
    task="image-classification",
    family="crossvit",
    model_type="crossvit",
    model_class=CrossViTForImageClassification,
    default_config=_CFG_15,
    params=27_528_464,
)
def crossvit_15_cls(
    pretrained: bool | str = False, *, weights: CrossViT15Weights | None = None, **overrides: object
) -> CrossViTForImageClassification:
    r"""CrossViT-15 image classifier — ``embed_dims=(192, 384)``, depths
    ``((1, 5, 0)) x 3``, 6 heads per branch.  ~27.4M params; paper
    Table 2 reports **81.5%** ImageNet-1k top-1 at 240x240.

    Parameters
    ----------
    pretrained : bool or str, default ``False``
        ``False`` returns randomly-initialised weights; ``True`` loads
        the default entry of :class:`CrossViT15Weights`; a string
        selects a named entry from that enum.
    weights : CrossViT15Weights, optional
        Explicit weight-enum member overriding ``pretrained``.
    **overrides : object
        Per-field overrides forwarded to :class:`CrossViTConfig`.

    Returns
    -------
    CrossViTForImageClassification
        The configured classifier.

    Notes
    -----
    Reference: Chen, Fan, and Panda, *"CrossViT: Cross-Attention
    Multi-Scale Vision Transformer for Image Classification"*, ICCV
    2021 (arXiv:2103.14899).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.crossvit import crossvit_15_cls
    >>> model = crossvit_15_cls()
    >>> out = model(lucid.randn(1, 3, 240, 240))
    >>> out.logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(CrossViT15Weights, pretrained, weights)
    model = CrossViTForImageClassification(apply_overrides(_CFG_15, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="crossvit_15_cls")
    return model


@register_model(
    task="image-classification",
    family="crossvit",
    model_type="crossvit",
    model_class=CrossViTForImageClassification,
    default_config=_CFG_18,
    params=43_271_408,
)
def crossvit_18_cls(
    pretrained: bool | str = False, *, weights: CrossViT18Weights | None = None, **overrides: object
) -> CrossViTForImageClassification:
    r"""CrossViT-18 image classifier — ``embed_dims=(224, 448)``, depths
    ``((1, 6, 0)) x 3``, 7 heads per branch.  ~43.3M params; paper
    Table 2 reports **82.5%** ImageNet-1k top-1 at 240x240.

    Parameters
    ----------
    pretrained : bool or str, default ``False``
        ``False`` returns randomly-initialised weights; ``True`` loads
        the default entry of :class:`CrossViT18Weights`; a string
        selects a named entry from that enum.
    weights : CrossViT18Weights, optional
        Explicit weight-enum member overriding ``pretrained``.
    **overrides : object
        Per-field overrides forwarded to :class:`CrossViTConfig`.

    Returns
    -------
    CrossViTForImageClassification
        The configured classifier.

    Notes
    -----
    Reference: Chen, Fan, and Panda, *"CrossViT: Cross-Attention
    Multi-Scale Vision Transformer for Image Classification"*, ICCV
    2021 (arXiv:2103.14899).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.crossvit import crossvit_18_cls
    >>> model = crossvit_18_cls()
    >>> out = model(lucid.randn(1, 3, 240, 240))
    >>> out.logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(CrossViT18Weights, pretrained, weights)
    model = CrossViTForImageClassification(apply_overrides(_CFG_18, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="crossvit_18_cls")
    return model
