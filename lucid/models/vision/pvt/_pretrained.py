"""Registry factories for PVT v2 variants."""

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models.vision.pvt._config import PVTConfig
from lucid.models.vision.pvt._model import PVT, PVTForImageClassification
from lucid.models.vision.pvt._weights import (
    PVTv2B0Weights,
    PVTv2B1Weights,
    PVTv2B2Weights,
    PVTv2B3Weights,
    PVTv2B4Weights,
    PVTv2B5Weights,
)
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained

# ── Canonical configs ─────────────────────────────────────────────────────────

_CFG_B0 = PVTConfig(
    variant="pvt_v2_b0",
    embed_dims=(32, 64, 160, 256),
    depths=(2, 2, 2, 2),
    num_heads=(1, 2, 5, 8),
    sr_ratios=(8, 4, 2, 1),
    mlp_ratios=(8.0, 8.0, 4.0, 4.0),
)

_CFG_B1 = PVTConfig(
    variant="pvt_v2_b1",
    embed_dims=(64, 128, 320, 512),
    depths=(2, 2, 2, 2),
    num_heads=(1, 2, 5, 8),
    sr_ratios=(8, 4, 2, 1),
    mlp_ratios=(8.0, 8.0, 4.0, 4.0),
)

# Divergence, deliberate: Table 1 prints L_2 = 3 for B2 and B3, but the
# authors' released configs use 4, and the paper's own parameter counts
# agree with the configs rather than with its table.  Measured here:
# B2 with L_2 = 4 is 25,362,856 parameters with the classifier, matching
# the 25.4M Table 2 reports; L_2 = 3 gives 24.76M.  B3 with L_2 = 4 is
# 45,238,696 against the reported 45.2M; L_2 = 3 gives 44.64M.  Following
# the configs the checkpoints were trained under.
_CFG_B2 = PVTConfig(
    variant="pvt_v2_b2",
    embed_dims=(64, 128, 320, 512),
    depths=(3, 4, 6, 3),
    num_heads=(1, 2, 5, 8),
    sr_ratios=(8, 4, 2, 1),
    mlp_ratios=(8.0, 8.0, 4.0, 4.0),
)

# PVT v2-B2-Li — Table 1's "B2-Li" column: the same B2 topology with
# *linear* SRA (average-pool to 7x7 at every stage instead of a stride-R_i
# convolution).  Table 2 reports 22.6M / 3.9 GFLOPs / 82.1 top-1.
_CFG_B2_LI = PVTConfig(
    variant="pvt_v2_b2_li",
    embed_dims=(64, 128, 320, 512),
    depths=(3, 4, 6, 3),
    num_heads=(1, 2, 5, 8),
    sr_ratios=(8, 4, 2, 1),
    mlp_ratios=(8.0, 8.0, 4.0, 4.0),
    linear_sra=True,
    linear_pool_size=7,
)

_CFG_B3 = PVTConfig(
    variant="pvt_v2_b3",
    embed_dims=(64, 128, 320, 512),
    depths=(3, 4, 18, 3),
    num_heads=(1, 2, 5, 8),
    sr_ratios=(8, 4, 2, 1),
    mlp_ratios=(8.0, 8.0, 4.0, 4.0),
)

_CFG_B4 = PVTConfig(
    variant="pvt_v2_b4",
    embed_dims=(64, 128, 320, 512),
    depths=(3, 8, 27, 3),
    num_heads=(1, 2, 5, 8),
    sr_ratios=(8, 4, 2, 1),
    mlp_ratios=(8.0, 8.0, 4.0, 4.0),
)

_CFG_B5 = PVTConfig(
    variant="pvt_v2_b5",
    embed_dims=(64, 128, 320, 512),
    depths=(3, 6, 40, 3),
    num_heads=(1, 2, 5, 8),
    sr_ratios=(8, 4, 2, 1),
    mlp_ratios=(4.0, 4.0, 4.0, 4.0),
)


# ── Backbones ─────────────────────────────────────────────────────────────────


@register_model(
    task="base", family="pvt", model_type="pvt", model_class=PVT, default_config=_CFG_B0
)
def pvt_v2_b0(pretrained: bool = False, **overrides: object) -> PVT:
    r"""PVT v2-B0 backbone (Wang et al., 2022).

    Builds the smallest PVT v2 variant: ``embed_dims=(32, 64, 160, 256)``,
    ``depths=(2, 2, 2, 2)``, ``num_heads=(1, 2, 5, 8)``,
    ``sr_ratios=(8, 4, 2, 1)``.  Approximately **3.7M parameters** —
    targeted at mobile / edge deployments.

    Parameters
    ----------
    pretrained : bool, optional
        No checkpoint is published for ``pvt_v2_b0``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``pvt_v2_b0_cls`` is the checkpointed variant.
    **overrides : object
        Keyword overrides on top of the canonical PVT v2-B0 config.

    Returns
    -------
    PVT
        A :class:`PVT` backbone returning a flat :math:`(B, 256)`
        feature.

    Notes
    -----
    PVT v2-B0 reaches **70.5% top-1 on ImageNet-1k** at 224x224 (Wang
    et al., 2022, Table 1).  See
    `arXiv:2106.13797 <https://arxiv.org/abs/2106.13797>`_.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.pvt import pvt_v2_b0
    >>> model = pvt_v2_b0()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> feat = model.forward_features(x)
    >>> feat.shape
    (1, 256)
    """
    if pretrained:
        reject_unavailable_pretrained("pvt_v2_b0", alternative="pvt_v2_b0_cls")
    return PVT(apply_overrides(_CFG_B0, overrides))


@register_model(
    task="base", family="pvt", model_type="pvt", model_class=PVT, default_config=_CFG_B1
)
def pvt_v2_b1(pretrained: bool = False, **overrides: object) -> PVT:
    r"""PVT v2-B1 backbone (Wang et al., 2022).

    Builds the canonical *PVT v2-B1* configuration:
    ``embed_dims=(64, 128, 320, 512)``, ``depths=(2, 2, 2, 2)``,
    ``num_heads=(1, 2, 5, 8)``, ``sr_ratios=(8, 4, 2, 1)``.
    Approximately **14.0M parameters** — ResNet-18 / 34 class on the
    accuracy/cost trade-off.

    Parameters
    ----------
    pretrained : bool, optional
        No checkpoint is published for ``pvt_v2_b1``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``pvt_v2_b1_cls`` is the checkpointed variant.
    **overrides : object
        Keyword overrides on top of the canonical PVT v2-B1 config.

    Returns
    -------
    PVT
        A :class:`PVT` backbone returning a flat :math:`(B, 512)`
        feature.

    Notes
    -----
    PVT v2-B1 reaches **78.7% top-1 on ImageNet-1k** at 224x224 (Wang
    et al., 2022, Table 1).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.pvt import pvt_v2_b1
    >>> model = pvt_v2_b1()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model.forward_features(x).shape
    (1, 512)
    """
    if pretrained:
        reject_unavailable_pretrained("pvt_v2_b1", alternative="pvt_v2_b1_cls")
    return PVT(apply_overrides(_CFG_B1, overrides))


@register_model(
    task="base", family="pvt", model_type="pvt", model_class=PVT, default_config=_CFG_B2
)
def pvt_v2_b2(pretrained: bool = False, **overrides: object) -> PVT:
    r"""PVT v2-B2 backbone (Wang et al., 2022).

    Builds the canonical *PVT v2-B2* configuration:
    ``embed_dims=(64, 128, 320, 512)``, ``depths=(3, 4, 6, 3)``.
    Approximately **25.4M parameters** — ResNet-50 class on accuracy/cost.

    Parameters
    ----------
    pretrained : bool, optional
        No checkpoint is published for ``pvt_v2_b2``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``pvt_v2_b2_cls`` is the checkpointed variant.
    **overrides : object
        Keyword overrides on top of the canonical PVT v2-B2 config.

    Returns
    -------
    PVT
        A :class:`PVT` backbone returning a flat :math:`(B, 512)`
        feature.

    Notes
    -----
    PVT v2-B2 reaches **82.0% top-1 on ImageNet-1k** at 224x224 (Wang
    et al., 2022, Table 1).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.pvt import pvt_v2_b2
    >>> model = pvt_v2_b2()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model.forward_features(x).shape
    (1, 512)
    """
    if pretrained:
        reject_unavailable_pretrained("pvt_v2_b2", alternative="pvt_v2_b2_cls")
    return PVT(apply_overrides(_CFG_B2, overrides))


@register_model(
    task="base", family="pvt", model_type="pvt", model_class=PVT, default_config=_CFG_B2_LI
)
def pvt_v2_b2_li(pretrained: bool = False, **overrides: object) -> PVT:
    r"""PVT v2-B2-Li backbone — B2 with *linear* SRA (Wang et al., 2022).

    Same topology as :func:`pvt_v2_b2`, but §3.5's linear spatial-reduction
    attention: every stage average-pools its key/value map to a fixed
    :math:`7 \times 7` grid before the :math:`1 \times 1` projection,
    instead of applying a stride-:math:`R_i` convolution.  Attention cost
    becomes linear in the token count rather than quadratic.

    Parameters
    ----------
    pretrained : bool, optional
        No checkpoint is published for ``pvt_v2_b2_li``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        Keyword overrides on top of the canonical PVT v2-B2-Li config.

    Returns
    -------
    PVT
        A :class:`PVT` backbone returning a flat :math:`(B, 512)` feature.

    Notes
    -----
    Table 1 carries a full "B2-Li" column and Table 2 reports it at
    **22.6M parameters, 3.9 GFLOPs and 82.1% top-1** on ImageNet-1k.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.pvt import pvt_v2_b2_li
    >>> model = pvt_v2_b2_li()
    >>> model.forward_features(lucid.randn(1, 3, 224, 224)).shape
    (1, 512)
    """
    if pretrained:
        reject_unavailable_pretrained("pvt_v2_b2_li")
    return PVT(apply_overrides(_CFG_B2_LI, overrides))


@register_model(
    task="base", family="pvt", model_type="pvt", model_class=PVT, default_config=_CFG_B3
)
def pvt_v2_b3(pretrained: bool = False, **overrides: object) -> PVT:
    r"""PVT v2-B3 backbone (Wang et al., 2022).

    Builds the canonical *PVT v2-B3* configuration:
    ``embed_dims=(64, 128, 320, 512)``, ``depths=(3, 4, 18, 3)`` (a
    deeper stage 3 than B2).  Approximately **45.2M parameters** —
    ResNet-101 class on accuracy/cost.

    Parameters
    ----------
    pretrained : bool, optional
        No checkpoint is published for ``pvt_v2_b3``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``pvt_v2_b3_cls`` is the checkpointed variant.
    **overrides : object
        Keyword overrides on top of the canonical PVT v2-B3 config.

    Returns
    -------
    PVT
        A :class:`PVT` backbone returning a flat :math:`(B, 512)`
        feature.

    Notes
    -----
    PVT v2-B3 reaches **83.1% top-1 on ImageNet-1k** at 224x224 (Wang
    et al., 2022, Table 1).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.pvt import pvt_v2_b3
    >>> model = pvt_v2_b3()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model.forward_features(x).shape
    (1, 512)
    """
    if pretrained:
        reject_unavailable_pretrained("pvt_v2_b3", alternative="pvt_v2_b3_cls")
    return PVT(apply_overrides(_CFG_B3, overrides))


@register_model(
    task="base", family="pvt", model_type="pvt", model_class=PVT, default_config=_CFG_B4
)
def pvt_v2_b4(pretrained: bool = False, **overrides: object) -> PVT:
    r"""PVT v2-B4 backbone (Wang et al., 2022).

    Builds the canonical *PVT v2-B4* configuration:
    ``embed_dims=(64, 128, 320, 512)``, ``depths=(3, 8, 27, 3)``.
    Approximately **62.6M parameters**.

    Parameters
    ----------
    pretrained : bool, optional
        No checkpoint is published for ``pvt_v2_b4``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``pvt_v2_b4_cls`` is the checkpointed variant.
    **overrides : object
        Keyword overrides on top of the canonical PVT v2-B4 config.

    Returns
    -------
    PVT
        A :class:`PVT` backbone returning a flat :math:`(B, 512)`
        feature.

    Notes
    -----
    PVT v2-B4 reaches **83.6% top-1 on ImageNet-1k** at 224x224 (Wang
    et al., 2022, Table 1).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.pvt import pvt_v2_b4
    >>> model = pvt_v2_b4()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model.forward_features(x).shape
    (1, 512)
    """
    if pretrained:
        reject_unavailable_pretrained("pvt_v2_b4", alternative="pvt_v2_b4_cls")
    return PVT(apply_overrides(_CFG_B4, overrides))


@register_model(
    task="base", family="pvt", model_type="pvt", model_class=PVT, default_config=_CFG_B5
)
def pvt_v2_b5(pretrained: bool = False, **overrides: object) -> PVT:
    r"""PVT v2-B5 backbone (Wang et al., 2022).

    Builds the largest PVT v2 variant *B5*:
    ``embed_dims=(64, 128, 320, 512)``, ``depths=(3, 6, 40, 3)``,
    and ``mlp_ratios=(4.0, 4.0, 4.0, 4.0)`` (uniform MLP ratio).
    81.4M parameters without a classifier head
    (:func:`pvt_v2_b5_cls` adds 0.51M more for the 1000-way head,
    reaching 82.0M) — the largest variant in the paper.

    Parameters
    ----------
    pretrained : bool, optional
        No checkpoint is published for ``pvt_v2_b5``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``pvt_v2_b5_cls`` is the checkpointed variant.
    **overrides : object
        Keyword overrides on top of the canonical PVT v2-B5 config.

    Returns
    -------
    PVT
        A :class:`PVT` backbone returning a flat :math:`(B, 512)`
        feature.

    Notes
    -----
    PVT v2-B5 reaches **83.8% top-1 on ImageNet-1k** at 224x224 (Wang
    et al., 2022, Table 1) — the headline result of the paper.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.pvt import pvt_v2_b5
    >>> model = pvt_v2_b5()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model.forward_features(x).shape
    (1, 512)
    """
    if pretrained:
        reject_unavailable_pretrained("pvt_v2_b5", alternative="pvt_v2_b5_cls")
    return PVT(apply_overrides(_CFG_B5, overrides))


@register_model(
    task="image-classification",
    family="pvt",
    model_type="pvt",
    model_class=PVTForImageClassification,
    default_config=_CFG_B0,
)
def pvt_v2_b0_cls(
    pretrained: bool | str = False, *, weights: PVTv2B0Weights | None = None, **overrides: object
) -> PVTForImageClassification:
    r"""PVT v2-B0 image classifier (Wang et al., 2022).

    Combines the :func:`pvt_v2_b0` backbone with a mean pool + single
    :class:`nn.Linear` classification head.  Default output is
    ``num_classes=1000`` (ImageNet-1k).  ~3.7M parameters.

    Parameters
    ----------
    pretrained : bool or str, optional
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`PVTv2B0Weights.IN1K`); a tag
        string → that specific checkpoint.  Mutually exclusive with
        ``weights`` (which wins if both are given).
    weights : PVTv2B0Weights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides : object
        Keyword overrides on top of the canonical PVT v2-B0 config.

    Returns
    -------
    PVTForImageClassification
        Classifier returning :class:`ImageClassificationOutput` whose
        ``logits`` has shape ``(B, num_classes)``.

    Notes
    -----
    PVT v2-B0 reaches **70.5% top-1 on ImageNet-1k** at 224x224 (Wang
    et al., 2022, Table 1).  Pretrained weights are converted from
    timm's ``pvt_v2_b0.in1k`` and hosted under ``lucid-dl/pvt-v2-b0``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.pvt import pvt_v2_b0_cls
    >>> model = pvt_v2_b0_cls(num_classes=1000)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model(x).logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(PVTv2B0Weights, pretrained, weights)
    model = PVTForImageClassification(apply_overrides(_CFG_B0, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="pvt_v2_b0_cls")
    return model


@register_model(
    task="image-classification",
    family="pvt",
    model_type="pvt",
    model_class=PVTForImageClassification,
    default_config=_CFG_B1,
)
def pvt_v2_b1_cls(
    pretrained: bool | str = False, *, weights: PVTv2B1Weights | None = None, **overrides: object
) -> PVTForImageClassification:
    r"""PVT v2-B1 image classifier (Wang et al., 2022).

    Combines the :func:`pvt_v2_b1` backbone with a mean pool + linear
    classification head.  ~14.0M parameters.

    Parameters
    ----------
    pretrained : bool or str, optional
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`PVTv2B1Weights.IN1K`); a tag
        string → that specific checkpoint.  Mutually exclusive with
        ``weights`` (which wins if both are given).
    weights : PVTv2B1Weights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides : object
        Keyword overrides on top of the canonical PVT v2-B1 config.

    Returns
    -------
    PVTForImageClassification
        Classifier whose ``logits`` has shape ``(B, num_classes)``.

    Notes
    -----
    PVT v2-B1 reaches **78.7% top-1 on ImageNet-1k** at 224x224 (Wang
    et al., 2022, Table 1).  Pretrained weights are converted from
    timm's ``pvt_v2_b1.in1k`` and hosted under ``lucid-dl/pvt-v2-b1``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.pvt import pvt_v2_b1_cls
    >>> model = pvt_v2_b1_cls(num_classes=1000)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model(x).logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(PVTv2B1Weights, pretrained, weights)
    model = PVTForImageClassification(apply_overrides(_CFG_B1, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="pvt_v2_b1_cls")
    return model


@register_model(
    task="image-classification",
    family="pvt",
    model_type="pvt",
    model_class=PVTForImageClassification,
    default_config=_CFG_B2,
)
def pvt_v2_b2_cls(
    pretrained: bool | str = False, *, weights: PVTv2B2Weights | None = None, **overrides: object
) -> PVTForImageClassification:
    r"""PVT v2-B2 image classifier (Wang et al., 2022).

    Combines the :func:`pvt_v2_b2` backbone with a mean pool + linear
    classification head.  ~25.4M parameters.

    Parameters
    ----------
    pretrained : bool or str, optional
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`PVTv2B2Weights.IN1K`); a tag
        string → that specific checkpoint.  Mutually exclusive with
        ``weights`` (which wins if both are given).
    weights : PVTv2B2Weights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides : object
        Keyword overrides on top of the canonical PVT v2-B2 config.

    Returns
    -------
    PVTForImageClassification
        Classifier whose ``logits`` has shape ``(B, num_classes)``.

    Notes
    -----
    PVT v2-B2 reaches **82.0% top-1 on ImageNet-1k** at 224x224 (Wang
    et al., 2022, Table 1).  Pretrained weights are converted from
    timm's ``pvt_v2_b2.in1k`` and hosted under ``lucid-dl/pvt-v2-b2``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.pvt import pvt_v2_b2_cls
    >>> model = pvt_v2_b2_cls(num_classes=1000)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model(x).logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(PVTv2B2Weights, pretrained, weights)
    model = PVTForImageClassification(apply_overrides(_CFG_B2, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="pvt_v2_b2_cls")
    return model


@register_model(
    task="image-classification",
    family="pvt",
    model_type="pvt",
    model_class=PVTForImageClassification,
    default_config=_CFG_B3,
)
def pvt_v2_b3_cls(
    pretrained: bool | str = False, *, weights: PVTv2B3Weights | None = None, **overrides: object
) -> PVTForImageClassification:
    r"""PVT v2-B3 image classifier (Wang et al., 2022).

    Combines the :func:`pvt_v2_b3` backbone with a mean pool + linear
    classification head.  ~45.2M parameters.

    Parameters
    ----------
    pretrained : bool or str, optional
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`PVTv2B3Weights.IN1K`); a tag
        string → that specific checkpoint.  Mutually exclusive with
        ``weights`` (which wins if both are given).
    weights : PVTv2B3Weights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides : object
        Keyword overrides on top of the canonical PVT v2-B3 config.

    Returns
    -------
    PVTForImageClassification
        Classifier whose ``logits`` has shape ``(B, num_classes)``.

    Notes
    -----
    PVT v2-B3 reaches **83.1% top-1 on ImageNet-1k** at 224x224 (Wang
    et al., 2022, Table 1).  Pretrained weights are converted from
    timm's ``pvt_v2_b3.in1k`` and hosted under ``lucid-dl/pvt-v2-b3``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.pvt import pvt_v2_b3_cls
    >>> model = pvt_v2_b3_cls(num_classes=1000)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model(x).logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(PVTv2B3Weights, pretrained, weights)
    model = PVTForImageClassification(apply_overrides(_CFG_B3, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="pvt_v2_b3_cls")
    return model


@register_model(
    task="image-classification",
    family="pvt",
    model_type="pvt",
    model_class=PVTForImageClassification,
    default_config=_CFG_B4,
)
def pvt_v2_b4_cls(
    pretrained: bool | str = False, *, weights: PVTv2B4Weights | None = None, **overrides: object
) -> PVTForImageClassification:
    r"""PVT v2-B4 image classifier (Wang et al., 2022).

    Combines the :func:`pvt_v2_b4` backbone with a mean pool + linear
    classification head.  ~62.6M parameters.

    Parameters
    ----------
    pretrained : bool or str, optional
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`PVTv2B4Weights.IN1K`); a tag
        string → that specific checkpoint.  Mutually exclusive with
        ``weights`` (which wins if both are given).
    weights : PVTv2B4Weights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides : object
        Keyword overrides on top of the canonical PVT v2-B4 config.

    Returns
    -------
    PVTForImageClassification
        Classifier whose ``logits`` has shape ``(B, num_classes)``.

    Notes
    -----
    PVT v2-B4 reaches **83.6% top-1 on ImageNet-1k** at 224x224 (Wang
    et al., 2022, Table 1).  Pretrained weights are converted from
    timm's ``pvt_v2_b4.in1k`` and hosted under ``lucid-dl/pvt-v2-b4``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.pvt import pvt_v2_b4_cls
    >>> model = pvt_v2_b4_cls(num_classes=1000)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model(x).logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(PVTv2B4Weights, pretrained, weights)
    model = PVTForImageClassification(apply_overrides(_CFG_B4, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="pvt_v2_b4_cls")
    return model


@register_model(
    task="image-classification",
    family="pvt",
    model_type="pvt",
    model_class=PVTForImageClassification,
    default_config=_CFG_B5,
)
def pvt_v2_b5_cls(
    pretrained: bool | str = False, *, weights: PVTv2B5Weights | None = None, **overrides: object
) -> PVTForImageClassification:
    r"""PVT v2-B5 image classifier (Wang et al., 2022).

    Combines the :func:`pvt_v2_b5` backbone with a mean pool + linear
    classification head.  ~82.0M parameters — the largest PVT v2
    variant.

    Parameters
    ----------
    pretrained : bool or str, optional
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`PVTv2B5Weights.IN1K`); a tag
        string → that specific checkpoint.  Mutually exclusive with
        ``weights`` (which wins if both are given).
    weights : PVTv2B5Weights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides : object
        Keyword overrides on top of the canonical PVT v2-B5 config.

    Returns
    -------
    PVTForImageClassification
        Classifier whose ``logits`` has shape ``(B, num_classes)``.

    Notes
    -----
    PVT v2-B5 reaches **83.8% top-1 on ImageNet-1k** at 224x224 (Wang
    et al., 2022, Table 1) — the headline result of the paper.
    Pretrained weights are converted from timm's ``pvt_v2_b5.in1k`` and
    hosted under ``lucid-dl/pvt-v2-b5``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.pvt import pvt_v2_b5_cls
    >>> model = pvt_v2_b5_cls(num_classes=1000)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model(x).logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(PVTv2B5Weights, pretrained, weights)
    model = PVTForImageClassification(apply_overrides(_CFG_B5, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="pvt_v2_b5_cls")
    return model
