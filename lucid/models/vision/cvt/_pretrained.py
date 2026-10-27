"""Registry factories for CvT variants."""

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models.vision.cvt._config import CvTConfig
from lucid.models.vision.cvt._model import CvT, CvTForImageClassification
from lucid.models.vision.cvt._weights import CvT13Weights, CvT21Weights, CvTW24Weights
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained

# ---------------------------------------------------------------------------
# Canonical configs
# ---------------------------------------------------------------------------

_CFG_13 = CvTConfig(
    variant="cvt_13",
    dims=(64, 192, 384),
    depths=(1, 2, 10),
    num_heads=(1, 3, 6),
    embed_strides=(4, 2, 2),
    drop_path_rate=0.1,  # Table 1
)

_CFG_21 = CvTConfig(
    variant="cvt_21",
    dims=(64, 192, 384),
    depths=(1, 4, 16),
    num_heads=(1, 3, 6),
    embed_strides=(4, 2, 2),
    drop_path_rate=0.1,  # Table 1
)

_CFG_W24 = CvTConfig(
    variant="cvt_w24",
    dims=(192, 768, 1024),
    depths=(2, 2, 20),
    num_heads=(3, 12, 16),
    embed_strides=(4, 2, 2),
    drop_path_rate=0.3,  # Table 1 (wide variant)
)

# ---------------------------------------------------------------------------
# Backbone registrations (task="base")
# ---------------------------------------------------------------------------


@register_model(
    task="base", family="cvt", model_type="cvt", model_class=CvT, default_config=_CFG_13
)
def cvt_13(pretrained: bool = False, **overrides: object) -> CvT:
    r"""CvT-13 backbone (Wu et al., 2021).

    Builds the canonical *CvT-13* configuration: ``dims=(64, 192, 384)``,
    ``depths=(1, 2, 10)``, ``num_heads=(1, 3, 6)``,
    ``embed_strides=(4, 2, 2)``.  Total spatial downsampling is
    :math:`4 \times 2 \times 2 = 16\times`, so a 224x224 input yields
    a 14x14 final token grid.  Approximately **20M parameters**.

    Parameters
    ----------
    pretrained : bool, optional
        No checkpoint is published for ``cvt_13``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``cvt_13_cls`` is the checkpointed variant.
    **overrides : object
        Keyword overrides applied on top of the canonical CvT-13
        config.  Each override must match a field of :class:`CvTConfig`.

    Returns
    -------
    CvT
        A :class:`CvT` backbone returning a flat :math:`(B, 384)`
        mean-pooled feature.

    Notes
    -----
    CvT-13 reaches **81.6% top-1 on ImageNet-1k** at 224x224 (Wu et al.,
    2021, Table 1).  See
    `arXiv:2103.15808 <https://arxiv.org/abs/2103.15808>`_.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.cvt import cvt_13
    >>> model = cvt_13()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> feat = model.forward_features(x)
    >>> feat.shape
    (1, 384)
    """
    if pretrained:
        reject_unavailable_pretrained("cvt_13", alternative="cvt_13_cls")
    cfg = apply_overrides(_CFG_13, overrides)
    return CvT(cfg)


@register_model(
    task="base", family="cvt", model_type="cvt", model_class=CvT, default_config=_CFG_21
)
def cvt_21(pretrained: bool = False, **overrides: object) -> CvT:
    r"""CvT-21 backbone (Wu et al., 2021).

    Builds the canonical *CvT-21* configuration: same widths and head
    counts as CvT-13 (``dims=(64, 192, 384)``, ``num_heads=(1, 3, 6)``)
    but deeper — ``depths=(1, 4, 16)``.  Approximately **31.6M
    parameters**.

    Parameters
    ----------
    pretrained : bool, optional
        No checkpoint is published for ``cvt_21``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``cvt_21_cls`` is the checkpointed variant.
    **overrides : object
        Keyword overrides on top of the canonical CvT-21 config.

    Returns
    -------
    CvT
        A :class:`CvT` backbone returning a flat :math:`(B, 384)`
        mean-pooled feature.

    Notes
    -----
    CvT-21 reaches **82.5% top-1 on ImageNet-1k** at 224x224 (Wu et al.,
    2021, Table 1).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.cvt import cvt_21
    >>> model = cvt_21()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model.forward_features(x).shape
    (1, 384)
    """
    if pretrained:
        reject_unavailable_pretrained("cvt_21", alternative="cvt_21_cls")
    cfg = apply_overrides(_CFG_21, overrides)
    return CvT(cfg)


@register_model(
    task="base", family="cvt", model_type="cvt", model_class=CvT, default_config=_CFG_W24
)
def cvt_w24(pretrained: bool = False, **overrides: object) -> CvT:
    r"""CvT-W24 wide backbone (Wu et al., 2021).

    Builds the largest reference configuration *CvT-W24*:
    ``dims=(192, 768, 1024)``, ``depths=(2, 2, 20)``,
    ``num_heads=(3, 12, 16)``.  Roughly an order of magnitude wider
    than CvT-13.  Approximately **277.2M parameters** — typically used
    only with large-scale (ImageNet-22k) pretraining.

    Parameters
    ----------
    pretrained : bool, optional
        No checkpoint is published for ``cvt_w24``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``cvt_w24_cls`` is the checkpointed variant.
    **overrides : object
        Keyword overrides on top of the canonical CvT-W24 config.

    Returns
    -------
    CvT
        A :class:`CvT` backbone returning a flat :math:`(B, 1024)`
        mean-pooled feature.

    Notes
    -----
    CvT-W24 reaches **87.7% top-1 on ImageNet-1k** at 384x384 fine-tune
    resolution after ImageNet-22k pretraining (Wu et al., 2021,
    Table 4) — the headline result of the paper.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.cvt import cvt_w24
    >>> model = cvt_w24()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model.forward_features(x).shape
    (1, 1024)
    """
    if pretrained:
        reject_unavailable_pretrained("cvt_w24", alternative="cvt_w24_cls")
    cfg = apply_overrides(_CFG_W24, overrides)
    return CvT(cfg)


# ---------------------------------------------------------------------------
# Classification head registrations (task="image-classification")
# ---------------------------------------------------------------------------


@register_model(
    task="image-classification",
    family="cvt",
    model_type="cvt",
    model_class=CvTForImageClassification,
    default_config=_CFG_13,
)
def cvt_13_cls(
    pretrained: bool | str = False, *, weights: CvT13Weights | None = None, **overrides: object
) -> CvTForImageClassification:
    r"""CvT-13 image classifier (Wu et al., 2021).

    Combines the :func:`cvt_13` backbone with a mean pool + LayerNorm
    + single :class:`nn.Linear` classification head.  Default output is
    ``num_classes=1000`` (ImageNet-1k).  Approximately **20M parameters**.

    Parameters
    ----------
    pretrained : bool, optional
        If ``True``, loads ImageNet-1k pretrained weights when
        available.  Defaults to ``False``.
    weights : CvT13Weights, optional, keyword-only
        Explicit weights enum member; takes precedence over ``pretrained``.
    **overrides : object
        Keyword overrides on top of the canonical CvT-13 config.

    Returns
    -------
    CvTForImageClassification
        Classifier returning :class:`ImageClassificationOutput` whose
        ``logits`` has shape ``(B, num_classes)``.

    Notes
    -----
    CvT-13 reaches **81.6% top-1 on ImageNet-1k** at 224x224 (Wu et al.,
    2021, Table 1).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.cvt import cvt_13_cls
    >>> model = cvt_13_cls(num_classes=1000)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model(x).logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(CvT13Weights, pretrained, weights)
    cfg = apply_overrides(_CFG_13, overrides)
    model = CvTForImageClassification(cfg)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="cvt_13_cls")
    return model


@register_model(
    task="image-classification",
    family="cvt",
    model_type="cvt",
    model_class=CvTForImageClassification,
    default_config=_CFG_21,
)
def cvt_21_cls(
    pretrained: bool | str = False, *, weights: CvT21Weights | None = None, **overrides: object
) -> CvTForImageClassification:
    r"""CvT-21 image classifier (Wu et al., 2021).

    Combines the :func:`cvt_21` backbone (``dims=(64, 192, 384)``,
    ``depths=(1, 4, 16)``) with a mean pool + LayerNorm + linear
    classification head.  Approximately **31.6M parameters**.

    Parameters
    ----------
    pretrained : bool, optional
        If ``True``, loads ImageNet-1k pretrained weights when
        available.  Defaults to ``False``.
    weights : CvT21Weights, optional, keyword-only
        Explicit weights enum member; takes precedence over ``pretrained``.
    **overrides : object
        Keyword overrides on top of the canonical CvT-21 config.

    Returns
    -------
    CvTForImageClassification
        Classifier whose ``logits`` has shape ``(B, num_classes)``.

    Notes
    -----
    CvT-21 reaches **82.5% top-1 on ImageNet-1k** at 224x224 (Wu et al.,
    2021, Table 1).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.cvt import cvt_21_cls
    >>> model = cvt_21_cls(num_classes=1000)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model(x).logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(CvT21Weights, pretrained, weights)
    cfg = apply_overrides(_CFG_21, overrides)
    model = CvTForImageClassification(cfg)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="cvt_21_cls")
    return model


@register_model(
    task="image-classification",
    family="cvt",
    model_type="cvt",
    model_class=CvTForImageClassification,
    default_config=_CFG_W24,
)
def cvt_w24_cls(
    pretrained: bool | str = False, *, weights: CvTW24Weights | None = None, **overrides: object
) -> CvTForImageClassification:
    r"""CvT-W24 wide image classifier (Wu et al., 2021).

    Combines the :func:`cvt_w24` backbone (``dims=(192, 768, 1024)``,
    ``depths=(2, 2, 20)``) with a mean pool + LayerNorm + linear
    classification head.  Approximately **277.2M parameters** — the
    largest CvT variant in the paper, typically used only with
    ImageNet-22k pretraining.

    Parameters
    ----------
    pretrained : bool, optional
        If ``True``, loads ImageNet-22k pretrained weights when
        available.  Defaults to ``False``.
    weights : CvTW24Weights, optional, keyword-only
        Explicit weights enum member; takes precedence over ``pretrained``.
    **overrides : object
        Keyword overrides on top of the canonical CvT-W24 config.

    Returns
    -------
    CvTForImageClassification
        Classifier whose ``logits`` has shape ``(B, num_classes)``.

    Notes
    -----
    CvT-W24 reaches **87.7% top-1 on ImageNet-1k** at 384x384
    fine-tune resolution after ImageNet-22k pretraining (Wu et al.,
    2021, Table 4) — the headline result of the paper.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.cvt import cvt_w24_cls
    >>> model = cvt_w24_cls(num_classes=1000)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model(x).logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(CvTW24Weights, pretrained, weights)
    cfg = apply_overrides(_CFG_W24, overrides)
    model = CvTForImageClassification(cfg)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="cvt_w24_cls")
    return model
