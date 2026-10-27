"""Registry factories for ResNeSt."""

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models.vision.resnest._config import ResNeStConfig
from lucid.models.vision.resnest._model import (
    ResNeSt,
    ResNeStForImageClassification,
)
from lucid.models.vision.resnest._weights import (
    ResNeSt50Weights,
    ResNeSt101Weights,
    ResNeSt200Weights,
    ResNeSt269Weights,
)
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained

# H11 marker: `resnest_14` and `resnest_26` are **not paper variants**.
# Zhang et al. (arXiv:2004.08955) define exactly four depths — 50, 101, 200
# and 269 — and its Table 1 variant column is the radix/cardinality ablation
# (0s1x64d ... 2s2x40d), not a depth table; no (1,1,1,1) or (2,2,2,2)
# configuration appears anywhere in it.  These two are kept as small,
# fast-building stand-ins for tests and examples, and ship no weights.
# Anything claiming to reproduce the paper must use 50/101/200/269.
_CFG_14 = ResNeStConfig(layers=(1, 1, 1, 1), radix=2, stem_width=32)
_CFG_26 = ResNeStConfig(layers=(2, 2, 2, 2), radix=2, stem_width=32)
_CFG_50 = ResNeStConfig(layers=(3, 4, 6, 3), radix=2)
_CFG_101 = ResNeStConfig(layers=(3, 4, 23, 3), radix=2, stem_width=64)
# Section 4.2: "A dropout layer with the dropout probability of 0.2 is
# applied before the final fully-connected layer to the networks with more
# than 200 layers."  200 and 269 are the two that qualify.
_CFG_200 = ResNeStConfig(layers=(3, 24, 36, 3), radix=2, stem_width=64, dropout=0.2)
_CFG_269 = ResNeStConfig(layers=(3, 30, 48, 8), radix=2, stem_width=64, dropout=0.2)


# ── Backbones ─────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="resnest",
    model_type="resnest",
    model_class=ResNeSt,
    default_config=_CFG_14,
    params=8562688,
)
def resnest_14(pretrained: bool = False, **overrides: object) -> ResNeSt:
    r"""ResNeSt-14 feature-extracting backbone (no classification head).

    Builds a :class:`ResNeSt` with block repeats ``[1, 1, 1, 1]``
    over four stages — a lightweight variant of the ResNeSt family
    targeted at low-latency deployments.  Uses ``radix = 2`` and
    ``stem_width = 32``.  Approximately 10.6M parameters.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``resnest_14``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides
        Keyword overrides forwarded into :class:`ResNeStConfig`.

    Returns
    -------
    ResNeSt
        Backbone with the ResNeSt-14 configuration applied (or
        with ``overrides`` merged on top of it).

    Notes
    -----
    See Zhang et al., "ResNeSt: Split-Attention Networks", CVPR
    Workshops 2022 (arXiv:2004.08955).  Final-stage output is 2048
    channels.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnest import resnest_14
    >>> model = resnest_14()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 2048, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained("resnest_14")
    cfg = apply_overrides(_CFG_14, overrides)
    return ResNeSt(cfg)


@register_model(
    task="base",
    family="resnest",
    model_type="resnest",
    model_class=ResNeSt,
    default_config=_CFG_26,
    params=15020448,
)
def resnest_26(pretrained: bool = False, **overrides: object) -> ResNeSt:
    r"""ResNeSt-26 feature-extracting backbone (no classification head).

    Builds a :class:`ResNeSt` with block repeats ``[2, 2, 2, 2]``
    over four stages — a middle-ground variant between the
    lightweight ResNeSt-14 and the canonical ResNeSt-50.  Uses
    ``radix = 2`` and ``stem_width = 32``.  Approximately 17.1M
    parameters.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``resnest_26``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides
        Keyword overrides forwarded into :class:`ResNeStConfig`.

    Returns
    -------
    ResNeSt
        Backbone with the ResNeSt-26 configuration applied (or
        with ``overrides`` merged on top of it).

    Notes
    -----
    See Zhang et al., "ResNeSt: Split-Attention Networks", CVPR
    Workshops 2022 (arXiv:2004.08955).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnest import resnest_26
    >>> model = resnest_26()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 2048, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained("resnest_26")
    cfg = apply_overrides(_CFG_26, overrides)
    return ResNeSt(cfg)


@register_model(
    task="base",
    family="resnest",
    model_type="resnest",
    model_class=ResNeSt,
    default_config=_CFG_50,
    params=25434240,
)
def resnest_50(pretrained: bool = False, **overrides: object) -> ResNeSt:
    r"""ResNeSt-50 feature-extracting backbone (no classification head).

    Builds a :class:`ResNeSt` with ResNet-50 topology
    (``[3, 4, 6, 3]``), ``radix = 2``, and the canonical
    ``stem_width = 32`` deep stem.  Approximately 27.5M parameters
    and 81.1% ImageNet-1k top-1 accuracy in Zhang et al., 2022
    (Table 4) — roughly 5 points higher than plain ResNet-50 at
    the same parameter budget.  The default ResNeSt configuration.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``resnest_50``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``resnest_50_cls`` is the checkpointed variant.
    **overrides
        Keyword overrides forwarded into :class:`ResNeStConfig`
        (e.g. ``radix=4`` to widen the split-attention).

    Returns
    -------
    ResNeSt
        Backbone with the ResNeSt-50 configuration applied (or
        with ``overrides`` merged on top of it).

    Notes
    -----
    See Zhang et al., "ResNeSt: Split-Attention Networks", CVPR
    Workshops 2022 (arXiv:2004.08955), Table 4.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnest import resnest_50
    >>> model = resnest_50()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 2048, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained("resnest_50", alternative="resnest_50_cls")
    cfg = apply_overrides(_CFG_50, overrides)
    return ResNeSt(cfg)


@register_model(
    task="base",
    family="resnest",
    model_type="resnest",
    model_class=ResNeSt,
    default_config=_CFG_101,
    params=46226016,
)
def resnest_101(pretrained: bool = False, **overrides: object) -> ResNeSt:
    r"""ResNeSt-101 feature-extracting backbone (no classification head).

    Builds a :class:`ResNeSt` with ResNet-101 topology
    (``[3, 4, 23, 3]``), ``radix = 2``, and a wider
    ``stem_width = 64`` deep stem.  Approximately 48.3M parameters
    and 82.8% ImageNet-1k top-1 accuracy in Zhang et al., 2022
    (Table 4).  Higher-accuracy drop-in replacement for ResNeSt-50.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``resnest_101``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``resnest_101_cls`` is the checkpointed variant.
    **overrides
        Keyword overrides forwarded into :class:`ResNeStConfig`.

    Returns
    -------
    ResNeSt
        Backbone with the ResNeSt-101 configuration applied (or
        with ``overrides`` merged on top of it).

    Notes
    -----
    See Zhang et al., "ResNeSt: Split-Attention Networks", CVPR
    Workshops 2022 (arXiv:2004.08955), Table 4.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnest import resnest_101
    >>> model = resnest_101()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 2048, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained("resnest_101", alternative="resnest_101_cls")
    cfg = apply_overrides(_CFG_101, overrides)
    return ResNeSt(cfg)


@register_model(
    task="base",
    family="resnest",
    model_type="resnest",
    model_class=ResNeSt,
    default_config=_CFG_200,
    params=68152544,
)
def resnest_200(pretrained: bool = False, **overrides: object) -> ResNeSt:
    r"""ResNeSt-200 feature-extracting backbone (no classification head).

    Builds a :class:`ResNeSt` with the deep topology
    ``[3, 24, 36, 3]``, ``radix = 2``, and ``stem_width = 64``.
    Approximately 70.2M parameters and 83.9% ImageNet-1k top-1
    accuracy in Zhang et al., 2022 (Table 4).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``resnest_200``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``resnest_200_cls`` is the checkpointed variant.
    **overrides
        Keyword overrides forwarded into :class:`ResNeStConfig`.

    Returns
    -------
    ResNeSt
        Backbone with the ResNeSt-200 configuration applied (or
        with ``overrides`` merged on top of it).

    Notes
    -----
    See Zhang et al., "ResNeSt: Split-Attention Networks", CVPR
    Workshops 2022 (arXiv:2004.08955), Table 4.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnest import resnest_200
    >>> model = resnest_200()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 2048, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained("resnest_200", alternative="resnest_200_cls")
    cfg = apply_overrides(_CFG_200, overrides)
    return ResNeSt(cfg)


@register_model(
    task="base",
    family="resnest",
    model_type="resnest",
    model_class=ResNeSt,
    default_config=_CFG_269,
    params=108880480,
)
def resnest_269(pretrained: bool = False, **overrides: object) -> ResNeSt:
    r"""ResNeSt-269 feature-extracting backbone (no classification head).

    Builds a :class:`ResNeSt` with the very deep topology
    ``[3, 30, 48, 8]``, ``radix = 2``, and ``stem_width = 64`` —
    the deepest variant from Zhang et al., 2022.  Approximately
    110.9M parameters and 84.5% ImageNet-1k top-1 accuracy
    (Table 4) at 416×416 input resolution.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``resnest_269``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``resnest_269_cls`` is the checkpointed variant.
    **overrides
        Keyword overrides forwarded into :class:`ResNeStConfig`.

    Returns
    -------
    ResNeSt
        Backbone with the ResNeSt-269 configuration applied (or
        with ``overrides`` merged on top of it).

    Notes
    -----
    See Zhang et al., "ResNeSt: Split-Attention Networks", CVPR
    Workshops 2022 (arXiv:2004.08955), Table 4.  Memory footprint
    is substantial; consider ``zero_init_residual=True`` for
    stable large-batch training.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnest import resnest_269
    >>> model = resnest_269()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 2048, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained("resnest_269", alternative="resnest_269_cls")
    cfg = apply_overrides(_CFG_269, overrides)
    return ResNeSt(cfg)


# ── Classifiers ───────────────────────────────────────────────────────────────


@register_model(
    task="image-classification",
    family="resnest",
    model_type="resnest",
    model_class=ResNeStForImageClassification,
    default_config=_CFG_14,
    params=10611688,
)
def resnest_14_cls(pretrained: bool = False, **overrides: object) -> ResNeStForImageClassification:
    r"""ResNeSt-14 image classifier (backbone + GAP + linear head).

    Builds a :class:`ResNeStForImageClassification` with the
    ResNeSt-14 backbone (block repeats ``[1, 1, 1, 1]``) followed
    by global average pooling and a linear projection to
    ``config.num_classes`` (default 1000 for ImageNet-1k).
    Approximately 10.6M parameters — the lightweight variant of
    the ResNeSt family.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``resnest_14_cls``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides
        Keyword overrides forwarded into :class:`ResNeStConfig`.

    Returns
    -------
    ResNeStForImageClassification
        Classifier with the ResNeSt-14 configuration applied (or
        with ``overrides`` merged on top of it).

    Notes
    -----
    See Zhang et al., "ResNeSt: Split-Attention Networks", CVPR
    Workshops 2022 (arXiv:2004.08955).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnest import resnest_14_cls
    >>> model = resnest_14_cls(num_classes=10)
    >>> x = lucid.randn(2, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (2, 10)
    """
    if pretrained:
        reject_unavailable_pretrained("resnest_14_cls")
    cfg = apply_overrides(_CFG_14, overrides)
    return ResNeStForImageClassification(cfg)


@register_model(
    task="image-classification",
    family="resnest",
    model_type="resnest",
    model_class=ResNeStForImageClassification,
    default_config=_CFG_26,
    params=17069448,
)
def resnest_26_cls(pretrained: bool = False, **overrides: object) -> ResNeStForImageClassification:
    r"""ResNeSt-26 image classifier (backbone + GAP + linear head).

    Builds a :class:`ResNeStForImageClassification` with the
    ResNeSt-26 backbone (block repeats ``[2, 2, 2, 2]``) followed
    by global average pooling and a linear projection.
    Approximately 17.1M parameters.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``resnest_26_cls``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides
        Keyword overrides forwarded into :class:`ResNeStConfig`.

    Returns
    -------
    ResNeStForImageClassification
        Classifier with the ResNeSt-26 configuration applied (or
        with ``overrides`` merged on top of it).

    Notes
    -----
    See Zhang et al., "ResNeSt: Split-Attention Networks", CVPR
    Workshops 2022 (arXiv:2004.08955).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnest import resnest_26_cls
    >>> model = resnest_26_cls()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    if pretrained:
        reject_unavailable_pretrained("resnest_26_cls")
    cfg = apply_overrides(_CFG_26, overrides)
    return ResNeStForImageClassification(cfg)


@register_model(
    task="image-classification",
    family="resnest",
    model_type="resnest",
    model_class=ResNeStForImageClassification,
    default_config=_CFG_50,
    params=27483240,
)
def resnest_50_cls(
    pretrained: bool | str = False,
    *,
    weights: ResNeSt50Weights | None = None,
    **overrides: object,
) -> ResNeStForImageClassification:
    r"""ResNeSt-50 image classifier (backbone + GAP + linear head).

    Builds a :class:`ResNeStForImageClassification` with the
    canonical ResNeSt-50 backbone (``[3, 4, 6, 3]``, ``radix = 2``)
    followed by global average pooling and a linear projection.
    Approximately 27.5M parameters and 81.1% ImageNet-1k top-1
    accuracy in Zhang et al., 2022 (Table 4) — the canonical
    ResNeSt configuration.

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`ResNeSt50Weights.IN1K`); a tag
        string → that specific checkpoint.  Mutually exclusive with
        ``weights`` (which wins if both are given).
    weights : ResNeSt50Weights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`ResNeStConfig`
        (typically ``num_classes`` to retarget the classifier).

    Returns
    -------
    ResNeStForImageClassification
        Classifier with the ResNeSt-50 configuration applied (or
        with ``overrides`` merged on top of it).

    Notes
    -----
    See Zhang et al., "ResNeSt: Split-Attention Networks", CVPR
    Workshops 2022 (arXiv:2004.08955), Table 4.  Pretrained weights are
    converted from timm's ``resnest50d.in1k`` and hosted under
    ``lucid-dl/resnest-50``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnest import resnest_50_cls
    >>> model = resnest_50_cls(num_classes=10)
    >>> x = lucid.randn(2, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (2, 10)
    """
    entry = weights_mod.resolve_weights(ResNeSt50Weights, pretrained, weights)
    cfg = apply_overrides(_CFG_50, overrides)
    model = ResNeStForImageClassification(cfg)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="resnest_50_cls")
    return model


@register_model(
    task="image-classification",
    family="resnest",
    model_type="resnest",
    model_class=ResNeStForImageClassification,
    default_config=_CFG_101,
    params=48275016,
)
def resnest_101_cls(
    pretrained: bool | str = False,
    *,
    weights: ResNeSt101Weights | None = None,
    **overrides: object,
) -> ResNeStForImageClassification:
    r"""ResNeSt-101 image classifier (backbone + GAP + linear head).

    Builds a :class:`ResNeStForImageClassification` with the
    ResNeSt-101 backbone (``[3, 4, 23, 3]``, ``stem_width = 64``)
    followed by global average pooling and a linear classifier.
    Approximately 48.3M parameters and 82.8% ImageNet-1k top-1
    accuracy in Zhang et al., 2022 (Table 4).

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`ResNeSt101Weights.IN1K`); a tag
        string → that specific checkpoint.  Mutually exclusive with
        ``weights`` (which wins if both are given).
    weights : ResNeSt101Weights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`ResNeStConfig`.

    Returns
    -------
    ResNeStForImageClassification
        Classifier with the ResNeSt-101 configuration applied (or
        with ``overrides`` merged on top of it).

    Notes
    -----
    See Zhang et al., "ResNeSt: Split-Attention Networks", CVPR
    Workshops 2022 (arXiv:2004.08955), Table 4.  Pretrained weights are
    converted from timm's ``resnest101e.in1k`` and hosted under
    ``lucid-dl/resnest-101``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnest import resnest_101_cls
    >>> model = resnest_101_cls()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(ResNeSt101Weights, pretrained, weights)
    cfg = apply_overrides(_CFG_101, overrides)
    model = ResNeStForImageClassification(cfg)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="resnest_101_cls")
    return model


@register_model(
    task="image-classification",
    family="resnest",
    model_type="resnest",
    model_class=ResNeStForImageClassification,
    default_config=_CFG_200,
    params=70201544,
)
def resnest_200_cls(
    pretrained: bool | str = False,
    *,
    weights: ResNeSt200Weights | None = None,
    **overrides: object,
) -> ResNeStForImageClassification:
    r"""ResNeSt-200 image classifier (backbone + GAP + linear head).

    Builds a :class:`ResNeStForImageClassification` with the
    ResNeSt-200 backbone (``[3, 24, 36, 3]``, ``stem_width = 64``)
    followed by global average pooling and a linear classifier.
    Approximately 70.2M parameters and 83.9% ImageNet-1k top-1
    accuracy in Zhang et al., 2022 (Table 4).

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`ResNeSt200Weights.IN1K`); a tag
        string → that specific checkpoint.  Mutually exclusive with
        ``weights`` (which wins if both are given).
    weights : ResNeSt200Weights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`ResNeStConfig`.

    Returns
    -------
    ResNeStForImageClassification
        Classifier with the ResNeSt-200 configuration applied (or
        with ``overrides`` merged on top of it).

    Notes
    -----
    See Zhang et al., "ResNeSt: Split-Attention Networks", CVPR
    Workshops 2022 (arXiv:2004.08955), Table 4.  Pretrained weights are
    converted from timm's ``resnest200e.in1k`` and hosted under
    ``lucid-dl/resnest-200``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnest import resnest_200_cls
    >>> model = resnest_200_cls()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(ResNeSt200Weights, pretrained, weights)
    cfg = apply_overrides(_CFG_200, overrides)
    model = ResNeStForImageClassification(cfg)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="resnest_200_cls")
    return model


@register_model(
    task="image-classification",
    family="resnest",
    model_type="resnest",
    model_class=ResNeStForImageClassification,
    default_config=_CFG_269,
    params=110929480,
)
def resnest_269_cls(
    pretrained: bool | str = False,
    *,
    weights: ResNeSt269Weights | None = None,
    **overrides: object,
) -> ResNeStForImageClassification:
    r"""ResNeSt-269 image classifier (backbone + GAP + linear head).

    Builds a :class:`ResNeStForImageClassification` with the
    deepest paper-cited backbone (``[3, 30, 48, 8]``,
    ``stem_width = 64``) followed by global average pooling and a
    linear classifier.  Approximately 110.9M parameters and 84.5%
    ImageNet-1k top-1 accuracy in Zhang et al., 2022 (Table 4) at
    416×416 input resolution.

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`ResNeSt269Weights.IN1K`); a tag
        string → that specific checkpoint.  Mutually exclusive with
        ``weights`` (which wins if both are given).
    weights : ResNeSt269Weights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`ResNeStConfig`.

    Returns
    -------
    ResNeStForImageClassification
        Classifier with the ResNeSt-269 configuration applied (or
        with ``overrides`` merged on top of it).

    Notes
    -----
    See Zhang et al., "ResNeSt: Split-Attention Networks", CVPR
    Workshops 2022 (arXiv:2004.08955), Table 4.  Memory footprint
    is substantial.  Pretrained weights are converted from timm's
    ``resnest269e.in1k`` and hosted under ``lucid-dl/resnest-269``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnest import resnest_269_cls
    >>> model = resnest_269_cls()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(ResNeSt269Weights, pretrained, weights)
    cfg = apply_overrides(_CFG_269, overrides)
    model = ResNeStForImageClassification(cfg)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="resnest_269_cls")
    return model
