"""Registry factories for Inception-v4."""

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained
from lucid.models.vision.inception_v4._config import InceptionV4Config
from lucid.models.vision.inception_v4._model import InceptionV4, InceptionV4ForImageClassification
from lucid.models.vision.inception_v4._weights import InceptionV4Weights

_CFG = InceptionV4Config()


@register_model(
    task="base",
    family="inception_v4",
    model_type="inception_v4",
    model_class=InceptionV4,
    default_config=_CFG,
    params=41_142_816,
    summary="auto",
)
def inception_v4(pretrained: bool = False, **overrides: object) -> InceptionV4:
    r"""Inception-v4 feature-extracting backbone.

    Builds an :class:`InceptionV4` with the topology of Szegedy et al.
    2017, Figure 9: multi-branch stem → 4× Inception-A → Reduction-A →
    7× Inception-B → Reduction-B → 3× Inception-C, ending in an
    :math:`8\times8\times1536` feature map for a :math:`299\times299`
    input.  41.1 M parameters (the published classifier's 42.7 M less
    its 1536 → 1000 head).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No pretrained weights are published for the headless backbone;
        ``True`` raises :class:`NotImplementedError` rather than returning
        a randomly initialised model.  The ImageNet-1k checkpoint belongs
        to :func:`inception_v4_cls`, which the error message names.
    **overrides
        Keyword overrides forwarded into :class:`InceptionV4Config`
        (e.g. ``in_channels``).

    Returns
    -------
    InceptionV4
        Backbone with the Inception-v4 configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See Szegedy et al., "Inception-v4, Inception-ResNet and the Impact of
    Residual Connections on Learning", AAAI 2017.  The paper defines a
    single network, so there are no size variants (H11).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.inception_v4 import inception_v4
    >>> model = inception_v4().eval()
    >>> out = model(lucid.randn(1, 3, 299, 299))
    >>> out.last_hidden_state.shape
    (1, 1536, 8, 8)
    """
    if pretrained:
        reject_unavailable_pretrained("inception_v4", alternative="inception_v4_cls")
    cfg = apply_overrides(_CFG, overrides)
    return InceptionV4(cfg)


@register_model(
    task="image-classification",
    family="inception_v4",
    model_type="inception_v4",
    model_class=InceptionV4ForImageClassification,
    default_config=_CFG,
    params=42_679_816,
    summary="auto",
)
def inception_v4_cls(
    pretrained: bool | str = False,
    *,
    weights: InceptionV4Weights | None = None,
    **overrides: object,
) -> InceptionV4ForImageClassification:
    r"""Inception-v4 image classifier (trunk + GAP + dropout + linear).

    Builds an :class:`InceptionV4ForImageClassification` with the paper's
    topology and head — global average pool, dropout with keep
    probability 0.8, and a :class:`~lucid.nn.Linear` layer producing
    ``config.num_classes`` logits.  42.7 M parameters, matching the
    released TF-Slim ImageNet checkpoint.

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`InceptionV4Weights.TF_IN1K`); a tag
        string (e.g. ``"TF_IN1K"``) → that specific checkpoint.  Mutually
        exclusive with ``weights`` (which wins if both are given).
    weights : InceptionV4Weights, optional, keyword-only
        Explicit weights enum member, e.g. ``InceptionV4Weights.TF_IN1K``.
        Takes precedence over ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`InceptionV4Config`.
        Common picks: ``num_classes=N`` to retarget the head,
        ``dropout=p`` to change the head regularisation.  Overriding
        ``num_classes`` away from the checkpoint's 1000 makes pretrained
        loading fail the strict key/shape check — load with the matching
        head, then call :meth:`reset_classifier`.

    Returns
    -------
    InceptionV4ForImageClassification
        Classifier with the Inception-v4 configuration applied (or with
        ``overrides`` merged on top of it), optionally initialised from
        pretrained weights.

    Notes
    -----
    See Szegedy et al., "Inception-v4, Inception-ResNet and the Impact of
    Residual Connections on Learning", AAAI 2017.  Single-crop ImageNet
    validation error reported in Table 2: 20.0% top-1 / 5.0% top-5.  The
    head is named ``last_linear`` so the TF-Slim checkpoint's keys load
    unchanged.

    Pretrained weights are converted from timm's ``inception_v4.tf_in1k``
    — the TensorFlow-Slim ImageNet-1k checkpoint — and hosted on the
    Hugging Face Hub under ``lucid-dl/inception-v4``.  timm reports
    80.144% top-1 / 94.982% top-5 for it at 299x299 with the preset that
    :meth:`InceptionV4Weights.TF_IN1K.transforms` reproduces (299 crop,
    341 resize, bicubic, ``(0.5, 0.5, 0.5)`` mean/std — the TF-Slim
    :math:`[-1, 1]` scaling, not the ImageNet statistics).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.inception_v4 import inception_v4_cls
    >>> model = inception_v4_cls(num_classes=10).eval()
    >>> out = model(lucid.randn(2, 3, 299, 299))
    >>> out.logits.shape
    (2, 10)

    Load ImageNet-pretrained weights:

    >>> model = inception_v4_cls(pretrained=True)  # doctest: +SKIP
    >>> from lucid.models.weights import InceptionV4Weights
    >>> model = inception_v4_cls(weights=InceptionV4Weights.TF_IN1K)  # doctest: +SKIP
    """
    entry = weights_mod.resolve_weights(InceptionV4Weights, pretrained, weights)
    cfg = apply_overrides(_CFG, overrides)
    model = InceptionV4ForImageClassification(cfg)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="inception_v4_cls")
    return model
