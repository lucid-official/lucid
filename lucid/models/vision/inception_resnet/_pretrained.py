"""Registry factories for Inception-ResNet v2."""

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models.vision.inception_resnet._config import InceptionResNetConfig
from lucid.models.vision.inception_resnet._model import (
    InceptionResNetV2,
    InceptionResNetV2ForImageClassification,
)
from lucid.models.vision.inception_resnet._weights import InceptionResNetV2Weights
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained

_CFG = InceptionResNetConfig()


@register_model(
    task="base",
    family="inception_resnet",
    model_type="inception_resnet",
    model_class=InceptionResNetV2,
    default_config=_CFG,
)
def inception_resnet_v2(pretrained: bool = False, **overrides: object) -> InceptionResNetV2:
    r"""Inception-ResNet v2 feature-extracting backbone.

    Builds an :class:`InceptionResNetV2` with the paper-cited Szegedy
    2017 topology: 5-conv stem → Mixed_5b → 10× Block35 → Reduction-A
    → 20× Block17 → Reduction-B → 9× Block8 (+ 1 no-ReLU block) →
    :math:`1\times1` projection to 1536 channels.  Designed for
    :math:`299\times299` RGB inputs.  Approximately 54.3 M parameters
    in the backbone alone.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``inception_resnet_v2``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``inception_resnet_v2_cls`` is the checkpointed variant.
    **overrides
        Keyword overrides forwarded into :class:`InceptionResNetConfig`.
        Use ``scale_a`` / ``scale_b`` / ``scale_c`` to tune the
        residual-branch scaling factors (defaults 0.17 / 0.10 / 0.20
        from the paper).

    Returns
    -------
    InceptionResNetV2
        Backbone with the Inception-ResNet v2 configuration applied
        (or with ``overrides`` merged on top of it).

    Notes
    -----
    See Szegedy et al., "Inception-v4, Inception-ResNet and the Impact
    of Residual Connections on Learning", AAAI 2017.  Top-5 ImageNet
    validation error of 3.08% (full classifier).  No paper-cited size
    variants (H11) — only v2 is implemented here; v1 differs primarily
    in widths.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.inception_resnet import inception_resnet_v2
    >>> model = inception_resnet_v2()
    >>> x = lucid.randn(1, 3, 299, 299)
    >>> out = model(x)
    >>> out.last_hidden_state.shape   # (B, 1536, 1, 1)
    (1, 1536, 8, 8)
    """
    if pretrained:
        reject_unavailable_pretrained("inception_resnet_v2", alternative="inception_resnet_v2_cls")
    cfg = apply_overrides(_CFG, overrides)
    return InceptionResNetV2(cfg)


@register_model(
    task="image-classification",
    family="inception_resnet",
    model_type="inception_resnet",
    model_class=InceptionResNetV2ForImageClassification,
    default_config=_CFG,
)
def inception_resnet_v2_cls(
    pretrained: bool | str = False,
    *,
    weights: InceptionResNetV2Weights | None = None,
    **overrides: object,
) -> InceptionResNetV2ForImageClassification:
    r"""Inception-ResNet v2 image classifier (backbone + GAP + dropout + linear).

    Builds an :class:`InceptionResNetV2ForImageClassification` with the
    paper-cited Szegedy 2017 topology and a single
    :class:`~lucid.nn.Linear` classifier head producing
    ``config.num_classes`` logits.  Approximately 55.8 M parameters
    total; achieves a top-5 ImageNet validation error of 3.08%.

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`InceptionResNetV2Weights.TF_IN1K`);
        a tag string (e.g. ``"TF_IN1K"``) → that specific checkpoint.
        Mutually exclusive with ``weights`` (which wins if both are
        given).
    weights : InceptionResNetV2Weights, optional, keyword-only
        Explicit weights enum member, e.g.
        ``InceptionResNetV2Weights.TF_IN1K``.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`InceptionResNetConfig`.
        Common picks: ``num_classes=N`` to retarget the head,
        ``dropout=p`` to adjust regularisation.  Note: overriding
        ``num_classes`` away from the checkpoint's class count makes
        pretrained loading fail the strict key/shape check — load with a
        matching head, then call :meth:`reset_classifier`.

    Returns
    -------
    InceptionResNetV2ForImageClassification
        Classifier with the Inception-ResNet v2 configuration applied
        (or with ``overrides`` merged on top of it), optionally
        initialised from pretrained weights.

    Notes
    -----
    See Szegedy et al., "Inception-v4, Inception-ResNet and the Impact
    of Residual Connections on Learning", AAAI 2017, §3.3.  The
    classifier attribute is named ``classif`` (not ``classifier``) for
    timm / TensorFlow-Slim state-dict compatibility.  Pretrained weights
    are converted from timm's ``inception_resnet_v2.tf_in1k`` checkpoint
    and hosted on the Hugging Face Hub under
    ``lucid-dl/inception-resnet-v2``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.inception_resnet import inception_resnet_v2_cls
    >>> model = inception_resnet_v2_cls()
    >>> x = lucid.randn(2, 3, 299, 299)
    >>> out = model(x)
    >>> out.logits.shape
    (2, 1000)

    Load ImageNet-pretrained weights:

    >>> model = inception_resnet_v2_cls(pretrained=True)        # DEFAULT tag
    >>> from lucid.models.vision.inception_resnet import InceptionResNetV2Weights
    >>> model = inception_resnet_v2_cls(weights=InceptionResNetV2Weights.TF_IN1K)
    """
    entry = weights_mod.resolve_weights(InceptionResNetV2Weights, pretrained, weights)
    cfg = apply_overrides(_CFG, overrides)
    model = InceptionResNetV2ForImageClassification(cfg)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="inception_resnet_v2_cls")
    return model
