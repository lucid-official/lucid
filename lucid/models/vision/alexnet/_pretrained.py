"""Registry factories for AlexNet."""

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models.vision.alexnet._config import AlexNetConfig
from lucid.models.vision.alexnet._model import AlexNet, AlexNetForImageClassification
from lucid.models.vision.alexnet._weights import AlexNetWeights
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained

_CFG = AlexNetConfig()


@register_model(
    task="base",
    family="alexnet",
    model_type="alexnet",
    model_class=AlexNet,
    default_config=_CFG,
    params=2_469_696,  # conv trunk only (Krizhevsky 2014, single-stream OWT)
)
def alexnet(pretrained: bool = False, **overrides: object) -> AlexNet:
    r"""AlexNet feature-extracting backbone (no fully-connected head).

    Builds an :class:`AlexNet` with the Krizhevsky 2014 single-stream
    OWT topology: five convolutional blocks (Conv1: 3→64 with
    :math:`11\times11` stride-4; Conv2: 64→192 with :math:`5\times5`;
    Conv3: 192→384, Conv4: 384→256, Conv5: 256→256, all
    :math:`3\times3`), and overlapping :math:`3\times3` max-pools
    after blocks 1, 2, 5.  Approximately 2.5 M parameters in the
    convolutional trunk alone (≈61.1 M for the full classifier
    variant).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``alexnet``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``alexnet_cls`` is the checkpointed variant.
    **overrides
        Keyword overrides forwarded into :class:`AlexNetConfig` to
        customise individual fields (``in_channels=1`` for grayscale,
        ``dropout=0.0`` to disable the dropout layers of the
        classifier head).

    Returns
    -------
    AlexNet
        Backbone with the AlexNet configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    Krizhevsky et al., NIPS 2012 reports a 15.3% ImageNet-1k top-5
    validation error — more than ten percentage points below the
    runner-up — and is widely credited with re-igniting deep-learning
    research after the 2006-2011 lull.  Channel widths follow
    Krizhevsky 2014 ("One weird trick for parallelizing convolutional
    neural networks"), the single-stream re-derivation of the original
    two-GPU model, since that is the topology every published
    reference checkpoint targets.  No paper-cited "small / large"
    variants exist (H11) — AlexNet is a single architecture.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.alexnet import alexnet
    >>> model = alexnet()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape   # (B, 256, 6, 6)
    (1, 256, 6, 6)
    """
    if pretrained:
        reject_unavailable_pretrained("alexnet", alternative="alexnet_cls")
    cfg = apply_overrides(_CFG, overrides)
    return AlexNet(cfg)


@register_model(
    task="image-classification",
    family="alexnet",
    model_type="alexnet",
    model_class=AlexNetForImageClassification,
    default_config=_CFG,
    params=61_100_840,  # full classifier (Krizhevsky 2014, single-stream OWT)
)
def alexnet_cls(
    pretrained: bool | str = False, *, weights: AlexNetWeights | None = None, **overrides: object
) -> AlexNetForImageClassification:
    r"""AlexNet image classifier (backbone + FC6 + FC7 + linear head).

    Builds an :class:`AlexNetForImageClassification` with the
    Krizhevsky 2014 single-stream OWT topology: the five-block conv
    trunk followed by FC6 (9216 → 4096), FC7 (4096 → 4096), and a
    final linear projection to ``config.num_classes`` (default 1000
    for ImageNet-1k).  Approximately 61.1 M parameters total,
    dominated by the two 4096-dim FC layers (≈54.5 M).
    :class:`~lucid.nn.Dropout` with ``p=config.dropout`` (default 0.5)
    is applied after both hidden activations.

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`AlexNetWeights.IMAGENET1K_V1`);
        a tag string (e.g. ``"IMAGENET1K_V1"``) → that specific
        checkpoint.  Mutually exclusive with ``weights`` (which wins
        if both are given).
    weights : AlexNetWeights, optional, keyword-only
        Explicit weights enum member, e.g.
        ``AlexNetWeights.IMAGENET1K_V1``.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`AlexNetConfig`.  Use
        ``num_classes=N`` to retarget the classifier (e.g.
        ``num_classes=10`` for CIFAR-10 fine-tuning) and ``dropout=p``
        to adjust the regularisation strength.  Note: overriding
        ``num_classes`` away from the checkpoint's class count makes
        pretrained loading fail the strict key/shape check.

    Returns
    -------
    AlexNetForImageClassification
        Classifier with the AlexNet configuration applied (or with
        ``overrides`` merged on top of it), optionally initialised
        from pretrained weights.

    Notes
    -----
    The two :class:`~lucid.nn.Dropout` layers were a key empirical
    contribution of the original NIPS 2012 paper: setting half of each
    4096-dim activation to zero during training was the principal
    technique that prevented overfitting of a 60 M-parameter network
    on 1.2 M images.  Pretrained weights are converted from
    reference_vision's ``AlexNet_Weights.IMAGENET1K_V1`` and hosted on the
    Hugging Face Hub under ``lucid-dl/alexnet``.

    Examples
    --------
    Run a forward pass without labels:

    >>> import lucid
    >>> from lucid.models.vision.alexnet import alexnet_cls
    >>> model = alexnet_cls()
    >>> x = lucid.randn(2, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (2, 1000)

    Load ImageNet-pretrained weights:

    >>> model = alexnet_cls(pretrained=True)            # DEFAULT tag
    >>> from lucid.models.vision.alexnet import AlexNetWeights
    >>> model = alexnet_cls(weights=AlexNetWeights.IMAGENET1K_V1)
    """
    entry = weights_mod.resolve_weights(AlexNetWeights, pretrained, weights)
    cfg = apply_overrides(_CFG, overrides)
    model = AlexNetForImageClassification(cfg)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="alexnet_cls")
    return model
