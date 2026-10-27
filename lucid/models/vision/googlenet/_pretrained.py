"""Registry factories for GoogLeNet."""

from dataclasses import replace

import lucid.weights as weights_mod
import lucid.nn as nn
from lucid.models._registry import register_model
from lucid.models.vision.googlenet._config import GoogLeNetConfig
from lucid.models.vision.googlenet._model import GoogLeNet, GoogLeNetForImageClassification
from lucid.models.vision.googlenet._weights import GoogLeNetWeights
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained

_CFG = GoogLeNetConfig()
_CFG_NO_AUX = GoogLeNetConfig(aux_logits=False)


@register_model(
    task="base",
    family="googlenet",
    model_type="googlenet",
    model_class=GoogLeNet,
    default_config=_CFG,
)
def googlenet(pretrained: bool = False, **overrides: object) -> GoogLeNet:
    r"""GoogLeNet (Inception v1) feature-extracting backbone.

    Builds a :class:`GoogLeNet` with the paper-cited Szegedy 2015
    topology: a Conv-MaxPool stem followed by nine
    :class:`_InceptionModule` blocks at three resolutions
    (28×28 → 14×14 → 7×7) and a final
    :class:`~lucid.nn.AdaptiveAvgPool2d` to :math:`1\times1`.
    5.60 M parameters in the backbone; :func:`googlenet_cls` adds the
    1000-way head and the two auxiliary classifiers for 13.00 M total,
    or 6.62 M with ``aux_logits=False`` — the figure the reference
    publishes, since its builder drops the aux heads after loading.
    Either way, far fewer than AlexNet despite being substantially
    deeper.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``googlenet``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
        ``googlenet_cls`` is the checkpointed variant.
    **overrides
        Keyword overrides forwarded into :class:`GoogLeNetConfig`
        (``in_channels``, etc.).  Auxiliary-classifier fields are
        irrelevant for the backbone.

    Returns
    -------
    GoogLeNet
        Backbone with the GoogLeNet configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See Szegedy et al., "Going Deeper with Convolutions", CVPR 2015 —
    the ILSVRC-2014 classification winner with a top-5 ImageNet
    validation error of 6.67%.  Single architecture; no paper-cited
    "tiny / large" variants (H11).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.googlenet import googlenet
    >>> model = googlenet()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 1024, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained("googlenet", alternative="googlenet_cls")
    cfg = apply_overrides(_CFG, overrides)
    return GoogLeNet(cfg)


@register_model(
    task="image-classification",
    family="googlenet",
    model_type="googlenet",
    model_class=GoogLeNetForImageClassification,
    default_config=_CFG,
)
def googlenet_cls(
    pretrained: bool | str = False, *, weights: GoogLeNetWeights | None = None, **overrides: object
) -> GoogLeNetForImageClassification:
    r"""GoogLeNet (Inception v1) image classifier with auxiliary heads.

    Builds a :class:`GoogLeNetForImageClassification` with the
    paper-cited Szegedy 2015 configuration: 22-layer Inception backbone
    + global-average-pool + dropout (``p=0.4``) + linear projection to
    ``config.num_classes``, plus two auxiliary classifiers attached at
    Inception 4a and 4d (enabled by default via ``aux_logits=True``).
    Approximately 13.0 M parameters total when auxiliary heads are
    included.  Reaches **69.778% top-1 on ImageNet-1k**.

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`GoogLeNetWeights.IMAGENET1K_V1`);
        a tag string → that specific checkpoint.  Mutually exclusive
        with ``weights`` (which wins if both are given).
    weights : GoogLeNetWeights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`GoogLeNetConfig`.  Use
        ``aux_logits=False`` to disable auxiliary classifiers for
        cheaper inference graphs, ``num_classes=N`` to retarget the
        head, or ``dropout=p`` / ``aux_dropout=p`` to adjust
        regularisation.

    Returns
    -------
    GoogLeNetForImageClassification
        Classifier with the GoogLeNet configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See Szegedy et al., "Going Deeper with Convolutions", CVPR 2015,
    §5.  Auxiliary classifiers were introduced specifically to combat
    vanishing gradients in this 22-layer network without the benefit of
    residual connections; they contribute to the loss with weight
    :math:`0.3` each during training and are discarded at inference.
    Pretrained weights are converted from reference_vision's
    ``GoogLeNet_Weights.IMAGENET1K_V1`` and hosted under
    ``lucid-dl/googlenet``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.googlenet import googlenet_cls
    >>> model = googlenet_cls().eval()
    >>> x = lucid.randn(2, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (2, 1000)
    """
    entry = weights_mod.resolve_weights(GoogLeNetWeights, pretrained, weights)
    cfg = apply_overrides(_CFG, overrides)
    if entry is not None and "transform_input" not in overrides:
        # Matches the reference builder, which force-sets this whenever
        # ImageNet weights are requested; the checkpoint is a TF port that
        # expects (x-0.5)/0.5 inputs.  An explicit override still wins.
        cfg = replace(cfg, transform_input=True)
    # The checkpoint always carries aux1/aux2 tensors, so a caller asking for
    # ``aux_logits=False`` alongside ``pretrained`` used to hit a load failure
    # -- even though the docstring above recommends exactly that pairing for a
    # cheaper inference graph.  Build with the aux heads so the load has
    # somewhere to put them, then drop them, which is what the reference
    # builder does.
    drop_aux = entry is not None and not cfg.aux_logits
    build_cfg = replace(cfg, aux_logits=True) if drop_aux else cfg
    model = GoogLeNetForImageClassification(build_cfg)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="googlenet_cls")
    if drop_aux:
        model.aux1 = nn.Identity()
        model.aux2 = nn.Identity()
        model.config = cfg
    return model
