"""Registry factories for ZFNet."""

from lucid.models._registry import register_model
from lucid.models.vision.zfnet._config import ZFNetConfig
from lucid.models.vision.zfnet._model import ZFNet, ZFNetForImageClassification
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained

# ---------------------------------------------------------------------------
# Canonical config
# ---------------------------------------------------------------------------

_CFG = ZFNetConfig()


# ---------------------------------------------------------------------------
# Backbone registration (task="base")
# ---------------------------------------------------------------------------


@register_model(
    task="base",
    family="zfnet",
    model_type="zfnet",
    model_class=ZFNet,
    default_config=_CFG,
    params=3726464,
)
def zfnet(pretrained: bool = False, **overrides: object) -> ZFNet:
    r"""ZFNet feature-extracting backbone (no fully-connected head).

    Builds a :class:`ZFNet` with the paper-cited Zeiler & Fergus 2014
    topology: five convolutional blocks with a tightened
    :math:`7\times7` stride-2 first convolution and a :math:`5\times5`
    stride-2 second convolution (versus AlexNet's :math:`11\times11`
    stride-4 and :math:`5\times5` stride-1).  Approximately 62 M
    parameters in the full classifier variant — the topology difference
    versus AlexNet does not change the headline parameter count.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No ZFNet weights are published.  Passing ``True`` raises
        :class:`NotImplementedError` rather than returning a randomly
        initialised model that looks pretrained.
    **overrides
        Keyword overrides forwarded into :class:`ZFNetConfig` (e.g.
        ``in_channels=1`` for grayscale inputs).

    Returns
    -------
    ZFNet
        Backbone with the ZFNet configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See Zeiler & Fergus, "Visualizing and Understanding Convolutional
    Networks", ECCV 2014.  The paper reports a top-5 ImageNet validation
    error of 16.5% for a single net (Table 2); §5.1 states this beats
    Krizhevsky's single-model test top-5 by 1.7%.  Achieved with
    essentially the same parameter count by retuning the first two conv
    layers based on deconvolutional-network visualisations.  ZFNet is a
    single architecture; there are no paper-cited size variants (H11).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.zfnet import zfnet
    >>> model = zfnet()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape   # (B, 256, 6, 6)
    (1, 256, 6, 6)
    """
    cfg = apply_overrides(_CFG, overrides)
    if pretrained:
        reject_unavailable_pretrained("zfnet")
    return ZFNet(cfg)


# ---------------------------------------------------------------------------
# Classification head registration (task="image-classification")
# ---------------------------------------------------------------------------


@register_model(
    task="image-classification",
    family="zfnet",
    model_type="zfnet",
    model_class=ZFNetForImageClassification,
    default_config=_CFG,
    params=62357608,
)
def zfnet_cls(pretrained: bool = False, **overrides: object) -> ZFNetForImageClassification:
    r"""ZFNet image classifier (backbone + FC6 + FC7 + linear head).

    Builds a :class:`ZFNetForImageClassification` with the paper-cited
    Zeiler & Fergus 2014 topology and the AlexNet-style two-layer
    4096-dim FC head.  Approximately 62 M parameters total.  Achieves
    a top-5 ImageNet-1k validation error of 16.5% (Table 2, single
    net) in the original
    paper.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``zfnet_cls``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides
        Keyword overrides forwarded into :class:`ZFNetConfig`.  Use
        ``num_classes=N`` to retarget the head and ``dropout=p`` to
        adjust regularisation.

    Returns
    -------
    ZFNetForImageClassification
        Classifier with the ZFNet configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See Zeiler & Fergus, "Visualizing and Understanding Convolutional
    Networks", ECCV 2014, §3.  The classifier head is identical to
    AlexNet's by design.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.zfnet import zfnet_cls
    >>> model = zfnet_cls()
    >>> x = lucid.randn(2, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (2, 1000)
    """
    cfg = apply_overrides(_CFG, overrides)
    if pretrained:
        reject_unavailable_pretrained("zfnet_cls")
    return ZFNetForImageClassification(cfg)
