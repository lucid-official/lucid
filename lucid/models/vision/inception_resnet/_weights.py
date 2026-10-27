"""Pretrained-weight declarations for the Inception-ResNet family.

One paper-cited variant (Szegedy et al., AAAI 2017) — sourced from
timm's ``tf_in1k`` checkpoint, the TensorFlow-Slim weights ported into
timm.  That preset is *not* the ImageNet default: it evaluates at
``299×299`` with ``crop_pct=0.8975`` (→ resize 333), **bicubic**
interpolation, and ``(0.5, 0.5, 0.5)`` mean/std (the TF-Slim ``[-1, 1]``
normalisation) — all replicated exactly in ``transforms`` below.
"""

from lucid.utils.transforms import ImageClassification
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights


@register_weights("inception_resnet_v2_cls")
class InceptionResNetV2Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.inception_resnet_v2_cls`.

    Szegedy et al. AAAI 2017 (55.8 M params, top-1 80.46%).

    Attributes
    ----------
    TF_IN1K : WeightEntry
        TensorFlow-Slim ImageNet-1k checkpoint (top-1 80.46% / top-5
        95.31%), sourced from ``timm/inception_resnet_v2.tf_in1k``.
    DEFAULT : WeightEntry
        Alias for :attr:`TF_IN1K`.

    Notes
    -----
    Reference: Szegedy, Ioffe, Vanhoucke, Alemi, *"Inception-v4,
    Inception-ResNet and the Impact of Residual Connections on
    Learning"*, AAAI 2017 (arXiv:1602.07261).

    Examples
    --------
    >>> from lucid.models import inception_resnet_v2_cls
    >>> model = inception_resnet_v2_cls(pretrained=True).eval()
    """

    TF_IN1K = WeightEntry(
        url=(f"{HUB_BASE}/inception-resnet-v2/resolve/main/" "TF_IN1K/model.safetensors"),
        sha256="d6125423dcc684daa2a422770ca8f00c091822ec36d14f98bfbdcd90e91740a0",
        num_classes=1000,
        transforms=ImageClassification(
            crop_size=299,
            resize_size=333,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            interpolation="bicubic",
        ),
        meta={
            "tag": "TF_IN1K",
            "source": "timm/inception_resnet_v2.tf_in1k",
            "license": "apache-2.0",
            "num_params": 55_843_464,
            "metrics": {"ImageNet-1k": {"acc@1": 80.46, "acc@5": 95.31}},
        },
    )
    DEFAULT = TF_IN1K
