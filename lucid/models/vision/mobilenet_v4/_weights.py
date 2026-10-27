"""Pretrained-weight declarations for the MobileNet v4 family.

All five paper variants (Qin et al., ECCV 2024) ship one ImageNet-1k
checkpoint each, converted from timm.  The authors have released no
weights, so these are timm's own training runs of the architecture —
recipes inspired by the paper's, not the paper's — and their accuracies
differ from Table 6.  Each tag is the timm pretrained tag, upper-cased:

==================  ======================  ================================
variant             tag                     choice
==================  ======================  ================================
Conv-Small          ``E2400_R224_IN1K``     timm's default
Conv-Medium         ``E500_R256_IN1K``      timm's default
Conv-Large          ``E600_R384_IN1K``      timm's default
Hybrid-Medium       ``IX_E550_R256_IN1K``   ImageNet-1k only, at the paper's
                                            256 (timm's default is
                                            ImageNet-12k pre-trained)
Hybrid-Large        ``IX_E600_R384_IN1K``   timm's default
==================  ======================  ================================

Every tag is ImageNet-1k-only and trained at the resolution the paper
uses for its variant.  Presets are the checkpoints' own eval pipelines:
the train resolution as the crop, ``crop_pct`` 0.875 (Conv-Small) or
0.95 (the rest) giving the resize, bicubic interpolation and ImageNet
mean/std.  The ``acc@1`` / ``acc@5`` figures are the rows of timm's
``results-imagenet.csv`` for each exact tag under that same pipeline;
timm also evaluates each tag at a larger test resolution, where it
scores higher, which these presets do not use.
"""

from lucid.utils.transforms import ImageClassification
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights


def _preset(crop: int, resize: int) -> ImageClassification:
    return ImageClassification(crop_size=crop, resize_size=resize, interpolation="bicubic")


@register_weights("mobilenet_v4_conv_small_cls")
class MobileNetV4ConvSmallWeights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.mobilenet_v4_conv_small_cls`.

    Qin et al. ECCV 2024 Conv-Small (3.8 M params, top-1 73.76%).

    Attributes
    ----------
    E2400_R224_IN1K : WeightEntry
        ImageNet-1k checkpoint trained 2400 epochs at 224x224 (top-1
        73.756% / top-5 91.430% at 224, per timm's
        ``results-imagenet.csv`` for this exact tag), sourced from
        ``timm/mobilenetv4_conv_small.e2400_r224_in1k``.
    DEFAULT : WeightEntry
        Alias for :attr:`E2400_R224_IN1K`.

    Notes
    -----
    Reference: Qin et al., *"MobileNetV4: Universal Models for the
    Mobile Ecosystem"*, ECCV 2024 (arXiv:2404.10518).

    Examples
    --------
    >>> from lucid.models.weights import MobileNetV4ConvSmallWeights
    >>> list(MobileNetV4ConvSmallWeights.__members__)
    ['E2400_R224_IN1K', 'DEFAULT']
    >>> tf = MobileNetV4ConvSmallWeights.DEFAULT.transforms()
    >>> tf.crop_size, tf.resize_size, tf.interpolation
    (224, 256, 'bicubic')
    """

    E2400_R224_IN1K = WeightEntry(
        url=(
            f"{HUB_BASE}/mobilenet-v4-conv-small/resolve/main/" "E2400_R224_IN1K/model.safetensors"
        ),
        sha256="dcf957719673adfa8c7a603fbfce50d6e3fe275e5a6be3a0379f1be6cfe517b6",
        num_classes=1000,
        transforms=_preset(224, 256),
        meta={
            "tag": "E2400_R224_IN1K",
            "source": "timm/mobilenetv4_conv_small.e2400_r224_in1k",
            "license": "apache-2.0",
            "num_params": 3_774_024,
            "metrics": {"ImageNet-1k": {"acc@1": 73.756, "acc@5": 91.430}},
        },
    )
    DEFAULT = E2400_R224_IN1K


@register_weights("mobilenet_v4_conv_medium_cls")
class MobileNetV4ConvMediumWeights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.mobilenet_v4_conv_medium_cls`.

    Qin et al. ECCV 2024 Conv-Medium (9.7 M params, top-1 79.92%).

    Attributes
    ----------
    E500_R256_IN1K : WeightEntry
        ImageNet-1k checkpoint trained 500 epochs at 256x256 (top-1
        79.916% / top-5 95.188% at 256, per timm's
        ``results-imagenet.csv`` for this exact tag), sourced from
        ``timm/mobilenetv4_conv_medium.e500_r256_in1k``.
    DEFAULT : WeightEntry
        Alias for :attr:`E500_R256_IN1K`.

    Notes
    -----
    Reference: Qin et al., *"MobileNetV4: Universal Models for the
    Mobile Ecosystem"*, ECCV 2024 (arXiv:2404.10518).

    Examples
    --------
    >>> from lucid.models.weights import MobileNetV4ConvMediumWeights
    >>> MobileNetV4ConvMediumWeights.DEFAULT is (
    ...     MobileNetV4ConvMediumWeights.E500_R256_IN1K
    ... )
    True
    >>> tf = MobileNetV4ConvMediumWeights.DEFAULT.transforms()
    >>> tf.crop_size, tf.resize_size
    (256, 269)
    """

    E500_R256_IN1K = WeightEntry(
        url=(
            f"{HUB_BASE}/mobilenet-v4-conv-medium/resolve/main/" "E500_R256_IN1K/model.safetensors"
        ),
        sha256="464e837e7d41ed0ec24bac989d056994cccacd212cc600d8bd886bafcb0d00a5",
        num_classes=1000,
        transforms=_preset(256, 269),
        meta={
            "tag": "E500_R256_IN1K",
            "source": "timm/mobilenetv4_conv_medium.e500_r256_in1k",
            "license": "apache-2.0",
            "num_params": 9_715_512,
            "metrics": {"ImageNet-1k": {"acc@1": 79.916, "acc@5": 95.188}},
        },
    )
    DEFAULT = E500_R256_IN1K


@register_weights("mobilenet_v4_conv_large_cls")
class MobileNetV4ConvLargeWeights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.mobilenet_v4_conv_large_cls`.

    Qin et al. ECCV 2024 Conv-Large (32.6 M params, top-1 82.97%).

    Attributes
    ----------
    E600_R384_IN1K : WeightEntry
        ImageNet-1k checkpoint trained 600 epochs at 384x384 (top-1
        82.974% / top-5 96.244% at 384, per timm's
        ``results-imagenet.csv`` for this exact tag), sourced from
        ``timm/mobilenetv4_conv_large.e600_r384_in1k``.
    DEFAULT : WeightEntry
        Alias for :attr:`E600_R384_IN1K`.

    Notes
    -----
    Reference: Qin et al., *"MobileNetV4: Universal Models for the
    Mobile Ecosystem"*, ECCV 2024 (arXiv:2404.10518).

    Examples
    --------
    >>> from lucid.models.weights import MobileNetV4ConvLargeWeights
    >>> tf = MobileNetV4ConvLargeWeights.DEFAULT.transforms()
    >>> tf.crop_size, tf.resize_size
    (384, 404)
    """

    E600_R384_IN1K = WeightEntry(
        url=(
            f"{HUB_BASE}/mobilenet-v4-conv-large/resolve/main/" "E600_R384_IN1K/model.safetensors"
        ),
        sha256="927da55b458d00a3438a396b9a68e1133c7c982a1d960a5882ec7160d6ae7809",
        num_classes=1000,
        transforms=_preset(384, 404),
        meta={
            "tag": "E600_R384_IN1K",
            "source": "timm/mobilenetv4_conv_large.e600_r384_in1k",
            "license": "apache-2.0",
            "num_params": 32_590_864,
            "metrics": {"ImageNet-1k": {"acc@1": 82.974, "acc@5": 96.244}},
        },
    )
    DEFAULT = E600_R384_IN1K


@register_weights("mobilenet_v4_hybrid_medium_cls")
class MobileNetV4HybridMediumWeights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.mobilenet_v4_hybrid_medium_cls`.

    Qin et al. ECCV 2024 Hybrid-Medium (11.1 M params, top-1 81.48%).

    Attributes
    ----------
    IX_E550_R256_IN1K : WeightEntry
        ImageNet-1k checkpoint trained 550 epochs at 256x256 (top-1
        81.478% / top-5 95.692% at 256, per timm's
        ``results-imagenet.csv`` for this exact tag), sourced from
        ``timm/mobilenetv4_hybrid_medium.ix_e550_r256_in1k``.  timm's
        own default tag for this variant is pre-trained on ImageNet-12k;
        this is its ImageNet-1k-only checkpoint at the paper's
        resolution.
    DEFAULT : WeightEntry
        Alias for :attr:`IX_E550_R256_IN1K`.

    Notes
    -----
    Reference: Qin et al., *"MobileNetV4: Universal Models for the
    Mobile Ecosystem"*, ECCV 2024 (arXiv:2404.10518).

    Examples
    --------
    >>> from lucid.models.weights import MobileNetV4HybridMediumWeights
    >>> list(MobileNetV4HybridMediumWeights.__members__)
    ['IX_E550_R256_IN1K', 'DEFAULT']
    >>> MobileNetV4HybridMediumWeights.DEFAULT.meta["metrics"]["ImageNet-1k"]
    {'acc@1': 81.478, 'acc@5': 95.692}
    """

    IX_E550_R256_IN1K = WeightEntry(
        url=(
            f"{HUB_BASE}/mobilenet-v4-hybrid-medium/resolve/main/"
            "IX_E550_R256_IN1K/model.safetensors"
        ),
        sha256="5d3ff6d83246a498d2624339c57ae76d9f5fcaeb4225ecc9b72bffaa82472b28",
        num_classes=1000,
        transforms=_preset(256, 269),
        meta={
            "tag": "IX_E550_R256_IN1K",
            "source": "timm/mobilenetv4_hybrid_medium.ix_e550_r256_in1k",
            "license": "apache-2.0",
            "num_params": 11_074_648,
            "metrics": {"ImageNet-1k": {"acc@1": 81.478, "acc@5": 95.692}},
        },
    )
    DEFAULT = IX_E550_R256_IN1K


@register_weights("mobilenet_v4_hybrid_large_cls")
class MobileNetV4HybridLargeWeights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.mobilenet_v4_hybrid_large_cls`.

    Qin et al. ECCV 2024 Hybrid-Large (37.8 M params, top-1 84.00%).

    Attributes
    ----------
    IX_E600_R384_IN1K : WeightEntry
        ImageNet-1k checkpoint trained 600 epochs at 384x384 (top-1
        83.996% / top-5 96.714% at 384, per timm's
        ``results-imagenet.csv`` for this exact tag), sourced from
        ``timm/mobilenetv4_hybrid_large.ix_e600_r384_in1k``.
    DEFAULT : WeightEntry
        Alias for :attr:`IX_E600_R384_IN1K`.

    Notes
    -----
    Reference: Qin et al., *"MobileNetV4: Universal Models for the
    Mobile Ecosystem"*, ECCV 2024 (arXiv:2404.10518).

    Examples
    --------
    >>> from lucid.models.weights import MobileNetV4HybridLargeWeights
    >>> tf = MobileNetV4HybridLargeWeights.DEFAULT.transforms()
    >>> tf.crop_size, tf.resize_size, tf.interpolation
    (384, 404, 'bicubic')
    """

    IX_E600_R384_IN1K = WeightEntry(
        url=(
            f"{HUB_BASE}/mobilenet-v4-hybrid-large/resolve/main/"
            "IX_E600_R384_IN1K/model.safetensors"
        ),
        sha256="489d73f6136a7d2199f3e152ed34329e83f62f3b718ad781a741c1f4b08b2213",
        num_classes=1000,
        transforms=_preset(384, 404),
        meta={
            "tag": "IX_E600_R384_IN1K",
            "source": "timm/mobilenetv4_hybrid_large.ix_e600_r384_in1k",
            "license": "apache-2.0",
            "num_params": 37_764_624,
            "metrics": {"ImageNet-1k": {"acc@1": 83.996, "acc@5": 96.714}},
        },
    )
    DEFAULT = IX_E600_R384_IN1K
