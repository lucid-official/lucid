"""Pretrained-weight declarations for the ConvNeXt family.

ImageNet-1k weights for the four canonical paper-cited variants
(``convnext_tiny`` / ``convnext_small`` / ``convnext_base`` /
``convnext_large``).  Converted from the reference vision library's
``ConvNeXt_*_Weights.IMAGENET1K_V1`` tag and re-hosted under the
``lucid-dl`` org so the per-family enum can pull a Lucid-native
SafeTensors blob without a reference dependency at load time.

Those are that library's *own retrained* weights, not a
redistribution of the original authors' checkpoints — its recipe and
reported accuracies differ from the paper's.  Attributing them to the
original lab also mis-states the licence they carry.

The ``convnext_xlarge`` variant (paper Table 9, 350M params) ships in
a follow-up commit sourced from timm's ``fb_in22k_ft_in1k`` checkpoint;
reference_vision does not publish a 1k-class xlarge head.
"""

from lucid.utils.transforms import ImageClassification
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights


@register_weights("convnext_tiny_cls")
class ConvNeXtTinyWeights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.convnext_tiny_cls`.

    Currently ships :attr:`IMAGENET1K_V1` — reference_vision's ImageNet-1k
    checkpoint (acc@1 = 82.520 / acc@5 = 96.146) re-hosted under
    ``huggingface.co/lucid-dl/convnext-tiny``.

    Examples
    --------
    >>> from lucid.models.vision.convnext._weights import ConvNeXtTinyWeights
    >>> list(ConvNeXtTinyWeights.__members__)
    ['IMAGENET1K_V1', 'DEFAULT']

    ``DEFAULT`` is an alias rather than a fourth entry, so a bare
    ``pretrained=True`` and the tag it resolves to cannot drift apart.

    >>> ConvNeXtTinyWeights.DEFAULT is ConvNeXtTinyWeights.IMAGENET1K_V1
    True
    >>> ConvNeXtTinyWeights.IMAGENET1K_V1.num_classes
    1000
    """

    IMAGENET1K_V1 = WeightEntry(
        url=f"{HUB_BASE}/convnext-tiny/resolve/main/IMAGENET1K_V1/model.safetensors",
        sha256="9bddab322a53aada74d717da131aba4c8fee460661209fe443c4bd743bd91c22",
        num_classes=1000,
        transforms=ImageClassification(crop_size=224, resize_size=236),
        meta={
            "tag": "IMAGENET1K_V1",
            "source": "reference_vision/ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
            "license": "bsd-3-clause",
            "num_params": 28_589_128,
            "gflops": 4.456,
            "file_size_mb": 109.1,
            "metrics": {"ImageNet-1k": {"acc@1": 82.520, "acc@5": 96.146}},
        },
    )
    DEFAULT = IMAGENET1K_V1


@register_weights("convnext_small_cls")
class ConvNeXtSmallWeights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.convnext_small_cls`.

    Currently ships :attr:`IMAGENET1K_V1` — reference_vision's ImageNet-1k
    checkpoint (acc@1 = 83.616 / acc@5 = 96.650).

    Examples
    --------
    >>> from lucid.models.vision.convnext._weights import ConvNeXtSmallWeights
    >>> list(ConvNeXtSmallWeights.__members__)
    ['IMAGENET1K_V1', 'DEFAULT']

    ``DEFAULT`` is an alias rather than a fourth entry, so a bare
    ``pretrained=True`` and the tag it resolves to cannot drift apart.

    >>> ConvNeXtSmallWeights.DEFAULT is ConvNeXtSmallWeights.IMAGENET1K_V1
    True
    >>> ConvNeXtSmallWeights.IMAGENET1K_V1.num_classes
    1000
    """

    IMAGENET1K_V1 = WeightEntry(
        url=f"{HUB_BASE}/convnext-small/resolve/main/IMAGENET1K_V1/model.safetensors",
        sha256="6412a3aab20fc5380288b68742a3e1b8015fcaadf0df3a0fc6ef3e4ab43f9212",
        num_classes=1000,
        transforms=ImageClassification(crop_size=224, resize_size=230),
        meta={
            "tag": "IMAGENET1K_V1",
            "source": "reference_vision/ConvNeXt_Small_Weights.IMAGENET1K_V1",
            "license": "bsd-3-clause",
            "num_params": 50_223_688,
            "gflops": 8.684,
            "file_size_mb": 191.6,
            "metrics": {"ImageNet-1k": {"acc@1": 83.616, "acc@5": 96.650}},
        },
    )
    DEFAULT = IMAGENET1K_V1


@register_weights("convnext_base_cls")
class ConvNeXtBaseWeights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.convnext_base_cls`.

    Currently ships :attr:`IMAGENET1K_V1` — reference_vision's ImageNet-1k
    checkpoint (acc@1 = 84.062 / acc@5 = 96.870).

    Examples
    --------
    >>> from lucid.models.vision.convnext._weights import ConvNeXtBaseWeights
    >>> list(ConvNeXtBaseWeights.__members__)
    ['IMAGENET1K_V1', 'DEFAULT']

    ``DEFAULT`` is an alias rather than a fourth entry, so a bare
    ``pretrained=True`` and the tag it resolves to cannot drift apart.

    >>> ConvNeXtBaseWeights.DEFAULT is ConvNeXtBaseWeights.IMAGENET1K_V1
    True
    >>> ConvNeXtBaseWeights.IMAGENET1K_V1.num_classes
    1000
    """

    IMAGENET1K_V1 = WeightEntry(
        url=f"{HUB_BASE}/convnext-base/resolve/main/IMAGENET1K_V1/model.safetensors",
        sha256="4c2c3389b534a134ed9c2531e9b89963aa0c8ab1484ce77b364d68eda437fcb7",
        num_classes=1000,
        transforms=ImageClassification(crop_size=224, resize_size=232),
        meta={
            "tag": "IMAGENET1K_V1",
            "source": "reference_vision/ConvNeXt_Base_Weights.IMAGENET1K_V1",
            "license": "bsd-3-clause",
            "num_params": 88_591_464,
            "gflops": 15.355,
            "file_size_mb": 338.0,
            "metrics": {"ImageNet-1k": {"acc@1": 84.062, "acc@5": 96.870}},
        },
    )
    DEFAULT = IMAGENET1K_V1


@register_weights("convnext_xlarge_cls")
class ConvNeXtXLargeWeights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.convnext_xlarge_cls`.

    Currently ships :attr:`FB_IN22K_FT_IN1K` — Facebook AI Research's
    ImageNet-22k pretraining → ImageNet-1k finetune checkpoint
    (paper Table 11, acc@1 ≈ 87.0%), sourced via timm's
    ``convnext_xlarge.fb_in22k_ft_in1k`` and re-hosted under
    ``huggingface.co/lucid-dl/convnext-xlarge``.  reference_vision does not
    publish a 1k-class ConvNeXt-XLarge head.

    Examples
    --------
    >>> from lucid.models.vision.convnext._weights import ConvNeXtXLargeWeights
    >>> list(ConvNeXtXLargeWeights.__members__)
    ['FB_IN22K_FT_IN1K', 'DEFAULT']

    ``DEFAULT`` is an alias rather than a fourth entry, so a bare
    ``pretrained=True`` and the tag it resolves to cannot drift apart.

    >>> ConvNeXtXLargeWeights.DEFAULT is ConvNeXtXLargeWeights.FB_IN22K_FT_IN1K
    True
    >>> ConvNeXtXLargeWeights.FB_IN22K_FT_IN1K.num_classes
    1000
    """

    FB_IN22K_FT_IN1K = WeightEntry(
        url=f"{HUB_BASE}/convnext-xlarge/resolve/main/FB_IN22K_FT_IN1K/model.safetensors",
        sha256="6c94f07df6893351e9d9b1079bca7e923eb1196c215f6649338f7ccad561a1fd",
        num_classes=1000,
        transforms=ImageClassification(crop_size=224, resize_size=256, interpolation="bicubic"),
        meta={
            "tag": "FB_IN22K_FT_IN1K",
            "source": "timm/convnext_xlarge.fb_in22k_ft_in1k",
            "license": "apache-2.0",
            "num_params": 350_196_968,
            "gflops": 60.98,
            "file_size_mb": 1336.6,
            "metrics": {"ImageNet-1k": {"acc@1": 87.0, "acc@5": 98.2}},
        },
    )
    DEFAULT = FB_IN22K_FT_IN1K


@register_weights("convnext_large_cls")
class ConvNeXtLargeWeights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.convnext_large_cls`.

    Currently ships :attr:`IMAGENET1K_V1` — reference_vision's ImageNet-1k
    checkpoint (acc@1 = 84.414 / acc@5 = 96.976).

    Examples
    --------
    >>> from lucid.models.vision.convnext._weights import ConvNeXtLargeWeights
    >>> list(ConvNeXtLargeWeights.__members__)
    ['IMAGENET1K_V1', 'DEFAULT']

    ``DEFAULT`` is an alias rather than a fourth entry, so a bare
    ``pretrained=True`` and the tag it resolves to cannot drift apart.

    >>> ConvNeXtLargeWeights.DEFAULT is ConvNeXtLargeWeights.IMAGENET1K_V1
    True
    >>> ConvNeXtLargeWeights.IMAGENET1K_V1.num_classes
    1000
    """

    IMAGENET1K_V1 = WeightEntry(
        url=f"{HUB_BASE}/convnext-large/resolve/main/IMAGENET1K_V1/model.safetensors",
        sha256="4f335d8136015448a3c62f58d4493a2dcc4adc5102e1c23dd6d3f57b7d695d26",
        num_classes=1000,
        transforms=ImageClassification(crop_size=224, resize_size=232),
        meta={
            "tag": "IMAGENET1K_V1",
            "source": "reference_vision/ConvNeXt_Large_Weights.IMAGENET1K_V1",
            "license": "bsd-3-clause",
            "num_params": 197_767_336,
            "gflops": 34.361,
            "file_size_mb": 754.4,
            "metrics": {"ImageNet-1k": {"acc@1": 84.414, "acc@5": 96.976}},
        },
    )
    DEFAULT = IMAGENET1K_V1
