"""Pretrained-weight declarations for the EfficientNet family.

Eight paper-cited variants (Tan & Le, ICML 2019) — all converted from
the reference-framework model zoo's ``EfficientNet_B{0..7}_Weights``
``IMAGENET1K_V1`` tag.  Unlike most ConvNets, every B-variant uses its
own native evaluation resolution and (for the smaller variants) a tight
resize→crop ratio; all use **bicubic** interpolation and the standard
ImageNet channel statistics.  The per-variant eval pipeline mirrors the
upstream presets exactly:

==========  =========  ===========  ==============
variant     crop size  resize size  interpolation
==========  =========  ===========  ==============
B0          224        256          bicubic
B1          240        256          bicubic
B2          288        288          bicubic
B3          300        320          bicubic
B4          380        384          bicubic
B5          456        456          bicubic
B6          528        528          bicubic
B7          600        600          bicubic
==========  =========  ===========  ==============

B1 ships a stronger ``IMAGENET1K_V2`` checkpoint upstream, but ``V1`` is
used as the :attr:`DEFAULT` for consistency across the family.
"""

from lucid.utils.transforms import ImageClassification
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights

_PRESET_B0 = ImageClassification(crop_size=224, resize_size=256, interpolation="bicubic")
_PRESET_B1 = ImageClassification(crop_size=240, resize_size=256, interpolation="bicubic")
# Divergence, deliberate: the paper's compound-scaling table puts B2 at
# 260x260, but the checkpoint this preset serves was trained and published
# at 288 — evaluating it at 260 measures a resolution it never saw.  The
# eval preset follows the weights; ``EfficientNetConfig`` still carries the
# paper's 260 for anyone training from scratch.
_PRESET_B2 = ImageClassification(crop_size=288, resize_size=288, interpolation="bicubic")
_PRESET_B3 = ImageClassification(crop_size=300, resize_size=320, interpolation="bicubic")
_PRESET_B4 = ImageClassification(crop_size=380, resize_size=384, interpolation="bicubic")
_PRESET_B5 = ImageClassification(crop_size=456, resize_size=456, interpolation="bicubic")
_PRESET_B6 = ImageClassification(crop_size=528, resize_size=528, interpolation="bicubic")
_PRESET_B7 = ImageClassification(crop_size=600, resize_size=600, interpolation="bicubic")


@register_weights("efficientnet_b0_cls")
class EfficientNetB0Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.efficientnet_b0_cls`.

    baseline B0 at 224 (5.3 M params, 0.39 GFLOPs, top-1 77.69%).

    Attributes
    ----------
    IMAGENET1K_V1 : WeightEntry
        ImageNet-1k V1 checkpoint (top-1 77.692% / top-5 93.532%),
        sourced from ``reference_vision/EfficientNet_B0_Weights.IMAGENET1K_V1``.
    DEFAULT : WeightEntry
        Alias for :attr:`IMAGENET1K_V1`.

    Notes
    -----
    Reference: Tan, Le, *"EfficientNet: Rethinking Model Scaling for
    Convolutional Neural Networks"*, ICML 2019 (arXiv:1905.11946).

    Examples
    --------
    >>> from lucid.models import efficientnet_b0_cls
    >>> model = efficientnet_b0_cls(pretrained=True).eval()
    """

    IMAGENET1K_V1 = WeightEntry(
        url=f"{HUB_BASE}/efficientnet-b0/resolve/main/IMAGENET1K_V1/model.safetensors",
        sha256="edd48083cff3cf742acb7a54a5df1a149106e6438f1370521bf97135ec2d7dfd",
        num_classes=1000,
        transforms=_PRESET_B0,
        meta={
            "tag": "IMAGENET1K_V1",
            "source": "reference_vision/EfficientNet_B0_Weights.IMAGENET1K_V1",
            "license": "apache-2.0",
            "num_params": 5_288_548,
            "gflops": 0.386,
            "metrics": {"ImageNet-1k": {"acc@1": 77.692, "acc@5": 93.532}},
        },
    )
    DEFAULT = IMAGENET1K_V1


@register_weights("efficientnet_b1_cls")
class EfficientNetB1Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.efficientnet_b1_cls`.

    compound-scaled B1 at 240 (7.8 M params, 0.69 GFLOPs, top-1 78.64%).

    Attributes
    ----------
    IMAGENET1K_V1 : WeightEntry
        ImageNet-1k V1 checkpoint (top-1 78.642% / top-5 94.186%),
        sourced from ``reference_vision/EfficientNet_B1_Weights.IMAGENET1K_V1``.
    DEFAULT : WeightEntry
        Alias for :attr:`IMAGENET1K_V1`.

    Notes
    -----
    Reference: Tan, Le, *"EfficientNet: Rethinking Model Scaling for
    Convolutional Neural Networks"*, ICML 2019 (arXiv:1905.11946).

    Examples
    --------
    >>> from lucid.models import efficientnet_b1_cls
    >>> model = efficientnet_b1_cls(pretrained=True).eval()
    """

    IMAGENET1K_V1 = WeightEntry(
        url=f"{HUB_BASE}/efficientnet-b1/resolve/main/IMAGENET1K_V1/model.safetensors",
        sha256="9ed9cae6f8e969623d5b8d366e6d882c6fd0b6e381004475dc3c719e4f032310",
        num_classes=1000,
        transforms=_PRESET_B1,
        meta={
            "tag": "IMAGENET1K_V1",
            "source": "reference_vision/EfficientNet_B1_Weights.IMAGENET1K_V1",
            "license": "apache-2.0",
            "num_params": 7_794_184,
            "gflops": 0.687,
            "metrics": {"ImageNet-1k": {"acc@1": 78.642, "acc@5": 94.186}},
        },
    )
    DEFAULT = IMAGENET1K_V1


@register_weights("efficientnet_b2_cls")
class EfficientNetB2Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.efficientnet_b2_cls`.

    compound-scaled B2 at 288 (9.1 M params, 1.09 GFLOPs, top-1 80.61%).

    Attributes
    ----------
    IMAGENET1K_V1 : WeightEntry
        ImageNet-1k V1 checkpoint (top-1 80.608% / top-5 95.310%),
        sourced from ``reference_vision/EfficientNet_B2_Weights.IMAGENET1K_V1``.
    DEFAULT : WeightEntry
        Alias for :attr:`IMAGENET1K_V1`.

    Notes
    -----
    Reference: Tan, Le, *"EfficientNet: Rethinking Model Scaling for
    Convolutional Neural Networks"*, ICML 2019 (arXiv:1905.11946).

    Examples
    --------
    >>> from lucid.models import efficientnet_b2_cls
    >>> model = efficientnet_b2_cls(pretrained=True).eval()
    """

    IMAGENET1K_V1 = WeightEntry(
        url=f"{HUB_BASE}/efficientnet-b2/resolve/main/IMAGENET1K_V1/model.safetensors",
        sha256="95a44dfcf358f6d59250be24822781e11582583ddfbfef9ac225d298771bf1fe",
        num_classes=1000,
        transforms=_PRESET_B2,
        meta={
            "tag": "IMAGENET1K_V1",
            "source": "reference_vision/EfficientNet_B2_Weights.IMAGENET1K_V1",
            "license": "apache-2.0",
            "num_params": 9_109_994,
            "gflops": 1.088,
            "metrics": {"ImageNet-1k": {"acc@1": 80.608, "acc@5": 95.310}},
        },
    )
    DEFAULT = IMAGENET1K_V1


@register_weights("efficientnet_b3_cls")
class EfficientNetB3Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.efficientnet_b3_cls`.

    compound-scaled B3 at 300 (12.2 M params, 1.83 GFLOPs, top-1 82.01%).

    Attributes
    ----------
    IMAGENET1K_V1 : WeightEntry
        ImageNet-1k V1 checkpoint (top-1 82.008% / top-5 96.054%),
        sourced from ``reference_vision/EfficientNet_B3_Weights.IMAGENET1K_V1``.
    DEFAULT : WeightEntry
        Alias for :attr:`IMAGENET1K_V1`.

    Notes
    -----
    Reference: Tan, Le, *"EfficientNet: Rethinking Model Scaling for
    Convolutional Neural Networks"*, ICML 2019 (arXiv:1905.11946).

    Examples
    --------
    >>> from lucid.models import efficientnet_b3_cls
    >>> model = efficientnet_b3_cls(pretrained=True).eval()
    """

    IMAGENET1K_V1 = WeightEntry(
        url=f"{HUB_BASE}/efficientnet-b3/resolve/main/IMAGENET1K_V1/model.safetensors",
        sha256="186f29541f1614a584748a8ca400e39ea3d7ecc3602c43623cad34b4585528ca",
        num_classes=1000,
        transforms=_PRESET_B3,
        meta={
            "tag": "IMAGENET1K_V1",
            "source": "reference_vision/EfficientNet_B3_Weights.IMAGENET1K_V1",
            "license": "apache-2.0",
            "num_params": 12_233_232,
            "gflops": 1.827,
            "metrics": {"ImageNet-1k": {"acc@1": 82.008, "acc@5": 96.054}},
        },
    )
    DEFAULT = IMAGENET1K_V1


@register_weights("efficientnet_b4_cls")
class EfficientNetB4Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.efficientnet_b4_cls`.

    compound-scaled B4 at 380 (19.3 M params, 4.39 GFLOPs, top-1 83.38%).

    Attributes
    ----------
    IMAGENET1K_V1 : WeightEntry
        ImageNet-1k V1 checkpoint (top-1 83.384% / top-5 96.594%),
        sourced from ``reference_vision/EfficientNet_B4_Weights.IMAGENET1K_V1``.
    DEFAULT : WeightEntry
        Alias for :attr:`IMAGENET1K_V1`.

    Notes
    -----
    Reference: Tan, Le, *"EfficientNet: Rethinking Model Scaling for
    Convolutional Neural Networks"*, ICML 2019 (arXiv:1905.11946).

    Examples
    --------
    >>> from lucid.models import efficientnet_b4_cls
    >>> model = efficientnet_b4_cls(pretrained=True).eval()
    """

    IMAGENET1K_V1 = WeightEntry(
        url=f"{HUB_BASE}/efficientnet-b4/resolve/main/IMAGENET1K_V1/model.safetensors",
        sha256="4a52744a4bf1c8125536c3097c30864fba387c2c70065bad931f383b5d7f847a",
        num_classes=1000,
        transforms=_PRESET_B4,
        meta={
            "tag": "IMAGENET1K_V1",
            "source": "reference_vision/EfficientNet_B4_Weights.IMAGENET1K_V1",
            "license": "apache-2.0",
            "num_params": 19_341_616,
            "gflops": 4.394,
            "metrics": {"ImageNet-1k": {"acc@1": 83.384, "acc@5": 96.594}},
        },
    )
    DEFAULT = IMAGENET1K_V1


@register_weights("efficientnet_b5_cls")
class EfficientNetB5Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.efficientnet_b5_cls`.

    compound-scaled B5 at 456 (30.4 M params, 10.27 GFLOPs, top-1 83.44%).

    Attributes
    ----------
    IMAGENET1K_V1 : WeightEntry
        ImageNet-1k V1 checkpoint (top-1 83.444% / top-5 96.628%),
        sourced from ``reference_vision/EfficientNet_B5_Weights.IMAGENET1K_V1``.
    DEFAULT : WeightEntry
        Alias for :attr:`IMAGENET1K_V1`.

    Notes
    -----
    Reference: Tan, Le, *"EfficientNet: Rethinking Model Scaling for
    Convolutional Neural Networks"*, ICML 2019 (arXiv:1905.11946).

    Examples
    --------
    >>> from lucid.models import efficientnet_b5_cls
    >>> model = efficientnet_b5_cls(pretrained=True).eval()
    """

    IMAGENET1K_V1 = WeightEntry(
        url=f"{HUB_BASE}/efficientnet-b5/resolve/main/IMAGENET1K_V1/model.safetensors",
        sha256="09538cbe7ec8dda318ba5d9e34915491391086abff99ec7fa571303289fb77c7",
        num_classes=1000,
        transforms=_PRESET_B5,
        meta={
            "tag": "IMAGENET1K_V1",
            "source": "reference_vision/EfficientNet_B5_Weights.IMAGENET1K_V1",
            "license": "apache-2.0",
            "num_params": 30_389_784,
            "gflops": 10.266,
            "metrics": {"ImageNet-1k": {"acc@1": 83.444, "acc@5": 96.628}},
        },
    )
    DEFAULT = IMAGENET1K_V1


@register_weights("efficientnet_b6_cls")
class EfficientNetB6Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.efficientnet_b6_cls`.

    compound-scaled B6 at 528 (43.0 M params, 19.07 GFLOPs, top-1 84.01%).

    Attributes
    ----------
    IMAGENET1K_V1 : WeightEntry
        ImageNet-1k V1 checkpoint (top-1 84.008% / top-5 96.916%),
        sourced from ``reference_vision/EfficientNet_B6_Weights.IMAGENET1K_V1``.
    DEFAULT : WeightEntry
        Alias for :attr:`IMAGENET1K_V1`.

    Notes
    -----
    Reference: Tan, Le, *"EfficientNet: Rethinking Model Scaling for
    Convolutional Neural Networks"*, ICML 2019 (arXiv:1905.11946).

    Examples
    --------
    >>> from lucid.models import efficientnet_b6_cls
    >>> model = efficientnet_b6_cls(pretrained=True).eval()
    """

    IMAGENET1K_V1 = WeightEntry(
        url=f"{HUB_BASE}/efficientnet-b6/resolve/main/IMAGENET1K_V1/model.safetensors",
        sha256="1cd3b2bf6f75a266f0df93d8d4607be2cd2f99b6e803e97ddf9c51c9fa99dc6e",
        num_classes=1000,
        transforms=_PRESET_B6,
        meta={
            "tag": "IMAGENET1K_V1",
            "source": "reference_vision/EfficientNet_B6_Weights.IMAGENET1K_V1",
            "license": "apache-2.0",
            "num_params": 43_040_704,
            "gflops": 19.068,
            "metrics": {"ImageNet-1k": {"acc@1": 84.008, "acc@5": 96.916}},
        },
    )
    DEFAULT = IMAGENET1K_V1


@register_weights("efficientnet_b7_cls")
class EfficientNetB7Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.efficientnet_b7_cls`.

    compound-scaled B7 at 600 (66.3 M params, 37.75 GFLOPs, top-1 84.12%).

    Attributes
    ----------
    IMAGENET1K_V1 : WeightEntry
        ImageNet-1k V1 checkpoint (top-1 84.122% / top-5 96.908%),
        sourced from ``reference_vision/EfficientNet_B7_Weights.IMAGENET1K_V1``.
    DEFAULT : WeightEntry
        Alias for :attr:`IMAGENET1K_V1`.

    Notes
    -----
    Reference: Tan, Le, *"EfficientNet: Rethinking Model Scaling for
    Convolutional Neural Networks"*, ICML 2019 (arXiv:1905.11946).

    Examples
    --------
    >>> from lucid.models import efficientnet_b7_cls
    >>> model = efficientnet_b7_cls(pretrained=True).eval()
    """

    IMAGENET1K_V1 = WeightEntry(
        url=f"{HUB_BASE}/efficientnet-b7/resolve/main/IMAGENET1K_V1/model.safetensors",
        sha256="2001634139ed24298f4afe1ad6e3a70314d169649ae08a79b3ec3668b213e3b6",
        num_classes=1000,
        transforms=_PRESET_B7,
        meta={
            "tag": "IMAGENET1K_V1",
            "source": "reference_vision/EfficientNet_B7_Weights.IMAGENET1K_V1",
            "license": "apache-2.0",
            "num_params": 66_347_960,
            "gflops": 37.746,
            "metrics": {"ImageNet-1k": {"acc@1": 84.122, "acc@5": 96.908}},
        },
    )
    DEFAULT = IMAGENET1K_V1
