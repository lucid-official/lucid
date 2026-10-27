"""Pretrained-weight declarations for the MobileNet v3 family.

Two paper-cited variants (Howard et al., ICCV 2019) — both converted
from reference_vision's ``MobileNet_V3_*_Weights.IMAGENET1K_V1`` tag.  Preset
is the standard ImageNet eval pipeline (224 crop / 256 resize / bilinear
/ ImageNet stats) for both variants.

Note: MobileNet v3 trains its batch-norm with ``eps=0.001`` (not the
1e-5 default); the model applies that override so the folded running
statistics in these checkpoints reproduce the source logits exactly
(verified max abs logit diff < 1e-5 on the Small variant).
"""

from lucid.utils.transforms import ImageClassification
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights

_PRESET = ImageClassification(crop_size=224, resize_size=256)


@register_weights("mobilenet_v3_large_cls")
class MobileNetV3LargeWeights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.mobilenet_v3_large_cls`.

    Howard et al. ICCV 2019 Large (5.5 M params, top-1 74.04%).

    Attributes
    ----------
    IMAGENET1K_V1 : WeightEntry
        ImageNet-1k V1 checkpoint (top-1 74.042% / top-5 91.340%),
        sourced from
        ``reference_vision/MobileNet_V3_Large_Weights.IMAGENET1K_V1``.
    DEFAULT : WeightEntry
        Alias for :attr:`IMAGENET1K_V1`.

    Notes
    -----
    Reference: Howard et al., *"Searching for MobileNetV3"*, ICCV 2019
    (arXiv:1905.02244).

    Examples
    --------
    >>> from lucid.models import mobilenet_v3_large_cls
    >>> model = mobilenet_v3_large_cls(pretrained=True).eval()
    """

    IMAGENET1K_V1 = WeightEntry(
        url=f"{HUB_BASE}/mobilenet-v3-large/resolve/main/" "IMAGENET1K_V1/model.safetensors",
        sha256="238269e463c10e08ba786f4b724de32c632458fc20f6489c46c3559a6d3e7023",
        num_classes=1000,
        transforms=_PRESET,
        meta={
            "tag": "IMAGENET1K_V1",
            "source": "reference_vision/MobileNet_V3_Large_Weights.IMAGENET1K_V1",
            "license": "bsd-3-clause",
            "num_params": 5_483_032,
            "gflops": 0.217,
            "metrics": {"ImageNet-1k": {"acc@1": 74.042, "acc@5": 91.340}},
        },
    )
    DEFAULT = IMAGENET1K_V1


@register_weights("mobilenet_v3_small_cls")
class MobileNetV3SmallWeights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.mobilenet_v3_small_cls`.

    Howard et al. ICCV 2019 Small (2.5 M params, top-1 67.67%).

    Attributes
    ----------
    IMAGENET1K_V1 : WeightEntry
        ImageNet-1k V1 checkpoint (top-1 67.668% / top-5 87.402%),
        sourced from
        ``reference_vision/MobileNet_V3_Small_Weights.IMAGENET1K_V1``.
    DEFAULT : WeightEntry
        Alias for :attr:`IMAGENET1K_V1`.

    Notes
    -----
    Reference: Howard et al., *"Searching for MobileNetV3"*, ICCV 2019
    (arXiv:1905.02244).

    Examples
    --------
    >>> from lucid.models import mobilenet_v3_small_cls
    >>> model = mobilenet_v3_small_cls(pretrained=True).eval()
    """

    IMAGENET1K_V1 = WeightEntry(
        url=f"{HUB_BASE}/mobilenet-v3-small/resolve/main/" "IMAGENET1K_V1/model.safetensors",
        sha256="6e96fd6d266b1291cf3fd24833e1196a774e4949d53180a3f5d7a6dc57fe2f66",
        num_classes=1000,
        transforms=_PRESET,
        meta={
            "tag": "IMAGENET1K_V1",
            "source": "reference_vision/MobileNet_V3_Small_Weights.IMAGENET1K_V1",
            "license": "bsd-3-clause",
            "num_params": 2_542_856,
            "gflops": 0.057,
            "metrics": {"ImageNet-1k": {"acc@1": 67.668, "acc@5": 87.402}},
        },
    )
    DEFAULT = IMAGENET1K_V1
