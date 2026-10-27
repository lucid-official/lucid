"""Pretrained-weight declarations for the SE-ResNet family.

Five paper-cited SE-ResNet variants (Hu et al., CVPR 2018).  Four ship
ImageNet-1k checkpoints converted from timm's ``legacy_seresnet*.in1k``
line (the only public weights for those depths); SE-ResNet-50 instead
uses the stronger ``seresnet50.ra2_in1k`` recipe (78.5 top-1).

Eval presets are sourced from each upstream ``default_cfg``: every
variant uses a 224 crop / 256 resize (``crop_pct = 0.875``) with
standard ImageNet stats.  SE-ResNet-18 and SE-ResNet-50 evaluate with
**bicubic** interpolation; SE-ResNet-34 / 101 / 152 use **bilinear**.
"""

from lucid.utils.transforms import ImageClassification
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights

_PRESET_BICUBIC = ImageClassification(crop_size=224, resize_size=256, interpolation="bicubic")
_PRESET_BILINEAR = ImageClassification(crop_size=224, resize_size=256, interpolation="bilinear")


@register_weights("se_resnet_18_cls")
class SEResNet18Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.se_resnet_18_cls`.

    SE-ResNet-18 (Hu et al. CVPR 2018; 11.8 M params, top-1 70.6%).

    Attributes
    ----------
    IN1K : WeightEntry
        ImageNet-1k checkpoint (top-1 70.6%), sourced from
        ``timm/legacy_seresnet18.in1k``.
    DEFAULT : WeightEntry
        Alias for :attr:`IN1K`.

    Notes
    -----
    Reference: Hu, Shen, Sun, *"Squeeze-and-Excitation Networks"*, CVPR
    2018 (arXiv:1709.01507).

    Examples
    --------
    >>> from lucid.models import se_resnet_18_cls
    >>> model = se_resnet_18_cls(pretrained=True).eval()
    """

    IN1K = WeightEntry(
        url=f"{HUB_BASE}/se-resnet-18/resolve/main/IN1K/model.safetensors",
        sha256="e89ffffd7fba340c9813d363f77a967a4fc4fa184a134db9a841828c7d9c65f8",
        num_classes=1000,
        transforms=_PRESET_BICUBIC,
        meta={
            "tag": "IN1K",
            "source": "timm/legacy_seresnet18.in1k",
            "license": "apache-2.0",
            "num_params": 11_778_592,
            "metrics": {"ImageNet-1k": {"acc@1": 70.6}},
        },
    )
    DEFAULT = IN1K


@register_weights("se_resnet_34_cls")
class SEResNet34Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.se_resnet_34_cls`.

    SE-ResNet-34 (Hu et al. CVPR 2018; 22.0 M params, top-1 73.31%).

    Attributes
    ----------
    IN1K : WeightEntry
        ImageNet-1k checkpoint (top-1 73.31%), sourced from
        ``timm/legacy_seresnet34.in1k``.
    DEFAULT : WeightEntry
        Alias for :attr:`IN1K`.

    Notes
    -----
    Reference: Hu, Shen, Sun, *"Squeeze-and-Excitation Networks"*, CVPR
    2018 (arXiv:1709.01507).

    Examples
    --------
    >>> from lucid.models import se_resnet_34_cls
    >>> model = se_resnet_34_cls(pretrained=True).eval()
    """

    IN1K = WeightEntry(
        url=f"{HUB_BASE}/se-resnet-34/resolve/main/IN1K/model.safetensors",
        sha256="af974ca3a9af1974cd053ec551d2aab8930fc02181ec91062a0243ada279c464",
        num_classes=1000,
        transforms=_PRESET_BILINEAR,
        meta={
            "tag": "IN1K",
            "source": "timm/legacy_seresnet34.in1k",
            "license": "apache-2.0",
            "num_params": 21_958_868,
            "metrics": {"ImageNet-1k": {"acc@1": 73.31}},
        },
    )
    DEFAULT = IN1K


@register_weights("se_resnet_50_cls")
class SEResNet50Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.se_resnet_50_cls`.

    Sourced from timm's ``seresnet50.ra2_in1k`` RandAugment recipe — a
    cleanly SE-augmented ResNet whose state-dict naming matches Lucid
    one-for-one (identity key map).

    Attributes
    ----------
    RA2_IN1K : WeightEntry
        timm RandAugment-v2 ImageNet-1k checkpoint (top-1 78.498%),
        sourced from ``timm/seresnet50.ra2_in1k``.
    DEFAULT : WeightEntry
        Alias for :attr:`RA2_IN1K`.

    Notes
    -----
    Reference: Hu, Shen, Sun, *"Squeeze-and-Excitation Networks"*, CVPR
    2018 (arXiv:1709.01507).

    Examples
    --------
    >>> from lucid.models import se_resnet_50_cls
    >>> model = se_resnet_50_cls(pretrained=True).eval()
    """

    RA2_IN1K = WeightEntry(
        url=f"{HUB_BASE}/se-resnet-50/resolve/main/RA2_IN1K/model.safetensors",
        sha256="a69697eeabc2b81b6f210d5d5cc2a51a86a80ced5b2e2b0a17e0a08cff528d4a",
        num_classes=1000,
        transforms=_PRESET_BICUBIC,
        meta={
            "tag": "RA2_IN1K",
            "source": "timm/seresnet50.ra2_in1k",
            "license": "apache-2.0",
            "num_params": 28_088_024,
            "metrics": {"ImageNet-1k": {"acc@1": 78.498}},
        },
    )
    DEFAULT = RA2_IN1K


@register_weights("se_resnet_101_cls")
class SEResNet101Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.se_resnet_101_cls`.

    SE-ResNet-101 (Hu et al. CVPR 2018; 49.3 M params, top-1 78.32%).

    Attributes
    ----------
    IN1K : WeightEntry
        ImageNet-1k checkpoint (top-1 78.32%), sourced from
        ``timm/legacy_seresnet101.in1k``.
    DEFAULT : WeightEntry
        Alias for :attr:`IN1K`.

    Notes
    -----
    Reference: Hu, Shen, Sun, *"Squeeze-and-Excitation Networks"*, CVPR
    2018 (arXiv:1709.01507).

    Examples
    --------
    >>> from lucid.models import se_resnet_101_cls
    >>> model = se_resnet_101_cls(pretrained=True).eval()
    """

    IN1K = WeightEntry(
        url=f"{HUB_BASE}/se-resnet-101/resolve/main/IN1K/model.safetensors",
        sha256="37ebdc71b448a8dc6a656b86eb0b54f526c00fcf3ec0a34c0fbdf5b7a72f1f84",
        num_classes=1000,
        transforms=_PRESET_BILINEAR,
        meta={
            "tag": "IN1K",
            "source": "timm/legacy_seresnet101.in1k",
            "license": "apache-2.0",
            "num_params": 49_326_872,
            "metrics": {"ImageNet-1k": {"acc@1": 78.32}},
        },
    )
    DEFAULT = IN1K


@register_weights("se_resnet_152_cls")
class SEResNet152Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.se_resnet_152_cls`.

    SE-ResNet-152 (Hu et al. CVPR 2018; 66.8 M params, top-1 78.66%).

    Attributes
    ----------
    IN1K : WeightEntry
        ImageNet-1k checkpoint (top-1 78.66%), sourced from
        ``timm/legacy_seresnet152.in1k``.
    DEFAULT : WeightEntry
        Alias for :attr:`IN1K`.

    Notes
    -----
    Reference: Hu, Shen, Sun, *"Squeeze-and-Excitation Networks"*, CVPR
    2018 (arXiv:1709.01507).

    Examples
    --------
    >>> from lucid.models import se_resnet_152_cls
    >>> model = se_resnet_152_cls(pretrained=True).eval()
    """

    IN1K = WeightEntry(
        url=f"{HUB_BASE}/se-resnet-152/resolve/main/IN1K/model.safetensors",
        sha256="6c5dfecd5371df766ba1340f742382220c3dcf4a429abd546b630856f584a15c",
        num_classes=1000,
        transforms=_PRESET_BILINEAR,
        meta={
            "tag": "IN1K",
            "source": "timm/legacy_seresnet152.in1k",
            "license": "apache-2.0",
            "num_params": 66_821_848,
            "metrics": {"ImageNet-1k": {"acc@1": 78.66}},
        },
    )
    DEFAULT = IN1K
