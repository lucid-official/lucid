"""Pretrained-weight declarations for the CvT family.

Three paper-cited variants (Wu et al., ICCV 2021) — all sourced from
Microsoft's HuggingFace Hub publications via the
``transformers.CvtForImageClassification`` loader.  ``cvt-13`` and
``cvt-21`` are ImageNet-1k finetunes at 224×224; ``cvt-w24`` is the
ImageNet-22k pretrain → ImageNet-1k finetune at 384×384.
"""

from lucid.utils.transforms import ImageClassification
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights

_PRESET_224 = ImageClassification(crop_size=224, resize_size=256, interpolation="bicubic")
_PRESET_384 = ImageClassification(crop_size=384, resize_size=439, interpolation="bicubic")


@register_weights("cvt_13_cls")
class CvT13Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.cvt_13_cls`.

    Wu et al. ICCV 2021 13-layer convolutional ViT (20 M params, top-1 81.6%).

    Attributes
    ----------
    IN1K : WeightEntry
        ImageNet-1k 224x224 finetune (top-1 81.6%), sourced from
        ``transformers/microsoft/cvt-13``.
    DEFAULT : WeightEntry
        Alias for :attr:`IN1K`.

    Notes
    -----
    Reference: Wu, Xiao, Codella, Liu, Dai, Yuan, Zhang, *"CvT:
    Introducing Convolutions to Vision Transformers"*, ICCV 2021
    (arXiv:2103.15808).

    Examples
    --------
    >>> from lucid.models import cvt_13_cls
    >>> model = cvt_13_cls(pretrained=True).eval()
    """

    IN1K = WeightEntry(
        url=f"{HUB_BASE}/cvt-13/resolve/main/IN1K/model.safetensors",
        sha256="a5e4d23f06c6c910cdb24fc0c350a106856d72ef1cd678863a1c6b380d9fe663",
        num_classes=1000,
        transforms=_PRESET_224,
        meta={
            "tag": "IN1K",
            "source": "transformers/microsoft/cvt-13",
            "license": "apache-2.0",
            "num_params": 20_000_000,
            "metrics": {"ImageNet-1k": {"acc@1": 81.6}},
        },
    )
    DEFAULT = IN1K


@register_weights("cvt_21_cls")
class CvT21Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.cvt_21_cls`.

    Wu et al. ICCV 2021 21-layer convolutional ViT (31.6 M params, top-1 82.5%).

    Attributes
    ----------
    IN1K : WeightEntry
        ImageNet-1k 224x224 finetune (top-1 82.5%), sourced from
        ``transformers/microsoft/cvt-21``.
    DEFAULT : WeightEntry
        Alias for :attr:`IN1K`.

    Notes
    -----
    Reference: Wu, Xiao, Codella, Liu, Dai, Yuan, Zhang, *"CvT:
    Introducing Convolutions to Vision Transformers"*, ICCV 2021
    (arXiv:2103.15808).

    Examples
    --------
    >>> from lucid.models import cvt_21_cls
    >>> model = cvt_21_cls(pretrained=True).eval()
    """

    IN1K = WeightEntry(
        url=f"{HUB_BASE}/cvt-21/resolve/main/IN1K/model.safetensors",
        sha256="f9d30c499fec9a21b55980bea15ed45638c16069a191c10011c4e6d52f1f60f0",
        num_classes=1000,
        transforms=_PRESET_224,
        meta={
            "tag": "IN1K",
            "source": "transformers/microsoft/cvt-21",
            "license": "apache-2.0",
            "num_params": 31_620_000,
            "metrics": {"ImageNet-1k": {"acc@1": 82.5}},
        },
    )
    DEFAULT = IN1K


@register_weights("cvt_w24_cls")
class CvTW24Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.cvt_w24_cls`.

    wide CvT-w24 IN-22k→IN-1k 384 finetune (Wu et al.; 277 M params, top-1 87.7%).

    Attributes
    ----------
    IN22K_FT_IN1K_384 : WeightEntry
        ImageNet-22k pretrain + ImageNet-1k 384x384 finetune (top-1
        87.7%), sourced from ``transformers/microsoft/cvt-w24-384-22k``.
    DEFAULT : WeightEntry
        Alias for :attr:`IN22K_FT_IN1K_384`.

    Notes
    -----
    Reference: Wu, Xiao, Codella, Liu, Dai, Yuan, Zhang, *"CvT:
    Introducing Convolutions to Vision Transformers"*, ICCV 2021
    (arXiv:2103.15808).

    Examples
    --------
    >>> from lucid.models import cvt_w24_cls
    >>> model = cvt_w24_cls(pretrained=True).eval()
    """

    IN22K_FT_IN1K_384 = WeightEntry(
        url=f"{HUB_BASE}/cvt-w24/resolve/main/IN22K_FT_IN1K_384/model.safetensors",
        sha256="65049228d1bcc258388825881ea56a9a93c501bc8e0f61a3005c386e9651494a",
        num_classes=1000,
        transforms=_PRESET_384,
        meta={
            "tag": "IN22K_FT_IN1K_384",
            "source": "transformers/microsoft/cvt-w24-384-22k",
            "license": "apache-2.0",
            "num_params": 277_200_000,
            "metrics": {"ImageNet-1k": {"acc@1": 87.7}},
        },
    )
    DEFAULT = IN22K_FT_IN1K_384
