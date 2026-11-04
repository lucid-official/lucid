"""Pretrained-weight declarations for the Swin Transformer family.

Per-model weight enums live *with the model* (not in
:mod:`lucid.weights`).  :mod:`lucid.weights` is purely the porting
infrastructure — base classes, transforms, hub download, loading, and
the discovery registry.  Importing this module (which happens when
:mod:`lucid.models.vision.swin` loads) registers the enums with the
discovery registry (:func:`lucid.weights.list_pretrained` /
:func:`lucid.weights.get_weight`).

Four paper-cited variants (Liu et al., ICCV 2021) ship ImageNet
checkpoints.  Tiny / Small / Base are converted from the
reference-framework model zoo's ImageNet-1k weights; Large is sourced
from the original Microsoft checkpoint (ImageNet-22k pretrain →
ImageNet-1k finetune) re-hosted via timm.  Each variant uses bicubic
resize with its own resize side length (the reference zoo tuned these
per model), and the standard ImageNet mean / std.

All four checkpoints are hosted under the ``lucid-dl`` Hugging Face
org and load directly via ``swin_<size>_cls(pretrained=True)``.
"""

from lucid.utils.transforms import ImageClassification
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights

# reference_vision tuned a distinct resize side length per Swin variant; all use
# a 224 crop with bicubic interpolation and the standard ImageNet stats.
_PRESET_T = ImageClassification(crop_size=224, resize_size=232, interpolation="bicubic")
_PRESET_S = ImageClassification(crop_size=224, resize_size=246, interpolation="bicubic")
_PRESET_B = ImageClassification(crop_size=224, resize_size=238, interpolation="bicubic")
# timm Swin-L: crop_pct=0.9 → resize = floor(224 / 0.9) = 248, bicubic.
_PRESET_L = ImageClassification(crop_size=224, resize_size=248, interpolation="bicubic")


@register_weights("swin_tiny_cls")
class SwinTinyWeights(WeightsEnum):
    r"""Pretrained weight tags for :func:`lucid.models.swin_tiny_cls`.

    Ships a single ImageNet-1k checkpoint (:attr:`IMAGENET1K_V1`)
    converted from the reference-framework ``Swin_T_Weights`` and
    re-hosted under ``huggingface.co/lucid-dl/swin-tiny``.

    Examples
    --------
    >>> from lucid.models.vision.swin._weights import SwinTinyWeights
    >>> list(SwinTinyWeights.__members__)
    ['IMAGENET1K_V1', 'DEFAULT']

    ``DEFAULT`` is an alias rather than a fourth entry, so a bare
    ``pretrained=True`` and the tag it resolves to cannot drift apart.

    >>> SwinTinyWeights.DEFAULT is SwinTinyWeights.IMAGENET1K_V1
    True
    >>> SwinTinyWeights.IMAGENET1K_V1.num_classes
    1000
    """

    IMAGENET1K_V1 = WeightEntry(
        url=f"{HUB_BASE}/swin-tiny/resolve/main/IMAGENET1K_V1/model.safetensors",
        sha256="34313dbaf0788b861b700cfc85c9933ad9d4bd62e823741e2064211ec10a0c25",
        num_classes=1000,
        transforms=_PRESET_T,
        meta={
            "tag": "IMAGENET1K_V1",
            "source": "reference_vision/Swin_T_Weights.IMAGENET1K_V1",
            "license": "mit",
            "num_params": 28_288_354,
            "gflops": 4.491,
            "metrics": {"ImageNet-1k": {"acc@1": 81.474, "acc@5": 95.776}},
        },
    )
    DEFAULT = IMAGENET1K_V1


@register_weights("swin_small_cls")
class SwinSmallWeights(WeightsEnum):
    r"""Pretrained weight tags for :func:`lucid.models.swin_small_cls`.

    Ships a single ImageNet-1k checkpoint (:attr:`IMAGENET1K_V1`)
    converted from the reference-framework ``Swin_S_Weights`` and
    re-hosted under ``huggingface.co/lucid-dl/swin-small``.

    Examples
    --------
    >>> from lucid.models.vision.swin._weights import SwinSmallWeights
    >>> list(SwinSmallWeights.__members__)
    ['IMAGENET1K_V1', 'DEFAULT']

    ``DEFAULT`` is an alias rather than a fourth entry, so a bare
    ``pretrained=True`` and the tag it resolves to cannot drift apart.

    >>> SwinSmallWeights.DEFAULT is SwinSmallWeights.IMAGENET1K_V1
    True
    >>> SwinSmallWeights.IMAGENET1K_V1.num_classes
    1000
    """

    IMAGENET1K_V1 = WeightEntry(
        url=f"{HUB_BASE}/swin-small/resolve/main/IMAGENET1K_V1/model.safetensors",
        sha256="b910f9447ddd22c67d47a73a56e03a1d07eb398469490532934620f816e6bcf7",
        num_classes=1000,
        transforms=_PRESET_S,
        meta={
            "tag": "IMAGENET1K_V1",
            "source": "reference_vision/Swin_S_Weights.IMAGENET1K_V1",
            "license": "mit",
            "num_params": 49_606_258,
            "gflops": 8.741,
            "metrics": {"ImageNet-1k": {"acc@1": 83.196, "acc@5": 96.360}},
        },
    )
    DEFAULT = IMAGENET1K_V1


@register_weights("swin_base_cls")
class SwinBaseWeights(WeightsEnum):
    r"""Pretrained weight tags for :func:`lucid.models.swin_base_cls`.

    Ships a single ImageNet-1k checkpoint (:attr:`IMAGENET1K_V1`)
    converted from the reference-framework ``Swin_B_Weights`` and
    re-hosted under ``huggingface.co/lucid-dl/swin-base``.

    Examples
    --------
    >>> from lucid.models.vision.swin._weights import SwinBaseWeights
    >>> list(SwinBaseWeights.__members__)
    ['IMAGENET1K_V1', 'DEFAULT']

    ``DEFAULT`` is an alias rather than a fourth entry, so a bare
    ``pretrained=True`` and the tag it resolves to cannot drift apart.

    >>> SwinBaseWeights.DEFAULT is SwinBaseWeights.IMAGENET1K_V1
    True
    >>> SwinBaseWeights.IMAGENET1K_V1.num_classes
    1000
    """

    IMAGENET1K_V1 = WeightEntry(
        url=f"{HUB_BASE}/swin-base/resolve/main/IMAGENET1K_V1/model.safetensors",
        sha256="4ad14a29605d612a06ec5b46081732482dfee8fad239d58d93524c0509d694aa",
        num_classes=1000,
        transforms=_PRESET_B,
        meta={
            "tag": "IMAGENET1K_V1",
            "source": "reference_vision/Swin_B_Weights.IMAGENET1K_V1",
            "license": "mit",
            "num_params": 87_768_224,
            "gflops": 15.431,
            "metrics": {"ImageNet-1k": {"acc@1": 83.582, "acc@5": 96.640}},
        },
    )
    DEFAULT = IMAGENET1K_V1


@register_weights("swin_large_cls")
class SwinLargeWeights(WeightsEnum):
    r"""Pretrained weight tags for :func:`lucid.models.swin_large_cls`.

    Ships the original Microsoft ImageNet-22k → ImageNet-1k finetune
    (:attr:`MS_IN22K_FT_IN1K`), re-hosted via timm
    (``swin_large_patch4_window7_224.ms_in22k_ft_in1k``) under
    ``huggingface.co/lucid-dl/swin-large``.  86.3% top-1 at 224×224 is
    the figure published for this checkpoint tag; the paper does not
    report a 224×224 number for Swin-L, and its Table 2 is COCO
    detection, not classification.

    Examples
    --------
    >>> from lucid.models.vision.swin._weights import SwinLargeWeights
    >>> list(SwinLargeWeights.__members__)
    ['MS_IN22K_FT_IN1K', 'DEFAULT']

    ``DEFAULT`` is an alias rather than a fourth entry, so a bare
    ``pretrained=True`` and the tag it resolves to cannot drift apart.

    >>> SwinLargeWeights.DEFAULT is SwinLargeWeights.MS_IN22K_FT_IN1K
    True
    >>> SwinLargeWeights.MS_IN22K_FT_IN1K.num_classes
    1000
    """

    MS_IN22K_FT_IN1K = WeightEntry(
        url=f"{HUB_BASE}/swin-large/resolve/main/MS_IN22K_FT_IN1K/model.safetensors",
        sha256="f1c8410824390625a3188d2bb5216b89e491447eb4197b8e9a2561238122593b",
        num_classes=1000,
        transforms=_PRESET_L,
        meta={
            "tag": "MS_IN22K_FT_IN1K",
            "source": "timm/swin_large_patch4_window7_224.ms_in22k_ft_in1k",
            "license": "mit",
            "num_params": 196_532_476,
            "metrics": {"ImageNet-1k": {"acc@1": 86.320, "acc@5": 97.890}},
        },
    )
    DEFAULT = MS_IN22K_FT_IN1K
