"""Pretrained-weight declarations for the ResNeXt family.

Three paper-cited variants (Xie et al., CVPR 2017):

* :class:`ResNeXt50_32x4dWeights` and :class:`ResNeXt101_32x8dWeights`
  ship the reference_vision ``IMAGENET1K_V2`` checkpoints (the improved
  training recipe), which evaluate with a **232** resize → 224 crop
  bilinear preset (not the usual 256).
* :class:`ResNeXt101_32x4dWeights` ships the timm Gluon ``GLUON_IN1K``
  checkpoint, which evaluates with a **bicubic** 0.875-crop_pct preset
  (256 resize → 224 crop).

Each preset below is replicated verbatim from its upstream source so the
on-Hub transforms stay in lock-step with the conversion recipe.
"""

from lucid.utils.transforms import ImageClassification
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights


@register_weights("resnext_50_32x4d_cls")
class ResNeXt50_32x4dWeights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.resnext_50_32x4d_cls`.

    ResNeXt-50 (32 groups × 4d width, 25.0 M params, top-1 81.20%).

    Attributes
    ----------
    IMAGENET1K_V2 : WeightEntry
        ImageNet-1k V2 checkpoint (top-1 81.198% / top-5 95.340%),
        sourced from ``reference_vision/ResNeXt50_32X4D_Weights.IMAGENET1K_V2``.
    DEFAULT : WeightEntry
        Alias for :attr:`IMAGENET1K_V2`.

    Notes
    -----
    Reference: Xie, Girshick, Dollár, Tu, He, *"Aggregated Residual
    Transformations for Deep Neural Networks"*, CVPR 2017
    (arXiv:1611.05431).

    Examples
    --------
    >>> from lucid.models import resnext_50_32x4d_cls
    >>> model = resnext_50_32x4d_cls(pretrained=True).eval()
    """

    IMAGENET1K_V2 = WeightEntry(
        url=(f"{HUB_BASE}/resnext-50-32x4d/resolve/main/" "IMAGENET1K_V2/model.safetensors"),
        sha256="95ab469ce4c14c1a7e1e517f8eeb140ea9b7747946d9ccd0258dd65f869fc704",
        num_classes=1000,
        transforms=ImageClassification(crop_size=224, resize_size=232),
        meta={
            "tag": "IMAGENET1K_V2",
            "source": "reference_vision/ResNeXt50_32X4D_Weights.IMAGENET1K_V2",
            "license": "bsd-3-clause",
            "num_params": 25_028_904,
            "metrics": {"ImageNet-1k": {"acc@1": 81.198, "acc@5": 95.340}},
        },
    )
    DEFAULT = IMAGENET1K_V2


@register_weights("resnext_101_32x8d_cls")
class ResNeXt101_32x8dWeights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.resnext_101_32x8d_cls`.

    ResNeXt-101 (32 groups × 8d width, 88.8 M params, top-1 82.83%).

    Attributes
    ----------
    IMAGENET1K_V2 : WeightEntry
        ImageNet-1k V2 checkpoint (top-1 82.834% / top-5 96.228%),
        sourced from ``reference_vision/ResNeXt101_32X8D_Weights.IMAGENET1K_V2``.
    DEFAULT : WeightEntry
        Alias for :attr:`IMAGENET1K_V2`.

    Notes
    -----
    Reference: Xie, Girshick, Dollár, Tu, He, *"Aggregated Residual
    Transformations for Deep Neural Networks"*, CVPR 2017
    (arXiv:1611.05431).

    Examples
    --------
    >>> from lucid.models import resnext_101_32x8d_cls
    >>> model = resnext_101_32x8d_cls(pretrained=True).eval()
    """

    IMAGENET1K_V2 = WeightEntry(
        url=(f"{HUB_BASE}/resnext-101-32x8d/resolve/main/" "IMAGENET1K_V2/model.safetensors"),
        sha256="39ddf40a89a715f0d9a4df83d94a3e338151076972917f6f1a29b3d7ebddccdb",
        num_classes=1000,
        transforms=ImageClassification(crop_size=224, resize_size=232),
        meta={
            "tag": "IMAGENET1K_V2",
            "source": "reference_vision/ResNeXt101_32X8D_Weights.IMAGENET1K_V2",
            "license": "bsd-3-clause",
            "num_params": 88_791_336,
            "metrics": {"ImageNet-1k": {"acc@1": 82.834, "acc@5": 96.228}},
        },
    )
    DEFAULT = IMAGENET1K_V2


@register_weights("resnext_101_32x4d_cls")
class ResNeXt101_32x4dWeights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.resnext_101_32x4d_cls`.

    Sourced from the timm Gluon ``gluon_in1k`` checkpoint — note the
    bicubic interpolation and 0.875 crop_pct (256 resize → 224 crop)
    rather than the bilinear / 232-resize preset of the reference_vision V2
    variants.

    Attributes
    ----------
    GLUON_IN1K : WeightEntry
        timm Gluon ImageNet-1k checkpoint (top-1 80.342% / top-5
        94.926%), sourced from ``timm/resnext101_32x4d.gluon_in1k``.
    DEFAULT : WeightEntry
        Alias for :attr:`GLUON_IN1K`.

    Notes
    -----
    Reference: Xie, Girshick, Dollár, Tu, He, *"Aggregated Residual
    Transformations for Deep Neural Networks"*, CVPR 2017
    (arXiv:1611.05431).

    Examples
    --------
    >>> from lucid.models import resnext_101_32x4d_cls
    >>> model = resnext_101_32x4d_cls(pretrained=True).eval()
    """

    GLUON_IN1K = WeightEntry(
        url=(f"{HUB_BASE}/resnext-101-32x4d/resolve/main/" "GLUON_IN1K/model.safetensors"),
        sha256="20e585e08b43f2771202755e1947e0f3ed5a98e6875061b765e994ef16216b6d",
        num_classes=1000,
        transforms=ImageClassification(crop_size=224, resize_size=256, interpolation="bicubic"),
        meta={
            "tag": "GLUON_IN1K",
            "source": "timm/resnext101_32x4d.gluon_in1k",
            "license": "apache-2.0",
            "num_params": 44_177_704,
            "metrics": {"ImageNet-1k": {"acc@1": 80.342, "acc@5": 94.926}},
        },
    )
    DEFAULT = GLUON_IN1K
