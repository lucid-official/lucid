"""Pretrained-weight declarations for the MaskFormer family.

Both paper-cited variants (Cheng et al., NeurIPS 2021) ship ADE20k
semantic-segmentation checkpoints converted from the
``facebook/maskformer-resnet{50,101}-ade``
``MaskFormerForInstanceSegmentation`` checkpoints:
:class:`MaskFormerResNet50Weights` and :class:`MaskFormerResNet101Weights`.

Each checkpoint is the 150-class ADE20k segmenter (mask-classification
formulation; ``class_predictor`` is ``(151, 256)`` = 150 foreground + 1
no-object).  The eval pipeline uses the 512² ADE20k semantic crop with
ImageNet normalisation sourced from the upstream image processor (the
:class:`~lucid.utils.transforms.Segmentation` preset), and the model's
semantic output drops the no-object slot before the per-query
softmax ⊗ sigmoid summation.
"""

from lucid.utils.transforms import Segmentation
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights

_PRESET = Segmentation(crop_size=512, resize_size=512)


@register_weights("maskformer_resnet50")
class MaskFormerResNet50Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.maskformer_resnet50`.

    Single ADE20k semantic checkpoint converted from
    ``facebook/maskformer-resnet50-ade`` (Cheng et al., 2021; ~41.3M
    params, 44.5 mIoU on the ADE20k validation set).

    Examples
    --------
    >>> from lucid.models.vision.maskformer import MaskFormerResNet50Weights
    >>> list(MaskFormerResNet50Weights.__members__)
    ['ADE20K', 'DEFAULT']

    ``DEFAULT`` is an alias for the tag a bare
    ``pretrained=True`` resolves to, so the two ways of asking for
    weights cannot drift apart.
    """

    ADE20K = WeightEntry(
        url=f"{HUB_BASE}/maskformer-resnet-50/resolve/main/" "ADE20K/model.safetensors",
        sha256="31f13cd088b9acf2871fc44421fbce7ff2252e9e0670c55c8b2ee58eb7f888e2",
        num_classes=150,
        transforms=_PRESET,
        meta={
            "tag": "ADE20K",
            "source": "facebook/maskformer-resnet50-ade",
            "license": "other",
            "num_params": 41_307_863,
            "metrics": {"ADE20K": {"mIoU": 44.5}},
        },
        # ENC_LAYERS: 0 in the semantic-segmentation config these come
        # from -- C5 goes straight into the decoder.  A six-layer encoder
        # has no weights behind it and fails on 72 keys.
        requires_config={"num_encoder_layers": 0},
    )
    DEFAULT = ADE20K


@register_weights("maskformer_resnet101")
class MaskFormerResNet101Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.maskformer_resnet101`.

    Single ADE20k semantic checkpoint converted from
    ``facebook/maskformer-resnet101-ade`` (Cheng et al., 2021; ~60.3M
    params, 45.5 mIoU on the ADE20k validation set).

    Examples
    --------
    >>> from lucid.models.vision.maskformer import MaskFormerResNet101Weights
    >>> list(MaskFormerResNet101Weights.__members__)
    ['ADE20K', 'DEFAULT']

    ``DEFAULT`` is an alias for the tag a bare
    ``pretrained=True`` resolves to, so the two ways of asking for
    weights cannot drift apart.
    """

    ADE20K = WeightEntry(
        url=f"{HUB_BASE}/maskformer-resnet-101/resolve/main/" "ADE20K/model.safetensors",
        sha256="224ba1214a3b2b5d5c6e4fb01cc9ccf5cbb79a297f0e5898eb0860e901aefffd",
        num_classes=150,
        transforms=_PRESET,
        meta={
            "tag": "ADE20K",
            "source": "facebook/maskformer-resnet101-ade",
            "license": "other",
            "num_params": 60_299_991,
            "metrics": {"ADE20K": {"mIoU": 45.5}},
        },
        # ENC_LAYERS: 0 in the semantic-segmentation config these come
        # from -- C5 goes straight into the decoder.  A six-layer encoder
        # has no weights behind it and fails on 72 keys.
        requires_config={"num_encoder_layers": 0},
    )
    DEFAULT = ADE20K
