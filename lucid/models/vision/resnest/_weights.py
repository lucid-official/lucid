"""Pretrained-weight declarations for the ResNeSt family.

Four paper-cited variants (Zhang et al., CVPR Workshops 2022) ship
ImageNet-1k checkpoints converted from timm's ``resnest{50d,101e,200e,
269e}.in1k`` model zoo: :class:`ResNeSt50Weights`,
:class:`ResNeSt101Weights`, :class:`ResNeSt200Weights`, and
:class:`ResNeSt269Weights`.

Unlike most families, each ResNeSt variant uses a *different* eval
pipeline — the deeper models were trained / evaluated at progressively
larger resolutions: ResNeSt-50 at 224 crop / bilinear, ResNeSt-101 at
256 crop / bilinear, ResNeSt-200 at 320 crop / bicubic, and ResNeSt-269
at 416 crop / bicubic.  Each enum therefore carries its own inline
:class:`ImageClassification` preset (all use ImageNet mean/std).
"""

from lucid.utils.transforms import ImageClassification
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights


@register_weights("resnest_50_cls")
class ResNeSt50Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.resnest_50_cls`.

    Single ImageNet-1k checkpoint converted from timm's
    ``resnest50d.in1k`` (Zhang et al., 2022; ~27.5M params, 81.1%
    top-1).  Evaluated at 224 crop / 256 resize / bilinear.

    Examples
    --------
    >>> from lucid.models.vision.resnest._weights import ResNeSt50Weights
    >>> list(ResNeSt50Weights.__members__)
    ['IN1K', 'DEFAULT']

    ``DEFAULT`` is an alias rather than a fourth entry, so a bare
    ``pretrained=True`` and the tag it resolves to cannot drift apart.

    >>> ResNeSt50Weights.DEFAULT is ResNeSt50Weights.IN1K
    True
    >>> ResNeSt50Weights.IN1K.num_classes
    1000
    """

    IN1K = WeightEntry(
        url=f"{HUB_BASE}/resnest-50/resolve/main/IN1K/model.safetensors",
        sha256="e9609ee480a096682e04c34acf0b13e2161258862060f09a1bacddd7eddf8b48",
        num_classes=1000,
        transforms=ImageClassification(crop_size=224, resize_size=256, interpolation="bilinear"),
        meta={
            "tag": "IN1K",
            "source": "timm/resnest50d.in1k",
            "license": "apache-2.0",
            "num_params": 27_483_240,
            "metrics": {"ImageNet-1k": {"acc@1": 81.1}},
        },
    )
    DEFAULT = IN1K


@register_weights("resnest_101_cls")
class ResNeSt101Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.resnest_101_cls`.

    Single ImageNet-1k checkpoint converted from timm's
    ``resnest101e.in1k`` (Zhang et al., 2022; ~48.3M params, 82.8%
    top-1).  Evaluated at 256 crop / 292 resize / bilinear.

    Examples
    --------
    >>> from lucid.models.vision.resnest._weights import ResNeSt101Weights
    >>> list(ResNeSt101Weights.__members__)
    ['IN1K', 'DEFAULT']

    ``DEFAULT`` is an alias rather than a fourth entry, so a bare
    ``pretrained=True`` and the tag it resolves to cannot drift apart.

    >>> ResNeSt101Weights.DEFAULT is ResNeSt101Weights.IN1K
    True
    >>> ResNeSt101Weights.IN1K.num_classes
    1000
    """

    IN1K = WeightEntry(
        url=f"{HUB_BASE}/resnest-101/resolve/main/IN1K/model.safetensors",
        sha256="b3403d554ee27612f25bd646c7202b0520f697e8150227538ae7ec5474a8479b",
        num_classes=1000,
        transforms=ImageClassification(crop_size=256, resize_size=292, interpolation="bilinear"),
        meta={
            "tag": "IN1K",
            "source": "timm/resnest101e.in1k",
            "license": "apache-2.0",
            "num_params": 48_275_016,
            "metrics": {"ImageNet-1k": {"acc@1": 82.8}},
        },
    )
    DEFAULT = IN1K


@register_weights("resnest_200_cls")
class ResNeSt200Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.resnest_200_cls`.

    Single ImageNet-1k checkpoint converted from timm's
    ``resnest200e.in1k`` (Zhang et al., 2022; ~70.2M params, 83.9%
    top-1).  Evaluated at 320 crop / 352 resize / bicubic.

    Examples
    --------
    >>> from lucid.models.vision.resnest._weights import ResNeSt200Weights
    >>> list(ResNeSt200Weights.__members__)
    ['IN1K', 'DEFAULT']

    ``DEFAULT`` is an alias rather than a fourth entry, so a bare
    ``pretrained=True`` and the tag it resolves to cannot drift apart.

    >>> ResNeSt200Weights.DEFAULT is ResNeSt200Weights.IN1K
    True
    >>> ResNeSt200Weights.IN1K.num_classes
    1000
    """

    IN1K = WeightEntry(
        url=f"{HUB_BASE}/resnest-200/resolve/main/IN1K/model.safetensors",
        sha256="9adc2f6282549bce1b1289a42fa4ddbc722ec5759b2e8d0decf154575ec5a065",
        num_classes=1000,
        transforms=ImageClassification(crop_size=320, resize_size=352, interpolation="bicubic"),
        meta={
            "tag": "IN1K",
            "source": "timm/resnest200e.in1k",
            "license": "apache-2.0",
            "num_params": 70_201_544,
            "metrics": {"ImageNet-1k": {"acc@1": 83.9}},
        },
        # timm's head is a bare Linear; §4.2's dropout makes Lucid's a
        # Sequential for the two deep variants, which moves the same
        # weights to classifier.1.  Dropout has no parameters, so this
        # renames and nothing else.
        key_map={
            "classifier.weight": "classifier.1.weight",
            "classifier.bias": "classifier.1.bias",
        },
    )
    DEFAULT = IN1K


@register_weights("resnest_269_cls")
class ResNeSt269Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.resnest_269_cls`.

    Single ImageNet-1k checkpoint converted from timm's
    ``resnest269e.in1k`` (Zhang et al., 2022; ~110.9M params, 84.5%
    top-1) — the deepest paper-cited variant.  Evaluated at 416 crop /
    448 resize / bicubic.

    Examples
    --------
    >>> from lucid.models.vision.resnest._weights import ResNeSt269Weights
    >>> list(ResNeSt269Weights.__members__)
    ['IN1K', 'DEFAULT']

    ``DEFAULT`` is an alias rather than a fourth entry, so a bare
    ``pretrained=True`` and the tag it resolves to cannot drift apart.

    >>> ResNeSt269Weights.DEFAULT is ResNeSt269Weights.IN1K
    True
    >>> ResNeSt269Weights.IN1K.num_classes
    1000
    """

    IN1K = WeightEntry(
        url=f"{HUB_BASE}/resnest-269/resolve/main/IN1K/model.safetensors",
        sha256="8ddfb4eb21e96a48a27fcf9d0b66105514cbfcab37a2b5289fa349dbb9e88034",
        num_classes=1000,
        transforms=ImageClassification(crop_size=416, resize_size=448, interpolation="bicubic"),
        meta={
            "tag": "IN1K",
            "source": "timm/resnest269e.in1k",
            "license": "apache-2.0",
            "num_params": 110_929_480,
            "metrics": {"ImageNet-1k": {"acc@1": 84.5}},
        },
        # timm's head is a bare Linear; §4.2's dropout makes Lucid's a
        # Sequential for the two deep variants, which moves the same
        # weights to classifier.1.  Dropout has no parameters, so this
        # renames and nothing else.
        key_map={
            "classifier.weight": "classifier.1.weight",
            "classifier.bias": "classifier.1.bias",
        },
    )
    DEFAULT = IN1K
