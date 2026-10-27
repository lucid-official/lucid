"""Pretrained-weight declarations for the MobileNet v1 family.

One paper-cited variant (Howard et al., 2017) ships an ImageNet-1k
checkpoint converted from timm's
``mobilenetv1_100.ra4_e3600_r224_in1k`` model zoo:
:class:`MobileNetWeights`.  This is the full-width
(:math:`\\alpha = 1.0`) classifier; the slim 0.75 / 0.5 / 0.25 width
variants are not shipped because no clean upstream checkpoint exists
for them.

The checkpoint uses timm's RA4 training recipe eval pipeline: 224 crop
/ 256 resize / bicubic interpolation, and — unlike the ImageNet default
— inception-style ``(0.5, 0.5, 0.5)`` mean/std normalisation.
"""

from lucid.utils.transforms import ImageClassification
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights

_PRESET = ImageClassification(
    crop_size=224,
    resize_size=256,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
    interpolation="bicubic",
)


@register_weights("mobilenet_cls")
class MobileNetWeights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.mobilenet_cls`.

    Single ImageNet-1k checkpoint converted from timm's
    ``mobilenetv1_100.ra4_e3600_r224_in1k`` (Howard et al., 2017;
    ~4.2M params, 75.4% top-1 at 224x224 under timm's RA4 recipe).

    Examples
    --------
    >>> from lucid.models.vision.mobilenet._weights import MobileNetWeights
    >>> list(MobileNetWeights.__members__)
    ['RA4_E3600_R224_IN1K', 'DEFAULT']

    ``DEFAULT`` is an alias rather than a fourth entry, so a bare
    ``pretrained=True`` and the tag it resolves to cannot drift apart.

    >>> MobileNetWeights.DEFAULT is MobileNetWeights.RA4_E3600_R224_IN1K
    True
    >>> MobileNetWeights.RA4_E3600_R224_IN1K.num_classes
    1000
    """

    RA4_E3600_R224_IN1K = WeightEntry(
        url=(f"{HUB_BASE}/mobilenet-v1/resolve/main/" "RA4_E3600_R224_IN1K/model.safetensors"),
        sha256="967fd814112842854b83875655502eacb87ed18dd808566eb84e686c058096aa",
        num_classes=1000,
        transforms=_PRESET,
        meta={
            "tag": "RA4_E3600_R224_IN1K",
            "source": "timm/mobilenetv1_100.ra4_e3600_r224_in1k",
            "license": "apache-2.0",
            "num_params": 4_231_976,
            "metrics": {"ImageNet-1k": {"acc@1": 75.4}},
        },
    )
    DEFAULT = RA4_E3600_R224_IN1K
