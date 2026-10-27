"""Pretrained-weight declarations for the YOLO family.

Four COCO detectors converted straight from the original darknet
``.weights`` releases by :mod:`tools.convert_weights.yolo` — no
intermediate framework is involved, so these are the authors' own trained
parameters rather than a reimplementation's.  Each checkpoint's tensor
budget matches its darknet blob exactly (see the converter's module
docstring for the arithmetic), and all four reproduce darknet's published
detections on ``data/dog.jpg``.

Preprocessing is darknet's, which is unusual for this zoo: letterbox to a
square canvas and feed raw :math:`[0, 1]` pixels.  No dataset mean/std is
subtracted anywhere in darknet's pipeline, so the presets below pass
``mean = 0`` / ``std = 1``.  The letterbox pad value is 0.5 grey.

``yolo`` has no entry here: the weights pjreddie published under
``yolov1/`` are for a different configuration than the paper's detector,
and they do not fit this family's ``yolo`` topology.
"""

from lucid.utils.transforms import Detection
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights

# Darknet letterbox + [0, 1] pixels.  v2 / v4 ship at 608, the v3 pair at 416
# — the resolution each release quotes its COCO mAP at.  The letterbox is
# centred, so ``postprocess`` boxes are in canvas coordinates: pass the canvas
# size as ``image_sizes``, then ``transforms().to_image_boxes`` maps them back
# onto the image, undoing the letterbox offset and the resize.
_PRESET_416 = Detection(
    max_size=416, pad_position="center", pad_value=0.5, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)
)
_PRESET_608 = Detection(
    max_size=608, pad_position="center", pad_value=0.5, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)
)


@register_weights("yolo_v2")
class YOLOV2Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.yolo_v2`.

    Single COCO checkpoint (:attr:`COCO_2014`) converted from darknet's
    ``yolov2.weights``, hosted under ``huggingface.co/lucid-dl/yolo-v2``.
    Evaluated at :math:`608 \times 608`, where the darknet release reports
    ``mAP@0.5 = 48.1``.

    Examples
    --------
    >>> from lucid.models.vision.yolo import YOLOV2Weights
    >>> list(YOLOV2Weights.__members__)
    ['COCO_2014', 'DEFAULT']

    ``DEFAULT`` is an alias for the tag a bare
    ``pretrained=True`` resolves to, so the two ways of asking for
    weights cannot drift apart.
    """

    COCO_2014 = WeightEntry(
        url=f"{HUB_BASE}/yolo-v2/resolve/main/COCO_2014/model.safetensors",
        sha256="2a6bf0b3e5ce1c2976017f6fbd736cd3d6259f0dec798fafc7718a0a6e0dc367",
        num_classes=80,
        transforms=_PRESET_608,
        meta={
            "tag": "COCO_2014",
            "source": "darknet/yolov2.weights",
            "license": "other",
            "num_params": 50_962_889,
            "file_size_mb": 194.5,
            "metrics": {"COCO": {"mAP@0.5": 48.1}},
        },
    )
    DEFAULT = COCO_2014


@register_weights("yolo_v3")
class YOLOV3Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.yolo_v3`.

    Single COCO checkpoint (:attr:`COCO_2014`) converted from darknet's
    ``yolov3.weights``, hosted under ``huggingface.co/lucid-dl/yolo-v3``.
    Evaluated at :math:`416 \times 416`, where the darknet release reports
    ``mAP@0.5 = 55.3``.

    Examples
    --------
    >>> from lucid.models.vision.yolo import YOLOV3Weights
    >>> list(YOLOV3Weights.__members__)
    ['COCO_2014', 'DEFAULT']

    ``DEFAULT`` is an alias for the tag a bare
    ``pretrained=True`` resolves to, so the two ways of asking for
    weights cannot drift apart.
    """

    COCO_2014 = WeightEntry(
        url=f"{HUB_BASE}/yolo-v3/resolve/main/COCO_2014/model.safetensors",
        sha256="ff360a2297a04af1ef11f7ffaf36b49204e298cf28748e0390aee10278ae70fb",
        num_classes=80,
        transforms=_PRESET_416,
        meta={
            "tag": "COCO_2014",
            "source": "darknet/yolov3.weights",
            "license": "other",
            "num_params": 61_949_149,
            "file_size_mb": 236.57,
            "metrics": {"COCO": {"mAP@0.5": 55.3}},
        },
    )
    DEFAULT = COCO_2014


@register_weights("yolo_v3_tiny")
class YOLOV3TinyWeights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.yolo_v3_tiny`.

    Single COCO checkpoint (:attr:`COCO_2014`) converted from darknet's
    ``yolov3-tiny.weights``, hosted under
    ``huggingface.co/lucid-dl/yolo-v3-tiny``.  Evaluated at
    :math:`416 \times 416`, where the darknet release reports
    ``mAP@0.5 = 33.1``.

    Examples
    --------
    >>> from lucid.models.vision.yolo import YOLOV3TinyWeights
    >>> list(YOLOV3TinyWeights.__members__)
    ['COCO_2014', 'DEFAULT']

    ``DEFAULT`` is an alias for the tag a bare
    ``pretrained=True`` resolves to, so the two ways of asking for
    weights cannot drift apart.
    """

    COCO_2014 = WeightEntry(
        url=f"{HUB_BASE}/yolo-v3-tiny/resolve/main/COCO_2014/model.safetensors",
        sha256="194beccbec3bfa7b228286290fbfe0056c4c08881ff7558f84cf8aacd63c9a3b",
        num_classes=80,
        transforms=_PRESET_416,
        meta={
            "tag": "COCO_2014",
            "source": "darknet/yolov3-tiny.weights",
            "license": "other",
            "num_params": 8_852_366,
            "file_size_mb": 33.8,
            "metrics": {"COCO": {"mAP@0.5": 33.1}},
        },
    )
    DEFAULT = COCO_2014


@register_weights("yolo_v4")
class YOLOV4Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.yolo_v4`.

    Single COCO checkpoint (:attr:`COCO_2017`) converted from the AlexeyAB
    darknet release's ``yolov4.weights``, hosted under
    ``huggingface.co/lucid-dl/yolo-v4``.  Evaluated at
    :math:`608 \times 608`, where that release reports
    ``mAP@0.5 = 65.7`` on COCO ``test-dev``.

    Examples
    --------
    >>> from lucid.models.vision.yolo import YOLOV4Weights
    >>> list(YOLOV4Weights.__members__)
    ['COCO_2017', 'DEFAULT']

    ``DEFAULT`` is an alias for the tag a bare
    ``pretrained=True`` resolves to, so the two ways of asking for
    weights cannot drift apart.
    """

    COCO_2017 = WeightEntry(
        url=f"{HUB_BASE}/yolo-v4/resolve/main/COCO_2017/model.safetensors",
        sha256="2969aff0a7e0326aa1e1f20996139807684a6fb37f463204217a79e8f10ab755",
        num_classes=80,
        transforms=_PRESET_608,
        meta={
            "tag": "COCO_2017",
            "source": "darknet/yolov4.weights",
            "license": "other",
            "num_params": 64_363_101,
            "file_size_mb": 245.85,
            "metrics": {"COCO": {"mAP@0.5": 65.7}},
        },
    )
    DEFAULT = COCO_2017
