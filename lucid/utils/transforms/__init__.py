"""``lucid.utils.transforms`` — image transform library.

An Albumentations-compatible transform library (matching class names +
constructor signatures) built tensor-native on Lucid (no numpy / PIL,
per H4).  Transforms consume / produce :class:`lucid.Tensor` images
``(C, H, W)`` or ``(B, C, H, W)``; image *decoding* (file → tensor)
lives outside this package.

Two entry points:

* **Class transforms** — :class:`Compose` a list of :class:`Transform`
  objects.  Each is applied with probability ``p`` and moves every
  typed target (:class:`Image` / :class:`Mask` / :class:`BoundingBoxes`
  / :class:`Keypoints`) consistently.
* **Functional** — stateless ops under
  :mod:`lucid.utils.transforms.functional`.

Presets (:class:`ImageClassification`) bundle a task's canonical
inference pipeline; pretrained weights ship preprocessing this way.

>>> import lucid
>>> import lucid.utils.transforms as T
>>> tf = T.Compose([
...     T.SmallestMaxSize(256),
...     T.CenterCrop(224, 224),
...     T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
... ])
>>> image = lucid.rand(3, 300, 400)             # (C, H, W)
>>> y = tf(image)
>>> y.shape
(3, 224, 224)
"""

from lucid.utils.transforms import functional
from lucid.utils.transforms._base import (
    BboxParams,
    Compose,
    GeometricTransform,
    PhotometricTransform,
    Transform,
)
from lucid.utils.transforms._datatypes import BoundingBoxes, Image, Keypoints, Mask
from lucid.utils.transforms._geometric import (
    CenterCrop,
    HorizontalFlip,
    LongestMaxSize,
    ResizeShortestEdge,
    RandomCrop,
    RandomResizedCrop,
    Resize,
    SmallestMaxSize,
    VerticalFlip,
)
from lucid.utils.transforms._blur import (
    Blur,
    Defocus,
    Downscale,
    GaussianBlur,
    GaussNoise,
    ISONoise,
    MedianBlur,
    MotionBlur,
    MultiplicativeNoise,
    ZoomBlur,
)
from lucid.utils.transforms._composition import OneOf, OneOrOther, ReplayCompose, Sequential, SomeOf
from lucid.utils.transforms._misc import (
    BBoxSafeRandomCrop,
    Lambda,
    MaskDropout,
    RandomCropNearBBox,
    RandomSizedBBoxSafeCrop,
)
from lucid.utils.transforms._color import (
    CLAHE,
    ChannelDropout,
    ChannelShuffle,
    Emboss,
    Equalize,
    FancyPCA,
    HueSaturationValue,
    InvertImg,
    PixelDropout,
    Posterize,
    RandomBrightness,
    RandomBrightnessContrast,
    RandomContrast,
    RandomGamma,
    RandomToneCurve,
    RGBShift,
    RingingOvershoot,
    Sharpen,
    Solarize,
    ToGray,
    ToSepia,
    UnsharpMask,
    XYMasking,
)
from lucid.utils.transforms._crop import Crop, CropAndPad, PadIfNeeded, RandomSizedCrop
from lucid.utils.transforms._distortion import (
    ElasticTransform,
    GridDistortion,
    GridElasticDeform,
    OpticalDistortion,
)
from lucid.utils.transforms._autoaugment import AutoAugment, RandAugment, TrivialAugmentWide
from lucid.utils.transforms._dropout import CoarseDropout, GridDropout
from lucid.utils.transforms._erasing import RandomErasing
from lucid.utils.transforms._interpolation import Interpolation
from lucid.utils.transforms._photometric import ColorJitter, FromFloat, Normalize, ToFloat
from lucid.utils.transforms._presets import (
    AutoTransformsPreset,
    Detection,
    ImageClassification,
    ImageClassificationAugment,
    VideoClassification,
    Pose,
    Segmentation,
    TransformsPreset,
)
from lucid.utils.transforms._spatial import (
    Affine,
    D4,
    Flip,
    Perspective,
    RandomGridShuffle,
    RandomRotate90,
    RandomScale,
    Rotate,
    SafeRotate,
    ShiftScaleRotate,
    Transpose,
)

__all__ = [
    # Base hierarchy
    "Transform",
    "GeometricTransform",
    "PhotometricTransform",
    "Compose",
    "BboxParams",
    # Interpolation
    "Interpolation",
    # Typed targets
    "Image",
    "Mask",
    "BoundingBoxes",
    "Keypoints",
    # Geometric — resize / crop
    "Resize",
    "SmallestMaxSize",
    "LongestMaxSize",
    "ResizeShortestEdge",
    "CenterCrop",
    "RandomCrop",
    "RandomResizedCrop",
    # Geometric — flips / transpose / rot90
    "HorizontalFlip",
    "VerticalFlip",
    "Flip",
    "Transpose",
    "RandomRotate90",
    "RandomScale",
    "D4",
    # Geometric — affine warps
    "Rotate",
    "SafeRotate",
    "ShiftScaleRotate",
    "Affine",
    "Perspective",
    # Geometric — distortion
    "ElasticTransform",
    "GridDistortion",
    "OpticalDistortion",
    "GridElasticDeform",
    "RandomGridShuffle",
    # Geometric — crop / pad
    "Crop",
    "PadIfNeeded",
    "RandomSizedCrop",
    "CropAndPad",
    "BBoxSafeRandomCrop",
    "RandomSizedBBoxSafeCrop",
    "RandomCropNearBBox",
    # Dropout / occlusion / erasing
    "CoarseDropout",
    "GridDropout",
    "MaskDropout",
    "PixelDropout",
    "RandomErasing",
    "XYMasking",
    # Photometric — value scaling
    "ToFloat",
    "FromFloat",
    "Normalize",
    "ColorJitter",
    # Photometric — colour / pixel
    "RandomBrightnessContrast",
    "RandomGamma",
    "HueSaturationValue",
    "RGBShift",
    "ChannelShuffle",
    "ChannelDropout",
    "Equalize",
    "CLAHE",
    "Solarize",
    "Posterize",
    "InvertImg",
    "ToGray",
    "ToSepia",
    "Sharpen",
    "Emboss",
    "RandomToneCurve",
    "RandomBrightness",
    "RandomContrast",
    "UnsharpMask",
    "RingingOvershoot",
    "FancyPCA",
    # Blur / noise
    "Blur",
    "MedianBlur",
    "MotionBlur",
    "GaussianBlur",
    "GaussNoise",
    "MultiplicativeNoise",
    "ISONoise",
    "Downscale",
    "Defocus",
    "ZoomBlur",
    # Composition
    "OneOf",
    "SomeOf",
    "Sequential",
    "OneOrOther",
    "ReplayCompose",
    # Utility
    "Lambda",
    # AutoAugment family
    "AutoAugment",
    "RandAugment",
    "TrivialAugmentWide",
    # Presets — pretrained preprocessing contract
    "TransformsPreset",
    "AutoTransformsPreset",
    "ImageClassification",
    "ImageClassificationAugment",
    "VideoClassification",
    "Detection",
    "Segmentation",
    "Pose",
    # Functional submodule
    "functional",
]
