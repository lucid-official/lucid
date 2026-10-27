"""Ready-made transform presets — Lucid's pretrained-pipeline contract.

A *preset* is a high-level, task-specific preprocessing pipeline that
knows how to serialise itself to a JSON-friendly dict and reconstruct
back — the canonical handoff between a :class:`~lucid.weights.WeightEntry`
and the actual transform sequence a pretrained model expects.

Design borrows from both Hugging Face ``transformers`` (model-family
``ImageProcessor`` + ``preprocessor_config.json``) and timm
(``default_cfg`` flat dict + builder).  Each preset is a class
(HF-style); the standard task presets (:class:`ImageClassification`,
:class:`ImageClassificationAugment`, :class:`Detection`,
:class:`Segmentation`, :class:`Pose`) cover the common cases without
per-model boilerplate (timm-style).  When a specific model needs
custom preprocessing (e.g. Inception's ``299×299`` bicubic), subclass
the matching task preset and override what differs.

Round-trip contract::

    cfg = preset.to_dict()                # JSON-friendly
    restored = AutoTransformsPreset.from_dict(cfg)
    assert type(restored) is type(preset)
    assert restored.to_dict() == cfg

Every preset is also a regular :class:`~lucid.utils.transforms.Transform`
— call it on an image tensor or a multi-target sample dict, both work.

Examples
--------
>>> import lucid
>>> from lucid.utils.transforms import (
...     ImageClassification, AutoTransformsPreset,
... )
>>> preset = ImageClassification(crop_size=224, resize_size=256)
>>> y = preset(lucid.rand(3, 300, 400))           # tensor in, tensor out
>>> cfg = preset.to_dict()                         # round-trip
>>> back = AutoTransformsPreset.from_dict(cfg)
>>> back.to_dict() == cfg
True
"""

import abc
from typing import ClassVar, cast, override

import lucid
from lucid._tensor import Tensor
from lucid.utils.transforms._autoaugment import AutoAugment, RandAugment, TrivialAugmentWide
from lucid.utils.transforms._base import (
    BboxParams,
    Compose,
    Empty,
    Transform,
    TransformLike,
    _NoParams,
)
from lucid.utils.transforms._crop import PadIfNeeded
from lucid.utils.transforms._erasing import RandomErasing
from lucid.utils.transforms._geometric import (
    CenterCrop,
    HorizontalFlip,
    LongestMaxSize,
    RandomResizedCrop,
    Resize,
    ResizeShortestEdge,
    SmallestMaxSize,
)
from lucid.utils.transforms._interpolation import Interpolation
from lucid.utils.transforms._photometric import ColorJitter, Normalize
from lucid.utils.transforms.functional import resize, resize_target

# ── registry + auto-resolver ────────────────────────────────────────


_PRESET_REGISTRY: dict[str, type[TransformsPreset]] = {}


def _register_preset(cls: type[TransformsPreset]) -> type[TransformsPreset]:
    """Class decorator: register ``cls`` under its ``preset_type`` key."""
    key = cls.preset_type
    if key in _PRESET_REGISTRY and _PRESET_REGISTRY[key] is not cls:
        raise ValueError(
            f"preset key {key!r} already registered to "
            f"{_PRESET_REGISTRY[key].__qualname__}; conflict with "
            f"{cls.__qualname__}"
        )
    _PRESET_REGISTRY[key] = cls
    return cls


# ── base class ──────────────────────────────────────────────────────


_IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
_IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


class TransformsPreset(_NoParams, Transform[Empty], abc.ABC):
    r"""Abstract base for task-specific preprocessing presets.

    A preset is a thin wrapper over an internal :class:`Compose`
    pipeline (assembled in :meth:`__init__` from the preset's task-
    specific defaults) plus a serialisation contract.  Subclasses
    must:

    1. Set the :attr:`preset_type` class variable to a unique short
       string used in ``config.json``.
    2. Declare the constructor's typed kwargs (no ``**kwargs`` —
       enforces the schema explicitly).
    3. Build ``self._pipeline`` (a :class:`Compose`) inside
       :meth:`__init__`.
    4. Implement :meth:`_init_kwargs` returning the dict of constructor
       arguments needed to reconstruct an identical instance.

    Attributes
    ----------
    preset_type : ClassVar[str]
        Short identifier (e.g. ``"ImageClassification"``) used as the
        ``preprocessor_type`` key in :meth:`to_dict` / :meth:`from_dict`.

    Examples
    --------
    The base of the ready-made pipelines — :class:`ImageClassification`,
    :class:`ImageClassificationAugment`, :class:`Detection`,
    :class:`Segmentation`, :class:`Pose`, :class:`VideoClassification`.
    Each is the preprocessing its task expects, so a caller reaches for
    one rather than assembling resize, crop and normalise by hand:

    >>> import lucid.utils.transforms as T
    >>> sorted(c.__name__ for c in T.TransformsPreset.__subclasses__())
    ['Detection', 'ImageClassification', 'ImageClassificationAugment', 'Pose', 'Segmentation', 'VideoClassification']
    """

    preset_type: ClassVar[str]

    _pipeline: Compose

    # ── serialisation contract ──────────────────────────────────────

    @abc.abstractmethod
    def _init_kwargs(self) -> dict[str, object]:
        """Return the constructor kwargs that reconstruct ``self``.

        The values must be JSON-friendly (``int`` / ``float`` / ``str``
        / ``list`` / ``dict`` / ``bool`` / ``None``).
        """

    def to_dict(self) -> dict[str, object]:
        r"""Serialise the preset to a JSON-friendly dict.

        Mirrors Hugging Face's ``preprocessor_config.json`` shape:
        ``{"preprocessor_type": <class-key>, "init_kwargs": {…}}``.

        Returns
        -------
        dict
            Two-key dict suitable for :func:`json.dump` and
            :meth:`AutoTransformsPreset.from_dict` round-trip.
        """
        return {"preprocessor_type": self.preset_type, "init_kwargs": self._init_kwargs()}

    @classmethod
    def from_dict(cls, cfg: dict[str, object]) -> TransformsPreset:
        r"""Reconstruct a preset instance from a :meth:`to_dict` payload.

        When called on the base :class:`TransformsPreset`, dispatches
        to the registered subclass named by ``cfg["preprocessor_type"]``;
        when called on a concrete subclass, verifies the key matches
        and constructs that subclass directly.

        Parameters
        ----------
        cfg : dict
            ``{"preprocessor_type": ..., "init_kwargs": {...}}``.

        Returns
        -------
        TransformsPreset
            New instance of the named subclass.

        Raises
        ------
        KeyError
            If ``preprocessor_type`` is not in the registry.
        ValueError
            If ``cfg`` is missing required keys or ``preprocessor_type``
            disagrees with the class :meth:`from_dict` was called on.
        """
        key = cfg.get("preprocessor_type")
        kwargs = cfg.get("init_kwargs", {})
        if key is None:
            raise ValueError(f"from_dict: missing 'preprocessor_type' in {cfg!r}")
        if not isinstance(key, str):
            raise TypeError(
                f"from_dict: 'preprocessor_type' must be a str, got {type(key).__name__}"
            )
        if not isinstance(kwargs, dict):
            raise TypeError(f"from_dict: 'init_kwargs' must be a dict, got {type(kwargs).__name__}")
        target = _PRESET_REGISTRY.get(key)
        if target is None:
            raise KeyError(
                f"unknown preset type {key!r}; registered: " f"{sorted(_PRESET_REGISTRY)}"
            )
        if cls is not TransformsPreset and target is not cls:
            raise ValueError(f"from_dict on {cls.__qualname__} got cfg for {target.__qualname__}")
        return target(**kwargs)

    # ── Transform interface (delegates to the inner pipeline) ───────

    @override
    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:
        return cast(Tensor, self._pipeline(img))

    @override
    def __call__(self, inputs: object) -> object:
        # Wholesale delegation so multi-target samples thread through
        # the inner stages with consistent params per call.
        return self._pipeline(inputs)

    @override
    def __repr__(self) -> str:
        kw = ", ".join(f"{k}={v!r}" for k, v in self._init_kwargs().items())
        return f"{type(self).__name__}({kw})"


class AutoTransformsPreset:
    r"""Resolver mirroring Hugging Face's ``AutoImageProcessor``.

    Stateless utility — no instances are constructed; the classmethods
    look up the registry by ``preprocessor_type`` and dispatch.

    Examples
    --------
    >>> from lucid.utils.transforms import AutoTransformsPreset
    >>> cfg = {
    ...     "preprocessor_type": "ImageClassification",
    ...     "init_kwargs": {"crop_size": 224, "resize_size": 256},
    ... }
    >>> preset = AutoTransformsPreset.from_dict(cfg)
    >>> type(preset).__name__
    'ImageClassification'
    """

    @staticmethod
    def from_dict(cfg: dict[str, object]) -> TransformsPreset:
        """Reconstruct a preset from a :meth:`TransformsPreset.to_dict` payload.

        Dispatches on the ``preprocessor_type`` key to the registered
        :class:`TransformsPreset` subclass and forwards ``init_kwargs``
        to its constructor.  Symmetric with :meth:`TransformsPreset.to_dict`
        — the round-trip ``from_dict(p.to_dict())`` produces a preset
        with identical behaviour.

        Parameters
        ----------
        cfg : dict
            ``{"preprocessor_type": <class-key>, "init_kwargs": {...}}``
            payload — typically loaded from a pretrained model's
            ``preprocessor_config.json``.

        Returns
        -------
        TransformsPreset
            Fresh instance of the named subclass.

        Raises
        ------
        ValueError
            If ``preprocessor_type`` is missing or names an unknown
            preset; if ``init_kwargs`` is not a dict.
        """
        return TransformsPreset.from_dict(cfg)

    @staticmethod
    def registered() -> list[str]:
        """Return the sorted list of registered preset type keys.

        Useful for discovery / introspection — every returned name is
        a legal value of the ``preprocessor_type`` field accepted by
        :meth:`from_dict`.
        """
        return sorted(_PRESET_REGISTRY)


# ── concrete presets ────────────────────────────────────────────────


def _resize_eight_bit(
    img: Tensor, size: tuple[int, int], interpolation: str | Interpolation
) -> Tensor:
    """8-bit pixels resized the way PIL resizes them, returned in ``[0, 1]``.

    The checkpoints these presets reproduce were evaluated on PIL images:
    an 8-bit resize, horizontal pass then vertical, each rounded and clipped
    back to 0..255.  Resizing the same pixels in float keeps fractions PIL
    throws away, and some models notice — ``convnext_xlarge`` agreed with
    its source on 96.4% of ImageNet-V2 that way and on 100% of the images
    once its numerics were fed the source's own preprocessing.  Spelled as
    PIL does it, the pixels match PIL's to the last level.
    """
    h, w = int(img.shape[-2]), int(img.shape[-1])
    th, tw = size
    x = img.to(lucid.float32)
    if tw != w:
        x = resize(x, (h, tw), interpolation=interpolation, antialias=True)
        x = x.round().clamp(0.0, 255.0)
    if th != h:
        x = resize(x, (th, tw), interpolation=interpolation, antialias=True)
        x = x.round().clamp(0.0, 255.0)
    return x / 255.0


class _ReferenceShorterSide(SmallestMaxSize):
    """``SmallestMaxSize`` sized the way the reference's ``Resize(int)`` is.

    Albumentations rounds the longer side and the reference truncates it,
    so an 876 x 1300 photo comes out 380 wide one way and 379 the other,
    and the centre crop after it lands a column apart.  The published
    checkpoints were evaluated the reference's way; the evaluation presets
    that reproduce them resize that way, and low-pass filter as they shrink
    the way the reference does, while
    :class:`~lucid.utils.transforms.SmallestMaxSize` itself keeps the
    Albumentations rules it documents.
    """

    @override
    def _target(self, h: int, w: int) -> tuple[int, int]:
        return resize_target(h, w, self.max_size)

    @override
    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:
        h, w = int(img.shape[-2]), int(img.shape[-1])
        if not img.is_floating_point():
            return _resize_eight_bit(img, self._target(h, w), self.interpolation)
        return resize(img, self._target(h, w), interpolation=self.interpolation, antialias=True)


class _ReferenceStretch(Resize):
    """``Resize`` that low-pass filters as it shrinks.

    For the presets that reproduce an upstream processor configured with an
    explicit ``(height, width)``: those resize through PIL, which always
    filters, where :class:`~lucid.utils.transforms.Resize` keeps the
    Albumentations (OpenCV) behaviour of not filtering.
    """

    @override
    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:
        if not img.is_floating_point():
            return _resize_eight_bit(img, (self.height, self.width), self.interpolation)
        return resize(
            img, (self.height, self.width), interpolation=self.interpolation, antialias=True
        )


@_register_preset
class ImageClassification(TransformsPreset):
    r"""Standard ImageNet classification *inference* preset.

    Pipeline: shorter side to ``resize_size`` (the reference's rule: the
    longer side truncated, and low-pass filtered as it shrinks) →
    ``CenterCrop(crop_size)`` → ``Normalize(mean, std)``.  This is the
    reference_vision evaluation pipeline shipped with most pretrained
    image-classification weights.

    Parameters
    ----------
    crop_size : int
        Square center-crop side fed to the model.
    resize_size : int, optional, default=256
        Shorter-side length before cropping.
    mean, std : tuple of float, optional
        Per-channel normalization stats; default ImageNet.
    interpolation : str or Interpolation, optional, default="bilinear"
        Resize interpolation mode.

    Notes
    -----
    Input is assumed already scaled to ``[0, 1]`` (prepend
    :class:`~lucid.utils.transforms.ToFloat` for uint8).

    Examples
    --------
    The evaluation pipeline every ImageNet model in the zoo expects —
    resize the shortest side, centre crop, normalise:

    >>> import lucid, lucid.utils.transforms as T
    >>> tf = T.ImageClassification(crop_size=224)
    >>> tuple(tf(T.Image(lucid.rand(3, 300, 400))).data.shape)
    (3, 224, 224)
    """

    preset_type: ClassVar[str] = "ImageClassification"

    def __init__(
        self,
        crop_size: int,
        *,
        resize_size: int = 256,
        mean: tuple[float, ...] | None = None,
        std: tuple[float, ...] | None = None,
        interpolation: str | Interpolation = Interpolation.BILINEAR,
    ) -> None:
        self.crop_size = crop_size
        self.resize_size = resize_size
        self.mean = mean if mean is not None else _IMAGENET_MEAN
        self.std = std if std is not None else _IMAGENET_STD
        self.interpolation = interpolation
        self._pipeline = Compose(
            [
                _ReferenceShorterSide(resize_size, interpolation=interpolation),
                CenterCrop(crop_size, crop_size),
                Normalize(self.mean, self.std, max_pixel_value=1.0),
            ]
        )

    @override
    def _init_kwargs(self) -> dict[str, object]:
        interp = self.interpolation
        return {
            "crop_size": self.crop_size,
            "resize_size": self.resize_size,
            "mean": list(self.mean),
            "std": list(self.std),
            "interpolation": (interp.value if isinstance(interp, Interpolation) else interp),
        }


def _parse_auto_augment(spec: str) -> TransformLike:
    r"""Resolve a timm-style ``auto_augment`` spec to a Transform instance.

    Accepted forms::

        "ta_wide"                       → TrivialAugmentWide()
        "ta"                            → TrivialAugmentWide(num_magnitude_bins=10)
        "ra"                            → RandAugment()  (defaults: n=2, m=9)
        "ra-mM"                         → RandAugment(magnitude=M)
        "ra-nN"                         → RandAugment(num_ops=N)
        "ra-mM-nN" / "ra-nN-mM"         → RandAugment(num_ops=N, magnitude=M)
        "aa_imagenet" / "aa_cifar10"    → AutoAugment(policy=...)
        "aa_svhn"

    Mirrors the reference-framework convention so user code that
    already supplies ``"ra-m9"`` etc. works without translation.
    """
    if spec == "ta_wide":
        return TrivialAugmentWide(p=1.0)
    if spec == "ta":
        return TrivialAugmentWide(num_magnitude_bins=10, p=1.0)
    if spec.startswith("ra"):
        parts = spec.split("-")
        m, n = 9, 2
        for part in parts[1:]:
            if part.startswith("m"):
                m = int(part[1:])
            elif part.startswith("n"):
                n = int(part[1:])
            else:
                raise ValueError(f"unrecognised RandAugment fragment {part!r} in spec {spec!r}")
        return RandAugment(num_ops=n, magnitude=m, p=1.0)
    if spec.startswith("aa_"):
        policy = spec[3:]
        return AutoAugment(policy=policy, p=1.0)
    raise ValueError(
        f"unknown auto_augment spec {spec!r}; expected one of "
        '"ta", "ta_wide", "ra[-mM][-nN]", "aa_imagenet", "aa_cifar10", "aa_svhn"'
    )


@_register_preset
class VideoClassification(TransformsPreset):
    r"""Inference preset for a decoded video clip.

    The released video encoders preprocess a clip by running the
    ordinary ImageNet evaluation pipeline on **every frame** — shortest
    side to ``crop_size * 256 / 224``, centre crop, rescale, ImageNet
    normalise, bilinear throughout.  Nothing is done across time: no
    temporal resampling, no frame selection.  Which frames reach this
    is the caller's decision, because it is the decoder's.

    Decoding is not done here and cannot be: reading a container is an
    external dependency, and Lucid's compute path takes none (H4).  The
    input is an already-decoded clip.

    Parameters
    ----------
    crop_size : int
        Square centre crop, per frame.  The released encoders use 256,
        or 384 for the high-resolution ViT-g.
    resize_size : int or None, optional
        Shortest side before the crop.  Defaults to the released rule,
        ``int(crop_size * 256 / 224)`` — 292 for a 256 crop; it
        truncates, where rounding would give 293.
    mean, std : tuple of float, optional
        Channel statistics; ImageNet's by default.
    interpolation : str or Interpolation, optional
        Resampling filter, bilinear by default as the release uses.

    Examples
    --------
    A clip in, a clip out — ``(T, C, H, W)`` both sides:

    >>> import lucid, lucid.utils.transforms as T
    >>> preset = T.VideoClassification(crop_size=64)
    >>> tuple(preset(T.Image(lucid.rand(4, 3, 90, 120))).data.shape)
    (4, 3, 64, 64)
    """

    preset_type: ClassVar[str] = "VideoClassification"

    def __init__(
        self,
        crop_size: int,
        *,
        resize_size: int | None = None,
        mean: tuple[float, ...] | None = None,
        std: tuple[float, ...] | None = None,
        interpolation: str | Interpolation = Interpolation.BILINEAR,
    ) -> None:
        self.crop_size = crop_size
        # The release derives it rather than declaring it, and it
        # *truncates*: ``int(256 * 256 / 224)`` is 292, not the 293 that
        # rounding 292.57 would give.  One pixel of resize moves the
        # centre crop's window and rescales every pixel inside it — the
        # released processor's own config records shortest_edge 292.
        self.resize_size = resize_size if resize_size is not None else int(crop_size * 256 / 224)
        self.mean = mean if mean is not None else _IMAGENET_MEAN
        self.std = std if std is not None else _IMAGENET_STD
        self.interpolation = interpolation
        # Frames thread through the image stages on the leading axis,
        # which those stages already treat as a batch.
        # ``floor``, not the zoo's usual ``round``: these weights are
        # published through Hugging Face's fast processors, which place
        # an odd margin down.  A 519-wide frame cropped to 256 starts at
        # column 131 there and 132 under the rounding convention the
        # reference vision package uses — one column, and on anything
        # but a smooth frame a different picture.
        self._pipeline = Compose(
            [
                _ReferenceShorterSide(self.resize_size, interpolation=interpolation),
                CenterCrop(crop_size, crop_size, offset="floor"),
                Normalize(self.mean, self.std, max_pixel_value=1.0),
            ]
        )

    @override
    def _init_kwargs(self) -> dict[str, object]:
        interp = self.interpolation
        return {
            "crop_size": self.crop_size,
            "resize_size": self.resize_size,
            "mean": list(self.mean),
            "std": list(self.std),
            "interpolation": (interp.value if isinstance(interp, Interpolation) else interp),
        }


@_register_preset
class ImageClassificationAugment(TransformsPreset):
    r"""Standard ImageNet classification *training* preset (augmentation).

    Pipeline (stages 4–7 conditional, all share the same Normalize stats)::

        1. RandomResizedCrop(crop_size, scale, ratio)
        2. HorizontalFlip(p=hflip_prob)            — skipped if hflip_prob = 0
        3. AutoAugment / RandAugment / TAW         — skipped if auto_augment = None
        4. ColorJitter(brightness=contrast=...)    — skipped if color_jitter = 0
        5. Normalize(mean, std, max_pixel_value=1)
        6. RandomErasing(p=random_erasing)         — skipped if random_erasing = 0

    Mirrors the reference-framework training recipe shipped with most
    classification models.  Setting ``auto_augment`` and/or
    ``random_erasing`` reaches the strong-augmentation recipe used by
    timm / reference_vision references.

    Parameters
    ----------
    crop_size : int
        Square crop fed to the model.
    scale : tuple of float, optional, default=(0.08, 1.0)
        ``RandomResizedCrop`` scale range.
    ratio : tuple of float, optional, default=(0.75, 1.3333)
        ``RandomResizedCrop`` aspect-ratio range.
    color_jitter : float, optional, default=0.4
        Brightness / contrast / saturation jitter strength.  Zero
        disables ``ColorJitter`` entirely.
    hflip_prob : float, optional, default=0.5
        Horizontal-flip probability.  Zero disables the flip.
    auto_augment : str or None, optional, default=None
        AutoAugment-family policy.  See :func:`_parse_auto_augment`
        for accepted spec strings.  ``None`` disables this stage —
        recommended unless you already have a tuned recipe.
    random_erasing : float, optional, default=0.0
        :class:`~lucid.utils.transforms.RandomErasing` probability
        applied *after* :class:`Normalize` (mirrors reference
        ordering).  ``0.0`` disables the stage.
    mean, std : tuple of float, optional
        Per-channel normalization stats; default ImageNet.
    interpolation : str or Interpolation, optional, default="bilinear"

    Examples
    --------
    The training counterpart of :class:`ImageClassification` — a random
    resized crop and a horizontal flip instead of a centre crop:

    >>> import lucid, lucid.utils.transforms as T
    >>> tf = T.ImageClassificationAugment(crop_size=224)
    >>> tuple(tf(T.Image(lucid.rand(3, 300, 400))).data.shape)
    (3, 224, 224)
    """

    preset_type: ClassVar[str] = "ImageClassificationAugment"

    def __init__(
        self,
        crop_size: int,
        *,
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (0.75, 1.3333333333333333),
        color_jitter: float = 0.4,
        hflip_prob: float = 0.5,
        auto_augment: str | None = None,
        random_erasing: float = 0.0,
        mean: tuple[float, ...] | None = None,
        std: tuple[float, ...] | None = None,
        interpolation: str | Interpolation = Interpolation.BILINEAR,
    ) -> None:
        if not 0.0 <= random_erasing <= 1.0:
            raise ValueError(f"random_erasing must be in [0, 1], got {random_erasing}")
        self.crop_size = crop_size
        self.scale = scale
        self.ratio = ratio
        self.color_jitter = color_jitter
        self.hflip_prob = hflip_prob
        self.auto_augment = auto_augment
        self.random_erasing = random_erasing
        self.mean = mean if mean is not None else _IMAGENET_MEAN
        self.std = std if std is not None else _IMAGENET_STD
        self.interpolation = interpolation

        stages: list[TransformLike] = [
            RandomResizedCrop(
                crop_size, crop_size, scale=scale, ratio=ratio, interpolation=interpolation, p=1.0
            )
        ]
        if hflip_prob > 0.0:
            stages.append(HorizontalFlip(p=hflip_prob))
        if auto_augment is not None:
            stages.append(_parse_auto_augment(auto_augment))
        if color_jitter > 0.0:
            stages.append(
                ColorJitter(
                    brightness=color_jitter,
                    contrast=color_jitter,
                    saturation=color_jitter,
                    hue=0.0,
                    p=1.0,
                )
            )
        stages.append(Normalize(self.mean, self.std, max_pixel_value=1.0))
        if random_erasing > 0.0:
            stages.append(RandomErasing(p=random_erasing))
        self._pipeline = Compose(stages)

    @override
    def _init_kwargs(self) -> dict[str, object]:
        interp = self.interpolation
        return {
            "crop_size": self.crop_size,
            "scale": list(self.scale),
            "ratio": list(self.ratio),
            "color_jitter": self.color_jitter,
            "hflip_prob": self.hflip_prob,
            "auto_augment": self.auto_augment,
            "random_erasing": self.random_erasing,
            "mean": list(self.mean),
            "std": list(self.std),
            "interpolation": (interp.value if isinstance(interp, Interpolation) else interp),
        }


@_register_preset
class Detection(TransformsPreset):
    r"""Object-detection preset — coordinates ride with the image.

    Pipeline: ``LongestMaxSize(max_size)`` (``ResizeShortestEdge(min_size,
    max_size)`` when ``min_size`` is given) → ``PadIfNeeded`` onto a
    square of side :attr:`canvas_size`, the image placed per
    ``pad_position`` → ``Normalize(mean, std)`` plus a
    :class:`BboxParams(min_area, min_visibility)` policy on the
    enclosing :class:`Compose` so out-of-frame boxes (and their
    labels) drop automatically after the pipeline runs.

    Parameters
    ----------
    max_size : int, optional, default=1333
        Longest-side target after resize; canonical detection
        recipes use ``800 / 1333``.
    min_size : int or None, optional, default=None
        Shortest-side target.  When given, the longest side is still
        capped at ``max_size``.
    size_divisible : int, optional, default=1
        The square's side is ``max_size`` rounded up to a multiple of
        this.  The R-CNN weights pass 32, as the reference detection
        transform rounds its batches: an R-CNN derives each pyramid
        level's stride as canvas height over level height, and on a
        canvas that does not divide by the strides those quotients fall
        short (1333 gives 3, 7, 15, 31 and 63 instead of 4 to 64).  The
        default pads to exactly ``max_size``, as a config saved before
        this option existed did.
    pad_position : str, optional, default="center"
        Where the resized image sits on the canvas — one of the
        :class:`PadIfNeeded` positions.  The R-CNN and DETR references
        leave it in the top-left corner (``"top_left"``), the frame a
        detector's ``image_sizes`` is measured in; darknet letterboxes it
        centred.  The default is the placement a config saved before this
        option existed had.
    pad_value : float or None, optional, default=None
        The value the padded area reaches the model at, after
        normalisation.  The R-CNN and DETR references pad an already
        normalised batch with 0, and darknet letterboxes raw pixels with
        0.5 grey; each family's weights pass what its reference uses.  The
        preset fills with ``pad_value * std + mean`` before normalising, so
        the padding comes out at exactly ``pad_value``.  ``None`` pads the
        raw pixels with 0 — about -2 after ImageNet normalisation — as a
        config saved before this option existed did.
    square : bool, optional, default=True
        Pad onto a square of side :attr:`canvas_size`, the canvas every
        image shares, so a batch of them stacks.  ``False`` pads each axis
        only up to the next multiple of ``size_divisible`` — the canvas the
        reference detection transform gives a single image, with no
        padding beyond what the strides need; images of different shapes
        then no longer stack.  :attr:`canvas_size` describes the square.
    min_area : float, optional, default=1.0
        Drop post-pipeline boxes whose absolute pixel area is below
        this — see :class:`BboxParams`.
    min_visibility : float, optional, default=0.0
        Drop boxes whose visible fraction (area / original area)
        falls below this.
    mean, std : tuple of float, optional
        Per-channel normalization stats; default ImageNet.
    interpolation : str or Interpolation, optional, default="bilinear"

    Examples
    --------
    >>> import lucid, lucid.utils.transforms as T
    >>> tf = T.Detection()
    >>> tuple(tf(T.Image(lucid.rand(3, 32, 32))).data.shape)
    (3, 1333, 1333)

    A preset, not a single transform: it resizes so the longest side
    is at most ``max_size`` and pads to a square, which is what a
    detector's backbone expects to batch.  The R-CNN weights round the
    square up to a multiple of 32 and keep the image in its top-left
    corner, where :meth:`image_size` says it ends.

    >>> rcnn = T.Detection(size_divisible=32, pad_position="top_left")
    >>> rcnn.canvas_size, rcnn.image_size(300, 500)
    (1344, (800, 1333))

    With ``pad_value=0.0`` the padding reaches the model at 0, as the R-CNN
    references feed it, rather than at the normalised value of a black pixel:

    >>> tf = T.Detection(max_size=64, size_divisible=32, pad_value=0.0,
    ...                  pad_position="top_left")
    >>> padded = tf(T.Image(lucid.rand(3, 20, 30))).data
    >>> float(padded[:, -1, :].abs().max().item()) < 1e-6
    True
    """

    preset_type: ClassVar[str] = "Detection"

    def __init__(
        self,
        *,
        max_size: int = 1333,
        min_size: int | None = None,
        size_divisible: int = 1,
        pad_position: str = "center",
        pad_value: float | None = None,
        square: bool = True,
        min_area: float = 1.0,
        min_visibility: float = 0.0,
        mean: tuple[float, ...] | None = None,
        std: tuple[float, ...] | None = None,
        interpolation: str | Interpolation = Interpolation.BILINEAR,
    ) -> None:
        if size_divisible < 1:
            raise ValueError(f"Detection: size_divisible must be at least 1, got {size_divisible}")
        self.max_size = max_size
        self.min_size = min_size
        self.size_divisible = size_divisible
        self.pad_position = pad_position
        self.pad_value = pad_value
        self.square = square
        self.min_area = min_area
        self.min_visibility = min_visibility
        self.mean = mean if mean is not None else _IMAGENET_MEAN
        self.std = std if std is not None else _IMAGENET_STD
        self.interpolation = interpolation
        # The reference detection transform scales by
        # ``min(min_size / short_side, max_size / long_side)`` — it targets the
        # SHORTEST side and only falls back to the longest-side cap when that
        # would overflow.  Resizing the longest side alone presents objects at
        # a systematically different scale (~25% larger for a 4:3 image), which
        # shifts FPN level assignment and anchor matching for a checkpoint
        # trained the other way.  Composing the two caps reproduces the rule.
        self._resize: LongestMaxSize | ResizeShortestEdge
        if min_size is not None:
            self._resize = ResizeShortestEdge(min_size, max_size, interpolation=interpolation)
        else:
            self._resize = LongestMaxSize(max_size, interpolation=interpolation)
        side = self.canvas_size
        m_seq, s_seq = self.mean, self.std
        if pad_value is not None:
            if len(m_seq) != len(s_seq) and 1 not in (len(m_seq), len(s_seq)):
                raise ValueError(
                    f"Detection: mean and std hold one value per channel, got "
                    f"{len(m_seq)} and {len(s_seq)}"
                )
            n = max(len(m_seq), len(s_seq))
            m_seq = m_seq * n if len(m_seq) == 1 else m_seq
            s_seq = s_seq * n if len(s_seq) == 1 else s_seq
        # Filling with ``pad_value * std + mean`` normalises to ``pad_value``.
        fill: float | tuple[float, ...] = (
            0.0
            if pad_value is None
            else tuple(pad_value * s + m for m, s in zip(m_seq, s_seq, strict=True))
        )
        self._pad = (
            PadIfNeeded(side, side, value=fill, position=pad_position)
            if square
            else PadIfNeeded(
                None,
                None,
                value=fill,
                position=pad_position,
                pad_height_divisor=size_divisible,
                pad_width_divisor=size_divisible,
            )
        )
        stages: list[TransformLike] = [
            self._resize,
            self._pad,
            Normalize(self.mean, self.std, max_pixel_value=1.0),
        ]
        self._pipeline = Compose(
            stages, bbox_params=BboxParams(min_area=min_area, min_visibility=min_visibility)
        )

    @property
    def canvas_size(self) -> int:
        r"""Side of the square canvas the image is padded onto.

        ``max_size`` rounded up to a multiple of ``size_divisible``.

        Examples
        --------
        >>> import lucid.utils.transforms as T
        >>> T.Detection().canvas_size
        1333
        >>> T.Detection(size_divisible=32).canvas_size
        1344
        """
        step = self.size_divisible
        return -(-self.max_size // step) * step

    def image_size(self, height: int, width: int) -> tuple[int, int]:
        r"""``(H, W)`` an image of this size has on the canvas, before padding.

        With ``pad_position="top_left"`` the image fills rows ``[0, H)`` and
        columns ``[0, W)`` of the canvas, so this is that image's entry in a
        detector's ``postprocess(image_sizes=...)``.

        Parameters
        ----------
        height, width : int
            Size of the image the preset is applied to.

        Returns
        -------
        tuple of int
            The size the resize stage gives it — the same rule it applies
            to the image and its boxes.

        Examples
        --------
        >>> import lucid.utils.transforms as T
        >>> T.Detection(min_size=800, max_size=1333).image_size(480, 640)
        (800, 1067)
        """
        return self._resize._target(height, width)

    def to_image_boxes(self, boxes: Tensor, height: int, width: int) -> Tensor:
        r"""Map boxes on the canvas back onto the image they came from.

        A detector's ``postprocess`` answers in the coordinates of the canvas
        it was fed.  This undoes what the preset did to an image of
        ``height x width`` — the pad offset, then the resize — and clips to
        that image.  A centred letterbox (YOLO's) shifts every box by the
        margin above and left of the image; a top-left placement (the
        R-CNNs', DETR's) leaves only the resize to undo.

        Parameters
        ----------
        boxes : Tensor
            ``(N, 4)`` boxes as ``(x1, y1, x2, y2)`` on the canvas.
        height, width : int
            Size of the image the preset was applied to.

        Returns
        -------
        Tensor
            ``(N, 4)`` boxes in that image's pixels.

        Examples
        --------
        >>> import lucid, lucid.utils.transforms as T
        >>> letterbox = T.Detection(max_size=64)
        >>> letterbox.image_size(16, 32)
        (32, 64)
        >>> canvas_box = lucid.tensor([[0.0, 16.0, 64.0, 48.0]])
        >>> letterbox.to_image_boxes(canvas_box, 16, 32).tolist()
        [[0.0, 0.0, 32.0, 16.0]]
        """
        rh, rw = self.image_size(height, width)
        left, _, top, _ = self._pad._pads(rh, rw)
        dtype, device = boxes.dtype, boxes.device
        offset = lucid.tensor([left, top, left, top], dtype=dtype, device=device)
        scale = lucid.tensor([width / rw, height / rh] * 2, dtype=dtype, device=device)
        limit = lucid.tensor([width, height, width, height], dtype=dtype, device=device)
        mapped = (boxes - offset) * scale
        return lucid.minimum(lucid.maximum(mapped, lucid.zeros_like(mapped)), limit)

    @override
    def _init_kwargs(self) -> dict[str, object]:
        interp = self.interpolation
        return {
            "max_size": self.max_size,
            "min_size": self.min_size,
            "size_divisible": self.size_divisible,
            "pad_position": self.pad_position,
            "pad_value": self.pad_value,
            "square": self.square,
            "min_area": self.min_area,
            "min_visibility": self.min_visibility,
            "mean": list(self.mean),
            "std": list(self.std),
            "interpolation": (interp.value if isinstance(interp, Interpolation) else interp),
        }


@_register_preset
class Segmentation(TransformsPreset):
    r"""Semantic-segmentation preset — image + mask share geometry.

    Pipeline: shorter side to ``resize_size`` (the reference's rule: the
    longer side truncated) → ``CenterCrop(crop_size)``
    → ``Normalize`` applied to the image only, or — with ``stretch=True``
    — ``Resize(resize_size, resize_size)`` → ``Normalize``, which is what
    an upstream processor configured with an explicit ``(height, width)``
    pair does.  The image is low-pass filtered as it shrinks, either way.
    Mask travels through
    the geometric stages with nearest-neighbour interpolation (label
    preservation guaranteed by every
    :class:`~lucid.utils.transforms._base.GeometricTransform`'s
    ``_apply_mask`` hook) and is *not* normalised.

    Parameters
    ----------
    crop_size : int
        Square crop fed to the model.
    resize_size : int, optional, default=520
        Shorter-side length before cropping; canonical for FCN /
        DeepLab style recipes.
    mean, std : tuple of float, optional
        Per-channel normalization stats; default ImageNet.
    interpolation : str or Interpolation, optional, default="bilinear"
        Image resize interpolation.  Masks always use nearest.
    stretch : bool, optional, default=False
        Resize to exactly ``resize_size`` x ``resize_size`` and skip the
        crop, distorting the aspect ratio — for checkpoints whose upstream
        processor names an explicit ``(height, width)``.  Reproducing
        those with shortest-edge and centre-crop frames the image
        differently from how the model was evaluated.

    Examples
    --------
    >>> import lucid, lucid.utils.transforms as T
    >>> tf = T.Segmentation()
    >>> tuple(tf(T.Image(lucid.rand(3, 32, 32))).data.shape)
    (3, 520, 520)

    A preset. The mask travels with the image and is resampled with
    nearest-neighbour so a label is never interpolated into one that
    does not exist.
    """

    preset_type: ClassVar[str] = "Segmentation"

    def __init__(
        self,
        crop_size: int | None = None,
        *,
        resize_size: int = 520,
        mean: tuple[float, ...] | None = None,
        std: tuple[float, ...] | None = None,
        interpolation: str | Interpolation = Interpolation.BILINEAR,
        stretch: bool = False,
    ) -> None:
        self.crop_size = crop_size
        self.resize_size = resize_size
        self.mean = mean if mean is not None else _IMAGENET_MEAN
        self.std = std if std is not None else _IMAGENET_STD
        self.interpolation = interpolation
        self.stretch = stretch
        # ``crop_size=None`` reproduces the reference SemanticSegmentation
        # preset, which only resizes the shortest side: cropping an
        # already-resized image discards the sides of every non-square input
        # (a 500x375 VOC image becomes 693x520 and then loses 173 columns),
        # so the reported mIoU would never be reachable.
        stages: list[TransformLike]
        if stretch:
            # Some upstream processors are configured with an explicit
            # ``size={"height": H, "width": W}`` pair, which resizes to
            # exactly that shape -- aspect ratio distorted, nothing cropped.
            # Reproducing them with shortest-edge + centre-crop feeds the
            # model a differently-framed image than it was evaluated on.
            stages = [_ReferenceStretch(resize_size, resize_size, interpolation=interpolation)]
        else:
            stages = [_ReferenceShorterSide(resize_size, interpolation=interpolation)]
            if crop_size is not None:
                stages.append(CenterCrop(crop_size, crop_size))
        stages.append(Normalize(self.mean, self.std, max_pixel_value=1.0))
        self._pipeline = Compose(stages)

    @override
    def _init_kwargs(self) -> dict[str, object]:
        interp = self.interpolation
        return {
            "crop_size": self.crop_size,
            "resize_size": self.resize_size,
            "mean": list(self.mean),
            "std": list(self.std),
            "interpolation": (interp.value if isinstance(interp, Interpolation) else interp),
            "stretch": self.stretch,
        }


@_register_preset
class Pose(TransformsPreset):
    r"""Keypoint / pose-estimation preset.

    Pipeline: ``SmallestMaxSize(resize_size)`` → ``CenterCrop(crop_size)``
    → ``Normalize``.  Keypoint coordinates ride along through every
    geometric stage's ``_apply_keypoints`` hook; extra columns
    (visibility / score / angle) pass through unchanged.

    Parameters
    ----------
    crop_size : int
        Square crop fed to the model.
    resize_size : int, optional, default=256
        Shorter-side length before cropping.
    mean, std : tuple of float, optional
        Per-channel normalization stats; default ImageNet.
    interpolation : str or Interpolation, optional, default="bilinear"

    Examples
    --------
    >>> import lucid, lucid.utils.transforms as T
    >>> tf = T.Pose(crop_size=256)
    >>> tuple(tf(T.Image(lucid.rand(3, 300, 400))).data.shape)
    (3, 256, 256)

    Keypoints handed in beside the image are moved with it, which is the
    reason this is a preset rather than a resize and a crop.
    """

    preset_type: ClassVar[str] = "Pose"

    def __init__(
        self,
        crop_size: int,
        *,
        resize_size: int = 256,
        mean: tuple[float, ...] | None = None,
        std: tuple[float, ...] | None = None,
        interpolation: str | Interpolation = Interpolation.BILINEAR,
    ) -> None:
        self.crop_size = crop_size
        self.resize_size = resize_size
        self.mean = mean if mean is not None else _IMAGENET_MEAN
        self.std = std if std is not None else _IMAGENET_STD
        self.interpolation = interpolation
        self._pipeline = Compose(
            [
                SmallestMaxSize(resize_size, interpolation=interpolation),
                CenterCrop(crop_size, crop_size),
                Normalize(self.mean, self.std, max_pixel_value=1.0),
            ]
        )

    @override
    def _init_kwargs(self) -> dict[str, object]:
        interp = self.interpolation
        return {
            "crop_size": self.crop_size,
            "resize_size": self.resize_size,
            "mean": list(self.mean),
            "std": list(self.std),
            "interpolation": (interp.value if isinstance(interp, Interpolation) else interp),
        }
