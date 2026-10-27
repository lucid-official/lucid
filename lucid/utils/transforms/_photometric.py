"""Photometric transforms — per-pixel value adjustment.

Albumentations-compatible colour / intensity transforms.  All are
:class:`~lucid.utils.transforms._base.PhotometricTransform` — they act
only on the image and leave masks / boxes / keypoints untouched.

This module currently holds :class:`Normalize` and :class:`ColorJitter`;
the wider Albumentations colour set (brightness/contrast, gamma, HSV,
channel ops, CLAHE, …) lands in later batches.
"""

from dataclasses import dataclass
from typing import override

from lucid._tensor import Tensor
from lucid.utils.transforms import _random
from lucid.utils.transforms import functional as F
from lucid.utils.transforms._base import Empty, PhotometricTransform, _NoParams

# ── parameter types ─────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ColorJitterParams:
    r"""Sampled per-call jitter factors + the order to apply them in.

    Returned by :meth:`ColorJitter.make_params` once per call —
    captures every random choice the transform makes so the apply
    step is fully deterministic given the same instance.

    Attributes
    ----------
    order : tuple of int
        Random permutation of ``(0, 1, 2, 3)`` indexing the
        ``(brightness, contrast, saturation, hue)`` adjustment
        sequence — matches the reference_vision reference.
    brightness, contrast, saturation, hue : float or None
        Sampled factors per adjustment, or ``None`` when the
        corresponding range was disabled at construction time.
    """

    order: tuple[int, ...]
    brightness: float | None
    contrast: float | None
    saturation: float | None
    hue: float | None


# ── deterministic ───────────────────────────────────────────────────


class ToFloat(_NoParams, PhotometricTransform[Empty]):
    r"""Scale an integer-range image to ``[0, 1]`` (Albumentations ``ToFloat``).

    Parameters
    ----------
    max_value : float, optional, default=255.0
        Value the image is divided by.
    p : float, optional, default=1.0

    Examples
    --------
    >>> import lucid, lucid.utils.transforms as T
    >>> tf = T.ToFloat(p=1.0)
    >>> tuple(tf(T.Image(lucid.rand(3, 32, 32))).data.shape)
    (3, 32, 32)

    Divides by ``max_value``, so an 8-bit image lands in 0-1. The
    input here is already float, which is why the values do not move.
    """

    def __init__(self, max_value: float = 255.0, p: float = 1.0) -> None:
        super().__init__(p=p)
        self.max_value = max_value

    @override
    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:
        return img / self.max_value

    @override
    def __repr__(self) -> str:
        return f"ToFloat(max_value={self.max_value}, p={self.p})"


class FromFloat(_NoParams, PhotometricTransform[Empty]):
    r"""Scale a ``[0, 1]`` image back up by ``max_value`` (Albumentations ``FromFloat``).

    Parameters
    ----------
    max_value : float, optional, default=255.0
    p : float, optional, default=1.0

    Notes
    -----
    Stays in floating point (Lucid keeps a float tensor); the Albu
    ``dtype`` argument is accepted for signature parity but the result
    is not re-quantized.

    Examples
    --------
    >>> import lucid, lucid.utils.transforms as T
    >>> tf = T.FromFloat(p=1.0)
    >>> tuple(tf(T.Image(lucid.rand(3, 32, 32))).data.shape)
    (3, 32, 32)

    The inverse of :class:`ToFloat`, and the pair is what moves an
    image between the 0-1 range a network wants and the 0-255 one a
    file holds.
    """

    def __init__(self, max_value: float = 255.0, dtype: str = "uint8", p: float = 1.0) -> None:
        super().__init__(p=p)
        self.max_value = max_value
        self.dtype = dtype

    @override
    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:
        return img * self.max_value

    @override
    def __repr__(self) -> str:
        return f"FromFloat(max_value={self.max_value}, p={self.p})"


class Normalize(_NoParams, PhotometricTransform[Empty]):
    r"""Normalize an image (Albumentations ``Normalize``).

    Computes ``(img - mean * max_pixel_value) / (std * max_pixel_value)``
    — i.e. scales by ``max_pixel_value`` then standardizes per channel.

    Parameters
    ----------
    mean, std : tuple of float
        Per-channel statistics.
    max_pixel_value : float, optional, default=255.0
        Value the inputs are divided by (``1.0`` for ``[0, 1]`` inputs).
    p : float, optional, default=1.0

    Examples
    --------
    >>> import lucid, lucid.utils.transforms as T
    >>> tf = T.Normalize(mean=(0.485, 0.456, 0.406),
    ...                  std=(0.229, 0.224, 0.225), max_pixel_value=1.0)
    >>> tuple(tf(T.Image(lucid.rand(3, 32, 32))).data.shape)
    (3, 32, 32)

    ``max_pixel_value`` divides before the mean is taken, so it is 255
    for an image read from a file and 1.0 for one already in 0-1.
    """

    def __init__(
        self,
        mean: tuple[float, ...],
        std: tuple[float, ...],
        max_pixel_value: float = 255.0,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.mean = tuple(mean)
        self.std = tuple(std)
        self.max_pixel_value = max_pixel_value

    @override
    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:
        mv = self.max_pixel_value
        eff_mean = tuple(m * mv for m in self.mean)
        eff_std = tuple(s * mv for s in self.std)
        return F.normalize(img, eff_mean, eff_std)

    @override
    def __repr__(self) -> str:
        return (
            f"Normalize(mean={self.mean}, std={self.std}, "
            f"max_pixel_value={self.max_pixel_value}, p={self.p})"
        )


# ── randomized ──────────────────────────────────────────────────────


def _jitter_range(
    value: float | tuple[float, float],
    *,
    center: float,
    floor: float | None = None,
) -> tuple[float, float] | None:
    """Normalize a ColorJitter arg to a ``(min, max)`` range, or ``None``."""
    if isinstance(value, (int, float)):
        if value == 0:
            return None
        lo, hi = center - value, center + value
    else:
        lo, hi = value[0], value[1]
    if floor is not None:
        lo = max(lo, floor)
    return (lo, hi)


class ColorJitter(PhotometricTransform[ColorJitterParams]):
    r"""Randomly jitter brightness/contrast/saturation/hue.

    Albumentations ``ColorJitter`` — each factor is sampled uniformly
    from its range and the four adjustments are applied in random order.

    Parameters
    ----------
    brightness, contrast, saturation : float or (float, float), optional, default=0.2
        Scalar ``v`` → range ``[max(0, 1-v), 1+v]``; tuple used directly.
    hue : float or (float, float), optional, default=0.2
        Scalar ``v`` → ``[-v, v]`` (``v`` ≤ 0.5).
    p : float, optional, default=0.5

    Examples
    --------
    >>> import lucid, lucid.utils.transforms as T
    >>> tf = T.ColorJitter(p=1.0)
    >>> tuple(tf(T.Image(lucid.rand(3, 32, 32))).data.shape)
    (3, 32, 32)
    """

    _ADJUST = (
        F.adjust_brightness,
        F.adjust_contrast,
        F.adjust_saturation,
        F.adjust_hue,
    )

    def __init__(
        self,
        brightness: float | tuple[float, float] = 0.2,
        contrast: float | tuple[float, float] = 0.2,
        saturation: float | tuple[float, float] = 0.2,
        hue: float | tuple[float, float] = 0.2,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self._ranges = (
            _jitter_range(brightness, center=1.0, floor=0.0),
            _jitter_range(contrast, center=1.0, floor=0.0),
            _jitter_range(saturation, center=1.0, floor=0.0),
            _jitter_range(hue, center=0.0),
        )

    @override
    def make_params(self, img: Tensor) -> ColorJitterParams:
        factors = [None if rng is None else _random.uniform(rng[0], rng[1]) for rng in self._ranges]
        order = list(range(4))
        for i in range(len(order) - 1, 0, -1):  # Fisher-Yates via Lucid RNG
            j = _random.randint(0, i + 1)
            order[i], order[j] = order[j], order[i]
        return ColorJitterParams(
            order=tuple(order),
            brightness=factors[0],
            contrast=factors[1],
            saturation=factors[2],
            hue=factors[3],
        )

    @override
    def _apply_image(self, img: Tensor, params: ColorJitterParams) -> Tensor:
        factors = (params.brightness, params.contrast, params.saturation, params.hue)
        for idx in params.order:
            factor = factors[idx]
            if factor is not None:
                img = self._ADJUST[idx](img, factor)
        return img

    @override
    def __repr__(self) -> str:
        b, c, s, h = self._ranges
        return f"ColorJitter(brightness={b}, contrast={c}, saturation={s}, " f"hue={h}, p={self.p})"
