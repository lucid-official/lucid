"""Colour / pixel-level transforms (Albumentations-compatible).

All :class:`~lucid.utils.transforms._base.PhotometricTransform` — they
act only on the image (RGB ``[0, 1]``) and leave masks / boxes /
keypoints untouched.
"""

from dataclasses import dataclass
from typing import override

import lucid
from lucid._tensor import Tensor
from lucid.utils.transforms import _random
from lucid.utils.transforms import functional as F
from lucid.utils.transforms._base import Empty, PhotometricTransform, _NoParams


def _rng(value: float | tuple[float, float]) -> tuple[float, float]:
    return (-value, value) if isinstance(value, (int, float)) else value


# ── parameter types ─────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class BCParams:
    r"""Per-call brightness offset and contrast delta.

    Carried by :class:`RandomBrightnessContrast`; the apply step fuses
    them into ``img * (1 + contrast) + brightness`` before clipping to
    ``[0, 1]``.

    Attributes
    ----------
    brightness : float
        Additive brightness offset on the unit ``[0, 1]`` scale.
    contrast : float
        Contrast delta; effective multiplier is ``1 + contrast``.
    """

    brightness: float
    contrast: float


@dataclass(frozen=True, slots=True)
class ScalarParams:
    r"""Per-call single-scalar parameter for tone / curve transforms.

    Reused across :class:`RandomGamma` (gamma exponent),
    :class:`Solarize` (inversion threshold on the unit scale),
    :class:`RandomBrightness` (brightness offset),
    :class:`RandomContrast` (contrast delta), and
    :class:`RandomToneCurve` (S-curve amount).  The host transform's
    apply step decides how to interpret ``value``.

    Attributes
    ----------
    value : float
        Sampled scalar; meaning depends on the host transform.
    """

    value: float


@dataclass(frozen=True, slots=True)
class TripletParams:
    r"""Per-call three-scalar parameter pack for multi-knob transforms.

    Used by :class:`HueSaturationValue` (hue / sat / val shifts),
    :class:`RGBShift` (per-channel offsets, pre-divided by 255),
    :class:`Sharpen` (alpha / lightness / unused),
    :class:`Emboss` (alpha / strength / unused), and
    :class:`UnsharpMask` (alpha / ksize-as-float / sigma).

    Attributes
    ----------
    a : float
        First sampled value; see host transform for semantics.
    b : float
        Second sampled value.
    c : float
        Third sampled value; often ``0.0`` when only two knobs are
        needed.
    """

    a: float
    b: float
    c: float


@dataclass(frozen=True, slots=True)
class PermParams:
    r"""Per-call RGB-channel permutation used by :class:`ChannelShuffle`.

    The apply step concatenates the input channels in ``order`` to
    produce the output, so the transform is a pure slice + concat
    (no pixel arithmetic, autograd-friendly).

    Attributes
    ----------
    order : tuple of (int, int, int)
        Permutation of ``(0, 1, 2)`` drawn via Fisher-Yates.
    """

    order: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class ChannelDropParams:
    r"""Per-call set of channel indices to replace with the fill value.

    Carried by :class:`ChannelDropout`; the apply step builds a
    multiplicative keep-mask + additive fill so the operation stays
    differentiable.

    Attributes
    ----------
    channels : tuple of int
        Distinct channel indices to drop (sampled without replacement).
    """

    channels: tuple[int, ...]


# ── brightness / contrast / gamma ───────────────────────────────────


class RandomBrightnessContrast(PhotometricTransform[BCParams]):
    r"""Randomly perturb brightness and contrast jointly (Albumentations ``RandomBrightnessContrast``).

    Samples a brightness offset and a contrast multiplier independently
    from their limit ranges, then applies ``img * (1 + c) + b`` in one
    pass and clips to ``[0, 1]``.  Equivalent to chaining
    :class:`RandomBrightness` and :class:`RandomContrast` but cheaper
    (one fused expression, one clip).

    Parameters
    ----------
    brightness_limit : float or (float, float), optional, default=0.2
        Additive brightness offset range.  A scalar ``v`` is interpreted
        as the symmetric range ``(-v, v)``.
    contrast_limit : float or (float, float), optional, default=0.2
        Multiplicative contrast offset range (multiplier is
        ``1 + sampled_value``).  Scalar ``v`` expands to ``(-v, v)``.
    p : float, optional, default=0.5
        Probability of applying the transform.

    Examples
    --------
    >>> import lucid
    >>> import lucid.utils.transforms as T
    >>> tf = T.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0)
    >>> out = tf(T.Image(lucid.rand(3, 32, 32))).data
    >>> tuple(out.shape)
    (3, 32, 32)
    """

    def __init__(
        self,
        brightness_limit: float | tuple[float, float] = 0.2,
        contrast_limit: float | tuple[float, float] = 0.2,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.brightness_limit = _rng(brightness_limit)
        self.contrast_limit = _rng(contrast_limit)

    @override
    def make_params(self, img: Tensor) -> BCParams:
        r"""Sample per-call random parameters for :class:`RandomBrightnessContrast`.

        Parameters
        ----------
        img : Tensor
            Image tensor; not inspected, carried through for dispatch.

        Returns
        -------
        BCParams
            Carries ``brightness`` (additive offset) and ``contrast``
            (multiplier delta), each drawn uniformly from their
            respective limit ranges.
        """
        return BCParams(
            brightness=_random.uniform(*self.brightness_limit),
            contrast=_random.uniform(*self.contrast_limit),
        )

    @override
    def _apply_image(self, img: Tensor, params: BCParams) -> Tensor:
        out = img * (1.0 + params.contrast) + params.brightness
        return lucid.clip(out, 0.0, 1.0)

    @override
    def __repr__(self) -> str:
        return (
            f"RandomBrightnessContrast(brightness_limit={self.brightness_limit}, "
            f"contrast_limit={self.contrast_limit}, p={self.p})"
        )


class RandomGamma(PhotometricTransform[ScalarParams]):
    r"""Random gamma correction (Albumentations ``RandomGamma``).

    Samples a gamma value uniformly from ``gamma_limit / 100`` and
    applies ``img ** gamma`` after clipping the input to ``[0, 1]``.
    Gamma ``< 1`` brightens midtones; gamma ``> 1`` darkens them.

    Parameters
    ----------
    gamma_limit : (int, int), optional, default=(80, 120)
        Inclusive range of the gamma exponent expressed in percent
        (matches Albumentations' integer-percent convention).  Sampled
        gamma is ``randint(gamma_limit[0], gamma_limit[1]) / 100``.
    p : float, optional, default=0.5
        Probability of applying the transform.

    Examples
    --------
    >>> import lucid
    >>> import lucid.utils.transforms as T
    >>> tf = T.RandomGamma(gamma_limit=(80, 120), p=1.0)
    >>> out = tf(T.Image(lucid.rand(3, 32, 32))).data
    >>> tuple(out.shape)
    (3, 32, 32)
    """

    def __init__(self, gamma_limit: tuple[int, int] = (80, 120), p: float = 0.5) -> None:
        super().__init__(p=p)
        self.gamma_limit = gamma_limit

    @override
    def make_params(self, img: Tensor) -> ScalarParams:
        r"""Sample per-call random parameters for :class:`RandomGamma`.

        Parameters
        ----------
        img : Tensor
            Image tensor; not inspected, carried through for dispatch.

        Returns
        -------
        ScalarParams
            Carries ``value`` — the gamma exponent, drawn uniformly
            from ``gamma_limit`` and divided by 100 to convert the
            integer-percent input to a real exponent.
        """
        return ScalarParams(value=_random.uniform(self.gamma_limit[0], self.gamma_limit[1]) / 100.0)

    @override
    def _apply_image(self, img: Tensor, params: ScalarParams) -> Tensor:
        return lucid.clip(img, 0.0, 1.0) ** params.value

    @override
    def __repr__(self) -> str:
        return f"RandomGamma(gamma_limit={self.gamma_limit}, p={self.p})"


# ── HSV / RGB shifts ────────────────────────────────────────────────


class HueSaturationValue(PhotometricTransform[TripletParams]):
    r"""Random hue / saturation / value shift (Albumentations ``HueSaturationValue``).

    Samples each shift uniformly from its limit range, then applies a
    single :func:`functional.adjust_hsv` round-trip — additive in HSV,
    hue wraps, saturation and value clip to ``[0, 1]``.  Channel-gated
    to RGB (:meth:`PhotometricTransform._require_channels` rejects
    non-3-channel input).

    Parameters
    ----------
    hue_shift_limit : float or (float, float), optional, default=20
        Hue offset range on the OpenCV scale ``[0, 179]``.  Scalar
        ``v`` is interpreted as ``(-v, v)``.
    sat_shift_limit : float or (float, float), optional, default=30
        Saturation offset range on the OpenCV scale ``[0, 255]``.
    val_shift_limit : float or (float, float), optional, default=20
        Value (brightness) offset range on the OpenCV scale ``[0, 255]``.
    p : float, optional, default=0.5
        Probability of applying the transform.

    Raises
    ------
    ValueError
        If the image has anything other than 3 channels.

    Notes
    -----
    Lucid stays in float for the full HSV round-trip; Albumentations
    quantises through ``uint8`` + cv2 HSV, so the two match to
    ``~0.02`` (G3 parity "ballpark" tier).  Lucid's path is the more
    precise of the two.

    Examples
    --------
    >>> import lucid
    >>> import lucid.utils.transforms as T
    >>> tf = T.HueSaturationValue(hue_shift_limit=10, p=1.0)
    >>> out = tf(T.Image(lucid.rand(3, 32, 32))).data
    >>> tuple(out.shape)
    (3, 32, 32)
    """

    def __init__(
        self,
        hue_shift_limit: float | tuple[float, float] = 20,
        sat_shift_limit: float | tuple[float, float] = 30,
        val_shift_limit: float | tuple[float, float] = 20,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.hue_shift_limit = _rng(hue_shift_limit)
        self.sat_shift_limit = _rng(sat_shift_limit)
        self.val_shift_limit = _rng(val_shift_limit)

    @override
    def make_params(self, img: Tensor) -> TripletParams:
        r"""Sample per-call random parameters for :class:`HueSaturationValue`.

        Parameters
        ----------
        img : Tensor
            Image tensor; not inspected, carried through for dispatch.

        Returns
        -------
        TripletParams
            Carries the hue shift (``a``, OpenCV ``[0, 179]`` scale),
            saturation shift (``b``, ``[0, 255]``), and value shift
            (``c``, ``[0, 255]``), each drawn uniformly from its
            limit range.
        """
        return TripletParams(
            a=_random.uniform(*self.hue_shift_limit),
            b=_random.uniform(*self.sat_shift_limit),
            c=_random.uniform(*self.val_shift_limit),
        )

    @override
    def _apply_image(self, img: Tensor, params: TripletParams) -> Tensor:
        self._require_channels(img, 3)
        return F.adjust_hsv(img, params.a, params.b, params.c)

    @override
    def __repr__(self) -> str:
        return (
            f"HueSaturationValue(hue_shift_limit={self.hue_shift_limit}, "
            f"sat_shift_limit={self.sat_shift_limit}, "
            f"val_shift_limit={self.val_shift_limit}, p={self.p})"
        )


class RGBShift(PhotometricTransform[TripletParams]):
    r"""Add per-channel additive shifts to an RGB image (Albumentations ``RGBShift``).

    Samples an independent offset for each of the R / G / B channels
    from the corresponding limit, scales by ``1/255`` so the limits stay
    on the familiar OpenCV 0-255 scale, then adds and clips to
    ``[0, 1]``.  Channel-gated to 3-channel input via
    :meth:`PhotometricTransform._require_channels`.

    Parameters
    ----------
    r_shift_limit : float or (float, float), optional, default=20
        Red-channel shift range on the 0-255 scale; scalar ``v`` expands
        to ``(-v, v)``.
    g_shift_limit : float or (float, float), optional, default=20
        Green-channel shift range on the 0-255 scale.
    b_shift_limit : float or (float, float), optional, default=20
        Blue-channel shift range on the 0-255 scale.
    p : float, optional, default=0.5
        Probability of applying the transform.

    Raises
    ------
    ValueError
        If the input image does not have exactly 3 channels.

    Examples
    --------
    >>> import lucid
    >>> import lucid.utils.transforms as T
    >>> tf = T.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1.0)
    >>> out = tf(T.Image(lucid.rand(3, 32, 32))).data
    >>> tuple(out.shape)
    (3, 32, 32)
    """

    def __init__(
        self,
        r_shift_limit: float | tuple[float, float] = 20,
        g_shift_limit: float | tuple[float, float] = 20,
        b_shift_limit: float | tuple[float, float] = 20,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.r = _rng(r_shift_limit)
        self.g = _rng(g_shift_limit)
        self.b = _rng(b_shift_limit)

    @override
    def make_params(self, img: Tensor) -> TripletParams:
        r"""Sample per-call random parameters for :class:`RGBShift`.

        Parameters
        ----------
        img : Tensor
            Image tensor; not inspected, carried through for dispatch.

        Returns
        -------
        TripletParams
            Carries the per-channel offsets ``a`` (red), ``b`` (green),
            ``c`` (blue), each sampled from the constructor's 0-255
            limit range and divided by 255 to land on the unit scale.
        """
        return TripletParams(
            a=_random.uniform(*self.r) / 255.0,
            b=_random.uniform(*self.g) / 255.0,
            c=_random.uniform(*self.b) / 255.0,
        )

    @override
    def _apply_image(self, img: Tensor, params: TripletParams) -> Tensor:
        self._require_channels(img, 3)
        shift = lucid.tensor([params.a, params.b, params.c], dtype=img.dtype, device=img.device)
        shift = shift.reshape(1, 3, 1, 1) if img.ndim == 4 else shift.reshape(3, 1, 1)
        return lucid.clip(img + shift, 0.0, 1.0)

    @override
    def __repr__(self) -> str:
        return f"RGBShift(r={self.r}, g={self.g}, b={self.b}, p={self.p})"


# ── channel ops ─────────────────────────────────────────────────────


class ChannelShuffle(PhotometricTransform[PermParams]):
    r"""Randomly permute the RGB channel order (Albumentations ``ChannelShuffle``).

    Samples a uniformly random permutation of ``(0, 1, 2)`` via a
    Fisher-Yates shuffle, then reorders the channel axis accordingly.
    No pixel arithmetic is performed — the operation is a pure slice +
    concat, so it composes cleanly with autograd.

    Parameters
    ----------
    p : float, optional, default=0.5
        Probability of applying the transform.

    Raises
    ------
    ValueError
        If the input image does not have exactly 3 channels.

    Examples
    --------
    >>> import lucid
    >>> import lucid.utils.transforms as T
    >>> tf = T.ChannelShuffle(p=1.0)
    >>> out = tf(T.Image(lucid.rand(3, 32, 32))).data
    >>> tuple(out.shape)
    (3, 32, 32)
    """

    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p=p)

    @override
    def make_params(self, img: Tensor) -> PermParams:
        r"""Sample per-call random parameters for :class:`ChannelShuffle`.

        Parameters
        ----------
        img : Tensor
            Image tensor; not inspected, carried through for dispatch.

        Returns
        -------
        PermParams
            Carries ``order`` — a uniformly-random permutation of
            ``(0, 1, 2)`` produced by a Fisher-Yates shuffle.
        """
        order = [0, 1, 2]
        for i in range(2, 0, -1):
            j = _random.randint(0, i + 1)
            order[i], order[j] = order[j], order[i]
        return PermParams(order=(order[0], order[1], order[2]))

    @override
    def _apply_image(self, img: Tensor, params: PermParams) -> Tensor:
        self._require_channels(img, 3)
        ax = -3
        ch = [img[..., i : i + 1, :, :] for i in params.order]
        return F._cat(ch, ax)

    @override
    def __repr__(self) -> str:
        return f"ChannelShuffle(p={self.p})"


class ChannelDropout(PhotometricTransform[ChannelDropParams]):
    r"""Replace random channels with a constant fill value (Albumentations ``ChannelDropout``).

    Samples a count ``n`` uniformly from ``channel_drop_range`` and
    selects ``n`` distinct channels without replacement.  Each chosen
    channel is replaced by ``fill_value`` via a multiplicative keep-mask
    + additive fill, so the op stays autograd-friendly.

    Parameters
    ----------
    channel_drop_range : (int, int), optional, default=(1, 1)
        Inclusive range from which the number of channels to drop is
        sampled (``randint(lo, hi + 1)``).
    fill_value : float, optional, default=0.0
        Value written into the dropped channels.
    p : float, optional, default=0.5
        Probability of applying the transform.

    Examples
    --------
    >>> import lucid
    >>> import lucid.utils.transforms as T
    >>> tf = T.ChannelDropout(channel_drop_range=(1, 1), fill_value=0.0, p=1.0)
    >>> out = tf(T.Image(lucid.rand(3, 32, 32))).data
    >>> tuple(out.shape)
    (3, 32, 32)
    """

    def __init__(
        self,
        channel_drop_range: tuple[int, int] = (1, 1),
        fill_value: float = 0.0,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.channel_drop_range = channel_drop_range
        self.fill_value = fill_value

    @override
    def make_params(self, img: Tensor) -> ChannelDropParams:
        r"""Sample per-call random parameters for :class:`ChannelDropout`.

        Parameters
        ----------
        img : Tensor
            Image tensor; its channel-axis size is read so that the
            sample-without-replacement loop never returns duplicate or
            out-of-range indices.

        Returns
        -------
        ChannelDropParams
            Carries ``channels`` — a tuple of distinct channel indices
            (sampled without replacement) to replace with
            ``fill_value`` in the apply step.

        Notes
        -----
        The number of dropped channels is drawn from
        ``channel_drop_range``; it is clamped to the available channel
        count if the high bound exceeds ``C``.
        """
        c = int(img.shape[-3])
        n = _random.randint(self.channel_drop_range[0], self.channel_drop_range[1] + 1)
        idxs = list(range(c))
        chosen: list[int] = []
        for _ in range(min(n, c)):
            k = _random.randint(0, len(idxs))
            chosen.append(idxs.pop(k))
        return ChannelDropParams(channels=tuple(chosen))

    @override
    def _apply_image(self, img: Tensor, params: ChannelDropParams) -> Tensor:
        c = int(img.shape[-3])
        keep = [0.0 if i in params.channels else 1.0 for i in range(c)]
        mask = lucid.tensor(keep, dtype=img.dtype, device=img.device)
        mask = mask.reshape(1, c, 1, 1) if img.ndim == 4 else mask.reshape(c, 1, 1)
        return img * mask + self.fill_value * (1.0 - mask)

    @override
    def __repr__(self) -> str:
        return f"ChannelDropout(channel_drop_range={self.channel_drop_range}, p={self.p})"


# ── histogram-based ─────────────────────────────────────────────────


class Equalize(_NoParams, PhotometricTransform[Empty]):
    r"""Per-channel histogram equalization (Albumentations ``Equalize``).

    Computes a cumulative distribution per channel and remaps intensities
    so the output histogram is as flat as possible.  Useful for stretching
    contrast on dim or hazy images.  Delegates to
    :func:`functional.equalize`, which handles the uint8 round-trip and
    per-channel CDF construction.

    Parameters
    ----------
    p : float, optional, default=0.5
        Probability of applying the transform.

    Examples
    --------
    >>> import lucid
    >>> import lucid.utils.transforms as T
    >>> tf = T.Equalize(p=1.0)
    >>> out = tf(T.Image(lucid.rand(3, 32, 32))).data
    >>> tuple(out.shape)
    (3, 32, 32)
    """

    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p=p)

    @override
    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:
        return F.equalize(img)

    @override
    def __repr__(self) -> str:
        return f"Equalize(p={self.p})"


class CLAHE(_NoParams, PhotometricTransform[Empty]):
    r"""Contrast-limited adaptive histogram equalization (Albumentations ``CLAHE``).

    Thin wrapper around :func:`functional.clahe`: per-tile
    clipped-histogram LUT + 4-neighbour bilinear interpolation between
    tile centres.  Single-channel images are equalized directly; RGB
    is routed through the HSV value channel so hue and saturation are
    preserved (mirrors Albumentations' "luminance-only" behaviour).

    Parameters
    ----------
    clip_limit : float, optional, default=4.0
        Contrast-clip threshold; per-tile histogram cap is
        ``clip_limit * tile_area / 256``.  Higher values allow more
        contrast amplification at the cost of noise.
    tile_grid_size : (int, int), optional, default=(8, 8)
        Number of tiles ``(rows, cols)`` the image is divided into.
        Larger grids give more local adaptivity at a per-tile cost.
    p : float, optional, default=0.5
        Probability of applying the transform.

    Notes
    -----
    Approximate cv2 / Albumentations parity (colour-space pivot
    differs: HSV value vs LAB L).  Currently ~30× slower than Albu
    on a 224² RGB input — see :func:`functional.clahe` notes for the
    bottleneck description.

    Examples
    --------
    >>> import lucid
    >>> import lucid.utils.transforms as T
    >>> tf = T.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0)
    >>> out = tf(T.Image(lucid.rand(3, 64, 64))).data
    >>> tuple(out.shape)
    (3, 64, 64)
    """

    def __init__(
        self,
        clip_limit: float = 4.0,
        tile_grid_size: tuple[int, int] = (8, 8),
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    @override
    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:
        return F.clahe(img, self.clip_limit, self.tile_grid_size)

    @override
    def __repr__(self) -> str:
        return (
            f"CLAHE(clip_limit={self.clip_limit}, "
            f"tile_grid_size={self.tile_grid_size}, p={self.p})"
        )


# ── tone / inversion ────────────────────────────────────────────────


class Solarize(PhotometricTransform[ScalarParams]):
    r"""Invert pixel intensities above a threshold (Albumentations ``Solarize``).

    Samples a threshold uniformly from ``threshold`` (on the 0-255
    scale, divided by 255 internally), then replaces every pixel
    ``x >= t`` with ``1 - x``, leaving pixels below the threshold
    untouched.  Classic photographic posterisation effect, also used by
    AutoAugment / RandAugment.

    Parameters
    ----------
    threshold : float or (float, float), optional, default=128
        Threshold (0-255 scale) for the inversion cutoff.  Scalar ``v``
        is interpreted as the constant range ``(v, v)``; a tuple samples
        the threshold uniformly per call.
    p : float, optional, default=0.5
        Probability of applying the transform.

    Examples
    --------
    >>> import lucid
    >>> import lucid.utils.transforms as T
    >>> tf = T.Solarize(threshold=128, p=1.0)
    >>> out = tf(T.Image(lucid.rand(3, 32, 32))).data
    >>> tuple(out.shape)
    (3, 32, 32)
    """

    def __init__(self, threshold: float | tuple[float, float] = 128, p: float = 0.5) -> None:
        super().__init__(p=p)
        self.threshold = (
            (threshold, threshold) if isinstance(threshold, (int, float)) else threshold
        )

    @override
    def make_params(self, img: Tensor) -> ScalarParams:
        r"""Sample per-call random parameters for :class:`Solarize`.

        Parameters
        ----------
        img : Tensor
            Image tensor; not inspected, carried through for dispatch.

        Returns
        -------
        ScalarParams
            Carries ``value`` — the inversion threshold on the unit
            scale (sampled from ``threshold`` on the 0-255 scale, then
            divided by 255).
        """
        return ScalarParams(value=_random.uniform(*self.threshold) / 255.0)

    @override
    def _apply_image(self, img: Tensor, params: ScalarParams) -> Tensor:
        return lucid.where(img >= params.value, 1.0 - img, img)

    @override
    def __repr__(self) -> str:
        return f"Solarize(threshold={self.threshold}, p={self.p})"


class Posterize(PhotometricTransform[Empty]):
    r"""Reduce bits per channel (Albumentations ``Posterize``).

    Two quantisation modes — pick based on whether you need
    bit-exact Albumentations / cv2 parity or just a cheap float-only
    posterise.

    Parameters
    ----------
    num_bits : int, optional, default=4
        Number of bits to keep per channel; output has ``2**num_bits``
        distinct levels.  Must be in ``[1, 7]``.
    mode : {"uint8_mask", "float"}, optional, default="uint8_mask"
        Quantisation algorithm:

        * ``"uint8_mask"`` — matches Albumentations / OpenCV
          bit-exactly.  Round-trips through ``uint8`` and applies the
          bit mask ``~((1 << (8 - num_bits)) - 1)``, then divides by
          ``255``.  Use this for parity with reference pipelines.
        * ``"float"`` — pure-float quantisation via
          ``floor(x * 2**num_bits) / 2**num_bits``.  Slightly
          different mid-bin placement than the OpenCV bit-mask;
          cheaper (no uint8 round-trip).  Use for novel pipelines
          where Albu parity isn't required.
    p : float, optional, default=0.5
        Probability of applying the transform.

    Raises
    ------
    ValueError
        If ``num_bits`` falls outside ``[1, 7]`` or ``mode`` is not one
        of the two recognised names.

    Notes
    -----
    The G3 parity suite verifies the ``"uint8_mask"`` mode matches
    Albumentations to float32 epsilon across ``num_bits ∈ {1, …, 7}``.
    A 200-seed statistical aggregate (G4e) confirms the mean-of-max
    diff stays in the same regime.

    Examples
    --------
    >>> import lucid
    >>> import lucid.utils.transforms as T
    >>> tf = T.Posterize(num_bits=3, mode="uint8_mask", p=1.0)
    >>> out = tf(T.Image(lucid.rand(3, 32, 32))).data
    >>> # 3-bit → 8 distinct quantisation levels per channel
    >>> len(set(out.numpy().reshape(-1).tolist())) <= 8
    True
    """

    def __init__(
        self,
        num_bits: int = 4,
        mode: str = "uint8_mask",
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        if not 1 <= num_bits <= 7:
            raise ValueError(f"num_bits must be in [1, 7], got {num_bits}")
        if mode not in ("uint8_mask", "float"):
            raise ValueError(f"mode must be 'uint8_mask' or 'float', got {mode!r}")
        self.num_bits = num_bits
        self.mode = mode

    @override
    def make_params(self, img: Tensor) -> Empty:
        r"""Return the no-op parameter sentinel for :class:`Posterize`.

        Parameters
        ----------
        img : Tensor
            Image tensor; not inspected.

        Returns
        -------
        Empty
            The shared :data:`NO_PARAMS` sentinel — Posterize is
            deterministic given ``num_bits`` / ``mode``, so no
            per-call sampling is required.
        """
        from lucid.utils.transforms._base import NO_PARAMS

        return NO_PARAMS

    @override
    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:
        if self.mode == "float":
            levels = float(2**self.num_bits)
            return lucid.floor(lucid.clip(img, 0.0, 1.0) * levels) / levels
        # uint8_mask — round to uint8, apply bit mask, back to float
        mask_int = (~((1 << (8 - self.num_bits)) - 1)) & 0xFF
        u8 = lucid.clip(lucid.round(img * 255.0), 0.0, 255.0).long()
        masked = u8 & mask_int
        return masked.to(img.dtype) / 255.0

    @override
    def __repr__(self) -> str:
        return f"Posterize(num_bits={self.num_bits}, mode={self.mode!r}, p={self.p})"


class InvertImg(_NoParams, PhotometricTransform[Empty]):
    r"""Invert pixel intensities via ``1 - img`` (Albumentations ``InvertImg``).

    Photographic negative — every channel of every pixel is replaced by
    ``1 - x``.  No parameters and no sampling; deterministic when
    triggered.  Often used as a building block for AutoAugment /
    RandAugment policies.

    Parameters
    ----------
    p : float, optional, default=0.5
        Probability of applying the transform.

    Examples
    --------
    >>> import lucid
    >>> import lucid.utils.transforms as T
    >>> tf = T.InvertImg(p=1.0)
    >>> out = tf(T.Image(lucid.rand(3, 32, 32))).data
    >>> tuple(out.shape)
    (3, 32, 32)
    """

    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p=p)

    @override
    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:
        return 1.0 - img

    @override
    def __repr__(self) -> str:
        return f"InvertImg(p={self.p})"


class ToGray(_NoParams, PhotometricTransform[Empty]):
    r"""Convert an RGB image to grayscale while keeping 3 channels (Albumentations ``ToGray``).

    Applies the BT.601 luminance weights ``Y = 0.299 R + 0.587 G + 0.114 B``
    and broadcasts the scalar luminance back to a 3-channel image so
    downstream ops that expect RGB still work.  Delegates to
    :func:`functional.rgb_to_grayscale` with ``keep_channels=True``.

    Parameters
    ----------
    p : float, optional, default=0.5
        Probability of applying the transform.

    Examples
    --------
    >>> import lucid
    >>> import lucid.utils.transforms as T
    >>> tf = T.ToGray(p=1.0)
    >>> out = tf(T.Image(lucid.rand(3, 32, 32))).data
    >>> tuple(out.shape)
    (3, 32, 32)
    """

    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p=p)

    @override
    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:
        return F.rgb_to_grayscale(img, keep_channels=True)

    @override
    def __repr__(self) -> str:
        return f"ToGray(p={self.p})"


class ToSepia(_NoParams, PhotometricTransform[Empty]):
    r"""Apply the canonical sepia-tone colour matrix (Albumentations ``ToSepia``).

    Multiplies the input by the fixed 3x3 sepia matrix used by
    Microsoft / PIL / Albumentations and clips to ``[0, 1]``.  Produces a
    warm brown-tinted output reminiscent of vintage photographs.

    Parameters
    ----------
    p : float, optional, default=0.5
        Probability of applying the transform.

    Raises
    ------
    ValueError
        If the input image does not have exactly 3 channels.

    Examples
    --------
    >>> import lucid
    >>> import lucid.utils.transforms as T
    >>> tf = T.ToSepia(p=1.0)
    >>> out = tf(T.Image(lucid.rand(3, 32, 32))).data
    >>> tuple(out.shape)
    (3, 32, 32)
    """

    _M = (
        (0.393, 0.769, 0.189),
        (0.349, 0.686, 0.168),
        (0.272, 0.534, 0.131),
    )

    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p=p)

    @override
    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:
        self._require_channels(img, 3)
        r = img[..., 0:1, :, :]
        g = img[..., 1:2, :, :]
        b = img[..., 2:3, :, :]
        m = self._M
        nr = m[0][0] * r + m[0][1] * g + m[0][2] * b
        ng = m[1][0] * r + m[1][1] * g + m[1][2] * b
        nb = m[2][0] * r + m[2][1] * g + m[2][2] * b
        return lucid.clip(F._cat([nr, ng, nb], -3), 0.0, 1.0)

    @override
    def __repr__(self) -> str:
        return f"ToSepia(p={self.p})"


# ── sharpen / emboss ────────────────────────────────────────────────


class Sharpen(PhotometricTransform[TripletParams]):
    r"""Sharpen an image via a 3x3 unsharp kernel blend (Albumentations ``Sharpen``).

    Convolves the image with a 3x3 Laplacian-of-Gaussian style kernel
    whose centre weight is ``8 + lightness``, then blends the sharpened
    result with the original using a per-call ``alpha``.  Higher
    ``alpha`` puts more weight on the sharpened image; ``lightness``
    controls the centre-weight of the kernel and thus the strength of
    the high-frequency boost.

    Parameters
    ----------
    alpha : (float, float), optional, default=(0.2, 0.5)
        Range from which the original / sharpened blend weight is
        sampled uniformly per call.
    lightness : (float, float), optional, default=(0.5, 1.0)
        Range for the kernel-centre additive term ``lightness``; larger
        values give a stronger sharpen.
    p : float, optional, default=0.5
        Probability of applying the transform.

    Examples
    --------
    >>> import lucid
    >>> import lucid.utils.transforms as T
    >>> tf = T.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0)
    >>> out = tf(T.Image(lucid.rand(3, 32, 32))).data
    >>> tuple(out.shape)
    (3, 32, 32)
    """

    def __init__(
        self,
        alpha: tuple[float, float] = (0.2, 0.5),
        lightness: tuple[float, float] = (0.5, 1.0),
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.alpha = alpha
        self.lightness = lightness

    @override
    def make_params(self, img: Tensor) -> TripletParams:
        r"""Sample per-call random parameters for :class:`Sharpen`.

        Parameters
        ----------
        img : Tensor
            Image tensor; not inspected, carried through for dispatch.

        Returns
        -------
        TripletParams
            Carries ``a`` (blend weight from ``alpha``), ``b``
            (centre-weight additive term from ``lightness``), and
            ``c = 0.0`` (unused — third slot kept for triplet
            uniformity with sibling sharpen-family ops).
        """
        return TripletParams(
            a=_random.uniform(*self.alpha),
            b=_random.uniform(*self.lightness),
            c=0.0,
        )

    @override
    def _apply_image(self, img: Tensor, params: TripletParams) -> Tensor:
        light = params.b
        kernel = [[-1.0, -1.0, -1.0], [-1.0, 8.0 + light, -1.0], [-1.0, -1.0, -1.0]]
        sharp = F.depthwise_conv2d(img, kernel)
        out = (1.0 - params.a) * img + params.a * sharp
        return lucid.clip(out, 0.0, 1.0)

    @override
    def __repr__(self) -> str:
        return f"Sharpen(alpha={self.alpha}, lightness={self.lightness}, p={self.p})"


class Emboss(PhotometricTransform[TripletParams]):
    r"""Emboss an image via a 3x3 directional gradient kernel (Albumentations ``Emboss``).

    Convolves the image with an asymmetric 3x3 kernel that highlights
    the top-left to bottom-right gradient, producing a faux 3-D relief
    effect.  Blends the embossed result with the original using a
    per-call ``alpha`` and a per-call ``strength`` that controls the
    off-diagonal kernel weights.

    Parameters
    ----------
    alpha : (float, float), optional, default=(0.2, 0.5)
        Range from which the original / embossed blend weight is sampled
        uniformly per call.
    strength : (float, float), optional, default=(0.2, 0.7)
        Range for the kernel off-diagonal magnitude; larger values give
        a more pronounced relief.
    p : float, optional, default=0.5
        Probability of applying the transform.

    Examples
    --------
    >>> import lucid
    >>> import lucid.utils.transforms as T
    >>> tf = T.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=1.0)
    >>> out = tf(T.Image(lucid.rand(3, 32, 32))).data
    >>> tuple(out.shape)
    (3, 32, 32)
    """

    def __init__(
        self,
        alpha: tuple[float, float] = (0.2, 0.5),
        strength: tuple[float, float] = (0.2, 0.7),
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.alpha = alpha
        self.strength = strength

    @override
    def make_params(self, img: Tensor) -> TripletParams:
        r"""Sample per-call random parameters for :class:`Emboss`.

        Parameters
        ----------
        img : Tensor
            Image tensor; not inspected, carried through for dispatch.

        Returns
        -------
        TripletParams
            Carries ``a`` (blend weight from ``alpha``), ``b``
            (off-diagonal kernel magnitude from ``strength``), and
            ``c = 0.0`` (unused — third slot kept for triplet
            uniformity).
        """
        return TripletParams(
            a=_random.uniform(*self.alpha),
            b=_random.uniform(*self.strength),
            c=0.0,
        )

    @override
    def _apply_image(self, img: Tensor, params: TripletParams) -> Tensor:
        s = params.b
        kernel = [[-1.0 - s, -s, 0.0], [-s, 1.0, s], [0.0, s, 1.0 + s]]
        emb = F.depthwise_conv2d(img, kernel)
        out = (1.0 - params.a) * img + params.a * emb
        return lucid.clip(out, 0.0, 1.0)

    @override
    def __repr__(self) -> str:
        return f"Emboss(alpha={self.alpha}, strength={self.strength}, p={self.p})"


class RandomToneCurve(PhotometricTransform[ScalarParams]):
    r"""Apply a random S-shaped tone curve around midtones (Albumentations ``RandomToneCurve``).

    Samples an amount ``a`` uniformly from ``(-scale, scale)`` and adds
    ``a * sin(2 * pi * x)`` to the clipped image.  The sine wave biases
    midtones up (if ``a > 0``) or down (if ``a < 0``) while leaving the
    deep shadows and bright highlights near identity, producing a
    classic photographic S-curve.

    Parameters
    ----------
    scale : float, optional, default=0.1
        Half-width of the symmetric range from which the tone-curve
        amount is sampled (uniform ``(-scale, scale)``).
    p : float, optional, default=0.5
        Probability of applying the transform.

    Examples
    --------
    >>> import lucid
    >>> import lucid.utils.transforms as T
    >>> tf = T.RandomToneCurve(scale=0.1, p=1.0)
    >>> out = tf(T.Image(lucid.rand(3, 32, 32))).data
    >>> tuple(out.shape)
    (3, 32, 32)
    """

    def __init__(self, scale: float = 0.1, p: float = 0.5) -> None:
        super().__init__(p=p)
        self.scale = scale

    @override
    def make_params(self, img: Tensor) -> ScalarParams:
        r"""Sample per-call random parameters for :class:`RandomToneCurve`.

        Parameters
        ----------
        img : Tensor
            Image tensor; not inspected, carried through for dispatch.

        Returns
        -------
        ScalarParams
            Carries ``value`` — the S-curve amount drawn uniformly
            from ``(-scale, scale)``; positive values brighten
            midtones, negative values darken them.
        """
        return ScalarParams(value=_random.uniform(-self.scale, self.scale))

    @override
    def _apply_image(self, img: Tensor, params: ScalarParams) -> Tensor:
        # Smooth S-curve around 0.5 controlled by the sampled amount.
        x = lucid.clip(img, 0.0, 1.0)
        out = x + params.value * lucid.sin(x * (2.0 * 3.141592653589793))
        return lucid.clip(out, 0.0, 1.0)

    @override
    def __repr__(self) -> str:
        return f"RandomToneCurve(scale={self.scale}, p={self.p})"


# ── B8: additional colour / pixel transforms ────────────────────────


class RandomBrightness(PhotometricTransform[ScalarParams]):
    r"""Randomly perturb brightness only (Albumentations ``RandomBrightness``).

    Samples an additive brightness offset uniformly from ``limit``,
    adds it to the image, and clips to ``[0, 1]``.  Equivalent to the
    brightness leg of :class:`RandomBrightnessContrast` — included as
    a standalone class for parity with the Albumentations API.

    Parameters
    ----------
    limit : float or (float, float), optional, default=0.2
        Additive brightness offset range.  Scalar ``v`` is interpreted
        as the symmetric range ``(-v, v)``.
    p : float, optional, default=0.5
        Probability of applying the transform.

    Examples
    --------
    >>> import lucid
    >>> import lucid.utils.transforms as T
    >>> tf = T.RandomBrightness(limit=0.2, p=1.0)
    >>> out = tf(T.Image(lucid.rand(3, 32, 32))).data
    >>> tuple(out.shape)
    (3, 32, 32)
    """

    def __init__(self, limit: float | tuple[float, float] = 0.2, p: float = 0.5) -> None:
        super().__init__(p=p)
        self.limit = _rng(limit)

    @override
    def make_params(self, img: Tensor) -> ScalarParams:
        r"""Sample per-call random parameters for :class:`RandomBrightness`.

        Parameters
        ----------
        img : Tensor
            Image tensor; not inspected, carried through for dispatch.

        Returns
        -------
        ScalarParams
            Carries ``value`` — the additive brightness offset drawn
            uniformly from ``limit``.
        """
        return ScalarParams(value=_random.uniform(*self.limit))

    @override
    def _apply_image(self, img: Tensor, params: ScalarParams) -> Tensor:
        return lucid.clip(img + params.value, 0.0, 1.0)

    @override
    def __repr__(self) -> str:
        return f"RandomBrightness(limit={self.limit}, p={self.p})"


class RandomContrast(PhotometricTransform[ScalarParams]):
    r"""Randomly perturb contrast only (Albumentations ``RandomContrast``).

    Samples a contrast factor uniformly from ``limit``, multiplies the
    image by ``1 + factor``, and clips to ``[0, 1]``.  Equivalent to
    the contrast leg of :class:`RandomBrightnessContrast` — included as
    a standalone class for parity with the Albumentations API.

    Parameters
    ----------
    limit : float or (float, float), optional, default=0.2
        Multiplicative contrast offset range (multiplier is
        ``1 + sampled_value``).  Scalar ``v`` expands to ``(-v, v)``.
    p : float, optional, default=0.5
        Probability of applying the transform.

    Examples
    --------
    >>> import lucid
    >>> import lucid.utils.transforms as T
    >>> tf = T.RandomContrast(limit=0.2, p=1.0)
    >>> out = tf(T.Image(lucid.rand(3, 32, 32))).data
    >>> tuple(out.shape)
    (3, 32, 32)
    """

    def __init__(self, limit: float | tuple[float, float] = 0.2, p: float = 0.5) -> None:
        super().__init__(p=p)
        self.limit = _rng(limit)

    @override
    def make_params(self, img: Tensor) -> ScalarParams:
        r"""Sample per-call random parameters for :class:`RandomContrast`.

        Parameters
        ----------
        img : Tensor
            Image tensor; not inspected, carried through for dispatch.

        Returns
        -------
        ScalarParams
            Carries ``value`` — the contrast delta drawn uniformly
            from ``limit``; the effective multiplier is ``1 + value``.
        """
        return ScalarParams(value=_random.uniform(*self.limit))

    @override
    def _apply_image(self, img: Tensor, params: ScalarParams) -> Tensor:
        return lucid.clip(img * (1.0 + params.value), 0.0, 1.0)

    @override
    def __repr__(self) -> str:
        return f"RandomContrast(limit={self.limit}, p={self.p})"


class UnsharpMask(PhotometricTransform[TripletParams]):
    r"""Unsharp-mask sharpen via ``img + alpha*(img - blur)`` (Albumentations ``UnsharpMask``).

    Subtracts a Gaussian-blurred copy from the original and adds the
    high-frequency residual back, scaled by ``alpha``.  This is the
    photographic "unsharp mask" used to enhance edges without amplifying
    noise as much as a raw Laplacian sharpen would.  The kernel size
    ``k`` and Gaussian ``sigma`` are sampled per call from
    ``blur_limit`` / ``sigma_limit``; ``sigma`` defaults to the
    OpenCV-style heuristic ``0.3 * ((k - 1) * 0.5 - 1.0) + 0.8`` when
    the sampled value is non-positive.

    Parameters
    ----------
    blur_limit : (int, int), optional, default=(3, 7)
        Inclusive range from which the Gaussian kernel size ``k`` is
        sampled (snapped to the next odd integer).
    sigma_limit : float or (float, float), optional, default=0.0
        Range from which the Gaussian standard deviation is sampled
        uniformly.  A scalar ``s`` is treated as ``(0, s)``; ``0`` falls
        back to the kernel-derived heuristic above.
    alpha : (float, float), optional, default=(0.2, 0.5)
        Range from which the residual blend weight is sampled
        uniformly; higher values yield a stronger sharpen.
    threshold : float, optional, default=10.0
        Reserved for parity with the reference framework's
        masked-residual variant; kept on the signature for forward
        compatibility but currently unused (every pixel is sharpened).
    p : float, optional, default=0.5
        Probability of applying the transform.

    Examples
    --------
    >>> import lucid
    >>> import lucid.utils.transforms as T
    >>> tf = T.UnsharpMask(blur_limit=(3, 7), alpha=(0.2, 0.5), p=1.0)
    >>> out = tf(T.Image(lucid.rand(3, 32, 32))).data
    >>> tuple(out.shape)
    (3, 32, 32)
    """

    def __init__(
        self,
        blur_limit: tuple[int, int] = (3, 7),
        sigma_limit: float | tuple[float, float] = 0.0,
        alpha: tuple[float, float] = (0.2, 0.5),
        threshold: float = 10.0,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.blur_limit = blur_limit
        self.sigma_limit = (
            (0.0, sigma_limit) if isinstance(sigma_limit, (int, float)) else sigma_limit
        )
        self.alpha = alpha
        self.threshold = threshold

    @override
    def make_params(self, img: Tensor) -> TripletParams:
        r"""Sample per-call random parameters for :class:`UnsharpMask`.

        Parameters
        ----------
        img : Tensor
            Image tensor; not inspected, carried through for dispatch.

        Returns
        -------
        TripletParams
            Carries ``a`` (residual blend weight from ``alpha``),
            ``b`` (Gaussian kernel side ``k`` stored as float so the
            triplet stays homogeneous), and ``c`` (Gaussian sigma).

        Notes
        -----
        ``k`` is forced odd via :func:`_odd`.  A non-positive ``sigma``
        falls back to OpenCV's kernel-derived
        ``0.3 * ((k - 1) * 0.5 - 1.0) + 0.8`` heuristic.
        """
        from lucid.utils.transforms._blur import _odd

        k = _odd(_random.randint(self.blur_limit[0], self.blur_limit[1] + 1))
        sigma = _random.uniform(self.sigma_limit[0], self.sigma_limit[1])
        if sigma <= 0.0:
            sigma = 0.3 * ((k - 1) * 0.5 - 1.0) + 0.8
        return TripletParams(a=_random.uniform(*self.alpha), b=float(k), c=sigma)

    @override
    def _apply_image(self, img: Tensor, params: TripletParams) -> Tensor:
        blur = F.gaussian_blur(img, params.c, ksize=int(params.b))
        return lucid.clip(img + params.a * (img - blur), 0.0, 1.0)

    @override
    def __repr__(self) -> str:
        return f"UnsharpMask(blur_limit={self.blur_limit}, alpha={self.alpha}, p={self.p})"


class RingingOvershoot(PhotometricTransform[Empty]):
    r"""Simulate ringing / overshoot artefacts via a high-pass kernel (Albumentations ``RingingOvershoot``).

    Convolves the image with a 3x3 high-pass kernel that exaggerates
    edges, producing the bright halo / undershoot pattern typical of
    aggressive sharpening or low-pass filter ringing.  Useful for
    simulating compression artefacts and JPEG-style edge halos during
    training.

    Parameters
    ----------
    blur_limit : (int, int), optional, default=(7, 15)
        Reserved for parity with the Albumentations API; the current
        implementation uses a fixed 3x3 high-pass kernel and ignores
        this argument, but it is kept on the signature for forward
        compatibility.
    p : float, optional, default=0.5
        Probability of applying the transform.

    Examples
    --------
    >>> import lucid
    >>> import lucid.utils.transforms as T
    >>> tf = T.RingingOvershoot(blur_limit=(7, 15), p=1.0)
    >>> out = tf(T.Image(lucid.rand(3, 32, 32))).data
    >>> tuple(out.shape)
    (3, 32, 32)
    """

    def __init__(self, blur_limit: tuple[int, int] = (7, 15), p: float = 0.5) -> None:
        super().__init__(p=p)
        self.blur_limit = blur_limit

    @override
    def make_params(self, img: Tensor) -> Empty:
        r"""Return the no-op parameter sentinel for :class:`RingingOvershoot`.

        Parameters
        ----------
        img : Tensor
            Image tensor; not inspected.

        Returns
        -------
        Empty
            The shared :data:`NO_PARAMS` sentinel — the current
            implementation uses a fixed high-pass kernel, so no
            per-call sampling is needed.  ``blur_limit`` is kept on
            the signature for forward compatibility but is unused.
        """
        from lucid.utils.transforms._base import NO_PARAMS

        return NO_PARAMS

    @override
    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:
        # Approximate ringing with a sharpening (high-pass) kernel.
        kernel = [[0.0, -0.25, 0.0], [-0.25, 2.0, -0.25], [0.0, -0.25, 0.0]]
        return lucid.clip(F.depthwise_conv2d(img, kernel), 0.0, 1.0)

    @override
    def __repr__(self) -> str:
        return f"RingingOvershoot(blur_limit={self.blur_limit}, p={self.p})"


class FancyPCA(PhotometricTransform[Empty]):
    r"""AlexNet-style PCA colour augmentation (Krizhevsky 2012, Albumentations ``FancyPCA``).

    Computes the per-image RGB covariance matrix, takes its
    eigendecomposition, and adds a random linear combination of the
    eigenvectors weighted by their eigenvalues to every pixel.  The
    weights are i.i.d. samples from ``alpha * U(-1, 1)`` per channel —
    same recipe used in the original AlexNet training pipeline.

    Parameters
    ----------
    alpha : float, optional, default=0.1
        Standard deviation scale on the eigenvalue-weighted perturbation;
        larger values give more colour jitter.
    p : float, optional, default=0.5
        Probability of applying the transform.

    Raises
    ------
    ValueError
        If the input image does not have exactly 3 channels.

    Examples
    --------
    >>> import lucid
    >>> import lucid.utils.transforms as T
    >>> tf = T.FancyPCA(alpha=0.1, p=1.0)
    >>> out = tf(T.Image(lucid.rand(3, 32, 32))).data
    >>> tuple(out.shape)
    (3, 32, 32)
    """

    def __init__(self, alpha: float = 0.1, p: float = 0.5) -> None:
        super().__init__(p=p)
        self.alpha = alpha

    @override
    def make_params(self, img: Tensor) -> Empty:
        r"""Return the no-op parameter sentinel for :class:`FancyPCA`.

        Parameters
        ----------
        img : Tensor
            Image tensor; not inspected here.

        Returns
        -------
        Empty
            The shared :data:`NO_PARAMS` sentinel — the random
            eigenvector weights are drawn inline in the apply step
            because they depend on the per-image covariance
            decomposition.
        """
        from lucid.utils.transforms._base import NO_PARAMS

        return NO_PARAMS

    @override
    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:
        self._require_channels(img, 3)
        c = 3
        flat = img.reshape(c, -1)  # (3, N)
        mean = lucid.mean(flat, dim=[1], keepdim=True)
        centered = flat - mean
        cov = lucid.matmul(centered, lucid.swapaxes(centered, 0, 1)) / float(flat.shape[1])
        evals, evecs = lucid.linalg.eigh(cov)
        alphas = [self.alpha * _random.uniform(-1.0, 1.0) for _ in range(c)]
        scaled = lucid.tensor(
            [alphas[i] for i in range(c)], dtype=img.dtype, device=img.device
        ).reshape(c, 1) * evals.reshape(c, 1)
        delta = lucid.matmul(evecs, scaled).reshape(c, 1, 1)  # (3,1,1)
        if img.ndim == 4:
            delta = delta[None]
        return lucid.clip(img + delta, 0.0, 1.0)

    @override
    def __repr__(self) -> str:
        return f"FancyPCA(alpha={self.alpha}, p={self.p})"


@dataclass(frozen=True, slots=True)
class PixelMaskParams:
    """Per-call dropout mask (broadcastable to the image) used by :class:`PixelDropout`."""

    mask: Tensor


class PixelDropout(PhotometricTransform[PixelMaskParams]):
    r"""Randomly set individual pixels to a fill value (Albumentations ``PixelDropout``).

    Samples a per-pixel Bernoulli mask with drop probability
    ``dropout_prob`` and replaces masked pixels with ``drop_value``
    using a multiplicative keep-mask + additive fill (autograd-friendly,
    no in-place writes).  When ``per_channel=True`` the mask is sampled
    independently for each channel; otherwise a single mask is broadcast
    across channels.

    Parameters
    ----------
    dropout_prob : float, optional, default=0.01
        Independent per-pixel drop probability.
    per_channel : bool, optional, default=False
        If ``True``, sample the dropout mask independently per channel.
        If ``False``, a single mask is shared across channels so dropped
        pixels go to the fill value in all channels at once.
    drop_value : float, optional, default=0.0
        Value written into dropped pixels.
    p : float, optional, default=0.5
        Probability of applying the transform.

    Examples
    --------
    >>> import lucid
    >>> import lucid.utils.transforms as T
    >>> tf = T.PixelDropout(dropout_prob=0.05, per_channel=False, p=1.0)
    >>> out = tf(T.Image(lucid.rand(3, 32, 32))).data
    >>> tuple(out.shape)
    (3, 32, 32)
    """

    def __init__(
        self,
        dropout_prob: float = 0.01,
        per_channel: bool = False,
        drop_value: float = 0.0,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.dropout_prob = dropout_prob
        self.per_channel = per_channel
        self.drop_value = drop_value

    @override
    def make_params(self, img: Tensor) -> PixelMaskParams:
        r"""Sample per-call random parameters for :class:`PixelDropout`.

        Parameters
        ----------
        img : Tensor
            Image tensor; its channel count and spatial size are read
            to size the Bernoulli keep-mask.

        Returns
        -------
        PixelMaskParams
            Carries ``mask`` — a per-pixel keep mask of shape
            ``(C, H, W)`` (when ``per_channel=True``) or ``(1, H, W)``
            (otherwise), with each entry independently 1 with
            probability ``1 - dropout_prob``.
        """
        c, h, w = int(img.shape[-3]), *F._spatial_hw(img)
        shape = (c, h, w) if self.per_channel else (1, h, w)
        keep = (lucid.rand(*shape, device=img.device) >= self.dropout_prob).to(img.dtype)
        return PixelMaskParams(mask=keep)

    @override
    def _apply_image(self, img: Tensor, params: PixelMaskParams) -> Tensor:
        keep = params.mask[None] if img.ndim == 4 else params.mask
        return img * keep + self.drop_value * (1.0 - keep)

    @override
    def __repr__(self) -> str:
        return f"PixelDropout(dropout_prob={self.dropout_prob}, p={self.p})"


@dataclass(frozen=True, slots=True)
class BandParams:
    """Per-call row / column band intervals used by :class:`XYMasking`.

    Each interval is ``(start, stop)`` in pixel coordinates (half-open).
    """

    rows: tuple[tuple[int, int], ...]
    cols: tuple[tuple[int, int], ...]


class XYMasking(PhotometricTransform[BandParams]):
    r"""Mask random horizontal and vertical bands of the image (Albumentations ``XYMasking``).

    Samples ``num_masks_x`` vertical column-bands of width
    ``mask_x_length`` and ``num_masks_y`` horizontal row-bands of height
    ``mask_y_length``, each placed at a uniformly-sampled offset.  All
    sampled bands are unioned into a single multiplicative keep-mask
    and applied via the standard mask + fill pattern.  Equivalent to
    SpecAugment-style time / frequency masking but for spatial inputs.

    Parameters
    ----------
    num_masks_x : int, optional, default=0
        Number of vertical (column-direction) bands to sample.
    num_masks_y : int, optional, default=0
        Number of horizontal (row-direction) bands to sample.
    mask_x_length : int, optional, default=10
        Width in pixels of each vertical band.
    mask_y_length : int, optional, default=10
        Height in pixels of each horizontal band.
    fill_value : float, optional, default=0.0
        Value written into the masked bands.
    p : float, optional, default=0.5
        Probability of applying the transform.

    Examples
    --------
    >>> import lucid
    >>> import lucid.utils.transforms as T
    >>> tf = T.XYMasking(num_masks_x=2, num_masks_y=2, mask_x_length=8, mask_y_length=8, p=1.0)
    >>> out = tf(T.Image(lucid.rand(3, 32, 32))).data
    >>> tuple(out.shape)
    (3, 32, 32)
    """

    def __init__(
        self,
        num_masks_x: int = 0,
        num_masks_y: int = 0,
        mask_x_length: int = 10,
        mask_y_length: int = 10,
        fill_value: float = 0.0,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.num_masks_x = num_masks_x
        self.num_masks_y = num_masks_y
        self.mask_x_length = mask_x_length
        self.mask_y_length = mask_y_length
        self.fill_value = fill_value

    @override
    def make_params(self, img: Tensor) -> BandParams:
        r"""Sample per-call random parameters for :class:`XYMasking`.

        Parameters
        ----------
        img : Tensor
            Image tensor; its spatial size is read to bound the
            uniformly-sampled band start positions.

        Returns
        -------
        BandParams
            Carries ``rows`` (``num_masks_y`` horizontal band intervals)
            and ``cols`` (``num_masks_x`` vertical band intervals),
            each interval ``(start, stop)`` with width ``mask_y_length``
            / ``mask_x_length`` (clipped to the image edge).
        """
        h, w = F._spatial_hw(img)
        cols = tuple(
            (lambda s: (s, min(s + self.mask_x_length, w)))(_random.randint(0, max(w - 1, 1)))
            for _ in range(self.num_masks_x)
        )
        rows = tuple(
            (lambda s: (s, min(s + self.mask_y_length, h)))(_random.randint(0, max(h - 1, 1)))
            for _ in range(self.num_masks_y)
        )
        return BandParams(rows=rows, cols=cols)

    @override
    def _apply_image(self, img: Tensor, params: BandParams) -> Tensor:
        h, w = F._spatial_hw(img)
        keep = lucid.ones(1, h, w, dtype=img.dtype, device=img.device)
        for r0, r1 in params.rows:
            band = F.pad(
                lucid.zeros(1, r1 - r0, w, dtype=img.dtype, device=img.device),
                (0, 0, r0, h - r1),
                value=1.0,
            )
            keep = keep * band
        for c0, c1 in params.cols:
            band = F.pad(
                lucid.zeros(1, h, c1 - c0, dtype=img.dtype, device=img.device),
                (c0, w - c1, 0, 0),
                value=1.0,
            )
            keep = keep * band
        keep_b = keep[None] if img.ndim == 4 else keep
        return img * keep_b + self.fill_value * (1.0 - keep_b)

    @override
    def __repr__(self) -> str:
        return (
            f"XYMasking(num_masks_x={self.num_masks_x}, "
            f"num_masks_y={self.num_masks_y}, p={self.p})"
        )
