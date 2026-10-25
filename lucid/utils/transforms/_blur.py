"""Blur + noise transforms (Albumentations-compatible).

All :class:`~lucid.utils.transforms._base.PhotometricTransform` — they
act only on the image and leave masks / boxes / keypoints untouched.
"""

import math
from typing import override
from dataclasses import dataclass

import lucid
from lucid._tensor import Tensor
from lucid.utils.transforms import _random
from lucid.utils.transforms import functional as F
from lucid.utils.transforms._base import PhotometricTransform


def _odd(k: int) -> int:
    return k if k % 2 == 1 else k + 1


@dataclass(frozen=True, slots=True)
class KSizeParam:
    r"""Per-call kernel-size parameter for box / median / motion blur.

    Carried by :class:`Blur` and :class:`MedianBlur` from
    :meth:`make_params` into ``_apply_image``; the kernel is built as a
    ``ksize x ksize`` window centred on each pixel.

    Attributes
    ----------
    ksize : int
        Odd kernel side length in pixels.  Sampling routes round even
        values up to the next odd integer so the window stays centred.
    """

    ksize: int


@dataclass(frozen=True, slots=True)
class SigmaParam:
    r"""Per-call kernel size + Gaussian standard deviation.

    Carried by :class:`GaussianBlur` (and :class:`UnsharpMask`) so the
    same odd ``ksize`` and sampled ``sigma`` feed both the kernel
    construction and the cv2-compatible fallback formula.

    Attributes
    ----------
    ksize : int
        Odd Gaussian-kernel side length in pixels.
    sigma : float
        Gaussian standard deviation; if the user requested ``0`` the
        OpenCV auto-sigma rule ``0.3 * ((k - 1) * 0.5 - 1.0) + 0.8`` is
        applied before this dataclass is constructed.
    """

    ksize: int
    sigma: float


@dataclass(frozen=True, slots=True)
class MotionParam:
    r"""Per-call motion-blur kernel size and streak angle.

    Carried by :class:`MotionBlur`; the kernel is built by drawing a
    1-pixel-wide line at ``angle`` through the centre of a
    ``ksize x ksize`` window and normalising.

    Attributes
    ----------
    ksize : int
        Odd kernel side length in pixels.
    angle : float
        Streak direction in degrees, sampled from ``[0, 180)``.
    """

    ksize: int
    angle: float


@dataclass(frozen=True, slots=True)
class NoiseParam:
    r"""Per-call noise magnitude for additive / sensor-style noise.

    Used by :class:`GaussNoise` (additive Gaussian on the unit-scale
    image) and :class:`ISONoise` (luminance Gaussian + saturation
    perturbation).  The semantics of ``std`` depend on the host:
    :class:`GaussNoise` interprets it as the unit-scale standard
    deviation already divided by 255, while :class:`ISONoise` uses it
    as a generic intensity scale (multiplied by ``0.1`` internally).

    Attributes
    ----------
    std : float
        Standard deviation / intensity of the per-pixel noise.
    """

    std: float


@dataclass(frozen=True, slots=True)
class MultiplierParam:
    r"""Per-call multiplicative-noise bounds.

    Carried by :class:`MultiplicativeNoise`.  The bounds are passed
    through to the apply step so a per-pixel uniform sample (when
    ``elementwise=True``) or a single scalar (otherwise) can be drawn
    from the same ``[lo, hi]`` range.

    Attributes
    ----------
    lo : float
        Inclusive lower bound of the multiplier range.
    hi : float
        Inclusive upper bound of the multiplier range.
    """

    lo: float
    hi: float


@dataclass(frozen=True, slots=True)
class ScaleParam:
    r"""Per-call downscale factor used by :class:`Downscale`.

    The :class:`Downscale` apply step resizes the image to
    ``(round(H * scale), round(W * scale))`` then resizes back to
    ``(H, W)`` via nearest-neighbour, simulating sensor / JPEG
    block-level degradation.

    Attributes
    ----------
    scale : float
        Downscale factor in ``(0, 1]``; smaller values lose more
        high-frequency detail.
    """

    scale: float


# ── blur family ─────────────────────────────────────────────────────


class Blur(PhotometricTransform[KSizeParam]):
    r"""Box (mean) blur with a random odd kernel size (Albumentations ``Blur``).

    Samples an odd kernel size ``k`` uniformly from ``[3, blur_limit]``
    and convolves the image with a normalised ``k x k`` all-ones kernel
    via :func:`functional.depthwise_conv2d`.

    Parameters
    ----------
    blur_limit : int, optional, default=7
        Inclusive upper bound on the sampled kernel size; values are
        rounded up to the next odd integer to keep the kernel centred.
    p : float, optional, default=0.5
        Probability of applying the transform.

    Examples
    --------
    >>> import lucid, lucid.utils.transforms as T
    >>> tf = T.Blur(p=1.0)
    >>> tuple(tf(T.Image(lucid.rand(3, 32, 32))).data.shape)
    (3, 32, 32)
    """

    def __init__(self, blur_limit: int = 7, p: float = 0.5) -> None:
        super().__init__(p=p)
        self.blur_limit = blur_limit

    @override
    def make_params(self, img: Tensor) -> KSizeParam:
        r"""Sample per-call random parameters for :class:`Blur`.

        Parameters
        ----------
        img : Tensor
            Image tensor used only to fix the parameter dtype / dispatch
            context; no spatial information is read.

        Returns
        -------
        KSizeParam
            Carries ``ksize`` — the odd box-kernel side length sampled
            uniformly from ``[3, blur_limit]``.

        Notes
        -----
        Even draws are forced to the next odd integer so the kernel
        stays centred on each output pixel.
        """
        return KSizeParam(ksize=_odd(_random.randint(3, self.blur_limit + 1)))

    @override
    def _apply_image(self, img: Tensor, params: KSizeParam) -> Tensor:
        k = params.ksize
        kernel = [[1.0 / (k * k)] * k for _ in range(k)]
        return F.depthwise_conv2d(img, kernel)

    @override
    def __repr__(self) -> str:
        return f"Blur(blur_limit={self.blur_limit}, p={self.p})"


class MedianBlur(PhotometricTransform[KSizeParam]):
    r"""Median filter with a random odd kernel (Albumentations ``MedianBlur``).

    Samples an odd kernel size ``k`` uniformly from ``[3, blur_limit]``,
    builds the ``k * k`` shifted-neighbour stack via :func:`lucid.roll`,
    sorts along the stack axis, and returns the median element — an
    impulse-noise denoiser that preserves edges better than ``Blur``.

    Parameters
    ----------
    blur_limit : int, optional, default=7
        Inclusive upper bound on the sampled kernel size; rounded up
        to the next odd integer so the window is centred.
    p : float, optional, default=0.5
        Probability of applying the transform.

    Examples
    --------
    >>> import lucid, lucid.utils.transforms as T
    >>> tf = T.MedianBlur(p=1.0)
    >>> tuple(tf(T.Image(lucid.rand(3, 32, 32))).data.shape)
    (3, 32, 32)
    """

    def __init__(self, blur_limit: int = 7, p: float = 0.5) -> None:
        super().__init__(p=p)
        self.blur_limit = blur_limit

    @override
    def make_params(self, img: Tensor) -> KSizeParam:
        r"""Sample per-call random parameters for :class:`MedianBlur`.

        Parameters
        ----------
        img : Tensor
            Image tensor; not inspected, only carried through for the
            transform dispatch.

        Returns
        -------
        KSizeParam
            Carries ``ksize`` — the odd window side length sampled
            uniformly from ``[3, blur_limit]``.

        Notes
        -----
        Even draws are forced to the next odd integer so the median
        window is centred on each output pixel.
        """
        return KSizeParam(ksize=_odd(_random.randint(3, self.blur_limit + 1)))

    @override
    def _apply_image(self, img: Tensor, params: KSizeParam) -> Tensor:
        k = params.ksize
        half = k // 2
        neigh = [
            lucid.roll(img, (dy, dx), dims=(-2, -1))
            for dy in range(-half, half + 1)
            for dx in range(-half, half + 1)
        ]
        stacked = lucid.stack(neigh, dim=0)
        ordered = lucid.sort(stacked, dim=0).values
        return ordered[(k * k) // 2]

    @override
    def __repr__(self) -> str:
        return f"MedianBlur(blur_limit={self.blur_limit}, p={self.p})"


class MotionBlur(PhotometricTransform[MotionParam]):
    r"""Directional (linear streak) motion blur (Albumentations ``MotionBlur``).

    Samples an odd kernel size and a streak angle in ``[0, 180)°``, draws
    a 1-pixel-wide line through the kernel centre at that angle, then
    convolves the normalised kernel with the image — approximates the
    look of a camera panning while the shutter is open.

    Parameters
    ----------
    blur_limit : int, optional, default=7
        Inclusive upper bound on the kernel size; rounded up to the
        next odd integer.
    p : float, optional, default=0.5
        Probability of applying the transform.

    Examples
    --------
    >>> import lucid, lucid.utils.transforms as T
    >>> tf = T.MotionBlur(p=1.0)
    >>> tuple(tf(T.Image(lucid.rand(3, 32, 32))).data.shape)
    (3, 32, 32)
    """

    def __init__(self, blur_limit: int = 7, p: float = 0.5) -> None:
        super().__init__(p=p)
        self.blur_limit = blur_limit

    @override
    def make_params(self, img: Tensor) -> MotionParam:
        r"""Sample per-call random parameters for :class:`MotionBlur`.

        Parameters
        ----------
        img : Tensor
            Image tensor; not inspected, carried through for dispatch.

        Returns
        -------
        MotionParam
            Carries ``ksize`` (odd kernel side, ``[3, blur_limit]``)
            and ``angle`` in degrees sampled uniformly from
            ``[0, 180)``.

        Notes
        -----
        Even kernel-size draws are forced to the next odd integer.
        Angles outside ``[0, 180)`` are unnecessary since the streak
        kernel is symmetric under 180° rotation.
        """
        return MotionParam(
            ksize=_odd(_random.randint(3, self.blur_limit + 1)),
            angle=_random.uniform(0.0, 180.0),
        )

    @override
    def _apply_image(self, img: Tensor, params: MotionParam) -> Tensor:
        k = params.ksize
        half = k // 2
        kernel = [[0.0] * k for _ in range(k)]
        rad = math.radians(params.angle)
        dx, dy = math.cos(rad), math.sin(rad)
        for t in range(-half, half + 1):
            x = int(round(half + t * dx))
            y = int(round(half + t * dy))
            if 0 <= x < k and 0 <= y < k:
                kernel[y][x] = 1.0
        total = sum(sum(r) for r in kernel) or 1.0
        kernel = [[v / total for v in r] for r in kernel]
        return F.depthwise_conv2d(img, kernel)

    @override
    def __repr__(self) -> str:
        return f"MotionBlur(blur_limit={self.blur_limit}, p={self.p})"


class GaussianBlur(PhotometricTransform[SigmaParam]):
    r"""Gaussian blur with random kernel size and sigma (Albumentations ``GaussianBlur``).

    Samples an odd kernel size from ``blur_limit`` and a sigma from
    ``sigma_limit``; when sigma is left at zero the OpenCV default
    ``0.3 * ((k - 1) * 0.5 - 1.0) + 0.8`` is used so the kernel matches
    cv2's auto-sigma policy.

    Parameters
    ----------
    blur_limit : (int, int), optional, default=(3, 7)
        Inclusive range for the kernel size; both ends are rounded up
        to the next odd integer.
    sigma_limit : float or (float, float), optional, default=0.0
        Range for the Gaussian sigma.  A scalar ``v`` expands to
        ``(0.0, v)``; ``0`` triggers the OpenCV auto-sigma formula.
    p : float, optional, default=0.5
        Probability of applying the transform.

    Examples
    --------
    >>> import lucid, lucid.utils.transforms as T
    >>> tf = T.GaussianBlur(p=1.0)
    >>> tuple(tf(T.Image(lucid.rand(3, 32, 32))).data.shape)
    (3, 32, 32)
    """

    def __init__(
        self,
        blur_limit: tuple[int, int] = (3, 7),
        sigma_limit: float | tuple[float, float] = 0.0,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.blur_limit = blur_limit
        self.sigma_limit = (
            (0.0, sigma_limit) if isinstance(sigma_limit, (int, float)) else sigma_limit
        )

    @override
    def make_params(self, img: Tensor) -> SigmaParam:
        r"""Sample per-call random parameters for :class:`GaussianBlur`.

        Parameters
        ----------
        img : Tensor
            Image tensor; not inspected, carried through for dispatch.

        Returns
        -------
        SigmaParam
            Carries ``ksize`` (odd kernel side from ``blur_limit``) and
            ``sigma`` (Gaussian standard deviation).

        Notes
        -----
        A non-positive sampled ``sigma`` falls back to OpenCV's
        kernel-derived default ``0.3 * ((k - 1) * 0.5 - 1.0) + 0.8``
        so the Gaussian matches cv2's auto-sigma policy.
        """
        k = _odd(_random.randint(self.blur_limit[0], self.blur_limit[1] + 1))
        sigma = _random.uniform(self.sigma_limit[0], self.sigma_limit[1])
        if sigma <= 0.0:
            sigma = 0.3 * ((k - 1) * 0.5 - 1.0) + 0.8  # OpenCV default
        return SigmaParam(ksize=k, sigma=sigma)

    @override
    def _apply_image(self, img: Tensor, params: SigmaParam) -> Tensor:
        return F.gaussian_blur(img, params.sigma, ksize=params.ksize)

    @override
    def __repr__(self) -> str:
        return f"GaussianBlur(blur_limit={self.blur_limit}, p={self.p})"


# ── noise family ────────────────────────────────────────────────────


class GaussNoise(PhotometricTransform[NoiseParam]):
    r"""Additive Gaussian noise (Albumentations ``GaussNoise``).

    ``var_limit`` is on the 0-255 scale (matching Albumentations).

    Examples
    --------
    >>> import lucid, lucid.utils.transforms as T
    >>> tf = T.GaussNoise(p=1.0)
    >>> tuple(tf(T.Image(lucid.rand(3, 32, 32))).data.shape)
    (3, 32, 32)
    """

    def __init__(
        self,
        var_limit: tuple[float, float] = (10.0, 50.0),
        mean: float = 0.0,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.var_limit = var_limit
        self.mean = mean

    @override
    def make_params(self, img: Tensor) -> NoiseParam:
        r"""Sample per-call random parameters for :class:`GaussNoise`.

        Parameters
        ----------
        img : Tensor
            Image tensor; not inspected, carried through for dispatch.

        Returns
        -------
        NoiseParam
            Carries ``std`` — the per-pixel Gaussian standard deviation
            on the unit ``[0, 1]`` scale.

        Notes
        -----
        ``var_limit`` is on the 0-255 scale (Albumentations
        convention); the sampled variance is square-rooted and
        divided by 255 so it can be added directly to the unit-scale
        image.
        """
        var = _random.uniform(self.var_limit[0], self.var_limit[1])
        return NoiseParam(std=math.sqrt(var) / 255.0)

    @override
    def _apply_image(self, img: Tensor, params: NoiseParam) -> Tensor:
        noise = (
            lucid.randn(*img.shape, device=img.device) * params.std + self.mean / 255.0
        )
        return lucid.clip(img + noise, 0.0, 1.0)

    @override
    def __repr__(self) -> str:
        return f"GaussNoise(var_limit={self.var_limit}, p={self.p})"


class MultiplicativeNoise(PhotometricTransform[MultiplierParam]):
    r"""Multiply pixel values by random noise (Albumentations ``MultiplicativeNoise``).

    With ``elementwise=False`` (default) a single scalar is sampled
    per call and the whole image is scaled by it; with
    ``elementwise=True`` an independent multiplier is sampled per
    pixel.  The output is always clipped back to ``[0, 1]``.

    Parameters
    ----------
    multiplier : (float, float), optional, default=(0.9, 1.1)
        Inclusive range of the per-pixel (or per-image) multiplier.
    elementwise : bool, optional, default=False
        If ``True``, sample one multiplier per pixel rather than one
        per image.
    p : float, optional, default=0.5
        Probability of applying the transform.

    Examples
    --------
    >>> import lucid, lucid.utils.transforms as T
    >>> tf = T.MultiplicativeNoise(p=1.0)
    >>> tuple(tf(T.Image(lucid.rand(3, 32, 32))).data.shape)
    (3, 32, 32)
    """

    def __init__(
        self,
        multiplier: tuple[float, float] = (0.9, 1.1),
        elementwise: bool = False,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.multiplier = multiplier
        self.elementwise = elementwise

    @override
    def make_params(self, img: Tensor) -> MultiplierParam:
        r"""Sample per-call random parameters for :class:`MultiplicativeNoise`.

        Parameters
        ----------
        img : Tensor
            Image tensor; not inspected, carried through for dispatch.

        Returns
        -------
        MultiplierParam
            Carries the constructor's ``multiplier`` bounds verbatim
            (``lo``, ``hi``).  The actual scalar / per-pixel multiplier
            draw is deferred to the apply step so ``elementwise`` can
            switch between the two regimes without re-running the
            sampling head.
        """
        return MultiplierParam(lo=self.multiplier[0], hi=self.multiplier[1])

    @override
    def _apply_image(self, img: Tensor, params: MultiplierParam) -> Tensor:
        if self.elementwise:
            mult = (
                lucid.rand(*img.shape, device=img.device) * (params.hi - params.lo)
                + params.lo
            )
            return lucid.clip(img * mult, 0.0, 1.0)
        scalar = _random.uniform(params.lo, params.hi)
        return lucid.clip(img * scalar, 0.0, 1.0)

    @override
    def __repr__(self) -> str:
        return f"MultiplicativeNoise(multiplier={self.multiplier}, p={self.p})"


class ISONoise(PhotometricTransform[NoiseParam]):
    r"""Camera-sensor-like noise (Albumentations ``ISONoise``, approximate).

    Adds luminance Gaussian noise scaled by ``intensity`` plus a small
    saturation perturbation from ``color_shift``.

    Examples
    --------
    >>> import lucid, lucid.utils.transforms as T
    >>> tf = T.ISONoise(p=1.0)
    >>> tuple(tf(T.Image(lucid.rand(3, 32, 32))).data.shape)
    (3, 32, 32)
    """

    def __init__(
        self,
        color_shift: tuple[float, float] = (0.01, 0.05),
        intensity: tuple[float, float] = (0.1, 0.5),
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.color_shift = color_shift
        self.intensity = intensity

    @override
    def make_params(self, img: Tensor) -> NoiseParam:
        r"""Sample per-call random parameters for :class:`ISONoise`.

        Parameters
        ----------
        img : Tensor
            Image tensor; not inspected, carried through for dispatch.

        Returns
        -------
        NoiseParam
            Carries ``std`` — sensor-noise intensity sampled uniformly
            from ``intensity``.  The colour-shift term is re-sampled
            inside the apply step.
        """
        return NoiseParam(std=_random.uniform(self.intensity[0], self.intensity[1]))

    @override
    def _apply_image(self, img: Tensor, params: NoiseParam) -> Tensor:
        cs = _random.uniform(self.color_shift[0], self.color_shift[1])
        out = F.adjust_saturation(img, 1.0 + cs)
        noise = lucid.randn(*out.shape, device=out.device) * (params.std * 0.1)
        return lucid.clip(out + noise, 0.0, 1.0)

    @override
    def __repr__(self) -> str:
        return (
            f"ISONoise(color_shift={self.color_shift}, intensity={self.intensity}, "
            f"p={self.p})"
        )


class Downscale(PhotometricTransform[ScaleParam]):
    r"""Downscale then upscale to lose high-frequency detail (Albumentations ``Downscale``).

    Samples a scale factor uniformly from ``[scale_min, scale_max]``,
    resizes the image to that fraction of its original ``(H, W)`` via
    nearest-neighbour, then resizes back to the original size — a
    cheap way to simulate sensor / JPEG block-level degradation.

    Parameters
    ----------
    scale_min : float, optional, default=0.25
        Lower bound of the downscale factor.
    scale_max : float, optional, default=0.25
        Upper bound of the downscale factor (set equal to
        ``scale_min`` for a deterministic factor).
    p : float, optional, default=0.5
        Probability of applying the transform.

    Examples
    --------
    >>> import lucid, lucid.utils.transforms as T
    >>> tf = T.Downscale(p=1.0)
    >>> tuple(tf(T.Image(lucid.rand(3, 32, 32))).data.shape)
    (3, 32, 32)
    """

    def __init__(
        self,
        scale_min: float = 0.25,
        scale_max: float = 0.25,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.scale_min = scale_min
        self.scale_max = scale_max

    @override
    def make_params(self, img: Tensor) -> ScaleParam:
        r"""Sample per-call random parameters for :class:`Downscale`.

        Parameters
        ----------
        img : Tensor
            Image tensor; not inspected, carried through for dispatch.

        Returns
        -------
        ScaleParam
            Carries ``scale`` — the downscale factor sampled uniformly
            from ``[scale_min, scale_max]``.
        """
        return ScaleParam(scale=_random.uniform(self.scale_min, self.scale_max))

    @override
    def _apply_image(self, img: Tensor, params: ScaleParam) -> Tensor:
        h, w = F._spatial_hw(img)
        dh = max(int(round(h * params.scale)), 1)
        dw = max(int(round(w * params.scale)), 1)
        small = F.resize(img, (dh, dw), interpolation="nearest")
        return F.resize(small, (h, w), interpolation="nearest")

    @override
    def __repr__(self) -> str:
        return (
            f"Downscale(scale_min={self.scale_min}, scale_max={self.scale_max}, "
            f"p={self.p})"
        )


# ── B8: defocus / zoom blur ─────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class RadiusParam:
    r"""Per-call disk-kernel radius used by :class:`Defocus`.

    The :class:`Defocus` apply step builds a normalised disk kernel of
    radius ``radius`` (pixels inside the disk get weight
    ``1 / area``) and convolves it with the image to approximate the
    bokeh circle of an out-of-focus lens.

    Attributes
    ----------
    radius : int
        Disk radius in pixels; larger values produce more aggressive
        defocus.
    """

    radius: int


class Defocus(PhotometricTransform[RadiusParam]):
    r"""Disk-kernel (out-of-focus) blur (Albumentations ``Defocus``).

    Samples an integer radius ``r`` from ``radius`` (inclusive) and
    convolves with a normalised disk of radius ``r`` — pixels inside
    the disk get weight ``1 / area``, those outside get zero.
    Approximates the bokeh circle of an out-of-focus lens better than
    Gaussian blur.

    Parameters
    ----------
    radius : (int, int), optional, default=(3, 10)
        Inclusive sampling range for the disk radius in pixels.
        Larger radii produce more aggressive defocus.
    p : float, optional, default=0.5
        Probability of applying the transform.

    Examples
    --------
    >>> import lucid, lucid.utils.transforms as T
    >>> tf = T.Defocus(p=1.0)
    >>> tuple(tf(T.Image(lucid.rand(3, 32, 32))).data.shape)
    (3, 32, 32)
    """

    def __init__(self, radius: tuple[int, int] = (3, 10), p: float = 0.5) -> None:
        super().__init__(p=p)
        self.radius = radius

    @override
    def make_params(self, img: Tensor) -> RadiusParam:
        r"""Sample per-call random parameters for :class:`Defocus`.

        Parameters
        ----------
        img : Tensor
            Image tensor; not inspected, carried through for dispatch.

        Returns
        -------
        RadiusParam
            Carries ``radius`` — the disk-kernel radius in pixels
            drawn uniformly from the constructor's ``radius`` range.
        """
        return RadiusParam(radius=_random.randint(self.radius[0], self.radius[1] + 1))

    @override
    def _apply_image(self, img: Tensor, params: RadiusParam) -> Tensor:
        r = params.radius
        k = 2 * r + 1
        disk = [
            [1.0 if (i - r) ** 2 + (j - r) ** 2 <= r * r else 0.0 for j in range(k)]
            for i in range(k)
        ]
        total = sum(sum(row) for row in disk) or 1.0
        disk = [[v / total for v in row] for row in disk]
        return F.depthwise_conv2d(img, disk)

    @override
    def __repr__(self) -> str:
        return f"Defocus(radius={self.radius}, p={self.p})"


@dataclass(frozen=True, slots=True)
class ZoomParam:
    r"""Per-call zoom factor used by :class:`ZoomBlur`.

    The :class:`ZoomBlur` apply step averages 5 centre-zoomed copies
    of the image at zoom levels evenly spaced between ``1.0`` and
    ``factor``, simulating a fast dolly into the scene.

    Attributes
    ----------
    factor : float
        Final zoom factor; ``> 1`` means zoom in.
    """

    factor: float


class ZoomBlur(PhotometricTransform[ZoomParam]):
    r"""Radial zoom blur via averaged centre-zoomed copies (Albumentations ``ZoomBlur``).

    Samples a final zoom factor uniformly from ``[1.0, max_factor]``,
    builds 5 centre-crops at progressively-increasing zoom levels
    (each resized back to the input ``(H, W)`` via bilinear), and
    averages them with the original image — simulates a fast forward
    dolly into the scene.

    Parameters
    ----------
    max_factor : float, optional, default=1.31
        Upper bound on the final zoom factor (must be ``>= 1.0``).
    step_factor : (float, float), optional, default=(0.01, 0.03)
        Reserved for future per-step jitter; currently the 5 zoom
        levels are evenly spaced toward ``max_factor`` regardless.
    p : float, optional, default=0.5
        Probability of applying the transform.

    Examples
    --------
    >>> import lucid, lucid.utils.transforms as T
    >>> tf = T.ZoomBlur(p=1.0)
    >>> tuple(tf(T.Image(lucid.rand(3, 32, 32))).data.shape)
    (3, 32, 32)
    """

    def __init__(
        self,
        max_factor: float = 1.31,
        step_factor: tuple[float, float] = (0.01, 0.03),
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.max_factor = max_factor
        self.step_factor = step_factor

    @override
    def make_params(self, img: Tensor) -> ZoomParam:
        r"""Sample per-call random parameters for :class:`ZoomBlur`.

        Parameters
        ----------
        img : Tensor
            Image tensor; not inspected, carried through for dispatch.

        Returns
        -------
        ZoomParam
            Carries ``factor`` — the final zoom factor drawn uniformly
            from ``[1.0, max_factor]``.  The apply step interpolates
            5 intermediate zoom levels between ``1.0`` and ``factor``.
        """
        return ZoomParam(factor=_random.uniform(1.0, self.max_factor))

    @override
    def _apply_image(self, img: Tensor, params: ZoomParam) -> Tensor:
        h, w = F._spatial_hw(img)
        n = 5
        acc = img
        for i in range(1, n + 1):
            f = 1.0 + (params.factor - 1.0) * i / n
            ch, cw = max(int(round(h / f)), 1), max(int(round(w / f)), 1)
            top, left = (h - ch) // 2, (w - cw) // 2
            zoomed = F.resize(
                F.crop(img, top, left, ch, cw), (h, w), interpolation="bilinear"
            )
            acc = acc + zoomed
        return lucid.clip(acc / (n + 1), 0.0, 1.0)

    @override
    def __repr__(self) -> str:
        return f"ZoomBlur(max_factor={self.max_factor}, p={self.p})"
