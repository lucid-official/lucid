"""Stateless functional transforms.

Mirrors ``reference_vision.transforms.functional``: plain functions that take
a :class:`lucid.Tensor` image (``(C, H, W)`` or ``(B, C, H, W)``) and
return a transformed tensor.  The class-based transforms in
:mod:`lucid.utils.transforms` are thin wrappers over these.

All operations are Lucid-native (engine ops only) — no numpy / PIL.
"""

import math
from typing import Literal

import lucid
import lucid.nn.functional as F
from lucid._dtype import iinfo
from lucid._tensor import Tensor
from lucid._types import DeviceLike

_RESIZE_ALIGN_MODES = frozenset({"bilinear", "bicubic", "linear", "trilinear"})
#: The modes ``antialias`` applies to; the others ignore it, as in the reference.
_ANTIALIASED_MODES = frozenset({"bilinear", "bicubic"})
_COPYING_MODES = frozenset({"nearest", "nearest-exact"})


def _cat(tensors: list[Tensor], dim: int) -> Tensor:
    """``lucid.concat`` wrapper (absorbs the int32-typed ``dim`` stub)."""
    return lucid.concat(tensors, dim=dim)


def _spatial_hw(img: Tensor) -> tuple[int, int]:
    """Return ``(H, W)`` for a CHW or NCHW tensor."""
    if img.ndim not in (3, 4):
        raise ValueError(f"expected a (C, H, W) or (B, C, H, W) tensor, got ndim={img.ndim}")
    return int(img.shape[-2]), int(img.shape[-1])


def resize_target(h: int, w: int, size: int | tuple[int, int]) -> tuple[int, int]:
    """Compute :func:`resize`'s output ``(H, W)`` for input ``(h, w)``.

    Shared by :func:`resize` and the mask / bounding-box paths so they
    stay consistent (shorter-side rule for an int ``size``).

    Parameters
    ----------
    h : int
        Input image height in pixels.
    w : int
        Input image width in pixels.
    size : int or (int, int)
        If an ``int``, the **shorter** side is scaled to ``size`` with
        the aspect ratio preserved and the longer side truncated to an
        integer, as the reference does.  If ``(h, w)``, the output shape
        is returned verbatim.

    Returns
    -------
    (int, int)
        Output ``(height, width)`` after applying the resize rule.
    """
    if isinstance(size, int):
        # Truncated, not rounded: the reference sizes the longer side as
        # int(size * long / short), and rounding put an 876 x 1300 photo at
        # 380 columns where the reference has 379.
        if h <= w:
            return size, int(w * size / h)
        return int(h * size / w), size
    return int(size[0]), int(size[1])


def resize(
    img: Tensor,
    size: int | tuple[int, int],
    *,
    interpolation: str = "bilinear",
    antialias: bool = True,
) -> Tensor:
    r"""Resize an image.

    Parameters
    ----------
    img : Tensor
        Image tensor ``(C, H, W)`` or ``(B, C, H, W)``.
    size : int or (int, int)
        If an ``int``, the **shorter** side is scaled to ``size`` with
        the aspect ratio preserved (reference_vision ``Resize(int)``
        semantics).  If ``(h, w)``, the image is resized to exactly that.
    interpolation : str, optional, default="bilinear"
        Mode forwarded to :func:`lucid.nn.functional.interpolate`.
    antialias : bool, optional, default=True
        Low-pass filter before shrinking, as the reference does by default:
        each output pixel averages the inputs under a kernel stretched by
        the downscale factor instead of sampling a few of them, which on a
        photo moves every value.  Applies to ``"bilinear"`` and
        ``"bicubic"`` (see ``antialias`` on
        :func:`lucid.nn.functional.interpolate`); the other modes ignore
        it.

    Returns
    -------
    Tensor
        Resized image with the same rank as the input.
    """
    unbatched = img.ndim == 3
    x = img[None] if unbatched else img
    h, w = _spatial_hw(x)
    new_h, new_w = resize_target(h, w, size)

    mode = str(getattr(interpolation, "value", interpolation))
    # An integer image is filtered in float32 and brought back the way the
    # reference does it: rounded, clamped to the dtype's range, cast.
    # Nearest only copies samples, so it keeps the integers as they are.
    integer = not x.is_floating_point() and mode not in _COPYING_MODES
    out = F.interpolate(
        x.to(lucid.float32) if integer else x,
        size=(new_h, new_w),
        mode=mode,
        align_corners=False if mode in _RESIZE_ALIGN_MODES else None,
        antialias=antialias and mode in _ANTIALIASED_MODES,
    )
    if integer:
        info = iinfo(x.dtype)
        out = out.round().clamp(info.min, info.max).to(x.dtype)
    return out[0] if unbatched else out


def crop(img: Tensor, top: int, left: int, height: int, width: int) -> Tensor:
    """Crop a ``height`` × ``width`` window with top-left at ``(top, left)``.

    Pure slicing — no copy on the contiguous case.  The crop window is
    expressed in pixel coordinates relative to the top-left origin.

    Parameters
    ----------
    img : Tensor
        Image ``(C, H, W)`` or ``(B, C, H, W)``.
    top : int
        Row index of the crop's top edge (inclusive).
    left : int
        Column index of the crop's left edge (inclusive).
    height : int
        Number of rows in the crop.
    width : int
        Number of columns in the crop.

    Returns
    -------
    Tensor
        Cropped image with spatial shape ``(height, width)``.
    """
    return img[..., top : top + height, left : left + width]


def center_crop(
    img: Tensor, size: int | tuple[int, int], offset: Literal["round", "floor"] = "round"
) -> Tensor:
    r"""Crop a centered window of ``size`` (square if ``size`` is an int).

    The crop window is centered on the input; if the requested ``size``
    exceeds the input, the offsets clamp to ``0`` and the crop returns
    the available pixels (no padding).

    Parameters
    ----------
    img : Tensor
        Image ``(C, H, W)`` or ``(B, C, H, W)``.
    size : int or (int, int)
        Output ``(height, width)``.  An ``int`` selects a square window.
    offset : {"round", "floor"}, optional
        How to place the window when the margin is odd, which is the
        only case the two conventions disagree on.  ``"round"`` is the
        default and what the reference vision package does — flooring
        put a 379-wide image's 224 crop one column left of it.
        ``"floor"`` is what Hugging Face's fast processors do, and a
        checkpoint published through those expects it: a 519-wide frame
        cropped to 256 starts at column 131 there and 132 here, which on
        anything but a smooth image is a different picture.

    Returns
    -------
    Tensor
        Cropped image centered on the input.
    """
    crop_h, crop_w = (size, size) if isinstance(size, int) else (size[0], size[1])
    h, w = _spatial_hw(img)
    if offset == "floor":
        top, left = max((h - crop_h) // 2, 0), max((w - crop_w) // 2, 0)
    else:
        top = max(int(round((h - crop_h) / 2.0)), 0)
        left = max(int(round((w - crop_w) / 2.0)), 0)
    return crop(img, top, left, crop_h, crop_w)


def hflip(img: Tensor) -> Tensor:
    """Horizontally flip (mirror along the width axis); leaves channels untouched.

    Parameters
    ----------
    img : Tensor
        Image ``(C, H, W)`` or ``(B, C, H, W)``.

    Returns
    -------
    Tensor
        Image flipped left-right; same shape and dtype as ``img``.
    """
    # lucid.flip's stub types dims as int32; a plain int works at runtime.
    return lucid.flip(img, dims=-1)


def vflip(img: Tensor) -> Tensor:
    """Vertically flip (mirror along the height axis); leaves channels untouched.

    Parameters
    ----------
    img : Tensor
        Image ``(C, H, W)`` or ``(B, C, H, W)``.

    Returns
    -------
    Tensor
        Image flipped top-bottom; same shape and dtype as ``img``.
    """
    return lucid.flip(img, dims=-2)


def pad(
    img: Tensor,
    padding: int | tuple[int, int, int, int],
    *,
    mode: str = "constant",
    value: float = 0.0,
) -> Tensor:
    r"""Pad spatial borders.

    Parameters
    ----------
    img : Tensor
        Image ``(C, H, W)`` or ``(B, C, H, W)``.
    padding : int or (left, right, top, bottom)
        Border widths.  A single int pads all four sides equally.
    mode : str, optional, default="constant"
        Forwarded to :func:`lucid.nn.functional.pad`.
    value : float, optional, default=0.0
        Fill value for ``"constant"`` mode.
    """
    if isinstance(padding, int):
        pad_spec = (padding, padding, padding, padding)
    else:
        pad_spec = padding
    return F.pad(img, pad_spec, mode=mode, value=value if mode == "constant" else None)


def normalize(img: Tensor, mean: tuple[float, ...], std: tuple[float, ...]) -> Tensor:
    r"""Normalize per channel: ``(img - mean) / std``.

    ``mean`` / ``std`` are broadcast over the channel axis (length ``1`` or ``C``).

    Parameters
    ----------
    img : Tensor
        Image ``(C, H, W)`` or ``(B, C, H, W)`` in floating dtype.
    mean : tuple of float
        Per-channel mean (length ``1`` or ``C``).
    std : tuple of float
        Per-channel standard deviation (length ``1`` or ``C``).

    Returns
    -------
    Tensor
        Normalized image with the same shape and dtype as ``img``.
    """
    mean_t = lucid.tensor(list(mean), dtype=img.dtype, device=img.device).reshape(-1, 1, 1)
    std_t = lucid.tensor(list(std), dtype=img.dtype, device=img.device).reshape(-1, 1, 1)
    return (img - mean_t) / std_t


def rescale(img: Tensor, scale: float = 1.0 / 255.0) -> Tensor:
    """Multiply pixel values by ``scale`` (e.g. uint8 ``[0,255]`` → ``[0,1]``).

    Parameters
    ----------
    img : Tensor
        Image of any shape; usually integer-typed when used for
        normalisation into the unit interval.
    scale : float, optional, default=1/255
        Multiplicative factor applied element-wise.

    Returns
    -------
    Tensor
        ``img * scale`` with the same shape as ``img``.
    """
    return img * scale


def resized_crop(
    img: Tensor,
    top: int,
    left: int,
    height: int,
    width: int,
    size: int | tuple[int, int],
    *,
    interpolation: str = "bilinear",
    antialias: bool = True,
) -> Tensor:
    """Crop ``(top, left, height, width)`` then resize to ``size``.

    Composes :func:`crop` followed by :func:`resize`; this is the
    canonical building block for ``RandomResizedCrop``.

    Parameters
    ----------
    img : Tensor
        Image ``(C, H, W)`` or ``(B, C, H, W)``.
    top, left : int
        Top-left corner of the crop window.
    height, width : int
        Size of the crop window in pixels.
    size : int or (int, int)
        Final output size, with the same shorter-side semantics as
        :func:`resize`.
    interpolation : str, optional, default="bilinear"
        Interpolation mode forwarded to :func:`resize`.
    antialias : bool, optional, default=True
        Forwarded to :func:`resize`: low-pass filter when the crop is
        shrunk, as the reference does by default.

    Returns
    -------
    Tensor
        Cropped-then-resized image.
    """
    return resize(
        crop(img, top, left, height, width), size, interpolation=interpolation, antialias=antialias
    )


def rgb_to_grayscale(img: Tensor, *, keep_channels: bool = True) -> Tensor:
    r"""Convert an RGB image to luminance (ITU-R 601-2 weights).

    Parameters
    ----------
    img : Tensor
        RGB image with 3 channels.
    keep_channels : bool, optional, default=True
        If ``True`` the single luma channel is broadcast back to 3
        channels (handy for ``saturation`` blending); else a 1-channel
        result is returned.
    """
    c = int(img.shape[-3])
    if c != 3:
        raise ValueError(f"rgb_to_grayscale expects 3 channels, got {c}")
    r = img[..., 0:1, :, :]
    g = img[..., 1:2, :, :]
    b = img[..., 2:3, :, :]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    if keep_channels:
        return _cat([gray, gray, gray], -3)
    return gray


def _blend(img1: Tensor, img2: Tensor, ratio: float) -> Tensor:
    """``ratio * img1 + (1 - ratio) * img2``, clipped to ``[0, 1]``."""
    return lucid.clip(ratio * img1 + (1.0 - ratio) * img2, 0.0, 1.0)


def adjust_brightness(img: Tensor, factor: float) -> Tensor:
    """Scale brightness by blending toward black; ``factor=1`` is a no-op.

    Equivalent to PIL ``ImageEnhance.Brightness``: ``factor=0`` returns
    a fully black image, ``factor=1`` returns the input, ``factor>1``
    over-brightens (extrapolated away from black, then clipped).

    Parameters
    ----------
    img : Tensor
        Image ``(C, H, W)`` or ``(B, C, H, W)`` in ``[0, 1]``.
    factor : float
        Non-negative brightness multiplier.

    Returns
    -------
    Tensor
        Brightened image, same shape and dtype as ``img``, clipped to ``[0, 1]``.
    """
    return _blend(img, img * 0.0, factor)


def adjust_contrast(img: Tensor, factor: float) -> Tensor:
    """Adjust contrast by blending toward the per-image mean gray.

    Equivalent to PIL ``ImageEnhance.Contrast``: ``factor=0`` returns a
    flat gray (the input's average luminance), ``factor=1`` is the
    identity, ``factor>1`` boosts contrast.

    Parameters
    ----------
    img : Tensor
        RGB image ``(3, H, W)`` or ``(B, 3, H, W)`` in ``[0, 1]``.
    factor : float
        Non-negative contrast multiplier.

    Returns
    -------
    Tensor
        Contrast-adjusted image, same shape and dtype as ``img``,
        clipped to ``[0, 1]``.
    """
    gray = rgb_to_grayscale(img, keep_channels=False)
    mean = lucid.mean(gray, dim=[-1, -2, -3], keepdim=True)
    return _blend(img, mean, factor)


def adjust_saturation(img: Tensor, factor: float) -> Tensor:
    """Adjust saturation by blending toward grayscale; ``factor=1`` is a no-op.

    Equivalent to PIL ``ImageEnhance.Color``: ``factor=0`` returns a
    grayscale image broadcast back to 3 channels, ``factor>1`` boosts
    saturation (extrapolated away from gray, then clipped).

    Parameters
    ----------
    img : Tensor
        RGB image ``(3, H, W)`` or ``(B, 3, H, W)`` in ``[0, 1]``.
    factor : float
        Non-negative saturation multiplier.

    Returns
    -------
    Tensor
        Saturation-adjusted image, same shape and dtype as ``img``,
        clipped to ``[0, 1]``.
    """
    return _blend(img, rgb_to_grayscale(img, keep_channels=True), factor)


def _frac1(z: Tensor) -> Tensor:
    """``z mod 1.0`` via ``z - floor(z)`` (Tensor has no ``%`` operator)."""
    return z - lucid.floor(z)


def rgb_to_hsv(img: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    r"""Convert an RGB image (in ``[0, 1]``) to HSV channels.

    Implements the standard hexcone HSV definition: the hue piecewise
    formula on the channel that holds the maximum, normalised to
    ``[0, 1)``, with achromatic pixels (zero chroma) explicitly
    snapped to ``0``.  Pure-tensor implementation — no Python loop,
    no colour-space library.

    .. math::

        V = \max(R, G, B), \quad
        S = \frac{V - \min(R, G, B)}{V}, \quad
        H = \frac{1}{6} h_\text{hex}(R, G, B) \;\bmod\; 1

    Parameters
    ----------
    img : Tensor
        RGB image of shape ``(3, H, W)`` or ``(B, 3, H, W)``, values
        in ``[0, 1]``.

    Returns
    -------
    h : Tensor
        Hue channel of shape ``(…, 1, H, W)``, in ``[0, 1)``.
    s : Tensor
        Saturation channel of shape ``(…, 1, H, W)``, in ``[0, 1]``.
    v : Tensor
        Value channel of shape ``(…, 1, H, W)``, in ``[0, 1]``.

    Notes
    -----
    Round-trip with :func:`hsv_to_rgb` is exact to float32 epsilon
    (verified by the G2 cv2-accuracy test suite).  Achromatic pixels
    (``delta < 1e-10``) get ``h = 0`` to avoid the ``0/0`` form in the
    hue definition.

    Examples
    --------
    >>> import lucid
    >>> img = lucid.tensor([[[1.0]], [[0.0]], [[0.0]]])  # pure red
    >>> h, s, v = rgb_to_hsv(img)
    >>> float(h.item()), float(s.item()), float(v.item())
    (0.0, 1.0, 1.0)
    """
    r = img[..., 0:1, :, :]
    g = img[..., 1:2, :, :]
    b = img[..., 2:3, :, :]
    maxc = lucid.max(img, dim=-3, keepdim=True).values
    minc = lucid.min(img, dim=-3, keepdim=True).values
    delta = maxc - minc
    eps = 1e-10

    rc = (maxc - r) / (delta + eps)
    gc = (maxc - g) / (delta + eps)
    bc = (maxc - b) / (delta + eps)
    hue = lucid.where(maxc == r, bc - gc, lucid.where(maxc == g, 2.0 + rc - bc, 4.0 + gc - rc))
    hue = _frac1(hue / 6.0)
    hue = lucid.where(delta < eps, hue * 0.0, hue)

    s = delta / (maxc + eps)
    v = maxc
    return hue, s, v


def hsv_to_rgb(h: Tensor, s: Tensor, v: Tensor) -> Tensor:
    r"""Convert HSV channels back to an RGB image (inverse of :func:`rgb_to_hsv`).

    Standard hexcone reconstruction: hue is partitioned into 6
    sectors, each sector selects which two of ``(v, p, q, t)`` map to
    which RGB channel.  Output is clipped to ``[0, 1]``.

    Parameters
    ----------
    h : Tensor
        Hue channel of shape ``(…, 1, H, W)``, expected in ``[0, 1)``
        (values outside are wrapped via ``i_mod``).
    s : Tensor
        Saturation channel of shape ``(…, 1, H, W)``, in ``[0, 1]``.
    v : Tensor
        Value channel of shape ``(…, 1, H, W)``, in ``[0, 1]``.

    Returns
    -------
    Tensor
        RGB image of shape ``(…, 3, H, W)``, values in ``[0, 1]``.

    Examples
    --------
    >>> import lucid
    >>> h = lucid.tensor([[[0.0]]])      # red
    >>> s = lucid.tensor([[[1.0]]])
    >>> v = lucid.tensor([[[1.0]]])
    >>> hsv_to_rgb(h, s, v).numpy().reshape(-1).tolist()
    [1.0, 0.0, 0.0]
    """
    i = lucid.floor(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i_mod = i - 6.0 * lucid.floor(i / 6.0)

    def _sel(c0: Tensor, c1: Tensor, c2: Tensor, c3: Tensor, c4: Tensor, c5: Tensor) -> Tensor:
        out = lucid.where(i_mod == 0, c0, c1)
        out = lucid.where(i_mod == 2, c2, out)
        out = lucid.where(i_mod == 3, c3, out)
        out = lucid.where(i_mod == 4, c4, out)
        out = lucid.where(i_mod == 5, c5, out)
        out = lucid.where(i_mod == 1, c1, out)
        return out

    new_r = _sel(v, q, p, p, t, v)
    new_g = _sel(t, v, v, q, p, p)
    new_b = _sel(p, p, t, v, v, q)
    return lucid.clip(_cat([new_r, new_g, new_b], -3), 0.0, 1.0)


def adjust_hue(img: Tensor, factor: float) -> Tensor:
    r"""Shift hue by ``factor`` (in ``[-0.5, 0.5]``) via an exact HSV round-trip.

    Round-trips through :func:`rgb_to_hsv` / :func:`hsv_to_rgb` so the
    transform is differentiable.  Hue wraps modulo 1 — ``factor=0.5``
    and ``factor=-0.5`` are equivalent (the chromatic antipode).

    Parameters
    ----------
    img : Tensor
        RGB image ``(3, H, W)`` or ``(B, 3, H, W)`` in ``[0, 1]``.
    factor : float
        Hue offset in ``[-0.5, 0.5]`` (fraction of a full rotation).

    Returns
    -------
    Tensor
        Hue-shifted image, same shape and dtype as ``img``.
    """
    if factor == 0.0:
        return img
    h, s, v = rgb_to_hsv(img)
    return hsv_to_rgb(_frac1(h + factor), s, v)


def adjust_hsv(img: Tensor, hue_shift: float, sat_shift: float, val_shift: float) -> Tensor:
    r"""Additive HSV shift (cv2 / Albumentations ``HueSaturationValue`` semantics).

    Round-trips through HSV via :func:`rgb_to_hsv` /
    :func:`hsv_to_rgb`.  Inputs use the OpenCV scales so values
    transfer between Lucid pipelines and any cv2 / Albumentations
    reference implementation unchanged — internally each shift is
    rescaled to the ``[0, 1]`` HSV representation, hue wraps mod 1
    and saturation / value clip to ``[0, 1]``.

    Parameters
    ----------
    img : Tensor
        RGB image of shape ``(3, H, W)`` or ``(B, 3, H, W)`` in ``[0, 1]``.
    hue_shift : float
        Hue offset on the OpenCV scale ``[0, 179]`` (``180`` ≡ a full
        revolution).  Divided by ``180`` and added to the normalised
        hue.
    sat_shift : float
        Saturation offset on the OpenCV scale ``[0, 255]``.  Divided
        by ``255`` and added to the normalised saturation, then
        clipped to ``[0, 1]``.
    val_shift : float
        Value (brightness) offset on the OpenCV scale ``[0, 255]``,
        same scaling as ``sat_shift``.

    Returns
    -------
    Tensor
        Shifted RGB image with the same shape and dtype as ``img``.

    Notes
    -----
    Lucid keeps the entire round-trip in float, while Albumentations
    quantises through ``uint8`` + cv2 HSV — Lucid's path is *more*
    precise (diff vs Albu ≈ 0.02 max, dominated by Albu's 1/255
    quantisation error).  Tracked in the G3 parity suite under the
    "ballpark" tier.

    Examples
    --------
    >>> import lucid
    >>> img = lucid.rand(3, 8, 8)
    >>> out = adjust_hsv(img, hue_shift=20.0, sat_shift=30.0, val_shift=10.0)
    >>> tuple(out.shape)
    (3, 8, 8)
    """
    h, s, v = rgb_to_hsv(img)
    h = _frac1(h + hue_shift / 180.0)
    s = lucid.clip(s + sat_shift / 255.0, 0.0, 1.0)
    v = lucid.clip(v + val_shift / 255.0, 0.0, 1.0)
    return hsv_to_rgb(h, s, v)


# ── affine / perspective warps ──────────────────────────────────────
#
# Coordinate convention matches OpenCV (so Albumentations-style matrices
# transfer directly): a forward 3x3 matrix ``M`` maps an *input* pixel
# ``[x, y, 1]`` to an *output* pixel.  Images are sampled by inverting M
# into ``affine_grid``'s normalized output→input ``theta``; boxes and
# keypoints apply M to their coordinates directly.


def _norm_to_pixel(h: int, w: int, device: DeviceLike = None) -> Tensor:
    """3x3 matrix mapping ``align_corners=True`` normalized coords → pixels.

    With ``align_corners=True`` the corner pixels map to ``-1`` / ``+1``:
    ``px = (nx + 1) * (W - 1) / 2``.  (Lucid's ``align_corners=False``
    grid path does not reproduce an identity ``theta``, so the warp path
    standardises on ``align_corners=True``.)
    """
    return lucid.tensor(
        [[(w - 1) / 2.0, 0.0, (w - 1) / 2.0], [0.0, (h - 1) / 2.0, (h - 1) / 2.0], [0.0, 0.0, 1.0]],
        device=device,
    )


def affine_theta(matrix: Tensor, in_hw: tuple[int, int], out_hw: tuple[int, int]) -> Tensor:
    """Convert a forward pixel matrix ``M`` to ``affine_grid`` ``theta`` (1,2,3).

    Composes the normalisation transforms on either side of ``M`` so
    the result can be fed directly to :func:`lucid.nn.functional.affine_grid`
    (which uses ``align_corners=True`` normalised coordinates).

    Parameters
    ----------
    matrix : Tensor
        Forward 3x3 pixel matrix mapping input → output pixels.
    in_hw : (int, int)
        Input image ``(height, width)``.
    out_hw : (int, int)
        Output image ``(height, width)``.

    Returns
    -------
    Tensor
        Theta tensor of shape ``(1, 2, 3)`` suitable for ``affine_grid``.
    """
    ih, iw = in_hw
    oh, ow = out_hw
    no = _norm_to_pixel(oh, ow, matrix.device)
    ni_inv = lucid.linalg.inv(_norm_to_pixel(ih, iw, matrix.device))
    m_inv = lucid.linalg.inv(matrix)
    theta = lucid.matmul(lucid.matmul(ni_inv, m_inv), no)  # norm_out → norm_in
    return theta[:2].reshape(1, 2, 3)


def warp_affine(
    img: Tensor,
    matrix: Tensor,
    out_hw: tuple[int, int],
    *,
    mode: str = "bilinear",
    fill: float = 0.0,
) -> Tensor:
    """Warp ``img`` by forward pixel matrix ``matrix`` (affine or homography).

    Backward-warps ``img`` via :func:`affine_theta` + ``affine_grid`` +
    ``grid_sample``.  The matrix may be a full 3x3 homography — the
    perspective divide is handled by ``affine_grid`` implicitly when
    the bottom row is non-degenerate.

    Parameters
    ----------
    img : Tensor
        Image ``(C, H, W)`` or ``(B, C, H, W)``.
    matrix : Tensor
        Forward 3x3 pixel matrix mapping input → output coordinates.
    out_hw : (int, int)
        Output ``(height, width)``.
    mode : str, optional, default="bilinear"
        Sampling mode; ``"nearest"`` falls back to nearest-neighbour.
    fill : float, optional, default=0.0
        Border fill value; ``0.0`` uses zero padding, any other value
        switches to border (edge-replicate) sampling.

    Returns
    -------
    Tensor
        Warped image with spatial shape ``out_hw``.  An image with no batch
        entries or no channels warps to an empty one of that shape.
    """
    if 0 in img.shape[:-2]:
        return lucid.zeros(*img.shape[:-2], *out_hw, dtype=img.dtype, device=img.device)
    unbatched = img.ndim == 3
    x = img[None] if unbatched else img
    b, c, ih, iw = (int(d) for d in x.shape)
    theta = affine_theta(matrix, (ih, iw), out_hw)
    if b > 1:
        theta = _cat([theta] * b, 0)
    grid = F.affine_grid(theta, [b, c, out_hw[0], out_hw[1]], align_corners=True)
    gmode = "nearest" if mode == "nearest" else "bilinear"
    pad = "zeros" if fill == 0.0 else "border"
    y = F.grid_sample(x, grid, mode=gmode, padding_mode=pad, align_corners=True)
    return y[0] if unbatched else y


def affine_points(pts: Tensor, matrix: Tensor) -> Tensor:
    """Apply a forward 3x3 ``matrix`` to ``(N, 2)`` ``(x, y)`` points.

    Promotes the points to homogeneous ``(x, y, 1)`` rows, multiplies by
    ``matrix^T``, and divides by the homogeneous coordinate — so this
    works for both affine and full-homography matrices.

    Parameters
    ----------
    pts : Tensor
        Point set of shape ``(N, 2)`` in pixel coordinates (``x``, ``y``).
    matrix : Tensor
        Forward 3x3 transform matrix.

    Returns
    -------
    Tensor
        Transformed points of shape ``(N, 2)``.
    """
    n = int(pts.shape[0])
    ones = lucid.ones(n, 1, dtype=pts.dtype, device=pts.device)
    hom = _cat([pts, ones], 1)  # (N, 3)
    out = lucid.matmul(hom, lucid.swapaxes(matrix, 0, 1))  # (N, 3)
    return out[:, :2] / out[:, 2:3]  # perspective divide (no-op for affine)


def rotation_matrix(
    angle_deg: float, cx: float, cy: float, scale: float = 1.0, device: DeviceLike = None
) -> Tensor:
    """OpenCV ``getRotationMatrix2D`` forward matrix (CCW degrees) about ``(cx, cy)``.

    Matches cv2's pixel coordinate convention so matrices transfer
    directly between Lucid and OpenCV / Albumentations pipelines.

    Parameters
    ----------
    angle_deg : float
        Rotation angle in degrees, counter-clockwise positive.
    cx : float
        x-coordinate of the rotation center (pixel).
    cy : float
        y-coordinate of the rotation center (pixel).
    scale : float, optional, default=1.0
        Uniform scale factor applied alongside the rotation.
    device : DeviceLike, optional
        Placement of the returned matrix.  ``None`` uses the current
        default device.

    Returns
    -------
    Tensor
        Forward 3x3 matrix mapping input pixel → output pixel.
    """
    rad = math.radians(angle_deg)
    alpha = scale * math.cos(rad)
    beta = scale * math.sin(rad)
    return lucid.tensor(
        [
            [alpha, beta, (1.0 - alpha) * cx - beta * cy],
            [-beta, alpha, beta * cx + (1.0 - alpha) * cy],
            [0.0, 0.0, 1.0],
        ],
        device=device,
    )


def affine_matrix(
    *,
    cx: float,
    cy: float,
    scale: float = 1.0,
    angle_deg: float = 0.0,
    shear_x_deg: float = 0.0,
    shear_y_deg: float = 0.0,
    translate_x: float = 0.0,
    translate_y: float = 0.0,
    device: DeviceLike = None,
) -> Tensor:
    """Compose a forward affine matrix about ``(cx, cy)`` then translate.

    Composition order is ``rotate ∘ shear ∘ scale`` (about the origin),
    followed by a re-centering so ``(cx, cy)`` stays fixed and then the
    explicit translation.  This matches reference_vision / Albumentations
    ``Affine`` semantics so matrices transfer directly.

    Parameters
    ----------
    cx, cy : float
        Pivot point in pixel coordinates.
    scale : float, optional, default=1.0
        Uniform scale factor.
    angle_deg : float, optional, default=0.0
        Rotation angle in degrees (CCW positive).
    shear_x_deg, shear_y_deg : float, optional, default=0.0
        Shear angles in degrees along each axis.
    translate_x, translate_y : float, optional, default=0.0
        Post-transform translation in pixels.
    device : DeviceLike, optional
        Placement of the returned matrix.  ``None`` uses the current
        default device.

    Returns
    -------
    Tensor
        Forward 3x3 pixel matrix.
    """
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    shx, shy = math.tan(math.radians(shear_x_deg)), math.tan(math.radians(shear_y_deg))
    # Rotate ∘ shear ∘ scale (about origin).
    a = scale * (cos_a - sin_a * shy)
    bb = scale * (cos_a * shx - sin_a)
    cc = scale * (sin_a + cos_a * shy)
    dd = scale * (sin_a * shx + cos_a)
    # Translate so the center maps to itself, then add the requested shift.
    e = cx - a * cx - bb * cy + translate_x
    f = cy - cc * cx - dd * cy + translate_y
    return lucid.tensor([[a, bb, e], [cc, dd, f], [0.0, 0.0, 1.0]], device=device)


def perspective_matrix(
    src: list[list[float]], dst: list[list[float]], device: DeviceLike = None
) -> Tensor:
    """Forward homography mapping the 4 ``src`` corners to ``dst`` (xy each).

    Solves the 8-DoF system for the 3x3 matrix (``h33 = 1``).

    Parameters
    ----------
    src : list of [float, float]
        Four source ``(x, y)`` corner points (pixel coords).
    dst : list of [float, float]
        Four destination ``(x, y)`` corner points; same order as ``src``.
    device : DeviceLike, optional
        Placement of the returned matrix.  ``None`` uses the current
        default device.

    Returns
    -------
    Tensor
        Forward 3x3 homography mapping ``src[i] → dst[i]`` exactly.
    """
    if len(src) != 4 or len(dst) != 4:
        raise ValueError(
            f"perspective_matrix: expected 4 (x, y) corners in src and in dst, got "
            f"{len(src)} and {len(dst)}"
        )
    rows: list[list[float]] = []
    rhs: list[float] = []
    for (sx, sy), (dx, dy) in zip(src, dst, strict=True):
        rows.append([sx, sy, 1.0, 0.0, 0.0, 0.0, -dx * sx, -dx * sy])
        rhs.append(dx)
        rows.append([0.0, 0.0, 0.0, sx, sy, 1.0, -dy * sx, -dy * sy])
        rhs.append(dy)
    a: Tensor = lucid.tensor(rows, device=device)
    b: Tensor = lucid.tensor([[v] for v in rhs], device=device)
    h = lucid.matmul(lucid.linalg.inv(a), b).reshape(-1)
    hl = h.tolist()
    return lucid.tensor(
        [[hl[0], hl[1], hl[2]], [hl[3], hl[4], hl[5]], [hl[6], hl[7], 1.0]], device=device
    )


# ── blur + displacement-field warps ─────────────────────────────────


def gaussian_blur(img: Tensor, sigma: float, ksize: int | None = None) -> Tensor:
    """Separable Gaussian blur (depthwise conv). ``sigma`` in pixels.

    Builds a 1-D Gaussian of width ``ksize`` (auto-derived as
    ``2 * round(3 * sigma) + 1`` when not supplied) and applies it twice
    — once along width, once along height — so the cost is ``O(C · k · H · W)``
    rather than ``O(C · k^2 · H · W)``.

    Parameters
    ----------
    img : Tensor
        Image ``(C, H, W)`` or ``(B, C, H, W)``.
    sigma : float
        Standard deviation of the Gaussian, in pixels.
    ksize : int, optional
        Odd kernel size; default ``2 * round(3 * sigma) + 1``.  Even
        values are rounded up to the next odd integer.

    Returns
    -------
    Tensor
        Blurred image, same shape and dtype as ``img``.
    """
    if ksize is None:
        ksize = int(2 * round(3.0 * sigma) + 1)
    if ksize % 2 == 0:
        ksize += 1
    half = ksize // 2
    w1 = [math.exp(-((i - half) ** 2) / (2.0 * sigma * sigma)) for i in range(ksize)]
    s = sum(w1)
    w1 = [v / s for v in w1]

    unbatched = img.ndim == 3
    x = img[None] if unbatched else img
    c = int(x.shape[1])
    kx = _cat([lucid.tensor(w1, device=x.device).reshape(1, 1, 1, ksize)] * c, 0)
    ky = _cat([lucid.tensor(w1, device=x.device).reshape(1, 1, ksize, 1)] * c, 0)
    x = F.conv2d(x, kx, padding=(0, half), groups=c)
    x = F.conv2d(x, ky, padding=(half, 0), groups=c)
    return x[0] if unbatched else x


def _pixel_grid(h: int, w: int, device: DeviceLike = None) -> tuple[Tensor, Tensor]:
    """Return ``(yy, xx)`` float pixel-coordinate grids of shape ``(H, W)``."""
    yy, xx = lucid.meshgrid(
        lucid.arange(0, h, dtype=lucid.float32, device=device),
        lucid.arange(0, w, dtype=lucid.float32, device=device),
        indexing="ij",
    )
    return yy, xx


def remap(img: Tensor, dx: Tensor, dy: Tensor, *, mode: str = "bilinear") -> Tensor:
    """Backward warp: ``out(y, x) = img(x + dx, y + dy)`` via ``grid_sample``.

    Equivalent to OpenCV ``cv2.remap`` with a displacement field; the
    sampling grid is built from the pixel index plus the displacement,
    then normalised to the ``[-1, 1]`` range ``grid_sample`` expects.

    Parameters
    ----------
    img : Tensor
        Image ``(C, H, W)`` or ``(B, C, H, W)``.
    dx, dy : Tensor
        Displacement fields of shape ``(H, W)`` (in pixels) added to the
        x / y sampling coordinates.
    mode : str, optional, default="bilinear"
        Sampling mode; ``"nearest"`` falls back to nearest-neighbour.

    Returns
    -------
    Tensor
        Warped image with the same shape and dtype as ``img``.  Border
        pixels sample with reflection padding.
    """
    unbatched = img.ndim == 3
    x = img[None] if unbatched else img
    b, _, h, w = (int(d) for d in x.shape)
    yy, xx = _pixel_grid(h, w, x.device)
    gx = 2.0 * (xx + dx) / (w - 1) - 1.0
    gy = 2.0 * (yy + dy) / (h - 1) - 1.0
    grid = lucid.stack([gx, gy], dim=-1)[None]  # (1, H, W, 2)
    if b > 1:
        grid = _cat([grid] * b, 0)
    gmode = "nearest" if mode == "nearest" else "bilinear"
    y = F.grid_sample(x, grid, mode=gmode, padding_mode="reflection", align_corners=True)
    return y[0] if unbatched else y


def sample_field_at_points(field: Tensor, pts: Tensor, canvas_hw: tuple[int, int]) -> Tensor:
    """Bilinearly sample a 2-D scalar field at arbitrary continuous points.

    Wraps :func:`grid_sample` for the common case of looking up
    displacement / flow / mask values at non-integer keypoint
    coordinates.  The points are converted from pixel space to the
    ``[-1, 1]`` normalised grid expected by :func:`grid_sample` using
    the same ``align_corners=True`` convention adopted across
    ``lucid.utils.transforms``.

    Parameters
    ----------
    field : Tensor of shape (H, W)
        Scalar field to sample.
    pts : Tensor of shape (N, 2)
        Pixel-space ``(x, y)`` query points; ``x`` along width,
        ``y`` along height.
    canvas_hw : (int, int)
        ``(H, W)`` of ``field`` — passed in explicitly to avoid an
        extra shape inspection inside the warp dispatch.

    Returns
    -------
    Tensor of shape (N, 1)
        Bilinearly-interpolated field values at the query points.
    """
    h, w = canvas_hw
    n = int(pts.shape[0])
    gx = 2.0 * pts[:, 0:1] / (w - 1) - 1.0
    gy = 2.0 * pts[:, 1:2] / (h - 1) - 1.0
    grid = _cat([gx, gy], 1).reshape(1, n, 1, 2)
    samp = F.grid_sample(field.reshape(1, 1, h, w), grid, mode="bilinear", align_corners=True)
    return samp.reshape(n, 1)


# ── histogram + kernel helpers (colour batch) ───────────────────────


def _equalize_channel(ch: Tensor, clip_limit: float | None = None) -> Tensor:
    """Histogram-equalize a single ``(H, W)`` channel in ``[0, 1]``."""
    h, w = int(ch.shape[0]), int(ch.shape[1])
    idx = lucid.clip(lucid.round(ch * 255.0), 0.0, 255.0).long().reshape(-1)
    hist = lucid.bincount(idx, minlength=256).to(lucid.float32)
    if clip_limit is not None:
        cap = clip_limit * (h * w) / 256.0
        clipped = lucid.clip(hist, 0.0, cap)
        excess = (hist - clipped).sum() / 256.0
        hist = clipped + excess
    cdf = lucid.cumsum(hist)
    big = lucid.ones(256, dtype=lucid.float32, device=ch.device) * float(h * w + 1)
    cdf_min = lucid.min(lucid.where(cdf > 0.0, cdf, big))
    total = cdf[255]
    denom = float(total.item()) - float(cdf_min.item())
    if denom <= 0.0:  # flat channel — leave unchanged
        return ch
    lut = (cdf - cdf_min) / denom * 255.0
    return lucid.take(lut, idx).reshape(h, w) / 255.0


def equalize(img: Tensor, clip_limit: float | None = None) -> Tensor:
    """Per-channel histogram equalization of a ``[0, 1]`` image.

    Computes a 256-bin histogram for each channel, builds the CDF, and
    remaps pixel intensities so the resulting histogram is roughly
    uniform.  Optional contrast limiting clips bins above
    ``clip_limit * pixels / 256`` and redistributes the excess.

    Parameters
    ----------
    img : Tensor
        Image ``(C, H, W)`` or ``(B, C, H, W)`` in ``[0, 1]``.
    clip_limit : float, optional
        Contrast-clip threshold; when ``None`` the standard non-clipped
        equalize is performed.

    Returns
    -------
    Tensor
        Equalized image, same shape and dtype as ``img``.
    """
    unbatched = img.ndim == 3
    x = img[None] if unbatched else img
    b, c, h, w = (int(d) for d in x.shape)
    imgs = []
    for bi in range(b):
        chans = [_equalize_channel(x[bi, ci], clip_limit).reshape(1, h, w) for ci in range(c)]
        imgs.append(_cat(chans, 0).reshape(1, c, h, w))
    out = _cat(imgs, 0)
    return out[0] if unbatched else out


def _clahe_lut(tile: Tensor, clip_limit: float) -> Tensor:
    r"""Build a 256-entry CLAHE mapping LUT (values in ``[0, 1]``) for one tile.

    Clips the histogram at ``clip_limit * area / 256`` and
    redistributes the clipped excess uniformly across all 256 bins
    (cv2 ``createCLAHE`` contrast-limiting), then accumulates into a
    CDF and rescales onto ``[0, 255]`` with the standard
    ``scale = 255 / area`` mapping.

    Parameters
    ----------
    tile : Tensor
        A 2-D ``(h, w)`` slice of one channel, values in ``[0, 1]``.
    clip_limit : float
        Contrast clip threshold; ``0`` (or negative) disables
        clipping and behaves as plain histogram equalization.

    Returns
    -------
    Tensor
        ``(1, 256)`` LUT where ``lut[v / 255]`` is the post-equalize
        normalised intensity for source ``v``.
    """
    n = int(tile.shape[0]) * int(tile.shape[1])
    idx = lucid.clip(lucid.round(tile * 255.0), 0.0, 255.0).long().reshape(-1)
    hist = lucid.bincount(idx, minlength=256).to(lucid.float32)
    if clip_limit > 0.0:
        cap = max(1.0, clip_limit * float(n) / 256.0)
        clipped = lucid.clip(hist, 0.0, cap)
        excess = float((hist - clipped).sum().item())
        hist = clipped + excess / 256.0
    cdf = lucid.cumsum(hist)
    lut = lucid.clip(cdf * (255.0 / float(n)), 0.0, 255.0) / 255.0
    return lut.reshape(1, 256)


def _clahe_channel(ch: Tensor, clip_limit: float, grid_h: int, grid_w: int) -> Tensor:
    r"""Contrast-limited adaptive equalization of a single ``(H, W)`` channel.

    Computes a clipped-histogram LUT per tile (:func:`_clahe_lut`),
    then bilinearly interpolates each pixel's mapped value between
    the four nearest tile centres.  Bilinear inter-tile blending is
    the defining step that separates CLAHE from a *global*
    clip-limited equalize (the latter produces visible tile
    boundaries; CLAHE does not).

    Parameters
    ----------
    ch : Tensor
        Single channel of shape ``(H, W)``, values in ``[0, 1]``.
    clip_limit : float
        Contrast clip threshold passed straight to :func:`_clahe_lut`.
    grid_h, grid_w : int
        Number of tile rows / columns the channel is divided into.

    Returns
    -------
    Tensor
        Equalized channel of shape ``(H, W)``, values in ``[0, 1]``.

    Notes
    -----
    The vectorised gather (``lucid.take`` over a flattened
    ``(grid_h * grid_w, 256)`` LUT bank) avoids per-pixel Python
    loops, but the per-tile LUT construction still iterates the grid
    in Python — this is the bottleneck behind the G4f benchmark
    finding that CLAHE is ~30× slower than cv2.
    """
    h, w = int(ch.shape[0]), int(ch.shape[1])
    # No more tiles than pixels per side: a grid finer than the image cut
    # tiles with no pixels in them, and their histogram divided by a pixel
    # count of zero (a 1 x 1 image raised ZeroDivisionError).
    grid_h, grid_w = min(grid_h, h), min(grid_w, w)
    luts = []
    for ty in range(grid_h):
        r0, r1 = ty * h // grid_h, (ty + 1) * h // grid_h
        for tx in range(grid_w):
            c0, c1 = tx * w // grid_w, (tx + 1) * w // grid_w
            luts.append(_clahe_lut(ch[r0:r1, c0:c1], clip_limit))
    lut_flat = _cat(luts, 0).reshape(-1)  # (grid_h * grid_w * 256,)

    v_idx = lucid.clip(lucid.round(ch * 255.0), 0.0, 255.0).long()  # (H, W)

    rows = lucid.arange(0, h, device=ch.device).reshape(h, 1)
    cols = lucid.arange(0, w, device=ch.device).reshape(1, w)
    fy = lucid.clip((rows + 0.5) * (grid_h / h) - 0.5, 0.0, float(grid_h - 1))
    fx = lucid.clip((cols + 0.5) * (grid_w / w) - 0.5, 0.0, float(grid_w - 1))
    ty0f, tx0f = lucid.floor(fy), lucid.floor(fx)
    wy, wx = fy - ty0f, fx - tx0f  # (H,1), (1,W)
    ty0 = ty0f.long()
    tx0 = tx0f.long()
    ty1 = lucid.clip(ty0f + 1.0, 0.0, float(grid_h - 1)).long()
    tx1 = lucid.clip(tx0f + 1.0, 0.0, float(grid_w - 1)).long()

    def _gather(ty: Tensor, tx: Tensor) -> Tensor:
        idx = (ty * grid_w + tx) * 256 + v_idx  # (H, W) via broadcast
        return lucid.take(lut_flat, idx.reshape(-1)).reshape(h, w)

    g00, g01 = _gather(ty0, tx0), _gather(ty0, tx1)
    g10, g11 = _gather(ty1, tx0), _gather(ty1, tx1)
    top = g00 * (1.0 - wx) + g01 * wx
    bot = g10 * (1.0 - wx) + g11 * wx
    return top * (1.0 - wy) + bot * wy


def clahe(img: Tensor, clip_limit: float = 4.0, tile_grid_size: tuple[int, int] = (8, 8)) -> Tensor:
    r"""Contrast-limited adaptive histogram equalization (cv2 ``CLAHE``).

    Per-tile clipped-histogram LUT plus 4-neighbour bilinear
    interpolation between tile centres; see :func:`_clahe_channel`
    for the algorithmic detail.  Multi-channel routing mirrors
    Albumentations' ``CLAHE`` (which goes through ``cv2.cvtColor``
    LAB), but stays in tensor space:

    * **single-channel** (``C == 1``) — equalize directly,
    * **RGB** (``C == 3``) — convert to HSV via :func:`rgb_to_hsv`,
      equalize the value channel, convert back via :func:`hsv_to_rgb`
      so hue and saturation are preserved.

    Parameters
    ----------
    img : Tensor
        Image of shape ``(C, H, W)`` or ``(B, C, H, W)``; supports
        ``C in {1, 3}``.  Values in ``[0, 1]``.
    clip_limit : float, optional, default=4.0
        Contrast-clip threshold passed to :func:`_clahe_lut`
        (histogram cap per tile = ``clip_limit * area / 256``).
    tile_grid_size : (int, int), optional, default=(8, 8)
        Number of tiles ``(rows, cols)`` the channel is divided into.

    Returns
    -------
    Tensor
        Equalized image with the same shape and dtype as ``img``.

    Notes
    -----
    Approximate cv2 / Albumentations parity (the colour-space pivot
    differs — HSV value vs LAB L — but both preserve hue /
    saturation by operating on luminance only).  The G4f benchmark
    records this at ~30× slower than Albu on a 224² RGB input; the
    pure-tensor per-tile path is the bottleneck.

    Examples
    --------
    >>> import lucid
    >>> import lucid.utils.transforms.functional as F
    >>> img = lucid.rand(3, 64, 64)
    >>> out = F.clahe(img, clip_limit=4.0, tile_grid_size=(8, 8))
    >>> tuple(out.shape)
    (3, 64, 64)
    """
    grid_h, grid_w = tile_grid_size
    # Nothing to equalize: an image with no pixels (or no channels) comes back
    # as it is.  It reached the colour conversion, whose channel ``max`` has
    # no answer for zero channels.
    if img.numel() == 0:
        return img.clone()
    unbatched = img.ndim == 3
    x = img[None] if unbatched else img
    b, c = int(x.shape[0]), int(x.shape[1])

    out_batch = []
    for bi in range(b):
        if c == 1:
            eq = _clahe_channel(x[bi, 0], clip_limit, grid_h, grid_w)
            out_batch.append(eq.reshape(1, 1, *eq.shape))
        else:
            hh, ss, vv = rgb_to_hsv(x[bi : bi + 1])  # each (1,1,H,W)
            h2, w2 = int(vv.shape[-2]), int(vv.shape[-1])
            v_eq = _clahe_channel(vv.reshape(h2, w2), clip_limit, grid_h, grid_w)
            rgb = hsv_to_rgb(hh, ss, v_eq.reshape(1, 1, h2, w2))
            out_batch.append(rgb)
    out = _cat(out_batch, 0)
    return out[0] if unbatched else out


def depthwise_conv2d(img: Tensor, kernel2d: list[list[float]]) -> Tensor:
    """Apply a 2-D kernel to every channel independently (same padding).

    Replicates the kernel ``C`` times along the output-channel axis and
    runs a grouped conv2d so each input channel is filtered by its own
    copy of ``kernel2d``.  Padding is set so the spatial shape is
    preserved (``same`` padding).

    Parameters
    ----------
    img : Tensor
        Image ``(C, H, W)`` or ``(B, C, H, W)``.
    kernel2d : list of list of float
        2-D kernel of shape ``(kh, kw)``; applied identically to each
        channel.

    Returns
    -------
    Tensor
        Filtered image, same shape and dtype as ``img``.
    """
    kh, kw = len(kernel2d), len(kernel2d[0])
    unbatched = img.ndim == 3
    x = img[None] if unbatched else img
    c = int(x.shape[1])
    k = lucid.tensor(kernel2d, device=x.device).reshape(1, 1, kh, kw)
    weight = _cat([k] * c, 0)
    y = F.conv2d(x, weight, padding=(kh // 2, kw // 2), groups=c)
    return y[0] if unbatched else y


# ── AutoAugment family — single-image functional ops ────────────────


def adjust_sharpness(img: Tensor, factor: float) -> Tensor:
    r"""Adjust sharpness by blending with a smoothed copy.

    Reference-framework compatible smoothing kernel (PIL ``ImageFilter.SMOOTH``)::

        [[1, 1, 1],
         [1, 5, 1],
         [1, 1, 1]] / 13

    ``factor == 0`` returns the blurred image, ``factor == 1`` is the
    identity, ``factor > 1`` sharpens (extrapolates away from blur).

    PIL convention is honoured for border pixels: the SMOOTH kernel
    has no valid neighbourhood on a 1-pixel border so those pixels
    pass through unchanged (preventing the zero-padded ``conv2d``
    bleed into the result that would otherwise diverge from
    reference-framework parity).

    Parameters
    ----------
    img : Tensor
        Image ``(C, H, W)`` or ``(B, C, H, W)`` in ``[0, 1]``.
    factor : float
        Non-negative interpolation factor.

    Returns
    -------
    Tensor
        Adjusted image, same shape and dtype as ``img``, clipped to ``[0, 1]``.
    """
    if factor < 0.0:
        raise ValueError(f"factor must be non-negative, got {factor}")
    if factor == 1.0:
        return img
    inv13 = 1.0 / 13.0
    kernel = [[inv13, inv13, inv13], [inv13, 5.0 * inv13, inv13], [inv13, inv13, inv13]]
    blurred = depthwise_conv2d(img, kernel)
    h, w = _spatial_hw(img)
    if h <= 2 or w <= 2:
        # No valid interior — kernel can't be applied; return identity.
        return img
    # Build a ``(1, H, W)`` interior mask: 1 in interior, 0 on the
    # 1-pixel border.  Broadcast-friendly: works for both ``(C, H, W)``
    # and ``(B, C, H, W)`` ``blurred``.
    inner_ones = lucid.ones(1, h - 2, w - 2, dtype=img.dtype, device=img.device)
    interior_mask = F.pad(inner_ones, (1, 1, 1, 1), value=0.0)
    border_mask = 1.0 - interior_mask
    smoothed = blurred * interior_mask + img * border_mask
    return _blend(img, smoothed, factor)


def autocontrast(img: Tensor) -> Tensor:
    r"""Per-channel min-max stretch to ``[0, 1]`` (PIL ``ImageOps.autocontrast``).

    Each channel's intensities are rescaled so its minimum maps to
    ``0`` and its maximum maps to ``1``.  A flat channel (``max ==
    min``) passes through unchanged.

    Parameters
    ----------
    img : Tensor
        Image ``(C, H, W)`` or ``(B, C, H, W)`` in ``[0, 1]``.

    Returns
    -------
    Tensor
        Contrast-stretched image, same shape and dtype as ``img``.
    """
    # min/max per channel over the spatial axes — both shape (..., C, 1, 1).
    minv = lucid.min(lucid.min(img, dim=-1, keepdim=True).values, dim=-2, keepdim=True).values
    maxv = lucid.max(lucid.max(img, dim=-1, keepdim=True).values, dim=-2, keepdim=True).values
    span = maxv - minv  # (..., C, 1, 1)
    # Multiplicative mask avoids the lucid.where shape constraint (which
    # requires cond + branches all the same shape, no broadcast).  The
    # mask is (..., C, 1, 1) and broadcasts over the spatial axes in
    # subsequent arithmetic.
    has_span = (span > 0.0).to(img.dtype)
    safe_span = has_span * span + (1.0 - has_span)  # 1 where flat → safe div
    stretched = (img - minv) / safe_span
    return has_span * stretched + (1.0 - has_span) * img


def posterize(img: Tensor, num_bits: int) -> Tensor:
    r"""Reduce bits per channel, bit-exact PIL ``ImageOps.posterize``.

    Round-trips through ``uint8`` and applies the bit mask
    ``~((1 << (8 - num_bits)) - 1)``, then divides by ``255``.  Output
    has ``2**num_bits`` distinct levels per channel.

    Parameters
    ----------
    img : Tensor
        Image ``(C, H, W)`` or ``(B, C, H, W)`` in ``[0, 1]``.
    num_bits : int
        Number of bits to keep per channel; must be in ``[1, 8]``.

    Returns
    -------
    Tensor
        Quantised image, same shape and dtype as ``img``.
    """
    if not 1 <= num_bits <= 8:
        raise ValueError(f"num_bits must be in [1, 8], got {num_bits}")
    if num_bits == 8:
        return img
    mask_int = (~((1 << (8 - num_bits)) - 1)) & 0xFF
    u8 = lucid.clip(lucid.round(img * 255.0), 0.0, 255.0).long()
    masked = u8 & mask_int
    out = masked.to(img.dtype) / 255.0
    # A NaN pixel has no bit pattern to mask, and casting one to an
    # integer is undefined — it came back 0, so a NaN entering the
    # pipeline left it as an ordinary black pixel with nothing raised.
    # Carry it through instead, as every other op here does.
    return lucid.where(img != img, lucid.full_like(out, float("nan")), out)


def solarize(img: Tensor, threshold: float) -> Tensor:
    r"""Invert pixels at or above ``threshold`` (PIL ``ImageOps.solarize``).

    Parameters
    ----------
    img : Tensor
        Image ``(C, H, W)`` or ``(B, C, H, W)`` in ``[0, 1]``.
    threshold : float
        Cut-off in ``[0, 1]``; pixels ``>= threshold`` become
        ``1 - pixel``.

    Returns
    -------
    Tensor
        Solarised image, same shape and dtype as ``img``.
    """
    return lucid.where(img >= threshold, 1.0 - img, img)


def invert(img: Tensor) -> Tensor:
    r"""Invert intensities (PIL ``ImageOps.invert``): ``1 - img``.

    Parameters
    ----------
    img : Tensor
        Image ``(C, H, W)`` or ``(B, C, H, W)`` in ``[0, 1]``.

    Returns
    -------
    Tensor
        Inverted image, same shape and dtype as ``img``.
    """
    return 1.0 - img
