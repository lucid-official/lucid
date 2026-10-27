"""Crop / pad family — Albumentations-compatible.

``Crop``, ``PadIfNeeded``, ``RandomSizedCrop``, ``CropAndPad`` — all
geometric, moving mask / boxes / keypoints with the image.
"""

from dataclasses import dataclass
from typing import override

import lucid
from lucid._tensor import Tensor
from lucid.utils.transforms import _random
from lucid.utils.transforms import functional as F
from lucid.utils.transforms._base import Empty, GeometricTransform, _NoParams
from lucid.utils.transforms._datatypes import (
    BoundingBoxes,
    Keypoints,
    crop_boxes,
    crop_keypoints,
    pad_boxes,
    pad_keypoints,
    resize_boxes,
    resize_keypoints,
)
from lucid.utils.transforms._interpolation import Interpolation, as_interpolation


@dataclass(frozen=True, slots=True)
class CropBox:
    r"""Per-call crop window for :class:`RandomSizedCrop`.

    The apply step extracts the rectangle ``[top:top+height,
    left:left+width]`` from the input and resizes it to the host
    transform's target size, with the same coordinates re-applied to
    mask / boxes / keypoints for alignment.

    Attributes
    ----------
    top : int
        Top edge of the crop window in input-image pixels.
    left : int
        Left edge of the crop window in input-image pixels.
    height : int
        Crop window height in pixels.
    width : int
        Crop window width in pixels.
    """

    top: int
    left: int
    height: int
    width: int


class Crop(_NoParams, GeometricTransform[Empty]):
    r"""Crop a fixed region ``[x_min, y_min, x_max, y_max)`` (Albumentations ``Crop``).

    Deterministic — every call returns the same rectangle regardless
    of input size.  Masks, boxes, and keypoints are translated in
    lock-step with the image (``GeometricTransform`` contract).

    Parameters
    ----------
    x_min, y_min : int, optional, default=0
        Top-left corner of the crop window (inclusive).
    x_max, y_max : int, optional, default=1024
        Bottom-right corner (exclusive).
    p : float, optional, default=1.0
        Probability of applying the crop.  Below ``1.0``, the input
        passes through unchanged on ``1 - p`` of calls.

    Examples
    --------
    >>> import lucid, lucid.utils.transforms as T
    >>> tf = T.Crop(0, 0, 16, 16, p=1.0)
    >>> tuple(tf(T.Image(lucid.rand(3, 32, 32))).data.shape)
    (3, 16, 16)

    The box is ``(x_min, y_min, x_max, y_max)``, so this takes the
    top-left quarter rather than a 16-pixel margin.
    """

    def __init__(
        self,
        x_min: int = 0,
        y_min: int = 0,
        x_max: int = 1024,
        y_max: int = 1024,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.x_min, self.y_min, self.x_max, self.y_max = x_min, y_min, x_max, y_max

    @property
    def _hw(self) -> tuple[int, int]:
        return self.y_max - self.y_min, self.x_max - self.x_min

    @override
    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:
        h, w = self._hw
        return F.crop(img, self.y_min, self.x_min, h, w)

    @override
    def _apply_mask(self, mask: Tensor, params: Empty) -> Tensor:
        h, w = self._hw
        return F.crop(mask, self.y_min, self.x_min, h, w)

    @override
    def _apply_boxes(self, boxes: BoundingBoxes, params: Empty) -> BoundingBoxes:
        h, w = self._hw
        return crop_boxes(boxes, self.y_min, self.x_min, h, w)

    @override
    def _apply_keypoints(self, kps: Keypoints, params: Empty) -> Keypoints:
        h, w = self._hw
        return crop_keypoints(kps, self.y_min, self.x_min, h, w)

    @override
    def __repr__(self) -> str:
        return (
            f"Crop(x_min={self.x_min}, y_min={self.y_min}, "
            f"x_max={self.x_max}, y_max={self.y_max}, p={self.p})"
        )


class PadIfNeeded(_NoParams, GeometricTransform[Empty]):
    r"""Pad to at least ``min_height`` x ``min_width`` (Albumentations ``PadIfNeeded``).

    Parameters
    ----------
    min_height, min_width : int or None
        Minimum output size; smaller inputs are padded up to it.  ``None``
        sets no minimum on that axis.
    border_mode : int, optional, default=4
    value : float or tuple of float, optional, default=0.0
        Fill for the image's padded area; a tuple gives one value per
        channel.
    mask_value : float, optional, default=0.0
    p : float, optional, default=1.0
    position : str, optional, default="center"
        Where a smaller input sits on the padded canvas: ``"center"``,
        ``"top_left"``, ``"top_right"``, ``"bottom_left"`` or
        ``"bottom_right"`` (the Albumentations names).  Boxes, masks and
        keypoints move with it.
    pad_height_divisor, pad_width_divisor : int or None, optional
        Pad that axis further, up to the next multiple of this (the
        Albumentations names).  ``None`` (default) leaves it at the
        minimum.

    Examples
    --------
    >>> import lucid, lucid.utils.transforms as T
    >>> tf = T.PadIfNeeded(64, 64, p=1.0)
    >>> tuple(tf(T.Image(lucid.rand(3, 32, 32))).data.shape)
    (3, 64, 64)

    Pads only what is short of the target; an image already large
    enough comes back untouched.  ``position="top_left"`` pads on the
    bottom and right only, so the input keeps its origin.

    >>> tl = T.PadIfNeeded(64, 64, position="top_left")
    >>> img = lucid.rand(3, 32, 32)
    >>> bool((tl(T.Image(img)).data[:, :32, :32] == img).all())
    True
    """

    def __init__(
        self,
        min_height: int | None = 1024,
        min_width: int | None = 1024,
        border_mode: int = 4,
        value: float | tuple[float, ...] = 0.0,
        mask_value: float = 0.0,
        p: float = 1.0,
        position: str = "center",
        pad_height_divisor: int | None = None,
        pad_width_divisor: int | None = None,
    ) -> None:
        for divisor in (pad_height_divisor, pad_width_divisor):
            if divisor is not None and divisor < 1:
                raise ValueError(f"PadIfNeeded: a divisor must be at least 1, got {divisor}")
        positions = ("center", "top_left", "top_right", "bottom_left", "bottom_right")
        if position not in positions:
            raise ValueError(f"PadIfNeeded: position must be one of {positions}, got {position!r}")
        super().__init__(p=p)
        self.min_height = min_height
        self.min_width = min_width
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value
        self.position = position
        self.pad_height_divisor = pad_height_divisor
        self.pad_width_divisor = pad_width_divisor

    @staticmethod
    def _target(size: int, minimum: int | None, divisor: int | None) -> int:
        target = size if minimum is None else max(minimum, size)
        return target if divisor is None else -(-target // divisor) * divisor

    def _pads(self, h: int, w: int) -> tuple[int, int, int, int]:
        dh = self._target(h, self.min_height, self.pad_height_divisor) - h
        dw = self._target(w, self.min_width, self.pad_width_divisor) - w
        if self.position == "center":
            top, left = dh // 2, dw // 2
        else:
            vertical, horizontal = self.position.split("_")
            top = 0 if vertical == "top" else dh
            left = 0 if horizontal == "left" else dw
        return left, dw - left, top, dh - top  # (l, r, t, b)

    @override
    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:
        h, w = F._spatial_hw(img)
        pads = self._pads(h, w)
        if not isinstance(self.value, tuple):
            return F.pad(img, pads, value=self.value)
        # One fill per channel: pad with 0, then add each channel's fill over
        # the padded ring only.  ``x + 0`` is exact, so the image keeps its
        # values bit for bit.
        ring = F.pad(lucid.zeros(1, h, w, dtype=img.dtype, device=img.device), pads, value=1.0)
        fill = lucid.tensor(list(self.value), dtype=img.dtype, device=img.device)
        return F.pad(img, pads, value=0.0) + ring * fill.reshape(-1, 1, 1)

    @override
    def _apply_mask(self, mask: Tensor, params: Empty) -> Tensor:
        h, w = F._spatial_hw(mask)
        return F.pad(mask, self._pads(h, w), value=self.mask_value)

    @override
    def _apply_boxes(self, boxes: BoundingBoxes, params: Empty) -> BoundingBoxes:
        h, w = boxes.canvas_size
        left, right, top, bottom = self._pads(h, w)
        return pad_boxes(boxes, left, top, h + top + bottom, w + left + right)

    @override
    def _apply_keypoints(self, kps: Keypoints, params: Empty) -> Keypoints:
        h, w = kps.canvas_size
        left, right, top, bottom = self._pads(h, w)
        return pad_keypoints(kps, left, top, h + top + bottom, w + left + right)

    @override
    def __repr__(self) -> str:
        where = "" if self.position == "center" else f", position={self.position!r}"
        return (
            f"PadIfNeeded(min_height={self.min_height}, "
            f"min_width={self.min_width}{where}, p={self.p})"
        )


class RandomSizedCrop(GeometricTransform[CropBox]):
    r"""Random-size crop then resize (Albumentations ``RandomSizedCrop``).

    Parameters
    ----------
    min_max_height : (int, int)
        Range of crop heights (pixels).
    height, width : int
        Output size after resizing.
    w2h_ratio : float, optional, default=1.0
        Crop width / height ratio.
    interpolation : int or str or Interpolation, optional, default=1
    p : float, optional, default=1.0

    Examples
    --------
    >>> import lucid, lucid.utils.transforms as T
    >>> tf = T.RandomSizedCrop((8, 16), 16, 16, p=1.0)
    >>> tuple(tf(T.Image(lucid.rand(3, 32, 32))).data.shape)
    (3, 16, 16)
    """

    def __init__(
        self,
        min_max_height: tuple[int, int],
        height: int,
        width: int,
        w2h_ratio: float = 1.0,
        interpolation: int | str | Interpolation = 1,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.min_max_height = min_max_height
        self.height = height
        self.width = width
        self.w2h_ratio = w2h_ratio
        self.interpolation = as_interpolation(interpolation)

    @override
    def make_params(self, img: Tensor) -> CropBox:
        r"""Sample per-call random parameters for :class:`RandomSizedCrop`.

        Parameters
        ----------
        img : Tensor
            Image tensor; its spatial size bounds both the sampled
            crop dimensions (clamped to fit) and the uniformly
            sampled ``(top, left)`` corner.

        Returns
        -------
        CropBox
            Crop window ``(top, left, height, width)``.  Height is
            drawn uniformly from ``min_max_height``; width is
            ``round(height * w2h_ratio)``; both are clipped to the
            input dimensions before the corner is sampled.
        """
        h, w = F._spatial_hw(img)
        ch = _random.randint(self.min_max_height[0], self.min_max_height[1] + 1)
        cw = int(round(ch * self.w2h_ratio))
        ch, cw = min(ch, h), min(cw, w)
        return CropBox(
            top=_random.randint(0, h - ch + 1),
            left=_random.randint(0, w - cw + 1),
            height=ch,
            width=cw,
        )

    @override
    def _apply_image(self, img: Tensor, params: CropBox) -> Tensor:
        return F.resized_crop(
            img,
            params.top,
            params.left,
            params.height,
            params.width,
            (self.height, self.width),
            interpolation=self.interpolation,
            antialias=False,  # Albumentations (OpenCV) does not low-pass filter
        )

    @override
    def _apply_mask(self, mask: Tensor, params: CropBox) -> Tensor:
        return F.resized_crop(
            mask,
            params.top,
            params.left,
            params.height,
            params.width,
            (self.height, self.width),
            interpolation=Interpolation.NEAREST,
        )

    @override
    def _apply_boxes(self, boxes: BoundingBoxes, params: CropBox) -> BoundingBoxes:
        cropped = crop_boxes(boxes, params.top, params.left, params.height, params.width)
        return resize_boxes(cropped, self.height, self.width)

    @override
    def _apply_keypoints(self, kps: Keypoints, params: CropBox) -> Keypoints:
        cropped = crop_keypoints(kps, params.top, params.left, params.height, params.width)
        return resize_keypoints(cropped, self.height, self.width)

    @override
    def __repr__(self) -> str:
        return (
            f"RandomSizedCrop(min_max_height={self.min_max_height}, "
            f"height={self.height}, width={self.width}, p={self.p})"
        )


class CropAndPad(_NoParams, GeometricTransform[Empty]):
    r"""Crop (negative) or pad (positive) every side by ``px`` pixels.

    A simplified Albumentations ``CropAndPad`` supporting a single int
    ``px`` applied to all four sides.

    Parameters
    ----------
    px : int
        Positive pads, negative crops, on every side.
    value : float, optional, default=0.0
        Pad fill (for positive ``px``).
    p : float, optional, default=1.0

    Examples
    --------
    >>> import lucid, lucid.utils.transforms as T
    >>> tf = T.CropAndPad(p=1.0)
    >>> tuple(tf(T.Image(lucid.rand(3, 32, 32))).data.shape)
    (3, 32, 32)
    """

    def __init__(self, px: int = 0, value: float = 0.0, p: float = 1.0) -> None:
        super().__init__(p=p)
        self.px = px
        self.value = value

    def _img_like(self, x: Tensor) -> Tensor:
        if self.px >= 0:
            return F.pad(x, self.px, value=self.value)
        c = -self.px
        h, w = F._spatial_hw(x)
        return F.crop(x, c, c, h - 2 * c, w - 2 * c)

    @override
    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:
        return self._img_like(img)

    @override
    def _apply_mask(self, mask: Tensor, params: Empty) -> Tensor:
        return self._img_like(mask)

    @override
    def _apply_boxes(self, boxes: BoundingBoxes, params: Empty) -> BoundingBoxes:
        h, w = boxes.canvas_size
        if self.px >= 0:
            return pad_boxes(boxes, self.px, self.px, h + 2 * self.px, w + 2 * self.px)
        c = -self.px
        return crop_boxes(boxes, c, c, h - 2 * c, w - 2 * c)

    @override
    def _apply_keypoints(self, kps: Keypoints, params: Empty) -> Keypoints:
        h, w = kps.canvas_size
        if self.px >= 0:
            return pad_keypoints(kps, self.px, self.px, h + 2 * self.px, w + 2 * self.px)
        c = -self.px
        return crop_keypoints(kps, c, c, h - 2 * c, w - 2 * c)

    @override
    def __repr__(self) -> str:
        return f"CropAndPad(px={self.px}, p={self.p})"
