"""Dropout-style occlusion augmentations — CoarseDropout / GridDropout.

Photometric (image-only by default): they blank rectangular regions via
a multiplicative keep-mask (no in-place assignment).  Boxes / keypoints
pass through unchanged, matching Albumentations' default.
"""

from dataclasses import dataclass, field
from typing import override

import lucid
from lucid._tensor import Tensor
from lucid.utils.transforms import _random
from lucid.utils.transforms import functional as F
from lucid.utils.transforms._base import PhotometricTransform


@dataclass(slots=True)
class HoleParams:
    """Per-call list of ``(top, left, height, width)`` cutout windows.

    Used by :class:`CoarseDropout` / :class:`GridDropout` to carry the
    sampled rectangle list from ``make_params`` into the shared apply
    path so image, mask, boxes, and keypoints all see the same holes.
    """

    holes: list[tuple[int, int, int, int]] = field(default_factory=list)


def _apply_holes(img: Tensor, holes: list[tuple[int, int, int, int]], fill_value: float) -> Tensor:
    """Blank each ``(top, left, h, w)`` hole via a multiplicative keep-mask."""
    if not holes:
        return img
    h, w = F._spatial_hw(img)
    c = int(img.shape[-3])
    keep = lucid.ones(1, h, w, dtype=img.dtype, device=img.device)
    for top, left, hh, ww in holes:
        inner = lucid.zeros(1, hh, ww, dtype=img.dtype, device=img.device)
        block = F.pad(inner, (left, w - left - ww, top, h - top - hh), value=1.0)
        keep = keep * block
    keep_c = F._cat([keep] * c, 0)
    if img.ndim == 4:
        keep_c = keep_c[None]
    return img * keep_c + fill_value * (1.0 - keep_c)


class CoarseDropout(PhotometricTransform[HoleParams]):
    r"""Drop several random rectangles (Albumentations ``CoarseDropout``).

    Parameters
    ----------
    max_holes : int, optional, default=8
    max_height, max_width : int, optional, default=8
    min_holes, min_height, min_width : int, optional
        Default to the corresponding max.
    fill_value : float, optional, default=0.0
    p : float, optional, default=0.5

    Examples
    --------
    >>> import lucid, lucid.utils.transforms as T
    >>> tf = T.CoarseDropout(p=1.0)
    >>> tuple(tf(T.Image(lucid.rand(3, 32, 32))).data.shape)
    (3, 32, 32)
    """

    def __init__(
        self,
        max_holes: int = 8,
        max_height: int = 8,
        max_width: int = 8,
        min_holes: int | None = None,
        min_height: int | None = None,
        min_width: int | None = None,
        fill_value: float = 0.0,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.max_holes = max_holes
        self.max_height = max_height
        self.max_width = max_width
        self.min_holes = max_holes if min_holes is None else min_holes
        self.min_height = max_height if min_height is None else min_height
        self.min_width = max_width if min_width is None else min_width
        self.fill_value = fill_value

    @override
    def make_params(self, img: Tensor) -> HoleParams:
        h, w = F._spatial_hw(img)
        count = _random.randint(self.min_holes, self.max_holes + 1)
        holes: list[tuple[int, int, int, int]] = []
        for _ in range(count):
            hh = _random.randint(self.min_height, self.max_height + 1)
            ww = _random.randint(self.min_width, self.max_width + 1)
            hh, ww = min(hh, h), min(ww, w)
            top = _random.randint(0, h - hh + 1)
            left = _random.randint(0, w - ww + 1)
            holes.append((top, left, hh, ww))
        return HoleParams(holes=holes)

    @override
    def _apply_image(self, img: Tensor, params: HoleParams) -> Tensor:
        return _apply_holes(img, params.holes, self.fill_value)

    @override
    def __repr__(self) -> str:
        return (
            f"CoarseDropout(max_holes={self.max_holes}, max_height={self.max_height}, "
            f"max_width={self.max_width}, p={self.p})"
        )


class GridDropout(PhotometricTransform[HoleParams]):
    r"""Drop a regular grid of squares (Albumentations ``GridDropout``).

    Parameters
    ----------
    ratio : float, optional, default=0.5
        Fraction of each grid unit that is dropped.
    unit_size : int, optional, default=32
        Grid period in pixels.
    fill_value : float, optional, default=0.0
    p : float, optional, default=0.5

    Examples
    --------
    >>> import lucid, lucid.utils.transforms as T
    >>> tf = T.GridDropout(p=1.0)
    >>> tuple(tf(T.Image(lucid.rand(3, 32, 32))).data.shape)
    (3, 32, 32)
    """

    def __init__(
        self, ratio: float = 0.5, unit_size: int = 32, fill_value: float = 0.0, p: float = 0.5
    ) -> None:
        super().__init__(p=p)
        self.ratio = ratio
        self.unit_size = unit_size
        self.fill_value = fill_value

    @override
    def make_params(self, img: Tensor) -> HoleParams:
        h, w = F._spatial_hw(img)
        unit = self.unit_size
        hole = max(int(round(unit * self.ratio)), 1)
        holes: list[tuple[int, int, int, int]] = []
        top = 0
        while top < h:
            left = 0
            while left < w:
                hh = min(hole, h - top)
                ww = min(hole, w - left)
                holes.append((top, left, hh, ww))
                left += unit
            top += unit
        return HoleParams(holes=holes)

    @override
    def _apply_image(self, img: Tensor, params: HoleParams) -> Tensor:
        return _apply_holes(img, params.holes, self.fill_value)

    @override
    def __repr__(self) -> str:
        return f"GridDropout(ratio={self.ratio}, unit_size={self.unit_size}, p={self.p})"
