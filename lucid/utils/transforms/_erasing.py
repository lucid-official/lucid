"""Random erasing — Zhong et al., 2017 (arXiv:1708.04896).

A single rectangular region of the image is replaced with a constant,
per-channel mean, or random noise.  The canonical companion to
RandAugment / Mixup in modern ImageNet training recipes — included in
the reference_vision / timm reference pipelines.

Photometric (image-only by convention): mask / boxes / keypoints pass
through unchanged.  Implementation follows the
:class:`~lucid.utils.transforms.CoarseDropout` multiplicative-mask
pattern so there is no in-place assignment.
"""

import math
from dataclasses import dataclass
from typing import final, override

import lucid
from lucid._tensor import Tensor
from lucid.utils.transforms import _random
from lucid.utils.transforms import functional as F
from lucid.utils.transforms._base import PhotometricTransform


@final
@dataclass(frozen=True, slots=True)
class _ErasingParams:
    """Sampled erase rectangle + fill tensor (or no-op when no fit found)."""

    top: int
    left: int
    h: int
    w: int
    fill: Tensor | None  # ``None`` → no-op for this call


class RandomErasing(PhotometricTransform[_ErasingParams]):
    r"""Erase a random rectangular region of the image (Zhong et al., 2017).

    At each call, with probability ``p`` one rectangle whose area is a
    fraction of the image (sampled uniformly from ``scale``) and whose
    aspect ratio is sampled log-uniformly from ``ratio`` is replaced by
    ``value``.  If no rectangle fits after 10 attempts the call is a
    no-op (matches the reference_vision contract).

    Parameters
    ----------
    p : float, optional, default=0.5
        Probability of applying the erase.
    scale : (float, float), optional, default=(0.02, 0.33)
        Range of the erased-region area as a fraction of the image
        area.  Sampled uniformly per call.
    ratio : (float, float), optional, default=(0.3, 3.3)
        Range of the erased-region aspect ratio (W / H).  Sampled
        log-uniformly per call (matches the reference implementation).
    value : float or tuple of float or {"random"}, optional, default=0.0
        Fill value:

        * ``float`` → fill the rectangle with this scalar (broadcast
          across channels).
        * tuple of length ``C`` → per-channel constants (e.g. the
          ImageNet pixel mean).
        * ``"random"`` → fill with i.i.d. samples from
          :math:`\mathcal{N}(0, 1)`, drawn fresh on each call.

    Notes
    -----
    The erase is performed via a multiplicative keep-mask + additive
    fill (same pattern as :class:`CoarseDropout`) so the operation
    composes cleanly with autograd and avoids any in-place writes.

    The log-uniform aspect ratio means ``ratio=(0.3, 3.3)`` samples
    landscape and portrait rectangles symmetrically, exactly matching
    the spec in Zhong et al., 2017 §3.2.

    Examples
    --------
    Standard ImageNet recipe (``p=0.25``):

    >>> import lucid
    >>> from lucid.utils.transforms import RandomErasing
    >>> tf = RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3))
    >>> x = lucid.rand(3, 224, 224)
    >>> y = tf(x)
    >>> tuple(y.shape)
    (3, 224, 224)

    Per-channel ImageNet-mean fill:

    >>> tf = RandomErasing(p=1.0, value=(0.485, 0.456, 0.406))
    """

    def __init__(
        self,
        p: float = 0.5,
        scale: tuple[float, float] = (0.02, 0.33),
        ratio: tuple[float, float] = (0.3, 3.3),
        value: float | tuple[float, ...] | str = 0.0,
    ) -> None:
        super().__init__(p=p)
        if not (0.0 <= scale[0] <= scale[1] <= 1.0):
            raise ValueError(f"scale must satisfy 0 <= scale[0] <= scale[1] <= 1, got {scale}")
        if not (0.0 < ratio[0] <= ratio[1]):
            raise ValueError(f"ratio must satisfy 0 < ratio[0] <= ratio[1], got {ratio}")
        if isinstance(value, str) and value != "random":
            raise ValueError(f"value string must be 'random', got {value!r}")
        self.scale = scale
        self.ratio = ratio
        self.value = value

    @override
    def make_params(self, img: Tensor) -> _ErasingParams:
        h, w = F._spatial_hw(img)
        c = int(img.shape[-3])
        area = float(h * w)
        log_lo, log_hi = math.log(self.ratio[0]), math.log(self.ratio[1])
        # 10 attempts to fit a rectangle; otherwise no-op (reference_vision spec).
        for _ in range(10):
            target_area = _random.uniform(self.scale[0], self.scale[1]) * area
            aspect = math.exp(_random.uniform(log_lo, log_hi))
            eh = int(round(math.sqrt(target_area * aspect)))
            ew = int(round(math.sqrt(target_area / aspect)))
            if 0 < eh < h and 0 < ew < w:
                top = _random.randint(0, h - eh + 1)
                left = _random.randint(0, w - ew + 1)
                fill = self._make_fill(img, c, eh, ew)
                return _ErasingParams(top=top, left=left, h=eh, w=ew, fill=fill)
        return _ErasingParams(top=0, left=0, h=0, w=0, fill=None)

    def _make_fill(self, img: Tensor, c: int, eh: int, ew: int) -> Tensor:
        r"""Build the ``(C, eh, ew)`` fill tensor for the erase rectangle."""
        v = self.value
        if isinstance(v, str):  # "random"
            return lucid.randn(c, eh, ew, dtype=img.dtype, device=img.device)
        if isinstance(v, (int, float)):
            return lucid.full((c, eh, ew), float(v), dtype=img.dtype, device=img.device)
        # per-channel tuple
        if len(v) != c:
            raise ValueError(f"value tuple length {len(v)} != image channels {c}")
        col = lucid.tensor(list(v), dtype=img.dtype, device=img.device).reshape(c, 1, 1)
        return col * lucid.ones(c, eh, ew, dtype=img.dtype, device=img.device)

    @override
    def _apply_image(self, img: Tensor, params: _ErasingParams) -> Tensor:
        if params.fill is None:
            return img
        h, w = F._spatial_hw(img)
        c = int(img.shape[-3])
        # Keep mask: 1 outside the erase rectangle, 0 inside.
        inner = lucid.zeros(1, params.h, params.w, dtype=img.dtype, device=img.device)
        keep_1c = F.pad(
            inner,
            (params.left, w - params.left - params.w, params.top, h - params.top - params.h),
            value=1.0,
        )
        keep_c = F._cat([keep_1c] * c, 0)
        # Fill expanded to (C, H, W) — zero outside, fill inside the rect.
        fill_padded = F.pad(
            params.fill,
            (params.left, w - params.left - params.w, params.top, h - params.top - params.h),
            value=0.0,
        )
        if img.ndim == 4:
            keep_c = keep_c[None]
            fill_padded = fill_padded[None]
        return img * keep_c + fill_padded

    @override
    def __repr__(self) -> str:
        return (
            f"RandomErasing(p={self.p}, scale={self.scale}, "
            f"ratio={self.ratio}, value={self.value!r})"
        )
