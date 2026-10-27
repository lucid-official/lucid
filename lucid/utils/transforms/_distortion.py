"""Displacement-field distortions — Elastic / Grid / Optical.

Albumentations-compatible non-rigid warps.  Each samples a per-pixel
displacement field ``(dx, dy)``; images/masks are remapped through
``grid_sample`` (exact), while boxes/keypoints are displaced by sampling
the field at their coordinates — an approximation that matches
Albumentations' own handling of these transforms.
"""

from dataclasses import dataclass
from typing import override

import lucid
from lucid._tensor import Tensor
from lucid.utils.transforms import _random
from lucid.utils.transforms import functional as F
from lucid.utils.transforms._base import GeometricTransform
from lucid.utils.transforms._datatypes import (
    BoundingBoxes,
    Keypoints,
    _kp_xy_rest,
    _rebuild,
    to_xyxy,
)
from lucid.utils.transforms._interpolation import Interpolation, as_interpolation


@dataclass(slots=True)
class DispParams:
    r"""A sampled displacement field for one call (pixel units).

    Returned by each ``_DisplacementTransform.make_params`` —
    encapsulates the per-pixel ``(dx, dy)`` shift that
    :class:`ElasticTransform`, :class:`GridDistortion`,
    :class:`OpticalDistortion`, and :class:`GridElasticDeform`
    feed into :func:`~lucid.utils.transforms.functional.remap`.

    Attributes
    ----------
    dx, dy : Tensor
        Per-pixel displacement fields of shape ``(H, W)`` in pixel
        units.
    out_hw : tuple of (int, int)
        Output ``(H, W)`` — usually equal to the input size, but
        kept explicit so the dispatch hooks know the target canvas
        for boxes / keypoints.
    """

    dx: Tensor
    dy: Tensor
    out_hw: tuple[int, int]


def _displace_points(pts: Tensor, params: DispParams) -> Tensor:
    """Forward-displace ``(N, 2)`` points (≈ ``pt - field(pt)``)."""
    h, w = params.out_hw
    sx = F.sample_field_at_points(params.dx, pts, (h, w))
    sy = F.sample_field_at_points(params.dy, pts, (h, w))
    return pts - F._cat([sx, sy], 1)


class _DisplacementTransform(GeometricTransform[DispParams]):
    """Shared image/mask/box/keypoint application for displacement warps."""

    interpolation: Interpolation

    def _img_mode(self) -> str:
        return "nearest" if self.interpolation == Interpolation.NEAREST else "bilinear"

    @override
    def _apply_image(self, img: Tensor, params: DispParams) -> Tensor:
        return F.remap(img, params.dx, params.dy, mode=self._img_mode())

    @override
    def _apply_mask(self, mask: Tensor, params: DispParams) -> Tensor:
        return F.remap(mask, params.dx, params.dy, mode="nearest")

    @override
    def _apply_boxes(self, boxes: BoundingBoxes, params: DispParams) -> BoundingBoxes:
        xy = to_xyxy(boxes)
        x1, y1, x2, y2 = xy[:, 0:1], xy[:, 1:2], xy[:, 2:3], xy[:, 3:4]
        corners = F._cat(
            [
                F._cat([x1, y1], 1),
                F._cat([x2, y1], 1),
                F._cat([x2, y2], 1),
                F._cat([x1, y2], 1),
            ],
            0,
        )
        moved = _displace_points(corners, params)
        n = int(xy.shape[0])
        mx = moved[:, 0:1].reshape(4, n)
        my = moved[:, 1:2].reshape(4, n)
        h, w = params.out_hw
        nx1 = lucid.clip(lucid.min(mx, dim=0, keepdim=True).values.reshape(n, 1), 0.0, float(w))
        nx2 = lucid.clip(lucid.max(mx, dim=0, keepdim=True).values.reshape(n, 1), 0.0, float(w))
        ny1 = lucid.clip(lucid.min(my, dim=0, keepdim=True).values.reshape(n, 1), 0.0, float(h))
        ny2 = lucid.clip(lucid.max(my, dim=0, keepdim=True).values.reshape(n, 1), 0.0, float(h))
        return _rebuild(boxes, F._cat([nx1, ny1, nx2, ny2], 1), (h, w))

    @override
    def _apply_keypoints(self, kps: Keypoints, params: DispParams) -> Keypoints:
        x, y, rest = _kp_xy_rest(kps)
        moved = _displace_points(F._cat([x, y], 1), params)
        cols = [moved[:, 0:1], moved[:, 1:2]]
        if rest is not None:
            cols.append(rest)
        return Keypoints(F._cat(cols, 1), params.out_hw)


class ElasticTransform(_DisplacementTransform):
    r"""Elastic deformation (Simard 2003; Albumentations ``ElasticTransform``).

    Random fields smoothed by a Gaussian then scaled by ``alpha``.

    Parameters
    ----------
    alpha : float, optional, default=1.0
        Displacement magnitude.
    sigma : float, optional, default=50.0
        Gaussian smoothing of the displacement field (pixels).
    interpolation : int or str or Interpolation, optional, default=1
    p : float, optional, default=0.5

    Examples
    --------
    >>> import lucid, lucid.utils.transforms as T
    >>> tf = T.ElasticTransform(p=1.0)
    >>> tuple(tf(T.Image(lucid.rand(3, 32, 32))).data.shape)
    (3, 32, 32)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        sigma: float = 50.0,
        interpolation: int | str | Interpolation = 1,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.alpha = alpha
        self.sigma = sigma
        self.interpolation = as_interpolation(interpolation)

    @override
    def make_params(self, img: Tensor) -> DispParams:
        r"""Sample per-call random parameters for :class:`ElasticTransform`.

        Parameters
        ----------
        img : Tensor
            Image tensor; its spatial size sets the displacement
            field size and the output canvas.

        Returns
        -------
        DispParams
            Carries per-pixel ``dx`` / ``dy`` displacement fields of
            shape ``(H, W)`` in pixel units, built by sampling two
            uniform ``(1, 1, H, W)`` random fields in ``[-1, 1]``,
            blurring each with a Gaussian of ``sigma``, and scaling by
            ``alpha`` — the Simard 2003 recipe.  ``out_hw`` equals
            the input ``(H, W)``.
        """
        h, w = F._spatial_hw(img)
        rx = lucid.rand(1, 1, h, w, device=img.device) * 2.0 - 1.0
        ry = lucid.rand(1, 1, h, w, device=img.device) * 2.0 - 1.0
        dx = F.gaussian_blur(rx, self.sigma)[0, 0] * self.alpha
        dy = F.gaussian_blur(ry, self.sigma)[0, 0] * self.alpha
        return DispParams(dx=dx, dy=dy, out_hw=(h, w))

    @override
    def __repr__(self) -> str:
        return f"ElasticTransform(alpha={self.alpha}, sigma={self.sigma}, p={self.p})"


class GridDistortion(_DisplacementTransform):
    r"""Grid distortion (Albumentations ``GridDistortion``).

    Perturbs a coarse ``num_steps`` x ``num_steps`` control grid by
    ``distort_limit`` and upsamples it to a smooth displacement field.

    Parameters
    ----------
    num_steps : int, optional, default=5
    distort_limit : float or (float, float), optional, default=0.3
    interpolation : int or str or Interpolation, optional, default=1
    p : float, optional, default=0.5

    Examples
    --------
    >>> import lucid, lucid.utils.transforms as T
    >>> tf = T.GridDistortion(p=1.0)
    >>> tuple(tf(T.Image(lucid.rand(3, 32, 32))).data.shape)
    (3, 32, 32)
    """

    def __init__(
        self,
        num_steps: int = 5,
        distort_limit: float | tuple[float, float] = 0.3,
        interpolation: int | str | Interpolation = 1,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.num_steps = num_steps
        lim = distort_limit
        self.distort_limit = (-lim, lim) if isinstance(lim, (int, float)) else lim
        self.interpolation = as_interpolation(interpolation)

    @override
    def make_params(self, img: Tensor) -> DispParams:
        r"""Sample per-call random parameters for :class:`GridDistortion`.

        Parameters
        ----------
        img : Tensor
            Image tensor; its spatial size sets the displacement
            field size and the output canvas.

        Returns
        -------
        DispParams
            Carries per-pixel ``dx`` / ``dy`` displacement fields of
            shape ``(H, W)`` in pixel units.  Built by sampling a
            ``(num_steps + 1) x (num_steps + 1)`` control grid with
            entries in ``distort_limit`` (scaled by the cell size in
            each axis) and bilinearly upsampling to ``(H, W)``.
            ``out_hw`` equals the input ``(H, W)``.
        """
        h, w = F._spatial_hw(img)
        n = self.num_steps + 1
        cdx = [
            [_random.uniform(self.distort_limit[0], self.distort_limit[1]) for _ in range(n)]
            for _ in range(n)
        ]
        cdy = [
            [_random.uniform(self.distort_limit[0], self.distort_limit[1]) for _ in range(n)]
            for _ in range(n)
        ]
        coarse_dx = lucid.tensor(cdx, device=img.device).reshape(1, 1, n, n) * (w / self.num_steps)
        coarse_dy = lucid.tensor(cdy, device=img.device).reshape(1, 1, n, n) * (h / self.num_steps)
        from lucid.nn.functional import interpolate

        dx = interpolate(coarse_dx, size=(h, w), mode="bilinear", align_corners=True)[0, 0]
        dy = interpolate(coarse_dy, size=(h, w), mode="bilinear", align_corners=True)[0, 0]
        return DispParams(dx=dx, dy=dy, out_hw=(h, w))

    @override
    def __repr__(self) -> str:
        return (
            f"GridDistortion(num_steps={self.num_steps}, "
            f"distort_limit={self.distort_limit}, p={self.p})"
        )


class OpticalDistortion(_DisplacementTransform):
    r"""Radial (barrel / pincushion) distortion (Albumentations ``OpticalDistortion``).

    Parameters
    ----------
    distort_limit : float or (float, float), optional, default=0.05
        Radial coefficient range.
    shift_limit : float or (float, float), optional, default=0.05
        Optical-center shift range (fraction of size).
    interpolation : int or str or Interpolation, optional, default=1
    p : float, optional, default=0.5

    Examples
    --------
    >>> import lucid, lucid.utils.transforms as T
    >>> tf = T.OpticalDistortion(p=1.0)
    >>> tuple(tf(T.Image(lucid.rand(3, 32, 32))).data.shape)
    (3, 32, 32)
    """

    def __init__(
        self,
        distort_limit: float | tuple[float, float] = 0.05,
        shift_limit: float | tuple[float, float] = 0.05,
        interpolation: int | str | Interpolation = 1,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        dl = distort_limit
        sl = shift_limit
        self.distort_limit = (-dl, dl) if isinstance(dl, (int, float)) else dl
        self.shift_limit = (-sl, sl) if isinstance(sl, (int, float)) else sl
        self.interpolation = as_interpolation(interpolation)

    @override
    def make_params(self, img: Tensor) -> DispParams:
        r"""Sample per-call random parameters for :class:`OpticalDistortion`.

        Parameters
        ----------
        img : Tensor
            Image tensor; its spatial size sets the displacement
            field size, the centred normalisation, and ``out_hw``.

        Returns
        -------
        DispParams
            Carries per-pixel ``dx`` / ``dy`` fields of shape
            ``(H, W)`` realising the radial backward map
            ``sample(p) = c + (p - c) * (1 + k * r^2)`` with ``k``
            drawn uniformly from ``distort_limit`` and the optical
            centre ``c`` shifted from the image centre by
            ``shift_limit`` fractions.  ``out_hw`` equals the input
            ``(H, W)``.
        """
        h, w = F._spatial_hw(img)
        k = _random.uniform(self.distort_limit[0], self.distort_limit[1])
        cx = w / 2.0 + _random.uniform(self.shift_limit[0], self.shift_limit[1]) * w
        cy = h / 2.0 + _random.uniform(self.shift_limit[0], self.shift_limit[1]) * h
        yy, xx = F._pixel_grid(h, w, img.device)
        nx = (xx - cx) / (w / 2.0)
        ny = (yy - cy) / (h / 2.0)
        r2 = nx * nx + ny * ny
        factor = 1.0 + k * r2
        # Backward map: sampling coordinate = center + (p - center) * factor.
        dx = (xx - cx) * factor + cx - xx
        dy = (yy - cy) * factor + cy - yy
        return DispParams(dx=dx, dy=dy, out_hw=(h, w))

    @override
    def __repr__(self) -> str:
        return (
            f"OpticalDistortion(distort_limit={self.distort_limit}, "
            f"shift_limit={self.shift_limit}, p={self.p})"
        )


class GridElasticDeform(_DisplacementTransform):
    r"""Grid-based elastic deformation (Albumentations ``GridElasticDeform``).

    Displaces a coarse ``num_grid_xy`` control grid by up to ``magnitude``
    pixels and upsamples it to a smooth displacement field.

    Parameters
    ----------
    num_grid_xy : (int, int), optional, default=(4, 4)
        Control-grid resolution (x, y).
    magnitude : int, optional, default=10
        Max control-point displacement in pixels.
    interpolation : int or str or Interpolation, optional, default=1
    p : float, optional, default=0.5

    Examples
    --------
    >>> import lucid, lucid.utils.transforms as T
    >>> tf = T.GridElasticDeform(p=1.0)
    >>> tuple(tf(T.Image(lucid.rand(3, 32, 32))).data.shape)
    (3, 32, 32)
    """

    def __init__(
        self,
        num_grid_xy: tuple[int, int] = (4, 4),
        magnitude: int = 10,
        interpolation: int | str | Interpolation = 1,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.num_grid_xy = num_grid_xy
        self.magnitude = magnitude
        self.interpolation = as_interpolation(interpolation)

    @override
    def make_params(self, img: Tensor) -> DispParams:
        r"""Sample per-call random parameters for :class:`GridElasticDeform`.

        Parameters
        ----------
        img : Tensor
            Image tensor; its spatial size sets the displacement
            field size and the output canvas.

        Returns
        -------
        DispParams
            Carries per-pixel ``dx`` / ``dy`` fields of shape
            ``(H, W)`` in pixel units, built by sampling a
            ``(num_grid_xy[1] + 1, num_grid_xy[0] + 1)`` control grid
            with entries uniform in ``[-magnitude, magnitude]`` and
            bilinearly upsampling to ``(H, W)``.  ``out_hw`` equals
            the input ``(H, W)``.
        """
        from lucid.nn.functional import interpolate

        h, w = F._spatial_hw(img)
        gx, gy = self.num_grid_xy[0] + 1, self.num_grid_xy[1] + 1
        cdx = [
            [_random.uniform(-self.magnitude, self.magnitude) for _ in range(gx)] for _ in range(gy)
        ]
        cdy = [
            [_random.uniform(-self.magnitude, self.magnitude) for _ in range(gx)] for _ in range(gy)
        ]
        dx = interpolate(
            lucid.tensor(cdx, device=img.device).reshape(1, 1, gy, gx),
            size=(h, w),
            mode="bilinear",
            align_corners=True,
        )[0, 0]
        dy = interpolate(
            lucid.tensor(cdy, device=img.device).reshape(1, 1, gy, gx),
            size=(h, w),
            mode="bilinear",
            align_corners=True,
        )[0, 0]
        return DispParams(dx=dx, dy=dy, out_hw=(h, w))

    @override
    def __repr__(self) -> str:
        return (
            f"GridElasticDeform(num_grid_xy={self.num_grid_xy}, "
            f"magnitude={self.magnitude}, p={self.p})"
        )
