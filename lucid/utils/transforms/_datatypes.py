"""Typed targets for multi-target transforms (reference_vision-v2 style).

A transform applied to a *sample* (image + companions) must move every
target consistently: a flip mirrors the image, its segmentation mask,
and its bounding boxes together.  To dispatch correctly, companions are
wrapped in lightweight typed objects:

* :class:`Image`        — the pixel image (a plain ``Tensor`` is also
  treated as an image, so single-image pipelines need no wrapping).
* :class:`Mask`         — a segmentation / label map; geometric
  transforms resample it with *nearest* interpolation, photometric
  transforms leave it untouched.
* :class:`BoundingBoxes`— ``(N, 4)`` boxes in a named ``format`` with a
  ``canvas_size``; geometric transforms update the coordinates.

Wrappers hold a :class:`lucid.Tensor` in ``.data`` plus metadata; they
are intentionally not ``Tensor`` subclasses (Lucid's tensor is
C-backed) — transforms dispatch on ``isinstance``.
"""

from dataclasses import dataclass

import lucid
from lucid._tensor import Tensor

# Supported bounding-box coordinate formats.
#   xyxy / xywh / cxcywh — absolute pixel coordinates.
#   albumentations       — normalized xyxy (x1,y1,x2,y2 in [0, 1]).
#   yolo                 — normalized cxcywh (cx,cy,w,h in [0, 1]).
# Albumentations names map: pascal_voc→xyxy, coco→xywh.
_BOX_FORMATS = frozenset({"xyxy", "xywh", "cxcywh", "albumentations", "yolo"})

# Albumentations format-name aliases → internal names.
_ALBU_FORMAT_ALIASES = {"pascal_voc": "xyxy", "coco": "xywh"}


def normalize_box_format(fmt: str) -> str:
    r"""Map an Albumentations format alias to the canonical internal name.

    Albumentations exposes two human-friendly aliases (``"pascal_voc"``
    for the corner-corner format and ``"coco"`` for the corner-size
    format); the internal representation uses the shorter
    ``"xyxy"`` / ``"xywh"`` names.  Unknown values pass through
    untouched so :class:`BoundingBoxes` can raise a clear error.

    Parameters
    ----------
    fmt : str
        A user-supplied format name.  Either an internal name
        (``"xyxy"``, ``"xywh"``, ``"cxcywh"``, ``"yolo"``,
        ``"albumentations"``) or an Albumentations alias
        (``"pascal_voc"``, ``"coco"``).

    Returns
    -------
    str
        The canonical internal name (one of ``"xyxy"``, ``"xywh"``,
        ``"cxcywh"``, ``"yolo"``, ``"albumentations"``), or ``fmt``
        unchanged if it isn't a known alias.

    Examples
    --------
    >>> normalize_box_format("pascal_voc")
    'xyxy'
    >>> normalize_box_format("coco")
    'xywh'
    >>> normalize_box_format("yolo")
    'yolo'
    """
    return _ALBU_FORMAT_ALIASES.get(fmt, fmt)


def _cat(tensors: list[Tensor], dim: int) -> Tensor:
    """``lucid.concat`` wrapper (absorbs the int32-typed ``dim`` stub)."""
    return lucid.concat(tensors, dim=dim)


@dataclass(slots=True)
class Image:
    """Wraps a pixel image tensor ``(C, H, W)`` or ``(B, C, H, W)``.

    A lightweight typed marker so multi-target transforms can dispatch
    on ``isinstance`` and route a pixel image differently from a
    segmentation :class:`Mask`, a :class:`BoundingBoxes` target, or a
    :class:`Keypoints` target.  Single-image pipelines need not wrap —
    a bare :class:`lucid.Tensor` is also treated as an image.

    Parameters
    ----------
    data : Tensor
        Pixel image with shape ``(C, H, W)`` or ``(B, C, H, W)``.
        Photometric transforms operate on ``data`` in place of
        per-channel statistics; geometric transforms resample with the
        configured interpolation mode.

    Examples
    --------
    The wrapper that says a tensor is a picture, so a transform knows to
    resample it rather than to treat it as labels:

    >>> import lucid, lucid.utils.transforms as T
    >>> holder = T.Image(lucid.rand(3, 32, 32))
    >>> tuple(T.Resize(16, 16)(holder).data.shape)
    (3, 16, 16)
    """

    data: Tensor


@dataclass(slots=True)
class Mask:
    """Wraps a segmentation / label mask, resampled with nearest mode.

    Companion target for semantic / instance segmentation pipelines.
    Geometric transforms (flip / crop / resize / rotate) move the mask
    in lock-step with its image but always resample with *nearest*
    interpolation so class indices survive unblended; photometric
    transforms (colour-jitter, blur, ...) leave the mask untouched.

    Parameters
    ----------
    data : Tensor
        Label / probability map with shape ``(H, W)``, ``(C, H, W)``,
        or ``(B, C, H, W)``.  Integer dtypes are recommended for class
        indices to preserve nearest-neighbour resampling exactly.

    Examples
    --------
    Says a tensor is labels rather than a picture, which changes how it
    is resampled — nearest-neighbour, so a class index is never
    interpolated into one that does not exist:

    >>> import lucid, lucid.utils.transforms as T
    >>> holder = T.Mask(lucid.zeros(1, 32, 32))
    >>> tuple(T.Resize(16, 16)(holder).data.shape)
    (1, 16, 16)

    A mask is ``(C, H, W)`` like an image — a bare ``(H, W)`` is refused
    by name rather than guessed at, since a two-dimensional tensor could
    as easily be a single channel or a batch of rows.
    """

    data: Tensor


@dataclass(slots=True)
class Keypoints:
    """Wraps ``(N, D)`` keypoints (``D >= 2``; first two columns ``x, y``).

    Extra columns (e.g. visibility / angle / scale) are carried through
    geometric transforms unchanged; only the ``x, y`` coordinates move.

    Parameters
    ----------
    data : Tensor
        ``(N, D)`` with ``D >= 2``; columns 0 and 1 are ``x`` and ``y``.
    canvas_size : (int, int)
        ``(H, W)`` of the image the points index into.

    Examples
    --------
    Points move with the image, so they need the canvas they were
    measured against:

    >>> import lucid, lucid.utils.transforms as T
    >>> points = T.Keypoints(lucid.tensor([[4.0, 4.0]]), canvas_size=(32, 32))
    >>> sample = {"image": T.Image(lucid.rand(3, 32, 32)), "keypoints": points}
    >>> moved = T.HorizontalFlip(p=1.0)(sample)
    >>> tuple(moved["keypoints"].data.shape)
    (1, 2)

    They have to travel with the image. On their own a geometric
    transform refuses them — "no image / mask in the sample to derive
    transform parameters" — because a flip needs to know the width it is
    flipping about, and the canvas size records where the points came
    from rather than what they are being mapped into.
    """

    data: Tensor
    canvas_size: tuple[int, int]


@dataclass(slots=True)
class BoundingBoxes:
    r"""Wraps ``(N, 4)`` bounding boxes with a format + canvas size.

    A typed target consumed by every :class:`GeometricTransform` —
    flips / crops / resizes update both the coordinates and the
    reported ``canvas_size`` consistently.  Optional ``labels`` ride
    along the same row dimension so :func:`filter_boxes` can drop
    out-of-frame boxes + their labels in one step.

    Parameters
    ----------
    data : Tensor
        ``(N, 4)`` coordinates interpreted under ``format``.  For the
        absolute formats (``xyxy`` / ``xywh`` / ``cxcywh``) units are
        pixels; for the normalized formats (``yolo`` / ``albumentations``)
        units are fractions of ``canvas_size``.
    format : {"xyxy", "xywh", "cxcywh", "yolo", "albumentations", \
              "pascal_voc", "coco"}
        Coordinate convention.  ``"pascal_voc"`` is an alias for
        ``"xyxy"``; ``"coco"`` for ``"xywh"``.  See
        :func:`normalize_box_format`.
    canvas_size : (int, int)
        ``(H, W)`` of the image the boxes index into.  Geometric
        transforms (crop / resize / pad / rotate) re-emit a new
        :class:`BoundingBoxes` with the matching canvas.
    labels : Tensor, optional
        ``(N,)`` class indices (or any per-box scalar — score, track
        id, ...).  Carried through every geometric transform; trimmed
        in lock-step with rows by :func:`filter_boxes`.  Default
        ``None`` (no per-box labels).

    Raises
    ------
    ValueError
        If ``format`` (after alias normalisation) is not one of the
        recognised names.

    Notes
    -----
    Format conventions (each row's 4 values):

    * ``xyxy`` — ``(x_min, y_min, x_max, y_max)``
    * ``xywh`` — ``(x_min, y_min, width, height)``
    * ``cxcywh`` — ``(cx, cy, width, height)``
    * ``yolo`` — ``(cx, cy, w, h)``, all in ``[0, 1]``
    * ``albumentations`` — ``(x_min, y_min, x_max, y_max)``, all in ``[0, 1]``

    Examples
    --------
    >>> import lucid
    >>> import lucid.utils.transforms as T
    >>> boxes = T.BoundingBoxes(
    ...     lucid.tensor([[10.0, 10.0, 40.0, 40.0]]),
    ...     "xyxy",
    ...     canvas_size=(100, 100),
    ...     labels=lucid.tensor([1.0]),
    ... )
    >>> out = T.HorizontalFlip(p=1.0)({"image": T.Image(lucid.rand(3, 100, 100)),
    ...                                "boxes": boxes})
    >>> out["boxes"].labels.numpy().tolist()
    [1.0]
    """

    data: Tensor
    format: str
    canvas_size: tuple[int, int]
    labels: Tensor | None = None

    def __post_init__(self) -> None:
        self.format = normalize_box_format(self.format)
        if self.format not in _BOX_FORMATS:
            raise ValueError(
                f"BoundingBoxes: unknown format {self.format!r}; "
                f"expected one of {sorted(_BOX_FORMATS)} (or pascal_voc/coco)"
            )


def to_xyxy(boxes: BoundingBoxes) -> Tensor:
    r"""Return the box coordinates as ``(N, 4)`` absolute-pixel xyxy.

    Normalisation-free output: regardless of whether ``boxes`` is in
    ``yolo`` (normalized cxcywh) or ``albumentations`` (normalized
    xyxy), the returned tensor is in pixel units of
    ``boxes.canvas_size``.  This is the canonical pivot every
    transform uses internally — flips / crops / resizes operate on
    xyxy and :func:`from_xyxy` re-encodes back to the original format.

    Parameters
    ----------
    boxes : BoundingBoxes
        Source boxes in any supported format.

    Returns
    -------
    Tensor
        ``(N, 4)`` absolute-pixel ``xyxy``.  Identical to ``boxes.data``
        when the source format is already ``xyxy``.
    """
    d = boxes.data
    h, w = boxes.canvas_size
    fmt = boxes.format
    if fmt == "xyxy":
        return d
    a, b, c, e = d[:, 0:1], d[:, 1:2], d[:, 2:3], d[:, 3:4]
    if fmt == "xywh":
        return _cat([a, b, a + c, b + e], 1)
    if fmt == "cxcywh":
        return _cat([a - c / 2, b - e / 2, a + c / 2, b + e / 2], 1)
    if fmt == "albumentations":  # normalized xyxy
        return _cat([a * w, b * h, c * w, e * h], 1)
    # yolo: normalized cxcywh
    cx, cy, bw, bh = a * w, b * h, c * w, e * h
    return _cat([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)


def from_xyxy(xyxy: Tensor, fmt: str, canvas: tuple[int, int]) -> Tensor:
    r"""Convert absolute ``(N, 4)`` xyxy coordinates back into ``fmt``.

    Inverse of :func:`to_xyxy`.  Normalised output formats divide by
    ``canvas`` so the round-trip ``from_xyxy(to_xyxy(b), b.format,
    b.canvas_size)`` reproduces the original ``data`` tensor.

    Parameters
    ----------
    xyxy : Tensor
        ``(N, 4)`` absolute-pixel ``xyxy`` coordinates.
    fmt : str
        Target format (any of ``xyxy``, ``xywh``, ``cxcywh``,
        ``albumentations``, ``yolo``).  Pass the alias-normalised name
        — use :func:`normalize_box_format` first if you have an Albu
        alias.
    canvas : (int, int)
        ``(H, W)`` used to normalise when ``fmt`` is ``"yolo"`` or
        ``"albumentations"``; ignored for the absolute formats.

    Returns
    -------
    Tensor
        ``(N, 4)`` coordinates in ``fmt``.
    """
    h, w = canvas
    x1, y1, x2, y2 = xyxy[:, 0:1], xyxy[:, 1:2], xyxy[:, 2:3], xyxy[:, 3:4]
    if fmt == "xyxy":
        return xyxy
    if fmt == "xywh":
        return _cat([x1, y1, x2 - x1, y2 - y1], 1)
    if fmt == "cxcywh":
        return _cat([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], 1)
    if fmt == "albumentations":
        return _cat([x1 / w, y1 / h, x2 / w, y2 / h], 1)
    # yolo
    return _cat([(x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h], 1)


def _rebuild(boxes: BoundingBoxes, xyxy: Tensor, canvas: tuple[int, int]) -> BoundingBoxes:
    """Re-wrap transformed xyxy coords back into the original format + labels."""
    return BoundingBoxes(from_xyxy(xyxy, boxes.format, canvas), boxes.format, canvas, boxes.labels)


def flip_boxes(boxes: BoundingBoxes, *, horizontal: bool) -> BoundingBoxes:
    r"""Mirror bounding boxes horizontally or vertically within the canvas.

    Geometric companion to :class:`HorizontalFlip` / :class:`VerticalFlip`
    so a sample's boxes stay aligned with its flipped image.  Operates
    on the :func:`to_xyxy` pivot internally and re-encodes the result
    back into the original ``boxes.format``; ``boxes.labels`` ride
    through unchanged.

    Parameters
    ----------
    boxes : BoundingBoxes
        Source boxes in any supported format
        (``xyxy`` / ``xywh`` / ``cxcywh`` / ``yolo`` / ``albumentations``).
    horizontal : bool
        ``True`` mirrors along the vertical axis (left/right swap);
        ``False`` mirrors along the horizontal axis (top/bottom swap).

    Returns
    -------
    BoundingBoxes
        Flipped boxes with the same format, canvas, and labels.
    """
    h, w = boxes.canvas_size
    xy = to_xyxy(boxes)
    x1, y1, x2, y2 = xy[:, 0:1], xy[:, 1:2], xy[:, 2:3], xy[:, 3:4]
    if horizontal:
        new = _cat([w - x2, y1, w - x1, y2], 1)
    else:
        new = _cat([x1, h - y2, x2, h - y1], 1)
    return _rebuild(boxes, new, boxes.canvas_size)


def resize_boxes(boxes: BoundingBoxes, new_h: int, new_w: int) -> BoundingBoxes:
    r"""Scale box coordinates to a new canvas size.

    Geometric companion to :class:`Resize` so a sample's boxes follow
    the resampled image.  Computes independent per-axis scale factors
    :math:`s_x = W' / W`, :math:`s_y = H' / H` on the :func:`to_xyxy`
    pivot and re-emits a :class:`BoundingBoxes` with ``canvas_size``
    set to ``(new_h, new_w)``.

    Parameters
    ----------
    boxes : BoundingBoxes
        Source boxes; ``canvas_size`` is read for the input height/width.
    new_h : int
        Target canvas height in pixels.
    new_w : int
        Target canvas width in pixels.

    Returns
    -------
    BoundingBoxes
        Rescaled boxes with the same format / labels and the new
        ``canvas_size = (new_h, new_w)``.
    """
    h, w = boxes.canvas_size
    sx, sy = new_w / w, new_h / h
    xy = to_xyxy(boxes)
    x1, y1, x2, y2 = xy[:, 0:1], xy[:, 1:2], xy[:, 2:3], xy[:, 3:4]
    new = _cat([x1 * sx, y1 * sy, x2 * sx, y2 * sy], 1)
    return _rebuild(boxes, new, (new_h, new_w))


def crop_boxes(boxes: BoundingBoxes, top: int, left: int, height: int, width: int) -> BoundingBoxes:
    r"""Translate boxes into a crop window and clip to its bounds.

    Geometric companion to :class:`Crop` / :class:`CenterCrop` /
    :class:`RandomCrop`.  Subtracts the ``(left, top)`` offset from
    every corner and clips to the ``[0, width] × [0, height]`` window
    so boxes spanning the crop edge are kept as their visible
    rectangle; fully-out-of-frame boxes become zero-area and are
    typically dropped downstream by :func:`filter_boxes`.

    Parameters
    ----------
    boxes : BoundingBoxes
        Source boxes; their canvas is implied to contain the crop.
    top : int
        Pixel row of the crop window's top edge.
    left : int
        Pixel column of the crop window's left edge.
    height : int
        Crop window height in pixels — becomes the new canvas height.
    width : int
        Crop window width in pixels — becomes the new canvas width.

    Returns
    -------
    BoundingBoxes
        Cropped + clipped boxes with the same format / labels and
        ``canvas_size = (height, width)``.
    """
    xy = to_xyxy(boxes)
    x1 = lucid.clip(xy[:, 0:1] - left, 0.0, float(width))
    y1 = lucid.clip(xy[:, 1:2] - top, 0.0, float(height))
    x2 = lucid.clip(xy[:, 2:3] - left, 0.0, float(width))
    y2 = lucid.clip(xy[:, 3:4] - top, 0.0, float(height))
    new = _cat([x1, y1, x2, y2], 1)
    return _rebuild(boxes, new, (height, width))


def pad_boxes(boxes: BoundingBoxes, left: int, top: int, new_h: int, new_w: int) -> BoundingBoxes:
    r"""Shift boxes by a pad offset onto a larger canvas.

    Geometric companion to ``Pad`` / :class:`PadIfNeeded`.  Adds
    ``(left, top)`` to every corner so each box keeps its place inside
    the original sub-region of the new, larger canvas.  Coordinates are
    not clipped because the new canvas fully contains the old.

    Parameters
    ----------
    boxes : BoundingBoxes
        Source boxes on the pre-pad canvas.
    left : int
        Number of pixels added to the left edge.
    top : int
        Number of pixels added to the top edge.
    new_h : int
        Post-pad canvas height in pixels.
    new_w : int
        Post-pad canvas width in pixels.

    Returns
    -------
    BoundingBoxes
        Translated boxes with the same format / labels and
        ``canvas_size = (new_h, new_w)``.
    """
    xy = to_xyxy(boxes)
    x1, y1, x2, y2 = xy[:, 0:1], xy[:, 1:2], xy[:, 2:3], xy[:, 3:4]
    new = _cat([x1 + left, y1 + top, x2 + left, y2 + top], 1)
    return _rebuild(boxes, new, (new_h, new_w))


# ── keypoints ───────────────────────────────────────────────────────


def _kp_xy_rest(kps: Keypoints) -> tuple[Tensor, Tensor, Tensor | None]:
    """Split keypoint data into ``x``, ``y``, and any trailing columns."""
    d = kps.data
    x = d[:, 0:1]
    y = d[:, 1:2]
    rest = d[:, 2:] if int(d.shape[1]) > 2 else None
    return x, y, rest


def _kp_rebuild(x: Tensor, y: Tensor, rest: Tensor | None, canvas: tuple[int, int]) -> Keypoints:
    cols = [x, y] if rest is None else [x, y, rest]
    return Keypoints(_cat(cols, 1), canvas)


def flip_keypoints(kps: Keypoints, *, horizontal: bool) -> Keypoints:
    r"""Mirror keypoint coordinates horizontally or vertically.

    Geometric companion to :class:`HorizontalFlip` / :class:`VerticalFlip`
    for the keypoints target.  Only the ``x, y`` columns move; any
    trailing columns (visibility / score / angle / scale) are passed
    through unchanged.

    Parameters
    ----------
    kps : Keypoints
        Source ``(N, D)`` keypoints with ``D >= 2``.
    horizontal : bool
        ``True`` mirrors along the vertical axis (``x' = W - x``);
        ``False`` mirrors along the horizontal axis (``y' = H - y``).

    Returns
    -------
    Keypoints
        Flipped keypoints with the same canvas size and trailing
        columns preserved.
    """
    h, w = kps.canvas_size
    x, y, rest = _kp_xy_rest(kps)
    if horizontal:
        x = w - x
    else:
        y = h - y
    return _kp_rebuild(x, y, rest, kps.canvas_size)


def resize_keypoints(kps: Keypoints, new_h: int, new_w: int) -> Keypoints:
    r"""Scale keypoint coordinates to a new canvas size.

    Geometric companion to :class:`Resize` for the keypoints target.
    Computes independent per-axis scale factors :math:`s_x = W' / W`,
    :math:`s_y = H' / H` and applies them to the ``x, y`` columns;
    trailing columns are preserved.

    Parameters
    ----------
    kps : Keypoints
        Source keypoints; ``canvas_size`` is read for the input
        height/width.
    new_h : int
        Target canvas height in pixels.
    new_w : int
        Target canvas width in pixels.

    Returns
    -------
    Keypoints
        Rescaled keypoints with ``canvas_size = (new_h, new_w)``.
    """
    h, w = kps.canvas_size
    x, y, rest = _kp_xy_rest(kps)
    return _kp_rebuild(x * (new_w / w), y * (new_h / h), rest, (new_h, new_w))


def crop_keypoints(kps: Keypoints, top: int, left: int, height: int, width: int) -> Keypoints:
    r"""Translate keypoints into a crop window (canvas to crop size).

    Geometric companion to :class:`Crop` / :class:`CenterCrop` /
    :class:`RandomCrop` for the keypoints target.  Subtracts the
    ``(left, top)`` offset from every point; unlike :func:`crop_boxes`
    the result is **not clipped**, so keypoints falling outside the
    crop window keep their (now-negative or out-of-range) coordinates
    — callers can detect them via the visibility column or by range
    test, since dropping rows would break joint-index assumptions.

    Parameters
    ----------
    kps : Keypoints
        Source keypoints; their canvas is implied to contain the crop.
    top : int
        Pixel row of the crop window's top edge.
    left : int
        Pixel column of the crop window's left edge.
    height : int
        Crop window height in pixels — becomes the new canvas height.
    width : int
        Crop window width in pixels — becomes the new canvas width.

    Returns
    -------
    Keypoints
        Translated keypoints with ``canvas_size = (height, width)``.
    """
    x, y, rest = _kp_xy_rest(kps)
    return _kp_rebuild(x - left, y - top, rest, (height, width))


def pad_keypoints(kps: Keypoints, left: int, top: int, new_h: int, new_w: int) -> Keypoints:
    r"""Shift keypoints by a pad offset onto a larger canvas.

    Geometric companion to ``Pad`` / :class:`PadIfNeeded` for the
    keypoints target.  Adds ``(left, top)`` to every point so each
    keeps its place inside the original sub-region of the new, larger
    canvas; trailing columns are preserved.

    Parameters
    ----------
    kps : Keypoints
        Source keypoints on the pre-pad canvas.
    left : int
        Number of pixels added to the left edge.
    top : int
        Number of pixels added to the top edge.
    new_h : int
        Post-pad canvas height in pixels.
    new_w : int
        Post-pad canvas width in pixels.

    Returns
    -------
    Keypoints
        Translated keypoints with ``canvas_size = (new_h, new_w)``.
    """
    x, y, rest = _kp_xy_rest(kps)
    return _kp_rebuild(x + left, y + top, rest, (new_h, new_w))


# ── affine / transpose / rot90 (boxes + keypoints) ──────────────────


def _clip_xyxy(x1: Tensor, y1: Tensor, x2: Tensor, y2: Tensor, h: int, w: int) -> Tensor:
    return _cat(
        [
            lucid.clip(x1, 0.0, float(w)),
            lucid.clip(y1, 0.0, float(h)),
            lucid.clip(x2, 0.0, float(w)),
            lucid.clip(y2, 0.0, float(h)),
        ],
        dim=1,
    )


def affine_boxes(boxes: BoundingBoxes, matrix: Tensor, out_hw: tuple[int, int]) -> BoundingBoxes:
    r"""Warp boxes by a forward affine matrix and re-axis-align them.

    Geometric companion to :class:`Affine` / :class:`Rotate` /
    :class:`ShiftScaleRotate`.  Each input rectangle is warped by mapping
    its four corners through ``matrix`` and then taking the smallest
    enclosing axis-aligned box of the warped quadrilateral.  Results
    are clipped to ``out_hw`` so partially-out-of-frame boxes survive
    as their visible rectangle.

    Parameters
    ----------
    boxes : BoundingBoxes
        Source boxes in any supported format.
    matrix : Tensor
        ``(2, 3)`` forward affine matrix in pixel coordinates, applied
        via :func:`lucid.utils.transforms.functional.affine_points`.
    out_hw : (int, int)
        ``(H', W')`` of the warped output canvas; becomes the new
        ``canvas_size`` and is used to clip the corner coordinates.

    Returns
    -------
    BoundingBoxes
        Axis-aligned enclosing boxes with the same format / labels and
        ``canvas_size = out_hw``.

    Notes
    -----
    Because the result is re-axis-aligned, the returned area is an
    upper bound on the warped rectangle's area whenever ``matrix``
    includes a non-zero rotation or shear component.
    """
    from lucid.utils.transforms import functional as _F

    xy = to_xyxy(boxes)
    x1, y1, x2, y2 = xy[:, 0:1], xy[:, 1:2], xy[:, 2:3], xy[:, 3:4]
    # Four corners stacked as (4N, 2).
    corners = _cat(
        [
            _cat([x1, y1], 1),
            _cat([x2, y1], 1),
            _cat([x2, y2], 1),
            _cat([x1, y2], 1),
        ],
        dim=0,
    )
    warped = _F.affine_points(corners, matrix)  # (4N, 2)
    n = int(xy.shape[0])
    wx = warped[:, 0:1].reshape(4, n)
    wy = warped[:, 1:2].reshape(4, n)
    nx1 = lucid.min(wx, dim=0, keepdim=True).values.reshape(n, 1)
    nx2 = lucid.max(wx, dim=0, keepdim=True).values.reshape(n, 1)
    ny1 = lucid.min(wy, dim=0, keepdim=True).values.reshape(n, 1)
    ny2 = lucid.max(wy, dim=0, keepdim=True).values.reshape(n, 1)
    new = _clip_xyxy(nx1, ny1, nx2, ny2, out_hw[0], out_hw[1])
    return _rebuild(boxes, new, out_hw)


def affine_keypoints(kps: Keypoints, matrix: Tensor, out_hw: tuple[int, int]) -> Keypoints:
    r"""Warp keypoint coordinates by a forward affine matrix.

    Geometric companion to :class:`Affine` / :class:`Rotate` /
    :class:`ShiftScaleRotate` for the keypoints target.  The ``x, y``
    columns are passed through
    :func:`lucid.utils.transforms.functional.affine_points`; trailing
    columns (visibility / score / angle / scale) are preserved.

    Parameters
    ----------
    kps : Keypoints
        Source ``(N, D)`` keypoints with ``D >= 2``.
    matrix : Tensor
        ``(2, 3)`` forward affine matrix in pixel coordinates.
    out_hw : (int, int)
        ``(H', W')`` of the warped output canvas; becomes the new
        ``canvas_size``.

    Returns
    -------
    Keypoints
        Warped keypoints with ``canvas_size = out_hw`` and trailing
        columns unchanged.  Coordinates are **not clipped** — out-of-frame
        points keep their warped position for the caller to inspect.
    """
    from lucid.utils.transforms import functional as _F

    x, y, rest = _kp_xy_rest(kps)
    xy = _F.affine_points(_cat([x, y], 1), matrix)
    return _kp_rebuild(xy[:, 0:1], xy[:, 1:2], rest, out_hw)


def transpose_boxes(boxes: BoundingBoxes) -> BoundingBoxes:
    r"""Swap the ``x`` and ``y`` axes of every box (image transpose).

    Geometric companion to :class:`Transpose` (matrix transpose of the
    image).  Each box ``(x1, y1, x2, y2)`` becomes
    ``(y1, x1, y2, x2)`` and the canvas swaps from ``(H, W)`` to
    ``(W, H)`` so the result still indexes a valid image.

    Parameters
    ----------
    boxes : BoundingBoxes
        Source boxes on a ``(H, W)`` canvas.

    Returns
    -------
    BoundingBoxes
        Transposed boxes with the same format / labels and
        ``canvas_size = (W, H)``.
    """
    h, w = boxes.canvas_size
    xy = to_xyxy(boxes)
    x1, y1, x2, y2 = xy[:, 0:1], xy[:, 1:2], xy[:, 2:3], xy[:, 3:4]
    new = _cat([y1, x1, y2, x2], 1)
    return _rebuild(boxes, new, (w, h))


def transpose_keypoints(kps: Keypoints) -> Keypoints:
    r"""Swap the ``x`` and ``y`` columns of every keypoint.

    Geometric companion to :class:`Transpose` for the keypoints target.
    Each point ``(x, y, *rest)`` becomes ``(y, x, *rest)`` and the
    canvas swaps from ``(H, W)`` to ``(W, H)``; trailing columns are
    preserved.

    Parameters
    ----------
    kps : Keypoints
        Source ``(N, D)`` keypoints with ``D >= 2`` on a ``(H, W)``
        canvas.

    Returns
    -------
    Keypoints
        Transposed keypoints with ``canvas_size = (W, H)`` and trailing
        columns unchanged.
    """
    h, w = kps.canvas_size
    x, y, rest = _kp_xy_rest(kps)
    return _kp_rebuild(y, x, rest, (w, h))


def rot90_boxes(boxes: BoundingBoxes, k: int) -> BoundingBoxes:
    r"""Rotate boxes by ``k`` CCW quarter-turns to match :func:`rot90`.

    Geometric companion to :class:`RandomRotate90` / pipeline use of
    :func:`lucid.rot90` on the image side.  Each quarter-turn maps a
    box ``(x1, y1, x2, y2)`` on a ``(H, W)`` canvas to
    ``(y1, (W-1)-x2, y2, (W-1)-x1)`` on a ``(W, H)`` canvas, applied
    ``k mod 4`` times; ``k = 0`` is a no-op.

    Parameters
    ----------
    boxes : BoundingBoxes
        Source boxes.
    k : int
        Number of 90° counter-clockwise rotations to apply.  Reduced
        modulo 4; negative values are valid (Python ``%`` semantics
        give a non-negative residue).

    Returns
    -------
    BoundingBoxes
        Rotated boxes with the same format / labels and the canvas
        appropriately swapped (``H`` <-> ``W``) for odd ``k``.
    """
    out = boxes
    for _ in range(k % 4):
        h, w = out.canvas_size
        xy = to_xyxy(out)
        x1, y1, x2, y2 = xy[:, 0:1], xy[:, 1:2], xy[:, 2:3], xy[:, 3:4]
        # CCW: x' = y, y' = (W-1) - x; canvas -> (W, H).
        nx1, nx2 = y1, y2
        ny1 = (w - 1) - x2
        ny2 = (w - 1) - x1
        new = _cat([nx1, ny1, nx2, ny2], 1)
        out = _rebuild(out, new, (w, h))
    return out


def rot90_keypoints(kps: Keypoints, k: int) -> Keypoints:
    r"""Rotate keypoints by ``k`` CCW quarter-turns to match :func:`rot90`.

    Geometric companion to :class:`RandomRotate90` for the keypoints
    target.  Each quarter-turn maps ``(x, y)`` on a ``(H, W)`` canvas to
    ``(y, (W-1)-x)`` on a ``(W, H)`` canvas, applied ``k mod 4`` times;
    trailing columns (visibility / score / ...) are preserved.

    Parameters
    ----------
    kps : Keypoints
        Source ``(N, D)`` keypoints with ``D >= 2``.
    k : int
        Number of 90° counter-clockwise rotations to apply.  Reduced
        modulo 4.

    Returns
    -------
    Keypoints
        Rotated keypoints with the canvas appropriately swapped
        (``H`` <-> ``W``) for odd ``k`` and trailing columns unchanged.
    """
    out = kps
    for _ in range(k % 4):
        h, w = out.canvas_size
        x, y, rest = _kp_xy_rest(out)
        out = _kp_rebuild(y, (w - 1) - x, rest, (w, h))
    return out


# ── bounding-box area + filtering (detection pipelines) ─────────────


def box_areas(boxes: BoundingBoxes) -> list[float]:
    r"""Absolute pixel area of every box, as a Python ``list[float]``.

    Computes :math:`\max(0, (x_2 - x_1)(y_2 - y_1))` on the
    :func:`to_xyxy` representation.  Used by :func:`filter_boxes` to
    drop degenerate / too-small / too-occluded rows, and by
    :class:`~lucid.utils.transforms.Compose` to record pre-pipeline
    areas for the ``min_visibility`` calculation.

    Parameters
    ----------
    boxes : BoundingBoxes
        Source boxes in any supported format.

    Returns
    -------
    list of float
        ``N`` entries, one per box, clamped at 0.  A Python list (not
        a Tensor) so callers can stash it across the pipeline without
        carrying an autograd handle.

    Examples
    --------
    >>> import lucid
    >>> import lucid.utils.transforms as T
    >>> b = T.BoundingBoxes(lucid.tensor([[0.0, 0.0, 10.0, 5.0]]),
    ...                     "xyxy", (100, 100))
    >>> box_areas(b)
    [50.0]
    """
    xy = to_xyxy(boxes)
    n = int(xy.shape[0])
    w = xy[:, 2] - xy[:, 0]
    h = xy[:, 3] - xy[:, 1]
    area = w * h
    return [max(float(area[i].item()), 0.0) for i in range(n)]


def filter_boxes(
    boxes: BoundingBoxes,
    *,
    orig_areas: list[float] | None = None,
    min_area: float = 0.0,
    min_visibility: float = 0.0,
) -> BoundingBoxes:
    r"""Drop degenerate / too-small / too-occluded boxes (+ their labels).

    A box at row ``i`` is dropped when **any** of the following holds:

    * its post-transform area :math:`\le 0` (degenerate),
    * its post-transform area :math:`<` ``min_area`` (absolute pixels),
    * ``orig_areas`` is supplied **and** the visible fraction
      :math:`a_i / o_i <` ``min_visibility`` (where :math:`a_i, o_i`
      are the post-transform and original areas).

    The companion ``labels`` tensor — when present — is trimmed in
    lock-step, so downstream loss / NMS code never sees a label
    without a matching box.

    Parameters
    ----------
    boxes : BoundingBoxes
        Boxes (and optional labels) after the transform whose effect
        you want to filter.
    orig_areas : list of float, optional
        Per-row areas captured *before* the transform was applied —
        typically the output of :func:`box_areas` on the pre-pipeline
        boxes, recorded by :class:`~lucid.utils.transforms.Compose`
        with ``bbox_params``.  When omitted, ``min_visibility`` is
        ignored (only ``min_area`` / degeneracy are checked).
    min_area : float, optional, default=0.0
        Absolute-pixel minimum area; rows below this are dropped.
    min_visibility : float, optional, default=0.0
        Minimum fraction of the original area that must still be
        visible.  Requires ``orig_areas``.

    Returns
    -------
    BoundingBoxes
        A new instance carrying only the surviving rows + their
        labels.  Format, canvas, and the labels-or-``None`` shape are
        all preserved.

    Examples
    --------
    Filter out a tiny box and a degenerate one (with labels):

    >>> import lucid
    >>> import lucid.utils.transforms as T
    >>> b = T.BoundingBoxes(
    ...     lucid.tensor([[0., 0., 30., 30.],   # area 900
    ...                   [0., 0.,  0.,  0.],   # degenerate
    ...                   [0., 0.,  2.,  2.]]), # area 4 (too small)
    ...     "xyxy", (100, 100),
    ...     labels=lucid.tensor([1.0, 2.0, 3.0]),
    ... )
    >>> out = filter_boxes(b, min_area=10.0)
    >>> out.labels.numpy().tolist()
    [1.0]
    """
    cur = box_areas(boxes)
    keep: list[int] = []
    for i, area in enumerate(cur):
        if area <= 0.0 or area < min_area:
            continue
        if (
            orig_areas is not None
            and i < len(orig_areas)
            and orig_areas[i] > 0.0
            and (area / orig_areas[i]) < min_visibility
        ):
            continue
        keep.append(i)

    if not keep:
        empty_lbl = boxes.labels[:0] if boxes.labels is not None else None
        return BoundingBoxes(boxes.data[:0], boxes.format, boxes.canvas_size, empty_lbl)

    data = _cat([boxes.data[i : i + 1] for i in keep], 0)
    labels = _cat([boxes.labels[i : i + 1] for i in keep], 0) if boxes.labels is not None else None
    return BoundingBoxes(data, boxes.format, boxes.canvas_size, labels)
