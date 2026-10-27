"""
Grid sampling the engine kernel does not cover, assembled from ops.

The engine's ``grid_sample`` reads a 4-D input with a bilinear or nearest
filter and pads with zeros or the border; its ``affine_grid`` builds 2-D
grids.  Everything else :func:`~lucid.nn.functional.grid_sample` and
:func:`~lucid.nn.functional.affine_grid` accept is built here:

- **reflection padding** of a 4-D input, as a grid the kernel then reads
  with border padding (:func:`reflect_grid`);
- **5-D (volumetric) input**, trilinear or nearest;
- **bicubic**, sixteen taps of the cubic convolution kernel;
- **3-D affine grids**.

Each follows the reference framework's coordinate rules step by step —
unnormalise, reflect, clip, then filter — so a sample lands on the same
taps with the same weights, far outside the image and at non-finite
coordinates too.  The reference has two sets of rules: its 4-D kernel
folds a reflection by twice the span in floating point, and its 5-D
kernel counts the flips in a 32-bit ``int``.  The two agree inside the
image and part only far away, so each rank follows its own
(``volumetric``).  A position becomes an index as the engine's
``resample::index_of`` makes it one: NaN reads sample 0, and anything
beyond the axis reads nothing.

The composites are differentiable in both the input (through ``gather``)
and the grid (through the filter weights, the reflection's ``±1`` slope
and the clip's ``0`` / ``1`` slope), the same two gradients the kernel
provides.  A half input is sampled in float32 and rounded once
(:func:`opmath`).
"""

import itertools
import math
from typing import TYPE_CHECKING

import lucid as _lucid
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid._dtype import _float32_or_wider

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


_BICUBIC_A = -0.75

# The reference's 5-D kernel counts reflection flips in a 32-bit int, and
# a count that does not fit saturates there — an odd count.
_INT32_LIMIT = 2.0**31


def cubic_coefficients[T: (float, Tensor)](t: T) -> tuple[T, T, T, T]:
    """Weights of the four taps a bicubic sample reads, at offset ``t``.

    The cubic convolution kernel with :math:`A = -0.75`, which is the
    coefficient the reference implementations use.  The weights are not
    normalised — bicubic is allowed to overshoot, and clipping it here
    would make it a different filter.  ``t`` may be a number (one sample)
    or a tensor (one weight per sample); the polynomial is the same.

    Parameters
    ----------
    t : float or Tensor
        Fractional distance from the second of the four taps, in ``[0, 1)``.

    Returns
    -------
    tuple of four float or Tensor
        Weights for taps at ``-1, 0, +1, +2`` relative to ``floor``.
    """
    a = _BICUBIC_A

    def near(x: T) -> T:
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0

    def far(x: T) -> T:
        return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a

    other = 1.0 - t
    return far(t + 1.0), near(t), near(other), far(other + 1.0)


def _distance(position: Tensor, low: float) -> Tensor:
    """``|position - low|``, with the slope the reference gives it.

    The slope is ``-1`` below ``low`` and ``+1`` everywhere else — at a NaN
    position too, where ``abs`` would have none.  The reference's reflection
    takes its sign from ``position - low < 0``, which is false for NaN, so a
    NaN coordinate still passes its grid gradient through.

    Parameters
    ----------
    position : Tensor
        Pixel-space positions.
    low : float
        The lower mirror.

    Returns
    -------
    Tensor
        The distances, same shape.
    """
    shifted = position - low
    return _lucid.where(shifted < 0.0, -shifted, shifted)


def _reflect(position: Tensor, low: float, span: float, volumetric: bool) -> Tensor:
    """Mirror ``position`` into ``[low, low + span]`` by the reference's rule.

    The 4-D reference reduces the distance from ``low`` by whole double
    spans — the count truncated in floating point, so the remainder is the
    rounded one — and reads a remainder in the far half back from the far
    end.  The 5-D reference splits the distance into a remainder
    (``fmod``) and a count of whole spans, and reads the remainder from the
    far end when the count is odd.  It counts in a 32-bit ``int``: a count
    that does not fit saturates, which makes it odd, and a NaN count is
    zero.  Its remainder is :func:`lucid.fmod`'s here, which is exact, so a
    distance past ``2**24`` spans reflects as the reference's does.  Both
    rules agree inside the image; the slope is ``±1`` either way.

    Parameters
    ----------
    position : Tensor
        Pixel-space positions.
    low : float
        Lower mirror.
    span : float
        Distance to the upper mirror; positive.
    volumetric : bool
        Follow the 5-D reference's rule instead of the 4-D one's.

    Returns
    -------
    Tensor
        The reflected positions, same shape.
    """
    distance = _distance(position, low)
    if volumetric:
        extra = _lucid.fmod(distance, span)
        flips = (distance / span).floor()
        odd = (flips >= _INT32_LIMIT) | (_lucid.fmod(flips, 2.0) == 1.0)
        return _lucid.where(odd, span - extra, extra) + low
    twice = 2.0 * span
    extra = distance - (distance / twice).trunc() * twice
    back = twice - extra
    return _lucid.where(extra > back, back, extra) + low


def _clip(position: Tensor, size: int) -> Tensor:
    """Clamp ``position`` onto ``[0, size - 1]``; NaN passes through.

    The slope is ``1`` strictly inside and ``0`` on or past either edge —
    the reference counts the edge as clipped — and ``1`` at NaN.

    Parameters
    ----------
    position : Tensor
        Pixel-space positions.
    size : int
        Length of the axis.

    Returns
    -------
    Tensor
        The clamped positions, same shape.
    """
    last = float(size - 1)
    zero = _lucid.where(_lucid.isfinite(position), position, 0.0) * 0.0
    inner = _lucid.where(position >= last, zero + last, position)
    return _lucid.where(position <= 0.0, zero, inner)


def _constant_zero(position: Tensor) -> Tensor:
    """``0`` at every position, NaN included, still in the grid's graph.

    Parameters
    ----------
    position : Tensor
        Any positions.

    Returns
    -------
    Tensor
        Zeros of the same shape and dtype, with a zero slope.
    """
    return _lucid.where(_lucid.isfinite(position), position, 0.0) * 0.0


def _unnormalize(coord: Tensor, size: int, align_corners: bool, volumetric: bool) -> Tensor:
    """Map a normalised ``[-1, 1]`` coordinate onto pixel positions of an axis.

    The 4-D reference scales by ``size / 2`` and the 5-D one contracts
    ``(coord + 1) * size - 1`` in one fused multiply-add before halving;
    they round alike inside the image and part where ``coord * size``
    reaches ``2**23`` or overflows.

    Parameters
    ----------
    coord : Tensor
        Normalised coordinates.
    size : int
        Length of the axis sampled.
    align_corners : bool
        ``True``: ``±1`` are the centres of the end pixels; ``False``:
        their outer edges.
    volumetric : bool
        Follow the 5-D reference's arithmetic instead of the 4-D one's.

    Returns
    -------
    Tensor
        Pixel-space coordinates, same shape.
    """
    if align_corners:
        return (coord + 1.0) * ((size - 1) / 2)
    if volumetric:
        c = _unwrap(coord + 1.0)
        s = _C_engine.full([], float(size), c.dtype, c.device)
        neg_one = _C_engine.full([], -1.0, c.dtype, c.device)
        return _wrap(_C_engine.fma(c, s, neg_one)) * 0.5
    return (coord + 1.0) * (size / 2) - 0.5


def _pad(
    position: Tensor, size: int, padding_mode: str, align_corners: bool, volumetric: bool
) -> Tensor:
    """Bring a pixel-space position inside the axis as ``padding_mode`` says.

    ``zeros`` leaves it where it is (the read is masked instead),
    ``border`` clips it, and ``reflection`` mirrors it about the image's
    edges — the end pixels' centres with ``align_corners``, their outer
    edges without — and then clips, as the reference does.

    A NaN position stays NaN under the 4-D rules.  Under the 5-D rules it
    clips to the last sample: the reference's 5-D forward does that
    (``std::min(max, std::max(NaN, 0))``), while its backward keeps NaN and
    differentiates at sample 0.  The two cannot be one graph, so the
    composite follows the forward and its gradients follow that same
    position.  For a NaN coordinate under border or reflection padding, and
    an infinite one under reflection, ``dx`` and the grid gradient are
    therefore the forward's, not the reference's.

    Parameters
    ----------
    position : Tensor
        Pixel-space positions.
    size : int
        Length of the axis.
    padding_mode : str
        ``"zeros"``, ``"border"`` or ``"reflection"``.
    align_corners : bool
        Corner convention the position was unnormalised with.
    volumetric : bool
        Follow the 5-D reference's rules instead of the 4-D one's.

    Returns
    -------
    Tensor
        The padded positions, same shape.
    """
    if padding_mode == "zeros":
        return position
    if padding_mode == "reflection":
        if not align_corners:
            position = _reflect(position, -0.5, float(size), volumetric)
        elif size > 1:
            position = _reflect(position, 0.0, float(size - 1), volumetric)
        else:
            # One pixel, mirrored at its own centre: every read is it.
            position = _constant_zero(position)
    clipped = _clip(position, size)
    if volumetric:
        clipped = _lucid.where(_lucid.isnan(clipped), float(size - 1), clipped)
    return clipped


def _index_of(position: Tensor) -> Tensor:
    """A whole-numbered position as the sample it reads: NaN reads sample 0.

    Parameters
    ----------
    position : Tensor
        Floored or rounded positions.

    Returns
    -------
    Tensor
        The same positions with NaN replaced by ``0``; infinite and far
        ones stay, and :func:`_gather` reads nothing there.
    """
    return _lucid.where(_lucid.isnan(position), 0.0, position)


def _gather(flat: Tensor, coords: list[Tensor], sizes: list[int]) -> tuple[Tensor, Tensor]:
    """Read ``flat`` at integral pixel coordinates; reads off the image are 0.

    A read off the image takes nothing from ``flat`` — not even a NaN or
    infinite value times zero — and sends it no gradient.

    Parameters
    ----------
    flat : Tensor
        Input with its spatial axes flattened, ``(N, C, S)``.
    coords : list of Tensor
        One ``(N, P)`` tensor of whole-numbered, NaN-free coordinates per
        spatial axis, fastest-varying (``x``) first.
    sizes : list of int
        The matching axis lengths.

    Returns
    -------
    tuple of Tensor
        ``(N, C, P)`` samples, zero where any coordinate is off its axis,
        and the ``(N, P)`` mask of the reads that landed on it.
    """
    n, c, _ = flat.shape
    inside = _lucid.ones_like(coords[0], dtype=_lucid.bool)
    for coord, size in zip(coords, sizes, strict=True):
        inside = inside & (coord >= 0.0) & (coord <= float(size - 1))
    index = _lucid.zeros_like(coords[0], dtype=_lucid.int64)
    stride = 1
    for coord, size in zip(coords, sizes, strict=True):
        index = index + _lucid.where(inside, coord, 0.0).to(_lucid.int64) * stride
        stride *= size
    points = int(index.shape[1])
    values = _lucid.gather(flat, 2, index.unsqueeze(1).expand(n, c, points))
    return _lucid.where(inside.unsqueeze(1), values, 0.0), inside


def _total(terms: list[Tensor]) -> Tensor:
    """Add up per-tap contributions, in tap order.

    Parameters
    ----------
    terms : list of Tensor
        Non-empty list of same-shape tensors.

    Returns
    -------
    Tensor
        Their sum.
    """
    total = terms[0]
    for term in terms[1:]:
        total = total + term
    return total


def _linear(flat: Tensor, coords: list[Tensor], sizes: list[int]) -> Tensor:
    """Trilinear (or bilinear): the ``2**k`` surrounding pixels, weighted.

    A corner off the image adds nothing, its weight included: at an
    infinite position the weights are NaN, and the reference skips such a
    corner rather than multiplying it by zero.  Every factor of such a
    corner's weight is zeroed before the product, not the product after
    it: a product's derivative multiplies by the other factors, so one NaN
    factor would turn the zero gradient of a skipped corner into NaN.  The
    corners are summed in the reference's order, the slowest axis
    outermost.

    A position at or past ``2**63`` reads nothing, as the engine's
    ``resample::next_index`` keeps it: the reference's 5-D kernel adds one
    to the saturated index, overflows, and reads its upper corner at
    sample 0.

    Parameters
    ----------
    flat : Tensor
        ``(N, C, S)`` input.
    coords : list of Tensor
        Padded pixel-space coordinates, one ``(N, P)`` per axis, ``x`` first.
    sizes : list of int
        Axis lengths.

    Returns
    -------
    Tensor
        ``(N, C, P)`` samples.
    """
    lower: list[Tensor] = []
    fractions: list[Tensor] = []
    for coord in coords:
        base = coord.floor()
        lower.append(_index_of(base))
        fractions.append(coord - base)
    terms = []
    for corner in itertools.product((0, 1), repeat=len(coords)):
        uppers = corner[::-1]
        taps = [low + float(upper) for low, upper in zip(lower, uppers, strict=True)]
        values, inside = _gather(flat, taps, sizes)
        weight = _lucid.ones_like(coords[0])
        for upper, frac in zip(uppers, fractions, strict=True):
            weight = weight * _lucid.where(inside, frac if upper else 1.0 - frac, 0.0)
        terms.append(values * weight.unsqueeze(1))
    return _total(terms)


def _nearest(flat: Tensor, coords: list[Tensor], sizes: list[int]) -> Tensor:
    """Nearest: the one pixel each coordinate rounds to (half to even).

    The sample does not move with the grid, but it still belongs to the
    grid's graph, as the kernel's nearest path does: the grid gets a zero
    gradient, and a caller differentiating only the grid gets that rather
    than an output that does not require grad.  ``round`` has zero slope
    and the clip pins every value to 0 (infinite ones too, which a
    product with 0 would turn into NaN), so the weight is exactly 1.

    Takes and returns what ``_linear`` does: the ``(N, C, S)`` input,
    padded coordinates (``x`` first) with their axis lengths, and
    ``(N, C, P)`` samples.
    """
    nearest = [_index_of(coord.round()) for coord in coords]
    weight = _total([index.clip(0.0, 0.0) for index in nearest]) + 1.0
    values, _ = _gather(flat, nearest, sizes)
    return values * weight.unsqueeze(1)


def _bicubic(
    flat: Tensor, coords: list[Tensor], sizes: list[int], padding_mode: str, align_corners: bool
) -> Tensor:
    """Bicubic: a 4 x 4 neighbourhood weighted by the cubic kernel.

    The coordinates are *not* padded first — clipping ``-0.1`` to ``0``
    would move ``floor`` and every tap with it.  Each tap is padded on
    its own instead, as the reference's ``get_value_bounded`` does, and
    the weights come from the unpadded fraction.  The grid gradient
    flows through the weights only; the taps are integral.  As in the
    reference, the four taps of each row are blended first and the rows
    after, and a tap off the image is a zero that still multiplies its
    weight (a NaN weight makes the sample NaN) but takes no gradient.

    Parameters
    ----------
    flat : Tensor
        ``(N, C, H * W)`` input.
    coords : list of Tensor
        Unpadded pixel-space ``[x, y]``, each ``(N, P)``.
    sizes : list of int
        ``[W, H]``.
    padding_mode : str
        Padding applied to each tap.
    align_corners : bool
        Corner convention, for reflection.

    Returns
    -------
    Tensor
        ``(N, C, P)`` samples.
    """
    taps: list[list[Tensor]] = []
    weights: list[list[Tensor]] = []
    for coord, size in zip(coords, sizes, strict=True):
        base = coord.floor()
        weights.append([w.unsqueeze(1) for w in cubic_coefficients(coord - base)])
        taps.append(
            [
                _index_of(_pad(base + float(offset), size, padding_mode, align_corners, False))
                for offset in (-1, 0, 1, 2)
            ]
        )
    rows = []
    for j in range(4):
        reads = [_gather(flat, [taps[0][i], taps[1][j]], sizes)[0] for i in range(4)]
        rows.append(_total([w * v for w, v in zip(weights[0], reads, strict=True)]))
    return _total([w * row for w, row in zip(weights[1], rows, strict=True)])


_MODES = ("bilinear", "nearest", "bicubic")
_PADDINGS = ("zeros", "border", "reflection")


def check_options(x: Tensor, mode: str, padding_mode: str) -> None:
    """Refuse, by name, a ``grid_sample`` setting it cannot compute.

    Parameters
    ----------
    x : Tensor
        The input; its rank decides which modes apply.
    mode : str
        Requested filter.
    padding_mode : str
        Requested padding.

    Raises
    ------
    ValueError
        Unknown ``mode`` or ``padding_mode``, an input that is not 4-D or
        5-D, or ``"bicubic"`` with a 5-D input (the reference has no
        tricubic filter either).
    """
    if mode not in _MODES:
        raise ValueError(f"grid_sample: unsupported mode {mode!r}; expected one of {_MODES}")
    if padding_mode not in _PADDINGS:
        raise ValueError(
            f"grid_sample: unsupported padding_mode {padding_mode!r}; expected one of {_PADDINGS}"
        )
    if x.ndim not in (4, 5):
        raise ValueError(
            "grid_sample: expected a 4-D (N, C, H, W) or 5-D (N, C, D, H, W) "
            f"input, got shape {tuple(x.shape)}"
        )
    if x.ndim == 5 and mode == "bicubic":
        raise ValueError("grid_sample: mode 'bicubic' supports only 4-D input")


def _as_real(t: Tensor) -> Tensor:
    """Promote an integral tensor to float32, as the engine kernel does.

    Parameters
    ----------
    t : Tensor
        Input or grid.

    Returns
    -------
    Tensor
        ``t`` if it is already floating point, otherwise a float32 copy.
    """
    return t if t.is_floating_point() else t.to(_lucid.float32)


def opmath(x: Tensor, grid: Tensor) -> tuple[Tensor, Tensor]:
    """``x`` and ``grid`` in the dtype the composite parts sample in.

    A half pair is widened to float32, and the caller rounds the result
    back once — as the reference's MPS kernels sample one.  Coordinates,
    weights and sums each rounded to float16 put a bicubic sample about
    2e-2 off the float32 answer, where a single rounding costs 5e-4.
    float32 and float64 are their own; a pair of two dtypes is returned as
    it is, for the caller's check or the kernel to refuse.

    Parameters
    ----------
    x : Tensor
        Floating input.
    grid : Tensor
        Floating grid.

    Returns
    -------
    tuple of Tensor
        The two operands, widened or as they were.
    """
    if x.dtype != grid.dtype:
        return x, grid
    work = _float32_or_wider(x.dtype)
    return x.to(work), grid.to(work)


def _check_operands(x: Tensor, grid: Tensor) -> None:
    """Refuse a grid that does not fit the input, before any read.

    Parameters
    ----------
    x : Tensor
        ``(N, C, *spatial)`` input.
    grid : Tensor
        Expected ``(N, *out_spatial, k)`` with ``k = len(spatial)``.

    Raises
    ------
    ValueError
        Shapes or devices do not match.
    TypeError
        Dtypes do not match.
    """
    rank = x.ndim - 2
    if grid.ndim != x.ndim or int(grid.shape[-1]) != rank:
        raise ValueError(
            f"grid_sample: a {x.ndim}-D input needs a {x.ndim}-D grid whose "
            f"last axis has {rank} coordinates, got grid of shape {tuple(grid.shape)}"
        )
    if int(grid.shape[0]) != int(x.shape[0]):
        raise ValueError(
            f"grid_sample: batch size mismatch, input {tuple(x.shape)} vs grid {tuple(grid.shape)}"
        )
    if x.device != grid.device:
        raise ValueError(f"grid_sample: input is on {x.device} but grid is on {grid.device}")
    if x.dtype != grid.dtype:
        raise TypeError(
            f"grid_sample: input and grid must have the same dtype, got {x.dtype} and {grid.dtype}"
        )


def composite_grid_sample(
    x: Tensor, grid: Tensor, mode: str, padding_mode: str, align_corners: bool
) -> Tensor:
    """Sample a 4-D or 5-D input at ``grid`` without the engine kernel.

    Parameters
    ----------
    x : Tensor
        ``(N, C, H, W)`` or ``(N, C, D, H, W)``.
    grid : Tensor
        ``(N, H_out, W_out, 2)`` or ``(N, D_out, H_out, W_out, 3)``.
    mode : str
        ``"bilinear"``, ``"nearest"`` or (4-D only) ``"bicubic"``.
    padding_mode : str
        ``"zeros"``, ``"border"`` or ``"reflection"``.
    align_corners : bool
        Corner convention.

    Returns
    -------
    Tensor
        ``(N, C, *out_spatial)`` samples.
    """
    x = _as_real(x)
    grid = _as_real(grid)
    _check_operands(x, grid)
    out_dtype = x.dtype
    x, grid = opmath(x, grid)
    n, c = int(x.shape[0]), int(x.shape[1])
    spatial = [int(s) for s in x.shape[2:]]
    out_spatial = [int(s) for s in grid.shape[1:-1]]
    points = math.prod(out_spatial)
    flat = x.reshape(n, c, math.prod(spatial))
    volumetric = x.ndim == 5
    # Grid components are (x, y[, z]) — fastest axis first, the reverse
    # of the input's (D, H, W).
    sizes = spatial[::-1]
    components = _lucid.unbind(grid.reshape(n, points, len(sizes)), -1)
    coords = [
        _unnormalize(component, size, align_corners, volumetric)
        for component, size in zip(components, sizes, strict=True)
    ]
    if mode == "bicubic":
        out = _bicubic(flat, coords, sizes, padding_mode, align_corners)
    else:
        coords = [
            _pad(coord, size, padding_mode, align_corners, volumetric)
            for coord, size in zip(coords, sizes, strict=True)
        ]
        sample = _nearest if mode == "nearest" else _linear
        out = sample(flat, coords, sizes)
    return out.reshape(n, c, *out_spatial).to(out_dtype)


def _normalize(position: Tensor, size: int, align_corners: bool) -> Tensor:
    """The grid coordinate whose pixel position along an axis is ``position``.

    Parameters
    ----------
    position : Tensor
        Pixel positions in ``[0, size - 1]``.
    size : int
        Length of the axis.
    align_corners : bool
        Corner convention.

    Returns
    -------
    Tensor
        Normalised coordinates, same shape.
    """
    if not align_corners:
        return (position + 0.5) * (2.0 / size) - 1.0
    if size > 1:
        return position * (2.0 / (size - 1)) - 1.0
    # Every coordinate reads the one pixel; 0 is the finite one to hand over.
    return _constant_zero(position)


def reflect_grid(x: Tensor, grid: Tensor, align_corners: bool, nearest: bool) -> Tensor:
    """The grid the 4-D kernel reads with border padding to sample with reflection.

    Each coordinate is unnormalised, reflected and clipped in pixel space
    with the reference's 4-D arithmetic, so a coordinate far outside the
    image lands where the reference's does, and is then handed back in
    normalised form.  A coordinate the padding leaves where it was is
    handed over untouched, so the kernel reads it exactly; one the padding
    moved goes through :func:`_normalize`, whose round trip can be a unit
    in the last place off — so under ``nearest`` it is rounded to its
    pixel first, and the kernel's own rounding cannot tip it.

    The grid gradient is the chain the reference applies: the unnormalise
    slope, the reflection's ``±1`` (``+1`` at NaN) and the clip's ``0`` /
    ``1``.

    Parameters
    ----------
    x : Tensor
        The ``(N, C, H, W)`` input; only its spatial size is read.
    grid : Tensor
        ``(N, H_out, W_out, 2)`` grid; an integral one is promoted to
        float32.
    align_corners : bool
        Corner convention.
    nearest : bool
        The kernel will round the positions rather than blend them.

    Returns
    -------
    Tensor
        A grid of the same shape whose coordinates all read inside the image;
        a grid of any other shape is returned as it is, for the kernel to
        refuse by name.
    """
    grid = _as_real(grid)
    if x.ndim != 4 or grid.ndim != 4 or int(grid.shape[-1]) != 2:
        return grid
    sizes = (int(x.shape[3]), int(x.shape[2]))
    reflected = []
    for coord, size in zip(_lucid.unbind(grid, -1), sizes, strict=True):
        position = _unnormalize(coord, size, align_corners, False)
        padded = _pad(position, size, "reflection", align_corners, False)
        if nearest:
            padded = padded.round()
        moved = _normalize(padded, size, align_corners)
        reflected.append(_lucid.where(padded == position, coord, moved))
    return _lucid.stack(reflected, -1)


def _base_axis(steps: int, align_corners: bool, like: Tensor) -> Tensor:
    """Normalised centres of ``steps`` output pixels along one axis.

    Parameters
    ----------
    steps : int
        Axis length.
    align_corners : bool
        ``True`` spans ``[-1, 1]`` end-centre to end-centre; ``False``
        pulls the centres in by half a pixel.
    like : Tensor
        Supplies dtype and device.

    Returns
    -------
    Tensor
        ``(steps,)`` coordinates; a single pixel sits at ``0``.
    """
    if steps <= 1:
        return _lucid.zeros(steps, dtype=like.dtype, device=like.device)
    axis = _lucid.linspace(-1.0, 1.0, steps, dtype=like.dtype, device=like.device)
    return axis if align_corners else axis * ((steps - 1) / steps)


def affine_grid_3d(theta: Tensor, size: list[int], align_corners: bool) -> Tensor:
    """Volumetric affine grid: ``theta @ [x, y, z, 1]`` at every output voxel.

    Parameters
    ----------
    theta : Tensor
        ``(N, 3, 4)`` affine matrices.
    size : list of int
        ``(N, C, D, H, W)`` target size.
    align_corners : bool
        Corner convention, as for :func:`composite_grid_sample`.

    Returns
    -------
    Tensor
        ``(N, D, H, W, 3)`` normalised sampling coordinates.

    Raises
    ------
    ValueError
        ``theta`` is not ``(N, 3, 4)`` for the ``N`` in ``size``.
    """
    n, _, d, h, w = size
    if tuple(int(s) for s in theta.shape) != (n, 3, 4):
        raise ValueError(
            f"affine_grid: a 5-D size needs theta of shape ({n}, 3, 4), got {tuple(theta.shape)}"
        )
    theta = _as_real(theta)
    xs = _base_axis(w, align_corners, theta).reshape(1, 1, w).expand(d, h, w)
    ys = _base_axis(h, align_corners, theta).reshape(1, h, 1).expand(d, h, w)
    zs = _base_axis(d, align_corners, theta).reshape(d, 1, 1).expand(d, h, w)
    base = _lucid.stack([xs, ys, zs, _lucid.ones_like(xs)], -1).reshape(1, d * h * w, 4)
    return _lucid.matmul(base, theta.transpose(1, 2)).reshape(n, d, h, w, 3)
