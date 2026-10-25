"""
The bodies every ``nn.functional`` pool and pooling module runs.

``pooling.py`` holds the public signatures and their documentation; the
work lives here once: the engine call per rank, the unbatched round trip,
the argmax a caller asks for with ``return_indices``, and the fallback for
adaptive windows the engine cannot express.  The arguments arrive already
read by :mod:`lucid.nn.functional._spatial_args`.
"""

import itertools
import math
import struct
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Literal, overload

import lucid as _lucid
from lucid import _promotion
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid._ops._autocast import _autocast_entry, _autocast_suspended
from lucid._tensor.tensor import Tensor
from lucid.nn.functional._spatial_args import (
    _PoolGeometry,
    _add_batch_axis,
    _drop_batch_axis,
    _ntuple,
    _require_positive,
)

# The composite pools with a row in the autocast policy table (FP32 on the
# CPU, from the reference's CPU list).  The others — adaptive pools of rank
# 1 and 2, ``max_unpool1d`` — have none and keep their input's dtype.
_POOL_ROWS = frozenset(
    {
        "adaptive_avg_pool3d",
        "adaptive_max_pool3d",
        "fractional_max_pool2d",
        "fractional_max_pool3d",
        "max_unpool2d",
        "max_unpool3d",
    }
)


@contextmanager
def _pool_autocast(op: str, x: Tensor) -> Iterator[Tensor]:
    """``x`` at ``op``'s row, or kept when it has none; autocast off inside."""
    if op not in _POOL_ROWS:
        with _autocast_suspended(_unwrap(x).device):
            yield x
        return
    with _autocast_entry(op, _unwrap(x)) as (xi,):
        yield _wrap(xi)


_MAX_POOL: dict[int, Callable[..., _C_engine.TensorImpl]] = {
    1: _C_engine.nn.max_pool1d,
    2: _C_engine.nn.max_pool2d,
    3: _C_engine.nn.max_pool3d,
}
_AVG_POOL: dict[int, Callable[..., _C_engine.TensorImpl]] = {
    1: _C_engine.nn.avg_pool1d,
    2: _C_engine.nn.avg_pool2d,
    3: _C_engine.nn.avg_pool3d,
}
_ADAPTIVE_POOL: dict[tuple[str, int], Callable[..., _C_engine.TensorImpl]] = {
    ("mean", 1): _C_engine.nn.adaptive_avg_pool1d,
    ("mean", 2): _C_engine.nn.adaptive_avg_pool2d,
    ("mean", 3): _C_engine.nn.adaptive_avg_pool3d,
    ("max", 1): _C_engine.nn.adaptive_max_pool1d,
    ("max", 2): _C_engine.nn.adaptive_max_pool2d,
    ("max", 3): _C_engine.nn.adaptive_max_pool3d,
}


def _pooled_extent(size: int, k: int, s: int, p: int, d: int, ceil_mode: bool) -> int:
    """How many windows fit along one axis, by the reference's arithmetic.

    With ``ceil_mode`` a last partial window is kept only if it starts
    inside the input or its left padding.
    """
    span = size + 2 * p - d * (k - 1) - 1
    out = (span + (s - 1 if ceil_mode else 0)) // s + 1
    if ceil_mode and (out - 1) * s >= size + p:
        out -= 1
    return out


def _digit(flat: Tensor, inner: int, extent: int) -> Tensor:
    """One axis's coordinate of a non-negative row-major flat index.

    ``(flat // inner) % extent``, spelled without ``%``: the two are equal
    for a non-negative ``flat``, and ``lucid.compile`` has no emitter for
    ``remainder`` yet (LCD-444), so a ``%`` here sent every compiled max
    pool that returns its indices to eager.
    """
    q = flat // inner
    return q - (q // extent) * extent


def _pool_windows(x: Tensor, g: _PoolGeometry, out_spatial: tuple[int, ...]) -> Tensor:
    """Every pooling window of ``x`` as ``(*lead, *out_spatial, prod(kernel))``.

    Padding is ``-inf``, so a maximum never comes from it.  A window spans
    ``dilation * (kernel - 1) + 1`` inputs and keeps every ``dilation``-th.
    """
    kernel, stride, padding, dilation = g
    n = len(kernel)
    lead = x.ndim - n
    spatial = tuple(int(s) for s in x.shape[lead:])
    span = tuple(dl * (k - 1) + 1 for k, dl in zip(kernel, dilation, strict=True))
    pads: list[int] = []
    for d in reversed(range(n)):
        # Enough on the right to hold every window, including the partial
        # last one ``ceil_mode`` keeps.
        right = (out_spatial[d] - 1) * stride[d] + span[d] - spatial[d] - padding[d]
        pads += [padding[d], max(right, 0)]
    windows = _lucid.nn.functional.pad(x, tuple(pads), value=float("-inf"))
    for d in range(n):
        windows = windows.unfold(lead + d, span[d], stride[d])
        windows = windows.narrow(lead + d, 0, out_spatial[d])
    for d in range(n):
        if dilation[d] > 1:
            taps = _lucid.arange(
                0, span[d], dilation[d], dtype=_lucid.int64, device=x.device
            )
            windows = windows.index_select(x.ndim + d, taps)
    return windows.reshape(*x.shape[:lead], *out_spatial, math.prod(kernel))


def _window_argmax_indices(
    x: Tensor, g: _PoolGeometry, out_spatial: tuple[int, ...], windows: Tensor
) -> Tensor:
    """Where each pooling window's maximum sits, as the reference reports it.

    Per leading ``(N, C)`` slice, a flat index into the input's unpadded
    spatial extent — the form :func:`max_unpool2d` and its siblings take.
    ``windows`` is :func:`_pool_windows` of ``x``; a tie goes to the first
    element of the window in row-major order, as the reference's does.
    """
    kernel, stride, padding, dilation = g
    n = len(kernel)
    spatial = tuple(int(s) for s in x.shape[x.ndim - n :])
    local = _lucid.argmax(windows, dim=-1)
    index = None
    for d in range(n):
        offset = _digit(local, math.prod(kernel[d + 1 :]), kernel[d]) * dilation[d]
        shape = [1] * n
        shape[d] = out_spatial[d]
        origin = _lucid.arange(out_spatial[d], dtype=_lucid.int64, device=x.device)
        coord = offset + (origin * stride[d] - padding[d]).reshape(*shape)
        index = coord if index is None else index * spatial[d] + coord
    assert index is not None
    return index


def _adaptive_ranges(n_in: int, n_out: int) -> list[tuple[int, int]]:
    """The reference's adaptive windows: ``[floor(i·in/out), ceil((i+1)·in/out))``."""
    return [((i * n_in) // n_out, -(-(i + 1) * n_in // n_out)) for i in range(n_out)]


def _adaptive_cells(
    x: Tensor, out_spatial: tuple[int, ...]
) -> list[tuple[tuple[tuple[int, int], ...], Tensor]]:
    """Every adaptive window of ``x``, row-major, with its ``(lo, hi)`` per axis."""
    n = len(out_spatial)
    lead = x.ndim - n
    spatial = tuple(int(s) for s in x.shape[lead:])
    cells = []
    for cell in itertools.product(
        *(_adaptive_ranges(i, o) for i, o in zip(spatial, out_spatial, strict=True))
    ):
        window = x
        for d, (lo, hi) in enumerate(cell):
            window = window.narrow(lead + d, lo, hi - lo)
        cells.append((cell, window))
    return cells


def _adaptive_argmax_indices(x: Tensor, out_spatial: tuple[int, ...]) -> Tensor:
    """:func:`_window_argmax_indices` for adaptive pooling's uneven windows."""
    n = len(out_spatial)
    lead = x.ndim - n
    spatial = tuple(int(s) for s in x.shape[lead:])
    if all(i % o == 0 for i, o in zip(spatial, out_spatial, strict=True)):
        kernel = tuple(i // o for i, o in zip(spatial, out_spatial, strict=True))
        g = _PoolGeometry(kernel, kernel, (0,) * n, (1,) * n)
        return _window_argmax_indices(
            x, g, out_spatial, _pool_windows(x, g, out_spatial)
        )
    found: list[Tensor] = []
    for cell, window in _adaptive_cells(x, out_spatial):
        extent = [hi - lo for lo, hi in cell]
        local = _lucid.argmax(window.reshape(*x.shape[:lead], -1), dim=-1)
        index = None
        for d in range(n):
            coord = _digit(local, math.prod(extent[d + 1 :]), extent[d]) + cell[d][0]
            index = coord if index is None else index * spatial[d] + coord
        assert index is not None
        found.append(index)
    return _lucid.stack(found, dim=-1).reshape(*x.shape[:lead], *out_spatial)


def _adaptive_pool_python(
    x: Tensor, output_size: tuple[int, ...], reduce: str
) -> Tensor:
    """Engine fallback for adaptive pooling with non-divisible sizes.

    Reduces each window ``[floor(i·in/out), ceil((i+1)·in/out))`` per axis,
    the reference's contract.  Each window is a ``narrow`` and a reduction,
    engine ops both, so the result stays on the input's device with no
    host round-trip.  ``reduce`` is ``"mean"`` for average pooling and
    ``"max"`` for max pooling; the window geometry is the same for both.
    """
    n = len(output_size)
    lead = x.ndim - n
    dims = tuple(range(lead, x.ndim))
    values = [
        window.max(dim=dims) if reduce == "max" else window.mean(dim=dims)
        for _, window in _adaptive_cells(x, output_size)
    ]
    return _lucid.stack(values, dim=-1).reshape(*x.shape[:lead], *output_size)


@overload
def _max_pool(
    x: Tensor, g: _PoolGeometry, ceil_mode: bool, return_indices: Literal[False]
) -> Tensor: ...


@overload
def _max_pool(
    x: Tensor, g: _PoolGeometry, ceil_mode: bool, return_indices: bool
) -> Tensor | tuple[Tensor, Tensor]: ...


def _max_pool(
    x: Tensor, g: _PoolGeometry, ceil_mode: bool, return_indices: bool
) -> Tensor | tuple[Tensor, Tensor]:
    """The one body of ``max_pool{1,2,3}d`` and their modules.

    The engine kernel pools undilated windows.  A dilated window is
    gathered from :func:`_pool_windows` and reduced here; the indices come
    from the same windows either way.
    """
    n = len(g.kernel)
    xb, unbatched = _add_batch_axis(x, n)
    if any(d != 1 for d in g.dilation):
        if xb.ndim != n + 2:
            raise ValueError(
                f"max_pool{n}d: expected {n + 1}D or {n + 2}D input, got {x.ndim}D"
            )
        spatial = tuple(int(v) for v in xb.shape[2:])
        out_spatial = tuple(
            _pooled_extent(size, *axis, ceil_mode)
            for size, axis in zip(spatial, zip(*g, strict=True), strict=True)
        )
        if min(out_spatial) < 1:
            raise ValueError(
                f"max_pool{n}d: an input of {spatial} is too small for "
                f"kernel_size={g.kernel} with dilation={g.dilation}"
            )
        windows = _pool_windows(xb, g, out_spatial)
        out = windows.max(dim=-1).values
    else:
        out = _wrap(
            _MAX_POOL[n](_unwrap(xb), *g.kernel, *g.stride, *g.padding, ceil_mode)
        )
        if not return_indices:
            return _drop_batch_axis(out, unbatched)
        out_spatial = tuple(int(v) for v in out.shape[-n:])
        windows = _pool_windows(xb, g, out_spatial)
    if not return_indices:
        return _drop_batch_axis(out, unbatched)
    index = _window_argmax_indices(xb, g, out_spatial, windows)
    return _drop_batch_axis(out, unbatched), _drop_batch_axis(index, unbatched)


def _padded_window_sizes(
    spatial: tuple[int, ...], g: _PoolGeometry, out_spatial: tuple[int, ...]
) -> list[list[int]]:
    """Per axis, how many input-or-padding cells each window covers.

    The reference's ``count_include_pad`` divisor: a window that runs past
    the padding (``ceil_mode``) counts only what it covers.
    """
    sizes: list[list[int]] = []
    for size, o, (k, s, p, _) in zip(
        spatial, out_spatial, zip(*g, strict=True), strict=True
    ):
        sizes.append([min(i * s - p + k, size + p) - (i * s - p) for i in range(o)])
    return sizes


def _avg_pool(
    x: Tensor,
    g: _PoolGeometry,
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: int | None,
) -> Tensor:
    """The one body of ``avg_pool{1,2,3}d`` and their modules.

    ``divisor_override`` rescales the engine's padded average to the
    window's sum over the given divisor.
    """
    n = len(g.kernel)
    if divisor_override is not None and int(divisor_override) == 0:
        raise ValueError(f"avg_pool{n}d: divisor_override must not be zero")
    xb, unbatched = _add_batch_axis(x, n)
    cip = count_include_pad or divisor_override is not None
    out = _wrap(
        _AVG_POOL[n](_unwrap(xb), *g.kernel, *g.stride, *g.padding, ceil_mode, cip)
    )
    if divisor_override is not None:
        sizes = _padded_window_sizes(
            tuple(int(v) for v in xb.shape[2:]), g, tuple(int(v) for v in out.shape[2:])
        )
        if all(v == k for axis, k in zip(sizes, g.kernel, strict=True) for v in axis):
            out = out * (math.prod(g.kernel) / int(divisor_override))
        else:
            for d, axis in enumerate(sizes):
                shape = [1] * n
                shape[d] = len(axis)
                count = _lucid.tensor(axis, dtype=out.dtype, device=out.device)
                out = out * count.reshape(*shape)
            out = out / int(divisor_override)
    return _drop_batch_axis(out, unbatched)


def _lp_pool(x: Tensor, norm_type: float, g: _PoolGeometry, ceil_mode: bool) -> Tensor:
    """The one body of ``lp_pool{1,2,3}d`` and the ``LPPool*d`` modules.

    The reference's power average: the window sum of ``x ** p`` (not of
    ``|x| ** p`` — an odd ``p`` keeps each element's sign), recovered as
    the average pool times the window size, then its ``p``-th root.  A
    window whose sum is zero gets no gradient, where the root's own
    derivative is infinite; the reference does this with
    ``sign(s) * relu(|s|)``, and a ``where`` does it here, because the
    engine's ``relu`` gradient also drops a ``nan`` sum's gradient, which
    the reference passes on.  ``p = inf`` is max pooling of ``|x|``: the
    power average would take ``inf ** 0`` and give 1 everywhere.
    """
    p = float(norm_type)
    n = len(g.kernel)
    if p == math.inf:
        return _max_pool(_lucid.abs(x), g, ceil_mode, False)
    if not p > 0.0:
        raise ValueError(f"lp_pool{n}d: norm_type must be > 0, got {p}")
    pooled = _avg_pool(x**p, g, ceil_mode, True, None)
    summed = _lucid.where(pooled != 0, pooled, _lucid.zeros_like(pooled))
    return (summed * float(math.prod(g.kernel))) ** (1.0 / p)


def _adaptive_size(
    output_size: int | tuple[int | None, ...], spatial: tuple[int, ...], op: str
) -> tuple[int, ...]:
    """``output_size`` per spatial axis; ``None`` keeps that axis's extent.

    A size of 0 is an empty axis, as in the reference; a negative one is
    refused rather than reaching the window arithmetic, which divides by it.
    """
    n = len(spatial)
    output_size = _promotion.as_number(output_size)
    given = (output_size,) * n if isinstance(output_size, int) else tuple(output_size)
    if len(given) != n:
        raise ValueError(
            f"{op}: expected output_size to be a single integer or {n} entries, "
            f"got {output_size!r}"
        )
    sizes = tuple(
        e if v is None else int(v) for v, e in zip(given, spatial, strict=True)
    )
    if min(sizes) < 0:
        raise ValueError(
            f"{op}: elements of output_size must be greater than or equal to 0, "
            f"got {sizes}"
        )
    return sizes


def _adaptive_pool(
    x: Tensor, output_size: int | tuple[int | None, ...], n: int, reduce: str
) -> Tensor:
    """The one body of the ``adaptive_{avg,max}_pool{1,2,3}d`` values.

    The engine op reduces adaptive pooling to a fixed-kernel pool, which
    only expresses the case where every input axis divides its output
    evenly; :func:`_adaptive_pool_python` covers the rest.  Avg and max both
    route through here, so they cannot drift apart on which sizes they take.
    """
    op = f"adaptive_{'avg' if reduce == 'mean' else 'max'}_pool{n}d"
    with _pool_autocast(op, x) as x:
        xb, unbatched = _add_batch_axis(x, n)
        if xb.ndim != n + 2:
            raise ValueError(
                f"{op}: expected {n + 1}D or {n + 2}D input, got {x.ndim}D"
            )
        spatial = tuple(int(s) for s in xb.shape[-n:])
        sizes = _adaptive_size(output_size, spatial, op)
        if 0 in sizes:
            # No windows: an empty result, still on ``x``'s graph.
            empty = xb.narrow(xb.ndim - n + sizes.index(0), 0, 0)
            out = empty.reshape(*xb.shape[: xb.ndim - n], *sizes)
        elif all(i % o == 0 for i, o in zip(spatial, sizes, strict=True)):
            out = _wrap(_ADAPTIVE_POOL[(reduce, n)](_unwrap(xb), *sizes))
        else:
            out = _adaptive_pool_python(xb, sizes, reduce)
        return _drop_batch_axis(out, unbatched)


def _adaptive_max_pool(
    x: Tensor, output_size: int | tuple[int | None, ...], n: int, return_indices: bool
) -> Tensor | tuple[Tensor, Tensor]:
    """:func:`_adaptive_pool` for max, with the argmax when it is asked for."""
    out = _adaptive_pool(x, output_size, n, "max")
    if not return_indices:
        return out
    if out.numel() == 0:
        return out, _lucid.zeros(*out.shape, dtype=_lucid.int64, device=out.device)
    xb, unbatched = _add_batch_axis(x, n)
    sizes = tuple(int(v) for v in out.shape[-n:])
    return out, _drop_batch_axis(_adaptive_argmax_indices(xb, sizes), unbatched)


def _round32(v: float) -> float:
    """``v`` rounded to single precision, as a ``float`` kernel holds it."""
    return float(struct.unpack("f", struct.pack("f", v))[0])


def _frac_pool_starts(
    size: int, kernel: int, out_size: int, sample: float, single: bool
) -> list[int]:
    """Fractional window starts on one axis (Graham 2014 §3), as the reference.

    ``alpha = (size - kernel) / (out_size - 1)`` and
    ``start[i] = floor((i + u) * alpha) - floor(u * alpha)``; the last window
    ends flush with the input.  The arithmetic is in the input's precision
    (``single``), as the reference's kernel does it: a start that lands
    near an integer rounds the same way.
    """
    if out_size == 1:
        return [size - kernel]
    r = _round32 if single else float
    alpha = r(r(size - kernel) / r(out_size - 1))
    base = int(r(sample * alpha))
    starts = [int(r(r(i + sample) * alpha)) - base for i in range(out_size - 1)]
    return [*starts, size - kernel]


def _fractional_output_size(
    output_size: int | Sequence[int] | None,
    output_ratio: float | Sequence[float] | None,
    spatial: tuple[int, ...],
    kernel: tuple[int, ...],
    op: str,
) -> tuple[int, ...]:
    """The pooled extent per axis, from ``output_size`` or ``output_ratio``."""
    n = len(spatial)
    if (output_size is None) == (output_ratio is None):
        raise ValueError(f"{op}: give exactly one of output_size and output_ratio")
    if output_size is not None:
        out = _ntuple(output_size, n, op, "output_size")
    else:
        assert output_ratio is not None
        if isinstance(output_ratio := _promotion.as_number(output_ratio), (int, float)):
            ratios = (float(output_ratio),) * n
        else:
            ratios = tuple(float(r) for r in output_ratio)
        if len(ratios) != n or not all(0.0 < r < 1.0 for r in ratios):
            raise ValueError(
                f"{op}: output_ratio must be {n} numbers in (0, 1), got {output_ratio!r}"
            )
        out = tuple(max(1, int(s * r)) for s, r in zip(spatial, ratios, strict=True))
    for o, k, s in zip(out, kernel, spatial, strict=True):
        if o < 1 or o + k - 1 > s:
            raise ValueError(
                f"{op}: output_size {out} with kernel_size {kernel} does not fit "
                f"an input of {spatial}"
            )
    return out


def _fractional_plane(
    plane: Tensor, starts: list[list[int]], kernel: tuple[int, ...]
) -> tuple[Tensor, Tensor]:
    """One ``(*spatial)`` plane's window maxima and their flat input positions."""
    n = len(kernel)
    spatial = tuple(int(v) for v in plane.shape)
    out = tuple(len(s) for s in starts)
    windows = plane
    for d in range(n):
        taps = [s + k for s in starts[d] for k in range(kernel[d])]
        rows = _lucid.tensor(taps, dtype=_lucid.int64, device=plane.device)
        windows = windows.index_select(2 * d, rows)
        windows = windows.reshape(
            *windows.shape[: 2 * d], out[d], kernel[d], *spatial[d + 1 :]
        )
    order = [2 * d for d in range(n)] + [2 * d + 1 for d in range(n)]
    flat = windows.permute(order).reshape(*out, math.prod(kernel))
    values, local = flat.max(dim=-1)
    index = None
    for d in range(n):
        shape = [1] * n
        shape[d] = out[d]
        origin = _lucid.tensor(starts[d], dtype=_lucid.int64, device=plane.device)
        offset = _digit(local, math.prod(kernel[d + 1 :]), kernel[d])
        coord = offset + origin.reshape(*shape)
        index = coord if index is None else index * spatial[d] + coord
    assert index is not None
    return values, index


def _no_planes(
    xb: Tensor, out: tuple[int, ...], unbatched: bool, return_indices: bool
) -> Tensor | tuple[Tensor, Tensor]:
    """A fractional pool of a batched input with no ``(N, C)`` plane.

    The answer is empty at the pooled shape, as the reference's is for an
    empty batch, rather than a stack of nothing; empty channels get the same
    answer, as every other pool here gives them.  It comes from the input,
    so it requires grad when the input does.
    """
    lead = (int(xb.shape[0]), int(xb.shape[1]))
    result = _drop_batch_axis(xb.reshape(*lead, *out).clone(), unbatched)
    if not return_indices:
        return result
    where = _lucid.zeros(*lead, *out, dtype=_lucid.int64, device=xb.device)
    return result, _drop_batch_axis(where, unbatched)


def _fractional_max_pool(
    x: Tensor,
    kernel_size: int | Sequence[int],
    output_size: int | Sequence[int] | None,
    output_ratio: float | Sequence[float] | None,
    return_indices: bool,
    samples: Tensor | None,
    n: int,
) -> Tensor | tuple[Tensor, Tensor]:
    """The one body of ``fractional_max_pool{2,3}d`` and their modules."""
    op = f"fractional_max_pool{n}d"
    with _pool_autocast(op, x) as x:
        xb, unbatched = _add_batch_axis(x, n)
        if xb.ndim != n + 2:
            raise ValueError(
                f"{op}: expected a {n + 2}-D or unbatched {n + 1}-D input, "
                f"got shape {tuple(x.shape)}"
            )
        kernel = _ntuple(kernel_size, n, op, "kernel_size")
        _require_positive(kernel, op, "kernel_size")
        spatial = tuple(int(v) for v in xb.shape[2:])
        out = _fractional_output_size(output_size, output_ratio, spatial, kernel, op)
        batch, channels = int(xb.shape[0]), int(xb.shape[1])
        if samples is None:
            samples = _lucid.rand(batch, channels, n, dtype=xb.dtype, device=xb.device)
        if tuple(samples.shape) != (batch, channels, n):
            raise ValueError(
                f"{op}: _random_samples must have shape {(batch, channels, n)}, "
                f"got {tuple(samples.shape)}"
            )
        if batch * channels == 0:
            return _no_planes(xb, out, unbatched, return_indices)
        # The reference reads a 2-D sample as (width, height) and a 3-D one as
        # (depth, height, width).
        axis_sample = (1, 0) if n == 2 else tuple(range(n))
        single = xb.dtype != _lucid.float64
        values: list[Tensor] = []
        indices: list[Tensor] = []
        # One host read for every sample, not one per plane.
        flat = samples.reshape(-1).tolist()
        assert isinstance(flat, list)
        u: list[float] = []
        for v in flat:
            v_num = _promotion.as_number(v)
            if not isinstance(v_num, (int, float)):
                raise TypeError(f"{op}: _random_samples must be real, got {v!r}")
            u.append(float(v_num))
        for b in range(batch):
            for c in range(channels):
                at = (b * channels + c) * n
                starts = [
                    _frac_pool_starts(
                        spatial[d], kernel[d], out[d], u[at + axis_sample[d]], single
                    )
                    for d in range(n)
                ]
                value, index = _fractional_plane(xb[b, c], starts, kernel)
                values.append(value)
                indices.append(index)
        result = _drop_batch_axis(
            _lucid.stack(values).reshape(batch, channels, *out), unbatched
        )
        if not return_indices:
            return result
        where = _lucid.stack(indices).reshape(batch, channels, *out)
        return result, _drop_batch_axis(where, unbatched)


def _scatter_unpool(
    x: Tensor,
    checked: tuple[Tensor, int],
    output_spatial: tuple[int, ...],
    n_spatial: int,
) -> Tensor:
    """Shared body of ``max_unpool{1,2,3}d``."""
    with _pool_autocast(f"max_unpool{n_spatial}d", x) as x:
        idx, spatial_numel = checked
        leading = [int(s) for s in x.shape[:-n_spatial]]
        zeros = _lucid.zeros(*leading, spatial_numel, dtype=x.dtype, device=x.device)
        flat_count = math.prod(int(s) for s in x.shape[-n_spatial:])
        x_flat = x.reshape(*leading, flat_count)
        idx_flat = idx.reshape(*leading, flat_count)
        summed = zeros.scatter_add(-1, idx_flat, x_flat)
        count = zeros.scatter_add(-1, idx_flat, _lucid.ones_like(x_flat)).clamp(min=1.0)
        return (summed / count).reshape(*leading, *output_spatial)
