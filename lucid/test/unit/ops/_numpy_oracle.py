"""numpy reference for the cases of :mod:`lucid.test.unit.compile._op_matrix`.

The replay matrix compares compiled with eager, so an op eager gets wrong the
same way on both devices passes it: ``arcsin`` of an integer stayed integer
(``arcsin(1)`` was 1) on CPU and Metal alike, and was found only because
MPSGraph happened to lack an int64 kernel.  This table answers each case
independently, in numpy, and :mod:`.test_numpy_oracle` holds eager — on CPU
and on Metal — to it.

H4: numpy is used here, in the test tree, and only to compute expected
values.  Values cross into it through ``Tensor.numpy()`` and nothing in
``lucid/`` outside ``lucid/test/`` imports this module.

Each entry is ``name → ref(x, c)`` where ``x`` is the case input as a numpy
array and ``c`` hands out the case's constants (weights, indices) as numpy
arrays.  A reference returns an array or a tuple of arrays.  Where numpy's
own dtype rules differ from the reference framework's — ``floor`` of an
integer is float64 in numpy, integer there — only values are compared; the
dtype *kind* is checked separately, and only for ops whose answer cannot be
an integer (:data:`REAL_VALUED`).
"""

import math
import statistics
from collections.abc import Callable

import numpy as np

_ND = statistics.NormalDist()


def _f(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _erf(x: np.ndarray) -> np.ndarray:
    return np.vectorize(math.erf, otypes=[np.float64])(_f(x))


def _erfinv_scalar(y: float) -> float:
    if y == 1.0:
        return math.inf
    if y == -1.0:
        return -math.inf
    if not -1.0 < y < 1.0:
        return math.nan
    return _ND.inv_cdf((y + 1.0) / 2.0) / math.sqrt(2.0)


def _erfinv(x: np.ndarray) -> np.ndarray:
    return np.vectorize(_erfinv_scalar, otypes=[np.float64])(_f(x))


def _flip0(x: np.ndarray) -> np.ndarray:
    return x[::-1]


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    e = np.exp(_f(x) - _f(x).max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _log_softmax(x: np.ndarray, axis: int) -> np.ndarray:
    z = _f(x) - _f(x).max(axis=axis, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=axis, keepdims=True))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-_f(x)))


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, _f(x))


def _cumext(x: np.ndarray, axis: int, is_max: bool) -> tuple[np.ndarray, np.ndarray]:
    """Running max / min and the index of the running extremum.

    On ties the index is the *latest* position, which is what the reference
    framework returns for ``cummax`` / ``cummin``.
    """
    x = np.moveaxis(x, axis, -1)
    vals = np.empty_like(x)
    idx = np.empty(x.shape, dtype=np.int64)
    best = x[..., 0].copy()
    at = np.zeros(best.shape, dtype=np.int64)
    for i in range(x.shape[-1]):
        cur = x[..., i]
        better = cur >= best if is_max else cur <= best
        best = np.where(better, cur, best)
        at = np.where(better, i, at)
        vals[..., i] = best
        idx[..., i] = at
    return np.moveaxis(vals, -1, axis), np.moveaxis(idx, -1, axis)


def _conv(
    x: np.ndarray, w: np.ndarray, b: np.ndarray | None, stride: int, padding: int
) -> np.ndarray:
    """N-d cross-correlation, NC… layout, groups = 1."""
    nd = w.ndim - 2
    if padding:
        x = np.pad(x, [(0, 0), (0, 0)] + [(padding, padding)] * nd)
    win = np.lib.stride_tricks.sliding_window_view(
        _f(x), w.shape[2:], axis=tuple(range(2, 2 + nd))
    )
    win = win[(slice(None), slice(None)) + (slice(None, None, stride),) * nd]
    # win: N, C, *out, *k  ·  w: O, C, *k  →  N, O, *out
    letters = "defgh"[:nd]  # not "c": that is the channel subscript
    kl = "pqrs"[:nd]
    out = np.einsum(f"nc{letters}{kl},oc{kl}->no{letters}", win, _f(w))
    if b is not None:
        out = out + _f(b).reshape((1, -1) + (1,) * nd)
    return out


def _pool2d(x: np.ndarray, k: int, op: Callable[..., np.ndarray]) -> np.ndarray:
    n, c, h, w = x.shape
    return op(_f(x).reshape(n, c, h // k, k, w // k, k), axis=(3, 5))


def _pool(
    x: np.ndarray,
    k: int,
    s: int,
    p: int,
    is_max: bool,
    count_pad: bool = True,
    ceil: bool = False,
) -> np.ndarray:
    """Pool the trailing axes of (N, C, *spatial), the reference's way.

    A ceil-mode window that would start in the right padding is dropped; an
    average that does not count padding divides by the in-bounds part.
    """
    x = _f(x)
    spatial = x.shape[2:]
    outs = []
    for size in spatial:
        span = size + 2 * p - k
        o = (-(-span // s) if ceil else span // s) + 1
        if ceil and (o - 1) * s >= size + p:
            o -= 1
        outs.append(o)
    y = np.empty(x.shape[:2] + tuple(outs))
    for pos in np.ndindex(*outs):
        lo = [q * s - p for q in pos]
        sl = tuple(slice(max(a, 0), min(a + k, n)) for a, n in zip(lo, spatial))
        win = x[(slice(None), slice(None)) + sl]
        if is_max:
            y[(slice(None), slice(None)) + pos] = win.max(axis=tuple(range(2, x.ndim)))
            continue
        total = win.sum(axis=tuple(range(2, x.ndim)))
        if count_pad:
            # The padded input's part of the window.
            count = np.prod([min(a + k, n + p) - a for a, n in zip(lo, spatial)])
        else:
            count = np.prod([t.stop - t.start for t in sl])
        y[(slice(None), slice(None)) + pos] = total / count
    return y


def _nearest(x: np.ndarray, size: tuple[int, ...]) -> np.ndarray:
    """Floor-index nearest resampling of the trailing axes."""
    y = _f(x)
    for ax, out in zip(range(2, 2 + len(size)), size):
        n = y.shape[ax]
        y = np.take(y, [min(i * n // out, n - 1) for i in range(out)], axis=ax)
    return y


def _grid_sample(
    x: np.ndarray, grid: np.ndarray, nearest: bool, border: bool, align: bool
) -> np.ndarray:
    """``F.grid_sample`` on (N, C, H, W): zeros drops the corners outside the
    image, border clamps the coordinate first, nearest rounds half to even."""
    x, grid = _f(x), _f(grid)
    n, _, h, w = x.shape

    def denorm(v: np.ndarray, extent: int) -> np.ndarray:
        if align:
            return (v + 1.0) * (extent - 1) / 2.0
        return (v + 1.0) * extent / 2.0 - 0.5

    ix, iy = denorm(grid[..., 0], w), denorm(grid[..., 1], h)

    def read(px: np.ndarray, py: np.ndarray) -> np.ndarray:
        inside = (px >= 0) & (px <= w - 1) & (py >= 0) & (py <= h - 1)
        cx = np.clip(px, 0, w - 1).astype(np.int64)
        cy = np.clip(py, 0, h - 1).astype(np.int64)
        v = np.moveaxis(x[np.arange(n)[:, None, None], :, cy, cx], -1, 1)
        return v if border else v * inside[:, None]

    if nearest:
        return read(np.rint(ix), np.rint(iy))
    if border:
        ix, iy = np.clip(ix, 0, w - 1), np.clip(iy, 0, h - 1)
    x0, y0 = np.floor(ix), np.floor(iy)
    x1, y1 = x0 + 1.0, y0 + 1.0
    return (
        read(x0, y0) * ((x1 - ix) * (y1 - iy))[:, None]
        + read(x0, y1) * ((x1 - ix) * (iy - y0))[:, None]
        + read(x1, y0) * ((ix - x0) * (y1 - iy))[:, None]
        + read(x1, y1) * ((ix - x0) * (iy - y0))[:, None]
    )


def _bilinear(x: np.ndarray, size: tuple[int, ...], align: bool) -> np.ndarray:
    """Linear resampling of the trailing axes, one axis at a time."""
    y = _f(x)
    for ax, out in zip(range(2, 2 + len(size)), size):
        n = y.shape[ax]
        if align:
            src = np.arange(out) * ((n - 1) / (out - 1) if out > 1 else 0.0)
        else:
            src = np.maximum((np.arange(out) + 0.5) * (n / out) - 0.5, 0.0)
        i0 = np.minimum(np.floor(src).astype(int), n - 1)
        i1 = np.minimum(i0 + 1, n - 1)
        lam = (src - i0).reshape((-1,) + (1,) * (y.ndim - ax - 1))
        y = np.take(y, i0, axis=ax) * (1 - lam) + np.take(y, i1, axis=ax) * lam
    return y


def _batch_norm_train(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-channel batch statistics, biased variance, eps 1e-5."""
    x = _f(x)
    axes = (0,) + tuple(range(2, x.ndim))
    shape = (1, -1) + (1,) * (x.ndim - 2)
    mean = x.mean(axis=axes, keepdims=True)
    var = x.var(axis=axes, keepdims=True)
    return (x - mean) / np.sqrt(var + 1e-5) * _f(w).reshape(shape) + _f(b).reshape(
        shape
    )


def _unfold(x: np.ndarray, axis: int, size: int, step: int) -> np.ndarray:
    win = np.lib.stride_tricks.sliding_window_view(x, size, axis=axis)
    return np.take(win, range(0, win.shape[axis], step), axis=axis)


def _scatter(
    base: np.ndarray, idx: np.ndarray, src: np.ndarray, add: bool
) -> np.ndarray:
    out = base.copy()
    rows = np.arange(idx.shape[0])[:, None]
    if add:
        np.add.at(out, (rows, idx), src)
    else:
        out[rows, idx] = src
    return out


def _sdpa(q: np.ndarray, k: np.ndarray, v: np.ndarray, causal: bool) -> np.ndarray:
    s = _f(q) @ np.swapaxes(_f(k), -1, -2) / math.sqrt(q.shape[-1])
    if causal:
        lq, lk = s.shape[-2:]
        s = np.where(np.tril(np.ones((lq, lk), dtype=bool)), s, -np.inf)
    return _softmax(s, -1) @ _f(v)


def _group_norm(
    x: np.ndarray, g: int, w: np.ndarray, b: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    n, c = x.shape[:2]
    xg = _f(x).reshape(n, g, -1)
    xg = (xg - xg.mean(-1, keepdims=True)) / np.sqrt(xg.var(-1, keepdims=True) + eps)
    shape = (1, c) + (1,) * (x.ndim - 2)
    return xg.reshape(x.shape) * _f(w).reshape(shape) + _f(b).reshape(shape)


Ref = Callable[..., object]

#: name → ref(x, c).  ``c.w(*shape)`` / ``c.idx(*vals)`` / ``c.eye(n)`` give
#: the case's constants as numpy arrays.
REFS: dict[str, Ref] = {
    # ── unary
    "abs": lambda x, c: np.abs(x),
    "neg": lambda x, c: -x.astype(np.int64) if x.dtype == np.bool_ else -x,
    # numpy has no ``sign`` of a bool; the reference framework's is the
    # identity (``sign(True)`` is ``True``, still bool).
    "sign": lambda x, c: x.copy() if x.dtype == np.bool_ else np.sign(x),
    "square": lambda x, c: x.astype(np.int64) ** 2 if x.dtype == np.bool_ else x * x,
    "round": lambda x, c: np.round(x),
    "floor": lambda x, c: np.floor(x),
    "ceil": lambda x, c: np.ceil(x),
    "trunc": lambda x, c: np.trunc(x),
    "relu": lambda x, c: np.maximum(x, 0),
    "isnan": lambda x, c: np.isnan(_f(x)),
    "isinf": lambda x, c: np.isinf(_f(x)),
    "isfinite": lambda x, c: np.isfinite(_f(x)),
    "logical_not": lambda x, c: np.logical_not(x),
    "nan_to_num": lambda x, c: np.nan_to_num(x),
    "clamp": lambda x, c: np.clip(x, -2, 2),
    "clip": lambda x, c: np.clip(x, -1, 3),
    "pow_scalar": lambda x, c: x.astype(np.int64) ** 2 if x.dtype == np.bool_ else x**2,
    "pow_scalar3": lambda x, c: (
        x.astype(np.int64) ** 3 if x.dtype == np.bool_ else x**3
    ),
    "rpow_scalar": lambda x, c: 2.0 ** _f(x),
    "astype_f32": lambda x, c: x.astype(np.float32),
    "astype_i64": lambda x, c: x.astype(np.int64),
    "astype_bool": lambda x, c: x.astype(np.bool_),
    "exp": lambda x, c: np.exp(_f(x)),
    "log": lambda x, c: np.log(_f(x)),
    "log2": lambda x, c: np.log2(_f(x)),
    "sqrt": lambda x, c: np.sqrt(_f(x)),
    "rsqrt": lambda x, c: 1.0 / np.sqrt(_f(x)),
    "reciprocal": lambda x, c: 1.0 / _f(x),
    "sin": lambda x, c: np.sin(_f(x)),
    "cos": lambda x, c: np.cos(_f(x)),
    "tan": lambda x, c: np.tan(_f(x)),
    "arcsin": lambda x, c: np.arcsin(_f(x)),
    "arccos": lambda x, c: np.arccos(_f(x)),
    "arctan": lambda x, c: np.arctan(_f(x)),
    "sinh": lambda x, c: np.sinh(_f(x)),
    "cosh": lambda x, c: np.cosh(_f(x)),
    "tanh": lambda x, c: np.tanh(_f(x)),
    "erf": lambda x, c: _erf(x),
    "erfinv": lambda x, c: _erfinv(x),
    "erfinv_edge": lambda x, c: _erfinv(np.round(np.clip(_f(x), -1, 1))),
    "sigmoid": lambda x, c: _sigmoid(x),
    "cbrt": lambda x, c: _f(x) ** (1.0 / 3.0),
    # ── binary (the recipes pair x with its own flip along dim 0)
    "add": lambda x, c: x + _flip0(x),
    "sub": lambda x, c: x - _flip0(x),
    "mul": lambda x, c: x * _flip0(x),
    "div": lambda x, c: _f(x) / (np.abs(_f(_flip0(x))) + 1),
    "floordiv": lambda x, c: np.floor_divide(x, np.abs(_flip0(x)) + 1),
    "remainder": lambda x, c: np.mod(x, 4),
    "fmod": lambda x, c: np.fmod(x, 4),
    "pow": lambda x, c: np.power(np.abs(x) + 1, np.minimum(np.abs(_flip0(x)), 2)),
    "maximum": lambda x, c: np.maximum(x, _flip0(x)),
    "minimum": lambda x, c: np.minimum(x, _flip0(x)),
    "eq": lambda x, c: x == _flip0(x),
    "ne": lambda x, c: x != _flip0(x),
    "lt": lambda x, c: x < _flip0(x),
    "le": lambda x, c: x <= _flip0(x),
    "gt": lambda x, c: x > _flip0(x),
    "ge": lambda x, c: x >= _flip0(x),
    "logical_and": lambda x, c: np.logical_and(x, _flip0(x)),
    "logical_or": lambda x, c: np.logical_or(x, _flip0(x)),
    "logical_xor": lambda x, c: np.logical_xor(x, _flip0(x)),
    "where": lambda x, c: np.where(x > 0, x, _flip0(x)),
    "masked_fill": lambda x, c: np.where(x > 1, np.zeros_like(x), x),
    "atan2": lambda x, c: np.arctan2(_f(x), _f(_flip0(x)) + 0.5),
    "hypot": lambda x, c: np.hypot(_f(x), _f(_flip0(x))),
    "xlogy": lambda x, c: np.where(
        _f(x) == 0, 0.0, _f(x) * np.log(np.abs(_f(_flip0(x))) + 1)
    ),
    "logaddexp": lambda x, c: np.logaddexp(_f(x), _f(_flip0(x))),
    "nextafter": lambda x, c: np.nextafter(x, _flip0(x)),
    "isclose": lambda x, c: np.isclose(_f(x), _f(_flip0(x))),
    "lerp": lambda x, c: x + 0.25 * (_flip0(x) - x),
    "bitwise_and": lambda x, c: x & _flip0(x),
    "invert": lambda x, c: ~x,
    "bitwise_or": lambda x, c: x | _flip0(x),
    "bitwise_xor": lambda x, c: x ^ _flip0(x),
    "shift_left": lambda x, c: np.left_shift(x, np.minimum(np.abs(_flip0(x)), 3)),
    "shift_right": lambda x, c: np.right_shift(x, np.minimum(np.abs(_flip0(x)), 3)),
    "shift_right_scalar": lambda x, c: np.right_shift(x, 1),
    # ── reductions
    "sum": lambda x, c: x.sum(),
    "sum_dim": lambda x, c: x.sum(axis=1),
    "sum_keep": lambda x, c: x.sum(axis=0, keepdims=True),
    "mean": lambda x, c: _f(x).mean(axis=1),
    "prod": lambda x, c: x.prod(axis=0),
    "max_all": lambda x, c: x.max(),
    "max_dim": lambda x, c: (x.max(axis=0), x.argmax(axis=0)),
    "min_dim": lambda x, c: (x.min(axis=1), x.argmin(axis=1)),
    "argmax": lambda x, c: x.argmax(axis=1),
    "argmin": lambda x, c: x.argmin(axis=0),
    "var": lambda x, c: _f(x).var(axis=1, ddof=1),
    "std": lambda x, c: _f(x).std(axis=0, ddof=1),
    "logsumexp": lambda x, c: np.log(np.exp(_f(x)).sum(axis=1)),
    "cumsum": lambda x, c: np.cumsum(x, axis=0),
    "cumprod": lambda x, c: np.cumprod(x, axis=1),
    "cummax": lambda x, c: _cumext(x, 0, True),
    "cummin": lambda x, c: _cumext(x, 1, False),
    "all": lambda x, c: np.all(x),
    "any": lambda x, c: np.any(x),
    "norm": lambda x, c: np.sqrt((_f(x) ** 2).sum()),
    "vector_norm": lambda x, c: np.sqrt((_f(x) ** 2).sum(axis=1)),
    "nansum": lambda x, c: np.nansum(x),
    "trace": lambda x, c: np.trace(x),
    # ── sort family (values and stable indices)
    "sort": lambda x, c: (
        np.sort(x, axis=-1),
        np.argsort(x, axis=-1, kind="stable"),
    ),
    "topk_values": lambda x, c: np.sort(x, axis=-1)[:, ::-1][:, :2],
    "topk": lambda x, c: (
        np.sort(x, axis=-1)[:, ::-1][:, :2],
        np.argsort(-x, axis=-1, kind="stable")[:, :2],
    ),
    # ── shape / layout / indexing
    "reshape": lambda x, c: x.reshape(6, 4) * 1,
    "view": lambda x, c: x.reshape(2, 12) * 1,
    "flatten": lambda x, c: x.reshape(-1) * 1,
    "unsqueeze": lambda x, c: x[:, None] * 1,
    "squeeze": lambda x, c: x * 1,
    "permute": lambda x, c: x.T * 1,
    "transpose_contig": lambda x, c: x.T.copy(),
    "broadcast_to": lambda x, c: np.broadcast_to(x[0], (3, 6)) * 1,
    "tile": lambda x, c: np.tile(x, (2, 1)),
    "repeat": lambda x, c: np.tile(x, (1, 2)),
    "repeat_interleave": lambda x, c: np.repeat(x, 2, axis=0),
    "cat": lambda x, c: np.concatenate([x, x[:, ::-1]], axis=1),
    "stack": lambda x, c: np.stack([x, _flip0(x)]),
    "split": lambda x, c: tuple(np.split(x, [2], axis=0)),
    "chunk": lambda x, c: tuple(np.split(x, 3, axis=1)),
    "unbind": lambda x, c: tuple(x[:, i] for i in range(x.shape[1])),
    "flip": lambda x, c: x[::-1, ::-1],
    "roll": lambda x, c: np.roll(x, (1, -2), axis=(0, 1)),
    "tril": lambda x, c: np.tril(x),
    "triu_k1": lambda x, c: np.triu(x, 1),
    "tril_km1": lambda x, c: np.tril(x, -1),
    "diagonal": lambda x, c: np.diagonal(x).copy(),
    "pad": lambda x, c: np.pad(x, [(0, 0), (1, 2)]),
    "slice": lambda x, c: x[1:3, ::2] * 1,
    "narrow": lambda x, c: x[:, 2:5] * 1,
    "gather": lambda x, c: np.take_along_axis(
        x, np.broadcast_to(c.idx(0, 5, 2, 1, 3, 4).reshape(1, 6), (4, 6)), 1
    ),
    "index_select": lambda x, c: np.take(x, c.idx(3, 0, 0, 2), axis=0),
    "fancy_index": lambda x, c: x[c.idx(2, 0, 3)],
    "scatter_add": lambda x, c: _scatter(
        np.zeros_like(x),
        np.broadcast_to(c.idx(0, 1, 0, 2, 5, 5).reshape(1, 6), (4, 6)),
        x,
        True,
    ),
    "scatter": lambda x, c: _scatter(
        np.zeros_like(x),
        np.broadcast_to(c.idx(5, 4, 3, 2, 1, 0).reshape(1, 6), (4, 6)),
        x,
        False,
    ),
    "unfold_dim": lambda x, c: _unfold(x, 1, 3, 2) * 1,
    "one_hot": lambda x, c: np.eye(5, dtype=np.int64)[
        np.minimum(np.abs(x).astype(np.int64), 4)
    ],
    "meshgrid": lambda x, c: tuple(np.meshgrid(x[0], x[1], indexing="ij")),
    "meshgrid_xy": lambda x, c: tuple(np.meshgrid(x[0][:3], x[1], indexing="xy")),
    # ── linear algebra
    "matmul": lambda x, c: x @ x.T,
    "matmul_batched": lambda x, c: _f(x) @ np.swapaxes(_f(x), -1, -2),
    "dot": lambda x, c: np.dot(x[0], x[1]),
    "inner": lambda x, c: np.inner(x, x),
    "outer": lambda x, c: np.outer(x[0], x[1]),
    "tensordot": lambda x, c: np.tensordot(_f(x), _f(c.w(6, 3)), axes=1),
    "matrix_power": lambda x, c: np.linalg.matrix_power(x, 3),
    "det": lambda x, c: np.linalg.det(_f(x) + 3 * np.eye(3)),
    "inv": lambda x, c: np.linalg.inv(_f(x) + 3 * np.eye(2)),
    "einsum": lambda x, c: np.einsum("ij,kj->ik", _f(x), _f(x)),
    # ── nn.functional
    "linear": lambda x, c: _f(x) @ _f(c.w(3, 6)).T + _f(c.w(3)),
    "bilinear": lambda x, c: np.einsum("bi,oij,bj->bo", _f(x), _f(c.w(2, 6, 6)), _f(x))
    + _f(c.w(2)),
    "softmax": lambda x, c: _softmax(x, -1),
    "log_softmax": lambda x, c: _log_softmax(x, -1),
    "relu6": lambda x, c: np.clip(_f(x), 0, 6),
    "elu": lambda x, c: np.where(_f(x) > 0, _f(x), np.expm1(_f(x))),
    "selu": lambda x, c: 1.0507009873554805
    * np.where(_f(x) > 0, _f(x), 1.6732632423543772 * np.expm1(_f(x))),
    "silu": lambda x, c: _f(x) * _sigmoid(x),
    "mish": lambda x, c: _f(x) * np.tanh(_softplus(x)),
    "softplus": lambda x, c: _softplus(x),
    "hardsigmoid": lambda x, c: np.clip(_f(x) / 6 + 0.5, 0, 1),
    "hardswish": lambda x, c: _f(x) * np.clip(_f(x) + 3, 0, 6) / 6,
    "gelu": lambda x, c: 0.5 * _f(x) * (1 + _erf(_f(x) / math.sqrt(2))),
    "gelu_tanh": lambda x, c: 0.5
    * _f(x)
    * (1 + np.tanh(math.sqrt(2 / math.pi) * (_f(x) + 0.044715 * _f(x) ** 3))),
    "leaky_relu": lambda x, c: np.where(_f(x) > 0, _f(x), 0.1 * _f(x)),
    "layer_norm": lambda x, c: (_f(x) - _f(x).mean(-1, keepdims=True))
    / np.sqrt(_f(x).var(-1, keepdims=True) + 1e-5)
    * _f(c.w(6))
    + _f(c.w(6)),
    "rms_norm": lambda x, c: _f(x)
    / np.sqrt((_f(x) ** 2).mean(-1, keepdims=True) + np.finfo(np.float32).eps)
    * _f(c.w(6)),
    "group_norm": lambda x, c: _group_norm(x, 2, c.w(4), c.w(4)),
    "batch_norm_eval": lambda x, c: (_f(x) - np.abs(_f(c.w(3))).reshape(1, 3, 1))
    / np.sqrt(np.abs(_f(c.w(3))).reshape(1, 3, 1) + 1 + 1e-5)
    * _f(c.w(3)).reshape(1, 3, 1)
    + _f(c.w(3)).reshape(1, 3, 1),
    "batch_norm_train": lambda x, c: _batch_norm_train(x, c.w(3), c.w(3, seed=8)),
    "batch_norm3d_train": lambda x, c: _batch_norm_train(x, c.w(2), c.w(2, seed=8)),
    "linspace_scale": lambda x, c: _f(x) * np.linspace(-1.0, 2.0, 6),
    "normalize": lambda x, c: _f(x)
    / np.maximum(np.sqrt((_f(x) ** 2).sum(1, keepdims=True)), 1e-12),
    "conv1d": lambda x, c: _conv(x, c.w(4, 3, 3), c.w(4), 1, 1),
    "conv2d": lambda x, c: _conv(x, c.w(4, 3, 3, 3), c.w(4), 2, 0),
    "conv3d": lambda x, c: _conv(x, c.w(2, 3, 2, 2, 2), None, 1, 0),
    "avg_pool2d": lambda x, c: _pool2d(x, 2, np.mean),
    "max_pool2d": lambda x, c: _pool2d(x, 2, np.max),
    "avg_pool2d_ceil_nopad": lambda x, c: _pool(x, 3, 2, 0, False, False, True),
    "avg_pool2d_t": lambda x, c: _pool2d(x, 2, np.mean).swapaxes(-1, -2),
    "max_pool2d_t": lambda x, c: _pool2d(x, 2, np.max).swapaxes(-1, -2),
    "pad_max_pool2d": lambda x, c: _pool(
        np.pad(_f(x), ((0, 0), (0, 0), (0, 1), (0, 1)), constant_values=-1e4),
        2,
        1,
        0,
        True,
    ),
    "pad_max_pool3d": lambda x, c: _pool(
        np.pad(_f(x), ((0, 0), (0, 0), (1, 0), (0, 1), (1, 0)), constant_values=-1e4),
        2,
        1,
        0,
        True,
    ),
    "max_pool1d": lambda x, c: _pool(x, 3, 2, 1, True),
    "avg_pool1d": lambda x, c: _pool(x, 3, 2, 1, False),
    "avg_pool1d_nopad": lambda x, c: _pool(x, 3, 2, 1, False, False),
    "max_pool3d": lambda x, c: _pool(x, 2, 2, 0, True),
    "avg_pool3d": lambda x, c: _pool(x, 3, 2, 1, False),
    "interp_nearest": lambda x, c: _nearest(x, (8, 8)),
    "interp_nearest_size": lambda x, c: _nearest(x, (3, 7)),
    "interp_bilinear": lambda x, c: _bilinear(x, (7, 5), False),
    "grid_sample_x": lambda x, c: _grid_sample(
        x, _f(c.w(2, 3, 4, 2, seed=11)) * 2.5, False, False, False
    ),
    "grid_sample_x_border_ac": lambda x, c: _grid_sample(
        x, _f(c.w(2, 3, 4, 2, seed=11)) * 2.5, False, True, True
    ),
    "grid_sample_x_nearest": lambda x, c: _grid_sample(
        x, _f(c.w(2, 3, 4, 2, seed=11)) * 2.5, True, False, False
    ),
    "grid_sample_grid": lambda x, c: _grid_sample(
        c.w(2, 3, 5, 4, seed=13), _f(x) * 0.2, False, False, False
    ),
    "grid_sample_grid_border_ac": lambda x, c: _grid_sample(
        c.w(2, 3, 5, 4, seed=13), _f(x) * 0.2, False, True, True
    ),
    "grid_sample_grid_nearest": lambda x, c: _grid_sample(
        c.w(2, 3, 5, 4, seed=13), _f(x) * 0.2, True, False, False
    ),
    "interp_bilinear_ac": lambda x, c: _bilinear(x, (7, 5), True),
    "interp_trilinear": lambda x, c: _bilinear(x, (3, 5, 4), False),
    "interp_trilinear_ac": lambda x, c: _bilinear(x, (3, 5, 4), True),
    "interp_nearest3d": lambda x, c: _nearest(x, (3, 5, 4)),
    "interp_bilinear_t": lambda x, c: _bilinear(x, (8, 6), False).swapaxes(-1, -2),
    "embedding": lambda x, c: _f(c.w(5, 3))[np.minimum(np.abs(x), 4)],
    "cross_entropy": lambda x, c: -_log_softmax(x, 1)[
        np.arange(4), c.idx(0, 5, 2, 1)
    ].mean(),
    "nll_loss": lambda x, c: -_log_softmax(x, 1)[
        np.arange(4), c.idx(0, 5, 2, 1)
    ].mean(),
    "mse_loss": lambda x, c: ((_f(x) - _f(_flip0(x))) ** 2).mean(),
    "huber_loss": lambda x, c: np.where(
        np.abs(_f(x) - _f(_flip0(x))) < 1.0,
        0.5 * (_f(x) - _f(_flip0(x))) ** 2,
        np.abs(_f(x) - _f(_flip0(x))) - 0.5,
    ).mean(),
    "bce": lambda x, c: -(
        (_flip0(x) > 0) * np.log(_sigmoid(x))
        + (_flip0(x) <= 0) * np.log(1 - _sigmoid(x))
    ).mean(),
    "bce_logits": lambda x, c: (
        np.maximum(_f(x), 0)
        - _f(x) * (_flip0(x) > 0)
        + np.log1p(np.exp(-np.abs(_f(x))))
    ).mean(),
    "sdpa": lambda x, c: _sdpa(x, x[:, :, ::-1], x * 0.5, False),
    "sdpa_causal": lambda x, c: _sdpa(x, x[:, :, ::-1], x * 0.5, True),
    "dropout_eval": lambda x, c: x * 1,
}

#: Ops whose answer cannot be an integer: an integer or bool input must come
#: back floating point.  ``arcsin`` / ``arccos`` / ``arctan`` failed this.
REAL_VALUED = frozenset(
    {
        "exp",
        "log",
        "log2",
        "sqrt",
        "rsqrt",
        "reciprocal",
        "sin",
        "cos",
        "tan",
        "arcsin",
        "arccos",
        "arctan",
        "sinh",
        "cosh",
        "tanh",
        "erf",
        "erfinv",
        "sigmoid",
        "cbrt",
        "rpow_scalar",
        "div",
        "atan2",
        "hypot",
        "xlogy",
        "logaddexp",
        "mean",
        "var",
        "std",
        "logsumexp",
        "norm",
        "vector_norm",
    }
)

#: (case, dtype) → (Lucid's answer, why) for inputs the reference framework
#: refuses and Lucid's eager answers anyway.  There is no reference answer to
#: hold these to, so the oracle does not generate them;
#: ``test_divergence_is_pinned`` holds eager to the answer written here, dtype
#: included, so the divergence stays on record.  A Lucid that starts refusing
#: fails that test — move the pair to ``EAGER_REJECTS`` then.
DIVERGES: dict[tuple[str, str], tuple[Ref, str]] = {
    # Empty: ``sub`` of two bools, the one entry, is now refused as the
    # reference framework refuses it (``EAGER_REJECTS``).
}

#: Cases with no numpy reference, and why — the ledger reads this list.
NO_REF: dict[str, str] = {
    "conv_transpose1d": "no compact numpy form; covered by the reference parity suite",
    "conv_transpose2d": "no compact numpy form; covered by the reference parity suite",
    "conv_transpose3d": "no compact numpy form; covered by the reference parity suite",
    "argsort": "index order among ties is unspecified — checked through take_along_dim",
    "kthvalue": "returns (values, indices); indices free on ties — values via sort",
    "take_along_dim": "self-consistent by construction: gathers x in argsort order",
    "dropout_train": "random by design",
    "rand_like": "random by design",
    "randn_like": "random by design",
}
