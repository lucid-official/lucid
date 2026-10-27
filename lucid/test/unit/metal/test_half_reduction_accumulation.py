"""Half-precision reductions on Metal accumulate in float32.

CHA-35.  MLX reduces in the dtype of its input: the accumulator of its
sum, prod and scan kernels is a ``half`` for a float16 input, and ``mean``
is that sum times 1/N.  So on Metal ``ones(64, 1024, float16).mean()`` was
inf (the sum, 65536, overflowed before the division), ``var`` and
``mse_loss`` followed it, a half ``BatchNorm2d`` trained its running
statistics to inf, and bfloat16 carried 8 bits through every partial sum
(``ones(100000).mean()`` answered 1.0078).  The reference accumulates both
half formats in float32 and rounds once.

Every expectation below is the exact answer for the exact half input,
computed in float64 and rounded to the input's dtype, so the tests hold
without the reference framework; the ``ref`` tests check the same cells
against it.  A float32 accumulation that rounds once can land one unit in
the last place from the exactly rounded answer, which is the tolerance.

The autocast side of the issue — ``sum`` / ``mean`` / ``prod`` cast down to
the autocast dtype before reducing — is pinned in
``unit/amp/test_autocast_reduction_dtype.py``.
"""

import math
from collections.abc import Callable

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid.test._fixtures.devices import metal_available

_HALVES = [lucid.float16, lucid.bfloat16]
#: Mantissa bits and the smallest subnormal of each half format.
_FORMAT: dict[lucid.dtype, tuple[int, float]] = {
    lucid.float16: (10, 2.0**-24),
    lucid.bfloat16: (7, 2.0**-133),
}


@pytest.fixture(autouse=True)
def _require_metal() -> None:
    if not metal_available():
        pytest.skip("Metal not available on this host")


def _name(dtype: lucid.dtype) -> str:
    return str(dtype).split(".")[-1]


def _half(values: np.ndarray, dtype: lucid.dtype) -> lucid.Tensor:
    """``values`` rounded to ``dtype``, on Metal."""
    return lucid.tensor(np.asarray(values, dtype=np.float64)).to(dtype).to("metal")


def _exact(t: lucid.Tensor) -> np.ndarray:
    """``t``'s values in float64 on the host — exact for a half dtype."""
    return t.to("cpu").to(lucid.float64).numpy()


def _rounded(value: float | np.ndarray, dtype: lucid.dtype) -> np.ndarray:
    """The float64 ``value`` rounded to ``dtype`` and read back."""
    return _exact(lucid.tensor(np.asarray(value, dtype=np.float64)).to(dtype))


def _ulp(value: np.ndarray, dtype: lucid.dtype) -> np.ndarray:
    """One unit in the last place of ``value`` in ``dtype``."""
    bits, tiny = _FORMAT[dtype]
    mag = np.abs(np.asarray(value, dtype=np.float64))
    with np.errstate(divide="ignore"):
        exp = np.floor(np.log2(np.where(mag > 0, mag, 1.0)))
    return np.maximum(2.0 ** (exp - bits), tiny)


def _assert_within_ulp(got: lucid.Tensor, truth: float | np.ndarray, ulps: int = 1) -> None:
    """``got`` is the float64 ``truth`` rounded to ``got``'s dtype, give or
    take ``ulps`` units in the last place."""
    want = _rounded(truth, got.dtype)
    have = _exact(got)
    assert np.all(np.isfinite(have)), have
    off = np.abs(have - want) / _ulp(want, got.dtype)
    assert np.all(off <= ulps), f"{have} is {off} ulp from {want}"


# ── the reproductions from the issue ─────────────────────────────────────────


def test_mean_of_65536_float16_ones_is_one() -> None:
    x = lucid.ones(64, 1024, dtype=lucid.float16, device="metal")
    got = x.mean()
    assert got.dtype == lucid.float16
    assert got.item() == 1.0


def test_mean_of_100000_bfloat16_ones_is_one() -> None:
    got = lucid.ones(100000, dtype=lucid.bfloat16, device="metal").mean()
    assert got.item() == 1.0


@pytest.mark.parametrize("dtype", _HALVES, ids=_name)
def test_variance_and_std_of_ones_are_zero(dtype: lucid.dtype) -> None:
    x = lucid.ones(64, 1024, dtype=dtype, device="metal")
    assert x.var().item() == 0.0
    assert x.std().item() == 0.0


@pytest.mark.parametrize("dtype", _HALVES, ids=_name)
def test_mse_and_l1_loss_over_65536_elements(dtype: lucid.dtype) -> None:
    x = lucid.ones(64, 1024, dtype=dtype, device="metal")
    y = lucid.zeros(64, 1024, dtype=dtype, device="metal")
    assert F.mse_loss(x, y).item() == 1.0
    assert F.l1_loss(x, y).item() == 1.0


# ── every accumulating reduction, on a sum float16 cannot hold ───────────────

_N = 70000  # more ones than float16 can count (65504)
_RNG = np.random.default_rng(35)
_NORMAL = _RNG.standard_normal(1 << 17)


def _cases() -> list[
    tuple[
        str,
        Callable[[], np.ndarray],
        Callable[..., lucid.Tensor],
        Callable[[np.ndarray], float],
    ]
]:
    """(id, input values, the op on a Metal tensor, the float64 truth)."""
    shifted = 3.0 + 2.0 * _NORMAL
    return [
        ("mean", lambda: np.full(_N, 3.0), lambda t: t.mean(), lambda v: v.mean()),
        ("mean-normal", lambda: shifted, lambda t: t.mean(), lambda v: v.mean()),
        ("sum-normal", lambda: _NORMAL * 4, lambda t: t.sum(), lambda v: v.sum()),
        ("var", lambda: shifted, lambda t: t.var(), lambda v: v.var(ddof=1)),
        ("std", lambda: shifted, lambda t: t.std(), lambda v: v.std(ddof=1)),
        (
            "var-axis",
            lambda: shifted,
            lambda t: t.reshape(4, -1).var(dim=1),
            lambda v: v.reshape(4, -1).var(axis=1, ddof=1),
        ),
        ("prod", lambda: np.full(1000, 1.01), lambda t: t.prod(), lambda v: v.prod()),
        (
            "cumsum",
            lambda: np.full(20000, 0.1),
            lambda t: lucid.cumsum(t, dim=0)[-1],
            lambda v: v.sum(),
        ),
        (
            "cumprod",
            lambda: np.full(1000, 1.01),
            lambda t: lucid.cumprod(t, dim=0)[-1],
            lambda v: v.prod(),
        ),
        (
            "logsumexp",
            lambda: np.zeros(_N),
            lambda t: lucid.logsumexp(t, dim=0),
            lambda v: math.log(np.exp(v).sum()),
        ),
        (
            "softmax",
            lambda: np.zeros(_N),
            lambda t: F.softmax(t, dim=0)[0],
            lambda v: 1.0 / v.size,
        ),
        (
            "log_softmax",
            lambda: np.zeros(_N),
            lambda t: F.log_softmax(t, dim=0)[0],
            lambda v: -math.log(v.size),
        ),
    ]


_CASES = _cases()


@pytest.mark.parametrize("dtype", _HALVES, ids=_name)
@pytest.mark.parametrize("case", _CASES, ids=[c[0] for c in _CASES])
def test_reduction_is_the_rounded_exact_answer(case: tuple, dtype: lucid.dtype) -> None:
    _, values, op, truth = case
    x = _half(values(), dtype)
    got = op(x)
    assert got.dtype == dtype
    # The truth is taken from the values the half tensor actually holds.
    # ``var`` and ``std`` round more than once on the way (the population
    # variance, the Bessel factor, the root), as the reference's do.
    _assert_within_ulp(got, truth(_exact(x)), ulps=2 if case[0].startswith(("var", "std")) else 1)


@pytest.mark.parametrize("dtype", _HALVES, ids=_name)
@pytest.mark.parametrize("case", _CASES, ids=[c[0] for c in _CASES])
def test_reduction_matches_reference(ref: object, case: tuple, dtype: lucid.dtype) -> None:
    """Against the reference on its own GPU stream: its CPU half kernels
    are not the comparison — they overflow ``logsumexp`` of 70000 zeros
    and lose 5% of a float16 ``prod`` of 1000 x 1.01."""
    if not ref.backends.mps.is_available():  # type: ignore[attr-defined]
        pytest.skip("the reference's GPU stream is not available")
    name, values, op, _ = case
    if name == "logsumexp" and dtype == lucid.float16:
        # The reference sums exp(x - max) in float16 and answers inf for
        # 70000 zeros; Lucid gives log(70000), the exact answer, which
        # ``test_reduction_is_the_rounded_exact_answer`` pins.
        pytest.skip("the reference overflows here")
    x = _half(values(), dtype)
    r = ref.tensor(_exact(x)).to(getattr(ref, _name(dtype))).to("mps")  # type: ignore[attr-defined]
    rf = ref.nn.functional  # type: ignore[attr-defined]
    ref_ops: dict[str, Callable[[object], object]] = {
        "mean": lambda t: t.mean(),
        "mean-normal": lambda t: t.mean(),
        "sum-normal": lambda t: t.sum(),
        "var": lambda t: t.var(),
        "std": lambda t: t.std(),
        "var-axis": lambda t: t.reshape(4, -1).var(dim=1),
        "prod": lambda t: t.prod(),
        "cumsum": lambda t: t.cumsum(0)[-1],
        "cumprod": lambda t: t.cumprod(0)[-1],
        "logsumexp": lambda t: t.logsumexp(0),
        "softmax": lambda t: rf.softmax(t, dim=0)[0],
        "log_softmax": lambda t: rf.log_softmax(t, dim=0)[0],
    }
    want = ref_ops[name](r).cpu().double().numpy()  # type: ignore[attr-defined]
    have = _exact(op(x))
    off = np.abs(have - want) / _ulp(want, dtype)
    assert np.all(off <= 1), f"{have} is {off} ulp from the reference's {want}"


@pytest.mark.parametrize("dtype", _HALVES, ids=_name)
@pytest.mark.parametrize(
    "shape,dims",
    [
        ((4096, 96), (0,)),  # a leading block, one dot product per column
        ((96, 4096), (1,)),  # a trailing block, one per row
        ((96, 4096), (-1,)),
        ((8, 512, 96), (0, 1)),
        ((96, 8, 512), (1, 2)),
        ((3, 70000), (1,)),  # few rows
        ((70000, 3), (0,)),  # few columns: transposed, then one per row
        ((16, 8, 512), (0, 2)),  # not a block: transposed, then one per row
        ((4096, 96), (0, 1)),  # everything: the widened reduction
    ],
)
@pytest.mark.parametrize("keepdim", [False, True])
def test_sum_over_axes_is_the_rounded_exact_answer(
    shape: tuple[int, ...], dims: tuple[int, ...], keepdim: bool, dtype: lucid.dtype
) -> None:
    """Every route a half ``sum`` takes on Metal — a matrix-vector product
    one way or the other, after a transposing copy when the reduced axes
    are not a block, and the widened reduction for a full sum —
    accumulates in float32 and rounds once."""
    values = 0.1 + _RNG.standard_normal(shape)
    x = _half(values, dtype)
    got = x.sum(dim=list(dims), keepdim=keepdim)
    want = _exact(x).sum(axis=dims, keepdims=keepdim)
    assert got.shape == want.shape
    _assert_within_ulp(got, want)


@pytest.mark.parametrize("dtype", _HALVES, ids=_name)
def test_sum_of_a_strided_view(dtype: lucid.dtype) -> None:
    x = _half(_RNG.standard_normal((512, 128)), dtype)
    xt = x.mT  # (128, 512), not contiguous
    _assert_within_ulp(xt.sum(dim=0), _exact(xt).sum(axis=0))
    _assert_within_ulp(xt.sum(dim=1), _exact(xt).sum(axis=1))


def test_half_sum_overflows_and_propagates_like_a_sum() -> None:
    """A sum float16 cannot hold is inf whichever route it takes, and a NaN
    or an infinity in the input reaches the answer."""
    big = lucid.full((4096, 128), 32.0, dtype=lucid.float16, device="metal")
    assert big.sum(dim=0).to("cpu").tolist() == [math.inf] * 128
    assert big.mT.sum(dim=1).to("cpu").tolist() == [math.inf] * 128
    x = lucid.zeros(4096, 128, dtype=lucid.float16, device="metal")
    x[5, 3] = math.nan
    x[7, 4] = math.inf
    x[9, 6] = math.inf
    x[11, 6] = -math.inf
    col = x.sum(dim=0).to("cpu").tolist()
    assert math.isnan(col[3]) and col[4] == math.inf and math.isnan(col[6])
    assert col[0] == 0.0


def test_log_softmax_does_not_underflow() -> None:
    """``log(softmax(x))`` is -inf once the probability underflows — at a
    logit 104 below the maximum in float32, 17 in float16."""
    for dtype in [lucid.float32, *_HALVES]:
        x = lucid.tensor([[0.0, -20.0, -200.0]], dtype=dtype, device="metal")
        got = F.log_softmax(x, dim=1)
        np.testing.assert_allclose(_exact(got), [[0.0, -20.0, -200.0]], rtol=1e-2)


@pytest.mark.parametrize("dtype", _HALVES, ids=_name)
def test_global_average_pool_of_a_large_map(dtype: lucid.dtype) -> None:
    x = lucid.full((2, 3, 56, 56), 30.0, dtype=dtype, device="metal")
    assert F.adaptive_avg_pool2d(x, 1).to("cpu").flatten().tolist() == [30.0] * 6
    assert F.avg_pool2d(x, 56).to("cpu").flatten().tolist() == [30.0] * 6


@pytest.mark.parametrize("dtype", _HALVES, ids=_name)
def test_broadcast_gradient_is_summed_in_float32(dtype: lucid.dtype) -> None:
    """The gradient of a broadcast operand is a sum over the batch."""
    a = lucid.zeros(_N, 3, dtype=dtype, device="metal")
    b = lucid.zeros(3, dtype=dtype, device="metal", requires_grad=True)
    ((a + b) * 0.5).sum().backward()
    _assert_within_ulp(b.grad, np.full(3, 0.5 * _N))


@pytest.mark.parametrize("dtype", _HALVES, ids=_name)
def test_linear_bias_gradient_is_summed_in_float32(dtype: lucid.dtype) -> None:
    lin = nn.Linear(4, 2).to("metal").to(dtype)
    x = lucid.zeros(_N, 4, dtype=dtype, device="metal")
    (lin(x) * 0.5).sum().backward()
    _assert_within_ulp(lin.bias.grad, np.full(2, 0.5 * _N))


# ── a variance whose first value is an outlier ───────────────────────────────

#: (shape, dims) — every route of a half variance on Metal.
_VARIANCE_ROUTES = [
    ((4096,), None),  # everything: per-threadgroup partials, merged after
    ((300, 16), (1,)),  # one row per simdgroup
    ((300, 4096), (1,)),
    ((4096, 96), (0,)),  # a leading block, a column per thread column
    ((65536, 64), (0,)),  # 2048 rows per thread column
    ((64, 8, 56, 56), (0, 2, 3)),  # per channel, as a BatchNorm reduces
    ((32, 64, 56, 56), (0, 2, 3)),  # two of the 32 batch rows per threadgroup
    ((8, 512, 4096), (0, 2)),  # one threadgroup per channel
    ((512, 32768), (1,)),  # one threadgroup per row
]


def _outlier_first(shape: tuple[int, ...], dims: tuple[int, ...] | None) -> np.ndarray:
    """Standard normal values with the first of every reduced set 1000."""
    values = _RNG.standard_normal(shape)
    first = tuple(0 if dims is None or d in dims else slice(None) for d in range(len(shape)))
    values[first] = 1000.0
    return values


@pytest.mark.parametrize("dtype", _HALVES, ids=_name)
@pytest.mark.parametrize("correction", [0, 1])
@pytest.mark.parametrize("shape,dims", _VARIANCE_ROUTES, ids=[str(r[0]) for r in _VARIANCE_ROUTES])
def test_variance_with_an_outlier_first_is_the_rounded_exact_answer(
    shape: tuple[int, ...],
    dims: tuple[int, ...] | None,
    correction: int,
    dtype: lucid.dtype,
) -> None:
    """The kernels took their deviations from the first element of the
    whole reduction, so M2 was the difference of two sums of the outlier's
    squared distance: 2 float16 ulp off over 4096 values, 44 per channel of
    a (64, 8, 56, 56) map, 13 with one threadgroup per channel, 42 over
    65536 rows of a column.  The Bessel factor rounds once more."""
    x = _half(_outlier_first(shape, dims), dtype)
    got = x.var(dim=dims, correction=correction)
    truth = _exact(x).var(axis=dims, ddof=correction)
    _assert_within_ulp(got, truth, ulps=1 + correction)


# ── normalisation layers ─────────────────────────────────────────────────────


def _half_vs_float32(
    make: Callable[[], nn.Module],
    shape: tuple[int, ...],
    dtype: lucid.dtype,
    scale: float,
) -> tuple[nn.Module, nn.Module]:
    """The half layer against the same layer in float32, on the same
    half-representable input, through forward and backward.

    The loss is a fixed random projection of the output: ``sum(y * y)`` of
    a normalised ``y`` is a constant, so its input gradient is only
    rounding noise and compares nothing.  Returns (half layer, float32
    layer) for the caller's own checks."""
    values = scale * (1.0 + _RNG.standard_normal(shape))
    project = _RNG.standard_normal(shape)
    outs, layers = {}, {}
    for dt in (dtype, lucid.float32):
        layer = make().to("metal").to(dt)
        layer.train()
        x = _half(values, dtype).to(dt)
        x.requires_grad_(True)
        y = layer(x)
        (y * _half(project, dtype).to(dt)).sum().backward()
        outs[dt] = [_exact(y), _exact(x.grad)] + [_exact(p.grad) for p in layer.parameters()]
        layers[dt] = layer
    tol = 2.0 ** -_FORMAT[dtype][0] * 8
    for h, f in zip(outs[dtype], outs[lucid.float32], strict=True):
        assert np.all(np.isfinite(h))
        np.testing.assert_allclose(h, f, rtol=tol, atol=tol * np.abs(f).max())
    return layers[dtype], layers[lucid.float32]


@pytest.mark.parametrize("dtype", _HALVES, ids=_name)
def test_half_batchnorm_trains_one_step(dtype: lucid.dtype) -> None:
    """The issue's case: a half ``BatchNorm2d`` trained its running mean
    to inf.  Forward, backward and the running statistics against the
    same step in float32."""
    half, full = _half_vs_float32(lambda: nn.BatchNorm2d(4), (64, 4, 32, 32), dtype, scale=3.0)
    for name in ("running_mean", "running_var"):
        h, f = _exact(getattr(half, name)), _exact(getattr(full, name))
        assert np.all(np.isfinite(h)), (name, h)
        np.testing.assert_allclose(h, f, rtol=2.0 ** -_FORMAT[dtype][0] * 2)


@pytest.mark.parametrize("dtype", _HALVES, ids=_name)
@pytest.mark.parametrize("shape", [(64, 4, 32, 32), (8, 512, 32, 32)], ids=str)
def test_half_batchnorm_statistics_with_an_outlier_first(
    shape: tuple[int, ...], dtype: lucid.dtype
) -> None:
    """The BatchNorm statistics share the variance kernels' partials; 512
    channels take one threadgroup each."""
    x = _half(_outlier_first(shape, (0, 2, 3)), dtype)
    bn = nn.BatchNorm2d(shape[1], momentum=1.0).to("metal").to(dtype)
    bn.train()
    bn(x)
    exact = _exact(x)
    _assert_within_ulp(bn.running_mean, exact.mean(axis=(0, 2, 3)))
    _assert_within_ulp(bn.running_var, exact.var(axis=(0, 2, 3), ddof=1), ulps=2)


@pytest.mark.parametrize("dtype", _HALVES, ids=_name)
def test_half_group_norm_matches_float32(dtype: lucid.dtype) -> None:
    _half_vs_float32(lambda: nn.GroupNorm(2, 4), (8, 4, 64, 64), dtype, scale=40.0)


@pytest.mark.parametrize("dtype", _HALVES, ids=_name)
def test_half_layer_norm_matches_float32(dtype: lucid.dtype) -> None:
    _half_vs_float32(lambda: nn.LayerNorm(8192), (4, 8192), dtype, scale=40.0)


@pytest.mark.parametrize("dtype", _HALVES, ids=_name)
def test_half_rms_norm_matches_float32(dtype: lucid.dtype) -> None:
    # |x| past 256 squares out of float16's range.
    _half_vs_float32(lambda: nn.RMSNorm(1024), (4, 1024), dtype, scale=300.0)


# ── losses ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", _HALVES, ids=_name)
@pytest.mark.parametrize(
    "loss",
    [
        lambda x, t: F.mse_loss(x, t),
        lambda x, t: F.l1_loss(x, t),
        lambda x, t: F.huber_loss(x, t),
        lambda x, t: F.binary_cross_entropy_with_logits(x, t),
        lambda x, t: F.binary_cross_entropy(F.sigmoid(x), t),
    ],
    ids=["mse", "l1", "huber", "bce_with_logits", "bce"],
)
def test_mean_loss_over_more_elements_than_float16_counts(
    loss: Callable[[lucid.Tensor, lucid.Tensor], lucid.Tensor], dtype: lucid.dtype
) -> None:
    """The mean divided by the element count as a half scalar — inf past
    65504 — so the forward was 0 or NaN and the gradient 0."""
    xv = _RNG.standard_normal(_N)
    tv = (_RNG.random(_N) > 0.5).astype(np.float64)
    got = {}
    for dt in (dtype, lucid.float32):
        x = _half(xv, dtype).to(dt)
        x.requires_grad_(True)
        t = _half(tv, dtype).to(dt)
        out = loss(x, t)
        out.backward()
        got[dt] = (out.item(), _exact(x.grad))
    (l_h, g_h), (l_f, g_f) = got[dtype], got[lucid.float32]
    tol = 2.0 ** -_FORMAT[dtype][0] * 4
    assert math.isfinite(l_h)
    assert abs(l_h - l_f) <= tol * abs(l_f)
    assert np.all(np.isfinite(g_h))
    np.testing.assert_allclose(g_h, g_f, rtol=tol, atol=tol * np.abs(g_f).max())


@pytest.mark.parametrize("dtype", _HALVES, ids=_name)
@pytest.mark.parametrize("which", ["cross_entropy_loss", "nll_loss"])
def test_engine_class_loss_mean_over_many_targets(which: str, dtype: lucid.dtype) -> None:
    """The engine's fused class losses counted the targets in the loss
    dtype — exact in float16 only to 2048, inf past 65504 — and divided
    the half sum by that count."""
    n, c = _N, 3
    logits = _RNG.standard_normal((n, c))
    target = lucid.tensor(_RNG.integers(0, c, n), dtype=lucid.int64, device="metal")
    got = {}
    for dt in (dtype, lucid.float32):
        x = _half(logits, dtype).to(dt)
        x.requires_grad_(True)
        inp = x if which == "cross_entropy_loss" else F.log_softmax(x, dim=1)
        out = _wrap(getattr(_C_engine.nn, which)(_unwrap(inp), _unwrap(target)))
        out.backward()
        got[dt] = (out.item(), _exact(x.grad))
    (l_h, g_h), (l_f, g_f) = got[dtype], got[lucid.float32]
    tol = 2.0 ** -_FORMAT[dtype][0] * 4
    assert abs(l_h - l_f) <= tol * abs(l_f)
    assert np.all(np.isfinite(g_h))
    np.testing.assert_allclose(g_h, g_f, rtol=tol, atol=tol * np.abs(g_f).max())
