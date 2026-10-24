"""CPU division, square root and reciprocal return the correctly rounded value.

IEEE 754 requires ``a / b``, ``sqrt(x)`` and ``1 / x`` to return the
representable value nearest the exact one, and the reference does that.
Four Accelerate entries do not across supported macOS releases. ``vDSP_vdivD``
(float64 division) was 1–2 ulp off on 41% of gaussian pairs: ``1 / 0.001``
gave 999.9999999999999, and a subnormal divisor gave NaN or an infinity of the
wrong sign. On macOS 26 ``vDSP_vdiv`` (float32 division) was likewise 1–2 ulp
off on 40.3% of gaussian pairs and flipped signs on subnormal and signed-zero
divisors (LCD-541). ``vvsqrtf`` was 1–2 ulp off on 41% of values, and ``vvrecf``
1 ulp off on 34%.  The CPU now computes those four with plain loops that
compile to the exact NEON instructions.  It keeps the vForce entries that
measured exact (float64 ``vvsqrt``, ``vvrec``).

The oracle is NumPy's arithmetic, which runs on the same IEEE hardware
instructions.  Python's own float division checks the float64 quotient a
second time.  Comparison is bitwise, so it checks the signs of zero and
infinity, and NaN only has to be NaN.  Exact entries that stay on Accelerate
are pinned here too, so a change in Accelerate shows up as a failure.
"""

import math
from collections.abc import Callable

import numpy as np
import pytest

import lucid

_N = 1_000_000

# Sizes either side of one vector lane, a 16-lane block and the 16384-element
# per-core chunk, so heads, tails and joins are all exercised.
_SIZES = [1, 3, 17, 16383, 16385, _N]

_F64_EDGES = [
    0.0,
    -0.0,
    5e-324,
    -5e-324,
    2.225073858507201e-308,
    2.2250738585072014e-308,
    1e-300,
    0.001,
    0.1,
    1.0,
    -1.0,
    3.0,
    -7.0,
    1e300,
    1.7976931348623157e308,
    math.inf,
    -math.inf,
    math.nan,
]
_F32_EDGES = [
    0.0,
    -0.0,
    1e-45,
    -1e-45,
    1.1754942e-38,
    1.17549435e-38,
    1e-30,
    0.001,
    0.1,
    1.0,
    -1.0,
    3.0,
    -7.0,
    1e30,
    3.4028235e38,
    math.inf,
    -math.inf,
    math.nan,
]
_EDGES = {np.float64: _F64_EDGES, np.float32: _F32_EDGES}
Op = Callable[[lucid.Tensor], lucid.Tensor]
Ref = Callable[[np.ndarray], np.ndarray]

_LUCID = {
    np.float64: lucid.float64,
    np.float32: lucid.float32,
    np.float16: lucid.float16,
}


def _assert_bits(got: np.ndarray, ref: np.ndarray) -> None:
    assert got.dtype == ref.dtype and got.shape == ref.shape
    gnan, rnan = np.isnan(got), np.isnan(ref)
    np.testing.assert_array_equal(gnan, rnan)
    uint = {8: np.uint64, 4: np.uint32, 2: np.uint16}[ref.itemsize]
    gb, rb = got[~gnan].view(uint), ref[~rnan].view(uint)
    bad = np.flatnonzero(gb != rb)
    assert bad.size == 0, (
        f"{bad.size} of {gb.size} not correctly rounded, first at "
        f"{got[~gnan][bad[0]]!r} vs {ref[~rnan][bad[0]]!r}"
    )


def _gauss(n: int, dtype: type, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n).astype(dtype)


def _t(x: np.ndarray) -> lucid.Tensor:
    return lucid.tensor(x, dtype=_LUCID[x.dtype.type])


def _edge_pairs(dtype: type) -> tuple[np.ndarray, np.ndarray]:
    e = np.array(_EDGES[dtype], dtype=dtype)
    return np.repeat(e, e.size), np.tile(e, e.size)


def _np_div(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    with np.errstate(all="ignore"):
        return np.divide(a, b)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("n", _SIZES)
def test_tensor_by_tensor_division(dtype: type, n: int) -> None:
    a, b = _gauss(n, dtype, 1), _gauss(n, dtype, 2)
    ref = _np_div(a, b)
    _assert_bits((_t(a) / _t(b)).numpy(), ref)
    _assert_bits(lucid.div(_t(a), _t(b)).numpy(), ref)
    x = _t(a)
    x /= _t(b)
    _assert_bits(x.numpy(), ref)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_division_edge_cases(dtype: type) -> None:
    a, b = _edge_pairs(dtype)
    _assert_bits((_t(a) / _t(b)).numpy(), _np_div(a, b))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize(
    "s", [0.001, 0.3, 3.0, -7.0, 1e-300, 5e-324, -0.0, math.inf, math.nan]
)
def test_division_by_and_of_a_python_scalar(dtype: type, s: float) -> None:
    # A Python float is weak: it takes the tensor's dtype, so the oracle
    # rounds it to that dtype first.
    x = np.concatenate([_gauss(100_000, dtype, 3), np.array(_EDGES[dtype], dtype)])
    sd = dtype(s)
    _assert_bits((_t(x) / s).numpy(), _np_div(x, sd))
    _assert_bits((s / _t(x)).numpy(), _np_div(sd, x))


def test_float64_quotients_match_python() -> None:
    a, b = _gauss(100_000, np.float64, 4), _gauss(100_000, np.float64, 5)
    got = (_t(a) / _t(b)).numpy().tolist()
    ref = [x / y for x, y in zip(a.tolist(), b.tolist(), strict=True)]
    assert got == ref
    one, milli, tiny = (
        lucid.tensor(v, dtype=lucid.float64) for v in (1.0, 0.001, 5e-324)
    )
    assert (one / milli).item() == 1000.0
    assert (1.0 / milli).item() == 1000.0
    assert (one / 0.001).item() == 1000.0
    assert (
        math.copysign(1.0, (lucid.zeros((), dtype=lucid.float64) / tiny).item()) == 1.0
    )
    assert (lucid.zeros((), dtype=lucid.float64) / tiny).item() == 0.0
    assert (one / tiny).item() == math.inf
    assert (-one / tiny).item() == -math.inf


def test_float64_division_on_strided_and_broadcast_operands() -> None:
    a, b = _gauss(4096, np.float64, 6), _gauss(4096, np.float64, 7)
    ta, tb = _t(a), _t(b)
    _assert_bits((ta[::2] / tb[1::2]).numpy(), _np_div(a[::2], b[1::2]))
    a2, b2 = a.reshape(64, 64), b[:64].reshape(1, 64)
    _assert_bits((ta.reshape(64, 64).T / _t(b2)).numpy(), _np_div(a2.T, b2))


def test_float64_division_relatives_match_python() -> None:
    # These never went through vDSP_vdivD (backend/cpu/DivMod.h); pinned with
    # the quotient so the whole family is held to Python's answer.
    a, b = _gauss(100_000, np.float64, 8), _gauss(100_000, np.float64, 9)
    ta, tb = _t(a), _t(b)
    pairs = list(zip(a.tolist(), b.tolist(), strict=True))
    assert lucid.floor_divide(ta, tb).numpy().tolist() == [x // y for x, y in pairs]
    assert lucid.remainder(ta, tb).numpy().tolist() == [x % y for x, y in pairs]
    assert lucid.fmod(ta, tb).numpy().tolist() == [math.fmod(x, y) for x, y in pairs]


def _unary_cases(dtype: type, n: int, seed: int) -> np.ndarray:
    x = _gauss(n, dtype, seed)
    if n == _N:
        x[: len(_EDGES[dtype])] = np.array(_EDGES[dtype], dtype)
    return x


def _ref_sqrt(x: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        return np.sqrt(x)


def _ref_rsqrt(x: np.ndarray) -> np.ndarray:
    # The reference's rsqrt is the rounded root, then the rounded reciprocal.
    return _np_div(x.dtype.type(1), _ref_sqrt(x))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("n", _SIZES)
@pytest.mark.parametrize(
    "op, ref",
    [
        (lucid.sqrt, _ref_sqrt),
        (lucid.reciprocal, lambda x: _np_div(x.dtype.type(1), x)),
        (lucid.rsqrt, _ref_rsqrt),
    ],
    ids=["sqrt", "reciprocal", "rsqrt"],
)
def test_unary_ieee_ops(dtype: type, n: int, op: Op, ref: Ref) -> None:
    x = _unary_cases(dtype, n, 10)
    xs = np.abs(x) if op is not lucid.reciprocal else x
    _assert_bits(op(_t(xs)).numpy(), ref(xs))
    _assert_bits(op(_t(x)).numpy(), ref(x))


@pytest.mark.parametrize(
    "op, ref",
    [
        (lucid.sqrt, _ref_sqrt),
        (lucid.reciprocal, lambda x: _np_div(np.float32(1), x)),
        (lucid.rsqrt, _ref_rsqrt),
    ],
    ids=["sqrt", "reciprocal", "rsqrt"],
)
def test_every_float16_value(op: Op, ref: Ref) -> None:
    # Half computes in float32 and rounds once.  A correctly rounded float32
    # result rounds to the correctly rounded half (24 >= 2*11 + 2 bits), so
    # every one of the 65536 inputs must come out exact.
    x = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16).view(np.float16)
    with np.errstate(all="ignore"):
        want = ref(x.astype(np.float32)).astype(np.float16)
    _assert_bits(op(_t(x)).numpy(), want)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_mean_matches_sum_divided_by_n_and_backward(dtype: type) -> None:
    rng = np.random.default_rng(11)
    ldt = _LUCID[dtype]
    for _ in range(200):
        n = int(rng.integers(3, 12))
        v = float(rng.standard_normal())
        c = lucid.full((n,), v, dtype=ldt)
        _assert_bits(c.mean().numpy(), (c.sum() / n).numpy())
        r = _t(rng.standard_normal(n).astype(dtype))
        _assert_bits(r.mean().numpy(), (r.sum() / n).numpy())
        m = int(rng.integers(3, 12))
        r2 = _t(rng.standard_normal((m, n)).astype(dtype))
        _assert_bits(r2.mean().numpy(), (r2.sum() / (m * n)).numpy())
        _assert_bits(r2.mean(dim=0).numpy(), (r2.sum(dim=0) / m).numpy())
        _assert_bits(r2.mean(dim=1).numpy(), (r2.sum(dim=1) / n).numpy())
        xr = lucid.ones((n,), dtype=ldt, requires_grad=True)
        g = lucid.full((), v, dtype=ldt)
        xr.mean().backward(g)
        assert xr.grad is not None
        _assert_bits(xr.grad.numpy(), (lucid.full((n,), v, dtype=ldt) / n).numpy())
    for dev in ("cpu", "metal") if dtype is np.float32 else ("cpu",):
        x2 = lucid.ones((3, 5), dtype=ldt, device=dev, requires_grad=True)
        g2 = lucid.full((3, 1), 0.3, dtype=ldt, device=dev)
        x2.mean(dim=1, keepdim=True).backward(g2)
        assert x2.grad is not None
        want2 = (g2 / 5).expand((3, 5)).cpu().numpy()
        _assert_bits(x2.grad.cpu().numpy(), want2)
        x3 = lucid.ones((2, 3, 7), dtype=ldt, device=dev, requires_grad=True)
        g3 = lucid.full((2, 3), -0.7, dtype=ldt, device=dev)
        x3.mean(dim=-1).backward(g3)
        assert x3.grad is not None
        want3 = (g3 / 7).unsqueeze(-1).expand((2, 3, 7)).cpu().numpy()
        _assert_bits(x3.grad.cpu().numpy(), want3)
