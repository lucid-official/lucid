"""CPU trigonometric and hyperbolic ops: Accelerate vForce plus exact special points.

H3 puts the CPU stream on Accelerate only, so ``sin cos tan asin acos atan
sinh cosh tanh`` run on vForce (``vvsinf`` / ``vvsin`` and so on). libm is
within 1 ulp but 7–14× slower per core. vForce is within a few ulp, except
at some special points where the reference returns the exact value. The
engine post-fixes those points with one compare-select over each tile
(``split_edges`` in ``backend/cpu/Vforce.cpp``, LCD-472):

* ``cos`` rounds to 1 for ``|x| <= 2**-12`` (float32) and ``|x| < 2**-26.5``
  (float64). vForce returned 1 - 1 ulp there, so ``cos(0)`` was
  ``0.99999994`` and ``hann_window`` did not start at 0.
* ``tan`` and ``tanh`` (and ``sin`` in float32) round to ``x`` for
  ``|x| <= 3.52e-4`` (float32) and ``1.35e-8`` (float64). vForce gave
  ``tan(-0.0) = +0.0`` and was 1 ulp off further in.

``vvsinf``, ``vvcosf`` and ``vvtanf`` also depend on their neighbours.
They work in blocks of 16 elements from the start of a call. A block that
holds a non-finite or large argument (from about 1e10 for sin / cos, 1e4
for tan) takes a slower path. There ``sin`` and ``tan`` flush subnormals to
0, turn ``-0`` into ``+0`` and are 1–3 ulp off tiny values, and every other
value of the block moves by about an ulp. The identity fix covers the tiny
values. The engine now cuts a call across cores only at 16-element
boundaries, so the bits no longer depend on the core count. No other
entry, and no float64 entry, does this.

The other five (``asin acos atan sinh cosh``) were exact at their special
points: the sign of ``-0``, subnormals, ``cosh(0) = 1``, ``±inf``, NaN, and
the domain edges of ``asin`` / ``acos``. ``tanh`` saturates to ±1 exactly
where the rounded value does. The tests below pin all of these, so a
change in Accelerate fails here.

Measured max ulp of vForce (macOS 26, M1 Pro, 2026-10-24). float32 is every
float in [-4π, 4π] (2.19e9 values) plus a stride of 61 over the rest of the
range. float64 is 4.2e7 values: dense in [-4π, 4π], random bit patterns,
and log-uniform magnitudes down to the subnormals.

======  =========  =========  =========  =========
op      f32 vs CR  f32 vs ref f64 vs np  f64 vs ref
======  =========  =========  =========  =========
sin     3          3          3          3
cos     3          3          3          3
tan     4          4          5          3
asin    2          2          2          2
acos    1          1          1          1
atan    3          3          1          1
sinh    2          2          2          2
cosh    2          2          2          2
tanh    3          3          3          4
======  =========  =========  =========  =========

"CR" is the correctly rounded float32, from NumPy's float64 then rounded.
"np" is NumPy's float64 (libm). "ref" is the reference framework, which is
itself within 1 ulp of CR except in three bands where it is the one that is
off: ``sinh`` / ``cosh`` overflow early (from 88.72 in float32 and 709.8 in
float64, where the rounded value is still finite), and ``tanh`` saturates
to ±1 from 8.66 (float32) and 18.7 (float64), before the rounded value does
at 9.01 and 19.06. The parity test leaves those bands out.

A deliberate departure: the ±1 plateaus of ``sin`` and ``cos`` at float
multiples of π/2 other than 0 are not fixed. vForce falls short of ±1 on
about 70% of the float32 plateau values (2847 of the 4096 around π/2: 2689
by 1 ulp, 158 by 2) and about 96% of the float64 ones (by 1–3 ulp). Metal
and the reference return ±1. Finding the plateau needs a range reduction,
which cost +24% on ``vvcosf`` (gated on the output) to +73% (every
element). The plateau tests pin both sides, so a change in vForce or MLX
shows up.
"""

import math
from collections.abc import Callable
from types import ModuleType

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F
from lucid.test._fixtures.devices import metal_available

_FUNCS = ["sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh"]
_NP: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
}
_LUCID = {np.float32: lucid.float32, np.float64: lucid.float64}

# The table in the module docstring.
_ULP_VS_ORACLE = {
    np.float32: {
        "sin": 3, "cos": 3, "tan": 4, "asin": 2, "acos": 1,
        "atan": 3, "sinh": 2, "cosh": 2, "tanh": 3,
    },
    np.float64: {
        "sin": 3, "cos": 3, "tan": 5, "asin": 2, "acos": 1,
        "atan": 1, "sinh": 2, "cosh": 2, "tanh": 3,
    },
}  # fmt: skip
_ULP_VS_REF = {
    np.float32: {
        "sin": 3, "cos": 3, "tan": 4, "asin": 2, "acos": 1,
        "atan": 3, "sinh": 2, "cosh": 2, "tanh": 3,
    },
    np.float64: {
        "sin": 3, "cos": 3, "tan": 3, "asin": 2, "acos": 1,
        "atan": 1, "sinh": 2, "cosh": 2, "tanh": 4,
    },
}  # fmt: skip

# float32 sin / cos / tan when every block also holds an inf (slow path),
# measured the same way over every float in [-4π, 4π].
_ULP_SLOW_BLOCK = {"sin": 3, "cos": 3, "tan": 4}

# Where the post-fix applies: the last |x| whose correctly rounded cos is 1,
# and whose correctly rounded tan / tanh is x itself.
_COS_ONE = {
    np.float32: float.fromhex("0x1p-12"),
    np.float64: float.fromhex("0x1.6a09e667f3bccp-27"),
}
_ODD_X = {
    np.float32: float.fromhex("0x1.713744p-12"),
    np.float64: float.fromhex("0x1.d12ed0af1a27ep-27"),
}


def _t(x: np.ndarray, device: str = "cpu") -> lucid.Tensor:
    return lucid.tensor(x, dtype=_LUCID[x.dtype.type], device=device)


def _oracle(name: str, x: np.ndarray) -> np.ndarray:
    """Correctly rounded for float32 (float64 then rounded); libm for float64."""
    with np.errstate(all="ignore"):
        return _NP[name](x.astype(np.float64)).astype(x.dtype)


def _ulp(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance in representable values; NaN against NaN is 0, against a
    number it is effectively infinite.  ±0 are the same value here — the
    special-point test checks their signs."""
    sint = np.int32 if a.dtype == np.float32 else np.int64
    nan = np.isnan(a) | np.isnan(b)

    def ordered(v: np.ndarray) -> np.ndarray:
        bits = np.where(nan, 0, v).astype(a.dtype).view(sint).astype(np.int64)
        lowest = np.int64(np.iinfo(sint).min)
        return np.where(bits < 0, lowest - bits, bits)

    d = np.abs(ordered(a) - ordered(b))
    return np.where(np.isnan(a) != np.isnan(b), np.int64(1) << 62, d)


def _assert_bits(got: np.ndarray, want: np.ndarray, where: np.ndarray) -> None:
    gnan, wnan = np.isnan(got), np.isnan(want)
    uint = np.uint32 if got.dtype == np.float32 else np.uint64
    bad = (gnan != wnan) | (~gnan & (got.view(uint) != want.view(uint)))
    idx = np.flatnonzero(bad)
    assert idx.size == 0, [
        (repr(where[i]), repr(got[i]), repr(want[i])) for i in idx[:8]
    ]


def _special_points(name: str, dtype: type) -> np.ndarray:
    fi = np.finfo(dtype)
    sub = float(np.nextafter(dtype(0), dtype(1)))
    pts = [0.0, sub, float(fi.smallest_normal) - sub, float(fi.smallest_normal)]
    pts += [1e-30, 1e-10, math.inf, math.nan]
    # well inside the plateaus: 1e-5 is one in float32 only
    pts += [1e-300, 1e-9] if dtype is np.float64 else [1e-5]
    if name == "cos":
        pts += [_COS_ONE[dtype]]
    elif name in ("sin", "tan", "tanh"):
        pts += [_ODD_X[dtype]]
    elif name in ("asin", "acos"):
        pts += [1.0, float(np.nextafter(dtype(1), dtype(2))), 2.0]
    elif name == "atan":
        pts += [1e30, float(fi.max)]
    elif name in ("sinh", "cosh"):
        pts += [1000.0, float(fi.max)]
    if name == "tanh":
        # first value whose correctly rounded tanh is 1, and two past it
        pts += [9.010914 if dtype is np.float32 else 19.07, 20.0, 50.0]
    x = np.array(pts, dtype=dtype)
    return np.concatenate([x, -x])


def _exact(name: str, x: np.ndarray) -> np.ndarray:
    """The correctly rounded value: the oracle, except on the plateaus the
    engine fixes, where it is known exactly — libm itself rounds up at the
    float64 edge of the tan plateau."""
    want = _oracle(name, x)
    a = np.abs(x)
    dt = x.dtype.type
    if name == "cos":
        want = np.where(a <= _COS_ONE[dt], dt(1.0), want)
    elif name in ("tan", "tanh") or (name == "sin" and dt is np.float32):
        want = np.where(a <= _ODD_X[dt], x, want)
    return want


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("name", _FUNCS)
def test_special_points_are_exact(name: str, dtype: type) -> None:
    # The points hold ±inf and NaN, so for sin / cos / tan in float32 their
    # 16-element blocks take vForce's slow path; the finite points alone,
    # one per call, take the fast one.  Both must be exact.
    x = _special_points(name, dtype)
    want = _exact(name, x)
    _assert_bits(getattr(lucid, name)(_t(x)).numpy(), want, x)
    finite = np.isfinite(x)
    for xi, wi in zip(x[finite], want[finite], strict=True):
        one = np.array([xi], dtype=dtype)
        _assert_bits(getattr(lucid, name)(_t(one)).numpy(), np.array([wi]), one)
    # The method and the in-place form reach the same kernel.
    _assert_bits(getattr(_t(x), name)().numpy(), want, x)
    inplace = _t(x)
    getattr(inplace, f"{name}_")()
    _assert_bits(inplace.numpy(), want, x)


def test_the_values_lcd_472_reported() -> None:
    assert lucid.cos(lucid.zeros(3)).tolist() == [1.0, 1.0, 1.0]
    assert lucid.cos(lucid.tensor([0.0, 1e-4])).tolist() == [1.0, 1.0]
    assert lucid.cos(lucid.zeros(3, dtype=lucid.float64)).tolist() == [1.0] * 3
    assert math.copysign(1.0, lucid.tan(lucid.tensor([-0.0])).item()) == -1.0
    assert lucid.hann_window(4)[0].item() == 0.0
    for dt in (lucid.float16, lucid.bfloat16):
        assert lucid.cos(lucid.zeros(2, dtype=dt)).float().tolist() == [1.0, 1.0]


def _grid(name: str, dtype: type) -> np.ndarray:
    rng = np.random.default_rng(472)
    if name in ("asin", "acos"):
        main = rng.uniform(-1.0, 1.0, 100_000)
    elif name in ("sinh", "cosh"):
        lim = 88.0 if dtype is np.float32 else 709.0
        main = rng.uniform(-lim, lim, 100_000)
    elif name == "tanh":
        main = rng.uniform(-20.0, 20.0, 100_000)
    else:
        main = rng.uniform(-4 * math.pi, 4 * math.pi, 100_000)
    lowest = -44.0 if dtype is np.float32 else -322.0
    sign = np.where(rng.integers(0, 2, 20_000) == 0, 1.0, -1.0)
    small = sign * 10.0 ** rng.uniform(lowest, 0.0, 20_000)
    parts = [main, small]
    if name in ("sin", "cos", "tan", "atan"):
        top = 30.0 if dtype is np.float32 else 300.0
        parts.append(sign[:10_000] * 10.0 ** rng.uniform(1.0, top, 10_000))
    return np.concatenate(parts).astype(dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("name", _FUNCS)
def test_within_the_measured_ulp_bound(name: str, dtype: type) -> None:
    x = _grid(name, dtype)
    got = getattr(lucid, name)(_t(x)).numpy()
    d = _ulp(got, _oracle(name, x))
    worst = int(d.max())
    assert worst <= _ULP_VS_ORACLE[dtype][name], (worst, repr(x[int(d.argmax())]))


@pytest.mark.parametrize("name", ["sin", "cos", "tan"])
def test_within_the_measured_ulp_bound_in_slow_blocks(name: str) -> None:
    # Every 16-element block also holds an inf, so all of float32 vForce's
    # blocks take the slow path.
    x = _grid(name, np.float32)
    x = x[: x.size - x.size % 16].reshape(-1, 16).copy()
    x[:, 15] = np.inf
    got = getattr(lucid, name)(_t(x.reshape(-1))).numpy().reshape(-1, 16)[:, :15]
    d = _ulp(got, _oracle(name, x[:, :15]))
    worst = int(d.max())
    assert worst <= _ULP_SLOW_BLOCK[name], (
        worst,
        repr(x[:, :15].flat[int(d.argmax())]),
    )


@pytest.mark.parametrize("name", _FUNCS)
def test_a_value_gets_the_bits_of_one_whole_call(name: str) -> None:
    # vForce's 16-element blocks restart at every call, so cutting a call at
    # anything but a block boundary changes the bits of a slow block.  The
    # engine cuts across cores at block boundaries only; one call per
    # 16-aligned piece must therefore reproduce the whole tensor exactly,
    # whatever the core count.  100_003 is several chunks on any machine.
    rng = np.random.default_rng(16)
    n = 100_003
    x = rng.uniform(-3.0, 3.0, n).astype(np.float32)
    x[rng.integers(0, n, 400)] = np.inf
    x[rng.integers(0, n, 400)] = np.float32(1e20)
    x[rng.integers(0, n, 400)] = np.float32(1e-30)
    x[rng.integers(0, n, 400)] = np.float32(-0.0)
    op = getattr(lucid, name)
    whole = op(_t(x)).numpy()
    piece = 4096
    parts = [op(_t(x[i : i + piece])).numpy() for i in range(0, n, piece)]
    _assert_bits(whole, np.concatenate(parts), x)


@pytest.mark.parametrize("n", [1, 3, 1023, 1024, 1025, 16385, 40_000])
@pytest.mark.parametrize("name", ["sin", "cos", "tan", "tanh"])
def test_a_fixed_tile_leaves_its_other_values_alone(name: str, n: int) -> None:
    # The fix runs per 1024-element tile, only where a tile holds a value
    # under the limit; the other values of that tile must come out exactly
    # as they do in a tile that takes the plain vForce path.
    rng = np.random.default_rng(n)
    for dtype in (np.float32, np.float64):
        x = rng.uniform(0.5, 1.5, n).astype(dtype)
        plain = getattr(lucid, name)(_t(x)).numpy()
        at = sorted({0, n - 1} | {i for i in (1023, 1024, 2047, 16383, 16384) if i < n})
        y = x.copy()
        y[at] = np.array(
            [-0.0, 0.0, 1e-20, -1e-9, 0.0, -0.0, 1e-30][: len(at)], dtype=dtype
        )
        got = getattr(lucid, name)(_t(y)).numpy()
        keep = np.ones(n, dtype=bool)
        keep[at] = False
        _assert_bits(got[keep], plain[keep], x[keep])
        _assert_bits(got[at], _exact(name, y[at]), y[at])


def test_strided_and_tanh_gelu_inputs() -> None:
    x = np.array([0.5, -0.0, 0.25, 0.0, 1e-30, -2.0] * 300, dtype=np.float32)
    t = _t(x)
    dense = np.ascontiguousarray(x[1::2])
    for name in ("sin", "cos", "tan"):
        got = getattr(lucid, name)(t[1::2]).numpy()
        _assert_bits(got, getattr(lucid, name)(_t(dense)).numpy(), dense)
        tiny = np.abs(dense) < 1e-20
        _assert_bits(got[tiny], _exact(name, dense[tiny]), dense[tiny])
    # GELU's tanh uses raw vtanh_unfixed_{f32,f64} (LCD-595) because 1 + tanh(u)
    # and 1 - tanh(u)^2 absorb a 1-ulp difference on a tiny u: its forward and
    # backward bits match the fixed lucid.tanh formula bit-for-bit and stay
    # within 2 ulp of the float64 oracle.
    c1 = 0.7978845608028654
    c2 = 0.044715
    rng = np.random.default_rng(595)
    for dtype in (np.float32, np.float64):
        edge = _ODD_X[dtype]
        xs = np.concatenate(
            [
                np.array([0.0, -0.0, 1e-30, -1e-30, 1e-6, -1e-4, edge, -edge, 0.5, -2.0], dtype=dtype),
                rng.uniform(-edge, edge, 2048).astype(dtype),
                rng.uniform(-4.0, 4.0, 2048).astype(dtype),
            ]
        )
        tx = _t(xs)
        tx.requires_grad_(True)
        out = F.gelu(tx, approximate="tanh")
        out.sum().backward()
        assert tx.grad is not None

        # On the entire |x| <= edge band where split_edges post-fixes tanh,
        # 1 + tanh(u) and 1 - tanh(u)^2 absorb the 1-ulp difference at tiny u
        # so the output and gradient bits are identical to the fixed tanh path.
        ux = _t(xs)
        c1_t = float(np.dtype(dtype).type(c1))
        c2_t = float(np.dtype(dtype).type(c2))
        inner_t = c1_t * (ux + c2_t * ux * ux * ux)
        t_fixed = lucid.tanh(inner_t)
        want_fwd = (0.5 * ux * (1.0 + t_fixed)).numpy()
        dinner = c1_t * (1.0 + 3.0 * c2_t * ux * ux)
        want_bwd = (0.5 * (1.0 + t_fixed) + 0.5 * ux * (1.0 - t_fixed * t_fixed) * dinner).numpy()
        in_edge = np.abs(xs) <= edge
        _assert_bits(out.detach().numpy()[in_edge], want_fwd[in_edge], xs[in_edge])
        _assert_bits(tx.grad.numpy()[in_edge], want_bwd[in_edge], xs[in_edge])

        x64 = xs[:9].astype(np.float64)
        inner = math.sqrt(2.0 / math.pi) * (x64 + 0.044715 * x64**3)
        oracle = (0.5 * x64 * (1.0 + np.tanh(inner))).astype(dtype)
        assert np.array_equal(np.signbit(out.detach().numpy()[:9]), np.signbit(oracle))
        assert int(_ulp(out.detach().numpy()[:9], oracle).max()) <= 2


# ── The ±1 plateaus at multiples of π/2 other than 0 ──────────────────────────


def _plateau(name: str, k: int, dtype: type) -> tuple[np.ndarray, np.ndarray]:
    """Values near k·π/2 whose correctly rounded sin / cos is ±1."""
    c = k * math.pi / 2
    if dtype is np.float32:
        lo, hi = np.float32(c - 4e-4), np.float32(c + 4e-4)
        x = np.arange(lo.view(np.uint32), hi.view(np.uint32) + 1, dtype=np.uint32)
        x = x.view(np.float32)
    else:
        x = np.linspace(c - 2e-8, c + 2e-8, 40_001, dtype=np.float64)
    want = _oracle(name, x)
    on = np.abs(want) == 1
    return x[on], want[on]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    ("name", "k"), [("sin", 1), ("sin", 3), ("cos", 2), ("cos", 4)]
)
def test_plateaus_away_from_zero_fall_short_on_cpu(
    name: str, k: int, dtype: type
) -> None:
    # A recorded bound, not a bug to be fixed here (see the module
    # docstring).  Measured: float32 misses ~70% of a plateau, by 1 or
    # 2 ulp; float64 ~96%, by 1 to 3.  If vForce starts returning ±1 here,
    # or falls further short, this fails: update the sin / cos docstrings.
    x, want = _plateau(name, k, dtype)
    got = getattr(lucid, name)(_t(x)).numpy()
    d = _ulp(got, want)
    share = float((d > 0).mean())
    lo, hi, worst = (0.6, 0.8, 2) if dtype is np.float32 else (0.9, 1.0, 3)
    assert lo <= share <= hi, share
    assert int(d.max()) == worst
    assert np.array_equal(np.sign(got), want)


@pytest.mark.skipif(not metal_available(), reason="needs Metal")
@pytest.mark.parametrize(
    ("name", "k"), [("sin", 1), ("sin", 3), ("cos", 2), ("cos", 4)]
)
def test_plateaus_away_from_zero_are_exact_on_metal(name: str, k: int) -> None:
    x, want = _plateau(name, k, np.float32)
    got = getattr(lucid, name)(_t(x, "metal")).cpu().numpy()
    _assert_bits(got, want, x)


# ── Against the reference ─────────────────────────────────────────────────────


def _ref_is_exact(name: str, x: np.ndarray) -> np.ndarray:
    """Leave out the bands where the reference, not vForce, is off."""
    a = np.abs(x.astype(np.float64))
    f32 = x.dtype == np.float32
    if name in ("sinh", "cosh"):
        return ~((a > (88.7 if f32 else 709.7)) & (a < (89.5 if f32 else 710.5)))
    if name == "tanh":
        return ~((a > (8.6 if f32 else 18.7)) & (a < (9.02 if f32 else 19.07)))
    return np.ones(x.shape, dtype=bool)


@pytest.mark.parity
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("name", _FUNCS)
def test_reference_parity_within_the_measured_ulp_bound(
    name: str, dtype: type, ref: ModuleType
) -> None:
    x = _grid(name, dtype)
    x = x[_ref_is_exact(name, x)]
    got = getattr(lucid, name)(_t(x)).numpy()
    want = getattr(ref, name)(ref.from_numpy(x)).numpy()
    d = _ulp(got, want)
    worst = int(d.max())
    assert worst <= _ULP_VS_REF[dtype][name], (worst, repr(x[int(d.argmax())]))
    # At the special points, wherever the reference is exact, so is Lucid.
    sp = _special_points(name, dtype)
    sp_want = getattr(ref, name)(ref.from_numpy(sp)).numpy()
    exact = _ulp(sp_want, _exact(name, sp)) == 0
    _assert_bits(getattr(lucid, name)(_t(sp)).numpy()[exact], sp_want[exact], sp[exact])


@pytest.mark.parity
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_mish_lstm_rope_and_randn_vforce_parity(dtype: type, ref: ModuleType) -> None:
    """CPU mish, LSTM, RoPE, and Box-Muller normal draws route through Vforce.h (LCD-608)."""
    l_dt = lucid.float32 if dtype is np.float32 else lucid.float64
    r_dt = ref.float32 if dtype is np.float32 else ref.float64
    atol = 1e-6 if dtype is np.float32 else 1e-12
    rtol = 1e-5 if dtype is np.float32 else 1e-12

    # 1. mish & mish_backward: calls cpu::vtanh(tile, tile, m) in-place (out == in).
    #    For x in [-16, -9] (f32) or [-35, -19] (f64), softplus(x) = exp(x) sits inside
    #    |sp| <= _ODD_X[dtype], directly exercising split_edges's out == in stack buffer
    #    without a (1 + t) term absorbing a 1-ulp error.
    rng = np.random.default_rng(608)
    xs = np.concatenate(
        [
            np.array(
                [-30.0, -20.0, -12.0, -10.0, -9.0, -1.0, -0.0, 0.0, 0.5, 2.0, 10.0],
                dtype=dtype,
            ),
            rng.uniform(-14.0, 4.0, 2055).astype(dtype),
        ]
    )
    lx = _t(xs)
    lx.requires_grad_(True)
    rx = ref.from_numpy(xs.copy()).requires_grad_(True)
    ly = F.mish(lx)
    ry = ref.nn.functional.mish(rx)
    np.testing.assert_allclose(ly.detach().numpy(), ry.detach().numpy(), rtol=rtol, atol=atol)
    ly.sum().backward()
    ry.sum().backward()
    assert lx.grad is not None and rx.grad is not None
    np.testing.assert_allclose(lx.grad.numpy(), rx.grad.numpy(), rtol=rtol, atol=atol)

    # 2. CPU LSTM (unprojected and projected, forward + backward) via vexp & vtanh.
    for proj in (0, 3):
        l_lstm = lucid.nn.LSTM(4, 6, proj_size=proj, dtype=l_dt)
        r_lstm = ref.nn.LSTM(4, 6, proj_size=proj, dtype=r_dt)
        with lucid.no_grad():
            for name_p, p in l_lstm.named_parameters():
                arr = rng.standard_normal(p.shape).astype(dtype) * 0.25
                p.copy_(_t(arr))
                getattr(r_lstm, name_p).data.copy_(ref.from_numpy(arr))
        x_seq = rng.standard_normal((5, 3, 4)).astype(dtype)
        lx_s = _t(x_seq)
        lx_s.requires_grad_(True)
        rx_s = ref.from_numpy(x_seq.copy()).requires_grad_(True)
        l_out, (l_hn, l_cn) = l_lstm(lx_s)
        r_out, (r_hn, r_cn) = r_lstm(rx_s)
        np.testing.assert_allclose(l_out.detach().numpy(), r_out.detach().numpy(), rtol=rtol, atol=atol)
        np.testing.assert_allclose(l_hn.detach().numpy(), r_hn.detach().numpy(), rtol=rtol, atol=atol)
        np.testing.assert_allclose(l_cn.detach().numpy(), r_cn.detach().numpy(), rtol=rtol, atol=atol)
        (l_out.sum() + l_hn.sum() + l_cn.sum()).backward()
        (r_out.sum() + r_hn.sum() + r_cn.sum()).backward()
        assert lx_s.grad is not None and rx_s.grad is not None
        np.testing.assert_allclose(lx_s.grad.numpy(), rx_s.grad.numpy(), rtol=rtol, atol=atol)
        for name_p, p in l_lstm.named_parameters():
            rp = getattr(r_lstm, name_p)
            assert p.grad is not None and rp.grad is not None
            np.testing.assert_allclose(p.grad.numpy(), rp.grad.numpy(), rtol=rtol, atol=atol)

    # 3. CPU RoPE: calls cpu::vcos(sp, cp, LDh) and in-place cpu::vsin(sp, sp, LDh).
    x_rope = rng.standard_normal((2, 5, 8)).astype(dtype)
    for interleaved in (False, True):
        got_rope = lucid.Tensor(
            lucid._C_engine.nn.rotary_pos_embedding(_t(x_rope)._impl, None, interleaved)
        ).numpy()
        theta = np.exp(-2.0 * np.arange(4, dtype=np.float64) * math.log(10000.0) / 8.0)
        angles = (np.arange(5, dtype=np.float64)[:, None] * theta[None, :]).astype(dtype)
        c = np.cos(angles.astype(np.float64)).astype(dtype)
        s = np.sin(angles.astype(np.float64)).astype(dtype)
        if interleaved:
            xe, xo = x_rope[..., 0::2], x_rope[..., 1::2]
            want_rope = np.empty_like(x_rope)
            want_rope[..., 0::2] = xe * c - xo * s
            want_rope[..., 1::2] = xo * c + xe * s
        else:
            x1, x2 = x_rope[..., :4], x_rope[..., 4:]
            want_rope = np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1)
        np.testing.assert_allclose(got_rope, want_rope, rtol=rtol, atol=atol)
        # Position 0 has angle == 0, where vcos is exact 1 and vsin is exact 0.
        np.testing.assert_array_equal(got_rope[:, 0, :], x_rope[:, 0, :])

    # 4. CPU Box-Muller normal_ / randn across tile boundary (odd numel = 2051).
    g1 = lucid.Generator(608)
    g2 = lucid.Generator(608)
    n1 = lucid.empty((2051,), dtype=l_dt).normal_(mean=1.5, std=0.75, generator=g1).numpy()
    n2 = lucid.empty((2051,), dtype=l_dt).normal_(mean=1.5, std=0.75, generator=g2).numpy()
    assert np.all(np.isfinite(n1))
    np.testing.assert_array_equal(n1, n2)
    np.testing.assert_array_equal(g1.get_state().numpy(), g2.get_state().numpy())
    assert abs(float(n1.mean()) - 1.5) < 0.08
    assert abs(float(n1.std()) - 0.75) < 0.08
