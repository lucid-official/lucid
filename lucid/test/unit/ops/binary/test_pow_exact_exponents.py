"""``x ** y`` is exact where the reference is, on Metal as on the CPU (LCD-549).

Metal ``pow`` was ``mx::power``, which is ``metal::pow``, for every exponent,
so the most common power was wrong: ``full(0.5) ** 2`` gave 0.24999997
(variance, MSE, squared norms, L2 penalties), ``2 ** -1`` 0.49999994, and
53-61% of float32 squares and reciprocals were not the correctly rounded
value.  Metal now runs one kernel
(``lucid/_C/backend/gpu/Pow.h``) for a tensor exponent, a scalar exponent
and a scalar base, and ``F.normalize`` reaches it too.  The CPU's tensor
exponent ran vForce's ``pow`` for every element, which is within 1 ulp but
not exact (``x ** full_like(x, 1.)`` was not ``x`` on 16% of values); it now
takes the scalar exponent's exact operations too.

``exact`` below is the reference's float64 answer rounded once to the dtype.
Each exponent is held to:

* 0, 1, 2, -1, 0.5: one correctly rounded operation (``1``, ``x``,
  ``x * x``, ``1 / x``, ``sqrt``), bit for bit on both devices and in every
  dtype.  0.5 and -0.5 keep IEEE ``pow`` at -0 and -inf (0.5: +0 and +inf;
  -0.5: +inf and +0), where the reference's sqrt and rsqrt answer -0, NaN,
  -inf and NaN.  The CPU answered IEEE already.
* 3, -2, -0.5: Metal float32 runs the reference's own formulas for 3 and -2
  (``x * x * x``, ``1 / (x * x)``), bit for bit its float32 answer, and
  rsqrt for -0.5.  Measured against ``exact``, ``x * x * x`` and rsqrt are
  within 1 ulp and ``1 / (x * x)`` within 2.
* Every other exponent: ``metal::pow``, as before, which is within 3 ulp of
  ``exact`` on Metal float32 (the measured bound on 6e5 values; LCD-557
  tracks a correctly rounded kernel).  The CPU (vForce) and the 16-bit floats
  are within 1 ulp.  Metal computes the special exponents of a 16-bit float
  in float32 and rounds once.
* An exact power of two raised to an integer (``2 ** n``, ``0.5 ** 4``) is
  exact on the CPU and in the 16-bit floats; Metal float32 is metal::pow
  there (LCD-557, pinned strict-xfail).
* A Metal float16 tensor exponent stays MLX's half pow, which is at the
  memory floor; the per-element dispatch cost it 1.6x.  Its specials are
  within 1 ulp there, and the cases these values hit are pinned
  strict-xfail (LCD-557).  A scalar float16 exponent is exact.

Metal flushes float32 subnormals, and so bfloat16 ones, to zero.  A
subnormal operand reads as a signed zero, and a subnormal answer comes back
as one.  float16 subnormals are normal float32 values, so they are kept.
"""

import math
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F
from lucid.test._fixtures.devices import device_dtype_params, metal_available

if TYPE_CHECKING:
    from _pytest.mark import ParameterSet

_DTYPES = [lucid.float32, lucid.float16, lucid.bfloat16]
_EXACT = [0.0, 1.0, 2.0, -1.0, 0.5]
_FORMULA = [3.0, -2.0, -0.5]
_GENERAL = [4.0, 1.5, 2.5, 0.3]
_EXPONENTS = _EXACT + _FORMULA + _GENERAL

_TINY = {lucid.float32: 2.0**-126, lucid.float16: 2.0**-14, lucid.bfloat16: 2.0**-126}
_BIG = {lucid.float32: 1e30, lucid.float16: 6e4, lucid.bfloat16: 1e30}
_SPAN = {lucid.float32: 40.0, lucid.float16: 9.0, lucid.bfloat16: 40.0}


# ── values and oracles ───────────────────────────────────────────────────────


def _round_bf16(v: np.ndarray) -> np.ndarray:
    """``v`` (float64) rounded once to bfloat16, ties to even, as float64."""
    out = v.copy()
    fin = np.isfinite(v) & (v != 0)
    _, e = np.frexp(v[fin])
    q = np.maximum(e, -125) - 8
    r = np.ldexp(np.round(np.ldexp(v[fin], -q)), q)
    big = (2.0 - 2.0**-7) * 2.0**127
    out[fin] = np.where(np.abs(r) > big, np.copysign(np.inf, r), r)
    return out


def _round_to(v: np.ndarray, dtype: lucid.dtype) -> np.ndarray:
    """``v`` (float64) rounded once to ``dtype``, held in float32."""
    with np.errstate(over="ignore"):
        if dtype is lucid.float16:
            return v.astype(np.float16).astype(np.float32)
        if dtype is lucid.bfloat16:
            return _round_bf16(v).astype(np.float32)
        return v.astype(np.float32)


def _bases(dtype: lucid.dtype) -> np.ndarray:
    """Values of ``dtype`` (held in float32): planted edges, then a spread."""
    rng = np.random.default_rng(549)
    tiny, big = _TINY[dtype], _BIG[dtype]
    planted = [0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0, -3.0, 0.1, -7.25]
    planted += [
        math.inf,
        -math.inf,
        math.nan,
        big,
        -big,
        tiny,
        -tiny,
        tiny / 16,
        -tiny / 16,
    ]
    wide = np.exp(rng.uniform(-_SPAN[dtype], _SPAN[dtype], 600))
    spread = np.concatenate([rng.standard_normal(600) * 3, wide, -wide[:200]])
    return _round_to(np.concatenate([planted, spread]).astype(np.float64), dtype)


def _rounded_exponent(e: float, dtype: lucid.dtype) -> float:
    # A scalar exponent is rounded to the tensor's dtype first, on both sides.
    return float(_round_to(np.array([e]), dtype)[0])


def _metal_flush(x: np.ndarray, device: str, dtype: lucid.dtype) -> np.ndarray:
    if device != "metal" or dtype is lucid.float16:
        return x
    return np.where(np.abs(x) < 2.0**-126, np.copysign(np.float32(0), x), x).astype(
        np.float32
    )


def _exact(
    ref: ModuleType, x: np.ndarray, e: float, dtype: lucid.dtype, device: str
) -> np.ndarray:
    """The reference's float64 power rounded once, under Lucid's IEEE contract."""
    if e == 1.0:
        return x  # a copy: no arithmetic, so nothing is flushed
    xin = _metal_flush(x, device, dtype).astype(np.float64)
    r = (ref.tensor(xin, dtype=ref.float64) ** e).numpy()
    if e == 0.5:
        r = np.where(xin == -math.inf, math.inf, r + 0.0)
    elif e == -0.5:
        r = np.where(xin == 0, math.inf, np.where(xin == -math.inf, 0.0, r))
    return _metal_flush(_round_to(r, dtype), device, dtype)


def _reference_f32(ref: ModuleType, x: np.ndarray, e: float) -> np.ndarray:
    return (ref.tensor(x, dtype=ref.float32) ** e).numpy()


def _ordered(v: np.ndarray, dtype: lucid.dtype) -> np.ndarray:
    if dtype is lucid.float16:
        i, bits = v.astype(np.float16).view(np.int16).astype(np.int64), 15
    elif dtype is lucid.bfloat16:
        i, bits = (v.view(np.int32) >> 16).astype(np.int64), 15
    else:
        i, bits = v.view(np.int32).astype(np.int64), 31
    return np.where(i < 0, -(i & ((1 << bits) - 1)), i)


def _assert_within(
    got: np.ndarray,
    want: np.ndarray,
    x: np.ndarray,
    dtype: lucid.dtype,
    max_ulp: int,
    what: str,
) -> None:
    both_nan = np.isnan(got) & np.isnan(want)
    finite = np.isfinite(want) & np.isfinite(got)
    dist = np.where(finite, np.abs(_ordered(got, dtype) - _ordered(want, dtype)), 0)
    same_sign = np.signbit(got) == np.signbit(want)
    ok = both_nan | (finite & (dist <= max_ulp) & ((want != 0) | same_sign))
    ok |= ~np.isfinite(want) & (got == want)
    if not ok.all():
        idx = np.flatnonzero(~ok)[:6]
        rows = [(float(x[i]), float(got[i]), float(want[i]), int(dist[i])) for i in idx]
        raise AssertionError(
            f"{what}: {int((~ok).sum())} off by more than {max_ulp} ulp, "
            f"(x, got, want, ulp) = {rows}"
        )


def _max_ulp(e: float, dtype: lucid.dtype, device: str) -> int:
    if e in _EXACT:
        return 0
    if dtype is not lucid.float32 or device == "cpu":
        return 1
    if e == -2.0:
        return 2
    return 1 if e in _FORMULA else 3


def _values(t: lucid.Tensor) -> np.ndarray:
    return t.cpu().to(lucid.float32).numpy()


def _on(x: np.ndarray, dtype: lucid.dtype, device: str) -> lucid.Tensor:
    return lucid.tensor(x).to(dtype).to(device)


# ── the reported value ───────────────────────────────────────────────────────


def test_reported_square_of_a_half_is_a_quarter(device: str) -> None:
    out = lucid.full((16,), 0.5, device=device) ** 2
    assert out.tolist() == [0.25] * 16


# ── scalar exponent, tensor exponent, every dtype ────────────────────────────


@pytest.mark.parametrize("e", _EXPONENTS)
@pytest.mark.parametrize(("device", "dtype"), device_dtype_params(_DTYPES))
def test_scalar_exponent_matches_the_reference(
    ref: ModuleType, device: str, dtype: lucid.dtype, e: float
) -> None:
    x = _bases(dtype)
    t = _on(x, dtype, device)
    want = _exact(ref, x, _rounded_exponent(e, dtype), dtype, device)
    for name, out in (
        ("**", t**e),
        ("lucid.pow", lucid.pow(t, e)),
        ("Tensor.pow", t.pow(e)),
    ):
        assert out.dtype == dtype
        _assert_within(
            _values(out), want, x, dtype, _max_ulp(e, dtype, device), f"x {name} {e}"
        )


_F16_TENSOR = (
    "LCD-557: float16 tensor-exponent specials ride MLX's half pow (<=1 ulp on "
    "80/28/46/24/5 of 65536 values for 2/3/-1/-2/0.5)"
)


def _strict_xfail(broken: bool, reason: str) -> list[pytest.MarkDecorator]:
    return [pytest.mark.xfail(strict=True, reason=reason)] if broken else []


def _tensor_exponent_params() -> list[ParameterSet]:
    # Metal float16 keeps MLX's half pow for a tensor exponent: these bases
    # meet its misses at 2 and 3 (-1, -2 and 0.5 miss on other values).
    out = []
    for p in device_dtype_params(_DTYPES):
        device, dtype = p.values
        for e in _EXPONENTS:
            broken = device == "metal" and dtype is lucid.float16 and e in (2.0, 3.0)
            marks = _strict_xfail(broken, _F16_TENSOR)
            out.append(pytest.param(device, dtype, e, marks=marks, id=f"{p.id}-{e}"))
    return out


@pytest.mark.parametrize(("device", "dtype", "e"), _tensor_exponent_params())
def test_tensor_exponent_matches_the_scalar_exponent(
    ref: ModuleType, device: str, dtype: lucid.dtype, e: float
) -> None:
    # The per-element dispatch gives a tensor of 2s what ``x ** 2`` gets.
    x = _bases(dtype)
    t = _on(x, dtype, device)
    ex = lucid.full_like(t, e)
    got = _values(t**ex)
    want = _exact(ref, x, float(_values(ex)[0]), dtype, device)
    _assert_within(got, want, x, dtype, _max_ulp(e, dtype, device), f"x ** tensor({e})")
    np.testing.assert_array_equal(got, _values(t**e))
    # ``ex`` is evaluated now: on Metal a broadcast of one value, read as one.
    np.testing.assert_array_equal(_values(t**ex), got)


@pytest.mark.skipif(not metal_available(), reason="Metal only")
@pytest.mark.parametrize("e", [3.0, -2.0])
def test_metal_runs_the_reference_formula_for_three_and_minus_two(
    ref: ModuleType, e: float
) -> None:
    # Where ``x * x`` or the answer is subnormal Metal flushes it; the
    # oracle test above holds those elements.
    x = _bases(lucid.float32)
    want = _reference_f32(ref, x, e)
    with np.errstate(over="ignore"):
        normal = (np.abs(x * x) >= 2.0**-126) & (np.abs(want) >= 2.0**-126)
    got = _values(_on(x, lucid.float32, "metal") ** e)
    np.testing.assert_array_equal(got[normal], want[normal])


# ── signs, zeros, infinities ─────────────────────────────────────────────────


_IEEE_CASES = [
    (-2.0, 3.0, -8.0),
    (-2.0, 2.0, 4.0),
    (-2.0, -1.0, -0.5),
    (-2.0, -2.0, 0.25),
    (-2.0, 4.0, 16.0),
    (-2.0, 5.0, -32.0),
    (-3.0, 3.0, -27.0),
    (-8.0, 1.0 / 3.0, math.nan),
    (-2.0, 0.5, math.nan),
    (0.0, 0.0, 1.0),
    (math.nan, 0.0, 1.0),
    (1.0, math.nan, 1.0),
    (0.0, -1.0, math.inf),
    (-0.0, -1.0, -math.inf),
    (-0.0, -2.0, math.inf),
    (-0.0, 3.0, -0.0),
    (-0.0, 2.0, 0.0),
    (-0.0, 0.5, 0.0),
    (-0.0, -0.5, math.inf),
    (0.0, -3.0, math.inf),
    (-math.inf, 3.0, -math.inf),
    (-math.inf, -3.0, -0.0),
    (-math.inf, 2.0, math.inf),
    (-math.inf, 0.5, math.inf),
    (-math.inf, -0.5, 0.0),
    (math.inf, -1.0, 0.0),
    (math.nan, 2.0, math.nan),
    (-1.0, math.inf, 1.0),
]


@pytest.mark.parametrize(("x", "e", "want"), _IEEE_CASES)
def test_special_values_follow_ieee_pow(
    device: str, x: float, e: float, want: float
) -> None:
    for out in (
        lucid.tensor([x], device=device) ** e,
        lucid.tensor([x], device=device) ** lucid.tensor([e], device=device),
    ):
        got = out.item()
        if math.isnan(want):
            assert math.isnan(got)
        else:
            assert got == want
            assert math.copysign(1.0, got) == math.copysign(1.0, want)


def _tie_params() -> list[ParameterSet]:
    out = []
    for device in ["cpu", "metal"] if metal_available() else ["cpu"]:
        for form in ("scalar", "tensor"):
            broken = device == "metal" and form == "tensor"
            marks = _strict_xfail(broken, _F16_TENSOR)
            out.append(pytest.param(device, form, marks=marks, id=f"{device}-{form}"))
    return out


@pytest.mark.parametrize(("device", "form"), _tie_params())
def test_float16_tie_rounds_to_even(ref: ModuleType, device: str, form: str) -> None:
    # 12.75 ** 2 = 162.5625 is a float16 tie, so it rounds to the even 162.5.
    # The CPU's vForce answer was 1 float32 ulp above it and went to 162.625.
    x = lucid.tensor([12.75, 0.24609375], dtype=lucid.float16, device=device)
    rx = ref.tensor([12.75, 0.24609375], dtype=ref.float16)
    e = 2.0 if form == "scalar" else lucid.full_like(x, 2.0)
    assert (x**e).tolist() == (rx**2).tolist() == [162.5, 0.060546875]


# ── a scalar base ────────────────────────────────────────────────────────────


def _power_of_two_params() -> list[ParameterSet]:
    # Metal float32 is metal::pow for a non-special exponent, up to 3 ulp
    # off even here (2 ** -3 is 0.12499999).  The 16-bit floats round that
    # error away; the CPU's vForce answers these exactly.
    out = []
    for p in device_dtype_params(_DTYPES):
        device, dtype = p.values
        broken = device == "metal" and dtype is lucid.float32
        marks = _strict_xfail(broken, "LCD-557: metal::pow misses 2**n")
        out.append(pytest.param(device, dtype, marks=marks, id=p.id))
    return out


@pytest.mark.parametrize("base", [2.0, 0.5, 4.0, -2.0, 0.125])
@pytest.mark.parametrize(("device", "dtype"), _power_of_two_params())
def test_power_of_two_base_with_an_integer_exponent_is_exact(
    ref: ModuleType, device: str, dtype: lucid.dtype, base: float
) -> None:
    n = np.arange(-40, 41, dtype=np.float64)
    if dtype is lucid.float16:
        n = n[np.abs(n * math.log2(abs(base))) <= 15]
    x = _round_to(n, dtype)
    got = _values(base ** _on(x, dtype, device))
    raw = ref.tensor(base, dtype=ref.float64) ** ref.tensor(n, dtype=ref.float64)
    want = _metal_flush(_round_to(raw.numpy(), dtype), device, dtype)
    _assert_within(got, want, x, dtype, 0, f"{base} ** n")
    if base > 0:
        _assert_within(
            _values(lucid.pow(base, _on(x, dtype, device))), want, x, dtype, 0, "pow"
        )


@pytest.mark.parametrize(("device", "dtype"), device_dtype_params(_DTYPES))
def test_other_scalar_bases_match_the_reference(
    ref: ModuleType, device: str, dtype: lucid.dtype
) -> None:
    e = _bases(dtype)
    e = e[np.isfinite(e) & (np.abs(e) < 30)]
    for base in (10.0, 3.0, math.e):
        b = _rounded_exponent(base, dtype)
        got = _values(base ** _on(e, dtype, device))
        raw = ref.tensor(b, dtype=ref.float64) ** ref.tensor(e.astype(np.float64))
        want = _metal_flush(_round_to(raw.numpy(), dtype), device, dtype)
        tol = 3 if (device == "metal" and dtype is lucid.float32) else 1
        _assert_within(got, want, e, dtype, tol, f"{base} ** e")


# ── layouts: views, broadcasts, 0-d, empty ───────────────────────────────────


def test_strided_and_broadcast_operands_take_the_same_paths(device: str) -> None:
    x = lucid.tensor(
        np.arange(1, 25, dtype=np.float32).reshape(4, 6) / 8, device=device
    )
    cpu = x.cpu()
    # The exact exponents only, so the CPU's answer is the oracle.
    exps = lucid.tensor([0.0, 1.0, 2.0, -1.0, 0.5, 2.0], device=device)
    cases = {
        "transposed": (lambda t, _e: t.mT**2),
        "sliced": (lambda t, _e: t[:, ::2] ** -1),
        "flipped": (lambda t, _e: lucid.flip(t, dims=[1]) ** 0.5),
        "row exponent": (lambda t, e: t**e),
        "column exponent": (lambda t, e: t[:, :1] ** e),
        "transposed base, row exponent": (lambda t, e: t.mT[:6, :4].mT ** e),
        "0-d exponent": (lambda t, e: t ** e[2]),
        "0-d base": (lambda t, e: t[1, 2] ** e),
    }
    for name, fn in cases.items():
        got = fn(x, exps)
        want = fn(cpu, exps.cpu())
        np.testing.assert_array_equal(_values(got), _values(want), err_msg=name)
        assert got.shape == want.shape, name


def test_zero_dim_and_empty_tensors(device: str) -> None:
    s = lucid.tensor(0.5, device=device)
    assert (s**2).shape == ()
    assert (s**2).item() == 0.25
    assert (s ** lucid.tensor(-1.0, device=device)).item() == 2.0
    assert (2.0 ** lucid.tensor(-1.0, device=device)).item() == 0.5
    empty = lucid.zeros((0, 3), device=device)
    assert (empty**2).shape == (0, 3)
    assert (empty ** lucid.ones((1, 3), device=device)).shape == (0, 3)
    assert (2.0**empty).shape == (0, 3)


# ── gradients ────────────────────────────────────────────────────────────────


_GRAD_X = [-2.0, -0.5, 0.0, 0.5, 1.5, 3.0]


@pytest.mark.parametrize("e", [1.0, 2.0, 3.0, 4.0, -1.0, -2.0, 0.5, -0.5, 2.5])
def test_gradient_is_the_power_rule(ref: ModuleType, device: str, e: float) -> None:
    # d/dx x**k = k * x**(k - 1), at zero and at negative bases too.
    ref_x = ref.tensor(_GRAD_X, requires_grad=True)
    (ref_x**e).sum().backward()
    want = ref_x.grad.numpy()
    for spelling in ("scalar", "tensor"):
        x = lucid.tensor(_GRAD_X, requires_grad=True, device=device)
        y = x**e if spelling == "scalar" else x ** lucid.full_like(x, e)
        y.sum().backward()
        np.testing.assert_allclose(
            _values(x.grad), want, rtol=1e-6, atol=0, equal_nan=True, err_msg=spelling
        )


@pytest.mark.parametrize("e", [2.0, 3.0, -1.0])
def test_second_derivative_is_the_power_rule(device: str, e: float) -> None:
    x = lucid.tensor([-2.0, 0.5, 3.0], requires_grad=True, device=device)
    (g,) = lucid.autograd.grad((x**e).sum(), [x], create_graph=True)
    (g2,) = lucid.autograd.grad(g.sum(), [x])
    xs = np.array([-2.0, 0.5, 3.0])
    np.testing.assert_allclose(_values(g), e * xs ** (e - 1), rtol=1e-6)
    np.testing.assert_allclose(_values(g2), e * (e - 1) * xs ** (e - 2), rtol=1e-6)


@pytest.mark.xfail(
    strict=True, reason="LCD-559: pow backward does not mask exponent 0 at x=0"
)
def test_gradient_of_the_zeroth_power_at_zero_is_zero(device: str) -> None:
    x = lucid.tensor([0.0, 1.5], requires_grad=True, device=device)
    (x**0).sum().backward()
    assert x.grad.tolist() == [0.0, 0.0]


# ── F.normalize reaches the same kernel ──────────────────────────────────────


@pytest.mark.skipif(not metal_available(), reason="Metal only")
def test_reported_normalize_of_equal_halves_is_exact() -> None:
    out = F.normalize(lucid.tensor([[0.5, 0.5]], device="metal"), p=2.0, dim=1)
    assert out.tolist() == [[0.7071067690849304, 0.7071067690849304]]


@pytest.mark.parametrize("p", [1.0, 2.0, 3.0])
def test_normalize_matches_the_reference(
    ref: ModuleType, device: str, p: float
) -> None:
    rng = np.random.default_rng(int(p))
    x = rng.standard_normal((5, 4)).astype(np.float32)
    x[0] = [0.5, 0.5, 0.5, 0.5]
    t = lucid.tensor(x, requires_grad=True, device=device)
    rt = ref.tensor(x, requires_grad=True)
    out = F.normalize(t, p=p, dim=1)
    rout = ref.nn.functional.normalize(rt, p=p, dim=1)
    w = rng.standard_normal((5, 4)).astype(np.float32)
    (out * lucid.tensor(w, device=device)).sum().backward()
    (rout * ref.tensor(w)).sum().backward()
    rtol = 1e-6 if p != 3.0 else 2e-6
    np.testing.assert_allclose(_values(out), rout.detach().numpy(), rtol=rtol, atol=0)
    np.testing.assert_allclose(_values(t.grad), rt.grad.numpy(), rtol=1e-5, atol=1e-7)
    if p != 3.0:
        np.testing.assert_array_equal(_values(out)[0], rout.detach().numpy()[0])
