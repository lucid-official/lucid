"""``ldexp`` and ``frexp`` are exact at every magnitude, and so are their gradients.

``x * 2**e`` is exact whenever it is representable, and so is the mantissa
``x * 2**-e`` that ``frexp`` returns.  So every value here is compared bit
for bit, on every device, at every magnitude: subnormal, normal and huge
operands, ``±0``, ``±inf`` and NaN, and exponents across ``[-160, 160]``.
The gradients are compared too: ``d ldexp / dx = 2**e``,
``d ldexp / de = x * 2**e * ln 2`` and ``d frexp / dx = 2**-e``.

Metal flushes float32 subnormals on the way in and on the way out, so
eager Metal reads and writes them as bits.  Its gradient used to be cut
there, and was 0 at a subnormal ``x`` (LCD-468).  A gradient is exact
wherever it is a normal number, whatever ``x`` is.  A gradient that is
itself subnormal may flush on Metal, as every Metal backward flushes one
(:func:`_assert_grad`).

The oracle is the reference evaluated in float64 and rounded once to the
dtype under test, which is the exact answer.  The reference's own float32
``ldexp`` is not exact.  For a float exponent it forms ``2**e`` first,
and that leaves float32 for ``|e| > 127``: ``ldexp(0, 160.)`` is NaN.  Its
derivative for an integer exponent is computed in integer arithmetic,
where ``2**-130`` is 0.
"""

import decimal
import math
import struct
from collections.abc import Callable
from types import ModuleType

import pytest

import lucid
from lucid._C import _C_engine
from lucid.test._fixtures.devices import metal_available

_INF, _NAN = math.inf, math.nan
_LN2 = math.log(2.0)

_F32_SUBNORMAL = [
    1.401298464324817e-45,  # the smallest, 2**-149
    4.203895392974451e-45,
    1e-40,
    3e-39,
    1.1754942106924411e-38,  # the largest
]
_F32_NORMAL = [
    1.1754943508222875e-38,  # the smallest, 2**-126
    1e-30,
    0.75,
    1.0,
    3.0,
    1e30,
    3.4028234663852886e38,  # the largest
]
_SPECIAL = [0.0, -0.0, _INF, -_INF, _NAN]
_EXPONENTS = [
    -160, -150, -149, -140, -130, -127, -126, -125, -100, -10, -1,
    0, 1, 10, 100, 125, 127, 128, 140, 149, 160,
]  # fmt: skip

_F32_X = [s * v for v in _F32_SUBNORMAL + _F32_NORMAL for s in (1.0, -1.0)] + _SPECIAL

#: 16-bit operands, as (dtype name, values, exponents).  The products stay
#: inside the dtype's range so the oracle is the dtype's own rounding.
_HALF_CASES = [
    (
        "float16",
        [6e-8, -6e-8, 1e-5, 0.75, -3.0, 60000.0, 0.0, -0.0, _INF, _NAN],
        [-30, -20, -10, -1, 0, 1, 5, 10, 20, 30],
    ),
    (
        "bfloat16",
        [1e-30, 0.75, -3.0, 1e30, 3e38, 0.0, -0.0, _INF, -_INF, _NAN],
        [-160, -100, -10, -1, 0, 1, 10, 100, 128, 160],
    ),
]


def _grid(xs: list[float], es: list[int]) -> tuple[list[float], list[int]]:
    """Every pair of an operand and an exponent, flattened."""
    return [x for x in xs for _ in es], [e for _ in xs for e in es]


def _bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", v))[0]


def _assert_same(got: list[float], want: list[float], what: str) -> None:
    """``got`` equals ``want`` element-wise, a zero's sign and NaN included."""
    for i, (g, w) in enumerate(zip(got, want, strict=True)):
        if math.isnan(w):
            assert math.isnan(g), f"{what}[{i}]: got {g!r}, want nan"
        else:
            assert g == w, f"{what}[{i}]: got {g!r}, want {w!r}"
            assert math.copysign(1.0, g) == math.copysign(1.0, w), f"{what}[{i}] sign"


def _assert_close(got: list[float], want: list[float], what: str, *, rel_tol: float = 3e-7) -> None:
    """``got`` is within rounding of ``want``: ``rel_tol`` (about 2.5 float32
    ulp by default), or two steps of the float32 subnormal grid."""
    for i, (g, w) in enumerate(zip(got, want, strict=True)):
        if math.isnan(w) or math.isinf(w):
            _assert_same([g], [w], f"{what}[{i}]")
        else:
            close = math.isclose(g, w, rel_tol=rel_tol, abs_tol=3e-45)
            assert close, f"{what}[{i}]: got {g!r}, want {w!r}"


def _assert_grad(
    got: list[float], want: list[float], what: str, *, exact: bool, rel_tol: float = 3e-7
) -> None:
    """A gradient: ``want`` where that is normal, zero or non-finite.

    A subnormal ``want`` only asks for a subnormal or zero of its sign.
    The CPU computes it and Metal flushes it.
    """
    tiny = 2.0**-126
    for i, (g, w) in enumerate(zip(got, want, strict=True)):
        if w != 0.0 and abs(w) < tiny:
            flushed_or_subnormal = abs(g) < tiny and g * w >= 0.0
            assert flushed_or_subnormal, f"{what}[{i}]: got {g!r}, want {w!r}"
        elif exact:
            _assert_same([g], [w], f"{what}[{i}]")
        else:
            _assert_close([g], [w], f"{what}[{i}]", rel_tol=rel_tol)


def _f32(v: float) -> float:
    """``v`` rounded to float32, as a tensor of it holds it."""
    return float(struct.unpack("<f", struct.pack("<f", v))[0])


def _ref_ldexp(
    ref: ModuleType, xs: list[float], es: list[int], dtype_name: str
) -> tuple[list[float], list[float], list[float]]:
    """The reference's float64 ``ldexp`` of ``xs`` (as ``dtype_name`` holds
    them) and its two gradients: the value rounded to ``dtype_name``, the
    gradients to float32."""
    held = ref.tensor(xs, dtype=getattr(ref, dtype_name)).to(ref.float64).tolist()
    rx = ref.tensor(held, dtype=ref.float64, requires_grad=True)
    re = ref.tensor([float(e) for e in es], dtype=ref.float64, requires_grad=True)
    y = ref.ldexp(rx, re)
    y.sum().backward()
    out = y.detach().to(getattr(ref, dtype_name)).to(ref.float32).tolist()
    return out, rx.grad.to(ref.float32).tolist(), re.grad.to(ref.float32).tolist()


@pytest.mark.parametrize("exp_dtype", ["int32", "float32"])
def test_ldexp_float32_is_exact_at_every_magnitude(
    ref: ModuleType, device: str, exp_dtype: str
) -> None:
    xs, es = _grid(_F32_X, _EXPONENTS)
    want, _, _ = _ref_ldexp(ref, xs, es, "float32")
    x = lucid.tensor(xs, dtype=lucid.float32, device=device)
    e = lucid.tensor(es, dtype=getattr(lucid, exp_dtype), device=device)
    got = lucid.ldexp(x, e).to("cpu").tolist()
    _assert_same(got, want, "ldexp")


# Whether the exponent needs a gradient decides the path off Metal: the
# plain composition, or the written-out derivatives.
@pytest.mark.parametrize("exp_grad", [False, True])
def test_ldexp_float32_gradients_are_those_of_the_exact_product(
    ref: ModuleType, device: str, exp_grad: bool
) -> None:
    xs, es = _grid(_F32_X, _EXPONENTS)
    _, want_gx, want_ge = _ref_ldexp(ref, xs, es, "float32")
    x = lucid.tensor(xs, dtype=lucid.float32, device=device, requires_grad=True)
    e = lucid.tensor(
        [float(v) for v in es], dtype=lucid.float32, device=device, requires_grad=exp_grad
    )
    lucid.ldexp(x, e).sum().backward()
    assert x.grad is not None
    # d/dx = 2**e is one power of two, rounded once: exact.
    _assert_grad(x.grad.to("cpu").tolist(), want_gx, "d ldexp / dx", exact=True)
    if exp_grad:
        assert e.grad is not None
        got_ge = e.grad.to("cpu").tolist()
        _assert_grad(got_ge, want_ge, "d ldexp / de", exact=False)


def _exact_ldexp(x: float, e: float, scale: float = 1.0) -> float:
    """``scale * x * 2**e`` to 40 digits, then to the nearest double."""
    with decimal.localcontext() as ctx:
        ctx.prec = 40
        two = decimal.Decimal(2)
        exact = decimal.Decimal(x) * two ** decimal.Decimal(e) * decimal.Decimal(scale)
        return float(exact)


#: Operands far from 1, and fractional exponents.  ``e + log2(x)`` taken in
#: the operand's dtype rounds away most of a fraction there (LCD-468 review).
_FRACTIONAL_E = [2.7, -1.5, 0.3, -7.25, 13.6, 20.7]
_FAR_X = {
    "float32": [1e30, -2.5e-31, 0.77 * 2.0**100, -0.61 * 2.0**-100, 3.0, -0.75, 1e-40],
    "float64": [0.9 * 2.0**900, -0.6 * 2.0**-900, 1.7e300, -3e-300, 3.0],
}


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_ldexp_fractional_exponent_gradient_is_within_a_few_ulp(
    device: str, dtype_name: str
) -> None:
    if device == "metal" and dtype_name == "float64":
        pytest.skip("Metal has no float64")
    dt = getattr(lucid, dtype_name)
    held = lucid.tensor(_FAR_X[dtype_name], dtype=dt).tolist()
    es = lucid.tensor(_FRACTIONAL_E, dtype=dt).tolist()
    xs, ks = [x for x in held for _ in es], [e for _ in held for e in es]
    x = lucid.tensor(xs, dtype=dt, device=device, requires_grad=True)
    k = lucid.tensor(ks, dtype=dt, device=device, requires_grad=True)
    y = lucid.ldexp(x, k)
    y.sum().backward()
    assert k.grad is not None and x.grad is not None
    want_y = lucid.tensor([_exact_ldexp(a, b) for a, b in zip(xs, ks, strict=True)], dtype=dt)
    want_ge = [_exact_ldexp(a, b, _LN2) for a, b in zip(xs, ks, strict=True)]
    want_gx = [_exact_ldexp(1.0, b) for b in ks]
    # Three ulp of the dtype.
    rel = 3.6e-7 if dtype_name == "float32" else 6.7e-16
    rounded = [lucid.tensor(w, dtype=dt).item() for w in want_ge]
    _assert_grad(k.grad.to("cpu").tolist(), rounded, "d/de", exact=False, rel_tol=rel)
    _assert_close(x.grad.to("cpu").tolist(), want_gx, "d/dx", rel_tol=rel)
    _assert_close(y.detach().to("cpu").tolist(), want_y.tolist(), "ldexp", rel_tol=rel)


def test_ldexp_fractional_exponent_reaches_subnormal_answers(device: str) -> None:
    # Metal shifted a subnormal x by 149 and wrote a subnormal answer from
    # trunc(e), dropping the fraction: ldexp(1, -130.5) came out 2**-130.
    # Then from e + 149, which is coarser than a small fraction of e: from
    # 2**-126 and -0.01 it came out 32 units of 2**-149 high.
    cases = [
        (1e-40, 2.7),
        (3e-39, 0.3),
        (1.0, -130.5),
        (0.75, -128.3),
        (1.5, -149.25),
        (2.0**-126, -0.01),
        (1.5 * 2.0**-126, -0.7),
        (2.0**-100, -26.6),
        (2.0**-80, -46.3),
        (2.0**-60, -66.1),  # e and e + 149 share a binade: exact either way
    ]
    xs = lucid.tensor([c[0] for c in cases], dtype=lucid.float32).tolist()
    es = lucid.tensor([c[1] for c in cases], dtype=lucid.float32).tolist()
    want = lucid.tensor([_exact_ldexp(a, b) for a, b in zip(xs, es, strict=True)])
    x = lucid.tensor(xs, dtype=lucid.float32, device=device)
    e = lucid.tensor(es, dtype=lucid.float32, device=device)
    got = lucid.ldexp(x, e).to("cpu").tolist()
    # Two units of 2**-149: the answers are all subnormal.
    _assert_close(got, want.tolist(), "ldexp", rel_tol=1e-9)


def test_ldexp_gradient_does_not_depend_on_a_subnormal_input(device: str) -> None:
    # The LCD-468 report: on Metal the gradient was 0 wherever x was subnormal.
    x = lucid.tensor(
        [1e-40, 3e-39, 1e-45, 1.0], dtype=lucid.float32, device=device, requires_grad=True
    )
    e = lucid.tensor([10, 3, 100, 1], device=device)
    lucid.ldexp(x, e).sum().backward()
    assert x.grad is not None
    assert x.grad.to("cpu").tolist() == [1024.0, 8.0, 2.0**100, 2.0]


@pytest.mark.parametrize(("dtype_name", "xs", "es"), _HALF_CASES)
def test_ldexp_half_is_exact_and_differentiable(
    ref: ModuleType, device: str, dtype_name: str, xs: list[float], es: list[int]
) -> None:
    gx_, ge_ = _grid(xs, es)
    want, want_gx, _ = _ref_ldexp(ref, gx_, ge_, dtype_name)
    dt = getattr(lucid, dtype_name)
    x = lucid.tensor(gx_, dtype=dt, device=device, requires_grad=True)
    e = lucid.tensor(ge_, dtype=lucid.int32, device=device)
    y = lucid.ldexp(x, e)
    assert y.dtype == dt
    _assert_same(y.to(dtype=lucid.float32).to("cpu").tolist(), want, "ldexp")
    y.sum().backward()
    assert x.grad is not None
    want_gx_dt = ref.tensor(want_gx, dtype=ref.float32).to(getattr(ref, dtype_name)).float()
    _assert_grad(
        x.grad.to(dtype=lucid.float32).to("cpu").tolist(),
        want_gx_dt.tolist(),
        "d ldexp / dx",
        exact=True,
    )


@pytest.mark.parametrize(
    ("dtype_name", "xs"), [("float32", _F32_X), *((name, xs) for name, xs, _ in _HALF_CASES)]
)
def test_frexp_is_exact_and_its_mantissa_gradient_is_two_to_minus_e(
    ref: ModuleType, device: str, dtype_name: str, xs: list[float]
) -> None:
    r_dt, l_dt = getattr(ref, dtype_name), getattr(lucid, dtype_name)
    rm, re = ref.frexp(ref.tensor(xs, dtype=r_dt))
    # The gradient's oracle is 2**-e itself, rounded to the dtype.  The
    # reference divides by exp2(e) taken in float32, which is inf at
    # e = 128, so its gradient at the largest float is 0, not 2**-128.
    want_grad = (
        ref.tensor([math.ldexp(1.0, -v) for v in re.tolist()], dtype=ref.float64)
        .to(r_dt)
        .float()
        .tolist()
    )
    x = lucid.tensor(xs, dtype=l_dt, device=device, requires_grad=True)
    m, e = lucid.frexp(x)
    assert m.dtype == l_dt and e.dtype == lucid.int32
    assert e.to("cpu").tolist() == re.tolist()
    _assert_same(
        m.to(dtype=lucid.float32).to("cpu").tolist(), rm.detach().float().tolist(), "mantissa"
    )
    m.sum().backward()
    assert x.grad is not None
    _assert_grad(
        x.grad.to(dtype=lucid.float32).to("cpu").tolist(), want_grad, "d mantissa / dx", exact=True
    )


def test_ldexp_second_derivatives_reach_both_arguments(device: str) -> None:
    xs, ks = [_f32(1e-40), 0.75, _f32(-3e-39), -1.25], [10.0, -4.0, 3.0, 0.0]
    x = lucid.tensor(xs, dtype=lucid.float32, device=device, requires_grad=True)
    k = lucid.tensor(ks, dtype=lucid.float32, device=device, requires_grad=True)
    gx, gk = lucid.autograd.grad(lucid.ldexp(x, k).sum(), [x, k], create_graph=True)
    d_gk_dx, d_gk_dk = lucid.autograd.grad(gk.sum(), [x, k], retain_graph=True)
    (d_gx_dk,) = lucid.autograd.grad(gx.sum(), [k])
    two_k = [2.0**v for v in ks]
    _assert_close(d_gk_dx.to("cpu").tolist(), [p * _LN2 for p in two_k], "d2/dkdx")
    _assert_close(d_gx_dk.to("cpu").tolist(), [p * _LN2 for p in two_k], "d2/dxdk")
    # At a subnormal x the exponent's own second derivative is a plain
    # product through 2**-164, which Metal flushes; it is checked at the
    # normal operands only.
    _assert_close(
        d_gk_dk.to("cpu").tolist()[1::2],
        [v * p * _LN2**2 for v, p in zip(xs, two_k, strict=True)][1::2],
        "d2/dk2",
    )


def test_ldexp_gradients_reduce_to_broadcast_operands(device: str) -> None:
    x = lucid.tensor([1e-40, 0.75, 3.0], dtype=lucid.float32, device=device, requires_grad=True)
    k = lucid.tensor([[1.0], [-2.0]], dtype=lucid.float32, device=device, requires_grad=True)
    lucid.ldexp(x, k).sum().backward()
    assert x.grad is not None and k.grad is not None
    assert x.grad.shape == (3,) and k.grad.shape == (2, 1)
    assert x.grad.to("cpu").tolist() == [2.25, 2.25, 2.25]
    _assert_close(
        k.grad.to("cpu").reshape(-1).tolist(), [3.75 * 2.0 * _LN2, 3.75 * 0.25 * _LN2], "d/dk"
    )


def test_metal_ldexp_saves_its_operand_only_for_the_exponents_gradient(
    device_gpu_only: str,
) -> None:
    # d/dx = 2**e does not read x, so writing x after the op is fine, as in
    # the reference.  d/de reads it, so then the write is refused.  The CPU
    # composite still refuses the first write: its mul saves x (LCD-584).
    x0 = lucid.tensor([1.0, 3.0], device=device_gpu_only, requires_grad=True)
    x = x0 * 1.0
    y = lucid.ldexp(x, lucid.tensor([2, 3], device=device_gpu_only))
    x.mul_(2.0)
    y.sum().backward()
    assert x0.grad is not None
    assert x0.grad.to("cpu").tolist() == [4.0, 8.0]

    x = x0 * 1.0
    k = lucid.tensor([2.0, 3.0], device=device_gpu_only, requires_grad=True)
    y = lucid.ldexp(x, k)
    x.mul_(2.0)
    with pytest.raises(_C_engine.VersionMismatch):
        y.sum().backward()


def test_frexp_returns_a_nan_operand_as_given(device: str) -> None:
    neg_nan = struct.unpack("<f", struct.pack("<I", 0xFFC00000))[0]
    x = lucid.tensor([neg_nan, _NAN], dtype=lucid.float32, device=device)
    m, e = lucid.frexp(x)
    assert [_bits(v) for v in m.to("cpu").tolist()] == [0xFFC00000, 0x7FC00000]
    assert e.to("cpu").tolist() == [0, 0]


# ── Compiled against eager (Metal) ─────────────────────────────────────────


def _is_subnormal(v: float) -> bool:
    return v != 0.0 and abs(v) < 1.1754943508222875e-38


def _split_cells(ref: ModuleType, subnormal: bool) -> tuple[list[float], list[int], list[float]]:
    """The float32 ``ldexp`` grid's cells and the ``frexp`` operands that do
    (``subnormal``) or do not read or write a subnormal."""
    xs, es = _grid(_F32_X, _EXPONENTS)
    want, _, _ = _ref_ldexp(ref, xs, es, "float32")
    cell_x: list[float] = []
    cell_e: list[int] = []
    for x, e, w in zip(xs, es, want, strict=True):
        if (_is_subnormal(_f32(x)) or _is_subnormal(w)) == subnormal:
            cell_x.append(x)
            cell_e.append(e)
    frexp_x = [x for x in _F32_X if _is_subnormal(_f32(x)) == subnormal]
    return cell_x, cell_e, frexp_x


def _compiled(fn: Callable[..., object]) -> Callable[..., object]:
    _C_engine.compile.session_cache_clear()
    return lucid.compile(fn)


def _assert_compiled_matches_eager(xs: list[float], es: list[int], frexp_xs: list[float]) -> None:
    x = lucid.tensor(xs, dtype=lucid.float32, device="metal")
    e = lucid.tensor(es, dtype=lucid.int32, device="metal")
    compiled = _compiled(lambda a, b: lucid.ldexp(a, b))(x, e)
    assert isinstance(compiled, lucid.Tensor)
    assert _C_engine.compile.session_cache_size() >= 1
    got = [_bits(v) for v in compiled.to("cpu").tolist()]
    want = [_bits(v) for v in lucid.ldexp(x, e).to("cpu").tolist()]
    bad = [(xs[i], es[i]) for i, (g, w) in enumerate(zip(got, want, strict=True)) if g != w]
    assert not bad, f"compiled ldexp differs from eager at (x, e) = {bad[:8]}"

    fx = lucid.tensor(frexp_xs, dtype=lucid.float32, device="metal")
    out = _compiled(lambda t: lucid.frexp(t))(fx)
    assert isinstance(out, tuple)
    cm, ce = out
    em, ee = lucid.frexp(fx)
    got_m = [_bits(v) for v in cm.to("cpu").tolist()]
    assert got_m == [_bits(v) for v in em.to("cpu").tolist()], "mantissa"
    assert ce.to("cpu").tolist() == ee.to("cpu").tolist(), "exponent"


_needs_metal = pytest.mark.skipif(not metal_available(), reason="needs Metal")


@_needs_metal
def test_compiled_ldexp_reaches_the_smallest_normal() -> None:
    x = lucid.tensor([1.0, 1.0, 0.75, -1.5], device="metal")
    e = lucid.tensor([-126, -125, -125, -126], device="metal")
    got = _compiled(lambda a, b: lucid.ldexp(a, b))(x, e)
    assert isinstance(got, lucid.Tensor)
    assert got.to("cpu").tolist() == [2.0**-126, 2.0**-125, 0.75 * 2.0**-125, -1.5 * 2.0**-126]


@_needs_metal
def test_compiled_ldexp_and_frexp_match_eager_bits_on_normal_values(ref: ModuleType) -> None:
    _assert_compiled_matches_eager(*_split_cells(ref, subnormal=False))


@_needs_metal
def test_compiled_ldexp_and_frexp_match_eager_bits_on_subnormals(ref: ModuleType) -> None:
    _assert_compiled_matches_eager(*_split_cells(ref, subnormal=True))
