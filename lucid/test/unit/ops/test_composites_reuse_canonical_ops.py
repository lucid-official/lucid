"""LCD-1035: composites that spelled a canonical op out by hand call it.

* ``logcumsumexp``'s scan step was ``m + log(exp(a - m) + exp(b - m))``: the sum rounds to 1 when
  ``b`` is more than ~16 below ``a`` (float32), so ``logcumsumexp([0, -20])`` was ``[0, 0]`` where it
  is ``[0, 2.06e-9]``.  The step is ``lucid.logaddexp``, whose tail is range-safe since LCD-1029.
  **``test_logcumsumexp_keeps_a_tail_below_one_ulp`` needs the LCD-1029 engine**: without it
  ``logaddexp`` itself is 0 there in float32 (and 5e-8 off in float64), and nothing the composite
  could do changes that.  Every other test here holds on either engine.
* ``sinc`` carried its own copy of the Maclaurin series that ``lucid.special.spherical_bessel_j0``
  (LCD-1030) has: ``sinc(x)`` is that function at ``pi x``, one body (``_sin_over``).

The oracles are plain float64 closed forms (``math``) at the values read back from the tensors.
"""

import math

import pytest

import lucid
import lucid.special as S
from lucid.test._fixtures.devices import device_dtype_params, devices_supporting

_FLOATS = device_dtype_params([lucid.float32, lucid.float64])
_REL = {lucid.float32: 2e-6, lucid.float64: 1e-12}
INF = float("inf")
NAN = float("nan")


# ── logcumsumexp ────────────────────────────────────────────────────────────


def _prefix_logsumexp(values: list[float]) -> list[float]:
    """``log(sum(exp(v)))`` over each prefix, the largest term factored out and the rest through ``log1p``."""
    out: list[float] = []
    for n in range(1, len(values) + 1):
        prefix = values[:n]
        if any(math.isnan(v) for v in prefix):
            out.append(NAN)
            continue
        top = max(prefix)
        if top == INF:
            out.append(INF)
        elif top == -INF:
            out.append(-INF)
        else:
            rest = list(prefix)
            rest.remove(top)
            out.append(top + math.log1p(math.fsum(math.exp(v - top) for v in rest)))
    return out


def _same(got: float, want: float, rel: float) -> bool:
    if math.isnan(want):
        return math.isnan(got)
    if math.isinf(want):
        return got == want
    return math.isclose(got, want, rel_tol=rel, abs_tol=1e-30)


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
def test_logcumsumexp_keeps_a_tail_below_one_ulp(device: str, dtype: lucid.dtype) -> None:
    """``[0, -20]`` is ``[0, log1p(e^-20)]``; every later term of a long tail adds to it."""
    for row in ([0.0, -20.0], [0.0, -20.0, -25.0, -30.0, -40.0], [-3.0, -30.0, -3.0]):
        x = lucid.tensor(row, dtype=dtype, device=device)
        got = lucid.logcumsumexp(x, 0).tolist()
        want = _prefix_logsumexp(x.tolist())
        for g, w in zip(got, want, strict=True):
            assert _same(g, w, 20 * _REL[dtype]), (row, got, want)


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
@pytest.mark.parametrize(
    "row",
    [
        [0.0, -2.0, 1.0, 2.0, -3.0, 0.5, 7.0],
        [-INF, -INF, 0.0, -INF, 1.0],
        [0.0, INF, 1.0, INF],
        [INF, -INF, 1.0],
        [0.0, NAN, 1.0],
        [1000.0, 1000.0, -1000.0, 999.0],
        [-1000.0, -1000.0, -1001.0],
        [3.0],
        [0.1 * i - 2.0 for i in range(37)],
    ],
    ids=[
        "plain",
        "neg_inf_head",
        "pos_inf",
        "mixed_inf",
        "nan",
        "large",
        "very_negative",
        "single",
        "long",
    ],
)
def test_logcumsumexp_agrees_with_the_closed_form(
    device: str, dtype: lucid.dtype, row: list[float]
) -> None:
    x = lucid.tensor(row, dtype=dtype, device=device)
    got = lucid.logcumsumexp(x, 0).tolist()
    want = _prefix_logsumexp(x.tolist())
    for g, w in zip(got, want, strict=True):
        assert _same(g, w, 20 * _REL[dtype]), (row, got, want)


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
def test_logcumsumexp_gradient_is_the_softmax_of_each_prefix(
    device: str, dtype: lucid.dtype
) -> None:
    """``d y_i / d x_j = exp(x_j - y_i)`` for ``j <= i``."""
    row = [0.0, -20.0, 1.0, -25.0, 0.5]
    w = [1.0, 1.1, 1.2, 1.3, 1.4]
    x = lucid.tensor(row, dtype=dtype, device=device, requires_grad=True)
    y = lucid.logcumsumexp(x, 0)
    (grad,) = lucid.autograd.grad((y * lucid.tensor(w, dtype=dtype, device=device)).sum(), x)
    xs, ys, ws = x.tolist(), _prefix_logsumexp(x.tolist()), lucid.tensor(w, dtype=dtype).tolist()
    for j, g in enumerate(grad.tolist()):
        want = sum(ws[i] * math.exp(xs[j] - ys[i]) for i in range(j, len(xs)))
        assert math.isclose(g, want, rel_tol=20 * _REL[dtype], abs_tol=1e-9), (j, g, want)


@pytest.mark.parametrize("device", devices_supporting(lucid.float32))
@pytest.mark.parametrize("dtype", [lucid.float16, lucid.bfloat16])
def test_logcumsumexp_half_dtypes_scan_in_float32(device: str, dtype: lucid.dtype) -> None:
    """The scan runs in float32 and rounds once at the end: the result keeps the input's dtype."""
    x = lucid.tensor([0.5, 0.25, -1.0, 2.0], dtype=dtype, device=device)
    got = lucid.logcumsumexp(x, 0)
    assert got.dtype == dtype
    for g, w in zip(got.tolist(), _prefix_logsumexp(x.tolist()), strict=True):
        assert math.isclose(g, w, rel_tol=1e-2), (g, w)


# ── sinc ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
def test_sinc_is_spherical_bessel_j0_at_pi_x(device: str, dtype: lucid.dtype) -> None:
    """One body: the same series and the same ``sin(u) / u``, so equal to the bit at every finite ``x``.

    Around ``|pi x| = 1`` (``x = 0.3183``) the value changes from the series to the quotient.
    """
    xs = [
        0.0,
        -0.0,
        1e-30,
        1e-8,
        1e-3,
        0.1,
        0.3183,
        0.31831,
        0.3184,
        0.5,
        1.0,
        -1.5,
        2.0,
        7.3,
        -40.25,
    ]
    x = lucid.tensor(xs, dtype=dtype, device=device)
    a = lucid.sinc(x).tolist()
    b = S.spherical_bessel_j0(x * math.pi).tolist()
    assert a == b


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
def test_sinc_matches_the_closed_form(device: str, dtype: lucid.dtype) -> None:
    """``sin(pi x) / (pi x)`` in float64, 1e-8 to 40 and across the seam at ``pi x = 1`` (``x = 0.3183``).

    The test above pins that ``sinc`` and ``spherical_bessel_j0`` are one body; this one is what the
    body is: both would pass it together if the series were wrong.
    """
    mags = [1e-8 * (40.0 / 1e-8) ** (i / 79) for i in range(80)]
    mags += [0.31, 0.3183, 0.31830988, 0.31831, 0.3184, 0.32, 1.0, 2.0, 3.0]
    xs = [s * m for m in mags for s in (1.0, -1.0)]
    x = lucid.tensor(xs, dtype=dtype, device=device)
    rel, atol = (2e-6, 2e-7) if dtype == lucid.float32 else (2e-13, 2e-15)
    for xv, g in zip(x.tolist(), lucid.sinc(x).tolist(), strict=True):
        u = math.pi * xv
        want = math.sin(u) / u
        assert abs(g - want) <= rel * abs(want) + atol, (xv, g, want)


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
def test_sinc_special_values(device: str, dtype: lucid.dtype) -> None:
    x = lucid.tensor([0.0, -0.0, 1e-30, 1.0, -2.0, 3.0, INF, -INF, NAN], dtype=dtype, device=device)
    got = lucid.sinc(x).tolist()
    assert got[:3] == [1.0, 1.0, 1.0]
    for g in got[3:6]:  # the zeros at the nonzero integers, to the rounding of ``pi x``
        assert abs(g) < 4 * _REL[dtype]
    assert all(math.isnan(g) for g in got[6:])  # sin(pi * inf) / (pi * inf): NaN, as the reference


@pytest.mark.parametrize("device", devices_supporting(lucid.float32))
def test_sinc_dtype_contract(device: str) -> None:
    for dtype in (lucid.float16, lucid.bfloat16, lucid.float32):
        assert lucid.sinc(lucid.tensor([0.25, 0.5], dtype=dtype, device=device)).dtype == dtype
    ints = lucid.tensor([0, 1, -2], dtype=lucid.int32, device=device)
    out = lucid.sinc(ints)
    assert out.dtype == lucid.float32  # an integer operand is the default float
    assert out.tolist()[0] == 1.0


@pytest.mark.parametrize("device", devices_supporting(lucid.float64))
def test_sinc_float64_keeps_its_derivatives_at_zero(device: str) -> None:
    """``sinc'(0) = 0``, ``sinc''(0) = -pi^2 / 3``, ``sinc'''(0) = 0``: a polynomial through the origin."""
    x = lucid.tensor(
        [0.0, -0.0, 1e-30, 1e-9], dtype=lucid.float64, device=device, requires_grad=True
    )
    d1 = lucid.autograd.grad(lucid.sinc(x).sum(), x, create_graph=True)[0]
    d2 = lucid.autograd.grad(d1.sum(), x, create_graph=True)[0]
    d3 = lucid.autograd.grad(d2.sum(), x)[0]
    for g in d1.tolist():
        assert abs(g) <= 1e-8 * math.pi**2 / 3
    for g in d2.tolist():
        assert math.isclose(g, -(math.pi**2) / 3, rel_tol=1e-12), g
    for g in d3.tolist():
        assert abs(g) <= 1e-8 * math.pi**4 / 5
