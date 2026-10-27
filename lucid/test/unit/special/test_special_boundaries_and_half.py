"""Domain boundary and half-precision tests for ``lucid.special`` and ``lucid.i0``
(LCD-101, LCD-102).

Covers:
1. ``ndtri`` at ``p = 0``, ``p = 1``, ``p < 0``, ``p > 1``, ``±inf``, ``NaN``,
   and interior probabilities across ``float16``, ``bfloat16``, ``float32``,
   and ``float64`` on CPU and Metal, plus gradient finiteness in ``(0, 1)``.
2. ``modified_bessel_k0``, ``modified_bessel_k1``, ``scaled_modified_bessel_k0``,
   ``scaled_modified_bessel_k1``, ``bessel_y0``, and ``bessel_y1`` at ``x = 0``,
   ``x < 0``, ``±inf``, ``NaN``, and positive arguments across all floating
   dtypes (including ``float16`` and ``bfloat16``) on CPU and Metal.
3. Half-precision ``lucid.i0``, ``modified_bessel_i0``, ``i1``,
   ``modified_bessel_i1``, ``i0e``, and ``i1e`` in the band ``|x| in [11, 13]``
   where ``exp(|x|)`` overflows ``float16`` while ``I_0(x)`` and ``I_1(x)``
   remain representable, plus ``lucid.signal.windows.kaiser`` in ``float16``
   and ``bfloat16`` on CPU and Metal.
"""

from collections.abc import Callable
import math
from typing import Any

import pytest

import lucid
import lucid.special as S
from lucid.test._fixtures.devices import metal_available

UnaryFn = Callable[[lucid.Tensor], lucid.Tensor]

_DEVICES: list[Any] = [
    "cpu",
    pytest.param("metal", marks=pytest.mark.skipif(not metal_available(), reason="needs Metal")),
]

_ALL_DTYPES: list[str] = ["float16", "bfloat16", "float32", "float64"]

_NDTRI_POINTS: list[float] = [
    -math.inf,
    -0.5,
    -1e-6,
    -0.0,
    0.0,
    1e-6,
    0.02,
    0.02425,
    0.25,
    0.5,
    0.75,
    0.97575,
    0.98,
    1.0 - 1e-6,
    1.0,
    1.0 + 1e-6,
    1.5,
    math.inf,
    math.nan,
]

_HALF_LINE_BESSEL: dict[str, tuple[UnaryFn, str, float]] = {
    "modified_bessel_k0": (S.modified_bessel_k0, "modified_bessel_k0", math.inf),
    "modified_bessel_k1": (S.modified_bessel_k1, "modified_bessel_k1", math.inf),
    "scaled_modified_bessel_k0": (
        S.scaled_modified_bessel_k0,
        "scaled_modified_bessel_k0",
        math.inf,
    ),
    "scaled_modified_bessel_k1": (
        S.scaled_modified_bessel_k1,
        "scaled_modified_bessel_k1",
        math.inf,
    ),
    "bessel_y0": (S.bessel_y0, "bessel_y0", -math.inf),
    "bessel_y1": (S.bessel_y1, "bessel_y1", -math.inf),
}

_BESSEL_BOUNDARY_POINTS: list[float] = [
    -math.inf,
    -10.0,
    -2.0,
    -1e-4,
    -0.0,
    0.0,
    0.25,
    1.0,
    2.0,
    5.0,
    8.0,
    12.0,
    math.inf,
    math.nan,
]


def _rtol_atol(dtype: str) -> tuple[float, float]:
    if dtype == "float16":
        return (2e-3, 1e-3)
    if dtype == "bfloat16":
        return (1.6e-2, 1e-2)
    if dtype == "float32":
        return (1e-5, 1e-6)
    return (1e-7, 1e-9)


def _assert_matches(ours: list[float], theirs: list[float], dtype: str) -> None:
    rtol, atol = _rtol_atol(dtype)
    for idx, (o, t) in enumerate(zip(ours, theirs, strict=True)):
        if math.isnan(t):
            assert math.isnan(o), f"index {idx}: expected NaN, got {o}"
        elif math.isinf(t):
            assert o == t, f"index {idx}: expected {t}, got {o}"
        else:
            assert math.isclose(
                o, t, rel_tol=rtol, abs_tol=atol
            ), f"index {idx} ({dtype}): {o} != {t}"


@pytest.mark.parity
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", _ALL_DTYPES)
def test_ndtri_boundaries_and_interior_match_reference(device: str, dtype: str, ref: Any) -> None:
    if device == "metal" and dtype == "float64":
        pytest.skip("Metal does not support float64")
    l_dt = getattr(lucid, dtype)
    r_dt = getattr(ref, dtype)
    x = lucid.tensor(_NDTRI_POINTS, dtype=l_dt, device=device)
    out = S.ndtri(x)
    assert out.dtype == l_dt
    rx = ref.tensor(_NDTRI_POINTS, dtype=r_dt)
    ref_in = rx.float() if dtype in ("float16", "bfloat16") else rx
    ref_out = ref.special.ndtri(ref_in).to(r_dt)
    _assert_matches(out.to("cpu").to(lucid.float64).tolist(), ref_out.double().tolist(), dtype)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", _ALL_DTYPES)
def test_ndtri_gradients_clean_across_boundaries(device: str, dtype: str) -> None:
    if device == "metal" and dtype == "float64":
        pytest.skip("Metal does not support float64")
    l_dt = getattr(lucid, dtype)
    pts = [-0.5, 0.0, 0.02, 0.5, 0.98, 1.0, 1.5, math.nan]
    x = lucid.tensor(pts, dtype=l_dt, device=device, requires_grad=True)
    y = S.ndtri(x)
    y.sum().backward()
    assert x.grad is not None
    grads = x.grad.to("cpu").to(lucid.float64).tolist()
    # Boundary and out-of-domain finite points do not leak NaN into interior grads.
    assert grads[0] == 0.0
    assert grads[1] == 0.0
    assert math.isfinite(grads[2]) and grads[2] > 0.0
    assert math.isfinite(grads[3]) and grads[3] > 0.0
    assert math.isfinite(grads[4]) and grads[4] > 0.0
    assert grads[5] == 0.0
    assert grads[6] == 0.0
    assert math.isnan(grads[7])


def test_ndtri_deep_tails_float64_and_float32() -> None:
    p64 = lucid.tensor([1e-300, 1e-100, 1e-30, 1.0 - 1e-15], dtype=lucid.float64)
    y64 = S.ndtri(p64).tolist()
    assert all(math.isfinite(v) for v in y64)
    assert y64[0] < y64[1] < y64[2] < 0.0 < y64[3]

    p32 = lucid.tensor([1e-38, 1e-20, 1e-7, 1.0 - 1e-7], dtype=lucid.float32)
    y32 = S.ndtri(p32).tolist()
    assert all(math.isfinite(v) for v in y32)
    assert y32[0] < y32[1] < y32[2] < 0.0 < y32[3]


@pytest.mark.parity
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", _ALL_DTYPES)
@pytest.mark.parametrize("name", list(_HALF_LINE_BESSEL))
def test_half_line_bessel_boundaries_match_reference(
    name: str, device: str, dtype: str, ref: Any
) -> None:
    if device == "metal" and dtype == "float64":
        pytest.skip("Metal does not support float64")
    fn, ref_attr, pole = _HALF_LINE_BESSEL[name]
    l_dt = getattr(lucid, dtype)
    r_dt = getattr(ref, dtype)
    x = lucid.tensor(_BESSEL_BOUNDARY_POINTS, dtype=l_dt, device=device)
    out = fn(x)
    assert out.dtype == l_dt
    got = out.to("cpu").to(lucid.float64).tolist()
    # x < 0 (-inf, -10, -2, -1e-4) -> NaN; x == ±0 -> pole (±inf).
    assert all(math.isnan(v) for v in got[:4])
    assert got[4] == pole and got[5] == pole
    assert math.isnan(got[-1])

    rx = ref.tensor(_BESSEL_BOUNDARY_POINTS, dtype=r_dt)
    ref_in = rx.float() if dtype in ("float16", "bfloat16") else rx
    want = getattr(ref.special, ref_attr)(ref_in).to(r_dt).double().tolist()
    _assert_matches(got, want, dtype)


@pytest.mark.parity
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_bessel_i0_i1_half_precision_no_premature_overflow(
    device: str, dtype: str, ref: Any
) -> None:
    l_dt = getattr(lucid, dtype)
    r_dt = getattr(ref, dtype)
    pts = [-13.0, -12.0, -11.5, -11.0, -3.75, 0.0, 3.75, 11.0, 11.5, 12.0, 13.0]
    for fn, ref_fn in (
        (lucid.i0, ref.i0),
        (S.modified_bessel_i0, ref.special.modified_bessel_i0),
        (S.i1, ref.special.i1),
        (S.modified_bessel_i1, ref.special.modified_bessel_i1),
        (S.i0e, ref.special.i0e),
        (S.i1e, ref.special.i1e),
        (S.erfcx, ref.special.erfcx),
    ):
        pts_use = [v for v in pts if not (fn is S.erfcx and v < -3.0)]
        x = lucid.tensor(pts_use, dtype=l_dt, device=device, requires_grad=True)
        out = fn(x)
        assert out.dtype == l_dt
        got = out.to("cpu").to(lucid.float64).tolist()
        assert all(math.isfinite(v) for v in got), f"{fn.__name__} overflowed: {got}"
        out.sum().backward()
        assert x.grad is not None
        grads = x.grad.to("cpu").to(lucid.float64).tolist()
        assert all(math.isfinite(g) for g in grads)

        rx = ref.tensor(pts_use, dtype=r_dt).float()
        want = ref_fn(rx).to(r_dt).double().tolist()
        _assert_matches(got, want, dtype)


@pytest.mark.parity
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("m", [1, 7, 8, 16])
@pytest.mark.parametrize("sym", [True, False])
def test_kaiser_window_half_precision_finite_and_matches_reference(
    device: str, dtype: str, m: int, sym: bool, ref: Any
) -> None:
    l_dt = getattr(lucid, dtype)
    r_dt = getattr(ref, dtype)
    win = lucid.signal.windows.kaiser(m, sym=sym, dtype=l_dt, device=device)
    assert win.dtype == l_dt
    assert win.shape == (m,)
    got = win.to("cpu").to(lucid.float64).tolist()
    assert all(math.isfinite(v) for v in got), f"kaiser({m}, {dtype}) non-finite: {got}"
    if m == 1 or (sym and m % 2 == 1) or (not sym and m % 2 == 0):
        assert max(got) == pytest.approx(1.0, rel=1e-2 if dtype == "float16" else 7e-2)
    ref_win = ref.signal.windows.kaiser(m, sym=sym, dtype=ref.float32).to(r_dt)
    want = ref_win.double().tolist()
    rtol = 1e-2 if dtype == "float16" else 7e-2
    atol = 5e-3 if dtype == "float16" else 3e-2
    for o, t in zip(got, want, strict=True):
        assert math.isclose(o, t, rel_tol=rtol, abs_tol=atol), f"{o} != {t}"


def test_discrete_inputs_promote_to_float32() -> None:
    xi = lucid.tensor([0, 1])
    assert S.ndtri(xi).tolist() == [-math.inf, math.inf]
    assert math.isinf(S.modified_bessel_k0(xi).tolist()[0])
    assert math.isinf(S.bessel_y0(xi).tolist()[0])
    assert math.isclose(S.i1(xi).tolist()[1], 0.5651591, rel_tol=1e-5)
    assert math.isclose(lucid.i0(xi).tolist()[1], 1.266066, rel_tol=1e-5)
