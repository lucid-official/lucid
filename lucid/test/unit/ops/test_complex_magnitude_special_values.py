"""The length of a (re, im) pair at every special value (LCD-744).

``abs`` of a complex tensor and ``hypot`` of two real ones are one rule,
``hypot_lanes``: overflow-safe, exact through the subnormal range, and
``+inf`` whenever a lane is infinite, a NaN beside it included (C99
Annex G).  Both used to break it, differently.  ``maximum`` turned
``(inf, nan)`` into NaN, and a ``max(m, FLT_MIN)`` floor on the divisor
changed the divisor but not the multiplier, so ``hypot(1e-40, 1e-40)``
came back ``1.00004e-40`` instead of ``1.4142e-40`` (``abs(0 + 1e-40j)``
came back ``8.5e-43``).

The matrix is {±inf, nan, ±0, ±subnormal, 1}² against the reference, on
every stream and dtype that can hold it.

Metal flushes subnormal operands in arithmetic (``x * 1.0`` of a
subnormal ``x`` is ``0``), so a length that needs arithmetic on a
subnormal lane cannot be exact there.  It must still be that flushed
``0`` or the reference value, never the garbage the CPU used to give.
"""

import itertools
import math

import pytest

import lucid
from lucid.test._fixtures.devices import device_dtype_params

_INF = float("inf")
_NAN = float("nan")

_LANE_OF = {lucid.complex64: lucid.float32, lucid.complex128: lucid.float64}
# real lane -> (a subnormal of it, smallest normal, relative tolerance)
_RANGE = {
    lucid.float32: (1e-40, 1.1754943508222875e-38, 1e-6),
    lucid.float64: (1e-310, 2.2250738585072014e-308, 1e-14),
}
_COMPLEX = device_dtype_params([lucid.complex64, lucid.complex128])
_REAL = device_dtype_params([lucid.float32, lucid.float64])


def _pairs(real_dt: lucid.dtype) -> tuple[list[float], list[float]]:
    sub = _RANGE[real_dt][0]
    pool = [_INF, -_INF, _NAN, 0.0, -0.0, sub, -sub, 1.0]
    pairs = list(itertools.product(pool, repeat=2))
    return [a for a, _ in pairs], [b for _, b in pairs]


def _agree(got: float, want: float, rtol: float) -> bool:
    if math.isnan(want):
        return math.isnan(got)
    if math.isinf(want) or want == 0.0:
        return got == want
    return abs(got - want) <= rtol * abs(want)


def _flushed(a: float, b: float, normal_min: float) -> bool:
    """Both lanes are zero or subnormal and at least one is not zero."""
    lanes = (abs(a), abs(b))
    return all(v < normal_min for v in lanes) and any(v != 0.0 for v in lanes)


def _assert_lengths(got, want, a, b, *, real_dt, device):
    _, normal_min, rtol = _RANGE[real_dt]
    bad = [
        (x, y, g, w)
        for x, y, g, w in zip(a, b, got, want, strict=True)
        if not _agree(g, w, rtol)
        and not (device == "metal" and g == 0.0 and _flushed(x, y, normal_min))
    ]
    assert not bad, f"(a, b, got, reference) differ on {device} {real_dt}: {bad}"


def _ref_dtype(ref, real_dt):
    return getattr(ref, str(real_dt).removeprefix("lucid."))


@pytest.mark.parametrize(("device", "dtype"), _COMPLEX)
def test_complex_abs_matches_the_reference_at_every_special_pair(ref, device, dtype):
    real_dt = _LANE_OF[dtype]
    re, im = _pairs(real_dt)
    z = lucid.complex(lucid.tensor(re, dtype=real_dt), lucid.tensor(im, dtype=real_dt)).to(device)
    out = lucid.abs(z)
    assert out.dtype == real_dt and out.device.type == device
    rdt = _ref_dtype(ref, real_dt)
    want = ref.abs(ref.complex(ref.tensor(re, dtype=rdt), ref.tensor(im, dtype=rdt)))
    _assert_lengths(out.tolist(), want.tolist(), re, im, real_dt=real_dt, device=device)


@pytest.mark.parametrize(("device", "real_dt"), _REAL)
def test_hypot_matches_the_reference_at_every_special_pair(ref, device, real_dt):
    a, b = _pairs(real_dt)
    out = lucid.hypot(
        lucid.tensor(a, dtype=real_dt).to(device), lucid.tensor(b, dtype=real_dt).to(device)
    )
    rdt = _ref_dtype(ref, real_dt)
    want = ref.hypot(ref.tensor(a, dtype=rdt), ref.tensor(b, dtype=rdt))
    _assert_lengths(out.tolist(), want.tolist(), a, b, real_dt=real_dt, device=device)


@pytest.mark.parametrize(("device", "real_dt"), device_dtype_params([lucid.float32]))
def test_hypot_applies_its_infinity_rule_across_a_broadcast(ref, device, real_dt):
    a = [[3.0], [_INF], [_NAN], [0.0]]
    b = [[4.0, _NAN, _INF, 0.0, -2.0]]
    got = lucid.hypot(lucid.tensor(a).to(device), lucid.tensor(b).to(device))
    want = ref.hypot(ref.tensor(a), ref.tensor(b))
    assert got.shape == tuple(want.shape)
    got_flat = [v for row in got.tolist() for v in row]
    want_flat = want.flatten().tolist()
    assert all(_agree(g, w, 1e-6) for g, w in zip(got_flat, want_flat, strict=True))


@pytest.mark.parametrize(("device", "dtype"), _COMPLEX)
def test_complex_abs_of_a_strided_zero_d_and_empty_input(device, dtype):
    real_dt = _LANE_OF[dtype]
    grid = lucid.complex(
        lucid.tensor([[1.0, _INF, 0.0], [_NAN, 3.0, _INF]], dtype=real_dt),
        lucid.tensor([[_NAN, _NAN, 0.0], [_INF, 4.0, _INF]], dtype=real_dt),
    ).to(device)
    strided = grid.transpose(0, 1)
    assert not strided.is_contiguous()
    got = [v for row in lucid.abs(strided).tolist() for v in row]
    want = [_NAN, _INF, _INF, 5.0, 0.0, _INF]
    assert all(_agree(g, w, 1e-6) for g, w in zip(got, want, strict=True))
    assert lucid.abs(grid[0, 1]).item() == _INF
    assert lucid.abs(lucid.zeros(0, dtype=dtype, device=device)).shape == (0,)


@pytest.mark.parametrize(("device", "dtype"), _COMPLEX)
def test_complex_abs_backward_is_unchanged_where_its_value_moved(ref, device, dtype):
    """The value moved from NaN to inf at (inf, nan); the derivative is still
    what the reference gives there, and the finite points stay exact."""
    real_dt = _LANE_OF[dtype]
    re = [_INF, _NAN, _INF, _NAN, 0.0, 3.0, -3.0]
    im = [_NAN, _INF, _INF, _NAN, 0.0, 4.0, 4.0]
    z = lucid.complex(lucid.tensor(re, dtype=real_dt), lucid.tensor(im, dtype=real_dt)).to(device)
    z.requires_grad_(True)
    lucid.abs(z).sum().backward()
    rdt = _ref_dtype(ref, real_dt)
    zr = ref.complex(ref.tensor(re, dtype=rdt), ref.tensor(im, dtype=rdt)).requires_grad_(True)
    ref.abs(zr).sum().backward()
    assert z.grad is not None and zr.grad is not None
    rtol = _RANGE[real_dt][2]
    for a, b, g, w in zip(re, im, z.grad.tolist(), zr.grad.tolist(), strict=True):
        assert _agree(g.real, w.real, rtol) and _agree(g.imag, w.imag, rtol), (a, b, g, w)
