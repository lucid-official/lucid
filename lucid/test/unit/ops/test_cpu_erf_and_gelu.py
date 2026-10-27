"""CPU float32 erf and GELU: within one ulp of the true value, on a dense grid.

``erf`` on the CPU used libm's scalar ``erff``, which kept the exact GELU's
loop scalar.  It is now a branch-free polynomial evaluated in float32
arithmetic (``backend/cpu/ErfPoly.h``, Linear CHA-9), four lanes to a vector
where the double-precision one before it ran two.  The GPT training parity
rests on an exact erf, so the bound is held here: at most one ulp from
``math.erf`` rounded to float32, on a grid and on every float around the
seams of the kernel (its region split, the point where erf starts rounding
to one, the clamp beyond it, the subnormals).  Over every positive float the
kernel is exact on all but 0.7% and one ulp off on the rest; the share that
is exact is held too, so a change that stays inside one ulp while rounding
the wrong way much more often still shows.

The exact GELU's gradient, Phi(x) + x * phi(x), took its density from
vForce's ``expf`` called between two passes over a tile; the call cost the
erf loop beside it its registers.  The density is now an inline float32
exp (``backend/cpu/GeluGrad.h``, Linear CHA-25) that also recovers the
rounding of x^2 the old route fed to the exp.  Where the CDF has rounded to
zero the gradient is x * phi(x) alone, so the density shows through: a few
ulp there, where the old route was 64 off.

The tanh-approximate GELU took a scalar ``tanhf`` per element on one core,
4 ms for 524k elements.  It now runs through vForce a tile at a time and is
held to the formula evaluated in float64.

GELU is held in absolute terms, scaled by the input: ``1 + erf`` and
``1 + tanh`` cancel for negative inputs, so a tiny output carries the
rounding of a value near one.  That is the formula's own precision, the
same before and after; a relative bound there would test the cancellation,
not the kernel.
"""

import math

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F

GRID = np.concatenate(
    [
        np.linspace(-6.0, 6.0, 400_001, dtype=np.float32),
        np.array([0.0, -0.0, 1.0, -1.0, 4.0, -4.0, 1e-30, -1e-30], np.float32),
    ]
)


def _assert_close_to_scale(got: np.ndarray, want: np.ndarray, x: np.ndarray, eps: float) -> None:
    bound = eps * np.maximum(1.0, np.abs(x.astype(np.float64)))
    worst = float((np.abs(got.astype(np.float64) - want) / bound).max())
    assert worst <= 1.0, f"off by {worst:.2f}x the bound"


def _ulps(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(
        a.astype(np.float32).view(np.int32).astype(np.int64)
        - b.astype(np.float32).view(np.int32).astype(np.int64)
    )


def _floats_around(value: float, count: int) -> np.ndarray:
    centre = int(np.array(value, np.float32).view(np.int32))
    return np.arange(centre - count, centre + count, dtype=np.int32).view(np.float32)


# Every float near the places the kernel could slip: its split between the two
# polynomials (1), the centre of the upper one, where erf starts rounding to 1
# (3.9192), the clamp beyond it (3.95), where the result crosses 0.5 (and its
# ulp halves), and the subnormals.  Both signs.
_SEAMS_POS = np.concatenate(
    [_floats_around(v, 1 << 14) for v in (1.0, 2.4645581, 3.9192, 3.95, 0.47693628)]
    + [np.arange(1, 1 << 15, dtype=np.int32).view(np.float32)]
)
SEAMS = np.concatenate([_SEAMS_POS, -_SEAMS_POS])


def _erf_reference(x: np.ndarray) -> np.ndarray:
    return np.array([math.erf(float(v)) for v in x]).astype(np.float32)


def test_erf_is_within_one_ulp_of_the_correctly_rounded_value() -> None:
    got = lucid.erf(lucid.tensor(GRID)).numpy()
    assert int(_ulps(got, _erf_reference(GRID)).max()) <= 1


def test_erf_holds_one_ulp_on_every_float_at_its_seams() -> None:
    got = lucid.erf(lucid.tensor(SEAMS)).numpy()
    assert int(_ulps(got, _erf_reference(SEAMS)).max()) <= 1


def test_erf_rounds_correctly_on_nearly_all_of_the_grid() -> None:
    # Measured 98.3% exact on this grid (the double-precision kernel before it:
    # 99.997%).  A fit or an evaluation order that rounds the wrong way twice
    # as often still passes the one-ulp bound, and fails here.
    got = lucid.erf(lucid.tensor(GRID)).numpy()
    exact = float((_ulps(got, _erf_reference(GRID)) == 0).mean())
    assert exact >= 0.97, f"only {exact:.2%} correctly rounded"


def test_erf_is_odd_bit_for_bit() -> None:
    x = np.abs(GRID)
    pos = lucid.erf(lucid.tensor(x)).numpy()
    neg = lucid.erf(lucid.tensor(-x)).numpy()
    np.testing.assert_array_equal(neg.view(np.int32), (-pos).view(np.int32))


@pytest.mark.parametrize("copies", [1, 13])
def test_erf_keeps_the_special_values(copies: int) -> None:
    # One copy runs the scalar tail, thirteen mostly the vector body.
    special = np.array(
        [np.inf, -np.inf, np.nan, -0.0, 0.0, 3.95, -3.95, 1e30, -1e30, 3e38], np.float32
    )
    got = lucid.erf(lucid.tensor(np.tile(special, copies))).numpy().reshape(copies, -1)
    assert (got[:, 0] == 1.0).all() and (got[:, 1] == -1.0).all()
    assert np.isnan(got[:, 2]).all()
    assert (got[:, 3] == 0.0).all() and np.signbit(got[:, 3]).all()
    assert (got[:, 4] == 0.0).all() and not np.signbit(got[:, 4]).any()
    np.testing.assert_array_equal(got[:, 5:], [[1.0, -1.0, 1.0, -1.0, 1.0]] * copies)


def test_exact_gelu_gives_an_element_the_same_bits_in_any_array() -> None:
    # erf runs inline in the GELU loops, split across cores and into vector
    # bodies and scalar tails; an element's answer, and its gradient, cannot
    # depend on which of those it lands in.
    x = 3.0 * np.random.default_rng(5).standard_normal(300_001).astype(np.float32)
    whole = lucid.tensor(x, requires_grad=True)
    out = F.gelu(whole)
    out.sum().backward()
    for lo, hi in ((0, 17), (65_530, 65_545), (299_990, 300_001)):
        part = lucid.tensor(x[lo:hi], requires_grad=True)
        piece = F.gelu(part)
        piece.sum().backward()
        np.testing.assert_array_equal(piece.numpy(), out.numpy()[lo:hi])
        np.testing.assert_array_equal(part.grad.numpy(), whole.grad.numpy()[lo:hi])


def test_exact_gelu_and_its_gradient_follow_erf() -> None:
    x64 = GRID.astype(np.float64)
    cdf = 0.5 * (1.0 + np.array([math.erf(v / math.sqrt(2.0)) for v in x64]))
    got = F.gelu(lucid.tensor(GRID)).numpy()
    _assert_close_to_scale(got, x64 * cdf, GRID, 2.0**-22)
    x = lucid.tensor(GRID, requires_grad=True)
    F.gelu(x).sum().backward()
    pdf = np.exp(-0.5 * x64 * x64) / math.sqrt(2.0 * math.pi)
    _assert_close_to_scale(x.grad.numpy(), cdf + x64 * pdf, GRID, 2.0**-21)


def _gelu_grad(x: np.ndarray, upstream: np.ndarray | None = None) -> np.ndarray:
    t = lucid.tensor(x, requires_grad=True)
    y = F.gelu(t)
    if upstream is None:
        y.sum().backward()
    else:
        y.backward(lucid.tensor(upstream))
    return t.grad.numpy()


def _signed_ulps(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Distance in representable floats, counted through zero.
    def ordered(v: np.ndarray) -> np.ndarray:
        bits = v.astype(np.float32).view(np.int32).astype(np.int64)
        return np.where(bits < 0, -(bits & 0x7FFFFFFF), bits)

    return np.abs(ordered(a) - ordered(b))


def _x_times_density(x: np.ndarray) -> np.ndarray:
    x64 = x.astype(np.float64)
    density = np.exp(-0.5 * x64 * x64) / math.sqrt(2.0 * math.pi)
    return (x64 * density).astype(np.float32)


@pytest.mark.parametrize(
    "lo, hi, bound",
    [
        # erf(x / sqrt 2) has rounded to -1 from x = -5.54, so the CDF is 0
        # and the gradient is x * phi(x) rounded: the density alone.
        (-13.0, -6.0, 3),
        # The density is subnormal from |x| = 13.39; it is rounded once into
        # the subnormals, then multiplied by x.
        (-14.5, -13.0, 8),
    ],
)
def test_exact_gelu_gradient_carries_the_density_to_a_few_ulp(
    lo: float, hi: float, bound: int
) -> None:
    # The vForce route rounded x^2 before the exp, which multiplies that
    # error by x^2 / 2: it was 64 ulp off in the first range, 69 in the
    # second.
    x = np.linspace(lo, hi, 200_001, dtype=np.float32)
    worst = int(_signed_ulps(_gelu_grad(x), _x_times_density(x)).max())
    assert worst <= bound, f"{worst} ulp from x * phi(x)"


def test_exact_gelu_gradient_is_within_one_ulp_for_positive_inputs() -> None:
    # Phi(x) + x phi(x) has no cancellation for x >= 0.  Over every positive
    # float the kernel is at most 2 ulp off (on 5e-5 of [0.5, 1)), and 1 ulp
    # on this grid.
    x = np.linspace(0.0, 6.0, 200_001, dtype=np.float32)
    x64 = x.astype(np.float64)
    cdf = 0.5 * np.array([math.erfc(-v / math.sqrt(2.0)) for v in x64])
    want = (cdf + x64 * np.exp(-0.5 * x64 * x64) / math.sqrt(2.0 * math.pi)).astype(np.float32)
    assert int(_signed_ulps(_gelu_grad(x), want).max()) <= 1


def test_exact_gelu_gradient_scales_by_the_upstream_gradient_in_one_rounding() -> None:
    # The incoming gradient multiplies the finished derivative, once.
    rng = np.random.default_rng(11)
    x = 3.0 * rng.standard_normal(70_001).astype(np.float32)
    g = rng.standard_normal(70_001).astype(np.float32)
    np.testing.assert_array_equal(_gelu_grad(x, g), _gelu_grad(x) * g)


@pytest.mark.parametrize("copies", [1, 13])
def test_exact_gelu_gradient_keeps_the_special_values(copies: int) -> None:
    # One copy runs the scalar tail, thirteen mostly the vector body.  The
    # density is clamped past |x| = 14.5, where it has rounded to zero, so a
    # huge input gives the step it tends to and +-inf give inf * 0 = nan.
    special = np.array(
        [np.inf, -np.inf, np.nan, 0.0, -0.0, 1e20, -1e20, 3e38, -3e38, 14.5, -14.5], np.float32
    )
    got = _gelu_grad(np.tile(special, copies)).reshape(copies, -1)
    assert np.isnan(got[:, :3]).all()
    np.testing.assert_array_equal(got[:, 3:], [[0.5, 0.5, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]] * copies)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_tanh_gelu_matches_its_formula(dtype: type) -> None:
    eps = 2.0**-22 if dtype is np.float32 else 2.0**-50
    x64 = GRID.astype(np.float64)
    c1, c2 = math.sqrt(2.0 / math.pi), 0.044715
    t = np.tanh(c1 * (x64 + c2 * x64**3))
    got = F.gelu(lucid.tensor(GRID.astype(dtype)), approximate="tanh").numpy()
    _assert_close_to_scale(got, 0.5 * x64 * (1.0 + t), GRID, eps)
    x = lucid.tensor(GRID.astype(dtype), requires_grad=True)
    F.gelu(x, approximate="tanh").sum().backward()
    dinner = c1 * (1.0 + 3.0 * c2 * x64 * x64)
    grad = 0.5 * (1.0 + t) + 0.5 * x64 * (1.0 - t * t) * dinner
    # 1 - t^2 cancels near t = +-1, multiplied by x * dinner (up to ~25):
    # an ulp of t becomes several of the gradient.
    _assert_close_to_scale(x.grad.numpy(), grad, GRID, 16 * eps)
