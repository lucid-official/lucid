"""LCD-1050: the range-safe gamma-family densities keep their accuracy at a lower cost.

Two of the pieces that made them cheap are checked here against what they replaced:

* ``_Bd0`` and ``_StirlingSeries`` are ``Function`` s with a hand-written slope.  The slope must be
  the derivative of the composite value they stand for, at every order and in every regime (the
  series lane, the direct lane, and the far lane where ``x / m`` underflows and ``log1p`` of
  ``(x - m) / m`` would be ``-inf``);
* the frame runs every per-category op once over ``(K, *batch)``: the number of engine ops of a
  density or an entropy must not grow with the number of categories ``K``;
* the counts' plain lanes read the positive-domain ``lgamma`` (LCD-1058): a count outside the
  support is still ``-inf``.
"""

import math

import pytest

import lucid
import lucid.distributions as D
from lucid.distributions import _stirling_math as M
from lucid.test._fixtures.devices import device_dtype_params

_FLOATS = device_dtype_params([lucid.float32, lucid.float64])
_EPS = {lucid.float32: 2.0**-23, lucid.float64: 2.0**-52}

# (x, m): inside the series lane, at the switch, in the direct lane, and far (x / m under / overflows)
_BD0_POINTS = (
    (1e6, 1000800.0),
    (50.0, 49.0),
    (10.0, 10.5),
    (3.0, 7.0),
    (7.0, 3.0),
    (0.5, 0.001),
    (0.001, 0.5),
    (10.0, 1e9),
    (1e9, 10.0),
    (30.0, 30.0),
)


def _t(v: float, device: str, dtype: lucid.dtype, grad: bool = True) -> lucid.Tensor:
    return lucid.tensor(v, dtype=dtype, device=device, requires_grad=grad)


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
@pytest.mark.parametrize(("x", "m"), _BD0_POINTS)
def test_bd0_slopes_are_the_derivative_of_its_value(
    x: float, m: float, device: str, dtype: lucid.dtype
) -> None:
    """``d bd0 / dx = log(x/m)`` and ``d bd0 / dm = 1 - x/m``, the Function's against the exact."""
    xt, mt = _t(x, device, dtype), _t(m, device, dtype)
    M._bd0(xt, mt).backward()
    assert xt.grad is not None and mt.grad is not None
    # the oracle reads the represented operands, in double: ``log1p`` keeps the digits near
    # ``x = m``, the difference of logs where ``1 + t`` is itself nearly 0
    xr, mr = float(xt.item()), float(mt.item())
    t = (xr - mr) / mr
    want_x = math.log1p(t) if t > -0.5 else math.log(xr) - math.log(mr)
    want_m = -t
    eps = _EPS[dtype]
    assert math.isfinite(xt.grad.item()) and math.isfinite(mt.grad.item())
    assert abs(xt.grad.item() - want_x) <= 8 * eps * max(abs(want_x), 1e-3)
    assert abs(mt.grad.item() - want_m) <= 8 * eps * max(abs(want_m), 1e-3)


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
@pytest.mark.parametrize(("x", "m"), _BD0_POINTS[:5] + _BD0_POINTS[9:])
def test_bd0_second_derivatives(x: float, m: float, device: str, dtype: lucid.dtype) -> None:
    """The slope is built of ordinary ops, so every higher order follows: ``1/x``, ``-1/m``, ``x/m^2``."""
    xt, mt = _t(x, device, dtype), _t(m, device, dtype)
    gx, gm = lucid.autograd.grad(M._bd0(xt, mt), [xt, mt], create_graph=True)
    d_xx, d_xm = lucid.autograd.grad(gx, [xt, mt], retain_graph=True)
    (d_mm,) = lucid.autograd.grad(gm, [mt])
    eps = _EPS[dtype]
    for got, want in ((d_xx, 1.0 / x), (d_xm, -1.0 / m), (d_mm, x / (m * m))):
        assert abs(got.item() - want) <= 64 * eps * abs(want), (x, m, got.item(), want)


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
@pytest.mark.parametrize("z", [10.0, 11.5, 25.0, 1e3, 1e6, 2.0**60])
def test_stirling_series_slope_is_minus_the_digamma_series(
    z: float, device: str, dtype: lucid.dtype
) -> None:
    """``S' = -D`` term by term: the Function's slope is the other series, to the bit."""
    zt = _t(z, device, dtype)
    M._stirling_series(zt).backward()
    assert zt.grad is not None
    assert zt.grad.item() == -M._digamma_series(_t(z, device, dtype, False)).item()


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
def test_equal_components_share_the_mass_evenly(device: str, dtype: lucid.dtype) -> None:
    """A tie goes to the first component, so ``log(a_i / s)`` is ``log(1/K)`` for every one."""
    for k in (1, 2, 3, 5):
        comps = lucid.full((k, 4), 25.0, dtype=dtype, device=device)
        got = M._log_shares(comps, comps.sum(dim=0))
        for v in got.reshape(-1).tolist():
            assert abs(v - math.log(1.0 / k)) <= 4 * _EPS[dtype] * max(abs(math.log(k)), 1.0)


def _event_count(fn: object) -> int:
    assert callable(fn)
    with lucid.profiler.profile() as prof:
        fn()
    return len(list(prof.events()))


def test_op_count_does_not_grow_with_the_number_of_categories() -> None:
    """Every per-category op runs once over ``(K, *batch)``: ``K`` is data, not dispatches.

    A loop over the components (the layout this replaced) costs a handful of ops per category
    on a device that pays per op; the Dirichlet with 3 categories and with 9 run the same graph.
    """
    counts: dict[str, set[int]] = {"log_prob": set(), "entropy": set()}
    for k in (3, 9):
        alpha = 20.0 + 100.0 * lucid.rand(16, k)
        x = alpha / alpha.sum(dim=-1, keepdim=True)
        counts["log_prob"].add(_event_count(lambda: D.Dirichlet(alpha).log_prob(x)))
        counts["entropy"].add(_event_count(lambda: D.Dirichlet(alpha).entropy()))
    assert all(len(c) == 1 for c in counts.values()), counts


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
def test_a_count_outside_the_support_still_scores_minus_infinity(
    device: str, dtype: lucid.dtype
) -> None:
    """LCD-1058: the counts' plain lanes read the positive-domain ``lgamma``, which has no left half.

    A count below 0, a count above ``total_count`` and a multinomial with a negative count were
    ``-inf`` through the pole of ``lgamma`` at the non-positive integers.  The lanes clamp (or mask)
    before the call, so they are ``-inf`` still, and not the NaN of ``log`` of a negative number.
    """
    inf = float("inf")

    def t(values: float | list[float]) -> lucid.Tensor:
        return lucid.tensor(values, dtype=dtype, device=device)

    poisson = D.Poisson(t(3.0), validate_args=False).log_prob(t([-1.0, -2.0, -7.0]))
    assert poisson.tolist() == [-inf, -inf, -inf]
    for total, counts in ((5.0, [-1.0, -3.0, 6.0, 9.0]), (50.0, [-1.0, 51.0, 60.0])):
        law = D.Binomial(t(total), probs=t(0.3), validate_args=False)
        assert law.log_prob(t(counts)).tolist() == [-inf] * len(counts)
    nb = D.NegativeBinomial(t(2.5), probs=t(0.3), validate_args=False)
    assert nb.log_prob(t([-1.0, -2.0, -6.0])).tolist() == [-inf, -inf, -inf]
    multinomial = D.Multinomial(5, probs=t([0.2, 0.3, 0.5]), validate_args=False)
    assert multinomial.log_prob(t([[-1.0, 2.0, 3.0]])).tolist() == [-inf]
