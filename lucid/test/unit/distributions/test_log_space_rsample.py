"""``rsample`` of ``Beta`` and ``Dirichlet`` keeps its ordering at tiny concentrations (LCD-707).

``Dirichlet([.01] * 3).rsample((20000,))`` put 7.7 % of its rows within 1e-3 of
``[1/3, 1/3, 1/3]``.  ``rsample`` normalised float32 gamma draws that were held
at the smallest normal float; below a concentration of about 0.03 a row whose
three components all fall under that floor reads as three equal numbers, and
the normalised row is uniform whichever component was larger.  ``sample`` had
been exact since LCD-677 because it normalises the *logarithms*; ``rsample``
now does too, through a ``Function`` over ``log G`` whose backward is the
implicit-reparameterisation derivative of the logarithm.

Guards, per concentration:

* the draws follow the reference (two-sample Kolmogorov-Smirnov) and so does
  the share of near-uniform rows -- at 1e-2 and 3e-2 only: below about 3e-3 the
  reference's own ``rsample`` and ``sample`` lose the ordering the same way
  (12 % of ``Dirichlet([1e-3] * 3)`` rows are uniform there), so the closed-form
  mass of the lower tail is the oracle at 1e-3 and 3e-3;
* the gradient of ``log G`` is the implicit formula -- against the closed-form
  small-sample limit where ``G`` has underflowed, against the reference where
  it has not, and in expectation against ``trigamma``;
* the gradient stays finite where ``G`` underflows.
"""

import math
from types import ModuleType

import pytest

import lucid
import lucid.distributions as D
from lucid.distributions.gamma import (
    _standard_gamma_grad,
    _standard_log_gamma,
    _standard_log_gamma_grad,
)
from lucid.test.unit.distributions.test_gamma_family_small_concentration import _ks

_CONCENTRATIONS = [1e-3, 3e-3, 1e-2, 3e-2]
_N = 20000
_KS_LIMIT = 1.95 * math.sqrt(2.0 / _N)
_NEAR_UNIFORM = 1e-3


def _t(lib: ModuleType, device: str, *values: float):
    return lib.tensor(list(values), dtype=lib.float32, device=device)


def _flat(x) -> list[float]:
    return x.detach().cpu().reshape(-1).tolist()


def _near_uniform_share(x) -> float:
    """Share of rows of a 3-simplex sample within ``1e-3`` of ``[1/3] * 3``."""
    rows = x.detach().cpu().reshape(-1, 3).tolist()
    return sum(all(abs(v - 1.0 / 3.0) < _NEAR_UNIFORM for v in row) for row in rows) / len(rows)


# ── the reported defect ─────────────────────────────────────────────────────


@pytest.mark.parametrize("c", _CONCENTRATIONS)
def test_dirichlet_rsample_has_no_phantom_uniform_rows(c: float, device: str) -> None:
    # A Dirichlet(c, c, c) row is within 1e-3 of uniform with probability well
    # under 1 % at every c here; the defect put 7.7 % of rows there at c = .01.
    lucid.manual_seed(0)
    x = D.Dirichlet(_t(lucid, device, c, c, c)).rsample((_N,))
    assert _near_uniform_share(x) < 0.005, c


@pytest.mark.parametrize("c", _CONCENTRATIONS)
def test_beta_rsample_has_no_phantom_midpoint(c: float, device: str) -> None:
    # The two-component case of the same defect: sigmoid(0) = .5 for every
    # draw whose two gammas had both been held at the floor.
    lucid.manual_seed(0)
    x = D.Beta(_t(lucid, device, c), _t(lucid, device, c)).rsample((_N,))
    share = sum(abs(v - 0.5) < _NEAR_UNIFORM for v in _flat(x)) / _N
    assert share < 0.005, c


@pytest.mark.parametrize("c", _CONCENTRATIONS)
def test_rsample_and_sample_agree_in_distribution(c: float, device: str) -> None:
    lucid.manual_seed(1)
    dist = D.Dirichlet(_t(lucid, device, c, c, c))
    r, s = dist.rsample((_N,)), dist.sample((_N,))
    for k in range(3):
        assert _ks(_flat(r[:, k]), _flat(s[:, k])) < _KS_LIMIT, (c, k)


# ── against the reference ───────────────────────────────────────────────────


def _lower_tail_mass(a: float, b: float, t: float) -> float:
    """``P(X < t)`` for ``X ~ Beta(a, b)`` at small ``t``: ``t^a / (a B(a, b))`` to first order."""
    log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    return math.exp(a * math.log(t) - math.log(a) - log_beta)


@pytest.mark.parametrize("c", _CONCENTRATIONS)
def test_rsample_lower_tail_mass_is_the_closed_form(c: float, device: str) -> None:
    # Mass below 1e-30 is mass the gammas cannot hold: it is where a
    # normalisation of clamped draws goes wrong.  Beta(c, c) and the marginal
    # of Dirichlet(c, c, c), Beta(c, 2c), have it in closed form.
    t = 1e-30
    lucid.manual_seed(2)
    beta = _flat(D.Beta(_t(lucid, device, c), _t(lucid, device, c)).rsample((_N,)))
    dirichlet = _flat(D.Dirichlet(_t(lucid, device, c, c, c)).rsample((_N,))[:, 0])
    for draws, (a, b) in ((beta, (c, c)), (dirichlet, (c, 2 * c))):
        p = _lower_tail_mass(a, b, t)
        got = sum(v < t for v in draws) / _N
        assert abs(got - p) < 5.0 * math.sqrt(p * (1.0 - p) / _N) + 1e-3, (c, a, b)


@pytest.mark.parity
@pytest.mark.parametrize("c", [1e-2, 3e-2])
def test_dirichlet_rsample_follows_the_reference(c: float, ref: ModuleType, device: str) -> None:
    lucid.manual_seed(7)
    ref.manual_seed(7)
    mine = D.Dirichlet(_t(lucid, device, c, c, c)).rsample((_N,))
    theirs = ref.distributions.Dirichlet(_t(ref, "cpu", c, c, c)).rsample((_N,))
    for k in range(3):
        assert _ks(_flat(mine[:, k]), _flat(theirs[:, k])) < _KS_LIMIT, (c, k)
    assert abs(_near_uniform_share(mine) - _near_uniform_share(theirs)) < 0.005, c


@pytest.mark.parity
@pytest.mark.parametrize("c", [1e-2, 3e-2])
def test_beta_rsample_follows_the_reference(c: float, ref: ModuleType, device: str) -> None:
    lucid.manual_seed(7)
    ref.manual_seed(7)
    mine = D.Beta(_t(lucid, device, c), _t(lucid, device, c)).rsample((_N,))
    theirs = ref.distributions.Beta(_t(ref, "cpu", c), _t(ref, "cpu", c)).rsample((_N,))
    assert _ks(_flat(mine), _flat(theirs)) < _KS_LIMIT, c


# ── the gradient of log G is the implicit formula ───────────────────────────


@pytest.mark.parametrize("alpha", [1e-3, 3e-3, 1e-2, 3e-2])
@pytest.mark.parametrize("log_z", [-40.0, -120.0, -1000.0, -30000.0])
def test_log_gamma_grad_is_the_small_sample_limit_where_the_gamma_underflowed(
    alpha: float, log_z: float, device: str
) -> None:
    # With P(a, z) ~ z^a / Gamma(a + 1) for small z, holding P fixed gives
    #   d log z / d a = (psi(a + 1) - log z) / a.
    # log z far below -87 is a gamma that float32 rounds to 0, where the
    # gradient of the gamma itself is 0 and its quotient with the gamma is 0 / 0.
    a = _t(lucid, device, alpha)
    got = _flat(_standard_log_gamma_grad(a, _t(lucid, device, log_z)))[0]
    want = (float(lucid.digamma(_t(lucid, "cpu", alpha + 1.0)).item()) - log_z) / alpha
    assert math.isfinite(got)
    assert got == pytest.approx(want, rel=2e-3)


@pytest.mark.parametrize("alpha", [0.5, 1.0, 3.0, 20.0])
@pytest.mark.parametrize("z", [0.01, 0.5, 1.0, 4.0, 30.0])
def test_log_gamma_grad_is_the_gamma_grad_over_the_gamma(
    alpha: float, z: float, device: str
) -> None:
    a, x = _t(lucid, device, alpha), _t(lucid, device, z)
    got = _flat(_standard_log_gamma_grad(a, x.log()))[0]
    want = _flat(_standard_gamma_grad(a, x))[0] / z
    assert got == pytest.approx(want, rel=2e-4, abs=1e-6)


@pytest.mark.parity
@pytest.mark.parametrize("alpha", [0.05, 0.5, 2.0, 12.0])
@pytest.mark.parametrize("z", [0.02, 0.5, 3.0, 25.0])
def test_log_gamma_grad_matches_the_reference(
    alpha: float, z: float, ref: ModuleType, device: str
) -> None:
    got = _flat(_standard_log_gamma_grad(_t(lucid, device, alpha), _t(lucid, device, z).log()))[0]
    want = float(ref._standard_gamma_grad(_t(ref, "cpu", alpha), _t(ref, "cpu", z)).item()) / z
    assert got == pytest.approx(want, rel=2e-3, abs=1e-6)


@pytest.mark.parametrize("alpha", _CONCENTRATIONS + [0.5, 4.0])
def test_log_gamma_backward_is_finite_and_has_the_expected_mean(alpha: float, device: str) -> None:
    # E[log G] = digamma(alpha), so the sample mean of d log G / d alpha
    # estimates trigamma(alpha) ~ 1 / alpha^2 + pi^2 / 6 for small alpha.
    a = _t(lucid, device, alpha).requires_grad_(True)
    lucid.manual_seed(3)
    _standard_log_gamma(a.expand(_N)).sum().backward()
    assert a.grad is not None
    total = _flat(a.grad)[0]
    assert math.isfinite(total)
    trigamma = 1.0 / alpha**2 + math.pi**2 / 6.0 if alpha < 0.1 else None
    if trigamma is not None:
        assert total / _N == pytest.approx(trigamma, rel=0.05), alpha


@pytest.mark.parametrize("c", _CONCENTRATIONS)
def test_rsample_gradients_are_finite_where_every_gamma_underflowed(c: float, device: str) -> None:
    a = _t(lucid, device, c, c, c).requires_grad_(True)
    b = _t(lucid, device, c).requires_grad_(True)
    lucid.manual_seed(5)
    (D.Dirichlet(a).rsample((2000,))[:, 0] ** 2).sum().backward()
    D.Beta(b, b).rsample((2000,)).sum().backward()
    assert a.grad is not None and b.grad is not None
    assert all(math.isfinite(v) for v in _flat(a.grad) + _flat(b.grad)), c
