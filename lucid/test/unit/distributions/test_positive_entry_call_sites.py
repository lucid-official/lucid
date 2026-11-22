"""LCD-1058: every call site of the positive-domain ``lgamma`` / ``digamma`` computes its quantity.

The gamma family reads ``_lgamma_positive`` / ``_digamma_positive`` wherever the law's constraint
keeps the argument above 0.  Each site below is held to a value that does not go through the code
under test:

* the closed forms of the moments and carrier measures at integer parameters, where
  ``psi(n) = H_{n-1} - gamma`` turns the digammas into harmonic numbers (the ``gamma`` of the
  Euler constant cancels in every one of them) and ``Gamma(n)`` is a factorial;
* the KL divergences against a Gauss-Legendre integral of ``p (log p - log q)`` with the laws' own
  ``log_prob`` (pure Python, float64), and their slopes against central differences;
* the Wishart entropy against the Gamma entropy it is for one dimension, and the LKJ normaliser
  against the integral it normalises;
* the Gamma sampler's implicit gradient against ``-dP/d alpha / pdf`` from ``gammainc``;
* the count samplers' helpers against ``math.lgamma``, and the samplers' moments.
"""

import math
from collections.abc import Callable

import pytest

import lucid
import lucid.distributions as D
from lucid._ops.composite import _count_samplers as samplers
from lucid.distributions.gamma import _standard_gamma_grad
from lucid.test._fixtures.devices import device_dtype_params

_FLOATS = device_dtype_params([lucid.float32, lucid.float64])
_EULER_GAMMA = 0.5772156649015329
_RTOL = {lucid.float32: 2e-5, lucid.float64: 1e-11}

# 5-point Gauss-Legendre on [-1, 1]
_GL_T = (0.0, 0.5384693101056831, -0.5384693101056831, 0.9061798459386640, -0.9061798459386640)
_GL_W = (
    0.5688888888888889,
    0.4786286704993665,
    0.4786286704993665,
    0.2369268850561891,
    0.2369268850561891,
)


def _t64(values: float | list[float]) -> lucid.Tensor:
    return lucid.tensor(values, dtype=lucid.float64)


def _integrate(
    fn: Callable[[lucid.Tensor], lucid.Tensor], a: float, b: float, panels: int
) -> float:
    """``int_a^b fn`` by composite 5-point Gauss-Legendre; ``fn`` takes a float64 tensor of nodes."""
    h = (b - a) / panels
    nodes: list[float] = []
    weights: list[float] = []
    for i in range(panels):
        mid = a + (i + 0.5) * h
        nodes += [mid + 0.5 * h * t for t in _GL_T]
        weights += [0.5 * h * w for w in _GL_W]
    values = fn(_t64(nodes)).tolist()
    return math.fsum(w * v for w, v in zip(weights, values, strict=True))


def _harmonic(n: int) -> float:
    """``H_n = 1 + 1/2 + ... + 1/n``; ``psi(n + 1) = H_n - gamma``."""
    return math.fsum(1.0 / k for k in range(1, n + 1))


def _close(got: float, want: float, rtol: float) -> bool:
    return abs(got - want) <= rtol * max(abs(want), 1.0)


# ── moments and carrier measures at integer parameters ──────────────────────


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
@pytest.mark.parametrize(
    ("scale", "concentration"), [(1.0, 1.0), (2.5, 2.0), (0.7, 3.0), (1.3, 5.0)]
)
def test_weibull_mean_and_variance_are_the_gamma_function_closed_forms(
    scale: float, concentration: float, device: str, dtype: lucid.dtype
) -> None:
    """``mean = lambda Gamma(1 + 1/k)``, ``var = lambda^2 [Gamma(1 + 2/k) - Gamma(1 + 1/k)^2]``."""
    law = D.Weibull(
        lucid.tensor(scale, dtype=dtype, device=device),
        lucid.tensor(concentration, dtype=dtype, device=device),
    )
    g1 = math.gamma(1.0 + 1.0 / concentration)
    g2 = math.gamma(1.0 + 2.0 / concentration)
    assert _close(law.mean.item(), scale * g1, 4 * _RTOL[dtype])
    assert _close(law.variance.item(), scale * scale * (g2 - g1 * g1), 64 * _RTOL[dtype])


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
@pytest.mark.parametrize(("a", "b"), [(0.7, 1), (2.0, 2), (3.5, 3), (1.4, 4)])
def test_kumaraswamy_entropy_at_integer_b_is_a_harmonic_number(
    a: float, b: int, device: str, dtype: lucid.dtype
) -> None:
    """``H = -log a - log b + (1 - 1/a) H_b + (1 - 1/b)``: ``psi(b + 1) = H_b - gamma``."""
    law = D.Kumaraswamy(
        lucid.tensor(a, dtype=dtype, device=device),
        lucid.tensor(float(b), dtype=dtype, device=device),
    )
    want = -math.log(a) - math.log(b) + (1.0 - 1.0 / a) * _harmonic(b) + (1.0 - 1.0 / b)
    assert _close(law.entropy().item(), want, 8 * _RTOL[dtype])


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
@pytest.mark.parametrize(("a", "b"), [(1, 1), (2, 3), (4, 2), (5, 5)])
def test_beta_carrier_measure_at_integers(a: int, b: int, device: str, dtype: lucid.dtype) -> None:
    """``2 psi(a + b) - psi(a) - psi(b) = 2 H_{a+b-1} - H_{a-1} - H_{b-1}``."""
    law = D.Beta(
        lucid.tensor(float(a), dtype=dtype, device=device),
        lucid.tensor(float(b), dtype=dtype, device=device),
    )
    want = 2.0 * _harmonic(a + b - 1) - _harmonic(a - 1) - _harmonic(b - 1)
    assert _close(law._mean_carrier_measure.item(), want, 16 * _RTOL[dtype])


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
@pytest.mark.parametrize("counts", [[1, 1], [2, 3, 4], [1, 5, 2, 3]])
def test_dirichlet_carrier_measure_at_integers(
    counts: list[int], device: str, dtype: lucid.dtype
) -> None:
    """``K psi(sum a) - sum psi(a_i) = K H_{s-1} - sum H_{a_i - 1}``."""
    law = D.Dirichlet(lucid.tensor([float(c) for c in counts], dtype=dtype, device=device))
    k, s = len(counts), sum(counts)
    want = k * _harmonic(s - 1) - math.fsum(_harmonic(c - 1) for c in counts)
    assert _close(law._mean_carrier_measure.item(), want, 16 * _RTOL[dtype])


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
@pytest.mark.parametrize(("alpha", "rate"), [(1.0, 1.0), (2.5, 0.8), (7.0, 3.0)])
def test_gamma_log_normalizer_is_lgamma_minus_alpha_log_rate(
    alpha: float, rate: float, device: str, dtype: lucid.dtype
) -> None:
    """The exponential-family normaliser at the natural parameters ``(alpha - 1, -rate)``."""
    law = D.Gamma(
        lucid.tensor(alpha, dtype=dtype, device=device),
        lucid.tensor(rate, dtype=dtype, device=device),
    )
    natural = tuple(p.detach() for p in law._natural_params)
    want = math.lgamma(alpha) - alpha * math.log(rate)
    assert _close(law._log_normalizer(*natural).item(), want, 8 * _RTOL[dtype])


@pytest.mark.parametrize("probs", [[0.2, 0.3, 0.5], [0.6, 0.4]])
@pytest.mark.parametrize("total", [1, 3, 5])
def test_multinomial_entropy_is_the_enumerated_one(probs: list[float], total: int) -> None:
    """``-sum p log p`` over every count vector, with the pmf written out in ``math``."""

    def vectors(n: int, k: int) -> list[tuple[int, ...]]:
        if k == 1:
            return [(n,)]
        return [(i, *rest) for i in range(n + 1) for rest in vectors(n - i, k - 1)]

    h = 0.0
    for v in vectors(total, len(probs)):
        log_pmf = math.lgamma(total + 1) + math.fsum(
            c * math.log(p) - math.lgamma(c + 1) for c, p in zip(v, probs, strict=True)
        )
        h -= math.exp(log_pmf) * log_pmf
    got = D.Multinomial(total, probs=_t64(probs)).entropy().item()
    assert _close(got, h, 1e-11)


# ── KL divergences against quadrature ───────────────────────────────────────


def _kl_by_quadrature(
    p: D.Distribution, q: D.Distribution, a: float, b: float, panels: int
) -> float:
    def integrand(x: lucid.Tensor) -> lucid.Tensor:
        lp = p.log_prob(x)
        return lp.exp() * (lp - q.log_prob(x))

    return _integrate(integrand, a, b, panels)


def test_kl_beta_gamma_matches_the_integral() -> None:
    p = D.Beta(_t64(2.5), _t64(3.5))
    q = D.Gamma(_t64(3.0), _t64(2.0))
    want = _kl_by_quadrature(p, q, 0.0, 1.0, 96)
    assert _close(D.kl_divergence(p, q).item(), want, 1e-7)


def test_kl_gamma_normal_matches_the_integral() -> None:
    p = D.Gamma(_t64(3.2), _t64(1.4))
    q = D.Normal(_t64(1.0), _t64(1.5))
    want = _kl_by_quadrature(p, q, 0.0, 45.0, 180)
    assert _close(D.kl_divergence(p, q).item(), want, 1e-7)


def test_kl_uniform_gamma_matches_the_integral() -> None:
    p = D.Uniform(_t64(0.5), _t64(2.5))
    q = D.Gamma(_t64(2.0), _t64(1.5))
    want = _kl_by_quadrature(p, q, 0.5, 2.5, 8)
    assert _close(D.kl_divergence(p, q).item(), want, 1e-9)


def test_kl_pareto_gamma_matches_the_integral() -> None:
    """``E_p f = int_0^1 f(scale / w) 4 w^3 dw`` for ``Pareto(scale, 4)`` (``x = scale w^(-1)``)."""
    scale, alpha = 1.2, 4.0
    p = D.Pareto(_t64(scale), _t64(alpha))
    q = D.Gamma(_t64(2.5), _t64(0.8))

    def integrand(w: lucid.Tensor) -> lucid.Tensor:
        x = scale / w
        return (p.log_prob(x) - q.log_prob(x)) * (4.0 * w**3)

    want = _integrate(integrand, 0.0, 1.0, 96)
    assert _close(D.kl_divergence(p, q).item(), want, 1e-7)


def test_kl_gumbel_gumbel_matches_the_integral() -> None:
    p = D.Gumbel(_t64(0.3), _t64(1.2))
    q = D.Gumbel(_t64(-0.5), _t64(0.9))
    want = _kl_by_quadrature(p, q, 0.3 - 7.0 * 1.2, 0.3 + 46.0 * 1.2, 240)
    assert _close(D.kl_divergence(p, q).item(), want, 1e-7)


def _central(fn: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    return (fn(x + h) - fn(x - h)) / (2.0 * h)


@pytest.mark.parametrize("which", ["beta_gamma", "gamma_normal", "pareto_gamma", "uniform_gamma"])
def test_kl_slope_in_the_gamma_concentration_is_the_central_difference(which: str) -> None:
    """The slopes read ``_digamma_positive``'s own slope (the trigamma): finite, and the derivative."""

    def kl(alpha: lucid.Tensor) -> lucid.Tensor:
        if which == "beta_gamma":
            return D.kl_divergence(D.Beta(_t64(2.5), _t64(3.5)), D.Gamma(alpha, _t64(2.0)))
        if which == "gamma_normal":
            return D.kl_divergence(D.Gamma(alpha, _t64(1.4)), D.Normal(_t64(1.0), _t64(1.5)))
        if which == "pareto_gamma":
            return D.kl_divergence(D.Pareto(_t64(1.2), _t64(4.0)), D.Gamma(alpha, _t64(0.8)))
        return D.kl_divergence(D.Uniform(_t64(0.5), _t64(2.5)), D.Gamma(alpha, _t64(1.5)))

    a0 = 3.2
    x = lucid.tensor(a0, dtype=lucid.float64, requires_grad=True)
    (g,) = lucid.autograd.grad(kl(x), [x])
    want = _central(lambda a: kl(_t64(a)).item(), a0)
    assert math.isfinite(g.item())
    assert _close(g.item(), want, 1e-6)


def test_kl_gamma_normal_slope_in_the_source_concentration() -> None:
    """``KL(Gamma(a, b) || Normal)`` in ``a``: the ``digamma`` and ``lgamma`` of ``p`` both differentiate."""
    p_rate = _t64(1.4)

    def kl(alpha: float) -> float:
        return D.kl_divergence(D.Gamma(_t64(alpha), p_rate), D.Normal(_t64(0.0), _t64(2.0))).item()

    x = lucid.tensor(2.7, dtype=lucid.float64, requires_grad=True)
    out = D.kl_divergence(D.Gamma(x, p_rate), D.Normal(_t64(0.0), _t64(2.0)))
    (g,) = lucid.autograd.grad(out, [x])
    assert _close(g.item(), _central(kl, 2.7), 1e-6)


# ── Wishart and LKJ ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(("df", "variance"), [(3.5, 0.6), (7.0, 2.0), (1.5, 1.0)])
def test_wishart_entropy_in_one_dimension_is_the_gamma_entropy(df: float, variance: float) -> None:
    """``Wishart(nu, s^2)`` in 1-d is ``s^2 chi^2_nu = Gamma(nu / 2, 1 / (2 s^2))``."""
    wishart = D.Wishart(_t64(df), covariance_matrix=_t64([[variance]]))
    gamma = D.Gamma(_t64(df / 2.0), _t64(1.0 / (2.0 * variance)))
    assert _close(wishart.entropy().item(), gamma.entropy().item(), 1e-10)


def test_wishart_entropy_in_two_dimensions_at_known_digammas() -> None:
    """``Wishart(4, I_2)``: ``psi(2) = 1 - gamma`` and ``psi(3/2) = 2 - gamma - 2 log 2`` are exact."""
    nu, p = 4.0, 2
    psi_p = (1.0 - _EULER_GAMMA) + (2.0 - _EULER_GAMMA - 2.0 * math.log(2.0))
    log_gamma_p = 0.5 * math.log(math.pi) + math.lgamma(2.0) + math.lgamma(1.5)
    want = (
        (p + 1) * (p * math.log(2.0) / 2.0)
        + log_gamma_p
        - (nu - p - 1.0) / 2.0 * psi_p
        + nu * p / 2.0
    )
    got = D.Wishart(_t64(nu), covariance_matrix=lucid.eye(2, dtype=lucid.float64)).entropy().item()
    assert _close(got, want, 1e-11)


@pytest.mark.parametrize("eta", [1.0, 2.0, 3.5])
def test_lkj_two_by_two_normalizer_and_density(eta: float) -> None:
    """In 2-d ``L = [[1, 0], [r, sqrt(1 - r^2)]]`` has density ``(1 - r^2)^(eta - 1) / Z`` in ``r``.

    ``Z = sqrt(pi) Gamma(eta) / Gamma(eta + 1/2)``, and the density integrates to 1.
    """
    law = D.LKJCholesky(2, _t64(eta))
    z = math.sqrt(math.pi) * math.gamma(eta) / math.gamma(eta + 0.5)
    assert _close(law._log_normalizer().item(), math.log(z), 1e-11)

    def density(r: lucid.Tensor) -> lucid.Tensor:
        s = (1.0 - r * r).sqrt()
        zeros = lucid.zeros_like(r)
        ones = lucid.ones_like(r)
        chol = lucid.stack(
            [lucid.stack([ones, zeros], dim=-1), lucid.stack([r, s], dim=-1)], dim=-2
        )
        return law.log_prob(chol).exp()

    assert _close(_integrate(density, -1.0, 1.0, 64), 1.0, 1e-5)


# ── the Gamma sampler's implicit gradient ───────────────────────────────────


@pytest.mark.parametrize("alpha", [0.3, 1.0, 2.5, 9.0])
@pytest.mark.parametrize("z", [0.05, 0.5, 1.2, 4.0, 15.0])
def test_standard_gamma_grad_is_minus_dp_dalpha_over_the_density(alpha: float, z: float) -> None:
    """``dz/dalpha`` at fixed ``P(alpha, z)``: ``-(dP/dalpha) / pdf``, ``dP/dalpha`` central."""
    h = 1e-5

    def cdf(a: float) -> float:
        return lucid.special.gammainc(_t64(a), _t64(z)).item()

    pdf = math.exp((alpha - 1.0) * math.log(z) - z - math.lgamma(alpha))
    want = -_central(cdf, alpha, h) / pdf
    got = _standard_gamma_grad(_t64(alpha), _t64(z)).item()
    # the Rice expansion and the rational fits are good to 1.5e-3 relative on this grid
    assert abs(got - want) <= 4e-3 * max(abs(want), 1e-3), (alpha, z, got, want)


# ── the count samplers ──────────────────────────────────────────────────────


def test_poisson_log_pmf_is_the_closed_form_on_both_sides_of_ten() -> None:
    """Below 10 the plain form (``lgamma(k + 1)``), from 10 the deviance form: one function."""
    lam = 7.3
    ks = [float(k) for k in (0, 1, 2, 5, 9, 10, 11, 20, 40, 100)]
    got = samplers._poisson_log_pmf(_t64(ks), _t64(lam), _t64(math.log(lam))).tolist()
    for k, g in zip(ks, got, strict=True):
        want = -lam + k * math.log(lam) - math.lgamma(k + 1.0)
        assert abs(g - want) <= 1e-10 * max(1.0, abs(want)), (k, g, want)


def test_stirling_tail_is_ln_factorial_minus_its_stirling_part() -> None:
    """``ln k! - ((k + 1/2) ln(k + 1) - (k + 1) + 1/2 ln 2 pi)``: lgamma side and series side."""
    ks = [float(k) for k in (0, 1, 3, 9, 10, 11, 25, 60)]
    got = samplers._stirling_tail(_t64(ks)).tolist()
    for k, g in zip(ks, got, strict=True):
        want = math.lgamma(k + 1.0) - (
            (k + 0.5) * math.log(k + 1.0) - (k + 1.0) + 0.9189385332046727
        )
        assert abs(g - want) <= 1e-9, (k, g, want)


def test_poisson_sampler_moments_use_the_exact_pmf() -> None:
    """``rate >= 10`` is PTRS, whose accept test is the log pmf: mean and variance are the rate."""
    lucid.manual_seed(7)
    rate = 42.0
    draws = samplers.poisson_counts(lucid.full((20000,), rate, dtype=lucid.float32)).tolist()
    mean = math.fsum(draws) / len(draws)
    var = math.fsum((d - mean) ** 2 for d in draws) / len(draws)
    assert abs(mean - rate) <= 0.3, mean
    assert abs(var - rate) <= 2.0, var


def test_binomial_sampler_moments_use_the_stirling_tail() -> None:
    """A large ``n p`` is BTRS, whose accept test reads ``ln k!`` through the tail."""
    lucid.manual_seed(11)
    n, p = 400.0, 0.35
    draws = samplers.binomial_counts(
        lucid.full((20000,), n, dtype=lucid.float32), lucid.full((20000,), p, dtype=lucid.float32)
    ).tolist()
    mean = math.fsum(draws) / len(draws)
    var = math.fsum((d - mean) ** 2 for d in draws) / len(draws)
    assert abs(mean - n * p) <= 0.4, mean
    assert abs(var - n * p * (1.0 - p)) <= 3.0, var


# ── a Gamma sample's gradient reaches the concentration ─────────────────────


def test_gamma_rsample_gradient_averages_to_the_mean_slope() -> None:
    """``E[z] = alpha / beta`` so ``E[dz/dalpha] = 1 / beta``; small ``alpha`` sits in the series regime."""
    lucid.manual_seed(3)
    beta = 2.0
    for alpha in (0.5, 6.0):
        concentration = lucid.full((40000,), alpha, dtype=lucid.float32, requires_grad=True)
        sample = D.Gamma(concentration, lucid.full((40000,), beta, dtype=lucid.float32)).rsample()
        (g,) = lucid.autograd.grad(sample.sum(), [concentration])
        assert abs(g.mean().item() - 1.0 / beta) <= 0.03, (alpha, g.mean().item())
