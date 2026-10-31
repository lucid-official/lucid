"""The matrix-valued distributions.

``distributions/matrix.py`` sat at 29.1%.  ``MultivariateNormal``,
``Wishart`` and ``LKJCholesky`` were exported and barely touched, and
they are the ones where a wrong factorisation is hardest to notice: a
sample from a Gaussian with the wrong covariance still looks like noise.

So the checks are on the structure the distribution promises — the
empirical covariance of many samples, the positive-definiteness of a
Wishart draw, the unit diagonal of an LKJ correlation — rather than on
individual numbers.
"""

import numpy as np
import pytest

import lucid
import lucid.distributions as D

MEAN = np.array([1.0, -2.0, 0.5])
COV = np.array([[2.0, 0.3, 0.1], [0.3, 1.0, -0.2], [0.1, -0.2, 1.5]])


def _t(a):
    return lucid.tensor(np.asarray(a, dtype=np.float64))


def _v(x):
    return np.asarray(x.numpy())


# ── MultivariateNormal ────────────────────────────────────────────────────────


def test_it_accepts_a_full_covariance():
    dist = D.MultivariateNormal(_t(MEAN), covariance_matrix=_t(COV))
    assert tuple(dist.mean.shape) == (3,)


def test_the_sample_mean_converges_on_the_mean():
    lucid.manual_seed(0)
    dist = D.MultivariateNormal(_t(MEAN), covariance_matrix=_t(COV))
    draws = _v(dist.sample((20000,)))
    assert draws.shape == (20000, 3)
    assert np.allclose(draws.mean(axis=0), MEAN, atol=0.05)


def test_the_sample_covariance_converges_on_the_covariance():
    """The part a wrong factorisation would break while still looking like
    plausible noise."""
    lucid.manual_seed(1)
    dist = D.MultivariateNormal(_t(MEAN), covariance_matrix=_t(COV))
    draws = _v(dist.sample((40000,)))
    assert np.allclose(np.cov(draws.T), COV, atol=0.08)


def test_log_prob_matches_the_written_out_density():
    dist = D.MultivariateNormal(_t(MEAN), covariance_matrix=_t(COV))
    point = np.array([0.5, -1.0, 1.0])
    got = float(_v(dist.log_prob(_t(point))).ravel()[0])

    delta = point - MEAN
    inv = np.linalg.inv(COV)
    expected = -0.5 * (delta @ inv @ delta + np.log(np.linalg.det(COV)) + 3 * np.log(2 * np.pi))
    assert np.isclose(got, expected, atol=1e-8)


def test_log_prob_peaks_at_the_mean():
    dist = D.MultivariateNormal(_t(MEAN), covariance_matrix=_t(COV))
    at_mean = float(_v(dist.log_prob(_t(MEAN))).ravel()[0])
    away = float(_v(dist.log_prob(_t(MEAN + 1.0))).ravel()[0])
    assert at_mean > away


def test_a_scale_tril_parameterisation_agrees_with_the_covariance_one():
    """``scale_tril @ scale_tril.T`` is the covariance, so the two spellings
    have to give the same density."""
    tril = np.linalg.cholesky(COV)
    by_cov = D.MultivariateNormal(_t(MEAN), covariance_matrix=_t(COV))
    by_tril = D.MultivariateNormal(_t(MEAN), scale_tril=_t(tril))
    point = _t(np.array([0.0, 0.0, 0.0]))
    assert np.isclose(
        float(_v(by_cov.log_prob(point)).ravel()[0]),
        float(_v(by_tril.log_prob(point)).ravel()[0]),
        atol=1e-8,
    )


def test_entropy_grows_with_the_spread():
    tight = D.MultivariateNormal(_t(MEAN), covariance_matrix=_t(COV * 0.1))
    wide = D.MultivariateNormal(_t(MEAN), covariance_matrix=_t(COV * 10.0))
    assert float(_v(wide.entropy()).ravel()[0]) > float(_v(tight.entropy()).ravel()[0])


def test_it_refuses_two_parameterisations_at_once():
    with pytest.raises(Exception):
        D.MultivariateNormal(
            _t(MEAN), covariance_matrix=_t(COV), scale_tril=_t(np.linalg.cholesky(COV))
        )


# ── Wishart ───────────────────────────────────────────────────────────────────


def test_a_wishart_draw_is_symmetric_positive_definite():
    """The defining property: it is a distribution over covariance
    matrices, so every draw has to be usable as one."""
    lucid.manual_seed(2)
    dist = D.Wishart(df=_t(5.0), covariance_matrix=_t(np.eye(3)))
    matrix = _v(dist.sample()).reshape(3, 3)
    assert np.allclose(matrix, matrix.T, atol=1e-8)
    assert np.all(np.linalg.eigvalsh(matrix) > 0)


def test_the_wishart_mean_is_df_times_the_scale():
    lucid.manual_seed(3)
    scale = np.eye(3) * 2.0
    dist = D.Wishart(df=_t(7.0), covariance_matrix=_t(scale))
    draws = np.stack([_v(dist.sample()).reshape(3, 3) for _ in range(1500)])
    assert np.allclose(draws.mean(axis=0), 7.0 * scale, atol=0.8)
    assert np.allclose(_v(dist.mean), 7.0 * scale, atol=1e-10)


def test_wishart_refuses_too_few_degrees_of_freedom():
    """``df ≤ d − 1`` puts the Bartlett diagonal at χ²(0) and χ²(−1),
    which is not a distribution.  It used to sample anyway and return a
    rank-deficient matrix — outside the ``positive_definite`` support the
    class declares, with nothing raised to say so."""
    with pytest.raises(ValueError, match="df must exceed"):
        D.Wishart(df=_t(1.0), covariance_matrix=_t(np.eye(3)))
    with pytest.raises(ValueError, match="df must exceed"):
        D.Wishart(df=_t(2.0), covariance_matrix=_t(np.eye(3)))
    D.Wishart(df=_t(2.5), covariance_matrix=_t(np.eye(3)))  # just above


def test_the_wishart_variance_is_the_textbook_one():
    """``Var[W_ij] = df · (Σ_ij² + Σ_ii Σ_jj)`` — checked against draws
    rather than against the formula it was written from."""
    lucid.manual_seed(9)
    scale = np.array([[2.0, 0.5], [0.5, 1.0]])
    dist = D.Wishart(df=_t(8.0), covariance_matrix=_t(scale))
    draws = np.stack([_v(dist.sample()).reshape(2, 2) for _ in range(4000)])
    assert np.allclose(draws.var(axis=0), _v(dist.variance), rtol=0.15)


# ── LKJCholesky ───────────────────────────────────────────────────────────────


def test_an_lkj_draw_is_a_cholesky_factor_of_a_correlation_matrix():
    """Lower-triangular with unit rows, so the product has a unit
    diagonal — that is what makes it a *correlation* matrix."""
    lucid.manual_seed(4)
    dist = D.LKJCholesky(dim=4, concentration=_t(1.0))
    factor = _v(dist.sample()).reshape(4, 4)
    assert np.allclose(np.triu(factor, 1), 0.0, atol=1e-10)
    corr = factor @ factor.T
    assert np.allclose(np.diag(corr), 1.0, atol=1e-8)
    assert np.all(np.linalg.eigvalsh(corr) > -1e-10)


def _lkj_correlations(dist, dim, draws):
    out = []
    for _ in range(draws):
        factor = _v(dist.sample()).reshape(dim, dim)
        corr = factor @ factor.T
        out.extend(corr[np.triu_indices(dim, 1)])
    return np.asarray(out)


@pytest.mark.parametrize("dim", [2, 3, 5])
@pytest.mark.parametrize("eta", [1.0, 2.0, 5.0])
def test_the_marginal_correlation_matches_its_closed_form(dim, eta):
    """The check the structural ones cannot make.

    Under LKJ(η) in dimension ``d`` every off-diagonal ``r`` satisfies
    ``(1 + r) / 2 ~ Beta(a, a)`` with ``a = η + (d − 2) / 2``, so
    ``Var[r] = 1 / (2a + 1)``.  Unit diagonal and positive-definiteness
    hold for *any* correlation matrix, so they cannot tell a correct
    sampler from one drawing at the wrong concentration — this can.

    It was wrong: the Onion sampler read the Beta parameters one row
    early and doubled ``i/2`` into ``i``, which roughly doubled the
    variance at every ``(d, η)``.  Every draw still had a unit diagonal
    and non-negative eigenvalues, so nothing downstream complained.
    """
    lucid.manual_seed(10 + dim)
    dist = D.LKJCholesky(dim=dim, concentration=_t(eta))
    r = _lkj_correlations(dist, dim, 400)
    a = eta + (dim - 2) / 2
    assert np.isclose(r.var(), 1.0 / (2 * a + 1), rtol=0.12)
    assert abs(r.mean()) < 0.06  # symmetric about zero


def test_a_high_concentration_pushes_toward_the_identity():
    """Large concentration concentrates the density on the uncorrelated
    matrix, so the off-diagonals shrink."""
    lucid.manual_seed(5)
    loose = _lkj_correlations(D.LKJCholesky(dim=3, concentration=_t(0.5)), 3, 60)
    tight = _lkj_correlations(D.LKJCholesky(dim=3, concentration=_t(50.0)), 3, 60)
    assert np.abs(tight).mean() < np.abs(loose).mean()


def test_a_concentration_below_a_half_is_accepted():
    """It used to raise: the shifted Beta offset drove the last
    ``concentration0`` to ``η − 1/2``, so any ``η ≤ 0.5`` was refused by
    the constraint check rather than by anything about LKJ."""
    dist = D.LKJCholesky(dim=4, concentration=_t(0.3))
    factor = _v(dist.sample()).reshape(4, 4)
    assert np.allclose(np.diag(factor @ factor.T), 1.0, atol=1e-6)


@pytest.mark.parametrize("eta", [0.5, 1.0, 2.0, 5.0])
@pytest.mark.parametrize("r", [-0.6, 0.0, 0.3])
def test_lkj_log_prob_matches_the_two_by_two_closed_form(eta, r):
    """At ``d = 2`` the factor is ``[[1, 0], [r, sqrt(1 − r²)]]`` and the
    density is ``(1 − r²)^(η−1) / Z`` with
    ``Z = sqrt(π) Γ(η) / Γ(η + 1/2)`` — normaliser included, so this
    pins the constant and not merely the shape."""
    import math

    dist = D.LKJCholesky(dim=2, concentration=_t(eta))
    factor = np.array([[1.0, 0.0], [r, math.sqrt(1 - r * r)]])
    got = float(_v(dist.log_prob(_t(factor))).ravel()[0])
    normaliser = math.sqrt(math.pi) * math.gamma(eta) / math.gamma(eta + 0.5)
    assert np.isclose(got, (eta - 1) * math.log(1 - r * r) - math.log(normaliser))


def test_lkj_log_prob_is_finite_on_its_own_sample():
    lucid.manual_seed(6)
    dist = D.LKJCholesky(dim=3, concentration=_t(2.0))
    sample = dist.sample()
    assert np.isfinite(float(_v(dist.log_prob(sample)).ravel()[0]))


def test_lkj_refuses_a_one_dimensional_correlation_matrix():
    with pytest.raises(ValueError, match="dim must be"):
        D.LKJCholesky(dim=1)


@pytest.mark.parametrize("dim", [2.0, 2.5, lucid.tensor(2.0)])
def test_lkj_cholesky_refuses_a_non_integer_dim(dim: int) -> None:
    # A float dim used to be stored and only failed inside log_prob, once
    # multigammaln started validating its order (LCD-772).
    with pytest.raises(TypeError, match="dim must be an integer"):
        D.LKJCholesky(dim)
