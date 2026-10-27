"""Reference parity for distributions — log_prob + closed-form KL."""

from typing import Any

import numpy as np
import pytest

import lucid
import lucid.distributions as D

# ── helpers ──────────────────────────────────────────────────────────────────


def _lp(dist: Any, x: Any) -> float:
    """Return scalar log_prob as Python float."""
    return dist.log_prob(x).item()


def _ent(dist: Any) -> float:
    return dist.entropy().item()


# ── univariate continuous ─────────────────────────────────────────────────────


@pytest.mark.parity
class TestNormalParity:
    def test_log_prob(self, ref: Any) -> None:
        loc, scale = 0.5, 1.5
        x = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        l = D.Normal(loc, scale).log_prob(lucid.tensor(x.copy())).numpy()
        r = (
            ref.distributions.Normal(loc, scale)
            .log_prob(ref.tensor(x.copy()))
            .detach()
            .cpu()
            .numpy()
        )
        np.testing.assert_allclose(l, r, atol=1e-5)

    def test_entropy(self, ref: Any) -> None:
        loc, scale = 0.5, 1.5
        assert (
            abs(_ent(D.Normal(loc, scale)) - ref.distributions.Normal(loc, scale).entropy().item())
            < 1e-5
        )


@pytest.mark.parity
class TestUniformParity:
    def test_log_prob(self, ref: Any) -> None:
        x = np.array([0.2, 0.5, 0.8], dtype=np.float32)
        l = D.Uniform(0.0, 1.0).log_prob(lucid.tensor(x.copy())).numpy()
        r = (
            ref.distributions.Uniform(0.0, 1.0)
            .log_prob(ref.tensor(x.copy()))
            .detach()
            .cpu()
            .numpy()
        )
        np.testing.assert_allclose(l, r, atol=1e-5)

    def test_entropy(self, ref: Any) -> None:
        assert (
            abs(_ent(D.Uniform(0.0, 2.0)) - ref.distributions.Uniform(0.0, 2.0).entropy().item())
            < 1e-5
        )


@pytest.mark.parity
class TestExponentialParity:
    def test_log_prob(self, ref: Any) -> None:
        x = np.array([0.5, 1.0, 2.0], dtype=np.float32)
        l = D.Exponential(2.0).log_prob(lucid.tensor(x.copy())).numpy()
        r = ref.distributions.Exponential(2.0).log_prob(ref.tensor(x.copy())).detach().cpu().numpy()
        np.testing.assert_allclose(l, r, atol=1e-5)

    def test_entropy(self, ref: Any) -> None:
        assert (
            abs(_ent(D.Exponential(2.0)) - ref.distributions.Exponential(2.0).entropy().item())
            < 1e-5
        )


@pytest.mark.parity
class TestGammaParity:
    def test_log_prob(self, ref: Any) -> None:
        x = np.array([0.5, 1.0, 2.0, 3.0], dtype=np.float32)
        l = D.Gamma(2.0, 1.0).log_prob(lucid.tensor(x.copy())).numpy()
        r = ref.distributions.Gamma(2.0, 1.0).log_prob(ref.tensor(x.copy())).detach().cpu().numpy()
        np.testing.assert_allclose(l, r, atol=1e-5)

    def test_entropy(self, ref: Any) -> None:
        assert (
            abs(_ent(D.Gamma(3.0, 2.0)) - ref.distributions.Gamma(3.0, 2.0).entropy().item()) < 1e-4
        )


@pytest.mark.parity
class TestBetaParity:
    def test_log_prob(self, ref: Any) -> None:
        x = np.array([0.1, 0.3, 0.5, 0.8], dtype=np.float32)
        l = D.Beta(2.0, 3.0).log_prob(lucid.tensor(x.copy())).numpy()
        r = ref.distributions.Beta(2.0, 3.0).log_prob(ref.tensor(x.copy())).detach().cpu().numpy()
        np.testing.assert_allclose(l, r, atol=1e-5)

    def test_entropy(self, ref: Any) -> None:
        assert (
            abs(_ent(D.Beta(2.0, 3.0)) - ref.distributions.Beta(2.0, 3.0).entropy().item()) < 1e-4
        )


@pytest.mark.parity
class TestLaplaceParity:
    def test_log_prob(self, ref: Any) -> None:
        x = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        l = D.Laplace(0.0, 1.0).log_prob(lucid.tensor(x.copy())).numpy()
        r = (
            ref.distributions.Laplace(0.0, 1.0)
            .log_prob(ref.tensor(x.copy()))
            .detach()
            .cpu()
            .numpy()
        )
        np.testing.assert_allclose(l, r, atol=1e-5)

    def test_entropy(self, ref: Any) -> None:
        assert (
            abs(_ent(D.Laplace(0.0, 2.0)) - ref.distributions.Laplace(0.0, 2.0).entropy().item())
            < 1e-5
        )


@pytest.mark.parity
class TestCauchyParity:
    def test_log_prob(self, ref: Any) -> None:
        x = np.array([-2.0, 0.0, 1.0, 3.0], dtype=np.float32)
        l = D.Cauchy(0.0, 1.0).log_prob(lucid.tensor(x.copy())).numpy()
        r = ref.distributions.Cauchy(0.0, 1.0).log_prob(ref.tensor(x.copy())).detach().cpu().numpy()
        np.testing.assert_allclose(l, r, atol=1e-5)

    def test_entropy(self, ref: Any) -> None:
        assert (
            abs(_ent(D.Cauchy(0.0, 1.0)) - ref.distributions.Cauchy(0.0, 1.0).entropy().item())
            < 1e-5
        )


@pytest.mark.parity
class TestStudentTParity:
    def test_log_prob(self, ref: Any) -> None:
        x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        l = D.StudentT(3.0, 0.0, 1.0).log_prob(lucid.tensor(x.copy())).numpy()
        r = (
            ref.distributions.StudentT(3.0, 0.0, 1.0)
            .log_prob(ref.tensor(x.copy()))
            .detach()
            .cpu()
            .numpy()
        )
        np.testing.assert_allclose(l, r, atol=1e-4)


@pytest.mark.parity
class TestLogNormalParity:
    def test_log_prob(self, ref: Any) -> None:
        x = np.array([0.5, 1.0, 2.0], dtype=np.float32)
        l = D.LogNormal(0.0, 1.0).log_prob(lucid.tensor(x.copy())).numpy()
        r = (
            ref.distributions.LogNormal(0.0, 1.0)
            .log_prob(ref.tensor(x.copy()))
            .detach()
            .cpu()
            .numpy()
        )
        np.testing.assert_allclose(l, r, atol=1e-5)


@pytest.mark.parity
class TestHalfNormalParity:
    def test_log_prob(self, ref: Any) -> None:
        x = np.array([0.5, 1.0, 2.0], dtype=np.float32)
        l = D.HalfNormal(1.0).log_prob(lucid.tensor(x.copy())).numpy()
        r = ref.distributions.HalfNormal(1.0).log_prob(ref.tensor(x.copy())).detach().cpu().numpy()
        np.testing.assert_allclose(l, r, atol=1e-5)


@pytest.mark.parity
class TestHalfCauchyParity:
    def test_log_prob(self, ref: Any) -> None:
        x = np.array([0.5, 1.0, 2.0], dtype=np.float32)
        l = D.HalfCauchy(1.0).log_prob(lucid.tensor(x.copy())).numpy()
        r = ref.distributions.HalfCauchy(1.0).log_prob(ref.tensor(x.copy())).detach().cpu().numpy()
        np.testing.assert_allclose(l, r, atol=1e-5)


@pytest.mark.parity
class TestParetoParity:
    def test_log_prob(self, ref: Any) -> None:
        x = np.array([1.5, 2.0, 3.0], dtype=np.float32)
        l = D.Pareto(1.0, 2.0).log_prob(lucid.tensor(x.copy())).numpy()
        r = ref.distributions.Pareto(1.0, 2.0).log_prob(ref.tensor(x.copy())).detach().cpu().numpy()
        np.testing.assert_allclose(l, r, atol=1e-5)


@pytest.mark.parity
class TestWeibullParity:
    def test_log_prob(self, ref: Any) -> None:
        x = np.array([0.5, 1.0, 2.0], dtype=np.float32)
        l = D.Weibull(2.0, 1.5).log_prob(lucid.tensor(x.copy())).numpy()
        r = (
            ref.distributions.Weibull(2.0, 1.5)
            .log_prob(ref.tensor(x.copy()))
            .detach()
            .cpu()
            .numpy()
        )
        np.testing.assert_allclose(l, r, atol=1e-5)


@pytest.mark.parity
class TestFisherSnedecorParity:
    def test_log_prob(self, ref: Any) -> None:
        x = np.array([0.5, 1.0, 2.0], dtype=np.float32)
        l = D.FisherSnedecor(2.0, 4.0).log_prob(lucid.tensor(x.copy())).numpy()
        r = (
            ref.distributions.FisherSnedecor(2.0, 4.0)
            .log_prob(ref.tensor(x.copy()))
            .detach()
            .cpu()
            .numpy()
        )
        np.testing.assert_allclose(l, r, atol=1e-4)


# ── univariate discrete ───────────────────────────────────────────────────────


@pytest.mark.parity
class TestBernoulliParity:
    def test_log_prob(self, ref: Any) -> None:
        x = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)
        l = D.Bernoulli(probs=0.7).log_prob(lucid.tensor(x.copy())).numpy()
        r = (
            ref.distributions.Bernoulli(probs=0.7)
            .log_prob(ref.tensor(x.copy()))
            .detach()
            .cpu()
            .numpy()
        )
        np.testing.assert_allclose(l, r, atol=1e-5)

    def test_entropy(self, ref: Any) -> None:
        assert (
            abs(_ent(D.Bernoulli(0.3)) - ref.distributions.Bernoulli(0.3).entropy().item()) < 1e-5
        )


@pytest.mark.parity
class TestGeometricParity:
    def test_log_prob(self, ref: Any) -> None:
        x = np.array([1.0, 2.0, 3.0, 5.0], dtype=np.float32)
        l = D.Geometric(probs=0.4).log_prob(lucid.tensor(x.copy())).numpy()
        r = (
            ref.distributions.Geometric(probs=0.4)
            .log_prob(ref.tensor(x.copy()))
            .detach()
            .cpu()
            .numpy()
        )
        np.testing.assert_allclose(l, r, atol=1e-5)


@pytest.mark.parity
class TestPoissonParity:
    def test_log_prob(self, ref: Any) -> None:
        x = np.array([0.0, 1.0, 2.0, 5.0], dtype=np.float32)
        l = D.Poisson(3.0).log_prob(lucid.tensor(x.copy())).numpy()
        r = ref.distributions.Poisson(3.0).log_prob(ref.tensor(x.copy())).detach().cpu().numpy()
        np.testing.assert_allclose(l, r, atol=1e-5)


@pytest.mark.parity
class TestBinomialParity:
    def test_log_prob(self, ref: Any) -> None:
        x = np.array([0.0, 2.0, 4.0, 5.0], dtype=np.float32)
        l = D.Binomial(total_count=5, probs=0.4).log_prob(lucid.tensor(x.copy())).numpy()
        r = (
            ref.distributions.Binomial(5, probs=0.4)
            .log_prob(ref.tensor(x.copy()))
            .detach()
            .cpu()
            .numpy()
        )
        np.testing.assert_allclose(l, r, atol=1e-4)


@pytest.mark.parity
class TestNegativeBinomialParity:
    def test_log_prob(self, ref: Any) -> None:
        x = np.array([0.0, 1.0, 3.0, 5.0], dtype=np.float32)
        l = D.NegativeBinomial(total_count=3, probs=0.5).log_prob(lucid.tensor(x.copy())).numpy()
        r = (
            ref.distributions.NegativeBinomial(3, probs=0.5)
            .log_prob(ref.tensor(x.copy()))
            .detach()
            .cpu()
            .numpy()
        )
        np.testing.assert_allclose(l, r, atol=1e-4)


@pytest.mark.parity
class TestCategoricalParity:
    def test_log_prob(self, ref: Any) -> None:
        p = [0.1, 0.5, 0.4]
        # Scalar index — shape matches the 1-D probs tensor.
        for idx in [0, 1, 2]:
            l = D.Categorical(probs=lucid.tensor(p)).log_prob(lucid.tensor(idx)).item()
            r = ref.distributions.Categorical(probs=ref.tensor(p)).log_prob(ref.tensor(idx)).item()
            assert abs(l - r) < 1e-5, f"idx={idx}"

    def test_entropy(self, ref: Any) -> None:
        p = [0.1, 0.5, 0.4]
        assert (
            abs(
                _ent(D.Categorical(probs=lucid.tensor(p)))
                - ref.distributions.Categorical(probs=ref.tensor(p)).entropy().item()
            )
            < 1e-5
        )


# ── multivariate ──────────────────────────────────────────────────────────────


@pytest.mark.parity
class TestDirichletParity:
    def test_log_prob(self, ref: Any) -> None:
        conc = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        x = np.array([0.2, 0.3, 0.5], dtype=np.float32)
        l = D.Dirichlet(lucid.tensor(conc)).log_prob(lucid.tensor(x)).item()
        r = ref.distributions.Dirichlet(ref.tensor(conc)).log_prob(ref.tensor(x)).item()
        assert abs(l - r) < 1e-4


@pytest.mark.parity
class TestMultivariateNormalParity:
    def test_log_prob(self, ref: Any) -> None:
        loc = np.array([0.0, 0.0], dtype=np.float32)
        cov = np.array([[1.0, 0.3], [0.3, 1.0]], dtype=np.float32)
        x = np.array([0.5, -0.5], dtype=np.float32)
        l = (
            D.MultivariateNormal(lucid.tensor(loc), lucid.tensor(cov))
            .log_prob(lucid.tensor(x))
            .item()
        )
        r = (
            ref.distributions.MultivariateNormal(ref.tensor(loc), ref.tensor(cov))
            .log_prob(ref.tensor(x))
            .item()
        )
        assert abs(l - r) < 1e-4


# ── composite / wrapper ───────────────────────────────────────────────────────


@pytest.mark.parity
class TestIndependentParity:
    def test_log_prob_sum(self, ref: Any) -> None:
        loc = np.array([0.0, 1.0, -1.0], dtype=np.float32)
        x = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        l = D.Independent(D.Normal(lucid.tensor(loc), 1.0), 1).log_prob(lucid.tensor(x)).item()
        r = (
            ref.distributions.Independent(ref.distributions.Normal(ref.tensor(loc), 1.0), 1)
            .log_prob(ref.tensor(x))
            .item()
        )
        assert abs(l - r) < 1e-4


@pytest.mark.parity
class TestTransformedDistributionParity:
    def test_log_normal_via_transform(self, ref: Any) -> None:
        x = np.array([0.5, 1.0, 2.0], dtype=np.float32)
        l_dist = D.TransformedDistribution(D.Normal(0.0, 1.0), D.ExpTransform())
        r_dist = ref.distributions.TransformedDistribution(
            ref.distributions.Normal(0.0, 1.0), ref.distributions.ExpTransform()
        )
        l = l_dist.log_prob(lucid.tensor(x.copy())).numpy()
        r = r_dist.log_prob(ref.tensor(x.copy())).detach().cpu().numpy()
        np.testing.assert_allclose(l, r, atol=1e-5)


# ── KL divergence registry ────────────────────────────────────────────────────


@pytest.mark.parity
class TestKLParity:
    def test_kl_normal(self, ref: Any) -> None:
        l = D.kl_divergence(D.Normal(0.0, 1.0), D.Normal(1.0, 2.0)).item()
        r = ref.distributions.kl_divergence(
            ref.distributions.Normal(0.0, 1.0), ref.distributions.Normal(1.0, 2.0)
        ).item()
        assert abs(l - r) < 1e-5

    def test_kl_categorical(self, ref: Any) -> None:
        p = [0.2, 0.5, 0.3]
        q = [0.3, 0.3, 0.4]
        l = D.kl_divergence(
            D.Categorical(probs=lucid.tensor(p)), D.Categorical(probs=lucid.tensor(q))
        ).item()
        r = ref.distributions.kl_divergence(
            ref.distributions.Categorical(probs=ref.tensor(p)),
            ref.distributions.Categorical(probs=ref.tensor(q)),
        ).item()
        assert abs(l - r) < 1e-5

    def test_kl_gamma(self, ref: Any) -> None:
        l = D.kl_divergence(D.Gamma(2.0, 1.0), D.Gamma(3.0, 2.0)).item()
        r = ref.distributions.kl_divergence(
            ref.distributions.Gamma(2.0, 1.0), ref.distributions.Gamma(3.0, 2.0)
        ).item()
        assert abs(l - r) < 1e-4

    def test_kl_beta(self, ref: Any) -> None:
        l = D.kl_divergence(D.Beta(2.0, 3.0), D.Beta(1.0, 4.0)).item()
        r = ref.distributions.kl_divergence(
            ref.distributions.Beta(2.0, 3.0), ref.distributions.Beta(1.0, 4.0)
        ).item()
        assert abs(l - r) < 1e-4

    def test_kl_exponential(self, ref: Any) -> None:
        l = D.kl_divergence(D.Exponential(2.0), D.Exponential(3.0)).item()
        r = ref.distributions.kl_divergence(
            ref.distributions.Exponential(2.0), ref.distributions.Exponential(3.0)
        ).item()
        assert abs(l - r) < 1e-5

    def test_kl_poisson(self, ref: Any) -> None:
        l = D.kl_divergence(D.Poisson(2.0), D.Poisson(4.0)).item()
        r = ref.distributions.kl_divergence(
            ref.distributions.Poisson(2.0), ref.distributions.Poisson(4.0)
        ).item()
        assert abs(l - r) < 1e-5

    def test_kl_dirichlet(self, ref: Any) -> None:
        a = lucid.tensor([1.0, 2.0, 3.0])
        b = lucid.tensor([2.0, 1.0, 2.0])
        l = D.kl_divergence(D.Dirichlet(a), D.Dirichlet(b)).item()
        r = ref.distributions.kl_divergence(
            ref.distributions.Dirichlet(ref.tensor([1.0, 2.0, 3.0])),
            ref.distributions.Dirichlet(ref.tensor([2.0, 1.0, 2.0])),
        ).item()
        assert abs(l - r) < 1e-4

    def test_kl_mvn(self, ref: Any) -> None:
        loc1 = np.array([0.0, 0.0], dtype=np.float32)
        cov1 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        loc2 = np.array([1.0, 0.5], dtype=np.float32)
        cov2 = np.array([[2.0, 0.3], [0.3, 1.5]], dtype=np.float32)
        l = D.kl_divergence(
            D.MultivariateNormal(lucid.tensor(loc1), lucid.tensor(cov1)),
            D.MultivariateNormal(lucid.tensor(loc2), lucid.tensor(cov2)),
        ).item()
        r = ref.distributions.kl_divergence(
            ref.distributions.MultivariateNormal(ref.tensor(loc1), ref.tensor(cov1)),
            ref.distributions.MultivariateNormal(ref.tensor(loc2), ref.tensor(cov2)),
        ).item()
        assert abs(l - r) < 1e-3
