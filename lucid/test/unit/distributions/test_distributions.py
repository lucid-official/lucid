"""``lucid.distributions`` — full surface coverage.

Tests are CPU-only (the gamma sampler iteratively retries with engine
ops in a tight loop and is significantly heavier on Metal — pinning
to CPU keeps the test tier fast and deterministic).  Distribution
correctness is the same on either device.
"""

import math

import numpy as np
import pytest

import lucid
import lucid.distributions as D

# ── continuous univariate ───────────────────────────────────────────────


class TestNormal:
    def test_log_prob_at_zero(self) -> None:
        v = D.Normal(0.0, 1.0).log_prob(lucid.tensor(0.0)).item()
        assert abs(v - (-0.5 * math.log(2.0 * math.pi))) < 1e-5

    def test_entropy(self) -> None:
        v = D.Normal(0.0, 1.0).entropy().item()
        assert abs(v - (0.5 + 0.5 * math.log(2.0 * math.pi))) < 1e-5

    def test_rsample_grad_flows(self) -> None:
        loc = lucid.tensor([0.0], requires_grad=True)
        scale = lucid.tensor([1.0], requires_grad=True)
        s = D.Normal(loc, scale).rsample((4,))
        assert s.requires_grad

    def test_cdf_icdf_roundtrip(self) -> None:
        n = D.Normal(0.0, 1.0)
        u = lucid.tensor([0.25, 0.5, 0.75])
        np.testing.assert_allclose(n.cdf(n.icdf(u)).numpy(), u.numpy(), atol=1e-4)


class TestLogNormalUniform:
    def test_lognormal_positive(self) -> None:
        s = D.LogNormal(0.0, 1.0).sample((20,)).numpy()
        assert (s > 0).all()

    def test_uniform_in_range(self) -> None:
        s = D.Uniform(2.0, 5.0).sample((50,)).numpy()
        assert (s >= 2.0).all() and (s < 5.0).all()


class TestExponentialLaplaceCauchy:
    def test_exp_log_prob_at_one(self) -> None:
        v = D.Exponential(1.0).log_prob(lucid.tensor(1.0)).item()
        assert abs(v + 1.0) < 1e-5

    def test_laplace_at_loc(self) -> None:
        v = D.Laplace(0.0, 1.0).log_prob(lucid.tensor(0.0)).item()
        assert abs(v + math.log(2.0)) < 1e-5

    def test_cauchy_at_loc(self) -> None:
        v = D.Cauchy(0.0, 1.0).log_prob(lucid.tensor(0.0)).item()
        assert abs(v + math.log(math.pi)) < 1e-5


class TestStudentT:
    def test_log_prob(self) -> None:
        # ψ(0 | df=5, loc=0, scale=1) = lgamma(3) − lgamma(2.5) − 0.5·log(5π).
        expected = math.lgamma(3.0) - math.lgamma(2.5) - 0.5 * math.log(5.0 * math.pi)
        v = D.StudentT(5.0, 0.0, 1.0).log_prob(lucid.tensor(0.0)).item()
        assert abs(v - expected) < 1e-4


class TestParetoWeibullHalfFisher:
    def test_pareto_min(self) -> None:
        s = D.Pareto(1.5, 3.0).sample((20,)).numpy()
        assert (s >= 1.5).all()

    def test_weibull_nonneg(self) -> None:
        s = D.Weibull(2.0, 1.5).sample((20,)).numpy()
        assert (s >= 0).all()

    def test_halfnormal_nonneg(self) -> None:
        s = D.HalfNormal(1.0).sample((50,)).numpy()
        assert (s >= 0).all()

    def test_halfnormal_mean(self) -> None:
        # E[|X|] = σ·√(2/π).
        v = D.HalfNormal(1.0).mean.item()
        assert abs(v - math.sqrt(2.0 / math.pi)) < 1e-5

    def test_halfcauchy_nonneg(self) -> None:
        s = D.HalfCauchy(1.0).sample((20,)).numpy()
        assert (s >= 0).all()

    def test_fisher_mean(self) -> None:
        # μ = d2/(d2−2) for d2>2.
        v = D.FisherSnedecor(5.0, 10.0).mean.item()
        assert abs(v - 1.25) < 1e-5


# ── discrete ────────────────────────────────────────────────────────────


class TestBernoulliGeometric:
    def test_bern_binary(self) -> None:
        s = D.Bernoulli(probs=0.5).sample((100,)).numpy()
        assert set(s.tolist()) <= {0.0, 1.0}

    def test_bern_log_prob(self) -> None:
        v = D.Bernoulli(probs=0.3).log_prob(lucid.tensor(1.0)).item()
        assert abs(v - math.log(0.3)) < 1e-5


class TestCategoricalOneHot:
    def test_log_prob(self) -> None:
        c = D.Categorical(probs=lucid.tensor([0.2, 0.5, 0.3]))
        assert abs(c.log_prob(lucid.tensor(1)).item() - math.log(0.5)) < 1e-5

    def test_one_hot(self) -> None:
        s = D.OneHotCategorical(probs=lucid.tensor([0.2, 0.5, 0.3])).sample().numpy()
        assert s.sum() == 1.0

    # ``value`` and ``batch_shape`` broadcast against each other, and each
    # direction has a case that only it can express.  Fixing one by
    # forcing the other used to break whichever was not being tested.

    def test_scalar_index_against_a_batched_dist(self) -> None:
        # ``value`` is lower-rank than ``batch_shape``: one category
        # scored against every distribution in the batch.
        c = D.Categorical(probs=lucid.tensor([[0.5, 0.5], [0.2, 0.8], [0.1, 0.9]]))
        out = c.log_prob(lucid.tensor(1))
        assert out.shape == (3,)
        expected = [math.log(0.5), math.log(0.8), math.log(0.9)]
        assert np.allclose(out.numpy(), expected, atol=1e-5)

    def test_same_rank_value_against_a_batched_dist(self) -> None:
        c = D.Categorical(probs=lucid.tensor([[0.5, 0.5], [0.2, 0.8], [0.1, 0.9]]))
        out = c.log_prob(lucid.tensor([1, 0, 1]))
        assert out.shape == (3,)
        expected = [math.log(0.5), math.log(0.2), math.log(0.9)]
        assert np.allclose(out.numpy(), expected, atol=1e-5)

    def test_sample_shaped_value_against_an_unbatched_dist(self) -> None:
        # ``value`` is higher-rank: the draw carries a sample_shape in
        # front of an empty batch_shape.  A distribution has to be able
        # to score its own samples.
        c = D.Categorical(probs=lucid.tensor([0.3, 0.7]))
        assert c.log_prob(c.sample((5,))).shape == (5,)

    def test_value_and_batch_broadcast_together(self) -> None:
        # Neither side contains the other: (2, 1) value against a (3,)
        # batch gives (2, 3).
        c = D.Categorical(probs=lucid.tensor([[0.5, 0.5], [0.2, 0.8], [0.1, 0.9]]))
        out = c.log_prob(lucid.tensor([[1], [0]]))
        assert out.shape == (2, 3)
        expected = [
            [math.log(0.5), math.log(0.8), math.log(0.9)],
            [math.log(0.5), math.log(0.2), math.log(0.1)],
        ]
        assert np.allclose(out.numpy(), expected, atol=1e-5)


class TestPoissonBinomial:
    def test_poisson_log_prob(self) -> None:
        # log p(2 | 3) = 2·log(3) − 3 − log(2).
        expected = 2.0 * math.log(3.0) - 3.0 - math.log(2.0)
        v = D.Poisson(3.0).log_prob(lucid.tensor(2.0)).item()
        assert abs(v - expected) < 1e-4

    def test_binomial_mean(self) -> None:
        v = D.Binomial(10, probs=0.3).mean.item()
        assert abs(v - 3.0) < 1e-6

    def test_negbin_mean(self) -> None:
        v = D.NegativeBinomial(5.0, probs=0.5).mean.item()
        assert abs(v - 5.0) < 1e-6

    def test_negbin_sample_shape(self) -> None:
        # The Gamma draw was handed ``sample_shape`` on top of a rate that
        # already carried it, so ``sample((100,))`` came back (100, 100).
        scalar = D.NegativeBinomial(5.0, probs=0.4)
        assert scalar.sample((100,)).shape == (100,)
        assert scalar.sample().shape == ()
        batched = D.NegativeBinomial(5.0, probs=lucid.tensor([0.2, 0.4, 0.6]))
        assert batched.sample((4, 2)).shape == (4, 2, 3)

    def test_negbin_broadcasts_total_count(self) -> None:
        # The batch was sized from ``probs`` alone: ``batch_shape`` came
        # out ``()`` beside a three-element ``total_count``.
        d = D.NegativeBinomial(lucid.tensor([2.0, 5.0, 9.0]), probs=0.4)
        assert d.batch_shape == (3,)
        assert d.sample((4,)).shape == (4, 3)
        assert np.allclose(d.mean.numpy(), [2 / 1.5, 5 / 1.5, 9 / 1.5])

    def test_negbin_samples_are_floating(self) -> None:
        # Floating counts, as the reference framework returns them; they
        # came back int64.
        s = D.NegativeBinomial(5.0, probs=0.4).sample((16,)).numpy()
        assert s.dtype == np.float32
        assert np.array_equal(s, np.round(s)) and (s >= 0).all()

    def test_binomial_broadcasts_total_count(self) -> None:
        # The same defect, and here ``sample`` raised a shape mismatch.
        d = D.Binomial(lucid.tensor([2.0, 5.0, 9.0]), probs=0.4)
        assert d.batch_shape == (3,)
        s = d.sample((4,)).numpy()
        assert s.shape == (4, 3)
        assert (s <= np.array([2, 5, 9])).all()


# Each entry builds one family from ``probs=`` or ``logits=``.
_BINARY_FAMILIES = {
    "Bernoulli": lambda **kw: D.Bernoulli(**kw),
    "Geometric": lambda **kw: D.Geometric(**kw),
    "Binomial": lambda **kw: D.Binomial(10, **kw),
    "NegativeBinomial": lambda **kw: D.NegativeBinomial(3.0, **kw),
    "RelaxedBernoulli": lambda **kw: D.RelaxedBernoulli(0.5, **kw),
    "ContinuousBernoulli": lambda **kw: D.ContinuousBernoulli(**kw),
}
_CATEGORICAL_FAMILIES = {
    "Categorical": lambda **kw: D.Categorical(**kw),
    "OneHotCategorical": lambda **kw: D.OneHotCategorical(**kw),
    "RelaxedOneHotCategorical": lambda **kw: D.RelaxedOneHotCategorical(0.5, **kw),
    "Multinomial": lambda **kw: D.Multinomial(10, **kw),
}


class TestDualParameterisation:
    """``probs`` and ``logits`` both answer, whichever one was given.

    The reference framework derives the missing one on demand; here it
    raised AttributeError.
    """

    @pytest.mark.parametrize("name", list(_BINARY_FAMILIES))
    def test_binary_families(self, name: str) -> None:
        make = _BINARY_FAMILIES[name]
        p = np.array([0.2, 0.5, 0.9], dtype=np.float32)
        logit = np.log(p) - np.log1p(-p)
        from_probs = make(probs=lucid.tensor(p))
        from_logits = make(logits=lucid.tensor(logit))
        assert np.allclose(from_probs.logits.numpy(), logit, atol=1e-5)
        assert np.allclose(from_logits.probs.numpy(), p, atol=1e-6)

    @pytest.mark.parametrize("name", list(_CATEGORICAL_FAMILIES))
    def test_categorical_families(self, name: str) -> None:
        make = _CATEGORICAL_FAMILIES[name]
        p = np.array([0.2, 0.3, 0.5], dtype=np.float32)
        from_probs = make(probs=lucid.tensor(p))
        assert np.allclose(from_probs.logits.numpy(), np.log(p), atol=1e-6)
        # Unnormalised logits still derive normalised probabilities.
        from_logits = make(logits=lucid.tensor(np.log(p) + 3.0))
        assert np.allclose(from_logits.probs.numpy(), p, atol=1e-6)

    def test_derived_logits_of_a_degenerate_probability_are_finite(self) -> None:
        # Clamped one epsilon inside [0, 1] first, as the reference does.
        eps = float(np.finfo(np.float32).eps)
        edge = math.log(eps) - math.log1p(-eps)
        got = D.Bernoulli(probs=lucid.tensor([0.0, 1.0])).logits.numpy()
        assert np.allclose(got, [edge, -edge], rtol=1e-5)
        cat = D.Categorical(probs=lucid.tensor([0.0, 1.0])).logits.numpy()
        assert np.allclose(cat, [math.log(eps), math.log1p(-eps)], rtol=1e-5)

    def test_only_the_given_parameter_is_validated(self) -> None:
        # Converting at construction validated the converted value, and
        # ``sigmoid(20)`` rounds to 1.0 in float32 — outside Geometric's
        # open interval — so a perfectly good argument was refused.
        assert D.Geometric(logits=20.0).batch_shape == ()
        assert repr(D.Bernoulli(logits=0.0)) == "Bernoulli(logits=())"

    def test_derived_parameter_follows_the_stored_one(self) -> None:
        # Derived on every read, never cached: it sees an in-place update
        # of the stored parameter, and gradients reach that parameter.
        stored = lucid.tensor([0.0])
        d = D.Bernoulli(logits=stored)
        assert np.allclose(d.probs.numpy(), [0.5])
        stored.add_(100.0)
        assert np.allclose(d.probs.numpy(), [1.0])
        leaf = lucid.tensor([0.0], requires_grad=True)
        D.Bernoulli(logits=leaf).probs.sum().backward()
        assert leaf.grad is not None
        assert np.allclose(leaf.grad.numpy(), [0.25])


# ── multivariate / wrappers ─────────────────────────────────────────────


class TestMultivariateNormal:
    def test_log_prob_at_origin(self) -> None:
        mvn = D.MultivariateNormal(lucid.zeros(2), scale_tril=lucid.eye(2))
        v = mvn.log_prob(lucid.zeros(2)).item()
        assert abs(v - (-math.log(2.0 * math.pi))) < 1e-4

    def test_entropy_unit(self) -> None:
        mvn = D.MultivariateNormal(lucid.zeros(2), scale_tril=lucid.eye(2))
        assert abs(mvn.entropy().item() - (1.0 + math.log(2.0 * math.pi))) < 1e-4


class TestIndependent:
    def test_log_prob_summed(self) -> None:
        base = D.Normal(lucid.tensor([0.0, 0.0]), lucid.tensor([1.0, 1.0]))
        ind = D.Independent(base, 1)
        v = lucid.tensor([0.0, 0.0])
        expected = 2.0 * base.log_prob(v).numpy()[0]
        assert abs(ind.log_prob(v).item() - expected) < 1e-5


class TestMixtureSameFamily:
    def test_symmetric_mean_zero(self) -> None:
        m = D.MixtureSameFamily(
            D.Categorical(probs=lucid.tensor([0.5, 0.5])),
            D.Normal(lucid.tensor([-2.0, 2.0]), lucid.tensor([0.5, 0.5])),
        )
        assert abs(m.mean.item()) < 1e-6


class TestRelaxed:
    def test_relaxed_bern_in_unit(self) -> None:
        s = D.RelaxedBernoulli(temperature=0.5, probs=0.5).rsample((20,)).numpy()
        assert (s > 0).all() and (s < 1).all()

    def test_relaxed_ohc_simplex(self) -> None:
        s = D.RelaxedOneHotCategorical(0.5, probs=lucid.tensor([0.2, 0.5, 0.3])).rsample().numpy()
        assert abs(s.sum() - 1.0) < 1e-5


# ── transforms ──────────────────────────────────────────────────────────


class TestTransforms:
    def test_exp_round_trip(self) -> None:
        x = lucid.tensor(1.5)
        e = D.ExpTransform()
        np.testing.assert_allclose(e.inv(e(x)).numpy(), x.numpy(), atol=1e-5)

    def test_normal_through_exp_eq_lognormal(self) -> None:
        n = D.Normal(0.0, 1.0)
        td = D.TransformedDistribution(n, D.ExpTransform())
        ref = D.LogNormal(0.0, 1.0)
        for v in (0.5, 1.0, 2.0, 3.0):
            assert (
                abs(td.log_prob(lucid.tensor(v)).item() - ref.log_prob(lucid.tensor(v)).item())
                < 1e-4
            )


# ── KL pairs ────────────────────────────────────────────────────────────


class TestKL:
    def test_kl_normal_known(self) -> None:
        # KL(N(0,1) || N(1,2)) = log(2) + (1+1)/(2·4) − 0.5.
        kl = D.kl_divergence(D.Normal(0.0, 1.0), D.Normal(1.0, 2.0)).item()
        expected = math.log(2.0) + 0.25 - 0.5
        assert abs(kl - expected) < 1e-5

    def test_kl_self_zero_normal(self) -> None:
        kl = D.kl_divergence(D.Normal(0.0, 1.0), D.Normal(0.0, 1.0)).item()
        assert abs(kl) < 1e-5

    def test_kl_poisson_known(self) -> None:
        expected = 3.0 * math.log(3.0 / 4.0) + 1.0
        v = D.kl_divergence(D.Poisson(3.0), D.Poisson(4.0)).item()
        assert abs(v - expected) < 1e-4

    def test_kl_mvn_translation(self) -> None:
        kl = D.kl_divergence(
            D.MultivariateNormal(lucid.zeros(2), scale_tril=lucid.eye(2)),
            D.MultivariateNormal(lucid.tensor([1.0, 0.0]), scale_tril=lucid.eye(2)),
        ).item()
        assert abs(kl - 0.5) < 1e-4

    def test_kl_independent_empty_batch_is_empty(self) -> None:
        """The reshape this pair used to sum with raised "cannot infer -1"
        on a ``(0, 3)`` batch (LCD-471)."""
        p = D.Independent(D.Normal(lucid.zeros(0, 3), lucid.ones(0, 3)), 1)
        q = D.Independent(D.Normal(lucid.zeros(0, 3), lucid.ones(0, 3)), 1)
        assert tuple(D.kl_divergence(p, q).shape) == (0,)
