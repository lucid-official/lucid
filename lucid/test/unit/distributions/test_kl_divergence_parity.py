"""Comprehensive parity and unit tests for ``lucid.distributions.kl``."""

from collections.abc import Callable
from typing import Any, override

import numpy as np
import pytest

import lucid
import lucid.distributions as D
from lucid.distributions.kl import (
    _KL_MEMOIZE,
    _KL_REGISTRY,
    _dispatch_kl,
    kl_divergence,
    register_kl,
)

_PLACES = [("cpu", np.float32), ("cpu", np.float64), ("metal", np.float32)]
_PLACE_IDS = ["cpu-f32", "cpu-f64", "metal-f32"]

_Builder = Callable[[Any, Callable[[Any], Any]], Any]

# Distinct P and Q builders so KL(P || Q) > 0 for every same-family pair as well.
_DIST_PAIRS: dict[str, tuple[_Builder, _Builder]] = {
    "Bernoulli": (
        lambda M, t: M.Bernoulli(probs=t([0.2, 0.7])),
        lambda M, t: M.Bernoulli(probs=t([0.6, 0.3])),
    ),
    "Beta": (
        lambda M, t: M.Beta(t([2.0, 1.5]), t([3.0, 2.5])),
        lambda M, t: M.Beta(t([1.2, 3.0]), t([2.0, 1.5])),
    ),
    "Binomial": (
        lambda M, t: M.Binomial(5, probs=t([0.3, 0.6])),
        lambda M, t: M.Binomial(5, probs=t([0.5, 0.2])),
    ),
    "Categorical": (
        lambda M, t: M.Categorical(probs=t([[0.2, 0.3, 0.5], [0.5, 0.2, 0.3]])),
        lambda M, t: M.Categorical(probs=t([[0.3, 0.4, 0.3], [0.2, 0.5, 0.3]])),
    ),
    "Cauchy": (
        lambda M, t: M.Cauchy(t([0.0, 1.0]), t([1.0, 2.0])),
        lambda M, t: M.Cauchy(t([0.5, -0.5]), t([1.5, 1.0])),
    ),
    "Chi2": (
        lambda M, t: M.Chi2(t([3.0, 5.0])),
        lambda M, t: M.Chi2(t([4.0, 2.0])),
    ),
    "ContinuousBernoulli": (
        lambda M, t: M.ContinuousBernoulli(probs=t([0.25, 0.75])),
        lambda M, t: M.ContinuousBernoulli(probs=t([0.6, 0.4])),
    ),
    "Dirichlet": (
        lambda M, t: M.Dirichlet(t([[1.5, 2.0, 3.0], [2.0, 1.0, 2.5]])),
        lambda M, t: M.Dirichlet(t([[2.0, 1.5, 2.0], [1.2, 2.5, 1.8]])),
    ),
    "Exponential": (
        lambda M, t: M.Exponential(t([1.5, 2.5])),
        lambda M, t: M.Exponential(t([0.8, 1.2])),
    ),
    "FisherSnedecor": (
        lambda M, t: M.FisherSnedecor(t(3.0), t(5.0)),
        lambda M, t: M.FisherSnedecor(t(4.0), t(6.0)),
    ),
    "Gamma": (
        lambda M, t: M.Gamma(t([2.0, 3.5]), t([1.5, 2.0])),
        lambda M, t: M.Gamma(t([1.5, 2.5]), t([1.0, 1.5])),
    ),
    "Geometric": (
        lambda M, t: M.Geometric(probs=t([0.3, 0.7])),
        lambda M, t: M.Geometric(probs=t([0.6, 0.4])),
    ),
    "Gumbel": (
        lambda M, t: M.Gumbel(t([0.5, -0.5]), t([1.2, 1.8])),
        lambda M, t: M.Gumbel(t([0.0, 1.0]), t([1.5, 1.0])),
    ),
    "HalfCauchy": (
        lambda M, t: M.HalfCauchy(t([1.0, 2.0])),
        lambda M, t: M.HalfCauchy(t([1.5, 0.8])),
    ),
    "HalfNormal": (
        lambda M, t: M.HalfNormal(t([1.0, 2.0])),
        lambda M, t: M.HalfNormal(t([1.5, 0.8])),
    ),
    "Independent": (
        lambda M, t: M.Independent(M.Normal(t([0.0, 1.0]), t([1.0, 2.0])), 1),
        lambda M, t: M.Independent(M.Normal(t([0.5, -0.5]), t([1.5, 1.0])), 1),
    ),
    "InverseGamma": (
        lambda M, t: M.InverseGamma(t([2.5, 3.5]), t([1.5, 2.0])),
        lambda M, t: M.InverseGamma(t([3.0, 2.0]), t([1.0, 2.5])),
    ),
    "Kumaraswamy": (
        lambda M, t: M.Kumaraswamy(t(2.0), t(3.0)),
        lambda M, t: M.Kumaraswamy(t(1.5), t(2.5)),
    ),
    "LKJCholesky": (
        lambda M, t: M.LKJCholesky(2, t(1.5)),
        lambda M, t: M.LKJCholesky(2, t(2.5)),
    ),
    "Laplace": (
        lambda M, t: M.Laplace(t([0.5, -0.5]), t([1.2, 1.8])),
        lambda M, t: M.Laplace(t([0.0, 1.0]), t([1.5, 1.0])),
    ),
    "LogNormal": (
        lambda M, t: M.LogNormal(t([0.5, -0.5]), t([1.2, 0.8])),
        lambda M, t: M.LogNormal(t([0.0, 0.5]), t([1.5, 1.2])),
    ),
    "LowRankMultivariateNormal": (
        lambda M, t: M.LowRankMultivariateNormal(t([0.0, 1.0]), t([[1.0], [0.5]]), t([1.0, 2.0])),
        lambda M, t: M.LowRankMultivariateNormal(t([0.5, -0.5]), t([[0.8], [-0.3]]), t([1.5, 1.0])),
    ),
    "MixtureSameFamily": (
        lambda M, t: M.MixtureSameFamily(
            M.Categorical(probs=t([0.3, 0.7])),
            M.Normal(t([0.0, 1.0]), t([1.0, 2.0])),
        ),
        lambda M, t: M.MixtureSameFamily(
            M.Categorical(probs=t([0.6, 0.4])),
            M.Normal(t([0.5, -0.5]), t([1.5, 1.0])),
        ),
    ),
    "Multinomial": (
        lambda M, t: M.Multinomial(5, probs=t([0.2, 0.3, 0.5])),
        lambda M, t: M.Multinomial(5, probs=t([0.4, 0.4, 0.2])),
    ),
    "MultivariateNormal": (
        lambda M, t: M.MultivariateNormal(
            t([0.0, 1.0]), covariance_matrix=t([[2.0, 0.3], [0.3, 1.0]])
        ),
        lambda M, t: M.MultivariateNormal(
            t([0.5, -0.5]), covariance_matrix=t([[1.5, -0.2], [-0.2, 2.0]])
        ),
    ),
    "NegativeBinomial": (
        lambda M, t: M.NegativeBinomial(t(3.0), probs=t(0.4)),
        lambda M, t: M.NegativeBinomial(t(4.0), probs=t(0.6)),
    ),
    "Normal": (
        lambda M, t: M.Normal(t([0.5, -0.5]), t([1.2, 0.8])),
        lambda M, t: M.Normal(t([0.0, 1.0]), t([1.5, 1.2])),
    ),
    "OneHotCategorical": (
        lambda M, t: M.OneHotCategorical(probs=t([[0.2, 0.3, 0.5], [0.5, 0.2, 0.3]])),
        lambda M, t: M.OneHotCategorical(probs=t([[0.3, 0.4, 0.3], [0.2, 0.5, 0.3]])),
    ),
    "Pareto": (
        lambda M, t: M.Pareto(t([1.5, 2.0]), t([2.5, 3.5])),
        lambda M, t: M.Pareto(t([1.0, 1.5]), t([2.0, 3.0])),
    ),
    "Poisson": (
        lambda M, t: M.Poisson(t([1.5, 3.0])),
        lambda M, t: M.Poisson(t([2.5, 1.2])),
    ),
    "RelaxedBernoulli": (
        lambda M, t: M.RelaxedBernoulli(t(0.5), probs=t(0.3)),
        lambda M, t: M.RelaxedBernoulli(t(0.5), probs=t(0.7)),
    ),
    "RelaxedOneHotCategorical": (
        lambda M, t: M.RelaxedOneHotCategorical(t(0.5), probs=t([0.2, 0.3, 0.5])),
        lambda M, t: M.RelaxedOneHotCategorical(t(0.5), probs=t([0.4, 0.4, 0.2])),
    ),
    "StudentT": (
        lambda M, t: M.StudentT(t(3.0), t(0.0), t(1.0)),
        lambda M, t: M.StudentT(t(5.0), t(1.0), t(2.0)),
    ),
    "TransformedDistribution": (
        lambda M, t: M.TransformedDistribution(
            M.Normal(t([0.0, 0.5]), t([1.0, 1.5])), [M.transforms.ExpTransform()]
        ),
        lambda M, t: M.TransformedDistribution(
            M.Normal(t([0.5, -0.5]), t([1.5, 1.0])), [M.transforms.ExpTransform()]
        ),
    ),
    "Uniform": (
        lambda M, t: M.Uniform(t([0.2, 1.2]), t([0.8, 2.5])),
        lambda M, t: M.Uniform(t([0.0, 1.0]), t([1.0, 3.0])),
    ),
    "Weibull": (
        lambda M, t: M.Weibull(t(1.0), t(1.5)),
        lambda M, t: M.Weibull(t(2.0), t(2.5)),
    ),
    "Wishart": (
        lambda M, t: M.Wishart(t(5.0), covariance_matrix=t([[2.0, 0.3], [0.3, 1.0]])),
        lambda M, t: M.Wishart(t(7.0), covariance_matrix=t([[1.5, -0.2], [-0.2, 2.0]])),
    ),
}


def _lucid_at(device: str, dtype: Any) -> Callable[[Any], lucid.Tensor]:
    return lambda v: lucid.tensor(np.asarray(v, dtype=dtype)).to(device)


def _ref_at(ref: Any, dtype: Any) -> Callable[[Any], Any]:
    return lambda v: ref.tensor(np.asarray(v, dtype=dtype))


def test_registry_covers_all_reference_pairs(ref: Any) -> None:
    """Every (P, Q) pair in the reference's ``distributions.kl._KL_REGISTRY``
    must resolve via _dispatch_kl in Lucid."""
    missing = []
    for p_ref, q_ref in ref.distributions.kl._KL_REGISTRY:
        pn, qn = p_ref.__name__, q_ref.__name__
        p_cls = getattr(D, pn)
        q_cls = getattr(D, qn)
        if _dispatch_kl(p_cls, q_cls) is None:
            missing.append((pn, qn))
    assert not missing, f"Missing KL registrations in Lucid: {missing}"


@pytest.mark.parametrize("place", _PLACES, ids=_PLACE_IDS)
def test_all_distribution_pairs_match_reference(place: Any, ref: Any) -> None:
    """Across the entire 36x36 matrix of concrete distributions, Lucid must
    implement KL iff the reference does, and match its values on CPU and Metal."""
    device, dtype = place
    lt = _lucid_at(device, dtype)
    rt = _ref_at(ref, dtype)
    rtol, atol = (1e-6, 1e-8) if dtype == np.float64 else (1e-4, 1e-5)

    names = sorted(_DIST_PAIRS)
    for pn in names:
        p_l = _DIST_PAIRS[pn][0](D, lt)
        p_r = _DIST_PAIRS[pn][0](ref.distributions, rt)
        for qn in names:
            q_l = _DIST_PAIRS[qn][1](D, lt)
            q_r = _DIST_PAIRS[qn][1](ref.distributions, rt)
            try:
                theirs = ref.distributions.kl_divergence(p_r, q_r)
                t_ok = True
            except NotImplementedError:
                t_ok = False
            if not t_ok:
                with pytest.raises(NotImplementedError):
                    kl_divergence(p_l, q_l)
                continue
            ours = kl_divergence(p_l, q_l)
            assert ours.device.type == device, f"({pn}, {qn}) device mismatch"
            ours_np = np.asarray(ours.detach().numpy(), dtype=np.float64)
            theirs_np = np.asarray(theirs.detach().numpy(), dtype=np.float64)
            assert ours_np.shape == theirs_np.shape, f"({pn}, {qn}) shape mismatch"
            assert np.allclose(
                ours_np, theirs_np, rtol=rtol, atol=atol, equal_nan=True
            ), f"({pn}, {qn}) value mismatch on {device}/{dtype}: {ours_np} vs {theirs_np}"


@pytest.mark.parametrize("place", _PLACES, ids=_PLACE_IDS)
def test_piecewise_support_boundaries_match_reference(place: Any, ref: Any) -> None:
    """Piecewise +inf masks for Binomial, Pareto, Beta, ContinuousBernoulli,
    and Uniform cross pairs match the reference."""
    device, dtype = place
    lt = _lucid_at(device, dtype)
    rt = _ref_at(ref, dtype)
    cases: list[tuple[str, Callable[[Any, Callable[[Any], Any]], tuple[Any, Any]]]] = [
        (
            "Binomial.gt",
            lambda M, t: (
                M.Binomial(t([5.0, 5.0]), probs=t([0.3, 0.6])),
                M.Binomial(t([5.0, 3.0]), probs=t([0.5, 0.4])),
            ),
        ),
        (
            "Pareto.Pareto.mask",
            lambda M, t: (
                M.Pareto(t([0.8, 2.0]), t([2.5, 3.0])),
                M.Pareto(t([1.0, 1.5]), t([2.0, 2.5])),
            ),
        ),
        (
            "Pareto.Exponential.alpha_le_1",
            lambda M, t: (
                M.Pareto(t([1.0, 1.0]), t([0.8, 2.0])),
                M.Exponential(t([1.5, 1.5])),
            ),
        ),
        (
            "Pareto.Gamma.alpha_le_1",
            lambda M, t: (
                M.Pareto(t([1.0, 1.0]), t([1.0, 2.5])),
                M.Gamma(t([2.0, 2.0]), t([1.5, 1.5])),
            ),
        ),
        (
            "Pareto.Normal.alpha_le_2",
            lambda M, t: (
                M.Pareto(t([1.0, 1.0]), t([1.8, 3.0])),
                M.Normal(t([0.0, 0.0]), t([1.0, 1.0])),
            ),
        ),
        (
            "Beta.Uniform.mask",
            lambda M, t: (
                M.Beta(t([2.0, 2.0]), t([3.0, 3.0])),
                M.Uniform(t([-0.5, 0.2]), t([1.5, 0.8])),
            ),
        ),
        (
            "ContinuousBernoulli.Uniform.mask",
            lambda M, t: (
                M.ContinuousBernoulli(probs=t([0.3, 0.3])),
                M.Uniform(t([-0.1, 0.0]), t([1.1, 1.0])),
            ),
        ),
        (
            "Uniform.Beta.mask",
            lambda M, t: (
                M.Uniform(t([0.1, -0.2]), t([0.9, 0.8])),
                M.Beta(t([2.0, 2.0]), t([3.0, 3.0])),
            ),
        ),
        (
            "Uniform.ContinuousBernoulli.mask",
            lambda M, t: (
                M.Uniform(t([0.1, 0.0]), t([0.9, 1.0])),
                M.ContinuousBernoulli(probs=t([0.4, 0.4])),
            ),
        ),
        (
            "Uniform.Exponential.mask",
            lambda M, t: (
                M.Uniform(t([0.1, -0.5]), t([1.0, 1.0])),
                M.Exponential(t([1.5, 1.5])),
            ),
        ),
        (
            "Uniform.Gamma.mask",
            lambda M, t: (
                M.Uniform(t([0.2, -0.2]), t([1.5, 1.5])),
                M.Gamma(t([2.0, 2.0]), t([1.5, 1.5])),
            ),
        ),
        (
            "Uniform.Pareto.mask",
            lambda M, t: (
                M.Uniform(t([1.5, 0.5]), t([3.0, 3.0])),
                M.Pareto(t([1.0, 1.0]), t([2.0, 2.0])),
            ),
        ),
    ]
    for label, build in cases:
        p_l, q_l = build(D, lt)
        p_r, q_r = build(ref.distributions, rt)
        ours = np.asarray(kl_divergence(p_l, q_l).detach().numpy(), dtype=np.float64)
        theirs = np.asarray(
            ref.distributions.kl_divergence(p_r, q_r).detach().numpy(),
            dtype=np.float64,
        )
        assert np.allclose(
            ours, theirs, rtol=1e-4, atol=1e-5, equal_nan=True
        ), f"{label} mismatch: {ours} vs {theirs}"

    # Binomial with p.total_count < q.total_count raises NotImplementedError.
    with pytest.raises(NotImplementedError):
        kl_divergence(
            D.Binomial(3, probs=lt(0.4)),
            D.Binomial(5, probs=lt(0.4)),
        )


def test_dispatch_specificity_and_ambiguity_warning() -> None:
    """_Match orders pairs by subclass specificity, warns on ambiguous matches,
    and register_kl clears _KL_MEMOIZE and validates types."""
    with pytest.raises(TypeError):
        register_kl(object, D.Normal)
    with pytest.raises(TypeError):
        register_kl(D.Normal, object)

    class _BaseP(D.Distribution):
        def __init__(self) -> None:
            super().__init__()

    class _DerivedP(_BaseP):
        pass

    class _BaseQ(D.Distribution):
        def __init__(self) -> None:
            super().__init__()

    class _DerivedQ(_BaseQ):
        pass

    try:

        @register_kl(_BaseP, _DerivedQ)
        def _kl_v1(p: _BaseP, q: _DerivedQ) -> lucid.Tensor:
            del p, q
            return lucid.tensor(1.0)

        @register_kl(_DerivedP, _BaseQ)
        def _kl_v2(p: _DerivedP, q: _BaseQ) -> lucid.Tensor:
            del p, q
            return lucid.tensor(2.0)

        with pytest.warns(RuntimeWarning, match="Ambiguous kl_divergence"):
            res = kl_divergence(_DerivedP(), _DerivedQ())
        assert float(res.item()) in (1.0, 2.0)

        # Registering the more specific (_DerivedP, _DerivedQ) clears memoization
        # and resolves the ambiguity without warning.
        register_kl(_DerivedP, _DerivedQ)(_kl_v2)
        assert (_DerivedP, _DerivedQ) not in _KL_MEMOIZE
        assert float(kl_divergence(_DerivedP(), _DerivedQ()).item()) == 2.0
    finally:
        for pair in [
            (_BaseP, _DerivedQ),
            (_DerivedP, _BaseQ),
            (_DerivedP, _DerivedQ),
        ]:
            _KL_REGISTRY.pop(pair, None)
        _KL_MEMOIZE.clear()


def test_expfamily_bregman_kl(ref: Any) -> None:
    """_kl_expfamily_expfamily computes the exact Bregman divergence for a custom
    ExponentialFamily subclass and refuses cross-family pairs."""

    class _CustomExpNormal(D.ExponentialFamily):
        def __init__(self, loc: lucid.Tensor, scale: lucid.Tensor) -> None:
            self.loc = loc
            self.scale = scale
            super().__init__(batch_shape=tuple(loc.shape), event_shape=())

        @property
        @override
        def _natural_params(self) -> tuple[lucid.Tensor, ...]:
            return (self.loc / (self.scale**2), -0.5 / (self.scale**2))

        @override
        def _log_normalizer(self, *natural_params: lucid.Tensor) -> lucid.Tensor:
            x, y = natural_params
            return -0.25 * (x**2) / y + 0.5 * (-0.5 / y).log()

    class _OtherExpFamily(D.ExponentialFamily):
        def __init__(self) -> None:
            super().__init__()

    p = _CustomExpNormal(lucid.tensor([0.5, -1.0]), lucid.tensor([1.2, 0.8]))
    q = _CustomExpNormal(lucid.tensor([0.0, 0.5]), lucid.tensor([1.5, 1.1]))
    got = kl_divergence(p, q).detach().numpy()
    want = (
        ref.distributions.kl_divergence(
            ref.distributions.Normal(ref.tensor([0.5, -1.0]), ref.tensor([1.2, 0.8])),
            ref.distributions.Normal(ref.tensor([0.0, 0.5]), ref.tensor([1.5, 1.1])),
        )
        .detach()
        .numpy()
    )
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)

    with pytest.raises(NotImplementedError):
        kl_divergence(p, _OtherExpFamily())


@pytest.mark.parametrize("device", ["cpu", "metal"])
def test_batched_kl_matches_unbatched_across_expfamily_and_wishart(
    device: str,
) -> None:
    """LCD-482: batched KL for Wishart and every registered ExponentialFamily pair
    (including mixed-event-rank natural parameters) agrees element-by-element with
    its unbatched scalar oracle."""
    t = _lucid_at(device, np.float32)

    # 1. Wishart: natural parameters have event rank 2 (precision_matrix) and 0 (df).
    cov_p = [[2.0, 0.3], [0.3, 1.0]]
    cov_q = [[1.5, -0.2], [-0.2, 2.0]]
    w_p0 = D.Wishart(t(4.0), covariance_matrix=t(cov_p))
    w_q0 = D.Wishart(t(5.0), covariance_matrix=t(cov_q))
    scalar_wishart = float(kl_divergence(w_p0, w_q0).to("cpu").item())

    for batch_shape in [(3,), (2, 3)]:
        df_p = lucid.full(batch_shape, 4.0, device=device)
        df_q = lucid.full(batch_shape, 5.0, device=device)
        mat_p = t(cov_p).expand(*batch_shape, 2, 2).contiguous()
        mat_q = t(cov_q).expand(*batch_shape, 2, 2).contiguous()
        got = kl_divergence(
            D.Wishart(df_p, covariance_matrix=mat_p),
            D.Wishart(df_q, covariance_matrix=mat_q),
        )
        assert got.device.type == device
        assert tuple(got.shape) == batch_shape
        np.testing.assert_allclose(
            got.to("cpu").numpy(),
            np.full(batch_shape, scalar_wishart, dtype=np.float32),
            rtol=1e-5,
            atol=1e-6,
        )

    # 2. _kl_expfamily_expfamily on a custom ExponentialFamily whose natural
    #    parameters have differing event ranks (rank 2 and rank 0).
    class _MixedRankExpFamily(D.ExponentialFamily):
        def __init__(self, mat: lucid.Tensor, nu: lucid.Tensor) -> None:
            self.mat = mat
            self.nu = nu
            super().__init__(batch_shape=tuple(nu.shape), event_shape=(2, 2))

        @property
        @override
        def _natural_params(self) -> tuple[lucid.Tensor, ...]:
            return (self.mat, self.nu)

        @override
        def _log_normalizer(self, *natural_params: lucid.Tensor) -> lucid.Tensor:
            x, y = natural_params
            return 0.5 * (x**2).sum(dim=(-2, -1)) + 0.25 * (y**2)

    m_p0 = _MixedRankExpFamily(t(cov_p), t(4.0))
    m_q0 = _MixedRankExpFamily(t(cov_q), t(5.0))
    scalar_mixed = float(kl_divergence(m_p0, m_q0).to("cpu").item())
    m_pb = _MixedRankExpFamily(
        t(cov_p).expand(3, 2, 2).contiguous(),
        lucid.full((3,), 4.0, device=device),
    )
    m_qb = _MixedRankExpFamily(
        t(cov_q).expand(3, 2, 2).contiguous(),
        lucid.full((3,), 5.0, device=device),
    )
    got_mixed = kl_divergence(m_pb, m_qb)
    assert tuple(got_mixed.shape) == (3,)
    np.testing.assert_allclose(
        got_mixed.to("cpu").numpy(),
        np.full((3,), scalar_mixed, dtype=np.float32),
        rtol=1e-5,
        atol=1e-6,
    )

    # 3. Every concrete ExponentialFamily subclass in Lucid: batched KL equals
    #    the per-slice unbatched KL.
    exp_cases: list[
        tuple[D.Distribution, D.Distribution, list[tuple[D.Distribution, D.Distribution]]]
    ] = [
        (
            D.Bernoulli(probs=t([0.2, 0.7])),
            D.Bernoulli(probs=t([0.6, 0.3])),
            [
                (D.Bernoulli(probs=t(0.2)), D.Bernoulli(probs=t(0.6))),
                (D.Bernoulli(probs=t(0.7)), D.Bernoulli(probs=t(0.3))),
            ],
        ),
        (
            D.Beta(t([2.0, 1.5]), t([3.0, 2.5])),
            D.Beta(t([1.2, 3.0]), t([2.0, 1.5])),
            [
                (D.Beta(t(2.0), t(3.0)), D.Beta(t(1.2), t(2.0))),
                (D.Beta(t(1.5), t(2.5)), D.Beta(t(3.0), t(1.5))),
            ],
        ),
        (
            D.Dirichlet(t([[1.5, 2.0, 3.0], [2.0, 1.0, 2.5]])),
            D.Dirichlet(t([[2.0, 1.5, 2.0], [1.2, 2.5, 1.8]])),
            [
                (D.Dirichlet(t([1.5, 2.0, 3.0])), D.Dirichlet(t([2.0, 1.5, 2.0]))),
                (D.Dirichlet(t([2.0, 1.0, 2.5])), D.Dirichlet(t([1.2, 2.5, 1.8]))),
            ],
        ),
        (
            D.Exponential(t([1.5, 2.5])),
            D.Exponential(t([0.8, 1.2])),
            [
                (D.Exponential(t(1.5)), D.Exponential(t(0.8))),
                (D.Exponential(t(2.5)), D.Exponential(t(1.2))),
            ],
        ),
        (
            D.Gamma(t([2.0, 3.5]), t([1.5, 2.0])),
            D.Gamma(t([1.5, 2.5]), t([1.0, 1.5])),
            [
                (D.Gamma(t(2.0), t(1.5)), D.Gamma(t(1.5), t(1.0))),
                (D.Gamma(t(3.5), t(2.0)), D.Gamma(t(2.5), t(1.5))),
            ],
        ),
        (
            D.Normal(t([0.5, -0.5]), t([1.2, 0.8])),
            D.Normal(t([0.0, 1.0]), t([1.5, 1.2])),
            [
                (D.Normal(t(0.5), t(1.2)), D.Normal(t(0.0), t(1.5))),
                (D.Normal(t(-0.5), t(0.8)), D.Normal(t(1.0), t(1.2))),
            ],
        ),
        (
            D.Poisson(t([1.5, 3.0])),
            D.Poisson(t([2.5, 1.2])),
            [
                (D.Poisson(t(1.5)), D.Poisson(t(2.5))),
                (D.Poisson(t(3.0)), D.Poisson(t(1.2))),
            ],
        ),
    ]
    for p_batched, q_batched, slices in exp_cases:
        got_b = kl_divergence(p_batched, q_batched).to("cpu").numpy()
        want_b = np.array(
            [float(kl_divergence(ps, qs).to("cpu").item()) for ps, qs in slices],
            dtype=np.float32,
        )
        np.testing.assert_allclose(got_b, want_b, rtol=1e-5, atol=1e-6)
