"""Parity and contract tests for LCD-149 distribution features.

Covers:
1. ``Distribution.expand`` across all 38 concrete distribution classes,
   plus ``Distribution._get_checked_instance`` subclass enforcement.
2. ``Distribution.perplexity`` and module-level ``set_default_validate_args``.
3. Implicit-reparameterisation ``rsample`` gradients on ``Gamma``, ``Chi2``,
   ``Beta``, ``Dirichlet``, ``FisherSnedecor``, ``InverseGamma``, and
   ``StudentT`` (with respect to ``df``).
4. ``OneHotCategoricalStraightThrough`` straight-through gradients.
5. ``LowRankMultivariateNormal`` moments, density, sampling, and KL divergence.
6. ``LogisticNormal``, ``GeneralizedPareto``, and ``VonMises``.
7. ``AffineTransform(event_dim=...)``, ``CorrCholeskyTransform()``,
   ``SoftplusTransform``, ``PositiveDefiniteTransform``, ``StackTransform``,
   ``CatTransform``, ``biject_to``, and ``transform_to``.
"""

import math
from typing import Any, Callable

import numpy as np
import pytest

import lucid
import lucid.distributions as D
import lucid.distributions.transforms as T
from lucid.distributions import constraints
from lucid.distributions.gamma import _standard_gamma, _standard_gamma_grad

DEVICES = ["cpu", "metal"]


def _lt(val: Any, device: str = "cpu", dtype: Any = np.float32) -> lucid.Tensor:
    return lucid.tensor(np.asarray(val, dtype=dtype)).to(device)


def _rt(ref: Any, val: Any, dtype: Any = np.float32) -> Any:
    return ref.tensor(np.asarray(val, dtype=dtype))


def _np(x: Any) -> np.ndarray:
    return np.asarray(x.detach().cpu().numpy(), dtype=np.float64)


# ── 1. Distribution.expand across all 38 concrete distributions ───────────────

_EXPAND_BUILDERS: list[tuple[str, Callable[[Callable[[Any], Any]], Any]]] = [
    ("Bernoulli", lambda t: D.Bernoulli(probs=t(0.35))),
    ("Beta", lambda t: D.Beta(t(2.0), t(3.0))),
    ("Binomial", lambda t: D.Binomial(5, probs=t(0.4))),
    ("Categorical", lambda t: D.Categorical(probs=t([0.2, 0.3, 0.5]))),
    ("Cauchy", lambda t: D.Cauchy(t(0.5), t(1.2))),
    ("Chi2", lambda t: D.Chi2(t(4.0))),
    ("ContinuousBernoulli", lambda t: D.ContinuousBernoulli(probs=t(0.3))),
    ("Dirichlet", lambda t: D.Dirichlet(t([1.5, 2.5, 3.0]))),
    ("Exponential", lambda t: D.Exponential(t(1.5))),
    ("FisherSnedecor", lambda t: D.FisherSnedecor(t(5.0), t(7.0))),
    ("Gamma", lambda t: D.Gamma(t(2.5), t(1.5))),
    ("GeneralizedPareto", lambda t: D.GeneralizedPareto(t(0.5), t(1.5), t(0.2))),
    ("Geometric", lambda t: D.Geometric(probs=t(0.4))),
    ("Gumbel", lambda t: D.Gumbel(t(0.5), t(1.2))),
    ("HalfCauchy", lambda t: D.HalfCauchy(t(1.5))),
    ("HalfNormal", lambda t: D.HalfNormal(t(1.5))),
    (
        "Independent",
        lambda t: D.Independent(D.Normal(t([0.0, 1.0]), t([1.0, 2.0])), 1),
    ),
    ("InverseGamma", lambda t: D.InverseGamma(t(3.5), t(2.0))),
    ("Kumaraswamy", lambda t: D.Kumaraswamy(t(2.0), t(3.0))),
    ("LKJCholesky", lambda t: D.LKJCholesky(3, t(2.0))),
    ("Laplace", lambda t: D.Laplace(t(0.5), t(1.2))),
    ("LogNormal", lambda t: D.LogNormal(t(0.2), t(0.8))),
    ("LogisticNormal", lambda t: D.LogisticNormal(t([0.0, 0.5]), t([1.0, 0.8]))),
    (
        "LowRankMultivariateNormal",
        lambda t: D.LowRankMultivariateNormal(t([0.5, -0.5]), t([[1.0], [0.5]]), t([0.8, 1.2])),
    ),
    (
        "MixtureSameFamily",
        lambda t: D.MixtureSameFamily(
            D.Categorical(probs=t([0.4, 0.6])),
            D.Normal(t([-1.0, 1.0]), t([0.5, 1.5])),
        ),
    ),
    ("Multinomial", lambda t: D.Multinomial(4, probs=t([0.2, 0.3, 0.5]))),
    (
        "MultivariateNormal",
        lambda t: D.MultivariateNormal(t([0.5, -0.5]), scale_tril=t([[1.0, 0.0], [0.3, 1.2]])),
    ),
    ("NegativeBinomial", lambda t: D.NegativeBinomial(t(3.0), probs=t(0.4))),
    ("Normal", lambda t: D.Normal(t(0.5), t(1.2))),
    ("OneHotCategorical", lambda t: D.OneHotCategorical(probs=t([0.2, 0.3, 0.5]))),
    (
        "OneHotCategoricalStraightThrough",
        lambda t: D.OneHotCategoricalStraightThrough(probs=t([0.2, 0.3, 0.5])),
    ),
    ("Pareto", lambda t: D.Pareto(t(1.5), t(3.5))),
    ("Poisson", lambda t: D.Poisson(t(2.5))),
    ("RelaxedBernoulli", lambda t: D.RelaxedBernoulli(t(0.5), probs=t(0.35))),
    (
        "RelaxedOneHotCategorical",
        lambda t: D.RelaxedOneHotCategorical(t(0.5), probs=t([0.2, 0.3, 0.5])),
    ),
    ("StudentT", lambda t: D.StudentT(t(5.0), t(0.5), t(1.2))),
    (
        "TransformedDistribution",
        lambda t: D.TransformedDistribution(D.Normal(t(0.2), t(0.8)), [T.ExpTransform()]),
    ),
    ("Uniform", lambda t: D.Uniform(t(-1.0), t(2.0))),
    ("VonMises", lambda t: D.VonMises(t(0.3), t(2.5))),
    ("Weibull", lambda t: D.Weibull(t(1.5), t(2.0))),
    (
        "Wishart",
        lambda t: D.Wishart(t(5.0), scale_tril=t([[1.0, 0.0], [0.2, 1.1]])),
    ),
]


def test_expand_builders_cover_every_concrete_distribution() -> None:
    expected = sorted(
        n
        for n in D.__all__
        if isinstance(getattr(D, n), type)
        and issubclass(getattr(D, n), D.Distribution)
        and n not in ("Distribution", "ExponentialFamily")
    )
    assert sorted(name for name, _ in _EXPAND_BUILDERS) == expected


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name,build", _EXPAND_BUILDERS, ids=[n for n, _ in _EXPAND_BUILDERS])
def test_expand_preserves_event_shape_and_matches_unexpanded_scoring(
    name: str, build: Callable[[Callable[[Any], Any]], Any], device: str
) -> None:
    del name
    lucid.manual_seed(0)
    base = build(lambda v: _lt(v, device))
    target_batch = (2, 3)
    expanded = base.expand(target_batch)
    assert expanded.batch_shape == target_batch
    assert expanded.event_shape == base.event_shape
    assert expanded._validate_args == base._validate_args

    draws = expanded.sample((2,))
    assert draws.shape == (2,) + target_batch + base.event_shape
    lp_exp = _np(expanded.log_prob(draws))
    lp_base = _np(base.log_prob(draws))
    np.testing.assert_allclose(lp_exp, lp_base, rtol=1e-5, atol=1e-5)


def test_get_checked_instance_refuses_unimplemented_subclass_expand() -> None:
    class CustomNormal(D.Normal):
        def __init__(self, loc: float, scale: float) -> None:
            super().__init__(loc, scale)

    dist = CustomNormal(0.0, 1.0)
    with pytest.raises(NotImplementedError, match="CustomNormal"):
        dist.expand((2,))


def test_perplexity_and_set_default_validate_args(ref: Any) -> None:
    d_ours = D.Normal(_lt([0.0, 1.0]), _lt([1.0, 2.0]))
    d_ref = ref.distributions.Normal(_rt(ref, [0.0, 1.0]), _rt(ref, [1.0, 2.0]))
    np.testing.assert_allclose(_np(d_ours.perplexity()), _np(d_ref.perplexity()), rtol=1e-6)

    try:
        D.set_default_validate_args(False)
        d_unvalidated = D.Normal(0.0, -1.0)
        assert not d_unvalidated._validate_args
    finally:
        D.set_default_validate_args(True)
    with pytest.raises(ValueError):
        D.Normal(0.0, -1.0)


# ── 2. Implicit-reparameterisation rsample gradients ──────────────────────────


@pytest.mark.parametrize("device", DEVICES)
def test_standard_gamma_grad_matches_reference_across_regimes(device: str, ref: Any) -> None:
    dtype = np.float64 if device == "cpu" else np.float32
    alphas = [0.02, 0.15, 0.6, 1.0, 2.5, 7.5, 15.0, 50.0]
    xs = [0.01, 0.08, 0.4, 0.9, 2.2, 6.8, 14.2, 48.5]
    a_l = _lt(alphas, device, dtype)
    x_l = _lt(xs, device, dtype)
    a_r = _rt(ref, alphas, dtype)
    x_r = _rt(ref, xs, dtype)
    got = _np(_standard_gamma_grad(a_l, x_l))
    expected = _np(ref._standard_gamma_grad(a_r, x_r))
    rtol = 1e-5 if dtype == np.float64 else 2e-4
    np.testing.assert_allclose(got, expected, rtol=rtol, atol=1e-5)


def test_standard_gamma_supports_higher_order_autograd() -> None:
    alpha = lucid.tensor([1.5, 3.0], dtype=lucid.float64, requires_grad=True)
    lucid.manual_seed(42)
    sample = _standard_gamma(alpha)
    (g1,) = lucid.autograd.grad(sample.sum(), (alpha,), create_graph=True)
    assert g1.requires_grad
    (g2,) = lucid.autograd.grad(g1.sum(), (alpha,))
    assert bool(lucid.isfinite(g2).all().item())


@pytest.mark.parametrize("device", DEVICES)
def test_gamma_family_and_student_t_rsample_gradients(device: str) -> None:
    lucid.manual_seed(7)
    a = _lt(2.5, device).requires_grad_(True)
    b = _lt(1.5, device).requires_grad_(True)
    D.Gamma(a, b).rsample((64,)).mean().backward()
    assert a.grad is not None and b.grad is not None
    assert np.isfinite(_np(a.grad)) and np.isfinite(_np(b.grad))

    df = _lt(4.0, device).requires_grad_(True)
    D.Chi2(df).rsample((64,)).mean().backward()
    assert df.grad is not None and np.isfinite(_np(df.grad))

    c1 = _lt(2.0, device).requires_grad_(True)
    c0 = _lt(3.0, device).requires_grad_(True)
    D.Beta(c1, c0).rsample((64,)).mean().backward()
    assert c1.grad is not None and c0.grad is not None

    conc = _lt([1.5, 2.5, 3.5], device).requires_grad_(True)
    (D.Dirichlet(conc).rsample((64,)) * _lt([1.0, 2.0, 3.0], device)).sum().backward()
    assert conc.grad is not None and np.isfinite(_np(conc.grad)).all()

    df1 = _lt(5.0, device).requires_grad_(True)
    df2 = _lt(7.0, device).requires_grad_(True)
    D.FisherSnedecor(df1, df2).rsample((64,)).mean().backward()
    assert df1.grad is not None and df2.grad is not None

    ig_a = _lt(3.5, device).requires_grad_(True)
    ig_b = _lt(2.0, device).requires_grad_(True)
    D.InverseGamma(ig_a, ig_b).rsample((64,)).mean().backward()
    assert ig_a.grad is not None and ig_b.grad is not None

    st_df = _lt(4.5, device).requires_grad_(True)
    st_loc = _lt(0.5, device).requires_grad_(True)
    st_scale = _lt(1.2, device).requires_grad_(True)
    D.StudentT(st_df, st_loc, st_scale).rsample((64,)).pow(2.0).mean().backward()
    assert st_df.grad is not None and float(abs(_np(st_df.grad))) > 0.0

    w_df = _lt(5.0, device).requires_grad_(True)
    w_tril = _lt([[1.5, 0.0], [0.3, 1.2]], device).requires_grad_(True)
    D.Wishart(df=w_df, scale_tril=w_tril).rsample((16,)).mean().backward()
    assert w_df.grad is not None and w_tril.grad is not None
    assert np.isfinite(_np(w_df.grad)) and np.isfinite(_np(w_tril.grad)).all()


# ── 3. OneHotCategoricalStraightThrough ───────────────────────────────────────


@pytest.mark.parametrize("device", DEVICES)
def test_one_hot_categorical_straight_through(device: str, ref: Any) -> None:
    probs_l = _lt([0.2, 0.5, 0.3], device).requires_grad_(True)
    probs_r = _rt(ref, [0.2, 0.5, 0.3]).requires_grad_(True)
    dist = D.OneHotCategoricalStraightThrough(probs=probs_l)
    dist_ref_p = ref.distributions.OneHotCategoricalStraightThrough(probs=probs_r)
    assert dist.has_rsample
    sample = dist.rsample((4,))
    assert sample.shape == (4, 3)
    # Forward sample is hard one-hot.
    np.testing.assert_allclose(_np(sample.sum(dim=-1)), np.ones(4), atol=1e-6)
    weights = _lt([1.5, -0.5, 2.0], device)
    weights_r = _rt(ref, [1.5, -0.5, 2.0])
    (sample * weights).sum().backward()
    (dist_ref_p.rsample((4,)) * weights_r).sum().backward()
    assert probs_l.grad is not None
    np.testing.assert_allclose(_np(probs_l.grad), _np(probs_r.grad), rtol=1e-5)

    logits_l = _lt([0.5, -1.0, 1.5], device).requires_grad_(True)
    logits_r = _rt(ref, [0.5, -1.0, 1.5]).requires_grad_(True)
    dist_l = D.OneHotCategoricalStraightThrough(logits=logits_l)
    dist_r = ref.distributions.OneHotCategoricalStraightThrough(logits=logits_r)
    (dist_l.rsample() * weights).sum().backward()
    (dist_r.rsample() * weights_r).sum().backward()
    np.testing.assert_allclose(_np(logits_l.grad), _np(logits_r.grad), rtol=1e-5)


# ── 4. LowRankMultivariateNormal & KL divergence ──────────────────────────────


@pytest.mark.parametrize("device", DEVICES)
def test_low_rank_multivariate_normal_parity(device: str, ref: Any) -> None:
    loc = [0.5, -1.0, 0.25]
    cov_factor = [[0.8, -0.2], [0.1, 0.5], [-0.3, 0.4]]
    cov_diag = [0.7, 1.1, 0.9]
    d_l = D.LowRankMultivariateNormal(
        _lt(loc, device), _lt(cov_factor, device), _lt(cov_diag, device)
    )
    d_r = ref.distributions.LowRankMultivariateNormal(
        _rt(ref, loc), _rt(ref, cov_factor), _rt(ref, cov_diag)
    )
    for attr in (
        "mean",
        "mode",
        "variance",
        "scale_tril",
        "covariance_matrix",
        "precision_matrix",
    ):
        np.testing.assert_allclose(
            _np(getattr(d_l, attr)), _np(getattr(d_r, attr)), rtol=1e-5, atol=1e-5
        )
    np.testing.assert_allclose(_np(d_l.entropy()), _np(d_r.entropy()), rtol=1e-5)
    probe = [0.2, -0.4, 0.7]
    np.testing.assert_allclose(
        _np(d_l.log_prob(_lt(probe, device))),
        _np(d_r.log_prob(_rt(ref, probe))),
        rtol=1e-5,
    )
    draws = d_l.rsample((8,))
    assert draws.shape == (8, 3)


def test_low_rank_multivariate_normal_kl_divergence(ref: Any) -> None:
    p_l = D.LowRankMultivariateNormal(_lt([0.5, -0.5]), _lt([[0.6], [0.3]]), _lt([0.8, 1.2]))
    q_l = D.LowRankMultivariateNormal(_lt([-0.2, 0.4]), _lt([[0.4], [-0.5]]), _lt([1.1, 0.9]))
    m_l = D.MultivariateNormal(_lt([0.1, 0.2]), scale_tril=_lt([[1.1, 0.0], [0.25, 0.9]]))
    p_r = ref.distributions.LowRankMultivariateNormal(
        _rt(ref, [0.5, -0.5]), _rt(ref, [[0.6], [0.3]]), _rt(ref, [0.8, 1.2])
    )
    q_r = ref.distributions.LowRankMultivariateNormal(
        _rt(ref, [-0.2, 0.4]), _rt(ref, [[0.4], [-0.5]]), _rt(ref, [1.1, 0.9])
    )
    m_r = ref.distributions.MultivariateNormal(
        _rt(ref, [0.1, 0.2]), scale_tril=_rt(ref, [[1.1, 0.0], [0.25, 0.9]])
    )
    for a_l, b_l, a_r, b_r in [
        (p_l, q_l, p_r, q_r),
        (p_l, m_l, p_r, m_r),
        (m_l, p_l, m_r, p_r),
    ]:
        np.testing.assert_allclose(
            _np(D.kl_divergence(a_l, b_l)),
            _np(ref.distributions.kl_divergence(a_r, b_r)),
            rtol=1e-5,
            atol=1e-6,
        )


# ── 5. LogisticNormal, GeneralizedPareto, VonMises ────────────────────────────


@pytest.mark.parametrize("device", DEVICES)
def test_logistic_normal_generalized_pareto_and_von_mises(device: str, ref: Any) -> None:
    ln_l = D.LogisticNormal(_lt([0.2, -0.3], device), _lt([0.8, 1.1], device))
    ln_r = ref.distributions.LogisticNormal(_rt(ref, [0.2, -0.3]), _rt(ref, [0.8, 1.1]))
    y = [0.25, 0.35, 0.40]
    np.testing.assert_allclose(
        _np(ln_l.log_prob(_lt(y, device))), _np(ln_r.log_prob(_rt(ref, y))), rtol=1e-5
    )

    for conc in [0.25, 0.0, -0.25]:
        gp_l = D.GeneralizedPareto(_lt(1.0, device), _lt(2.0, device), _lt(conc, device))
        gp_r = ref.distributions.GeneralizedPareto(_rt(ref, 1.0), _rt(ref, 2.0), _rt(ref, conc))
        x = 1.8
        for actual, expected in (
            (gp_l.log_prob(_lt(x, device)), gp_r.log_prob(_rt(ref, x))),
            (gp_l.cdf(_lt(x, device)), gp_r.cdf(_rt(ref, x))),
            (gp_l.log_cdf(_lt(x, device)), gp_r.log_cdf(_rt(ref, x))),
            (
                gp_l.log_survival_function(_lt(x, device)),
                gp_r.log_survival_function(_rt(ref, x)),
            ),
        ):
            np.testing.assert_allclose(_np(actual), _np(expected), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(
            _np(gp_l.icdf(_lt(0.4, device))),
            _np(gp_r.icdf(_rt(ref, 0.4))),
            rtol=1e-4,
            atol=1e-5,
        )
        np.testing.assert_allclose(_np(gp_l.entropy()), _np(gp_r.entropy()), rtol=1e-5)

    vm_l = D.VonMises(_lt([0.4, -0.5], device), _lt([0.3, 12.0], device))
    vm_r = ref.distributions.VonMises(_rt(ref, [0.4, -0.5]), _rt(ref, [0.3, 12.0]))
    for attr in ("mean", "mode", "variance"):
        np.testing.assert_allclose(_np(getattr(vm_l, attr)), _np(getattr(vm_r, attr)), rtol=1e-5)
    np.testing.assert_allclose(
        _np(vm_l.log_prob(_lt([0.2, -0.4], device))),
        _np(vm_r.log_prob(_rt(ref, [0.2, -0.4]))),
        rtol=1e-5,
    )
    lucid.manual_seed(0)
    vm_draws = vm_l.sample((256,))
    assert vm_draws.shape == (256, 2)
    assert bool(((vm_draws >= -math.pi) & (vm_draws <= math.pi)).all().item())


# ── 6. Transforms & ConstraintRegistry (biject_to / transform_to) ─────────────


@pytest.mark.parametrize("device", DEVICES)
def test_new_transforms_and_constraint_registry(device: str, ref: Any) -> None:
    x = _lt([[0.3, -0.7, 1.1], [-0.4, 0.8, -0.2]], device)
    xr = _rt(ref, [[0.3, -0.7, 1.1], [-0.4, 0.8, -0.2]])

    # AffineTransform with event_dim=1
    af_l = T.AffineTransform(loc=_lt([1.0, 2.0, 3.0], device), scale=2.0, event_dim=1)
    assert af_l.sign == 1
    af_r = ref.distributions.transforms.AffineTransform(
        loc=_rt(ref, [1.0, 2.0, 3.0]), scale=2.0, event_dim=1
    )
    y_l, y_r = af_l(x), af_r(xr)
    np.testing.assert_allclose(_np(y_l), _np(y_r), rtol=1e-5)
    np.testing.assert_allclose(_np(af_l.inv(y_l)), _np(x), rtol=1e-5)
    np.testing.assert_allclose(
        _np(af_l.log_abs_det_jacobian(x, y_l)),
        _np(af_r.log_abs_det_jacobian(xr, y_r)),
        rtol=1e-5,
    )

    # SoftplusTransform
    sp_l = T.SoftplusTransform()
    sp_r = ref.distributions.transforms.SoftplusTransform()
    sy_l, sy_r = sp_l(x), sp_r(xr)
    np.testing.assert_allclose(_np(sy_l), _np(sy_r), rtol=1e-5)
    np.testing.assert_allclose(_np(sp_l.inv(sy_l)), _np(x), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        _np(sp_l.log_abs_det_jacobian(x, sy_l)),
        _np(sp_r.log_abs_det_jacobian(xr, sy_r)),
        rtol=1e-5,
    )

    # CorrCholeskyTransform() no-arg constructor (free vector of length D*(D-1)/2)
    cc_l = T.CorrCholeskyTransform()
    cc_r = ref.distributions.transforms.CorrCholeskyTransform()
    z_l = _lt([0.2, -0.4, 0.6], device)
    z_r = _rt(ref, [0.2, -0.4, 0.6])
    L_l, L_r = cc_l(z_l), cc_r(z_r)
    assert cc_l.forward_shape((3,)) == (3, 3)
    assert cc_l.inverse_shape((3, 3)) == (3,)
    np.testing.assert_allclose(_np(L_l), _np(L_r), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(_np(cc_l.inv(L_l)), _np(z_l), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        _np(cc_l.log_abs_det_jacobian(z_l, L_l)),
        _np(cc_r.log_abs_det_jacobian(z_r, L_r)),
        rtol=1e-5,
    )

    # PositiveDefiniteTransform (composes LowerCholeskyTransform with L @ L.mT)
    pd_l = T.PositiveDefiniteTransform()
    m_l = _lt([[0.2, 0.0], [0.4, -0.1]], device)
    P_l = pd_l(m_l)
    L_chol = T.LowerCholeskyTransform()(m_l)
    np.testing.assert_allclose(_np(P_l), _np(L_chol @ L_chol.mT), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(_np(pd_l.inv(P_l)), _np(m_l), rtol=1e-5, atol=1e-5)
    diag = L_chol.diagonal(dim1=-2, dim2=-1)
    expected_ladj = (
        T.LowerCholeskyTransform().log_abs_det_jacobian(m_l, L_chol)
        + 2.0 * diag[0].log()
        + diag[1].log()
        + 2.0 * math.log(2.0)
    )
    np.testing.assert_allclose(
        _np(pd_l.log_abs_det_jacobian(m_l, P_l)),
        _np(expected_ladj),
        rtol=1e-5,
    )


@pytest.mark.parametrize("device", DEVICES)
def test_biject_to_and_transform_to_cover_constraints(device: str) -> None:
    x1 = _lt([0.3, -0.7, 1.1], device)
    x_mat = _lt([[0.2, 0.0], [0.4, -0.1]], device)
    cases: list[tuple[constraints.Constraint, lucid.Tensor]] = [
        (constraints.real, x1),
        (constraints.real_vector, x1),
        (constraints.positive, x1),
        (constraints.nonnegative, x1),
        (constraints.greater_than(2.0), x1),
        (constraints.greater_than_eq(2.0), x1),
        (constraints.less_than(5.0), x1),
        (constraints.interval(-1.0, 3.0), x1),
        (constraints.half_open_interval(-1.0, 3.0), x1),
        (constraints.unit_interval, x1),
        (constraints.simplex, _lt([0.2, -0.5], device)),
        (constraints.lower_cholesky, x_mat),
        (constraints.corr_cholesky, _lt([0.2, -0.4, 0.6], device)),
        (constraints.positive_definite, x_mat),
        (constraints.independent(constraints.positive, 1), x1),
        (
            constraints.stack([constraints.positive, constraints.unit_interval], dim=0),
            _lt([[0.2, -0.3], [0.5, 0.1]], device),
        ),
        (
            constraints.cat([constraints.positive, constraints.unit_interval], dim=-1),
            _lt([0.2, -0.3], device),
        ),
    ]
    for constraint, inp in cases:
        t_bij = D.biject_to(constraint)
        y_bij = t_bij(inp)
        assert bool(constraint.check(y_bij).all().item())
        np.testing.assert_allclose(_np(t_bij.inv(y_bij)), _np(inp), rtol=1e-4, atol=1e-5)

        t_tr = D.transform_to(constraint)
        y_tr = t_tr(inp)
        assert bool(constraint.check(y_tr).all().item())
        np.testing.assert_allclose(_np(t_tr(t_tr.inv(y_tr))), _np(y_tr), rtol=1e-4, atol=1e-5)
