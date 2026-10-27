"""Cross-family and composite ``kl_divergence`` registrations."""

import math

import lucid
from lucid._tensor.tensor import Tensor
from lucid.distributions._util import _sum_rightmost
from lucid.distributions.bernoulli import Bernoulli, Geometric
from lucid.distributions.categorical import OneHotCategorical
from lucid.distributions.continuous_extra import HalfCauchy, Pareto, Weibull
from lucid.distributions.discrete import Binomial, Poisson
from lucid.distributions.distribution import Distribution, ExponentialFamily
from lucid.distributions.exponential import Exponential, Laplace
from lucid.distributions.extra import ContinuousBernoulli, Gumbel, InverseGamma, Kumaraswamy
from lucid.distributions.gamma import Beta, Gamma
from lucid.distributions.independent import Independent
from lucid.distributions.kl import (
    _EULER_GAMMA,
    _kl_categorical_categorical,
    kl_divergence,
    register_kl,
)
from lucid.distributions.matrix import Wishart
from lucid.distributions.multivariate import (
    LowRankMultivariateNormal,
    MultivariateNormal,
    _batch_lowrank_logdet,
    _batch_lowrank_mahalanobis,
)
from lucid.distributions.normal import LogNormal, Normal
from lucid.distributions.transformed_distribution import Transform, TransformedDistribution
from lucid.distributions.transforms import ExpTransform
from lucid.distributions.uniform import Uniform
from lucid.special import multigammaln, xlog1py


def _x_log_x(tensor: Tensor) -> Tensor:
    return lucid.xlogy(tensor, tensor)


def _transform_eq(a: Transform, b: Transform) -> bool:
    if type(a).__eq__ is not object.__eq__:
        return bool(a == b)
    if type(a) is not type(b) or type(a) is Transform:
        return False
    keys = [k for k in a.__dict__ if k != "_inv"]
    return keys == [k for k in b.__dict__ if k != "_inv"] and all(
        _value_eq(a.__dict__[k], b.__dict__[k]) for k in keys
    )


def _value_eq(a: object, b: object) -> bool:
    if a is b:
        return True
    if isinstance(a, Tensor) and isinstance(b, Tensor):
        return a.shape == b.shape and bool((a == b).all().item())
    if isinstance(a, Transform) and isinstance(b, Transform):
        return _transform_eq(a, b)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_value_eq(x, y) for x, y in zip(a, b, strict=False))
    return bool(a == b)


def _register_inf_pairs() -> None:
    specs: tuple[tuple[type, tuple[type, ...], str], ...] = (
        (Beta, (Pareto,), "concentration1"),
        (ContinuousBernoulli, (Pareto,), "probs"),
        (Exponential, (Beta, ContinuousBernoulli, Pareto, Uniform), "rate"),
        (Gamma, (Beta, ContinuousBernoulli, Pareto, Uniform), "concentration"),
        (Gumbel, (Beta, ContinuousBernoulli, Exponential, Gamma, Pareto, Uniform), "loc"),
        (Laplace, (Beta, ContinuousBernoulli, Exponential, Gamma, Pareto, Uniform), "loc"),
        (Normal, (Beta, ContinuousBernoulli, Exponential, Gamma, Pareto, Uniform), "loc"),
        (Pareto, (Beta, ContinuousBernoulli, Uniform), "scale"),
        (Poisson, (Bernoulli, Binomial), "rate"),
    )
    for p_cls, q_classes, attr in specs:

        def _inf_fn(p: Distribution, q: Distribution, _attr: str = attr) -> Tensor:
            del q
            param: Tensor = getattr(p, _attr)
            return lucid.full_like(param, float("inf"))

        for q_cls in q_classes:
            register_kl(p_cls, q_cls)(_inf_fn)


_register_inf_pairs()


@register_kl(Binomial, Binomial)
def _kl_binomial_binomial(p: Binomial, q: Binomial) -> Tensor:
    if bool((p.total_count < q.total_count).any().item()):
        raise NotImplementedError(
            "KL between Binomials where q.total_count > p.total_count is not implemented"
        )
    kl = p.total_count * (p.probs * (p.logits - q.logits) + (-p.probs).log1p() - (-q.probs).log1p())
    return lucid.where(p.total_count > q.total_count, float("inf"), kl)


@register_kl(ContinuousBernoulli, ContinuousBernoulli)
def _kl_continuous_bernoulli_continuous_bernoulli(
    p: ContinuousBernoulli, q: ContinuousBernoulli
) -> Tensor:
    t1 = p.mean * (p.logits - q.logits) + p._log_normalizer() + (-p.probs).log1p()
    return t1 - q._log_normalizer() - (-q.probs).log1p()


@register_kl(ExponentialFamily, ExponentialFamily)
def _kl_expfamily_expfamily(p: ExponentialFamily, q: ExponentialFamily) -> Tensor:
    if type(p) is not type(q):
        raise NotImplementedError(
            "The cross KL-divergence between different exponential families cannot "
            "be computed using Bregman divergences"
        )
    p_nparams = [np.detach().requires_grad_(True) for np in p._natural_params]
    q_nparams = q._natural_params
    lg_normal = p._log_normalizer(*p_nparams)
    gradients = lucid.autograd.grad(lg_normal.sum(), p_nparams, create_graph=True)
    result = q._log_normalizer(*q_nparams) - lg_normal
    for pnp, qnp, g in zip(p_nparams, q_nparams, gradients, strict=False):
        result = result - _sum_rightmost((qnp - pnp) * g, pnp.ndim - len(p.batch_shape))
    return result


@register_kl(OneHotCategorical, OneHotCategorical)
def _kl_onehotcategorical_onehotcategorical(p: OneHotCategorical, q: OneHotCategorical) -> Tensor:
    return _kl_categorical_categorical(p._cat, q._cat)


@register_kl(Pareto, Pareto)
def _kl_pareto_pareto(p: Pareto, q: Pareto) -> Tensor:
    scale_ratio, alpha_ratio = p.scale / q.scale, q.alpha / p.alpha
    result = q.alpha * scale_ratio.log() - alpha_ratio.log() + alpha_ratio - 1.0
    return lucid.where(p.scale < q.scale, float("inf"), result)


@register_kl(HalfCauchy, HalfCauchy)
def _kl_halfcauchy_halfcauchy(p: HalfCauchy, q: HalfCauchy) -> Tensor:
    return kl_divergence(p._base, q._base)


@register_kl(InverseGamma, InverseGamma)
def _kl_inversegamma_inversegamma(p: InverseGamma, q: InverseGamma) -> Tensor:
    return kl_divergence(p._gamma, q._gamma)


@register_kl(Weibull, Weibull)
def _kl_weibull_weibull(p: Weibull, q: Weibull) -> Tensor:
    if not (_value_eq(p.concentration, q.concentration) and _value_eq(p.scale, q.scale)):
        raise NotImplementedError
    return lucid.zeros_like(p.scale)


@register_kl(Kumaraswamy, Kumaraswamy)
def _kl_kumaraswamy_kumaraswamy(p: Kumaraswamy, q: Kumaraswamy) -> Tensor:
    if not (
        _value_eq(p.concentration0, q.concentration0)
        and _value_eq(p.concentration1, q.concentration1)
    ):
        raise NotImplementedError
    return lucid.zeros_like(p.concentration0)


@register_kl(Wishart, Wishart)
def _kl_wishart_wishart(p: Wishart, q: Wishart) -> Tensor:
    if p.event_shape != q.event_shape:
        raise NotImplementedError
    dim = p._dim
    nu_p = p.df.broadcast_to(p.batch_shape) if p.batch_shape else p.df
    nu_q = q.df.broadcast_to(q.batch_shape) if q.batch_shape else q.df
    p_nparams = [
        (-0.5 * p.precision_matrix).detach().requires_grad_(True),
        (0.5 * (nu_p - dim - 1.0)).detach().requires_grad_(True),
    ]
    q_nparams = [-0.5 * q.precision_matrix, 0.5 * (nu_q - dim - 1.0)]

    def _log_norm(x: Tensor, y: Tensor) -> Tensor:
        half_df = y + 0.5 * (dim + 1.0)
        return half_df * (-lucid.linalg.slogdet(-2.0 * x)[1] + math.log(2.0) * dim) + multigammaln(
            half_df, dim
        )

    lg_p = _log_norm(p_nparams[0], p_nparams[1])
    grads = lucid.autograd.grad(lg_p.sum(), p_nparams, create_graph=True)
    result = _log_norm(q_nparams[0], q_nparams[1]) - lg_p
    for pnp, qnp, g in zip(p_nparams, q_nparams, grads, strict=False):
        result = result - _sum_rightmost((qnp - pnp) * g, pnp.ndim - len(p.batch_shape))
    return result


@register_kl(TransformedDistribution, TransformedDistribution)
def _kl_transformed_transformed(p: TransformedDistribution, q: TransformedDistribution) -> Tensor:
    if not _value_eq(p.transforms, q.transforms) or p.event_shape != q.event_shape:
        raise NotImplementedError
    return kl_divergence(p.base_dist, q.base_dist)


@register_kl(LogNormal, TransformedDistribution)
def _kl_lognormal_transformed(p: LogNormal, q: TransformedDistribution) -> Tensor:
    if not _value_eq(q.transforms, [ExpTransform()]) or q.event_shape != ():
        raise NotImplementedError
    return kl_divergence(p._base, q.base_dist)


@register_kl(TransformedDistribution, LogNormal)
def _kl_transformed_lognormal(p: TransformedDistribution, q: LogNormal) -> Tensor:
    if not _value_eq(p.transforms, [ExpTransform()]) or p.event_shape != ():
        raise NotImplementedError
    return kl_divergence(p.base_dist, q._base)


@register_kl(Uniform, Uniform)
def _kl_uniform_uniform(p: Uniform, q: Uniform) -> Tensor:
    result: Tensor = ((q.high - q.low) / (p.high - p.low)).log()
    return lucid.where((q.low > p.low) | (q.high < p.high), float("inf"), result)


@register_kl(Geometric, Geometric)
def _kl_geometric_geometric(p: Geometric, q: Geometric) -> Tensor:
    pp, qp = p.probs, q.probs
    mean_failures = (1.0 - pp) / pp
    return pp.log() - qp.log() + xlog1py(mean_failures, -pp) - xlog1py(mean_failures, -qp)


@register_kl(Independent, Independent)
def _kl_independent_independent(p: Independent, q: Independent) -> Tensor:
    if p.reinterpreted_batch_ndims != q.reinterpreted_batch_ndims:
        raise NotImplementedError(
            "kl_divergence(Independent, Independent) requires equal reinterpreted_batch_ndims."
        )
    return _sum_rightmost(kl_divergence(p.base_dist, q.base_dist), p.reinterpreted_batch_ndims)


@register_kl(Bernoulli, Poisson)
def _kl_bernoulli_poisson(p: Bernoulli, q: Poisson) -> Tensor:
    return -p.entropy() - (p.probs * q.rate.log() - q.rate)


@register_kl(Beta, ContinuousBernoulli)
def _kl_beta_continuous_bernoulli(p: Beta, q: ContinuousBernoulli) -> Tensor:
    return -p.entropy() - p.mean * q.logits - (-q.probs).log1p() - q._log_normalizer()


@register_kl(Beta, Exponential)
def _kl_beta_exponential(p: Beta, q: Exponential) -> Tensor:
    return (
        -p.entropy()
        - q.rate.log()
        + q.rate * (p.concentration1 / (p.concentration1 + p.concentration0))
    )


@register_kl(Beta, Gamma)
def _kl_beta_gamma(p: Beta, q: Gamma) -> Tensor:
    sum_p = p.concentration1 + p.concentration0
    t2 = lucid.lgamma(q.concentration) - q.concentration * q.rate.log()
    t3 = (q.concentration - 1.0) * (lucid.digamma(p.concentration1) - lucid.digamma(sum_p))
    return -p.entropy() + t2 - t3 + q.rate * p.concentration1 / sum_p


@register_kl(Beta, Normal)
def _kl_beta_normal(p: Beta, q: Normal) -> Tensor:
    e_beta = p.concentration1 / (p.concentration1 + p.concentration0)
    var_normal = q.scale**2
    t3 = (e_beta * (1.0 - e_beta) / (p.concentration1 + p.concentration0 + 1.0) + e_beta**2) * 0.5
    return (
        -p.entropy()
        + 0.5 * (var_normal * 2.0 * math.pi).log()
        + (t3 - q.loc * e_beta + 0.5 * (q.loc**2)) / var_normal
    )


@register_kl(Beta, Uniform)
def _kl_beta_uniform(p: Beta, q: Uniform) -> Tensor:
    result = -p.entropy() + (q.high - q.low).log()
    return lucid.where((q.low > 0.0) | (q.high < 1.0), float("inf"), result)


@register_kl(ContinuousBernoulli, Exponential)
def _kl_continuous_bernoulli_exponential(p: ContinuousBernoulli, q: Exponential) -> Tensor:
    return -p.entropy() - q.rate.log() + q.rate * p.mean


@register_kl(ContinuousBernoulli, Normal)
def _kl_continuous_bernoulli_normal(p: ContinuousBernoulli, q: Normal) -> Tensor:
    t2 = 0.5 * (math.log(2.0 * math.pi) + (q.loc / q.scale) ** 2) + q.scale.log()
    t3 = (p.variance + p.mean**2 - 2.0 * q.loc * p.mean) / (2.0 * q.scale**2)
    return -p.entropy() + t2 + t3


@register_kl(ContinuousBernoulli, Uniform)
def _kl_continuous_bernoulli_uniform(p: ContinuousBernoulli, q: Uniform) -> Tensor:
    result = -p.entropy() + (q.high - q.low).log()
    return lucid.where((q.low >= 0.0) | (q.high <= 1.0), float("inf"), result)


@register_kl(Exponential, Gamma)
def _kl_exponential_gamma(p: Exponential, q: Gamma) -> Tensor:
    ratio = q.rate / p.rate
    return (
        -q.concentration * ratio.log()
        + ratio
        + lucid.lgamma(q.concentration)
        + q.concentration * _EULER_GAMMA
        - (1.0 + _EULER_GAMMA)
    )


@register_kl(Exponential, Gumbel)
def _kl_exponential_gumbel(p: Exponential, q: Gumbel) -> Tensor:
    sr, lsr = p.rate * q.scale, q.loc / q.scale
    return sr.log() - 1.0 - lsr + lsr.exp() * sr / (sr + 1.0) + 1.0 / sr


@register_kl(Exponential, Normal)
def _kl_exponential_normal(p: Exponential, q: Normal) -> Tensor:
    var_n, rate_sqr = q.scale**2, p.rate**2
    return (
        0.5 * (rate_sqr * var_n * 2.0 * math.pi).log()
        - 1.0
        + (1.0 / rate_sqr - q.loc / p.rate + 0.5 * (q.loc**2)) / var_n
    )


@register_kl(Gamma, Exponential)
def _kl_gamma_exponential(p: Gamma, q: Exponential) -> Tensor:
    return -p.entropy() - q.rate.log() + q.rate * p.concentration / p.rate


@register_kl(Gamma, Gumbel)
def _kl_gamma_gumbel(p: Gamma, q: Gumbel) -> Tensor:
    bs, lsr = p.rate * q.scale, q.loc / q.scale
    t1 = (
        (p.concentration - 1.0) * lucid.digamma(p.concentration)
        - lucid.lgamma(p.concentration)
        - p.concentration
    )
    return (
        t1
        + bs.log()
        + p.concentration / bs
        + lsr.exp() * (1.0 + 1.0 / bs) ** (-p.concentration)
        - lsr
    )


@register_kl(Gamma, Normal)
def _kl_gamma_normal(p: Gamma, q: Normal) -> Tensor:
    var_n, beta_sqr = q.scale**2, p.rate**2
    t1 = (
        0.5 * (beta_sqr * var_n * 2.0 * math.pi).log()
        - p.concentration
        - lucid.lgamma(p.concentration)
    )
    t2 = (
        0.5 * (p.concentration**2 + p.concentration) / beta_sqr
        - q.loc * p.concentration / p.rate
        + 0.5 * (q.loc**2)
    )
    return t1 + (p.concentration - 1.0) * lucid.digamma(p.concentration) + t2 / var_n


@register_kl(Gumbel, Normal)
def _kl_gumbel_normal(p: Gumbel, q: Normal) -> Tensor:
    pr = p.scale / q.scale
    return (
        -(pr / math.sqrt(2.0 * math.pi)).log()
        + (math.pi * pr * 0.5) ** 2 / 3.0
        + ((p.loc + p.scale * _EULER_GAMMA - q.loc) / q.scale) ** 2 * 0.5
        - (_EULER_GAMMA + 1.0)
    )


@register_kl(Laplace, Normal)
def _kl_laplace_normal(p: Laplace, q: Normal) -> Tensor:
    var_n = q.scale**2
    sr = (p.scale**2) / var_n
    return (
        -0.5 * (2.0 * sr / math.pi).log()
        + sr
        + (0.5 * (p.loc**2) - p.loc * q.loc + 0.5 * (q.loc**2)) / var_n
        - 1.0
    )


@register_kl(Normal, Gumbel)
def _kl_normal_gumbel(p: Normal, q: Gumbel) -> Tensor:
    msr, vsr, lsr = p.loc / q.scale, (p.scale / q.scale) ** 2, q.loc / q.scale
    return (
        -0.5 * vsr.log()
        + (msr - lsr)
        + (-msr + 0.5 * vsr + lsr).exp()
        - 0.5 * (1.0 + math.log(2.0 * math.pi))
    )


@register_kl(Normal, Laplace)
def _kl_normal_laplace(p: Normal, q: Laplace) -> Tensor:
    loc_diff, sr = p.loc - q.loc, p.scale / q.scale
    ldsr = loc_diff / p.scale
    t2 = math.sqrt(2.0 / math.pi) * p.scale * (-0.5 * ldsr**2).exp()
    t3 = loc_diff * lucid.erf(math.sqrt(0.5) * ldsr)
    return -sr.log() + (t2 + t3) / q.scale - 0.5 * (1.0 + math.log(0.5 * math.pi))


@register_kl(Pareto, Exponential)
def _kl_pareto_exponential(p: Pareto, q: Exponential) -> Tensor:
    sr = p.scale * q.rate
    res = (p.alpha / sr).log() - 1.0 / p.alpha + p.alpha * sr / (p.alpha - 1.0) - 1.0
    return lucid.where(p.alpha <= 1.0, float("inf"), res)


@register_kl(Pareto, Gamma)
def _kl_pareto_gamma(p: Pareto, q: Gamma) -> Tensor:
    c = p.scale.log() + 1.0 / p.alpha
    t2 = lucid.lgamma(q.concentration) - q.concentration * q.rate.log()
    res = (
        (p.alpha.log() - c)
        + t2
        + (1.0 - q.concentration) * c
        + q.rate * p.alpha * p.scale / (p.alpha - 1.0)
        - 1.0
    )
    return lucid.where(p.alpha <= 1.0, float("inf"), res)


@register_kl(Pareto, Normal)
def _kl_pareto_normal(p: Pareto, q: Normal) -> Tensor:
    var_n, c = 2.0 * (q.scale**2), p.scale / (p.alpha - 1.0)
    t1 = (math.sqrt(2.0 * math.pi) * q.scale * p.alpha / p.scale).log() - 1.0 / p.alpha
    res = t1 + (p.alpha * (c**2) / (p.alpha - 2.0) + (p.alpha * c - q.loc) ** 2) / var_n - 1.0
    return lucid.where(p.alpha <= 2.0, float("inf"), res)


@register_kl(Uniform, Beta)
def _kl_uniform_beta(p: Uniform, q: Beta) -> Tensor:
    c = p.high - p.low
    t2 = (q.concentration1 - 1.0) * (_x_log_x(p.high) - _x_log_x(p.low) - c) / c
    t3 = (q.concentration0 - 1.0) * (_x_log_x(1.0 - p.high) - _x_log_x(1.0 - p.low) + c) / c
    t4 = (
        lucid.lgamma(q.concentration1)
        + lucid.lgamma(q.concentration0)
        - lucid.lgamma(q.concentration1 + q.concentration0)
    )
    return lucid.where((p.high > 1.0) | (p.low < 0.0), float("inf"), t3 + t4 - c.log() - t2)


@register_kl(Uniform, ContinuousBernoulli)
def _kl_uniform_continuous_bernoulli(p: Uniform, q: ContinuousBernoulli) -> Tensor:
    res = -p.entropy() - p.mean * q.logits - (-q.probs).log1p() - q._log_normalizer()
    return lucid.where((p.high >= 1.0) | (p.low <= 0.0), float("inf"), res)


@register_kl(Uniform, Exponential)
def _kl_uniform_exponential(p: Uniform, q: Exponential) -> Tensor:
    res = q.rate * (p.high + p.low) / 2.0 - ((p.high - p.low) * q.rate).log()
    return lucid.where(p.low < 0.0, float("inf"), res)


@register_kl(Uniform, Gamma)
def _kl_uniform_gamma(p: Uniform, q: Gamma) -> Tensor:
    c = p.high - p.low
    t2 = lucid.lgamma(q.concentration) - q.concentration * q.rate.log()
    t3 = (1.0 - q.concentration) * (_x_log_x(p.high) - _x_log_x(p.low) - c) / c
    res = -c.log() + t2 + t3 + q.rate * (p.high + p.low) / 2.0
    return lucid.where(p.low < 0.0, float("inf"), res)


@register_kl(Uniform, Gumbel)
def _kl_uniform_gumbel(p: Uniform, q: Gumbel) -> Tensor:
    c = q.scale / (p.high - p.low)
    h, l = (p.high - q.loc) / q.scale, (p.low - q.loc) / q.scale
    return c.log() + 0.5 * (h + l) - c * ((-h).exp() - (-l).exp())


@register_kl(Uniform, Normal)
def _kl_uniform_normal(p: Uniform, q: Normal) -> Tensor:
    c = p.high - p.low
    return (math.sqrt(math.pi * 2.0) * q.scale / c).log() + 0.5 * (
        (c**2) / 12.0 + ((p.high + p.low - 2.0 * q.loc) / 2.0) ** 2
    ) / (q.scale**2)


@register_kl(Uniform, Pareto)
def _kl_uniform_pareto(p: Uniform, q: Pareto) -> Tensor:
    su = p.high - p.low
    res = ((_x_log_x(p.high) - _x_log_x(p.low) - su) / su) * (q.alpha + 1.0) - (
        q.alpha * (q.scale**q.alpha) * su
    ).log()
    return lucid.where(p.low < q.scale, float("inf"), res)


def _check_mvn_event_shapes(
    p: LowRankMultivariateNormal | MultivariateNormal,
    q: LowRankMultivariateNormal | MultivariateNormal,
    label: str,
) -> None:
    if p.event_shape != q.event_shape:
        raise ValueError(
            f"KL-divergence between {label} with different event shapes cannot be computed"
        )


def _lowrank_whitened_factor(q: LowRankMultivariateNormal) -> Tensor:
    q_Wt_Dinv = q._unbroadcasted_cov_factor.mT / q._unbroadcasted_cov_diag.unsqueeze(-2)
    return lucid.linalg.solve_triangular(q._capacitance_tril, q_Wt_Dinv, upper=False)


@register_kl(LowRankMultivariateNormal, LowRankMultivariateNormal)
def _kl_lowrank_lowrank(p: LowRankMultivariateNormal, q: LowRankMultivariateNormal) -> Tensor:
    _check_mvn_event_shapes(p, q, "two LowRankMultivariateNormals")
    term1 = _batch_lowrank_logdet(
        q._unbroadcasted_cov_diag, q._capacitance_tril
    ) - _batch_lowrank_logdet(p._unbroadcasted_cov_diag, p._capacitance_tril)
    term3 = _batch_lowrank_mahalanobis(
        q._unbroadcasted_cov_factor, q._unbroadcasted_cov_diag, q.loc - p.loc, q._capacitance_tril
    )
    A = _lowrank_whitened_factor(q)
    term21 = (p._unbroadcasted_cov_diag / q._unbroadcasted_cov_diag).sum(dim=-1)
    S1 = p._unbroadcasted_cov_factor * (1.0 / q._unbroadcasted_cov_diag.sqrt().unsqueeze(-1))
    term22 = (S1 * S1).sum(dim=(-2, -1))
    S2 = A * p._unbroadcasted_cov_diag.sqrt().unsqueeze(-2)
    term23 = (S2 * S2).sum(dim=(-2, -1))
    S3 = A @ p._unbroadcasted_cov_factor
    term24 = (S3 * S3).sum(dim=(-2, -1))
    term2 = term21 + term22 - term23 - term24
    return 0.5 * (term1 + term2 + term3 - float(p.event_shape[0]))


@register_kl(LowRankMultivariateNormal, MultivariateNormal)
def _kl_lowrank_mvn(p: LowRankMultivariateNormal, q: MultivariateNormal) -> Tensor:
    _check_mvn_event_shapes(p, q, "LowRankMultivariateNormal and MultivariateNormal")
    term1 = 2.0 * q.scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1) - _batch_lowrank_logdet(
        p._unbroadcasted_cov_diag, p._capacitance_tril
    )
    diff = (q.loc - p.loc).unsqueeze(-1)
    z = lucid.linalg.solve_triangular(q.scale_tril, diff, upper=False)
    term3 = (z * z).sum(dim=(-2, -1))
    L1 = lucid.linalg.solve_triangular(
        q.scale_tril, lucid.diag_embed(p._unbroadcasted_cov_diag.sqrt()), upper=False
    )
    L2 = lucid.linalg.solve_triangular(q.scale_tril, p._unbroadcasted_cov_factor, upper=False)
    term2 = (L1 * L1).sum(dim=(-2, -1)) + (L2 * L2).sum(dim=(-2, -1))
    return 0.5 * (term1 + term2 + term3 - float(p.event_shape[0]))


@register_kl(MultivariateNormal, LowRankMultivariateNormal)
def _kl_mvn_lowrank(p: MultivariateNormal, q: LowRankMultivariateNormal) -> Tensor:
    _check_mvn_event_shapes(p, q, "MultivariateNormal and LowRankMultivariateNormal")
    term1 = _batch_lowrank_logdet(
        q._unbroadcasted_cov_diag, q._capacitance_tril
    ) - 2.0 * p.scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
    term3 = _batch_lowrank_mahalanobis(
        q._unbroadcasted_cov_factor, q._unbroadcasted_cov_diag, q.loc - p.loc, q._capacitance_tril
    )
    A = _lowrank_whitened_factor(q)
    S1 = p.scale_tril / q._unbroadcasted_cov_diag.sqrt().unsqueeze(-1)
    S2 = A @ p.scale_tril
    term2 = (S1 * S1).sum(dim=(-2, -1)) - (S2 * S2).sum(dim=(-2, -1))
    return 0.5 * (term1 + term2 + term3 - float(p.event_shape[0]))
