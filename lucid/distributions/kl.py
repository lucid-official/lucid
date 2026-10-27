"""``kl_divergence`` registry — pairwise dispatch keyed on distribution
type, mirroring the reference framework's ``register_kl`` mechanism.
"""

from collections.abc import Callable
import importlib
from typing import override
import warnings

import lucid
from lucid._tensor.tensor import Tensor
from lucid.distributions.bernoulli import Bernoulli
from lucid.distributions.categorical import Categorical
from lucid.distributions.continuous_extra import HalfNormal
from lucid.distributions.discrete import Poisson
from lucid.distributions.distribution import Distribution
from lucid.distributions.exponential import Cauchy, Exponential, Laplace
from lucid.distributions.extra import Gumbel
from lucid.distributions.gamma import Beta, Dirichlet, Gamma
from lucid.distributions.multivariate import MultivariateNormal
from lucid.distributions.normal import LogNormal, Normal

_KLFn = Callable[..., Tensor]
_KL_REGISTRY: dict[tuple[type, type], _KLFn] = {}
_KL_MEMOIZE: dict[tuple[type, type], _KLFn | None] = {}
_EULER_GAMMA: float = 0.5772156649015329


class _Match:
    """Partial ordering over ``(type_p, type_q)`` pairs by subclass specificity."""

    __slots__ = ("types",)

    def __init__(self, *types: type) -> None:
        self.types = types

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Match):
            return NotImplemented
        return self.types == other.types

    def __le__(self, other: _Match) -> bool:
        for x, y in zip(self.types, other.types, strict=False):
            if not issubclass(x, y):
                return False
            if x is not y:
                break
        return True

    def __lt__(self, other: _Match) -> bool:
        return self <= other and self.types != other.types


def _dispatch_kl(type_p: type, type_q: type) -> _KLFn | None:
    """Find the most specific registered ``(super_p, super_q)`` match."""
    matches = [
        (super_p, super_q)
        for super_p, super_q in _KL_REGISTRY
        if issubclass(type_p, super_p) and issubclass(type_q, super_q)
    ]
    if not matches:
        return None
    left_p, left_q = min(_Match(*m) for m in matches).types
    right_q, right_p = min(_Match(*reversed(m)) for m in matches).types
    left_fun = _KL_REGISTRY[left_p, left_q]
    right_fun = _KL_REGISTRY[right_p, right_q]
    if left_fun is not right_fun:
        warnings.warn(
            f"Ambiguous kl_divergence({type_p.__name__}, {type_q.__name__}). "
            f"Please register_kl({left_p.__name__}, {right_q.__name__})",
            RuntimeWarning,
            stacklevel=2,
        )
    return left_fun


def register_kl[F: Callable[..., Tensor]](p_cls: type, q_cls: type) -> Callable[[F], F]:
    r"""Register a closed-form KL implementation for a distribution pair.

    Decorator used to add an analytical
    :math:`\mathrm{KL}(p \,\|\, q)` formula to the global dispatch
    registry, keyed on the *exact* (or ancestor) types of the two
    arguments.  Once registered, :func:`kl_divergence` will route calls
    matching ``(p_cls, q_cls)`` (or any subclass thereof, via MRO walk)
    to the decorated function.

    Parameters
    ----------
    p_cls : type
        Class of the "left" distribution :math:`p` (the reference
        distribution in :math:`\mathrm{KL}(p \,\|\, q)`).
    q_cls : type
        Class of the "right" distribution :math:`q`.

    Returns
    -------
    Callable[[F], F]
        A decorator that takes a two-argument callable
        ``fn(p, q) -> Tensor`` and registers it in the internal KL
        registry, returning ``fn`` unchanged so the decorated function
        remains directly callable.

    Raises
    ------
    TypeError
        If ``p_cls`` or ``q_cls`` is not a :class:`Distribution` subclass.

    Notes
    -----
    Dispatch precedence inside :func:`kl_divergence`:

    1. Exact class match ``(type(p), type(q))``.
    2. Most-derived ancestor pair ordered by subclass specificity; emits
       a :class:`RuntimeWarning` if left- and right-lexicographic minima
       disagree.
    3. ``NotImplementedError`` when no closed-form pair is registered.

    Examples
    --------
    >>> from lucid.distributions import Normal
    >>> from lucid.distributions.kl import register_kl, kl_divergence
    >>> @register_kl(Normal, Normal)
    ... def _kl_normal_normal(p, q):
    ...     return (q.scale.log() - p.scale.log()
    ...             + (p.variance + (p.loc - q.loc) ** 2) / (2.0 * q.variance)
    ...             - 0.5)
    >>> p = Normal(0.0, 1.0); q = Normal(0.0, 2.0)
    >>> kl_divergence(p, q)  # log(2) + 1/8 - 1/2
    tensor(0.3181)
    """
    for cls, name in ((p_cls, "p_cls"), (q_cls, "q_cls")):
        chk: object = cls
        if not (isinstance(chk, type) and issubclass(chk, Distribution)):
            raise TypeError(f"Expected {name} to be a Distribution subclass but got {cls}")

    def _decorator(fn: F) -> F:
        """Insert ``fn`` into the KL registry under ``(p_cls, q_cls)``."""
        _KL_REGISTRY[(p_cls, q_cls)] = fn
        _KL_MEMOIZE.clear()
        return fn

    return _decorator


def kl_divergence(p: Distribution, q: Distribution) -> Tensor:
    r"""Compute the Kullback–Leibler divergence :math:`\mathrm{KL}(p \,\|\, q)`.

    The KL divergence is the expected log-ratio of two probability
    measures :math:`p` and :math:`q` defined on the same sample space:

    .. math::

        \mathrm{KL}(p \,\|\, q) =
            \mathbb{E}_{x \sim p}\!\left[\log \frac{p(x)}{q(x)}\right]
        = \int p(x) \log \frac{p(x)}{q(x)} \, dx

    It is non-negative
    (:math:`\mathrm{KL}(p \,\|\, q) \geq 0`, with equality iff
    :math:`p = q` almost everywhere), is not symmetric in general
    (:math:`\mathrm{KL}(p \,\|\, q) \neq \mathrm{KL}(q \,\|\, p)`), and
    does not satisfy the triangle inequality — so it is a *divergence*,
    not a metric.

    Dispatch looks up the most-specific registered ancestor pair in the
    registry built by :func:`register_kl` and raises
    :class:`NotImplementedError` when no analytical formula is registered.

    Parameters
    ----------
    p : Distribution
        Left-hand distribution.
    q : Distribution
        Right-hand distribution (must share the support / event shape
        of ``p``).

    Returns
    -------
    Tensor
        Non-negative tensor of shape ``batch_shape`` giving the per-batch
        KL divergence in nats.

    Raises
    ------
    NotImplementedError
        If the distribution types have not been registered via
        :func:`register_kl`.

    Notes
    -----
    KL divergence appears throughout machine learning:

    * **Variational inference**: the ELBO equals
      :math:`\log p(\mathbf{x}) - \mathrm{KL}(q(\mathbf{z})\,\|\,
      p(\mathbf{z} \mid \mathbf{x}))`.
    * **VAE training**: the encoder regulariser is
      :math:`\mathrm{KL}(q(\mathbf{z} \mid \mathbf{x})\,\|\,
      p(\mathbf{z}))`, available analytically for
      Normal-vs-Normal pairs.
    * **Maximum likelihood**: minimising negative log-likelihood is
      equivalent to minimising :math:`\mathrm{KL}(p_{\text{data}}
      \,\|\, p_{\theta})`.
    * **Mode-seeking vs mode-covering**:
      :math:`\mathrm{KL}(p\,\|\,q)` is mode-covering in :math:`q`;
      :math:`\mathrm{KL}(q\,\|\,p)` is mode-seeking — important for
      variational approximations.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import Normal
    >>> from lucid.distributions.kl import kl_divergence
    >>> p = Normal(loc=0.0, scale=1.0)
    >>> q = Normal(loc=1.0, scale=2.0)
    >>> kl_divergence(p, q)  # log(2) + (1 + 1)/8 - 1/2
    tensor(0.4431)
    """
    key = (type(p), type(q))
    try:
        fun = _KL_MEMOIZE[key]
    except KeyError:
        fun = _dispatch_kl(type(p), type(q))
        _KL_MEMOIZE[key] = fun
    if fun is None:
        raise NotImplementedError(
            f"No KL(p || q) is implemented for p type {type(p).__name__} "
            f"and q type {type(q).__name__}"
        )
    return fun(p, q)


# ── concrete pairs ────────────────────────────────────────────────────────


@register_kl(Normal, Normal)
def _kl_normal_normal(p: Normal, q: Normal) -> Tensor:
    r"""Closed-form KL divergence between two univariate Normals.

    .. math::

        \mathrm{KL}(p \,\|\, q) = \log\frac{\sigma_q}{\sigma_p}
            + \frac{\sigma_p^2 + (\mu_p - \mu_q)^2}{2\sigma_q^2} - \frac{1}{2}
    """
    var_p = p.variance
    var_q = q.variance
    return (q.scale.log() - p.scale.log()) + (var_p + (p.loc - q.loc) ** 2) / (2.0 * var_q) - 0.5


def _bernoulli_outcomes(d: Bernoulli) -> tuple[Tensor, Tensor]:
    """``(P(X = 1), P(X = 0))`` of ``d``, each to full relative precision.

    Built from logits, ``P(X = 0)`` is ``sigmoid(-ℓ)`` rather than
    ``1 - sigmoid(ℓ)``: at ``ℓ = 15`` the subtraction keeps one significant
    digit of a number near ``3e-7``, and its logarithm is off in the second.
    """
    if d._is_logits:
        return d.logits.sigmoid(), (-d.logits).sigmoid()
    return d.probs, 1.0 - d.probs


@register_kl(Bernoulli, Bernoulli)
def _kl_bernoulli_bernoulli(p: Bernoulli, q: Bernoulli) -> Tensor:
    r"""Closed-form KL divergence between two Bernoulli distributions.

    .. math::

        \mathrm{KL}(p \,\|\, q) = p \log\frac{p}{q} + (1-p) \log\frac{1-p}{1-q}

    Each :math:`x \log y` is :func:`lucid.xlogy`, which owns the boundary
    convention: an outcome ``p`` cannot produce contributes ``0`` whatever
    ``q`` says of it, and an outcome ``p`` can produce but ``q`` cannot makes
    the divergence ``+inf`` (``KL(Bernoulli(0.3) || Bernoulli(0)) = inf``),
    as the reference framework answers.  The probabilities are not clipped:
    a clip at ``1e-7`` turned that ``inf`` into ``4.22``.

    Notes
    -----
    Built from logits, each outcome's probability comes from its own
    sigmoid, so a ``q`` whose probability rounds to 1 in float32 (a logit
    above about 17) still has a positive ``1 - q``: ``KL(Bernoulli(logits=-20)
    || Bernoulli(logits=20))`` is 20.  The reference framework tests
    ``q.probs == 1`` and answers ``inf`` there.
    """
    p1, p0 = _bernoulli_outcomes(p)
    q1, q0 = _bernoulli_outcomes(q)
    return (lucid.xlogy(p1, p1) - lucid.xlogy(p1, q1)) + (lucid.xlogy(p0, p0) - lucid.xlogy(p0, q0))


@register_kl(Categorical, Categorical)
def _kl_categorical_categorical(p: Categorical, q: Categorical) -> Tensor:
    r"""Closed-form KL divergence between two Categorical distributions.

    .. math::

        \mathrm{KL}(p \,\|\, q) = \sum_k p_k (\log p_k - \log q_k)

    A category with :math:`p_k = 0` contributes ``0`` (not
    ``0 · (-inf) = NaN``); one with :math:`p_k > 0 = q_k` makes the
    divergence ``+inf``.  The log-probabilities are replaced *before* the
    product wherever the term is decided by those rules, so the gradient
    never multiplies an infinity by zero.
    """
    p_probs = p._probs
    p_zero = p_probs == 0
    q_log = q._log_probs
    q_zero = lucid.isinf(q_log) & (q_log < 0)
    p_log = lucid.where(p_zero, 0.0, p._log_probs)
    q_log = lucid.where(p_zero | q_zero, 0.0, q_log)
    terms = lucid.where(q_zero & ~p_zero, float("inf"), p_probs * (p_log - q_log))
    return terms.sum(dim=-1)


@register_kl(Exponential, Exponential)
def _kl_exponential_exponential(p: Exponential, q: Exponential) -> Tensor:
    r"""Closed-form KL divergence between two Exponential distributions.

    .. math::

        \mathrm{KL}(p \,\|\, q) = \frac{\lambda_q}{\lambda_p} - 1
            - \log\frac{\lambda_q}{\lambda_p}
    """
    rate_ratio = q.rate / p.rate
    return rate_ratio - 1.0 - rate_ratio.log()


@register_kl(Gamma, Gamma)
def _kl_gamma_gamma(p: Gamma, q: Gamma) -> Tensor:
    r"""Closed-form KL divergence between two Gamma distributions.

    With shape :math:`\alpha` and rate :math:`\beta`:

    .. math::

        \mathrm{KL}(p \,\|\, q) = \alpha_q \log\frac{\beta_p}{\beta_q}
            + \log\frac{\Gamma(\alpha_q)}{\Gamma(\alpha_p)}
            + (\alpha_p - \alpha_q)\,\psi(\alpha_p)
            + (\beta_q - \beta_p)\,\frac{\alpha_p}{\beta_p}
    """
    a_p, b_p = p.concentration, p.rate
    a_q, b_q = q.concentration, q.rate
    t1 = a_q * (b_p / b_q).log()
    t2 = lucid.lgamma(a_q) - lucid.lgamma(a_p)
    t3 = (a_p - a_q) * lucid.digamma(a_p)
    t4 = (b_q - b_p) * (a_p / b_p)
    return t1 + t2 + t3 + t4


@register_kl(Beta, Beta)
def _kl_beta_beta(p: Beta, q: Beta) -> Tensor:
    r"""Closed-form KL divergence between two Beta distributions.

    Uses the digamma/log-gamma identity with shape parameters
    :math:`(\alpha_1, \alpha_0)` and :math:`(\beta_1, \beta_0)`.
    """
    sum_p = p.concentration1 + p.concentration0
    sum_q = q.concentration1 + q.concentration0
    t1 = lucid.lgamma(sum_p) - lucid.lgamma(sum_q)
    t2 = (
        lucid.lgamma(q.concentration1)
        - lucid.lgamma(p.concentration1)
        + lucid.lgamma(q.concentration0)
        - lucid.lgamma(p.concentration0)
    )
    t3 = (p.concentration1 - q.concentration1) * (
        lucid.digamma(p.concentration1) - lucid.digamma(sum_p)
    )
    t4 = (p.concentration0 - q.concentration0) * (
        lucid.digamma(p.concentration0) - lucid.digamma(sum_p)
    )
    return t1 + t2 + t3 + t4


@register_kl(Poisson, Poisson)
def _kl_poisson_poisson(p: Poisson, q: Poisson) -> Tensor:
    r"""Closed-form KL divergence between two Poisson distributions.

    .. math::

        \mathrm{KL}(p \,\|\, q) = \lambda_p \log\frac{\lambda_p}{\lambda_q}
            - \lambda_p + \lambda_q

    The :math:`\lambda_p \log \lambda` terms are :func:`lucid.xlogy`, so a
    rate of 0 — all mass at 0 — has its exact divergence
    :math:`\lambda_q`, and a :math:`\lambda_q` of 0 against a positive
    :math:`\lambda_p` is ``+inf``.

    Notes
    -----
    The reference framework multiplies :math:`\lambda_p` by
    :math:`\log \lambda_p` directly and answers NaN when
    :math:`\lambda_p = 0`.
    """
    return lucid.xlogy(p.rate, p.rate) - lucid.xlogy(p.rate, q.rate) - p.rate + q.rate


@register_kl(Dirichlet, Dirichlet)
def _kl_dirichlet_dirichlet(p: Dirichlet, q: Dirichlet) -> Tensor:
    r"""Closed-form KL divergence between two Dirichlet distributions.

    Uses the standard digamma + log-gamma form, summed over the K simplex
    components along the last axis.
    """
    a = p.concentration
    b = q.concentration
    sum_a = a.sum(dim=-1, keepdim=True)
    sum_b = b.sum(dim=-1, keepdim=True)
    t1 = (
        lucid.lgamma(sum_a).squeeze(-1)
        - lucid.lgamma(a).sum(dim=-1)
        - lucid.lgamma(sum_b).squeeze(-1)
        + lucid.lgamma(b).sum(dim=-1)
    )
    t2 = ((a - b) * (lucid.digamma(a) - lucid.digamma(sum_a))).sum(dim=-1)
    return t1 + t2


@register_kl(MultivariateNormal, MultivariateNormal)
def _kl_mvn_mvn(p: MultivariateNormal, q: MultivariateNormal) -> Tensor:
    r"""Closed-form KL divergence between two multivariate Normals.

    .. math::

        2\,\mathrm{KL} = \operatorname{tr}(\Sigma_q^{-1}\Sigma_p)
            + (\mu_q - \mu_p)^{\top}\Sigma_q^{-1}(\mu_q - \mu_p)
            - D + \log\frac{|\Sigma_q|}{|\Sigma_p|}

    All operations work on the lower-triangular Cholesky factors to avoid
    forming explicit inverses.
    """
    if p.event_shape != q.event_shape:
        raise ValueError(
            "KL-divergence between two Multivariate Normals with "
            "different event shapes cannot be computed"
        )
    Lp, Lq = p.scale_tril, q.scale_tril
    D = int(p._D)
    log_det_p = 2.0 * Lp.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
    log_det_q = 2.0 * Lq.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
    M = lucid.linalg.solve_triangular(Lq, Lp, upper=False)
    trace_term = (M * M).sum(dim=(-2, -1))
    diff = (q.loc - p.loc).unsqueeze(-1)
    z = lucid.linalg.solve_triangular(Lq, diff, upper=False)
    maha = (z * z).sum(dim=(-2, -1))
    return 0.5 * (trace_term + maha - float(D) + log_det_q - log_det_p)


@register_kl(Laplace, Laplace)
def _kl_laplace_laplace(p: Laplace, q: Laplace) -> Tensor:
    r"""Closed-form KL divergence between two Laplace distributions.

    Uses the formula from Gil et al. (2011) §3.2:

    .. math::

        \mathrm{KL} = -\log\frac{b_p}{b_q} + \frac{|\mu_p - \mu_q|}{b_q}
            + \frac{b_p}{b_q}\,\exp\!\left(-\frac{|\mu_p - \mu_q|}{b_p}\right) - 1
    """
    scale_ratio: Tensor = p.scale / q.scale
    loc_diff: Tensor = (p.loc - q.loc).abs()
    return (
        -scale_ratio.log() + loc_diff / q.scale + scale_ratio * (-(loc_diff / p.scale)).exp() - 1.0
    )


@register_kl(HalfNormal, HalfNormal)
def _kl_halfnormal_halfnormal(p: HalfNormal, q: HalfNormal) -> Tensor:
    """KL divergence between two HalfNormal distributions.

    Reduces to the KL of the underlying zero-centred Normal distributions
    since :math:`\\text{HalfNormal}(\\sigma) = |\\text{Normal}(0, \\sigma)|`.
    """
    return _kl_normal_normal(p._base, q._base)


@register_kl(LogNormal, LogNormal)
def _kl_lognormal_lognormal(p: LogNormal, q: LogNormal) -> Tensor:
    return _kl_normal_normal(Normal(p.loc, p.scale), Normal(q.loc, q.scale))


@register_kl(Gumbel, Gumbel)
def _kl_gumbel_gumbel(p: Gumbel, q: Gumbel) -> Tensor:
    ct1: Tensor = p.scale / q.scale
    ct23: Tensor = (q.loc - p.loc) / q.scale
    t12: Tensor = -ct1.log() - ct23 + ct1 * _EULER_GAMMA
    return t12 + (ct23 + lucid.lgamma(1.0 + ct1)).exp() - (1.0 + _EULER_GAMMA)


@register_kl(Cauchy, Cauchy)
def _kl_cauchy_cauchy(p: Cauchy, q: Cauchy) -> Tensor:
    t1: Tensor = ((p.scale + q.scale) ** 2 + (p.loc - q.loc) ** 2).log()
    t2: Tensor = (4.0 * p.scale * q.scale).log()
    return t1 - t2


importlib.import_module("lucid.distributions._kl_cross")
