"""``Gamma``, ``Chi2``, ``Beta``, ``Dirichlet`` — gamma-family distributions.

The reparameterised sampler for ``Gamma`` uses Marsaglia & Tsang's
acceptance-rejection algorithm wrapped in a fixed-cap retry loop: the
expected reject rate is well under 5 % per sample, so 8 rounds drives
the residual probability of any unsampled cell below ``2 ** −60``.

A small concentration puts most of the mass below the smallest float32, so
the draws are made as logarithms and held in a range the dtype can hold:
a ``Gamma`` draw at or above ``tiny``, a ``Beta`` or ``Dirichlet`` draw in
``[tiny, 1 - eps/2]`` — the reference's ranges.  The ``sample`` of ``Beta``
and ``Dirichlet`` normalises the logarithms directly and stays exact down
to any concentration; their ``rsample`` normalises the clamped Gamma draws,
so below a concentration of a few hundredths it loses the ordering of
components that are all under ``tiny``.
"""

from collections.abc import Sequence
from typing import Self, override

import lucid
from lucid._C import engine as _C_engine
from lucid._dtype import finfo
from lucid._tensor.tensor import Tensor
from lucid.autograd.function import Function, FunctionCtx
from lucid.distributions.constraints import (
    Constraint,
    independent,
    nonnegative,
    positive,
    simplex,
    unit_interval,
)
from lucid.distributions.distribution import ExponentialFamily

_MAX_GAMMA_RETRIES: int = 8


from lucid.distributions._util import _as_tensor, _broadcast_pair
from lucid.special import gammainc, xlog1py


def _sample_standard_log_gamma(concentration: Tensor, sample_shape: tuple[int, ...]) -> Tensor:
    r"""Sample :math:`\log X` for :math:`X \sim \text{Gamma}(\text{concentration}, 1)`.

    Marsaglia & Tsang's acceptance-rejection for :math:`\alpha \ge 1`; below
    that, the boost :math:`X = G_{\alpha+1}\,U^{1/\alpha}` — carried out on
    the logarithm, :math:`\log G_{\alpha+1} + \log U / \alpha`.  The power
    :math:`U^{1/\alpha}` itself falls under the smallest float32 once
    :math:`\alpha` is a few hundredths (a third of the draws at
    :math:`\alpha = 0.01`, nearly all at :math:`\alpha = 10^{-3}`), and a
    draw that rounds to ``0`` is then wrong everywhere it is used: a ratio of
    two of them is ``NaN``, ``log_prob`` is ``-inf``.  The logarithm is a
    modest number at any :math:`\alpha`, so callers that normalise
    (:class:`Beta`, :class:`Dirichlet`) can do it here before anything
    underflows.

    Detached — the engine has no reparameterisation kernel for Gamma.
    Implemented in pure Lucid with a bounded retry loop (rejection rate
    is < 5 % so 8 rounds is overkill).
    """
    out_shape = tuple(sample_shape) + tuple(concentration.shape)
    alpha = concentration
    tiny = float(finfo(alpha.dtype).tiny)
    # Branch for α ≥ 1; α < 1 uses the Gamma(α + 1) · U^{1/α} boost.
    boost = (alpha < 1.0).to(alpha.dtype)
    alpha_eff = alpha + boost  # Use α + 1 when α < 1.

    d = alpha_eff - 1.0 / 3.0
    c = 1.0 / (9.0 * d).sqrt()

    accepted = lucid.zeros(out_shape, dtype=alpha.dtype, device=alpha.device)
    has_sample = lucid.zeros(out_shape, dtype=alpha.dtype, device=alpha.device)

    for _ in range(_MAX_GAMMA_RETRIES):
        z = lucid.randn(*out_shape, dtype=alpha.dtype, device=alpha.device)
        v = (1.0 + c * z) ** 3
        u = lucid.rand(*out_shape, dtype=alpha.dtype, device=alpha.device)
        u = u.clip(1e-30, 1.0)
        # Acceptance: v > 0  AND  log(u) < 0.5·z² + d − d·v + d·log(v).
        v_safe = v.clip(1e-30, float("inf"))
        log_v = v_safe.log()
        condition = (v > 0) & (u.log() < 0.5 * z * z + d - d * v + d * log_v)
        new_accept = condition & (has_sample == 0)
        new_mask = new_accept.to(alpha.dtype)
        accepted = accepted + new_mask * d * v
        has_sample = has_sample + new_mask
        if _C_engine.compile.current_tracer() is None and bool((has_sample == 1).all().item()):
            break

    # A cell no round accepted (odds below 1e-10) takes d, near the mode of
    # Gamma(α_eff), rather than 0: the logarithm of 0 would be -inf.
    draw = lucid.where(has_sample > 0, accepted, d)
    # The α<1 boost x_α = x_{α+1} · U^{1/α}, as a sum of logarithms; a no-op
    # (boost = 0) when α ≥ 1.  α is floored so that 1/α stays finite.
    log_u = lucid.rand(*out_shape, dtype=alpha.dtype, device=alpha.device).clip(tiny, 1.0).log()
    alpha_floor = alpha.clip(max(tiny, 1e-30), float("inf"))
    return draw.log() + boost * log_u / alpha_floor


def _sample_standard_gamma(concentration: Tensor, sample_shape: tuple[int, ...]) -> Tensor:
    """Sample ``Gamma(concentration, 1)``, never below the smallest normal float.

    The draw is made in log space (:func:`_sample_standard_log_gamma`) and
    exponentiated, then held at ``finfo(dtype).tiny`` as the reference does:
    float32 cannot hold the mass a small ``concentration`` puts below it
    (42 % of ``Gamma(0.01)``), and an exact ``0`` there would turn every
    ratio, reciprocal and logarithm taken of the sample into ``NaN`` or
    ``inf``.
    """
    log_x = _sample_standard_log_gamma(concentration, sample_shape)
    return log_x.exp().clamp(min=float(finfo(concentration.dtype).tiny))


def _hold_in_range(x: Tensor, lo: float | None = None, hi: float | None = None) -> Tensor:
    """Clamp the *value* of ``x`` into ``[lo, hi]``, leaving its gradient alone.

    A reparameterised sample is held off the boundary the way the reference
    does it, by clamping a detached alias in place: the gradient is the one
    of the unclamped expression.  A plain ``clamp`` would zero the gradient
    of every clamped entry, which is most of them for a small concentration.
    """
    held = x.detach()
    return x + (held.clamp(min=lo, max=hi) - held)


def _hold_off_the_boundary(x: Tensor) -> Tensor:
    """``x`` held in ``[tiny, 1 - eps/2]``: the reference's range for Beta and Dirichlet samples.

    The upper end is the largest float below 1.  An exact ``0`` or ``1`` makes
    ``log_prob`` of the sample ``-inf`` (or ``NaN`` where the exponent is
    ``0``), so no sample may sit on either.
    """
    info = finfo(x.dtype)
    return _hold_in_range(x, float(info.tiny), 1.0 - float(info.eps) / 2.0)


_GAMMA_GRAD_COEF_UV: tuple[tuple[float, ...], ...] = (
    (
        0.16009398,
        -0.094634809,
        0.025146376,
        -0.0030648343,
        1.0,
        0.32668115,
        0.10406089,
        0.0014179084,
    ),
    (
        0.53487893,
        0.1298071,
        0.065735949,
        -0.0015649758,
        0.16639465,
        0.020070113,
        -0.0035938915,
        -0.00058392623,
    ),
    (
        0.040121004,
        -0.0065914022,
        -0.0026286047,
        -0.0013441777,
        0.017050642,
        -0.0021309326,
        0.00085092367,
        -1.5247877e-07,
    ),
)


def _standard_gamma_grad(alpha: Tensor, z: Tensor) -> Tensor:
    r"""Implicit-reparameterisation derivative :math:`\partial z / \partial \alpha = -(\partial P(\alpha, z) / \partial \alpha) / p(z; \alpha)`.

    Matches the three-regime approximation in ``standard_gamma_grad_one``
    (Jankowiak & Obermeyer, 2018):
      1. ``z < 0.8``: 6-term Taylor series with :math:`z^\alpha / z^{\alpha-1} = z`
         cancelled analytically.
      2. ``alpha > 8.0``: Rice saddle-point expansion (with Taylor series near
         :math:`0.9\alpha \le z \le 1.1\alpha`).
      3. Otherwise: bivariate rational approximation in
         :math:`u = \ln(z / \alpha)` and :math:`v = \ln \alpha`.
    """
    # Floored at the smallest normal float, which is where a sample is held:
    # a larger floor would price a sample of 1e-38 as one of 1e-30, a
    # derivative 1e8 times too large for the ratios taken of it.
    x = z.clamp(min=float(finfo(z.dtype).tiny))
    log_x = x.log()

    # 1. Small-x Taylor series (x < 0.8).
    numer = lucid.ones_like(x)
    denom = alpha
    series1 = numer / denom
    series2 = numer / (denom * denom)
    for i in range(1, 6):
        numer = numer * (-x / float(i))
        denom = denom + 1.0
        series1 = series1 + numer / denom
        series2 = series2 + numer / (denom * denom)
    small_x_res = (
        -x * x.clamp(max=20.0).exp() * ((log_x - lucid.digamma(alpha)) * series1 - series2)
    )

    # 2. Rice saddle-point expansion (alpha > 8.0).
    near_mode = (x >= 0.9 * alpha) & (x <= 1.1 * alpha)
    numer_1 = 1.0 + 24.0 * alpha * (1.0 + 12.0 * alpha)
    numer_2 = (
        1440.0 * (alpha * alpha)
        + 6.0 * x * (53.0 - 120.0 * x)
        - 65.0 * x * x / alpha
        + alpha * (107.0 + 3600.0 * x)
    )
    a2 = alpha * alpha
    rice_near = numer_1 * numer_2 / (1244160.0 * a2 * a2)

    denom_r = (8.0 * alpha).sqrt()
    diff = alpha - x
    diff_safe = lucid.where(diff.abs() < 1e-6, lucid.full_like(diff, 1e-6), diff)
    term2 = denom_r / diff_safe
    u_ratio = log_x - alpha.log()
    rad = (x - alpha - alpha * u_ratio).clamp(min=1e-12)
    term3 = rad.pow(-1.5)
    term23 = lucid.where(x < alpha, term2 - term3, term2 + term3)
    term1 = u_ratio * term23 - (2.0 / alpha).sqrt() * (alpha + x) / (diff_safe * diff_safe)
    stirling = 1.0 + (1.0 / (12.0 * alpha)) * (1.0 + 1.0 / (24.0 * alpha))
    rice_far = -stirling * (x * term1) / denom_r
    rice_res = lucid.where(near_mode, rice_near, rice_far)

    # 3. Bivariate rational approximation (x >= 0.8 and alpha <= 8.0).
    u = u_ratio
    v = alpha.log()
    c0, c1, c2 = _GAMMA_GRAD_COEF_UV
    cv = [c0[i] + u * (c1[i] + u * c2[i]) for i in range(8)]
    p = cv[0] + v * (cv[1] + v * (cv[2] + v * cv[3]))
    q = cv[4] + v * (cv[5] + v * (cv[6] + v * cv[7]))
    rat_res = (p / q).exp()

    out = lucid.where(alpha > 8.0, rice_res, rat_res)
    out = lucid.where(x < 0.8, small_x_res, out)
    return lucid.where(lucid.isnan(out), lucid.zeros_like(out), out)


class _StandardGamma(Function):
    """Differentiable standard Gamma sampler with implicit-reparameterisation backward."""

    _traced_backward = True

    @staticmethod
    @override
    def forward(ctx: FunctionCtx, concentration: Tensor) -> Tensor:
        with lucid.no_grad():
            sample = _sample_standard_gamma(concentration, ())
        ctx.save_for_backward(concentration, sample)
        return sample

    @staticmethod
    @override
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tensor:
        concentration, sample = ctx.saved_tensors
        return grad_output * _standard_gamma_grad(concentration, sample)


def _standard_gamma(concentration: Tensor) -> Tensor:
    """Draw reparameterised samples from ``Gamma(concentration, 1)``."""
    out = _StandardGamma.apply(concentration)
    assert isinstance(out, Tensor)
    return out


class Gamma(ExponentialFamily):
    r"""Gamma distribution on :math:`[0, \infty)` — shape/rate parameterisation.

    Two-parameter continuous distribution that generalises the
    :class:`~lucid.distributions.Exponential` (shape :math:`\alpha = 1`)
    and :class:`~lucid.distributions.Chi2` (shape :math:`k/2`, rate
    :math:`1/2`).  Widely used as the **conjugate prior** for the rate
    parameter of a Poisson likelihood and for positive-valued
    measurements such as waiting times, life-times, and rainfall amounts.

    Parameters
    ----------
    concentration : Tensor or float
        Shape (concentration) parameter :math:`\alpha > 0`.  Sometimes
        called :math:`k`.  When :math:`\alpha` is a positive integer the
        Gamma equals the **Erlang** distribution (sum of :math:`\alpha`
        IID Exponentials).
    rate : Tensor or float
        Rate (inverse-scale) parameter :math:`\beta > 0`.  Equivalent
        scale parameterisation: :math:`\theta = 1/\beta`.
    validate_args : bool, optional
        If ``True``, validate parameter constraints at construction time.

    Notes
    -----
    Probability density on :math:`x > 0`:

    .. math::

        p(x; \alpha, \beta) =
            \frac{\beta^\alpha}{\Gamma(\alpha)}
            x^{\alpha - 1} e^{-\beta x}

    Moments:

    .. math::

        \mathbb{E}[X] = \frac{\alpha}{\beta}, \qquad
        \mathrm{Var}[X] = \frac{\alpha}{\beta^2}, \qquad
        \mathrm{Mode} = \frac{\max(\alpha - 1, 0)}{\beta}

    Entropy:

    .. math::

        H[X] = \alpha - \log \beta + \log \Gamma(\alpha)
        + (1 - \alpha)\psi(\alpha)

    where :math:`\psi` is the digamma function.

    Special cases and relations:

    * :math:`\mathrm{Gamma}(1, \beta) = \mathrm{Exponential}(\beta)`
    * :math:`\mathrm{Gamma}(k/2, 1/2) = \chi^2(k)`
    * Sum of independent Gammas with the same rate adds shapes:
      :math:`\mathrm{Gamma}(\alpha_1, \beta) + \mathrm{Gamma}(\alpha_2, \beta)
      = \mathrm{Gamma}(\alpha_1 + \alpha_2, \beta)`.
    * The ratio :math:`G_\alpha/(G_\alpha + G_\beta)` is
      :math:`\mathrm{Beta}(\alpha, \beta)`.

    Sampling uses Marsaglia–Tsang acceptance-rejection — accept rate is
    consistently above 95 %, so eight retry rounds are more than enough.
    Because the kernel is rejection-based, ``has_rsample = False`` and
    samples are detached.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import Gamma
    >>> d = Gamma(concentration=2.0, rate=1.0)
    >>> d.mean  # α/β
    tensor(2.)
    >>> d.sample((4,)).shape
    (4,)
    >>> d.log_prob(lucid.tensor(1.5))  # log(1.5) - 1.5
    tensor(-1.095)
    """

    arg_constraints = {"concentration": positive, "rate": positive}
    # ``[0, ∞)``, as in the reference framework: at 0 the density is
    # finite (``α = 1``), infinite (``α < 1``) or 0 (``α > 1``), and
    # :meth:`log_prob` scores all three.  ``Chi2`` inherits it.
    support: Constraint | None = nonnegative
    has_rsample = True

    def __init__(
        self, concentration: Tensor | float, rate: Tensor | float, validate_args: bool | None = None
    ) -> None:
        r"""Construct a Gamma distribution.

        Parameters
        ----------
        concentration : Tensor | float
            Shape (concentration) parameter :math:`\alpha > 0`.  Also known
            as the shape parameter :math:`k`.
        rate : Tensor | float
            Rate (inverse-scale) parameter :math:`\beta > 0`.  The scale
            is :math:`\theta = 1/\beta`.
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.

        Notes
        -----
        The Gamma distribution has PDF:

        .. math::

            p(x; \alpha, \beta) = \frac{\beta^\alpha}{\Gamma(\alpha)}
            x^{\alpha-1} e^{-\beta x}, \quad x > 0

        Special cases include:

        - :math:`\text{Gamma}(1, \beta) = \text{Exponential}(\beta)`
        - :math:`\text{Gamma}(k/2, 1/2) = \chi^2(k)`

        Sampling uses the Marsaglia–Tsang acceptance-rejection algorithm
        (see module docstring); the accept rate is > 95 % so eight retry
        rounds are more than sufficient.

        Examples
        --------
        >>> from lucid.distributions import Gamma
        >>> d = Gamma(concentration=2.0, rate=1.0)
        >>> d.mean  # α/β = 2.0
        tensor(2.)
        """
        self.concentration = _as_tensor(concentration)
        self.rate = _as_tensor(rate)
        self.concentration, self.rate = _broadcast_pair(self.concentration, self.rate)
        super().__init__(
            batch_shape=tuple(self.concentration.shape), event_shape=(), validate_args=validate_args
        )

    @override
    def expand(self, batch_shape: Sequence[int], _instance: Self | None = None) -> Self:
        """Return a new :class:`Gamma` with batch dimensions expanded to ``batch_shape``."""
        return self._expand_attributes(Gamma, batch_shape, _instance, "concentration", "rate")

    @override
    @property
    def mean(self) -> Tensor:
        r"""Expected value of the Gamma distribution.

        .. math::

            E[X] = \frac{\alpha}{\beta}

        Returns
        -------
        Tensor
            Mean :math:`\alpha/\beta`, shape ``batch_shape``.

        Examples
        --------
        >>> Gamma(concentration=3.0, rate=2.0).mean
        tensor(1.5)
        """
        return self.concentration / self.rate

    @override
    @property
    def mode(self) -> Tensor:
        r"""Mode of the Gamma distribution.

        .. math::

            \text{mode} = \frac{\max(\alpha - 1,\; 0)}{\beta}

        The mode is zero for :math:`\alpha \leq 1` (the density is
        monotonically decreasing from infinity) and positive for
        :math:`\alpha > 1`.

        Returns
        -------
        Tensor
            Mode :math:`\max(\alpha-1, 0)/\beta`, shape ``batch_shape``.

        Examples
        --------
        >>> Gamma(concentration=3.0, rate=1.0).mode
        tensor(2.)
        """
        return ((self.concentration - 1.0).clip(0.0, float("inf"))) / self.rate

    @override
    @property
    def variance(self) -> Tensor:
        r"""Variance of the Gamma distribution.

        .. math::

            \operatorname{Var}[X] = \frac{\alpha}{\beta^2}

        Returns
        -------
        Tensor
            Variance :math:`\alpha/\beta^2`, shape ``batch_shape``.

        Examples
        --------
        >>> Gamma(concentration=4.0, rate=2.0).variance
        tensor(1.)
        """
        return self.concentration / (self.rate * self.rate)

    @override
    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        r"""Draw reparameterised samples from the Gamma distribution via implicit reparameterisation."""
        shape = self._extended_shape(sample_shape)
        value = _standard_gamma(self.concentration.expand(shape)) / self.rate.expand(shape)
        return _hold_in_range(value, float(finfo(value.dtype).tiny))

    @override
    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        r"""Draw non-reparameterised samples from the Gamma distribution via ``_sample_standard_gamma``.

        Examples
        --------
        >>> lucid.manual_seed(0)
        >>> d = Gamma(concentration=2.0, rate=1.0)
        >>> x = d.sample((500,))
        >>> x.shape
        (500,)
        >>> bool((x.mean() - 2.0).abs() < 0.2)  # the sample mean approaches α/β
        True
        """
        with lucid.no_grad():
            std = _sample_standard_gamma(self.concentration, sample_shape)
            value = (std / self.rate).detach()
            return value.clamp(min=float(finfo(value.dtype).tiny))

    @override
    def log_prob(self, value: Tensor) -> Tensor:
        r"""Log-density of ``value`` under the Gamma distribution.

        .. math::

            \log p(x; \alpha, \beta) = \alpha \log \beta
            + (\alpha - 1) \log x - \beta x - \log \Gamma(\alpha)

        :math:`(\alpha - 1)\log x` is taken as ``0`` when :math:`\alpha = 1`,
        so the density at :math:`x = 0` is the finite :math:`\beta` of the
        exponential rather than ``0 · (-inf) = NaN``.

        Parameters
        ----------
        value : Tensor
            Non-negative real values :math:`x \geq 0`.

        Returns
        -------
        Tensor
            Element-wise log-densities, shape ``batch_shape``.

        Examples
        --------
        >>> Gamma(concentration=1.0, rate=1.0).log_prob(lucid.tensor(1.0))
        tensor(-1.)
        """
        return (
            self.concentration * self.rate.log()
            + lucid.xlogy(self.concentration - 1.0, value)
            - self.rate * value
            - lucid.lgamma(self.concentration)
        )

    @override
    def cdf(self, value: Tensor) -> Tensor:
        r"""Cumulative distribution function of the Gamma distribution.

        .. math::

            F(x; \alpha, \beta) = P(\alpha, \beta x), \quad x \geq 0

        the regularized lower incomplete gamma
        (:func:`lucid.special.gammainc`).  ``Chi2`` inherits it.

        Returns probabilities in :math:`[0, 1]`; 0 below 0.

        Raises
        ------
        NotImplementedError
            From ``backward``, when ``concentration`` requires a gradient —
            :math:`P` is differentiable in its upper limit only, as in the
            reference framework.  ``rate`` and ``value`` are differentiable.

        Examples
        --------
        >>> Gamma(concentration=1.0, rate=1.0).cdf(lucid.tensor(1.0))  # 1 - 1/e
        tensor(0.6321)
        """
        # Clamped: P is NaN for a negative limit, and a CDF is 0 left of
        # its support.
        return gammainc(self.concentration, self.rate * value.clamp(min=0.0))

    @override
    def entropy(self) -> Tensor:
        r"""Shannon entropy of the Gamma distribution (in nats).

        .. math::

            H(X) = \alpha - \log \beta + \log \Gamma(\alpha)
            + (1 - \alpha) \psi(\alpha)

        where :math:`\psi` is the digamma function.

        Returns
        -------
        Tensor
            Entropy in nats, shape ``batch_shape``.

        Examples
        --------
        >>> Gamma(concentration=1.0, rate=1.0).entropy()  # Exp(1): H = 1
        tensor(1.)
        """
        return (
            self.concentration
            - self.rate.log()
            + lucid.lgamma(self.concentration)
            + (1.0 - self.concentration) * lucid.digamma(self.concentration)
        )

    @override
    @property
    def _natural_params(self) -> tuple[Tensor, Tensor]:
        return (self.concentration - 1.0, -self.rate)

    @override
    def _log_normalizer(self, *natural_params: Tensor) -> Tensor:
        x, y = natural_params
        return lucid.lgamma(x + 1.0) + (x + 1.0) * (-1.0 / y).log()


class Chi2(Gamma):
    r"""Chi-squared distribution with :math:`k` degrees of freedom.

    Distribution of the sum of squares of :math:`k` independent standard
    Normal random variables.  Foundational in classical statistics:
    arises in the likelihood-ratio test (Wilks' theorem), Pearson's
    goodness-of-fit, confidence intervals for variance, and many F-test
    /t-test derivations.

    Equivalent to a Gamma with shape :math:`k/2` and rate :math:`1/2`:
    :math:`\chi^2(k) = \mathrm{Gamma}(k/2,\, 1/2)`.

    Parameters
    ----------
    df : Tensor or float
        Degrees of freedom :math:`k > 0`.  Need not be an integer.
    validate_args : bool, optional
        If ``True``, validate parameter constraints at construction time.

    Notes
    -----
    Probability density on :math:`x > 0`:

    .. math::

        p(x; k) = \frac{1}{2^{k/2} \Gamma(k/2)}
                  x^{k/2 - 1} e^{-x/2}

    Moments:

    .. math::

        \mathbb{E}[X] = k, \qquad
        \mathrm{Var}[X] = 2k, \qquad
        \mathrm{Mode} = \max(k - 2, 0)

    Special cases:

    * :math:`k = 1` → squared standard Normal.
    * :math:`k = 2` → :math:`\mathrm{Exponential}(1/2)`.
    * Sum: :math:`\chi^2(k_1) + \chi^2(k_2) \sim \chi^2(k_1 + k_2)`.

    By the central limit theorem,
    :math:`(\chi^2(k) - k)/\sqrt{2k} \to \mathcal{N}(0, 1)` as
    :math:`k \to \infty`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import Chi2
    >>> d = Chi2(df=4.0)
    >>> d.mean
    tensor(4.)
    >>> d.sample((4,)).shape
    (4,)
    """

    arg_constraints = {"df": positive}

    def __init__(self, df: Tensor | float, validate_args: bool | None = None) -> None:
        r"""Construct a Chi-squared distribution.

        Parameters
        ----------
        df : Tensor | float
            Degrees of freedom :math:`k > 0`.
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.

        Notes
        -----
        :math:`\chi^2(k)` is a special case of the Gamma distribution:

        .. math::

            \chi^2(k) = \text{Gamma}\!\left(\frac{k}{2},\; \frac{1}{2}\right)

        It arises as the distribution of the sum of squares of :math:`k`
        independent standard Normal variables, and is fundamental in
        hypothesis testing (e.g., goodness-of-fit tests, likelihood-ratio
        tests).

        Examples
        --------
        >>> from lucid.distributions import Chi2
        >>> d = Chi2(df=4.0)
        >>> d.mean  # k = 4
        tensor(4.)
        """
        self.df = _as_tensor(df)
        super().__init__(
            concentration=self.df * 0.5,
            # ``_as_tensor(0.5)`` produces a 0-D scalar; using
            # ``lucid.tensor(0.5)`` would yield a (1,) tensor and force
            # an unwanted broadcast that propagates into the batch shape.
            rate=_as_tensor(0.5),
            validate_args=validate_args,
        )

    @override
    def expand(self, batch_shape: Sequence[int], _instance: Self | None = None) -> Self:
        """Return a new :class:`Chi2` with batch dimensions expanded to ``batch_shape``."""
        new = self._get_checked_instance(Chi2, _instance)
        batch_shape_t = tuple(batch_shape)
        new.df = self.df.expand(batch_shape_t)
        return super().expand(batch_shape_t, _instance=new)


class Beta(ExponentialFamily):
    r"""Beta distribution on the open interval :math:`(0, 1)`.

    Two-parameter continuous distribution that is the **conjugate prior**
    of the :class:`~lucid.distributions.Bernoulli` / Binomial likelihoods.
    When :math:`\alpha = \beta = 1` it reduces to a Uniform on
    :math:`(0, 1)`; for large :math:`\alpha + \beta` it concentrates
    around :math:`\alpha/(\alpha + \beta)`.

    Parameters
    ----------
    concentration1 : Tensor or float
        First shape parameter :math:`\alpha > 0` (sometimes called *a*).
    concentration0 : Tensor or float
        Second shape parameter :math:`\beta > 0` (sometimes called *b*).
    validate_args : bool, optional
        If ``True``, validate parameter constraints at construction time.

    Notes
    -----
    Probability density on :math:`x \in (0, 1)`:

    .. math::

        p(x; \alpha, \beta) =
            \frac{x^{\alpha - 1} (1 - x)^{\beta - 1}}{B(\alpha, \beta)}

    where :math:`B(\alpha, \beta) = \Gamma(\alpha)\Gamma(\beta)/\Gamma(\alpha + \beta)`.

    Moments:

    .. math::

        \mathbb{E}[X] = \frac{\alpha}{\alpha + \beta}, \qquad
        \mathrm{Var}[X] = \frac{\alpha \beta}
                               {(\alpha + \beta)^2 (\alpha + \beta + 1)}

    Mode: :math:`(\alpha - 1) / (\alpha + \beta - 2)` for
    :math:`\alpha, \beta > 1`.

    Special cases / shapes:

    * :math:`\alpha = \beta = 1` → :math:`\mathrm{Uniform}(0, 1)`.
    * :math:`\alpha = \beta` → symmetric around :math:`0.5`.
    * :math:`\alpha, \beta \to \infty` with fixed mean → concentrates as
      a Gaussian about the mean.
    * Bimodal (U-shaped) when both :math:`\alpha, \beta < 1`.

    **Conjugacy**: observing :math:`k` successes out of :math:`n`
    Bernoulli trials updates ``Beta(α, β) → Beta(α + k, β + n - k)``.

    Sampling uses the ratio-of-Gammas method:
    :math:`X = G_\alpha / (G_\alpha + G_\beta)` with independent
    :math:`G_\alpha \sim \mathrm{Gamma}(\alpha, 1)`,
    :math:`G_\beta \sim \mathrm{Gamma}(\beta, 1)`.  Since the underlying
    Gamma sampler is rejection-based, samples are detached.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import Beta
    >>> d = Beta(concentration1=2.0, concentration0=5.0)
    >>> d.mean  # 2/(2+5) ≈ 0.2857
    tensor(0.2857)
    >>> d.sample((4,)).shape
    (4,)
    >>> d.log_prob(lucid.tensor(0.3))  # log(30 * 0.3 * 0.7**4)
    tensor(0.7705)
    """

    arg_constraints = {"concentration1": positive, "concentration0": positive}
    support: Constraint | None = unit_interval
    has_rsample = True

    def __init__(
        self,
        concentration1: Tensor | float,
        concentration0: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        r"""Construct a Beta distribution.

        Parameters
        ----------
        concentration1 : Tensor | float
            First shape parameter :math:`\alpha > 0` (also called
            ``concentration1``).  Controls how much mass is near 1.
        concentration0 : Tensor | float
            Second shape parameter :math:`\beta > 0` (also called
            ``concentration0``).  Controls how much mass is near 0.
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.

        Notes
        -----
        The Beta distribution has PDF:

        .. math::

            p(x; \alpha, \beta) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)},
            \quad x \in [0, 1]

        where :math:`B(\alpha, \beta) = \Gamma(\alpha)\Gamma(\beta)/\Gamma(\alpha+\beta)`.

        Sampling uses the ratio-of-Gammas method:
        :math:`X = G_\alpha / (G_\alpha + G_\beta)` where
        :math:`G_\alpha \sim \text{Gamma}(\alpha, 1)`.

        Examples
        --------
        >>> from lucid.distributions import Beta
        >>> d = Beta(concentration1=2.0, concentration0=5.0)
        >>> d.mean  # α/(α+β) = 2/7 ≈ 0.286
        tensor(0.2857)
        """
        self.concentration1 = _as_tensor(concentration1)
        self.concentration0 = _as_tensor(concentration0)
        self.concentration1, self.concentration0 = _broadcast_pair(
            self.concentration1, self.concentration0
        )
        super().__init__(
            batch_shape=tuple(self.concentration1.shape),
            event_shape=(),
            validate_args=validate_args,
        )

    @override
    def expand(self, batch_shape: Sequence[int], _instance: Self | None = None) -> Self:
        """Return a new :class:`Beta` with batch dimensions expanded to ``batch_shape``."""
        return self._expand_attributes(
            Beta, batch_shape, _instance, "concentration1", "concentration0"
        )

    @override
    @property
    def mean(self) -> Tensor:
        r"""Expected value of the Beta distribution.

        .. math::

            E[X] = \frac{\alpha}{\alpha + \beta}

        Returns
        -------
        Tensor
            Mean :math:`\alpha/(\alpha+\beta)`, shape ``batch_shape``.

        Examples
        --------
        >>> Beta(concentration1=1.0, concentration0=1.0).mean
        tensor(0.5)
        """
        return self.concentration1 / (self.concentration1 + self.concentration0)

    @override
    @property
    def mode(self) -> Tensor:
        r"""Mode of the Beta distribution.

        .. math::

            \frac{\alpha - 1}{\alpha + \beta - 2} \quad (\alpha, \beta \geq 1)

        with each excess :math:`\alpha - 1`, :math:`\beta - 1` floored at 0,
        so one concentration below 1 puts the mode at that end.  With both
        below 1 the density is unbounded at both ends and ``1`` is returned;
        at :math:`\alpha = \beta = 1` (uniform) the answer is ``nan`` — both
        as the reference framework answers.
        """
        excess1 = (self.concentration1 - 1.0).clip(0.0, None)
        excess0 = (self.concentration0 - 1.0).clip(0.0, None)
        interior = excess1 / (excess1 + excess0)
        both_below = (self.concentration1 < 1.0) & (self.concentration0 < 1.0)
        return lucid.where(both_below, 1.0, interior)

    @override
    @property
    def variance(self) -> Tensor:
        r"""Variance of the Beta distribution.

        .. math::

            \operatorname{Var}[X] =
            \frac{\alpha \beta}{(\alpha+\beta)^2 (\alpha+\beta+1)}

        Returns
        -------
        Tensor
            Variance, shape ``batch_shape``.

        Examples
        --------
        >>> Beta(concentration1=2.0, concentration0=2.0).variance
        tensor(0.05)
        """
        a = self.concentration1
        b = self.concentration0
        ab = a + b
        return (a * b) / (ab * ab * (ab + 1.0))

    @override
    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        r"""Draw reparameterised samples from the Beta distribution.

        Uses :math:`X = G_\alpha / (G_\alpha + G_\beta)` where
        :math:`G_\alpha \sim \text{Gamma}(\alpha, 1)` and
        :math:`G_\beta \sim \text{Gamma}(\beta, 1)` carry implicit-reparameterisation
        gradients.
        """
        shape = self._extended_shape(sample_shape)
        pair = lucid.stack(
            [self.concentration1.expand(shape), self.concentration0.expand(shape)], dim=0
        )
        g = _standard_gamma(pair)
        # g[0] / (g[0] + g[1]) as a sigmoid of the logs: both draws can sit
        # at the float floor, and the quotient's backward then divides by a
        # square that underflows to 0.
        return _hold_off_the_boundary(lucid.sigmoid(g[0].log() - g[1].log()))

    @override
    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        r"""Draw samples from the Beta distribution.

        Uses the ratio-of-independent-Gamma-samples identity:

        .. math::

            X = \frac{G_\alpha}{G_\alpha + G_\beta},
            \quad G_\alpha \sim \text{Gamma}(\alpha, 1),\;
            G_\beta \sim \text{Gamma}(\beta, 1)

        Parameters
        ----------
        sample_shape : tuple[int, ...], optional
            Leading shape dimensions for the sample batch.  Default is ``()``.

        Returns
        -------
        Tensor
            Samples of shape ``sample_shape + batch_shape``, held in
            ``[tiny, 1 - eps/2]`` — off both ends of the unit interval, so
            ``log_prob`` of a draw is finite — where ``tiny`` and ``eps`` are
            the dtype's smallest normal number and machine epsilon.

        Examples
        --------
        >>> lucid.manual_seed(0)
        >>> d = Beta(concentration1=2.0, concentration0=5.0)
        >>> x = d.sample((500,))
        >>> x.shape
        (500,)
        >>> bool((x.mean() - 2 / 7).abs() < 0.03)  # the sample mean approaches 2/7
        True
        """
        with lucid.no_grad():
            # X = sigmoid(log G_a - log G_b) is the same ratio, taken before
            # either Gamma draw can round to 0 (a/(a+b) is NaN where both do).
            log_x = _sample_standard_log_gamma(self.concentration1, sample_shape)
            log_y = _sample_standard_log_gamma(self.concentration0, sample_shape)
            return _hold_off_the_boundary(lucid.sigmoid(log_x - log_y)).detach()

    @override
    def log_prob(self, value: Tensor) -> Tensor:
        r"""Log-density of ``value`` under the Beta distribution.

        .. math::

            \log p(x; \alpha, \beta) =
            (\alpha - 1) \log x + (\beta - 1) \log(1-x)
            - \log B(\alpha, \beta)

        Each power term is taken as ``0`` when its exponent is, so
        ``Beta(1, 1)`` scores both endpoints as :math:`\log 1 = 0` rather
        than ``0 · (-inf) = NaN``.

        Parameters
        ----------
        value : Tensor
            Values in :math:`[0, 1]`.

        Returns
        -------
        Tensor
            Element-wise log-densities, shape ``batch_shape``.
        """
        a = self.concentration1
        b = self.concentration0
        log_b = lucid.lgamma(a) + lucid.lgamma(b) - lucid.lgamma(a + b)
        return lucid.xlogy(a - 1.0, value) + xlog1py(b - 1.0, -value) - log_b

    @override
    def entropy(self) -> Tensor:
        r"""Shannon entropy of the Beta distribution (in nats).

        .. math::

            H(X) = \log B(\alpha,\beta)
            - (\alpha-1)\psi(\alpha)
            - (\beta-1)\psi(\beta)
            + (\alpha+\beta-2)\psi(\alpha+\beta)

        where :math:`\psi` is the digamma function and
        :math:`B(\alpha,\beta) = \Gamma(\alpha)\Gamma(\beta)/\Gamma(\alpha+\beta)`.

        Returns
        -------
        Tensor
            Entropy in nats, shape ``batch_shape``.
        """
        a = self.concentration1
        b = self.concentration0
        ab = a + b
        log_b = lucid.lgamma(a) + lucid.lgamma(b) - lucid.lgamma(ab)
        return (
            log_b
            - (a - 1.0) * lucid.digamma(a)
            - (b - 1.0) * lucid.digamma(b)
            + (ab - 2.0) * lucid.digamma(ab)
        )

    @override
    @property
    def _natural_params(self) -> tuple[Tensor, Tensor]:
        return (self.concentration1, self.concentration0)

    @override
    def _log_normalizer(self, *natural_params: Tensor) -> Tensor:
        x, y = natural_params
        return lucid.lgamma(x) + lucid.lgamma(y) - lucid.lgamma(x + y)

    @override
    @property
    def _mean_carrier_measure(self) -> Tensor:
        a = self.concentration1
        b = self.concentration0
        return 2.0 * lucid.digamma(a + b) - lucid.digamma(a) - lucid.digamma(b)


class Dirichlet(ExponentialFamily):
    r"""Dirichlet distribution on the :math:`K`-simplex.

    Multivariate generalisation of the :class:`~lucid.distributions.Beta`
    distribution: a distribution over probability vectors
    :math:`\mathbf{x} \in \Delta^{K-1}` with :math:`x_i \geq 0` and
    :math:`\sum_i x_i = 1`.  It is the **conjugate prior** of the
    :class:`~lucid.distributions.Categorical` /
    :class:`~lucid.distributions.Multinomial` likelihoods and a foundational
    building block in topic models (LDA), Bayesian mixture models, and
    population genetics.

    The last dimension of ``concentration`` is the simplex/event
    dimension; all preceding dimensions form the batch shape.

    Parameters
    ----------
    concentration : Tensor
        Concentration vector :math:`\boldsymbol{\alpha}` with all
        entries :math:`\alpha_i > 0`.  Shape ``(..., K)``.
    validate_args : bool, optional
        If ``True``, validate parameter constraints at construction time.

    Notes
    -----
    Probability density on the :math:`K`-simplex
    (:math:`x_i > 0,\;\sum_i x_i = 1`):

    .. math::

        p(\mathbf{x}; \boldsymbol{\alpha}) =
            \frac{1}{B(\boldsymbol{\alpha})}
            \prod_{i=1}^{K} x_i^{\alpha_i - 1},
        \qquad
        B(\boldsymbol{\alpha}) =
            \frac{\prod_i \Gamma(\alpha_i)}{\Gamma(\alpha_0)},
        \quad \alpha_0 = \sum_i \alpha_i

    Moments (with :math:`\alpha_0 = \sum_i \alpha_i`,
    :math:`\mu_i = \alpha_i/\alpha_0`):

    .. math::

        \mathbb{E}[X_i] = \mu_i, \qquad
        \mathrm{Var}[X_i] = \frac{\mu_i (1 - \mu_i)}{\alpha_0 + 1},
        \qquad
        \mathrm{Cov}[X_i, X_j] = -\frac{\mu_i \mu_j}{\alpha_0 + 1}
        \;\;(i \neq j)

    Special cases:

    * :math:`K = 2` → :math:`\mathrm{Beta}(\alpha_1, \alpha_2)`
      (after dropping one redundant coordinate).
    * :math:`\boldsymbol{\alpha} = \mathbf{1}` → uniform over the
      simplex.
    * :math:`\alpha_0 \to \infty` with fixed
      :math:`\boldsymbol{\mu}` → mass concentrates at
      :math:`\boldsymbol{\mu}`.

    **Conjugacy**: observing categorical counts
    :math:`\mathbf{n} = (n_1, \ldots, n_K)` updates
    :math:`\mathrm{Dirichlet}(\boldsymbol{\alpha}) \to
    \mathrm{Dirichlet}(\boldsymbol{\alpha} + \mathbf{n})`.

    Sampling uses the normalised-Gamma method: draw independent
    :math:`G_i \sim \mathrm{Gamma}(\alpha_i, 1)` and set
    :math:`X_i = G_i / \sum_j G_j`.  Samples are detached.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import Dirichlet
    >>> d = Dirichlet(lucid.tensor([1.0, 2.0, 3.0]))
    >>> d.mean  # α / Σ α
    tensor([0.1667, 0.3333, 0.5])
    >>> d.sample((4,)).shape
    (4, 3)
    """

    arg_constraints = {"concentration": independent(positive, 1)}
    support: Constraint | None = simplex
    has_rsample = True

    def __init__(self, concentration: Tensor, validate_args: bool | None = None) -> None:
        r"""Construct a Dirichlet distribution.

        Parameters
        ----------
        concentration : Tensor
            Concentration vector :math:`\boldsymbol{\alpha}` with all
            entries :math:`> 0`.  The last dimension is the event (simplex)
            dimension :math:`K`; all preceding dimensions form the batch
            shape.
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.

        Notes
        -----
        The Dirichlet distribution with concentration :math:`\boldsymbol{\alpha}`
        has PDF over the :math:`K`-simplex:

        .. math::

            p(\mathbf{x}; \boldsymbol{\alpha}) =
            \frac{1}{B(\boldsymbol{\alpha})}
            \prod_{i=1}^{K} x_i^{\alpha_i - 1}

        where
        :math:`B(\boldsymbol{\alpha}) = \prod_i \Gamma(\alpha_i) / \Gamma(\sum_i \alpha_i)`.

        Sampling uses the normalised-Gamma trick: draw independent
        :math:`G_i \sim \text{Gamma}(\alpha_i, 1)` then return
        :math:`\mathbf{x} = \mathbf{G} / \sum_i G_i`.

        Examples
        --------
        >>> import lucid
        >>> from lucid.distributions import Dirichlet
        >>> d = Dirichlet(lucid.tensor([1.0, 2.0, 3.0]))
        >>> d.mean  # proportional to concentration
        tensor([0.1667, 0.3333, 0.5])
        """
        self.concentration = _as_tensor(concentration)
        shape = tuple(self.concentration.shape)
        super().__init__(
            batch_shape=shape[:-1], event_shape=shape[-1:], validate_args=validate_args
        )

    @override
    def expand(self, batch_shape: Sequence[int], _instance: Self | None = None) -> Self:
        """Return a new :class:`Dirichlet` with batch dimensions expanded to ``batch_shape``."""
        new = self._get_checked_instance(Dirichlet, _instance)
        batch_shape_t = tuple(batch_shape)
        new.concentration = self.concentration.expand(batch_shape_t + self._event_shape)
        return self._finish_expand(new, batch_shape_t, self._event_shape)

    @override
    @property
    def mean(self) -> Tensor:
        r"""Expected value of the Dirichlet distribution.

        Each component of the mean equals the normalised concentration:

        .. math::

            E[X_i] = \frac{\alpha_i}{\sum_j \alpha_j}

        Returns
        -------
        Tensor
            Mean vector on the simplex, shape ``batch_shape + event_shape``.

        Examples
        --------
        >>> Dirichlet(lucid.tensor([2.0, 2.0])).mean
        tensor([0.5, 0.5])
        """
        s = self.concentration.sum(dim=-1, keepdim=True)
        return self.concentration / s

    @override
    @property
    def mode(self) -> Tensor:
        r"""Mode of the Dirichlet distribution.

        .. math::

            \frac{\alpha_i - 1}{\sum_j (\alpha_j - 1)}

        with each excess floored at 0.  With every concentration below 1 the
        density is unbounded at every vertex and the first vertex
        :math:`e_0` is returned; with all equal to 1 (uniform on the
        simplex) the answer is ``nan`` — both as the reference framework
        answers.
        """
        excess = (self.concentration - 1.0).clip(0.0, None)
        interior = excess / excess.sum(dim=-1, keepdim=True)
        all_below = (self.concentration < 1.0).min(dim=-1).values.unsqueeze(-1)
        k = int(self.concentration.shape[-1])
        first_vertex = (lucid.arange(k, device=self.concentration.device) == 0).to(interior.dtype)
        return lucid.where(all_below, first_vertex, interior)

    @override
    @property
    def variance(self) -> Tensor:
        r"""Variance of the Dirichlet distribution (component-wise).

        .. math::

            \operatorname{Var}[X_i] =
            \frac{\mu_i (1 - \mu_i)}{\alpha_0 + 1},
            \quad \alpha_0 = \sum_j \alpha_j,\;
            \mu_i = \alpha_i / \alpha_0

        Returns
        -------
        Tensor
            Variance vector, shape ``batch_shape + event_shape``.
        """
        a = self.concentration
        s = a.sum(dim=-1, keepdim=True)
        m = a / s
        return m * (1.0 - m) / (s + 1.0)

    @override
    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        r"""Draw reparameterised samples from the Dirichlet distribution.

        Uses :math:`X_i = G_i / \sum_j G_j` where :math:`G_i \sim \text{Gamma}(\alpha_i, 1)`
        carry implicit-reparameterisation gradients.
        """
        shape = self._extended_shape(sample_shape)
        log_g = _standard_gamma(self.concentration.expand(shape)).log()
        # A softmax of the logs, not gammas / gammas.sum(): at the float floor
        # the quotient's backward divides by a square that underflows to 0.
        weights = (log_g - log_g.amax(dim=-1, keepdim=True).detach()).exp()
        return _hold_off_the_boundary(weights / weights.sum(dim=-1, keepdim=True))

    @override
    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        r"""Draw non-reparameterised simplex samples from the Dirichlet distribution.

        Examples
        --------
        >>> d = Dirichlet(lucid.tensor([1.0, 1.0, 1.0]))
        >>> x = d.sample((100,))
        >>> x.shape
        (100, 3)
        >>> bool(((x.sum(dim=-1) - 1.0).abs() < 1e-5).all())  # every row sums to one
        True
        """
        with lucid.no_grad():
            # Normalised as a softmax of the log-draws: the Gamma draws
            # themselves can all round to 0, and 0/0 is a NaN row.
            log_g = _sample_standard_log_gamma(self.concentration, sample_shape)
            weights = (log_g - log_g.amax(dim=-1, keepdim=True)).exp()
            return _hold_off_the_boundary(weights / weights.sum(dim=-1, keepdim=True)).detach()

    @override
    def log_prob(self, value: Tensor) -> Tensor:
        r"""Log-density of ``value`` under the Dirichlet distribution.

        .. math::

            \log p(\mathbf{x}; \boldsymbol{\alpha}) =
            \sum_i (\alpha_i - 1) \log x_i - \log B(\boldsymbol{\alpha})

        A term with :math:`\alpha_i = 1` is ``0`` even where
        :math:`x_i = 0`, on the boundary of the simplex.

        Parameters
        ----------
        value : Tensor
            Simplex-valued observations, last dimension is :math:`K`.

        Returns
        -------
        Tensor
            Log-densities, shape ``batch_shape``.
        """
        a = self.concentration
        log_b = lucid.lgamma(a).sum(dim=-1) - lucid.lgamma(a.sum(dim=-1))
        return lucid.xlogy(a - 1.0, value).sum(dim=-1) - log_b

    @override
    def entropy(self) -> Tensor:
        r"""Shannon entropy of the Dirichlet distribution (in nats).

        .. math::

            H = \log B(\boldsymbol{\alpha})
            + (\alpha_0 - K) \psi(\alpha_0)
            - \sum_i (\alpha_i - 1) \psi(\alpha_i)

        where :math:`\alpha_0 = \sum_i \alpha_i`, :math:`K` is the
        number of categories, and :math:`\psi` is the digamma function.

        Returns
        -------
        Tensor
            Entropy in nats, shape ``batch_shape``.
        """
        a = self.concentration
        s = a.sum(dim=-1)
        k = a.shape[-1]
        log_b = lucid.lgamma(a).sum(dim=-1) - lucid.lgamma(s)
        # Each term of the formula once: ψ(α₀) appears only in
        # ``(α₀ − K) ψ(α₀)``.  Writing the last sum as
        # ``Σ (αᵢ − 1)(ψ(αᵢ) − ψ(α₀))`` counted it a second time.
        return log_b + (s - k) * lucid.digamma(s) - ((a - 1.0) * lucid.digamma(a)).sum(dim=-1)

    @override
    @property
    def _natural_params(self) -> tuple[Tensor]:
        return (self.concentration,)

    @override
    def _log_normalizer(self, *natural_params: Tensor) -> Tensor:
        (x,) = natural_params
        return lucid.lgamma(x).sum(dim=-1) - lucid.lgamma(x.sum(dim=-1))

    @override
    @property
    def _mean_carrier_measure(self) -> Tensor:
        a = self.concentration
        k = float(a.shape[-1])
        return k * lucid.digamma(a.sum(dim=-1)) - lucid.digamma(a).sum(dim=-1)
