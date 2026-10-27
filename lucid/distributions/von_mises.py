"""Additional continuous and directional distributions:
``LogisticNormal``, ``GeneralizedPareto``, ``VonMises``.
"""

import math
from collections.abc import Sequence
from typing import Self, override

import lucid
from lucid._tensor.tensor import Tensor
from lucid.distributions._util import _as_tensor, _broadcast_pair
from lucid.distributions.constraints import Constraint, interval, positive, real, simplex
from lucid.distributions.distribution import Distribution
from lucid.distributions.independent import Independent
from lucid.distributions.normal import Normal
from lucid.distributions.transformed_distribution import TransformedDistribution
from lucid.distributions.transforms import StickBreakingTransform


class LogisticNormal(TransformedDistribution):
    r"""Logistic-Normal distribution on the open :math:`K`-simplex.

    Pushes a batch of :math:`K - 1` Normal random variables through
    :class:`~lucid.distributions.transforms.StickBreakingTransform` to produce
    a distribution on the :math:`K`-simplex :math:`\Delta^{K-1}`:

    .. math::

        \mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \operatorname{diag}(\boldsymbol{\sigma}^2)),
        \qquad
        \mathbf{y} = \operatorname{StickBreakingTransform}(\mathbf{x}) \in \Delta^{K-1}

    Parameters
    ----------
    loc : Tensor or float
        Mean :math:`\boldsymbol{\mu}` of the underlying Normal distribution.
    scale : Tensor or float
        Standard deviation :math:`\boldsymbol{\sigma} > 0` of the underlying
        Normal distribution.
    validate_args : bool | None, optional
        If ``True``, validate parameter constraints at construction time.
    """

    arg_constraints = {"loc": real, "scale": positive}
    support: Constraint | None = simplex
    has_rsample = True

    def __init__(
        self, loc: Tensor | float, scale: Tensor | float, validate_args: bool | None = None
    ) -> None:
        """Initialise a :class:`LogisticNormal` distribution."""
        base_dist = Normal(loc, scale, validate_args=validate_args)
        if not base_dist.batch_shape:
            base_dist = base_dist.expand([1])
        super().__init__(base_dist, StickBreakingTransform(), validate_args=validate_args)

    @override
    def expand(self, batch_shape: Sequence[int], _instance: Self | None = None) -> Self:
        """Return a new :class:`LogisticNormal` instance expanded to ``batch_shape``."""
        new = self._get_checked_instance(LogisticNormal, _instance)
        return super().expand(batch_shape, _instance=new)

    @property
    def loc(self) -> Tensor:
        """Mean parameter of the underlying Normal distribution."""
        assert isinstance(self.base_dist, Independent)
        assert isinstance(self.base_dist.base_dist, Normal)
        return self.base_dist.base_dist.loc

    @property
    def scale(self) -> Tensor:
        """Scale parameter of the underlying Normal distribution."""
        assert isinstance(self.base_dist, Independent)
        assert isinstance(self.base_dist.base_dist, Normal)
        return self.base_dist.base_dist.scale


class GeneralizedPareto(Distribution):
    r"""Generalised Pareto distribution for modelling distribution tails.

    Parameterised by location :math:`\mu \in \mathbb{R}`, scale
    :math:`\sigma > 0`, and shape / concentration :math:`\xi \in \mathbb{R}`.

    **Support**:

    .. math::

        x \in
        \begin{cases}
            [\mu, \infty) & \xi \geq 0 \\
            [\mu, \mu - \sigma / \xi] & \xi < 0
        \end{cases}

    **Density**:

    .. math::

        p(x; \mu, \sigma, \xi) =
        \begin{cases}
            \sigma^{-1} \bigl(1 + \xi \frac{x - \mu}{\sigma}\bigr)^{-(1 + 1/\xi)} & \xi \neq 0 \\
            \sigma^{-1} \exp\!\bigl(-\frac{x - \mu}{\sigma}\bigr) & \xi = 0
        \end{cases}

    Parameters
    ----------
    loc : Tensor or float
        Location parameter :math:`\mu \in \mathbb{R}`.
    scale : Tensor or float
        Scale parameter :math:`\sigma > 0`.
    concentration : Tensor or float
        Shape / tail index parameter :math:`\xi \in \mathbb{R}`.
    validate_args : bool | None, optional
        If ``True``, validate parameter constraints at construction time.
    """

    arg_constraints = {"loc": real, "scale": positive, "concentration": real}
    has_rsample = True

    def __init__(
        self,
        loc: Tensor | float,
        scale: Tensor | float,
        concentration: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        """Initialise a :class:`GeneralizedPareto` distribution."""
        self.loc = _as_tensor(loc)
        self.scale = _as_tensor(scale)
        self.concentration = _as_tensor(concentration)
        self.loc, self.scale = _broadcast_pair(self.loc, self.scale)
        self.loc, self.concentration = _broadcast_pair(self.loc, self.concentration)
        self.scale, self.concentration = _broadcast_pair(self.scale, self.concentration)
        super().__init__(
            batch_shape=tuple(self.loc.shape), event_shape=(), validate_args=validate_args
        )

    @override
    def expand(self, batch_shape: Sequence[int], _instance: Self | None = None) -> Self:
        """Return a new :class:`GeneralizedPareto` instance expanded to ``batch_shape``."""
        new = self._get_checked_instance(GeneralizedPareto, _instance)
        batch_shape = tuple(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.concentration = self.concentration.expand(batch_shape)
        super(GeneralizedPareto, new).__init__(
            batch_shape=batch_shape, event_shape=(), validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    @override
    @property
    def mean(self) -> Tensor:
        r"""Mean :math:`\mu + \sigma / (1 - \xi)` for :math:`\xi < 1`, ``NaN`` otherwise."""
        valid = self.concentration < 1.0
        safe_conc = lucid.where(valid, self.concentration, 0.5)
        return lucid.where(valid, self.loc + self.scale / (1.0 - safe_conc), math.nan)

    @override
    @property
    def variance(self) -> Tensor:
        r"""Variance :math:`\sigma^2 / ((1 - \xi)^2 (1 - 2\xi))` for :math:`\xi < 1/2`, ``NaN`` otherwise."""
        valid = self.concentration < 0.5
        safe_conc = lucid.where(valid, self.concentration, 0.25)
        return lucid.where(
            valid, self.scale**2 / ((1.0 - safe_conc) ** 2 * (1.0 - 2.0 * safe_conc)), math.nan
        )

    @override
    @property
    def mode(self) -> Tensor:
        r"""Mode of the :class:`GeneralizedPareto` distribution (equal to ``loc``)."""
        return self.loc

    @override
    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        """Draw reparameterised samples via inverse-CDF transformation."""
        shape = self._extended_shape(sample_shape)
        u = lucid.rand(*shape, dtype=self.loc.dtype, device=self.loc.device)
        return self._icdf_unvalidated(u)

    @override
    def log_prob(self, value: Tensor) -> Tensor:
        """Log-probability density evaluated at ``value``."""
        z = self._z(value)
        eq_zero = lucid.isclose(self.concentration, lucid.zeros_like(self.concentration))
        safe_conc = lucid.where(eq_zero, lucid.ones_like(self.concentration), self.concentration)
        y = 1.0 / safe_conc + lucid.ones_like(z)
        where_nonzero = lucid.where(y == 0.0, y, y * lucid.log1p(safe_conc * z))
        return -self.scale.log() - lucid.where(eq_zero, z, where_nonzero)

    def log_survival_function(self, value: Tensor) -> Tensor:
        r"""Log survival function :math:`\log(1 - F(x))`."""
        return self._log_survival_unvalidated(value)

    def _log_survival_unvalidated(self, value: Tensor) -> Tensor:
        z = self._z(value)
        eq_zero = lucid.isclose(self.concentration, lucid.zeros_like(self.concentration))
        safe_conc = lucid.where(eq_zero, lucid.ones_like(self.concentration), self.concentration)
        where_nonzero = -lucid.log1p(safe_conc * z) / safe_conc
        return lucid.where(eq_zero, -z, where_nonzero)

    def log_cdf(self, value: Tensor) -> Tensor:
        r"""Log cumulative distribution function :math:`\log F(x)`."""
        return lucid.log1p(-lucid.exp(self._log_survival_unvalidated(value)))

    @override
    def cdf(self, value: Tensor) -> Tensor:
        """Cumulative distribution function :math:`F(x)`."""
        return lucid.exp(lucid.log1p(-lucid.exp(self._log_survival_unvalidated(value))))

    def _icdf_unvalidated(self, value: Tensor) -> Tensor:
        loc = self.loc
        scale = self.scale
        conc = self.concentration
        eq_zero = lucid.isclose(conc, lucid.zeros_like(conc))
        safe_conc = lucid.where(eq_zero, lucid.ones_like(conc), conc)
        logu = lucid.log1p(-value)
        where_nonzero = loc + scale / safe_conc * lucid.expm1(-safe_conc * logu)
        where_zero = loc - scale * logu
        return lucid.where(eq_zero, where_zero, where_nonzero)

    @override
    def icdf(self, value: Tensor) -> Tensor:
        """Inverse cumulative distribution (quantile) function :math:`F^{-1}(u)`."""
        return self._icdf_unvalidated(value)

    @override
    def entropy(self) -> Tensor:
        r"""Differential entropy :math:`\log\sigma + \xi + 1`."""
        ans = self.scale.log() + self.concentration + 1.0
        return ans.broadcast_to(self._batch_shape)

    def _z(self, x: Tensor) -> Tensor:
        return (x - self.loc) / self.scale

    @override
    @property
    def support(self) -> Constraint:
        """Support interval :math:`[\\mu, \\infty)` for :math:`\\xi \\geq 0`, or :math:`[\\mu, \\mu - \\sigma/\\xi]` for :math:`\\xi < 0`."""
        neg = self.concentration < 0.0
        safe_conc = lucid.where(neg, self.concentration, -1.0)
        upper = lucid.where(neg, self.loc - self.scale / safe_conc, math.inf)
        return interval(self.loc, upper)


_I0_COEF_SMALL: list[float] = [
    1.0,
    3.5156229,
    3.0899424,
    1.2067492,
    0.2659732,
    0.360768e-1,
    0.45813e-2,
]
_I0_COEF_LARGE: list[float] = [
    0.39894228,
    0.1328592e-1,
    0.225319e-2,
    -0.157565e-2,
    0.916281e-2,
    -0.2057706e-1,
    0.2635537e-1,
    -0.1647633e-1,
    0.392377e-2,
]
_I1_COEF_SMALL: list[float] = [
    0.5,
    0.87890594,
    0.51498869,
    0.15084934,
    0.2658733e-1,
    0.301532e-2,
    0.32411e-3,
]
_I1_COEF_LARGE: list[float] = [
    0.39894228,
    -0.3988024e-1,
    -0.362018e-2,
    0.163801e-2,
    -0.1031555e-1,
    0.2282967e-1,
    -0.2895312e-1,
    0.1787654e-1,
    -0.420059e-2,
]

_COEF_SMALL = [_I0_COEF_SMALL, _I1_COEF_SMALL]
_COEF_LARGE = [_I0_COEF_LARGE, _I1_COEF_LARGE]


def _eval_poly(y: Tensor, coef: list[float]) -> Tensor:
    result = lucid.full_like(y, coef[-1])
    for c in reversed(coef[:-1]):
        result = c + y * result
    return result


def _log_modified_bessel_fn(x: Tensor, order: int = 0) -> Tensor:
    """Compute ``log(I_order(x))`` for ``x > 0`` and ``order in (0, 1)``."""
    if order not in (0, 1):
        raise ValueError("order must be 0 or 1")

    y = x / 3.75
    y = y * y
    small = _eval_poly(y, _COEF_SMALL[order])
    if order == 1:
        small = x.abs() * small
    small = small.log()

    y_large = 3.75 / x
    large = x - 0.5 * x.log() + _eval_poly(y_large, _COEF_LARGE[order]).log()

    return lucid.where(x < 3.75, small, large)


def _sample_von_mises(loc: Tensor, concentration: Tensor) -> Tensor:
    """Best & Fisher (1979) rejection sampler for the Von Mises distribution."""
    work_dtype = lucid.float64 if loc.device.type == "cpu" else lucid.float32
    x_conc = concentration.to(work_dtype)
    tau = 1.0 + (1.0 + 4.0 * x_conc * x_conc).sqrt()
    rho = (tau - (2.0 * tau).sqrt()) / (2.0 * x_conc)
    r = (1.0 + rho * rho) / (2.0 * rho)

    z_res = lucid.zeros(tuple(loc.shape), dtype=work_dtype, device=loc.device)
    u3 = lucid.rand(*loc.shape, dtype=work_dtype, device=loc.device)
    done = lucid.zeros(tuple(loc.shape), dtype=lucid.bool, device=loc.device)
    while not bool(done.all().item()):
        u1 = lucid.rand(*loc.shape, dtype=work_dtype, device=loc.device)
        u2 = lucid.rand(*loc.shape, dtype=work_dtype, device=loc.device)
        z = (math.pi * u1).cos()
        f = (1.0 + r * z) / (r + z)
        c = x_conc * (r - f)
        accept = ((c * (2.0 - c) - u2) > 0.0) | ((c / u2).log() + 1.0 - c >= 0.0)
        z_res = lucid.where(done, z_res, f)
        done = done | accept

    f_clamped = z_res.clip(-1.0, 1.0)
    theta = lucid.where(u3 > 0.5, f_clamped.acos(), -f_clamped.acos())
    out = (loc.to(work_dtype) + theta + math.pi) % (2.0 * math.pi) - math.pi
    return out.to(loc.dtype)


class VonMises(Distribution):
    r"""Circular Von Mises distribution on :math:`[-\pi, \pi]`.

    The circular analogue of the Normal distribution on the unit circle, with
    density

    .. math::

        p(x; \mu, \kappa) = \frac{\exp(\kappa \cos(x - \mu))}{2\pi I_0(\kappa)}

    where :math:`\mu \in \mathbb{R}` is the circular mean / mode,
    :math:`\kappa > 0` is the concentration, and :math:`I_0` is the modified
    Bessel function of order zero.

    Parameters
    ----------
    loc : Tensor or float
        Circular mean / mode :math:`\mu \in \mathbb{R}` (radians).
    concentration : Tensor or float
        Concentration parameter :math:`\kappa > 0`.
    validate_args : bool | None, optional
        If ``True``, validate parameter constraints at construction time.
    """

    arg_constraints = {"loc": real, "concentration": positive}
    support: Constraint | None = real
    has_rsample = False

    def __init__(
        self, loc: Tensor | float, concentration: Tensor | float, validate_args: bool | None = None
    ) -> None:
        """Initialise a :class:`VonMises` distribution."""
        self.loc, self.concentration = _broadcast_pair(_as_tensor(loc), _as_tensor(concentration))
        super().__init__(
            batch_shape=tuple(self.loc.shape), event_shape=(), validate_args=validate_args
        )

    @override
    def log_prob(self, value: Tensor) -> Tensor:
        """Log-probability density evaluated at ``value``."""
        return (
            self.concentration * lucid.cos(value - self.loc)
            - math.log(2.0 * math.pi)
            - _log_modified_bessel_fn(self.concentration, order=0)
        )

    @override
    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        """Draw non-reparameterised samples via Best & Fisher (1979) rejection sampling."""
        shape = self._extended_shape(sample_shape)
        loc = self.loc.expand(shape)
        conc = self.concentration.expand(shape)
        return _sample_von_mises(loc, conc).detach()

    @override
    def expand(self, batch_shape: Sequence[int], _instance: Self | None = None) -> Self:
        """Return a new :class:`VonMises` instance expanded to ``batch_shape``."""
        new = self._get_checked_instance(VonMises, _instance)
        batch_shape = tuple(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.concentration = self.concentration.expand(batch_shape)
        super(VonMises, new).__init__(batch_shape=batch_shape, event_shape=(), validate_args=False)
        new._validate_args = self._validate_args
        return new

    @override
    @property
    def mean(self) -> Tensor:
        """Circular mean :math:`\\mu`."""
        return self.loc

    @override
    @property
    def mode(self) -> Tensor:
        """Circular mode :math:`\\mu`."""
        return self.loc

    @override
    @property
    def variance(self) -> Tensor:
        r"""Circular variance :math:`1 - I_1(\kappa) / I_0(\kappa)`."""
        return (
            1.0
            - (
                _log_modified_bessel_fn(self.concentration, order=1)
                - _log_modified_bessel_fn(self.concentration, order=0)
            ).exp()
        )
