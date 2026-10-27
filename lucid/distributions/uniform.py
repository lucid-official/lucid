"""Continuous ``Uniform(low, high)``."""

import math
from collections.abc import Sequence
from typing import Self, override

import lucid
from lucid._tensor.tensor import Tensor
from lucid.distributions.constraints import Constraint, interval, real
from lucid.distributions.distribution import Distribution


from lucid.distributions._util import _as_tensor, _broadcast_pair


class Uniform(Distribution):
    r"""Continuous Uniform distribution on the half-open interval :math:`[a, b)`.

    Maximum-entropy distribution over a bounded interval of fixed length:
    every point in :math:`[a, b)` is equally likely.  Widely used as a
    non-informative prior, as the source of randomness for inverse-CDF
    sampling of arbitrary distributions, and as a quasi-Monte-Carlo
    integration measure.

    Parameters
    ----------
    low : Tensor or float
        Lower bound :math:`a \in \mathbb{R}`.
    high : Tensor or float
        Upper bound :math:`b \in \mathbb{R}` with :math:`b > a`.
    validate_args : bool, optional
        If ``True``, validate parameter constraints at construction time.

    Notes
    -----
    Probability density (constant on the support):

    .. math::

        p(x; a, b) = \begin{cases}
            \dfrac{1}{b - a} & x \in [a, b) \\[2pt]
            0 & \text{otherwise}
        \end{cases}

    Cumulative distribution:

    .. math::

        F(x; a, b) = \mathrm{clip}\!\left(\frac{x - a}{b - a},\; 0,\; 1\right)

    Moments:

    .. math::

        \mathbb{E}[X] = \frac{a + b}{2}, \qquad
        \mathrm{Var}[X] = \frac{(b - a)^2}{12}, \qquad
        H[X] = \log(b - a)

    Higher moments (centred at the midpoint :math:`m = (a+b)/2`):
    :math:`\mathbb{E}[(X-m)^{2k}] = (b-a)^{2k}/(2k+1)\,2^{2k}`; odd
    central moments are zero by symmetry.

    The Uniform is the **maximum-entropy distribution** over a bounded
    interval of fixed length, and it underpins the inverse-CDF
    (Smirnov) sampling identity: if :math:`U \sim \mathrm{Uniform}(0, 1)`
    and :math:`F^{-1}` is the quantile function of any 1-D distribution,
    then :math:`F^{-1}(U)` has that distribution.

    Reparameterised sampling uses the location-scale transform
    :math:`X = a + (b - a) U` so gradients flow through both endpoints.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import Uniform
    >>> d = Uniform(low=0.0, high=1.0)
    >>> d.mean
    tensor(0.5)
    >>> d.rsample((4,)).shape
    (4,)
    >>> d.log_prob(lucid.tensor(0.5))
    tensor(0.)
    """

    arg_constraints = {"low": real, "high": real}
    has_rsample = True

    def __init__(
        self, low: Tensor | float, high: Tensor | float, validate_args: bool | None = None
    ) -> None:
        r"""Construct a Uniform distribution on ``[low, high)``.

        Parameters
        ----------
        low : Tensor | float
            Lower bound :math:`a \in \mathbb{R}`.
        high : Tensor | float
            Upper bound :math:`b \in \mathbb{R}` with :math:`b > a`.
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.

        Notes
        -----
        The Uniform distribution has constant PDF on its support:

        .. math::

            p(x; a, b) = \frac{1}{b - a}, \quad x \in [a, b)

        All values in the interval are equally likely.  The distribution is
        the maximum-entropy distribution over a bounded interval, and is
        widely used for non-informative priors and Monte-Carlo sampling via
        the inverse-CDF method.

        Examples
        --------
        >>> from lucid.distributions import Uniform
        >>> d = Uniform(low=0.0, high=1.0)
        >>> d.mean
        tensor(0.5)
        """
        self.low = _as_tensor(low)
        self.high = _as_tensor(high)
        self.low, self.high = _broadcast_pair(self.low, self.high)
        super().__init__(
            batch_shape=tuple(self.low.shape), event_shape=(), validate_args=validate_args
        )

    @override
    def expand(self, batch_shape: Sequence[int], _instance: Self | None = None) -> Self:
        """Return a new :class:`Uniform` with batch dimensions expanded to ``batch_shape``."""
        new = self._get_checked_instance(Uniform, _instance)
        batch_shape_t = tuple(batch_shape)
        new.low = self.low.expand(batch_shape_t)
        new.high = self.high.expand(batch_shape_t)
        super(Uniform, new).__init__(batch_shape_t, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @override
    @property
    def support(self) -> Constraint:
        """The support constraint of the Uniform distribution.

        Returns
        -------
        Constraint
            ``interval(low, high)`` — the bounds are the parameter tensors,
            so each batch member is held to its own interval.  Closed at
            both ends, as in the reference framework, although
            :meth:`log_prob` scores ``high`` itself as ``-inf``.
        """
        return interval(self.low, self.high)

    @override
    def _validate_params(self) -> None:
        """Check ``low`` and ``high`` against ``real``, then ``low < high``.

        The ordering involves both parameters, so no entry of
        ``arg_constraints`` can hold it; an empty or reversed interval has
        no density at all.

        Raises
        ------
        ValueError
            If a bound is NaN, or ``low >= high`` anywhere in the batch.
        """
        super()._validate_params()
        if not bool((self.low < self.high).all().item()):
            raise ValueError("Uniform: `low` must be less than `high`")

    @override
    @property
    def mean(self) -> Tensor:
        r"""Expected value of the Uniform distribution.

        .. math::

            E[X] = \frac{a + b}{2}

        Returns
        -------
        Tensor
            Midpoint :math:`(a+b)/2`, shape ``batch_shape``.

        Examples
        --------
        >>> Uniform(low=2.0, high=8.0).mean
        tensor(5.)
        """
        return 0.5 * (self.low + self.high)

    @override
    @property
    def mode(self) -> Tensor:
        """Mode of the Uniform distribution — ``nan``.

        Every point of ``[low, high]`` is equally likely, so there is no
        single mode; ``nan`` of the batch shape, as the reference framework
        answers.
        """
        return self.high * float("nan")

    @override
    @property
    def variance(self) -> Tensor:
        r"""Variance of the Uniform distribution.

        .. math::

            \operatorname{Var}[X] = \frac{(b - a)^2}{12}

        Returns
        -------
        Tensor
            Variance :math:`(b-a)^2/12`, shape ``batch_shape``.

        Examples
        --------
        >>> Uniform(low=0.0, high=1.0).variance  # 1/12 ≈ 0.0833
        tensor(0.08333)
        """
        d = self.high - self.low
        return d * d / 12.0

    @override
    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        r"""Reparameterised sample using the location-scale transform.

        .. math::

            X = a + (b - a) \cdot U, \quad U \sim \text{Uniform}(0, 1)

        Gradients flow through both :math:`a` and :math:`b`.

        Parameters
        ----------
        sample_shape : tuple[int, ...], optional
            Leading shape dimensions for the sample batch.  Default is ``()``.

        Returns
        -------
        Tensor
            Samples in :math:`[a, b)` of shape ``sample_shape + batch_shape``.

        Examples
        --------
        >>> d = Uniform(low=-1.0, high=1.0)
        >>> x = d.rsample((100,))
        >>> bool(((x >= -1.0) & (x < 1.0)).all())
        True
        """
        shape = self._extended_shape(sample_shape)
        u = lucid.rand(*shape, dtype=self.low.dtype, device=self.low.device)
        return self.low + (self.high - self.low) * u

    @override
    def log_prob(self, value: Tensor) -> Tensor:
        r"""Log-density of ``value`` under the Uniform distribution.

        Returns :math:`-\log(b - a)` inside the support :math:`[a, b)` and
        :math:`-\infty` outside:

        .. math::

            \log p(x; a, b) = \begin{cases}
                -\log(b - a) & x \in [a, b) \\
                -\infty & \text{otherwise}
            \end{cases}

        Parameters
        ----------
        value : Tensor
            Values at which to evaluate the log-density.

        Returns
        -------
        Tensor
            Element-wise log-densities, shape ``batch_shape``.

        Examples
        --------
        >>> Uniform(low=0.0, high=2.0).log_prob(lucid.tensor(1.0))
        tensor(-0.6931)
        """
        # log(1 / (high - low)) inside support, -inf outside.
        log_density = -(self.high - self.low).log()
        # Match value's shape for the comparison so 0-dim params + N-dim value work.
        low_b = self.low + value * 0
        high_b = self.high + value * 0
        in_support = (value >= low_b) & (value < high_b)
        return lucid.where(
            in_support,
            log_density + value * 0,
            lucid.full(tuple(value.shape), -math.inf, device=value.device, dtype=value.dtype),
        )

    @override
    def cdf(self, value: Tensor) -> Tensor:
        r"""Cumulative distribution function of the Uniform distribution.

        .. math::

            F(x; a, b) = \text{clip}\!\left(\frac{x - a}{b - a},\; 0,\; 1\right)

        Parameters
        ----------
        value : Tensor
            Values at which to evaluate the CDF.

        Returns
        -------
        Tensor
            CDF values in :math:`[0, 1]`, shape ``batch_shape``.

        Examples
        --------
        >>> Uniform(low=0.0, high=4.0).cdf(lucid.tensor(1.0))
        tensor(0.25)
        """
        return ((value - self.low) / (self.high - self.low)).clip(0.0, 1.0)

    @override
    def icdf(self, value: Tensor) -> Tensor:
        r"""Inverse CDF (quantile function) of the Uniform distribution.

        .. math::

            F^{-1}(u; a, b) = a + u \cdot (b - a)

        Parameters
        ----------
        value : Tensor
            Probability values :math:`u \in [0, 1]`.

        Returns
        -------
        Tensor
            Quantiles in :math:`[a, b]`, shape ``batch_shape``.

        Examples
        --------
        >>> Uniform(low=0.0, high=10.0).icdf(lucid.tensor(0.5))
        tensor(5.)
        """
        return self.low + value * (self.high - self.low)

    @override
    def entropy(self) -> Tensor:
        r"""Shannon entropy of the Uniform distribution (in nats).

        .. math::

            H(X) = \log(b - a)

        The Uniform distribution maximises entropy among all distributions
        supported on a bounded interval of fixed length.

        Returns
        -------
        Tensor
            Entropy in nats, shape ``batch_shape``.

        Examples
        --------
        >>> Uniform(low=0.0, high=1.0).entropy()  # log(1) = 0.0
        tensor(0.)
        """
        return (self.high - self.low).log()
