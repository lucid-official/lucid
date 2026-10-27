"""``Independent`` — re-interpret rightmost batch dims of a base
distribution as event dims.

Used to turn a batch of independent univariates into a single
multivariate distribution with diagonal covariance, e.g.

    base = Normal(loc.shape == (B, D), scale.shape == (B, D))
    Independent(base, 1)  # event_shape == (D,), batch_shape == (B,)
"""

from collections.abc import Sequence
from typing import Self, override

from lucid._tensor.tensor import Tensor
from lucid.distributions._util import _sum_rightmost
from lucid.distributions.constraints import Constraint, independent
from lucid.distributions.distribution import Distribution


class Independent(Distribution):
    r"""Re-interpret rightmost batch dimensions of a base distribution as event dims.

    ``Independent(base_distribution, reinterpreted_batch_ndims=n)`` takes
    the last :math:`n` dimensions of ``base_distribution.batch_shape`` and
    moves them into ``event_shape``.  Log-probabilities are *summed* over the
    re-interpreted axes, turning a product of independent marginals into the
    joint log-probability.

    This is a pure structural wrapper — it does **not** change the sampler or
    the underlying computation.  Its main use is to express

    - **Diagonal multivariate Normals** from batches of scalar Normals, or
    - **Independent Bernoulli likelihoods** over image pixels from a
      per-pixel Bernoulli batch.

    Parameters
    ----------
    base_distribution : Distribution
        The underlying distribution.  Its ``batch_shape`` must have at least
        ``reinterpreted_batch_ndims`` dimensions.
    reinterpreted_batch_ndims : int
        Number of rightmost batch dimensions to absorb into ``event_shape``.
        Must satisfy ``0 <= reinterpreted_batch_ndims <= len(batch_shape)``.
    validate_args : bool | None, optional
        If ``True``, validate parameter constraints at construction time.

    Attributes
    ----------
    base_dist : Distribution
        The wrapped base distribution.
    reinterpreted_batch_ndims : int
        Number of absorbed batch dimensions.

    Notes
    -----
    Given ``base`` with ``batch_shape = (B, D)`` and ``event_shape = ()``,
    wrapping with ``reinterpreted_batch_ndims=1`` yields:

    - ``batch_shape = (B,)``
    - ``event_shape = (D,)``

    The resulting ``log_prob(x)`` sums the :math:`D` scalar log-probabilities:

    .. math::

        \log p(x_1, \ldots, x_D) = \sum_{i=1}^{D} \log p_i(x_i)

    which is correct because the components are independent by construction.

    **Entropy** is similarly summed:

    .. math::

        H[X_1, \ldots, X_D] = \sum_{i=1}^{D} H[X_i]

    (equality holds because independence implies
    :math:`H[X_1, \ldots, X_D] = \sum_i H[X_i]`).

    The ``has_rsample`` property mirrors ``base_dist.has_rsample``, so
    gradient flow is preserved when the base distribution supports it.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import Independent
    >>> from lucid.distributions import Normal
    >>> # Diagonal Normal: batch of D=4 scalars → single 4-d event
    >>> base = Normal(lucid.zeros(4), lucid.ones(4))
    >>> dist = Independent(base, reinterpreted_batch_ndims=1)
    >>> dist.batch_shape, dist.event_shape
    ((), (4,))
    >>> x = dist.rsample()
    >>> x.shape  # (4,)
    (4,)
    >>> dist.log_prob(x).shape  # scalar — sum of 4 log-probs
    ()
    >>> dist.log_prob(lucid.zeros(4))  # 4 * log N(0 | 0, 1)
    tensor(-3.676)
    """

    def __init__(
        self,
        base_distribution: Distribution,
        reinterpreted_batch_ndims: int,
        validate_args: bool | None = None,
    ) -> None:
        """Initialise an Independent distribution wrapper.

        Parameters
        ----------
        base_distribution : Distribution
            The underlying distribution.  Its ``batch_shape`` must have at
            least ``reinterpreted_batch_ndims`` dimensions.
        reinterpreted_batch_ndims : int
            Number of rightmost batch dimensions to move into ``event_shape``.
            Must satisfy
            ``0 <= reinterpreted_batch_ndims <= len(batch_shape)``.
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.

        Raises
        ------
        ValueError
            If ``reinterpreted_batch_ndims`` exceeds the number of batch
            dimensions of ``base_distribution``.
        """
        if reinterpreted_batch_ndims > len(base_distribution.batch_shape):
            raise ValueError(
                f"Independent: reinterpreted_batch_ndims "
                f"{reinterpreted_batch_ndims} exceeds base batch ndim "
                f"{len(base_distribution.batch_shape)}"
            )
        self.base_dist = base_distribution
        self.reinterpreted_batch_ndims = int(reinterpreted_batch_ndims)
        b = tuple(base_distribution.batch_shape)
        e = tuple(base_distribution.event_shape)
        new_batch = b[: len(b) - self.reinterpreted_batch_ndims]
        new_event = b[len(b) - self.reinterpreted_batch_ndims :] + e
        super().__init__(batch_shape=new_batch, event_shape=new_event, validate_args=validate_args)

    @override
    @property
    def has_rsample(self) -> bool:  # type: ignore[override]
        """Whether reparameterised sampling is supported.

        Mirrors ``base_dist.has_rsample``, so gradient flow is preserved
        when the base distribution supports it.

        Returns
        -------
        bool
            ``True`` if and only if the base distribution supports
            reparameterised sampling.
        """
        return self.base_dist.has_rsample

    @override
    @property
    def support(self) -> Constraint | None:
        """Support of the distribution — the base support, read per event.

        Returns
        -------
        Constraint or None
            ``independent(base_dist.support, reinterpreted_batch_ndims)``:
            its ``event_dim`` counts the reinterpreted axes, and it answers
            once per event.  ``None`` when the base declares no support.
        """
        base = self.base_dist.support
        if base is None:
            return None
        return independent(base, self.reinterpreted_batch_ndims)

    @override
    @property
    def mean(self) -> Tensor:
        """Mean of the distribution — delegates to ``base_dist.mean``.

        Returns
        -------
        Tensor
            Mean with the same shape as ``base_dist.mean``.
        """
        return self.base_dist.mean

    @override
    @property
    def mode(self) -> Tensor:
        """Mode of the distribution — delegates to ``base_dist.mode``.

        Returns
        -------
        Tensor
            Mode with the same shape as ``base_dist.mode``.
        """
        return self.base_dist.mode

    @override
    @property
    def variance(self) -> Tensor:
        """Variance of the distribution — delegates to ``base_dist.variance``.

        Returns
        -------
        Tensor
            Variance with the same shape as ``base_dist.variance``.
        """
        return self.base_dist.variance

    @override
    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        """Draw reparameterised samples — delegates to ``base_dist.rsample``.

        Parameters
        ----------
        sample_shape : tuple[int, ...], optional
            Leading shape of the output sample batch.

        Returns
        -------
        Tensor
            Samples of shape ``(*sample_shape, *batch_shape, *event_shape)``.
        """
        return self.base_dist.rsample(sample_shape)

    @override
    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        """Draw samples — delegates to ``base_dist.sample``.

        Parameters
        ----------
        sample_shape : tuple[int, ...], optional
            Leading shape of the output sample batch.

        Returns
        -------
        Tensor
            Samples of shape ``(*sample_shape, *batch_shape, *event_shape)``.
        """
        return self.base_dist.sample(sample_shape)

    @override
    def log_prob(self, value: Tensor) -> Tensor:
        r"""Log joint probability summed over re-interpreted event dimensions.

        Computes the base distribution's log-probabilities and then sums
        over the rightmost ``reinterpreted_batch_ndims`` axes:

        .. math::

            \log p(x_1, \ldots, x_D) = \sum_{i=1}^{D} \log p_i(x_i)

        Parameters
        ----------
        value : Tensor
            Observations of shape ``(*batch_shape, *event_shape)``.

        Returns
        -------
        Tensor
            Log joint probability values of shape ``batch_shape``.
        """
        return _sum_rightmost(self.base_dist.log_prob(value), self.reinterpreted_batch_ndims)

    @override
    def entropy(self) -> Tensor:
        r"""Joint entropy summed over re-interpreted event dimensions.

        Because the components are independent:

        .. math::

            H[X_1, \ldots, X_D] = \sum_{i=1}^{D} H[X_i]

        Returns
        -------
        Tensor
            Entropy values of shape ``batch_shape`` (nats).
        """
        return _sum_rightmost(self.base_dist.entropy(), self.reinterpreted_batch_ndims)

    def _has_enumerate_support(self) -> bool:
        """Whether finite support enumeration is supported.

        Mirrors ``base_dist.has_enumerate_support`` when
        ``reinterpreted_batch_ndims == 0``; cartesian-product enumeration
        over re-interpreted batch dimensions is not implemented.
        """
        if self.reinterpreted_batch_ndims > 0:
            return False
        return self.base_dist.has_enumerate_support

    has_enumerate_support = property(_has_enumerate_support, doc=_has_enumerate_support.__doc__)

    @override
    def enumerate_support(self, expand: bool = True) -> Tensor:
        """Delegate support enumeration to ``base_dist`` when no batch axes are re-interpreted."""
        if self.reinterpreted_batch_ndims > 0:
            raise NotImplementedError("Enumeration over cartesian product is not implemented")
        return self.base_dist.enumerate_support(expand=expand)

    @override
    def expand(self, batch_shape: Sequence[int], _instance: Self | None = None) -> Self:
        """Return a new :class:`Independent` instance with parameters expanded to ``batch_shape``."""
        new = self._get_checked_instance(Independent, _instance)
        batch_shape = tuple(batch_shape)
        base_batch = batch_shape + self.event_shape[: self.reinterpreted_batch_ndims]
        new.base_dist = self.base_dist.expand(base_batch)
        new.reinterpreted_batch_ndims = self.reinterpreted_batch_ndims
        super(Independent, new).__init__(
            batch_shape=batch_shape, event_shape=self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new
