"""``TransformedDistribution`` and the structural transforms (base, compose, inverse, reshape)."""

import math
from collections.abc import Sequence
from typing import Self, final, override

import lucid
from lucid._tensor.tensor import Tensor
from lucid.distributions import constraints
from lucid.distributions._util import _sum_rightmost
from lucid.distributions.constraints import Constraint
from lucid.distributions.distribution import Distribution
from lucid.distributions.independent import Independent


class Transform:
    r"""Abstract bijection between two measurable spaces.

    A :class:`Transform` represents an invertible map
    :math:`T : \mathcal{X} \to \mathcal{Y}` together with its
    log-absolute-determinant Jacobian, used to push a base distribution
    through a measurable bijection.  Composed with a base distribution
    via :class:`TransformedDistribution`, it implements normalising flows,
    coupling layers, and standard change-of-variable arguments.

    The change-of-variable formula for densities is

    .. math::

        \log p_Y(\mathbf{y}) = \log p_X(\mathbf{x})
        - \log \left|\det \frac{\partial T(\mathbf{x})}{\partial \mathbf{x}}\right|,
        \quad \mathbf{y} = T(\mathbf{x})

    so the per-transform Jacobian determinant must be tractable.

    Subclasses must override:

    * :meth:`_call` — forward map :math:`\mathbf{y} = T(\mathbf{x})`.
    * :meth:`_inverse` — inverse map
      :math:`\mathbf{x} = T^{-1}(\mathbf{y})`.
    * :meth:`log_abs_det_jacobian` —
      :math:`\log \bigl|\det \partial T / \partial \mathbf{x}\bigr|`,
      broadcast over the batch dimensions of the input.

    Attributes
    ----------
    bijective : bool
        ``True`` if the transform is invertible (most are; notable
        exception: :class:`AbsTransform`).
    sign : int or Tensor
        :math:`+1` for monotone-increasing scalar bijections,
        :math:`-1` for monotone-decreasing (a tensor when the slope is a
        tensor, as with an :class:`AffineTransform` whose ``scale`` is).
        :class:`TransformedDistribution` reads it to orient its CDF.  A
        transform with no meaningful sign (:class:`AbsTransform`, the
        vector-valued transforms) raises :class:`NotImplementedError` on
        access, so a missing sign is never mistaken for ``+1``.
    event_dim : int
        Number of trailing tensor dimensions that the transform treats
        as a single event.  ``0`` for element-wise maps,
        :math:`\geq 1` for vector / matrix transforms (e.g.
        :class:`SoftmaxTransform` has ``event_dim = 1``,
        :class:`LowerCholeskyTransform` has ``event_dim = 2``).

    Notes
    -----
    Composition and inversion are provided structurally:

    * :class:`ComposeTransform` — :math:`T = T_n \circ \cdots \circ T_1`.
    * ``~transform`` / :attr:`inv` — lazy :class:`_InverseTransform` view.

    so users can build complex bijectors without manually tracking
    Jacobian terms.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions.transforms import ExpTransform, AffineTransform, ComposeTransform
    >>> # y = exp(2*x + 1) — a log-normal-like transform
    >>> T = ComposeTransform([AffineTransform(loc=1.0, scale=2.0), ExpTransform()])
    >>> x = lucid.tensor(0.0)
    >>> y = T(x)
    >>> T.log_abs_det_jacobian(x, y)  # log(2) + (2x + 1) at x = 0
    tensor(1.693)
    """

    bijective: bool = True
    event_dim: int = 0

    def __init__(self) -> None:
        """Initialise the transform with an empty inverse cache."""
        self._inv: Transform | None = None

    @property
    def sign(self) -> int | Tensor:
        """Orientation of the map: ``+1`` increasing, ``-1`` decreasing.

        Only meaningful for a monotone scalar bijection.  A subclass states
        its own (a class attribute for a fixed one, a property for one that
        depends on parameters); the base refuses, so a transform that never
        declared a sign cannot be silently read as increasing.

        Raises
        ------
        NotImplementedError
            For a transform with no defined sign.
        """
        raise NotImplementedError(f"{type(self).__name__}.sign")

    @property
    def domain(self) -> Constraint:
        """Constraint describing valid inputs to this transform."""
        raise NotImplementedError(f"{type(self).__name__}.domain")

    @property
    def codomain(self) -> Constraint:
        """Constraint describing valid outputs of this transform."""
        raise NotImplementedError(f"{type(self).__name__}.codomain")

    def forward_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        """Infer the output shape from input ``shape`` (defaults to identity)."""
        return tuple(shape)

    def inverse_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        """Infer the input shape from output ``shape`` (defaults to identity)."""
        return tuple(shape)

    @property
    def inv(self) -> Transform:
        """Lazy inverse — caches an :class:`_InverseTransform` view."""
        if self._inv is None:
            self._inv = _InverseTransform(self)
        return self._inv

    def __invert__(self) -> Transform:
        """Return ``self.inv``. Enables the ``~transform`` syntax."""
        return self.inv

    def __call__(self, x: Tensor) -> Tensor:
        """Apply the forward map ``T(x)`` by delegating to :meth:`_call`."""
        return self._call(x)

    def _call(self, x: Tensor) -> Tensor:
        """Forward map ``y = T(x)`` — must be overridden by subclasses."""
        raise NotImplementedError(f"{type(self).__name__}._call")

    def _inverse(self, y: Tensor) -> Tensor:
        """Inverse map ``x = T^{-1}(y)`` — must be overridden by subclasses."""
        raise NotImplementedError(f"{type(self).__name__}._inverse")

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        r"""Log-absolute-determinant of the Jacobian :math:`\partial T / \partial x`.

        Subclasses must override and return
        :math:`\log|\det \partial y / \partial x|` broadcast over the
        input batch dimensions.

        Parameters
        ----------
        x : Tensor
            Pre-transform input.
        y : Tensor
            Post-transform output (typically ``y = self(x)``).  Some
            transforms can compute the log-Jacobian in terms of
            ``y`` more cheaply than ``x``, hence both are required.

        Returns
        -------
        Tensor
            Element-wise log-absolute-Jacobian-determinant, broadcastable
            to ``x.shape``.
        """
        raise NotImplementedError(f"{type(self).__name__}.log_abs_det_jacobian")

    @override
    def __repr__(self) -> str:
        """Return a developer-facing string representation of the instance."""
        return f"{type(self).__name__}()"


@final
class _InverseTransform(Transform):
    """View of the inverse of an underlying transform."""

    def __init__(self, base: Transform) -> None:
        """Wrap ``base`` and inherit its event-dim/sign metadata."""
        super().__init__()
        self._base = base
        self.bijective = base.bijective
        self.event_dim = base.event_dim
        self._inv = base

    @override
    @property
    def sign(self) -> int | Tensor:
        """Sign of the base transform: a monotone map and its inverse share one."""
        return self._base.sign

    @override
    @property
    def domain(self) -> Constraint:
        """Domain of the inverse view (= codomain of the base transform)."""
        return self._base.codomain

    @override
    @property
    def codomain(self) -> Constraint:
        """Codomain of the inverse view (= domain of the base transform)."""
        return self._base.domain

    @override
    def forward_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        """Forward shape of the inverse view (= ``base.inverse_shape``)."""
        return self._base.inverse_shape(shape)

    @override
    def inverse_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        """Inverse shape of the inverse view (= ``base.forward_shape``)."""
        return self._base.forward_shape(shape)

    @override
    @property
    def inv(self) -> Transform:
        """Return the wrapped base transform (inverse of this view)."""
        return self._base

    @override
    def _call(self, x: Tensor) -> Tensor:
        """Forward direction of the inverse view = ``base._inverse``."""
        return self._base._inverse(x)

    @override
    def _inverse(self, y: Tensor) -> Tensor:
        """Inverse direction of the inverse view = ``base._call``."""
        return self._base._call(y)

    @override
    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        r"""Jacobian of the inverse view — negate the base with roles swapped.

        For an inverse view of a bijection ``T``,
        :math:`\log|\det \partial T^{-1}/\partial y|
        = -\log|\det \partial T/\partial x|`,
        so we delegate to the base transform with ``x`` and ``y`` swapped
        and flip sign.

        Parameters
        ----------
        x : Tensor
            Inverse-view input (= base-view output).
        y : Tensor
            Inverse-view output (= base-view input).

        Returns
        -------
        Tensor
            Negated log-Jacobian of the base transform.
        """
        return -self._base.log_abs_det_jacobian(y, x)


class ComposeTransform(Transform):
    r"""Function composition of a list of transforms — left-to-right.

    Implements the composite map
    :math:`T = T_n \circ T_{n-1} \circ \cdots \circ T_1` so that the
    first transform in ``parts`` is applied first.  The composite
    Jacobian is the sum of the per-step Jacobians along the trajectory,
    by the chain rule.  This is the workhorse for stacking normalising
    flows.

    Parameters
    ----------
    parts : list[Transform]
        Ordered list of sub-transforms.  Must be non-empty.  Their
        ``event_dim`` values are reduced via :func:`max`.

    Raises
    ------
    ValueError
        If ``parts`` is empty.

    Notes
    -----
    Forward:

    .. math::

        \mathbf{y} = T_n(T_{n-1}(\cdots T_1(\mathbf{x}) \cdots ))

    Inverse (transforms run in reverse with each one inverted):

    .. math::

        \mathbf{x} = T_1^{-1}(T_2^{-1}(\cdots T_n^{-1}(\mathbf{y}) \cdots ))

    Log Jacobian determinant (chain rule):

    .. math::

        \log|\det J_T(\mathbf{x})|
        = \sum_{k=1}^{n}
            \log|\det J_{T_k}(\mathbf{x}^{(k-1)})|

    where :math:`\mathbf{x}^{(0)} = \mathbf{x}` and
    :math:`\mathbf{x}^{(k)} = T_k(\mathbf{x}^{(k-1)})`.  The
    implementation re-evaluates the forward chain to collect the
    intermediate states needed by each sub-Jacobian.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions.transforms import (
    ...     ExpTransform, AffineTransform, ComposeTransform)
    >>> # Log-normal-like: y = exp(loc + scale * x)
    >>> T = ComposeTransform([AffineTransform(loc=0.0, scale=1.0), ExpTransform()])
    >>> T(lucid.tensor(0.0))
    tensor(1.)
    """

    def __init__(self, parts: list[Transform]) -> None:
        """Store the ordered list of sub-transforms.

        Raises
        ------
        ValueError
            If ``parts`` is empty.
        """
        super().__init__()
        if not parts:
            raise ValueError("ComposeTransform: parts cannot be empty.")
        self.parts = list(parts)
        self.event_dim = max(p.event_dim for p in parts)

    @override
    @property
    def sign(self) -> int | Tensor:
        """Product of the parts' signs.

        Raises
        ------
        NotImplementedError
            If any part has no defined sign.
        """
        sign: int | Tensor = 1
        for part in self.parts:
            sign = sign * part.sign
        return sign

    @override
    @property
    def domain(self) -> Constraint:
        """Input domain of the composite transform."""
        if not self.parts:
            return constraints.real
        domain = self.parts[0].domain
        reinterpreted_batch_ndims = self.event_dim - domain.event_dim
        if reinterpreted_batch_ndims > 0:
            return constraints.independent(domain, reinterpreted_batch_ndims)
        return domain

    @override
    @property
    def codomain(self) -> Constraint:
        """Output codomain of the composite transform."""
        if not self.parts:
            return constraints.real
        codomain = self.parts[-1].codomain
        reinterpreted_batch_ndims = self.event_dim - codomain.event_dim
        if reinterpreted_batch_ndims > 0:
            return constraints.independent(codomain, reinterpreted_batch_ndims)
        return codomain

    @override
    def forward_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        """Propagate ``shape`` forward through every sub-transform in order."""
        for part in self.parts:
            shape = part.forward_shape(shape)
        return shape

    @override
    def inverse_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        """Propagate ``shape`` backward through every sub-transform in reverse order."""
        for part in reversed(self.parts):
            shape = part.inverse_shape(shape)
        return shape

    @override
    def _call(self, x: Tensor) -> Tensor:
        """Forward: apply every sub-transform in order."""
        for p in self.parts:
            x = p(x)
        return x

    @override
    def _inverse(self, y: Tensor) -> Tensor:
        """Inverse: apply each sub-transform's inverse in reverse order."""
        for p in reversed(self.parts):
            y = p._inverse(y)
        return y

    @override
    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        """Cumulative log-Jacobian across the composition.

        Re-evaluates the chain of forward maps to obtain the
        intermediate states needed by each sub-Jacobian, then sums the
        contributions.

        Parameters
        ----------
        x : Tensor
            Input to the first transform in the chain.
        y : Tensor
            Output of the last transform; ignored — the chain is
            re-evaluated from ``x``.

        Returns
        -------
        Tensor
            Element-wise sum of every sub-transform's log-Jacobian,
            broadcastable to ``x.shape``.
        """
        # Need intermediate values to evaluate each Jacobian.
        total = lucid.zeros(tuple(x.shape), dtype=x.dtype, device=x.device)
        cur = x
        for p in self.parts:
            nxt = p(cur)
            total = total + p.log_abs_det_jacobian(cur, nxt)
            cur = nxt
        return total


class ReshapeTransform(Transform):
    r"""Pure-shape reinterpretation of the event shape :math:`y = \mathrm{reshape}(x)`.

    Maps the trailing event tail of a tensor from ``in_shape`` to
    ``out_shape``.  This is a *volume-preserving* bijection — the
    underlying storage is unchanged, only the interpretation of which
    axes constitute the event differs.  The product of dimensions must
    match.

    Parameters
    ----------
    in_shape : tuple[int, ...]
        Event shape of the input tail.
    out_shape : tuple[int, ...]
        Event shape of the output tail.  Must satisfy
        :math:`\prod_i \mathrm{in\_shape}_i = \prod_i \mathrm{out\_shape}_i`.

    Raises
    ------
    ValueError
        If ``prod(in_shape) != prod(out_shape)``.

    Notes
    -----
    Forward:  :math:`\mathbf{y} = \mathrm{reshape}(\mathbf{x},
    \mathrm{out\_shape})`.

    Inverse:  :math:`\mathbf{x} = \mathrm{reshape}(\mathbf{y},
    \mathrm{in\_shape})`.

    Log Jacobian determinant:

    .. math::

        \log|\det J| = 0

    everywhere — a reshape preserves Lebesgue measure exactly.  Useful
    for stitching together transforms that expect different event
    layouts (e.g. flatten a matrix event for a Cholesky-style
    transform, then reshape back).

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions.transforms import ReshapeTransform
    >>> T = ReshapeTransform(in_shape=(2, 3), out_shape=(6,))
    >>> T(lucid.zeros(2, 3)).shape
    (6,)
    """

    def __init__(self, in_shape: tuple[int, ...], out_shape: tuple[int, ...]) -> None:
        """Validate that ``in_shape`` and ``out_shape`` have the same total size and store them.

        Raises
        ------
        ValueError
            If ``prod(in_shape) != prod(out_shape)``.
        """
        super().__init__()
        in_n = math.prod(in_shape) if in_shape else 1
        out_n = math.prod(out_shape) if out_shape else 1
        if in_n != out_n:
            raise ValueError(
                f"ReshapeTransform: in_shape {in_shape} ({in_n} elements) "
                f"!= out_shape {out_shape} ({out_n} elements)."
            )
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.event_dim = max(len(in_shape), len(out_shape))

    @override
    @property
    def domain(self) -> Constraint:
        """``independent(real, len(in_shape))``."""
        return constraints.independent(constraints.real, len(self.in_shape))

    @override
    @property
    def codomain(self) -> Constraint:
        """``independent(real, len(out_shape))``."""
        return constraints.independent(constraints.real, len(self.out_shape))

    @override
    def forward_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        """Replace the trailing ``in_shape`` dimensions with ``out_shape``."""
        if len(shape) < len(self.in_shape):
            return tuple(shape)
        cut = len(shape) - len(self.in_shape)
        return tuple(shape[:cut]) + tuple(self.out_shape)

    @override
    def inverse_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        """Replace the trailing ``out_shape`` dimensions with ``in_shape``."""
        if len(shape) < len(self.out_shape):
            return tuple(shape)
        cut = len(shape) - len(self.out_shape)
        return tuple(shape[:cut]) + tuple(self.in_shape)

    @override
    def _call(self, x: Tensor) -> Tensor:
        """Reshape the event tail of ``x`` from ``in_shape`` to ``out_shape``."""
        batch = x.shape[: x.dim() - len(self.in_shape)]
        return x.reshape(*batch, *self.out_shape)

    @override
    def _inverse(self, y: Tensor) -> Tensor:
        """Reshape the event tail of ``y`` from ``out_shape`` back to ``in_shape``."""
        batch = y.shape[: y.dim() - len(self.out_shape)]
        return y.reshape(*batch, *self.in_shape)

    @override
    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        """Zero Jacobian — pure reshape is volume-preserving."""
        batch = x.shape[: x.dim() - len(self.in_shape)]
        return lucid.zeros(tuple(batch), dtype=x.dtype, device=x.device)


class TransformedDistribution(Distribution):
    r"""Pushforward of a base distribution through a (composite) bijector.

    Constructs a new :class:`Distribution` whose samples are obtained by
    pushing samples from ``base_distribution`` through the supplied
    chain of :class:`Transform` instances, with :meth:`log_prob`
    accounting for the Jacobian correction via the change-of-variable
    formula.  This is the canonical way to build normalising flows in
    Lucid: stack any number of bijections on top of a tractable base
    (typically a Normal) to obtain expressive densities while retaining
    exact log-likelihood evaluation and reparameterised sampling.

    Parameters
    ----------
    base_distribution : Distribution
        Underlying distribution whose samples will be pushed through
        ``transforms``.
    transforms : Transform or list[Transform]
        A single transform or an ordered list applied left-to-right.
        Internally wrapped in a list.
    validate_args : bool, optional
        Forwarded to :class:`Distribution`.

    Notes
    -----
    Sampling (with :math:`T = T_n \circ \cdots \circ T_1` the composite
    bijector):

    .. math::

        Y = T(X), \quad X \sim p_{\text{base}}

    Reparameterised sampling is available iff the base distribution
    supports it (``has_rsample`` is forwarded).

    Density (change of variables):

    .. math::

        \log p_Y(\mathbf{y}) =
            \log p_X(T^{-1}(\mathbf{y}))
            - \sum_{i=1}^{n}
                \log\!\left|\det
                \frac{\partial T_i(\mathbf{x}_{i-1})}{\partial \mathbf{x}_{i-1}}\right|

    where :math:`\mathbf{x}_0 = T^{-1}(\mathbf{y})` and
    :math:`\mathbf{x}_i = T_i(\mathbf{x}_{i-1})`.  The implementation
    walks the transforms in reverse, inverting one step at a time and
    accumulating the Jacobian correction.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import Normal
    >>> from lucid.distributions.transforms import (
    ...     ExpTransform, TransformedDistribution)
    >>> # LogNormal = ExpTransform(Normal(0, 1))
    >>> log_normal = TransformedDistribution(Normal(loc=0.0, scale=1.0), [ExpTransform()])
    >>> log_normal.rsample((4,)).shape
    (4,)
    >>> log_normal.log_prob(lucid.tensor(1.0))  # log N(0 | 0, 1) - log(1)
    tensor(-0.9189)
    """

    def __init__(
        self,
        base_distribution: Distribution,
        transforms: Transform | list[Transform],
        validate_args: bool | None = None,
    ) -> None:
        """Construct a transformed distribution.

        Parameters
        ----------
        base_distribution : Distribution
            Underlying distribution whose samples will be pushed through
            ``transforms``.
        transforms : Transform | list[Transform]
            A single transform or an ordered list applied left-to-right.
        validate_args : bool | None, optional
            Forwarded to :class:`Distribution`.
        """
        if isinstance(transforms, Transform):
            self.transforms: list[Transform] = [transforms]
        else:
            self.transforms = list(transforms)
        base_shape = tuple(base_distribution.batch_shape) + tuple(base_distribution.event_shape)
        base_event_dim = len(base_distribution.event_shape)
        if self.transforms:
            transform = ComposeTransform(self.transforms)
            forward_shape = transform.forward_shape(base_shape)
            reinterpreted_batch_ndims = transform.domain.event_dim - base_event_dim
            if reinterpreted_batch_ndims > 0 and len(base_shape) >= transform.domain.event_dim:
                base_distribution = Independent(
                    base_distribution, reinterpreted_batch_ndims, validate_args=validate_args
                )
            transform_change_in_event_dim = (
                transform.codomain.event_dim - transform.domain.event_dim
            )
            event_dim = max(
                transform.codomain.event_dim, base_event_dim + transform_change_in_event_dim
            )
            cut = max(0, len(forward_shape) - event_dim)
            batch_shape = forward_shape[:cut]
            event_shape = forward_shape[cut:]
        else:
            batch_shape = tuple(base_distribution.batch_shape)
            event_shape = tuple(base_distribution.event_shape)
        self.base_dist = base_distribution
        super().__init__(
            batch_shape=batch_shape, event_shape=event_shape, validate_args=validate_args
        )

    @override
    @property
    def support(self) -> Constraint | None:
        """Support of the transformed distribution — codomain of the outer transform."""
        if not self.transforms:
            return self.base_dist.support
        support = self.transforms[-1].codomain
        if len(self.event_shape) > support.event_dim:
            support = constraints.independent(support, len(self.event_shape) - support.event_dim)
        return support

    def _has_rsample(self) -> bool:
        """Whether reparameterised sampling is available — inherited from the base distribution."""
        return self.base_dist.has_rsample

    has_rsample = property(_has_rsample, doc=_has_rsample.__doc__)

    def _push(self, x: Tensor) -> Tensor:
        """Apply every transform in ``self.transforms`` in order to ``x``."""
        for t in self.transforms:
            x = t(x)
        return x

    @override
    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        """Push a reparameterised base sample through the transform chain."""
        return self._push(self.base_dist.rsample(sample_shape))

    @override
    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        """Push a (non-reparameterised) base sample through the transform chain."""
        return self._push(self.base_dist.sample(sample_shape))

    @override
    def log_prob(self, value: Tensor) -> Tensor:
        r"""Evaluate the log-density of ``value`` under the transformed distribution.

        Uses the change-of-variable formula:

        .. math::

            \log p(y) = \log p_{\text{base}}(T^{-1}(y))
                - \sum_i \log\!\bigl|\det \partial T_i / \partial x_i\bigr|

        with the chain unwound by walking the transforms in reverse.
        """
        event_dim = len(self.event_shape)
        log_det: Tensor | float = 0.0
        cur = value
        for t in reversed(self.transforms):
            prev = t._inverse(cur)
            event_dim += t.domain.event_dim - t.codomain.event_dim
            log_det = log_det + _sum_rightmost(
                t.log_abs_det_jacobian(prev, cur), event_dim - t.domain.event_dim
            )
            cur = prev
        return (
            _sum_rightmost(
                self.base_dist.log_prob(cur), event_dim - len(self.base_dist.event_shape)
            )
            - log_det
        )

    def _orientation(self) -> int | Tensor:
        """Product of the transforms' signs: ``+1`` when the chain increases.

        Read before anything is evaluated, so a chain containing a transform
        with no sign — and so no CDF — is refused for that reason, rather
        than by whatever its inverse does to the point first.
        """
        sign: int | Tensor = 1
        for transform in self.transforms:
            sign = sign * transform.sign
        return sign

    @staticmethod
    def _monotonize_cdf(value: Tensor, sign: int | Tensor) -> Tensor:
        """Flip ``value -> 1 - value`` when ``sign`` says the chain decreases.

        The CDF of ``T(X)`` is ``F_X(T^-1(y))`` for an increasing ``T`` and
        ``1 - F_X(T^-1(y))`` for a decreasing one; the quantile is the same
        flip applied to the probability.  ``sign`` may be a tensor (a negative
        ``scale`` in one element of a batch, say), in which case the flip is
        per element.
        """
        if isinstance(sign, int) and sign == 1:
            return value
        return sign * (value - 0.5) + 0.5

    @override
    def cdf(self, value: Tensor) -> Tensor:
        r"""Cumulative distribution function of the transformed variable.

        Inverts the transform chain, scores the preimage under the base
        distribution's CDF, and flips the answer to :math:`1 - F` when the
        chain is decreasing:

        .. math::

            F_Y(y) = F_X(T^{-1}(y)) \quad (T \text{ increasing}),
            \qquad 1 - F_X(T^{-1}(y)) \quad (T \text{ decreasing}).

        Parameters
        ----------
        value : Tensor
            Point(s) of the transformed variable.

        Returns
        -------
        Tensor
            Probabilities in :math:`[0, 1]`.

        Raises
        ------
        NotImplementedError
            If the base distribution has no CDF, or a transform in the
            chain has no defined :attr:`~Transform.sign`.

        Examples
        --------
        >>> import lucid
        >>> from lucid.distributions import Normal
        >>> from lucid.distributions.transforms import ExpTransform, TransformedDistribution
        >>> d = TransformedDistribution(Normal(0.0, 1.0), [ExpTransform()])
        >>> d.cdf(lucid.tensor(1.0))  # the median of a LogNormal(0, 1) is 1
        tensor(0.5)
        """
        sign = self._orientation()
        for transform in self.transforms[::-1]:
            value = transform.inv(value)
        return self._monotonize_cdf(self.base_dist.cdf(value), sign)

    @override
    def icdf(self, value: Tensor) -> Tensor:
        r"""Inverse CDF (quantile function) of the transformed variable.

        Flips the probability to :math:`1 - p` when the chain is decreasing,
        takes the base distribution's quantile, and pushes it forward through
        the transforms.

        Parameters
        ----------
        value : Tensor
            Probabilities in :math:`[0, 1]`.

        Returns
        -------
        Tensor
            Quantiles of the transformed variable.

        Raises
        ------
        NotImplementedError
            If the base distribution has no inverse CDF, or a transform in
            the chain has no defined :attr:`~Transform.sign`.

        Examples
        --------
        >>> import lucid
        >>> from lucid.distributions import Normal
        >>> from lucid.distributions.transforms import AffineTransform, TransformedDistribution
        >>> d = TransformedDistribution(Normal(0.0, 1.0), [AffineTransform(0.0, -1.0)])
        >>> d.icdf(lucid.tensor(0.975))  # a decreasing map reflects the quantile
        tensor(1.96)
        """
        return self._push(self.base_dist.icdf(self._monotonize_cdf(value, self._orientation())))

    @override
    def expand(self, batch_shape: Sequence[int], _instance: Self | None = None) -> Self:
        """Return a new :class:`TransformedDistribution` instance with parameters expanded to ``batch_shape``."""
        new = self._get_checked_instance(TransformedDistribution, _instance)
        batch_shape = tuple(batch_shape)
        shape = batch_shape + self.event_shape
        for t in reversed(self.transforms):
            shape = t.inverse_shape(shape)
        base_batch_shape = shape[: len(shape) - len(self.base_dist.event_shape)]
        new.base_dist = self.base_dist.expand(base_batch_shape)
        new.transforms = self.transforms
        super(TransformedDistribution, new).__init__(
            batch_shape=batch_shape, event_shape=self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new
