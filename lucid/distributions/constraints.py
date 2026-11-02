"""Domain constraints for ``lucid.distributions``.

A :class:`Constraint` answers the question *"is this value in the support
of some random variable?"* via :meth:`Constraint.check`, which returns a
boolean :class:`Tensor` broadcast over the input.

The set mirrors the reference framework's: each constraint admits exactly
the values the reference's constraint of the same name admits, so a
distribution's ``arg_constraints`` and ``support`` describe the same domain
in both (``lucid/test/parity/test_distribution_constraints_parity.py``).
Every domain has one definition here — an interval with either end open is
one class, not one class per combination.
"""

from collections.abc import Callable
from typing import Self, final, overload, override

import lucid
from lucid._dtype import finfo
from lucid._tensor.tensor import Tensor


__all__ = [
    "Constraint",
    "MixtureSameFamilyConstraint",
    "boolean",
    "cat",
    "corr_cholesky",
    "dependent",
    "dependent_property",
    "greater_than",
    "greater_than_eq",
    "half_open_interval",
    "independent",
    "integer_interval",
    "interval",
    "is_dependent",
    "less_than",
    "lower_cholesky",
    "lower_triangular",
    "multinomial",
    "nonnegative",
    "nonnegative_integer",
    "one_hot",
    "open_unit_interval",
    "positive",
    "positive_definite",
    "positive_integer",
    "positive_semidefinite",
    "real",
    "real_vector",
    "simplex",
    "square",
    "stack",
    "symmetric",
    "unit_interval",
]


class Constraint:
    """Base class for every domain constraint in ``lucid.distributions``.

    A constraint encapsulates the *support* of a distribution — the
    set of values its log-probability is defined on.  Subclasses override
    :meth:`check` to return a boolean tensor flagging which entries of
    the candidate input fall inside the support.

    Subclasses also override the two class attributes below when
    relevant: ``is_discrete=True`` for lattice supports
    (Categorical / Bernoulli / …) and ``event_dim>0`` for multivariate
    supports where the constraint applies to a trailing batch of axes
    rather than a single scalar.

    Attributes
    ----------
    is_discrete : bool
        ``True`` when the support is a discrete set; ``False`` (default)
        for continuous reals / non-negative reals / simplex / …
    event_dim : int
        Number of trailing dimensions the constraint applies to as a
        unit.  Default ``0`` (scalar per element).
    """

    is_discrete: bool = False
    event_dim: int = 0

    def check(self, value: Tensor) -> Tensor:
        """Test whether ``value`` lies in the constraint's support.

        Parameters
        ----------
        value : Tensor
            Candidate values to validate.

        Returns
        -------
        Tensor
            Boolean tensor broadcast over ``value`` — ``True`` where the
            element satisfies the constraint.

        Raises
        ------
        NotImplementedError
            The base class does not implement a concrete predicate.
        """
        raise NotImplementedError(f"{type(self).__name__}.check is not implemented")

    @override
    def __repr__(self) -> str:
        """Return a developer-facing string representation of the instance."""
        return f"{type(self).__name__.lstrip('_')}()"


def _all_trailing(result: Tensor, ndims: int) -> Tensor:
    """``result`` reduced with logical *and* over its last ``ndims`` axes.

    ``Tensor.all`` reduces to a single 0-dim answer, and an event-shaped
    constraint needs one answer per batch element; the minimum of a boolean
    tensor along an axis is the *and* along it.
    """
    for _ in range(ndims):
        result = result.min(dim=-1).values
    return result


def _is_integral(value: Tensor) -> Tensor:
    """Element-wise: ``value`` is a finite whole number.

    ``floor(x) == x`` alone holds for ``±inf``, which is not a count; the
    reference framework's ``x % 1 == 0`` refuses it.
    """
    return lucid.isfinite(value) & (lucid.floor(value) == value)


class _StaticAttribute[T]:
    """A constraint attribute a :class:`_DependentBase` may leave undetermined.

    ``is_discrete`` and ``event_dim`` are plain class attributes on every
    other constraint.  A dependent one stands for a support that follows
    other variables, so they are usually not known until the support is
    built; reading one that was never declared raises instead of answering
    with a default that would be wrong half the time.
    """

    def __init__(self) -> None:
        self._name = ""

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = name

    @overload
    def __get__(self, instance: None, owner: type | None = None) -> Self: ...

    @overload
    def __get__(self, instance: object, owner: type | None = None) -> T: ...

    def __get__(self, instance: object | None, owner: type | None = None) -> Self | T:
        if instance is None:
            return self
        value: T | None = vars(instance).get("_" + self._name)
        if value is None:
            raise NotImplementedError(f".{self._name} cannot be determined statically")
        return value

    def __set__(self, instance: object, value: T | None) -> None:
        vars(instance)["_" + self._name] = value


class _DependentBase(Constraint):
    """What :func:`is_dependent` recognises: a support that depends on other variables.

    These variables obey no simple coordinate-wise constraint, so
    :meth:`check` cannot answer.  ``is_discrete`` and ``event_dim`` may be
    declared when they can be known statically and raise
    :class:`NotImplementedError` when read otherwise.
    """

    is_discrete = _StaticAttribute[bool]()
    event_dim = _StaticAttribute[int]()

    def __init__(self, *, is_discrete: bool | None = None, event_dim: int | None = None) -> None:
        """Declare ``is_discrete`` and ``event_dim`` where they are known."""
        self.is_discrete = is_discrete
        self.event_dim = event_dim

    @override
    def check(self, value: Tensor) -> Tensor:
        """Refuse: validity of a dependent constraint cannot be determined.

        Raises
        ------
        ValueError
            Always.
        """
        raise ValueError("Cannot determine validity of dependent constraint")

    def _declared(self) -> tuple[bool | None, int | None]:
        """``(is_discrete, event_dim)`` as declared, ``None`` for each left undetermined."""
        return vars(self).get("_is_discrete"), vars(self).get("_event_dim")


@final
class _Dependent(_DependentBase):
    """Placeholder for a support that depends on other variables.

    Calling it with ``is_discrete`` and/or ``event_dim`` returns a copy with
    those declared, keeping whatever it already declared for the other.

    Examples
    --------
    >>> from lucid.distributions import constraints
    >>> constraints.is_dependent(constraints.dependent(event_dim=1))
    True
    >>> constraints.dependent(event_dim=1).event_dim
    1
    """

    def __call__(
        self, *, is_discrete: bool | None = None, event_dim: int | None = None
    ) -> _Dependent:
        """Return a copy declaring ``is_discrete`` and ``event_dim`` where given."""
        declared_discrete, declared_event_dim = self._declared()
        return _Dependent(
            is_discrete=declared_discrete if is_discrete is None else is_discrete,
            event_dim=declared_event_dim if event_dim is None else event_dim,
        )


class _DependentProperty(property, _DependentBase):
    """A property that is also a :class:`_Dependent` constraint when read off the class.

    ``@dependent_property`` turns a method into a property whose value is
    computed per instance, while ``Class.support`` — the property object
    itself — answers :func:`is_dependent` and carries the static
    ``is_discrete`` / ``event_dim`` the decorator declared.  This is how a
    distribution says "my support is built from my parameters" without
    building it.

    Parameters
    ----------
    fn : callable, optional
        The method to wrap.  Omitted when the decorator is called with
        keyword arguments first, ``@dependent_property(is_discrete=True)``.
    is_discrete : bool, optional
        Static value of ``is_discrete``, if it can be known.
    event_dim : int, optional
        Static value of ``event_dim``, if it can be known.

    Examples
    --------
    >>> from lucid.distributions import constraints
    >>> class Box:
    ...     def __init__(self, high):
    ...         self.high = high
    ...     @constraints.dependent_property(is_discrete=False, event_dim=0)
    ...     def support(self):
    ...         return constraints.interval(0.0, self.high)
    >>> constraints.is_dependent(Box.support)
    True
    >>> Box(2.0).support
    Interval(lower_bound=0.0, upper_bound=2.0)
    """

    def __init__(
        self,
        fn: Callable[..., Constraint] | None = None,
        *,
        is_discrete: bool | None = None,
        event_dim: int | None = None,
    ) -> None:
        """Wrap ``fn`` as a property and declare the static attributes."""
        property.__init__(self, fn)
        _DependentBase.__init__(self, is_discrete=is_discrete, event_dim=event_dim)

    def __call__(self, fn: Callable[..., Constraint]) -> _DependentProperty:
        """Wrap ``fn`` keeping the declared ``is_discrete`` / ``event_dim``."""
        is_discrete, event_dim = self._declared()
        return _DependentProperty(fn, is_discrete=is_discrete, event_dim=event_dim)


def is_dependent(constraint: object) -> bool:
    """Whether ``constraint`` is a placeholder for a support that depends on other variables.

    Parameters
    ----------
    constraint : object
        A :class:`Constraint` — ``dependent``, a ``dependent_property`` read
        off its class, or any other constraint.

    Returns
    -------
    bool
        ``True`` for ``dependent`` and for ``dependent_property``; ``False``
        for every other constraint, whose support is a fixed domain.

    Examples
    --------
    >>> from lucid.distributions import constraints
    >>> constraints.is_dependent(constraints.dependent)
    True
    >>> constraints.is_dependent(constraints.positive)
    False
    """
    return isinstance(constraint, _DependentBase)


@final
class _Real(Constraint):
    """``ℝ`` extended by ``±inf`` — everything except NaN.

    Infinities are admitted on purpose, as the reference framework admits
    them: a logit of ``-inf`` is how a category is masked out
    (``Categorical(logits=[-inf, 0, 0])`` is the action-mask idiom), and a
    ``Normal`` scores ``±inf`` as ``-inf`` rather than refusing it.  NaN is
    the one value that is never a meaningful parameter or observation.
    """

    @override
    def check(self, value: Tensor) -> Tensor:
        """Return ``True`` element-wise where ``value`` is not NaN."""
        return ~lucid.isnan(value)


@final
class _Boolean(Constraint):
    """``{0, 1}``."""

    is_discrete = True

    @override
    def check(self, value: Tensor) -> Tensor:
        """Return ``True`` element-wise where ``value`` equals ``0`` or ``1``."""
        zero = lucid.zeros_like(value)
        one = lucid.ones_like(value)
        return (value == zero) | (value == one)


@final
class _GreaterThan(Constraint):
    """``(lower_bound, ∞)`` — ``positive`` is this with a bound of ``0``.

    The bound may be a tensor, compared element by element.
    """

    def __init__(self, lower_bound: float | Tensor) -> None:
        """Store the strict lower bound used by :meth:`check`."""
        self.lower_bound = lower_bound

    @override
    def check(self, value: Tensor) -> Tensor:
        """Return ``True`` element-wise where ``value > lower_bound``."""
        return value > self.lower_bound

    @override
    def __repr__(self) -> str:
        return f"GreaterThan(lower_bound={self.lower_bound})"


@final
class _GreaterThanEq(Constraint):
    """``[lower_bound, ∞)`` — ``nonnegative`` is this with a bound of ``0``.

    The bound may be a tensor, compared element by element: ``Pareto``'s
    support is ``[scale, ∞)`` for each of its batch members.
    """

    def __init__(self, lower_bound: float | Tensor) -> None:
        """Store the non-strict lower bound used by :meth:`check`."""
        self.lower_bound = lower_bound

    @override
    def check(self, value: Tensor) -> Tensor:
        """Return ``True`` element-wise where ``value >= lower_bound``."""
        return value >= self.lower_bound

    @override
    def __repr__(self) -> str:
        return f"GreaterThanEq(lower_bound={self.lower_bound})"


@final
class _LessThan(Constraint):
    """``(-∞, upper_bound)``.  The bound may be a tensor."""

    def __init__(self, upper_bound: float | Tensor) -> None:
        """Store the strict upper bound used by :meth:`check`."""
        self.upper_bound = upper_bound

    @override
    def check(self, value: Tensor) -> Tensor:
        """Return ``True`` element-wise where ``value < upper_bound``."""
        return value < self.upper_bound

    @override
    def __repr__(self) -> str:
        return f"LessThan(upper_bound={self.upper_bound})"


@final
class _Interval(Constraint):
    """An interval of the real line whose ends are each open or closed.

    One definition serves ``[a, b]`` (:func:`interval`, ``unit_interval``),
    ``[a, b)`` (:func:`half_open_interval`), ``(a, b)``
    (``open_unit_interval``) and ``(a, b]``.  The bounds may be tensors,
    compared element by element: ``Uniform``'s support is ``[low, high]``
    for each of its batch members.
    """

    def __init__(
        self,
        lower_bound: float | Tensor,
        upper_bound: float | Tensor,
        *,
        lower_open: bool = False,
        upper_open: bool = False,
    ) -> None:
        """Store the bounds, and whether each end excludes its bound."""
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.lower_open = lower_open
        self.upper_open = upper_open

    @override
    def check(self, value: Tensor) -> Tensor:
        """Return ``True`` element-wise where ``value`` lies between the bounds.

        An open end excludes its bound, a closed end includes it.
        """
        above = value > self.lower_bound if self.lower_open else value >= self.lower_bound
        below = value < self.upper_bound if self.upper_open else value <= self.upper_bound
        return above & below

    @override
    def __repr__(self) -> str:
        name = {
            (False, False): "Interval",
            (False, True): "HalfOpenInterval",
            (True, False): "LeftOpenInterval",
            (True, True): "OpenInterval",
        }[(self.lower_open, self.upper_open)]
        return f"{name}(lower_bound={self.lower_bound}, upper_bound={self.upper_bound})"


@final
class _IntegerInterval(Constraint):
    """``{lower_bound, lower_bound+1, ..., upper_bound}``.

    The bounds may be tensors: ``Binomial``'s support is
    ``{0, …, total_count}`` for each of its batch members.
    """

    is_discrete = True

    def __init__(self, lower_bound: int | Tensor, upper_bound: int | Tensor) -> None:
        """Store the inclusive integer interval bounds used by :meth:`check`."""
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    @override
    def check(self, value: Tensor) -> Tensor:
        """Return ``True`` element-wise where ``value`` is a whole number in
        ``[lower_bound, upper_bound]``."""
        in_range = (value >= self.lower_bound) & (value <= self.upper_bound)
        return in_range & _is_integral(value)

    @override
    def __repr__(self) -> str:
        return (
            f"IntegerInterval(lower_bound={self.lower_bound}, " f"upper_bound={self.upper_bound})"
        )


@final
class _IntegerGreaterThan(Constraint):
    """``{lower_bound, lower_bound+1, ...}`` — ``nonnegative_integer`` is
    this with a bound of ``0``."""

    is_discrete = True

    def __init__(self, lower_bound: int) -> None:
        """Store the inclusive lower bound used by :meth:`check`."""
        self.lower_bound = lower_bound

    @override
    def check(self, value: Tensor) -> Tensor:
        """Return ``True`` element-wise where ``value`` is a whole number
        ``>= lower_bound`` (``inf`` is not)."""
        return (value >= self.lower_bound) & _is_integral(value)

    @override
    def __repr__(self) -> str:
        return f"IntegerGreaterThan(lower_bound={self.lower_bound})"


@final
class _Simplex(Constraint):
    """The K-simplex: ``x ≥ 0`` and ``Σ x = 1`` along the last dimension."""

    event_dim = 1

    @override
    def check(self, value: Tensor) -> Tensor:
        """Return ``True`` where ``value`` lies on the unit simplex.

        Verifies non-negativity of every component and that the sum along
        the last axis equals ``1`` (within a ``1e-6`` tolerance).
        """
        nonneg = value.min(dim=-1).values >= 0
        sums = value.sum(dim=-1)
        unit = (sums - 1.0).abs() < 1e-6
        return nonneg & unit


@final
class _Multinomial(Constraint):
    """Count vectors of at most ``upper_bound`` draws: entries ``>= 0`` along
    the last dimension, summing to at most ``upper_bound`` — the support of
    :class:`~lucid.distributions.Multinomial`.

    The reference framework checks this weaker condition rather than
    whole counts summing to exactly ``upper_bound``, and so does this.
    """

    is_discrete = True
    event_dim = 1

    def __init__(self, upper_bound: float | Tensor) -> None:
        """Store the most draws a count vector may hold."""
        self.upper_bound = upper_bound

    @override
    def check(self, value: Tensor) -> Tensor:
        """Return ``True`` per vector where it is non-negative and sums to at
        most ``upper_bound``."""
        nonneg = _all_trailing(value >= 0, 1)
        return nonneg & (value.sum(dim=-1) <= self.upper_bound)

    @override
    def __repr__(self) -> str:
        return f"Multinomial(upper_bound={self.upper_bound})"


class _LowerTriangular(Constraint):
    """Lower-triangular matrices — everything above the diagonal is zero."""

    event_dim = 2

    @override
    def check(self, value: Tensor) -> Tensor:
        """Return ``True`` per matrix where it equals its lower triangle."""
        return _all_trailing(lucid.tril(value) == value, 2)


class _LowerCholesky(_LowerTriangular):
    """Lower-triangular square matrices with a positive diagonal — a
    Cholesky factor, the domain of a ``scale_tril``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import constraints
    >>> constraints.lower_cholesky.check(lucid.tensor([[1.0, 0.0], [0.5, 2.0]]))
    tensor(True, dtype=lucid.bool)
    >>> constraints.lower_cholesky.check(lucid.tensor([[-1.0, 0.0], [0.5, 2.0]]))
    tensor(False, dtype=lucid.bool)
    """

    @override
    def check(self, value: Tensor) -> Tensor:
        """Return ``True`` per matrix where it is lower-triangular with a
        positive diagonal."""
        positive_diagonal = _all_trailing(value.diagonal(dim1=-2, dim2=-1) > 0, 1)
        return super().check(value) & positive_diagonal


@final
class _OneHot(Constraint):
    """One-hot vectors: entries in ``{0, 1}`` that sum to ``1`` along the
    last dimension — the support of
    :class:`~lucid.distributions.OneHotCategorical`.

    A point inside the simplex such as ``[0.2, 0.3, 0.5]`` is not a one-hot
    sample, so ``simplex`` is not this support.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import constraints
    >>> constraints.one_hot.check(lucid.tensor([[0.0, 1.0, 0.0], [0.2, 0.3, 0.5]]))
    tensor([True, False], dtype=lucid.bool)
    """

    is_discrete = True
    event_dim = 1

    @override
    def check(self, value: Tensor) -> Tensor:
        """Return ``True`` per vector where every entry is 0 or 1 and they sum to 1."""
        binary = _all_trailing((value == 0) | (value == 1), 1)
        return binary & (value.sum(dim=-1) == 1)


def _is_square(value: Tensor) -> bool:
    """Whether the last two axes of ``value`` have the same size."""
    return value.shape[-1] == value.shape[-2]


class _Square(Constraint):
    """Square matrices — the last two axes are the same size."""

    event_dim = 2

    @override
    def check(self, value: Tensor) -> Tensor:
        """Return ``True`` for every matrix when the last two axes match,
        ``False`` for every one when they do not."""
        return lucid.full(
            tuple(value.shape[:-2]), _is_square(value), dtype=lucid.bool, device=value.device
        )


class _Symmetric(_Square):
    """Symmetric square matrices, compared within ``1e-6``."""

    @override
    def check(self, value: Tensor) -> Tensor:
        """Return ``True`` per matrix where it is square and equals its transpose."""
        square = super().check(value)
        if not _is_square(value):
            return square
        return _all_trailing(lucid.isclose(value, value.mT, atol=1e-6), 2)


class _PositiveSemidefinite(_Symmetric):
    """Symmetric positive-semidefinite matrices: every eigenvalue is ``>= 0``."""

    @override
    def check(self, value: Tensor) -> Tensor:
        """Return ``True`` per matrix where it is symmetric with no negative eigenvalue."""
        symmetric = super().check(value)
        if not _is_square(value):
            return symmetric
        return symmetric & _all_trailing(lucid.linalg.eigvalsh(value) >= 0, 1)


class _PositiveDefinite(_Symmetric):
    """Symmetric positive-definite matrices.

    Symmetry is checked first, within ``1e-6``: a Cholesky factorisation
    reads only the lower triangle, so on its own it accepts any matrix
    whose lower triangle happens to be the lower triangle of an SPD one —
    a Cholesky *factor* among them.
    """

    @override
    def check(self, value: Tensor) -> Tensor:
        """Return ``True`` per matrix where it is symmetric and has a
        Cholesky factorisation."""
        symmetric = super().check(value)
        if not _is_square(value):
            return symmetric
        _, info = lucid.linalg.cholesky_ex(value)
        return symmetric & (info == 0)


@final
class _CorrCholesky(_LowerCholesky):
    """Cholesky factors of correlation matrices: lower-triangular, positive
    diagonal, every row of unit Euclidean norm — the support of
    :class:`~lucid.distributions.LKJCholesky`.

    The row norm is compared within ``10 · K · eps`` of 1, the tolerance
    the reference framework uses.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import constraints
    >>> constraints.corr_cholesky.check(lucid.tensor([[1.0, 0.0], [0.6, 0.8]]))
    tensor(True, dtype=lucid.bool)
    >>> constraints.corr_cholesky.check(lucid.tensor([[1.0, 0.0], [0.5, 0.5]]))
    tensor(False, dtype=lucid.bool)
    """

    @override
    def check(self, value: Tensor) -> Tensor:
        """Return ``True`` per matrix where it is the Cholesky factor of a
        correlation matrix."""
        tol = float(finfo(value.dtype).eps) * value.shape[-1] * 10
        row_norm = lucid.linalg.vector_norm(value.detach(), dim=-1)
        unit_rows = _all_trailing((row_norm - 1.0).abs() <= tol, 1)
        return super().check(value) & unit_rows


@final
class _IndependentConstraint(Constraint):
    """A constraint applied to each element, read as one answer per event.

    Wraps ``base_constraint`` so its last ``reinterpreted_batch_ndims`` batch
    axes count as event axes: the answer is ``True`` for an event only when
    every element of it satisfies the base constraint.  This is the support
    of :class:`~lucid.distributions.Independent`.
    """

    def __init__(self, base_constraint: Constraint, reinterpreted_batch_ndims: int) -> None:
        """Store the wrapped constraint and how many batch axes it absorbs."""
        if reinterpreted_batch_ndims < 0:
            raise ValueError(
                "independent: reinterpreted_batch_ndims must be >= 0, got "
                f"{reinterpreted_batch_ndims}"
            )
        self.base_constraint = base_constraint
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims
        self.is_discrete = base_constraint.is_discrete
        self.event_dim = base_constraint.event_dim + reinterpreted_batch_ndims

    @override
    def check(self, value: Tensor) -> Tensor:
        """Return the base check reduced with *and* over the reinterpreted axes.

        Raises
        ------
        ValueError
            If ``value`` has fewer dimensions than the event needs.
        """
        result = self.base_constraint.check(value)
        if result.ndim < self.reinterpreted_batch_ndims:
            raise ValueError(
                f"independent: expected value.ndim >= {self.event_dim}, " f"got {value.ndim}"
            )
        return _all_trailing(result, self.reinterpreted_batch_ndims)

    @override
    def __repr__(self) -> str:
        return (
            f"IndependentConstraint({self.base_constraint!r}, " f"{self.reinterpreted_batch_ndims})"
        )


@final
class MixtureSameFamilyConstraint(Constraint):
    """The support of a :class:`~lucid.distributions.MixtureSameFamily`.

    The components carry a mixture axis that a sample does not: a mixture
    value of shape ``batch + event`` is scored against components of shape
    ``batch + (K,) + event``.  This constraint puts that axis back, checks
    the value against the support of every component, and answers ``True``
    for a value only when all of them admit it.

    Parameters
    ----------
    base_constraint : Constraint
        The component distribution's support.

    Raises
    ------
    TypeError
        If ``base_constraint`` is not a :class:`Constraint`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import constraints
    >>> c = constraints.MixtureSameFamilyConstraint(constraints.positive)
    >>> c.check(lucid.tensor([1.0, -1.0]))
    tensor([True, False], dtype=lucid.bool)
    """

    def __init__(self, base_constraint: Constraint) -> None:
        """Store the component support; ``is_discrete`` and ``event_dim`` are its."""
        if not isinstance(base_constraint, Constraint):
            raise TypeError(
                "MixtureSameFamilyConstraint: base_constraint must be a Constraint, "
                f"got {type(base_constraint).__name__}"
            )
        self.base_constraint = base_constraint
        self.is_discrete = base_constraint.is_discrete
        self.event_dim = base_constraint.event_dim

    @override
    def check(self, value: Tensor) -> Tensor:
        """Check ``value`` against the component support with the mixture axis restored.

        Raises
        ------
        ValueError
            If ``value`` has fewer dimensions than the event needs.
        """
        if value.ndim < self.event_dim:
            raise ValueError(f"Expected value.ndim >= {self.event_dim} but got {value.ndim}")
        result = self.base_constraint.check(value.unsqueeze(-1 - self.event_dim))
        kept = value.ndim - self.event_dim
        return result.reshape(*result.shape[:kept], -1).min(dim=-1).values

    @override
    def __repr__(self) -> str:
        return f"MixtureSameFamilyConstraint({self.base_constraint!r})"


# Public singletons (functions/objects users actually reach for).
real = _Real()
boolean = _Boolean()
positive = _GreaterThan(0.0)
nonnegative = _GreaterThanEq(0.0)
unit_interval = _Interval(0.0, 1.0)
open_unit_interval = _Interval(0.0, 1.0, lower_open=True, upper_open=True)
simplex = _Simplex()
one_hot = _OneHot()
positive_definite = _PositiveDefinite()
corr_cholesky = _CorrCholesky()
lower_cholesky = _LowerCholesky()
lower_triangular = _LowerTriangular()
square = _Square()
symmetric = _Symmetric()
positive_semidefinite = _PositiveSemidefinite()
nonnegative_integer = _IntegerGreaterThan(0)
positive_integer = _IntegerGreaterThan(1)
dependent = _Dependent()
dependent_property = _DependentProperty
# ``ℝⁿ`` read as one event: a vector with no NaN in it.
real_vector = _IndependentConstraint(real, 1)

# ``(0, 1]`` — a success probability that may be certain but not impossible
# (``Geometric``: with p = 0 no trial ever succeeds).  The reference
# framework declares ``unit_interval`` and refuses 0 in the constructor;
# held here, the table is the whole domain.  It has no reference name, so
# it is not public.
_positive_unit_interval = _Interval(0.0, 1.0, lower_open=True)


def greater_than(lower_bound: float | Tensor) -> Constraint:
    """Construct a strict-greater-than constraint ``(lower_bound, ∞)``.

    Parameters
    ----------
    lower_bound : float or Tensor
        Strict lower bound for the support; a tensor is compared element by
        element.

    Returns
    -------
    Constraint
        Constraint instance accepting values ``> lower_bound``.
    """
    return _GreaterThan(lower_bound)


def greater_than_eq(lower_bound: float | Tensor) -> Constraint:
    """Construct a non-strict-greater-than constraint ``[lower_bound, ∞)``.

    Parameters
    ----------
    lower_bound : float or Tensor
        Inclusive lower bound for the support; a tensor is compared element
        by element.

    Returns
    -------
    Constraint
        Constraint instance accepting values ``>= lower_bound``.
    """
    return _GreaterThanEq(lower_bound)


def less_than(upper_bound: float | Tensor) -> Constraint:
    """Construct a strict-less-than constraint ``(-∞, upper_bound)``.

    Parameters
    ----------
    upper_bound : float or Tensor
        Strict upper bound for the support; a tensor is compared element by
        element.

    Returns
    -------
    Constraint
        Constraint instance accepting values ``< upper_bound``.
    """
    return _LessThan(upper_bound)


def interval(lower_bound: float | Tensor, upper_bound: float | Tensor) -> Constraint:
    """Construct a closed-interval constraint ``[lower_bound, upper_bound]``.

    Parameters
    ----------
    lower_bound, upper_bound : float or Tensor
        Inclusive bounds for the support; tensors are compared element by
        element.

    Returns
    -------
    Constraint
        Constraint instance accepting values in ``[lower_bound, upper_bound]``.
    """
    return _Interval(lower_bound, upper_bound)


def half_open_interval(lower_bound: float | Tensor, upper_bound: float | Tensor) -> Constraint:
    """Construct a half-open interval constraint ``[lower_bound, upper_bound)``.

    The domain of a probability that may be impossible but not certain —
    :class:`~lucid.distributions.NegativeBinomial`'s ``probs``, where
    ``p = 1`` would mean every trial fails and the count never ends.

    Parameters
    ----------
    lower_bound : float or Tensor
        Inclusive lower bound.
    upper_bound : float or Tensor
        Exclusive upper bound.

    Returns
    -------
    Constraint
        Constraint instance accepting ``lower_bound <= value < upper_bound``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import constraints
    >>> constraints.half_open_interval(0.0, 1.0).check(lucid.tensor([0.0, 0.5, 1.0]))
    tensor([True, True, False], dtype=lucid.bool)
    """
    return _Interval(lower_bound, upper_bound, upper_open=True)


def integer_interval(lower_bound: int | Tensor, upper_bound: int | Tensor) -> Constraint:
    """Construct an inclusive integer-interval constraint.

    Parameters
    ----------
    lower_bound, upper_bound : int or Tensor
        Inclusive integer bounds defining the discrete support
        ``{lower_bound, ..., upper_bound}``; tensors are compared element by
        element.

    Returns
    -------
    Constraint
        Constraint instance accepting whole numbers in the inclusive range.
    """
    return _IntegerInterval(lower_bound, upper_bound)


def multinomial(upper_bound: float | Tensor) -> Constraint:
    """Construct the support of a multinomial of ``upper_bound`` draws.

    Parameters
    ----------
    upper_bound : float or Tensor
        The number of draws; a tensor is compared batch member by batch
        member.

    Returns
    -------
    Constraint
        Constraint accepting vectors, along the last dimension, whose entries
        are non-negative and sum to at most ``upper_bound``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import constraints
    >>> c = constraints.multinomial(5)
    >>> c.check(lucid.tensor([[1.0, 2.0, 2.0], [3.0, 3.0, 0.0]]))
    tensor([True, False], dtype=lucid.bool)
    """
    return _Multinomial(upper_bound)


def independent(base_constraint: Constraint, reinterpreted_batch_ndims: int) -> Constraint:
    """Reinterpret batch axes of a constraint as event axes.

    The returned constraint answers once per event: ``True`` when every
    element of the event's trailing ``reinterpreted_batch_ndims`` axes
    satisfies ``base_constraint``.  Its ``event_dim`` is the base's plus
    ``reinterpreted_batch_ndims``, and it is discrete when the base is.

    Parameters
    ----------
    base_constraint : Constraint
        The element-wise constraint to wrap.
    reinterpreted_batch_ndims : int
        Number of trailing axes folded into each event.  Must be ``>= 0``.

    Returns
    -------
    Constraint
        The wrapped constraint.

    Raises
    ------
    ValueError
        If ``reinterpreted_batch_ndims`` is negative.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import constraints
    >>> c = constraints.independent(constraints.positive, 1)
    >>> c.event_dim
    1
    >>> c.check(lucid.tensor([[1.0, 2.0], [1.0, -2.0]]))
    tensor([True, False], dtype=lucid.bool)
    """
    return _IndependentConstraint(base_constraint, reinterpreted_batch_ndims)


@final
class _Cat(Constraint):
    """Apply a sequence of constraints to slices along ``dim`` of sizes ``lengths``."""

    def __init__(
        self, cseq: list[Constraint], dim: int = 0, lengths: list[int] | None = None
    ) -> None:
        """Store the per-slice constraints, concatenation axis, and slice lengths."""
        self.cseq = list(cseq)
        self.lengths = [1] * len(self.cseq) if lengths is None else list(lengths)
        if len(self.lengths) != len(self.cseq):
            raise ValueError(
                f"cat: lengths ({len(self.lengths)}) must match cseq ({len(self.cseq)})"
            )
        self.dim = dim
        self.is_discrete = any(c.is_discrete for c in self.cseq)
        self.event_dim = max((c.event_dim for c in self.cseq), default=0)

    @override
    def check(self, value: Tensor) -> Tensor:
        """Return ``True`` where each slice along ``dim`` satisfies its constraint."""
        if not (-value.ndim <= self.dim < value.ndim):
            raise ValueError(
                f"cat: dim {self.dim} out of range for value with {value.ndim} dimensions"
            )
        checks: list[Tensor] = []
        start = 0
        for constr, length in zip(self.cseq, self.lengths, strict=True):
            v = value.narrow(self.dim, start, length)
            checks.append(constr.check(v))
            start = start + length
        return lucid.cat(checks, dim=self.dim)


@final
class _Stack(Constraint):
    """Apply a sequence of constraints to indexed slices along ``dim``."""

    def __init__(self, cseq: list[Constraint], dim: int = 0) -> None:
        """Store the per-slice constraints and stacking dimension."""
        self.cseq = list(cseq)
        self.dim = dim
        self.is_discrete = any(c.is_discrete for c in self.cseq)
        event_dim = max((c.event_dim for c in self.cseq), default=0)
        if self.dim + event_dim < 0:
            event_dim += 1
        self.event_dim = event_dim

    @override
    def check(self, value: Tensor) -> Tensor:
        """Return ``True`` where each slice along ``dim`` satisfies its constraint."""
        if not (-value.ndim <= self.dim < value.ndim):
            raise ValueError(
                f"stack: dim {self.dim} out of range for value with {value.ndim} dimensions"
            )
        vs = value.unbind(self.dim)
        if len(vs) != len(self.cseq):
            raise ValueError(
                f"stack: got {len(vs)} slices along dim {self.dim} but "
                f"{len(self.cseq)} constraints"
            )
        return lucid.stack(
            [constr.check(v) for v, constr in zip(vs, self.cseq, strict=True)], dim=self.dim
        )


def cat(cseq: list[Constraint], dim: int = 0, lengths: list[int] | None = None) -> Constraint:
    """Apply a sequence of constraints to contiguous slices along ``dim``.

    Parameters
    ----------
    cseq : list[Constraint]
        One constraint per slice along ``dim``.
    dim : int, optional
        Dimension along which the slices are concatenated.  Default ``0``.
    lengths : list[int] or None, optional
        Length of each slice along ``dim``.  Defaults to ``[1, ..., 1]``.

    Returns
    -------
    Constraint
        Composite constraint checking each slice against the corresponding
        entry of ``cseq``.
    """
    return _Cat(cseq, dim=dim, lengths=lengths)


def stack(cseq: list[Constraint], dim: int = 0) -> Constraint:
    """Apply a sequence of constraints to indexed slices along ``dim``.

    Parameters
    ----------
    cseq : list[Constraint]
        One constraint per index along ``dim``.
    dim : int, optional
        Dimension along which the slices are stacked.  Default ``0``.

    Returns
    -------
    Constraint
        Composite constraint checking each slice against the corresponding
        entry of ``cseq``.
    """
    return _Stack(cseq, dim=dim)
