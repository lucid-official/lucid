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

from typing import final, override

import lucid
from lucid._dtype import finfo
from lucid._tensor.tensor import Tensor


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
        if self.lower_open:
            above = value > self.lower_bound
        else:
            above = value >= self.lower_bound
        if self.upper_open:
            below = value < self.upper_bound
        else:
            below = value <= self.upper_bound
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


@final
class _LowerCholesky(Constraint):
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

    event_dim = 2

    @override
    def check(self, value: Tensor) -> Tensor:
        """Return ``True`` per matrix where it is lower-triangular with a
        positive diagonal."""
        lower = _all_trailing(lucid.tril(value) == value, 2)
        positive_diagonal = _all_trailing(value.diagonal(dim1=-2, dim2=-1) > 0, 1)
        return lower & positive_diagonal


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


@final
class _PositiveDefinite(Constraint):
    """Symmetric positive-definite matrices.

    Symmetry is checked first, within ``1e-6``: a Cholesky factorisation
    reads only the lower triangle, so on its own it accepts any matrix
    whose lower triangle happens to be the lower triangle of an SPD one —
    a Cholesky *factor* among them.
    """

    event_dim = 2

    @override
    def check(self, value: Tensor) -> Tensor:
        """Return ``True`` per matrix where it is symmetric and has a
        Cholesky factorisation."""
        if value.shape[-1] != value.shape[-2]:
            return lucid.zeros(tuple(value.shape[:-2]), dtype=lucid.bool, device=value.device)
        symmetric = _all_trailing(lucid.isclose(value, value.mT, atol=1e-6), 2)
        _, info = lucid.linalg.cholesky_ex(value)
        return symmetric & (info == 0)


@final
class _CorrCholesky(Constraint):
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

    event_dim = 2

    @override
    def check(self, value: Tensor) -> Tensor:
        """Return ``True`` per matrix where it is the Cholesky factor of a
        correlation matrix."""
        tol = float(finfo(value.dtype).eps) * value.shape[-1] * 10
        lower = _all_trailing(lucid.tril(value) == value, 2)
        positive_diagonal = _all_trailing(value.diagonal(dim1=-2, dim2=-1) > 0, 1)
        row_norm = lucid.linalg.vector_norm(value.detach(), dim=-1)
        unit_rows = _all_trailing((row_norm - 1.0).abs() <= tol, 1)
        return lower & positive_diagonal & unit_rows


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
nonnegative_integer = _IntegerGreaterThan(0)
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
        self,
        cseq: list[Constraint],
        dim: int = 0,
        lengths: list[int] | None = None,
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
            [constr.check(v) for v, constr in zip(vs, self.cseq, strict=True)],
            dim=self.dim,
        )


def cat(
    cseq: list[Constraint],
    dim: int = 0,
    lengths: list[int] | None = None,
) -> Constraint:
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
