"""Constraint-to-Transform registry (``ConstraintRegistry``, ``biject_to``, ``transform_to``)
and composite transforms (``SoftplusTransform``, ``PositiveDefiniteTransform``,
``StackTransform``, ``CatTransform``).
"""

import math
from collections.abc import Callable
from typing import override

import lucid
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.distributions import constraints
from lucid.distributions._util import _sum_rightmost
from lucid.distributions.constraints import (
    Constraint,
    _Cat,
    _CorrCholesky,
    _GreaterThan,
    _GreaterThanEq,
    _IndependentConstraint,
    _Interval,
    _LessThan,
    _LowerCholesky,
    _PositiveDefinite,
    _Real,
    _Simplex,
    _Stack,
)
from lucid.distributions.transformed_distribution import (
    ComposeTransform,
    Transform,
)
from lucid.distributions.transforms import (
    AffineTransform,
    CorrCholeskyTransform,
    ExpTransform,
    IndependentTransform,
    LowerCholeskyTransform,
    SigmoidTransform,
    SoftmaxTransform,
    StickBreakingTransform,
)


class SoftplusTransform(Transform):
    r"""Element-wise softplus bijection :math:`y = \log(1 + e^x)`.

    Maps :math:`\mathbb{R} \to (0, \infty)` element-wise with linear growth
    for large positive inputs, making it more numerically stable than
    :class:`ExpTransform`.  Monotone increasing, bijective, ``event_dim = 0``.

    Notes
    -----
    Forward: :math:`y = \operatorname{softplus}(x) = \log(1 + e^x)`.

    Inverse: :math:`x = \log(e^y - 1) = y + \log(1 - e^{-y})`.

    Log Jacobian determinant:

    .. math::

        \log\!\left|\frac{\partial y}{\partial x}\right|
        = \log \sigma(x) = -\operatorname{softplus}(-x)
    """

    domain: Constraint = constraints.real
    codomain: Constraint = constraints.positive
    bijective: bool = True
    sign: int = 1

    @override
    def _call(self, x: Tensor) -> Tensor:
        """Forward: :math:`y = \\operatorname{softplus}(x)`."""
        return F.softplus(x)

    @override
    def _inverse(self, y: Tensor) -> Tensor:
        """Inverse: :math:`x = y + \\log(-\\operatorname{expm1}(-y))`."""
        return lucid.log(-lucid.expm1(-y)) + y

    @override
    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        r"""Log-Jacobian :math:`-\operatorname{softplus}(-x)`."""
        return -F.softplus(-x)


class PositiveDefiniteTransform(Transform):
    r"""Bijection from unconstrained matrices to symmetric positive-definite matrices.

    Applies :class:`LowerCholeskyTransform` to obtain a lower-triangular
    Cholesky factor :math:`L` with positive diagonal and returns
    :math:`Y = L L^\top`.  ``event_dim = 2``.
    """

    domain: Constraint = constraints.independent(constraints.real, 2)
    codomain: Constraint = constraints.positive_definite
    event_dim: int = 2

    @override
    def _call(self, x: Tensor) -> Tensor:
        """Forward: :math:`Y = L L^\\top` where :math:`L = \\text{LowerCholeskyTransform}(x)`."""
        x_tril = LowerCholeskyTransform()(x)
        return x_tril @ x_tril.mT

    @override
    def _inverse(self, y: Tensor) -> Tensor:
        """Inverse: Cholesky-factorise ``y`` and invert :class:`LowerCholeskyTransform`."""
        y_tril = lucid.linalg.cholesky(y)
        return LowerCholeskyTransform().inv(y_tril)

    @override
    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        r"""Log-Jacobian combining :class:`LowerCholeskyTransform` and :math:`L \mapsto L L^\top`."""
        d = int(x.shape[-1])
        x_tril = LowerCholeskyTransform()(x)
        diag = x_tril.diagonal(dim1=-2, dim2=-1)
        order = lucid.arange(d, 0, -1, dtype=x.dtype, device=x.device)
        cholesky_ladj = LowerCholeskyTransform().log_abs_det_jacobian(x, x_tril)
        return cholesky_ladj + (order * diag.log()).sum(dim=-1) + d * math.log(2.0)


class StackTransform(Transform):
    r"""Apply a list of transforms to indexed slices along a stack axis.

    The :math:`k`-th transform is applied to the :math:`k`-th slice
    obtained by :meth:`~Tensor.unbind` along ``dim``, and the results
    are stacked back along the same axis.  This is the heterogeneous
    counterpart of :class:`IndependentTransform`: different bijections
    can be applied to different coordinates of an event vector.

    Parameters
    ----------
    transforms : list[Transform]
        One transform per slice.  Must be non-empty and the same length
        as ``x.shape[dim]``.
    dim : int, optional
        Dimension along which to slice.  Default ``0``.

    Raises
    ------
    ValueError
        If ``transforms`` is empty, or if the size of ``dim`` does not
        match ``len(transforms)`` at call time.
    """

    def __init__(self, transforms: list[Transform], dim: int = 0) -> None:
        """Store the per-slice transforms and the stacking axis ``dim``."""
        super().__init__()
        if not transforms:
            raise ValueError("StackTransform: transforms cannot be empty.")
        self.transforms = list(transforms)
        self.dim = dim
        self.event_dim = max(t.event_dim for t in transforms)

    @override
    @property
    def domain(self) -> Constraint:
        """Stacked domain constraint across all slice transforms."""
        return constraints.stack([t.domain for t in self.transforms], self.dim)

    @override
    @property
    def codomain(self) -> Constraint:
        """Stacked codomain constraint across all slice transforms."""
        return constraints.stack([t.codomain for t in self.transforms], self.dim)

    def _slices(self, x: Tensor) -> list[Tensor]:
        """Unbind ``x`` along ``dim``: one slice per transform, or ``ValueError``."""
        if len(slices := x.unbind(self.dim)) != len(self.transforms):
            raise ValueError(
                f"StackTransform: got {len(slices)} slices but "
                f"{len(self.transforms)} transforms."
            )
        return slices

    @override
    def _call(self, x: Tensor) -> Tensor:
        """Unbind ``x`` along ``dim`` and apply each transform to the matching slice."""
        slices = self._slices(x)
        return lucid.stack(
            [t(s) for t, s in zip(self.transforms, slices, strict=True)], dim=self.dim
        )

    @override
    def _inverse(self, y: Tensor) -> Tensor:
        """Unbind ``y`` along ``dim`` and apply each transform's inverse per slice."""
        slices = self._slices(y)
        return lucid.stack(
            [t._inverse(s) for t, s in zip(self.transforms, slices, strict=True)],
            dim=self.dim,
        )

    @override
    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        """Per-slice Jacobians stacked back along ``dim``."""
        xs, ys = self._slices(x), self._slices(y)
        ladjs = [
            t.log_abs_det_jacobian(xi, yi)
            for t, xi, yi in zip(self.transforms, xs, ys, strict=True)
        ]
        return lucid.stack(ladjs, dim=self.dim)


class CatTransform(Transform):
    r"""Apply different transforms to contiguous partitions along an axis.

    Splits the input along ``dim`` into contiguous chunks of size
    ``lengths[i]`` (one element each if ``lengths`` is ``None``),
    applies the :math:`i`-th transform to the :math:`i`-th chunk, and
    concatenates the results back along the same axis.

    Parameters
    ----------
    transforms : list[Transform]
        One transform per partition.
    dim : int, optional
        Concatenation dimension.  Default ``0``.
    lengths : list[int], optional
        Length of each partition.  Default ``None``: one per transform.
    """

    def __init__(
        self,
        transforms: list[Transform],
        dim: int = 0,
        lengths: list[int] | None = None,
    ) -> None:
        """Store the per-partition transforms, concat axis, and partition lengths."""
        super().__init__()
        if not transforms:
            raise ValueError("CatTransform: transforms cannot be empty.")
        if lengths is not None and len(lengths) != len(transforms):
            raise ValueError(
                f"CatTransform: got {len(lengths)} lengths but {len(transforms)} transforms."
            )
        self.transforms = list(transforms)
        self.dim = dim
        self.lengths = [1] * len(transforms) if lengths is None else list(lengths)
        self.event_dim = max(t.event_dim for t in transforms)

    @override
    @property
    def domain(self) -> Constraint:
        """Concatenated domain constraint across all partition transforms."""
        return constraints.cat([t.domain for t in self.transforms], self.dim, self.lengths)

    @override
    @property
    def codomain(self) -> Constraint:
        """Concatenated codomain constraint across all partition transforms."""
        return constraints.cat([t.codomain for t in self.transforms], self.dim, self.lengths)

    def _split(self, x: Tensor) -> list[Tensor]:
        """Split ``x`` along ``self.dim`` into pieces of ``self.lengths``."""
        return list(x.split(self.lengths, dim=self.dim))

    @override
    def _call(self, x: Tensor) -> Tensor:
        """Split ``x`` along ``dim``, apply each transform, and concatenate."""
        return lucid.cat(
            [t(p) for t, p in zip(self.transforms, self._split(x), strict=True)],
            dim=self.dim,
        )

    @override
    def _inverse(self, y: Tensor) -> Tensor:
        """Split ``y`` along ``dim``, apply each inverse, and concatenate."""
        return lucid.cat(
            [t._inverse(p) for t, p in zip(self.transforms, self._split(y), strict=True)],
            dim=self.dim,
        )

    @override
    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        """Per-partition Jacobians concatenated or summed according to ``event_dim``."""
        xs, ys = self._split(x), self._split(y)
        ladjs = [
            _sum_rightmost(t.log_abs_det_jacobian(xi, yi), self.event_dim - t.event_dim)
            for t, xi, yi in zip(self.transforms, xs, ys, strict=True)
        ]
        dim = (self.dim - x.ndim if self.dim >= 0 else self.dim) + self.event_dim
        return lucid.cat(ladjs, dim=dim) if dim < 0 else sum(ladjs[1:], ladjs[0])


class _IdentityTransform(Transform):
    """Identity bijection on :data:`constraints.real`."""

    domain: Constraint = constraints.real
    codomain: Constraint = constraints.real
    bijective: bool = True
    sign: int = 1

    @override
    def _call(self, x: Tensor) -> Tensor:
        return x

    @override
    def _inverse(self, y: Tensor) -> Tensor:
        return y

    @override
    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return lucid.zeros(tuple(x.shape), dtype=x.dtype, device=x.device)


def _is_scalar_number(val: object, target: float) -> bool:
    """Return ``True`` when ``val`` is a Python ``int`` or ``float`` equal to ``target``."""
    return isinstance(val, (int, float)) and not isinstance(val, bool) and val == target


class ConstraintRegistry:
    """Registry linking :class:`Constraint` instances or classes to :class:`Transform` factories.

    Instances of this registry back :data:`biject_to` and :data:`transform_to`:
    calling ``registry(constraint)`` dispatches on ``constraint`` (or its
    class) and returns a :class:`Transform` whose codomain satisfies
    ``constraint``.
    """

    def __init__(self) -> None:
        """Initialise an empty constraint-to-transform registry."""
        self._registry: dict[Constraint | type[Constraint], Callable[[Constraint], Transform]] = {}

    def register(
        self,
        constraint: Constraint | type[Constraint],
        factory: Callable[[Constraint], Transform] | None = None,
    ) -> Callable[..., object]:
        """Register a :class:`Transform` factory for a :class:`Constraint` or subclass."""
        if factory is None:

            def decorator(
                fn: Callable[[Constraint], Transform],
            ) -> Callable[[Constraint], Transform]:
                self.register(constraint, fn)
                return fn

            return decorator

        if isinstance(constraint, Constraint):
            self._registry[type(constraint)] = factory
        self._registry[constraint] = factory
        return factory

    def __call__(self, constraint: Constraint) -> Transform:
        """Look up and construct a :class:`Transform` targeting ``constraint``."""
        try:
            factory = self._registry[constraint]
        except KeyError:
            try:
                factory = self._registry[type(constraint)]
            except KeyError as exc:
                raise NotImplementedError(
                    f"Cannot transform {type(constraint).__name__} constraints"
                ) from exc
        return factory(constraint)


biject_to = ConstraintRegistry()
transform_to = ConstraintRegistry()
_IDENTITY_TRANSFORM = _IdentityTransform()


def _register(
    *targets: ConstraintRegistry | Constraint | type[Constraint],
) -> Callable[[Callable[..., Transform]], Callable[..., Transform]]:
    """Register a transform factory on every passed :class:`ConstraintRegistry` and constraint."""
    registries = [t for t in targets if isinstance(t, ConstraintRegistry)]
    constraints_list = [t for t in targets if not isinstance(t, ConstraintRegistry)]

    def decorator(fn: Callable[..., Transform]) -> Callable[..., Transform]:
        for reg in registries:
            for c in constraints_list:
                reg.register(c, fn)
        return fn

    return decorator


@_register(biject_to, transform_to, constraints.real)
def _transform_to_real(_constraint: _Real) -> Transform:
    return _IDENTITY_TRANSFORM


@_register(biject_to, transform_to, _GreaterThan, _GreaterThanEq)
def _transform_to_greater_than(
    constraint: _GreaterThan | _GreaterThanEq,
) -> Transform:
    if _is_scalar_number(constraint.lower_bound, 0.0):
        return ExpTransform()
    return ComposeTransform([ExpTransform(), AffineTransform(constraint.lower_bound, 1.0)])


@_register(biject_to, transform_to, _LessThan)
def _transform_to_less_than(constraint: _LessThan) -> Transform:
    return ComposeTransform([ExpTransform(), AffineTransform(constraint.upper_bound, -1.0)])


@_register(biject_to, transform_to, _Interval)
def _transform_to_interval(constraint: _Interval) -> Transform:
    if _is_scalar_number(constraint.lower_bound, 0.0) and _is_scalar_number(
        constraint.upper_bound, 1.0
    ):
        return SigmoidTransform()
    loc = constraint.lower_bound
    scale = constraint.upper_bound - constraint.lower_bound
    return ComposeTransform([SigmoidTransform(), AffineTransform(loc, scale)])


@_register(biject_to, _Simplex)
def _biject_to_simplex(_constraint: _Simplex) -> Transform:
    return StickBreakingTransform()


@_register(transform_to, _Simplex)
def _transform_to_simplex(_constraint: _Simplex) -> Transform:
    return SoftmaxTransform()


@_register(biject_to, transform_to, _LowerCholesky)
def _transform_to_lower_cholesky(_constraint: _LowerCholesky) -> Transform:
    return LowerCholeskyTransform()


@_register(biject_to, transform_to, _PositiveDefinite)
def _transform_to_positive_definite(_constraint: _PositiveDefinite) -> Transform:
    return PositiveDefiniteTransform()


@_register(biject_to, transform_to, _CorrCholesky)
def _transform_to_corr_cholesky(_constraint: _CorrCholesky) -> Transform:
    return CorrCholeskyTransform()


@_register(biject_to, _IndependentConstraint)
def _biject_to_independent(constraint: _IndependentConstraint) -> Transform:
    return IndependentTransform(
        biject_to(constraint.base_constraint),
        constraint.reinterpreted_batch_ndims,
    )


@_register(transform_to, _IndependentConstraint)
def _transform_to_independent(constraint: _IndependentConstraint) -> Transform:
    return IndependentTransform(
        transform_to(constraint.base_constraint),
        constraint.reinterpreted_batch_ndims,
    )


@_register(biject_to, _Cat)
def _biject_to_cat(constraint: _Cat) -> Transform:
    return CatTransform(
        [biject_to(c) for c in constraint.cseq],
        constraint.dim,
        constraint.lengths,
    )


@_register(transform_to, _Cat)
def _transform_to_cat(constraint: _Cat) -> Transform:
    return CatTransform(
        [transform_to(c) for c in constraint.cseq],
        constraint.dim,
        constraint.lengths,
    )


@_register(biject_to, _Stack)
def _biject_to_stack(constraint: _Stack) -> Transform:
    return StackTransform([biject_to(c) for c in constraint.cseq], constraint.dim)


@_register(transform_to, _Stack)
def _transform_to_stack(constraint: _Stack) -> Transform:
    return StackTransform([transform_to(c) for c in constraint.cseq], constraint.dim)
