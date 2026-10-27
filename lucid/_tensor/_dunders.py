"""
Dunder methods injected into the Tensor class.

All arithmetic/comparison operators are implemented here and attached to
the Tensor class by _inject_dunders() at module import time.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING
from lucid._C import engine as _C_engine
from lucid import _promotion
from lucid._dispatch import (
    _binary_operands,
    _refuse_bool_subtraction,
    _scalar_power,
    _unwrap_or_scalar,
    _wrap,
)
from lucid._tensor._indexing import _adopt_inplace, _getitem, _setitem

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor
    from lucid._types import TensorOrScalar, _IndexType


def _compare(
    op: Callable[[_C_engine.TensorImpl, _C_engine.TensorImpl], _C_engine.TensorImpl],
    tensor: Tensor,
    other: TensorOrScalar,
) -> Tensor:
    """Compare at the operands' common dtype, as arithmetic does.

    The engine compares only equal dtypes, and the comparison operators
    handed it the operands as they came — so ``bool_tensor > 0`` and
    ``int_tensor > 0.5`` raised ``DtypeMismatch`` while ``bool_tensor + 1``
    and the functional ``lucid.greater`` promoted and answered.  A Python
    number is rounded to the common dtype first, as the reference rounds
    it: ``float16_tensor == 0.1`` holds where the tensor holds ``0.1``.
    """
    a, b = _binary_operands(tensor, other)
    return _wrap(op(a, b))


def __add__(self: Tensor, other: TensorOrScalar) -> Tensor:
    r"""Element-wise addition: ``self + other`` with broadcasting and dtype promotion.

    Forwards to the engine ``add`` op.  The right operand may be another
    Tensor (broadcast-compatible) or a Python scalar, which is weak: it
    can lift the result's kind (``int32 + 1.5`` is float) but never its
    width (``float16 + 1.5`` stays float16).

    Parameters
    ----------
    other : Tensor or scalar
        Right-hand operand.

    Returns
    -------
    Tensor
        Sum with shape ``broadcast(self.shape, other.shape)`` and dtype
        ``promote_types(self.dtype, other.dtype)``.

    Notes
    -----
    Math:
    .. math::
        \text{out}_i = \text{self}_i + \text{other}_i

    Mixed-dtype operands are promoted to their common type before the
    add.  Both operands receive gradient flow with unit Jacobian
    (:math:`\partial \text{out} / \partial \text{self} = 1`,
    :math:`\partial \text{out} / \partial \text{other} = 1`).

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([1.0, 2.0, 3.0])
    >>> b = lucid.tensor([10.0, 20.0, 30.0])
    >>> a + b
    tensor([11., 22., 33.])
    >>> a + 5
    tensor([6., 7., 8.])
    """
    a, b = _binary_operands(self, other)
    return _wrap(_C_engine.add(a, b))


def __radd__(self: Tensor, other: TensorOrScalar) -> Tensor:
    r"""Reflected addition: ``other + self``.

    Triggered when the left operand does not implement ``__add__`` with
    a Tensor right operand (typically when ``other`` is a plain Python
    scalar).  Addition is commutative so the result equals
    :meth:`__add__`.

    Parameters
    ----------
    other : Tensor or scalar
        Left-hand operand of the original expression.

    Returns
    -------
    Tensor
        Sum with broadcast shape and promoted dtype.

    Notes
    -----
    Math:
    .. math::
        \text{out}_i = \text{other}_i + \text{self}_i

    Order is preserved for symmetry with :meth:`__rsub__` even though
    addition commutes.  Dtype promotion and broadcasting follow the
    same rules as :meth:`__add__`.

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([1.0, 2.0, 3.0])
    >>> 5 + a
    tensor([6., 7., 8.])
    """
    a, b = _binary_operands(other, self)
    return _wrap(_C_engine.add(a, b))


def __iadd__(self: Tensor, other: TensorOrScalar) -> Tensor:
    r"""In-place addition: ``self += other``.

    Mutates ``self._impl`` to hold the sum and returns ``self`` so the
    expression can be chained.  Useful for memory-tight loops; for
    leaves that participate in autograd prefer the out-of-place
    :meth:`__add__`.

    Parameters
    ----------
    other : Tensor or scalar
        Right-hand operand.

    Returns
    -------
    Tensor
        ``self`` after mutation.

    Notes
    -----
    Math (in place):
    .. math::
        \text{self}_i \leftarrow \text{self}_i + \text{other}_i

    Performing an in-place add on a leaf tensor with
    ``requires_grad=True`` is typically rejected by autograd because it
    would destroy the value needed for the backward pass.  Use the
    non-in-place form there.

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([1.0, 2.0, 3.0])
    >>> a += 10
    >>> a
    tensor([11., 12., 13.])
    """
    a, b = _binary_operands(self, other, inplace=True)
    return _adopt_inplace(self, _C_engine.add_(a, b), "+=")


def __sub__(self: Tensor, other: TensorOrScalar) -> Tensor:
    r"""Element-wise subtraction: ``self - other`` with broadcasting.

    Parameters
    ----------
    other : Tensor or scalar
        Right-hand operand.

    Returns
    -------
    Tensor
        Difference with broadcast shape and promoted dtype.

    Notes
    -----
    Math:
    .. math::
        \text{out}_i = \text{self}_i - \text{other}_i

    Dtype promotion follows the standard kind/width rules.  Gradients
    flow with Jacobian ``+1`` for ``self`` and ``-1`` for ``other``.

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([10.0, 20.0, 30.0])
    >>> b = lucid.tensor([1.0, 2.0, 3.0])
    >>> a - b
    tensor([9., 18., 27.])
    >>> a - 1
    tensor([9., 19., 29.])
    """
    _refuse_bool_subtraction(self, other)
    a, b = _binary_operands(self, other)
    return _wrap(_C_engine.sub(a, b))


def __rsub__(self: Tensor, other: TensorOrScalar) -> Tensor:
    r"""Reflected subtraction: ``other - self``.

    Triggered when the left operand does not implement ``__sub__`` with
    a Tensor right operand.  Subtraction is **not** commutative, so the
    order of operands matters.

    Parameters
    ----------
    other : Tensor or scalar
        Left-hand operand of the original expression.

    Returns
    -------
    Tensor
        Difference with broadcast shape and promoted dtype.

    Notes
    -----
    Math:
    .. math::
        \text{out}_i = \text{other}_i - \text{self}_i

    Jacobians are ``+1`` for ``other`` and ``-1`` for ``self``.

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([1.0, 2.0, 3.0])
    >>> 10 - a
    tensor([9., 8., 7.])
    """
    _refuse_bool_subtraction(other, self)
    a, b = _binary_operands(other, self)
    return _wrap(_C_engine.sub(a, b))


def __isub__(self: Tensor, other: TensorOrScalar) -> Tensor:
    r"""In-place subtraction: ``self -= other``.

    Mutates ``self._impl`` and returns ``self``.

    Parameters
    ----------
    other : Tensor or scalar
        Right-hand operand.

    Returns
    -------
    Tensor
        ``self`` after mutation.

    Notes
    -----
    Math (in place):
    .. math::
        \text{self}_i \leftarrow \text{self}_i - \text{other}_i

    In-place mutation of an autograd leaf with ``requires_grad=True``
    is typically rejected because it destroys the saved value needed
    for backward.

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([10.0, 20.0, 30.0])
    >>> a -= 5
    >>> a
    tensor([5., 15., 25.])
    """
    _refuse_bool_subtraction(self, other)
    a, b = _binary_operands(self, other, inplace=True)
    return _adopt_inplace(self, _C_engine.sub_(a, b), "-=")


def __mul__(self: Tensor, other: TensorOrScalar) -> Tensor:
    r"""Element-wise multiplication: ``self * other`` (Hadamard product).

    Parameters
    ----------
    other : Tensor or scalar
        Right-hand operand.

    Returns
    -------
    Tensor
        Product with broadcast shape and promoted dtype.

    Notes
    -----
    Math:
    .. math::
        \text{out}_i = \text{self}_i \cdot \text{other}_i

    This is the **element-wise** product; for matrix multiplication
    use :meth:`__matmul__` (``@`` operator).  Jacobians are
    :math:`\partial \text{out}_i / \partial \text{self}_i = \text{other}_i`
    and
    :math:`\partial \text{out}_i / \partial \text{other}_i = \text{self}_i`.

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([1.0, 2.0, 3.0])
    >>> b = lucid.tensor([4.0, 5.0, 6.0])
    >>> a * b
    tensor([4., 10., 18.])
    >>> a * 2
    tensor([2., 4., 6.])
    """
    a, b = _binary_operands(self, other, opmath=True)
    return _wrap(_C_engine.mul(a, b))


def __rmul__(self: Tensor, other: TensorOrScalar) -> Tensor:
    r"""Reflected multiplication: ``other * self``.

    Triggered when the left operand does not implement ``__mul__``
    with a Tensor right operand (e.g. plain scalar on the left).
    Multiplication is commutative, so the result equals
    :meth:`__mul__`.

    Parameters
    ----------
    other : Tensor or scalar
        Left-hand operand of the original expression.

    Returns
    -------
    Tensor
        Product with broadcast shape and promoted dtype.

    Notes
    -----
    Math:
    .. math::
        \text{out}_i = \text{other}_i \cdot \text{self}_i

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([1.0, 2.0, 3.0])
    >>> 3 * a
    tensor([3., 6., 9.])
    """
    a, b = _binary_operands(other, self, opmath=True)
    return _wrap(_C_engine.mul(a, b))


def __imul__(self: Tensor, other: TensorOrScalar) -> Tensor:
    r"""In-place multiplication: ``self *= other``.

    Mutates ``self._impl`` and returns ``self``.

    Parameters
    ----------
    other : Tensor or scalar
        Right-hand operand.

    Returns
    -------
    Tensor
        ``self`` after mutation.

    Notes
    -----
    Math (in place):
    .. math::
        \text{self}_i \leftarrow \text{self}_i \cdot \text{other}_i

    In-place multiplication on an autograd leaf with
    ``requires_grad=True`` is typically rejected.

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([1.0, 2.0, 3.0])
    >>> a *= 10
    >>> a
    tensor([10., 20., 30.])
    """
    a, b = _binary_operands(self, other, opmath=True, inplace=True)
    return _adopt_inplace(self, _C_engine.mul_(a, b), "*=")


def __truediv__(self: Tensor, other: TensorOrScalar) -> Tensor:
    r"""Element-wise true division: ``self / other``.

    Always produces a floating-point result.  Integer operands are
    promoted to float before the division.

    Parameters
    ----------
    other : Tensor or scalar
        Right-hand operand.

    Returns
    -------
    Tensor
        Quotient with broadcast shape and a float dtype.

    Notes
    -----
    Math:
    .. math::
        \text{out}_i = \frac{\text{self}_i}{\text{other}_i}

    Division by zero follows IEEE 754: positive / 0 -> ``+inf``,
    0 / 0 -> ``nan``.  Jacobians are
    :math:`\partial \text{out}_i / \partial \text{self}_i = 1 / \text{other}_i`
    and
    :math:`\partial \text{out}_i / \partial \text{other}_i = -\text{self}_i / \text{other}_i^2`.

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([10.0, 20.0, 30.0])
    >>> b = lucid.tensor([2.0, 4.0, 5.0])
    >>> a / b
    tensor([5., 5., 6.])
    >>> a / 10
    tensor([1., 2., 3.])
    >>> lucid.tensor([7, 8], dtype=lucid.int64) / 2
    tensor([3.5, 4.])
    """
    a, b = _binary_operands(self, other, true_div=True, opmath=True)
    return _wrap(_C_engine.div(a, b))


def __rtruediv__(self: Tensor, other: TensorOrScalar) -> Tensor:
    r"""Reflected true division: ``other / self``.

    Triggered when the left operand does not implement ``__truediv__``
    with a Tensor right operand.  Division is **not** commutative.

    Parameters
    ----------
    other : Tensor or scalar
        Left-hand operand of the original expression.

    Returns
    -------
    Tensor
        Quotient with broadcast shape and float dtype.

    Notes
    -----
    Math:
    .. math::
        \text{out}_i = \frac{\text{other}_i}{\text{self}_i}

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([2.0, 4.0, 5.0])
    >>> 20 / a
    tensor([10.,  5.,  4.])
    """
    a, b = _binary_operands(other, self, true_div=True, opmath=True)
    return _wrap(_C_engine.div(a, b))


def __itruediv__(self: Tensor, other: TensorOrScalar) -> Tensor:
    r"""In-place true division: ``self /= other``.

    Mutates ``self._impl`` and returns ``self``.

    Parameters
    ----------
    other : Tensor or scalar
        Right-hand operand.

    Returns
    -------
    Tensor
        ``self`` after mutation.

    Notes
    -----
    Math (in place):
    .. math::
        \text{self}_i \leftarrow \frac{\text{self}_i}{\text{other}_i}

    Requires ``self`` to already have a float dtype since the result
    cannot be cast back to integer in place.  Rejected on autograd
    leaves with ``requires_grad=True``.

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([10.0, 20.0, 30.0])
    >>> a /= 10
    >>> a
    tensor([1., 2., 3.])
    """
    a, b = _binary_operands(self, other, true_div=True, opmath=True, inplace=True)
    return _adopt_inplace(self, _C_engine.div_(a, b), "/=")


def __floordiv__(self: Tensor, other: TensorOrScalar) -> Tensor:
    r"""Element-wise floor division: ``self // other``.

    Computes the largest integer less than or equal to the true
    quotient, broadcast element-wise.

    Parameters
    ----------
    other : Tensor or scalar
        Right-hand operand.

    Returns
    -------
    Tensor
        Floor-divided result with broadcast shape and promoted dtype.

    Notes
    -----
    Math:
    .. math::
        \text{out}_i = \left\lfloor \frac{\text{self}_i}{\text{other}_i} \right\rfloor

    For integer operands the result is integer; for float operands
    the result is float but takes integral values; both are exact.  An
    integer zero divisor raises ``RuntimeError`` on the CPU and gives -1
    on Metal (no device sync, as on the reference's accelerator).  Not
    differentiable at integer-quotient boundaries.

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([7.0, 8.0, 9.0])
    >>> a // 2
    tensor([3., 4., 4.])
    """
    a, b = _binary_operands(self, other, opmath=True)
    return _wrap(_C_engine.floordiv(a, b))


def __rfloordiv__(self: Tensor, other: TensorOrScalar) -> Tensor:
    r"""Reflected floor division: ``other // self``.

    Triggered when the left operand does not implement
    ``__floordiv__`` with a Tensor right operand.

    Parameters
    ----------
    other : Tensor or scalar
        Left-hand operand of the original expression.

    Returns
    -------
    Tensor
        Floor-divided result with broadcast shape and promoted dtype.

    Notes
    -----
    Math:
    .. math::
        \text{out}_i = \left\lfloor \frac{\text{other}_i}{\text{self}_i} \right\rfloor

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([2.0, 3.0, 4.0])
    >>> 10 // a
    tensor([5., 3., 2.])
    """
    a, b = _binary_operands(other, self, opmath=True)
    return _wrap(_C_engine.floordiv(a, b))


def __pow__(self: Tensor, other: TensorOrScalar) -> Tensor:
    r"""Element-wise exponentiation: ``self ** other``.

    Raises ``self`` to the ``other`` power element-by-element with
    broadcasting and dtype promotion.

    Parameters
    ----------
    other : Tensor or scalar
        Exponent.

    Returns
    -------
    Tensor
        Powered result with broadcast shape and promoted dtype.

    Notes
    -----
    Math:
    .. math::
        \text{out}_i = \text{self}_i^{\text{other}_i}

    Gradients are
    :math:`\partial \text{out}_i / \partial \text{self}_i =
    \text{other}_i \cdot \text{self}_i^{\text{other}_i - 1}`
    and
    :math:`\partial \text{out}_i / \partial \text{other}_i =
    \text{self}_i^{\text{other}_i} \cdot \ln(\text{self}_i)`.
    Negative base with non-integer exponent produces NaN under IEEE
    754 unless complex dtype is used.

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([1.0, 2.0, 3.0])
    >>> a ** 2
    tensor([1., 4., 9.])
    >>> a ** lucid.tensor([3.0, 2.0, 1.0])
    tensor([1., 4., 3.])
    """
    scalar = _scalar_power(self._impl, other)
    if scalar is not None:
        return _wrap(scalar)
    a, b = _binary_operands(self, other)
    return _wrap(_C_engine.pow(a, b))


def __rpow__(self: Tensor, other: TensorOrScalar) -> Tensor:
    r"""Reflected exponentiation: ``other ** self``.

    Triggered when the left operand does not implement ``__pow__``
    with a Tensor right operand.  Exponentiation is **not**
    commutative.

    Parameters
    ----------
    other : Tensor or scalar
        Base of the original expression.

    Returns
    -------
    Tensor
        Powered result with broadcast shape and promoted dtype.

    Notes
    -----
    Math:
    .. math::
        \text{out}_i = \text{other}_i^{\text{self}_i}

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([1.0, 2.0, 3.0])
    >>> 2 ** a
    tensor([2., 4., 8.])
    """
    a, b = _binary_operands(other, self)
    return _wrap(_C_engine.pow(a, b))


def __ipow__(self: Tensor, other: TensorOrScalar) -> Tensor:
    r"""In-place exponentiation: ``self **= other``.

    Mutates ``self._impl`` and returns ``self``.

    Parameters
    ----------
    other : Tensor or scalar
        Exponent.

    Returns
    -------
    Tensor
        ``self`` after mutation.

    Notes
    -----
    Math (in place):
    .. math::
        \text{self}_i \leftarrow \text{self}_i^{\text{other}_i}

    Rejected on autograd leaves with ``requires_grad=True`` because
    the operation cannot be reversed.

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([1.0, 2.0, 3.0])
    >>> a **= 2
    >>> a
    tensor([1., 4., 9.])
    """
    _promotion.check_pow_scalar(self._impl.dtype, other)
    a, b = _binary_operands(self, other, inplace=True)
    return _adopt_inplace(self, _C_engine.pow_(a, b), "**=")


def __matmul__(self: Tensor, other: Tensor) -> Tensor:
    r"""Matrix multiplication: ``self @ other`` with batched semantics.

    Supports 2-D (matrix) and N-D (batched) inputs.  For 1-D vectors
    the usual NumPy/BLAS dot-product semantics apply (1-D @ 1-D is
    scalar, 1-D @ 2-D is a row vector, 2-D @ 1-D is a column vector).
    For N-D operands the trailing two dimensions are matrix-multiplied
    while leading dimensions broadcast.

    Parameters
    ----------
    other : Tensor
        Right-hand operand; inner dimension must equal ``self``'s last
        dimension.

    Returns
    -------
    Tensor
        Matrix product with shape determined by broadcasting batch
        dims and contracting the inner dimension.

    Notes
    -----
    Math (2-D case):
    .. math::
        \text{out}_{ij} = \sum_k \text{self}_{ik} \cdot \text{other}_{kj}

    For N-D inputs the same contraction is applied to the last two
    axes:

    .. math::
        \text{out}_{\dots ij} = \sum_k \text{self}_{\dots ik} \cdot \text{other}_{\dots kj}

    Backward passes both matmuls' transposes through the chain rule.
    Use :meth:`__mul__` (``*``) for element-wise multiplication.

    Examples
    --------
    >>> import lucid
    >>> A = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
    >>> B = lucid.tensor([[5.0, 6.0], [7.0, 8.0]])
    >>> A @ B
    tensor([[19., 22.], [43., 50.]])
    """
    return _wrap(_C_engine.matmul(self._impl, _unwrap_or_scalar(other, self._impl)))


def __rmatmul__(self: Tensor, other: Tensor) -> Tensor:
    r"""Reflected matrix multiplication: ``other @ self``.

    Triggered when the left operand does not implement ``__matmul__``
    with a Tensor right operand.  Matrix multiplication is **not**
    commutative.

    Parameters
    ----------
    other : Tensor
        Left-hand operand of the original expression.

    Returns
    -------
    Tensor
        Matrix product with broadcasted batch dims and contracted
        inner dimension.

    Notes
    -----
    Math (2-D case):
    .. math::
        \text{out}_{ij} = \sum_k \text{other}_{ik} \cdot \text{self}_{kj}

    Examples
    --------
    >>> import lucid
    >>> A = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
    >>> B = lucid.tensor([[5.0, 6.0], [7.0, 8.0]])
    >>> B.__rmatmul__(A)
    tensor([[19., 22.], [43., 50.]])
    """
    return _wrap(_C_engine.matmul(_unwrap_or_scalar(other, self._impl), self._impl))


def __neg__(self: Tensor) -> Tensor:
    r"""Unary negation: ``-self``.

    Returns
    -------
    Tensor
        Element-wise negation with the same shape and dtype as
        ``self``.

    Notes
    -----
    Math:
    .. math::
        \text{out}_i = -\text{self}_i

    Defined for all signed numeric dtypes.  Applied to an unsigned
    integer dtype the engine raises ``TypeError`` because the result
    would not be representable.  Jacobian is ``-1``.

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([1.0, -2.0, 3.0])
    >>> -a
    tensor([-1., 2., -3.])
    """
    return _wrap(_C_engine.neg(self._impl))


def __abs__(self: Tensor) -> Tensor:
    r"""Element-wise absolute value: ``abs(self)``.

    Returns
    -------
    Tensor
        Magnitude tensor with the same shape as ``self``.  For complex
        input the result has the corresponding real float dtype.

    Notes
    -----
    Math:
    .. math::
        \text{out}_i = \lvert \text{self}_i \rvert

    For complex input :math:`\lvert z \rvert = \sqrt{\Re(z)^2 + \Im(z)^2}`.
    The derivative is :math:`\operatorname{sign}(\text{self}_i)` and is
    undefined at zero (the engine returns ``0`` there).

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([-1.0, 2.0, -3.0])
    >>> abs(a)
    tensor([1., 2., 3.])
    """
    return _wrap(_C_engine.abs(self._impl))


def __invert__(self: Tensor) -> Tensor:
    r"""Bitwise / logical NOT: ``~self``.

    For boolean tensors this is logical negation; for integer dtypes
    it is the standard two's-complement bitwise complement.  Float
    inputs are rejected by the engine.

    Returns
    -------
    Tensor
        Inverted tensor with the same shape and dtype as ``self``.

    Notes
    -----
    Math (boolean case):
    .. math::
        \text{out}_i = \neg\,\text{self}_i

    Math (integer case):
    .. math::
        \text{out}_i = \sim\!\text{self}_i = -\text{self}_i - 1

    Not differentiable; produced tensor never carries autograd
    information.

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([True, False, True])
    >>> ~a
    tensor([False, True, False], dtype=lucid.bool)
    """
    return _wrap(_C_engine.invert(self._impl))


def __int__(self: Tensor) -> int:
    """Convert a single-element tensor to a Python ``int``.

    Returns
    -------
    int
        The tensor's only element, truncated toward zero if it is a
        float — matching what ``int()`` does to a Python float.

    Raises
    ------
    LucidError
        If the tensor holds more than one element.  Which element
        would have been meant is not something to guess at.

    Examples
    --------
    >>> import lucid
    >>> int(lucid.tensor(3))
    3
    >>> int(lucid.tensor(3.7))
    3
    """
    return int(self.item())


def __index__(self: Tensor) -> int:
    """Use a single-element integer tensor where an index is expected.

    This is what lets a tensor appear in ``range()``, in a slice, or
    as a list subscript.  Unlike :meth:`__int__` it refuses floats:
    silently truncating an index is how off-by-one bugs are made.

    Returns
    -------
    int
        The tensor's only element.

    Raises
    ------
    TypeError
        If the dtype is not integral.
    LucidError
        If the tensor holds more than one element.

    Examples
    --------
    >>> import lucid
    >>> [10, 20, 30][lucid.tensor(1)]
    20
    """
    value = self.item()
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"only integer tensors can be used as an index, " f"got dtype {self.dtype}")
    return value


def __float__(self: Tensor) -> float:
    """Convert a single-element tensor to a Python ``float``.

    The ordinary way to read a loss or a metric out of the graph —
    ``float(loss)`` — and the reason it is worth having as an
    operator rather than only as ``.item()``.

    Returns
    -------
    float
        The tensor's only element.

    Raises
    ------
    LucidError
        If the tensor holds more than one element.

    Examples
    --------
    >>> import lucid
    >>> float(lucid.tensor(2.5))
    2.5
    """
    return float(self.item())


def __complex__(self: Tensor) -> complex:
    """Convert a single-element tensor to a Python ``complex``.

    Returns
    -------
    complex
        The tensor's only element.  Real dtypes give a zero
        imaginary part, as Python's own ``complex()`` does.

    Raises
    ------
    LucidError
        If the tensor holds more than one element.

    Examples
    --------
    >>> import lucid
    >>> complex(lucid.tensor(2.0))
    (2+0j)
    """
    return complex(self.item())


def __mod__(self: Tensor, other: TensorOrScalar) -> Tensor:
    r"""Element-wise remainder: ``self % other``.

    Follows Python's and the reference framework's convention rather
    than C's: the result takes the **sign of the divisor**, so
    ``-7 % 3`` is ``2``.  Use :func:`lucid.fmod` for the C behaviour,
    where the sign follows the dividend.

    Parameters
    ----------
    other : Tensor or scalar
        Divisor; broadcast against ``self``.

    Returns
    -------
    Tensor
        Remainder with broadcast shape and promoted dtype.

    Notes
    -----
    Math:
    .. math::
        \text{out}_i = \text{self}_i
            - \text{other}_i \left\lfloor
                \frac{\text{self}_i}{\text{other}_i} \right\rfloor

    Examples
    --------
    >>> import lucid
    >>> lucid.tensor([-7.0, 7.0]) % 3
    tensor([2., 1.])
    """
    a, b = _binary_operands(self, other)
    return _wrap(_C_engine.remainder(a, b))


def __rmod__(self: Tensor, other: TensorOrScalar) -> Tensor:
    r"""Reflected remainder: ``other % self``.

    Parameters
    ----------
    other : Tensor or scalar
        Dividend of the original expression.

    Returns
    -------
    Tensor
        Remainder with broadcast shape and promoted dtype.

    Examples
    --------
    >>> import lucid
    >>> 7 % lucid.tensor([3.0, 4.0])
    tensor([1., 3.])
    """
    a, b = _binary_operands(other, self)
    return _wrap(_C_engine.remainder(a, b))


def __divmod__(self: Tensor, other: TensorOrScalar) -> tuple[Tensor, Tensor]:
    """Floor quotient and remainder together: ``divmod(self, other)``.

    Parameters
    ----------
    other : Tensor or scalar
        Divisor; broadcast against ``self``.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(self // other, self % other)``, the pair Python's builtin
        promises.

    Examples
    --------
    >>> import lucid
    >>> q, r = divmod(lucid.tensor([7.0]), 3)
    >>> q, r
    (tensor([2.]), tensor([1.]))
    """
    return self // other, self % other


def __rdivmod__(self: Tensor, other: TensorOrScalar) -> tuple[Tensor, Tensor]:
    """Reflected pair: ``divmod(other, self)``.

    Parameters
    ----------
    other : Tensor or scalar
        Dividend of the original expression.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(other // self, other % self)``.

    Examples
    --------
    >>> import lucid
    >>> divmod(7, lucid.tensor([3.0]))
    (tensor([2.]), tensor([1.]))
    """
    return other // self, other % self


def __rand__(self: Tensor, other: TensorOrScalar) -> Tensor:
    r"""Reflected bitwise / logical AND: ``other & self``.

    AND is commutative, so this exists to make the operator work when
    the left operand is a scalar or a type that does not know about
    tensors — not because the order changes the answer.

    Parameters
    ----------
    other : Tensor or scalar
        Left-hand operand of the original expression.

    Returns
    -------
    Tensor
        AND-combined tensor with broadcast shape.

    Examples
    --------
    >>> import lucid
    >>> True & lucid.tensor([True, False])
    tensor([True, False], dtype=lucid.bool)
    """
    a, b = _binary_operands(other, self)
    return _wrap(_C_engine.bitwise_and(a, b))


def __ror__(self: Tensor, other: TensorOrScalar) -> Tensor:
    r"""Reflected bitwise / logical OR: ``other | self``.

    Parameters
    ----------
    other : Tensor or scalar
        Left-hand operand of the original expression.

    Returns
    -------
    Tensor
        OR-combined tensor with broadcast shape.

    Examples
    --------
    >>> import lucid
    >>> False | lucid.tensor([True, False])
    tensor([True, False], dtype=lucid.bool)
    """
    a, b = _binary_operands(other, self)
    return _wrap(_C_engine.bitwise_or(a, b))


def __rxor__(self: Tensor, other: TensorOrScalar) -> Tensor:
    r"""Reflected bitwise / logical XOR: ``other ^ self``.

    Parameters
    ----------
    other : Tensor or scalar
        Left-hand operand of the original expression.

    Returns
    -------
    Tensor
        XOR-combined tensor with broadcast shape.

    Examples
    --------
    >>> import lucid
    >>> True ^ lucid.tensor([True, False])
    tensor([False, True], dtype=lucid.bool)
    """
    a, b = _binary_operands(other, self)
    return _wrap(_C_engine.bitwise_xor(a, b))


def __lshift__(self: Tensor, other: TensorOrScalar) -> Tensor:
    r"""Element-wise bitwise left shift: ``self << other``.

    Integer dtypes only; each element is shifted by the matching
    element of ``other``, which is a multiplication by
    :math:`2^{\text{other}}`.

    Parameters
    ----------
    other : Tensor or scalar
        Shift amount; broadcast against ``self``.

    Returns
    -------
    Tensor
        Shifted tensor with broadcast shape.  Not differentiable.

    Examples
    --------
    >>> import lucid
    >>> lucid.tensor([1, 2, 3]) << 2
    tensor([4, 8, 12], dtype=lucid.int64)
    """
    a, b = _binary_operands(self, other)
    return _wrap(_C_engine.bitwise_left_shift(a, b))


def __rlshift__(self: Tensor, other: TensorOrScalar) -> Tensor:
    r"""Reflected left shift: ``other << self``.

    Parameters
    ----------
    other : Tensor or scalar
        Value being shifted.

    Returns
    -------
    Tensor
        Shifted tensor with broadcast shape.

    Examples
    --------
    >>> import lucid
    >>> 1 << lucid.tensor([1, 2, 3])
    tensor([2, 4, 8], dtype=lucid.int64)
    """
    a, b = _binary_operands(other, self)
    return _wrap(_C_engine.bitwise_left_shift(a, b))


def __rshift__(self: Tensor, other: TensorOrScalar) -> Tensor:
    r"""Element-wise bitwise right shift: ``self >> other``.

    Integer dtypes only; a floor division by
    :math:`2^{\text{other}}`.

    Parameters
    ----------
    other : Tensor or scalar
        Shift amount; broadcast against ``self``.

    Returns
    -------
    Tensor
        Shifted tensor with broadcast shape.  Not differentiable.

    Examples
    --------
    >>> import lucid
    >>> lucid.tensor([4, 8, 12]) >> 2
    tensor([1, 2, 3], dtype=lucid.int64)
    """
    a, b = _binary_operands(self, other)
    return _wrap(_C_engine.bitwise_right_shift(a, b))


def __rrshift__(self: Tensor, other: TensorOrScalar) -> Tensor:
    r"""Reflected right shift: ``other >> self``.

    Parameters
    ----------
    other : Tensor or scalar
        Value being shifted.

    Returns
    -------
    Tensor
        Shifted tensor with broadcast shape.

    Examples
    --------
    >>> import lucid
    >>> 16 >> lucid.tensor([1, 2, 3])
    tensor([8, 4, 2], dtype=lucid.int64)
    """
    a, b = _binary_operands(other, self)
    return _wrap(_C_engine.bitwise_right_shift(a, b))


def __and__(self: Tensor, other: TensorOrScalar) -> Tensor:
    r"""Element-wise bitwise / logical AND: ``self & other``.

    For boolean tensors this is logical AND; for integer dtypes it is
    bitwise AND.  Float dtypes are rejected.

    Parameters
    ----------
    other : Tensor or scalar
        Right-hand operand, bool or integer; the pair meets at its
        common dtype first (``int32 & int64`` is int64).

    Returns
    -------
    Tensor
        AND-combined tensor with broadcast shape.

    Notes
    -----
    Math (boolean):
    .. math::
        \text{out}_i = \text{self}_i \wedge \text{other}_i

    Math (integer): bitwise AND across the two operands' binary
    representations.  Not differentiable.

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([True, True, False, False])
    >>> b = lucid.tensor([True, False, True, False])
    >>> a & b
    tensor([True, False, False, False], dtype=lucid.bool)
    """
    a, b = _binary_operands(self, other)
    return _wrap(_C_engine.bitwise_and(a, b))


def __or__(self: Tensor, other: TensorOrScalar) -> Tensor:
    r"""Element-wise bitwise / logical OR: ``self | other``.

    For boolean tensors this is logical OR; for integer dtypes it is
    bitwise OR.  Float dtypes are rejected.

    Parameters
    ----------
    other : Tensor or scalar
        Right-hand operand, bool or integer; the pair meets at its
        common dtype first.

    Returns
    -------
    Tensor
        OR-combined tensor with broadcast shape.

    Notes
    -----
    Math (boolean):
    .. math::
        \text{out}_i = \text{self}_i \vee \text{other}_i

    Not differentiable.

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([True, True, False, False])
    >>> b = lucid.tensor([True, False, True, False])
    >>> a | b
    tensor([True, True, True, False], dtype=lucid.bool)
    """
    a, b = _binary_operands(self, other)
    return _wrap(_C_engine.bitwise_or(a, b))


def __xor__(self: Tensor, other: TensorOrScalar) -> Tensor:
    r"""Element-wise bitwise / logical XOR: ``self ^ other``.

    For boolean tensors this is logical exclusive-or; for integer
    dtypes it is bitwise XOR.

    Parameters
    ----------
    other : Tensor or scalar
        Right-hand operand, bool or integer; the pair meets at its
        common dtype first.

    Returns
    -------
    Tensor
        XOR-combined tensor with broadcast shape.

    Notes
    -----
    Math (boolean):
    .. math::
        \text{out}_i = \text{self}_i \oplus \text{other}_i

    Not differentiable.

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([True, True, False, False])
    >>> b = lucid.tensor([True, False, True, False])
    >>> a ^ b
    tensor([False, True, True, False], dtype=lucid.bool)
    """
    a, b = _binary_operands(self, other)
    return _wrap(_C_engine.bitwise_xor(a, b))


def __eq__(self: Tensor, other: TensorOrScalar) -> Tensor:
    r"""Element-wise equality comparison: ``self == other``.

    Parameters
    ----------
    other : Tensor or scalar
        Right-hand operand; broadcast against ``self``.

    Returns
    -------
    Tensor
        Boolean tensor with broadcast shape; ``True`` where the
        corresponding elements match.

    Notes
    -----
    Math:
    .. math::
        \text{out}_i = \mathbb{1}\!\left[\text{self}_i = \text{other}_i\right]

    Comparison with NaN follows IEEE 754 — ``NaN == NaN`` is always
    ``False``.  Not differentiable; the output tensor never carries
    gradient.

    Returns an element-wise boolean Tensor when ``other`` is a Tensor
    or numeric scalar (broadcast against ``self``).  For arguments
    that are neither (e.g. ``None``, lists, arbitrary objects),
    Python's ``==`` falls back to object identity and yields a plain
    ``bool`` rather than a Tensor.

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([1.0, 2.0, 3.0])
    >>> b = lucid.tensor([1.0, 5.0, 3.0])
    >>> a == b
    tensor([True, False, True], dtype=lucid.bool)
    """
    if not _promotion.is_comparable(other):
        return NotImplemented
    return _compare(_C_engine.equal, self, other)


def __ne__(self: Tensor, other: TensorOrScalar) -> Tensor:
    r"""Element-wise inequality comparison: ``self != other``.

    Parameters
    ----------
    other : Tensor or scalar
        Right-hand operand; broadcast against ``self``.

    Returns
    -------
    Tensor
        Boolean tensor with broadcast shape.

    Notes
    -----
    Math:
    .. math::
        \text{out}_i = \mathbb{1}\!\left[\text{self}_i \neq \text{other}_i\right]

    Under IEEE 754, ``NaN != NaN`` evaluates to ``True`` — this is the
    only comparison operator that is "true" against NaN.  Not
    differentiable.

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([1.0, 2.0, 3.0])
    >>> b = lucid.tensor([1.0, 5.0, 3.0])
    >>> a != b
    tensor([False, True, False], dtype=lucid.bool)
    """
    if not _promotion.is_comparable(other):
        return NotImplemented
    return _compare(_C_engine.not_equal, self, other)


def __lt__(self: Tensor, other: TensorOrScalar) -> Tensor:
    r"""Element-wise less-than comparison: ``self < other``.

    Parameters
    ----------
    other : Tensor or scalar
        Right-hand operand; broadcast against ``self``.

    Returns
    -------
    Tensor
        Boolean tensor with broadcast shape.

    Notes
    -----
    Math:
    .. math::
        \text{out}_i = \mathbb{1}\!\left[\text{self}_i < \text{other}_i\right]

    Any comparison involving NaN returns ``False`` per IEEE 754.  Not
    differentiable.

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([1.0, 2.0, 3.0])
    >>> a < 2.5
    tensor([True, True, False], dtype=lucid.bool)
    """
    return _compare(_C_engine.less, self, other)


def __le__(self: Tensor, other: TensorOrScalar) -> Tensor:
    r"""Element-wise less-than-or-equal comparison: ``self <= other``.

    Parameters
    ----------
    other : Tensor or scalar
        Right-hand operand; broadcast against ``self``.

    Returns
    -------
    Tensor
        Boolean tensor with broadcast shape.

    Notes
    -----
    Math:
    .. math::
        \text{out}_i = \mathbb{1}\!\left[\text{self}_i \leq \text{other}_i\right]

    Any comparison involving NaN returns ``False`` per IEEE 754.  Not
    differentiable.

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([1.0, 2.0, 3.0])
    >>> a <= 2.0
    tensor([True, True, False], dtype=lucid.bool)
    """
    return _compare(_C_engine.less_equal, self, other)


def __gt__(self: Tensor, other: TensorOrScalar) -> Tensor:
    r"""Element-wise greater-than comparison: ``self > other``.

    Parameters
    ----------
    other : Tensor or scalar
        Right-hand operand; broadcast against ``self``.

    Returns
    -------
    Tensor
        Boolean tensor with broadcast shape.

    Notes
    -----
    Math:
    .. math::
        \text{out}_i = \mathbb{1}\!\left[\text{self}_i > \text{other}_i\right]

    Any comparison involving NaN returns ``False`` per IEEE 754.  Not
    differentiable.

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([1.0, 2.0, 3.0])
    >>> a > 1.5
    tensor([False, True, True], dtype=lucid.bool)
    """
    return _compare(_C_engine.greater, self, other)


def __ge__(self: Tensor, other: TensorOrScalar) -> Tensor:
    r"""Element-wise greater-than-or-equal comparison: ``self >= other``.

    Parameters
    ----------
    other : Tensor or scalar
        Right-hand operand; broadcast against ``self``.

    Returns
    -------
    Tensor
        Boolean tensor with broadcast shape.

    Notes
    -----
    Math:
    .. math::
        \text{out}_i = \mathbb{1}\!\left[\text{self}_i \geq \text{other}_i\right]

    Any comparison involving NaN returns ``False`` per IEEE 754.  Not
    differentiable.

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([1.0, 2.0, 3.0])
    >>> a >= 2.0
    tensor([False, True, True], dtype=lucid.bool)
    """
    return _compare(_C_engine.greater_equal, self, other)


def __getitem__(self: Tensor, idx: _IndexType) -> Tensor:
    r"""Tensor indexing: ``self[idx]``.

    Dispatches to :func:`lucid._tensor._indexing._getitem`, which
    supports both basic and advanced indexing semantics.

    Parameters
    ----------
    idx : int, slice, None, Ellipsis, Tensor, or tuple of these
        Index specification.  Supported atoms:

        * ``int`` — pick a single position along an axis (reduces rank
          by 1).
        * ``slice`` — Python ``start:stop:step`` window (preserves
          rank).
        * ``None`` — insert a new axis of length 1.
        * ``...`` (Ellipsis) — fill in remaining axes with full slices.
        * integer Tensor — gather along an axis (advanced indexing).
        * boolean Tensor — mask selection (advanced indexing).
        * tuple — combine the above across multiple axes.

    Returns
    -------
    Tensor
        View or gather result.  On the CPU basic indexing produces a
        view that shares storage with ``self``, a positive step such as
        ``x[::2]`` included; advanced indexing copies, as do a negative
        step and any index on metal.

    Notes
    -----
    Math (basic case, single axis):
    .. math::
        \text{out}_{j_1 \dots j_{k}} = \text{self}_{j_1 \dots j_{r-1} \, i \, j_{r+1} \dots}

    Advanced indexing follows NumPy's gather semantics: integer
    tensors are broadcast against one another and used to fetch
    elements; boolean masks select a flat 1-D subset along the masked
    axes.  Indexing participates in autograd via a sparse scatter in
    the backward pass.

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    >>> a[0]
    tensor([1., 2., 3.])
    >>> a[:, 1]
    tensor([2., 5.])
    >>> a[a > 3]
    tensor([4., 5., 6.])
    """
    return _getitem(self, idx)


def __setitem__(self: Tensor, idx: _IndexType, value: TensorOrScalar) -> None:
    r"""In-place indexed assignment: ``self[idx] = value``.

    Dispatches to :func:`lucid._tensor._indexing._setitem`.  Supports
    the same index forms as :meth:`__getitem__`.

    Parameters
    ----------
    idx : int, slice, None, Ellipsis, Tensor, or tuple of these
        Index specification — see :meth:`__getitem__` for the full
        list.
    value : Tensor or scalar
        Right-hand value to assign.  Broadcast against the shape of
        the selected region and promoted to ``self.dtype``.

    Returns
    -------
    None
        The assignment mutates ``self`` in place.

    Notes
    -----
    Math (basic case):
    .. math::
        \text{self}_{j_1 \dots j_{r-1} \, i \, j_{r+1} \dots} \leftarrow \text{value}_{j_1 \dots}

    The mutated region is disconnected from the autograd graph for
    the previous values — subsequent backward passes will see only
    the new values.  Assigning into a leaf tensor with
    ``requires_grad=True`` raises because it would invalidate saved
    activations.

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([1.0, 2.0, 3.0, 4.0])
    >>> a[1:3] = 0
    >>> a
    tensor([1., 0., 0., 4.])
    """
    _setitem(self, idx, value)


_DUNDER_METHODS = (
    __add__,
    __radd__,
    __iadd__,
    __sub__,
    __rsub__,
    __isub__,
    __mul__,
    __rmul__,
    __imul__,
    __truediv__,
    __rtruediv__,
    __itruediv__,
    __floordiv__,
    __rfloordiv__,
    __pow__,
    __rpow__,
    __ipow__,
    __matmul__,
    __rmatmul__,
    __neg__,
    __abs__,
    __invert__,
    __and__,
    __rand__,
    __or__,
    __ror__,
    __xor__,
    __rxor__,
    __int__,
    __index__,
    __float__,
    __complex__,
    __mod__,
    __rmod__,
    __divmod__,
    __rdivmod__,
    __lshift__,
    __rlshift__,
    __rshift__,
    __rrshift__,
    __eq__,
    __ne__,
    __lt__,
    __le__,
    __gt__,
    __ge__,
    __getitem__,
    __setitem__,
)


def _inject_dunders(cls: type) -> None:
    """Attach all dunder methods to the Tensor class."""
    type.__setattr__(cls, "__array_priority__", 1000)
    for fn in _DUNDER_METHODS:
        fn.__qualname__ = f"_inject_dunders.<locals>.{fn.__name__}"
        setattr(cls, fn.__name__, fn)
