"""Predicates and zero-cost identity ops surfaced for reference-framework parity."""

from typing import TYPE_CHECKING

import lucid
from lucid._ops.composite._shared import _is_tensor
from lucid._types import TensorLike

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def numel(input: Tensor) -> int:
    r"""Return the total number of elements in a tensor.

    Equivalent to the product of the shape dimensions, returned as a
    plain Python ``int`` so it can be used in pure-Python control flow
    (loops, conditionals) without triggering autograd tracing.

    Parameters
    ----------
    input : Tensor
        Input tensor.

    Returns
    -------
    int
        Total number of scalar elements:
        :math:`\prod_{d} \text{shape}[d]`.

    Notes
    -----
    For a tensor of shape :math:`(s_0, s_1, \dots, s_{n-1})`:

    .. math::

        \text{numel}(x) = \prod_{i = 0}^{n - 1} s_i.

    A 0-D (scalar) tensor has ``numel == 1``; a tensor with any dimension
    of size ``0`` has ``numel == 0``.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.zeros((2, 3, 4))
    >>> lucid.numel(x)
    24
    """
    return int(input.numel())


def is_storage(obj: object, /) -> bool:
    r"""Predicate: is the object a separate Storage container?

    Lucid does not expose a distinct Storage type — tensor data ownership
    is folded into the :class:`Tensor` class itself — so this predicate
    always returns ``False``. Provided for API parity with reference
    frameworks that maintain a Tensor/Storage split.

    Parameters
    ----------
    obj : object
        Object to test.

    Returns
    -------
    bool
        Always ``False`` in Lucid.

    Notes
    -----
    Lucid uses an arc-buffer model where every :class:`Tensor` directly
    owns its byte buffer (see the ``arch-storage-ownership`` ADR). There
    is therefore no separate Storage handle to query.

    Examples
    --------
    >>> import lucid
    >>> lucid.is_storage(lucid.zeros(3))
    False
    """
    del obj  # reference-signature parity; Lucid has no separate Storage type
    return False


def is_nonzero(input: Tensor) -> bool:
    r"""Predicate: is a single-element tensor non-zero?

    Reduces a scalar (one-element) tensor to a Python ``bool``. Raises if
    the tensor has more than one element so that the caller is forced to
    pick an explicit reduction (e.g. ``.any()`` / ``.all()``) for
    multi-element inputs.

    Parameters
    ----------
    input : Tensor
        Single-element tensor (i.e. ``x.numel() == 1``).

    Returns
    -------
    bool
        ``True`` iff the (sole) element is non-zero.

    Raises
    ------
    RuntimeError
        If ``x.numel() != 1``.

    Notes
    -----
    Equivalent to:

    .. math::

        \text{out} = (x_0 \neq 0)

    where :math:`x_0` is the single scalar value extracted via
    :meth:`Tensor.item`. The strict shape check prevents the silent
    bug where one would accidentally call ``bool(tensor)`` on an
    unintentionally multi-element tensor.

    Examples
    --------
    >>> import lucid
    >>> lucid.is_nonzero(lucid.tensor(0.0))
    False
    >>> lucid.is_nonzero(lucid.tensor(3.5))
    True
    """
    if input.numel() != 1:
        raise RuntimeError("is_nonzero is defined only for scalar tensors (numel == 1)")
    return bool(input.item() != 0)


def is_same_size(input: Tensor, other: Tensor) -> bool:
    r"""Predicate: do two tensors have identical shapes?

    Compares the shape tuples exactly — no broadcasting compatibility is
    implied. Use :func:`lucid.broadcast_to` if broadcastability is the
    question.

    Parameters
    ----------
    input : Tensor
        First tensor.
    other : Tensor
        Second tensor.

    Returns
    -------
    bool
        ``True`` iff ``tuple(a.shape) == tuple(b.shape)``.

    Notes
    -----
    Compares all dimensions including singleton axes:

    .. math::

        \text{out} = \bigwedge_{i} (\text{shape}_a[i] = \text{shape}_b[i]).

    A ``(1, 3)`` tensor is not the same size as a ``(3,)`` tensor, even
    though they broadcast.

    Examples
    --------
    >>> import lucid
    >>> a = lucid.zeros(2, 3)
    >>> b = lucid.ones(2, 3)
    >>> lucid.is_same_size(a, b)
    True
    """
    return tuple(input.shape) == tuple(other.shape)


def equal(input: Tensor, other: Tensor) -> bool:
    r"""Predicate: do two tensors have the same shape and the same elements?

    One Python ``bool`` for the whole pair, unlike the element-wise
    :func:`lucid.eq`.  The shapes must match exactly — no broadcasting —
    and the elements are compared at the pair's promoted dtype, so an
    integer tensor equals the float tensor holding the same values.

    Parameters
    ----------
    input : Tensor
        First tensor.
    other : Tensor
        Second tensor.

    Returns
    -------
    bool
        ``True`` iff ``input.shape == other.shape`` and every element pair
        compares equal.  ``NaN`` equals nothing, itself included, and
        ``-0.0`` equals ``0.0``.  Two empty tensors of one shape are equal.

    Raises
    ------
    TypeError
        If either argument is not a :class:`Tensor`.

    Notes
    -----
    Reading the answer synchronises any pending Metal work on the pair.

    Examples
    --------
    >>> import lucid
    >>> lucid.equal(lucid.tensor([1, 2]), lucid.tensor([1.0, 2.0]))
    True
    >>> lucid.equal(lucid.tensor([1, 2]), lucid.tensor([[1, 2]]))
    False
    """
    for name, x in (("input", input), ("other", other)):
        if not _is_tensor(x):
            raise TypeError(
                f"lucid.equal(): argument '{name}' must be a Tensor, " f"got {type(x).__name__}"
            )
    if tuple(input.shape) != tuple(other.shape):
        return False
    return bool(lucid.eq(input, other).all().item())


def is_neg(input: Tensor) -> bool:
    r"""Predicate: does the tensor carry a lazy "negated" flag?

    Some reference frameworks track negation as a lazy view flag that is
    materialised on access. Lucid eagerly materialises every operation,
    so this predicate is always ``False``.

    Parameters
    ----------
    input : Tensor
        Tensor to query.

    Returns
    -------
    bool
        Always ``False``.

    Notes
    -----
    Lucid's tensor model has no lazy-flag bits — negation is realised by
    the engine ``neg`` op which produces a fresh buffer (or, under
    fusion, an in-pipeline transformation that still yields concrete
    values). There is therefore nothing to query.

    Examples
    --------
    >>> import lucid
    >>> lucid.is_neg(-lucid.tensor([1.0, 2.0]))
    False
    """
    del input  # reference-signature parity; Lucid materialises negation eagerly
    return False


def is_conj(input: Tensor) -> bool:
    r"""Predicate: does the tensor carry a lazy "conjugated" flag?

    Some reference frameworks track complex conjugation as a lazy view
    flag. Lucid materialises every operation eagerly, so this predicate
    is always ``False``.

    Parameters
    ----------
    input : Tensor
        Tensor to query.

    Returns
    -------
    bool
        Always ``False``.

    Notes
    -----
    See :func:`is_neg` for the analogous discussion on lazy flag bits.
    Conjugation in Lucid is performed by :func:`lucid.conj`, which
    produces a materialised result rather than toggling a flag.

    Examples
    --------
    >>> import lucid
    >>> lucid.is_conj(lucid.tensor([1.0, 2.0]))
    False
    """
    del input  # reference-signature parity; Lucid materialises conjugation eagerly
    return False


def isin(
    elements: Tensor | TensorLike,
    test_elements: Tensor | TensorLike,
    *,
    assume_unique: bool = False,
    invert: bool = False,
) -> Tensor:
    r"""Per-element set-membership test.

    For each entry of ``elements``, checks whether the value appears
    anywhere in ``test_elements``. Implemented as a broadcasted
    equality compare followed by an OR reduction.

    Parameters
    ----------
    elements : Tensor | TensorLike
        Values to test.  Reshaped to 1-D internally; the output is
        reshaped back to ``elements.shape``.
    test_elements : Tensor | TensorLike
        The set of values to test against. Flattened internally.
    assume_unique : bool, optional
        If ``True``, assumes both ``elements`` and ``test_elements``
        contain unique items. Accepted for reference-framework parity;
        defaults to ``False``.
    invert : bool, optional
        If ``True``, return the negation of the membership test.
        Defaults to ``False``.

    Returns
    -------
    Tensor
        Boolean tensor of the same shape as ``elements``.

    Notes
    -----
    Mathematical definition (with :math:`E = \text{elements}` and
    :math:`T = \text{test\_elements}`):

    .. math::

        \text{out}_i = \bigvee_{j} (E_i = T_j).

    Cost is :math:`\mathcal{O}(|E| \cdot |T|)`. For large ``test_elements``
    a sort-based implementation would be cheaper, but this composite
    form keeps the operation differentiable-by-default (the gradient
    is zero everywhere — the output dtype is boolean — but the structure
    is preserved).

    Examples
    --------
    >>> import lucid
    >>> e = lucid.tensor([1, 2, 3, 4])
    >>> t = lucid.tensor([2, 4])
    >>> lucid.isin(e, t)
    tensor([False, True, False, True], dtype=lucid.bool)
    """
    del assume_unique
    if not _is_tensor(elements):
        elements = lucid.tensor(elements)
    if not _is_tensor(test_elements):
        test_elements = lucid.tensor(test_elements)
    e_flat = elements.reshape(-1)
    t_flat = test_elements.reshape(-1)
    n, m = e_flat.shape[0], t_flat.shape[0]
    e_bc = e_flat.unsqueeze(1).expand(n, m)
    t_bc = t_flat.unsqueeze(0).expand(n, m)
    matches = (e_bc == t_bc).to(dtype=lucid.float32).sum(1)
    out = (matches > 0.0).reshape(elements.shape)
    return ~out if invert else out


def isneginf(input: Tensor) -> Tensor:
    r"""Element-wise test for negative infinity.

    Identifies entries that are exactly :math:`-\infty` in the IEEE 754
    floating-point sense. Finite values and :math:`+\infty` both report
    ``False``.

    Parameters
    ----------
    input : Tensor
        Floating-point input tensor.

    Returns
    -------
    Tensor
        Boolean tensor of the same shape as ``input``.

    Notes
    -----
    Defined as the conjunction of "is infinite" and "is negative":

    .. math::

        \text{out}_i =
        \operatorname{isinf}(x_i) \wedge (x_i < 0).

    NaN inputs report ``False`` because NaN compares false against any
    real value, including in ``isinf``.

    Examples
    --------
    >>> import lucid
    >>> import math
    >>> x = lucid.tensor([-math.inf, -1.0, 0.0, math.inf])
    >>> lucid.isneginf(x)
    tensor([True, False, False, False], dtype=lucid.bool)
    """
    return lucid.logical_and(lucid.isinf(input), input < 0.0)


def isposinf(input: Tensor) -> Tensor:
    r"""Element-wise test for positive infinity.

    Identifies entries that are exactly :math:`+\infty` in the IEEE 754
    floating-point sense. Finite values and :math:`-\infty` both report
    ``False``.

    Parameters
    ----------
    input : Tensor
        Floating-point input tensor.

    Returns
    -------
    Tensor
        Boolean tensor of the same shape as ``input``.

    Notes
    -----
    Defined as the conjunction of "is infinite" and "is positive":

    .. math::

        \text{out}_i =
        \operatorname{isinf}(x_i) \wedge (x_i > 0).

    NaN inputs report ``False``.

    Examples
    --------
    >>> import lucid
    >>> import math
    >>> x = lucid.tensor([-math.inf, -1.0, 0.0, math.inf])
    >>> lucid.isposinf(x)
    tensor([False, False, False, True], dtype=lucid.bool)
    """
    return lucid.logical_and(lucid.isinf(input), input > 0.0)


def isreal(input: Tensor) -> Tensor:
    r"""Element-wise test for real-valued entries.

    An entry is real when its imaginary part is exactly zero, so every
    entry of a real (or integer, or bool) tensor is, and an entry of a
    complex tensor is where its imaginary part is ``0``.

    Parameters
    ----------
    input : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Boolean tensor of the same shape as ``input``.

    Notes
    -----
    .. math::

        \text{out}_i = (\operatorname{Im}(\text{input}_i) = 0).

    ``NaN`` and infinite real parts do not matter; a ``NaN`` imaginary
    part is not zero.

    Examples
    --------
    >>> import lucid
    >>> lucid.isreal(lucid.tensor([1 + 0j, 1j, float('nan') + 0j])).tolist()
    [True, False, True]
    """
    if input.is_complex():
        return lucid.imag(input) == 0.0
    return lucid.ones_like(input, dtype=lucid.bool_)


def conj_physical(input: Tensor) -> Tensor:
    r"""Eagerly materialised complex conjugate.

    Some reference frameworks distinguish ``conj`` (which may set a lazy
    "conjugated" flag on the tensor's view) from ``conj_physical`` (which
    always writes a new buffer with the conjugated values).  Lucid does
    not carry any lazy conjugation metadata, so both call into the same
    engine op — ``conj_physical`` is provided for API parity and as a
    clear signal to readers that the conjugation is materialised.

    Parameters
    ----------
    input : Tensor
        Input tensor.  Typically ``complex64``; for real dtypes the
        conjugate is the identity.

    Returns
    -------
    Tensor
        Element-wise complex conjugate of ``input``, same shape and dtype.

    Notes
    -----
    Mathematical definition for a complex element :math:`z = a + ib`:

    .. math::

        \overline{z} = a - i\,b.

    Use this spelling when porting code that explicitly depends on the
    conjugation being physically present in the storage (e.g. for
    interop with external libraries that would otherwise see the lazy
    flag).  Functionally identical to :func:`lucid.conj` in Lucid.

    Examples
    --------
    >>> import lucid
    >>> z = lucid.tensor([1+2j, 3-4j])
    >>> lucid.conj_physical(z)
    tensor([(1.-2.j), (3.+4.j)], dtype=lucid.complex64)
    """
    return lucid.conj(input) if input.is_complex() else input.clone()


def resolve_conj(input: Tensor) -> Tensor:
    r"""Materialise any pending lazy conjugation — no-op in Lucid.

    Reference frameworks that maintain a lazy "is_conj" flag on complex
    tensors use ``resolve_conj`` to force the flag to be applied to the
    underlying buffer.  Lucid has no such flag (see
    :func:`conj_physical`), so this routine simply returns the input
    unchanged.  It is provided for API parity so that ported code keeps
    working without behavioural drift.

    Parameters
    ----------
    input : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        The input ``input`` returned unchanged (no copy, no autograd op
        inserted).

    Notes
    -----
    Equivalent to the identity map:

    .. math::

        \text{resolve\_conj}(x) = x.

    See :func:`conj_physical` for an explicit, always-materialised
    conjugation.

    Examples
    --------
    >>> import lucid
    >>> z = lucid.tensor([1+2j, 3-4j])
    >>> lucid.resolve_conj(z) is z
    True
    """
    return input


def resolve_neg(input: Tensor) -> Tensor:
    r"""Materialise any pending lazy negation — no-op in Lucid.

    Reference frameworks that maintain a lazy-negation flag use this op
    to force the flag to be applied to the underlying buffer. Lucid has
    no such flag (see :func:`is_neg`), so this returns the input
    unchanged.

    Parameters
    ----------
    input : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        The input ``input``, returned unchanged.

    Notes
    -----
    Equivalent to the identity map:

    .. math::

        \text{out} = x.

    Provided strictly for API parity. Calling this op is free — no copy
    is made, no autograd node is inserted.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.tensor([1.0, 2.0])
    >>> lucid.resolve_neg(x) is x
    True
    """
    return input


def allclose(
    input: Tensor, other: Tensor, rtol: float = 1e-5, atol: float = 1e-8, equal_nan: bool = False
) -> bool:
    r"""Return ``True`` iff every element pair is close within tolerance.

    Closeness predicate:

    .. math::

        |\text{input}_i - \text{other}_i| \le \text{atol}
            + \text{rtol} \cdot |\text{other}_i|.

    Note the asymmetry — the relative term uses ``|other|``, so swapping
    arguments can change the answer for marginal cases.  This matches
    the NumPy / reference-framework convention.

    Parameters
    ----------
    input, other : Tensor
        Inputs to compare.  Must be broadcastable.
    rtol : float, optional
        Relative tolerance.  Default ``1e-5``.
    atol : float, optional
        Absolute tolerance — dominates the bound when ``|other|`` is small
        (think of comparisons near zero).  Default ``1e-8``.
    equal_nan : bool, optional
        When ``True``, ``NaN`` positions are treated as equal to other
        ``NaN`` positions.  Default ``False`` (any ``NaN`` makes the
        predicate ``False``).

    Returns
    -------
    bool
        Reduces to a single Python ``bool`` — synchronises any pending
        GPU work to evaluate the final reduction.

    See Also
    --------
    isclose : element-wise variant that returns a Tensor of bools.
    """
    diff = lucid.abs(input - other)
    tol = atol + rtol * lucid.abs(other)
    close = diff <= tol
    if equal_nan:
        both_nan = lucid.isnan(input) & lucid.isnan(other)
        close = close | both_nan
    return bool(close.all().item())


__all__ = [
    "numel",
    "is_storage",
    "is_nonzero",
    "is_same_size",
    "equal",
    "is_neg",
    "is_conj",
    "isin",
    "isneginf",
    "isposinf",
    "isreal",
    "conj_physical",
    "resolve_conj",
    "resolve_neg",
    "allclose",
]

# Bound at runtime, not only for type checking: PEP 649 evaluates a
# function's annotations in its own module globals when something asks
# for them, so a name that exists under ``TYPE_CHECKING`` alone makes
# ``inspect.signature`` raise NameError — which is what help(), an IDE
# and the docs build all call. Imported at the bottom because this
# module is inside the cycle ``Tensor`` closes.
from lucid._tensor.tensor import Tensor
