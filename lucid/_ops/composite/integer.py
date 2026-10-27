"""Integer-only element-wise composites: ``gcd`` and ``lcm``.

The engine has no kernel for either, so each runs through Python's
``math`` per element.  Both are defined on integer tensors only and are
not differentiable.
"""

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import lucid
from lucid import _promotion
from lucid._C import engine as _C_engine
from lucid._ops.composite._opmath import _promote_pair
from lucid._ops.composite._shared import _binary_device, _is_tensor

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def _tensor_operand(op: str, name: str, x: object) -> Tensor:
    """``x``, refused with ``TypeError`` unless it is a tensor."""
    if not _is_tensor(x):
        raise TypeError(
            f"lucid.{op}(): argument '{name}' must be a Tensor, " f"got {type(x).__name__}"
        )
    return x


def _integer_pair(op: str, input: object, other: object) -> tuple[Tensor, Tensor, _C_engine.Device]:
    """``input`` and ``other`` at their common dtype and broadcast shape.

    With them, the device the answer lands on (:func:`_binary_device`),
    read before broadcasting, which would give a 0-d operand a shape.

    Both must be tensors whose common dtype is an integer one, as the
    reference requires: an integer beside a bool promotes to the integer,
    while two bools, or anything beside a float, is refused.  A float used
    to be truncated without a word, and a Python number failed deep inside
    the element loop.

    Raises
    ------
    TypeError
        If either operand is not a tensor, or their common dtype is not an
        integer dtype.
    lucid._C.engine.DeviceMismatch
        If the operands are on two devices and neither is a 0-d CPU tensor.
    """
    x, y = _tensor_operand(op, "input", input), _tensor_operand(op, "other", other)
    device = _binary_device(op, x, y)
    a, b = _promote_pair(x, y)
    if not _promotion.is_integer(a._impl.dtype):
        raise TypeError(
            f"lucid.{op}(): expected integer tensors (int8, int16, int32 or "
            f"int64), got {x.dtype} and {y.dtype}, which meet at {a.dtype}"
        )
    shape = lucid.broadcast_shapes(a.shape, b.shape)
    # Only an operand of another shape becomes a view: a view of ``input``
    # still alive when ``gcd_`` writes would make it an aliased write.
    return (
        a if tuple(a.shape) == shape else a.broadcast_to(shape),
        b if tuple(b.shape) == shape else b.broadcast_to(shape),
        device,
    )


def _per_element(
    fn: Callable[[int, int, int], int], op: str, input: object, other: object
) -> Tensor:
    """``fn(x, y, bits)`` of each broadcast pair of integer elements, in Python.

    ``bits`` is the result dtype's width: ``fn`` answers within it, as the
    reference's integer kernel would.
    """
    a, b, device = _integer_pair(op, input, other)
    bits = _promotion.bit_width(a._impl.dtype)
    pairs = zip(_int_values(a), _int_values(b), strict=True)
    result = [fn(x, y, bits) for x, y in pairs]
    return lucid.tensor(result, dtype=a.dtype, device=device).reshape(a.shape)


def _wrapped(value: int, bits: int) -> int:
    """``value`` wrapped to a ``bits``-wide two's-complement integer."""
    value &= (1 << bits) - 1
    return value - (1 << bits) if value >> (bits - 1) else value


def _gcd(x: int, y: int, bits: int) -> int:
    """The reference's ``gcd``: ``math.gcd``, except at the most negative value.

    In int32 and int64 the answer is ``math.gcd`` wrapped to the dtype, so
    only a gcd of ``2**(bits-1)`` itself — ``gcd(-2**63, 0)`` — comes back
    negative.  An int8 or int16 operand is widened to C's ``int`` for its
    absolute value and narrowed back, so the most negative value stays
    negative and Euclid's remainders (C's, signed like the dividend) carry
    the sign: in int8 ``gcd(-128, 0)`` is ``-128`` and ``gcd(117, -128)``
    is ``-1``.
    """
    if bits >= 32:
        return _wrapped(math.gcd(x, y), bits)
    a, b = _wrapped(abs(x), bits), _wrapped(abs(y), bits)
    if a >= 0 and b >= 0:
        return math.gcd(a, b)
    while a != 0:
        a, b = _c_remainder(b, a), a
    return b


def _c_remainder(b: int, a: int) -> int:
    """``b % a`` as C computes it: the quotient truncated, so signed like ``b``."""
    r = abs(b) % abs(a)
    return -r if b < 0 else r


def _lcm(x: int, y: int, bits: int) -> int:
    """The reference's ``lcm``: ``|x / gcd * y|`` in C integer arithmetic.

    The product is taken at the C-promoted width — at least 32 bits, so an
    int8 or int16 one does not wrap there — and wraps in it; its absolute
    value (which leaves the most negative value as it is) is then narrowed
    to the dtype.  So the int8 ``lcm(100, 99)`` is ``9900`` narrowed,
    ``-84``, while the int64 ``lcm(2**62, 3)`` wraps to ``-2**62`` first and
    answers ``2**62``.
    """
    g = math.gcd(x, y)
    if g == 0:
        return 0
    wide = max(bits, 32)
    return _wrapped(_wrapped(abs(_wrapped(x // g * y, wide)), wide), bits)


def _int_values(x: Tensor) -> list[int]:
    """The elements of an integer tensor in order, read back in one transfer."""
    vals = x.reshape(-1).tolist()
    assert isinstance(vals, list)
    return [v for v in vals if isinstance(v, int)]


def gcd(input: Tensor, other: Tensor) -> Tensor:
    """Element-wise greatest common divisor for integer tensors.

    Falls through to Python's ``math.gcd`` per element — the engine
    has no native GCD kernel and this path is for low-rate utility use
    (e.g. shape arithmetic), not hot-loop computation.  Non-differentiable.

    Parameters
    ----------
    input, other : Tensor
        Tensors whose common dtype is an integer one — an integer tensor
        beside a bool one promotes to the integer — broadcastable against
        each other.  A 0-d CPU tensor follows the other operand's device.

    Returns
    -------
    Tensor
        The broadcast shape at the operands' common dtype, on their common
        device.  Each element is ``math.gcd(input_i, other_i)`` wrapped to
        that dtype, as the reference's kernel wraps: non-negative except
        ``gcd(-2**63, 0)`` in int64, which wraps back to ``-2**63``.

    Raises
    ------
    TypeError
        If either operand is not a tensor (a Python number is refused, as
        the reference refuses it), or the operands meet at a float or bool
        dtype.
    lucid._C.engine.DeviceMismatch
        If the operands are on two devices and neither is a 0-d CPU tensor.

    Examples
    --------
    >>> import lucid
    >>> lucid.gcd(lucid.tensor([12, 18]), lucid.tensor([8])).tolist()
    [4, 2]
    """
    return _per_element(_gcd, "gcd", input, other)


def lcm(input: Tensor, other: Tensor) -> Tensor:
    """Element-wise least common multiple for integer tensors.

    :func:`gcd`'s contract — the same operands, broadcast, promotion,
    devices and refusals — with the least common multiple per element,
    ``0`` wherever either operand is ``0``.  A multiple past the dtype's
    range wraps as the reference's kernel wraps it: ``lcm`` of the int8
    ``100`` and ``99`` is ``-84``, and of the int64 ``2**62`` and ``3`` is
    ``2**62``.  Non-differentiable.

    Parameters
    ----------
    input, other : Tensor
        Integer tensors, broadcastable against each other.

    Returns
    -------
    Tensor
        The least common multiple of each pair, at the broadcast shape and
        the operands' common dtype and device.

    Examples
    --------
    >>> import lucid
    >>> lucid.lcm(lucid.tensor([4, 6]), lucid.tensor([6])).tolist()
    [12, 6]
    """
    return _per_element(_lcm, "lcm", input, other)


__all__ = ["gcd", "lcm"]

# Bound at runtime, not only for type checking: PEP 649 evaluates a
# function's annotations in its own module globals when something asks for
# them, so a name that exists only under ``TYPE_CHECKING`` makes
# ``inspect.signature`` raise NameError.  Imported at the bottom because
# the module is part of the cycle that ``Tensor`` closes.
from lucid._tensor.tensor import Tensor
