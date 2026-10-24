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
from lucid._ops.composite._opmath import _promote_pair
from lucid._ops.composite._shared import _is_tensor

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def _integer_operand(op: str, name: str, x: object) -> Tensor:
    """``x``, refused unless it is a tensor of an integer dtype, bool excluded.

    The reference defines the integer-only functions on nothing else.  A
    float used to be truncated without a word, and a Python number failed
    deep inside the element loop.

    Raises
    ------
    TypeError
        If ``x`` is not a tensor, or its dtype is not an integer one.
    """
    if not _is_tensor(x):
        raise TypeError(
            f"lucid.{op}(): argument '{name}' must be a Tensor, "
            f"got {type(x).__name__}"
        )
    if not _promotion.is_integer(x._impl.dtype):
        raise TypeError(
            f"lucid.{op}(): argument '{name}' must have an integer dtype "
            f"(int8, int16, int32 or int64), got {x.dtype}"
        )
    return x


def _integer_pair(op: str, input: object, other: object) -> tuple[Tensor, Tensor]:
    """``input`` and ``other`` at their common integer dtype and broadcast shape."""
    a, b = _promote_pair(
        _integer_operand(op, "input", input), _integer_operand(op, "other", other)
    )
    shape = lucid.broadcast_shapes(a.shape, b.shape)
    # Only an operand of another shape becomes a view: a view of ``input``
    # still alive when ``gcd_`` writes would make it an aliased write.
    return (
        a if tuple(a.shape) == shape else a.broadcast_to(shape),
        b if tuple(b.shape) == shape else b.broadcast_to(shape),
    )


def _per_element(
    fn: Callable[[int, int], int], op: str, input: object, other: object
) -> Tensor:
    """``fn`` of each broadcast pair of integer elements, through Python."""
    a, b = _integer_pair(op, input, other)
    pairs = zip(_int_values(a), _int_values(b), strict=True)
    result = [fn(x, y) for x, y in pairs]
    return lucid.tensor(result, dtype=a.dtype, device=a.device).reshape(a.shape)


def _int_values(x: Tensor) -> list[int]:
    """The elements of an integer tensor in order, read back in one transfer.

    Flattened here rather than through ``x.reshape(-1)``: a view of
    ``input``, even one already dropped, makes the write of ``gcd_`` bump
    ``input``'s version twice.
    """
    flat: list[int] = []
    pending: list[object] = [x.tolist()]
    while pending:
        item = pending.pop()
        if isinstance(item, list):
            pending.extend(reversed(item))
        elif isinstance(item, int):
            flat.append(item)
    return flat


def gcd(input: Tensor, other: Tensor) -> Tensor:
    """Element-wise greatest common divisor for integer tensors.

    Falls through to Python's ``math.gcd`` per element — the engine
    has no native GCD kernel and this path is for low-rate utility use
    (e.g. shape arithmetic), not hot-loop computation.  Non-differentiable.

    Parameters
    ----------
    input, other : Tensor
        Integer tensors, broadcastable against each other.

    Returns
    -------
    Tensor
        The broadcast shape at the operands' common dtype.  Each element
        is ``math.gcd(input_i, other_i)``, never negative.

    Raises
    ------
    TypeError
        If either operand is not a tensor of an integer dtype — a float or
        bool tensor, or a Python number.

    Examples
    --------
    >>> import lucid
    >>> lucid.gcd(lucid.tensor([12, 18]), lucid.tensor([8])).tolist()
    [4, 2]
    """
    return _per_element(math.gcd, "gcd", input, other)


def lcm(input: Tensor, other: Tensor) -> Tensor:
    """Element-wise least common multiple for integer tensors.

    :func:`gcd`'s contract — the same operands, broadcast, promotion and
    refusals — with ``math.lcm`` per element, never negative, and ``0``
    wherever either operand is ``0``.  Non-differentiable.

    Parameters
    ----------
    input, other : Tensor
        Integer tensors, broadcastable against each other.

    Returns
    -------
    Tensor
        ``math.lcm(input_i, other_i)`` at the broadcast shape and the
        operands' common dtype.

    Examples
    --------
    >>> import lucid
    >>> lucid.lcm(lucid.tensor([4, 6]), lucid.tensor([6])).tolist()
    [12, 6]
    """
    return _per_element(math.lcm, "lcm", input, other)


__all__ = ["gcd", "lcm"]

# Bound at runtime, not only for type checking: PEP 649 evaluates a
# function's annotations in its own module globals when something asks for
# them, so a name that exists only under ``TYPE_CHECKING`` makes
# ``inspect.signature`` raise NameError.  Imported at the bottom because
# the module is part of the cycle that ``Tensor`` closes.
from lucid._tensor.tensor import Tensor
