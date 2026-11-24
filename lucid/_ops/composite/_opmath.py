"""The dtype a composite special function computes in.

A special function composed of engine ops computes in whatever dtype its
operands arrive in, and two kinds of operand are not fit for that:

* a discrete one (:func:`_real`) — the engine's own ``real_valued``
  schemas promote an integer before the kernel sees it, but a composite
  has no schema, so the promotion has to be asked for.  It is the one
  every float-valued op makes, :func:`lucid._dispatch._float_input`;
* a 16-bit float (:func:`_in_float32`) — series, continued fractions and
  polynomial fits carry constants and intermediates outside float16's
  range (``2**24`` as a rescaling threshold, ``Γ`` and ``exp`` of the
  argument) and coefficients past bfloat16's 8 bits.  ``zeta`` and
  ``gammainc`` answered NaN in float16, ``bessel_y0`` NaN on ``(0, 8]``,
  ``bessel_j0(300)`` in bfloat16 was 30 times too small.  The reference
  framework evaluates these kernels in float32 ("opmath") and rounds the
  answer to the 16-bit dtype; so does this.

:func:`_in_float32_as` is the one place an operand set is widened and the answer
narrowed: the special functions reach it through :func:`_in_float32`, the accumulating
products (``addmm``, ``addcmul``, ``lerp``...) call it with the dtype their own operand
rule gives.
"""

from collections.abc import Callable

import lucid
from lucid import _promotion
from lucid._C import engine as _C_engine
from lucid._dispatch import _binary_operands, _float_input, _unwrap, _wrap
from lucid._dtype import _ENGINE_TO_DTYPE
from lucid._ops.composite._shared import _is_tensor
from lucid._tensor.tensor import Tensor
from lucid._types import TensorOrScalar


def _real(x: Tensor) -> Tensor:
    """``x`` as the operand of a function of a real variable.

    A bool or integer operand is cast to the default float, as every
    float-valued op casts one (:func:`lucid._dispatch._float_input`).  It
    used to cast to float32 whatever the default, so ``i0e`` of an int64 was
    float32 under a float64 default, where the reference answers float64.

    It matters where the composition negates before it reaches a
    promoting op.  ``entr`` computes ``-x * log(x)``, and with a bool
    operand the negation runs first — MLX refuses to negate a bool and
    says to use logical_not, so the whole function failed on Metal while
    the CPU answered.  Promoting at the door removes the question.

    A complex operand is refused, as the reference refuses it: the cast
    this used to make dropped the imaginary part without a word, so
    ``i0e(1+1j)`` answered ``i0e(1)``.

    Raises
    ------
    TypeError
        If ``x`` is complex.
    """
    x = _float_input(x)
    if lucid.is_complex(x):
        raise TypeError(f"expected a real input to a function of a real variable, got {x.dtype}")
    return x


def _promote_pair(
    a: TensorOrScalar, b: TensorOrScalar, *, float_valued: bool = False
) -> tuple[Tensor, Tensor]:
    """Bring ``a`` and ``b`` to their common tensor dtype through :func:`_binary_operands`.

    ``float_valued=True`` lifts an integral or bool pair to the default float
    dtype, as ``true_divide``, ``xlogy``, and ``copysign`` do.  An operand
    already at that dtype is handed on as the tensor the caller holds, not a
    second wrapper of its storage.
    """
    ai, bi = _binary_operands(a, b, true_div=float_valued)
    return (
        a if _is_tensor(a) and a._impl is ai else _wrap(ai),
        b if _is_tensor(b) and b._impl is bi else _wrap(bi),
    )


def _real_pair(op: str, a: TensorOrScalar, b: TensorOrScalar) -> tuple[Tensor, Tensor]:
    """``a`` and ``b`` as the operands of a function of two real variables.

    Either may be a Python number, as the reference's ``Scalar`` overloads
    take one: it follows the tensor's device and is weak in dtype, so
    ``zeta(x, 2.0)`` answers in ``x``'s dtype.  The pair is brought to its
    common dtype (:func:`_promote_pair`, an integral one lifted to the
    default float) and each operand checked by :func:`_real`.  Two Python
    numbers are refused, as the reference refuses them: there is no tensor
    to take a device from.

    Raises
    ------
    TypeError
        If neither operand is a tensor, or either is complex.
    """
    if not (_is_tensor(a) or _is_tensor(b)):
        raise TypeError(
            f"{op}: expected a Tensor for at least one operand, got "
            f"{type(a).__name__} and {type(b).__name__}"
        )
    a, b = _promote_pair(a, b, float_valued=True)
    return _real(a), _real(b)


def _in_float32_as(common: _C_engine.Dtype, fn: Callable[..., Tensor], *operands: Tensor) -> Tensor:
    """``fn(*operands)``, evaluated in float32 when ``common``, the dtype it answers in, is 16-bit.

    The reference's kernels for a float16 or bfloat16 tensor compute in float32 ("op math") and
    round once, when the result is stored.  A composite of engine ops rounds after each one: a
    scale factor to the half type, then a product, then a sum.  Every operand is widened first
    here and only the answer is narrowed, so the whole expression is the float32 one.  The casts
    are differentiable, so a gradient reaches each operand in the operand's own dtype.  Any other
    ``common`` takes ``fn`` as it is, at the cost of one test.

    ``fn`` is usually the op itself, called again on the widened operands: the half path then
    needs no second copy of the arithmetic, and the float32 path is the code it always was.

    Parameters
    ----------
    common : Dtype
        The dtype ``fn``'s answer has on ``operands`` as they are.  The caller decides it by its
        own operand rule, so a set of operands the op refuses is never widened into one it
        accepts.
    fn : callable
        The arithmetic over tensors, returning one tensor.
    *operands : Tensor
        What ``fn`` is called with.

    Returns
    -------
    Tensor
        ``fn``'s answer at ``common``.
    """
    if not _promotion.is_half(common):
        return fn(*operands)
    return fn(*(t.to(lucid.float32) for t in operands)).to(_ENGINE_TO_DTYPE[common])


def _in_float32(fn: Callable[..., Tensor], *operands: Tensor) -> Tensor:
    """``fn(*operands)``, evaluated in float32 when the answer is 16-bit.

    The answer's dtype is the operands' common one (:func:`~lucid._promotion.result_type`);
    when that is float16 or bfloat16 every operand is widened to float32 first and the result
    rounded back, so only the answer carries 16-bit precision (:func:`_in_float32_as`).  Any
    other dtype is passed through untouched.
    """
    return _in_float32_as(_promotion.result_type(*(_unwrap(t) for t in operands)), fn, *operands)


def _nan_quiet_pick(input: Tensor, other: Tensor, take_first: Tensor, is_max: bool) -> Tensor:
    """``where(take_first | isnan(other), input, other)`` with the signed-zero rule.

    The reference ranks a zero tie: the max is ``+0`` unless both are ``-0``, the min is
    ``-0`` unless both are ``+0``.  A plain ``where`` keeps the first operand's zero, so a
    tie of zeros takes ``input + other`` (``-(-input + -other)`` for the min), the sum whose
    sign is the ranked one.  ``other`` is detached there: a tie's gradient goes to the first
    operand only, as ``take_first`` already decides everywhere else.  Integers have no signed
    zero and skip the extra work.
    """
    picked = lucid.where(lucid.logical_or(take_first, lucid.isnan(other)), input, other)
    if not input.is_floating_point():
        return picked
    other_value = other.detach()
    if is_max:
        ranked = input + other_value
    else:
        ranked = lucid.negative(lucid.negative(input) + lucid.negative(other_value))
    zero_tie = lucid.logical_and(input == 0, other == 0)
    return lucid.where(zero_tie, ranked, picked)
