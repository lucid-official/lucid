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
"""

from collections.abc import Callable

import lucid
from lucid._dispatch import _binary_operands, _float_input, _wrap
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
        raise TypeError(
            f"expected a real input to a function of a real variable, got {x.dtype}"
        )
    return x


def _real_pair(op: str, a: TensorOrScalar, b: TensorOrScalar) -> tuple[Tensor, Tensor]:
    """``a`` and ``b`` as the operands of a function of two real variables.

    Either may be a Python number, as the reference's ``Scalar`` overloads
    take one: it follows the tensor's device and is weak in dtype, so
    ``zeta(x, 2.0)`` answers in ``x``'s dtype.  The pair is brought to its
    common dtype, an integral one lifted to the default float, and each
    operand checked by :func:`_real`.  Two Python numbers are refused, as
    the reference refuses them: there is no tensor to take a device from.

    Raises
    ------
    TypeError
        If neither operand is a tensor, or either is complex.
    """
    if not (isinstance(a, Tensor) or isinstance(b, Tensor)):
        raise TypeError(
            f"{op}: expected a Tensor for at least one operand, got "
            f"{type(a).__name__} and {type(b).__name__}"
        )
    ai, bi = _binary_operands(a, b, true_div=True)
    # An operand already at the common dtype is handed on as the tensor the
    # caller holds, not a second wrapper of its storage.
    return (
        _real(a if isinstance(a, Tensor) and a._impl is ai else _wrap(ai)),
        _real(b if isinstance(b, Tensor) and b._impl is bi else _wrap(bi)),
    )


def _in_float32(fn: Callable[..., Tensor], *operands: Tensor) -> Tensor:
    """``fn(*operands)``, evaluated in float32 when the answer is 16-bit.

    The answer's dtype is the operands' common one; when that is float16
    or bfloat16 every operand is widened to float32 first and the result
    rounded back, so only the answer carries 16-bit precision.  Any other
    dtype is passed through untouched.  The casts are differentiable, so
    gradients flow as they would without them.
    """
    common = operands[0].dtype
    if len(operands) == 2:
        common = lucid.result_type(operands[0], operands[1])
    if common not in (lucid.float16, lucid.bfloat16):
        return fn(*operands)
    return fn(*(t.to(lucid.float32) for t in operands)).to(common)
