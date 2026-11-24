"""A Python scalar read as the element type of the tensor it scales (LCD-1076).

``alpha`` and ``beta`` of the ``add`` family and the scaled products, ``value`` of
``addcmul`` / ``addcdiv`` and the ``weight`` of ``lerp`` scale a tensor by a Python
number.  A complex number on a real tensor was cast to the tensor's dtype by that scale,
which dropped its imaginary part without a word (``add(x, y, alpha=1j)`` was ``x + y``), or
multiplied through the operator and turned a real result into a complex one.  The
reference answers neither: it refuses the number, in two rules that are its own and are
kept apart here.

* :func:`check_alpha` -- the ``alpha`` of ``add``, ``sub``, ``rsub`` and of ``addr`` on an
  integral tensor.  A float is refused for a bool or integer result, and a complex number is
  judged by its *type*: it is refused on every result that is not complex, even when its
  imaginary part is zero.
* :func:`cast_scalar` -- every other scalar, which the reference converts to the tensor's
  element type (``addmm``, ``addcmul``, ``lerp``, ``index_add``...).  A float becomes an
  integer on an integer tensor, a complex number with no imaginary part is its real part,
  and one with an imaginary part has no value in the type and is refused.

The caller that owns the scalar applies the rule (:func:`lucid._ops._adapters._scale_by_alpha`
for the first, the composites of :mod:`lucid._ops.composite.blas` and ``index_add`` for the
second), so a free function, its ``Tensor`` method and its in-place twin refuse alike.
"""

from lucid import _promotion
from lucid._C import engine as _C_engine
from lucid._types import Scalar

_D = _C_engine.Dtype

#: The element type the reference's kernels convert a scalar to, as its overflow refusal
#: names it.  A half tensor's scalar is read in float32 (op-math), so it says ``float``.
_READ_AS: dict[_C_engine.Dtype, str] = {
    _D.Bool: "bool",
    _D.I8: "int8_t",
    _D.I16: "int16_t",
    _D.I32: "int",
    _D.I64: "int64_t",
    _D.F16: "float",
    _D.BF16: "float",
    _D.F32: "float",
    _D.F64: "double",
}


def check_alpha(dtype: _C_engine.Dtype, alpha: Scalar | complex, name: str = "alpha") -> None:
    """Refuse an ``alpha`` that an ``add`` / ``sub`` result of ``dtype`` cannot hold.

    A float, or a complex number, on a bool or integer result is "a floating point
    number"; a complex number on a floating result is "a complex number".  A complex
    result holds any of them.

    Parameters
    ----------
    dtype : Dtype
        The dtype of the result the scalar scales.
    alpha : bool or int or float or complex
        The scalar.
    name : str, optional
        How the refusal spells the scalar (``addr`` has a ``beta`` too).

    Raises
    ------
    RuntimeError
        ``alpha`` is a float or complex number on a bool or integer result, or a complex
        number on a floating one.
    """
    if isinstance(alpha, (float, complex)) and _promotion.is_integral(dtype):
        raise RuntimeError(
            f"For integral input tensors, argument {name} must not be a floating point number."
        )
    if isinstance(alpha, complex) and not _promotion.is_complex(dtype):
        raise RuntimeError(
            f"For non-complex input tensors, argument {name} must not be a complex number."
        )


def real_scalar(dtype: _C_engine.Dtype, scalar: Scalar | complex) -> Scalar:
    """``scalar`` as a tensor of the non-complex ``dtype`` reads it: made real, or refused.

    A real scalar comes back as it is.  A complex scalar whose imaginary part is zero is its
    real part; one with an imaginary part has no value in ``dtype`` and is refused, as
    the reference's scalar conversion refuses it.

    Parameters
    ----------
    dtype : Dtype
        A non-complex dtype; it names the type in the refusal.
    scalar : bool or int or float or complex
        The scalar.

    Returns
    -------
    bool or int or float
        ``scalar``, or its real part.

    Raises
    ------
    RuntimeError
        ``scalar`` has an imaginary part.
    """
    scalar = _promotion.as_number(scalar)
    if not isinstance(scalar, complex):
        return scalar
    if scalar.imag != 0:
        raise RuntimeError(f"value cannot be converted to type {_READ_AS[dtype]} without overflow")
    return scalar.real


def cast_scalar(
    dtype: _C_engine.Dtype, factor: Scalar | complex, *, cast_integral: bool = True
) -> Scalar | complex:
    """``factor`` as the reference's scaled ops read it for a tensor of ``dtype``.

    ``addmm``, ``addmv``, ``baddbmm``, ``addbmm``, ``addcmul`` and ``index_add`` cast their
    scalars to the tensor dtype (``beta=0.5`` on ``int32`` is ``0``), where ``addr``, ``add``
    and ``sub`` refuse a float for an integer tensor: with ``cast_integral=False`` an
    integral ``dtype`` keeps ``factor`` for :func:`check_alpha` to refuse by name.  A complex
    ``factor`` is made real (:func:`real_scalar`) on every dtype that is not complex.

    Parameters
    ----------
    dtype : Dtype
        The dtype of the tensor the scalar scales.
    factor : bool or int or float or complex
        The scalar.
    cast_integral : bool, optional
        Whether an integral ``dtype`` casts a float ``factor`` (the default) or leaves it be.

    Returns
    -------
    bool or int or float or complex
        ``factor`` as ``dtype`` reads it.

    Raises
    ------
    RuntimeError
        ``factor`` has an imaginary part and ``dtype`` is not complex.
    """
    if not cast_integral and _promotion.is_integral(dtype):
        return factor
    if not _promotion.is_complex(dtype):
        factor = real_scalar(dtype, factor)
    if isinstance(factor, float) and _promotion.is_integer(dtype):
        return int(factor)
    return factor
