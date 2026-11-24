"""A Python scalar read as the element type of the tensor it scales (LCD-1076, LCD-1078).

``alpha`` and ``beta`` of the ``add`` family and the scaled products, ``value`` of
``addcmul`` / ``addcdiv``, the ``weight`` of ``lerp`` and the exponent of an in-place
``pow`` are Python numbers a kernel converts to the tensor's element type.  The reference
converts them in two ways, and this module is the one owner of both -- no entry point
repeats either, so a free function, its ``Tensor`` method and its in-place twin refuse alike.

* :func:`check_alpha` -- the ``alpha`` of ``add``, ``sub``, ``rsub`` and of ``addr`` on an
  integral tensor.  The scalar is judged by its *type* first: a bool is refused for a result
  that is not bool, a float for a bool or integer result, a complex number for any result that
  is not complex (even ``0j``).  What passes is then read as the element type.
* :func:`cast_scalar` -- every other scale factor, which the reference reads by its *value*
  (``Scalar::to<T>``): ``addmm``, ``addmv``, ``addbmm``, ``baddbmm``, ``addcmul``,
  ``addcdiv``, ``lerp``, ``index_add`` and ``addr`` on a floating tensor.  A bool is 0 or 1, a
  float becomes an integer on an integer tensor, a complex number with no imaginary part is
  its real part, and one with an imaginary part has no value in the type and is refused.

Reading a scalar can fail with ``value cannot be converted to type <T> without overflow``,
``<T>`` the C++ type the kernel reads: an integer must lie inside the width of its dtype
(a float is truncated, but NaN, infinity and anything past the width are refused), and a
finite number past the largest value of a floating type is refused (infinity and NaN are
not).  A bool tensor reads any scalar as ``value != 0`` and refuses none.

Which type a *half* tensor's scalar is read as depends on the op (:class:`ScalarRead`).

Both start from a Python number, and a scale factor does not always arrive as one.  The
reference takes a 0-d tensor for a ``Scalar`` argument and reads it as the number it holds
(``.item()``: a bool, int, float or complex of the tensor's own dtype), so ``alpha=tensor(2.5)``
is refused for an integer tensor as ``alpha=2.5`` is.  :func:`scale_factor` is the one entry
that makes that reading, at the top of every op that takes a scale factor: what the two rules
above receive is always a number.
"""

import abc
import enum
import functools
import math
from collections.abc import Callable, Iterable
from typing import NoReturn, override

from lucid import _promotion
from lucid._C import engine as _C_engine
from lucid._types import Scalar

_D = _C_engine.Dtype

#: The largest finite value of the floating types a scalar is read as.
_F16_MAX = 65504.0
_BF16_MAX = float.fromhex("0x1.fep+127")
_F32_MAX = float.fromhex("0x1.fffffep+127")


class ScalarRead(enum.Enum):
    """The type the reference's kernel reads a scale factor as for a half tensor.

    An integer or a float32 / float64 tensor reads its scalar as its own element type
    whatever the op.  A float16 / bfloat16 tensor does not: ``add``, ``sub``, ``addmv``,
    ``addr`` and ``index_add`` convert the scalar to the storage type (``c10::Half``,
    ``c10::BFloat16``), and the products that accumulate in float32 (``addmm``, ``addbmm``,
    ``baddbmm``, ``addcmul``, ``addcdiv``, ``lerp``) convert it to ``float``.  The two differ in
    the range a scalar may have and in the type the refusal names.  The ``float`` ops also
    *compute* in float32 and round once (:func:`lucid._ops.composite._opmath._in_float32_as`), so
    ``alpha=300.7`` is 300.7 in the product and not the 300 a bfloat16 holds.
    """

    STORAGE = enum.auto()
    OP_MATH = enum.auto()


class _Element(abc.ABC):
    """How the reference reads a Python scalar as one element type.

    Parameters
    ----------
    name : str
        The C++ type, as the reference's refusal names it.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abc.abstractmethod
    def read(self, value: Scalar | complex) -> Scalar | complex:
        """``value`` as this type holds it, or a ``RuntimeError`` when it has none."""

    def check_exponent(self, exponent: int | float) -> None:
        """Refuse a Python exponent a power of this base type cannot take (none, by default)."""
        del exponent  # the types that refuse one override this

    def refuse(self) -> NoReturn:
        raise RuntimeError(f"value cannot be converted to type {self.name} without overflow")

    def real_part(self, value: Scalar | complex) -> Scalar:
        """``value``, or its real part when it has no imaginary part (refused otherwise)."""
        if isinstance(value, complex):
            if value.imag != 0:
                self.refuse()
            return value.real
        return value


def _check_integral_power(exponent: int | float) -> None:
    """A bool or integer base refuses a negative integer exponent, and one past int64."""
    if isinstance(exponent, int):
        if exponent < 0:
            raise RuntimeError("Integers to negative integer powers are not allowed.")
        _INT64.read(exponent)


class _Bool(_Element):
    """A bool tensor reads a scalar as ``value != 0``: nothing overflows."""

    @override
    def read(self, value: Scalar | complex) -> bool:
        return bool(value)

    @override
    def check_exponent(self, exponent: int | float) -> None:
        _check_integral_power(exponent)


class _Integer(_Element):
    """An integer tensor of ``bits`` bits.

    The range check is the reference's: on the float, in double precision, which for 64 bits
    lets ``2.0 ** 63`` through (the bound rounds up to it) and the cast then saturates.
    """

    def __init__(self, name: str, bits: int) -> None:
        super().__init__(name)
        self._lowest = -(1 << (bits - 1))
        self._highest = (1 << (bits - 1)) - 1
        self._lowest_f, self._highest_f = float(self._lowest), float(self._highest)

    @override
    def read(self, value: Scalar | complex) -> int:
        value = self.real_part(value)
        if isinstance(value, float):
            if not self._lowest_f <= value <= self._highest_f:
                self.refuse()
            return min(int(value), self._highest)
        if not self._lowest <= value <= self._highest:
            self.refuse()
        return int(value)

    @override
    def check_exponent(self, exponent: int | float) -> None:
        _check_integral_power(exponent)
        if isinstance(exponent, int):
            self.read(exponent)


class _Bounded(_Element):
    """An element type whose finite values stop at ``limit``.

    Infinity and NaN are values of the type, so they are not refused; a bool is 0 or 1.
    """

    def __init__(self, name: str, limit: float) -> None:
        super().__init__(name)
        self._limit = limit

    def fit(self, value: Scalar) -> Scalar:
        if self._limit < abs(value) < math.inf:
            self.refuse()
        return int(value) if isinstance(value, bool) else value


class _Floating(_Bounded):
    """A real floating tensor: a complex scalar is its real part."""

    @override
    def read(self, value: Scalar | complex) -> Scalar:
        return self.fit(self.real_part(value))


class _Half(_Floating):
    """A half tensor, whose power reads its exponent as the storage type (float32's does not)."""

    @override
    def check_exponent(self, exponent: int | float) -> None:
        self.read(exponent)


class _Complex(_Bounded):
    """A complex tensor holds a complex scalar whole; each of its parts is bounded."""

    @override
    def read(self, value: Scalar | complex) -> Scalar | complex:
        if isinstance(value, complex):
            self.fit(value.real)
            self.fit(value.imag)
            return value
        return self.fit(value)


def _elements() -> dict[int, _Element]:
    """Every dtype's element type, keyed by the dtype's code (an engine enum hashes slowly)."""
    named: dict[_C_engine.Dtype, _Element] = {
        _D.Bool: _Bool("bool"),
        _D.I8: _Integer("int8_t", _promotion.bit_width(_D.I8)),
        _D.I16: _Integer("int16_t", _promotion.bit_width(_D.I16)),
        _D.I32: _Integer("int", _promotion.bit_width(_D.I32)),
        _D.I64: _Integer("int64_t", _promotion.bit_width(_D.I64)),
        _D.F16: _Half("c10::Half", _F16_MAX),
        _D.BF16: _Half("c10::BFloat16", _BF16_MAX),
        _D.F32: _Floating("float", _F32_MAX),
        _D.F64: _Floating("double", math.inf),
        _D.C64: _Complex("c10::complex<float>", _F32_MAX),
        _D.C128: _Complex("c10::complex<double>", math.inf),
    }
    return {dtype.value: element for dtype, element in named.items()}


_STORAGE = _elements()
#: A half tensor's scalar is a ``float`` where an op accumulates in float32.
_OP_MATH = {**_STORAGE, _D.F16.value: _STORAGE[_D.F32.value], _D.BF16.value: _STORAGE[_D.F32.value]}
_READERS = {ScalarRead.STORAGE: _STORAGE, ScalarRead.OP_MATH: _OP_MATH}
_INT64 = _STORAGE[_D.I64.value]


def _element(dtype: _C_engine.Dtype, read: ScalarRead) -> _Element:
    return _READERS[read][dtype.value]


def _negated(alpha: Scalar | complex) -> Scalar | complex:
    """``-alpha`` as the reference's scalar negation: an int64 wraps, so ``-2 ** 63`` stays."""
    return alpha if isinstance(alpha, int) and alpha == -(1 << 63) else -alpha


#: What a scale factor already is when it needs no reading: the four Python number types.
_PLAIN = frozenset(_promotion.NUMBER)


def scale_factor(factor: object, op: str, name: str) -> Scalar | complex:
    """The Python number the scale factor ``name`` of ``op`` stands for.

    A number (a NumPy scalar included) is itself.  A 0-d tensor that does not require grad
    is the number it holds, of the kind its dtype has -- bool, int, float or complex -- so the
    rules of :func:`check_alpha` and :func:`cast_scalar` judge it as they judge that number:
    ``alpha=tensor(2.5)`` on an integer tensor is refused as ``alpha=2.5`` is, and
    ``value=tensor(2.0)`` leaves an ``int32`` result ``int32``.  Reading it is a host read
    (``item()``), taken only on this path; a number costs one type test.

    Anything else is refused where the reference's argument parser refuses it, before any
    other check of the op: a tensor with a dimension, one that requires grad, a ``str``,
    ``None``, a list.

    Parameters
    ----------
    factor : object
        The argument as the caller passed it.
    op : str
        The op's name, as the refusal spells it.
    name : str
        The argument's name (``alpha``, ``beta``, ``value``).

    Returns
    -------
    bool or int or float or complex
        The number the argument is.

    Raises
    ------
    TypeError
        ``factor`` is neither a number nor a 0-d tensor that does not require grad.
    """
    number = factor if type(factor) in _PLAIN else _promotion.as_number(factor)
    if isinstance(number, _promotion.NUMBER):
        return number
    impl = _promotion.try_unwrap(factor)
    if impl is not None and not impl.shape and not impl.requires_grad:
        return impl.item()
    raise TypeError(f"{op}(): argument '{name}' must be Number, not {type(factor).__name__}")


def read_scale_factors(op: str, kwargs: dict[str, object], names: Iterable[str]) -> None:
    """Replace each scale factor of ``names`` in ``kwargs`` by the number it stands for.

    The call arguments, by keyword: every op that takes a scale factor takes it keyword-only
    (``alpha``, ``beta``, ``value``), so this reads all of them at the op's entry, before
    anything else of the op is checked, as the reference's argument parser does.  A factor
    that is already a Python number is left alone, which is the one type test the common
    call pays.  See :func:`scale_factor`.

    Parameters
    ----------
    op : str
        The op's name, as a refusal spells it.
    kwargs : dict
        The call's keyword arguments; changed in place.
    names : iterable of str
        The scale factors of the op.

    Raises
    ------
    TypeError
        A scale factor is neither a number nor a 0-d tensor that does not require grad.
    """
    for name in names:
        factor = kwargs.get(name, 1)
        if type(factor) not in _PLAIN:
            kwargs[name] = scale_factor(factor, op, name)


def scale_factors[**P, R](*names: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorate an op so its keyword-only scale factors ``names`` are read at entry.

    The declaration of which arguments of an op are scale factors.  ``_amp_op(scalars=...)``
    applies it outside the autocast entry, so the products with a scaled accumulator read
    theirs before the row can cast a tensor one.  The wrapped function sees numbers only, and
    keeps its signature, name and docstring.

    Parameters
    ----------
    *names : str
        The scale factors of the op: keyword-only parameters, as in the reference.

    Returns
    -------
    Callable
        The decorator.
    """

    def decorate(fn: Callable[P, R]) -> Callable[P, R]:
        op = fn.__name__

        @functools.wraps(fn)
        def at_entry(*args: P.args, **kwargs: P.kwargs) -> R:
            read_scale_factors(op, kwargs, names)
            return fn(*args, **kwargs)

        return at_entry

    return decorate


def check_alpha(
    dtype: _C_engine.Dtype,
    alpha: Scalar | complex,
    name: str = "alpha",
    *,
    read: ScalarRead = ScalarRead.STORAGE,
    subtracts: bool = False,
) -> None:
    """Refuse an ``alpha`` that an ``add`` / ``sub`` result of ``dtype`` cannot hold.

    A float, or a complex number, on a bool or integer result is "a floating point
    number"; a complex number on a floating result is "a complex number".  What is left
    must be readable as the element type (an integer that fits, a finite number the floating
    type can hold).  A complex result holds any of them.  The reference subtracts by adding
    ``-alpha``, so for ``sub`` and ``rsub`` it is the negated scalar that must fit: ``128`` is
    an ``int8`` alpha there and ``-128`` is not.

    Parameters
    ----------
    dtype : Dtype
        The dtype of the result the scalar scales.
    alpha : bool or int or float or complex
        The scalar, not a bool unless ``dtype`` is bool (the caller refuses that by name).
    name : str, optional
        How the refusal spells the scalar (``addr`` has a ``beta`` too).
    read : ScalarRead, optional
        What a half result reads the scalar as.
    subtracts : bool, optional
        Whether the op subtracts the scaled operand (``sub``, ``rsub``).

    Raises
    ------
    RuntimeError
        ``alpha`` is a float or complex number on a bool or integer result, a complex
        number on a floating one, or it has no value in the element type.
    """
    if isinstance(alpha, (float, complex)) and _promotion.is_integral(dtype):
        raise RuntimeError(
            f"For integral input tensors, argument {name} must not be a floating point number."
        )
    if isinstance(alpha, complex) and not _promotion.is_complex(dtype):
        raise RuntimeError(
            f"For non-complex input tensors, argument {name} must not be a complex number."
        )
    _element(dtype, read).read(_negated(alpha) if subtracts else alpha)


def cast_scalar(
    dtype: _C_engine.Dtype,
    factor: Scalar | complex,
    *,
    read: ScalarRead = ScalarRead.STORAGE,
    by_type: bool = False,
) -> Scalar | complex:
    """``factor`` as the reference's scaled ops read it for a tensor of ``dtype``.

    ``addmm``, ``addmv``, ``baddbmm``, ``addbmm``, ``addcmul``, ``addcdiv``, ``lerp`` and
    ``index_add`` convert their scalars to the tensor dtype (``beta=0.5`` on ``int32`` is
    ``0``, ``alpha=True`` is 1), where ``addr``, ``add`` and ``sub`` judge the scalar by its
    type: with ``by_type=True`` a bool, or any scalar on an integral ``dtype``, is kept for
    :func:`check_alpha` and the bool check to refuse by name.

    Parameters
    ----------
    dtype : Dtype
        The dtype of the tensor the scalar scales.
    factor : bool or int or float or complex
        The scalar.
    read : ScalarRead, optional
        What a half ``dtype`` reads the scalar as.
    by_type : bool, optional
        Whether the op judges a bool or an integral tensor's scalar by its type first.

    Returns
    -------
    bool or int or float or complex
        ``factor`` as ``dtype`` reads it: a bool for a bool tensor, an int for an integer
        tensor, a bool as 0 or 1 elsewhere, a complex number's real part on a real tensor.

    Raises
    ------
    RuntimeError
        ``factor`` has an imaginary part and ``dtype`` is not complex, or it has no value
        in the element type of ``dtype`` (NaN or a float past its width on an integer
        tensor, a finite number past the largest value of a floating one).
    """
    if by_type and (isinstance(factor, bool) or _promotion.is_integral(dtype)):
        return factor
    return _element(dtype, read).read(factor)


def check_pow_scalar(dtype: _C_engine.Dtype, exponent: object) -> None:
    """Refuse a Python scalar exponent a ``dtype`` power cannot take.

    A negative Python integer is refused for a bool or integer base, as the reference refuses
    it; a tensor exponent never reaches here, and answers the value (``1 ** -n`` is 1,
    ``(-1) ** -n`` is +-1, the rest 0) on both streams.  An integer exponent is an int64 first
    and then the base's own type, so one past int64 is refused as ``int64_t`` whatever the
    base, and one past an integer base's width as that base.  A half base reads the exponent as
    its storage type; float32 and float64 bases take any exponent.

    Parameters
    ----------
    dtype : _C_engine.Dtype
        Base tensor engine dtype.
    exponent : object
        Candidate scalar exponent.

    Raises
    ------
    RuntimeError
        A negative integer exponent on an integral base, or one the base cannot hold.
    """
    exponent = _promotion.as_number(exponent)
    if isinstance(exponent, bool) or not isinstance(exponent, (int, float)):
        return
    _STORAGE[dtype.value].check_exponent(exponent)
