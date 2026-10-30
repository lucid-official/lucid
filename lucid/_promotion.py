"""The one owner of binary dtype promotion.

Every spelling of a binary op — ``a + b``, ``lucid.add``, ``Tensor.add``,
the in-place forms, the comparisons, the bitwise and shift operators,
``where`` / ``cat`` / ``einsum``, and the public ``result_type`` /
``promote_types`` / ``can_cast`` — takes its result dtype from here.  This is
the only module under ``lucid/`` allowed to hold a dtype kind/width table;
``test_promotion_single_source`` fails if another one appears.

The rule is the reference framework's.  Operands fall into three tiers:

1. tensors with at least one dimension,
2. 0-d tensors,
3. Python numbers.

Within a tier the dtypes meet on the lattice :func:`promote_types`.  Across
tiers a lower tier can raise the result's *category* (bool < integer <
floating < complex) but never its width inside a category, so
``float16_tensor * tensor(0.5)`` stays float16 and ``int8_tensor + 1`` stays
int8, while ``int32_tensor + 2.5`` becomes floating.

A Python ``bool`` is bool, an ``int`` is int64, a ``float`` is the default
dtype and a ``complex`` the complex dtype of the default (complex64, or
complex128 under a float64 default).  Off the CPU stream, which cannot hold
float64, a float64 default yields float32 / complex64 instead.

One documented deviation: the reference promotes float16 against a complex
operand to complex32, which Lucid lacks; Lucid answers complex64.

The module is a leaf — it imports only the engine and ``lucid._globals`` —
so the dispatch layer, the dunders and the op adapters
can all import it without a cycle.  It decides dtypes only; casting the
operands and building scalar operands is :mod:`lucid._dispatch`'s job.
"""

import math
import numbers
from typing import overload

from lucid._C import engine as _C_engine
from lucid._globals import get_default_dtype

_D = _C_engine.Dtype

#: Category of each dtype; a larger category wins a mixed-category pair.
BOOL, INTEGER, FLOATING, COMPLEX = 0, 1, 2, 3

# (category, bit width) for every engine dtype — the single promotion table.
_KIND_WIDTH: dict[_C_engine.Dtype, tuple[int, int]] = {
    _D.Bool: (BOOL, 1),
    _D.I8: (INTEGER, 8),
    _D.I16: (INTEGER, 16),
    _D.I32: (INTEGER, 32),
    _D.I64: (INTEGER, 64),
    _D.F16: (FLOATING, 16),
    _D.BF16: (FLOATING, 16),
    _D.F32: (FLOATING, 32),
    _D.F64: (FLOATING, 64),
    _D.C64: (COMPLEX, 64),
    _D.C128: (COMPLEX, 128),
}

#: The half-precision floating dtypes, the ones mul/div/floordiv op-math is for.
HALF: frozenset[_C_engine.Dtype] = frozenset({_D.F16, _D.BF16})

#: A Python number, the third tier.
NUMBER = (bool, int, float, complex)

Operand = _C_engine.TensorImpl | bool | int | float | complex

# The table again, keyed by each dtype's integer code.  An engine enum hashes
# and compares through C++ (150-250 ns a time) and this runs on every binary
# op, so the rules below work on ints and turn the answer back into a dtype
# once.
_BY_CODE: dict[int, tuple[int, int]] = {d.value: kw for d, kw in _KIND_WIDTH.items()}
_DTYPE: dict[int, _C_engine.Dtype] = {d.value: d for d in _KIND_WIDTH}
_BOOL_C, _I64_C, _F16_C, _F32_C, _F64_C = (
    _D.Bool.value,
    _D.I64.value,
    _D.F16.value,
    _D.F32.value,
    _D.F64.value,
)
_C64_C, _C128_C = _D.C64.value, _D.C128.value
_HALF_C = frozenset(d.value for d in HALF)
_F16_MAX = 65504.0
_BF16_MAX = float.fromhex("0x1.fep+127")
_NUMBER_KIND: dict[type, int] = {bool: BOOL, int: INTEGER, float: FLOATING, complex: COMPLEX}
_CPU = _C_engine.Device.CPU


def bit_width(d: _C_engine.Dtype) -> int:
    """The width of one element of ``d`` in bits, read from the one table.

    Used to wrap a Python integer to an integer dtype's width exactly.
    """
    return _BY_CODE[d.value][1]


def is_integral(d: _C_engine.Dtype) -> bool:
    """Whether ``d`` is bool or an integer dtype — the dtypes that hold no
    fraction, which true division moves to the default float."""
    return _BY_CODE[d.value][0] <= INTEGER


def is_integer(d: _C_engine.Dtype) -> bool:
    """Whether ``d`` is an integer dtype, bool excluded — the dtypes a Python
    integer can be written into exactly."""
    return _BY_CODE[d.value][0] == INTEGER


def is_half(d: _C_engine.Dtype) -> bool:
    """Whether ``d`` is float16 or bfloat16 — the dtypes the engine's
    mul/div/floordiv op-math computes in float32 for."""
    return d.value in _HALF_C


def can_cast(da: _C_engine.Dtype, db: _C_engine.Dtype) -> bool:
    """Whether ``da`` can be cast to ``db`` under the reference's rule.

    Parameters
    ----------
    da : _C_engine.Dtype
        Source engine dtype.
    db : _C_engine.Dtype
        Destination engine dtype.
    """
    return _BY_CODE[da.value][0] <= _BY_CODE[db.value][0]


@overload
def as_number(x: None) -> None: ...
@overload
def as_number[T](x: T) -> T: ...
def as_number(x: object) -> object:
    """``x`` as the Python number it holds when it is another library's scalar.

    A NumPy scalar registers with :mod:`numbers` (``float32`` as Real,
    ``int64`` as Integral) without deriving from the builtin, and ``bool_``
    registers with neither, so they reached the operand checks as neither a
    tensor nor a number: arithmetic raised and ``==`` handed the comparison
    to NumPy, which answered with an ndarray.  It is a weak scalar, as the
    reference takes it.  Anything else — a Python number, a tensor, ``None``
    — comes back as is.
    """
    if type(x) in _NUMBER_KIND:
        return x
    if isinstance(x, bool) or (
        type(x).__name__ in ("bool", "bool_") and type(x).__module__ == "numpy"
    ):
        return bool(x)
    if isinstance(x, numbers.Integral):
        return int(x)
    if isinstance(x, numbers.Real):
        return float(x)
    if isinstance(x, numbers.Complex):
        return complex(x)
    return x


_IMPL = _C_engine.TensorImpl
_NON_TENSOR_TYPES: frozenset[type] = frozenset(
    {int, float, bool, complex, str, list, tuple, dict, slice, type(None)}
)


def try_unwrap(x: object) -> _C_engine.TensorImpl | None:
    """Return the underlying ``TensorImpl`` from a ``Tensor`` or ``TensorImpl``, or ``None``."""
    if type(x) in _NON_TENSOR_TYPES:
        return None
    impl: object = getattr(x, "_impl", x)
    return impl if type(impl) is _IMPL or isinstance(impl, _IMPL) else None


def is_comparable(other: object) -> bool:
    """Whether ``other`` is a Tensor or a number that ``==`` / ``!=`` compares element-wise."""
    return isinstance(as_number(other), NUMBER) or try_unwrap(other) is not None


def check_pow_scalar(d: _C_engine.Dtype, exponent: object) -> None:
    """Refuse a Python scalar exponent a ``d`` power cannot take.

    A negative Python integer is refused for a bool or integer base, as the
    reference refuses it; a tensor exponent never reaches here, and answers
    the value (``1 ** -n`` is 1, ``(-1) ** -n`` is +-1, the rest 0) on both
    streams.  A finite exponent that overflows ``d`` is refused too.

    Parameters
    ----------
    d : _C_engine.Dtype
        Base tensor engine dtype.
    exponent : object
        Candidate scalar exponent.
    """
    exponent = as_number(exponent)
    if isinstance(exponent, bool) or not isinstance(exponent, (int, float)):
        return
    c = d.value
    if isinstance(exponent, int) and exponent < 0 and is_integral(d):
        raise RuntimeError("Integers to negative integer powers are not allowed.")
    if is_integer(d):
        if isinstance(exponent, int):
            bits = _BY_CODE[c][1]
            if not -(1 << (bits - 1)) <= exponent < (1 << (bits - 1)):
                raise RuntimeError(f"value cannot be converted to type {d.name} without overflow")
        return
    if c in _HALF_C:
        val = float(exponent)
        limit = _F16_MAX if c == _F16_C else _BF16_MAX
        if math.isfinite(val) and abs(val) > limit:
            raise RuntimeError(f"value cannot be converted to type {d.name} without overflow")


def _complex_code(c: int) -> int:
    return _C128_C if c == _F64_C else _C64_C


def _promote(a: int, b: int) -> int:
    if a == b:
        return a
    ka, wa = _BY_CODE[a]
    kb, wb = _BY_CODE[b]
    if ka == COMPLEX and kb != COMPLEX:
        return _C128_C if b == _F64_C else a
    if kb == COMPLEX and ka != COMPLEX:
        return _C128_C if a == _F64_C else b
    if ka != kb:
        return a if ka > kb else b
    if wa == wb:
        return _F32_C
    return a if wa > wb else b


def promote_types(da: _C_engine.Dtype, db: _C_engine.Dtype) -> _C_engine.Dtype:
    """The lattice join of two dtypes, tiers aside.

    Parameters
    ----------
    da, db : lucid._C.engine.Dtype
        The two dtypes.

    Returns
    -------
    lucid._C.engine.Dtype
        The higher category's dtype; within one category the wider one.
        Two complex dtypes give the wider, a complex beside float64 gives
        complex128, and float16 beside bfloat16 — neither holds the other —
        gives float32.
    """
    return _DTYPE[_promote(da.value, db.value)]


def _default_code(device: _C_engine.Device | None) -> int:
    c = get_default_dtype()._engine.value
    if c == _F64_C and device is not None and device != _CPU:
        return _F32_C
    return c


def default_float(device: _C_engine.Device) -> _C_engine.Dtype:
    """The default floating dtype, narrowed to what ``device`` can hold.

    The GPU stream cannot hold float64, so a float64 default answers
    float32 there.
    """
    return _DTYPE[_default_code(device)]


def _number_kind(x: bool | int | float | complex) -> int:
    kind = _NUMBER_KIND.get(type(x))
    if kind is not None:
        return kind
    # A subclass — NumPy's float64 derives from float, its bool_ from nothing.
    if isinstance(x, bool):
        return BOOL
    if isinstance(x, int):
        return INTEGER
    return FLOATING if isinstance(x, float) else COMPLEX


def _number_code_for_default(kind: int, float_code: int) -> int:
    if kind <= INTEGER:
        return _BOOL_C if kind == BOOL else _I64_C
    return float_code if kind == FLOATING else _complex_code(float_code)


def _number_code(kind: int, holder: _C_engine.TensorImpl | None) -> int:
    """The code of a number of ``kind`` beside ``holder`` (a tensor or None).

    Reads ``holder.device`` only when a float64 default must be narrowed.
    """
    if kind <= INTEGER:
        return _BOOL_C if kind == BOOL else _I64_C
    c = get_default_dtype()._engine.value
    if c == _F64_C and holder is not None and holder.device != _CPU:
        c = _F32_C
    return _number_code_for_default(kind, c)


def number_dtype(
    x: bool | int | float | complex, default_float_dtype: _C_engine.Dtype
) -> _C_engine.Dtype:
    """The engine dtype a Python number ``x`` asks for under ``default_float_dtype``.

    Parameters
    ----------
    x : bool or int or float or complex
        The Python scalar value.
    default_float_dtype : lucid._C.engine.Dtype
        Current default floating dtype (``F32`` or ``F64``).
    """
    return _DTYPE[_number_code_for_default(_number_kind(x), default_float_dtype.value)]


def literal_types_dtype(
    types: set[type] | frozenset[type], default_float_dtype: _C_engine.Dtype
) -> _C_engine.Dtype:
    """The engine dtype for a non-empty set of builtin Python number ``types``.

    Parameters
    ----------
    types : set of type or frozenset of type
        Builtin Python scalar types (``bool``, ``int``, ``float``, ``complex``).
    default_float_dtype : lucid._C.engine.Dtype
        Current default floating dtype (``F32`` or ``F64``).
    """
    kind = max(_NUMBER_KIND[t] for t in types)
    return _DTYPE[_number_code_for_default(kind, default_float_dtype.value)]


def _lifts(kh: int, kl: int) -> bool:
    """Whether a lower tier of category ``kl`` changes a higher of ``kh``.

    The lower tier can lift the category only: a complex lower makes a real
    higher complex; a floating lower meets an integral higher on the
    lattice, as does any non-bool beside a bool higher.
    """
    return kl > kh


def _join(higher: int, lower: int) -> int:
    """Fold a lower tier's dtype code into a higher tier's."""
    kh = _BY_CODE[higher][0]
    if not _lifts(kh, _BY_CODE[lower][0]):
        return higher
    if kh == FLOATING:  # lifted by a complex lower: keep the precision
        return _complex_code(higher)
    return _promote(higher, lower)


def _combine(higher: int | None, lower: int | None) -> int | None:
    if higher is None or lower is None:
        return lower if higher is None else higher
    return _join(higher, lower)


def beside_number(
    t: _C_engine.TensorImpl, d: _C_engine.Dtype, x: bool | int | float | complex
) -> _C_engine.Dtype:
    """``result_type(t, x)`` for one tensor and one number.

    The common binary case — ``x * 0.5``, ``i + 1`` — answered without the
    general walk.

    Parameters
    ----------
    t : TensorImpl
        The tensor; its ``device`` is read only to narrow a float64 default.
    d : lucid._C.engine.Dtype
        ``t``'s dtype, already read by the caller.
    x : bool or int or float or complex
        The number.

    Returns
    -------
    lucid._C.engine.Dtype
        ``d`` itself, the same object, when the number does not lift it, so
        the caller can tell with ``is`` that ``t`` needs no cast.
    """
    c = d.value
    kl = _number_kind(x)
    if not _lifts(_BY_CODE[c][0], kl):
        return d
    return _DTYPE[_join(c, _number_code(kl, t))]


def tensor_pair_type(
    a: _C_engine.TensorImpl, da: _C_engine.Dtype, b: _C_engine.TensorImpl, db: _C_engine.Dtype
) -> _C_engine.Dtype:
    """``result_type(a, b)`` for two tensors whose dtypes the caller has read.

    Parameters
    ----------
    a, b : TensorImpl
        The tensors; only their ``shape`` is read, for the tier.
    da, db : lucid._C.engine.Dtype
        Their dtypes, read once by the binary hot path rather than again here.

    Returns
    -------
    lucid._C.engine.Dtype
        The dtype the pair computes in.
    """
    ca, cb = da.value, db.value
    a_dim, b_dim = bool(a.shape), bool(b.shape)
    if a_dim == b_dim:
        return _DTYPE[_promote(ca, cb)]
    return _DTYPE[_join(ca, cb) if a_dim else _join(cb, ca)]


def result_type(*operands: Operand) -> _C_engine.Dtype:
    """The dtype a binary or n-ary op over ``operands`` computes in.

    Parameters
    ----------
    *operands : TensorImpl or bool or int or float or complex
        Engine tensors and Python numbers, in any order.  A tensor is read
        through ``dtype``, ``shape`` and ``device`` only.

    Returns
    -------
    lucid._C.engine.Dtype
        ``combine(dimensioned, combine(zero_dim, numbers))``, each tier the
        lattice join of its members.

    Raises
    ------
    TypeError
        If no operand is given.
    """
    dim: int | None = None
    zero: int | None = None
    first: _C_engine.TensorImpl | None = None
    kind = -1
    for x in operands:
        if isinstance(x := as_number(x), NUMBER):
            kind = max(kind, _number_kind(x))
            continue
        c = x.dtype.value
        if first is None:
            first = x
        if x.shape:
            dim = c if dim is None else _promote(dim, c)
        else:
            zero = c if zero is None else _promote(zero, c)
    # The numbers' own join is the code of the highest kind among them:
    # bool < int64 < the default float < its complex counterpart.
    wrapped = None if kind < 0 else _number_code(kind, first)
    out = _combine(dim, _combine(zero, wrapped))
    if out is None:
        raise TypeError("result_type: expected at least one operand")
    return _DTYPE[out]
