"""
Value construction shared by every tensor factory.

A factory owes its caller two things, in this order: the value, held
exactly in the requested dtype on the requested device, and then — as
the very last step — a leaf with the requested ``requires_grad``.

Each half has a way to go quietly wrong.  Setting ``requires_grad``
before a dtype conversion hands back the conversion's *output*, a
non-leaf whose ``.grad`` never fills, so training silently does
nothing.  Routing a value through a C ``double`` (which the engine's
``full`` takes) rounds integers past :math:`2^{53}` and has no room for
an imaginary part.  This module builds values from their own Python
type straight into the target dtype's bytes, and :func:`_leaf` is the
single place a factory's ``requires_grad`` is applied.

Pure Python + the engine: no external imports (H4).
"""

import functools
import math
import struct
from collections.abc import Sequence, Set

from lucid import _promotion
from lucid._C import engine as _C_engine
from lucid._dispatch import _default_dtype_enum_cached, _impl_with_grad, _scalar_impl
from lucid._dtype import _ENGINE_TO_DTYPE

_D = _C_engine.Dtype

type PyScalar = bool | int | float | complex

# struct format per dtype.  bfloat16 has no struct code; it is packed as
# float32 and narrowed by the engine *before* the leaf is made.
_STRUCT_CODE: dict[_C_engine.Dtype, str] = {
    _D.F16: "e",
    _D.F32: "f",
    _D.F64: "d",
    _D.I8: "b",
    _D.I16: "h",
    _D.I32: "i",
    _D.I64: "q",
    _D.Bool: "?",
    _D.C64: "f",
    _D.C128: "d",
}
_COMPLEX = (_D.C64, _D.C128)

# NumPy scalar dtype names → engine dtype.  Unsigned types widen to the
# next signed type, as unsigned arrays do (Lucid has no unsigned dtype).
_NUMPY_SCALAR_DTYPE: dict[str, _C_engine.Dtype] = {
    "bool": _D.Bool,
    "int8": _D.I8,
    "int16": _D.I16,
    "int32": _D.I32,
    "int64": _D.I64,
    "uint8": _D.I16,
    "uint16": _D.I32,
    "uint32": _D.I64,
    "uint64": _D.I64,
    "float16": _D.F16,
    "float32": _D.F32,
    "float64": _D.F64,
    "complex64": _D.C64,
    "complex128": _D.C128,
}

_PY_SCALAR_TYPES = (bool, int, float, complex)
_PY_SCALAR_SET = frozenset(_PY_SCALAR_TYPES)
_PY_REAL_SET = frozenset((bool, int, float))
_PY_INT_SET = frozenset((bool, int))
# Dtypes a plain Python real packs into without per-element coercion:
# ``struct`` takes bool / int / float for every float code as they are.
_FLOATING = frozenset((_D.F16, _D.F32, _D.F64))


def _leaf(impl: _C_engine.TensorImpl, requires_grad: bool) -> _C_engine.TensorImpl:
    r"""The last step of every factory: make ``impl`` a leaf.

    ``impl`` must already hold the final value in the final dtype on the
    final device, built without grad — nothing may convert it afterwards,
    or the caller gets that conversion's output instead of a leaf.

    Raises
    ------
    RuntimeError
        If ``requires_grad`` is set on a bool or integer ``impl``.
    """
    return _impl_with_grad(impl, True) if requires_grad else impl


def _is_numpy_scalar(value: object) -> bool:
    cls = type(value)
    return cls.__module__ == "numpy" and cls.__name__ != "ndarray"


def _is_scalar_leaf(value: object) -> bool:
    r"""Whether ``value`` is an element the pure-Python path can pack."""
    if _is_numpy_scalar(value):
        return str(getattr(value, "dtype", "")) in _NUMPY_SCALAR_DTYPE
    return isinstance(value, _PY_SCALAR_TYPES)


def _python_scalar(value: object) -> PyScalar:
    r"""``value`` as a plain Python scalar, keeping its kind.

    NumPy scalars and one-element tensors expose ``item()``, which
    returns the Python scalar of their own kind (``np.bool_`` → ``bool``,
    ``np.int64`` → ``int``) — unlike ``float(value)``, which would lose
    the integer and refuse the complex.
    """
    value = _promotion.as_number(value)
    if isinstance(value, _PY_SCALAR_TYPES):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        value = _promotion.as_number(item())
        if isinstance(value, _PY_SCALAR_TYPES):
            return value
    raise TypeError(f"expected a number (bool, int, float or complex), got {type(value).__name__}")


def _scalar_dtype(value: object, default_float: _C_engine.Dtype) -> _C_engine.Dtype:
    r"""The dtype a literal element asks for on its own."""
    impl = _promotion.try_unwrap(value)
    if impl is not None:
        return impl.dtype
    if _is_numpy_scalar(value):
        return _NUMPY_SCALAR_DTYPE[str(getattr(value, "dtype", ""))]
    return _promotion.number_dtype(_python_scalar(value), default_float)


def _infer_dtype(flat: Sequence[object]) -> _C_engine.Dtype:
    r"""Dtype of a literal: its elements' dtypes, promoted together.

    Python ``float`` / ``complex`` elements take the default dtype (and
    its complex counterpart); NumPy scalars keep their own.  Mixed
    elements promote as operands of arithmetic do — two different 16-bit
    floats meet at ``float32``, and a ``float64`` beside a complex makes
    it ``complex128``.
    """
    default_float = _default_dtype_enum_cached()
    if not flat:
        return default_float
    types = {type(v) for v in flat}
    if types <= _PY_SCALAR_SET:  # the common literal: plain Python numbers
        return _promotion.literal_types_dtype(types, default_float)
    dtypes = sorted({_scalar_dtype(v, default_float) for v in flat}, key=lambda d: d.value)
    return functools.reduce(_promotion.promote_types, dtypes)


def _real(value: object) -> float:
    v = _python_scalar(value)
    if isinstance(v, complex):
        raise TypeError(f"expected a real number, got the complex value {v!r}")
    return float(v)


def _name(dtype: _C_engine.Dtype) -> str:
    return str(_ENGINE_TO_DTYPE[dtype])


def _coerce(value: object, dtype: _C_engine.Dtype) -> PyScalar:
    r"""``value`` converted exactly into ``dtype``'s Python type.

    Raises at the boundary instead of letting the engine wrap or clamp:
    a complex value into a real dtype is a ``TypeError``, a non-finite
    float into an integer dtype a ``ValueError``, and an integer outside
    the dtype's range an ``OverflowError``.  A finite float truncates
    toward zero, as a cast does.
    """
    v = _python_scalar(value)
    if dtype in _COMPLEX:
        return complex(v)
    if isinstance(v, complex):
        raise TypeError(f"cannot store the complex value {v!r} in a {_name(dtype)} tensor")
    if dtype == _D.Bool:
        return bool(v)
    if not _promotion.is_integer(dtype):
        return float(v)
    bits = _promotion.bit_width(dtype)
    if isinstance(v, float):
        if not math.isfinite(v):
            raise ValueError(f"cannot store {v!r} in an integer ({_name(dtype)}) tensor")
        v = int(v)
    low, high = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    iv = int(v)
    if not low <= iv <= high:
        raise OverflowError(f"{iv} does not fit in {_name(dtype)}, which holds [{low}, {high}]")
    return iv


def _to_inf_on_overflow(x: float, code: str) -> float:
    r"""``x``, or infinity of its sign when ``code`` would round it there.

    ``struct`` refuses a finite value that rounds past the format's
    largest finite number; an IEEE cast turns it into infinity.
    """
    try:
        struct.pack(f"={code}", x)
    except OverflowError:
        return math.copysign(math.inf, x)
    return x


def _struct_parts(
    flat: Sequence[object], dtype: _C_engine.Dtype, types: Set[type]
) -> Sequence[object]:
    r"""``flat`` as the values ``struct`` packs for ``dtype``."""
    if dtype in _COMPLEX:
        zs = [complex(_coerce(v, dtype)) for v in flat]
        return [part for z in zs for part in (z.real, z.imag)]
    if dtype in _FLOATING and types <= _PY_REAL_SET:
        return flat
    return [_coerce(v, dtype) for v in flat]


def _pack(flat: Sequence[object], dtype: _C_engine.Dtype) -> bytes:
    if not flat:
        return b""
    code = _STRUCT_CODE[dtype]
    types = {type(v) for v in flat}
    if _promotion.is_integer(dtype) and types <= _PY_INT_SET:
        try:
            return struct.pack(f"={len(flat)}{code}", *flat)
        except struct.error:
            pass  # out of range: the coercion below names the value and range
    parts = _struct_parts(flat, dtype, types)
    try:
        return struct.pack(f"={len(parts)}{code}", *parts)
    except OverflowError:
        if code not in ("e", "f"):
            raise
        narrowed = [_to_inf_on_overflow(_real(x), code) for x in parts]
        return struct.pack(f"={len(narrowed)}{code}", *narrowed)


def _from_values(
    flat: Sequence[object],
    shape: list[int],
    dtype: _C_engine.Dtype,
    device: _C_engine.Device,
) -> _C_engine.TensorImpl:
    r"""A tensor (not yet a leaf of the caller's choosing) holding ``flat``."""
    if dtype == _D.BF16:
        f32 = _C_engine.TensorImpl.from_bytes(_pack(flat, _D.F32), shape, _D.F32, device, False)
        return _C_engine.astype(f32, _D.BF16)
    return _C_engine.TensorImpl.from_bytes(_pack(flat, dtype), shape, dtype, device, False)


# Below this magnitude a double holds every integer, so the engine's
# double-valued ``full`` is exact.
_EXACT_IN_DOUBLE = 2**53


def _fill(
    shape: list[int],
    value: object,
    dtype: _C_engine.Dtype,
    device: _C_engine.Device,
) -> _C_engine.TensorImpl:
    r"""``shape`` filled with ``value``, exact in ``dtype``.

    The engine fill takes a ``double``; a value it cannot carry — a
    complex, or an integer past :math:`2^{53}` — is built as a 0-d
    tensor of the target dtype via :func:`lucid._dispatch._scalar_impl`
    and broadcast instead.
    """
    value = _promotion.as_number(value)
    if dtype in _FLOATING and isinstance(value, (bool, int, float)):
        return _C_engine.full(shape, float(value), dtype, device)
    v = _coerce(value, dtype)
    if (
        not shape
        or isinstance(v, complex)
        or (isinstance(v, int) and not isinstance(v, bool) and abs(v) > _EXACT_IN_DOUBLE)
    ):
        scalar = _scalar_impl(v, dtype, device)
        return scalar if not shape else _C_engine.contiguous(_C_engine.broadcast_to(scalar, shape))
    return _C_engine.full(shape, float(v), dtype, device)
