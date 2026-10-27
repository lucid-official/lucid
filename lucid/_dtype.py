"""Lucid scalar dtype descriptors and dtype-info helpers.

This module defines the :class:`dtype` class together with the canonical
singleton instances (``float32``, ``int64``, ``bool_`` …) that parameterize
every tensor operation. It also exposes :class:`finfo` / :class:`iinfo` for
querying numerical limits of floating / integer dtypes, mirroring the
NumPy API.
"""

from typing import override

from lucid._C import engine as _C_engine


class dtype:
    """Lucid scalar data type descriptor.

    Represents the element type of a tensor (``float32``, ``int64``, ``bool``
    …). Instances are module-level singletons compared by identity (``is``);
    do not instantiate :class:`dtype` directly — use the predefined
    constants exported by :mod:`lucid`.

    Attributes
    ----------
    _name : str
        Canonical type name (e.g. ``"float32"``).
    _engine : lucid._C.engine.Dtype
        C++ engine enum value for this dtype.
    itemsize : int
        Size of one element in bytes.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.zeros(3, dtype=lucid.float32)
    >>> x.dtype
    lucid.float32
    >>> x.dtype is lucid.float32
    True
    """

    def __init__(self, name: str, engine_dtype: _C_engine.Dtype, itemsize: int) -> None:
        """Initialize a dtype singleton.

        Parameters
        ----------
        name : str
            Canonical type name (e.g. ``"float32"``).
        engine_dtype : lucid._C.engine.Dtype
            Corresponding C++ engine enum value.
        itemsize : int
            Element size in bytes.
        """
        self._name = name
        self._engine = engine_dtype
        self.itemsize = itemsize if itemsize else _C_engine.dtype_size(engine_dtype)

    @override
    def __repr__(self) -> str:
        """Return ``"lucid.<name>"`` (e.g. ``"lucid.float32"``)."""
        return f"lucid.{self._name}"

    @override
    def __eq__(self, other: object) -> bool:
        """Identity comparison with engine-dtype fallback."""
        if self is other:
            return True
        if isinstance(other, dtype):
            return self._engine == other._engine
        return False

    @override
    def __hash__(self) -> int:
        """Hash by canonical name so dtypes can key dicts/sets across pickle."""
        return hash(self._name)

    @override
    def __reduce__(self) -> str:
        """Pickle by module-level singleton name so identity is preserved."""
        return "bool_" if self._name == "bool" else self._name


float16: dtype = dtype("float16", _C_engine.Dtype.F16, 2)
float32: dtype = dtype("float32", _C_engine.Dtype.F32, 4)
float64: dtype = dtype("float64", _C_engine.Dtype.F64, 8)
# Its own engine dtype, not an alias for float16.  bfloat16 carries
# float32's exponent range in sixteen bits — that range is the entire
# reason to choose it — so pointing it at F16 meant code that picked
# bfloat16 specifically to avoid overflow got the format that overflows,
# and ``lucid.tensor(..., dtype=lucid.bfloat16).dtype`` answered
# ``lucid.float16``.
bfloat16: dtype = dtype("bfloat16", _C_engine.Dtype.BF16, 2)
int8: dtype = dtype("int8", _C_engine.Dtype.I8, 1)
int16: dtype = dtype("int16", _C_engine.Dtype.I16, 2)
int32: dtype = dtype("int32", _C_engine.Dtype.I32, 4)
int64: dtype = dtype("int64", _C_engine.Dtype.I64, 8)
bool_: dtype = dtype("bool", _C_engine.Dtype.Bool, 1)
complex64: dtype = dtype("complex64", _C_engine.Dtype.C64, 8)
complex128: dtype = dtype("complex128", _C_engine.Dtype.C128, 16)

half: dtype = float16
double: dtype = float64
short: dtype = int16
long: dtype = int64


def _float32_or_wider(d: dtype) -> dtype:
    """The dtype position and time tables are computed in for a ``d`` model.

    Sinusoid angles reach a thousand radians and resampling coordinates
    hundreds of pixels; float16 and bfloat16 cannot hold either to a
    useful fraction.  Such tables are built in float32 — the reference's
    accumulation type for half activations — or float64 for a float64
    model, then cast once to the activations' dtype.  Unlike the loss and
    reduction rules, which widen half only, every non-float64 input maps to
    float32: an integer timestep gets a float32 table.

    Examples
    --------
    >>> import lucid
    >>> from lucid._dtype import _float32_or_wider
    >>> _float32_or_wider(lucid.float16), _float32_or_wider(lucid.float64)
    (lucid.float32, lucid.float64)
    """
    return float64 if d == float64 else float32


_ENGINE_TO_DTYPE: dict[_C_engine.Dtype, dtype] = {
    _C_engine.Dtype.F16: float16,
    _C_engine.Dtype.BF16: bfloat16,
    _C_engine.Dtype.F32: float32,
    _C_engine.Dtype.F64: float64,
    _C_engine.Dtype.I8: int8,
    _C_engine.Dtype.I16: int16,
    _C_engine.Dtype.I32: int32,
    _C_engine.Dtype.I64: int64,
    _C_engine.Dtype.Bool: bool_,
    _C_engine.Dtype.C64: complex64,
    _C_engine.Dtype.C128: complex128,
}

_NAME_TO_DTYPE: dict[str, dtype] = {
    "float16": float16,
    "float32": float32,
    "float64": float64,
    "bfloat16": bfloat16,
    "int8": int8,
    "int16": int16,
    "int32": int32,
    "int64": int64,
    "bool": bool_,
    "bool_": bool_,
    "complex64": complex64,
    "complex128": complex128,
    "cfloat": complex64,
    "cdouble": complex128,
    "half": float16,
    "float": float32,
    "double": float64,
    "short": int16,
    "int": int32,
    "long": int64,
}


def to_engine_dtype(d: dtype | type[dtype] | _C_engine.Dtype | str | None) -> _C_engine.Dtype:
    """Coerce any accepted dtype-spec to the C++ engine's ``Dtype`` enum.

    Front door for every Python → engine call site that needs to pass
    a dtype.  Accepts the union of forms users / internal code throws
    at the framework: a :class:`lucid.dtype` instance, a raw engine
    enum value, a string name (``"float32"``), or ``None`` for the
    framework default.  Rejects bare ``dtype`` classes with a clear
    error to catch ``lucid.dtype`` (the type) being passed instead of
    ``lucid.float32`` (an instance).

    Parameters
    ----------
    d : dtype, dtype-class, engine.Dtype, str, or None
        Requested dtype.

    Returns
    -------
    engine.Dtype
        The corresponding C++ enum value, consumable by every engine
        factory and kernel call.

    Raises
    ------
    TypeError
        If ``d`` is a dtype *class* (e.g. ``lucid.dtype``) rather than
        an instance.
    ValueError
        If ``d`` is a string that doesn't name a known dtype.
    """
    if d is None:
        return _C_engine.Dtype.F32
    if isinstance(d, _C_engine.Dtype):
        return d
    if isinstance(d, dtype):
        return d._engine
    # dtype subclass or uninstantiated dtype class — e.g. lucid.float32 (already
    # an instance), but also handle the rare case where a type object is passed.
    if isinstance(d, type) and issubclass(d, dtype):
        raise TypeError(
            f"Pass a dtype instance, not the class {d!r}. "
            "Example: lucid.float32, not lucid.dtype"
        )
    if isinstance(d, str):
        name = d.lower()
        if name in _NAME_TO_DTYPE:
            return _NAME_TO_DTYPE[name]._engine
        raise ValueError(f"Unknown dtype string: {d!r}")
    raise TypeError(f"Cannot convert {type(d).__name__} to engine Dtype")


# ── finfo / iinfo ────────────────────────────────────────────────────────────


import dataclasses as _dc

# Alias so that `dtype` (the class) is accessible inside finfo/iinfo even
# after the dataclass field `dtype: str` shadows the outer name.
_DtypeCls = dtype


def _resolve_dtype_name(d: _DtypeCls | type[_DtypeCls] | _C_engine.Dtype | str) -> str:
    """Map any dtype-like input to its canonical lucid dtype name.

    Used by ``finfo`` / ``iinfo`` to discriminate ``float16`` from
    ``bfloat16`` (both share engine enum ``F16``) and to produce the
    user-facing ``dtype: str`` field on the info objects.
    """
    if isinstance(d, dtype):
        return d._name
    if isinstance(d, str):
        name = d.lower()
        if name in _NAME_TO_DTYPE:
            return _NAME_TO_DTYPE[name]._name
        raise ValueError(f"Unknown dtype string: {d!r}")
    # Engine Dtype enum — best-effort reverse lookup (loses bfloat16/float16
    # distinction; both report ``float16``).
    if isinstance(d, _C_engine.Dtype):
        for lucid_dt in _NAME_TO_DTYPE.values():
            if lucid_dt._engine == d:
                return lucid_dt._name
    raise TypeError(f"Cannot resolve dtype name for {type(d).__name__}: {d!r}")


@_dc.dataclass(frozen=True)
class finfo:
    """Floating-point dtype info — mirrors ``numpy.finfo`` (and the reference framework's ``finfo``).

    Constructed from a floating dtype (``float16`` / ``bfloat16`` /
    ``float32`` / ``float64``).
    """

    bits: int = 0
    eps: float = 0.0
    max: float = 0.0
    min: float = 0.0
    tiny: float = 0.0
    smallest_normal: float = 0.0
    resolution: float = 0.0
    dtype: str = ""

    def __init__(self, dt: _DtypeCls | type[_DtypeCls] | _C_engine.Dtype | str) -> None:
        """Populate floating-point limits for ``dt``.

        Parameters
        ----------
        dt : dtype or lucid._C.engine.Dtype or str
            Floating dtype to query (``float16`` / ``bfloat16`` / ``float32``
            / ``float64``). Passing an integer / boolean / complex dtype
            raises :class:`TypeError`.

        Raises
        ------
        TypeError
            If ``dt`` is not a recognized floating-point dtype.

        Examples
        --------
        >>> import lucid
        >>> lucid.finfo(lucid.float32).eps
        1.1920929e-07
        >>> lucid.finfo(lucid.float32).max
        3.4028235e+38
        """
        # Resolve to a lucid dtype name first — bfloat16 and float16 share the
        # same engine enum (F16), so we can't distinguish them via the engine.
        name = _resolve_dtype_name(dt)
        table: dict[str, tuple[int, float, float, float, float, float]] = {
            "float64": (
                64,
                2.220446049250313e-16,
                1.7976931348623157e308,
                -1.7976931348623157e308,
                2.2250738585072014e-308,
                1e-15,
            ),
            "float32": (32, 1.1920929e-7, 3.4028235e38, -3.4028235e38, 1.1754944e-38, 1e-6),
            "float16": (16, 9.7656e-4, 65504.0, -65504.0, 6.1035e-5, 1e-3),
            "bfloat16": (16, 7.8125e-3, 3.3895314e38, -3.3895314e38, 1.1754944e-38, 1e-2),
        }
        if name not in table:
            raise TypeError(f"finfo expects a floating dtype, got {dt!r}")
        bits, eps, mx, mn, tiny, res = table[name]
        object.__setattr__(self, "bits", bits)
        object.__setattr__(self, "eps", eps)
        object.__setattr__(self, "max", mx)
        object.__setattr__(self, "min", mn)
        object.__setattr__(self, "tiny", tiny)
        object.__setattr__(self, "smallest_normal", tiny)
        object.__setattr__(self, "resolution", res)
        object.__setattr__(self, "dtype", name)


@_dc.dataclass(frozen=True)
class iinfo:
    """Integer dtype info — mirrors ``numpy.iinfo`` (and the reference framework's ``iinfo``).

    Constructed from an integer dtype (``int8`` / ``int16`` / ``int32`` / ``int64``).
    """

    bits: int = 0
    max: int = 0
    min: int = 0
    dtype: str = ""

    def __init__(self, dt: _DtypeCls | type[_DtypeCls] | _C_engine.Dtype | str) -> None:
        """Populate integer limits for ``dt``.

        Parameters
        ----------
        dt : dtype or lucid._C.engine.Dtype or str
            Integer dtype to query (``int8`` / ``int16`` / ``int32`` /
            ``int64``). Passing a non-integer dtype raises
            :class:`TypeError`.

        Raises
        ------
        TypeError
            If ``dt`` is not a recognized integer dtype.

        Examples
        --------
        >>> import lucid
        >>> lucid.iinfo(lucid.int32).max
        2147483647
        >>> lucid.iinfo(lucid.int8).min
        -128
        """
        name = _resolve_dtype_name(dt)
        table: dict[str, tuple[int, int, int]] = {
            "int8": (8, -(2**7), 2**7 - 1),
            "int16": (16, -(2**15), 2**15 - 1),
            "int32": (32, -(2**31), 2**31 - 1),
            "int64": (64, -(2**63), 2**63 - 1),
        }
        if name not in table:
            raise TypeError(f"iinfo expects an integer dtype, got {dt!r}")
        bits, lo, hi = table[name]
        object.__setattr__(self, "bits", bits)
        object.__setattr__(self, "min", lo)
        object.__setattr__(self, "max", hi)
        object.__setattr__(self, "dtype", name)
