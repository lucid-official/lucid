"""The ops that refuse a bool tensor, and who refuses them (LCD-1013, LCD-1020).

The reference's CPU backend turns a bool operand away from a long list of
ops -- a kernel with no bool instantiation raises ``NotImplementedError``, a
guard on the input dtype raises ``RuntimeError`` -- and Lucid computed them
instead, promoting the mask to a number behind the caller's back
(``mask.mean()``, ``mask.abs()``, ``softmax(mask)``).  A bool is a truth
value, not a count or a magnitude, so the program that meant one of those
casts at the call site (``mask.float().mean()``) and the one that did not
learns of it where it happened.

The rule has one owner per layer, and an op has exactly one of them:

* An op with an engine schema (``abs``, ``relu``, ``softmax``, ``mean``, the
  pools and norms, ...) is refused by the engine: its ``OpSchema`` declares
  the refusal and ``SchemaGuard`` raises it for every caller -- the free
  function, the ``Tensor`` method, the in-place twin, ``nn.functional`` and a
  direct engine call alike.  :func:`engine_bool_refused` reads those
  declarations back out of the engine.
* An op with no schema of its own -- a composite built from other ops, or an
  adapter whose answer depends on the form of the call (``pow`` takes a bool
  base to a number, ``clamp`` a single tensor bound) -- is refused by its
  Python entry, through :data:`BOOL_REFUSED`.  Each entry calls
  :func:`refuse_bool` / :func:`bool_refusal` with the op's name, so the error
  class and the message never diverge between a free function, its ``Tensor``
  method and the in-place twin.

Neither list is a census of the ops that refuse.  The bool-refusal sweep
(``lucid/test/unit/ops/test_bool_refusal_sweep.py``) derives its candidates
from the audit's op census and compares Lucid's answer with the reference's
recorded one, and ``lucid/test/unit/ops/test_bool_refusal_table.py`` pins
every row of both lists against the reference.
"""

import functools
from collections.abc import Mapping
from types import MappingProxyType

from lucid import _promotion
from lucid._C import engine as _C_engine

#: The reference's ``NotImplementedError`` -- a kernel with no bool
#: instantiation.  Lucid's class is also a ``RuntimeError``, as the reference's.
_KERNEL = _C_engine.NotImplementedError

#: The reference's plain ``RuntimeError`` -- a guard on the input dtype.  The
#: engine's dtype-refusal class is a ``RuntimeError`` (and a ``TypeError``),
#: and what ``mean(dtype=...)`` already raises for the dtype it refuses.
_GUARD = _C_engine.DtypeMismatch

#: ``Dtype.Bool``'s integer code: ``impl.dtype.value == BOOL_CODE`` is 150 ns
#: cheaper than comparing the enum.
BOOL_CODE: int = _C_engine.Bool.value

# The cast hints.  The engine words the same three in ``OpSchema.h``
# (``bool_cast_hint``); a test holds the two spellings equal.
_FLOATING = "a floating dtype, e.g. x.float()"
_INTEGER = "an integer dtype, e.g. x.long()"
_NUMBER = "an integer or floating dtype, e.g. x.long() or x.float()"

#: op -> (error class the reference raises for a bool operand, what to cast to)
#: for the ops this layer refuses -- the ones with no engine schema to do it.
#: In-place twins (``pow_``) take the row of their out-of-place op, and an
#: alias (``clip``, ``upsample``) the row of the op it forwards to.
BOOL_REFUSED: Mapping[str, tuple[type[RuntimeError], str]] = MappingProxyType(
    {
        # Order statistics and counting over a sorted sequence.
        "quantile": (_GUARD, _FLOATING),
        "nanquantile": (_GUARD, _FLOATING),
        "searchsorted": (_KERNEL, _INTEGER),
        "bucketize": (_KERNEL, _INTEGER),
        "isin": (_GUARD, _INTEGER),
        # Moments.  ``mean``, ``var`` and ``std`` are the engine's.
        "std_mean": (_GUARD, _FLOATING),
        "var_mean": (_GUARD, _FLOATING),
        "nanmean": (_GUARD, _FLOATING),
        "cov": (_GUARD, _NUMBER),
        "corrcoef": (_GUARD, _NUMBER),
        "histogram": (_KERNEL, _FLOATING),
        "gradient": (_GUARD, _NUMBER),
        "poisson": (_KERNEL, _FLOATING),
        # Elementwise.  ``pow`` refuses a bool base only beside a tensor exponent
        # (``mask ** True`` is a number), ``clamp`` a bool result of two bounds.
        "clamp": (_KERNEL, _INTEGER),
        "pow": (_KERNEL, _NUMBER),
        "frexp": (_GUARD, _FLOATING),
        "polar": (_GUARD, _FLOATING),
        # Special functions and linear algebra.
        "gammainc": (_KERNEL, _FLOATING),
        "gammaincc": (_KERNEL, _FLOATING),
        "multigammaln": (_GUARD, _NUMBER),
        "vecdot": (_GUARD, _FLOATING),
        # Activations composed from other ops.
        "celu": (_KERNEL, _FLOATING),
        "hardshrink": (_KERNEL, _FLOATING),
        "softshrink": (_KERNEL, _FLOATING),
        "hardtanh": (_GUARD, _NUMBER),
        "threshold": (_KERNEL, _NUMBER),
        "rrelu": (_KERNEL, _FLOATING),
        "prelu": (_KERNEL, _FLOATING),
        # Resampling.
        "interpolate": (_KERNEL, _FLOATING),
        "grid_sample": (_KERNEL, _FLOATING),
    }
)

_KIND_CLASS: Mapping[str, type[RuntimeError]] = MappingProxyType(
    {"kernel": _KERNEL, "guard": _GUARD}
)


@functools.cache
def engine_bool_refused() -> Mapping[str, tuple[type[RuntimeError], str]]:
    """The ops whose engine schema refuses a bool tensor, in :data:`BOOL_REFUSED`'s shape.

    The engine owns these refusals: ``OpSchema.refusing_bool`` declares one
    and ``SchemaGuard`` raises it for every caller, so no Python entry
    repeats it.  This reads the declarations back out of the registry, for the
    tests that hold the two layers to one rule.  An op the engine refuses
    without registering a schema (``var``, ``frac``) is not in the registry
    and so not in this mapping.

    Returns
    -------
    Mapping[str, tuple[type[RuntimeError], str]]
        ``op -> (error class, cast hint)``: the engine's
        ``NotImplementedError`` for a ``"kernel"`` refusal, ``DtypeMismatch``
        for a ``"guard"`` one.
    """
    refusals: dict[str, tuple[type[RuntimeError], str]] = {}
    for schema in _C_engine.op_registry_all():
        declared = schema.bool_refusal
        if declared is not None:
            kind, hint = declared
            refusals[schema.name] = (_KIND_CLASS[kind], hint)
    return MappingProxyType(refusals)


def bool_refusal(op: str, role: str = "") -> RuntimeError:
    """The error ``op`` raises for a bool operand, for a call site to ``raise``.

    Parameters
    ----------
    op : str
        The name the caller used.  A trailing ``_`` (the in-place twin) reads
        the row of the out-of-place op, and is kept in the message.
    role : str, optional
        Names the operand when it is not the first one (``"bins"``,
        ``"weight"``).

    Returns
    -------
    RuntimeError
        The reference's class for ``op`` (the engine's ``NotImplementedError``
        or ``DtypeMismatch``, both ``RuntimeError`` subclasses), with a message
        that names the op and the cast to make.
    """
    error, cast = BOOL_REFUSED[op.removesuffix("_")]
    what = f"a bool {role}" if role else "bool"
    return error(f"{op} does not accept {what} — cast to {cast}")


def is_bool(x: object) -> bool:
    """Whether ``x`` is a bool tensor, as a ``Tensor`` or as the engine's impl.

    Parameters
    ----------
    x : object
        Any operand.  A Python ``bool`` is a number, not a bool tensor, and so
        are ``None`` and a list.

    Returns
    -------
    bool
        ``True`` only for a tensor whose dtype is bool.
    """
    impl = _promotion.try_unwrap(x)
    return impl is not None and impl.dtype.value == BOOL_CODE


def refuse_bool(op: str, *operands: object, role: str = "") -> None:
    """Raise ``op``'s refusal when any of ``operands`` is a bool tensor.

    Parameters
    ----------
    op : str
        The refusing op's name, a key of :data:`BOOL_REFUSED`.
    *operands : object
        The operands to look at.  Numbers, ``None`` and non-tensors pass: only
        a tensor's dtype can be bool.
    role : str, optional
        Names the operand in the message when it is not the first one.

    Raises
    ------
    RuntimeError
        The class :func:`bool_refusal` builds for ``op``.
    """
    for operand in operands:
        impl = _promotion.try_unwrap(operand)
        if impl is not None and impl.dtype.value == BOOL_CODE:
            raise bool_refusal(op, role)


def refuse_all_bool(op: str, *operands: object) -> None:
    """Raise ``op``'s refusal when *every* one of ``operands`` is a bool tensor.

    For the ops whose operands meet at a common dtype (``searchsorted``,
    ``bucketize``, ``gammainc``): the reference has no kernel for a bool pair,
    and answers a bool beside another dtype by promoting.  Lucid's engine does
    not promote for ``searchsorted`` and ``bucketize`` -- a bool beside an
    integer is a ``DtypeMismatch`` there (LCD-1021) -- so only the pair is
    refused here, and the mixed pair is left to whatever the op does with it.

    Parameters
    ----------
    op : str
        The refusing op's name, a key of :data:`BOOL_REFUSED`.
    *operands : object
        The operands that meet at one dtype.

    Raises
    ------
    RuntimeError
        The class :func:`bool_refusal` builds for ``op``.
    """
    if all(is_bool(operand) for operand in operands):
        raise bool_refusal(op)


__all__ = [
    "BOOL_CODE",
    "BOOL_REFUSED",
    "bool_refusal",
    "engine_bool_refused",
    "is_bool",
    "refuse_all_bool",
    "refuse_bool",
]
