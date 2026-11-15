"""The ops that refuse a bool tensor, in one table (LCD-1013).

The reference's CPU backend turns a bool operand away from a long list of
ops -- a kernel with no bool instantiation raises ``NotImplementedError``, a
guard on the input dtype raises ``RuntimeError`` -- and Lucid computed them
instead, promoting the mask to a number behind the caller's back
(``mask.mean()``, ``mask.abs()``, ``softmax(mask)``).  A bool is a truth
value, not a count or a magnitude, so the program that meant one of those
casts at the call site (``mask.float().mean()``) and the one that did not
learns of it where it happened.

This module is the one owner of that rule.  :data:`BOOL_REFUSED` names every
op that refuses bool and the class the reference raises for it; the op's
entry points (registry entry, adapter, composite, ``nn.functional``) each
call :func:`refuse_bool` / :func:`bool_refusal` with the op's name, so the
error class and the message never diverge between a free function, its
``Tensor`` method and the in-place twin.  The test table
``lucid/test/unit/ops/test_bool_refusal_table.py`` pins every row against the
reference and reads this table to prove no refusing op is missing from it.
"""

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

#: ``Dtype.Bool``'s integer code.  The entries of the hot unary ops (``abs``,
#: ``relu``) test ``impl.dtype.value == BOOL_CODE``, which is 150 ns cheaper
#: than comparing the enum, on calls that cost about a microsecond.
BOOL_CODE: int = _C_engine.Bool.value

_FLOATING = "a floating dtype, e.g. x.float()"
_INTEGER = "an integer dtype, e.g. x.long()"
_NUMBER = "an integer or floating dtype, e.g. x.long() or x.float()"

#: op -> (error class the reference raises for a bool operand, what to cast to).
#: In-place twins (``abs_``) take the row of their out-of-place op, and an alias
#: (``fix``, ``absolute``, ``clip``) the row of the op it forwards to.
BOOL_REFUSED: Mapping[str, tuple[type[RuntimeError], str]] = MappingProxyType(
    {
        # Order statistics and counting over a sorted sequence.
        "quantile": (_GUARD, _FLOATING),
        "nanquantile": (_GUARD, _FLOATING),
        "searchsorted": (_KERNEL, _INTEGER),
        "bucketize": (_KERNEL, _INTEGER),
        "isin": (_GUARD, _INTEGER),
        "clamp": (_KERNEL, _INTEGER),
        # Moments.
        "mean": (_GUARD, _FLOATING),
        "std": (_GUARD, _FLOATING),
        "var": (_GUARD, _FLOATING),
        "std_mean": (_GUARD, _FLOATING),
        "var_mean": (_GUARD, _FLOATING),
        "nanmean": (_GUARD, _FLOATING),
        "cov": (_GUARD, _NUMBER),
        "corrcoef": (_GUARD, _NUMBER),
        "histogram": (_KERNEL, _FLOATING),
        "gradient": (_GUARD, _NUMBER),
        "poisson": (_KERNEL, _FLOATING),
        # Elementwise.
        "abs": (_KERNEL, _NUMBER),
        "ceil": (_KERNEL, _NUMBER),
        "floor": (_KERNEL, _NUMBER),
        "round": (_KERNEL, _NUMBER),
        "trunc": (_KERNEL, _NUMBER),
        "frac": (_KERNEL, _FLOATING),
        "frexp": (_GUARD, _FLOATING),
        "polar": (_GUARD, _FLOATING),
        # Activations.
        "relu": (_GUARD, _NUMBER),
        "selu": (_KERNEL, _FLOATING),
        "softmax": (_KERNEL, _FLOATING),
        "log_softmax": (_KERNEL, _FLOATING),
    }
)


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
        if is_bool(operand):
            raise bool_refusal(op, role)


def refuse_all_bool(op: str, *operands: object) -> None:
    """Raise ``op``'s refusal when *every* one of ``operands`` is a bool tensor.

    For the ops whose operands meet at a common dtype (``searchsorted``,
    ``bucketize``): the reference has no kernel for a bool pair, and answers
    a bool beside any other dtype by promoting.

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


__all__ = ["BOOL_CODE", "BOOL_REFUSED", "bool_refusal", "is_bool", "refuse_all_bool", "refuse_bool"]
