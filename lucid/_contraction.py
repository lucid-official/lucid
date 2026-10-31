"""The result dtype of a contraction, in one place.

A contraction is a product summed over an axis the operands share —
``matmul``, ``bmm``, ``einsum("ij,jk->ik")``, the ``mat1 @ mat2`` inside
``addmm``.  The reference accumulates it in the operands' own dtype, so an
``int32`` contraction is ``int32``.  A stand-alone sum is a different
operation: ``sum`` widens every integer dtype (and bool) to ``int64``, and
``einsum("ij->j")`` or a trace is that sum.

The engine builds both out of ``mul`` and ``sum``, so a contraction comes
back as the ``int64`` its ``sum`` produced.  Every composite that
contracts decides the dtype through this module instead of each choosing
its own spelling, which is how ``einsum`` answered ``int64`` where
``matmul`` answered ``int32`` and ``addbmm`` raised on its own batch sum.

A bool contraction is refused: summing a product of booleans has no
boolean meaning, and the reference's contraction kernels reject it.
"""

import functools
from collections.abc import Sequence

from lucid import _promotion
from lucid._C import engine as _C_engine

_Impl = _C_engine.TensorImpl


def require_summable(dtype: _C_engine.Dtype, op: str) -> None:
    """Refuse a contraction over bool operands.

    Parameters
    ----------
    dtype : lucid._C.engine.Dtype
        The dtype the operands meet at.
    op : str
        The caller's name, for the message.

    Raises
    ------
    lucid._C.engine.NotImplementedError
        ``dtype`` is bool.
    """
    if dtype == _C_engine.Bool:
        raise _C_engine.NotImplementedError(
            f'{op}: "contraction" is not implemented for the Bool dtype'
        )


def settle(result: _Impl, operand: _C_engine.Dtype) -> _Impl:
    """Narrow an integer contraction's ``int64`` sum back to the operand dtype.

    The product is exact modulo the operand width either way, so narrowing
    the wide sum gives what an accumulator of the operand's own width would
    have.  Floating and complex results already carry the operand dtype and
    pass through.

    Parameters
    ----------
    result : TensorImpl
        What the engine returned for the contraction.
    operand : lucid._C.engine.Dtype
        The dtype the operands meet at.

    Returns
    -------
    TensorImpl
        ``result``, cast to ``operand`` when both are integer dtypes.
    """
    if (
        _promotion.is_integer(operand)
        and _promotion.is_integer(result.dtype)
        and result.dtype != operand
    ):
        return _C_engine.astype(result, operand)
    return result


def operand_dtype(operands: Sequence[_Impl]) -> _C_engine.Dtype:
    """Find the dtype a contraction over ``operands`` keeps.

    Parameters
    ----------
    operands : sequence of TensorImpl
        The contraction's operands.

    Returns
    -------
    lucid._C.engine.Dtype
        The dtype the operands promote to.
    """
    return functools.reduce(_promotion.promote_types, (op.dtype for op in operands))


def split_sums(equation: str) -> tuple[bool, bool]:
    """Say which kinds of sum an explicit einsum equation performs.

    Parameters
    ----------
    equation : str
        An equation with ``->`` and no ``...`` in which each operand names
        a subscript at most once, as :func:`lucid.einops._einsum.normalise`
        returns it.

    Returns
    -------
    tuple of (bool, bool)
        ``(stand_alone, contracted)``: whether some subscript is summed away
        within a single operand (a ``sum``, which widens an integer), and
        whether some subscript is shared by operands and summed away (a
        contraction, which keeps the operand dtype).
    """
    lhs, rhs = equation.split("->")
    kept = set(rhs)
    uses: dict[str, int] = {}
    for term in lhs.split(","):
        for label in term:
            uses[label] = uses.get(label, 0) + 1
    summed = {label: n for label, n in uses.items() if label not in kept}
    return (any(n == 1 for n in summed.values()), any(n > 1 for n in summed.values()))


def widen(operands: Sequence[_Impl], dtype: _C_engine.Dtype) -> list[_Impl]:
    """Carry integer and bool operands at ``int64``, as the engine's ``sum`` will.

    ``sum`` widens every narrow integer to ``int64``, so a chained contraction
    would meet its own ``int64`` partial result with an ``int32`` operand and
    refuse the pair.  Widening first makes the chain uniform; :func:`settle`
    narrows the answer back.

    Parameters
    ----------
    operands : sequence of TensorImpl
        The contraction's operands.
    dtype : lucid._C.engine.Dtype
        The dtype they meet at.

    Returns
    -------
    list of TensorImpl
        The operands, cast to ``int64`` when ``dtype`` is a narrower integer
        or bool, unchanged otherwise.
    """
    if _promotion.is_integral(dtype) and dtype != _C_engine.I64:
        return [_C_engine.astype(op, _C_engine.I64) for op in operands]
    return list(operands)
