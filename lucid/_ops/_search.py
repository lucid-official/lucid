"""``searchsorted`` and ``bucketize``: the sorted-sequence search adapters.

The sequence and the values are searched at one dtype, the reference's rule for
a pair of operands (:func:`lucid._promotion.result_type`): a bool beside an
integer, an integer beside a float, an ``int32`` beside an ``int64`` are each
promoted to their common dtype, where the engine's kernels want the two alike
and raised ``DtypeMismatch`` (LCD-1021).  A Python number as the values is a
0-d constant on the sequence's device, which gives a 0-d result, as the
reference's does.

Both refuse a bool sequence searched by bool values, as the reference's CPU
does (a kernel it has no bool instantiation of -- LCD-1013,
:mod:`lucid._ops._bool_refusal`).  The pair is judged before the promotion: a
bool beside another dtype is promoted and answered, not refused.
"""

from typing import TYPE_CHECKING

from lucid import _promotion
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _unwrap_or_scalar
from lucid._ops._bool_refusal import refuse_common
from lucid._types import Scalar

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

_Impl = _C_engine.TensorImpl


def _search_operands(
    op: str, sequence: Tensor | _Impl, values: Tensor | _Impl | Scalar
) -> tuple[_Impl, _Impl]:
    """``sequence`` and ``values`` as engine operands of one dtype.

    Parameters
    ----------
    op : str
        The calling op's name, a key of :data:`lucid._ops._bool_refusal.BOOL_REFUSED`.
    sequence : Tensor or TensorImpl
        The sorted boundaries; always a tensor.
    values : Tensor or TensorImpl or bool or int or float
        What is searched for.  A Python number takes ``sequence``'s device and
        the dtype it is weakly beside it (:func:`lucid._dispatch._unwrap_or_scalar`).

    Returns
    -------
    tuple[TensorImpl, TensorImpl]
        ``(sequence, values)`` cast to their common dtype.

    Raises
    ------
    TypeError
        If ``sequence`` is not a tensor, or ``values`` is neither a tensor nor a number.
    RuntimeError
        If both are bool (the class :func:`lucid._ops._bool_refusal.bool_refusal` builds).
    """
    sequence_impl = _unwrap(sequence)
    values_impl = _unwrap_or_scalar(values, sequence_impl)
    refuse_common(op, sequence_impl, values_impl)
    dtype = _promotion.result_type(sequence_impl, values_impl)
    return (
        sequence_impl if sequence_impl.dtype == dtype else _C_engine.astype(sequence_impl, dtype),
        values_impl if values_impl.dtype == dtype else _C_engine.astype(values_impl, dtype),
    )


def _searchsorted_adapter(
    sorted_sequence: Tensor,
    input: Tensor | Scalar,
    *,
    out_int32: bool = False,
    right: bool = False,
    side: str | None = None,
    sorter: Tensor | None = None,
) -> _Impl:
    """searchsorted(sorted_seq, values, *, out_int32=False, right=False, side=None, sorter=None)."""
    if side is not None:
        if side not in ("left", "right"):
            raise RuntimeError(f"searchsorted(): side can only be 'left' or 'right', got {side!r}")
        if right and side == "left":
            raise RuntimeError(
                "searchsorted(): side and right can't be set to opposites, got side of left while right was True"
            )
        right = side == "right"
    seq_impl, values_impl = _search_operands("searchsorted", sorted_sequence, input)
    if sorter is not None:
        s_impl = _unwrap(sorter)
        if s_impl.dtype != _C_engine.I64 or list(s_impl.shape) != list(seq_impl.shape):
            raise RuntimeError(
                "searchsorted: sorter must be int64 dtype and match sorted_sequence shape"
            )
        seq_impl = _C_engine.gather(seq_impl, s_impl, len(seq_impl.shape) - 1)
    out = _C_engine.searchsorted(seq_impl, values_impl, bool(right))
    return _C_engine.astype(out, _C_engine.I32) if out_int32 else out


def _bucketize_adapter(
    input: Tensor | Scalar, boundaries: Tensor, *, out_int32: bool = False, right: bool = False
) -> _Impl:
    """bucketize(values, boundaries, *, out_int32=False, right=False)."""
    boundaries_impl, values_impl = _search_operands("bucketize", boundaries, input)
    out = _C_engine.bucketize(values_impl, boundaries_impl, bool(right))
    return _C_engine.astype(out, _C_engine.I32) if out_int32 else out
