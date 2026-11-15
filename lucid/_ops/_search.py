"""``searchsorted`` and ``bucketize``: the sorted-sequence search adapters.

Both refuse a bool sequence searched by bool values, as the reference's CPU
does (a kernel it has no bool instantiation of -- LCD-1013,
:mod:`lucid._ops._bool_refusal`); a sequence and values that differ in dtype
are the engine's to reconcile, and it reconciles them as it always did.
"""

from typing import TYPE_CHECKING

from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap
from lucid._ops._bool_refusal import refuse_all_bool

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

_Impl = _C_engine.TensorImpl


def _searchsorted_adapter(
    sorted_sequence: Tensor,
    input: Tensor,
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
    seq_impl = _unwrap(sorted_sequence)
    refuse_all_bool("searchsorted", seq_impl, input)
    if sorter is not None:
        s_impl = _unwrap(sorter)
        if s_impl.dtype != _C_engine.I64 or list(s_impl.shape) != list(seq_impl.shape):
            raise RuntimeError(
                "searchsorted: sorter must be int64 dtype and match sorted_sequence shape"
            )
        seq_impl = _C_engine.gather(seq_impl, s_impl, len(seq_impl.shape) - 1)
    out = _C_engine.searchsorted(seq_impl, _unwrap(input), bool(right))
    return _C_engine.astype(out, _C_engine.I32) if out_int32 else out


def _bucketize_adapter(
    input: Tensor, boundaries: Tensor, *, out_int32: bool = False, right: bool = False
) -> _Impl:
    """bucketize(values, boundaries, *, out_int32=False, right=False)."""
    values_impl, boundaries_impl = _unwrap(input), _unwrap(boundaries)
    refuse_all_bool("bucketize", values_impl, boundaries_impl)
    out = _C_engine.bucketize(values_impl, boundaries_impl, bool(right))
    return _C_engine.astype(out, _C_engine.I32) if out_int32 else out
