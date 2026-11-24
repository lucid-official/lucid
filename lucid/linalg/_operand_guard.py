"""Which operands the linear-system solvers refuse, in the reference's words.

``solve``, ``solve_ex`` and ``lstsq`` read a 1-D right-hand side as one vector.
``solve_triangular``, ``lu_solve`` and ``ldl_solve`` do not: the reference takes a
matrix ``B`` of at least two dimensions there and refuses anything else, so a
vector has to be given its column (``b.unsqueeze(-1)``) first.  The engine's
right-hand-side contract is the wider one ``solve`` needs, so a public wrapper
that is narrower says so here, before the engine reads the operands.

A refusal is a ``ShapeMismatch`` — a ``RuntimeError`` and a ``ValueError`` — with
the reference's message; ``ldl_solve``'s is an ``IndexError``, which is what the
reference raises there.
"""

from lucid._C import engine as _C_engine
from lucid._dtype import int8, int16, int32, int64
from lucid._tensor._indexing import broadcast_shapes
from lucid._tensor.tensor import Tensor
from lucid.linalg._dtype_guard import _check_ldl_factor


def _require_matrix(op: str, name: str, t: Tensor) -> None:
    """Refuse ``t`` of fewer than two dimensions."""
    if t.ndim < 2:
        raise _C_engine.ShapeMismatch(
            f"{op}: The input tensor {name} must have at least 2 dimensions."
        )


def _require_broadcastable(matrix_batch: tuple[int, ...], rhs_batch: tuple[int, ...]) -> None:
    """Refuse batch shapes that do not broadcast, naming the last axis that disagrees.

    The reference words it as tensor ``a`` = the right-hand side and tensor ``b`` = the
    matrix, and counts the axis in the broadcast batch.
    """
    ndim = max(len(matrix_batch), len(rhs_batch))
    matrix_pad, rhs_pad = ndim - len(matrix_batch), ndim - len(rhs_batch)
    for axis in range(ndim - 1, -1, -1):
        matrix_size = matrix_batch[axis - matrix_pad] if axis >= matrix_pad else 1
        rhs_size = rhs_batch[axis - rhs_pad] if axis >= rhs_pad else 1
        if matrix_size != rhs_size and matrix_size != 1 and rhs_size != 1:
            raise _C_engine.ShapeMismatch(
                f"The size of tensor a ({rhs_size}) must match the size of tensor b "
                f"({matrix_size}) at non-singleton dimension {axis}"
            )


def _require_triangular_system(op: str, a: Tensor, b: Tensor, *, left: bool) -> None:
    """Refuse the operands of ``solve_triangular(a, b, left=left)`` the reference refuses.

    Checked in the reference's order: the matrix's rank, then that it is square, the
    right-hand side's rank, whether its rows (columns, for ``left=False``) fit the
    matrix, and last whether the batch dimensions broadcast.
    """
    _require_matrix(op, "A", a)
    n, cols = int(a.shape[-2]), int(a.shape[-1])
    if n != cols:
        raise _C_engine.ShapeMismatch(
            f"{op}: A must be batches of square matrices, but they are {n} by {cols} matrices"
        )
    _require_matrix(op, "B", b)
    b_rows, b_cols = int(b.shape[-2]), int(b.shape[-1])
    if (b_rows if left else b_cols) != n:
        equation = "AX = B" if left else "XA = B"
        raise _C_engine.ShapeMismatch(
            f"{op}: Incompatible shapes of A and B for the equation {equation} "
            f"({n}x{n} and {b_rows}x{b_cols})"
        )
    _require_broadcastable(tuple(a.shape[:-2]), tuple(b.shape[:-2]))


def _require_ldl_rhs(b: Tensor) -> None:
    """Refuse the right-hand side ``ldl_solve`` cannot read, as the reference's index error."""
    if b.ndim == 0:
        raise _C_engine.IndexError("Dimension specified as -2 but tensor has no dimensions")
    if b.ndim == 1:
        raise _C_engine.IndexError(
            "Dimension out of range (expected to be in range of [-1, 0], but got -2)"
        )


def _check_ldl_solve_operands(ld: Tensor, pivots: Tensor, b: Tensor) -> tuple[int, ...]:
    """Refuse the operands ``ldl_solve`` cannot read; the batch ``ld`` and ``b`` broadcast to."""
    _check_ldl_factor(ld)
    if pivots.dtype not in (int8, int16, int32, int64):
        raise TypeError(
            f"ldl_solve: pivots must be an integer tensor, got {pivots.dtype}; "
            f"pass the vector returned by ldl_factor"
        )
    if tuple(pivots.shape) != tuple(ld.shape[:-1]):
        raise ValueError(
            f"ldl_solve: pivots must have shape {tuple(ld.shape[:-1])} to match "
            f"LD of shape {tuple(ld.shape)}, got {tuple(pivots.shape)}"
        )
    _require_ldl_rhs(b)
    if int(b.shape[-2]) != int(ld.shape[-1]):
        raise ValueError(
            f"ldl_solve: B must have {ld.shape[-1]} rows to match LD of shape "
            f"{tuple(ld.shape)}, got shape {tuple(b.shape)}"
        )
    try:
        return tuple(broadcast_shapes(tuple(ld.shape[:-2]), tuple(b.shape[:-2])))
    except ValueError:
        raise ValueError(
            f"ldl_solve: the batch dimensions of LD {tuple(ld.shape)} and B "
            f"{tuple(b.shape)} do not broadcast"
        ) from None
