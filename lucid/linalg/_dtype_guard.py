"""Dtype rules of the two-operand solvers, in one place.

``solve`` (and ``solve_ex`` / ``tensorsolve`` through it) and
``householder_product`` refuse operands of different dtypes, as the reference
does; ``solve_triangular`` reads the right-hand side in the matrix's dtype and
answers in it.  The engine's kernels are single-dtype, and used to read the
second operand as the first's, narrowing a ``float64`` right-hand side
silently.
"""

from lucid._C import engine as _C_engine
from lucid._tensor.tensor import Tensor


def _require_same_dtype(op: str, a: Tensor, b: Tensor, names: tuple[str, str] = ("A", "B")) -> None:
    """Refuse ``a`` and ``b`` of different dtypes, in ``op``'s name."""
    if a.dtype != b.dtype:
        raise _C_engine.DtypeMismatch(
            f"{op}: expected {names[0]} and {names[1]} to have the same dtype, "
            f"but found {a.dtype} and {b.dtype}"
        )
