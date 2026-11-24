"""The dtype rule of ``ldl_solve``'s packed factor.

``solve`` (and ``solve_ex`` / ``tensorsolve`` through it) and ``householder_product``
refuse operands of different dtypes, as the reference does, through
:func:`lucid._ops._bool_refusal.refuse_unlike`, which owns that rule for every op that
refuses a pair; ``solve_triangular`` reads the right-hand side in the matrix's dtype
and answers in it.
"""

from lucid._ops._bool_refusal import refuse
from lucid._tensor.tensor import Tensor


def _check_ldl_factor(factor: Tensor) -> None:
    """Refuse a packed LDL factor ``ldl_solve`` cannot read: an integral one, or not square."""
    refuse("ldl_solve", factor)
    if factor.ndim < 2 or factor.shape[-1] != factor.shape[-2]:
        raise ValueError(
            f"ldl_solve: LD must be a square matrix or a batch of them, got "
            f"shape {tuple(factor.shape)}"
        )
