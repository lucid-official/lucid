"""Where a ``linalg`` entry judges an operand's dtype before the engine would.

``solve`` (and ``solve_ex`` / ``tensorsolve`` through it) and ``householder_product`` refuse
operands of different dtypes, as the reference does, through
:func:`lucid._ops._bool_refusal.refuse_unlike`, which owns that rule for every op that
refuses a pair; ``solve_triangular`` reads the right-hand side in the matrix's dtype
and answers in it.

An entry judges a dtype itself where the engine's own refusal would come too late, not at
all, or in another class: ``solve`` reads ``A``'s dtype before any shape, as the reference does,
and ``inv`` answers an empty matrix without calling the engine.  The rule stays the table's
(:func:`lucid._ops._bool_refusal.refuse_integral`); these helpers only say *when* it is asked.
``ldl_solve``'s packed factor is refused here too.
"""

from lucid._ops._bool_refusal import refuse, refuse_integral, refuse_unlike
from lucid._tensor.tensor import Tensor


def _check_solve_operands(a: Tensor, b: Tensor) -> None:
    """Refuse the operands of ``solve`` in the reference's order: ``A``'s dtype, then the pair.

    Both come before any shape check, so an integral ``A`` of the wrong shape is a dtype error
    here as it is in the reference.
    """
    refuse_integral("solve", a)
    refuse_unlike("linalg.solve", a, b)


def _inv_operand(a: Tensor) -> Tensor:
    """``a``, once ``inv``'s refusal of a bool or integer matrix has been raised for it.

    A square matrix is judged here, empty or not: an empty one is its own inverse and never
    reaches the engine, and the engine words a bool as a kernel's ``NotImplementedError`` (its
    schema declares the integers, not bool) where the reference raises a dtype guard.  A matrix
    that is not square is left to the engine, which reads a shape before a dtype, as the
    reference does.
    """
    if len(a.shape) >= 2 and a.shape[-1] == a.shape[-2]:
        refuse_integral("inv", a)
    return a


def _check_ldl_factor(factor: Tensor) -> None:
    """Refuse a packed LDL factor ``ldl_solve`` cannot read: an integral one, or not square."""
    refuse("ldl_solve", factor)
    if factor.ndim < 2 or factor.shape[-1] != factor.shape[-2]:
        raise ValueError(
            f"ldl_solve: LD must be a square matrix or a batch of them, got "
            f"shape {tuple(factor.shape)}"
        )
