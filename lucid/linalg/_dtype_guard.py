"""Where a ``linalg`` entry judges an operand's dtype before the engine would.

``solve`` (and ``solve_ex`` / ``tensorsolve`` through it) and ``householder_product`` refuse
operands of different dtypes, as the reference does, through
:func:`lucid._ops._bool_refusal.refuse_unlike`, which owns that rule for every op that
refuses a pair; ``solve_triangular`` reads the right-hand side in the matrix's dtype
and answers in it.

An entry judges a dtype itself where the engine's own refusal would come too late or not at
all: ``solve`` reads ``A``'s dtype before any shape, as the reference does, and ``inv`` answers
an empty matrix without calling the engine.  The rule stays the table's
(:func:`lucid._ops._bool_refusal.refuse_integral`); these helpers only say *when* it is asked.
``ldl_solve``'s packed factor is refused here too.
"""

from lucid._C import engine as _C_engine
from lucid._dispatch import _wrap
from lucid._ops._bool_refusal import refuse, refuse_integral, refuse_unlike
from lucid._tensor.tensor import Tensor


def _check_solve_operands(a: Tensor, b: Tensor) -> None:
    """Refuse the operands of ``solve`` in the reference's order: ``A``'s dtype, then the pair.

    Both come before any shape check, so an integral ``A`` of the wrong shape is a dtype error
    here as it is in the reference.
    """
    refuse_integral("solve", a)
    refuse_unlike("linalg.solve", a, b)


def _inverse_of_empty(a: Tensor) -> Tensor:
    """``a`` itself, the inverse of a matrix with no entry, if it is of a dtype ``inv`` takes.

    Inversion preserves shape and there are no elements to factorise, so the input *is* the
    answer -- of a floating matrix.  An integral one is what the engine refuses and the
    reference refuses for an empty one too; the engine is not asked here, so this is.
    """
    refuse_integral("inv", a)
    return a


def _checked_operand(op: str, impl: _C_engine.TensorImpl) -> Tensor:
    """``impl`` as a ``Tensor``, once ``op``'s row has turned its dtype away if it does.

    For an op whose routes do not all reach the engine's own refusal: ``pinv`` and ``matrix_rank``
    send a ``hermitian`` input through ``eigh``, whose refusal is a kernel's where the reference's
    for these two is a dtype guard.
    """
    refuse(op, impl)
    return _wrap(impl)


def _check_ldl_factor(factor: Tensor) -> None:
    """Refuse a packed LDL factor ``ldl_solve`` cannot read: an integral one, or not square."""
    refuse("ldl_solve", factor)
    if factor.ndim < 2 or factor.shape[-1] != factor.shape[-2]:
        raise ValueError(
            f"ldl_solve: LD must be a square matrix or a batch of them, got "
            f"shape {tuple(factor.shape)}"
        )
