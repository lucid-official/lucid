r"""The SVD split OSF trains in, and the projection of a gradient onto what the split leaves free.

OSF (Nayak et al., 2025, arXiv:2504.07097, section 3.2) takes the SVD of every weight,
:math:`W = U\Sigma V^\top` with the singular values descending, and freezes the top ``effective_rank``
triplets, the *high* subspace that carries what the model already knows.  The rest, the *low*
subspace, is what trains: :math:`W' = U_{high}\Sigma_{high}V_{high}^\top + U_{low}\Sigma_{low}V_{low}^\top`.
Section 3.5 (Eq. 4) keeps the update from leaking into the high subspace by projecting gradients onto
its orthogonal complement; on the factors that train, that is

.. math:: \nabla U_{low} \leftarrow (I - U_{high}U_{high}^\top)\,\nabla U_{low}, \qquad
          \nabla V_{low} \leftarrow \nabla V_{low}\,(I - V_{high}^\top V_{high}) ,

which :class:`_Complement` computes.  The singular vectors are a basis of the complement only up to
a rotation, and a sign: nothing here depends on which one a backend picks, because the update is
the difference of two products :math:`U\Sigma V^\top` and the projections are those of spans.
"""

from typing import NamedTuple

import lucid
import lucid.linalg as linalg
from lucid._tensor.tensor import Tensor
from lucid._types import DTypeLike


class _Split(NamedTuple):
    """A weight's SVD cut after ``k`` triplets: the frozen top and the trainable rest."""

    u_high: Tensor
    s_high: Tensor
    v_high: Tensor
    u_low: Tensor
    s_low: Tensor
    v_low: Tensor

    @classmethod
    def of(cls, weight: Tensor, rank: int, dtype: DTypeLike) -> _Split:
        """The split of ``weight`` after its top ``rank`` triplets, detached and in ``dtype`` (the SVD itself runs in float32)."""
        wide = weight.detach().to(lucid.float64 if dtype == lucid.float64 else lucid.float32)
        u, s, vh = linalg.svd(wide, full_matrices=False)
        cut = min(rank, int(s.shape[0]))
        parts = (u[:, :cut], s[:cut], vh[:cut], u[:, cut:], s[cut:], vh[cut:])
        return cls(*(part.contiguous().to(dtype) for part in parts))


def _resolve_rank(value: int | float | None, shortest: int) -> int:
    """The number of frozen triplets ``value`` asks for in a weight whose smaller side is ``shortest``.

    ``None`` is half of it; a float in ``(0, 1]`` is that fraction; anything else a count.  Never
    more than ``shortest`` and never negative.
    """
    if value is None:
        return max(shortest // 2, 0)
    fraction = isinstance(value, float) and 0 < value <= 1
    return max(min(shortest, int(shortest * value) if fraction else int(value)), 0)


class _Complement:
    """Gradients projected onto the orthogonal complement of the frozen high factors.

    A factor of the high subspace is kept only where the low one cannot stand in for it: the
    left factors of an SVD are square when the weight has no more rows than columns, then
    :math:`I - U_{high}U_{high}^\\top = U_{low}U_{low}^\\top` of the initial low factors, and the
    right ones when it has no more columns than rows.  Every method takes the high factor, or ``None``
    when the initial low factor replaces it.
    """

    @staticmethod
    def left(grad: Tensor, u_init: Tensor, u_high: Tensor | None) -> Tensor:
        """``(I - U_high U_high^T) grad`` for a gradient of the left factor ``(out, m)``."""
        if u_high is not None:
            return grad - u_high @ (u_high.mT @ grad)
        return u_init @ (u_init.mT @ grad)

    @staticmethod
    def right(grad: Tensor, v_init: Tensor, v_high: Tensor | None) -> Tensor:
        """``grad (I - V_high^T V_high)`` for a gradient of the right factor ``(m, in)``."""
        if v_high is not None:
            return grad - (grad @ v_high.mT) @ v_high
        return (grad @ v_init.mT) @ v_init
