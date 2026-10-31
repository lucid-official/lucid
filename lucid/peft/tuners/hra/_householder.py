r"""The product of Householder reflections HRA multiplies a weight by.

A reflection :math:`H = I - 2\hat u\hat u^\top` is orthogonal, symmetric and its
own inverse, so a chain of them is orthogonal and its inverse is the chain in
reverse order.  Two forms give the same kind of matrix: the chain itself, or —
after Gram-Schmidt has made the vectors orthonormal — the single product
:math:`I - 2UU^\top`.
"""

import lucid
import lucid.linalg as linalg
from lucid._tensor.tensor import Tensor


class _Householder:
    """Reflection chains of the columns of a matrix; stateless."""

    @staticmethod
    def orthonormal(vectors: Tensor) -> Tensor:
        """The columns of ``vectors`` orthonormalised by Gram-Schmidt, in order (each against the ones before it)."""
        columns: list[Tensor] = []
        for i in range(vectors.shape[1]):
            column = vectors[:, i : i + 1]
            for earlier in columns:
                column = column - (earlier.mT @ column) * earlier
            columns.append(column / linalg.vector_norm(column))
        return lucid.cat(columns, dim=1)

    @classmethod
    def product(cls, vectors: Tensor, *, gram_schmidt: bool, reverse: bool = False) -> Tensor:
        r"""The ``(d, d)`` orthogonal matrix :math:`H_1 H_2 \cdots H_r` of the ``(d, r)`` ``vectors``' columns.

        Parameters
        ----------
        vectors : Tensor
            ``(d, r)``; column :math:`i` is :math:`u_i`.
        gram_schmidt : bool
            Orthonormalise the columns first and return :math:`I - 2UU^\top`.
        reverse : bool, default=False
            Multiply the reflections in the opposite order, which gives the inverse
            (the Gram-Schmidt form is its own inverse already, so this changes nothing there).
        """
        eye = lucid.eye(vectors.shape[0], dtype=vectors.dtype, device=vectors.device)
        if gram_schmidt:
            basis = cls.orthonormal(vectors)
            return eye - 2.0 * (basis @ basis.mT)
        unit = vectors / linalg.vector_norm(vectors, dim=0)
        order = range(vectors.shape[1] - 1, -1, -1) if reverse else range(vectors.shape[1])
        product = eye
        for i in order:
            column = unit[:, i : i + 1]
            product = product - 2.0 * ((product @ column) @ column.mT)
        return product
