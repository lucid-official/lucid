"""Spectral helpers the tuners share: decompositions of a weight that a method may need."""

import lucid
import lucid.linalg as linalg
from lucid._tensor.tensor import Tensor


def _svd_lowrank(matrix: Tensor, rank: int, niter: int) -> tuple[Tensor, Tensor, Tensor]:
    r"""Randomised rank-``rank`` SVD (Halko et al.) with ``niter`` power iterations.

    Returns ``(U, S, V)`` with :math:`M \approx U\,\mathrm{diag}(S)\,V^\top`.
    The basis is found for the tall orientation of ``matrix``; no
    oversampling, so ``niter = 0`` is a coarse sketch and a few iterations
    approach the exact truncation.
    """
    flipped = matrix.shape[0] < matrix.shape[1]
    tall = matrix.mT if flipped else matrix
    sketch = lucid.randn(tall.shape[1], rank, device=tall.device, dtype=tall.dtype)
    basis, _ = linalg.qr(tall @ sketch)
    for _ in range(niter):
        basis, _ = linalg.qr(tall.mT @ basis)
        basis, _ = linalg.qr(tall @ basis)
    small_u, s, small_vh = linalg.svd(basis.mT @ tall, full_matrices=False)
    u, v = basis @ small_u, small_vh.mT
    return (v, s, u) if flipped else (u, s, v)
