r"""The frozen factors of PSOFT's principal subspace, from the SVD of the weight.

PSOFT (Wu et al., 2025, arXiv:2505.11235, section 4.1) splits :math:`W = U\Sigma V^\top` into the
principal part of its top :math:`r` singular triplets and the residual, and keeps the principal part
as two factors with the trained matrix :math:`C` between them (Eq. 3 and 6).  The reference folds the
scale of the update, :math:`s = \alpha / r`, into the factors (they hold :math:`\Sigma / s`) and
multiplies the update by :math:`s` again, so ``scaling`` cancels in the function: what the update is
does not depend on ``psoft_alpha``.

A singular pair :math:`(u_k, v_k)` is only defined up to a common sign, and the trained :math:`C`
is not invariant to it (:math:`C \to DCD` for the sign matrix :math:`D`).  A backend's SVD may
choose differently on each device, so the decomposition is always run on the CPU, whatever device
the weight is on: an adapter trained on Metal then means the same thing when it is loaded on the CPU.
"""

from collections.abc import Iterator
from contextlib import contextmanager

import lucid
import lucid.linalg as linalg
from lucid._tensor.tensor import Tensor
from lucid.peft.tuners._spectral import _svd_lowrank
from lucid.peft.tuners.psoft._config import PsoftConfig


@contextmanager
def _seeded(seed: int) -> Iterator[None]:
    """Draw from the default generator started at ``seed``, and leave it where it was found."""
    state = lucid.get_rng_state()
    lucid.manual_seed(seed)
    try:
        yield
    finally:
        lucid.set_rng_state(state)


def _top_triplets(weight: Tensor, config: PsoftConfig) -> tuple[Tensor, Tensor, Tensor]:
    """``(U_r, S_r, Vh_r)`` of ``weight``: ``(out, r)``, ``(r,)``, ``(r, in)``.

    The full SVD, or the randomised one (``psoft_svd="lowrank"``) drawn from ``random_seed`` so
    the same weight and seed always give the same factors.
    """
    rank = config.r
    if config.psoft_svd == "full":
        u, s, vh = linalg.svd(weight, full_matrices=False)
        return u[:, :rank], s[:rank], vh[:rank]
    with _seeded(config.random_seed):
        u, s, v = _svd_lowrank(weight, rank, config.psoft_svd_lowrank_niter)
    return u[:, :rank], s[:rank], v.mT[:rank]


def _principal_factors(
    weight: Tensor, config: PsoftConfig, scaling: float
) -> tuple[Tensor, Tensor]:
    """``(A, B)``, ``(r, in)`` and ``(out, r)``, with ``scaling * B A`` the principal part of ``weight``.

    ``"psoft_init"``: ``A = V_r^T`` has orthonormal rows (so the guarantee of the paper's Theorem 4.1
    holds with a plain orthogonal ``R``) and ``B = U_r Sigma_r / s``.  ``"pissa_init"``: both carry
    ``sqrt(Sigma_r / s)``, the symmetric split of Eq. 3.
    """
    with lucid.no_grad():
        u, s, vh = _top_triplets(weight.detach().to("cpu"), config)
        s = s / scaling
        if config.ab_svd_init == "psoft_init":
            input_side, output_side = vh, u * s
        else:
            root = s.sqrt()
            input_side, output_side = root.unsqueeze(1) * vh, u * root
        return input_side.contiguous().to(weight.device), output_side.contiguous().to(weight.device)
