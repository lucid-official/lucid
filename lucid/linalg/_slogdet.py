"""``slogdet``: the sign and ``log|det A|`` of a matrix, from its LU diagonal.

The determinant is never formed, so it cannot overflow or underflow on the way
to its logarithm, and the gradient is ``A⁻ᵀ`` rather than ``cof(A) / det(A)``.
Re-exported by :mod:`lucid.linalg`, which owns the factorisation it reads.
"""

from typing import NamedTuple, final, override

import lucid
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap
from lucid._tensor.tensor import Tensor
from lucid.autograd.function import Function, FunctionCtx
from lucid.linalg import _la, det, diagonal, inv
from lucid.linalg._lu import _lu_factor_packed


class _SlogdetResult(NamedTuple):
    sign: Tensor
    logabsdet: Tensor


def slogdet(A: Tensor) -> tuple[Tensor, Tensor]:
    r"""Sign and natural log of the absolute determinant.

    Returns the pair :math:`(\mathrm{sign},\, \log|\det A|)` such that

    .. math::

        \det(A) \,=\, \mathrm{sign} \cdot \exp(\log|\det A|).

    Stays correct where :math:`\det A` itself leaves the floating-point
    range — large or small covariance matrices in log-likelihood
    calculations, or any matrix whose determinant would overflow to
    :math:`\infty` or underflow to :math:`0`.

    Parameters
    ----------
    A : Tensor
        Square matrix of shape ``(*, n, n)``.

    Returns
    -------
    sign : Tensor
        :math:`\pm 1` (or :math:`0` for singular matrices), shape
        ``(*,)``.
    logabsdet : Tensor
        :math:`\log|\det A|`, shape ``(*,)``.  Returns :math:`-\infty`
        for singular inputs.

    Notes
    -----
    Computed from the LU factorisation :math:`PA = LU` as
    :math:`\log|\det A| = \sum_i \log|U_{ii}|`, with the sign the product
    of the signs of :math:`U_{ii}` and of the permutation :math:`P` — the
    determinant itself is never formed, so nothing here can overflow.  Cost
    is :math:`O(n^3)`.  A zero pivot makes the matrix singular: the sign is
    the signed zero :math:`\det A` would be, and ``logabsdet`` is
    :math:`-\infty`.

    The gradient of ``logabsdet`` is :math:`A^{-\top}`, taken from an
    inverse of :math:`A` rather than as :math:`\mathrm{cof}(A)/\det(A)`, so
    it stays finite wherever the matrix is invertible.  At a singular matrix
    it is :math:`\mathrm{cof}(A)/\det(A)` — :math:`\pm\infty` along the
    adjugate of a matrix of rank :math:`n-1`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import slogdet
    >>> sign, logabs = slogdet(lucid.tensor([[1.0, 2.0], [3.0, 4.0]]))
    >>> sign, logabs
    (tensor(-1.), tensor(0.6931))
    """
    n = int(A.shape[-1]) if A.ndim else 0
    if A.ndim < 2 or int(A.shape[-2]) != n or A.dtype not in (lucid.float32, lucid.float64):
        # Not a float square matrix: ``det`` owns the refusal and its type.
        _la.det(_unwrap(A))
    if A.numel() == 0:
        # The empty product: det = 1, so (sign, logabsdet) = (1, 0).
        batch = tuple(int(s) for s in A.shape[:-2])
        return _SlogdetResult(
            lucid.ones(batch, dtype=A.dtype, device=A.device),
            lucid.zeros(batch, dtype=A.dtype, device=A.device),
        )
    # det(A) = det(Aᵀ); the reference factors the transpose, and which zero a
    # singular matrix ends on — ``+0`` or ``-0`` — depends on that choice.
    LU, pivots, info = _lu_factor_packed(A.detach().mT)
    pivot = diagonal(LU)
    swaps = (pivots != lucid.arange(1, n + 1, dtype=pivots.dtype, device=A.device)).sum(dim=-1)
    parity = (1 - 2 * (swaps % 2)).to(A.dtype)
    # A zero pivot signs as a literal ``+0``, so a singular matrix's sign is
    # the permutation's sign on a zero — the signed zero the reference gives.
    sign = parity * pivot.sign().prod(dim=-1)
    logabs = lucid.log(pivot.abs()).sum(dim=-1)
    singular = info != 0
    d: Tensor | None = None
    if _C_engine.compile.current_tracer() is not None or bool(singular.any()):
        # A zero pivot has no inverse for the gradient to use; ``det`` is where
        # its cofactor form lives.  Every other matrix of the batch enters it
        # as the identity, so it adds nothing there.
        d = det(lucid.where(_matrix_mask(singular), A, _batched_eye(A)))
    out = _SlogdetFromLU.apply(A, d, sign, logabs, singular)
    assert isinstance(out, tuple)  # two outputs, so ``apply`` returns the tuple
    sign, logabs = out
    return _SlogdetResult(sign, logabs)


def _batched_eye(A: Tensor) -> Tensor:
    """The identity of ``A``'s matrix size, broadcast over its batch."""
    n = int(A.shape[-1])
    return lucid.eye(n, dtype=A.dtype, device=A.device).expand(*A.shape[:-2], n, n)


def _matrix_mask(per_matrix: Tensor) -> Tensor:
    """A ``(*,)`` mask spread over the trailing matrix axes."""
    return per_matrix.unsqueeze(-1).unsqueeze(-1)


@final
class _SlogdetFromLU(Function):
    """``(sign, log|det A|)`` read off an LU factorisation, differentiable in ``A``.

    ``log|det A|`` is the sum of ``log|U_ii|`` — finite whenever the
    determinant is not, which ``log(abs(det(A)))`` is not.  Its gradient is
    ``A⁻ᵀ``, an inverse, where ``cof(A) / det(A)``'s cofactor overflows exactly
    where this exists.  The sign has no derivative but rides on the same node,
    so both outputs are tracked as the reference's are.

    A matrix with a zero pivot has no inverse.  It is flagged ``singular``,
    handed the identity in the inverse's place, and its gradient comes through
    ``d = det(A)`` instead: ``g / d`` into ``det``'s own backward, which is
    ``g · cof(A) / det(A)`` — ``±inf`` along the adjugate of a matrix of rank
    ``n - 1``, the sign of the infinity that of the signed zero ``d`` is.
    """

    @override
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        A: Tensor,
        d: Tensor | None,
        sign: Tensor,
        logabs: Tensor,
        singular: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if d is None:
            ctx.save_for_backward(A, singular)
            return sign, logabs
        ctx.save_for_backward(A, singular, d)
        return sign, lucid.where(singular, lucid.log(d.abs()), logabs)

    @override
    @staticmethod
    def backward(
        ctx: FunctionCtx, grad_sign: Tensor, grad_logabs: Tensor
    ) -> tuple[Tensor, Tensor | None, None, None, None]:
        del grad_sign  # the sign is piecewise constant
        A, singular, *rest = ctx.saved_tensors
        mask = _matrix_mask(singular)
        invertible = lucid.where(mask, _batched_eye(A), A)
        grad_A = _matrix_mask(grad_logabs) * inv(invertible).mT
        grad_d = lucid.where(singular, grad_logabs / rest[0], 0.0) if rest else None
        return grad_A, grad_d, None, None, None
