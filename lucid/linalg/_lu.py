"""The LU family: packed factor, its unpacking, and the explicit ``lu``.

``getrf`` factors any ``m × n`` matrix, so everything here does — the
factors, the permutation, and the gradient.  Re-exported by
:mod:`lucid.linalg`, which owns the triangular solves and the permutation
builder this reads.
"""

from typing import NamedTuple, final, override

import lucid
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid._tensor.tensor import Tensor
from lucid.autograd.function import Function, FunctionCtx
from lucid.linalg import _build_permutation_matrix, _check_pivots, _la, solve_triangular


class _LuResult(NamedTuple):
    P: Tensor
    L: Tensor
    U: Tensor


def _lu_factor_packed(A: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """``getrf``'s packed factor (differentiable in ``A``), pivots and ``info``.

    A singular matrix is factored and reported, never raised; whether that is
    an error is the caller's call.
    """
    lu_data, pivots, info = _la.lu_factor(_unwrap(A))
    LU, piv = _wrap(lu_data), _wrap(pivots)
    if A.requires_grad and _C_engine.grad_enabled():
        P = _build_permutation_matrix(piv, int(A.shape[-2]), A.dtype, A.device)
        graphed = _LUFactorGrad.apply(A, LU, P)
        assert isinstance(graphed, Tensor)
        LU = graphed
    return LU, piv, _wrap(info)


def _lu_solve_packed(LU: Tensor, pivots: Tensor, B: Tensor) -> Tensor:
    """``getrs`` on a packed factor and pivots :func:`_lu_factor_packed` returned.

    Not :func:`lucid.linalg.lu_solve`: those pivots are the engine's own, so the
    range check it runs on a caller's would read them back to the host for nothing.
    """
    return _wrap(_la.lu_solve(_unwrap(LU), _unwrap(pivots), _unwrap(B)))


def _square_lu_backward(L: Tensor, U: Tensor, gL: Tensor, gU: Tensor) -> Tensor:
    """``L⁻ᵀ · (tril(Lᵀ gL, -1) + triu(gU Uᵀ)) · U⁻ᵀ`` for square, triangular ``L``, ``U``."""
    M = lucid.tril(lucid.matmul(L.mT, gL), -1) + lucid.triu(lucid.matmul(gU, U.mT))
    X = solve_triangular(U.mT, M, upper=False, left=False)
    return solve_triangular(L.mT, X, upper=True, unitriangular=True)


@final
class _LUFactorGrad(Function):
    """The packed factor of ``A = P L U``, differentiable in ``A``.

    The kernel recorded no gradient, so a loss through ``lu_factor`` — or
    through ``lu``, ``det`` written by hand from it — trained nothing.  With
    ``gL = tril(G, -1)`` and ``gU = triu(G)`` read off the packed gradient,
    the reference framework's backward for a square ``A`` is::

        dA = P · L⁻ᵀ · (tril(Lᵀ gL, -1) + triu(gU Uᵀ)) · U⁻ᵀ

    A rectangular ``A`` splits the square factorisation off and carries the
    rest.  Wide (``m < n``, ``U = [U₁ | U₂]``): ``U₂ = L⁻¹ A₂``, so
    ``dA₂ = L⁻ᵀ gU₂`` and ``L`` also receives ``-dA₂ U₂ᵀ``; the square rule
    runs on ``(L, U₁)``.  Tall (``m > n``, ``L = [L₁; L₂]``): ``L₂ = A₂ U⁻¹``,
    so ``dA₂ = gL₂ U⁻ᵀ`` and ``U`` also receives ``-L₂ᵀ dA₂``; the square rule
    runs on ``(L₁, U)``.

    The triangular solves go through the differentiable solve, and ``L`` and
    ``U`` are read from this Function's own output, so the result
    differentiates again.
    """

    @override
    @staticmethod
    def forward(ctx: FunctionCtx, A: Tensor, LU: Tensor, P: Tensor) -> Tensor:
        out = LU.detach()
        ctx.save_for_backward(out, P)
        return out

    @override
    @staticmethod
    def backward(ctx: FunctionCtx, grad_out: Tensor) -> tuple[Tensor, None, None]:
        LU, P = ctx.saved_tensors
        m, n = int(LU.shape[-2]), int(LU.shape[-1])
        k = min(m, n)
        eye = lucid.eye(m, k, dtype=LU.dtype, device=LU.device)
        L = lucid.tril(LU[..., :, :k], -1) + eye
        U = lucid.triu(LU[..., :k, :])
        gL = lucid.tril(grad_out[..., :, :k], -1)
        gU = lucid.triu(grad_out[..., :k, :])
        if m == n:
            dA = _square_lu_backward(L, U, gL, gU)
        elif m < n:
            U2 = U[..., :, k:]
            dA2 = solve_triangular(L.mT, gU[..., :, k:], upper=True, unitriangular=True)
            dA1 = _square_lu_backward(L, U[..., :, :k], gL - dA2 @ U2.mT, gU[..., :, :k])
            dA = lucid.cat([dA1, dA2], dim=-1)
        else:
            L2 = L[..., k:, :]
            dA2 = solve_triangular(U.mT, gL[..., k:, :], upper=False, left=False)
            dA1 = _square_lu_backward(L[..., :k, :], U, gL[..., :k, :], gU - L2.mT @ dA2)
            dA = lucid.cat([dA1, dA2], dim=-2)
        return lucid.matmul(P, dA), None, None


def lu_unpack(
    LU_data: Tensor, LU_pivots: Tensor, unpack_data: bool = True, unpack_pivots: bool = True
) -> tuple[Tensor, Tensor, Tensor]:
    """Unpack the packed LU factorization `(LU_data, LU_pivots)` into `(P, L, U)`.

    Parameters
    ----------
    LU_data : Tensor
        Packed LU matrix from `lu_factor`.
    LU_pivots : Tensor
        1-based pivot indices from `lu_factor`.
    unpack_data : bool, optional
        Whether to unpack `L` and `U`.
    unpack_pivots : bool, optional
        Whether to unpack `P`.
    """
    sh = tuple(int(s) for s in LU_data.shape)
    if len(sh) < 2:
        raise ValueError(f"lu_unpack: LU_data must have at least 2 dimensions, got {sh}")
    m, n = sh[-2], sh[-1]
    k = min(m, n)
    piv = _check_pivots(LU_pivots, LU_data, "lu_unpack")
    if unpack_pivots:
        P = _build_permutation_matrix(piv, m, LU_data.dtype, LU_data.device)
    else:
        P = lucid.empty(0, dtype=LU_data.dtype, device=LU_data.device)
    if unpack_data:
        eye = lucid.eye(m, k, dtype=LU_data.dtype, device=LU_data.device)
        L = lucid.tril(LU_data[..., :, :k], -1) + eye
        U = lucid.triu(LU_data[..., :k, :])
    else:
        L = lucid.empty(0, dtype=LU_data.dtype, device=LU_data.device)
        U = lucid.empty(0, dtype=LU_data.dtype, device=LU_data.device)
    return _LuResult(P, L, U)


def lu(A: Tensor, *, pivot: bool = True) -> tuple[Tensor, Tensor, Tensor]:
    r"""Full LU decomposition with explicit factors :math:`(P, L, U)`.

    Decomposes a matrix :math:`A` as

    .. math::

        A \,=\, P\,L\,U,

    where :math:`P` is a row permutation, :math:`L` is unit-lower
    triangular (ones on the diagonal), and :math:`U` is upper
    triangular.  Unlike :func:`lu_factor` (which returns a packed
    factor + integer pivots), this routine returns the three factors
    as explicit dense tensors — convenient for inspection or for
    re-using :math:`P,L,U` in downstream linear-algebra expressions.

    Parameters
    ----------
    A : Tensor
        Matrix of shape ``(*, m, n)`` (batch dims allowed), not
        necessarily square.
    pivot : bool, keyword-only, optional
        Must be ``True`` (the default).  ``False`` would request an
        unpivoted LU; Lucid does not currently ship that kernel and
        raises ``NotImplementedError``.

    Returns
    -------
    P : Tensor
        Permutation matrix of shape ``(*, m, m)``.
    L : Tensor
        Unit-lower-trapezoidal factor of shape ``(*, m, k)``.
    U : Tensor
        Upper-trapezoidal factor of shape ``(*, k, n)``.

    Notes
    -----
    Here :math:`k = \min(m, n)`.  Implemented as :func:`lu_factor`
    followed by :func:`lu_unpack`.  Cost is
    :math:`O(\tfrac{2}{3} n^3)` for a square input, dominated by the
    underlying :math:`LU` factorization.  Differentiable for any shape.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import lu
    >>> A = lucid.tensor([[2.0, 1.0], [4.0, 7.0]])
    >>> P, L, U = lu(A)
    >>> (P @ L @ U).numpy()
    array([[2., 1.],
           [4., 7.]], dtype=float32)
    """
    if not pivot:
        raise NotImplementedError("lu: pivot=False is not supported")
    if A.ndim < 2:
        raise ValueError(f"lu requires a matrix in the last two dims, got {tuple(A.shape)}")
    # A singular matrix has an LU factorization; only lu_factor refuses it.
    LU, pivots, _ = _lu_factor_packed(A)
    return lu_unpack(LU, pivots)
