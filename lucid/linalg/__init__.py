"""
lucid.linalg: linear algebra operations.
"""

import functools
from typing import Callable, cast, final, override

import lucid
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid._dtype import _ENGINE_TO_DTYPE, to_engine_dtype
from lucid._ops._adapters import _outer_adapter
from lucid._ops._autocast import _autocast_entry
from lucid._ops._dims import _wrap_dim
from lucid._ops._reduce import _p_norm, _reduction_axes, _require_normable
from lucid._promotion import try_unwrap as _try_unwrap
from lucid._tensor._indexing import broadcast_shapes as _broadcast_batch
from lucid._tensor.tensor import Tensor

_la = _C_engine.linalg

#: Raised for a matrix a routine could not factor — singular, not
#: positive-definite, an iteration that did not converge.  A
#: ``RuntimeError``, distinct from the argument refusals (``ShapeMismatch``,
#: ``DtypeMismatch``, ...), so ``except LinAlgError`` catches exactly the
#: numerical failures.  The ``*_ex`` variants report it through ``info``.
LinAlgError = _C_engine.LinAlgError


# ── Decorator: auto-unwrap Tensor inputs / re-wrap TensorImpl outputs ─────────


def _linalg_op[R](fn: Callable[..., R]) -> Callable[..., R]:
    """Unwrap every ``Tensor`` argument before the call; re-wrap ``TensorImpl``
    results (including those inside tuples) back to ``Tensor`` afterwards.

    This eliminates the boilerplate ``_wrap(_la.foo(_unwrap(x)))`` pattern
    from every trivial linalg wrapper while keeping full type annotations on
    the public function signature.
    """

    @functools.wraps(fn)
    def wrapper(*args: object, **kwargs: object) -> R:
        """Decorator-generated wrapper that applies the surrounding behaviour to the wrapped callable."""
        ua = tuple(_try_unwrap(a) or a for a in args)
        uk = {k: _try_unwrap(v) or v for k, v in kwargs.items()}
        out: object = fn(*ua, **uk)
        if isinstance(out, tuple):
            out = tuple(_wrap(o) if isinstance(o, _C_engine.TensorImpl) else o for o in out)
        elif isinstance(out, _C_engine.TensorImpl):
            out = _wrap(out)
        # ``fn`` is annotated with what the caller gets once it is re-wrapped,
        # so call sites see ``inv(A)`` as a ``Tensor`` without a cast each.
        return cast(R, out)

    return wrapper


# ── Engine-backed ops ─────────────────────────────────────────────────────────


@_linalg_op
def inv(A: Tensor) -> Tensor:
    r"""Compute the multiplicative inverse of a square matrix.

    Returns the unique matrix :math:`A^{-1}` such that

    .. math::

        A A^{-1} = A^{-1} A = I

    Inversion is performed via LU decomposition with partial pivoting.
    When the goal is to apply :math:`A^{-1}` to a known right-hand side,
    prefer :func:`solve` — explicit inversion is both slower and less
    numerically stable than back-substitution.

    Parameters
    ----------
    A : Tensor
        Square matrix of shape ``(*, n, n)`` (batch dims allowed).  Must
        be non-singular; raises a runtime error on singular input.

    Returns
    -------
    Tensor
        Inverse matrix of shape ``(*, n, n)`` with the same dtype as
        ``A``.

    Notes
    -----
    Internally computes :math:`PA = LU` and then solves ``LU X = P`` for
    ``X`` via two triangular sweeps.  Cost is :math:`O(n^3)` per matrix.
    For ill-conditioned ``A`` consider :func:`pinv` (SVD-based) for a
    more robust pseudo-inverse.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import inv
    >>> A = lucid.tensor([[4.0, 7.0], [2.0, 6.0]])
    >>> inv(A)
    tensor([[0.6, -0.7], [-0.2, 0.4]])

    Notes
    -----
    A matrix with a zero dimension is returned unchanged rather than
    handed to the factorisation.  The 0x0 matrix is its own inverse — it
    is the identity on the zero-dimensional space — so nothing is lost,
    and LAPACK's behaviour at ``n = 0`` is not uniform across the
    implementations this runs against: the same call is harmless on one
    Accelerate build and a segmentation fault on another.  Guarding here
    makes the answer depend on the mathematics rather than on which
    machine is asking.
    """
    if len(A.shape) >= 2 and A.shape[-1] == 0:
        # Inversion preserves shape, and there are no elements to
        # factorise, so the input *is* the answer.
        return A
    return _la.inv(A)  # type: ignore[arg-type, return-value]


@_linalg_op
def det(A: Tensor) -> Tensor:
    r"""Compute the determinant of a square matrix.

    For a square matrix :math:`A \in \mathbb{R}^{n \times n}` returns the
    scalar :math:`\det(A)`.  The determinant is the signed volume of the
    parallelepiped spanned by the rows (or columns) of :math:`A`; it is
    non-zero if and only if :math:`A` is invertible.

    Parameters
    ----------
    A : Tensor
        Square matrix of shape ``(*, n, n)``.

    Returns
    -------
    Tensor
        Determinant of shape ``(*,)`` (one scalar per batch entry).

    Notes
    -----
    Computed as the product of the diagonal of :math:`U` from
    :math:`PA = LU` together with the sign of the row-permutation
    :math:`P`.  Cost is :math:`O(n^3)`.

    For numerically large matrices :math:`\det(A)` can overflow or
    underflow easily — prefer :func:`slogdet` which returns
    :math:`(\mathrm{sign},\,\log|\det A|)` and is stable across scales.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import det
    >>> det(lucid.tensor([[1.0, 2.0], [3.0, 4.0]]))
    tensor(-2.)
    """
    return _la.det(A)  # type: ignore[arg-type, return-value]


def solve(A: Tensor, B: Tensor, *, left: bool = True) -> Tensor:
    r"""Solve a square linear system :math:`AX = B` (or :math:`XA = B`).

    Returns the unique solution :math:`X` of the system

    .. math::

        A X = B

    where :math:`A` is a non-singular square matrix.  The system is
    solved by LU decomposition with partial pivoting, which is both
    faster and more accurate than forming :math:`A^{-1}` explicitly.

    Parameters
    ----------
    A : Tensor
        Square coefficient matrix of shape ``(*, n, n)``.
    B : Tensor
        Right-hand side of shape ``(*, n, k)`` (or ``(*, k, n)`` when
        ``left=False``) or ``(*, n)`` (single RHS vector).
    left : bool, keyword-only, optional
        If ``True`` (default) solves :math:`AX = B`; if ``False`` solves
        :math:`XA = B` via transposition.

    Returns
    -------
    Tensor
        Solution :math:`X` with the same shape as ``B``.

    Notes
    -----
    Algorithm: factor :math:`PA = LU`, then perform forward-substitution
    :math:`L Y = P B` followed by back-substitution :math:`U X = Y`.
    Total cost is :math:`O(n^3 + k n^2)`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import solve
    >>> A = lucid.tensor([[3.0, 1.0], [1.0, 2.0]])
    >>> b = lucid.tensor([9.0, 8.0])
    >>> solve(A, b)
    tensor([2., 3.])
    """
    if not left:
        if B.ndim < 2 or tuple(B.shape) == tuple(A.shape[:-1]):
            raise _C_engine.ShapeMismatch(
                f"solve(left=False): B must be a matrix or batch of matrices, got shape {tuple(B.shape)}"
            )
        rhs = B.unsqueeze(0).mT if tuple(B.mT.shape) == tuple(A.shape[:-1]) else B.mT
        return _wrap(_la.solve(_unwrap(A.mT), _unwrap(rhs))).mT
    return _wrap(_la.solve(_unwrap(A), _unwrap(B)))


def cholesky(input: Tensor, *, upper: bool = False) -> Tensor:
    r"""Cholesky decomposition of a symmetric positive-definite matrix.

    For a real symmetric positive-definite (SPD) matrix :math:`A`
    returns the unique lower-triangular factor :math:`L` with positive
    diagonal such that

    .. math::

        A = L L^\top \qquad \text{(or } A = U^\top U \text{ if } \texttt{upper=True}\text{)}.

    The Cholesky factor is the standard tool for solving SPD linear
    systems, sampling from a multivariate Gaussian, and computing
    matrix square roots.  It is roughly twice as fast as LU and avoids
    any pivoting.

    Parameters
    ----------
    input : Tensor
        Symmetric positive-definite matrix of shape ``(*, n, n)``.
    upper : bool, optional
        If ``True`` return the upper-triangular factor :math:`U` such
        that :math:`A = U^\top U`.  Default ``False`` returns the
        lower-triangular factor :math:`L`.

    Returns
    -------
    Tensor
        Triangular Cholesky factor of shape ``(*, n, n)``.

    Notes
    -----
    Algorithm: LAPACK ``potrf`` via Apple Accelerate on CPU, MLX on GPU.
    Cost is :math:`O(n^3 / 3)`, half the work of LU.

    Backward is implemented in Python via Murray's (2016) formula

    .. math::

        \frac{\partial L}{\partial A} \,=\, \mathrm{sym}\!\big(L^{-\top}\,\Phi(L^\top G)\,L^{-1}\big),

    where :math:`\Phi` zeros the strict upper triangle and halves the
    diagonal.  Two triangular solves implement the inversion implicitly.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import cholesky
    >>> A = lucid.tensor([[4.0, 2.0], [2.0, 3.0]])  # SPD
    >>> L = cholesky(A)
    >>> L
    tensor([[2., 0.], [1., 1.414]])
    >>> L @ L.T
    tensor([[4., 2.], [2., 3.]])
    """
    with _autocast_entry("cholesky", _unwrap(input)) as (x_impl,):
        return cast(Tensor, _CholeskyAutograd.apply(_wrap(x_impl), upper))


from lucid.autograd.function import Function as _AutogradFunction
from lucid.autograd.function import FunctionCtx
from lucid._unsupported import unsupported_if


@final
class _CholeskyAutograd(_AutogradFunction):
    """Custom-autograd wrapper around the engine's non-differentiable
    cholesky_op. Forward calls the engine; backward computes the input
    gradient via Murray (2016)."""

    @override
    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor, upper: bool) -> Tensor:
        out = _wrap(_la.cholesky(_unwrap(x), upper))
        ctx.save_for_backward(out)
        ctx.upper = bool(upper)
        return out

    @override
    @staticmethod
    def backward(ctx: FunctionCtx, grad_out: Tensor) -> Tensor:
        (factor,) = ctx.saved_tensors  # L (upper=False) or U (upper=True)
        upper: bool = cast(bool, ctx.upper)

        # Normalise to lower-triangular form L; gradient w.r.t. that L.
        if upper:
            L = factor.mT
            gL = grad_out.mT
        else:
            L = factor
            gL = grad_out

        n: int = int(L.shape[-1])
        eye_n = lucid.eye(n, dtype=L.dtype)
        if L.device != "cpu":
            eye_n = eye_n.to(L.device)
        # Mask gL to its lower triangle — the strictly upper half doesn't
        # contribute to L (which is lower-triangular by construction).
        gL_tril = lucid.tril(gL)
        # Phi(M): tril(M) with diagonal halved.
        M = lucid.matmul(L.mT, gL_tril)
        Phi = lucid.tril(M) - 0.5 * (M * eye_n)

        # S = L^{-T} @ Phi @ L^{-1}, computed via two triangular solves.
        # Step 1: Y = L^{-T} Phi  →  solve L^T Y = Phi (upper=True against L^T).
        Y = solve_triangular(L.mT, Phi, upper=True)
        # Step 2: Z = Y L^{-1}.  Take transposes: Z^T = L^{-T} Y^T, so solve
        # L^T Z^T = Y^T then transpose back.
        Z = solve_triangular(L.mT, Y.mT, upper=True).mT
        # Murray's formula gives the symmetric Riemannian gradient for A = A^T:
        # grad_A = 1/2 (L^{-T} Phi(L^T gL) L^{-1} + sym), matching the reference
        # framework's cholesky backward convention for upper=False and upper=True.
        return 0.5 * (Z + Z.mT)


def norm(
    input: Tensor,
    ord: int | float | str | None = None,
    dim: int | list[int] | tuple[int, ...] | None = None,
    keepdim: bool = False,
    *,
    dtype: object = None,
) -> Tensor:
    r"""Compute a vector or matrix norm.

    Generic norm dispatcher that delegates to :func:`vector_norm` or
    :func:`matrix_norm` based on the input rank and reduction axes.
    The default behaviour computes the Frobenius norm of a matrix input
    and the Euclidean (:math:`\ell_2`) norm of a vector input:

    .. math::

        \|x\|_2 \,=\, \Big(\sum_i |x_i|^2\Big)^{1/2}, \qquad
        \|A\|_F \,=\, \Big(\sum_{i,j} |A_{ij}|^2\Big)^{1/2}.

    Parameters
    ----------
    input : Tensor
        Input tensor.  May be a vector ``(n,)``, a matrix ``(m, n)``, or
        higher-rank with explicit ``dim``.
    ord : int, float, str or None, optional
        Order of the norm.  Vector orders: ``0``, ``1``, ``2`` (default
        when None), ``inf``, ``-inf``, or any positive real.  Matrix
        orders: ``"fro"``, ``"nuc"``, ``1``, ``-1``, ``2``, ``-2``,
        ``inf``, ``-inf``.
    dim : int, list of int, tuple of int or None, optional
        Reduction axis (or pair of axes for matrix norms).  ``None``
        reduces over all elements.
    keepdim : bool, optional
        If ``True``, retains reduced dimensions with size 1.
    dtype : dtype, keyword-only, optional
        If given, ``input`` is cast to ``dtype`` before computing the norm.

    Returns
    -------
    Tensor
        Norm value(s); shape depends on ``dim`` / ``keepdim``.

    Raises
    ------
    DtypeMismatch
        If ``input`` is an integer or bool tensor (a ``TypeError``).
    ValueError
        If ``ord`` is given without ``dim`` and the input is neither a
        vector nor a single matrix — a batch of matrices leaves it
        ambiguous whether one norm or one per matrix was meant.  Also if
        ``dim`` names more than two axes, or if a matrix order is asked
        of a vector.
    NotImplementedError
        If a complex input is asked for a singular-value order (``"nuc"``
        or a matrix ``ord`` of :math:`\pm 2`), which :func:`matrix_norm`
        cannot take yet.

    Notes
    -----
    Which of the two norms runs is decided as follows:

    ============================  ==========================================
    arguments                     reduction
    ============================  ==========================================
    ``dim=None, ord=None``        flatten, Euclidean norm, any rank
    ``dim=None, ord`` given       vector norm if 1-D, matrix norm if 2-D
    ``dim`` names one axis        vector norm along it
    ``dim`` names two axes        matrix norm over that plane
    ``ord`` is ``"fro"``/``"nuc"``  matrix norm regardless
    ============================  ==========================================

    So ``ord`` means different things at different ranks, and the same
    order names two different quantities: ``ord=2`` is the Euclidean norm
    of a vector but the *spectral* norm — the largest singular value — of
    a matrix, which is not its Frobenius norm.  Pass an explicit ``dim``,
    or call :func:`vector_norm` or :func:`matrix_norm` directly, wherever
    that distinction matters.

    Many norm orders (spectral, nuclear) require an SVD and therefore
    cost :math:`O(\min(m,n) \cdot mn)`; entry-wise norms reduce in a
    single pass.

    Overflow is not a failure mode here: the input is rescaled by its
    largest magnitude before reducing, so a norm whose value is
    representable is computed even when its squares are not.

    A complex input is reduced through its magnitudes :math:`|x_i|` by
    whichever of the two norms runs, and the result is real.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import norm
    >>> norm(lucid.tensor([3.0, 4.0]))
    tensor(5.)
    >>> A = lucid.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    >>> norm(A, ord=1)          # max absolute column sum
    tensor(9.)
    >>> norm(A, ord=1, dim=1)   # per-row vector 1-norm
    tensor([6., 15.])
    """
    _require_normable(_unwrap(input), "norm")
    x = _wrap(_cast_for_norm(input, dtype)) if dtype is not None else input
    with _autocast_entry("norm", _unwrap(x)) as (x_impl,):
        return _norm_body(_wrap(x_impl), ord, dim, keepdim)


def _norm_body(
    x: Tensor,
    ord: int | float | str | None,
    dim: int | list[int] | tuple[int, ...] | None,
    keepdim: bool,
) -> Tensor:
    rank = len(_unwrap(x).shape)
    axes: tuple[int, ...] | None
    if dim is None:
        axes = None
    elif isinstance(dim, (list, tuple)):
        axes = tuple(int(d) for d in dim)
    else:
        axes = (int(dim),)

    if axes is None:
        if ord is None:
            return _scaled(x, ord, lambda t: vector_norm(t, 2, None, keepdim))
        if rank == 2:
            return _scaled(x, ord, lambda t: matrix_norm(t, ord, (0, 1), keepdim))
        if rank == 1 and not isinstance(ord, str):
            return _scaled(x, ord, lambda t: vector_norm(t, ord, 0, keepdim))
        if rank == 1:
            raise ValueError(
                f"norm: ord={ord!r} is a matrix order and needs at least " "2 dimensions, got 1-D"
            )
        raise ValueError(
            "norm: when dim is not given but ord is, the input must be 1-D "
            f"or 2-D, got {rank}-D — pass dim to say which axes to reduce"
        )

    if isinstance(ord, str) or len(axes) == 2:
        if len(axes) != 2:
            raise ValueError(
                f"norm: ord={ord!r} is a matrix order and needs a pair of " f"axes, got dim={dim!r}"
            )
        pair = (axes[0], axes[1])
        m_ord: int | float | str = "fro" if ord is None else ord
        return _scaled(x, ord, lambda t: matrix_norm(t, m_ord, pair, keepdim))
    if len(axes) == 1:
        v_ord: int | float = 2 if ord is None else ord
        return _scaled(x, ord, lambda t: vector_norm(t, v_ord, [axes[0]], keepdim))
    raise ValueError(f"norm: dim must name one or two axes, got {dim!r}")


def _scaled(
    x: Tensor, ord: int | float | str | None, compute: Callable[[Tensor], Tensor]
) -> Tensor:
    """Run ``compute`` on a rescaled ``x`` and undo the scaling.

    Every norm order here is absolutely homogeneous — ``‖αx‖ = |α|‖x‖`` —
    so the largest magnitude can be divided out first and multiplied back
    at the end.  Every order but one: ``ord=0`` counts non-zero entries,
    which no positive scaling changes, so multiplying the scale back in
    inflates the count by it — ``norm([1, -2, 3], ord=0)`` answered 9
    instead of 3.  It also cannot overflow, so it skips the rescaling
    rather than being corrected for it.  That matters because the squaring in a 2-norm reaches
    infinity before the answer does: ``norm([1e200, 1e200])`` is 1.41e200,
    an ordinary double, and the direct evaluation says ``inf``.  The other
    end underflows to zero.  Dividing by the largest magnitude puts every
    ratio in [0, 1], so nothing squares out of range.  BLAS's ``nrm2`` has
    rescaled for the same reason since the seventies.

    The scale is only used when it is finite and positive.  An all-zero
    input has scale 0, and an input containing an infinity has scale inf —
    dividing by either turns the answer into NaN, which is how the first
    version of this rescaling reported ``norm([inf, 1])``.  Both fall back
    to 1, where the unscaled evaluation is already right.
    """
    if ord == 0 and not isinstance(ord, str):
        return compute(x)
    # Nothing to rescale, and no largest magnitude to find: ``max`` of an
    # empty tensor is refused, as it has no identity.
    if x.numel() == 0:
        return compute(x)
    scale = lucid.max(lucid.abs(x))
    usable = lucid.isfinite(scale) & (scale > lucid.zeros_like(scale))
    safe = lucid.where(usable, scale, lucid.ones_like(scale))
    return compute(x / safe) * safe


# ── SVD with backward ─────────────────────────────────────────────────────────
#
# Backward formula (Giles 2008, extended to rectangular A(m×n), k=min(m,n)):
#   F[i,j] = 1 / (s_j² - s_i²)  for i≠j,  F[i,i] = 0
#   dA from S:  U diag(G_S) Vh
#   dA from U:  J = F ⊙ (U^T G_U)
#               U ((J + J^T) diag(s)) Vh + (I_m - U U^T) G_U Σ⁻¹ Vh    [if m>k]
#   dA from Vh: K = F ⊙ (Vh G_Vh^T)
#               U (diag(s) (K + K^T)) Vh + U Σ⁻¹ G_Vh (I_n - Vh^T Vh)  [if n>k]


def _svd_loewner(S: Tensor) -> Tensor:
    """Build the Loewner matrix F[i,j] = 1/(s_j²-s_i²) for i≠j."""
    k = int(S.shape[-1])
    Si = S.unsqueeze(-1)  # (..., k, 1)
    Sj = S.unsqueeze(-2)  # (..., 1, k)
    denom = Sj * Sj - Si * Si  # (..., k, k)
    eye_k = lucid.eye(k, dtype=S.dtype, device=S.device)
    safe_denom = denom + eye_k  # avoid div-by-zero on diagonal
    return (1.0 - eye_k) / safe_denom  # zero diagonal


@final
class _SVD(_AutogradFunction):
    """Reduced SVD with Giles' (2008) backward across ``(U, S, Vh)``."""

    _materialize_grads = False

    @override
    @staticmethod
    def forward(ctx: FunctionCtx, A: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        u_impl, s_impl, vh_impl = _la.svd(_unwrap(A))
        U, S, Vh = _wrap(u_impl), _wrap(s_impl), _wrap(vh_impl)
        ctx.save_for_backward(U, S, Vh)
        return U, S, Vh

    @override
    @staticmethod
    def backward(
        ctx: FunctionCtx,
        G_U: Tensor | None,
        G_S: Tensor | None,
        G_Vh: Tensor | None,
    ) -> Tensor:
        U, S, Vh = ctx.saved_tensors
        dA: Tensor | None = None
        if G_S is not None:
            dA = U @ lucid.diag_embed(G_S) @ Vh
        if G_U is not None or G_Vh is not None:
            k = int(S.shape[-1])
            F = _svd_loewner(S)
            if G_U is not None:
                m = int(U.shape[-2])
                UtgU = U.mT @ G_U  # (..., k, k)
                J = F * UtgU
                dA_U = U @ ((J + J.mT) * S.unsqueeze(-2)) @ Vh
                if m > k:
                    S_inv = lucid.diag_embed(1.0 / S)
                    proj_gU = G_U - U @ UtgU
                    dA_U = dA_U + proj_gU @ S_inv @ Vh
                dA = dA_U if dA is None else dA + dA_U
            if G_Vh is not None:
                n = int(Vh.shape[-1])
                gV = G_Vh.mT  # (..., n, k)
                K = F * (Vh @ gV)  # (..., k, k)
                dA_Vh = U @ (S.unsqueeze(-1) * (K + K.mT)) @ Vh
                if n > k:
                    S_inv = lucid.diag_embed(1.0 / S)
                    proj_gVh = G_Vh - (G_Vh @ Vh.mT) @ Vh
                    dA_Vh = dA_Vh + U @ S_inv @ proj_gVh
                dA = dA_Vh if dA is None else dA + dA_Vh
        assert dA is not None
        return dA


def svd(
    A: Tensor, full_matrices: bool = True, *, driver: str | None = None
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Singular value decomposition of a matrix.

    Factorizes any (possibly rectangular) matrix :math:`A \in
    \mathbb{R}^{m \times n}` into

    .. math::

        A \,=\, U \,\Sigma\, V^\top,

    where :math:`U` and :math:`V` are orthogonal and
    :math:`\Sigma = \mathrm{diag}(\sigma_1, \ldots, \sigma_k)` with
    :math:`\sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_k \ge 0` and
    :math:`k = \min(m, n)`.  The SVD is the foundation of low-rank
    approximation, pseudo-inverse, matrix rank, condition number, and
    PCA.

    Parameters
    ----------
    A : Tensor
        Input matrix of shape ``(*, m, n)`` (batch dims allowed).
    full_matrices : bool, optional
        If ``True`` (default), :math:`U` is ``(m, m)`` and
        :math:`V^\top` is ``(n, n)`` — the full orthogonal factors.
        If ``False``, returns the reduced SVD with :math:`U` shaped
        ``(m, k)`` and :math:`V^\top` shaped ``(k, n)``.
    driver : str or None, keyword-only, optional
        LAPACK driver name.  Must be ``None``.

    Returns
    -------
    U : Tensor
        Left singular vectors, shape ``(*, m, m)`` or ``(*, m, k)``.
    S : Tensor
        Singular values in descending order, shape ``(*, k)``.
    Vh : Tensor
        Right singular vectors (conjugate-transposed), shape
        ``(*, n, n)`` or ``(*, k, n)``.

    Notes
    -----
    Backward is implemented via the multi-output ``_SVD`` ``Function``
    wrapper so gradients from :math:`U`, :math:`S`, and :math:`V^\top`
    accumulate into the input via Giles' (2008) formula using the
    Loewner matrix :math:`F_{ij} = \sigma_i / (\sigma_i^2 - \sigma_j^2)`.
    Gradients blow up when singular values are nearly repeated —
    degenerate or rank-deficient inputs are not differentiation-friendly.

    Cost is :math:`O(\min(m,n)^2 \cdot \max(m,n))`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import svd
    >>> A = lucid.tensor([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
    >>> U, S, Vh = svd(A, full_matrices=False)
    >>> S
    tensor([2., 1.])
    """
    unsupported_if(driver is not None, "svd", "driver", driver)
    with _autocast_entry("svd", _unwrap(A)) as (a_impl,):
        A_cast = _wrap(a_impl)
        if not _C_engine.grad_enabled() or not A_cast.requires_grad:
            u_impl, s_impl, vh_impl = _la.svd(a_impl)
            return _complete_svd_factors(
                A_cast, _wrap(u_impl), _wrap(s_impl), _wrap(vh_impl), full_matrices
            )
        out = _SVD.apply(A_cast)
        assert isinstance(out, tuple)
        U, S, Vh = out
        return _complete_svd_factors(A_cast, U, S, Vh, full_matrices)


def svdvals(A: Tensor, *, driver: str | None = None) -> Tensor:
    r"""Compute only the singular values of a matrix.

    Returns the singular values :math:`\sigma_1 \ge \cdots \ge
    \sigma_k \ge 0` (with :math:`k = \min(m, n)`) of an input matrix
    :math:`A` without forming :math:`U` or :math:`V^\top`.  Equivalent
    to ``svd(A, full_matrices=False)[1]`` but avoids the work of
    constructing the singular vectors.

    Parameters
    ----------
    A : Tensor
        Input matrix of shape ``(*, m, n)``.
    driver : str or None, keyword-only, optional
        LAPACK driver name.  Must be ``None``.

    Returns
    -------
    Tensor
        Singular values in descending order, shape ``(*, k)``.

    Notes
    -----
    When gradients are required this routes through the reduced
    :func:`svd` so backward still works.  Without ``requires_grad``,
    the engine kernel skips assembly of the singular vectors and is
    roughly :math:`2\times` faster.  Useful for computing
    :func:`matrix_rank`, :func:`cond`, or the nuclear / spectral norms.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import svdvals
    >>> svdvals(lucid.tensor([[3.0, 0.0], [0.0, 4.0]]))
    tensor([4., 3.])
    """
    unsupported_if(driver is not None, "svdvals", "driver", driver)
    with _autocast_entry("svdvals", _unwrap(A)) as (a_impl,):
        A_cast = _wrap(a_impl)
        if _C_engine.grad_enabled() and A_cast.requires_grad:
            _, S, _ = svd(A_cast, full_matrices=False)
            return S
        result = _la.svd(a_impl, False)
        if isinstance(result, (list, tuple)):
            return _wrap(result[0])
        return _wrap(result)


# ── QR with backward ─────────────────────────────────────────────────────────
#
# Strategy: express R via the Cholesky decomposition of A^T A.
#
# For A = Q R (m≥n, R upper triangular):
#   A^T A = R^T R  (since Q^T Q = I)
#
# The lower Cholesky factor L = chol(A^T A) satisfies L L^T = A^T A,
# with positive diagonal (by definition).  The relationship between L and R
# from LAPACK's dgeqrf is: R = D L^T  where D = diag(sign(diag(R))).
#
# This lets us route the R backward through the existing (correct) Cholesky
# backward without implementing the Gu-Eisenstat formula, which requires
# positive-diagonal R and fails for LAPACK's sign convention.
#
# Q backward: from A = Q R, Q = A R^{-1} for square A.  For thin QR (m>n),
# Q doesn't have a simple closed-form inverse through A.  We route Q through
# the same Cholesky path or via projection.
#
# For simplicity, the combined QR backward is implemented in a single
# Function wrapper that handles both Q and R gradients together.


def _copyltu(M: Tensor) -> Tensor:
    """Mirror the strict lower triangle of ``M`` upward, keeping its diagonal.

    ``tril(M, -1) + tril(M, -1)ᵀ + diag(M)`` — not ``sym(M)``; the two agree
    only for a symmetric ``M``.
    """
    strict_lower = lucid.tril(M, -1)
    return strict_lower + strict_lower.mT + lucid.diag_embed(M.diagonal(dim1=-2, dim2=-1))


def _qr_wide_backward(
    A: Tensor, Q: Tensor, R: Tensor, G_Q: Tensor | None, G_R: Tensor | None
) -> Tensor:
    r"""``dA`` for the reduced QR of a wide ``A`` (``m < n``).

    Split ``A = [X | Y]`` and ``R = [U | V]`` after the first ``m``
    columns: ``X = Q U`` is a square QR and ``V = Qᵀ Y``.  ``V``'s gradient
    reaches ``Y`` as ``Q G_V`` and ``Q`` as ``Y G_Vᵀ``; what remains is the
    square case for ``X``:

    .. math::

        G_X = \big(G_Q' + Q\,\mathrm{copyltu}(U G_U^\top - G_Q'^\top Q)\big)
              U^{-\top}, \qquad G_Q' = G_Q + Y G_V^\top.

    The tall / square formula needed ``R`` square to invert it, so a wide
    matrix's backward failed on a shape mismatch.  Either cotangent may be
    absent — each output's Function supplies its own and the
    contributions add up in ``A.grad``.
    """
    m = int(Q.shape[-1])
    U = R[..., :m]
    gQ = G_Q if G_Q is not None else lucid.zeros_like(Q)
    gY: Tensor | None = None
    if G_R is not None:
        gV = G_R[..., m:]
        gQ = gQ + A[..., m:] @ gV.mT
        M = U @ G_R[..., :m].mT - gQ.mT @ Q
        gY = Q @ gV
    else:
        M = -(gQ.mT @ Q)
    gX = solve_triangular(U, (gQ + Q @ _copyltu(M)).mT, upper=True).mT
    if gY is None:
        gY = lucid.zeros_like(A[..., m:])
    return lucid.cat([gX, gY], -1)


def _qr_tall_r_backward(A: Tensor, R: Tensor, G_R: Tensor) -> Tensor:
    """R contribution to ``dA`` via the Cholesky factor of ``A^T A`` (``m >= n``)."""
    n = int(R.shape[-1])
    diag_R = R.diagonal(dim1=-2, dim2=-1)
    abs_d = diag_R.abs()
    safe_abs_d = lucid.where(abs_d < 1e-12, lucid.ones_like(abs_d), abs_d)
    sign_d = diag_R / safe_abs_d
    D = lucid.diag_embed(sign_d)
    # L = R.mT @ D  (lower triangular, positive diagonal).  Kept on the graph
    # so second derivatives through R retain R's dependence on A.
    L = R.mT @ D
    G_L = G_R.mT @ D
    eye_n = lucid.eye(n, dtype=L.dtype, device=L.device)
    M = L.mT @ G_L
    Phi = lucid.tril(M) - 0.5 * (M * eye_n)
    Y = solve_triangular(L.mT, Phi, upper=True)
    Z = solve_triangular(L.mT, Y.mT, upper=True).mT
    G_B = (Z + Z.mT) * 0.5
    return (2.0 * A) @ G_B


@final
class _QR(_AutogradFunction):
    """Reduced QR with combined ``(Q, R)`` backward."""

    _materialize_grads = False

    @override
    @staticmethod
    def forward(ctx: FunctionCtx, A: Tensor) -> tuple[Tensor, Tensor]:
        q_impl, r_impl = _la.qr(_unwrap(A))
        Q, R = _wrap(q_impl), _wrap(r_impl)
        ctx.save_for_backward(A, Q, R)
        return Q, R

    @override
    @staticmethod
    def backward(ctx: FunctionCtx, G_Q: Tensor | None, G_R: Tensor | None) -> Tensor:
        A, Q, R = ctx.saved_tensors
        if int(A.shape[-2]) < int(A.shape[-1]):
            return _qr_wide_backward(A, Q, R, G_Q, G_R)
        dA: Tensor | None = None
        if G_Q is not None:
            M = -(G_Q.mT @ Q)
            numerator = G_Q + Q @ _copyltu(M)
            dA = solve_triangular(R, numerator.mT, upper=True).mT
        if G_R is not None:
            dA_R = _qr_tall_r_backward(A, R, G_R)
            dA = dA_R if dA is None else dA + dA_R
        assert dA is not None
        return dA


def qr(A: Tensor, mode: str = "reduced") -> tuple[Tensor, Tensor]:
    r"""QR decomposition of a matrix.

    Factorizes a matrix :math:`A \in \mathbb{R}^{m \times n}` as

    .. math::

        A \,=\, Q\,R,

    where :math:`Q` has orthonormal columns (:math:`Q^\top Q = I`) and
    :math:`R` is upper-triangular.  Used for orthogonalizing a basis,
    solving least-squares (:math:`\min \|Ax - b\|_2`), and computing
    eigenvalues via QR iteration.

    Parameters
    ----------
    A : Tensor
        Input matrix of shape ``(*, m, n)``.
    mode : {"reduced", "complete", "r"}, optional
        ``"reduced"`` (default): :math:`Q` is ``(m, k)`` and :math:`R`
        is ``(k, n)`` with :math:`k = \min(m, n)`.
        ``"complete"``: :math:`Q` is ``(m, m)`` and :math:`R` is
        ``(m, n)``.
        ``"r"``: return only :math:`R` (``Q`` is an empty tensor).

    Returns
    -------
    Q : Tensor
        Orthogonal factor.
    R : Tensor
        Upper-triangular factor.

    Notes
    -----
    Computed via Householder reflections (LAPACK ``geqrf``).  Cost is
    :math:`O(2 m n^2 - \tfrac{2}{3} n^3)` for :math:`m \ge n`.

    The diagonal of :math:`R` may carry arbitrary signs (LAPACK
    convention) — the factorization is unique only up to a diagonal
    sign matrix.  Backward routes :math:`R` through a Cholesky of
    :math:`A^\top A` (sign-robust) and :math:`Q` through the
    Stiefel-manifold tangent projection.  A wide input
    (:math:`m < n`) splits as :math:`A = [X \mid Y]`, :math:`R = [U \mid
    V]`: :math:`X = QU` is differentiated as a square QR and
    :math:`V = Q^\top Y` carries the rest.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import qr
    >>> A = lucid.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    >>> Q, R = qr(A)
    >>> lucid.allclose(Q.T @ Q, lucid.eye(2), atol=1e-5)   # Q is orthonormal
    True
    """
    with _autocast_entry("qr", _unwrap(A)) as (a_impl,):
        A_cast = _wrap(a_impl)
        if not _C_engine.grad_enabled() or not A_cast.requires_grad:
            q_impl, r_impl = _la.qr(a_impl)
            return _complete_qr_factors(A_cast, _wrap(q_impl), _wrap(r_impl), mode)
        out = _QR.apply(A_cast)
        assert isinstance(out, tuple)
        Q, R = out
        return _complete_qr_factors(A_cast, Q, R, mode)


@final
class _MatrixPowerZero(_AutogradFunction):
    """``A⁰ = I`` — a fresh identity per matrix, on ``A``'s graph.

    The identity was built without a device, so a Metal input got a CPU
    answer and the next op raised DeviceMismatch; and it had no
    ``grad_fn``, so a loss through it could not be backpropagated at all.
    Its gradient is zero — the identity does not depend on ``A`` — which
    is what this node hands back.
    """

    @override
    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor) -> Tensor:
        shape = tuple(int(s) for s in x.shape)
        eye = lucid.eye(shape[-1], dtype=x.dtype, device=x.device)
        if len(shape) == 2:
            return eye
        # A fresh identity per batch element, writable as the reference's
        # is — broadcast_to alone would be a read-only view of one.
        return lucid.broadcast_to(eye, shape).contiguous()

    @override
    @staticmethod
    def backward(ctx: FunctionCtx, grad_out: Tensor) -> Tensor:
        return lucid.zeros_like(grad_out)


def matrix_power(input: Tensor, n: int) -> Tensor:
    r"""Raise a square matrix to an integer power.

    Computes :math:`A^n` for an integer exponent :math:`n`:

    .. math::

        A^n \,=\, \begin{cases}
            \underbrace{A A \cdots A}_{n\ \text{times}} & n > 0 \\
            I & n = 0 \\
            (A^{-1})^{|n|} & n < 0
        \end{cases}

    Parameters
    ----------
    input : Tensor
        Square matrix of shape ``(*, m, m)``.
    n : int
        Integer exponent.  Negative values require :math:`A` to be
        invertible.

    Returns
    -------
    Tensor
        :math:`A^n`, shape ``(*, m, m)``.

    Notes
    -----
    Uses binary exponentiation (repeated squaring) so the cost is
    :math:`O(\log |n|)` matrix multiplies rather than :math:`|n|`.
    Implemented as a Python composite over :func:`matmul` and
    :func:`inv` so autograd flows naturally — the engine
    ``matrix_power_op`` is not differentiable.

    The result is always a new tensor on ``input``'s device: ``n = 0`` gives
    an identity per matrix (with a zero gradient) and ``n = 1`` a copy
    of ``input``, never ``input`` itself.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import matrix_power
    >>> A = lucid.tensor([[1.0, 1.0], [0.0, 1.0]])
    >>> matrix_power(A, 5)
    tensor([[1., 5.], [0., 1.]])
    """
    if not isinstance(n, int):
        raise TypeError(f"matrix_power exponent must be int, got {type(n).__name__}")

    sh: tuple[int, ...] = tuple(_unwrap(input).shape)
    if len(sh) < 2 or sh[-1] != sh[-2]:
        raise ValueError(f"matrix_power requires a square matrix in the last two dims, got {sh}")

    if n == 0:
        return cast(Tensor, _MatrixPowerZero.apply(input))

    base: Tensor = inv(input) if n < 0 else input
    exponent: int = -n if n < 0 else n
    if exponent == 1:
        # A new tensor, never the argument itself: handing ``input`` back made
        # ``matrix_power(a, 1).add_(1)`` write into ``a``.
        return base.clone() if base is input else base

    # Standard binary exponentiation: result starts at base if the lowest
    # bit is set, otherwise it gets multiplied in on the first set bit.
    #
    # Each factor joins on the left.  Powers of one matrix commute, so the
    # value is the same either way; the order is for Core ML, whose CPU graph
    # compiler never finishes on ``x @ (x @ x)`` — the right-multiplied n = 3
    # — and loads ``(x @ x) @ x`` and every other power tried (CHA-20).
    result: Tensor | None = None
    cur: Tensor = base
    while exponent > 0:
        if exponent & 1:
            result = cur if result is None else lucid.matmul(cur, result)
        exponent >>= 1
        if exponent:
            cur = lucid.matmul(cur, cur)
    assert result is not None  # exponent was non-zero on entry
    return result


def pinv(
    input: Tensor,
    rcond: float | Tensor | None = None,
    hermitian: bool = False,
    *,
    atol: float | Tensor | None = None,
    rtol: float | Tensor | None = None,
) -> Tensor:
    r"""Moore-Penrose pseudo-inverse of a matrix.

    Returns the unique matrix :math:`A^+` satisfying the four
    Moore-Penrose conditions

    .. math::

        A A^+ A = A, \quad A^+ A A^+ = A^+, \quad
        (A A^+)^\top = A A^+, \quad (A^+ A)^\top = A^+ A.

    For a thin SVD :math:`A = U\Sigma V^\top`, the pseudo-inverse is

    .. math::

        A^+ \,=\, V\,\Sigma^+\,U^\top, \qquad
        \Sigma^+_{ii} = \begin{cases} 1/\sigma_i & \sigma_i > \tau \\ 0 & \text{else} \end{cases}.

    Parameters
    ----------
    input : Tensor
        Input matrix of shape ``(*, m, n)``.  Need not be square or
        full-rank.
    rcond : float, Tensor or None, optional
        Alias for ``rtol``; mutually exclusive with ``atol`` and ``rtol``.
    hermitian : bool, optional
        If ``True``, assumes ``input`` is symmetric / Hermitian and uses
        :func:`eigh` of the lower triangle instead of :func:`svd`.
    atol : float, Tensor or None, keyword-only, optional
        Absolute tolerance threshold (treated as ``0.0`` when ``None``).
    rtol : float, Tensor or None, keyword-only, optional
        Relative tolerance threshold; defaults to
        :math:`\max(m, n) \cdot \varepsilon` unless ``atol > 0``, in
        which case ``0.0``.

    Returns
    -------
    Tensor
        Pseudo-inverse of shape ``(*, n, m)``.

    Notes
    -----
    Every shape goes through the SVD-based kernel, a singular square
    matrix included (routing square input through :func:`inv` answered
    a singular one with a non-finite matrix).  The gradient follows the
    reference framework's closed form for a matrix of constant rank,
    which needs no singular-value gaps, and differentiates again.

    The pseudo-inverse provides the least-squares solution of
    :math:`Ax = b` even when :math:`A` is singular: :math:`x = A^+ b`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import pinv
    >>> A = lucid.tensor([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
    >>> lucid.allclose(pinv(A) @ A, lucid.eye(2), atol=1e-5)
    True
    """
    if rcond is not None:
        if atol is not None or rtol is not None:
            raise TypeError("pinv: rcond cannot be specified together with atol or rtol")
        rtol = rcond
    if atol is not None or rtol is not None or hermitian:
        return _pinv_with_tol(input, atol=atol, rtol=rtol, hermitian=hermitian)
    if input.requires_grad and lucid.is_grad_enabled():
        return cast(Tensor, _Pinv.apply(input))
    return _wrap(_la.pinv(_unwrap(input)))


@final
class _Pinv(_AutogradFunction):
    """``A⁺`` with the reference framework's backward.

    The kernel recorded no gradient, so a loss through a rectangular
    ``pinv`` trained nothing.  For ``P = A⁺`` and an incoming ``G``, with
    ``K`` chosen to keep the products at the smaller dimension::

        m <= n:  K = Gᵀ P
                 dA = -(P K)ᵀ + K Pᵀ - (A P) K Pᵀ + (Pᵀ P)(Gᵀ - K A)
        m >  n:  K = P Gᵀ
                 dA = -(K P)ᵀ + (Gᵀ - A K) P Pᵀ + Pᵀ K - Pᵀ K P A

    Built from differentiable ops on the saved ``A`` and ``P``, so it
    differentiates again.
    """

    @override
    @staticmethod
    def forward(ctx: FunctionCtx, A: Tensor, p_impl: _C_engine.TensorImpl | None = None) -> Tensor:
        P = _wrap(_la.pinv(_unwrap(A)) if p_impl is None else p_impl)
        ctx.save_for_backward(A, P)
        return P

    @override
    @staticmethod
    def backward(ctx: FunctionCtx, grad_out: Tensor) -> Tensor:
        A, P = ctx.saved_tensors
        m, n = int(A.shape[-2]), int(A.shape[-1])
        Gt = grad_out.mT
        Pt = P.mT
        if m <= n:
            K = lucid.matmul(Gt, P)
            KPt = lucid.matmul(K, Pt)
            return (
                -lucid.matmul(P, K).mT
                + KPt
                - lucid.matmul(lucid.matmul(A, P), KPt)
                + lucid.matmul(lucid.matmul(Pt, P), Gt - lucid.matmul(K, A))
            )
        K = lucid.matmul(P, Gt)
        PtK = lucid.matmul(Pt, K)
        return (
            -lucid.matmul(K, P).mT
            + lucid.matmul(lucid.matmul(Gt - lucid.matmul(A, K), P), Pt)
            + PtK
            - lucid.matmul(lucid.matmul(PtK, P), A)
        )


def eig(input: Tensor) -> tuple[Tensor, Tensor]:
    r"""Eigenvalue decomposition of a general square matrix.

    For a general (not necessarily symmetric) square matrix :math:`A`
    returns eigenvalues :math:`\lambda_i` and right eigenvectors
    :math:`v_i` such that

    .. math::

        A v_i \,=\, \lambda_i\, v_i,
        \qquad A \,=\, V\,\Lambda\,V^{-1}.

    For matrices with complex eigenvalues, the result is in general
    complex-valued.  When :math:`A` is known to be symmetric / Hermitian
    prefer :func:`eigh` — it is faster, more stable, and produces real
    eigenvalues with orthogonal eigenvectors.

    Parameters
    ----------
    input : Tensor
        Square matrix of shape ``(*, n, n)``.

    Returns
    -------
    eigenvalues : Tensor
        Tensor of shape ``(*, n)``.  Real for real-spectrum matrices,
        complex otherwise.
    eigenvectors : Tensor
        Tensor of shape ``(*, n, n)`` whose columns are the right
        eigenvectors.

    Notes
    -----
    Backed by LAPACK ``geev``.  Cost is :math:`O(n^3)`.  This op
    currently has **no autograd support** — gradients through
    eigendecomposition of a general matrix are notoriously unstable
    near defective spectra and not implemented.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import eig
    >>> A = lucid.tensor([[2.0, 0.0], [0.0, 3.0]])
    >>> w, V = eig(A)
    >>> w
    tensor([(2.+0.j), (3.+0.j)], dtype=lucid.complex64)
    """
    with _autocast_entry("eig", _unwrap(input)) as (a_impl,):
        A_cast = _wrap(a_impl)
        if A_cast.requires_grad and lucid.is_grad_enabled():
            w, V = cast(tuple[Tensor, Tensor], _Eig.apply(A_cast))
            return _EigResult(w, V)
        raw_vals, raw_vecs = _la.eig(a_impl)
        return _EigResult(_wrap(raw_vals), _wrap(raw_vecs))


def eigvals(input: Tensor) -> Tensor:
    r"""Compute only the eigenvalues of a general square matrix.

    Returns the roots :math:`\lambda_1, \ldots, \lambda_n` of the
    characteristic polynomial :math:`\det(A - \lambda I) = 0` without
    computing the eigenvectors.  Equivalent to ``eig(A)[0]`` but skips
    the eigenvector assembly.

    Parameters
    ----------
    input : Tensor
        Square matrix of shape ``(*, n, n)``.

    Returns
    -------
    Tensor
        Eigenvalues of shape ``(*, n)``.

    Notes
    -----
    Cost is :math:`O(n^3)` and dominated by the Hessenberg reduction.
    No autograd support — use :func:`eigvalsh` for symmetric / Hermitian
    matrices when gradients are required.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import eigvals
    >>> eigvals(lucid.tensor([[2.0, 0.0], [0.0, 3.0]]))
    tensor([(2.+0.j), (3.+0.j)], dtype=lucid.complex64)
    """
    with _autocast_entry("eigvals", _unwrap(input)) as (a_impl,):
        A_cast = _wrap(a_impl)
        if A_cast.requires_grad and lucid.is_grad_enabled():
            return cast(tuple[Tensor, Tensor], _Eig.apply(A_cast))[0]
        vals, _ = _la.eig(a_impl)
        return _wrap(vals)


def _complex_matmul(a: Tensor, b: Tensor) -> Tensor:
    """``a @ b`` for complex operands, from four real products.

    The CPU matmul has no complex kernel; spelled in real parts it runs on
    both devices and differentiates like any other product.
    """
    ar, ai = lucid.real(a), lucid.imag(a)
    br, bi = lucid.real(b), lucid.imag(b)
    return lucid.complex(
        lucid.matmul(ar, br) - lucid.matmul(ai, bi),
        lucid.matmul(ar, bi) + lucid.matmul(ai, br),
    )


def _complex_solve(A: Tensor, B: Tensor) -> Tensor:
    """``A⁻¹ B`` for complex operands, as the real system of twice the size.

    ``(Ar + i Ai)(Xr + i Xi) = Br + i Bi`` is ``[[Ar, -Ai], [Ai, Ar]]`` times
    ``[Xr; Xi]`` equal to ``[Br; Bi]``.
    """
    n = int(A.shape[-1])
    Ar, Ai = lucid.real(A), lucid.imag(A)
    M = lucid.cat([lucid.cat([Ar, -Ai], dim=-1), lucid.cat([Ai, Ar], dim=-1)], dim=-2)
    rhs = lucid.cat([lucid.real(B), lucid.imag(B)], dim=-2)
    X = solve(M, rhs)
    return lucid.complex(X[..., :n, :], X[..., n:, :])


@final
class _Eig(_AutogradFunction):
    """``eig`` with the reference framework's backward.

    The kernel recorded no gradient, so a loss through ``eig`` or
    ``eigvals`` trained nothing.  For ``A = V diag(w) V⁻¹`` and incoming
    ``gw``, ``gV``, with ``E[i, j] = conj(w_j - w_i)``::

        dA = V⁻ᴴ (diag(gw) + offdiag((Vᴴ gV − Vᴴ V diag(Re diag(Vᴴ gV))) / E)) Vᴴ

    — the middle term removes the part of ``gV`` that only rescales an
    eigenvector, which the unit-norm convention does not let move.  A real
    input takes the real part.  Eigenvectors are defined up to a phase, so
    only a loss that does not depend on it (``abs``, ``V diag(w) V⁻¹``) has
    a meaningful gradient, as with the reference.
    """

    @override
    @staticmethod
    def forward(ctx: FunctionCtx, A: Tensor) -> tuple[Tensor, Tensor]:
        vals, vecs = _la.eig(_unwrap(A))
        w, V = _wrap(vals), _wrap(vecs)
        ctx.save_for_backward(w, V)
        ctx.real_input = not A.is_complex()
        return w, V

    @override
    @staticmethod
    def backward(ctx: FunctionCtx, gw: Tensor, gV: Tensor) -> Tensor:
        w, V = ctx.saved_tensors
        n = int(V.shape[-1])
        real_dtype = lucid.float64 if V.dtype == lucid.complex128 else lucid.float32
        eye = lucid.eye(n, dtype=real_dtype, device=V.device)
        eye_c = lucid.complex(eye, lucid.zeros_like(eye))
        Vh = lucid.conj(V).mT
        VhgV = _complex_matmul(Vh, gV)
        scale = lucid.real(lucid.diagonal(VhgV, 0, -2, -1))
        scale_c = lucid.complex(scale, lucid.zeros_like(scale))
        inner = VhgV - _complex_matmul(Vh, V * scale_c.unsqueeze(-2))
        wc = lucid.conj(w)
        E = wc.unsqueeze(-2) - wc.unsqueeze(-1) + eye_c
        inner = inner / E * (1 - eye_c) + eye_c * gw.unsqueeze(-2)
        gA = _complex_solve(Vh, _complex_matmul(inner, Vh))
        if cast(bool, ctx.real_input):
            return lucid.real(gA)
        return gA


# ── Eigh with backward ────────────────────────────────────────────────────────
#
# For symmetric A = V diag(w) V^T, given G_w and G_V:
#   F[i,j] = 1/(w_i − w_j)  for i≠j,  F[i,i] = 0    (Loewner matrix)
#   dA from w: V diag(G_w) V^T
#   dA from V: V (F ⊙ (V^T G_V)) V^T  (then symmetrised)
#
# Split into two Function wrappers so the engine accumulates contributions.


@final
class _Eigh(_AutogradFunction):
    """Symmetric / Hermitian eigendecomposition with Loewner backward."""

    _materialize_grads = False

    @override
    @staticmethod
    def forward(ctx: FunctionCtx, A: Tensor, UPLO: str) -> tuple[Tensor, Tensor]:
        w_impl, V_impl = _la.eigh(_unwrap(A), UPLO)
        w, V = _wrap(w_impl), _wrap(V_impl)
        ctx.save_for_backward(w, V)
        return w, V

    @override
    @staticmethod
    def backward(ctx: FunctionCtx, G_w: Tensor | None, G_V: Tensor | None) -> Tensor:
        w, V = ctx.saved_tensors
        dA: Tensor | None = None
        if G_w is not None:
            dA = V @ lucid.diag_embed(G_w) @ V.mT
        if G_V is not None:
            k = int(w.shape[-1])
            Wi = w.unsqueeze(-1)
            Wj = w.unsqueeze(-2)
            # F[i, j] = 1 / (w[j] - w[i]).
            denom = Wj - Wi
            eye_k = lucid.eye(k, dtype=w.dtype, device=w.device)
            safe_denom = denom + eye_k
            F = (1.0 / safe_denom) * (1.0 - eye_k)
            VtGV = V.mT @ G_V
            inner = F * VtGV
            dA_V = V @ inner @ V.mT
            dA_V = (dA_V + dA_V.mT) * 0.5
            dA = dA_V if dA is None else dA + dA_V
        assert dA is not None
        return dA


def eigh(input: Tensor, UPLO: str = "L") -> tuple[Tensor, Tensor]:
    r"""Eigendecomposition of a Hermitian / symmetric matrix.

    Returns the eigenvalues and orthonormal eigenvectors of a real
    symmetric (or complex Hermitian) matrix :math:`A`.  Eigenvalues are
    returned in **ascending order** and eigenvectors form an orthogonal
    matrix :math:`V` such that

    .. math::

        A \,=\, V \,\Lambda\, V^\top,

    where :math:`\Lambda = \mathrm{diag}(\lambda_1, \ldots, \lambda_n)`
    with :math:`\lambda_1 \le \cdots \le \lambda_n`.

    Parameters
    ----------
    input : Tensor
        Square Hermitian / symmetric matrix of shape ``(*, n, n)``.
    UPLO : str, optional
        ``"L"`` (default) reads only the lower triangle of ``input``;
        ``"U"`` reads only the upper triangle.  The other triangle is
        ignored, so a non-Hermitian input is accepted as long as the
        chosen triangle holds the correct values.

    Returns
    -------
    eigenvalues : Tensor
        Real-valued tensor of shape ``(*, n)`` in ascending order.
    eigenvectors : Tensor
        Tensor of shape ``(*, n, n)`` whose columns are the
        corresponding orthonormal eigenvectors.

    Notes
    -----
    Prefer this over :func:`eig` whenever ``A`` is symmetric / Hermitian
    — it is faster, numerically more stable, and guarantees real
    eigenvalues with orthogonal eigenvectors (:math:`V^\top V = I`).

    Backward uses the Loewner-matrix formula
    :math:`F_{ij} = 1/(\lambda_i - \lambda_j)`:

    .. math::

        \frac{\partial \mathcal{L}}{\partial A}
        \,=\, V\,\big(\mathrm{diag}(G_\lambda) + F \odot (V^\top G_V)\big)\,V^\top,

    symmetrised to enforce the symmetry of :math:`A`.  Gradients blow up
    near repeated eigenvalues.

    Implementation: LAPACK ``syevd`` on the CPU stream (via Apple
    Accelerate); MLX on the GPU stream.  Both run in :math:`O(n^3)`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import eigh
    >>> A = lucid.tensor([[2.0, 1.0], [1.0, 3.0]])
    >>> w, V = eigh(A)
    >>> w
    tensor([1.382, 3.618])
    """
    with _autocast_entry("eigh", _unwrap(input)) as (a_impl,):
        A_cast = _wrap(a_impl)
        if not _C_engine.grad_enabled() or not A_cast.requires_grad:
            w_impl, V_impl = _la.eigh(a_impl, UPLO)
            return _EighResult(_wrap(w_impl), _wrap(V_impl))
        out = _Eigh.apply(A_cast, UPLO)
        assert isinstance(out, tuple)
        w, V = out
        return _EighResult(w, V)


def eigvalsh(input: Tensor, UPLO: str = "L") -> Tensor:
    r"""Eigenvalues of a Hermitian / symmetric matrix.

    Returns only the real eigenvalues :math:`\lambda_1 \le \cdots \le
    \lambda_n` of a symmetric (real) or Hermitian (complex) matrix
    without forming the eigenvectors.  Equivalent to ``eigh(A)[0]``
    but skips eigenvector assembly when no gradients are requested.

    Parameters
    ----------
    input : Tensor
        Symmetric / Hermitian matrix of shape ``(*, n, n)``.
    UPLO : str, optional
        ``"L"`` reads the lower triangle (default), ``"U"`` the upper.

    Returns
    -------
    Tensor
        Eigenvalues in ascending order, shape ``(*, n)``.

    Notes
    -----
    When ``input.requires_grad`` is true the call routes through
    :func:`eigh` so that backward via the Loewner formula remains
    available.  Otherwise the engine kernel is invoked directly.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import eigvalsh
    >>> eigvalsh(lucid.tensor([[2.0, 1.0], [1.0, 3.0]]))
    tensor([1.382, 3.618])
    """
    with _autocast_entry("eigvalsh", _unwrap(input)) as (a_impl,):
        A_cast = _wrap(a_impl)
        if _C_engine.grad_enabled() and A_cast.requires_grad:
            w, _ = eigh(A_cast, UPLO)
            return w
        vals, _ = _la.eigh(a_impl, UPLO)
        return _wrap(vals)


def lu_factor(A: Tensor, *, pivot: bool = True) -> tuple[Tensor, Tensor]:
    r"""LU factorization with partial pivoting (packed form).

    Computes the packed LU factorization

    .. math::

        P\,A \,=\, L\,U,

    where :math:`P` is a row-permutation matrix, :math:`L` is
    unit-lower-triangular, and :math:`U` is upper-triangular.  The
    result is returned in LAPACK's packed format suitable for repeated
    solves via :func:`lu_solve`.

    Parameters
    ----------
    A : Tensor
        Matrix of shape ``(*, m, n)``.  Need not be square.
    pivot : bool, keyword-only, optional
        Must be ``True`` (the default).  Unpivoted LU is not implemented
        and raises ``NotImplementedError``.

    Returns
    -------
    LU : Tensor
        Packed factorization of shape ``(*, m, n)``.  :math:`U`
        occupies the diagonal and above; :math:`L` (without its unit
        diagonal) occupies the strict lower triangle.
    pivots : Tensor
        ``int32`` tensor of shape ``(*, min(m, n))`` containing 1-based
        pivot indices (LAPACK convention) — one per elimination step,
        and there are only as many steps as the shorter side.

    Notes
    -----
    Backed by LAPACK ``getrf``.  Cost is :math:`O(\tfrac{2}{3} n^3)` for
    a square input.

    For a rectangular :math:`A \in \mathbb{R}^{m \times n}` with
    :math:`k = \min(m, n)`, :math:`L` is :math:`m \times k`
    unit-lower-trapezoidal and :math:`U` is :math:`k \times n`
    upper-trapezoidal.  :func:`lu_solve` still requires a square factor:
    only a square system has a solution.

    Use :func:`lu_factor` + :func:`lu_solve` instead of :func:`solve`
    when the same coefficient matrix :math:`A` is reused with many
    right-hand sides — factorization is shared.  For the explicit
    :math:`(P, L, U)` triple see :func:`lu`.

    Raises
    ------
    LinAlgError
        If a pivot :math:`U_{kk}` is exactly zero — :func:`lu_solve` would
        divide by it.  :func:`lu_factor_ex` returns the factors of such a
        matrix with a non-zero ``info`` instead.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import lu_factor, lu_solve
    >>> A = lucid.tensor([[2.0, 1.0], [4.0, 7.0]])
    >>> LU, piv = lu_factor(A)
    >>> b = lucid.tensor([[3.0], [13.0]])
    >>> lu_solve(LU, piv, b)
    tensor([[0.8], [1.4]])
    """
    if not pivot:
        raise NotImplementedError("lu_factor: pivot=False is not supported")
    LU, pivots, _ = _lu_factor_ex(A, check_errors=True, op="lu_factor")
    return _LuFactorResult(LU, pivots)


def _lu_factor_packed(A: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """``getrf``'s packed factor (differentiable in square ``A``), pivots and
    ``info``.  A singular matrix is factored and reported, never raised;
    whether that is an error is the caller's call."""
    lu, pivots, info = _la.lu_factor(_unwrap(A))
    LU, piv = _wrap(lu), _wrap(pivots)
    if A.requires_grad and lucid.is_grad_enabled():
        n = int(A.shape[-1])
        if int(A.shape[-2]) != n:
            raise NotImplementedError(
                "lu_factor: the gradient of a non-square factorization is not "
                "implemented — factor a square matrix, or call it under "
                "lucid.no_grad()"
            )
        P = _build_permutation_matrix(piv, n, A.dtype, A.device)
        LU = cast(Tensor, _LUFactorGrad.apply(A, LU, P))
    return LU, piv, _wrap(info)


@final
class _LUFactorGrad(_AutogradFunction):
    """The packed factor of ``A = P L U``, differentiable in square ``A``.

    The kernel recorded no gradient, so a loss through ``lu_factor`` — or
    through ``lu``, ``det`` written by hand from it — trained nothing.  With
    ``dL = tril(G, -1)`` and ``dU = triu(G)`` read off the packed gradient,
    the reference framework's backward is::

        dA = P · L⁻ᵀ · (tril(Lᵀ dL, -1) + triu(dU Uᵀ)) · U⁻ᵀ

    Both triangular solves go through the differentiable solve, and ``L``
    and ``U`` are read from this Function's own output, so the result
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
        n = int(LU.shape[-1])
        eye = lucid.eye(n, dtype=LU.dtype, device=LU.device)
        L = lucid.tril(LU, -1) + eye
        U = lucid.triu(LU)
        M = lucid.tril(lucid.matmul(L.mT, lucid.tril(grad_out, -1)), -1) + lucid.triu(
            lucid.matmul(lucid.triu(grad_out), U.mT)
        )
        X = solve_triangular(U.mT, M, upper=False, left=False)
        Y = solve_triangular(L.mT, X, upper=True, unitriangular=True)
        return lucid.matmul(P, Y), None, None


# ── Pure-Python compositions ───────────────────────────────────────────────────


def slogdet(A: Tensor) -> tuple[Tensor, Tensor]:
    r"""Sign and natural log of the absolute determinant.

    Returns the pair :math:`(\mathrm{sign},\, \log|\det A|)` such that

    .. math::

        \det(A) \,=\, \mathrm{sign} \cdot \exp(\log|\det A|).

    Numerically stable for matrices whose determinant would overflow
    or underflow if computed directly — for instance, large covariance
    matrices in log-likelihood calculations.

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
    Computed via LU as :math:`\log|\det A| = \sum_i \log|U_{ii}|` with
    the sign accumulated from row swaps.  Cost is :math:`O(n^3)`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import slogdet
    >>> sign, logabs = slogdet(lucid.tensor([[1.0, 2.0], [3.0, 4.0]]))
    >>> sign, logabs
    (tensor(-1.), tensor(0.6931))
    """
    d = det(A)
    di, z, ai = _unwrap(d), _C_engine.zeros_like(_unwrap(d)), _unwrap(A)
    with _C_engine.NoGradGuard():
        ds = _unwrap(det(_wrap(_C_engine.add(ai, _C_engine.zeros_like(ai)))))
    eq = _C_engine.equal(di, z)
    sign = _wrap(_C_engine.where(eq, _C_engine.mul(ds, z), _C_engine.sign(di)))
    return _SlogdetResult(sign, cast(Tensor, _LogAbsDet.apply(d)))


@final
class _LogAbsDet(_AutogradFunction):
    """``log|d|`` of a determinant, with derivative ``1 / d``.

    Composed as ``log(abs(d))`` the chain rule read ``sign(d) / |d|``, and
    at a singular matrix that is ``0 · inf`` — a NaN gradient where the
    reference answers ``±inf``.  ``1 / d`` is the same derivative wherever
    ``d ≠ 0`` and keeps the sign of a signed zero, so through ``det``'s own
    backward the gradient is ``g · cof(A) / det(A)``: ``g · A⁻ᵀ`` for an
    invertible ``A`` and ``±inf`` along the adjugate of a singular one.
    """

    @override
    @staticmethod
    def forward(ctx: FunctionCtx, d: Tensor) -> Tensor:
        ctx.save_for_backward(d)
        return _wrap(_C_engine.log(_C_engine.abs(_unwrap(d))))

    @override
    @staticmethod
    def backward(ctx: FunctionCtx, grad_out: Tensor) -> Tensor:
        (d,) = ctx.saved_tensors
        return grad_out / d


def matrix_rank(
    input: Tensor,
    tol: float | Tensor | None = None,
    hermitian: bool = False,
    *,
    atol: float | Tensor | None = None,
    rtol: float | Tensor | None = None,
) -> Tensor:
    r"""Compute the numerical rank of a matrix or a batch of matrices.

    Counts singular values (or eigenvalue magnitudes when ``hermitian=True``)
    above :math:`\tau = \max(\text{atol},\, \sigma_{\max} \cdot \text{rtol})`:

    .. math::

        \mathrm{rank}(A) \,=\, |\{\,i : \sigma_i > \tau\,\}|.

    Parameters
    ----------
    input : Tensor
        Input matrix of shape ``(*, m, n)``.
    tol : float, Tensor or None, optional
        Alias for ``atol`` with ``rtol=0``; mutually exclusive with ``atol``/``rtol``.
    hermitian : bool, optional
        If ``True``, use :func:`eigvalsh` of the lower triangle instead of :func:`svdvals`.
    atol : float, Tensor or None, keyword-only, optional
        Absolute tolerance (treated as ``0.0`` when ``None``).
    rtol : float, Tensor or None, keyword-only, optional
        Relative tolerance; defaults to :math:`\max(m, n) \cdot \varepsilon`
        unless ``atol > 0``, in which case ``0.0``.

    Returns
    -------
    Tensor
        ``int64`` tensor of numerical ranks per matrix.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import matrix_rank
    >>> matrix_rank(lucid.tensor([[1.0, 2.0], [2.0, 4.0]]))
    tensor(1, dtype=lucid.int64)
    """
    if input.ndim < 2:
        raise ValueError(
            f"matrix_rank: input must have at least 2 dimensions, got shape {tuple(input.shape)}"
        )
    if tol is not None:
        if atol is not None or rtol is not None:
            raise ValueError("matrix_rank: tol cannot be specified together with atol or rtol")
        atol, rtol = tol, 0.0
    with _autocast_entry("matrix_rank", _unwrap(input)) as (a_impl,):
        A_cast = _wrap(a_impl)
        with lucid.no_grad():
            sv = eigvalsh(A_cast).abs() if hermitian else svdvals(A_cast)
            if sv.numel() == 0:
                return lucid.zeros(
                    tuple(A_cast.shape[:-2]), dtype=lucid.int64, device=A_cast.device
                )
            return (sv > _sv_cutoff(A_cast, sv, atol=atol, rtol=rtol)).sum(dim=-1)


_COND_ORDERS: tuple[int | float | str | None, ...] = (
    None,
    2,
    -2,
    1,
    -1,
    float("inf"),
    float("-inf"),
    "fro",
    "nuc",
)


def cond(input: Tensor, p: int | float | str | None = None) -> Tensor:
    r"""Compute the condition number of a matrix.

    The condition number under norm :math:`\|\cdot\|_p` is

    .. math::

        \kappa_p(A) \,=\, \|A\|_p \,\|A^{-1}\|_p.

    For the spectral (:math:`p = 2`) norm this simplifies to the ratio
    of the largest to smallest singular value:

    .. math::

        \kappa_2(A) \,=\, \sigma_{\max}(A) \,/\, \sigma_{\min}(A).

    The condition number quantifies how sensitive the solution of
    :math:`Ax = b` is to perturbations in :math:`A` or :math:`b`.

    Parameters
    ----------
    input : Tensor
        Input matrix of shape ``(*, m, n)``.
    p : int, float, str or None, optional
        Norm order: ``None`` (default) or ``2`` for the spectral norm,
        ``-2`` for its reciprocal :math:`\sigma_{\min} / \sigma_{\max}`,
        ``"nuc"``, or one of the :func:`matrix_norm` orders ``"fro"``,
        ``1``, ``-1``, ``inf``, ``-inf``.

    Returns
    -------
    Tensor
        Condition number, shape ``(*,)``.

    Raises
    ------
    ValueError
        For an unknown ``p``, an input with fewer than two dimensions, or a
        non-square input under any order but :math:`\pm 2`.
    NotImplementedError
        For a complex ``input``: every order needs its singular values or its
        inverse, and neither has a complex kernel yet.

    Notes
    -----
    A condition number near :math:`1/\varepsilon_{\mathrm{mach}}`
    indicates numerical singularity.  Non-spectral orders require an
    explicit inverse, so prefer ``p = 2`` for rank-deficient or
    rectangular matrices.

    The spectral orders and ``"nuc"`` are read off the singular values;
    the others are :math:`\|A\|_p \|A^{-1}\|_p` with the inverse from
    :func:`inv_ex`, so a singular matrix has condition number
    :math:`\infty` rather than an error, each matrix of a batch on its
    own.  A matrix with no entries has condition number ``0``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import cond
    >>> cond(lucid.tensor([[1.0, 0.0], [0.0, 1e-6]]))
    tensor(1e+06)
    >>> cond(lucid.tensor([[1.0, 2.0], [2.0, 4.0]]), "fro")
    tensor(inf)
    """
    if p not in _COND_ORDERS:
        raise ValueError(
            f"cond: unsupported order p={p!r}; expected None, ±1, ±2, ±inf, " f"'fro' or 'nuc'"
        )
    if input.ndim < 2:
        raise ValueError(
            f"cond: input must have at least 2 dimensions, got shape {tuple(input.shape)}"
        )
    square = input.shape[-1] == input.shape[-2]
    if isinstance(p, str) and not square:
        raise ValueError(
            f"cond(p={p!r}): input must be a square matrix or a batch of them, got "
            f"shape {tuple(input.shape)}"
        )
    if input.is_complex():
        needs = "SVD" if p is None or p in (2, -2, "nuc") else "inverse"
        raise NotImplementedError(
            f"cond(p={p!r}): a complex A needs a complex {needs}, which linalg " "does not have yet"
        )
    with _autocast_entry("cond", _unwrap(input)) as (a_impl,):
        return _cond_compute(_wrap(a_impl), p, square)


def multi_dot(tensors: list[Tensor]) -> Tensor:
    r"""Multiply a sequence of matrices as a single chained product.

    Computes the product of a list of matrices

    .. math::

        A_1 \, A_2 \, \cdots \, A_n,

    associating left-to-right.  Optimal parenthesization can
    substantially reduce flops for chains with widely varying inner
    dimensions; the current implementation associates left-to-right
    (the most common case is already locally optimal).

    Parameters
    ----------
    tensors : list of Tensor
        Sequence of at least one matrix.  Inner dimensions must agree
        (``tensors[i].shape[-1] == tensors[i+1].shape[-2]``).

    Returns
    -------
    Tensor
        The chained matrix product.

    Notes
    -----
    For long chains, choosing the optimal split can reduce work from
    :math:`O(\sum_i d_i d_{i+1} d_{i+2})` (left-to-right) to a
    significantly smaller bound found via dynamic programming.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import multi_dot
    >>> A = lucid.tensor([[1.0, 2.0]])
    >>> B = lucid.tensor([[3.0], [4.0]])
    >>> C = lucid.tensor([[5.0]])
    >>> multi_dot([A, B, C])
    tensor([[55.]])
    """
    if len(tensors) == 0:
        raise ValueError("multi_dot requires at least one tensor")
    if len(tensors) == 1:
        return tensors[0]
    result = _wrap(_C_engine.matmul(_unwrap(tensors[0]), _unwrap(tensors[1])))
    for t in tensors[2:]:
        result = _wrap(_C_engine.matmul(_unwrap(result), _unwrap(t)))
    return result


def solve_triangular(
    input: Tensor,
    B: Tensor,
    *,
    upper: bool,
    left: bool = True,
    unitriangular: bool = False,
) -> Tensor:
    r"""Solve a triangular linear system by back/forward substitution.

    Solves the system :math:`AX = B` (or :math:`XA = B`) in which the
    coefficient matrix :math:`A` is triangular.  For upper-triangular
    :math:`A` the system is solved by back-substitution starting from
    the last row; for lower-triangular :math:`A` by forward
    substitution from the first row.  Either direction runs in
    :math:`O(n^2 k)` time and is numerically stable when the diagonal
    of :math:`A` is well-conditioned.

    Parameters
    ----------
    input : Tensor
        Triangular coefficient matrix of shape ``(*, n, n)``.  Only
        the relevant triangle is read; the other half is ignored.
    B : Tensor
        Right-hand side ``(*, n, k)`` or ``(n,)``; batch dims broadcast with ``input``.
    upper : bool, keyword-only
        If ``True`` :math:`A` is upper-triangular; if ``False``
        lower-triangular.
    left : bool, keyword-only, optional
        If ``True`` (default) solves :math:`AX = B`; if ``False``
        solves :math:`XA = B` via transposition.
    unitriangular : bool, keyword-only, optional
        If ``True`` the diagonal of :math:`A` is treated as all-ones
        regardless of its stored values (LAPACK's "unit-diagonal"
        mode).

    Returns
    -------
    Tensor
        Solution :math:`X` — the broadcast batch, then ``(n, k)`` (or ``(n,)``).

    Notes
    -----
    Backed by BLAS ``trsm`` on both devices.  Triangular solves are the
    workhorse used inside Cholesky, LU, and QR back-substitution paths.

    A singular :math:`A` (an exact zero on the diagonal, with
    ``unitriangular=False``) is not refused: substitution divides by the
    zero, so the solution holds :math:`\pm\infty`, or NaN where it meets
    :math:`0 / 0`, as the reference framework does.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import solve_triangular
    >>> A = lucid.tensor([[2.0, 1.0], [0.0, 3.0]])  # upper
    >>> b = lucid.tensor([[5.0], [9.0]])
    >>> solve_triangular(A, b, upper=True)
    tensor([[1.], [3.]])
    """
    # The engine owns the right-hand-side contract (batch broadcast, vector
    # right-hand side) and the gradient in both operands.
    if not left:
        # X A = B  ⟺  Aᵀ Xᵀ = Bᵀ — solve the transposed system, transpose result.
        XT = _la.solve_triangular(_unwrap(input.mT), _unwrap(B.mT), not upper, unitriangular)
        return _wrap(XT).mT
    return _wrap(_la.solve_triangular(_unwrap(input), _unwrap(B), upper, unitriangular))


def vander(x: Tensor, *, N: int | None = None, increasing: bool = True) -> Tensor:
    r"""Construct a Vandermonde matrix from a 1-D or batched vector.

    Given an input :math:`x` of shape ``(*, n)``, returns the
    ``(*, n, N)`` tensor whose :math:`j`-th column is a power of
    :math:`x`:

    .. math::

        V_{\ldots ij} \,=\, x_{\ldots i}^{\,j} \quad (\text{increasing, default})
        \qquad\text{or}\qquad
        V_{\ldots ij} \,=\, x_{\ldots i}^{\,N-1-j} \quad (\text{decreasing}).

    Vandermonde matrices arise naturally in polynomial fitting and
    interpolation: the columns are the basis :math:`\{1, x, x^2,
    \ldots\}` evaluated at the data points.

    Parameters
    ----------
    x : Tensor
        Input tensor of shape ``(*, n)``; a 0-d tensor is a vector of one.
        Floating, complex or integer; an integer input stays integral.
    N : int or None, keyword-only, optional
        Number of columns.  Defaults to :math:`n` (square output per vector).
    increasing : bool, keyword-only, optional
        Default ``True``: column 0 is :math:`x^0`, as the reference orders
        it.  ``False`` reverses the columns.

    Returns
    -------
    Tensor
        ``(*, n, N)`` Vandermonde tensor; ``int64`` for an integer input.

    Raises
    ------
    TypeError
        If ``x`` is a bool tensor.

    Notes
    -----
    The powers are a running product of ``x``, as the reference builds
    them, so they are exact for an integer input and the derivative at
    :math:`x = 0` is the exact one.  The default was decreasing powers.

    Vandermonde matrices become highly ill-conditioned as
    :math:`N` grows (condition number grows exponentially) — for
    polynomial regression beyond degree :math:`\sim 10` prefer an
    orthogonal-polynomial basis or QR-based fitting.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import vander
    >>> vander(lucid.tensor([1.0, 2.0, 3.0]))
    tensor([[1., 1., 1.], [1., 2., 4.], [1., 3., 9.]])
    """
    if x.dtype == lucid.bool:
        raise TypeError("linalg.vander: expected a numeric input, got lucid.bool")
    vec = x.unsqueeze(-1) if x.ndim == 0 else x
    shape = tuple(vec.shape)
    cols = shape[-1] if N is None else int(N)
    if cols < 0:
        raise _C_engine.InvalidArgument(f"linalg.vander: N must be non-negative, got {cols}")
    powers = lucid.cumprod(vec.unsqueeze(-1).expand(*shape, max(cols - 1, 0)), -1)
    ones = lucid.ones(*shape, min(cols, 1), dtype=powers.dtype, device=vec.device)
    out = lucid.cat([ones, powers], -1)
    return out if increasing else out.flip(-1)


_COMPLEX_DTYPES = (lucid.complex64, lucid.complex128)


def _cast_for_norm(x: Tensor, dtype: object, op: str = "vector_norm") -> _C_engine.TensorImpl:
    """``x`` cast to the ``dtype`` a norm was asked to compute in.

    The argument used to be accepted and dropped: ``dtype=float64`` on a
    float32 input overflowed to ``inf`` in float32 and came back float32.
    The reference's rules come with it — the kind (real / complex) must
    match the input's, and the cast must not narrow.
    """
    target = _ENGINE_TO_DTYPE[to_engine_dtype(cast(lucid.dtype | str, dtype))]
    complex_in = x.dtype in _COMPLEX_DTYPES
    if complex_in != (target in _COMPLEX_DTYPES):
        kind = "complex" if complex_in else "real"
        raise TypeError(f"{op}: dtype must be {kind} for a {kind} input, got {target}")
    if lucid.promote_types(x.dtype, target) != target:
        raise TypeError(
            f"{op}: dtype={target} would narrow the {x.dtype} input; "
            f"pass a dtype it converts to without loss, or cast it first"
        )
    xi = _unwrap(x)
    engine_dtype = to_engine_dtype(target)
    return xi if xi.dtype == engine_dtype else _C_engine.astype(xi, engine_dtype)


def vector_norm(
    input: Tensor,
    ord: int | float = 2,
    dim: int | list[int] | tuple[int, ...] | None = None,
    keepdim: bool = False,
    *,
    dtype: object = None,
) -> Tensor:
    r"""Compute a vector :math:`p`-norm along an axis.

    Reduces along ``dim`` (or all elements if ``dim is None``) using

    .. math::

        \|x\|_p \,=\, \Big(\sum_i |x_i|^{\,p}\Big)^{1/p}

    for any positive real :math:`p`.  Special-cased values:

    * :math:`p = 0` — count of non-zero entries (not a true norm).
    * :math:`p = 1` — :math:`\sum_i |x_i|`.
    * :math:`p = 2` — Euclidean norm.
    * :math:`p = +\infty` — :math:`\max_i |x_i|`.
    * :math:`p = -\infty` — :math:`\min_i |x_i|`.

    Parameters
    ----------
    input : Tensor
        Input tensor.
    ord : int or float, optional
        Order of the norm.  Default ``2``.
    dim : int, list of int, tuple of int or None, optional
        Axis or axes to reduce.  ``None`` reduces over every axis — the
        rank is retained, so ``keepdim=True`` yields one size-1 axis per
        input axis rather than a single one.
    keepdim : bool, optional
        If ``True``, reduced dimensions are retained with size 1.
    dtype : dtype, keyword-only, optional
        Cast ``input`` to this dtype before reducing, so the whole computation
        — and the result — is in it; a complex ``dtype`` gives its real
        counterpart.  It must be complex for a complex ``input`` and real
        otherwise, and must not narrow ``input``.  ``None`` (default) reduces
        in ``input``'s own dtype.

    Returns
    -------
    Tensor
        Norm along the specified axes — real, also for a complex ``input``.

    Raises
    ------
    TypeError
        If ``input`` is an integer or bool tensor (a ``DtypeMismatch``), or if
        ``dtype`` would narrow it or its kind (real / complex) differs.
    ValueError
        If ``dim`` names one axis twice.

    Notes
    -----
    All operations are routed through autograd-aware engine kernels, so
    gradients flow naturally even for non-integer :math:`p` (via
    :math:`p`-power and root).  Where the norm is zero the gradient is
    zero — the subgradient the reference picks — rather than the ``nan``
    the root's infinite slope there would give; so is a zero entry's for
    :math:`p < 1`.

    A ``float16`` or ``bfloat16`` input is reduced in ``float32`` and the
    result cast back, so a norm that fits the input dtype is not lost to
    an intermediate sum that does not: the 2-norm of 65 536 half-precision
    ones is 256, not ``inf``.

    A complex ``input`` is reduced through its magnitudes :math:`|x_i|`, as
    the definition says.  Squaring the entries themselves gave a complex
    "norm" for ``ord=2`` on Metal and an error on the CPU.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import vector_norm
    >>> vector_norm(lucid.tensor([3.0, 4.0]))
    tensor(5.)
    >>> vector_norm(lucid.tensor([1.0, -2.0, 3.0]), ord=1)
    tensor(6.)
    >>> vector_norm(lucid.tensor([3.0, 4.0]), dtype=lucid.float64)
    tensor(5., dtype=lucid.float64)
    >>> vector_norm(lucid.tensor([3 + 4j, 0j]))
    tensor(5.)
    """
    xi = _require_normable(_unwrap(input), "vector_norm")
    with _autocast_entry("vector_norm", xi) as (xi_cast,):
        xi = _cast_for_norm(input, dtype, "vector_norm") if dtype is not None else xi_cast
        # Every order below is a function of the magnitudes |x_i| alone, and
        # this is the one place in linalg where a norm takes them: a complex
        # input is replaced by its (real) magnitudes before any order sees it.
        # ``matrix_norm``'s entry-wise orders and ``norm``'s dispatch reduce
        # through this function instead of repeating the step (CHA-147, CHA-231).
        r = _C_engine.abs(xi) if xi.dtype in (_C_engine.C64, _C_engine.C128) else xi
        # Every axis, rather than a flatten, for ``dim=None`` so ``keepdim`` retains rank.
        axes = list(range(len(r.shape))) if dim is None else _reduction_axes(r, dim, "vector_norm")
        if ord == 0:
            nz = _C_engine.not_equal(r, _C_engine.zeros_like(r))
            return _wrap(_C_engine.astype(_C_engine.sum(nz, axes, keepdim), r.dtype))
        return _wrap(_p_norm(r, float(ord), axes, keepdim))


def cross(input: Tensor, other: Tensor, *, dim: int = -1) -> Tensor:
    r"""Compute the cross product of two 3-D vectors.

    Returns the standard 3-D vector cross product

    .. math::

        x \times y \,=\, \big(x_2 y_3 - x_3 y_2,\;
                              x_3 y_1 - x_1 y_3,\;
                              x_1 y_2 - x_2 y_1\big).

    Parameters
    ----------
    input : Tensor
        First operand.  Must have size 3 along ``dim``.
    other : Tensor
        Second operand.  Same shape as ``input``.
    dim : int, keyword-only, optional
        Axis along which the 3 components are stored.  Default ``-1``.

    Returns
    -------
    Tensor
        Cross product, same shape as inputs.

    Notes
    -----
    Implemented via :func:`gather` + element-wise products so that
    autograd flows through naturally.  Recall that
    :math:`x \times y` is orthogonal to both :math:`x` and :math:`y`,
    with magnitude :math:`\|x\|\,\|y\|\,\sin\theta`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import cross
    >>> cross(lucid.tensor([1.0, 0.0, 0.0]), lucid.tensor([0.0, 1.0, 0.0]))
    tensor([0., 0., 1.])
    """
    xi, yi = _unwrap(input), _unwrap(other)
    if len(xi.shape) != len(yi.shape):
        raise _C_engine.ShapeMismatch(
            "linalg.cross: inputs must have the same number of dimensions"
        )
    d = _wrap_dim(dim, len(xi.shape), "linalg.cross")
    if not xi.shape:
        raise IndexError(f"linalg.cross: axis out of range: got {dim} for a 0-D tensor")
    if xi.shape[d] != 3 or yi.shape[d] != 3:
        raise _C_engine.ShapeMismatch(
            f"linalg.cross: inputs dimension {d} must have length 3, got {xi.shape[d]} and {yi.shape[d]}"
        )

    def _idx(t: _C_engine.TensorImpl, i: int) -> _C_engine.TensorImpl:
        idx_shape = list(t.shape)
        idx_shape[d] = 1
        idx = _C_engine.full(idx_shape, float(i), _C_engine.I32, t.device)
        return _C_engine.squeeze(_C_engine.gather(t, idx, d), d)

    x0, x1, x2 = _idx(xi, 0), _idx(xi, 1), _idx(xi, 2)
    y0, y1, y2 = _idx(yi, 0), _idx(yi, 1), _idx(yi, 2)

    c0 = _C_engine.sub(_C_engine.mul(x1, y2), _C_engine.mul(x2, y1))
    c1 = _C_engine.sub(_C_engine.mul(x2, y0), _C_engine.mul(x0, y2))
    c2 = _C_engine.sub(_C_engine.mul(x0, y1), _C_engine.mul(x1, y0))
    return _wrap(_C_engine.concatenate([_C_engine.unsqueeze(c, d) for c in (c0, c1, c2)], d))


def vecdot(x: Tensor, y: Tensor, *, dim: int = -1) -> Tensor:
    r"""Compute a batched vector dot product along an axis.

    Reduces the chosen axis with a sum of element-wise products, the
    first operand conjugated:

    .. math::

        (x \cdot y)_{\ldots} \,=\, \sum_{k} \overline{x_{\ldots, k, \ldots}}\,
                                              y_{\ldots, k, \ldots}.

    Parameters
    ----------
    x : Tensor
        First operand.
    y : Tensor
        Second operand, broadcast-compatible with ``x``.
    dim : int, keyword-only, optional
        Axis to contract.  Default ``-1``.

    Returns
    -------
    Tensor
        Reduced tensor with ``dim`` removed.

    Notes
    -----
    Equivalent to ``(x.conj() * y).sum(dim=dim)`` — for real inputs
    simply ``(x * y).sum(dim=dim)``.  Useful for computing many
    independent dot products in one shot (e.g., per-row inner products
    of two matrices).

    The conjugate makes ``vecdot(x, x)`` the squared norm
    :math:`\sum_k |x_k|^2` of a complex vector.  Without it, the
    complex result was the bilinear :math:`\sum_k x_k y_k` instead.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import vecdot
    >>> x = lucid.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    >>> y = lucid.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    >>> vecdot(x, y)
    tensor([-2., 5.])
    >>> vecdot(lucid.tensor([1 + 2j, 3 - 1j]), lucid.tensor([2 - 1j, 1 + 1j]))
    tensor((2.-1.j), dtype=lucid.complex64)
    """
    with _autocast_entry("vecdot", _unwrap(x), _unwrap(y)) as (xi, yi):
        if xi.dtype in (_C_engine.C64, _C_engine.C128):
            xi = _C_engine.conj(xi)
        prod = _C_engine.mul(xi, yi)
        return _wrap(_C_engine.sum(prod, [dim], False))


def dot(x: Tensor, y: Tensor) -> Tensor:
    r"""Dot product of two 1-D tensors.

    Computes the scalar inner product

    .. math::

        x \cdot y \,=\, \sum_{i} x_i\, y_i.

    Parameters
    ----------
    x, y : Tensor
        1-D tensors of equal length.

    Returns
    -------
    Tensor
        Scalar dot product.

    Notes
    -----
    For higher-rank tensors use :func:`inner` (last-dim contraction)
    or :func:`vecdot` (explicit axis), and :func:`matmul` for matrix
    products.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import dot
    >>> dot(lucid.tensor([1.0, 2.0, 3.0]), lucid.tensor([4.0, 5.0, 6.0]))
    tensor(32.)
    """
    return _wrap(_C_engine.dot(_unwrap(x), _unwrap(y)))


def inner(x: Tensor, y: Tensor) -> Tensor:
    r"""Inner product over the last axes of two tensors.

    Behaves like :func:`dot` for 1-D inputs and like a generalised
    outer-then-contract for higher ranks.

    Parameters
    ----------
    x, y : Tensor
        Tensors whose last dimensions match.

    Returns
    -------
    Tensor
        Result with shape ``x.shape[:-1] + y.shape[:-1]``.

    Notes
    -----
    Contracts the last axis of both operands and sums:

    .. math::

        \mathrm{inner}(x, y)_{\ldots, \ldots'} \,=\,
            \sum_{k} x_{\ldots, k}\, y_{\ldots', k}.

    For 1-D inputs, this is the standard vector dot product.  For higher
    ranks the leading axes of ``x`` and ``y`` are kept independent and
    appear consecutively in the output — handy for batched dot products
    when the contraction axis is the trailing one.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import inner
    >>> inner(lucid.tensor([1.0, 2.0]), lucid.tensor([3.0, 4.0]))
    tensor(11.)
    """
    return _wrap(_C_engine.inner(_unwrap(x), _unwrap(y)))


def outer(x: Tensor, y: Tensor) -> Tensor:
    r"""Outer product of two 1-D tensors.

    Parameters
    ----------
    x : Tensor
        1-D tensor of length :math:`m`.
    y : Tensor
        1-D tensor of length :math:`n`.

    Returns
    -------
    Tensor
        Matrix of shape ``(m, n)``.

    Notes
    -----
    Computes the rank-1 matrix

    .. math::

        (x \otimes y)_{ij} \,=\, x_i\, y_j,

    yielding the same result as :math:`x\,y^\top` viewed as a 2-D array.
    Outer products underpin rank-1 updates (BFGS), Kronecker-product
    factorisations, and certain attention patterns.  Unlike
    :func:`matmul`, the inputs are not contracted — every pair
    :math:`(x_i, y_j)` produces an independent entry.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import outer
    >>> outer(lucid.tensor([1.0, 2.0]), lucid.tensor([3.0, 4.0]))
    tensor([[3., 4.], [6., 8.]])
    """
    # The adapter ``Tensor.outer`` uses too: it brings mixed dtypes to their
    # common one, which the engine's single-dtype kernel refused.
    return _wrap(_outer_adapter(_unwrap(x), _unwrap(y)))


def _matrix_norm_induced1(
    input: Tensor, ord: int | float, d0: int, d1: int, keepdim: bool, dtype: object
) -> Tensor:
    """Induced 1- and inf-norms (and their -1 / -inf counterparts)."""
    along, across = (d0, d1) if ord in (1, -1) else (d1, d0)
    sums = _unwrap(vector_norm(input, 1, along, True, dtype=dtype))
    pick = _C_engine.max if ord in (1, float("inf")) else _C_engine.min
    picked = pick(sums, [across], True)
    if not keepdim:
        picked = _C_engine.reshape(
            picked, [n for i, n in enumerate(picked.shape) if i not in (d0, d1)]
        )
    return _wrap(picked)


def matrix_norm(
    input: Tensor,
    ord: int | float | str = "fro",
    dim: tuple[int, int] = (-2, -1),
    keepdim: bool = False,
    *,
    dtype: object = None,
) -> Tensor:
    r"""Compute a matrix norm.

    Reduces the trailing two axes of an input to a scalar matrix norm.
    Supported orders:

    * ``"fro"`` — Frobenius norm
      :math:`\|A\|_F = \big(\sum_{ij} |A_{ij}|^2\big)^{1/2}`.
    * ``"nuc"`` — nuclear norm
      :math:`\|A\|_* = \sum_i \sigma_i(A)` (sum of singular values).
    * ``1`` / ``-1`` — max / min absolute column sum.
    * ``inf`` / ``-inf`` — max / min absolute row sum.
    * ``2`` / ``-2`` — largest / smallest singular value (spectral
      norm and its reciprocal).

    Parameters
    ----------
    input : Tensor
        Input of shape ``(*, m, n)``.
    ord : int, float or str, optional
        Norm order.  Default ``"fro"``.
    dim : tuple of two ints, optional
        Axis pair identifying the matrix dimensions.  Default
        ``(-2, -1)``.
    keepdim : bool, optional
        If ``True``, reduced dims are retained with size 1.
    dtype : dtype, keyword-only, optional
        Cast ``input`` to this dtype before computing the norm; must match
        the real/complex kind of ``input`` and must not narrow it.  ``None``
        (default) uses ``input``'s own dtype.

    Returns
    -------
    Tensor
        Matrix norm of each batch — real, also for a complex ``input``.

    Raises
    ------
    DtypeMismatch
        If ``input`` is an integer or bool tensor (a ``TypeError``).
    ValueError
        If ``input`` has fewer than two dimensions, ``dim`` does not name two
        distinct axes, or ``ord`` is not one of the orders above.
    NotImplementedError
        If ``ord`` is ``"nuc"``, ``2`` or ``-2`` and ``input`` is complex:
        those orders are read off the singular values, and :func:`svd`
        has no complex kernel yet.

    Notes
    -----
    Spectral and nuclear norms require an SVD and so cost
    :math:`O(\min(m,n)^2 \max(m,n))`.  Entry-wise norms reduce in a
    single pass.

    The entry-wise orders (``"fro"``, :math:`\pm 1`, :math:`\pm\infty`)
    are taken through :func:`vector_norm`, so a complex ``input`` is reduced
    through its magnitudes :math:`|A_{ij}|` exactly as a vector would be.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import matrix_norm
    >>> A = lucid.tensor([[3.0, 4.0], [0.0, 0.0]])
    >>> matrix_norm(A, ord="fro")
    tensor(5.)
    >>> matrix_norm(lucid.tensor([[3 + 4j, 0j], [0j, 0j]]), ord="fro")
    tensor(5.)
    """
    xi = _require_normable(_unwrap(input), "matrix_norm")
    with _autocast_entry("matrix_norm", xi) as (xi_cast,):
        xi = _cast_for_norm(input, dtype, "matrix_norm") if dtype is not None else xi_cast
        if dtype is None:
            input = _wrap(xi)
        d0, d1 = _check_matrix_norm_dims(len(xi.shape), dim)

        # Entry-wise orders reduce through ``vector_norm``, which owns taking |x|.
        if ord == "fro" or (ord in (1, float("inf")) and (xi.shape[d0] == 0 or xi.shape[d1] == 0)):
            return vector_norm(input, 2, [d0, d1], keepdim, dtype=dtype)

        if ord in (1, -1, float("inf"), float("-inf")):
            return _matrix_norm_induced1(input, ord, d0, d1, keepdim, dtype)

        sv_reduce = {"nuc": _C_engine.sum, 2: _C_engine.max, -2: _C_engine.min}.get(ord)
        if sv_reduce is None:
            raise ValueError(f"matrix_norm: unsupported ord={ord!r}")
        if xi.dtype in (_C_engine.C64, _C_engine.C128):
            raise NotImplementedError(
                f"matrix_norm: ord={ord!r} is read off the singular values, and "
                "svd has no complex kernel yet; ord='fro', 1, -1, inf and -inf "
                "reduce through |x| and accept a complex input"
            )
        if ord in ("nuc", 2) and (xi.shape[d0] == 0 or xi.shape[d1] == 0):
            return vector_norm(input, 2, [d0, d1], keepdim, dtype=dtype)
        return _matrix_norm_sv(xi, d0, d1, keepdim, sv_reduce)


def lstsq(
    input: Tensor,
    b: Tensor,
    rcond: float | None = None,
    *,
    driver: str | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Solve a least-squares linear system.

    Finds :math:`X` minimising the squared residual

    .. math::

        \min_X \,\|A\,X - B\|_2^2,

    where :math:`A \in \mathbb{R}^{m \times n}` may be over- or
    underdetermined.  For full-rank :math:`A` and :math:`m \ge n` the
    solution is unique:

    .. math::

        X \,=\, (A^\top A)^{-1} A^\top B,

    obtained more stably via QR or SVD without forming the normal
    equations.

    Parameters
    ----------
    input : Tensor
        Coefficient matrix of shape ``(*, m, n)``.
    b : Tensor
        Right-hand side of shape ``(*, m, k)`` (or ``(*, m)``).
    rcond : float or None, optional
        Cutoff for small singular values (passed to the underlying
        driver).  ``None`` selects the default driver heuristic.
    driver : str or None, keyword-only, optional
        Solver choice (``"gels"``, ``"gelsy"``, ``"gelsd"``, ``"gelss"``).
        ``None`` selects ``"gelsy"`` on CPU and ``"gels"`` on Metal.

    Returns
    -------
    solution : Tensor
        Least-squares solution of shape ``(*, n, k)`` (or ``(*, n)``).
    residuals : Tensor
        Sum of squared residuals of shape ``(*, k)`` when :math:`m > n` and full
        rank (or ``driver == "gels"``); empty ``(0,)`` otherwise or for ``"gelsy"``.
    rank : Tensor
        Effective rank of :math:`A` (``int64``, shape ``(*)``), or empty ``(0,)``
        when ``driver == "gels"``.
    singular_values : Tensor
        Singular values of :math:`A` of shape ``(*, min(m, n))`` for ``"gelsd"``
        and ``"gelss"``; empty ``(0,)`` for ``"gels"`` and ``"gelsy"``.

    Notes
    -----
    The solution is :math:`A^+ B`, the minimum-norm least-squares solution for
    every shape and rank, and differentiates in ``input`` and ``b`` (twice).

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import lstsq
    >>> A = lucid.tensor([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    >>> b = lucid.tensor([[6.0], [9.0], [12.0]])
    >>> sol, *_ = lstsq(A, b)
    >>> sol
    tensor([[3.], [3.]])
    """
    with _autocast_entry("lstsq", _unwrap(input), _unwrap(b)) as (ai, bi):
        return _lstsq_solution(_wrap(ai), _wrap(bi), _broadcast_batch, rcond, driver)


def lu_solve(
    LU: Tensor,
    pivots: Tensor,
    B: Tensor,
    *,
    left: bool = True,
    adjoint: bool = False,
) -> Tensor:
    r"""Solve a linear system from a precomputed LU factorization.

    Given the packed factorization :math:`PA = LU` returned by
    :func:`lu_factor`, solves

    .. math::

        A\,X \,=\, B

    by applying the permutation and performing two triangular solves
    (forward + back substitution).

    Parameters
    ----------
    LU : Tensor
        Packed LU factor of shape ``(*, n, n)`` from :func:`lu_factor`.
    pivots : Tensor
        Pivot indices of shape ``(*, n)`` from :func:`lu_factor`
        (1-based, LAPACK convention).
    B : Tensor
        Right-hand side ``(*, n, k)`` or ``(n,)``, broadcast against ``LU``.
    left : bool, keyword-only, optional
        If ``True`` (default), solve :math:`A X = B`; ``False`` solves
        :math:`X A = B`.
    adjoint : bool, keyword-only, optional
        If ``True``, solve with :math:`A^H` instead of :math:`A`.
        Default ``False``.

    Returns
    -------
    Tensor
        Solution :math:`X`: broadcast batch, then ``(n, k)``, or ``(n,)`` for 1-D ``B``.

    Notes
    -----
    Backed by LAPACK ``getrs``.  Cost per solve is :math:`O(n^2 k)` —
    much cheaper than a fresh :func:`solve` (:math:`O(n^3)`) when the
    same :math:`A` is reused.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import lu_factor, lu_solve
    >>> A = lucid.tensor([[3.0, 1.0], [1.0, 2.0]])
    >>> LU, piv = lu_factor(A)
    >>> b = lucid.tensor([[9.0], [8.0]])
    >>> lu_solve(LU, piv, b)
    tensor([[2.], [3.]])
    """
    return _lu_solve_general(LU, pivots, B, left=left, adjoint=adjoint)


def householder_product(input: Tensor, tau: Tensor) -> Tensor:
    r"""Reconstruct an orthogonal matrix from Householder reflectors.

    Computes the implicit product

    .. math::

        Q \,=\, H_1\,H_2\,\cdots\,H_k,
        \qquad H_i \,=\, I - \tau_i\, v_i\, v_i^\top,

    where each :math:`v_i` is a Householder vector stored in the
    :math:`i`-th column of the packed input ``input`` and :math:`\tau_i`
    is its scalar factor.  This is the standard way to materialise the
    :math:`Q` factor from a packed QR (``geqrf``) result.

    Parameters
    ----------
    input : Tensor
        Packed reflector matrix of shape ``(*, m, k)`` — columns
        contain the Householder vectors (typically the output of an
        unpacked ``geqrf``).
    tau : Tensor
        Scalar factors of shape ``(*, k)``.

    Returns
    -------
    Tensor
        Orthogonal matrix :math:`Q` of shape ``(*, m, k)``.

    Notes
    -----
    Backed by LAPACK ``orgqr``.  Cost is :math:`O(m k^2)`.  Useful
    when a routine returns the packed Householder form (cheaper to
    store) but the explicit :math:`Q` is needed downstream.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import qr, householder_product
    >>> A = lucid.randn(4, 3)
    >>> Q, R = qr(A, mode="reduced")
    >>> # The same Q can be reconstructed from packed Householder reflectors
    >>> # returned by lower-level geqrf-style factorisations.
    >>> Q.shape
    (4, 3)
    """
    return _householder_product(input, tau)


# ── Full LU decomposition (P, L, U) ────────────────────────────────────────


def lu(A: Tensor, *, pivot: bool = True) -> tuple[Tensor, Tensor, Tensor]:
    r"""Full LU decomposition with explicit factors :math:`(P, L, U)`.

    Decomposes a square matrix :math:`A` as

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
        Square matrix of shape ``(*, n, n)`` (batch dims allowed).
    pivot : bool, keyword-only, optional
        Must be ``True`` (the default).  ``False`` would request an
        unpivoted LU; Lucid does not currently ship that kernel and
        raises ``NotImplementedError``.

    Returns
    -------
    P : Tensor
        Permutation matrix of shape ``(*, n, n)``.
    L : Tensor
        Unit-lower-triangular factor of shape ``(*, n, n)``.
    U : Tensor
        Upper-triangular factor of shape ``(*, n, n)``.

    Notes
    -----
    Implemented as a Python composite over :func:`lu_factor` followed
    by explicit triangular masking and pivot-to-matrix conversion.
    Cost is :math:`O(\tfrac{2}{3} n^3)`, dominated by the underlying
    :math:`LU` factorization.

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
    sh = tuple(_unwrap(A).shape)
    if len(sh) < 2 or sh[-1] != sh[-2]:
        raise ValueError(f"lu requires a square matrix in the last two dims, got {sh}")
    n = int(sh[-1])

    # A singular matrix has an LU factorization; only lu_factor refuses it.
    LU, pivots, _ = _lu_factor_packed(A)

    # Split the packed LU into L (unit-lower, by index mask so a non-finite
    # diagonal entry does not poison L via inf * 0) and U (upper).
    L = lucid.tril(LU, -1) + lucid.eye(n, dtype=A.dtype, device=A.device)
    U = lucid.triu(LU)

    # Reconstruct the permutation matrix from the LAPACK pivot vector.
    # LAPACK pivots are 1-based: ``pivots[i]`` holds the row swapped with
    # row ``i`` during step ``i``.  We start from the identity and apply
    # the swap sequence in reverse to recover ``P`` such that ``P · A``
    # equals the *unpermuted* LU product.
    P = _build_permutation_matrix(pivots, n, A.dtype, A.device)
    return _LuResult(P, L, U)


def _check_pivots(pivots: Tensor, factor: Tensor, op: str) -> Tensor:
    """Validate a LAPACK pivot vector and return it at the width LAPACK reads.

    LAPACK's ``ipiv`` argument is ``const int*``, so the engine reads the
    pivot buffer 32 bits at a time no matter what dtype it arrived as.
    Nothing checked that, and all three ways of getting it wrong were
    silent or fatal rather than an error:

    * **int8 / bool** — the buffer is shorter than the read, so the last
      entries come from past its own allocation.  ``lu_solve`` took the
      process down with SIGBUS, which is how the audit's own sweep died.
    * **float32 / float64** — the mantissa bits are read as indices,
      landing far outside the matrix.  Also SIGBUS.
    * **int16 / int64** — the buffer is long enough to survive the
      over-read, so this one *returns*.  On a matrix that genuinely
      pivots, int16 answered ``5.9e+133`` and int64 answered a plausible
      vector that was simply not the solution.  A wrong number that looks
      like a right one is the worst of the three.

    Integer pivots are converted rather than rejected — ``lu_factor``
    hands back int32, but an int64 pivot vector is a reasonable thing for
    a caller to have built, and converting it is what makes it correct.

    Parameters
    ----------
    pivots : Tensor
        1-based pivot indices, as returned by :func:`lu_factor`.
    factor : Tensor
        The factorization the pivots belong to; supplies ``n`` for the
        range check.
    op : str
        Calling op name, used as the prefix of the raised message.

    Returns
    -------
    Tensor
        ``pivots`` as int32.

    Raises
    ------
    TypeError
        If ``pivots`` is not an integer tensor.
    IndexError
        If any pivot falls outside ``[1, n]``.
    """
    if pivots.dtype not in (lucid.int8, lucid.int16, lucid.int32, lucid.int64):
        raise TypeError(
            f"{op}: pivots must be an integer tensor, got {pivots.dtype}; "
            f"pass the vector returned by lu_factor, or cast with "
            f".to(lucid.int32)"
        )
    normalised = pivots if pivots.dtype == lucid.int32 else pivots.to(lucid.int32)
    if normalised.numel() == 0:
        return normalised

    # LAPACK pivots are 1-based: entry i names the row swapped with row i.
    # An out-of-range entry indexes outside the matrix, and the engine did
    # not notice — a pivot of 99 on a 4x4 was accepted and answered.
    n = int(factor.shape[-2]) if len(factor.shape) >= 2 else int(factor.shape[0])
    # Reduced in int64, not int32.  The CPU reduce kernels do not cover
    # every integer width and int32 min/max raises — the same detour
    # ``check_embedding_indices`` takes for the same reason.
    wide = normalised.to(lucid.int64)
    lo = int(wide.min().item())
    hi = int(wide.max().item())
    if lo < 1 or hi > n:
        bad = lo if lo < 1 else hi
        raise IndexError(
            f"{op}: pivot {bad} is out of range for an order-{n} "
            f"factorization (LAPACK pivots are 1-based, valid range [1, {n}])"
        )
    return normalised


def _build_permutation_matrix(
    pivots: Tensor,
    n: int,
    dtype: lucid.dtype | type[lucid.dtype] | _C_engine.Dtype | None,
    device: lucid.device | _C_engine.Device | str | None,
) -> Tensor:
    """Convert LAPACK's 1-based pivot vectors to explicit ``(*, n, n)`` P
    matrices such that ``A = P · L · U`` (LAPACK's contract is
    ``P · A = L · U``; we transpose at the end so callers can use the
    factor product directly).

    The swaps are replayed on host integers read with ``tolist`` — the
    pivots are a handful of indices, and the conversion used to go through
    ``Tensor.numpy``, which put numpy on the compute path (H4) and limited
    it to a single matrix.  Every matrix of a batch gets its own ``P``.
    """
    k = int(pivots.shape[-1]) if pivots.ndim else 0
    batch = tuple(int(s) for s in pivots.shape[:-1])
    rows = cast(list[list[int]], pivots.reshape(-1, k).tolist()) if k else [[]]
    perms: list[int] = []
    for piv in rows:
        perm = list(range(n))
        for i, p in enumerate(piv):
            j = int(p) - 1  # 1-based → 0-based
            perm[i], perm[j] = perm[j], perm[i]
        perms.extend(perm)
    # Row i of LAPACK's P is e_{perm[i]}: pick those rows of the identity.
    eye = lucid.eye(n, dtype=dtype, device=device)
    index = lucid.tensor(perms, dtype=lucid.int64, device=eye.device)
    P = eye.index_select(0, index).reshape(*batch, n, n)
    return P.mT  # transpose: A = P·L·U


# ── ldl_solve — back-substitution using the LDL factorization ──────────────


def _bunch_kaufman_steps(piv: list[int], n: int) -> list[tuple[int, int, int]]:
    """Read one ``sytrf`` pivot vector as its sequence of elimination steps.

    Each step is ``(k, size, partner)``: a diagonal block of ``size`` 1 or 2
    starting at row ``k`` (0-based), and the row its interchange swapped
    with row ``k`` (a 1×1 block) or row ``k + 1`` (a 2×2 block).

    Raises
    ------
    ValueError
        If the vector is not one ``sytrf`` could have written: a zero, an
        index outside ``[1, n]``, or a negative entry not paired with an
        equal one in the next position.
    """
    steps: list[tuple[int, int, int]] = []
    k = 0
    while k < n:
        p = int(piv[k])
        if p == 0 or abs(p) > n:
            raise ValueError(
                f"ldl_solve: pivot {p} at position {k} is out of range for an "
                f"order-{n} factorization (valid: ±1 … ±{n}, LAPACK is 1-based)"
            )
        if p > 0:
            steps.append((k, 1, p - 1))
            k += 1
            continue
        if k + 1 >= n or int(piv[k + 1]) != p:
            raise ValueError(
                f"ldl_solve: pivot {p} at position {k} opens a 2x2 block, so "
                f"position {k + 1} must hold the same value"
            )
        steps.append((k, 2, -p - 1))
        k += 2
    return steps


def ldl_solve(LD: Tensor, pivots: Tensor, B: Tensor, *, hermitian: bool = False) -> Tensor:
    r"""Solve a symmetric linear system using an LDL factorization.

    Given :math:`A = L\,D\,L^\top` produced by :func:`ldl_factor`,
    solves

    .. math::

        A\,X \,=\, B.

    ``sytrf``'s :math:`L` is a product :math:`P_1 L_1 P_2 L_2 \cdots` of
    elimination steps, each preceded by its own row interchange.  Moving
    every interchange to the left gives

    .. math::

        A \,=\, P\,\tilde L\,D\,\tilde L^\top P^\top,

    with :math:`\tilde L` unit-lower-triangular — :math:`L` with each
    column carrying the interchanges of the steps after it, as LAPACK's
    ``syconv`` arranges it — so the solve is a permutation, two triangular
    solves around the :math:`1 \times 1` / :math:`2 \times 2` blocks of
    :math:`D` (solved in closed form, as ``sytrs`` does), and the inverse
    permutation.

    Parameters
    ----------
    LD : Tensor
        Packed LDL factor from :func:`ldl_factor`, shape ``(*, n, n)``.
        Only its lower triangle is read.
    pivots : Tensor
        Integer pivot indices from :func:`ldl_factor`, shape ``(*, n)``
        (1-based, LAPACK convention).  Both :math:`1 \times 1` and
        :math:`2 \times 2` Bunch-Kaufman blocks are supported.
    B : Tensor
        Right-hand side of shape ``(*, n, k)`` (or ``(n,)``).  Its batch
        dimensions broadcast with those of ``LD``.
    hermitian : bool, keyword-only, optional
        If ``True``, treat ``LD`` as a Hermitian :math:`L D L^H`
        factorisation.  For real inputs this is identical to ``False``.
        Default ``False``.

    Returns
    -------
    Tensor
        Solution :math:`X`, shaped like ``B`` broadcast against ``LD``'s
        batch.

    Raises
    ------
    ValueError
        If ``LD`` is not square, ``pivots`` does not match it, or a pivot
        is not one ``sytrf`` could have written.
    TypeError
        If ``pivots`` is not an integer tensor.

    Notes
    -----
    Supports indefinite symmetric :math:`A` (unlike Cholesky), so it
    is appropriate for KKT / saddle-point systems where Cholesky
    would fail.  Cost per solve is :math:`O(n^2 k)` once the LDL
    factor is in hand: two triangular solves, and :math:`O(nk)` for
    :math:`D`.  The pivots are replayed on the host in :math:`O(n)` per
    matrix; only when an interchange reaches earlier columns of
    :math:`L` is an :math:`n \times n` gather index built for
    :math:`\tilde L` (:math:`O(n^2)` host work).

    The interchanges are applied in the order ``sytrf`` performed them
    (LAPACK ``sytrs``).  Applying them as a single up-front row
    permutation of :math:`B` — as this function once did — is right only
    when no later step swaps rows an earlier column of :math:`L` has
    entries in, and silently wrong otherwise.  The pivots are read on the
    host with ``tolist``; the arithmetic stays on ``LD``'s device and is
    differentiable in ``LD`` and ``B``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import ldl_factor, ldl_solve
    >>> A = lucid.tensor([[4.0, 1.0], [1.0, 3.0]])
    >>> LD, piv = ldl_factor(A)
    >>> b = lucid.tensor([[5.0], [4.0]])
    >>> ldl_solve(LD, piv, b)
    tensor([[1.], [1.]])
    """
    del hermitian
    if LD.ndim < 2 or LD.shape[-1] != LD.shape[-2]:
        raise ValueError(
            f"ldl_solve: LD must be a square matrix or a batch of them, got "
            f"shape {tuple(LD.shape)}"
        )
    if pivots.dtype not in (lucid.int8, lucid.int16, lucid.int32, lucid.int64):
        raise TypeError(
            f"ldl_solve: pivots must be an integer tensor, got {pivots.dtype}; "
            f"pass the vector returned by ldl_factor"
        )
    n = int(LD.shape[-1])
    if tuple(pivots.shape) != tuple(LD.shape[:-1]):
        raise ValueError(
            f"ldl_solve: pivots must have shape {tuple(LD.shape[:-1])} to match "
            f"LD of shape {tuple(LD.shape)}, got {tuple(pivots.shape)}"
        )
    vector = B.ndim == 1
    rhs = B.unsqueeze(-1) if vector else B
    if rhs.ndim < 2 or int(rhs.shape[-2]) != n:
        raise ValueError(
            f"ldl_solve: B must have {n} rows to match LD of shape "
            f"{tuple(LD.shape)}, got shape {tuple(B.shape)}"
        )
    try:
        batch = _broadcast_batch(
            tuple(int(s) for s in LD.shape[:-2]),
            tuple(int(s) for s in rhs.shape[:-2]),
        )
    except ValueError:
        raise ValueError(
            f"ldl_solve: the batch dimensions of LD {tuple(LD.shape)} and B "
            f"{tuple(B.shape)} do not broadcast"
        ) from None
    k = int(rhs.shape[-1])
    if n == 0 or k == 0 or 0 in batch:
        empty = lucid.zeros(*batch, n, k, dtype=rhs.dtype, device=rhs.device)
        return empty.squeeze(-1) if vector else empty

    LD = lucid.broadcast_to(lucid.tril(LD), (*batch, n, n))
    rhs = lucid.broadcast_to(rhs, (*batch, n, k))
    piv_rows = cast(
        list[list[int]],
        lucid.broadcast_to(pivots, (*batch, n)).reshape(-1, n).tolist(),
    )

    # Replay every matrix's interchanges on host indices, O(n) each:
    #   order[i]    — the row of B the forward sweep sees at position i
    #                 (B permuted by Pᵀ);
    #   partner[i]  — the other row of i's 2x2 block of D, or i itself;
    #   moves       — (columns, row, row) for each interchange that reaches
    #                 earlier columns of L, as syconv swaps them.
    orders: list[int] = []
    partners: list[int] = []
    moves: list[list[tuple[int, int, int]]] = []
    for piv in piv_rows:
        order = list(range(n))
        partner = list(range(n))
        moved: list[tuple[int, int, int]] = []
        for start, size, other in _bunch_kaufman_steps(piv, n):
            a = start + size - 1  # the row this step interchanged
            order[a], order[other] = order[other], order[a]
            if other != a and start > 0:
                moved.append((start, a, other))
            if size == 2:
                partner[start], partner[start + 1] = start + 1, start
        orders.extend(order)
        partners.extend(partner)
        moves.append(moved)

    device = LD.device
    rows = lucid.arange(n, dtype=lucid.int64, device=device).unsqueeze(-1)
    partner_index = lucid.tensor(partners, dtype=lucid.int64, device=device)
    partner_col = partner_index.reshape(*batch, n, 1)
    # D's off-diagonal entry of a 2x2 block sits at LD[k + 1, k]; it belongs
    # to D, not L.  For row i that is LD[i, partner] + LD[partner, i], the
    # other term lying in the cleared upper triangle.
    paired = partner_col != rows
    offdiag = lucid.gather(LD, -1, partner_col) + lucid.gather(LD.mT, -1, partner_col)
    in_d = (rows.mT == partner_col) & (partner_col < rows)
    L = lucid.where(in_d, lucid.zeros_like(LD), lucid.tril(LD, -1))
    if any(moves):
        # Only now an n x n gather index: the rows a swap touched, every
        # other row reading from itself.
        sources: list[list[int]] = []
        for moved in moves:
            touched: dict[int, list[int]] = {}
            for start, a, other in moved:
                ra = touched.setdefault(a, [a] * n)
                ro = touched.setdefault(other, [other] * n)
                ra[:start], ro[:start] = ro[:start], ra[:start]
            sources.extend(touched.get(r, [r] * n) for r in range(n))
        index = lucid.tensor(sources, dtype=lucid.int64, device=device)
        L = lucid.gather(L, -2, index.reshape(*batch, n, n))
    L = L + lucid.eye(n, dtype=LD.dtype, device=device)

    order_index = lucid.tensor(orders, dtype=lucid.int64, device=device)
    order_index = order_index.reshape(*batch, n, 1)
    inverse_index = lucid.argsort(order_index, dim=-2)
    Y = lucid.gather(rhs, -2, lucid.broadcast_to(order_index, (*batch, n, k)))
    Y = solve_triangular(L, Y, upper=False, unitriangular=True)
    Y = _solve_bunch_kaufman_d(LD, offdiag, partner_col, paired, Y)
    Y = solve_triangular(L.mT, Y, upper=True, unitriangular=True)
    X = lucid.gather(Y, -2, lucid.broadcast_to(inverse_index, (*batch, n, k)))
    return X.squeeze(-1) if vector else X


def _solve_bunch_kaufman_d(
    LD: Tensor, offdiag: Tensor, partner_col: Tensor, paired: Tensor, Y: Tensor
) -> Tensor:
    """``D⁻¹ Y`` for ``sytrf``'s block-diagonal ``D``, in closed form.

    A 1×1 block divides.  A 2×2 block ``[[a, b], [b, c]]`` takes ``sytrs``'s
    scaled inverse — every quantity divided by ``b`` first, so the
    determinant ``ac - b²`` is never formed — and the formula is the same
    for both rows of the block, written from that row's side::

        z_i = ((d_o / b)(y_i / b) - y_o / b) / ((d_i / b)(d_o / b) - 1)

    with ``o`` the row's partner.  O(n k), against a dense solve's O(n³).
    The masked-out branch of each ``where`` is kept finite, so neither
    leaks a NaN into the other's gradient.
    """
    k = int(Y.shape[-1])
    d = LD.diagonal(dim1=-2, dim2=-1).unsqueeze(-1)  # (*, n, 1)
    d_o = lucid.gather(d, -2, partner_col)
    Y_o = lucid.gather(Y, -2, lucid.broadcast_to(partner_col, tuple(Y.shape[:-1]) + (k,)))
    one = lucid.ones_like(d)
    b = lucid.where(paired, offdiag, one)
    blocks = ((d_o / b) * (Y / b) - Y_o / b) / lucid.where(paired, (d / b) * (d_o / b) - 1.0, one)
    return lucid.where(paired, blocks, Y / lucid.where(paired, one, d))


# ── linalg.diagonal — batched-view alias of lucid.diagonal ─────────────────


def matrix_exp(input: Tensor) -> Tensor:
    r"""Matrix exponential :math:`e^A`.

    Returns the matrix exponential

    .. math::

        e^A \,=\, \sum_{k=0}^{\infty} \frac{A^k}{k!},

    which solves the matrix ODE :math:`\dot Y = A Y` with initial
    condition :math:`Y(0) = I`.  Computed by the scaling-and-squaring
    algorithm of Higham (2005) with a Padé [6/6] rational approximant:

    1. Scale :math:`A' = A / 2^s` so that :math:`\|A'\| \le \theta_6`.
    2. Approximate :math:`R \approx e^{A'}` via Padé [6/6] using the
      even/odd polynomial splitting :math:`R = D^{-1} N`.
    3. Square :math:`R` a total of :math:`s` times to recover
       :math:`e^A = R^{2^s}`.

    Parameters
    ----------
    input : Tensor
        Square matrix of shape ``(*, n, n)``.

    Returns
    -------
    Tensor
        :math:`e^A`, shape ``(*, n, n)``.

    Notes
    -----
    Computational cost is :math:`O((s + 6)\, n^3)` where the scaling
    parameter :math:`s = \lceil \log_2 (\|A\| / \theta_6) \rceil`.  The
    Frobenius norm is used here as a (conservative) proxy for the
    1-norm — this may add one extra squaring step but keeps the result
    correct.

    Differentiable: built entirely on :func:`matmul` and :func:`inv`,
    so autograd flows naturally.  Batched inputs (``ndim > 2``) are
    handled element-wise.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import matrix_exp
    >>> A = lucid.tensor([[0.0, 1.0], [-1.0, 0.0]])  # 90-deg rotation generator
    >>> matrix_exp(A)
    tensor([[0.5403, 0.8415], [-0.8415, 0.5403]])
    """
    import math as _math

    A = input
    sh: tuple[int, ...] = tuple(A.shape)
    if len(sh) < 2 or sh[-1] != sh[-2]:
        raise ValueError(
            f"matrix_exp requires a square matrix in the last two dims, got shape {sh}"
        )
    n: int = int(sh[-1])

    # -- Padé [6/6] coefficients c_j = (12-j)! * 6! / (12! * j! * (6-j)!) ----
    _c0: float = 1.0
    _c1: float = 0.5
    _c2: float = 5.0 / 44.0
    _c3: float = 1.0 / 66.0
    _c4: float = 1.0 / 792.0
    _c5: float = 1.0 / 15840.0
    _c6: float = 1.0 / 665280.0
    # theta_6: Frobenius-norm threshold at which Padé [6/6] is machine-accurate
    _theta: float = 2.0

    # -- Scaling ---------------------------------------------------------------
    # Upper bound for the 1-norm: max element × n (safe over-estimate).
    # An empty matrix has nothing to bound (and ``max`` of nothing is refused).
    norm_bound: float = float(A.abs().max().item()) * n if A.numel() else 0.0
    s: int = max(0, _math.ceil(_math.log2(norm_bound / _theta))) if norm_bound > _theta else 0
    A_sc: Tensor = A * (2.0 ** (-s))

    # -- Matrix powers ---------------------------------------------------------
    A2: Tensor = A_sc @ A_sc
    A4: Tensor = A2 @ A2
    A6: Tensor = A2 @ A4

    # -- Identity (broadcasts over any batch dims) -----------------------------
    eye_2d: Tensor = lucid.eye(n, dtype=A.dtype, device=A.device)
    if len(sh) > 2:
        # Use arithmetic broadcast: eye_2d + zeros of batch shape keeps autograd.
        I: Tensor = eye_2d + lucid.zeros(*sh[:-2], n, n, dtype=A.dtype, device=A.device)
    else:
        I = eye_2d

    # -- Padé polynomials (even/odd split for efficiency) ----------------------
    # Even — c0*I + c2*A2 + c4*A4 + c6*A6
    Even: Tensor = _c0 * I + _c2 * A2 + _c4 * A4 + _c6 * A6
    # Odd  — A * (c1*I + c3*A2 + c5*A4)
    Odd: Tensor = A_sc @ (_c1 * I + _c3 * A2 + _c5 * A4)

    # N = Even + Odd,  D = Even - Odd
    N: Tensor = Even + Odd
    D: Tensor = Even - Odd

    # -- Solve D @ R = N  →  R = D^{-1} @ N -----------------------------------
    R: Tensor = inv(D) @ N

    # -- Squaring step ---------------------------------------------------------
    for _ in range(s):
        R = R @ R

    return R


def diagonal(
    A: Tensor,
    *,
    offset: int = 0,
    dim1: int = -2,
    dim2: int = -1,
) -> Tensor:
    r"""Extract a (off-)diagonal from each matrix in a batched input.

    Returns the elements of :math:`A` lying on the diagonal selected by
    ``offset`` from the matrix slice formed by ``(dim1, dim2)``.  For
    a 2-D matrix and ``offset=0`` this is

    .. math::

        d_i \,=\, A_{i,\,i}.

    Positive ``offset`` selects super-diagonals (:math:`A_{i, i +
    \text{offset}}`); negative offsets select sub-diagonals.

    Parameters
    ----------
    A : Tensor
        Input of shape ``(*, m, n)`` (or higher rank).
    offset : int, keyword-only, optional
        Diagonal index relative to the main diagonal.  Default ``0``.
    dim1 : int, keyword-only, optional
        First matrix dimension.  Default ``-2``.
    dim2 : int, keyword-only, optional
        Second matrix dimension.  Default ``-1``.

    Returns
    -------
    Tensor
        Diagonal values with the two matrix axes replaced by a single
        axis of length :math:`\min(m, n) - |\text{offset}|`.

    Notes
    -----
    Shares the engine kernel with the top-level :func:`lucid.diagonal`;
    the only difference is the keyword-only / matrix-aware defaults
    (``dim1=-2``, ``dim2=-1``) that match the standard linalg
    convention.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import diagonal
    >>> A = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
    >>> diagonal(A)
    tensor([1., 4.])
    >>> diagonal(A, offset=1)
    tensor([2.])
    """
    return lucid.diagonal(A, offset=offset, dim1=dim1, dim2=dim2)


# The ``*_ex`` family reads LAPACK's ``info``; ``lu_factor`` and
# ``ldl_factor`` above are its checked forms.
from lucid.linalg._ex import (
    _check_matrix_norm_dims,
    _complete_qr_factors,
    _complete_svd_factors,
    _cond_compute,
    _EighResult,
    _EigResult,
    _householder_product,
    _LstsqResult,
    _lstsq_solution,
    _lu_factor_ex,
    _lu_solve_general,
    _LuFactorResult,
    _LuResult,
    _matrix_norm_sv,
    _pinv_with_tol,
    _QrResult,
    _SlogdetResult,
    _sv_cutoff,
    cholesky_ex,
    inv_ex,
    ldl_factor,
    ldl_factor_ex,
    lu_factor_ex,
    solve_ex,
    tensorinv,
    tensorsolve,
)

__all__ = [
    "LinAlgError",
    "inv",
    "det",
    "solve",
    "cholesky",
    "norm",
    "qr",
    "svd",
    "svdvals",
    "matrix_power",
    "pinv",
    "eig",
    "eigvals",
    "eigh",
    "eigvalsh",
    "slogdet",
    "matrix_rank",
    "cond",
    "multi_dot",
    "vander",
    "lu_factor",
    "lu_factor_ex",
    "solve_triangular",
    "cross",
    "vecdot",
    "dot",
    "inner",
    "outer",
    "matrix_norm",
    "vector_norm",
    "lstsq",
    "lu_solve",
    "householder_product",
    "ldl_factor",
    "ldl_factor_ex",
    "ldl_solve",
    "lu",
    "cholesky_ex",
    "inv_ex",
    "solve_ex",
    "tensorinv",
    "tensorsolve",
    "matrix_exp",
    "diagonal",
]
