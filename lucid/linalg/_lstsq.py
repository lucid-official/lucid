"""``lstsq``: QR for a full-rank ``A``, the SVD otherwise.

The reference's default driver (``gelsy``) solves a full-rank system through
a QR factorisation and spends the rank-revealing work only on a matrix that
needs it.  ``gels`` and ``gelsy`` (so the default) do the same: the QR route
is taken only when ``A`` is *certified* full rank, and answers the same
``A⁺ b`` the SVD route does — ``R⁻¹ Qᵀ b`` for ``m ≥ n``, ``Q R⁻ᵀ b`` from the
QR of ``Aᵀ`` for ``m < n``.  A matrix the certificate does not clear (rank
deficient, ill-conditioned, non-finite) goes to the SVD route, so a rank
decision is never made by the cheap route.  ``gelsd`` / ``gelss`` return the
singular values and always take the SVD route.

Re-exported by :mod:`lucid.linalg`, which owns ``qr``, ``svd`` and the
triangular solves this reads.
"""

from typing import NamedTuple, Self

import lucid
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid._dtype import finfo as _finfo
from lucid._ops._autocast import _autocast_entry
from lucid._tensor._indexing import broadcast_shapes as _broadcast_batch
from lucid._tensor.tensor import Tensor
from lucid.linalg import _Pinv, qr, solve_triangular
from lucid.linalg._ex import _pinv_with_tol, _sv_cutoff


class _LstsqResult(NamedTuple):
    solution: Tensor
    residuals: Tensor
    rank: Tensor
    singular_values: Tensor


def _check_lstsq_shapes(
    a_shape: tuple[int, ...], b_shape: tuple[int, ...]
) -> tuple[bool, tuple[int, ...]]:
    """Return ``(vec_rhs, b_batch)`` or raise ``ValueError`` for malformed shapes."""
    a_ndim, b_ndim = len(a_shape), len(b_shape)
    if a_ndim < 2:
        raise ValueError(f"lstsq: A must have at least 2 dimensions, got shape {a_shape}")
    if b_ndim < 1 or not (0 <= a_ndim - b_ndim <= 1):
        raise ValueError(
            f"lstsq: A.ndim must be >= B.ndim and A.ndim - B.ndim <= 1, "
            f"got A of shape {a_shape} and B of shape {b_shape}"
        )
    vec_rhs = (b_ndim == 1) or (a_ndim - 1 == b_ndim and a_shape[:-2] == b_shape[:-1])
    b_rows = b_shape[-1] if vec_rhs else (b_shape[-2] if b_ndim >= 2 else -1)
    if b_rows != a_shape[-2]:
        raise ValueError(
            f"lstsq: B must have one row per row of A — A is {a_shape} "
            f"({a_shape[-2]} rows), B is {b_shape}"
        )
    b_batch = b_shape[:-1] if a_ndim > b_ndim else b_shape[:-2]
    return vec_rhs, b_batch


_LSTSQ_DRIVERS = frozenset({None, "gels", "gelsy", "gelsd", "gelss"})

#: Drivers that decide nothing from the singular values, so the QR route may
#: answer them: ``gels`` assumes full rank, ``gelsy`` reports the rank only.
_QR_DRIVERS = frozenset({"gels", "gelsy"})


def _lstsq_rank_and_residuals(
    A: Tensor,
    b_mat: Tensor,
    sol_mat: Tensor,
    sv: Tensor,
    rank: Tensor,
    batch: tuple[int, ...],
    drv: str,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute ``(residuals, rank, singular_values)`` for ``lstsq`` under ``drv``."""
    empty_f = _wrap(_C_engine.zeros([0], _unwrap(A).dtype, _unwrap(A).device))
    empty_i = _wrap(_C_engine.zeros([0], _C_engine.I64, _unwrap(A).device))
    if drv == "gelsy":
        return empty_f, rank, empty_f
    m, n, k = int(A.shape[-2]), int(A.shape[-1]), int(b_mat.shape[-1])
    full_rank = bool((rank == n).all().item()) if rank.numel() > 0 else True
    if m > n and (drv == "gels" or full_rank):
        if m == 0 or k == 0 or 0 in batch:
            residuals = lucid.zeros(*batch, k, dtype=A.dtype, device=A.device)
        elif n == 0:
            # The solution is empty, so the residual is all of ``b`` for the SVD
            # drivers; the reference's ``gels`` reports zeros here.
            residuals = (
                lucid.zeros(*batch, k, dtype=A.dtype, device=A.device)
                if drv == "gels"
                else (b_mat.abs() ** 2).sum(dim=-2)
            )
        else:
            diff = lucid.matmul(A, sol_mat) - b_mat
            residuals = (diff.abs() ** 2).sum(dim=-2)
    else:
        residuals = empty_f
    if drv == "gels":
        return residuals, empty_i, empty_f
    return residuals, rank, sv


def _non_finite_batches(a: Tensor) -> Tensor:
    """Which matrices of the batch ``a`` hold a NaN or an infinity."""
    if a.numel() == 0:
        return lucid.zeros(a.shape[:-2], dtype=lucid.bool, device=a.device)
    return ~lucid.isfinite(a).all(dim=(-2, -1))


# ── The QR route ─────────────────────────────────────────────────────────────

#: Power / inverse iterations of the conditioning estimate.  The estimate is a
#: lower bound on the true condition number that tightens with each, and four
#: from a fixed generic vector land within a small factor on the matrices a
#: least-squares solve meets.
_ESTIMATE_STEPS = 4

#: A matrix goes to the QR route only when its estimated condition number is
#: this many times *inside* the rank cutoff.  The estimate undershoots the true
#: number (measured on Gaussian, graded, Kahan, Hilbert and near-duplicate-column
#: matrices: 0.66-1.05 of it), so a factor of two keeps an under-converged one
#: from calling a deficient matrix full rank; a matrix in the band between goes
#: to the SVD route, which decides it.
_ESTIMATE_MARGIN = 2.0

#: Start vector of both iterations: generic, deterministic, and cheap.
_START_PHASE = 1.618033988749895


def _unit_columns(v: Tensor) -> Tensor:
    """``v`` scaled to unit 2-norm along its column axis."""
    return v / lucid.sqrt((v * v).sum(dim=-2, keepdim=True))


def _estimate_condition(R: Tensor) -> Tensor:
    """Estimate ``σ_max / σ_min`` of each triangular ``R`` of the batch.

    Both extremes come from a few power steps on ``RᵀR`` (``σ_max``) and on its
    inverse through two triangular solves (``σ_min``), so the cost is
    ``O(k²)`` per step against the factorisation's ``O(k³)``.  ``R`` is scaled
    to unit magnitude first, so a matrix of ``1e±20`` entries neither
    underflows the inverse step nor overflows it.  A singular ``R`` divides by
    its zero diagonal and comes out ``inf`` / ``nan``, which no comparison
    against a cutoff passes.
    """
    k = int(R.shape[-1])
    scale = R.abs().amax(dim=(-2, -1), keepdim=True)
    Rn = R / scale
    phase = lucid.arange(k, dtype=R.dtype, device=R.device) * _START_PHASE + 1.0
    start = lucid.cos(phase).reshape(k, 1).expand(*tuple(R.shape[:-2]), k, 1)
    Rt = Rn.mT
    inv_v = _unit_columns(start)
    inv_gain = lucid.ones_like(scale)
    for _ in range(_ESTIMATE_STEPS):
        y = solve_triangular(Rn, solve_triangular(Rt, inv_v, upper=False), upper=True)
        inv_gain = lucid.sqrt((y * y).sum(dim=-2, keepdim=True))
        inv_v = y / inv_gain
    fwd_v = _unit_columns(start)
    fwd_gain = lucid.ones_like(scale)
    for _ in range(_ESTIMATE_STEPS):
        z = lucid.matmul(Rt, lucid.matmul(Rn, fwd_v))
        fwd_gain = lucid.sqrt((z * z).sum(dim=-2, keepdim=True))
        fwd_v = z / fwd_gain
    return lucid.sqrt(inv_gain * fwd_gain).reshape(tuple(R.shape[:-2]))


class _QrLeastSquares:
    """The minimum-norm least-squares solve of full-rank matrices by QR.

    ``A = Q R`` for a tall ``A``; a wide one factors ``Aᵀ = Q R``, so that
    ``A = Rᵀ Qᵀ`` and ``A⁺ = Q R⁻ᵀ``.  Nothing here is differentiable: the
    gradient of ``A⁺`` is ``pinv``'s closed form, which reads only ``A`` and
    the answer, so :func:`_qr_solve` attaches it around :meth:`pinv`.
    """

    def __init__(self, tall: bool, q: Tensor, r: Tensor) -> None:
        self._tall, self._q, self._r = tall, q, r

    @classmethod
    def of(cls, A: Tensor) -> Self:
        tall = int(A.shape[-2]) >= int(A.shape[-1])
        with lucid.no_grad():
            q, r = qr(A.detach() if tall else A.detach().mT)
        return cls(tall, q, r)

    def select(self, index: Tensor) -> Self:
        """The solve of the matrices ``index`` names, of a flattened batch."""
        return type(self)(
            self._tall, self._q.index_select(0, index), self._r.index_select(0, index)
        )

    def certified(self, rtol: float) -> list[bool]:
        """Which matrices of the flattened batch are certified full rank.

        The cutoff keeps ``σ`` above ``rtol · σ_max``, so full rank is
        ``κ < 1 / rtol``.  The estimate is made on the host: the answer is one
        data-dependent bit per matrix, and a Metal tensor would pay a device
        round trip for each of the dozen small solves.
        """
        with lucid.no_grad():
            r = self._r if self._r.device.type == "cpu" else self._r.to("cpu")
            kappa = _estimate_condition(r.reshape(-1, *tuple(r.shape[-2:])))
            flags = (kappa * (rtol * _ESTIMATE_MARGIN) < 1.0).tolist()
        assert isinstance(flags, list)
        return [bool(v) for v in flags]

    def solve(self, b: Tensor) -> Tensor:
        """``A⁺ b`` without forming ``A⁺``."""
        with lucid.no_grad():
            if self._tall:
                return solve_triangular(self._r, lucid.matmul(self._q.mT, b), upper=True)
            return lucid.matmul(self._q, solve_triangular(self._r.mT, b, upper=False))

    def pinv(self) -> Tensor:
        """``A⁺`` itself, for a solve that must differentiate."""
        with lucid.no_grad():
            P = solve_triangular(self._r, self._q.mT, upper=True)
            return P if self._tall else P.mT.contiguous()


def _qr_solve(route: _QrLeastSquares, A: Tensor, b: Tensor) -> Tensor:
    """``A⁺ b`` from ``route``, differentiable in whichever of ``A``, ``b`` needs it."""
    if not (_C_engine.grad_enabled() and (A.requires_grad or b.requires_grad)):
        return route.solve(b)
    pinv_a = route.pinv()
    if A.requires_grad:
        value = pinv_a
        out = _Pinv.apply(A, lambda _: value)
        assert isinstance(out, Tensor)
        pinv_a = out
    return lucid.matmul(pinv_a, b)


def _default_rtol(A: Tensor, rcond: float | None) -> float:
    """The relative singular-value cutoff ``pinv`` and the rank both use."""
    if rcond is not None:
        return float(rcond)
    return max(int(A.shape[-2]), int(A.shape[-1])) * float(_finfo(A.dtype).eps)


def _index(rows: list[int], like: Tensor) -> Tensor:
    return lucid.tensor(rows, dtype=lucid.int64, device=like.device)


def _qr_route(
    A: Tensor, b_mat: Tensor, rcond: float | None, batch: tuple[int, ...]
) -> tuple[Tensor, Tensor] | None:
    """``(A⁺ b, rank)`` by QR where ``A`` is certified full rank, else ``None``.

    A batch is split by the certificate: the matrices it clears take the QR
    route, the rest take the SVD route, and the two are put back in order.  One
    ill-conditioned matrix in a batch of hundreds then costs one SVD, not the
    whole batch's.  ``None`` means the SVD route should answer everything — no
    matrix cleared, or an operand batch that was broadcast, which the split
    does not index.
    """
    m, n, k = int(A.shape[-2]), int(A.shape[-1]), int(b_mat.shape[-1])
    if min(m, n) == 0 or k == 0 or 0 in batch or A.dtype not in (lucid.float32, lucid.float64):
        return None
    route = _QrLeastSquares.of(A)
    ok = route.certified(_default_rtol(A, rcond))
    full = min(m, n)
    if all(ok):
        sol = _qr_solve(route, A, b_mat)
        return sol, lucid.full(batch, full, dtype=lucid.int64, device=A.device)
    same = tuple(A.shape[:-2]) == batch == tuple(b_mat.shape[:-2])
    if not any(ok) or not same:
        return None
    good = [i for i, c in enumerate(ok) if c]
    bad = [i for i, c in enumerate(ok) if not c]
    flat_a, flat_b = A.reshape(-1, m, n), b_mat.reshape(-1, m, k)
    ig, ib = _index(good, A), _index(bad, A)
    sol_good = _qr_solve(route.select(ig), flat_a.index_select(0, ig), flat_b.index_select(0, ig))
    sol_bad, rank_bad, _ = _svd_route(flat_a.index_select(0, ib), flat_b.index_select(0, ib), rcond)
    rank_good = lucid.full((len(good),), full, dtype=lucid.int64, device=A.device)
    place = [0] * len(ok)
    for pos, i in enumerate(good + bad):
        place[i] = pos
    order = _index(place, A)
    sol = lucid.cat([sol_good, sol_bad], dim=0).index_select(0, order)
    rank = lucid.cat([rank_good, rank_bad], dim=0).index_select(0, order)
    return sol.reshape(*batch, n, k), rank.reshape(batch)


# ── The SVD route ────────────────────────────────────────────────────────────


def _svd_route(A: Tensor, b_mat: Tensor, rcond: float | None) -> tuple[Tensor, Tensor, Tensor]:
    """``(A⁺ b, rank, singular values)`` through the SVD.

    ``pinv`` refuses a non-finite matrix (LCD-674), but a least-squares solve
    over one is NaN, so only that batch element is masked around the
    factorization.
    """
    a_det = A.detach()
    bad = _non_finite_batches(a_det)
    bad_mat = bad.unsqueeze(-1).unsqueeze(-1) if bool(bad.any().item()) else None
    fact_a = A if bad_mat is None else lucid.where(bad_mat, lucid.zeros_like(A), A)
    pinv_a = (
        lucid.linalg.pinv(fact_a)
        if rcond is None
        else _pinv_with_tol(fact_a, atol=None, rtol=rcond, hermitian=False)
    )
    sol_mat = lucid.matmul(pinv_a, b_mat)
    with lucid.no_grad():
        safe_a = a_det if bad_mat is None else lucid.where(bad_mat, lucid.zeros_like(a_det), a_det)
        sv = lucid.linalg.svdvals(safe_a)
        cutoff = _sv_cutoff(safe_a, sv, atol=None, rtol=rcond, op="lstsq")
        rank = (sv > cutoff).to(lucid.int64).sum(dim=-1)
        if bad_mat is not None:
            rank = lucid.where(bad, lucid.zeros_like(rank), rank)
            sv = lucid.where(bad.unsqueeze(-1), lucid.full_like(sv, float("nan")), sv)
    if bad_mat is not None:
        sol_mat = lucid.where(bad_mat, lucid.full_like(sol_mat, float("nan")), sol_mat)
    return sol_mat, rank, sv


def _lstsq_solution(
    A: Tensor, B: Tensor, rcond: float | None = None, driver: str | None = None
) -> _LstsqResult:
    """Validate shapes and compute ``(solution, residuals, rank, singular_values)``."""
    if driver not in _LSTSQ_DRIVERS:
        raise ValueError(
            f"linalg.lstsq: invalid driver {driver!r}; expected one of "
            "None, 'gels', 'gelsy', 'gelsd', 'gelss'"
        )
    drv = ("gelsy" if A.device.type == "cpu" else "gels") if driver is None else driver
    a_shape = tuple(int(s) for s in A.shape)
    b_shape = tuple(int(s) for s in B.shape)
    vec_rhs, b_batch = _check_lstsq_shapes(a_shape, b_shape)
    batch = _broadcast_batch(a_shape[:-2], b_batch)
    n = a_shape[-1]
    b_mat = B.unsqueeze(-1) if vec_rhs else B
    k = int(b_mat.shape[-1])
    fast = _qr_route(A, b_mat, rcond, batch) if drv in _QR_DRIVERS else None
    if fast is not None:
        sol_mat, rank = fast
        # ``gels`` and ``gelsy`` return no singular values.
        sv = _wrap(_C_engine.zeros([0], _unwrap(A).dtype, _unwrap(A).device))
    else:
        sol_mat, rank, sv = _svd_route(A, b_mat, rcond)
    target_shape = (*batch, n, k)
    if tuple(int(s) for s in sol_mat.shape) != target_shape:
        sol_mat = sol_mat.expand(*target_shape).contiguous()
    with lucid.no_grad():
        residuals, rank_out, sv_out = _lstsq_rank_and_residuals(
            A.detach(), b_mat.detach(), sol_mat.detach(), sv, rank, batch, drv
        )
    sol = sol_mat.squeeze(-1) if vec_rhs else sol_mat
    return _LstsqResult(sol, residuals, rank_out, sv_out)


def lstsq(
    input: Tensor, b: Tensor, rcond: float | None = None, *, driver: str | None = None
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

    ``"gels"`` and ``"gelsy"`` (so the default) factor a full-rank ``input``
    with a QR decomposition, as the reference's default driver does.  Every
    matrix is first checked for full rank with a condition-number estimate; one
    that does not pass — rank deficient, ill-conditioned, non-finite — takes the
    SVD route, which decides the rank.  ``"gelsd"`` and ``"gelss"`` return the
    singular values and always take the SVD route.

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
        return _lstsq_solution(_wrap(ai), _wrap(bi), rcond, driver)
