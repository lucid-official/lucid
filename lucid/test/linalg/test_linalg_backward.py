"""Phase 7 — linalg backward audit.

Verifies that analytical gradients from backward() match finite-difference
estimates for all linalg ops that support differentiation:

  * Already had correct backward: inv, det, norm, matrix_power
  * Newly added backward:        cholesky, eigh, svd, qr

Uses lucid.autograd.gradcheck (eps=1e-5, atol=2e-3) throughout.
"""

import lucid
import lucid.linalg as LA
from lucid.autograd import gradcheck

# ── helpers ───────────────────────────────────────────────────────────────────


def _spd2() -> lucid.Tensor:
    """2×2 symmetric positive-definite matrix."""
    return lucid.tensor([[4.0, 1.0], [1.0, 3.0]], requires_grad=True)


def _spd3() -> lucid.Tensor:
    """3×3 symmetric positive-definite matrix."""
    return lucid.tensor(
        [[5.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]],
        requires_grad=True,
    )


def _sym(A: lucid.Tensor) -> lucid.Tensor:
    """``A`` symmetrised, the input a Cholesky gradient check differentiates.

    The factor reads only one triangle (LCD-165, as the reference's does), so
    a coordinate-wise difference in the other one moves nothing, while the
    gradient is the symmetric one.  Checking through ``(A + Aᵀ) / 2`` asks
    about the symmetric matrices the gradient is defined on, which is how the
    reference checks its own.
    """
    return (A + A.mT) * 0.5


def _sq2() -> lucid.Tensor:
    """2×2 invertible matrix."""
    return lucid.tensor([[3.0, 1.0], [1.0, 2.0]], requires_grad=True)


def _rect32() -> lucid.Tensor:
    """3×2 full-column-rank matrix."""
    return lucid.tensor([[3.0, 1.0], [1.0, 2.0], [0.0, 1.0]], requires_grad=True)


def _rect23() -> lucid.Tensor:
    """2×3 full-row-rank matrix (m < n for SVD off-range coverage)."""
    return lucid.tensor([[3.0, 1.0, 0.0], [1.0, 2.0, 1.0]], requires_grad=True)


# ── Cholesky ──────────────────────────────────────────────────────────────────


class TestCholeskyBackward:
    def test_2x2_lower(self) -> None:
        ok = gradcheck(lambda A: LA.cholesky(_sym(A)).sum(), [_spd2()], eps=1e-5, atol=2e-3)
        assert ok

    def test_3x3_lower(self) -> None:
        ok = gradcheck(lambda A: LA.cholesky(_sym(A)).sum(), [_spd3()], eps=1e-5, atol=2e-3)
        assert ok

    def test_upper(self) -> None:
        ok = gradcheck(
            lambda A: LA.cholesky(_sym(A), upper=True).sum(),
            [_spd2()],
            eps=1e-5,
            atol=2e-3,
        )
        assert ok

    def test_combined_loss(self) -> None:
        def fn(A: lucid.Tensor) -> lucid.Tensor:
            L = LA.cholesky(_sym(A))
            return (L * L).sum() + L.sum()

        ok = gradcheck(fn, [_spd2()], eps=1e-5, atol=2e-3)
        assert ok


# ── SVD ───────────────────────────────────────────────────────────────────────


class TestSVDBackward:
    def test_S_only_rect(self) -> None:
        ok = gradcheck(lambda A: LA.svd(A)[1].sum(), [_rect23()], eps=1e-5, atol=2e-3)
        assert ok

    def test_U_only_rect(self) -> None:
        ok = gradcheck(lambda A: (LA.svd(A)[0] ** 2).sum(), [_rect23()], eps=1e-5, atol=2e-3)
        assert ok

    def test_Vh_only_rect(self) -> None:
        ok = gradcheck(lambda A: (LA.svd(A)[2] ** 2).sum(), [_rect23()], eps=1e-5, atol=2e-3)
        assert ok

    def test_all_three_rect(self) -> None:
        def fn(A: lucid.Tensor) -> lucid.Tensor:
            U, S, Vh = LA.svd(A)
            return S.sum() + (U**2).sum() + (Vh**2).sum()

        ok = gradcheck(fn, [_rect23()], eps=1e-5, atol=2e-3)
        assert ok

    def test_S_square(self) -> None:
        ok = gradcheck(lambda A: LA.svd(A)[1].sum(), [_sq2()], eps=1e-5, atol=2e-3)
        assert ok

    def test_svdvals(self) -> None:
        ok = gradcheck(lambda A: LA.svdvals(A).sum(), [_rect23()], eps=1e-5, atol=2e-3)
        assert ok


# ── Eigh ──────────────────────────────────────────────────────────────────────


class TestEighBackward:
    def test_eigenvalues_2x2(self) -> None:
        ok = gradcheck(lambda A: LA.eigh(A)[0].sum(), [_spd2()], eps=1e-5, atol=2e-3)
        assert ok

    def test_eigenvectors_2x2(self) -> None:
        ok = gradcheck(lambda A: (LA.eigh(A)[1] ** 2).sum(), [_spd2()], eps=1e-5, atol=2e-3)
        assert ok

    def test_both_2x2(self) -> None:
        def fn(A: lucid.Tensor) -> lucid.Tensor:
            w, V = LA.eigh(A)
            return w.sum() + (V**2).sum()

        ok = gradcheck(fn, [_spd2()], eps=1e-5, atol=2e-3)
        assert ok

    def test_eigenvalues_3x3(self) -> None:
        ok = gradcheck(lambda A: LA.eigh(A)[0].sum(), [_spd3()], eps=1e-5, atol=2e-3)
        assert ok

    def test_eigvalsh(self) -> None:
        ok = gradcheck(lambda A: LA.eigvalsh(A).sum(), [_spd2()], eps=1e-5, atol=2e-3)
        assert ok


# ── QR ────────────────────────────────────────────────────────────────────────


class TestQRBackward:
    def test_R_square(self) -> None:
        ok = gradcheck(lambda A: LA.qr(A)[1].sum(), [_sq2()], eps=1e-5, atol=2e-3)
        assert ok

    def test_R_rect(self) -> None:
        ok = gradcheck(lambda A: LA.qr(A)[1].sum(), [_rect32()], eps=1e-5, atol=2e-3)
        assert ok

    def test_Q_square(self) -> None:
        ok = gradcheck(lambda A: (LA.qr(A)[0] ** 2).sum(), [_sq2()], eps=1e-5, atol=2e-3)
        assert ok

    def test_Q_rect(self) -> None:
        ok = gradcheck(lambda A: (LA.qr(A)[0] ** 2).sum(), [_rect32()], eps=1e-5, atol=2e-3)
        assert ok

    def test_both_square(self) -> None:
        def fn(A: lucid.Tensor) -> lucid.Tensor:
            Q, R = LA.qr(A)
            return R.sum() + (Q**2).sum()

        ok = gradcheck(fn, [_sq2()], eps=1e-5, atol=2e-3)
        assert ok

    def test_both_rect(self) -> None:
        def fn(A: lucid.Tensor) -> lucid.Tensor:
            Q, R = LA.qr(A)
            return R.sum() + (Q**2).sum()

        ok = gradcheck(fn, [_rect32()], eps=1e-5, atol=2e-3)
        assert ok


# ── Existing ops (regression) ─────────────────────────────────────────────────


class TestExistingBackwards:
    def test_inv(self) -> None:
        ok = gradcheck(lambda A: LA.inv(A).sum(), [_sq2()], eps=1e-5, atol=2e-3)
        assert ok

    def test_det(self) -> None:
        ok = gradcheck(lambda A: LA.det(A), [_sq2()], eps=1e-5, atol=2e-3)
        assert ok

    def test_matrix_power(self) -> None:
        A = lucid.tensor([[2.0, 1.0], [0.0, 2.0]], requires_grad=True)
        ok = gradcheck(lambda A: LA.matrix_power(A, 3).sum(), [A], eps=1e-5, atol=2e-3)
        assert ok

    def test_cholesky_ex(self) -> None:
        # cholesky_ex wraps cholesky → backward flows through.
        ok = gradcheck(
            lambda A: LA.cholesky_ex(_sym(A))[0].sum(),
            [_spd2()],
            eps=1e-5,
            atol=2e-3,
        )
        assert ok
