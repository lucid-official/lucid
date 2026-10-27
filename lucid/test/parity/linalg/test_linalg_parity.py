"""Reference parity for linalg ops."""

from typing import Any

import numpy as np
import pytest

import lucid
import lucid.linalg as LA
from lucid.test._helpers.compare import assert_close


def _spd(n: int = 4, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n)).astype(np.float32)
    return (M @ M.T + n * np.eye(n)).astype(np.float32)


def _sym(n: int = 4, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n)).astype(np.float32)
    return (M + M.T).astype(np.float32)


@pytest.mark.parity
class TestLinalgParity:
    def test_norm_l2(self, ref: Any) -> None:
        v = np.array([3.0, 4.0], dtype=np.float32)
        assert_close(
            LA.norm(lucid.tensor(v.copy())),
            ref.linalg.norm(ref.tensor(v.copy())),
            atol=1e-5,
        )

    def test_solve(self, ref: Any) -> None:
        A = _spd()
        b = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
        assert_close(
            LA.solve(lucid.tensor(A.copy()), lucid.tensor(b.copy())),
            ref.linalg.solve(ref.tensor(A.copy()), ref.tensor(b.copy())),
            atol=1e-3,
        )

    def test_inv(self, ref: Any) -> None:
        A = _spd()
        assert_close(
            LA.inv(lucid.tensor(A.copy())),
            ref.linalg.inv(ref.tensor(A.copy())),
            atol=1e-3,
        )

    def test_cholesky_reconstruction(self, ref: Any) -> None:
        A_np = _spd()
        A_l = lucid.tensor(A_np.copy())
        A_r = ref.tensor(A_np.copy())
        assert_close(
            LA.cholesky(A_l) @ LA.cholesky(A_l).mT,
            ref.linalg.cholesky(A_r) @ ref.linalg.cholesky(A_r).mT,
            atol=1e-3,
        )


@pytest.mark.parity
class TestLinalgNewOpsParity:
    """P6 additions: lu, cholesky_ex, inv_ex, solve_ex, diagonal."""

    def test_lu_reconstruction(self, ref: Any) -> None:
        A_np = _spd()
        lP, lL, lU = LA.lu(lucid.tensor(A_np.copy()))
        rP, rL, rU = ref.linalg.lu(ref.tensor(A_np.copy()))
        # P·L·U must equal A for both frameworks (sign of P rows may differ).
        l_recon = lP @ lL @ lU
        r_recon = rP @ rL @ rU
        assert_close(l_recon, r_recon, atol=1e-4)

    def test_lu_shapes(self, ref: Any) -> None:
        A_np = _spd(3)
        P, L, U = LA.lu(lucid.tensor(A_np.copy()))
        assert P.shape == (3, 3)
        assert L.shape == (3, 3)
        assert U.shape == (3, 3)

    def test_cholesky_ex_value(self, ref: Any) -> None:
        A_np = _spd()
        lL, linfo = LA.cholesky_ex(lucid.tensor(A_np.copy()))
        rL, rinfo = ref.linalg.cholesky_ex(ref.tensor(A_np.copy()))
        assert_close(lL, rL, atol=1e-4)
        assert linfo.item() == rinfo.item()  # both 0 for PD matrix

    def test_cholesky_ex_info_singular(self, ref: Any) -> None:
        # Rank-deficient → info should be non-zero.
        A_np = np.zeros((3, 3), dtype=np.float32)
        _, linfo = LA.cholesky_ex(lucid.tensor(A_np.copy()))
        _, rinfo = ref.linalg.cholesky_ex(ref.tensor(A_np.copy()))
        assert linfo.item() != 0
        assert rinfo.item() != 0

    def test_inv_ex_value(self, ref: Any) -> None:
        A_np = _spd()
        lInv, linfo = LA.inv_ex(lucid.tensor(A_np.copy()))
        rInv, rinfo = ref.linalg.inv_ex(ref.tensor(A_np.copy()))
        assert_close(lInv, rInv, atol=1e-3)
        assert linfo.item() == rinfo.item()

    def test_solve_ex_value(self, ref: Any) -> None:
        A_np = _spd()
        b_np = np.random.default_rng(0).standard_normal((4, 2)).astype(np.float32)
        lx, linfo = LA.solve_ex(lucid.tensor(A_np.copy()), lucid.tensor(b_np.copy()))
        rx, rinfo = ref.linalg.solve_ex(ref.tensor(A_np.copy()), ref.tensor(b_np.copy()))
        assert_close(lx, rx, atol=1e-3)
        assert linfo.item() == rinfo.item()

    def test_diagonal(self, ref: Any) -> None:
        A_np = _spd()
        l = LA.diagonal(lucid.tensor(A_np.copy()), offset=0, dim1=-2, dim2=-1)
        r = ref.linalg.diagonal(ref.tensor(A_np.copy()), offset=0, dim1=-2, dim2=-1)
        assert_close(l, r, atol=1e-5)

    def test_diagonal_offset(self, ref: Any) -> None:
        A_np = _spd()
        l = LA.diagonal(lucid.tensor(A_np.copy()), offset=1, dim1=-2, dim2=-1)
        r = ref.linalg.diagonal(ref.tensor(A_np.copy()), offset=1, dim1=-2, dim2=-1)
        assert_close(l, r, atol=1e-5)


@pytest.mark.parity
class TestLinalgDecompParity:
    """Existing decomps: det, slogdet, qr, svd, eigh."""

    def test_det(self, ref: Any) -> None:
        A_np = _spd()
        l = LA.det(lucid.tensor(A_np.copy()))
        r = ref.linalg.det(ref.tensor(A_np.copy()))
        assert_close(l, r, atol=1e-2)

    def test_slogdet(self, ref: Any) -> None:
        A_np = _spd()
        l_sign, l_logabs = LA.slogdet(lucid.tensor(A_np.copy()))
        r_sign, r_logabs = ref.linalg.slogdet(ref.tensor(A_np.copy()))
        assert_close(l_logabs, r_logabs, atol=1e-3)
        assert_close(l_sign, r_sign, atol=1e-5)

    def test_qr_reconstruction(self, ref: Any) -> None:
        np.random.seed(2)
        A_np = np.random.randn(4, 3).astype(np.float32)
        lQ, lR = LA.qr(lucid.tensor(A_np.copy()))
        rQ, rR = ref.linalg.qr(ref.tensor(A_np.copy()))
        # Q·R must equal A (sign ambiguity in columns).
        assert_close(lQ @ lR, rQ @ rR, atol=1e-4)

    def test_svd_reconstruction(self, ref: Any) -> None:
        np.random.seed(3)
        A_np = np.random.randn(4, 3).astype(np.float32)
        lU, lS, lVh = LA.svd(lucid.tensor(A_np.copy()), full_matrices=False)
        rU, rS, rVh = ref.linalg.svd(ref.tensor(A_np.copy()), full_matrices=False)
        # Singular values are unique.
        assert_close(lS, rS, atol=1e-4)

    def test_eigh_eigenvalues(self, ref: Any) -> None:
        A_np = _spd()
        l_vals, _ = LA.eigh(lucid.tensor(A_np.copy()))
        r_vals, _ = ref.linalg.eigh(ref.tensor(A_np.copy()))
        assert_close(l_vals, r_vals, atol=1e-3)
