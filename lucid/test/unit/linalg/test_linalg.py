"""``lucid.linalg`` — decomposition / norm / solve / inverse."""

import numpy as np
import pytest

import lucid


def _spd(seed: int = 0) -> lucid.Tensor:
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((4, 4)).astype(np.float32)
    return lucid.tensor((M @ M.T + 4 * np.eye(4)).astype(np.float32))


class TestCholesky:
    def test_reconstruction(self) -> None:
        A = _spd()
        L = lucid.linalg.cholesky(A)
        recon = (L @ L.mT).numpy()
        np.testing.assert_allclose(recon, A.numpy(), atol=1e-3)

    def test_lower_triangular(self) -> None:
        L = lucid.linalg.cholesky(_spd()).numpy()
        # Strictly upper triangle is zero.
        for i in range(4):
            for j in range(i + 1, 4):
                assert abs(L[i, j]) < 1e-5


class TestCholeskyEx:
    def test_success_zero_info(self) -> None:
        L, info = lucid.linalg.cholesky_ex(_spd())
        assert int(info.item()) == 0

    def test_failure_nonzero_info(self) -> None:
        zeroA = lucid.zeros(3, 3)
        _, info = lucid.linalg.cholesky_ex(zeroA)
        assert int(info.item()) != 0


class TestQR:
    def test_decomposition(self) -> None:
        A = _spd()
        Q, R = lucid.linalg.qr(A)
        recon = (Q @ R).numpy()
        np.testing.assert_allclose(recon, A.numpy(), atol=1e-4)

    def test_q_orthonormal(self) -> None:
        Q, _ = lucid.linalg.qr(_spd())
        QtQ = (Q.mT @ Q).numpy()
        np.testing.assert_allclose(QtQ, np.eye(4), atol=1e-4)


class TestSVD:
    def test_basic(self) -> None:
        A = _spd()
        out = lucid.linalg.svd(A)
        # Engine surface returns either (U, S, Vt) or a single SVD object.
        if isinstance(out, tuple) and len(out) == 3:
            U, S, Vt = out
            recon = (U @ lucid.tensor(np.diag(S.numpy()), dtype=A.dtype) @ Vt).numpy()
            np.testing.assert_allclose(recon, A.numpy(), atol=1e-3)


class TestEig:
    def test_symmetric_real_eigenvalues(self) -> None:
        # SPD matrix has all-positive real eigenvalues.
        A = _spd()
        out = lucid.linalg.eigh(A) if hasattr(lucid.linalg, "eigh") else None
        if out is None:
            pytest.skip("eigh not implemented")
        if isinstance(out, tuple):
            evals = out[0].numpy()
        else:
            evals = out.numpy()
        assert np.all(evals > 0)


class TestSolve:
    def test_basic(self) -> None:
        A = _spd()
        b = lucid.tensor([[1.0], [2.0], [3.0], [4.0]])
        x = lucid.linalg.solve(A, b)
        np.testing.assert_allclose((A @ x).numpy(), b.numpy(), atol=1e-3)


class TestInv:
    def test_round_trip(self) -> None:
        A = _spd()
        Ai = lucid.linalg.inv(A)
        np.testing.assert_allclose((A @ Ai).numpy(), np.eye(4), atol=1e-3)


class TestInvEx:
    def test_success(self) -> None:
        A = _spd()
        Ai, info = lucid.linalg.inv_ex(A)
        assert int(info.item()) == 0
        np.testing.assert_allclose((A @ Ai).numpy(), np.eye(4), atol=1e-3)

    def test_singular(self) -> None:
        S = lucid.zeros(3, 3)
        _, info = lucid.linalg.inv_ex(S)
        assert int(info.item()) != 0


class TestSolveEx:
    def test_success(self) -> None:
        A = _spd()
        B = lucid.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]])
        X, info = lucid.linalg.solve_ex(A, B)
        assert int(info.item()) == 0
        np.testing.assert_allclose((A @ X).numpy(), B.numpy(), atol=1e-3)


class TestLU:
    def test_pl_u_recovers(self) -> None:
        A = lucid.tensor([[4.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 2.0]])
        P, L, U = lucid.linalg.lu(A)
        np.testing.assert_allclose((P @ L @ U).numpy(), A.numpy(), atol=1e-4)


class TestDiagonal:
    def test_basic(self) -> None:
        A = lucid.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        np.testing.assert_array_equal(lucid.linalg.diagonal(A).numpy(), [1.0, 5.0, 9.0])

    def test_offset(self) -> None:
        A = lucid.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        np.testing.assert_array_equal(lucid.linalg.diagonal(A, offset=1).numpy(), [2.0, 6.0])


class TestNorm:
    def test_l2_vector(self) -> None:
        v = lucid.tensor([3.0, 4.0])
        assert abs(lucid.linalg.norm(v).item() - 5.0) < 1e-5


class TestMatrixRank:
    def test_full_rank(self) -> None:
        if not hasattr(lucid.linalg, "matrix_rank"):
            pytest.skip("matrix_rank not exposed")
        I = lucid.eye(4)
        assert int(lucid.linalg.matrix_rank(I).item()) == 4


class TestOuter:
    def test_mixed_dtypes_meet_at_the_common_dtype(self, device: str) -> None:
        # The engine multiplies one dtype only, so an int64 vector beside a
        # float32 one raised DtypeMismatch; the reference answers in float32.
        ints = lucid.tensor([1, 2], dtype=lucid.int64, device=device)
        floats = lucid.tensor([0.5, 1.5], device=device)
        out = lucid.linalg.outer(ints, floats)
        assert out.dtype == lucid.float32
        assert out.tolist() == [[0.5, 1.5], [1.0, 3.0]]
        assert ints.outer(floats).tolist() == out.tolist()

    def test_integers_stay_exact(self, device: str) -> None:
        ints = lucid.tensor([1, 2], dtype=lucid.int64, device=device)
        out = lucid.linalg.outer(ints, ints)
        assert out.dtype == lucid.int64
        assert out.tolist() == [[1, 2], [2, 4]]
