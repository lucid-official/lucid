"""Linear algebra, checked by the identity each routine is defined by.

``linalg/__init__.py`` sat at 82.9% with 115 statements never run — the
decompositions past their happy path, the norm orders past the default,
and the small helpers.

A factorisation is the one kind of result that carries its own proof:
``Q @ R`` has to be the input, ``A @ v`` has to be ``lambda * v``, and
``L @ L.T`` has to be the matrix that went in.  Those are asserted here
rather than the factor values, because the factors are only unique up to
sign and ordering conventions and pinning them would test the convention
instead of the mathematics.
"""

import numpy as np
import pytest

import lucid
import lucid.linalg as linalg

RNG = np.random.default_rng(0)
A = RNG.standard_normal((4, 4))
SPD = A @ A.T + 4 * np.eye(4)
TALL = RNG.standard_normal((6, 4))
WIDE = RNG.standard_normal((4, 6))
VEC = RNG.standard_normal(4)


def _t(a):
    return lucid.tensor(np.asarray(a, dtype=np.float64))


def _v(x):
    return np.asarray(x.numpy())


# ── the routines with a closed form ───────────────────────────────────────────


def test_det_matches_numpy():
    assert np.isclose(_v(linalg.det(_t(A))), np.linalg.det(A))


def test_inv_undoes_the_matrix():
    assert np.allclose(_v(linalg.inv(_t(A))) @ A, np.eye(4), atol=1e-8)


def test_solve_solves():
    x = _v(linalg.solve(_t(A), _t(VEC)))
    assert np.allclose(A @ x, VEC, atol=1e-8)


def test_slogdet_agrees_with_det_without_overflowing():
    sign, logabs = linalg.slogdet(_t(A))
    want_sign, want_log = np.linalg.slogdet(A)
    assert np.isclose(_v(sign), want_sign)
    assert np.isclose(_v(logabs), want_log, atol=1e-8)
    assert np.isclose(float(_v(sign)) * np.exp(float(_v(logabs))), np.linalg.det(A))


def test_matrix_rank_counts_the_independent_directions():
    assert int(_v(linalg.matrix_rank(_t(A)))) == 4
    singular = A.copy()
    singular[1] = singular[0]
    assert int(_v(linalg.matrix_rank(_t(singular)))) == 3


def test_cond_is_the_ratio_of_the_extreme_singular_values():
    assert np.isclose(_v(linalg.cond(_t(A))), np.linalg.cond(A), rtol=1e-7)


@pytest.mark.parametrize("power", [0, 1, 2, 3, -1, -2])
def test_matrix_power_matches_repeated_multiplication(power):
    assert np.allclose(
        _v(linalg.matrix_power(_t(A), power)),
        np.linalg.matrix_power(A, power),
        atol=1e-7,
    )


def test_matrix_power_zero_is_the_identity():
    assert np.allclose(_v(linalg.matrix_power(_t(A), 0)), np.eye(4), atol=1e-12)


def test_the_pseudo_inverse_satisfies_the_moore_penrose_conditions():
    """Four conditions, and they are what *define* the pseudo-inverse —
    checking it against a formula would only re-test the formula."""
    for matrix in (TALL, WIDE):
        m = matrix
        p = _v(linalg.pinv(_t(m)))
        assert np.allclose(m @ p @ m, m, atol=1e-7)
        assert np.allclose(p @ m @ p, p, atol=1e-7)
        assert np.allclose((m @ p).T, m @ p, atol=1e-7)
        assert np.allclose((p @ m).T, p @ m, atol=1e-7)


def test_the_pseudo_inverse_of_an_invertible_matrix_is_the_inverse():
    assert np.allclose(_v(linalg.pinv(_t(A))), np.linalg.inv(A), atol=1e-7)


def test_lstsq_minimises_the_residual():
    target = RNG.standard_normal(6)
    solution = _v(linalg.lstsq(_t(TALL), _t(target))[0])
    residual = np.linalg.norm(TALL @ solution - target)
    for _ in range(20):
        nudged = solution + 1e-3 * RNG.standard_normal(solution.shape)
        assert np.linalg.norm(TALL @ nudged - target) >= residual - 1e-9


# ── the factorisations, checked by reconstruction ─────────────────────────────


def test_cholesky_reconstructs_a_positive_definite_matrix():
    factor = _v(linalg.cholesky(_t(SPD)))
    assert np.allclose(np.triu(factor, 1), 0.0, atol=1e-12)
    assert np.allclose(factor @ factor.T, SPD, atol=1e-8)


def test_cholesky_refuses_a_matrix_that_is_not_positive_definite():
    with pytest.raises(Exception):
        linalg.cholesky(_t(-SPD))


@pytest.mark.parametrize("matrix", [A, TALL, WIDE], ids=["square", "tall", "wide"])
def test_qr_reconstructs_and_is_orthonormal(matrix):
    q, r = linalg.qr(_t(matrix))
    q, r = _v(q), _v(r)
    assert np.allclose(q @ r, matrix, atol=1e-8)
    assert np.allclose(q.T @ q, np.eye(q.shape[1]), atol=1e-8)
    assert np.allclose(np.tril(r, -1), 0.0, atol=1e-10)


@pytest.mark.parametrize("matrix", [A, TALL, WIDE], ids=["square", "tall", "wide"])
def test_svd_reconstructs_with_non_negative_ordered_values(matrix):
    u, s, vh = linalg.svd(_t(matrix))
    u, s, vh = _v(u), _v(s), _v(vh)
    assert (s >= 0).all()
    assert (np.diff(s) <= 1e-12).all(), "singular values must be non-increasing"
    k = len(s)
    assert np.allclose(u[:, :k] @ np.diag(s) @ vh[:k], matrix, atol=1e-7)


def test_the_singular_values_are_the_square_roots_of_the_gram_eigenvalues():
    """An independent route to the same numbers."""
    s = _v(linalg.svd(_t(TALL))[1])
    gram = np.linalg.eigvalsh(TALL.T @ TALL)[::-1]
    assert np.allclose(s**2, gram, atol=1e-7)


def test_eig_satisfies_its_own_definition():
    values, vectors = linalg.eig(_t(A))
    values, vectors = _v(values), _v(vectors)
    assert np.allclose(A @ vectors, vectors * values[None, :], atol=1e-6)


def test_eigh_is_orthonormal_and_ordered():
    values, vectors = linalg.eigh(_t(SPD))
    values, vectors = _v(values), _v(vectors)
    assert np.allclose(vectors.T @ vectors, np.eye(4), atol=1e-9)
    assert (np.diff(values) >= -1e-9).all()
    assert np.allclose(SPD @ vectors, vectors * values[None, :], atol=1e-7)
    assert (values > 0).all()  # positive definite


def test_eigvals_agrees_with_eig():
    assert np.allclose(
        np.sort_complex(np.asarray(_v(linalg.eigvals(_t(A))), dtype=complex)),
        np.sort_complex(np.asarray(_v(linalg.eig(_t(A))[0]), dtype=complex)),
        atol=1e-8,
    )


def test_eigvalsh_agrees_with_eigh():
    assert np.allclose(_v(linalg.eigvalsh(_t(SPD))), _v(linalg.eigh(_t(SPD))[0]), atol=1e-9)


def test_the_determinant_is_the_product_of_the_eigenvalues():
    values = np.asarray(_v(linalg.eig(_t(A))[0]), dtype=complex)
    assert np.isclose(np.prod(values).real, float(_v(linalg.det(_t(A)))), atol=1e-8)


# ── norms ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("order", [1, 2, np.inf, -np.inf, "fro", "nuc"], ids=str)
def test_matrix_norm_matches_numpy(order):
    assert np.isclose(
        _v(linalg.matrix_norm(_t(A), ord=order)),
        np.linalg.norm(A, ord=order),
        rtol=1e-7,
    )


def test_the_matrix_norm_default_is_frobenius():
    assert np.isclose(_v(linalg.matrix_norm(_t(A))), np.linalg.norm(A, ord="fro"), rtol=1e-9)


def test_matrix_norm_refuses_an_order_it_has_no_meaning_for():
    """``ord=None`` means Frobenius to NumPy and is a type error to the
    reference.  Lucid follows the reference — the default already *is*
    Frobenius, so nothing is lost and the two conventions do not get
    quietly merged."""
    with pytest.raises((ValueError, TypeError)):
        linalg.matrix_norm(_t(A), ord=None)


@pytest.mark.parametrize("order", [1, 2, 3, np.inf, -np.inf, 0], ids=str)
def test_vector_norm_matches_numpy(order):
    assert np.isclose(
        _v(linalg.vector_norm(_t(VEC), ord=order)),
        np.linalg.norm(VEC, ord=order),
        rtol=1e-9,
    )


def test_the_nuclear_norm_is_the_sum_of_the_singular_values():
    assert np.isclose(
        _v(linalg.matrix_norm(_t(A), ord="nuc")),
        _v(linalg.svd(_t(A))[1]).sum(),
        rtol=1e-7,
    )


def test_the_spectral_norm_is_the_largest_singular_value():
    assert np.isclose(
        _v(linalg.matrix_norm(_t(A), ord=2)), _v(linalg.svd(_t(A))[1]).max(), rtol=1e-7
    )


def test_keepdim_keeps_the_reduced_axes():
    assert _v(linalg.matrix_norm(_t(A), keepdim=True)).shape == (1, 1)
    assert _v(linalg.matrix_norm(_t(A))).shape == ()


# ── the small ones ────────────────────────────────────────────────────────────


def test_cross_is_the_right_handed_product():
    assert np.allclose(_v(linalg.cross(_t([1.0, 0.0, 0.0]), _t([0.0, 1.0, 0.0]))), [0.0, 0.0, 1.0])
    left = _v(linalg.cross(_t([1.0, 2.0, 3.0]), _t([4.0, 5.0, 6.0])))
    assert np.allclose(left, np.cross([1, 2, 3], [4, 5, 6]))


def test_a_vector_crossed_with_itself_is_zero():
    assert np.allclose(_v(linalg.cross(_t(VEC[:3]), _t(VEC[:3]))), 0.0, atol=1e-12)


def test_outer_is_the_rank_one_product():
    assert np.allclose(_v(linalg.outer(_t(VEC), _t(VEC))), np.outer(VEC, VEC))
    assert int(np.linalg.matrix_rank(_v(linalg.outer(_t(VEC), _t(VEC))))) == 1


def test_diagonal_reads_the_diagonal():
    assert np.allclose(_v(linalg.diagonal(_t(A))), np.diagonal(A))


def test_matrix_exp_of_zero_is_the_identity():
    assert np.allclose(_v(linalg.matrix_exp(_t(np.zeros((3, 3))))), np.eye(3), atol=1e-12)


def test_matrix_exp_of_a_diagonal_is_the_elementwise_exponential():
    diag = np.diag([0.5, -1.0, 2.0])
    assert np.allclose(
        _v(linalg.matrix_exp(_t(diag))), np.diag(np.exp([0.5, -1.0, 2.0])), atol=1e-9
    )


def test_matrix_exp_composes_along_a_scalar_multiple():
    """``exp(A) @ exp(A) == exp(2A)`` holds because ``A`` commutes with
    itself — a cheap check that the series is actually being summed."""
    small = A * 0.1
    once = _v(linalg.matrix_exp(_t(small)))
    twice = _v(linalg.matrix_exp(_t(2 * small)))
    assert np.allclose(once @ once, twice, atol=1e-8)
