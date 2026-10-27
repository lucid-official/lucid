"""Degenerate matrices have decompositions; LAPACK just cannot compute them.

An empty matrix is a legal object, not a malformed one.  A 0x3 matrix is
the unique linear map from R^3 to R^0, its rank is 0, its reduced SVD has
no singular values, and the determinant of the 0x0 matrix is the empty
product — 1, for the same reason an empty sum is 0.  Every op here has a
defined answer and none of them used to give it.

They dispatched, and LAPACK refused: its leading dimensions must be at
least 1 even when the extent they describe is 0, so ``dgesdd`` on a 0x3
matrix received ``LDA = 0`` and reported argument 5 illegal.  That failed
twice over.  The Fortran runtime prints its own complaint straight to
file descriptor 2 —

    ** On entry to DGESDD, parameter number  5 had an illegal value

— which no Python-level redirection can catch, and the negative ``info``
then surfaced as ``LucidError: LAPACK invalid argument index5``, which
reads as though the caller's data were at fault rather than this
library's argument marshalling.

Expected shapes follow Lucid's own conventions: ``svd`` and ``qr`` are
the *reduced* forms, so with ``k = min(m, n) = 0`` every factor is empty.
"""

import numpy as np
import pytest

import lucid
import lucid.linalg as L

EMPTY = [(0, 0), (0, 3), (3, 0)]


def _t(shape: tuple) -> lucid.Tensor:
    return lucid.tensor(np.zeros(shape))


def _shapes(out: object) -> object:
    if isinstance(out, (tuple, list)):
        return [tuple(x.shape) for x in out]
    return tuple(out.shape)  # type: ignore[union-attr]


# ── the two rectangular decompositions ────────────────────────────────────────


@pytest.mark.parametrize(
    "shape,expected_full,expected_reduced",
    [
        ((0, 0), [(0, 0), (0,), (0, 0)], [(0, 0), (0,), (0, 0)]),
        ((0, 3), [(0, 0), (0,), (3, 3)], [(0, 0), (0,), (0, 3)]),
        ((3, 0), [(3, 3), (0,), (0, 0)], [(3, 0), (0,), (0, 0)]),
    ],
)
def test_svd_of_an_empty_matrix(shape, expected_full, expected_reduced) -> None:
    """Full SVD is (m, m), (0,), (n, n); reduced SVD is (m, 0), (0,), (0, n)."""
    assert _shapes(L.svd(_t(shape))) == expected_full
    assert _shapes(L.svd(_t(shape), full_matrices=False)) == expected_reduced


@pytest.mark.parametrize(
    "shape,expected",
    [
        ((0, 0), [(0, 0), (0, 0)]),
        ((0, 3), [(0, 0), (0, 3)]),
        ((3, 0), [(3, 0), (0, 0)]),
    ],
)
def test_qr_of_an_empty_matrix(shape, expected) -> None:
    assert _shapes(L.qr(_t(shape))) == expected


@pytest.mark.parametrize("shape", EMPTY)
def test_svdvals_and_rank(shape) -> None:
    assert tuple(L.svdvals(_t(shape)).shape) == (0,)
    assert float(np.asarray(L.matrix_rank(_t(shape)).numpy()).ravel()[0]) == 0.0


@pytest.mark.parametrize("shape,expected", [((0, 0), (0, 0)), ((0, 3), (3, 0)), ((3, 0), (0, 3))])
def test_pinv_transposes_the_matrix_axes(shape, expected) -> None:
    assert tuple(L.pinv(_t(shape)).shape) == expected


@pytest.mark.parametrize("shape", [(0, 0), (2, 0, 0), (3, 2, 0, 0)])
def test_inv_of_an_empty_matrix_never_reaches_the_factorisation(shape) -> None:
    """A segfault on CI that this machine could not reproduce.

    ``pinv`` routes a square input to ``inv``, and ``(0, 0)`` is square,
    so an empty pseudo-inverse called LAPACK with ``n = 0``.  Whether
    that returns or crashes is not uniform across Accelerate builds — it
    survived here and segfaulted on the runner for days.  The 0x0 matrix
    is its own inverse, so the answer is decided in Python now and does
    not depend on which machine is asking.
    """
    assert tuple(L.inv(_t(shape)).shape) == shape


# ── the square ones ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "fn,expected",
    [
        (L.inv, (0, 0)),
        (L.cholesky, (0, 0)),
        (L.matrix_exp, (0, 0)),
        (L.eigvals, (0,)),
        (L.eigvalsh, (0,)),
    ],
)
def test_square_ops_on_the_zero_by_zero_matrix(fn, expected) -> None:
    assert tuple(fn(_t((0, 0))).shape) == expected


@pytest.mark.parametrize("fn,expected", [(L.eig, [(0,), (0, 0)]), (L.eigh, [(0,), (0, 0)])])
def test_eigendecompositions(fn, expected) -> None:
    assert _shapes(fn(_t((0, 0)))) == expected


def test_lu_factor() -> None:
    lu, piv = L.lu_factor(_t((0, 0)))
    assert tuple(lu.shape) == (0, 0)
    assert tuple(piv.shape) == (0,)


def test_solve_takes_the_shape_of_its_right_hand_side() -> None:
    x = L.solve(_t((0, 0)), _t((0, 2)))
    assert tuple(x.shape) == (0, 2)


# ── the one that is not empty ─────────────────────────────────────────────────


def test_determinant_of_the_empty_matrix_is_one() -> None:
    """The empty product, exactly as the empty sum is zero.

    This is the only degenerate result here with a value rather than a
    shape: the output has no matrix axes left to be empty in.
    """
    assert float(np.asarray(L.det(_t((0, 0))).numpy()).ravel()[0]) == 1.0


def test_slogdet_of_the_empty_matrix() -> None:
    sign, logabsdet = L.slogdet(_t((0, 0)))
    assert float(np.asarray(sign.numpy()).ravel()[0]) == 1.0
    assert float(np.asarray(logabsdet.numpy()).ravel()[0]) == 0.0


# ── batches ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("batch", [(4,), (2, 3)])
def test_batch_dimensions_survive(batch) -> None:
    """The empty axes are the trailing two; the batch is carried through."""
    a = _t((*batch, 0, 0))
    assert tuple(L.svd(a)[1].shape) == (*batch, 0)
    assert tuple(L.qr(a)[0].shape) == (*batch, 0, 0)
    assert tuple(L.inv(a).shape) == (*batch, 0, 0)
    assert np.all(np.asarray(L.det(a).numpy()) == 1.0)
    assert tuple(L.det(a).shape) == batch


# ── the complaint itself ──────────────────────────────────────────────────────


def test_nothing_is_written_to_the_descriptors() -> None:
    """LAPACK's own error printer bypasses every Python-level redirect.

    ``capfd`` would not do here: the write happens from the Fortran
    runtime inside a library call, and what needs asserting is that a
    fresh interpreter running these ops leaves fd 2 untouched.
    """
    import subprocess
    import sys

    code = (
        "import numpy as np, lucid, lucid.linalg as L\n"
        "for shape in [(0,0),(0,3),(3,0)]:\n"
        "    a = lucid.tensor(np.zeros(shape))\n"
        "    L.svd(a); L.qr(a); L.pinv(a); L.svdvals(a); L.matrix_rank(a)\n"
        "a = lucid.tensor(np.zeros((0,0)))\n"
        "for fn in (L.inv, L.det, L.cholesky, L.eig, L.eigh, L.lu_factor):\n"
        "    fn(a)\n"
    )
    done = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", code], capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    assert "On entry to" not in done.stderr, done.stderr
    assert done.stderr.strip() == "", done.stderr


# ── and the ordinary case is untouched ────────────────────────────────────────


@pytest.mark.parametrize("shape", [(3, 3), (4, 3), (3, 4), (1, 1)])
def test_non_degenerate_matrices_still_decompose(shape) -> None:
    """Guard the guard: the early return must not catch a real matrix."""
    a = np.random.default_rng(0).standard_normal(shape)
    u, s, vh = L.svd(lucid.tensor(a))
    m, n = shape
    k = min(shape)
    assert tuple(u.shape) == (m, m)
    assert tuple(s.shape) == (k,)
    assert tuple(vh.shape) == (n, n)
    u_np = np.asarray(u.numpy())
    vh_np = np.asarray(vh.numpy())
    assert np.allclose(u_np.T @ u_np, np.eye(m), atol=1e-6)
    assert np.allclose(vh_np @ vh_np.T, np.eye(n), atol=1e-6)
    assert np.allclose(
        u_np[:, :k] @ np.diag(np.asarray(s.numpy())) @ vh_np[:k, :],
        a,
        atol=1e-8,
    )
