"""``eig`` returns complex, because a real matrix's eigenvalues are complex.

It used to return the real parts alone.  LAPACK's ``?geev`` reports the
spectrum in two real arrays — ``wr`` and ``wi`` — and this backend read
only the first, so a conjugate pair ``a ± bi`` came back as ``a, a``: two
equal reals where there were two distinct numbers, with no error and
nothing to say the imaginary halves had been dropped.  The eigenvectors
were worse off still, copied out in LAPACK's packed real form where a
conjugate pair occupies two adjacent columns, so neither column was an
eigenvector of anything.

A real matrix having complex eigenvalues is the ordinary case, not a
corner one.  Every rotation has them.  It went unnoticed because a
symmetric or triangular test matrix has a real spectrum, and those are
the matrices test suites reach for.

The matrices here are chosen so the answer is known without computing
it: a quarter-turn rotation has eigenvalues ±i, and the companion matrix
of ``x³ + 1`` has the three cube roots of −1.
"""

import cmath
import math

import numpy as np
import pytest

import lucid
import lucid.linalg as L

ROTATION = np.array([[0.0, -1.0], [1.0, 0.0]])
COMPANION = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
SYMMETRIC = np.array([[2.0, 1.0], [1.0, 2.0]])


def _sorted(t: lucid.Tensor) -> np.ndarray:
    return np.sort_complex(np.asarray(t.numpy()))


# ── the eigenvalues themselves ────────────────────────────────────────────────


def test_a_quarter_turn_has_eigenvalues_plus_and_minus_i() -> None:
    """The rotation by 90 degrees fixes no real direction, so neither
    eigenvalue is real.  This answered ``[0, 0]``."""
    got = _sorted(L.eigvals(lucid.tensor(ROTATION)))
    assert np.allclose(got, [-1j, 1j])


def test_the_cube_roots_of_minus_one() -> None:
    """``COMPANION``'s last column is ``(-1, 0, 0)``, which makes it the
    companion matrix of ``x³ + 1``; its spectrum is the three cube roots
    of −1.  Two of them are complex, and both used to come back as
    ``0.5`` — the shared real part, twice."""
    got = _sorted(L.eigvals(lucid.tensor(COMPANION)))
    expected = np.sort_complex(
        np.array([cmath.exp(1j * math.pi * (2 * k + 1) / 3) for k in range(3)])
    )
    assert np.allclose(got, expected)


def test_a_real_spectrum_still_comes_back_real_valued() -> None:
    """Guard the guard: the complex dtype must not perturb the easy case."""
    got = _sorted(L.eigvals(lucid.tensor(SYMMETRIC)))
    assert np.allclose(got, [1.0, 3.0])
    assert np.allclose(np.asarray(lucid.imag(L.eigvals(lucid.tensor(SYMMETRIC))).numpy()), 0.0)


# ── the dtype is not a choice ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "real_dtype,complex_dtype",
    [(lucid.float32, lucid.complex64), (lucid.float64, lucid.complex128)],
)
def test_the_lane_width_follows_the_input(real_dtype, complex_dtype) -> None:
    """An f64 matrix whose spectrum came back complex64 would shed eight
    decimal digits — the same class of quiet loss as dropping the
    imaginary part outright, which is why complex128 exists."""
    a = lucid.tensor(ROTATION, dtype=real_dtype)
    w, v = L.eig(a)
    assert w.dtype is complex_dtype
    assert v.dtype is complex_dtype
    assert L.eigvals(a).dtype is complex_dtype


def test_complex128_keeps_the_digits_complex64_would_lose() -> None:
    a = np.array([[0.0, -1.0], [1.000000001, 0.0]])
    w = _sorted(L.eigvals(lucid.tensor(a)))
    expected = np.sort_complex(np.linalg.eigvals(a))
    assert np.allclose(w, expected, rtol=0, atol=1e-15)


# ── the eigenvectors ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["rotation", "companion", "symmetric", "random"])
def test_each_column_of_v_is_an_eigenvector(name) -> None:
    """LAPACK packs a conjugate pair as two adjacent *real* columns.
    Copied out verbatim, neither column solved ``Av = λv``."""
    a = {
        "rotation": ROTATION,
        "companion": COMPANION,
        "symmetric": SYMMETRIC,
        "random": np.random.default_rng(0).standard_normal((5, 5)),
    }[name]
    w, v = L.eig(lucid.tensor(a))
    W, V = np.asarray(w.numpy()), np.asarray(v.numpy())
    for j in range(a.shape[0]):
        residual = np.abs(a.astype(complex) @ V[:, j] - W[j] * V[:, j]).max()
        assert residual < 1e-12, (name, j, residual)


@pytest.mark.parametrize("name", ["rotation", "companion", "random"])
def test_the_decomposition_reconstructs_the_matrix(name) -> None:
    """A = V Λ V⁻¹ — the property the whole factorisation exists for."""
    a = {
        "rotation": ROTATION,
        "companion": COMPANION,
        "random": np.random.default_rng(1).standard_normal((4, 4)),
    }[name]
    w, v = L.eig(lucid.tensor(a))
    W, V = np.asarray(w.numpy()), np.asarray(v.numpy())
    assert np.allclose(V @ np.diag(W) @ np.linalg.inv(V), a, atol=1e-12)


def test_conjugate_pairs_come_back_as_conjugates() -> None:
    """Both halves of the pair, not the same real number twice."""
    w = np.asarray(L.eigvals(lucid.tensor(ROTATION)).numpy())
    assert w[0] == np.conj(w[1])
    assert w[0] != w[1]
    assert np.abs(w.imag).min() > 0.5


# ── batches ───────────────────────────────────────────────────────────────────


def test_a_batch_mixing_real_and_complex_spectra() -> None:
    """The unpacking is per matrix; one real spectrum in the batch must
    not straighten out the other."""
    batch = np.stack([ROTATION, SYMMETRIC])
    w, v = L.eig(lucid.tensor(batch))
    assert tuple(w.shape) == (2, 2)
    assert tuple(v.shape) == (2, 2, 2)
    W = np.asarray(w.numpy())
    assert np.allclose(np.sort_complex(W[0]), [-1j, 1j])
    assert np.allclose(np.sort_complex(W[1]), [1.0, 3.0])
    for b in range(2):
        V = np.asarray(v.numpy())[b]
        for j in range(2):
            assert np.abs(batch[b] @ V[:, j] - W[b][j] * V[:, j]).max() < 1e-12


# ── eigh is unaffected ────────────────────────────────────────────────────────


def test_hermitian_eigenvalues_stay_real() -> None:
    """``eigh`` promises a real spectrum and must not have acquired a
    complex dtype from its neighbour."""
    assert L.eigvalsh(lucid.tensor(SYMMETRIC)).dtype is lucid.float64
    assert np.allclose(np.asarray(L.eigvalsh(lucid.tensor(SYMMETRIC)).numpy()), [1.0, 3.0])
    w, v = L.eigh(lucid.tensor(SYMMETRIC))
    assert w.dtype is lucid.float64 and v.dtype is lucid.float64
