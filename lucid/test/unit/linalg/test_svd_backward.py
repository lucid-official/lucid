"""Regression test: the SVD backward was wrong for U and Vh.

Found 2026-08-05 by the audit's ``grad`` axis, which had been reporting
``lucid.linalg.svd`` at rel 1.36 for as long as the axis has existed.  It
was easy to dismiss: a left singular vector is defined only up to a sign,
so comparing ``U`` against another framework's shows columns differing by
a sign, and it is tempting to stop there.

What settles it is Lucid's *own* forward.  A central difference of it
agreed with the reference's analytic gradient to 6e-17 and disagreed with
Lucid's by 6e-1, on a matrix whose singular values are well separated —
so neither the sign convention nor a degeneracy was the explanation.

Three things were wrong in the Loewner term and they compounded rather
than cancelling: ``s_i`` where the derivation puts ``s_j``, the sign that
follows from writing ``s_i² - s_j²`` for its negative, and the
symmetrisation ``J + Jᵀ`` missing entirely so that only half the term
survived.

The check here is against Lucid's own forward rather than against the
reference, because that is the comparison the sign convention cannot
confuse.
"""

import numpy as np
import pytest

import lucid
import lucid.linalg

_SHAPES = [(3, 3), (4, 3), (3, 4), (5, 5), (6, 2), (2, 6)]
_PATHS = ["U", "S", "Vh"]


def _loss(matrix: np.ndarray, weight: np.ndarray, path: str) -> float:
    u, s, vh = lucid.linalg.svd(lucid.tensor(matrix), full_matrices=False)
    picked = {"U": u, "S": s, "Vh": vh}[path]
    return float((picked * lucid.tensor(weight)).sum().item())


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("path", _PATHS)
def test_svd_backward_matches_its_own_forward(shape, path: str) -> None:
    rows, cols = shape
    rank = min(rows, cols)
    rng = np.random.default_rng(11)
    # +2I keeps the singular values well separated, so a disagreement
    # cannot be blamed on a degenerate pair.
    matrix = rng.random(shape) + np.eye(rows, cols) * 2
    weight = rng.random({"U": (rows, rank), "S": (rank,), "Vh": (rank, cols)}[path])

    x = lucid.tensor(matrix.copy(), requires_grad=True)
    u, s, vh = lucid.linalg.svd(x, full_matrices=False)
    ({"U": u, "S": s, "Vh": vh}[path] * lucid.tensor(weight)).sum().backward()
    analytic = np.asarray(x.grad.numpy(), dtype=np.float64)

    step = 1e-6
    difference = np.zeros_like(matrix)
    for i in range(matrix.size):
        up, down = matrix.copy().ravel(), matrix.copy().ravel()
        up[i] += step
        down[i] -= step
        difference.ravel()[i] = (
            _loss(up.reshape(shape), weight, path) - _loss(down.reshape(shape), weight, path)
        ) / (2 * step)

    worst = float(np.max(np.abs(analytic - difference)))
    assert worst < 1e-5, f"{shape} {path}: analytic and difference differ by {worst:.3e}"


def test_svd_reconstructs_what_it_decomposed() -> None:
    """Guard the instrument: the forward has to still be an SVD."""
    rng = np.random.default_rng(3)
    matrix = rng.random((4, 3))
    u, s, vh = lucid.linalg.svd(lucid.tensor(matrix), full_matrices=False)
    rebuilt = np.asarray(u.numpy()) @ np.diag(np.asarray(s.numpy())) @ np.asarray(vh.numpy())
    assert np.allclose(rebuilt, matrix, atol=1e-10)


# ── norm, scaled ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "values,expected",
    [
        ([1e200, 1e200, 3.0], 1.4142135623730951e200),
        ([1e-200, 1e-200], 1.4142135623730951e-200),
        ([3.0, 4.0], 5.0),
        ([0.0, 0.0], 0.0),
    ],
)
def test_norm_does_not_overflow_on_the_way_to_a_finite_answer(values, expected) -> None:
    """``sqrt(Σ x²)`` reaches infinity at the first square and never returns.

    ``norm([1e200, 1e200])`` is 1.41e200 — an ordinary double — and the
    unscaled evaluation answered inf.  The other end underflowed to 0.
    The reference does both; this is a place Lucid can simply be right.
    """
    got = float(np.asarray(lucid.linalg.norm(lucid.tensor(np.array(values))).numpy()).ravel()[0])
    assert np.isfinite(got), got
    if expected == 0.0:
        assert got == 0.0
    else:
        assert abs(got - expected) / expected < 1e-12, (got, expected)


def test_norm_is_unchanged_for_ordinary_inputs() -> None:
    """Guard the instrument: rescaling must not move the everyday answer."""
    values = np.arange(1, 7, dtype=np.float64).reshape(2, 3)
    assert np.isclose(
        float(np.asarray(lucid.linalg.norm(lucid.tensor(values)).numpy()).ravel()[0]),
        float(np.sqrt((values**2).sum())),
    )
    per_row = np.asarray(lucid.linalg.norm(lucid.tensor(values), dim=1).numpy())
    assert np.allclose(per_row, np.sqrt((values**2).sum(axis=1)))
