"""The grad2 axis compares a one-triangle input on the triangle it reads.

``eigh`` reads one triangle of a symmetric input and states its gradient
symmetrically.  Differenced coordinate by coordinate, the unread triangle
moves nothing and each read off-diagonal entry carries its mirror's share
too, so a right second derivative came back "rel 0.5".  The fold that
makes the two comparable has to accept exactly that shape of agreement and
nothing looser — a wrong gradient must still fail it.
"""

import numpy as np

from lucid.test.audit._axes import _fold_onto_read_triangle


def _one_triangle_difference(symmetric: np.ndarray, lower: bool) -> np.ndarray:
    """What a coordinate-wise difference of a one-triangle op reports."""
    n = symmetric.shape[-1]
    diagonal = np.eye(n, dtype=bool)
    read = np.tril(np.ones((n, n), bool), -1) if lower else np.triu(np.ones((n, n), bool), 1)
    return np.where(read, 2 * symmetric, np.where(diagonal, symmetric, 0.0))


def test_a_symmetric_gradient_folds_onto_either_triangle() -> None:
    rng = np.random.default_rng(0)
    a = rng.standard_normal((4, 4))
    symmetric = (a + a.T) / 2
    for lower in (True, False):
        fd = _one_triangle_difference(symmetric, lower)
        folded = _fold_onto_read_triangle(symmetric.reshape(-1), fd.reshape(-1), (4, 4))
        assert folded is not None
        np.testing.assert_allclose(folded, fd.reshape(-1))


def test_a_wrong_gradient_still_disagrees_after_folding() -> None:
    rng = np.random.default_rng(1)
    a = rng.standard_normal((3, 3))
    symmetric = (a + a.T) / 2
    fd = _one_triangle_difference(symmetric, lower=True)
    folded = _fold_onto_read_triangle(-symmetric.reshape(-1), fd.reshape(-1), (3, 3))
    assert folded is not None
    assert not np.allclose(folded, fd.reshape(-1))


def test_nothing_is_folded_unless_a_triangle_is_unread() -> None:
    rng = np.random.default_rng(2)
    g = rng.standard_normal(9)
    assert _fold_onto_read_triangle(g, rng.standard_normal(9), (3, 3)) is None
    assert _fold_onto_read_triangle(g[:6], np.zeros(6), (2, 3)) is None
    assert _fold_onto_read_triangle(g[:3], np.zeros(3), (3,)) is None
