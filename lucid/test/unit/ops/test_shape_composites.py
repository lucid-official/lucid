"""The shape composites, checked against NumPy's definitions.

``_ops/composite/shape.py`` sat at 35%, and the dark entries were whole
functions rather than branches: ``swapdims``, ``adjoint``, ``t``, the
stack family, the ``atleast_*`` family, every splitter, ``take_along_dim``,
the triangle-index helpers, ``combinations`` and ``rot90``.  They are
exported and none of them ran.

Each is checked against NumPy, which defines the same operations and was
not consulted while they were written — so agreement means the semantics
are right rather than merely stable.
"""

import numpy as np
import pytest

import lucid

V = np.arange(1.0, 5.0)
M = np.arange(1.0, 13.0).reshape(3, 4)
C = np.arange(1.0, 25.0).reshape(2, 3, 4)


def _t(a):
    return lucid.tensor(a.copy())


def _v(x):
    return np.asarray(x.numpy())


# ── axis reordering ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("a,b", [(0, 1), (0, 2), (1, 2), (-1, -3)])
def test_swapdims_matches_numpy(a, b):
    assert np.array_equal(_v(lucid.swapdims(_t(C), a, b)), np.swapaxes(C, a, b))


def test_t_transposes_a_matrix_and_leaves_a_vector():
    assert np.array_equal(_v(lucid.t(_t(M))), M.T)
    assert np.array_equal(_v(lucid.t(_t(V))), V)


def test_row_stack_is_vstack():
    """NumPy 2 removed ``row_stack``; Lucid keeps it as the alias it was."""
    got = _v(lucid.row_stack([_t(M), _t(M)]))
    assert np.array_equal(got, np.vstack([M.copy(), M.copy()]))


def test_adjoint_of_a_complex_matrix_is_its_conjugate_transpose():
    """``adjoint`` is defined as the conjugate transpose.

    It used to be unreachable on complex input — ``permute`` had no complex
    branch — and was pinned as a gap.  A CPU transpose is now a view, which
    reads any dtype, so the definition holds.
    """
    z = lucid.tensor(np.array([[1 + 2j, 3 - 1j]], dtype=np.complex64))
    assert np.allclose(_v(lucid.adjoint(z)), np.array([[1 - 2j], [3 + 1j]]))


def test_the_engines_NotImplementedError_is_the_builtin_one_too():
    """``except NotImplementedError`` catches the engine's refusal.

    ``lucid._C`` defines a ``NotImplementedError`` under ``LucidError`` ->
    ``RuntimeError``.  It used to shadow the builtin without inheriting from
    it, so the natural

        try:  ...
        except NotImplementedError:  fallback()

    silently did not fire.  It now derives from the builtin as well, so both
    that and ``except RuntimeError`` / ``LucidError`` catch it — and it names
    its own module rather than ``importlib._bootstrap``.
    """
    z = lucid.tensor(np.array([[1 + 2j]], dtype=np.complex64))
    with pytest.raises(NotImplementedError) as info:
        lucid.sigmoid(z)
    exc = info.value
    assert isinstance(exc, RuntimeError)
    assert type(exc).__module__ == "lucid._C.engine"


def test_adjoint_of_a_real_matrix_is_the_transpose():
    assert np.array_equal(_v(lucid.adjoint(_t(M))), M.T)


@pytest.mark.parametrize("k", [0, 1, 2, 3, -1])
@pytest.mark.parametrize("axes", [(0, 1), (1, 0)])
def test_rot90_matches_numpy(k, axes):
    assert np.array_equal(_v(lucid.rot90(_t(M), k, axes)), np.rot90(M, k, axes))


def test_four_quarter_turns_return_the_original():
    assert np.array_equal(_v(lucid.rot90(_t(M), 4)), M)


# ── the stack family ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name,arrays",
    [
        ("column_stack", [V, V]),
        ("column_stack", [M, M]),
        ("dstack", [V, V]),
        ("dstack", [M, M]),
        ("hstack", [V, V]),
        ("vstack", [M, M]),
    ],
)
def test_the_stack_family_matches_numpy(name, arrays):
    got = _v(getattr(lucid, name)([_t(a) for a in arrays]))
    expected = getattr(np, name)([a.copy() for a in arrays])
    assert got.shape == expected.shape
    assert np.array_equal(got, expected)


# ── atleast_* ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("rank", [1, 2, 3])
@pytest.mark.parametrize("source", ["scalar", "vector", "matrix", "cube"])
def test_atleast_promotes_to_at_least_that_rank(rank, source):
    arr = {"scalar": np.array(3.0), "vector": V, "matrix": M, "cube": C}[source]
    got = _v(getattr(lucid, f"atleast_{rank}d")(_t(arr)))
    expected = getattr(np, f"atleast_{rank}d")(arr.copy())
    assert got.shape == expected.shape
    assert np.array_equal(got, expected)
    assert got.ndim >= rank


def test_atleast_leaves_a_higher_rank_alone():
    assert _v(lucid.atleast_1d(_t(C))).shape == C.shape


# ── splitting ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name,arr,parts",
    [
        ("vsplit", M, 3),
        ("hsplit", M, 2),
        ("dsplit", C, 2),
        ("vsplit", C, 2),
    ],
)
def test_the_split_family_matches_numpy(name, arr, parts):
    got = getattr(lucid, name)(_t(arr), parts)
    expected = getattr(np, name)(arr.copy(), parts)
    assert len(got) == len(expected)
    for g, e in zip(got, expected):
        assert np.array_equal(_v(g), e)


@pytest.mark.parametrize("sections", [2, 3, [1, 3]])
def test_tensor_split_matches_numpy(sections):
    got = lucid.tensor_split(_t(M), sections, dim=1)
    expected = np.array_split(M.copy(), sections, axis=1)
    assert len(got) == len(expected)
    for g, e in zip(got, expected):
        assert np.array_equal(_v(g), e)


def test_tensor_split_handles_an_uneven_division():
    """``np.array_split`` puts the remainder in the leading pieces, which
    is the behaviour that separates it from ``split``."""
    got = [tuple(_v(p).shape) for p in lucid.tensor_split(_t(M), 3, dim=1)]
    expected = [p.shape for p in np.array_split(M.copy(), 3, axis=1)]
    assert got == expected


# ── gathering along an axis ───────────────────────────────────────────────────


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_take_along_dim_matches_numpy(dim):
    indices = np.argsort(M, axis=dim)
    got = _v(lucid.take_along_dim(_t(M), lucid.tensor(indices, dtype=lucid.int32), dim=dim))
    assert np.array_equal(got, np.take_along_axis(M.copy(), indices, axis=dim))


def test_take_along_dim_with_argsort_sorts():
    """The composition that makes it worth having."""
    indices = np.argsort(M, axis=1)
    got = _v(lucid.take_along_dim(_t(M), lucid.tensor(indices, dtype=lucid.int32), dim=1))
    assert np.array_equal(got, np.sort(M, axis=1))


# ── index helpers ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("n,m", [(3, 3), (4, 3), (3, 5)])
@pytest.mark.parametrize("offset", [0, 1, -1])
def test_tril_indices_matches_numpy(n, m, offset):
    got = _v(lucid.tril_indices(n, m, offset=offset))
    expected = np.tril_indices(n, k=offset, m=m)
    assert np.array_equal(got[0], expected[0])
    assert np.array_equal(got[1], expected[1])


@pytest.mark.parametrize("n,m", [(3, 3), (4, 3), (3, 5)])
@pytest.mark.parametrize("offset", [0, 1, -1])
def test_triu_indices_matches_numpy(n, m, offset):
    got = _v(lucid.triu_indices(n, m, offset=offset))
    expected = np.triu_indices(n, k=offset, m=m)
    assert np.array_equal(got[0], expected[0])
    assert np.array_equal(got[1], expected[1])


def test_the_two_index_sets_partition_the_matrix():
    """Off the diagonal, every position is in exactly one of them."""
    lower = _v(lucid.tril_indices(4, 4, offset=-1))
    upper = _v(lucid.triu_indices(4, 4, offset=1))
    positions = set(zip(*lower)) | set(zip(*upper))
    assert len(positions) == 4 * 4 - 4  # everything but the diagonal


# ── combinations ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("r", [1, 2, 3])
@pytest.mark.parametrize("with_replacement", [False, True])
def test_combinations_counts_correctly(r, with_replacement):
    import math

    got = _v(lucid.combinations(_t(V), r=r, with_replacement=with_replacement))
    n = V.size
    expected = math.comb(n + r - 1, r) if with_replacement else math.comb(n, r)
    assert got.shape == (expected, r)


def test_combinations_are_the_values_not_the_indices():
    got = _v(lucid.combinations(_t(np.array([10.0, 20.0, 30.0])), r=2))
    assert sorted(map(tuple, got)) == [(10.0, 20.0), (10.0, 30.0), (20.0, 30.0)]
