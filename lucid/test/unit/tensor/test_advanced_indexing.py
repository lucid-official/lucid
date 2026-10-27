"""Advanced indexing, checked expression by expression.

``_tensor/_indexing.py`` sat at 62%, and the dark part was the hard
part: the branch that runs when the array indices are *not* adjacent.
Indexing is where a wrong answer is quietest — the result has the right
element count and the right dtype, and a transposed axis pair reads as
plausible data all the way into a loss curve.

NumPy is the oracle for everything here except one expression, noted
where it appears, on which the reference framework and NumPy genuinely
disagree; Lucid follows the reference.  The split cases checked against
the reference itself live in
``lucid/test/parity/tensor/test_advanced_indexing_parity.py``, because
they need it installed.
"""

import numpy as np
import pytest

import lucid

A = np.arange(2 * 3 * 4 * 5, dtype=np.float64).reshape(2, 3, 4, 5)
B = np.arange(2 * 3 * 4 * 5 * 6, dtype=np.float64).reshape(2, 3, 4, 5, 6)
M = np.arange(4 * 5, dtype=np.float64).reshape(4, 5)
V = np.arange(6, dtype=np.float64)

I1 = np.array([0, 2, 1])
I2 = np.array([1, 0, 3])


def _t(a):
    return lucid.tensor(np.asarray(a).copy())


def _i(a):
    return lucid.tensor(np.asarray(a, dtype=np.int32), dtype=lucid.int32)


def _v(x):
    return np.asarray(x.numpy())


def _same(got, want):
    assert got.shape == want.shape, f"shape {got.shape} != {want.shape}"
    assert np.array_equal(got, want)


# ── one integer array ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "take,expect",
    [
        (lambda t: t[_i(I1)], lambda n: n[I1]),
        (lambda t: t[:, _i(I2)], lambda n: n[:, I2]),
        (lambda t: t[_i(I1), :], lambda n: n[I1, :]),
        (lambda t: t[1, _i(I2)], lambda n: n[1, I2]),
        (lambda t: t[_i(I1), 2], lambda n: n[I1, 2]),
        (lambda t: t[::2, _i(I2)], lambda n: n[::2, I2]),
        (lambda t: t[_i(I1), ::2], lambda n: n[I1, ::2]),
        (lambda t: t[None, _i(I1)], lambda n: n[None, I1]),
        (lambda t: t[_i(I1), None], lambda n: n[I1, None]),
        (lambda t: t[_i(I1[:, None])], lambda n: n[I1[:, None]]),
    ],
)
def test_a_single_index_array_on_a_matrix(take, expect):
    _same(_v(take(_t(M))), expect(M))


@pytest.mark.parametrize(
    "take,expect",
    [
        (lambda t: t[:, _i(I1)], lambda n: n[:, I1]),
        (lambda t: t[:, :, _i(I1)], lambda n: n[:, :, I1]),
        (lambda t: t[..., _i(I1)], lambda n: n[..., I1]),
        (lambda t: t[0, _i(I1)], lambda n: n[0, I1]),
        (lambda t: t[:, 1, _i(I1)], lambda n: n[:, 1, I1]),
        (lambda t: t[1:, _i(I1), 2], lambda n: n[1:, I1, 2]),
        (lambda t: t[:, _i(I1), 1:3], lambda n: n[:, I1, 1:3]),
    ],
)
def test_a_single_index_array_on_a_four_dimensional(take, expect):
    _same(_v(take(_t(A))), expect(A))


# ── two index arrays, adjacent ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "take,expect",
    [
        (lambda t: t[_i(I1), _i(I2)], lambda n: n[I1, I2]),
        (lambda t: t[_i(I1[:, None]), _i(I2)], lambda n: n[I1[:, None], I2]),
    ],
)
def test_two_adjacent_index_arrays_broadcast_together(take, expect):
    _same(_v(take(_t(M))), expect(M))


def test_adjacent_arrays_keep_the_result_where_the_block_was():
    """``a[:, i, j]`` puts the broadcast result at axis 1, not at the
    front — that is what separates the adjacent case from the split one."""
    _same(_v(_t(A)[:, _i(I1), _i([1, 2, 3])]), A[:, I1, [1, 2, 3]])


# ── two index arrays, split by a slice ────────────────────────────────────────


@pytest.mark.parametrize(
    "take,expect,array",
    [
        (lambda t: t[:, _i(I1), :, _i([1, 2, 3])], lambda n: n[:, I1, :, [1, 2, 3]], A),
        (lambda t: t[_i([0, 1]), :, _i([2, 3])], lambda n: n[[0, 1], :, [2, 3]], A),
        (lambda t: t[_i([0, 1]), :, :, _i([0, 1])], lambda n: n[[0, 1], :, :, [0, 1]], A),
        (lambda t: t[:, _i(I1), 1:3, _i([1, 2, 3])], lambda n: n[:, I1, 1:3, [1, 2, 3]], A),
        (lambda t: t[1:, _i(I1), :, _i([1, 2, 3])], lambda n: n[1:, I1, :, [1, 2, 3]], A),
        (lambda t: t[:, _i(I1), ::2, _i([1, 2, 3])], lambda n: n[:, I1, ::2, [1, 2, 3]], A),
        (lambda t: t[:, _i(I1), :, _i([1, 2, 3]), :], lambda n: n[:, I1, :, [1, 2, 3], :], B),
        (
            lambda t: t[_i([0, 1]), :, _i([1, 2]), :, _i([0, 1])],
            lambda n: n[[0, 1], :, [1, 2], :, [0, 1]],
            B,
        ),
    ],
)
def test_split_index_arrays_move_the_result_to_the_front(take, expect, array):
    """The branch this file exists for.

    When the array indices are separated, the broadcast result goes to
    axis 0 and *every* surviving axis follows in its original order —
    the ones a slice touched and the ones no index mentioned alike.

    Splitting those two groups is the bug this pins: ``A[:, i, :, j]``
    on a ``(2, 3, 4, 5)`` was returning ``(3, 4, 2)`` where the answer is
    ``(3, 2, 4)``.  Same elements, two axes swapped, and for
    ``A[:, i, 1:3, j]`` the wrong answer even had the right *shape*.
    """
    _same(_v(take(_t(array))), expect(array))


@pytest.mark.parametrize(
    "take,expect",
    [
        (lambda t: t[:, _i(I1), None, :, _i([1, 2, 3])], lambda n: n[:, I1, None, :, [1, 2, 3]]),
        (lambda t: t[None, :, _i(I1), :, _i([1, 2, 3])], lambda n: n[None, :, I1, :, [1, 2, 3]]),
    ],
)
def test_a_new_axis_inside_a_split_block(take, expect):
    """``None`` used to raise ``permute: perm length must equal tensor
    ndim`` here: it occupies a slot in the index expression but not a
    dimension of the tensor, and the permutation counted it as both."""
    _same(_v(take(_t(A))), expect(A))


# ── boolean masks ─────────────────────────────────────────────────────────────


def test_a_full_mask_flattens_to_the_selected_elements():
    mask = M > 10
    _same(_v(_t(M)[_t(mask)]), M[mask])


def test_a_row_mask_selects_rows():
    mask = np.array([True, False, True, False])
    _same(_v(_t(M)[_t(mask)]), M[mask])
    _same(_v(_t(M)[_t(mask), :]), M[mask, :])
    _same(_v(_t(M)[_t(mask), 1]), M[mask, 1])


def test_a_mask_over_a_leading_pair_of_dims():
    mask = A[:, :, 0, 0] > 20
    _same(_v(_t(A)[_t(mask)]), A[mask])
    _same(_v(_t(A)[_t(mask), 0]), A[mask, 0])


def test_a_mask_in_a_trailing_position():
    mask = M[0] > 2
    _same(_v(_t(M)[:, _t(mask)]), M[:, mask])


def test_comparison_masking_a_vector():
    t = _t(V)
    _same(_v(t[t > 2]), V[V > 2])


def test_an_all_false_mask_gives_an_empty_result():
    mask = np.zeros(4, dtype=bool)
    got = _v(_t(M)[_t(mask)])
    assert got.shape == M[mask].shape == (0, 5)


# ── where the reference and NumPy part ways ───────────────────────────────────


def test_an_integer_split_from_an_array_follows_the_reference():
    """``A[0, :, i, 1]`` is the one expression here where NumPy and the
    reference framework disagree.

    NumPy counts a scalar integer as an *advanced* index, so the ints at
    positions 0 and 3 split the block and the result is transposed to the
    front.  The reference counts it as basic, leaving the array result in
    place.  Lucid is a reference-compatible interface, so it follows the
    reference — the two answers are exact transposes of each other, which
    is worth knowing before someone ports code across.
    """
    got = _v(_t(A)[0, :, _i(I1), 1])
    from_numpy = A[0, :, I1, 1]
    assert got.shape == from_numpy.shape == (3, 3)
    assert np.array_equal(got, from_numpy.T)
    assert not np.array_equal(got, from_numpy)


# ── assignment ────────────────────────────────────────────────────────────────


SET_CASES = [
    # basic keys
    ("whole", lambda t: t.__setitem__(slice(None), 9.0), lambda n: n.__setitem__(slice(None), 9.0)),
    ("ellipsis", lambda t: t.__setitem__(Ellipsis, 9.0), lambda n: n.__setitem__(Ellipsis, 9.0)),
    ("row", lambda t: t.__setitem__(1, 9.0), lambda n: n.__setitem__(1, 9.0)),
    ("row from the end", lambda t: t.__setitem__(-1, 9.0), lambda n: n.__setitem__(-1, 9.0)),
    ("cell", lambda t: t.__setitem__((1, 2), 9.0), lambda n: n.__setitem__((1, 2), 9.0)),
    (
        "row slice",
        lambda t: t.__setitem__(slice(1, 3), 9.0),
        lambda n: n.__setitem__(slice(1, 3), 9.0),
    ),
    (
        "strided",
        lambda t: t.__setitem__(slice(None, None, 2), 9.0),
        lambda n: n.__setitem__(slice(None, None, 2), 9.0),
    ),
    (
        "column",
        lambda t: t.__setitem__((slice(None), 1), 9.0),
        lambda n: n.__setitem__((slice(None), 1), 9.0),
    ),
    (
        "sub-block",
        lambda t: t.__setitem__((slice(1, 3), slice(None, None, 2)), 9.0),
        lambda n: n.__setitem__((slice(1, 3), slice(None, None, 2)), 9.0),
    ),
    # one index array
    ("rows by array", lambda t: t.__setitem__(_i(I1), 9.0), lambda n: n.__setitem__(I1, 9.0)),
    (
        "columns by array",
        lambda t: t.__setitem__((slice(None), _i(I2)), 9.0),
        lambda n: n.__setitem__((slice(None), I2), 9.0),
    ),
    (
        "array then int",
        lambda t: t.__setitem__((_i(I1), 2), 9.0),
        lambda n: n.__setitem__((I1, 2), 9.0),
    ),
    (
        "int then array",
        lambda t: t.__setitem__((0, _i(I2)), 9.0),
        lambda n: n.__setitem__((0, I2), 9.0),
    ),
    # two index arrays — the pairs, not the rectangle
    (
        "zipped pairs",
        lambda t: t.__setitem__((_i(I1), _i(I2)), 9.0),
        lambda n: n.__setitem__((I1, I2), 9.0),
    ),
    # masks
    ("full mask", lambda t: t.__setitem__(_t(M > 10), 9.0), lambda n: n.__setitem__(M > 10, 9.0)),
    (
        "row mask",
        lambda t: t.__setitem__(_t([True, False, True, False]), 9.0),
        lambda n: n.__setitem__(np.array([True, False, True, False]), 9.0),
    ),
    (
        "column mask",
        lambda t: t.__setitem__((slice(None), _t(M[0] > 2)), 9.0),
        lambda n: n.__setitem__((slice(None), M[0] > 2), 9.0),
    ),
    (
        "empty mask",
        lambda t: t.__setitem__(_t(np.zeros(4, bool)), 9.0),
        lambda n: n.__setitem__(np.zeros(4, bool), 9.0),
    ),
    # tensor values
    (
        "rows from a block",
        lambda t: t.__setitem__(_i(I1), _t(np.full((3, 5), -2.0))),
        lambda n: n.__setitem__(I1, -2.0),
    ),
    (
        "a row from a vector",
        lambda t: t.__setitem__(1, _t(np.arange(5.0) * -1)),
        lambda n: n.__setitem__(1, np.arange(5.0) * -1),
    ),
    (
        "a column from a vector",
        lambda t: t.__setitem__((slice(None), 1), _t(np.arange(4.0) * -1)),
        lambda n: n.__setitem__((slice(None), 1), np.arange(4.0) * -1),
    ),
    (
        "pairs from a vector",
        lambda t: t.__setitem__((_i(I1), _i(I2)), _t([-1.0, -2.0, -3.0])),
        lambda n: n.__setitem__((I1, I2), np.array([-1.0, -2.0, -3.0])),
    ),
    (
        "a broadcast row",
        lambda t: t.__setitem__(slice(1, 3), _t(np.arange(5.0) * -1)),
        lambda n: n.__setitem__(slice(1, 3), np.arange(5.0) * -1),
    ),
]


@pytest.mark.parametrize("assign,expect", [c[1:] for c in SET_CASES], ids=[c[0] for c in SET_CASES])
def test_assignment_writes_where_reading_would_have_read(assign, expect):
    """Assignment used to re-derive the target set instead of reusing the
    reading path, and got it wrong in two ways that wrote a *superset*:

    * ``t[rows, cols] = v`` crossed the two index arrays instead of
      zipping them, filling the whole ``rows x cols`` rectangle;
    * ``t[mask] = v`` on a full-shape mask took the mask's rows and
      columns separately, so any row the mask touched anywhere was
      written across.

    Neither raised, and both leave a tensor of the right shape and dtype.
    """
    tensor, array = _t(M), M.copy()
    assign(tensor)
    expect(array)
    _same(_v(tensor), array)


def test_a_reversed_slice_assigns_in_reverse():
    tensor, array = _t(V), V.copy()
    tensor[::-1] = _t(np.arange(6.0) * -1)
    array[::-1] = np.arange(6.0) * -1
    _same(_v(tensor), array)


@pytest.mark.parametrize(
    "assign,expect",
    [
        (lambda t: t.__setitem__(0, 9.0), lambda n: n.__setitem__(0, 9.0)),
        (
            lambda t: t.__setitem__((slice(None), _i(I1)), 9.0),
            lambda n: n.__setitem__((slice(None), I1), 9.0),
        ),
        (lambda t: t.__setitem__((0, 1, 2, 3), 9.0), lambda n: n.__setitem__((0, 1, 2, 3), 9.0)),
        (lambda t: t.__setitem__((Ellipsis, 0), 9.0), lambda n: n.__setitem__((Ellipsis, 0), 9.0)),
        (
            lambda t: t.__setitem__((slice(None), _i(I1), slice(None), _i([1, 2, 3])), 9.0),
            lambda n: n.__setitem__((slice(None), I1, slice(None), [1, 2, 3]), 9.0),
        ),
    ],
)
def test_assignment_on_a_four_dimensional(assign, expect):
    tensor, array = _t(A), A.copy()
    assign(tensor)
    expect(array)
    _same(_v(tensor), array)


def test_assignment_does_not_alias_the_value():
    """``x[:] = y`` has to copy: sharing ``y``'s storage would make a
    later write to ``y`` rewrite ``x`` behind its back."""
    source = _t(np.ones((3, 4)))
    target = _t(np.zeros((3, 4)))
    target[:] = source
    source[0] = 5.0
    assert np.array_equal(_v(target), np.ones((3, 4)))


# ── refusals ──────────────────────────────────────────────────────────────────


def test_an_out_of_range_index_is_refused():
    with pytest.raises((IndexError, RuntimeError, ValueError)):
        _t(M)[_i([0, 99])]


def test_an_unsupported_index_type_is_refused():
    with pytest.raises((IndexError, TypeError)):
        _t(M)[_i(I1), "column"]
