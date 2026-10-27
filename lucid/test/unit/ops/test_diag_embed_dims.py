"""``diag_embed`` puts the matrix at ``dim1`` / ``dim2``, as the reference does.

``dim1`` and ``dim2`` were read and ignored: the matrix always went to the
last two dims, so ``diag_embed(t, 0, 0, 2)`` of ``[[1, 2], [3, 4]]`` gave
``[[1, 0], [3, 0]]`` for its first row-slice where the reference gives
``[[1, 0], [0, 2]]`` — a silently wrong result (LCD-410 review).
"""

import itertools
from types import ModuleType

import numpy as np
import pytest

import lucid


def _dim_pairs(nd: int) -> list[tuple[int, int]]:
    return [(d1, d2) for d1, d2 in itertools.permutations(range(-nd, nd), 2) if (d1 - d2) % nd]


@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 2, 3)], ids=str)
@pytest.mark.parametrize("offset", [0, 1, -2])
def test_the_matrix_lands_where_dim1_and_dim2_say(
    ref: ModuleType, device: str, shape: tuple[int, ...], offset: int
) -> None:
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1
    x = lucid.tensor(values, device=device)
    for dim1, dim2 in _dim_pairs(len(shape) + 1):
        want = ref.diag_embed(ref.tensor(values), offset, dim1, dim2).numpy()
        got = lucid.diag_embed(x, offset, dim1, dim2).numpy()
        assert got.shape == want.shape, (dim1, dim2)
        np.testing.assert_array_equal(got, want, err_msg=f"{dim1}, {dim2}")


def test_the_reported_case() -> None:
    got = lucid.diag_embed(lucid.tensor([[1, 2], [3, 4]]), 0, 0, 2)
    assert got[:, 0, :].tolist() == [[1, 0], [0, 2]]


def test_dims_out_of_range_or_equal_are_refused() -> None:
    x = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="same dim"):
        lucid.diag_embed(x, 0, 1, 1)
    with pytest.raises(ValueError, match="same dim"):
        lucid.diag_embed(x, 0, -1, 2)
    with pytest.raises(IndexError):
        lucid.diag_embed(x, 0, 0, 5)
