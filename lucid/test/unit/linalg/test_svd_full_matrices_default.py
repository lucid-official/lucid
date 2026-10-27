"""``linalg.svd`` returns full orthogonal bases by default (LCD-431).

``full_matrices`` defaulted to ``False`` and the reference defaults it to
``True``, so ``U, S, Vh = svd(A)`` of a non-square ``A`` gave reduced
factors where code written against the reference expected square ones.
Every caller inside Lucid names ``full_matrices=False`` where it wants the
reduced factors.
"""

from types import ModuleType

import pytest

import lucid

_TALL = [[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]]
_WIDE = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 9.0]]


def _shapes(factors: tuple[lucid.Tensor, ...]) -> list[tuple[int, ...]]:
    return [tuple(f.shape) for f in factors]


def _close(a: lucid.Tensor, b: lucid.Tensor, tol: float = 1e-4) -> bool:
    return bool((a - b).abs().max().item() < tol)


@pytest.mark.parametrize(
    "data, full, reduced",
    [
        (_TALL, [(3, 3), (2,), (2, 2)], [(3, 2), (2,), (2, 2)]),
        (_WIDE, [(2, 2), (2,), (4, 4)], [(2, 2), (2,), (2, 4)]),
    ],
    ids=["tall", "wide"],
)
def test_the_default_is_the_full_factorisation(
    data: list[list[float]], full: list[tuple[int, ...]], reduced: list[tuple[int, ...]]
) -> None:
    a = lucid.tensor(data)
    assert _shapes(lucid.linalg.svd(a)) == full
    assert _shapes(lucid.linalg.svd(a, full_matrices=True)) == full
    assert _shapes(lucid.linalg.svd(a, full_matrices=False)) == reduced
    assert _shapes(lucid.linalg.svd(a, False)) == reduced


@pytest.mark.parametrize("data", [_TALL, _WIDE], ids=["tall", "wide"])
def test_the_full_bases_are_orthonormal_and_reconstruct(data: list[list[float]]) -> None:
    a = lucid.tensor(data)
    u, s, vh = lucid.linalg.svd(a)
    m, n = a.shape
    k = min(m, n)
    assert _close(u.mT @ u, lucid.eye(m))
    assert _close(vh @ vh.mT, lucid.eye(n))
    assert _close((u[:, :k] * s) @ vh[:k], a)


def test_a_batch_has_full_bases_per_matrix() -> None:
    a = lucid.tensor([_TALL, [[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]]])
    assert _shapes(lucid.linalg.svd(a)) == [(2, 3, 3), (2, 2), (2, 2, 2)]


@pytest.mark.parity
@pytest.mark.parametrize("data", [_TALL, _WIDE], ids=["tall", "wide"])
def test_the_reference_gives_the_same_shapes_and_values(
    ref: ModuleType, data: list[list[float]]
) -> None:
    got = lucid.linalg.svd(lucid.tensor(data))
    want = ref.linalg.svd(ref.tensor(data))
    assert _shapes(got) == [tuple(w.shape) for w in want]
    assert _close(got.S, lucid.tensor(want.S.tolist()))
    # Singular vectors agree up to sign; their products do not depend on it.
    for g, w in ((got.U, want.U), (got.Vh.mT, want.Vh.mT)):
        k = min(len(data), len(data[0]))
        g_abs = (g[:, :k].mT @ lucid.tensor(w[:, :k].tolist())).abs()
        assert _close(g_abs, lucid.eye(k))
