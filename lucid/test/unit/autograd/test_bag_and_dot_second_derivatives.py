"""embedding_bag and 2-D Tensor.dot differentiate twice.

The last graph-mode refusals among the ops a model reaches: the rows an
embedding bag drew from are data (read on the host from the saved indices,
offsets and — for max — the weight, as the eager backward reads them) and
the gradient is a scatter-add of the bag gradients onto those rows;
``Tensor.dot`` on matrices is two matmuls.

Checked against central differences of the first gradient in float64.
"""

from collections.abc import Callable

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F
from lucid.autograd import grad

INDICES = [3, 1, 4, 1, 5, 9, 2, 6, 0, 5]
OFFSETS = [0, 3, 3, 7]  # the second bag is empty

CASES: list[tuple[str, dict[str, object]]] = [
    ("sum", {"mode": "sum"}),
    ("mean", {"mode": "mean"}),
    ("max", {"mode": "max"}),
    ("mean padding", {"mode": "mean", "padding_idx": 1}),
    ("sum last offset", {"mode": "sum", "include_last_offset": True}),
]


def _second_derivative_matches(
    fn: Callable[[lucid.Tensor], lucid.Tensor], w0: np.ndarray, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    r0 = rng.standard_normal(fn(lucid.tensor(w0)).shape)
    u = rng.standard_normal(w0.shape)

    def loss(w: lucid.Tensor) -> lucid.Tensor:
        y = fn(w)
        return (lucid.tensor(r0) * y * y).sum()

    def projected_gradient(values: np.ndarray) -> float:
        w = lucid.tensor(values, requires_grad=True)
        (g,) = grad(loss(w), [w])
        return float((np.asarray(g.numpy()) * u).sum())

    w = lucid.tensor(w0, requires_grad=True)
    (g,) = grad(loss(w), [w], create_graph=True)
    assert float((np.asarray(g.numpy()) * u).sum()) == pytest.approx(
        projected_gradient(w0), rel=1e-12
    )
    (hvp,) = grad((g * lucid.tensor(u)).sum(), [w])
    e = rng.standard_normal(w0.shape)
    h = 1e-6
    numeric = (projected_gradient(w0 + h * e) - projected_gradient(w0 - h * e)) / (2 * h)
    assert float((np.asarray(hvp.numpy()) * e).sum()) == pytest.approx(numeric, rel=1e-6, abs=1e-6)


@pytest.mark.parametrize("name,kw", CASES, ids=[c[0] for c in CASES])
def test_embedding_bag(name: str, kw: dict[str, object]) -> None:
    idx = lucid.tensor(INDICES, dtype=lucid.int64)
    off = lucid.tensor(OFFSETS, dtype=lucid.int64)
    # Distinct values a unit apart, so a small step changes no max winner.
    w0 = np.random.default_rng(1).permutation(40).reshape(10, 4).astype(np.float64) * 0.1

    def bag(w: lucid.Tensor) -> lucid.Tensor:
        return F.embedding_bag(idx, w, off, **kw)

    _second_derivative_matches(bag, w0, len(name))


def test_dot_on_matrices() -> None:
    b = lucid.tensor(np.random.default_rng(3).standard_normal((4, 2)))
    a0 = np.random.default_rng(4).standard_normal((3, 4))
    _second_derivative_matches(lambda a: a.dot(b), a0, 5)
