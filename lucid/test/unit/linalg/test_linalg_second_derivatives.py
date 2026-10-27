"""The factorisations differentiate — once correctly, and then again.

Three defects, found together by a second-derivative check (CHA-8):

* ``solve_triangular`` recorded no gradient at all.  Its output came back
  with ``requires_grad=False`` whatever its inputs asked, so a loss through
  it lost the solve's contribution without a word — and ``cholesky``'s
  backward, two of these solves, came back detached under
  ``create_graph=True``.  Its forward also paired batches one to one, so a
  single triangle against a batch of right-hand sides returned wrong
  values.
* ``eigh``'s eigenvector gradient had the wrong sign: ``F[i, j]`` was
  ``1 / (w[i] - w[j])`` where it is ``1 / (w[j] - w[i])``.  An
  eigenvalue-only loss never reaches that term, which is how it hid.
* ``qr``'s R-path backward detached its Cholesky factor, dropping a term of
  the second derivative.

With those fixed the three stopped refusing ``create_graph=True``.  All
checks are central differences in float64 — the eigenvector one against
numpy's own ``eigh`` — so no oracle is involved.
"""

from collections.abc import Callable

import numpy as np
import pytest

import lucid
import lucid.linalg as LA
from lucid.autograd import grad

Fn = Callable[..., lucid.Tensor]


def _central(f: Callable[[float], float], h: float = 1e-6) -> float:
    return (f(h) - f(-h)) / (2 * h)


def _spd(a: lucid.Tensor) -> lucid.Tensor:
    return a @ a.mT + 4 * lucid.eye(int(a.shape[-1]), dtype=a.dtype)


def _sym(a: lucid.Tensor) -> lucid.Tensor:
    return a + a.mT


def test_the_eigenvector_gradient_has_the_right_sign() -> None:
    rng = np.random.default_rng(1)
    a0 = rng.standard_normal((4, 4))
    a0 = a0 + a0.T
    r = rng.standard_normal((4, 4))
    a = lucid.tensor(a0, requires_grad=True)
    (lucid.tensor(r) * LA.eigh(a)[1] ** 4).sum().backward()
    e = rng.standard_normal((4, 4))
    e = e + e.T  # a symmetric step: unambiguous whatever triangle is read
    numeric = _central(lambda s: float((r * np.linalg.eigh(a0 + s * e)[1] ** 4).sum()))
    assert float((np.asarray(a.grad.numpy()) * e).sum()) == pytest.approx(numeric, rel=1e-6)


def test_a_triangular_solve_records_its_gradient() -> None:
    rng = np.random.default_rng(2)
    a = lucid.tensor(np.triu(rng.standard_normal((4, 4))) + 4 * np.eye(4), requires_grad=True)
    b = lucid.tensor(rng.standard_normal((4, 3)), requires_grad=True)
    x = LA.solve_triangular(a, b, upper=True)
    assert x.requires_grad and x.grad_fn is not None


def test_one_triangle_solves_a_batch_of_right_hand_sides() -> None:
    rng = np.random.default_rng(3)
    a0 = np.tril(rng.standard_normal((4, 4))) + 4 * np.eye(4)
    b0 = rng.standard_normal((2, 4, 3))
    x = LA.solve_triangular(lucid.tensor(a0), lucid.tensor(b0), upper=False)
    want = np.stack([np.linalg.solve(a0, b0[i]) for i in range(2)])
    np.testing.assert_allclose(np.asarray(x.numpy()), want, rtol=1e-12)


CASES: list[tuple[str, list[tuple[int, ...]], Fn]] = [
    (
        "solve_triangular",
        [(4, 4), (4, 3)],
        lambda a, b: LA.solve_triangular(a, b, upper=True),
    ),
    (
        "solve_triangular unit lower",
        [(4, 4), (4, 2)],
        lambda a, b: LA.solve_triangular(a, b, upper=False, unitriangular=True),
    ),
    (
        "solve_triangular right side",
        [(3, 3), (2, 3)],
        lambda a, b: LA.solve_triangular(a, b, upper=True, left=False),
    ),
    (
        "solve_triangular broadcast",
        [(4, 4), (2, 4, 3)],
        lambda a, b: LA.solve_triangular(a, b, upper=False),
    ),
    ("cholesky", [(4, 4)], lambda a: LA.cholesky(_spd(a))),
    ("cholesky upper", [(3, 3)], lambda a: LA.cholesky(_spd(a), upper=True)),
    ("cholesky_ex", [(4, 4)], lambda a: LA.cholesky_ex(_spd(a))[0]),
    ("qr Q", [(4, 3)], lambda a: LA.qr(a)[0]),
    ("qr R", [(4, 3)], lambda a: LA.qr(a)[1]),
    ("eigh values", [(4, 4)], lambda a: LA.eigh(_sym(a))[0]),
    ("eigh vectors", [(4, 4)], lambda a: LA.eigh(_sym(a))[1] ** 2),
    ("eigvalsh", [(4, 4)], lambda a: LA.eigvalsh(_sym(a))),
]


@pytest.mark.parametrize("name,shapes,fn", CASES, ids=[c[0] for c in CASES])
def test_the_second_derivative_matches_central_differences(
    name: str, shapes: list[tuple[int, ...]], fn: Fn
) -> None:
    rng = np.random.default_rng(len(name))
    base = [rng.standard_normal(s) for s in shapes]
    if shapes[0][-1] == shapes[0][-2]:
        base[0] = base[0] + 4 * np.eye(shapes[0][-1])  # well away from singular
    r = rng.standard_normal(fn(*[lucid.tensor(b) for b in base]).shape)
    u = [rng.standard_normal(b.shape) for b in base]

    def loss(*args: lucid.Tensor) -> lucid.Tensor:
        return (lucid.tensor(r) * fn(*args) ** 2).sum()

    def projected_gradient(values: list[np.ndarray]) -> float:
        args = [lucid.tensor(v, requires_grad=True) for v in values]
        grads = grad(loss(*args), args)
        return sum(float((np.asarray(g.numpy()) * w).sum()) for g, w in zip(grads, u))

    args = [lucid.tensor(b, requires_grad=True) for b in base]
    grads = grad(loss(*args), args, create_graph=True)
    projected = sum((g * lucid.tensor(w)).sum() for g, w in zip(grads, u))
    second = grad(projected, args, allow_unused=True)
    for k, analytic in enumerate(second):
        e = rng.standard_normal(base[k].shape)
        numeric = _central(
            lambda s: projected_gradient([b + s * e if i == k else b for i, b in enumerate(base)])
        )
        exact = 0.0 if analytic is None else float((np.asarray(analytic.numpy()) * e).sum())
        assert exact == pytest.approx(numeric, rel=1e-5, abs=1e-6), k
