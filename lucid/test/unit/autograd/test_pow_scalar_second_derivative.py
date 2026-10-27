"""A scalar power keeps its second derivative.

``pow_scalar``'s graph-mode gradient scaled ``x ** (e - 1)`` by ``e`` on the
raw storage and wrapped the result in a fresh tensor, which cut it off from
``x``: the gradient was right, and its derivative silently lost the
``e (e - 1) x ** (e - 2)`` term — wrong for every exponent but 0 and 1.
Nothing refused, so nothing noticed until a second derivative through
``local_response_norm`` disagreed with finite differences.  The audit's
grad2 axis calls each function with its defaults, and the public paths
that reach this op mostly default to an order that happened to hide it.

Every path below goes through ``pow_scalar`` with an exponent other than 0
or 1, checked against central differences of the first gradient in
float64.
"""

from collections.abc import Callable

import numpy as np
import pytest

import lucid
import lucid.linalg
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.autograd import grad

TARGET = np.array([1, 0, 2])

CASES: list[tuple[str, tuple[int, ...], Callable[[lucid.Tensor], lucid.Tensor]]] = [
    ("vector_norm ord 3", (4, 5), lambda x: lucid.linalg.vector_norm(x, ord=3, dim=1)),
    ("vector_norm ord 1.5", (6,), lambda x: lucid.linalg.vector_norm(x, ord=1.5)),
    # The pool keeps the sign of ``x ** 3``; a window with a negative sum
    # has no real cube root, so it pools the magnitudes.
    ("LPPool2d norm 3", (1, 2, 4, 4), lambda x: nn.LPPool2d(3, kernel_size=2)(lucid.abs(x))),
    (
        "multi_margin_loss p 2",
        (3, 4),
        lambda x: F.multi_margin_loss(x, lucid.tensor(TARGET), p=2, reduction="none"),
    ),
]


@pytest.mark.parametrize("name,shape,op", CASES, ids=[c[0] for c in CASES])
def test_the_second_derivative_matches_central_differences(
    name: str, shape: tuple[int, ...], op: Callable[[lucid.Tensor], lucid.Tensor]
) -> None:
    rng = np.random.default_rng(len(name))
    # Kept away from zero: |x| ** e is not twice differentiable there.
    x0 = rng.uniform(0.5, 1.5, shape) * rng.choice([-1.0, 1.0], shape)
    r0 = rng.standard_normal(op(lucid.tensor(x0)).shape)
    u = rng.standard_normal(shape)

    def loss(x: lucid.Tensor) -> lucid.Tensor:
        return (lucid.tensor(r0) * op(x)).sum()

    def projected_gradient(values: np.ndarray) -> float:
        x = lucid.tensor(values, requires_grad=True)
        (g,) = grad(loss(x), [x])
        return float((np.asarray(g.numpy()) * u).sum())

    x = lucid.tensor(x0, requires_grad=True)
    (g,) = grad(loss(x), [x], create_graph=True)
    (hvp,) = grad((g * lucid.tensor(u)).sum(), [x])
    e = rng.standard_normal(shape)
    h = 1e-6
    numeric = (projected_gradient(x0 + h * e) - projected_gradient(x0 - h * e)) / (2 * h)
    assert float((np.asarray(hvp.numpy()) * e).sum()) == pytest.approx(numeric, rel=1e-6, abs=1e-6)
