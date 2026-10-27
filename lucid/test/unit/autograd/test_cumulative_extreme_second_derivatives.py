"""cummax and cummin differentiate twice.

Each output of a running maximum is a copy of one input, so its gradient
is a scatter-add onto the positions the extremes came from — positions
read from the saved output, the same strict comparison the eager backward
uses, so a tie keeps crediting the earlier element.  The graph-mode form
is that scatter-add as an op: differentiable in the incoming gradient and,
rightly, piecewise constant in the input.

Checked against central differences of the first gradient in float64,
with inputs spread far enough apart that a small step changes no winner.
"""

from collections.abc import Callable

import numpy as np
import pytest

import lucid
from lucid.autograd import grad
from lucid.test._fixtures.devices import metal_available

CASES: list[tuple[str, tuple[int, ...], Callable[[lucid.Tensor], lucid.Tensor]]] = [
    ("cummax axis 0", (7, 3), lambda x: lucid.cummax(x, 0).values),
    ("cummax last axis", (2, 3, 6), lambda x: x.cummax(-1).values),
    ("cummin axis 1", (3, 8, 2), lambda x: lucid.cummin(x, 1).values),
]


def _spread(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    # Distinct values a unit apart: no two within reach of a 1e-6 step.
    return rng.permutation(np.arange(np.prod(shape), dtype=np.float64)).reshape(shape)


@pytest.mark.parametrize("name,shape,op", CASES, ids=[c[0] for c in CASES])
def test_the_second_derivative_matches_central_differences(
    name: str, shape: tuple[int, ...], op: Callable[[lucid.Tensor], lucid.Tensor]
) -> None:
    rng = np.random.default_rng(len(name))
    x0 = _spread(rng, shape) * 0.1
    r0 = rng.standard_normal(shape)
    u = rng.standard_normal(shape)

    def loss(x: lucid.Tensor) -> lucid.Tensor:
        y = op(x)
        return (lucid.tensor(r0) * y * y).sum()

    def projected_gradient(values: np.ndarray) -> float:
        x = lucid.tensor(values, requires_grad=True)
        (g,) = grad(loss(x), [x])
        return float((np.asarray(g.numpy()) * u).sum())

    x = lucid.tensor(x0, requires_grad=True)
    (g,) = grad(loss(x), [x], create_graph=True)
    assert float((np.asarray(g.numpy()) * u).sum()) == pytest.approx(
        projected_gradient(x0), rel=1e-12
    )
    (hvp,) = grad((g * lucid.tensor(u)).sum(), [x])
    e = rng.standard_normal(shape)
    h = 1e-6
    numeric = (projected_gradient(x0 + h * e) - projected_gradient(x0 - h * e)) / (2 * h)
    assert float((np.asarray(hvp.numpy()) * e).sum()) == pytest.approx(numeric, rel=1e-6, abs=1e-6)
    assert np.abs(np.asarray(hvp.numpy())).max() > 1e-3


def test_a_tie_credits_the_earlier_position_in_both_modes() -> None:
    values = [2.0, 1.0, 2.0, 3.0, 3.0]
    grads = []
    for create_graph in (False, True):
        x = lucid.tensor(values, requires_grad=True)
        (g,) = grad(lucid.cummax(x, 0).values.sum(), [x], create_graph=create_graph)
        grads.append(np.asarray(g.numpy()))
    np.testing.assert_array_equal(grads[0], [3.0, 0.0, 0.0, 2.0, 0.0])
    np.testing.assert_array_equal(grads[1], grads[0])


@pytest.mark.skipif(not metal_available(), reason="metal unavailable")
def test_metal_differentiates_twice_as_the_cpu_does() -> None:
    rng = np.random.default_rng(4)
    x0 = (_spread(rng, (3, 9)) * 0.1).astype(np.float32)
    answers = {}
    for device in ("cpu", "metal"):
        x = lucid.tensor(x0, device=device, requires_grad=True)
        y = lucid.cummax(x, 1).values
        (g,) = grad((y * y).sum(), [x], create_graph=True)
        (hvp,) = grad((g * g).sum(), [x])
        answers[device] = hvp.numpy()
    np.testing.assert_allclose(answers["metal"], answers["cpu"], rtol=1e-5, atol=1e-5)
