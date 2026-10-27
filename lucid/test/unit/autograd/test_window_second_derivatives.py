"""Window extraction differentiates twice: unfold, fold and Tensor.unfold.

``F.unfold`` copies sliding windows out and ``F.fold`` adds them back, so
each is the other's adjoint and each graph-mode backward is the other op.
``Tensor.unfold`` already built its backward from ops (a scatter-add over
the windows each element fed) but only as storage; the same ops now record
themselves.  ``local_response_norm`` is built on ``Tensor.unfold`` and
refused a second derivative through it — which put AlexNet out of reach of
a gradient penalty.

Checked against central differences of the first gradient in float64, on
losses curved enough that a zero second derivative would fail.
"""

from collections.abc import Callable

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F
from lucid.autograd import grad

CASES: list[tuple[str, tuple[int, ...], Callable[[lucid.Tensor], lucid.Tensor]]] = [
    (
        "unfold",
        (2, 2, 6, 7),
        lambda x: F.unfold(x, kernel_size=(3, 2), stride=(2, 1), padding=1, dilation=(1, 2)),
    ),
    (
        "fold",
        (2, 2 * 3 * 2, 9),
        lambda x: F.fold(x, output_size=(5, 5), kernel_size=(3, 2), stride=(1, 2), padding=(0, 1)),
    ),
    ("Tensor.unfold overlapping", (3, 9, 2), lambda x: x.unfold(1, 4, 2)),
    ("Tensor.unfold last axis", (2, 3, 10), lambda x: x.unfold(-1, 3, 3)),
    (
        "local_response_norm",
        (2, 6, 3, 3),
        lambda x: F.local_response_norm(x, size=3, alpha=0.5, beta=0.75, k=1.0),
    ),
]


@pytest.mark.parametrize("name,shape,op", CASES, ids=[c[0] for c in CASES])
def test_the_second_derivative_matches_central_differences(
    name: str, shape: tuple[int, ...], op: Callable[[lucid.Tensor], lucid.Tensor]
) -> None:
    rng = np.random.default_rng(len(name))
    x0 = rng.standard_normal(shape)
    r0 = rng.standard_normal(op(lucid.tensor(x0)).shape)
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
