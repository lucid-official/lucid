"""F.bilinear differentiates twice, cross terms included.

``y = x1^T W x2 + b`` is linear in each factor, so every second derivative
through a single factor is zero and all the curvature lives in the cross
terms — which is what a check along one argument alone would miss.  The
graph-mode backward is the three contractions as matmuls, with any leading
axes flattened into one.

Checked against central differences of the first gradient in float64,
through a loss quadratic in the output.
"""

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F
from lucid.autograd import grad

CASES = [
    ("batched with bias", (5, 3), (5, 4), (2, 3, 4), True),
    ("leading axes, no bias", (2, 3, 3), (2, 3, 2), (4, 3, 2), False),
]


@pytest.mark.parametrize("first", [0, 1, 2])
@pytest.mark.parametrize("name,s1,s2,sw,bias", CASES, ids=[c[0] for c in CASES])
def test_the_second_derivative_matches_central_differences(
    first: int, name: str, s1: tuple[int, ...], s2: tuple[int, ...], sw: tuple[int, ...], bias: bool
) -> None:
    rng = np.random.default_rng(len(name) + first)
    base = [rng.standard_normal(s) for s in (s1, s2, sw)]
    b0 = rng.standard_normal(sw[0]) if bias else None
    r0 = rng.standard_normal(s1[:-1] + (sw[0],))
    u = rng.standard_normal(base[first].shape)

    def loss(args: list[lucid.Tensor]) -> lucid.Tensor:
        y = F.bilinear(args[0], args[1], args[2], None if b0 is None else lucid.tensor(b0))
        return (lucid.tensor(r0) * y * y).sum()

    def projected_gradient(values: list[np.ndarray]) -> float:
        args = [lucid.tensor(v, requires_grad=True) for v in values]
        (g,) = grad(loss(args), [args[first]])
        return float((np.asarray(g.numpy()) * u).sum())

    args = [lucid.tensor(v, requires_grad=True) for v in base]
    (g,) = grad(loss(args), [args[first]], create_graph=True)
    assert float((np.asarray(g.numpy()) * u).sum()) == pytest.approx(
        projected_gradient(base), rel=1e-12
    )
    second = grad((g * lucid.tensor(u)).sum(), args, allow_unused=True)

    h = 1e-6
    for k, analytic in enumerate(second):
        e = rng.standard_normal(base[k].shape)
        plus = [v + h * e if i == k else v for i, v in enumerate(base)]
        minus = [v - h * e if i == k else v for i, v in enumerate(base)]
        numeric = (projected_gradient(plus) - projected_gradient(minus)) / (2 * h)
        exact = 0.0 if analytic is None else float((np.asarray(analytic.numpy()) * e).sum())
        assert exact == pytest.approx(numeric, rel=1e-6, abs=1e-6), (first, k)
