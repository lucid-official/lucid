"""``expm1`` keeps full relative precision near zero.

``exp(x) - 1`` cancels in float32: it returned ``1.19e-7`` for ``x = 1e-7``
(19% off) and ``0`` below it.  The composite now evaluates Kahan's form for
the value and carries ``exp(x)``'s gradient.
"""

import numpy as np
import pytest

import lucid

XS = np.array(
    [1e-10, -1e-10, 1e-7, -1e-7, 1e-5, 3e-4, -3e-4, 0.01, -0.5, 1.0, 5.0, 20.0, -20.0],
    dtype=np.float32,
)


@pytest.mark.parametrize("device", ["cpu", "metal"])
def test_relative_error_stays_at_float32_precision(device):
    got = lucid.expm1(lucid.tensor(XS, device=device)).numpy().astype(np.float64)
    want = np.expm1(XS.astype(np.float64))
    assert float(np.max(np.abs(got - want) / np.abs(want))) < 3e-7


@pytest.mark.parametrize("device", ["cpu", "metal"])
def test_the_limits(device):
    got = lucid.expm1(
        lucid.tensor([0.0, 200.0, -1e4, float("inf"), -float("inf")], device=device)
    ).numpy()
    assert got[0] == 0.0 and np.isinf(got[1]) and got[2] == -1.0
    assert np.isinf(got[3]) and got[4] == -1.0


def test_the_gradient_is_exp():
    x = lucid.tensor([1e-7, 0.5, -3.0, 0.0], requires_grad=True)
    lucid.expm1(x).sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), np.exp([1e-7, 0.5, -3.0, 0.0]), rtol=1e-6)
    x = lucid.tensor([0.5], requires_grad=True)
    (g,) = lucid.autograd.grad(lucid.expm1(x).sum(), [x], create_graph=True)
    (h,) = lucid.autograd.grad(g.sum(), [x])
    np.testing.assert_allclose(h.numpy(), np.exp([0.5]), rtol=1e-6)
