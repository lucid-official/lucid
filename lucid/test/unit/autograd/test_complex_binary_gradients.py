"""Complex VJPs conjugate the Jacobian and reduce broadcast dimensions."""

import operator

import numpy as np
import pytest

import lucid


@pytest.mark.parametrize("operation", [operator.add, operator.sub, operator.mul, operator.truediv])
@pytest.mark.parametrize("create_graph", [False, True])
@pytest.mark.parametrize(
    ("dtype", "device"),
    [(lucid.complex64, "cpu"), (lucid.complex128, "cpu"), (lucid.complex64, "metal")],
)
def test_broadcast_complex_vjp(
    operation, create_graph: bool, dtype: lucid.dtype, device: str
) -> None:
    av = np.array([[1 + 2j, -3 + 4j], [2 - 1j, 3 + 2j]])
    bv = np.array([[2 - 3j, 1 + 2j]])
    a = lucid.tensor(av, dtype=dtype, device=device, requires_grad=True)
    b = lucid.tensor(bv, dtype=dtype, device=device, requires_grad=True)
    y = operation(a, b)
    loss = (lucid.real(y) + 2 * lucid.imag(y)).sum()
    if create_graph:
        ga, gb = lucid.autograd.grad(loss, [a, b], create_graph=True)
    else:
        loss.backward()
        ga, gb = a.grad, b.grad
    g = np.full_like(av, 1 + 2j)
    if operation is operator.add:
        wanted_a, wanted_b = g, g
    elif operation is operator.sub:
        wanted_a, wanted_b = g, -g
    elif operation is operator.mul:
        wanted_a, wanted_b = g * bv.conj(), g * av.conj()
    else:
        wanted_a, wanted_b = g / bv.conj(), -g * (av / (bv * bv)).conj()
    np.testing.assert_allclose(ga.numpy(), wanted_a, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        gb.numpy(), wanted_b.sum(axis=0, keepdims=True), rtol=1e-6, atol=1e-6
    )


def test_complex_square_second_derivative() -> None:
    z = lucid.tensor([1 + 2j, -3 + 4j], dtype=lucid.complex128, requires_grad=True)
    (g,) = lucid.autograd.grad(lucid.real(z * z).sum(), [z], create_graph=True)
    np.testing.assert_allclose(g.numpy(), [2 - 4j, -6 - 8j])
    (h,) = lucid.autograd.grad(lucid.real(g).sum(), [z])
    np.testing.assert_allclose(h.numpy(), [2 + 0j, 2 + 0j])
