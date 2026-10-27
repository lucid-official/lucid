"""Complex projections must preserve both lanes through higher derivatives."""

import numpy as np
import pytest

import lucid


@pytest.mark.parametrize(
    ("dtype", "device"),
    [(lucid.complex64, "cpu"), (lucid.complex128, "cpu"), (lucid.complex64, "metal")],
)
@pytest.mark.parametrize("conjugate", [False, True])
def test_projection_hessian(dtype: lucid.dtype, device: str, conjugate: bool) -> None:
    z = lucid.tensor([1 + 2j, -3 + 4j], dtype=dtype, device=device, requires_grad=True)
    y = lucid.conj(z) if conjugate else z
    real, imag = lucid.real(y), lucid.imag(y)
    loss = (real * real + 3 * imag * imag).sum()
    (gradient,) = lucid.autograd.grad(loss, [z], create_graph=True)
    np.testing.assert_allclose(gradient.numpy(), [2 + 12j, -6 + 24j])
    (hessian,) = lucid.autograd.grad((lucid.real(gradient) + lucid.imag(gradient)).sum(), [z])
    np.testing.assert_allclose(hessian.numpy(), [2 + 6j, 2 + 6j])


@pytest.mark.parametrize("dtype", [lucid.float32, lucid.float64])
def test_complex_assembly_preserves_second_derivative(dtype: lucid.dtype) -> None:
    x = lucid.tensor([1.0, 2.0], dtype=dtype, requires_grad=True)
    z = lucid.complex(x * x, x * x * x)
    (gradient,) = lucid.autograd.grad((lucid.real(z) + lucid.imag(z)).sum(), [x], create_graph=True)
    np.testing.assert_allclose(gradient.numpy(), [5, 16])
    (hessian,) = lucid.autograd.grad(gradient.sum(), [x])
    np.testing.assert_allclose(hessian.numpy(), [8, 14])
