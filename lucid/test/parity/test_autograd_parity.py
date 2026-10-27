"""Reference parity for backward / gradient values."""

from typing import Any

import numpy as np
import pytest

import lucid
from lucid.test._helpers.compare import assert_close


@pytest.mark.parity
class TestAutogradParity:
    def test_square_sum_backward(self, ref: Any) -> None:
        x_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        x_l = lucid.tensor(x_np.copy(), requires_grad=True)
        (x_l * x_l).sum().backward()

        x_r = ref.tensor(x_np.copy(), requires_grad=True)
        (x_r * x_r).sum().backward()

        np.testing.assert_allclose(x_l.grad.numpy(), x_r.grad.detach().cpu().numpy(), atol=1e-5)

    def test_chain_backward(self, ref: Any) -> None:
        x_np = np.array([1.5, 2.5], dtype=np.float32)

        x_l = lucid.tensor(x_np.copy(), requires_grad=True)
        (x_l.exp().sum() + x_l.sin().sum()).backward()

        x_r = ref.tensor(x_np.copy(), requires_grad=True)
        (x_r.exp().sum() + x_r.sin().sum()).backward()

        np.testing.assert_allclose(x_l.grad.numpy(), x_r.grad.detach().cpu().numpy(), atol=1e-5)

    def test_matmul_backward(self, ref: Any) -> None:
        np.random.seed(0)
        a_np = np.random.standard_normal(size=(3, 4)).astype(np.float32)
        b_np = np.random.standard_normal(size=(4, 2)).astype(np.float32)

        a_l = lucid.tensor(a_np.copy(), requires_grad=True)
        b_l = lucid.tensor(b_np.copy(), requires_grad=True)
        (a_l @ b_l).sum().backward()

        a_r = ref.tensor(a_np.copy(), requires_grad=True)
        b_r = ref.tensor(b_np.copy(), requires_grad=True)
        (a_r @ b_r).sum().backward()

        np.testing.assert_allclose(a_l.grad.numpy(), a_r.grad.detach().cpu().numpy(), atol=1e-4)
        np.testing.assert_allclose(b_l.grad.numpy(), b_r.grad.detach().cpu().numpy(), atol=1e-4)


@pytest.mark.parity
class TestSecondOrderParity:
    """2nd-order gradients: d²(f)/dx² must match the reference framework."""

    def test_exp_hessian_diagonal(self, ref: Any) -> None:
        # d²(exp)/dx² = exp(x); use small x to keep values manageable.
        x_np = np.array([0.5, 1.0, 1.5], dtype=np.float32)

        x_l = lucid.tensor(x_np.copy(), requires_grad=True)
        y_l = x_l.exp().sum()
        (g_l,) = lucid.autograd.grad(y_l, [x_l], create_graph=True)
        g_l.sum().backward()

        x_r = ref.tensor(x_np.copy(), requires_grad=True)
        y_r = x_r.exp().sum()
        (g_r,) = ref.autograd.grad(y_r, [x_r], create_graph=True)
        g_r.sum().backward()

        np.testing.assert_allclose(x_l.grad.numpy(), x_r.grad.detach().cpu().numpy(), atol=1e-5)

    def test_quadratic_hessian(self, ref: Any) -> None:
        # f(x) = sum(x²) → grad = 2x → hess diagonal = 2.
        x_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        x_l = lucid.tensor(x_np.copy(), requires_grad=True)
        (g_l,) = lucid.autograd.grad((x_l * x_l).sum(), [x_l], create_graph=True)
        g_l.sum().backward()

        x_r = ref.tensor(x_np.copy(), requires_grad=True)
        (g_r,) = ref.autograd.grad((x_r * x_r).sum(), [x_r], create_graph=True)
        g_r.sum().backward()

        np.testing.assert_allclose(x_l.grad.numpy(), x_r.grad.detach().cpu().numpy(), atol=1e-5)

    def test_jacobian_shape(self, ref: Any) -> None:
        # jacobian of f(x)=(x², 2x, 3x): shape should be (3, 3).
        x_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        x_l = lucid.tensor(x_np.copy())
        x_r = ref.tensor(x_np.copy())

        # f: R³ → R³ via elementwise scaling.
        scales_l = lucid.tensor([1.0, 2.0, 3.0])
        scales_r = ref.tensor([1.0, 2.0, 3.0])

        jac_l = lucid.autograd.jacobian(lambda x: x * scales_l, x_l)
        jac_r = ref.autograd.functional.jacobian(lambda x: x * scales_r, x_r)

        assert_close(jac_l, jac_r, atol=1e-5)

    def test_hessian_diagonal_matches(self, ref: Any) -> None:
        x_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        x_l = lucid.tensor(x_np.copy())
        x_r = ref.tensor(x_np.copy())

        hess_l = lucid.autograd.hessian(lambda x: (x * x).sum(), x_l)
        hess_r = ref.autograd.functional.hessian(lambda x: (x * x).sum(), x_r)

        # Diagonal should be 2; off-diagonal 0.
        assert_close(hess_l, hess_r, atol=1e-5)

    def test_vjp_matches(self, ref: Any) -> None:
        x_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        v_np = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        x_l = lucid.tensor(x_np.copy())
        v_l = lucid.tensor(v_np.copy())
        x_r = ref.tensor(x_np.copy())
        v_r = ref.tensor(v_np.copy())

        _, vjp_l = lucid.autograd.vjp(lambda x: x * x, x_l, v_l)
        _, vjp_r = ref.autograd.functional.vjp(lambda x: x * x, x_r, v_r)
        # A bare tensor in, a bare tensor out — on both sides.
        assert not isinstance(vjp_l, tuple) and not isinstance(vjp_r, tuple)
        assert_close(vjp_l, vjp_r, atol=1e-5)

    def test_jvp_matches(self, ref: Any) -> None:
        x_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        t_np = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        x_l = lucid.tensor(x_np.copy())
        t_l = lucid.tensor(t_np.copy())
        x_r = ref.tensor(x_np.copy())
        t_r = ref.tensor(t_np.copy())

        # jvp of f(x)=x² at x with tangent t: 2*x*t.
        _, jvp_l = lucid.autograd.jvp(lambda x: x * x, x_l, t_l)
        _, jvp_r = ref.autograd.functional.jvp(lambda x: x * x, x_r, t_r)
        assert_close(jvp_l, jvp_r, atol=1e-3)
