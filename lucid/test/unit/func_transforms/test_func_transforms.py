"""Unit tests for lucid.func functional transforms (grad, vjp, jvp, jacrev, jacfwd, hessian)."""

import pytest
import lucid
import lucid.func as func
from lucid.func import _forward as forward

# ── func.grad ─────────────────────────────────────────────────────────────────


class TestFuncGrad:
    def test_scalar_quadratic(self) -> None:
        f = lambda x: (x**2).sum()
        df = func.grad(f)
        x = lucid.tensor([1.0, 2.0, 3.0])
        g = df(x)
        assert lucid.allclose(g, x * 2)

    def test_argnums_single(self) -> None:
        f = lambda x, y: (x * y).sum()
        df = func.grad(f, argnums=1)
        x = lucid.tensor([2.0, 3.0])
        y = lucid.tensor([1.0, 4.0])
        g = df(x, y)
        assert lucid.allclose(g, x)

    def test_argnums_tuple(self) -> None:
        f = lambda x, y: (x * y).sum()
        df = func.grad(f, argnums=(0, 1))
        x = lucid.tensor([2.0, 3.0])
        y = lucid.tensor([1.0, 4.0])
        gx, gy = df(x, y)
        assert lucid.allclose(gx, y)
        assert lucid.allclose(gy, x)

    def test_has_aux(self) -> None:
        f = lambda x: ((x**2).sum(), x * 10)
        df = func.grad(f, has_aux=True)
        x = lucid.tensor([1.0, 2.0])
        g, aux = df(x)
        assert lucid.allclose(g, x * 2)
        assert lucid.allclose(aux, x * 10)

    def test_second_derivative_via_hessian(self) -> None:
        """Second derivative via func.hessian (the correct composable path)."""
        f = lambda x: (x**3).sum()
        H = func.hessian(f)(lucid.tensor([2.0]))
        # d²/dx²(x³) = 6x → at x=2: 12
        assert abs(float(H.item()) - 12.0) < 1e-2


# ── func.grad_and_value ───────────────────────────────────────────────────────


class TestGradAndValue:
    def test_returns_grad_and_value(self) -> None:
        f = lambda x: (x**2).sum()
        gv = func.grad_and_value(f)
        x = lucid.tensor([1.0, 2.0, 3.0])
        g, v = gv(x)
        assert lucid.allclose(g, x * 2)
        assert abs(float(v.item()) - 14.0) < 1e-5

    def test_value_is_detached(self) -> None:
        # The backward that produced the gradient has already consumed the
        # graph behind the value; a value still claiming requires_grad
        # could only fail later, on a second backward.
        f = lambda x: (x**2).sum()
        _, v = func.grad_and_value(f)(lucid.tensor([1.0, 2.0, 3.0]))
        assert not v.requires_grad

    def test_value_is_detached_with_aux(self) -> None:
        f = lambda x: ((x**2).sum(), x * 10)
        _, (v, _aux) = func.grad_and_value(f, has_aux=True)(lucid.tensor([1.0, 2.0]))
        assert not v.requires_grad
        assert abs(float(v.item()) - 5.0) < 1e-5


# ── func.vjp ─────────────────────────────────────────────────────────────────


class TestFuncVJP:
    def test_vjp_quadratic(self) -> None:
        f = lambda x: x**2
        x = lucid.tensor([1.0, 2.0, 3.0])
        y, vjp_fn = func.vjp(f, x)
        (g,) = vjp_fn(lucid.ones_like(y))
        assert lucid.allclose(g, x * 2)

    def test_vjp_output_shape(self) -> None:
        f = lambda x: x @ x.T
        x = lucid.randn(3, 4)
        y, vjp_fn = func.vjp(f, x)
        assert list(y.shape) == [3, 3]
        (g,) = vjp_fn(lucid.ones(3, 3))
        assert list(g.shape) == [3, 4]

    def test_vjp_multiple_primals(self) -> None:
        f = lambda x, y: x * y
        x = lucid.tensor([2.0, 3.0])
        y = lucid.tensor([4.0, 5.0])
        out, vjp_fn = func.vjp(f, x, y)
        gx, gy = vjp_fn(lucid.ones_like(out))
        assert lucid.allclose(gx, y)
        assert lucid.allclose(gy, x)

    def test_vjp_has_aux(self) -> None:
        f = lambda x: (x**2, x * 3)
        x = lucid.tensor([1.0, 2.0])
        (y, aux), vjp_fn = func.vjp(f, x, has_aux=True)
        assert lucid.allclose(aux, x * 3)
        (g,) = vjp_fn(lucid.ones_like(y))
        assert lucid.allclose(g, x * 2)


# ── func.jvp ─────────────────────────────────────────────────────────────────


class TestFuncJVP:
    def test_jvp_scalar_exact(self) -> None:
        """JVP of x²: d/dt (x+tv)² at t=0 = 2xv."""
        f = lambda x: (x**2).sum()
        x = lucid.tensor([1.0, 2.0, 3.0])
        v = lucid.ones(3)
        primals_out, tangents_out = func.jvp(f, (x,), (v,))
        # d/dt sum((x+tv)²) = 2*(x·v) = 2*(1+2+3) = 12
        assert abs(float(tangents_out.item()) - 12.0) < 1e-3

    def test_jvp_linear(self) -> None:
        """JVP of a diagonal scale function: J = diag(scales)."""
        # f(x) = [x[0], 2*x[1]] via elementwise multiply — avoids 1D matmul
        scales = lucid.tensor([1.0, 2.0])
        f = lambda x: x * scales
        x = lucid.tensor([1.0, 1.0])
        v = lucid.tensor([1.0, 1.0])
        primals_out, tangents_out = func.jvp(f, (x,), (v,))
        # J = diag(scales), JVP = scales * v = [1, 2]
        assert lucid.allclose(tangents_out, lucid.tensor([1.0, 2.0]), atol=1e-3)

    def test_jvp_matches_vjp(self) -> None:
        """For diagonal Jacobian: JVP and VJP with same vector should agree."""
        f = lambda x: x * 2
        x = lucid.tensor([1.0, 2.0, 3.0])
        v = lucid.tensor([1.0, 0.0, 0.0])
        _, jvp_out = func.jvp(f, (x,), (v,))
        _, vjp_fn = func.vjp(f, x)
        (vjp_out,) = vjp_fn(v)
        # For f(x)=2x: J = 2I, so Jv = 2v[0]*e_0 = [2, 0, 0] — same as VJP
        assert lucid.allclose(jvp_out, vjp_out, atol=1e-3)

    def test_jvp_on_each_device(self, device: str) -> None:
        """The perturbation lives with the tangent — on the CPU it could not
        multiply a Metal tangent, and every jvp on Metal raised."""
        x = lucid.tensor([1.0, 2.0, 3.0]).to(device)
        out, tangent = func.jvp(lambda v: v**2, (x,), (lucid.ones_like(x),))
        assert tangent.device == x.device
        assert tangent.tolist() == [2.0, 4.0, 6.0]

    def test_jvp_vector_output_through_attention(
        self, device: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A vector output takes two backward passes, not one per element.

        One per element made MeanFlow's training loss — a jvp through a
        whole transformer — thousands of passes long.  The two-pass route
        differentiates a backward, which the fused attention's is not; jvp
        runs attention in its explicit form, and the route checks itself
        against one plain backward before trusting the result.
        """
        import lucid.nn as nn
        import lucid.nn.functional as F

        lucid.manual_seed(0)
        lin = nn.Linear(8, 8).to(device)
        ln = nn.LayerNorm(8).to(device)

        def f(x: lucid.Tensor) -> lucid.Tensor:
            h = ln(lin(x)).reshape(2, 2, 4, 4)
            return F.gelu(F.scaled_dot_product_attention(h, h, h)).reshape(8, 8)

        taken: list[bool] = []
        fast = forward._jvp_by_double_backward

        def spy(*a: object, **k: object) -> object:
            r = fast(*a, **k)  # type: ignore[arg-type]
            taken.append(r is not None)
            return r

        monkeypatch.setattr(forward, "_jvp_by_double_backward", spy)
        x = lucid.randn(8, 8).to(device)
        t = lucid.randn(8, 8).to(device)
        _, jt = func.jvp(f, (x,), (t,))
        eps = 1e-3
        fd = (f(x + eps * t) - f(x - eps * t)) / (2 * eps)
        assert taken == [True]
        assert float((jt - fd).abs().max().item()) < 1e-3

    def test_jvp_leaves_parameter_gradients_alone(self) -> None:
        """jvp's own backward passes accumulate into no leaf.

        They used ``backward()``, which fills every leaf's ``.grad`` — a
        model's parameters picked up one junk gradient per output element,
        and MeanFlow trained on those instead of on its loss.
        """
        import lucid.nn as nn

        lucid.manual_seed(0)
        lin = nn.Linear(3, 3)
        x = lucid.randn(4, 3)
        func.jvp(lambda v: lin(v).tanh(), (x,), (lucid.ones_like(x),))
        func.jvp(lambda v: lin(v).tanh().sum(), (x,), (lucid.ones_like(x),))
        assert all(p.grad is None for p in lin.parameters())

    def test_jvp_primal_output_is_differentiable(self) -> None:
        """A loss built from jvp's primal output trains the parameters —
        it used to be computed under ``no_grad`` and carried nothing."""
        import lucid.nn as nn

        lucid.manual_seed(0)
        lin = nn.Linear(3, 3)
        x = lucid.randn(4, 3)
        y, _ = func.jvp(lambda v: lin(v).tanh(), (x,), (lucid.ones_like(x),))
        (y * y).sum().backward()
        lucid.manual_seed(0)
        ref = nn.Linear(3, 3)
        (ref(x).tanh() ** 2).sum().backward()
        for p, q in zip(lin.parameters(), ref.parameters()):
            assert p.grad is not None
            assert lucid.allclose(p.grad, q.grad, atol=1e-6)

    def test_jvp_strict_raises(self) -> None:
        x = lucid.tensor([1.0, 2.0])
        v = lucid.ones(2)
        # constant output does not depend on x
        f = lambda x: lucid.ones(2)
        with pytest.raises((ValueError, RuntimeError)):
            func.jvp(f, (x,), (v,), strict=True)


# ── func.jacrev ───────────────────────────────────────────────────────────────


class TestJacRev:
    def test_identity_jacobian(self) -> None:
        f = lambda x: x
        x = lucid.tensor([1.0, 2.0, 3.0])
        J = func.jacrev(f)(x)
        assert list(J.shape) == [3, 3]
        assert lucid.allclose(J, lucid.eye(3))

    def test_linear_jacobian(self) -> None:
        # f(x) = x * [1, 3] — diagonal linear map, J = diag([1, 3])
        scales = lucid.tensor([1.0, 3.0])
        f = lambda x: x * scales
        x = lucid.tensor([2.0, 5.0])
        J = func.jacrev(f)(x)
        assert list(J.shape) == [2, 2]
        expected = lucid.tensor([[1.0, 0.0], [0.0, 3.0]])
        assert lucid.allclose(J, expected, atol=1e-4)

    def test_scalar_output_matches_grad(self) -> None:
        f = lambda x: (x**2).sum()
        x = lucid.tensor([1.0, 2.0, 3.0])
        J = func.jacrev(f)(x)
        df = func.grad(f)
        g = df(x)
        assert lucid.allclose(J.reshape(-1), g.reshape(-1), atol=1e-4)


# ── func.jacfwd ───────────────────────────────────────────────────────────────


class TestJacFwd:
    def test_identity_jacobian(self) -> None:
        f = lambda x: x
        x = lucid.tensor([1.0, 2.0, 3.0])
        J = func.jacfwd(f)(x)
        assert list(J.shape) == [3, 3]
        assert lucid.allclose(J, lucid.eye(3), atol=1e-3)

    def test_jacfwd_matches_jacrev(self) -> None:
        # Use elementwise scaling — works for any shape, no matmul issues
        scales = lucid.tensor([2.0, 3.0])
        f = lambda x: x * scales
        x = lucid.tensor([1.0, 1.0])
        Jrev = func.jacrev(f)(x)
        Jfwd = func.jacfwd(f)(x)
        assert lucid.allclose(Jrev, Jfwd, atol=1e-3)


# ── func.hessian ──────────────────────────────────────────────────────────────


class TestHessian:
    def test_quadratic_hessian(self) -> None:
        """H of sum(x²) = 2I."""
        f = lambda x: (x**2).sum()
        x = lucid.tensor([1.0, 2.0])
        H = func.hessian(f)(x)
        assert list(H.shape) == [2, 2]
        assert lucid.allclose(H, lucid.eye(2) * 2, atol=1e-3)


# ── func.linearize ────────────────────────────────────────────────────────────


class TestLinearize:
    def test_linearize_at_zero(self) -> None:
        """Linearize of exp(x) at x=0: lin_fn(v) ≈ J @ v = v."""
        f = lambda x: lucid.exp(x)
        x = lucid.zeros(3)
        primals_out, lin_fn = func.linearize(f, x)
        v = lucid.tensor([1.0, 0.0, 0.0])
        tangent = lin_fn(v)
        assert lucid.allclose(tangent, v, atol=1e-3)
