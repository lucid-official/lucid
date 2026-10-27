"""Parity tests for lucid.func against the reference framework's func module.

Each test verifies that lucid.func transforms produce numerically identical
results to the reference framework's equivalent transform.

Coverage:
  vmap       — batch vectorisation: elementwise, matmul, in_dims, composition
  grad       — gradient transform: scalar, argnums, has_aux
  grad_and_value — returns (grads, value)
  vjp        — vector-Jacobian product: primals_out + vjp_fn cotangents
  jvp        — Jacobian-vector product: scalar and vector output, tangents
  jacrev     — reverse-mode Jacobian: identity, diagonal, per-sample
  jacfwd     — forward-mode Jacobian: agrees with jacrev
  hessian    — second-order: quadratic, diagonal
"""

from typing import Any

import numpy as np
import pytest

import lucid
import lucid.func as func
from lucid.test._helpers.compare import assert_close

# ── helpers ───────────────────────────────────────────────────────────────────


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# ── vmap ─────────────────────────────────────────────────────────────────────


@pytest.mark.parity
class TestVmapParity:
    def test_elementwise(self, ref: Any) -> None:
        """vmap over elementwise fn matches reference."""
        rng = _rng(0)
        x_np = rng.standard_normal((8, 4)).astype(np.float32)

        f_l = lambda x: x**2 + x
        f_r = lambda x: x**2 + x

        y_l = func.vmap(f_l)(lucid.tensor(x_np.copy()))
        y_r = ref.func.vmap(f_r)(ref.tensor(x_np.copy()))
        assert_close(y_l, y_r, atol=1e-5)

    def test_scalar_reduction(self, ref: Any) -> None:
        """vmap over fn returning scalar per element."""
        rng = _rng(1)
        x_np = rng.standard_normal((6, 5)).astype(np.float32)

        f_l = lambda x: (x**2).sum(dim=-1)
        f_r = lambda x: (x**2).sum(dim=-1)

        y_l = func.vmap(f_l)(lucid.tensor(x_np.copy()))
        y_r = ref.func.vmap(f_r)(ref.tensor(x_np.copy()))
        assert_close(y_l, y_r, atol=1e-5)

    def test_in_dims_none_broadcast(self, ref: Any) -> None:
        """Broadcast arg (in_dims=None) passes through unchanged."""
        rng = _rng(2)
        x_np = rng.standard_normal((7, 3)).astype(np.float32)
        w_np = rng.standard_normal((5, 3)).astype(np.float32)

        f_l = lambda x, w: x @ w.T
        f_r = lambda x, w: x @ w.T

        y_l = func.vmap(f_l, in_dims=(0, None))(
            lucid.tensor(x_np.copy()), lucid.tensor(w_np.copy())
        )
        y_r = ref.func.vmap(f_r, in_dims=(0, None))(
            ref.tensor(x_np.copy()), ref.tensor(w_np.copy())
        )
        assert_close(y_l, y_r, atol=1e-5)

    def test_out_dims(self, ref: Any) -> None:
        """out_dims=1 places batch at dim 1 of output."""
        rng = _rng(3)
        x_np = rng.standard_normal((4, 3)).astype(np.float32)

        f_l = lambda x: x * 2
        f_r = lambda x: x * 2

        y_l = func.vmap(f_l, in_dims=0, out_dims=1)(lucid.tensor(x_np.copy()))
        y_r = ref.func.vmap(f_r, in_dims=0, out_dims=1)(ref.tensor(x_np.copy()))
        assert list(y_l.shape) == list(y_r.shape)
        assert_close(y_l, y_r, atol=1e-5)

    def test_vmap_grad_composition(self, ref: Any) -> None:
        """vmap(grad(fn)) gives per-sample gradients matching reference."""
        rng = _rng(4)
        x_np = rng.standard_normal((5, 3)).astype(np.float32)

        f_l = lambda x: (x**2).sum()
        f_r = lambda x: (x**2).sum()

        grads_l = func.vmap(func.grad(f_l))(lucid.tensor(x_np.copy()))
        grads_r = ref.func.vmap(ref.func.grad(f_r))(ref.tensor(x_np.copy()))
        assert_close(grads_l, grads_r, atol=1e-5)

    def test_chunk_size(self, ref: Any) -> None:
        """chunk_size produces same result as full-batch dispatch."""
        rng = _rng(5)
        x_np = rng.standard_normal((12, 4)).astype(np.float32)

        f_l = lambda x: lucid.relu(x) * 2
        f_r = lambda x: ref.relu(x) * 2

        full_l = func.vmap(f_l)(lucid.tensor(x_np.copy()))
        chunk_l = func.vmap(f_l, chunk_size=4)(lucid.tensor(x_np.copy()))
        ref_l = ref.func.vmap(f_r)(ref.tensor(x_np.copy()))

        assert_close(full_l, ref_l, atol=1e-5)
        assert_close(chunk_l, ref_l, atol=1e-5)


# ── grad ─────────────────────────────────────────────────────────────────────


@pytest.mark.parity
class TestGradParity:
    def test_scalar_quadratic(self, ref: Any) -> None:
        """grad of sum(x²) = 2x."""
        x_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        f_l = lambda x: (x**2).sum()
        f_r = lambda x: (x**2).sum()

        g_l = func.grad(f_l)(lucid.tensor(x_np.copy()))
        g_r = ref.func.grad(f_r)(ref.tensor(x_np.copy()))
        assert_close(g_l, g_r, atol=1e-5)

    def test_argnums_1(self, ref: Any) -> None:
        """grad w.r.t. second argument."""
        x_np = np.array([2.0, 3.0], dtype=np.float32)
        y_np = np.array([1.0, 4.0], dtype=np.float32)

        f_l = lambda x, y: (x * y).sum()
        f_r = lambda x, y: (x * y).sum()

        g_l = func.grad(f_l, argnums=1)(lucid.tensor(x_np.copy()), lucid.tensor(y_np.copy()))
        g_r = ref.func.grad(f_r, argnums=1)(ref.tensor(x_np.copy()), ref.tensor(y_np.copy()))
        assert_close(g_l, g_r, atol=1e-5)

    def test_argnums_tuple(self, ref: Any) -> None:
        """grad w.r.t. both arguments."""
        x_np = np.array([2.0, 3.0], dtype=np.float32)
        y_np = np.array([1.0, 4.0], dtype=np.float32)

        f_l = lambda x, y: (x * y).sum()
        f_r = lambda x, y: (x * y).sum()

        gx_l, gy_l = func.grad(f_l, argnums=(0, 1))(
            lucid.tensor(x_np.copy()), lucid.tensor(y_np.copy())
        )
        gx_r, gy_r = ref.func.grad(f_r, argnums=(0, 1))(
            ref.tensor(x_np.copy()), ref.tensor(y_np.copy())
        )
        assert_close(gx_l, gx_r, atol=1e-5)
        assert_close(gy_l, gy_r, atol=1e-5)

    def test_has_aux(self, ref: Any) -> None:
        """grad with has_aux=True returns (grad, aux)."""
        x_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        f_l = lambda x: ((x**2).sum(), x * 3)
        f_r = lambda x: ((x**2).sum(), x * 3)

        g_l, aux_l = func.grad(f_l, has_aux=True)(lucid.tensor(x_np.copy()))
        g_r, aux_r = ref.func.grad(f_r, has_aux=True)(ref.tensor(x_np.copy()))
        assert_close(g_l, g_r, atol=1e-5)
        assert_close(aux_l, aux_r, atol=1e-5)


# ── grad_and_value ────────────────────────────────────────────────────────────


@pytest.mark.parity
class TestGradAndValueParity:
    def test_basic(self, ref: Any) -> None:
        """grad_and_value returns matching (grads, value)."""
        x_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        f_l = lambda x: (x**2).sum()
        f_r = lambda x: (x**2).sum()

        g_l, v_l = func.grad_and_value(f_l)(lucid.tensor(x_np.copy()))
        g_r, v_r = ref.func.grad_and_value(f_r)(ref.tensor(x_np.copy()))
        assert_close(g_l, g_r, atol=1e-5)
        assert_close(v_l, v_r, atol=1e-5)


# ── vjp ──────────────────────────────────────────────────────────────────────


@pytest.mark.parity
class TestVJPParity:
    def test_basic_elementwise(self, ref: Any) -> None:
        """vjp cotangent computation matches reference."""
        x_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ct_np = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        f_l = lambda x: x**2
        f_r = lambda x: x**2

        y_l, vjp_l = func.vjp(f_l, lucid.tensor(x_np.copy()))
        y_r, vjp_r = ref.func.vjp(f_r, ref.tensor(x_np.copy()))

        assert_close(y_l, y_r, atol=1e-5)

        (g_l,) = vjp_l(lucid.tensor(ct_np.copy()))
        (g_r,) = vjp_r(ref.tensor(ct_np.copy()))
        assert_close(g_l, g_r, atol=1e-5)

    def test_multiple_primals(self, ref: Any) -> None:
        """vjp with two primals — cotangents split correctly."""
        x_np = np.array([2.0, 3.0], dtype=np.float32)
        y_np = np.array([4.0, 5.0], dtype=np.float32)
        ct_np = np.array([1.0, 1.0], dtype=np.float32)

        f_l = lambda x, y: x * y
        f_r = lambda x, y: x * y

        out_l, vjp_l = func.vjp(f_l, lucid.tensor(x_np.copy()), lucid.tensor(y_np.copy()))
        out_r, vjp_r = ref.func.vjp(f_r, ref.tensor(x_np.copy()), ref.tensor(y_np.copy()))

        gx_l, gy_l = vjp_l(lucid.tensor(ct_np.copy()))
        gx_r, gy_r = vjp_r(ref.tensor(ct_np.copy()))
        assert_close(gx_l, gx_r, atol=1e-5)
        assert_close(gy_l, gy_r, atol=1e-5)

    def test_matrix_fn(self, ref: Any) -> None:
        """vjp through a matrix function."""
        rng = _rng(6)
        x_np = rng.standard_normal((3, 4)).astype(np.float32)

        f_l = lambda x: (x**2).sum(dim=-1)
        f_r = lambda x: (x**2).sum(dim=-1)

        ct_np = np.ones(3, dtype=np.float32)

        _, vjp_l = func.vjp(f_l, lucid.tensor(x_np.copy()))
        _, vjp_r = ref.func.vjp(f_r, ref.tensor(x_np.copy()))

        (g_l,) = vjp_l(lucid.tensor(ct_np.copy()))
        (g_r,) = vjp_r(ref.tensor(ct_np.copy()))
        assert_close(g_l, g_r, atol=1e-5)


# ── jvp ──────────────────────────────────────────────────────────────────────


@pytest.mark.parity
class TestJVPParity:
    def test_scalar_output(self, ref: Any) -> None:
        """jvp of scalar fn matches reference."""
        x_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        v_np = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        f_l = lambda x: (x**2).sum()
        f_r = lambda x: (x**2).sum()

        pout_l, jvp_l = func.jvp(f_l, (lucid.tensor(x_np.copy()),), (lucid.tensor(v_np.copy()),))
        pout_r, jvp_r = ref.func.jvp(f_r, (ref.tensor(x_np.copy()),), (ref.tensor(v_np.copy()),))

        assert_close(pout_l, pout_r, atol=1e-4)
        assert_close(jvp_l, jvp_r, atol=1e-4)

    def test_vector_output(self, ref: Any) -> None:
        """jvp of vector fn (elementwise scale) matches reference."""
        x_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        v_np = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        scales_np = np.array([2.0, 3.0, 4.0], dtype=np.float32)

        scales_l = lucid.tensor(scales_np.copy())
        scales_r = ref.tensor(scales_np.copy())

        f_l = lambda x: x * scales_l
        f_r = lambda x: x * scales_r

        _, jvp_l = func.jvp(f_l, (lucid.tensor(x_np.copy()),), (lucid.tensor(v_np.copy()),))
        _, jvp_r = ref.func.jvp(f_r, (ref.tensor(x_np.copy()),), (ref.tensor(v_np.copy()),))
        assert_close(jvp_l, jvp_r, atol=1e-4)

    def test_tangent_direction(self, ref: Any) -> None:
        """JVP with one-hot tangent gives single column of Jacobian."""
        x_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        # one-hot tangent for dim 1
        v_np = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        scales_np = np.array([2.0, 3.0, 4.0], dtype=np.float32)

        scales_l = lucid.tensor(scales_np.copy())
        scales_r = ref.tensor(scales_np.copy())

        f_l = lambda x: x * scales_l
        f_r = lambda x: x * scales_r

        _, jvp_l = func.jvp(f_l, (lucid.tensor(x_np.copy()),), (lucid.tensor(v_np.copy()),))
        _, jvp_r = ref.func.jvp(f_r, (ref.tensor(x_np.copy()),), (ref.tensor(v_np.copy()),))
        # JVP[i] = scales[i] * v[i]; only dim 1 is non-zero
        assert_close(jvp_l, jvp_r, atol=1e-4)


# ── jacrev ───────────────────────────────────────────────────────────────────


@pytest.mark.parity
class TestJacRevParity:
    def test_identity(self, ref: Any) -> None:
        """Jacobian of identity is I."""
        x_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        f_l = lambda x: x
        f_r = lambda x: x

        J_l = func.jacrev(f_l)(lucid.tensor(x_np.copy()))
        J_r = ref.func.jacrev(f_r)(ref.tensor(x_np.copy()))
        assert list(J_l.shape) == list(J_r.shape)
        assert_close(J_l, J_r, atol=1e-5)

    def test_diagonal_scale(self, ref: Any) -> None:
        """Jacobian of x * scales = diag(scales)."""
        x_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        s_np = np.array([2.0, 3.0, 4.0], dtype=np.float32)

        scales_l = lucid.tensor(s_np.copy())
        scales_r = ref.tensor(s_np.copy())

        f_l = lambda x: x * scales_l
        f_r = lambda x: x * scales_r

        J_l = func.jacrev(f_l)(lucid.tensor(x_np.copy()))
        J_r = ref.func.jacrev(f_r)(ref.tensor(x_np.copy()))
        assert_close(J_l, J_r, atol=1e-5)

    def test_scalar_output_matches_grad(self, ref: Any) -> None:
        """jacrev of scalar fn agrees with grad."""
        x_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        f_l = lambda x: (x**2).sum()
        f_r = lambda x: (x**2).sum()

        J_l = func.jacrev(f_l)(lucid.tensor(x_np.copy()))
        J_r = ref.func.jacrev(f_r)(ref.tensor(x_np.copy()))
        assert_close(J_l.reshape(-1), J_r.reshape(-1), atol=1e-5)

    def test_per_sample_grad_via_vmap(self, ref: Any) -> None:
        """vmap(grad(fn)) gives per-sample gradients matching reference.

        Note: vmap(grad(fn)) works correctly because grad produces scalar
        output per sample. vmap(jacrev(fn)) requires true vmap isolation
        (Stage 2) and is deferred.
        """
        rng = _rng(7)
        X_np = rng.standard_normal((4, 3)).astype(np.float32)

        f_l = lambda x: (x**2).sum()
        f_r = lambda x: (x**2).sum()

        # Per-sample gradients via vmap(grad): shape (4, 3)
        g_batch_l = func.vmap(func.grad(f_l))(lucid.tensor(X_np.copy()))
        g_batch_r = ref.func.vmap(ref.func.grad(f_r))(ref.tensor(X_np.copy()))
        assert_close(g_batch_l, g_batch_r, atol=1e-5)


# ── jacfwd ───────────────────────────────────────────────────────────────────


@pytest.mark.parity
class TestJacFwdParity:
    def test_identity(self, ref: Any) -> None:
        """Jacobian of identity via jacfwd matches jacrev."""
        x_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        f_l = lambda x: x
        f_r = lambda x: x

        J_fwd_l = func.jacfwd(f_l)(lucid.tensor(x_np.copy()))
        J_rev_l = func.jacrev(f_l)(lucid.tensor(x_np.copy()))
        J_fwd_r = ref.func.jacfwd(f_r)(ref.tensor(x_np.copy()))

        assert_close(J_fwd_l, J_rev_l, atol=1e-4)
        assert_close(J_fwd_l, J_fwd_r, atol=1e-4)

    def test_diagonal_agrees_with_jacrev(self, ref: Any) -> None:
        """jacfwd and jacrev agree on a diagonal-Jacobian function."""
        x_np = np.array([1.0, 2.0], dtype=np.float32)
        s_np = np.array([3.0, 5.0], dtype=np.float32)

        scales_l = lucid.tensor(s_np.copy())
        scales_r = ref.tensor(s_np.copy())

        f_l = lambda x: x * scales_l
        f_r = lambda x: x * scales_r

        J_fwd_l = func.jacfwd(f_l)(lucid.tensor(x_np.copy()))
        J_rev_l = func.jacrev(f_l)(lucid.tensor(x_np.copy()))
        J_fwd_r = ref.func.jacfwd(f_r)(ref.tensor(x_np.copy()))

        assert_close(J_fwd_l, J_rev_l, atol=1e-4)
        assert_close(J_fwd_l, J_fwd_r, atol=1e-4)


# ── hessian ───────────────────────────────────────────────────────────────────


@pytest.mark.parity
class TestHessianParity:
    def test_quadratic(self, ref: Any) -> None:
        """Hessian of sum(x²) = 2I, matching reference."""
        x_np = np.array([1.0, 2.0], dtype=np.float32)

        f_l = lambda x: (x**2).sum()
        f_r = lambda x: (x**2).sum()

        H_l = func.hessian(f_l)(lucid.tensor(x_np.copy()))
        H_r = ref.func.hessian(f_r)(ref.tensor(x_np.copy()))
        assert_close(H_l, H_r, atol=1e-3)

    def test_cubic_diagonal(self, ref: Any) -> None:
        """Hessian of sum(x³) = diag(6x), matching reference."""
        x_np = np.array([1.0, 2.0], dtype=np.float32)

        f_l = lambda x: (x**3).sum()
        f_r = lambda x: (x**3).sum()

        H_l = func.hessian(f_l)(lucid.tensor(x_np.copy()))
        H_r = ref.func.hessian(f_r)(ref.tensor(x_np.copy()))
        # H[i,i] = 6*x[i]; off-diagonal = 0
        assert_close(H_l, H_r, atol=1e-2)


# ── vmap(jacrev) / vmap(jacfwd) parity ───────────────────────────────────────


@pytest.mark.parity
class TestVmapJacrevParity:
    """vmap(jacrev(fn)) — Stage 2 isolation correctness vs reference framework."""

    def test_vector_output(self, ref: Any) -> None:
        """Per-batch Jacobian shape and values match reference."""
        rng = _rng(50)
        x_np = rng.standard_normal((4, 3)).astype(np.float32)

        f_l = lambda x: lucid.stack([x.sum(), (x**2).sum()])
        f_r = lambda x: ref.stack([x.sum(), (x**2).sum()])

        J_l = func.vmap(func.jacrev(f_l))(lucid.tensor(x_np.copy()))
        J_r = ref.func.vmap(ref.func.jacrev(f_r))(ref.tensor(x_np.copy()))
        assert_close(J_l, J_r, atol=1e-4)

    def test_scalar_fn_matches_grad(self, ref: Any) -> None:
        """vmap(jacrev(f)) == vmap(grad(f)) for scalar-output f."""
        rng = _rng(51)
        x_np = rng.standard_normal((5, 4)).astype(np.float32)

        f_l = lambda x: (x**2).sum()
        f_r = lambda x: (x**2).sum()

        J_l = func.vmap(func.jacrev(f_l))(lucid.tensor(x_np.copy()))
        g_l = func.vmap(func.grad(f_l))(lucid.tensor(x_np.copy()))
        J_r = ref.func.vmap(ref.func.jacrev(f_r))(ref.tensor(x_np.copy()))

        assert_close(J_l, J_r, atol=1e-4)
        assert_close(J_l, g_l, atol=1e-5)

    def test_non_trivial_fn(self, ref: Any) -> None:
        """Jacobian of a mixed-feature function vs reference."""
        rng = _rng(52)
        x_np = rng.standard_normal((3, 4)).astype(np.float32)

        f_l = lambda x: lucid.stack(
            [
                x[0] * x[1],
                x[1] ** 2 + x[2],
                x[3] - x[0],
            ]
        )
        f_r = lambda x: ref.stack(
            [
                x[0] * x[1],
                x[1] ** 2 + x[2],
                x[3] - x[0],
            ]
        )

        J_l = func.vmap(func.jacrev(f_l))(lucid.tensor(x_np.copy()))
        J_r = ref.func.vmap(ref.func.jacrev(f_r))(ref.tensor(x_np.copy()))
        assert_close(J_l, J_r, atol=1e-4)


@pytest.mark.parity
class TestVmapJacfwdParity:
    """vmap(jacfwd(fn)) — forward-mode Jacobian, batched."""

    def test_matches_jacrev(self, ref: Any) -> None:
        """vmap(jacfwd(fn)) and vmap(jacrev(fn)) agree."""
        rng = _rng(53)
        x_np = rng.standard_normal((4, 3)).astype(np.float32)

        f_l = lambda x: lucid.stack([x.sum(), (x**2).sum()])
        f_r = lambda x: ref.stack([x.sum(), (x**2).sum()])

        Jrev = func.vmap(func.jacrev(f_l))(lucid.tensor(x_np.copy()))
        Jfwd = func.vmap(func.jacfwd(f_l))(lucid.tensor(x_np.copy()))
        J_r = ref.func.vmap(ref.func.jacrev(f_r))(ref.tensor(x_np.copy()))

        assert_close(Jfwd, J_r, atol=1e-4)
        assert_close(Jrev, Jfwd, atol=1e-4)


@pytest.mark.parity
class TestVmapHessianParity:
    """vmap(hessian(fn)) — batched second-order derivatives."""

    def test_quadratic(self, ref: Any) -> None:
        """Hessian of x² is 2I for each batch element."""
        rng = _rng(54)
        x_np = rng.standard_normal((3, 4)).astype(np.float32)

        f_l = lambda x: (x**2).sum()
        f_r = lambda x: (x**2).sum()

        H_l = func.vmap(func.hessian(f_l))(lucid.tensor(x_np.copy()))
        H_r = ref.func.vmap(ref.func.hessian(f_r))(ref.tensor(x_np.copy()))
        assert_close(H_l, H_r, atol=1e-3)
