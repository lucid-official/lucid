"""Regression tests for LCD-207, LCD-335, LCD-355, LCD-395, LCD-455, and LCD-456."""

import inspect
import math
from types import ModuleType
from typing import override

import mpmath
import numpy as np
import pytest

import lucid
import lucid.func as func
from lucid._tensor.tensor import Tensor
from lucid.distributions import (
    Bernoulli,
    Beta,
    Chi2,
    Dirichlet,
    Exponential,
    Gamma,
    Normal,
)
from lucid.distributions.distribution import ExponentialFamily
from lucid.special import gammainc, gammaincc, modified_bessel_i0
from lucid.test._fixtures.devices import metal_available

_DEVICES = [
    "cpu",
    pytest.param(
        "metal",
        marks=pytest.mark.skipif(not metal_available(), reason="needs Metal"),
    ),
]


# ── LCD-207: lucid.vdot complex conjugation and Wirtinger backward ──────────


@pytest.mark.parity
@pytest.mark.parametrize("device", _DEVICES)
def test_vdot_complex64_forward_and_backward_parity(ref: ModuleType, device: str) -> None:
    a_np = np.array([1.0 + 2.0j, -3.0 + 0.5j, 2.5 - 1.5j], dtype=np.complex64)
    b_np = np.array([-0.5 + 1.0j, 4.0 - 2.0j, 1.5 + 3.0j], dtype=np.complex64)

    a_l = lucid.tensor(a_np, device=device, requires_grad=True)
    b_l = lucid.tensor(b_np, device=device, requires_grad=True)
    a_t = ref.tensor(a_np, requires_grad=True)
    b_t = ref.tensor(b_np, requires_grad=True)

    out_l = lucid.vdot(a_l, b_l)
    out_t = ref.vdot(a_t, b_t)
    np.testing.assert_allclose(out_l.cpu().numpy(), out_t.detach().numpy(), rtol=1e-5)

    # Self vdot <a, a> must be real and positive.
    aa_l = lucid.vdot(a_l, a_l)
    assert abs(float(lucid.imag(aa_l).cpu().item())) < 1e-6
    assert float(lucid.real(aa_l).cpu().item()) > 0.0

    # Check Wirtinger gradients under a complex weight w = 2.0 - 1.5j.
    w_np = np.array(2.0 - 1.5j, dtype=np.complex64)
    loss_l = lucid.real(out_l * lucid.tensor(w_np, device=device))
    loss_t = (out_t * ref.tensor(w_np)).real
    loss_l.backward()
    loss_t.backward()

    assert a_l.grad is not None and b_l.grad is not None
    assert a_t.grad is not None and b_t.grad is not None
    np.testing.assert_allclose(a_l.grad.cpu().numpy(), a_t.grad.numpy(), rtol=1e-5)
    np.testing.assert_allclose(b_l.grad.cpu().numpy(), b_t.grad.numpy(), rtol=1e-5)


@pytest.mark.parity
def test_vdot_complex128_cpu_parity(ref: ModuleType) -> None:
    a_np = np.array([1.25 - 3.5j, -2.0 + 4.75j], dtype=np.complex128)
    b_np = np.array([0.75 + 1.25j, -1.5 - 2.25j], dtype=np.complex128)

    a_l = lucid.tensor(a_np, dtype=lucid.complex128, requires_grad=True)
    b_l = lucid.tensor(b_np, dtype=lucid.complex128, requires_grad=True)
    a_t = ref.tensor(a_np, dtype=ref.complex128, requires_grad=True)
    b_t = ref.tensor(b_np, dtype=ref.complex128, requires_grad=True)

    out_l = lucid.vdot(a_l, b_l)
    out_t = ref.vdot(a_t, b_t)
    np.testing.assert_allclose(out_l.numpy(), out_t.detach().numpy(), rtol=1e-12)

    lucid.imag(out_l).backward()
    out_t.imag.backward()
    assert a_l.grad is not None and b_l.grad is not None
    assert a_t.grad is not None and b_t.grad is not None
    np.testing.assert_allclose(a_l.grad.numpy(), a_t.grad.numpy(), rtol=1e-12)
    np.testing.assert_allclose(b_l.grad.numpy(), b_t.grad.numpy(), rtol=1e-12)


# ── LCD-335 & LCD-355: func.jvp unification, strict=True, and transforms ────


def test_func_jvp_delegates_to_autograd_functional_jvp() -> None:
    from lucid.autograd._functional import jvp as af_jvp

    assert func._jvp_by_double_backward is af_jvp

    # Steep scale (1e6) that would trip a heuristic 1e-3 relative check.
    x = lucid.tensor([1.0, -2.0, 3.0])
    v = lucid.tensor([0.5, -1.0, 2.0])
    y, dy = func.jvp(lambda z: 1e6 * (z * z), (x,), (v,))
    np.testing.assert_allclose(y.numpy(), (1e6 * x * x).numpy(), rtol=1e-6)
    np.testing.assert_allclose(dy.numpy(), (2e6 * x * v).numpy(), rtol=1e-6)


def test_func_transforms_shared_helpers_and_validation() -> None:
    def f(x: Tensor, y: Tensor, scale: float) -> tuple[Tensor, Tensor]:
        loss = ((x * y) * scale).sum()
        return loss, x + y

    x = lucid.tensor([1.0, 2.0, 3.0])
    y = lucid.tensor([4.0, 5.0, 6.0])

    (gx, gy), aux = func.grad(f, argnums=(0, 1), has_aux=True)(x, y, 2.0)
    (gx2, gy2), (val, aux2) = func.grad_and_value(f, argnums=(0, 1), has_aux=True)(x, y, 2.0)
    np.testing.assert_allclose(gx.numpy(), (y * 2.0).numpy())
    np.testing.assert_allclose(gy.numpy(), (x * 2.0).numpy())
    np.testing.assert_allclose(gx2.numpy(), gx.numpy())
    np.testing.assert_allclose(gy2.numpy(), gy.numpy())
    np.testing.assert_allclose(aux.numpy(), (x + y).numpy())
    np.testing.assert_allclose(aux2.numpy(), (x + y).numpy())
    assert float(val.item()) == pytest.approx(64.0)

    def vec_fn(a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
        return lucid.stack([(a * b).sum(), (a + b).sum()]), a - b

    (jrev_a, jrev_b), aux_r = func.jacrev(vec_fn, argnums=(0, 1), has_aux=True)(x, y)
    (jfwd_a, jfwd_b), aux_f = func.jacfwd(vec_fn, argnums=(0, 1), has_aux=True)(x, y)
    np.testing.assert_allclose(jrev_a.numpy(), jfwd_a.numpy(), rtol=1e-5)
    np.testing.assert_allclose(jrev_b.numpy(), jfwd_b.numpy(), rtol=1e-5)
    np.testing.assert_allclose(aux_r.numpy(), aux_f.numpy())

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        func.jacrev(lambda z: z, chunk_size=0)


# ── LCD-395: ldl_factor keyword-only hermitian and cholesky_ex partial L ────


@pytest.mark.parity
def test_ldl_factor_hermitian_is_keyword_only_and_cholesky_ex_partial(
    ref: ModuleType,
) -> None:
    for fn in (lucid.linalg.ldl_factor, lucid.linalg.ldl_factor_ex):
        sig = inspect.signature(fn)
        param = sig.parameters["hermitian"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
        assert param.default is False

    a = lucid.tensor([[4.0, 2.0], [2.0, 3.0]])
    with pytest.raises(TypeError):
        lucid.linalg.ldl_factor(a, False)  # type: ignore[misc]
    with pytest.raises(TypeError):
        lucid.linalg.ldl_factor_ex(a, False)  # type: ignore[misc]

    # Fails at minor k = 3: leading 2x2 Cholesky factor must match the reference.
    bad_np = np.array([[4.0, 2.0, 0.0], [2.0, 5.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32)
    l_res, l_info = lucid.linalg.cholesky_ex(lucid.tensor(bad_np))
    t_res, t_info = ref.linalg.cholesky_ex(ref.tensor(bad_np))
    assert int(l_info.item()) == int(t_info.item()) == 3
    np.testing.assert_allclose(l_res[:2, :2].numpy(), t_res[:2, :2].numpy(), rtol=1e-6)
    np.testing.assert_allclose(l_res[2:, :].numpy(), 0.0)


# ── LCD-455: gammainc/gammaincc at large a outside 4.5*sqrt(a) band ─────────


@pytest.mark.parametrize("a_val", [1e7, 1e8])
@pytest.mark.parametrize("k_std", [-10.0, -6.0, -4.6, -4.5, 4.5, 4.6, 6.0])
def test_gammainc_large_a_matches_mpmath_outside_temme_band(a_val: float, k_std: float) -> None:
    mpmath.mp.dps = 50
    x_val = a_val + k_std * math.sqrt(a_val)
    a_t = lucid.tensor([a_val], dtype=lucid.float64)
    x_t = lucid.tensor([x_val], dtype=lucid.float64)

    got_p = float(gammainc(a_t, x_t).item())
    got_q = float(gammaincc(a_t, x_t).item())

    want_q = float(mpmath.gammainc(a_val, x_val, mpmath.inf, regularized=True))
    want_p = float(1 - mpmath.gammainc(a_val, x_val, mpmath.inf, regularized=True))

    assert math.isclose(got_p, want_p, rel_tol=1e-11, abs_tol=0.0)
    assert math.isclose(got_q, want_q, rel_tol=1e-11, abs_tol=0.0)


# ── LCD-456: ExponentialFamily.entropy and modified_bessel_i0 in float16 ─────


def test_exponential_family_bregman_entropy_matches_closed_form() -> None:
    families: list[ExponentialFamily] = [
        Normal(loc=lucid.tensor([0.5, -1.0]), scale=lucid.tensor([1.2, 2.5])),
        Bernoulli(probs=lucid.tensor([0.2, 0.75])),
        Exponential(rate=lucid.tensor([0.5, 3.0])),
        Gamma(concentration=lucid.tensor([2.5, 4.0]), rate=lucid.tensor([1.5, 0.8])),
        Chi2(df=lucid.tensor([3.0, 6.0])),
        Beta(
            concentration1=lucid.tensor([2.0, 0.7]),
            concentration0=lucid.tensor([3.5, 1.8]),
        ),
        Dirichlet(lucid.tensor([[1.5, 2.5, 3.5], [0.8, 1.2, 2.0]])),
    ]
    for dist in families:
        closed = dist.entropy()
        bregman = ExponentialFamily.entropy(dist)
        np.testing.assert_allclose(
            bregman.detach().numpy(),
            closed.detach().numpy(),
            rtol=1e-5,
            err_msg=f"Mismatch for {type(dist).__name__}",
        )

    # Custom ExponentialFamily subclass implementing only _natural_params and _log_normalizer.
    class _CustomExp(ExponentialFamily):
        def __init__(self, rate: Tensor) -> None:
            self.rate = rate
            super().__init__(batch_shape=tuple(rate.shape))

        @override
        @property
        def _natural_params(self) -> tuple[Tensor, ...]:
            return (-self.rate,)

        @override
        def _log_normalizer(self, *natural_params: Tensor) -> Tensor:
            (x,) = natural_params
            return -(-x).log()

    rate = lucid.tensor([1.0, 2.0, 4.0])
    custom = _CustomExp(rate)
    np.testing.assert_allclose(
        custom.entropy().detach().numpy(),
        (1.0 - rate.log()).numpy(),
        rtol=1e-5,
    )


@pytest.mark.parity
@pytest.mark.parametrize("device", _DEVICES)
def test_modified_bessel_i0_float16_no_overflow(ref: ModuleType, device: str) -> None:
    vals = [0.3, 1.5, 7.5, 11.5, 12.0]
    x_l = lucid.tensor(vals, dtype=lucid.float16, device=device)
    x_t = ref.tensor(vals, dtype=ref.float16).float()
    out_l = modified_bessel_i0(x_l)
    out_t = ref.special.modified_bessel_i0(x_t).to(ref.float16)
    assert out_l.dtype == lucid.float16
    np.testing.assert_allclose(
        out_l.cpu().to(lucid.float32).numpy(),
        out_t.float().numpy(),
        rtol=2e-3,
        atol=1e-3,
    )
