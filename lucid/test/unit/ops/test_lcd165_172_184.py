"""Regression and reference-parity tests for LCD-165, LCD-172, and LCD-184.

* **LCD-165**: ``bincount`` accepts ``int8`` / ``int16`` inputs (promoting the
  scatter index to ``int64``) and preserves ``weights`` dtype on CPU and Metal.
* **LCD-184**: ``addmm``, ``addmv``, ``baddbmm``, ``addbmm``, and ``addr`` skip
  multiplying ``input`` by ``beta`` when ``beta == 0`` so ``NaN`` / ``Inf`` in
  an unused accumulator never poisons the output or gradients.
* **LCD-172**: ``linalg.cholesky`` backward returns the symmetrised Murray
  gradient ``0.5 * (Z + Z.mT)`` matching the reference framework's, and
  scalar-constructed distributions (including ``LKJCholesky``) score Metal
  tensors without ``DeviceMismatch``.
"""

import math
from types import ModuleType

import numpy as np
import pytest

import lucid
import lucid.distributions as D
import lucid.linalg as LA
from lucid.test._helpers.compare import assert_close

# ── LCD-165: bincount integer dtypes & weighted output dtype ────────────────


@pytest.mark.parity
@pytest.mark.parametrize("index_dtype", ["int8", "int16", "int32", "int64"])
@pytest.mark.parametrize("weighted", [False, True])
def test_bincount_all_signed_integer_dtypes(
    ref: ModuleType, device: str, index_dtype: str, weighted: bool
) -> None:
    vals = [0, 2, 2, 5, 1, 5, 5]
    x_ref = ref.tensor(vals, dtype=getattr(ref, index_dtype))
    x_luc = lucid.tensor(vals, dtype=getattr(lucid, index_dtype), device=device)
    if weighted:
        w = [0.5, 1.25, 2.0, 3.5, 4.0, 0.75, 1.0]
        w_ref = ref.tensor(w, dtype=ref.float32)
        w_luc = lucid.tensor(w, dtype=lucid.float32, device=device)
        want = ref.bincount(x_ref, weights=w_ref, minlength=8)
        got = lucid.bincount(x_luc, weights=w_luc, minlength=8)
        assert got.dtype == lucid.float32
        assert got.device.type == device
        assert_close(got, want.numpy(), rtol=1e-6, atol=1e-6)
    else:
        want = ref.bincount(x_ref, minlength=8)
        got = lucid.bincount(x_luc, minlength=8)
        assert got.dtype == lucid.int64
        assert got.device.type == device
        assert got.tolist() == want.tolist()


# ── LCD-184: BLAS beta=0 ignores NaN/Inf in accumulator ─────────────────────


@pytest.mark.parity
@pytest.mark.parametrize("fill", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("alpha", [1.0, 2.5, -0.5])
def test_addmm_beta_zero_ignores_nonfinite_input(
    ref: ModuleType, device: str, fill: float, alpha: float
) -> None:
    m1_np = np.array([[1.0, 2.0, -1.0], [3.0, 0.5, 2.0]], dtype=np.float32)
    m2_np = np.array([[2.0, -1.0], [0.5, 3.0], [1.5, 2.0]], dtype=np.float32)
    inp_ref = ref.full((2, 2), fill, dtype=ref.float32)
    inp_luc = lucid.full((2, 2), fill, dtype=lucid.float32, device=device)
    want = ref.addmm(inp_ref, ref.tensor(m1_np), ref.tensor(m2_np), beta=0.0, alpha=alpha)
    got = lucid.addmm(
        inp_luc,
        lucid.tensor(m1_np, device=device),
        lucid.tensor(m2_np, device=device),
        beta=0.0,
        alpha=alpha,
    )
    assert got.device.type == device
    assert bool(lucid.isfinite(got).all().item())
    assert_close(got, want.numpy(), rtol=1e-5, atol=1e-6)


@pytest.mark.parity
@pytest.mark.parametrize("fill", [math.nan, math.inf, -math.inf])
def test_addmv_baddbmm_addbmm_addr_beta_zero_ignore_nonfinite_input(
    ref: ModuleType, device: str, fill: float
) -> None:
    # addmv
    mat_np = np.array([[1.0, 2.0, 3.0], [4.0, -1.0, 0.5]], dtype=np.float32)
    vec_np = np.array([2.0, -0.5, 1.5], dtype=np.float32)
    want_mv = ref.addmv(
        ref.full((2,), fill),
        ref.tensor(mat_np),
        ref.tensor(vec_np),
        beta=0,
        alpha=1.5,
    )
    got_mv = lucid.addmv(
        lucid.full((2,), fill, device=device),
        lucid.tensor(mat_np, device=device),
        lucid.tensor(vec_np, device=device),
        beta=0,
        alpha=1.5,
    )
    assert_close(got_mv, want_mv.numpy(), rtol=1e-5, atol=1e-6)

    # baddbmm & addbmm
    b1_np = np.arange(12, dtype=np.float32).reshape(2, 2, 3) * 0.25
    b2_np = np.arange(12, dtype=np.float32).reshape(2, 3, 2) * 0.5
    want_badd = ref.baddbmm(
        ref.full((2, 2, 2), fill),
        ref.tensor(b1_np),
        ref.tensor(b2_np),
        beta=0.0,
        alpha=2.0,
    )
    got_badd = lucid.baddbmm(
        lucid.full((2, 2, 2), fill, device=device),
        lucid.tensor(b1_np, device=device),
        lucid.tensor(b2_np, device=device),
        beta=0.0,
        alpha=2.0,
    )
    assert_close(got_badd, want_badd.numpy(), rtol=1e-5, atol=1e-6)

    want_addb = ref.addbmm(
        ref.full((2, 2), fill),
        ref.tensor(b1_np),
        ref.tensor(b2_np),
        beta=0.0,
        alpha=2.0,
    )
    got_addb = lucid.addbmm(
        lucid.full((2, 2), fill, device=device),
        lucid.tensor(b1_np, device=device),
        lucid.tensor(b2_np, device=device),
        beta=0.0,
        alpha=2.0,
    )
    assert_close(got_addb, want_addb.numpy(), rtol=1e-5, atol=1e-6)

    # addr
    v1_np = np.array([1.5, -2.0], dtype=np.float32)
    v2_np = np.array([3.0, 0.5, -1.0], dtype=np.float32)
    want_addr = ref.addr(
        ref.full((2, 3), fill),
        ref.tensor(v1_np),
        ref.tensor(v2_np),
        beta=0.0,
        alpha=0.5,
    )
    got_addr = lucid.addr(
        lucid.full((2, 3), fill, device=device),
        lucid.tensor(v1_np, device=device),
        lucid.tensor(v2_np, device=device),
        beta=0.0,
        alpha=0.5,
    )
    assert_close(got_addr, want_addr.numpy(), rtol=1e-5, atol=1e-6)


def test_addmm_beta_zero_backward_with_nan_input(device: str) -> None:
    inp = lucid.full((2, 2), math.nan, device=device, requires_grad=True)
    m1 = lucid.tensor([[1.0, 2.0], [3.0, 4.0]], device=device, requires_grad=True)
    m2 = lucid.tensor([[0.5, -1.0], [1.5, 2.0]], device=device, requires_grad=True)
    out = lucid.addmm(inp, m1, m2, beta=0.0, alpha=2.0)
    out.sum().backward()
    assert inp.grad is not None
    assert m1.grad is not None
    assert m2.grad is not None
    assert_close(inp.grad, np.zeros((2, 2), dtype=np.float32))
    assert bool(lucid.isfinite(m1.grad).all().item())
    assert bool(lucid.isfinite(m2.grad).all().item())


# ── LCD-172: linalg.cholesky symmetric backward ─────────────────────────────


@pytest.mark.parity
@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("batched", [False, True])
def test_cholesky_backward_matches_reference_symmetry(
    ref: ModuleType, device: str, upper: bool, batched: bool
) -> None:
    rng = np.random.default_rng(42)
    shape = (2, 4, 4) if batched else (4, 4)
    raw = rng.standard_normal(shape).astype(np.float32)
    spd = np.matmul(raw, np.swapaxes(raw, -1, -2)) + 4.0 * np.eye(4, dtype=np.float32)
    w_np = rng.standard_normal(shape).astype(np.float32)

    a_ref = ref.tensor(spd, requires_grad=True)
    f_ref = ref.linalg.cholesky(a_ref, upper=upper)
    (f_ref * ref.tensor(w_np)).sum().backward()

    a_luc = lucid.tensor(spd, device=device, requires_grad=True)
    f_luc = LA.cholesky(a_luc, upper=upper)
    (f_luc * lucid.tensor(w_np, device=device)).sum().backward()

    assert a_luc.grad is not None
    assert a_ref.grad is not None
    # Gradient is symmetric: grad == grad.mT
    assert_close(a_luc.grad, a_luc.grad.mT.numpy(), rtol=1e-5, atol=1e-6)
    assert_close(a_luc.grad, a_ref.grad.numpy(), rtol=1e-4, atol=1e-5)


# ── LCD-172: scalar-constructed distributions scoring on device ─────────────


@pytest.mark.parity
def test_scalar_constructed_distributions_score_on_device(ref: ModuleType, device: str) -> None:
    x_pos_np = np.array([0.4, 1.2, 2.5], dtype=np.float32)
    x_unit_np = np.array([0.2, 0.5, 0.8], dtype=np.float32)
    x_int_np = np.array([0.0, 1.0, 3.0], dtype=np.float32)
    x_bern_np = np.array([0.0, 1.0, 1.0], dtype=np.float32)

    x_pos = lucid.tensor(x_pos_np, device=device)
    x_unit = lucid.tensor(x_unit_np, device=device)
    x_int = lucid.tensor(x_int_np, device=device)
    x_bern = lucid.tensor(x_bern_np, device=device)

    rx_pos = ref.tensor(x_pos_np)
    rx_unit = ref.tensor(x_unit_np)
    rx_int = ref.tensor(x_int_np)
    rx_bern = ref.tensor(x_bern_np)

    RD = ref.distributions
    cases = [
        (D.Normal(0.0, 1.0), RD.Normal(0.0, 1.0), x_pos, rx_pos, True),
        (D.Uniform(0.0, 3.0), RD.Uniform(0.0, 3.0), x_pos, rx_pos, True),
        (D.Exponential(1.5), RD.Exponential(1.5), x_pos, rx_pos, True),
        (D.Bernoulli(probs=0.3), RD.Bernoulli(probs=0.3), x_bern, rx_bern, False),
        (D.Gamma(2.0, 1.5), RD.Gamma(2.0, 1.5), x_pos, rx_pos, False),
        (D.Beta(2.0, 3.0), RD.Beta(2.0, 3.0), x_unit, rx_unit, False),
        (D.Cauchy(0.0, 1.0), RD.Cauchy(0.0, 1.0), x_pos, rx_pos, True),
        (D.Laplace(0.0, 1.0), RD.Laplace(0.0, 1.0), x_pos, rx_pos, True),
        (D.StudentT(3.0, 0.0, 1.0), RD.StudentT(3.0, 0.0, 1.0), x_pos, rx_pos, False),
        (D.Poisson(2.5), RD.Poisson(2.5), x_int, rx_int, False),
        (D.Geometric(probs=0.4), RD.Geometric(probs=0.4), x_int, rx_int, False),
        (D.LogNormal(0.0, 1.0), RD.LogNormal(0.0, 1.0), x_pos, rx_pos, True),
        (D.Weibull(1.5, 2.0), RD.Weibull(1.5, 2.0), x_pos, rx_pos, True),
        (D.Gumbel(0.0, 1.0), RD.Gumbel(0.0, 1.0), x_pos, rx_pos, True),
        (D.HalfNormal(1.5), RD.HalfNormal(1.5), x_pos, rx_pos, True),
        (D.HalfCauchy(1.5), RD.HalfCauchy(1.5), x_pos, rx_pos, True),
        (D.Pareto(1.0, 2.0), RD.Pareto(1.0, 2.0), x_pos + 1.0, rx_pos + 1.0, True),
        (D.Chi2(3.0), RD.Chi2(3.0), x_pos, rx_pos, False),
    ]
    for dist_luc, dist_ref, val_luc, val_ref, has_cdf_icdf in cases:
        lp = dist_luc.log_prob(val_luc)
        assert lp.device.type == device
        assert_close(lp, dist_ref.log_prob(val_ref).numpy(), rtol=1e-4, atol=1e-4)
        if has_cdf_icdf:
            c = dist_luc.cdf(val_luc)
            assert c.device.type == device
            assert_close(c, dist_ref.cdf(val_ref).numpy(), rtol=1e-4, atol=1e-4)
            ic = dist_luc.icdf(x_unit)
            assert ic.device.type == device
            assert_close(ic, dist_ref.icdf(rx_unit).numpy(), rtol=1e-4, atol=1e-4)


@pytest.mark.parity
def test_lkj_cholesky_scalar_concentration_scores_on_device(ref: ModuleType, device: str) -> None:
    dist_luc = D.LKJCholesky(3, 2.0)
    dist_ref = ref.distributions.LKJCholesky(3, 2.0)
    sample_ref = dist_ref.sample((2,))
    sample_luc = lucid.tensor(sample_ref.numpy(), device=device)
    lp_luc = dist_luc.log_prob(sample_luc)
    lp_ref = dist_ref.log_prob(sample_ref)
    assert lp_luc.device.type == device
    assert_close(lp_luc, lp_ref.numpy(), rtol=1e-4, atol=1e-4)
