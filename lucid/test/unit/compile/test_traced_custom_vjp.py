"""Tests for traced custom VJP regions (``Function._traced_backward = True``).

Covers:
1. Built-in ``_traced_backward = True`` ``Function`` subclasses:
   - ``_FakeQuantizeAffine`` (``Q.fake_quantize`` and ``Q.prepare_qat``)
   - ``_StandardGamma`` (``Gamma.rsample``, ``Beta.rsample``, ``Dirichlet.rsample``)
   - ``_LogAbsDet`` (``lucid.linalg.slogdet``)
   - ``_MatrixPowerZero`` (``lucid.linalg.matrix_power(A, 0)``)
2. User-defined ``Function`` subclasses with ``_traced_backward = True``:
   - Identity STE (``return grad_output``, 0 ops in ``bwd_ops``)
   - Scaled / reversed gradient (``return -grad_output * 5.0``)
   - Saved-tensor backward (``ctx.save_for_backward``)
   - Multi-output with ``ctx.mark_non_differentiable`` and unused output
   - Broadcast input gradient reduction (``_validate`` / ``_sum_to``)
3. Entry points: ``make_step``, ``make_step(..., segments=2)``, ``fused_step``,
   ``compiled_step``, and ``diagnose``.
"""

import warnings
from collections.abc import Callable

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
import lucid.quantization as Q
from lucid.autograd import Function
from lucid.compile import diagnose, fused_step, make_step
from lucid.compile._core.fallback import CompileFallbackWarning
from lucid.compile._entry.function import compiled_step
from lucid.distributions import (
    Beta,
    Chi2,
    Dirichlet,
    FisherSnedecor,
    Gamma,
    InverseGamma,
    LowRankMultivariateNormal,
    MultivariateNormal,
    StudentT,
    Wishart,
)
from lucid.models.text.roformer import RoFormerConfig, RoFormerForSequenceClassification


class _TracedSTE(Function):
    _traced_backward = True

    @staticmethod
    def forward(ctx, h):
        del ctx
        return h.round()

    @staticmethod
    def backward(ctx, g):
        del ctx
        return g


class _TracedScaledReversal(Function):
    _traced_backward = True

    @staticmethod
    def forward(ctx, h):
        ctx.save_for_backward(h)
        return h * 2.0

    @staticmethod
    def backward(ctx, g):
        (h,) = ctx.saved_tensors
        return -g * (h.abs() + 1.5)


class _TracedWithMaskAndUnused(Function):
    _traced_backward = True

    @staticmethod
    def forward(ctx, h, bias):
        mask = (h > 0).to(h.dtype)
        ctx.save_for_backward(mask)
        ctx.mark_non_differentiable(mask)
        return h + bias, mask, h * 3.0

    @staticmethod
    def backward(ctx, g_main, g_mask, g_unused):
        del g_mask
        (mask,) = ctx.saved_tensors
        grad_h = g_main * mask * 2.0 + g_unused * 7.0
        grad_bias = g_main * 0.5
        return grad_h, grad_bias


class _CustomVjpModule(nn.Module):
    def __init__(self, mode: str) -> None:
        super().__init__()
        self.mode = mode
        self.fc1 = nn.Linear(4, 4)
        self.bias = nn.Parameter(lucid.ones(1, 4) * 0.25)
        self.fc2 = nn.Linear(4, 3)

    def _apply_linalg_mode(self, h: lucid.Tensor) -> lucid.Tensor:
        dt, dev = h.dtype, h.device
        if self.mode == "slogdet":
            mat = h[:2, :2] * 0.25 + lucid.eye(2, dtype=dt, device=dev) * 2.0
            sign, logabsdet = lucid.linalg.slogdet(mat)
            return h + sign * logabsdet + self.bias
        if self.mode == "matrix_power_zero":
            mat = h.unsqueeze(-1) * h.unsqueeze(-2) + lucid.eye(4, dtype=dt, device=dev)
            p0 = lucid.linalg.matrix_power(mat, 0)
            return (h + self.bias) @ p0[0]
        if self.mode == "cholesky_solve":
            spd = (h.mT @ h) * 0.15 + lucid.eye(4, dtype=dt, device=dev) * 2.0
            factor = lucid.linalg.cholesky(spd)
            sol = lucid.linalg.solve_triangular(factor, (h + self.bias).mT, upper=False).mT
            return sol @ factor
        if self.mode == "qr":
            mat = (h.mT @ h[:, :3]) * 0.15 + lucid.eye(4, 3, dtype=dt, device=dev) * 1.5
            q_mat, r_mat = lucid.linalg.qr(mat)
            recon = q_mat @ (r_mat * 1.25)
            return (h + self.bias) + F.pad((h + self.bias) @ recon, (0, 1))
        if self.mode == "svd":
            shift = lucid.diag_embed(lucid.tensor([4.5, 3.2, 2.0, 1.0], dtype=dt, device=dev))
            mat = (h.mT @ h) * 0.1 + shift
            u_mat, s_vec, vh_mat = lucid.linalg.svd(mat, full_matrices=False)
            recon = (u_mat * (s_vec * s_vec).unsqueeze(-2)) @ vh_mat
            return (h + self.bias) @ (recon * 0.1)
        if self.mode == "eigh":
            shift = lucid.diag_embed(lucid.tensor([1.0, 2.2, 3.6, 5.2], dtype=dt, device=dev))
            gram = h.mT @ h
            sym = (gram + gram.mT) * 0.08 + shift
            w_vec, v_mat = lucid.linalg.eigh(sym)
            recon = (v_mat * (w_vec * w_vec).unsqueeze(-2)) @ v_mat.mT
            return (h + self.bias) @ (recon * 0.08)
        if self.mode == "pinv":
            mat = (h.mT @ h[:, :3]) * 0.15 + lucid.eye(4, 3, dtype=dt, device=dev) * 2.0
            p_mat = lucid.linalg.pinv(mat)
            return (h[:, :3] @ p_mat) + self.bias
        raise ValueError(f"Unknown mode: {self.mode}")

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        h = self.fc1(x)
        if self.mode == "ste":
            h = _TracedSTE.apply(h) + self.bias
        elif self.mode == "scaled_reversal":
            h = _TracedScaledReversal.apply(h) + self.bias
        elif self.mode == "mask_and_unused":
            y, mask, _unused = _TracedWithMaskAndUnused.apply(h, self.bias)
            h = y + mask
        elif self.mode == "fake_quantize":
            w = Q.fake_quantize(self.fc2.weight, 0.05, 0, -128, 127)
            return F.linear(h + self.bias, w, self.fc2.bias)
        else:
            h = self._apply_linalg_mode(h)
        return self.fc2(h)


def _grads(model: nn.Module) -> dict[str, list | None]:
    return {n: None if p.grad is None else p.grad.tolist() for n, p in model.named_parameters()}


def _assert_close_dicts(got: dict[str, list | None], want: dict[str, list | None]) -> None:
    assert got.keys() == want.keys()
    for name in want:
        a, b = got[name], want[name]
        if a is None or b is None:
            assert a is b, name
            continue
        ta = lucid.tensor(a).reshape(-1)
        tb = lucid.tensor(b).reshape(-1)
        assert bool(lucid.allclose(ta, tb, rtol=1e-4, atol=1e-5)), f"Mismatch on {name}"


@pytest.mark.parametrize(
    "mode",
    [
        "ste",
        "scaled_reversal",
        "mask_and_unused",
        "fake_quantize",
        "slogdet",
        "matrix_power_zero",
        "cholesky_solve",
        "qr",
        "svd",
        "eigh",
        "pinv",
    ],
)
def test_traced_custom_vjp_compiles_across_entry_points(mode: str, device_gpu_only: str) -> None:
    device = device_gpu_only
    lucid.manual_seed(42)
    x = lucid.randn(5, 4).to(device)
    t = lucid.randn(5, 3).to(device)

    # 1. Eager reference
    lucid.manual_seed(7)
    eager = _CustomVjpModule(mode).to(device)
    eager_loss = F.mse_loss(eager(x), t)
    eager_loss.backward()
    want_grads = _grads(eager)
    optim.SGD(eager.parameters(), lr=0.25).step()
    want_params = {n: p.tolist() for n, p in eager.named_parameters()}

    # Verify STE / fake_quantize actually produced non-zero fc1/fc2 gradients
    if mode in ("ste", "fake_quantize"):
        fc1_grad = lucid.tensor(want_grads["fc1.weight"]).abs().sum().item()
        assert fc1_grad > 0.0

    # 2. make_step
    lucid.manual_seed(7)
    m_step = _CustomVjpModule(mode).to(device)
    step = make_step(m_step, F.mse_loss)
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompileFallbackWarning)
        loss_step = step(x, t)
        loss_step.backward()
    assert len(step.cache) == 1
    assert not step.eager_only
    _assert_close_dicts(_grads(m_step), want_grads)

    # 3. compiled_step
    lucid.manual_seed(7)
    m_cstep = _CustomVjpModule(mode).to(device)
    loss_cstep = compiled_step(m_cstep, x, lambda y: F.mse_loss(y, t))
    assert bool(lucid.allclose(loss_cstep.cpu(), eager_loss.cpu(), rtol=1e-4, atol=1e-5))
    _assert_close_dicts(_grads(m_cstep), want_grads)

    # 4. fused_step
    lucid.manual_seed(7)
    m_fused = _CustomVjpModule(mode).to(device)
    fstep = fused_step(m_fused, F.mse_loss, optim.SGD(m_fused.parameters(), lr=0.25))
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompileFallbackWarning)
        fstep(x, t)
    info = fstep.cache_info()
    assert info["entries"] == 1
    assert not info["eager_only"]
    _assert_close_dicts({n: p.tolist() for n, p in m_fused.named_parameters()}, want_params)

    # 5. diagnose reports 0 uncovered ops
    rpt = diagnose(m_step, x)
    assert not rpt.uncovered


def test_traced_custom_vjp_segmented_step_compiles(device_gpu_only: str) -> None:
    device = device_gpu_only
    lucid.manual_seed(11)
    x = lucid.randn(4, 4).to(device)
    t = lucid.randn(4, 3).to(device)

    class _Block(nn.Module):
        def __init__(self, fn: Callable[[lucid.Tensor], lucid.Tensor]) -> None:
            super().__init__()
            self.fn = fn

        def forward(self, h: lucid.Tensor) -> lucid.Tensor:
            return self.fn(h)

    def build() -> nn.Sequential:
        lucid.manual_seed(3)
        return nn.Sequential(
            nn.Linear(4, 4), _Block(_TracedScaledReversal.apply), nn.Linear(4, 3)
        ).to(device)

    eager = build()
    F.mse_loss(eager(x), t).backward()
    want_grads = _grads(eager)

    model = build()
    step = make_step(model, F.mse_loss, segments=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompileFallbackWarning)
        step(x, t).backward()
    assert not step.eager_only["flag"]
    _assert_close_dicts(_grads(model), want_grads)


class _GammaVAEBlock(nn.Module):
    def __init__(self, dist_kind: str) -> None:
        super().__init__()
        self.dist_kind = dist_kind
        self.encoder = nn.Linear(4, 4)
        self.decoder = nn.Linear(4, 3)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        raw = F.softplus(self.encoder(x)) + 0.5
        if self.dist_kind == "gamma":
            z = Gamma(raw, lucid.ones_like(raw)).rsample()
        elif self.dist_kind == "beta":
            z = Beta(raw, raw + 0.5).rsample()
        elif self.dist_kind == "dirichlet":
            z = Dirichlet(raw).rsample()
        elif self.dist_kind == "chi2":
            z = Chi2(raw + 1.5).rsample()
        elif self.dist_kind == "student_t":
            z = StudentT(raw + 2.5, loc=x * 0.2, scale=raw * 0.3 + 0.5).rsample()
        elif self.dist_kind == "fisher_snedecor":
            z = FisherSnedecor(raw + 3.0, raw + 4.5).rsample()
        elif self.dist_kind == "inverse_gamma":
            z = InverseGamma(raw + 2.5, raw + 1.2).rsample()
        elif self.dist_kind == "mvn":
            eye = lucid.eye(4, dtype=raw.dtype, device=raw.device)
            cov = (raw.mT @ raw) * 0.1 + eye * 1.5
            z = MultivariateNormal(raw, covariance_matrix=cov).rsample()
        elif self.dist_kind == "lowrank_mvn":
            factor = raw.unsqueeze(-1) * 0.2
            z = LowRankMultivariateNormal(raw, factor, raw * 0.5 + 1.0).rsample()
        elif self.dist_kind == "wishart":
            eye = lucid.eye(4, dtype=raw.dtype, device=raw.device)
            scale = (raw.mT @ raw) * 0.1 + eye * 1.5
            df = lucid.tensor(6.0, dtype=raw.dtype, device=raw.device)
            w_sample = Wishart(df, covariance_matrix=scale).rsample()
            z = raw @ w_sample
        else:
            raise ValueError(self.dist_kind)
        return self.decoder(z)


@pytest.mark.parametrize(
    "dist_kind",
    [
        "gamma",
        "beta",
        "dirichlet",
        "chi2",
        "student_t",
        "fisher_snedecor",
        "inverse_gamma",
        "mvn",
        "lowrank_mvn",
        "wishart",
    ],
)
def test_standard_gamma_rsample_compiles_in_make_step_and_fused_step(
    dist_kind: str, device_gpu_only: str
) -> None:
    device = device_gpu_only
    lucid.manual_seed(19)
    x = lucid.randn(4, 4).to(device)
    t = lucid.randn(4, 3).to(device)

    lucid.manual_seed(23)
    model = _GammaVAEBlock(dist_kind).to(device)
    step = make_step(model, F.mse_loss)
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompileFallbackWarning)
        loss = step(x, t)
        loss.backward()
    assert len(step.cache) == 1
    assert not step.eager_only
    for name, p in model.named_parameters():
        assert p.grad is not None, f"Missing grad for {name}"
        assert bool(lucid.isfinite(p.grad).all().item()), f"Non-finite grad for {name}"
        assert float(p.grad.abs().sum().item()) > 0.0, f"Zero grad for {name}"

    lucid.manual_seed(23)
    m_fused = _GammaVAEBlock(dist_kind).to(device)
    fstep = fused_step(m_fused, F.mse_loss, optim.Adam(m_fused.parameters(), lr=1e-2))
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompileFallbackWarning)
        loss_f = fstep(x, t)
    assert bool(lucid.isfinite(loss_f).all().item())
    info = fstep.cache_info()
    assert info["entries"] == 1
    assert not info["eager_only"]


def test_roformer_training_compiles_in_make_step_and_fused_step(device_gpu_only: str) -> None:
    device = device_gpu_only
    cfg = RoFormerConfig(
        vocab_size=32,
        embedding_size=16,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=16,
        num_labels=3,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    ids = lucid.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=lucid.int64, device=device)
    target = lucid.randn(2, 3, device=device)

    def _cls_mse(out: object, t: Tensor) -> Tensor:
        return F.mse_loss(out.logits, t)  # type: ignore[attr-defined]

    lucid.manual_seed(31)
    eager = RoFormerForSequenceClassification(cfg).to(device).train()
    _cls_mse(eager(ids), target).backward()
    want_grads = _grads(eager)

    lucid.manual_seed(31)
    model = RoFormerForSequenceClassification(cfg).to(device).train()
    step = make_step(model, _cls_mse)
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompileFallbackWarning)
        step(ids, target).backward()
    assert len(step.cache) == 1
    assert not step.eager_only
    _assert_close_dicts(_grads(model), want_grads)

    lucid.manual_seed(31)
    m_fused = RoFormerForSequenceClassification(cfg).to(device).train()
    fstep = fused_step(m_fused, _cls_mse, optim.AdamW(m_fused.parameters(), lr=1e-3))
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompileFallbackWarning)
        loss_f = fstep(ids, target)
    assert bool(lucid.isfinite(loss_f).all().item())
    info = fstep.cache_info()
    assert info["entries"] == 1
    assert not info["eager_only"]


class _MixedArgsAndMatmulCotangentFn(Function):
    _traced_backward = True

    @staticmethod
    def forward(ctx, x_grad, x_nograd, idx_int, scale_float, flag_bool, mode_str, opt_none):
        ctx.save_for_backward(x_grad, x_nograd)
        ctx.scale = scale_float
        ctx.flag = flag_bool
        ctx.mode = mode_str
        base = x_grad * scale_float + x_nograd * 0.5
        if flag_bool and mode_str == "cubic":
            return base + (x_grad * x_grad * x_grad) * 0.1, x_grad @ x_grad.mT
        return base, x_grad @ x_grad.mT

    @staticmethod
    def backward(ctx, g_out0, g_out1):
        x_grad, _x_nograd = ctx.saved_tensors
        nig = ctx.needs_input_grad
        assert nig == (True, False, False, False, False, False, False)
        # g_out1 is a zero-splat cotangent when output 1 is unused; combining
        # elementwise + matmul on g_out1 exercises constant-splat materialization.
        dx = g_out0 * ctx.scale + (g_out1 + x_grad) * (g_out1 @ x_grad)
        if ctx.flag and ctx.mode == "cubic":
            dx = dx + g_out0 * (0.3 * x_grad * x_grad)
        return dx, None, None, None, None, None, None


class _DiamondFanoutFn(Function):
    _traced_backward = True

    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a * b + a, a - b * 0.5, a + b

    @staticmethod
    def backward(ctx, gu, gv, gw):
        a, b = ctx.saved_tensors
        return gu * (b + 1.0) + gv + gw, gu * a - gv * 0.5 + gw


def test_complex_vjp_region_topologies_and_mixed_args(device_gpu_only: str) -> None:
    """Diamond fan-out, dead VJP branch, unused matmul-cotangent output, and mixed args match eager."""
    device = device_gpu_only

    class _TopologyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(3, 3)
            self.bias_vec = nn.Parameter(lucid.tensor([0.2, -0.1, 0.3], dtype=lucid.float32))
            self.fc2 = nn.Linear(3, 2)
            self.register_buffer("const_buf", lucid.tensor([[0.1, 0.2, 0.3]], dtype=lucid.float32))
            self.register_buffer("idx_buf", lucid.tensor([0, 1, 2], dtype=lucid.int64))

        def forward(self, x: lucid.Tensor) -> lucid.Tensor:
            h = self.fc1(x)
            _dead = _TracedScaledReversal.apply(h)
            y0, _unused_y1 = _MixedArgsAndMatmulCotangentFn.apply(
                h.mT.mT, self.const_buf, self.idx_buf, 1.25, True, "cubic", None
            )
            z = y0 + self.bias_vec
            u, v, _unused_w = _DiamondFanoutFn.apply(z, z)
            return self.fc2(u * 0.6 - v * 0.4)

    lucid.manual_seed(77)
    x = lucid.randn(3, 3, device=device)
    t = lucid.randn(3, 2, device=device)

    lucid.manual_seed(13)
    eager = _TopologyModel().to(device)
    F.mse_loss(eager(x), t).backward()
    want_grads = _grads(eager)

    lucid.manual_seed(13)
    comp = _TopologyModel().to(device)
    step = make_step(comp, F.mse_loss)
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompileFallbackWarning)
        step(x, t).backward()
    assert len(step.cache) == 1
    assert not step.eager_only
    _assert_close_dicts(_grads(comp), want_grads)


def test_degenerate_and_mode_r_linalg_compile_and_match_eager(device_gpu_only: str) -> None:
    """1x1 / 3x1 SVD, QR (including mode='r'), slogdet, and negative matrix_power compile and match eager."""
    device = device_gpu_only

    class _DegenerateLinalgModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a11 = nn.Parameter(lucid.tensor([[2.5]], dtype=lucid.float32))
            self.a31 = nn.Parameter(lucid.tensor([[1.8], [-0.7], [1.2]], dtype=lucid.float32))

        def forward(self, x: lucid.Tensor) -> lucid.Tensor:
            m11 = self.a11 + x * 0.0
            m31 = self.a31 + x * 0.0
            _, r11 = lucid.linalg.qr(m11, mode="r")
            q31, r31 = lucid.linalg.qr(m31, mode="reduced")
            u11, s11, vh11 = lucid.linalg.svd(m11, full_matrices=False)
            u31, s31, vh31 = lucid.linalg.svd(m31, full_matrices=True)
            sign11, lad11 = lucid.linalg.slogdet(m11)
            inv11 = lucid.linalg.matrix_power(m11, -1)
            rec31 = (u31[:, :1] * s31.unsqueeze(-2)) @ vh31[:1, :]
            total = (
                (r11 * 1.2).sum()
                + ((q31 @ r31) * 0.5).sum()
                + ((u11 * s11) @ vh11).sum()
                + (rec31 * 0.8).sum()
                + (sign11 * lad11).sum()
                + (inv11 * 0.4).sum()
            )
            return total.reshape(1)

    x = lucid.ones(1, dtype=lucid.float32, device=device)
    t = lucid.zeros(1, dtype=lucid.float32, device=device)

    eager = _DegenerateLinalgModel().to(device)
    F.mse_loss(eager(x), t).backward()
    want_grads = _grads(eager)

    comp = _DegenerateLinalgModel().to(device)
    step = make_step(comp, F.mse_loss)
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompileFallbackWarning)
        step(x, t).backward()
    assert len(step.cache) == 1
    assert not step.eager_only
    _assert_close_dicts(_grads(comp), want_grads)
