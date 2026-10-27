"""Compiled training steps record in the caller's grad mode (LCD-580, LCD-607).

Covers the C' training-step matrix across ``make_step``, ``make_step(..., segments=2)``,
and ``fused_step`` on Metal:
- ``is_grad_enabled()``-gated straight-through estimator (STE) falls back to
  eager with a ``CompileFallbackWarning`` and trains like eager;
- ``is_grad_enabled()``-gated ``Function``-less branch compiles and trains like eager;
- ``is_grad_enabled()``-gated ``odeint_adjoint`` falls back to eager with a
  ``CompileFallbackWarning`` and trains like eager;
- ``is_grad_enabled()``-gated ``checkpoint`` compiles and trains like eager;
- ``requires_grad``-guarded ``register_hook`` and ``retain_grad()`` fall back to
  eager and train like eager;
- Bookkeeping trace forms (``trunc`` division, ``round(decimals=...)``,
  ``setitem``, ``fill_``, ``index_fill_``, ``ReLU(inplace=True)``) and
  ``BatchNorm2d + ReLU(inplace=True)`` CNN blocks compile and train like eager;
- RNG draws are bit-equal to eager on the first (trace), second (cached), and
  refused (``eager_only``, including buffer-reassigning) calls.
"""

from collections.abc import Callable
from typing import cast
import warnings

import numpy as np
import pytest

import lucid
import lucid.diffeq as diffeq
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid.autograd import Function, FunctionCtx, checkpoint
from lucid.compile import fused_step, make_step
from lucid.compile._core.fallback import _NOTICED, CompileFallbackWarning
from lucid.test._fixtures.devices import metal_available

pytestmark = pytest.mark.skipif(not metal_available(), reason="Metal unavailable")

_ENTRY_KINDS = ("make_step", "segments_2", "fused_step")


@pytest.fixture(autouse=True)
def _clear_verbose(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure ``LUCID_COMPILE_VERBOSE`` left by earlier matrix runs does not suppress warnings."""
    monkeypatch.delenv("LUCID_COMPILE_VERBOSE", raising=False)


class _RoundSTE(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, x: lucid.Tensor) -> lucid.Tensor:
        del ctx
        return x.round()

    @staticmethod
    def backward(ctx: FunctionCtx, grad_out: lucid.Tensor) -> lucid.Tensor:
        del ctx
        return grad_out


class _GatedSTEBlock(nn.Module):
    def forward(self, h: lucid.Tensor) -> lucid.Tensor:
        if lucid.is_grad_enabled():
            return cast(lucid.Tensor, _RoundSTE.apply(h))
        return h.round()


class _GatedPlainBlock(nn.Module):
    def forward(self, h: lucid.Tensor) -> lucid.Tensor:
        if lucid.is_grad_enabled():
            return h * 2.0 + h.sin()
        return h * 0.5


def _rhs(t: lucid.Tensor, y: lucid.Tensor) -> lucid.Tensor:
    del t
    return -0.25 * y


class _GatedAdjointBlock(nn.Module):
    def forward(self, h: lucid.Tensor) -> lucid.Tensor:
        grid = [0.0, 0.5]
        opts = {"step_size": 0.25}
        if lucid.is_grad_enabled():
            traj = cast(
                lucid.Tensor,
                diffeq.odeint_adjoint(_rhs, h, grid, method="euler", options=opts),
            )
            return traj[-1]
        return cast(
            lucid.Tensor,
            diffeq.odeint(
                _rhs,
                h,
                grid,
                method="euler",
                options=opts,
                return_trajectory=False,
            ),
        )


class _GatedCheckpointBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def _body(self, h: lucid.Tensor) -> lucid.Tensor:
        return F.silu(self.lin(h))

    def forward(self, h: lucid.Tensor) -> lucid.Tensor:
        if lucid.is_grad_enabled():
            return cast(
                lucid.Tensor,
                checkpoint(self._body, h, use_reentrant=False),
            )
        return self._body(h)


class _GuardedHookBlock(nn.Module):
    def __init__(self, kind: str) -> None:
        super().__init__()
        self.kind = kind

    def forward(self, h: lucid.Tensor) -> lucid.Tensor:
        out = h * 1.5
        if out.requires_grad:
            if self.kind == "hook":
                out.register_hook(lambda g: g * 3.0)
            else:
                out.retain_grad()
        return out


class _BookkeepingBlock(nn.Module):
    def __init__(self, op_kind: str) -> None:
        super().__init__()
        self.op_kind = op_kind

    def forward(self, h: lucid.Tensor) -> lucid.Tensor:
        if self.op_kind == "trunc_div":
            return h + lucid.div(h * 3.0, 2.0, rounding_mode="trunc")
        if self.op_kind == "round_decimals":
            return h + lucid.round(h, decimals=2)
        if self.op_kind == "setitem":
            buf = h.clone()
            buf[:, :2] = h[:, :2] * 2.0
            return buf
        if self.op_kind == "fill_":
            buf = h.clone()
            buf.fill_(0.25)
            return h + buf
        if self.op_kind == "index_fill_":
            idx = lucid.tensor([0, 2], dtype=lucid.int64, device=h.device)
            buf = h.clone()
            buf.index_fill_(1, idx, 0.5)
            return buf
        if self.op_kind == "relu_inplace":
            return F.relu(h.clone(), inplace=True)
        raise ValueError(self.op_kind)


def _seq_model(middle_factory: Callable[[], nn.Module], seed: int = 0) -> nn.Sequential:
    lucid.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(4, 4),
        middle_factory(),
        nn.Linear(4, 3),
    ).to("metal")


def _batch(seed: int = 1) -> tuple[lucid.Tensor, lucid.Tensor]:
    lucid.manual_seed(seed)
    return (
        lucid.randn(5, 4, device="metal"),
        lucid.randn(5, 3, device="metal"),
    )


def _run_entry(
    kind: str,
    model_factory: Callable[[], nn.Sequential],
    *,
    expect_compiled: bool,
    warn_match: str | None = None,
) -> None:
    x, t = _batch()

    # Eager reference step
    eager = model_factory()
    eager_opt = optim.SGD(eager.parameters(), lr=0.25)
    eager_opt.zero_grad()
    want_loss = F.mse_loss(eager(x), t)
    want_loss.backward()
    want_grads = {
        n: p.grad.detach().cpu().numpy().copy()
        for n, p in eager.named_parameters()
        if p.grad is not None
    }
    eager_opt.step()
    want_params = {n: p.detach().cpu().numpy().copy() for n, p in eager.named_parameters()}

    # Compiled step under the requested entry point
    model = model_factory()
    _NOTICED.clear()
    if kind == "fused_step":
        opt = optim.SGD(model.parameters(), lr=0.25)
        step = fused_step(model, F.mse_loss, opt)
        if warn_match is not None:
            with pytest.warns(CompileFallbackWarning, match=warn_match):
                got_loss = step(x, t)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("error", CompileFallbackWarning)
                got_loss = step(x, t)
        info = step.cache_info()
        if expect_compiled:
            assert info["entries"] == 1
            assert not info["eager_only"]
        else:
            assert info["entries"] == 0
            assert len(cast(tuple[object, ...], info["eager_only"])) == 1
    else:
        segs = 2 if kind == "segments_2" else 1
        step_fn = make_step(model, F.mse_loss, segments=segs)
        if warn_match is not None:
            with pytest.warns(CompileFallbackWarning, match=warn_match):
                got_loss = step_fn(x, t)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("error", CompileFallbackWarning)
                got_loss = step_fn(x, t)
        got_loss.backward()
        for n, p in model.named_parameters():
            assert p.grad is not None, n
            np.testing.assert_allclose(
                p.grad.detach().cpu().numpy(),
                want_grads[n],
                rtol=1e-4,
                atol=1e-5,
                err_msg=n,
            )
        optim.SGD(model.parameters(), lr=0.25).step()
        if kind == "segments_2":
            if expect_compiled:
                assert not step_fn.eager_only["flag"]
                assert all(len(c) == 1 for c in step_fn.cache)
            else:
                assert step_fn.eager_only["flag"]
        else:
            if expect_compiled:
                assert len(step_fn.cache) == 1
                assert not step_fn.eager_only
            else:
                assert not step_fn.cache
                assert len(step_fn.eager_only) == 1

    np.testing.assert_allclose(
        float(got_loss.item()), float(want_loss.item()), rtol=1e-4, atol=1e-5
    )
    for n, p in model.named_parameters():
        np.testing.assert_allclose(
            p.detach().cpu().numpy(),
            want_params[n],
            rtol=1e-4,
            atol=1e-5,
            err_msg=n,
        )


@pytest.mark.parametrize("kind", _ENTRY_KINDS)
def test_gated_ste_falls_back_to_eager_and_matches_gradients(kind: str) -> None:
    _run_entry(
        kind,
        lambda: _seq_model(_GatedSTEBlock),
        expect_compiled=False,
        warn_match="_RoundSTE",
    )


@pytest.mark.parametrize("kind", _ENTRY_KINDS)
def test_gated_functionless_branch_compiles_and_matches_eager(kind: str) -> None:
    _run_entry(
        kind,
        lambda: _seq_model(_GatedPlainBlock),
        expect_compiled=True,
    )


@pytest.mark.parametrize("kind", _ENTRY_KINDS)
def test_gated_odeint_adjoint_falls_back_to_eager_with_warning(kind: str) -> None:
    _run_entry(
        kind,
        lambda: _seq_model(_GatedAdjointBlock),
        expect_compiled=False,
        warn_match="_AdjointSolve",
    )


@pytest.mark.parametrize("kind", _ENTRY_KINDS)
def test_gated_checkpoint_compiles_and_matches_eager(kind: str) -> None:
    _run_entry(
        kind,
        lambda: _seq_model(lambda: _GatedCheckpointBlock(4)),
        expect_compiled=True,
    )


@pytest.mark.parametrize("hook_kind", ["hook", "retain_grad"])
@pytest.mark.parametrize("kind", _ENTRY_KINDS)
def test_requires_grad_guarded_hook_or_retain_grad_falls_back_to_eager(
    kind: str, hook_kind: str
) -> None:
    match = "register_hook" if hook_kind == "hook" else "retain_grad"
    _run_entry(
        kind,
        lambda: _seq_model(lambda: _GuardedHookBlock(hook_kind)),
        expect_compiled=False,
        warn_match=match,
    )


@pytest.mark.parametrize(
    "op_kind",
    [
        "trunc_div",
        "round_decimals",
        "setitem",
        "fill_",
        "index_fill_",
        "relu_inplace",
    ],
)
@pytest.mark.parametrize("kind", _ENTRY_KINDS)
def test_bookkeeping_trace_forms_compile_and_match_eager(kind: str, op_kind: str) -> None:
    _run_entry(
        kind,
        lambda: _seq_model(lambda: _BookkeepingBlock(op_kind)),
        expect_compiled=True,
    )


@pytest.mark.parametrize("kind", _ENTRY_KINDS)
def test_bn_relu_inplace_cnn_compiles_and_matches_eager(kind: str) -> None:
    def make_cnn() -> nn.Sequential:
        lucid.manual_seed(0)
        return nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(4 * 4 * 4, 3),
        ).to("metal")

    lucid.manual_seed(2)
    x = lucid.randn(4, 3, 4, 4, device="metal")
    t = lucid.randn(4, 3, device="metal")

    eager = make_cnn()
    eager_opt = optim.SGD(eager.parameters(), lr=0.1)
    eager_opt.zero_grad()
    want_loss = F.mse_loss(eager(x), t)
    want_loss.backward()
    eager_opt.step()

    model = make_cnn()
    if kind == "fused_step":
        opt = optim.SGD(model.parameters(), lr=0.1)
        step = fused_step(model, F.mse_loss, opt)
        with warnings.catch_warnings():
            warnings.simplefilter("error", CompileFallbackWarning)
            got_loss = step(x, t)
        assert step.cache_info()["entries"] == 1
    else:
        segs = 2 if kind == "segments_2" else 1
        step_fn = make_step(model, F.mse_loss, segments=segs)
        with warnings.catch_warnings():
            warnings.simplefilter("error", CompileFallbackWarning)
            got_loss = step_fn(x, t)
        got_loss.backward()
        optim.SGD(model.parameters(), lr=0.1).step()
        if kind == "segments_2":
            assert not step_fn.eager_only["flag"]
        else:
            assert len(step_fn.cache) == 1

    np.testing.assert_allclose(
        float(got_loss.item()), float(want_loss.item()), rtol=1e-4, atol=1e-5
    )
    for (n, p), (_, want_p) in zip(model.named_parameters(), eager.named_parameters(), strict=True):
        np.testing.assert_allclose(
            p.detach().cpu().numpy(),
            want_p.detach().cpu().numpy(),
            rtol=1e-4,
            atol=1e-5,
            err_msg=n,
        )
    if kind != "segments_2":
        for (n, b), (_, want_b) in zip(model.named_buffers(), eager.named_buffers(), strict=True):
            np.testing.assert_allclose(
                b.detach().cpu().numpy(),
                want_b.detach().cpu().numpy(),
                rtol=1e-4,
                atol=1e-5,
                err_msg=n,
            )


class _RandomLinearBlock(nn.Module):
    def __init__(self, *, refuse_compile: bool = False) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.refuse_compile = refuse_compile
        if refuse_compile:
            self.register_buffer("seen", lucid.zeros(1, 4))

    def forward(self, h: lucid.Tensor) -> lucid.Tensor:
        r = lucid.rand(*h.shape, device=h.device)
        if self.refuse_compile:
            seen = cast(lucid.Tensor, self.seen)
            with lucid.no_grad():
                self.seen = 0.9 * seen + 0.1 * r.mean(dim=0, keepdim=True)
            return self.lin(h + r) + cast(lucid.Tensor, self.seen)
        return self.lin(h + r)


@pytest.mark.parametrize(
    ("kind", "refuse_compile"),
    [
        ("make_step", False),
        ("make_step", True),
        ("segments_2", True),
        ("fused_step", False),
        ("fused_step", True),
    ],
)
def test_rng_bit_equal_on_first_second_and_refused_calls(kind: str, refuse_compile: bool) -> None:
    def build() -> nn.Sequential:
        lucid.manual_seed(0)
        return nn.Sequential(
            _RandomLinearBlock(refuse_compile=refuse_compile),
            nn.Linear(4, 3),
        ).to("metal")

    x, t = _batch()
    eager = build()
    eager_opt = optim.SGD(eager.parameters(), lr=0.1)

    model = build()
    if kind == "fused_step":
        opt = optim.SGD(model.parameters(), lr=0.1)
        step: Callable[..., lucid.Tensor] = fused_step(model, F.mse_loss, opt)
    else:
        segs = 2 if kind == "segments_2" else 1
        step = make_step(model, F.mse_loss, segments=segs)
        opt = optim.SGD(model.parameters(), lr=0.1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CompileFallbackWarning)
        for seed in (101, 102, 103):
            lucid.manual_seed(seed)
            eager_opt.zero_grad()
            want_loss = F.mse_loss(eager(x), t)
            want_loss.backward()
            eager_opt.step()
            want_rng = lucid.get_rng_state().numpy().copy()

            lucid.manual_seed(seed)
            if kind == "fused_step":
                got_loss = step(x, t)
            else:
                opt.zero_grad()
                got_loss = step(x, t)
                got_loss.backward()
                opt.step()
            got_rng = lucid.get_rng_state().numpy().copy()

            np.testing.assert_array_equal(got_rng, want_rng)
            np.testing.assert_allclose(
                float(got_loss.item()),
                float(want_loss.item()),
                rtol=1e-4,
                atol=1e-5,
            )

    if kind == "fused_step" and refuse_compile:
        info = cast(dict[str, object], getattr(step, "cache_info")())
        assert len(cast(tuple[object, ...], info["eager_only"])) == 1
