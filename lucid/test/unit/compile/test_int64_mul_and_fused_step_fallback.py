"""Guard tests for 64-bit integer self-multiplication in MPSGraph and ``fused_step`` eager fallback (LCD-293)."""

import warnings
from typing import override

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid._C import engine as _C_engine
from lucid.amp import GradScaler
from lucid.compile import fused_step
from lucid.compile._core.fallback import _NOTICED, CompileFallbackWarning
from lucid.test._fixtures.devices import metal_available

pytestmark = pytest.mark.skipif(not metal_available(), reason="Metal unavailable")


@pytest.fixture(autouse=True)
def _clean_compile_verbose(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure ``LUCID_COMPILE_VERBOSE`` left by ``_op_matrix.run`` does not suppress warnings."""
    monkeypatch.delenv("LUCID_COMPILE_VERBOSE", raising=False)


@pytest.mark.parametrize("dtype", [lucid.int64, lucid.int32])
@pytest.mark.parametrize(
    "kind",
    [
        "same_operand",
        "cse_abs",
        "cse_argmax",
        "max_left",
        "min_right",
        "floordiv",
        "remainder",
        "fmod",
    ],
)
def test_integer_self_mul_compiles_and_matches_eager(dtype: lucid.NumericDType, kind: str) -> None:
    """``int64`` and ``int32`` self-products compile without hitting ``square_i64``."""
    _C_engine.compile.session_cache_clear()

    def _run_case(a: lucid.Tensor) -> lucid.Tensor:
        if kind == "same_operand":
            return a * a
        if kind == "cse_abs":
            return lucid.abs(a) * lucid.abs(a)
        if kind == "cse_argmax":
            idx = a.reshape(2, 4).argmax(dim=1).to(dtype)
            idx2 = a.reshape(2, 4).argmax(dim=1).to(dtype)
            return idx * idx2
        if kind == "max_left":
            return lucid.maximum(a, a) * a
        if kind == "min_right":
            return a * lucid.minimum(a, a)
        if kind == "remainder":
            return a % a
        if kind == "fmod":
            return lucid.fmod(a, a)
        return a // a

    vals = (
        [-3_000_000_000, -65537, -7, -1, 1, 7, 65537, 3_000_000_000]
        if dtype == lucid.int64
        else [-40000, -257, -7, -1, 1, 7, 257, 40000]
    )
    x_metal = lucid.tensor(vals, dtype=dtype, device="metal")
    x_cpu = lucid.tensor(vals, dtype=dtype, device="cpu")

    compiled = lucid.compile(_run_case)
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompileFallbackWarning)
        got = compiled(x_metal).to("cpu").numpy()
        got_second = compiled(x_metal).to("cpu").numpy()

    want = _run_case(x_cpu).numpy()
    info = compiled.cache_info()
    assert info["entries"] == 1, info
    assert info["eager_only"] == (), info
    assert got.dtype == want.dtype
    assert np.array_equal(got, want)
    assert np.array_equal(got_second, want)


@pytest.mark.parametrize("dtype", [lucid.int64, lucid.int32])
@pytest.mark.parametrize("shared_var", [False, True])
def test_fused_step_with_argmax_self_mul_compiles_without_fallback(
    dtype: lucid.NumericDType, shared_var: bool
) -> None:
    """``fused_step`` compiles ``y.argmax(1) * y.argmax(1)`` and ``u * u`` on GPU."""
    lucid.manual_seed(0)
    comp_model = nn.Linear(8, 4).to("metal")
    eager_model = nn.Linear(8, 4).to("metal")
    eager_model.load_state_dict(comp_model.state_dict())

    def loss_fn(p: lucid.Tensor, y: lucid.Tensor) -> lucid.Tensor:
        if shared_var:
            u = y.argmax(dim=1).to(dtype)
            prod = u * u
        else:
            prod = y.argmax(dim=1).to(dtype) * y.argmax(dim=1).to(dtype)
        return (prod.to(p.dtype) * p.sum(dim=1)).sum()

    comp_opt = optim.Adam(comp_model.parameters(), lr=0.05)
    eager_opt = optim.Adam(eager_model.parameters(), lr=0.05)
    step = fused_step(comp_model, loss_fn, comp_opt)

    with warnings.catch_warnings():
        warnings.simplefilter("error", CompileFallbackWarning)
        for _ in range(3):
            x = lucid.randn(4, 8, device="metal")
            y = lucid.randn(4, 4, device="metal")
            got_loss = float(step(x, y).item())

            eager_opt.zero_grad()
            want_t = loss_fn(eager_model(x), y)
            want_t.backward()
            eager_opt.step()
            want_loss = float(want_t.item())

            assert np.isclose(got_loss, want_loss, rtol=1e-5, atol=1e-5)

    for p_c, p_e in zip(comp_model.parameters(), eager_model.parameters(), strict=True):
        assert np.allclose(p_c.to("cpu").numpy(), p_e.to("cpu").numpy(), rtol=1e-5, atol=1e-5)


def test_fused_step_falls_back_to_eager_on_uncompilable_op() -> None:
    """When a loss uses an op whose emitter declines lowering, ``fused_step`` falls back to eager."""
    lucid.manual_seed(42)
    comp_model = nn.Linear(6, 3).to("metal")
    eager_model = nn.Linear(6, 3).to("metal")
    eager_model.load_state_dict(comp_model.state_dict())

    # Integer ``inner`` runs as ``A @ B^T`` where MPSGraph has no integer matmul kernel,
    # so ``InnerEmitter`` declines lowering.
    def loss_fn(p: lucid.Tensor, y: lucid.Tensor) -> lucid.Tensor:
        idx = y.argmax(dim=1)
        scale = lucid.linalg.inner(idx, idx).to(p.dtype)
        return ((p * y).sum(dim=1) * (scale + 1.0)).mean()

    comp_opt = optim.Adam(comp_model.parameters(), lr=0.02)
    eager_opt = optim.Adam(eager_model.parameters(), lr=0.02)
    step = fused_step(comp_model, loss_fn, comp_opt)

    _NOTICED.discard("lowering")
    with pytest.warns(CompileFallbackWarning, match="running eagerly"):
        x0 = lucid.randn(4, 6, device="metal")
        y0 = lucid.randn(4, 3, device="metal")
        got0 = step(x0, y0)
    assert got0.grad_fn is None

    eager_opt.zero_grad()
    want0 = loss_fn(eager_model(x0), y0)
    want0.backward()
    eager_opt.step()
    assert np.isclose(float(got0.item()), float(want0.item()), rtol=1e-5, atol=1e-5)

    # Subsequent steps use the blacklisted eager path without re-warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompileFallbackWarning)
        for _ in range(3):
            x = lucid.randn(4, 6, device="metal")
            y = lucid.randn(4, 3, device="metal")
            got = float(step(x, y).item())

            eager_opt.zero_grad()
            want_t = loss_fn(eager_model(x), y)
            want_t.backward()
            eager_opt.step()
            assert np.isclose(got, float(want_t.item()), rtol=1e-5, atol=1e-5)

    for p_c, p_e in zip(comp_model.parameters(), eager_model.parameters(), strict=True):
        assert np.allclose(p_c.to("cpu").numpy(), p_e.to("cpu").numpy(), rtol=1e-5, atol=1e-5)
    comp_sd = comp_opt.state_dict()
    eager_sd = eager_opt.state_dict()
    assert comp_sd["state"].keys() == eager_sd["state"].keys()


def test_fused_step_falls_back_on_cumulative_bn_and_grad_scaler() -> None:
    """Cumulative BatchNorm (``momentum=None``) + ``GradScaler`` falls back cleanly in ``fused_step``."""

    class _CumBNNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 4)
            self.bn = nn.BatchNorm1d(4, momentum=None)

        @override
        def forward(self, x: lucid.Tensor) -> lucid.Tensor:
            return self.bn(self.fc(x))

    lucid.manual_seed(7)
    comp_model = _CumBNNet().to("metal").train()
    eager_model = _CumBNNet().to("metal").train()
    eager_model.load_state_dict(comp_model.state_dict())

    comp_opt = optim.SGD(comp_model.parameters(), lr=0.05, momentum=0.9)
    eager_opt = optim.SGD(eager_model.parameters(), lr=0.05, momentum=0.9)
    comp_scaler = GradScaler(init_scale=128.0)
    eager_scaler = GradScaler(init_scale=128.0)

    step = fused_step(
        comp_model,
        F.mse_loss,
        comp_opt,
        grad_scaler=comp_scaler,
    )

    _NOTICED.discard("cumulative-bn")
    with pytest.warns(CompileFallbackWarning, match="momentum=None"):
        for _ in range(3):
            x = lucid.randn(8, 4, device="metal")
            t = lucid.randn(8, 4, device="metal")
            got = float(step(x, t).item())

            eager_opt.zero_grad()
            want_loss = F.mse_loss(eager_model(x), t)
            scaled = eager_scaler.scale(want_loss)
            assert isinstance(scaled, lucid.Tensor)
            scaled.backward()
            eager_scaler.step(eager_opt)
            eager_scaler.update()

            assert np.isclose(got, float(want_loss.item()), rtol=1e-5, atol=1e-5)

    assert int(comp_model.bn.num_batches_tracked.item()) == 3
    for p_c, p_e in zip(comp_model.parameters(), eager_model.parameters(), strict=True):
        assert np.allclose(p_c.to("cpu").numpy(), p_e.to("cpu").numpy(), rtol=1e-5, atol=1e-5)


def test_fused_step_falls_back_on_first_run_mpsgraph_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ``run_executable_inplace`` raises ``MPSGraph failed`` on first run, ``fused_step`` falls back."""
    lucid.manual_seed(11)
    comp_model = nn.Linear(4, 2).to("metal")
    eager_model = nn.Linear(4, 2).to("metal")
    eager_model.load_state_dict(comp_model.state_dict())

    comp_opt = optim.SGD(comp_model.parameters(), lr=0.1, momentum=0.5)
    eager_opt = optim.SGD(eager_model.parameters(), lr=0.1, momentum=0.5)
    step = fused_step(comp_model, F.mse_loss, comp_opt)

    orig_run_inplace = _C_engine.compile.run_executable_inplace
    fused_calls = 0

    def _fail_fused_inplace(exe: object, feeds: list[object], targets: list[object]) -> None:
        nonlocal fused_calls
        # The fused executable writes [loss, new_w, new_b, new_mom_w, new_mom_b] (5 targets),
        # whereas standalone copt.step() writes [new_w, new_b, new_mom_w, new_mom_b] (4 targets).
        if len(targets) == 5:
            fused_calls += 1
            raise RuntimeError(
                "run_executable_inplace: MPSGraph failed: simulated first-run failure"
            )
        orig_run_inplace(exe, feeds, targets)

    monkeypatch.setattr(_C_engine.compile, "run_executable_inplace", _fail_fused_inplace)
    _NOTICED.discard("first-run")

    x = lucid.randn(3, 4, device="metal")
    t = lucid.randn(3, 2, device="metal")
    with pytest.warns(CompileFallbackWarning, match="MPSGraph failed"):
        got1 = float(step(x, t).item())
    # Second call hits _eager_only without invoking the fused executable again.
    got2 = float(step(x, t).item())
    assert fused_calls == 1

    want_losses: list[float] = []
    for _ in range(2):
        eager_opt.zero_grad()
        wl = F.mse_loss(eager_model(x), t)
        wl.backward()
        eager_opt.step()
        want_losses.append(float(wl.item()))

    assert np.isclose(got1, want_losses[0], rtol=1e-5, atol=1e-5)
    assert np.isclose(got2, want_losses[1], rtol=1e-5, atol=1e-5)
    for p_c, p_e in zip(comp_model.parameters(), eager_model.parameters(), strict=True):
        assert np.allclose(p_c.to("cpu").numpy(), p_e.to("cpu").numpy(), rtol=1e-5, atol=1e-5)
