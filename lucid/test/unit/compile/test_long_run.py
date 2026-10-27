"""Long-run training parity: eager vs compile over 100 steps.

A single-step parity test catches algorithmic bugs in the
forward/backward graph, but it doesn't catch slow-drift issues:

  * Optimizer state buffers diverging by epsilon each step (e.g.
    Adam's ``v`` accumulator using a different reduction tree).
  * Numerical instability when a tiny per-step delta gets amplified
    by curvature over many steps.
  * Cache key collisions surfacing only after the second iteration
    (the first iteration uses the same code path but a fresh cache).

These tests pin the loss trajectory of a small model trained for 100
steps in both modes — the curves must remain within a generous-but-
finite tolerance across the entire trajectory.  fp32 reductions
diverge predictably; a real bug shows up as either an O(1) jump or
a monotone drift that compounds over steps.
"""

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid.compile import compile_optimizer, fused_step

from lucid.test.unit.compile._helpers import COMPILE_DEVICE


def _make_problem() -> tuple[lucid.Tensor, lucid.Tensor, callable]:
    """Synthetic regression dataset + model factory.

    The data is deterministic (seeded inside the call) and small
    enough that 100 steps converge to a non-trivial loss minimum
    without needing fp64.
    """
    lucid.manual_seed(0)
    x = lucid.randn(64, 8).to(COMPILE_DEVICE)
    y = lucid.randn(64, 4).to(COMPILE_DEVICE)

    def factory() -> nn.Module:
        class _M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc1 = nn.Linear(8, 16)
                self.fc2 = nn.Linear(16, 4)

            def forward(self, z: lucid.Tensor) -> lucid.Tensor:
                return self.fc2(self.fc1(z).relu())

        return _M().to(COMPILE_DEVICE)

    return x, y, factory


def _sync_params(src: nn.Module, dst: nn.Module) -> None:
    """Copy ``src.parameters()`` into ``dst.parameters()`` slot-by-slot."""
    for (_, p_s), (_, p_d) in zip(src.named_parameters(), dst.named_parameters()):
        with lucid.no_grad():
            p_d.copy_(p_s)


def _train_eager(model: nn.Module, x: lucid.Tensor, y: lucid.Tensor, n_steps: int) -> list[float]:
    opt = optim.Adam(list(model.parameters()), lr=0.01)
    losses: list[float] = []
    for _ in range(n_steps):
        opt.zero_grad()
        loss = F.mse_loss(model(x), y)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return losses


def _train_compile_optimizer(
    model: nn.Module, x: lucid.Tensor, y: lucid.Tensor, n_steps: int
) -> list[float]:
    """Compile only the optimizer step; forward/backward stays eager."""
    opt = optim.Adam(list(model.parameters()), lr=0.01)
    copt = compile_optimizer(opt)
    losses: list[float] = []
    for _ in range(n_steps):
        copt.zero_grad()
        loss = F.mse_loss(model(x), y)
        loss.backward()
        copt.step()
        losses.append(float(loss.item()))
    return losses


def _train_fused(model: nn.Module, x: lucid.Tensor, y: lucid.Tensor, n_steps: int) -> list[float]:
    """End-to-end fused: forward + loss + backward + update in one executable."""
    opt = optim.Adam(list(model.parameters()), lr=0.01)
    step = fused_step(model, F.mse_loss, opt)
    losses: list[float] = []
    for _ in range(n_steps):
        loss = step(x, y)
        losses.append(float(loss.item()))
    return losses


def _max_step_diff(a: list[float], b: list[float]) -> tuple[int, float, float]:
    """Worst absolute and relative loss-curve drift across ``a`` vs ``b``."""
    worst_abs = 0.0
    worst_rel = 0.0
    worst_step = -1
    for i, (la, lb) in enumerate(zip(a, b)):
        d = abs(la - lb)
        r = d / max(abs(la), 1e-9)
        if d > worst_abs:
            worst_abs = d
            worst_rel = r
            worst_step = i
    return worst_step, worst_abs, worst_rel


def test_compile_optimizer_long_run() -> None:
    """``compile_optimizer`` matches eager Adam over 100 training steps."""
    x, y, factory = _make_problem()

    model_eager = factory()
    model_comp = factory()
    _sync_params(model_eager, model_comp)

    losses_eager = _train_eager(model_eager, x, y, n_steps=100)
    losses_comp = _train_compile_optimizer(model_comp, x, y, n_steps=100)

    step, abs_diff, rel_diff = _max_step_diff(losses_eager, losses_comp)
    # Loss must decrease (training is effective)
    assert losses_eager[0] > losses_eager[-1], f"eager training stalled: {losses_eager[:3]}"
    assert losses_comp[0] > losses_comp[-1], f"compile training stalled: {losses_comp[:3]}"
    # Trajectories must agree to fp32 tolerance — the optimizer update
    # is bit-exact apart from MPSGraph's reduction reordering, so the
    # compounded drift over 100 steps stays small.
    assert rel_diff < 1e-3, (
        f"compile_optimizer drift at step {step}: "
        f"eager={losses_eager[step]:.6f}, compile={losses_comp[step]:.6f}, "
        f"abs={abs_diff:.3e}, rel={rel_diff:.3e}"
    )


def test_fused_step_long_run() -> None:
    """``fused_step`` matches eager Adam over 100 training steps."""
    x, y, factory = _make_problem()

    model_eager = factory()
    model_fused = factory()
    _sync_params(model_eager, model_fused)

    losses_eager = _train_eager(model_eager, x, y, n_steps=100)
    losses_fused = _train_fused(model_fused, x, y, n_steps=100)

    step, abs_diff, rel_diff = _max_step_diff(losses_eager, losses_fused)
    assert losses_eager[0] > losses_eager[-1]
    assert losses_fused[0] > losses_fused[-1]
    # fused_step does forward + backward in MPSGraph autodiff, which
    # has slightly more reorder freedom than the eager autograd
    # engine — allow a slightly wider tolerance than the
    # compile_optimizer-only case.
    assert rel_diff < 5e-3, (
        f"fused_step drift at step {step}: "
        f"eager={losses_eager[step]:.6f}, fused={losses_fused[step]:.6f}, "
        f"abs={abs_diff:.3e}, rel={rel_diff:.3e}"
    )


def test_long_run_final_param_drift() -> None:
    """After 100 steps the *parameter values* shouldn't drift > 1%.

    Loss-curve parity is a derived metric; the underlying parameters
    are what production training cares about.  This test pins both.
    """
    x, y, factory = _make_problem()

    model_eager = factory()
    model_comp = factory()
    _sync_params(model_eager, model_comp)

    _train_eager(model_eager, x, y, n_steps=100)
    _train_compile_optimizer(model_comp, x, y, n_steps=100)

    worst = 0.0
    for (n, p_e), (_, p_c) in zip(model_eager.named_parameters(), model_comp.named_parameters()):
        d = float((p_e - p_c).abs().max().item())
        scale = float(p_e.abs().max().item())
        rel = d / max(scale, 1e-9)
        if rel > worst:
            worst = rel
    assert worst < 1e-2, f"final param drift = {worst:.3e}"
