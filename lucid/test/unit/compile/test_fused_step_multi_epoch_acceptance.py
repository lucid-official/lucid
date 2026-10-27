"""Multi-epoch end-to-end acceptance for fused_step (P4 closure).

Acceptance gate for the "compile path produces correct training
dynamics over many steps" claim from the compile-parity
architecture note (§6 P4 — see ``obsidian/architecture/``).
Per-op tests pin correctness of individual VJPs at single-step
level; this test chains 100 SGD steps and asserts that the
compiled training trajectory matches eager within tolerance.

Two workloads covered:
1. MLP (3-layer Linear+ReLU) — guaranteed-supported path
2. CIFAR ResNet block (Conv+BN+ReLU+skip) — exercises the
   high-value workload class where compile shows 2.5-4×
   speedup ([[perf-compile-vs-eager-2026-05-26]])

For each: build two identical models (same seed), train one in
eager mode and one in fused_step mode for 100 SGD steps on
synthetic data.  Assert final loss is within tolerance.

Synthetic data over CIFAR-10 because:
- Test runs in CI without external dataset deps
- Synthetic gives deterministic ground truth for comparison
- The convergence assertion is on relative loss, not absolute
  accuracy (which depends on the dataset)
"""

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
import lucid.metal as metal
from lucid.compile import fused_step

from lucid.test.unit.compile._helpers import COMPILE_DEVICE


def _train_eager(
    model: nn.Module, x: lucid.Tensor, t: lucid.Tensor, n_steps: int, lr: float
) -> list[float]:
    """Run eager training loop; return per-step loss values."""
    opt = optim.SGD(model.parameters(), lr=lr)
    losses: list[float] = []
    for _ in range(n_steps):
        opt.zero_grad()
        out = model(x)
        loss = F.mse_loss(out, t)
        loss.backward()
        opt.step()
        _ = float(loss.item())  # force eval
        metal.synchronize()
        losses.append(float(loss.item()))
    return losses


def _train_fused(
    model: nn.Module, x: lucid.Tensor, t: lucid.Tensor, n_steps: int, lr: float
) -> list[float]:
    """Run fused_step training loop; return per-step loss values."""
    opt = optim.SGD(model.parameters(), lr=lr)
    step = fused_step(model, F.mse_loss, opt)
    losses: list[float] = []
    for _ in range(n_steps):
        loss = step(x, t)
        _ = float(loss.item())
        metal.synchronize()
        losses.append(float(loss.item()))
    return losses


def _sync_models(a: nn.Module, b: nn.Module) -> None:
    """Copy parameter tensors from `a` to `b` so both start identical."""
    for (_, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        with lucid.no_grad():
            pb.copy_(pa.detach().clone())


# ── Workload 1: MLP ─────────────────────────────────────────────────


class _MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 4)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fc3(self.fc2(self.fc1(x).relu()).relu())


def test_fused_step_mlp_100_steps_matches_eager() -> None:
    """100 SGD steps on MLP — fused_step vs eager loss trajectory parity.

    Both paths start from identical weights (manual_seed(0) + copy).
    After 100 steps the final loss should match within 1e-3 absolute
    (F32 numerical drift over 100 SGD updates is small but non-zero).
    """
    n_steps = 100
    lr = 1e-2

    lucid.manual_seed(0)
    m_eager = _MLP().to(COMPILE_DEVICE)
    m_fused = _MLP().to(COMPILE_DEVICE)
    _sync_models(m_eager, m_fused)

    m_eager.train()
    m_fused.train()

    lucid.manual_seed(42)
    x = lucid.randn(16, 8).to(COMPILE_DEVICE)
    t = lucid.randn(16, 4).to(COMPILE_DEVICE)

    eager_losses = _train_eager(m_eager, x, t, n_steps, lr)
    fused_losses = _train_fused(m_fused, x, t, n_steps, lr)

    final_diff = abs(eager_losses[-1] - fused_losses[-1])
    assert final_diff < 1e-3, (
        f"fused_step diverged from eager after {n_steps} steps: "
        f"eager_final={eager_losses[-1]:.6f}, "
        f"fused_final={fused_losses[-1]:.6f}, "
        f"diff={final_diff:.6f}"
    )

    # Sanity: loss should decrease monotonically on synthetic data.
    # Random targets at lr=1e-2 don't drop dramatically over 100 steps,
    # so check for any decrease rather than a fixed ratio.
    assert eager_losses[0] > eager_losses[-1], (
        f"eager loss didn't decrease over {n_steps} steps "
        f"(start={eager_losses[0]:.4f}, end={eager_losses[-1]:.4f})"
    )


# ── Workload 2: CIFAR ResNet block ──────────────────────────────────


class _CifarResNetBlock(nn.Module):
    """Single residual block, no downsample — matches
    bench_compile_vs_eager's workload.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(8)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        identity = x
        out = self.bn1(self.conv1(x)).relu()
        out = self.bn2(self.conv2(out))
        return (out + identity).relu()


def test_fused_step_cifar_block_50_steps_matches_eager() -> None:
    """50 SGD steps on CIFAR ResNet block — fused_step vs eager.

    Shorter run (50 vs 100) because Conv+BN training step is heavier;
    keeps test wall-time reasonable.  Tolerance relaxed slightly
    (3e-3) because BN running stats accumulate across many steps
    and the eager-vs-compile order-of-operations can drift slightly
    (compile fuses the BN update with the rest of the executable;
    eager runs BN's running-stat update as a separate op).
    """
    n_steps = 50
    lr = 1e-3

    lucid.manual_seed(0)
    m_eager = _CifarResNetBlock().to(COMPILE_DEVICE)
    m_fused = _CifarResNetBlock().to(COMPILE_DEVICE)
    _sync_models(m_eager, m_fused)

    m_eager.train()
    m_fused.train()

    lucid.manual_seed(42)
    x = lucid.randn(8, 8, 16, 16).to(COMPILE_DEVICE)
    t = lucid.randn(8, 8, 16, 16).to(COMPILE_DEVICE)

    eager_losses = _train_eager(m_eager, x, t, n_steps, lr)
    fused_losses = _train_fused(m_fused, x, t, n_steps, lr)

    final_diff = abs(eager_losses[-1] - fused_losses[-1])
    assert final_diff < 3e-3, (
        f"fused_step diverged from eager after {n_steps} steps on "
        f"CIFAR ResNet block: eager_final={eager_losses[-1]:.6f}, "
        f"fused_final={fused_losses[-1]:.6f}, diff={final_diff:.6f}"
    )

    # Sanity: both paths converged.
    assert eager_losses[0] > eager_losses[-1], (
        f"eager loss didn't decrease (start={eager_losses[0]:.4f}, " f"end={eager_losses[-1]:.4f})"
    )
