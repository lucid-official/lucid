"""Manual VJP — BatchNorm acceptance (P2).

Exercises BN train + eval VJPs in a tiny conv-less block:
  x (B, C, H, W) → BatchNorm(C, train) → ReLU → mean → Linear → MSE
"""

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid.compile import fused_step

from lucid.test.unit.compile._helpers import COMPILE_DEVICE, metal_tensor

B, C, H, W = 2, 4, 3, 3


class _BNNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(C)
        self.fc = nn.Linear(C, 1)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        h = self.bn(x).relu()
        h = h.mean(dim=(2, 3), keepdim=False)  # (B, C)
        return self.fc(h)


def _loss(pred: lucid.Tensor, tgt: lucid.Tensor) -> lucid.Tensor:
    return F.mse_loss(pred, tgt)


def _matched_models() -> tuple[nn.Module, nn.Module]:
    lucid.manual_seed(0)
    a = _BNNet().to(COMPILE_DEVICE)
    b = _BNNet().to(COMPILE_DEVICE)
    for (_, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        with lucid.no_grad():
            pb.copy_(pa.detach().clone())
    return a, b


@pytest.fixture(autouse=True)
def _manual_vjp_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LUCID_MANUAL_VJP", "1")
    monkeypatch.setenv("LUCID_MANUAL_VJP_REQUIRE", "1")


def test_manual_vjp_batchnorm_train_parity() -> None:
    """BatchNorm train + Linear + MSE: manual VJP matches eager within 1e-3."""
    lucid.manual_seed(0)
    x = metal_tensor(B, C, H, W)
    t = metal_tensor(B, 1)
    eager_model, comp_model = _matched_models()

    opt_eager = optim.SGD(list(eager_model.parameters()), lr=1e-2)
    eager_losses: list[float] = []
    for _ in range(5):
        opt_eager.zero_grad()
        loss = _loss(eager_model(x), t)
        loss.backward()
        opt_eager.step()
        eager_losses.append(float(loss.item()))

    opt_comp = optim.SGD(list(comp_model.parameters()), lr=1e-2)
    step = fused_step(comp_model, _loss, opt_comp)
    comp_losses: list[float] = []
    for _ in range(5):
        loss = step(x, t)
        comp_losses.append(float(loss.item() if hasattr(loss, "item") else loss[0].item()))

    for k in range(5):
        diff = abs(eager_losses[k] - comp_losses[k])
        assert diff < 1e-3, (
            f"BN-train manual VJP drift at step {k}: "
            f"eager={eager_losses[k]:.6f}, compile={comp_losses[k]:.6f}, diff={diff:.6f}"
        )
