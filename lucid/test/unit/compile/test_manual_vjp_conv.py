"""Manual VJP — Conv2d acceptance (P4).

Exercises conv2d VJP using MPSGraph's
``convolution2DDataGradient`` / ``convolution2DWeightsGradient`` APIs.
"""

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid.compile import fused_step

from lucid.test.unit.compile._helpers import COMPILE_DEVICE, metal_tensor


class _ConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.fc = nn.Linear(4, 1)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        h = self.conv(x).relu()
        h = h.mean(dim=(2, 3), keepdim=False)  # (B, 4)
        return self.fc(h)


def _loss(pred: lucid.Tensor, tgt: lucid.Tensor) -> lucid.Tensor:
    return F.mse_loss(pred, tgt)


def _matched_models() -> tuple[nn.Module, nn.Module]:
    lucid.manual_seed(0)
    a = _ConvNet().to(COMPILE_DEVICE)
    b = _ConvNet().to(COMPILE_DEVICE)
    for (_, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        with lucid.no_grad():
            pb.copy_(pa.detach().clone())
    return a, b


@pytest.fixture(autouse=True)
def _manual_vjp_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LUCID_MANUAL_VJP", "1")
    monkeypatch.setenv("LUCID_MANUAL_VJP_REQUIRE", "1")


def test_manual_vjp_conv2d_parity() -> None:
    """Conv2d + ReLU + mean + Linear + MSE: manual VJP matches eager within 2e-3."""
    lucid.manual_seed(0)
    x = metal_tensor(2, 3, 4, 4)
    t = metal_tensor(2, 1)
    eager_model, comp_model = _matched_models()

    opt_eager = optim.SGD(list(eager_model.parameters()), lr=1e-2)
    eager_losses: list[float] = []
    for _ in range(3):
        opt_eager.zero_grad()
        loss = _loss(eager_model(x), t)
        loss.backward()
        opt_eager.step()
        eager_losses.append(float(loss.item()))

    opt_comp = optim.SGD(list(comp_model.parameters()), lr=1e-2)
    step = fused_step(comp_model, _loss, opt_comp)
    comp_losses: list[float] = []
    for _ in range(3):
        loss = step(x, t)
        comp_losses.append(float(loss.item() if hasattr(loss, "item") else loss[0].item()))

    for k in range(3):
        diff = abs(eager_losses[k] - comp_losses[k])
        assert diff < 2e-3, (
            f"Conv2d manual VJP drift at step {k}: "
            f"eager={eager_losses[k]:.6f}, compile={comp_losses[k]:.6f}, diff={diff:.6f}"
        )
