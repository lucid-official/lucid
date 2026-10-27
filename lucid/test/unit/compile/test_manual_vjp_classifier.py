"""Manual VJP — classification-head acceptance test (P1: CE/NLL VJPs).

Exercises ``cross_entropy_loss`` + ``nll_loss`` VJPs on a small
classifier pipeline.  This is the canonical transformer-LM /
image-classifier training loss; without these VJPs ``fused_step`` on
a CE-loss model would fall through to MPSGraph autograd (which is
exactly the path we want to avoid for transformer training).

Network:
  x (B, D) → Linear(D, C) → CE-loss against integer targets

Backward path under ``LUCID_MANUAL_VJP_REQUIRE=1`` exercises:
linear VJP → cross_entropy_loss VJP (softmax+sparse-onehot
combination).

Pass criterion: 5 SGD steps within 1e-3 of eager.
"""

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid.compile import fused_step

from lucid.test.unit.compile._helpers import COMPILE_DEVICE, metal_tensor

BATCH = 8
DIM = 16
N_CLASSES = 5


class _Classifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(DIM, N_CLASSES)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fc(x)


def _ce_loss(logits: lucid.Tensor, target: lucid.Tensor) -> lucid.Tensor:
    return F.cross_entropy(logits, target)


def _matched_models() -> tuple[nn.Module, nn.Module]:
    lucid.manual_seed(0)
    a = _Classifier().to(COMPILE_DEVICE)
    b = _Classifier().to(COMPILE_DEVICE)
    for (_, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        with lucid.no_grad():
            pb.copy_(pa.detach().clone())
    return a, b


@pytest.fixture(autouse=True)
def _manual_vjp_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LUCID_MANUAL_VJP", "1")
    monkeypatch.setenv("LUCID_MANUAL_VJP_REQUIRE", "1")


def test_manual_vjp_cross_entropy_parity() -> None:
    """Linear + CE-loss: manual VJP matches eager within 1e-3 over 5 SGD steps."""
    lucid.manual_seed(0)
    x = metal_tensor(BATCH, DIM)
    target = lucid.randint(0, N_CLASSES, size=(BATCH,)).to(COMPILE_DEVICE)
    eager_model, comp_model = _matched_models()

    # Eager trajectory.
    opt_eager = optim.SGD(list(eager_model.parameters()), lr=1e-2)
    eager_losses: list[float] = []
    for _ in range(5):
        opt_eager.zero_grad()
        loss = _ce_loss(eager_model(x), target)
        loss.backward()
        opt_eager.step()
        eager_losses.append(float(loss.item()))

    # Compile trajectory.
    opt_comp = optim.SGD(list(comp_model.parameters()), lr=1e-2)
    step = fused_step(comp_model, _ce_loss, opt_comp)
    comp_losses: list[float] = []
    for _ in range(5):
        loss = step(x, target)
        comp_losses.append(float(loss.item() if hasattr(loss, "item") else loss[0].item()))

    for k in range(5):
        diff = abs(eager_losses[k] - comp_losses[k])
        assert diff < 1e-3, (
            f"CE manual VJP drift at step {k}: "
            f"eager={eager_losses[k]:.6f}, compile={comp_losses[k]:.6f}, diff={diff:.6f}"
        )
