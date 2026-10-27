"""End-to-end MLP training — verifies forward, backward, and the
optimizer move loss meaningfully on a synthetic toy problem.

The point of an integration test isn't to chase the global minimum —
it's to catch regressions that look fine in unit tests but break the
full forward+backward+step+zero_grad chain.  We assert that loss
shrinks by a clear margin (≥ 50 %) over a handful of steps.
"""

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim


@pytest.mark.slow
class TestMLPTraining:
    def _build_data(self, device: str) -> tuple[lucid.Tensor, lucid.Tensor]:
        rng = np.random.default_rng(0)
        x = rng.uniform(-1.0, 1.0, size=(32, 4)).astype(np.float32)
        # y = x · w + b with a known target so the model can fit it.
        w = np.array([[1.0, -1.0, 0.5, 2.0]], dtype=np.float32)
        y = x @ w.T + 0.3
        return (lucid.tensor(x, device=device), lucid.tensor(y, device=device))

    def test_loss_shrinks(self, device: str) -> None:
        x, y = self._build_data(device)

        model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1)).to(device=device)
        opt = optim.SGD(model.parameters(), lr=0.05)

        losses: list[float] = []
        for _ in range(30):
            opt.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        assert (
            losses[-1] < 0.5 * losses[0]
        ), f"loss did not shrink enough: {losses[0]:.4f} → {losses[-1]:.4f}"

    def test_adam_also_shrinks(self, device: str) -> None:
        x, y = self._build_data(device)
        model = nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 1)).to(device=device)
        opt = optim.Adam(model.parameters(), lr=0.05)

        first = float(F.mse_loss(model(x), y).item())
        for _ in range(30):
            opt.zero_grad()
            loss = F.mse_loss(model(x), y)
            loss.backward()
            opt.step()
        last = float(loss.item())
        assert last < 0.5 * first
