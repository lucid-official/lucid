"""End-to-end AMP training — autocast + GradScaler keep the loss
shrinking on the same toy problem the MLP test uses."""

import numpy as np
import pytest

import lucid
import lucid.amp as amp
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim


@pytest.mark.slow
class TestAMPTraining:
    def test_autocast_loss_shrinks(self, device: str) -> None:
        rng = np.random.default_rng(0)
        x = lucid.tensor(rng.uniform(-1.0, 1.0, size=(32, 4)).astype(np.float32), device=device)
        y = lucid.tensor(rng.uniform(-1.0, 1.0, size=(32, 1)).astype(np.float32), device=device)

        model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1)).to(device=device)
        opt = optim.Adam(model.parameters(), lr=0.05)

        first = float(F.mse_loss(model(x), y).item())
        for _ in range(50):
            opt.zero_grad()
            with amp.autocast(device_type=device):
                out = model(x)
                # 3.3 AMP fix: under autocast on metal the model output is
                # F16 (Promote policy on Linear took effect — before the fix
                # autocast was silently a no-op).  Loss functions are
                # numerically sensitive and run in F32 by convention; cast
                # the output explicitly so the F32 target dtype matches.
                out_f32 = out.float() if out.dtype != lucid.float32 else out
                loss = F.mse_loss(out_f32, y)
            loss.backward()
            opt.step()
        last = float(loss.item())
        # AMP should not prevent a meaningful drop on this toy problem.
        assert last < 0.5 * first
