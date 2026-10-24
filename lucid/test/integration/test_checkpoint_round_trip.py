"""Save → load → resume training: parameters survive intact and the
loss curve continues from where it left off."""

from pathlib import Path

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim


def _build() -> tuple[nn.Sequential, optim.Optimizer, lucid.Tensor, lucid.Tensor]:
    rng = np.random.default_rng(0)
    x = lucid.tensor(rng.uniform(-1.0, 1.0, size=(16, 4)).astype(np.float32))
    y = lucid.tensor(rng.uniform(-1.0, 1.0, size=(16, 1)).astype(np.float32))
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    opt = optim.Adam(model.parameters(), lr=0.05)
    return model, opt, x, y


@pytest.mark.slow
class TestCheckpointRoundTrip:
    def test_state_dict_round_trip(self, tmp_path: Path) -> None:
        model, opt, x, y = _build()

        # 5 warm-up steps.
        for _ in range(5):
            opt.zero_grad()
            F.mse_loss(model(x), y).backward()
            opt.step()
        loss_before = float(F.mse_loss(model(x), y).item())

        ckpt = tmp_path / "ckpt.lcd"
        lucid.save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
            },
            str(ckpt),
        )

        # Build fresh, load.
        model2, opt2, _, _ = _build()
        loaded = lucid.load(str(ckpt), weights_only=False)
        model2.load_state_dict(loaded["model"])
        opt2.load_state_dict(loaded["opt"])

        # After load, identical model+optim must reproduce identical loss.
        loss_after = float(F.mse_loss(model2(x), y).item())
        assert abs(loss_before - loss_after) < 1e-5

        # And training continues from there as if it had never stopped: the
        # resumed pair takes the uninterrupted pair's next 20 steps.  "The
        # loss did not rise" asked the initialisation a question, whose answer
        # moved with every draw before it (LCD-615); this asks the checkpoint.
        for _ in range(20):
            losses = []
            for module, optimizer in ((model, opt), (model2, opt2)):
                optimizer.zero_grad()
                loss = F.mse_loss(module(x), y)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
            assert losses[1] == pytest.approx(losses[0], rel=1e-6, abs=1e-7)


@pytest.mark.parametrize("kind", ["SGD", "Adam", "AdamW"])
def test_checkpoint_resume_reproduces_next_updates(tmp_path: Path, kind: str) -> None:
    """A decreasing loss alone cannot detect a lost momentum/variance buffer."""
    lucid.manual_seed(17)
    model, _, x, y = _build()
    options = {"lr": 0.01}
    if kind == "SGD":
        options["momentum"] = 0.9
    constructor = getattr(optim, kind)
    optimizer = constructor(model.parameters(), **options)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.7)

    def step(module, opt, schedule) -> float:
        opt.zero_grad()
        loss = F.mse_loss(module(x), y)
        loss.backward()
        opt.step()
        schedule.step()
        return float(loss.item())

    for _ in range(3):
        step(model, optimizer, scheduler)
    checkpoint = tmp_path / "resume.lcd"
    lucid.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        str(checkpoint),
    )
    restored, _, _, _ = _build()
    restored_optimizer = constructor(restored.parameters(), **options)
    restored_scheduler = optim.lr_scheduler.StepLR(
        restored_optimizer, step_size=2, gamma=0.7
    )
    # Optimizer state contains arrays, not just Tensor payloads. This file
    # was written above in this test; arbitrary external checkpoints are not trusted.
    with pytest.warns(UserWarning, match="arbitrary object deserialization"):
        loaded = lucid.load(str(checkpoint), weights_only=False)
    restored.load_state_dict(loaded["model"])
    restored_optimizer.load_state_dict(loaded["optimizer"])
    restored_scheduler.load_state_dict(loaded["scheduler"])

    for _ in range(4):
        assert step(restored, restored_optimizer, restored_scheduler) == pytest.approx(
            step(model, optimizer, scheduler),
            abs=1e-7,
            rel=1e-6,
        )
        assert (
            restored_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
        )
        for original, resumed in zip(
            model.parameters(), restored.parameters(), strict=True
        ):
            np.testing.assert_allclose(
                resumed.numpy(), original.numpy(), atol=1e-7, rtol=1e-6
            )
