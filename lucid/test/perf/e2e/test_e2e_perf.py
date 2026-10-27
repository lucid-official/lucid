"""End-to-end perf — full forward+backward of representative models."""

from pathlib import Path

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.test._fixtures.perf import assert_no_regression, load_thresholds

_THRESHOLDS = load_thresholds(Path(__file__).parent)


@pytest.mark.perf
class TestMLPForwardBackward:
    def test_mlp_step(self, bench, device: str) -> None:
        x = lucid.tensor(np.random.standard_normal((64, 32)).astype(np.float32), device=device)
        y = lucid.tensor(np.random.standard_normal((64, 1)).astype(np.float32), device=device)
        model = nn.Sequential(
            nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1)
        ).to(device=device)

        def step() -> tuple[lucid.Tensor, list[lucid.Tensor | None]]:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad = None
            loss = F.mse_loss(model(x), y)
            loss.backward()
            return loss, [p.grad for p in model.parameters()]

        bench(step)
        if hasattr(bench, "last_elapsed"):
            assert_no_regression(f"mlp_step_{device}", bench.last_elapsed, _THRESHOLDS)


@pytest.mark.perf
class TestTransformerForward:
    def test_block_forward(self, bench, device: str) -> None:
        x = lucid.tensor(np.random.standard_normal((4, 32, 64)).astype(np.float32), device=device)
        block = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, batch_first=True
        ).to(device=device)

        bench(lambda: block(x))
        if hasattr(bench, "last_elapsed"):
            assert_no_regression(f"transformer_block_fwd_{device}", bench.last_elapsed, _THRESHOLDS)
