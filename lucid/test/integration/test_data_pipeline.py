"""End-to-end data ingest → train: ``Dataset`` + ``DataLoader`` feed
a model and the loss decreases."""

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid.utils.data import DataLoader, Dataset


class _ToyDataset(Dataset):
    def __init__(self, n: int = 64, d: int = 4) -> None:
        rng = np.random.default_rng(0)
        self.x = rng.uniform(-1.0, 1.0, size=(n, d)).astype(np.float32)
        self.y = (self.x.sum(axis=1, keepdims=True) > 0).astype(np.float32)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        return self.x[idx], self.y[idx]


@pytest.mark.slow
class TestDataPipeline:
    def test_loader_drives_training(self, device: str) -> None:
        ds = _ToyDataset()
        # The batch order has a stream of its own, so the model's
        # initialisation cannot reorder it (LCD-615).  The bound is far from
        # any draw: across 60 seeds the loss fell to at most 0.29 of its start.
        loader = DataLoader(
            ds, batch_size=8, shuffle=True, generator=lucid.Generator(0)
        )

        model = nn.Sequential(
            nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid()
        ).to(device=device)
        opt = optim.Adam(model.parameters(), lr=0.05)

        # Initial full-batch loss as the baseline.
        x_full = lucid.tensor(ds.x, device=device)
        y_full = lucid.tensor(ds.y, device=device)
        first = float(F.binary_cross_entropy(model(x_full), y_full).item())

        for _ in range(5):  # 5 epochs.
            for xb, yb in loader:
                xb_t = (
                    xb.to(device=device)
                    if isinstance(xb, lucid.Tensor)
                    else lucid.tensor(np.asarray(xb), device=device)
                )
                yb_t = (
                    yb.to(device=device)
                    if isinstance(yb, lucid.Tensor)
                    else lucid.tensor(np.asarray(yb), device=device)
                )
                opt.zero_grad()
                loss = F.binary_cross_entropy(model(xb_t), yb_t)
                loss.backward()
                opt.step()

        last = float(F.binary_cross_entropy(model(x_full), y_full).item())
        assert last < 0.7 * first
