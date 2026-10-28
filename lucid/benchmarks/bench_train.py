"""
benchmarks/bench_train.py — end-to-end training loop throughput.

Model: MLP(256 → 512 → 256 → 128 → 10) with ReLU activations.
Measures:
  · Forward pass
  · Forward + backward pass
  · Full step (forward + backward + SGD optimizer)

Runs on both CPU and GPU (if available).
Reports ms/step and samples/sec for batch sizes [64, 256, 1024].
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from typing import override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid.benchmarks._core import BenchResult, fmt_table, metal_available


class _MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)

    @override
    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        x = lucid.relu(self.fc1(x))
        x = lucid.relu(self.fc2(x))
        x = lucid.relu(self.fc3(x))
        return self.fc4(x)


def _sync(tensor: lucid.Tensor) -> None:
    """Force GPU synchronisation for Metal tensors; no-op for CPU."""
    if tensor.is_metal:
        from lucid._C import engine as _C_engine

        _C_engine.eval_gpu(tensor._impl)


def _bench_step(
    model: _MLP,
    opt: optim.SGD,
    x: lucid.Tensor,
    y: lucid.Tensor,
    *,
    warmup: int,
    iters: int,
    mode: str,
) -> BenchResult:
    """mode: \"forward\" | \"forward_backward\" | \"full_step\"."""
    for _ in range(warmup):
        out = model.forward(x)
        if mode in ("forward_backward", "full_step"):
            loss = F.cross_entropy(out, y)
            loss.backward()
        if mode == "full_step":
            opt.step()
            opt.zero_grad()
        _sync(out)

    times: list[int] = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        out = model.forward(x)
        if mode in ("forward_backward", "full_step"):
            loss = F.cross_entropy(out, y)
            loss.backward()
        if mode == "full_step":
            opt.step()
            opt.zero_grad()
        _sync(out)
        times.append(time.perf_counter_ns() - t0)
    return BenchResult(mode, times)


def _bench_device(device: str, batch_size: int, iters: int) -> dict[str, object]:
    model = _MLP()
    if device == "metal":
        model = model.to("metal")
    opt = optim.SGD(model.parameters(), lr=0.01)

    x = lucid.randn(batch_size, 256)
    y = lucid.randint(0, 10, (batch_size,))
    if device == "metal":
        x = x.to("metal")
        y = y.to("metal")

    results: dict[str, object] = {}
    for mode in ("forward", "forward_backward", "full_step"):
        br = _bench_step(model, opt, x, y, warmup=3, iters=iters, mode=mode)
        mean_ms = br.mean_us / 1_000.0
        samples_per_sec = batch_size / (br.mean_us * 1e-6)
        results[f"{mode}_mean_ms"] = round(mean_ms, 3)
        results[f"{mode}_p95_ms"] = round(br.p95_us / 1_000.0, 3)
        results[f"{mode}_samples_sec"] = round(samples_per_sec, 1)
    return results


def run(verbose: bool = True) -> dict[str, object]:
    if verbose:
        print("\n── Training loop (MLP 256→512→256→128→10) ──────────────────────")

    has_gpu = metal_available()
    batch_sizes = [64, 256, 1024]
    iters_map = {64: 50, 256: 30, 1024: 20}

    out: dict[str, object] = {}
    header = ("batch", "device", "fwd ms", "fwd+bwd ms", "step ms", "samples/s")
    rows: list[tuple[str, ...]] = [header]

    for bs in batch_sizes:
        iters = iters_map[bs]
        for device in ["cpu"] + (["metal"] if has_gpu else []):
            r = _bench_device(device, bs, iters)
            key = f"train/bs{bs}/{device}"
            for k, v in r.items():
                out[f"{key}/{k}"] = v

            rows.append(
                (
                    str(bs),
                    device,
                    f"{r['forward_mean_ms']:.2f}",
                    f"{r['forward_backward_mean_ms']:.2f}",
                    f"{r['full_step_mean_ms']:.2f}",
                    f"{r['full_step_samples_sec']:.0f}",
                )
            )

    if verbose:
        print(fmt_table(rows))

    return out


if __name__ == "__main__":
    run()
