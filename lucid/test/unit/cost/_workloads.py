"""What a training step costs, counted rather than timed.

A wall-clock gate on a shared runner measures the runner.  These count what
the code decides: storages allocated and freed, the peak they reach, each
forward op dispatched, host storages created while the step runs on the
GPU, and — for a compiled step — the executables built and the signatures
sent back to eager.  All are the same on every run of the same code, on any
machine, so the budget is exact: :mod:`.test_step_cost` compares each one
with ``step_cost.json`` and fails on any difference, up or down.
"""

import collections
import gc
from collections.abc import Callable
from dataclasses import dataclass

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid._C import engine as _C_engine

WARMUP = 3

_DEVICE = {"cpu": _C_engine.Device.CPU, "metal": _C_engine.Device.GPU}


@dataclass(frozen=True)
class Workload:
    name: str
    build: Callable[[str], Callable[[], object]]
    devices: tuple[str, ...] = ("cpu", "metal")


def _mlp_adam(device: str) -> Callable[[], object]:
    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10)).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    x = lucid.randn(16, 32, device=device)
    y = lucid.randint(0, 10, (16,), device=device)

    def step() -> object:
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        return loss.item()

    return step


def _cnn(device: str) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(8 * 8 * 8, 10),
    ).to(device)


def _cnn_sgd(device: str) -> Callable[[], object]:
    model = _cnn(device)
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    x = lucid.randn(4, 3, 16, 16, device=device)
    y = lucid.randint(0, 10, (4,), device=device)

    def step() -> object:
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        return loss.item()

    return step


def _cnn_inference(device: str) -> Callable[[], object]:
    model = _cnn(device).eval()
    x = lucid.randn(4, 3, 16, 16, device=device)

    def step() -> object:
        with lucid.no_grad():
            return model(x).sum().item()

    return step


def _transformer_adamw(device: str) -> Callable[[], object]:
    layer = nn.TransformerEncoderLayer(32, 4, dim_feedforward=64, dropout=0.0, batch_first=True).to(
        device
    )
    opt = optim.AdamW(layer.parameters(), lr=1e-3)
    x = lucid.randn(2, 8, 32, device=device)
    target = lucid.randn(2, 8, 32, device=device)

    def step() -> object:
        opt.zero_grad()
        loss = F.mse_loss(layer(x), target)
        loss.backward()
        opt.step()
        return loss.item()

    return step


def _recurrent_adam(cls: type[nn.LSTM] | type[nn.GRU], device: str) -> Callable[[], object]:
    rnn = cls(16, 32, batch_first=True).to(device)
    head = nn.Linear(32, 1).to(device)
    opt = optim.Adam(list(rnn.parameters()) + list(head.parameters()), lr=1e-3)
    x = lucid.randn(2, 6, 16, device=device)
    target = lucid.randn(2, 1, device=device)

    def step() -> object:
        opt.zero_grad()
        out, _ = rnn(x)
        loss = F.mse_loss(head(out[:, -1]), target)
        loss.backward()
        opt.step()
        return loss.item()

    return step


def _lstm_adam(device: str) -> Callable[[], object]:
    return _recurrent_adam(nn.LSTM, device)


def _gru_adam(device: str) -> Callable[[], object]:
    # GRU and RNN step a cell per timestep rather than making one engine
    # call, so a cost paid per step — a whole-batch slice that became a
    # view of its own — shows up here once per timestep and is invisible
    # in the LSTM workload.
    return _recurrent_adam(nn.GRU, device)


def _compiled_mlp_adam(device: str) -> Callable[[], object]:
    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10)).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    compiled = lucid.compile.make_step(model, lambda out, t: F.cross_entropy(out, t))
    x = lucid.randn(16, 32, device=device)
    y = lucid.randint(0, 10, (16,), device=device)

    def step() -> object:
        opt.zero_grad()
        loss = compiled(x, y)
        loss.backward()
        opt.step()
        return loss.item()

    step.compiled = compiled  # type: ignore[attr-defined]
    return step


WORKLOADS: tuple[Workload, ...] = (
    Workload("mlp_adam", _mlp_adam),
    Workload("cnn_sgd", _cnn_sgd),
    Workload("cnn_inference", _cnn_inference),
    Workload("transformer_adamw", _transformer_adamw),
    Workload("lstm_adam", _lstm_adam),
    Workload("gru_adam", _gru_adam),
    Workload("compiled_mlp_adam", _compiled_mlp_adam, devices=("metal",)),
)


def measure(workload: Workload, device: str) -> dict[str, object]:
    """One step's counts, taken after :data:`WARMUP` steps have run."""
    lucid.manual_seed(0)
    step = workload.build(device)
    for _ in range(WARMUP):
        step()
    here = _DEVICE[device]
    # The cycle collector runs when its allocation counters say so, and
    # they depend on everything the process did before — so in a long test
    # run it sometimes swept the earlier steps' garbage *during* the
    # measured one: 71 extra frees and a peak below the starting point,
    # once in five nightly runs.  Collect first, and keep it out of the
    # step.
    gc.collect()
    collecting = gc.isenabled()
    gc.disable()
    try:
        _C_engine.reset_peak_memory_stats(here)
        before = _C_engine.memory_stats(here)
        host_before = _C_engine.memory_stats(_C_engine.Device.CPU)
        syncs_before = _C_engine.host_sync_count()
        with lucid.profiler.profile() as prof:
            step()
        after = _C_engine.memory_stats(here)
        host_after = _C_engine.memory_stats(_C_engine.Device.CPU)
        syncs_after = _C_engine.host_sync_count()
    finally:
        if collecting:
            gc.enable()
    # The backward pass and each node it runs are events too, marked as
    # backward work; the budget counts the ops the step dispatched.
    ops = collections.Counter(event.name for event in prof.events() if not event.backward)
    cost: dict[str, object] = {
        "allocations": after.alloc_count - before.alloc_count,
        "frees": after.free_count - before.free_count,
        "peak_bytes": after.peak_bytes - before.current_bytes,
        "forward_ops": dict(sorted(ops.items())),
        # Each read of an evaluated array by the host waits for the lazy
        # graph behind it: a loss read with ``.item()`` every step, or an
        # op that falls back to the CPU, stalls the pipeline without
        # allocating anything the counts above would see.
        "host_syncs": syncs_after - syncs_before,
    }
    if device == "metal":
        # A storage made on the host while the step runs on the GPU is a
        # round trip — a CPU fallback or a copy nothing asked for.
        cost["host_allocations"] = host_after.alloc_count - host_before.alloc_count
    compiled = getattr(step, "compiled", None)
    if compiled is not None:
        cost["executables"] = len(compiled.cache)
        cost["eager_fallbacks"] = len(compiled.eager_only.snapshot())
    return cost


def measure_all() -> dict[str, dict[str, object]]:
    return {f"{w.name}/{device}": measure(w, device) for w in WORKLOADS for device in w.devices}
