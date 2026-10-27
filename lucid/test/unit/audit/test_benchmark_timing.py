"""Timers must submit lazy outputs before stopping the clock."""

import pytest
import subprocess
import sys

import lucid
from tools import _bench_timing as timing
from tools import bench_compile


def test_cold_timer_materializes_before_final_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    events = []
    monkeypatch.setattr(timing.lucid.metal, "synchronize", lambda: events.append("sync"))
    monkeypatch.setattr(timing, "materialize", lambda value: events.append(("value", value)))
    ticks = iter([1.0, 1.25])
    monkeypatch.setattr(timing.time, "perf_counter", lambda: next(ticks))
    assert timing.cold_ms(lambda: 42) == 250
    assert events == ["sync", ("value", 42), "sync"]


def test_each_warmup_and_sample_observes_output(monkeypatch: pytest.MonkeyPatch) -> None:
    observed = []
    monkeypatch.setattr(timing, "materialize", observed.append)
    timing.warm_ms(lambda: 42, 2, 3)
    assert observed == [42] * 5


def test_unknown_output_cannot_earn_a_timing() -> None:
    with pytest.raises(TypeError, match="materialization"):
        timing.cold_ms(object)
    with pytest.raises(ValueError, match="iterations"):
        timing.warm_ms(lambda: None, 0, 0)


def test_compile_benchmark_calls_the_public_module_entrypoint() -> None:
    result = bench_compile._bench_case(
        "small-linear", lambda: lucid.nn.Linear(4, 2), lambda: lucid.ones(2, 4), 2
    )
    assert result["note"] == "ok"
    assert result["eager_ms"] > 0
    assert result["compile_ms"] > 0


def test_training_benchmark_rejects_a_step_that_never_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_fused_step(model, loss_fn, optimizer):
        return lambda x, target: loss_fn(model(x), target)

    monkeypatch.setattr(bench_compile, "fused_step", broken_fused_step)
    result = bench_compile._bench_training_case(
        "broken", lambda: lucid.nn.Linear(4, 2), lambda: (lucid.ones(2, 4), lucid.zeros(2, 2)), 2
    )
    assert result["note"] == "parity-diverged"


def test_precision_benchmark_casts_buffers_but_not_index_inputs() -> None:
    # Isolate benchmark imports and its model/compile global state from pytest.
    code = """
import lucid
from tools.bench_compile_vs_eager import _cast_model, _cast_inputs
model = lucid.nn.BatchNorm1d(4)
parameters = list(model.parameters())
_cast_model(model, lucid.float16)
assert all(p.dtype == lucid.float16 for p in model.parameters())
assert all(a is b for a, b in zip(parameters, model.parameters()))
assert model.running_mean.dtype == lucid.float16
assert model.num_batches_tracked.dtype == lucid.int64
indices = lucid.tensor([1, 2], dtype=lucid.int16)
mask = lucid.tensor([True, False], dtype=lucid.bool)
got = _cast_inputs((indices, mask), lucid.float16)
assert got[0] is indices and got[1] is mask
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


def test_autocast_benchmark_checks_updates_and_propagates_failure(tmp_path) -> None:
    code = """
import sys
import lucid
from tools import bench_autocast as bench
factory = lambda: lucid.nn.Linear(4, 2)
inputs = lambda n: lucid.ones(n, 4)
targets = lambda n: lucid.zeros(n, 2)
bench._bench_fused_step(factory, inputs, targets, 2, autocast=True, n_iter=1)
bench.fused_step = lambda model, loss_fn, opt: lambda x, y: loss_fn(model(x), y)
try:
    bench._bench_fused_step(factory, inputs, targets, 2, autocast=False, n_iter=1)
except RuntimeError as exc:
    assert 'training parity diverged' in str(exc)
else:
    raise AssertionError('an unchanged model earned a training timing')
bench.WORKLOADS = [('broken', factory, inputs, targets)]
sys.argv = ['bench', '--batch', '2', '--iter', '1', '--csv', sys.argv[1]]
assert bench.main() == 1
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "broken.csv")],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
