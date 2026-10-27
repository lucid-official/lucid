"""Materialized end-to-end timings shared by the development benchmarks.

These include a host observation and synchronization, not isolated kernel time.
Callers must return their output (or materialize it themselves), and keep model
construction, transfer and correctness checks outside the timed callable.
"""

from collections.abc import Callable
import statistics
import time

import lucid


def materialize(output: object) -> None:
    if isinstance(output, lucid.Tensor):
        output.sum().item()
    elif isinstance(output, tuple | list):
        for value in output:
            materialize(value)
    elif isinstance(output, dict):
        for value in output.values():
            materialize(value)
    elif output is not None and not isinstance(output, bool | int | float | complex):
        for name in ("logits", "prediction", "last_hidden_state"):
            value = getattr(output, name, None)
            if isinstance(value, lucid.Tensor):
                materialize(value)
                return
        raise TypeError(f"benchmark output needs explicit materialization: {type(output).__name__}")


def cold_ms(call: Callable[[], object]) -> float:
    lucid.metal.synchronize()
    start = time.perf_counter()
    materialize(call())
    lucid.metal.synchronize()
    return (time.perf_counter() - start) * 1000


def warm_ms(call: Callable[[], object], n_warmup: int, n_iter: int) -> tuple[float, float, float]:
    if n_warmup < 0 or n_iter < 1:
        raise ValueError("warmup must be nonnegative and iterations must be positive")
    for _ in range(n_warmup):
        materialize(call())
    lucid.metal.synchronize()
    samples = sorted(cold_ms(call) for _ in range(n_iter))
    return (
        statistics.median(samples),
        samples[min(n_iter - 1, int(n_iter * 0.05))],
        samples[min(n_iter - 1, int(n_iter * 0.95))],
    )
