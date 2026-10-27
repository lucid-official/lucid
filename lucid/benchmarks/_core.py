"""
benchmarks/_core.py — shared timing primitives (stdlib only).

All timer functions return raw nanosecond samples.
Statistics (mean, median, p95) are computed by BenchResult.
"""

import statistics
import time
from typing import Callable, override

# ── defaults ──────────────────────────────────────────────────────────────────

WARMUP_CPU: int = 10
WARMUP_GPU: int = 10
ITERS_CPU: int = 200
ITERS_GPU: int = 50


# ── result container ──────────────────────────────────────────────────────────


class BenchResult:
    """Holds raw nanosecond samples and computes derived statistics."""

    def __init__(self, name: str, times_ns: list[int]) -> None:
        self.name = name
        self._ns = sorted(times_ns)

    @property
    def mean_us(self) -> float:
        return statistics.mean(self._ns) / 1_000.0

    @property
    def median_us(self) -> float:
        return statistics.median(self._ns) / 1_000.0

    @property
    def p95_us(self) -> float:
        idx = max(0, int(len(self._ns) * 0.95) - 1)
        return self._ns[idx] / 1_000.0

    @property
    def p99_us(self) -> float:
        idx = max(0, int(len(self._ns) * 0.99) - 1)
        return self._ns[idx] / 1_000.0

    def to_dict(self) -> dict[str, object]:
        return {
            "mean_us": round(self.mean_us, 3),
            "median_us": round(self.median_us, 3),
            "p95_us": round(self.p95_us, 3),
            "p99_us": round(self.p99_us, 3),
            "n_iter": len(self._ns),
        }

    @override
    def __repr__(self) -> str:
        return (
            f"BenchResult({self.name!r}  "
            f"mean={self.mean_us:.1f}µs  "
            f"median={self.median_us:.1f}µs  "
            f"p95={self.p95_us:.1f}µs)"
        )


# ── timer helpers ─────────────────────────────────────────────────────────────


def bench_cpu(
    fn: Callable[[], object],
    warmup: int = WARMUP_CPU,
    iters: int = ITERS_CPU,
) -> list[int]:
    """Time a CPU-synchronous callable. Returns nanosecond samples."""
    for _ in range(warmup):
        fn()
    times: list[int] = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        times.append(time.perf_counter_ns() - t0)
    return times


def bench_gpu_lucid(
    fn: Callable[[], object],
    warmup: int = WARMUP_GPU,
    iters: int = ITERS_GPU,
) -> list[int]:
    """Time a Lucid GPU op, forcing MLX eval after each call.

    ``fn`` must return a single Tensor (not a tuple).
    Timing includes kernel dispatch + GPU synchronisation.

    Uses ``eval_gpu(impl)`` — the single-tensor fast path — rather than
    ``eval_tensors([impl])`` to avoid Python list creation overhead (~25 µs)
    that would skew the comparison against raw ``mx.eval(arr)``.
    """
    from lucid._C import engine as _C_engine

    def _run() -> None:
        out = fn()
        impl = getattr(out, "_impl", out)
        _C_engine.eval_gpu(impl)  # type: ignore[arg-type]

    for _ in range(warmup):
        _run()
    times: list[int] = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        _run()
        times.append(time.perf_counter_ns() - t0)
    return times


def bench_gpu_mlx(
    fn: Callable[[], object],
    warmup: int = WARMUP_GPU,
    iters: int = ITERS_GPU,
) -> list[int]:
    """Time a raw MLX op, forcing eval after each call.

    ``fn`` must return a single mlx.core.array.
    Timing includes kernel dispatch + GPU synchronisation.
    This provides a lower-bound reference: the cost if Lucid's Python
    layer had zero overhead.
    """
    import mlx.core as mx

    def _run() -> None:
        out = fn()
        mx.eval(out)

    for _ in range(warmup):
        _run()
    times: list[int] = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        _run()
        times.append(time.perf_counter_ns() - t0)
    return times


# ── metal availability guard ──────────────────────────────────────────────────


def metal_available() -> bool:
    import lucid.metal as _m

    return _m.is_available()


# ── formatting helpers ────────────────────────────────────────────────────────


def fmt_table(rows: list[tuple[str, ...]]) -> str:
    """Format a list of tuples as a fixed-width ASCII table."""
    if not rows:
        return ""
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    lines = [sep]
    for i, row in enumerate(rows):
        line = "|" + "|".join(f" {cell:<{widths[j]}} " for j, cell in enumerate(row)) + "|"
        lines.append(line)
        if i == 0:  # header separator
            lines.append(sep)
    lines.append(sep)
    return "\n".join(lines)
