"""
benchmarks/bench_transfer.py — CPU↔GPU device transfer latency.

Measures .to("metal") and .to("cpu") at four tensor sizes:
  1 KB / 64 KB / 1 MB / 64 MB

Also compares the two transfer paths:
  · SharedStorage path (lucid.metal.to_shared → .to("metal"))   — allocate once;
    every .to() after that relabels the same buffer, no copy
  · Direct .to("metal")                                          — the default:
    one copy into a GPU-private buffer, which later GPU ops read faster

Every size takes the same path; there is no size threshold.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lucid
import lucid.metal as metal
from lucid.benchmarks._core import BenchResult, bench_cpu, metal_available

# ── size presets ──────────────────────────────────────────────────────────────

_SIZES: list[tuple[str, int]] = [
    ("1 KB", 1 * 1024 // 4),  # 256 float32 elements
    ("64 KB", 64 * 1024 // 4),  # 16 384 elements
    ("1 MB", 1 * 1024 * 1024 // 4),  # 262 144 elements
    ("64 MB", 64 * 1024 * 1024 // 4),  # 16 777 216 elements
]


def _gbps(n_elements: int, mean_us: float) -> float:
    """Effective bandwidth in GB/s."""
    nbytes = n_elements * 4  # float32
    return nbytes / (mean_us * 1e-6) / 1e9


def _bench_size(label: str, n: int, iters: int) -> dict[str, object]:
    has_gpu = metal_available()
    r: dict[str, object] = {"size": label, "n_elements": n}

    cpu_tensor = lucid.randn(n)

    # ── CPU → GPU (direct .to("metal")) ──────────────────────────────────────
    if has_gpu:
        from lucid._C import engine as _C_engine

        def _to_metal_direct() -> None:
            t = cpu_tensor.to("metal")
            _C_engine.eval_tensors([t._impl])

        ns = bench_cpu(_to_metal_direct, warmup=5, iters=iters)
        br = BenchResult(f"cpu_to_gpu_{label}", ns)
        r["cpu_to_gpu_mean_us"] = br.mean_us
        r["cpu_to_gpu_p95_us"] = br.p95_us
        r["cpu_to_gpu_gbps"] = _gbps(n, br.mean_us)

        # ── Already-shared tensor → GPU (zero-copy path) ──────────────────
        # Promote once to SharedStorage (one-time cost, not measured here),
        # then measure the repeated transfer cost — this is the steady-state
        # latency for a tensor that lives in shared memory across iterations.
        already_shared = metal.to_shared(cpu_tensor)

        def _to_metal_shared() -> None:
            t = already_shared.to("metal")
            _C_engine.eval_tensors([t._impl])

        ns_s = bench_cpu(_to_metal_shared, warmup=5, iters=iters)
        br_s = BenchResult(f"cpu_to_gpu_shared_{label}", ns_s)
        r["shared_to_gpu_mean_us"] = br_s.mean_us
        r["shared_to_gpu_p95_us"] = br_s.p95_us
        r["shared_to_gpu_gbps"] = _gbps(n, br_s.mean_us)

        # Speedup: direct (includes Metal allocation) vs shared (zero-copy).
        speedup = br.mean_us / max(br_s.mean_us, 0.001)
        r["shared_speedup_x"] = round(speedup, 2)

        # ── GPU → CPU ─────────────────────────────────────────────────────
        gpu_tensor = cpu_tensor.to("metal")

        def _to_cpu() -> None:
            t = gpu_tensor.to("cpu")
            _ = t.numpy()  # force eval

        ns_c = bench_cpu(_to_cpu, warmup=5, iters=iters)
        br_c = BenchResult(f"gpu_to_cpu_{label}", ns_c)
        r["gpu_to_cpu_mean_us"] = br_c.mean_us
        r["gpu_to_cpu_p95_us"] = br_c.p95_us
        r["gpu_to_cpu_gbps"] = _gbps(n, br_c.mean_us)

    return r


def run(verbose: bool = True) -> dict[str, object]:
    if not metal_available():
        if verbose:
            print("\n── Transfer benchmark: Metal not available, skipping ────────")
        return {}

    if verbose:
        print("\n── Device transfer latency ──────────────────────────────────────")

    iters_map = {"1 KB": 200, "64 KB": 200, "1 MB": 100, "64 MB": 20}
    rows_cpu_gpu: list[tuple[str, ...]] = [
        ("size", "direct µs", "shared µs", "speedup", "bandwidth (GB/s)")
    ]
    rows_gpu_cpu: list[tuple[str, ...]] = [("size", "µs", "GB/s")]

    out: dict[str, object] = {}
    for label, n in _SIZES:
        r = _bench_size(label, n, iters_map.get(label, 50))

        key = f"transfer/{label.replace(' ', '_')}"
        if "cpu_to_gpu_mean_us" in r:
            out[f"{key}/cpu_to_gpu_mean_us"] = r["cpu_to_gpu_mean_us"]
            out[f"{key}/cpu_to_gpu_gbps"] = r["cpu_to_gpu_gbps"]
            out[f"{key}/shared_to_gpu_mean_us"] = r["shared_to_gpu_mean_us"]
            out[f"{key}/shared_to_gpu_gbps"] = r["shared_to_gpu_gbps"]
            out[f"{key}/gpu_to_cpu_mean_us"] = r["gpu_to_cpu_mean_us"]
            out[f"{key}/gpu_to_cpu_gbps"] = r["gpu_to_cpu_gbps"]

            rows_cpu_gpu.append(
                (
                    label,
                    f"{r['cpu_to_gpu_mean_us']:.1f}",
                    f"{r['shared_to_gpu_mean_us']:.1f}",
                    f"{r['shared_speedup_x']:.2f}×",
                    f"{r['cpu_to_gpu_gbps']:.1f}",
                )
            )
            rows_gpu_cpu.append(
                (label, f"{r['gpu_to_cpu_mean_us']:.1f}", f"{r['gpu_to_cpu_gbps']:.1f}")
            )

    if verbose:
        from lucid.benchmarks._core import fmt_table

        print("\nCPU → GPU:")
        print(fmt_table(rows_cpu_gpu))
        print("\nGPU → CPU:")
        print(fmt_table(rows_gpu_cpu))

    return out


if __name__ == "__main__":
    run()
