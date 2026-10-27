"""
benchmarks/bench_ops.py — element-wise, reduction, and matmul throughput.

For each op, measures:
  A) Lucid CPU  (Apple Accelerate backend)
  B) Lucid GPU  (MLX backend, forced eval)
  B) Raw MLX    (lower-bound reference — Lucid layer overhead = B - MLX)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lucid
from lucid.benchmarks._core import (
    BenchResult,
    bench_cpu,
    bench_gpu_lucid,
    bench_gpu_mlx,
    metal_available,
)

# ── benchmark definitions ─────────────────────────────────────────────────────


def _run_op_suite(n_elements: int = 10_000_000) -> list[dict[str, object]]:
    """
    Run all op benchmarks and return a list of result dicts.
    Each dict has keys: name, cpu_us, gpu_us, mlx_us, overhead_pct.
    """
    has_gpu = metal_available()
    results: list[dict[str, object]] = []

    # ── inputs ────────────────────────────────────────────────────────────────
    cpu_a = lucid.randn(n_elements)
    cpu_b = lucid.randn(n_elements)
    mat_n = 1024
    cpu_mat_a = lucid.randn(mat_n, mat_n)
    cpu_mat_b = lucid.randn(mat_n, mat_n)

    if has_gpu:
        import mlx.core as mx

        gpu_a = cpu_a.to("metal")
        gpu_b = cpu_b.to("metal")
        gpu_mat_a = cpu_mat_a.to("metal")
        gpu_mat_b = cpu_mat_b.to("metal")
        mlx_a = mx.array(cpu_a.numpy())
        mlx_b = mx.array(cpu_b.numpy())
        mlx_mat_a = mx.array(cpu_mat_a.numpy())
        mlx_mat_b = mx.array(cpu_mat_b.numpy())
        mx.eval(mlx_a, mlx_b, mlx_mat_a, mlx_mat_b)

    def _make_entry(
        name: str,
        cpu_ns: list[int],
        gpu_ns: list[int] | None,
        mlx_ns: list[int] | None,
    ) -> dict[str, object]:
        r: dict[str, object] = {"name": name}
        cpu_r = BenchResult(f"{name}_cpu", cpu_ns)
        cpu_r = BenchResult(f"{name}_cpu", cpu_ns)
        r["cpu_mean_us"] = cpu_r.mean_us
        r["cpu_median_us"] = cpu_r.median_us
        r["cpu_p95_us"] = cpu_r.p95_us
        if gpu_ns is not None:
            gpu_r = BenchResult(f"{name}_gpu", gpu_ns)
            r["gpu_mean_us"] = gpu_r.mean_us
            r["gpu_median_us"] = gpu_r.median_us
            r["gpu_p95_us"] = gpu_r.p95_us
        if mlx_ns is not None and gpu_ns is not None:
            mlx_r = BenchResult(f"{name}_mlx", mlx_ns)
            r["mlx_mean_us"] = mlx_r.mean_us
            # Overhead (%) — (lucid_gpu - mlx) / mlx * 100
            overhead = (gpu_r.mean_us - mlx_r.mean_us) / max(mlx_r.mean_us, 0.001) * 100.0
            r["layer_overhead_pct"] = round(overhead, 1)
        return r

    # ── relu ──────────────────────────────────────────────────────────────────
    cpu_ns = bench_cpu(lambda: lucid.relu(cpu_a))
    gpu_ns = bench_gpu_lucid(lambda: lucid.relu(gpu_a)) if has_gpu else None
    mlx_ns = bench_gpu_mlx(lambda: mx.maximum(mlx_a, 0.0)) if has_gpu else None
    results.append(_make_entry(f"relu_{n_elements//1_000_000}M", cpu_ns, gpu_ns, mlx_ns))

    # ── add ───────────────────────────────────────────────────────────────────
    cpu_ns = bench_cpu(lambda: lucid.add(cpu_a, cpu_b))
    gpu_ns = bench_gpu_lucid(lambda: lucid.add(gpu_a, gpu_b)) if has_gpu else None
    mlx_ns = bench_gpu_mlx(lambda: mlx_a + mlx_b) if has_gpu else None
    results.append(_make_entry(f"add_{n_elements//1_000_000}M", cpu_ns, gpu_ns, mlx_ns))

    # ── mul ───────────────────────────────────────────────────────────────────
    cpu_ns = bench_cpu(lambda: lucid.mul(cpu_a, cpu_b))
    gpu_ns = bench_gpu_lucid(lambda: lucid.mul(gpu_a, gpu_b)) if has_gpu else None
    mlx_ns = bench_gpu_mlx(lambda: mlx_a * mlx_b) if has_gpu else None
    results.append(_make_entry(f"mul_{n_elements//1_000_000}M", cpu_ns, gpu_ns, mlx_ns))

    # ── sum ───────────────────────────────────────────────────────────────────
    cpu_ns = bench_cpu(lambda: lucid.sum(cpu_a))
    gpu_ns = bench_gpu_lucid(lambda: lucid.sum(gpu_a)) if has_gpu else None
    mlx_ns = bench_gpu_mlx(lambda: mx.sum(mlx_a)) if has_gpu else None
    results.append(_make_entry(f"sum_{n_elements//1_000_000}M", cpu_ns, gpu_ns, mlx_ns))

    # ── exp ───────────────────────────────────────────────────────────────────
    cpu_ns = bench_cpu(lambda: lucid.exp(cpu_a))
    gpu_ns = bench_gpu_lucid(lambda: lucid.exp(gpu_a)) if has_gpu else None
    mlx_ns = bench_gpu_mlx(lambda: mx.exp(mlx_a)) if has_gpu else None
    results.append(_make_entry(f"exp_{n_elements//1_000_000}M", cpu_ns, gpu_ns, mlx_ns))

    # ── matmul 1024×1024 ──────────────────────────────────────────────────────
    cpu_ns = bench_cpu(lambda: lucid.matmul(cpu_mat_a, cpu_mat_b), iters=20)
    gpu_ns = (
        bench_gpu_lucid(lambda: lucid.matmul(gpu_mat_a, gpu_mat_b), iters=20) if has_gpu else None
    )
    mlx_ns = bench_gpu_mlx(lambda: mx.matmul(mlx_mat_a, mlx_mat_b), iters=20) if has_gpu else None
    results.append(_make_entry(f"matmul_{mat_n}x{mat_n}", cpu_ns, gpu_ns, mlx_ns))

    return results


def run(verbose: bool = True) -> dict[str, object]:
    if verbose:
        print("\n── Op throughput ────────────────────────────────────────────────")
    raw = _run_op_suite()
    out: dict[str, object] = {}
    for r in raw:
        name = str(r["name"])
        out[f"ops/{name}/cpu_mean_us"] = r["cpu_mean_us"]
        out[f"ops/{name}/cpu_median_us"] = r["cpu_median_us"]
        out[f"ops/{name}/cpu_p95_us"] = r["cpu_p95_us"]
        if "gpu_mean_us" in r:
            out[f"ops/{name}/gpu_mean_us"] = r["gpu_mean_us"]
            out[f"ops/{name}/gpu_median_us"] = r["gpu_median_us"]
            out[f"ops/{name}/gpu_p95_us"] = r["gpu_p95_us"]
        if "mlx_mean_us" in r:
            out[f"ops/{name}/mlx_mean_us"] = r["mlx_mean_us"]
            out[f"ops/{name}/layer_overhead_pct"] = r["layer_overhead_pct"]
    if verbose:
        _print_table(raw)
    return out


def _print_table(results: list[dict[str, object]]) -> None:
    has_gpu = any("gpu_mean_us" in r for r in results)
    has_mlx = any("mlx_mean_us" in r for r in results)

    header: list[str] = ["op", "cpu µs"]
    if has_gpu:
        header += ["gpu µs", "gpu p95"]
    if has_mlx:
        header += ["mlx µs", "overhead%"]

    rows: list[tuple[str, ...]] = [tuple(header)]
    for r in results:
        row: list[str] = [
            str(r["name"]),
            f"{r['cpu_mean_us']:.1f}",
        ]
        if has_gpu:
            row += [
                f"{r.get('gpu_mean_us', '-'):.1f}" if "gpu_mean_us" in r else "-",
                f"{r.get('gpu_p95_us', '-'):.1f}" if "gpu_p95_us" in r else "-",
            ]
        if has_mlx:
            row += [
                f"{r.get('mlx_mean_us', '-'):.1f}" if "mlx_mean_us" in r else "-",
                (f"{r.get('layer_overhead_pct', '-'):+.1f}%" if "layer_overhead_pct" in r else "-"),
            ]
        rows.append(tuple(row))

    from lucid.benchmarks._core import fmt_table

    print(fmt_table(rows))


if __name__ == "__main__":
    run()
