"""
benchmarks/bench_ops.py — element-wise, reduction, matmul, conv2d, SDPA, and dispatch throughput.

For each op, measures:
  A) Lucid CPU  (Apple Accelerate backend)
  B) Lucid GPU  (Metal backend, forced eval)
  C) Raw MLX    (lower-bound reference — Lucid layer overhead = B - MLX)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lucid
import lucid.nn.functional as F
from lucid.benchmarks._core import (
    BenchResult,
    bench_cpu,
    bench_gpu_lucid,
    bench_gpu_mlx,
    fmt_table,
    metal_available,
)


def _make_entry(
    name: str, cpu_ns: list[int], gpu_ns: list[int] | None, mlx_ns: list[int] | None
) -> dict[str, object]:
    r: dict[str, object] = {"name": name}
    cpu_r = BenchResult(f"{name}_cpu", cpu_ns)
    cpu_dict = cpu_r.to_dict()
    r["cpu_mean_us"] = cpu_r.mean_us
    r["cpu_median_us"] = cpu_r.median_us
    r["cpu_p95_us"] = cpu_r.p95_us
    r["cpu_p99_us"] = cpu_dict["p99_us"]
    if gpu_ns is not None:
        gpu_r = BenchResult(f"{name}_gpu", gpu_ns)
        r["gpu_mean_us"] = gpu_r.mean_us
        r["gpu_median_us"] = gpu_r.median_us
        r["gpu_p95_us"] = gpu_r.p95_us
        if mlx_ns is not None:
            mlx_r = BenchResult(f"{name}_mlx", mlx_ns)
            r["mlx_mean_us"] = mlx_r.mean_us
            overhead = (gpu_r.mean_us - mlx_r.mean_us) / max(mlx_r.mean_us, 0.001) * 100.0
            r["layer_overhead_pct"] = round(overhead, 1)
    return r


def _run_kernel_benches(has_gpu: bool) -> list[dict[str, object]]:
    """Measure Conv2d, SDPA, and small-tensor Python dispatch overhead."""
    out: list[dict[str, object]] = []

    # 1. Small-tensor dispatch (64-element add, CPU eager hot path)
    small_a = lucid.randn(64)
    small_b = lucid.randn(64)
    cpu_ns = bench_cpu(lambda: small_a + small_b, warmup=20, iters=500)
    gpu_s_a = small_a.to("metal") if has_gpu else small_a
    gpu_s_b = small_b.to("metal") if has_gpu else small_b
    gpu_ns = bench_gpu_lucid(lambda: gpu_s_a + gpu_s_b, iters=50) if has_gpu else None
    out.append(_make_entry("dispatch_add_64", cpu_ns, gpu_ns, None))

    # 2. Conv2d 3x3 (B=8, C=32, H=32, W=32)
    conv_x = lucid.randn(8, 32, 32, 32)
    conv_w = lucid.randn(32, 32, 3, 3)
    cpu_ns = bench_cpu(lambda: F.conv2d(conv_x, conv_w, padding=1), warmup=3, iters=15)
    gpu_cx = conv_x.to("metal") if has_gpu else conv_x
    gpu_cw = conv_w.to("metal") if has_gpu else conv_w
    gpu_ns = (
        bench_gpu_lucid(lambda: F.conv2d(gpu_cx, gpu_cw, padding=1), warmup=3, iters=15)
        if has_gpu
        else None
    )
    out.append(_make_entry("conv2d_3x3_b8_c32", cpu_ns, gpu_ns, None))

    # 3. Scaled Dot-Product Attention (B=4, H=8, T=64, D=64)
    q = lucid.randn(4, 8, 64, 64)
    k = lucid.randn(4, 8, 64, 64)
    v = lucid.randn(4, 8, 64, 64)
    cpu_ns = bench_cpu(
        lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True), warmup=5, iters=30
    )
    gpu_q = q.to("metal") if has_gpu else q
    gpu_k = k.to("metal") if has_gpu else k
    gpu_v = v.to("metal") if has_gpu else v
    gpu_ns = (
        bench_gpu_lucid(
            lambda: F.scaled_dot_product_attention(gpu_q, gpu_k, gpu_v, is_causal=True),
            warmup=5,
            iters=30,
        )
        if has_gpu
        else None
    )
    out.append(_make_entry("sdpa_b4_h8_t64_d64", cpu_ns, gpu_ns, None))
    return out


def _run_op_suite(n_elements: int = 10_000_000) -> list[dict[str, object]]:
    """Run all op benchmarks and return a list of result dicts."""
    has_gpu = metal_available()
    results: list[dict[str, object]] = []

    cpu_a = lucid.randn(n_elements)
    cpu_b = lucid.randn(n_elements)
    mat_n = 1024
    cpu_mat_a = lucid.randn(mat_n, mat_n)
    cpu_mat_b = lucid.randn(mat_n, mat_n)

    gpu_a = cpu_a.to("metal") if has_gpu else cpu_a
    gpu_b = cpu_b.to("metal") if has_gpu else cpu_b
    gpu_mat_a = cpu_mat_a.to("metal") if has_gpu else cpu_mat_a
    gpu_mat_b = cpu_mat_b.to("metal") if has_gpu else cpu_mat_b

    if has_gpu:
        import mlx.core as mx

        mlx_a = mx.random.normal((n_elements,))
        mlx_b = mx.random.normal((n_elements,))
        mlx_mat_a = mx.random.normal((mat_n, mat_n))
        mlx_mat_b = mx.random.normal((mat_n, mat_n))
        mx.eval(mlx_a, mlx_b, mlx_mat_a, mlx_mat_b)

        tag = f"{n_elements // 1_000_000}M"
        results.append(
            _make_entry(
                f"relu_{tag}",
                bench_cpu(lambda: lucid.relu(cpu_a)),
                bench_gpu_lucid(lambda: lucid.relu(gpu_a)),
                bench_gpu_mlx(lambda: mx.maximum(mlx_a, 0.0)),
            )
        )
        results.append(
            _make_entry(
                f"add_{tag}",
                bench_cpu(lambda: lucid.add(cpu_a, cpu_b)),
                bench_gpu_lucid(lambda: lucid.add(gpu_a, gpu_b)),
                bench_gpu_mlx(lambda: mlx_a + mlx_b),
            )
        )
        results.append(
            _make_entry(
                f"mul_{tag}",
                bench_cpu(lambda: lucid.mul(cpu_a, cpu_b)),
                bench_gpu_lucid(lambda: lucid.mul(gpu_a, gpu_b)),
                bench_gpu_mlx(lambda: mlx_a * mlx_b),
            )
        )
        results.append(
            _make_entry(
                f"sum_{tag}",
                bench_cpu(lambda: lucid.sum(cpu_a)),
                bench_gpu_lucid(lambda: lucid.sum(gpu_a)),
                bench_gpu_mlx(lambda: mx.sum(mlx_a)),
            )
        )
        results.append(
            _make_entry(
                f"exp_{tag}",
                bench_cpu(lambda: lucid.exp(cpu_a)),
                bench_gpu_lucid(lambda: lucid.exp(gpu_a)),
                bench_gpu_mlx(lambda: mx.exp(mlx_a)),
            )
        )
        results.append(
            _make_entry(
                f"matmul_{mat_n}x{mat_n}",
                bench_cpu(lambda: lucid.matmul(cpu_mat_a, cpu_mat_b), iters=20),
                bench_gpu_lucid(lambda: lucid.matmul(gpu_mat_a, gpu_mat_b), iters=20),
                bench_gpu_mlx(lambda: mx.matmul(mlx_mat_a, mlx_mat_b), iters=20),
            )
        )
    else:
        tag = f"{n_elements // 1_000_000}M"
        results.append(_make_entry(f"relu_{tag}", bench_cpu(lambda: lucid.relu(cpu_a)), None, None))
        results.append(
            _make_entry(f"add_{tag}", bench_cpu(lambda: lucid.add(cpu_a, cpu_b)), None, None)
        )
        results.append(
            _make_entry(f"mul_{tag}", bench_cpu(lambda: lucid.mul(cpu_a, cpu_b)), None, None)
        )
        results.append(_make_entry(f"sum_{tag}", bench_cpu(lambda: lucid.sum(cpu_a)), None, None))
        results.append(_make_entry(f"exp_{tag}", bench_cpu(lambda: lucid.exp(cpu_a)), None, None))
        results.append(
            _make_entry(
                f"matmul_{mat_n}x{mat_n}",
                bench_cpu(lambda: lucid.matmul(cpu_mat_a, cpu_mat_b), iters=20),
                None,
                None,
            )
        )

    results.extend(_run_kernel_benches(has_gpu))
    return results


def run(verbose: bool = True) -> dict[str, object]:
    if verbose:
        print("\n── Op & kernel throughput ───────────────────────────────────────")
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
        row: list[str] = [str(r["name"]), f"{r['cpu_mean_us']:.1f}"]
        if has_gpu:
            row += [
                f"{r['gpu_mean_us']:.1f}" if "gpu_mean_us" in r else "-",
                f"{r['gpu_p95_us']:.1f}" if "gpu_p95_us" in r else "-",
            ]
        if has_mlx:
            row += [
                f"{r['mlx_mean_us']:.1f}" if "mlx_mean_us" in r else "-",
                f"{r['layer_overhead_pct']:+.1f}%" if "layer_overhead_pct" in r else "-",
            ]
        rows.append(tuple(row))

    print(fmt_table(rows))


if __name__ == "__main__":
    run()
