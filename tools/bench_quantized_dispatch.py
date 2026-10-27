"""Measure Metal packed GEMM against cached and per-call dequantization.

No dispatch threshold is inferred from one machine. Each row first checks the
same quantized weights numerically. Timings include materialization and sync.
"""

import argparse
import json
from pathlib import Path
import platform
import subprocess

import lucid
from lucid.quantization import _qgemm
from tools._bench_timing import cold_ms, warm_ms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iter", type=int, default=20)
    parser.add_argument("--width", type=int, default=1024)
    args = parser.parse_args()
    if args.iter < 1 or args.width < 64 or args.width % 64:
        parser.error("iterations must be positive and width a positive multiple of 64")
    lucid.manual_seed(20260920)
    rows = []
    with lucid.no_grad():
        for bits in (4, 8):
            weight = lucid.randn(args.width, args.width, device="metal")
            packed, scales, biases = _qgemm.quantize(weight, group_size=64, bits=bits)
            dense = _qgemm.dequantize(packed, scales, biases, group_size=64, bits=bits)
            for batch in (1, 2, 8, 32):
                x = lucid.randn(batch, args.width, device="metal")

                def packed_call() -> lucid.Tensor:
                    return _qgemm.quantized_matmul(
                        x, packed, scales, biases, transpose=True, group_size=64, bits=bits
                    )

                def cached_call() -> lucid.Tensor:
                    return lucid.matmul(x, dense.mT)

                def unpack_call() -> lucid.Tensor:
                    w = _qgemm.dequantize(packed, scales, biases, group_size=64, bits=bits)
                    return lucid.matmul(x, w.mT)

                expected, actual = cached_call(), packed_call()
                error = float((expected - actual).abs().max().item())
                scale = max(float(expected.abs().max().item()), 1e-12)
                if not error / scale <= 1e-4:
                    raise RuntimeError(
                        f"parity failure for bits={bits}, M={batch}: {error / scale}"
                    )
                for name, call in (
                    ("packed", packed_call),
                    ("cached_dense", cached_call),
                    ("dequant_each_call", unpack_call),
                ):
                    first = cold_ms(call)
                    median, p5, p95 = warm_ms(call, 5, args.iter)
                    row = {
                        "bits": bits,
                        "m": batch,
                        "n": args.width,
                        "k": args.width,
                        "path": name,
                        "first_ms": first,
                        "median_ms": median,
                        "p5_ms": p5,
                        "p95_ms": p95,
                        "parity_relative_error": error / scale,
                    }
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    report = {
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "chip": subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip(),
            "version": lucid.__version__,
        },
        "contract": "end-to-end per-call materialization and sync; 5 warmups; first call is not cold compilation",
        "iterations": args.iter,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
