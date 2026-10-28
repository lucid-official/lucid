"""
benchmarks/run_all.py — benchmark suite runner and baseline regression checker.

Usage:
    # Run all benchmarks and print summary tables:
    python -m lucid.benchmarks.run_all

    # Run a single benchmark suite (ops, transfer, train, models):
    python -m lucid.benchmarks.run_all --suite models

    # Save current results as the new baseline:
    python -m lucid.benchmarks.run_all --save

    # Compare current results against saved baseline (fail if > threshold%):
    python -m lucid.benchmarks.run_all --check [--threshold 15]
"""

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lucid.benchmarks import bench_models, bench_ops, bench_train, bench_transfer

_BASELINE_DIR = os.path.join(os.path.dirname(__file__), "baseline")
_BASELINE_PATH = os.path.join(_BASELINE_DIR, "baseline.json")
_DEFAULT_THRESHOLD = 15.0  # percent


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def run_all(verbose: bool = True, suite: str = "all") -> dict[str, object]:
    """Run selected benchmark suites and return a flat results dict."""
    results: dict[str, object] = {}
    if suite in ("all", "ops"):
        results.update(bench_ops.run(verbose=verbose))
    if suite in ("all", "transfer"):
        results.update(bench_transfer.run(verbose=verbose))
    if suite in ("all", "train"):
        results.update(bench_train.run(verbose=verbose))
    if suite in ("all", "models"):
        results.update(bench_models.run(verbose=verbose))
    return results


def _payload(results: dict[str, object]) -> dict[str, object]:
    import platform

    return {
        "commit": _git_sha(),
        "date": _now_iso(),
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "results": results,
    }


def save_json(results: dict[str, object], path: str) -> None:
    """Write this run on its own — the nightly job keeps one per night."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_payload(results), f, indent=2)
    print(f"\n   Results written → {path}")


def save_baseline(results: dict[str, object]) -> None:
    payload = _payload(results)
    os.makedirs(os.path.dirname(_BASELINE_PATH), exist_ok=True)
    with open(_BASELINE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n✅ Baseline saved → {_BASELINE_PATH}  (commit {payload['commit']})")


def _is_latency_key(key: str) -> bool:
    """Return True for keys where a higher value = regression (latency/time)."""
    return any(key.endswith(suf) for suf in ("_median_us", "_mean_us", "_median_ms", "_mean_ms"))


def _is_throughput_key(key: str) -> bool:
    """Return True for keys where a lower value = regression (throughput/rate)."""
    return any(key.endswith(suf) for suf in ("_gbps", "_samples_sec", "_tok_s"))


def _regression_pct(
    key: str, cur_val: object, base_val: object
) -> tuple[float, float, float, str] | None:
    """Return ``(base_f, cur_f, regression_pct, raw_pct_str)`` for comparable keys.

    ``regression_pct > 0`` means slower/worse; ``regression_pct < 0`` means faster/better.
    """
    if not isinstance(cur_val, (int, float)) or not isinstance(base_val, (int, float)):
        return None
    base_f, cur_f = float(base_val), float(cur_val)
    if base_f == 0.0:
        return None
    raw_pct = (cur_f - base_f) / base_f * 100.0
    if _is_latency_key(key):
        return base_f, cur_f, raw_pct, f"{raw_pct:+.1f}%"
    if _is_throughput_key(key):
        return base_f, cur_f, -raw_pct, f"{raw_pct:+.1f}%"
    return None


def check_regressions(
    current: dict[str, object], baseline: dict[str, object], threshold_pct: float
) -> list[str]:
    """Return a list of human-readable regression strings (empty = no regressions)."""
    regressions: list[str] = []
    for key, cur_val in current.items():
        if key not in baseline:
            continue
        stat = _regression_pct(key, cur_val, baseline[key])
        if stat is not None and stat[2] > threshold_pct:
            base_f, cur_f, _, pct_str = stat
            regressions.append(
                f"  ❌  {key}: {base_f:.2f} → {cur_f:.2f} ({pct_str}  >  {threshold_pct:.0f}% limit)"
            )
    return regressions


def print_comparison(
    current: dict[str, object], baseline: dict[str, object], threshold_pct: float
) -> list[str]:
    regressions = check_regressions(current, baseline, threshold_pct)
    improvements: list[str] = []
    for key, cur_val in current.items():
        if key not in baseline:
            continue
        stat = _regression_pct(key, cur_val, baseline[key])
        if stat is not None and stat[2] < -threshold_pct:
            base_f, cur_f, _, pct_str = stat
            improvements.append(f"  ✅  {key}: {base_f:.2f} → {cur_f:.2f} ({pct_str})")

    print(f"\n── Comparison vs baseline (threshold={threshold_pct:.0f}%) ────────")
    if improvements:
        print("Improvements:")
        for s in improvements:
            print(s)
    if regressions:
        print("Regressions:")
        for s in regressions:
            print(s)
    if not improvements and not regressions:
        print("  No significant changes (all within threshold).")
    return regressions


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Lucid performance benchmark suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--suite",
        choices=("all", "ops", "transfer", "train", "models"),
        default="all",
        help="Benchmark suite to run (default: all)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results as the new baseline (baseline/baseline.json)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare results against the saved baseline and report regressions",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=_DEFAULT_THRESHOLD,
        metavar="PCT",
        help=f"Regression threshold in %% (default: {_DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        help="Also write this run's results (with commit and machine) to PATH",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-suite tables (only print comparison/summary)",
    )
    return parser


def _run_check(results: dict[str, object], threshold: float) -> None:
    if not os.path.exists(_BASELINE_PATH):
        print(f"\n⚠️  Baseline not found: {_BASELINE_PATH}")
        print("   Run with --save first to establish a baseline.")
        sys.exit(2)
    with open(_BASELINE_PATH, encoding="utf-8") as f:
        payload = json.load(f)
    baseline_results = payload.get("results", {})
    baseline_commit = payload.get("commit", "unknown")
    print(f"\n   Baseline commit: {baseline_commit}")
    regressions = print_comparison(results, baseline_results, threshold)
    if regressions:
        print(f"\n💥 {len(regressions)} regression(s) detected.")
        sys.exit(1)
    print("\n✅ No regressions detected.")


def main() -> None:
    args = _build_parser().parse_args()
    verbose = not args.quiet

    print(f"\n{'='*62}")
    print(f"  Lucid benchmark suite  |  commit {_git_sha()}  |  {_now_iso()}")
    print(f"{'='*62}")

    results = run_all(verbose=verbose, suite=args.suite)
    if args.json:
        save_json(results, args.json)
    if args.save:
        save_baseline(results)
        return
    if args.check:
        _run_check(results, args.threshold)
        return

    print(f"\n── Summary ({len(results)} metrics) ──────────────────────────────────")
    for key in sorted(results):
        val = results[key]
        print(f"  {key}: {val:.3f}" if isinstance(val, float) else f"  {key}: {val}")
    print("\nRun with --save to store as baseline, --check to compare.\n")


if __name__ == "__main__":
    main()
