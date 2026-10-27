"""
lucid/benchmarks/run_all.py — CLI orchestrator for the Lucid benchmark suite.

Usage:
  python lucid/benchmarks/run_all.py                  # run + print table
  python lucid/benchmarks/run_all.py --save           # run + save baseline.json
  python lucid/benchmarks/run_all.py --check          # run + compare vs baseline
  python lucid/benchmarks/run_all.py --check --threshold 5  # 5% regression limit

Exit codes:
  0  — all clear (or --save)
  1  — regression(s) detected (only with --check)
  2  — baseline.json not found (only with --check)
"""

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from lucid.benchmarks import bench_ops, bench_transfer, bench_train

_BASELINE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "baseline", "baseline.json"
)

_DEFAULT_THRESHOLD = 15.0  # percent regression limit


# ── helpers ───────────────────────────────────────────────────────────────────


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, OSError):
        # Not a repository, or no VCS binary on PATH.
        return "unknown"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# ── run ───────────────────────────────────────────────────────────────────────


def run_all(verbose: bool = True) -> dict[str, object]:
    """Run all benchmark suites and return a flat results dict."""
    results: dict[str, object] = {}
    results.update(bench_ops.run(verbose=verbose))
    results.update(bench_transfer.run(verbose=verbose))
    results.update(bench_train.run(verbose=verbose))
    return results


# ── save ──────────────────────────────────────────────────────────────────────


def _payload(results: dict[str, object]) -> dict[str, object]:
    import platform

    return {
        "commit": _git_sha(),
        "date": _now_iso(),
        # A timing is only comparable with one from the same machine, so it
        # travels with what produced it.
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


# ── check ─────────────────────────────────────────────────────────────────────


def _is_latency_key(key: str) -> bool:
    """Return True for keys where a higher value = regression (latency/time).

    Prefer *_median_us keys for comparison — median is more robust to
    measurement noise than mean or p95.  Fall back to mean if median
    is not present.
    """
    return any(
        key.endswith(suf)
        for suf in ("_median_us", "_mean_us", "_median_ms", "_mean_ms")
    )


def _is_throughput_key(key: str) -> bool:
    """Return True for keys where a lower value = regression (throughput/rate)."""
    return any(key.endswith(suf) for suf in ("_gbps", "_samples_sec"))


def check_regressions(
    current: dict[str, object],
    baseline: dict[str, object],
    threshold_pct: float,
) -> list[str]:
    """
    Return a list of human-readable regression strings.
    Empty list = no regressions.
    """
    regressions: list[str] = []
    for key, cur_val in current.items():
        if key not in baseline:
            continue
        base_val = baseline[key]
        if not isinstance(cur_val, (int, float)) or not isinstance(
            base_val, (int, float)
        ):
            continue
        if base_val == 0:
            continue

        if _is_latency_key(key):
            # Higher is worse
            pct_change = (float(cur_val) - float(base_val)) / float(base_val) * 100.0
            if pct_change > threshold_pct:
                regressions.append(
                    f"  ❌  {key}: {base_val:.2f} → {cur_val:.2f} "
                    f"(+{pct_change:.1f}%  >  {threshold_pct:.0f}% limit)"
                )
        elif _is_throughput_key(key):
            # Lower is worse
            pct_change = (float(base_val) - float(cur_val)) / float(base_val) * 100.0
            if pct_change > threshold_pct:
                regressions.append(
                    f"  ❌  {key}: {base_val:.2f} → {cur_val:.2f} "
                    f"(-{pct_change:.1f}%  >  {threshold_pct:.0f}% limit)"
                )
    return regressions


def print_comparison(
    current: dict[str, object],
    baseline: dict[str, object],
    threshold_pct: float,
) -> list[str]:
    regressions = check_regressions(current, baseline, threshold_pct)
    improvements: list[str] = []

    for key, cur_val in current.items():
        if key not in baseline:
            continue
        base_val = baseline[key]
        if not isinstance(cur_val, (int, float)) or not isinstance(
            base_val, (int, float)
        ):
            continue
        if base_val == 0:
            continue

        if _is_latency_key(key):
            pct = (float(cur_val) - float(base_val)) / float(base_val) * 100.0
            if pct < -threshold_pct:
                improvements.append(
                    f"  ✅  {key}: {base_val:.2f} → {cur_val:.2f} ({pct:+.1f}%)"
                )
        elif _is_throughput_key(key):
            pct = (float(cur_val) - float(base_val)) / float(base_val) * 100.0
            if pct > threshold_pct:
                improvements.append(
                    f"  ✅  {key}: {base_val:.2f} → {cur_val:.2f} (+{pct:.1f}%)"
                )

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


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lucid performance benchmark suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
    args = parser.parse_args()

    verbose = not args.quiet

    print(f"\n{'='*62}")
    print(f"  Lucid benchmark suite  |  commit {_git_sha()}  |  {_now_iso()}")
    print(f"{'='*62}")

    results = run_all(verbose=verbose)
    if args.json:
        save_json(results, args.json)

    if args.save:
        save_baseline(results)
        return

    if args.check:
        if not os.path.exists(_BASELINE_PATH):
            print(f"\n⚠️  Baseline not found: {_BASELINE_PATH}")
            print("   Run with --save first to establish a baseline.")
            sys.exit(2)
        with open(_BASELINE_PATH, encoding="utf-8") as f:
            payload = json.load(f)
        baseline_results = payload.get("results", {})
        baseline_commit = payload.get("commit", "unknown")
        print(f"\n   Baseline commit: {baseline_commit}")
        regressions = print_comparison(results, baseline_results, args.threshold)
        if regressions:
            print(f"\n💥 {len(regressions)} regression(s) detected.")
            sys.exit(1)
        else:
            print("\n✅ No regressions detected.")
        return

    # Default: just print results summary
    print(f"\n── Summary ({len(results)} metrics) ──────────────────────────────────")
    for key in sorted(results):
        val = results[key]
        if isinstance(val, float):
            print(f"  {key}: {val:.3f}")
        else:
            print(f"  {key}: {val}")
    print("\nRun with --save to store as baseline, --check to compare.\n")


if __name__ == "__main__":
    main()
