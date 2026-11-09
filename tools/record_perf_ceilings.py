#!/usr/bin/env python3
"""tools/record_perf_ceilings.py — record timing ceilings from N hosted runs, never one.

A ceiling in ``_thresholds.json`` is read by ``assert_no_regression`` with a tolerance band on
top.  Recorded from a single CI sample it fails the moment the runner pool hands out a slower
machine (LCD-867: six FFT and signal ceilings taken from one fast sample failed in 31 of 55
later runs, while the engine had not changed).  This tool takes the ``pytest_benchmark.json``
of several runs (the nightly uploads one per run as ``perf-history-<n>``), reduces each case to
one value per run (the plugin's median), and sets the ceiling so the checked limit
(``ceiling * (1 + tolerance)``) sits at a chosen quantile of those values.

    gh run download <run-id> -n perf-history-<n> -D runs/<n>      # once per run
    python3 -m tools.record_perf_ceilings lucid/test/perf/microbench/_thresholds.json \\
        runs/*/pytest_benchmark.json --cases '^(fft|ifft)_' --issue LCD-867 --write

Without ``--write`` it prints the table and changes nothing.  Fewer than ``--min-runs`` files
is refused: a quantile of three samples is the single-sample defect again.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

DEFAULT_TOLERANCE = 0.25  # assert_no_regression's default band
DEFAULT_QUANTILE = 0.95
DEFAULT_MIN_RUNS = 10
_NAME = re.compile(r"^test_(?P<case>\w+?)\[(?P<device>\w+)\]$")


def case_key(test_name: str) -> str | None:
    """``test_fft_64[cpu]`` -> ``fft_64_cpu``, the key ``assert_no_regression`` is called with."""
    m = _NAME.match(test_name)
    return f"{m['case']}_{m['device']}" if m else None


def run_medians(path: Path) -> dict[str, float]:
    """One pytest-benchmark JSON -> {case key: median seconds}."""
    out: dict[str, float] = {}
    for bench in json.loads(path.read_text())["benchmarks"]:
        key = case_key(bench["name"])
        if key is not None:
            out[key] = float(bench["stats"]["median"])
    return out


def quantile(values: list[float], q: float) -> float:
    """Linear-interpolated quantile; ``values`` must be non-empty."""
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def sig3(x: float) -> float:
    """Three significant digits, so the committed ceiling is readable."""
    return float(f"{x:.3g}")


def record(
    runs: list[dict[str, float]],
    old: dict[str, float],
    cases: re.Pattern[str],
    *,
    q: float,
    tolerance: float,
) -> dict[str, tuple[int, float, float]]:
    """{case: (runs seen, quantile value, new ceiling)} for every selected case in ``old``."""
    chosen: dict[str, tuple[int, float, float]] = {}
    for key in old:
        if not cases.search(key):
            continue
        values = [r[key] for r in runs if key in r]
        if not values:
            continue
        limit = quantile(values, q)
        chosen[key] = (len(values), limit, sig3(limit / (1.0 + tolerance)))
    return chosen


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("thresholds", type=Path)
    ap.add_argument("runs", type=Path, nargs="+", help="pytest-benchmark JSON of one run each")
    ap.add_argument("--cases", required=True, help="regex over the case keys to re-record")
    ap.add_argument("--issue", required=True, help="tracking issue, written into the provenance")
    ap.add_argument("--quantile", type=float, default=DEFAULT_QUANTILE)
    ap.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    ap.add_argument("--min-runs", type=int, default=DEFAULT_MIN_RUNS)
    ap.add_argument("--write", action="store_true", help="update the thresholds file")
    args = ap.parse_args(argv)

    if len(args.runs) < args.min_runs:
        print(
            f"refusing: {len(args.runs)} run file(s) < --min-runs {args.min_runs}; a ceiling "
            "from so few samples is the single-sample defect",
            file=sys.stderr,
        )
        return 2
    raw = json.loads(args.thresholds.read_text())
    old = {k: float(v) for k, v in raw.items() if not k.startswith("_")}
    chosen = record(
        [run_medians(p) for p in args.runs],
        old,
        re.compile(args.cases),
        q=args.quantile,
        tolerance=args.tolerance,
    )
    if not chosen:
        print(f"no case in {args.thresholds} matches {args.cases!r} in these runs", file=sys.stderr)
        return 2

    pct = int(args.quantile * 100)
    print(f"{'case':<28}{'runs':>5}{'p' + str(pct) + ' ms':>11}{'old ms':>9}{'new ms':>9}")
    for key, (n, limit, ceiling) in chosen.items():
        print(f"{key:<28}{n:>5}{limit * 1e3:>11.3f}{old[key] * 1e3:>9.3f}{ceiling * 1e3:>9.3f}")
    if not args.write:
        return 0

    provenance = raw.setdefault("_provenance", {})
    for key, (n, limit, ceiling) in chosen.items():
        raw[key] = ceiling
        provenance[key] = (
            f"{args.issue}: p{pct} of {n} hosted runs = {limit * 1e3:.3f} ms; "
            f"ceiling {ceiling * 1e3:.3f} ms * (1 + {args.tolerance}) puts the limit there"
        )
    args.thresholds.write_text(json.dumps(raw, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
