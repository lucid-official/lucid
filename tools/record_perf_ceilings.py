#!/usr/bin/env python3
"""tools/record_perf_ceilings.py — record timing ceilings from N hosted runs, never one.

A ceiling in ``_thresholds.json`` is read by ``assert_no_regression`` with a tolerance band on
top.  Recorded from a single CI sample it fails the moment the runner pool hands out a slower
machine (LCD-867: six FFT and signal ceilings taken from one fast sample failed in 31 of 55
later runs, while the engine had not changed).  This tool takes the ``pytest_benchmark.json``
of several runs (the nightly uploads one per run as ``perf-history-<n>``, the phase gate's checks
part one per push as ``perf-core-<run id>-<attempt>``; ``tools/README.md``, "Perf ceiling
history", lists and downloads both), reduces each case to one value per run (the plugin's
median), and sets the ceiling so the checked limit (``ceiling * (1 + tolerance)``) sits
``headroom`` above a chosen quantile of those values.

The default is the slowest run seen plus 15%.  A limit *at* a quantile fails by construction:
at p95 each case is red in one run of twenty, and six such cases made the nightly red in one
run of four (the first nightly after LCD-867 failed at 3.939 ms against a 3.938 ms limit).

    gh run download <run-id> -n <artifact-name> -D runs/<artifact-name>   # once per run
    python3 -m tools.record_perf_ceilings lucid/test/perf/microbench/_thresholds.json \\
        runs/*/pytest_benchmark.json --cases '^(fft|ifft)_' --issue LCD-867 --write

Without ``--write`` it prints the table and changes nothing.  Fewer than ``--min-runs`` files
is refused: a quantile of three samples is the single-sample defect again.

A recording that would *lower* a ceiling is refused without ``--allow-lower``: the runs on hand
can be fewer than the ones the old ceiling came from (LCD-947: the ten nightly artifacts top out
at 13.0 ms for ``argsort_16384x64_int64`` while the hosted CI logs the ceiling was derived from
read 16.6 ms, so recording the artifacts alone would have re-opened the flake).

``--audit`` is the sweep over every ceiling in the file instead of a recording: for each one,
how often the checked limit failed in these runs, and how far above the slowest run it sits.

    python3 -m tools.record_perf_ceilings lucid/test/perf/microbench/_thresholds.json \\
        runs/*/pytest_benchmark.json --audit
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import NamedTuple

DEFAULT_TOLERANCE = 0.25  # assert_no_regression's default band
DEFAULT_QUANTILE = 1.0  # the slowest run seen
DEFAULT_HEADROOM = 1.15  # how far above it the checked limit sits
DEFAULT_MIN_RUNS = 10
DEFAULT_EXCEED_RATE = 0.02  # a ceiling whose limit fails in this share of runs is flaky
_FAILING_FLAGS = frozenset({"NO-HISTORY", "EXCEEDS", "TIGHT"})
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
    headroom: float = 1.0,
) -> dict[str, tuple[int, float, float]]:
    """{case: (runs seen, quantile value, new ceiling)} for every selected case in ``old``.

    The checked limit, ``ceiling * (1 + tolerance)``, is ``headroom`` times the quantile value.
    """
    chosen: dict[str, tuple[int, float, float]] = {}
    for key in old:
        if not cases.search(key):
            continue
        values = [r[key] for r in runs if key in r]
        if not values:
            continue
        limit = quantile(values, q)
        chosen[key] = (len(values), limit, sig3(limit * headroom / (1.0 + tolerance)))
    return chosen


class Audit(NamedTuple):
    """How one ceiling holds up against the runs: ``runs == 0`` means no run measured it."""

    runs: int
    p95: float
    slowest: float
    limit: float
    exceeded: int
    flags: tuple[str, ...]


def audit(
    runs: list[dict[str, float]],
    ceilings: dict[str, float],
    cases: re.Pattern[str],
    *,
    tolerance: float,
    headroom: float,
    exceed_rate: float,
    min_runs: int,
) -> dict[str, Audit]:
    """{case: Audit} for every selected ceiling, whether or not any run measured it.

    ``NO-HISTORY``: no run carries the case under the name the tests assert with, so
    ``record`` can never re-record it.  ``EXCEEDS``: the checked limit failed in at least
    ``exceed_rate`` of the runs.  ``TIGHT``: it never or rarely failed, but sits less than
    ``headroom`` above the slowest run -- the limit ``record`` would set.  ``FEW-RUNS`` is
    informational: fewer than ``min_runs`` runs measured it.
    """
    out: dict[str, Audit] = {}
    for key, ceiling in ceilings.items():
        if not cases.search(key):
            continue
        values = [r[key] for r in runs if key in r]
        if not values:
            out[key] = Audit(0, 0.0, 0.0, ceiling * (1.0 + tolerance), 0, ("NO-HISTORY",))
            continue
        limit = ceiling * (1.0 + tolerance)
        slowest = max(values)
        exceeded = sum(v > limit for v in values)
        flags: list[str] = []
        if exceeded / len(values) >= exceed_rate:
            flags.append("EXCEEDS")
        elif limit < headroom * slowest:
            flags.append("TIGHT")
        if len(values) < min_runs:
            flags.append("FEW-RUNS")
        out[key] = Audit(
            len(values), quantile(values, 0.95), slowest, limit, exceeded, tuple(flags)
        )
    return out


def _print_audit(
    rows: dict[str, Audit], ceilings: dict[str, float], provenance: dict[str, str]
) -> None:
    print(
        f"{'case':<28}{'runs':>5}{'p95 ms':>9}{'max ms':>9}{'ceil ms':>9}{'limit ms':>10}"
        f"{'over':>5}{'limit/max':>10}  {'source':<9}flags"
    )
    for key, row in rows.items():
        src = "recorded" if key in provenance else "legacy"
        ratio = f"{row.limit / row.slowest:>10.2f}" if row.runs else f"{'-':>10}"
        print(
            f"{key:<28}{row.runs:>5}{row.p95 * 1e3:>9.3f}{row.slowest * 1e3:>9.3f}"
            f"{ceilings[key] * 1e3:>9.3f}{row.limit * 1e3:>10.3f}{row.exceeded:>5}{ratio}"
            f"  {src:<9}{' '.join(row.flags)}"
        )


def _audit_main(args: argparse.Namespace, raw: dict[str, object]) -> int:
    ceilings = {k: float(v) for k, v in raw.items() if isinstance(v, int | float)}
    provenance = raw.get("_provenance")
    rows = audit(
        [run_medians(p) for p in args.runs],
        ceilings,
        re.compile(args.cases or ""),
        tolerance=args.tolerance,
        headroom=args.headroom,
        exceed_rate=args.exceed_rate,
        min_runs=args.min_runs,
    )
    _print_audit(rows, ceilings, provenance if isinstance(provenance, dict) else {})
    return 1 if any(_FAILING_FLAGS.intersection(row.flags) for row in rows.values()) else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("thresholds", type=Path)
    ap.add_argument("runs", type=Path, nargs="+", help="pytest-benchmark JSON of one run each")
    ap.add_argument("--cases", help="regex over the case keys to re-record (required to record)")
    ap.add_argument("--issue", help="tracking issue, written into the provenance (to record)")
    ap.add_argument("--quantile", type=float, default=DEFAULT_QUANTILE)
    ap.add_argument("--headroom", type=float, default=DEFAULT_HEADROOM)
    ap.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    ap.add_argument("--min-runs", type=int, default=DEFAULT_MIN_RUNS)
    ap.add_argument("--write", action="store_true", help="update the thresholds file")
    ap.add_argument("--allow-lower", action="store_true", help="let --write lower a ceiling")
    ap.add_argument("--audit", action="store_true", help="sweep every ceiling, record nothing")
    ap.add_argument("--exceed-rate", type=float, default=DEFAULT_EXCEED_RATE, help="for --audit")
    args = ap.parse_args(argv)

    raw = json.loads(args.thresholds.read_text())
    if args.audit:
        if args.write:
            ap.error("--audit records nothing; drop --write")
        return _audit_main(args, raw)
    if not (args.cases and args.issue):
        ap.error("--cases and --issue are required to record (or pass --audit)")
    if len(args.runs) < args.min_runs:
        print(
            f"refusing: {len(args.runs)} run file(s) < --min-runs {args.min_runs}; a ceiling "
            "from so few samples is the single-sample defect",
            file=sys.stderr,
        )
        return 2
    old = {k: float(v) for k, v in raw.items() if not k.startswith("_")}
    selected = re.compile(args.cases)
    chosen = record(
        [run_medians(p) for p in args.runs],
        old,
        selected,
        q=args.quantile,
        tolerance=args.tolerance,
        headroom=args.headroom,
    )
    if not chosen:
        print(f"no case in {args.thresholds} matches {args.cases!r} in these runs", file=sys.stderr)
        return 2
    unread = [k for k in old if selected.search(k) and k not in chosen]
    if unread:
        print(f"no run measured {', '.join(unread)}; left unchanged", file=sys.stderr)

    pct = int(args.quantile * 100)
    print(
        f"{'case':<28}{'runs':>5}{'p' + str(pct) + ' ms':>11}{'old ms':>9}{'new ms':>9}{'new/old':>9}"
    )
    for key, (n, limit, ceiling) in chosen.items():
        print(
            f"{key:<28}{n:>5}{limit * 1e3:>11.3f}{old[key] * 1e3:>9.3f}{ceiling * 1e3:>9.3f}"
            f"{ceiling / old[key]:>9.2f}"
        )
    if not args.write:
        return 0
    lowered = [k for k, (_, _, ceiling) in chosen.items() if ceiling < old[k]]
    if lowered and not args.allow_lower:
        print(
            f"refusing to lower {', '.join(lowered)}: the runs on hand may be fewer than the "
            "ones the old ceiling came from; pass --allow-lower if they are not",
            file=sys.stderr,
        )
        return 2

    provenance = raw.setdefault("_provenance", {})
    for key, (n, limit, ceiling) in chosen.items():
        raw[key] = ceiling
        provenance[key] = (
            f"{args.issue}: p{pct} of {n} hosted runs = {limit * 1e3:.3f} ms; "
            f"ceiling {ceiling * 1e3:.3f} ms * (1 + {args.tolerance}) puts the limit "
            f"{args.headroom}x above it"
        )
    args.thresholds.write_text(json.dumps(raw, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
