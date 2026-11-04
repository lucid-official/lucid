"""The gate's modes: check, update, rebaseline, diff, report."""

import io
import subprocess
import sys
import tarfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from tools.quality_gate import config
from tools.quality_gate.collectors import Collector
from tools.quality_gate.core import (
    Baseline,
    Counts,
    Delta,
    GateError,
    Key,
    apply_renames,
    baseline_commit,
    baseline_raises,
    changed_files,
    class_totals,
    git,
    git_env,
    load_baseline,
    rebaseline_entry,
    renames,
    rule_totals,
    save_baseline,
    scope_of,
    tracked_files,
)

# tool -> files it measured (None: every file)
Measured = dict[str, set[str] | None]


@dataclass
class Run:
    counts: Counts
    measured: Measured


def measure(root: Path, collectors: list[Collector], files: list[str] | None, workdir: Path) -> Run:
    """Run *collectors* on *root*; local ones on *files* when given, others on everything."""
    workdir.mkdir(parents=True, exist_ok=True)
    everything = scope_of(root, tracked_files(root), workdir)
    subset = everything if files is None else scope_of(root, files, workdir)
    counts: Counts = {}
    measured: Measured = {}
    for c in collectors:
        scope = subset if c.local else everything
        print(f"  [{c.name}] {len(scope.files)} file(s)…", file=sys.stderr, flush=True)
        counts.update(c.collect(scope))
        measured[c.name] = None if (files is None or not c.local) else set(files)
    return Run(counts, measured)


def restrict(counts: Counts, measured: Measured) -> Counts:
    out: Counts = {}
    for k, n in counts.items():
        if k[0] in measured:
            files = measured[k[0]]
            if files is None or k[2] in files:
                out[k] = n
    return out


def _fmt_key(k: Key) -> str:
    return f"{k[0]:9s} {k[1]:28s} {k[2]}"


def print_delta(title: str, delta: Delta, limit: int = 60) -> None:
    before, after = class_totals(delta.before), class_totals(delta.after)
    print(title)
    print(f"  {'class':24s} {'before':>8s} {'after':>8s} {'delta':>7s}")
    for cls, label in config.CLASSES.items():
        d = after[cls] - before[cls]
        mark = "  ← INCREASE" if d > 0 else ""
        print(f"  {cls + ' ' + label:24s} {before[cls]:8d} {after[cls]:8d} {d:+7d}{mark}")
    inc, dec = delta.increases, delta.decreases
    if inc:
        print(f"  increases ({len(inc)}):")
        for k, (b, a) in list(inc.items())[:limit]:
            print(f"    + {_fmt_key(k)}  {b} -> {a}")
    if dec:
        print(f"  decreases ({len(dec)}):")
        for k, (b, a) in list(dec.items())[:limit]:
            print(f"    - {_fmt_key(k)}  {b} -> {a}")
        if len(dec) > limit:
            print(f"    … {len(dec) - limit} more")


def _baseline_view(
    root: Path, baseline: Baseline, measured: Measured
) -> tuple[Counts, dict[str, str]]:
    """The baseline with renames since it was written applied, cut to what was measured."""
    since = baseline_commit(root)
    moved = renames(root, since) if since else {}
    listed = {k[2] for k in baseline.counts}
    in_scope = set().union(*(f for f in measured.values() if f is not None))
    whole = any(f is None for f in measured.values())
    stale = {old: new for old, new in moved.items() if old in listed and (whole or new in in_scope)}
    return restrict(apply_renames(baseline.counts, moved), measured), stale


def _warn_versions(baseline: Baseline, collectors: Iterable[Collector]) -> None:
    for c in collectors:
        want = baseline.tools.get(c.name)
        if want is not None and want != (have := c.version()):
            print(
                f"⚠️  {c.name} {have} differs from the baseline's {want}; counts may shift",
                file=sys.stderr,
            )


def contract_drift(baseline: Baseline, collectors: Iterable[Collector]) -> list[str]:
    """The contract entries of *collectors* the baseline recorded otherwise (or not at all)."""
    contract = config.contract()
    drift: list[str] = []
    for c in collectors:
        for key, value in contract.get(c.name, {}).items():
            if key not in baseline.thresholds:
                drift.append(f"{c.name:9s} {key}: not recorded -> {value!r}")
            elif baseline.thresholds[key] != value:
                drift.append(f"{c.name:9s} {key}: {baseline.thresholds[key]!r} -> {value!r}")
    return drift


def _print_drift(drift: list[str]) -> None:
    print(f"\n✗ the gate's contract differs from the one the baseline records ({len(drift)}):")
    for line in drift:
        print(f"    {line}")
    print(
        "  Counts taken under another contract are not comparable, so a change to\n"
        "  config.py, the vulture whitelist or an allowlist ships with (orchestrator only)\n"
        '    .venv/bin/python3 -m tools.quality_gate --rebaseline --reason "<why>"'
    )


def _print_unrecorded_raises(raises: dict[Key, tuple[int, int]]) -> None:
    print(f"\n✗ the baseline raises {len(raises)} key(s) without a rebaselines entry:")
    for k, (b, a) in list(raises.items())[:40]:
        print(f"    + {_fmt_key(k)}  {b} -> {a}")
    print(
        "  The baseline only goes down; raising a count requires (orchestrator only)\n"
        '    .venv/bin/python3 -m tools.quality_gate --rebaseline --reason "<why>"'
    )


def check(
    root: Path,
    collectors: list[Collector],
    files: list[str] | None,
    workdir: Path,
    allow_slack: bool = False,
    since: str = "HEAD",
) -> int:
    """Fail when a count rose; also when one fell unrecorded, unless *allow_slack*.

    A baseline raised anywhere since *since* needs its rebaselines entry: a bypassed
    commit can raise a count and the baseline together, which HEAD alone cannot see.
    """
    baseline_path = root / config.BASELINE_PATH
    baseline = load_baseline(baseline_path)
    _warn_versions(baseline, collectors)
    if drift := contract_drift(baseline, collectors):
        _print_drift(drift)
        return 1
    if raises := baseline_raises(root, baseline, since):
        _print_unrecorded_raises(raises)
        return 1
    run = measure(root, collectors, files, workdir)
    before, stale = _baseline_view(root, baseline, run.measured)
    delta = Delta(before, restrict(run.counts, run.measured))
    print_delta("QUALITY GATE vs tools/quality_baseline.json", delta)
    if delta.increases:
        print(
            f"\n✗ {len(delta.increases)} count(s) rose above the baseline — new slop. "
            "Fix it at its owner (lucid-worker.md §2 'slop 금지'); the baseline only goes down."
        )
        return 1
    if delta.decreases or stale:
        for old, new in stale.items():
            print(f"  moved: {old} -> {new}")
        record = (
            "    .venv/bin/python3 -m tools.quality_gate --update"
            + ("" if files is None else " --fast")
            + f"\n    git add {config.BASELINE_PATH}"
        )
        if allow_slack:
            print(f"\n⚠️  the baseline is higher than the code; lower it:\n{record}")
            print("✓ no count rose")
            return 0
        print(
            "\n✗ the baseline is out of date (counts fell or files moved). Record it in this "
            f"commit:\n{record}"
        )
        return 1
    print("\n✓ no count rose; baseline is current")
    return 0


def update(root: Path, collectors: list[Collector], files: list[str] | None, workdir: Path) -> int:
    """Lower the baseline to the measured counts; refuse if any count rose."""
    baseline_path = root / config.BASELINE_PATH
    baseline = load_baseline(baseline_path)
    if drift := contract_drift(baseline, collectors):
        # A looser contract lowers counts; recording that as a decrease is DS-18.
        _print_drift(drift)
        return 1
    if raises := baseline_raises(root, baseline):
        _print_unrecorded_raises(raises)
        return 1
    run = measure(root, collectors, files, workdir)
    since = baseline_commit(root)
    moved = renames(root, since) if since else {}
    moved_counts = apply_renames(baseline.counts, moved)
    before = restrict(moved_counts, run.measured)
    delta = Delta(before, restrict(run.counts, run.measured))
    if delta.increases:
        print_delta("QUALITY GATE --update refused", delta)
        print("\n✗ --update only lowers counts; fix the increases above first.")
        return 1
    # Keys outside this run's reach stay; keys inside take the measured count
    # (never higher — refused above), and a count of zero drops the key.
    untouched = {k: n for k, n in moved_counts.items() if k not in before}
    remeasured = {k: n for k, n in run.counts.items() if n > 0 and k in before}
    baseline.counts = untouched | remeasured
    for c in collectors:
        baseline.tools[c.name] = c.version()
    save_baseline(baseline_path, baseline)
    dec = delta.decreases
    print(f"✓ baseline lowered at {len(dec)} key(s); {len(moved)} move(s) applied")
    for k, (b, a) in list(dec.items())[:40]:
        print(f"    - {_fmt_key(k)}  {b} -> {a}")
    return 0


def _recorded_contract(recorded: dict[str, object], ran: list[Collector]) -> dict[str, object]:
    """The contract a rebaseline writes: current for the collectors it measured and
    for entries not yet recorded, as recorded for the rest — a narrowed
    ``--collectors`` run must not bless another collector's changed contract."""
    contract = config.contract()
    measured = {c.name for c in ran}
    return {
        key: value if name in measured or key not in recorded else recorded[key]
        for name, entries in contract.items()
        for key, value in entries.items()
    }


def rebaseline(root: Path, collectors: list[Collector], reason: str, workdir: Path) -> int:
    """Write the measured counts as they are — raises included.  Orchestrator only."""
    if not reason.strip():
        raise GateError("--rebaseline needs --reason")
    baseline_path = root / config.BASELINE_PATH
    baseline = load_baseline(baseline_path)
    run = measure(root, collectors, None, workdir)
    kept = {k: n for k, n in baseline.counts.items() if k[0] not in run.measured}
    old = restrict(baseline.counts, run.measured)
    baseline.counts = {**kept, **{k: n for k, n in run.counts.items() if n > 0}}
    for c in collectors:
        baseline.tools[c.name] = c.version()
    baseline.thresholds = _recorded_contract(baseline.thresholds, collectors)
    baseline.rebaselines.append(rebaseline_entry(reason))
    save_baseline(baseline_path, baseline)
    print_delta(f"REBASELINE ({reason})", Delta(old, restrict(run.counts, run.measured)))
    print(f"\n✓ wrote {config.BASELINE_PATH}")
    return 0


def _extract(root: Path, ref: str, dest: Path) -> None:
    """Materialise *ref*'s tree (the parts the gate reads) without a git worktree."""
    present = git(root, "ls-tree", "--name-only", ref).split("\n")
    # tools/: mypy follows lucid/'s imports into it and counts what it finds.
    wanted = [
        p for p in ("lucid", "tools", "pyproject.toml", "mypy.ini", ".gitignore") if p in present
    ]
    proc = subprocess.run(
        ["git", "archive", "--format=tar", ref, "--", *wanted],
        cwd=root,
        env=git_env(),
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise GateError(f"git archive {ref} failed: {proc.stderr.decode()[-500:]}")
    dest.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(proc.stdout)) as tar:
        tar.extractall(dest, filter="data")


def diff(root: Path, collectors: list[Collector], ref: str, workdir: Path) -> int:
    """SLOP DELTA of the working tree against *ref*, then the baseline check.

    Local collectors measure only the files that differ from *ref* (on both
    sides); cross-file ones measure both whole trees.  A file moved since
    *ref* keeps its counts under its new name, so a move is not new debt.
    """
    sha = git(root, "rev-parse", "--verify", f"{ref}^{{commit}}").strip()
    changed = changed_files(root, sha)
    moved = renames(root, sha)
    ref_root = workdir / "ref"
    _extract(root, sha, ref_root)
    old_files = [moved_from for moved_from in changed if (ref_root / moved_from).is_file()]
    print(f"SLOP DELTA: measuring {ref} ({sha[:9]})…", file=sys.stderr)
    ref_run = measure(ref_root, collectors, old_files, workdir / "w-ref")
    print("SLOP DELTA: measuring the working tree…", file=sys.stderr)
    head_run = measure(root, collectors, changed, workdir / "w-head")
    head_measured: Measured = {
        t: (None if files is None else files | set(moved.values()))
        for t, files in head_run.measured.items()
    }
    before = restrict(apply_renames(ref_run.counts, moved), head_measured)
    after = restrict(head_run.counts, head_measured)
    delta = Delta(before, after)
    names = ", ".join(c.name for c in collectors)
    print_delta(f"SLOP DELTA vs {ref} ({sha[:9]}) — {names}", delta)

    baseline = load_baseline(root / config.BASELINE_PATH)
    base_view, _ = _baseline_view(root, baseline, head_measured)
    over = Delta(base_view, after)
    if over.increases:
        print(f"  over the committed baseline ({len(over.increases)}):")
        for k, (b, a) in over.increases.items():
            print(f"    ! {_fmt_key(k)}  baseline {b}, actual {a}")
    slack = sum(b - a for b, a in over.decreases.values())
    if slack:
        print(f"  note: the baseline can be lowered by {slack} (run --update)")
    drift = contract_drift(baseline, collectors)
    if drift:
        _print_drift(drift)
    raises = baseline_raises(root, baseline, sha) | baseline_raises(root, baseline)
    if raises:
        _print_unrecorded_raises(raises)
    bad = len(delta.increases) + len(over.increases) + len(drift) + len(raises)
    verdict = "✗" if bad else "✓"
    print(
        f"RESULT {verdict}: {len(delta.increases)} increase(s) vs {ref}, "
        f"{len(over.increases)} over baseline, {len(delta.decreases)} decrease(s), "
        f"{len(drift)} contract change(s)"
    )
    return 1 if bad else 0


def report(root: Path, collectors: list[Collector], workdir: Path, top: int) -> int:
    run = measure(root, collectors, None, workdir)
    totals = class_totals(run.counts)
    files_by_class: dict[str, set[str]] = {cls: set() for cls in config.CLASSES}
    for tool, rule, path in run.counts:
        files_by_class[config.class_of(tool, rule)].add(path)
    print(f"QUALITY REPORT — {', '.join(c.name for c in collectors)}")
    print(f"  {'class':24s} {'count':>8s} {'files':>6s}")
    for cls, label in config.CLASSES.items():
        print(f"  {cls + ' ' + label:24s} {totals[cls]:8d} {len(files_by_class[cls]):6d}")
    print(f"  {'total':24s} {sum(totals.values()):8d}")
    print(f"\n  top {top} rules:")
    ranked = sorted(rule_totals(run.counts).items(), key=lambda kv: -kv[1])[:top]
    for (tool, rule), n in ranked:
        print(f"    {n:6d}  {config.class_of(tool, rule):3s} {tool:9s} {rule}")
    return 0
