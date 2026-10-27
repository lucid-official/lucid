"""``python -m lucid.test.audit`` — the exhaustive sweep.

Installed as ``lucid-audit``; ``python -m lucid.test.audit`` is the same
program and works without reinstalling.

Examples
--------
::

    lucid-audit                          # everything
    lucid-audit --coverage               # reach and depth, running nothing
    lucid-audit --quick                  # fewer domains, smaller probes
    lucid-audit --axis grad,nonfinite
    lucid-audit --subsystem nn.functional --select 'conv|pool'
    lucid-audit --json audit.json --known lucid/test/audit/known.json
    lucid-audit --list-uncovered         # what the harness cannot reach

Exit status is ``0`` when no defect survived, ``1`` when one did, ``2``
when the harness itself broke.  ``--list-uncovered`` is not decoration:
a census that cannot say what it missed is not a census, and that list is
the work queue for extending :mod:`~lucid.test.audit._specs`.
"""

import argparse
import atexit
import contextlib
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from lucid.test.audit import _axes, _console_rich, _probe, _suite, _surface
from lucid.test.audit._console import Console, Suppress, Timer, fmt_duration
from lucid.test.audit._result import STATUS_STYLE, Baseline, Finding, Report, Status

if TYPE_CHECKING:
    from collections.abc import Sequence

_DEFAULT_KNOWN = Path(__file__).with_name("known.json")
_DEFAULT_COVERAGE = Path(__file__).with_name("coverage.json")
_DEFAULT_SUITE = Path(__file__).with_name("suite.json")
_DEFAULT_DOCTEST = Path(__file__).with_name("doctest.json")
# Beside the suite it reports on, not beside the harness that produces it —
# ``lucid/test/audit.log`` is where someone looking for "what did the last
# run say" will look first.  Git-ignored: the gate is required to leave the
# checkout as it found it, and this is the one file it deliberately writes.
_DEFAULT_LOG = Path(__file__).resolve().parents[1] / "audit.log"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m lucid.test.audit",
        description="Exhaustive correctness sweep over Lucid, excluding the model zoo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Statuses: PASS correct · FAIL a defect · TRNC the finite-difference probe "
            "was the limit, not the op · GAUG defined only up to a symmetry · "
            "VAC the check could not have failed · UNSP the op refused by design · "
            "SKIP the harness could not build inputs · KNWN accepted in the baseline."
        ),
    )
    sel = p.add_argument_group("selection")
    sel.add_argument("--axis", default="all", help="comma-separated axis names, or 'all'")
    sel.add_argument("--subsystem", default="all", help="comma-separated subsystem keys, or 'all'")
    sel.add_argument("--select", default="", help="regular expression over symbol names")
    sel.add_argument("--limit", type=int, default=0, help="stop after N symbols (smoke run)")

    run = p.add_argument_group("run")
    run.add_argument("--quick", action="store_true", help="fewer domains, looser sweep")
    run.add_argument("--no-metal", action="store_true", help="skip every GPU comparison")
    run.add_argument("--step", type=float, default=1e-5, help="finite-difference step")
    run.add_argument("--tolerance", type=float, default=2e-5, help="relative tolerance")
    run.add_argument("--fail-fast", action="store_true", help="stop at the first defect")

    stage = p.add_argument_group("stages")
    stage.add_argument(
        "--audit-only",
        action="store_true",
        help="the probe sweep alone, without the test suite",
    )
    stage.add_argument(
        "--tests-only",
        action="store_true",
        help="the test suite and line-coverage floor alone, without the sweep",
    )
    stage.add_argument(
        "--doctests-only",
        action="store_true",
        help="the docstring-example stage alone, against its recorded floor",
    )
    stage.add_argument(
        "--suite-path",
        default="lucid/test",
        help="what to hand pytest (default: the whole tree; the sweep is "
        "deselected by its marker)",
    )
    stage.add_argument(
        "--no-line-coverage",
        action="store_true",
        help="run the suite without measuring line coverage",
    )
    stage.add_argument(
        "--suite-baseline",
        type=Path,
        default=_DEFAULT_SUITE,
        help="recorded line-coverage floor, compared against on every run",
    )
    stage.add_argument(
        "--suite-ignore",
        action="append",
        default=[],
        metavar="PATH",
        help="a subtree pytest should leave out; repeatable "
        "(e.g. --suite-ignore lucid/test/parity/models)",
    )
    stage.add_argument(
        "--no-doctests",
        action="store_true",
        help="skip the docstring-example stage",
    )
    stage.add_argument(
        "--no-self-check",
        action="store_true",
        help="skip proving the axes can go red",
    )
    stage.add_argument(
        "--doctest-baseline",
        type=Path,
        default=_DEFAULT_DOCTEST,
        help="recorded per-module doctest failures, compared on every run",
    )
    stage.add_argument(
        "--update-doctests",
        action="store_true",
        help="record the current docstring failures as the new floor",
    )
    stage.add_argument(
        "--update-suite",
        action="store_true",
        help="record the current line coverage as the new floor",
    )

    out = p.add_argument_group("output")
    out.add_argument("--json", type=Path, default=None, help="write the full report here")
    out.add_argument("--known", type=Path, default=_DEFAULT_KNOWN, help="baseline file")
    out.add_argument(
        "--coverage-baseline",
        type=Path,
        default=_DEFAULT_COVERAGE,
        help="recorded set of answered cells, compared against on every run",
    )
    out.add_argument(
        "--update-coverage",
        action="store_true",
        help="record the current answered set as the new floor",
    )
    out.add_argument(
        "--no-coverage-diff",
        action="store_true",
        help="skip the comparison against the recorded set",
    )
    out.add_argument(
        "--update-known",
        action="store_true",
        help="fold this run's defects into the baseline as accepted",
    )
    out.add_argument("--no-color", action="store_true")
    out.add_argument("--quiet", action="store_true", help="only the summary and findings")
    out.add_argument(
        "--log",
        type=Path,
        default=_DEFAULT_LOG,
        help=f"mirror the run to this file as it happens (default {_DEFAULT_LOG})",
    )
    out.add_argument(
        "--no-log",
        action="store_true",
        help="do not write the transcript",
    )
    out.add_argument("--width", type=int, default=None)
    out.add_argument("--show", default="fail,error,vacuous", help="statuses to list in full")

    info = p.add_argument_group("information")
    info.add_argument("--list-axes", action="store_true")
    info.add_argument("--list-subsystems", action="store_true")
    info.add_argument("--list-uncovered", action="store_true", help="symbols no axis reached")
    info.add_argument(
        "--self-check",
        action="store_true",
        help="prove each axis can go red, by breaking the framework on purpose",
    )
    info.add_argument(
        "--coverage",
        action="store_true",
        help="measure reach and depth against an independent walk (runs nothing)",
    )
    return p


# ── information modes ────────────────────────────────────────────────────────


def _list_axes(console: Console) -> int:
    console.banner("Audit axes", "each one has already caught a defect in this framework")
    console.table(
        ["axis", "applies to", "question"],
        [[a.name, ", ".join(sorted(a.kinds)), a.summary] for a in _axes.ALL_AXES],
        always=True,
    )
    return 0


def _list_subsystems(console: Console) -> int:
    console.banner("Subsystems", "the audit denominator")
    symbols = _surface.enumerate_surface()
    rows = []
    for key in _surface.SUBSYSTEMS:
        of_key = [s for s in symbols if s.subsystem == key]
        if not of_key:
            continue
        inert = sum(1 for s in of_key if s.inert)
        rows.append(
            [
                key,
                str(len(of_key)),
                str(len(of_key) - inert),
                _surface.SUBSYSTEMS[key][1],
            ]
        )
    console.table(
        ["subsystem", "symbols", "stateful", "kind"],
        rows,
        ["l", "r", "r", "l"],
        always=True,
    )
    console.always("")
    for name, why in _surface.EXCLUDED.items():
        console.always("  " + console.paint(f"excluded  {name}", "grey") + f"  — {why}")
    return 0


def _self_check(console: Console) -> int:
    """Which axes have been shown to produce a red light.

    A green axis is not evidence until it has been made to fail.  This
    breaks the framework on purpose, once per axis, in exactly the way
    that axis exists to notice — and reports the axes nobody has proven
    yet rather than omitting them.
    """
    from lucid.test.audit import _mutants

    console.banner("Self-check", "each axis is handed the defect it claims to catch")
    results = _mutants.verify()
    rows = [
        [
            "caught" if r.caught else "MISSED",
            r.axis,
            r.why,
            r.status if not r.caught else r.detail[:44],
        ]
        for r in results
    ]
    console.table(["", "axis", "the defect", "what the axis said"], rows, always=True)

    missed = [r for r in results if not r.caught]
    unproven = _mutants.unproven_axes()
    console.always("")
    console.always(
        "  "
        + console.paint("proven".ljust(22), "grey")
        + console.paint(f"{len(results) - len(missed)}/{len(results)} mutants caught ", "white")
        + console.paint(
            f"({len({r.axis for r in results}) - len({r.axis for r in missed})}"
            f" of {len(_axes.ALL_AXES)} axes)",
            "cyan",
        )
    )
    if unproven:
        console.always("")
        console.rule(f"unproven · {len(unproven)}", "yellow")
        console.always(
            console.paint(
                "  no mutant exercises these, so nothing here says they can fail:",
                "grey",
            )
        )
        for name in unproven:
            reason = _mutants.UNPROVEN_REASONS.get(name, "no mutant written yet")
            console.always("      " + console.paint(name.ljust(12), "yellow") + reason)
    if missed:
        console.always("")
        console.rule(f"blind · {len(missed)}", "red")
        for r in missed:
            console.always(
                "  "
                + console.paint(r.axis.ljust(12), "red", "bold")
                + f"{r.why} — the axis said {r.status}"
            )
    console.always("")
    return 1 if missed else 0


def _report_coverage(console: Console) -> int:
    """What fraction of Lucid this tool can speak about, measured not claimed.

    Two numbers, because they fail independently and only one of them is
    usually quoted:

    *Reach* is the surface against an **independent** recursive walk of
    the package.  The enumeration used to name its modules in a literal
    and the literal named nineteen; the walk finds a hundred and thirty,
    and the difference was 26% of the framework — including all of
    ``utils.transforms`` and ``nn.init`` — sitting silently outside the
    denominator while the tool reported a coverage figure over the rest.
    Deriving the check from a different traversal than the thing it
    checks is the whole point; a self-consistent count would have agreed
    with itself at 73.8% forever.

    *Depth* is how many reached symbols get an axis that can actually
    fail, rather than only the smoke axis.  Reaching a symbol and calling
    it once is not verifying it, and the two must not be added together.
    """
    console.banner("Coverage", "measured against an independent walk of the package")

    symbols = _surface.enumerate_surface()
    walked = _surface.independent_walk()
    reached = {id(s.obj) for s in symbols}
    absent = {name for oid, name in walked.items() if oid not in reached}
    reach = 100.0 * (len(walked) - len(absent)) / max(len(walked), 1)

    per_symbol = {s.qualname: [a.name for a in _axes.ALL_AXES if a.applies(s)] for s in symbols}
    cells = sum(len(v) for v in per_symbol.values())
    shallow = [s for s in symbols if per_symbol[s.qualname] == ["smoke"]]
    stateful_shallow = sum(1 for s in shallow if not s.inert)
    depth = 100.0 * (len(symbols) - len(shallow)) / max(len(symbols), 1)

    console.table(
        ["measure", "value", "of", "%"],
        [
            [
                "reach — objects the surface enumerates",
                str(len(walked) - len(absent)),
                str(len(walked)),
                f"{reach:.1f}",
            ],
            [
                "depth — symbols with an axis past smoke",
                str(len(symbols) - len(shallow)),
                str(len(symbols)),
                f"{depth:.1f}",
            ],
            ["cells — symbol x applicable axis", str(cells), "", ""],
        ],
        ["l", "r", "r", "r"],
        always=True,
    )
    console.always("")
    if absent:
        console.always(
            console.paint(
                f"  {len(absent)} object(s) the walk found and the surface did not:",
                "red",
            )
        )
        for name in sorted(absent)[:20]:
            console.always(f"      {name}")
    else:
        console.always(
            console.paint("  every public object the walk found is in the surface.", "green")
        )
    console.always(
        f"  {len(shallow)} symbol(s) reach only the smoke axis, of which "
        f"{stateful_shallow} mutate process state and are called under a guard "
        "because no numeric axis can express them."
    )
    console.always("")

    rows = []
    for key in _surface.SUBSYSTEMS:
        of_key = [s for s in symbols if s.subsystem == key]
        if not of_key:
            continue
        thin = sum(1 for s in of_key if per_symbol[s.qualname] == ["smoke"])
        rows.append(
            [
                key,
                str(len(of_key)),
                str(len(of_key) - thin),
                f"{100.0 * (len(of_key) - thin) / len(of_key):.0f}",
                str(sum(len(per_symbol[s.qualname]) for s in of_key)),
            ]
        )
    console.table(
        ["subsystem", "symbols", "with an axis", "%", "cells"],
        rows,
        ["l", "r", "r", "r", "r"],
        always=True,
    )
    return 0


# ── the run ──────────────────────────────────────────────────────────────────


def _selected_axes(spec: str) -> list[_axes.Axis]:
    if spec.strip() in ("all", ""):
        return list(_axes.ALL_AXES)
    wanted = [s.strip() for s in spec.split(",") if s.strip()]
    out = []
    for name in wanted:
        axis = _axes.axis_by_name(name)
        if axis is None:
            raise SystemExit(f"unknown axis {name!r}; known axes: {', '.join(_axes.axis_names())}")
        out.append(axis)
    return out


def _selected_symbols(args: argparse.Namespace) -> list[_surface.Symbol]:
    subsystems = (
        None
        if args.subsystem.strip() in ("all", "")
        else [s.strip() for s in args.subsystem.split(",") if s.strip()]
    )
    symbols = _surface.enumerate_surface(subsystems)
    if args.select:
        pattern = re.compile(args.select)
        symbols = [s for s in symbols if pattern.search(s.qualname)]
    if args.limit:
        symbols = symbols[: args.limit]
    return symbols


def _tally(counts: dict[Status, int], console: Console) -> str:
    order = (
        Status.PASS,
        Status.FAIL,
        Status.VACUOUS,
        Status.TRUNCATION,
        Status.UNSUPPORTED,
        Status.SKIP,
        Status.KNOWN,
        Status.ERROR,
    )
    parts = []
    for status in order:
        n = counts.get(status, 0)
        if not n:
            continue
        label, colour = STATUS_STYLE[status]
        parts.append(console.paint(f"{label.strip().lower()} {n}", colour))
    return "  ".join(parts)


def run(args: argparse.Namespace, console: Console) -> Report:
    axes = _selected_axes(args.axis)
    symbols = _selected_symbols(args)
    metal = (not args.no_metal) and _probe.metal_available()
    ctx = _axes.Context(quick=args.quick, metal=metal, step=args.step, tolerance=args.tolerance)

    report = Report()
    report.surface_total = len(symbols)
    report.axes_run = [a.name for a in axes]
    report.config = {
        "axis": args.axis,
        "subsystem": args.subsystem,
        "select": args.select,
        "quick": args.quick,
        "metal": metal,
        "step": args.step,
        "tolerance": args.tolerance,
    }
    baseline = Baseline.load(args.known)

    console.banner(
        "Lucid — exhaustive audit",
        f"{len(axes)} axes · {len(symbols)} symbols · "
        f"metal {'on' if metal else 'off'} · model zoo excluded",
    )

    total_work = sum(sum(1 for s in symbols if a.applies(s)) for a in axes)
    report.applicable_cells = total_work
    overall = Timer()
    done_total = 0
    tick = 0
    stop = False

    # One bar per subsystem when ``rich`` is installed, one bar per axis
    # otherwise.  ``rich`` is an optional extra, so both paths stay live.
    reporter = _console_rich.build(console, axes, symbols)
    with contextlib.ExitStack() as stack:
        if reporter is not None:
            stack.enter_context(reporter)

        for axis in axes:
            applicable = [s for s in symbols if axis.applies(s)]
            if not applicable:
                continue
            if reporter is None:
                console.rule(f"{axis.name}  —  {axis.summary}", "cyan")
            counts: dict[Status, int] = dict.fromkeys(Status, 0)
            axis_timer = Timer()

            for index, symbol in enumerate(applicable, start=1):
                with Suppress(not console.quiet), _probe.preserved_globals():
                    try:
                        finding = axis.run(symbol, ctx)
                    except KeyboardInterrupt:
                        raise
                    except Exception as exc:  # noqa: BLE001 - the harness must survive the survey
                        finding = Finding(
                            axis.name,
                            symbol.qualname,
                            Status.ERROR,
                            f"probe raised {type(exc).__name__}: {str(exc)[:90]}",
                        )
                finding = baseline.apply(finding)
                report.add(finding)
                counts[finding.status] += 1
                done_total += 1
                tick += 1

                if reporter is not None:
                    # Counted, not printed.  Every defect is listed in
                    # full by ``summarise`` once the sweep is over;
                    # printing them as they arrive scrolled the live
                    # region on each one and said it twice.
                    reporter.record(symbol, axis.name, finding)
                elif finding.status is Status.FAIL and not console.live:
                    label, colour = STATUS_STYLE[Status.FAIL]
                    console.always(
                        "  "
                        + console.paint(label, colour)
                        + f"  {symbol.qualname}  {finding.detail}"
                    )

                if (
                    reporter is None
                    and console.live
                    and (tick % 3 == 0 or index == len(applicable))
                ):
                    console.live_block(
                        [
                            "  "
                            + console.paint(console.spinner(tick), "cyan")
                            + "  "
                            + console.paint(axis.name.ljust(12), "bold")
                            + console.paint(console.bar(index, len(applicable)), "cyan")
                            + f"  {index:>5}/{len(applicable)}"
                            + console.paint(
                                f"  {100 * index // max(len(applicable), 1):>3}%",
                                "grey",
                            )
                            + console.paint(
                                f"   {axis_timer}  "
                                f"eta {axis_timer.eta(index, len(applicable))}",
                                "grey",
                            ),
                            "     " + console.paint(symbol.qualname[: console.width - 8], "grey"),
                            "     " + _tally(counts, console),
                        ]
                    )

                if args.fail_fast and finding.status.is_defect:
                    stop = True
                    break

            if reporter is None:
                if console.live:
                    console.live_done()
                console.write(
                    "  "
                    + console.paint(axis.name.ljust(12), "bold")
                    + console.paint(f"{len(applicable):>5} symbols  ", "grey")
                    + _tally(counts, console)
                    + console.paint(f"   [{axis_timer}]", "grey")
                )
            if stop:
                console.write()
                console.always(
                    console.paint("  stopped at the first defect (--fail-fast)", "yellow")
                )
                break

    report.duration = overall.elapsed
    return report


# ── reporting ────────────────────────────────────────────────────────────────


def summarise(report: Report, console: Console, show: Sequence[str]) -> None:
    console.always("")
    console.rule("summary", "cyan")

    per_axis = report.counts_by_axis()
    rows = []
    for axis in report.axes_run:
        counts = per_axis.get(axis)
        if not counts:
            continue
        rows.append(
            [
                axis,
                str(sum(counts.values())),
                str(counts[Status.PASS]),
                str(counts[Status.FAIL]),
                str(counts[Status.VACUOUS]),
                str(counts[Status.TRUNCATION] + counts[Status.GAUGE]),
                str(counts[Status.UNSUPPORTED]),
                str(counts[Status.SKIP]),
            ]
        )
    console.table(
        ["axis", "run", "pass", "fail", "vacuous", "artefact", "unsupported", "skip"],
        rows,
        ["l", "r", "r", "r", "r", "r", "r", "r"],
        always=True,
    )

    reached, total, frac = report.coverage()
    filled, cells, cell_frac = report.cell_coverage()
    console.always("")
    console.always(
        "  "
        + console.paint("coverage".ljust(22), "grey")
        + console.paint(f"{reached}/{total} symbols reached ", "white")
        + console.paint(f"({100 * frac:.1f}%)", "cyan")
    )
    console.always(
        "  "
        + console.paint("applicable cells".ljust(22), "grey")
        + console.paint(f"{filled}/{cells} produced a verdict ", "white")
        + console.paint(f"({100 * cell_frac:.1f}%)", "cyan")
        + console.paint("   — skips and refusals do not count", "grey")
    )
    console.always(
        "  "
        + console.paint("elapsed".ljust(22), "grey")
        + console.paint(fmt_duration(report.duration), "white")
    )

    wanted = {s.strip().lower() for s in show if s.strip()}
    for status in (Status.FAIL, Status.ERROR, Status.VACUOUS, Status.KNOWN):
        if status.value not in wanted:
            continue
        items = report.by_status(status)
        if not items:
            continue
        label, colour = STATUS_STYLE[status]
        console.always("")
        console.rule(f"{label.strip()} · {len(items)}", colour)
        for f in items:
            console.always(
                "  "
                + console.paint(label, colour)
                + "  "
                # The gap is part of the string being padded, not padding
                # that a long name can consume.  ``ljust`` on a name wider
                # than the column returns it unchanged, which ran symbol
                # and detail together for every name over 34 characters —
                # and the widest names are the deeply-qualified ones whose
                # detail matters most.
                + console.paint(f"{f.symbol}  ".ljust(34), "white")
                + console.paint(f.detail, "grey")
            )

    console.always("")
    defects = report.defects
    if defects:
        console.always(
            console.paint(f"  {len(defects)} defect(s) — see the FAIL list above", "red", "bold")
        )
    else:
        console.always(console.paint("  no defects survived verification", "green", "bold"))


# ── coverage baseline ────────────────────────────────────────────────────────


def _answered(report: Report) -> dict[str, str]:
    """``axis::symbol -> status`` for every cell that produced a verdict.

    Only the verdicts.  ``skip`` and ``unsupported`` move around with
    probe details that have nothing to do with the framework, and a
    baseline that churns is a baseline nobody re-reads.
    """
    out: dict[str, str] = {}
    for finding in report.findings:
        if finding.status in (Status.PASS, Status.FAIL, Status.ERROR, Status.VACUOUS):
            out[f"{finding.axis}::{finding.symbol}"] = finding.status.value
    return out


def load_coverage(path: Path) -> dict[str, str] | None:
    """The recorded set, or ``None`` when there is nothing to compare to."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    answered = data.get("answered")
    return answered if isinstance(answered, dict) else None


def save_coverage(path: Path, report: Report) -> int:
    answered = _answered(report)
    payload = {
        "comment": [
            "Which (axis, symbol) cells produced a verdict, and which.",
            "",
            "The audit reports absolute counts, and an absolute count cannot",
            "show a regression: an op that stops being reachable moves one",
            "cell from 'pass' to 'unsupported' among fifteen hundred, and",
            "nothing in the summary changes visibly.  This file is what a run",
            "is compared against, so 'did my change break something' has an",
            "answer that does not depend on remembering yesterday's numbers.",
            "",
            "Regenerate with:  lucid-audit --update-coverage",
            "Do that only when the new state is the intended one.",
        ],
        "cells": len(answered),
        "answered": dict(sorted(answered.items())),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return len(answered)


def _scoped_coverage(
    recorded: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, str]:
    """A filtered sweep cannot answer cells outside the requested scope.

    Keep the complete baseline for full runs, including removed symbols,
    which the diff names as REMOVED rather than dropping without a word.
    A scoped run compares only the selected symbols and axes, including
    selected cells that now skip or refuse their input.
    """
    axes = {axis.name for axis in _selected_axes(args.axis)}
    scoped_symbols = args.subsystem.strip() not in ("all", "") or args.select or args.limit
    symbols = {symbol.qualname for symbol in _selected_symbols(args)} if scoped_symbols else None
    return {
        key: value
        for key, value in recorded.items()
        if key.partition("::")[0] in axes and (symbols is None or key.partition("::")[2] in symbols)
    }


def _removed_cells(report: Report, unanswered: set[str]) -> dict[str, str]:
    """The unanswered cells a removal from the public surface accounts for.

    ``key -> why``.  LOST says the op is still exported and the audit can
    no longer reach it; a symbol taken out on purpose is neither.  LCD-473
    removed ``lucid.pad`` and ``Tensor.pad`` and updated the released
    surface, and their 24 cells, reported as LOST, held the gate red over a
    deliberate change.  The entry axis answers a symbol by comparing it with
    its other spellings, so once all but one are gone its question has no
    answer either — ``entry::F.pad`` went with them.
    """
    gone = _surface.removed(key.partition("::")[2] for key in unanswered)
    if not gone:
        return {}
    now = {finding.key: finding.status for finding in report.findings}
    out: dict[str, str] = {}
    for key in unanswered:
        axis, _, symbol = key.partition("::")
        if symbol in gone:
            out[key] = f"{symbol} is no longer public"
            continue
        spelled = _surface.spellings(symbol.rpartition(".")[2]) & gone
        if axis == _axes.EntryPointAxis.name and spelled and now.get(key) is Status.NOT_APPLICABLE:
            out[key] = f"the spelling it was compared with, {min(spelled)}, is gone"
    return out


def _capped(console: Console, label: str, colour: str, lines: list[str]) -> None:
    for line in lines[:20]:
        console.always("  " + console.paint(label, colour) + f" {line}")
    if len(lines) > 20:
        console.always(console.paint(f"       ... and {len(lines) - 20} more", "grey"))


def report_coverage_diff(report: Report, recorded: dict[str, str], console: Console) -> int:
    """Print what moved since the baseline.  Returns the regression count.

    A cell that answered and no longer does is the interesting direction:
    the op is still there, the audit simply cannot reach it any more,
    which is what a refactor breaks without breaking a test.  A cell whose
    symbol has left the public surface is reported as REMOVED and is not
    a regression: the released-surface gate is what judges a removal.
    """
    current = _answered(report)
    unanswered = set(recorded) - set(current)
    removed = _removed_cells(report, unanswered)
    lost = sorted(unanswered - set(removed))
    gained = sorted(set(current) - set(recorded))
    worsened = sorted(
        key
        for key in set(recorded) & set(current)
        if recorded[key] == "pass" and current[key] != "pass"
    )

    if not (lost or gained or worsened or removed):
        console.always("")
        console.always(console.paint("  coverage unchanged against the baseline", "green"))
        return 0

    if lost or worsened:
        console.always("")
        console.rule(f"coverage regressions · {len(lost) + len(worsened)}", "red")
        for key in lost:
            console.always(
                "  " + console.paint("LOST ", "red", "bold") + f" {key}"
                f"  (was {recorded[key]}, now unanswered)"
            )
        for key in worsened:
            console.always(
                "  " + console.paint("WORSE", "red", "bold") + f" {key}  (pass -> {current[key]})"
            )
    if removed:
        console.always("")
        console.rule(f"removed from the surface · {len(removed)}", "grey")
        _capped(
            console,
            "REMOVED",
            "grey",
            [f"{key}  (was {recorded[key]}; {why})" for key, why in sorted(removed.items())],
        )
    if gained:
        console.always("")
        console.rule(f"newly answered · {len(gained)}", "green")
        _capped(console, "NEW  ", "green", gained)
    if removed or gained:
        console.always(
            console.paint("  run with --update-coverage to record the new floor", "grey")
        )
    return len(lost) + len(worsened)


def report_uncovered(console: Console, args: argparse.Namespace) -> int:
    """What the harness could not build inputs for — the work queue."""
    from lucid.test.audit import _autospec, _specs

    symbols = _selected_symbols(args)
    ops = [s for s in symbols if s.kind in ("op", "method") and s.inert]
    without = [s for s in ops if not _specs.has_spec(s.short, _surface.resolve(s))]

    console.banner(
        "Symbols with no invocation, hand-written or derived",
        "not even their own signature says enough to call them",
    )
    by_sub: dict[str, list[str]] = {}
    for s in without:
        by_sub.setdefault(s.subsystem, []).append(s.short)
    rows = [
        [
            key,
            str(len(names)),
            ", ".join(sorted(names)[:6]) + ("…" if len(names) > 6 else ""),
        ]
        for key, names in sorted(by_sub.items())
    ]
    console.table(["subsystem", "count", "examples"], rows, ["l", "r", "l"], always=True)
    console.always("")
    console.always(
        console.paint(
            f"  {len(without)} of {len(ops)} callable symbols fall through every tier.",
            "yellow" if without else "green",
        )
    )
    for symbol in without[:12]:
        console.always(
            "      "
            + console.paint(symbol.qualname, "grey")
            + f"  — {_autospec.explain(_surface.resolve(symbol), symbol.short)}"
        )
    console.always("")
    console.always(
        console.paint(
            "  Each reason names the parameter that stopped it.  A required argument",
            "grey",
        )
    )
    console.always(
        console.paint(
            "  the derivation has no value for is closed by one entry in _autospec's",
            "grey",
        )
    )
    console.always(
        console.paint(
            "  name table, which then covers every other op that spells it the same",
            "grey",
        )
    )
    console.always(
        console.paint(
            "  way — writing a per-op spec is the last resort, not the first.",
            "grey",
        )
    )
    console.always(
        console.paint(
            "  Reaching a symbol is still not verifying it: the authoritative gap is",
            "grey",
        )
    )
    console.always(
        console.paint(
            "  the SKIP list of an actual run, from `--json report.json`.",
            "grey",
        )
    )
    return 0


# ── entry point ──────────────────────────────────────────────────────────────


def _run_audit_stage(args: argparse.Namespace, console: Console) -> tuple[int, int]:
    """The probe sweep.  Returns ``(defects, coverage-cell regressions)``.

    The record-a-new-baseline modes return zero for whatever they just
    accepted and let the run continue: the point of folding the stages
    together is one verdict, and a stage that exits early takes the other
    stage's answer with it.
    """
    started = time.time()
    report = run(args, console)
    report.started = started
    summarise(report, console, args.show.split(","))

    if args.json is not None:
        report.write_json(args.json)
        console.always(console.paint(f"  report written to {args.json}", "grey"))

    # Recording a new baseline used to exit here.  With two stages that
    # silences the other one: ``--update-suite`` would record a floor and
    # report success while the sweep sat on a live defect.  Record, then
    # carry on to the verdict — the recorded stage simply contributes
    # nothing to it, which is what "accepted" means.
    if args.update_coverage:
        recorded = save_coverage(args.coverage_baseline, report)
        console.always(
            console.paint(
                f"  {recorded} answered cells recorded in {args.coverage_baseline}",
                "grey",
            )
        )
        return 0, 0

    regressions = 0
    if not args.no_coverage_diff:
        recorded_cells = load_coverage(args.coverage_baseline)
        if recorded_cells is None:
            console.always("")
            console.always(
                console.paint(
                    "  no coverage baseline yet — run --update-coverage to record one",
                    "grey",
                )
            )
        else:
            regressions = report_coverage_diff(
                report, _scoped_coverage(recorded_cells, args), console
            )

    if args.update_known:
        Baseline.load(args.known).save(args.known, report.defects)
        console.always(
            console.paint(f"  {len(report.defects)} defect(s) folded into {args.known}", "grey")
        )
        return 0, regressions

    return len(report.defects), regressions


def _run_suite_stage(args: argparse.Namespace, console: Console) -> tuple[int, int]:
    """The test suite and its line-coverage floor.

    Returns ``(broken tests, line-coverage regressions)``.
    """
    console.banner("Suite", "the values, where the sweep checks the contracts")
    result = _suite.run_suite(
        console,
        args.suite_path,
        with_coverage=not args.no_line_coverage,
        ignore=args.suite_ignore,
    )
    _suite.report_suite(result, console)

    if not result.ran:
        return 0, 0

    if args.update_suite:
        if result.percent is None:
            console.always(
                console.paint("  nothing to record — the run measured no coverage", "grey")
            )
        else:
            _suite.save_floor(args.suite_baseline, result)
            console.always(
                console.paint(
                    f"  {result.percent:.2f}% recorded in {args.suite_baseline}",
                    "grey",
                )
            )
        # A recorded floor is not a licence to ignore a broken test — the
        # two are independent, and only the floor was just accepted.
        return result.broken, 0

    regressions = 0
    if result.percent is not None:
        floor = _suite.load_floor(args.suite_baseline)
        if floor is None:
            console.always("")
            console.always(
                console.paint(
                    "  no line-coverage floor yet — run --update-suite to record one",
                    "grey",
                )
            )
        else:
            regressions = _suite.report_line_coverage_diff(result, floor, console)

    return result.broken, regressions


def _run_doctest_stage(args: argparse.Namespace, console: Console) -> int:
    """The documentation, against the code it documents.

    Returns the number of modules whose docstrings got *worse*.  The
    standing failures are a recorded backlog: a stage that is red on
    arrival teaches people to pass it with a flag, and the number that
    matters for a gate is whether this change made it grow.
    """
    from lucid.test.audit import _doctests

    console.banner("Docstrings", "every documented example, run as written")
    result = _doctests.run()
    if not result.ran or not result.attempted:
        console.always(console.paint("  no docstring examples found", "grey"))
        return 0

    console.always(
        "  "
        + console.paint("examples".ljust(22), "grey")
        + console.paint(
            f"{result.attempted - result.failed}/{result.attempted} run as written ",
            "white",
        )
        + console.paint(
            f"({100 * (result.attempted - result.failed) / result.attempted:.1f}%)",
            "cyan",
        )
    )

    if args.update_doctests:
        _doctests.save_floor(args.doctest_baseline, result)
        console.always(
            console.paint(
                f"  {result.failed} failing example(s) in {len(result.per_module)} "
                f"module(s) recorded in {args.doctest_baseline}",
                "grey",
            )
        )
        return 0

    floor = _doctests.load_floor(args.doctest_baseline)
    if floor is None:
        console.always("")
        console.always(
            console.paint("  no doctest floor yet — run --update-doctests to record one", "grey")
        )
        return 0
    worse = _doctests.regressions(result, floor)
    if not worse:
        console.always(
            "  "
            + console.paint("against the floor".ljust(22), "grey")
            + console.paint("no module got worse", "green")
        )
        return 0
    console.always("")
    console.rule(f"docstring regressions · {len(worse)}", "red")
    for name, was, now in worse:
        console.always(
            "  "
            + console.paint("WORSE", "red", "bold")
            + f"  {name}  ({was} -> {now} failing examples)"
        )
    return len(worse)


def _run_self_check_stage(console: Console) -> int:
    """Whether the instruments can still produce a red light.

    Cheap — no framework symbol is swept, only the mutants — and it is
    the one stage without which the others cannot be believed: a blind
    axis and a clean framework print the same thing.
    """
    from lucid.test.audit import _mutants

    console.banner("Self-check", "each axis is handed the defect it claims to catch")
    results = _mutants.verify()
    missed = [r for r in results if not r.caught]
    console.always(
        "  "
        + console.paint("axes proven".ljust(22), "grey")
        + console.paint(f"{len(results) - len(missed)}/{len(results)} mutants caught ", "white")
        + console.paint(f"({len(_mutants.unproven_axes())} axes unproven, by design)", "cyan")
    )
    if missed:
        console.always("")
        console.rule(f"blind · {len(missed)}", "red")
        for verdict in missed:
            console.always(
                "  "
                + console.paint(verdict.axis.ljust(12), "red", "bold")
                + f"{verdict.why} — the axis said {verdict.status}"
            )
    return len(missed)


def _verdict(console: Console, tallies: dict[str, int]) -> int:
    """One line per stage, then the answer.

    The whole point of folding the two stages together is that a single
    exit code means something.  Printing the parts as well means a red
    run says *which* half went red without anyone re-reading the scroll.
    """
    console.always("")
    console.rule("verdict", "red" if any(tallies.values()) else "green")
    for label, count in tallies.items():
        console.always(
            console.paint(f"  {label}".ljust(34), "grey")
            + console.paint(str(count), "red" if count else "green")
        )
    total = sum(tallies.values())
    console.always("")
    if total:
        console.always(console.paint(f"  {total} problem(s) — see above", "red", "bold"))
    else:
        console.always(console.paint("  clean on every stage that ran", "green", "bold"))
    return 1 if total else 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # The informational modes answer a question about the harness rather
    # than running the gate.  They get no transcript: the log is truncated
    # on open, and losing the last real run to a ``--list-axes`` typed
    # afterwards is the one way this feature could destroy what it exists
    # to preserve.
    informational = bool(
        args.list_axes
        or args.list_subsystems
        or args.self_check
        or args.coverage
        or args.list_uncovered
    )
    console = Console(
        colour=False if args.no_color else None,
        width=args.width,
        quiet=args.quiet,
        log=None if (args.no_log or informational) else args.log,
    )
    if console.log_path is not None:
        atexit.register(console.close)
        stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        console.note(f"# lucid-audit  {stamp}")
        console.note("# " + " ".join([Path(sys.argv[0]).name, *(argv or sys.argv[1:])]))

    if args.list_axes:
        return _list_axes(console)
    if args.list_subsystems:
        return _list_subsystems(console)
    if args.self_check:
        return _self_check(console)
    if args.coverage:
        return _report_coverage(console)
    if args.list_uncovered:
        return report_uncovered(console, args)

    if sum((args.audit_only, args.tests_only, args.doctests_only)) > 1:
        console.always(
            console.paint(
                "  --audit-only, --tests-only and --doctests-only exclude each "
                "other; pass none to run everything",
                "red",
            )
        )
        return 2

    if args.update_coverage and (
        args.axis.strip() not in ("all", "")
        or args.subsystem.strip() not in ("all", "")
        or args.select
        or args.limit
        or args.fail_fast
        or args.no_metal
        or args.quick
    ):
        console.always("--update-coverage requires a complete, unfiltered sweep")
        return 2

    # Both stages by default.  A gate that has to be invoked twice is a
    # gate that gets invoked once, and this session's defects split
    # roughly evenly between what each stage can see.
    do_audit = not (args.tests_only or args.doctests_only)
    do_suite = not (args.audit_only or args.doctests_only)
    # The doctest stage rides with the suite, and alone under
    # --doctests-only.  CI's gate ran the sweep with --audit-only, which
    # skipped the examples, and 63 of them shipped without ever running.
    do_doctests = (do_suite or args.doctests_only) and not args.no_doctests

    tallies: dict[str, int] = {}
    try:
        # The self-check runs *first*, on a clean interpreter.
        #
        # Ordering is load-bearing, not cosmetic.  Run after the sweep,
        # four mutants stopped being caught — the sweep leaves device and
        # module state behind, and the axes that need Metal or a freshly
        # built module then answered UNSUPPORTED instead of FAIL.  An
        # instrument check that the run itself perturbs proves nothing,
        # and proving the instruments before using them is the order the
        # argument needs anyway.
        if do_audit and not args.no_self_check:
            tallies["blind axes"] = _run_self_check_stage(console)
        if do_audit:
            defects, cell_regressions = _run_audit_stage(args, console)
            tallies["audit defects"] = defects
            tallies["audit coverage regressions"] = cell_regressions
        if do_suite:
            broken, line_regressions = _run_suite_stage(args, console)
            tallies["suite failures"] = broken
            tallies["line coverage regressions"] = line_regressions
        if do_doctests:
            tallies["docstring regressions"] = _run_doctest_stage(args, console)
    except KeyboardInterrupt:
        console.always("")
        console.always(console.paint("  interrupted — no report written", "yellow"))
        return 130
    except SystemExit as exit_:
        return int(exit_.code or 0)
    except Exception as exc:  # noqa: BLE001
        console.always("")
        console.always(console.paint(f"  the audit harness failed: {exc!r}", "red", "bold"))
        return 2

    return _verdict(console, tallies)


if __name__ == "__main__":
    sys.exit(main())
