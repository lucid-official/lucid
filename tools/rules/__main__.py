"""python -m tools.rules: list the rules, show one, run a level's checks, audit the claims."""

import argparse
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Final

from tools.rules.audit import Audit
from tools.rules.checks import CannotMeasure, changed_files, run_all
from tools.rules.model import Exit, Level, Rule, RuleError, Verdict
from tools.rules.registry import Registry

_ROOT: Final = Path(__file__).resolve().parents[2]


def _levels(rule: Rule) -> str:
    return " ".join(f"{e.level}{'?' if e.pending else ''}" for e in rule.enforce) or "-"


def _list(registry: Registry, args: argparse.Namespace) -> Exit:
    rules = list(registry.rules.values())
    for rule in rules:
        columns = f"{rule.verdict:<10s}{_levels(rule):<26s}" if args.verdict else ""
        print(f"{rule.id:<14s}{columns}{rule.statement.strip().splitlines()[0]}")
    if args.verdict:
        counts = Counter(rule.verdict for rule in rules)
        print(", ".join(f"{counts[v]} {v}" for v in Verdict), f"of {len(rules)}")
    return Exit.PASS


def _show(registry: Registry, args: argparse.Namespace) -> Exit:
    rule = registry.rules.get(args.id)
    if rule is None:
        print(f"[rules] no rule {args.id}", file=sys.stderr)
        return Exit.UNMEASURED
    print(f"verdict: {rule.verdict}")
    for name, value in asdict(rule).items():
        if value:
            print(f"{name}: {str(value).strip()}")
    return Exit.PASS


def _run(registry: Registry, args: argparse.Namespace) -> Exit:
    ws = registry.workspace
    try:
        changes = changed_files(ws, args.range, args.staged)
    except CannotMeasure as exc:
        print(f"[rules] cannot measure: {exc}", file=sys.stderr)
        return Exit.UNMEASURED
    named = registry.at_level(Level(args.level))
    if not named:
        print(f"[rules] nothing claims level {args.level}; running nothing proves nothing")
        return Exit.UNMEASURED
    what = "the whole tree" if changes is None else f"{len(changes.paths)} changed file(s)"
    print(f"level {args.level}: {len(named)} check(s) over {what}")
    results = run_all([registry.checks[name] for name in named], ws, changes)
    for result, rules in zip(results, named.values(), strict=True):
        word = "SKIP" if result.skipped else result.outcome.name
        note = result.skipped or f"{result.seconds:.1f}s"
        print(f"  {word:<11s}{result.check:<20s}[{' '.join(r.id for r in rules)}]  {note}")
        if result.outcome is not Exit.PASS:
            print("".join(f"      {line}\n" for line in result.output.splitlines()), end="")
    return max(result.outcome for result in results)


def _audit(registry: Registry, args: argparse.Namespace) -> Exit:
    unknown = (set(args.rule) - registry.rules.keys()) | (set(args.check) - registry.checks.keys())
    if unknown:
        print(f"[rules] no such rule or check: {sorted(unknown)}", file=sys.stderr)
        return Exit.UNMEASURED
    report = Audit(registry, frozenset(args.rule), frozenset(args.check)).run()
    print(report)
    return report.outcome


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.rules", description=__doc__)
    parser.add_argument("--root", type=Path, default=_ROOT, help="repository root")
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="the rules")
    listing.add_argument("--verdict", action="store_true", help="and how far each is enforced")
    listing.set_defaults(handler=_list)
    show = commands.add_parser("show", help="one rule, why included")
    show.add_argument("id")
    show.set_defaults(handler=_show)
    run = commands.add_parser("run", help="the checks one level runs; exit is the worst")
    run.add_argument("--level", required=True, choices=list(Level))
    scope = run.add_mutually_exclusive_group()
    scope.add_argument("--range", help="A..B: only what those commits changed")
    scope.add_argument("--staged", action="store_true", help="only what the index changes")
    run.set_defaults(handler=_run)
    audit = commands.add_parser("audit", help="is each claim true, and can each check fail")
    audit.add_argument("--rule", action="append", default=[], help="only these rules' controls")
    audit.add_argument("--check", action="append", default=[], help="only these checks' controls")
    audit.set_defaults(handler=_audit)
    args = parser.parse_args(argv)
    try:
        registry = Registry.load(args.root.resolve())
    except RuleError as exc:
        print(f"[rules] {exc}", file=sys.stderr)
        return Exit.VIOLATION
    return int(args.handler(registry, args))


if __name__ == "__main__":
    sys.exit(main())
