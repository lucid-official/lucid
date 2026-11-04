"""python -m tools.rules: list the rules, show one, run a level, judge a push, audit the claims."""

import argparse
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Final

from tools.rules.audit import Audit
from tools.rules.checks import CannotMeasure, Changes, staged_changes
from tools.rules.model import Exit, Level, Rule, RuleError, Verdict
from tools.rules.push import PushGate, PushRange, run_level
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
    changes: Changes | None = None
    if args.range:
        changes = PushRange.parse(args.range).changes(registry.workspace)
    elif args.staged:
        changes = staged_changes(registry.workspace)
    return run_level(registry, Level(args.level), changes)


def _push(registry: Registry, args: argparse.Namespace) -> Exit:
    gate = PushGate(registry)
    if args.range:
        return gate.judge([PushRange.parse(args.range)])
    if sys.stdin.isatty():
        raise CannotMeasure("no ref updates: git writes them to the hook, or give --range A..B")
    return gate.judge(gate.ranges(sys.stdin.read(), args.remote))


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
    push = commands.add_parser("push", help="what pre-push runs: refs on stdin, git's arguments")
    push.add_argument("remote", nargs="?", default="origin")
    push.add_argument("url", nargs="?", help="what git passes; the refs are what is judged")
    push.add_argument("--range", help="A..B: judge these, as if they were being pushed")
    push.set_defaults(handler=_push)
    audit = commands.add_parser("audit", help="is each claim true, and can each check fail")
    audit.add_argument("--rule", action="append", default=[], help="only these rules' controls")
    audit.add_argument("--check", action="append", default=[], help="only these checks' controls")
    audit.set_defaults(handler=_audit)
    args = parser.parse_args(argv)
    try:
        return int(args.handler(Registry.load(args.root.resolve()), args))
    except RuleError as exc:
        print(f"[rules] {exc}", file=sys.stderr)
        return Exit.VIOLATION
    except CannotMeasure as exc:
        print(f"[rules] cannot measure: {exc}", file=sys.stderr)
        return Exit.UNMEASURED


if __name__ == "__main__":
    sys.exit(main())
