#!/usr/bin/env python3
"""
tools/check_audit_baselines.py — the audit's recorded baselines agree with themselves.

``lucid/test/audit/{coverage,doctest,suite}.json`` each carry a header total
next to the body it summarises (``cells`` next to ``answered``, ``failed``
next to ``failing``, ``line_covered`` / ``line_total`` / ``percent`` next to
``files``).  The audit's writers derive the totals from the body, so a file
they wrote is consistent by construction; what breaks it is an edit that
bypasses them.  LCD-358: a one-line hand deletion from ``answered`` left
``cells`` one high, and the repository-baseline test failed on main.

So this checks only that agreement — no audit run, a JSON parse per file —
and is cheap enough to run on every commit that stages one of the files
(``--staged``, the pre-commit gate) and on every land that changes one
(``tools/agent_ws.py land``).  The fix it asks for is to regenerate, never to
patch the header::

    python3 tools/check_audit_baselines.py              # working tree
    python3 tools/check_audit_baselines.py --staged     # the index (pre-commit)
    python3 tools/check_audit_baselines.py --root WT    # another checkout

Stdlib only: ``agent_ws.py`` runs it with whatever interpreter it has.
"""

import argparse
import json
import math
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TypeGuard

ROOT = Path(__file__).resolve().parent.parent

# The statuses ``lucid/test/audit/__main__._answered`` records.
VERDICTS = frozenset({"pass", "fail", "error", "vacuous"})


def _is_count(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _coverage(payload: dict[str, object]) -> list[str]:
    answered = payload.get("answered")
    cells = payload.get("cells")
    if not isinstance(answered, dict):
        return ["'answered' is not an object"]
    if not _is_count(cells):
        return [f"'cells' is not a count: {cells!r}"]
    problems = []
    if cells != len(answered):
        problems.append(f"'cells' is {cells} but 'answered' holds {len(answered)} entries")
    bad_keys = [k for k in answered if "::" not in k]
    if bad_keys:
        problems.append(f"{len(bad_keys)} key(s) not 'axis::symbol', e.g. {bad_keys[0]!r}")
    bad = [k for k, v in answered.items() if v not in VERDICTS]
    if bad:
        problems.append(
            f"{len(bad)} value(s) not a verdict {sorted(VERDICTS)}, "
            f"e.g. {bad[0]!r}: {answered[bad[0]]!r}"
        )
    return problems


def _doctest(payload: dict[str, object]) -> list[str]:
    failing = payload.get("failing")
    failed = payload.get("failed")
    attempted = payload.get("attempted")
    if not isinstance(failing, dict):
        return ["'failing' is not an object"]
    if not _is_count(failed) or not _is_count(attempted):
        return [f"'failed' / 'attempted' are not counts: {failed!r} / {attempted!r}"]
    if not all(_is_count(n) and n > 0 for n in failing.values()):
        return ["'failing' holds a value that is not a positive count"]
    problems = []
    total = sum(failing.values())
    if failed != total:
        problems.append(f"'failed' is {failed} but 'failing' sums to {total}")
    if failed > attempted:
        problems.append(f"'failed' ({failed}) exceeds 'attempted' ({attempted})")
    return problems


def _suite(payload: dict[str, object]) -> list[str]:
    files = payload.get("files")
    covered = payload.get("line_covered")
    total = payload.get("line_total")
    if not isinstance(files, dict):
        return ["'files' is not an object"]
    if not _is_count(covered) or not _is_count(total):
        return [f"'line_covered' / 'line_total' are not counts: {covered!r} / {total!r}"]
    rows: dict[str, tuple[int, int]] = {}
    for name, row in files.items():
        if not isinstance(row, dict):
            return [f"'files' entry {name!r} is not an object"]
        file_covered, file_total = row.get("covered"), row.get("total")
        if not _is_count(file_covered) or not _is_count(file_total):
            return [f"'files' entry {name!r} lacks count 'covered' and 'total'"]
        rows[str(name)] = (file_covered, file_total)
    problems = []
    sum_covered = sum(c for c, _ in rows.values())
    sum_total = sum(t for _, t in rows.values())
    if covered != sum_covered:
        problems.append(f"'line_covered' is {covered} but 'files' sums to {sum_covered}")
    if total != sum_total:
        problems.append(f"'line_total' is {total} but 'files' sums to {sum_total}")
    over = [name for name, (c, t) in rows.items() if c > t]
    if over:
        problems.append(f"{len(over)} file(s) covering more lines than they have, e.g. {over[0]}")
    percent = payload.get("percent")
    expected = 100.0 * covered / total if total else None
    if expected is None:
        if percent is not None:
            problems.append(f"'percent' is {percent!r} over zero statements")
    elif not isinstance(percent, (int, float)) or not math.isclose(percent, expected, rel_tol=1e-9):
        problems.append(f"'percent' is {percent!r} but the counts give {expected!r}")
    return problems


Check = Callable[[dict[str, object]], list[str]]

# path -> (consistency check, the command that rewrites the file correctly)
BASELINES: dict[str, tuple[Check, str]] = {
    "lucid/test/audit/coverage.json": (_coverage, "lucid-audit --audit-only --update-coverage"),
    "lucid/test/audit/doctest.json": (_doctest, "lucid-audit --doctests-only --update-doctests"),
    "lucid/test/audit/suite.json": (_suite, "lucid-audit --tests-only --update-suite"),
}


def problems(rel: str, text: str) -> list[str]:
    """Every way ``text``, the contents of baseline ``rel``, disagrees with itself."""
    check, _ = BASELINES[rel]
    try:
        payload = json.loads(text)
    except ValueError as exc:
        return [f"not JSON: {exc}"]
    if not isinstance(payload, dict):
        return ["the top level is not an object"]
    return check(payload)


def _git(root: Path, *args: str) -> str:
    # The hook's environment on purpose: GIT_INDEX_FILE is how git hands a
    # pre-commit hook the index being committed (a temporary one for
    # `git commit -- <path>`), and that is the content to judge.
    return subprocess.run(
        ["git", *args], cwd=root, env=dict(os.environ), capture_output=True, text=True, check=True
    ).stdout


def staged(root: Path) -> dict[str, str]:
    """The baselines this commit adds or modifies, as staged."""
    names = _git(root, "diff", "--cached", "--name-only", "--diff-filter=ACMR").split()
    return {rel: _git(root, "show", f":{rel}") for rel in names if rel in BASELINES}


def on_disk(root: Path) -> dict[str, str]:
    return {
        rel: (root / rel).read_text(encoding="utf-8") for rel in BASELINES if (root / rel).exists()
    }


def report(contents: dict[str, str]) -> int:
    """Print what disagrees; 1 if anything does."""
    bad = 0
    for rel, text in sorted(contents.items()):
        found = problems(rel, text)
        if not found:
            continue
        bad += 1
        print(f"[check_audit_baselines] {rel} is inconsistent:", file=sys.stderr)
        for line in found:
            print(f"    · {line}", file=sys.stderr)
        print(
            f"    Regenerate it with `{BASELINES[rel][1]}` "
            "rather than editing it by hand (lucid/test/audit/README.md).",
            file=sys.stderr,
        )
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT, help="checkout to check")
    parser.add_argument(
        "--staged", action="store_true", help="check the index instead of the working tree"
    )
    args = parser.parse_args(argv)
    return report(staged(args.root) if args.staged else on_disk(args.root))


if __name__ == "__main__":
    sys.exit(main())
