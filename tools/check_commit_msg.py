#!/usr/bin/env python3
"""Validate commit message(s) against the Lucid commit convention.

The convention (see CONTRIBUTING.md (Commit style)):

    <type>(<scope>): <subject>

    [body]

    [trailers]

* ``type``    — one of the allowed set below (lower-case).
* ``scope``   — optional; a lower-case dotted source path mirroring the tree,
                e.g. ``models.text.bert`` / ``nn.functional`` / ``compile``.
                Multiple comma-separated scopes are allowed.
* ``!``       — optional breaking-change marker (``type(scope)!: ...``).
* ``subject`` — non-empty, no trailing period.  Header ≤ 72 chars recommended,
                ≤ 100 enforced.

Usage
-----
    # Single message file (commit-msg hook):
    python3 tools/check_commit_msg.py --file .git/COMMIT_EDITMSG

    # A revision range (CI):
    python3 tools/check_commit_msg.py --range origin/main..HEAD

    # Default: validate HEAD's subject.
    python3 tools/check_commit_msg.py

Exit code 0 = all valid, 1 = at least one hard error.  Warnings never fail.

Pure standard library — no Lucid import, no engine, no third-party deps; runs
anywhere (local hook + CI).
"""

import argparse
import re
import subprocess
import sys

# ── Convention vocabulary ─────────────────────────────────────────────────────
# Code / user-facing types feed the CHANGELOG pipeline (tools/changelog.py).
USER_FACING_TYPES = ("feat", "fix", "perf", "refactor", "revert", "remove", "deprecate", "security")
# Non-code types — no CHANGELOG entry expected.
OTHER_TYPES = ("docs", "style", "test", "build", "ci", "chore", "release")
ALLOWED_TYPES = USER_FACING_TYPES + OTHER_TYPES

# <type>(<scope>)!: <subject>
_HEADER_RE = re.compile(
    r"^(?P<type>[a-z]+)" r"(?:\((?P<scope>[a-z0-9._,\-]+)\))?" r"(?P<bang>!)?" r": (?P<subject>.+)$"
)

# git-generated / system subjects that the convention does not govern.
_SKIP_RE = re.compile(r'^(Merge |merge |fixup! |squash! |Revert ")')

RECOMMENDED_LEN = 72
MAX_LEN = 100


def validate_subject(subject: str) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for one commit subject line.

    A non-empty ``errors`` list means the commit is rejected.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not subject.strip():
        errors.append("empty commit subject")
        return errors, warnings

    if _SKIP_RE.match(subject):
        return errors, warnings  # merges / reverts / fixups are exempt

    m = _HEADER_RE.match(subject)
    if m is None:
        errors.append(
            "header must match '<type>(<scope>): <subject>' "
            "(e.g. 'feat(models.text.bert): add SQuAD weights')"
        )
        return errors, warnings

    ctype = m.group("type")
    if ctype not in ALLOWED_TYPES:
        errors.append(f"unknown type {ctype!r}. Allowed: {', '.join(ALLOWED_TYPES)}")

    body_subject = m.group("subject").strip()
    if not body_subject:
        errors.append("subject text after the colon is empty")
    if body_subject.endswith("."):
        errors.append("subject must not end with a period")

    if len(subject) > MAX_LEN:
        errors.append(f"header is {len(subject)} chars (hard limit {MAX_LEN})")
    elif len(subject) > RECOMMENDED_LEN:
        warnings.append(f"header is {len(subject)} chars (recommended ≤ {RECOMMENDED_LEN})")

    return errors, warnings


def _first_subject_from_file(path: str) -> str:
    """First non-blank, non-comment line of a commit message file."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            stripped = line.rstrip("\n")
            if stripped.strip() == "" or stripped.lstrip().startswith("#"):
                continue
            return stripped
    return ""


def _subjects_from_range(rev_range: str) -> list[str]:
    """Commit subjects in ``rev_range`` (merges excluded), newest first."""
    out = subprocess.run(
        ["git", "log", "--no-merges", "--format=%s", rev_range],
        capture_output=True,
        text=True,
        check=True,
    )
    return [ln for ln in out.stdout.splitlines() if ln.strip()]


def _report(subject: str, errors: list[str], warnings: list[str]) -> None:
    for w in warnings:
        print(f"  ⚠️  {subject!r}\n      {w}", file=sys.stderr)
    if errors:
        print(f"  ❌  {subject!r}", file=sys.stderr)
        for e in errors:
            print(f"      {e}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check_commit_msg",
        description="Validate commit message(s) against the Lucid convention.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--file", help="commit message file (commit-msg hook)")
    group.add_argument("--range", dest="rev_range", help="git revision range (CI)")
    args = parser.parse_args(argv)

    if args.file is not None:
        subjects = [_first_subject_from_file(args.file)]
    elif args.rev_range is not None:
        try:
            subjects = _subjects_from_range(args.rev_range)
        except subprocess.CalledProcessError as exc:
            print(
                f"check_commit_msg: cannot read range {args.rev_range!r}: " f"{exc.stderr.strip()}",
                file=sys.stderr,
            )
            return 1
        if not subjects:
            print("check_commit_msg: no commits in range (nothing to validate).")
            return 0
    else:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%s"], capture_output=True, text=True, check=True
        )
        subjects = [out.stdout.strip()]

    failed = 0
    for subject in subjects:
        errors, warnings = validate_subject(subject)
        if errors or warnings:
            _report(subject, errors, warnings)
        if errors:
            failed += 1

    if failed:
        print(
            f"\n  {failed} commit(s) violate the Lucid commit convention "
            "(CONTRIBUTING.md (Commit style)).\n"
            "  Format: <type>(<scope>): <subject>\n"
            f"  Types:  {', '.join(ALLOWED_TYPES)}\n"
            "  Bypass a single local commit (discouraged): "
            "LUCID_SKIP_COMMIT_CONVENTION=1 git commit ...  (or --no-verify)",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
