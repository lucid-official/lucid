#!/usr/bin/env python3
"""
tools/changelog.py — CHANGELOG.md helper.

Sub-commands:
  add <category> <message>   Add a bullet to the [Unreleased] section.
  propose                    Read the most recent commit and suggest an
                             entry for [Unreleased] (interactive confirm).
  release <version>          Promote [Unreleased] to a dated [<version>]
                             section. Updates compare links at the bottom.
  check [--strict]           Verify CHANGELOG is consistent with git state.
                             --strict fails if [Unreleased] is empty when
                             new feat/fix/perf commits exist since the last
                             tag/release section.

Categories (Keep a Changelog + project-specific):
  added | changed | deprecated | removed | fixed | security
  performance | tooling | documentation

Examples:
  python tools/changelog.py add fixed "Cholesky upper=True backward Triu fix"
  python tools/changelog.py propose
  python tools/changelog.py release 3.0.1
  python tools/changelog.py check --strict
"""

import argparse
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

# ── paths / constants ────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
CHANGELOG = ROOT / "CHANGELOG.md"

CATEGORIES: dict[str, str] = {
    "added": "Added",
    "changed": "Changed",
    "deprecated": "Deprecated",
    "removed": "Removed",
    "fixed": "Fixed",
    "security": "Security",
    "performance": "Performance",
    "tooling": "Tooling",
    "documentation": "Documentation",
}

# Conventional-commit prefix → CHANGELOG category.
PREFIX_MAP: dict[str, str] = {
    "feat": "added",
    "fix": "fixed",
    "perf": "performance",
    "refactor": "changed",
    "revert": "changed",
    "remove": "removed",
    "deprecate": "deprecated",
    "security": "security",
    "docs": "documentation",
    "test": "tooling",
    "chore": "tooling",
    "build": "tooling",
    "ci": "tooling",
    "style": "tooling",
}

# Commit-prefix patterns we DO require a CHANGELOG entry for in --strict mode.
USER_FACING_PREFIXES = {
    "feat",
    "fix",
    "perf",
    "refactor",
    "revert",
    "remove",
    "deprecate",
    "security",
}

# `check` only looks at commits since the most recent version section.
RELEASE_HEADER_RE = re.compile(r"^## \[(\d+\.\d+\.\d+)\]")


# ── file helpers ─────────────────────────────────────────────────────────────


def _read() -> str:
    if not CHANGELOG.exists():
        sys.exit(f"❌  {CHANGELOG.relative_to(ROOT)} does not exist")
    return CHANGELOG.read_text(encoding="utf-8")


def _write(text: str) -> None:
    CHANGELOG.write_text(text, encoding="utf-8")


def _git(*args: str) -> str:
    """Run git in the repo root and return stdout (rstripped)."""
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        sys.exit(f"❌  git {' '.join(args)} failed:\n{result.stderr}")
    return result.stdout.rstrip()


def _commit_prefix(subject: str) -> str | None:
    """Return the conventional-commit prefix (e.g. 'feat', 'fix') or None."""
    m = re.match(r"^([a-z]+)(?:\([^)]*\))?(!)?:\s", subject)
    if m is None:
        return None
    return m.group(1)


def _category_for_prefix(prefix: str) -> str:
    return PREFIX_MAP.get(prefix, "changed")


# ── add ──────────────────────────────────────────────────────────────────────


def _ensure_unreleased(text: str) -> tuple[str, int, int]:
    """Locate the [Unreleased] section and return (text, body_start, body_end).

    body_end points to the line index where the section's body ENDS — i.e.
    one past the last bullet, not including any ``---`` separator that may
    follow.  If a ``---`` separator precedes the next ``## `` header, body_end
    stops just before the blank line preceding the ``---``.

    If [Unreleased] is missing, raise.
    """
    lines = text.splitlines(keepends=True)
    h_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "## [Unreleased]":
            h_idx = i
            break
    if h_idx is None:
        sys.exit("❌  CHANGELOG missing '## [Unreleased]' header")

    # Find the next ## header after Unreleased.
    next_header = len(lines)
    for j in range(h_idx + 1, len(lines)):
        if lines[j].startswith("## "):
            next_header = j
            break

    # Walk back from next_header, skipping trailing blank lines and a single
    # ``---`` separator with its surrounding blanks.  This makes the body
    # boundary insensitive to the separator so insertions don't displace it.
    e_idx = next_header
    while e_idx > h_idx + 1 and lines[e_idx - 1].strip() == "":
        e_idx -= 1
    if e_idx > h_idx + 1 and lines[e_idx - 1].strip() == "---":
        e_idx -= 1
        while e_idx > h_idx + 1 and lines[e_idx - 1].strip() == "":
            e_idx -= 1
    return "".join(lines), h_idx, e_idx


def _add_bullet(category: str, message: str, dry_run: bool = False) -> None:
    if category not in CATEGORIES:
        sys.exit(f"❌  unknown category {category!r} (allowed: " f"{', '.join(CATEGORIES)})")
    text = _read()
    text, h_idx, e_idx = _ensure_unreleased(text)
    lines = text.splitlines(keepends=True)
    # Body of the Unreleased section, EXCLUDING the trailing separator/blanks.
    body_lines = lines[h_idx + 1 : e_idx]
    body = "".join(body_lines)

    cat_label = CATEGORIES[category]
    cat_header = f"### {cat_label}"
    bullet = f"- {message.strip()}\n"

    # Strip any "[Unreleased] is empty" placeholder line and the surrounding
    # blank lines that frame it.  Historically we've used two phrasings:
    #
    #     - _Pending the next release._    (old bullet form)
    #     _No changes yet._                (current italics-only form)
    #
    # Both are recognised here; new code paths in this script standardise on
    # the second form (no leading dash) so the placeholder isn't valid
    # markdown for a category bullet — it can't accidentally be mistaken
    # for a real entry.
    placeholder_patterns = ("- _Pending the next release._", "_No changes yet._")

    def _is_placeholder(stripped: str) -> bool:
        return stripped in placeholder_patterns

    if any(_is_placeholder(line.strip()) for line in body.splitlines()):
        # Drop the placeholder line.  Collapse the runs of blank lines that
        # surrounded it so we don't leave a double-blank where the entry will
        # land.
        kept: list[str] = []
        prev_blank = False
        for line in body.splitlines(keepends=True):
            if _is_placeholder(line.strip()):
                continue
            is_blank = line.strip() == ""
            if is_blank and prev_blank:
                continue  # collapse consecutive blanks
            kept.append(line)
            prev_blank = is_blank
        # Strip a leading blank so the body starts cleanly under the
        # ## [Unreleased] header.
        while kept and kept[0].strip() == "":
            kept.pop(0)
        body = "".join(kept)

    if cat_header in body:
        # Append bullet under the existing category, before next ### or end.
        sec_lines = body.splitlines(keepends=True)
        new_lines = []
        in_target = False
        appended = False
        for k, line in enumerate(sec_lines):
            stripped = line.rstrip("\n")
            if stripped == cat_header:
                in_target = True
                new_lines.append(line)
                continue
            if in_target and (line.startswith("### ") or line.startswith("## ")):
                # Boundary into next sub-section — emit bullet + blank first.
                if not appended:
                    # Strip trailing blank so we don't double up.
                    while new_lines and new_lines[-1].strip() == "":
                        new_lines.pop()
                    new_lines.append("\n")  # blank between header and bullet
                    new_lines.append(bullet)
                    new_lines.append("\n")  # blank between bullet and next ###
                    appended = True
                in_target = False
            new_lines.append(line)
        if in_target and not appended:
            new_lines.append(bullet)
        new_body = "".join(new_lines)
    else:
        # Append a new category block at the end of the body.
        # Normalize trailing whitespace then add blank line + header + bullet.
        new_body = body.rstrip("\n") + "\n\n" + cat_header + "\n\n" + bullet

    # Re-emit canonical separator after the section body so the next ##
    # version section is preceded by exactly one blank + `---` + blank.
    sep = "\n---\n\n"
    new_section = lines[h_idx] + new_body.rstrip("\n") + "\n" + sep

    new_text = "".join(lines[:h_idx]) + new_section + "".join(lines[e_idx:]).lstrip("\n")
    # Ensure there is no leftover stray separator immediately after our newly
    # emitted one (can happen if the original file already had its own).
    new_text = re.sub(r"\n---\n\n(?:\n*---\n\n)+", "\n---\n\n", new_text)

    if dry_run:
        print(new_text)
        return
    _write(new_text)
    print(f"  ✅  Added to [Unreleased] / {cat_label}: {message}")


# ── propose ──────────────────────────────────────────────────────────────────


def _propose(*, auto: bool = False, message_file: str | None = None) -> None:
    """Propose an [Unreleased] entry from a commit message.

    Args:
        auto:         If True, skip the interactive confirmation and add the
                      entry directly.  Used by the ``prepare-commit-msg``
                      Git hook to keep CHANGELOG.md in sync automatically.
        message_file: Path to a file containing the commit message to parse
                      (used by ``prepare-commit-msg`` since HEAD doesn't yet
                      reflect the commit being created).  Falls back to
                      ``git log -1`` when ``None``.
    """
    if message_file is not None:
        with open(message_file, encoding="utf-8") as f:
            full = f.read()
        # Strip git-comment lines (starting with '#').
        lines = [ln for ln in full.splitlines() if not ln.startswith("#")]
        subject = lines[0] if lines else ""
        body = "\n".join(lines[1:]).strip()
    else:
        subject = _git("log", "-1", "--format=%s")
        body = _git("log", "-1", "--format=%b")
    prefix = _commit_prefix(subject)
    if prefix is None:
        if not auto:
            print(f"  ⚠️   Commit subject has no conventional prefix: {subject!r}")
            print("       Skip — use `add <category> <message>` manually.")
        return
    category = _category_for_prefix(prefix)
    # Strip the conventional prefix from the message.
    message = re.sub(r"^[a-z]+(?:\([^)]*\))?!?:\s*", "", subject)

    if auto:
        # Quietly add and return — no prompt.
        _add_bullet(category, message)
        return

    print(f"  HEAD: {subject}")
    print(f"  → category: {category}  ({CATEGORIES[category]})")
    print(f"  → message : {message}")
    if body:
        print(f"  body preview:\n    {body.splitlines()[0][:120]}")
    try:
        reply = input("\nAdd this entry? [Y/n/edit] ").strip().lower()
    except EOFError:
        reply = "n"
    if reply == "n":
        print("  cancelled.")
        return
    if reply == "edit":
        new_msg = input("New message: ").strip() or message
        new_cat = input(f"Category [{category}]: ").strip() or category
        _add_bullet(new_cat, new_msg)
    else:
        _add_bullet(category, message)


# ── release ──────────────────────────────────────────────────────────────────


def _release(version: str) -> None:
    if not re.match(r"^\d+\.\d+\.\d+(?:-[\w.]+)?$", version):
        sys.exit(f"❌  '{version}' is not a valid semver string")

    text = _read()
    text, h_idx, e_idx = _ensure_unreleased(text)
    lines = text.splitlines(keepends=True)

    # Body of the Unreleased section (excluding its header).
    body_lines = lines[h_idx + 1 : e_idx]
    body = "".join(body_lines).strip()

    # Replace [Unreleased] with [version] - date, prepend new empty Unreleased.
    today = date.today().isoformat()
    new_unreleased = (
        "## [Unreleased]\n" "\n" "### Added\n" "- _Pending the next release._\n" "\n" "---\n" "\n"
    )
    new_release = f"## [{version}] — {today}\n\n{body}\n"

    new_text = "".join(lines[:h_idx]) + new_unreleased + new_release + "\n" + "".join(lines[e_idx:])

    # Update compare links at the bottom of the file.
    new_text = _update_compare_links(new_text, version)

    _write(new_text)
    print(f"  ✅  Released {version} ({today})")


def _update_compare_links(text: str, version: str) -> str:
    """Update or insert the [Unreleased] / [version] compare URLs at the bottom."""
    # Try to find the existing [Unreleased] link to extract repo URL.
    m = re.search(r"^\[Unreleased\]:\s*(https?://[^\s/]+/[^/]+/[^/]+)/compare/", text, re.MULTILINE)
    if not m:
        # Could not infer repo URL; skip silently.
        return text
    repo = m.group(1)

    # Build new tail.
    new_unreleased_link = f"[Unreleased]: {repo}/compare/v{version}...HEAD"
    new_version_link = f"[{version}]: {repo}/releases/tag/v{version}"

    # Replace the old [Unreleased]: line with both new lines.
    text = re.sub(
        r"^\[Unreleased\]:\s*.*$",
        new_unreleased_link + "\n" + new_version_link,
        text,
        count=1,
        flags=re.MULTILINE,
    )
    return text


# ── check ────────────────────────────────────────────────────────────────────


def _last_release_section(text: str) -> str | None:
    """Return the most recent semver header found, or None."""
    for line in text.splitlines():
        m = RELEASE_HEADER_RE.match(line)
        if m:
            return m.group(1)
    return None


def _check(strict: bool) -> None:
    text = _read()
    text2, h_idx, e_idx = _ensure_unreleased(text)
    lines = text2.splitlines(keepends=True)
    body = "".join(lines[h_idx + 1 : e_idx])
    has_real_entry = "_Pending the next release._" not in body and bool(
        re.search(r"^\s*-\s+\S", body, re.MULTILINE)
    )

    last_version = _last_release_section(text)
    if last_version is None:
        print("  no release sections yet — skipping `check`.")
        return

    tag = f"v{last_version}"
    # Use the tag if it exists, otherwise fall back to the merge-base of the
    # release-section line (best-effort: don't fail if tag is missing).
    have_tag = (
        subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", tag], cwd=ROOT, capture_output=True
        ).returncode
        == 0
    )
    rev_range = f"{tag}..HEAD" if have_tag else "HEAD~50..HEAD"

    log = _git("log", "--format=%s", rev_range)
    new_user_facing = []
    for subject in log.splitlines():
        prefix = _commit_prefix(subject)
        if prefix in USER_FACING_PREFIXES:
            new_user_facing.append(subject)

    if new_user_facing and not has_real_entry:
        msg = (
            f"⚠️   {len(new_user_facing)} user-facing commit(s) since "
            f"v{last_version} but [Unreleased] is empty:\n  " + "\n  ".join(new_user_facing[:10])
        )
        if strict:
            sys.exit("❌  " + msg)
        print(msg)
        return

    print(
        f"  ✅  CHANGELOG OK "
        f"({len(new_user_facing)} user-facing commit(s) since v{last_version}, "
        f"[Unreleased] {'has entries' if has_real_entry else 'is empty'})"
    )


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(
        description="CHANGELOG.md helper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("add", help="add a bullet to [Unreleased]")
    p_add.add_argument("category", choices=list(CATEGORIES))
    p_add.add_argument("message", nargs="+", help="bullet text (multiple words OK)")

    p_prop = sub.add_parser("propose", help="propose an [Unreleased] entry from the latest commit")
    p_prop.add_argument(
        "--auto",
        action="store_true",
        help="skip the confirmation prompt and add the entry unconditionally "
        "(used by the prepare-commit-msg hook)",
    )
    p_prop.add_argument(
        "--message-file",
        default=None,
        help="parse this file as the commit message instead of reading from "
        "git log (used by prepare-commit-msg, which fires before HEAD updates)",
    )

    p_rel = sub.add_parser("release", help="promote [Unreleased] to a dated version section")
    p_rel.add_argument("version", help="semver string, e.g. 3.0.1")

    p_chk = sub.add_parser("check", help="verify CHANGELOG is consistent with git history")
    p_chk.add_argument(
        "--strict", action="store_true", help="fail (exit 1) if user-facing commits are missing"
    )

    args = p.parse_args()

    if args.cmd == "add":
        _add_bullet(args.category, " ".join(args.message))
    elif args.cmd == "propose":
        _propose(auto=args.auto, message_file=args.message_file)
    elif args.cmd == "release":
        _release(args.version)
    elif args.cmd == "check":
        _check(args.strict)


if __name__ == "__main__":
    main()
