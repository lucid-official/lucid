#!/usr/bin/env bash
# tools/install_hooks.sh — one-shot activation of the Lucid git hooks.
#
# Run once after cloning the repo (or after a fresh ``git init``):
#
#     ./tools/install_hooks.sh
#
# What it does:
#
#   1. Points ``core.hooksPath`` at ``.githooks/``, so the four Lucid
#      hooks (``pre-commit`` / ``commit-msg`` / ``post-commit`` /
#      ``pre-push``) actually fire.  By default git only looks in
#      ``.git/hooks/``, which means the hooks committed into the repo
#      are inert until this line runs.
#
#   2. Says whether the interpreter the hooks need is there: Python 3.14
#      at ``.venv/bin/python3``.  Without it they refuse every commit and
#      every push.  ``CONTRIBUTING.md`` sets it up.
#
#   3. Verifies the optional dependency for ``pre-commit`` — ``cloc``
#      (used to refresh the Lines-of-Code badge in README.md).  Without
#      ``cloc`` the badge update silently no-ops; the hook still
#      succeeds.  Print a brief ``brew install cloc`` hint if missing.
#
#   4. (Optional, ``--check``) Re-runs the hooks once over the current
#      tree to surface any drift — useful right after install.
#
# Idempotent: safe to run multiple times.  Re-running just re-asserts
# the config and re-checks.
#
# This script does NOT modify any tracked file on its own; it only
# changes the per-clone git config.  The actual hook scripts live under
# ``.githooks/`` and are version-controlled.

set -euo pipefail

cd "$(dirname "$0")/.."

# ── 1. Activate the hooks directory ─────────────────────────────────────────
git config core.hooksPath .githooks
echo "✓ git hooks active — core.hooksPath = .githooks"

# Show what each hook does (one-line summary).
cat <<EOF

  Installed hooks (see .githooks/README.md for details):
    pre-commit   — the rules at commit, then the ![Lines of Code] badge via cloc
    commit-msg   — the commit convention; warn when conventional-commit subjects
                   (feat/fix/perf/…) land without a CHANGELOG.md entry
    post-commit  — record a bypass; auto-fold a CHANGELOG.md entry via
                   tools/changelog.py when the just-landed commit matches the
                   conventional prefix and didn't touch CHANGELOG itself
    pre-push     — judge again what leaves for main or a v* tag
EOF

# ── 2. Check the interpreter the hooks pin ──────────────────────────────────
if .venv/bin/python3 -c 'import sys; sys.exit(sys.version_info < (3, 14))' 2>/dev/null; then
    echo ""
    echo "✓ Python 3.14 at .venv/bin/python3"
else
    cat <<EOF

  ⚠️  no Python 3.14 at .venv/bin/python3 — the hooks refuse every commit and
     push until there is one (CONTRIBUTING.md sets it up).

EOF
fi

# ── 3. Check optional `cloc` ────────────────────────────────────────────────
if command -v cloc &>/dev/null; then
    CLOC_VERSION="$(cloc --version 2>/dev/null | head -n1)"
    echo ""
    echo "✓ cloc $CLOC_VERSION available (LOC badge will refresh on each commit)"
else
    cat <<EOF

  ⚠️  cloc not installed — the pre-commit hook will skip LOC badge
     refresh (the badge stays at its last hand-edited value).  To
     enable automatic refresh:

         brew install cloc

EOF
fi

# ── 4. Optional --check: dry-run the hooks ──────────────────────────────────
if [[ "${1:-}" == "--check" ]]; then
    echo ""
    echo "==> --check: running pre-commit hook against current tree"
    .githooks/pre-commit
    echo "✓ pre-commit dry-run OK"
fi

echo ""
echo "Done.  Next ``git commit`` will exercise the hook chain."
