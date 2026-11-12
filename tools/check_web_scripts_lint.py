#!/usr/bin/env python3
"""``ruff check`` on the Python under ``web/scripts``, at commit time (LCD-976).

The ratchet gate counts ``lucid/`` only, and no commit-time stage ran ruff, so the
docs builder piled up a dozen lint errors that nothing flagged until the Docs
workflow's lint step (LCD-970) -- after the push.  This is that step's check
(``ruff check`` under the project's ``[tool.ruff]``) on what a commit changes.

The ruff is the distribution installed with this interpreter, the one the gate
counts with (``tools.quality_gate.collectors.lint.ruff_bin``), and not
``python -m ruff``: from a worktree's light venv that resolves to whatever ``ruff``
is first on PATH, a different version than the pinned one.

Usage:  python tools/check_web_scripts_lint.py [PATH ...]    (default: web/scripts)
Exit:   0 clean, 1 findings, 2 cannot measure (the tools.rules convention).
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.quality_gate.collectors.lint import ruff_bin
from tools.quality_gate.core import GateError

_DEFAULT = "web/scripts"


def main(paths: list[str]) -> int:
    try:
        ruff = ruff_bin()
    except GateError as exc:
        print(f"check_web_scripts_lint: cannot measure: {exc}", file=sys.stderr)
        return 2
    # --force-exclude: a path named here obeys [tool.ruff] extend-exclude like the
    # directory the Docs workflow lints does.
    command = [ruff, "check", "--no-cache", "--force-exclude", *(paths or [_DEFAULT])]
    done = subprocess.run(command, cwd=ROOT, check=False)
    return done.returncode if done.returncode in (0, 1) else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
