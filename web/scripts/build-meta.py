"""build-meta.py — emit ``public/build-meta.json`` with version + commit info.

Runs at prebuild time (alongside the api-data cache wrapper) so the docs
site can surface "Built from <sha> on <date>, lucid <version>" freshness
metadata.  Cheap enough to run unconditionally — no caching needed.

Schema (consumed by ``src/lib/build-meta.ts``)::

    {
      "lucid_version":  "3.2.2",
      "git_sha":        "abc1234",
      "git_sha_full":   "abc1234567890abcdef...",
      "git_branch":     "main",
      "built_at":       "2026-05-22T12:34:56Z"
    }

Missing values fall back to ``null`` rather than crashing the build — the
UI handles a partial meta gracefully (renders "—" placeholders).
"""

import datetime as _dt
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEB_ROOT = HERE.parent
REPO_ROOT = WEB_ROOT.parent
OUT = WEB_ROOT / "public" / "build-meta.json"


def _git(*args: str) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return out.strip() or None


def _lucid_version() -> str | None:
    """Parse pyproject.toml for ``project.version``.  Plain regex — we
    don't want a tomllib dependency on older Pythons + the format is
    stable."""
    pyproject = REPO_ROOT / "pyproject.toml"
    try:
        text = pyproject.read_text()
    except OSError:
        return None
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return m.group(1) if m else None


def main() -> int:
    meta = {
        "lucid_version": _lucid_version(),
        "git_sha": _git("rev-parse", "--short", "HEAD"),
        "git_sha_full": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "built_at": _dt.datetime.now(_dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    sha = meta["git_sha"] or "??"
    ver = meta["lucid_version"] or "?"
    print(f"[build-meta] lucid {ver} @ {sha} ({meta['built_at']}) → {OUT.relative_to(WEB_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
