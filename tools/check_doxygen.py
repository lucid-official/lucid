#!/usr/bin/env python3
"""
tools/check_doxygen.py — Check that C++ public headers have doc comment coverage.

A header is considered "documented" if it contains at least one `//` doc comment
block (single-line or block comment) in the first 30 lines.

Usage:  python tools/check_doxygen.py [--threshold N]   (default: 70%)
"""

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
HEADERS_DIR = ROOT / "lucid/_C"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--threshold",
        type=float,
        default=70.0,
        help="Minimum documentation coverage percentage (default: 70)",
    )
    args = parser.parse_args()

    headers = [
        h for h in HEADERS_DIR.rglob("*.h") if "__pycache__" not in str(h) and "test/" not in str(h)
    ]

    documented = 0
    undocumented: list[str] = []

    for h in sorted(headers):
        lines = h.read_text(encoding="utf-8", errors="replace").splitlines()[:30]
        has_doc = any(
            line.strip().startswith("//") or line.strip().startswith("/*") for line in lines
        )
        if has_doc:
            documented += 1
        else:
            undocumented.append(str(h.relative_to(ROOT)))

    total = len(headers)
    if total == 0:
        print("[check_doxygen] No headers found.")
        return 0

    coverage = 100.0 * documented / total
    print(f"[check_doxygen] Documentation coverage: {documented}/{total} = {coverage:.1f}%")

    if coverage < args.threshold:
        print(f"[check_doxygen] BELOW THRESHOLD ({args.threshold}%). Undocumented headers:")
        for h in undocumented[:20]:
            print(f"  {h}")
        if len(undocumented) > 20:
            print(f"  ... and {len(undocumented) - 20} more")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
