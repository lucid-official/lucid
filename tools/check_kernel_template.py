#!/usr/bin/env python3
"""
tools/check_kernel_template.py — Verify kernel/.cpp files each have a matching header.

Every .cpp in lucid/_C/kernel/ must have a corresponding .h file.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
KERNEL_DIR = ROOT / "lucid/_C/kernel"


def main() -> int:
    if not KERNEL_DIR.exists():
        print("[check_kernel_template] kernel/ directory not found — skipping")
        return 0

    issues: list[str] = []
    cpp_files = sorted(KERNEL_DIR.glob("*.cpp"))

    for cpp in cpp_files:
        header = cpp.with_suffix(".h")
        if not header.exists():
            issues.append(f"  MISSING HEADER: {cpp.relative_to(ROOT)} has no matching .h")

    if issues:
        print("[check_kernel_template] Kernel template violations:")
        for i in issues:
            print(i)
        return 1

    print(f"[check_kernel_template] {len(cpp_files)} kernel files all have matching headers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
