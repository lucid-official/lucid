#!/usr/bin/env python3
"""
tools/check_op_template.py — Verify BackwardNode schema_v1 conformance.

Every .h file in ops/ufunc/ and ops/bfunc/ that defines a BackwardNode should:
  - Have a `static constexpr const char* schema_v1` member
  - Define a `backward(...)` method
"""

import sys
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
OP_DIRS = [ROOT / "lucid/_C/ops/ufunc", ROOT / "lucid/_C/ops/bfunc"]

SCHEMA_PATTERN = re.compile(r"schema_v1\s*=")
BACKWARD_PATTERN = re.compile(r"\bbackward\s*\(")
BACKWARD_NODE_PATTERN = re.compile(r"struct\s+\w+Backward\s*[:{]")


def main() -> int:
    issues: list[str] = []

    for op_dir in OP_DIRS:
        for h in sorted(op_dir.glob("*.h")):
            if h.name.startswith("_"):
                continue
            text = h.read_text(encoding="utf-8", errors="replace")

            has_backward_node = bool(BACKWARD_NODE_PATTERN.search(text))
            if not has_backward_node:
                continue  # not an op with backward — skip

            if not SCHEMA_PATTERN.search(text):
                issues.append(f"  MISSING schema_v1: {h.relative_to(ROOT)}")
            if not BACKWARD_PATTERN.search(text):
                issues.append(f"  MISSING backward(): {h.relative_to(ROOT)}")

    if issues:
        print("[check_op_template] BackwardNode template violations:")
        for i in issues:
            print(i)
        return 1

    checked = sum(
        1
        for d in OP_DIRS
        for h in d.glob("*.h")
        if not h.name.startswith("_")
        and BACKWARD_NODE_PATTERN.search(h.read_text(encoding="utf-8", errors="replace"))
    )
    print(f"[check_op_template] {checked} BackwardNode headers all conform.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
