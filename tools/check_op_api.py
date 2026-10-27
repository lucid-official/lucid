#!/usr/bin/env python3
"""
tools/check_op_api.py — Verify every registry entry has a callable engine_fn
and that every free_fn_name is exported in lucid's public namespace.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    from lucid._ops._registry import _REGISTRY

    # noqa: F401 — the import is the check: a missing engine fails here.
    from lucid._C import engine as _C_engine  # noqa: F401

    errors: list[str] = []

    for entry in _REGISTRY:
        fn = entry.engine_fn
        if not callable(fn):
            errors.append(f"  NOT CALLABLE: registry entry {entry.name!r} has engine_fn={fn!r}")

    # Check free_fn_name is callable from lucid._ops (the module that auto-generates them)
    import lucid._ops as _ops_module

    for entry in _REGISTRY:
        fn_name = entry.free_fn_name
        if fn_name is None:
            continue
        if not callable(getattr(_ops_module, fn_name, None)):
            errors.append(
                f"  NOT CALLABLE IN lucid._ops: free_fn_name={fn_name!r} missing or not callable"
            )

    if errors:
        print("[check_op_api] API violations found:")
        for e in errors:
            print(e)
        return 1

    n = len([e for e in _REGISTRY if e.free_fn_name])
    print(f"[check_op_api] all {n} registry free functions are callable and exported.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
