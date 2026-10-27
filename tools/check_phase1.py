#!/usr/bin/env python3
"""
tools/check_phase1.py — Verify that the Phase 1 foundation types are accessible
and have the required attributes.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

CHECKS = [
    # (import_path, attr_list)
    ("lucid._C.engine", ["TensorImpl", "Dtype", "Device", "ABI_VERSION", "engine_backward"]),
    ("lucid._dispatch", ["_unwrap", "_wrap", "_impl_with_grad"]),
    ("lucid._tensor.tensor", ["Tensor"]),
    ("lucid._ops._registry", ["_REGISTRY", "OpEntry"]),
    ("lucid._factories.creation", ["zeros", "ones", "eye", "arange"]),
    ("lucid._factories.random", ["rand", "randn", "randint"]),
    ("lucid.autograd._backward", []),  # importable
    ("lucid.nn.modules.linear", ["Linear"]),
    ("lucid.optim.sgd", ["SGD"]),
]


def main() -> int:
    errors: list[str] = []

    for mod_path, attrs in CHECKS:
        try:
            import importlib

            mod = importlib.import_module(mod_path)
        except ImportError as exc:
            errors.append(f"  IMPORT FAILED: {mod_path} — {exc}")
            continue
        for attr in attrs:
            if not hasattr(mod, attr):
                errors.append(f"  MISSING ATTR: {mod_path}.{attr}")

    if errors:
        print("[check_phase1] Phase 1 foundation check FAILED:")
        for e in errors:
            print(e)
        return 1

    print(f"[check_phase1] all {len(CHECKS)} modules importable with required attributes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
