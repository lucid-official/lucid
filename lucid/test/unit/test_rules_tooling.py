"""Expose ``tools/test_rules.py`` under ``lucid/test/`` so default pytest CI collects it (LCD-841)."""

import importlib.util
from pathlib import Path

_TEST_FILE = Path(__file__).resolve().parents[3] / "tools" / "test_rules.py"
_SPEC = importlib.util.spec_from_file_location("_tools_test_rules", _TEST_FILE)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)

for _name in dir(_MOD):
    if _name in ("scratch", "real_copy") or _name.startswith("test_"):
        globals()[_name] = getattr(_MOD, _name)
