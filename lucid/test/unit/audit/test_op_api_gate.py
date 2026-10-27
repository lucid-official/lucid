"""The export gate must reject names that exist but cannot be called."""

import pytest

import lucid._ops as _ops_module
from lucid._ops._registry import _REGISTRY
from tools.check_op_api import main


def test_present_but_noncallable_export_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    name = next(entry.free_fn_name for entry in _REGISTRY if entry.free_fn_name)
    monkeypatch.setattr(_ops_module, name, object())
    assert main() == 1
