"""``lucid.optim`` may not import ``lucid.nn``, except the modules that are named.

The reference ``swa_utils`` subclasses ``nn.Module`` and resets BatchNorm
statistics, so ``lucid.optim.swa_utils`` is excepted — that one module and those
two imports.  Any other optimizer module reaching into ``nn`` still fails.
"""

import sys
from pathlib import Path

import pytest

_TOOLS_DIR = str(Path(__file__).resolve().parents[3] / "tools")
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

import check_layers


def _tree(root: Path, module: str, source: str) -> Path:
    path = root / "lucid" / "optim" / f"{module}.py"
    path.parent.mkdir(parents=True)
    path.write_text(source)
    return root / "lucid"


@pytest.mark.parametrize(
    "imported",
    ["lucid.nn.module", "lucid.nn.modules.normalization"],
)
def test_swa_utils_may_import_the_two_named_nn_modules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, imported: str
) -> None:
    monkeypatch.setattr(check_layers, "ROOT", tmp_path)
    lucid_dir = _tree(tmp_path, "swa_utils", f"import {imported}\n")
    assert check_layers.check_python_layers(lucid_dir) == []


@pytest.mark.parametrize("module", ["adam", "swa_utils_clone"])
@pytest.mark.parametrize(
    "imported", ["lucid.nn.module", "lucid.nn.modules.normalization", "lucid.nn"]
)
def test_any_other_optim_module_importing_nn_still_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module: str, imported: str
) -> None:
    monkeypatch.setattr(check_layers, "ROOT", tmp_path)
    lucid_dir = _tree(tmp_path, module, f"import {imported}\n")
    assert len(check_layers.check_python_layers(lucid_dir)) == 1


def test_swa_utils_importing_other_nn_modules_still_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(check_layers, "ROOT", tmp_path)
    lucid_dir = _tree(tmp_path, "swa_utils", "import lucid.nn.functional\n")
    assert len(check_layers.check_python_layers(lucid_dir)) == 1


def test_the_real_tree_has_no_python_layer_violation() -> None:
    assert check_layers.check_python_layers() == []
