"""No report means unverified; a registry entry is not a successful probe."""

import json
from pathlib import Path

import pytest

from tools.support_manifest import read_evidence, snapshot
from tools._support_contracts import (
    canonical,
    documentation,
    stub_declarations,
    test_references as index_test_calls,
)


def test_missing_evidence_is_not_a_pass() -> None:
    report = snapshot()
    assert report["evidence"] == {"audit": None, "pretrained": None}
    assert report["counts"]["model_factories"] > 0
    assert report["contract"]["missing_evidence"] == "unverified, not unsupported"
    assert all("supported" not in entry for entry in report["registrations"]["native_schemas"])


def test_scoped_failures_and_skips_survive_attachment(tmp_path: Path) -> None:
    path = tmp_path / "audit.json"
    original = {
        "config": {"axis": "layout"},
        "findings": [
            {"symbol": "example", "status": "fail"},
            {"symbol": "other", "status": "skip"},
        ],
        "coverage": {},
        "platform": {"machine": "arm64"},
    }
    path.write_text(json.dumps(original))
    evidence = read_evidence(path, "audit")
    assert evidence["report"] == original
    assert evidence["source_match"] == "not_verified"
    assert len(evidence["sha256"]) == 64


def test_wrong_report_kind_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_text('{"complete": true}')
    with pytest.raises(ValueError, match="not a audit evidence report"):
        read_evidence(path, "audit")


def test_contract_indexes_preserve_exact_names_and_locations(tmp_path: Path) -> None:
    tests = tmp_path / "lucid/test"
    tests.mkdir(parents=True)
    (tests / "test_example.py").write_text(
        "import lucid as lc\nfrom lucid.nn import Linear as Dense\nlc.sin(1)\nDense(2, 3)\n"
    )
    calls = index_test_calls(tmp_path)
    assert calls["lucid.sin"] == ["lucid/test/test_example.py:3"]
    assert calls["lucid.nn.Linear"] == ["lucid/test/test_example.py:4"]
    (tmp_path / "lucid/__init__.pyi").write_text("class complex128: ...\n")
    assert stub_declarations(tmp_path)["lucid.complex128"] == ["lucid/__init__.pyi:1"]
    docs = tmp_path / "web/public/api-data"
    docs.mkdir(parents=True)
    (docs / "lucid.json").write_text('{"members": [{"path": "lucid.sin"}]}')
    assert documentation(tmp_path)["lucid.sin"] == ["web/public/api-data/lucid.json"]
    assert canonical("F.relu") == "lucid.nn.functional.relu"
    assert canonical("Tensor.sum") == "lucid.Tensor.sum"


def test_complex128_is_declared_in_the_generated_stub() -> None:
    root = Path(__file__).resolve().parents[4]
    assert "lucid.complex128" in stub_declarations(root)
