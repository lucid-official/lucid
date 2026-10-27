"""Aggregation must preserve provenance and must not select only passing runs."""

import json
from pathlib import Path

import pytest

from tools.merge_pretrained_evidence import merge


def _report(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(json.dumps({"evidence": rows, "finished": False, "complete": False}))
    return path


def _pass(name: str) -> dict[str, object]:
    return {"model": name, "max_diff": 1e-6, "top1_agrees": True}


def test_partial_reports_can_cover_disjoint_factories_with_provenance(tmp_path: Path) -> None:
    first = _report(tmp_path / "first.json", [_pass("a")])
    second = _report(tmp_path / "second.json", [_pass("b")])
    result = merge([first, second], {"a", "b"})
    assert result["complete"] is True
    assert result["compared"] == 2
    assert result["evidence"][1]["evidence_report"] == 1
    assert len(result["reports"][0]["sha256"]) == 64
    assert result["reports"][0]["finished"] is False


@pytest.mark.parametrize(
    "later",
    [
        {"model": "a", "max_diff": 0.5, "top1_agrees": True},
        {"model": "a", "error": "download failed"},
        {"model": "a", "unreachable": "missing reference"},
    ],
)
def test_later_failure_or_unverified_observation_supersedes_pass(
    tmp_path: Path, later: dict[str, object]
) -> None:
    first = _report(tmp_path / "first.json", [_pass("a")])
    second = _report(tmp_path / "second.json", [later])
    result = merge([first, second], {"a"})
    assert result["complete"] is False
    assert result["evidence"][0] == {**later, "evidence_report": 1}


@pytest.mark.parametrize("difference", [float("nan"), float("inf"), -1, True, "0"])
def test_invalid_numeric_evidence_never_passes(tmp_path: Path, difference: object) -> None:
    report = _report(tmp_path / "invalid.json", [{**_pass("a"), "max_diff": difference}])
    result = merge([report], {"a"})
    assert result["complete"] is False
    assert result["problems"]


def test_missing_and_unexpected_factories_are_explicit(tmp_path: Path) -> None:
    report = _report(tmp_path / "partial.json", [_pass("a"), _pass("unregistered")])
    result = merge([report], {"a", "b"})
    assert result["complete"] is False
    assert result["missing"] == ["b"]
    assert result["unexpected"] == ["unregistered"]


def test_duplicate_observations_in_one_report_are_rejected(tmp_path: Path) -> None:
    report = _report(tmp_path / "duplicate.json", [_pass("a"), _pass("a")])
    with pytest.raises(ValueError, match="duplicate observation"):
        merge([report], {"a"})


def test_successful_recheck_keeps_original_report_failure_diagnostics(tmp_path: Path) -> None:
    failed = tmp_path / "failed.json"
    failed.write_text(
        json.dumps(
            {
                "evidence": [{"model": "a", "error": "checksum mismatch"}],
                "problems": ["a: checksum mismatch"],
                "unreachable": 0,
                "complete": False,
            }
        )
    )
    recheck = _report(tmp_path / "recheck.json", [_pass("a")])
    result = merge([failed, recheck], {"a"})
    assert result["complete"] is True
    assert result["reports"][0]["problems"] == ["a: checksum mismatch"]
    assert result["reports"][0]["complete"] is False
    assert result["evidence"][0]["evidence_report"] == 1
