"""The audit compares against a recorded set, not just its own counts.

An absolute count cannot show a regression.  An op that stops being
reachable moves one cell from ``pass`` to ``unsupported`` among fifteen
hundred already-unsupported ones, and nothing in the summary changes
visibly — so "did my change break something" had no answer that did not
depend on remembering yesterday's numbers.

``coverage.json`` records which ``(axis, symbol)`` cells produced a
verdict.  Every run is diffed against it, and a cell that used to answer
and no longer does is reported as a regression.  That is the direction
that matters: the op is still there, the audit simply cannot reach it any
more, which is exactly what a refactor breaks without breaking a test.

Only verdicts are recorded.  ``skip`` and ``unsupported`` carry probe
details that move for reasons unrelated to the framework, and a baseline
that churns is a baseline nobody re-reads.
"""

import importlib
import json

import pytest

from lucid.test.audit.__main__ import (
    _answered,
    _scoped_coverage,
    build_parser,
    load_coverage,
    report_coverage_diff,
    save_coverage,
)
from lucid.test.audit._console import Console
from lucid.test.audit._result import Finding, Report, Status


def _report(*cells: tuple[str, str, Status]) -> Report:
    report = Report()
    for axis, symbol, status in cells:
        report.add(Finding(axis, symbol, status, "detail"))
    return report


def _quiet() -> Console:
    return Console(colour=False, quiet=False)


def test_axis_filter_does_not_report_unrequested_cells_as_lost() -> None:
    recorded = {"layout::lucid.exp": "pass", "grad::lucid.exp": "pass"}
    args = build_parser().parse_args(["--axis", "layout"])
    scoped = _scoped_coverage(recorded, args)
    assert scoped == {"layout::lucid.exp": "pass"}
    assert (
        report_coverage_diff(_report(("layout", "lucid.exp", Status.PASS)), scoped, _quiet()) == 0
    )
    assert (
        report_coverage_diff(_report(("layout", "lucid.exp", Status.SKIP)), scoped, _quiet()) == 1
    )


def test_full_sweep_keeps_removed_symbols_in_the_baseline() -> None:
    recorded = {"layout::lucid.removed_symbol": "pass"}
    assert _scoped_coverage(recorded, build_parser().parse_args([])) == recorded


@pytest.mark.parametrize("selection", [["--axis", "layout"], ["--limit", "1"], ["--quick"]])
def test_partial_sweep_cannot_overwrite_full_coverage_baseline(
    selection: list[str],
) -> None:
    from lucid.test.audit.__main__ import main

    assert main(["--audit-only", "--update-coverage", "--no-log", *selection]) == 2


# ── what gets recorded ────────────────────────────────────────────────────────


def test_only_verdicts_are_recorded() -> None:
    report = _report(
        ("grad", "lucid.exp", Status.PASS),
        ("grad", "lucid.log", Status.FAIL),
        ("grad", "lucid.sin", Status.SKIP),
        ("grad", "lucid.cos", Status.UNSUPPORTED),
        ("grad", "lucid.tan", Status.NOT_APPLICABLE),
    )
    assert _answered(report) == {
        "grad::lucid.exp": "pass",
        "grad::lucid.log": "fail",
    }


def test_a_baseline_round_trips(tmp_path) -> None:
    path = tmp_path / "coverage.json"
    report = _report(("grad", "lucid.exp", Status.PASS))
    assert save_coverage(path, report) == 1
    assert load_coverage(path) == {"grad::lucid.exp": "pass"}


def test_a_missing_baseline_reads_as_nothing_to_compare(tmp_path) -> None:
    assert load_coverage(tmp_path / "absent.json") is None


def test_a_corrupt_baseline_does_not_take_the_run_down(tmp_path) -> None:
    path = tmp_path / "coverage.json"
    path.write_text("{ this is not json")
    assert load_coverage(path) is None


# ── what the diff reports ─────────────────────────────────────────────────────


def test_an_unchanged_run_reports_no_drift(capsys) -> None:
    report = _report(("grad", "lucid.exp", Status.PASS))
    regressions = report_coverage_diff(report, _answered(report), _quiet())
    assert regressions == 0
    assert "unchanged" in capsys.readouterr().out


def test_a_cell_that_stops_answering_is_a_regression(capsys) -> None:
    """The case the whole file exists for: the op is still exported, the
    audit just cannot reach it any more."""
    recorded = {"grad::lucid.exp": "pass", "grad::lucid.log": "pass"}
    now = _report(
        ("grad", "lucid.exp", Status.PASS),
        ("grad", "lucid.log", Status.UNSUPPORTED),
    )
    regressions = report_coverage_diff(now, recorded, _quiet())
    assert regressions == 1
    out = capsys.readouterr().out
    assert "LOST" in out and "lucid.log" in out


def test_a_pass_turning_into_a_failure_is_a_regression(capsys) -> None:
    recorded = {"grad::lucid.exp": "pass"}
    now = _report(("grad", "lucid.exp", Status.FAIL))
    assert report_coverage_diff(now, recorded, _quiet()) == 1
    assert "WORSE" in capsys.readouterr().out


def test_a_newly_answered_cell_is_progress_not_a_regression(capsys) -> None:
    recorded = {"grad::lucid.exp": "pass"}
    now = _report(
        ("grad", "lucid.exp", Status.PASS),
        ("grad", "lucid.log", Status.PASS),
    )
    assert report_coverage_diff(now, recorded, _quiet()) == 0
    out = capsys.readouterr().out
    assert "NEW" in out and "update-coverage" in out


def test_a_failure_that_was_already_a_failure_is_not_new(capsys) -> None:
    """The defect list already reports it; the coverage diff is about
    reachability moving, not about a known failure persisting."""
    recorded = {"grad::lucid.exp": "fail"}
    now = _report(("grad", "lucid.exp", Status.FAIL))
    assert report_coverage_diff(now, recorded, _quiet()) == 0


def test_regressions_and_progress_are_reported_together(capsys) -> None:
    recorded = {"grad::a": "pass", "grad::b": "pass"}
    now = _report(("grad", "a", Status.PASS), ("grad", "c", Status.PASS))
    assert report_coverage_diff(now, recorded, _quiet()) == 1
    out = capsys.readouterr().out
    assert "LOST" in out and "NEW" in out


# ── a symbol taken out of the public surface ─────────────────────────────────
#
# LCD-532: LCD-473 removed ``lucid.pad`` and ``Tensor.pad`` on purpose and
# their 24 recorded cells came back as LOST, holding the gate red over a
# deliberate change.  The probe below is a real public name in a real
# module, so the diff's proof of removal walks the same surface a sweep does.

_PROBE = "audit_removal_probe"


def _probe_fn(x: object) -> object:
    return x


@pytest.fixture
def public_probe(monkeypatch: pytest.MonkeyPatch):
    """Publish ``_PROBE`` in a module; the returned callable takes it out again."""

    def publish(module_path: str) -> object:
        module = importlib.import_module(module_path)

        def probe(x: object) -> object:
            return x

        probe.__module__ = module_path  # owned by Lucid, so the walk counts it
        monkeypatch.setattr(module, _PROBE, probe, raising=False)
        monkeypatch.setattr(module, "__all__", [*module.__all__, _PROBE])
        return module

    def withdraw(module: object) -> None:
        monkeypatch.delattr(module, _PROBE)
        monkeypatch.setattr(module, "__all__", [n for n in module.__all__ if n != _PROBE])

    return publish, withdraw


@pytest.fixture
def tensor_probe(monkeypatch: pytest.MonkeyPatch):
    from lucid._tensor.tensor import Tensor

    monkeypatch.setattr(Tensor, _PROBE, _probe_fn, raising=False)
    return lambda: monkeypatch.delattr(Tensor, _PROBE)


def test_a_cell_of_a_still_public_symbol_is_lost(public_probe, capsys) -> None:
    publish, _ = public_probe
    publish("lucid.special")
    recorded = {f"grad::lucid.special.{_PROBE}": "pass"}
    assert report_coverage_diff(Report(), recorded, _quiet()) == 1
    out = capsys.readouterr().out
    assert "LOST" in out and "REMOVED" not in out


def test_a_cell_of_a_removed_symbol_is_removed_not_lost(public_probe, capsys) -> None:
    publish, withdraw = public_probe
    withdraw(publish("lucid.special"))
    key = f"grad::lucid.special.{_PROBE}"
    assert report_coverage_diff(Report(), {key: "pass"}, _quiet()) == 0
    out = capsys.readouterr().out
    assert "REMOVED" in out and key in out and "LOST" not in out
    assert "update-coverage" in out
    assert "unchanged" not in out


def test_a_removed_tensor_method_is_removed_not_lost(tensor_probe, capsys) -> None:
    key = f"smoke::Tensor.{_PROBE}"
    tensor_probe()
    assert report_coverage_diff(Report(), {key: "pass"}, _quiet()) == 0
    assert "REMOVED" in capsys.readouterr().out


def test_a_removal_and_a_regression_are_told_apart(public_probe, capsys) -> None:
    publish, withdraw = public_probe
    withdraw(publish("lucid.special"))
    recorded = {f"grad::lucid.special.{_PROBE}": "pass", "grad::lucid.exp": "pass"}
    now = _report(("grad", "lucid.exp", Status.SKIP))
    assert report_coverage_diff(now, recorded, _quiet()) == 1
    lines = capsys.readouterr().out.splitlines()
    assert any("LOST" in line and "grad::lucid.exp" in line for line in lines)
    assert any("REMOVED" in line and _PROBE in line for line in lines)


def test_no_removal_is_claimed_while_a_module_fails_to_import(
    public_probe, monkeypatch, capsys
) -> None:
    """A sweep skips an unimportable module silently, so its symbols are
    missing without having been removed — they stay LOST."""
    from lucid.test.audit import _surface

    publish, withdraw = public_probe
    withdraw(publish("lucid.special"))
    real = _surface._import
    monkeypatch.setattr(
        _surface, "_import", lambda path: None if path == "lucid.fft" else real(path)
    )
    key = f"grad::lucid.special.{_PROBE}"
    assert report_coverage_diff(Report(), {key: "pass"}, _quiet()) == 1
    assert "LOST" in capsys.readouterr().out


def test_an_entry_cell_goes_with_the_spelling_it_was_compared_with(public_probe, capsys) -> None:
    """``entry::F.pad`` answered by comparing ``F.pad`` with ``lucid.pad``;
    with the free spelling removed there is one entry point and no question."""
    publish, _ = public_probe
    publish("lucid.nn.functional")
    recorded = {f"entry::F.{_PROBE}": "pass", f"entry::lucid.{_PROBE}": "pass"}
    now = _report(("entry", f"F.{_PROBE}", Status.NOT_APPLICABLE))
    assert report_coverage_diff(now, recorded, _quiet()) == 0
    out = capsys.readouterr().out
    assert out.count("REMOVED") == 2 and "LOST" not in out
    assert f"lucid.{_PROBE}, is gone" in out


@pytest.mark.parametrize("status", [Status.SKIP, Status.UNSUPPORTED], ids=["skip", "unsupported"])
def test_an_entry_cell_that_still_has_its_question_stays_lost(
    public_probe, capsys, status: Status
) -> None:
    """Only "one entry point left" follows a removed spelling.  A cell
    that now skips or refuses was asked and could not be answered."""
    publish, _ = public_probe
    publish("lucid.nn.functional")
    recorded = {f"entry::F.{_PROBE}": "pass", f"entry::lucid.{_PROBE}": "pass"}
    now = _report(("entry", f"F.{_PROBE}", status))
    assert report_coverage_diff(now, recorded, _quiet()) == 1
    assert "LOST" in capsys.readouterr().out


def test_an_entry_cell_with_no_removed_spelling_stays_lost(public_probe, capsys) -> None:
    publish, _ = public_probe
    publish("lucid.nn.functional")
    recorded = {f"entry::F.{_PROBE}": "pass"}
    now = _report(("entry", f"F.{_PROBE}", Status.NOT_APPLICABLE))
    assert report_coverage_diff(now, recorded, _quiet()) == 1
    assert "LOST" in capsys.readouterr().out


# ── the checked-in baseline ───────────────────────────────────────────────────


def test_the_repository_baseline_is_readable_and_substantial() -> None:
    import pathlib

    path = pathlib.Path(__file__).parents[2] / "audit" / "coverage.json"
    if not path.exists():
        pytest.skip("no baseline recorded in this checkout")
    recorded = load_coverage(path)
    assert recorded is not None
    assert len(recorded) > 5000, len(recorded)
    assert all("::" in key for key in recorded)
    payload = json.loads(path.read_text())
    assert payload["cells"] == len(recorded)
