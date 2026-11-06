"""The second review of LCD-840: what was still wrong after the first round of fixes.

* ``nohup`` ignores ``SIGHUP`` so a run started for the night survives its
  terminal; installing a handler over that ignore turned the closed terminal
  back into the death ``nohup`` exists to prevent.
* A family's tests were found by the factory names in their parameter ids, so
  the tests spelled by *class* (``TestGPT2Cache``, ``[AlexNetForImageClassification]``)
  belonged to nobody.
* ``--zoo-device`` had a default, so on a run with no zoo stage it looked unused.
* ``--suite-path`` / ``--suite-ignore`` run a tree the floor was not measured
  over, and still got a verdict that could fail the run.
* ``lucid-audit`` needed pytest to print ``--list-packages``.
* Compiled training ignored ``--zoo-device`` and the note had to say so; it is now a
  stage that declares itself Metal only.
* The verdict said "clean on every stage that ran" for a run that left things
  out, and its zoo number did not say what it counted.
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from lucid.test.audit import _zoo
from lucid.test.audit._console import Console
from lucid.test.audit._suite import report_line_coverage_diff
from lucid.test.audit._zoo_budget import MachineBudget
from lucid.test.audit._zoo_device import COMPILED_TRAINING_DEVICE, OnlyOn
from lucid.test.audit._zoo_guard import exit_on_terminate
from lucid.test.audit._zoo_record import RunFacts, render_note
from lucid.test.audit._zoo_select import FamilySelector
from lucid.test.audit._zoo_stage import StageResult, ZooCatalog
from lucid.test.audit.__main__ import _verdict, build_parser, main
from lucid.test.audit.test_audit_honesty import _lost, _plan, _r
from lucid.test.audit.test_audit_zoo import _budget, _Run

ROOT = Path(__file__).resolve().parents[3]
P = "lucid/test/unit/models/"


# ── an ignored hangup stays ignored ──────────────────────────────────────────


def test_a_hangup_that_is_ignored_stays_ignored_and_a_terminate_is_still_handled() -> None:
    before_term = signal.getsignal(signal.SIGTERM)
    before_hup = signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        with exit_on_terminate():
            assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN, "nohup's ignore was replaced"
            assert signal.getsignal(signal.SIGTERM) not in (before_term, signal.SIG_IGN)
        assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN
        assert signal.getsignal(signal.SIGTERM) is before_term
    finally:
        signal.signal(signal.SIGHUP, before_hup)


def test_a_hangup_that_is_not_ignored_still_ends_the_run() -> None:
    before = signal.signal(signal.SIGHUP, signal.SIG_DFL)
    try:
        with exit_on_terminate():
            assert callable(signal.getsignal(signal.SIGHUP))
    finally:
        signal.signal(signal.SIGHUP, before)


def _write_driver(tmp_path: Path) -> Path:
    (tmp_path / "child.py").write_text(
        "import os, sys, time\nopen(sys.argv[1], 'w').write(str(os.getpid()))\ntime.sleep(60)\n"
    )
    driver = tmp_path / "driver.py"
    driver.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "from lucid.test.audit._zoo_budget import MachineBudget\n"
        "from lucid.test.audit._zoo_guard import StageGuard, exit_on_terminate\n"
        "tmp = Path(sys.argv[1])\n"
        "guard = StageGuard(tmp, MachineBudget(64.0, 40.0, 0.0, 0.0), poll_s=0.05)\n"
        "with exit_on_terminate():\n"
        "    guard.run((sys.executable, str(tmp / 'child.py'), str(tmp / 'stage.pid')), tmp / 'log')\n"
    )
    return driver


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_a_run_started_under_nohup_survives_a_hangup_and_still_ends_on_terminate(
    tmp_path: Path,
) -> None:
    """``nohup python3 -m lucid.test.audit --models ...`` then ``kill -HUP`` exited 129."""
    driver = _write_driver(tmp_path)
    proc = subprocess.Popen(
        ["nohup", sys.executable, str(driver), str(tmp_path)],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    pidfile = tmp_path / "stage.pid"
    try:
        deadline = time.monotonic() + 60
        while not pidfile.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert pidfile.exists(), "the stage never started"
        child = int(pidfile.read_text())
        proc.send_signal(signal.SIGHUP)
        time.sleep(1.0)
        assert proc.poll() is None, f"the hangup killed a nohup'd run (exit {proc.returncode})"
        assert _alive(child), "and took its stage with it"
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=30) == 128 + signal.SIGTERM
    finally:
        if proc.poll() is None:
            proc.kill()
    deadline = time.monotonic() + 5
    while _alive(child) and time.monotonic() < deadline:
        time.sleep(0.05)
    if _alive(child):
        os.kill(child, signal.SIGKILL)
        pytest.fail("the stage outlived the run that was terminated")


# ── a family's tests, spelled by class ───────────────────────────────────────


@pytest.mark.parametrize(
    ("family", "nodeid", "claims"),
    [
        ("gpt2", P + "test_cache_generation.py::TestGPT2Cache::test_x", True),
        ("alexnet", P + "test_family_contract.py::test_x[AlexNetForImageClassification]", True),
        ("gpt2", P + "test_infra.py::TestWrap::test_x[GPT2DoubleHeadsModel]", True),
        ("gpt", P + "test_infra.py::TestWrap::test_x[GPTDoubleHeadsModel]", True),
        ("bert", P + "test_zoo_dtype.py::test_x[bert0-cpu-float64]", True),
        ("gpt2", P + "test_zoo_dtype.py::test_x[gpt2_0-metal-float16]", True),
        ("yolo", P + "test_models_detection.py::TestYOLOV3::test_factory_and_forward[cpu]", True),
        ("diamond", P + "test_recent.py::TestDIAMONDCSGO::test_the_denoiser_trains", True),
        ("senet", P + "test_x.py::test_y[se_resnet_50]", True),
        ("densenet", P + "test_x.py::test_y[densenet_121]", True),
        # and the ones substring matching got wrong stay wrong
        ("senet", P + "test_x.py::TestDenseNetForward::test_y[cpu]", False),
        ("senet", P + "test_x.py::test_y[DenseNetModel]", False),
        ("senet", P + "test_x.py::test_y[densenet_121]", False),
        ("resnet", P + "test_x.py::test_y[se_resnet_50]", False),
        ("lenet", P + "test_x.py::test_y[googlenet]", False),
    ],
)
def test_a_family_is_found_by_the_class_names_its_tests_are_written_with(
    family: str, nodeid: str, claims: bool
) -> None:
    assert FamilySelector(*ZooCatalog(ROOT).selection_for(family)).matches(nodeid) is claims


def test_the_registry_gives_a_family_its_model_and_config_classes() -> None:
    names, labels, classes = ZooCatalog(ROOT).selection_for("alexnet")
    assert "AlexNetForImageClassification" in classes
    assert "alexnet" in names and labels == ("alexnet",)


def test_a_selector_camel_splits_a_test_head_and_spells_runs_of_words() -> None:
    selector = FamilySelector(["se_resnet_50"], ["senet"], ["SENetModel"])
    assert selector.matches("t.py::TestSENetForward::test_x")
    assert selector.matches("t.py::test_senet_forward")
    assert not selector.matches("t.py::TestDenseNetForward::test_x"), "Dense + Net is not senet"
    named = FamilySelector([], ["alexnet"], [])
    assert named.matches("t.py::TestAlexNet::test_x"), "Alex + Net spells alexnet"


def test_the_stage_hands_the_classes_to_the_plugin(tmp_path: Path) -> None:
    stage = ZooCatalog(ROOT).family_stages("alexnet", tmp_path, MachineBudget.for_memory(36))[-1]
    argv = stage.argv()
    assert any(
        a.startswith("--zoo-classes=") and "AlexNetForImageClassification" in a for a in argv
    )


# ── a zoo option on a run with no zoo stage ──────────────────────────────────


@pytest.mark.parametrize(
    "argv",
    [
        ["--no-models", "--zoo-device", "cpu"],
        ["--audit-only", "--zoo-device", "cpu"],
        ["--tests-only", "--zoo-device", "cpu"],
        ["--doctests-only", "--zoo-device", "metal"],
        ["--nn", "--zoo-device", "cpu"],
    ],
)
def test_a_zoo_device_given_to_a_run_that_skips_the_zoo_is_exit_2(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([*argv, "--no-log"]) == 2
    assert "--zoo-device" in capsys.readouterr().out


def test_a_zoo_device_is_fine_where_the_zoo_runs() -> None:
    assert _plan("--models", "--zoo-device", "cpu").zoo
    assert _plan("--zoo-device", "cpu").zoo, "no package flag: the zoo is included"


# ── a tree the floor was not measured over cannot fail the run ───────────────


def test_a_suite_path_or_a_suite_ignore_makes_the_floor_comparison_advisory(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--tests-only --suite-path lucid/test/unit/fft`` gave 202 regressions and exit 1."""
    plan = _plan("--tests-only", "--no-doctests", "--suite-path", "lucid/test/unit/fft")
    args = build_parser().parse_args(
        ["--tests-only", "--no-doctests", "--suite-path", "lucid/test/unit/fft"]
    )
    where = plan.selection.suite_plan(ROOT, args.suite_path, extra_ignore=args.suite_ignore)
    assert where.paths == ("lucid/test/unit/fft",) and where.advisory
    result, floor = _lost()
    console = Console(colour=False)
    assert (
        report_line_coverage_diff(
            result, floor, console, in_scope=where.in_scope, advisory=where.advisory
        )
        == 0
    )
    said = capsys.readouterr().out
    assert "advisory only" in said and "--update-suite to accept" not in said

    ignoring = build_parser().parse_args(["--tests-only", "--suite-ignore", "lucid/test/unit/nn"])
    plan = _plan("--tests-only", "--suite-ignore", "lucid/test/unit/nn")
    where = plan.selection.suite_plan(ROOT, "lucid/test", extra_ignore=ignoring.suite_ignore)
    assert where.ignore == ("lucid/test/unit/nn",) and where.advisory


def test_the_whole_suite_is_still_the_one_run_that_can_fail() -> None:
    where = _plan("--tests-only").selection.suite_plan(ROOT, "lucid/test", extra_ignore=())
    assert not where.advisory and where.ignore == ()


# ── lucid-audit does not need pytest ─────────────────────────────────────────


def test_lucid_audit_runs_with_pytest_unimportable() -> None:
    code = (
        "import sys\n"
        "sys.modules['pytest'] = None\n"
        "from lucid.test.audit.__main__ import main\n"
        "assert main(['--list-packages', '--no-log']) == 0\n"
        "assert main(['--models', '--family', 'no_such', '--no-log']) == 2\n"
        "assert sys.modules['pytest'] is None\n"
        "print('ok')\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert done.stdout.rstrip().endswith("ok"), done.stdout[-400:] + done.stderr[-400:]


# ── compiled training is Metal only, and says so in its scope ────────────────


def test_compiled_training_is_scoped_to_the_device_its_matrix_runs_on() -> None:
    """The scope names Metal; the matrix moves every model to ``COMPILE_DEVICE`` itself."""
    from lucid.test.unit.compile._helpers import COMPILE_DEVICE

    assert COMPILED_TRAINING_DEVICE == COMPILE_DEVICE
    stages = {s.name: s for s in ZooCatalog(ROOT).family_stages("lenet", Path("/x"), _budget(40.0))}
    assert stages["compiled-training"].scope == OnlyOn(COMPILE_DEVICE)


def test_the_note_names_the_device_of_each_row_and_the_scope_of_the_stages(tmp_path: Path) -> None:
    metal = StageResult("compiled-training", "pass", 1.0, 1.0, device="metal")
    results = [metal, StageResult("own-tests", "pass", 1.0, 1.0, device="metal")]
    results.append(StageResult("own-tests", "pass", 1.0, 1.0, device="cpu"))
    note = render_note("alpha", "vision", results, RunFacts("a" * 40, "c", _budget(36.0)))
    rows = [line.split("|")[1:3] for line in note.splitlines() if line.startswith("| metal")]
    assert [[c.strip() for c in row] for row in rows] == [
        ["metal", "compiled-training"],
        ["metal", "own-tests"],
    ]
    assert "| cpu | own-tests |" in note
    assert "contract = 한 번 (장치와 무관)" in note and "compiled-training = metal 에서만" in note


# ── the verdict says when a clean run left things out, and what it counts ────


def test_the_verdict_does_not_call_a_reduced_run_clean_on_every_stage(
    capsys: pytest.CaptureFixture[str],
) -> None:
    console = Console(colour=False)
    assert (
        _verdict(console, {"model zoo failing stages": 0}, {"zoo group(s) passed reduced": 3}) == 0
    )
    out = capsys.readouterr().out
    assert "zoo group(s) passed reduced" in out and "3" in out
    assert "clean on every stage that ran, but 3 zoo group(s) passed reduced" in out
    assert _verdict(console, {"x": 0}, {"zoo group(s) passed reduced": 0}) == 0
    again = capsys.readouterr().out
    assert "clean on every stage that ran" in again and "but" not in again
    assert (
        _verdict(console, {"model zoo failing stages": 2}, {"zoo group(s) passed reduced": 3}) == 1
    )


def test_a_reduced_zoo_run_reaches_the_verdict(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    report = _zoo.ZooReport(reduced=["lenet", "t5"])
    monkeypatch.setattr(_zoo, "run_zoo_stage", lambda *args: report)
    assert main(["--models", "--no-log"]) == 0
    assert (
        "clean on every stage that ran, but 2 zoo group(s) passed reduced"
        in capsys.readouterr().out
    )


def _failing(groups: int, total: int = 71) -> _zoo.ZooReport:
    report = _zoo.ZooReport()
    for n in range(total):
        if n < groups:
            stages = [_r("fail", failures=[f"t.py::test_{n} - boom"]), _r("fail")]
        else:
            stages = [_r()]
        report.add(f"fam{n:02d}", stages, resumed=False)
        if n < groups:
            report.adapted.append((f"fam{n:02d}", "build", "past this machine's budget"))
    return report


def test_the_zoo_summary_labels_its_numbers_and_points_at_the_complete_list(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = _failing(20)
    report.summary_path = tmp_path / "summary.txt"
    report.summary_path.write_text(_zoo.render_summary(report))
    _zoo.summarise(Console(colour=False), report, MachineBudget.for_memory(36), Path("x"))
    out = capsys.readouterr().out
    assert "failing stages" in out and "40 in 20 of 71 group(s)" in out
    assert "stages, not tests" in out, "the number says what it counts"
    assert out.count("FAIL  fam") == 8, "the screen is short"
    assert out.count("past this machine's budget") == 8
    assert str(report.summary_path) in out, "and names the file with the rest"
    full = report.summary_path.read_text()
    assert (
        full.count("(fail)") == 40 and "fam19/s" in full and full.count("past this machine's") == 20
    )
    assert report.failures == 40 and len(report.failing_groups) == 20


def test_a_short_summary_does_not_send_you_to_a_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = _failing(2, total=5)
    report.summary_path = tmp_path / "summary.txt"
    _zoo.summarise(Console(colour=False), report, MachineBudget.for_memory(36), Path("x"))
    assert "complete lists" not in capsys.readouterr().out


def test_a_run_writes_its_complete_lists_beside_the_logs(tmp_path: Path) -> None:
    runs = _Run(tmp_path, failing=("alpha",))
    report = runs.go()
    assert report.summary_path is not None and report.summary_path.parent.name == "a" * 40
    text = report.summary_path.read_text()
    assert text.startswith("failing stages 2 in 1 of 3 group(s)")
    assert "alpha/contract (fail, metal)" in text and "== failing stages · 2" in text
    assert (report.summary_path.parent / "logs" / "alpha" / "build.log").exists()
