"""A record, a baseline or a verdict must not say more than was checked.

The review of the first version of this stage found places where it did, and
each test below would have failed against it:

* a baseline written from a narrowed run (a 26.98% floor over an 85.92% one),
  and a line-coverage "regression" that was only fewer tests running;
* a zoo record made on a small machine read as a pass by a large one, a
  machine running short of memory answered by running *less* of the stage, a
  guard that watched only the process it started, and matrix pairs that were
  skipped reporting ``ok``;
* a family selected by substring, a tree that changed mid-run, a stage
  killed by ``SIGTERM`` leaving its child behind.
"""

import argparse
import dataclasses
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import override

import pytest

from lucid.test.audit import _zoo, _zoo_probes
from lucid.test.audit._console import Console
from lucid.test.audit._plan import PlanError, RunPlan
from lucid.test.audit._suite import SuiteResult, report_line_coverage_diff
from lucid.test.audit._zoo_budget import MachineBudget, fits_build, group_pids
from lucid.test.audit._zoo_guard import StageGuard, StageRunner, exit_on_terminate
from lucid.test.audit._zoo_record import (
    FamilyRecord,
    Notebook,
    RunFacts,
    ZooBusy,
    ZooState,
    overall,
    render_note,
    repo_state,
    write_atomic,
)
from lucid.test.audit._zoo_report import LineReporter
from lucid.test.audit._zoo_select import FamilySelector
from lucid.test.audit._zoo_stage import (
    ZOO_WIDE,
    CommandStage,
    Stage,
    StageResult,
    ZooCatalog,
    read_result,
)
from lucid.test.audit.__main__ import build_parser, main

ROOT = Path(__file__).resolve().parents[3]
SHA = "a" * 40


def _plan(*argv: str) -> RunPlan:
    return RunPlan.from_args(build_parser().parse_args(list(argv)))


def _budget(limit_gb: float, floor_gb: float = 0.0, start_room_gb: float = 0.0) -> MachineBudget:
    return MachineBudget(
        total_gb=64.0, limit_gb=limit_gb, start_room_gb=start_room_gb, floor_gb=floor_gb
    )


def _python(code: str) -> tuple[str, ...]:
    return (sys.executable, "-c", code)


def _guard(tmp_path: Path, budget: MachineBudget) -> StageGuard:
    return StageGuard(tmp_path, budget, poll_s=0.05)


# ── baselines are written only by the run that is the whole of them ──────────


@pytest.mark.parametrize(
    "argv",
    [
        ["--fft", "--tests-only", "--no-doctests", "--update-suite"],  # 26.98% over 85.92%
        ["--fft", "--doctests-only", "--update-doctests"],  # 6,932 examples became 135
        ["--no-models", "--tests-only", "--update-suite"],  # a floor from a filtered tree
        ["--update-suite"],  # the zoo's trees go to the zoo stage
        ["--tests-only", "--update-suite", "--suite-ignore", "lucid/test/unit/ops"],
        ["--tests-only", "--update-suite", "--suite-path", "lucid/test/unit/ops"],
        ["--no-nn", "--doctests-only", "--update-doctests"],
        ["--audit-only", "--update-coverage", "--no-nn"],
    ],
)
def test_a_baseline_is_refused_from_a_run_that_is_not_the_whole_of_it(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([*argv, "--no-log"]) == 2
    assert "requires" in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv",
    [
        ["--tests-only", "--update-suite"],
        ["--doctests-only", "--update-doctests"],
        ["--no-models", "--doctests-only", "--update-doctests"],  # doctests never held the zoo
        ["--audit-only", "--update-coverage"],
        ["--no-models", "--audit-only", "--update-coverage"],
    ],
)
def test_the_official_rewrite_commands_are_allowed(argv: list[str]) -> None:
    assert _plan(*argv)


def test_the_hints_tell_you_the_command_that_writes_each_baseline_whole() -> None:
    from tools.check_audit_baselines import BASELINES

    hints = {rel.rsplit("/", 1)[1]: hint for rel, (_, hint) in BASELINES.items()}
    assert hints["suite.json"] == "lucid-audit --tests-only --update-suite"
    assert hints["coverage.json"] == "lucid-audit --audit-only --update-coverage"
    assert hints["doctest.json"] == "lucid-audit --doctests-only --update-doctests"


def _lost() -> tuple[SuiteResult, dict[str, object]]:
    floor: dict[str, object] = {
        "percent": 90.0,
        "files": {"lucid/nn/a.py": {"covered": 90, "total": 100}},
    }
    result = SuiteResult(
        ran=True, line_covered=10, line_total=100, per_file={"lucid/nn/a.py": (10, 100)}
    )
    return result, floor


def test_a_run_that_did_not_run_the_whole_suite_cannot_fail_on_the_floor(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--autograd`` "lost" ``_functional.py`` 90.9 -> 82.7 and exited 1, on fewer tests."""
    result, floor = _lost()
    console = Console(colour=False)
    narrowed = _plan("--nn").selection.file_scope
    assert report_line_coverage_diff(result, floor, console, in_scope=narrowed, advisory=True) == 0
    said = capsys.readouterr().out
    assert "line coverage (advisory)" in said and "LOST   lucid/nn/a.py" in said
    assert "advisory only" in said
    # The same finding from a run entitled to give a verdict still fails it.
    assert report_line_coverage_diff(result, floor, console, in_scope=narrowed) == 1


def test_the_zoo_files_are_neither_compared_nor_missed_when_their_tests_are_not_run() -> None:
    """Measured: with their trees ignored, three floor files fell >2 points on fewer tests."""
    for argv, zoo_stage in (([], True), (["--no-models"], False)):
        plan = _plan(*argv).selection.suite_plan(ROOT, zoo_stage=zoo_stage)
        assert plan.advisory and plan.in_scope is not None
        assert not plan.in_scope("lucid/models/vision/lenet/_model.py")
    assert not _plan("--tests-only").selection.suite_plan(ROOT).advisory


# ── options that would be silently ignored on a zoo-only run ─────────────────


@pytest.mark.parametrize(
    "argv",
    [
        ["--cpu-only"],
        ["--json", "report.json"],
        ["--select", "conv"],
        ["--quick"],
        ["--fail-fast"],
        ["--axis", "grad"],
        ["--no-self-check"],
        ["--update-known"],
        ["--no-line-coverage"],
        ["--suite-path", "lucid/test/unit/ops"],
        ["--suite-ignore", "lucid/test/unit/ops"],
        ["--no-doctests"],
    ],
)
def test_an_option_for_a_stage_a_zoo_only_run_skips_is_exit_2(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--models", *argv, "--no-log"]) == 2
    assert "zoo-only run skips" in capsys.readouterr().out


def test_the_same_options_are_fine_when_their_stage_runs() -> None:
    assert _plan("--models", "--nn", "--cpu-only", "--quick", "--select", "conv").sweep


# ── a record is only as good as the machine that made it ─────────────────────


@dataclass(frozen=True, kw_only=True)
class _Lever(CommandStage):
    """A stage that can be reduced, the way a pytest stage drops ``heavy``."""

    lever: bool = True

    @override
    def without_heavy(self, why: str) -> Stage | None:
        if not self.lever:
            return None
        return dataclasses.replace(self, lever=False, notes=(*self.notes, why))


class _Catalog(ZooCatalog):
    def __init__(self, root: Path, counter: Path) -> None:
        super().__init__(root)
        self.counter = counter

    @override
    def families(self) -> dict[str, str]:
        return {"alpha": "vision", "beta": "text"}

    def _stage(self, group: str, name: str) -> _Lever:
        code = f"open({str(self.counter)!r}, 'a').write({group!r} + '/' + {name!r} + '\\n')"
        return _Lever(name=name, command=_python(code))

    @override
    def family_stages(self, family: str, temp: Path, budget: MachineBudget) -> list[Stage]:
        return [self._stage(family, "build"), self._stage(family, "own-tests")]

    @override
    def wide_stages(self, temp: Path) -> list[Stage]:
        return [self._stage(ZOO_WIDE, "tests/test_wide.py")]


class _Runs:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.counter = root / "counter"
        self.revision = (SHA, False)

    def runs(self) -> int:
        return len(self.counter.read_text().split()) if self.counter.exists() else 0

    def go(
        self, limit_gb: float, *, notes: bool = False, families: tuple[str, ...] = ()
    ) -> _zoo.ZooReport:
        console = Console(colour=False)
        settings = _zoo.ZooSettings(
            families=families,
            state_root=self.root / "state",
            poll_s=0.05,
            record_dir=self.root / "notes" if notes else None,
        )
        run = _zoo.ZooRun(
            self.root,
            console,
            settings,
            budget=MachineBudget(64.0, limit_gb, 0.0, 0.0),
            reporter=LineReporter(console, limit_gb),
            catalog=_Catalog(self.root, self.counter),
            revision=lambda: self.revision,
        )
        return run.run()


def test_a_record_made_on_a_smaller_machine_is_not_reused_by_a_larger_one(tmp_path: Path) -> None:
    """lenet at 6 GB, then at 28 GB, printed "recorded for this commit" and ran no heavy test."""
    runs = _Runs(tmp_path)
    small = runs.go(6.0, families=("alpha",))
    assert small.reduced == ["alpha"], "the small machine left heavy tests out, and says so"
    ran = runs.runs()
    big = runs.go(28.0, families=("alpha",))
    assert big.resumed == 0 and runs.runs() > ran, "a larger machine would have run more"
    assert big.reduced == []
    # The other way round, and the same size, are fine: nothing more was available.
    assert runs.go(28.0, families=("alpha",)).resumed == 1
    assert runs.go(6.0, families=("alpha",)).resumed == 1


def test_a_record_that_left_nothing_out_is_good_for_any_machine(tmp_path: Path) -> None:
    runs = _Runs(tmp_path)
    runs.go(40.0)
    assert runs.go(6.0).resumed == 3
    assert runs.go(200.0).resumed == 3


def test_a_machine_short_record_is_never_reused() -> None:
    record = FamilyRecord([StageResult("build", "machine-short", 1.0, 1.0)], limit_gb=40.0)
    assert not record.reusable_under(_budget(40.0))
    assert FamilyRecord([StageResult("build", "pass", 1.0, 1.0)], 6.0).reusable_under(_budget(40.0))
    killed = FamilyRecord([StageResult("t", "mem-guard", 1.0, 7.0)], limit_gb=6.0)
    assert not killed.reusable_under(_budget(28.0)) and killed.reusable_under(_budget(6.0))


# ── reduced is not pass ──────────────────────────────────────────────────────


def _r(status: str = "pass", **fields: list[str]) -> StageResult:
    return StageResult("s", status, 1.0, 1.0, **fields)


@pytest.mark.parametrize(
    ("results", "expected"),
    [
        ([_r()], "pass"),
        ([_r(notes=["heavy tests deselected up front: ..."])], "reduced"),
        ([_r(notes=["retry with heavy deselected: pass"])], "reduced"),
        ([_r("none")], "reduced"),  # ran nothing and said nothing
        ([_r("none", skips=["1 × t.py:3: needs the hub"])], "pass"),  # a designed skip
        ([_r(skips=["3 × t.py:9: set LUCID_TEST_NETWORK=1"])], "pass"),
        ([_r(events=["waited 30:00 for 14 GB of free memory"])], "pass"),
        ([_r(notes=["x"]), _r("fail")], "fail"),
        ([_r("machine-short")], "machine-short"),
        ([_r("mem-guard"), _r(notes=["x"])], "mem-guard"),
    ],
)
def test_overall_does_not_call_reduced_verification_a_pass(
    results: list[StageResult], expected: str
) -> None:
    assert overall(results) == expected


def test_the_note_states_what_reduced_means() -> None:
    facts = RunFacts(SHA, "lucid-audit --models", MachineBudget.for_memory(36))
    note = render_note("alpha", "vision", [_r(notes=["heavy deselected"])], facts)
    assert "> **reduced**" in note
    assert "reduced = " in note and "설계된 skip" in note


# ── a killed stage is answered by the kind of kill it was ────────────────────


def test_the_guard_counts_the_whole_process_group_not_just_the_leader(tmp_path: Path) -> None:
    """A child allocating 600 MB under a 150 MB limit gave ``pass, peak 0.017``."""
    code = (
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', "
        "'import time; x = b\"x\" * (300 * 2**20); time.sleep(30)'])\n"
        "time.sleep(30)\n"
    )
    outcome = _guard(tmp_path, _budget(0.15)).run(_python(code), tmp_path / "log")
    assert outcome.status == "mem-guard" and outcome.peak_gb > 0.15
    assert outcome.seconds < 25


def test_group_pids_lists_the_leader_and_its_children(tmp_path: Path) -> None:
    code = "import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(5)']); time.sleep(5)"
    proc = subprocess.Popen(_python(code), start_new_session=True)
    try:
        deadline = time.monotonic() + 10
        while len(group_pids(proc.pid)) < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert proc.pid in group_pids(proc.pid) and len(group_pids(proc.pid)) >= 2
    finally:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def test_a_worker_that_outlives_its_leader_is_killed_with_the_stage(tmp_path: Path) -> None:
    pidfile = tmp_path / "worker.pid"
    code = (
        "import subprocess, sys\n"
        "w = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(pidfile)!r}, 'w').write(str(w.pid))\n"
    )
    outcome = _guard(tmp_path, _budget(8.0)).run(_python(code), tmp_path / "log")
    assert outcome.status == "pass", "the leader exited cleanly and left a worker behind"
    pid = int(pidfile.read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    os.kill(pid, signal.SIGKILL)
    pytest.fail("the worker outlived the stage")


def _runner(tmp_path: Path, budget: MachineBudget) -> StageRunner:
    return StageRunner(_guard(tmp_path, budget), tmp_path / "logs", tmp_path / "tmp")


def test_a_machine_short_kill_is_not_answered_by_running_less(tmp_path: Path) -> None:
    """Someone else's build dipped the floor; the family then read ``pass`` with heavy never run."""
    stage = _Lever(name="own-tests", command=_python("import time; time.sleep(30)"))
    result = _runner(tmp_path, _budget(40.0, floor_gb=10_000.0)).run(stage, "alpha")
    assert result.status == "machine-short" and result.failed
    assert not result.notes, "nothing was deselected"
    assert any("reran it unreduced" in event for event in result.events)
    assert (tmp_path / "logs" / "alpha" / "own-tests-again.log").exists()
    assert not (tmp_path / "logs" / "alpha" / "own-tests-retry.log").exists()


def test_a_machine_short_kill_that_clears_is_a_full_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    flag = tmp_path / "started"
    code = (
        f"import os, time\nexisted = os.path.exists({str(flag)!r})\n"
        f"open({str(flag)!r}, 'a').write('x')\ntime.sleep(0 if existed else 30)\n"
    )
    guard = _guard(tmp_path, _budget(40.0))
    killed = []

    def machine_dips_once(peak: float, elapsed: float) -> str:
        if not killed and flag.exists():
            killed.append(True)
            return "machine-short"
        return ""

    monkeypatch.setattr(guard, "_verdict", machine_dips_once)
    runner = StageRunner(guard, tmp_path / "logs", tmp_path / "tmp")
    result = runner.run(CommandStage(name="own-tests", command=_python(code)), "alpha")
    assert result.status == "pass" and not result.notes and not result.reduced
    assert result.events and "reran it unreduced: pass" in result.events[0]


def test_logs_are_kept_per_group_so_one_family_does_not_overwrite_another(tmp_path: Path) -> None:
    runner = _runner(tmp_path, _budget(8.0))
    stage = CommandStage(name="build", command=_python("print('mine')"))
    runner.run(stage, "alpha")
    runner.run(dataclasses.replace(stage, command=_python("print('theirs')")), "beta")
    assert "mine" in (tmp_path / "logs" / "alpha" / "build.log").read_text()
    assert "theirs" in (tmp_path / "logs" / "beta" / "build.log").read_text()


def test_a_terminated_run_takes_its_stage_with_it(tmp_path: Path) -> None:
    """``kill`` and a closed terminal skipped the ``finally`` that kills the stage."""
    pidfile = tmp_path / "stage.pid"
    (tmp_path / "child.py").write_text(
        "import os, sys, time\nopen(sys.argv[1], 'w').write(str(os.getpid()))\ntime.sleep(60)\n"
    )
    (tmp_path / "driver.py").write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "from lucid.test.audit._zoo_budget import MachineBudget\n"
        "from lucid.test.audit._zoo_guard import StageGuard, exit_on_terminate\n"
        "tmp = Path(sys.argv[1])\n"
        "guard = StageGuard(tmp, MachineBudget(64.0, 40.0, 0.0, 0.0), poll_s=0.05)\n"
        "with exit_on_terminate():\n"
        "    guard.run((sys.executable, str(tmp / 'child.py'), str(tmp / 'stage.pid')), tmp / 'log')\n"
    )
    proc = subprocess.Popen(
        [sys.executable, str(tmp_path / "driver.py"), str(tmp_path)],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    try:
        deadline = time.monotonic() + 60
        while not pidfile.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert pidfile.exists(), "the stage never started"
        child = int(pidfile.read_text())
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=30) == 128 + signal.SIGTERM
    finally:
        if proc.poll() is None:
            proc.kill()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    os.kill(child, signal.SIGKILL)
    pytest.fail("the stage outlived the run that was terminated")


def test_terminate_handlers_are_restored_after_the_run() -> None:
    before = signal.getsignal(signal.SIGTERM)
    with exit_on_terminate():
        assert signal.getsignal(signal.SIGTERM) is not before
    assert signal.getsignal(signal.SIGTERM) is before


def test_two_runs_on_one_commit_do_not_share_a_state_directory(tmp_path: Path) -> None:
    first = ZooState(tmp_path, SHA, dirty=False, resume=True)
    second = ZooState(tmp_path, SHA, dirty=False, resume=True)
    with first.lock():
        with pytest.raises(ZooBusy, match="another zoo run"):
            with second.lock():
                pytest.fail("two runs held one directory")
    with second.lock():
        pass


# ── waiting for memory is shown, and a wait that gave up is on the record ────


def test_a_wait_that_runs_out_is_put_on_the_record_and_shown(tmp_path: Path) -> None:
    seen: list[tuple[float, float, float]] = []
    guard = StageGuard(
        tmp_path,
        _budget(40.0, start_room_gb=10_000.0),
        poll_s=0.05,
        room_wait_s=0.3,
        room_poll_s=0.05,
        on_wait=lambda *args: seen.append(args),
    )
    result = StageRunner(guard, tmp_path / "logs", tmp_path / "tmp").run(
        CommandStage(name="build", command=_python("pass")), "alpha"
    )
    assert result.status == "pass"
    assert seen and all(needed == 10_000.0 for _, _, needed in seen)
    assert any("waited" in event and "10000 GB" in event for event in result.events)
    assert "waited" in (tmp_path / "logs" / "alpha" / "build.log").read_text()


def test_the_display_says_it_is_waiting_not_that_the_stage_uses_0_gb() -> None:
    reporter = LineReporter(Console(colour=False), 28.0)
    reporter.start(1)
    reporter.family(1, "alpha")
    reporter.stage("build")
    reporter.wait(65.0, 3.2, 14.0)
    status = reporter.status()
    assert "waiting for memory" in status and "3.2 GB free" in status and "14 GB needed" in status
    assert "footprint" not in status
    reporter.tick(1.0, 0.4)
    assert "footprint 0.4 GB" in reporter.status()


# ── the matrix probe does not call a skipped pair "ok" ───────────────────────


def _matrix(monkeypatch: pytest.MonkeyPatch, records: list[dict[str, object]]) -> None:
    from lucid.test.unit.compile import _zoo_matrix as matrix

    pending = iter(records)
    monkeypatch.setattr(matrix, "TASKS", ("world-modeling",))
    monkeypatch.setattr(matrix, "families", lambda task: ["genie", "diamond"])
    monkeypatch.setattr(matrix, "run_isolated", lambda label, task: next(pending))
    monkeypatch.setattr(matrix, "verdict", lambda rec: None)


def test_a_skipped_matrix_pair_is_reported_as_skipped_and_ran_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _matrix(monkeypatch, [{"status": "skip", "detail": "needs the hub"}])
    assert _zoo_probes.matrix_family("genie") == 5, "nothing ran, which is not a pass"
    out = capsys.readouterr().out
    assert "SKIPPED [1] world-modeling genie: needs the hub" in out
    assert "ok world-modeling" not in out and "0 pairs run, 1 skipped" in out
    log = tmp_path / "log"
    log.write_text(out)
    result = read_result(CommandStage(name="compiled-training", command=("x",)), log, "none", 1, 0)
    assert result.skips and not result.reduced, "a skip with a reason is a designed skip"


def test_a_matrix_with_one_pair_that_ran_is_a_pass_with_the_skip_noted(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _matrix(monkeypatch, [{"status": "ok"}])
    assert _zoo_probes.matrix_family("genie") == 0
    assert "ok world-modeling genie" in capsys.readouterr().out


def test_a_family_with_no_pair_at_all_says_so(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _matrix(monkeypatch, [])
    assert _zoo_probes.matrix_family("lenet") == 5
    assert "SKIPPED [1] lenet: no compiled-training pair is declared" in capsys.readouterr().out


def test_a_family_whose_every_factory_is_past_the_budget_is_not_a_pass(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _zoo_probes.build_family("alexnet", budget_gb=0.00001) == 5
    out = capsys.readouterr().out
    assert "ADAPTIVE alexnet/alexnet_cls" in out and "0 built, 2 past budget" in out


def test_the_build_budget_has_one_rule() -> None:
    assert _zoo_probes.fits_build is fits_build
    assert MachineBudget.for_memory(36).fits_build(5_000_000_000)


# ── a family's tests are the ones its factories name ─────────────────────────


def test_a_family_is_selected_by_its_factory_names_and_word_safely() -> None:
    senet = FamilySelector(["se_resnet_50", "se_resnet_50_cls"], ["senet"])
    assert senet.matches("lucid/test/unit/models/test_x.py::test_y[se_resnet_50]")
    assert senet.matches("lucid/test/unit/models/test_x.py::test_y[se_resnet_50_cls-cpu]")
    assert senet.matches("lucid/test/unit/models/test_x.py::test_y[senet]")
    # `-k senet` matched every densenet test; a name is not found inside a longer one.
    assert not senet.matches("lucid/test/unit/models/test_x.py::test_y[densenet_121]")
    assert not senet.matches("lucid/test/unit/models/test_x.py::test_y[resnet_50]")
    resnet = FamilySelector(["resnet_50"], ["resnet"])
    assert resnet.matches("t.py::test_y[resnet_50]")
    assert not resnet.matches("t.py::test_y[se_resnet_50]"), "_ is part of a word"
    assert resnet.matches("t.py::test_resnet_forward"), "_ separates words in a function name"


def test_the_real_registry_gives_senet_its_se_resnets_and_not_densenets() -> None:
    names, labels = ZooCatalog(ROOT).selection_for("senet")
    assert "se_resnet_50" in names and labels == ("senet",)
    selector = FamilySelector(names, labels)
    assert selector.matches("t.py::test[se_resnet_50]")
    assert not selector.matches("t.py::test[densenet_121]")
    for family in ("sknet", "cspnet"):
        found, kept = ZooCatalog(ROOT).selection_for(family)
        assert found and family in kept


def test_a_family_with_no_registered_factory_still_selects_by_its_directory_name() -> None:
    names, labels = ZooCatalog(ROOT).selection_for("no_such_family")
    assert names == () and labels == ("no_such_family",)


def test_the_select_plugin_deselects_everything_else(tmp_path: Path) -> None:
    (tmp_path / "test_sel.py").write_text(
        "import pytest\n"
        "@pytest.mark.parametrize('name', ['se_resnet_50', 'densenet_121', 'resnet_50'])\n"
        "def test_model(name):\n    pass\n"
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-p",
            "lucid.test.audit._zoo_select",
            "--zoo-names=se_resnet_50",
            "--zoo-labels=senet",
            str(tmp_path / "test_sel.py"),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert "1 passed, 2 deselected" in proc.stdout, proc.stdout + proc.stderr


# ── the tree a record is filed under is the tree that was verified ───────────


def test_a_tree_that_moves_mid_run_stops_the_run_and_files_nothing_late(tmp_path: Path) -> None:
    calls = iter([(SHA, False), (SHA, False), (SHA, False), ("b" * 40, False)])
    console = Console(colour=False)
    run = _zoo.ZooRun(
        tmp_path,
        console,
        _zoo.ZooSettings(state_root=tmp_path / "state", poll_s=0.05),
        budget=_budget(40.0),
        reporter=LineReporter(console, 40.0),
        catalog=_Catalog(tmp_path, tmp_path / "counter"),
        revision=lambda: next(calls),
    )
    with pytest.raises(_zoo.TreeMoved, match="scratch clone"):
        run.run()
    state = tmp_path / "state" / SHA
    assert (state / "alpha.json").exists(), "what finished before the move stands"
    assert not (state / "beta.json").exists(), "what finished after it is not filed under SHA"


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def test_repo_state_counts_an_untracked_test_file_as_a_change(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "a@b.c")
    _git(tmp_path, "config", "user.name", "x")
    (tmp_path / "a.py").write_text("x = 1\n")
    _git(tmp_path, "add", "a.py")
    _git(tmp_path, "commit", "-q", "-m", "one")
    sha, dirty = repo_state(tmp_path)
    assert len(sha) == 40 and not dirty
    (tmp_path / "test_new.py").write_text("def test_x(): pass\n")
    assert repo_state(tmp_path) == (sha, True), "the catalog would read this file"
    (tmp_path / ".gitignore").write_text("build/\n")
    (tmp_path / "test_new.py").unlink()
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "engine.so").write_text("x")
    _git(tmp_path, "add", ".gitignore")
    _git(tmp_path, "commit", "-q", "-m", "two")
    assert not repo_state(tmp_path)[1], "ignored build products are not changes"


# ── what is written down ─────────────────────────────────────────────────────


def test_a_resumed_family_still_gets_its_note_and_the_ledger_is_rewritten_once(
    tmp_path: Path,
) -> None:
    runs = _Runs(tmp_path)
    runs.go(40.0)  # no notes
    assert not (tmp_path / "notes").exists()
    report = runs.go(40.0, notes=True)
    assert report.resumed == 3 and report.notes_written == 3
    notes = sorted(p.name for p in (tmp_path / "notes").iterdir())
    assert notes == [
        "zoo-verify-alpha.md",
        "zoo-verify-beta.md",
        "zoo-verify-ledger.md",
        "zoo-verify-zoo-wide.md",
    ]
    assert "진행: 3 / 3" in (tmp_path / "notes" / "zoo-verify-ledger.md").read_text()
    again = runs.go(40.0, notes=True)
    assert again.notes_written == 0, "the notes were already there"


def test_a_run_that_writes_no_notes_says_so(capsys: pytest.CaptureFixture[str]) -> None:
    console = Console(colour=False)
    _zoo.summarise(console, _zoo.ZooReport(groups=1), MachineBudget.for_memory(36), None)
    out = capsys.readouterr().out
    assert "no notes written" in out and "--zoo-record" in out
    _zoo.summarise(console, _zoo.ZooReport(groups=1), MachineBudget.for_memory(36), Path("x"))
    assert "no notes written" not in capsys.readouterr().out


def test_the_summary_names_groups_that_passed_in_reduced_form(
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = _zoo.ZooReport()
    report.add("lenet", [_r(notes=["heavy tests deselected up front: ..."])], resumed=False)
    _zoo.summarise(Console(colour=False), report, MachineBudget.for_memory(8), Path("x"))
    out = capsys.readouterr().out
    assert "1 group(s) passed in reduced form" in out and "lenet" in out
    assert "lenet/s" in out, "and what was left out is listed with it"


def test_notes_and_state_are_written_whole_or_not_at_all(tmp_path: Path) -> None:
    target = tmp_path / "note.md"
    write_atomic(target, "one")
    write_atomic(target, "two")
    assert target.read_text() == "two"
    assert [p.name for p in tmp_path.iterdir()] == ["note.md"], "no scratch file is left behind"
    state = ZooState(tmp_path / "s", SHA, dirty=False, resume=True)
    state.save("alpha", [_r()], _budget(8.0))
    assert json.loads((state.root / "alpha.json").read_text())["limit_gb"] == 8.0
    assert not list(state.root.glob("*.tmp"))
    book = Notebook(tmp_path / "n", state, {"alpha": "vision"}, RunFacts(SHA, "c", _budget(8.0)))
    book.write("alpha", [_r()])
    assert not list((tmp_path / "n").glob("*.tmp"))


# ── the options are checked before hours are spent ───────────────────────────


def _zoo_args(**given: object) -> argparse.Namespace:
    parsed = build_parser().parse_args(["--models"])
    for key, value in given.items():
        setattr(parsed, key, value)
    return parsed


def test_zoo_options_are_validated_up_front(tmp_path: Path) -> None:
    _zoo.validate_options(ROOT, _zoo_args())
    for bad in (0, -4):
        with pytest.raises(PlanError, match="--mem-limit-gb must be positive"):
            _zoo.validate_options(ROOT, _zoo_args(mem_limit_gb=bad))
    afile = tmp_path / "afile"
    afile.write_text("x")
    with pytest.raises(PlanError, match="is a file"):
        _zoo.validate_options(ROOT, _zoo_args(zoo_record=afile))
    if os.geteuid() != 0:
        with pytest.raises(PlanError, match="cannot be created"):
            _zoo.validate_options(ROOT, _zoo_args(zoo_record=Path("/no-such-root/notes")))
    _zoo.validate_options(ROOT, _zoo_args(zoo_record=tmp_path / "new" / "deeper"))


def test_a_bad_zoo_option_exits_2_before_any_stage(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--models", "--mem-limit-gb", "-1", "--no-log"]) == 2
    assert "positive" in capsys.readouterr().out


def test_a_limit_larger_than_the_machine_does_not_wait_for_memory_that_does_not_exist() -> None:
    budget = MachineBudget.for_memory(36, limit_gb=100.0)
    assert budget.start_room_gb <= 14.0, "it would wait the full half hour before every stage"
    assert budget.build_budget_gb <= 36 * 0.8


def test_the_regenerate_hints_no_longer_mean_hours(tmp_path: Path) -> None:
    from lucid.test.audit._doctests import DoctestResult, save_floor
    from lucid.test.audit._result import Report
    from lucid.test.audit.__main__ import save_coverage

    save_coverage(tmp_path / "c.json", Report())
    save_floor(tmp_path / "d.json", DoctestResult(0, 0, {}, ran=True))
    assert "lucid-audit --audit-only --update-coverage" in (tmp_path / "c.json").read_text()
    assert "lucid-audit --doctests-only --update-doctests" in (tmp_path / "d.json").read_text()


def test_no_comment_names_a_flag_that_was_removed() -> None:
    for name in ("_surface.py", "test_audit_smoke.py", "__main__.py", "__init__.py"):
        text = (ROOT / "lucid" / "test" / "audit" / name).read_text()
        assert "--list-subsystems" not in text and "--subsystem" not in text, name
