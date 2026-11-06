"""The model-zoo stage: the machine budget, the guard, resuming, and what it writes.

None of this builds a model.  The stage's moving parts are a memory budget, a
guard that kills a process, a retry, a per-commit record and a display; each is
exercised here with a process that is a few lines of ``python -c`` — the real
families are the Studio's job.
"""

import argparse
import dataclasses
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import override

import pytest

from lucid.test.audit import _zoo, _zoo_report
from lucid.test.audit._console import Console
from lucid.test.audit._zoo_budget import (
    HEAVY_HEADROOM,
    HEAVY_PEAK_GB,
    MachineBudget,
    available_gb,
    footprint_gb,
)
from lucid.test.audit._zoo_guard import StageGuard, StageRunner
from lucid.test.audit._zoo_record import Notebook, RunFacts, ZooState, render_ledger, render_note
from lucid.test.audit._zoo_report import LineReporter
from lucid.test.audit._zoo_stage import (
    ZOO_WIDE,
    CommandStage,
    PytestStage,
    Stage,
    StageResult,
    ZooCatalog,
    read_result,
)

ROOT = Path(__file__).resolve().parents[3]
SHA = "a" * 40


# ── MachineBudget ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("gb", "limit", "start", "floor", "build"),
    [
        (8, 6.0, 3.0, 1.0, 4.8),
        (16, 12.5, 6.5, 1.0, 10.0),
        (32, 25.0, 12.5, 2.0, 20.0),
        (36, 28.0, 14.0, 2.5, 22.4),  # the Studio: the values the script was tuned on
        (128, 99.5, 50.0, 9.0, 79.6),
    ],
)
def test_the_budget_scales_the_studios_numbers_to_the_machine(
    gb: float, limit: float, start: float, floor: float, build: float
) -> None:
    budget = MachineBudget.for_memory(gb)
    assert (budget.limit_gb, budget.start_room_gb, budget.floor_gb) == (limit, start, floor)
    assert budget.build_budget_gb == pytest.approx(build)


def test_the_studio_keeps_exactly_its_old_values() -> None:
    budget = MachineBudget.for_memory(36)
    assert budget.limit_gb == 28.0
    assert budget.start_room_gb == 14.0
    assert budget.floor_gb == 2.5
    assert budget.build_budget_gb == pytest.approx(0.8 * 28.0)


def test_a_memory_limit_override_moves_the_room_to_start_and_not_the_floor() -> None:
    budget = MachineBudget.for_memory(36, limit_gb=10.0)
    assert (budget.limit_gb, budget.start_room_gb, budget.floor_gb) == (10.0, 5.0, 2.5)
    assert budget.build_budget_gb == pytest.approx(8.0)


@pytest.mark.parametrize("bad", [(0, None), (-8, None), (16, 0), (16, -1)])
def test_a_budget_needs_positive_sizes(bad: tuple[float, float | None]) -> None:
    with pytest.raises(ValueError, match="positive"):
        MachineBudget.for_memory(bad[0], bad[1])


def test_a_build_past_the_budget_is_refused_by_declared_size() -> None:
    budget = MachineBudget.for_memory(36)
    assert budget.fits_build(5_000_000_000)  # 18.6 GiB of float32
    assert not budget.fits_build(6_500_000_000)  # 24.2 GiB, past 22.4


def test_heavy_tests_are_offered_only_to_machines_with_room_for_them() -> None:
    assert not MachineBudget.for_memory(8).admits_heavy
    assert MachineBudget.for_memory(16).admits_heavy
    assert MachineBudget.for_memory(36).admits_heavy
    assert MachineBudget.for_memory(36, limit_gb=HEAVY_PEAK_GB * HEAVY_HEADROOM).admits_heavy
    assert not MachineBudget.for_memory(36, limit_gb=HEAVY_PEAK_GB).admits_heavy


def test_the_machine_can_be_read() -> None:
    assert MachineBudget.detect().total_gb > 0
    assert available_gb() > 0
    assert footprint_gb(os.getpid()) > 0.01, "a process can read its own footprint"


# ── a stage and its result ───────────────────────────────────────────────────


def test_a_stage_log_is_read_into_a_result(tmp_path: Path) -> None:
    log = tmp_path / "stage.log"
    log.write_text(
        "built lenet: 61,706 params, declared 61706, 0.1s\n"
        "ADAPTIVE vit/vit_huge_14: 632,000,000 params need 2.4 GB, past this machine's budget\n"
        "CHECK foo: built 5, declared 6\n"
        "SKIPPED [2] t.py:3: no network\n"
        "FAILED t.py::test_a - boom\n"
        "===== 1 failed, 3 passed in 1.23s =====\n"
    )
    stage = CommandStage(name="build", command=("x",), notes=("decided before it started",))
    result = read_result(stage, log, "fail", 4.26, 1.234)
    assert (result.status, result.seconds, result.peak_gb, result.failed) == (
        "fail",
        4.3,
        1.23,
        True,
    )
    assert result.summary == "1 failed, 3 passed"
    assert result.failures == ["t.py::test_a - boom"]
    assert result.skips == ["2 × t.py:3: no network"]
    assert result.checks == ["foo: built 5, declared 6"]
    assert result.notes == [
        "decided before it started",
        "vit/vit_huge_14: 632,000,000 params need 2.4 GB, past this machine's budget",
    ]


def test_a_stage_that_died_without_saying_why_still_has_a_failure(tmp_path: Path) -> None:
    log = tmp_path / "stage.log"
    log.write_text("one\ntwo\nthree\nfour\n")
    result = read_result(CommandStage(name="x", command=("x",)), log, "mem-guard", 1.0, 30.0)
    assert result.failed
    assert result.failures == [
        "killed by the memory guard at 30.0 GB after 1s; the log ends: two / three / four"
    ]
    plain = read_result(CommandStage(name="x", command=("x",)), log, "fail", 1.0, 0.1)
    assert plain.failures == ["two / three / four"]


def test_a_stage_with_no_tests_is_not_a_failure(tmp_path: Path) -> None:
    log = tmp_path / "stage.log"
    log.write_text("no tests ran\n")
    assert not read_result(CommandStage(name="x", command=("x",)), log, "none", 0.1, 0.1).failed


def _marker(argv: tuple[str, ...]) -> str:
    """The marker expression: the ``-m`` after ``python -m pytest``'s own."""
    return argv[argv.index("-m", 3) + 1]


def test_a_pytest_stage_clears_the_default_markers_and_can_drop_heavy(tmp_path: Path) -> None:
    stage = PytestStage(name="own-tests", files=("a.py",), basetemp=tmp_path, marker="")
    assert _marker(stage.argv()) == ""
    light = stage.without_heavy("why")
    assert light is not None and light.notes == ("why",)
    assert _marker(light.argv()) == "not heavy"
    assert light.without_heavy("again") is None, "one retry, not a loop"

    zoo = PytestStage(
        name="zoo-tests",
        files=("m.py",),
        basetemp=tmp_path,
        marker="not control",
        names=("vit_base_16", "vit_large_16"),
        labels=("vit",),
    )
    dropped = zoo.without_heavy("why")
    assert dropped is not None
    argv = dropped.argv()
    assert _marker(argv) == "(not control) and not heavy"
    assert "--zoo-names=vit_base_16,vit_large_16" in argv and "--zoo-labels=vit" in argv
    assert argv[argv.index("lucid.test.audit._zoo_select") - 1] == "-p"
    assert CommandStage(name="build", command=("x",)).without_heavy("why") is None


# ── the guard ────────────────────────────────────────────────────────────────


def _allocate(megabytes: int, seconds: float = 30.0) -> tuple[str, ...]:
    code = f"import time; x = b'x' * ({megabytes} * 2**20); time.sleep({seconds})"
    return (sys.executable, "-c", code)


def _budget(limit_gb: float) -> MachineBudget:
    """A budget whose machine-wide checks cannot fire, so only the footprint limit can."""
    return MachineBudget(total_gb=64.0, limit_gb=limit_gb, start_room_gb=0.0, floor_gb=0.0)


def _guard(tmp_path: Path, limit_gb: float, timeout_s: float = 60.0) -> StageGuard:
    return StageGuard(tmp_path, _budget(limit_gb), poll_s=0.05, timeout_s=timeout_s)


def test_the_guard_kills_a_stage_that_passes_the_footprint_limit(tmp_path: Path) -> None:
    outcome = _guard(tmp_path, 0.15).run(_allocate(300), tmp_path / "log")
    assert outcome.status == "mem-guard"
    assert outcome.peak_gb > 0.15
    assert outcome.seconds < 25, "it waited out the child's sleep instead of killing it"


@pytest.mark.parametrize(
    ("code", "status"),
    [("pass", "pass"), ("raise SystemExit(1)", "fail"), ("raise SystemExit(5)", "none")],
)
def test_the_guard_reports_how_an_ordinary_process_ended(
    tmp_path: Path, code: str, status: str
) -> None:
    outcome = _guard(tmp_path, 8.0).run((sys.executable, "-c", code), tmp_path / "log")
    assert outcome.status == status


def test_a_stage_shorter_than_the_poll_interval_still_has_a_footprint(tmp_path: Path) -> None:
    """A one-second build used to report a peak of 0.0 GB: it ended before the first look."""
    guard = StageGuard(tmp_path, _budget(8.0), poll_s=2.0)
    outcome = guard.run(_allocate(120, seconds=1.0), tmp_path / "log")
    assert outcome.status == "pass"
    assert outcome.peak_gb > 0.05, "a stage that ends inside its first poll was never measured"


def test_the_guard_times_out_a_stage_that_never_ends(tmp_path: Path) -> None:
    outcome = _guard(tmp_path, 8.0, timeout_s=0.3).run(_allocate(1), tmp_path / "log")
    assert outcome.status == "timeout"


def test_the_guard_ticks_so_a_display_can_show_the_footprint(tmp_path: Path) -> None:
    ticks: list[tuple[float, float]] = []
    guard = StageGuard(
        tmp_path, _budget(8.0), poll_s=0.05, on_tick=lambda s, gb: ticks.append((s, gb))
    )
    guard.run(_allocate(50, seconds=0.4), tmp_path / "log")
    assert ticks and all(gb >= 0 for _, gb in ticks)


@dataclass(frozen=True, kw_only=True)
class _Heavy(Stage):
    """A stage that is too big until its heavy part is dropped."""

    heavy: bool = True
    retry_fits: bool = True

    @override
    def argv(self) -> tuple[str, ...]:
        return _allocate(300) if self.heavy else _allocate(1, seconds=0.0)

    @override
    def without_heavy(self, why: str) -> Stage | None:
        if not self.heavy:
            return None
        return dataclasses.replace(self, heavy=not self.retry_fits, notes=(*self.notes, why))


def _runner(tmp_path: Path, limit_gb: float = 0.15) -> StageRunner:
    logs = tmp_path / "logs"
    logs.mkdir()
    return StageRunner(_guard(tmp_path, limit_gb), logs, tmp_path / "tmp")


def test_a_killed_stage_is_retried_once_without_heavy_and_says_so(tmp_path: Path) -> None:
    result = _runner(tmp_path).run(_Heavy(name="own-tests"))
    assert result.status == "pass" and not result.failed
    assert any("memory guard killed the first attempt" in note for note in result.notes)
    assert result.notes[-1] == "retry with heavy deselected: pass"
    assert (tmp_path / "logs" / "own-tests.log").exists()
    assert (tmp_path / "logs" / "own-tests-retry.log").exists()


def test_a_stage_still_killed_on_its_retry_is_a_failure(tmp_path: Path) -> None:
    result = _runner(tmp_path).run(_Heavy(name="own-tests", retry_fits=False))
    assert result.status == "mem-guard" and result.failed
    assert result.notes[-1] == "retry with heavy deselected: mem-guard"


def test_a_stage_with_nothing_to_drop_is_a_failure_when_killed(tmp_path: Path) -> None:
    stage = CommandStage(name="build", command=_allocate(300))
    result = _runner(tmp_path).run(stage)
    assert result.status == "mem-guard" and result.failed
    assert not any("retry" in note for note in result.notes)


def test_the_scratch_directory_is_removed_after_each_stage(tmp_path: Path) -> None:
    runner = _runner(tmp_path, limit_gb=8.0)
    (tmp_path / "tmp").mkdir()
    (tmp_path / "tmp" / "leftover").write_text("x")
    runner.run(CommandStage(name="quick", command=(sys.executable, "-c", "pass")))
    assert not (tmp_path / "tmp").exists()


def test_the_report_counts_a_killed_stage_unless_its_retry_passed() -> None:
    killed = StageResult("a", "mem-guard", 1.0, 30.0)
    rescued = StageResult("b", "pass", 2.0, 30.0, notes=["retry with heavy deselected: pass"])
    report = _zoo.ZooReport()
    report.add("fam", [killed, rescued], resumed=False)
    assert report.failures == 1 and report.failed[0][1] is killed
    assert any("retry" in note for _, _, note in report.adapted)


# ── resuming, and a run through the stage ────────────────────────────────────


class _FakeCatalog(ZooCatalog):
    """Two families and a zoo-wide group, each stage a few lines of Python."""

    def __init__(self, root: Path, counter: Path, failing: tuple[str, ...] = ()) -> None:
        super().__init__(root)
        self.counter = counter
        self.failing = failing

    @override
    def families(self) -> dict[str, str]:
        return {"alpha": "vision", "beta": "text"}

    def _stage(self, name: str, group: str) -> CommandStage:
        code = (
            f"open({str(self.counter)!r}, 'a').write({group!r} + '/' + {name!r} + '\\n');"
            "print('1 passed in 0.01s')"
        )
        if group in self.failing:
            code += ";print('FAILED t.py::test_x - boom');raise SystemExit(1)"
        return CommandStage(name=name, command=(sys.executable, "-c", code))

    @override
    def family_stages(self, family: str, temp: Path, budget: MachineBudget) -> list[Stage]:
        return [self._stage("contract", family), self._stage("build", family)]

    @override
    def wide_stages(self, temp: Path) -> list[Stage]:
        return [self._stage("tests/test_wide.py", ZOO_WIDE)]


@dataclass
class _Run:
    root: Path
    failing: tuple[str, ...] = ()

    @property
    def counter(self) -> Path:
        return self.root / "counter"

    def runs(self) -> list[str]:
        return self.counter.read_text().split() if self.counter.exists() else []

    def go(
        self,
        *,
        sha: str = SHA,
        dirty: bool = False,
        families: tuple[str, ...] = (),
        limit: int = 0,
        fresh: bool = False,
        notes: bool = True,
        device: str = "metal",
    ) -> _zoo.ZooReport:
        console = Console(colour=False)
        base = _zoo.ZooSettings(
            state_root=self.root / "state", poll_s=0.05, record_dir=self.root / "notes"
        )
        run = _zoo.ZooRun(
            self.root,
            console,
            dataclasses.replace(
                base,
                families=families,
                limit=limit,
                fresh=fresh,
                record_dir=base.record_dir if notes else None,
                device=device,
            ),
            budget=_budget(40.0),
            reporter=LineReporter(console, 40.0),
            catalog=_FakeCatalog(self.root, self.counter, self.failing),
            revision=lambda: (sha, dirty),
        )
        return run.run()


def test_every_group_runs_once_and_the_stage_never_stops_on_a_failure(tmp_path: Path) -> None:
    runs = _Run(tmp_path, failing=("alpha",))
    report = runs.go()
    assert report.groups == 3 and report.resumed == 0
    assert report.failures == 2, "both of alpha's stages failed, and beta still ran"
    assert sorted(set(runs.runs())) == [
        "alpha/build",
        "alpha/contract",
        "beta/build",
        "beta/contract",
        "zoo-wide/tests/test_wide.py",
    ]


def test_a_family_recorded_for_this_commit_is_skipped(tmp_path: Path) -> None:
    runs = _Run(tmp_path)
    runs.go()
    first = len(runs.runs())
    second = runs.go()
    assert second.resumed == 3 and len(runs.runs()) == first


def test_an_interrupted_run_resumes_where_it_stopped(tmp_path: Path) -> None:
    runs = _Run(tmp_path)
    runs.go(families=("alpha",))
    assert {r.split("/")[0] for r in runs.runs()} == {"alpha"}
    report = runs.go()
    assert report.resumed == 1, "alpha was recorded; beta and zoo-wide were not"
    assert {r.split("/")[0] for r in runs.runs()} == {"alpha", "beta", "zoo-wide"}


def test_fresh_ignores_the_records(tmp_path: Path) -> None:
    runs = _Run(tmp_path)
    runs.go()
    first = len(runs.runs())
    assert runs.go(fresh=True).resumed == 0
    assert len(runs.runs()) == 2 * first


def test_another_commit_does_not_reuse_the_records(tmp_path: Path) -> None:
    runs = _Run(tmp_path)
    runs.go()
    assert runs.go(sha="b" * 40).resumed == 0


def test_a_dirty_tree_never_reuses_a_record_and_keeps_its_own(tmp_path: Path) -> None:
    runs = _Run(tmp_path)
    runs.go()
    first = len(runs.runs())
    assert runs.go(dirty=True).resumed == 0
    assert len(runs.runs()) == 2 * first
    assert (tmp_path / "state" / SHA).is_dir() and (tmp_path / "state" / f"{SHA}-dirty").is_dir()
    assert runs.go().resumed == 3, "the clean record is still the clean record"


def test_the_state_is_per_commit_and_per_family(tmp_path: Path) -> None:
    state = ZooState(tmp_path, SHA, dirty=False, resume=True)
    results = [StageResult("build", "pass", 1.0, 2.0, notes=["n"])]
    state.save("alpha", results, _budget(40.0), "cpu")
    on_cpu = [dataclasses.replace(r, device="cpu") for r in results]
    assert state.load("alpha", _budget(40.0), "cpu") == on_cpu, "the record says where it ran"
    assert state.load("beta", _budget(40.0), "cpu") is None
    assert (
        ZooState(tmp_path, SHA, dirty=False, resume=False).load("alpha", _budget(40.0), "cpu")
        is None
    )
    assert (
        ZooState(tmp_path, SHA, dirty=True, resume=True).load("alpha", _budget(40.0), "cpu") is None
    )
    (tmp_path / SHA / "alpha.cpu.json").write_text("{ not json")
    assert (
        state.load("alpha", _budget(40.0), "cpu") is None
    ), "a record this version cannot read is rerun"


def test_a_full_run_leaves_a_done_marker_and_a_narrowed_one_does_not(tmp_path: Path) -> None:
    runs = _Run(tmp_path)
    runs.go(families=("alpha",))
    assert not (tmp_path / "state" / SHA / "DONE").exists()
    runs.go(device="metal")
    assert not (tmp_path / "state" / SHA / "DONE").exists(), "one device is not the whole zoo"
    runs.go(device="both")
    assert (tmp_path / "state" / SHA / "DONE").read_text().strip() == SHA


def test_limit_takes_the_first_groups(tmp_path: Path) -> None:
    runs = _Run(tmp_path)
    assert runs.go(limit=1).groups == 1
    assert {r.split("/")[0] for r in runs.runs()} == {"alpha"}


# ── what is written down ─────────────────────────────────────────────────────


def _facts() -> RunFacts:
    return RunFacts(SHA, "lucid-audit --models --family alpha", MachineBudget.for_memory(36))


def test_a_note_keeps_its_format_and_names_the_command_and_the_budget() -> None:
    results = [
        StageResult("contract", "pass", 3.0, 1.0, "1 passed"),
        StageResult(
            "own-tests",
            "pass",
            60.0,
            9.5,
            "2 passed",
            skips=["1 × t.py:3: no network"],
            notes=["heavy tests deselected up front: small machine"],
        ),
    ]
    note = render_note("alpha", "vision", results, _facts())
    assert note.startswith("# zoo-verify: alpha\n")
    assert f"`{SHA[:9]}`" in note and "vision" in note and "최대 9.5 GB" in note
    assert "명령: `lucid-audit --models --family alpha`" in note
    assert "36 GB machine" in note and "footprint limit 28 GB" in note
    assert "| 장치 | 단계 | 결과 | 시간 (s) | 최대 (GB) | 요약 |" in note
    assert "## 건너뛴 것" in note and "## 이 머신에서 정한 것" in note
    assert "heavy tests deselected up front" in note
    assert note.rstrip().endswith("계약: [[arch-models-family-contract]]")


def test_the_ledger_lists_every_family_recorded_or_waiting(tmp_path: Path) -> None:
    state = ZooState(tmp_path, SHA, dirty=False, resume=True)
    state.save("alpha", [StageResult("build", "fail", 60.0, 3.0)], _budget(40.0), "cpu")
    ledger = render_ledger(_facts(), state, {"alpha": "vision", "beta": "text"})
    assert "| family | 분야 | 결과 | metal | cpu | 분 | 최대 GB | 기록 |" in ledger
    assert "| alpha | vision | fail | 대기 | fail | 1.0 | 3.0 | [[zoo-verify-alpha]] |" in ledger
    assert "| beta | text | 대기 | 대기 | 대기 | | | |" in ledger
    assert f"`{SHA}`" in ledger and "진행: 0 / 3" in ledger, "metal is not recorded yet"


def test_a_run_writes_a_note_per_family_and_the_ledger(tmp_path: Path) -> None:
    runs = _Run(tmp_path)
    runs.go()
    notes = tmp_path / "notes"
    assert sorted(p.name for p in notes.iterdir()) == [
        "zoo-verify-alpha.md",
        "zoo-verify-beta.md",
        "zoo-verify-ledger.md",
        "zoo-verify-zoo-wide.md",
    ]
    assert "진행: 3 / 3" in (notes / "zoo-verify-ledger.md").read_text()


def test_without_a_vault_a_run_writes_no_notes(tmp_path: Path) -> None:
    runs = _Run(tmp_path)
    runs.go(notes=False)
    assert not (tmp_path / "notes").exists()


def test_the_notebook_creates_its_directory(tmp_path: Path) -> None:
    state = ZooState(tmp_path / "s", SHA, dirty=False, resume=True)
    book = Notebook(tmp_path / "deep" / "notes", state, {"alpha": "vision"}, _facts())
    state.save("alpha", [StageResult("build", "pass", 1.0, 1.0)], _budget(40.0), "metal")
    book.write("alpha")
    assert (tmp_path / "deep" / "notes" / "zoo-verify-alpha.md").exists()


def _namespace(**given: object) -> argparse.Namespace:
    base: dict[str, object] = {
        "family": [],
        "zoo_record": None,
        "fresh": False,
        "mem_limit_gb": None,
        "limit": 0,
        "zoo_device": "metal",
    }
    return argparse.Namespace(**{**base, **given})


def test_the_record_directory_defaults_to_the_vault_when_there_is_one(tmp_path: Path) -> None:
    settings = _zoo.ZooSettings.from_args(_namespace(), tmp_path, "cmd")
    assert settings.record_dir is None, "a scratch clone has no vault"
    vault = tmp_path / "obsidian" / "zoo-verify"
    vault.mkdir(parents=True)
    assert _zoo.ZooSettings.from_args(_namespace(), tmp_path, "cmd").record_dir == vault
    chosen = tmp_path / "elsewhere"
    assert (
        _zoo.ZooSettings.from_args(_namespace(zoo_record=chosen), tmp_path, "c").record_dir
        == chosen
    )


def test_the_default_state_lives_in_the_cache_not_the_checkout() -> None:
    settings = _zoo.ZooSettings.from_args(_namespace(), ROOT, "cmd")
    assert settings.state_root == Path.home() / "Library" / "Caches" / "lucid" / "audit-zoo"


# ── what this machine adapts ─────────────────────────────────────────────────


def test_a_small_machine_deselects_heavy_up_front_and_says_so(tmp_path: Path) -> None:
    catalog = ZooCatalog(ROOT)
    console = Console(colour=False, quiet=True)
    small = _zoo.ZooRun(ROOT, console, _zoo.ZooSettings(), budget=MachineBudget.for_memory(8))
    stages = small._stages("lenet", catalog, tmp_path, "metal")
    pytest_stages = [s for s in stages if isinstance(s, PytestStage)]
    assert pytest_stages and all(s.heavy_off for s in pytest_stages)
    assert all("heavy tests deselected up front on metal" in s.notes[0] for s in pytest_stages)
    assert [s.notes for s in stages if isinstance(s, CommandStage)] == [(), (), ()]
    on_cpu = small._stages("lenet", catalog, tmp_path, "cpu")
    assert all("up front on cpu" in s.notes[0] for s in on_cpu if isinstance(s, PytestStage))

    big = _zoo.ZooRun(ROOT, console, _zoo.ZooSettings(), budget=MachineBudget.for_memory(36))
    assert not any(
        s.heavy_off
        for s in big._stages("lenet", catalog, tmp_path, "metal")
        if isinstance(s, PytestStage)
    )


def test_the_catalog_finds_the_zoo_as_the_tree_has_it() -> None:
    catalog = ZooCatalog(ROOT)
    families = catalog.families()
    assert families["lenet"] == "vision" and not any(name.startswith("_") for name in families)
    assert catalog.own_files("lenet")
    stages = catalog.family_stages("lenet", Path("/tmp/x"), MachineBudget.for_memory(36))
    assert [s.name for s in stages][:3] == ["contract", "build", "compiled-training"]
    assert stages[-1].name == "zoo-tests"
    wide = catalog.wide_stages(Path("/tmp/x"))
    assert wide and all(isinstance(s, PytestStage) and s.marker == "" for s in wide)


def test_the_build_probe_skips_a_factory_past_the_budget_by_name(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from lucid.test.audit import _zoo_probes

    # Whether a factory declares ``params`` is beside the point (test_audit_zoo_budget.py).
    assert (
        _zoo_probes.build_family("alexnet", budget_gb=0.00001) == 5
    ), "nothing built is not a pass"
    out = capsys.readouterr().out
    assert "ADAPTIVE alexnet/alexnet:" in out and "past this machine's build budget" in out
    assert "0 built, 2 past budget" in out


# ── the display ──────────────────────────────────────────────────────────────


def test_the_plain_display_names_the_family_the_stage_and_the_tally(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _Run(tmp_path).go()
    out = capsys.readouterr().out
    assert "[1/3] metal alpha" in out and "[2/3] metal beta" in out
    assert "[3/3] metal zoo-wide" in out
    assert "    contract ..." in out, "a stage that is about to run says so"
    assert "alpha: pass" in out and "families ok 3 fail 0" in out and "stages ok 5 fail 0" in out
    assert "GB" in out and "min" in out


def test_a_failing_stage_turns_the_tally_red_in_the_plain_display(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _Run(tmp_path, failing=("beta",)).go()
    out = capsys.readouterr().out
    assert "beta: fail" in out and "families ok 2 fail 1" in out and "stages ok 3 fail 2" in out


def test_the_live_frame_shows_family_stage_elapsed_footprint_and_tally() -> None:
    console = Console(colour=False)
    console.live = True
    reporter = LineReporter(console, 28.0)
    reporter.start(75)
    reporter.family(3, "alexnet")
    reporter.stage("build")
    reporter.tick(75.0, 3.4)
    frame = "\n".join(reporter._frame())
    assert "family 3/75" in frame and "alexnet" in frame
    assert "build" in frame and "1:15" in frame
    assert "footprint 3.4 GB" in frame and "limit 28" in frame
    assert "families ok 0 fail 0" in frame


def test_closing_the_live_display_leaves_no_stale_frame(capsys: pytest.CaptureFixture[str]) -> None:
    console = Console(colour=False)
    console.live = True
    with LineReporter(console, 28.0) as reporter:
        reporter.start(2)
        reporter.family(1, "alpha")
        assert console._live_lines == 3
    assert console._live_lines == 0
    assert "\x1b[3A" in capsys.readouterr().out, "the region was erased, not left behind"


def test_a_stage_that_runs_long_says_it_is_still_alive(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(_zoo_report, "HEARTBEAT_S", 0.0)
    reporter = LineReporter(Console(colour=False), 28.0)
    reporter.start(2)
    reporter.family(1, "alpha")
    reporter.stage("own-tests")
    reporter.tick(400.0, 5.5)
    assert "... own-tests" in capsys.readouterr().out


def test_the_reporter_picks_the_stdlib_display_without_a_terminal() -> None:
    reporter = _zoo_report.build_reporter(Console(colour=False), 28.0)
    assert isinstance(reporter, LineReporter)


# ── the stage inside the audit ───────────────────────────────────────────────


def test_the_zoo_stage_failures_are_the_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    from lucid.test.audit.__main__ import main

    monkeypatch.setattr(_zoo, "run_zoo_stage", lambda *args: _zoo.ZooReport())
    assert main(["--models", "--no-log"]) == 0
    three = [("alpha", StageResult("s", "fail", 1.0, 1.0))] * 3
    monkeypatch.setattr(_zoo, "run_zoo_stage", lambda *args: _zoo.ZooReport(failed=three))
    assert main(["--models", "--no-log"]) == 1


def test_a_zoo_stage_that_breaks_is_a_broken_harness_not_a_failed_framework(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lucid.test.audit.__main__ import main

    def broken(*args: object) -> int:
        raise RuntimeError("cannot read the machine")

    monkeypatch.setattr(_zoo, "run_zoo_stage", broken)
    assert main(["--models", "--no-log"]) == 2


def test_models_alone_runs_the_zoo_and_nothing_else(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from lucid.test.audit.__main__ import main

    monkeypatch.setattr(_zoo, "run_zoo_stage", lambda *args: _zoo.ZooReport())
    assert main(["--models", "--no-log"]) == 0
    out = capsys.readouterr().out
    assert "model zoo failing stages" in out
    assert "audit defects" not in out and "suite failures" not in out
