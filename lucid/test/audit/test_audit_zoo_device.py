"""The zoo is verified on both devices a model runs on, by design.

The zoo is a set of deep-learning models, so ``--zoo-device`` defaults to
``both``: Metal first, where models train and the pass is quick, then the CPU.
``metal`` and ``cpu`` narrow it on purpose.  The guard hands the device to every
child, the tests and probes apply it, a stage declares which devices it is for,
the record is kept per (family, device) and is never an answer for the other
device, and the verdict is the worse of the two.  A machine with no Metal is an
error — never a quiet run on the CPU that reports itself as a Metal verification.
"""

import argparse
import dataclasses
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import override

import pytest

import lucid
from lucid.test import _default_device
from lucid.test._default_device import (
    DEFAULT_DEVICE_ENV,
    DEFAULT_DEVICES,
    apply_default_device,
    metal_available,
)
from lucid.test.audit import _zoo, _zoo_probes
from lucid.test.audit._console import Console
from lucid.test.audit._plan import PlanError
from lucid.test.audit._zoo_budget import MachineBudget
from lucid.test.audit._zoo_device import (
    CHOICES,
    EACH_DEVICE,
    DeviceScope,
    ONCE_ONLY,
    ORDER,
    OnlyOn,
    devices_for,
    where_failed,
)
from lucid.test.audit._zoo_guard import StageGuard, StageRunner, child_env
from lucid.test.audit._zoo_record import (
    FamilyRecord,
    Notebook,
    RunFacts,
    ZooState,
    device_verdict,
    overall,
    render_ledger,
    render_note,
)
from lucid.test.audit._zoo_report import LineReporter
from lucid.test.audit._zoo_stage import (
    ZOO_WIDE,
    CommandStage,
    Stage,
    StageResult,
    ZooCatalog,
)
from lucid.test.audit.__main__ import build_parser, main
from lucid.test.audit.test_audit_zoo import SHA, _budget

ROOT = Path(__file__).resolve().parents[3]


# ── the option ───────────────────────────────────────────────────────────────


def test_the_zoo_runs_on_both_devices_unless_told_otherwise() -> None:
    """The option has no default of its own, so a run with no zoo stage can tell it was given."""
    assert build_parser().parse_args(["--models"]).zoo_device is None
    assert build_parser().parse_args(["--models", "--zoo-device", "cpu"]).zoo_device == "cpu"
    assert _zoo.ZooSettings().device == "both"
    root = Path(__file__).resolve().parents[3]
    unset = _zoo.ZooSettings.from_args(build_parser().parse_args(["--models"]), root, "c")
    assert unset.device == "both" and unset.devices == ("metal", "cpu")
    for narrowed in ("metal", "cpu"):
        parsed = build_parser().parse_args(["--models", "--zoo-device", narrowed])
        settings = _zoo.ZooSettings.from_args(parsed, root, "c")
        assert settings.devices == (narrowed,)


def test_the_choices_and_their_order_have_one_owner() -> None:
    """Metal first: it is the quick pass and the device models train on."""
    assert CHOICES == ("both", "metal", "cpu") and ORDER == ("metal", "cpu")
    assert devices_for("both") == ORDER
    assert devices_for("metal") == ("metal",) and devices_for("cpu") == ("cpu",)
    assert set(ORDER) == set(DEFAULT_DEVICES), "the zoo verifies every device a test can default to"
    with pytest.raises(ValueError, match="unknown zoo device"):
        devices_for("tpu")


def test_an_unknown_device_is_an_argument_error() -> None:
    with pytest.raises(SystemExit) as raised:
        build_parser().parse_args(["--models", "--zoo-device", "tpu"])
    assert raised.value.code == 2


def _args(device: str) -> argparse.Namespace:
    return build_parser().parse_args(["--models", "--zoo-device", device])


@pytest.mark.parametrize("choice", ["both", "metal"])
def test_a_machine_without_metal_is_an_error_not_a_cpu_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], choice: str
) -> None:
    monkeypatch.setattr(_zoo, "metal_available", lambda: False)
    with pytest.raises(PlanError, match="no usable Metal device") as raised:
        _zoo.validate_options(ROOT, _args(choice))
    assert "--zoo-device cpu" in str(raised.value), "and it says how to ask for the CPU"
    _zoo.validate_options(ROOT, _args("cpu"))  # on purpose, that is fine
    assert main(["--models", "--no-log"]) == 2
    out = capsys.readouterr().out
    assert "no usable Metal device" in out and "(the default)" in out


def test_a_machine_with_metal_runs_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_zoo, "metal_available", lambda: True)
    for choice in CHOICES:
        _zoo.validate_options(ROOT, _args(choice))


# ── every child is told, and tells the tests ─────────────────────────────────


def test_the_environment_states_the_device_and_does_not_inherit_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(DEFAULT_DEVICE_ENV, "cpu")
    assert child_env(ROOT, "metal")[DEFAULT_DEVICE_ENV] == "metal"
    monkeypatch.setenv(DEFAULT_DEVICE_ENV, "metal")
    assert child_env(ROOT, "cpu")[DEFAULT_DEVICE_ENV] == "cpu"


@pytest.mark.parametrize("device", ["metal", "cpu"])
def test_the_guard_hands_its_device_to_the_stage(tmp_path: Path, device: str) -> None:
    code = f"import os; print('device=' + os.environ[{DEFAULT_DEVICE_ENV!r}])"
    guard = StageGuard(tmp_path, _budget(8.0), poll_s=0.05, device=device)
    outcome = guard.run((sys.executable, "-c", code), tmp_path / "log")
    assert outcome.status == "pass"
    assert f"device={device}" in (tmp_path / "log").read_text()


def test_the_default_device_is_applied_from_the_environment_and_nowhere_else(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    applied: list[str] = []
    monkeypatch.setattr(lucid, "set_default_device", applied.append)
    monkeypatch.setattr(_default_device, "metal_available", lambda: True)
    monkeypatch.delenv(DEFAULT_DEVICE_ENV, raising=False)
    assert apply_default_device() is None and applied == [], "nobody asked, nothing changes"
    monkeypatch.setenv(DEFAULT_DEVICE_ENV, "metal")
    assert apply_default_device() == "metal" and applied == ["metal"]
    assert apply_default_device("cpu") == "cpu" and applied == ["metal", "cpu"]


def test_asking_the_tests_for_metal_on_a_machine_without_it_fails_loudly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    applied: list[str] = []
    monkeypatch.setattr(lucid, "set_default_device", applied.append)
    monkeypatch.setattr(_default_device, "metal_available", lambda: False)
    with pytest.raises(RuntimeError, match="no usable Metal device"):
        apply_default_device("metal")
    assert applied == [], "it did not fall back to anything"
    with pytest.raises(ValueError, match="unknown default device"):
        apply_default_device("tpu")


def test_the_default_device_the_zoo_stage_asked_for_is_the_one_a_test_sees() -> None:
    """Run by the test below in a process of its own; skipped when nobody asked."""
    wanted = os.environ.get(DEFAULT_DEVICE_ENV)
    if not wanted:
        pytest.skip(f"{DEFAULT_DEVICE_ENV} is not set")
    assert wanted in str(lucid.get_default_device())


def _inner(env_value: str) -> subprocess.CompletedProcess[str]:
    node = (
        f"{Path(__file__)}::test_the_default_device_the_zoo_stage_asked_for_is_the_one_a_test_sees"
    )
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", node],
        cwd=ROOT,
        env={**os.environ, DEFAULT_DEVICE_ENV: env_value, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_conftest_applies_the_device_to_the_tests_of_the_process_it_starts() -> None:
    cpu = _inner("cpu")
    assert "1 passed" in cpu.stdout, cpu.stdout + cpu.stderr
    if metal_available():
        metal = _inner("metal")
        assert "1 passed" in metal.stdout, metal.stdout + metal.stderr


def test_conftest_refuses_a_device_it_does_not_know() -> None:
    bogus = _inner("tpu")
    assert bogus.returncode != 0 and "unknown default device" in bogus.stdout + bogus.stderr


# ── a build really allocates ─────────────────────────────────────────────────


def test_a_build_forces_every_parameter_to_be_computed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """MLX is lazy: a "real-size build" that stops at ``create_model`` allocates nothing."""
    import lucid.metal
    import lucid.models
    from lucid.models import _registry

    forced: list[str] = []

    class Param:
        def __init__(self, name: str) -> None:
            self.name = name

        def sum(self) -> Param:
            return self

        def item(self) -> float:
            forced.append(self.name)
            return 0.0

        def numel(self) -> int:
            return 3

    class Model:
        def parameters(self) -> list[Param]:
            return [Param("w"), Param("b")]

    monkeypatch.setattr(lucid.models, "create_model", lambda name: Model())
    monkeypatch.setattr(lucid.metal, "empty_cache", lambda: None)
    monkeypatch.setattr(
        _registry, "_REGISTRY", {"fake_net": SimpleNamespace(params=None, factory=None)}
    )
    monkeypatch.setattr(
        _zoo_probes,
        "factories_by_family",
        lambda: {"fake": _zoo_probes.Factories(names=["fake_net"], labels={"fake"})},
    )
    assert _zoo_probes.build_family("fake", budget_gb=8.0) == 0
    assert forced == ["w", "b"], "each parameter was read, so the device had to compute it"
    assert "built fake_net: 6 params" in capsys.readouterr().out


def test_an_allocation_that_fails_is_a_failure_of_that_factory(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import lucid.models
    from lucid.models import _registry

    class Param:
        def sum(self) -> Param:
            return self

        def item(self) -> float:
            raise RuntimeError("out of device memory")

        def numel(self) -> int:
            return 3

    class Model:
        def parameters(self) -> list[Param]:
            return [Param()]

    monkeypatch.setattr(lucid.models, "create_model", lambda name: Model())
    monkeypatch.setattr(_registry, "_REGISTRY", {"big": SimpleNamespace(params=None, factory=None)})
    monkeypatch.setattr(
        _zoo_probes,
        "factories_by_family",
        lambda: {"fake": _zoo_probes.Factories(names=["big"], labels={"fake"})},
    )
    assert _zoo_probes.build_family("fake", budget_gb=8.0) == 1
    assert "FAILED big - RuntimeError: out of device memory" in capsys.readouterr().out


# ── a stage declares which devices it is for ─────────────────────────────────


def test_a_scope_says_which_passes_a_stage_runs_in() -> None:
    both = ("metal", "cpu")
    assert [EACH_DEVICE.covers(d, both) for d in both] == [True, True]
    assert [ONCE_ONLY.covers(d, both) for d in both] == [True, False], "one pass: the first"
    assert [ONCE_ONLY.covers(d, ("cpu",)) for d in ("cpu",)] == [True]
    assert [OnlyOn("metal").covers(d, both) for d in both] == [True, False]
    assert not OnlyOn("metal").covers("cpu", ("cpu",)), "a CPU-only run has no Metal pass"


def test_the_catalog_declares_the_scopes_and_the_run_does_not_branch_on_names() -> None:
    stages = {s.name: s for s in ZooCatalog(ROOT).family_stages("lenet", Path("/x"), _budget(40.0))}
    assert stages["contract"].scope == ONCE_ONLY
    assert stages["compiled-training"].scope == OnlyOn("metal")
    assert stages["build"].scope == EACH_DEVICE and stages["zoo-tests"].scope == EACH_DEVICE
    assert dataclasses.replace(stages["zoo-tests"], notes=("x",)).scope == EACH_DEVICE


# ── a run over both devices ──────────────────────────────────────────────────


class _TwoDeviceCatalog(ZooCatalog):
    """Each stage appends ``group/stage@device`` to a file, and fails where told to.

    The stages declare the scopes the real catalog does, so the run decides what
    reaches which device; the stage itself only reports where it was started.
    """

    def __init__(self, root: Path, journal: Path, fails: tuple[str, ...] = ()) -> None:
        super().__init__(root)
        self.journal = journal
        self.fails = fails

    @override
    def families(self) -> dict[str, str]:
        return {"alpha": "vision", "beta": "text"}

    def _stage(self, group: str, name: str, scope: DeviceScope) -> CommandStage:
        code = (
            "import os, sys; dev = os.environ['LUCID_TEST_DEFAULT_DEVICE'];"
            f"open({str(self.journal)!r}, 'a').write('{group}/{name}@' + dev + '\\n');"
            "print('1 passed in 0.01s');"
            f"bad = {self.fails!r};"
            f"sys.exit(1 if '{group}/{name}@' + dev in bad else 0)"
        )
        return CommandStage(
            name=name,
            command=(sys.executable, "-c", code),
            scope=scope,
        )

    @override
    def family_stages(self, family: str, temp: Path, budget: MachineBudget) -> list[Stage]:
        return [
            self._stage(family, "contract", ONCE_ONLY),
            self._stage(family, "build", EACH_DEVICE),
            self._stage(family, "compiled-training", OnlyOn("metal")),
        ]

    @override
    def wide_stages(self, temp: Path) -> list[Stage]:
        return [self._stage(ZOO_WIDE, "wide.py", EACH_DEVICE)]


class _Both:
    """A fake zoo that a test runs again and again, to see what is run and what is kept."""

    def __init__(self, root: Path, fails: tuple[str, ...] = ()) -> None:
        self.root = root
        self.fails = fails

    @property
    def journal(self) -> Path:
        return self.root / "journal"

    def started(self) -> list[str]:
        return self.journal.read_text().split() if self.journal.exists() else []

    def go(
        self,
        device: str = "both",
        *,
        families: tuple[str, ...] = (),
        revision: Callable[[], tuple[str, bool]] | None = None,
    ) -> _zoo.ZooReport:
        console = Console(colour=False)
        settings = _zoo.ZooSettings(
            families=families,
            state_root=self.root / "state",
            poll_s=0.05,
            record_dir=self.root / "notes",
            device=device,
        )
        run = _zoo.ZooRun(
            self.root,
            console,
            settings,
            budget=_budget(40.0),
            reporter=LineReporter(console, 40.0),
            catalog=_TwoDeviceCatalog(self.root, self.journal, self.fails),
            revision=revision or (lambda: (SHA, False)),
        )
        return run.run()


def test_both_devices_run_metal_first_and_each_stage_runs_where_it_is_for(tmp_path: Path) -> None:
    zoo = _Both(tmp_path)
    report = zoo.go(families=("alpha",))
    assert zoo.started() == [
        "alpha/contract@metal",  # once, in the first pass
        "alpha/build@metal",
        "alpha/compiled-training@metal",  # Metal only
        "alpha/build@cpu",  # and the CPU pass has only what is for the CPU
    ]
    assert report.devices == ("metal", "cpu") and report.groups == 1
    assert [(r.name, r.device) for r in report.results["alpha"]] == [
        ("contract", "metal"),
        ("build", "metal"),
        ("compiled-training", "metal"),
        ("build", "cpu"),
    ]


def test_a_narrowed_run_takes_only_its_device_and_its_scopes(tmp_path: Path) -> None:
    metal = _Both(tmp_path / "m")
    metal.go("metal", families=("alpha",))
    assert metal.started() == [
        "alpha/contract@metal",
        "alpha/build@metal",
        "alpha/compiled-training@metal",
    ]
    cpu = _Both(tmp_path / "c")
    cpu.go("cpu", families=("alpha",))
    assert cpu.started() == ["alpha/contract@cpu", "alpha/build@cpu"], (
        "the contract is device independent so it runs once wherever the run starts; "
        "compiled training is Metal only and a CPU run has no Metal pass"
    )


def test_records_are_per_device_and_a_device_that_finished_is_not_rerun(tmp_path: Path) -> None:
    zoo = _Both(tmp_path)
    zoo.go(families=("alpha",))
    state = next((tmp_path / "state").iterdir())
    assert sorted(p.name for p in state.glob("*.json")) == ["alpha.cpu.json", "alpha.metal.json"]
    ran = len(zoo.started())
    again = zoo.go(families=("alpha",))
    assert again.resumed == 1 and len(zoo.started()) == ran, "both devices were recorded"


def test_a_run_that_dies_in_the_cpu_pass_resumes_only_the_cpu(tmp_path: Path) -> None:
    """Finish Metal, die in the CPU pass: the rerun does the CPU and nothing of Metal."""
    zoo = _Both(tmp_path)
    ticks = {"n": 0}

    def dies_when_the_cpu_pass_starts() -> tuple[str, bool]:
        ticks["n"] += 1
        if "alpha/build@cpu" in zoo.started() or ticks["n"] > 5:
            raise KeyboardInterrupt
        return SHA, False

    with pytest.raises(KeyboardInterrupt):
        zoo.go(families=("alpha", "beta"), revision=dies_when_the_cpu_pass_starts)
    state = next((tmp_path / "state").iterdir())
    done = sorted(p.name for p in state.glob("*.json"))
    assert done == ["alpha.metal.json", "beta.metal.json"], "Metal finished, the CPU did not"
    before = len(zoo.started())

    zoo.go(families=("alpha", "beta"))
    assert [r for r in zoo.started()[before:] if r.endswith("@metal")] == [], "Metal is not rerun"
    assert sorted(r for r in zoo.started()[before:]) == ["alpha/build@cpu", "beta/build@cpu"]
    assert sorted(p.name for p in state.glob("*.json")) == [
        "alpha.cpu.json",
        "alpha.metal.json",
        "beta.cpu.json",
        "beta.metal.json",
    ]


def test_a_record_for_one_device_never_answers_for_the_other(tmp_path: Path) -> None:
    zoo = _Both(tmp_path)
    zoo.go("metal", families=("alpha",))
    cpu = zoo.go("cpu", families=("alpha",))
    assert cpu.resumed == 0 and "alpha/build@cpu" in zoo.started()
    assert zoo.go("both", families=("alpha",)).resumed == 1, "now both are on record"
    passed = [StageResult("build", "pass", 1.0, 1.0)]
    assert not FamilyRecord(passed, 40.0, "cpu").reusable_under(_budget(40.0), "metal")
    assert not FamilyRecord(passed, 40.0, "metal").reusable_under(_budget(40.0), "cpu")


def test_a_record_from_before_per_device_state_is_not_read_as_either_device(
    tmp_path: Path,
) -> None:
    """The single-device runs wrote ``<family>.json``; it must not stand for both devices."""
    state = ZooState(tmp_path, SHA, dirty=False, resume=True)
    legacy = [dataclasses.asdict(StageResult("build", "pass", 1.0, 1.0))]
    (state.root / "alpha.json").write_text(
        json.dumps({"limit_gb": 40.0, "device": "metal", "results": legacy})
    )
    for device in ORDER:
        assert state.recorded("alpha", device) is None
        assert state.load("alpha", _budget(40.0), device) is None
    assert state.latest("alpha") == []
    ledger = render_ledger(RunFacts(SHA, "c", _budget(40.0)), state, {"alpha": "vision"})
    assert "| alpha | vision | 대기 | 대기 | 대기 | | | |" in ledger
    # and a record filed under one device's name that says it is the other's is refused
    state.save("alpha", [StageResult("build", "pass", 1.0, 1.0)], _budget(40.0), "metal")
    (state.root / "alpha.cpu.json").write_text((state.root / "alpha.metal.json").read_text())
    assert state.recorded("alpha", "cpu") is None


# ── the verdict is the worse of the two, and says which one ──────────────────


def test_the_worse_device_is_the_family_verdict_and_the_device_is_named(tmp_path: Path) -> None:
    zoo = _Both(tmp_path, fails=("alpha/build@cpu",))
    report = zoo.go(families=("alpha", "beta"))
    assert report.failures == 1 and list(report.failing_groups) == ["alpha"]
    [(group, failed)] = report.failed
    assert (group, failed.name, failed.device) == ("alpha", "build", "cpu")
    assert report.label("alpha", failed) == "build (fail, cpu only)"
    assert report.failing_by_device() == {"cpu only": 1}
    results = report.results["alpha"]
    assert overall(results) == "fail" and where_failed(results) == "cpu only"
    assert device_verdict(results) == "metal pass · cpu fail (cpu only)"
    assert overall(report.results["beta"]) == "pass" and where_failed(report.results["beta"]) == ""


def test_a_failure_on_metal_alone_and_on_both_are_told_apart(tmp_path: Path) -> None:
    metal_only = _Both(tmp_path / "a", fails=("alpha/build@metal",)).go(families=("alpha",))
    assert metal_only.failing_by_device() == {"metal only": 1}
    both = _Both(tmp_path / "b", fails=("alpha/build@metal", "alpha/build@cpu")).go(
        families=("alpha",)
    )
    assert both.failing_by_device() == {"both devices": 1}
    label = both.label("alpha", both.failed[0][1])
    assert label == "build (fail, both devices)"


def test_a_stage_for_one_device_or_a_narrowed_run_makes_no_only_claim(tmp_path: Path) -> None:
    """``cpu only`` says the other device ran the same stage and passed; here it did not run."""
    compiled = _Both(tmp_path / "a", fails=("alpha/compiled-training@metal",)).go(
        families=("alpha",)
    )
    [(group, failed)] = compiled.failed
    assert compiled.label(group, failed) == "compiled-training (fail, metal)"
    assert where_failed(compiled.results["alpha"], "compiled-training") == ""
    narrowed = _Both(tmp_path / "b", fails=("alpha/build@cpu",)).go("cpu", families=("alpha",))
    assert narrowed.failing_by_device() == {}
    assert narrowed.label("alpha", narrowed.failed[0][1]) == "build (fail, cpu)"


def test_the_screen_summary_and_the_file_say_which_device_failed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    zoo = _Both(tmp_path, fails=("alpha/build@cpu",))
    report = zoo.go(families=("alpha",))
    _zoo.summarise(Console(colour=False), report, _budget(40.0), tmp_path / "notes")
    out = capsys.readouterr().out
    assert "metal → cpu" in out and "1 group(s) fail on cpu only" in out
    assert "alpha: build (fail, cpu only)" in out
    assert report.summary_path is not None
    text = report.summary_path.read_text()
    assert "devices metal → cpu · 1 group(s) fail on cpu only" in text
    assert "alpha/build (fail, cpu only)" in text


def test_the_ledger_and_the_note_show_each_devices_result(tmp_path: Path) -> None:
    _Both(tmp_path, fails=("alpha/build@cpu",)).go(families=("alpha", "beta"))
    ledger = (tmp_path / "notes" / "zoo-verify-ledger.md").read_text()
    assert "| family | 분야 | 결과 | metal | cpu | 분 | 최대 GB | 기록 |" in ledger
    assert "| alpha | vision | fail (cpu only) | pass | fail |" in ledger
    assert "| beta | text | pass | pass | pass |" in ledger
    assert "이번 실행 장치**: both (metal → cpu)" in ledger
    note = (tmp_path / "notes" / "zoo-verify-alpha.md").read_text()
    assert "> 장치별: metal pass · cpu fail (cpu only)" in note
    assert "> **fail**" in note and "요청 장치: both" in note
    assert "| metal | contract | pass |" in note and "| cpu | build | fail |" in note
    assert "| cpu | contract |" not in note and "| cpu | compiled-training |" not in note
    assert "- `build` [cpu]:" in note, "a failure is listed with the device it happened on"


def test_a_ledger_waits_for_the_device_that_has_not_run(tmp_path: Path) -> None:
    _Both(tmp_path).go("metal", families=("alpha",))
    ledger = (tmp_path / "notes" / "zoo-verify-ledger.md").read_text()
    assert "| alpha | vision | pass | pass | 대기 |" in ledger
    note = (tmp_path / "notes" / "zoo-verify-alpha.md").read_text()
    assert "> 장치별: metal pass · cpu 기록 없음" in note
    both = RunFacts(SHA, "c", _budget(40.0), zoo_device="both")
    state = ZooState(tmp_path / "state", SHA, dirty=False, resume=True)
    assert "진행: 0 / 3" in render_ledger(both, state, {"alpha": "vision", "beta": "text"})


def test_a_hand_made_record_and_note_say_where_each_row_ran() -> None:
    results = [
        StageResult("build", "pass", 1.0, 1.0, device="metal"),
        StageResult("build", "mem-guard", 1.0, 9.0, failures=["killed"], device="cpu"),
    ]
    note = render_note("alpha", "vision", results, RunFacts(SHA, "c", _budget(8.0)))
    assert "- `build` [cpu]: killed" in note and "metal pass · cpu mem-guard (cpu only)" in note
    assert "| metal | build | pass |" in note and "| cpu | build | mem-guard |" in note


def test_notes_are_written_after_each_device_so_a_death_leaves_the_first(tmp_path: Path) -> None:
    state = ZooState(tmp_path / "s", SHA, dirty=False, resume=True)
    book = Notebook(
        tmp_path / "n", state, {"alpha": "vision"}, RunFacts(SHA, "c", _budget(8.0), zoo_device="both")
    )
    state.save("alpha", [StageResult("build", "pass", 1.0, 1.0)], _budget(8.0), "metal")
    book.write("alpha")
    assert "metal pass · cpu 기록 없음" in book.note_path("alpha").read_text()
    state.save("alpha", [StageResult("build", "fail", 1.0, 1.0)], _budget(8.0), "cpu")
    book.write("alpha")
    assert "metal pass · cpu fail (cpu only)" in book.note_path("alpha").read_text()


# ── what a skip or a kill applied to is named ────────────────────────────────


def test_a_kill_and_its_retry_say_which_device_they_happened_on(tmp_path: Path) -> None:
    code = "import time; x = b'x' * (300 * 2**20); time.sleep(30)"
    from lucid.test.audit._zoo_stage import PytestStage

    class Killed(PytestStage):
        def argv(self) -> tuple[str, ...]:
            return (sys.executable, "-c", code)

    stage = Killed(name="own-tests", files=(), basetemp=tmp_path)
    guard = StageGuard(tmp_path, _budget(0.15), poll_s=0.05, device="cpu")
    result = StageRunner(guard, tmp_path / "logs", tmp_path / "tmp").run(stage, "alpha")
    assert result.status == "mem-guard" and result.device == "cpu"
    assert any("killed the first attempt on cpu" in note for note in result.notes)


def test_the_report_labels_what_was_adapted_with_its_device() -> None:
    report = _zoo.ZooReport()
    note = "heavy tests deselected up front on cpu: ..."
    report.add(
        "lenet",
        [
            StageResult("zoo-tests", "pass", 1.0, 1.0, notes=[note], device="cpu"),
            StageResult("zoo-tests", "pass", 1.0, 1.0, device="metal"),
        ],
        resumed=False,
    )
    assert report.adapted == [("lenet", "zoo-tests (cpu)", note)]


def test_each_pass_counts_its_own_families(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _Both(tmp_path).go(families=("alpha", "beta"))
    out = capsys.readouterr().out
    assert "── metal ── 2 group(s)" in out and "── cpu ── 2 group(s)" in out
    assert "[2/2] metal beta: pass" in out and "[2/2] cpu beta: pass" in out
    assert "families ok 2" in out and "families ok 4" not in out, "the tally starts again per device"
