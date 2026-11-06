"""The zoo is verified on the device it is asked to be verified on — Metal by default.

"Who runs a deep-learning model on the CPU": the zoo stage's children used to
default to whatever the process did, which is the CPU.  ``--zoo-device`` states
it (``metal`` unless told otherwise), the guard hands it to every child, the
tests and probes apply it, the record keeps it, and a record made on one device
is not an answer for a run on the other.  A machine with no Metal is an error —
never a quiet run on the CPU that reports itself as the Metal verification.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import lucid
from lucid.test import _default_device
from lucid.test._default_device import DEFAULT_DEVICE_ENV, apply_default_device, metal_available
from lucid.test.audit import _zoo, _zoo_probes
from lucid.test.audit._plan import PlanError
from lucid.test.audit._zoo_budget import MachineBudget
from lucid.test.audit._zoo_guard import StageGuard, child_env
from lucid.test.audit._zoo_record import FamilyRecord
from lucid.test.audit._zoo_stage import StageResult
from lucid.test.audit.__main__ import build_parser, main
from lucid.test.audit.test_audit_zoo import _budget, _Run

ROOT = Path(__file__).resolve().parents[3]


# ── the option ───────────────────────────────────────────────────────────────


def test_the_zoo_runs_on_metal_unless_told_otherwise() -> None:
    """The option has no default of its own, so a run with no zoo stage can tell it was given."""
    assert build_parser().parse_args(["--models"]).zoo_device is None
    assert build_parser().parse_args(["--models", "--zoo-device", "cpu"]).zoo_device == "cpu"
    assert _zoo.ZooSettings().device == "metal"
    root = Path(__file__).resolve().parents[3]
    unset = _zoo.ZooSettings.from_args(build_parser().parse_args(["--models"]), root, "c")
    assert unset.device == "metal"
    cpu = _zoo.ZooSettings.from_args(
        build_parser().parse_args(["--models", "--zoo-device", "cpu"]), root, "c"
    )
    assert cpu.device == "cpu"


def test_an_unknown_device_is_an_argument_error() -> None:
    with pytest.raises(SystemExit) as raised:
        build_parser().parse_args(["--models", "--zoo-device", "tpu"])
    assert raised.value.code == 2


def _args(device: str) -> argparse.Namespace:
    return build_parser().parse_args(["--models", "--zoo-device", device])


def test_a_machine_without_metal_is_an_error_not_a_cpu_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(_zoo, "metal_available", lambda: False)
    with pytest.raises(PlanError, match="no usable Metal device") as raised:
        _zoo.validate_options(ROOT, _args("metal"))
    assert "--zoo-device cpu" in str(raised.value), "and it says how to ask for the CPU"
    _zoo.validate_options(ROOT, _args("cpu"))  # on purpose, that is fine
    assert main(["--models", "--no-log"]) == 2
    assert "no usable Metal device" in capsys.readouterr().out


def test_a_machine_with_metal_runs_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_zoo, "metal_available", lambda: True)
    _zoo.validate_options(ROOT, _args("metal"))


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


# ── the record keeps the device, and a different one is a different run ──────


def test_a_record_made_on_one_device_is_not_an_answer_for_the_other() -> None:
    passed = [StageResult("build", "pass", 1.0, 1.0)]
    on_cpu = FamilyRecord(passed, limit_gb=40.0, device="cpu")
    assert on_cpu.reusable_under(_budget(40.0), "cpu")
    assert not on_cpu.reusable_under(_budget(40.0), "metal")
    assert not FamilyRecord(passed, 40.0, "metal").reusable_under(_budget(40.0), "cpu")


def test_changing_the_device_reruns_the_family_and_the_same_device_resumes(tmp_path: Path) -> None:
    runs = _Run(tmp_path)
    assert runs.go(device="metal").resumed == 0
    first = len(runs.runs())
    assert runs.go(device="metal").resumed == 3 and len(runs.runs()) == first
    assert runs.go(device="cpu").resumed == 0 and len(runs.runs()) == 2 * first
    assert runs.go(device="cpu").resumed == 3
    assert runs.go(device="metal").resumed == 0, "the cpu run replaced the metal record"


def test_the_notes_and_the_ledger_say_which_device_the_zoo_was_verified_on(tmp_path: Path) -> None:
    runs = _Run(tmp_path)
    runs.go(device="cpu")
    note = (tmp_path / "notes" / "zoo-verify-alpha.md").read_text()
    ledger = (tmp_path / "notes" / "zoo-verify-ledger.md").read_text()
    assert "장치: cpu" in note
    assert "이번 실행 장치**: cpu" in ledger
    assert "| 장치 |" in ledger and "| alpha | vision | pass |" in ledger and "| cpu |" in ledger


def test_the_banner_and_the_state_show_the_device(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs = _Run(tmp_path)
    runs.go(device="metal")
    out = capsys.readouterr().out
    assert "device" in out and "metal" in out
    state = next((tmp_path / "state").iterdir())
    assert '"device": "metal"' in (state / "alpha.json").read_text()


def test_the_regular_machine_budget_is_unchanged_by_the_device() -> None:
    assert MachineBudget.for_memory(36).limit_gb == 28.0
