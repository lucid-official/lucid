"""The package registry, the flags generated from it, and what a command line runs.

What a run covers used to be spelled three ways (a ``--subsystem`` string,
stage flags, and a separate script for the zoo), and the zoo was not in the
audit at all.  These tests hold the one spelling that replaced them:

* a flag exists for every package, generated, so a new ``SUBSYSTEMS`` entry
  needs no other edit;
* a parent takes its children, ``--no-*`` subtracts, and a contradiction is
  exit ``2`` rather than a silent winner;
* stage flags stay orthogonal and exclude the zoo stage;
* no generated flag shadows a declared one (``--no-metal`` did, once);
* CI's command lines cannot select the model zoo, whatever else changes.
"""

import argparse
import re
import shlex
from pathlib import Path

import pytest

from lucid.test.audit import _surface
from lucid.test.audit.__main__ import _verdict, build_parser, main
from lucid.test.audit._console import Console
from lucid.test.audit._packages import (
    MODELS,
    PACKAGES,
    PackageRegistry,
    PackageSelectionError,
    SubsystemPackage,
)
from lucid.test.audit._plan import PlanError, RunPlan

ROOT = Path(__file__).resolve().parents[3]


def _args(*argv: str) -> argparse.Namespace:
    return build_parser().parse_args(list(argv))


def _keys(*argv: str) -> frozenset[str]:
    return PACKAGES.select(_args(*argv)).keys


# ── the flags are generated from the registry ────────────────────────────────


def test_every_subsystem_key_and_the_zoo_is_a_package() -> None:
    assert set(PACKAGES.keys()) == {*_surface.SUBSYSTEMS, MODELS}


def test_every_package_gets_a_flag_and_its_negation_with_dots_kept() -> None:
    parser = build_parser()
    for package in PACKAGES:
        assert package.flag == f"--{package.key}"
        assert package.negated_flag == f"--no-{package.key}"
        assert parser._option_string_actions[package.flag].dest == package.dest
        assert parser._option_string_actions[package.negated_flag].dest == package.negated_dest
    assert "--nn.functional" in PACKAGES.flags()
    assert "--no-nn.functional" in PACKAGES.flags()


def test_a_new_subsystem_entry_gets_its_flags_without_another_edit(monkeypatch) -> None:
    monkeypatch.setitem(_surface.SUBSYSTEMS, "brand_new", ("lucid.brand_new", "op"))
    registry = PackageRegistry.default()
    assert {"--brand_new", "--no-brand_new"} <= registry.flags()
    parser = argparse.ArgumentParser(allow_abbrev=False)
    registry.add_flags(parser)
    namespace = parser.parse_args(["--brand_new"])
    assert registry.select(namespace).keys == {"brand_new"}


def test_a_parent_includes_its_children() -> None:
    assert {"nn", "nn.functional", "nn.init", "nn.utils", "nn.quantized"} <= _keys("--nn")
    assert _keys("--optim") == {"optim", "optim.lr_scheduler"}
    assert _keys("--utils") >= {"utils", "utils.data", "utils.transforms.functional"}
    # A child alone is itself, and a key that merely shares a prefix is not a child.
    assert _keys("--nn.functional") == {"nn.functional"}
    assert _keys("--lucid") == {"lucid"}


def test_the_default_is_every_package_including_the_zoo() -> None:
    selection = PACKAGES.select(_args())
    assert selection.keys == frozenset(PACKAGES.keys())
    assert selection.runs_zoo and not selection.explicit


def test_flags_are_exact_so_an_abbreviation_cannot_pick_a_package() -> None:
    with pytest.raises(SystemExit) as raised:
        _args("--nn.func")
    assert raised.value.code == 2


def test_an_unknown_package_fails_as_argparse_does() -> None:
    with pytest.raises(SystemExit) as raised:
        _args("--no-such-package")
    assert raised.value.code == 2


# ── selection semantics ──────────────────────────────────────────────────────

_ALL = frozenset(PACKAGES.keys())
_NN = PACKAGES.subtree("nn")


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], _ALL),
        (["--no-models"], _ALL - {MODELS}),
        (["--models"], {MODELS}),
        (["--nn"], _NN),
        (["--nn", "--linalg"], _NN | {"linalg"}),
        (["--nn", "--models"], _NN | {MODELS}),
        (["--no-nn"], _ALL - _NN),
        (["--no-nn", "--no-models"], _ALL - _NN - {MODELS}),
        (["--nn", "--no-nn.functional"], _NN - {"nn.functional"}),
        (["--no-nn.functional"], _ALL - {"nn.functional"}),
    ],
)
def test_selection_semantics(argv: list[str], expected: set[str]) -> None:
    assert _keys(*argv) == frozenset(expected)


@pytest.mark.parametrize(
    "argv",
    [
        ["--models", "--no-models"],
        ["--nn", "--no-nn"],
        ["--nn.functional", "--no-nn"],  # excluding what contains what you asked for
        [f"--no-{key}" for key in PACKAGES.keys()],  # nothing left
    ],
)
def test_contradictions_are_selection_errors(argv: list[str]) -> None:
    with pytest.raises(PackageSelectionError):
        PACKAGES.select(_args(*argv))


@pytest.mark.parametrize(
    "argv",
    [
        ["--models", "--no-models"],
        ["--nn", "--no-nn"],
        ["--models", "--audit-only"],
        ["--models", "--tests-only"],
        ["--models", "--doctests-only"],
        ["--audit-only", "--tests-only"],
        ["--audit-only", "--family", "lenet"],
        ["--no-models", "--fresh"],
        ["--nn", "--zoo-record", "/tmp/never-written"],
        ["--no-models", "--mem-limit-gb", "8"],
        ["--update-coverage", "--nn"],
        ["--update-coverage", "--models"],
        ["--update-coverage", "--cpu-only"],
    ],
)
def test_a_contradictory_command_line_exits_2_and_says_why(argv: list[str], capsys) -> None:
    assert main([*argv, "--no-log"]) == 2
    assert capsys.readouterr().out.strip(), "exit 2 with no explanation"


def test_an_unknown_family_exits_2_before_any_stage_runs(capsys) -> None:
    assert main(["--models", "--family", "no_such_family", "--no-log"]) == 2
    assert "no_such_family" in capsys.readouterr().out


# ── stages are orthogonal to packages ────────────────────────────────────────


def _plan(*argv: str) -> RunPlan:
    return RunPlan.from_args(_args(*argv))


@pytest.mark.parametrize(
    ("argv", "stages"),
    [
        ([], {"self_check", "sweep", "suite", "doctests", "zoo"}),
        (["--no-models"], {"self_check", "sweep", "suite", "doctests"}),
        (["--models"], {"zoo"}),
        (["--models", "--nn"], {"self_check", "sweep", "suite", "doctests", "zoo"}),
        (["--audit-only"], {"self_check", "sweep"}),
        (["--tests-only"], {"suite", "doctests"}),
        (["--doctests-only"], {"doctests"}),
        (["--tests-only", "--no-doctests"], {"suite"}),
        (["--audit-only", "--no-self-check"], {"sweep"}),
        (["--nn", "--audit-only"], {"self_check", "sweep"}),
    ],
)
def test_the_stages_a_command_line_runs(argv: list[str], stages: set[str]) -> None:
    plan = _plan(*argv)
    ran = {
        name for name in ("self_check", "sweep", "suite", "doctests", "zoo") if getattr(plan, name)
    }
    assert ran == stages


def test_stage_flags_exclude_the_zoo_even_when_it_is_selected() -> None:
    for flag in ("--audit-only", "--tests-only", "--doctests-only"):
        plan = _plan(flag)
        assert plan.selection.runs_zoo and not plan.zoo


def test_no_models_is_not_a_filter_for_a_coverage_update() -> None:
    plan = _plan("--no-models", "--audit-only", "--update-coverage")
    assert plan.sweep and plan.selection.sweep_keys() is None


def test_a_partial_sweep_still_cannot_be_recorded() -> None:
    with pytest.raises(PlanError, match="complete"):
        _plan("--audit-only", "--update-coverage", "--no-nn")


def test_the_sweep_walks_the_selected_packages_only() -> None:
    assert PACKAGES.select(_args()).sweep_keys() is None
    assert PACKAGES.select(_args("--no-models")).sweep_keys() is None
    assert PACKAGES.select(_args("--nn.functional", "--linalg")).sweep_keys() == [
        "nn.functional",
        "linalg",
    ]
    assert PACKAGES.select(_args("--no-nn")).sweep_keys() == [
        k for k in _surface.SUBSYSTEMS if k not in _NN
    ]


# ── what the suite and the doctests are asked to cover ───────────────────────


def test_the_suite_leaves_the_zoo_to_its_own_stage() -> None:
    zoo_trees = ("lucid/test/unit/models", "lucid/test/parity/models")
    plain = _plan().selection.suite_plan(ROOT, zoo_stage=True)
    assert plain.paths == ("lucid/test",)
    assert set(plain.ignore) == set(zoo_trees)
    # Excluding the zoo leaves it out of the suite as well.
    assert set(_plan("--no-models").selection.suite_plan(ROOT).ignore) == set(zoo_trees)


def test_a_tests_only_run_still_carries_the_zoo_trees_as_it_always_did() -> None:
    assert _plan("--tests-only").selection.suite_plan(ROOT, zoo_stage=False).ignore == ()


def test_an_explicit_selection_runs_those_packages_trees() -> None:
    plan = _plan("--nn", "--linalg").selection.suite_plan(ROOT)
    assert "lucid/test/unit/nn" in plan.paths
    assert "lucid/test/unit/linalg" in plan.paths
    assert not any(path.endswith("models") for path in plan.paths)
    assert plan.ignore == ()


def test_a_tree_shared_with_a_selected_package_is_never_ignored() -> None:
    """``nn.functional`` shares ``unit/nn`` with ``nn``; dropping one keeps the other's tests."""
    plan = _plan("--no-nn.functional", "--no-models").selection.suite_plan(ROOT)
    assert "lucid/test/unit/nn" not in plan.ignore


def test_excluding_a_package_ignores_its_trees() -> None:
    plan = _plan("--no-linalg", "--no-models").selection.suite_plan(ROOT)
    assert "lucid/test/unit/linalg" in plan.ignore


def test_an_explicit_suite_path_wins() -> None:
    plan = _plan("--nn").selection.suite_plan(ROOT, "lucid/test/unit/ops")
    assert plan.paths == ("lucid/test/unit/ops",) and plan.ignore == ()


def test_a_complete_run_compares_the_whole_line_floor() -> None:
    assert _plan().selection.suite_plan(ROOT).in_scope is None
    assert _plan("--no-models").selection.suite_plan(ROOT).in_scope is None


def test_line_coverage_is_judged_only_for_the_selected_packages_files() -> None:
    nn = _plan("--nn").selection
    assert nn.owns_file("lucid/nn/modules/linear.py")
    assert not nn.owns_file("lucid/optim/sgd.py")
    assert not nn.owns_file("lucid/_dispatch.py"), "a file no package owns is not an nn file"
    without = _plan("--no-nn.functional", "--no-models").selection
    assert without.owns_file("lucid/nn/modules/linear.py")
    assert not without.owns_file("lucid/nn/functional/_loss.py"), "the longest prefix owns it"
    assert without.owns_file("lucid/_dispatch.py"), "a negative run keeps unowned files"
    assert not without.owns_file("lucid/models/vision/lenet/_model.py")


def test_the_doctests_follow_the_longest_module_prefix() -> None:
    nn = _plan("--nn").selection
    assert nn.owns_module("lucid.nn.functional.activation")
    assert not nn.owns_module("lucid.optim.sgd")
    without = _plan("--no-nn").selection
    assert not without.owns_module("lucid.nn.modules.linear")
    assert without.owns_module("lucid._dispatch")


# ── no generated flag shadows a declared one ─────────────────────────────────


def test_no_generated_flag_collides_with_a_declared_flag() -> None:
    parser = build_parser()
    generated = PACKAGES.flags()
    declared = {opt for action in parser._actions for opt in action.option_strings} - generated
    assert not generated & declared
    owners: dict[str, int] = {}
    for action in parser._actions:
        for opt in action.option_strings:
            owners[opt] = owners.get(opt, 0) + 1
    assert all(owners[flag] == 1 for flag in generated)


def test_a_flag_that_is_already_declared_is_an_error_not_a_silent_shadow() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--no-metal", action="store_true")
    with pytest.raises(argparse.ArgumentError):
        PACKAGES.add_flags(parser)


def test_the_cpu_only_run_flag_replaced_the_one_that_shadowed_the_metal_package() -> None:
    parser = build_parser()
    assert "--cpu-only" in parser._option_string_actions
    assert parser._option_string_actions["--no-metal"].dest == "no_pkg_metal"
    assert _keys("--no-metal") == _ALL - {"metal"}
    assert _args("--cpu-only").cpu_only is True


def test_the_removed_flags_are_gone() -> None:
    parser = build_parser()
    assert "--subsystem" not in parser._option_string_actions
    assert "--list-subsystems" not in parser._option_string_actions
    assert "--list-packages" in parser._option_string_actions


def test_list_packages_names_the_zoo(capsys) -> None:
    assert main(["--list-packages", "--no-log"]) == 0
    out = capsys.readouterr().out
    assert "--models" in out and "zoo" in out and "families" in out
    assert "--nn.functional" in out


# ── CI cannot select the model zoo ───────────────────────────────────────────

_COMMAND = re.compile(r"(?:lucid-audit|-m lucid\.test\.audit)\s+(--[^`\n]*)")
_CI_FILES = [
    *sorted((ROOT / ".github" / "workflows").glob("*.yml")),
    *sorted((ROOT / "scripts").glob("*.sh")),
    *sorted((ROOT / "scripts").glob("*.py")),
    ROOT / "tools" / "check_audit_baselines.py",
]


def _audit_commands(path: Path) -> list[list[str]]:
    """Every ``lucid-audit ...`` / ``-m lucid.test.audit ...`` command a file spells out."""
    found = []
    for line in path.read_text().splitlines():
        for match in _COMMAND.finditer(line):
            text = match.group(1)
            try:
                found.append(shlex.split(text))
            except ValueError:  # the command ends at a closing quote of a Python string
                found.append(shlex.split(text.rstrip('",.)')))
    return found


def test_the_command_scanner_finds_the_ci_calls() -> None:
    """The scan below proves nothing if it silently reads zero commands."""
    assert len(_audit_commands(ROOT / "scripts" / "ci_full.sh")) >= 3
    assert len(_audit_commands(ROOT / "tools" / "check_audit_baselines.py")) == 3


def test_no_ci_or_script_command_line_selects_the_model_zoo() -> None:
    checked = 0
    for path in _CI_FILES:
        for tokens in _audit_commands(path):
            plan = RunPlan.from_args(build_parser().parse_args(tokens))
            where = f"{path.relative_to(ROOT)}: lucid-audit {' '.join(tokens)}"
            assert not plan.selection.runs_zoo, f"{where} would include the model zoo"
            assert not plan.zoo, f"{where} would run the zoo stage"
            checked += 1
    assert checked >= 5


def test_the_package_flags_alone_would_have_selected_the_zoo() -> None:
    """The reason CI passes --no-models: a bare stage flag leaves the zoo selected."""
    plan = RunPlan.from_args(build_parser().parse_args(["--audit-only"]))
    assert plan.selection.runs_zoo and not plan.zoo


# ── the verdict ──────────────────────────────────────────────────────────────


def test_the_verdict_has_a_zoo_row_and_the_exit_code_follows_it(capsys) -> None:
    console = Console(colour=False)
    assert _verdict(console, {"audit defects": 0, "model zoo failures": 0}) == 0
    assert "model zoo failures" in capsys.readouterr().out
    assert _verdict(console, {"audit defects": 0, "model zoo failures": 2}) == 1


def test_a_subsystem_package_describes_itself_from_the_surface_table() -> None:
    package = SubsystemPackage("nn.functional")
    assert package.summary() == "lucid.nn.functional (op)"
    assert package.module_prefix() == "lucid.nn.functional"
    assert package.symbol_keys() == ("nn.functional",)
    assert package.file_prefixes() == ("lucid/nn/functional/", "lucid/nn/functional.py")
