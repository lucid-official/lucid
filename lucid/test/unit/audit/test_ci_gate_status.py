"""A broken instrument cannot turn a required CI stage green, and a stage
failure in ``part_checks`` / ``part_core`` / ``part_coreml`` / ``Coverage``
does not mask later stages (LCD-530)."""

import ast
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "stage,next_marker",
    [
        ("Symbol x axis audit", "# Docstring examples"),
        (
            "Docstring examples (against the doctest floor)",
            "# Model-zoo family contract",
        ),
    ],
)
@pytest.mark.parametrize("status", [0, 1, 2])
def test_required_ci_stage_preserves_failure(
    tmp_path: Path,
    stage: str,
    next_marker: str,
    status: int,
) -> None:
    root = Path(__file__).resolve().parents[4]
    script = (root / "scripts/ci_full.sh").read_text()
    block = script.split(f'echo "==> {stage}"', 1)[1].split(next_marker, 1)[0]
    fake_python = tmp_path / "python"
    fake_python.write_text(
        f"#!{sys.executable}\nimport os, sys\nsys.exit(int(os.environ['TEST_STAGE_STATUS']))\n"
    )
    fake_python.chmod(0o700)
    harness = (
        "set -euo pipefail\n"
        "failed_stages=()\n"
        + block
        + "\nprintf 'NEXT_STAGE\\n'\n"
        + 'if [ "${#failed_stages[@]}" -ne 0 ]; then exit 1; fi\n'
    )
    result = subprocess.run(
        ["bash", "-c", harness],
        env={
            **os.environ,
            "PYTHON_BIN": str(fake_python),
            "TEST_STAGE_STATUS": str(status),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == status, result.stdout + result.stderr
    # Exit 1 records the failure and continues to later stages; exit 2
    # ("harness broke") stops immediately.
    assert ("NEXT_STAGE" in result.stdout) == (status in (0, 1))


def _function(script: str, name: str) -> str:
    """The text of shell function ``name`` in ``script``, header to brace."""
    return (
        f"{name}() {{"
        + script.split(f"\n{name}() {{", 1)[1].split("\n}", 1)[0]
        + "\n}\n"
    )


def test_part_core_and_coverage_do_not_abort_on_first_failure_or_collection_error() -> (
    None
):
    root = Path(__file__).resolve().parents[4]
    ci_full = (root / "scripts/ci_full.sh").read_text()
    assert (
        "\nFAST_TIER_FAILURES=(--maxfail=50 --continue-on-collection-errors)\n"
        in ci_full
    )
    for part in ("part_core", "part_coreml"):
        body = _function(ci_full, part)
        commands = [c for c in _logical_lines(body) if "-m pytest" in c]
        assert commands, part
        for command in commands:
            assert '"${FAST_TIER_FAILURES[@]}"' in command, (part, command)

    ci_yml = (root / ".github/workflows/ci.yml").read_text()
    cov_lines = [
        line.strip()
        for line in ci_yml.splitlines()
        if "coverage run --parallel-mode" in line
    ]
    assert len(cov_lines) == 3
    for line in cov_lines:
        assert "--continue-on-collection-errors" in line


def test_part_coreml_runs_the_export_smoke_after_a_red_first_process(
    tmp_path: Path,
) -> None:
    """Since 10-21 one red Core ML test meant the export smoke never ran."""
    root = Path(__file__).resolve().parents[4]
    ci_full = (root / "scripts/ci_full.sh").read_text()
    calls = tmp_path / "calls"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        f"#!{sys.executable}\nimport sys\n"
        f"with open({str(calls)!r}, 'a') as log:\n"
        "    log.write(' '.join(sys.argv[1:]) + '\\n')\n"
        "sys.exit(1 if any(a.startswith('--ignore=') for a in sys.argv) else 0)\n"
    )
    fake_python.chmod(0o700)
    definitions = "".join(
        line + "\n"
        for line in ci_full.splitlines()
        if line.startswith(("COREML_TESTS=", "EXPORT_SMOKE=", "FAST_TIER_FAILURES="))
    )
    script = (
        "set -euo pipefail\n"
        + definitions
        + _function(ci_full, "part_coreml")
        + "part_coreml\n"
    )
    result = subprocess.run(
        ["bash", "-c", script],
        env={**os.environ, "PYTHON_BIN": str(fake_python)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    ran = calls.read_text().splitlines()
    assert len(ran) == 2, ran
    assert ran[1].startswith(
        "-m pytest lucid/test/unit/coreml/test_family_export_smoke.py"
    ), ran


#: A pytest flag that ends the run at its first failure.
_STOP_AT_FIRST = re.compile(r"-[qvsl]*x[qvsl]*|--exitfirst|--maxfail=1")


def _stops_at_first(args: list[str]) -> bool:
    """Whether pytest ``args`` end the run at its first failure."""
    return any(
        _STOP_AT_FIRST.fullmatch(arg)
        or (arg == "--maxfail" and args[i + 1 : i + 2] == ["1"])
        for i, arg in enumerate(args)
    )


@pytest.mark.parametrize(
    ("args", "stops"),
    [
        (["-x"], True),
        (["-qx"], True),
        (["--exitfirst"], True),
        (["--maxfail=1"], True),
        (["--maxfail", "1"], True),
        (["--maxfail", "50"], False),
        (["--maxfail=50", "-q"], False),
        (["-rsx"], False),  # -r takes "sx" as report characters
        (["-c", "x"], False),
    ],
)
def test_the_first_failure_detector(args: list[str], stops: bool) -> None:
    assert _stops_at_first(args) is stops
_SCRIPT_REF = re.compile(r"(?:scripts|tools)/[\w./-]+\.(?:sh|py)")


def _logical_lines(text: str) -> list[str]:
    """Shell lines with their continuations joined, comments dropped."""
    joined = re.sub(r"\\\n\s*", " ", text)
    return [
        line.strip()
        for line in joined.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _ci_files(root: Path) -> list[Path]:
    """Every workflow and action, and every script they reach, transitively."""
    seen = set((root / ".github").rglob("*.yml"))
    queue = list(seen)
    while queue:
        for ref in _SCRIPT_REF.findall(queue.pop().read_text()):
            path = root / ref
            if path.is_file() and path not in seen:
                seen.add(path)
                queue.append(path)
    return sorted(seen)


def _pytest_arguments(path: Path) -> list[list[str]]:
    """The arguments of every pytest command ``path`` runs."""
    if path.suffix == ".py":
        # A command built as a list: the strings of any list or tuple that
        # names pytest, after it.
        commands = []
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, (ast.List, ast.Tuple)):
                words = [
                    e.value
                    for e in node.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)
                ]
                if "pytest" in words:
                    commands.append(words[words.index("pytest") + 1 :])
        return commands
    return [
        match.group(1).split()
        for line in _logical_lines(path.read_text())
        if (match := re.search(r"(?:^|[\s(])pytest\s(.*)", line))
    ]


def test_no_ci_pytest_command_stops_at_its_first_failure() -> None:
    """A run that stops at the first red test reports one defect and hides
    the rest — 91% of core under ``-x`` (LCD-530), then the coreml part's
    export smoke."""
    root = Path(__file__).resolve().parents[4]
    files = _ci_files(root)
    commands = {path: _pytest_arguments(path) for path in files}
    # The scan has to reach the gate's scripts, or it proves nothing.
    for script in ("scripts/ci_full.sh", "scripts/ci_nightly_suite.py"):
        assert commands.get(root / script), f"no pytest command found in {script}"
    offenders = [
        f"{path.relative_to(root)}: pytest {' '.join(args)}"
        for path, found in commands.items()
        for args in found
        if _stops_at_first(args)
    ]
    assert not offenders, "\n".join(offenders)


def test_part_checks_stage_helper_continues_and_exits_one_at_end(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[4]
    ci_full = (root / "scripts/ci_full.sh").read_text()
    part_checks = ci_full.split("part_checks() {", 1)[1].split("\npart_zoo()", 1)[0]
    stage_def = part_checks.split("# ── 3. Parity tier", 1)[0]
    summary_tail = '    if [ "${#failed_stages[@]}" -ne 0 ]; then' + (
        part_checks.split('    if [ "${#failed_stages[@]}" -ne 0 ]; then', 1)[1].rsplit(
            "\n}", 1
        )[0]
    )
    summary_file = tmp_path / "step_summary.md"
    script = (
        "set -euo pipefail\n"
        "run_test() {\n"
        + stage_def
        + '\nstage "First failing stage" bash -c "exit 1"\n'
        + 'stage "Second passing stage" printf "RAN_SECOND\\n"\n'
        + summary_tail
        + "\n}\nrun_test\n"
    )
    result = subprocess.run(
        ["bash", "-c", script],
        env={**os.environ, "GITHUB_STEP_SUMMARY": str(summary_file)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "RAN_SECOND" in result.stdout
    assert "First failing stage" in result.stderr
    assert "First failing stage" in summary_file.read_text()


def test_pyproject_test_extra_includes_quality_gate_collectors() -> None:
    """``[project.optional-dependencies].test`` must include lizard and vulture for unit tests."""
    import tomllib

    root = Path(__file__).resolve().parents[4]
    data = tomllib.loads((root / "pyproject.toml").read_text())
    test_deps = data["project"]["optional-dependencies"]["test"]
    quality_deps = data["dependency-groups"]["quality"]
    for pkg in ("lizard", "vulture"):
        pinned = next(d for d in quality_deps if d.startswith(f"{pkg}=="))
        assert (
            pinned in test_deps
        ), f"{pinned!r} missing from [project.optional-dependencies].test"
