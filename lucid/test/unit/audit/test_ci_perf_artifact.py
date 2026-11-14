"""The phase gate's perf tier leaves its timings behind as an artifact (LCD-1006).

Only the nightly uploaded ``perf-history-<n>``, so a ceiling for a case added
recently was recorded from three to ten runs, while the checks part's perf tier
ran on every push and printed its numbers to a log that expires.  The script
writes the plugin's JSON where ``LUCID_CI_PERF_JSON`` says, the workflow uploads
it as ``perf-core-<run>-<attempt>``, and the tier stays warn-only throughout.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
_PERF_TIER_ARGS = ["-m", "pytest", "lucid/test/perf/", "-m", "perf", "--tb=short", "-q"]


def _phase_gate() -> str:
    """The text of the ``phase-gate`` job in ci.yml."""
    text = (ROOT / ".github/workflows/ci.yml").read_text()
    return text.split("\n  phase-gate:\n", 1)[1].split("\n  mlx-compat:\n", 1)[0]


def _upload_step() -> str:
    """The step of ``phase-gate`` that uploads the perf timings."""
    steps = re.split(r"\n      - ", _phase_gate())
    found = [s for s in steps if "name: perf-core-" in s]
    assert len(found) == 1, f"expected one perf-core upload step, found {len(found)}"
    return found[0]


def test_the_workflow_and_the_script_agree_on_the_file() -> None:
    gate = _phase_gate()
    written = re.search(r"^      LUCID_CI_PERF_JSON: (\S+)$", gate, re.MULTILINE)
    assert written, "phase-gate must set LUCID_CI_PERF_JSON for scripts/ci_full.sh"
    assert "path: ${{ env.LUCID_CI_PERF_JSON }}" in _upload_step()
    assert "LUCID_CI_PERF_JSON" in (ROOT / "scripts/ci_full.sh").read_text()


def test_the_upload_survives_a_red_run_but_not_a_cancelled_one() -> None:
    step = _upload_step()
    condition = re.search(r"^        if: (.+)$", step, re.MULTILINE)
    assert condition, step
    assert "!cancelled()" in condition[1]
    assert "always()" not in condition[1]  # a cancelled run leaves an empty or partial file
    assert "matrix.part == 'checks'" in condition[1]  # the one part that runs the perf tier
    assert "pull_request" in condition[1]  # a branch's timings are not the ceilings' code
    assert "if-no-files-found: ignore" in step  # a part that died before the tier is not a failure


def test_the_artifact_is_named_and_kept_so_the_recorder_can_use_it() -> None:
    step = _upload_step()
    name = re.search(r"^          name: (.+)$", step, re.MULTILINE)
    assert name
    # `perf-` is the prefix the recorder's download flow lists by (the artifacts API
    # filters by exact name only); the run attempt keeps a re-run from colliding.
    assert name[1] == "perf-core-${{ github.run_id }}-${{ github.run_attempt }}"
    days = re.search(r"^          retention-days: (\d+)$", step, re.MULTILINE)
    assert days and int(days[1]) <= 14
    nightly = (ROOT / ".github/workflows/ci.yml").read_text().split("\n  nightly-perf:\n", 1)[1]
    assert re.search(r"^          name: perf-history-", nightly, re.MULTILINE), (
        "the nightly's artifacts must share the `perf-` prefix, or one listing cannot find both"
    )


def _perf_block() -> str:
    script = (ROOT / "scripts/ci_full.sh").read_text()
    return script.split('echo "==> Perf tier"', 1)[1].split("# ── 5. Validator tools", 1)[0]


def _run_perf_block(
    tmp_path: Path, *, json_path: Path | None, pytest_status: int, writes: str | None
) -> tuple[subprocess.CompletedProcess[str], list[list[str]]]:
    """The perf-tier block of ``part_checks`` against a stand-in for python.

    The stand-in logs every call.  ``-m pytest`` writes ``writes`` to the file named by
    ``--benchmark-json=`` (as the plugin does) and exits ``pytest_status``; ``-c`` runs
    the real interpreter, so the block's own readability check is the real code.
    """
    calls = tmp_path / "calls.jsonl"
    fake = tmp_path / "python"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        f"with open({str(calls)!r}, 'a') as log:\n"
        "    log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if sys.argv[1] == '-c':\n"
        f"    os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])\n"
        "for arg in sys.argv:\n"
        "    if arg.startswith('--benchmark-json='):\n"
        "        with open(arg.split('=', 1)[1], 'w') as out:\n"
        "            out.write(os.environ['TEST_PERF_WRITES'])\n"
        "sys.exit(int(os.environ['TEST_PERF_STATUS']))\n"
    )
    fake.chmod(0o700)
    env = {
        **os.environ,
        "PYTHON_BIN": str(fake),
        "TEST_PERF_STATUS": str(pytest_status),
        "TEST_PERF_WRITES": writes or "",
    }
    env.pop("LUCID_CI_PERF_JSON", None)
    if json_path is not None:
        env["LUCID_CI_PERF_JSON"] = str(json_path)
    result = subprocess.run(
        ["bash", "-c", f"set -euo pipefail\nperf() {{\n{_perf_block()}\n}}\nperf\n"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    logged = [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
    return result, logged


def test_without_the_variable_the_perf_tier_is_what_it_was(tmp_path: Path) -> None:
    result, calls = _run_perf_block(tmp_path, json_path=None, pytest_status=0, writes=None)
    assert result.returncode == 0, result.stderr
    assert calls == [_PERF_TIER_ARGS]


def test_with_the_variable_the_plugin_writes_there_and_the_file_is_kept(tmp_path: Path) -> None:
    path = tmp_path / "build" / "perf" / "pytest_benchmark.json"  # the directory does not exist
    doc = json.dumps({"benchmarks": []})
    result, calls = _run_perf_block(tmp_path, json_path=path, pytest_status=0, writes=doc)
    assert result.returncode == 0, result.stderr
    assert calls[0] == [*_PERF_TIER_ARGS, f"--benchmark-json={path}"]
    assert path.read_text() == doc


@pytest.mark.parametrize("writes", ['{"benchmarks": []}', ""])
def test_a_failing_perf_tier_still_only_warns(tmp_path: Path, writes: str) -> None:
    path = tmp_path / "perf.json"
    result, _ = _run_perf_block(tmp_path, json_path=path, pytest_status=1, writes=writes)
    assert result.returncode == 0, result.stderr
    assert "[WARN] Perf tier failed" in result.stdout
    # A red run's timings are kept; an empty file (the process died) is not an artifact.
    assert path.exists() == bool(writes)


@pytest.mark.parametrize("writes", ["", "{", '{"machine_info": {}}'])
def test_an_unreadable_benchmark_file_is_removed_not_uploaded(tmp_path: Path, writes: str) -> None:
    path = tmp_path / "perf.json"
    result, _ = _run_perf_block(tmp_path, json_path=path, pytest_status=0, writes=writes)
    assert result.returncode == 0, result.stderr
    assert not path.exists()
    assert "nothing to upload" in result.stdout
