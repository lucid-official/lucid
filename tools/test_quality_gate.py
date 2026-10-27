"""The ratchet gate fails on each seeded kind of slop and only ever lowers its baseline.

Each end-to-end case builds a scratch git repository with a small ``lucid/``
tree, writes its baseline with ``--rebaseline``, commits, seeds one
regression and runs the gate the way the hook and land do — as a process.

Not under ``lucid/test`` (the default ``testpaths``), so it runs on request::

    .venv/bin/python3 -m pytest tools/test_quality_gate.py
"""

import ast
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools.quality_gate import config
from tools.quality_gate.collectors.counters import count_python
from tools.quality_gate.core import Baseline, MergeRefused, merge_baselines
from tools.quality_gate.unlaunder import strip_ignores, strip_transitional, unwrap_casts

REPO = Path(__file__).resolve().parents[1]
FAST = "ruff,counters,lizard"

CLEAN = '''\
def add(a: int, b: int) -> int:
    """Sum."""
    return a + b
'''

BLOCK = "".join(
    f"    total_{i} = values[{i}] * weights[{i}] + offsets[{i}] - bias_{i}\n" for i in range(12)
)
DUPLICATED = f"def first(values, weights, offsets, bias_0):\n{BLOCK}    return 0\n\n\n" + (
    f"def second(values, weights, offsets, bias_0):\n{BLOCK}    return 1\n"
)


# ── unit: counters ──────────────────────────────────────────────────────────


def test_counters_read_comments_and_calls_not_strings() -> None:
    src = (
        "from typing import cast\n"
        "x = cast(int, y)  # type: ignore[arg-type, misc]\n"
        "z = 1  # type: ignore\n"
        "s = 'cast(int, y)  # type: ignore[override]'\n"
        "w = f(1)  # noqa: BLE001, S110\n"
        "v = 2  # noqa\n"
        "import typing\n"
        "from typing import Any\n"
        "a: Any = typing.Any  # Any in a comment is not counted\n"
        "t = 'Any'\n"
    )
    assert count_python(src) == {
        "cast": 1,
        "type-ignore[arg-type]": 1,
        "type-ignore[misc]": 1,
        "type-ignore-bare": 1,
        "noqa[BLE001]": 1,
        "noqa[S110]": 1,
        "noqa-bare": 1,
        "any": 3,
    }


# ── unit: the merge driver ──────────────────────────────────────────────────

A = ("ruff", "B905", "lucid/a.py")
B = ("ruff", "B905", "lucid/b.py")
REBASE = {"date": "2026-10-05", "reason": "new collector", "by": "orchestrator"}


def _b(
    counts: dict[tuple[str, str, str], int], history: list[dict[str, str]] | None = None
) -> Baseline:
    return Baseline(dict(counts), rebaselines=list(history or []))


def test_merge_disjoint_lowerings_keep_both() -> None:
    merged = merge_baselines(_b({A: 5, B: 5}), _b({A: 3, B: 5}), _b({A: 5, B: 2}))
    assert merged.counts == {A: 3, B: 2}


def test_merge_same_key_lowered_on_both_sides_takes_the_lower() -> None:
    merged = merge_baselines(_b({A: 5}), _b({A: 3}), _b({A: 4}))
    assert merged.counts == {A: 3}


def test_merge_delete_beats_lower() -> None:
    merged = merge_baselines(_b({A: 5, B: 1}), _b({B: 1}), _b({A: 4, B: 1}))
    assert merged.counts == {B: 1}


def test_merge_refuses_an_unauthorised_raise() -> None:
    with pytest.raises(MergeRefused):
        merge_baselines(_b({A: 5}), _b({A: 6}), _b({A: 5}))
    with pytest.raises(MergeRefused):
        merge_baselines(_b({}), _b({}), _b({B: 1}))


def test_merge_keeps_a_raise_carried_by_a_rebaseline() -> None:
    merged = merge_baselines(_b({A: 5}), _b({A: 4}), _b({A: 5, B: 9}, [REBASE]))
    assert merged.counts == {A: 4, B: 9}
    assert merged.rebaselines == [REBASE]


def test_merge_combines_deltas_when_one_side_carries_a_rebaseline() -> None:
    """LCD-522: when base has an unrecorded slack (27 vs 26), main rebaselines
    from 26 -> 27, and a side branch updates 27 -> 26, combining deltas
    (``o + t - b``) keeps the rebaselined 27 instead of taking ``min(27, 26) = 26``."""
    merged = merge_baselines(_b({A: 27, B: 31}), _b({A: 27, B: 28}, [REBASE]), _b({A: 26, B: 27}))
    assert merged.counts == {A: 26, B: 24}
    # And when base was stale (27 in baseline, 26 in code) and main rebaselined
    # after a formatting change (27 -> 27 in baseline) while side ran --update
    # before the formatting change (27 -> 26), or when base was 26, main
    # rebaselined to 27 (+1), and side lowered another key:
    merged2 = merge_baselines(_b({A: 26, B: 28}), _b({A: 27, B: 28}, [REBASE]), _b({A: 26, B: 27}))
    assert merged2.counts == {A: 27, B: 27}


# ── end to end ──────────────────────────────────────────────────────────────


def _env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(
        GIT_AUTHOR_NAME="t",
        GIT_AUTHOR_EMAIL="t@t",
        GIT_COMMITTER_NAME="t",
        GIT_COMMITTER_EMAIL="t@t",
        PYTHONPATH=str(REPO),
    )
    return env


def gate(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "tools.quality_gate", "--root", str(root), *args],
        cwd=REPO,
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )


def git(root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", *args],
        cwd=root,
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "lucid" / "pkg").mkdir(parents=True)
    (root / "tools").mkdir()
    (root / "lucid" / "pkg" / "mod.py").write_text(CLEAN)
    (root / "lucid" / "pkg" / "other.py").write_text(CLEAN + "x = 1  # type: ignore[misc]\n")
    git(root, "init", "-q", "-b", "main")
    done = gate(root, "--rebaseline", "--reason", "test", "--collectors", FAST)
    assert done.returncode == 0, done.stdout + done.stderr
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "base")
    return root


def _append(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.write_text(path.read_text() + text)


SEEDS = {
    "type-ignore": "\ny: int = 'a'  # type: ignore[assignment]\n",
    "except-pass": "\ntry:\n    add(1, 2)\nexcept Exception:\n    pass\n",
    "long-function": "\n\ndef long() -> int:\n    n = 0\n"
    + "    n += 1\n" * 200
    + "    return n\n",
    "any": "\nfrom typing import Any\n\nv: Any = 1\n",
}


@pytest.mark.parametrize("seed", sorted(SEEDS))
def test_fast_check_fails_on_seeded_slop(repo: Path, seed: str) -> None:
    assert gate(repo, "--fast").returncode == 0
    _append(repo, "lucid/pkg/mod.py", SEEDS[seed])
    out = gate(repo, "--fast")
    assert out.returncode == 1, out.stdout + out.stderr
    assert "rose above the baseline" in out.stdout


@pytest.mark.parametrize("seed", sorted(SEEDS))
def test_diff_reports_the_increase_against_a_ref(repo: Path, seed: str) -> None:
    _append(repo, "lucid/pkg/mod.py", SEEDS[seed])
    git(repo, "commit", "-qam", "slop")
    out = gate(repo, "--diff", "HEAD~1", "--collectors", FAST)
    assert out.returncode == 1, out.stdout + out.stderr
    assert "SLOP DELTA vs HEAD~1" in out.stdout
    assert "← INCREASE" in out.stdout


def test_trading_a_type_ignore_for_any_is_an_increase(repo: Path) -> None:
    """DS-16: 3681fd918 swapped 17 ``type: ignore`` for ``Any`` and the gate,
    which did not count ``Any``, read the swap as a decrease."""
    (repo / "lucid" / "pkg" / "other.py").write_text(
        CLEAN + "from typing import Any\n\nx: Any = 1\n"
    )
    out = gate(repo, "--fast")
    assert out.returncode == 1, out.stdout + out.stderr
    assert "+ counters  any " in out.stdout and "lucid/pkg/other.py  0 -> 2" in out.stdout
    assert "- counters  type-ignore[misc]" in out.stdout


def _function(lines: int, branches: int) -> str:
    body = "    if n:\n        n += 1\n" * branches + "    n += 1\n" * lines
    return f"def big(n: int) -> int:\n{body}    return n\n"


@pytest.mark.parametrize(
    ("rule", "before", "after"),
    [
        ("function-length-excess", _function(150, 0), _function(170, 0)),
        ("function-ccn-excess", _function(0, 20), _function(0, 25)),
    ],
)
def test_an_already_long_function_may_not_grow(
    repo: Path, rule: str, before: str, after: str
) -> None:
    """DS-29: counting the functions over a limit missed `_build_plan` growing
    425 -> 480 lines; the excess over the limit is what may not rise."""
    (repo / "lucid" / "pkg" / "big.py").write_text(before)
    assert gate(repo, "--rebaseline", "--reason", "t", "--collectors", FAST).returncode == 0
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "a long function")
    (repo / "lucid" / "pkg" / "big.py").write_text(after)
    # FAST alone: the repeated lines would be a jscpd clone besides.
    out = gate(repo, "--fast", "--collectors", FAST)
    assert out.returncode == 1, out.stdout + out.stderr
    rises = [line.split()[2] for line in out.stdout.splitlines() if line.startswith("    + ")]
    assert rises == [rule], out.stdout  # the count of long functions did not move
    git(repo, "commit", "-qam", "grow it")
    diff = gate(repo, "--diff", "HEAD~1", "--collectors", FAST)
    assert diff.returncode == 1 and rule in diff.stdout, diff.stdout


ENGINE_STUB = """\
def cube(x: int) -> int: ...
def cube_root(x: int) -> int: ...

class _nn_namespace:
    @staticmethod
    def bce_loss(x: int) -> int: ...
nn: _nn_namespace
"""

ENGINE_USER = """\
from lucid._C import engine as _C_engine


def f(x: int) -> int:
    return _C_engine.cube(_C_engine.cube_root(_C_engine.nn.bce_loss(x)))
"""


def test_dropping_the_last_caller_of_a_binding_is_an_increase(tmp_path: Path) -> None:
    """DS-24: a binding whose last production caller goes is a dead entry point."""
    root = tmp_path / "repo"
    (root / "lucid" / "_C").mkdir(parents=True)
    (root / "tools").mkdir()
    (root / "lucid" / "_C" / "engine.pyi").write_text(ENGINE_STUB)
    (root / "lucid" / "use.py").write_text(ENGINE_USER)
    git(root, "init", "-q", "-b", "main")
    done = gate(root, "--rebaseline", "--reason", "t", "--collectors", "engine")
    assert done.returncode == 0, done.stdout + done.stderr
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "base")
    assert gate(root, "--full", "--collectors", "engine").returncode == 0
    (root / "lucid" / "use.py").write_text(ENGINE_USER.replace("_C_engine.cube_root(", "("))
    out = gate(root, "--full", "--collectors", "engine")
    assert out.returncode == 1, out.stdout + out.stderr
    assert "+ engine    unbound" in out.stdout and "lucid/_C/engine.pyi  0 -> 1" in out.stdout
    git(root, "commit", "-qam", "drop a caller")
    assert gate(root, "--diff", "HEAD~1", "--collectors", "engine").returncode == 1


def _tamper_contract(root: Path, key: str, value: object) -> Path:
    """Record *value* for *key*, as a baseline written under another contract would."""
    path = root / "tools" / "quality_baseline.json"
    data = json.loads(path.read_text())
    data["thresholds"][key] = value
    path.write_text(json.dumps(data, indent=1) + "\n")
    git(root, "commit", "-qam", f"baseline recorded under another {key}")
    return path


def test_a_contract_change_needs_a_rebaseline(repo: Path) -> None:
    """DS-18: an edit that changes what the counts mean (a threshold, the
    vulture whitelist, an allowlist) may not reach the baseline as --update."""
    path = _tamper_contract(repo, "function_length_max", 90)
    out = gate(repo, "--fast")
    assert out.returncode == 1, out.stdout + out.stderr
    assert "contract differs" in out.stdout and "function_length_max: 90 -> 100" in out.stdout
    before = path.read_text()
    assert gate(repo, "--update", "--fast").returncode == 1
    assert path.read_text() == before
    diff = gate(repo, "--diff", "HEAD", "--collectors", FAST)
    assert diff.returncode == 1 and "1 contract change(s)" in diff.stdout, diff.stdout
    # Only the collector the setting belongs to is held to it ...
    assert gate(repo, "--fast", "--collectors", "ruff,counters").returncode == 0
    # ... and a rebaseline that did not measure it leaves its record alone.
    assert gate(repo, "--rebaseline", "--reason", "t", "--collectors", "ruff").returncode == 0
    assert json.loads(path.read_text())["thresholds"]["function_length_max"] == 90
    assert gate(repo, "--rebaseline", "--reason", "t", "--collectors", FAST).returncode == 0
    assert json.loads(path.read_text())["thresholds"]["function_length_max"] == 100
    assert gate(repo, "--fast").returncode == 0


def test_a_whitelist_change_holds_vulture_runs_only(repo: Path) -> None:
    _tamper_contract(repo, "vulture_whitelist_ast_sha256", "0" * 16)
    assert gate(repo, "--fast").returncode == 0  # pre-commit does not run vulture
    for mode in ("--full", "--update"):
        out = gate(repo, mode, "--collectors", "vulture")
        assert out.returncode == 1, out.stdout + out.stderr
        assert "vulture_whitelist_ast_sha256: '0000000000000000' ->" in out.stdout


def test_every_whitelist_entry_is_needed() -> None:
    """DS-18: vulture's used-name set is global, so an entry for a name Python
    code already uses hides nothing today and a namesake's death tomorrow.
    Every entry must be one vulture would report without the whitelist."""
    cmd = [
        sys.executable,
        "-m",
        "vulture",
        "lucid",
        f"--min-confidence={config.VULTURE_MIN_CONFIDENCE}",
        "--ignore-names=" + ",".join(config.VULTURE_IGNORE_NAMES),
        "--ignore-decorators=" + ",".join(config.VULTURE_IGNORE_DECORATORS),
    ]
    out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, check=False)
    assert out.returncode in (0, 3), out.stderr
    reported = {
        line.split("'")[1]
        for line in out.stdout.splitlines()
        if line.startswith("lucid/") and not line.startswith("lucid/test/") and "'" in line
    }
    tree = ast.parse(config.VULTURE_WHITELIST.read_text())
    entries = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert entries and not entries - reported, sorted(entries - reported)


def test_new_file_may_not_start_over_the_loc_cap(repo: Path) -> None:
    (repo / "lucid" / "pkg" / "big.py").write_text("x = 1\n" * 1600)
    out = gate(repo, "--fast")
    assert out.returncode == 1
    assert "file-loc-over-cap" in out.stdout


def test_a_decrease_must_be_recorded_and_update_only_lowers(repo: Path) -> None:
    (repo / "lucid" / "pkg" / "other.py").write_text(CLEAN)
    out = gate(repo, "--fast")
    assert out.returncode == 1 and "out of date" in out.stdout
    slack = gate(repo, "--fast", "--allow-slack")
    assert slack.returncode == 0 and "higher than the code" in slack.stdout
    assert gate(repo, "--update", "--fast").returncode == 0
    assert gate(repo, "--fast").returncode == 0
    data = json.loads((repo / "tools" / "quality_baseline.json").read_text())
    assert "type-ignore[misc]" not in data["counts"].get("counters", {})
    # A raise is refused and leaves the baseline untouched.
    before = (repo / "tools" / "quality_baseline.json").read_text()
    _append(repo, "lucid/pkg/mod.py", SEEDS["type-ignore"])
    assert gate(repo, "--update", "--fast").returncode == 1
    assert (repo / "tools" / "quality_baseline.json").read_text() == before


def test_a_move_is_not_new_debt(repo: Path) -> None:
    git(repo, "mv", "lucid/pkg/other.py", "lucid/pkg/moved.py")
    out = gate(repo, "--fast")
    assert "rose above" not in out.stdout, out.stdout
    assert "moved: lucid/pkg/other.py -> lucid/pkg/moved.py" in out.stdout
    assert gate(repo, "--update", "--fast").returncode == 0
    assert gate(repo, "--fast").returncode == 0
    git(repo, "commit", "-qam", "move")
    diff = gate(repo, "--diff", "HEAD~1", "--collectors", FAST)
    assert diff.returncode == 0, diff.stdout


@pytest.mark.skipif(
    shutil.which("npx") is None and not Path("/opt/homebrew/bin/npx").exists(),
    reason="jscpd runs through npx",
)
def test_duplicated_block_fails_full_and_diff(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "lucid").mkdir(parents=True)
    (root / "tools").mkdir()
    (root / "lucid" / "a.py").write_text(CLEAN)
    git(root, "init", "-q", "-b", "main")
    assert gate(root, "--rebaseline", "--reason", "t", "--collectors", "jscpd").returncode == 0
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "base")
    (root / "lucid" / "a.py").write_text(DUPLICATED)
    for mode in ("--full", "--fast"):
        out = gate(root, mode, "--collectors", "jscpd")
        assert out.returncode == 1 and "clone-pair" in out.stdout, out.stdout + out.stderr
    git(root, "commit", "-qam", "dup")
    assert gate(root, "--diff", "HEAD~1", "--collectors", "jscpd").returncode == 1


REDUNDANT_EXPR = '''\
def check(reduction: str) -> bool:
    """A validator whose ``str`` guard mypy proves is always false."""
    return not isinstance(reduction, str) or reduction == "mean"


print(check("sum"))
'''


@pytest.mark.skipif(
    shutil.which("npx") is None and not Path("/opt/homebrew/bin/npx").exists(),
    reason="the default collector set includes jscpd, which runs through npx",
)
def test_diff_counts_a_mypy_finding_in_a_new_file(tmp_path: Path) -> None:
    """DS-GATE-1: land's ``--diff`` once skipped mypy, so a new file's
    ``redundant-expr`` reached main and failed only CI's ``--full``.

    The default ``--diff`` must run what ``--full`` runs, and a file with no
    baseline row starts at 0, so its one finding is an increase.
    """
    root = tmp_path / "repo"
    (root / "lucid" / "pkg").mkdir(parents=True)
    (root / "tools").mkdir()
    for init in ("lucid/__init__.py", "lucid/pkg/__init__.py"):
        (root / init).write_text("")
    (root / "lucid" / "pkg" / "mod.py").write_text(CLEAN)
    git(root, "init", "-q", "-b", "main")
    done = gate(root, "--rebaseline", "--reason", "t")
    assert done.returncode == 0, done.stdout + done.stderr
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "base")
    (root / "lucid" / "pkg" / "checks.py").write_text(REDUNDANT_EXPR)
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "new file")

    out = gate(root, "--diff", "HEAD~1")
    assert out.returncode == 1, out.stdout + out.stderr
    assert "redundant-expr" in out.stdout and "lucid/pkg/checks.py" in out.stdout, out.stdout
    assert "0 over baseline" not in out.stdout  # the baseline has no row: 0, not "clean"
    full = gate(root, "--full")
    assert full.returncode == 1 and "redundant-expr" in full.stdout, full.stdout


def test_mypy_counts_an_error_in_a_tools_module_lucid_imports(tmp_path: Path) -> None:
    """LCD-317: ``mypy --strict lucid/`` follows lucid/'s imports into tools/
    and fails on what it finds there; the gate runs that same command, so it
    must count those findings too instead of keeping lucid/ files only."""
    root = tmp_path / "repo"
    (root / "lucid").mkdir(parents=True)
    (root / "tools").mkdir()
    (root / "lucid" / "__init__.py").write_text("from tools import helper\n")
    (root / "tools" / "helper.py").write_text(CLEAN)
    git(root, "init", "-q", "-b", "main")
    done = gate(root, "--rebaseline", "--reason", "t", "--collectors", "mypy")
    assert done.returncode == 0, done.stdout + done.stderr
    (root / "tools" / "helper.py").write_text(CLEAN + "\nwrong: int = add(1, 2) + 'a'\n")
    out = gate(root, "--full", "--collectors", "mypy")
    assert out.returncode == 1, out.stdout + out.stderr
    assert "tools/helper.py" in out.stdout, out.stdout


def test_a_finding_moved_into_a_new_file_is_an_increase(repo: Path) -> None:
    """Counts are never netted across files: only a git rename carries them."""
    (repo / "lucid" / "pkg" / "other.py").write_text(CLEAN)
    (repo / "lucid" / "pkg" / "fresh.py").write_text("x = 1  # type: ignore[misc]\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "move a finding")
    out = gate(repo, "--diff", "HEAD~1", "--collectors", FAST)
    assert out.returncode == 1, out.stdout
    assert "+ counters  type-ignore[misc]" in out.stdout and "lucid/pkg/fresh.py" in out.stdout


def test_rebase_merges_concurrent_lowerings_through_the_driver(tmp_path: Path) -> None:
    """Two branches lower neighbouring keys; the rebase neither conflicts nor loses either."""
    root = tmp_path / "repo"
    (root / "lucid" / "pkg").mkdir(parents=True)
    (root / "tools").mkdir()
    ignores = "".join(f"x{i} = 1  # type: ignore[misc]\n" for i in range(3))
    for name in ("a.py", "b.py"):
        (root / "lucid" / "pkg" / name).write_text(CLEAN + ignores)
    (root / ".gitattributes").write_text(
        "tools/quality_baseline.json merge=lucid-quality-baseline\n"
    )
    git(root, "init", "-q", "-b", "main")
    driver = f"{sys.executable} -m tools.quality_gate --merge-baseline %O %A %B"
    git(root, "config", "merge.lucid-quality-baseline.driver", driver)
    assert gate(root, "--rebaseline", "--reason", "t", "--collectors", FAST).returncode == 0
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "base")

    def lower(name: str, keep: int) -> None:
        kept = "x0 = 1  # type: ignore[misc]\n" * keep
        (root / "lucid" / "pkg" / name).write_text(CLEAN + kept)
        assert gate(root, "--update", "--collectors", FAST).returncode == 0
        git(root, "commit", "-qam", f"lower {name}")

    git(root, "checkout", "-q", "-b", "side")
    lower("a.py", 1)
    git(root, "checkout", "-q", "main")
    lower("b.py", 2)
    git(root, "checkout", "-q", "side")
    git(root, "rebase", "-q", "main")  # git() asserts it succeeded
    counts = json.loads((root / "tools" / "quality_baseline.json").read_text())["counts"]
    assert counts["counters"]["type-ignore[misc]"] == {"lucid/pkg/a.py": 1, "lucid/pkg/b.py": 2}
    assert gate(root, "--full", "--collectors", FAST).returncode == 0


def test_rebase_over_a_rebaseline_combines_deltas_without_hand_edit(tmp_path: Path) -> None:
    """LCD-522: replay the 2e88ab8bf + 8e0938a48 scenario where main rebaselines
    a file (3 -> 4) while a side branch lowers it by 1 (3 -> 2) and another file
    (3 -> 1), then rebases onto main without needing a hand edit."""
    root = tmp_path / "repo"
    (root / "lucid" / "pkg").mkdir(parents=True)
    (root / "tools").mkdir()

    def ig(n: int) -> str:
        return "".join(f"x{i} = 1  # type: ignore[misc]\n" for i in range(n))

    spacer = "".join(f"line_{i} = {i}\n" for i in range(8))
    (root / "lucid" / "pkg" / "a.py").write_text(CLEAN + spacer + ig(3))
    (root / "lucid" / "pkg" / "b.py").write_text(CLEAN + ig(3))
    (root / ".gitattributes").write_text(
        "tools/quality_baseline.json merge=lucid-quality-baseline\n"
    )
    git(root, "init", "-q", "-b", "main")
    driver = f"{sys.executable} -m tools.quality_gate --merge-baseline %O %A %B"
    git(root, "config", "merge.lucid-quality-baseline.driver", driver)
    assert gate(root, "--rebaseline", "--reason", "init", "--collectors", FAST).returncode == 0
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "base")

    # Side branch lowers a.py (3 -> 2) at the bottom and b.py (3 -> 1) via --update:
    git(root, "checkout", "-q", "-b", "side")
    (root / "lucid" / "pkg" / "a.py").write_text(CLEAN + spacer + ig(2))
    (root / "lucid" / "pkg" / "b.py").write_text(CLEAN + ig(1))
    assert gate(root, "--update", "--collectors", FAST).returncode == 0
    git(root, "commit", "-qam", "side lowers a and b")

    # Main rebaselines a.py (3 -> 4) at the top with --rebaseline:
    git(root, "checkout", "-q", "main")
    (root / "lucid" / "pkg" / "a.py").write_text(
        "z = 1  # type: ignore[misc]\n" + CLEAN + spacer + ig(3)
    )
    assert (
        gate(root, "--rebaseline", "--reason", "main rebaseline", "--collectors", FAST).returncode
        == 0
    )
    git(root, "commit", "-qam", "main rebaseline")

    # Rebase side onto main: merged a.py has 3 ignores (2 + 4 - 3 = 3), b.py has 1:
    git(root, "checkout", "-q", "side")
    git(root, "rebase", "-q", "main")
    counts = json.loads((root / "tools" / "quality_baseline.json").read_text())["counts"]
    assert counts["counters"]["type-ignore[misc]"] == {"lucid/pkg/a.py": 3, "lucid/pkg/b.py": 1}
    assert gate(root, "--full", "--collectors", FAST).returncode == 0
    assert gate(root, "--diff", "main", "--collectors", FAST).returncode == 0


def test_unrecorded_baseline_raise_fails_check_update_and_diff(repo: Path) -> None:
    """LCD-522: raising a baseline count without a new ``rebaselines`` entry
    is refused by ``check``, ``--update``, and ``--diff``."""
    _append(repo, "lucid/pkg/other.py", "y = 2  # type: ignore[misc]\n")
    bl_path = repo / "tools" / "quality_baseline.json"
    data = json.loads(bl_path.read_text())
    data["counts"]["counters"]["type-ignore[misc]"]["lucid/pkg/other.py"] = 2
    bl_path.write_text(json.dumps(data, indent=1) + "\n")

    for args in (("--fast",), ("--full", "--collectors", FAST), ("--update", "--fast")):
        out = gate(repo, *args)
        assert out.returncode == 1, out.stdout + out.stderr
        assert "without a rebaselines entry" in out.stdout

    git(repo, "commit", "-qam", "hand-raised baseline")
    diff = gate(repo, "--diff", "HEAD~1", "--collectors", FAST)
    assert diff.returncode == 1, diff.stdout + diff.stderr
    assert "without a rebaselines entry" in diff.stdout


def test_malformed_prior_baseline_fails_closed(repo: Path) -> None:
    """LCD-522: a corrupted or schema-invalid baseline committed at *ref*
    must fail the gate with ``GateError`` (exit 2), never fail open as ``{}``."""
    bl_path = repo / "tools" / "quality_baseline.json"
    good_text = bl_path.read_text()

    for bad_text in ("{not json\n", json.dumps({"version": 999}) + "\n", ""):
        bl_path.write_text(bad_text)
        git(repo, "commit", "-qam", "corrupt baseline at HEAD")
        bl_path.write_text(good_text)
        for args in (("--fast",), ("--update", "--fast"), ("--diff", "HEAD", "--collectors", FAST)):
            out = gate(repo, *args)
            assert out.returncode == 2, out.stdout + out.stderr
            assert "cannot measure: " in out.stderr
            assert "tools/quality_baseline.json" in out.stderr


@pytest.mark.skipif(
    shutil.which("npx") is None and not Path("/opt/homebrew/bin/npx").exists(),
    reason="jscpd runs through npx",
)
def test_jscpd_measures_cpp_headers_and_cross_format_clones(tmp_path: Path) -> None:
    """LCD-522: jscpd scans ``.h``/``.hpp`` headers and detects header↔``.cpp`` clones."""
    root = tmp_path / "repo"
    (root / "lucid" / "_C").mkdir(parents=True)
    (root / "tools").mkdir()
    cpp_block = "".join(
        f"    acc_{i} = lhs[{i}] * rhs[{i}] + bias[{i}] - scale_{i};\n" for i in range(12)
    )
    fn_body = f"int compute(const int* lhs, const int* rhs, const int* bias, int scale_0) {{\n{cpp_block}    return 0;\n}}\n"
    (root / "lucid" / "_C" / "a.h").write_text(fn_body)
    (root / "lucid" / "_C" / "b.cpp").write_text("int ok() { return 1; }\n")
    git(root, "init", "-q", "-b", "main")
    assert gate(root, "--rebaseline", "--reason", "t", "--collectors", "jscpd").returncode == 0
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "base")

    # Duplicate the header function into a .cpp file (cross-format c-header <-> cpp):
    (root / "lucid" / "_C" / "b.cpp").write_text(fn_body)
    out = gate(root, "--fast", "--collectors", "jscpd")
    assert out.returncode == 1, out.stdout + out.stderr
    assert "lucid/_C/a.h" in out.stdout and "lucid/_C/b.cpp" in out.stdout


# ── unlaunder: the mypy-driven codemod ──────────────────────────────────────


def test_unlaunder_strips_only_the_marked_sections() -> None:
    ini = (
        "[mypy]\nstrict = True\n\n"
        "# transitional (LCD-260) — removed later.\n# more words\n"
        "[mypy-lucid.pkg.*]\nwarn_unused_ignores = False\n\n"
        "[mypy-lucid.keep.*]\nignore_errors = True\n"
        "# transitional (LCD-260) is a stray marker here\nfoo = 1\n"
        "[mypy-lucid.also_kept]\nignore_errors = True\n"
    )
    out = strip_transitional(ini)
    assert "[mypy-lucid.pkg.*]" not in out and "warn_unused_ignores" not in out
    assert "[mypy-lucid.keep.*]" in out and "[mypy-lucid.also_kept]" in out
    assert "strict = True" in out


def test_unlaunder_strips_a_section_marked_with_any_issue() -> None:
    """LCD-315: the marker is ``# transitional (<issue>)``, not LCD-260 alone."""
    ini = (
        "[mypy]\nstrict = True\n\n"
        "# transitional (LCD-999) — another owner's type change.\n"
        "[mypy-lucid.other.*]\nwarn_unused_ignores = False\n\n"
        "# transitional (see LCD-999) is not the marker\n"
        "[mypy-lucid.kept]\nignore_errors = True\n"
    )
    out = strip_transitional(ini)
    assert "[mypy-lucid.other.*]" not in out
    assert "[mypy-lucid.kept]" in out


@pytest.mark.parametrize(
    ("line", "unused", "after"),
    [
        ("x = 1  # type: ignore\n", None, "x = 1\n"),
        ("x = 1  # type: ignore[misc]\n", None, "x = 1\n"),
        ("x = 1  # type: ignore[a, b]\n", {"a"}, "x = 1  # type: ignore[b]\n"),
        ("x = 1  # type: ignore[a, b]  # why\n", {"a"}, "x = 1  # type: ignore[b]  # why\n"),
        ("x = 1  # type: ignore[a, b]\n", {"a", "b"}, "x = 1\n"),
        ("x = 1  # type: ignore[misc]  # it is fine at runtime\n", None, "x = 1\n"),
        ("x = 1  # type: ignore # noqa: F821\n", None, "x = 1  # noqa: F821\n"),
        ("x = 1  # noqa: E501  # type: ignore[misc]\n", None, "x = 1  # noqa: E501\n"),
        ("s = '# type: ignore'  # type: ignore\n", None, "s = '# type: ignore'\n"),
    ],
)
def test_unlaunder_removes_only_the_unused_codes(
    line: str, unused: set[str] | None, after: str
) -> None:
    out, skipped = strip_ignores(line, {1: None if unused is None else frozenset(unused)})
    assert (out, skipped) == (after, [])


def test_unlaunder_leaves_an_ignore_whose_codes_do_not_match() -> None:
    out, skipped = strip_ignores("x = 1  # type: ignore[misc]\n", {1: frozenset({"override"})})
    assert out == "x = 1  # type: ignore[misc]\n" and len(skipped) == 1


def _sites(src: str) -> set[tuple[int, int]]:
    """Every cast call's (line, byte column), as mypy reports them."""
    return {
        (n.lineno, n.col_offset)
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", "")) == "cast"
    }


@pytest.mark.parametrize(
    ("src", "after"),
    [
        ("y = cast(int, cast(int, x)) + cast(int, x)\n", "y = x + x\n"),
        ("y = cast(\n    int,\n    f(\n        x,\n    ),\n)\n", "y = f(\n        x,\n    )\n"),
        ("y = cast(int, a + b) * 2\n", "y = (a + b) * 2\n"),
        ("y = cast(int, a + b)\n", "y = a + b\n"),
        ("f(cast(int, a if b else c))\n", "f(a if b else c)\n"),
        ("y = cast(int, a or b).bit_length()\n", "y = (a or b).bit_length()\n"),
        ("y = cast(int, a\n    + b)\n", "y = (a\n    + b)\n"),
        ("f(cast(int, a\n    + b))\n", "f(a\n    + b)\n"),
        ("y = -cast(int, (z := 1))\n", "y = -(z := 1)\n"),
        ("y = typing.cast(int, x)\n", "y = x\n"),
        ("y = cast(typ=int, val=x)\n", "y = x\n"),
        ("s = 'é'; y = cast(int, x)\n", "s = 'é'; y = x\n"),
    ],
)
def test_unlaunder_unwraps_casts_without_changing_the_code(src: str, after: str) -> None:
    out, n, skipped = unwrap_casts(src, _sites(src))
    assert (out, skipped) == (after, [])
    assert n == len(_sites(src))


def test_unlaunder_leaves_a_cast_whose_comment_would_be_lost() -> None:
    src = "y = cast(\n    int,  # the engine returns int here\n    x,\n)\n"
    out, n, skipped = unwrap_casts(src, _sites(src))
    assert out == src and n == 0
    assert skipped == ["1: a comment inside the cast would be lost"]


def test_unlaunder_unwraps_only_the_reported_cast_of_a_nest() -> None:
    src = "y = cast(int, cast(str, x))\n"
    out, n, _ = unwrap_casts(src, {(1, 14)})
    assert (out, n) == ("y = cast(int, x)\n", 1)


UNLAUNDER_INI = """\
[mypy]
python_version = 3.14
strict = True

# transitional (LCD-260) — keeps the package green until its sweep lands.
[mypy-lucid.pkg.*]
warn_unused_ignores = False
disable_error_code = redundant-cast
"""

UNLAUNDER_MOD = """\
from typing import cast


def f(x: int, s: str) -> int:
    a = cast(int, cast(int, x)) + cast(int, x)
    b = cast(
        int,
        x,
    )
    c: int = s  # type: ignore[assignment, misc]
    d = 1  # type: ignore[misc]  # the reason this was needed
    e = 2  # type: ignore # noqa: E501
    return a + b + c + d + e
"""

UNLAUNDER_DONE = """\
def f(x: int, s: str) -> int:
    a = x + x
    b = x
    c: int = s  # type: ignore[assignment]
    d = 1
    e = 2  # noqa: E501
    return a + b + c + d + e
"""


def unlaunder(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "tools.quality_gate.unlaunder", "--root", str(root), *args],
        cwd=REPO,
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.skipif(
    importlib.util.find_spec("black") is None and shutil.which("black") is None,
    reason="unlaunder reformats with black",
)
def test_unlaunder_end_to_end_reaches_a_fixpoint(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    for pkg in ("lucid/pkg", "lucid/other"):
        (root / pkg).mkdir(parents=True)
        (root / pkg / "__init__.py").write_text("")
    (root / "lucid" / "__init__.py").write_text("")
    (root / "mypy.ini").write_text(UNLAUNDER_INI)
    (root / "lucid" / "pkg" / "mod.py").write_text(UNLAUNDER_MOD)
    (root / "lucid" / "other" / "mod.py").write_text(UNLAUNDER_MOD)

    # A path mypy never reports on would read as "0 left": refused instead.
    (root / "tools").mkdir()
    (root / "lucid" / "test").mkdir()
    for wrong in ("lucid/nope", "tools", "lucid/test", str(tmp_path)):
        assert unlaunder(root, "--check", "--paths", wrong).returncode == 2, wrong
    check = unlaunder(root, "--check", "--paths", str(root / "lucid" / "pkg"))
    assert check.returncode == 1, check.stdout + check.stderr
    assert "2 redundant-cast, 3 unused-ignore" in check.stdout  # one per line
    assert (root / "lucid" / "pkg" / "mod.py").read_text() == UNLAUNDER_MOD

    run = unlaunder(root, "--paths", "lucid/pkg")
    assert run.returncode == 0, run.stdout + run.stderr
    assert "round 2:" in run.stdout  # mypy reports one cast per line and message
    assert (root / "lucid" / "pkg" / "mod.py").read_text() == UNLAUNDER_DONE
    assert (root / "lucid" / "other" / "mod.py").read_text() == UNLAUNDER_MOD
    done = unlaunder(root, "--check", "--paths", "lucid/pkg")
    assert done.returncode == 0 and "0 redundant-cast, 0 unused-ignore" in done.stdout


@pytest.mark.skipif(
    shutil.which("clang-tidy") is None
    and not (Path(sys.executable).parent / "clang-tidy").is_file(),
    reason="clang-tidy is not installed",
)
def test_clang_tidy_collector_counts_seeded_cpp_regression(tmp_path: Path) -> None:
    """DS-3 (LCD-262): clang-tidy counts findings from compile_commands.json."""
    root = tmp_path / "repo"
    (root / "lucid" / "_C").mkdir(parents=True)
    (root / "tools").mkdir()
    (root / "build").mkdir()
    cpp_file = root / "lucid" / "_C" / "sample.cpp"
    cpp_file.write_text("int clean(int x) { return x + 1; }\n")
    sdk = subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip()
    (root / "build" / "compile_commands.json").write_text(
        json.dumps(
            [
                {
                    "directory": str(root),
                    "command": f"/usr/bin/clang++ -std=c++20 -isysroot {sdk} -c lucid/_C/sample.cpp",
                    "file": str(cpp_file),
                }
            ]
        )
    )
    git(root, "init", "-q", "-b", "main")
    done = gate(root, "--rebaseline", "--reason", "t", "--collectors", "clang-tidy")
    assert done.returncode == 0, done.stdout + done.stderr
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "base")

    cpp_file.write_text("int bad() { int uninit; return uninit; }\n")
    out = gate(root, "--full", "--collectors", "clang-tidy")
    assert out.returncode == 1, out.stdout + out.stderr
    assert "cppcoreguidelines-init-variables" in out.stdout
    assert "lucid/_C/sample.cpp" in out.stdout


@pytest.mark.skipif(
    shutil.which("semgrep") is None and not (Path(sys.executable).parent / "semgrep").is_file(),
    reason="semgrep is not installed",
)
def test_semgrep_error_taxonomy_rules_catch_seeded_violations(tmp_path: Path) -> None:
    """DS-4 (LCD-263): semgrep rules catch out-of-range ValueError, inline
    loss reduction checks, and inline nn.utils parameter checks."""
    root = tmp_path / "repo"
    (root / "lucid" / "nn" / "utils").mkdir(parents=True)
    (root / "tools").mkdir()
    (root / "lucid" / "ops.py").write_text(CLEAN)
    (root / "lucid" / "nn" / "loss.py").write_text(CLEAN)
    (root / "lucid" / "nn" / "utils" / "norm.py").write_text(CLEAN)
    git(root, "init", "-q", "-b", "main")
    done = gate(root, "--rebaseline", "--reason", "t", "--collectors", "semgrep")
    assert done.returncode == 0, done.stdout + done.stderr
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "base")

    (root / "lucid" / "ops.py").write_text(
        'def check(dim: int) -> None:\n    raise ValueError(f"op: dim {dim} is out of range")\n'
    )
    (root / "lucid" / "nn" / "loss.py").write_text(
        'def check(r: str) -> None:\n    if r not in ("none", "mean", "sum"):\n        raise ValueError(r)\n'
    )
    (root / "lucid" / "nn" / "utils" / "norm.py").write_text(
        "def check(m: object, name: str) -> None:\n"
        "    if not hasattr(m, name):\n"
        '        raise AttributeError(f"module has no parameter {name}")\n'
    )
    out = gate(root, "--full", "--collectors", "semgrep")
    assert out.returncode == 1, out.stdout + out.stderr
    assert "lucid-dim-range-value-error" in out.stdout
    assert "lucid-inline-reduction-check" in out.stdout
    assert "lucid-nn-utils-inline-param-check" in out.stdout


def test_backend_methods_loc_cap_enforced_at_1500_lines(tmp_path: Path) -> None:
    """DS-5 (LCD-264): backend/{cpu,gpu}/methods/*.cpp files use the 1,500-line cap."""
    root = tmp_path / "repo"
    (root / "lucid" / "_C" / "backend" / "cpu" / "methods").mkdir(parents=True)
    (root / "tools").mkdir()
    m_cpp = root / "lucid" / "_C" / "backend" / "cpu" / "methods" / "Binary.cpp"
    hdr = root / "lucid" / "_C" / "backend" / "cpu" / "CpuBackend.h"
    m_cpp.write_text("// ok\n" * config.LOC_CAP_BACKEND_METHODS)
    hdr.write_text("// ok\n" * (config.LOC_CAP_BACKEND_METHODS + 100))
    git(root, "init", "-q", "-b", "main")
    done = gate(root, "--rebaseline", "--reason", "t", "--collectors", "counters")
    assert done.returncode == 0, done.stdout + done.stderr
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "base")

    m_cpp.write_text("// over\n" * (config.LOC_CAP_BACKEND_METHODS + 5))
    out = gate(root, "--full", "--collectors", "counters")
    assert out.returncode == 1, out.stdout + out.stderr
    assert "file-loc-over-cap" in out.stdout
    assert "lucid/_C/backend/cpu/methods/Binary.cpp" in out.stdout
