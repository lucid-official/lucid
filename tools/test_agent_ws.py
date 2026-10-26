"""The claim gate judges every tree by that tree's own index and config.

git exports repository-local variables to the hooks it runs: GIT_INDEX_FILE
on every commit (relative ``.git/index`` in the main checkout, a temporary
``next-index-*.lock`` for ``git commit -- <path>``) and GIT_DIR from a linked
worktree.  A gate that hands them on to its queries of the *other* trees reads
the committing tree's index and labels there instead, and refuses every commit
with the committing tree named as the owner.  So these cases drive real commits
through a pre-commit hook in a scratch repository — main plus two linked
worktrees — once staged with ``git add`` and once as a partial commit, which is
also what proves the gate still reads the commit's own staged set from the
index git handed it.

Not under ``lucid/test`` (the default ``testpaths``), so it runs on request::

    .venv/bin/python3 -m pytest tools/test_agent_ws.py
"""

import importlib.util
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

AGENT_WS = Path(__file__).resolve().with_name("agent_ws.py")
MODES = ("staged", "partial")


@dataclass
class Scratch:
    main: Path
    a: Path
    b: Path
    env: dict[str, str] = field(repr=False)

    def git(self, cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            env=self.env,
            capture_output=True,
            text=True,
            check=False,
        )

    def edit(self, tree: Path, name: str) -> None:
        path = tree / name
        path.write_text(path.read_text() + f"{tree.name}\n")

    def commit(
        self, tree: Path, name: str, mode: str
    ) -> subprocess.CompletedProcess[str]:
        """Change ``name`` in ``tree`` and commit just that file through the hook."""
        self.edit(tree, name)
        message = f"{tree.name}: {name}"
        if mode == "staged":
            assert self.git(tree, "add", name).returncode == 0
            return self.git(tree, "commit", "-q", "-m", message)
        # `git commit -- <path>` builds a temporary index and hands it to the
        # hook as GIT_INDEX_FILE; the worktree's own index has nothing staged.
        return self.git(tree, "commit", "-q", "-m", message, "--", name)


@pytest.fixture
def scratch(tmp_path: Path) -> Scratch:
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("GIT_") and k != "LUCID_SKIP_CLAIMS"
    }
    env.update(
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_CONFIG_NOSYSTEM="1",
        LUCID_AGENT_STATE=str(tmp_path / "state"),
    )
    main = tmp_path / "main"
    main.mkdir()
    scratch = Scratch(main, tmp_path / "wt-a", tmp_path / "wt-b", env)

    def run(cwd: Path, *args: str) -> None:
        proc = subprocess.run(
            list(args), cwd=cwd, env=env, capture_output=True, text=True, check=False
        )
        assert proc.returncode == 0, f"{args}: {proc.stderr}"

    run(main, "git", "init", "-q", "-b", "main")
    run(main, "git", "config", "user.name", "scratch")
    run(main, "git", "config", "user.email", "scratch@example.invalid")
    # As in the Lucid repository: `agent_ws.py label` writes `--worktree`
    # config, which is per worktree only with this extension on.
    run(main, "git", "config", "extensions.worktreeConfig", "true")
    for name in ("a.txt", "b.txt", "c.txt"):
        (main / name).write_text(f"{name}\n")
    run(main, "git", "add", ".")
    run(main, "git", "commit", "-q", "-m", "init")

    hooks = tmp_path / "hooks"
    hooks.mkdir()
    hook = hooks / "pre-commit"
    hook.write_text(
        "#!/bin/sh\n"
        f"exec {shlex.quote(sys.executable)} {shlex.quote(str(AGENT_WS))} check-commit\n"
    )
    hook.chmod(0o755)
    run(main, "git", "config", "core.hooksPath", str(hooks))

    for tree, branch in ((scratch.a, "task-a"), (scratch.b, "task-b")):
        run(main, "git", "worktree", "add", "-q", "-b", branch, str(tree))
    run(scratch.a, sys.executable, str(AGENT_WS), "label", "label of A")
    run(scratch.b, sys.executable, str(AGENT_WS), "label", "label of B")
    return scratch


@pytest.mark.parametrize("mode", MODES)
def test_worktree_commits_a_file_nobody_else_changed(
    scratch: Scratch, mode: str
) -> None:
    scratch.edit(scratch.b, "b.txt")  # B is busy elsewhere
    result = scratch.commit(scratch.a, "a.txt", mode)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("mode", MODES)
def test_worktree_commit_of_a_file_another_worktree_changed_names_that_worktree(
    scratch: Scratch, mode: str
) -> None:
    assert scratch.commit(scratch.a, "a.txt", mode).returncode == 0
    scratch.edit(scratch.b, "a.txt")
    result = scratch.commit(scratch.a, "a.txt", mode)
    assert result.returncode != 0
    # B and only B, under B's own label — not the committing tree's.
    assert "a.txt is claimed by worktree wt-b [task-b] — label of B. " in result.stderr
    assert "label of A" not in result.stderr


@pytest.mark.parametrize("mode", MODES)
def test_main_commits_a_file_no_worktree_changed(scratch: Scratch, mode: str) -> None:
    scratch.edit(scratch.a, "a.txt")
    scratch.edit(scratch.b, "b.txt")
    result = scratch.commit(scratch.main, "c.txt", mode)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("mode", MODES)
def test_main_commit_of_a_file_a_worktree_changed_names_that_worktree(
    scratch: Scratch, mode: str
) -> None:
    scratch.edit(scratch.a, "a.txt")
    scratch.edit(scratch.b, "b.txt")
    result = scratch.commit(scratch.main, "b.txt", mode)
    assert result.returncode != 0
    assert "b.txt is claimed by worktree wt-b [task-b] — label of B. " in result.stderr


# ── land: the audit baselines (LCD-358) ───────────────────────────────────────

COVERAGE = "lucid/test/audit/coverage.json"


def commit_coverage(scratch: Scratch, cells: int) -> None:
    """Commit, in A, a coverage baseline whose header claims ``cells`` over two entries."""
    path = scratch.a / COVERAGE
    path.parent.mkdir(parents=True, exist_ok=True)
    answered = {"grad::lucid.exp": "pass", "grad::lucid.log": "pass"}
    path.write_text(json.dumps({"cells": cells, "answered": answered}))
    assert scratch.git(scratch.a, "add", COVERAGE).returncode == 0
    commit = scratch.git(scratch.a, "commit", "-q", "-m", "record coverage")
    assert commit.returncode == 0, commit.stderr


def land(scratch: Scratch, *flags: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(AGENT_WS), "land", str(scratch.a), "--no-changelog", *flags],
        cwd=scratch.main,
        env=scratch.env,
        capture_output=True,
        text=True,
        check=False,
    )


def main_tip(scratch: Scratch) -> str:
    return scratch.git(scratch.main, "rev-parse", "main").stdout.strip()


@pytest.mark.parametrize("flags", [(), ("--check",)])
def test_land_refuses_an_audit_baseline_whose_total_disagrees_with_its_body(
    scratch: Scratch, flags: tuple[str, ...]
) -> None:
    before = main_tip(scratch)
    commit_coverage(scratch, cells=3)
    result = land(scratch, *flags)
    output = result.stdout + result.stderr
    assert result.returncode == 7, output
    assert "'cells' is 3 but 'answered' holds 2 entries" in output
    assert main_tip(scratch) == before


def test_land_takes_an_audit_baseline_that_agrees_with_itself(scratch: Scratch) -> None:
    commit_coverage(scratch, cells=2)
    result = land(scratch)
    assert result.returncode == 0, result.stdout + result.stderr
    assert scratch.git(scratch.main, "cat-file", "-e", f"main:{COVERAGE}").returncode == 0


def test_worktree_venv_resolves_lucid_and_tools_without_the_root_on_sys_path(
    tmp_path: Path,
) -> None:
    """LCD-317: mypy silences errors in modules under any ``sys.path`` entry,
    so the worktree venv maps ``lucid`` and ``tools`` with a finder instead of
    putting the worktree root on ``sys.path``."""
    spec = importlib.util.spec_from_file_location("agent_ws", AGENT_WS)
    assert spec is not None and spec.loader is not None
    agent_ws = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent_ws)
    root, site, main_site = tmp_path / "wt", tmp_path / "site", tmp_path / "main-site"
    for d in (root / "lucid", root / "tools", site, main_site):
        d.mkdir(parents=True)
    (root / "lucid" / "__init__.py").write_text("")
    (root / "tools" / "helper.py").write_text("")
    for name, text in agent_ws.venv_site_files(root, main_site).items():
        (site / name).write_text(text)
    probe = (
        "import site, sys; site.addsitedir(sys.argv[1]);"
        "import lucid, tools.helper;"
        "print(lucid.__file__); print(tools.helper.__file__);"
        "print(sys.argv[2] in sys.path, sys.argv[3] in sys.path)"
    )
    out = subprocess.run(
        [sys.executable, "-S", "-c", probe, str(site), str(root), str(main_site)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert out == [
        str(root / "lucid" / "__init__.py"),
        str(root / "tools" / "helper.py"),
        "False True",
    ]


def test_setup_sets_worktree_config_and_doctor_verifies_it(
    scratch: Scratch,
) -> None:
    """LCD-30: `setup` sets `extensions.worktreeConfig = true` and `doctor` checks it."""
    scratch.git(scratch.main, "config", "--unset", "extensions.worktreeConfig")
    setup_res = subprocess.run(
        [sys.executable, str(AGENT_WS), "setup"],
        cwd=scratch.main,
        env=scratch.env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert setup_res.returncode == 0, setup_res.stderr
    assert "git config extensions.worktreeConfig = true" in setup_res.stdout
    val = scratch.git(
        scratch.main, "config", "--bool", "extensions.worktreeConfig"
    ).stdout.strip()
    assert val == "true"

    # When worktreeConfig is unset/false, doctor reports the problem and exits 1.
    scratch.git(scratch.main, "config", "extensions.worktreeConfig", "false")
    doc_res = subprocess.run(
        [sys.executable, str(AGENT_WS), "doctor"],
        cwd=scratch.a,
        env=scratch.env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert doc_res.returncode == 1
    assert "extensions.worktreeConfig is not true" in doc_res.stdout


def test_bootstrap_refreshes_stale_so_and_warns_when_cpp_is_newer(
    tmp_path: Path,
) -> None:
    """LCD-136: `bootstrap` re-clones a newer main `.so` and warns when C++ sources are newer."""
    spec = importlib.util.spec_from_file_location("agent_ws", AGENT_WS)
    assert spec is not None and spec.loader is not None
    agent_ws = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent_ws)

    main_root = tmp_path / "main"
    wt_root = tmp_path / "wt"
    (main_root / "lucid" / "_C").mkdir(parents=True)
    (wt_root / "lucid" / "_C").mkdir(parents=True)

    so_rel = Path("lucid/_C/engine.cpython-314-darwin.so")
    main_so = main_root / so_rel
    wt_so = wt_root / so_rel
    main_so.write_bytes(b"v2")
    wt_so.write_bytes(b"v1")
    os.utime(wt_so, (1000.0, 1000.0))
    os.utime(main_so, (2000.0, 2000.0))

    done = agent_ws.bootstrap(wt_root, main_root)
    assert f"cloned {so_rel}" in done
    assert wt_so.read_bytes() == b"v2"

    # Touch a C++ source in the worktree so it is newer than the .so.
    cpp = wt_root / "lucid" / "_C" / "foo.cpp"
    cpp.write_text("// changed\n")
    future_mtime = wt_so.stat().st_mtime + 100.0
    os.utime(cpp, (future_mtime, future_mtime))

    done2 = agent_ws.bootstrap(wt_root, main_root)
    assert any("engine .so is older than lucid/_C sources" in line for line in done2)
    assert not agent_ws._check_engine_freshness(wt_root)


def test_bootstrap_links_semgrep_executables_from_the_main_venv(tmp_path: Path) -> None:
    """LCD-536: the quality gate's semgrep collector shells out to a literal
    `semgrep` executable beside the interpreter that ran it, not an importable
    module (`python -m semgrep` is itself deprecated and exits nonzero), so the
    site-packages link bootstrap already writes is not enough — every
    worktree's self-check exited 2 ("cannot measure") until this. `pysemgrep`
    ships alongside it because semgrep's own CLI re-execs it from the same
    directory."""
    spec = importlib.util.spec_from_file_location("agent_ws", AGENT_WS)
    assert spec is not None and spec.loader is not None
    agent_ws = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent_ws)

    main_root = tmp_path / "main"
    wt_root = tmp_path / "wt"
    for root in (main_root, wt_root):
        (root / ".venv" / "bin").mkdir(parents=True)
        (root / ".venv" / "lib" / "python3.14" / "site-packages").mkdir(parents=True)
        (root / ".venv" / "bin" / "python3").write_text("#!/bin/sh\n")
    (main_root / ".venv" / "bin" / "semgrep").write_text("#!/bin/sh\necho semgrep\n")
    (main_root / ".venv" / "bin" / "pysemgrep").write_text(
        "#!/bin/sh\necho pysemgrep\n"
    )

    done = agent_ws.bootstrap(wt_root, main_root)
    assert "linked .venv/bin/semgrep" in done
    assert "linked .venv/bin/pysemgrep" in done
    wt_semgrep = wt_root / ".venv" / "bin" / "semgrep"
    assert wt_semgrep.is_symlink()
    assert wt_semgrep.resolve() == (main_root / ".venv" / "bin" / "semgrep").resolve()

    # Idempotent: an existing link (or a tool the main venv lacks) is left alone.
    done2 = agent_ws.bootstrap(wt_root, main_root)
    assert "linked .venv/bin/semgrep" not in done2
    assert "linked .venv/bin/jscpd" not in done2


def test_doctor_trusts_matching_source_stamp_or_abi_over_mtime(
    tmp_path: Path,
) -> None:
    """LCD-536: a rebase or checkout that bumps C++ mtime without changing
    either the C++ source hash or ``ABI_VERSION`` does not false-positive in ``doctor``."""
    spec = importlib.util.spec_from_file_location("agent_ws", AGENT_WS)
    assert spec is not None and spec.loader is not None
    agent_ws = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent_ws)

    main_root = tmp_path / "main"
    wt_root = tmp_path / "wt"
    (main_root / "lucid" / "_C").mkdir(parents=True)
    (wt_root / "lucid" / "_C").mkdir(parents=True)

    so_rel = Path("lucid/_C/engine.cpython-314-darwin.so")
    for root in (main_root, wt_root):
        (root / "lucid" / "_C" / "version.h").write_text("#define LUCID_ABI_VERSION 7\n")
        (root / "lucid" / "_C" / "foo.cpp").write_text("// identical\n")
        os.utime(root / "lucid" / "_C" / "version.h", (1000.0, 1000.0))
        os.utime(root / "lucid" / "_C" / "foo.cpp", (1000.0, 1000.0))

    main_so = main_root / so_rel
    main_so.write_bytes(b"v2")
    os.utime(main_so, (2000.0, 2000.0))

    done = agent_ws.bootstrap(wt_root, main_root)
    assert f"cloned {so_rel}" in done
    assert agent_ws._engine_stamp_path(wt_root).is_file()

    # Simulate a git rebase touching C++ mtimes without changing source bytes:
    wt_so = wt_root / so_rel
    future = wt_so.stat().st_mtime + 500.0
    os.utime(wt_root / "lucid" / "_C" / "foo.cpp", (future, future))
    assert agent_ws._check_engine_freshness(wt_root)

    # Actual C++ source change invalidates the stamp:
    (wt_root / "lucid" / "_C" / "foo.cpp").write_text("// modified\n")
    os.utime(wt_root / "lucid" / "_C" / "foo.cpp", (future + 10.0, future + 10.0))
    assert not agent_ws._check_engine_freshness(wt_root)

    # Without a stamp, a matching ABI_VERSION does NOT override mtime (LCD-536
    # regression: a prior version treated "no stamp + ABI match" as fresh,
    # false-negativing this still-stale .cpp-newer-than-.so state):
    agent_ws._engine_stamp_path(wt_root).unlink()
    version_h = wt_root / "lucid" / "_C" / "version.h"
    cpp = wt_root / "lucid" / "_C" / "foo.cpp"
    os.utime(version_h, (future, future))  # writes below must not leak wall-clock times
    venv_bin = wt_root / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    fake_py = venv_bin / "python3"
    fake_py.write_text("#!/bin/sh\necho 7\n")
    fake_py.chmod(0o755)
    assert not agent_ws._check_engine_freshness(wt_root)

    # A mismatched ABI_VERSION always flags stale, stamp or not:
    version_h.write_text("#define LUCID_ABI_VERSION 8\n")
    os.utime(version_h, (future, future))
    assert not agent_ws._check_engine_freshness(wt_root)

    # Restore a fresh .so/.cpp pair (matching mtimes) and confirm a matching
    # ABI_VERSION with no stamp and no mtime skew reads as fresh:
    version_h.write_text("#define LUCID_ABI_VERSION 7\n")
    os.utime(version_h, (future, future))
    fresh_mtime = future + 10.0
    os.utime(cpp, (fresh_mtime, fresh_mtime))
    os.utime(wt_so, (fresh_mtime, fresh_mtime))
    assert agent_ws._check_engine_freshness(wt_root)


_COVERAGE_XML = (
    '<?xml version="1.0" ?>\n'
    '<coverage version="7.0">\n'
    '  <packages><package name="lucid"><classes>\n'
    '    <class name="mod.py" filename="lucid/mod.py">\n'
    "      <lines>\n"
    '        <line number="1" hits="1"/>\n'
    '        <line number="2" hits="{hits}"/>\n'
    '        <line number="3" hits="{hits}"/>\n'
    "      </lines>\n"
    "    </class>\n"
    "  </classes></package></packages>\n"
    "</coverage>\n"
)


def commit_a_measured_lucid_change(scratch: Scratch) -> Path:
    """Branch A changes ``lucid/mod.py``; returns where its coverage.xml goes."""
    (scratch.main / ".gitignore").write_text("build/\ncoverage.xml\n.venv/\n")
    (scratch.main / "lucid").mkdir(parents=True, exist_ok=True)
    (scratch.main / "lucid" / "mod.py").write_text("def f(x: int) -> int:\n    return x\n")
    assert scratch.git(scratch.main, "add", ".gitignore", "lucid/mod.py").returncode == 0
    assert scratch.git(scratch.main, "commit", "-q", "-m", "add mod").returncode == 0
    assert scratch.git(scratch.a, "rebase", "main").returncode == 0

    (scratch.a / "lucid" / "mod.py").write_text(
        "def f(x: int) -> int:\n    y = x + 1\n    return y\n"
    )
    assert scratch.git(scratch.a, "add", "lucid/mod.py").returncode == 0
    assert scratch.git(scratch.a, "commit", "-q", "-m", "change mod").returncode == 0

    cov_dir = scratch.a / "build" / "coverage"
    cov_dir.mkdir(parents=True, exist_ok=True)
    return cov_dir


def test_land_check_reports_and_enforces_diff_cover_threshold(
    scratch: Scratch,
) -> None:
    """LCD-269: ``land --check`` runs ``diff-cover`` on changed ``lucid/*.py`` files
    and requires >= 90% changed-line coverage."""
    # diff-cover is in the `quality` dependency group, not the `test` extra
    # CI installs; without it nothing is measured (the test below).
    pytest.importorskip("diff_cover")
    cov_dir = commit_a_measured_lucid_change(scratch)

    # 0% changed-line coverage -> refused with exit 8
    (cov_dir / "coverage.xml").write_text(_COVERAGE_XML.format(hits=0))
    bad = land(scratch, "--check")
    bad_out = bad.stdout + bad.stderr
    assert bad.returncode == 8, bad_out
    assert "lucid/mod.py" in bad_out
    assert "below 90%" in bad_out

    # 100% changed-line coverage -> passes with exit 0
    (cov_dir / "coverage.xml").write_text(_COVERAGE_XML.format(hits=1))
    good = land(scratch, "--check")
    good_out = good.stdout + good.stderr
    assert good.returncode == 0, good_out
    assert "100%" in good_out


def test_land_check_without_diff_cover_cannot_measure_rather_than_fail_coverage(
    scratch: Scratch, tmp_path: Path
) -> None:
    """LCD-561: a worktree interpreter without diff-cover is "cannot measure"
    (exit 9), not "coverage below 90%" (exit 8) — CI's `.[test]` install has none."""
    cov_dir = commit_a_measured_lucid_change(scratch)
    (cov_dir / "coverage.xml").write_text(_COVERAGE_XML.format(hits=1))

    # `land` runs diff-cover with the worktree's .venv python; this one finds
    # a `diff_cover` that refuses to import ahead of any installed copy.
    hidden = tmp_path / "no-diff-cover"
    hidden.mkdir()
    (hidden / "diff_cover.py").write_text(
        "raise ModuleNotFoundError(\"No module named 'diff_cover'\", name='diff_cover')\n"
    )
    venv_py = scratch.a / ".venv" / "bin" / "python3"
    venv_py.parent.mkdir(parents=True)
    venv_py.write_text(
        "#!/bin/sh\n"
        f"PYTHONPATH={shlex.quote(str(hidden))} exec {shlex.quote(sys.executable)} \"$@\"\n"
    )
    venv_py.chmod(0o755)

    result = land(scratch, "--check")
    out = result.stdout + result.stderr
    assert result.returncode == 9, out
    assert "diff-cover cannot measure" in out
    assert "below 90%" not in out


def test_land_check_on_a_diff_cover_crash_cannot_measure_rather_than_fail_coverage(
    scratch: Scratch,
) -> None:
    """LCD-561: only diff-cover's own --fail-under verdict is exit 8; a run that
    dies before measuring (here an unreadable coverage.xml) is exit 9."""
    pytest.importorskip("diff_cover")
    cov_dir = commit_a_measured_lucid_change(scratch)
    (cov_dir / "coverage.xml").write_text("<coverage><unclosed>\n")

    result = land(scratch, "--check")
    out = result.stdout + result.stderr
    assert result.returncode == 9, out
    assert "diff-cover cannot measure (exit" in out
    assert "below 90%" not in out


def test_land_check_with_missing_coverage_xml_fails_closed(
    scratch: Scratch,
) -> None:
    """LCD-269: a branch that changes covered ``lucid/*.py`` with no coverage.xml,
    no .coverage data, and no touched tests fails closed with exit 9 rather than
    passing without measurement; a branch with no covered ``lucid/*.py`` changes
    still skips diff-cover with exit 0."""
    assert scratch.commit(scratch.a, "a.txt", "staged").returncode == 0
    no_lucid = land(scratch, "--check")
    no_lucid_out = no_lucid.stdout + no_lucid.stderr
    assert no_lucid.returncode == 0, no_lucid_out
    assert "no covered lucid/*.py changes — diff-cover skipped" in no_lucid_out

    commit_a_measured_lucid_change(scratch)
    result = land(scratch, "--check")
    out = result.stdout + result.stderr
    assert result.returncode == 9, out
    assert "diff-cover cannot measure" in out
    assert "no coverage.xml, no .coverage data, and no touched tests" in out


#: A pytest config that, like Lucid's, deselects a marker by default.
_DESELECTING_PYPROJECT = (
    "[tool.pytest.ini_options]\n"
    'markers = ["parity: needs the reference framework"]\n'
    "addopts = \"-m 'not parity'\"\n"
)


def commit_test_files(scratch: Scratch, files: dict[str, str]) -> None:
    """Branch A also changes ``lucid/test/`` files, with a pytest config."""
    (scratch.a / "pyproject.toml").write_text(_DESELECTING_PYPROJECT)
    for rel, text in files.items():
        path = scratch.a / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    assert scratch.git(scratch.a, "add", "pyproject.toml", *files).returncode == 0
    assert scratch.git(scratch.a, "commit", "-q", "-m", "tests").returncode == 0


def test_land_check_whose_touched_test_fails_under_coverage_cannot_measure(
    scratch: Scratch,
) -> None:
    """LCD-561: a coverage run that fails is an attempt that did not measure
    (exit 9), not the "nothing to measure" pass it used to fold into — with
    or without diff-cover or coverage installed."""
    commit_a_measured_lucid_change(scratch)
    commit_test_files(
        scratch, {"lucid/test/test_mod.py": "def test_mod() -> None:\n    assert False\n"}
    )

    result = land(scratch, "--check")
    out = result.stdout + result.stderr
    assert result.returncode == 9, out
    assert "diff-cover cannot measure: coverage run exited" in out


@pytest.mark.parametrize(
    "files",
    [
        {
            "lucid/test/conftest.py": "import pytest\n",
            "lucid/test/_fixtures/helper.py": "VALUE = 1\n",
        },
        {
            "lucid/test/test_mod.py": (
                "import pytest\n\n\n@pytest.mark.parity\n"
                "def test_mod() -> None:\n    assert False\n"
            )
        },
    ],
    ids=["helpers-only", "all-deselected"],
)
def test_land_check_whose_touched_files_run_no_test_fails_closed(
    scratch: Scratch, files: dict[str, str]
) -> None:
    """LCD-269: when covered ``lucid/*.py`` changed and the touched ``lucid/test/``
    files run no test (pytest exits 5) with no coverage.xml present, ``land --check``
    fails closed with exit 9 rather than passing unmeasured."""
    pytest.importorskip("coverage")
    commit_a_measured_lucid_change(scratch)
    commit_test_files(scratch, files)

    result = land(scratch, "--check")
    out = result.stdout + result.stderr
    assert result.returncode == 9, out
    assert "touched test files ran no tests (pytest exited 5)" in out


def test_land_check_rejects_stale_coverage_xml(scratch: Scratch) -> None:
    """LCD-269: a coverage.xml older than a modified ``lucid/*.py`` file is rejected
    as stale rather than reused."""
    pytest.importorskip("diff_cover")
    cov_dir = commit_a_measured_lucid_change(scratch)
    xml_file = cov_dir / "coverage.xml"
    xml_file.write_text(_COVERAGE_XML.format(hits=1))
    mod_py = scratch.a / "lucid" / "mod.py"
    stale_time = mod_py.stat().st_mtime - 60.0
    os.utime(xml_file, (stale_time, stale_time))

    stale = land(scratch, "--check")
    stale_out = stale.stdout + stale.stderr
    assert stale.returncode == 9, stale_out
    assert "stale build/coverage/coverage.xml (older than lucid/mod.py)" in stale_out


def test_land_measures_lucid_models_and_enforces_diff_cover_at_land(
    scratch: Scratch,
) -> None:
    """LCD-269: ``lucid/models/*.py`` is included in coverage and diff-cover, and
    non-``--check`` ``land`` also enforces diff-cover before fast-forwarding main."""
    pytest.importorskip("diff_cover")
    (scratch.main / ".gitignore").write_text("build/\ncoverage.xml\n.venv/\n")
    model_rel = "lucid/models/vision/tiny.py"
    (scratch.main / "lucid" / "models" / "vision").mkdir(parents=True, exist_ok=True)
    (scratch.main / model_rel).write_text("def forward(x: int) -> int:\n    return x\n")
    assert scratch.git(scratch.main, "add", ".gitignore", model_rel).returncode == 0
    assert scratch.git(scratch.main, "commit", "-q", "-m", "add model").returncode == 0
    assert scratch.git(scratch.a, "rebase", "main").returncode == 0

    (scratch.a / model_rel).write_text(
        "def forward(x: int) -> int:\n    y = x + 1\n    return y\n"
    )
    assert scratch.git(scratch.a, "add", model_rel).returncode == 0
    assert scratch.git(scratch.a, "commit", "-q", "-m", "change model").returncode == 0

    # Without coverage.xml, non-check `land` refuses with exit 9:
    before = main_tip(scratch)
    unmeasured = land(scratch, "--skip-stub-check")
    assert unmeasured.returncode == 9, unmeasured.stdout + unmeasured.stderr
    assert main_tip(scratch) == before

    # With 0% coverage on lucid/models/vision/tiny.py, `land` refuses with exit 8:
    cov_dir = scratch.a / "build" / "coverage"
    cov_dir.mkdir(parents=True, exist_ok=True)
    model_xml = _COVERAGE_XML.replace("lucid/mod.py", model_rel).replace("mod.py", "tiny.py")
    (cov_dir / "coverage.xml").write_text(model_xml.format(hits=0))
    low = land(scratch, "--skip-stub-check")
    low_out = low.stdout + low.stderr
    assert low.returncode == 8, low_out
    assert model_rel in low_out
    assert main_tip(scratch) == before

    # With 100% coverage on lucid/models/vision/tiny.py, `land` succeeds:
    (cov_dir / "coverage.xml").write_text(model_xml.format(hits=1))
    ok = land(scratch, "--skip-stub-check")
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert main_tip(scratch) != before
