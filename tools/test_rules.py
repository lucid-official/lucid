"""The registry refuses a record that enforces nothing, and the audit refuses a claim that is untrue.

Most cases build a small repository whose one tool is a script written here and drive it
through ``python -S -m tools.rules``: the exit convention is the contract the hooks will read,
and ``-S`` holds the package to the standard library.  The mutant cases break one line, of this
package or of a real tool and its records, and name what must say so; a mutant that survives
fails its case.  What no case can catch is written in ``tools/rules/audit.py``.

Not under ``lucid/test`` (the default ``testpaths``); ``lucid/test/unit/test_rules_tooling.py``
exposes it to CI.  On request::

    .venv/bin/python3 -m pytest tools/test_rules.py
"""

import contextlib
import functools
import importlib.util
import json
import os
import pty
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest

from tools import check_source_rules
from tools.rules import attest
from tools.rules.__main__ import main
from tools.rules.audit import Audit
from tools.rules.checks import EMPTY_TREE, CommandCheck, PytestCheck, Workspace, run_nodes
from tools.rules.controls import SeedControl
from tools.rules.model import Exit, Level, RuleError, Verdict
from tools.rules.registry import Registry

REPO = Path(__file__).resolve().parents[1]

# These run the real quality gate, which answers "cannot measure" without its
# `quality` dependency group.  CI's core job doesn't install it.
_needs_the_gate = pytest.mark.skipif(
    importlib.util.find_spec("ruff") is None and shutil.which("ruff") is None,
    reason="the quality gate needs ruff: pip install --group quality",
)

TOOL = """\
import pathlib, sys
named = sys.argv[1:] or [str(p) for p in pathlib.Path("src").rglob("*")]
bad = [p for p in named if "bad" in pathlib.Path(p).read_text()]
print("".join(f"{p}:1: [bad-word]\\n" for p in bad), end="")
sys.exit(1 if bad else 0)
"""
BLOB_TOOL = """\
import subprocess, sys
def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout
names = git("diff", "--cached", "--name-only", "--diff-filter=ACMR").split()
bad = [name for name in names if "bad" in git("show", f":{name}")]
print("".join(f"{name}:1: [bad-word]\\n" for name in bad), end="")
sys.exit(1 if bad else 0)
"""
FLAGS_ANY_TEXT = "import sys\nsys.exit(1 if open(sys.argv[1]).read() else 0)\n"
OFF_MESSAGE = """\
import sys
if "bad" in open(sys.argv[1]).read():
    print("some other complaint")
    sys.exit(1)
"""
CRASHES_SEEDED = """\
import sys
if "bad" in open(sys.argv[1]).read():
    raise ValueError("a crash is no finding")
"""
HEAD = 'id = "R1"\nstatement = "No bad word."\nwhy = "Bad."\nscope = ["src/**"]\ndoc = "global"\n'
CHECK = '[check.x]\nrun = ["{py}", "-S", "tools/x.py", "{changed}"]\ntrigger = ["src/**"]\n\n'
COMMIT = 'enforce = [{ level = "commit", check = "x" }]\n'
SEED = (
    '[[rule.control]]\nkind = "seed"\ncheck = ["x"]\npath = "src/deep/_seed.txt"\n'
    'clean = "ok\\n"\ntext = "bad\\n"\nexpect = "bad-word"\n'
)
RULE = f"{CHECK}[[rule]]\n{HEAD}{COMMIT}{SEED}"
# What a claim's level has to run: the python a hook trial stubs, then the tool.
RUNS_X = ".venv/bin/python3 tools/x.py\n"
INDEX_RULE = (
    '[check.y]\nrun = ["{py}", "-S", "tools/y.py", "--staged"]\ntrigger = ["src/**"]\n\n'
    + f"[[rule]]\n{HEAD.replace('R1', 'R2')}"
    + 'enforce = [{ level = "commit", check = "y" }]\n'
    + SEED.replace('["x"]', '["y"]')
)
TESTS = """\
import pytest
def test_ok(): pass
def test_bad(): assert False
@pytest.mark.skip
def test_skipped(): pass
@pytest.mark.xfail(strict=True)
def test_xfailed(): assert False
@pytest.fixture
def broken(): raise ValueError
def test_error(broken): pass
@pytest.mark.parametrize("n", [1, 2])
def test_many(n): pass
@pytest.mark.heavy
def test_marked(): pass
"""
PYPROJECT = (
    '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n'
    'markers = ["heavy: not by default"]\naddopts = "-m \'not heavy\'"\n'
)
AUDIT = ("audit",)
RUN = ("run", "--level", "commit")
PUSH = ("push", "--range", "base..main")
PUSH_TIP = ("push", "--range", "main..refs/heads/main~0")


@dataclass
class Scratch:
    root: Path
    env: dict[str, str] = field(repr=False)

    def write(self, relative: str, text: str) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def hook(self, name: str, body: str) -> None:
        self.write(f".githooks/{name}", f"#!/bin/sh\n{body}")
        (self.root / ".githooks" / name).chmod(0o755)

    def records(self, text: str, name: str = "x") -> None:
        self.write(f"tools/rules/registry/{name}.toml", text)

    def rules(
        self, *args: str, home: Path = REPO, stdin: str = ""
    ) -> subprocess.CompletedProcess[str]:
        argv = [sys.executable, "-S", "-m", "tools.rules", "--root", str(self.root), *args]
        return subprocess.run(
            argv, cwd=home, env=self.env, input=stdin, capture_output=True, text=True
        )

    def attempt(self, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        """Git as a person runs it, hooks and all; a refusal is an answer, not an error."""
        return subprocess.run(
            ["git", *args], cwd=self.root, env={**self.env, **env}, capture_output=True, text=True
        )

    def sha(self, rev: str = "HEAD") -> str:
        return self.attempt("rev-parse", rev).stdout.strip()

    def message(self, rev: str = "HEAD") -> str:
        return self.attempt("log", "-1", "--format=%B", rev).stdout

    @property
    def bypass_log(self) -> str:
        log = self.root / ".git/lucid-bypass.log"
        return log.read_text() if log.exists() else ""

    def git(self, *args: str) -> None:
        subprocess.run(["git", *args], cwd=self.root, env=self.env, check=True, capture_output=True)

    def commit_all(self, message: str = "commit") -> None:
        if not (self.root / ".git").exists():
            self.git("init", "-q", "-b", "main")
            self.git("config", "user.name", "t")
            self.git("config", "user.email", "t@example.invalid")
        self.git("add", "-A")
        self.git("commit", "-qm", message)

    @property
    def registry(self) -> Registry:
        return Registry.load(self.root)

    @property
    def ws(self) -> Workspace:
        return replace(Workspace.of(self.root), isolated=True)


@pytest.fixture
def scratch(tmp_path: Path) -> Scratch:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_") and k != "PYTHONPATH"}
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    made = Scratch(tmp_path, env)
    made.write("pyproject.toml", PYPROJECT)
    made.hook("pre-commit", RUNS_X)
    made.write("tools/x.py", TOOL)
    made.write("src/a.txt", "ok\n")
    made.records(RULE)
    return made


# ── load ─────────────────────────────────────────────────────────────────────

_EXEMPT = "[[rule.exempt]]\nmatch = 'a'\nsince = 2026-11-05\nreason = 'r'\nissue = 'LCD-1'\n"
_BAD_RECORDS = [
    (f"[[rule]]\n{HEAD}", "needs one of enforce"),
    (f"{CHECK}[[rule]]\n{HEAD}{COMMIT}", "no control proves"),
    (f"{CHECK}[[rule]]\n{HEAD}{SEED}", "controls prove"),
    (RULE.replace(COMMIT, f'{COMMIT}advisory = "because"\n'), "needs one of"),
    (f'[[rule]]\n{HEAD}advisory = "  "\n', "advisory is blank"),
    (f'[[rule]]\n{HEAD}pending = "soon"\n', "not an LCD issue"),
    (f'[[rule]]\n{HEAD}pending = "LCD-999999"\n', "not an LCD issue"),
    (RULE.replace(COMMIT, f"{COMMIT}enforcement = 1\n"), "unknown key"),
    (RULE.replace('"R1"', '"r1"'), "bad rule id"),
    (RULE.replace('"No bad word."', '"""a\nb\nc"""'), "statement over two lines"),
    (RULE.replace('"No bad word."', '""'), "statement is blank"),
    (RULE.replace('"Bad."', '"""a\nb"""'), "why is one line"),
    (RULE.replace('"Bad."', '""'), "why is one line"),
    (RULE.replace('"global"', '"nowhere"'), "unknown doc"),
    (RULE.replace('["src/**"]\ndoc', "[]\ndoc"), "scope is empty"),
    (RULE.replace('["src/**"]\ndoc', '["/src/**"]\ndoc'), "leaves the tree"),
    (RULE.replace('level = "commit"', 'level = "lunch"'), "unknown Level"),
    (RULE.replace('level = "commit"', 'level = "test"'), "cannot run at test"),
    (RULE.replace('check = "x" }', 'check = "y" }'), "no control proves"),
    (RULE.replace('kind = "seed"', 'kind = "vibes"'), "unknown kind"),
    (
        RULE.replace('run = ["{py}", "-S", "tools/x.py", "{changed}"]', 'run = ["{py}"]'),
        "runs nothing",
    ),
    (RULE.replace('trigger = ["src/**"]', 'trigger = "src/**"'), "must list strings"),
    (RULE.replace('trigger = ["src/**"]', 'trigger = ["../**"]'), "leaves the tree"),
    (f'{CHECK}kind = "psychic"\n[[rule]]\n{HEAD}{COMMIT}{SEED}', "unknown kind"),
    (f'{RULE}\n[check.y]\nrun = ["{{py}}", "tools/x.py"]\n', "no rule claims"),
    (f"{RULE}\n[budget.b]\nunit = 'bytes'\nceiling = true\npaths = ['x']\n", "is not"),
    (f"{RULE}\n[budget.b]\nunit = 'parsecs'\nceiling = 1\npaths = ['x']\n", "unknown unit"),
    (f"{RULE}\n[budget.b]\nunit = 'bytes'\nceiling = 1\npaths = ['/x']\n", "leaves the tree"),
    (f"wish = 'more'\n{RULE}", "unknown table"),
    (f"{RULE}\n[check]\nx = 1\n", "Cannot"),  # TOML itself: a table declared twice
    ("check = 1\n[[rule]]\n" + HEAD + 'advisory = "a"\n', "must be tables"),
    ("", "no rules"),
    (RULE.replace("[check.x]\n", '[check.x]\nname = "y"\n'), "not its to set"),
    (f"{RULE}{_EXEMPT}", "unknown key"),  # no check reads one yet: LCD-841-B1
    (RULE.replace('clean = "ok\\n"\n', ""), "'clean' is missing"),
    (RULE.replace('["x"]', '"x"'), "must list strings"),
    (RULE.replace('path = "src/deep/_seed.txt"', 'path = "/tmp/out.py"'), "leaves the tree"),
    (RULE.replace('path = "src/deep/_seed.txt"', 'path = "../out.py"'), "leaves the tree"),
    (RULE.replace('expect = "bad-word"', "expect = '['"), "expect"),
    (
        '[check.x]\nkind = "pytest"\nnodes = ["tests/test_m.py::test_ok"]\n\n'
        f'[[rule]]\n{HEAD}enforce = [{{ level = "test", check = "x" }}]\n{SEED}',
        "cannot prove",
    ),
]


@pytest.mark.parametrize(("text", "problem"), _BAD_RECORDS, ids=[p for _, p in _BAD_RECORDS])
def test_a_bad_record_fails_load(scratch: Scratch, text: str, problem: str) -> None:
    scratch.records(text)
    with pytest.raises(RuleError, match=problem):
        Registry.load(scratch.root)


def test_ids_are_unique_across_files_and_the_private_layer_merges(scratch: Scratch) -> None:
    advisory = f"[[rule]]\n{HEAD.replace('R1', 'R2')}advisory = 'judgement'\n"
    scratch.write(".claude/rules.local.toml", advisory)
    assert sorted(scratch.registry.rules) == ["R1", "R2"]
    scratch.records(advisory, "y")
    with pytest.raises(RuleError, match="R2 is declared twice"):
        Registry.load(scratch.root)


def test_a_rule_is_found_by_its_level(scratch: Scratch) -> None:
    assert list(scratch.registry.at_level(Level.COMMIT)) == ["x"]
    assert scratch.registry.at_level(Level.PUSH) == {}


# ── end to end: each scenario is a claim the audit or the runner must settle ─────


def _tool(body: str) -> Callable[[Scratch], None]:
    return lambda s: s.write("tools/x.py", body)


def _record(text: str) -> Callable[[Scratch], None]:
    return lambda s: s.records(text)


def _hook(body: str, name: str = "pre-commit") -> Callable[[Scratch], None]:
    return lambda s: s.hook(name, body)


def _no_tool(s: Scratch) -> None:
    (s.root / "tools/x.py").unlink()


def _index_rule(body: str = BLOB_TOOL) -> Callable[[Scratch], None]:
    def setup(s: Scratch) -> None:
        s.write("tools/y.py", body)
        s.hook("pre-commit", f"{RUNS_X}.venv/bin/python3 tools/y.py --staged\n")
        s.records(INDEX_RULE, "y")

    return setup


def _pytest_rule(check: str, control: str) -> Callable[[Scratch], None]:
    node = "tests/test_m.py::"

    def setup(s: Scratch) -> None:
        s.write("tests/test_m.py", TESTS)
        s.records(
            f'[check.t]\nkind = "pytest"\nnodes = ["{node}{check}"]\n\n[[rule]]\n{HEAD}'
            'enforce = [{ level = "test", check = "t" }]\n\n'
            f'[[rule.control]]\nkind = "pytest"\ncheck = ["t"]\nnodes = ["{node}{control}"]\n'
        )

    return setup


def _pending(*wired: str) -> Callable[[Scratch], None]:
    def setup(s: Scratch) -> None:
        claim = '{ level = "push", check = "x", pending = "LCD-841-R2" }, '
        s.records(RULE.replace("enforce = [", f"enforce = [{claim}"))
        for line in wired:
            s.hook("pre-push", line)

    return setup


def _ci(file: str, text: str) -> Callable[[Scratch], None]:
    def setup(s: Scratch) -> None:
        s.records(RULE.replace('level = "commit"', 'level = "ci"'))
        s.write(file, text)

    return setup


def _push_rule(body: str) -> Callable[[Scratch], None]:
    def setup(s: Scratch) -> None:
        s.records(RULE.replace('level = "commit"', 'level = "push"'))
        s.hook("pre-push", body)

    return setup


def _budget(
    unit: str, ceiling: int, paths: str = 'paths = ["big.txt"]\n'
) -> Callable[[Scratch], None]:
    def setup(s: Scratch) -> None:
        s.write("big.txt", "a\n# b\n\nc\n")
        s.records(f'{RULE}\n[budget.b]\nunit = "{unit}"\nceiling = {ceiling}\n{paths}')

    return setup


def _pending_budget(ceiling: int) -> Callable[[Scratch], None]:
    def setup(s: Scratch) -> None:
        s.records(f"[[rule]]\n{HEAD.replace('R1', 'R2')}pending = 'LCD-1'\n", "y")
        _budget("pending", ceiling, "")(s)

    return setup


def _allowlist(ceiling: int, match: str = "", name: str = "T") -> Callable[[Scratch], None]:
    def setup(s: Scratch) -> None:
        s.write(
            "tools/table.py", "T = {('a', 'goto'): 1, ('b', 'c'): 2}\nT.update({('d', 'e'): 3})\n"
        )
        line = f'allowlist = {{ source = "tools.table:{name}", match = "{match}", ceiling = {ceiling} }}\n'
        s.records(RULE.replace(COMMIT, COMMIT + line))

    return setup


@dataclass(frozen=True)
class Scenario:
    setup: Callable[[Scratch], None]
    args: tuple[str, ...]
    code: int
    said: str


def _nothing(_: Scratch) -> None:
    pass


def _staged_differs(s: Scratch) -> None:
    s.commit_all()
    s.write("src/n.txt", "bad\n")
    s.git("add", "src/n.txt")
    s.write("src/n.txt", "ok\n")


def _seed_outside_trigger(s: Scratch) -> None:
    s.write("other/a.txt", "ok\n")
    s.records(RULE.replace('trigger = ["src/**"]', 'trigger = ["other/**"]'))


def _pushable(bad: str = "", dirty: bool = False, tip: str = "") -> Callable[[Scratch], None]:
    def setup(s: Scratch) -> None:
        s.records(RULE.replace('level = "commit"', 'level = "push"'))
        s.commit_all("base")
        s.git("tag", "base")
        s.write("src/n.txt", f"{bad}\\n")
        s.commit_all("tip")
        if dirty:
            s.write("tools/x.py", TOOL + "\\n")
        if tip:
            s.git("checkout", "-q", tip)

    return setup


def _staged_outside_trigger(s: Scratch) -> None:
    s.commit_all()
    s.write("web/n.txt", "bad\n")
    s.git("add", "web/n.txt")


_SCENARIOS = {
    # seed controls
    "proves": Scenario(_nothing, AUDIT, 0, "1/1 proven"),
    "green-forever": Scenario(
        _tool("raise SystemExit(0)"), AUDIT, 1, "R1: x control: stayed green"
    ),
    "red-forever": Scenario(_tool("raise SystemExit(1)"), AUDIT, 2, "not green on the near miss"),
    "flags-the-near-miss": Scenario(_tool(FLAGS_ANY_TEXT), AUDIT, 2, "not green on the near miss"),
    "off-message": Scenario(_tool(OFF_MESSAGE), AUDIT, 1, "went red without /bad-word/"),
    "crashes-seeded": Scenario(_tool(CRASHES_SEEDED), AUDIT, 2, "could not run seeded"),
    "missing-tool": Scenario(_no_tool, AUDIT, 2, "could not copy tools/x.py"),
    "index-seed": Scenario(_index_rule(), AUDIT, 0, "2/2 proven"),
    "index-tool-reads-nothing": Scenario(
        _index_rule("raise SystemExit(0)"), AUDIT, 1, "R2: y control: stayed green"
    ),
    # triggers and scope
    "seed-outside-trigger": Scenario(
        _seed_outside_trigger,
        AUDIT,
        1,
        "its trigger leaves out src/deep/_seed.txt",
    ),
    "dead-trigger": Scenario(
        _record(RULE.replace('trigger = ["src/**"]', 'trigger = ["nowhere/**"]')),
        AUDIT,
        1,
        "x: nowhere/** matches no file",
    ),
    "dead-scope": Scenario(
        _record(RULE.replace('scope = ["src/**"]', 'scope = ["nowhere/**"]')),
        AUDIT,
        1,
        "R1: nowhere/** matches no file",
    ),
    # test nodes: the control's and the claimed level's own
    "own-node-skipped": Scenario(
        _pytest_rule("test_skipped", "test_ok"), AUDIT, 2, "test_skipped did not run"
    ),
    "own-node-deselected": Scenario(
        _pytest_rule("test_marked", "test_ok"), AUDIT, 2, "test_marked did not run"
    ),
    "own-node-absent": Scenario(_pytest_rule("test_gone", "test_ok"), AUDIT, 2, "did not run"),
    # wiring: a real commit or push runs the tool, and stops when it fails
    "runs": Scenario(_nothing, AUDIT, 0, "1/1 claimed levels"),
    "run-only-in-a-comment": Scenario(
        _hook("# .venv/bin/python3 tools/x.py\n"),
        AUDIT,
        1,
        "R1: level=commit check=x: a real commit in the main checkout did not run tools/x.py",
    ),
    "exit-0-above-the-line": Scenario(_hook(f"exit 0\n{RUNS_X}"), AUDIT, 1, "did not run"),
    "if-false": Scenario(_hook(f"if false; then {RUNS_X}fi\n"), AUDIT, 1, "did not run"),
    "name-in-an-echo": Scenario(_hook("echo tools/x.py\n"), AUDIT, 1, "did not run"),
    "name-with-a-suffix": Scenario(
        _hook(".venv/bin/python3 tools/x.py.disabled\n"), AUDIT, 1, "did not run"
    ),
    "guard-never-fires": Scenario(
        _hook(f"if git diff --cached --name-only | grep -q ^nowhere/; then {RUNS_X}fi\n"),
        AUDIT,
        1,
        "did not run",
    ),
    "guard-fires-on-the-trigger": Scenario(
        _hook(f"git diff --cached --name-only | grep -q ^src/ || exit 0\n{RUNS_X}"),
        AUDIT,
        0,
        "1/1 claimed",
    ),
    "exit-dropped": Scenario(
        _hook(".venv/bin/python3 tools/x.py || true\n"),
        AUDIT,
        1,
        "a real commit in the main checkout went through with tools/x.py failing",
    ),
    "only-past-a-worktree-exit": Scenario(
        _hook(f"[ -f .git ] && exit 0\n{RUNS_X}"),
        AUDIT,
        1,
        "a real commit in a linked worktree did not run tools/x.py",
    ),
    "not-executable": Scenario(
        lambda s: (s.root / ".githooks/pre-commit").chmod(0o644), AUDIT, 1, "did not run"
    ),
    "tool-fails-for-all": Scenario(
        _hook(f"exit 1\n{RUNS_X}"), AUDIT, 1, "failed with its tool passing"
    ),
    "run-by-the-runner": Scenario(
        _hook(".venv/bin/python3 -m tools.rules run --level commit --staged\n"),
        AUDIT,
        0,
        "1/1 claimed",
    ),
    "runner-off-the-index": Scenario(
        _hook(".venv/bin/python3 -m tools.rules run --level commit\n"), AUDIT, 1, "did not run"
    ),
    "the-wrong-runner": Scenario(
        _hook(".venv/bin/python3 -m tools.rules run --level push\n"), AUDIT, 1, "did not run"
    ),
    "ci-script": Scenario(_ci("scripts/ci_full.sh", '"$PY" tools/x.py\n'), AUDIT, 0, "1/1 claimed"),
    "ci-workflow": Scenario(
        _ci(".github/workflows/x.yml", "run: python tools/x.py\n"), AUDIT, 0, "1/1 claimed"
    ),
    "ci-other-script": Scenario(
        _ci("scripts/o.sh", "python tools/x.py\n"), AUDIT, 1, "no tools/x.py"
    ),
    "ci-only-in-a-comment": Scenario(
        _ci("scripts/ci_full.sh", "# tools/x.py is what this would run\n"),
        AUDIT,
        1,
        "R1: level=ci check=x: scripts/ci_full.sh or .github/workflows/*.yml: no tools/x.py",
    ),
    "ci-in-a-trailing-comment": Scenario(
        _ci("scripts/ci_full.sh", "exit 0  # tools/x.py\n"), AUDIT, 1, "no tools/x.py"
    ),
    "ci-after-a-semicolon": Scenario(
        _ci("scripts/ci_full.sh", "true;#tools/x.py\n"), AUDIT, 1, "no tools/x.py"
    ),
    "ci-with-a-suffix": Scenario(
        _ci("scripts/ci_full.sh", "X=tools/x.py.disabled\n"), AUDIT, 1, "no tools/x.py"
    ),
    "ci-with-a-prefix": Scenario(
        _ci("scripts/ci_full.sh", "X=retools/x.py\n"), AUDIT, 1, "no tools/x.py"
    ),
    "pending-counted": Scenario(_pending(), AUDIT, 0, "1 pending on LCD-841-R2"),
    "pending-gone-stale": Scenario(
        _pending('.venv/bin/python3 -m tools.rules push "$@"\n'),
        AUDIT,
        1,
        "is wired now: drop pending",
    ),
    "push-with-no-hook": Scenario(
        _record(RULE.replace('level = "commit"', 'level = "push"')),
        AUDIT,
        1,
        "a real push in the main checkout did not run tools/x.py",
    ),
    "push-run-by-the-gate": Scenario(
        _push_rule('.venv/bin/python3 -m tools.rules push "$@"\n'), AUDIT, 0, "1/1 claimed"
    ),
    "push-gate-off-the-refs": Scenario(
        _push_rule('.venv/bin/python3 -m tools.rules push "$@" </dev/null\n'),
        AUDIT,
        1,
        "did not run tools/x.py",
    ),
    "push-exit-dropped": Scenario(
        _push_rule(f"{RUNS_X.strip()} || true\n"),
        AUDIT,
        1,
        "push in the main checkout went through",
    ),
    # ceilings
    "budget-at": Scenario(_budget("bytes", 9), AUDIT, 0, "b 9/9"),
    "budget-over": Scenario(_budget("bytes", 8), AUDIT, 1, "b: 9 bytes is over its ceiling of 8"),
    "budget-lines-free-comments": Scenario(_budget("lines", 2), AUDIT, 0, "b 2/2"),
    "budget-lines-over": Scenario(_budget("lines", 1), AUDIT, 1, "b: 2 lines is over"),
    "budget-no-match": Scenario(
        _budget("bytes", 7, 'paths = ["no.txt"]\n'), AUDIT, 2, "which match nothing"
    ),
    "pending-at": Scenario(_pending_budget(1), AUDIT, 0, "b 1/1"),
    "pending-over": Scenario(_pending_budget(0), AUDIT, 1, "b: 1 pending is over"),
    "pending-fell": Scenario(_pending_budget(2), AUDIT, 1, "b: fell to 1: lower ceiling 2"),
    "allowlist-at": Scenario(_allowlist(3), AUDIT, 0, "1/1 at their ceiling"),
    "allowlist-grew": Scenario(_allowlist(2), AUDIT, 1, "grew: 3 > 2"),
    "allowlist-shrank": Scenario(_allowlist(4), AUDIT, 1, "shrank to 3: lower ceiling 4"),
    "allowlist-match": Scenario(_allowlist(0, "goto"), AUDIT, 1, "grew: 1 > 0"),
    "allowlist-unreadable": Scenario(_allowlist(1, "", "NOPE"), AUDIT, 2, "could not be read"),
    # the runner's exit convention
    "exit-0": Scenario(_tool("raise SystemExit(0)"), RUN, 0, "PASS"),
    "exit-1": Scenario(_tool("raise SystemExit(1)"), RUN, 1, "VIOLATION"),
    "exit-2": Scenario(_tool("raise SystemExit(2)"), RUN, 2, "UNMEASURED"),
    "exit-3": Scenario(_tool("raise SystemExit(3)"), RUN, 2, "UNMEASURED"),
    "tool-crashes": Scenario(_tool("raise ValueError('a bug')"), RUN, 2, "UNMEASURED"),
    "tool-killed": Scenario(
        _tool("import os, signal; os.kill(os.getpid(), signal.SIGKILL)"), RUN, 2, "UNMEASURED"
    ),
    "run-tool-missing": Scenario(_no_tool, RUN, 2, "UNMEASURED"),
    "no-level-claims": Scenario(_nothing, ("run", "--level", "ci"), 2, "running nothing proves"),
    "staged-working-copy-differs": Scenario(
        _staged_differs, (*RUN, "--staged"), 2, "not what is staged: src/n.txt"
    ),
    "staged-outside-trigger": Scenario(
        _staged_outside_trigger, (*RUN, "--staged"), 0, "no changed file matches"
    ),
    # the push gate, on a range
    "push-judged": Scenario(_pushable(), PUSH, 0, "push passes"),
    "push-a-violation": Scenario(_pushable("bad"), PUSH, 1, "is refused"),
    "push-a-dirty-tree": Scenario(_pushable(dirty=True), PUSH, 2, "M tools/x.py"),
    "push-a-tip-not-checked-out": Scenario(
        _pushable(tip="base"), PUSH_TIP, 2, "check out what you are pushing"
    ),
    # load, through the command line
    "no-form": Scenario(
        _record(f"[[rule]]\n{HEAD}"), AUDIT, 1, "R1: needs one of enforce + control"
    ),
    "no-control": Scenario(
        _record(f"{CHECK}[[rule]]\n{HEAD}{COMMIT}"), AUDIT, 1, "no control proves"
    ),
    "typo": Scenario(
        _record(RULE.replace(COMMIT, f"{COMMIT}gapp = 'x'\n")), AUDIT, 1, "unknown key"
    ),
    "unclaimed-check": Scenario(
        _record(f'{RULE}\n[check.y]\nrun = ["{{py}}", "tools/x.py"]\n'), AUDIT, 1, "no rule claims"
    ),
    "declared-twice": Scenario(
        lambda s: s.records(RULE.replace("[check.x]", "[check.z]").replace('"x"', '"z"'), "y"),
        AUDIT,
        1,
        "R1 is declared twice",
    ),
    "unknown-rule": Scenario(_nothing, ("audit", "--rule", "R9"), 2, "no such rule or check"),
    "unknown-check": Scenario(_nothing, ("audit", "--check", "z"), 2, "no such rule or check"),
    "show-unknown": Scenario(_nothing, ("show", "R9"), 2, "no rule R9"),
    "show": Scenario(_nothing, ("show", "R1"), 0, "why: Bad."),
    "list": Scenario(
        _nothing, ("list", "--verdict"), 0, "1 enforced, 0 partly, 0 advisory, 0 pending of 1"
    ),
}


@pytest.mark.parametrize("name", _SCENARIOS)
def test_a_scenario_ends_as_it_must(
    scratch: Scratch, capsys: pytest.CaptureFixture[str], name: str
) -> None:
    scenario = _SCENARIOS[name]
    scenario.setup(scratch)
    code = main(["--root", str(scratch.root), *scenario.args])
    out, err = capsys.readouterr()
    assert code == scenario.code and scenario.said in out + err, out + err


# (file in tools/rules, line, its replacement, the scenario that must then end otherwise)
_PACKAGE_MUTANTS = [
    (
        "checks.py",
        '_COMMENT.sub("", f.read_text(errors="replace"))',
        'f.read_text(errors="replace")',
        "ci-only-in-a-comment",
    ),
    ("checks.py", r"(?:^|[\s;])#", r"(?:^|\s)#", "ci-after-a-semicolon"),
    ("checks.py", r"(?![\w./-])", "", "ci-with-a-suffix"),
    ("checks.py", r"(?<![\w.-])", "", "ci-with-a-prefix"),
    ("checks.py", "re.search(self.wanted, log.read_text()) is None", "False", "the-wrong-runner"),
    (
        "checks.py",
        'if self._attempt(tree, "failing").returncode == 0:',
        "if False:",
        "exit-dropped",
    ),
    (
        "checks.py",
        """return self._tried(base, repo, "the main checkout") or self._tried(
                base, linked, "a linked worktree"
            )""",
        'return self._tried(base, repo, "the main checkout")',
        "only-past-a-worktree-exit",
    ),
    (
        "checks.py",
        "if passing.returncode != 0:",
        "if False:",
        "tool-fails-for-all",
    ),
    (
        "checks.py",
        'crashed = done.returncode == 1 and "Traceback (most recent call last)" in done.stderr',
        "crashed = False",
        "tool-crashes",
    ),
    ("checks.py", 'children & {"error", "skipped"}', "children & set()", "own-node-skipped"),
    (
        "checks.py",
        "max(found, default=Exit.UNMEASURED)",
        "max(found, default=Exit.PASS)",
        "own-node-deselected",
    ),
    (
        "checks.py",
        "differ = [] if check.reads_index else sorted(changes.differ.intersection(relevant))",
        "differ = []",
        "staged-working-copy-differs",
    ),
    (
        "checks.py",
        "if not self.trigger:\n            return list(changed)",
        "if True:\n            return list(changed)",
        "staged-outside-trigger",
    ),
    ("push.py", "if tip != head:", "if False:", "push-a-tip-not-checked-out"),
    ("push.py", "if dirty:", "if False:", "push-a-dirty-tree"),
    (
        "push.py",
        "return max((self._judge(push) for push in pushes), default=Exit.PASS)",
        "return Exit.PASS",
        "push-a-violation",
    ),
    (
        "push.py",
        "return max(result.outcome for result in results)",
        "return Exit.PASS",
        "push-a-violation",
    ),
    ("controls.py", "if clean.outcome is not Exit.PASS:", "if False:", "red-forever"),
    ("controls.py", "if re.search(self.expect, red.output) is None:", "if False:", "off-message"),
    ("controls.py", "if red.outcome is Exit.PASS:", "if False:", "green-forever"),
    ("controls.py", "if not check.relevant([self.path]):", "if False:", "seed-outside-trigger"),
    (
        "controls.py",
        'if check.reads_index:\n            scratch.spawn(["git", "add", "--", self.path])',
        "if False:\n            pass",
        "index-seed",
    ),
    ("audit.py", "if e.pending and problem is None:", "if False:", "pending-gone-stale"),
    ("audit.py", "if size < allowlist.ceiling:", "if False:", "allowlist-shrank"),
    ("audit.py", "if size > allowlist.ceiling:", "if False:", "allowlist-grew"),
    ("audit.py", "elif size > budget.ceiling:", "elif False:", "budget-over"),
    (
        "audit.py",
        'elif budget.unit == "pending" and size < budget.ceiling:',
        "elif False:",
        "pending-fell",
    ),
    (
        "audit.py",
        "if not any(path.is_file() for path in self.registry.root.glob(glob)):",
        "if False:",
        "dead-trigger",
    ),
    (
        "audit.py",
        "if not e.pending and isinstance(check := checks[e.check], PytestCheck)",
        "if False and isinstance(check := checks[e.check], PytestCheck)",
        "own-node-skipped",
    ),
    (
        "audit.py",
        'return bool(stripped) and not stripped.startswith("#")',
        "return True",
        "budget-lines-free-comments",
    ),
    (
        "model.py",
        'need(sum(forms) == 1, f"{self.id}: needs one of enforce + control, advisory, pending")',
        "pass",
        "no-form",
    ),
    (
        "model.py",
        'need(claimed <= proven, f"{self.id}: no control proves {sorted(claimed - proven)}")',
        "pass",
        "no-control",
    ),
    ("model.py", 'need(not left, f"{where}: unknown key(s) {sorted(left)}")', "pass", "typo"),
    (
        "registry.py",
        'need(not unclaimed, f"no rule claims check(s) {sorted(unclaimed)}")',
        "pass",
        "unclaimed-check",
    ),
    (
        "registry.py",
        'need(rule.id not in rules, f"{where}: rule {rule.id} is declared twice")',
        "pass",
        "declared-twice",
    ),
]


@pytest.mark.parametrize(
    ("file", "line", "new", "name"), _PACKAGE_MUTANTS, ids=[m[3] for m in _PACKAGE_MUTANTS]
)
def test_a_line_broken_in_this_package_is_seen(
    scratch: Scratch,
    tmp_path_factory: pytest.TempPathFactory,
    file: str,
    line: str,
    new: str,
    name: str,
) -> None:
    home = tmp_path_factory.mktemp("mutant")
    shutil.copytree(
        REPO / "tools/rules", home / "tools/rules", ignore=shutil.ignore_patterns("__pycache__")
    )
    source = home / "tools/rules" / file
    assert line in source.read_text(), line
    source.write_text(source.read_text().replace(line, new, 1))
    scenario = _SCENARIOS[name]
    scenario.setup(scratch)
    done = scratch.rules(*scenario.args, home=home)
    # A copy that cannot import "kills" every mutant: attest.py's top-level import of
    # tools/agent_ws.py, absent from the copy, did so for every case here once.
    assert "Traceback" not in done.stderr, done.stderr
    ended_as_before = (
        done.returncode == scenario.code and scenario.said in done.stdout + done.stderr
    )
    assert not ended_as_before, f"survived: {line}"


# ── the rest of the runner ───────────────────────────────────────────────────


def test_triggers_and_changed_files_come_from_the_range(scratch: Scratch) -> None:
    scratch.write("tools/x.py", 'import sys\nprint(sys.argv[1:], file=open("args", "w"))\n')
    scratch.write("src/kept.txt", "a\n")
    scratch.write("src/gone.txt", "a\n")
    scratch.write("web/other.txt", "a\n")
    scratch.commit_all("base")
    range_ = ("run", "--level", "commit", "--range", "HEAD~1..HEAD")

    scratch.write("web/other.txt", "b\n")
    scratch.commit_all("outside the trigger")
    done = scratch.rules(*range_)
    assert done.returncode == 0 and "no changed file matches its trigger" in done.stdout

    (scratch.root / "src/gone.txt").unlink()
    scratch.commit_all("only a deletion")
    done = scratch.rules(*range_)
    assert done.returncode == 0 and "its changed files were deleted" in done.stdout

    scratch.write("src/kept.txt", "b\n")
    scratch.write("web/other.txt", "c\n")
    scratch.commit_all("both")
    assert scratch.rules(*range_).returncode == 0
    assert (scratch.root / "args").read_text().strip() == "['src/kept.txt']"

    for bad in ("nope..HEAD", "HEAD"):
        assert scratch.rules("run", "--level", "commit", "--range", bad).returncode == 2


def test_a_staged_run_judges_the_index_or_says_it_cannot(scratch: Scratch) -> None:
    _index_rule()(scratch)
    scratch.commit_all("base")
    run = ("run", "--level", "commit", "--staged")

    # A bad blob with a clean working copy: the blob reader sees it, the other cannot judge.
    scratch.write("src/n.txt", "bad\n")
    scratch.git("add", "src/n.txt")
    scratch.write("src/n.txt", "ok\n")
    done = scratch.rules(*run)
    assert done.returncode == 2
    assert re.search(r"VIOLATION\s+y\s", done.stdout) and "src/n.txt:1: [bad-word]" in done.stdout
    assert (
        re.search(r"UNMEASURED\s+x\s", done.stdout)
        and "not what is staged: src/n.txt" in done.stdout
    )

    # The working copy gone: still the blob, never a skip.
    (scratch.root / "src/n.txt").unlink()
    done = scratch.rules(*run)
    assert done.returncode == 2 and re.search(r"VIOLATION\s+y\s", done.stdout), done.stdout

    # A clean blob with a bad working copy is no finding about the commit.
    scratch.write("src/n.txt", "ok\n")
    scratch.git("add", "src/n.txt")
    scratch.write("src/n.txt", "bad\n")
    done = scratch.rules(*run)
    assert done.returncode == 2 and "VIOLATION" not in done.stdout, done.stdout
    assert re.search(r"PASS\s+y\s", done.stdout)

    # Index and working copy agree: both judge the same text.
    scratch.write("src/n.txt", "ok\n")
    assert scratch.rules(*run).returncode == 0
    scratch.write("src/n.txt", "bad\n")
    scratch.git("add", "src/n.txt")
    assert scratch.rules(*run).returncode == 1

    # With no --staged there is no index to read, so the blob reader has nothing to say.
    done = scratch.rules("run", "--level", "commit")
    assert "only --staged can say" in done.stdout


def test_a_real_staged_run_judges_the_blob_and_refuses_the_working_copy(real_copy: Path) -> None:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_") and k != "PYTHONPATH"}
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    seed = real_copy / "lucid/_staged.py"
    bad = "from __future__ import annotations\n"

    def stage(text: str) -> None:
        seed.write_text(text)
        subprocess.run(["git", "add", "lucid/_staged.py"], cwd=real_copy, env=env, check=True)

    def run() -> subprocess.CompletedProcess[str]:
        argv = [sys.executable, "-S", "-m", "tools.rules", "run", "--level", "commit", "--staged"]
        return subprocess.run(argv, cwd=real_copy, env=env, capture_output=True, text=True)

    subprocess.run(["git", "init", "-q"], cwd=real_copy, env=env, check=True)
    try:
        stage(bad)
        seed.write_text("x = 1\n")
        blob_is_bad = run()
        assert blob_is_bad.returncode != 0 and "[future-annotations]" in blob_is_bad.stdout

        seed.unlink()
        copy_is_gone = run()
        assert "[future-annotations]" in copy_is_gone.stdout, copy_is_gone.stdout

        stage("x = 1\n")
        seed.write_text(bad)
        copy_is_bad = run()
        assert copy_is_bad.returncode == 2 and "[future-annotations]" not in copy_is_bad.stdout
        assert "not what is staged: lucid/_staged.py" in copy_is_bad.stdout
    finally:
        seed.unlink(missing_ok=True)
        shutil.rmtree(real_copy / ".git")


def test_a_node_that_did_not_run_proved_nothing(scratch: Scratch) -> None:
    scratch.write("tests/test_m.py", TESTS)
    want = {
        "test_ok": Exit.PASS,
        "test_many": Exit.PASS,
        "test_many[1]": Exit.PASS,
        "test_bad": Exit.VIOLATION,
        "test_skipped": Exit.UNMEASURED,
        "test_xfailed": Exit.UNMEASURED,
        "test_error": Exit.UNMEASURED,
        "test_marked": Exit.UNMEASURED,
    }
    nodes = {f"tests/test_m.py::{name}": outcome for name, outcome in want.items()}
    assert dict(run_nodes(scratch.ws, list(nodes)).outcomes) == nodes
    gone = run_nodes(scratch.ws, ["tests/test_m.py::test_ok", "tests/test_m.py::test_gone"])
    assert set(gone.outcomes.values()) == {Exit.UNMEASURED}  # pytest refuses both


def test_each_test_node_is_judged_for_the_rule_that_owes_it(scratch: Scratch) -> None:
    def record(rule: str, own: str, control: str) -> str:
        node = "tests/test_m.py::"
        head = HEAD.replace("R1", rule)
        return (
            f'[check.{rule}]\nkind = "pytest"\nnodes = ["{node}{own}"]\n\n[[rule]]\n{head}'
            f'enforce = [{{ level = "test", check = "{rule}" }}]\n\n[[rule.control]]\n'
            f'kind = "pytest"\ncheck = ["{rule}"]\nnodes = ["{node}{control}"]\n\n'
        )

    scratch.write("tests/test_m.py", TESTS)
    scratch.records(
        record("R1", "test_ok", "test_many")
        + record("R2", "test_ok", "test_bad")
        + record("R3", "test_ok", "test_skipped")
        + record("R4", "test_bad", "test_ok")
    )
    done = scratch.rules("audit")
    assert done.returncode == 2
    assert "R1:" not in done.stdout
    assert "R2: R2 control: tests/test_m.py::test_bad failed" in done.stdout
    assert "R3: R3 control: tests/test_m.py::test_skipped did not run" in done.stdout
    assert "R4: R4 control: tests/test_m.py::test_bad failed" in done.stdout


def test_pytest_nodes_share_one_process_and_name_their_own_rule(scratch: Scratch) -> None:
    _pytest_rule("test_ok", "test_ok")(scratch)
    scratch.write(
        "tests/test_m.py", TESTS + "print(__import__('os').getpid(), file=open('pid', 'a'))\n"
    )
    other = (Path(scratch.root) / "tools/rules/registry/x.toml").read_text()
    other = other.replace('"R1"', '"R2"').replace("[check.t]", "[check.u]").replace('"t"', '"u"')
    scratch.records(other.replace("test_ok", "test_bad"), "y")
    done = scratch.rules("audit")
    assert done.returncode == 1 and "R2: u control:" in done.stdout and "R1:" not in done.stdout
    assert len((scratch.root / "pid").read_text().split()) == 1


def test_a_control_does_not_see_the_committing_tree_through_git_variables(
    scratch: Scratch, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch.write(
        "tools/x.py", "import os\nraise SystemExit(1 if 'GIT_DIR' in os.environ else 0)\n"
    )
    monkeypatch.setenv("GIT_DIR", "/nowhere")
    assert scratch.ws.spawn([sys.executable, "tools/x.py"]).returncode == 0


def test_the_name_a_level_must_use_is_the_tool_not_its_first_word(scratch: Scratch) -> None:
    generate = CommandCheck(name="g", run=("{py}", "-m", "tools.rules", "generate", "--check"))
    assert generate.tool == "tools.rules generate"
    scratch.hook("pre-commit", ".venv/bin/python3 -m tools.rules list\n")
    assert generate.unwired(Level.COMMIT, scratch.root) is not None
    scratch.hook("pre-commit", ".venv/bin/python3 -m tools.rules generate --check\n")
    assert generate.unwired(Level.COMMIT, scratch.root) is None
    assert CommandCheck(name="q", run=("{py}", "-m", "tools.x", "--fast")).tool == "tools.x"
    assert CommandCheck(name="s", run=("{py}", "tools/x.py", "{changed}")).tool == "tools/x.py"


# ── the push gate: what it takes from git, on a small repository ─────────────

_ZERO = "0" * 40
_MAIN = "refs/heads/main"


def _update(local: str, remote: str, ref: str = _MAIN) -> str:
    return f"{_MAIN} {local} {ref} {remote}\n"


@dataclass(frozen=True)
class Pushed:
    """Two commits, as far as a gate can see them; ``tracking``: the remote's main, if known."""

    stdin: Callable[[str, str], str]
    code: int
    said: str
    tracking: str = ""


_PUSHES = {
    "main": Pushed(lambda b, t: _update(t, b), 0, "1 commit(s), 1 file(s)"),
    "nothing-to-push": Pushed(lambda b, t: "", 0, ""),
    "another-branch": Pushed(lambda b, t: _update(t, b, "refs/heads/side"), 0, ""),
    "another-tag": Pushed(lambda b, t: _update(t, _ZERO, "refs/tags/nightly"), 0, ""),
    "new-to-the-remote": Pushed(lambda b, t: _update(t, _ZERO), 0, "2 commit(s), 6 file(s)"),
    "main-deleted": Pushed(lambda b, t: _update(_ZERO, b), 2, f"this deletes {_MAIN}"),
    "tag-deleted": Pushed(lambda b, t: _update(_ZERO, t, "refs/tags/v1.0.0"), 0, ""),
    "tag-the-remote-holds": Pushed(lambda b, t: _update(b, _ZERO, "refs/tags/v1.0.0"), 0, "", "t"),
    "tag-with-commits-of-its-own": Pushed(
        lambda b, t: _update(t, _ZERO, "refs/tags/v1.0.0"), 0, "1 commit(s), 1 file(s)", "b"
    ),
    "tag-elsewhere-with-commits": Pushed(
        lambda b, t: _update(b, _ZERO, "refs/tags/v1.0.0"), 2, "check out what you are pushing"
    ),
    "tip-other-than-head": Pushed(lambda b, t: _update(b, t), 2, "check out what you are pushing"),
    "unknown-base": Pushed(lambda b, t: _update(t, "1" * 40), 2, "cannot measure"),
    "not-a-ref-update": Pushed(lambda b, t: "garbage\n", 2, "not a ref update"),
}


@pytest.mark.parametrize("name", _PUSHES)
def test_push_gate_reads_what_git_writes_to_the_hook(scratch: Scratch, name: str) -> None:
    case = _PUSHES[name]
    scratch.records(RULE.replace('level = "commit"', 'level = "push"'))
    scratch.commit_all("base")
    base = scratch.sha()
    scratch.write("src/n.txt", "ok\n")
    scratch.commit_all("tip")
    tip = scratch.sha()
    if case.tracking:
        tracked = base if case.tracking == "b" else tip
        scratch.git("update-ref", "refs/remotes/origin/main", tracked)
    done = scratch.rules("push", "origin", "url", stdin=case.stdin(base, tip))
    out = done.stdout + done.stderr
    assert done.returncode == case.code and case.said in out, out
    if not case.said:
        assert out == ""


def test_push_gate_judges_each_range_once_and_all_of_them(scratch: Scratch) -> None:
    scratch.records(RULE.replace('level = "commit"', 'level = "push"'))
    scratch.commit_all("base")
    base = scratch.sha()
    scratch.write("src/bad.txt", "bad\n")
    scratch.commit_all("tip")
    tip = scratch.sha()
    scratch.git("update-ref", "refs/remotes/origin/main", base)
    done = scratch.rules(
        "push", "origin", stdin=_update(tip, base) + _update(tip, _ZERO, "refs/tags/v1.0.0")
    )
    assert done.returncode == 1 and "src/bad.txt:1: [bad-word]" in done.stdout, done.stdout
    assert done.stdout.count("level push:") == 1
    assert "is refused" in done.stdout
    # A branch alongside changes nothing, and neither does the order.
    done = scratch.rules(
        "push", "origin", stdin=_update(tip, base, "refs/heads/s") + _update(tip, base)
    )
    assert done.returncode == 1 and done.stdout.count("level push:") == 1


@pytest.mark.parametrize(
    ("path", "code", "said"),
    [
        ("tools/x.py", 2, "tools/x.py"),
        ("tools/new.py", 2, "tools/new.py"),
        ("pyproject.toml", 2, "pyproject.toml"),
        (".githooks/pre-commit", 2, ".githooks/pre-commit"),
        ("src/a.txt", 0, "1 commit(s)"),
        ("web/untouched.txt", 0, "1 commit(s)"),
    ],
)
def test_push_gate_refuses_a_working_copy_that_is_not_what_leaves(
    scratch: Scratch, path: str, code: int, said: str
) -> None:
    scratch.records(RULE.replace('level = "commit"', 'level = "push"'))
    scratch.commit_all("base")
    base = scratch.sha()
    scratch.write("src/n.txt", "ok\n")
    scratch.commit_all("tip")
    existing = scratch.root / path
    scratch.write(path, (existing.read_text() if existing.exists() else "") + "\n")
    done = scratch.rules("push", "--range", f"{base}..HEAD")
    assert done.returncode == code and said in done.stdout + done.stderr, done.stdout + done.stderr
    ran = "level push:" in done.stdout
    assert ran is (code == 0)


def test_push_gate_prints_the_bypasses_in_the_range(scratch: Scratch) -> None:
    scratch.records(RULE.replace('level = "commit"', 'level = "push"'))
    scratch.commit_all("base")
    base = scratch.sha()
    scratch.write("src/n.txt", "ok\n")
    scratch.commit_all("chore: skipped\n\nLucid-Bypass: no-verify")
    scratch.write("src/m.txt", "ok\n")
    scratch.commit_all("chore: cleared")
    done = scratch.rules("push", "--range", f"{base}..HEAD")
    assert done.returncode == 0 and "2 commit(s)" in done.stdout
    assert "bypassed at commit (no-verify)" in done.stdout and "chore: skipped" in done.stdout
    assert "chore: cleared" not in done.stdout


def test_push_gate_shows_what_a_pass_notes_and_nothing_else(scratch: Scratch) -> None:
    scratch.records(RULE.replace('level = "commit"', 'level = "push"'))
    scratch.write("tools/x.py", 'print("note: let through, and why")\nprint("chatter")\n')
    scratch.commit_all("base")
    base = scratch.sha()
    scratch.write("src/n.txt", "ok\n")
    scratch.commit_all("tip")
    done = scratch.rules("push", "--range", f"{base}..HEAD")
    assert done.returncode == 0 and "      note: let through, and why" in done.stdout, done.stdout
    assert "chatter" not in done.stdout


def test_push_gate_refuses_what_is_no_range_and_a_terminal(scratch: Scratch) -> None:
    scratch.records(RULE.replace('level = "commit"', 'level = "push"'))
    assert scratch.rules("push", "--range", "HEAD").returncode == 2
    _, terminal = pty.openpty()
    argv = [sys.executable, "-S", "-m", "tools.rules", "--root", str(scratch.root), "push"]
    done = subprocess.run(
        argv, cwd=REPO, env=scratch.env, stdin=terminal, capture_output=True, text=True
    )
    assert done.returncode == 2 and "no ref updates" in done.stderr, done.stderr


def test_push_gate_fills_since_from_the_range_the_index_or_nothing(scratch: Scratch) -> None:
    scratch.write("tools/x.py", 'import sys\nprint(sys.argv[1], end="", file=open("since", "w"))\n')
    scratch.records(
        RULE.replace('"tools/x.py", "{changed}"', '"tools/x.py", "{since}"').replace(
            SEED, SEED.replace("bad-word", "x")
        )
    )
    scratch.commit_all("base")
    base = scratch.sha()
    scratch.write("src/n.txt", "ok\n")
    scratch.commit_all("tip")
    assert scratch.rules(*RUN, "--range", f"{base}..HEAD").returncode == 0
    assert (scratch.root / "since").read_text() == base
    assert scratch.rules(*RUN).returncode == 0
    assert (scratch.root / "since").read_text() == EMPTY_TREE
    scratch.write("src/m.txt", "ok\n")
    scratch.git("add", "src/m.txt")
    assert scratch.rules(*RUN, "--staged").returncode == 0
    assert (scratch.root / "since").read_text() == "HEAD"


# ── the ratchet from a ref: the two ways a committed rise could pass ─────────


def _gate_repo(s: Scratch) -> Callable[..., subprocess.CompletedProcess[str]]:
    def gate(*args: str) -> subprocess.CompletedProcess[str]:
        argv = [sys.executable, "-m", "tools.quality_gate", "--root", str(s.root), *args]
        collectors = ("--collectors", "ruff,counters,lizard")
        return subprocess.run(
            [*argv, *collectors], cwd=REPO, env=s.env, capture_output=True, text=True
        )

    s.write("lucid/mod.py", "x = 1\n")
    s.commit_all("first")
    done = gate("--rebaseline", "--reason", "test")
    assert done.returncode == 0, done.stdout + done.stderr
    s.commit_all("base")
    return gate


_RISE = "y: int = 'a'  # type: ignore[assignment]\n"
_RANGE_FLAGS = ("--fast", "--since", "HEAD~1", "--allow-slack")


@_needs_the_gate
def test_push_gate_since_sees_a_rise_already_committed(scratch: Scratch) -> None:
    gate = _gate_repo(scratch)
    scratch.write("lucid/mod.py", f"x = 1\n{_RISE}")
    scratch.commit_all("a rise")
    assert gate("--fast").returncode == 0  # nothing differs from HEAD: the vacuous pass
    done = gate(*_RANGE_FLAGS)
    assert done.returncode == 1 and "rose above the baseline" in done.stdout, done.stdout
    assert gate("--diff", "HEAD~1").returncode == 1


@_needs_the_gate
def test_push_gate_since_sees_a_baseline_raised_with_the_count(scratch: Scratch) -> None:
    gate = _gate_repo(scratch)
    baseline = scratch.root / "tools/quality_baseline.json"
    data = json.loads(baseline.read_text())
    data["counts"].setdefault("counters", {})["type-ignore[assignment]"] = {"lucid/mod.py": 1}
    baseline.write_text(json.dumps(data))
    scratch.write("lucid/mod.py", f"x = 1\n{_RISE}")
    scratch.commit_all("a rise, and the baseline to match")
    assert gate("--fast").returncode == 0  # HEAD's baseline agrees with itself
    done = gate(*_RANGE_FLAGS)
    assert done.returncode == 1 and "without a rebaselines entry" in done.stdout, done.stdout
    # The recorded way to raise one is not refused, and the record is what shows.
    data["rebaselines"].append({"date": "2026-11-05", "reason": "test", "by": "test"})
    baseline.write_text(json.dumps(data))
    scratch.commit_all("the rebaseline entry")
    assert gate("--fast", "--since", "HEAD~2", "--allow-slack").returncode == 0


@_needs_the_gate
def test_push_gate_since_only_judges(scratch: Scratch) -> None:
    gate = _gate_repo(scratch)
    for mode in (("--update", "--fast"), ("--full",), ("--report", "--fast")):
        done = gate(*mode, "--since", "HEAD")
        assert done.returncode == 2 and "--since only judges" in done.stderr, done.stderr
    assert gate("--fast", "--since", "nowhere").returncode == 2


# ── the bypass record: real hooks, real commits, a stub for every tool ───────


def _stub_python(tree: Path, body: str = "exit 0\n") -> None:
    python = tree / ".venv/bin/python3"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text(f"#!/bin/sh\n{body}")
    python.chmod(0o755)


def _real_hooks(s: Scratch, hooks: Path = REPO / ".githooks", python: bool = True) -> None:
    """Commits in ``s`` go through ``hooks``; the scripts they look for are there, and pass."""
    text = (hooks / "pre-commit").read_text().replace("${RULES_MAIN}", "tools/rules/__main__.py")
    for script in re.findall(r"\btools/[\w/]+\.py", text):
        s.write(script, "")
    s.write(".gitignore", ".venv\n")
    s.write("README.md", "x\n")
    if python:
        _stub_python(s.root)
    s.commit_all("chore: base")
    s.git("config", "core.hooksPath", str(hooks))


def _commit(s: Scratch, *args: str, text: str = "x\n", **env: str) -> None:
    s.write("src/a.txt", text)
    done = s.attempt("commit", "-qam", "chore: work\n\nThe body.", *args, **env)
    assert done.returncode == 0, done.stderr


def test_bypass_a_cleared_commit_leaves_nothing(scratch: Scratch) -> None:
    _real_hooks(scratch)
    _commit(scratch)
    assert "Lucid-Bypass" not in scratch.message() and scratch.bypass_log == ""
    assert "The body." in scratch.message()
    assert not (scratch.root / ".git/lucid-hook-ran").exists()


def test_bypass_no_verify_leaves_a_trailer_and_a_log_line(scratch: Scratch) -> None:
    _real_hooks(scratch)
    _commit(scratch, "--no-verify")
    message = scratch.message()
    assert "Lucid-Bypass: no-verify" in message and "The body." in message
    assert re.search(rf"{scratch.sha()[:7]}\w* no-verify: chore: work\n$", scratch.bypass_log)
    _commit(scratch, text="y\n")
    assert "Lucid-Bypass" not in scratch.message(), "the next cleared commit is clean"
    assert scratch.bypass_log.count("\n") == 1


def test_bypass_the_amend_takes_only_what_the_commit_took(scratch: Scratch) -> None:
    """A path-limited commit leaves other staged files in the index, for the next commit."""
    _real_hooks(scratch)
    scratch.write("src/staged.txt", "s\n")
    scratch.git("add", "src/staged.txt")
    scratch.write("src/a.txt", "x\n")
    done = scratch.attempt("commit", "-qm", "chore: path", "--no-verify", "--", "src/a.txt")
    assert done.returncode == 0, done.stderr
    assert "Lucid-Bypass: no-verify" in scratch.message()
    shown = scratch.attempt("show", "--name-only", "--format=", "HEAD").stdout.split()
    assert shown == ["src/a.txt"]
    assert scratch.attempt("diff", "--cached", "--name-only").stdout.split() == ["src/staged.txt"]


def test_bypass_a_receipt_from_an_aborted_commit_does_not_pass(scratch: Scratch) -> None:
    _real_hooks(scratch)
    scratch.write("src/a.txt", "x\n")
    assert scratch.attempt("commit", "-qam", "").returncode != 0
    assert (scratch.root / ".git/lucid-hook-ran").read_text().endswith("done\n")
    _commit(scratch, "--no-verify")
    assert "Lucid-Bypass: no-verify" in scratch.message()


def test_bypass_an_exit_above_the_last_check_shows(scratch: Scratch, tmp_path: Path) -> None:
    hooks = tmp_path / "hooks"
    shutil.copytree(REPO / ".githooks", hooks)
    text = (hooks / "pre-commit").read_text()
    assert 'echo done >> "${RECEIPT}"' in text
    (hooks / "pre-commit").write_text(text.replace('echo done >> "${RECEIPT}"', "exit 0"))
    _real_hooks(scratch, hooks)
    _commit(scratch)
    assert "Lucid-Bypass: pre-commit-unfinished" in scratch.message()


def test_bypass_each_gate_variable_is_named(scratch: Scratch) -> None:
    _real_hooks(scratch)
    gates = ("LUCID_SKIP_CLAIMS", "LUCID_SKIP_DOCSTRING", "LUCID_SKIP_XREFS")
    for number, gate in enumerate((*gates, "LUCID_SKIP_COMMIT_CONVENTION")):
        _commit(scratch, text=f"{number}\n", **{gate: "1"})
        assert f"Lucid-Bypass: {gate}" in scratch.message()
    _commit(scratch, text="cost\n", LUCID_SKIP_LOC_BADGE="1", LUCID_SKIP_API_DATA="1")
    assert "Lucid-Bypass" not in scratch.message(), "a cost variable is not a gate"


def test_bypass_a_script_the_tree_lacks_is_named(scratch: Scratch) -> None:
    _real_hooks(scratch)
    (scratch.root / "tools/check_stubs.py").unlink()
    _commit(scratch)
    assert "Lucid-Bypass: missing:check_stubs.py" in scratch.message()


def test_bypass_a_tree_without_the_rules_says_so_and_a_stale_branch_is_refused(
    scratch: Scratch,
) -> None:
    rules = "tools/rules/__main__.py"
    _real_hooks(scratch)
    linked = scratch.root.with_name(f"{scratch.root.name}-linked")
    scratch.git("worktree", "add", "-q", "-b", "side", str(linked))
    (linked / rules).unlink()
    there = ("-C", str(linked), "commit", "-qam")
    refused = scratch.attempt(*there, "chore: a branch that predates the rules")
    assert refused.returncode != 0 and "predates tools/rules" in refused.stderr, refused.stderr

    # With no main checkout's copy either, nothing could judge it, and the commit says that.
    (scratch.root / rules).unlink()
    _commit(scratch)
    assert "Lucid-Bypass: missing:rules/__main__.py" in scratch.message()


def test_bypass_a_replay_is_none_and_is_not_rewritten(scratch: Scratch) -> None:
    _real_hooks(scratch)
    scratch.git("checkout", "-qb", "side")
    _commit(scratch)
    scratch.write("src/b.txt", "b\n")
    scratch.git("add", "src/b.txt")
    assert scratch.attempt("commit", "-qm", "chore: second").returncode == 0
    side = scratch.sha()
    scratch.git("checkout", "-q", "main")
    scratch.write("src/m.txt", "m\n")
    scratch.git("add", "src/m.txt")
    assert scratch.attempt("commit", "-qm", "chore: on main").returncode == 0

    picked = scratch.attempt("cherry-pick", side)
    assert picked.returncode == 0, picked.stderr
    assert "Lucid-Bypass" not in scratch.message() and scratch.bypass_log == ""
    scratch.git("reset", "-q", "--hard", "HEAD~1")

    rebased = scratch.attempt("rebase", "main", "side")
    assert rebased.returncode == 0, rebased.stderr
    assert "Lucid-Bypass" not in scratch.attempt("log", "--format=%B", "main..HEAD").stdout
    assert scratch.bypass_log == ""
    assert scratch.attempt("rev-list", "--count", "main..HEAD").stdout.strip() == "2"


def test_bypass_lands_own_amend_is_none(scratch: Scratch) -> None:
    """``agent_ws land`` folds CHANGELOG in with this exact command."""
    _real_hooks(scratch)
    _commit(scratch)
    scratch.write("src/b.txt", "b\n")
    scratch.git("add", "src/b.txt")
    done = scratch.attempt(
        "commit", "--amend", "--no-edit", "--no-verify", LUCID_CHANGELOG_AMEND_IN_PROGRESS="1"
    )
    assert done.returncode == 0 and "Lucid-Bypass" not in scratch.message()


def test_bypass_the_changelog_fold_takes_only_the_changelog(scratch: Scratch) -> None:
    """The same amend, the same hazard: what else is staged waits for the next commit."""
    _real_hooks(scratch)
    scratch.write("tools/changelog.py", "open('CHANGELOG.md', 'a').write('an entry\\n')\n")
    scratch.write("CHANGELOG.md", "# log\n")
    scratch.git("add", "-A")
    assert scratch.attempt("commit", "-qm", "chore: tools").returncode == 0
    scratch.write("src/staged.txt", "s\n")
    scratch.git("add", "src/staged.txt")
    scratch.write("src/a.txt", "x\n")
    done = scratch.attempt("commit", "-qm", "fix: a path", "--", "src/a.txt")
    assert done.returncode == 0, done.stderr
    shown = scratch.attempt("show", "--name-only", "--format=", "HEAD").stdout.split()
    assert shown == ["CHANGELOG.md", "src/a.txt"]
    assert scratch.attempt("diff", "--cached", "--name-only").stdout.split() == ["src/staged.txt"]


def test_bypass_in_a_worktree_with_the_main_checkouts_python(scratch: Scratch) -> None:
    _real_hooks(scratch)
    linked = scratch.root.with_name(f"{scratch.root.name}-linked")
    scratch.git("worktree", "add", "-q", "-b", "side", str(linked))
    (linked / "src/a.txt").write_text("x\n")
    there = ("-C", str(linked), "commit", "-qam")
    cleared = scratch.attempt(*there, "chore: cleared")
    assert cleared.returncode == 0, cleared.stderr
    assert "Lucid-Bypass" not in scratch.attempt("log", "-1", "--format=%B", "side").stdout
    (linked / "src/a.txt").write_text("y\n")
    skipped = scratch.attempt(*there, "chore: skipped", "--no-verify")
    assert skipped.returncode == 0, skipped.stderr
    assert "Lucid-Bypass: no-verify" in scratch.attempt("log", "-1", "--format=%B", "side").stdout
    assert "chore: skipped" in scratch.bypass_log, "the log is the repository's, not the worktree's"


@pytest.mark.parametrize("python", ["missing", "the-wrong-version"])
def test_push_and_commit_refuse_without_the_pinned_interpreter(
    scratch: Scratch, python: str
) -> None:
    _real_hooks(scratch, python=False)
    remote = scratch.root.with_name(f"{scratch.root.name}-remote.git")
    scratch.attempt("init", "-q", "--bare", str(remote))
    scratch.git("remote", "add", "origin", str(remote))
    if python == "the-wrong-version":
        _stub_python(scratch.root, 'case "$1" in -c) exit 1 ;; esac\nexit 0\n')
    scratch.write("src/a.txt", "x\n")
    head = scratch.sha()
    commit = scratch.attempt("commit", "-qam", "chore: no")
    assert commit.returncode != 0 and "no Python 3.14" in commit.stderr, commit.stderr
    assert scratch.sha() == head
    push = scratch.attempt("push", "origin", "main")
    assert push.returncode != 0 and "no Python 3.14" in push.stderr, push.stderr
    assert scratch.attempt("--git-dir", str(remote), "rev-parse", "main").returncode != 0


# ── a real hook, broken: the audit's trial is what has to say so ─────────────

_RUN_LINE = (
    'if have "${REPO_ROOT}/${RULES_MAIN}" && ! (cd "${REPO_ROOT}" && '
    '"${PY}" -m tools.rules run --level commit --staged >&2); then'
)
_RUN_ONLY = '"${PY}" -m tools.rules run --level commit --staged'
_PUSH_LINE = 'cd "${REPO_ROOT}" && exec "${PY}" -m tools.rules push "$@"'


def _swap(old: str, new: str) -> Callable[[str], str]:
    def apply(text: str) -> str:
        assert old in text, old
        return text.replace(old, new, 1)

    return apply


_BROKEN_HOOKS = {
    "commit-exit-0-at-the-top": (
        "pre-commit",
        _swap("set -uo pipefail\n", "set -uo pipefail\nexit 0\n"),
        "did not run",
    ),
    "commit-line-gone": ("pre-commit", _swap(_RUN_LINE, "if false; then"), "did not run"),
    "commit-line-in-an-echo": (
        "pre-commit",
        _swap(_RUN_LINE, f'echo "{_RUN_ONLY}"\nif false; then'),
        "did not run",
    ),
    "commit-exit-dropped": (
        "pre-commit",
        _swap(_RUN_LINE, f"{_RUN_ONLY}\nif false; then"),
        "went through",
    ),
    "commit-off-the-index": ("pre-commit", _swap("--staged >&2", ">&2"), "did not run"),
    "commit-line-below-the-worktree-exit": (
        "pre-commit",
        lambda text: _swap(_RUN_LINE, "if false; then")(text) + f"\n{_RUN_ONLY} || exit 1\n",
        "did not run",
    ),
    "push-exit-0-at-the-top": (
        "pre-push",
        _swap("set -uo pipefail\n", "set -uo pipefail\nexit 0\n"),
        "did not run",
    ),
    "push-line-gone": ("pre-push", _swap(_PUSH_LINE, "true"), "did not run"),
    "push-off-the-refs": ("pre-push", _swap(_PUSH_LINE, f"{_PUSH_LINE} </dev/null"), "did not run"),
    "push-exit-dropped": (
        "pre-push",
        _swap(_PUSH_LINE, 'cd "${REPO_ROOT}" && "${PY}" -m tools.rules push "$@" || true'),
        "went through",
    ),
}


@pytest.mark.parametrize("name", _BROKEN_HOOKS)
def test_a_real_hook_broken_is_seen(tmp_path: Path, name: str) -> None:
    hook, broken, said = _BROKEN_HOOKS[name]
    shutil.copytree(REPO / ".githooks", tmp_path / ".githooks")
    level = Level.COMMIT if hook == "pre-commit" else Level.PUSH
    check = _real().checks["numpy-h4"]
    assert check.unwired(level, tmp_path) is None
    path = tmp_path / ".githooks" / hook
    path.write_text(broken(path.read_text()))
    problem = check.unwired(level, tmp_path)
    assert problem is not None and said in problem, problem


def test_a_real_hook_that_is_not_executable_is_seen(tmp_path: Path) -> None:
    shutil.copytree(REPO / ".githooks", tmp_path / ".githooks")
    (tmp_path / ".githooks/pre-push").chmod(0o644)
    problem = _real().checks["numpy-h4"].unwired(Level.PUSH, tmp_path)
    assert problem is not None and "did not run" in problem, problem


# ── real tools, real git: the incidents themselves ───────────────────────────


@contextlib.contextmanager
def _real_repository(root: Path, remote: Path) -> Iterator[Scratch]:
    """``root`` under git with the real hooks and a bare remote holding its one commit."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_") and k != "PYTHONPATH"}
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    repo = Scratch(root, env)
    repo.git("init", "-q", "-b", "main")
    repo.git("config", "user.name", "t")
    repo.git("config", "user.email", "t@example.invalid")
    repo.git("add", "-A")
    repo.git("commit", "-qm", "chore: base")
    repo.attempt("init", "-q", "--bare", str(remote))
    repo.git("remote", "add", "origin", str(remote))
    repo.git("push", "-q", "origin", "main")
    repo.git("config", "core.hooksPath", str(root / ".githooks"))
    try:
        yield repo
    finally:
        repo.git("config", "--unset", "core.hooksPath")
        repo.git(
            "reset",
            "-q",
            "--hard",
            repo.attempt("rev-list", "--max-parents=0", "HEAD").stdout.strip(),
        )
        repo.git("clean", "-fdq")
        shutil.rmtree(root / ".git")


_SLOP = "lucid/_seeded_slop.py"


@_needs_the_gate
def test_push_gate_refuses_a_rise_through_real_git(real_copy: Path, tmp_path: Path) -> None:
    with _real_repository(real_copy, tmp_path / "remote.git") as repo:
        out_there = repo.sha()
        repo.write(_SLOP, f'"""Seeded."""\n\n{_RISE}')
        repo.git("add", _SLOP)
        skipped = repo.attempt("commit", "-qm", "chore: three rises", "--no-verify")
        assert skipped.returncode == 0, skipped.stderr
        assert "Lucid-Bypass: no-verify" in repo.message()

        def on_the_remote() -> str:
            return repo.attempt("ls-remote", "origin", "refs/heads/main").stdout.split()[0]

        push = repo.attempt("push", "origin", "main")
        said = push.stdout + push.stderr
        assert push.returncode != 0, said
        assert re.search(r"VIOLATION\s+quality-gate-range", said), said
        assert "bypassed at commit (no-verify)" in said and "is refused" in said
        assert on_the_remote() == out_there

        # The R1 hole: the same range through the runner, and the gate against a ref, agree.
        def tool(*args: str) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                [sys.executable, "-S", "-m", *args],
                cwd=real_copy,
                env=repo.env,
                capture_output=True,
                text=True,
            )

        ranged = tool("tools.rules", "run", "--level", "push", "--range", "HEAD~1..HEAD")
        assert ranged.returncode == 1 and "quality-gate-range" in ranged.stdout, ranged.stdout
        collectors = ("--collectors", "ruff,counters,lizard")
        python = real_copy / ".venv/bin/python3"
        diff = subprocess.run(
            [python, "-m", "tools.quality_gate", "--diff", "HEAD~1", *collectors],
            cwd=real_copy,
            env=repo.env,
            capture_output=True,
            text=True,
        )
        assert diff.returncode == 1, diff.stdout + diff.stderr

        # A branch is none of the gate's business, whatever it holds.
        branch = repo.attempt("push", "origin", "HEAD:refs/heads/side")
        assert branch.returncode == 0, branch.stderr
        assert on_the_remote() == out_there

        # And the same push with the rise gone goes through, judged.
        repo.git("reset", "-q", "--hard", "HEAD~1")
        repo.write(_SLOP, '"""Seeded."""\n')
        repo.git("add", _SLOP)
        assert repo.attempt("commit", "-qm", "chore: no rise", "--no-verify").returncode == 0
        clean = repo.attempt("push", "origin", "main")
        assert clean.returncode == 0, clean.stdout + clean.stderr
        assert "push passes" in clean.stdout + clean.stderr
        assert on_the_remote() == repo.sha()


@_needs_the_gate
def test_a_real_commit_cannot_carry_a_blob_its_working_copy_hides(
    real_copy: Path, tmp_path: Path
) -> None:
    seed = "lucid/_seeded_blob.py"
    with _real_repository(real_copy, tmp_path / "remote.git") as repo:
        head = repo.sha()
        repo.write(seed, '"""Seeded."""\n\nimport numpy\n')
        repo.git("add", seed)
        repo.write(seed, '"""Seeded."""\n')
        refused = repo.attempt("commit", "-qm", "chore: hides a blob")
        said = refused.stdout + refused.stderr
        assert refused.returncode != 0 and f"not what is staged: {seed}" in said, said
        assert repo.sha() == head

        # The workers' normal form: named paths, staged as they are on disk.
        repo.git("reset", "-q")
        other = "lucid/_seeded_note.txt"
        repo.write(other, "a note\n")
        repo.git("add", "-N", other)
        kept = repo.attempt(
            "commit", "-qm", "chore: one path", "--", other, LUCID_SKIP_LOC_BADGE="1"
        )
        assert kept.returncode == 0, kept.stdout + kept.stderr
        assert "Lucid-Bypass" not in repo.message()
        assert repo.attempt("show", "--name-only", "--format=", "HEAD").stdout.split() == [other]


# ── the C++ at push: its ABI bump, and the record that its suites passed ─────

_ABI_TOOL = "tools/check_abi_bump.py"
_THING_H = "lucid/_C/core/Thing.h"
_THING = """\
#pragma once
#include "../api.h"

namespace lucid {

// A thing { with a brace in the comment.
class LUCID_API Thing {
public:
    explicit Thing(int size) : size_(size), seen_{0} {}
    virtual ~Thing() = default;
    virtual int count(int limit = 3, const char* why = "{") const;
    int size() const { return size_ + 1; }

private:
    int size_;
    int seen_ = 0;
};

struct Hidden {
    float weight;
};

struct alignas(16) Packed {
    alignas(8) float a;
};

using Index = int;

LUCID_API int make_thing(const Thing& thing, int n);
LUCID_API int shift_thing(int bits = 1 << 2, int count = 0);

}  // namespace lucid
"""
_BIND = 'void bind(py::module_& m) {\n    py::class_<Thing>(m, "Thing").def("count", &Thing::count);\n}\n'
_STUB = "class Thing:\n    def count(self) -> int: ...\n"
_ABI = "#define LUCID_ABI_VERSION {}\n"

# (id, path, the text there, its replacement, what the check must say about it)
_ABI_MOVES = [
    (
        "member",
        _THING_H,
        "    int seen_ = 0;\n",
        "    int seen_ = 0;\n    bool ready_;\n",
        "layout + bool ready_",
    ),
    (
        "internal-member",
        _THING_H,
        "float weight;\n",
        "float weight;\n    float bias;\n",
        "lucid::Hidden (internal) layout + float bias",
    ),
    (
        "member-order",
        _THING_H,
        "    int size_;\n    int seen_ = 0;\n",
        "    int seen_ = 0;\n    int size_;\n",
        "members reordered",
    ),
    (
        "virtual",
        _THING_H,
        "    virtual ~Thing() = default;\n",
        "    virtual ~Thing() = default;\n    virtual void reset();\n",
        "layout + virtual void reset",
    ),
    (
        "member-function",
        _THING_H,
        "    int size() const",
        "    int area() const;\n    int size() const",
        "(exported) + int area ( ) const",
    ),
    (
        "exported-type",
        _THING_H,
        "struct Hidden {",
        "struct LUCID_API Shown {};\nstruct Hidden {",
        "lucid::Shown (exported) added",
    ),
    (
        "export",
        _THING_H,
        "LUCID_API int make_thing",
        "LUCID_API void drop_thing();\nLUCID_API int make_thing",
        "LUCID_API + LUCID_API void drop_thing ( )",
    ),
    (
        "signature",
        _THING_H,
        "const Thing& thing, int n)",
        "const Thing& thing, long n)",
        "LUCID_API + LUCID_API int make_thing ( const Thing &, long )",
    ),
    (
        "binding",
        "lucid/_C/bindings/bind.cpp",
        "&Thing::count)",
        '&Thing::count).def("reset", &Thing::reset)',
        "binding + def reset",
    ),
    (
        "stub",
        "lucid/_C/engine.pyi",
        "-> int: ...\n",
        "-> int: ...\n    def reset(self) -> None: ...\n",
        "engine.pyi + Thing.reset",
    ),
    (
        "aligned-struct",
        _THING_H,
        "    alignas(8) float a;\n",
        "    alignas(8) float a;\n    float b;\n",
        "lucid::Packed (internal) layout + float b",
    ),
    (
        "aligned-member",
        _THING_H,
        "alignas(8) float a",
        "alignas(16) float a",
        "lucid::Packed (internal) layout + alignas ( 16 ) float a",
    ),
    (
        "alias",
        _THING_H,
        "using Index = int;",
        "using Index = long;",
        "alias lucid::Index: int -> long",
    ),
    (
        "after-a-shift",
        _THING_H,
        "int count = 0)",
        "long count = 0)",
        "LUCID_API + LUCID_API int shift_thing ( int, long )",
    ),
]
_ABI_STILL = [
    (
        "comment-and-names",
        _THING_H,
        "// A thing { with a brace in the comment.\nclass LUCID_API Thing {",
        "/* Renamed { */ class LUCID_API\nThing\n{",
    ),
    (
        "parameter-names",
        _THING_H,
        'int limit = 3, const char* why = "{")',
        'int most = 4, const char *reason = "}")',
    ),
    ("inline-body", _THING_H, "{ return size_ + 1; }", "{\n        return size_ * 2;\n    }"),
    ("initializers", _THING_H, "seen_{0} {}", "seen_{1} { size_ += 0; }"),
    ("default-member", _THING_H, "int seen_ = 0;", "int seen_{7};"),
    (
        "free-parameter",
        _THING_H,
        "const Thing& thing, int n)",
        "const Thing &other,\n                    int count)",
    ),
    ("moved-to-another-header", _THING_H, "struct Hidden {\n    float weight;\n};\n", ""),
    ("source-body", "lucid/_C/core/Thing.cpp", "return 1;", "return 2;"),
    ("new-alias", _THING_H, "using Index = int;", "using Index = int;\nusing Offset = long;"),
    (
        "named-class-binding",
        "lucid/_C/bindings/bind.cpp",
        '    py::class_<Thing>(m, "Thing").def(',
        '    py::class_<Thing> thing(m, "Thing");\n    thing.def(',
    ),
]


def _abi_repo(s: Scratch) -> Callable[..., subprocess.CompletedProcess[str]]:
    """A repository holding one exported class, its binding and its stub, at ABI 7."""
    s.write(_ABI_TOOL, (REPO / _ABI_TOOL).read_text())
    s.write(_THING_H, _THING)
    s.write("lucid/_C/core/Thing.cpp", "int thing() { return 1; }\n")
    s.write("lucid/_C/bindings/bind.cpp", _BIND)
    s.write("lucid/_C/engine.pyi", _STUB)
    s.write("lucid/_C/version.h", _ABI.format(7))
    s.commit_all("chore: base")

    def check(*args: str) -> subprocess.CompletedProcess[str]:
        argv = [sys.executable, "-S", _ABI_TOOL, *args]
        return subprocess.run(argv, cwd=s.root, env=s.env, capture_output=True, text=True)

    return check


def _swap_in(s: Scratch, path: str, old: str, new: str) -> None:
    text = (s.root / path).read_text()
    assert old in text, old
    s.write(path, text.replace(old, new, 1))


@pytest.mark.parametrize(
    ("path", "old", "new", "said"), [m[1:] for m in _ABI_MOVES], ids=[m[0] for m in _ABI_MOVES]
)
def test_abi_bump_sees_each_way_the_interface_moves(
    scratch: Scratch, path: str, old: str, new: str, said: str
) -> None:
    check = _abi_repo(scratch)
    _swap_in(scratch, path, old, new)
    scratch.commit_all("feat: a move")
    done = check("--since", "HEAD~1")
    assert done.returncode == 1 and said in done.stdout, done.stdout + done.stderr
    assert "LUCID_ABI_VERSION is still 7" in done.stdout and "moved by" in done.stdout
    assert "ABI: none (<why>)" in done.stdout


@pytest.mark.parametrize(
    ("path", "old", "new"), [m[1:] for m in _ABI_STILL], ids=[m[0] for m in _ABI_STILL]
)
def test_abi_bump_lets_through_what_moves_nothing(
    scratch: Scratch, path: str, old: str, new: str
) -> None:
    check = _abi_repo(scratch)
    _swap_in(scratch, path, old, new)
    if not new:  # the struct moves to a header of its own
        scratch.write(
            "lucid/_C/core/Hidden.h", "namespace lucid {\nstruct Hidden { float weight; };\n}\n"
        )
    scratch.commit_all("refactor: nothing moves")
    done = check("--since", "HEAD~1")
    assert (done.returncode, done.stdout) == (0, ""), done.stdout + done.stderr


def test_abi_bump_takes_a_bump_anywhere_in_the_range(scratch: Scratch) -> None:
    check = _abi_repo(scratch)
    base = scratch.sha()
    _swap_in(scratch, _THING_H, "    int size_;\n", "    int size_;\n    bool ready_;\n")
    scratch.commit_all("feat: a member")
    scratch.write("lucid/_C/version.h", _ABI.format(8))
    scratch.commit_all("chore: bump the ABI to 8")
    done = check("--range", f"{base}..HEAD")
    assert done.returncode == 0 and "went 7 -> 8" in done.stdout, done.stdout
    # Commit by commit, the member came without one: the backtest's view.
    each = check("--range", f"{base}..HEAD", "--each")
    assert each.returncode == 1 and "MISSING" in each.stdout and "feat: a member" in each.stdout
    assert "chore: bump" not in each.stdout  # moves nothing itself
    # A counter that went down moved nowhere.
    scratch.write("lucid/_C/version.h", _ABI.format(6))
    scratch.commit_all("chore: the counter backwards")
    assert check("--range", f"{base}..HEAD").returncode == 1


_READY = ("    int size_;\n", "    int size_;\n    bool ready_;\n")
_MORE = ("    int seen_ = 0;\n", "    int seen_ = 0;\n    int more_;\n")
_BIND_H = "lucid/_C/bindings/bind.cpp"
_RESET = ("&Thing::count)", '&Thing::count).def("reset", &Thing::reset)')
_SAYS_WHY = "ABI: none (the member is read by no one outside)"


def _move(s: Scratch, edit: tuple[str, str], trailer: str = "", path: str = _THING_H) -> str:
    """Commit one move of the interface, say what the trailer says; its sha."""
    _swap_in(s, path, *edit)
    s.commit_all("feat: a move" + (f"\n\n{trailer}" if trailer else ""))
    return s.sha()


def _said(s: Scratch, trailer: str) -> None:
    """A commit that moves nothing and says only this."""
    s.write(f"src/{s.sha()[:8]}.txt", "ok\n")
    s.commit_all(f"chore: say it\n\n{trailer}")


def _held(done: subprocess.CompletedProcess[str]) -> list[str]:
    return [line.split()[2] for line in done.stdout.splitlines() if "moved by" in line]


@pytest.mark.parametrize(
    ("trailer", "passes"),
    [
        ("ABI: none", False),  # what most commits say by habit
        ("ABI: none (40)", False),  # a number is no reason
        ("ABI: bump needed; Thing gains ready_", False),
        (_SAYS_WHY, True),
    ],
    ids=["bare", "a-number", "bump-needed", "a-reason"],
)
def test_abi_bump_takes_a_reason_not_a_habit(scratch: Scratch, trailer: str, passes: bool) -> None:
    check = _abi_repo(scratch)
    base = scratch.sha()
    moved = _move(scratch, _READY, trailer)
    done = check("--since", base)
    assert (done.returncode == 0) is passes, done.stdout
    if passes:
        assert (
            "note: abi-bump excused by" in done.stdout and "read by no one outside" in done.stdout
        )
        # A reason covers what its commit moved, not what comes after.
        later = _move(scratch, _MORE)
        done = check("--since", base)
        assert done.returncode == 1 and _held(done) == [later[:9]], done.stdout
    else:
        assert _held(done) == [moved[:9]], done.stdout


def test_abi_bump_excuses_what_a_commit_moved_not_what_another_did(scratch: Scratch) -> None:
    """The LCD-900 shape: a move, then an unrelated commit with a reason of its own."""
    check = _abi_repo(scratch)
    base = scratch.sha()
    first = _move(scratch, _READY)
    _said(scratch, "ABI: none (Python only)")
    done = check("--since", base)
    assert done.returncode == 1 and _held(done) == [first[:9]], done.stdout
    second = _move(scratch, _MORE, _SAYS_WHY)  # its own move is its to excuse
    done = check("--since", base)
    assert done.returncode == 1 and _held(done) == [first[:9]], done.stdout
    assert second[:9] not in "".join(_held(done))


def test_abi_bump_takes_a_later_reason_that_names_the_commit(scratch: Scratch) -> None:
    check = _abi_repo(scratch)
    base = scratch.sha()
    first = _move(scratch, _READY)
    _said(scratch, f"ABI: none ({first[:6]} is a member nothing outside reads)")  # six is no sha
    assert check("--since", base).returncode == 1
    _said(scratch, f"ABI: none ({first[:7]} is a member nothing outside reads)")
    done = check("--since", base)
    assert done.returncode == 0, done.stdout
    assert f", and {first[:9]}" in done.stdout, done.stdout
    # It names that commit, so it cannot speak for another.
    other = _move(scratch, _MORE)
    done = check("--since", base)
    assert done.returncode == 1 and _held(done) == [other[:9]], done.stdout


def test_abi_bump_takes_a_later_reason_that_names_every_symbol(scratch: Scratch) -> None:
    check = _abi_repo(scratch)
    base = scratch.sha()
    _swap_in(scratch, _THING_H, *_READY)
    _move(scratch, _RESET, path=_BIND_H)  # one commit, two things: Thing and reset
    _said(scratch, "ABI: none (a Python-side note only)")
    assert check("--since", base).returncode == 1
    _said(scratch, "ABI: none (Thing is private to the engine)")
    done = check("--since", base)
    assert done.returncode == 1 and "[Thing, reset]" in done.stdout, done.stdout  # reset unnamed
    _said(scratch, "ABI: none (Thing and reset are private to the engine)")
    done = check("--since", base)
    assert done.returncode == 0 and "note: abi-bump excused by" in done.stdout, done.stdout


def test_abi_bump_asks_for_a_sha_where_a_name_is_too_short_to_mean_anything(
    scratch: Scratch,
) -> None:
    check = _abi_repo(scratch)
    base = scratch.sha()
    first = _move(scratch, ("&Thing::count)", '&Thing::count).def("go", &Thing::go)'), path=_BIND_H)
    _said(scratch, "ABI: none (go is a name no caller outside uses)")
    done = check("--since", base)
    assert done.returncode == 1 and _held(done) == [first[:9]], done.stdout
    _said(scratch, f"ABI: none ({first[:7]} binds a name no caller outside uses)")
    assert check("--since", base).returncode == 0


def test_abi_bump_needed_is_excused_by_nothing_but_the_bump(scratch: Scratch) -> None:
    check = _abi_repo(scratch)
    base = scratch.sha()
    first = _move(scratch, _READY, "ABI: bump needed; Thing gains ready_")
    for trailer in (
        "ABI: none (Python only)",
        f"ABI: none ({first[:7]} Thing gains ready_, which nothing outside reads)",
    ):
        _said(scratch, trailer)
        done = check("--since", base)
        assert done.returncode == 1 and _held(done) == [first[:9]], done.stdout
        assert "bump needed" in done.stdout
    # Not even a reason of its own, written beside the demand for a bump.
    _move(scratch, _MORE, "ABI: bump needed; more_\nABI: none (it is read by no one outside)")
    assert check("--since", base).returncode == 1
    scratch.write("lucid/_C/version.h", _ABI.format(8))
    scratch.commit_all("chore: bump the ABI to 8")
    done = check("--since", base)
    assert done.returncode == 0 and "went 7 -> 8" in done.stdout, done.stdout


def test_abi_bump_forgets_a_move_that_a_later_commit_undid(scratch: Scratch) -> None:
    check = _abi_repo(scratch)
    base = scratch.sha()
    _move(scratch, _READY)
    _move(scratch, _READY[::-1])  # takes the member out again
    hidden = ("float weight;\n", "float weight;\n    float bias;\n")
    last = _move(scratch, hidden, _SAYS_WHY)
    done = check("--since", base)
    assert done.returncode == 0 and last[:9] in done.stdout, done.stdout
    assert "moved by" not in done.stdout


def test_abi_bump_reads_a_range_or_says_it_cannot(scratch: Scratch) -> None:
    check = _abi_repo(scratch)
    assert check("--range", "nowhere..HEAD").returncode == 2
    assert check("--range", "HEAD").returncode == 2
    done = check("--since", EMPTY_TREE)  # a remote with nothing on it
    assert done.returncode == 0 and "holds nothing" in done.stdout, done.stdout + done.stderr
    scratch.write("lucid/_C/engine.pyi", "def broken(:\n")
    scratch.commit_all("chore: a stub that does not parse")
    done = check("--since", "HEAD~1")
    assert done.returncode == 2 and "does not parse" in done.stderr, done.stderr


def test_abi_bump_flags_the_commits_that_shipped_without_one() -> None:
    """The backtest, pinned: 578247379 moved Node's vtable at ABI 40; LCD-837 bumped in range."""
    history = ("578247379", "60b4519a0", "d324733a2")
    known = [_git_ok(REPO, "cat-file", "-e", f"{sha}^{{commit}}") for sha in history]
    if not all(known):
        pytest.skip("a shallow clone holds no history to backtest")

    def check(spec: str) -> subprocess.CompletedProcess[str]:
        argv = [sys.executable, "-S", _ABI_TOOL, "--range", spec]
        return subprocess.run(argv, cwd=REPO, capture_output=True, text=True)

    done = check("578247379~1..578247379")
    assert done.returncode == 1, done.stdout + done.stderr
    assert "lucid::Node (exported) layout + virtual void pack_saved" in done.stdout
    assert "still 40" in done.stdout
    assert check("60b4519a0~1..60b4519a0").returncode == 1  # the Tracer's members, alone
    done = check("60b4519a0~1..d324733a2")  # with the bump that followed them
    assert done.returncode == 0 and "went 42 -> 43" in done.stdout, done.stdout


def _git_ok(root: Path, *args: str) -> bool:
    return subprocess.run(["git", *args], cwd=root, capture_output=True).returncode == 0


_FAKE_ENGINE = """\
import pathlib, types
_C_engine = types.SimpleNamespace(
    ABI_VERSION={abi}, __file__=str(pathlib.Path({where!r}) / "engine.cpython-314-darwin.so")
)
"""
_SO = "lucid/_C/engine.cpython-314-darwin.so"


_PROJECT = '[project]\nname = "thing"\nversion = "1"\n\n[tool.lucid.mlx]\nversion = "0.1"\n'


def _attest_repo(s: Scratch, suite: str = "def test_ok():\n    pass\n") -> None:
    """A checkout whose engine is built from its C++, and whose suites pass."""
    site = Path(pytest.__file__).resolve().parents[1]
    # -S: the light venv's finder would hand over this worktree's lucid in place of the scratch's.
    s.write(
        ".venv/bin/python3", f'#!/bin/sh\nPYTHONPATH="{site}" exec "{sys.executable}" -S "$@"\n'
    )
    (s.root / ".venv/bin/python3").chmod(0o755)
    s.write(".gitignore", ".venv/\n__pycache__/\n*.so\n")
    s.write("lucid/__init__.py", _FAKE_ENGINE.format(abi=7, where=str(s.root / "lucid/_C")))
    s.write("lucid/_C/version.h", _ABI.format(7))
    s.write("lucid/_C/core/Thing.cpp", "int thing() { return 1; }\n")
    for native in attest.NATIVE:  # what builds the engine besides its sources
        s.write(native, "built = True\n")
    s.write("pyproject.toml", f"{(s.root / 'pyproject.toml').read_text()}\n{_PROJECT}")
    for path in attest.SUITES:
        s.write(f"{path}/test_suite.py", suite)
    s.commit_all("chore: base")
    s.write(_SO, "")  # newer than every source, as a build leaves it


def _rules_in(
    s: Scratch, *args: str, home: Path = REPO, **env: str
) -> subprocess.CompletedProcess[str]:
    argv = [sys.executable, "-S", "-m", "tools.rules", "--root", str(s.root), *args]
    return subprocess.run(argv, cwd=home, env={**s.env, **env}, capture_output=True, text=True)


def _record_of(s: Scratch) -> Path:
    key = attest.engine_key(Workspace(s.root, sys.executable, isolated=True))
    return s.root / ".git/lucid-attest" / f"{key}.json"


def test_attest_records_a_pass_that_push_then_finds(scratch: Scratch) -> None:
    _attest_repo(scratch)
    done = _rules_in(scratch, "attest")
    assert done.returncode == 0, done.stdout + done.stderr
    record = json.loads(_record_of(scratch).read_text())
    assert [run["tests"] for run in record["suites"]] == [1, 1] and record["abi"] == 7
    assert record["commit"] == scratch.sha() and "passed on" in done.stdout
    assert _rules_in(scratch, "attest", "--verify").returncode == 0
    # Python moves on, the C++ tree does not: the record still holds, in any worktree.
    scratch.write("lucid/extra.py", "x = 1\n")
    scratch.commit_all("feat: python only")
    assert _rules_in(scratch, "attest", "--verify").returncode == 0
    linked = scratch.root.parent / "linked"
    scratch.git("worktree", "add", "-q", str(linked))
    assert _rules_in(Scratch(linked, scratch.env), "attest", "--verify").returncode == 0
    # What builds the engine moves (the MLX pin, setup.py): the key follows; a version does not.
    for path, old, new, code in (
        ("pyproject.toml", 'version = "0.1"', 'version = "9"', 1),
        ("setup.py", "built = True", "built = False", 1),
        ("pyproject.toml", 'version = "1"', 'version = "2"', 0),
    ):
        _swap_in(scratch, path, old, new)
        scratch.commit_all(f"build: {path} moves")
        assert _rules_in(scratch, "attest", "--verify").returncode == code, (path, new)
        scratch.git("reset", "-q", "--hard", "HEAD~1")
    assert _rules_in(scratch, "attest", "--verify").returncode == 0
    # The C++ moves: no record names its key, and the refusal says what to run.
    scratch.write("lucid/_C/core/Thing.cpp", "int thing() { return 2; }\n")
    scratch.commit_all("fix: the C++")
    done = _rules_in(scratch, "attest", "--verify")
    assert done.returncode == 1 and "has no record" in done.stdout, done.stdout
    assert ".venv/bin/python3 -m tools.rules attest" in done.stdout


def _dirty_cpp(s: Scratch) -> None:
    s.write("lucid/_C/core/Thing.cpp", "int thing() { return 3; }\n")


def _untracked(s: Scratch) -> None:
    s.write("lucid/_C/core/New.cpp", "int added() { return 0; }\n")


def _stale(s: Scratch) -> None:
    s.write("lucid/_C/core/Thing.cpp", "int thing() { return 3; }\n")
    s.commit_all("fix: C++ the engine was not built from")
    later = (s.root / _SO).stat().st_mtime + 5
    os.utime(s.root / "lucid/_C/core/Thing.cpp", (later, later))


def _no_engine(s: Scratch) -> None:
    (s.root / _SO).unlink()


def _engine_elsewhere(s: Scratch) -> None:
    s.write("lucid/__init__.py", _FAKE_ENGINE.format(abi=7, where="/elsewhere/lucid/_C"))
    s.commit_all("chore: an engine from another checkout")
    s.write(_SO, "")


def _dirty_pyproject(s: Scratch) -> None:
    s.write("pyproject.toml", (s.root / "pyproject.toml").read_text() + "\n# an edit\n")


def _dirty_setup(s: Scratch) -> None:
    s.write("setup.py", "built = False\n")


def _committed(path: str, old: str, new: str) -> Callable[[Scratch], None]:
    def spoil(s: Scratch) -> None:
        _swap_in(s, path, old, new)
        s.commit_all(f"build: {path} moves")

    return spoil


def _suite(text: str, *, both: bool = False) -> Callable[[Scratch], None]:
    """The compile suite (or both) as ``text``, committed: what the run then finds."""

    def spoil(s: Scratch) -> None:
        for path in attest.SUITES if both else attest.SUITES[:1]:
            s.write(f"{path}/test_suite.py", text)
        s.commit_all("test: the suite as it is now")
        s.write(_SO, "")

    return spoil


_SKIPPED = "import pytest\n@pytest.mark.skip\ndef test_skip():\n    pass\n"
_FAILING = "def test_ok():\n    pass\ndef test_bad():\n    assert False\n"
# What a suite does to the checkout while it runs.
_MID_RUN = "import pathlib, subprocess\ndef test_moves_the_checkout():\n    {}\n"
_RUN = 'subprocess.run(["git", *{}], check=True)'
_CPP_COMMIT = f'pathlib.Path("lucid/_C/core/Thing.cpp").write_text("int t;"); {_RUN.format(["commit", "-qam", "x"])}'
_PY_COMMIT = f'pathlib.Path("lucid/late.py").write_text("x = 1"); {_RUN.format(["add", "lucid/late.py"])}; {_RUN.format(["commit", "-qm", "x"])}'
_EDIT = 'pathlib.Path("lucid/late.py").write_text("x = 1")'
_UNSELECTED = (
    "import os\ndef test_env():\n"
    "    assert not {'PYTEST_PLUGINS', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD'} & set(os.environ)\n"
)
_VERIFY = ("attest", "--verify")


@dataclass(frozen=True)
class Attempt:
    """An attest run in a checkout spoiled one way, and how it must end.

    ``first``: a passing attest before the spoil, by the same code, so that ``args`` (a
    lookup) answers for the key that code computes.
    """

    spoil: Callable[[Scratch], None]
    code: int
    said: str
    env: dict[str, str] = field(default_factory=dict)
    args: tuple[str, ...] = ("attest",)
    first: bool = False


_ATTEMPTS = {
    "dirty": Attempt(_dirty_cpp, 2, "not committed"),
    "untracked": Attempt(_untracked, 2, "not committed"),
    "dirty-pyproject": Attempt(_dirty_pyproject, 2, "not committed"),
    "dirty-setup": Attempt(_dirty_setup, 2, "not committed"),
    "stale-engine": Attempt(_stale, 2, "missing or older than lucid/_C"),
    "no-engine": Attempt(_no_engine, 2, "missing or older than lucid/_C"),
    "engine-elsewhere": Attempt(_engine_elsewhere, 2, "and its engine from /elsewhere/lucid/_C"),
    "no-tests": Attempt(_suite("import pytest\n", both=True), 2, "ran no test (0 selected"),
    "all-skipped": Attempt(_suite(_SKIPPED), 2, "ran no test (1 selected, 1 skipped)"),
    "mostly-skipped": Attempt(_suite(f"{_SKIPPED}def test_ok():\n    pass\n"), 2, "skipped 1 of 2"),
    "cpp-moved-mid-run": Attempt(_suite(_MID_RUN.format(_CPP_COMMIT)), 2, "during the run"),
    "head-moved-mid-run": Attempt(_suite(_MID_RUN.format(_PY_COMMIT)), 2, "during the run"),
    "edited-mid-run": Attempt(_suite(_MID_RUN.format(_EDIT)), 2, "during the run"),
    # A selection from the environment would have hidden the failure: it is dropped.
    "deselected": Attempt(
        _suite(_FAILING),
        1,
        "FAILED on",
        {"PYTEST_ADDOPTS": "--deselect=lucid/test/unit/compile/test_suite.py::test_bad"},
    ),
    # Autoload off loses the benchmark plugin a compile test's fixture needs.
    "plugins-env": Attempt(
        _suite(_UNSELECTED),
        0,
        "attest ignores PYTEST_DISABLE_PLUGIN_AUTOLOAD=",
        {"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTEST_PLUGINS": ""},
    ),
    # The key reads what builds the engine, and only that.
    "setup-moved": Attempt(
        _committed("setup.py", "built = True", "built = False"),
        1,
        "has no record",
        args=_VERIFY,
        first=True,
    ),
    "version-moved": Attempt(
        _committed("pyproject.toml", 'version = "1"', 'version = "2"'),
        0,
        "attested: key",
        args=_VERIFY,
        first=True,
    ),
}
_REFUSED = [name for name, attempt in _ATTEMPTS.items() if attempt.code == 2]


def _attempt(s: Scratch, name: str, home: Path = REPO) -> subprocess.CompletedProcess[str]:
    attempt = _ATTEMPTS[name]
    _attest_repo(s)
    if attempt.first:
        _rules_in(s, "attest", home=home)
    attempt.spoil(s)
    return _rules_in(s, *attempt.args, home=home, **attempt.env)


def _ended_as_said(done: subprocess.CompletedProcess[str], name: str) -> bool:
    attempt = _ATTEMPTS[name]
    return done.returncode == attempt.code and attempt.said in done.stdout + done.stderr


@pytest.mark.parametrize("name", _REFUSED)
def test_attest_refuses_what_it_cannot_vouch_for(scratch: Scratch, name: str) -> None:
    done = _attempt(scratch, name)
    assert _ended_as_said(done, name) and _ATTEMPTS[name].said in done.stderr, (
        done.stdout + done.stderr
    )
    assert not (scratch.root / ".git/lucid-attest").exists()


@pytest.mark.parametrize("name", ["deselected", "plugins-env"])
def test_attest_runs_the_suites_as_committed(scratch: Scratch, name: str) -> None:
    done = _attempt(scratch, name)
    assert _ended_as_said(done, name), done.stdout + done.stderr
    assert all(f"attest ignores {var}=" in done.stdout for var in _ATTEMPTS[name].env)
    record = json.loads(_record_of(scratch).read_text())
    assert record["suites"][0]["failures"] == int(name == "deselected")


@pytest.mark.parametrize("name", ["setup-moved", "version-moved"])
def test_attest_keys_what_builds_the_engine(scratch: Scratch, name: str) -> None:
    done = _attempt(scratch, name)
    assert _ended_as_said(done, name), done.stdout + done.stderr


# (id, the line in attest.py, its replacement, the attempt that must now end otherwise)
_ATTEST_MUTANTS = [
    (
        "skips-read-as-runs",
        "if self.tests - self.skipped <= 0:",
        "if self.tests <= 0:",
        "all-skipped",
    ),
    (
        "skips-unbounded",
        "_MOST_SKIPPED: Final = 0.05",
        "_MOST_SKIPPED: Final = 1.0",
        "mostly-skipped",
    ),
    ("selection-passed-on", "if name not in _SELECTION_ENV}", "if name}", "deselected"),
    (
        "pyproject-not-read",
        '_CLEAN: Final = ("lucid", "pyproject.toml", *NATIVE)',
        '_CLEAN: Final = ("lucid", *NATIVE)',
        "dirty-pyproject",
    ),
    (
        "no-recheck",
        'elif runs and (engine_key(ws), _head(ws), _dirty(ws)) != (key, commit, ""):',
        "elif False:",
        "cpp-moved-mid-run",
    ),
    (
        "head-not-rechecked",
        '(engine_key(ws), _head(ws), _dirty(ws)) != (key, commit, "")',
        '(engine_key(ws), _dirty(ws)) != (key, "")',
        "head-moved-mid-run",
    ),
    (
        "K4-native-inputs-unkeyed",
        'blobs += [ws.spawn(["git", "rev-parse", "-q", "--verify", f"HEAD:{p}"]).stdout for p in NATIVE]',
        "blobs += []",
        "setup-moved",
    ),
    ("K6-whole-pyproject-keyed", "json.dumps(pin, sort_keys=True)", "project", "version-moved"),
    (
        "K7-native-inputs-not-clean",
        '_CLEAN: Final = ("lucid", "pyproject.toml", *NATIVE)',
        '_CLEAN: Final = ("lucid", "pyproject.toml")',
        "dirty-setup",
    ),
    ("autoload-passed-on", ', "PYTEST_DISABLE_PLUGIN_AUTOLOAD")', ")", "plugins-env"),
    ("plugins-passed-on", '"PYTEST_PLUGINS", ', "", "plugins-env"),
]


@pytest.mark.parametrize(
    ("line", "new", "name"), [m[1:] for m in _ATTEST_MUTANTS], ids=[m[0] for m in _ATTEST_MUTANTS]
)
def test_a_line_broken_in_attest_is_seen(
    scratch: Scratch, tmp_path_factory: pytest.TempPathFactory, line: str, new: str, name: str
) -> None:
    home = tmp_path_factory.mktemp("mutant")
    shutil.copytree(
        REPO / "tools/rules", home / "tools/rules", ignore=shutil.ignore_patterns("__pycache__")
    )
    shutil.copy(REPO / "tools/agent_ws.py", home / "tools/agent_ws.py")
    source = home / "tools/rules/attest.py"
    assert line in source.read_text(), line
    source.write_text(source.read_text().replace(line, new, 1))
    done = _attempt(scratch, name, home=home)
    out = done.stdout + done.stderr
    assert "Traceback" not in out, out  # a crash is no proof that the line is needed
    assert not _ended_as_said(done, name), f"survived: {line}"


def test_attest_records_a_failure_and_push_refuses_it(scratch: Scratch) -> None:
    _attest_repo(scratch, suite="def test_ok():\n    pass\ndef test_bad():\n    assert False\n")
    done = _rules_in(scratch, "attest")
    assert done.returncode == 1 and "FAILED on" in done.stdout, done.stdout + done.stderr
    assert json.loads(_record_of(scratch).read_text())["suites"][0]["failures"] == 1
    done = _rules_in(scratch, "attest", "--verify")
    assert done.returncode == 1 and "has a record that FAILED" in done.stdout, done.stdout


def test_attest_a_record_that_does_not_read_is_none(scratch: Scratch) -> None:
    _attest_repo(scratch)
    assert _rules_in(scratch, "attest").returncode == 0
    record = _record_of(scratch)
    table = json.loads(record.read_text())
    for spoiled in (
        "{",
        json.dumps({**table, "suites": table["suites"][:1]}),
        json.dumps({**table, "abi": "7"}),
    ):
        record.write_text(spoiled)
        done = _rules_in(scratch, "attest", "--verify")
        assert done.returncode == 1, (spoiled, done.stdout)


@_needs_the_gate
def test_a_cpp_push_is_judged_through_real_git(real_copy: Path, tmp_path: Path) -> None:
    """The records' own run lines through the real hook: ``{since}``, ``--verify``, triggers.

    A control that runs only the tool stays green with ``{since}`` gone or a trigger
    narrowed; a header below lucid/_C and one at its top each go out alone.
    """
    with _real_repository(real_copy, tmp_path / "remote.git") as repo:

        def on_the_remote() -> str:
            return repo.attempt("ls-remote", "origin", "refs/heads/main").stdout.split()[0]

        def refused(header: str, name: str) -> None:
            seeded = f"namespace lucid {{\nstruct LUCID_API {name} {{\n    int seeded_;\n}};\n}}\n"
            repo.write(header, (repo.root / header).read_text() + seeded)
            repo.git("add", header)
            commit = repo.attempt("commit", "-qm", f"feat: {name}", "--no-verify")
            assert commit.returncode == 0, commit.stderr
            push = repo.attempt("push", "origin", "main")
            said = push.stdout + push.stderr
            assert push.returncode != 0 and on_the_remote() == out_there, said
            assert re.search(r"VIOLATION\s+abi-bump", said), said
            assert f"lucid::{name} (exported) added" in said, said
            assert re.search(r"VIOLATION\s+cpp-attested", said), said
            assert ".venv/bin/python3 -m tools.rules attest" in said

        out_there = repo.sha()
        refused("lucid/_C/core/Shape.h", "SeededBelow")
        repo.git("reset", "-q", "--hard", out_there)
        refused("lucid/_C/api.h", "SeededAtTheTop")

        # The bump in the same range, and the record an attest run would have left.
        version = repo.root / "lucid/_C/version.h"
        found = re.search(r"LUCID_ABI_VERSION (\d+)\n", version.read_text())
        assert found is not None
        abi = int(found[1]) + 1
        version.write_text(version.read_text().replace(found[0], f"LUCID_ABI_VERSION {abi}\n"))
        repo.git("add", "lucid/_C/version.h")
        assert repo.attempt("commit", "-qm", "chore: bump the ABI", "--no-verify").returncode == 0
        ws = Workspace(repo.root, sys.executable, isolated=True)
        runs = tuple(attest.SuiteRun(path, 0, 1, 0, 0) for path in attest.SUITES)
        made = attest.Attestation(attest.engine_key(ws), repo.sha(), "t", "now", 1.0, abi, runs)
        attest.Ledger.of(ws).record(made)
        push = repo.attempt("push", "origin", "main")
        said = push.stdout + push.stderr
        assert re.search(r"PASS\s+abi-bump", said) and re.search(r"PASS\s+cpp-attested", said)
        # Debt main carries is the ratchet's to judge, not these two records'.
        refusers = set(re.findall(r"(?:VIOLATION|UNMEASURED)\s+(\S+)", said))
        assert (push.returncode == 0) == (not refusers), said
        assert (on_the_remote() == repo.sha()) == (not refusers), said


# ── the real registry ────────────────────────────────────────────────────────


@functools.cache
def _real() -> Registry:
    return Registry.load(REPO)


def _clone(root: Path, *relatives: str) -> None:
    for relative in relatives:
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["cp", "-cR", str(REPO / relative), str(root / relative)], check=True)


@pytest.fixture(scope="session")
def real_copy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """This repository's rule-bearing files, shared by the mutant cases, which restore them."""
    root = tmp_path_factory.mktemp("real").resolve()
    _clone(root, "tools", ".githooks", "scripts", ".github", "pyproject.toml", "lucid")
    _clone(root, ".gitignore", "setup.py")  # E-VERIFY-CPP's key and trigger read setup.py
    finder = REPO / ".venv/lib/python3.14/site-packages/_lucid_worktree_finder.py"
    if finder.exists():  # a worktree's light venv pins lucid and tools to the worktree
        _clone(root, ".venv")
        copy = root / finder.relative_to(REPO)
        copy.write_text(copy.read_text().replace(str(REPO), str(root)))
    else:  # the hooks ask for this path and take no other
        python = root / ".venv/bin/python3"
        python.parent.mkdir(parents=True)
        python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
        python.chmod(0o755)
    return root


def _audit_copy(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_") and k != "PYTHONPATH"}
    argv = [sys.executable, "-S", "-m", "tools.rules", "audit", *args]
    return subprocess.run(argv, cwd=root, env=env, capture_output=True, text=True)


_SRC = "tools/check_source_rules.py"
_NUM = "tools/check_numpy_h4.py"
_CPP = "tools/check_cpp_layers.py"
_TSR = "lucid/test/unit/test_source_rules.py"
_TCC = "lucid/test/unit/test_check_cpp_layers.py"
_QG = "tools/quality_gate/__main__.py"
_CODE = "tools/rules/registry/code.toml"
_ENG = "tools/rules/registry/engine.toml"
_META = "tools/rules/registry/meta.toml"
_TRIGGER = (
    'run = ["{py}", "tools/check_source_rules.py", "{changed}"]\ntrigger = ["lucid/**", "tools/**"]'
)
_SKIP = "@pytest.mark.skip\n"
_FULL = "    collectors = _pick(args.collectors, fast=args.fast)\n"

# (id, path, line, its replacement, the check whose controls to run, what the audit must say)
_REAL_MUTANTS = [
    (
        "C1-staged-mode-reads-nothing",
        _SRC,
        'names = _git(root, "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z")',
        'names = ""',
        "source-rules-staged",
        "H1: source-rules-staged control: stayed green",
    ),
    (
        "C3-gate-full-returns-0",
        _QG,
        _FULL,
        f"    if args.full:\n        return 0\n{_FULL}",
        "quality-gate-full",
        "quality-gate-full control: tools/test_quality_gate.py",
    ),
    (
        "C6-whole-tree-read-empty",
        _TSR,
        "return tuple(rules.check(rules.tree_sources(), rules.ref_name(), excuses=()))",
        "return ()",
        "test-source-tree",
        "test_every_excuse_is_still_needed failed",
    ),
    (
        "C4-h1-test-skipped",
        _TSR,
        "def test_h1_no_future_annotations_import",
        f"{_SKIP}def test_h1_no_future_annotations_import",
        "test-h1",
        "test_h1_no_future_annotations_import did not run",
    ),
    (
        "C5-h13-test-skipped",
        _TCC,
        "def test_real_engine_tree_passes",
        f"{_SKIP}def test_real_engine_tree_passes",
        "test-cpp-layers",
        "test_real_engine_tree_passes did not run",
    ),
    (
        "C10-h4-flags-every-import",
        _NUM,
        'return name == "numpy" or name.startswith("numpy.")',
        "return True",
        "numpy-h4",
        "H4: numpy-h4 control: not green on the near miss",
    ),
    (
        "C11-h1-flags-every-future-import",
        _SRC,
        'if any(alias.name == "annotations" for alias in node.names):',
        "if True:",
        "source-rules-staged",
        "source-rules-staged control: not green on the near miss",
    ),
    (
        "C12-h4-scan-top-level-only",
        _NUM,
        'for path in sorted(root.rglob("*.py")):',
        'for path in sorted(root.glob("*.py")):',
        "numpy-h4",
        "numpy-h4 control: stayed green",
    ),
    (
        "T1-source-rules-trigger-typo",
        _CODE,
        _TRIGGER,
        _TRIGGER.replace('["lucid/**", "tools/**"]', '["nowhere/**"]'),
        "source-rules",
        "source-rules: nowhere/** matches no file",
    ),
    (
        "E1-exemption-record",
        _ENG,
        'id = "H13"',
        'id = "H13"\nexempt = [{ match = "lucid/_C/core/Scope.h:1" }]',
        "numpy-h4",
        "unknown key",
    ),
    (
        "E3-allowlist-entry-is-a-goto",
        _CPP,
        "\n\n_COMMENTS_AND_STRINGS = re.compile(",
        '\n\n_ALLOWLIST.update({("ops/_Evil.cpp", "goto", "goto"): ("2026-11-05", "LCD-1")})\n_COMMENTS_AND_STRINGS = re.compile(',
        "numpy-h4",
        "tools.check_cpp_layers:_ALLOWLIST grew: 1 > 0",
    ),
    (
        "E5-h4-sanctioned-grows",
        _NUM,
        '"_factories/converters.py",\n',
        '"_factories/converters.py",\n        "nn/functional/__init__.py",\n',
        "numpy-h4",
        "tools.check_numpy_h4:SANCTIONED grew: 9 > 8",
    ),
    (
        "P1-pending-on-no-issue",
        _CODE,
        '{ level = "push", check = "source-rules" }',
        '{ level = "push", check = "source-rules", pending = "LCD-999999" }',
        "numpy-h4",
        "not an LCD issue",
    ),
    (
        "X1-bad-expect-regex",
        _ENG,
        r"expect = '_seed\.py:1:\d+\s+import numpy'",
        "expect = '['",
        "numpy-h4",
        "expect '['",
    ),
    (
        "Q1-ceiling-quietly-raised-pending",
        _META,
        "ceiling = 0",
        "ceiling = 1",
        "numpy-h4",
        "fell to 0",
    ),
    (
        "A1-abi-counter-unmoved-reads-as-a-bump",
        _ABI_TOOL,
        "(before is None or after > before)",
        "(before is None or after >= before)",
        "abi-bump",
        "abi-bump control: tools/test_rules.py::test_abi_bump_sees_each_way_the_interface_moves",
    ),
    (
        "A3-an-unrelated-reason-excuses-every-earlier-move",
        _ABI_TOOL,
        "held = [entry for entry in held if entry not in covered]",
        "held = []",
        "abi-bump",
        "abi-bump control: tools/test_rules.py::test_abi_bump_excuses_what_a_commit_moved_not_what_another_did",
    ),
    (
        "A4-a-reason-excuses-bump-needed",
        _ABI_TOOL,
        "if not other.needs_bump and _speaks_for(",
        "if _speaks_for(",
        "abi-bump",
        "abi-bump control: tools/test_rules.py::test_abi_bump_needed_is_excused_by_nothing_but_the_bump",
    ),
    (
        "A5-one-symbol-named-is-enough",
        _ABI_TOOL,
        "return all(len(move.subject)",
        "return any(len(move.subject)",
        "abi-bump",
        "abi-bump control: tools/test_rules.py::test_abi_bump_takes_a_later_reason_that_names_every_symbol",
    ),
    (
        "A6-a-move-undone-still-holds",
        _ABI_TOOL,
        "if move.subject in standing]",
        "if move.subject]",
        "abi-bump",
        "abi-bump control: tools/test_rules.py::test_abi_bump_forgets_a_move_that_a_later_commit_undid",
    ),
    (
        "A7-a-two-letter-word-names-a-symbol",
        _ABI_TOOL,
        "_MIN_NAME: Final = 3",
        "_MIN_NAME: Final = 1",
        "abi-bump",
        "abi-bump control: tools/test_rules.py::test_abi_bump_asks_for_a_sha_where_a_name_is_too_short_to_mean_anything",
    ),
    (
        "A8-a-six-digit-prefix-names-a-commit",
        _ABI_TOOL,
        "{7,40}",
        "{6,40}",
        "abi-bump",
        "abi-bump control: tools/test_rules.py::test_abi_bump_takes_a_later_reason_that_names_the_commit",
    ),
    (
        "A9-bump-needed-is-not-read",
        _ABI_TOOL,
        'r"\\s*bump\\s+needed\\b"',
        'r"\\s*bump\\s+neededX\\b"',
        "abi-bump",
        "abi-bump control: tools/test_rules.py::test_abi_bump_needed_is_excused_by_nothing_but_the_bump",
    ),
    (
        "A2-a-failed-attestation-passes",
        "tools/rules/attest.py",
        "if found is not None and found.passed:",
        "if found is not None:",
        "cpp-attested",
        "cpp-attested control: tools/test_rules.py::test_attest_records_a_failure_and_push",
    ),
    (
        "R1-abi-bump-range-emptied",
        _ENG,
        '"--since", "{since}"',
        '"--since", "HEAD"',
        "abi-bump",
        "abi-bump control: tools/test_rules.py::test_a_cpp_push_is_judged_through_real_git",
    ),
    (
        "T2-abi-bump-trigger-narrowed",
        _ENG,
        'trigger = ["lucid/_C/**", "tools/check_abi_bump.py"]',
        'trigger = ["lucid/_C/*.h", "tools/check_abi_bump.py"]',
        "numpy-h4",
        "E-ABI-BUMP: abi-bump's trigger leaves out",
    ),
]


@pytest.mark.parametrize(
    ("path", "line", "new", "check", "said"),
    [m[1:] for m in _REAL_MUTANTS],
    ids=[m[0] for m in _REAL_MUTANTS],
)
def test_a_real_tool_or_record_broken_is_seen(
    real_copy: Path, path: str, line: str, new: str, check: str, said: str
) -> None:
    target = real_copy / path
    original = target.read_bytes()
    assert line in original.decode(), line
    target.write_text(original.decode().replace(line, new, 1))
    try:
        done = _audit_copy(real_copy, "--check", check)
    finally:
        target.write_bytes(original)
    assert done.returncode != 0 and said in done.stdout + done.stderr, done.stdout + done.stderr
    assert "Traceback" not in done.stderr, done.stderr


def test_the_real_copy_passes_until_it_is_broken(real_copy: Path) -> None:
    done = _audit_copy(real_copy, "--check", "numpy-h4", "--check", "source-rules-staged")
    assert done.returncode == 0, done.stdout + done.stderr


def test_a_file_in_a_subpackage_counts_against_the_code_budget(real_copy: Path) -> None:
    big = real_copy / "tools/rules/_subpackage/big.py"
    big.parent.mkdir()
    big.write_text("x = 1\n" * 5000)
    try:
        done = _audit_copy(real_copy, "--check", "numpy-h4")
    finally:
        shutil.rmtree(big.parent)
    assert done.returncode == 1 and "code-lines:" in done.stdout, done.stdout


def test_each_real_seed_goes_red_and_a_tool_gone_green_would_show(tmp_path: Path) -> None:
    real = _real()
    seeds = [
        (control, real.checks[name])
        for rule in real.rules.values()
        for control in rule.control
        for name in control.check
        if isinstance(control, SeedControl)
    ]
    ws = replace(real.workspace, isolated=True)
    proofs = SeedControl.prove_all(ws, seeds)
    assert [p.detail for p in proofs if p.outcome is not Exit.PASS] == []
    by_tool = {check.tool: (control, check) for control, check in seeds}
    assert set(by_tool) == {_SRC, _NUM, _CPP}, "a new seed needs a case here"
    for number, (control, check) in enumerate(by_tool.values()):
        assert isinstance(control, SeedControl)
        root = (tmp_path / str(number)).resolve()
        _clone(root, check.tool, *control.tree)
        tool = root / check.tool
        text, patched = re.subn(r"sys\.exit\(main\(.*\)\)", "sys.exit(0)", tool.read_text())
        assert patched == 1, check.tool
        tool.write_text(text)
        proof = control.prove(replace(ws, root=root), check)
        assert proof.outcome is Exit.VIOLATION, control.expect


def test_the_real_records_say_what_the_inventory_says() -> None:
    rules = _real().rules
    verdicts = {name: rule.verdict for name, rule in rules.items()}
    enforced = dict.fromkeys(("H1", "H5", "H7", "H13", "S22"), Verdict.ENFORCED)
    partly = dict.fromkeys(("H4", "E-ABI-BUMP", "E-VERIFY-CPP"), Verdict.PARTLY)
    assert verdicts == {**enforced, **partly}
    # The C++ at push: judged where it leaves, and nowhere a worker could not satisfy it.
    for name in ("E-ABI-BUMP", "E-VERIFY-CPP"):
        assert [e.level for e in rules[name].enforce] == [Level.PUSH], name
    sanctioned, goto = rules["H4"].allowlist, rules["H13"].allowlist
    assert sanctioned is not None and sanctioned.ceiling == 8
    assert goto is not None and (goto.match, goto.ceiling) == ("goto", 0)


def test_the_real_registry_holds_its_static_claims() -> None:
    audit = Audit(_real())
    for section in (audit.wiring(), audit.allowlists(), audit.budgets()):
        assert section.findings == (), section


def test_a_test_level_outside_testpaths_is_refused(scratch: Scratch) -> None:
    check = PytestCheck(name="t", nodes=("tests/test_m.py::test_ok", "other/test_m.py::test_ok"))
    assert "other/test_m.py is outside testpaths" in str(check.unwired(Level.TEST, scratch.root))
    assert PytestCheck(name="t", nodes=("tests/t.py::a",)).unwired(Level.TEST, scratch.root) is None


def test_this_package_holds_itself_to_the_source_rules() -> None:
    """tools/ is held to H1 alone; this package took on H5, H6 and H7 as well."""
    ours = [*(REPO / "tools/rules").rglob("*.py"), REPO / "tools/test_rules.py"]
    ours += [REPO / "lucid/test/unit/test_rules_tooling.py"]
    sources = [(f"lucid/{p.name}", p.read_text()) for p in ours]
    tables = (REPO / "tools/rules").rglob("*.toml")
    sources += [(f"lucid/{p.stem}.md", p.read_text()) for p in tables]
    found = check_source_rules.check(sources, check_source_rules.ref_name())
    assert [str(v) for v in found if v.rule != "dead-path"] == []
