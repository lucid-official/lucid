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

import functools
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest

from tools import check_source_rules
from tools.rules.__main__ import main
from tools.rules.audit import Audit
from tools.rules.checks import CommandCheck, PytestCheck, Workspace, run_nodes
from tools.rules.controls import SeedControl
from tools.rules.model import Exit, Level, RuleError, Verdict
from tools.rules.registry import Registry

REPO = Path(__file__).resolve().parents[1]

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


@dataclass
class Scratch:
    root: Path
    env: dict[str, str] = field(repr=False)

    def write(self, relative: str, text: str) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def records(self, text: str, name: str = "x") -> None:
        self.write(f"tools/rules/registry/{name}.toml", text)

    def rules(self, *args: str, home: Path = REPO) -> subprocess.CompletedProcess[str]:
        argv = [sys.executable, "-S", "-m", "tools.rules", "--root", str(self.root), *args]
        return subprocess.run(argv, cwd=home, env=self.env, capture_output=True, text=True)

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
    made.write(".githooks/pre-commit", '"${PY}" tools/x.py\n')
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


def _hook(text: str) -> Callable[[Scratch], None]:
    return lambda s: s.write(".githooks/pre-commit", text)


def _no_tool(s: Scratch) -> None:
    (s.root / "tools/x.py").unlink()


def _index_rule(body: str = BLOB_TOOL) -> Callable[[Scratch], None]:
    def setup(s: Scratch) -> None:
        s.write("tools/y.py", body)
        s.write(".githooks/pre-commit", '"${PY}" tools/x.py\n"${PY}" tools/y.py --staged\n')
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
            s.write(".githooks/pre-push", line)

    return setup


def _ci(file: str, text: str) -> Callable[[Scratch], None]:
    def setup(s: Scratch) -> None:
        s.records(RULE.replace('level = "commit"', 'level = "ci"'))
        s.write(file, text)

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
        lambda s: (
            (
                s.write("other/a.txt", "ok\n"),
                s.records(RULE.replace('trigger = ["src/**"]', 'trigger = ["other/**"]')),
            )
            and None
        ),
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
    # wiring: a level names the tool
    "named": Scenario(_nothing, AUDIT, 0, "1/1 claimed levels name their tool"),
    "named-only-in-a-comment": Scenario(
        _hook("# tools/x.py is what this hook would run\nexit 0\n"),
        AUDIT,
        1,
        "R1: level=commit check=x: .githooks/pre-commit does not name tools/x.py",
    ),
    "named-in-a-trailing-comment": Scenario(
        _hook("exit 0  # tools/x.py\n"), AUDIT, 1, "does not name"
    ),
    "named-after-a-semicolon": Scenario(_hook("true;#tools/x.py\n"), AUDIT, 1, "does not name"),
    "named-with-a-suffix": Scenario(_hook("X=tools/x.py.disabled\n"), AUDIT, 1, "does not name"),
    "named-with-a-prefix": Scenario(_hook("X=retools/x.py\n"), AUDIT, 1, "does not name"),
    "named-by-the-runner": Scenario(
        _hook('"${PY}" -m tools.rules run --level commit --staged\n'), AUDIT, 0, "1/1 claimed"
    ),
    "named-by-the-wrong-runner": Scenario(
        _hook('"${PY}" -m tools.rules run --level push\n'), AUDIT, 1, "does not name"
    ),
    "ci-script": Scenario(_ci("scripts/ci_full.sh", '"$PY" tools/x.py\n'), AUDIT, 0, "1/1 claimed"),
    "ci-workflow": Scenario(
        _ci(".github/workflows/x.yml", "run: python tools/x.py\n"), AUDIT, 0, "1/1 claimed"
    ),
    "ci-other-script": Scenario(
        _ci("scripts/o.sh", "python tools/x.py\n"), AUDIT, 1, "does not name"
    ),
    "pending-counted": Scenario(_pending(), AUDIT, 0, "1 pending on LCD-841-R2"),
    "pending-gone-stale": Scenario(
        _pending('"${PY}" -m tools.rules push\n'), AUDIT, 1, "is named now: drop pending"
    ),
    "push-with-no-hook": Scenario(
        _record(RULE.replace('level = "commit"', 'level = "push"')),
        AUDIT,
        1,
        ".githooks/pre-push does not name",
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
        "named-only-in-a-comment",
    ),
    ("checks.py", r"(?:^|[\s;])#", r"(?:^|\s)#", "named-after-a-semicolon"),
    ("checks.py", r"(?![\w./-])", "", "named-with-a-suffix"),
    ("checks.py", r"(?<![\w.-])", "", "named-with-a-prefix"),
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
    scratch.write(".githooks/pre-commit", '"${PY}" -m tools.rules list\n')
    assert generate.unwired(Level.COMMIT, scratch.root) is not None
    scratch.write(".githooks/pre-commit", '"${PY}" -m tools.rules generate --check\n')
    assert generate.unwired(Level.COMMIT, scratch.root) is None
    assert CommandCheck(name="q", run=("{py}", "-m", "tools.x", "--fast")).tool == "tools.x"
    assert CommandCheck(name="s", run=("{py}", "tools/x.py", "{changed}")).tool == "tools/x.py"


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
    finder = REPO / ".venv/lib/python3.14/site-packages/_lucid_worktree_finder.py"
    if finder.exists():  # a worktree's light venv pins lucid and tools to the worktree
        _clone(root, ".venv")
        copy = root / finder.relative_to(REPO)
        copy.write_text(copy.read_text().replace(str(REPO), str(root)))
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
        'pending = "LCD-841-R2"',
        'pending = "LCD-999999"',
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
        "ceiling = 7",
        "ceiling = 8",
        "numpy-h4",
        "fell to 7",
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
    assert set(by_tool) == {_SRC, _NUM, _CPP}
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
    verdicts = {name: rules[name].verdict for name in ("H1", "H4", "H5", "H7", "H13", "S22")}
    enforced = dict.fromkeys(("H1", "H5", "H7", "H13", "S22"), Verdict.ENFORCED)
    assert verdicts == {**enforced, "H4": Verdict.PARTLY}
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
