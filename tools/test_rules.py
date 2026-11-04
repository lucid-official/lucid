"""The registry refuses a record that enforces nothing, and the audit refuses a claim that is untrue.

Most cases build a small repository whose one tool is a script written here and
drive it through ``python -m tools.rules`` as a process: the exit convention is the
contract the hooks will read.  The rest hold the real registry to the same audit.

Not under ``lucid/test`` (the default ``testpaths``); ``lucid/test/unit/test_rules_tooling.py``
exposes it to CI.  On request::

    .venv/bin/python3 -m pytest tools/test_rules.py
"""

import functools
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path

import pytest

from tools import check_source_rules
from tools.rules.audit import Audit
from tools.rules.checks import PytestCheck, Workspace, run_nodes
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
HEAD = 'id = "R1"\nstatement = "No bad word."\nwhy = "Bad."\nscope = ["src/**"]\ndoc = "global"\n'
CHECK = '[check.x]\nrun = ["{py}", "tools/x.py", "{changed}"]\ntrigger = ["src/**"]\n\n'
COMMIT = 'enforce = [{ level = "commit", check = "x" }]\n'
SEED = (
    '[[rule.control]]\nkind = "seed"\ncheck = "x"\npath = "src/_seed.txt"\n'
    'text = "bad\\n"\nexpect = "bad-word"\n'
)
RULE = f"{CHECK}[[rule]]\n{HEAD}{COMMIT}{SEED}"
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
"""


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

    def rules(self, *args: str) -> subprocess.CompletedProcess[str]:
        argv = [sys.executable, "-m", "tools.rules", "--root", str(self.root), *args]
        return subprocess.run(argv, cwd=REPO, env=self.env, capture_output=True, text=True)

    def git(self, *args: str) -> None:
        subprocess.run(["git", *args], cwd=self.root, env=self.env, check=True, capture_output=True)

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
    made.write("pyproject.toml", '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n')
    made.write(".githooks/pre-commit", '"${PY}" tools/x.py\n')
    made.write("tools/x.py", TOOL)
    made.records(RULE)
    return made


# ── load ─────────────────────────────────────────────────────────────────────


_EXEMPT_HEAD = "\n[[rule.exempt]]\nmatch = 'a'\n"
_BAD_RECORDS = [
    (f"[[rule]]\n{HEAD}", "needs one of enforce"),
    (f"{CHECK}[[rule]]\n{HEAD}{COMMIT}", "no control proves"),
    (f"{CHECK}[[rule]]\n{HEAD}{SEED}", "controls prove"),
    (RULE.replace(COMMIT, f'{COMMIT}advisory = "because"\n'), "needs one of"),
    (f'{CHECK}[[rule]]\n{HEAD}advisory = "  "\n', "advisory is blank"),
    (f'[[rule]]\n{HEAD}pending = "soon"\n', "not an LCD issue"),
    (RULE.replace(COMMIT, f"{COMMIT}enforcement = 1\n"), "unknown key"),
    (RULE.replace('"R1"', '"r1"'), "bad rule id"),
    (RULE.replace('"No bad word."', '"""a\nb\nc"""'), "statement over two lines"),
    (RULE.replace('"Bad."', '"""a\nb"""'), "why over one line"),
    (RULE.replace('"global"', '"nowhere"'), "unknown doc"),
    (RULE.replace('["src/**"]\ndoc', "[]\ndoc"), "scope is empty"),
    (RULE.replace('level = "commit"', 'level = "lunch"'), "unknown Level"),
    (RULE.replace('level = "commit"', 'level = "test"'), "cannot run at test"),
    (RULE.replace('check = "x" }', 'check = "y" }'), "no control proves"),
    (RULE.replace('kind = "seed"', 'kind = "vibes"'), "unknown kind"),
    (RULE.replace('run = ["{py}", "tools/x.py", "{changed}"]', 'run = ["{py}"]'), "runs nothing"),
    (RULE.replace('trigger = ["src/**"]', 'trigger = "src/**"'), "must list strings"),
    (f'{CHECK}kind = "psychic"\n[[rule]]\n{HEAD}{COMMIT}{SEED}', "unknown kind"),
    (f'{RULE}\n[check.y]\nrun = ["{{py}}", "tools/x.py"]\n', "no rule claims"),
    (f"{RULE}\n[budget.b]\nunit = 'bytes'\nceiling = true\npaths = ['x']\n", "is not"),
    (f"{RULE}\n[budget.b]\nunit = 'parsecs'\nceiling = 1\npaths = ['x']\n", "unknown unit"),
    (f"wish = 'more'\n{RULE}", "unknown table"),
    (f"{RULE}\n[check]\nx = 1\n", "Cannot"),  # TOML itself: a table declared twice
    ("check = 1\n[[rule]]\n" + HEAD + 'advisory = "a"\n', "must be tables"),
    (f"{RULE}{_EXEMPT_HEAD}since = '2026-11-05'\nreason = 'r'\nissue = 'LCD-1'\n", "is not"),
    (f"{RULE}{_EXEMPT_HEAD}since = 2026-11-05\nreason = 'r'\n", "'issue' is missing"),
    (f"{RULE}{_EXEMPT_HEAD}since = 2026-11-05\nreason = ' '\nissue = 'LCD-1'\n", "its reason"),
    ("", "no rules"),
    (RULE.replace("[check.x]\n", '[check.x]\nname = "y"\n'), "not its to set"),
]


@pytest.mark.parametrize(("text", "problem"), _BAD_RECORDS, ids=[p for _, p in _BAD_RECORDS])
def test_a_bad_record_fails_load(scratch: Scratch, text: str, problem: str) -> None:
    scratch.records(text)
    with pytest.raises(RuleError, match=problem):
        Registry.load(scratch.root)


def test_a_record_with_no_form_exits_1_and_names_the_rule(scratch: Scratch) -> None:
    scratch.records(f"[[rule]]\n{HEAD}")
    done = scratch.rules("audit")
    assert done.returncode == 1
    assert "R1: needs one of enforce + control, advisory, pending" in done.stderr


def test_ids_are_unique_across_files_and_the_private_layer_merges(scratch: Scratch) -> None:
    advisory = f'[[rule]]\n{HEAD.replace("R1", "R2")}advisory = "judgement"\n'
    scratch.write(".claude/rules.local.toml", advisory)
    assert sorted(scratch.registry.rules) == ["R1", "R2"]
    scratch.records(advisory, "y")
    with pytest.raises(RuleError, match="R2 is declared twice"):
        Registry.load(scratch.root)


def test_queries_by_level_and_path(scratch: Scratch) -> None:
    registry = scratch.registry
    assert list(registry.at_level(Level.COMMIT)) == ["x"]
    assert registry.at_level(Level.PUSH) == {}
    assert [rule.id for rule in registry.for_path("src/a.txt")] == ["R1"]
    assert registry.for_path("web/a.txt") == []


# ── run: the exit convention ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("body", "code", "word"),
    [
        ("raise SystemExit(0)", 0, "PASS"),
        ("raise SystemExit(1)", 1, "VIOLATION"),
        ("raise SystemExit(2)", 2, "UNMEASURED"),
        ("raise SystemExit(3)", 2, "UNMEASURED"),
        ("raise ValueError('a bug in the tool, not a finding')", 2, "UNMEASURED"),
        ("import os, signal; os.kill(os.getpid(), signal.SIGKILL)", 2, "UNMEASURED"),
    ],
    ids=["exit-0", "exit-1", "exit-2", "exit-3", "traceback", "killed"],
)
def test_0_passes_1_is_a_violation_anything_else_cannot_measure(
    scratch: Scratch, body: str, code: int, word: str
) -> None:
    scratch.write("tools/x.py", body)
    done = scratch.rules("run", "--level", "commit")
    assert done.returncode == code, done.stdout + done.stderr
    assert re.search(rf"^  {word}\s+x\s+\[R1\]", done.stdout, re.MULTILINE), done.stdout


def test_a_missing_tool_cannot_measure(scratch: Scratch) -> None:
    (scratch.root / "tools/x.py").unlink()
    assert scratch.rules("run", "--level", "commit").returncode == 2


def test_the_worst_exit_is_returned(scratch: Scratch) -> None:
    scratch.write("tools/x.py", "raise SystemExit(1)")
    scratch.write("tools/y.py", "raise SystemExit(2)")
    second = RULE.replace("[check.x]", "[check.y]").replace("tools/x.py", "tools/y.py")
    second = second.replace('"x"', '"y"').replace('"R1"', '"R2"')
    scratch.records(second, "y")
    assert scratch.rules("run", "--level", "commit").returncode == 2


def test_a_level_nothing_claims_is_not_a_pass(scratch: Scratch) -> None:
    assert scratch.rules("run", "--level", "ci").returncode == 2


def test_triggers_and_changed_files_come_from_the_range(scratch: Scratch) -> None:
    scratch.write("tools/x.py", 'import sys\nprint(sys.argv[1:], file=open("args", "w"))\n')
    scratch.write("src/kept.txt", "a\n")
    scratch.write("src/gone.txt", "a\n")
    scratch.write("web/other.txt", "a\n")
    scratch.git("init", "-q", "-b", "main")
    scratch.git("config", "user.name", "t")
    scratch.git("config", "user.email", "t@example.invalid")
    scratch.git("add", "-A")
    scratch.git("commit", "-qm", "base")

    scratch.write("web/other.txt", "b\n")
    scratch.git("commit", "-qam", "outside the trigger")
    done = scratch.rules("run", "--level", "commit", "--range", "HEAD~1..HEAD")
    assert done.returncode == 0 and "no changed file matches its trigger" in done.stdout

    (scratch.root / "src/gone.txt").unlink()
    scratch.git("commit", "-qam", "only a deletion")
    done = scratch.rules("run", "--level", "commit", "--range", "HEAD~1..HEAD")
    assert done.returncode == 0 and "its changed files were deleted" in done.stdout

    scratch.write("src/kept.txt", "b\n")
    scratch.write("web/other.txt", "c\n")
    scratch.git("commit", "-qam", "both")
    assert scratch.rules("run", "--level", "commit", "--range", "HEAD~1..HEAD").returncode == 0
    assert (scratch.root / "args").read_text().strip() == "['src/kept.txt']"

    scratch.write("src/kept.txt", "c\n")
    scratch.git("add", "src/kept.txt")
    assert "1 changed file(s)" in scratch.rules("run", "--level", "commit", "--staged").stdout

    for bad in ("nope..HEAD", "HEAD"):
        assert scratch.rules("run", "--level", "commit", "--range", bad).returncode == 2


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
    }
    nodes = {f"tests/test_m.py::{name}": outcome for name, outcome in want.items()}
    assert dict(run_nodes(scratch.ws, list(nodes)).outcomes) == nodes
    gone = run_nodes(scratch.ws, ["tests/test_m.py::test_ok", "tests/test_m.py::test_gone"])
    assert set(gone.outcomes.values()) == {Exit.UNMEASURED}, gone.output  # pytest refuses both
    nothing = run_nodes(scratch.ws, ["tests/test_absent.py::test_ok"])
    assert set(nothing.outcomes.values()) == {Exit.UNMEASURED}


# ── controls ─────────────────────────────────────────────────────────────────


CRASHES_SEEDED = """\
import sys
if open(sys.argv[1]).read():
    raise ValueError("a crash is no finding")
"""
OFF_MESSAGE = """\
import sys
if open(sys.argv[1]).read():
    print("some other complaint")
    raise SystemExit(1)
"""


@pytest.mark.parametrize(
    ("tool", "code", "said"),
    [
        (TOOL, 0, "1/1 red when seeded"),
        ("raise SystemExit(0)", 1, "R1: x control: stayed green with src/_seed.txt seeded"),
        ("raise SystemExit(1)", 2, "R1: x control: not green on a clean tree"),
        (OFF_MESSAGE, 1, "went red without /bad-word/"),
        (CRASHES_SEEDED, 2, "could not run seeded"),
    ],
    ids=["proves", "green-forever", "red-forever", "off-message", "crashes-seeded"],
)
def test_a_seed_control_proves_clean_then_red(
    scratch: Scratch, tool: str, code: int, said: str
) -> None:
    scratch.write("tools/x.py", tool)
    done = scratch.rules("audit")
    assert done.returncode == code, done.stdout + done.stderr
    assert said in done.stdout


def test_a_proof_keeps_its_verdict_when_the_tool_says_a_lot(scratch: Scratch) -> None:
    scratch.write("tools/x.py", "print('x' * 100_000)\nraise SystemExit(1)")
    done = scratch.rules("audit")
    assert done.returncode == 2 and "not green on a clean tree" in done.stdout
    assert len(done.stdout) < 2_000


def test_a_test_file_that_does_not_parse_is_no_wiring(scratch: Scratch) -> None:
    scratch.write("tests/test_m.py", "def (:\n")
    problem = PytestCheck(name="t", nodes=("tests/test_m.py::test_ok",)).unwired(
        Level.TEST, scratch.root
    )
    assert problem is not None and "does not parse" in problem


def test_a_seed_control_with_no_tool_could_not_run(scratch: Scratch) -> None:
    (scratch.root / "tools/x.py").unlink()
    done = scratch.rules("audit")
    assert done.returncode == 2 and "could not copy tools/x.py" in done.stdout


def test_a_control_does_not_see_the_committing_tree_through_git_variables(
    scratch: Scratch, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = "import os\nraise SystemExit(1 if 'GIT_DIR' in os.environ else 0)\n"
    scratch.write("tools/x.py", probe)
    monkeypatch.setenv("GIT_DIR", "/nowhere")
    assert scratch.ws.spawn([sys.executable, "tools/x.py"]).returncode == 0


def _pytest_rule(*nodes: str) -> str:
    listed = ", ".join(f'"tests/test_m.py::{node}"' for node in nodes)
    return (
        f'[check.t]\nkind = "pytest"\nnodes = ["tests/test_m.py::test_ok"]\n\n[[rule]]\n{HEAD}'
        f'enforce = [{{ level = "test", check = "t" }}]\n\n'
        f'[[rule.control]]\nkind = "pytest"\ncheck = "t"\nnodes = [{listed}]\n'
    )


@pytest.mark.parametrize(
    ("nodes", "code", "said"),
    [
        (("test_ok", "test_many"), 0, "1/1 red when seeded"),
        (("test_ok", "test_bad"), 1, "R1: t control: tests/test_m.py::test_bad failed"),
        (("test_ok", "test_skipped"), 2, "tests/test_m.py::test_skipped did not run"),
        (("test_ok", "test_gone"), 2, "did not run"),
    ],
)
def test_a_pytest_control_passes_only_on_tests_that_ran(
    scratch: Scratch, nodes: tuple[str, ...], code: int, said: str
) -> None:
    scratch.write("tests/test_m.py", TESTS)
    scratch.records(_pytest_rule(*nodes))
    done = scratch.rules("audit")
    assert done.returncode == code, done.stdout + done.stderr
    assert said in done.stdout


def test_pytest_controls_share_one_process_and_name_their_own_rule(scratch: Scratch) -> None:
    scratch.write(
        "tests/test_m.py", TESTS + "import os\nprint(os.getpid(), file=open('pid', 'a'))\n"
    )
    scratch.records(_pytest_rule("test_ok"))
    other = _pytest_rule("test_bad").replace('"R1"', '"R2"').replace("[check.t]", "[check.u]")
    scratch.records(other.replace('"t"', '"u"'), "y")
    done = scratch.rules("audit")
    assert done.returncode == 1 and "R2: u control:" in done.stdout and "R1:" not in done.stdout
    assert len((scratch.root / "pid").read_text().split()) == 1


HOOK = """\
import json, sys
call = json.load(sys.stdin)
if "--no-verify" in call["tool_input"]["command"]:
    decision = "deny" if "agent_id" in call else "ask"
    print(json.dumps({"hookSpecificOutput": {"permissionDecision": decision}}))
"""


def _hook_rule(body: str) -> str:
    return (
        f'[check.h]\nkind = "hook"\nrun = ["{{py}}", "tools/h.py"]\n\n[[rule]]\n{HEAD}'
        f'enforce = [{{ level = "hook", check = "h" }}]\n\n'
        f'[[rule.control]]\nkind = "hook"\ncheck = "h"\ntool = "Bash"\n{body}'
    )


@pytest.mark.parametrize(
    ("hook", "body", "code", "said"),
    [
        (HOOK, 'input = "git commit --no-verify"\nexpect = "deny"\nmain = "ask"\n', 0, "1/1"),
        (
            HOOK,
            'input = "git commit --no-verify"\nexpect = "deny"\n',
            1,
            "answered main session ask",
        ),
        (HOOK, 'input = "git commit"\nexpect = "deny"\n', 1, "answered subagent allow, not deny"),
        ("raise SystemExit(1)", 'input = "x"\nexpect = "deny"\n', 2, "gave subagent no answer"),
        ("print('not json')", 'input = "x"\nexpect = "deny"\n', 2, "gave subagent no answer"),
    ],
    ids=["both", "main-differs", "subagent-allowed", "hook-fails", "hook-babbles"],
)
def test_a_hook_control_asks_as_subagent_and_as_main(
    scratch: Scratch, hook: str, body: str, code: int, said: str
) -> None:
    scratch.write("tools/h.py", hook)
    scratch.write(".claude/settings.json", '{"command": "python3 tools/h.py"}')
    scratch.records(_hook_rule(body))
    done = scratch.rules("audit")
    assert done.returncode == code, done.stdout + done.stderr
    assert said in done.stdout


# ── wiring: a claimed level really runs the check ────────────────────────────


@pytest.mark.parametrize(
    "hook",
    [
        "# tools/x.py is what this hook would run, if it ran it\nexit 0\n",
        "exit 0  # tools/x.py\n",
        "exit 0\n",
    ],
    ids=["comment-line", "trailing-comment", "absent"],
)
def test_a_commit_claim_on_a_tool_the_hook_does_not_run_fails(scratch: Scratch, hook: str) -> None:
    scratch.write(".githooks/pre-commit", hook)
    done = scratch.rules("audit")
    assert done.returncode == 1
    assert "R1: level=commit check=x: .githooks/pre-commit does not run tools/x.py" in done.stdout


def test_a_config_file_that_never_runs_wires_nothing(scratch: Scratch) -> None:
    scratch.write(".githooks/pre-commit", "exit 0\n")
    scratch.write(".pre-commit-config.yaml", "entry: python tools/x.py\n")
    assert scratch.rules("audit").returncode == 1


def test_the_rules_runner_wires_every_check_at_its_level(scratch: Scratch) -> None:
    scratch.write(".githooks/pre-commit", '"${PY}" -m tools.rules run --level commit --staged\n')
    assert scratch.rules("audit").returncode == 0


@pytest.mark.parametrize(
    ("file", "text", "code"),
    [
        ("scripts/ci_full.sh", 'stage "x" "$PY" tools/x.py\n', 0),
        (".github/workflows/x.yml", "run: python tools/x.py\n", 0),
        (".github/workflows/x.yml", "run: python tools/y.py  # tools/x.py\n", 1),
        ("scripts/other.sh", "python tools/x.py\n", 1),
    ],
)
def test_a_ci_claim_needs_the_script_or_a_workflow(
    scratch: Scratch, file: str, text: str, code: int
) -> None:
    scratch.records(RULE.replace('level = "commit"', 'level = "ci"'))
    scratch.write(file, text)
    assert scratch.rules("audit").returncode == code


def test_a_pending_level_is_counted_and_goes_stale_once_wired(scratch: Scratch) -> None:
    claim = '{ level = "push", check = "x", pending = "LCD-841-R2" }, '
    scratch.records(RULE.replace("enforce = [", f"enforce = [{claim}"))
    done = scratch.rules("audit")
    assert done.returncode == 0 and "1 pending on LCD-841-R2" in done.stdout
    assert scratch.registry.rules["R1"].verdict is Verdict.ENFORCED
    scratch.write(".githooks/pre-push", '"${PY}" -m tools.rules push\n')
    done = scratch.rules("audit")
    assert done.returncode == 1 and "is wired now: drop pending" in done.stdout


def test_a_push_claim_with_no_pre_push_fails(scratch: Scratch) -> None:
    scratch.records(RULE.replace('level = "commit"', 'level = "push"'))
    done = scratch.rules("audit")
    assert done.returncode == 1 and ".githooks/pre-push does not run" in done.stdout


def test_a_rule_pending_on_every_level_is_not_enforced(scratch: Scratch) -> None:
    claim = '{ level = "commit", check = "x", pending = "LCD-1" }'
    scratch.records(RULE.replace('{ level = "commit", check = "x" }', claim))
    assert scratch.registry.rules["R1"].verdict is Verdict.PENDING


@pytest.mark.parametrize(
    ("node", "said"),
    [
        ("tests/test_m.py::test_ok", None),
        ("tests/test_m.py::test_many[1]", None),
        ("tests/test_m.py::test_gone", "defines no test_gone"),
        ("tests/test_absent.py::test_ok", "does not exist"),
        ("other/test_m.py::test_ok", "is outside testpaths"),
    ],
)
def test_a_test_claim_names_a_test_something_collects(
    scratch: Scratch, node: str, said: str | None
) -> None:
    scratch.write("tests/test_m.py", TESTS)
    scratch.write("other/test_m.py", TESTS)
    problem = PytestCheck(name="t", nodes=(node,)).unwired(Level.TEST, scratch.root)
    assert (problem is None) == (said is None) and (said is None or said in str(problem))


# ── ceilings ─────────────────────────────────────────────────────────────────


def _budget(unit: str, ceiling: int, extra: str = 'paths = ["big.txt"]\n') -> str:
    return f'{RULE}\n[budget.b]\nunit = "{unit}"\nceiling = {ceiling}\n{extra}'


@pytest.mark.parametrize(
    ("record", "code", "said"),
    [
        (_budget("bytes", 9), 0, "b 9/9"),
        (_budget("bytes", 8), 1, "b: 9 bytes is over its ceiling of 8"),
        (_budget("lines", 2), 0, "b 2/2"),  # the comment and the blank line are free
        (_budget("lines", 1), 1, "b: 2 lines is over"),
        (_budget("bytes", 7, 'paths = ["no.txt"]\n'), 2, "which match nothing"),
        (_budget("bytes", 7, 'paths = ["no.txt"]\nabsent_ok = true\n'), 0, "b absent/7"),
        (_budget("pending-rules", 0, ""), 0, "b 0/0"),
    ],
    ids=["at", "over", "lines-at", "lines-over", "no-match", "absent-ok", "pending"],
)
def test_a_budget_is_a_ceiling_and_measures_something(
    scratch: Scratch, record: str, code: int, said: str
) -> None:
    scratch.write("big.txt", "a\n# b\n\nc\n")
    scratch.records(record)
    done = scratch.rules("audit")
    assert done.returncode == code, done.stdout + done.stderr
    assert said in done.stdout


def test_pending_rules_are_held_to_their_ceiling(scratch: Scratch) -> None:
    pending = f'[[rule]]\n{HEAD.replace("R1", "R2")}pending = "LCD-1"\n'
    scratch.records(pending, "y")
    scratch.records(_budget("pending-rules", 0, ""))
    done = scratch.rules("audit")
    assert done.returncode == 1 and "b: 1 pending-rules is over" in done.stdout


@pytest.mark.parametrize(
    ("ceiling", "code", "said"),
    [(2, 0, "1/1 allowlists"), (1, 1, "grew: 2 > 1"), (3, 1, "shrank to 2: lower ceiling 3")],
)
def test_an_allowlist_ceiling_only_falls_in_step_with_the_list(
    scratch: Scratch, ceiling: int, code: int, said: str
) -> None:
    scratch.write("tools/x.py", f"{TOOL}\nT: dict[str, str] = {{'a': 'b', 'c': 'd'}}\n")
    line = f'allowlist = {{ source = "tools.x:T", ceiling = {ceiling} }}\n'
    scratch.records(RULE.replace(COMMIT, COMMIT + line))
    done = scratch.rules("audit")
    assert done.returncode == code, done.stdout + done.stderr
    assert said in done.stdout


def test_an_allowlist_that_cannot_be_read_could_not_be_measured(scratch: Scratch) -> None:
    line = 'allowlist = { source = "tools.x:NOPE", ceiling = 1 }\n'
    scratch.records(RULE.replace(COMMIT, COMMIT + line))
    done = scratch.rules("audit")
    assert done.returncode == 2 and "could not be read" in done.stdout


EXEMPT = """
[[rule.exempt]]
match = "tools/x.py:3"
since = 2026-11-05
reason = "patches another library's own attribute name"
issue = "LCD-1"
"""


def test_an_exemption_goes_stale_with_its_file_and_expires_on_its_date(scratch: Scratch) -> None:
    scratch.records(f"{RULE}{EXEMPT}expires = 2026-12-01\n")
    before, after = date(2026, 12, 1), date(2026, 12, 2)
    assert Audit(scratch.registry, today=before).exemptions().findings == ()
    assert (
        "expired 2026-12-01"
        in Audit(scratch.registry, today=after).exemptions().findings[0].message
    )
    scratch.records(f"{RULE}{EXEMPT}".replace("tools/x.py:3", "tools/gone.py:3"))
    done = scratch.rules("audit")
    assert done.returncode == 1 and "is stale: tools/gone.py is gone" in done.stdout


# ── the command line ─────────────────────────────────────────────────────────


def test_list_show_and_their_refusals(scratch: Scratch) -> None:
    advisory = f'[[rule]]\n{HEAD.replace("R1", "R2")}advisory = "judgement"\n'
    scratch.records(advisory, "y")
    assert "R1" in scratch.rules("list").stdout
    assert "web" not in scratch.rules("list", "--path", "src/a.txt").stdout
    assert (
        "1 enforced, 0 partly, 1 advisory, 0 pending of 2"
        in scratch.rules("list", "--verdict").stdout
    )
    assert "why: Bad." in scratch.rules("show", "R1").stdout
    assert scratch.rules("show", "R9").returncode == 2
    assert scratch.rules("audit", "--rule", "R9").returncode == 2


# ── the real registry ────────────────────────────────────────────────────────


@functools.cache
def _real() -> Registry:
    return Registry.load(REPO)


def _clone(root: Path, *relatives: str) -> None:
    for relative in relatives:
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["cp", "-cR", str(REPO / relative), str(root / relative)], check=True)


def test_each_real_seed_control_goes_red_and_would_show_a_tool_gone_green(tmp_path: Path) -> None:
    real = _real()
    seeds = [c for r in real.rules.values() for c in r.control if isinstance(c, SeedControl)]
    assert {c.check for c in seeds} == {"source-rules", "numpy-h4", "cpp-layers"}
    for number, control in enumerate(seeds):
        check = real.checks[control.check]
        ws = replace(real.workspace, isolated=True)
        assert control.prove(ws, check).outcome is Exit.PASS, control.expect
        root = (tmp_path / str(number)).resolve()
        _clone(root, check.tool, *control.tree)
        tool = root / check.tool
        text, patched = re.subn(r"sys\.exit\(main\(.*\)\)", "sys.exit(0)", tool.read_text())
        assert patched == 1, check.tool
        tool.write_text(text)
        assert control.prove(replace(ws, root=root), check).outcome is Exit.VIOLATION, (
            control.expect
        )


def _real_tree(tmp_path: Path) -> Path:
    """Enough of this repository for the real records' audit, with the H4 control alone."""
    root = tmp_path.resolve()
    _clone(root, "tools", ".githooks", "scripts", ".github", "pyproject.toml", "lucid/test/unit")
    return root


def _audit_h4(root: Path) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    argv = [sys.executable, "-m", "tools.rules", "--root", str(root), "audit", "--rule", "H4"]
    return subprocess.run(argv, cwd=REPO, env=env, capture_output=True, text=True)


def test_a_real_tool_turned_green_exits_1_naming_its_rule(tmp_path: Path) -> None:
    real_tree = _real_tree(tmp_path)
    fine = _audit_h4(real_tree)
    assert fine.returncode == 0, fine.stdout + fine.stderr
    tool = real_tree / "tools/check_numpy_h4.py"
    tool.write_text(tool.read_text().replace("sys.exit(main(sys.argv[1:]))", "sys.exit(0)"))
    broken = _audit_h4(real_tree)
    assert broken.returncode == 1, broken.stdout + broken.stderr
    assert "H4: numpy-h4 control: stayed green with lucid/_seed.py seeded" in broken.stdout


def test_a_real_claim_the_hook_does_not_honour_exits_1_naming_its_rule(tmp_path: Path) -> None:
    """H13's commit level is pending because no hook runs it; say otherwise and the audit refuses."""
    real_tree = _real_tree(tmp_path)
    records = real_tree / "tools/rules/registry/engine.toml"
    claim = '{ level = "commit", check = "cpp-layers", pending = "LCD-841-R2" }'
    assert claim in records.read_text()
    records.write_text(
        records.read_text().replace(claim, '{ level = "commit", check = "cpp-layers" }')
    )
    wrong = _audit_h4(real_tree)
    assert wrong.returncode == 1, wrong.stdout + wrong.stderr
    expected = "H13: level=commit check=cpp-layers: .githooks/pre-commit does not run tools/check_cpp_layers.py"
    assert expected in wrong.stdout


def test_the_real_records_say_what_the_inventory_says() -> None:
    real = _real()
    verdicts = {rule.id: rule.verdict for rule in real.rules.values()}
    enforced = {name: Verdict.ENFORCED for name in ("H1", "H5", "H7", "H13", "S22")}
    assert verdicts == {**enforced, "H4": Verdict.PARTLY}
    allowlist = real.rules["H13"].allowlist
    assert allowlist is not None and allowlist.ceiling == 23


def test_the_real_registry_holds_its_static_claims() -> None:
    audit = Audit(_real())
    for section in (audit.wiring(), audit.exemptions(), audit.budgets()):
        assert section.findings == (), section


def test_this_package_holds_itself_to_the_source_rules() -> None:
    """tools/ is held to H1 alone; this package took on H5, H6 and H7 as well."""
    ours = [*(REPO / "tools/rules").rglob("*.py"), REPO / "tools/test_rules.py"]
    ours += [REPO / "lucid/test/unit/test_rules_tooling.py"]
    sources = [(f"lucid/{p.name}", p.read_text()) for p in ours]
    sources += [
        (f"lucid/{p.stem}.md", p.read_text()) for p in (REPO / "tools/rules").rglob("*.toml")
    ]
    found = check_source_rules.check(sources, check_source_rules.ref_name())
    assert [str(v) for v in found if v.rule != "dead-path"] == []
