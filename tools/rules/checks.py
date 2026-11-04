"""A check is a subprocess with one reading of its exit: 0 pass, 1 violation, 2 cannot measure."""

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import tomllib
import xml.etree.ElementTree as ElementTree
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import ClassVar, Final, Self, override

from tools.rules.model import Exit, Level, need

WORKERS: Final = 4
_TIMEOUT: Final = 600
# Where a claimed level has to name a check.  ``.pre-commit-config.yaml`` is not
# here: it claimed mypy and ruff at commit for years and never ran.
_WIRED_IN: Final[Mapping[Level, tuple[str, ...]]] = {
    Level.COMMIT: (".githooks/pre-commit",),
    Level.PUSH: (".githooks/pre-push",),
    Level.CI: ("scripts/ci_full.sh", ".github/workflows/*.yml"),
    Level.HOOK: (".claude/settings.json",),
}
_RUNNERS: Final[Mapping[Level, tuple[str, ...]]] = {
    Level.COMMIT: ("tools.rules run --level commit",),
    Level.PUSH: ("tools.rules run --level push", "tools.rules push"),
    Level.CI: ("tools.rules run --level ci",),
}
_COMMENT: Final = re.compile(r"(?:^|\s)#.*$", re.MULTILINE)


class CannotMeasure(RuntimeError):
    """The question could not be put: an unreadable range, for one."""


@dataclass(frozen=True)
class Workspace:
    root: Path
    python: str
    isolated: bool = False

    @classmethod
    def of(cls, root: Path) -> Self:
        venv = root / ".venv/bin/python3"
        return cls(root, str(venv) if venv.exists() else sys.executable)

    def spawn(
        self, argv: Sequence[str], stdin: str | None = None
    ) -> subprocess.CompletedProcess[str]:
        # A hook exports GIT_DIR and GIT_INDEX_FILE, which in a control would
        # send git in a scratch tree to the real repository.
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        try:
            return subprocess.run(
                argv,
                cwd=self.root,
                env=env if self.isolated else None,
                input=stdin,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=_TIMEOUT,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return subprocess.CompletedProcess(argv, Exit.UNMEASURED, "", f"{exc!r}")


@dataclass(frozen=True)
class Result:
    check: str
    outcome: Exit
    output: str = ""
    skipped: str = ""
    seconds: float = 0.0


@dataclass(frozen=True, kw_only=True)
class Check(ABC):
    name: str
    run: tuple[str, ...] = ()
    trigger: tuple[str, ...] = ()

    kinds: ClassVar[dict[str, type[Check]]] = {}
    levels: ClassVar[frozenset[Level]] = frozenset()

    @override
    def __init_subclass__(cls, *, kind: str, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        Check.kinds[kind] = cls

    def __post_init__(self) -> None:
        need(bool(self.tool), f"{self.name} runs nothing")

    @property
    def tool(self) -> str:
        """What a hook or a workflow names when it runs this check, and what a seed copies."""
        return next((w for w in self.run if w != "{py}" and not w.startswith("-")), "")

    def argv(self, ws: Workspace, changed: Sequence[str] = ()) -> list[str]:
        words = (changed if w == "{changed}" else [w.replace("{py}", ws.python)] for w in self.run)
        return [word for group in words for word in group]

    def relevant(self, changed: Sequence[str]) -> list[str]:
        """The changed paths this check is about; none means it has nothing to say."""
        if not self.trigger:
            return list(changed)
        return [p for p in changed if any(PurePosixPath(p).full_match(g) for g in self.trigger)]

    def unwired(self, level: Level, root: Path) -> str | None:
        """Why ``level`` does not run this check, or None when it does."""
        places = _WIRED_IN[level]
        files = [file for place in places for file in root.glob(place) if file.is_file()]
        text = "\n".join(_COMMENT.sub("", f.read_text(errors="replace")) for f in files)
        if self.tool in text or any(runner in text for runner in _RUNNERS.get(level, ())):
            return None
        return f"{' or '.join(places)} does not run {self.tool}"

    @abstractmethod
    def execute(self, ws: Workspace, changed: Sequence[str] = ()) -> Result:
        """Run it on ``ws``; ``changed`` fills ``{changed}``, and empty means the tool's default."""


@dataclass(frozen=True, kw_only=True)
class CommandCheck(Check, kind="command"):
    levels = frozenset({Level.COMMIT, Level.PUSH, Level.CI})

    @override
    def execute(self, ws: Workspace, changed: Sequence[str] = ()) -> Result:
        done = ws.spawn(self.argv(ws, changed))
        # An uncaught exception exits 1 as well, and it is no finding about the code.
        crashed = done.returncode == 1 and "Traceback (most recent call last)" in done.stderr
        good = done.returncode in (0, 1) and not crashed
        return Result(
            self.name, Exit(done.returncode) if good else Exit.UNMEASURED, done.stdout + done.stderr
        )


@dataclass(frozen=True)
class NodeRun:
    outcomes: Mapping[str, Exit]
    output: str


def run_nodes(ws: Workspace, nodes: Sequence[str]) -> NodeRun:
    """One process for all of ``nodes``.  A node that did not run proved nothing: 2, not 0."""
    with tempfile.TemporaryDirectory(prefix="lucid-rules-") as tmp:
        report = Path(tmp) / "junit.xml"
        pytest = ("-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider")
        done = ws.spawn([ws.python, *pytest, f"--junitxml={report}", *nodes])
        cases = {} if not report.is_file() else dict(_testcases(report))
    return NodeRun({n: _node_outcome(n, cases) for n in nodes}, done.stdout + done.stderr)


def _testcases(report: Path) -> Sequence[tuple[str, Exit]]:
    found = []
    for case in ElementTree.parse(report).iter("testcase"):
        children = {child.tag for child in case}
        outcome = Exit.VIOLATION if "failure" in children else Exit.PASS
        outcome = Exit.UNMEASURED if children & {"error", "skipped"} else outcome
        found.append((f"{case.get('classname')}.{case.get('name')}", outcome))
    return found


def _node_outcome(node: str, cases: Mapping[str, Exit]) -> Exit:
    path, _, tail = node.partition(".py::")
    want = f"{path.replace('/', '.')}.{tail.replace('::', '.')}"
    found = [v for k, v in cases.items() if k == want or k.startswith(f"{want}[")]
    return max(found, default=Exit.UNMEASURED)


@dataclass(frozen=True, kw_only=True)
class PytestCheck(Check, kind="pytest"):
    nodes: tuple[str, ...]

    levels = frozenset({Level.TEST})

    @override
    def __post_init__(self) -> None:
        need(bool(self.nodes), f"{self.name} names no test")

    @override
    def execute(self, ws: Workspace, changed: Sequence[str] = ()) -> Result:
        ran = run_nodes(ws, self.nodes)
        return Result(self.name, max(ran.outcomes.values()), ran.output)

    @override
    def unwired(self, level: Level, root: Path) -> str | None:
        return next(filter(None, (_node_problem(node, root) for node in self.nodes)), None)


def _node_problem(node: str, root: Path) -> str | None:
    """A test nothing collects enforces nothing: ``tools/test_quality_gate.py`` ran nowhere."""
    path, _, tail = node.partition("::")
    pyproject = root / "pyproject.toml"
    config = tomllib.loads(pyproject.read_bytes().decode()) if pyproject.is_file() else {}
    pytest_ini = config.get("tool", {}).get("pytest", {}).get("ini_options", {})
    trees = pytest_ini.get("testpaths", [])
    if not any(PurePosixPath(path).is_relative_to(tree) for tree in trees):
        return f"{path} is outside testpaths {trees}: nothing collects it"
    if not (root / path).is_file():
        return f"{path} does not exist"
    name = tail.partition("[")[0].rpartition("::")[2]
    try:
        nodes = ast.walk(ast.parse((root / path).read_bytes()))
    except SyntaxError as exc:
        return f"{path} does not parse: {exc}"
    defined = any(
        isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name == name for n in nodes
    )
    return None if defined else f"{path} defines no {name}"


@dataclass(frozen=True, kw_only=True)
class HookCheck(Check, kind="hook"):
    """Reads a tool call on stdin and answers as a PreToolUse hook does."""

    levels = frozenset({Level.HOOK})

    @override
    def execute(self, ws: Workspace, changed: Sequence[str] = ()) -> Result:
        return Result(self.name, Exit.UNMEASURED, "a hook answers tool calls; it has no batch run")

    def decide(self, ws: Workspace, payload: Mapping[str, object]) -> str | None:
        """``allow``, ``ask`` or ``deny``, or None when the hook gave no answer to stand by."""
        done = ws.spawn(self.argv(ws), stdin=json.dumps(payload))
        if done.returncode != 0:
            return None
        if not done.stdout.strip():
            return "allow"
        try:
            return str(json.loads(done.stdout)["hookSpecificOutput"]["permissionDecision"])
        except (json.JSONDecodeError, KeyError, TypeError):
            return None


def changed_files(ws: Workspace, commits: str | None, staged: bool) -> tuple[str, ...] | None:
    """The paths a range or the index changed; None when neither was asked for: the whole tree."""
    if commits is None and not staged:
        return None
    if commits is not None and ".." not in commits:
        raise CannotMeasure(f"{commits!r} is not a range: A..B")
    which = ["--cached"] if staged else ["--end-of-options", str(commits)]
    done = ws.spawn(["git", "diff", "--name-only", "--no-renames", "-z", *which])
    if done.returncode != 0:
        raise CannotMeasure(done.stderr.strip())
    return tuple(filter(None, done.stdout.split("\0")))


def _run_one(check: Check, ws: Workspace, changed: Sequence[str] | None) -> Result:
    start = time.monotonic()
    named: Sequence[str] = ()
    if changed is not None:
        relevant = check.relevant(changed)
        if not relevant:
            return Result(check.name, Exit.PASS, skipped="no changed file matches its trigger")
        named = [path for path in relevant if (ws.root / path).is_file()]
        if not named and "{changed}" in check.run:
            return Result(check.name, Exit.PASS, skipped="its changed files were deleted")
    return replace(check.execute(ws, named), seconds=time.monotonic() - start)


def run_all(checks: Sequence[Check], ws: Workspace, changed: Sequence[str] | None) -> list[Result]:
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        return list(pool.map(lambda check: _run_one(check, ws, changed), checks))
