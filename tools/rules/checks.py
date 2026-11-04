"""A check is a subprocess with one reading of its exit: 0 pass, 1 violation, 2 cannot measure."""

import itertools
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

from tools.rules.model import Exit, Level, inside, need

WORKERS: Final = 4
_TIMEOUT: Final = 600
# Where a level has to name a check.  ``.pre-commit-config.yaml`` is not here: it
# named mypy and ruff at commit for years and never ran.  Naming is all this proves.
_WIRED_IN: Final[Mapping[Level, tuple[str, ...]]] = {
    Level.COMMIT: (".githooks/pre-commit",),
    Level.PUSH: (".githooks/pre-push",),
    Level.CI: ("scripts/ci_full.sh", ".github/workflows/*.yml"),
}
_RUNNERS: Final[Mapping[Level, tuple[str, ...]]] = {
    Level.COMMIT: ("tools.rules run --level commit",),
    Level.PUSH: ("tools.rules run --level push", "tools.rules push"),
    Level.CI: ("tools.rules run --level ci",),
}
_COMMENT: Final = re.compile(r"(?:^|[\s;])#.*$", re.MULTILINE)


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

    def spawn(self, argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        # A hook exports GIT_DIR and GIT_INDEX_FILE, which in a control would
        # send git in a scratch tree to the real repository.
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        try:
            return subprocess.run(
                argv,
                cwd=self.root,
                env=env if self.isolated else None,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=_TIMEOUT,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return subprocess.CompletedProcess(argv, Exit.UNMEASURED, "", f"{exc!r}")


@dataclass(frozen=True)
class Changes:
    """What a range or the index changed.  ``differ``: paths whose working copy is not staged."""

    paths: tuple[str, ...]
    staged: bool = False
    differ: frozenset[str] = frozenset()


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
        inside(self.trigger, self.name)

    @property
    def tool(self) -> str:
        """What names this check where it runs: the script, or the module and its subcommand."""
        words = itertools.dropwhile(lambda w: w == "{py}" or w.startswith("-"), self.run)
        return " ".join(itertools.takewhile(lambda w: not w.startswith(("-", "{")), words))

    @property
    def reads_index(self) -> bool:
        return "--staged" in self.run

    def argv(self, ws: Workspace, changed: Sequence[str] = ()) -> list[str]:
        words = (changed if w == "{changed}" else [w.replace("{py}", ws.python)] for w in self.run)
        return [word for group in words for word in group]

    def relevant(self, changed: Sequence[str]) -> list[str]:
        """The changed paths this check is about; none means it has nothing to say."""
        if not self.trigger:
            return list(changed)
        return [p for p in changed if any(PurePosixPath(p).full_match(g) for g in self.trigger)]

    def unwired(self, level: Level, root: Path) -> str | None:
        """Why ``level`` does not name this check's tool, or None when it does."""
        places = _WIRED_IN[level]
        files = [file for place in places for file in root.glob(place) if file.is_file()]
        text = "\n".join(_COMMENT.sub("", f.read_text(errors="replace")) for f in files)
        named = rf"(?<![\w.-]){re.escape(self.tool)}(?![\w./-])"
        if re.search(named, text) or any(runner in text for runner in _RUNNERS[level]):
            return None
        return f"{' or '.join(places)} does not name {self.tool}"

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
        outcome = Exit(done.returncode) if good else Exit.UNMEASURED
        return Result(self.name, outcome, done.stdout + done.stderr)


@dataclass(frozen=True)
class NodeRun:
    outcomes: Mapping[str, Exit]
    outputs: Mapping[str, str]


def run_nodes(ws: Workspace, nodes: Sequence[str]) -> NodeRun:
    """At most ``WORKERS`` processes, a test file whole in one: whole-tree nodes cost seconds each.

    A node that did not run proved nothing: 2, not 0.
    """
    files = sorted({node.partition("::")[0] for node in nodes})
    bins: dict[int, list[str]] = {}
    for node in nodes:
        bins.setdefault(files.index(node.partition("::")[0]) % WORKERS, []).append(node)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        runs = list(pool.map(lambda group: _run_nodes_once(ws, group), bins.values()))
    outcomes = {node: outcome for run in runs for node, outcome in run.outcomes.items()}
    return NodeRun(outcomes, {node: text for run in runs for node, text in run.outputs.items()})


def _run_nodes_once(ws: Workspace, nodes: Sequence[str]) -> NodeRun:
    with tempfile.TemporaryDirectory(prefix="lucid-rules-") as tmp:
        report = Path(tmp) / "junit.xml"
        pytest = ("-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider")
        done = ws.spawn([ws.python, *pytest, f"--junitxml={report}", *nodes])
        cases = dict(_testcases(report)) if report.is_file() else {}
    outcomes = {node: _node_outcome(node, cases) for node in nodes}
    return NodeRun(outcomes, dict.fromkeys(nodes, done.stdout + done.stderr))


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
        inside(self.trigger, self.name)

    @override
    def execute(self, ws: Workspace, changed: Sequence[str] = ()) -> Result:
        ran = run_nodes(ws, self.nodes)
        return Result(self.name, max(ran.outcomes.values()), "".join(set(ran.outputs.values())))

    @override
    def unwired(self, level: Level, root: Path) -> str | None:
        """A test nothing collects enforces nothing: ``tools/test_quality_gate.py`` ran nowhere."""
        pyproject = root / "pyproject.toml"
        config = tomllib.loads(pyproject.read_bytes().decode()) if pyproject.is_file() else {}
        trees = config.get("tool", {}).get("pytest", {}).get("ini_options", {}).get("testpaths", [])
        for node in self.nodes:
            path = PurePosixPath(node.partition("::")[0])
            if not any(path.is_relative_to(tree) for tree in trees):
                return f"{path} is outside testpaths {trees}: nothing collects it"
        return None


def _git_names(ws: Workspace, *which: str) -> tuple[str, ...]:
    done = ws.spawn(["git", "diff", "--name-only", "--no-renames", "-z", *which])
    if done.returncode != 0:
        raise CannotMeasure(done.stderr.strip())
    return tuple(filter(None, done.stdout.split("\0")))


def changed_files(ws: Workspace, commits: str | None, staged: bool) -> Changes | None:
    """What a range or the index changed; None when neither was asked for: the whole tree."""
    if staged:
        return Changes(_git_names(ws, "--cached"), True, frozenset(_git_names(ws)))
    if commits is None:
        return None
    if ".." not in commits:
        raise CannotMeasure(f"{commits!r} is not a range: A..B")
    return Changes(_git_names(ws, "--end-of-options", commits))


def _run_one(check: Check, ws: Workspace, changes: Changes | None) -> Result:
    start = time.monotonic()
    if check.reads_index and (changes is None or not changes.staged):
        return Result(check.name, Exit.PASS, skipped="it reads the index: only --staged can say")
    named: Sequence[str] = ()
    if changes is not None:
        relevant = check.relevant(changes.paths)
        if not relevant:
            return Result(check.name, Exit.PASS, skipped="no changed file matches its trigger")
        differ = [] if check.reads_index else sorted(changes.differ.intersection(relevant))
        if differ:
            message = f"it reads the working copy, which is not what is staged: {', '.join(differ)}"
            return Result(check.name, Exit.UNMEASURED, message)
        named = [path for path in relevant if (ws.root / path).is_file()]
        if not named and "{changed}" in check.run:
            return Result(check.name, Exit.PASS, skipped="its changed files were deleted")
    return replace(check.execute(ws, named), seconds=time.monotonic() - start)


def run_all(checks: Sequence[Check], ws: Workspace, changes: Changes | None) -> list[Result]:
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        return list(pool.map(lambda check: _run_one(check, ws, changes), checks))
