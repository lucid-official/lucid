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
# ``git hash-object -t tree /dev/null``: what a tree is measured from when all of it counts.
EMPTY_TREE: Final = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
# Where a CI claim has to name a check.  ``.pre-commit-config.yaml`` is not here: it
# named mypy and ruff at commit for years and never ran.  Naming is all this proves.
_CI_FILES: Final = ("scripts/ci_full.sh", ".github/workflows/*.yml")
# What stands for every check of a level.  ``--staged``, because without it a blob reader
# skips; the refs, because without them the push gate has nothing to judge and passes.
_RUNNERS: Final[Mapping[Level, str]] = {
    Level.COMMIT: r"tools\.rules run --level commit --staged",
    Level.PUSH: r"tools\.rules push \S+ \S+ \S+ \S+ refs/heads/main ",
    Level.CI: r"tools\.rules run --level ci",
}
_COMMENT: Final = re.compile(r"(?:^|[\s;])#.*$", re.MULTILINE)
# A stub python: it says what it was asked, and fails when told to.  The interpreter
# question a hook asks first must still be answered, or the failure proves nothing.
_STUB: Final = """\
#!/bin/sh
asked="$*"
case "$asked" in *tools.rules\\ push*) asked="$asked $(cat)" ;; esac
printf '%s\\n' "$asked" >> "{log}"
case "$asked" in -c\\ *) exit 0 ;; esac
test ! -e "{fail}"
"""


class CannotMeasure(RuntimeError):
    """The question could not be put: an unreadable range, for one."""


@dataclass(frozen=True)
class Workspace:
    root: Path
    python: str
    isolated: bool = False
    since: str = EMPTY_TREE  # where a check that measures changes starts

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
    since: str = "HEAD"
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
        filled = {"{py}": ws.python, "{since}": ws.since}
        words = (changed if w == "{changed}" else [filled.get(w, w)] for w in self.run)
        return [word for group in words for word in group]

    def relevant(self, changed: Sequence[str]) -> list[str]:
        """The changed paths this check is about; none means it has nothing to say."""
        if not self.trigger:
            return list(changed)
        return [p for p in changed if any(PurePosixPath(p).full_match(g) for g in self.trigger)]

    def unwired(self, level: Level, root: Path) -> str | None:
        """Why ``level`` does not run this check, or None: a hook runs it, CI still only names it."""
        wanted = rf"(?<![\w.-]){re.escape(self.tool)}(?![\w./-])|{_RUNNERS[level]}"
        if level is not Level.CI:
            return HookTrial(level, root / ".githooks", wanted, self.tool, self.trigger).problem()
        files = [file for place in _CI_FILES for file in root.glob(place) if file.is_file()]
        text = "\n".join(_COMMENT.sub("", f.read_text(errors="replace")) for f in files)
        return None if re.search(wanted, text) else f"{' or '.join(_CI_FILES)}: no {self.tool}"

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


@dataclass(frozen=True)
class HookTrial:
    """Does a real commit or push run ``wanted`` and stop when it fails?

    Git runs the real hook in a scratch repository whose python is a stub, in a main
    checkout and in a linked worktree.  So an ``exit 0`` above the line, an ``if false``,
    a name in an ``echo`` and a dropped exit status all fail here, as text cannot show.
    """

    level: Level
    hooks: Path
    wanted: str
    tool: str
    trigger: tuple[str, ...]

    def problem(self) -> str | None:
        with tempfile.TemporaryDirectory(prefix="lucid-rules-") as tmp:
            base = Path(tmp).resolve()
            repo, linked = base / "repo", base / "linked"
            self._repository(base, repo, linked)
            return self._tried(base, repo, "the main checkout") or self._tried(
                base, linked, "a linked worktree"
            )

    def _repository(self, base: Path, repo: Path, linked: Path) -> None:
        off = ("-c", "core.hooksPath=/dev/null")
        _git(base, "init", "-q", "--bare", "remote.git")
        _git(base, "init", "-q", "-b", "main", "repo")
        (repo / ".gitignore").write_text(".venv\n")
        (repo / "README.md").write_text("trial\n")
        (repo / "tools/rules").mkdir(parents=True)  # a hook asks for it before running it
        (repo / "tools/rules/__main__.py").write_text("")
        for key, value in (
            ("user.name", "trial"),
            ("user.email", "trial@example.invalid"),
            ("commit.gpgsign", "false"),
            ("core.hooksPath", str(self.hooks)),
            ("remote.origin.url", str(base / "remote.git")),
        ):
            _git(repo, "config", key, value)
        _git(repo, "add", "-A")
        _git(repo, *off, "commit", "-qm", "chore: base")
        _git(repo, *off, "push", "-q", "origin", "main")
        _git(repo, "worktree", "add", "-q", "-b", "side", str(linked))

    def _attempt(self, tree: Path, mark: str) -> subprocess.CompletedProcess[str]:
        # Named for its tree: within a second, two trees would make one commit, and a
        # push of it would be up to date, with no refs for the hook.
        subject = f"chore: {mark} in {tree.name}"
        # A path each trigger matches, so that a hook's own path guard fires.
        for glob in self.trigger:
            path = tree / re.sub(r"\*+", "f", glob.replace("**/", "d/"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(subject)
        _git(tree, "add", "-A")
        commit = ("commit", "--allow-empty", "-qm", subject)
        if self.level is Level.COMMIT:
            return _git(tree, *commit)
        _git(tree, "-c", "core.hooksPath=/dev/null", *commit)
        return _git(tree, "push", "-q", "-f", "origin", "HEAD:main")

    def _tried(self, base: Path, tree: Path, where: str) -> str | None:
        log, fail = base / "log", base / "fail"
        stub = tree / ".venv/bin/python3"
        stub.parent.mkdir(parents=True)
        stub.write_text(_STUB.format(log=log, fail=fail))
        stub.chmod(0o755)
        log.unlink(missing_ok=True)
        fail.unlink(missing_ok=True)
        verb = f"a real {'commit' if self.level is Level.COMMIT else 'push'} in {where}"
        passing = self._attempt(tree, "passing")
        if passing.returncode != 0:
            return f"{verb} failed with its tool passing: {passing.stderr.strip()[-200:]}"
        if not log.is_file() or re.search(self.wanted, log.read_text()) is None:
            return f"{verb} did not run {self.tool}"
        fail.touch()
        if self._attempt(tree, "failing").returncode == 0:
            return f"{verb} went through with {self.tool} failing"
        return None


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return Workspace(cwd, "", isolated=True).spawn(["git", *args])


def git_names(ws: Workspace, *which: str) -> tuple[str, ...]:
    done = ws.spawn(["git", "diff", "--name-only", "--no-renames", "-z", *which])
    if done.returncode != 0:
        raise CannotMeasure(done.stderr.strip())
    return tuple(filter(None, done.stdout.split("\0")))


def staged_changes(ws: Workspace) -> Changes:
    return Changes(git_names(ws, "--cached"), staged=True, differ=frozenset(git_names(ws)))


def _run_one(check: Check, ws: Workspace, changes: Changes | None) -> Result:
    start = time.monotonic()
    ws = replace(ws, since=EMPTY_TREE if changes is None else changes.since)
    if check.reads_index and (changes is None or not changes.staged):
        return Result(check.name, Exit.PASS, skipped="it reads the index: only --staged can say")
    named: Sequence[str] = ()
    if changes is not None:
        relevant = check.relevant(changes.paths)
        if not relevant:
            return Result(check.name, Exit.PASS, skipped="no changed file matches its trigger")
        differ = [] if check.reads_index else sorted(changes.differ.intersection(relevant))
        if differ:
            message = (
                f"it reads the working copy, which is not what is staged: {', '.join(differ)}. "
                "Stage all of each file, or set the rest aside."
            )
            return Result(check.name, Exit.UNMEASURED, message)
        named = [path for path in relevant if (ws.root / path).is_file()]
        if not named and "{changed}" in check.run:
            return Result(check.name, Exit.PASS, skipped="its changed files were deleted")
    return replace(check.execute(ws, named), seconds=time.monotonic() - start)


def run_all(checks: Sequence[Check], ws: Workspace, changes: Changes | None) -> list[Result]:
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        return list(pool.map(lambda check: _run_one(check, ws, changes), checks))
