"""The last stop before main: what leaves is judged again, whatever happened at commit.

A check reads the working copy, so the working copy has to be what leaves.  Exit 2 refuses
here as exit 1 does: a check that could not measure has said nothing about the commits.
"""

import re
import time
from dataclasses import dataclass
from typing import Final, Self

from tools.rules.checks import EMPTY_TREE, CannotMeasure, Changes, Workspace, git_names, run_all
from tools.rules.model import Exit, Level
from tools.rules.registry import Registry

_MAIN: Final = "refs/heads/main"
_RELEASE: Final = re.compile(r"refs/tags/v[^/]*")
_NO_OBJECT: Final = re.compile(r"0+")
# What the checks read.  A change to one of these would be judged in place of what leaves.
_READ: Final = ("lucid", "tools", "scripts", ".githooks", "pyproject.toml")


def _git(ws: Workspace, *args: str) -> str:
    done = ws.spawn(["git", *args])
    if done.returncode != 0:
        raise CannotMeasure(f"git {args[0]}: {done.stderr.strip()}")
    return done.stdout.strip()


def run_level(registry: Registry, level: Level, changes: Changes | None) -> Exit:
    """Run what ``level`` claims over ``changes`` (None: the whole tree); the worst exit."""
    named = registry.at_level(level)
    if not named:
        print(f"[rules] nothing claims level {level}; running nothing proves nothing")
        return Exit.UNMEASURED
    what = "the whole tree" if changes is None else f"{len(changes.paths)} changed file(s)"
    print(f"level {level}: {len(named)} check(s) over {what}")
    results = run_all([registry.checks[name] for name in named], registry.workspace, changes)
    for result, rules in zip(results, named.values(), strict=True):
        word = "SKIP" if result.skipped else result.outcome.name
        note = result.skipped or f"{result.seconds:.1f}s"
        print(f"  {word:<11s}{result.check:<20s}[{' '.join(r.id for r in rules)}]  {note}")
        if result.outcome is not Exit.PASS:
            print("".join(f"      {line}\n" for line in result.output.splitlines()), end="")
    return max(result.outcome for result in results)


@dataclass(frozen=True)
class PushRange:
    """The commits that leave.  ``base`` may be the empty tree: nothing of them is there yet."""

    base: str
    tip: str

    @classmethod
    def parse(cls, text: str) -> Self:
        base, dots, tip = text.partition("..")
        if not (base and dots and tip):
            raise CannotMeasure(f"{text!r} is not a range: A..B")
        return cls(base, tip)

    def changes(self, ws: Workspace) -> Changes:
        """Refuses unless the working copy is the tip, for every path a check reads."""
        tip = _git(ws, "rev-parse", "--verify", "--end-of-options", f"{self.tip}^{{commit}}")
        head = _git(ws, "rev-parse", "--verify", "HEAD^{commit}")
        if tip != head:
            raise CannotMeasure(
                f"{self.tip} is {tip[:9]} and HEAD is {head[:9]}: the checks read the working "
                "copy, so check out what you are pushing"
            )
        dirty = _git(ws, "status", "--porcelain", "--", *_READ)
        if dirty:
            raise CannotMeasure(
                f"the checks read the working copy, and it is not {tip[:9]}. Commit these or "
                f"set them aside:\n{dirty}"
            )
        return Changes(git_names(ws, "--end-of-options", self.base, tip), since=self.base)

    def log(self, ws: Workspace) -> list[list[str]]:
        """Each commit: short hash, its bypass trailer, subject."""
        spec = self.tip if self.base == EMPTY_TREE else f"{self.base}..{self.tip}"
        fields = "%h%x00%(trailers:key=Lucid-Bypass,valueonly,separator=%x2C)%x00%s"
        lines = _git(ws, "log", f"--format={fields}", "--end-of-options", spec).splitlines()
        return [line.split("\0", 2) for line in lines]


@dataclass(frozen=True)
class RefUpdate:
    """One line of what git gives a pre-push hook."""

    local_sha: str
    remote_ref: str
    remote_sha: str

    @classmethod
    def parse(cls, line: str) -> Self:
        words = line.split()
        if len(words) != 4:
            raise CannotMeasure(f"not a ref update: {line!r}")
        return cls(*words[1:])

    def push_range(self, ws: Workspace, remote: str) -> PushRange | None:
        """What this update takes to the remote, or None when it is none of the gate's business."""
        release = _RELEASE.fullmatch(self.remote_ref) is not None
        if not release and self.remote_ref != _MAIN:
            return None
        if _NO_OBJECT.fullmatch(self.local_sha):
            if release:
                return None
            raise CannotMeasure(f"this deletes {self.remote_ref}: there is nothing to judge")
        if not _NO_OBJECT.fullmatch(self.remote_sha):
            return PushRange(self.remote_sha, self.local_sha)
        if not release:
            return PushRange(EMPTY_TREE, self.local_sha)
        # A new tag brings only what the remote's main does not hold, usually nothing.
        main = ws.spawn(["git", "rev-parse", "--verify", "-q", f"refs/remotes/{remote}/main"])
        base = main.stdout.strip() if main.returncode == 0 else EMPTY_TREE
        known = ws.spawn(["git", "merge-base", "--is-ancestor", self.local_sha, base])
        return None if known.returncode == 0 else PushRange(base, self.local_sha)


@dataclass(frozen=True)
class PushGate:
    registry: Registry

    def ranges(self, updates: str, remote: str) -> list[PushRange]:
        """The ranges in what git wrote to the hook; the same range twice is judged once."""
        lines = [line for line in updates.splitlines() if line.strip()]
        found = (
            RefUpdate.parse(line).push_range(self.registry.workspace, remote) for line in lines
        )
        return list(dict.fromkeys(push for push in found if push is not None))

    def judge(self, pushes: list[PushRange]) -> Exit:
        return max((self._judge(push) for push in pushes), default=Exit.PASS)

    def _judge(self, push: PushRange) -> Exit:
        start = time.monotonic()
        ws = self.registry.workspace
        changes = push.changes(ws)
        commits = push.log(ws)
        print(f"push: {len(commits)} commit(s), {len(changes.paths)} file(s), {push.tip[:9]}")
        for short, bypass, subject in commits:
            if bypass:
                print(f"  bypassed at commit ({bypass}): {short} {subject}")
        outcome = run_level(self.registry, Level.PUSH, changes)
        verdict = "passes" if outcome is Exit.PASS else "is refused"
        print(f"push {verdict}, {time.monotonic() - start:.1f}s")
        return outcome
