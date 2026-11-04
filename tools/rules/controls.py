"""A green check is no evidence until it has gone red.  Each kind shows that in its own way."""

import re
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import ClassVar, Final, override

from tools.rules.checks import WORKERS, Check, CommandCheck, NodeRun, Result, Workspace, run_nodes
from tools.rules.model import Exit, RuleError, inside, need

_TAIL: Final = 400


@dataclass(frozen=True)
class Proof:
    outcome: Exit
    detail: str


@dataclass(frozen=True, kw_only=True)
class Control(ABC):
    check: tuple[str, ...]

    kinds: ClassVar[dict[str, type[Control]]] = {}

    @override
    def __init_subclass__(cls, *, kind: str, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        Control.kinds[kind] = cls

    def fits(self, check: Check) -> bool:
        """Whether this kind of control can prove that kind of check."""
        return True

    @abstractmethod
    def prove(self, ws: Workspace, check: Check) -> Proof: ...

    @classmethod
    def prove_all(cls, ws: Workspace, items: Sequence[tuple[Control, Check]]) -> list[Proof]:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            return list(pool.map(lambda item: item[0].prove(ws, item[1]), items))


@dataclass(frozen=True, kw_only=True)
class SeedControl(Control, kind="seed"):
    """Proves a tool on the least tree it needs: its ``Path(__file__)`` root is the scratch.

    ``clean`` is a near miss at ``path`` and must pass; ``text`` there must fail with
    ``expect`` in the output.  The clean run catches a tool that cannot run at all and
    one that flags everything.  A check reading the index is shown the index: the seed
    is staged in a scratch repository.
    """

    path: str
    clean: str
    text: str
    expect: str
    tree: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        inside((self.path, *self.tree), f"seed for {self.check}")
        try:
            re.compile(self.expect)
        except re.error as exc:
            raise RuleError(f"seed for {self.check}: expect {self.expect!r}: {exc}") from exc

    @override
    def fits(self, check: Check) -> bool:
        return isinstance(check, CommandCheck)

    @override
    def prove(self, ws: Workspace, check: Check) -> Proof:
        if not check.relevant([self.path]):
            return Proof(Exit.VIOLATION, f"its trigger leaves out {self.path}: runs would skip it")
        with tempfile.TemporaryDirectory(prefix="lucid-rules-") as tmp:
            scratch = replace(ws, root=Path(tmp).resolve(), isolated=True)
            for relative in (check.tool, *self.tree):
                target = scratch.root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                copied = scratch.spawn(["cp", "-cR", str(ws.root / relative), str(target)])
                if copied.returncode != 0:
                    return Proof(Exit.UNMEASURED, f"could not copy {relative}: {copied.stderr}")
            if check.reads_index:
                scratch.spawn(["git", "init", "-q"])
            return self._judge(scratch, check)

    def _planted(self, scratch: Workspace, check: Check, text: str) -> Result:
        seed = scratch.root / self.path
        seed.parent.mkdir(parents=True, exist_ok=True)
        seed.write_text(text)
        if check.reads_index:
            scratch.spawn(["git", "add", "--", self.path])
        return check.execute(scratch, [self.path])

    def _judge(self, scratch: Workspace, check: Check) -> Proof:
        clean = self._planted(scratch, check, self.clean)
        if clean.outcome is not Exit.PASS:
            detail = f"not green on the near miss: {clean.output[-_TAIL:]}"
            return Proof(Exit.UNMEASURED, detail)
        red = self._planted(scratch, check, self.text)
        if red.outcome is Exit.PASS:
            return Proof(Exit.VIOLATION, f"stayed green with {self.path} seeded")
        if red.outcome is Exit.UNMEASURED:
            return Proof(Exit.UNMEASURED, f"could not run seeded: {red.output[-_TAIL:]}")
        if re.search(self.expect, red.output) is None:
            detail = f"went red without /{self.expect}/: {red.output[-_TAIL:]}"
            return Proof(Exit.VIOLATION, detail)
        return Proof(Exit.PASS, f"red on {self.path}")


@dataclass(frozen=True, kw_only=True)
class PytestControl(Control, kind="pytest"):
    """Named nodes, all in one process.  Each must run and pass; an empty body cannot be told."""

    nodes: tuple[str, ...]

    def __post_init__(self) -> None:
        need(bool(self.nodes), f"a pytest control for {self.check} names no test")

    @override
    def prove(self, ws: Workspace, check: Check) -> Proof:
        return self.prove_all(ws, [(self, check)])[0]

    def _judge(self, ran: NodeRun) -> Proof:
        worst = max(ran.outcomes[node] for node in self.nodes)
        if worst is Exit.PASS:
            return Proof(Exit.PASS, f"{len(self.nodes)} test(s) ran and passed")
        off = [node for node in self.nodes if ran.outcomes[node] is not Exit.PASS]
        word = "failed" if worst is Exit.VIOLATION else "did not run"
        return Proof(worst, f"{', '.join(off)} {word}\n{ran.outputs[off[0]][-_TAIL:]}")

    @classmethod
    @override
    def prove_all(cls, ws: Workspace, items: Sequence[tuple[Control, Check]]) -> list[Proof]:
        controls = [control for control, _ in items if isinstance(control, PytestControl)]
        nodes = dict.fromkeys(node for control in controls for node in control.nodes)
        ran = run_nodes(ws, list(nodes))
        return [control._judge(ran) for control in controls]
