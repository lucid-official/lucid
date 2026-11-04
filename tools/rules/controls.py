"""A green check is no evidence until it has gone red.  Each kind shows that in its own way."""

import re
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import ClassVar, Final, override

from tools.rules.checks import WORKERS, Check, CommandCheck, HookCheck, Workspace, run_nodes
from tools.rules.model import Exit, need

_TAIL: Final = 400


@dataclass(frozen=True)
class Proof:
    outcome: Exit
    detail: str


@dataclass(frozen=True, kw_only=True)
class Control(ABC):
    check: str

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

    The same path, empty, must pass; with the seed it must fail with ``expect`` in
    the output.  The clean run is what catches a tool that cannot run at all, which
    otherwise looks either clean or like every file failing.
    """

    path: str
    text: str
    expect: str
    tree: tuple[str, ...] = ()

    @override
    def fits(self, check: Check) -> bool:
        return isinstance(check, CommandCheck)

    @override
    def prove(self, ws: Workspace, check: Check) -> Proof:
        with tempfile.TemporaryDirectory(prefix="lucid-rules-") as tmp:
            scratch = replace(ws, root=Path(tmp).resolve(), isolated=True)
            for relative in (check.tool, *self.tree):
                target = scratch.root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                copied = scratch.spawn(["cp", "-cR", str(ws.root / relative), str(target)])
                if copied.returncode != 0:
                    return Proof(Exit.UNMEASURED, f"could not copy {relative}: {copied.stderr}")
            return self._judge(scratch, check)

    def _judge(self, scratch: Workspace, check: Check) -> Proof:
        seed = scratch.root / self.path
        seed.parent.mkdir(parents=True, exist_ok=True)
        seed.touch()
        clean = check.execute(scratch, [self.path])
        if clean.outcome is not Exit.PASS:
            detail = f"not green on a clean tree: {clean.output[-_TAIL:]}"
            return Proof(Exit.UNMEASURED, detail)
        seed.write_text(self.text)
        red = check.execute(scratch, [self.path])
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
    """Named nodes that already assert the failure, all in one process."""

    nodes: tuple[str, ...]

    def __post_init__(self) -> None:
        need(bool(self.nodes), f"a pytest control for {self.check} names no test")

    @override
    def prove(self, ws: Workspace, check: Check) -> Proof:
        return self.prove_all(ws, [(self, check)])[0]

    def _judge(self, outcomes: Mapping[str, Exit], output: str) -> Proof:
        worst = max(outcomes[node] for node in self.nodes)
        if worst is Exit.PASS:
            return Proof(Exit.PASS, f"{len(self.nodes)} test(s) saw it fail")
        off = [node for node in self.nodes if outcomes[node] is not Exit.PASS]
        word = "failed: the check did not go red" if worst is Exit.VIOLATION else "did not run"
        return Proof(worst, f"{', '.join(off)} {word}\n{output[-_TAIL:]}")

    @classmethod
    @override
    def prove_all(cls, ws: Workspace, items: Sequence[tuple[Control, Check]]) -> list[Proof]:
        controls = [control for control, _ in items if isinstance(control, PytestControl)]
        nodes = dict.fromkeys(node for control in controls for node in control.nodes)
        ran = run_nodes(ws, list(nodes))
        return [control._judge(ran.outcomes, ran.output) for control in controls]


@dataclass(frozen=True, kw_only=True)
class HookControl(Control, kind="hook"):
    """Feeds a tool call to the hook, as a subagent and as the main session."""

    tool: str
    input: str
    expect: str
    main: str | None = None

    @override
    def fits(self, check: Check) -> bool:
        return isinstance(check, HookCheck)

    @override
    def prove(self, ws: Workspace, check: Check) -> Proof:
        assert isinstance(check, HookCheck)  # the registry refuses any other pairing
        call = {"tool_name": self.tool, "tool_input": {"command": self.input}, "cwd": str(ws.root)}
        asked = (
            ("subagent", self.expect, {**call, "agent_id": "control"}),
            ("main session", self.main or self.expect, call),
        )
        for who, expected, payload in asked:
            answer = check.decide(ws, payload)
            if answer is None:
                return Proof(Exit.UNMEASURED, f"the hook gave {who} no answer")
            if answer != expected:
                return Proof(Exit.VIOLATION, f"answered {who} {answer}, not {expected}")
        return Proof(Exit.PASS, f"{self.tool}: {self.input}")
