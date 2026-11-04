"""Turns "the config says so" into a checked statement, section by section.

Exit 1 is a finding: a control that stayed green, a claimed level not wired, a
ceiling broken.  Exit 2 means a control could not run, which is not the same thing.
"""

import ast
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import override

from tools.rules.checks import Check
from tools.rules.controls import Control
from tools.rules.model import Allowlist, Budget, Exit, Rule, Verdict
from tools.rules.registry import Registry


@dataclass(frozen=True)
class Finding:
    subject: str
    outcome: Exit
    message: str


@dataclass(frozen=True)
class Section:
    name: str
    summary: str
    findings: tuple[Finding, ...] = ()


@dataclass(frozen=True)
class Report:
    sections: tuple[Section, ...]

    @property
    def findings(self) -> list[Finding]:
        return [finding for section in self.sections for finding in section.findings]

    @property
    def outcome(self) -> Exit:
        return max((finding.outcome for finding in self.findings), default=Exit.PASS)

    @override
    def __str__(self) -> str:
        lines = [f"  {s.name:<11s}{s.summary}" for s in self.sections]
        lines += [f"  {f.outcome.name:<11s}{f.subject}: {f.message}" for f in self.findings]
        return "\n".join(lines)


@dataclass(frozen=True)
class Audit:
    registry: Registry
    today: date = field(default_factory=date.today)
    only: frozenset[str] = frozenset()

    def run(self) -> Report:
        return Report(
            (self.tally(), self.controls(), self.wiring(), self.exemptions(), self.budgets())
        )

    @property
    def _rules(self) -> Iterable[Rule]:
        return self.registry.rules.values()

    def tally(self) -> Section:
        counts = Counter(rule.verdict for rule in self._rules)
        parts = ", ".join(f"{counts[verdict]} {verdict}" for verdict in Verdict)
        return Section("rules", f"{sum(counts.values())}: {parts}")

    def controls(self) -> Section:
        chosen = [r for r in self._rules if not self.only or r.id in self.only]
        groups: dict[type[Control], list[tuple[Rule, Control, Check]]] = {}
        for rule in chosen:
            for control in rule.control:
                item = (rule, control, self.registry.checks[control.check])
                groups.setdefault(type(control), []).append(item)
        ws = replace(self.registry.workspace, isolated=True)
        with ThreadPoolExecutor(max_workers=max(len(groups), 1)) as pool:
            done = [
                (group, pool.submit(kind.prove_all, ws, [(c, check) for _, c, check in group]))
                for kind, group in groups.items()
            ]
            findings = tuple(
                Finding(rule.id, proof.outcome, f"{control.check} control: {proof.detail}")
                for group, future in done
                for (rule, control, _), proof in zip(group, future.result(), strict=True)
                if proof.outcome is not Exit.PASS
            )
        total = sum(len(group) for group in groups.values())
        summary = f"{total - len(findings)}/{total} red when seeded, green when clean"
        return Section("controls", summary, findings)

    def wiring(self) -> Section:
        wired = 0
        pending: Counter[str] = Counter()
        findings: list[Finding] = []
        for rule in self._rules:
            for e in rule.enforce:
                claim = f"level={e.level} check={e.check}"
                problem = self.registry.checks[e.check].unwired(e.level, self.registry.root)
                if e.pending and problem is None:
                    message = f"{claim} is wired now: drop pending = {e.pending!r}"
                    findings.append(Finding(rule.id, Exit.VIOLATION, message))
                elif e.pending:
                    pending[e.pending] += 1
                elif problem is None:
                    wired += 1
                else:
                    findings.append(Finding(rule.id, Exit.VIOLATION, f"{claim}: {problem}"))
        waits = "".join(f"; {n} pending on {issue}" for issue, n in sorted(pending.items()))
        summary = f"{wired}/{wired + len(findings)} claimed levels really run the check{waits}"
        return Section("wiring", summary, tuple(findings))

    def exemptions(self) -> Section:
        findings: list[Finding] = []
        live = 0
        for rule in self._rules:
            for exemption in rule.exempt:
                path = exemption.match.partition(":")[0]
                message = None
                if not (self.registry.root / path).exists():
                    message = f"exemption for {exemption.match} is stale: {path} is gone"
                elif exemption.expires is not None and exemption.expires < self.today:
                    message = f"exemption for {exemption.match} expired {exemption.expires}"
                if message:
                    findings.append(Finding(rule.id, Exit.VIOLATION, message))
                live += message is None
        lists = [(r, r.allowlist) for r in self._rules if r.allowlist is not None]
        broken = [found for rule, a in lists if (found := self._allowlist(rule, a))]
        summary = (
            f"{live} live; {len(lists) - len(broken)}/{len(lists)} allowlists at their ceiling"
        )
        return Section("exemptions", summary, tuple(findings + broken))

    def _allowlist(self, rule: Rule, allowlist: Allowlist) -> Finding | None:
        size = _allowlist_size(self.registry.root, allowlist.source)
        name = f"allowlist {allowlist.source}"
        if size is None:
            return Finding(rule.id, Exit.UNMEASURED, f"{name} could not be read")
        if size > allowlist.ceiling:
            return Finding(rule.id, Exit.VIOLATION, f"{name} grew: {size} > {allowlist.ceiling}")
        if size < allowlist.ceiling:
            message = f"{name} shrank to {size}: lower ceiling {allowlist.ceiling} in this commit"
            return Finding(rule.id, Exit.VIOLATION, message)
        return None

    def budgets(self) -> Section:
        parts: list[str] = []
        findings: list[Finding] = []
        for budget in self.registry.budgets:
            size = self._measure(budget)
            parts.append(f"{budget.name} {'absent' if size is None else size}/{budget.ceiling}")
            if size is None and not budget.absent_ok:
                message = f"measures {list(budget.paths)}, which match nothing"
                findings.append(Finding(budget.name, Exit.UNMEASURED, message))
            if size is not None and size > budget.ceiling:
                message = f"{size} {budget.unit} is over its ceiling of {budget.ceiling}"
                findings.append(Finding(budget.name, Exit.VIOLATION, message))
        return Section("budgets", ", ".join(parts), tuple(findings))

    def _measure(self, budget: Budget) -> int | None:
        if budget.unit == "pending-rules":
            return sum(1 for rule in self._rules if rule.verdict is Verdict.PENDING)
        files = [f for p in budget.paths for f in self.registry.root.glob(p) if f.is_file()]
        if not files:
            return None
        if budget.unit == "bytes":
            return sum(file.stat().st_size for file in files)
        lines = (line for f in files for line in f.read_text(errors="replace").splitlines())
        return sum(1 for line in lines if _is_code(line))


def _is_code(line: str) -> bool:
    # A why-comment is what the rules ask for, so a ceiling must not charge for one.
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith("#")


def _allowlist_size(root: Path, source: str) -> int | None:
    """Read by AST: auditing a tool's table should not depend on importing the tool."""
    module, _, name = source.partition(":")
    file = root / (module.replace(".", "/") + ".py")
    if not file.is_file():
        return None
    try:
        body = ast.parse(file.read_bytes()).body
    except SyntaxError:
        return None
    for node in body:
        if isinstance(node, ast.Assign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                return _literal_size(node.value)
    return None


def _literal_size(value: ast.expr | None) -> int | None:
    if isinstance(value, ast.Dict):
        return len(value.keys)
    if isinstance(value, ast.List | ast.Tuple | ast.Set):
        return len(value.elts)
    if isinstance(value, ast.Call) and len(value.args) == 1:
        return _literal_size(value.args[0])
    return None
