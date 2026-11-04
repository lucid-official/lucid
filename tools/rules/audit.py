"""Turns "the config says so" into a checked statement, as far as a static reading can.

Exit 1 is a finding: a control that stayed green, a level that does not name its tool,
a ceiling broken.  Exit 2 means something could not be measured, which is not the same
thing.  Wiring proves that a level names the tool, not that it runs it: that takes a
real hook run.  A test node is proven to run and pass, not to contain an assertion.
"""

from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from typing import Final, override

from tools.rules.checks import Check, PytestCheck
from tools.rules.controls import Control, PytestControl
from tools.rules.model import Allowlist, Budget, Exit, Rule, Verdict
from tools.rules.registry import Registry

_TABLE_SIZE: Final = """\
import runpy, sys
module, _, name = sys.argv[1].partition(":")
table = runpy.run_path(module.replace(".", "/") + ".py")[name]
print(sum(sys.argv[2] in repr(entry) for entry in table))
"""


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
    only_rules: frozenset[str] = frozenset()
    only_checks: frozenset[str] = frozenset()

    def run(self) -> Report:
        sections = (self.tally(), self.controls(), self.wiring(), self.allowlists(), self.budgets())
        return Report(sections)

    @property
    def _rules(self) -> Iterable[Rule]:
        return self.registry.rules.values()

    def tally(self) -> Section:
        counts = Counter(rule.verdict for rule in self._rules)
        parts = ", ".join(f"{counts[verdict]} {verdict}" for verdict in Verdict)
        return Section("rules", f"{sum(counts.values())}: {parts}")

    def _proofs_owed(self, rule: Rule) -> Iterator[tuple[Control, Check]]:
        """Its controls, and each claimed test level's own nodes, which must run and pass."""
        checks = self.registry.checks
        own = [
            PytestControl(check=(e.check,), nodes=check.nodes)
            for e in rule.enforce
            if not e.pending and isinstance(check := checks[e.check], PytestCheck)
        ]
        for control in (*rule.control, *own):
            for name in control.check:
                if not self.only_checks or name in self.only_checks:
                    yield control, checks[name]

    def controls(self) -> Section:
        chosen = [r for r in self._rules if not self.only_rules or r.id in self.only_rules]
        groups: dict[type[Control], list[tuple[Rule, Control, Check]]] = {}
        for rule in chosen:
            for control, check in self._proofs_owed(rule):
                groups.setdefault(type(control), []).append((rule, control, check))
        ws = replace(self.registry.workspace, isolated=True)
        findings: list[Finding] = []
        for kind, group in groups.items():
            proofs = kind.prove_all(ws, [(control, check) for _, control, check in group])
            for (rule, _, check), proof in zip(group, proofs, strict=True):
                if proof.outcome is not Exit.PASS:
                    detail = f"{check.name} control: {proof.detail}"
                    findings.append(Finding(rule.id, proof.outcome, detail))
        total = sum(len(group) for group in groups.values())
        summary = f"{total - len(findings)}/{total} proven: seeded red, near miss green, tests ran"
        return Section("controls", summary, tuple(findings))

    def wiring(self) -> Section:
        named = 0
        pending: Counter[str] = Counter()
        findings = list(self._dead_globs())
        unnamed = 0
        for rule in self._rules:
            for e in rule.enforce:
                claim = f"level={e.level} check={e.check}"
                problem = self.registry.checks[e.check].unwired(e.level, self.registry.root)
                if e.pending and problem is None:
                    message = f"{claim} is named now: drop pending = {e.pending!r}"
                    findings.append(Finding(rule.id, Exit.VIOLATION, message))
                elif e.pending:
                    pending[e.pending] += 1
                elif problem is None:
                    named += 1
                else:
                    unnamed += 1
                    findings.append(Finding(rule.id, Exit.VIOLATION, f"{claim}: {problem}"))
        waits = "".join(f"; {n} pending on {issue}" for issue, n in sorted(pending.items()))
        summary = f"{named}/{named + unnamed} claimed levels name their tool{waits}"
        return Section("wiring", summary, tuple(findings))

    def _dead_globs(self) -> Iterator[Finding]:
        """A glob that matches nothing makes its check skip forever, or its scope empty."""
        owned = [(c.name, c.trigger) for c in self.registry.checks.values()]
        owned += [(r.id, r.scope) for r in self._rules]
        for owner, globs in owned:
            for glob in globs:
                if not any(path.is_file() for path in self.registry.root.glob(glob)):
                    yield Finding(owner, Exit.VIOLATION, f"{glob} matches no file")

    def allowlists(self) -> Section:
        lists = [(r, r.allowlist) for r in self._rules if r.allowlist is not None]
        broken = [found for rule, a in lists if (found := self._allowlist(rule, a))]
        summary = f"{len(lists) - len(broken)}/{len(lists)} at their ceiling"
        return Section("allowlists", summary, tuple(broken))

    def _allowlist(self, rule: Rule, allowlist: Allowlist) -> Finding | None:
        ws = self.registry.workspace
        done = ws.spawn([ws.python, "-c", _TABLE_SIZE, allowlist.source, allowlist.match])
        name = f"allowlist {allowlist.source}"
        if done.returncode != 0:
            return Finding(rule.id, Exit.UNMEASURED, f"{name} could not be read: {done.stderr}")
        size = int(done.stdout)
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
            parts.append(f"{budget.name} {size}/{budget.ceiling}")
            if size is None:
                message = f"measures {list(budget.paths)}, which match nothing"
                findings.append(Finding(budget.name, Exit.UNMEASURED, message))
            elif size > budget.ceiling:
                message = f"{size} {budget.unit} is over its ceiling of {budget.ceiling}"
                findings.append(Finding(budget.name, Exit.VIOLATION, message))
            elif budget.unit == "pending" and size < budget.ceiling:
                message = f"fell to {size}: lower ceiling {budget.ceiling} in this commit"
                findings.append(Finding(budget.name, Exit.VIOLATION, message))
        return Section("budgets", ", ".join(parts), tuple(findings))

    def _measure(self, budget: Budget) -> int | None:
        if budget.unit == "pending":
            levels = sum(1 for rule in self._rules for e in rule.enforce if e.pending)
            return levels + sum(1 for rule in self._rules if rule.pending)
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
