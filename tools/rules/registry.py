"""The registry: every record in ``registry/*.toml`` and the private layer, cross-validated."""

import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Self

from tools.rules.checks import Check, Workspace
from tools.rules.controls import Control
from tools.rules.model import (
    Allowlist,
    Budget,
    Enforcement,
    Level,
    Rule,
    RuleError,
    need,
    read,
    read_kind,
)

REGISTRY_DIR: Final = "tools/rules/registry"
LOCAL_LAYER: Final = ".claude/rules.local.toml"


def _tables[T](
    table: dict[str, object], key: str, where: str, build: Callable[[Mapping[str, object], str], T]
) -> tuple[T, ...]:
    value = table.pop(key, [])
    items = [value] if isinstance(value, Mapping) else value
    if not isinstance(items, list) or not all(isinstance(item, Mapping) for item in items):
        raise RuleError(f"{where}: {key!r} must hold tables")
    return tuple(build(item, f"{where}.{key}[{n}]") for n, item in enumerate(items))


def _named(
    table: dict[str, object], key: str, where: str
) -> list[tuple[str, Mapping[str, object]]]:
    value = table.pop(key, {})
    fine = isinstance(value, Mapping) and all(isinstance(v, Mapping) for v in value.values())
    need(fine, f"{where}: {key}.* must be tables")
    return list(value.items()) if isinstance(value, Mapping) else []


def _rule(table: Mapping[str, object], where: str) -> Rule:
    left = dict(table)
    allowlist = _tables(left, "allowlist", where, lambda t, w: read(Allowlist, t, w))
    need(len(allowlist) < 2, f"{where}: one allowlist per rule")
    return read(
        Rule,
        left,
        where,
        enforce=_tables(left, "enforce", where, lambda t, w: read(Enforcement, t, w)),
        control=_tables(left, "control", where, lambda t, w: read_kind(Control.kinds, t, w)),
        allowlist=allowlist[0] if allowlist else None,
    )


@dataclass(frozen=True)
class Registry:
    root: Path
    rules: Mapping[str, Rule]
    checks: Mapping[str, Check]
    budgets: tuple[Budget, ...]

    def __post_init__(self) -> None:
        need(
            bool(self.rules),
            f"{self.root / REGISTRY_DIR}: no rules, and nothing says that is right",
        )
        for rule in self.rules.values():
            for enforcement in rule.enforce:
                check = self.checks.get(enforcement.check)
                need(check is not None, f"{rule.id}: unknown check {enforcement.check!r}")
                fits = check is not None and enforcement.level in check.levels
                need(fits, f"{rule.id}: {enforcement.check!r} cannot run at {enforcement.level}")
            for control in rule.control:
                unfit = [name for name in control.check if not control.fits(self.checks[name])]
                need(not unfit, f"{rule.id}: a {type(control).__name__} cannot prove {unfit}")
        unclaimed = self.checks.keys() - {e.check for r in self.rules.values() for e in r.enforce}
        need(not unclaimed, f"no rule claims check(s) {sorted(unclaimed)}")

    @classmethod
    def load(cls, root: Path) -> Self:
        files = sorted((root / REGISTRY_DIR).glob("*.toml"))
        files += [file for file in [root / LOCAL_LAYER] if file.is_file()]
        rules: dict[str, Rule] = {}
        checks: dict[str, Check] = {}
        budgets: dict[str, Budget] = {}
        for file in files:
            where = str(file.relative_to(root))
            try:
                doc = tomllib.loads(file.read_bytes().decode())
            except ValueError as exc:  # the TOML, or its encoding
                raise RuleError(f"{where}: {exc}") from exc
            for rule in _tables(doc, "rule", where, _rule):
                need(rule.id not in rules, f"{where}: rule {rule.id} is declared twice")
                rules[rule.id] = rule
            for name, table in _named(doc, "check", where):
                need(name not in checks, f"{where}: check {name} is declared twice")
                checks[name] = read_kind(Check.kinds, table, f"{where}: check.{name}", name=name)
            for name, table in _named(doc, "budget", where):
                need(name not in budgets, f"{where}: budget {name} is declared twice")
                budgets[name] = read(Budget, table, f"{where}: budget.{name}", name=name)
            need(not doc, f"{where}: unknown table(s) {sorted(doc)}")
        return cls(root, rules, checks, tuple(budgets.values()))

    @property
    def workspace(self) -> Workspace:
        return Workspace.of(self.root)

    def at_level(self, level: Level) -> dict[str, tuple[Rule, ...]]:
        """The check names ``level`` runs, each with the rules it answers for."""
        found: dict[str, tuple[Rule, ...]] = {}
        for rule in self.rules.values():
            for name in dict.fromkeys(e.check for e in rule.enforce if e.level is level):
                found[name] = (*found.get(name, ()), rule)
        return found
