"""What a record may say.  Stdlib only: the hook and push paths read it on a bare interpreter."""

import re
from dataclasses import MISSING, dataclass, fields
from datetime import date
from enum import IntEnum, StrEnum
from types import UnionType
from typing import TYPE_CHECKING, Final, get_args, get_origin

if TYPE_CHECKING:
    from collections.abc import Mapping

    from _typeshed import DataclassInstance

    from tools.rules.controls import Control

_RULE_ID: Final = re.compile(r"[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*")
_ISSUE: Final = re.compile(r"LCD-\d+(?:-[A-Z][A-Z0-9]*)?")
_DOCS: Final = frozenset({"global", "python", "engine", "models", "tests", "web", "ci"})
_UNITS: Final = frozenset({"bytes", "lines", "pending-rules"})


class RuleError(ValueError):
    """A record breaks the schema.  Nothing is judged on half a registry."""


def need(holds: bool, problem: str) -> None:
    if not holds:
        raise RuleError(problem)


class Level(StrEnum):
    HOOK = "hook"
    COMMIT = "commit"
    PUSH = "push"
    CI = "ci"
    TEST = "test"


class Exit(IntEnum):
    """Every check's exit convention.  ``UNMEASURED`` is never read as a pass."""

    PASS = 0
    VIOLATION = 1
    UNMEASURED = 2


class Verdict(StrEnum):
    ENFORCED = "enforced"
    PARTLY = "partly"
    ADVISORY = "advisory"
    PENDING = "pending"


def _coerce(value: object, hint: object, where: str) -> object:
    hint = get_args(hint)[0] if isinstance(hint, UnionType) else hint
    if get_origin(hint) is tuple:
        items = value if isinstance(value, list) else [None]
        need(all(isinstance(item, str) for item in items), f"{where} must list strings")
        return tuple(items)
    if hint is str and isinstance(value, list):  # parts, where the whole may not be written here
        return "".join(str(_coerce(part, str, where)) for part in value)
    if isinstance(hint, type) and issubclass(hint, StrEnum):
        need(value in list(hint), f"{where}: unknown {hint.__name__} {value!r}")
        return hint(str(value))
    fits = isinstance(hint, type) and isinstance(value, hint)
    fits = fits and (hint is bool or not isinstance(value, bool))
    need(fits, f"{where}: {value!r} is not {hint}")
    return value


def read[T: DataclassInstance](
    cls: type[T], table: Mapping[str, object], where: str, **given: object
) -> T:
    """``cls`` from a TOML table.  Its annotations are the schema; what nobody read is a typo."""
    left = dict(table)
    need(left.keys().isdisjoint(given), f"{where}: {sorted(given)} are not its to set")
    values = dict(given)
    for field in fields(cls):
        if field.name in left:
            value = left.pop(field.name)
            values[field.name] = _coerce(value, field.type, f"{where}: {field.name}")
        elif field.name not in given and field.default is MISSING:
            raise RuleError(f"{where}: {field.name!r} is missing")
    need(not left, f"{where}: unknown key(s) {sorted(left)}")
    return cls(**values)


def read_kind[T: DataclassInstance](
    kinds: Mapping[str, type[T]], table: Mapping[str, object], where: str, **given: object
) -> T:
    """The member of a kind registry that ``table`` names; a check defaults to ``command``."""
    left = dict(table)
    kind = str(left.pop("kind", "command"))
    need(kind in kinds, f"{where}: unknown kind {kind!r}")
    return read(kinds[kind], left, where, **given)


@dataclass(frozen=True)
class Enforcement:
    """A claim that ``level`` runs ``check``.  ``pending``: not wired yet, and the audit says so."""

    level: Level
    check: str
    pending: str | None = None

    def __post_init__(self) -> None:
        need(_is_issue(self.pending), f"pending {self.pending!r} is not an LCD issue")


@dataclass(frozen=True)
class Exemption:
    match: str
    since: date
    reason: str
    issue: str
    expires: date | None = None

    def __post_init__(self) -> None:
        need(_is_issue(self.issue), f"{self.match}: {self.issue!r} is not an LCD issue")
        need(bool(self.reason.strip()), f"{self.match}: an exemption states its reason")


@dataclass(frozen=True)
class Allowlist:
    """A list living in a tool (``module:NAME``); the record holds its size, which only falls."""

    source: str
    ceiling: int


@dataclass(frozen=True)
class Budget:
    name: str
    unit: str
    ceiling: int
    paths: tuple[str, ...] = ()
    absent_ok: bool = False

    def __post_init__(self) -> None:
        need(self.unit in _UNITS, f"budget {self.name}: unknown unit {self.unit!r}")


@dataclass(frozen=True, kw_only=True)
class Rule:
    """One rule.  It is enforced and proven, or says why it cannot be, or names who owes it."""

    id: str
    statement: str
    why: str
    scope: tuple[str, ...]
    doc: str
    enforce: tuple[Enforcement, ...] = ()
    control: tuple[Control, ...] = ()
    advisory: str | None = None
    pending: str | None = None
    gap: str | None = None
    allowlist: Allowlist | None = None
    exempt: tuple[Exemption, ...] = ()

    def __post_init__(self) -> None:
        need(_RULE_ID.fullmatch(self.id) is not None, f"bad rule id {self.id!r}")
        need(self.statement.strip().count("\n") < 2, f"{self.id}: statement over two lines")
        need("\n" not in self.why.strip(), f"{self.id}: why over one line")
        need(bool(self.scope), f"{self.id}: scope is empty")
        need(self.doc in _DOCS, f"{self.id}: unknown doc {self.doc!r}")
        forms = (bool(self.enforce or self.control), self.advisory is not None, bool(self.pending))
        need(sum(forms) == 1, f"{self.id}: needs one of enforce + control, advisory, pending")
        need(self.advisory is None or bool(self.advisory.strip()), f"{self.id}: advisory is blank")
        need(_is_issue(self.pending), f"{self.id}: pending {self.pending!r} is not an LCD issue")
        need(self.gap is None or bool(self.enforce), f"{self.id}: a gap is in an enforcement")
        claimed = {e.check for e in self.enforce}
        proven = {c.check for c in self.control}
        need(claimed <= proven, f"{self.id}: no control proves {sorted(claimed - proven)}")
        need(proven <= claimed, f"{self.id}: controls prove {sorted(proven - claimed)}, unclaimed")

    @property
    def verdict(self) -> Verdict:
        if self.advisory is not None:
            return Verdict.ADVISORY
        if self.pending or all(e.pending for e in self.enforce):
            return Verdict.PENDING
        return Verdict.PARTLY if self.gap else Verdict.ENFORCED


def _is_issue(text: str | None) -> bool:
    return text is None or _ISSUE.fullmatch(text) is not None
