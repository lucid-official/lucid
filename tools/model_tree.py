"""Hold ``lucid/models/MODELS.tree`` to the model registry.

The tree is written by hand, because its groups are a judgement the code does not
make, so nothing but a check can keep it true.  This reads the tree's structure back
and compares it with what is registered: every family once, each family's count, each
variant named on its lines, and every group's total.
"""

import dataclasses
import re
from collections.abc import Iterable, Mapping
from pathlib import Path

TREE_PATH = Path(__file__).resolve().parents[1] / "lucid" / "models" / "MODELS.tree"

_INDENT = 3
_BRANCH = re.compile(r"^((?:[│ ]  )*)[├└]─ (.+)$")
_CONTINUATION = re.compile(r"^(?:[│ ]\s*)+\S")
_FAMILY = re.compile(r"\[(\w+)\] \((\d+)\)\s*(.*)$")
_GROUP = re.compile(r"\((\d+)\)\s*$")
_SECTION = re.compile(r"— (\d+) famil(?:y|ies) · (\d+) models?\s*$")
_NOTE = re.compile(r"\([^)]*\)")
_SEPARATORS = re.compile(r"\s+|·|~|×|\+")


@dataclasses.dataclass
class _Node:
    """One line of the tree: a section, a group, or a family with its variant text."""

    label: str
    depth: int
    models: int
    families: int | None = None
    family: str | None = None
    variants: str = ""
    children: list["_Node"] = dataclasses.field(default_factory=list)


def _parse_node(label: str, depth: int) -> _Node:
    if found := _FAMILY.search(label):
        return _Node(label, depth, int(found[2]), 1, found[1], found[3])
    if found := _SECTION.search(label):
        return _Node(label, depth, int(found[2]), int(found[1]))
    if found := _GROUP.search(label):
        return _Node(label, depth, int(found[1]))
    raise ValueError(f"MODELS.tree: cannot read {label!r}")


def parse(text: str) -> _Node:
    """The tree in ``text``, comments dropped, as nested nodes under its root line."""
    lines = [line for line in text.splitlines() if line.strip() and not line.startswith("#")]
    root = _parse_node(lines[0], -1)
    path = [root]
    last: _Node = root
    for line in lines[1:]:
        if branch := _BRANCH.match(line):
            depth = len(branch[1]) // _INDENT
            last = _parse_node(branch[2], depth)
            del path[depth + 1 :]
            path[-1].children.append(last)
            path.append(last)
        elif _CONTINUATION.match(line) and last.family:
            last.variants += " " + line.lstrip("│ ")
        else:
            raise ValueError(f"MODELS.tree: cannot read {line!r}")
    return root


def _walk(node: _Node) -> Iterable[_Node]:
    yield node
    for child in node.children:
        yield from _walk(child)


def _variant_tokens(text: str) -> list[str]:
    return [token for token in _SEPARATORS.split(_NOTE.sub(" ", text)) if token]


def _check_totals(node: _Node, problems: list[str]) -> tuple[int, int]:
    """Check each group against its children; returns ``(models, families)`` beneath it."""
    if node.family:
        return node.models, 1
    models = families = 0
    for child in node.children:
        child_models, child_families = _check_totals(child, problems)
        models += child_models
        families += child_families
    if models != node.models:
        problems.append(f"{node.label!r}: says {node.models} models, its children total {models}")
    if node.families is not None and families != node.families:
        problems.append(
            f"{node.label!r}: says {node.families} families, its children total {families}"
        )
    return models, families


def check(text: str, registered: Mapping[str, Iterable[str]]) -> list[str]:
    """What ``text`` gets wrong about ``registered`` (family directory to factory names)."""
    root = parse(text)
    problems: list[str] = []
    _check_totals(root, problems)
    names = {family: list(models) for family, models in registered.items()}
    seen: set[str] = set()
    for node in _walk(root):
        if node.family is None:
            continue
        if node.family in seen:
            problems.append(f"family {node.family!r} is listed twice")
        seen.add(node.family)
        models = names.get(node.family)
        if models is None:
            problems.append(f"family {node.family!r} is listed but registers nothing")
            continue
        if node.models != len(models):
            problems.append(
                f"family {node.family!r}: says {node.models} models, {len(models)} are registered"
            )
        for token in _variant_tokens(node.variants):
            if not any(token in name for name in models):
                problems.append(f"family {node.family!r}: no factory matches {token!r}")
    problems.extend(
        f"family {family!r} is registered but not listed" for family in sorted(names.keys() - seen)
    )
    return problems


def registered_by_family() -> dict[str, list[str]]:
    """Every registered factory name, grouped by the family directory that defines it."""
    import lucid.models  # noqa: F401 — registers every factory
    from lucid.models._registry import _REGISTRY

    grouped: dict[str, list[str]] = {}
    for name, entry in _REGISTRY.items():
        grouped.setdefault(entry.factory.__module__.split(".")[3], []).append(name)
    return grouped


def check_tree() -> list[str]:
    """What the committed tree gets wrong about the live registry."""
    return check(TREE_PATH.read_text(encoding="utf-8"), registered_by_family())
