"""Keep ``lucid/models/MODELS.tree`` exactly true to the model registry.

The file is drawn the way the Unix ``tree -F`` command draws a directory, down to
every registered factory by its full name.  A lowercase name ending in ``/`` is a
directory under ``lucid/models/``, a capitalized one is a group chosen for reading
(the code has no such level), and a name with no ``/`` is a factory.

Only the groups are written by hand; the registry decides everything under a family.

    python -m tools.model_tree           # report what the file gets wrong
    python -m tools.model_tree --write   # redraw the factories from the registry

``--write`` cannot place a new family, since that is a judgement: add its ``name/`` line
under the right group and run it again.
"""

import argparse
import dataclasses
import re
import sys
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

TREE_PATH = Path(__file__).resolve().parents[1] / "lucid" / "models" / "MODELS.tree"

_INDENT = 4
_LINE = re.compile(r"^((?:│   |    )*)(?:├── |└── )(.+)$")
_DIGITS = re.compile(r"(\d+)")


@dataclasses.dataclass
class Node:
    """One name in the tree and what is drawn under it."""

    name: str
    children: list["Node"] = dataclasses.field(default_factory=list)

    @property
    def is_factory(self) -> bool:
        return not self.name.endswith("/")

    @property
    def is_family(self) -> bool:
        return not self.is_factory and all(child.is_factory for child in self.children)

    @property
    def directory(self) -> str:
        return self.name.removesuffix("/")


def parse(text: str) -> Node:
    """The tree in ``text``, up to the summary line that closes it."""
    lines = text.splitlines()
    root = Node(lines[0])
    path = [root]
    for line in lines[1:]:
        if not line.strip():
            break
        found = _LINE.match(line)
        if found is None:
            raise ValueError(f"MODELS.tree: cannot read {line!r}")
        node = Node(found[2])
        del path[len(found[1]) // _INDENT + 1 :]
        path[-1].children.append(node)
        path.append(node)
    return root


def _families(root: Node) -> Iterator[tuple[str, Node]]:
    """Every family, with the domain it is drawn under."""

    def walk(node: Node, domain: str) -> Iterator[tuple[str, Node]]:
        for child in node.children:
            if child.is_family:
                yield domain, child
            elif not child.is_factory:
                yield from walk(child, domain)

    for domain in root.children:
        yield from walk(domain, domain.directory)


def _draw(node: Node, prefix: str = "") -> Iterator[str]:
    for index, child in enumerate(node.children):
        last = index == len(node.children) - 1
        yield f"{prefix}{'└── ' if last else '├── '}{child.name}"
        yield from _draw(child, prefix + ("    " if last else "│   "))


def render(root: Node) -> str:
    """Draw ``root``, closing with the count line ``tree`` itself ends on."""
    families = list(_families(root))
    models = sum(len(family.children) for _, family in families)
    summary = f"{len(root.children)} domains, {len(families)} families, {models} models"
    return "\n".join([root.name, *_draw(root), "", summary, ""])


def _natural(name: str) -> list[str | int]:
    """Sort order that puts ``resnet_50`` before ``resnet_101``."""
    return [int(part) if part.isdigit() else part for part in _DIGITS.split(name)]


def redrawn(root: Node, registered: Mapping[str, Iterable[str]]) -> Node:
    """``root`` with each family's factories replaced by what is registered, in order."""
    for _, family in _families(root):
        names = sorted(registered.get(family.directory, ()), key=_natural)
        family.children = [Node(name) for name in names]
    return root


def _check_family(family: Node, registered: Mapping[str, Iterable[str]]) -> Iterator[str]:
    label = family.directory
    if label not in registered:
        yield f"family {label!r} is listed but registers nothing"
        return
    drawn = [child.name for child in family.children]
    names = set(registered[label])
    for name in sorted(names - set(drawn)):
        yield f"family {label!r}: {name!r} is registered but not listed"
    for name in sorted(set(drawn) - names):
        yield f"family {label!r}: {name!r} is listed but not registered"
    for name in sorted({name for name in drawn if drawn.count(name) > 1}):
        yield f"family {label!r}: {name!r} is listed twice"


def check(
    text: str, registered: Mapping[str, Iterable[str]], domains: Mapping[str, str] | None = None
) -> list[str]:
    """What ``text`` gets wrong about ``registered`` (family directory to factory names)."""
    root = parse(text)
    problems: list[str] = []
    seen: list[str] = []
    for domain, family in _families(root):
        problems.extend(_check_family(family, registered))
        if family.directory in seen:
            problems.append(f"family {family.directory!r} is listed twice")
        seen.append(family.directory)
        if domains and domains.get(family.directory, domain) != domain:
            problems.append(
                f"family {family.directory!r} is under {domain!r}, "
                f"but lives in {domains[family.directory]!r}"
            )
    problems.extend(
        f"family {family!r} is registered but not listed"
        for family in sorted(set(registered) - set(seen))
    )
    if not problems and text != render(redrawn(parse(text), registered)):
        problems.append("the drawing is not canonical: run python -m tools.model_tree --write")
    return problems


def _registry() -> tuple[dict[str, list[str]], dict[str, str]]:
    import lucid.models  # noqa: F401 — registers every factory
    from lucid.models._registry import _REGISTRY

    names: dict[str, list[str]] = {}
    domains: dict[str, str] = {}
    for name, entry in _REGISTRY.items():
        parts = entry.factory.__module__.split(".")
        names.setdefault(parts[3], []).append(name)
        domains[parts[3]] = parts[2]
    return names, domains


def registered_by_family() -> dict[str, list[str]]:
    """Every registered factory name, grouped by the family directory that defines it."""
    return _registry()[0]


def check_tree() -> list[str]:
    """What the committed tree gets wrong about the live registry."""
    names, domains = _registry()
    return check(TREE_PATH.read_text(encoding="utf-8"), names, domains)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.partition("\n")[0])
    parser.add_argument("--write", action="store_true", help="redraw the factories")
    args = parser.parse_args()
    if args.write:
        names, _ = _registry()
        root = parse(TREE_PATH.read_text(encoding="utf-8"))
        listed = {family.directory for _, family in _families(root)}
        for family in sorted(set(names) - listed):
            print(f"family {family!r} is not in the tree: add '{family}/' under its group")
        TREE_PATH.write_text(render(redrawn(root, names)), encoding="utf-8")
    problems = check_tree()
    for problem in problems:
        print(problem)
    print(f"{TREE_PATH.name}: {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
