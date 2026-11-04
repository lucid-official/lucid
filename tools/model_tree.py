"""Draw ``lucid/models/MODELS.tree`` from the model registry, and hold it to it.

The file is the whole zoo in the Unix ``tree -F`` format, every factory by its full name,
and nothing in it is written by hand::

    lucid/models/
    └── <domain>/       lucid.models.<domain>, the module the family lives in
        └── <task>/     register_model(task=...), as list_models(task=...) takes it
            └── <family>/     lucid.models.<domain>.<family>, as list_models(family=...) takes it
                └── <factory>     the name create_model takes

Every factory has one line.  A task-specific factory goes under its own task, so
``list_models(task=t, family=f)`` is what the tree shows there.  A family's ``base``
factories, its models with no task head, would otherwise repeat the family under a
``base`` branch beside the same family's task branch; they go under the family's primary
task instead, the non-base task it registers most factories for, with ties going to the
alphabetically first.  A family with no task-specific factory stays under ``base``.  A
family therefore appears under more than one task only when it really has factories for
several, and no factory ever appears twice.

    python -m tools.model_tree           # report what the file gets wrong
    python -m tools.model_tree --write   # draw it again
"""

import argparse
import collections
import dataclasses
import re
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

TREE_PATH = Path(__file__).resolve().parents[1] / "lucid" / "models" / "MODELS.tree"

_ROOT = "lucid/models/"
_BASE = "base"
_INDENT = 4
_LINE = re.compile(r"^((?:│   |    )*)(?:├── |└── )(.+)$")
_DIGITS = re.compile(r"(\d+)")
_FACTORY_DEPTH = 3


@dataclasses.dataclass(frozen=True, order=True)
class Entry:
    """One registered factory, as the registry and the module path describe it."""

    domain: str
    family: str
    task: str
    name: str


def registered() -> list[Entry]:
    """Every factory in the live registry."""
    import lucid.models  # noqa: F401 — registers every factory
    from lucid.models._registry import _REGISTRY

    entries = []
    for name, entry in _REGISTRY.items():
        domain = entry.factory.__module__.split(".")[2]
        entries.append(Entry(domain, entry.family, entry.task, name))
    return entries


def _primary_tasks(entries: Iterable[Entry]) -> dict[str, str]:
    """Each family's most-registered non-base task, the alphabetically first on a tie."""
    counts: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for entry in entries:
        if entry.task != _BASE:
            counts[entry.family][entry.task] += 1
    return {
        family: min(tasks, key=lambda task: (-tasks[task], task))
        for family, tasks in counts.items()
    }


def _placed(entries: Iterable[Entry]) -> list[Entry]:
    """``entries`` with each base factory moved to its family's primary task."""
    entries = list(entries)
    primary = _primary_tasks(entries)
    return [
        (
            dataclasses.replace(entry, task=primary.get(entry.family, _BASE))
            if entry.task == _BASE
            else entry
        )
        for entry in entries
    ]


def _natural(name: str) -> list[str | int]:
    """Sort order that puts ``resnet_50`` before ``resnet_101``."""
    return [int(part) if part.isdigit() else part for part in _DIGITS.split(name)]


def _nested(entries: Iterable[Entry]) -> dict[str, dict[str, dict[str, list[str]]]]:
    tree: dict[str, dict[str, dict[str, list[str]]]] = {}
    for entry in _placed(entries):
        family = tree.setdefault(entry.domain, {}).setdefault(entry.task, {})
        family.setdefault(entry.family, []).append(entry.name)
    return tree


def _branches(items: list[tuple[str, object]], prefix: str) -> Iterator[str]:
    for index, (label, below) in enumerate(items):
        last = index == len(items) - 1
        yield f"{prefix}{'└── ' if last else '├── '}{label}"
        yield from _below(below, prefix + ("    " if last else "│   "))


def _below(node: object, prefix: str) -> Iterator[str]:
    if isinstance(node, dict):
        yield from _branches([(f"{key}/", node[key]) for key in sorted(node)], prefix)
    elif isinstance(node, list):
        yield from _branches([(name, None) for name in sorted(node, key=_natural)], prefix)


def draw(entries: Iterable[Entry]) -> str:
    """The tree, closing with the count line ``tree`` itself ends on."""
    entries = list(entries)
    placed = _placed(entries)
    summary = (
        f"{len({e.domain for e in entries})} domains, {len({e.task for e in placed})} tasks, "
        f"{len({e.family for e in entries})} families, {len(entries)} models"
    )
    return "\n".join([_ROOT, *_below(_nested(entries), ""), "", summary, ""])


def parse(text: str) -> list[Entry]:
    """The factories drawn in ``text``, each with the path it is drawn under."""
    lines = text.splitlines()
    if not lines or lines[0] != _ROOT:
        raise ValueError(f"MODELS.tree: expected {_ROOT!r} on the first line")
    path: list[str] = []
    drawn = []
    for line in lines[1:]:
        if not line.strip():
            break
        found = _LINE.match(line)
        if found is None:
            raise ValueError(f"MODELS.tree: cannot read {line!r}")
        depth = len(found[1]) // _INDENT
        del path[depth:]
        path.append(found[2].removesuffix("/"))
        if depth == _FACTORY_DEPTH:
            drawn.append(Entry(path[0], path[2], path[1], path[3]))
    return drawn


def check(text: str, entries: Iterable[Entry]) -> list[str]:
    """What ``text`` gets wrong about ``entries``."""
    entries = list(entries)
    drawn = parse(text)
    want = set(_placed(entries))
    problems = [
        f"family {e.family!r}: {e.name!r} is registered as {e.domain}/{e.task} but not drawn there"
        for e in sorted(want - set(drawn))
    ]
    problems += [
        f"family {e.family!r}: {e.name!r} is drawn under {e.domain}/{e.task} but not registered there"
        for e in sorted(set(drawn) - want)
    ]
    names = collections.Counter(e.name for e in drawn)
    problems += [f"{name!r} is drawn {n} times" for name, n in sorted(names.items()) if n > 1]
    if not problems and text != draw(entries):
        problems.append("the drawing is out of order: run python -m tools.model_tree --write")
    return problems


def check_tree() -> list[str]:
    """What the committed tree gets wrong about the live registry."""
    return check(TREE_PATH.read_text(encoding="utf-8"), registered())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.partition("\n")[0])
    parser.add_argument("--write", action="store_true", help="draw the file again")
    args = parser.parse_args()
    if args.write:
        TREE_PATH.write_text(draw(registered()), encoding="utf-8")
    problems = check_tree()
    for problem in problems:
        print(problem)
    print(f"{TREE_PATH.name}: {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
