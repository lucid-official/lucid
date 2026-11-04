"""A pytest plugin that picks a family's tests by its model names, not by substring.

The zoo-wide test files are shared by every family, so a family's slice of them
used to be ``-k <directory name>``.  That is a substring match on the node id,
and it is wrong both ways at once:

* it **over-selects**: ``-k senet`` also selects every ``densenet`` test (61 of
  the 76 it picked) — work done twice and credited to the wrong family;
* it **under-selects**: a family's factories need not contain its directory's
  name.  ``senet``'s are ``se_resnet_*``, ``sknet``'s ``sk_resnet_*``, ``cspnet``'s
  ``csp_*`` — 45, 48 and 27 tests that no family's slice ever ran.

So a family is selected by the names its factories are registered under, and
the match is word-safe: a name matches only where it is not part of a longer
name.  ``resnet`` is not found inside ``se_resnet_50`` — ``_`` counts as part of
a word, as it does in an identifier.

Loaded with ``-p lucid.test.audit._zoo_select``; a run with neither option is
unaffected.
"""

import re
from typing import TYPE_CHECKING

# Imported for real, not under TYPE_CHECKING: pytest reads the hooks' annotations
# when it registers this module, and an annotation naming something that is not
# imported is a NameError on Python 3.14's lazy annotations.
import pytest

if TYPE_CHECKING:
    from collections.abc import Iterable

_WORD = r"A-Za-z0-9_"


def _alternation(names: Iterable[str], word: str) -> re.Pattern[str] | None:
    """``names`` as a pattern that only matches where none of them is inside a longer word."""
    found = sorted({name for name in names if name}, key=len, reverse=True)
    if not found:
        return None
    body = "|".join(re.escape(name) for name in found)
    return re.compile(rf"(?<![{word}])(?:{body})(?![{word}])", re.IGNORECASE)


class FamilySelector:
    """Which test node ids belong to one family.

    ``names`` are the family's factory names; they are matched in a test's
    parameter id, where a factory is spelled out, and ``_`` is part of a name
    there.  ``labels`` are the family's directory and registry labels; they are
    matched there too and also in the test's file and function name, where
    ``_`` separates words (``test_alexnet_forward`` belongs to ``alexnet``).
    """

    def __init__(self, names: Iterable[str], labels: Iterable[str]) -> None:
        everything = [*names, *labels]
        self._param = _alternation(everything, _WORD)
        self._head = _alternation(labels, "A-Za-z0-9")

    def matches(self, nodeid: str) -> bool:
        head, bracket, params = nodeid.partition("[")
        if self._head is not None and self._head.search(head):
            return True
        return bool(bracket and self._param is not None and self._param.search(params))


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("zoo")
    group.addoption("--zoo-names", default="", help="comma-separated factory names of a family")
    group.addoption("--zoo-labels", default="", help="comma-separated labels of a family")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    names = [n for n in str(config.getoption("--zoo-names")).split(",") if n]
    labels = [n for n in str(config.getoption("--zoo-labels")).split(",") if n]
    if not (names or labels):
        return
    selector = FamilySelector(names, labels)
    kept: list[pytest.Item] = []
    dropped: list[pytest.Item] = []
    for item in items:
        (kept if selector.matches(item.nodeid) else dropped).append(item)
    if dropped:
        config.hook.pytest_deselected(items=dropped)
        items[:] = kept
