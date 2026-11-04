"""A pytest plugin that picks a family's tests by its model names, not by substring.

The zoo-wide test files are shared by every family, so a family's slice of them
used to be ``-k <directory name>``.  That is a substring match on the node id,
and it is wrong both ways at once:

* it **over-selects**: ``-k senet`` also selects every ``densenet`` test (61 of
  the 76 it picked) — work done twice and credited to the wrong family;
* it **under-selects**: a family's factories need not contain its directory's
  name.  ``senet``'s are ``se_resnet_*``, ``sknet``'s ``sk_resnet_*``, ``cspnet``'s
  ``csp_*`` — 45, 48 and 27 tests that no family's slice ever ran.

So a family is selected by the names its tests are written with, and the match
is word-safe: a name matches only where it is not part of a longer name.

Tests are written three ways, and each is matched the way it is spelled:

* **a factory name or label in a parameter id** (``[se_resnet_50]``): found as a
  whole word, where ``_`` is part of the word — ``resnet`` is not inside
  ``se_resnet_50``;
* **a model class in a parameter id** (``[AlexNetForImageClassification]``): the
  registry's model and config class names, found the same way;
* **the family or a class in a test's own name** (``TestGPT2Cache``,
  ``test_alexnet_forward``): split into words on ``_`` and on CamelCase, and a
  name matches where a run of whole words spells it — ``Alex`` + ``Net`` is
  ``alexnet``, ``Dense`` + ``Net`` is ``densenet`` and is not ``senet``.

Loaded with ``-p lucid.test.audit._zoo_select``; a run with none of the options
is unaffected.
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
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_NOT_ALNUM = re.compile(r"[^A-Za-z0-9]+")
_IDENTIFIER = re.compile(rf"[{_WORD}]+")


def _alternation(names: Iterable[str]) -> str:
    return "|".join(re.escape(n) for n in sorted({n for n in names if n}, key=len, reverse=True))


def _words(text: str) -> list[str]:
    """``TestGPT2Cache::test_alex_net`` as ``test gpt2 cache test alex net``."""
    return [w.lower() for w in _NOT_ALNUM.split(_CAMEL.sub("_", text)) if w]


def _spelled(name: str) -> str:
    return name.replace("_", "").lower()


class FamilySelector:
    """Which test node ids belong to one family.

    ``names`` are its factory names, ``labels`` its directory and registry
    labels, ``classes`` the registry's model and config class names.
    """

    def __init__(
        self, names: Iterable[str], labels: Iterable[str], classes: Iterable[str] = ()
    ) -> None:
        names, labels, classes = list(names), list(labels), list(classes)
        exact, label = _alternation([*names, *classes]), _alternation(labels)
        alternatives = [f"(?:{exact})"] if exact else []
        # A label also names a test pytest numbered because two params looked alike
        # (``bert0``, ``gpt2_0``); a factory name never carries such a suffix.
        alternatives += [f"(?:{label})(?:_?[0-9]+)?"] if label else []
        self._param = (
            re.compile(rf"(?<![{_WORD}])(?:{'|'.join(alternatives)})(?![{_WORD}])", re.IGNORECASE)
            if alternatives
            else None
        )
        self._spelled = {_spelled(n) for n in (*names, *labels, *classes) if n}
        self._longest = max((len(s) for s in self._spelled), default=0)

    def _spells(self, text: str) -> bool:
        """Whether a run of whole words in ``text`` spells a name, a label or a class."""
        words = _words(text)
        for start in range(len(words)):
            run = ""
            for word in words[start:]:
                run += word
                if len(run) > self._longest:
                    break
                if run in self._spelled:
                    return True
        return False

    def matches(self, nodeid: str) -> bool:
        head, bracket, params = nodeid.partition("[")
        if self._spells(head):
            return True
        if not bracket:
            return False
        if self._param is not None and self._param.search(params):
            return True
        # A CamelCase identifier in a parameter id is a class (``GPT2DoubleHeadsModel``),
        # which may be a family's without the registry knowing it: spell it by words.
        return any(
            self._spells(token)
            for token in _IDENTIFIER.findall(params)
            if token != token.lower() and token != token.upper()
        )


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("zoo")
    group.addoption("--zoo-names", default="", help="comma-separated factory names of a family")
    group.addoption("--zoo-labels", default="", help="comma-separated labels of a family")
    group.addoption("--zoo-classes", default="", help="comma-separated class names of a family")


def _listed(config: pytest.Config, option: str) -> list[str]:
    return [n for n in str(config.getoption(option)).split(",") if n]


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    names = _listed(config, "--zoo-names")
    labels = _listed(config, "--zoo-labels")
    classes = _listed(config, "--zoo-classes")
    if not (names or labels or classes):
        return
    selector = FamilySelector(names, labels, classes)
    kept: list[pytest.Item] = []
    dropped: list[pytest.Item] = []
    for item in items:
        (kept if selector.matches(item.nodeid) else dropped).append(item)
    if dropped:
        config.hook.pytest_deselected(items=dropped)
        items[:] = kept
