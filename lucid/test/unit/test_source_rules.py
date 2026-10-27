"""H1, H5, H6 and H7 hold across the whole ``lucid/`` tree, tests included.

The runtime package was clean, but the test tree was not. It held 14 mentions
of the reference framework's packages by name and 401 string annotations, and
nothing checked either rule. They surfaced only when closed issues were re-read
(Linear CHA-15, CHA-16), so this does the checking.  After that, tests that
imported the reference framework directly and opened with
``from __future__ import annotations`` landed anyway, because nothing ran the
check before a commit.  The rules now live in ``tools/check_source_rules.py``,
which ``.githooks/pre-commit`` runs on the staged files and this runs on the
whole tree.

* **H1** — no ``from __future__ import annotations``, in ``tools/`` too.
* **H5** — the reference framework is imported and named only in
  :mod:`lucid.test._fixtures.ref_framework`; every other test takes the
  ``ref`` fixture.  The tool reads the name from that module.
* **H6** — the other vendor's GPU API is not named anywhere.
* **H7** — no annotation is a string (PEP 649 makes the quotes unnecessary).
* **dead-path** — no comment, docstring, C++ or Markdown line names a
  ``lucid.`` path that is gone.  The deleted Metal view layer was still
  named as the live owner of Metal views in nine places (LCD-587).
"""

import functools

import pytest

from lucid.test._fixtures import ref_framework
from tools import check_source_rules as rules


@functools.cache
def _tree() -> tuple[rules.Violation, ...]:
    # Unexcused, so the excuses can be held to the hits they let through.
    return tuple(rules.check(rules.tree_sources(), rules.ref_name(), excuses=()))


def _dead_name(violation: rules.Violation) -> str:
    return violation.text.partition(" ")[0]


def _hits(*names: str) -> list[str]:
    return [str(v) for v in _tree() if v.rule in names]


def test_the_tool_reads_the_fixture_s_name() -> None:
    assert rules.ref_name() == ref_framework._REF_NAME


def test_h1_no_future_annotations_import() -> None:
    hits = _hits("future-annotations", "syntax")
    assert not hits, "H1 — drop the import (PEP 649):\n" + "\n".join(hits)


def test_h5_only_the_fixture_imports_the_reference_framework() -> None:
    hits = _hits("ref-import")
    assert not hits, "H5 — take the ref fixture:\n" + "\n".join(hits)


def test_h5_the_reference_framework_is_named_in_one_file() -> None:
    hits = _hits("ref-name")
    assert not hits, "H5 — use the ref fixture or 'reference framework':\n" + "\n".join(hits)


def test_h6_no_other_gpu_api_is_named() -> None:
    hits = _hits("gpu-name")
    assert not hits, "H6 — Lucid's GPU is 'metal':\n" + "\n".join(hits)


def test_h7_no_annotation_is_a_string() -> None:
    hits = _hits("string-annotation")
    assert not hits, "H7 — drop the quotes (PEP 649):\n" + "\n".join(hits)


def test_except_tuples_are_parenthesized() -> None:
    hits = _hits("except-parens")
    assert not hits, "Parenthesize multi-exception clauses:\n" + "\n".join(hits)


def test_no_prose_names_a_dead_lucid_path() -> None:
    hits = [
        str(v)
        for v in _tree()
        if v.rule == "dead-path" and rules.excuse_for(v.path, _dead_name(v)) is None
    ]
    assert not hits, "Name what owns it now, or say the path was deleted:\n" + "\n".join(hits)


@pytest.mark.parametrize("excuse", rules.EXCUSES, ids=lambda e: e.pattern)
def test_every_excuse_is_still_needed(excuse: rules.Excuse) -> None:
    # An excuse nothing needs any more would let the next dead path through.
    used = [
        v
        for v in _tree()
        if v.rule == "dead-path" and rules.excuse_for(v.path, _dead_name(v)) == excuse
    ]
    assert used, f"no dead path needs {excuse.pattern!r} ({excuse.reason}); drop it"


_NAME = ref_framework._REF_NAME
_GPU = "cu" + "da"

# (source, the rule it breaks).  Each is checked as a test file and must be
# reported; the clean twins below must not be.
_BROKEN = [
    ("from __future__ import annotations\n", "future-annotations"),
    (f"import {_NAME}\n", "ref-import"),
    (f"import {_NAME}.nn as nn\n", "ref-import"),
    (f"from {_NAME} import nn\n", "ref-import"),
    (f"import {_NAME}vision\n", "ref-import"),
    (f"mod = pytest.importorskip({_NAME!r})\n", "ref-import"),
    (f"mod = importlib.import_module({_NAME!r})\n", "ref-import"),
    (f"# matches {_NAME.capitalize()}'s rounding\n", "ref-name"),
    (f"x = 1  # the {_GPU.upper()} path\n", "gpu-name"),
    ("def f(x: 'Tensor') -> None: ...\n", "string-annotation"),
    ("try:\n    pass\nexcept ValueError, TypeError:\n    pass\n", "except-parens"),
]
_CLEAN = [
    "from __future__ import division\n",
    "def f(ref: object) -> None:\n    ref.tensor([1.0])\n",
    "def f(mode: Literal['a', 'b']) -> None: ...\n",
    "x: Annotated[int, 'metadata'] = 1\n",
    "# the reference framework rounds to even\n",
    "try:\n    pass\nexcept (ValueError, TypeError):\n    pass\n",
]


@pytest.mark.parametrize("source,rule", _BROKEN, ids=[r for _, r in _BROKEN])
def test_each_rule_reports_its_violation(source: str, rule: str) -> None:
    found = rules.check([("lucid/test/unit/test_x.py", source)], _NAME)
    assert rule in {v.rule for v in found}, found


@pytest.mark.parametrize("source", _CLEAN)
def test_clean_source_is_not_reported(source: str) -> None:
    assert rules.check([("lucid/test/unit/test_x.py", source)], _NAME) == []


def test_the_fixture_may_name_and_import_the_reference_framework() -> None:
    source = f"import {_NAME}\n_REF_NAME = {_NAME!r}\n"
    assert rules.check([(rules.REF_FILE, source)], _NAME) == []


def test_tools_are_held_to_h1_only() -> None:
    # tools/ converts the reference framework's checkpoints, so it may name it.
    future = rules.check([("tools/x.py", "from __future__ import annotations\n")], _NAME)
    assert {v.rule for v in future} == {"future-annotations"}
    assert rules.check([("tools/x.py", f"import {_NAME}  # {_NAME}\n")], _NAME) == []


def test_files_outside_lucid_and_tools_are_not_checked() -> None:
    source = f"from __future__ import annotations\nimport {_NAME}\n"
    assert rules.check([("scripts/x.py", source)], _NAME) == []


# ── dead-path, against a small tree ─────────────────────────────────────────

_MODULES = {
    "lucid/__init__.py": "from lucid.pkg import thing\n__all__ = ['listed']\ndef top(): ...\n",
    "lucid/pkg/__init__.py": "from lucid.pkg._impl import *\n",
    "lucid/pkg/_impl.py": "def thing(): ...\nclass Kind: ...\n",
    "lucid/lazy.py": "def __getattr__(name: str) -> object: ...\n",
    # A lazy module that says what it serves: __all__, or a stub.
    "lucid/listed.py": "__all__ = ['served']\ndef __getattr__(name: str) -> object: ...\n",
    "lucid/stubbed.py": (
        "def plain(): ...\n" "def __getattr__(name: str) -> object:\n    globals()[name] = 1\n"
    ),
    "lucid/stubbed.pyi": "def lazy() -> None: ...\n",
    "lucid/_C/engine.pyi": "class Op: ...\nlinalg: object\n",
    "lucid/_C/core/Tensor.h": "",
}
_INDEX = rules.ModuleIndex(_MODULES, _MODULES.__getitem__)

# (repo-relative path, source).  Each names one dead path in prose.
_DEAD = [
    ("lucid/x.py", "# owned by lucid.pkg._gone\n"),
    ("lucid/x.py", "x = 1  # see lucid.nothing.here\n"),
    ("lucid/x.py", '"""Calls :func:`lucid.pkg.missing`."""\n'),
    ("lucid/x.py", "def f() -> None:\n    '''Via lucid.pkg._impl.Gone.'''\n"),
    ("lucid/x.pyi", "# lucid.pkg.missing\n"),
    ("lucid/_C/x.cpp", "// Bound as lucid._C.Op.\n"),
    ("lucid/_C/x.cpp", 'throw Error("call lucid.pkg.missing first");\n'),
    ("lucid/README.md", "See ``lucid.pkg.Gone``.\n"),
    # A removal word in another sentence says nothing about this path.
    ("lucid/x.py", "# Owned by lucid.pkg._gone.  The old one was deleted.\n"),
    # Nor does one inside a path.
    ("lucid/x.py", "# lucid.pkg.removed is the owner\n"),
    # __all__ or a stub lists what a lazy module serves; nothing else is there.
    ("lucid/x.py", "# lucid.listed.unlisted\n"),
    ("lucid/x.py", "# lucid.stubbed.unlisted\n"),
]
_LIVE = [
    "# lucid.top, lucid.listed and lucid.thing\n",
    "# lucid.pkg.thing and lucid.pkg._impl.Kind.any_attribute_after_it\n",
    "# lucid.pkg.Kind, through the star import\n",
    "# lucid.lazy.anything: the module has __getattr__\n",
    "# lucid._C.engine.Op and lucid._C.engine.linalg, from the stub\n",
    "# lucid._C.core, a directory\n",
    "# lucid.pkg.__file__\n",
    "# lucid.listed.served, lucid.stubbed.lazy and lucid.stubbed.plain\n",
    "# lucid.pkg._gone was deleted in 730ef06c5.\n",
    '"""``lucid.pkg._old`` (since renamed) did this."""\n',
    "# there is no ``lucid.pkg.alias``: one path per op\n",
    'repr = "<lucid.Op seq=1>"  # <lucid.Op seq=1>\n',
    "# the payload's format is 'lucid.pkg.trace/v1'\n",
    "# https://lucid-official.github.io/lucid/lucid.pkg.gone.html\n",
    "x = lucid.pkg.missing  # code, not prose: it fails when it runs\n",
    "# Tensor.type() gives 'lucid.metal.HalfTensor' on Metal\n",
]


@pytest.mark.parametrize("path,source", _DEAD)
def test_a_dead_path_in_prose_is_reported(path: str, source: str) -> None:
    found = rules.check([(path, source)], _NAME, _INDEX)
    assert [v.rule for v in found] == ["dead-path"], found


@pytest.mark.parametrize("source", _LIVE)
def test_a_live_path_history_or_excuse_is_not_reported(source: str) -> None:
    assert rules.check([("lucid/x.py", source)], _NAME, _INDEX) == []


def test_a_dead_path_is_reported_on_its_line() -> None:
    source = (
        "import lucid\n\n\ndef f() -> None:\n" '    """Title.\n\n    Via lucid.pkg.gone.\n    """\n'
    )
    (found,) = rules.check([("lucid/x.py", source)], _NAME, _INDEX)
    assert (found.line, _dead_name(found)) == (7, "lucid.pkg.gone")


def test_an_excuse_can_be_held_to_one_file() -> None:
    excuse = rules.Excuse(r"lucid\.pkg\.gone", "pending: held by another branch", "lucid/x.py")
    source = "# lucid.pkg.gone\n"
    assert rules.check([("lucid/x.py", source)], _NAME, _INDEX, (excuse,)) == []
    assert rules.check([("lucid/y.py", source)], _NAME, _INDEX, (excuse,)) != []


def test_tools_prose_is_not_held_to_lucid_paths() -> None:
    assert rules.check([("tools/x.py", "# lucid.pkg.gone\n")], _NAME, _INDEX) == []
