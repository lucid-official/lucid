"""Every public name ``lucid`` binds is published (LCD-613, LCD-641).

``lucid.eval`` was a public ``def`` that ``lucid.metal`` documents and the
test suite calls, but neither ``__all__`` nor ``__init__.pyi`` listed it:
a type checker refused ``lucid.eval(x)``, and the docs and the released
surface did not know it.  Then 39 more — ``gcd``, ``put``, ``amax``,
``histogram`` … — callable and stubbed, but bound by the lazy loaders
rather than by a ``def`` here, so the first guard missed them, and every
reader of ``__all__`` (``from lucid import *``, the audit's surface walk,
the docs) skipped them.  A public name the package binds is in
``__all__``, or is listed below with the reason it is not.
"""

import ast
from pathlib import Path
from types import ModuleType

import pytest

import lucid

_ROOT = Path(lucid.__file__).parent

#: Public top-level functions of ``lucid/__init__.py`` that are not API.
_UNPUBLISHED: frozenset[str] = frozenset()


def _defs(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }


def test_every_public_toplevel_def_is_in_all_and_the_stub() -> None:
    public = {n for n in _defs(_ROOT / "__init__.py") if not n.startswith("_")}
    stubbed = _defs(_ROOT / "__init__.pyi")
    unlisted = sorted(n for n in public - _UNPUBLISHED if n not in lucid.__all__)
    unstubbed = sorted(n for n in public - _UNPUBLISHED if n not in stubbed)
    assert (unlisted, unstubbed) == ([], [])
    assert sorted(_UNPUBLISHED - public) == []


#: Public names ``lucid`` binds that ``__all__`` leaves out, and why.
_NOT_IN_ALL: dict[str, str] = {
    "bool": "the dtype alias of bool_: `from lucid import *` would shadow the builtin",
    "float": "the dtype alias of float32: `import *` would shadow the builtin",
    "int": "the dtype alias of int32: `import *` would shadow the builtin",
}


def _bound_public_names() -> set[str]:
    """What ``lucid`` binds, the lazily loaded names included.

    A lazy loader binds its names on first access, so ``dir(lucid)`` shows
    only those something has asked for already.  Submodules are not names
    this list governs: ``__all__`` lists the ones the package presents.
    """
    for name in lucid._GROUP_LOADERS:
        getattr(lucid, name)
    return {
        n
        for n in dir(lucid)
        if not n.startswith("_") and not isinstance(getattr(lucid, n), ModuleType)
    }


def test_every_public_name_lucid_binds_is_in_all() -> None:
    unlisted = _bound_public_names() - set(lucid.__all__)
    assert sorted(unlisted - set(_NOT_IN_ALL)) == []
    # The list holds only names that are there to be excused.
    assert sorted(set(_NOT_IN_ALL) - unlisted) == []


def test_every_name_in_all_resolves() -> None:
    assert [n for n in lucid.__all__ if not hasattr(lucid, n)] == []
    assert len(lucid.__all__) == len(set(lucid.__all__))


def test_eval_evaluates_the_tensors_it_is_given(device: str) -> None:
    a = lucid.ones(3, device=device) * 2
    b = a + 1
    assert lucid.eval(a, b) is None
    assert lucid.eval() is None
    assert b.tolist() == [3.0, 3.0, 3.0]


@pytest.mark.parametrize("bad", [None, [lucid.ones(2)], 3.0])
def test_eval_refuses_what_is_not_a_tensor(bad: object) -> None:
    with pytest.raises(TypeError, match="lucid.eval"):
        lucid.eval(lucid.ones(2), bad)
