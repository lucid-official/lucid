"""Every public function ``lucid/__init__.py`` defines is published (LCD-613).

``lucid.eval`` was a public ``def`` that ``lucid.metal`` documents and the
test suite calls, but neither ``__all__`` nor ``__init__.pyi`` listed it:
a type checker refused ``lucid.eval(x)``, and the docs and the released
surface did not know it.  A public name the package defines is in both, or
is listed below as deliberately unpublished.
"""

import ast
from pathlib import Path

import pytest

import lucid

_ROOT = Path(lucid.__file__).parent

#: Public top-level functions of ``lucid/__init__.py`` that are not API.
_UNPUBLISHED: frozenset[str] = frozenset()


def _defs(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }


def test_every_public_toplevel_def_is_in_all_and_the_stub() -> None:
    public = {n for n in _defs(_ROOT / "__init__.py") if not n.startswith("_")}
    stubbed = _defs(_ROOT / "__init__.pyi")
    unlisted = sorted(n for n in public - _UNPUBLISHED if n not in lucid.__all__)
    unstubbed = sorted(n for n in public - _UNPUBLISHED if n not in stubbed)
    assert (unlisted, unstubbed) == ([], [])
    assert sorted(_UNPUBLISHED - public) == []


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
