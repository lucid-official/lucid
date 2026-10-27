"""No ``Tensor`` member ``tensor.py`` defines is overwritten by an injector (LCD-610).

``tensor.py`` defined ``index_select`` with a reshape-based body that never
ran: ``_methods._inject_methods`` replaced it with the registry method at
import, and six more (``clone``, ``detach``, ``contiguous``,
``scatter_add``, ``masked_select``, ``bmm``) were dead the same way — code
that read as the behaviour and was not.  Each injector here runs against a
blank class, so what it attaches is read, not restated.
"""

import ast
from collections.abc import Callable
from pathlib import Path

import lucid
from lucid._tensor import _dunders, _free_fn_methods, _methods, _to

_SOURCE = Path(lucid.__file__).parent / "_tensor" / "tensor.py"

_INJECTORS: tuple[Callable[[type], None], ...] = (
    _methods._inject_methods,
    _dunders._inject_dunders,
    _to._inject_to,
    _free_fn_methods.inject_free_fn_methods,
)

#: ``tensor.py`` members an injector may replace on purpose.
_REPLACED_ON_PURPOSE: frozenset[str] = frozenset()


def _class_members() -> dict[str, int]:
    tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Tensor")
    members: dict[str, int] = {}
    for node in cls.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            members.setdefault(node.name, node.lineno)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    members.setdefault(target.id, node.lineno)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            members.setdefault(node.target.id, node.lineno)
    return members


def _attached_by(inject: Callable[[type], None]) -> set[str]:
    blank = type("Blank", (), {})
    before = set(vars(blank))
    inject(blank)
    return set(vars(blank)) - before


def _injected() -> set[str]:
    return set().union(*(_attached_by(inject) for inject in _INJECTORS))


def test_no_two_injectors_attach_one_name() -> None:
    # They run one after another, so a name two of them set is the later
    # one's, and the earlier body is dead the same way tensor.py's were.
    sets = {inject.__qualname__: _attached_by(inject) for inject in _INJECTORS}
    overlaps = sorted(
        f"{a} & {b}: {sorted(sets[a] & sets[b])}"
        for i, a in enumerate(sets)
        for b in list(sets)[i + 1 :]
        if sets[a] & sets[b]
    )
    assert overlaps == []


def test_no_injector_overwrites_a_member_tensor_py_defines() -> None:
    members = _class_members()
    shadowed = sorted(
        f"{name} (tensor.py:{members[name]})"
        for name in _injected() & members.keys()
        if name not in _REPLACED_ON_PURPOSE
    )
    assert shadowed == []


def test_the_injectors_are_read_not_restated() -> None:
    injected = _injected()
    for name in (
        "index_select",
        "clone",
        "eval",
        "masked_fill_",
        "__add__",
        "to",
        "expm1",
    ):
        assert name in injected, name
    assert sorted(_REPLACED_ON_PURPOSE - _class_members().keys()) == []


def test_an_injector_run_on_another_class_leaves_tensor_alone() -> None:
    # The shadowing checks above run every injector on a blank class.  An
    # injector that stashed its argument in a module global made a later
    # ``x.to(other)`` refuse a real tensor ("takes a device, a dtype or a
    # tensor first, got Tensor"), but only when this file ran first.
    import lucid

    for inject in _INJECTORS:
        _attached_by(inject)
    x = lucid.tensor([1.0, 2.0])
    other = lucid.tensor([1.0], dtype=lucid.float64)
    assert x.to(other).dtype == lucid.float64
