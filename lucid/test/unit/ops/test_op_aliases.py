"""An alias takes exactly the calls its target takes (LCD-1052).

``lucid.multiply(2.0, t)`` raised ``TypeError`` while ``lucid.mul(2.0, t)`` took
it, and so did ``divide``, ``true_divide``, ``subtract`` and ``arctan2``: each
alias was a function written out again, annotated ``input: Tensor``, and the
tensor-argument guard enforces what a function's annotations say.

Every alias is declared once, in :data:`lucid._ops._aliases.ALIASES`, and an
alias of a registry op is built from its target's entry (so it cannot differ).
Held here, for every row of that table:

* its signature is its target's — parameter names, kinds, defaults and
  annotations — less the keywords it fixes; a row that differs on purpose
  (the reference spells ``swapaxes``'s axes ``axis0`` / ``axis1``) says why and
  pins the parameters it has;
* the stubs say the same, for the free function and for the ``Tensor`` method
  and its in-place twin;
* a derived alias runs its target's adapter, and takes and refuses the same
  operands on either device — a number first, ``None``, a list or a tuple;
* a hand-written function that only forwards to another public op is a row of
  the table, so a new alias cannot skip the guard.
"""

import ast
import inspect
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

import lucid
from lucid._ops._aliases import ALIAS_NAMES, ALIASES, OpAlias
from lucid._ops._registry import _REGISTRY, OpEntry

_ROOT = Path(lucid.__file__).parent
_FREE_STUB = _ROOT / "__init__.pyi"
_METHOD_STUB = _ROOT / "_tensor" / "tensor.pyi"

_DERIVED = [a for a in ALIASES if a.derived]
_DECLARED = [a for a in ALIASES if a.differs]
_EQUAL = [a for a in ALIASES if not a.differs]


def _ids(aliases: Sequence[OpAlias]) -> list[str]:
    return [f"{a.name}->{a.target}" for a in aliases]


def _entry(free_fn_name: str) -> OpEntry:
    return next(e for e in _REGISTRY if e.free_fn_name == free_fn_name)


# ── the table ───────────────────────────────────────────────────────────────


def test_the_table_names_each_alias_once() -> None:
    names = [a.name for a in ALIASES]
    assert len(names) == len(set(names))
    assert not {a.target for a in ALIASES} & set(names), "a target is itself an alias"


@pytest.mark.parametrize("alias", ALIASES, ids=_ids(ALIASES))
def test_a_row_names_two_public_ops(alias: OpAlias) -> None:
    assert alias.name in lucid.__all__
    assert alias.target in lucid.__all__
    assert callable(getattr(lucid, alias.name))
    assert callable(getattr(lucid, alias.target))


@pytest.mark.parametrize("alias", ALIASES, ids=_ids(ALIASES))
def test_a_difference_is_declared_with_a_reason_and_its_parameters(alias: OpAlias) -> None:
    assert bool(alias.differs) == bool(alias.params)
    assert not (alias.derived and alias.differs), "a derived alias has its target's signature"
    assert not (alias.fixed and not alias.derived), "only a derived alias can fix a keyword"


# ── the signature ───────────────────────────────────────────────────────────


def _described(
    fn: Callable[..., object], drop: frozenset[str] = frozenset()
) -> tuple[list[tuple[object, ...]], object]:
    """What ``inspect`` reads of ``fn``: every parameter in full, and the return annotation."""
    sig = inspect.signature(fn)
    params = [
        (p.name, p.kind, p.default, p.annotation)
        for p in sig.parameters.values()
        if p.name not in drop
    ]
    return params, sig.return_annotation


@pytest.mark.parametrize("alias", _EQUAL, ids=_ids(_EQUAL))
def test_an_alias_has_its_targets_signature(alias: OpAlias) -> None:
    got = _described(getattr(lucid, alias.name))
    want = _described(getattr(lucid, alias.target), frozenset(alias.fixed))
    assert got == want


@pytest.mark.parametrize("alias", _DECLARED, ids=_ids(_DECLARED))
def test_a_declared_difference_has_the_parameters_it_declares(alias: OpAlias) -> None:
    names = list(inspect.signature(getattr(lucid, alias.name)).parameters)
    assert names == list(alias.params), alias.differs


@pytest.mark.parametrize("alias", _DERIVED, ids=_ids(_DERIVED))
def test_a_derived_alias_runs_its_targets_adapter(alias: OpAlias) -> None:
    target, entry = _entry(alias.target), _entry(alias.name)
    assert entry.name == alias.name
    assert (entry.engine_fn is target.engine_fn) == (not alias.fixed)
    assert entry.n_tensor_args == target.n_tensor_args
    assert getattr(lucid, alias.name).__module__ == getattr(lucid, alias.target).__module__


# ── the stubs ───────────────────────────────────────────────────────────────

#: The method stubs spell "a tensor or a number" three ways: ``Tensor | _float``
#: (a registry method), ``Tensor | Scalar`` (a free function as a method) and
#: ``TensorOrScalar`` (an in-place twin).  They mean one thing.
_TENSOR_OR_NUMBER = {"Tensor | _float", "Tensor | Scalar", "TensorOrScalar"}


def _stub_signature(fn: ast.FunctionDef, drop: frozenset[str]) -> list[tuple[str, str, bool, str]]:
    """``(name, kind, has default, annotation)`` of each parameter of a stub ``def``."""
    a = fn.args
    positional = [*a.posonlyargs, *a.args]
    first_default = len(positional) - len(a.defaults)
    rows = [
        (
            arg.arg,
            "positional-only" if i < len(a.posonlyargs) else "positional",
            i >= first_default,
            ast.unparse(arg.annotation) if arg.annotation else "",
        )
        for i, arg in enumerate(positional)
    ]
    if a.vararg:
        rows.append(
            (a.vararg.arg, "variadic", False, ast.unparse(a.vararg.annotation or ast.Constant("")))
        )
    rows += [
        (
            arg.arg,
            "keyword",
            default is not None,
            ast.unparse(arg.annotation) if arg.annotation else "",
        )
        for arg, default in zip(a.kwonlyargs, a.kw_defaults, strict=True)
    ]
    return [
        (n, k, d, "Tensor | Scalar" if t in _TENSOR_OR_NUMBER else t)
        for n, k, d, t in rows
        if n not in drop
    ]


def _stub_functions(path: Path, in_class: str | None) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body: list[ast.stmt] = tree.body
    if in_class is not None:
        body = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == in_class).body
    found: dict[str, ast.FunctionDef] = {}
    for node in body:
        if isinstance(node, ast.FunctionDef):
            found.setdefault(node.name, node)
    return found


_FREE = _stub_functions(_FREE_STUB, None)
_METHODS = _stub_functions(_METHOD_STUB, "Tensor")


@pytest.mark.parametrize("alias", _EQUAL, ids=_ids(_EQUAL))
def test_the_stub_of_an_alias_is_its_targets(alias: OpAlias) -> None:
    drop = frozenset(alias.fixed)
    assert _stub_signature(_FREE[alias.name], frozenset()) == _stub_signature(
        _FREE[alias.target], drop
    )


@pytest.mark.parametrize("alias", _EQUAL, ids=_ids(_EQUAL))
@pytest.mark.parametrize("suffix", ["", "_"], ids=["method", "inplace"])
def test_the_method_stub_of_an_alias_is_its_targets(alias: OpAlias, suffix: str) -> None:
    mine, theirs = _METHODS.get(alias.name + suffix), _METHODS.get(alias.target + suffix)
    if mine is None or theirs is None:
        pytest.skip("no such method on both spellings")
    drop = frozenset(alias.fixed)
    assert _stub_signature(mine, frozenset()) == _stub_signature(theirs, drop)


@pytest.mark.parametrize("alias", _DECLARED, ids=_ids(_DECLARED))
def test_the_stub_of_a_declared_difference_has_its_declared_parameters(alias: OpAlias) -> None:
    free = [row[0] for row in _stub_signature(_FREE[alias.name], frozenset())]
    assert free == list(alias.params)
    method = _METHODS.get(alias.name)
    if method is not None:
        assert [row[0] for row in _stub_signature(method, frozenset())] == [
            "self",
            *alias.params[1:],
        ]


# ── the operands ────────────────────────────────────────────────────────────

_REFUSALS = (TypeError, ValueError, RuntimeError, NotImplementedError, IndexError)


def _probes(arity: int, device: str) -> list[tuple[object, ...]]:
    """Operand tuples that tell a tensor from a number from a mistake, for an op of ``arity``."""
    t = lucid.tensor([1.0, 2.0], device=device)
    i = lucid.tensor([7, 3], device=device)
    if arity == 1:
        return [(t,), (i,), (2.0,), (None,), ("x",)]
    if arity == 2:
        return [
            (t, t),
            (i, i),
            (2.0, t),
            (t, 2.0),
            (2.0, 2.0),
            (None, t),
            (t, None),
            ("x", t),
            (i, 2),
        ]
    return [([t, t],), ((t, t),), ([t, 2.0],), (None,), ([],), ([t, t], 0), ([t, t], 5)]


def _call(fn: Callable[..., object], args: tuple[object, ...], kwargs: dict[str, object]) -> object:
    try:
        return fn(*args, **kwargs)
    except _REFUSALS as err:
        return type(err)


def _same(a: object, b: object) -> bool:
    if isinstance(a, lucid.Tensor) and isinstance(b, lucid.Tensor):
        return a.dtype == b.dtype and a.shape == b.shape and a.tolist() == b.tolist()
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    return a == b


@pytest.mark.parametrize("alias", _DERIVED, ids=_ids(_DERIVED))
def test_an_alias_takes_and_refuses_the_operands_its_target_does(
    alias: OpAlias, device: str
) -> None:
    arity = _entry(alias.target).n_tensor_args
    mine, theirs = getattr(lucid, alias.name), getattr(lucid, alias.target)
    for args in _probes(arity, device):
        got = _call(mine, args, {})
        want = _call(theirs, args, dict(alias.fixed))
        assert _same(got, want), f"{alias.name}{args!r}: {got!r}, but {alias.target} gives {want!r}"


def test_a_number_first_is_taken_by_the_five_that_refused_it(device: str) -> None:
    t = lucid.tensor([1.0, 2.0], device=device)
    for name in ("multiply", "divide", "true_divide", "subtract", "arctan2"):
        assert getattr(lucid, name)(2.0, t).shape == t.shape, name


def test_true_divide_refuses_a_rounding_mode_as_the_reference_does() -> None:
    t = lucid.tensor([1.0, 2.0])
    with pytest.raises(TypeError, match="rounding_mode"):
        lucid.true_divide(t, t, rounding_mode="floor")  # type: ignore[call-arg]


# ── the one place ───────────────────────────────────────────────────────────


#: The operator a one-line body may spell instead of a call: ``input * other`` is ``mul``.
_OPERATORS: dict[type[ast.operator], str] = {
    ast.Add: "add",
    ast.Sub: "sub",
    ast.Mult: "mul",
    ast.Div: "div",
    ast.FloorDiv: "floor_divide",
    ast.Pow: "pow",
    ast.Mod: "remainder",
}


def _passes_through(fn: ast.FunctionDef, args: list[ast.expr], keywords: list[ast.keyword]) -> bool:
    """Whether ``args`` / ``keywords`` hand ``fn``'s own parameters on, by the names it has them."""
    params = [a.arg for a in [*fn.args.posonlyargs, *fn.args.args]]
    passed = [a.id for a in args if isinstance(a, ast.Name)]
    if not args and not keywords or len(passed) != len(args) or passed != params[: len(passed)]:
        return False
    named = {a.arg for a in [*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs]}
    return all(
        k.arg in named and isinstance(k.value, ast.Name) and k.value.id == k.arg for k in keywords
    )


def _forwarded_op(fn: ast.FunctionDef) -> str | None:
    """The public op ``fn`` does nothing but hand its parameters to, or ``None``."""
    body = fn.body[1:] if ast.get_docstring(fn) is not None else fn.body
    if len(body) != 1 or not isinstance(body[0], ast.Return) or body[0].value is None:
        return None
    value = body[0].value
    if isinstance(value, ast.BinOp):
        name = _OPERATORS.get(type(value.op))
        return name if name and _passes_through(fn, [value.left, value.right], []) else None
    if not isinstance(value, ast.Call):
        return None
    callee = value.func
    if isinstance(callee, ast.Name):
        name = callee.id
    elif (
        isinstance(callee, ast.Attribute)
        and isinstance(callee.value, ast.Name)
        and callee.value.id == "lucid"
    ):
        name = callee.attr
    else:
        return None
    if not _passes_through(fn, value.args, value.keywords):
        return None
    op = getattr(lucid, name, None)
    return name if callable(op) and not isinstance(op, type) and name in lucid.__all__ else None


def _forwarders() -> list[tuple[str, str]]:
    found = []
    for path in sorted((_ROOT / "_ops").rglob("*.py")):
        if path.name.startswith("test_"):
            continue
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                target = _forwarded_op(node)
                if target is not None and node.name in lucid.__all__:
                    found.append((node.name, target))
    return found


@pytest.mark.parametrize(("name", "target"), _forwarders(), ids=lambda v: str(v))
def test_a_function_that_only_forwards_is_a_row_of_the_table(name: str, target: str) -> None:
    assert name in ALIAS_NAMES, (
        f"{name}() only hands its parameters to {target}(): it is an alias, and a hand-written "
        "one re-spells the target's signature (LCD-1052).  Add it to "
        "lucid._ops._aliases.ALIASES — derived, if the target is a registry op."
    )
