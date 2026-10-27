"""Hand-written stub classes vs the runtime objects they describe.

``_nn_namespace``, ``_linalg_namespace`` and the rest of ``lucid/_C/engine.pyi``'s
sub-namespace classes are typed out in gen_pyi's template, so regenerating
cannot correct them and the freshness diff cannot see them drift:
``_linalg_namespace`` had declared a ``foo`` the engine never had and omitted
the real ``matrix_power`` (LCD-402, :func:`namespace_drift`).  ``tensor.pyi``'s
``Tensor`` had the same blind spot: gen_pyi skipped thirteen registry methods
as "already in the header" when nothing there defined them, and the header
omitted another thirteen methods (LCD-357, :func:`class_drift`).
``tools/check_stubs.py`` runs both on every generated stub.  Stdlib only, so
the check type-checks on its own.
"""

import annotationlib
import ast
import inspect
import re
from collections.abc import Callable, Sequence


def _split_top(params: str) -> list[str]:
    out: list[str] = []
    depth, cur = 0, ""
    for ch in params + ",":
        if ch == "," and depth == 0:
            if cur.strip():
                out.append(cur.strip())
            cur = ""
            continue
        depth += ch in "[("
        depth -= ch in "])"
        cur += ch
    return out


def _runtime_params(name: str, fn: object) -> list[str] | None:
    """Parameter names from pybind11's first docstring line; None when overloaded."""
    doc = getattr(fn, "__doc__", None) or ""
    first = doc.splitlines()[0] if doc else ""
    m = re.match(rf"^{re.escape(name)}\((?P<params>.*)\)\s*->", first)
    if m is None:
        return None
    return [p.split(":")[0].split("=")[0].strip() for p in _split_top(m["params"])]


def _stub_params(fn: ast.FunctionDef) -> list[str]:
    a = fn.args
    names = [p.arg for p in (*a.posonlyargs, *a.args)]
    if a.vararg is not None:
        names.append("*" + a.vararg.arg)
    names += [p.arg for p in a.kwonlyargs]
    if a.kwarg is not None:
        names.append("**" + a.kwarg.arg)
    return names


def _stub_members(cls: ast.ClassDef) -> dict[str, ast.FunctionDef | None]:
    members: dict[str, ast.FunctionDef | None] = {}
    for node in cls.body:
        if isinstance(node, ast.FunctionDef):
            members[node.name] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            members[node.target.id] = None
        elif isinstance(node, ast.Assign):
            members.update({t.id: None for t in node.targets if isinstance(t, ast.Name)})
    return members


def _namespace_attrs(body: list[ast.stmt], classes: set[str]) -> dict[str, str]:
    """``attr: _x_namespace`` declarations in *body*, attribute -> class name."""
    return {
        node.target.id: node.annotation.id
        for node in body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and isinstance(node.annotation, ast.Name)
        and node.annotation.id in classes
    }


def namespace_drift(stub: str, engine: object) -> list[str]:
    """Every disagreement between the stub's sub-namespace classes and *engine*."""
    tree = ast.parse(stub)
    classes = {
        n.name: n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name.endswith("_namespace")
    }
    todo = [(attr, cls, engine) for attr, cls in _namespace_attrs(tree.body, set(classes)).items()]
    problems: list[str] = []
    while todo:
        path, cls_name, parent = todo.pop()
        runtime = getattr(parent, path.rsplit(".", 1)[-1], None)
        if runtime is None:  # optional build feature (``quantized``): hasattr-guarded
            continue
        cls = classes[cls_name]
        members = _stub_members(cls)
        have = {n for n in dir(runtime) if not n.startswith("__")}
        problems += [
            f"{path}.{n}: in the stub, not in the engine" for n in sorted(set(members) - have)
        ]
        problems += [
            f"{path}.{n}: in the engine, not in the stub" for n in sorted(have - set(members))
        ]
        for name, fn in members.items():
            if fn is None or name not in have:
                continue
            want = _runtime_params(name, getattr(runtime, name))
            got = _stub_params(fn)
            if want is not None and got != want:
                problems.append(
                    f"{path}.{name}: stub ({', '.join(got)}), engine ({', '.join(want)})"
                )
        todo += [
            (f"{path}.{attr}", sub, runtime)
            for attr, sub in _namespace_attrs(cls.body, set(classes)).items()
        ]
    return problems


def _signature_params(fn: object) -> list[str] | None:
    """``fn``'s parameters after ``self``, spelled as :func:`_stub_params` spells them.

    ``Format.STRING`` leaves annotations unevaluated: inside ``Tensor`` an
    annotation's ``int`` resolves to the ``Tensor.int`` method, so
    evaluating ``stride``'s ``int | None`` raises.
    """
    if not callable(fn):
        return None
    try:
        signature = inspect.signature(fn, annotation_format=annotationlib.Format.STRING)
    except (TypeError, ValueError):
        return None
    names = []
    for p in signature.parameters.values():
        prefix = (
            "*"
            if p.kind is p.VAR_POSITIONAL
            else "**"
            if p.kind is p.VAR_KEYWORD
            else ""
        )
        names.append(prefix + p.name)
    return names[1:]


def _stub_methods(cls: ast.ClassDef) -> dict[str, list[ast.FunctionDef]]:
    """Each method's definitions, several for an overload; properties left out."""
    methods: dict[str, list[ast.FunctionDef]] = {}
    for node in cls.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        decorators = [ast.unparse(d) for d in node.decorator_list]
        if any(d == "property" or d.endswith((".setter", ".deleter")) for d in decorators):
            continue
        methods.setdefault(node.name, []).append(node)
    return methods


def _method_drift(
    qualname: str, defs: list[ast.FunctionDef], want: list[str]
) -> list[str]:
    """A lone definition names exactly ``want``; each overload, a subset of it."""
    if len(defs) == 1:
        got = _stub_params(defs[0])[1:]
        if got == want:
            return []
        return [f"{qualname}: stub ({', '.join(got)}), runtime ({', '.join(want)})"]
    bare = {p.lstrip("*") for p in want}
    problems = []
    for overload in defs:
        unknown = [p for p in _stub_params(overload)[1:] if p.lstrip("*") not in bare]
        if unknown:
            problems.append(
                f"{qualname}: overload names ({', '.join(unknown)}), "
                f"runtime ({', '.join(want)})"
            )
    return problems


def class_drift(
    stub: str, runtime: type, forwarded: Callable[[str], list[str] | None]
) -> list[str]:
    """Every disagreement between the stub's class ``runtime.__name__`` and *runtime*.

    Each public name on the class needs a stub, and each stubbed method
    names the parameters it takes.  A method that takes ``*args, **kwargs``
    and passes them on is held to what ``forwarded(name)`` says it binds;
    ``None`` from either skips the comparison, not the name.
    """
    name = runtime.__name__
    cls = next(
        n
        for n in ast.parse(stub).body
        if isinstance(n, ast.ClassDef) and n.name == name
    )
    have = {n for n in dir(runtime) if not n.startswith("_")}
    stubbed = {n for n in _stub_members(cls) if not n.startswith("_")}
    problems = [f"{name}.{n}: in the stub, not at runtime" for n in sorted(stubbed - have)]
    problems += [f"{name}.{n}: at runtime, not in the stub" for n in sorted(have - stubbed)]
    for method, defs in sorted(_stub_methods(cls).items()):
        if method.startswith("_") or method not in have:
            continue
        want = _signature_params(getattr(runtime, method))
        if want is not None and any(p.startswith("**") for p in want):
            want = forwarded(method)
        if want is not None:
            problems += _method_drift(f"{name}.{method}", defs, want)
    return problems


def module_all_drift(stub: str, all_names: Sequence[str], *, module: str = "lucid") -> list[str]:
    """Every name in *all_names* not explicitly exported by the module *stub*.

    Under PEP 484 (``mypy --strict`` / ``--no-implicit-reexport``, pyright),
    an import in a ``.pyi`` stub only re-exports a symbol when written in the
    redundant ``X as X`` alias form.
    """
    exported: set[str] = set()
    for node in ast.parse(stub).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            exported.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            exported.add(node.target.id)
        elif isinstance(node, ast.Assign):
            exported.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            exported.update(
                alias.asname for alias in node.names if alias.asname is not None
            )
    return [
        f"{module}.{n}: in {module}.__all__, not exported in {module}/__init__.pyi"
        for n in all_names
        if n not in exported
    ]

