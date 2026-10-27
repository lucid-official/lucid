"""Every parameter a public docstring documents is one the signature takes.

LCD-410 renamed parameters to the reference's names and the docstrings
kept the old ones: ``tril``'s ``Parameters`` still listed ``k``,
``where``'s ``cond`` / ``x`` / ``y``, ``searchsorted``'s ``sorted_seq``,
and ``lucid.triu(A, k=1)`` stood in an example that now raises.  The
registry ops' docstrings live in ``lucid/__init__.pyi`` (``tools/gen_pyi``
carries them over), so nothing that runs code read them.

Held here: each name in the ``Parameters`` / ``Other Parameters`` section
of a public function's or class's docstring — in the stubs, and on the
runtime objects of the public namespaces — is a parameter of its
signature (the class's ``__init__`` for a class).  A signature that takes
``**kwargs`` documents what it forwards and is left out.
"""

import annotationlib
import ast
import importlib
import inspect
import re
from pathlib import Path

import pytest

import lucid

_ROOT = Path(lucid.__file__).parent
_STUBS = (_ROOT / "__init__.pyi", _ROOT / "_tensor" / "tensor.pyi")
_NAMESPACES = (
    "lucid",
    "lucid.nn",
    "lucid.nn.functional",
    "lucid.linalg",
    "lucid.fft",
    "lucid.special",
    "lucid.signal",
    "lucid.einops",
    "lucid.distributions",
    "lucid.autograd",
    "lucid.optim",
)

_ENTRY = re.compile(r"^\s*\*{0,2}([A-Za-z_]\w*(?:\s*,\s*\*{0,2}[A-Za-z_]\w*)*)\s*(?::|$)")


def _documented(doc: str) -> list[str]:
    """The names in ``doc``'s ``Parameters`` / ``Other Parameters`` sections."""
    lines = inspect.cleandoc(doc).expandtabs().split("\n")
    names: list[str] = []
    i = 0
    while i < len(lines):
        header = lines[i].strip()
        underlined = i + 1 < len(lines) and set(lines[i + 1].strip()) == {"-"}
        if header not in ("Parameters", "Other Parameters") or not underlined:
            i += 1
            continue
        indent = len(lines[i]) - len(lines[i].lstrip())
        i += 2
        while i < len(lines):
            line = lines[i]
            next_is_rule = i + 1 < len(lines) and set(lines[i + 1].strip()) == {"-"}
            if line.strip() and len(line) - len(line.lstrip()) == indent:
                if next_is_rule:
                    break  # the next section's header
                match = _ENTRY.match(line)
                if match:
                    names += [n.strip().lstrip("*") for n in match.group(1).split(",")]
            i += 1
    return [n for n in names if n != "None"]  # "None" says there are none


def _stub_cases() -> list[object]:
    cases = []
    for path in _STUBS:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
            nodes += [n for n in cls.body if isinstance(n, ast.FunctionDef)]
        for node in nodes:
            doc = ast.get_docstring(node, clean=False)
            if not doc or node.name.startswith("_"):
                continue
            a = node.args
            params = {p.arg for p in a.posonlyargs + a.args + a.kwonlyargs}
            params |= {p.arg for p in (a.vararg, a.kwarg) if p is not None}
            label = f"{path.name}:{node.name}:{node.lineno}"
            cases.append(pytest.param(doc, sorted(params), id=label))
    return cases


def _runtime_cases() -> list[object]:
    cases, seen = [], set()
    for module_name in _NAMESPACES:
        module = importlib.import_module(module_name)
        names = set(dir(module)) | set(getattr(module, "__all__", []))
        if module_name == "lucid":  # the lazily loaded names too
            stub = ast.parse(_STUBS[0].read_text(encoding="utf-8"))
            names |= {n.name for n in stub.body if isinstance(n, ast.FunctionDef)}
        for name in sorted(names):
            obj = None if name.startswith("_") else getattr(module, name, None)
            if obj is None or id(obj) in seen or not callable(obj):
                continue
            if not str(getattr(obj, "__module__", "")).startswith("lucid"):
                continue
            seen.add(id(obj))
            target = obj.__init__ if isinstance(obj, type) else obj
            try:
                signature = inspect.signature(
                    target, annotation_format=annotationlib.Format.FORWARDREF
                )
            except (TypeError, ValueError):
                continue
            kinds = {p.kind for p in signature.parameters.values()}
            if inspect.Parameter.VAR_KEYWORD in kinds or not obj.__doc__:
                continue
            params = sorted(signature.parameters)
            cases.append(pytest.param(obj.__doc__, params, id=f"{module_name}.{name}"))
    return cases


@pytest.mark.parametrize("doc,params", _stub_cases() + _runtime_cases())
def test_documented_parameters_are_taken(doc: str, params: list[str]) -> None:
    assert [n for n in _documented(doc) if n not in params] == []
