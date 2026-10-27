"""Every stubbed signature is the one the runtime binds.

``lucid/__init__.pyi`` is partly hand-written in ``tools/gen_pyi.py``, and
the hand-written lines did not follow the runtime when LCD-410 renamed
its parameters: ``is_floating_point(t)``, ``from_numpy(arr)`` without
``/``, ``zeros_like(t, ...)``, ``bernoulli(p, ...)``, ``split(x, ...)``
— a type checker accepted keywords the runtime refused and refused the
ones it took, and ``lucid.lerp`` lost ``end`` / ``weight`` altogether.
``tools/check_stubs`` only checks that the generator's output is
committed, and ``class_drift`` compares Tensor parameter names, not their
kinds or defaults.

Here each stubbed function of ``lucid`` and each method of ``Tensor``
whose runtime signature is concrete — not a ``*args, **kwargs``
forwarder — is held to that signature: the same parameter names, in the
same order, of the same kinds (positional-only, either, ``*args``,
keyword-only, ``**kwargs``), optional exactly where the runtime is.  An
overloaded stub may split the signature between its overloads, so each
overload is held to naming only parameters the runtime has, of the same
kind.  A stubbed name missing at runtime fails (``dot`` / ``inner`` /
``outer`` and seven activations were stubbed on ``lucid`` and lived only
under ``linalg`` / ``nn.functional``); the few left unchecked are listed
in ``_CONTRACT_ONLY``.  A Tensor method generated from the registry is
held to its callee's signature — that branch was skipped for 185 of 246
methods until the review.  ``nn.functional`` has no stub file: its
source is the stub.
"""

import annotationlib
import ast
import inspect
from pathlib import Path
from typing import NamedTuple

import pytest

import lucid
import lucid._ops as ops
from lucid._ops._registry import _REGISTRY
from lucid._tensor.tensor import Tensor

_ROOT = Path(lucid.__file__).parent
_INIT_STUB = _ROOT / "__init__.pyi"
_TENSOR_STUB = _ROOT / "_tensor" / "tensor.pyi"

_KINDS = {
    inspect.Parameter.POSITIONAL_ONLY: "positional-only",
    inspect.Parameter.POSITIONAL_OR_KEYWORD: "either",
    inspect.Parameter.VAR_POSITIONAL: "*args",
    inspect.Parameter.KEYWORD_ONLY: "keyword-only",
    inspect.Parameter.VAR_KEYWORD: "**kwargs",
}


class _Param(NamedTuple):
    name: str
    kind: str
    optional: bool


def _stub_params(fn: ast.FunctionDef, method: bool) -> list[_Param]:
    a = fn.args
    positional = a.posonlyargs + a.args
    first_default = len(positional) - len(a.defaults)
    out = [
        _Param(
            arg.arg, "positional-only" if i < len(a.posonlyargs) else "either", i >= first_default
        )
        for i, arg in enumerate(positional)
    ]
    if a.vararg is not None:
        out.append(_Param(a.vararg.arg, "*args", True))
    out += [
        _Param(arg.arg, "keyword-only", default is not None)
        for arg, default in zip(a.kwonlyargs, a.kw_defaults, strict=True)
    ]
    if a.kwarg is not None:
        out.append(_Param(a.kwarg.arg, "**kwargs", True))
    return out[1:] if method else out


def _runtime_params(fn: object, method: bool) -> list[_Param] | None:
    try:
        signature = inspect.signature(  # type: ignore[arg-type]
            fn, annotation_format=annotationlib.Format.FORWARDREF
        )
    except (TypeError, ValueError):
        return None
    return _signature_params(signature, method)


def _signature_params(signature: inspect.Signature, method: bool) -> list[_Param]:
    params = list(signature.parameters.values())
    if method and params and params[0].name == "self":
        params = params[1:]
    out = [_Param(p.name, _KINDS[p.kind], p.default is not p.empty) for p in params]
    for i, p in enumerate(out):  # ``*args`` and ``**kwargs`` are optional
        if p.kind in ("*args", "**kwargs"):
            out[i] = p._replace(optional=True)
    return out


def _forwards(params: list[_Param]) -> bool:
    """A ``*args, **kwargs`` forwarder: it binds nothing of its own."""
    kinds = {p.kind for p in params}
    return "*args" in kinds and "**kwargs" in kinds


def _stub_defs(path: Path, cls: str | None) -> dict[str, list[ast.FunctionDef]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = tree.body
    if cls is not None:
        body = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
    defs: dict[str, list[ast.FunctionDef]] = {}
    for node in body:
        if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
            continue
        decorators = {ast.unparse(d) for d in node.decorator_list}
        if decorators & {"property", "classmethod", "staticmethod"} or any(
            d.endswith((".setter", ".deleter")) for d in decorators
        ):
            continue
        defs.setdefault(node.name, []).append(node)
    return defs


def _drift(stub: list[list[_Param]], runtime: list[_Param]) -> list[str]:
    if len(stub) == 1:
        return [] if stub[0] == runtime else [f"stub {stub[0]}", f"runtime {runtime}"]
    known = {(p.name, p.kind) for p in runtime}
    names = {p.name for p in runtime}
    problems = []
    for overload in stub:
        for p in overload:
            if p.name not in names:
                problems.append(f"an overload names {p.name}, which the runtime lacks")
            elif (p.name, p.kind) not in known and p.kind != "*args":
                problems.append(f"an overload makes {p.name} {p.kind}")
    return problems


_BY_METHOD = {e.method_name: e for e in _REGISTRY if e.method_name}


def _tensor_runtime(name: str) -> list[_Param] | None:
    params = _runtime_params(getattr(Tensor, name), method=True)
    if params is None or not _forwards(params):
        return params
    entry = _BY_METHOD.get(name)
    if entry is None or entry.n_tensor_args == -1:
        return None
    params = _signature_params(ops._signature_for_entry(entry), method=False)[1:]
    if _forwards(params):
        return None
    # The method body takes ``*args, **kwargs``: every parameter is either.
    return params


def _cases() -> list[object]:
    cases = []
    for name, defs in sorted(_stub_defs(_INIT_STUB, None).items()):
        cases.append(pytest.param("lucid", name, defs, id=f"lucid.{name}"))
    for name, defs in sorted(_stub_defs(_TENSOR_STUB, "Tensor").items()):
        cases.append(pytest.param("Tensor", name, defs, id=f"Tensor.{name}"))
    return cases


#: Stubbed names the runtime does not spell out — the only cases left
#: unchecked, and listed so the guard cannot go quiet: a name that stops
#: being one of these fails, as does a new one.  ``var`` / ``std`` and the
#: pairs read the reference's overloads by type; ``to`` its four shapes;
#: ``no_grad`` / ``enable_grad`` are classes.
_CONTRACT_ONLY = frozenset(
    {
        "lucid.enable_grad",
        "lucid.no_grad",
        "lucid.std",
        "lucid.std_mean",
        "lucid.var",
        "lucid.var_mean",
        "Tensor.std",
        "Tensor.to",
        "Tensor.var",
    }
)


def _runtime(owner: str, name: str) -> tuple[list[_Param] | None, bool]:
    """``name``'s runtime parameters (``None`` for a forwarder) and whether a method."""
    if owner == "Tensor":
        assert hasattr(Tensor, name), f"Tensor.{name} is stubbed, not at runtime"
        return _tensor_runtime(name), True
    obj = getattr(lucid, name, None)
    assert obj is not None, f"lucid.{name} is stubbed, not at runtime"
    if isinstance(obj, type):
        return None, False
    return _runtime_params(obj, method=False), False


@pytest.mark.parametrize("owner,name,defs", _cases())
def test_the_stub_binds_what_the_runtime_binds(
    owner: str, name: str, defs: list[ast.FunctionDef]
) -> None:
    runtime, method = _runtime(owner, name)
    unchecked = runtime is None or _forwards(runtime)
    assert unchecked == (f"{owner}.{name}" in _CONTRACT_ONLY)
    if runtime is None or unchecked:
        return
    stub = [_stub_params(d, method) for d in defs]
    assert _drift(stub, runtime) == []
