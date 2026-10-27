"""Every keyword a stub offers, the runtime accepts.

``Tensor`` methods built from the op registry take ``*args, **kwargs`` and
forward them to an engine function or adapter, so ``inspect`` cannot
compare them with ``tensor.pyi``.  The stub and the callee drifted apart
unnoticed: ``x.flatten(start=1)``, ``x.softmax(axis=-1)``,
``x.unbind(axis=0)``, ``x.swapaxes(axis0=0, axis1=1)``,
``x.topk(k, largest=False)``, ``x.sort(descending=True)`` and
``x.sum(correction=1)`` were all in the stub and all raised
``TypeError``.  The registry's own ``extra_kwargs`` listed ``axis`` /
``axes`` / ``keepdims`` aliases that no callee took.

Both lists are held to the callee's real parameter names here — the
callee's own, not the engine function an adapter wraps: following
``__wrapped__`` to the engine's ``(a, b)`` hid that ``x.add(other=2)`` and
``lucid.add(input=x, other=y)`` raised ``TypeError`` (LCD-341).  The
leading tensor operands are the wrapper's to bind: their advertised names
(``input`` / ``other``) are put back in position before the callee runs.
"""

import annotationlib
import ast
import inspect
import keyword
import re
from pathlib import Path
from types import ModuleType
from typing import Callable, NamedTuple

import pytest

import lucid
import lucid._ops as ops
from lucid._ops._adapters import _make_arith_adapter
from lucid._ops._registry import _REGISTRY, OpEntry
from lucid._tensor.tensor import Tensor

STUB = Path(lucid.__file__).parent / "_tensor" / "tensor.pyi"


def _parameters(fn: Callable[..., object]) -> list[str] | None:
    """The callee's parameter names, or None when it takes ``**kwargs``."""
    try:
        params = list(inspect.signature(fn, follow_wrapped=False).parameters.values())
    except (TypeError, ValueError):
        # A pybind11 function: the first docstring line is its signature.
        first = (fn.__doc__ or "").split("\n")[0]
        m = re.match(r"\w+\((.*)\)\s*->", first)
        return re.findall(r"(\w+):", m.group(1)) if m else None
    if any(p.kind is p.VAR_KEYWORD for p in params):
        return None
    return [p.name for p in params]


@pytest.mark.parametrize("entry", [e for e in _REGISTRY if e.extra_kwargs], ids=lambda e: e.name)
def test_registry_keywords_name_real_parameters(entry) -> None:  # type: ignore[no-untyped-def]
    accepted = _parameters(entry.engine_fn)
    if accepted is None:
        pytest.skip("callee takes **kwargs")
    assert [k for k in entry.extra_kwargs if k not in accepted] == []


def _stub_methods() -> list[tuple[str, list[str]]]:
    tree = ast.parse(STUB.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Tensor")
    out = []
    for fn in cls.body:
        if not isinstance(fn, ast.FunctionDef) or fn.name.startswith("__"):
            continue
        if fn.args.vararg is not None:
            continue
        names = [a.arg for a in fn.args.args[1:]] + [a.arg for a in fn.args.kwonlyargs]
        if names:
            out.append((fn.name, names))
    return out


_BY_METHOD = {e.method_name: e for e in _REGISTRY if e.method_name}


#: Stub keywords no callee takes yet — strict, so a fix flips them to XPASS.
_KNOWN_GAPS: dict[str, str] = {}


def _stub_cases() -> list[object]:
    return [
        pytest.param(
            name,
            keywords,
            id=str((name, keywords)),
            marks=(
                [pytest.mark.xfail(strict=True, reason=_KNOWN_GAPS[name])]
                if name in _KNOWN_GAPS
                else []
            ),
        )
        for name, keywords in _stub_methods()
    ]


@pytest.mark.parametrize("name,keywords", _stub_cases())
def test_tensor_stub_keywords_reach_the_callee(name: str, keywords: list[str]) -> None:
    method = getattr(Tensor, name, None)
    assert method is not None, f"tensor.pyi declares Tensor.{name}, which does not exist"
    direct = _parameters(method)
    if direct is not None and "args" not in direct:
        assert [k for k in keywords if k not in direct] == []
        return
    entry = _BY_METHOD.get(name)
    if entry is None:
        pytest.skip("forwards **kwargs to something other than a registry callee")
    accepted = _parameters(entry.engine_fn)
    if accepted is None:
        pytest.skip("callee takes **kwargs")
    # ``self`` fills the first parameter; the method binds the other
    # advertised tensor operands itself, and passes the rest on.
    bound = ops._leading_tensor_names(entry)[1:]
    assert [k for k in keywords if k not in accepted[1:] and k not in bound] == []


_FREE = [e for e in _REGISTRY if e.free_fn_name and e.n_tensor_args != -1]


@pytest.mark.parametrize("entry", _FREE, ids=lambda e: e.free_fn_name)
def test_free_function_keywords_reach_the_callee(entry: OpEntry) -> None:
    # The registry's own wrapper: ``lucid.softmax`` and a few others are
    # other functions of the same name, checked by their own signatures.
    advertised = list(inspect.signature(getattr(ops, entry.free_fn_name)).parameters)
    accepted = _parameters(entry.engine_fn)
    if accepted is None:
        pytest.skip("callee takes **kwargs")
    bound = ops._leading_tensor_names(entry)
    assert [k for k in advertised if k not in accepted and k not in bound] == []


# ── Every public signature, held to the reference's ──────────────────────────
#
# The keyword checks above follow a call into its callee.  This one asks the
# reference's own parser what it binds — the aten schema for a builtin, the
# Python signature otherwise — and holds every public function and Tensor
# method to it: each positional slot named as the reference names it (a
# positional-only one by kind alone), each keyword-only argument keyword-only,
# no argument required that the reference leaves optional, and no boolean
# default turned round.  ~60 top-level functions named their operands
# ``x`` / ``a`` / ``t`` where the reference says ``input``, and code passing
# them by keyword raised ``TypeError`` (LCD-410); ``F.max_pool2d``'s sixth
# positional argument was ``return_indices`` where the reference reads
# ``ceil_mode`` (LCD-421).


class _Param(NamedTuple):
    name: str
    kind: str  # "po" / "pk" / "var" / "kw" / "varkw"
    optional: bool
    default: object = None
    int_list: bool = False


_KINDS = {
    inspect.Parameter.POSITIONAL_ONLY: "po",
    inspect.Parameter.POSITIONAL_OR_KEYWORD: "pk",
    inspect.Parameter.VAR_POSITIONAL: "var",
    inspect.Parameter.KEYWORD_ONLY: "kw",
    inspect.Parameter.VAR_KEYWORD: "varkw",
}

#: Arguments the whole surface leaves out: output buffers, layouts, memory
#: formats — and the reference's own deprecated autograd and scheduler
#: arguments, which are not added (LCD-410 / LCD-430 decision).
_OMITTED = frozenset(
    {
        "out",
        "layout",
        "memory_format",
        "pin_memory",
        "grad_variables",
        "epoch",
        "name",
        "recompile_limit",
        "isolate_recompiles",
        "shapes_spec",
    }
)
#: Factory keywords compared only where Lucid offers them.
_FACTORY = frozenset({"dtype", "device", "requires_grad", "generator"})

#: The reference names these by overload (``sections`` for an int,
#: ``indices`` for a list); Lucid keeps the one name its documentation uses.
_BY_OVERLOAD = {"sections": "indices_or_sections", "indices": "indices_or_sections"}
_OVERLOADED_NAMES = frozenset({"hsplit", "vsplit", "dsplit", "tensor_split"})


def _params(signature: inspect.Signature, drop_self: bool) -> list[_Param]:
    params = list(signature.parameters.values())
    if drop_self and params and params[0].name == "self":
        params = params[1:]
    return [
        _Param(
            p.name,
            _KINDS[p.kind],
            p.default is not p.empty,
            None if p.default is p.empty else p.default,
        )
        for p in params
    ]


def _signature(fn: object) -> inspect.Signature:
    # Annotations left unevaluated: ``from_numpy``'s names a lazily imported
    # ``np``, and only the parameters matter here.
    return inspect.signature(fn, annotation_format=annotationlib.Format.FORWARDREF)  # type: ignore[arg-type]


def _catch_all(params: list[_Param]) -> bool:
    return any(p.kind == "varkw" for p in params) and not any(p.kind == "kw" for p in params)


def _stub_params(path: Path, name: str, cls: str | None) -> list[_Param] | None:
    """``name``'s parameters as the stub writes them (its first overload)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = tree.body
    if cls is not None:
        body = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
    fn = next((n for n in body if isinstance(n, ast.FunctionDef) and n.name == name), None)
    if fn is None:
        return None
    a = fn.args
    positional = a.posonlyargs + a.args
    first_default = len(positional) - len(a.defaults)
    out = [
        _Param(arg.arg, "po" if i < len(a.posonlyargs) else "pk", i >= first_default)
        for i, arg in enumerate(positional)
    ]
    if a.vararg is not None:
        out.append(_Param(a.vararg.arg, "var", True))
    out += [
        _Param(arg.arg, "kw", default is not None)
        for arg, default in zip(a.kwonlyargs, a.kw_defaults, strict=True)
    ]
    return out[1:] if cls is not None else out


def _lucid_function(name: str, fn: object, stub: Path | None) -> list[_Param] | None:
    """What ``fn`` binds; the stub's spelling where it reads its own arguments."""
    params = _params(_signature(fn), drop_self=False)
    if _catch_all(params) and stub is not None:
        return _stub_params(stub, name, None)
    return params


def _lucid_method(name: str) -> list[_Param] | None:
    """What ``Tensor.<name>`` binds: its own signature, its registry callee's,
    or — for one that reads its own arguments — the stub's."""
    params = _params(_signature(getattr(Tensor, name)), drop_self=True)
    if not _catch_all(params):
        return params
    entry = _BY_METHOD.get(name)
    if entry is not None and entry.n_tensor_args != -1:
        params = _params(ops._signature_for_entry(entry), drop_self=False)[1:]
        if not _catch_all(params):
            return params
    return _stub_params(STUB, name, "Tensor")


def _reference_overloads(
    ref: ModuleType, obj: object, aten: str | None, method: bool
) -> list[list[_Param]]:
    """Every way the reference's parser reads ``obj``."""
    if inspect.isfunction(obj):  # the table is weak-keyed: functions only
        dispatched = ref._jit_internal.boolean_dispatched.get(obj)
        if dispatched is not None:  # ``max_pool2d``: one signature, two returns
            obj = dispatched["if_true"]
    try:
        params = _params(inspect.signature(obj), drop_self=method)  # type: ignore[arg-type]
        if not _catch_all(params):
            return [params]
    except (TypeError, ValueError):
        pass
    if aten is None:
        return []
    try:
        schemas = ref._C._jit_get_schemas_for_operator(f"aten::{aten}")
    except RuntimeError:
        return []
    out = []
    for schema in schemas:
        args = schema.arguments
        # An ``out=`` variant writes to an argument other than ``self``; an
        # in-place one (``add_``) writes ``self`` and is the method's own.
        written = any(
            a.alias_info is not None and a.alias_info.is_write and a.name != "self" for a in args
        )
        if written or any("Dimname" in str(a.type) for a in args):
            continue  # an ``out=`` variant, or a named-dims one
        params = []
        for a in args:
            name = a.name
            if name == "self":
                if method:
                    continue
                name = "input"  # the parser's name for ``self``
            kind = str(a.type).replace("SymInt", "int")
            params.append(
                _Param(
                    name,
                    "kw" if a.kwarg_only else "pk",
                    a.has_default_value(),
                    a.default_value if a.has_default_value() else None,
                    kind.startswith("int[") or kind.startswith("List[int]"),
                )
            )
        out.append(params)
    return out


def _unspellable(want: _Param, got: _Param) -> bool:
    """``want`` is named by a Python keyword (``from``), which a call can pass
    only by position, and ``got`` is that slot, positional-only, as ``from_``."""
    return keyword.iskeyword(want.name) and got.kind == "po" and got.name == f"{want.name}_"


def _mismatches(ours: list[_Param], theirs: list[_Param], method: bool) -> list[str]:
    """How ``ours`` binds a call differently from ``theirs``; empty if it does not."""
    have = {p.name for p in ours}
    ours = [p for p in ours if p.name not in _OMITTED]
    theirs = [
        p
        for p in theirs
        if p.name not in _OMITTED
        and not (p.name.startswith("_") and p.name not in have)
        and not (p.name in _FACTORY and p.name not in have)
    ]
    problems = []
    slots = [p for p in ours if p.kind in ("po", "pk")]
    spread = next((p for p in ours if p.kind == "var"), None)
    wanted = [p for p in theirs if p.kind in ("po", "pk")]
    for i, want in enumerate(wanted):
        if i < len(slots):
            got = slots[i]
            if want.kind == "po" and got.kind != "po":
                problems.append(f"{got.name} is positional-only there")
            elif _unspellable(want, got):
                continue  # ``uniform_(from=...)``: no Python call can name it
            elif want.kind == "pk" and got.name != want.name:
                problems.append(f"slot {i} is {want.name}, not {got.name}")
            elif want.kind == "pk" and got.kind == "po":
                problems.append(f"{got.name} takes a keyword there")
        elif spread is not None and i == len(slots) and want.int_list and (i == 0 or method):
            continue  # ``f(*size)`` — the parser spreads a leading int list
        else:
            problems.append(f"no slot {i} ({want.name})")
    if spread is not None and not any(p.kind == "var" for p in theirs):
        rest = wanted[len(slots) :]
        if not (rest and rest[0].int_list and (len(slots) == 0 or method)):
            problems.append(f"*{spread.name}: the reference takes no varargs here")
    keywords = {p.name for p in theirs if p.kind == "kw"}
    by_name = {p.name: p for p in ours}
    problems += [f"{p.name} is keyword-only there" for p in slots if p.name in keywords]
    problems += [f"no keyword {n}" for n in sorted(keywords - by_name.keys())]
    for want in theirs:
        got = by_name.get(want.name)
        if got is None or got.kind in ("var", "varkw"):
            continue
        if want.optional and not got.optional:
            problems.append(f"{want.name} is optional there")
        elif (
            isinstance(got.default, bool)
            and isinstance(want.default, bool)
            and got.default != want.default
        ):
            problems.append(f"{want.name} defaults to {want.default} there")
    return problems


def _public(module: ModuleType, stub: Path | None = None) -> list[str]:
    """Every Lucid function reachable as ``module.<name>``.

    ``__all__`` is not the whole surface: 42 functions reachable as
    ``lucid.<name>`` and declared in its stub — ``amax``, ``softmax``,
    ``allclose``, ``erfc`` ... — were outside it and kept their ``x``.  The
    package's own public attributes are read, and for ``lucid`` the names
    its stub declares too, which covers the lazily loaded ones.
    """
    names = set(getattr(module, "__all__", [])) | set(dir(module))
    if stub is not None:
        tree = ast.parse(stub.read_text(encoding="utf-8"))
        names |= {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    out = []
    for n in sorted(names):
        obj = getattr(module, n, None) if not n.startswith("_") else None
        if (
            obj is None
            or isinstance(obj, type)
            or (isinstance(obj, ModuleType) and n != "compile")
            or not callable(obj)
        ):
            continue
        if str(getattr(obj, "__module__", "")).startswith("lucid"):
            out.append(n)
    return out


def _tensor_methods() -> list[str]:
    out = []
    for name in sorted(dir(Tensor)):
        member = inspect.getattr_static(Tensor, name)
        if name.startswith("_") or isinstance(member, (property, staticmethod, classmethod)):
            continue
        if callable(member):
            out.append(name)
    return out


def _class_method_targets(
    module: ModuleType, classes: list[str], methods: tuple[str, ...]
) -> list[str]:
    out = []
    for cls_name in classes:
        cls = getattr(module, cls_name, None)
        if not isinstance(cls, type):
            continue
        for method in methods:
            if hasattr(cls, method):
                out.append(f"{cls_name}.{method}")
    return out


#: ``(namespace, Lucid names, the reference's module path, aten prefix)``.
_NAMESPACES: list[tuple[str, list[str], str, str | None]] = [
    ("lucid", _public(lucid, Path(lucid.__file__).with_name("__init__.pyi")), "", ""),
    ("Tensor", _tensor_methods(), "Tensor", ""),
    ("nn.functional", _public(lucid.nn.functional), "nn.functional", ""),
    (
        "nn",
        [n for n in lucid.nn.__all__ if isinstance(getattr(lucid.nn, n, None), type)],
        "nn",
        None,
    ),
    ("linalg", _public(lucid.linalg), "linalg", "linalg_"),
    ("fft", _public(lucid.fft), "fft", "fft_"),
    ("special", _public(lucid.special), "special", "special_"),
    (
        "optim",
        _class_method_targets(
            lucid.optim,
            # Skip LBFGS while lucid/optim/lbfgs.py is dirty on main.
            [
                n
                for n in lucid.optim.__all__
                if n != "LBFGS"
                and isinstance(getattr(lucid.optim, n, None), type)
                and issubclass(getattr(lucid.optim, n), lucid.optim.Optimizer)
            ],
            ("__init__", "step", "zero_grad", "state_dict", "load_state_dict"),
        ),
        "optim",
        None,
    ),
    (
        "optim.lr_scheduler",
        _class_method_targets(
            lucid.optim.lr_scheduler,
            [
                n
                for n, obj in sorted(vars(lucid.optim.lr_scheduler).items())
                if not n.startswith("_")
                and isinstance(obj, type)
                and obj.__module__ == "lucid.optim.lr_scheduler"
            ],
            ("__init__", "step", "state_dict", "load_state_dict", "get_last_lr"),
        ),
        "optim.lr_scheduler",
        None,
    ),
    ("autograd", ["backward", "grad"], "autograd", None),
    (
        "autograd.functional",
        ["jacobian", "hessian", "vjp", "jvp", "vhp", "hvp"],
        "autograd.functional",
        None,
    ),
    (
        "amp",
        _class_method_targets(
            lucid.amp,
            ["autocast", "GradScaler"],
            (
                "__init__",
                "scale",
                "unscale_",
                "step",
                "update",
                "get_scale",
                "state_dict",
                "load_state_dict",
            ),
        ),
        "amp",
        None,
    ),
]

#: Signatures that still differ from the reference — strict, so a fix flips
#: its case to XPASS and the entry leaves with it.
_SIGNATURE_GAPS: dict[str, str] = {}


def _signature_cases() -> list[object]:
    cases = []
    for namespace, names, _, _ in _NAMESPACES:
        for name in names:
            key = f"{namespace}.{name}"
            marks = (
                [pytest.mark.xfail(strict=True, reason=_SIGNATURE_GAPS[key])]
                if key in _SIGNATURE_GAPS
                else []
            )
            cases.append(pytest.param(namespace, name, id=key, marks=marks))
    return cases


@pytest.mark.parametrize("namespace,name", _signature_cases())
def test_signatures_bind_as_the_reference_binds(ref: ModuleType, namespace: str, name: str) -> None:
    _, _, ref_path, aten_prefix = next(n for n in _NAMESPACES if n[0] == namespace)
    owner: object = ref
    for part in filter(None, ref_path.split(".")):
        owner = getattr(owner, part)
    if "." in name:
        cls_name, attr_name = name.split(".", 1)
        theirs_cls = getattr(owner, cls_name, None)
        if not isinstance(theirs_cls, type) or not hasattr(theirs_cls, attr_name):
            pytest.skip("the reference has no matching class method")
        module = lucid
        for part in filter(None, ref_path.split(".")):
            module = getattr(module, part)
        ours = _params(_signature(getattr(getattr(module, cls_name), attr_name)), True)
        theirs = getattr(theirs_cls, attr_name)
        method = True
    else:
        theirs = getattr(owner, name, None)
        if theirs is None or (namespace != "nn" and isinstance(theirs, type)):
            pytest.skip("the reference has no function of this name")
        method = namespace in ("Tensor", "nn")
        if namespace == "Tensor":
            ours = _lucid_method(name)
        elif namespace == "nn":
            if not isinstance(theirs, type):
                pytest.skip("the reference's name is not a class")
            ours = _params(_signature(getattr(lucid.nn, name).__init__), True)
            theirs = theirs.__init__
        else:
            lucid_path = "autograd" if namespace == "autograd.functional" else ref_path
            module = lucid
            for part in filter(None, lucid_path.split(".")):
                module = getattr(module, part)
            stub = Path(lucid.__file__).with_name("__init__.pyi") if not lucid_path else None
            ours = _lucid_function(name, getattr(module, name), stub)
    if ours is None:
        pytest.skip("Lucid's signature is a catch-all with no stub")
    aten = None if aten_prefix is None else aten_prefix + name
    overloads = _reference_overloads(ref, theirs, aten, method)
    if not overloads:
        pytest.skip("the reference exposes no signature")
    if name in _OVERLOADED_NAMES:
        overloads = [
            [p._replace(name=_BY_OVERLOAD.get(p.name, p.name)) for p in o] for o in overloads
        ]
    found = [_mismatches(ours, theirs_params, method) for theirs_params in overloads]
    closest = min(found, key=len)
    assert closest == []


def _pair(device: str) -> tuple[Tensor, Tensor]:
    return (
        lucid.tensor([0.5, 2.0, -3.0], device=device),
        lucid.tensor([1.5, -1.0, 2.0], device=device),
    )


_ARITH = [
    e for e in _REGISTRY if e.engine_fn.__qualname__.startswith(_make_arith_adapter.__qualname__)
]


@pytest.mark.parametrize("entry", _ARITH, ids=lambda e: e.name)
def test_arithmetic_ops_take_input_and_other_by_keyword(entry: OpEntry, device: str) -> None:
    def fresh() -> tuple[Tensor, Tensor]:
        a, b = _pair(device)
        if entry.name.startswith(("bitwise", "logical")):
            return a.to(lucid.int32), b.to(lucid.int32)
        return a, b

    a, b = fresh()
    want = (
        getattr(a.clone(), entry.method_name)(b)
        if entry.inplace
        else getattr(lucid, entry.free_fn_name or entry.name)(a, b)
    )
    # ``pow`` names its second operand ``exponent``, as the reference does.
    second = ops._leading_tensor_names(entry)[1]
    if entry.free_fn_name:
        got = getattr(lucid, entry.free_fn_name)(input=a, **{second: b})
        assert got.tolist() == want.tolist()
    if entry.method_name:
        got = getattr(a.clone(), entry.method_name)(**{second: b})
        assert got.tolist() == want.tolist()


@pytest.mark.parametrize(
    "call",
    [
        lambda x, y: (lucid.exp(input=x), lucid.exp(x)),
        lambda x, y: (lucid.matmul(input=x, other=y), lucid.matmul(x, y)),
        lambda x, y: (lucid.cumsum(input=x, dim=0), lucid.cumsum(x, 0)),
        lambda x, y: (
            lucid.take(input=x, index=lucid.tensor([2, 0], device=x.device)),
            lucid.take(x, lucid.tensor([2, 0], device=x.device)),
        ),
        lambda x, y: (lucid.histc(input=x, bins=4), lucid.histc(x, 4)),
        lambda x, y: (lucid.complex(real=x, imag=y), lucid.complex(x, y)),
        lambda x, y: (
            lucid.squeeze(input=x.unsqueeze(0), dim=0),
            lucid.squeeze(x.unsqueeze(0), 0),
        ),
        lambda x, y: (lucid.sum(input=x, dim=0), lucid.sum(x, 0)),
        lambda x, y: (lucid.reshape(input=x, shape=(3,)), lucid.reshape(x, (3,))),
        lambda x, y: (
            lucid.split(tensor=x, split_size_or_sections=2)[0],
            lucid.split(x, 2)[0],
        ),
        lambda x, y: (
            lucid.tensordot(a=x, b=y, dims=1),
            lucid.tensordot(x, y, 1),
        ),
        lambda x, y: (
            lucid.clamp(input=x, min=0.0, max=1.0),
            lucid.clamp(x, 0.0, 1.0),
        ),
        lambda x, y: (
            lucid.gather(input=x, dim=0, index=lucid.tensor([2, 0], device=x.device)),
            lucid.gather(x, 0, lucid.tensor([2, 0], device=x.device)),
        ),
        lambda x, y: (
            lucid.masked_fill(
                input=x,
                mask=lucid.tensor([True, False, True], device=x.device),
                value=0.0,
            ),
            lucid.masked_fill(x, lucid.tensor([True, False, True], device=x.device), 0.0),
        ),
        lambda x, y: (lucid.unique(input=x), lucid.unique(x)),
        lambda x, y: (x.matmul(other=y), x.matmul(y)),
        lambda x, y: (x.isclose(other=y), x.isclose(y)),
        lambda x, y: (x.add(other=2), x.add(2)),
    ],
    ids=[
        "exp",
        "matmul",
        "cumsum",
        "take",
        "histc",
        "complex",
        "squeeze",
        "sum",
        "reshape",
        "split",
        "tensordot",
        "clamp",
        "gather",
        "masked_fill",
        "unique",
        "Tensor.matmul",
        "Tensor.isclose",
        "Tensor.add",
    ],
)
def test_advertised_tensor_keywords_bind(
    call: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]], device: str
) -> None:
    x, y = _pair(device)
    got, want = call(x, y)
    assert got.tolist() == want.tolist()


def test_every_registry_free_fn_advertises_leading_tensor_names() -> None:
    for entry in _REGISTRY:
        fn_name = entry.free_fn_name
        if not fn_name or entry.n_tensor_args <= 0:
            continue
        fn = getattr(ops, fn_name)
        sig = _signature(fn)
        leading = ops._leading_tensor_names(entry)
        sig_leading = tuple(list(sig.parameters)[: len(leading)])
        assert (
            leading == sig_leading
        ), f"{fn_name}: _leading_tensor_names={leading} != signature={sig_leading}"
        assert not (
            {"x_impl", "a_impl", "b_impl", "y_impl"} & set(leading)
        ), f"{fn_name} advertises internal parameter name: {leading}"


def test_a_tensor_operand_given_twice_or_skipped_is_refused() -> None:
    x, y = _pair("cpu")
    with pytest.raises(TypeError, match="multiple values for argument 'input'"):
        lucid.add(x, input=y)
    with pytest.raises(TypeError, match="missing required argument 'input'"):
        lucid.add(other=y)


def test_atan2_reads_input_over_other(ref: ModuleType, device: str) -> None:
    # The engine names its operands (y, x), which the signature used to
    # advertise as (other, input): the order was the reference's, the names
    # were swapped, and a keyword call could not bind at all.
    x, y = _pair(device)
    assert list(inspect.signature(lucid.atan2).parameters) == ["input", "other"]
    want = ref.atan2(ref.tensor(x.tolist()), ref.tensor(y.tolist())).tolist()
    for got in (lucid.atan2(x, y), lucid.atan2(input=x, other=y), x.atan2(other=y)):
        assert got.tolist() == pytest.approx(want, rel=1e-6)


def test_lcd412_ops_and_tensor_keywords_match_reference(ref: ModuleType, device: str) -> None:
    x = lucid.tensor([1.0, 2.0, 3.0], device=device)
    y = lucid.tensor([0.5, -1.0, 2.0], device=device)
    assert lucid.add(x, y, alpha=2.0).tolist() == pytest.approx([2.0, 0.0, 7.0])
    assert lucid.sub(x, y, alpha=2.0).tolist() == pytest.approx([0.0, 4.0, -1.0])
    xc = x.clone()
    xc.add_(y, alpha=2.0)
    assert xc.tolist() == pytest.approx([2.0, 0.0, 7.0])
    xc.sub_(y, alpha=2.0)
    assert xc.tolist() == pytest.approx([1.0, 2.0, 3.0])

    r = lucid.tensor([1.2345, -2.555, 125.0], device=device)
    assert lucid.round(r, decimals=2).tolist() == pytest.approx([1.23, -2.56, 125.0])
    assert lucid.round(r, decimals=-1).tolist() == pytest.approx([0.0, -0.0, 120.0])
    rc = r.clone()
    rc_view = rc[:]
    rc.round_(decimals=2)
    assert rc_view.tolist() == pytest.approx([1.23, -2.56, 125.0])

    rep = lucid.tensor([1, 2, 1], device=device)
    assert lucid.repeat_interleave(x, rep, dim=0, output_size=4).tolist() == [
        1.0,
        2.0,
        2.0,
        3.0,
    ]

    seq = lucid.tensor([1.0, 3.0, 5.0], device=device)
    vals = lucid.tensor([3.0, 0.0], device=device)
    s_i32 = lucid.searchsorted(seq, vals, out_int32=True, side="right")
    assert s_i32.dtype == lucid.int32 and s_i32.tolist() == [2, 0]
    b_i32 = lucid.bucketize(vals, seq, out_int32=True, right=True)
    assert b_i32.dtype == lucid.int32 and b_i32.tolist() == [2, 0]

    prep = lucid.tensor([0.0], device=device)
    app = lucid.tensor([6.0], device=device)
    assert x.diff(n=1, dim=0, prepend=prep, append=app).tolist() == [
        1.0,
        1.0,
        1.0,
        3.0,
    ]
    assert lucid.isin(x, lucid.tensor([2.0, 4.0], device=device), assume_unique=True).tolist() == [
        False,
        True,
        False,
    ]

    mat = lucid.tensor([[1.0, 2.0, 4.0], [2.0, 0.0, 1.0]], device=device)
    fw = lucid.tensor([1, 2, 1], device=device)
    aw = lucid.tensor([0.5, 1.5, 1.0], device=device)
    rmat = ref.tensor([[1.0, 2.0, 4.0], [2.0, 0.0, 1.0]])
    rfw = ref.tensor([1, 2, 1])
    raw = ref.tensor([0.5, 1.5, 1.0])
    assert lucid.cov(mat, fweights=fw, aweights=aw).reshape(-1).tolist() == pytest.approx(
        ref.cov(rmat, fweights=rfw, aweights=raw).reshape(-1).tolist(), rel=1e-5
    )

    pts = lucid.tensor([[0.1, 0.2], [0.8, 0.9]], device=device)
    w = lucid.tensor([2.5, 1.5], device=device)
    counts, _ = lucid.histogramdd(pts, bins=2, range=[(0.0, 1.0), (0.0, 1.0)], weight=w)
    assert counts.reshape(-1).tolist() == pytest.approx([2.5, 0.0, 0.0, 1.5])
    rpts = ref.tensor([[0.1, 0.2], [0.8, 0.9]])
    assert lucid.cdist(pts, pts, p=2.0, compute_mode="donot_use_mm_for_euclid_dist").reshape(
        -1
    ).tolist() == pytest.approx(
        ref.cdist(rpts, rpts, p=2.0, compute_mode="donot_use_mm_for_euclid_dist")
        .reshape(-1)
        .tolist(),
        abs=1e-6,
    )
    dl = lucid.from_dlpack(lucid.tensor([1.0, 2.0]), device=device, copy=True)
    assert dl.device.type == device and dl.tolist() == [1.0, 2.0]


def test_lcd412_optim_autograd_and_amp_keywords_match_reference() -> None:
    ctx = lucid.amp.autocast("cpu", dtype=lucid.bfloat16, cache_enabled=False)
    assert ctx.cache_enabled is False

    x = lucid.tensor([2.0, 3.0], requires_grad=True)
    y = lucid.tensor([5.0], requires_grad=True)
    gx, gy = lucid.autograd.grad(
        (x**2).sum(),
        (x, y),
        allow_unused=True,
        materialize_grads=True,
    )
    assert gx.tolist() == [4.0, 6.0] and gy.tolist() == [0.0]

    v = lucid.tensor([[1.0, 0.0], [0.0, 1.0]])
    (bg,) = lucid.autograd.grad(
        x * lucid.tensor([2.0, 3.0]),
        x,
        grad_outputs=v,
        is_grads_batched=True,
    )
    assert bg.tolist() == [[2.0, 0.0], [0.0, 3.0]]

    j = lucid.autograd.jacobian(
        lambda t: t**2,
        lucid.tensor([2.0, 3.0]),
        strategy="forward-mode",
        vectorize=True,
    )
    assert j.tolist() == [[4.0, 0.0], [0.0, 6.0]]
    h = lucid.autograd.hessian(
        lambda t: (t**3).sum(),
        lucid.tensor([1.0, 2.0]),
        outer_jacobian_strategy="reverse-mode",
    )
    assert h.tolist() == [[6.0, 0.0], [0.0, 12.0]]

    p = lucid.nn.Parameter(lucid.tensor([1.0, -2.0]))
    opt = lucid.optim.SGD([p], lr=0.1, momentum=0.8, maximize=True)
    p.grad = lucid.tensor([0.5, -1.0])
    opt.step()
    assert p.tolist() == pytest.approx([1.05, -2.1])

    cyc = lucid.optim.lr_scheduler.CyclicLR(
        opt,
        base_lr=0.01,
        max_lr=0.1,
        step_size_up=2,
        step_size_down=4,
        cycle_momentum=True,
        base_momentum=0.7,
        max_momentum=0.9,
    )
    cyc.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.055)
    assert opt.param_groups[0]["momentum"] == pytest.approx(0.8)

    one = lucid.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=0.1,
        epochs=2,
        steps_per_epoch=5,
        three_phase=True,
        cycle_momentum=True,
    )
    one.step()
    assert 0.004 < opt.param_groups[0]["lr"] < 0.1

    pr = lucid.nn.Parameter(lucid.tensor([2.0]))
    radam = lucid.optim.RAdam([pr], lr=0.1, weight_decay=0.5, decoupled_weight_decay=True)
    pr.grad = lucid.tensor([0.0])
    radam.step()
    assert pr.tolist() == pytest.approx([1.9])
    radam.param_groups[0]["lr"] = 0.2
    pr.grad = lucid.tensor([0.0])
    radam.step()
    assert pr.tolist() == pytest.approx([1.71])


def test_nn_module_forward_parameter_names_match_reference(ref: ModuleType) -> None:
    import lucid.nn.quantized.dynamic as qdynamic

    mismatches: list[str] = []
    suites = [
        ("nn", lucid.nn, ref.nn),
        ("nn.quantized", lucid.nn.quantized, ref.ao.nn.quantized),
        ("nn.quantized.dynamic", qdynamic, ref.ao.nn.quantized.dynamic),
    ]
    for label, mod, rmod in suites:
        for name in getattr(mod, "__all__", ()):
            cls = getattr(mod, name, None)
            rcls = getattr(rmod, name, None)
            if (
                not isinstance(cls, type)
                or not issubclass(cls, lucid.nn.Module)
                or not isinstance(rcls, type)
            ):
                continue
            if cls.forward is lucid.nn.Module.forward or rcls.forward is ref.nn.Module.forward:
                continue
            lsig = _signature(cls.forward)
            rsig = _signature(rcls.forward)
            lp = [p for p in lsig.parameters.values() if p.name != "self"]
            rp = [p for p in rsig.parameters.values() if p.name != "self"]
            if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in lp + rp):
                continue
            k = min(len(lp), len(rp))
            lnames = [p.name for p in lp[:k]]
            rnames = [p.name for p in rp[:k]]
            if lnames != rnames:
                mismatches.append(f"{label}.{name}.forward: {lnames} != {rnames}")
    assert not mismatches, "\n".join(mismatches)


def test_lcd434_signature_parity_stragglers(device: str) -> None:
    import io
    import pickle

    probs = lucid.tensor([0.0, 1.0, 0.0, 1.0], device=device)
    drawn = lucid.bernoulli(probs)
    assert drawn.device == probs.device
    assert drawn.dtype == probs.dtype
    assert drawn.tolist() == [0.0, 1.0, 0.0, 1.0]

    scalar_drawn = lucid.bernoulli(lucid.empty(3, device=device), p=1.0)
    assert scalar_drawn.tolist() == [1.0, 1.0, 1.0]
    assert probs.bernoulli().tolist() == [0.0, 1.0, 0.0, 1.0]
    assert probs.bernoulli(0.0).tolist() == [0.0, 0.0, 0.0, 0.0]

    dst = lucid.empty(4, dtype=lucid.int64, device=device)
    assert dst.bernoulli_(probs) is dst
    assert dst.tolist() == [0, 1, 0, 1]
    assert dst.bernoulli_(0.0).tolist() == [0, 0, 0, 0]

    with pytest.raises(RuntimeError, match=r"p_in >= 0 && p_in <= 1"):
        lucid.bernoulli(lucid.tensor([1.5], device=device))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        lucid.bernoulli(probs, p=-0.1)

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        probs.to(bogus=1)  # type: ignore[call-overload]

    def _fn(x: Tensor) -> Tensor:
        return x + 1.0

    assert lucid.compile(_fn, fullgraph=True, mode="reduce-overhead", disable=True) is _fn
    assert lucid.compile(fullgraph=True, disable=True)(_fn) is _fn

    buf = io.BytesIO()
    cpu_t = lucid.tensor([1.0, 2.0, 3.0])
    lucid.save(cpu_t, buf, pickle, 4)
    buf.seek(0)
    loaded = lucid.load(buf, "cpu", pickle, weights_only=True, mmap=False)
    assert isinstance(loaded, Tensor)
    assert loaded.tolist() == [1.0, 2.0, 3.0]


def test_lcd430_legacy_reduction_and_elu_inplace_scale(device: str) -> None:
    import warnings
    import lucid.nn as nn
    import lucid.nn.functional as F
    from lucid.nn.functional._loss_checks import _legacy_reduction

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _legacy_reduction(None, None, "mean", "mse_loss") == "mean"
        assert _legacy_reduction(False, None, "mean", "mse_loss") == "sum"
        assert _legacy_reduction(True, False, "mean", "mse_loss") == "none"
        assert _legacy_reduction(False, False, "mean", "mse_loss") == "none"
        assert _legacy_reduction(False, True, "mean", "mse_loss") == "sum"
        assert _legacy_reduction(True, True, "sum", "mse_loss") == "mean"
        assert _legacy_reduction(None, None, "batchmean", "kl_div", batchmean=True) == "batchmean"

        x = lucid.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
        y = lucid.tensor([[0.0, 1.0], [1.0, 2.0]], device=device)
        assert F.mse_loss(x, y, False).item() == pytest.approx(
            F.mse_loss(x, y, reduction="sum").item()
        )
        assert F.mse_loss(x, y, True, False).shape == (2, 2)
        assert nn.MSELoss(False)(x, y).item() == pytest.approx(
            nn.MSELoss(reduction="sum")(x, y).item()
        )
        assert nn.MSELoss(True, False)(x, y).shape == (2, 2)
        assert nn.KLDivLoss(size_average=False).reduction == "sum"
        assert nn.CrossEntropyLoss(None, False).reduction == "sum"
        assert nn.NLLLoss(None, None, -100, False).reduction == "none"
        assert nn.PoissonNLLLoss(True, False, False).reduction == "sum"
        assert nn.MultiLabelMarginLoss(reduce=False).reduction == "none"

    t = lucid.tensor([-1.0, 2.0], device=device)
    assert F.elu_(t, 1.0, False) is t
    assert t.tolist() == [0.0, 0.0]
