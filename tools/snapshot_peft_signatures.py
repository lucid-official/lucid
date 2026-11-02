#!/usr/bin/env python3
"""tools/snapshot_peft_signatures.py — the reference library's public signatures, as JSON.

``lucid.peft`` follows the reference adapter library's API name for name, and
that claim rots silently: a parameter renamed, an argument moved, a default
changed — nothing fails until a user's script does.  This tool reads the
reference library's *source* (never imports it) and writes every exported
name's signature to a JSON file that ``lucid/test/unit/peft/
test_reference_signatures.py`` holds ``lucid.peft`` to.

Usage::

    pip download peft==X.Y.Z --no-deps -d /tmp/peftsig/whl
    unzip -q /tmp/peftsig/whl/peft-*.whl 'peft/*' -d /tmp/peftsig/src
    python3 -I tools/snapshot_peft_signatures.py /tmp/peftsig/src \\
        lucid/test/unit/peft/_reference_signatures.json

The extracted source is untrusted data: it is parsed with ``ast`` only and is
never imported or executed.  Re-run on a newer wheel and review the JSON diff
to see exactly what the reference changed.

What is recorded
----------------
* every name in the reference ``peft/__init__.py`` ``__all__``, resolved
  through the chain of ``from .x import y`` re-exports to its defining module;
* for a function: its parameters;
* for a class: the dataclass fields (including those inherited along the
  base-class chain inside the reference package, in dataclass order), the
  ``__init__`` signature, and every public method / property (own or
  inherited from a base defined in the package);
* for an ``Enum``: its members.

A parameter is ``[name, kind, has_default, default_repr]``; ``kind`` is one of
``positional_only``, ``positional_or_keyword``, ``var_positional``,
``keyword_only``, ``var_keyword`` (the ``inspect.Parameter`` kinds), and
``default_repr`` is the unparsed default expression (``None`` without one).
``self`` / ``cls`` are dropped.
"""

import ast
import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

PACKAGE = "peft"

POS_ONLY = "positional_only"
POS_OR_KW = "positional_or_keyword"
VAR_POS = "var_positional"
KW_ONLY = "keyword_only"
VAR_KW = "var_keyword"

_FACTORY = "<factory>:"


def _param(name: str, kind: str, default: ast.expr | None) -> list[object]:
    return [
        name,
        kind,
        default is not None,
        None if default is None else ast.unparse(default),
    ]


def parameters(args: ast.arguments, *, drop_first: bool) -> list[list[object]]:
    """The parameters of a ``def`` as ``[name, kind, has_default, default_repr]``."""
    out: list[list[object]] = []
    positional = [*args.posonlyargs, *args.args]
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(args.defaults))
    defaults.extend(args.defaults)
    for i, (arg, default) in enumerate(zip(positional, defaults, strict=True)):
        kind = POS_ONLY if i < len(args.posonlyargs) else POS_OR_KW
        out.append(_param(arg.arg, kind, default))
    if drop_first and out:
        out.pop(0)
    if args.vararg is not None:
        out.append(_param(args.vararg.arg, VAR_POS, None))
    for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
        out.append(_param(arg.arg, KW_ONLY, default))
    if args.kwarg is not None:
        out.append(_param(args.kwarg.arg, VAR_KW, None))
    return out


def _decorator_names(node: ast.FunctionDef | ast.ClassDef) -> set[str]:
    names: set[str] = set()
    for deco in node.decorator_list:
        target = deco.func if isinstance(deco, ast.Call) else deco
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, ast.Attribute):
            names.add(target.attr)
    return names


@dataclass
class Module:
    """A parsed reference module and the names it imports."""

    dotted: str
    tree: ast.Module
    is_package: bool
    # local name -> (source dotted module, original name | None for `import x`)
    imports: dict[str, tuple[str, str]] = field(default_factory=dict)
    star_imports: list[str] = field(default_factory=list)

    def own(self, name: str) -> ast.AST | None:
        """The last top-level definition of ``name`` in this module."""
        found: ast.AST | None = None
        for node in _walk_toplevel(self.tree.body):
            if isinstance(node, ast.FunctionDef | ast.ClassDef) and node.name == name:
                found = node
            elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets
            ):
                found = node
            elif (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == name
            ):
                found = node
        return found


def _walk_toplevel(body: list[ast.stmt]) -> Iterator[ast.stmt]:
    """Top-level statements, looking inside ``if`` / ``try`` blocks."""
    for node in body:
        yield node
        if isinstance(node, ast.If):
            yield from _walk_toplevel(node.body)
            yield from _walk_toplevel(node.orelse)
        elif isinstance(node, ast.Try):
            yield from _walk_toplevel(node.body)
            for handler in node.handlers:
                yield from _walk_toplevel(handler.body)


class Reference:
    """A parsed copy of the reference package: symbol and base-class resolution."""

    def __init__(self, src: Path) -> None:
        self._root = src / PACKAGE
        self._modules: dict[str, Module | None] = {}

    def module(self, dotted: str) -> Module | None:
        if dotted in self._modules:
            return self._modules[dotted]
        rel = Path(*dotted.split("."))
        path = self._root.parent / rel
        mod: Module | None = None
        if (path / "__init__.py").is_file():
            mod = self._parse(dotted, path / "__init__.py", is_package=True)
        elif path.with_suffix(".py").is_file():
            mod = self._parse(dotted, path.with_suffix(".py"), is_package=False)
        self._modules[dotted] = mod
        return mod

    def _parse(self, dotted: str, path: Path, *, is_package: bool) -> Module:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        mod = Module(dotted, tree, is_package)
        package = dotted if is_package else dotted.rpartition(".")[0]
        for node in _walk_toplevel(tree.body):
            if not isinstance(node, ast.ImportFrom):
                continue
            source = self._import_source(package, node)
            if source is None:
                continue
            for alias in node.names:
                if alias.name == "*":
                    mod.star_imports.append(source)
                else:
                    mod.imports[alias.asname or alias.name] = (source, alias.name)
        return mod

    @staticmethod
    def _import_source(package: str, node: ast.ImportFrom) -> str | None:
        if node.level == 0:
            name = node.module or ""
            return name if name == PACKAGE or name.startswith(PACKAGE + ".") else None
        parts = package.split(".")
        base = parts[: len(parts) - (node.level - 1)]
        return ".".join([*base, *([node.module] if node.module else [])])

    def resolve(
        self, dotted: str, name: str, _seen: frozenset[tuple[str, str]] = frozenset()
    ) -> tuple[Module, ast.AST] | None:
        """Follow ``name`` imported into ``dotted`` back to its defining node."""
        if (dotted, name) in _seen:
            return None
        seen = _seen | {(dotted, name)}
        mod = self.module(dotted)
        if mod is None:
            return None
        node = mod.own(name)
        if node is not None:
            return mod, node
        if name in mod.imports:
            source, original = mod.imports[name]
            return self.resolve(source, original, seen)
        for star in mod.star_imports:
            hit = self.resolve(star, name, seen)
            if hit is not None:
                return hit
        return None


@dataclass
class ClassInfo:
    """A reference class: its node, defining module and package-internal bases."""

    module: Module
    node: ast.ClassDef
    bases: list["ClassInfo"]

    @property
    def is_dataclass(self) -> bool:
        return "dataclass" in _decorator_names(self.node)

    @property
    def is_enum(self) -> bool:
        names = {b.id for b in self.node.bases if isinstance(b, ast.Name)}
        names |= {b.attr for b in self.node.bases if isinstance(b, ast.Attribute)}
        return bool(names & {"Enum", "IntEnum", "StrEnum"}) or any(b.is_enum for b in self.bases)

    def mro(self) -> list["ClassInfo"]:
        """Depth-first, left-to-right linearisation (enough for these hierarchies)."""
        out: list[ClassInfo] = [self]
        for base in self.bases:
            for c in base.mro():
                if all(c.node is not o.node for o in out):
                    out.append(c)
        return out


class Snapshotter:
    def __init__(self, ref: Reference) -> None:
        self.ref = ref
        self._classes: dict[tuple[str, str], ClassInfo] = {}

    def class_info(self, mod: Module, node: ast.ClassDef) -> ClassInfo:
        key = (mod.dotted, node.name)
        if key in self._classes:
            return self._classes[key]
        bases: list[ClassInfo] = []
        info = ClassInfo(mod, node, bases)
        self._classes[key] = info
        for base in node.bases:
            if not isinstance(base, ast.Name):
                continue
            hit = self.ref.resolve(mod.dotted, base.id)
            if hit is not None and isinstance(hit[1], ast.ClassDef):
                bases.append(self.class_info(hit[0], hit[1]))
        return info

    # -- classes -----------------------------------------------------------

    def fields(self, info: ClassInfo) -> list[list[object]]:
        """Dataclass fields in dataclass order, bases first."""
        ordered: dict[str, list[object]] = {}
        for base in reversed(info.bases):
            if base.is_dataclass:
                for row in self.fields(base):
                    ordered[str(row[0])] = row
        if not info.is_dataclass:
            return list(ordered.values())
        for stmt in info.node.body:
            if not (isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)):
                continue
            if "ClassVar" in ast.unparse(stmt.annotation):
                continue
            declared = self._field_row(stmt)
            if declared is not None:
                ordered[str(declared[0])] = declared
        return list(ordered.values())

    @staticmethod
    def _field_row(stmt: ast.AnnAssign) -> list[object] | None:
        assert isinstance(stmt.target, ast.Name)
        name = stmt.target.id
        value = stmt.value
        kind = POS_OR_KW
        default: str | None = None
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name | ast.Attribute)
            and ast.unparse(value.func).split(".")[-1] == "field"
        ):
            kwargs = {k.arg: k.value for k in value.keywords if k.arg}
            init = kwargs.get("init")
            if isinstance(init, ast.Constant) and init.value is False:
                return None
            kw_only = kwargs.get("kw_only")
            if isinstance(kw_only, ast.Constant) and kw_only.value is True:
                kind = KW_ONLY
            if "default" in kwargs:
                default = ast.unparse(kwargs["default"])
            elif "default_factory" in kwargs:
                default = _FACTORY + ast.unparse(kwargs["default_factory"])
        elif value is not None:
            default = ast.unparse(value)
        return [name, kind, default is not None, default]

    def methods(self, info: ClassInfo) -> dict[str, dict[str, object]]:
        """Public callables of a class, nearest definition along the MRO winning."""
        out: dict[str, dict[str, object]] = {}
        for cls in info.mro():
            defs = self._own_methods(cls)
            for name, entry in defs.items():
                out.setdefault(name, entry)
        return out

    def _own_methods(self, cls: ClassInfo) -> dict[str, dict[str, object]]:
        out: dict[str, dict[str, object]] = {}
        for stmt in cls.node.body:
            if not isinstance(stmt, ast.FunctionDef):
                continue
            name = stmt.name
            if name.startswith("_") and name not in ("__init__", "__call__"):
                continue
            decos = _decorator_names(stmt)
            if "overload" in decos:
                continue
            if "setter" in decos or "deleter" in decos:
                if name in out:
                    out[name]["has_setter"] = True
                continue
            if "property" in decos or "cached_property" in decos:
                kind = "property"
            elif "classmethod" in decos:
                kind = "classmethod"
            elif "staticmethod" in decos:
                kind = "staticmethod"
            else:
                kind = "method"
            out[name] = {
                "callable": kind,
                "defined_in": f"{cls.module.dotted}.{cls.node.name}",
                "params": parameters(
                    stmt.args, drop_first=kind in ("method", "classmethod", "property")
                ),
            }
        return out

    @staticmethod
    def enum_members(info: ClassInfo) -> list[list[str]]:
        out: list[list[str]] = []
        for stmt in info.node.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target = stmt.targets[0]
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    out.append([target.id, ast.unparse(stmt.value)])
        return out


def exports(ref: Reference) -> list[str]:
    """The reference ``__all__``."""
    mod = ref.module(PACKAGE)
    assert mod is not None, "peft/__init__.py not found under the source root"
    for node in mod.tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets)
            and isinstance(node.value, ast.List)
        ):
            return [
                e.value
                for e in node.value.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)
            ]
    raise SystemExit("no __all__ list in peft/__init__.py")


def version(ref: Reference) -> str:
    mod = ref.module(PACKAGE)
    assert mod is not None
    for node in mod.tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets)
            and isinstance(node.value, ast.Constant)
        ):
            return str(node.value.value)
    return "unknown"


class Snapshot(TypedDict):
    """The JSON the tool writes."""

    peft_version: str
    exports: list[str]
    kinds: dict[str, str]
    fields: dict[str, list[list[object]]]
    enum_members: dict[str, list[list[str]]]
    signatures: dict[str, dict[str, object]]


def snapshot(src: Path) -> Snapshot:
    ref = Reference(src)
    snap = Snapshotter(ref)
    kinds: dict[str, str] = {}
    signatures: dict[str, dict[str, object]] = {}
    fields: dict[str, list[list[object]]] = {}
    enums: dict[str, list[list[str]]] = {}
    names = exports(ref)
    for name in names:
        hit = ref.resolve(PACKAGE, name)
        if hit is None:
            kinds[name] = "unresolved"
            continue
        mod, node = hit
        if isinstance(node, ast.FunctionDef):
            kinds[name] = "function"
            signatures[name] = {
                "callable": "function",
                "defined_in": f"{mod.dotted}.{name}",
                "params": parameters(node.args, drop_first=False),
            }
        elif isinstance(node, ast.ClassDef):
            info = snap.class_info(mod, node)
            if info.is_enum:
                kinds[name] = "enum"
                enums[name] = snap.enum_members(info)
                continue
            kinds[name] = "class"
            if info.is_dataclass:
                fields[name] = snap.fields(info)
            for member, entry in snap.methods(info).items():
                signatures[f"{name}.{member}"] = entry
        else:
            kinds[name] = "constant"
    return Snapshot(
        peft_version=version(ref),
        exports=names,
        kinds=kinds,
        fields=fields,
        enum_members=enums,
        signatures=signatures,
    )


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    src, out = Path(argv[1]), Path(argv[2])
    data = snapshot(src)
    out.write_text(json.dumps(data, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    unresolved = [n for n, k in data["kinds"].items() if k == "unresolved"]
    print(
        f"peft {data['peft_version']}: {len(data['exports'])} exports,"
        f" {len(data['signatures'])} signatures -> {out}"
    )
    if unresolved:
        print(f"unresolved: {unresolved}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
