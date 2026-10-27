#!/usr/bin/env python3
"""tools/check_docstring_xrefs.py — flag broken sphinx-style cross-references.

The docs site renders ``:func:`lucid.linalg.norm```, ``:class:`Tensor```,
``:meth:`Module.forward```, etc. as inline references.  When the target
gets renamed or removed, the reference is silently left dangling — the
docs page shows ``lucid.linalg.norm`` as plain text and nobody notices
until a reader clicks and lands on a 404.

This linter walks the emitted API JSONs (which contain every
documentable symbol path) and the lucid Python sources (for the raw
docstring text), extracts every ``:role:`target``` reference, and
reports targets that can't be resolved to an emitted symbol.

Run AFTER ``web/scripts/build-api-data.py`` populates
``web/public/api-data/``::

    python tools/check_docstring_xrefs.py           # exits 1 on any unresolved ref
    python tools/check_docstring_xrefs.py --list    # show every unresolved ref

Resolution policy
-----------------
A reference resolves when its target is either:
  1. A fully-qualified path that matches an emitted symbol's ``path``
     (e.g. ``lucid.linalg.norm`` → ``lucid.linalg.norm``), or
  2. A short name uniquely matching the basename of some emitted
     symbol (e.g. ``dropout`` → ``lucid.nn.functional.dropout``), or
  3. A class-qualified path that matches a known Tensor method
     (``Tensor.matmul`` → ``lucid.tensor::matmul``).

Exit codes
----------
0 — every reference resolves.
1 — at least one reference is unresolved; details printed to stderr.
"""

import argparse
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LUCID_SRC = ROOT / "lucid"
API_DATA_DIR = ROOT / "web" / "public" / "api-data"

# Sphinx domain roles we care about — any reference written as
# ``:role:`target``` with one of these role names.  Other roles
# (``:py:obj:``, ``:any:``, …) are silently ignored.
_XREF_ROLES = {"func", "meth", "class", "attr", "data", "mod", "obj"}

_XREF_RE = re.compile(r":(" + "|".join(_XREF_ROLES) + r"):`(?:~)?([^`<>\s]+)(?:\s*<[^>]+>)?`")

# Docstring fields the build emits.  ``parameters`` / ``raises`` /
# ``attributes`` / ``returns`` carry their own embedded text, so we
# recurse into them.
_TEXT_FIELDS = ("summary", "extended")
_LIST_TEXT_FIELDS = ("examples", "notes")
_NESTED_DESCRIPTION_FIELDS = ("parameters", "raises", "attributes", "warns")


def _collect_emitted_paths() -> set[str]:
    """Union of every ``path`` value across the API JSON tree.

    Walks each ``lucid.*.json`` recursively (modules contain members
    that themselves contain members for classes).  Includes synthetic
    paths the docs site generates for class methods (``lucid.tensor::name``).
    """
    paths: set[str] = set()

    def walk(node: dict[str, object]) -> None:
        p = node.get("path")
        if isinstance(p, str):
            paths.add(p)
        members = node.get("members")
        if isinstance(members, list):
            for m in members:
                if isinstance(m, dict):
                    walk(m)
        methods = node.get("methods")
        if isinstance(methods, list):
            owner_path = node.get("path") if isinstance(node.get("path"), str) else ""
            for m in methods:
                if isinstance(m, dict):
                    walk(m)
                    name = m.get("name")
                    # Class-method dual paths so :meth:`Tensor.foo` resolves
                    # via either ``lucid.tensor.foo`` or ``lucid.Tensor.foo``.
                    if isinstance(name, str) and isinstance(owner_path, str):
                        paths.add(f"{owner_path}.{name}")
                        # Friendly form: ``Tensor.foo``.
                        if owner_path.endswith(".tensor"):
                            paths.add(f"Tensor.{name}")

    for p in sorted(API_DATA_DIR.glob("*.json")):
        if p.name.startswith("_"):
            continue
        try:
            walk(json.loads(p.read_text()))
        except Exception:
            continue
    return paths


def _basename_index(paths: set[str]) -> dict[str, list[str]]:
    """Map ``last-segment → [full paths]`` for short-name resolution."""
    idx: dict[str, list[str]] = {}
    for full in paths:
        # Use ``.`` and ``::`` as segment separators (the docs site
        # uses ``::`` for the Tensor class-method namespace).
        last = re.split(r"[.::]", full)[-1]
        idx.setdefault(last, []).append(full)
    return idx


# Python builtins / stdlib / third-party names that show up as
# ``:class:`...``` or ``:mod:`...``` targets but live outside the Lucid
# surface.  Resolving these here keeps the linter focused on actual
# Lucid-side typos.
_PY_STDLIB_NAMES: set[str] = {
    # Exceptions
    "Exception",
    "BaseException",
    "TypeError",
    "ValueError",
    "RuntimeError",
    "NotImplementedError",
    "IndexError",
    "KeyError",
    "AttributeError",
    "AssertionError",
    "ArithmeticError",
    "OverflowError",
    "ZeroDivisionError",
    "FloatingPointError",
    "FileNotFoundError",
    "OSError",
    "IOError",
    "StopIteration",
    "StopAsyncIteration",
    "GeneratorExit",
    "ImportError",
    "ModuleNotFoundError",
    "MemoryError",
    "NameError",
    "UnboundLocalError",
    "RecursionError",
    # typing / collections.abc
    "Callable",
    "Iterable",
    "Iterator",
    "Sequence",
    "Mapping",
    "Generator",
    "Optional",
    "Union",
    "Any",
    "Self",
    "TypeVar",
    "ParamSpec",
    # Builtins
    "list",
    "dict",
    "set",
    "tuple",
    "frozenset",
    "str",
    "bytes",
    "int",
    "float",
    "complex",
    "bool",
    "object",
    "type",
    "slice",
    "range",
    "None",
    "Ellipsis",
    "True",
    "False",
    # stdlib modules referenced via :mod:
    "abc",
    "ast",
    "asyncio",
    "collections",
    "concurrent",
    "contextlib",
    "copy",
    "ctypes",
    "dataclasses",
    "datetime",
    "enum",
    "functools",
    "hashlib",
    "io",
    "itertools",
    "json",
    "logging",
    "math",
    "multiprocessing",
    "operator",
    "os",
    "pathlib",
    "pickle",
    "platform",
    "queue",
    "random",
    "re",
    "shutil",
    "signal",
    "socket",
    "ssl",
    "struct",
    "subprocess",
    "sys",
    "tempfile",
    "textwrap",
    "threading",
    "time",
    "traceback",
    "typing",
    "unittest",
    "urllib",
    "uuid",
    "warnings",
    "weakref",
    "zipfile",
    # numpy / MLX bridge symbols — H4 allows numpy only at boundary; MLX
    # is the GPU stream backend.
    "ndarray",
    "np.ndarray",
    "numpy.ndarray",
    "numpy",
    "np",
    "mlx",
    "mlx.core",
    "mlx.core.array",
    # C++ / Metal Performance Shaders types referenced from compile/
    # — engine internals, not Python-side symbols.
    "MPSGraph",
    "MPSGraphExecutable",
    "MTLDevice",
    "MTLBuffer",
    "MTLResourceStorageModeShared",
    "MpsBuilder",
    "CompiledStepBackward",
    "compile_trace_with_backward",
    "_C_engine",
    # Builtin warning categories.
    "UserWarning",
    "DeprecationWarning",
    "PendingDeprecationWarning",
    "SyntaxWarning",
    "RuntimeWarning",
    "FutureWarning",
    "ImportWarning",
    "UnicodeWarning",
    "BytesWarning",
    "ResourceWarning",
}


def _collect_source_symbols() -> set[str]:
    """Every public class / function name defined anywhere under lucid/.

    Used as a fallback for refs that point to internal symbols the docs
    site doesn't expose (e.g. ``lucid._globals`` private helpers that
    are still real Python objects).  Walks the AST once per file —
    cheap given lucid is ~500 files.

    Also harvests names that are added to the lucid namespace
    dynamically — OpEntry registry generates ``lucid.add`` /
    ``lucid.median`` etc. at import time, dtype singletons
    (``lucid.float32``) are class instances assigned at module scope,
    and the lazy ``__getattr__`` loader pulls composite names from
    submodules.  Those aren't visible as ``ast.FunctionDef`` nodes, so
    we parse the relevant tables out of ``lucid/__init__.py`` and
    ``lucid/_ops/_registry.py`` separately.
    """
    names: set[str] = set()
    # Dynamic-loader name sets in lucid/__init__.py — _FACTORY_NAMES,
    # _OPS_NAMES, _SCATTER_NAMES, _GRAD_NAMES, _PREDICATE_NAMES,
    # _SERIALIZATION_NAMES, _TYPE_ALIAS_NAMES.  Each is a frozenset
    # literal; harvest its string contents.
    try:
        init_tree = ast.parse((LUCID_SRC / "__init__.py").read_text())
        for node in ast.walk(init_tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                tgt = node.targets[0]
                if (
                    isinstance(tgt, ast.Name)
                    and tgt.id.startswith("_")
                    and tgt.id.endswith("_NAMES")
                ):
                    try:
                        val = node.value
                        if (
                            isinstance(val, ast.Call)
                            and getattr(val.func, "id", None) == "frozenset"
                            and val.args
                        ):
                            val = val.args[0]
                        for s in ast.literal_eval(val):
                            if isinstance(s, str):
                                names.add(s)
                                names.add(f"lucid.{s}")
                    except Exception:
                        pass
    except Exception:
        pass
    # OpEntry registry — every ``OpEntry("name", ...)`` adds a
    # ``lucid.<name>`` free function via _populate_free_fns.
    try:
        reg_tree = ast.parse((LUCID_SRC / "_ops" / "_registry.py").read_text())
        for node in ast.walk(reg_tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "OpEntry"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                n = node.args[0].value
                names.add(n)
                names.add(f"lucid.{n}")
    except Exception:
        pass
    # Composite ops — each submodule's __all__ contributes lazy-loaded names.
    for comp in (LUCID_SRC / "_ops" / "composite").glob("*.py"):
        if comp.name == "__init__.py":
            continue
        try:
            tree = ast.parse(comp.read_text())
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "__all__"
                ):
                    try:
                        for s in ast.literal_eval(node.value):
                            if isinstance(s, str):
                                names.add(s)
                                names.add(f"lucid.{s}")
                    except Exception:
                        pass
        except Exception:
            pass
    for src in LUCID_SRC.rglob("*.py"):
        try:
            tree = ast.parse(src.read_text())
        except Exception:
            continue
        # Module dotted path (``lucid._globals``).  Source files inside
        # the package mirror import paths 1-to-1.
        rel = src.relative_to(LUCID_SRC.parent).with_suffix("")
        mod_path = ".".join(rel.parts)
        if mod_path.endswith(".__init__"):
            mod_path = mod_path[: -len(".__init__")]
        names.add(mod_path)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
                names.add(f"{mod_path}.{node.name}")
                # Class methods — surface bare method names so
                # ``:meth:`forward``` resolves regardless of which class.
                # Also catch class attributes (``arg_constraints = ...``)
                # so ``:attr:`Distribution.arg_constraints``` resolves.
                # ``register_buffer("name", …)`` / ``register_parameter``
                # calls inside ``__init__`` add real attributes that
                # don't appear as ``ast.Assign`` targets — harvest them
                # explicitly.
                if isinstance(node, ast.ClassDef):
                    for child in node.body:
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            names.add(child.name)
                            names.add(f"{mod_path}.{node.name}.{child.name}")
                            names.add(f"{node.name}.{child.name}")
                            # Walk method body for register_buffer /
                            # register_parameter calls.  These add
                            # attributes whose names live in the first
                            # positional string argument.
                            for sub in ast.walk(child):
                                if (
                                    isinstance(sub, ast.Call)
                                    and isinstance(sub.func, ast.Attribute)
                                    and sub.func.attr in ("register_buffer", "register_parameter")
                                    and sub.args
                                    and isinstance(sub.args[0], ast.Constant)
                                    and isinstance(sub.args[0].value, str)
                                ):
                                    attr_name = sub.args[0].value
                                    names.add(attr_name)
                                    names.add(f"{node.name}.{attr_name}")
                        elif isinstance(child, (ast.Assign, ast.AnnAssign)):
                            tgts = (
                                child.targets if isinstance(child, ast.Assign) else [child.target]
                            )
                            for tgt in tgts:
                                if isinstance(tgt, ast.Name):
                                    names.add(tgt.id)
                                    names.add(f"{node.name}.{tgt.id}")
            elif isinstance(node, ast.Assign):
                # Top-level constants / aliases (``float32 = dtype(...)``).
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        names.add(tgt.id)
                        names.add(f"{mod_path}.{tgt.id}")
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                # ``from lucid._dtype import float32`` makes ``float32``
                # an attribute of the importing module.  Without this,
                # ``:data:`lucid.float32``` doesn't resolve even though
                # the dtype singleton is a real public attribute.
                for alias in node.names:
                    name = alias.asname or alias.name
                    if name == "*":
                        continue
                    names.add(name)
                    names.add(f"{mod_path}.{name}")
    return names


def _iter_docstring_text(node: dict[str, object]):
    """Yield every chunk of free-text inside one docstring payload."""
    for key in _TEXT_FIELDS:
        v = node.get(key)
        if isinstance(v, str) and v:
            yield v
    for key in _LIST_TEXT_FIELDS:
        v = node.get(key)
        if isinstance(v, list):
            for item in v:
                if isinstance(item, str) and item:
                    yield item
    for key in _NESTED_DESCRIPTION_FIELDS:
        v = node.get(key)
        if isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    d = item.get("description")
                    if isinstance(d, str) and d:
                        yield d
    ret = node.get("returns")
    if isinstance(ret, dict):
        d = ret.get("description")
        if isinstance(d, str) and d:
            yield d


def _resolve(
    target: str, paths: set[str], basename_idx: dict[str, list[str]], source_names: set[str]
) -> bool:
    """``True`` when ``target`` matches at least one known symbol.

    Resolution attempts, in order:

    1. Targets containing unicode ellipsis, parens, or other
       non-identifier punctuation — treat as prose placeholders and
       skip silently.
    2. Stdlib-rooted dotted paths (``json.dumps``, ``os.path``) — the
       module prefix is whitelisted, so the member is assumed valid.
    3. Emitted JSON ``path`` value (docs-visible exact match).
    4. Python source AST: top-level defs / classes / class attrs /
       imports + lazy-loader name sets harvested from ``__init__.py``
       and ``_ops/_registry.py``.
    5. Suffix on an emitted path (``norm`` ↔ ``lucid.linalg.norm``).
    6. Bare-name basename match against ``source_names``.
    """
    # Prose placeholders — not real references.
    if any(c in target for c in "…()[] "):
        return True
    # ``tools.convert_weights.dit`` and its siblings are modules in this
    # repository, and the index above walks ``lucid/`` only. Checked on
    # disk rather than whitelisted, so a reference to a converter that
    # was renamed or deleted still fails.
    if target.startswith("tools."):
        return (ROOT / Path(*target.split("."))).with_suffix(".py").exists() or (
            ROOT / Path(*target.split("."))
        ).is_dir()
    # An engine symbol is emitted under its C++ name — ``lucid::TensorImpl``
    # — while a docstring naturally spells the Python path it is reached
    # by. Both name the same class, and the docs site links either.
    if target.startswith("lucid._C.engine."):
        leaf = target.rsplit(".", 1)[-1]
        if f"lucid::{leaf}" in paths or leaf in paths:
            return True
    # Module-prefix whitelist: ``json.dumps`` resolves via ``json``.
    if "." in target:
        prefix = target.split(".", 1)[0]
        if prefix in _PY_STDLIB_NAMES:
            return True
    if target in paths:
        return True
    if target in source_names:
        return True
    if target in _PY_STDLIB_NAMES:
        return True
    suffix_hits = [p for p in paths if p.endswith("." + target) or p.endswith("::" + target)]
    if suffix_hits:
        return True
    if target in basename_idx:
        return True
    # Last fallback: bare basename match against source symbols
    # (``set_default_dtype`` resolves via the function def even when it
    # lives in a non-emitted module).
    leaf = re.split(r"[.::]", target)[-1]
    if leaf in source_names:
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument(
        "--list", action="store_true", help="print every unresolved reference (default: first 20)"
    )
    args = ap.parse_args()

    if not API_DATA_DIR.is_dir():
        sys.stderr.write(
            f"[check_docstring_xrefs] {API_DATA_DIR} not found — "
            "run ``pnpm build:api`` (or ``python web/scripts/build-api-data.py``) first.\n"
        )
        return 1

    paths = _collect_emitted_paths()
    if not paths:
        sys.stderr.write("[check_docstring_xrefs] no emitted symbols — api-data dir is empty.\n")
        return 1
    basename_idx = _basename_index(paths)
    source_names = _collect_source_symbols()

    # Per-source-file scan of raw Python so we can blame each unresolved
    # ref on its exact site, not just the docs page that surfaced it.
    unresolved: list[tuple[str, str, str, str]] = []
    for src in sorted(LUCID_SRC.rglob("*.py")):
        rel = src.relative_to(ROOT)
        if any(seg in rel.parts for seg in ("test", "benchmarks")):
            continue
        try:
            tree = ast.parse(src.read_text())
        except Exception:
            continue
        for node in ast.walk(tree):
            if not isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)
            ):
                continue
            doc = ast.get_docstring(node) or ""
            for m in _XREF_RE.finditer(doc):
                role = m.group(1)
                target = m.group(2)
                if not _resolve(target, paths, basename_idx, source_names):
                    sym = getattr(node, "name", "<module>")
                    unresolved.append((str(rel), sym, role, target))

    if not unresolved:
        print(
            f"[check_docstring_xrefs] OK — every :func:/:class:/:meth: ref resolves "
            f"({len(paths)} symbols indexed)."
        )
        return 0

    limit = len(unresolved) if args.list else min(20, len(unresolved))
    sys.stderr.write(f"[check_docstring_xrefs] {len(unresolved)} unresolved reference(s):\n")
    for rel, sym, role, target in unresolved[:limit]:
        sys.stderr.write(f"  {rel}: in `{sym}` — :{role}:`{target}` does not resolve\n")
    if len(unresolved) > limit:
        sys.stderr.write(f"  ... {len(unresolved) - limit} more (re-run with --list)\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
