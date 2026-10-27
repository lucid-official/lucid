"""build-usedby.py — derive a Python ``Used by`` backlink map.

Walks every ``lucid/**/*.py`` file, harvests its ``import`` /
``from … import`` statements, and emits the inverted map: for each
fully-qualified Lucid symbol, the list of files that import it.

The output schema (matched 1:1 by ``src/lib/usedby.ts``) is::

    {
      "lucid.nn.functional.linear": [
        { "module": "lucid.nn.modules.linear", "kind": "module" },
        { "module": "lucid.models.vision.resnet._model", "kind": "module" },
        ...
      ],
      "lucid.nn.Module": [
        { "module": "lucid.nn.modules.conv",   "kind": "module" },
        ...
      ]
    }

Heuristic, not a call-graph analysis — we treat ``import X`` as
evidence the importing module *uses* X.  That covers the vast
majority of real-world dependencies and avoids the ~10× complexity
of an AST visitor that follows attribute lookups through aliases.

False-positive surface: imports that are referenced only inside a
``TYPE_CHECKING`` block.  Those still count as "uses" semantically —
they document a typing relationship — so we don't filter them out.

Run order in the prebuild chain: after ``build-api-data.py`` so the
list of emitted Lucid paths is already on disk; we union those with
the imports map to suppress entries for non-public symbols.
"""

from __future__ import annotations  # noqa: F401  — tooling script, runtime constraints don't apply

import ast
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEB_ROOT = HERE.parent
REPO_ROOT = WEB_ROOT.parent
LUCID_SRC = REPO_ROOT / "lucid"
API_DATA = WEB_ROOT / "public" / "api-data"
OUT = API_DATA / "_usedby.json"

# Skip these subtrees entirely — they're test fixtures or perf scripts
# that don't represent "production" usage of the public surface.
SKIP_DIRS = {"test", "benchmarks"}


# Split-file modules that belong to a single logical module group in docs.
_MODULE_ALIASES: dict[str, str] = {"lucid.nn.modules.conv_transpose": "lucid.nn.modules.conv"}


def _module_path_of(p: Path) -> str:
    """Convert ``lucid/nn/functional/linear.py`` → ``lucid.nn.functional.linear``."""
    rel = p.relative_to(LUCID_SRC.parent).with_suffix("")
    parts = rel.parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    raw = ".".join(parts)
    return _MODULE_ALIASES.get(raw, raw)


def _resolve_relative(importer_pkg: str, level: int, module: str | None) -> str | None:
    """``from . import X`` / ``from ..foo import Y`` style.  Resolves the
    relative spec against the importing module's *package* path."""
    pkg_parts = importer_pkg.split(".")
    if level > len(pkg_parts):
        return None
    base = pkg_parts[: len(pkg_parts) - level + 1]
    if module:
        base = base + module.split(".")
    return ".".join(base) if base else None


def _collect_imports_per_file() -> dict[str, set[str]]:
    """Map ``importing_module_path → { imported_symbol_path, … }``.

    ``import lucid.nn.functional`` records ``lucid.nn.functional``.
    ``from lucid.nn.functional import linear`` records
    ``lucid.nn.functional.linear``.  Star imports are dropped (we
    don't know what they pull in) but they're rare in lucid/.
    """
    out: dict[str, set[str]] = {}
    for p in sorted(LUCID_SRC.rglob("*.py")):
        parts = p.relative_to(LUCID_SRC).parts
        if any(seg in SKIP_DIRS for seg in parts):
            continue
        try:
            tree = ast.parse(p.read_text())
        except SyntaxError:
            continue
        importer = _module_path_of(p)
        # Package the file belongs to — needed for relative-import
        # resolution (``__init__.py`` is its own package; everything
        # else takes the directory's path).
        importer_pkg = importer if p.name == "__init__.py" else ".".join(importer.split(".")[:-1])
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("lucid"):
                        imported.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module
                if node.level:
                    resolved = _resolve_relative(importer_pkg, node.level, module)
                    if resolved is None:
                        continue
                    base = resolved
                else:
                    base = module or ""
                if not base.startswith("lucid"):
                    continue
                # ``from X import *`` — skip (no name set).
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    imported.add(f"{base}.{alias.name}")
                # ``from lucid.nn import functional`` — also count the
                # base import, otherwise pages for ``lucid.nn`` itself
                # would never appear in any "used by" list.
                imported.add(base)
        if imported:
            out.setdefault(importer, set()).update(imported)
    return out


def _emitted_symbol_paths() -> set[str]:
    """Union of every ``path`` field across the emitted API JSONs.

    Used to filter the imports map down to *documented* Lucid symbols
    — importing a private helper (``lucid._dispatch._unwrap``) shouldn't
    create a stale entry that links to nowhere.
    """
    paths: set[str] = set()

    def walk(node):
        p = node.get("path") if isinstance(node, dict) else None
        if isinstance(p, str):
            paths.add(p)
        members = node.get("members") if isinstance(node, dict) else None
        if isinstance(members, list):
            for m in members:
                walk(m)
        methods = node.get("methods") if isinstance(node, dict) else None
        if isinstance(methods, list):
            for m in methods:
                walk(m)

    if not API_DATA.is_dir():
        return paths
    for jp in API_DATA.glob("*.json"):
        if jp.name.startswith("_"):
            continue
        try:
            walk(json.loads(jp.read_text()))
        except Exception:
            continue
    return paths


def _invert(imports: dict[str, set[str]], emitted: set[str]) -> dict[str, list[dict[str, str]]]:
    """Flip ``importer → imported`` to ``target → [{module, kind}]``.

    Filters out targets that don't correspond to an emitted JSON path —
    so the docs site never offers a "Used by" entry pointing to a
    symbol that has no docs page to land on.
    """
    inverted: dict[str, set[str]] = {}
    for importer, targets in imports.items():
        for t in targets:
            resolved_t = t
            if resolved_t not in emitted:
                for src_mod, canonical_mod in _MODULE_ALIASES.items():
                    prefix = f"{canonical_mod}."
                    if resolved_t.startswith(prefix):
                        cand = f"{src_mod}.{resolved_t[len(prefix):]}"
                        if cand in emitted:
                            resolved_t = cand
                            break
            inverted.setdefault(resolved_t, set()).add(importer)
    result: dict[str, list[dict[str, str]]] = {}
    for target, importers in inverted.items():
        if target not in emitted:
            continue
        target_parent = target.rsplit(".", 1)[0]
        target_group = _MODULE_ALIASES.get(target_parent, target_parent)
        rows = [
            {"module": imp, "kind": "module"}
            for imp in sorted(importers)
            if imp != target and imp != target_group  # self-import noise
        ]
        if rows:
            result[target] = rows
    return result


def main() -> int:
    if not API_DATA.is_dir():
        print(
            f"[build-usedby] api-data dir missing — " f"run ``build-api-data.py`` first.",
            file=sys.stderr,
        )
        return 1
    print("[build-usedby] scanning lucid/ for imports…")
    imports = _collect_imports_per_file()
    print(f"[build-usedby]   {len(imports)} module(s) with at least one import.")
    emitted = _emitted_symbol_paths()
    print(f"[build-usedby]   {len(emitted)} emitted symbol path(s).")
    result = _invert(imports, emitted)
    print(f"[build-usedby]   {len(result)} target(s) with one or more callers.")

    OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[build-usedby] wrote {OUT.relative_to(REPO_ROOT)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
