"""build-cross-links.py — derive Python ↔ C++ symbol mapping for docs.

Walks every Python module JSON plus the C++ engine JSON under
``web/public/api-data/``, infers cross-links via naming conventions, and
emits ``_cross_links.json`` for the UI to consume.

Naming conventions Lucid follows (and this script keys off):

  C++ class ``LinearBackward``         → Python class ``Linear`` (in ``lucid.nn``)
                                       → Python fn   ``linear`` (in ``lucid.nn.functional``)
  C++ class ``Conv2dBackward``         → Python class ``Conv2d``
                                       → Python fn   ``conv2d``
  C++ class ``BatchNormNdBackward``    → Python classes ``BatchNorm{1,2,3}d``
  C++ fn    ``linear_op``              → Python fn   ``linear``
  C++ fn    ``fftn_op``                → Python fn   ``fftn``

The output schema (matched 1:1 by ``src/lib/cross-links.ts``) is::

    {
      "python_to_cpp": {
        "lucid.nn.Linear": [
          { "name": "LinearBackward", "kind": "backward_node" }
        ],
        ...
      },
      "cpp_to_python": {
        "LinearBackward": [
          { "path": "lucid.nn.Linear",            "module": "lucid.nn",            "kind": "class"    },
          { "path": "lucid.nn.functional.linear", "module": "lucid.nn.functional", "kind": "function" }
        ],
        ...
      }
    }

Missing matches are silently dropped — better to omit a link than guess
wrong.  When a future op exposes a non-standard naming scheme, add a
hand-curated override in ``web/data/cross-links.overrides.json``
(merged in at the end).
"""

import json
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
WEB_ROOT = HERE.parent
API_DATA = WEB_ROOT / "public" / "api-data"
OUT = API_DATA / "_cross_links.json"
OVERRIDES = WEB_ROOT / "data" / "cross-links.overrides.json"

ENGINE_SLUG = "lucid._C.engine"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _camel_to_snake(name: str) -> str:
    """``Conv2d`` → ``conv2d``; ``MaxPoolNd`` → ``max_pool_nd``;
    ``LinearBackward`` → ``linear_backward``."""
    out = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    # Collapse digit-letter underscores that bury common suffixes:
    # ``conv_2d`` → ``conv2d`` keeps the canonical Python form.
    out = re.sub(r"(?<=\d)_(?=[a-z])", "", out)
    out = re.sub(r"(?<=[a-z])_(?=\d)", "", out)
    return out


def _backward_base(cpp_class_name: str) -> str | None:
    """Strip the trailing ``Backward`` from a C++ class name; return
    ``None`` if it doesn't end with that suffix (anything else isn't an
    autograd node and doesn't participate in the mapping)."""
    if cpp_class_name.endswith("Backward"):
        return cpp_class_name[: -len("Backward")]
    return None


def _op_base(cpp_fn_name: str) -> str | None:
    """Strip a single trailing ``_op`` / ``_inplace_op`` suffix.  Returns
    ``None`` for free functions that don't follow that convention (they
    won't auto-match to Python)."""
    if cpp_fn_name.endswith("_inplace_op"):
        return cpp_fn_name[: -len("_inplace_op")] + "_"
    if cpp_fn_name.endswith("_op"):
        return cpp_fn_name[: -len("_op")]
    return None


def _expand_nd_variants(base: str) -> list[str]:
    """``BatchNormNd`` → ``BatchNorm1d``, ``BatchNorm2d``, ``BatchNorm3d``.
    A no-op when ``Nd`` isn't present at the tail."""
    if base.endswith("Nd"):
        stem = base[:-2]
        return [f"{stem}{n}d" for n in (1, 2, 3)]
    return [base]


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def _load_engine() -> dict[str, Any] | None:
    p = API_DATA / f"{ENGINE_SLUG}.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def _load_python_modules() -> dict[str, dict[str, Any]]:
    """Return ``{slug: module_json}`` for every Python-side module file
    in ``api-data/`` — excluding the C++ engine and underscore-prefixed
    caches."""
    out: dict[str, dict[str, Any]] = {}
    for p in sorted(API_DATA.glob("*.json")):
        if p.name.startswith("_"):
            continue
        slug = p.stem
        if slug == ENGINE_SLUG:
            continue
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        out[slug] = data
    return out


# ---------------------------------------------------------------------------
# Index Python symbols by name
# ---------------------------------------------------------------------------


def _index_python_symbols(modules: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, str]]]:
    """Map ``name`` → ``[{path, module, kind}]``.  Classes and functions
    co-exist in the same index by their bare name; the consumer
    disambiguates by ``kind``."""
    idx: dict[str, list[dict[str, str]]] = {}
    for slug, data in modules.items():
        kind = data.get("kind")
        if kind == "module":
            members = data.get("members", [])
        elif kind == "class-module":
            # The module IS a class (e.g. lucid.tensor → Tensor).  Treat
            # its methods as members of the class itself rather than the
            # module — they don't participate in the cross-link mapping
            # (methods aren't autograd nodes).  But the class itself
            # does.
            members = [
                {"name": data.get("name", ""), "kind": "class", "path": data.get("path", "")}
            ]
        else:
            continue
        for m in members:
            name = m.get("name")
            if not name:
                continue
            entry = {"path": f"{slug}.{name}", "module": slug, "kind": m.get("kind", "function")}
            idx.setdefault(name, []).append(entry)
    return idx


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def _match_backward_class(
    cpp_name: str, py_idx: dict[str, list[dict[str, str]]]
) -> list[dict[str, str]]:
    """Find Python symbols backed by a C++ ``XBackward`` class."""
    base = _backward_base(cpp_name)
    if base is None:
        return []
    matches: list[dict[str, str]] = []
    seen: set[str] = set()
    for candidate in _expand_nd_variants(base):
        # 1. Python class with the same camelCase name (e.g. ``Linear``).
        for entry in py_idx.get(candidate, []):
            if entry["kind"] != "class":
                continue
            if entry["path"] in seen:
                continue
            seen.add(entry["path"])
            matches.append(entry)
        # 2. Python function with snake_case name (e.g. ``linear``).
        snake = _camel_to_snake(candidate)
        for entry in py_idx.get(snake, []):
            if entry["kind"] != "function":
                continue
            if entry["path"] in seen:
                continue
            seen.add(entry["path"])
            matches.append(entry)
    return matches


def _match_op_function(
    cpp_name: str, py_idx: dict[str, list[dict[str, str]]]
) -> list[dict[str, str]]:
    """Find Python functions implemented by a C++ ``foo_op`` free fn."""
    base = _op_base(cpp_name)
    if base is None:
        return []
    matches: list[dict[str, str]] = []
    seen: set[str] = set()
    for entry in py_idx.get(base, []):
        if entry["kind"] != "function":
            continue
        if entry["path"] in seen:
            continue
        seen.add(entry["path"])
        matches.append(entry)
    return matches


# ---------------------------------------------------------------------------
# Build tables
# ---------------------------------------------------------------------------


def _build(engine: dict[str, Any], py_idx: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    cpp_members = engine.get("members", [])

    cpp_to_python: dict[str, list[dict[str, str]]] = {}
    python_to_cpp: dict[str, list[dict[str, str]]] = {}

    for m in cpp_members:
        cpp_name = m.get("name")
        cpp_kind = m.get("kind")
        if not cpp_name or not cpp_kind:
            continue

        if cpp_kind == "class":
            matches = _match_backward_class(cpp_name, py_idx)
            link_kind = "backward_node"
        elif cpp_kind == "function":
            matches = _match_op_function(cpp_name, py_idx)
            link_kind = "free_function"
        else:
            continue

        if not matches:
            continue

        cpp_to_python[cpp_name] = matches
        for py in matches:
            python_to_cpp.setdefault(py["path"], []).append({"name": cpp_name, "kind": link_kind})

    return {"python_to_cpp": python_to_cpp, "cpp_to_python": cpp_to_python}


def _apply_overrides(table: dict[str, Any]) -> dict[str, Any]:
    """Merge the hand-curated overrides file on top of inferred matches.
    Overrides shape mirrors the output schema and is permissive — any key
    present replaces the inferred entry for that key entirely."""
    if not OVERRIDES.is_file():
        return table
    try:
        ovr = json.loads(OVERRIDES.read_text())
    except json.JSONDecodeError:
        print(f"[cross-links] failed to parse {OVERRIDES} — skipping overrides", file=sys.stderr)
        return table
    for section in ("python_to_cpp", "cpp_to_python"):
        for k, v in ovr.get(section, {}).items():
            table[section][k] = v
    return table


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def main() -> int:
    engine = _load_engine()
    if engine is None:
        print(f"[cross-links] {ENGINE_SLUG}.json missing — nothing to do", file=sys.stderr)
        return 0
    py_modules = _load_python_modules()
    py_idx = _index_python_symbols(py_modules)
    table = _build(engine, py_idx)
    table = _apply_overrides(table)
    OUT.write_text(json.dumps(table, indent=2, ensure_ascii=False) + "\n")
    p2c = len(table["python_to_cpp"])
    c2p = len(table["cpp_to_python"])
    print(
        f"[cross-links] wrote {p2c} python→cpp, {c2p} cpp→python entries → {OUT.relative_to(WEB_ROOT)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
