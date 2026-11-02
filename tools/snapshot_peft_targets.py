#!/usr/bin/env python3
"""tools/snapshot_peft_targets.py — the reference library's default ``target_modules`` tables, as JSON.

A user moving from the reference adapter library writes ``LoraConfig(target_modules=[...])``
with the names that library's models use, or leaves it out and relies on its per-architecture
default.  Both only work here if those names reach modules of the zoo's classes, and that
claim rots silently (the reference adds a row, a zoo class renames a module).  This tool reads
the reference library's *source* (never imports it) and writes every default table to a JSON
file that ``lucid/test/unit/peft/test_reference_targets.py`` holds ``lucid.peft`` to.

Usage::

    pip download peft==X.Y.Z --no-deps -d /tmp/peftsig/whl
    unzip -q /tmp/peftsig/whl/peft-*.whl 'peft/*' -d /tmp/peftsig/src
    python3 -I tools/snapshot_peft_targets.py /tmp/peftsig/src \\
        lucid/test/unit/peft/_reference_targets.json

The extracted source is untrusted data: it is parsed with ``ast`` only and is never imported
or executed.

What is recorded
----------------
Every module-level ``TRANSFORMERS_MODELS_TO_<METHOD>_(TARGET|FEEDFORWARD)_MODULES_MAPPING``
in ``peft/utils/constants.py``, as the library leaves it after the whole file has run:
a dict literal, a ``.copy()`` of another table, and the later ``TABLE[model_type] = [...]``
assignments that edit it.  A table that began as a copy is stored as its parent's name plus
the rows that differ from the parent (``rows``) and the parent's rows it dropped
(``removed``), so the file shows what each method changed from the LoRA table and stays small.
A value is a list of module names or one regex string, as the reference spells it.
"""

import ast
import json
import re
import sys
from pathlib import Path
from typing import NamedTuple

_TABLE = re.compile(
    r"TRANSFORMERS_MODELS_TO_(?P<method>[A-Z0-9_]+?)_(?P<kind>TARGET|FEEDFORWARD)_MODULES_MAPPING"
)

type Row = list[str] | str
type Rows = dict[str, Row]


class _Table(NamedTuple):
    rows: Rows
    parent: str | None


def _name(node: ast.expr) -> str | None:
    return node.id if isinstance(node, ast.Name) else None


def _literal_rows(node: ast.expr) -> Rows | None:
    try:
        value = ast.literal_eval(node)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _copied(node: ast.expr, tables: dict[str, _Table]) -> str | None:
    """The table ``node`` copies (``PARENT.copy()``), if it is one the file defined already."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
        return None
    if node.func.attr != "copy":
        return None
    parent = _name(node.func.value)
    return parent if parent in tables else None


def read_tables(source: str) -> dict[str, _Table]:
    """Every default table in ``source`` (the text of ``constants.py``), by constant name."""
    tables: dict[str, _Table] = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        name = _name(target)
        if name is not None and _TABLE.fullmatch(name):
            parent = _copied(node.value, tables)
            if parent is not None:
                tables[name] = _Table(dict(tables[parent].rows), parent)
                continue
            literal = _literal_rows(node.value)
            if literal is not None:
                tables[name] = _Table(literal, None)
        elif isinstance(target, ast.Subscript):
            owner = _name(target.value)
            if owner not in tables:
                continue
            try:
                key, value = ast.literal_eval(target.slice), ast.literal_eval(node.value)
            except ValueError:
                continue
            tables[owner].rows[key] = value
    return tables


def _key(constant: str) -> str:
    """``TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING`` → ``IA3_FEEDFORWARD``."""
    found = _TABLE.fullmatch(constant)
    if found is None:
        raise ValueError(constant)
    method, kind = found["method"], found["kind"]
    return method if kind == "TARGET" else f"{method}_{kind}"


def encode(tables: dict[str, _Table]) -> dict[str, dict[str, object]]:
    """``tables`` as the JSON holds them: a copy stores only how it differs from its parent."""
    out: dict[str, dict[str, object]] = {}
    for constant, table in tables.items():
        if table.parent is None:
            out[_key(constant)] = {"base": None, "rows": table.rows, "removed": []}
            continue
        parent = tables[table.parent].rows
        out[_key(constant)] = {
            "base": _key(table.parent),
            "rows": {k: v for k, v in table.rows.items() if parent.get(k) != v},
            "removed": sorted(set(parent) - set(table.rows)),
        }
    return dict(sorted(out.items()))


def version(root: Path) -> str:
    """The reference library's ``__version__``, read from its ``__init__.py``."""
    for node in ast.parse((root / "peft" / "__init__.py").read_text()).body:
        if isinstance(node, ast.Assign) and _name(node.targets[0]) == "__version__":
            value = ast.literal_eval(node.value)
            if isinstance(value, str):
                return value
    raise ValueError("no __version__ in peft/__init__.py")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    root, out = Path(argv[1]), Path(argv[2])
    tables = encode(read_tables((root / "peft" / "utils" / "constants.py").read_text()))
    data = {"peft_version": version(root), "tables": tables}
    out.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    print(f"peft {data['peft_version']}: {len(tables)} tables -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
