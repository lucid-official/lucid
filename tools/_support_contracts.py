"""Trace public names to source, typing, documentation, test references and audit.

Static test references are navigation aids, not evidence that a test ran or
asserted a particular contract. Missing links remain explicit empty lists.
"""

import ast
import inspect
import json
from pathlib import Path
from typing import Any

from lucid.test.audit._surface import Symbol


def canonical(name: str) -> str:
    if name.startswith("F."):
        return "lucid.nn.functional." + name[2:]
    if name.startswith("nn."):
        return "lucid." + name
    if name.startswith("Tensor."):
        return "lucid." + name
    return name.replace("lucid._tensor.tensor.Tensor", "lucid.Tensor")


def test_references(root: Path) -> dict[str, list[str]]:
    links: dict[str, list[str]] = {}
    for path in sorted((root / "lucid/test").rglob("test_*.py")):
        tree = ast.parse(path.read_text())
        aliases = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("lucid"):
                        aliases[alias.asname or alias.name.split(".")[0]] = (
                            alias.name if alias.asname else alias.name.split(".")[0]
                        )
            elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("lucid"):
                for alias in node.names:
                    aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        found = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            cursor = node.func
            parts = []
            while isinstance(cursor, ast.Attribute):
                parts.append(cursor.attr)
                cursor = cursor.value
            if not isinstance(cursor, ast.Name) or cursor.id not in aliases:
                continue
            name = canonical(".".join([aliases[cursor.id], *reversed(parts)]))
            link = f"{path.relative_to(root)}:{node.lineno}"
            if (name, link) not in found:
                links.setdefault(name, []).append(link)
                found.add((name, link))
    return links


def documentation(root: Path) -> dict[str, list[str]]:
    links: dict[str, list[str]] = {}

    def visit(value: object, path: Path) -> None:
        if isinstance(value, dict):
            name = value.get("path")
            if isinstance(name, str) and name.startswith("lucid."):
                links.setdefault(canonical(name), []).append(str(path.relative_to(root)))
            for child in value.values():
                if isinstance(child, list | dict):
                    visit(child, path)
        elif isinstance(value, list):
            for child in value:
                visit(child, path)

    for path in sorted((root / "web/public/api-data").glob("*.json")):
        if not path.name.startswith(("lucid.models", "_")):
            visit(json.loads(path.read_text()), path)
    return {name: sorted(set(paths)) for name, paths in links.items()}


def stub_declarations(root: Path) -> dict[str, list[str]]:
    links: dict[str, list[str]] = {}
    for relative, prefix in (
        ("lucid/__init__.pyi", "lucid"),
        ("lucid/_tensor/tensor.pyi", "lucid._tensor.tensor"),
    ):
        path = root / relative
        if not path.exists():
            continue

        def visit(nodes: list[ast.stmt], parent: str) -> None:
            for node in nodes:
                names = []
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                    names = [node.name]
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    names = [node.target.id]
                elif isinstance(node, ast.Assign):
                    names = [target.id for target in node.targets if isinstance(target, ast.Name)]
                elif isinstance(node, ast.ImportFrom):
                    names = [alias.asname for alias in node.names if alias.asname]
                for name in names:
                    key = canonical(f"{parent}.{name}")
                    links.setdefault(key, []).append(f"{relative}:{node.lineno}")
                if isinstance(node, ast.ClassDef):
                    visit(node.body, f"{parent}.{node.name}")

        visit(ast.parse(path.read_text()).body, prefix)
    return links


def contract_links(
    root: Path, symbols: list[Symbol], audit_report: dict[str, Any] | None
) -> list[dict[str, Any]]:
    tests = test_references(root)
    docs = documentation(root)
    stubs = stub_declarations(root)
    findings: dict[str, list[dict[str, Any]]] = {}
    for finding in (audit_report or {}).get("findings", []):
        findings.setdefault(canonical(finding["symbol"]), []).append(
            {key: finding.get(key) for key in ("axis", "status", "detail")}
        )
    result = []
    for symbol in symbols:
        name = canonical(symbol.qualname)
        obj: Any = symbol.obj.fget if isinstance(symbol.obj, property) else symbol.obj
        definition = canonical(
            f"{getattr(obj, '__module__', '')}.{getattr(obj, '__qualname__', '')}"
        )
        documented = sorted(set(docs.get(name, []) + docs.get(definition, [])))
        source = None
        try:
            # A decorated function (a loss under ``_amp_op``) is read where its
            # def is, not where the wrapper was written.
            defined = inspect.unwrap(obj)
            filename = inspect.getsourcefile(defined)
            path = Path(filename) if filename else None
            if path is not None and path.is_relative_to(root):
                source = f"{path.relative_to(root)}:{inspect.getsourcelines(defined)[1]}"
        except (TypeError, OSError):
            pass
        result.append(
            {
                "name": name,
                "audit_name": symbol.qualname,
                "source": source,
                "stub_declarations": stubs.get(name, []),
                "typing_mode": "stub" if name in stubs else "inline_or_unverified",
                "definition_name": definition,
                "documentation": documented,
                "static_test_calls": tests.get(name, []),
                "audit_findings": findings.get(name, []),
            }
        )
    return result
