#!/usr/bin/env python3
"""
tools/check_stubs.py — Verify that committed .pyi stubs are up to date.

Runs gen_pyi.py to a temp directory and diffs output against committed stubs.
Exits non-zero if any stub is stale.
"""

import ast
import difflib
import sys
import types
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def _run_gen_pyi_to_tempdir() -> dict[str, str]:
    """Run gen_pyi logic and return {relative_path: content}."""
    import tools.gen_pyi as gen

    engine_content, _ = gen.gen_engine_pyi()
    tensor_content, _ = gen.gen_tensor_pyi()
    init_content, _ = gen.gen_init_pyi()

    return {
        "lucid/_C/engine.pyi": engine_content,
        "lucid/_tensor/tensor.pyi": tensor_content,
        "lucid/__init__.pyi": init_content,
    }


def _collect_runtime_classes(mod: object) -> dict[str, type]:
    """Collect bound classes from *engine* and its non-``compile`` submodules."""
    classes: dict[str, type] = {}
    for k in sorted(dir(mod)):
        if k.startswith("__") or k == "compile":
            continue
        val = getattr(mod, k)
        if isinstance(val, type):
            classes[k] = val
        elif isinstance(val, types.ModuleType):
            classes.update(_collect_runtime_classes(val))
    return classes


def _stub_class_member_names(cls_node: ast.ClassDef) -> set[str]:
    names: set[str] = set()
    for node in cls_node.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


def _stub_setter_names(cls_node: ast.ClassDef) -> set[str]:
    return {
        node.name
        for node in cls_node.body
        if isinstance(node, ast.FunctionDef)
        and any(ast.unparse(d).endswith(".setter") for d in node.decorator_list)
    }


def _stub_method_defs(cls_node: ast.ClassDef) -> dict[str, list[ast.FunctionDef]]:
    methods: dict[str, list[ast.FunctionDef]] = {}
    for node in cls_node.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        decs = [ast.unparse(d) for d in node.decorator_list]
        if any(d == "property" or d.endswith((".setter", ".deleter")) for d in decs):
            continue
        methods.setdefault(node.name, []).append(node)
    return methods


def _split_top_sig_params(params: str) -> list[str]:
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


def _runtime_doc_params(name: str, fn: object) -> list[str] | None:
    import re

    doc = getattr(fn, "__doc__", None) or ""
    first = doc.splitlines()[0] if doc else ""
    m = re.match(rf"^{re.escape(name)}\((?P<params>.*)\)\s*->", first)
    if m is None:
        return None
    return [
        p.split(":")[0].split("=")[0].strip()
        for p in _split_top_sig_params(m["params"])
        if p.strip() != "/"
    ]


def _ast_fn_params(fn: ast.FunctionDef) -> list[str]:
    a = fn.args
    names = [p.arg for p in (*a.posonlyargs, *a.args)]
    if a.vararg is not None:
        names.append("*" + a.vararg.arg)
    names += [p.arg for p in a.kwonlyargs]
    if a.kwarg is not None:
        names.append("**" + a.kwarg.arg)
    return names


def _check_callable_params(
    qualname: str, fn_name: str, defs: list[ast.FunctionDef], runtime_fn: object, *, is_method: bool
) -> list[str]:
    if len(defs) != 1:
        return []
    want = _runtime_doc_params(fn_name, runtime_fn)
    if want is None:
        return []
    if is_method and want and want[0] == "self":
        want = want[1:]
    if want == ["*args"] or any(p.startswith("arg") and p[3:].isdigit() for p in want):
        return []
    got = _ast_fn_params(defs[0])
    if is_method and got and got[0] == "self":
        got = got[1:]
    if got != want:
        return [f"{qualname}: stub ({', '.join(got)}), engine ({', '.join(want)})"]
    return []


def _class_member_drift(qual: str, cls_node: ast.ClassDef, cls: type) -> list[str]:
    from tools.gen_pyi import is_stubbed_class_member

    stubbed = _stub_class_member_names(cls_node)
    if issubclass(cls, BaseException):
        return [f"{qual}.{m}: in the stub, not in the engine" for m in sorted(stubbed)]

    have = {m for m, raw in cls.__dict__.items() if is_stubbed_class_member(m, raw)}
    problems = [f"{qual}.{m}: in the stub, not in the engine" for m in sorted(stubbed - have)]
    problems += [f"{qual}.{m}: in the engine, not in the stub" for m in sorted(have - stubbed)]

    want_setters = {
        m for m, raw in cls.__dict__.items() if isinstance(raw, property) and raw.fset is not None
    }
    got_setters = _stub_setter_names(cls_node)
    problems += [
        f"{qual}.{m}.setter: in the stub, not in the engine"
        for m in sorted(got_setters - want_setters)
    ]
    problems += [
        f"{qual}.{m}.setter: in the engine, not in the stub"
        for m in sorted(want_setters - got_setters)
    ]

    for mname, defs in sorted(_stub_method_defs(cls_node).items()):
        if mname not in have:
            continue
        raw = cls.__dict__[mname]
        if isinstance(raw, (property, type)):
            continue
        is_static = isinstance(raw, staticmethod)
        target = raw.__func__ if is_static else getattr(cls, mname, raw)
        problems += _check_callable_params(
            f"{qual}.{mname}", mname, defs, target, is_method=not is_static
        )

    nested_stub = {n.name: n for n in cls_node.body if isinstance(n, ast.ClassDef)}
    for mname, sub_node in sorted(nested_stub.items()):
        sub_cls = cls.__dict__.get(mname)
        if isinstance(sub_cls, type):
            problems += _class_member_drift(f"{qual}.{mname}", sub_node, sub_cls)
    return problems


def _top_level_binding_drift(tree: ast.Module, engine: object) -> list[str]:
    from tools.gen_pyi import SKIP_PRIVATE_ENGINE_FNS

    have_fns = {
        k
        for k in dir(engine)
        if not k.startswith("__")
        and k not in SKIP_PRIVATE_ENGINE_FNS
        and not isinstance(getattr(engine, k), (type, types.ModuleType))
        and callable(getattr(engine, k))
    }
    stub_fns: dict[str, list[ast.FunctionDef]] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            stub_fns.setdefault(node.name, []).append(node)

    problems = [f"{fn}: in the stub, not in the engine" for fn in sorted(set(stub_fns) - have_fns)]
    problems += [f"{fn}: in the engine, not in the stub" for fn in sorted(have_fns - set(stub_fns))]
    for fn in sorted(set(stub_fns) & have_fns):
        problems += _check_callable_params(
            fn, fn, stub_fns[fn], getattr(engine, fn), is_method=False
        )

    have_consts = {
        k
        for k in dir(engine)
        if not k.startswith("__")
        and not isinstance(getattr(engine, k), (type, types.ModuleType))
        and not callable(getattr(engine, k))
    }
    stub_consts = {
        node.target.id
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and not (
            isinstance(node.annotation, ast.Name) and node.annotation.id.endswith("_namespace")
        )
    }
    problems += [f"{c}: in the stub, not in the engine" for c in sorted(stub_consts - have_consts)]
    problems += [f"{c}: in the engine, not in the stub" for c in sorted(have_consts - stub_consts)]
    return problems


def engine_drift(stub: str, engine: object) -> list[str]:
    """Verify 1:1 parity between ``lucid._C.engine`` and ``engine.pyi``."""
    tree = ast.parse(stub)
    stub_classes = {
        n.name: n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and not n.name.endswith("_namespace")
    }
    runtime_classes = _collect_runtime_classes(engine)

    problems = [
        f"{c}: in the stub, not in the engine"
        for c in sorted(set(stub_classes) - set(runtime_classes))
    ]
    problems += [
        f"{c}: in the engine, not in the stub"
        for c in sorted(set(runtime_classes) - set(stub_classes))
    ]
    for cname in sorted(set(stub_classes) & set(runtime_classes)):
        problems += _class_member_drift(cname, stub_classes[cname], runtime_classes[cname])
    problems += _top_level_binding_drift(tree, engine)
    return problems


def _report_drift(label: str, items: list[str], hint: str) -> bool:
    for line in items:
        print(f"[check_stubs] {label} {line}")
    if items:
        print(f"\n[check_stubs] {len(items)} {hint}", file=sys.stderr)
        return True
    return False


def _diff_committed_stubs(generated: dict[str, str]) -> list[str]:
    stale: list[str] = []
    for rel_path, new_content in generated.items():
        committed_path = ROOT / rel_path
        if not committed_path.exists():
            print(f"[check_stubs] MISSING   {rel_path}")
            stale.append(rel_path)
            continue

        committed = committed_path.read_text(encoding="utf-8")
        if committed == new_content:
            print(f"[check_stubs] OK       {rel_path}")
            continue

        diff = list(
            difflib.unified_diff(
                committed.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                fromfile=f"{rel_path} (committed)",
                tofile=f"{rel_path} (generated)",
                n=3,
            )
        )
        print(f"[check_stubs] STALE    {rel_path}")
        sys.stdout.writelines(diff[:40])
        if len(diff) > 40:
            print(f"  ... ({len(diff) - 40} more diff lines)")
        stale.append(rel_path)
    return stale


def main() -> int:
    try:
        generated = _run_gen_pyi_to_tempdir()
    except Exception as exc:
        print(f"[check_stubs] ERROR running gen_pyi: {exc}", file=sys.stderr)
        return 1

    import importlib
    import lucid

    from lucid._C import engine as _C_engine
    from lucid._tensor.tensor import Tensor
    from tools.gen_pyi import tensor_method_params

    stub_ns = importlib.import_module("tools.stub_namespaces")

    if _report_drift(
        "INIT",
        stub_ns.module_all_drift(generated["lucid/__init__.pyi"], lucid.__all__),
        "lucid/__init__.pyi stub(s) omit symbol(s) listed in lucid.__all__.\n"
        "  Add them to _INIT_HEADER in tools/gen_pyi.py, then regenerate.",
    ):
        return 1

    if _report_drift(
        "TENSOR",
        stub_ns.class_drift(generated["lucid/_tensor/tensor.pyi"], Tensor, tensor_method_params),
        "Tensor stub(s) disagree with the runtime class.\n"
        "  Fix _TENSOR_HEADER / python_only in tools/gen_pyi.py, then regenerate.",
    ):
        return 1

    if _report_drift(
        "NAMESPACE",
        stub_ns.namespace_drift(generated["lucid/_C/engine.pyi"], _C_engine),
        "engine sub-namespace stub(s) disagree with the loaded engine.\n"
        "  Regenerate with `python tools/gen_pyi.py`.",
    ):
        return 1

    if _report_drift(
        "ENGINE",
        engine_drift(generated["lucid/_C/engine.pyi"], _C_engine),
        "engine class/member/function stub(s) disagree with the loaded engine.\n"
        "  Regenerate with `python tools/gen_pyi.py`.",
    ):
        return 1

    stale = _diff_committed_stubs(generated)
    if stale:
        print(
            f"\n[check_stubs] {len(stale)} stub(s) out of date. Run:\n"
            "    python tools/gen_pyi.py\n"
            f"    git add {' '.join(stale)}\n",
            file=sys.stderr,
        )
        print("\n  Commit blocked. Run the following to fix, then re-stage and commit:\n")
        print("    python tools/gen_pyi.py")
        print("    git add lucid/_C/engine.pyi lucid/_tensor/tensor.pyi lucid/__init__.pyi")
        return 1

    print("[check_stubs] all stubs up to date.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
