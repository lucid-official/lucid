#!/usr/bin/env python3
"""
tools/check_numpy_h4.py — enforce CLAUDE.md Hard Rule H4.

H4 says: numpy may enter Lucid only at the six sanctioned bridge surfaces
listed in CLAUDE.md.  Everywhere else, ``import numpy`` (or
``from numpy import …``) is forbidden — including in docstrings of
test files we sometimes copy from upstream.

The 3.0.1 → 3.0.2 hotfix sequence found *four* unsanctioned numpy
import sites that had crept in silently (``_tensor/_to.py``,
``_factories/converters.py`` for the non-ndarray path,
``autograd/{_backward,_hooks,_functional}.py``,
``nn/utils/rnn.py``).  All four were latent in the codebase before
3.0.0 shipped — there was no PR-time guard to catch them.  This script
is that guard.

Usage:

    python3 tools/check_numpy_h4.py            # exit 1 on any violation
    python3 tools/check_numpy_h4.py --list     # also print the sanctioned list
    python3 tools/check_numpy_h4.py --quiet    # exit code only, no output

Wire into pre-commit or CI:

    python3 tools/check_numpy_h4.py || exit 1

Whitelist semantics:

* ``import numpy`` / ``from numpy import …`` (and ``numpy.*`` submodules,
  e.g. ``numpy.typing``) are flagged.
* Imports inside ``if TYPE_CHECKING:`` blocks are *allowed everywhere* —
  they evaluate to no-ops at runtime under PEP 649 (Python 3.14).
* Imports inside the sanctioned files are allowed regardless of position.
* Test files (``lucid/test/**``) are excluded entirely — tests opt into
  numpy via the parity-framework tier.
"""

import ast
import pathlib
import sys
from typing import NamedTuple

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
LUCID_ROOT = REPO_ROOT / "lucid"


# ── H4 sanctioned bridge files (CLAUDE.md §6.1 H4) ──────────────────────────
# Paths are relative to ``lucid/``.
#
# CLAUDE.md lists six bridge boundaries.  Implementation files we map them to:
#
#   ① ``_factories/converters.py``       (tensor / from_numpy / from_dlpack / to_dlpack)
#   ② ``_tensor/tensor.py``              (.numpy() / __dlpack__ / _to_impl — method-level)
#   ③ ``_tensor/_repr.py``               (display only)
#   ④ ``_types.py``                      (typing protocol; runtime is TYPE_CHECKING-only)
#   ⑤ ``serialization/__init__.py``      (checkpoint state_dict)
#       ``optim/optimizer.py``           (state_dict round-trip)
#       ``optim/lbfgs.py``               (state_dict round-trip; large internal state)
#       ``compile/_optim/compiler.py``   (state_dict / _step_entry only; method-scoped)
#   ⑥ ``utils/data/dataloader.py``       (external data ingest)
#
# We whitelist the core bridge files at *file* granularity for simplicity —
# CLAUDE.md's stricter method-level guarantee (only ``.numpy()`` /
# ``__dlpack__`` / ``_to_impl`` in ``tensor.py``) is enforced by code review.
# Large modules where only checkpoint state_dict export touches ``.numpy()``
# are scoped to the specific methods in ``SANCTIONED_METHODS`` below.
SANCTIONED: frozenset[str] = frozenset(
    {
        "_factories/converters.py",
        "_tensor/tensor.py",
        "_tensor/_repr.py",
        "_types.py",
        "serialization/__init__.py",
        "optim/optimizer.py",
        "optim/lbfgs.py",
        "utils/data/dataloader.py",
    }
)

SANCTIONED_METHODS: dict[str, frozenset[str]] = {
    "compile/_optim/compiler.py": frozenset({"state_dict", "_step_entry"})
}


class Violation(NamedTuple):
    path: pathlib.Path
    lineno: int
    col: int
    statement: str


def _is_numpy_module(name: str) -> bool:
    """``numpy`` or any ``numpy.*`` submodule."""
    return name == "numpy" or name.startswith("numpy.")


def _under_blocks(node: ast.AST, blocks: list[tuple[int, int]]) -> bool:
    """True when ``node`` is textually inside any ``(start, end)`` range in ``blocks``."""
    lineno = getattr(node, "lineno", 0)
    for start, end in blocks:
        if start <= lineno <= end:
            return True
    return False


def _under_type_checking(node: ast.AST, type_checking_blocks: list[tuple[int, int]]) -> bool:
    """True when ``node`` is textually inside an ``if TYPE_CHECKING:`` block.

    PEP 649 (Python 3.14) makes these blocks zero-cost at runtime — they
    only execute when a type checker walks them.  Numpy imports inside
    them never reach ``sys.modules``.
    """
    return _under_blocks(node, type_checking_blocks)


def _collect_type_checking_blocks(tree: ast.AST) -> list[tuple[int, int]]:
    """All ``if TYPE_CHECKING:`` block line ranges in ``tree``."""
    blocks: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        # Match ``if TYPE_CHECKING:`` and ``if typing.TYPE_CHECKING:``.
        test = node.test
        if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
            pass
        elif isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
            pass
        else:
            continue
        if not node.body:
            continue
        # ``node.end_lineno`` is set on every statement when Python parses
        # with feature-version >= 3.8 (always true on 3.14).  Use the
        # If-statement's own end_lineno rather than walking children —
        # ast.walk visits expression contexts (``Load``, ``Store``) that
        # lack lineno entirely.
        end = node.end_lineno or node.body[-1].end_lineno or node.lineno
        blocks.append((node.lineno, end))
    return blocks


def _collect_method_blocks(tree: ast.AST, method_names: frozenset[str]) -> list[tuple[int, int]]:
    """Line ranges of ``def <name>`` blocks whose name is in ``method_names``."""
    if not method_names:
        return []
    blocks: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            node.name in method_names and node.body
        ):
            end = node.end_lineno or node.body[-1].end_lineno or node.lineno
            blocks.append((node.lineno, end))
    return blocks


def _scan_file(path: pathlib.Path, rel_path: pathlib.Path | None = None) -> list[Violation]:
    """Return every unsanctioned numpy import or .numpy() call in ``path``."""
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as e:
        # Bubble up as a violation so the developer fixes the syntax first.
        return [Violation(path, e.lineno or 0, e.offset or 0, f"SyntaxError: {e.msg}")]

    if rel_path is None:
        try:
            rel_path = path.resolve().relative_to(LUCID_ROOT)
        except ValueError:
            rel_path = None
    allowed_methods = (
        SANCTIONED_METHODS.get(rel_path.as_posix(), frozenset())
        if rel_path is not None
        else frozenset()
    )

    type_checking_blocks = _collect_type_checking_blocks(tree)
    method_blocks = _collect_method_blocks(tree, allowed_methods)
    violations: list[Violation] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_numpy_module(alias.name) and not _under_type_checking(
                    node, type_checking_blocks
                ):
                    violations.append(
                        Violation(path, node.lineno, node.col_offset, f"import {alias.name}")
                    )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if _is_numpy_module(module) and not _under_type_checking(node, type_checking_blocks):
                names = ", ".join(a.name for a in node.names)
                violations.append(
                    Violation(path, node.lineno, node.col_offset, f"from {module} import {names}")
                )
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "numpy"
            and not _under_type_checking(node, type_checking_blocks)
            and not _under_blocks(node, method_blocks)
        ):
            violations.append(Violation(path, node.lineno, node.col_offset, ".numpy() call"))

    return violations


def _is_skipped(rel_path: pathlib.Path) -> bool:
    """``lucid/test/**`` and ``lucid/benchmarks/**`` are excluded."""
    parts = rel_path.parts
    return bool(parts and parts[0] in ("test", "benchmarks"))


def _is_sanctioned(rel_path: pathlib.Path) -> bool:
    return rel_path.as_posix() in SANCTIONED


def scan(root: pathlib.Path) -> list[Violation]:
    """Walk ``root`` and collect every unsanctioned numpy import."""
    violations: list[Violation] = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        if _is_skipped(rel):
            continue
        if _is_sanctioned(rel):
            continue
        violations.extend(_scan_file(path, rel_path=rel))
    return violations


def main(argv: list[str]) -> int:
    list_sanctioned = "--list" in argv
    quiet = "--quiet" in argv

    if list_sanctioned and not quiet:
        print(f"H4 sanctioned bridge files ({len(SANCTIONED)}):")
        for s in sorted(SANCTIONED):
            print(f"  lucid/{s}")
        print()

    violations = scan(LUCID_ROOT)

    if not violations:
        if not quiet:
            print(
                f"[check_numpy_h4] OK — scanned {sum(1 for _ in LUCID_ROOT.rglob('*.py'))} "
                "files, zero unsanctioned numpy imports."
            )
        return 0

    if not quiet:
        print(f"[check_numpy_h4] {len(violations)} H4 violation(s):\n")
        for v in violations:
            rel = v.path.relative_to(REPO_ROOT)
            print(f"  {rel}:{v.lineno}:{v.col}  {v.statement}")
        print(
            "\nH4 (CLAUDE.md §6.1) forbids ``import numpy`` outside the six "
            "sanctioned bridge\nfiles.  Either move the import inside an "
            "``if TYPE_CHECKING:`` block (runtime\nno-op under PEP 649) or "
            "route the data through ``to_bytes()`` / ``from_bytes()`` /\n"
            "``transfer_to_device()`` / the existing struct.pack helpers.  "
            "See [[retro-3-0-2-numpy-free-standalone]]\nin obsidian/ for the "
            "list of patterns we already replaced."
        )

    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
