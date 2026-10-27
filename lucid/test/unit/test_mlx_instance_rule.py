"""Nothing in ``lucid/`` reaches for the ``mlx`` Python package by accident.

The engine runs on its own pinned MLX (``liblucid_mlx``, LCD-292).  A Python
``mlx`` in the same process is a second MLX instance with its own allocator,
buffer cache and command queue, so asking it about memory, the cache or
synchronisation answers for the wrong instance.  Two tests did exactly that
after the switch: a peak-memory comparison read ``0 < 0`` and a leak probe
read zero drift whether or not it leaked.  The engine binds every such query
(``metal_active_memory``, ``metal_peak_memory``, ``metal_cache_memory``,
``synchronize_gpu``, ...), so the only reason left to load ``mlx`` is to
exercise the user's package on purpose.

The scan is syntactic, not textual.  The repository loads modules from code
held in strings — ``python -c`` scripts, subprocess probes — so every string
constant is parsed as Python and scanned the same way, recursively.  A string
that is not Python (prose, a fragment) falls back to a tolerant pattern.
"""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]  # lucid/
_INTEROP = ROOT / "test/unit/interop"

# Places that load the user's `mlx` on purpose.
_ALLOWED = (
    # Interop with the `mlx` package is the subject under test.
    _INTEROP,
    # Opt-in speed comparisons against `mlx` itself, never run by the suite.
    ROOT / "benchmarks",
)

# Calls whose first argument names a module to load.
_LOADERS = frozenset({"importorskip", "import_module", "find_spec", "__import__"})

# For strings that are not Python: an import statement naming mlx anywhere in
# its comma list, or a loader call on it.
_TOLERANT = re.compile(
    r"\bimport\s+(?:[\w.]+(?:\s+as\s+\w+)?\s*,\s*)*mlx\b"
    r"|\bfrom\s+mlx(?:\.\w+)*\s+import\b"
    r"|\b(?:importorskip|import_module|find_spec|__import__)\(\s*[\"']mlx\b"
)

_MAX_DEPTH = 4  # code in a string in code in a string ... is still scanned


def _is_mlx(name: str | None) -> bool:
    return name is not None and (name == "mlx" or name.startswith("mlx."))


def _loader_name(func: ast.expr) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _loads_mlx(node: ast.AST) -> bool:
    if isinstance(node, ast.Import):
        return any(_is_mlx(alias.name) for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        return node.level == 0 and _is_mlx(node.module)
    if isinstance(node, ast.Call) and _loader_name(node.func) in _LOADERS and node.args:
        first = node.args[0]
        return (
            isinstance(first, ast.Constant)
            and isinstance(first.value, str)
            and _is_mlx(first.value)
        )
    return False


def _tree_hits(tree: ast.AST, depth: int) -> list[int]:
    """Line numbers in *tree* that load mlx, including inside string constants."""
    hits = []
    for node in ast.walk(tree):
        if _loads_mlx(node):
            hits.append(getattr(node, "lineno", 0))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if depth < _MAX_DEPTH and _text_loads_mlx(node.value, depth + 1):
                hits.append(node.lineno)
    return hits


def _text_loads_mlx(text: str, depth: int = 0) -> bool:
    if "mlx" not in text:
        return False
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return bool(_TOLERANT.search(text))
    return bool(_tree_hits(tree, depth))


def _scan(paths: list[Path]) -> list[str]:
    hits = []
    for path in paths:
        source = path.read_text(errors="ignore")
        for line in _tree_hits(ast.parse(source, filename=str(path)), 0):
            text = source.splitlines()[line - 1].strip() if line else ""
            hits.append(f"{path.relative_to(ROOT.parent)}:{line}: {text[:100]}")
    return hits


def _python_files(under: Path) -> list[Path]:
    return sorted(p for p in under.rglob("*.py") if "__pycache__" not in p.parts)


def _checked_files() -> list[Path]:
    return [p for p in _python_files(ROOT) if not any(p.is_relative_to(a) for a in _ALLOWED)]


def test_only_interop_tests_load_the_mlx_package() -> None:
    files = _checked_files()
    assert len(files) > 1000, f"scanned only {len(files)} files: the walk is broken"
    hits = _scan(files)
    assert not hits, (
        "the `mlx` package is a different MLX instance from the engine's; use the "
        "engine bindings (lucid._C.engine.metal_* / synchronize_gpu):\n" + "\n".join(hits)
    )


def test_the_rule_fires_where_mlx_is_loaded_on_purpose() -> None:
    """A positive control: without it, a scanner that never fires would pass."""
    assert _scan(_python_files(_INTEROP)), "the interop tests load mlx; the scan saw none"


# Assembled so that this file does not trip its own rule.
_M = "ml" + "x"


@pytest.mark.parametrize(
    "source",
    [
        f"import {_M}",
        f"import {_M}.core as mx",
        f"from {_M}.core import array",
        f"import gc, sys, {_M}.core as mx",
        f'probe = "import {_M}.core as mx\\n"',
        f'cmd = [sys.executable, "-c", "import {_M}.core"]',
        f'probe = "import os\\nimport {_M}.core"',
        f'mod = __import__("{_M}")',
        f'mx = pytest.importorskip("{_M}.core")',
        f'mx = importlib.import_module("{_M}.core")',
        f'ok = importlib.util.find_spec("{_M}") is not None',
        f'outer = "inner = \\"import {_M}.core\\""',
        f'prose = "first run: import gc, {_M}.core as mx; then"',
    ],
)
def test_every_loading_form_is_seen(source: str) -> None:
    assert _tree_hits(ast.parse(source), 0), source


@pytest.mark.parametrize(
    "source",
    [
        f'doc = "an ``{_M}.core.array`` is accepted"',
        f"# import {_M} in a comment is not code",
        f'name = "{_M}"',
        f"from .{_M} import helper",
        f'mod = __import__("{_M}_tools")',
    ],
)
def test_mentions_that_load_nothing_pass(source: str) -> None:
    assert not _tree_hits(ast.parse(source), 0), source
