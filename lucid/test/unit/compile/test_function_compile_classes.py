"""Every Lucid ``Function`` subclass is classified for ``lucid.compile`` (LCD-570b).

Each ``Function`` defined in ``lucid/`` (outside ``lucid/test/``) belongs to
one of five compile categories:

* ``TRACE_FORM`` — bookkeeping nodes whose compile recording takes their
  :attr:`~lucid.autograd.Function._trace_form` instead of registering the node.
* ``EXACT`` — nodes whose ``backward`` is the exact derivative of their
  recorded ``forward`` (:attr:`~lucid.autograd.Function._exact_derivative`), so
  a compiled backward may differentiate the recording when eager would also run
  ``backward``.
* ``TRACED_BACKWARD`` — nodes whose ``backward`` is traced into a
  ``VjpRegion`` (:attr:`~lucid.autograd.Function._traced_backward`) and lowered
  alongside the forward graph into the compiled executable.
* ``CUSTOM`` — nodes whose ``backward`` a compiled backward cannot run (either
  because ``backward`` is not the recorded ``forward``'s derivative, or because
  the op has no compile emitter or runs a host / nested-autograd backward).
* ``UNTRACED`` — nodes invoked only under ``untraced()`` so a compile recording
  never reaches ``Function.apply``.
* ``COMPILE_INTERNAL`` — nodes ``lucid.compile`` itself registers to execute a
  compiled step or segment.
"""

import ast
from pathlib import Path

TRACE_FORM: frozenset[str] = frozenset(
    {
        "_Snapshot",
        "_Overwritten",
        "_TruncQuotient",
        "_RoundDecimals",
        "_Erfcx",
        # Special edge-value backwards (LCD-772)
        "_Entr",
        "_HalfLineEdge",
        "_I0",
        "_LogGamma",
        "_Ndtri",
        "_Polygamma",
    }
)

EXACT: frozenset[str] = frozenset({"_CheckpointFn"})

TRACED_BACKWARD: frozenset[str] = frozenset(
    {
        "_MatrixPowerZero",
        "_LogAbsDet",
        "_StandardGamma",
        "_FakeQuantizeAffine",
        "_CholeskyAutograd",
        "_SVD",
        "_QR",
        "_Pinv",
        "_Eigh",
    }
)

CUSTOM: frozenset[str] = frozenset(
    {
        # FFT (no MPSGraph emitter yet)
        "_FftnAutograd",
        "_RfftnAutograd",
        "_IrfftnAutograd",
        # Linalg (complex / pivot-table factorizations without MPSGraph emitter)
        "_Eig",
        "_LUFactorGrad",
        "_LDLFactorGrad",
        # Special (host loop in forward)
        "_RegularizedGamma",
        # Host / nested-autograd ODE backward
        "_EventTimeGradient",
        "_AdjointSolve",
    }
)

UNTRACED: frozenset[str] = frozenset({"_Ldexp", "_MetalFrexp"})

COMPILE_INTERNAL: frozenset[str] = frozenset({"_CompiledStepFunction", "_SegmentFunction"})

ALL_CATEGORIES: tuple[tuple[str, frozenset[str]], ...] = (
    ("TRACE_FORM", TRACE_FORM),
    ("EXACT", EXACT),
    ("TRACED_BACKWARD", TRACED_BACKWARD),
    ("CUSTOM", CUSTOM),
    ("UNTRACED", UNTRACED),
    ("COMPILE_INTERNAL", COMPILE_INTERNAL),
)

_FUNCTION_BASES = frozenset({"Function", "_AutogradFunction"})


def _base_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _class_sets_attribute(node: ast.ClassDef, attr_name: str) -> bool:
    for stmt in node.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id == attr_name:
                    return True
        elif isinstance(stmt, ast.AnnAssign):
            if isinstance(stmt.target, ast.Name) and stmt.target.id == attr_name:
                return True
    return False


def _scan_lucid_functions() -> tuple[set[str], set[str], set[str], set[str]]:
    """Return ``(all_functions, trace_form_set, exact_derivative_set, traced_backward_set)``."""
    root = Path(__file__).resolve().parents[3]
    test_root = root / "test"
    found: set[str] = set()
    trace_forms: set[str] = set()
    exact_derivatives: set[str] = set()
    traced_backwards: set[str] = set()
    for path in sorted(root.rglob("*.py")):
        if test_root in path.parents:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            bases = {_base_name(b) for b in node.bases}
            if not (bases & _FUNCTION_BASES):
                continue
            found.add(node.name)
            if _class_sets_attribute(node, "_trace_form"):
                trace_forms.add(node.name)
            if _class_sets_attribute(node, "_exact_derivative"):
                exact_derivatives.add(node.name)
            if _class_sets_attribute(node, "_traced_backward"):
                traced_backwards.add(node.name)
    return found, trace_forms, exact_derivatives, traced_backwards


def test_categories_are_mutually_disjoint() -> None:
    seen: dict[str, str] = {}
    for category, members in ALL_CATEGORIES:
        for name in members:
            assert name not in seen, f"{name} is listed in both {seen[name]} and {category}"
            seen[name] = category


def test_every_lucid_function_is_classified_in_exactly_one_category() -> None:
    found, trace_forms, exact_derivatives, traced_backwards = _scan_lucid_functions()
    classified = frozenset().union(*(members for _, members in ALL_CATEGORIES))
    unclassified = sorted(found - classified)
    stale = sorted(classified - found)
    assert not unclassified, f"Unclassified Function subclasses in lucid/: {unclassified}"
    assert not stale, f"Classified names no longer in lucid/: {stale}"
    assert trace_forms == TRACE_FORM
    assert exact_derivatives == EXACT
    assert traced_backwards == TRACED_BACKWARD
