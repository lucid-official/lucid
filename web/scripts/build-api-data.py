"""
build-api-data.py — Griffe → JSON API pipeline for Lucid docs

Reads lucid/ Python source statically (NumPy docstring parser),
outputs one JSON file per module to web/public/api-data/.

Usage:
    python scripts/build-api-data.py [--module MODULE] [--verbose]

    # Build all (default)
    python scripts/build-api-data.py

    # Single module PoC
    python scripts/build-api-data.py --module lucid.fft
"""

# fmt: off

from __future__ import annotations  # allowed in build scripts (not lucid/ source)

import argparse
import json
import logging
import re
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from collections.abc import Callable
from typing import Any

try:
    from tqdm import tqdm
except ImportError:                                          # graceful fallback
    tqdm = None                                              # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Griffe robustness shim — callable-module / overload collision
# ---------------------------------------------------------------------------
#
# ``lucid.compile`` is a *callable module*: the runtime swaps its class to a
# ModuleType subclass so ``lucid.compile(model)`` works, and ``lucid/__init__.pyi``
# advertises ``compile`` as ``@overload``'d functions so type-checkers can model
# the call.  Griffe's stub-overload merger then looks up ``lucid.compile`` while
# merging the package stub, finds the *submodule* (a ``Module``) under that name,
# and crashes on ``Module.parameters`` — taking down whichever module happens to
# be loaded first on a worker (the api-data for that module silently goes
# unbuilt).  Guard the merger so a name that resolves to a non-Function member is
# skipped rather than fatal; real function-overload merges are untouched.

_GRIFFE_GUARD_INSTALLED = False


def _install_griffe_overload_guard() -> None:
    """Make Griffe's stub-overload merge tolerant of callable-module name
    collisions (idempotent; safe to call before every loader build)."""
    global _GRIFFE_GUARD_INSTALLED
    if _GRIFFE_GUARD_INSTALLED:
        return
    try:
        import griffe._internal.merger as _merger
        from griffe import Function as _GFunction
    except Exception:
        return
    _orig = _merger._merge_overload_annotations

    def _guarded(function: Any, overloads: Any) -> Any:
        if not isinstance(function, _GFunction):
            return None
        return _orig(function, overloads)

    _merger._merge_overload_annotations = _guarded
    _GRIFFE_GUARD_INSTALLED = True

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT   = Path(__file__).parent.parent.parent  # lucid repo root
LUCID_SRC   = REPO_ROOT                            # lucid/ lives here
WEB_ROOT    = Path(__file__).parent.parent         # web/
OUT_DIR     = WEB_ROOT / "public" / "api-data"
GITHUB_BASE = "https://github.com/lucid-official/lucid/blob"

# ---------------------------------------------------------------------------
# Module manifest — auto-discovered from lucid/ directory structure
# ---------------------------------------------------------------------------
#
# The discovery walk recurses two levels under ``lucid/``, looking for
# directories that contain an ``__init__.py``.  Anything matching is exposed
# as ``lucid.<dir>`` (or ``lucid.<dir>.<subdir>`` at the second level).
#
# - Private packages (leading ``_``) are skipped automatically.
# - Internal-only packages are listed in ``_EXCLUDED_SUBPKGS`` below.
# - Special cases (class-modules, slug-path remappings, .py-file modules)
#   live in ``_MANIFEST_OVERRIDES`` — applied AFTER discovery.
#
# Adding a new public subpackage = create the directory with an
# ``__init__.py``.  No manual manifest edit required.

# Entire subtree excluded — neither this slug nor any descendant is documented.
_EXCLUDED_SUBTREE: set[str] = {
    "lucid.metal",       # GPU backend internals
    "lucid.backends",    # CPU/GPU dispatch internals
    "lucid.test",        # test infrastructure (parity fixtures, etc.)
    "lucid.benchmarks",  # internal perf scripts
    # Per-family weight-enum aggregator: NOT documented as its own package.
    # Each ``<Family>Weights`` class is already a member of its model family
    # (e.g. lucid.models.vision.alexnet.AlexNetWeights) and is surfaced there
    # under the family's "Weights" sidebar slot.  The aggregator would just be
    # a flat 128-entry duplicate.
    "lucid.models.weights",
}

# Self excluded — this slug is dropped but descendants still get discovered.
# Used for umbrella packages whose ``__init__.py`` has no user-facing content
# of its own but whose children DO get documented separately.
#
# Historical note: ``lucid.utils`` used to live here on the assumption it
# was a pure umbrella, but its ``__all__`` re-exports ``checkpoint`` from
# ``lucid.utils.checkpoint`` so the slug DOES have a user-facing function
# and must be documented.
_EXCLUDED_SELF: set[str] = {
    "lucid.nn.modules",         # umbrella; content re-exposed via lucid.nn path-grouping
    "lucid.models._utils",      # internal helper subpackage of lucid.models
}

# Family-root slugs: their ``__init__.py`` is intentionally minimal but each
# is given its own browsable index page populated with cards for every
# direct sub-family.  The build script post-processes these JSONs to
# inject ``family_groups`` (one entry per child package).
_FAMILY_ROOTS: set[str] = {
    "lucid.models.vision",
    "lucid.models.text",
    "lucid.models.generative",
    "lucid.models.multimodal",
}


def _is_family_leaf(slug: str) -> bool:
    """A slug like ``lucid.models.vision.resnet`` — depth-4 under a
    family root.  These are the actual model-family pages (resnet, vit,
    bert, …) that own ``ModelConfig._canonical_name`` and the 4-slot
    sidebar structure."""
    parts = slug.split(".")
    if len(parts) != 4:
        return False
    return ".".join(parts[:3]) in _FAMILY_ROOTS

# Special-case slug→griffe-path mappings (layered ON TOP of auto-discovery).
_MANIFEST_OVERRIDES: dict[str, str] = {
    "lucid.tensor":  "lucid._tensor.tensor",   # class-module: → Tensor only
    "lucid.signal":  "lucid.signal.windows",   # redirect to subpkg with content
    "lucid.nn.init": "lucid.nn.init",          # .py file at module root, not a dir
}

# Synthesised slugs — these are NOT real Python packages with their own
# discoverable ``__init__.py`` content.  Instead, the build pipeline loads
# the lucid top-level once, partitions its members by the subcategory each
# falls into (via ``_LUCID_NAME_OVERRIDES``), and emits one JSON per
# partition.  Used for groups whose runtime population is dynamic and
# therefore invisible to Griffe's static walker:
#   - ``_factories/``     → registered via lucid's _load_factories hook
#   - ``_ops/``           → bound at import time by ``_populate_free_fns()``
#   - ``_ops/composite/`` → dynamically populated from submodule ``__all__``s
# Each entry maps the public slug to the subcategory tag set by
# ``_LUCID_NAME_OVERRIDES``.
_SYNTH_SLUGS: dict[str, str] = {
    "lucid.creation":      "factories",
    "lucid.ops":           "ops",
    "lucid.ops.composite": "composite",
}

# These slugs have no module object behind them, so unlike every other page
# there is no ``__init__`` docstring for the header to come from — without
# this map their pages render with a blank summary.  Keep the voice matched
# to the real module docstrings: what the group is, then what is non-obvious.
#
# Written in the POST-conversion form (markdown, single backticks): real
# docstrings reach the JSON through the reST->markdown converter, but these
# are injected straight into it and never pass through one.  Double backticks
# here reach the page as literal reST and trip the audit's no-rest-leak rule.
_SYNTH_SUMMARIES: dict[str, tuple[str, str]] = {
    "lucid.creation": (
        "Tensor construction, and the seeding of the default generator.",
        "Thirty entry points: conversion (`tensor`, `as_tensor`, "
        "`from_numpy`), filled and `*_like` constructors, the range "
        "builders `arange` / `linspace`, the sampling family, the "
        "generator controls (`seed`, `get_rng_state` / `set_rng_state`, "
        "`initial_seed`, `randperm`) and DLPack exchange.\n\n"
        "Every constructor takes `device=`, and passing it is what keeps a "
        "pipeline on one device: a helper built without it lands on the CPU "
        "and forces a round trip the moment it meets Metal data.\n\n"
        "CPU and Metal share one default generator but consume it "
        "differently per draw — a CPU `rand(4)` advances its counter by 4, "
        "a Metal `rand(4)` by one block — so the same seed does not "
        "reproduce the same stream across devices; comparing them means "
        "comparing deterministic ops, replaying state with "
        "`get_rng_state()` / `set_rng_state()`, or `eval()` for anything "
        "stochastic."
    ),
    "lucid.ops": (
        "The engine-bound primitive ops, grouped by arity.",
        "158 ops registered in `lucid/_ops/_registry.py` and bound onto the "
        "top level at import time — elementwise maths, comparisons, "
        "reductions, shape moves and their in-place `_`-suffixed forms.\n\n"
        "These are the leaves of the autograd graph: each maps to one engine "
        "kernel with a hand-written backward, which is what distinguishes "
        "them from the composites in `lucid.ops.composite`.  The page is "
        "split by arity (unary / binary / variadic) because a flat list of "
        "158 is not navigable."
    ),
    "lucid.ops.composite": (
        "Ops assembled from the primitives rather than from a kernel.",
        "118 functions defined in `lucid/_ops/composite/` — the aliases "
        "(`absolute`, `arctan2`, ...), the fused arithmetic helpers "
        "(`addmm`, `addcdiv`, ...), and the statistical and linear-algebra "
        "conveniences.\n\n"
        "They carry no kernel and no backward of their own: gradients fall "
        "out of the primitives they are written in.  That makes them cheap "
        "to add and correct by construction, at the cost of materialising "
        "the intermediates a fused kernel would not."
    ),
}


def _discover_manifest() -> dict[str, str]:
    """Walk lucid/ to auto-discover all documentable subpackages.

    Returns a manifest dict ``{slug → griffe_path}``.  Walks two levels of
    public directories under lucid/, then layers the explicit overrides
    above.  The bare ``lucid`` top-level is intentionally NOT included —
    its content is fully re-exported from the private subpackages that
    ``_MANIFEST_OVERRIDES`` already surfaces under friendly aliases
    (``lucid.creation``, ``lucid.ops``, ``lucid.ops.composite``).
    """
    manifest: dict[str, str] = {}
    lucid_dir = LUCID_SRC / "lucid"
    if not lucid_dir.is_dir():
        return manifest

    def _is_public_pkg(p: Path) -> bool:
        return (
            p.is_dir()
            and not p.name.startswith("_")
            and (p / "__init__.py").exists()
        )

    for first in sorted(lucid_dir.iterdir()):
        if not _is_public_pkg(first):
            continue
        slug = f"lucid.{first.name}"
        if slug in _EXCLUDED_SUBTREE:
            continue
        if slug not in _EXCLUDED_SELF:
            manifest[slug] = slug
        for second in sorted(first.iterdir()):
            if not _is_public_pkg(second):
                continue
            sub_slug = f"{slug}.{second.name}"
            if sub_slug in _EXCLUDED_SUBTREE or sub_slug in _EXCLUDED_SELF:
                continue
            manifest[sub_slug] = sub_slug
            # Family-root walk: lucid.models has a 4-level structure
            # (lucid/models/{vision,text,generative}/{family}/) — every
            # leaf family directory becomes its own page so the family
            # cards on the family-root index are clickable.
            if sub_slug in _FAMILY_ROOTS:
                for third in sorted(second.iterdir()):
                    if not _is_public_pkg(third):
                        continue
                    leaf_slug = f"{sub_slug}.{third.name}"
                    if leaf_slug in _EXCLUDED_SUBTREE or leaf_slug in _EXCLUDED_SELF:
                        continue
                    manifest[leaf_slug] = leaf_slug

    manifest.update(_MANIFEST_OVERRIDES)
    return manifest


MODULE_MANIFEST: dict[str, str] = _discover_manifest()

# ---------------------------------------------------------------------------
# Griffe helpers
# ---------------------------------------------------------------------------

def _get_commit_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except Exception:
        return "main"


def _source_link(obj: Any, sha: str) -> str | None:
    """Return a GitHub permalink for a griffe object."""
    try:
        if obj.filepath is None:
            return None
        rel = Path(obj.filepath).relative_to(REPO_ROOT)
        line = obj.lineno or 1
        return f"{GITHUB_BASE}/{sha}/{rel}#L{line}"
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Source-file → subcategory mapping (used for the lucid top-level sidebar)
# ---------------------------------------------------------------------------

# Order matters: more specific paths first.  Each entry maps a relative path
# fragment under lucid/ to a short subcategory slug.  These rules apply only
# to files in lucid's PRIVATE subpackages — public subpackages like
# ``lucid.autograd`` / ``lucid.serialization`` get per-file grouping via the
# generic fallback inside :func:`_subcategory`.
_SUBCATEGORY_RULES: list[tuple[str, str]] = [
    ("lucid/_ops/composite", "composite"),
    ("lucid/_ops",           "ops"),
    ("lucid/_factories",     "factories"),
    ("lucid/_tensor",        "tensor"),
    ("lucid/_types.py",      "types"),
    ("lucid/_dtype.py",      "dtypes"),
    ("lucid/dtypes.py",      "dtypes"),
    ("lucid/_device.py",     "device"),
    ("lucid/_globals.py",    "globals"),
    ("lucid/_threads.py",    "threads"),
    ("lucid/_dispatch.py",   "dispatch"),
    ("lucid/_vmap_ctx.py",   "vmap"),
]


# File stems split out for size that belong to the same docs subcategory group.
_STEM_ALIASES: dict[str, str] = {
    "conv_transpose": "conv",
    # LCD-614 split ``nn/functional/activations.py``'s softmax / distance /
    # shrinkage families into their own private modules for size, but they
    # re-export through ``activations.py`` and are still documented as one
    # family — without this, the sidebar would grow three new one-function
    # groups ("Softmax", "Distance", "Shrinkage") instead of staying under
    # "Activations" (LCD-628).
    "softmax": "activations",
    "distance": "activations",
    "shrinkage": "activations",
}


def _subcategory(
    obj: Any,
    module_root: Path | None = None,
    *,
    is_lucid_toplevel: bool = False,
) -> str | None:
    """Determine the source-directory-based subcategory for a Griffe object.

    Returns a slash-separated path slug based on where the symbol is actually
    defined in the lucid/ source tree, **relative to the module being
    serialised**.  Used by the sidebar to mirror the on-disk directory
    structure rather than the flat re-exported namespace.

    Resolution order (deterministic, applied in this exact order):

      1. **(lucid top-level only)** Name-based override — dynamic loaders
         (``_FACTORY_NAMES`` / ``_OPS_NAMES`` / etc.) that Griffe can't
         statically resolve.  ``is_lucid_toplevel=True`` MUST be set, otherwise
         this step is skipped — preventing names like ``relu`` (which is in
         ``_OPS_NAMES``) from miscategorising members of unrelated modules
         like ``lucid.nn.functional``.
      2. **(lucid top-level only)** Path-based rules for lucid's private
         subpackages (``_factories/``, ``_ops/``, ``_tensor/``, ...).
      3. **Module-root-relative path** — for any module, the member's source
         file path relative to the module's ``__init__.py`` directory.
         ``lucid/nn/modules/conv.py`` (module root ``lucid/nn/``) → subcategory
         ``"modules/conv"``.  Multi-segment slugs let the sidebar render a
         tree that mirrors ``ls lucid/nn/``.
      4. **Fallback** (no module_root context, e.g. Tensor class methods):
         file basename with leading underscores stripped; ``__init__.py`` → None.
    """
    try:
        if is_lucid_toplevel and obj.name in _LUCID_NAME_OVERRIDES:
            return _LUCID_NAME_OVERRIDES[obj.name]
        if obj.filepath is None:
            return None
        p = Path(obj.filepath)
        if is_lucid_toplevel:
            rel = str(p.relative_to(REPO_ROOT)).replace("\\", "/")
            for path_frag, slug in _SUBCATEGORY_RULES:
                if rel.startswith(path_frag):
                    return slug
        # Module-root-relative path (slash-separated, multi-segment).
        if module_root is not None:
            try:
                rr = p.relative_to(module_root)
                parts = list(rr.parts)
                if not parts:
                    return None
                last = parts[-1]
                if last == "__init__.py":
                    return None    # member is defined in the package's own init
                if last.endswith(".py"):
                    parts[-1] = last[:-3]
                parts = [_STEM_ALIASES.get(pp.lstrip("_"), pp.lstrip("_")) for pp in parts]
                parts = [pp for pp in parts if pp]
                if not parts:
                    return None
                return "/".join(parts)
            except ValueError:
                pass
        # Fallback when no module_root context: file basename only.
        if p.name == "__init__.py":
            return None
        name = _STEM_ALIASES.get(p.stem.lstrip("_"), p.stem.lstrip("_"))
        return name or None
    except Exception:
        return None


def _load_lucid_name_overrides() -> dict[str, str]:
    """Parse ``lucid/__init__.py`` to map dynamic-loaded names to subcategories.

    Reads the ``_FACTORY_NAMES``, ``_OPS_NAMES``, and ``_SCATTER_NAMES``
    frozensets defined at module scope, plus ``COMPOSITE_NAMES`` from the
    ``lucid._ops.composite`` package.  These sets are what the runtime
    ``__getattr__`` actually checks, so they are the single source of truth
    for the top-level namespace.
    """
    import ast

    mapping: dict[str, str] = {}
    init_path = LUCID_SRC / "lucid" / "__init__.py"

    NAME_TO_SLUG = {
        "_FACTORY_NAMES":       "factories",
        "_OPS_NAMES":           "ops",
        "_SCATTER_NAMES":       "ops",
        "_METHOD_ALIASES":      "ops",
        "_GRAD_NAMES":          "autograd",
        "_PREDICATE_NAMES":     "predicates",
        "_SERIALIZATION_NAMES": "serialization",
        "_TYPE_ALIAS_NAMES":    "types",
    }

    # Hard-coded overrides for top-level names that aren't in any *_NAMES set
    # but still need a sensible bucket (dtype singletons, deterministic toggles).
    mapping.update({
        "float16": "dtypes", "float32": "dtypes", "float64": "dtypes",
        "bfloat16": "dtypes", "int8": "dtypes", "int16": "dtypes",
        "int32": "dtypes", "int64": "dtypes", "bool_": "dtypes",
        "complex64": "dtypes",
        "use_deterministic_algorithms": "globals",
        "are_deterministic_algorithms_enabled": "globals",
    })

    try:
        tree = ast.parse(init_path.read_text(encoding="utf-8"))
    except Exception:
        return mapping

    def _iter_assigns(node_iter: Any) -> Any:
        """Yield (target_name, value_node) for both Assign and AnnAssign."""
        for n in node_iter:
            if isinstance(n, ast.Assign) and len(n.targets) == 1:
                tgt = n.targets[0]
                if isinstance(tgt, ast.Name):
                    yield tgt.id, n.value
            elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.value is not None:
                yield n.target.id, n.value

    def _extract_string_set(value_node: Any) -> set[str]:
        """Extract a set of string literals from `frozenset([...])`, `{...}`, or `[...]`."""
        try:
            if (
                isinstance(value_node, ast.Call)
                and getattr(value_node.func, "id", None) == "frozenset"
                and value_node.args
            ):
                value_node = value_node.args[0]
            return {n for n in ast.literal_eval(value_node) if isinstance(n, str)}
        except Exception:
            return set()

    # ``sorted`` at every insertion point: this mapping's *insertion order*
    # decides the member order of the synthetic slugs (``build_synth`` appends
    # in ``_LUCID_NAME_OVERRIDES`` order).  Iterating a set of strings varies
    # with PYTHONHASHSEED, so without this the emitted JSON differs run to run
    # and the drift gates can't tell a real change from reshuffled output.
    for target_id, value_node in _iter_assigns(ast.walk(tree)):
        if target_id in NAME_TO_SLUG:
            slug = NAME_TO_SLUG[target_id]
            for n in sorted(_extract_string_set(value_node)):
                mapping[n] = slug

    # Composite ops: COMPOSITE_NAMES is built at runtime from each submodule's
    # ``__all__``.  Static analysis: walk every submodule in
    # ``lucid/_ops/composite/*.py`` and union their ``__all__`` lists.
    comp_dir = LUCID_SRC / "lucid" / "_ops" / "composite"
    if comp_dir.exists():
        for comp_file in sorted(comp_dir.glob("*.py")):
            if comp_file.name == "__init__.py":
                continue
            try:
                comp_tree = ast.parse(comp_file.read_text(encoding="utf-8"))
                for target_id, value_node in _iter_assigns(ast.walk(comp_tree)):
                    if target_id == "__all__":
                        for n in sorted(_extract_string_set(value_node)):
                            mapping[n] = "composite"
            except Exception:
                pass

    # OpEntry registry (``lucid/_ops/_registry.py``): the runtime source of
    # truth for which engine ops are exposed.  Extract every ``OpEntry(...)``
    # call's first positional arg (the op name) — these include some names
    # not in ``_OPS_NAMES`` (``dot``, ``inner``, ``outer``) plus all the
    # in-place variants (``add_``, ``sub_``, ``mul_``, ...).
    reg_path = LUCID_SRC / "lucid" / "_ops" / "_registry.py"
    try:
        reg_tree = ast.parse(reg_path.read_text(encoding="utf-8"))
        for node in ast.walk(reg_tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "OpEntry"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                op_name = node.args[0].value
                # Don't clobber a more-specific slug if one is already set.
                mapping.setdefault(op_name, "ops")
    except Exception:
        pass

    return mapping


def _load_op_arity() -> dict[str, int]:
    """Parse ``lucid/_ops/_registry.py`` to map each OpEntry name to its arity.

    OpEntry signature is ``OpEntry(name, callable, arity, ...)`` — the third
    positional arg is the arity.  Used by ``build_synth`` to subcategorise
    the ``lucid.ops`` slug by unary / binary / ternary / etc. so the page
    isn't one giant flat list of 143 ops.
    """
    import ast
    result: dict[str, int] = {}
    reg_path = LUCID_SRC / "lucid" / "_ops" / "_registry.py"
    try:
        tree = ast.parse(reg_path.read_text(encoding="utf-8"))
    except Exception:
        return result
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "OpEntry"
            and len(node.args) >= 3
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and isinstance(node.args[2], ast.Constant)
            and isinstance(node.args[2].value, int)
        ):
            result[node.args[0].value] = node.args[2].value
    return result


_OP_ARITY: dict[str, int] = _load_op_arity()


_LUCID_NAME_OVERRIDES: dict[str, str] = _load_lucid_name_overrides()


def _annotation_str(ann: Any) -> str | None:
    """Convert a griffe annotation expression to a plain string.

    Strips reST inline markup — attribute / param annotations occasionally
    carry prose-y type lines like ``Parameter, shape ``(N, *)``` that would
    otherwise leak double-backticks into the rendered type pill."""
    if ann is None:
        return None
    try:
        return _rst_strip_for_code(str(ann))
    except Exception:
        return None


def _rst_inline_markup(text: str) -> str:
    """Non-math reST inline markup → markdown.  Shared by ``_rst_to_text``
    (docstring bodies) and ``_rst_math_to_markdown`` (family ``theory`` /
    ``citation`` blocks) so every prose surface gets identical treatment —
    otherwise roles like ``:func:`` leak only on the family theory pages.

    Order matters: hyperlinks first (they own the backticks), then
    double-backtick code, then ``:cite:`` removal, then the generic role pass
    (so a citation never collapses into stray inline code)."""
    # reST external hyperlinks: `text <url>`_ / `text <url>`__ → [text](url).
    # Must run before the single-backtick text reaches the markdown renderer,
    # which would otherwise treat the whole ``text <url>`` as an inline-code
    # span and leak the trailing ``_`` (the citation links in model docstrings).
    text = re.sub(
        r"`([^`<]+?)\s*<((?:https?|ftp|mailto):[^>]+)>`__?",
        r"[\1](\2)",
        text,
    )
    # Double-backtick code: ``code`` → `code`
    text = re.sub(r"``([^`]+)``", r"`\1`", text)
    # Remove citations first so the generic role pass below doesn't collapse
    # them into stray inline code.  Covers :cite:`k`, :cite:t:`k`, :cite:p:`k`.
    text = re.sub(r":cite:[a-z]*:?`[^`]+`", "", text)
    # Cross-references / inline roles → `target`.  Generic over ANY Sphinx
    # role (:class:`Foo`, :func:`bar`, :file:`p`, :py:meth:`x`, …) so a role we
    # never enumerated can't leak to the reader as raw ``:role:`x``` text — it
    # was an un-handled :file: role that surfaced on the compile page.
    text = re.sub(
        r":[a-zA-Z][\w.+-]*(?::[a-zA-Z][\w.+-]*)*:`~?([^`]+)`",
        r"`\1`",
        text,
    )
    # reST literal-block marker ``::`` at end of a line → ``:`` (the indented
    # block that follows is already a 4-space code block markdown renders
    # verbatim); a lone ``::`` line is dropped.  Only matches line-end so an
    # inline ``std::vector`` mid-sentence is left intact.
    text = re.sub(r"(\S)[ \t]*::([ \t]*(?:\n|$))", r"\1:\2", text)
    text = re.sub(r"(?m)^[ \t]*::[ \t]*$", "", text)
    return text


def _rst_strip_for_code(text: str) -> str:
    """Strip reST inline markup down to the *bare* token — for content that
    renders inside a code block (Examples), where inline-code backticks would
    show up literally.  ``:class:`BatchNorm1d``` → ``BatchNorm1d``."""
    if not text:
        return text
    text = re.sub(
        r"`([^`<]+?)\s*<((?:https?|ftp|mailto):[^>]+)>`__?", r"\1 (\2)", text
    )
    text = re.sub(r"``([^`]+)``", r"\1", text)
    text = re.sub(r":cite:[a-z]*:?`[^`]+`", "", text)
    text = re.sub(
        r":[a-zA-Z][\w.+-]*(?::[a-zA-Z][\w.+-]*)*:`~?([^`]+)`", r"\1", text
    )
    # reST literal-block marker at line end (``form::`` → ``form:``); lone ``::``
    # line dropped.  Inline ``std::vector`` mid-line is left intact.
    text = re.sub(r"(\S)[ \t]*::([ \t]*(?:\n|$))", r"\1:\2", text)
    text = re.sub(r"(?m)^[ \t]*::[ \t]*$", "", text)
    return text


# Header of a reST explicit-markup directive: ``.. <name>:: <arg>`` at the
# start of a line.  The body is the following block indented deeper than the
# directive itself (reST's indentation-delimited block rule), captured by the
# walk in ``_convert_directive_blocks``.
_DIRECTIVE_HEADER = re.compile(
    r"^(?P<indent>[ \t]*)\.\.[ \t]+(?P<name>[\w+-]+)::(?P<arg>[^\n]*)\n",
    re.MULTILINE,
)

# reST admonitions → a markdown blockquote with a bold label.
_ADMONITIONS = {
    "note": "Note", "warning": "Warning", "tip": "Tip", "hint": "Hint",
    "important": "Important", "caution": "Caution", "attention": "Attention",
    "danger": "Danger", "error": "Error", "seealso": "See also",
    "admonition": "Note",
}


def _convert_directive_blocks(text: str) -> str:
    """Convert reST explicit-markup directives to markdown.

    Handles the block forms whose body is indentation-delimited (a pure regex
    can't honour that rule — blank lines share the zero-indent state), so we
    walk line-by-line:

      - ``.. math::``                  → ``$$ … $$``      (display math)
      - ``.. code-block:: <lang>`` /
        ``.. code:: <lang>`` /
        ``.. sourcecode:: <lang>``     → fenced ```` ```<lang> … ``` ````
      - ``.. note:: / .. warning:: …`` → ``> **Label:** …`` (blockquote)
      - any other directive            → body kept, header dropped

    Generic so a directive we haven't special-cased never leaks its
    ``.. name::`` header as raw text on the page.
    """
    out: list[str] = []
    cursor = 0
    for m in _DIRECTIVE_HEADER.finditer(text):
        if m.start() < cursor:
            continue
        out.append(text[cursor:m.start()])
        name = m.group("name").lower()
        arg = m.group("arg").strip()
        directive_indent = len(m.group("indent").expandtabs(4))

        # Walk the body: blank lines, or lines indented strictly deeper than
        # the directive.  Stop at the first non-blank line at/under its indent.
        lines: list[str] = []
        scan = m.end()
        while scan < len(text):
            nl = text.find("\n", scan)
            line = text[scan : nl if nl != -1 else len(text)]
            if not line.strip():
                lines.append(line)
                scan = nl + 1 if nl != -1 else len(text)
                continue
            lead = len(line) - len(line.lstrip(" \t"))
            if len(line[:lead].expandtabs(4)) > directive_indent:
                lines.append(line)
                scan = nl + 1 if nl != -1 else len(text)
                continue
            break
        while lines and not lines[-1].strip():
            lines.pop()
        body = textwrap.dedent("\n".join(lines)).strip("\n")

        if name == "math":
            b = body.strip()
            # KaTeX renders TeX quote ligatures (``word'') literally — normalise
            # the paired form to straight quotes so they don't show as backticks.
            b = re.sub(r"``([^`']+?)''", r'"\1"', b)
            if "&" in b and "\\begin{" not in b and "\\end{" not in b:
                b = f"\\begin{{aligned}}\n{b}\n\\end{{aligned}}"
            out.append(f"\n\n$$\n{b}\n$$\n\n")
        elif name in ("code-block", "code", "sourcecode"):
            lang = arg if re.fullmatch(r"[A-Za-z0-9_+-]+", arg) else ""
            out.append(f"\n\n```{lang}\n{body}\n```\n\n")
        elif name in _ADMONITIONS:
            quoted = "\n".join(
                f"> {ln}" if ln.strip() else ">" for ln in body.split("\n")
            )
            out.append(f"\n\n> **{_ADMONITIONS[name]}:**\n>\n{quoted}\n\n")
        else:
            # Unknown directive: keep the (dedented) body, drop the header.
            out.append(f"\n\n{body}\n\n" if body else "\n\n")
        cursor = scan
    out.append(text[cursor:])
    return "".join(out)


def _fence_doctests(text: str) -> str:
    """Wrap doctest blocks (consecutive ``>>>`` / ``...`` lines) in a fenced
    ```` ```python ```` code block.  Without this, a doctest authored in prose
    (not under an ``Examples`` header) renders as a markdown blockquote — the
    ``>>>`` triggers ``>`` blockquote nesting — and the code collapses into a
    run-on line (the transforms module overview)."""
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith(">>>"):
            block: list[str] = []
            while i < len(lines) and lines[i].lstrip().startswith((">>>", "...")):
                block.append(lines[i])
                i += 1
            out.append("```python")
            out.extend(block)
            out.append("```")
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)


def _inline_math_sub(text: str) -> str:
    """Inline math ``:math:`expr``` → ``$expr$`` (multi-line collapsed)."""
    def _repl(m: re.Match[str]) -> str:
        inner = " ".join(m.group(1).split())
        inner = re.sub(r"``([^`']+?)''", r'"\1"', inner)   # TeX quotes → "
        return f"${inner}$"
    return re.sub(r":math:`([^`]+)`", _repl, text, flags=re.DOTALL)


def _apply_outside_protected(text: str, fn: Callable[[str], str]) -> str:
    r"""Apply ``fn`` to every segment of ``text`` that is NOT inside a protected
    region — a fenced code block (```` ```…``` ````) or math (``$$…$$`` /
    ``$…$``).  The inline-markup passes (double-backtick collapse, ``::``,
    roles) must not run inside these: they would corrupt code fences and mangle
    LaTeX markup (e.g. ``\text{``mean''}`` — LaTeX left-quotes — would lose a
    backtick)."""
    parts = re.split(r"(```[\s\S]*?```|\$\$[\s\S]*?\$\$|\$[^$\n]+?\$)", text)
    return "".join(p if i % 2 else fn(p) for i, p in enumerate(parts))


def _rst_to_text(text: str) -> str:
    """Convert reST markup to markdown-friendly text with $...$ math notation.

    Converts:
      - ``.. math::`` / ``.. code-block::`` / ``.. note::`` …  block directives
      - ``>>>`` doctest blocks         → fenced ```` ```python ``` ````
      - ``:math:`expr```               → $expr$   (inline math)
      - ````code````                   → `code`   (inline code)
      - ``:cite:`k```, ``:cite:t:`k``` → (removed)
      - ``\\`text <url>\\`_``            → [text](url)  (reST hyperlink)
      - ``::`` literal-block markers    → ``:``
      - ANY other role ``:role:`Foo``` → `Foo`    (generic cross-reference pass)
    """
    text = _convert_directive_blocks(text)   # .. math:: / code-block:: / note::
    text = _fence_doctests(text)             # >>> blocks → ```python fences
    # Inline passes (math + roles/code/cite/::) run only OUTSIDE fenced code so
    # they never mangle the ``` fences produced above.
    text = _apply_outside_protected(
        text, lambda seg: _rst_inline_markup(_inline_math_sub(seg))
    )
    return text.strip()


_AST_DOC_INDEX: list[tuple[str, dict[str, list[ast.ClassDef]], dict[str, list[tuple[ast.AST, str]]]]] | None = None


def _get_ast_doc_index() -> list[tuple[str, dict[str, list[ast.ClassDef]], dict[str, list[tuple[ast.AST, str]]]]]:
    """Build and cache the per-file AST docstring index across IMPL_DIRS once per process."""
    global _AST_DOC_INDEX
    if _AST_DOC_INDEX is not None:
        return _AST_DOC_INDEX
    import ast

    impl_dirs = [
        LUCID_SRC / "lucid" / "_factories",
        LUCID_SRC / "lucid" / "_ops",
        LUCID_SRC / "lucid" / "autograd",
        LUCID_SRC / "lucid" / "serialization",
        LUCID_SRC / "lucid",
    ]
    index: list[tuple[str, dict[str, list[ast.ClassDef]], dict[str, list[tuple[ast.AST, str]]]]] = []
    seen: set[Path] = set()
    for impl_dir in impl_dirs:
        for fpath in sorted(impl_dir.rglob("*.py")):
            if fpath in seen:
                continue
            seen.add(fpath)
            try:
                text = fpath.read_text(encoding="utf-8")
                tree = ast.parse(text)
            except (SyntaxError, OSError):
                continue
            class_nodes: dict[str, list[ast.ClassDef]] = {}
            by_name: dict[str, list[tuple[ast.AST, str]]] = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_nodes.setdefault(node.name, []).append(node)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    doc = ast.get_docstring(node)
                    if doc and doc.strip():
                        by_name.setdefault(node.name, []).append((node, doc))
            if by_name:
                index.append((text, class_nodes, by_name))
    _AST_DOC_INDEX = index
    return index


def _get_ast_docstring(fn_name: str, class_name: str | None = None) -> str | None:
    """Search implementation .py files for fn_name's docstring via AST."""
    import ast

    for text, class_nodes, by_name in _get_ast_doc_index():
        hits = by_name.get(fn_name)
        if not hits:
            continue
        file_mentions_class = class_name is not None and class_name in text
        for node, doc in hits:
            if class_name is not None:
                c_list = class_nodes.get(class_name, ())
                if any(child is node for c in c_list for child in ast.walk(c)):
                    return doc
                if file_mentions_class:
                    return doc
            else:
                return doc
    return None


_GOOGLE_HEADER_RE = re.compile(
    r"^[ \t]*(Args|Arguments|Returns|Yields|Raises|Note|Notes|"
    r"Warning|Warnings|Example|Examples|Attributes|See Also|References)\s*:\s*$",
    re.MULTILINE,
)
_NUMPY_UNDERLINE_RE = re.compile(r"^[ \t]*-{3,}[ \t]*$", re.MULTILINE)


def _detect_docstring_style(raw_text: str) -> str | None:
    """Return ``"google"`` / ``"numpy"`` / ``None`` based on header markers.

    Lucid's source tree mixes Google-style (``Args:``) and NumPy-style
    (``Parameters\n----------``) docstrings.  The Griffe parser picked at
    call sites used to be hard-coded to ``Parser.numpy``, which silently
    treated Google-style ``Args:`` bodies as prose — every parameter list
    was dropped on the floor.  We sniff for either family's marker and let
    :func:`_parse_docstring` dispatch the right parser.
    """
    if not raw_text:
        return None
    has_numpy = bool(_NUMPY_UNDERLINE_RE.search(raw_text))
    has_google = bool(_GOOGLE_HEADER_RE.search(raw_text))
    if has_google and not has_numpy:
        return "google"
    if has_numpy and not has_google:
        return "numpy"
    return None


def _parse_docstring(obj: Any, parser: Any) -> dict[str, Any]:
    """Parse a griffe object's docstring into structured sections."""
    result: dict[str, Any] = {
        "summary": None,
        "extended": None,
        "parameters": [],
        "returns": None,
        "raises": [],
        "examples": [],
        "notes": [],
        "attributes": [],
        "warns": [],
        # ``See Also`` admonitions are parsed out into structured
        # ``{name, description}`` entries so the frontend can render
        # each item as a hyperlink to the referenced symbol's docs
        # page.  Previously they landed in ``notes`` as plain text and
        # readers had to copy-paste names to navigate.
        "see_also": [],
    }

    # Prefer the Griffe docstring object; fall back to the AST-extracted __doc__
    # when the stub (.pyi) has no docstring but the .py implementation does.
    if not obj.docstring:
        fn_name = getattr(obj, "name", None) or obj.path.split(".")[-1]
        # When this method belongs to a class, pass the class name so the
        # AST fallback doesn't pick up a same-named method on a different
        # class (e.g. Tensor.__eq__ vs device.__eq__).  We extract the
        # enclosing-class name from ``obj.path`` (more reliable than
        # ``obj.parent`` which Griffe doesn't always populate the way we
        # expect for stub-loaded methods).
        class_ctx: str | None = None
        path_parts = (obj.path or "").split(".")
        if len(path_parts) >= 2:
            candidate = path_parts[-2]
            # Heuristic: a class name starts with an uppercase letter (PEP 8)
            # and isn't a typical module path segment.
            if candidate and candidate[0].isupper():
                class_ctx = candidate
        raw_text = _get_ast_docstring(fn_name, class_name=class_ctx)
        if not raw_text:
            return result
        from griffe import Docstring as _GDoc
        doc_obj = _GDoc(raw_text)
    else:
        doc_obj = obj.docstring

    # Auto-detect Google vs NumPy from the raw text; pick the matching Griffe
    # parser instead of blindly using the one the caller passed.  Lucid's
    # sources are mostly NumPy-style but a sizeable minority (≈36 modules,
    # incl. ``nn.functional.linear``) use Google-style ``Args:`` headers.
    style = _detect_docstring_style(getattr(doc_obj, "value", "") or "")
    if style is not None:
        try:
            from griffe import Parser as _GParser
            parser = _GParser.google if style == "google" else _GParser.numpy
        except Exception:
            pass

    try:
        sections = doc_obj.parse(parser)
    except Exception:
        result["summary"] = _rst_to_text(doc_obj.value.split("\n")[0])
        return result

    from griffe import DocstringSectionKind as K

    for section in sections:
        kind = section.kind
        val  = section.value

        if kind == K.text:
            text = _rst_to_text(str(val))
            if result["summary"] is None:
                # Split first paragraph → summary, rest → extended
                paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
                result["summary"] = paragraphs[0] if paragraphs else text
                rest = "\n\n".join(paragraphs[1:])
                if rest:
                    result["extended"] = rest
            else:
                # Second K.text block: append to extended
                if result["extended"]:
                    result["extended"] += "\n\n" + text
                else:
                    result["extended"] = text

        elif kind == K.parameters:
            for item in val:
                result["parameters"].append({
                    "name":        item.name,
                    "annotation":  _annotation_str(item.annotation),
                    "description": _rst_to_text(item.description or ""),
                    "default":     _rst_strip_for_code(str(item.default)) if item.default is not None and str(item.default) != "required" else None,
                })

        elif kind == K.returns:
            items = list(val) if hasattr(val, "__iter__") and not isinstance(val, str) else [val]
            if items:
                first = items[0]
                result["returns"] = {
                    "annotation":  _annotation_str(getattr(first, "annotation", None)) or _annotation_str(obj.returns),
                    "description": _rst_to_text(getattr(first, "description", "") or ""),
                }

        elif kind == K.raises:
            for item in val:
                result["raises"].append({
                    "annotation":  _annotation_str(item.annotation),
                    "description": _rst_to_text(item.description or ""),
                })

        elif kind == K.examples:
            # Griffe NumPy parser yields a list of (DocstringSectionKind, str) tuples
            parts: list[str] = []
            items_iter = val if hasattr(val, "__iter__") and not isinstance(val, str) else [val]
            for ex in items_iter:
                if isinstance(ex, tuple) and len(ex) == 2:
                    # (DocstringSectionKind.examples|text, code_str) tuple
                    parts.append(str(ex[1]).strip())
                elif isinstance(ex, str):
                    parts.append(ex.strip())
                elif hasattr(ex, "value"):
                    parts.append(str(ex.value).strip())
                else:
                    parts.append(str(ex).strip())
            block = "\n".join(p for p in parts if p)
            if not block:
                block = str(val).strip()
            if block:
                # Examples render as a shiki code block, so strip reST roles to
                # the bare name (inline-code backticks would show literally).
                result["examples"].append(_rst_strip_for_code(block))

        elif kind in (K.admonition,):
            # Notes / warnings / See-Also blocks all arrive as
            # admonitions (reST style).  ``See Also`` is the structural
            # exception — its body is a list of ``name : description``
            # entries that we want rendered as hyperlinks rather than
            # plain prose, so we extract it into ``see_also`` and skip
            # the ``notes`` append for that case.
            sub_kind = getattr(val, "kind", None)
            sub_kind_str = str(sub_kind).lower() if sub_kind is not None else ""
            body = getattr(val, "description", None) or (str(val) if isinstance(val, str) else "")
            title = getattr(section, "title", "") or ""
            is_see_also = (
                "see-also" in sub_kind_str
                or "see_also" in sub_kind_str
                or title.strip().lower() == "see also"
            )
            if is_see_also and body:
                # Each line is ``name : description`` or just ``name``.
                # Comma-separated names on one line — split each.
                for line in body.split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    if ":" in line:
                        names_part, desc = line.split(":", 1)
                        desc = desc.strip()
                    else:
                        names_part, desc = line, ""
                    for raw_name in names_part.split(","):
                        n = raw_name.strip()
                        if n:
                            result["see_also"].append({
                                # Names are usually symbol refs, but docstrings
                                # sometimes use prose with reST code (``x``);
                                # strip to bare so it renders cleanly.
                                "name":        _rst_strip_for_code(n),
                                "description": _rst_to_text(desc),
                            })
                continue
            note_text = _rst_to_text(body)
            if note_text:
                result["notes"].append(note_text)

        elif hasattr(K, "notes") and kind == K.notes:
            # NumPy "Notes" section → dedicated kind in newer Griffe
            note_text = _rst_to_text(str(val))
            if note_text:
                result["notes"].append(note_text)

        elif kind == K.attributes:
            for item in val:
                result["attributes"].append({
                    # Usually an identifier, but some docstrings document naming
                    # conventions as prose attribute names with reST code.
                    "name":        _rst_strip_for_code(item.name),
                    "annotation":  _annotation_str(item.annotation),
                    "description": _rst_to_text(item.description or ""),
                })

        elif kind == K.warns:
            for item in val:
                result["warns"].append({
                    "annotation":  _annotation_str(item.annotation),
                    "description": _rst_to_text(item.description or ""),
                })

    return result


def _build_signature(obj: Any) -> str | None:
    """Build a human-readable function/class __init__ signature string."""
    try:
        params = obj.parameters
        parts: list[str] = []
        for p in params:
            if p.name in ("self", "cls"):
                continue
            ann = _annotation_str(p.annotation)
            default = str(p.default) if p.default is not None and str(p.default) != "required" else None
            chunk = p.name
            if ann:
                chunk = f"{chunk}: {ann}"
            if default is not None:
                chunk = f"{chunk} = {default}"
            parts.append(chunk)
        name = obj.name
        return f"{name}({', '.join(parts)})"
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Object serialisers
# ---------------------------------------------------------------------------

_KNOWN_LABELS = {"property", "staticmethod", "classmethod", "abstractmethod", "writable"}

def _extract_labels(fn: Any) -> list[str]:
    """Collect meaningful labels from griffe labels + decorators."""
    result: set[str] = set()
    # griffe .labels: set of str
    if hasattr(fn, "labels") and fn.labels:
        result |= set(fn.labels) & _KNOWN_LABELS
    # .decorators: list of Decorator objects whose .value is a string/expression
    if hasattr(fn, "decorators") and fn.decorators:
        for d in fn.decorators:
            val = str(d.value).strip().lstrip("@").split("(")[0]
            if val in _KNOWN_LABELS:
                result.add(val)
    return sorted(result)


_MODEL_SUMMARIES: dict[str, Any] | None = None


def _load_model_summaries() -> dict[str, Any]:
    """Lazy-load the cached layer-summary tree map (built once by
    ``tools/build_model_summaries.py``).  Cached in-process so the
    JSON read happens once per build run."""
    global _MODEL_SUMMARIES
    if _MODEL_SUMMARIES is not None:
        return _MODEL_SUMMARIES
    cache = OUT_DIR / "_summaries.json"
    if cache.is_file():
        try:
            _MODEL_SUMMARIES = json.loads(cache.read_text())
        except (OSError, json.JSONDecodeError):
            _MODEL_SUMMARIES = {}
    else:
        _MODEL_SUMMARIES = {}
    return _MODEL_SUMMARIES


def _extract_register_model_params(fn: Any) -> int | None:
    """Pull the ``params=<int>`` kwarg out of a ``@register_model(...)``
    decorator on a factory function.  Returns ``None`` when the
    decorator is absent, the kwarg is missing, or the value isn't a
    literal int / numeric expression Python can ``literal_eval``.

    Surfaced on the JSON as ``param_count`` so the docs site can render
    factory cards / detail pages with the family's paper-cited model
    size (e.g. ``"61.1M"`` for ``alexnet_cls``).
    """
    import ast as _ast

    if not getattr(fn, "decorators", None):
        return None
    for d in fn.decorators:
        try:
            tree = _ast.parse(str(d.value), mode="eval")
        except SyntaxError:
            continue
        call = tree.body
        if not isinstance(call, _ast.Call):
            continue
        func = call.func
        fname = (
            func.id if isinstance(func, _ast.Name)
            else (func.attr if isinstance(func, _ast.Attribute) else None)
        )
        if fname != "register_model":
            continue
        for kw in call.keywords:
            if kw.arg != "params":
                continue
            try:
                v = _ast.literal_eval(kw.value)
            except (ValueError, SyntaxError):
                continue
            if isinstance(v, int) and v > 0:
                return v
    return None


def _serialise_function(
    fn: Any, parser: Any, sha: str, *,
    module_root: Path | None = None,
    is_lucid_toplevel: bool = False,
) -> dict[str, Any]:
    doc = _parse_docstring(fn, parser)
    ret_ann = _annotation_str(fn.returns)
    if doc["returns"] is None and ret_ann:
        doc["returns"] = {"annotation": ret_ann, "description": ""}

    out = {
        "name":        fn.name,
        "path":        fn.path,
        "kind":        "function",
        "labels":      _extract_labels(fn),
        "signature":   _build_signature(fn),
        "source":      _source_link(fn, sha),
        "subcategory": _subcategory(fn, module_root=module_root, is_lucid_toplevel=is_lucid_toplevel),
        **doc,
    }
    param_count = _extract_register_model_params(fn)
    if param_count is not None:
        out["param_count"] = param_count
    # Layer-summary tree for the docs site's expandable "Model Size"
    # card — generated by ``tools/build_model_summaries.py`` and stored
    # in the ``_summaries.json`` cache.  Keyed by the factory name.
    summary = _load_model_summaries().get(fn.name)
    if summary is not None:
        out["model_summary"] = summary
    return out


def _detect_class_kind(cls: Any) -> str:
    """Return 'protocol', 'abstract', 'dataclass', or 'regular'.

    Resolution order is intentional: Protocol is checked first because a
    Protocol subclass *is also* technically abstract (no concrete bodies);
    dataclass next because ``@dataclass`` is a strong, explicit marker;
    abstract last as the catch-all for ABC-based classes.
    """
    # ``typing.Protocol`` base — checked first so Protocol subclasses
    # surface as their own category rather than as "abstract".
    if hasattr(cls, "bases") and cls.bases:
        for base in cls.bases:
            base_str = str(base)
            # Bare ``Protocol`` or qualified ``typing.Protocol``.  Also
            # accept ``runtime_checkable``-decorated bare Protocols whose
            # bases are other Protocols composed via multiple inheritance.
            if base_str == "Protocol" or base_str.endswith(".Protocol"):
                return "protocol"
    if hasattr(cls, "decorators") and cls.decorators:
        for d in cls.decorators:
            dec_str = str(d.value)
            # ``@runtime_checkable`` is the Protocol idiom — when present
            # alongside a Protocol-derived class it confirms the category.
            if "runtime_checkable" in dec_str:
                return "protocol"
    # @dataclass decorator check
    if hasattr(cls, "decorators") and cls.decorators:
        for d in cls.decorators:
            if "dataclass" in str(d.value):
                return "dataclass"
    # Any *direct* (non-alias) method has abstractmethod label
    from griffe import Function as _GF, Attribute as _GA
    for _name, member in cls.members.items():
        try:
            if isinstance(member, (_GF, _GA)) and hasattr(member, "labels"):
                if "abstractmethod" in member.labels:
                    return "abstract"
        except Exception:
            pass
    # Explicit ABC / ABCMeta base class
    if hasattr(cls, "bases") and cls.bases:
        for base in cls.bases:
            if "ABC" in str(base):
                return "abstract"
    return "regular"


def _extract_bases(cls: Any) -> list[str]:
    """Return simplified base class names, excluding 'object'."""
    if not hasattr(cls, "bases") or not cls.bases:
        return []
    result: list[str] = []
    for base in cls.bases:
        base_str = str(base).strip()
        if base_str in ("object", ""):
            continue
        # Take the last dotted component for display brevity
        result.append(base_str.split(".")[-1])
    return result


def _serialise_property(attr: Any, sha: str) -> dict[str, Any]:
    """Serialise a Griffe Attribute that has the 'property' label as a pseudo-function."""
    from griffe import Parser
    labels = sorted(set(attr.labels) & _KNOWN_LABELS)
    ann = _annotation_str(attr.annotation)
    doc = _parse_docstring(attr, Parser.numpy) if hasattr(attr, "docstring") else {
        "summary": None, "extended": None, "parameters": [], "returns": None,
        "raises": [], "examples": [], "notes": [], "attributes": [], "warns": [],
    }
    if doc["returns"] is None and ann:
        doc["returns"] = {"annotation": ann, "description": ""}
    return {
        "name":      attr.name,
        "path":      attr.path,
        "kind":      "function",
        "labels":    labels,
        "signature": f"{attr.name}: {ann}" if ann else attr.name,
        "source":    _source_link(attr, sha),
        **doc,
    }


def _merge_inherited_parameters(cls: Any, doc: dict[str, Any], parser: Any) -> None:
    """Fill in parameters the class accepts but does not itself describe.

    A subclass documents what it adds and leaves the rest to its base —
    which is correct in source and invisible on the site, because only
    the class's own docstring was ever parsed.  ``PlaNetConfig`` takes
    nineteen fields and showed ten; a reader could not learn that
    ``cnn_depth`` exists, let alone what it does.  The base classes are
    no help either: ``GenerativeModelConfig`` and friends are exported
    from ``lucid.models`` but never emitted, so the descriptions were
    on no page at all.

    The rule is deliberately narrow: only a parameter that appears in
    the class's *real* signature is filled in, and only from a base
    that documents it.  Nothing is invented — an inherited field the
    subclass does not accept, or one nobody documents, stays absent.
    ``inherited_from`` records where the text came from so the page can
    say so.
    """
    try:
        accepted = [p.name for p in cls.parameters if p.name not in ("self", "cls")]
    except Exception:
        return
    if not accepted:
        return

    described = {p["name"] for p in doc["parameters"]}
    missing = [n for n in accepted if n not in described]
    if not missing:
        return

    def _bases(obj: Any) -> list[Any]:
        # ``resolved_bases`` resolves lazily and raises on a base Griffe
        # cannot follow (a C extension, a conditional import).  A class
        # whose ancestry is unreadable simply gets nothing merged.
        try:
            return list(getattr(obj, "resolved_bases", []) or [])
        except Exception:
            return []

    seen: set[str] = set()
    queue = _bases(cls)
    while queue and missing:
        base = queue.pop(0)
        path = getattr(base, "path", None)
        if not path or path in seen:
            continue
        seen.add(path)
        queue.extend(_bases(base))
        try:
            base_doc = _parse_docstring(base, parser)
        except Exception:
            continue
        for item in base_doc["parameters"]:
            if item["name"] in missing:
                doc["parameters"].append(
                    {**item, "inherited_from": getattr(base, "name", None)}
                )
                missing.remove(item["name"])

    # Signature order, so the table reads like the constructor call.
    order = {name: i for i, name in enumerate(accepted)}
    doc["parameters"].sort(key=lambda p: order.get(p["name"], len(order)))


def _serialise_class(
    cls: Any, parser: Any, sha: str, *,
    module_root: Path | None = None,
    is_lucid_toplevel: bool = False,
) -> dict[str, Any]:
    doc = _parse_docstring(cls, parser)
    _merge_inherited_parameters(cls, doc, parser)

    # Collect public methods + properties
    # Keep selected dunders that are part of the public interface
    KEEP_DUNDERS = {"__init__", "__call__", "__len__", "__iter__",
                    "__getitem__", "__setitem__", "__repr__", "__str__",
                    "__add__", "__radd__", "__iadd__", "__sub__", "__rsub__",
                    "__mul__", "__rmul__", "__matmul__", "__truediv__",
                    "__floordiv__", "__mod__", "__pow__", "__neg__",
                    "__pos__", "__abs__", "__bool__", "__float__", "__int__",
                    "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__",
                    "__enter__", "__exit__"}

    methods: list[dict[str, Any]] = []
    for name, member in cls.members.items():
        from griffe import Function as GriffeFunction, Attribute as GriffeAttr

        # Skip private names (but keep selected dunders)
        if name.startswith("_") and name not in KEEP_DUNDERS:
            continue

        if isinstance(member, GriffeFunction):
            methods.append(_serialise_function(member, parser, sha))
        elif isinstance(member, GriffeAttr):
            # Include Attribute members that are @property (have 'property' label)
            attr_labels = set(getattr(member, "labels", set()))
            if "property" in attr_labels:
                methods.append(_serialise_property(member, sha))
            # Skip plain class/instance attributes (not properties)

    return {
        "name":        cls.name,
        "path":        cls.path,
        "kind":        "class",
        "class_kind":  _detect_class_kind(cls),
        "bases":       _extract_bases(cls),
        "labels":      [],   # classes don't have labels; field kept for schema consistency
        "signature":   _build_signature(cls),
        "source":      _source_link(cls, sha),
        "subcategory": _subcategory(cls, module_root=module_root, is_lucid_toplevel=is_lucid_toplevel),
        "methods":     methods,
        **doc,
    }


def _resolve_member(member: Any) -> Any:
    """Resolve an Alias to its final target, return None on failure."""
    from griffe import Alias as GriffeAlias
    if isinstance(member, GriffeAlias):
        try:
            return member.final_target
        except Exception:
            return None
    return member


def _resolve_attribute_alias(attr: Any) -> Any:
    """Resolve an :class:`Attribute` whose value is a bare identifier
    pointing at another callable in the SAME module (or its parents)
    to that callable.

    Returns ``None`` when the value isn't a simple identifier or doesn't
    resolve to a Function/Class.  Built primarily to surface non-inplace
    aliases like ``kaiming_uniform = kaiming_uniform_`` — both names
    are part of the public API, but only the underscore-suffixed source
    is a real ``def``, the bare-name alias is an :class:`Attribute`.

    Without this hop, the assignment would silently disappear from the
    docs JSON (Attribute isn't one of the kinds the member loop emits)
    and we'd get a build-time warning from the ``__all__`` blind-spot
    guard instead of a real entry."""
    from griffe import Attribute as GriffeAttr, Function as GriffeFn, Class as GriffeCls
    if not isinstance(attr, GriffeAttr):
        return None
    value = attr.value
    if value is None:
        return None
    val_str = str(value).strip()
    if not val_str.isidentifier():
        return None
    parent = attr.parent
    while parent is not None:
        candidate = getattr(parent, "members", {}).get(val_str)
        candidate = _resolve_member(candidate) if candidate is not None else None
        if isinstance(candidate, (GriffeFn, GriffeCls)):
            return candidate
        parent = getattr(parent, "parent", None)
    return None


def _parse_dunder_all(mod: Any) -> set[str] | None:
    """Return the set of names in __all__ if it exists, otherwise None."""
    import ast
    from griffe import Attribute as GriffeAttr
    all_attr = mod.members.get("__all__")
    if not isinstance(all_attr, GriffeAttr):
        return None
    try:
        names = ast.literal_eval(str(all_attr.value))
        if isinstance(names, (list, tuple)):
            return set(names)
    except Exception:
        pass
    return None


def _rst_math_to_markdown(text: str) -> str:
    """Convert a family ``theory`` body (reST) to markdown / remark-math.

    Same pipeline as ``_rst_to_text`` — block directives (``.. math::`` /
    ``.. code-block::`` / admonitions), doctest fencing, inline math, then the
    shared inline-markup pass — so theory pages render identically to docstring
    prose with no reST leaking through.
    """
    text = _convert_directive_blocks(text)
    text = _fence_doctests(text)
    text = _apply_outside_protected(
        text, lambda seg: _rst_inline_markup(_inline_math_sub(seg))
    )
    return text


# Task-wrapper class suffix → HF-style task identifier.  Each family-leaf's
# supported tasks are inferred at build time from the names of its public
# ``*For*`` classes — no per-Config bookkeeping needed.  Identifiers use
# kebab-case to match Hugging Face Hub conventions so the same tags travel
# unchanged if/when the docs cross-link to external repos.


def _extract_family_tasks(slug: str) -> list[str]:
    """The tasks a family registers, read from its ``@register_model``.

    The site used to re-derive this from class-name suffixes through a
    table of its own, which made it a fourth task taxonomy beside the
    class names, the registry tags and the ``AutoModelFor*`` shells —
    and the four disagreed.  ``lucid.models._tasks`` is now the single
    list, so the pills come from what each factory actually registers.

    Parsed with ``ast`` rather than imported, like
    :func:`_extract_family_meta`, because this script must not depend on
    importing the Lucid runtime.  ``"base"`` is skipped: a backbone with
    no head is not a task the reader is looking for.  Order follows the
    source so the JSON stays stable.
    """
    import ast as _ast

    family_dir = _slug_to_family_dir(slug)
    if not family_dir.is_dir():
        return []

    seen: list[str] = []
    for path in sorted(family_dir.glob("*.py")):
        try:
            tree = _ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for node in _ast.walk(tree):
            if not isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                continue
            for dec in node.decorator_list:
                if not isinstance(dec, _ast.Call):
                    continue
                fn = dec.func
                name = (
                    fn.id if isinstance(fn, _ast.Name)
                    else (fn.attr if isinstance(fn, _ast.Attribute) else None)
                )
                if name != "register_model":
                    continue
                for kw in dec.keywords:
                    if kw.arg != "task" or not isinstance(kw.value, _ast.Constant):
                        continue
                    task = kw.value.value
                    if isinstance(task, str) and task != "base" and task not in seen:
                        seen.append(task)
    return seen


def _slug_to_family_dir(slug: str) -> Path:
    """Map a family-leaf slug to its source directory.

    e.g. ``lucid.models.vision.resnet`` → ``<repo>/lucid/models/vision/resnet/``
    """
    parts = slug.split(".")
    return REPO_ROOT.joinpath(*parts)


def _extract_family_meta(slug: str) -> dict[str, str]:
    """Parse a family-leaf source to extract ``@model_family_meta``
    decorator arguments.

    Convention (see ``arch-models-canonical-name``): each family's Config
    dataclass is wrapped with ``@model_family_meta(canonical_name=...,
    citation=..., theory=...)`` whose keyword arguments are all string
    literals (or implicit concatenations of literals).  Most families
    own a single ``_config.py``; a few (e.g. ``yolo`` with ``_v1.py``,
    ``_v2.py``, ...) split the configs across several files.  We scan
    every ``.py`` file in the family directory and return the first
    decorator we successfully parse.

    Returns a dict with any of the keys ``canonical_name`` / ``citation``
    / ``theory`` that were found and successfully decoded.  Empty dict
    when no decorator is present in any source file.
    """
    import ast as _ast
    import textwrap

    family_dir = _slug_to_family_dir(slug)
    if not family_dir.is_dir():
        return {}

    wanted = {"canonical_name", "citation", "theory"}

    # Scan order: prefer ``_config.py`` first (the conventional location),
    # then any other ``.py`` files at the family-dir root (handles YOLO's
    # per-version configs and similar splits).  Skip ``__init__.py`` —
    # re-exports won't carry the decorator literally.
    candidates: list[Path] = []
    primary = family_dir / "_config.py"
    if primary.is_file():
        candidates.append(primary)
    for p in sorted(family_dir.iterdir()):
        if p.is_file() and p.suffix == ".py" and p.name not in {"__init__.py", "_config.py"}:
            candidates.append(p)

    for src_file in candidates:
        try:
            tree = _ast.parse(src_file.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue

        out: dict[str, str] = {}
        for node in _ast.walk(tree):
            if not isinstance(node, _ast.ClassDef):
                continue
            for dec in node.decorator_list:
                if not isinstance(dec, _ast.Call):
                    continue
                func = dec.func
                if isinstance(func, _ast.Name):
                    name = func.id
                elif isinstance(func, _ast.Attribute):
                    name = func.attr
                else:
                    continue
                if name != "model_family_meta":
                    continue
                for kw in dec.keywords:
                    if kw.arg not in wanted:
                        continue
                    try:
                        val = _ast.literal_eval(kw.value)
                    except (ValueError, SyntaxError):
                        continue
                    if not isinstance(val, str):
                        continue
                    # ``theory`` is typically a raw-triple-quoted block
                    # with leading indentation; dedent so the rST
                    # renderer sees clean column-0 directives, then
                    # translate rST math syntax to markdown so the
                    # MathText (remark-math) component can render it.
                    if kw.arg == "theory":
                        val = textwrap.dedent(val).strip("\n")
                        val = _rst_math_to_markdown(val)
                    else:
                        # citation: convert reST hyperlinks / roles to markdown.
                        val = _rst_inline_markup(val.strip())
                    if val:
                        out[kw.arg] = val
                if out:
                    return out
    return {}


def _inject_lazy_loaded_members(
    loader: Any, parser: Any, sha: str, lucid_data: dict[str, Any],
) -> None:
    """Rescue composite / ops names exposed only through lucid's lazy
    ``__getattr__`` loader so they show up in the synth-bucket JSONs.

    Blind spot pattern (category 5):
      * ``lucid/__init__.py`` registers e.g. ``COMPOSITE_NAMES`` with a
        lazy loader — ``lucid.erfc`` works at runtime, but Griffe's
        static AST walk sees nothing under ``lucid.erfc`` because the
        name is never explicitly imported or re-exported from the
        top-level ``__init__.py``.
      * The synth pipeline below partitions ``lucid_data["members"]``
        by ``subcategory`` to emit ``lucid.ops.composite.json`` (et al.).
        With Griffe's static view, the composite bucket misses ~30 ops
        (erfc / lgamma / i0 / allclose / amax / corrcoef / std_mean /
        argwhere / index_copy / nanquantile / …).

    Fix: for every ``name → subcategory`` mapping in
    :data:`_LUCID_NAME_OVERRIDES` that isn't present in
    ``lucid_data["members"]``, locate the symbol in its actual home
    module (composite submodule, ``_ops/_registry`` generated function,
    etc.) and append a serialised entry with the correct subcategory tag.

    Mutates ``lucid_data["members"]`` in place; safe to call exactly
    once after :func:`_serialise_module` produces the lucid top-level
    payload.
    """
    from griffe import (
        Function as GriffeFunction,
        Class as GriffeClass,
        Module as GriffeModule,
    )

    existing = {m.get("name") for m in lucid_data.get("members", [])}
    missing = [n for n in _LUCID_NAME_OVERRIDES if n not in existing]
    if not missing:
        return

    # Home-module lookup table: subcategory → list of (griffe_path,
    # already_loaded_module).  Loaded once per call.  Names tagged
    # "composite" live in lucid._ops.composite.<submodule>; "ops"
    # names come from the runtime _registry (we resolve those via the
    # generated ``lucid._ops`` module).
    home_candidates: dict[str, list[str]] = {
        "composite": [
            "lucid._ops.composite.elementwise",
            "lucid._ops.composite.reductions",
            "lucid._ops.composite.shape",
            "lucid._ops.composite.blas",
            "lucid._ops.composite.predicates",
            "lucid._ops.composite.dtype",
            "lucid._ops.composite.constants",
            "lucid._ops.composite.statistics",
            "lucid._ops.composite.indexing",
            "lucid._ops.composite.complex",
        ],
        "ops":      ["lucid._ops"],
        "factories": ["lucid._factories"],
        "autograd":  ["lucid.autograd"],
    }
    home_cache: dict[str, Any] = {}

    def _load_home(path: str) -> Any | None:
        if path in home_cache:
            return home_cache[path]
        try:
            m = loader.load(path)
            home_cache[path] = m
            return m
        except Exception:
            home_cache[path] = None
            return None

    appended = 0
    skipped: list[str] = []
    for name in missing:
        subcat = _LUCID_NAME_OVERRIDES[name]
        candidates = home_candidates.get(subcat, [])
        target = None
        target_home: Any = None
        for home_path in candidates:
            home_mod = _load_home(home_path)
            if home_mod is None:
                continue
            raw = home_mod.members.get(name)
            resolved = _resolve_member(raw) if raw is not None else None
            if resolved is None:
                continue
            target = resolved
            target_home = home_mod
            break
        if target is None:
            skipped.append(name)
            continue

        try:
            if isinstance(target, GriffeFunction):
                serialised = _serialise_function(
                    target, parser, sha,
                    module_root=None, is_lucid_toplevel=True,
                )
            elif isinstance(target, GriffeClass):
                serialised = _serialise_class(
                    target, parser, sha,
                    module_root=None, is_lucid_toplevel=True,
                )
            else:
                skipped.append(name)
                continue
        except Exception as exc:
            if VERBOSE:
                print(f"  [warn] lazy-rescue failed for {name}: {exc}")
            skipped.append(name)
            continue

        # Relabel under the lucid.* namespace so the subcategory rule
        # (``is_lucid_toplevel=True`` → name-override hit) attaches the
        # right bucket tag.
        serialised["name"] = name
        serialised["path"] = f"lucid.{name}"
        serialised["subcategory"] = subcat
        lucid_data.setdefault("members", []).append(serialised)
        appended += 1

    if VERBOSE and appended:
        print(f"  [lazy-rescue] injected {appended} member(s) "
              f"(skipped {len(skipped)}: {skipped[:5]}{'…' if len(skipped) > 5 else ''})")


def _serialise_module(mod: Any, parser: Any, sha: str, *, slug: str) -> dict[str, Any]:
    """Top-level serialiser for a module."""
    from griffe import Function as GriffeFunction, Class as GriffeClass, Module as GriffeModule

    doc = _parse_docstring(mod, parser)
    members: list[dict[str, Any]] = []
    seen_paths: set[str] = set()

    # Compute the module's source directory so members get subcategories
    # relative to it (e.g. ``lucid.nn`` members from ``lucid/nn/modules/conv.py``
    # → subcategory ``"modules/conv"``).
    module_root: Path | None = None
    try:
        if mod.filepath is not None:
            module_root = Path(mod.filepath).parent
    except Exception:
        module_root = None

    # Honour __all__ when present — this is the authoritative public API surface.
    # Without this, Griffe picks up every imported name (e.g. Tensor imported into
    # lucid.nn.init for type-annotation use) and leaks cross-package symbols.
    allowed_names = _parse_dunder_all(mod)

    for name, raw_member in mod.members.items():
        if name.startswith("_"):
            continue
        # __all__ filter: skip names not explicitly exported
        if allowed_names is not None and name not in allowed_names:
            continue

        member = _resolve_member(raw_member)
        if member is None:
            continue
        # Deduplicate — same class re-exported from multiple aliases
        if member.path in seen_paths:
            continue
        seen_paths.add(member.path)

        is_lucid_top = slug == "lucid"
        # Attribute-alias rescue: ``foo = foo_`` style assignments are
        # public API but live in Griffe as ``Attribute`` values, which
        # the kind branches below don't emit.  Resolve here and treat as
        # the underlying callable (re-labelled with the alias's name).
        from griffe import Attribute as GriffeAttribute  # local import to keep top imports tight
        attr_alias_target = None
        if isinstance(member, GriffeAttribute):
            attr_alias_target = _resolve_attribute_alias(member)
        try:
            if isinstance(member, GriffeFunction):
                members.append(_serialise_function(
                    member, parser, sha,
                    module_root=module_root,
                    is_lucid_toplevel=is_lucid_top,
                ))
            elif isinstance(member, GriffeClass):
                members.append(_serialise_class(
                    member, parser, sha,
                    module_root=module_root,
                    is_lucid_toplevel=is_lucid_top,
                ))
            elif attr_alias_target is not None:
                # Re-serialise the alias target under the alias's name
                # so consumers see the docstring + signature while the
                # URL anchor matches ``__all__``.
                if isinstance(attr_alias_target, GriffeFunction):
                    serialised = _serialise_function(
                        attr_alias_target, parser, sha,
                        module_root=module_root,
                        is_lucid_toplevel=is_lucid_top,
                    )
                else:
                    serialised = _serialise_class(
                        attr_alias_target, parser, sha,
                        module_root=module_root,
                        is_lucid_toplevel=is_lucid_top,
                    )
                serialised["name"] = name
                serialised["path"] = f"{mod.path}.{name}"
                members.append(serialised)
            elif isinstance(member, GriffeModule):
                # Name-collision rescue: a submodule ``linear`` (file
                # ``linear.py``) shadows the parent's ``from .linear
                # import linear`` alias in Griffe's namespace.  When
                # the submodule itself holds a same-named public
                # function or class, surface THAT — that's what the
                # ``__all__`` re-export actually means at the public
                # API level.
                nested_raw = member.members.get(name)
                nested = _resolve_member(nested_raw) if nested_raw is not None else None
                if nested is None or nested.path in seen_paths:
                    continue
                if isinstance(nested, GriffeFunction):
                    seen_paths.add(nested.path)
                    members.append(_serialise_function(
                        nested, parser, sha,
                        module_root=module_root,
                        is_lucid_toplevel=is_lucid_top,
                    ))
                elif isinstance(nested, GriffeClass):
                    seen_paths.add(nested.path)
                    members.append(_serialise_class(
                        nested, parser, sha,
                        module_root=module_root,
                        is_lucid_toplevel=is_lucid_top,
                    ))
        except Exception as exc:
            if VERBOSE:
                print(f"  [warn] skipping {name}: {exc}")

    # Blind-spot guard: any name in ``__all__`` that didn't make it
    # into ``members`` after collection is a regression risk — print a
    # build-time warning so the next sweep catches it before it ships.
    # Excludes submodule names (caller documents those via their own
    # slug) and class members (already deduped via ``seen_paths``).
    if allowed_names is not None:
        emitted = {m.get("name") for m in members}
        # Module-level constants (``RK4 = ButcherTableau(...)``) are Griffe
        # Attributes whose value is a call, so no member branch above emits
        # them.  Documenting one in the module docstring's ``Attributes``
        # section is the supported way to surface it — the module page
        # renders that table — so those names are covered, not missing.
        documented_attrs = {a.get("name") for a in doc.get("attributes", [])}
        missing = []
        for n in allowed_names:
            if n in emitted or n in documented_attrs:
                continue
            sub = mod.members.get(n)
            sub = _resolve_member(sub) if sub is not None else None
            if sub is None:
                continue
            from griffe import Module as _GModuleCheck
            if isinstance(sub, _GModuleCheck):
                # A bare submodule re-export (e.g. ``data`` in
                # ``lucid.utils.__all__``) is fine — that submodule gets
                # its own page.
                continue
            missing.append(n)
        if missing:
            print(f"  [warn] {slug}: __all__ exposes {missing} but they are missing "
                  f"from the emitted members list — possible Griffe shadowing.")

    return {
        "slug":    slug,
        "name":    mod.name,
        "path":    mod.path,
        "kind":    "module",
        "source":  _source_link(mod, sha),
        "members": members,
        **doc,
    }


# ---------------------------------------------------------------------------
# Special handler: lucid.tensor (Tensor class only)
# ---------------------------------------------------------------------------

def _build_tensor_data(loader: Any, parser: Any, sha: str) -> dict[str, Any]:
    """Extract only the Tensor class from lucid._tensor.tensor.

    Filters the methods list so that operators / factories / composites that
    are already documented as top-level ``lucid.*`` functions are NOT
    duplicated here — the Tensor docs surface only what is genuinely
    Tensor-specific: properties, dunders, in-place mutations, ``new_*``
    constructors, autograd lifecycle (``backward`` / ``retain_grad`` /
    ``register_hook`` / ``requires_grad_``), and bridge methods
    (``item`` / ``numpy`` / ``tolist`` / ``__dlpack__`` / ...).
    """
    from griffe import Class as GriffeClass

    mod = loader.load("lucid._tensor.tensor")
    tensor_cls = mod.members.get("Tensor")
    if tensor_cls is None or not isinstance(tensor_cls, GriffeClass):
        return {"slug": "lucid.tensor", "kind": "module", "name": "Tensor",
                "path": "lucid.Tensor", "members": [], "summary": None}

    serialised = _serialise_class(tensor_cls, parser, sha)

    # Names that have a top-level `lucid.<name>` function — listing them as
    # Tensor methods too is redundant.  Pulled from the same name sets that
    # drive the sidebar subcategory assignment, so the two views stay in sync.
    redundant_names: set[str] = {
        name for name, slug in _LUCID_NAME_OVERRIDES.items()
        if slug in ("ops", "factories", "composite", "predicates")
    }

    def _keep_method(m: dict[str, Any]) -> bool:
        name: str = m["name"]
        labels = m.get("labels") or []
        if "property" in labels:
            return True                       # properties are Tensor-only state
        if name.startswith("__"):
            return True                       # dunders (constructor, repr, ops, ...)
        if name.startswith("new_"):
            return True                       # new_zeros / new_full / ...
        # In-place mutation: filter if its non-mutating counterpart is a
        # top-level op (e.g. add_ → add).  Otherwise keep (zero_, fill_,
        # share_memory_, etc. have no free-function counterpart).
        if name.endswith("_") and not name.startswith("__"):
            base = name[:-1]
            if name in redundant_names or base in redundant_names:
                return False
            return True
        if name in redundant_names:
            return False                      # duplicate of lucid.<name>
        return True

    serialised["methods"] = [m for m in serialised["methods"] if _keep_method(m)]

    return {
        "slug":    "lucid.tensor",
        "kind":    "class-module",   # special kind for the site
        "name":    "Tensor",
        "path":    "lucid.Tensor",
        "source":  _source_link(tensor_cls, sha),
        **{k: v for k, v in serialised.items() if k not in ("name", "path", "kind", "source")},
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

VERBOSE = False


def _emit(msg: str) -> None:
    """Print a status line without breaking an active tqdm bar."""
    if tqdm is not None:
        tqdm.write(msg)
    else:
        print(msg)


_LUCID_MODELS_TIER1_SUBS = {
    "auto", "base", "hub", "mixins", "output", "protocols", "registry",
}
"""Subcategories kept on the ``lucid.models`` landing page.  Everything
with a slash (e.g. ``vision/resnet/pretrained``) is a family member and
belongs on its own family page — keeping them here would make the index
page ~500 members long and obscure the user-facing dispatch surface."""

def _is_domain_infrastructure(subcategory: str | None) -> bool:
    """Is this a domain's shared base layer rather than one family's code?

    The filter above drops anything with a slash on the grounds that it
    belongs to a family page.  That is right for ``generative/nice/config``
    and wrong for ``generative/config`` — the second is the base layer the
    families inherit from, and dropping it meant ``GenerativeModelConfig``,
    ``DiffusionModelConfig``, ``NormalizingFlowConfig`` and
    ``LanguageModelConfig`` were exported from ``lucid.models`` and emitted
    to no page at all.  The descriptions of every inherited field lived
    there, so they reached no page either.

    Depth alone used to separate them — a family member carried two
    slashes and a domain module one.  That stopped being true when the
    shared modules moved under ``<domain>/_common/`` to keep the domain's
    directory listing to families only: ``generative/_common/config`` now
    has the same depth as ``generative/nice/config``.  So the second
    segment is what decides, and the leading underscore is the marker no
    family name can collide with (a family directory never starts with
    one).
    """
    if not subcategory:
        return False
    parts = subcategory.split("/")
    if len(parts) not in (2, 3):
        return False
    domains = {g["slug"].rsplit(".", 1)[-1] for g in _LUCID_MODELS_FAMILY_GROUPS}
    if parts[0] not in domains:
        return False
    return len(parts) == 2 or parts[1].startswith("_")


_LUCID_MODELS_FAMILY_GROUPS = [
    {"slug": "lucid.models.vision",     "label": "Vision",     "icon": "image"},
    {"slug": "lucid.models.text",       "label": "Text",       "icon": "text"},
    {"slug": "lucid.models.generative", "label": "Generative", "icon": "sparkles"},
    {"slug": "lucid.models.multimodal", "label": "Multimodal", "icon": "layers"},
]


def _build_family_groups(family_root_slug: str) -> list[dict[str, Any]]:
    """Walk the filesystem under a family-root and emit one entry per
    direct sub-package (model family).  Each entry is a card on the
    family-root's index page linking to that family's detail page."""
    parts = family_root_slug.split(".")  # e.g. ["lucid", "models", "vision"]
    family_dir = LUCID_SRC / Path(*parts)
    if not family_dir.is_dir():
        return []

    groups: list[dict[str, Any]] = []
    for sub in sorted(family_dir.iterdir()):
        if not sub.is_dir() or sub.name.startswith("_"):
            continue
        if not (sub / "__init__.py").exists():
            continue
        groups.append({
            "slug": f"{family_root_slug}.{sub.name}",
            "label": sub.name,
        })
    return groups


# ── ProcessPool worker plumbing ────────────────────────────────────────────
# These two functions sit at module scope (not nested) so the worker
# processes can re-import this script and call them by name — that's
# how ``ProcessPoolExecutor`` smuggles work across the fork boundary.
# The Griffe loader is created lazily in each worker the first time
# it's needed; subsequent calls reuse the cached object so the
# expensive walk-of-lucid happens at most once per worker.

_WORKER_LOADER: Any = None


def _worker_init() -> None:
    """ProcessPool initialiser.  Suppresses Griffe's per-mismatch
    warnings (the parent process already printed them; duplicating
    across N workers triples the noise on real ones)."""
    import logging as _logging
    _logging.getLogger("griffe").setLevel(_logging.ERROR)
    _logging.getLogger("mkdocstrings").setLevel(_logging.ERROR)


def _worker_loader() -> Any:
    """Lazy per-worker Griffe loader.  Cached on a module-global so
    every task on the same worker process shares one loader instance
    — the AST walk of ``lucid/`` is the dominant cost and we want it
    paid once per worker, not once per task."""
    global _WORKER_LOADER
    if _WORKER_LOADER is None:
        _install_griffe_overload_guard()
        from griffe import GriffeLoader, Parser
        _WORKER_LOADER = GriffeLoader(
            docstring_parser=Parser.numpy,
            search_paths=[str(LUCID_SRC)],
            allow_inspection=False,
        )
    return _WORKER_LOADER


def _worker_build_one(slug: str, griffe_path: str, sha: str) -> str:
    """Entry point ``ProcessPoolExecutor`` calls per task.  Wraps
    :func:`build_one` with worker-local loader / parser plumbing and
    returns the slug so the parent's ``as_completed`` loop can stamp
    a progress line."""
    from griffe import Parser
    build_one(slug, griffe_path, _worker_loader(), Parser.numpy, sha)
    return slug


def build_one(slug: str, griffe_path: str, loader: Any, parser: Any, sha: str) -> None:
    from griffe import Module as GriffeModule, Class as GriffeClass

    out_file = OUT_DIR / f"{slug}.json"

    try:
        if slug == "lucid.tensor":
            data = _build_tensor_data(loader, parser, sha)
        else:
            obj = loader.load(griffe_path)
            if isinstance(obj, GriffeModule):
                data = _serialise_module(obj, parser, sha, slug=slug)
                # Family-leaf metadata: each ``lucid.models.<domain>.<family>``
                # Config class is wrapped with ``@model_family_meta(...)``
                # carrying three docs-facing fields (``canonical_name`` /
                # ``citation`` / ``theory``).  Parsed directly from the
                # source file via ``ast`` — does not depend on Griffe.
                # See ``arch-models-canonical-name``.
                if _is_family_leaf(slug):
                    meta = _extract_family_meta(slug)
                    for key in ("canonical_name", "citation", "theory"):
                        if meta.get(key):
                            data[key] = meta[key]
                    # HF-style task tags inferred from the family's
                    # ``*For*`` task-wrapper class names.  Surfaced on the
                    # family-leaf JSON's top level so family-root cards
                    # can render them next to the canonical name.
                    tasks = _extract_family_tasks(slug)
                    if tasks:
                        data["tasks"] = tasks
            elif isinstance(obj, GriffeClass):
                serialised = _serialise_class(obj, parser, sha)
                data = {"slug": slug, **serialised}
            else:
                data = {"slug": slug, "kind": "unknown", "name": griffe_path}

        if slug == "lucid.models":
            members = data.get("members", []) or []
            filtered = [
                m for m in members
                if (m.get("subcategory") or "") in _LUCID_MODELS_TIER1_SUBS
                or _is_domain_infrastructure(m.get("subcategory"))
            ]
            data["members"] = filtered
            data["family_groups"] = _LUCID_MODELS_FAMILY_GROUPS

        if slug in _FAMILY_ROOTS:
            data["members"] = []  # __init__.py is intentionally empty
            data["family_groups"] = _build_family_groups(slug)

        out_file.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        member_count = len(data.get("members", data.get("methods", [])))
        _emit(f"  ✓ {slug:32}  ({member_count} members)")

    except Exception as exc:
        _emit(f"  ✗ {slug:32}  {exc}")
        if VERBOSE:
            import traceback
            traceback.print_exc()


def build_synth(slug: str, filter_subcat: str, lucid_data: dict[str, Any]) -> None:
    """Build a synthesized slug by filtering the lucid top-level by subcategory.

    Special-case for ``lucid.ops``: re-bucket members by arity (unary/binary/
    ternary) so the page isn't a flat 143-item list.  All other synthesised
    slugs drop subcategory entirely so their members render flat.
    """
    out_file = OUT_DIR / f"{slug}.json"

    filtered = [
        m for m in lucid_data.get("members", [])
        if m.get("subcategory") == filter_subcat
    ]

    def _arity_bucket(name: str) -> str:
        """Map an op name to its arity bucket.  In-place variants share the
        bucket of their base op (so ``add_`` → binary alongside ``add``).
        Arity 0 (variadic; e.g. ``cat``, ``where``, ``meshgrid``) and
        unknown arity fall into ``variadic`` so nothing is left orphaned.
        """
        a = _OP_ARITY.get(name)
        if a is None and name.endswith("_"):
            a = _OP_ARITY.get(name[:-1])
        if a == 1: return "unary"
        if a == 2: return "binary"
        if a == 3: return "ternary"
        return "variadic"

    if slug == "lucid.ops":
        for m in filtered:
            m["subcategory"] = _arity_bucket(m["name"])
    else:
        for m in filtered:
            m["subcategory"] = None

    summary, extended = _SYNTH_SUMMARIES.get(slug, (None, None))

    data = {
        "slug":    slug,
        "name":    slug.split(".")[-1],
        "path":    slug,
        "kind":    "module",
        "source":  None,
        "summary": summary,
        "extended": extended,
        "parameters": [],
        "returns": None,
        "raises": [],
        "examples": [],
        "notes": [],
        "attributes": [],
        "warns": [],
        "members": filtered,
    }

    out_file.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    _emit(f"  ✓ {slug:32}  ({len(filtered)} members)")


class _ElapsedSpinner:
    """Print '[mm:ss] msg' on a timer until ``stop()``.  Used during the
    upfront ``loader.load('lucid')`` walk which has no granular progress."""

    def __init__(self, msg: str, interval: float = 2.0) -> None:
        self._msg = msg
        self._interval = interval
        self._t0 = time.monotonic()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _fmt(self) -> str:
        dt = int(time.monotonic() - self._t0)
        return f"\r[{dt//60:02d}:{dt%60:02d}] {self._msg}"

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            sys.stdout.write(self._fmt())
            sys.stdout.flush()

    def __enter__(self) -> "_ElapsedSpinner":
        sys.stdout.write(self._fmt()); sys.stdout.flush()
        self._thread.start()
        return self

    def __exit__(self, *a: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=0.1)
        dt = int(time.monotonic() - self._t0)
        sys.stdout.write(f"\r[{dt//60:02d}:{dt%60:02d}] {self._msg} ✓\n")
        sys.stdout.flush()


def main() -> None:
    global VERBOSE

    parser_args = argparse.ArgumentParser(description="Build Lucid API JSON data")
    parser_args.add_argument("--module", help="Build only this slug (e.g. lucid.fft)")
    parser_args.add_argument(
        "--slugs",
        nargs="+",
        help="Build only these slugs (used by the incremental cache wrapper)",
    )
    parser_args.add_argument("--verbose", action="store_true")
    args = parser_args.parse_args()
    VERBOSE = args.verbose

    # Griffe's docstring parser emits a warning per signature/docstring mismatch.
    # In verbose mode keep them visible; otherwise silence so the progress bar
    # isn't drowned out.
    if not VERBOSE:
        logging.getLogger("griffe").setLevel(logging.ERROR)
        logging.getLogger("mkdocstrings").setLevel(logging.ERROR)

    # Verify griffe is available
    try:
        from griffe import GriffeLoader, Parser, DocstringSectionKind  # noqa: F401
    except ImportError:
        print("ERROR: griffe not installed. Run: pip install griffe")
        sys.exit(1)

    _install_griffe_overload_guard()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sha = _get_commit_sha()

    from griffe import GriffeLoader, Parser, Module as GriffeModule
    loader = GriffeLoader(
        docstring_parser=Parser.numpy,
        search_paths=[str(LUCID_SRC)],
        allow_inspection=False,  # static analysis only — no C extension import needed
    )

    manifest = MODULE_MANIFEST
    if args.module:
        if args.module not in manifest and args.module not in _SYNTH_SLUGS:
            print(f"ERROR: unknown slug '{args.module}'. Valid: {list(manifest) + list(_SYNTH_SLUGS)}")
            sys.exit(1)

    # Build lucid top-level once (internally) — used as the source for any
    # synthesized slugs.  It's NEVER emitted to disk as ``lucid.json`` because
    # the bare ``lucid`` namespace shouldn't appear in the sidebar.
    #
    # Slug-filtering: ``--module`` selects a single slug; ``--slugs`` is
    # the multi-slug form used by the incremental cache wrapper.  We
    # apply whichever was passed and narrow both ``manifest`` and
    # ``synth_slugs`` accordingly so the build does the minimum work
    # to refresh just the requested set.
    selected_slugs: set[str] | None = None
    if args.module:
        selected_slugs = {args.module}
    elif args.slugs:
        selected_slugs = set(args.slugs)

    synth_slugs = _SYNTH_SLUGS
    if selected_slugs is not None:
        synth_slugs = {
            slug: payload
            for slug, payload in _SYNTH_SLUGS.items()
            if slug in selected_slugs
        }

    lucid_data: dict[str, Any] | None = None
    if synth_slugs:
        try:
            with _ElapsedSpinner("loading lucid/ tree (Griffe AST walk)..."):
                lucid_mod = loader.load("lucid")
                if isinstance(lucid_mod, GriffeModule):
                    lucid_data = _serialise_module(lucid_mod, Parser.numpy, sha, slug="lucid")
                    # Blind-spot rescue (category 5): lazy ``__getattr__``
                    # imports composite ops on demand, so names like
                    # ``lucid.erfc`` / ``lucid.allclose`` are runtime-visible
                    # but absent from ``lucid/__init__.py``'s ``__all__`` /
                    # explicit imports.  Griffe's static walk therefore misses
                    # them and the synth-bucket pipeline below has nothing to
                    # partition.  Walk each lazy-name's home module and inject
                    # the serialised member directly into ``lucid_data`` so
                    # the synth slugs (``lucid.ops.composite`` etc.) pick
                    # them up.
                    _inject_lazy_loaded_members(
                        loader, Parser.numpy, sha, lucid_data,
                    )
        except Exception as exc:
            print(f"  [warn] failed to load lucid for synthesis: {exc}")

    # Regular slugs from MODULE_MANIFEST — narrow by ``--module`` /
    # ``--slugs`` so partial rebuilds only touch the requested set.
    if selected_slugs is not None:
        manifest = {
            slug: griffe_path
            for slug, griffe_path in manifest.items()
            if slug in selected_slugs
        }

    total = len(manifest) + len(synth_slugs)
    print(f"Building {total} module(s)  [commit: {sha}]")

    manifest_items: list[tuple[str, str]] = [
        (slug, griffe_path) for slug, griffe_path in manifest.items()
    ]
    synth_items: list[tuple[str, str]] = []
    if lucid_data:
        synth_items = [
            (slug, filter_subcat) for slug, filter_subcat in synth_slugs.items()
        ]

    # ── Parallel manifest builds ────────────────────────────────────────
    # Each manifest slug is independent — distribute across a process
    # pool so the per-module work overlaps.  Measured speedup vs.
    # sequential is modest (~20-30 %) because each worker re-walks
    # ``lucid/`` once for its first task — that cost dominates over the
    # per-module emit work — but it's still wall-clock cheaper than
    # serial, and the worker model leaves room to share more state in
    # the future.
    #
    # Synth slugs need ``lucid_data`` (heavy to pickle into workers)
    # and are cheap individually, so we keep them on the main process
    # below.
    #
    # ``LUCID_API_BUILD_WORKERS`` overrides the pool size for benchmarking;
    # ``0`` or ``1`` disables parallelism (single-process fallback for
    # easier debugging of build crashes).
    import os as _os

    n_workers_env = _os.environ.get("LUCID_API_BUILD_WORKERS")
    if n_workers_env is not None:
        try:
            n_workers = int(n_workers_env)
        except ValueError:
            n_workers = 0
    else:
        # Cap at 8 even on 16-core M-Max — Griffe's AST walk per worker
        # is GIL-free but spawns ``importlib`` machinery whose I/O
        # contention dominates beyond ~8 parallel readers.
        n_workers = 1  # single-process reuses warm GriffeLoader + _AST_DOC_INDEX

    completed: list[str] = []
    # ``failed`` tracks every slug whose worker threw — we surface
    # them at the end with a hard exit code so CI / pre-commit never
    # accepts a partially-emitted JSON dir.  Earlier behaviour was to
    # print the failure and move on, which left stale builds passing
    # silently.
    failed: list[tuple[str, str]] = []

    def _on_complete(slug: str) -> None:
        completed.append(slug)
        _emit(f"  ✓ {slug:32}  ({len(completed)}/{total})")

    if n_workers > 1 and len(manifest_items) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import signal as _signal
        _emit(f"  [pool] {n_workers} workers building {len(manifest_items)} manifest module(s)…")
        # SIGINT cleanup: a bare ``with ProcessPoolExecutor`` leaves
        # worker processes running for a few seconds after Ctrl-C
        # while the executor drains in-flight tasks.  Installing an
        # explicit handler that calls ``pool.shutdown(wait=False,
        # cancel_futures=True)`` makes the cancel snappy.  We restore
        # the prior handler on the normal-exit path so subsequent
        # phases of the build pipeline don't inherit it.
        prev_sigint = _signal.getsignal(_signal.SIGINT)
        pool: Any = ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_init,
        )

        def _on_sigint(_signum: int, _frame: Any) -> None:
            print("\n  [pool] SIGINT received — shutting down workers…", file=sys.stderr)
            try:
                pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            # Re-raise as KeyboardInterrupt so the outer ``with``
            # cleanup runs and the script exits with the canonical
            # 130 exit code.
            raise KeyboardInterrupt

        _signal.signal(_signal.SIGINT, _on_sigint)
        try:
            with pool:
                futures = {
                    pool.submit(_worker_build_one, slug, griffe_path, sha): slug
                    for slug, griffe_path in manifest_items
                }
                for fut in as_completed(futures):
                    slug = futures[fut]
                    try:
                        fut.result()
                    except Exception as exc:
                        print(f"  [error] {slug}: {exc}", file=sys.stderr)
                        failed.append((slug, str(exc)))
                    else:
                        _on_complete(slug)
        finally:
            _signal.signal(_signal.SIGINT, prev_sigint)
    else:
        for slug, griffe_path in manifest_items:
            try:
                build_one(slug, griffe_path, loader, Parser.numpy, sha)
                _on_complete(slug)
            except Exception as exc:
                print(f"  [error] {slug}: {exc}", file=sys.stderr)
                failed.append((slug, str(exc)))

    # Synth slugs stay on the main process — ``lucid_data`` is large
    # (whole-package serialised tree) and pickling it into each worker
    # would dominate the wall-clock gain.
    for slug, filter_subcat in synth_items:
        assert lucid_data is not None
        try:
            build_synth(slug, filter_subcat, lucid_data)
            _on_complete(slug)
        except Exception as exc:
            print(f"  [error] {slug}: {exc}", file=sys.stderr)
            failed.append((slug, str(exc)))

    # ── Post-build sanity check ─────────────────────────────────────────
    # Verify every expected slug emitted a JSON.  Catches the failure
    # mode where a worker raised silently or wrote nothing.  Without
    # this, ``pnpm dev`` could ship a docs site missing whole modules
    # and the only signal would be 404s on user click.
    expected_slugs = {slug for slug, _ in manifest_items} | {
        slug for slug, _ in synth_items
    }
    missing_files = sorted(
        slug for slug in expected_slugs if not (OUT_DIR / f"{slug}.json").exists()
    )
    if missing_files:
        print("", file=sys.stderr)
        print(
            f"  ❌  {len(missing_files)} expected JSON(s) missing from output:",
            file=sys.stderr,
        )
        for slug in missing_files:
            print(f"      - {slug}", file=sys.stderr)

    if failed or missing_files:
        print(
            f"\n  Build incomplete — {len(failed)} worker error(s), "
            f"{len(missing_files)} missing output(s).",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\nDone. Files in {OUT_DIR}")


if __name__ == "__main__":
    main()
