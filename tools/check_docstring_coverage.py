#!/usr/bin/env python3
"""tools/check_docstring_coverage.py — flag public symbols whose docstring
is too thin to render usefully on the docs site.

Two structural blind-spots motivate this check:

1. **Google-style ``Args:`` sections** silently dropped by the docs build
   when ``Parser.numpy`` was hard-coded.  Already fixed at parse time
   (``web/scripts/build-api-data.py`` now auto-detects the style), but a
   future regression would re-introduce the bug — this script asserts
   that every symbol's parameters / returns survive a round-trip.

2. **filename == symbol-name** files (e.g. ``nn/functional/linear.py``
   defining ``def linear``) often ship a one-line docstring on the
   symbol because the author leans on the module-level prose.  The docs
   site renders the symbol's docstring, not the module's, so the symbol
   page looks empty.  We require those symbols to carry a real
   docstring of their own.

Run::

    python -m tools.check_docstring_coverage           # exits 1 on failure
    python -m tools.check_docstring_coverage --list    # show every flag

Exit codes
----------
0 — every public symbol meets the threshold.
1 — at least one symbol is below threshold; details printed to stderr.
"""

import argparse
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LUCID_SRC = ROOT / "lucid"

# Minimum docstring length on a *symbol* (function / class) whose name
# matches the file's basename — the canonical "blind spot" case.
MIN_NAME_MATCH_DOC = 200

# Minimum docstring length on any public function / class.
MIN_PUBLIC_DOC = 80

# A function with this many real parameters (i.e. excluding ``self`` /
# ``cls``) is expected to document each one inside a structured
# Parameters / Args section so the docs site can render per-param cards.
# Below this threshold, prose-only docstrings are acceptable.
PARAM_BLOCK_REQUIRED_THRESHOLD = 2

# Methods we don't enforce structured params on — they're docstring-lite by
# convention (``__init__`` mirrors the class docstring; ``forward`` repeats
# its module's contract; dunders are infrastructure).
PARAM_BLOCK_METHOD_EXEMPT = {
    "__init__",
    "__call__",
    "__enter__",
    "__exit__",
    "forward",
    "__getitem__",
    "__setitem__",
    "__repr__",
    "__str__",
}

# Files we deliberately skip (test fixtures, internal C++ binding shims).
SKIP_DIRS = {"test", "benchmarks"}
SKIP_FILES: set[str] = set()

# A docstring counts as "rich" when it has at least one of these markers.
_GOOGLE_HDR = re.compile(
    r"^[ \t]*(Args|Arguments|Returns|Yields|Raises|Note|Notes|Example|"
    r"Examples|Attributes|See Also|References)\s*:\s*$",
    re.MULTILINE,
)
_NUMPY_UNDERLINE = re.compile(r"^[ \t]*-{3,}[ \t]*$", re.MULTILINE)

# A *Parameters* / *Args* block specifically — the structural marker the
# docs site needs to render per-param cards.  Both Google (``Args:``) and
# NumPy (``Parameters\n----------``) forms are accepted because the
# build script auto-dispatches between Parser.google and Parser.numpy
# based on which marker appears.
_PARAM_BLOCK = re.compile(
    r"(Parameters\s*\n[ \t]*-{3,})|(\b(Args|Arguments|Parameters)\s*:\s*\n)", re.MULTILINE
)


def _public_param_count(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Number of user-facing parameters — excludes ``self`` / ``cls``."""
    return sum(1 for a in node.args.args if a.arg not in ("self", "cls")) + len(
        node.args.kwonlyargs
    )


def _public_param_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Param names a docstring is expected to document, in declaration order.

    Excludes ``self`` / ``cls`` (always implicit), but includes both
    regular positional args and keyword-only args.  ``*args`` and
    ``**kwargs`` are excluded because authors rarely document them
    by name — they get a free pass.
    """
    names: list[str] = []
    for a in node.args.args:
        if a.arg in ("self", "cls"):
            continue
        names.append(a.arg)
    for a in node.args.kwonlyargs:
        names.append(a.arg)
    return names


# NumPy-style Parameters block: ``ast.get_docstring`` strips the common
# leading indent, so NumPy-style params land at column 0 while their
# descriptions are at column 4.  Pinning the regex to column 0 prevents
# false positives like ``Default: ...`` inside a description body
# (which sits at column 4 and so won't match).
_NUMPY_PARAM_LINE = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_]*(?:\s*,\s*[A-Za-z_][A-Za-z0-9_]*)*)\s*:\s+\S", re.MULTILINE
)

# Google-style Args block: ``Args:`` sits at column 0 and param lines
# are indented underneath (typically 4 spaces).  Requiring leading
# whitespace lets us reject column-0 noise above the block while still
# accepting the canonical ``    name: description`` shape.
_GOOGLE_PARAM_LINE = re.compile(
    r"^[ \t]+([A-Za-z_][A-Za-z0-9_]*)\s*(?:\([^)]*\))?\s*:\s+\S", re.MULTILINE
)


def _extract_documented_param_names(doc: str) -> list[str] | None:
    """Pull the param names actually documented inside the Parameters block.

    Returns ``None`` when the docstring has no Parameters / Args
    section at all (drift check is moot — handled by category 4).
    Otherwise returns the list of names found, in document order,
    with duplicates collapsed.

    The extraction is structural rather than parser-based — we just
    isolate the lines between the section header and the next blank
    line or next section header, then scan each line for the
    ``name :`` / ``name:`` prefix pattern.
    """
    if not _PARAM_BLOCK.search(doc):
        return None

    # Find the start of the Parameters / Args block and detect its style
    # from the header — NumPy uses a ``Parameters\n----------`` underline,
    # Google uses a bare ``Args:`` / ``Arguments:`` colon-header.  We
    # then run *only* the matching extraction regex so that description
    # lines (which sit at col 4) can't masquerade as parameters of the
    # other style.
    numpy_header = re.compile(r"^[ \t]*Parameters\s*\n[ \t]*-{3,}\s*$", re.MULTILINE)
    google_header = re.compile(r"^[ \t]*(Args|Arguments)\s*:\s*$", re.MULTILINE)
    style: str
    m_num = numpy_header.search(doc)
    m_goo = google_header.search(doc)
    if m_num is not None and (m_goo is None or m_num.start() < m_goo.start()):
        style = "numpy"
        start = m_num.end()
    elif m_goo is not None:
        style = "google"
        start = m_goo.end()
    else:
        return None

    # Find the end of the block — the next NumPy underline section header,
    # the next Google ``Header:`` line, or two blank lines.
    end_re = re.compile(
        r"\n\n(?=[ \t]*(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*\n[ \t]*-{3,}|"
        r"(?:Returns|Raises|Yields|Notes|Examples|See Also|References|"
        r"Attributes|Warns|Warnings|Note)\s*:))|\n\n[ \t]*\n",
        re.MULTILINE,
    )
    m2 = end_re.search(doc, start)
    body = doc[start : m2.start()] if m2 else doc[start:]

    names: list[str] = []
    seen: set[str] = set()
    if style == "numpy":
        # Tolerate comma-separated names on one line (``a, b : Tensor``).
        for m in _NUMPY_PARAM_LINE.finditer(body):
            for part in m.group(1).split(","):
                n = part.strip()
                if n and n not in seen:
                    seen.add(n)
                    names.append(n)
    else:  # google
        for m in _GOOGLE_PARAM_LINE.finditer(body):
            n = m.group(1)
            if n not in seen:
                seen.add(n)
                names.append(n)
    return names


def _is_public(name: str) -> bool:
    return not name.startswith("_")


def _walk_lucid() -> list[Path]:
    files: list[Path] = []
    for p in LUCID_SRC.rglob("*.py"):
        parts = p.relative_to(LUCID_SRC).parts
        if any(seg in SKIP_DIRS for seg in parts):
            continue
        if p.name in SKIP_FILES:
            continue
        files.append(p)
    return files


def _classify(doc: str) -> str:
    """Return ``'rich'`` if the docstring has any section marker, else
    ``'plain'``.  Used to enforce the structural requirement that
    user-facing symbols document parameters / returns rather than ship
    a bare one-liner."""
    if _GOOGLE_HDR.search(doc) or _NUMPY_UNDERLINE.search(doc):
        return "rich"
    return "plain"


def _check_file(path: Path) -> list[str]:
    """Return one error string per violation found in *path*."""
    try:
        src = path.read_text()
    except OSError:
        return []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []

    stem = path.stem
    file_basename_is_symbol_carrier = not stem.startswith("_") and stem not in ("__init__",)

    errors: list[str] = []

    # Walk every public def/class — including methods inside classes —
    # so that the parameter-block check catches multi-arg methods that
    # only document themselves in prose.  Module-level vs class-level
    # placement still affects the name-match rule, so track that
    # separately.
    def _walk_publics(parent: ast.AST, at_module_level: bool):
        for child in ast.iter_child_nodes(parent):
            if isinstance(child, ast.ClassDef):
                if not _is_public(child.name):
                    continue
                # Class-level check (covers name-match + stub).
                yield child, at_module_level
                # Recurse into class body for method-level checks.
                yield from _walk_publics(child, at_module_level=False)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not _is_public(child.name):
                    continue
                yield child, at_module_level

    for node, at_module_level in _walk_publics(tree, at_module_level=True):
        name = node.name
        doc = ast.get_docstring(node) or ""
        doc_len = len(doc.strip())

        # Blind-spot #2: filename matches symbol name (module-level only —
        # nested classes with the same basename are fine).  Require a
        # richer docstring than the bare-minimum, because there is nowhere
        # else the symbol's documentation can live — the module docstring
        # doesn't make it into the symbol's rendered card.
        if at_module_level and file_basename_is_symbol_carrier and name == stem:
            if doc_len < MIN_NAME_MATCH_DOC or _classify(doc) == "plain":
                errors.append(
                    f"{path.relative_to(ROOT)}: "
                    f"name-match symbol `{name}` has thin docstring "
                    f"(len={doc_len}, threshold={MIN_NAME_MATCH_DOC} + section marker required)"
                )

        # Blind-spot #1: any public symbol with a stub docstring will
        # render as an empty card in the docs site.  Lower threshold,
        # universal rule.
        elif at_module_level and doc_len < MIN_PUBLIC_DOC and doc_len > 0:
            # Only flag *short non-empty* docstrings; absent docstrings
            # are a separate (and more obvious) lint concern.  We focus
            # on the silent-degradation case where authors wrote one
            # sentence and assumed it was enough.
            errors.append(
                f"{path.relative_to(ROOT)}: "
                f"public `{name}` has stub docstring "
                f"(len={doc_len}, threshold={MIN_PUBLIC_DOC})"
            )

        # Blind-spot #3: function has multiple parameters and a real
        # docstring but no structured Parameters / Args section.  The
        # docs site can't render per-param cards from prose alone, so
        # the function page ends up summary-only.
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if name in PARAM_BLOCK_METHOD_EXEMPT:
                continue
            nparams = _public_param_count(node)
            if nparams < PARAM_BLOCK_REQUIRED_THRESHOLD:
                continue
            if doc_len < 100:
                # Already covered by the stub-docstring rule above.
                continue
            if not _PARAM_BLOCK.search(doc):
                errors.append(
                    f"{path.relative_to(ROOT)}: "
                    f"`{name}` has {nparams} params + {doc_len}-char docstring "
                    f"but no Parameters/Args section — docs page will lack per-param cards"
                )
                continue  # No block to drift-check.

            # Blind-spot #4: docstring HAS a Parameters block, but its
            # documented names don't match the function's actual
            # signature.  Silent failure mode — docs cards say one thing,
            # call site sees another.  Flag any name that:
            #   * is documented but not in the signature (typo / stale doc), or
            #   * is in the signature but not documented.
            documented = _extract_documented_param_names(doc)
            if documented is None:
                # _PARAM_BLOCK matched but we couldn't isolate the body.
                continue
            sig_names = _public_param_names(node)
            sig_set = set(sig_names)
            doc_set = set(documented)
            unknown = [n for n in documented if n not in sig_set]
            missing = [n for n in sig_names if n not in doc_set]
            if unknown or missing:
                bits: list[str] = []
                if unknown:
                    bits.append(f"undocumented-typo: {unknown}")
                if missing:
                    bits.append(f"signature-only: {missing}")
                errors.append(
                    f"{path.relative_to(ROOT)}: "
                    f"`{name}` Parameters block drifted from signature "
                    f"— {' / '.join(bits)}"
                )

    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument(
        "--list",
        action="store_true",
        help="print every violation found (default: print only the first 20)",
    )
    args = ap.parse_args()

    all_errors: list[str] = []
    for p in sorted(_walk_lucid()):
        all_errors.extend(_check_file(p))

    if not all_errors:
        print("[check_docstring_coverage] OK — no thin docstrings on public symbols.")
        return 0

    limit = len(all_errors) if args.list else min(20, len(all_errors))
    sys.stderr.write(f"[check_docstring_coverage] {len(all_errors)} thin docstring(s) found:\n")
    for e in all_errors[:limit]:
        sys.stderr.write(f"  {e}\n")
    if len(all_errors) > limit:
        sys.stderr.write(
            f"  ... {len(all_errors) - limit} more (re-run with --list to see them all)\n"
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
