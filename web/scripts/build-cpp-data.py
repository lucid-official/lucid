"""build-cpp-data.py — C++ engine docs extractor for the Lucid docs site.

Walks ``lucid/_C/**.h`` headers, parses each via libclang, extracts every
public class / struct / free function / enum / typedef along with the
adjacent ``//`` doc-comment block, and emits per-compound JSON files
matching the existing :class:`ApiModule` / :class:`ApiClass` /
:class:`ApiFunction` schemas the Next.js site already renders.

The result is that no frontend changes are needed — the new JSON files
drop into ``web/public/api-data/`` and the existing
``getAllModuleSlugs()`` sidebar walker picks them up.  A single label
override in ``api/layout.tsx`` (``"lucid._C.engine"`` → ``"C++ Engine"``)
positions it under the same Model Zoo / Tensor pattern.

Source-comment convention
-------------------------
Lucid's C++ uses plain ``//`` block comments — *not* Doxygen ``///`` or
``/** */``.  ``libclang.cursor.raw_comment`` only surfaces the latter,
so we walk the original source line-by-line and collect the consecutive
``//`` lines immediately preceding each declaration.  Blank line ends
the run; non-comment code ends it harder.

Usage
-----
::

    python web/scripts/build-cpp-data.py               # full sweep
    python web/scripts/build-cpp-data.py --root <dir>  # alternate header root
    python web/scripts/build-cpp-data.py --debug       # verbose parse logs
"""

from __future__ import annotations  # tooling — H1 OK

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from clang.cindex import Config, Cursor, CursorKind, Index, TranslationUnit

# Resolve repo + libclang path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
HEADERS_ROOT = REPO_ROOT / "lucid" / "_C"
OUT_DIR = REPO_ROOT / "web" / "public" / "api-data"

# libclang dylib — Homebrew LLVM on Apple Silicon.  Override via
# LUCID_LIBCLANG env var on other systems.
_LIBCLANG = "/opt/homebrew/opt/llvm/lib/libclang.dylib"
Config.set_library_file(_LIBCLANG)


GITHUB_BASE = "https://github.com/lucid-official/lucid/blob"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "main"


# ---------------------------------------------------------------------------
# Comment extraction (plain ``//`` style — libclang's ``raw_comment`` skips these)
# ---------------------------------------------------------------------------


def _collect_comment_above(source_lines: list[str], line0: int) -> str:
    """Walk upward from ``line0`` (0-indexed) collecting the run of
    consecutive ``//`` lines.  A blank line breaks the run only if
    something has been collected; non-``//`` non-blank code breaks
    unconditionally.  Returns the joined text stripped of comment markers
    and leading whitespace."""
    out: list[str] = []
    i = line0 - 1
    while i >= 0:
        s = source_lines[i].lstrip()
        if s.startswith("//"):
            # Strip leading "//" + optional space
            stripped = s[2:]
            if stripped.startswith(" "):
                stripped = stripped[1:]
            out.append(stripped.rstrip())
            i -= 1
        elif s == "":
            if out:
                break
            i -= 1
        else:
            break
    return "\n".join(reversed(out)).strip()


# ---------------------------------------------------------------------------
# Doc-comment → NumPy-style section split
# ---------------------------------------------------------------------------


_SECTION_HEADERS = frozenset(
    (
        "Parameters",
        "Returns",
        "Yields",
        "Raises",
        "Warns",
        "Notes",
        "Examples",
        "Attributes",
        "Math",
        "Shape",
        "References",
        "See Also",
    )
)


def _empty_doc() -> dict[str, Any]:
    return {
        "summary": None,
        "extended": None,
        "parameters": [],
        "returns": None,
        "raises": [],
        "examples": [],
        "notes": [],
        "attributes": [],
        "warns": [],
    }


def _split_into_sections(comment: str) -> tuple[str, dict[str, str]]:
    """Split a NumPy-style docstring into the leading body + per-section
    text blocks.  A section is identified by a header line whose next
    line is dashes-only (NumPy convention)::

        Parameters
        ----------
        x : Tensor
            Input.

    Returns ``(body_text, {section_name: section_text})``."""
    lines = comment.splitlines()
    sections: dict[str, list[str]] = {"_body": []}
    current = "_body"

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        nxt = lines[i + 1].rstrip() if i + 1 < len(lines) else ""
        header_candidate = line.strip()
        if header_candidate in _SECTION_HEADERS and nxt.strip() and set(nxt.strip()) == {"-"}:
            current = header_candidate
            sections.setdefault(current, [])
            i += 2
            continue
        sections[current].append(lines[i])
        i += 1

    body = "\n".join(sections.pop("_body", [])).strip()
    return body, {k: "\n".join(v).strip() for k, v in sections.items()}


def _parse_field_list(block: str) -> list[dict[str, Any]]:
    """Parse NumPy-style ``name : type`` entries with indented descriptions.

    Multi-line descriptions are joined with single spaces.  A trailing
    ``optional`` or ``= <value>`` in the type slot becomes the
    ``default`` field."""
    items: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw in block.splitlines():
        if not raw.strip():
            if current is not None:
                # Blank inside an entry — treat as soft paragraph break.
                current["description"] = (current["description"] + " ").rstrip()
            continue
        if not raw.startswith((" ", "\t")):
            if current is not None:
                items.append(current)
            m = re.match(r"^\s*(\S[\S ]*?)(?:\s*:\s*(.+?))?\s*$", raw)
            if not m:
                current = None
                continue
            name = m.group(1)
            anno = m.group(2)
            default: str | None = None
            if anno:
                m2 = re.search(r"\s*=\s*(\S.*)$", anno)
                if m2:
                    default = m2.group(1).strip()
                    anno = anno[: m2.start()].rstrip(" ,")
                m3 = re.search(r",\s*optional\s*$", anno)
                if m3:
                    anno = anno[: m3.start()].rstrip()
                    if default is None:
                        default = "None"
            current = {"name": name, "annotation": anno, "default": default, "description": ""}
        else:
            if current is None:
                continue
            txt = raw.strip()
            current["description"] = (
                (current["description"] + " " + txt).strip() if current["description"] else txt
            )
    if current is not None:
        items.append(current)
    return items


def _parse_raise_list(block: str) -> list[dict[str, Any]]:
    """Same shape as field-list but emit ``{annotation, description}``
    rather than ``{name, annotation, default, description}`` so the
    web renderer's :class:`DocstringRaise` schema is satisfied."""
    out: list[dict[str, Any]] = []
    for f in _parse_field_list(block):
        out.append({"annotation": f["annotation"] or f["name"], "description": f["description"]})
    return out


def _parse_returns(block: str) -> dict[str, Any] | None:
    """Parse a Returns block.  Two NumPy idioms supported:

    - ``TypeName\\n    Description.``  (type on first line, body indented)
    - ``Description spanning lines.``  (no explicit type)
    """
    text = block.strip()
    if not text:
        return None
    lines = text.splitlines()
    first = lines[0].strip()
    rest = [l.strip() for l in lines[1:] if l.strip()]
    if rest:
        return {"annotation": first, "description": " ".join(rest)}
    return {"annotation": None, "description": first}


def _parse_examples(block: str) -> list[str]:
    """Examples block → list of code snippets (one entry per blank-line
    separated paragraph), preserving original indentation."""
    text = block.rstrip()
    if not text:
        return []
    return [p.strip("\n") for p in re.split(r"\n\s*\n", text) if p.strip()]


def _rst_inline_markup(text: str) -> str:
    """Non-math reST inline markup → markdown.  Mirrors the helper of the same
    name in ``build-api-data.py`` (kept as a local copy because the two build
    scripts are standalone entrypoints).  Without it, cross-reference roles in
    C++ doc-comments (``:class:`CpuBackend```) leak as raw text on the engine
    page."""
    if not text:
        return text
    # reST external hyperlinks: `text <url>`_ → [text](url)
    text = re.sub(r"`([^`<]+?)\s*<((?:https?|ftp|mailto):[^>]+)>`__?", r"[\1](\2)", text)
    # double-backtick code: ``code`` → `code`
    text = re.sub(r"``([^`]+)``", r"`\1`", text)
    # remove citations, then collapse any remaining role to inline code
    text = re.sub(r":cite:[a-z]*:?`[^`]+`", "", text)
    text = re.sub(r":[a-zA-Z][\w.+-]*(?::[a-zA-Z][\w.+-]*)*:`~?([^`]+)`", r"`\1`", text)
    return text


def _split_sections(comment: str) -> dict[str, Any]:
    """Map a NumPy-style ``//`` comment block to the docstring schema the
    web site renders.

    Recognises the standard NumPy headers (Parameters / Returns / Notes /
    Examples / Attributes / Raises / Warns / Math / Shape / References /
    See Also).  Free-form prose with no headers stays backward-compatible:
    first paragraph → ``summary``, remainder → ``extended``."""
    if not comment:
        return _empty_doc()
    body, sections = _split_into_sections(comment)

    body_paras = re.split(r"\n\s*\n", body) if body else []
    summary = body_paras[0].replace("\n", " ").strip() if body_paras else None
    extended_parts: list[str] = []
    if len(body_paras) > 1:
        extended_parts.extend(p.strip() for p in body_paras[1:])

    parameters = _parse_field_list(sections.get("Parameters", ""))
    attributes = _parse_field_list(sections.get("Attributes", ""))
    raises_list = _parse_raise_list(sections.get("Raises", ""))
    warns_list = _parse_raise_list(sections.get("Warns", ""))
    returns = _parse_returns(sections.get("Returns", ""))

    notes_text = sections.get("Notes", "").strip()
    notes = [p.strip() for p in re.split(r"\n\s*\n", notes_text) if p.strip()] if notes_text else []

    examples = _parse_examples(sections.get("Examples", ""))

    # Math / Shape / References / See Also are folded into ``extended`` as
    # labelled blocks so they render below the main summary on the page.
    # MathText already passes through ``$...$`` / ``$$...$$`` and rST
    # ``.. math::`` directives, so the author can write either form.
    for label, key in (
        ("Math", "Math"),
        ("Shape", "Shape"),
        ("References", "References"),
        ("See Also", "See Also"),
    ):
        txt = sections.get(key, "").strip()
        if txt:
            extended_parts.append(f"**{label}**\n\n{txt}")

    extended = "\n\n".join(extended_parts) if extended_parts else None

    # Normalise reST cross-reference roles / hyperlinks in every prose field so
    # they don't leak as raw ``:role:`x``` text on the rendered engine page.
    for _fl in (parameters, attributes, raises_list, warns_list):
        for _item in _fl:
            if _item.get("description"):
                _item["description"] = _rst_inline_markup(_item["description"])
    if returns and returns.get("description"):
        returns["description"] = _rst_inline_markup(returns["description"])

    return {
        "summary": _rst_inline_markup(summary) if summary else summary,
        "extended": _rst_inline_markup(extended) if extended else extended,
        "parameters": parameters,
        "returns": returns,
        "raises": raises_list,
        "examples": examples,
        "notes": [_rst_inline_markup(n) for n in notes],
        "attributes": attributes,
        "warns": warns_list,
    }


# ---------------------------------------------------------------------------
# AST → JSON
# ---------------------------------------------------------------------------


def _is_private_namespace(name: str) -> bool:
    """Lucid's C++ marks implementation-detail namespaces in two shapes:
    a nested ``::detail`` / ``::internal`` (e.g. ``lucid::nn::detail``)
    AND a flat ``*_detail`` / ``*_internal`` (e.g. ``lucid::optim_detail``,
    ``lucid::ufunc_detail``).  Treat both as private."""
    if name in ("detail", "internal"):
        return True
    return name.endswith("_detail") or name.endswith("_internal")


def _is_public(c: Cursor) -> bool:
    """Skip private/anonymous declarations and anything inside an
    implementation-detail namespace (see :func:`_is_private_namespace`)."""
    if not c.spelling or c.spelling.startswith("_"):
        return False
    parent = c.semantic_parent
    while parent is not None and parent.kind == CursorKind.NAMESPACE:
        if _is_private_namespace(parent.spelling or ""):
            return False
        parent = parent.semantic_parent
    return True


def _qualified_name(c: Cursor) -> str:
    """``lucid::Tensor::data`` style fully-qualified name."""
    parts: list[str] = [c.spelling]
    p = c.semantic_parent
    while p is not None and p.kind in (
        CursorKind.NAMESPACE,
        CursorKind.CLASS_DECL,
        CursorKind.STRUCT_DECL,
    ):
        if p.spelling:
            parts.append(p.spelling)
        p = p.semantic_parent
    return "::".join(reversed(parts))


def _slug_from_qname(qname: str) -> str:
    """Engine docs use the Python-side prefix (``lucid._C.engine``) so the
    sidebar groups them under the existing pybind11-exposed surface.

    The C++ namespace ``lucid::`` wraps every public type — including
    it in the slug would produce redundant ``lucid._C.engine.lucid.Foo``.
    Strip that leading namespace so the URL reads
    ``lucid._C.engine.Foo`` (the qualified name is preserved separately
    in the JSON's ``path`` field).
    """
    parts = qname.split("::")
    if parts and parts[0] == "lucid":
        parts = parts[1:]
    return "lucid._C.engine." + ".".join(parts)


_METHOD_DEF_INDEX: dict[tuple[str, str], tuple[str, int]] | None = None


def _backend_method_def_index() -> dict[tuple[str, str], tuple[str, int]]:
    """Index out-of-line ``CpuBackend::`` / ``GpuBackend::`` definitions in ``methods/*.cpp``."""
    global _METHOD_DEF_INDEX
    if _METHOD_DEF_INDEX is not None:
        return _METHOD_DEF_INDEX
    idx: dict[tuple[str, str], tuple[str, int]] = {}
    for bk, cls_name in (("cpu", "CpuBackend"), ("gpu", "GpuBackend")):
        methods_dir = HEADERS_ROOT / "backend" / bk / "methods"
        if not methods_dir.is_dir():
            continue
        pat = re.compile(rf"\b{cls_name}::([A-Za-z_]\w*)\s*\(")
        for cpp_file in sorted(methods_dir.glob("*.cpp")):
            rel = cpp_file.resolve().relative_to(REPO_ROOT).as_posix()
            for lineno, line in enumerate(
                cpp_file.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
            ):
                if line.lstrip().startswith("//"):
                    continue
                m = pat.search(line)
                if m is not None:
                    key = (cls_name, m.group(1))
                    if key not in idx:
                        idx[key] = (rel, lineno)
    _METHOD_DEF_INDEX = idx
    return idx


def _source_link(c: Cursor, sha: str) -> str | None:
    loc = c.location
    if loc is None or loc.file is None:
        return None
    if c.kind in (CursorKind.CXX_METHOD, CursorKind.FUNCTION_TEMPLATE):
        parent = c.semantic_parent
        if parent is not None and parent.spelling in ("CpuBackend", "GpuBackend"):
            hit = _backend_method_def_index().get((parent.spelling, c.spelling))
            if hit is not None:
                rel_cpp, def_line = hit
                return f"{GITHUB_BASE}/{sha}/{rel_cpp}#L{def_line}"
    try:
        rel = Path(loc.file.name).resolve().relative_to(REPO_ROOT)
    except ValueError:
        return None
    return f"{GITHUB_BASE}/{sha}/{rel}#L{loc.line}"


def _build_signature(c: Cursor) -> str | None:
    """Best-effort one-line signature for the entry header.  Function
    and method declarations include parameters; classes include
    template parameters when present."""
    if c.kind in (
        CursorKind.FUNCTION_DECL,
        CursorKind.CXX_METHOD,
        CursorKind.CONSTRUCTOR,
        CursorKind.DESTRUCTOR,
        CursorKind.FUNCTION_TEMPLATE,
    ):
        params = []
        for arg in c.get_arguments():
            t = arg.type.spelling
            n = arg.spelling
            params.append(f"{t} {n}".strip())
        ret = c.result_type.spelling if c.result_type else ""
        sig = f"{ret} {c.spelling}({', '.join(params)})".strip()
        return sig
    if c.kind in (CursorKind.CLASS_DECL, CursorKind.STRUCT_DECL, CursorKind.CLASS_TEMPLATE):
        return c.spelling
    return None


def _comment_for(c: Cursor, source_lines: list[str] | None) -> str:
    """Look up the ``//`` comment block immediately above ``c``.  ``source_lines``
    is the file content lazily cached at the TU level — passing ``None``
    falls back to an empty string (safe for cross-file cursors where we
    don't have the source available)."""
    if source_lines is None:
        return ""
    line = c.extent.start.line
    return _collect_comment_above(source_lines, line - 1)


def _cpp_labels(c: Cursor) -> list[str]:
    """Extract C++-specific method kind labels for the docs site to
    colour-code (mirrors Python's ``@property`` / ``@staticmethod`` /
    etc. detection in ``_collect_labels``).

    Emitted labels live in the ``ApiLabel`` union on the TS side and
    drive both the kind-badge pill (e.g. ``CTOR`` / ``DTOR`` / ``OP``)
    and the per-name colour in ``api-kind-utils.ts``.  Order matches
    the precedence in the renderer (most specific first):

      cpp-ctor       — constructor
      cpp-dtor       — destructor
      cpp-operator   — ``operator+`` / ``operator=`` / ``operator()`` overload
      cpp-pure-virtual — pure-virtual method (``= 0``)
      cpp-virtual    — non-pure virtual
      cpp-static     — static member function
      cpp-const      — const-qualified non-static method
      cpp-template   — function template

    A method can carry multiple labels (e.g. ``virtual void foo() const``
    → ``cpp-virtual`` + ``cpp-const``); the badge renderer picks the
    highest-precedence one but the full list is preserved so future
    UI can compose more nuanced indicators."""
    labels: list[str] = []
    kind = c.kind
    if kind == CursorKind.CONSTRUCTOR:
        labels.append("cpp-ctor")
    elif kind == CursorKind.DESTRUCTOR:
        labels.append("cpp-dtor")
    elif c.spelling and c.spelling.startswith("operator") and not c.spelling[8:9].isalnum():
        # ``operator+`` / ``operator()`` / ``operator[]`` / ``operator=`` —
        # NOT ``operator_overload_helper`` (alphanumeric continuation).
        labels.append("cpp-operator")
    if kind in (CursorKind.CXX_METHOD, CursorKind.FUNCTION_TEMPLATE):
        try:
            if c.is_pure_virtual_method():
                labels.append("cpp-pure-virtual")
            elif c.is_virtual_method():
                labels.append("cpp-virtual")
        except Exception:  # noqa: BLE001
            pass
        try:
            if c.is_static_method():
                labels.append("cpp-static")
        except Exception:  # noqa: BLE001
            pass
        try:
            if c.is_const_method():
                labels.append("cpp-const")
        except Exception:  # noqa: BLE001
            pass
    if kind == CursorKind.FUNCTION_TEMPLATE:
        labels.append("cpp-template")
    return labels


def _serialise_function(
    c: Cursor, sha: str, source_lines: list[str] | None = None, subcategory: str | None = None
) -> dict[str, Any]:
    doc = _split_sections(_comment_for(c, source_lines))
    return {
        "name": c.spelling,
        "path": _qualified_name(c),
        "kind": "function",
        "labels": _cpp_labels(c),
        "signature": _build_signature(c),
        "source": _source_link(c, sha),
        "subcategory": subcategory,
        **doc,
    }


def _serialise_class(
    c: Cursor, sha: str, source_lines: list[str] | None = None, subcategory: str | None = None
) -> dict[str, Any]:
    doc = _split_sections(_comment_for(c, source_lines))
    methods: list[dict[str, Any]] = []
    # Dedup C++ overloads by bare method name.  Overloaded constructors
    # (``Foo()``, ``Foo(int)``, ``Foo(const Foo&)``) share the spelling
    # ``Foo`` — without dedup the renderer would emit multiple ``<h3
    # id="Foo">`` siblings, producing duplicate-key React warnings AND
    # ambiguous ``#Foo`` URL anchors.  Keeps the first encounter so the
    # signature ``_build_signature`` captures matches what the header
    # declares first, which by convention is the canonical form.
    seen_method_names: set[str] = set()
    for child in c.get_children():
        if not _is_public(child):
            continue
        if child.kind in (
            CursorKind.CXX_METHOD,
            CursorKind.CONSTRUCTOR,
            CursorKind.DESTRUCTOR,
            CursorKind.FUNCTION_TEMPLATE,
        ):
            if child.spelling in seen_method_names:
                continue
            seen_method_names.add(child.spelling)
            methods.append(_serialise_function(child, sha, source_lines))
    return {
        "name": c.spelling,
        "path": _qualified_name(c),
        "kind": "class",
        "class_kind": "regular",
        "bases": [b.spelling for b in c.get_children() if b.kind == CursorKind.CXX_BASE_SPECIFIER],
        "labels": [],
        "signature": _build_signature(c),
        "source": _source_link(c, sha),
        "subcategory": subcategory,
        "methods": methods,
        **doc,
    }


# ---------------------------------------------------------------------------
# Per-header traversal
# ---------------------------------------------------------------------------


_FUNCTION_KINDS = (CursorKind.FUNCTION_DECL, CursorKind.FUNCTION_TEMPLATE)
_CLASS_KINDS = (CursorKind.CLASS_DECL, CursorKind.STRUCT_DECL, CursorKind.CLASS_TEMPLATE)


def _subcategory_for_header(header: Path) -> str | None:
    """Map a header's location under ``lucid/_C/`` to the sidebar
    subcategory path so the engine docs nest the same way the Python
    docs do — by directory.  ``lucid/_C/ops/ufunc/Astype.h`` becomes
    ``"ops/ufunc"``; root-level headers (``version.h``, ``api.h``)
    return ``None`` and surface at the engine top level."""
    try:
        rel = header.resolve().relative_to(HEADERS_ROOT.resolve())
    except ValueError:
        return None
    parts = rel.parts[:-1]  # drop the filename
    if not parts:
        return None
    return "/".join(parts)


def _process_translation_unit(
    tu: TranslationUnit,
    header: Path,
    classes: dict[str, dict[str, Any]],
    functions: dict[str, dict[str, Any]],
    sha: str,
) -> None:
    """Walk the TU's preorder; collect public functions / classes whose
    *definition* lives in ``header`` (skip transitive includes).  The
    header's source lines are read once and threaded into the serialisers
    so each declaration can pull its own comment block from line index."""
    src_lines = header.read_text(encoding="utf-8", errors="replace").splitlines()
    sub = _subcategory_for_header(header)

    header_str = str(header.resolve())
    for c in tu.cursor.walk_preorder():
        if c.location.file is None:
            continue
        if str(c.location.file.name) != header_str:
            continue
        if not _is_public(c):
            continue
        if c.kind in _FUNCTION_KINDS:
            if c.semantic_parent is None or c.semantic_parent.kind == CursorKind.NAMESPACE:
                key = _qualified_name(c)
                if key not in functions:
                    functions[key] = _serialise_function(c, sha, src_lines, sub)
        elif c.kind in _CLASS_KINDS and c.is_definition():
            key = _qualified_name(c)
            classes[key] = _serialise_class(c, sha, src_lines, sub)


def _parse_header(idx: Index, header: Path) -> TranslationUnit | None:
    """Parse one header in C++20 mode with the engine's include root."""
    try:
        return idx.parse(
            str(header),
            args=["-x", "c++", "-std=c++20", f"-I{HEADERS_ROOT}", f"-I{HEADERS_ROOT.parent}"],
            options=TranslationUnit.PARSE_SKIP_FUNCTION_BODIES,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  ✗ parse failed: {header.name} — {exc}")
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=str(HEADERS_ROOT), help="C++ headers root (default: lucid/_C)")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    root = Path(args.root).resolve()
    sha = _git_sha()[:9]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    idx = Index.create()
    classes: dict[str, dict[str, Any]] = {}
    functions: dict[str, dict[str, Any]] = {}

    headers = sorted(root.rglob("*.h"))
    print(f"Scanning {len(headers)} headers under {root}/ …")
    for h in headers:
        tu = _parse_header(idx, h)
        if tu is None:
            continue
        _process_translation_unit(tu, h, classes, functions, sha)
        if args.debug:
            print(f"  ✓ {h.relative_to(REPO_ROOT)}")

    # The Python pipeline packs full class detail (methods + docstring) inline
    # into the module's ``members`` list — the dynamic ``/api/[...slug]/page.tsx``
    # route resolves ``/api/lucid._C.engine/Adam`` by ``findMember`` over the
    # module JSON.  We mirror that here: every C++ class lands inline with all
    # its methods, and we emit ONLY the single ``lucid._C.engine.json`` index.
    # Per-class detail JSONs are intentionally not written — they would be
    # picked up by ``getAllModuleSlugs()`` and pollute the sidebar with 200
    # phantom top-level entries.
    #
    # Also wipe any stale per-class JSONs left behind by previous runs of an
    # earlier version of this script that did emit them.
    for stale in OUT_DIR.glob("lucid._C.engine.*.json"):
        stale.unlink()

    members: list[dict[str, Any]] = []
    for qname, data in sorted(classes.items()):
        members.append(data)
    for qname, data in sorted(functions.items()):
        members.append(data)

    engine_json = {
        "slug": "lucid._C.engine",
        "name": "engine",
        "path": "lucid._C.engine",
        "kind": "module",
        "source": None,
        "summary": "C++ engine — tensor storage, ops, autograd graph, "
        "backend dispatch (CPU=Accelerate / GPU=MLX).",
        "extended": "Lucid's compute core.  All Python-side ops route here "
        "via pybind11 bindings (`from lucid._C import engine`).",
        "parameters": [],
        "returns": None,
        "raises": [],
        "examples": [],
        "notes": [],
        "attributes": [],
        "warns": [],
        "members": members,
    }
    (OUT_DIR / "lucid._C.engine.json").write_text(
        json.dumps(engine_json, indent=2, ensure_ascii=False)
    )

    print(
        f"\nEmitted 1 engine index with {len(classes)} classes inline "
        f"+ {len(functions)} free functions → {OUT_DIR / 'lucid._C.engine.json'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
