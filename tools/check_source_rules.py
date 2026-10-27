#!/usr/bin/env python3
"""tools/check_source_rules.py — Hard Rules H1, H5, H6 and H7, and dead paths.

Each rule is a property of one source file, so each is checked per file:

* **H1** (``future-annotations``) — no ``from __future__ import
  annotations``, in ``lucid/`` and ``tools/``.  Python 3.14 evaluates
  annotations lazily (PEP 649); the import turns every annotation back
  into a string.
* ``except-parens`` — multiple exception types in an ``except`` clause are
  parenthesized (``except (A, B):``), in ``lucid/`` and ``tools/``, so tools
  and environments whose Python parser predates PEP 758 can still parse the
  source (LCD-648).  The rules below hold in ``lucid/`` only: ``tools/``
  converts the reference framework's checkpoints and names it.
* **H5** (``ref-import``, ``ref-name``) — the reference framework is
  imported, and named, only in ``lucid/test/_fixtures/ref_framework.py``.
  Every other test takes the ``ref`` fixture.  The name is read from that
  file's ``_REF_NAME``, so it is written down once.
* **H6** (``gpu-name``) — the other vendor's GPU API is not named; Lucid's
  GPU is ``metal``.
* **H7** (``string-annotation``) — no annotation is a string.  A name used
  only for typing is imported under ``TYPE_CHECKING``.  ``Literal[...]``
  arguments and ``Annotated`` metadata are values, not types.
* ``dead-path`` — prose does not name a ``lucid.`` path that is gone.  Prose
  is a comment or a bare string (a docstring) in Python, and any line of
  C++ or Markdown.  Nothing is imported to check a path.  Its longest
  prefix that is a module (a ``.py`` or ``.pyi`` file, or a directory) must
  be followed by a name that module binds at its top level: a def, a class,
  an assignment, an import, ``__all__`` or a star import.  Only that first
  name is checked, not the attributes after it.  A module with
  ``__getattr__`` or a ``globals()`` write binds every name.  After
  ``lucid._tensor._metal_views`` was deleted, nine places still named it as
  the live owner of Metal views (LCD-587).  Two kinds of prose are let
  through.  A sentence that says the path was deleted, removed or renamed,
  or that there is no such path, is history or a contract (``_GONE``).  A
  name in ``EXCUSES`` is not a path at all, and each entry gives the reason.
  Where a module has a stub, the stub and the ``.py``'s own bindings are
  its names: ``lucid`` serves its API lazily, so its ``__getattr__`` says
  nothing about which names exist.

Usage::

    python3 tools/check_source_rules.py            # the whole lucid/ and tools/
    python3 tools/check_source_rules.py --staged   # the staged blobs (pre-commit)
    python3 tools/check_source_rules.py PATH ...   # these files

Exit 1 on any violation.  ``lucid/test/unit/test_source_rules.py`` runs the
whole tree; ``.githooks/pre-commit`` runs ``--staged``, so a commit can
neither add a violation nor carry one along in a file it touches.  The
script uses the standard library only.
"""

import argparse
import ast
import functools
import io
import os
import re
import subprocess
import sys
import tokenize
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path, PurePosixPath
from typing import NamedTuple, Self, override

REPO_ROOT = Path(__file__).resolve().parent.parent
LUCID = "lucid"
TOOLS = "tools"
REF_FILE = "lucid/test/_fixtures/ref_framework.py"
TEXT_SUFFIXES = frozenset({".py", ".pyi", ".h", ".hpp", ".cpp", ".mm", ".md"})
PYTHON_SUFFIXES = frozenset({".py", ".pyi"})
# Assembled so that this file passes its own H6 check.
_GPU_WORD = re.compile(r"\b" + "cu" + "da" + r"\b", re.IGNORECASE)
# Calls whose first argument names a module to import.
_IMPORTERS = frozenset({"import_module", "importorskip", "__import__"})
# A ``lucid.`` path in prose.  Not part of a URL, a file name, a longer path,
# a ``<lucid.Node ...>`` repr or a ``lucid.compile.trace/v1`` format tag.
_DOTTED = re.compile(r"(?<![\w./<-])lucid(?:\.[A-Za-z_]\w*)+(?![\w/])")
_SENTENCE_END = re.compile(r"[.!?](?=\s)|\n\s*\n")
# Words by which a sentence says a path is gone, or was never there.
_GONE = re.compile(
    r"\b(?:deleted|removed|retired|renamed|superseded|formerly|took out"
    r"|there is no|there are no|neither|does not exist|do not exist)\b",
    re.IGNORECASE,
)
_CPP_COMMENT = re.compile(r"\s*(?://+|/\*+|\*+/?)")


class Violation(NamedTuple):
    path: str
    line: int
    rule: str
    text: str

    @override
    def __str__(self) -> str:
        return f"{self.path}:{self.line}: [{self.rule}] {self.text}"


def ref_name(root: Path = REPO_ROOT) -> str:
    """The reference framework's package name, from ``_REF_NAME`` in its file."""
    tree = ast.parse((root / REF_FILE).read_text(encoding="utf-8"))
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "_REF_NAME" for t in node.targets
            )
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return node.value.value
    raise LookupError(f"{REF_FILE} assigns no _REF_NAME string")


class _Rules:
    """The checks, bound to the reference framework's name."""

    def __init__(self, ref: str) -> None:
        self.ref_word = re.compile(re.escape(ref), re.IGNORECASE)
        self.ref_module = re.compile(re.escape(ref) + r"\w*", re.IGNORECASE)

    def _is_ref_module(self, name: str) -> bool:
        return self.ref_module.fullmatch(name.split(".")[0]) is not None

    def text(self, path: str, source: str) -> Iterator[Violation]:
        """H5 ``ref-name`` and H6, line by line."""
        for number, line in enumerate(source.splitlines(), 1):
            if path != REF_FILE and self.ref_word.search(line):
                yield Violation(path, number, "ref-name", line.strip()[:100])
            if _GPU_WORD.search(line):
                yield Violation(path, number, "gpu-name", line.strip()[:100])

    def python(self, path: str, source: str, full: bool) -> Iterator[Violation]:
        """H1, and with ``full`` H5 ``ref-import`` and H7, from the syntax tree."""
        try:
            tree = ast.parse(source, filename=path)
        except SyntaxError as e:
            yield Violation(path, e.lineno or 0, "syntax", e.msg)
            return
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                if any(alias.name == "annotations" for alias in node.names):
                    yield Violation(
                        path, node.lineno, "future-annotations", ast.unparse(node)
                    )
            if isinstance(node, ast.ExceptHandler) and isinstance(node.type, ast.Tuple):
                segment = ast.get_source_segment(source, node.type) or ""
                if not (segment.startswith("(") and segment.endswith(")")):
                    yield Violation(
                        path, node.lineno, "except-parens", f"except ({segment}):"
                    )
            if (
                full
                and path != REF_FILE
                and isinstance(node, (ast.Import, ast.ImportFrom, ast.Call))
                and self._imports_ref(node)
            ):
                yield Violation(path, node.lineno, "ref-import", ast.unparse(node))
        if not full:
            return
        for annotation in _annotations(tree):
            for found in _string_types(annotation):
                yield Violation(
                    path, found.lineno, "string-annotation", repr(found.value)
                )

    def _imports_ref(self, node: ast.Import | ast.ImportFrom | ast.Call) -> bool:
        if isinstance(node, ast.Import):
            return any(self._is_ref_module(alias.name) for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            return node.level == 0 and self._is_ref_module(node.module or "")
        if not node.args:
            return False
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        first = node.args[0]
        return (
            name in _IMPORTERS
            and isinstance(first, ast.Constant)
            and isinstance(first.value, str)
            and self._is_ref_module(first.value)
        )


def _string_types(annotation: ast.expr) -> Iterator[ast.Constant]:
    """String constants used as types, skipping ``Literal`` and ``Annotated`` metadata."""
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        yield annotation
        return
    if isinstance(annotation, ast.Subscript):
        base = annotation.value
        name = base.attr if isinstance(base, ast.Attribute) else getattr(base, "id", "")
        if name == "Literal":
            return
        yield from _string_types(base)
        if name == "Annotated" and isinstance(annotation.slice, ast.Tuple):
            yield from _string_types(annotation.slice.elts[0])
            return
        yield from _string_types(annotation.slice)
        return
    for child in ast.iter_child_nodes(annotation):
        if isinstance(child, ast.expr):
            yield from _string_types(child)


def _annotations(tree: ast.AST) -> Iterator[ast.expr]:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            for arg in [
                *args.posonlyargs,
                *args.args,
                *args.kwonlyargs,
                args.vararg,
                args.kwarg,
            ]:
                if arg is not None and arg.annotation is not None:
                    yield arg.annotation
            if node.returns is not None:
                yield node.returns
        elif isinstance(node, ast.AnnAssign):
            yield node.annotation


# ── dead-path ────────────────────────────────────────────────────────────────


class Excuse(NamedTuple):
    """A ``lucid.`` name in prose that is not a path to check, and why."""

    pattern: str  # matched against the whole dotted name
    reason: str
    path: str = ""  # only in this file; "" is every file


EXCUSES: tuple[Excuse, ...] = (
    Excuse(
        r"lucid\.(?:metal\.)?[A-Z]\w*Tensor",
        "a legacy type name: a string Tensor.type() returns and takes back "
        "(lucid/_tensor/_to.py), never an attribute",
    ),
    Excuse(
        r"lucid\.lucid\.bfloat16",
        "the wrong legacy name a bfloat16 tensor used to report, quoted",
        "lucid/test/unit/tensor/test_tensor_type_takes_dtypes_and_legacy_names.py",
    ),
    Excuse(
        r"lucid\.(?:xxx|foo|X)",
        "a placeholder for any top-level name",
    ),
)


class _Binds(NamedTuple):
    """What a module's top level binds: names, star-import sources, or anything."""

    names: frozenset[str]
    stars: tuple[str, ...]
    any_name: bool


class ModuleIndex:
    """The ``lucid`` modules and the names each binds, read without importing.

    A module is a ``.py`` or ``.pyi`` file, or a directory — Python imports a
    directory without ``__init__.py`` as a namespace package.  The engine's
    names are its stub's, ``lucid/_C/engine.pyi``.
    """

    def __init__(self, paths: Iterable[str], read: Callable[[str], str]) -> None:
        self._read = read
        self._files: dict[str, list[str]] = {}
        self._binds: dict[str, _Binds] = {}
        self._resolved: dict[str, bool] = {}
        for path in paths:
            parts = PurePosixPath(path).parts
            if parts[0] != LUCID:
                continue
            for depth in range(1, len(parts)):
                self._files.setdefault(".".join(parts[:depth]), [])
            stem = PurePosixPath(path).stem
            if PurePosixPath(path).suffix in PYTHON_SUFFIXES:
                module = parts[:-1] if stem == "__init__" else (*parts[:-1], stem)
                self._files.setdefault(".".join(module), []).append(path)

    @classmethod
    def of_tree(cls, root: Path) -> Self:
        """Every module under ``root/lucid`` as it is on disk."""
        paths: list[str] = []
        for directory, subdirs, files in os.walk(root / LUCID):
            subdirs[:] = [d for d in subdirs if d != "__pycache__"]
            relative = Path(directory).relative_to(root).as_posix()
            paths.extend(f"{relative}/{name}" for name in files)
        return cls(paths, lambda path: (root / path).read_text(errors="ignore"))

    def resolves(self, dotted: str) -> bool:
        """Whether ``dotted``'s modules exist and the first name after them is bound."""
        if dotted not in self._resolved:
            self._resolved[dotted] = self._resolve(dotted)
        return self._resolved[dotted]

    def _resolve(self, dotted: str) -> bool:
        module, *rest = dotted.split(".")
        while rest and f"{module}.{rest[0]}" in self._files:
            module = f"{module}.{rest.pop(0)}"
        if not rest or (rest[0].startswith("__") and rest[0].endswith("__")):
            return True
        return self._defines(module, rest[0], set())

    def _defines(self, module: str, name: str, seen: set[str]) -> bool:
        if module in seen:
            return False
        seen.add(module)
        binds = self._binds_of(module)
        if binds.any_name or name in binds.names:
            return True
        for star in binds.stars:
            if star.partition(".")[0] != LUCID:
                return True  # another package's names are not indexed
            if star in self._files and self._defines(star, name, seen):
                return True
        return False

    def _binds_of(self, module: str) -> _Binds:
        if module not in self._binds:
            files = [(p, self._read(p)) for p in self._files.get(module, [])]
            self._binds[module] = _module_binds(module, files)
        return self._binds[module]


@functools.cache
def tree_index(root: Path = REPO_ROOT) -> ModuleIndex:
    """The index of ``root``'s tree, built once per process."""
    return ModuleIndex.of_tree(root)


def _module_binds(module: str, files: list[tuple[str, str]]) -> _Binds:
    # Where a stub lists the names a lazy module serves, the ``.py``'s
    # ``__getattr__`` or ``globals()`` write says nothing more.
    has_stub = any(PurePosixPath(path).suffix == ".pyi" for path, _ in files)
    names: set[str] = set()
    stars: list[str] = []
    any_name = False
    for path, source in files:
        try:
            tree = ast.parse(source, filename=path)
        except SyntaxError:
            return _Binds(frozenset(), (), True)  # ``syntax`` reports it
        package = (
            module
            if PurePosixPath(path).stem == "__init__"
            else module.rpartition(".")[0]
        )
        bound: set[str] = set()
        for node in _module_level(tree.body):
            bound.update(_bound_by(node))
            if isinstance(node, ast.ImportFrom) and any(
                a.name == "*" for a in node.names
            ):
                stars.append(_absolute(node, package))
        # A lazy ``__getattr__`` serves the names ``__all__`` lists (PEP 562);
        # without ``__all__`` nothing says which, nor after a ``globals()`` write.
        lazy = "__getattr__" in bound and "__all__" not in bound
        any_name = any_name or (not has_stub and (lazy or _writes_globals(tree)))
        names |= bound
    return _Binds(frozenset(names), tuple(stars), any_name)


def _module_level(statements: list[ast.stmt]) -> Iterator[ast.stmt]:
    """Every statement that runs in the module's namespace, nested blocks included."""
    for node in statements:
        yield node
        for block in _blocks(node):
            yield from _module_level(block)


def _blocks(node: ast.stmt) -> list[list[ast.stmt]]:
    if isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While)):
        return [node.body, node.orelse]
    if isinstance(node, (ast.With, ast.AsyncWith)):
        return [node.body]
    if isinstance(node, (ast.Try, ast.TryStar)):
        return [
            node.body,
            node.orelse,
            node.finalbody,
            *(h.body for h in node.handlers),
        ]
    if isinstance(node, ast.Match):
        return [case.body for case in node.cases]
    return []


def _bound_by(node: ast.stmt) -> Iterator[str]:
    """The names ``node`` binds where it runs, and the names ``__all__`` lists."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        yield node.name
    elif isinstance(node, (ast.Import, ast.ImportFrom)):
        for alias in node.names:
            if alias.name != "*":
                yield alias.asname or alias.name.partition(".")[0]
    targets = _targets(node)
    for target in targets:
        yield from (n.id for n in ast.walk(target) if isinstance(n, ast.Name))
    if (
        isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))
        and node.value is not None
        and any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets)
    ):
        for constant in ast.walk(node.value):
            if isinstance(constant, ast.Constant) and isinstance(constant.value, str):
                yield constant.value


def _targets(node: ast.stmt) -> list[ast.expr]:
    if isinstance(node, ast.Assign):
        return node.targets
    if isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.For, ast.AsyncFor)):
        return [node.target]
    if isinstance(node, ast.TypeAlias):
        return [node.name]
    if isinstance(node, (ast.With, ast.AsyncWith)):
        return [i.optional_vars for i in node.items if i.optional_vars is not None]
    return []


def _absolute(node: ast.ImportFrom, package: str) -> str:
    """The module ``from ... import *`` reads, made absolute."""
    if node.level == 0:
        return node.module or ""
    parts = package.split(".")
    base = parts[: len(parts) - node.level + 1]
    return ".".join([*base, node.module] if node.module else base)


def _writes_globals(tree: ast.Module) -> bool:
    return any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "globals"
        for n in ast.walk(tree)
    )


class _Prose(NamedTuple):
    """A block of prose and the source line it starts on."""

    line: int
    text: str


def _prose(path: str, source: str) -> list[_Prose]:
    suffix = PurePosixPath(path).suffix
    if suffix in PYTHON_SUFFIXES:
        return _python_prose(source)
    if suffix == ".md":
        return [_Prose(1, source)]
    comments: list[tuple[int, str]] = []
    blocks: list[_Prose] = []
    for number, line in enumerate(source.splitlines(), 1):
        if marker := _CPP_COMMENT.match(line):
            comments.append((number, line[marker.end() :]))
        else:
            blocks.append(_Prose(number, line))  # string literals: messages
    return blocks + _runs(comments)


def _python_prose(source: str) -> list[_Prose]:
    """Each run of comment lines, and each bare string — a docstring."""
    comments: list[tuple[int, str]] = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT:
                comments.append((token.start[0], token.string.lstrip("#")))
        tree = ast.parse(source)
    except (tokenize.TokenError, SyntaxError):
        return []  # ``syntax`` reports it
    strings = [
        _Prose(node.value.lineno, node.value.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    ]
    return _runs(comments) + strings


def _runs(lines: list[tuple[int, str]]) -> list[_Prose]:
    """``(line number, text)`` pairs joined into blocks of consecutive lines."""
    blocks: list[_Prose] = []
    texts: list[str] = []
    start = 0
    for number, text in lines:
        if texts and number != start + len(texts):
            blocks.append(_Prose(start, "\n".join(texts)))
            texts = []
        if not texts:
            start = number
        texts.append(text)
    if texts:
        blocks.append(_Prose(start, "\n".join(texts)))
    return blocks


def _told_as_history(text: str, match: re.Match[str]) -> bool:
    """Whether the sentence around ``match`` says the path was deleted or renamed."""
    ends = [end.end() for end in _SENTENCE_END.finditer(text, 0, match.start())]
    after = _SENTENCE_END.search(text, match.end())
    sentence = text[ends[-1] if ends else 0 : after.start() if after else len(text)]
    # The words count, not the paths: ``lucid.x.gone`` says nothing.
    return _GONE.search(_DOTTED.sub(" ", sentence)) is not None


def excuse_for(
    path: str, name: str, excuses: Iterable[Excuse] = EXCUSES
) -> Excuse | None:
    """The excuse that lets ``name`` stand in ``path``'s prose, if any."""
    for excuse in excuses:
        if excuse.path in ("", path) and re.fullmatch(excuse.pattern, name):
            return excuse
    return None


def dead_paths(
    path: str, source: str, index: ModuleIndex, excuses: Iterable[Excuse] = EXCUSES
) -> Iterator[Violation]:
    """Each ``lucid.`` path ``path``'s prose names that does not resolve."""
    excuses = tuple(excuses)
    # Resolve every mention first; only a file with a dead one is worth
    # tokenizing to find which mentions are prose.
    dead = {
        name
        for name in {m.group() for m in _DOTTED.finditer(source)}
        if not index.resolves(name) and excuse_for(path, name, excuses) is None
    }
    if not dead:
        return
    for prose in _prose(path, source):
        for match in _DOTTED.finditer(prose.text):
            name = match.group()
            if name in dead and not _told_as_history(prose.text, match):
                line = prose.line + prose.text.count("\n", 0, match.start())
                yield Violation(path, line, "dead-path", f"{name} does not resolve")


def _scope(path: str) -> str | None:
    """``lucid`` or ``tools`` for a repo-relative path the rules read, else None."""
    top, _, _ = path.partition("/")
    suffix = Path(path).suffix
    if top == LUCID and suffix in TEXT_SUFFIXES:
        return LUCID
    if top == TOOLS and suffix in PYTHON_SUFFIXES:
        return TOOLS
    return None


def check(
    sources: Iterable[tuple[str, str]],
    ref: str,
    index: ModuleIndex | None = None,
    excuses: Iterable[Excuse] = EXCUSES,
) -> list[Violation]:
    """Every violation in ``(repo-relative path, source text)`` pairs.

    ``index`` resolves the ``lucid.`` paths prose names; by default it is
    this checkout's tree.
    """
    rules = _Rules(ref)
    modules = index or tree_index()
    excuses = tuple(excuses)
    found: list[Violation] = []
    for path, source in sources:
        scope = _scope(path)
        if scope == LUCID:
            found.extend(rules.text(path, source))
            found.extend(dead_paths(path, source, modules, excuses))
        if scope is not None and Path(path).suffix in PYTHON_SUFFIXES:
            found.extend(rules.python(path, source, full=scope == LUCID))
    return found


def tree_sources(root: Path = REPO_ROOT) -> Iterator[tuple[str, str]]:
    """Every checked file under ``root/lucid`` and ``root/tools``, as on disk."""
    for top in (LUCID, TOOLS):
        for file in sorted((root / top).rglob("*")):
            path = file.relative_to(root).as_posix()
            if _scope(path) is not None and "__pycache__" not in file.parts:
                yield path, file.read_text(errors="ignore")


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout


def staged_sources(root: Path) -> Iterator[tuple[str, str]]:
    """The staged blob of every added, copied, modified or renamed file.

    ``root`` is any directory of the repository whose index is read — the
    hook's checkout, which is not this script's when a worktree that
    predates the script borrows the main checkout's copy.
    """
    names = _git(root, "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z")
    for path in filter(None, names.split("\0")):
        if _scope(path) is not None:
            yield path, _git(root, "show", f":{path}")


def path_sources(
    paths: Iterable[str], root: Path = REPO_ROOT
) -> Iterator[tuple[str, str]]:
    """The given files, as they are on disk; a file outside ``root`` is skipped."""
    for given in paths:
        file = Path(given).resolve()
        if file.is_relative_to(root):
            yield file.relative_to(root).as_posix(), file.read_text(errors="ignore")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check Hard Rules H1, H5, H6, H7 and dead lucid. paths in prose."
    )
    parser.add_argument("paths", nargs="*", help="files (default: lucid/, tools/)")
    parser.add_argument("--staged", action="store_true", help="check the staged blobs")
    args = parser.parse_args(argv)
    root = REPO_ROOT
    if args.staged:
        # The hook's checkout, which is not this script's when a worktree
        # borrows the main checkout's copy: its modules are the ones to find.
        root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel").strip())
        sources = staged_sources(Path.cwd())
    elif args.paths:
        sources = path_sources(args.paths)
    else:
        sources = tree_sources()
    found = check(sources, ref_name(), tree_index(root))
    for violation in found:
        print(violation, file=sys.stderr)
    if found:
        print(
            f"\n{len(found)} source-rule violation(s) — CLAUDE.md H1 / H5 / H6 / H7"
            " and dead paths.\n"
            "  H5: take the ``ref`` fixture and say 'reference framework'.\n"
            "  dead-path: name what owns it now, or say the path was deleted"
            " (tools/check_source_rules.py EXCUSES for a name that is no path).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
