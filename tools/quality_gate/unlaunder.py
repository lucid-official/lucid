"""Remove the type laundering mypy can prove dead: redundant casts and unused ignores.

    python -m tools.quality_gate.unlaunder --paths lucid/models/text [--check]

mypy is the judge, so nothing is removed on a guess.  Each round runs mypy
over ``lucid/`` with the gate's flags (``config.MYPY_FLAGS``) and a copy of
``mypy.ini`` that drops every section introduced by a ``# transitional
(LCD-260)`` comment — those sections are what keeps the pending packages
green until their sweep lands, and they hide exactly the findings wanted
here.  The ``redundant-cast`` and ``unused-ignore`` findings under
``--paths`` are then applied:

- ``cast(T, e)`` becomes ``e``.  Parentheses are added when ``e`` binds
  looser than its new context or would leave a bare newline, and a cast
  whose removed parts hold a comment is left alone.  An AST comparison
  proves each file's result equals the original with the casts replaced.
- ``# type: ignore`` loses the codes mypy names; an ignore left with no
  code goes with its trailing reason (``# noqa`` / ``# pragma`` tails stay).

mypy reports one finding per line and message, so the next of several
casts on a line only shows up once the first is gone: rounds repeat until
mypy reports nothing more that can be applied.  Touched files then get
``ruff --select F401 --fix`` (a ``cast`` import that became unused) and
black at 88.  Any other mypy finding that appears along the way fails the
run.  ``--check`` edits nothing and exits 1 while anything is left.

Workflow per sweep card: tools/README.md "Quality gate" → "Unlaundering".
"""

import argparse
import ast
import configparser
import importlib.util
import io
import json
import re
import shutil
import sys
import tempfile
import tokenize
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import override

from tools.quality_gate import config
from tools.quality_gate.collectors import run_tool
from tools.quality_gate.collectors.lint import ruff_bin
from tools.quality_gate.core import GateError

# The convention is `# transitional (<issue>)` for any issue, not one id (LCD-315).
TRANSITIONAL_MARKER = re.compile(r"#\s*transitional \(LCD-\d+\)")
CAST = "redundant-cast"
IGNORE = "unused-ignore"
BLACK_LINE_LENGTH = 88  # what lucid/ is formatted with
MAX_ROUNDS = 20

_IGNORE_COMMENT = re.compile(r"#\s*type:\s*ignore(?:\[(?P<codes>[^\]]*)\])?")
_UNUSED_MESSAGE = re.compile(r'Unused "type: ignore(?:\[(?P<codes>[^\]]*)\])?" comment')
# Tails after a removed ignore that are another tool's directive, not its reason.
_DIRECTIVE = re.compile(r"(noqa|pragma|fmt:|pyright:|nosec|isort:|pylint:)", re.IGNORECASE)


@dataclass(frozen=True)
class Finding:
    file: str
    line: int
    column: int  # 0-based, in UTF-8 bytes like ast.col_offset
    code: str
    message: str


# ── mypy ──────────────────────────────────────────────────────────────────


def strip_transitional(ini_text: str) -> str:
    """mypy.ini without the sections a transitional marker introduces.

    The marker must sit in the comment block directly above the section
    header; anything else between them cancels it, so a stray marker cannot
    drop an unrelated section.
    """
    doomed: set[str] = set()
    pending = False
    for raw in ini_text.splitlines():
        line = raw.strip()
        if TRANSITIONAL_MARKER.match(line):
            pending = True
        elif pending and line.startswith("[") and line.endswith("]"):
            doomed.add(line[1:-1].strip())
            pending = False
        elif not line.startswith("#"):
            pending = False
    parser = configparser.ConfigParser(interpolation=None)
    parser.read_string(ini_text)
    for name in doomed:
        parser.remove_section(name)
    out = io.StringIO()
    parser.write(out)
    return out.getvalue()


def run_mypy(root: Path, workdir: Path) -> list[Finding]:
    ini = root / "mypy.ini"
    if not ini.is_file():
        raise GateError(f"{ini}: not found")
    cfg = workdir / "mypy.ini"
    cfg.write_text(strip_transitional(ini.read_text()))
    out = run_tool(
        [
            sys.executable,
            "-m",
            *config.MYPY_COMMAND,
            *config.MYPY_FLAGS,
            f"--config-file={cfg}",
            # Its own cache: the stripped config would otherwise invalidate
            # the regular cache of every transitional module on each run.
            f"--cache-dir={root / '.mypy_cache' / 'unlaunder'}",
            "--output=json",
            "--no-error-summary",
        ],
        root,
        ok_codes=(0, 1),  # 2 is a crash or a usage error
    )
    findings: list[Finding] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GateError(f"mypy: unreadable line {line[:200]!r}") from exc
        if row.get("severity") != "error":
            continue
        findings.append(
            Finding(
                row["file"], row["line"], row["column"], row.get("code") or "misc", row["message"]
            )
        )
    return findings


def under(path: str, prefixes: list[str]) -> bool:
    return any(path == p or path.startswith(p.rstrip("/") + "/") for p in prefixes)


# ── source positions ──────────────────────────────────────────────────────


class _Source:
    """A file's text with (line, byte column) → character offset."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.lines = text.splitlines(keepends=True)
        self.starts = [0]
        for line in self.lines:
            self.starts.append(self.starts[-1] + len(line))

    def offset(self, lineno: int, byte_col: int) -> int:
        line = self.lines[lineno - 1] if lineno <= len(self.lines) else ""
        return self.starts[lineno - 1] + len(line.encode()[:byte_col].decode())

    def char_offset(self, lineno: int, char_col: int) -> int:
        return self.starts[lineno - 1] + char_col

    def span(self, node: ast.expr) -> tuple[int, int]:
        end_line = node.end_lineno if node.end_lineno is not None else node.lineno
        end_col = node.end_col_offset if node.end_col_offset is not None else node.col_offset
        return self.offset(node.lineno, node.col_offset), self.offset(end_line, end_col)


def _tokens(text: str) -> list[tokenize.TokenInfo]:
    return list(tokenize.generate_tokens(io.StringIO(text).readline))


# ── unused ignores ────────────────────────────────────────────────────────


def _rewrite_ignore(comment: str, unused: frozenset[str] | None) -> str | None:
    """The comment without the unused codes; None when it cannot be matched."""
    m = _IGNORE_COMMENT.search(comment)
    if m is None:
        return None
    codes = [c.strip() for c in (m["codes"] or "").split(",") if c.strip()]
    if unused is not None and codes and not unused & set(codes):
        return None
    keep = [c for c in codes if unused is not None and c not in unused]
    if keep:
        return f"{comment[: m.start()]}# type: ignore[{', '.join(keep)}]{comment[m.end() :]}"
    return _drop_ignore(comment, m)


def _drop_ignore(comment: str, m: re.Match[str]) -> str:
    """The comment without the ignore and its reason; other tools' directives stay."""
    head = comment[: m.start()].rstrip()
    tails = [seg.strip() for seg in comment[m.end() :].split("#")[1:]]
    kept = [f"# {seg}" for seg in tails if _DIRECTIVE.match(seg)]
    return "  ".join(([head] if head else []) + kept)


def strip_ignores(text: str, wanted: dict[int, frozenset[str] | None]) -> tuple[str, list[str]]:
    """Apply unused-ignore findings: line → codes to drop (None: the whole ignore)."""
    comments = {t.start[0]: t for t in _tokens(text) if t.type == tokenize.COMMENT}
    lines = text.splitlines(keepends=True)
    skipped: list[str] = []
    for lineno, unused in sorted(wanted.items()):
        tok = comments.get(lineno)
        new = None if tok is None else _rewrite_ignore(tok.string, unused)
        if tok is None or new is None:
            skipped.append(f"{lineno}: no matching `type: ignore` comment")
            continue
        line = lines[lineno - 1]
        code, rest = line[: tok.start[1]], line[tok.end[1] :]
        if new and code.strip():
            code = f"{code.rstrip()}  {new}"
        else:
            code = code + new if new else code.rstrip()
        lines[lineno - 1] = code + rest
    out = "".join(lines)
    if ast.dump(ast.parse(out)) != ast.dump(ast.parse(text)):
        raise GateError("removing ignore comments changed the code (unlaunder bug)")
    return out, skipped


# ── redundant casts ───────────────────────────────────────────────────────

# Expressions that bind at least as tightly as a call, so they can stand in
# for ``cast(...)`` anywhere.
_ATOMS = (
    ast.Name,
    ast.Attribute,
    ast.Call,
    ast.Subscript,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Set,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.JoinedStr,
)
# Never valid bare where a call stood.
_ALWAYS_PARENTHESISED = (ast.NamedExpr, ast.Yield, ast.YieldFrom, ast.Starred)


def _is_cast(node: ast.Call) -> bool:
    f = node.func
    return (isinstance(f, ast.Name) and f.id == "cast") or (
        isinstance(f, ast.Attribute) and f.attr == "cast"
    )


def _cast_value(node: ast.Call) -> ast.expr | None:
    if len(node.args) == 2 and not node.keywords:
        return node.args[1]
    values = [k.value for k in node.keywords if k.arg == "val"]
    return values[0] if len(values) == 1 and len(node.args) + len(node.keywords) == 2 else None


def _in_loose_slot(node: ast.expr, parent: ast.AST | None) -> bool:
    """Whether any expression may replace *node* there without parentheses."""
    slots: list[ast.expr | None] = []
    match parent:
        case ast.Assign() | ast.AnnAssign() | ast.AugAssign() | ast.Return() | ast.Expr():
            slots = [parent.value]
        case ast.keyword():
            slots = [parent.value]
        case ast.Call():
            slots = list(parent.args)
        case ast.List() | ast.Set() | ast.Tuple():
            slots = list(parent.elts)
        case ast.Dict():
            slots = [*parent.keys, *parent.values]
        case ast.Subscript():
            slots = [parent.slice]
    return any(s is node for s in slots)


def _bare_newline(text: str) -> bool:
    """A newline outside brackets — legal inside ``cast(...)``, not on its own."""
    if "\n" not in text:
        return False
    depth = 0
    try:
        for tok in _tokens(text):
            if tok.type == tokenize.OP and tok.string in "([{":
                depth += 1
            elif tok.type == tokenize.OP and tok.string in ")]}":
                depth -= 1
            elif tok.type in (tokenize.NL, tokenize.NEWLINE) and tok.string and depth == 0:
                return True
    except (tokenize.TokenError, SyntaxError):
        return True
    return False


@dataclass
class _Unwrap:
    start: int  # the cast call
    end: int
    inner_start: int  # its value
    inner_end: int
    wrap: bool
    node: ast.Call


class _CastPlanner:
    """Which reported casts can go, and how each is rewritten."""

    def __init__(self, text: str) -> None:
        self.src = _Source(text)
        self.tree = ast.parse(text)
        self.parents = {c: n for n in ast.walk(self.tree) for c in ast.iter_child_nodes(n)}
        self.comments: list[int] = []
        self.depth: dict[int, int] = {}
        depth = 0
        for tok in _tokens(text):
            at = self.src.char_offset(*tok.start)
            self.depth.setdefault(at, depth)
            if tok.type == tokenize.COMMENT:
                self.comments.append(at)
            elif tok.type == tokenize.OP and tok.string in "([{":
                depth += 1
            elif tok.type == tokenize.OP and tok.string in ")]}":
                depth -= 1

    def plan(self, node: ast.Call) -> _Unwrap | str:
        value = _cast_value(node)
        if value is None:
            return "cast with an unexpected signature"
        start, end = self.src.span(node)
        inner_start, inner_end = self.src.span(value)
        if any(start <= c < inner_start or inner_end <= c < end for c in self.comments):
            return "a comment inside the cast would be lost"
        if isinstance(value, _ATOMS):
            wrap = False
        elif isinstance(value, _ALWAYS_PARENTHESISED):
            wrap = True
        else:
            wrap = not _in_loose_slot(node, self.parents.get(node))
        if self.depth.get(start, 0) == 0 and _bare_newline(self.src.text[inner_start:inner_end]):
            wrap = True
        return _Unwrap(start, end, inner_start, inner_end, wrap, node)

    def render(self, edits: list[_Unwrap], *, wrap_all: bool = False) -> str:
        text = self.src.text

        def emit(lo: int, hi: int, todo: list[_Unwrap]) -> str:
            out: list[str] = []
            i, k = lo, 0
            while k < len(todo):
                e = todo[k]
                # Sorted by start, so the casts nested in this one follow it.
                inner = [x for x in todo[k + 1 :] if x.end <= e.end]
                body = emit(e.inner_start, e.inner_end, inner)
                out += [text[i : e.start], f"({body})" if e.wrap or wrap_all else body]
                i, k = e.end, k + 1 + len(inner)
            out.append(text[i:hi])
            return "".join(out)

        return emit(0, len(text), sorted(edits, key=lambda e: (e.start, -e.end)))

    def expected(self, edits: list[_Unwrap]) -> str:
        """The original tree with these casts replaced by their values, dumped."""
        gone = {_where(e.node) for e in edits}

        class _Drop(ast.NodeTransformer):
            @override
            def visit_Call(self, node: ast.Call) -> ast.AST:
                self.generic_visit(node)
                value = _cast_value(node)
                return value if _where(node) in gone and value is not None else node

        return ast.dump(_Drop().visit(ast.parse(self.src.text)))


def _where(node: ast.expr) -> tuple[int, int, int | None, int | None]:
    return node.lineno, node.col_offset, node.end_lineno, node.end_col_offset


def unwrap_casts(text: str, sites: set[tuple[int, int]]) -> tuple[str, int, list[str]]:
    """Apply redundant-cast findings at (line, byte column); returns text, count, skips."""
    planner = _CastPlanner(text)
    edits: list[_Unwrap] = []
    skipped: list[str] = []
    found: set[tuple[int, int]] = set()
    for node in ast.walk(planner.tree):
        site = (getattr(node, "lineno", 0), getattr(node, "col_offset", 0))
        if not isinstance(node, ast.Call) or site not in sites or not _is_cast(node):
            continue
        found.add(site)
        planned = planner.plan(node)
        if isinstance(planned, str):
            skipped.append(f"{site[0]}: {planned}")
        else:
            edits.append(planned)
    skipped += [f"{line}: no cast call at column {col}" for line, col in sorted(sites - found)]
    if not edits:
        return text, 0, skipped
    expected = planner.expected(edits)
    for wrap_all in (False, True):
        out = planner.render(edits, wrap_all=wrap_all)
        try:
            if ast.dump(ast.parse(out)) == expected:
                return out, len(edits), skipped
        except SyntaxError:
            continue
    raise GateError("unwrapping a cast changed the code (unlaunder bug)")


# ── one file, one round ───────────────────────────────────────────────────


def apply_file(path: Path, findings: list[Finding]) -> tuple[int, int, list[str]]:
    """Edit one file; returns (casts removed, ignores edited, skip notes)."""
    ignores: dict[int, frozenset[str] | None] = {}
    for f in findings:
        if f.code == IGNORE:
            m = _UNUSED_MESSAGE.match(f.message)
            codes = m["codes"] if m else None
            ignores[f.line] = (
                None if codes is None else frozenset(c.strip() for c in codes.split(","))
            )
    casts = {(f.line, f.column) for f in findings if f.code == CAST}
    original = path.read_text(encoding="utf-8")
    try:
        text, skipped = strip_ignores(original, ignores) if ignores else (original, [])
        n_ignores = len(ignores) - len(skipped)
        text, n_casts, cast_skips = unwrap_casts(text, casts) if casts else (text, 0, [])
    except GateError as exc:
        raise GateError(f"{path}: {exc}") from exc
    if text != original:
        path.write_text(text, encoding="utf-8")
    return n_casts, n_ignores, skipped + cast_skips


def _black() -> list[str]:
    if importlib.util.find_spec("black") is not None:
        return [sys.executable, "-m", "black"]
    found = shutil.which("black")
    if found is None:
        raise GateError("black is not installed (needed to reformat the touched files)")
    return [found]


def reformat(root: Path, files: list[str]) -> None:
    if not files:
        return
    # 1: a fix left behind (an __init__ re-export) is not ours to remove.
    run_tool([ruff_bin(), "check", "--select", "F401", "--fix", "--quiet", *files], root, (0, 1))
    run_tool([*_black(), "--quiet", f"--line-length={BLACK_LINE_LENGTH}", *files], root)


# ── driver ────────────────────────────────────────────────────────────────


@dataclass
class Outcome:
    remaining: list[Finding]
    touched: set[str]
    skips: dict[str, list[str]]
    new_errors: list[Finding]
    rounds: int
    converged: bool


def _split(findings: list[Finding], prefixes: list[str]) -> tuple[list[Finding], list[Finding]]:
    ours = [f for f in findings if f.code in (CAST, IGNORE) and under(f.file, prefixes)]
    rest = [f for f in findings if f.code not in (CAST, IGNORE)]
    return ours, rest


def _key(f: Finding) -> tuple[str, str]:
    return f.file, f.code


def sweep(root: Path, prefixes: list[str], workdir: Path, max_rounds: int) -> Outcome:
    ours, others = _split(run_mypy(root, workdir), prefixes)
    before = Counter(map(_key, others))
    touched: set[str] = set()
    skips: dict[str, list[str]] = {}
    for rnd in range(1, max_rounds + 1):
        by_file: dict[str, list[Finding]] = defaultdict(list)
        for f in ours:
            by_file[f.file].append(f)
        skips, changed, n_casts, n_ignores = {}, [], 0, 0
        for rel in sorted(by_file):
            casts, ignores, skipped = apply_file(root / rel, by_file[rel])
            n_casts, n_ignores = n_casts + casts, n_ignores + ignores
            if skipped:
                skips[rel] = skipped
            if casts or ignores:
                changed.append(rel)
        print(f"round {rnd}: {n_casts} cast(s) unwrapped, {n_ignores} ignore(s) edited")
        if not changed:
            return Outcome(ours, touched, skips, [], rnd, converged=True)
        reformat(root, changed)
        touched.update(changed)
        ours, others = _split(run_mypy(root, workdir), prefixes)
        grown = Counter(map(_key, others)) - before
        if grown:
            new = [f for f in others if _key(f) in grown]
            return Outcome(ours, touched, skips, new, rnd, converged=False)
        if not ours:
            return Outcome([], touched, {}, [], rnd, converged=True)
    return Outcome(ours, touched, skips, [], max_rounds, converged=False)


def _summary(findings: list[Finding]) -> str:
    counts = Counter(f.code for f in findings)
    return f"{counts[CAST]} redundant-cast, {counts[IGNORE]} unused-ignore"


def _check(root: Path, prefixes: list[str], workdir: Path) -> int:
    ours, _ = _split(run_mypy(root, workdir), prefixes)
    print(f"unlaunder --check {' '.join(prefixes)}: {_summary(ours)}")
    per_file: Counter[str] = Counter(f.file for f in ours)
    for rel, n in sorted(per_file.items()):
        print(f"  {rel}: {n}")
    return 1 if ours else 0


def _report(outcome: Outcome) -> int:
    for rel in sorted(outcome.touched):
        print(f"  edited {rel}")
    for rel, notes in sorted(outcome.skips.items()):
        for note in notes:
            print(f"  left   {rel}:{note}")
    if outcome.new_errors:
        print("unlaunder: these mypy errors appeared during the sweep — review the edits:")
        for f in outcome.new_errors:
            print(f"  {f.file}:{f.line}: {f.message} [{f.code}]")
        return 2
    if not outcome.converged:
        print(f"unlaunder: still finding new sites after {outcome.rounds} rounds")
    print(f"unlaunder: {len(outcome.touched)} file(s) edited; left {_summary(outcome.remaining)}")
    return 1 if outcome.remaining or not outcome.converged else 0


def _prefix(root: Path, given: str) -> str:
    """*given* as mypy names files; a path mypy never reports on would read as 0 left."""
    path = Path(given)
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not path.exists():
        raise ValueError(f"no such path: {given}")
    try:
        rel = path.relative_to(root).as_posix()
    except ValueError:
        raise ValueError(f"{given} is outside {root}") from None
    as_dir = rel + "/"
    if not as_dir.startswith(config.PY_ROOT) or as_dir.startswith(config.PY_EXCLUDE_PREFIXES):
        raise ValueError(f"{given}: mypy is run over {config.PY_ROOT} minus tests only")
    return rel


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m tools.quality_gate.unlaunder",
        description="Remove the redundant casts and unused ignores mypy reports under --paths.",
    )
    p.add_argument("--paths", nargs="+", required=True, metavar="PREFIX", help="e.g. lucid/nn")
    p.add_argument("--check", action="store_true", help="report what is left; edit nothing")
    p.add_argument("--max-rounds", type=int, default=MAX_ROUNDS)
    p.add_argument("--root", default=".", help="repository root (default: cwd)")
    args = p.parse_args(argv)
    root = Path(args.root).resolve()
    try:
        prefixes = [_prefix(root, x) for x in args.paths]
    except ValueError as exc:
        p.error(str(exc))

    workdir = Path(tempfile.mkdtemp(prefix="unlaunder-"))
    try:
        if args.check:
            return _check(root, prefixes, workdir)
        return _report(sweep(root, prefixes, workdir, args.max_rounds))
    except GateError as exc:
        print(f"[unlaunder] cannot proceed: {exc}", file=sys.stderr)
        return 2
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
