#!/usr/bin/env python3
"""Does a range that moves the engine's binary interface bump ``LUCID_ABI_VERSION``?

``lucid/_C/version.h`` states the rule: a member added to or removed from an engine
class, a virtual, a ``LUCID_API`` function (even an additive one) and a binding Python
newly calls all move the interface.  A checkout whose engine was built before such a
change imports cleanly and fails at its first call, unless the counter moved and the
import refuses the old build.  578247379 added five virtuals to ``Node``, changed
``TensorImpl`` and the bindings, and left the counter where it was.

What moved is read from the two ends of the range, never from diff lines, so a move
between files, a reflow or a comment is no change:

- each class or struct in a header: its data members and virtuals, in order, and for
  an exported (``LUCID_API``) type every other member as well;
- free ``LUCID_API`` declarations in headers;
- the names a binding gives Python (``.def*("name"``, ``class_<…>(m, "Name")``,
  ``.value("NAME"``, ``.attr("name") =``), counted across every changed C++ file;
- the names in ``lucid/_C/engine.pyi``.

A range that moves any of them passes when ``LUCID_ABI_VERSION`` rose between its ends,
or when every commit that moves it is excused.  A commit speaks for its own moves: a
reason in its ``ABI:`` trailer, as in ``ABI: none (only a comment in Node.h moved)``,
excuses what that commit moved and nothing else.  A later commit excuses another's moves
only by naming them in its reason: the commit's sha, or every symbol it moved
(``ABI: none (Node::pack_saved is private to the CPU backend)``).  A bare ``ABI: none``
is no reason, most commits carry one by habit, and ``ABI: bump needed`` is no excuse at
all: it keeps the range failing until the counter moves.

Usage::

    python tools/check_abi_bump.py --since <base>          # base..HEAD, as pre-push runs it
    python tools/check_abi_bump.py --range A..B
    python tools/check_abi_bump.py --range A..B --each     # every commit on its own

Exit 0 passes, 1 is a move without a bump, 2 means the range could not be read.
Standard library only: it runs on the push path.
"""

import argparse
import ast
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, NamedTuple

ROOT: Final = Path(__file__).resolve().parents[1]
ENGINE: Final = "lucid/_C"
VERSION_H: Final = "lucid/_C/version.h"
STUB: Final = "lucid/_C/engine.pyi"
# ``git hash-object -t tree /dev/null``: a base with nothing in it.
EMPTY_TREE: Final = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
_HEADERS: Final = (".h", ".hpp")
_SOURCES: Final = (".h", ".hpp", ".cpp", ".mm")
_SHOWN: Final = 12

_ABI_LINE: Final = re.compile(r"^\s*#\s*define\s+LUCID_ABI_VERSION\s+(\d+)\b", re.MULTILINE)
# A reason is words, not a number: ``ABI: none (40)`` is in the history and says nothing.
_EXCUSE: Final = re.compile(r"\s*none\b\W*(?P<reason>.*)", re.IGNORECASE | re.DOTALL)
# A commit that says the counter has to move: no reason, its own or another's, undoes that.
_BUMP_NEEDED: Final = re.compile(r"\s*bump\s+needed\b", re.IGNORECASE)
_WORD: Final = re.compile(r"[A-Za-z]{2,}")
_SHA: Final = re.compile(r"\b[0-9a-f]{7,40}\b")
# A symbol shorter than this is in too much prose to be named by it; a sha names it instead.
_MIN_NAME: Final = 3
# Raw strings first, so that a quote or a brace inside one ends nothing.
_NOISE: Final = re.compile(
    r'R"(?P<delim>[^(\s"]{0,16})\((?s:.*?)\)(?P=delim)"'
    r"|//[^\n]*|/\*(?s:.*?)\*/"
    r'|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\''
)
_COMMENT: Final = re.compile(r"//[^\n]*|/\*(?s:.*?)\*/")
_DIRECTIVE: Final = re.compile(r"^[ \t]*#(?:[^\n]*\\\n)*[^\n]*", re.MULTILINE)
_TOKEN: Final = re.compile(r"[A-Za-z_]\w*|\d[\w.']*|::|->|\.\.\.|[^\s\w]")
_BINDINGS: Final = (
    ("def", re.compile(r'\.def(?:_\w+)?\s*\(\s*"([^"]+)"')),
    ("init", re.compile(r"\.def\s*\(\s*(?:py|pybind11)::init\s*<([^;]*?)>\s*\(")),
    # A temporary or a named one: ``class_<T>(m, "T")``, ``class_<T> t(m, "T")``.
    (
        "class",
        re.compile(
            r'\b(?:class_|enum_|exception)\s*<[^;]*?>\s*(?:\w+\s*)?\(\s*\w+\s*,\s*"([^"]+)"'
        ),
    ),
    ("value", re.compile(r'\.value\s*\(\s*"([^"]+)"')),
    ("attr", re.compile(r'\.attr\s*\(\s*"([^"]+)"\s*\)\s*=[^=]')),
)
_NAME: Final = re.compile(r"[A-Za-z_]\w*")
# A last word that is a type, not a parameter's name: ``unsigned int`` names nothing.
_BUILTIN: Final = frozenset(
    "int long short char unsigned signed float double bool void const volatile auto".split()
)
_OPEN: Final = ("(", "[")
_CLOSE: Final = (")", "]")
_ACCESS: Final = frozenset({"public", "private", "protected"})
_NOT_MEMBERS: Final = frozenset({"friend", "using", "typedef", "static_assert", "enum"})
_TYPE_KEYS: Final = frozenset({"class", "struct", "union"})
# A word whose parentheses are no parameter list: ``struct alignas(16) X``, ``decltype(y) z_``.
_SPECIFIERS: Final = frozenset({"alignas", "decltype", "__attribute__", "__declspec"})
_NAME_SKIP: Final = frozenset({"LUCID_API", "LUCID_INTERNAL", "final", "::", *_SPECIFIERS})
_SCOPE_SKIP: Final = frozenset({"namespace", "inline", "extern"})


class Unreadable(RuntimeError):
    """Git could not answer, so nothing is known about the range."""


def _git(*args: str) -> str:
    done = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace"
    )
    if done.returncode != 0:
        raise Unreadable(f"git {args[0]}: {done.stderr.strip()}")
    return done.stdout


def _blob(rev: str, path: str) -> str:
    """``path`` at ``rev``; a file that is not there reads as empty."""
    done = subprocess.run(
        ["git", "cat-file", "blob", f"{rev}:{path}"],
        cwd=ROOT,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    return done.stdout if done.returncode == 0 else ""


# ── what one end of a range offers across the boundary ───────────────────────


@dataclass
class _Type:
    exported: bool
    layout: list[str] = field(default_factory=list)  # data members and virtuals, in order
    calls: list[str] = field(default_factory=list)  # every other member: symbols when exported


@dataclass
class Interface:
    types: dict[str, _Type] = field(default_factory=dict)
    exports: Counter[str] = field(default_factory=Counter)
    bindings: Counter[str] = field(default_factory=Counter)
    stub: frozenset[str] = frozenset()
    aliases: dict[str, str] = field(default_factory=dict)  # ``using X = T``: X's T


def _depth0(tokens: Sequence[str]) -> list[tuple[int, str]]:
    """Tokens outside parentheses and template brackets (an opening one included)."""
    out: list[tuple[int, str]] = []
    parens = angles = 0
    after_operator = False
    for n, tok in enumerate(tokens):
        top = parens == 0 and angles == 0
        if tok in _OPEN:
            out += [(n, tok)] if top and not (n and tokens[n - 1] in _SPECIFIERS) else []
            parens += 1
        elif tok in _CLOSE:
            parens = max(parens - 1, 0)
        elif parens == 0 and tok == "<" and not after_operator:
            angles += 1
        elif parens == 0 and tok == ">" and angles and not after_operator:
            angles -= 1
        elif top:
            out.append((n, tok))
        after_operator = tok == "operator" or (after_operator and tok in ("<", ">", "="))
    return out


def _is_function(tokens: Sequence[str]) -> bool:
    """A declarator with a parameter list before any initializer."""
    for _, tok in _depth0(tokens):
        if tok in ("operator", "("):
            return True
        if tok == "=":
            return False
    return False


def _parameter(tokens: Sequence[str]) -> str:
    """One parameter as its type: a name and a default are no part of the interface."""
    cut = next((n for n, tok in _depth0(tokens) if tok == "="), len(tokens))
    kept = list(tokens[:cut])
    named = len(kept) >= 2 and _NAME.fullmatch(kept[-1]) and kept[-1] not in _BUILTIN
    return " ".join(kept[:-1] if named and kept[-2] != "::" else kept)


def _signature(tokens: Sequence[str]) -> str:
    """A function declaration with its parameters reduced to their types."""
    top = _depth0(tokens)
    opens = [
        n for k, (n, t) in enumerate(top) if t == "(" and not (k and top[k - 1][1] == "operator")
    ]
    if not opens:
        return " ".join(tokens)
    params: list[str] = []
    part: list[str] = []
    depth, end = 0, len(tokens)
    for n in range(opens[0] + 1, len(tokens)):
        after = tokens[n + 1] if n + 1 < len(tokens) else ""
        shift = tokens[n] == "<" and "<" in (tokens[n - 1], after)  # ``a << b``: no template
        depth += (tokens[n] in _OPEN or (tokens[n] == "<" and not shift)) - (
            tokens[n] in _CLOSE or tokens[n] == ">"
        )
        if depth < 0 or (depth == 0 and tokens[n] == ","):
            params, part = [*params, _parameter(part)], []
        else:
            part.append(tokens[n])
        if depth < 0:
            end = n
            break
    listed = ", ".join(p for p in params if p)
    return " ".join(filter(None, (*tokens[: opens[0]], "(", listed, ")", *tokens[end + 1 :])))


def _strip_access(tokens: list[str]) -> list[str]:
    while len(tokens) >= 2 and tokens[0] in _ACCESS and tokens[1] == ":":
        tokens = tokens[2:]
    return tokens


def _type_head(head: Sequence[str]) -> tuple[str, bool] | None:
    """The name a ``class``/``struct``/``union`` head opens, and whether it is exported."""
    top = [tok for _, tok in _depth0(head)]
    keys = [n for n, tok in enumerate(top) if tok in _TYPE_KEYS]
    if not keys or {"enum", "(", "="} & set(top):
        return None
    rest = [tok for tok in top[keys[0] + 1 :] if tok not in _NAME_SKIP]
    name = next((tok for tok in rest if _NAME.fullmatch(tok)), "<anonymous>")
    return name, "LUCID_API" in head


class _Scanner:
    """Declarations by scope: namespaces nest, a type records its members, a body is skipped."""

    def __init__(self, text: str, into: Interface) -> None:
        self.tokens = _TOKEN.findall(_DIRECTIVE.sub(" ", _NOISE.sub(" ", text)))
        self.into = into

    def skip_group(self, i: int) -> int:
        """``i`` is at an opening brace; the index after its match."""
        depth = 0
        for n in range(i, len(self.tokens)):
            depth += {"{": 1, "}": -1}.get(self.tokens[n], 0)
            if depth == 0:
                return n + 1
        return len(self.tokens)

    def scope(self, i: int, path: str, owner: _Type | None) -> int:
        """Statements from ``i`` to the brace that closes this scope."""
        stmt: list[str] = []
        parens = 0  # a brace inside parentheses is a default argument, not a scope
        while i < len(self.tokens):
            tok = self.tokens[i]
            parens = max(parens + (tok == "(") - (tok == ")"), 0)
            if parens == 0 and tok == "}":
                return i + 1
            if parens == 0 and tok == ";":
                self.statement(stmt, path, owner)
                stmt, i = [], i + 1
            elif parens == 0 and tok == "{":
                stmt, i = self.brace(_strip_access(stmt), i, path, owner)
            else:
                stmt.append(tok)
                i += 1
        return i

    def brace(
        self, head: list[str], i: int, path: str, owner: _Type | None
    ) -> tuple[list[str], int]:
        """What a ``{`` opens; the statement still open after it, and where to go on."""
        top = [tok for _, tok in _depth0(head)]
        if "namespace" in top or top == ["extern"]:
            name = "".join(tok for tok in top if tok not in _SCOPE_SKIP)
            return [], self.scope(i + 1, f"{path}{name}::" if name else path, None)
        opened = _type_head(head)
        if opened is not None:
            key = f"{path}{opened[0]}"
            twins = sum(k == key or k.startswith(f"{key}#") for k in self.into.types)
            key += f"#{twins}" if twins else ""  # a specialization, or a name in two headers
            made = self.into.types[key] = _Type(opened[1])
            i = self.scope(i + 1, f"{key}::", made)
            while i < len(self.tokens) and self.tokens[i] != ";":  # declarators after the body
                i += 1
            return [], i + 1
        i = self.skip_group(i)
        if "enum" in top or not _is_function(head):
            return [*head, "{}"], i  # an initializer: the statement runs on to its ``;``
        while i < len(self.tokens) and self.tokens[i] in (",", "{"):  # a constructor's inits
            while i < len(self.tokens) and self.tokens[i] != "{":
                i += 1
            i = self.skip_group(i)
        i += int(i < len(self.tokens) and self.tokens[i] == ";")
        colon = next((n for n, tok in _depth0(head) if tok == ":"), len(head))
        self.statement(head[:colon], path, owner)
        return [], i

    def statement(self, stmt: list[str], path: str, owner: _Type | None) -> None:
        stmt = _strip_access(stmt)
        if stmt[:1] == ["using"] and stmt[2:3] == ["="]:  # a layout reads the type through it
            self.into.aliases[f"{path}{stmt[1]}"] = " ".join(stmt[3:])
        if not stmt or stmt[0] in _NOT_MEMBERS:
            return
        function = _is_function(stmt)
        if stmt[0] in _TYPE_KEYS and not function:
            return  # a forward declaration
        if function:
            text = _signature(stmt)
        else:  # a value: what it starts as is no part of the layout
            cut = next((n for n, tok in _depth0(stmt) if tok in ("=", "{}")), len(stmt))
            text = " ".join(stmt[:cut])
        if owner is None:
            self.into.exports.update([text] if "LUCID_API" in stmt else [])
        elif function:
            virtual = {"virtual", "override"} & set(stmt) or stmt[-1] == "final"
            (owner.layout if virtual else owner.calls).append(text)
        elif "static" not in stmt:
            owner.layout.append(text)
        elif not {"constexpr", "inline"} & set(stmt):  # either needs no symbol
            owner.calls.append(text)


def _stub_names(text: str) -> frozenset[str]:
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        raise Unreadable(f"{STUB} does not parse: {exc}") from exc
    names: set[str] = set()
    pending: list[tuple[Sequence[ast.stmt], str]] = [(tree.body, "")]
    while pending:
        body, prefix = pending.pop()
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(f"{prefix}{node.name}")
            if isinstance(node, ast.ClassDef):
                pending.append((node.body, f"{prefix}{node.name}."))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(f"{prefix}{node.target.id}")
            elif isinstance(node, ast.Assign):
                names.update(f"{prefix}{t.id}" for t in node.targets if isinstance(t, ast.Name))
    return frozenset(names)


def read_interface(rev: str, paths: Iterable[str]) -> Interface:
    """What ``paths`` offer across the boundary at ``rev``."""
    found = Interface()
    for path in paths:
        text = _blob(rev, path)
        if path == STUB:
            found.stub = _stub_names(text)
        if path.endswith(_HEADERS):
            _Scanner(text, found).scope(0, "", None)
        if path.endswith(_SOURCES):
            plain = _DIRECTIVE.sub(" ", _COMMENT.sub(" ", text))
            for kind, pattern in _BINDINGS:
                found.bindings.update(
                    f"{kind} {' '.join(m.split())}" for m in pattern.findall(plain)
                )
    return found


class Move(NamedTuple):
    """One line of how the interface moved, and the name a reason can speak for it by."""

    subject: str
    text: str


def _counted(old: Iterable[str], new: Iterable[str]) -> list[tuple[str, str]]:
    """``(sign, key)`` for what one end has more of than the other, removals first."""
    before, after = Counter(old), Counter(new)
    return [("-", k) for k in sorted(before - after)] + [("+", k) for k in sorted(after - before)]


def _leaf(qualified: str) -> str:
    """``lucid::Node#1`` -> ``Node``: the name a person writes for it."""
    return qualified.rsplit("::", 1)[-1].partition("#")[0]


def _declared(declaration: str) -> str:
    """The name a free declaration introduces: the word before its parameter list."""
    head = declaration.partition(" (")[0]
    return next(reversed(_NAME.findall(head)), declaration)


def _type_moves(name: str, before: _Type | None, after: _Type | None) -> list[Move]:
    either = before or after
    leaf = _leaf(name)
    if before is None or after is None:
        gone = "removed" if after is None else "added"
        return [Move(leaf, f"{name} (exported) {gone}")] if either and either.exported else []
    tag = "exported" if before.exported or after.exported else "internal"
    found = [
        Move(leaf, f"{name} ({tag}) layout {sign} {key}")
        for sign, key in _counted(before.layout, after.layout)
    ]
    if not found and before.layout != after.layout:
        found.append(Move(leaf, f"{name} ({tag}) layout: members reordered"))
    if tag == "exported":
        found += [
            Move(leaf, f"{name} (exported) {sign} {key}")
            for sign, key in _counted(before.calls, after.calls)
        ]
    return found


def _bound_name(key: str) -> str:
    """``def reset`` -> ``reset``; a constructor binding has no name of its own."""
    kind, _, name = key.partition(" ")
    return "init" if kind == "init" else name


def moves(base: str, tip: str) -> list[Move]:
    """How the interface moved from ``base`` to ``tip``, a line each; empty when it did not."""
    names = _git("diff", "--name-only", "--no-renames", "-z", base, tip, "--", ENGINE)
    paths = sorted(p for p in names.split("\0") if p.endswith(_SOURCES) or p == STUB)
    if not paths:
        return []
    old, new = read_interface(base, paths), read_interface(tip, paths)
    found = [
        move
        for name in sorted(old.types.keys() | new.types.keys())
        for move in _type_moves(name, old.types.get(name), new.types.get(name))
    ]
    found += [
        Move(_declared(key), f"LUCID_API {sign} {key}")
        for sign, key in _counted(old.exports.elements(), new.exports.elements())
    ]
    found += [
        Move(_bound_name(key), f"binding {sign} {key}")
        for sign, key in _counted(old.bindings.elements(), new.bindings.elements())
    ]
    # A new alias or one gone moves nothing by itself; one that means another type does.
    meant = sorted(
        k for k in old.aliases.keys() & new.aliases.keys() if old.aliases[k] != new.aliases[k]
    )
    found += [Move(_leaf(k), f"alias {k}: {old.aliases[k]} -> {new.aliases[k]}") for k in meant]
    found += [
        Move(key.rsplit(".", 1)[-1], f"engine.pyi {sign} {key}")
        for sign, key in _counted(old.stub, new.stub)
    ]
    return found


# ── the range ────────────────────────────────────────────────────────────────


def abi_version(rev: str) -> int | None:
    found = _ABI_LINE.search(_blob(rev, VERSION_H))
    return int(found.group(1)) if found else None


@dataclass(frozen=True)
class Commit:
    sha: str
    parent: str
    subject: str
    reason: str  # what its ``ABI: none`` trailer gives as the reason; empty when nothing
    needs_bump: bool = False  # its trailer says ``ABI: bump needed``: nothing excuses it

    @property
    def line(self) -> str:
        return f"{self.sha[:9]} {self.subject}"


def _reason(trailers: str) -> str:
    for value in trailers.split("\x1e"):
        excuse = _EXCUSE.fullmatch(value.strip())
        if excuse and len(_WORD.findall(excuse["reason"])) >= 2:
            return " ".join(excuse["reason"].split())
    return ""


def _needs_bump(trailers: str) -> bool:
    return any(_BUMP_NEEDED.match(value) for value in trailers.split("\x1e"))


def commits(spec: str, *options: str) -> list[Commit]:
    """Oldest first; a root commit's parent is the empty tree."""
    fields = "%H%x1f%P%x1f%s%x1f%(trailers:key=ABI,valueonly,separator=%x1e)%x00"
    text = _git("log", "--reverse", f"--format={fields}", *options, "--end-of-options", spec)
    found = []
    for record in filter(str.strip, text.split("\0")):
        sha, parents, subject, trailers = record.strip("\n").split("\x1f")
        parent = (parents.split() or [EMPTY_TREE])[0]
        found.append(Commit(sha, parent, subject, _reason(trailers), _needs_bump(trailers)))
    return found


def _bumped(base: str, tip: str) -> tuple[int | None, int | None, bool]:
    before, after = abi_version(base), abi_version(tip)
    return before, after, after is not None and (before is None or after > before)


def _speaks_for(excuse: Commit, other: Commit, made: Sequence[Move]) -> bool:
    """Does ``excuse``'s reason name ``other``, by its sha or by every symbol it moved?"""
    if any(other.sha.startswith(sha) for sha in _SHA.findall(excuse.reason)):
        return True
    named = set(_NAME.findall(excuse.reason))
    return all(len(move.subject) >= _MIN_NAME and move.subject in named for move in made)


def _settle(
    history: Sequence[Commit], found: Sequence[Move]
) -> tuple[list[tuple[Commit, list[Move]]], list[tuple[Commit, list[Commit]]]]:
    """Which commits still hold a move nobody excused, and which commits excused something.

    A commit's reason speaks for its own moves.  It speaks for an earlier commit's only by
    naming it.  A commit that says a bump is needed is never excused.  A move whose symbol
    is not among those that moved end to end was undone by a later commit and holds nothing.
    """
    standing = {move.subject for move in found}
    held: list[tuple[Commit, list[Move]]] = []
    excused: list[tuple[Commit, list[Commit]]] = []
    for commit in history:
        made = [move for move in moves(commit.parent, commit.sha) if move.subject in standing]
        if commit.needs_bump or not commit.reason:
            if made or commit.needs_bump:
                held.append((commit, made))
            continue
        covered = [
            (other, theirs)
            for other, theirs in held
            if not other.needs_bump and _speaks_for(commit, other, theirs)
        ]
        held = [entry for entry in held if entry not in covered]
        if made or covered:
            excused.append((commit, [other for other, _ in covered]))
    return held, excused


def judge(base: str, tip: str) -> int:
    if base == EMPTY_TREE:
        print("abi-bump: the remote holds nothing yet, so nothing there can have moved")
        return 0
    found = moves(base, tip)
    if not found:
        return 0
    before, after, bumped = _bumped(base, tip)
    if bumped:
        print(f"abi-bump: the interface moved and LUCID_ABI_VERSION went {before} -> {after}")
        return 0
    held, excused = _settle(commits(f"{base}..{tip}", "--first-parent"), found)
    if excused and not held:
        for commit, covering in excused:
            # "note: " is what a passing push still prints: the trailer is the excuse's only record.
            also = "".join(f", and {c.sha[:9]}" for c in covering)
            print(f"note: abi-bump excused by {commit.line}{also}: ABI: none ({commit.reason})")
        return 0
    print(f"abi-bump: lucid/_C moved its binary interface and LUCID_ABI_VERSION is still {after}:")
    shown = [
        *(move.text for move in found[:_SHOWN]),
        *([f"... and {len(found) - _SHOWN} more"] if found[_SHOWN:] else []),
    ]
    for commit, made in held:
        what = ", ".join(sorted({move.subject for move in made})) or "nothing it can name"
        says = "; it says `ABI: bump needed`" if commit.needs_bump else ""
        shown.append(f"moved by {commit.line} [{what}]{says}")
    print("".join(f"  {line}\n" for line in shown), end="")
    print(
        f"Bump LUCID_ABI_VERSION in {VERSION_H} in this range.  If a commit's move changes no\n"
        "binary interface, say why in that commit's own trailer: `ABI: none (<why>)`.  A later\n"
        "commit excuses another's move only by naming it in its reason: the other commit's sha,\n"
        "or every symbol it moved listed in brackets above.  `ABI: bump needed` is excused by\n"
        "nothing but the bump."
    )
    return 1


def judge_each(base: str, tip: str) -> int:
    """Every non-merge commit as its own range: the backtest's view."""
    worst = 0
    for commit in commits(f"{base}..{tip}", "--no-merges"):
        found = moves(commit.parent, commit.sha)
        if not found:
            continue
        before, after, bumped = _bumped(commit.parent, commit.sha)
        verdict = "bumped" if bumped else "excused" if commit.reason else "MISSING"
        worst = max(worst, int(verdict == "MISSING"))
        print(f"{verdict:<8s}{commit.line}  [{len(found)} move(s); ABI {before} -> {after}]")
        print("".join(f"          {move.text}\n" for move in found[:_SHOWN]), end="")
    return worst


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="check_abi_bump", description=__doc__.split("\n")[0])
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--range", help="A..B")
    where.add_argument("--since", help="the range's base; its tip is HEAD")
    parser.add_argument("--each", action="store_true", help="judge every commit on its own")
    args = parser.parse_args(argv)
    base, dots, tip = (args.range or f"{args.since}..HEAD").partition("..")
    try:
        if not (base and dots and tip):
            raise Unreadable(f"{args.range!r} is not a range: A..B")
        if base != EMPTY_TREE:
            base = _git("rev-parse", "--verify", "--end-of-options", f"{base}^{{commit}}").strip()
        tip = _git("rev-parse", "--verify", "--end-of-options", f"{tip}^{{commit}}").strip()
        return (judge_each if args.each else judge)(base, tip)
    except Unreadable as exc:
        print(f"abi-bump: cannot measure: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
