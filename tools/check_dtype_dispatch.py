#!/usr/bin/env python3
"""
tools/check_dtype_dispatch.py — every engine dtype dispatch names what it
handles and refuses the rest.

The defect class (CHA-225)
--------------------------
A kernel branches on its dtype, names the dtypes it has code for, and
lets a bare ``else`` take everything else::

    if (dt == Dtype::F32) { ...float*... }
    else                   { ...double*... }   // and F16, BF16, C64, ...

Any dtype the chain does not name is then read and written at the width
of the branch the ``else`` was written for.  CPU ``embedding_backward``
did exactly this for half: a buffer of 2-byte elements was walked in
8-byte ``double`` lanes, which wrote into rows no index named and, on the
last row, past the end of the allocation.

The companion pattern is a hand-written "is it a float" test spelled
``dt == Dtype::F32 || dt == Dtype::F64``.  It reads as a float check and
answers ``false`` for float16 and bfloat16; that is how ``sort`` and
``topk`` stopped recording a gradient for half inputs.

The owner
---------
Each kernel's own dispatch.  It ends in an explicit refusal
(``ErrorBuilder(...).not_implemented(...)`` or a ``throw``), so a dtype
nobody wrote code for raises instead of being reinterpreted.  On the CPU,
half widens to float32 at the door and narrows on the way out
(``detail::as_f32`` / ``detail::back_to_f16``).  Whether a dtype is
floating point or complex is answered by ``core/Dtype.h``'s
``is_floating_point`` / ``is_complex``, not by a list.

Rules
-----
``bare-else``
    An ``if`` / ``else if`` chain whose conditions are all equality tests
    of one dtype expression against ``Dtype::`` constants, naming F32 or
    F64, that ends in a bare ``else`` which is not a refusal.  The
    early-return spelling is the same chain: ``if (dt == Dtype::F32)
    return ...;`` followed by a statement that reads memory at a fixed
    float width (``reinterpret_cast<double*>`` and the like).
    A chain is fine when an earlier guard in the same function already
    refused every dtype but the ones it names plus one, for example
    ``if (dt != Dtype::F32 && dt != Dtype::F64) not_implemented(...)``
    ahead of ``if (dt == Dtype::F32) ... else ...``.
``float-list``
    An ``||`` of equality tests that names both F32 and F64 but not both
    half formats: a float test that misses dtypes.  An ``if`` with that
    condition whose ``else`` (or next statement) is a refusal is a
    capability guard, not a float test, and passes.  The negated refusal
    form ``dt != Dtype::F32 && dt != Dtype::F64`` passes too.

``half-pair``
    A ``switch`` whose own ``case Dtype::...`` labels name F16 but not
    BF16.  The two 16-bit floats share every code path on the CPU (widen
    to float32, or move the sixteen bits as they are), so a dispatch that
    wrote the float16 case and stopped refuses bfloat16 in its default
    while float16 works: bfloat16 training died in gradient accumulation
    and in the backward of every transpose that way (LCD-248).  Name both,
    or take the pair through ``is_half_float`` before the switch.

Known sites
-----------
Sites that predate the rule are listed in ``_KNOWN`` by
``(file, function, rule)`` with their hit count and the reason each one
is not the defect today.  A function with more hits than its entry fails
as new; one with fewer fails as a stale entry, so the list only shrinks.

Usage
-----
    python tools/check_dtype_dispatch.py            # check lucid/_C
    python tools/check_dtype_dispatch.py --list     # also print known hits
    python tools/check_dtype_dispatch.py PATH ...   # check these files only
    python tools/check_dtype_dispatch.py --root DIR # check another checkout
    python tools/check_dtype_dispatch.py --self-test
"""

import argparse
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENGINE = Path("lucid") / "_C"
SUFFIXES = (".h", ".hpp", ".cpp", ".mm")

#: The real floating dtypes, in step with ``core/Dtype.h::is_floating_point``.
#: That file is where they may be spelled out, so the float-list rule
#: skips it.
FLOATS = frozenset({"F16", "BF16", "F32", "F64"})
HALVES = frozenset({"F16", "BF16"})
DEFINING_FILE = "lucid/_C/core/Dtype.h"

_CPU = "lucid/_C/backend/cpu/CpuBackend.h"
_GPU = "lucid/_C/backend/gpu/GpuBackend.h"
_CPU_METHODS = "lucid/_C/backend/cpu/methods/"
_GPU_METHODS = "lucid/_C/backend/gpu/methods/"


def _known_unit(rel: str) -> str:
    """Backend header and its out-of-line method TUs share one ``_KNOWN`` unit."""
    if rel == _CPU or (rel.startswith(_CPU_METHODS) and rel.endswith(".cpp")):
        return _CPU
    if rel == _GPU or (rel.startswith(_GPU_METHODS) and rel.endswith(".cpp")):
        return _GPU
    return rel


#: Sites that predate the rule: ``(file, function, rule) -> (hits, reason)``.
#: Each reason says why the site is not the defect today and what retires
#: it.  Do not add an entry to get a new kernel through; give its dispatch
#: a refusal instead.
_KNOWN: dict[tuple[str, str, str], tuple[int, str]] = {
    (_CPU, "gather_backward", "float-list"): (
        1,
        "typed fast-path guard `rows && (dt == F32 || dt == F64)`: half "
        "was widened at the top of the function and the general path "
        "below ends in a refusal. Retire: name the fast path's dtypes in "
        "its own dispatch and refuse there.",
    ),
    (_CPU, "gather_backward", "bare-else"): (
        1,
        "the `if (F32) float else double` inside that fast-path guard, so "
        "only F64 reaches the else. Retire: `else if (dt == Dtype::F64)` "
        "plus not_implemented.",
    ),
    (_CPU, "ctc_loss_forward", "bare-else"): (
        2,
        "the `lp` reader and the loss writer use `double` for anything "
        "that is not F32; the op's float_only validator admits F32/F64 "
        "only (half raises before the kernel). Retire: refuse non-F32/F64 "
        "at the top of the kernel.",
    ),
    (_CPU, "ctc_loss_backward", "bare-else"): (
        2,
        "the `read` / `put` lambdas, as ctc_loss_forward; same validator. "
        "Retire with the same refusal.",
    ),
    (_CPU, "project_lane", "bare-else"): (
        1,
        "`real_lane_of(cs.dtype)` is F32 or F64 for the complex storages "
        "complex_real / complex_imag receive, and the ops refuse a real "
        "input before the kernel. Retire: refuse !is_complex(cs.dtype).",
    ),
    ("lucid/_C/ops/bfunc/_Opmath.h", "is_wide_scalar", "float-list"): (
        1,
        "not a float test: it asks whether a 0-d operand is a float32 (or "
        "CPU float64) scalar a half tensor may compute against; half "
        "operands are the other side of the pair. Retire: spell it "
        "`!is_half_float(dt) && is_floating_point(dt)` plus the device rule.",
    ),
    ("lucid/_C/nn/Interpolate.cpp", "resample_matrix", "bare-else"): (
        1,
        "not a reinterpretation: the else only picks float precision for "
        "the arithmetic, the buffer is always F64 and is cast to `dt` "
        "afterwards. Retire: `else if (is_floating_point(dt))` plus a "
        "refusal.",
    ),
    ("lucid/_C/ops/diffeq/RkErrorNorm.cpp", "read_scalar", "bare-else"): (
        1,
        "reads a one-element buffer as double for F64 and float otherwise; "
        "its only caller runs after rk_error_norm's F32/F64 refusal. "
        "Retire: give read_scalar the refusal itself.",
    ),
}


# ── source preparation ───────────────────────────────────────────────────

#: The lexemes whose insides are not code: comments, string and character
#: literals (raw strings included) and preprocessor lines.  A character
#: literal must not follow a word character, so a digit separator
#: (``1'000``) is left alone.
_NOT_CODE = re.compile(
    r"//[^\n]*"
    r"|/\*.*?\*/"
    r'|(?<!\w)R"(?P<delim>[^(\s"]{0,16})\(.*?\)(?P=delim)"'
    r'|"(?:\\.|[^"\\\n])*"'
    r"|(?<!\w)'(?:\\.|[^'\\\n])+'"
    r"|^[ \t]*#(?:\\\n|[^\n])*",
    re.DOTALL | re.MULTILINE,
)


def _blank(m: re.Match[str]) -> str:
    """A lexeme with all but its newlines (and a literal's quotes) blanked."""
    text = m.group(0)
    body = "".join("\n" if c == "\n" else " " for c in text)
    if text[0] in "\"'R":
        return text[0] + body[1:-1] + text[-1]
    return body


def strip_source(text: str) -> str:
    """Comments, literals' contents and preprocessor lines blanked out.

    Offsets and line numbers are preserved, so a position in the result
    is the same position in the file.
    """
    return _NOT_CODE.sub(_blank, text)


# ── a statement walker, just deep enough for if / else chains ────────────

_OPEN = {"(": ")", "[": "]", "{": "}"}
_WORD = re.compile(r"[A-Za-z_]\w*")


def _skip_ws(s: str, i: int) -> int:
    while i < len(s) and s[i].isspace():
        i += 1
    return i


def _match(s: str, i: int) -> int:
    """Index of the bracket closing the one at ``s[i]``."""
    stack = [_OPEN[s[i]]]
    j = i + 1
    while j < len(s) and stack:
        c = s[j]
        if c in _OPEN:
            stack.append(_OPEN[c])
        elif c in ")]}":
            if c != stack[-1]:
                # Unbalanced (a macro trick); give up on this construct.
                return len(s) - 1
            stack.pop()
        j += 1
    return j - 1


def _word_at(s: str, i: int) -> str:
    m = _WORD.match(s, i)
    return m.group(0) if m else ""


def _statement_end(s: str, i: int) -> int:
    """Index just past the statement starting at ``s[i]``."""
    i = _skip_ws(s, i)
    if i >= len(s):
        return i
    if s[i] == "{":
        return _match(s, i) + 1
    word = _word_at(s, i)
    if word == "if":
        return _if_chain(s, i).end
    if word in ("for", "while", "switch"):
        p = s.find("(", i)
        return _statement_end(s, _match(s, p) + 1)
    if word == "do":
        body_end = _statement_end(s, i + 2)
        p = s.find("(", body_end)
        return s.find(";", _match(s, p)) + 1
    depth = 0
    j = i
    while j < len(s):
        c = s[j]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            if depth == 0:
                return j
            depth -= 1
        elif c == ";" and depth == 0:
            return j + 1
        j += 1
    return j


@dataclass
class Chain:
    """``if (c0) b0 else if (c1) b1 ... [else e]`` as offsets into the source."""

    start: int
    conditions: list[tuple[int, int]]
    bodies: list[tuple[int, int]]
    else_body: tuple[int, int] | None
    end: int


def _if_chain(s: str, i: int) -> Chain:
    """Parse ``if (...) stmt [else if (...) stmt]* [else stmt]`` at ``s[i]``."""
    chain = Chain(i, [], [], None, i)
    j = i
    while True:
        j = _skip_ws(s, j + 2)  # past "if"
        if _word_at(s, j) == "constexpr":
            j = _skip_ws(s, j + len("constexpr"))
        if j >= len(s) or s[j] != "(":
            chain.end = j
            return chain
        close = _match(s, j)
        chain.conditions.append((j + 1, close))
        body = _skip_ws(s, close + 1)
        j = _statement_end(s, body)
        chain.bodies.append((body, j))
        chain.end = j
        k = _skip_ws(s, j)
        if _word_at(s, k) != "else":
            return chain
        k = _skip_ws(s, k + 4)
        if _word_at(s, k) == "if":
            j = k
            continue
        chain.end = _statement_end(s, k)
        chain.else_body = (k, chain.end)
        return chain


# ── what a condition and a branch say ────────────────────────────────────

_SUBJECT = r"[A-Za-z_]\w*(?:(?:\.|->)[A-Za-z_]\w*)*(?:\(\s*[\w.\->]*\s*\))?"
_TEST = re.compile(
    rf"(?P<lhs>{_SUBJECT})\s*(?P<op>[=!]=)\s*Dtype::(?P<a>\w+)"
    rf"|Dtype::(?P<b>\w+)\s*(?P<op2>[=!]=)\s*(?P<rhs>{_SUBJECT})"
)
_ONE_EQ = rf"(?:\(\s*)?(?:{_SUBJECT}\s*==\s*Dtype::\w+|Dtype::\w+\s*==\s*{_SUBJECT})(?:\s*\))?"
_OR_OF_TESTS = re.compile(rf"{_ONE_EQ}(?:\s*\|\|\s*{_ONE_EQ})+")
_REFUSAL = re.compile(r"\bErrorBuilder\b|\bthrow\b|\bnot_implemented\b")
_EXITS = re.compile(r"\b(?:return|throw)\b[^;]*;\s*\}?\s*$|\bnot_implemented\b")
#: Reading or writing memory at one fixed float width.
_FIXED_WIDTH = re.compile(
    r"\breinterpret_cast\s*<\s*(?:const\s+)?(?:float|double)\s*\*"
    r"|\bstatic_cast\s*<\s*(?:const\s+)?(?:float|double)\s*\*"
    r"|\b(?:float|double)\s*\{\s*\}"
)


def _dtype_test(cond: str, op: str) -> tuple[str, frozenset[str]] | None:
    """``(subject, dtypes)`` when ``cond`` is only dtype tests joined one way.

    ``op="=="`` accepts ``subject == Dtype::A || subject == Dtype::B ...``;
    ``op="!="`` accepts ``subject != Dtype::A && subject != Dtype::B ...``.
    """
    joiner = "|" if op == "==" else "&"
    subjects: set[str] = set()
    dtypes: set[str] = set()
    for m in _TEST.finditer(cond):
        if (m.group("op") or m.group("op2")) != op:
            return None
        subjects.add((m.group("lhs") or m.group("rhs")).replace(" ", ""))
        dtypes.add(m.group("a") or m.group("b"))
    rest = re.sub(rf"[\s(){re.escape(joiner)}]", "", _TEST.sub("", cond))
    if not dtypes or len(subjects) != 1 or rest:
        return None
    return subjects.pop(), frozenset(dtypes)


def _is_refusal(body: str) -> bool:
    return bool(_REFUSAL.search(body))


# ── which function a position is in ──────────────────────────────────────

_NOT_FUNCTIONS = frozenset(
    {"if", "for", "while", "switch", "catch", "return", "sizeof", "decltype"}
)
#: What may stand between a parameter list's ``)`` and the body's ``{``.
_SIGNATURE_TAIL = re.compile(
    r"(?:\s|const\b|override\b|noexcept\b|final\b|mutable\b|volatile\b|&)*" r"(?:->[^;{}()]*)?"
)


@dataclass(frozen=True)
class Function:
    open: int
    close: int
    name: str


def _functions(s: str) -> list[Function]:
    """Every function body in ``s``.

    A ``{`` opens a function body when the text before it is a parameter
    list (plus qualifiers and a trailing return type) whose ``(`` follows
    a name that is not a control keyword.  Lambdas (``](``) are skipped,
    so a dispatch inside one counts against the function around it.
    """
    found: list[Function] = []
    for brace in re.finditer(r"\{", s):
        i = brace.start()
        close = s.rfind(")", 0, i)
        if close < 0 or not _SIGNATURE_TAIL.fullmatch(s, close + 1, i):
            continue
        depth = 0
        k = close
        while k >= 0:
            c = s[k]
            if c == ")":
                depth += 1
            elif c == "(":
                depth -= 1
                if depth == 0:
                    break
            elif c == ";":
                k = -1
                break
            k -= 1
        if k < 0:
            continue
        j = k - 1
        while j >= 0 and s[j].isspace():
            j -= 1
        if j < 0 or s[j] == "]":
            continue
        e = j + 1
        while j >= 0 and (s[j].isalnum() or s[j] in "_~"):
            j -= 1
        name = s[j + 1 : e]
        if not name or name in _NOT_FUNCTIONS or name[0].isdigit():
            continue
        found.append(Function(i, _match(s, i), name))
    return found


def _function_at(functions: list[Function], pos: int) -> Function | None:
    """The innermost named function containing ``pos``."""
    best = None
    for f in functions:
        if f.open < pos < f.close and (best is None or f.close - f.open < best.close - best.open):
            best = f
    return best


# ── the check ────────────────────────────────────────────────────────────


@dataclass
class Hit:
    path: str
    line: int
    function: str
    rule: str
    detail: str


@dataclass(frozen=True)
class Guard:
    """``if (x != A && x != B ...) refuse`` — only A, B, ... get past it."""

    pos: int
    function: Function | None
    subject: str
    admits: frozenset[str]


def _fallthrough(s: str, chain: Chain) -> tuple[int, int] | None:
    """The statement an early-return chain falls through to, if any."""
    if not all(_EXITS.search(s[lo:hi]) for lo, hi in chain.bodies):
        return None
    k = _skip_ws(s, chain.end)
    if k >= len(s) or s[k] in "}":
        return None
    return k, _statement_end(s, k)


def scan(path: Path, root: Path = ROOT) -> list[Hit]:
    """Every hit in one file; ``root`` is the checkout the file belongs to."""
    rel = path.resolve().relative_to(root).as_posix()
    return scan_text(path.read_text(encoding="utf-8", errors="replace"), rel)


def scan_text(text: str, rel: str) -> list[Hit]:
    """Every hit in ``text``, reported against the repository path ``rel``."""
    s = strip_source(text)
    functions = _functions(s)
    hits: list[Hit] = []
    guards: list[Guard] = []
    capability_guards: set[int] = set()

    def line_of(pos: int) -> int:
        return s.count("\n", 0, pos) + 1

    def where(pos: int) -> str:
        f = _function_at(functions, pos)
        return f.name if f else "<file scope>"

    chains: list[Chain] = []
    for m in re.finditer(r"\bif\b", s):
        if re.search(r"(?<!\w)else\s*$", s[max(0, m.start() - 16) : m.start()]):
            continue  # an ``else if`` belongs to the chain its ``if`` began
        chain = _if_chain(s, m.start())
        if not chain.conditions:
            continue
        chains.append(chain)
        first = s[chain.conditions[0][0] : chain.conditions[0][1]]
        refused = _dtype_test(first, "!=")
        lo, hi = chain.bodies[0]
        if refused and len(chain.conditions) == 1 and _is_refusal(s[lo:hi]):
            subject, admits = refused
            guards.append(Guard(m.start(), _function_at(functions, m.start()), subject, admits))
        # ``if (dt == F32 || dt == F64) {...} else refuse`` is a guard too.
        if len(chain.conditions) == 1 and _dtype_test(first, "=="):
            rest = chain.else_body or _fallthrough(s, chain)
            if rest is not None and _is_refusal(s[rest[0] : rest[1]]):
                capability_guards.add(line_of(chain.conditions[0][0]))

    for chain in chains:
        tests = [_dtype_test(s[lo:hi], "==") for lo, hi in chain.conditions]
        if any(t is None for t in tests):
            continue
        subjects = {t[0] for t in tests if t is not None}
        named = frozenset().union(*(t[1] for t in tests if t is not None))
        if len(subjects) != 1 or not named & {"F32", "F64"}:
            continue
        subject = subjects.pop()
        if chain.else_body is not None:
            tail = chain.else_body
            if _is_refusal(s[tail[0] : tail[1]]):
                continue
        else:
            fall = _fallthrough(s, chain)
            if fall is None or _is_refusal(s[fall[0] : fall[1]]):
                continue
            if not _FIXED_WIDTH.search(s, fall[0], fall[1]):
                continue
            if _word_at(s, fall[0]) == "if":
                nxt = _if_chain(s, fall[0])
                cond = s[nxt.conditions[0][0] : nxt.conditions[0][1]] if nxt.conditions else ""
                found = _dtype_test(cond, "==")
                if found is not None and found[0] == subject:
                    continue  # the chain goes on; its last link is checked on its own
            tail = fall
        home = _function_at(functions, chain.start)
        if any(
            g.function == home
            and g.pos < chain.start
            and g.subject == subject
            and named < g.admits
            and len(g.admits - named) == 1
            for g in guards
        ):
            continue  # an earlier refusal left exactly one dtype for the else
        hits.append(
            Hit(
                rel,
                line_of(tail[0]),
                where(chain.start),
                "bare-else",
                f"`{subject}` names {{{', '.join(sorted(named))}}}; every other "
                f"dtype falls to {'the else' if chain.else_body else 'the next statement'} "
                f"without a refusal",
            )
        )

    if rel != DEFINING_FILE:
        for m in _OR_OF_TESTS.finditer(s):
            found = _dtype_test(m.group(0), "==")
            if found is None:
                continue
            subject, dtypes = found
            if not {"F32", "F64"} <= dtypes or HALVES <= dtypes:
                continue
            if line_of(m.start()) in capability_guards:
                continue
            hits.append(
                Hit(
                    rel,
                    line_of(m.start()),
                    where(m.start()),
                    "float-list",
                    f"`{subject}` tested against {{{', '.join(sorted(dtypes))}}}, "
                    f"which misses {{{', '.join(sorted(FLOATS - dtypes))}}}; ask "
                    f"is_floating_point / is_complex",
                )
            )
    hits.extend(_half_pair_hits(s, rel, functions, line_of))
    return hits


_SWITCH = re.compile(r"\bswitch\b")
_CASE_DTYPE = re.compile(r"\bcase\s+Dtype::(\w+)")


def _half_pair_hits(
    s: str, rel: str, functions: list[Function], line_of: Callable[[int], int]
) -> list[Hit]:
    """Switches whose own case labels name F16 but not BF16."""
    bodies: list[tuple[int, int, int]] = []  # (switch keyword, ``{``, ``}``)
    for m in _SWITCH.finditer(s):
        paren = _skip_ws(s, m.end())
        if paren >= len(s) or s[paren] != "(":
            continue
        brace = _skip_ws(s, _match(s, paren) + 1)
        if brace < len(s) and s[brace] == "{":
            bodies.append((m.start(), brace, _match(s, brace)))
    hits: list[Hit] = []
    for start, lo, hi in bodies:
        nested = [(a, b) for _, a, b in bodies if lo < a and b < hi]
        labels = {
            c.group(1)
            for c in _CASE_DTYPE.finditer(s, lo, hi)
            if not any(a < c.start() < b for a, b in nested)
        }
        if "F16" in labels and "BF16" not in labels:
            f = _function_at(functions, start)
            hits.append(
                Hit(
                    rel,
                    line_of(start),
                    f.name if f else "<file scope>",
                    "half-pair",
                    "switch names F16 but not BF16; bfloat16 reaches the "
                    "default. Name both 16-bit floats (they share a path)",
                )
            )
    return hits


# ── self-test: the rule still catches what it was written for ────────────

#: ``(snippet, expected hits as (rule, function) pairs)``.
_SELF_TEST: list[tuple[str, list[tuple[str, str]]]] = [
    (  # the CHA-225 shape: a bare else at another width
        "Storage k(const Storage& g, Dtype dt) {\n"
        "    if (dt == Dtype::F32) { add(reinterpret_cast<float*>(p)); }\n"
        "    else { add(reinterpret_cast<double*>(p)); }\n}\n",
        [("bare-else", "k")],
    ),
    (  # the early-return spelling of the same chain, inside a lambda
        "Storage k(Dtype dt) {\n"
        "    auto read = [&](std::size_t i) -> double {\n"
        "        if (dt == Dtype::F32) return reinterpret_cast<const float*>(p)[i];\n"
        "        return reinterpret_cast<const double*>(p)[i];\n"
        "    };\n    return read(0);\n}\n",
        [("bare-else", "k")],
    ),
    (  # an F64-first chain whose else is float
        "void k(Dtype dt) { if (dt == Dtype::F64) run(double{}); else run(float{}); }\n",
        [("bare-else", "k")],
    ),
    (  # the CHA-225 float test
        "bool differentiable(Dtype dt) { return dt == Dtype::F32 || dt == Dtype::F64; }\n",
        [("float-list", "differentiable")],
    ),
    (  # ... and one that misses bfloat16 alone
        "bool f(Dtype dt) {\n"
        "    return dt == Dtype::F16 || dt == Dtype::F32 || dt == Dtype::F64;\n}\n",
        [("float-list", "f")],
    ),
    (  # a chain that ends in a refusal
        "void k(Dtype dt) {\n"
        "    if (dt == Dtype::F32) run(float{});\n"
        "    else if (dt == Dtype::F64) run(double{});\n"
        '    else ErrorBuilder("k").not_implemented("dtype");\n}\n',
        [],
    ),
    (  # an earlier refusal leaves exactly one dtype for the else
        "void k(Dtype dt) {\n"
        "    if (dt != Dtype::F32 && dt != Dtype::F64)\n"
        '        ErrorBuilder("k").not_implemented("only F32/F64");\n'
        "    if (dt == Dtype::F32) run(float{}); else run(double{});\n}\n",
        [],
    ),
    (  # early returns that end in a refusal
        "Storage k(Dtype dt) {\n"
        "    if (dt == Dtype::F32) { return a(reinterpret_cast<float*>(p)); }\n"
        "    if (dt == Dtype::F64) { return a(reinterpret_cast<double*>(p)); }\n"
        '    ErrorBuilder("k").not_implemented("dtype");\n}\n',
        [],
    ),
    (  # a capability guard spelled positively, refusing in its else
        "void k(Dtype dt) {\n"
        "    if (dt == Dtype::F32 || dt == Dtype::F64) { go(); }\n"
        '    else { ErrorBuilder("k").not_implemented("dtype"); }\n}\n',
        [],
    ),
    (  # a switch with a float16 case and no bfloat16 one (LCD-248)
        "void k(Dtype dt) {\n"
        "    switch (dt) {\n"
        "    case Dtype::F32: a(); break;\n"
        "    case Dtype::F16: h(); break;\n"
        '    default: ErrorBuilder("k").not_implemented("dtype");\n'
        "    }\n}\n",
        [("half-pair", "k")],
    ),
    (  # both halves named; a nested switch's labels are its own
        "void k(Dtype dt, Dtype it) {\n"
        "    switch (dt) {\n"
        "    case Dtype::F16: case Dtype::BF16: h(); break;\n"
        "    case Dtype::I32:\n"
        "        switch (it) { case Dtype::I8: break; default: break; }\n"
        "        break;\n"
        "    default: break;\n"
        "    }\n"
        "    switch (dt) { case Dtype::I8: case Dtype::Bool: b(); break; default: break; }\n}\n",
        [],
    ),
    (  # the predicates, the half test, and a bad chain inside a comment
        "bool f(Dtype dt) { return is_floating_point(dt) || is_complex(dt); }\n"
        "bool h(Dtype dt) { return dt == Dtype::F16 || dt == Dtype::BF16; }\n"
        "// if (dt == Dtype::F32) a(); else b(reinterpret_cast<double*>(p));\n",
        [],
    ),
]


def self_test() -> int:
    """Run the rules on ``_SELF_TEST``; nonzero if any case comes out different."""
    failures = 0
    for i, (snippet, want) in enumerate(_SELF_TEST):
        got = sorted((h.rule, h.function) for h in scan_text(snippet, "snippet.h"))
        if got != sorted(want):
            failures += 1
            print(f"[check_dtype_dispatch] self-test {i}: want {want}, got {got}")
    if failures:
        return 1
    print(f"[check_dtype_dispatch] self-test OK: {len(_SELF_TEST)} case(s).")
    return 0


def _files(paths: list[str], root: Path) -> list[Path]:
    if paths:
        return [Path(p).resolve() for p in paths]
    return sorted(p for p in (root / ENGINE).rglob("*") if p.suffix in SUFFIXES)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Engine dtype dispatches must refuse.")
    ap.add_argument("paths", nargs="*", help="files to check (default: lucid/_C)")
    ap.add_argument("--list", action="store_true", help="also print known hits")
    ap.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="checkout to check (default: the one this script is in)",
    )
    ap.add_argument("--self-test", action="store_true", help="run the rules on their own examples")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    root = args.root.resolve()

    files = _files(args.paths, root)
    by_key: dict[tuple[str, str, str], list[Hit]] = {}
    for f in files:
        for h in scan(f, root):
            by_key.setdefault((_known_unit(h.path), h.function, h.rule), []).append(h)

    checked = {f.relative_to(root).as_posix() for f in files}
    unit_files: dict[str, set[str]] = {}
    for f in _files([], root):
        rel = f.relative_to(root).as_posix()
        unit_files.setdefault(_known_unit(rel), set()).add(rel)
    checked_units = {unit for unit, members in unit_files.items() if members and members <= checked}
    new: list[Hit] = []
    for key, hits in sorted(by_key.items()):
        allowed = _KNOWN.get(key, (0, ""))[0]
        if len(hits) > allowed:
            new.extend(hits)
        elif args.list:
            for h in hits:
                print(f"  known  {h.path}:{h.line} [{h.rule}] {h.function}")
    stale = [
        f"  {path} {func} [{rule}]: listed {count}, found {len(by_key.get((path, func, rule), []))}"
        for (path, func, rule), (count, _) in sorted(_KNOWN.items())
        if path in checked_units and len(by_key.get((path, func, rule), [])) < count
    ]

    for h in new:
        print(f"{h.path}:{h.line}: [{h.rule}] in {h.function}: {h.detail}", file=sys.stderr)
    if stale:
        print("[check_dtype_dispatch] stale _KNOWN entries; lower or delete them:", file=sys.stderr)
        for line in stale:
            print(line, file=sys.stderr)
    if new:
        print(
            f"\n[check_dtype_dispatch] {len(new)} dtype dispatch(es) without a "
            "refusal.  End the chain in `else ErrorBuilder(...).not_implemented(...)`, "
            "widen half through float32 (detail::as_f32 / back_to_f16), name F16 "
            "and BF16 together, and ask is_floating_point / is_complex instead of "
            "listing dtypes.",
            file=sys.stderr,
        )
    if new or stale:
        return 1
    known = sum(count for count, _ in _KNOWN.values())
    print(
        f"[check_dtype_dispatch] OK: {len(files)} file(s), no new dtype dispatch "
        f"without a refusal ({known} known site(s) listed in _KNOWN)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
