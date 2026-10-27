"""Lucid's own counters: suppressions, casts, ``Any``, and file size over the cap.

Read with ``tokenize`` so a ``cast(``, ``Any`` or ``type: ignore`` inside a
string or docstring is not counted; only real comments, calls and names are.
``Any`` is counted as a name, so ``typing.Any`` and its import count too:
trading a ``type: ignore`` for an ``Any`` is laundering, not a fix (DS-16).
"""

import io
import re
import tokenize
from collections import defaultdict
from typing import override

from tools.quality_gate import config
from tools.quality_gate.collectors import Collector
from tools.quality_gate.core import Counts, Scope

_TYPE_IGNORE = re.compile(r"type:\s*ignore(?:\[([^\]]*)\])?")
_NOQA = re.compile(r"noqa(?::\s*([A-Z]+[0-9]+(?:[\s,]+[A-Z]+[0-9]+)*))?", re.IGNORECASE)


def _comment_rules(comment: str) -> list[str]:
    """One rule per suppressed code; a suppression without a code is its own rule."""
    rules: list[str] = []
    for m in _TYPE_IGNORE.finditer(comment):
        codes = [c.strip() for c in (m.group(1) or "").split(",") if c.strip()]
        rules += [f"type-ignore[{c}]" for c in codes] or ["type-ignore-bare"]
    for m in _NOQA.finditer(comment):
        codes = [c.upper() for c in re.split(r"[\s,]+", m.group(1) or "") if c]
        rules += [f"noqa[{c}]" for c in codes] or ["noqa-bare"]
    return rules


def count_python(text: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    prev: tokenize.TokenInfo | None = None
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.COMMENT:
                for rule in _comment_rules(tok.string):
                    counts[rule] += 1
            elif tok.type == tokenize.NAME and tok.string == "Any":
                counts["any"] += 1
            elif (
                tok.type == tokenize.OP
                and tok.string == "("
                and prev is not None
                and prev.type == tokenize.NAME
                and prev.string == "cast"
            ):
                counts["cast"] += 1
            if tok.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT):
                prev = tok
    except (tokenize.TokenError, SyntaxError):
        counts["untokenizable"] += 1
    return dict(counts)


def loc(text: str) -> int:
    return text.count("\n") + (0 if text.endswith("\n") or not text else 1)


class CountersCollector(Collector):
    name = "counters"
    local = True
    fast = True

    @override
    def version(self) -> str:
        return "3"  # 3: applies LOC_CAP_BACKEND_METHODS to backend/{cpu,gpu}/methods/*.cpp

    @override
    def collect(self, scope: Scope) -> Counts:
        counts: Counts = {}
        for path in scope.py:
            text = (scope.root / path).read_text(encoding="utf-8", errors="replace")
            for rule, n in count_python(text).items():
                counts[("counters", rule, path)] = n
            over = loc(text) - config.LOC_CAP_PY
            if over > 0:
                counts[("counters", "file-loc-over-cap", path)] = over
        for path in scope.cpp:
            text = (scope.root / path).read_text(encoding="utf-8", errors="replace")
            cap = (
                config.LOC_CAP_BACKEND_METHODS
                if path.startswith(config.BACKEND_METHODS_PREFIXES)
                else config.LOC_CAP_CPP
            )
            over = loc(text) - cap
            if over > 0:
                counts[("counters", "file-loc-over-cap", path)] = over
        return counts
