"""A compile timing guard measures through ``_helpers.compare_ms`` (LCD-937).

One wall-clock sample per side flaked every time the machine was loaded: a spike on one side
read as a regression, and each flake cost an attest rerun or a red CI run.  The helper times
both sides in alternation, keeps each side's minimum and retries from a fresh build; a test
that reads the clock itself brings the flake back, so no ``test_*.py`` here may.
"""

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

_DIR = Path(__file__).parent
_CLOCKS = frozenset({"perf_counter", "perf_counter_ns"})


def _raw_clock_reads(source: str) -> Iterator[int]:
    """Line numbers of a ``perf_counter`` use or a ``timeit`` import (not strings: a script
    run in a child process may time itself)."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and node.attr in _CLOCKS:
            yield node.lineno
        elif isinstance(node, ast.ImportFrom) and node.module in {"time", "timeit"}:
            if node.module == "timeit" or any(alias.name in _CLOCKS for alias in node.names):
                yield node.lineno
        elif isinstance(node, ast.Import) and any(a.name == "timeit" for a in node.names):
            yield node.lineno


def test_no_compile_test_reads_the_clock_itself() -> None:
    offenders = {
        path.name: lines
        for path in sorted(_DIR.glob("test_*.py"))
        if (lines := list(_raw_clock_reads(path.read_text())))
    }
    assert not offenders, (
        "time a guard with lucid.test.unit.compile._helpers.compare_ms / assert_ratio_within, "
        f"not the clock: {offenders}"
    )


@pytest.mark.parametrize(
    "source",
    [
        "import time\nt = time.perf_counter()\n",
        "from time import perf_counter\n",
        "from time import perf_counter_ns as now\n",
        "import timeit\n",
        "from timeit import timeit\n",
    ],
)
def test_the_detector_sees_each_way_to_read_the_clock(source: str) -> None:
    assert list(_raw_clock_reads(source))


@pytest.mark.parametrize(
    "source", ['_CHILD = "import time; time.perf_counter()"\n', "import time\ntime.sleep(0)\n"]
)
def test_the_detector_ignores_what_is_not_a_clock_read(source: str) -> None:
    assert not list(_raw_clock_reads(source))
