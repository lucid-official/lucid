"""Wall-clock comparisons belong to the compile performance tier."""

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
        "measure compile performance under lucid/test/perf/compile, " f"not the clock: {offenders}"
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


_CLOCK_NAMES = _CLOCKS | {"monotonic", "monotonic_ns", "time", "time_ns", "timeit"}
_RATIO_HELPERS = {"assert_ratio_within", "compare_ms"}


def _ratio_lines(source: str) -> set[int]:
    tree = ast.parse(source)
    aliases = {
        name.asname: name.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.ImportFrom, ast.Import))
        for name in node.names
        if name.asname
    }
    values: dict[str, int] = {}
    point, duration, ratio = 1, 2, 3

    def kind(expr: ast.AST) -> int:
        if isinstance(expr, ast.Name):
            return values.get(expr.id, 0)
        if isinstance(expr, ast.Call):
            name = getattr(expr.func, "attr", getattr(expr.func, "id", ""))
            name = aliases.get(name, name)
            if name in _CLOCK_NAMES:
                return duration if name == "timeit" else point
            if name in _RATIO_HELPERS:
                return ratio
            return max(values.get(name, 0), max((kind(arg) for arg in expr.args), default=0))
        if isinstance(expr, ast.BinOp):
            left, right = kind(expr.left), kind(expr.right)
            if isinstance(expr.op, ast.Sub) and left == right == point:
                return duration
            if isinstance(expr.op, ast.Div) and left == right == duration:
                return ratio
            return max(left, right)
        if isinstance(expr, ast.Compare):
            sides = [kind(expr.left), *(kind(value) for value in expr.comparators)]
            if sides.count(duration) >= 2 or ratio in sides:
                return ratio
        return max((kind(child) for child in ast.iter_child_nodes(expr)), default=0)

    nodes = list(ast.walk(tree))
    # Assignments and helper returns can chain in either source order.
    while True:
        before = dict(values)
        for node in nodes:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    for name in ast.walk(target):
                        if isinstance(name, ast.Name):
                            values[name.id] = max(values.get(name.id, 0), kind(node.value))
            if isinstance(node, ast.FunctionDef):
                for ret in ast.walk(node):
                    if isinstance(ret, ast.Return) and ret.value is not None:
                        values[node.name] = max(values.get(node.name, 0), kind(ret.value))
        if before == values:
            break
    return {
        node.lineno
        for node in nodes
        if (isinstance(node, ast.Assert) and kind(node.test) == ratio)
        or (isinstance(node, ast.ImportFrom) and any(a.name in _RATIO_HELPERS for a in node.names))
        or (isinstance(node, ast.Call) and getattr(node.func, "attr", None) in _RATIO_HELPERS)
    }


def test_compile_unit_files_do_not_gate_on_relative_wall_time() -> None:
    offenders = {
        str(path.relative_to(_DIR)): lines
        for path in _DIR.rglob("*.py")
        if (lines := _ratio_lines(path.read_text()))
    }
    assert not offenders, offenders


@pytest.mark.parametrize("comparison", ["a / b <= 1.5", "a <= 2 * b + 0.1", "a < b"])
def test_relative_durations_are_detected(comparison: str) -> None:
    source = (
        "from time import perf_counter_ns as now\n"
        "def measure(fn):\n    start = now()\n    fn()\n    return now() - start\n"
        "a, b = measure(f), measure(g)\n"
        f"assert {comparison}\n"
    )
    assert _ratio_lines(source)


def test_an_absolute_deadline_and_a_numerical_ratio_are_allowed() -> None:
    assert not _ratio_lines("until = time.monotonic() + 1\nassert time.monotonic() < until\n")
    assert not _ratio_lines("assert computed / reference < 1.001")
