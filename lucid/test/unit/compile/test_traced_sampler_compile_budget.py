"""Every traced sampler compiles on Metal within a budget (LCD-786).

The Metal graph compiler's stitching pass recurses without memoisation, so a
traced graph shaped like a deep diamond chain compiles in time exponential in
its depth.  An unrolled rejection loop in the Gamma sampler carried a mask from
round to round, read it twice per round, and made one compiled step take 56 s
to minutes (LCD-784).  The suite saw a hang, not a failure.

This is the class-wide guard.  It enumerates, from the packages rather than a
hand list,

* every ``lucid.distributions`` class that has ``rsample`` (wrappers included),
* every top-level random op (``lucid.rand``, ``lucid.poisson`` ...),
* the public call that reaches each ``Function`` in
  ``test_function_compile_classes.TRACED_BACKWARD``,
* every ``lucid.special`` function of one or two tensors (composites built on
  unrolled series loops: ``zeta`` was already at 7 s),

builds a one-parameter model around each, and compiles one ``make_step`` of it
in a fresh process.  The compile is native and cannot be interrupted, so the
parent only watches the clock: a target that is silent for ``_STALL_S`` is
killed and reported, and the targets after it still run.

Metal is required for the compile to mean anything; on a runner without it the
timed tests skip, and the enumeration tests (which need no GPU) still run.
"""

import queue
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import pytest

import lucid
from lucid.test._fixtures.devices import metal_available
from lucid.test.unit.compile import _sampler_probes as probes
from lucid.test.unit.compile.test_function_compile_classes import TRACED_BACKWARD

pytestmark = pytest.mark.deadline(900)

#: A first compile of one of these takes 0.3-2 s on an M1 Pro; the exponential
#: sampler took 56 s.  The budget separates them by an order of magnitude.
_BUDGET_S = 10.0
#: Targets that are slow to compile today.  Their own budget is a ceiling set about 4x above
#: the measurement, so that getting worse still fails; lower it when the cause is fixed.
_SLOW_TODAY: dict[str, float] = {
    # 6.5 s on an M1 Pro (every other special function: <= 3.3 s).  ``_zeta.py`` threads
    # ``rising`` / ``a_pow`` / ``tail`` through a 12+13-term unrolled loop, each read twice
    # per term -- the shape that blew up the Gamma sampler, at a depth where it is still
    # merely slow.
    "special:zeta": 25.0,
}
#: Silence from the child for this long means it is inside a compile that is not coming back.
_STALL_S = 30.0
#: Targets that run eagerly by design (a traced value is read on the host), so there is no
#: graph to compile.  Strict both ways: one that starts to compile, or one missing here
#: that stops compiling, fails the guard.
_EAGER_BY_DESIGN = frozenset({"random:poisson"})


def _targets() -> list[str]:
    found = [f"dist:{name}" for name in probes.distribution_names()]
    found += [f"random:{name}" for name in sorted(probes.random_table())]
    found += [f"special:{name}" for name in probes.special_names()]
    functions = {probes.FUNCTION_ENTRY[name] for name in sorted(TRACED_BACKWARD)}
    found += sorted(functions - set(found))
    return found


@dataclass(frozen=True)
class _Outcome:
    seconds: float | None
    compiled: bool
    problem: str
    skip: str = ""


def _pump(stream: IO[str], lines: queue.Queue[str | None]) -> None:
    for line in stream:
        lines.put(line)
    lines.put(None)


def _run_children(targets: list[str]) -> dict[str, _Outcome]:
    root = str(Path(lucid.__file__).resolve().parent.parent)
    outcomes: dict[str, _Outcome] = {}
    remaining = list(targets)
    while remaining:
        proc = subprocess.Popen(
            [sys.executable, "-u", "-m", probes.__name__, *remaining],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        lines: queue.Queue[str | None] = queue.Queue()
        threading.Thread(target=_pump, args=(proc.stdout, lines), daemon=True).start()
        current: str | None = None
        tail: list[str] = []
        stalled = False
        while True:
            try:
                line = lines.get(timeout=_STALL_S)
            except queue.Empty:
                stalled = True
                proc.kill()
                break
            if line is None:
                break
            parts = line.split()
            if parts[:1] == ["START"]:
                current = parts[1]
            elif parts[:1] == ["SKIP"]:
                outcomes[parts[1]] = _Outcome(None, False, "", skip=" ".join(parts[2:]))
                current = None
            elif parts[:1] == ["DONE"]:
                outcomes[parts[1]] = _Outcome(float(parts[2]), parts[3] == "1", "")
                current = None
            else:
                tail = (tail + [line.rstrip()])[-12:]
        proc.wait()
        if current is None and not stalled and proc.returncode == 0:
            break
        current = current or remaining[0]
        why = (
            f"no progress for {_STALL_S:.0f} s inside the first compile (killed)"
            if stalled
            else "child died: " + " | ".join(tail)[-400:]
        )
        outcomes[current] = _Outcome(None, False, why)
        remaining = remaining[remaining.index(current) + 1 :]
    return outcomes


@pytest.fixture(scope="module")
def outcomes() -> dict[str, _Outcome]:
    if not metal_available():
        pytest.skip("Metal device not available on this host")
    return _run_children(_targets())


@pytest.mark.parametrize("target", _targets())
def test_first_compile_is_within_budget(target: str, outcomes: dict[str, _Outcome]) -> None:
    outcome = outcomes.get(target)
    assert outcome is not None, f"{target}: never reached"
    assert not outcome.problem, f"{target}: {outcome.problem}"
    if outcome.skip:
        pytest.skip(outcome.skip)
    assert outcome.seconds is not None
    budget = _SLOW_TODAY.get(target, _BUDGET_S)
    assert (
        outcome.seconds <= budget
    ), f"{target}: first make_step compile took {outcome.seconds:.1f} s (budget {budget:.0f} s)"
    if target.startswith("special:"):
        return  # many series functions read their argument on the host; they run eagerly
    assert outcome.compiled == (target not in _EAGER_BY_DESIGN), (
        f"{target}: compiled={outcome.compiled}; the guard only means something for a target "
        "that reaches the graph compiler (update _EAGER_BY_DESIGN if that changed on purpose)"
    )


# The enumeration itself, which needs no GPU.


def test_enumeration_is_not_vacuous() -> None:
    names = probes.distribution_names()
    assert {"Gamma", "Beta", "Dirichlet", "Wishart", "Independent"} <= set(names)
    assert len(names) >= 30


@pytest.mark.parametrize("name", probes.distribution_names())
def test_every_rsample_distribution_can_be_probed(name: str) -> None:
    dist = probes.build_distribution(name)
    assert dist.has_rsample, f"D.{name} no longer has rsample; the enumeration is out of date"


def test_every_traced_backward_function_has_a_probe() -> None:
    assert set(probes.FUNCTION_ENTRY) == set(TRACED_BACKWARD)
    for entry in probes.FUNCTION_ENTRY.values():
        kind, _, key = entry.partition(":")
        assert key in (probes.distribution_names() if kind == "dist" else probes.FUNCTIONS)


def test_every_random_op_has_a_probe() -> None:
    assert probes.random_names_in_package() <= probes.random_table()
