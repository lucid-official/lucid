"""A compile must come back, on the graph shapes MPSGraph walks path by path (LCD-838).

``lucid.compile(lambda a: lucid.linalg.eigvalsh(a))`` on a 4x4 never returned,
and could not be interrupted.  Two MPSGraph passes keep no visited set, and the
two-sided Jacobi graph then behind ``eigh`` was the deep diamond chain that makes
them exponential:

* ``cloneForFeeds:`` (inside ``compileWithDevice:``) doubles on a chain of tight
  diamonds.  A 4x4 ``eigvalsh`` hung here, and an ``x = x * a + x * b`` loop with
  no linalg in it took 34 s at 24 rounds on an M4 Max, doubling with every round
  after.  Its worklist grows too, 15 GB in two minutes.
* ``MemrefDeallocationPass`` (inside the first run) enumerates forward paths.
  A 6x6 ``eigh`` hung here, and with it full-matrices ``svd``, whose completion
  is an ``eigh``.

``eigh`` now shares the graph ``svd`` was measured on, and ``compile_graph``
replays the first walk before starting it.  The second walk runs on the optimized
module and cannot be replayed, so user code can still reach it (LCD-847).

Each group runs in a child that is killed past a time or a memory ceiling, so a
regression fails this test rather than hanging the suite or taking the machine.
"""

import subprocess
import sys
import time
from pathlib import Path

import pytest

import lucid
from lucid.test.audit._zoo_budget import footprint_gb
from lucid.test.unit.compile._jacobi_probes import GROUP_NAMES, WRONG_ANSWER

#: The slowest group took 13.7 s on a loaded Studio.  Exponential cost overshoots any ceiling.
_CEILING_S = 90
#: A regression in the clone walk reached 6 GB in under 30 s; a hosted runner has about 7 GB.
_CEILING_GB = 3.0
_POLL_S = 0.2
#: 256 rounds of the LCD-847 chain take 7 s and 320 do not return in 40; the probe runs 384.
_PINNED_HANG_CEILING_S = 5


class _NoReturn(Exception):
    """The child was still running at its time ceiling."""


class _OverMemory(Exception):
    """The child passed the footprint ceiling."""


class _WrongAnswer(AssertionError):
    """The child finished and its answer was wrong."""


def _finished_probes(partial_stdout: bytes | None) -> str:
    done = (partial_stdout or b"").decode().strip().splitlines()
    return done[-1] if done else "nothing"


def _watch(child: subprocess.Popen[str], ceiling_s: float) -> str:
    """Return the child's stderr once it exits; raise past a ceiling."""
    deadline = time.monotonic() + ceiling_s
    while True:
        try:
            return child.communicate(timeout=_POLL_S)[1]
        except subprocess.TimeoutExpired as waited:
            footprint = footprint_gb(child.pid)
            if footprint > _CEILING_GB:
                raise _OverMemory(
                    f"{footprint:.1f} GB, over {_CEILING_GB} GB; "
                    f"last finished: {_finished_probes(waited.output)}"
                ) from None
            if time.monotonic() > deadline:
                raise _NoReturn(
                    f"no return after {ceiling_s} s; "
                    f"last finished: {_finished_probes(waited.output)}"
                ) from None


def _run_group(group: str, ceiling_s: float = _CEILING_S) -> None:
    with subprocess.Popen(
        [sys.executable, "-m", "lucid.test.unit.compile._jacobi_probes", group],
        cwd=Path(lucid.__file__).resolve().parent.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as child:
        try:
            err = _watch(child, ceiling_s)
        finally:
            child.kill()
    if child.returncode == WRONG_ANSWER:
        raise _WrongAnswer(err[-600:])
    assert child.returncode == 0, err[-600:]


@pytest.mark.parametrize("group", GROUP_NAMES)
def test_compile_returns_with_the_right_answer(group: str, device_gpu_only: str) -> None:
    del device_gpu_only
    _run_group(group)


@pytest.mark.xfail(
    strict=True,
    raises=_WrongAnswer,
    reason="LCD-846: the shift rounds away eigenpairs under eps * ||A||; accurate to that, "
    "not relatively",
)
def test_small_eigenpairs_beside_a_large_entry(device_gpu_only: str) -> None:
    del device_gpu_only
    _run_group("small_beside_large")


# Not strict: the hang depends on the machine's MPSGraph.  It never returns on an M4 Max
# and returned inside the ceiling on CI's macos-26 runner.
@pytest.mark.xfail(
    strict=False,
    raises=_NoReturn,
    reason="LCD-847: MemrefDeallocationPass enumerates paths in the optimized module, which "
    "the clone walk's budget cannot see",
)
def test_long_rotation_chain_returns(device_gpu_only: str) -> None:
    del device_gpu_only
    _run_group("rotation_chain", ceiling_s=_PINNED_HANG_CEILING_S)
