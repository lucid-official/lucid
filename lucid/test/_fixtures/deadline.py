"""A wall-clock deadline for tests that reach a native graph compile (LCD-786).

The Metal graph compiler (MPSGraph) runs natively, inside the first call of a
compiled step, and cannot be interrupted from Python: a traced graph shaped
like a deep diamond chain compiles in exponential time (LCD-784, a Gamma
sampler that took 56 s to a minute and a half per step) and the suite shows
that as a hang -- the job dies at its 210-minute timeout with no stack.

Python cannot interrupt a native call, so the deadline is a watchdog that
does not need Python to run: :func:`faulthandler.dump_traceback_later` is a
C thread.  When a test outlives its deadline it prints every thread's stack
to the real stderr (pytest's capture would swallow it) and ends the process,
so the run fails within the deadline and says where it was.  It ends the
*session*, not just the test: a thread stuck in native code cannot be
unwound, and a child process per test would not give these tests their
fixtures.  The class-wide guard that reports each offender separately lives
in ``unit/compile/test_traced_sampler_compile_budget.py`` and does use a
child process.

Which tests get a deadline:

* every test under ``unit/compile/`` -- 120 s;
* a test in any other module that names the compile entry points
  (``make_step`` / ``fused_step`` / ``compiled_step`` / ``CompiledModule`` /
  ``lucid.compile``) -- 300 s, those tests also train and build models;
* ``slow`` / ``heavy`` tests among them -- 900 s;
* ``@pytest.mark.deadline(seconds)`` overrides all of the above.

``LUCID_TEST_DEADLINE_SCALE`` multiplies every deadline (a slow machine);
``0`` turns the watchdog off.
"""

import faulthandler
import functools
import os
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

COMPILE_DIR_DEADLINE_S = 120.0
ELSEWHERE_DEADLINE_S = 300.0
SLOW_DEADLINE_S = 900.0
SCALE_ENV = "LUCID_TEST_DEADLINE_SCALE"

_COMPILE_DIR = Path(__file__).resolve().parent.parent / "unit" / "compile"
_REACHES_COMPILE = re.compile(
    r"\b(?:make_step|fused_step|compiled_step|CompiledModule)\b"
    r"|\blucid\.compile\b|\bfrom\s+lucid\s+import\s+compile\b"
)


def _scale() -> float:
    raw = os.environ.get(SCALE_ENV, "")
    if not raw:
        return 1.0
    try:
        return max(0.0, float(raw))
    except ValueError as exc:
        raise ValueError(f"{SCALE_ENV}={raw!r} is not a number") from exc


@functools.cache
def _module_reaches_compile(path: str) -> bool:
    try:
        return _REACHES_COMPILE.search(Path(path).read_text(encoding="utf-8")) is not None
    except OSError:
        return False


def deadline_seconds(
    path: Path, *, explicit: float | None = None, slow: bool = False
) -> float | None:
    """The deadline for a test in ``path``, or ``None`` when it has none."""
    scale = _scale()
    if scale == 0.0:
        return None
    if explicit is not None:
        return explicit * scale
    resolved = path.resolve()
    if _COMPILE_DIR in resolved.parents:
        base = SLOW_DEADLINE_S if slow else COMPILE_DIR_DEADLINE_S
    elif _module_reaches_compile(str(resolved)):
        base = SLOW_DEADLINE_S if slow else ELSEWHERE_DEADLINE_S
    else:
        return None
    return base * scale


def arm(seconds: float, fd: int) -> None:
    """Start the watchdog: after ``seconds`` dump all stacks to ``fd`` and exit."""
    faulthandler.dump_traceback_later(seconds, repeat=False, file=fd, exit=True)


def disarm() -> None:
    faulthandler.cancel_dump_traceback_later()


@pytest.fixture(scope="session")
def _deadline_stderr_fd(request: pytest.FixtureRequest) -> int:
    """A dup of the real stderr, taken while pytest's capture is suspended."""
    capman = request.config.pluginmanager.getplugin("capturemanager")
    if capman is None:
        return os.dup(2)
    with capman.global_and_fixture_disabled():
        return os.dup(2)


@pytest.fixture(autouse=True)
def _native_compile_deadline(
    request: pytest.FixtureRequest, _deadline_stderr_fd: int
) -> Iterator[None]:
    marker = request.node.get_closest_marker("deadline")
    explicit = float(marker.args[0]) if marker is not None and marker.args else None
    seconds = deadline_seconds(
        request.node.path,
        explicit=explicit,
        slow=request.node.get_closest_marker("slow") is not None
        or request.node.get_closest_marker("heavy") is not None,
    )
    if seconds is None:
        yield
        return
    arm(seconds, _deadline_stderr_fd)
    try:
        yield
    finally:
        disarm()
