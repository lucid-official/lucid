"""A compile process that has reported nothing, for every test (LCD-915).

``lucid.compile`` says that a call ran eagerly in one of two ways, decided per
call by process-global state a test can leave behind:

* once per *kind of reason* per process (``fallback._NOTICED``) -- the first
  test of a run to hit a kind consumes the notice;
* not at all as a warning when ``LUCID_COMPILE_VERBOSE=1`` -- the reason is
  printed to stderr instead.  The matrix helpers (``unit/compile/_op_matrix.py``
  ``run``) set it with a bare ``os.environ[...] = "1"``, which outlives the
  test that called them.

Either way a later ``pytest.warns(CompileFallbackWarning)`` sees nothing, so a
test that passes alone fails after whichever test ran before it (the six
``make_step`` / ``fused_step`` cases of
``unit/autograd/test_function_outputs_under_compile.py`` after ``unit/compile``).
The ``unit/compile`` files each reset one or the other by hand; a file outside
that directory cannot be expected to know.  So the reset lives here, in one
autouse fixture wired from the root ``conftest``: no test inherits what an
earlier one announced or switched on, wherever it sits, and a test that wants
verbose output sets it itself (``monkeypatch.setenv``) after this runs.

The notice set is cleared **in place**.  Several tests do ``from ...fallback
import _NOTICED`` and ``discard`` from it; rebinding the module attribute would
leave them editing a set the product no longer reads.
"""

import os
from collections.abc import Iterator

import pytest

from lucid.compile._core import fallback

VERBOSE_ENV = "LUCID_COMPILE_VERBOSE"


def forget_reports() -> None:
    """Make the process one that has announced no fallback and is not verbose."""
    fallback._NOTICED.clear()
    os.environ.pop(VERBOSE_ENV, None)


@pytest.fixture(autouse=True)
def _fresh_compile_notices() -> Iterator[None]:
    """Start every test -- and leave the next one -- with nothing reported."""
    forget_reports()
    yield
    forget_reports()
