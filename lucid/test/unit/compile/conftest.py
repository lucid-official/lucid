"""Fixtures shared by the compile tests."""

import os
from collections.abc import Iterator

import pytest

import lucid
from lucid.test._fixtures.compile_notices import VERBOSE_ENV


@pytest.fixture
def inference() -> Iterator[None]:
    """Run the test under ``no_grad``, where a compiled forward is used.

    Under grad mode a call that reads a tensor requiring grad — any model
    with trainable parameters — runs eagerly so that its result is
    differentiable (LCD-582).  A test of the compiled forward itself opts
    in with ``pytestmark = pytest.mark.usefixtures("inference")``.
    """
    with lucid.no_grad():
        yield


@pytest.fixture(autouse=True)
def _compile_verbose_not_left_changed(request: pytest.FixtureRequest) -> Iterator[None]:
    """Fail the test that leaves ``LUCID_COMPILE_VERBOSE`` different from how it found it.

    The root ``_fresh_compile_notices`` fixture quietly takes the variable back
    after every test, so a bare ``os.environ[...] = "1"`` (the old matrix
    helpers, LCD-868) no longer breaks the next test -- but it would also go
    unnoticed.  This one makes it loud, and names the test: set the variable
    through ``monkeypatch.setenv`` or ``_helpers.verbose_compile_notices``.

    Being nearer to the test than the root fixture, it is torn down first and
    still sees what the test left.  The root fixture then resets, so one
    offender fails once instead of cascading.
    """
    before = os.environ.get(VERBOSE_ENV)
    yield
    after = os.environ.get(VERBOSE_ENV)
    if after != before:
        pytest.fail(
            f"{request.node.nodeid} left {VERBOSE_ENV} as {after!r} (it was {before!r}); "
            "set it with monkeypatch.setenv or _helpers.verbose_compile_notices",
            pytrace=False,
        )
