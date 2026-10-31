"""Lucid test suite — root pytest configuration.

This file wires the per-area fixtures into pytest's discovery, sets
the autouse determinism hook, and registers a single
``configure_markers`` step that mirrors the marker matrix declared in
``pyproject.toml``.

What you get for free in any test:

* ``device`` — parametrize over CPU and (when present) Metal.
* ``device_cpu_only`` / ``device_gpu_only`` / ``cross_device_pair``.
* ``float_dtype`` — parametrize over float32 + float64.
* ``int_dtype`` — parametrize over int8/16/32/64.
* ``device`` together with a dtype fixture — swept as pairs, leaving out
  the ones the device cannot hold (Metal has no float64), so no test has
  to skip them.  See ``_fixtures/devices.py``.
* ``ref`` — lazy reference framework module (skips test if missing).
* ``tensor_factory`` — device-aware ``make_tensor`` shorthand.
* ``bench`` — ``pytest-benchmark``-compatible benchmark callable
  (degrades gracefully when the dep is missing).
* Autouse ``manual_seed(0)`` per test.
* Autouse wall-clock deadline for tests that reach a native graph compile
  (``_fixtures/deadline.py``); ``@pytest.mark.deadline(seconds)`` overrides.
"""

import pytest

import lucid
from lucid.test import _memory

# Re-export every fixture from the per-area modules so test files just
# need to declare the fixture name in their argument list.  The hook is
# re-exported for the same reason: pytest only registers hooks it finds
# in a conftest (or a plugin).
from lucid.test._fixtures.devices import (  # noqa: F401
    device,
    device_cpu_only,
    device_gpu_only,
    cross_device_pair,
    pytest_generate_tests,
)
from lucid.test._fixtures.deadline import (  # noqa: F401
    _deadline_stderr_fd,
    _native_compile_deadline,
)
from lucid.test._fixtures.dtypes import float_dtype, float_dtype_extended, int_dtype  # noqa: F401
from lucid.test._fixtures.ref_framework import ref  # noqa: F401
from lucid.test._fixtures.tensors import tensor_factory  # noqa: F401
from lucid.test._fixtures.perf import bench  # noqa: F401

_GOVERNOR = _memory.Governor()


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Reclaim before the machine runs out, and skip only if it still has.

    See :mod:`lucid.test._memory` for the measurements this is sized
    from.  The short version: the suite was being ``SIGKILL``-ed by
    jetsam partway through the model zoo, and two thirds of what it was
    holding was a reclaimable MLX buffer cache.
    """
    reason = _GOVERNOR.before_test(item.nodeid)
    if reason is not None:
        pytest.skip(reason)


def pytest_runtest_teardown(item: pytest.Item, nextitem: pytest.Item | None) -> None:
    _GOVERNOR.after_test()


def pytest_terminal_summary(terminalreporter: object) -> None:
    """Say what the governor did.  A run that checked less has to admit it."""
    for line in _GOVERNOR.summary():
        terminalreporter.write_line(line)  # type: ignore[attr-defined]


@pytest.fixture(autouse=True)
def _seed_per_test() -> None:
    """Seed Lucid's default RNG to 0 before every test for
    bit-reproducibility of any sampling op.

    Reference-framework seeding (when applicable) is handled inside
    parity tests via ``lucid.test._helpers.seeding.seed_all``.
    """
    lucid.manual_seed(0)
