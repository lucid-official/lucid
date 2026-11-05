"""The device a test process defaults to, and whether Metal is there at all.

``lucid-audit --models`` verifies the zoo on Metal unless told otherwise, by
exporting ``LUCID_TEST_DEFAULT_DEVICE`` to every process it starts; the test
``conftest`` and the audit's probes apply it here.

This module imports nothing from pytest on purpose.  ``lucid-audit`` reaches
it from its own modules, and a tool whose suite stage degrades without pytest
must not need pytest to print ``--list-packages``.  The pytest fixtures
(``_fixtures/devices.py``) import from here, not the other way round.
"""

import contextlib
import functools
import itertools
import os
from collections.abc import Iterator

import lucid
import lucid.nn as nn

#: The environment variable ``lucid-audit --models`` sets for every process it
#: starts, and the one place a test process learns what device to default to.
DEFAULT_DEVICE_ENV = "LUCID_TEST_DEFAULT_DEVICE"

DEFAULT_DEVICES = ("cpu", "metal")

#: What the zoo is verified on when nobody says: models train on Metal.
DEFAULT_ZOO_DEVICE = "metal"


@functools.lru_cache(maxsize=1)
def metal_available() -> bool:
    """Return True if Apple Metal is usable — try a tiny allocation
    and let any failure bubble up as ``False``."""
    try:
        from lucid._C import engine as _C_engine

        _ = _C_engine.zeros([1], _C_engine.F32, _C_engine.GPU)
    except Exception:
        return False
    return True


def apply_default_device(name: str | None = None) -> str | None:
    """Make ``name`` — or, with none, ``$LUCID_TEST_DEFAULT_DEVICE`` — the default device.

    Models train on Metal, so the zoo is verified on Metal unless it is told
    otherwise.  Asking for a device the machine does not have is an error, not a
    fallback: running the zoo on CPU because Metal was missing would report a
    verification of something other than what was asked for.  Returns the device
    applied, or ``None`` when nothing was asked.
    """
    device = name if name is not None else os.environ.get(DEFAULT_DEVICE_ENV)
    if not device:
        return None
    if device not in DEFAULT_DEVICES:
        raise ValueError(f"unknown default device {device!r}; expected one of {DEFAULT_DEVICES}")
    if device == "metal" and not metal_available():
        raise RuntimeError(
            f"the default device is metal ({DEFAULT_DEVICE_ENV}, or lucid-audit --zoo-device), "
            "but this machine has no usable Metal device; pass --zoo-device cpu to verify "
            "on CPU on purpose"
        )
    lucid.set_default_device(device)
    return device


@contextlib.contextmanager
def default_device(name: str) -> Iterator[str]:
    """Make ``name`` the default device inside the block, then put the old one back.

    For a test that needs one device whatever the process defaults to, so that
    it says so where it does it and leaves no trace for the next test.
    """
    previous = lucid.get_default_device()
    lucid.set_default_device(name)
    try:
        yield name
    finally:
        lucid.set_default_device(previous)


def device_of(module: nn.Module) -> str:
    """The device ``module`` holds its tensors on, or the default device if it holds none.

    Inputs a test builds for a model go where the model is, not where the
    process defaults to: a model moved to the CPU still answers there under a
    Metal default.
    """
    held = next(itertools.chain(module.parameters(), module.buffers()), None)
    return lucid.get_default_device() if held is None else held.device.type
