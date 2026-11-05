"""Per-directory pytest config for ``lucid/test/unit/models``.

Pending families declare an explicit module-level skip in their test file.
Directory collect_ignore is deliberately not used: pytest bypasses it for
explicit file arguments, which the memory-bounded audit runner supplies.
Explicit skips remain visible and activate when the family is implemented.

The default device
------------------
These tests follow the process's default device: the CPU unless
``lucid-audit --models`` (Metal by default) or ``LUCID_TEST_DEFAULT_DEVICE``
says otherwise.  Inputs go where the model is (``device_of``), not to a
device named in the test.  A test that needs one device whatever the
default is takes ``cpu_default`` or ``metal_default`` and says why, and no
test may leave the default other than it found it.
"""

import sys
from collections.abc import Callable, Iterator

import pytest

import lucid
from lucid.test._default_device import default_device, metal_available

type HostReads = Callable[[Callable[[], object]], list[str]]


@pytest.fixture(autouse=True)
def _the_default_device_is_left_as_found() -> Iterator[None]:
    before = lucid.get_default_device()
    yield
    after = lucid.get_default_device()
    lucid.set_default_device(before)
    assert after == before, f"the test left the default device at {after!r}, not {before!r}"


@pytest.fixture
def cpu_default() -> Iterator[None]:
    """The CPU as the default device for the test, then the old default back."""
    with default_device("cpu"):
        yield


@pytest.fixture
def metal_default() -> Iterator[None]:
    """Metal as the default device for the test, or a skip where there is none."""
    if not metal_available():
        pytest.skip("needs Metal")
    with default_device("metal"):
        yield


@pytest.fixture
def host_reads(monkeypatch: pytest.MonkeyPatch) -> HostReads:
    """Run a callable and return ``file:line`` of every ``.item()`` /
    ``.tolist()`` / ``.numpy()`` that model code under ``lucid/models`` made.

    Each is a device sync on Metal.  Reads made by the framework's own
    index guards are not the model's and are not counted.
    """
    seen: list[str] = []
    for name in ("item", "tolist", "numpy"):
        original = getattr(lucid.Tensor, name)

        def spy(t: lucid.Tensor, *a: object, _f=original, **k: object) -> object:
            caller = sys._getframe(1)
            if "lucid/models/" in caller.f_code.co_filename:
                seen.append(f"{caller.f_code.co_filename}:{caller.f_lineno}")
            return _f(t, *a, **k)

        monkeypatch.setattr(lucid.Tensor, name, spy)

    def count(run: Callable[[], object]) -> list[str]:
        seen.clear()
        run()
        return list(seen)

    return count
