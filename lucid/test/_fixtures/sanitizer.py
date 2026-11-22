"""Which sanitizer runtimes this process loaded, and the marker that acts on it (LCD-1056).

The nightly native job (``scripts/ci_sanitizer.sh``) runs the engine suites under ASan + UBSan.
AddressSanitizer holds every freed block in a quarantine (256 MB by default) and pads every
allocation with redzones, so a test that asserts on how much memory the *process* held --
a child's ``ru_maxrss``, ``phys_footprint`` -- measures the sanitizer, not Lucid.  Until
LCD-1047 the job stopped at its first UBSan report, before any of these ran; once it reached the
end, ``test_a_read_or_write_through_a_view_takes_no_full_size_snapshot[pickle]`` read 212 MB
against a 140 MB bound.

One answer, asked of the process itself.  :func:`loaded_sanitizers` lists the sanitizer runtime
images dyld mapped into this process (``libclang_rt.asan_osx_dynamic.dylib`` and its siblings).
The ASan build links its launcher against that dylib, so it is a load command of the main
executable, and the engine built with ``-fsanitize=...`` links its own.  The environment says
nothing reliable: ``ASAN_OPTIONS`` can be exported over a release engine, and an instrumented
engine can be run without it; ``LUCID_SANITIZE`` exists only at build time.

``@pytest.mark.requires_unsanitized_memory`` skips a test when :func:`inflates_memory` holds.
Only ASan (and TSan) inflate the footprint; a UBSan-only build keeps running it.  The marker goes
on a test whose assertion reads *process* memory.  It does not go on one that reads an allocator
counter -- ``memory_stats(...).current_bytes`` / ``peak_bytes``, ``metal_peak_memory`` -- because
those count the bytes the engine asked for, which a sanitizer does not change, and those tests
exercise the very code a sanitizer exists to run.  ``unit/test_unsanitized_memory_marker.py``
fails a test in the nightly target that reaches a process-memory reader without the marker.
"""

import ctypes
import re
from collections.abc import Iterable
from typing import Protocol

import pytest

MEMORY_MARKER = "requires_unsanitized_memory"
MEMORY_REASON = (
    "ASan quarantine and redzones inflate the footprint "
    "(sanitizer runtime loaded: {loaded}; LCD-1056)"
)

# Clang's runtime for each sanitizer, one dynamic library per runtime on Apple platforms.
_RUNTIME_IMAGE = re.compile(r"libclang_rt\.(?P<runtime>asan|ubsan|tsan)_\w+_dynamic\.dylib$")
_SANITIZER_OF_RUNTIME = {"asan": "address", "ubsan": "undefined", "tsan": "thread"}

# The sanitizers whose runtime changes how much memory the process holds.
_INFLATES_MEMORY = frozenset({"address", "thread"})


def _loaded_image_paths() -> list[str]:
    """Paths of every image dyld has mapped into this process, in load order."""
    libsystem = ctypes.CDLL(None)
    try:
        image_count = libsystem._dyld_image_count
        image_name = libsystem._dyld_get_image_name
    except AttributeError:  # not a Mach-O host: nothing here can be sanitized this way
        return []
    image_count.restype = ctypes.c_uint32
    image_name.argtypes = [ctypes.c_uint32]
    image_name.restype = ctypes.c_char_p
    names = (image_name(i) for i in range(image_count()))
    return [raw.decode(errors="replace") for raw in names if raw]


def sanitizers_in(paths: Iterable[str]) -> frozenset[str]:
    """The sanitizers (``address`` / ``undefined`` / ``thread``) whose runtime is among ``paths``."""
    found: set[str] = set()
    for path in paths:
        match = _RUNTIME_IMAGE.search(path)
        if match is not None:
            found.add(_SANITIZER_OF_RUNTIME[match["runtime"]])
    return frozenset(found)


def loaded_sanitizers() -> frozenset[str]:
    """The sanitizers whose runtime this process has loaded; empty for a release build."""
    return sanitizers_in(_loaded_image_paths())


def is_sanitized() -> bool:
    """Whether any sanitizer runtime is loaded (instrumentation slows every test down)."""
    return bool(loaded_sanitizers())


def inflates_memory() -> bool:
    """Whether a loaded sanitizer changes the process footprint (ASan, TSan; not UBSan)."""
    return not loaded_sanitizers().isdisjoint(_INFLATES_MEMORY)


class _Skippable(Protocol):
    """The two things of a collected test that the skip needs (``pytest.Item`` has both)."""

    def get_closest_marker(self, name: str) -> object | None: ...

    def add_marker(self, marker: pytest.MarkDecorator) -> None: ...


def skip_where_memory_is_inflated(items: Iterable[_Skippable]) -> None:
    """Skip every collected test that carries ``requires_unsanitized_memory``, under ASan/TSan.

    Called from the root conftest's ``pytest_collection_modifyitems``, so the skip shows in the
    collection summary and in ``-rs`` with its reason rather than as a late failure.
    """
    loaded = loaded_sanitizers()
    if loaded.isdisjoint(_INFLATES_MEMORY):
        return
    skip = pytest.mark.skip(reason=MEMORY_REASON.format(loaded=", ".join(sorted(loaded))))
    for item in items:
        if item.get_closest_marker(MEMORY_MARKER) is not None:
            item.add_marker(skip)
