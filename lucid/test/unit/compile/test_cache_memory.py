"""Cache LRU eviction + memory-growth bound tests.

The process-global ``ExecutableCache::session()`` is LRU-bounded
(default 32, overridable via ``LUCID_COMPILE_MAX_CACHE``).  Three
properties must hold for compile mode to be safe in long-running
services and shape-variable workloads (NLP variable-length sequences,
dataloader batching that includes a remainder batch, …):

  1. The session cache size never exceeds the LRU cap.
  2. Repeatedly compiling new signatures evicts the oldest entries
     rather than growing unboundedly.
  3. The memory the session cache's own entries keep pinned stays inside the
     byte cap (``LUCID_COMPILE_MAX_CACHE_BYTES``): an MPSGraphExecutable holds
     the intermediate buffers MPSGraph planned for it until it is released, in
     proportion to the tensors, which 32 entries do not bound (LCD-1000).

If either of the first two breaks, what dead modules leave in the cache grows
without bound; if the third does, it grows by 32 entries of any size.

What the byte cap does NOT bound: it drops only the session cache's reference.
A live ``CompiledModule`` kept every signature's executable in its own unbounded
``_cache`` (LCD-1008): one module over eight shapes pinned the same with the session
cap as without.  That dict is a ``SignatureCache`` now, with a byte cap and an entry
cap of its own (``test_signature_cache.py``), and the last test here is its
footprint, with the session cap alone as the control.  Still outside both caps: the
symbolic-batch path (one executable, specialised per runtime shape inside MPSGraph)
and a segmented step's per-segment caches.
"""

import json
import os
import subprocess
import sys
import textwrap
from collections.abc import Callable, Iterator

import pytest

import lucid
import lucid.nn as nn
from lucid._C import engine as _C_engine

from lucid.test._fixtures.devices import metal_available
from lucid.test.unit.compile._helpers import (
    COMPILE_DEVICE,
    UNUSABLE_BYTE_COUNTS,
    count_id,
    metal_tensor,
)

# Compiled forwards under grad mode run eagerly (LCD-582); these test them.
pytestmark = pytest.mark.usefixtures("inference")


def _mlp() -> nn.Module:
    class _M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(8, 16)
            self.fc2 = nn.Linear(16, 4)

        def forward(self, x: lucid.Tensor) -> lucid.Tensor:
            return self.fc2(self.fc1(x).relu())

    return _M().to(COMPILE_DEVICE)


def test_session_cache_clears_explicitly() -> None:
    """``session_cache_clear`` drops every entry from the C++ cache."""
    cm = lucid.compile(_mlp())
    for b in (2, 4, 8):
        cm(metal_tensor(b, 8))
    # The cache may also contain entries from earlier tests in the
    # session — we just need to verify the clear knob works.
    _C_engine.compile.session_cache_clear()
    assert _C_engine.compile.session_cache_size() == 0


def test_session_cache_bounded_by_lru_cap() -> None:
    """Compiling > LRU-cap distinct signatures triggers eviction.

    The default cap is 32.  We force 50 distinct signatures and
    confirm the cache never exceeds the cap — proves the eviction
    path is wired in.
    """
    _C_engine.compile.session_cache_clear()
    cm = lucid.compile(_mlp())

    # Each batch size produces a fresh signature; 50 > default cap 32.
    for b in range(1, 51):
        cm(metal_tensor(b, 8))

    final = _C_engine.compile.session_cache_size()
    assert final <= 32, (
        f"session cache exceeded LRU cap: {final} > 32. "
        f"The cache will grow unboundedly under shape-variable workloads."
    )


def test_session_cache_lru_eviction_order() -> None:
    """After overflowing the cap, the *earliest* signatures are gone.

    Compile signatures 1..50, then re-call on signature 1.  If LRU is
    correct, signature 1's executable was evicted (oldest) and the
    re-call triggers a recompile rather than a hit.  We detect the
    recompile via a fresh ``compile_ms > 0`` timing entry.
    """
    _C_engine.compile.session_cache_clear()
    cm = lucid.compile(_mlp())

    # Step 1: prime the cache to overflow.
    for b in range(1, 51):
        cm(metal_tensor(b, 8))

    # The Python-side cache is unbounded today, so cm._cache has all
    # 50 entries.  The *C++* session cache is the bounded layer.
    # Clear the Python cache so the next call definitely re-enters
    # the C++ compile_or_cached path.
    cm.clear_cache()
    _ = cm(metal_tensor(1, 8))
    # If sig-1 had been retained in the C++ cache, this would be a
    # hit (compile_ms ≈ 0 for cache hits).  After eviction it's a
    # fresh compile (compile_ms > 0).
    timing = cm.timing()
    assert timing, "expected at least one timing entry after compile"
    assert timing[-1]["compile_ms"] > 0.5, (
        f"expected fresh compile (>0.5ms) for evicted signature, got "
        f"{timing[-1]['compile_ms']:.3f}ms — eviction may not be working"
    )


def test_env_var_lru_cap_documented() -> None:
    """``LUCID_COMPILE_MAX_CACHE`` env var is honoured by ``session()``.

    We can't change it mid-run (it's read on first cache access), but
    we can document the contract — the implementation in
    ``ExecutableCache::session()`` reads ``getenv`` exactly once.
    The user-guide doc should reference this knob.
    """
    # The env-var binding is a one-time read; we just check the size
    # API exists so the doc reference doesn't go stale.
    assert hasattr(_C_engine.compile, "session_cache_size")
    assert hasattr(_C_engine.compile, "session_cache_clear")


# ── the byte cap (LCD-1000) ─────────────────────────────────────────────────

_MB = 1024 * 1024


def _skip_if_nothing_is_pinned(grew_mb: float, graphs: int, pinned_per_graph_mb: float) -> None:
    """Skip a footprint test whose uncapped control did not grow: there is nothing to bound.

    The hosted macOS runner's virtual GPU does not pin MPSGraph intermediates.  On
    2026-11-14 the control of the footprint tests below saw 6-8 MB of growth for eight
    graphs there, where Apple Silicon hardware pins about 150 MB.  On hardware that pins,
    the control still has to hold, and the cap is asserted against it.  The child gave the
    control 15 s to grow before it reported (``_footprint``: a baseline still draining
    reads the control small, which is not a GPU that pins nothing; LCD-1112).
    """
    if grew_mb <= graphs * pinned_per_graph_mb / 2:
        pytest.skip(
            "this GPU does not pin MPSGraph intermediates, so there is no footprint for the "
            f"cap to bound (the uncapped control grew {grew_mb:.0f} MB)"
        )


@pytest.fixture
def byte_cap() -> Iterator[Callable[[int], int]]:
    """An empty session cache with no byte cap; the setter, then the old cap restored."""
    previous = _C_engine.compile.session_cache_set_max_bytes(0)
    _C_engine.compile.session_cache_clear()
    yield _C_engine.compile.session_cache_set_max_bytes
    _C_engine.compile.session_cache_set_max_bytes(previous)
    _C_engine.compile.session_cache_clear()


def _entry_bytes(cm: lucid.nn.Module, batches: list[int]) -> list[int]:
    """The estimate each ``batches`` entry adds, compiled in turn into an uncapped cache."""
    sizes: list[int] = []
    before = _C_engine.compile.session_cache_bytes()
    for b in batches:
        cm(metal_tensor(b, 8))
        now = _C_engine.compile.session_cache_bytes()
        sizes.append(now - before)
        before = now
    return sizes


def test_session_cache_counts_what_its_entries_hold(byte_cap: Callable[[int], int]) -> None:
    """A bigger graph adds more to ``session_cache_bytes``, and ``clear`` takes it all back."""
    sizes = _entry_bytes(lucid.compile(_mlp()), [16, 64])
    assert 0 < sizes[0] < sizes[1], sizes
    assert _C_engine.compile.session_cache_bytes() == sum(sizes)
    _C_engine.compile.session_cache_clear()
    assert _C_engine.compile.session_cache_bytes() == 0


def test_byte_cap_evicts_the_oldest_entries_first(byte_cap: Callable[[int], int]) -> None:
    """Over the cap the oldest go, until the newest ones that fit are what is left."""
    batches = [16, 32, 48, 64, 80, 96]
    sizes = _entry_bytes(lucid.compile(_mlp()), batches)
    _C_engine.compile.session_cache_clear()

    cap = sizes[-1] + sizes[-2] + 1  # the two newest fit, a third does not
    byte_cap(cap)
    _entry_bytes(lucid.compile(_mlp()), batches)

    assert _C_engine.compile.session_cache_size() == 2
    assert _C_engine.compile.session_cache_bytes() == sizes[-1] + sizes[-2] <= cap


def test_the_newest_entry_stays_when_it_alone_is_over_the_byte_cap(
    byte_cap: Callable[[int], int],
) -> None:
    """The executable just built is the one about to run: a tiny cap must not drop it."""
    byte_cap(1)
    cm = lucid.compile(_mlp())
    cm(metal_tensor(16, 8))
    assert _C_engine.compile.session_cache_size() == 1
    cm(metal_tensor(32, 8))
    assert _C_engine.compile.session_cache_size() == 1
    assert _C_engine.compile.session_cache_bytes() > 1


_CHILD_ENV = {
    k: v
    for k, v in os.environ.items()
    if not k.startswith(("LUCID_COMPILE_MAX_CACHE", "LUCID_COMPILE_MAX_SIGNATURE"))
}

# What MPSGraph pins, measured in a process of its own: the floor is then the Metal
# set-up of one graph, not whatever the tests before this one left to release.  A released
# executable's memory comes back to the footprint late, by 2-13 s under load, so a reading
# waits for the condition the assertion needs (``_footprint.Footprints``, LCD-1112) rather
# than for a clock: the control for what grew, the capped run for being under half of it.
_FOOTPRINT_CHILD = textwrap.dedent("""
    import json

    import lucid
    from lucid._C import engine as E
    from lucid.test.unit.compile._footprint import GRAPHS as N
    from lucid.test.unit.compile._footprint import LEAD, PINNED_MB, Footprints


    def widen(x):
        # A graph that must write its 2x broadcast out: MPSGraph pins that buffer.
        return x.repeat(1, 1, 1, 2).reshape(-1).sum()


    def run_graphs(count):
        for i in range(count):
            # Made on the device: randn(...).to leaves three host copies of the input in
            # malloc's cache, 25 MB each round, which would bury what is measured here.
            x = lucid.randn(LEAD + 32 * i, 8, 16, 16, device="metal")
            lucid.compile(widen)(x).item()
            del x


    def grew(reading, floor):
        return reading - floor > N * PINNED_MB / 2


    fp = Footprints()
    with lucid.no_grad():
        E.compile.session_cache_set_max_bytes(0)
        # The first graph of a process also sets up Metal, which stays: out of the baseline.
        run_graphs(1)
        E.compile.session_cache_clear()
        fp.settle()

        run_graphs(N)
        held = fp.settle(until=grew, timeout_s=15.0)
        entry = E.compile.session_cache_bytes() // N

        E.compile.session_cache_clear()
        E.compile.session_cache_set_max_bytes(entry * 5 // 2)
        run_graphs(N)
        capped = fp.settle(until=lambda reading, floor: reading - floor < (held - floor) / 2)
    print(json.dumps({
        "uncapped": held - fp.floor,
        "capped": capped - fp.floor,
        "entries": E.compile.session_cache_size(),
        "pinned_per_graph_mb": PINNED_MB,
        "graphs": N,
    }))
    """)


@pytest.mark.parametrize("grew_mb", [6.0, 8.0, 64.0])
def test_a_control_that_did_not_grow_skips_instead_of_failing(grew_mb: float) -> None:
    """The hosted runner's GPU grew 6-8 MB over eight 16 MB graphs (2026-11-14): skip, not fail."""
    with pytest.raises(pytest.skip.Exception, match="does not pin MPSGraph intermediates"):
        _skip_if_nothing_is_pinned(grew_mb, graphs=8, pinned_per_graph_mb=16.0)


@pytest.mark.parametrize("grew_mb", [128.0, 150.0])
def test_a_control_that_grew_is_asserted_against(grew_mb: float) -> None:
    """Hardware that pins (about 150 MB for eight graphs): the control passes, the cap is judged."""
    _skip_if_nothing_is_pinned(grew_mb, graphs=8, pinned_per_graph_mb=16.0)


@pytest.mark.slow
@pytest.mark.skipif(not metal_available(), reason="metal unavailable")
def test_footprint_stays_inside_the_byte_cap() -> None:
    """Distinct graphs pin memory until their executables go; the byte cap lets it go.

    Eight graphs each holding a 128 MB broadcast: with no byte cap the process keeps all
    eight, with a cap of two entries it keeps two.  The uncapped run is the control: where
    it does not grow (a GPU that pins nothing) the test skips rather than pass vacuously.
    Readings are taken the way ``_footprint`` settles them (LCD-1112): on a loaded Studio a
    released executable's memory came back 2-13 s late, or at the next GPU work.
    """
    done = subprocess.run(
        [sys.executable, "-c", _FOOTPRINT_CHILD],
        env=_CHILD_ENV,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    got = json.loads(done.stdout.strip().splitlines()[-1])
    uncapped, capped = got["uncapped"], got["capped"]

    assert got["entries"] == 2
    _skip_if_nothing_is_pinned(uncapped, got["graphs"], got["pinned_per_graph_mb"])
    assert capped < uncapped / 2, f"capped {capped:.0f} MB vs uncapped {uncapped:.0f} MB"


# ``LUCID_COMPILE_MAX_CACHE_BYTES`` is read when the session cache is first made, so
# each value needs a process of its own.
_CAP_CHILD = "from lucid._C import engine as E; print(E.compile.session_cache_set_max_bytes(0))"


def _initial_cap(value: str | None) -> tuple[int, str]:
    """The byte cap a fresh process starts with under ``value``, and what it said on stderr."""
    env = dict(_CHILD_ENV)
    if value is not None:
        env["LUCID_COMPILE_MAX_CACHE_BYTES"] = value
    done = subprocess.run(
        [sys.executable, "-c", _CAP_CHILD], env=env, capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, done.stderr[-2000:]
    return int(done.stdout.strip().splitlines()[-1]), done.stderr


@pytest.mark.parametrize(
    ("text", "bytes_"),
    [("0", 0), ("123456", 123456), ("64k", 64 * 1024), ("64m", 64 * _MB), ("2G", 2 * 1024**3)],
)
def test_byte_cap_environment_value_is_honoured(text: str, bytes_: int) -> None:
    """A number of bytes, with an optional k / m / g suffix; ``0`` turns the cap off."""
    got, said = _initial_cap(text)
    assert got == bytes_
    assert "ignoring" not in said


@pytest.mark.parametrize("text", UNUSABLE_BYTE_COUNTS, ids=count_id)
def test_an_unusable_byte_cap_falls_back_to_the_default_and_says_so(text: str) -> None:
    """``1g`` read as one byte would evict every entry but the newest: reject, do not guess."""
    default, quiet = _initial_cap(None)
    assert default > 0 and "ignoring" not in quiet
    got, said = _initial_cap(text)
    assert got == default
    assert "ignoring LUCID_COMPILE_MAX_CACHE_BYTES" in said


# One live module over many shapes (LCD-1008).  The module's own cap is read when the
# module is made, so each arm sets it first.  Readings wait as the byte-cap child's do: the
# arm that ends a module's life leaves its memory to come back, which the next arm's baseline
# may still carry (the control read 0 when it did), so every reading is judged against the
# lowest footprint seen.
_LIVE_MODULE_CHILD = textwrap.dedent("""
    import json
    import os

    import lucid
    from lucid._C import engine as E
    from lucid.test.unit.compile._footprint import GRAPHS as N
    from lucid.test.unit.compile._footprint import LEAD, PINNED_MB, Footprints

    MODULE_BYTES = "LUCID_COMPILE_MAX_SIGNATURE_BYTES"


    def widen(x):
        return x.repeat(1, 1, 1, 2).reshape(-1).sum()


    def drive(cm):
        for i in range(N):
            # Made on the device: randn(...).to leaves host copies in malloc's cache.
            x = lucid.randn(LEAD + 32 * i, 8, 16, 16, device="metal")
            cm(x).item()
            del x


    def grew(reading, floor):
        return reading - floor > N * PINNED_MB / 2


    fp = Footprints()


    def arm(session_cap, module_cap, until, timeout_s):
        E.compile.session_cache_clear()
        E.compile.session_cache_set_max_bytes(session_cap)
        os.environ[MODULE_BYTES] = str(module_cap)
        fp.settle()
        cm = lucid.compile(widen)
        drive(cm)
        reading = fp.settle(until=until, timeout_s=timeout_s)
        entries = cm.cache_info()["entries"]
        del cm
        E.compile.session_cache_clear()
        return reading, entries


    with lucid.no_grad():
        # The first graph of a process also sets up Metal, which stays; the same pass
        # tells what the module's estimate of one entry is.
        E.compile.session_cache_set_max_bytes(0)
        os.environ[MODULE_BYTES] = "0"
        warm = lucid.compile(widen)
        drive(warm)
        entry = warm._cache.retained_bytes // N
        del warm

        cap = entry * 5 // 2
        # The session cache capped to two entries, the module not: the report of LCD-1008.
        session_only, session_only_entries = arm(cap, 0, grew, 15.0)
        # Both capped to two entries.
        both, both_entries = arm(
            cap,
            cap,
            lambda reading, floor: reading - floor < (session_only - floor) / 2,
            40.0,
        )
    print(json.dumps({
        "session_only": session_only - fp.floor,
        "session_only_entries": session_only_entries,
        "both": both - fp.floor,
        "both_entries": both_entries,
        "pinned_per_graph_mb": PINNED_MB,
        "graphs": N,
    }))
    """)


@pytest.mark.slow
@pytest.mark.skipif(not metal_available(), reason="metal unavailable")
def test_a_live_modules_footprint_stays_inside_its_own_cap() -> None:
    """One live module over eight shapes: bounded by its cap, not by the session cache's.

    Both runs cap the session cache to two entries, as the LCD-1008 reviewer did.  The
    module with no cap of its own keeps all eight executables and their pins (the control:
    where they do not grow, a GPU that pins nothing, the test skips rather than pass
    vacuously); with a cap of two entries it keeps two.
    """
    done = subprocess.run(
        [sys.executable, "-c", _LIVE_MODULE_CHILD],
        env=_CHILD_ENV,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    got = json.loads(done.stdout.strip().splitlines()[-1])

    assert got["session_only_entries"] == got["graphs"]
    assert got["both_entries"] == 2
    _skip_if_nothing_is_pinned(got["session_only"], got["graphs"], got["pinned_per_graph_mb"])
    assert (
        got["both"] < got["session_only"] / 2
    ), f"capped {got['both']:.0f} MB vs session-capped-only {got['session_only']:.0f} MB"
