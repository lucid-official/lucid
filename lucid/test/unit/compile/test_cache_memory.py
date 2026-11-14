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
A live ``CompiledModule`` keeps every signature's executable in its own
unbounded ``_cache``, training-step and function-compile executables never
enter the session cache, and the dynamic-batch path bypasses it, so a
long-lived module fed many shapes still pins them all (LCD-1008).
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
from lucid.test.unit.compile._helpers import COMPILE_DEVICE, metal_tensor

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


_CHILD_ENV = {k: v for k, v in os.environ.items() if not k.startswith("LUCID_COMPILE_MAX_CACHE")}

# What MPSGraph pins, measured in a process of its own: the floor is then the Metal
# set-up of one graph, not whatever the tests before this one left to release.
_FOOTPRINT_CHILD = textwrap.dedent("""
    import gc
    import json
    import os
    import time

    import lucid
    from lucid._C import engine as E
    from lucid.test.audit._zoo_budget import footprint_gb

    N, LEAD = 8, 1024
    # What a release needs to reach the footprint: MPS keeps a retired heap about a
    # second and Metal returns released buffers in a step 1-2 s after the release.
    RELEASE_S = 3.0


    def widen(x):
        # A graph that must write its 2x broadcast out: MPSGraph pins that buffer.
        return x.repeat(1, 1, 1, 2).reshape(-1).sum()


    def settled_mb():
        gc.collect()
        lucid.metal.empty_cache()  # the freed inputs sit in MLX's cache, not in an executable
        time.sleep(RELEASE_S)
        return footprint_gb(os.getpid()) * 1024


    def run_graphs(count):
        for i in range(count):
            # Made on the device: randn(...).to leaves three host copies of the input in
            # malloc's cache, 25 MB each round, which would bury what is measured here.
            x = lucid.randn(LEAD + 32 * i, 8, 16, 16, device="metal")
            lucid.compile(widen)(x).item()
            del x


    with lucid.no_grad():
        E.compile.session_cache_set_max_bytes(0)
        # The first graph of a process also sets up Metal, which stays: out of the baseline.
        run_graphs(1)
        E.compile.session_cache_clear()
        base = settled_mb()

        run_graphs(N)
        uncapped = settled_mb() - base
        entry = E.compile.session_cache_bytes() // N

        E.compile.session_cache_clear()
        E.compile.session_cache_set_max_bytes(entry * 5 // 2)
        run_graphs(N)
        capped = settled_mb() - base
    print(json.dumps({
        "uncapped": uncapped,
        "capped": capped,
        "entries": E.compile.session_cache_size(),
        "pinned_per_graph_mb": 2 * LEAD * 8 * 16 * 16 * 4 / 2**20,
        "graphs": N,
    }))
    """)


@pytest.mark.slow
@pytest.mark.skipif(not metal_available(), reason="metal unavailable")
def test_footprint_stays_inside_the_byte_cap() -> None:
    """Distinct graphs pin memory until their executables go; the byte cap lets it go.

    Eight graphs each holding a 17 MB broadcast: with no byte cap the process keeps all
    eight, with a cap of two entries it keeps two.  The uncapped run is the control, so
    an MPSGraph that stops pinning fails here instead of passing vacuously.
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

    assert uncapped > got["graphs"] * got["pinned_per_graph_mb"] / 2, (
        f"uncapped growth {uncapped:.0f} MB: MPSGraph no longer pins the broadcast, "
        "so this test proves nothing"
    )
    assert got["entries"] == 2
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


@pytest.mark.parametrize(
    "text", ["", "-5", "1x", "abc", "1.5g", " 12", "12 ", "99999999999999999999", "99999999999g"]
)
def test_an_unusable_byte_cap_falls_back_to_the_default_and_says_so(text: str) -> None:
    """``1g`` read as one byte would evict every entry but the newest: reject, do not guess."""
    default, quiet = _initial_cap(None)
    assert default > 0 and "ignoring" not in quiet
    got, said = _initial_cap(text)
    assert got == default
    assert "ignoring LUCID_COMPILE_MAX_CACHE_BYTES" in said
