"""Cache LRU eviction + memory-growth bound tests.

The process-global ``ExecutableCache::session()`` is LRU-bounded
(default 32, overridable via ``LUCID_COMPILE_MAX_CACHE``).  Three
properties must hold for compile mode to be safe in long-running
services and shape-variable workloads (NLP variable-length sequences,
dataloader batching that includes a remainder batch, …):

  1. The session cache size never exceeds the LRU cap.
  2. Repeatedly compiling new signatures evicts the oldest entries
     rather than growing unboundedly.
  3. The memory its entries keep pinned stays inside the byte cap
     (``LUCID_COMPILE_MAX_CACHE_BYTES``): an MPSGraphExecutable holds the
     intermediate buffers MPSGraph planned for it until it is released, in
     proportion to the tensors, which 32 entries do not bound (LCD-1000).

If either of the first two breaks, a long-running service compiling a model
on a shape-variable input stream will OOM on the cumulative MPSGraph
executables; if the third does, it does so on 32 large ones.
"""

import gc
import os
import time
from collections.abc import Callable, Iterator

import pytest

import lucid
import lucid.nn as nn
from lucid._C import engine as _C_engine

from lucid.test.audit._zoo_budget import footprint_gb
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


def _widen(x: lucid.Tensor) -> lucid.Tensor:
    """A graph that must write its 2x broadcast out: MPSGraph pins that buffer."""
    return x.repeat(1, 1, 1, 2).reshape(-1).sum()


#: What a release needs to reach the footprint: MPS keeps a retired heap about a second and
#: Metal returns released buffers in a step that lands 1-2 s after the release.
_RELEASE_S = 3.0


def _settled_footprint_mb() -> float:
    gc.collect()
    lucid.metal.empty_cache()  # the freed inputs sit in MLX's cache, not in an executable
    time.sleep(_RELEASE_S)
    return footprint_gb(os.getpid()) * 1024


@pytest.mark.slow
def test_footprint_stays_inside_the_byte_cap(byte_cap: Callable[[int], int]) -> None:
    """Distinct graphs pin memory until their executables go; the byte cap lets it go.

    Eight graphs each holding a 17 MB broadcast: with no byte cap the process keeps all
    eight, with a cap of two entries it keeps two.  The uncapped run is the control, so
    an MPSGraph that stops pinning fails here instead of passing vacuously.
    """
    n, lead = 8, 1024

    def run_graphs(count: int) -> None:
        for i in range(count):
            # Made on the device: ``randn(...).to`` leaves three host copies of the input in
            # malloc's cache, 25 MB each round, which would bury what is measured here.
            x = lucid.randn(lead + 32 * i, 8, 16, 16, device=COMPILE_DEVICE)
            lucid.compile(_widen)(x).item()
            del x

    # The first graph of a process also sets up Metal, which stays: take it out of the
    # baseline.  The floor is read after the release of the cache that clearing starts.
    run_graphs(1)
    _C_engine.compile.session_cache_clear()
    base = _settled_footprint_mb()

    run_graphs(n)
    uncapped = _settled_footprint_mb() - base
    entry = _C_engine.compile.session_cache_bytes() // n

    _C_engine.compile.session_cache_clear()
    byte_cap(entry * 5 // 2)
    run_graphs(n)
    capped = _settled_footprint_mb() - base

    pinned_per_graph_mb = 2 * lead * 8 * 16 * 16 * 4 / _MB
    assert uncapped > n * pinned_per_graph_mb / 2, (
        f"uncapped growth {uncapped:.0f} MB: MPSGraph no longer pins the broadcast, "
        "so this test proves nothing"
    )
    assert _C_engine.compile.session_cache_size() == 2
    assert capped < uncapped / 2, f"capped {capped:.0f} MB vs uncapped {uncapped:.0f} MB"
