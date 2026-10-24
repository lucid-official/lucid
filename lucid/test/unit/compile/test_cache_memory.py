"""Cache LRU eviction + memory-growth bound tests.

The process-global ``ExecutableCache::session()`` is LRU-bounded
(default 32, overridable via ``LUCID_COMPILE_MAX_CACHE``).  Two
properties must hold for compile mode to be safe in long-running
services and shape-variable workloads (NLP variable-length sequences,
dataloader batching that includes a remainder batch, …):

  1. The session cache size never exceeds the LRU cap.
  2. Repeatedly compiling new signatures evicts the oldest entries
     rather than growing unboundedly.

If either invariant breaks, a long-running service compiling a model
on a shape-variable input stream will OOM on the cumulative MPSGraph
executables.
"""

import pytest

import lucid
import lucid.nn as nn
from lucid._C import engine as _C_engine

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
    ``CompiledModule``'s docstring names this knob.
    """
    # The env-var binding is a one-time read; we just check the size
    # API exists so the doc reference doesn't go stale.
    assert hasattr(_C_engine.compile, "session_cache_size")
    assert hasattr(_C_engine.compile, "session_cache_clear")
