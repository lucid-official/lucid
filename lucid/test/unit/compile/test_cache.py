"""ExecutableCache + CompiledModule cache state tests.

Three behaviours that **cannot regress** if compile mode is to stay
production-safe on long-running services / data-aug pipelines:

  1. **Identical signatures hit the cache** — calling the same model
     on the same shape/dtype must produce ``entries=1, n_calls=N``.

  2. **Distinct signatures create distinct entries, bounded by the
     LRU cap** — varying batch dim across a sweep populates per-sig
     entries; once past ``LUCID_COMPILE_MAX_CACHE`` (default 32)
     the oldest signature is evicted.

  3. **Clearing cache + lifecycle hooks** — ``train()`` /
     ``load_state_dict()`` / ``clear_cache()`` must drop the cache
     so stale executables don't outlive the model state they were
     compiled against.
"""

import pytest

import lucid
import lucid.nn as nn

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


def test_same_signature_hits_cache() -> None:
    """Same shape across N calls ⇒ exactly 1 cache entry, N hits."""
    cm = lucid.compile(_mlp())
    x = metal_tensor(4, 8)
    for _ in range(5):
        cm(x)
    info = cm.cache_info()
    assert info["entries"] == 1, info
    # n_calls dict has exactly one key with count 5.
    assert sum(info["n_calls"].values()) == 5


def test_distinct_shapes_create_distinct_entries() -> None:
    """Varying batch dim across N shapes ⇒ N cache entries."""
    cm = lucid.compile(_mlp())
    for b in (2, 4, 8, 16):
        x = metal_tensor(b, 8)
        cm(x)
    info = cm.cache_info()
    assert info["entries"] == 4, info


def _mul_add(a: lucid.Tensor, b: lucid.Tensor) -> lucid.Tensor:
    return a * b + 1.0


def test_aliased_and_distinct_inputs_get_their_own_executables() -> None:
    """``f(x, x)`` reads one tensor twice; ``f(a, b)`` reads two.

    Both trace to the same ops at the same shapes, and both caches keyed on
    that alone.  A module compiled on ``(x, x)`` answered ``(a, b)`` with
    ``a * a + 1`` from its own cache, and a fresh module compiled afterwards
    got the same wrong answer from the session cache.
    """
    x, a, b = metal_tensor(2, 8), metal_tensor(2, 8), metal_tensor(2, 8)
    want = (a * b + 1.0).to("cpu")

    same = lucid.compile(_mul_add)
    same(x, x)
    assert bool(lucid.allclose(same(a, b).to("cpu"), want))
    assert same.cache_info()["entries"] == 2

    fresh = lucid.compile(_mul_add)
    assert bool(lucid.allclose(fresh(a, b).to("cpu"), want))


def test_distinct_then_aliased_inputs_both_compile() -> None:
    """The other order fell back to eager for good.

    The session cache handed the aliased trace a two-feed executable, the
    feed binding refused it, and the signature was marked eager-only.
    """
    a, b, x = metal_tensor(2, 8), metal_tensor(2, 8), metal_tensor(2, 8)
    lucid.compile(_mul_add)(a, b)
    aliased = lucid.compile(_mul_add)
    out = aliased(x, x)
    assert bool(lucid.allclose(out.to("cpu"), (x * x + 1.0).to("cpu")))
    info = aliased.cache_info()
    assert info["entries"] == 1 and not info["eager_only"], info


def test_train_eval_flip_clears_cache() -> None:
    """Toggling ``train()`` / ``eval()`` invalidates compiled graphs.

    BN / Dropout codepaths diverge between modes; the safe default is
    to drop the cache so eval-mode call doesn't reuse a training-mode
    executable (which would silently apply dropout, etc.).
    """
    cm = lucid.compile(_mlp())
    cm.eval()
    cm(metal_tensor(4, 8))
    assert cm.cache_info()["entries"] == 1
    cm.train()
    assert cm.cache_info()["entries"] == 0  # cleared
    cm(metal_tensor(4, 8))
    assert cm.cache_info()["entries"] == 1


def test_load_state_dict_clears_cache() -> None:
    """Loading new weights drops the cache.

    The compiled graph captures parameter buffer identities; replacing
    the values is safe but pulling in a different state-dict shape /
    dtype would not be.  The safe default is to clear.
    """
    cm = lucid.compile(_mlp())
    cm(metal_tensor(4, 8))
    assert cm.cache_info()["entries"] == 1
    cm.load_state_dict(cm.state_dict())
    assert cm.cache_info()["entries"] == 0


def test_clear_cache_explicit() -> None:
    """``clear_cache()`` empties everything regardless of state."""
    cm = lucid.compile(_mlp())
    for b in (2, 4):
        cm(metal_tensor(b, 8))
    assert cm.cache_info()["entries"] == 2
    cm.clear_cache()
    assert cm.cache_info()["entries"] == 0
    assert sum(cm.cache_info()["n_calls"].values()) == 0


def test_timing_breakdown_populated() -> None:
    """``timing()`` returns per-signature compile + run cost."""
    cm = lucid.compile(_mlp())
    cm(metal_tensor(4, 8))
    rows = cm.timing()
    assert len(rows) == 1
    row = rows[0]
    assert row["compile_ms"] > 0.0, "compile_ms should be measurable"
    assert row["last_run_ms"] >= 0.0
    assert row["n_hits"] == 1


def test_lru_cap_default_32() -> None:
    """Default LRU cap is 32 entries (per ExecutableCache.h).

    This sweep populates 35 distinct signatures; the CompiledModule's
    Python-side cache holds them all (it's a dict, not LRU), but the
    *process-global* C++ ``ExecutableCache::session()`` is LRU-bounded
    — so the timing fingerprints stay coherent.  The Python-side cap
    is a different layer (Phase 1.6 deferral); this test pins the
    behaviour we currently have.
    """
    cm = lucid.compile(_mlp())
    for b in range(1, 36):
        cm(metal_tensor(b, 8))
    info = cm.cache_info()
    # Python-side cache is unbounded today; 35 entries should all be
    # present.  If this changes, the test should be updated to reflect
    # whatever policy lands.
    assert info["entries"] == 35
