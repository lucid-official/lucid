"""Parity: ``lucid.utils.cache.DynamicCache`` vs the reference framework.

Validates that the cache's core accumulation matches the reference ``cat``
along the sequence axis, and — when the reference cache library is present —
that lucid's HF-mirrored ``DynamicCache`` is behaviourally identical (same
accumulated key/value, ``get_seq_length`` and ``to_legacy_cache`` shape),
confirming the API-portability goal.
"""

from typing import Any

import numpy as np
import pytest

import lucid
from lucid.test._helpers.compare import assert_close
from lucid.utils.cache import DynamicCache


@pytest.mark.parity
class TestDynamicCacheParity:
    def test_growth_matches_ref_cat(self, ref: Any) -> None:
        rng = np.random.default_rng(0)
        sizes = (3, 1, 1, 2)
        chunks = [rng.standard_normal((1, 2, s, 4)).astype(np.float32) for s in sizes]

        cache = DynamicCache()
        for c in chunks:
            cache.update(lucid.tensor(c), lucid.tensor(c), 0)

        ref_k = ref.cat([ref.tensor(c) for c in chunks], dim=-2)
        assert_close(cache.key_cache[0], ref_k, atol=0.0)
        assert cache.get_seq_length() == sum(sizes)

    def test_matches_reference_dynamic_cache(self, ref: Any) -> None:
        transformers = pytest.importorskip("transformers")
        ref_cache_cls = getattr(transformers, "DynamicCache", None)
        if ref_cache_cls is None:
            pytest.skip("reference framework has no DynamicCache")

        rng = np.random.default_rng(1)
        chunks = [rng.standard_normal((1, 2, s, 4)).astype(np.float32) for s in (3, 1, 1)]

        lucid_cache = DynamicCache()
        ref_cache = ref_cache_cls()
        for c in chunks:
            lucid_cache.update(lucid.tensor(c), lucid.tensor(c), 0)
            ref_cache.update(ref.tensor(c), ref.tensor(c), 0)

        # The reference moved its per-layer storage from ``key_cache[i]`` to
        # ``layers[i].keys`` and dropped ``to_legacy_cache`` (5.x).  Read either
        # form; a third one should fail here, not skip — a skip on "API differs"
        # left this comparison unrun for as long as nobody looked.
        if hasattr(ref_cache, "layers"):
            ref_key = ref_cache.layers[0].keys
            ref_layers = len(ref_cache.layers)
        else:
            ref_key = ref_cache.key_cache[0]
            ref_layers = len(ref_cache.to_legacy_cache())

        assert_close(lucid_cache.key_cache[0], ref_key, atol=0.0)
        assert lucid_cache.get_seq_length() == ref_cache.get_seq_length()
        assert len(lucid_cache.to_legacy_cache()) == ref_layers
