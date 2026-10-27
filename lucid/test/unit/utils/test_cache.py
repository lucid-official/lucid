"""Unit tests for ``lucid.utils.cache`` — DynamicCache / EncoderDecoderCache.

Covers the HF-parity surface (update / get_seq_length / to_legacy_cache /
from_legacy_cache / reorder_cache / seen_tokens) plus the cache-aware
``nn.MultiheadAttention`` path (incremental decoding must equal a full pass).
"""

import lucid
import lucid.nn as nn
from lucid.test._helpers.compare import assert_close
from lucid.utils.cache import Cache, DynamicCache, EncoderDecoderCache


def _kv(seq: int, b: int = 1, h: int = 2, d: int = 4) -> tuple[lucid.Tensor, lucid.Tensor]:
    return lucid.randn(b, h, seq, d), lucid.randn(b, h, seq, d)


class TestDynamicCache:
    def test_is_a_cache(self) -> None:
        assert isinstance(DynamicCache(), Cache)

    def test_seed_then_grow_along_seq_dim(self) -> None:
        cache = DynamicCache()
        k0, v0 = _kv(3)
        fk, fv = cache.update(k0, v0, layer_idx=0)
        assert tuple(fk.shape) == (1, 2, 3, 4)
        assert cache.get_seq_length() == 3
        assert cache.seen_tokens == 3

        k1, v1 = _kv(1)
        fk, fv = cache.update(k1, v1, layer_idx=0)
        assert tuple(fk.shape) == (1, 2, 4, 4)  # grew along dim -2
        assert cache.get_seq_length() == 4
        assert cache.seen_tokens == 4

    def test_update_returns_full_tensors(self) -> None:
        cache = DynamicCache()
        k0, v0 = _kv(2)
        cache.update(k0, v0, 0)
        k1, v1 = _kv(1)
        fk, _ = cache.update(k1, v1, 0)
        # first two rows are the seed, last row is the new token
        assert_close(fk[:, :, :2, :], k0)
        assert_close(fk[:, :, 2:, :], k1)

    def test_multi_layer_independent(self) -> None:
        cache = DynamicCache()
        cache.update(*_kv(2), layer_idx=0)
        cache.update(*_kv(2), layer_idx=1)
        cache.update(*_kv(1), layer_idx=0)
        assert len(cache) == 2
        assert cache.get_seq_length(0) == 3
        assert cache.get_seq_length(1) == 2

    def test_dunder_protocol(self) -> None:
        cache = DynamicCache()
        k, v = _kv(2)
        cache.update(k, v, 0)
        gk, gv = cache[0]
        assert_close(gk, k)
        pairs = list(cache)
        assert len(pairs) == 1 == len(cache)

    def test_empty_cache_seq_length_zero(self) -> None:
        cache = DynamicCache()
        assert cache.get_seq_length() == 0
        assert cache.get_seq_length(5) == 0
        assert len(cache) == 0

    def test_get_max_cache_shape_and_usable_length(self) -> None:
        cache = DynamicCache()
        cache.update(*_kv(4), layer_idx=0)
        assert cache.get_max_cache_shape() is None  # unbounded
        # unbounded → usable length is just the current length
        assert cache.get_usable_length(10, 0) == 4

    def test_legacy_roundtrip(self) -> None:
        cache = DynamicCache()
        cache.update(*_kv(3), layer_idx=0)
        cache.update(*_kv(3), layer_idx=1)
        legacy = cache.to_legacy_cache()
        assert len(legacy) == 2
        assert len(legacy[0]) == 2  # (key, value)

        rebuilt = DynamicCache.from_legacy_cache(legacy)
        assert rebuilt.get_seq_length(0) == 3
        assert rebuilt.get_seq_length(1) == 3
        assert_close(rebuilt.key_cache[0], cache.key_cache[0])

    def test_from_legacy_none_is_empty(self) -> None:
        assert len(DynamicCache.from_legacy_cache(None)) == 0

    def test_reorder_cache(self) -> None:
        cache = DynamicCache()
        k = lucid.randn(2, 2, 3, 4)
        v = lucid.randn(2, 2, 3, 4)
        cache.update(k, v, 0)
        beam = lucid.tensor([1, 0]).long()
        cache.reorder_cache(beam)
        assert_close(cache.key_cache[0][0], k[1])
        assert_close(cache.key_cache[0][1], k[0])

    def test_crop(self) -> None:
        cache = DynamicCache()
        cache.update(*_kv(5), layer_idx=0)
        cache.update(*_kv(5), layer_idx=1)
        cache.crop(3)
        assert cache.get_seq_length(0) == 3
        assert cache.get_seq_length(1) == 3
        assert cache.seen_tokens == 3
        assert tuple(cache.key_cache[0].shape) == (1, 2, 3, 4)
        cache.crop(10)  # at/above current length is a no-op
        assert cache.get_seq_length() == 3

    def test_crop_negative(self) -> None:
        cache = DynamicCache()
        cache.update(*_kv(5), layer_idx=0)
        cache.crop(-2)  # drop 2 off the end
        assert cache.get_seq_length() == 3

    def test_reset(self) -> None:
        cache = DynamicCache()
        cache.update(*_kv(4), layer_idx=0)
        cache.reset()
        assert len(cache) == 0
        assert cache.get_seq_length() == 0
        assert cache.seen_tokens == 0
        cache.update(*_kv(2), layer_idx=0)  # reusable
        assert cache.get_seq_length() == 2

    def test_batch_repeat_interleave(self) -> None:
        cache = DynamicCache()
        cache.update(*_kv(3), layer_idx=0)  # batch=1
        cache.batch_repeat_interleave(4)
        assert tuple(cache.key_cache[0].shape) == (4, 2, 3, 4)

    def test_batch_select_indices(self) -> None:
        cache = DynamicCache()
        k = lucid.randn(2, 2, 3, 4)
        v = lucid.randn(2, 2, 3, 4)
        cache.update(k, v, 0)
        cache.batch_select_indices(lucid.tensor([1, 0]).long())
        assert_close(cache.key_cache[0][0], k[1])
        assert_close(cache.key_cache[0][1], k[0])


class TestEncoderDecoderCache:
    def test_is_a_cache(self) -> None:
        edc = EncoderDecoderCache(DynamicCache(), DynamicCache())
        assert isinstance(edc, Cache)

    def test_self_grows_cross_fixed(self) -> None:
        edc = EncoderDecoderCache(DynamicCache(), DynamicCache())
        # self-attention grows
        edc.update(*_kv(1), layer_idx=0)
        edc.update(*_kv(1), layer_idx=0)
        assert edc.get_seq_length() == 2
        # cross-attention written once
        ck, cv = _kv(7)
        edc.cross_attention_cache.update(ck, cv, 0)
        edc.is_updated[0] = True
        assert edc.cross_attention_cache.get_seq_length(0) == 7
        assert edc.is_updated == {0: True}

    def test_getitem_returns_four(self) -> None:
        edc = EncoderDecoderCache(DynamicCache(), DynamicCache())
        edc.self_attention_cache.update(*_kv(2), layer_idx=0)
        edc.cross_attention_cache.update(*_kv(5), layer_idx=0)
        sk, sv, xk, xv = edc[0]
        assert tuple(sk.shape)[2] == 2
        assert tuple(xk.shape)[2] == 5

    def test_legacy_roundtrip(self) -> None:
        edc = EncoderDecoderCache(DynamicCache(), DynamicCache())
        edc.self_attention_cache.update(*_kv(2), layer_idx=0)
        edc.cross_attention_cache.update(*_kv(5), layer_idx=0)
        legacy = edc.to_legacy_cache()
        assert len(legacy[0]) == 4
        rebuilt = EncoderDecoderCache.from_legacy_cache(legacy)
        assert rebuilt.get_seq_length(0) == 2
        assert rebuilt.cross_attention_cache.get_seq_length(0) == 5

    def test_crop_self_only(self) -> None:
        edc = EncoderDecoderCache(DynamicCache(), DynamicCache())
        edc.self_attention_cache.update(*_kv(4), layer_idx=0)
        edc.cross_attention_cache.update(*_kv(7), layer_idx=0)
        edc.crop(2)
        assert edc.get_seq_length() == 2  # self-attn truncated
        assert edc.cross_attention_cache.get_seq_length() == 7  # cross untouched

    def test_batch_ops_and_iter(self) -> None:
        edc = EncoderDecoderCache(DynamicCache(), DynamicCache())
        edc.self_attention_cache.update(lucid.randn(1, 2, 3, 4), lucid.randn(1, 2, 3, 4), 0)
        edc.cross_attention_cache.update(lucid.randn(1, 2, 5, 4), lucid.randn(1, 2, 5, 4), 0)
        edc.batch_repeat_interleave(2)
        assert tuple(edc.self_attention_cache.key_cache[0].shape)[0] == 2
        assert tuple(edc.cross_attention_cache.key_cache[0].shape)[0] == 2
        assert len(list(edc)) == 1  # one layer
        edc.reset()
        assert len(edc.self_attention_cache) == 0
        assert edc.is_updated == {}


class TestMultiheadAttentionCache:
    def test_incremental_self_attention_equals_full(self) -> None:
        lucid.manual_seed(0)
        mha = nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True).eval()
        x = lucid.randn(1, 5, 16)

        causal = (1.0 - lucid.tril(lucid.ones((5, 5)))) * -1e4
        full, _ = mha(x, x, x, attn_mask=causal, need_weights=False)

        cache = DynamicCache()
        steps = []
        for t in range(5):
            step = x[:, t : t + 1, :]
            out, _ = mha(
                step,
                step,
                step,
                need_weights=False,
                use_cache=True,
                past_key_value=cache,
                layer_idx=0,
            )
            steps.append(out[:, -1, :])
        inc = lucid.stack(steps, dim=1)
        assert cache.get_seq_length() == 5
        assert_close(inc, full, atol=1e-4)
