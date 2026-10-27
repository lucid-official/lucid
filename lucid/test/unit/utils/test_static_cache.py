"""Unit tests for ``lucid.utils.cache.StaticCache`` (fixed-buffer KV cache).

The buffer is pre-allocated at ``max_cache_len`` and never grows; new tokens
are written in place at ``cache_position`` via the traceable ``index_copy``.
"""

import lucid
from lucid.utils.cache import Cache, StaticCache


def _ones(seq: int, val: float) -> tuple[lucid.Tensor, lucid.Tensor]:
    t = lucid.ones(1, 2, seq, 4) * val
    return t, t


class TestStaticCache:
    def test_is_a_cache(self) -> None:
        assert isinstance(StaticCache(max_cache_len=8), Cache)
        assert StaticCache(max_cache_len=8).get_max_cache_shape() == 8

    def test_fixed_shape_and_in_place_write(self) -> None:
        cache = StaticCache(max_cache_len=8)
        fk, _ = cache.update(*_ones(3, 1.0), layer_idx=0)  # prefill positions 0,1,2
        assert tuple(fk.shape) == (1, 2, 8, 4)  # full fixed buffer
        assert cache.get_seq_length() == 3
        assert cache.seen_tokens == 3
        # written prefix non-zero, unwritten tail zero
        assert float(fk[:, :, :3, :].sum().item()) == 3 * 2 * 4
        assert float(fk[:, :, 3:, :].sum().item()) == 0.0

        fk, _ = cache.update(*_ones(1, 2.0), layer_idx=0)  # decode at position 3
        assert tuple(fk.shape) == (1, 2, 8, 4)  # shape unchanged — compile-friendly
        assert cache.get_seq_length() == 4
        assert float(fk[:, :, 3, :].sum().item()) == 2 * 2 * 4

    def test_explicit_cache_position(self) -> None:
        cache = StaticCache(max_cache_len=8)
        cache.update(
            *_ones(1, 5.0), layer_idx=0, cache_kwargs={"cache_position": lucid.tensor([4]).long()}
        )
        assert float(cache.key_cache[0][:, :, 4, :].sum().item()) == 5 * 2 * 4

    def test_read_len_narrows_return_only(self) -> None:
        # Opt-in read_len narrows the RETURNED view (attention reads O(filled)),
        # while the STORED buffer stays full max_cache_len (compile-stable write).
        cache = StaticCache(max_cache_len=16)
        # No read_len -> full buffer (back-compatible default).
        fk, _ = cache.update(*_ones(3, 1.0), layer_idx=0)
        assert tuple(fk.shape) == (1, 2, 16, 4)
        # read_len=4 -> returned view narrowed to width 4, holding the 4 written
        # positions (0..3); the stored buffer is still the full 16.
        nk, nv = cache.update(*_ones(1, 2.0), layer_idx=0, cache_kwargs={"read_len": 4})
        assert tuple(nk.shape) == (1, 2, 4, 4)
        assert tuple(nv.shape) == (1, 2, 4, 4)
        assert tuple(cache.key_cache[0].shape) == (1, 2, 16, 4)  # buffer full
        # The narrowed view equals the full buffer's first 4 positions exactly.
        full = cache.key_cache[0]
        assert float((nk - full[:, :, :4, :]).abs().sum().item()) == 0.0

    def test_multi_layer_lazy_alloc(self) -> None:
        cache = StaticCache(max_cache_len=8)
        cache.update(*_ones(2, 1.0), layer_idx=0)
        cache.update(*_ones(2, 1.0), layer_idx=1)
        assert len(cache) == 2
        assert cache.get_seq_length(0) == 2
        assert cache.get_seq_length(1) == 2

    def test_empty_seq_length_zero(self) -> None:
        cache = StaticCache(max_cache_len=8)
        assert cache.get_seq_length() == 0
        assert len(cache) == 0

    def test_crop_and_reset(self) -> None:
        cache = StaticCache(max_cache_len=8)
        cache.update(*_ones(5, 1.0), layer_idx=0)
        cache.crop(2)
        assert cache.get_seq_length() == 2  # counter rewound (buffer untouched)
        cache.crop(-1)
        assert cache.get_seq_length() == 1
        cache.reset()
        assert cache.get_seq_length() == 0
        assert float(cache.key_cache[0].sum().item()) == 0.0  # buffer zeroed

    def test_batch_ops(self) -> None:
        cache = StaticCache(max_cache_len=8)
        cache.update(*_ones(3, 1.0), layer_idx=0)  # batch=1
        cache.batch_repeat_interleave(4)
        assert tuple(cache.key_cache[0].shape) == (4, 2, 8, 4)
        cache.batch_select_indices(lucid.tensor([0, 2]).long())
        assert tuple(cache.key_cache[0].shape) == (2, 2, 8, 4)

    def test_dunder_protocol(self) -> None:
        cache = StaticCache(max_cache_len=8)
        cache.update(*_ones(2, 1.0), layer_idx=0)
        gk, gv = cache[0]
        assert tuple(gk.shape) == (1, 2, 8, 4)
        assert len(list(cache)) == 1
