"""Cached-decoding correctness across every generation path lucid caches.

The core invariant: incremental decoding with a KV cache must produce
**token-identical** output (and bit-close logits) to re-encoding the full
prefix every step.  Verified for GPT-1, GPT-2 (decoder-only) and the
encoder-decoder Transformer and T5.
"""

from collections.abc import Callable

import pytest

import lucid
from lucid.models.text.gpt._config import GPTConfig
from lucid.models.text.gpt._model import GPTLMHeadModel
from lucid.models.text.gpt2._config import GPT2Config
from lucid.models.text.gpt2._model import GPT2LMHeadModel
from lucid.models.text.t5._config import T5Config
from lucid.models.text.t5._model import T5ForSeq2SeqLM, T5Model
from lucid.models.text.transformer._config import TransformerConfig
from lucid.models.text.transformer._model import TransformerForSeq2SeqLM, TransformerModel
from lucid.test._helpers.compare import assert_close
from lucid.utils.cache import DynamicCache


def _gpt2() -> GPT2LMHeadModel:
    cfg = GPT2Config(
        vocab_size=50,
        hidden_size=32,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=64,
        max_position_embeddings=32,
    )
    return GPT2LMHeadModel(cfg).eval()


def _gpt() -> GPTLMHeadModel:
    cfg = GPTConfig(
        vocab_size=50,
        hidden_size=32,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=64,
        max_position_embeddings=32,
    )
    return GPTLMHeadModel(cfg).eval()


def _seq2seq() -> TransformerForSeq2SeqLM:
    cfg = TransformerConfig(
        vocab_size=50,
        hidden_size=32,
        num_attention_heads=4,
        num_hidden_layers=2,
        num_decoder_layers=2,
        intermediate_size=64,
        max_position_embeddings=32,
    )
    return TransformerForSeq2SeqLM(cfg).eval()


def _t5() -> T5ForSeq2SeqLM:
    cfg = T5Config(
        vocab_size=50,
        hidden_size=32,
        num_attention_heads=4,
        head_dim=8,
        num_hidden_layers=2,
        num_decoder_layers=2,
        intermediate_size=64,
    )
    return T5ForSeq2SeqLM(cfg).eval()


type _Seq2Seq = TransformerForSeq2SeqLM | T5ForSeq2SeqLM

#: The encoder-decoder families: each trunk exposes ``encode`` / ``decode``
#: with the same signature, so one set of cache invariants covers them all.
_SEQ2SEQ = [pytest.param(_seq2seq, id="transformer"), pytest.param(_t5, id="t5")]


def _trunk(model: _Seq2Seq) -> TransformerModel | T5Model:
    return model.transformer if isinstance(model, TransformerForSeq2SeqLM) else model.t5


def _decoder_only_incremental_equals_full(model: object) -> None:
    lucid.manual_seed(0)
    ids = lucid.tensor([[3, 7, 1, 9, 2, 8]]).long()
    T = int(ids.shape[1])

    full = model(ids).logits  # type: ignore[operator]

    cache = DynamicCache()
    steps = []
    last = None
    for t in range(T):
        out = model(ids[:, t : t + 1], use_cache=True, past_key_values=cache)  # type: ignore[operator]
        steps.append(out.logits[:, -1, :])
        last = out
    inc = lucid.stack(steps, dim=1)

    assert_close(inc, full, atol=1e-4)
    assert cache.get_seq_length() == T
    # past_key_values populated, one (key, value) pair per layer, (B, H, T, D)
    assert last is not None and last.past_key_values is not None
    assert len(last.past_key_values) == 2
    assert tuple(last.past_key_values.key_cache[0].shape) == (1, 2, T, 16)


class TestGPT2Cache:
    def test_incremental_equals_full(self) -> None:
        _decoder_only_incremental_equals_full(_gpt2())

    def test_generate_cached_equals_uncached(self) -> None:
        lucid.manual_seed(0)
        m = _gpt2()
        prompt = lucid.tensor([[3, 7, 1]]).long()
        g_cache = m.generate(prompt, max_new_tokens=6, do_sample=False, use_cache=True)
        g_nocache = m.generate(prompt, max_new_tokens=6, do_sample=False, use_cache=False)
        assert_close(g_cache, g_nocache, atol=0.0)

    def test_static_cache_equals_dynamic(self) -> None:
        lucid.manual_seed(0)
        m = _gpt2()
        prompt = lucid.tensor([[3, 7, 1]]).long()
        g_static = m.generate(
            prompt, max_new_tokens=6, do_sample=False, use_cache=True, cache_implementation="static"
        )
        g_dynamic = m.generate(
            prompt,
            max_new_tokens=6,
            do_sample=False,
            use_cache=True,
            cache_implementation="dynamic",
        )
        assert_close(g_static, g_dynamic, atol=0.0)

    def test_static_cache_oversized_equals_dynamic(self) -> None:
        # An OVER-SIZED max_cache_len (buffer far larger than tokens decoded) must
        # be both correct AND free: the read_len narrowing makes eager StaticCache
        # attend over only the filled prefix, so an oversized buffer stays
        # bit-identical to DynamicCache (no O(max_cache_len) attention penalty).
        lucid.manual_seed(0)
        m = _gpt2()
        prompt = lucid.tensor([[3, 7, 1]]).long()
        g_static = m.generate(
            prompt,
            max_new_tokens=6,
            do_sample=False,
            use_cache=True,
            cache_implementation="static",
            max_cache_len=32,  # 3 + 6 = 9 filled, buffer is 32 → mostly empty
        )
        g_dynamic = m.generate(prompt, max_new_tokens=6, do_sample=False, use_cache=True)
        assert_close(g_static, g_dynamic, atol=0.0)

    def test_static_cache_incremental_equals_full(self) -> None:
        from lucid.utils.cache import StaticCache

        lucid.manual_seed(0)
        m = _gpt2()
        ids = lucid.tensor([[3, 7, 1, 9, 2, 8]]).long()
        T = int(ids.shape[1])
        full = m(ids).logits
        cache = StaticCache(max_cache_len=16)
        steps = []
        for t in range(T):
            out = m(ids[:, t : t + 1], use_cache=True, past_key_values=cache)
            steps.append(out.logits[:, -1, :])
        inc = lucid.stack(steps, dim=1)
        assert_close(inc, full, atol=1e-4)
        assert cache.get_seq_length() == T
        assert tuple(cache.key_cache[0].shape) == (1, 2, 16, 16)  # fixed buffer


class TestGPTCache:
    def test_incremental_equals_full(self) -> None:
        _decoder_only_incremental_equals_full(_gpt())

    def test_generate_cached_equals_uncached(self) -> None:
        lucid.manual_seed(0)
        m = _gpt()
        prompt = lucid.tensor([[3, 7, 1]]).long()
        g_cache = m.generate(prompt, max_new_tokens=6, do_sample=False, use_cache=True)
        g_nocache = m.generate(prompt, max_new_tokens=6, do_sample=False, use_cache=False)
        assert_close(g_cache, g_nocache, atol=0.0)

    def test_static_cache_equals_dynamic(self) -> None:
        # GPT-1's eager StaticCache path uses the past_len-slice branches (its
        # trunk does not default cache_position), distinct from GPT-2's — cover
        # it explicitly so a future edit can't silently regress it.
        lucid.manual_seed(0)
        m = _gpt()
        prompt = lucid.tensor([[3, 7, 1]]).long()
        g_static = m.generate(
            prompt, max_new_tokens=6, do_sample=False, cache_implementation="static"
        )
        g_dynamic = m.generate(prompt, max_new_tokens=6, do_sample=False)
        assert_close(g_static, g_dynamic, atol=0.0)


class TestStaticCacheValidation:
    """Eager StaticCache up-front validation (writes by absolute position, so an
    out-of-range target must raise, not silently corrupt on Metal)."""

    def test_target_exceeds_max_position_raises(self) -> None:
        import pytest

        lucid.manual_seed(0)
        m = _gpt2()  # max_position_embeddings=32
        prompt = lucid.tensor([[3, 7, 1, 9]]).long()
        with pytest.raises(ValueError, match="max_position_embeddings"):
            m.generate(prompt, max_length=40, do_sample=False, cache_implementation="static")

    def test_max_cache_len_too_small_raises(self) -> None:
        import pytest

        lucid.manual_seed(0)
        m = _gpt2()
        prompt = lucid.tensor([[3, 7, 1, 9]]).long()
        with pytest.raises(ValueError, match="max_cache_len"):
            m.generate(
                prompt,
                max_new_tokens=10,
                do_sample=False,
                cache_implementation="static",
                max_cache_len=8,  # < 4 + 10 target
            )


@pytest.mark.parametrize("build", _SEQ2SEQ)
class TestSeq2SeqCache:
    def test_decode_incremental_equals_full(self, build: Callable[[], _Seq2Seq]) -> None:
        lucid.manual_seed(0)
        trunk = _trunk(build())
        src = lucid.tensor([[3, 7, 1, 9, 2]]).long()
        tgt = lucid.tensor([[1, 4, 2, 5, 6]]).long()
        T = int(tgt.shape[1])

        memory = trunk.encode(src)
        full = trunk.decode(tgt, memory)  # (B, T, d)

        from lucid.utils.cache import EncoderDecoderCache

        cache = EncoderDecoderCache(DynamicCache(), DynamicCache())
        steps = []
        for t in range(T):
            dec = trunk.decode(tgt[:, t : t + 1], memory, past_key_value=cache, use_cache=True)
            steps.append(dec[:, -1, :])
        inc = lucid.stack(steps, dim=1)

        assert_close(inc, full, atol=1e-4)
        assert cache.self_attention_cache.get_seq_length() == T
        # cross-attention computed once, then reused
        assert cache.cross_attention_cache.get_seq_length() == int(memory.shape[1])
        assert all(cache.is_updated.values())

    def test_generate_cached_equals_uncached(self, build: Callable[[], _Seq2Seq]) -> None:
        lucid.manual_seed(0)
        m = build()
        src = lucid.tensor([[3, 7, 1, 9, 2]]).long()
        g_cache = m.generate(src, max_length=8, use_cache=True)
        g_nocache = m.generate(src, max_length=8, use_cache=False)
        assert_close(g_cache, g_nocache, atol=0.0)

    def test_decoder_applies_padding_masks(self, build: Callable[[], _Seq2Seq]) -> None:
        # Regression: TransformerDecoder.forward dropped tgt/memory key-padding
        # masks before the layers.  Correctness check: with the mask applied,
        # the decoder output must be INVARIANT to whatever tokens sit in the
        # padded source positions (encoder self-attn + cross-attn both mask).
        lucid.manual_seed(0)
        trunk = _trunk(build())
        amask = lucid.tensor([[1.0, 1, 1, 1, 1, 0, 0]])
        dec_in = lucid.tensor([[1, 4, 2]]).long()
        src_a = lucid.tensor([[3, 7, 1, 9, 2, 0, 0]]).long()
        src_b = lucid.tensor([[3, 7, 1, 9, 2, 5, 8]]).long()  # differs only in pad cols
        out_a = trunk.decode(dec_in, trunk.encode(src_a, amask), memory_attention_mask=amask)
        out_b = trunk.decode(dec_in, trunk.encode(src_b, amask), memory_attention_mask=amask)
        assert_close(out_a, out_b, atol=1e-4)  # padded positions do not leak
        # and dropping the mask DOES let them leak (the bug's signature)
        leak_a = trunk.decode(dec_in, trunk.encode(src_a, amask))
        leak_b = trunk.decode(dec_in, trunk.encode(src_b, amask))
        assert float((leak_a - leak_b).abs().max().item()) > 1e-3
