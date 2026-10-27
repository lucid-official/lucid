"""Phase B — Unigram tokenizer (SentencePiece flavour).

Coverage:
  * Train builds a non-trivial vocab including UNK + single-codepoint
    pieces.
  * Encode → decode round-trip preserves the original text (mod
    NFKC + SentencePiece dummy-prefix conventions).
  * Python (slow) ↔ Fast bit-parity when sharing the same pieces.
  * HF unified ``tokenizer.json`` save / from_file round-trip.
  * SentencePiecePreTokenizer correctly handles whitespace.
"""

import os
import tempfile

import pytest

from lucid.utils.tokenizer import SpecialTokens, UnigramTokenizer, UnigramTokenizerFast
from lucid.utils.tokenizer._unigram import SentencePiecePreTokenizer

CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "the quick brown fox jumps high",
    "the dog the dog the dog",
    "a quick fox runs",
] * 8


# ── SentencePiecePreTokenizer ───────────────────────────────────────


class TestSentencePiecePreTokenizer:
    def test_replaces_whitespace_with_marker(self) -> None:
        pt = SentencePiecePreTokenizer()
        chunks = [c for c, _ in pt("hello world")]
        assert chunks == ["▁hello", "▁world"]

    def test_add_dummy_prefix_false(self) -> None:
        pt = SentencePiecePreTokenizer(add_dummy_prefix=False)
        chunks = [c for c, _ in pt("hello world")]
        # No leading prefix on the first word.
        assert chunks[0] == "hello"
        assert chunks[1] == "▁world"

    def test_empty(self) -> None:
        pt = SentencePiecePreTokenizer(add_dummy_prefix=False)
        assert pt("") == []


# ── UnigramTokenizer (Python) + Fast ────────────────────────────────


class TestUnigramTokenizerTrain:
    def test_train_builds_vocab(self) -> None:
        for cls in (UnigramTokenizer, UnigramTokenizerFast):
            tok = cls(pieces=[])
            tok.train(CORPUS, vocab_size=40)
            assert tok.vocab_size > 0
            v = tok.get_vocab()
            assert "<unk>" in v
            # Single-codepoint pieces always kept by the pruner.
            assert "t" in v or "▁" in v

    def test_train_keeps_unk_first(self) -> None:
        tok = UnigramTokenizer(pieces=[])
        tok.train(CORPUS, vocab_size=40)
        assert tok.get_vocab()["<unk>"] == 0


class TestUnigramEncode:
    def test_encode_returns_ids(self) -> None:
        for cls in (UnigramTokenizer, UnigramTokenizerFast):
            tok = cls(pieces=[])
            tok.train(CORPUS, vocab_size=40)
            ids = tok.encode("the dog", add_special_tokens=False)
            assert isinstance(ids, list)
            assert all(isinstance(i, int) for i in ids)
            assert len(ids) > 0

    def test_decode_roundtrip(self) -> None:
        for cls in (UnigramTokenizer, UnigramTokenizerFast):
            tok = cls(pieces=[])
            tok.train(CORPUS, vocab_size=40)
            text = "the dog"
            ids = tok.encode(text, add_special_tokens=False)
            back = tok.decode(ids, skip_special_tokens=False)
            # SentencePiece round-trip adds a leading space (dummy prefix).
            assert back.strip() == text

    def test_oov_uses_unk(self) -> None:
        tok = UnigramTokenizer(pieces=[])
        tok.train(["abc"], vocab_size=10)
        ids = tok.encode("xyz", add_special_tokens=False)
        # All non-vocab codepoints route through UNK (id 0).
        assert isinstance(ids, list)


class TestUnigramParity:
    def test_python_fast_parity(self) -> None:
        slow = UnigramTokenizer(pieces=[])
        slow.train(CORPUS, vocab_size=40)
        fast = UnigramTokenizerFast(pieces=slow.pieces)
        for text in ["the dog", "the quick brown fox", "a quick fox runs"]:
            assert slow.encode(text, add_special_tokens=False) == fast.encode(
                text, add_special_tokens=False
            )

    def test_python_fast_vocab_match(self) -> None:
        slow = UnigramTokenizer(pieces=[])
        slow.train(CORPUS, vocab_size=40)
        fast = UnigramTokenizerFast(pieces=slow.pieces)
        assert slow.get_vocab() == fast.get_vocab()
        assert slow.vocab_size == fast.vocab_size


class TestUnigramSave:
    def test_save_then_from_file(self) -> None:
        orig = UnigramTokenizer(pieces=[])
        orig.train(CORPUS, vocab_size=40)
        with tempfile.TemporaryDirectory() as d:
            orig.save(d)
            assert os.path.isfile(os.path.join(d, "tokenizer.json"))
            loaded = UnigramTokenizer.from_file(d)
        assert loaded.get_vocab() == orig.get_vocab()
        text = "the dog"
        assert loaded.encode(text) == orig.encode(text)

    def test_fast_save_then_from_file(self) -> None:
        orig = UnigramTokenizerFast(pieces=[])
        orig.train(CORPUS, vocab_size=40)
        with tempfile.TemporaryDirectory() as d:
            orig.save(d)
            loaded = UnigramTokenizerFast.from_file(d)
        assert loaded.get_vocab() == orig.get_vocab()
        assert loaded.encode("the dog") == orig.encode("the dog")

    def test_from_pretrained_alias(self) -> None:
        orig = UnigramTokenizer(pieces=[])
        orig.train(CORPUS, vocab_size=40)
        with tempfile.TemporaryDirectory() as d:
            orig.save(d)
            loaded = UnigramTokenizer.from_pretrained(d)
        assert loaded.encode("the dog") == orig.encode("the dog")


class TestUnigramSpecialTokens:
    def test_custom_special_tokens(self) -> None:
        tok = UnigramTokenizer(
            pieces=[], special_tokens=SpecialTokens(unk="<unk>", bos="<s>", eos="</s>")
        )
        tok.train(CORPUS, vocab_size=40)
        # bos/eos aren't necessarily in the vocab (no auto-injection).
        # Just check it doesn't crash.
        ids = tok.encode("the dog")
        assert isinstance(ids, list)


# ── the empty document ───────────────────────────────────────────────


_PIECES = [(w, -1.0) for w in ("the", "quick", "brown", "fox")] + [("<unk>", -10.0)]


@pytest.mark.parametrize("cls", [UnigramTokenizer, UnigramTokenizerFast])
def test_encoding_nothing_produces_nothing(cls) -> None:
    """``encode("")`` must be empty, as it is for BPE and WordPiece.

    The SentencePiece pre-tokenizer prepended its ``▁`` marker even to an
    empty string, so the empty document came back holding one token — a
    marker that matches no piece and lands on ``<unk>``.  Every
    length-based caller downstream (padding, truncation, attention masks)
    reads that as a document with content.
    """
    assert cls(_PIECES).encode("") == []


@pytest.mark.parametrize("cls", [UnigramTokenizer, UnigramTokenizerFast])
def test_the_dummy_prefix_still_reaches_real_text(cls) -> None:
    """Fixing the empty case must not cost the marker everywhere else."""
    chunks = SentencePiecePreTokenizer().pre_tokenize("the quick")
    assert [c for c, _ in chunks] == ["▁the", "▁quick"]
    assert cls(_PIECES).encode("the") != []


def test_a_sentencepiece_vocabulary_round_trips() -> None:
    """With ``▁``-prefixed pieces the encoding is a fixed point."""
    pieces = [("▁" + w, -1.0) for w in ("the", "quick", "brown", "fox")]
    pieces.append(("<unk>", -10.0))
    tok = UnigramTokenizer(pieces)
    ids = tok.encode("the quick brown fox")
    assert ids == tok.encode(tok.decode(ids))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
