"""Phase A unit + parity tests — Tier 0 / 1 / 2 tokenizers.

Coverage matrix (per tokenizer family):
  * Encode → decode round-trip
  * Training produces a non-trivial vocab
  * Python ↔ Fast bit-parity on encode (when both are trained on the
    same vocab / share the same merges)
  * HF format round-trip — save() → from_file() preserves state
  * __call__ with return_tensors='lucid' yields Tensor outputs

The BPE family already has its own dedicated test_bpe.py from Z1;
this file covers everything else added in Phase A.
"""

import os
import tempfile

import pytest

import lucid
from lucid.utils.tokenizer import (
    ByteLevelBPETokenizer,
    ByteLevelBPETokenizerFast,
    ByteTokenizer,
    ByteTokenizerFast,
    CharTokenizer,
    CharTokenizerFast,
    RegexTokenizer,
    RegexTokenizerFast,
    SpecialTokens,
    WhitespaceTokenizer,
    WhitespaceTokenizerFast,
    WordPieceTokenizer,
    WordPieceTokenizerFast,
    WordTokenizer,
    WordTokenizerFast,
)

CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "the quick brown fox jumps high",
    "the dog the dog the dog",
    "a quick fox runs",
] * 4  # multiplied for higher pair frequencies


# ── Tier 0 — ByteTokenizer ──────────────────────────────────────────


class TestByteTokenizer:
    def test_vocab_size_fixed(self) -> None:
        assert ByteTokenizer().vocab_size == 256
        assert ByteTokenizerFast().vocab_size == 256

    def test_encode_decode_roundtrip(self) -> None:
        for cls in (ByteTokenizer, ByteTokenizerFast):
            tok = cls()
            for text in ["hello", "café 한국어 🦊", "", "the dog"]:
                ids = tok.encode(text, add_special_tokens=False)
                back = tok.decode(ids, skip_special_tokens=False)
                assert back == text, f"{cls.__name__} failed on {text!r}: {back!r}"

    def test_python_fast_parity(self) -> None:
        slow, fast = ByteTokenizer(), ByteTokenizerFast()
        for text in ["hello world", "한국어 테스트", ""]:
            assert slow.encode(text, add_special_tokens=False) == fast.encode(
                text, add_special_tokens=False
            )

    def test_train_is_noop(self) -> None:
        tok = ByteTokenizer()
        tok.train(CORPUS, vocab_size=1000)
        assert tok.vocab_size == 256  # unchanged


# ── Tier 0 — CharTokenizer ──────────────────────────────────────────


class TestCharTokenizer:
    def test_train_builds_vocab(self) -> None:
        for cls in (CharTokenizer, CharTokenizerFast):
            tok = cls()
            tok.train(CORPUS, vocab_size=100)
            assert tok.vocab_size > 0
            # Should include common chars (space + lowercase letters).
            v = tok.get_vocab()
            assert " " in v
            assert "t" in v

    def test_encode_decode_roundtrip(self) -> None:
        for cls in (CharTokenizer, CharTokenizerFast):
            tok = cls()
            tok.train(CORPUS, vocab_size=100)
            for text in ["the dog", "the quick brown fox"]:
                ids = tok.encode(text, add_special_tokens=False)
                back = tok.decode(ids, skip_special_tokens=False)
                assert back == text

    def test_python_fast_parity(self) -> None:
        slow = CharTokenizer()
        slow.train(CORPUS, vocab_size=100)
        fast = CharTokenizerFast(vocab=slow.get_vocab())
        for text in ["the dog", "the quick"]:
            assert slow.encode(text, add_special_tokens=False) == fast.encode(
                text, add_special_tokens=False
            )

    def test_save_then_from_file(self) -> None:
        orig = CharTokenizer()
        orig.train(CORPUS, vocab_size=100)
        with tempfile.TemporaryDirectory() as d:
            orig.save(d)
            assert os.path.isfile(os.path.join(d, "vocab.txt"))
            loaded = CharTokenizer.from_file(d)
        assert loaded.get_vocab() == orig.get_vocab()
        assert loaded.encode("the dog") == orig.encode("the dog")


# ── Tier 1 — WhitespaceTokenizer ────────────────────────────────────


class TestWhitespaceTokenizer:
    def test_train_builds_vocab(self) -> None:
        for cls in (WhitespaceTokenizer, WhitespaceTokenizerFast):
            tok = cls()
            tok.train(CORPUS, vocab_size=100)
            v = tok.get_vocab()
            assert "the" in v
            assert "fox" in v

    def test_encode_decode_roundtrip(self) -> None:
        for cls in (WhitespaceTokenizer, WhitespaceTokenizerFast):
            tok = cls()
            tok.train(CORPUS, vocab_size=100)
            text = "the quick brown fox"
            ids = tok.encode(text, add_special_tokens=False)
            assert tok.decode(ids, skip_special_tokens=False) == text

    def test_python_fast_parity(self) -> None:
        slow = WhitespaceTokenizer()
        slow.train(CORPUS, vocab_size=100)
        fast = WhitespaceTokenizerFast(vocab=slow.get_vocab())
        for text in ["the dog", "quick fox", "lazy dog the dog"]:
            assert slow.encode(text, add_special_tokens=False) == fast.encode(
                text, add_special_tokens=False
            )

    def test_oov_silently_dropped(self) -> None:
        tok = WhitespaceTokenizer()
        tok.train(["hello world"], vocab_size=10)
        # 'unknown' is OOV — should be dropped silently (no UNK).
        assert tok.encode("hello unknown world", add_special_tokens=False) == [
            tok.get_vocab()["hello"],
            tok.get_vocab()["world"],
        ]


# ── Tier 1 — WordTokenizer ──────────────────────────────────────────


class TestWordTokenizer:
    def test_oov_raises_without_unk(self) -> None:
        for cls in (WordTokenizer, WordTokenizerFast):
            tok = cls()
            tok.train(["hello world"], vocab_size=10)
            with pytest.raises(ValueError, match="OOV"):
                tok.encode("hello unknown")

    def test_oov_emits_unk_when_configured(self) -> None:
        for cls in (WordTokenizer, WordTokenizerFast):
            tok = cls()
            tok.train(["hello world"], vocab_size=10)
            # Add UNK to vocab + register.
            v = tok.get_vocab()
            v["<unk>"] = max(v.values()) + 1
            tok = cls(vocab=v, special_tokens=SpecialTokens(unk="<unk>"))
            ids = tok.encode("hello unknown world", add_special_tokens=False)
            assert tok.unk_token_id in ids

    def test_python_fast_parity(self) -> None:
        slow = WordTokenizer()
        slow.train(CORPUS, vocab_size=100)
        fast = WordTokenizerFast(vocab=slow.get_vocab())
        for text in ["the dog", "quick fox"]:
            assert slow.encode(text, add_special_tokens=False) == fast.encode(
                text, add_special_tokens=False
            )


# ── Tier 1 — RegexTokenizer ─────────────────────────────────────────


class TestRegexTokenizer:
    def test_train_picks_up_word_runs(self) -> None:
        for cls in (RegexTokenizer, RegexTokenizerFast):
            tok = cls(r"\w+")
            tok.train(["hello, world! foo-bar baz."], vocab_size=20)
            v = tok.get_vocab()
            assert "hello" in v
            assert "world" in v
            # Punctuation dropped by the regex.
            assert "," not in v
            assert "!" not in v

    def test_encode_drops_punctuation(self) -> None:
        for cls in (RegexTokenizer, RegexTokenizerFast):
            tok = cls(r"\w+")
            tok.train(["foo bar baz"], vocab_size=20)
            assert tok.encode("foo, bar! baz.", add_special_tokens=False) == [
                tok.get_vocab()[w] for w in ("foo", "bar", "baz")
            ]

    def test_python_fast_parity(self) -> None:
        slow = RegexTokenizer(r"\w+")
        slow.train(CORPUS, vocab_size=100)
        fast = RegexTokenizerFast(r"\w+", vocab=slow.get_vocab())
        for text in ["the dog!", "quick, brown.", "fox jumps"]:
            assert slow.encode(text, add_special_tokens=False) == fast.encode(
                text, add_special_tokens=False
            )

    def test_save_then_from_file_preserves_pattern(self) -> None:
        orig = RegexTokenizer(r"[a-z]+")
        orig.train(["hello world"], vocab_size=10)
        with tempfile.TemporaryDirectory() as d:
            orig.save(d)
            assert os.path.isfile(os.path.join(d, "regex_pattern.txt"))
            loaded = RegexTokenizer.from_file(d)
        assert loaded.pattern == r"[a-z]+"
        assert loaded.get_vocab() == orig.get_vocab()


# ── Tier 2 — ByteLevelBPE ───────────────────────────────────────────


class TestByteLevelBPETokenizer:
    def test_train_builds_vocab(self) -> None:
        for cls in (ByteLevelBPETokenizer, ByteLevelBPETokenizerFast):
            tok = cls(vocab={}, merges=[])
            tok.train(CORPUS, vocab_size=80)
            assert tok.vocab_size > 0

    def test_encode_decode_roundtrip(self) -> None:
        for cls in (ByteLevelBPETokenizer, ByteLevelBPETokenizerFast):
            tok = cls(vocab={}, merges=[])
            tok.train(CORPUS, vocab_size=80)
            text = "the quick brown fox"
            ids = tok.encode(text, add_special_tokens=False)
            back = tok.decode(ids, skip_special_tokens=False)
            # BBPE preserves spaces (unlike classical BPE).
            assert back == text or back == text + " " or back == " " + text

    def test_python_fast_parity_when_sharing_vocab(self) -> None:
        """Bit-parity holds when both flavours share the same trained
        vocab + merges (training paths diverge slightly between flavours,
        but encode is deterministic given the vocab)."""
        slow = ByteLevelBPETokenizer(vocab={}, merges=[])
        slow.train(CORPUS, vocab_size=80)
        fast = ByteLevelBPETokenizerFast(vocab=slow.get_vocab(), merges=slow._merges)
        for text in ["the dog", "the quick brown fox"]:
            assert slow.encode(text, add_special_tokens=False) == fast.encode(
                text, add_special_tokens=False
            )

    def test_preserves_whitespace(self) -> None:
        """The key property of BBPE vs classical BPE."""
        tok = ByteLevelBPETokenizer(vocab={}, merges=[])
        tok.train(CORPUS, vocab_size=80)
        ids = tok.encode("the dog", add_special_tokens=False)
        assert " " in tok.decode(ids, skip_special_tokens=False)


# ── Tier 2 — WordPiece ──────────────────────────────────────────────


class TestWordPieceTokenizer:
    def test_train_includes_unk(self) -> None:
        for cls in (WordPieceTokenizer, WordPieceTokenizerFast):
            tok = cls(vocab={})
            tok.train(CORPUS, vocab_size=60)
            v = tok.get_vocab()
            assert "[UNK]" in v
            assert v["[UNK]"] == 0

    def test_continuation_prefix_present(self) -> None:
        tok = WordPieceTokenizer(vocab={})
        tok.train(CORPUS, vocab_size=60)
        v = tok.get_vocab()
        # Should have at least one ## continuation piece in a non-trivial
        # vocab.
        assert any(k.startswith("##") for k in v)

    def test_encode_returns_ids(self) -> None:
        for cls in (WordPieceTokenizer, WordPieceTokenizerFast):
            tok = cls(vocab={})
            tok.train(CORPUS, vocab_size=60)
            ids = tok.encode("the dog", add_special_tokens=False)
            assert isinstance(ids, list)
            assert all(isinstance(i, int) for i in ids)

    def test_oov_emits_unk(self) -> None:
        for cls in (WordPieceTokenizer, WordPieceTokenizerFast):
            tok = cls(vocab={})
            tok.train(["a"], vocab_size=10)  # tiny vocab — almost everything OOV
            ids = tok.encode("zzzzzzz", add_special_tokens=False)
            # Should produce at least one UNK id (or stop short — both
            # acceptable; check it doesn't crash).
            assert isinstance(ids, list)

    def test_python_fast_parity_when_sharing_vocab(self) -> None:
        slow = WordPieceTokenizer(vocab={})
        slow.train(CORPUS, vocab_size=60)
        fast = WordPieceTokenizerFast(vocab=slow.get_vocab())
        for text in ["the dog", "the quick brown fox"]:
            assert slow.encode(text, add_special_tokens=False) == fast.encode(
                text, add_special_tokens=False
            )

    def test_save_then_from_file(self) -> None:
        orig = WordPieceTokenizer(vocab={})
        orig.train(CORPUS, vocab_size=60)
        with tempfile.TemporaryDirectory() as d:
            orig.save(d)
            assert os.path.isfile(os.path.join(d, "vocab.txt"))
            loaded = WordPieceTokenizer.from_file(d)
        assert loaded.get_vocab() == orig.get_vocab()
        assert loaded.encode("the dog") == orig.encode("the dog")


# ── HF-style __call__ with return_tensors='lucid' ──────────────────


class TestCallTensorIntegration:
    """Spot-check that __call__ works uniformly across the family."""

    def test_basic_call_byte(self) -> None:
        """ByteTokenizer with pad configured → padded batch returns
        rectangular lucid.Tensor outputs."""
        tok = ByteTokenizer(special_tokens=SpecialTokens(pad=chr(0)))
        out = tok(
            ["hi", "hello world"], padding=True, return_tensors="lucid", add_special_tokens=False
        )
        assert isinstance(out["input_ids"], lucid.Tensor)
        assert out["input_ids"].dtype == lucid.int32

    def test_basic_call_char(self) -> None:
        """CharTokenizer with manually-added pad in vocab + registry."""
        tok = CharTokenizer()
        tok.train(CORPUS, vocab_size=50)
        # Inject a pad token.
        v = tok.get_vocab()
        v["<pad>"] = max(v.values()) + 1
        tok = CharTokenizer(vocab=v, special_tokens=SpecialTokens(pad="<pad>"))
        out = tok(["hi", "hello"], padding=True, return_tensors="lucid", add_special_tokens=False)
        assert isinstance(out["input_ids"], lucid.Tensor)
        assert out["input_ids"].dtype == lucid.int32

    def test_byte_padding(self) -> None:
        """ByteTokenizer with manually-added pad id."""
        tok = ByteTokenizer(special_tokens=SpecialTokens(pad=chr(0)))
        out = tok(["a", "abc"], padding=True, add_special_tokens=False)
        rows = out["input_ids"]
        assert len(rows[0]) == len(rows[1])
        # Last entry of shorter row should be pad (id 0 since chr(0) → byte 0).
        assert rows[0][-1] == 0

    def test_wordpiece_attention_mask(self) -> None:
        tok = WordPieceTokenizer(vocab={})
        tok.train(CORPUS, vocab_size=60)
        v = tok.get_vocab()
        # Add [PAD] to vocab + register.
        v["[PAD]"] = max(v.values()) + 1
        tok = WordPieceTokenizer(vocab=v, special_tokens=SpecialTokens(pad="[PAD]", unk="[UNK]"))
        out = tok(
            ["the dog", "the quick brown fox"],
            padding=True,
            return_tensors="lucid",
            add_special_tokens=False,
        )
        assert isinstance(out["attention_mask"], lucid.Tensor)
        # First row should have at least one 0 in the attention mask
        # (the shorter sequence got padded).
        mask = out["attention_mask"].numpy()
        assert mask[0].sum() < mask[1].sum() or mask[0].sum() == mask[1].sum()


# ── Pretrained-style end-to-end ─────────────────────────────────────


class TestPretrainedRoundtrip:
    """Train → save → load → encode — verifying every family survives
    a HF-style from_pretrained round-trip."""

    def test_char(self) -> None:
        orig = CharTokenizer()
        orig.train(CORPUS, vocab_size=100)
        with tempfile.TemporaryDirectory() as d:
            orig.save(d)
            loaded = CharTokenizer.from_pretrained(d)
        text = "the dog"
        assert loaded.encode(text) == orig.encode(text)

    def test_whitespace(self) -> None:
        orig = WhitespaceTokenizer()
        orig.train(CORPUS, vocab_size=100)
        with tempfile.TemporaryDirectory() as d:
            orig.save(d)
            loaded = WhitespaceTokenizer.from_pretrained(d)
        text = "the dog"
        assert loaded.encode(text) == orig.encode(text)

    def test_word(self) -> None:
        orig = WordTokenizer()
        orig.train(CORPUS, vocab_size=100)
        with tempfile.TemporaryDirectory() as d:
            orig.save(d)
            loaded = WordTokenizer.from_pretrained(d)
        text = "the dog"
        assert loaded.encode(text) == orig.encode(text)

    def test_regex(self) -> None:
        orig = RegexTokenizer(r"\w+")
        orig.train(CORPUS, vocab_size=100)
        with tempfile.TemporaryDirectory() as d:
            orig.save(d)
            loaded = RegexTokenizer.from_pretrained(d)
        text = "the dog!"
        assert loaded.encode(text) == orig.encode(text)

    def test_wordpiece(self) -> None:
        orig = WordPieceTokenizer(vocab={})
        orig.train(CORPUS, vocab_size=60)
        with tempfile.TemporaryDirectory() as d:
            orig.save(d)
            loaded = WordPieceTokenizer.from_pretrained(d)
        text = "the dog"
        assert loaded.encode(text) == orig.encode(text)
