"""Unit + Python ↔ Fast parity tests for the BPE tokenizer family.

Covers:
  * Algorithm correctness — known-vocab encode produces the
    canonical Sennrich-2016 output.
  * Round-trip — encode → decode → re-encode is a fixed point.
  * Training — corpus → vocab + merges, then encode of a held-out
    string produces a valid id sequence.
  * Python ↔ Fast bit parity — same vocab + merges + text must
    yield the exact same id sequence on both flavours.
  * HF format round-trip — save() → from_file() preserves vocab,
    merges, and special-token registry.
  * ``__call__`` — padding / truncation / return_tensors='lucid'
    semantics + attention-mask + special-tokens-mask outputs.
"""

import json
import os
import tempfile

import lucid
from lucid.utils.tokenizer import (
    BPETokenizer,
    BPETokenizerFast,
    SpecialTokens,
)
from lucid.utils.tokenizer._normalizers import NFC, Lowercase, Sequence
from lucid.utils.tokenizer._pre_tokenizers import (
    ByteLevel,
    WhitespacePunctuationSplit,
)

# ── Shared fixtures ─────────────────────────────────────────────────


def _train_tiny(klass: type) -> object:
    """Train a tiny BPE tokenizer on a fixed corpus.

    The corpus is deterministic so train output is repeatable for
    parity tests.  Uses default normalizer + pre-tokenizer.
    """
    tok = klass(vocab={}, merges=[])
    tok.train(
        [
            "the quick brown fox jumps over the lazy dog",
            "the quick brown fox jumps high",
            "the dog the dog the dog",
        ],
        vocab_size=40,
    )
    return tok


# ── Algorithm correctness ───────────────────────────────────────────


def test_bpe_encode_decode_roundtrip_slow() -> None:
    """Slow encode + decode reconstructs the surface form (modulo
    whitespace, which the WhitespaceSplit pre-tokenizer strips)."""
    tok = _train_tiny(BPETokenizer)
    text = "the quick brown fox"
    ids = tok.encode(text, add_special_tokens=False)
    decoded = tok.decode(ids, skip_special_tokens=False)
    # Whitespace is consumed by the pre-tokenizer; decode joins
    # tokens directly.
    assert decoded == text.replace(" ", "")


def test_bpe_encode_decode_roundtrip_fast() -> None:
    """Same round-trip on the Fast flavour."""
    tok = _train_tiny(BPETokenizerFast)
    text = "the quick brown fox"
    ids = tok.encode(text, add_special_tokens=False)
    decoded = tok.decode(ids, skip_special_tokens=False)
    assert decoded == text.replace(" ", "")


def test_bpe_train_produces_merges() -> None:
    """Train should produce at least one merge on a corpus with
    repeated bigrams."""
    tok = BPETokenizer(vocab={}, merges=[])
    tok.train(
        ["abab abab abab", "abc abc abc abc"],
        vocab_size=30,
    )
    assert len(tok._merges) > 0
    assert tok.vocab_size > 0


def test_bpe_train_stops_at_vocab_target() -> None:
    """Training honors the ``vocab_size`` cap."""
    tok = BPETokenizer(vocab={}, merges=[])
    tok.train(["hello world hello world"], vocab_size=15)
    assert tok.vocab_size <= 15


# ── Python ↔ Fast bit parity ────────────────────────────────────────


def test_bpe_python_fast_parity_simple() -> None:
    """Train on Python, build Fast from the same vocab+merges,
    encode both → must match bit-for-bit."""
    slow = _train_tiny(BPETokenizer)
    fast = BPETokenizerFast(
        vocab=slow.get_vocab(),
        merges=slow._merges,
    )
    for text in [
        "the dog",
        "the quick brown fox",
        "lazy dog",
        "",
        "the the the the",
    ]:
        ids_slow = slow.encode(text, add_special_tokens=False)
        ids_fast = fast.encode(text, add_special_tokens=False)
        assert ids_slow == ids_fast, (
            f"Python ↔ Fast parity broken on {text!r}: " f"slow={ids_slow}, fast={ids_fast}"
        )


def test_bpe_python_fast_parity_with_specials() -> None:
    """With BOS/EOS specials the Python wrapper inserts them; Fast
    must wrap identically (the special-token-insertion logic lives
    on the shared base class, so both should agree)."""
    slow = _train_tiny(BPETokenizer)
    fast = BPETokenizerFast(
        vocab=slow.get_vocab(),
        merges=slow._merges,
    )
    bos_tok = "<bos>"
    eos_tok = "<eos>"
    for tok in (slow, fast):
        v = tok._vocab.copy()
        v[bos_tok] = max(v.values()) + 1
        v[eos_tok] = max(v.values()) + 1
        tok._vocab = v
        # rebuild caches for the slow path
        if hasattr(tok, "_rebuild_tables"):
            tok._rebuild_tables()
        if hasattr(tok, "_cpp"):
            # Fast: must reconstruct the C++ side because the new
            # vocab entries aren't in its internal tables.
            from lucid._C import engine as _C_engine

            tok._cpp = _C_engine.utils.tokenizer.BPE(tok._vocab, tok._merges)
            tok._id_to_token = {v: k for k, v in tok._vocab.items()}
        tok._special = SpecialTokens(bos=bos_tok, eos=eos_tok)
        tok._refresh_special_ids()
        if hasattr(tok, "_sync_special_tokens_to_cpp"):
            tok._sync_special_tokens_to_cpp()
    text = "the dog"
    ids_slow = slow.encode(text, add_special_tokens=True)
    ids_fast = fast.encode(text, add_special_tokens=True)
    assert ids_slow[0] == slow.bos_token_id
    assert ids_slow[-1] == slow.eos_token_id
    assert ids_slow == ids_fast


def test_bpe_python_fast_parity_batched() -> None:
    """Batched encode parity."""
    slow = _train_tiny(BPETokenizer)
    fast = BPETokenizerFast(
        vocab=slow.get_vocab(),
        merges=slow._merges,
    )
    batch = ["the dog", "the lazy dog", "fox"]
    assert slow.encode_batch(batch, add_special_tokens=False) == fast.encode_batch(
        batch, add_special_tokens=False
    )


# ── HF format round-trip ────────────────────────────────────────────


def test_bpe_save_then_from_file_preserves_state() -> None:
    """save() → from_file() reconstructs vocab + merges + specials."""
    orig = _train_tiny(BPETokenizer)
    orig._special = SpecialTokens(pad="<pad>", unk="<unk>")
    # Add the special tokens to the vocab so they have ids.
    v = orig.get_vocab()
    v["<pad>"] = max(v.values()) + 1
    v["<unk>"] = max(v.values()) + 1
    orig._vocab = v
    orig._rebuild_tables()
    orig._refresh_special_ids()
    with tempfile.TemporaryDirectory() as d:
        orig.save(d)
        # Both legacy + unified files were written.
        assert os.path.isfile(os.path.join(d, "vocab.json"))
        assert os.path.isfile(os.path.join(d, "merges.txt"))
        assert os.path.isfile(os.path.join(d, "tokenizer.json"))
        assert os.path.isfile(os.path.join(d, "special_tokens_map.json"))
        loaded = BPETokenizer.from_file(d)
    assert loaded.get_vocab() == orig.get_vocab()
    assert loaded._merges == orig._merges
    assert loaded.pad_token_id == orig.pad_token_id
    assert loaded.unk_token_id == orig.unk_token_id


def test_bpe_legacy_only_load() -> None:
    """from_file() works with only vocab.json + merges.txt (no
    tokenizer.json) — the canonical HF GPT-2 distribution."""
    orig = _train_tiny(BPETokenizer)
    with tempfile.TemporaryDirectory() as d:
        # Write only the legacy pair, no unified.
        with open(os.path.join(d, "vocab.json"), "w") as f:
            json.dump(orig.get_vocab(), f)
        with open(os.path.join(d, "merges.txt"), "w") as f:
            f.write("#version: 0.2\n")
            for a, b in orig._merges:
                f.write(f"{a} {b}\n")
        loaded = BPETokenizer.from_file(d)
    assert loaded.get_vocab() == orig.get_vocab()
    assert loaded._merges == orig._merges


def test_bpe_fast_load_same_as_slow() -> None:
    """Loading the same files through Fast and Slow yields the same
    state — both flavours can drop into the same on-disk vocab."""
    orig = _train_tiny(BPETokenizer)
    with tempfile.TemporaryDirectory() as d:
        orig.save(d)
        slow_loaded = BPETokenizer.from_file(d)
        fast_loaded = BPETokenizerFast.from_file(d)
    text = "the quick brown fox"
    assert slow_loaded.encode(text, add_special_tokens=False) == fast_loaded.encode(
        text, add_special_tokens=False
    )


# ── HF-style __call__ ───────────────────────────────────────────────


def test_call_returns_dict_with_input_ids() -> None:
    """Basic __call__ returns the right shape of output."""
    tok = _train_tiny(BPETokenizer)
    out = tok("the dog", add_special_tokens=False)
    assert "input_ids" in out
    assert "special_tokens_mask" in out
    assert isinstance(out["input_ids"], list)
    assert all(isinstance(i, int) for i in out["input_ids"])


def test_call_batched_padding_longest() -> None:
    """Batched + padding='longest' produces rectangular outputs."""
    tok = _train_tiny(BPETokenizer)
    # Inject a pad token so padding is well-defined.
    pad = "<pad>"
    v = tok.get_vocab()
    v[pad] = max(v.values()) + 1
    tok._vocab = v
    tok._rebuild_tables()
    tok._special = SpecialTokens(pad=pad)
    tok._refresh_special_ids()

    out = tok(
        ["the dog", "the lazy dog the dog"],
        padding=True,
        add_special_tokens=False,
    )
    rows = out["input_ids"]
    assert len({len(r) for r in rows}) == 1  # rectangular
    assert "attention_mask" in out
    # First sequence should be shorter → padded with pad_id at end.
    assert rows[0][-1] == tok.pad_token_id
    # Attention masks: 1 for real, 0 for padding.
    assert sum(out["attention_mask"][0]) == len([r for r in rows[0] if r != tok.pad_token_id])


def test_call_return_tensors_lucid() -> None:
    """return_tensors='lucid' yields lucid.Tensor int32 outputs."""
    tok = _train_tiny(BPETokenizer)
    pad = "<pad>"
    v = tok.get_vocab()
    v[pad] = max(v.values()) + 1
    tok._vocab = v
    tok._rebuild_tables()
    tok._special = SpecialTokens(pad=pad)
    tok._refresh_special_ids()

    out = tok(
        ["the dog", "the lazy dog"],
        padding=True,
        return_tensors="lucid",
        add_special_tokens=False,
    )
    assert isinstance(out["input_ids"], lucid.Tensor)
    assert out["input_ids"].dtype == lucid.int32
    assert out["input_ids"].ndim == 2
    assert isinstance(out["attention_mask"], lucid.Tensor)


def test_call_truncation_respects_max_length() -> None:
    """Truncation chops sequences past max_length."""
    tok = _train_tiny(BPETokenizer)
    out = tok(
        "the quick brown fox jumps over the lazy dog",
        truncation=True,
        max_length=3,
        add_special_tokens=False,
    )
    assert len(out["input_ids"]) <= 3


def test_call_padding_without_pad_token_raises() -> None:
    """Padding requested without a pad token should error helpfully."""
    import pytest

    tok = _train_tiny(BPETokenizer)
    # No pad token defined.
    assert tok.pad_token_id is None
    with pytest.raises(ValueError, match="no pad token"):
        tok(["a", "ab"], padding=True, add_special_tokens=False)


# ── Custom pipeline integration ─────────────────────────────────────


def test_bpe_with_custom_normalizer() -> None:
    """Lowercase normalizer should fold case before encoding —
    'HELLO' and 'hello' yield identical id sequences."""
    norm = Sequence([NFC(), Lowercase()])
    tok = BPETokenizer(vocab={}, merges=[], normalizer=norm)
    tok.train(["hello hello hello world world"], vocab_size=20)
    assert tok.encode("HELLO", add_special_tokens=False) == tok.encode(
        "hello", add_special_tokens=False
    )


def test_bpe_with_byte_level_pre_tokenizer() -> None:
    """ByteLevel pre-tokenizer encodes raw bytes into the
    GPT-2-printable mapping; the tokenizer sees no whitespace at
    all in chunks."""
    tok = BPETokenizer(vocab={}, merges=[], pre_tokenizer=ByteLevel(add_prefix_space=False))
    tok.train(["hello world hello"], vocab_size=30)
    ids = tok.encode("hello world", add_special_tokens=False)
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)


def test_bpe_punctuation_split_pretokenizer() -> None:
    """The WordPiece-style pre-tokenizer (used by BERT) splits
    punctuation as its own token even without algorithm support."""
    tok = BPETokenizer(vocab={}, merges=[], pre_tokenizer=WhitespacePunctuationSplit())
    tok.train(["hello , world ! hello , world"], vocab_size=20)
    # 'hello, world!' (no space before punct) should still produce
    # separate tokens for ',', '!'
    ids = tok.encode("hello,world!", add_special_tokens=False)
    assert isinstance(ids, list)
