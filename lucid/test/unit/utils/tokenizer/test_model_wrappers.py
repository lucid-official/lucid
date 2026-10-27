"""Phase C — per-model tokenizer wrappers.

Each text-model family ships a `_tokenizer/` package with
`{Model}Tokenizer` + `{Model}TokenizerFast` subclassing the matching
base algorithm with model-specific defaults baked in (special-token
registry, normalizer settings).  These tests cover:

* The wrappers import from the family package's top-level `__init__`.
* Default special-tokens registry matches the model convention.
* Train / encode / decode work end-to-end.
* Python ↔ Fast parity on the wrapper level (same vocab → same ids).
"""

import glob
import os

import pytest

from lucid.models.text.bert import BERTTokenizer, BERTTokenizerFast
from lucid.models.text.gpt import GPTTokenizer, GPTTokenizerFast
from lucid.models.text.gpt2 import GPT2Tokenizer, GPT2TokenizerFast
from lucid.models.text.roformer import (
    RoFormerTokenizer,
    RoFormerTokenizerFast,
)

CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "the quick brown fox jumps high",
    "the dog the dog the dog",
    "a quick fox runs",
] * 8


# ── BERT ────────────────────────────────────────────────────────────


class TestBERTTokenizer:
    def test_default_special_tokens(self) -> None:
        for cls in (BERTTokenizer, BERTTokenizerFast):
            tok = cls(vocab={})
            st = tok.special_tokens
            assert st.unk == "[UNK]"
            assert st.pad == "[PAD]"
            assert st.cls == "[CLS]"
            assert st.sep == "[SEP]"
            assert st.mask == "[MASK]"

    def test_train_encode_decode(self) -> None:
        for cls in (BERTTokenizer, BERTTokenizerFast):
            tok = cls(vocab={})
            tok.train(CORPUS, vocab_size=60)
            ids = tok.encode("the dog", add_special_tokens=False)
            assert isinstance(ids, list)
            assert len(ids) > 0

    def test_python_fast_parity(self) -> None:
        slow = BERTTokenizer(vocab={})
        slow.train(CORPUS, vocab_size=60)
        fast = BERTTokenizerFast(vocab=slow.get_vocab())
        for text in ["the dog", "the quick brown fox"]:
            assert slow.encode(text, add_special_tokens=False) == fast.encode(
                text, add_special_tokens=False
            )

    def test_lowercasing_default(self) -> None:
        """BERT default is uncased — encode("THE") should match
        encode("the") because of the bundled BERTNormalizer."""
        tok = BERTTokenizer(vocab={})
        tok.train(CORPUS, vocab_size=60)
        assert tok.encode("the dog", add_special_tokens=False) == tok.encode(
            "THE DOG", add_special_tokens=False
        )

    def test_frames_with_cls_and_sep(self) -> None:
        """[CLS] A [SEP] — the registry named both, the base inserted neither."""
        vocab = {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[PAD]": 3, "hello": 4, "world": 5}
        for cls in (BERTTokenizer, BERTTokenizerFast):
            tok = cls(vocab=vocab)
            assert tok.encode("hello world") == [1, 4, 5, 2]
            assert tok.encode("hello", add_special_tokens=False) == [4]
            assert tok.num_special_tokens_to_add() == 2
            assert tok.num_special_tokens_to_add(pair=True) == 3
            out = tok("hello world hello", truncation=True, max_length=3)
            assert out["input_ids"] == [1, 4, 2]
            assert out["special_tokens_mask"] == [1, 0, 1]

    def test_missing_framing_tokens_are_skipped(self) -> None:
        """A vocabulary without [CLS] / [SEP] encodes unframed, not an error.

        Training does not reserve the special tokens, so this is the state
        every freshly trained wrapper is in.
        """
        tok = BERTTokenizer(vocab={})
        tok.train(CORPUS, vocab_size=60)
        assert tok.cls_token_id is None
        assert tok.encode("the dog") == tok.encode("the dog", add_special_tokens=False)
        assert tok.num_special_tokens_to_add() == 0

    def test_cased_keeps_accents(self) -> None:
        vocab = {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "café": 3, "cafe": 4}
        cased = BERTTokenizer(vocab=vocab, do_lower_case=False)
        assert cased.encode("café", add_special_tokens=False) == [3]
        uncased = BERTTokenizer(vocab=vocab)
        assert uncased.encode("Café", add_special_tokens=False) == [4]

    def test_matches_hugging_face(self) -> None:
        tokenizers = pytest.importorskip("tokenizers")
        snapshot = _hub_snapshot("bert-base-uncased")
        if snapshot is None or not os.path.isfile(os.path.join(snapshot, "vocab.txt")):
            pytest.skip("bert-base-uncased is not in the Hugging Face cache")
        theirs = tokenizers.Tokenizer.from_file(os.path.join(snapshot, "tokenizer.json"))
        texts = [
            "Hello world",
            "The quick brown fox jumps over the lazy dog.",
            "naïve café, résumé!",
            "  spaced   out  ",
            "東京 123",
        ]
        for cls in (BERTTokenizer, BERTTokenizerFast):
            ours = cls.from_file(snapshot)
            for text in texts:
                assert ours.encode(text) == theirs.encode(text).ids, text


# ── RoFormer ────────────────────────────────────────────────────────


class TestRoFormerTokenizer:
    def test_default_special_tokens(self) -> None:
        for cls in (RoFormerTokenizer, RoFormerTokenizerFast):
            tok = cls(vocab={})
            st = tok.special_tokens
            assert st.unk == "[UNK]"
            assert st.cls == "[CLS]"
            assert st.sep == "[SEP]"
            assert st.mask == "[MASK]"

    def test_python_fast_parity(self) -> None:
        slow = RoFormerTokenizer(vocab={})
        slow.train(CORPUS, vocab_size=60)
        fast = RoFormerTokenizerFast(vocab=slow.get_vocab())
        for text in ["the dog", "the quick brown fox"]:
            assert slow.encode(text, add_special_tokens=False) == fast.encode(
                text, add_special_tokens=False
            )

    def test_frames_with_cls_and_sep(self) -> None:
        vocab = {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "he": 3, "##llo": 4}
        for cls in (RoFormerTokenizer, RoFormerTokenizerFast):
            assert cls(vocab=vocab).encode("hello") == [1, 3, 4, 2]


# ── GPT-1 ───────────────────────────────────────────────────────────


class TestGPTTokenizer:
    """GPT-1 is word BPE with ``</w>``, not GPT-2's byte-level scheme."""

    @staticmethod
    def _marked(cls: type = GPTTokenizer) -> GPTTokenizer:
        vocab = {"i": 0, "n": 1, "n</w>": 2, "in": 3, "in</w>": 4, "s": 5, "s</w>": 6}
        return cls(vocab=vocab, merges=[("i", "n</w>"), ("i", "n")])

    def test_end_of_word_separates_word_from_subword(self) -> None:
        """The same letters must not collapse onto the same id.

        Without the marker, ``in`` the word and ``in`` inside ``ins`` share
        one entry, and that entry silently means two things.
        """
        tok = self._marked()
        assert [tok.id_to_token(i) for i in tok.encode("in")] == ["in</w>"]
        assert [tok.id_to_token(i) for i in tok.encode("ins")] == ["in", "s</w>"]

    def test_round_trip_restores_word_boundaries(self) -> None:
        tok = self._marked()
        assert tok.decode(tok.encode("in ins")) == "in ins"

    def test_input_is_lowercased(self) -> None:
        """§4.1 lowercases the text, so the default normalizer must too."""
        tok = self._marked()
        assert tok.encode("IN") == tok.encode("in")

    def test_fast_matches_python(self) -> None:
        assert self._marked(GPTTokenizerFast).encode("ins") == self._marked().encode("ins")

    def test_train_encode(self) -> None:
        for cls in (GPTTokenizer, GPTTokenizerFast):
            tok = cls(vocab={}, merges=[])
            tok.train(CORPUS, vocab_size=80)
            ids = tok.encode("the dog", add_special_tokens=False)
            assert isinstance(ids, list)
            assert len(ids) > 0

    def test_adds_no_special_tokens(self) -> None:
        """Registered special tokens are for callers; encode inserts none."""
        from lucid.utils.tokenizer import SpecialTokens

        vocab = {"i": 0, "n</w>": 1, "in</w>": 2, "<s>": 3, "</s>": 4}
        for cls in (GPTTokenizer, GPTTokenizerFast):
            tok = cls(
                vocab=vocab,
                merges=[("i", "n</w>")],
                special_tokens=SpecialTokens(bos="<s>", eos="</s>"),
            )
            assert tok.bos_token_id == 3
            assert tok.encode("in") == [2]
            assert tok.num_special_tokens_to_add() == 0

    def test_python_fast_parity(self) -> None:
        slow = GPTTokenizer(vocab={}, merges=[])
        slow.train(CORPUS, vocab_size=80)
        fast = GPTTokenizerFast(vocab=slow.get_vocab(), merges=slow._merges)
        for text in ["the dog", "the quick brown fox"]:
            assert slow.encode(text, add_special_tokens=False) == fast.encode(
                text, add_special_tokens=False
            )


# ── GPT-2 ───────────────────────────────────────────────────────────


class TestGPT2Tokenizer:
    def test_default_endoftext(self) -> None:
        for cls in (GPT2Tokenizer, GPT2TokenizerFast):
            tok = cls(vocab={}, merges=[])
            st = tok.special_tokens
            assert st.bos == "<|endoftext|>"
            assert st.eos == "<|endoftext|>"
            assert st.unk == "<|endoftext|>"

    def test_train_encode(self) -> None:
        for cls in (GPT2Tokenizer, GPT2TokenizerFast):
            tok = cls(vocab={}, merges=[])
            tok.train(CORPUS, vocab_size=80)
            ids = tok.encode("the dog", add_special_tokens=False)
            assert isinstance(ids, list)
            assert len(ids) > 0

    def test_python_fast_parity(self) -> None:
        slow = GPT2Tokenizer(vocab={}, merges=[])
        slow.train(CORPUS, vocab_size=80)
        fast = GPT2TokenizerFast(vocab=slow.get_vocab(), merges=slow._merges)
        for text in ["the dog", "the quick brown fox"]:
            assert slow.encode(text, add_special_tokens=False) == fast.encode(
                text, add_special_tokens=False
            )

    def test_adds_no_special_tokens(self) -> None:
        """<|endoftext|> is registered for callers, never inserted."""
        vocab = {"h": 0, "i": 1, "hi": 2, "<|endoftext|>": 3}
        for cls in (GPT2Tokenizer, GPT2TokenizerFast):
            tok = cls(vocab=vocab, merges=[("h", "i")])
            assert tok.eos_token_id == 3 and tok.bos_token_id == 3
            assert tok.encode("hi") == [2]
            assert tok.num_special_tokens_to_add() == 0
            assert tok("hi hi hi", truncation=True, max_length=1)["input_ids"] == [2]

    def test_matches_hugging_face(self) -> None:
        tokenizers = pytest.importorskip("tokenizers")
        snapshot = _hub_snapshot("gpt2")
        if snapshot is None:
            pytest.skip("gpt2 is not in the Hugging Face cache")
        theirs = tokenizers.Tokenizer.from_file(os.path.join(snapshot, "tokenizer.json"))
        # Single spaces only: runs of whitespace go through Lucid's
        # approximation of the GPT-2 split expression, which is a separate
        # matter from the framing checked here.
        texts = ["Hello world", "The quick brown fox.", "naïve café, 123!", "hi"]
        for cls in (GPT2Tokenizer, GPT2TokenizerFast):
            ours = cls.from_file(snapshot)
            for text in texts:
                assert ours.encode(text) == theirs.encode(text).ids, text

    def test_preserves_whitespace(self) -> None:
        """Byte-level BPE preserves spaces (unlike WordPiece / classical BPE)."""
        tok = GPT2Tokenizer(vocab={}, merges=[])
        tok.train(CORPUS, vocab_size=80)
        ids = tok.encode("the dog", add_special_tokens=False)
        assert " " in tok.decode(ids, skip_special_tokens=False)


def _hub_snapshot(repo: str) -> str | None:
    """Directory of a cached Hugging Face snapshot holding ``tokenizer.json``."""
    roots = []
    if os.environ.get("HF_HUB_CACHE"):
        roots.append(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        roots.append(os.path.join(os.environ["HF_HOME"], "hub"))
    roots.append(os.path.expanduser("~/.cache/huggingface/hub"))
    folder = "models--" + repo.replace("/", "--")
    for root in roots:
        for snap in sorted(glob.glob(os.path.join(root, folder, "snapshots", "*"))):
            if os.path.isfile(os.path.join(snap, "tokenizer.json")):
                return snap
    return None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
