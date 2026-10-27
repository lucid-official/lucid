"""The ``tokenizer.json`` pipeline, and framing with special tokens.

A Hugging Face ``tokenizer.json`` describes more than a vocabulary: a
normalizer, a pre-tokenizer, a post-processor that decides the special
tokens around a sequence, a decoder, and ``added_tokens`` cut out of the
raw text first.  Unigram loading used to read only the pieces, so a T5
vocabulary came back NFKC-normalised, BOS-prefixed and with ``</s>`` spelled
out of punctuation.  These tests pin the pipeline stage by stage, then as a
whole against a hand-written fixture, then against the published t5-small
tokenizer through the Hugging Face ``tokenizers`` package when it and the
cached file are present.

Truncation is here too because it is the same question — which special
tokens frame a sequence — asked at a length limit: it used to cut the
framed sequence and so dropped ``</s>``.
"""

import glob
import json
import os
import random
import re
import struct
import tempfile

import pytest

from lucid.utils.tokenizer import (
    SpecialTokens,
    UnigramTokenizer,
    UnigramTokenizerFast,
    WordTokenizer,
)
from lucid.utils.tokenizer._added_tokens import AddedToken, AddedVocabulary
from lucid.utils.tokenizer._decoders import (
    ByteFallback,
    ByteLevel as ByteLevelDecoder,
    Fuse,
    Metaspace as MetaspaceDecoder,
    Replace as ReplaceDecoder,
    Sequence as DecoderSequence,
    Strip as StripDecoder,
    decoder_from_config,
)
from lucid.utils.tokenizer._normalizers import (
    BERTNormalizer,
    Nmt,
    Precompiled,
    Prepend,
    Replace,
    StripAccents,
    normalizer_from_config,
)
from lucid.utils.tokenizer._post_processors import (
    BertProcessing,
    ByteLevel as ByteLevelProcessing,
    RobertaProcessing,
    Sequence as ProcessorSequence,
    TemplateProcessing,
    post_processor_from_config,
)
from lucid.utils.tokenizer._pre_tokenizers import (
    Digits,
    Metaspace,
    Punctuation,
    Sequence as PreTokenizerSequence,
    Split,
    Whitespace,
    WhitespaceSplit,
    pre_tokenizer_from_config,
)

# ── helpers ─────────────────────────────────────────────────────────


def _chunks(pre_tokenizer: object, text: str) -> list[str]:
    """Just the chunk strings of a pre-tokenizer's output."""
    return [c for c, _ in pre_tokenizer.pre_tokenize(text)]  # type: ignore[attr-defined]


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


# ── truncation keeps the framing ────────────────────────────────────


def _framed_word_tokenizer() -> WordTokenizer:
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 3}
    vocab.update({ch: 4 + i for i, ch in enumerate("abcdefgh")})
    return WordTokenizer(
        vocab, special_tokens=SpecialTokens(unk="<unk>", bos="<s>", eos="</s>", pad="<pad>")
    )


class TestTruncationKeepsSpecialTokens:
    def test_eos_survives_truncation(self) -> None:
        """The reported case: the content is cut, ``</s>`` is not."""
        tok = _framed_word_tokenizer()
        ids = tok("a b c d e f g h", truncation=True, max_length=4)["input_ids"]
        assert ids == [1, 4, 5, 2]

    def test_batch_and_padding(self) -> None:
        tok = _framed_word_tokenizer()
        out = tok(["a b c d e f", "a"], truncation=True, max_length=4, padding="max_length")
        assert out["input_ids"] == [[1, 4, 5, 2], [1, 4, 2, 3]]
        assert out["attention_mask"] == [[1, 1, 1, 1], [1, 1, 1, 0]]

    def test_short_input_is_untouched(self) -> None:
        tok = _framed_word_tokenizer()
        assert tok("a b", truncation=True, max_length=10)["input_ids"] == [1, 4, 5, 2]

    def test_no_room_for_the_specials_raises(self) -> None:
        tok = _framed_word_tokenizer()
        with pytest.raises(ValueError, match="special tokens"):
            tok("a b", truncation=True, max_length=1)

    def test_without_special_tokens_cuts_plainly(self) -> None:
        tok = _framed_word_tokenizer()
        out = tok("a b c d e", truncation=True, max_length=3, add_special_tokens=False)
        assert out["input_ids"] == [4, 5, 6]

    def test_num_special_tokens_to_add(self) -> None:
        tok = _framed_word_tokenizer()
        assert tok.num_special_tokens_to_add() == 2
        assert tok.num_special_tokens_to_add(pair=True) == 4
        assert len(tok.encode("a b")) == 2 + tok.num_special_tokens_to_add()

    def test_clip_keeps_its_sentinels(self) -> None:
        """CLIP's ``encode`` defaults to no framing; ``__call__`` still frames."""
        from lucid.models.multimodal.clip import CLIPTokenizer

        vocab = {"a</w>": 0, "b</w>": 1, "<|startoftext|>": 2, "<|endoftext|>": 3}
        tok = CLIPTokenizer(vocab=vocab, merges=[])
        ids = tok("a b a b", truncation=True, max_length=4)["input_ids"]
        assert ids == [2, 0, 1, 3]

    def test_explicit_post_processor_is_what_gets_measured(self) -> None:
        tok = _framed_word_tokenizer()
        tok.post_processor = TemplateProcessing("$A </s>", special_tokens={"</s>": 2})
        assert tok.num_special_tokens_to_add() == 1
        assert tok("a b c", truncation=True, max_length=3)["input_ids"] == [4, 5, 2]
        tok.post_processor = None
        assert tok.encode("a") == [1, 4, 2]


# ── post-processors ─────────────────────────────────────────────────


_T5_TEMPLATE = {
    "type": "TemplateProcessing",
    "single": [
        {"Sequence": {"id": "A", "type_id": 0}},
        {"SpecialToken": {"id": "</s>", "type_id": 0}},
    ],
    "pair": [
        {"Sequence": {"id": "A", "type_id": 0}},
        {"SpecialToken": {"id": "</s>", "type_id": 0}},
        {"Sequence": {"id": "B", "type_id": 0}},
        {"SpecialToken": {"id": "</s>", "type_id": 0}},
    ],
    "special_tokens": {"</s>": {"id": "</s>", "ids": [1], "tokens": ["</s>"]}},
}


class TestPostProcessors:
    def test_template_from_config(self) -> None:
        proc = post_processor_from_config(_T5_TEMPLATE)
        assert proc.process([7]) == [7, 1]
        assert proc.process([7], [8]) == [7, 1, 8, 1]
        assert proc.num_special_tokens_to_add() == 1
        assert proc.num_special_tokens_to_add(pair=True) == 2

    def test_template_round_trips_through_config(self) -> None:
        proc = post_processor_from_config(_T5_TEMPLATE)
        again = post_processor_from_config(proc.to_config())
        assert again.process([5], [6]) == proc.process([5], [6])

    def test_template_string_syntax(self) -> None:
        proc = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens={"[CLS]": 101, "[SEP]": 102},
        )
        assert proc.process([7], [8]) == [101, 7, 102, 8, 102]

    def test_template_default_pair_adds_nothing(self) -> None:
        """Hugging Face's default pair template is ``$A $B:1``."""
        proc = TemplateProcessing("[CLS] $A", special_tokens={"[CLS]": 1})
        assert proc.process([7], [8]) == [7, 8]

    def test_template_validation(self) -> None:
        with pytest.raises(ValueError, match="no id"):
            TemplateProcessing("$A </s>")
        with pytest.raises(ValueError, match=r"\$A"):
            TemplateProcessing("$A $A")
        with pytest.raises(ValueError, match=r"\$B"):
            TemplateProcessing("$A", pair="$A")

    def test_bert_and_roberta(self) -> None:
        bert = post_processor_from_config(
            {"type": "BertProcessing", "sep": ["[SEP]", 102], "cls": ["[CLS]", 101]}
        )
        assert isinstance(bert, BertProcessing)
        assert bert.process([7], [8]) == [101, 7, 102, 8, 102]
        roberta = post_processor_from_config(
            {
                "type": "RobertaProcessing",
                "sep": ["</s>", 2],
                "cls": ["<s>", 0],
                "trim_offsets": True,
                "add_prefix_space": True,
            }
        )
        assert isinstance(roberta, RobertaProcessing)
        assert roberta.process([7], [8]) == [0, 7, 2, 2, 8, 2]

    def test_byte_level_and_null_add_nothing(self) -> None:
        byte_level = post_processor_from_config(
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True}
        )
        assert isinstance(byte_level, ByteLevelProcessing)
        assert byte_level.process([7], [8]) == [7, 8]
        assert post_processor_from_config(None).process([7]) == [7]

    def test_sequence_keeps_the_pair_apart(self) -> None:
        """A ByteLevel step must not merge a pair before the template runs."""
        proc = post_processor_from_config(
            {"type": "Sequence", "processors": [{"type": "ByteLevel"}, _T5_TEMPLATE]}
        )
        assert isinstance(proc, ProcessorSequence)
        assert proc.process([7], [8]) == [7, 1, 8, 1]

    def test_unsupported_type_raises(self) -> None:
        with pytest.raises(ValueError, match="unsupported"):
            post_processor_from_config({"type": "Wordpiece"})


# ── pre-tokenizers ──────────────────────────────────────────────────


class TestPreTokenizers:
    def test_metaspace_prepend_schemes(self) -> None:
        # Expected chunks recorded from Hugging Face ``tokenizers``.
        assert _chunks(Metaspace(), "a  b c ") == ["▁a", "▁", "▁b", "▁c", "▁"]
        assert _chunks(Metaspace(prepend_scheme="never"), "a  b") == ["a", "▁", "▁b"]
        assert _chunks(Metaspace(split=False), "a b") == ["▁a▁b"]
        assert _chunks(Metaspace(), " x") == ["▁x"]
        assert _chunks(Metaspace(), "") == []

    def test_metaspace_first_marks_only_the_start(self) -> None:
        """With ``first``, words after a whitespace split get no marker."""
        seq = PreTokenizerSequence([WhitespaceSplit(), Metaspace(prepend_scheme="first")])
        assert _chunks(seq, "a  b c ") == ["▁a", "b", "c"]

    def test_metaspace_replaces_only_the_ascii_space(self) -> None:
        assert _chunks(Metaspace(split=False), "a\tb") == ["▁a\tb"]

    def test_split_behaviors(self) -> None:
        text = "a  b "
        assert _chunks(Split(" ", "removed"), text) == ["a", "b"]
        assert _chunks(Split(" ", "isolated"), text) == ["a", " ", " ", "b", " "]
        assert _chunks(Split(" ", "merged_with_previous"), text) == ["a ", " ", "b "]
        assert _chunks(Split(" ", "merged_with_next"), text) == ["a", " ", " b", " "]
        assert _chunks(Split(" ", "contiguous"), text) == ["a", "  ", "b", " "]
        assert _chunks(Split(re.compile(r"\d+"), "isolated", invert=True), "ab12c") == [
            "ab",
            "12",
            "c",
        ]
        with pytest.raises(ValueError, match="behavior"):
            Split(" ", "sideways")

    def test_digits_whitespace_punctuation(self) -> None:
        assert _chunks(Digits(), "ab123c45") == ["ab", "123", "c", "45"]
        assert _chunks(Digits(individual_digits=True), "a12") == ["a", "1", "2"]
        assert _chunks(Whitespace(), "héllo, wörld!! a_b é") == [
            "héllo",
            ",",
            "wörld",
            "!!",
            "a_b",
            "é",
        ]
        assert _chunks(Punctuation(), "hi, you!") == ["hi", ",", " you", "!"]

    def test_from_config(self) -> None:
        t5 = pre_tokenizer_from_config(
            {
                "type": "Sequence",
                "pretokenizers": [
                    {"type": "WhitespaceSplit"},
                    {
                        "type": "Metaspace",
                        "replacement": "▁",
                        "str_rep": "▁",
                        "add_prefix_space": True,
                    },
                ],
            }
        )
        assert _chunks(t5, "the  quick fox") == ["▁the", "▁quick", "▁fox"]
        never = pre_tokenizer_from_config(
            {"type": "Metaspace", "replacement": "▁", "add_prefix_space": False}
        )
        assert _chunks(never, "hi you") == ["hi", "▁you"]
        split = pre_tokenizer_from_config(
            {"type": "Split", "pattern": {"Regex": r"\s+"}, "behavior": "Removed", "invert": False}
        )
        assert _chunks(split, "a  b") == ["a", "b"]
        assert _chunks(pre_tokenizer_from_config(None), "as is") == ["as is"]
        assert _chunks(pre_tokenizer_from_config(None), "") == []
        with pytest.raises(ValueError, match="unsupported"):
            pre_tokenizer_from_config({"type": "CharDelimiterSplit"})

    def test_matches_hugging_face(self) -> None:
        pre_tokenizers = pytest.importorskip("tokenizers.pre_tokenizers")
        cases = [
            (Metaspace(), pre_tokenizers.Metaspace()),
            (Metaspace(prepend_scheme="never"), pre_tokenizers.Metaspace(prepend_scheme="never")),
            (Whitespace(), pre_tokenizers.Whitespace()),
            (Digits(), pre_tokenizers.Digits()),
            (Punctuation(), pre_tokenizers.Punctuation()),
            (Split(" ", "merged_with_next"), pre_tokenizers.Split(" ", "merged_with_next")),
            (Split(" ", "contiguous"), pre_tokenizers.Split(" ", "contiguous")),
        ]
        texts = [
            "Hello  world",
            " leading",
            "trailing  ",
            "naïve café, 12 or 345!",
            "a_b-c--d",
            "   ",
            "é x",
        ]
        for ours, theirs in cases:
            for text in texts:
                expected = [c for c, _ in theirs.pre_tokenize_str(text)]
                assert _chunks(ours, text) == expected, (ours, text)


# ── decoders ────────────────────────────────────────────────────────


class TestDecoders:
    def test_metaspace(self) -> None:
        assert MetaspaceDecoder().decode(["▁hello", "▁wor", "ld"]) == "hello world"
        assert MetaspaceDecoder(prepend_scheme="never").decode(["▁a", "▁b"]) == " a b"

    def test_llama_chain(self) -> None:
        chain = decoder_from_config(
            {
                "type": "Sequence",
                "decoders": [
                    {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
                    {"type": "ByteFallback"},
                    {"type": "Fuse"},
                    {"type": "Strip", "content": " ", "start": 1, "stop": 0},
                ],
            }
        )
        assert isinstance(chain, DecoderSequence)
        assert chain.decode(["▁caf", "<0xC3>", "<0xA9>", "▁ok"]) == "café ok"

    def test_byte_fallback_invalid_run(self) -> None:
        assert ByteFallback().decode(["<0xC3>", "x"]) == "�x"

    def test_strip_and_fuse(self) -> None:
        assert StripDecoder(" ", start=1, stop=1).decode(["  a  "]) == " a "
        assert Fuse().decode_chain(["a", "b"]) == ["ab"]
        assert ReplaceDecoder(re.compile("_+"), " ").decode(["a__b"]) == "a b"

    def test_byte_level(self) -> None:
        assert ByteLevelDecoder().decode(["Hello", "Ġworld"]) == "Hello world"

    def test_null_and_unsupported(self) -> None:
        assert decoder_from_config(None) is None
        with pytest.raises(ValueError, match="unsupported"):
            decoder_from_config({"type": "CTC"})

    def test_matches_hugging_face(self) -> None:
        decoders = pytest.importorskip("tokenizers.decoders")
        tokens = ["▁The", "▁caf", "<0xC3>", "<0xA9>", "▁", "▁x", "Ġy"]
        cases = [
            (MetaspaceDecoder(), decoders.Metaspace()),
            (ByteFallback(), decoders.ByteFallback()),
            (Fuse(), decoders.Fuse()),
            (StripDecoder(" ", 1, 0), decoders.Strip(" ", 1, 0)),
        ]
        for ours, theirs in cases:
            assert ours.decode(tokens) == theirs.decode(tokens), ours


# ── normalizers ─────────────────────────────────────────────────────


def _tiny_charsmap() -> bytes:
    """A hand-built precompiled map with one rule: ``"A"`` → ``"a"``.

    Root unit 0 carries offset 1, so byte ``0x41`` lands on unit
    ``1 ^ 0x41 = 64``; that unit has label ``0x41``, the leaf flag and
    offset 1, putting its value unit at ``64 ^ 1 = 65``, whose value 0 is
    the position of ``"a\\0"`` in the pool.
    """
    units = [0] * 66
    units[0] = 1 << 10
    units[64] = (1 << 10) | (1 << 8) | 0x41
    units[65] = (1 << 31) | 0
    trie = struct.pack(f"<{len(units)}I", *units)
    return struct.pack("<I", len(trie)) + trie + b"a\0"


class TestNormalizers:
    def test_replace_regex_and_literal(self) -> None:
        assert Replace(re.compile(" {2,}"), " ")("a    b  c") == "a b c"
        assert Replace("``", '"')("``x") == '"x'
        # The replacement is literal even when it looks like a group reference.
        assert Replace(re.compile("x"), r"\1")("axb") == r"a\1b"

    def test_prepend_and_nmt(self) -> None:
        assert Prepend("▁")("hi") == "▁hi"
        assert Prepend("▁")("") == ""
        assert Nmt()("a\x01b\tc​d") == "ab c d"

    def test_strip_accents_removes_every_mark(self) -> None:
        # A Thai vowel sign has combining class 0 and is still a mark.
        assert StripAccents()("éั") == "e"

    def test_bert_normalizer_cased_keeps_composed_accents(self) -> None:
        cased = BERTNormalizer(lowercase=False, strip_accents=False)
        assert cased("Café") == "Café"
        assert BERTNormalizer()("Café") == "cafe"

    def test_bert_normalizer_config_null_strip_follows_lowercase(self) -> None:
        cased = normalizer_from_config(
            {"type": "BertNormalizer", "lowercase": False, "strip_accents": None}
        )
        assert cased is not None and cased("Café") == "Café"
        uncased = normalizer_from_config(
            {"type": "BertNormalizer", "lowercase": True, "strip_accents": None}
        )
        assert uncased is not None and uncased("Café") == "cafe"

    def test_precompiled_trie(self) -> None:
        norm = Precompiled(_tiny_charsmap())
        assert norm("ABA") == "aBa"
        assert norm("") == ""
        assert Precompiled(b"")("ABA") == "ABA"
        with pytest.raises(ValueError):
            Precompiled(b"\x00\x01")
        with pytest.raises(ValueError):
            Precompiled(struct.pack("<I", 400) + b"\0" * 8)

    def test_precompiled_empty_config_is_identity(self) -> None:
        norm = normalizer_from_config({"type": "Precompiled", "precompiled_charsmap": ""})
        assert norm is not None and norm("Ｆｕｌｌ") == "Ｆｕｌｌ"

    def test_from_config(self) -> None:
        chain = normalizer_from_config(
            {
                "type": "Sequence",
                "normalizers": [
                    {"type": "NFKC"},
                    {"type": "Lowercase"},
                    {"type": "Strip", "strip_left": True, "strip_right": False},
                    {"type": "Prepend", "prepend": "▁"},
                ],
            }
        )
        assert chain is not None and chain("  ＡB ") == "▁ab "
        assert normalizer_from_config(None) is None
        with pytest.raises(ValueError, match="unsupported"):
            normalizer_from_config({"type": "ByteLevel"})
        with pytest.raises(ValueError, match="does not compile"):
            normalizer_from_config({"type": "Replace", "pattern": {"Regex": "("}, "content": ""})

    def test_matches_hugging_face(self) -> None:
        normalizers = pytest.importorskip("tokenizers.normalizers")
        cases = [
            (StripAccents(), normalizers.StripAccents()),
            (Nmt(), normalizers.Nmt()),
            (
                BERTNormalizer(lowercase=False, strip_accents=False),
                normalizers.BertNormalizer(lowercase=False),
            ),
            (BERTNormalizer(), normalizers.BertNormalizer()),
        ]
        texts = ["Café Ünïcode", "คุณ कि", "a\x01b\tc​", "東京 ｘ", "\x1c x \x1c"]
        for ours, theirs in cases:
            for text in texts:
                assert ours(text) == theirs.normalize_str(text), (ours, text)


# ── added tokens ────────────────────────────────────────────────────


class TestAddedTokens:
    def test_split_raw(self) -> None:
        vocab = AddedVocabulary(
            [AddedToken("</s>", 1, special=True), AddedToken("<pad>", 0, special=True)]
        )
        assert vocab.split_raw("a</s><pad>b") == [
            ("a", None, 0),
            ("</s>", 1, 1),
            ("<pad>", 0, 5),
            ("b", None, 10),
        ]
        assert vocab.split_raw("") == [("", None, 0)]

    def test_longest_match_wins(self) -> None:
        vocab = AddedVocabulary([AddedToken("<a>", 1), AddedToken("<a><b>", 2)])
        assert [seg[1] for seg in vocab.split_raw("x<a><b>")] == [None, 2]

    def test_strip_and_single_word(self) -> None:
        vocab = AddedVocabulary(
            [AddedToken("<m>", 5, lstrip=True, rstrip=True), AddedToken("cat", 6, single_word=True)]
        )
        assert vocab.split_raw("a  <m>  b") == [("a", None, 0), ("  <m>  ", 5, 1), ("b", None, 8)]
        assert [seg[1] for seg in vocab.split_raw("concat cat")] == [None, 6]

    def test_normalized_tokens_match_after_normalizing(self) -> None:
        from lucid.utils.tokenizer._normalizers import Lowercase

        vocab = AddedVocabulary([AddedToken("HeLLo", 9, normalized=True)], Lowercase())
        assert vocab.split_raw("HELLO") == [("HELLO", None, 0)]
        assert vocab.split_normalized("hello") == [("hello", 9, 0)]


# ── Unigram from a tokenizer.json ───────────────────────────────────

_PIECES: list[list[object]] = [
    ["<pad>", 0.0],
    ["</s>", 0.0],
    ["<s>", 0.0],
    ["<unk>", 0.0],
    ["▁", -2.0],
    ["▁hi", -3.0],
    ["▁there", -3.5],
    ["▁the", -4.0],
    ["re", -5.0],
    ["h", -6.0],
    ["i", -6.0],
    ["t", -6.0],
    ["e", -6.0],
    ["r", -6.0],
    ["x", -6.5],
]
_ID = {piece: i for i, (piece, _) in enumerate(_PIECES)}


def _added(entries: list[tuple[str, int]]) -> list[dict[str, object]]:
    return [
        {
            "id": tid,
            "content": content,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
        for content, tid in entries
    ]


def _t5_style_json(**overrides: object) -> dict[str, object]:
    """A UMT5-shaped tokenizer.json: regex Replace, Metaspace, ``$A </s>``."""
    data: dict[str, object] = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": _added([("<pad>", 0), ("</s>", 1), ("<s>", 2), ("<unk>", 3)]),
        "normalizer": {"type": "Replace", "pattern": {"Regex": " {2,}"}, "content": " "},
        "pre_tokenizer": {
            "type": "Metaspace",
            "replacement": "▁",
            "prepend_scheme": "always",
            "split": True,
        },
        "post_processor": _T5_TEMPLATE,
        "decoder": {
            "type": "Metaspace",
            "replacement": "▁",
            "prepend_scheme": "always",
            "split": True,
        },
        "model": {"type": "Unigram", "unk_id": 3, "vocab": _PIECES},
    }
    data.update(overrides)
    return data


def _write(directory: str, data: dict[str, object], with_bos_map: bool = True) -> None:
    with open(os.path.join(directory, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    if with_bos_map:
        # What made the reported BOS appear: the map names one.
        special_map = {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
        }
        with open(os.path.join(directory, "special_tokens_map.json"), "w") as f:
            json.dump(special_map, f)


_FLAVOURS = [UnigramTokenizer, UnigramTokenizerFast]


@pytest.mark.parametrize("cls", _FLAVOURS)
class TestUnigramFromTokenizerJson:
    def test_template_frames_without_bos(self, cls: type) -> None:
        with tempfile.TemporaryDirectory() as d:
            _write(d, _t5_style_json())
            tok = cls.from_file(d)
        assert tok.encode("hi") == [_ID["▁hi"], 1]
        # The map still populates the registry, just not the framing.
        assert tok.bos_token_id == 2 and tok.pad_token_id == 0
        assert tok.unk_token_id == 3

    def test_normalizer_collapses_space_runs(self, cls: type) -> None:
        with tempfile.TemporaryDirectory() as d:
            _write(d, _t5_style_json())
            tok = cls.from_file(d)
        assert tok.encode("hi    there", add_special_tokens=False) == [_ID["▁hi"], _ID["▁there"]]

    def test_literal_added_token_is_cut_out(self, cls: type) -> None:
        with tempfile.TemporaryDirectory() as d:
            _write(d, _t5_style_json())
            tok = cls.from_file(d)
        assert tok.encode("hi</s>there") == [_ID["▁hi"], 1, _ID["▁there"], 1]

    def test_unknowns_fuse(self, cls: type) -> None:
        with tempfile.TemporaryDirectory() as d:
            _write(d, _t5_style_json())
            tok = cls.from_file(d)
        assert tok.encode("hi zzz", add_special_tokens=False) == [_ID["▁hi"], _ID["▁"], 3]

    def test_decode(self, cls: type) -> None:
        with tempfile.TemporaryDirectory() as d:
            _write(d, _t5_style_json())
            tok = cls.from_file(d)
        ids = tok.encode("hi there")
        assert tok.decode(ids) == "hi there"
        assert tok.decode(ids, skip_special_tokens=False) == "hi there</s>"

    def test_truncation_keeps_eos(self, cls: type) -> None:
        with tempfile.TemporaryDirectory() as d:
            _write(d, _t5_style_json())
            tok = cls.from_file(d)
        out = tok("hi there hi there", truncation=True, max_length=3)
        assert out["input_ids"] == [_ID["▁hi"], _ID["▁there"], 1]

    def test_null_post_processor_adds_nothing(self, cls: type) -> None:
        with tempfile.TemporaryDirectory() as d:
            _write(d, _t5_style_json(post_processor=None))
            tok = cls.from_file(d)
        assert tok.encode("hi") == [_ID["▁hi"]]

    def test_missing_blocks_keep_the_old_behaviour(self, cls: type) -> None:
        """A file without pipeline blocks is an older Lucid save: BOS/EOS
        from the map, NFKC and the SentencePiece pre-tokenizer."""
        data = {"algo": "unigram", "model": {"type": "Unigram", "vocab": _PIECES}}
        with tempfile.TemporaryDirectory() as d:
            _write(d, data)
            tok = cls.from_file(d)
        assert tok.encode("hi") == [2, _ID["▁hi"], 1]

    def test_explicit_normalizer_wins(self, cls: type) -> None:
        from lucid.utils.tokenizer._normalizers import Lowercase

        with tempfile.TemporaryDirectory() as d:
            _write(d, _t5_style_json())
            tok = cls.from_file(d, normalizer=Lowercase())
        assert tok.encode("HI", add_special_tokens=False) == [_ID["▁hi"]]

    def test_save_round_trip_keeps_the_pipeline(self, cls: type) -> None:
        with tempfile.TemporaryDirectory() as d:
            _write(d, _t5_style_json())
            tok = cls.from_file(d)
        with tempfile.TemporaryDirectory() as d2:
            tok.save(d2)
            again = cls.from_file(d2)
        for text in ["hi", "hi    there", "hi</s>there", "hi zzz", ""]:
            assert again.encode(text) == tok.encode(text), text
            assert again.decode(tok.encode(text)) == tok.decode(tok.encode(text))

    def test_matches_hugging_face(self, cls: type) -> None:
        tokenizers = pytest.importorskip("tokenizers")
        data = _t5_style_json()
        theirs = tokenizers.Tokenizer.from_str(json.dumps(data))
        with tempfile.TemporaryDirectory() as d:
            _write(d, data)
            ours = cls.from_file(d)
        for text in ["hi", "hi    there", "hi</s>there", "hi zzz", "", " x x "]:
            expected = theirs.encode(text).ids
            assert ours.encode(text) == expected, text
            assert ours.decode(expected) == theirs.decode(expected), text


@pytest.mark.parametrize("cls", _FLAVOURS)
def test_byte_fallback_matches_hugging_face(cls: type) -> None:
    """Unknown characters spelled as byte pieces when every byte exists."""
    tokenizers = pytest.importorskip("tokenizers")
    pieces = [*_PIECES, ["<0xC3>", -9.0], ["<0xA9>", -9.0]]
    data = _t5_style_json(
        model={"type": "Unigram", "unk_id": 3, "vocab": pieces, "byte_fallback": True},
        decoder={
            "type": "Sequence",
            "decoders": [
                {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
                {"type": "ByteFallback"},
                {"type": "Fuse"},
            ],
        },
    )
    theirs = tokenizers.Tokenizer.from_str(json.dumps(data))
    with tempfile.TemporaryDirectory() as d:
        _write(d, data, with_bos_map=False)
        ours = cls.from_file(d)
    for text in ["hi é", "hié😀", "éé x", "😀"]:
        expected = theirs.encode(text).ids
        assert ours.encode(text) == expected, text
        assert ours.decode(expected) == theirs.decode(expected), text


# ── the published t5-small tokenizer ────────────────────────────────

_T5_TEXTS = [
    "hi",
    "Hello, world!",
    "multiple   spaces    here",
    "  leading and trailing  ",
    "naïve café résumé",
    "café with a decomposed accent",
    "Ｆｕｌｌ－ｗｉｄｔｈ　ＡＢＣ　１２３",
    "Ａ́ — the grapheme rule",
    "3.14159 and 2,718,281",
    "①②③ ½ ﬁ ™",
    "tab\tand\nnewline\r\nend",
    "東京 and Москва",
    "hello</s>world <extra_id_0> x <pad>",
    "😀😀 emoji 👍🏽",
    "",
]


@pytest.fixture(scope="module")
def t5_pair() -> tuple[object, object, object]:
    tokenizers = pytest.importorskip("tokenizers")
    snapshot = _hub_snapshot("t5-small")
    if snapshot is None:
        pytest.skip("t5-small is not in the Hugging Face cache")
    theirs = tokenizers.Tokenizer.from_file(os.path.join(snapshot, "tokenizer.json"))
    return (theirs, UnigramTokenizer.from_file(snapshot), UnigramTokenizerFast.from_file(snapshot))


class TestT5SmallParity:
    def test_encode(self, t5_pair: tuple[object, object, object]) -> None:
        theirs, slow, fast = t5_pair
        for text in _T5_TEXTS:
            expected = theirs.encode(text).ids  # type: ignore[attr-defined]
            assert fast.encode(text) == expected, text  # type: ignore[attr-defined]
            assert slow.encode(text) == expected, text  # type: ignore[attr-defined]

    def test_decode(self, t5_pair: tuple[object, object, object]) -> None:
        theirs, _, fast = t5_pair
        for text in _T5_TEXTS:
            ids = theirs.encode(text).ids  # type: ignore[attr-defined]
            for skip in (True, False):
                expected = theirs.decode(ids, skip_special_tokens=skip)  # type: ignore[attr-defined]
                assert fast.decode(ids, skip_special_tokens=skip) == expected  # type: ignore[attr-defined]

    def test_reported_case_has_no_bos(self, t5_pair: tuple[object, object, object]) -> None:
        _, _, fast = t5_pair
        assert fast.encode("hi") == [7102, 1]  # type: ignore[attr-defined]

    def test_precompiled_normalizer_fuzz(self, t5_pair: tuple[object, object, object]) -> None:
        """Random code points, seeded, through both normalizers."""
        theirs, _, fast = t5_pair
        ours = fast._normalizer  # type: ignore[attr-defined]
        rnd = random.Random(0)
        for _ in range(3000):
            chars = [chr(rnd.randint(0, 0x2FFFF)) for _ in range(rnd.randint(1, 6))]
            text = "".join(c for c in chars if not 0xD800 <= ord(c) <= 0xDFFF)
            expected = theirs.normalizer.normalize_str(text)  # type: ignore[attr-defined]
            assert ours(text) == expected, repr(text)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
