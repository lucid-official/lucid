"""UnigramTokenizer — SentencePiece-flavour subword tokenizer (Kudo 2018).

The Unigram Language Model tokenizer powers T5 / mBART / ALBERT /
XLNet / LLaMA / Mistral (all via SentencePiece).  Unlike BPE — which
builds a deterministic merge sequence — Unigram maintains a *fixed*
vocabulary of sub-word pieces, each with an associated
log-probability, and at encode time picks the segmentation that
maximises the product of piece probabilities (Viterbi over the
piece-with-max-log-probability lattice).

This file holds **both flavours**:

* :class:`UnigramTokenizer` — pure-Python reference implementation.
  Viterbi decode runs in Python (slow but easy to step through);
  training delegates to C++ because EM with vocab pruning is a tight
  numerical loop and a Python rewrite would be 100-1000x slower for
  any non-trivial corpus.

* :class:`UnigramTokenizerFast` — C++-backed via the engine
  ``Unigram`` binding.  Same on-disk format, bit-identical encode
  output, used in production.

**On-disk format.** A single unified ``tokenizer.json`` with
``model.vocab = [[piece, log_prob], ...]`` (matching the Hugging
Face Fast-tokenizers / Rust ``tokenizers`` schema).  Loaded
verbatim by :meth:`UnigramTokenizer.from_file` /
:meth:`UnigramTokenizerFast.from_file`.

**SentencePiece convention.** Words are prefixed with ``▁``
(U+2581, LOWER ONE EIGHTH BLOCK) to mark word starts so that decode
can perfectly reconstruct the original whitespace without
ambiguity.  The default :class:`SentencePiecePreTokenizer`
applies this remapping; pass a different :class:`PreTokenizer` to
disable (matches plain "non-SentencePiece" Unigram behaviour).

**Loading a published checkpoint.**  A Hugging Face ``tokenizer.json``
describes the whole pipeline, not just the pieces: its ``normalizer``,
``pre_tokenizer``, ``post_processor`` and ``decoder`` blocks and its
``added_tokens``.  ``from_file`` builds every one of them, because each
changes the ids — T5 normalises with a compiled SentencePiece map rather
than NFKC, frames with ``… </s>`` and no BOS, and cuts ``</s>`` out of
the raw text before anything else sees it.  Files Lucid wrote before it
read these blocks carry none of them and load exactly as they did.
"""

import json
import os
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Self, override

from lucid._C import engine as _C_engine

from lucid.utils.tokenizer._added_tokens import (
    AddedToken,
    AddedVocabulary,
    added_tokens_from_config,
)
from lucid.utils.tokenizer._base import SpecialTokens, Tokenizer
from lucid.utils.tokenizer._bpe import _special_tokens_from_map
from lucid.utils.tokenizer._decoders import Decoder, decoder_from_config
from lucid.utils.tokenizer._normalizers import (
    NFKC,
    Normalizer,
    normalizer_from_config,
)
from lucid.utils.tokenizer._post_processors import post_processor_from_config
from lucid.utils.tokenizer._pre_tokenizers import (
    PreTokenizer,
    pre_tokenizer_from_config,
)

#: Hugging Face scores an unknown character at the lowest piece score
#: minus this, so an unknown is always the costliest step on a path.
_UNK_PENALTY = 10.0

#: The ``tokenizer.json`` pipeline blocks ``from_file`` reads and ``save``
#: writes back.
_PIPELINE_BLOCKS = ("normalizer", "pre_tokenizer", "decoder")

# ── SentencePiece-style pre-tokenizer ──────────────────────────────


class SentencePiecePreTokenizer(PreTokenizer):
    """SentencePiece pre-tokenization.

    Replaces every whitespace run with a single ``▁`` (U+2581)
    prefix on each word so decode can perfectly reconstruct the
    original spacing.  This is the canonical pre-tokenizer for any
    Unigram / SentencePiece checkpoint (T5, LLaMA, mBART, ...).

    Parameters
    ----------
    add_dummy_prefix : bool, default True
        Prepend ``▁`` to the very first word (matches the canonical
        SentencePiece behaviour).  Set ``False`` for plain Unigram
        without the SentencePiece word-start marker.

    Notes
    -----
    Each emitted chunk keeps the leading ``▁`` so that decode is a
    simple ``"".join(pieces).replace("▁", " ")`` — see
    :meth:`UnigramTokenizer._decode_one`.
    """

    SP_SPACE = "▁"  # ▁

    def __init__(self, add_dummy_prefix: bool = True) -> None:
        r"""Record the SentencePiece prefix-prepend flag.

        Parameters
        ----------
        add_dummy_prefix : bool, default True
            When ``True`` (the canonical setting), prepend ``▁`` to
            the first word so it carries the word-start marker.
        """
        self._add_dummy_prefix = add_dummy_prefix

    @override
    def pre_tokenize(self, text: str) -> list[tuple[str, tuple[int, int]]]:
        """Replace whitespace with ``▁`` and split on ``▁`` boundaries."""
        # Replace every whitespace run with SP_SPACE.  Optionally
        # prepend SP_SPACE so the first word also carries the
        # word-start marker.
        text = "".join(self.SP_SPACE if c.isspace() else c for c in text)
        # Empty in, empty out.  The ``not text`` arm used to prepend the
        # marker to the empty string, so ``encode("")`` came back holding
        # one token — a ``▁`` that matches nothing and lands on ``<unk>``.
        # BPE and WordPiece both return ``[]`` here, and a tokenizer that
        # invents a token for no input breaks every length-based caller
        # (padding, truncation, attention masks) on the empty document.
        # The arm was redundant besides: ``"".startswith("▁")`` is already
        # False, so the second condition covered every non-empty case on
        # its own.
        if self._add_dummy_prefix and text and not text.startswith(self.SP_SPACE):
            text = self.SP_SPACE + text
        # Split on SP_SPACE boundaries, keeping the SP_SPACE prefix
        # attached to each word for round-trip decode.
        out: list[tuple[str, tuple[int, int]]] = []
        i = 0
        n = len(text)
        while i < n:
            start = i
            # Each chunk = SP_SPACE + word characters (or just word
            # characters for the very first chunk if add_dummy_prefix
            # is False and there's no leading whitespace).
            if text[i] == self.SP_SPACE:
                i += 1
            while i < n and text[i] != self.SP_SPACE:
                i += 1
            out.append((text[start:i], (start, i)))
        return out


# ── Vocab format helpers ───────────────────────────────────────────


def _pieces_from_model(model: dict[str, object], path: str) -> list[tuple[str, float]]:
    """Read ``[(piece, log_prob), ...]`` out of a parsed ``model`` block."""
    vocab = model.get("vocab", [])
    if not isinstance(vocab, list):
        raise TypeError(f"_load_unigram_pieces_json: model.vocab is not a list in {path}")
    out: list[tuple[str, float]] = []
    for entry in vocab:
        if isinstance(entry, list) and len(entry) == 2:
            out.append((str(entry[0]), float(entry[1])))
        elif isinstance(entry, dict) and "piece" in entry and "score" in entry:
            out.append((str(entry["piece"]), float(entry["score"])))
        else:
            raise ValueError(
                f"_load_unigram_pieces_json: malformed vocab entry " f"{entry!r} in {path}"
            )
    return out


def _save_unigram_pieces_json(
    pieces: list[tuple[str, float]],
    path: str,
    unk_token: str,
    unk_log_prob: float,
) -> None:
    """Write the unified ``tokenizer.json`` for Unigram."""
    payload = {
        "algo": "unigram",
        "model": {
            "type": "Unigram",
            "vocab": [list(p) for p in pieces],
            "unk_token": unk_token,
            "unk_log_prob": unk_log_prob,
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ── Loading a tokenizer.json ───────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _UnigramFileConfig:
    """Everything ``from_file`` reads out of a checkpoint directory."""

    pieces: list[tuple[str, float]]
    unk_token: str
    unk_log_prob: float
    fuse_unk: bool
    byte_fallback: bool
    special_tokens: SpecialTokens
    added_tokens: list[AddedToken]
    data: dict[str, object]


def _read_json_object(path: str) -> dict[str, object]:
    """Load a JSON file that must hold an object."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected a JSON object at the top level")
    return {str(k): v for k, v in data.items()}


def _read_unigram_config(directory: str, who: str) -> _UnigramFileConfig:
    """Parse ``tokenizer.json`` (and the special-token files beside it).

    Parameters
    ----------
    directory : str
        The checkpoint directory.
    who : str
        Class name for error messages.

    Returns
    -------
    _UnigramFileConfig
        Pieces, unknown-token handling, the special-token registry and the
        raw file for the pipeline blocks.

    Notes
    -----
    Two dialects are read.  Lucid's own files name the unknown piece
    (``unk_token``) and its score (``unk_log_prob``).  Hugging Face files
    give the unknown piece's *index* (``unk_id``) and no score — the model
    scores an unknown at the lowest piece score minus ten and fuses runs of
    unknown characters into one token, so both are reproduced for them.
    """
    path = os.path.join(directory, "tokenizer.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{who}.from_file: tokenizer.json not found in {directory}")
    data = _read_json_object(path)
    raw_model = data.get("model", {})
    if not isinstance(raw_model, dict):
        raise TypeError(f"{who}.from_file: 'model' is not an object in {path}")
    model: dict[str, object] = {str(k): v for k, v in raw_model.items()}
    pieces = _pieces_from_model(model, path)
    lucid_dialect = "unk_log_prob" in model

    unk_id = model.get("unk_id")
    if "unk_token" in model:
        unk_token = str(model["unk_token"])
    elif isinstance(unk_id, int) and 0 <= unk_id < len(pieces):
        unk_token = pieces[unk_id][0]
    else:
        unk_token = "<unk>"
    if lucid_dialect:
        raw_score = model["unk_log_prob"]
        if not isinstance(raw_score, (int, float)):
            raise ValueError(f"{who}.from_file: unk_log_prob is not a number in {path}")
        unk_log_prob = float(raw_score)
    elif pieces:
        unk_log_prob = min(score for _, score in pieces) - _UNK_PENALTY
    else:
        unk_log_prob = -100.0
    fuse_unk = bool(model.get("fuse_unk", not lucid_dialect))
    byte_fallback = bool(model.get("byte_fallback", False))

    added = added_tokens_from_config(data.get("added_tokens"))

    # The registry decides which ids count as special (masks, skipping on
    # decode); it does not decide framing when the file has a
    # post-processor.  special_tokens_map.json is the usual source;
    # tokenizer_config.json carries the same keys when it is absent.
    special = SpecialTokens()
    for name in ("special_tokens_map.json", "tokenizer_config.json"):
        side = os.path.join(directory, name)
        if os.path.isfile(side):
            special = _special_tokens_from_map(_read_json_object(side))
            break
    if special.unk is None and any(p == unk_token for p, _ in pieces):
        special.unk = unk_token
    known = set(special.all_tokens())
    for token in added:
        if token.special and token.content not in known:
            special.extra[token.content] = token.content
            known.add(token.content)

    return _UnigramFileConfig(
        pieces=pieces,
        unk_token=unk_token,
        unk_log_prob=unk_log_prob,
        fuse_unk=fuse_unk,
        byte_fallback=byte_fallback,
        special_tokens=special,
        added_tokens=added,
        data=data,
    )


# ── Shared mixin ───────────────────────────────────────────────────


class _UnigramCommonMixin(Tokenizer):
    """Shared text pipeline for both flavours.

    Subclasses supply the per-chunk Viterbi (Python or C++) through
    :meth:`_encode_chunk_ids`; everything around it — added-token
    splitting, normalization, pre-tokenization, unknown-token handling and
    decoding — is shared, so the two flavours cannot drift apart.
    """

    _normalizer: Normalizer | None
    _pre_tokenizer: PreTokenizer
    _pieces: list[tuple[str, float]]
    _id_to_piece: dict[int, str]
    _unk_token: str
    _unk_log_prob: float
    _decoder: Decoder | None
    _decoder_null: bool
    _added: AddedVocabulary
    _fuse_unk: bool
    _byte_fallback: bool
    _unk_piece_id: int | None
    _byte_piece_ids: dict[int, int]
    _hf_blocks: dict[str, object]

    def _init_pipeline(self) -> None:
        """Reset the ``tokenizer.json``-driven state to Lucid's defaults.

        Must run before ``Tokenizer.__init__``, whose special-id refresh
        reads :meth:`get_vocab` and so the added tokens.
        """
        self._decoder = None
        self._decoder_null = False
        self._added = AddedVocabulary([])
        self._fuse_unk = False
        self._byte_fallback = False
        self._unk_piece_id = None
        self._byte_piece_ids = {}
        self._hf_blocks = {}

    def _index_unk(self) -> None:
        """Cache the unknown piece's id and the byte pieces' ids.

        Rebuilt whenever the pieces change; looking them up per chunk
        would mean a pass over the vocabulary for every word encoded.
        """
        self._unk_piece_id = None
        self._byte_piece_ids = {}
        for i, (piece, _) in enumerate(self._pieces):
            if piece == self._unk_token and self._unk_piece_id is None:
                self._unk_piece_id = i
            if self._byte_fallback and len(piece) == 6 and piece.startswith("<0x"):
                digits = piece[3:5]
                if piece.endswith(">") and all(c in "0123456789ABCDEFabcdef" for c in digits):
                    self._byte_piece_ids.setdefault(int(digits, 16), i)

    @abstractmethod
    def _encode_chunk_ids(self, chunk: str) -> list[int]:
        """Viterbi-segment one pre-tokenized chunk (flavour-specific)."""

    def _prepare_chunks(self, text: str) -> list[str]:
        """Apply normalizer + pre-tokenizer, return chunk strings."""
        if self._normalizer is not None:
            text = self._normalizer(text)
        return [chunk for chunk, _ in self._pre_tokenizer(text)]

    def _encode_text(self, text: str) -> list[int]:
        """The full encode pipeline, minus special-token framing.

        Added tokens are cut out of the raw text first; each remaining
        piece is normalized, searched again for normalized added tokens,
        pre-tokenized at its position in the input, and Viterbi-segmented.
        """
        out: list[int] = []
        normalizer = self._normalizer
        for segment, token_id, start in self._added.split_raw(text):
            if token_id is not None:
                out.append(token_id)
                continue
            normalized = normalizer(segment) if normalizer is not None else segment
            for piece, piece_id, piece_start in self._added.split_normalized(normalized):
                if piece_id is not None:
                    out.append(piece_id)
                    continue
                chunks = self._pre_tokenizer._pre_tokenize_at(piece, start + piece_start)
                for chunk, _ in chunks:
                    out.extend(self._handle_unknowns(chunk, self._encode_chunk_ids(chunk)))
        return out

    def _handle_unknowns(self, chunk: str, ids: list[int]) -> list[int]:
        """Fuse runs of unknowns, or spell them as byte pieces.

        Hugging Face's Unigram merges consecutive unknown characters into
        one ``<unk>`` (``fuse_unk``), and with ``byte_fallback`` replaces an
        unknown span by its UTF-8 bytes when every ``<0x..>`` piece exists.
        Lucid-trained vocabularies do neither unless told to.
        """
        unk = self._unk_piece_id
        if unk is None or unk not in ids or not (self._fuse_unk or self._byte_fallback):
            return ids
        if not self._byte_fallback:
            fused: list[int] = []
            for tid in ids:
                if not (tid == unk and fused and fused[-1] == unk):
                    fused.append(tid)
            return fused
        # Recover each id's span of the chunk.  A Viterbi unknown covers one
        # character; a *literal* ``<unk>`` in the text matched as a piece
        # covers its own length and must not be spelled out as bytes.
        spans: list[tuple[int, int, int]] = []
        pos = 0
        for tid in ids:
            if tid == unk:
                width = len(self._unk_token) if chunk.startswith(self._unk_token, pos) else 1
            else:
                width = len(self._id_to_piece.get(tid, ""))
            spans.append((tid, pos, pos + width))
            pos += width
        out: list[int] = []
        k = 0
        while k < len(spans):
            tid, begin, end = spans[k]
            k += 1
            if tid != unk:
                out.append(tid)
                continue
            if self._fuse_unk:
                while k < len(spans) and spans[k][0] == unk:
                    end = spans[k][2]
                    k += 1
            text = chunk[begin:end]
            raw = text.encode("utf-8")
            if text != self._unk_token and all(b in self._byte_piece_ids for b in raw):
                out.extend(self._byte_piece_ids[b] for b in raw)
            else:
                out.append(unk)
        return out

    def _token_for_id(self, token_id: int) -> str | None:
        """Surface of a piece or an added token."""
        piece = self._id_to_piece.get(token_id)
        return piece if piece is not None else self._added.content_for(token_id)

    def _with_added(self, vocab: dict[str, int]) -> dict[str, int]:
        """Add the added tokens a vocabulary does not already hold."""
        for token in self._added.tokens:
            vocab.setdefault(token.content, token.id)
        return vocab

    def _decode_with_pipeline(self, ids: list[int]) -> str | None:
        """Decode through the file's decoder; ``None`` means use Lucid's."""
        if self._decoder is None and not self._decoder_null:
            return None
        tokens = [t for t in map(self._token_for_id, ids) if t is not None]
        if self._decoder is None:
            # ``"decoder": null`` — Hugging Face joins the tokens with spaces.
            return " ".join(tokens)
        return self._decoder.decode(tokens)

    def _apply_file_config(
        self,
        config: _UnigramFileConfig,
        *,
        normalizer_given: bool,
        pre_tokenizer_given: bool,
    ) -> None:
        """Install the pipeline a ``tokenizer.json`` describes.

        A block the file does not contain leaves Lucid's default in place —
        that is what keeps older Lucid saves loading as they did.  A block
        present as ``null`` means "no such stage", which is different from
        absent.  Explicit ``normalizer`` / ``pre_tokenizer`` arguments win
        over the file.
        """
        data = config.data
        blocks: dict[str, object] = {}
        if "normalizer" in data and not normalizer_given:
            self._normalizer = normalizer_from_config(data["normalizer"])
            blocks["normalizer"] = data["normalizer"]
        if "pre_tokenizer" in data and not pre_tokenizer_given:
            self._pre_tokenizer = pre_tokenizer_from_config(data["pre_tokenizer"])
            blocks["pre_tokenizer"] = data["pre_tokenizer"]
        if "post_processor" in data:
            self._post_processor = post_processor_from_config(data["post_processor"])
        if "decoder" in data:
            self._decoder = decoder_from_config(data["decoder"])
            self._decoder_null = data["decoder"] is None
            blocks["decoder"] = data["decoder"]
        self._added = AddedVocabulary(config.added_tokens, self._normalizer)
        self._fuse_unk = config.fuse_unk
        self._byte_fallback = config.byte_fallback
        self._hf_blocks = blocks
        self._index_unk()
        self._refresh_special_ids()

    def _unigram_extras(self) -> dict[str, object]:
        """The ``tokenizer.json`` content both flavours save."""
        extras: dict[str, object] = {
            "model": {
                "type": "Unigram",
                "vocab": [list(p) for p in self._pieces],
                "unk_token": self._unk_token,
                "unk_id": self._unk_piece_id,
                "unk_log_prob": self._unk_log_prob,
                "fuse_unk": self._fuse_unk,
                "byte_fallback": self._byte_fallback,
            }
        }
        # Only what was loaded is written back, verbatim: a stage that came
        # from Lucid's defaults has no block, so it reloads as the default.
        for name in _PIPELINE_BLOCKS:
            if name in self._hf_blocks:
                extras[name] = self._hf_blocks[name]
        if self._post_processor is not None:
            extras["post_processor"] = self._post_processor.to_config()
        if self._added.tokens:
            extras["added_tokens"] = [t.to_config() for t in self._added.tokens]
        return extras

    @override
    @property
    def algo(self) -> str:
        """Algorithm identifier (always ``"unigram"``)."""
        return "unigram"

    @override
    def id_to_token(self, token_id: int) -> str | None:
        """Look up the piece string for a token id."""
        return self._token_for_id(token_id)

    @property
    def pieces(self) -> list[tuple[str, float]]:
        """Raw ``(piece, log_prob)`` list."""
        return list(self._pieces)

    if TYPE_CHECKING:

        def __init__(
            self,
            pieces: list[tuple[str, float]] | None = None,
            *,
            unk_token: str = "<unk>",
            unk_log_prob: float = -10.0,
            normalizer: Normalizer | None = None,
            pre_tokenizer: PreTokenizer | None = None,
            special_tokens: SpecialTokens | None = None,
        ) -> None: ...

    @override
    def _encode_one(self, text: str) -> list[int]:
        """Split added tokens, normalize, pre-tokenize, Viterbi-encode."""
        return self._encode_text(text)

    @override
    def save(self, directory: str) -> None:
        """Persist as unified ``tokenizer.json`` + ``special_tokens_map.json``."""
        os.makedirs(directory, exist_ok=True)
        _save_unigram_pieces_json(
            self._pieces,
            os.path.join(directory, "tokenizer.json"),
            self._unk_token,
            self._unk_log_prob,
        )
        super().save(directory)

    @override
    def _save_extras(self) -> dict[str, object]:
        """Add the ``model`` block and any loaded pipeline blocks."""
        return self._unigram_extras()

    @classmethod
    def from_file(
        cls,
        directory: str,
        *,
        normalizer: Normalizer | None = None,
        pre_tokenizer: PreTokenizer | None = None,
        special_tokens: SpecialTokens | None = None,
    ) -> Self:
        """Load from a directory containing ``tokenizer.json``."""
        config = _read_unigram_config(directory, cls.__name__)
        tok = cls(
            config.pieces,
            unk_token=config.unk_token,
            unk_log_prob=config.unk_log_prob,
            normalizer=normalizer,
            pre_tokenizer=pre_tokenizer,
            special_tokens=(
                special_tokens if special_tokens is not None else config.special_tokens
            ),
        )
        tok._apply_file_config(
            config,
            normalizer_given=normalizer is not None,
            pre_tokenizer_given=pre_tokenizer is not None,
        )
        return tok

    from_pretrained = from_file


# ── Pure-Python Unigram ────────────────────────────────────────────


class UnigramTokenizer(_UnigramCommonMixin, Tokenizer):
    r"""Reference (pure-Python) Unigram tokenizer.

    Encodes via Viterbi over the lattice of all candidate
    sub-piece segmentations: for each chunk we build a DP table
    ``dp[i]`` = best (highest log-prob) path ending at byte offset
    ``i``, with single-codepoint UNK fallback when no piece spans
    a region.  Decode is a trivial ``"".join(pieces).replace("▁", " ")``.

    Training is delegated to C++ (see :meth:`train` for the
    rationale) — even this "pure-Python" flavour does not implement
    EM in Python because the runtime would be unusable on any
    real corpus.

    For production / latency-sensitive use, prefer
    :class:`UnigramTokenizerFast` — same vocab format, bit-identical
    encode output, but the Viterbi hot loop runs in C++.

    Parameters
    ----------
    pieces : list of (str, float)
        Ordered ``(piece_str, log_prob)`` list; index = token id.
        Larger (less-negative) log-probs are preferred during
        Viterbi decode.
    unk_token : str, default ``"<unk>"``
        Fallback piece string used when no entry in ``pieces``
        spans a region of the input.  Must appear in ``pieces`` for
        the UNK id to be defined; otherwise encode silently drops
        unmatchable codepoints.
    unk_log_prob : float, default ``-100.0``
        Log probability assigned to UNK substitutions in the
        Viterbi recurrence.  Very negative so any non-UNK path
        dominates when available.
    normalizer : Normalizer, optional
        Pre-encode text normalisation chain.  Defaults to
        :class:`~lucid.utils.tokenizer._normalizers.NFKC` (matches
        LLaMA / Mistral / T5).
    pre_tokenizer : PreTokenizer, optional
        Chunk-splitter applied after normalisation.  Defaults to
        :class:`SentencePiecePreTokenizer` for canonical
        SentencePiece behaviour with ``▁`` word-start markers.
    special_tokens : SpecialTokens, optional
        Special-token registry — see
        :class:`lucid.utils.tokenizer.SpecialTokens`.  Defaults to
        ``SpecialTokens(unk=unk_token)``.

    Notes
    -----
    The Viterbi DP runs in :math:`O(N \cdot M)` per chunk, where
    ``N`` is the chunk byte length and ``M`` is the max piece byte
    length.  UTF-8 boundary masking ensures sub-piece offsets only
    land on codepoint boundaries (no mid-codepoint cuts), which
    matches the C++ flavour bit-for-bit.

    See Also
    --------
    UnigramTokenizerFast : C++-backed flavour with identical
        encode output and a much faster :meth:`~UnigramTokenizerFast.encode`.

    Examples
    --------
    >>> from lucid.utils.tokenizer import UnigramTokenizer
    >>> tok = UnigramTokenizer(pieces=[])
    >>> tok.train(["the quick brown fox", "the lazy dog sleeps"],
    ...           vocab_size=48)
    >>> tok.decode(tok.encode("the quick dog"))
    ' the quick dog'

    The leading space is the SentencePiece convention this follows — a
    piece carries the boundary before it — and it survives the round
    trip rather than being trimmed away.
    """

    def __init__(
        self,
        pieces: list[tuple[str, float]],
        *,
        unk_token: str = "<unk>",
        unk_log_prob: float = -100.0,
        normalizer: Normalizer | None = None,
        pre_tokenizer: PreTokenizer | None = None,
        special_tokens: SpecialTokens | None = None,
    ) -> None:
        r"""Construct a pure-Python Unigram tokenizer.

        Parameters
        ----------
        pieces : list of (str, float)
            Ordered ``(piece, log_prob)`` list; index = token id.
        unk_token : str, default "<unk>"
            Fallback piece string for unmatchable input regions.
        unk_log_prob : float, default -100.0
            Log probability assigned to UNK substitutions.
        normalizer : Normalizer or None, optional, keyword-only
            Pre-encode normalisation.  Defaults to :class:`NFKC`.
        pre_tokenizer : PreTokenizer or None, optional, keyword-only
            Chunk splitter.  Defaults to
            :class:`SentencePiecePreTokenizer`.
        special_tokens : SpecialTokens or None, optional, keyword-only
            Special-token registry.  Defaults to
            ``SpecialTokens(unk=unk_token)``.

        Notes
        -----
        Builds piece-id maps and caches the longest piece in bytes
        for the Viterbi DP via :meth:`_rebuild_tables`.
        """
        self._pieces = list(pieces)
        self._unk_token = unk_token
        self._unk_log_prob = unk_log_prob
        self._init_pipeline()
        self._normalizer = normalizer if normalizer is not None else NFKC()
        self._pre_tokenizer = (
            pre_tokenizer if pre_tokenizer is not None else SentencePiecePreTokenizer()
        )
        self._piece_to_id: dict[str, int] = {}
        self._id_to_piece: dict[int, str] = {}
        self._max_piece_bytes = 0
        self._rebuild_tables()
        if special_tokens is None:
            special_tokens = SpecialTokens(unk=unk_token)
        super().__init__(special_tokens=special_tokens)

    def _rebuild_tables(self) -> None:
        """Recompute piece-to-id maps + max-piece byte cache."""
        self._piece_to_id = {p: i for i, (p, _) in enumerate(self._pieces)}
        self._id_to_piece = {i: p for i, (p, _) in enumerate(self._pieces)}
        self._max_piece_bytes = max((len(p.encode("utf-8")) for p, _ in self._pieces), default=0)
        self._index_unk()

    @override
    @property
    def vocab_size(self) -> int:
        r"""Number of pieces in the Unigram vocabulary.

        Returns
        -------
        int
            ``len(self._pieces)``.
        """
        return len(self._pieces)

    @override
    def get_vocab(self) -> dict[str, int]:
        r"""Return a copy of the piece → id map.

        Returns
        -------
        dict[str, int]
            Shallow copy; mutating it does not affect the tokenizer.
            Added tokens a ``tokenizer.json`` declares outside the piece
            list are included.
        """
        return self._with_added(dict(self._piece_to_id))

    def _viterbi_encode_chunk(self, chunk: str) -> list[int]:
        """Reference Viterbi DP — operates on bytes for parity with
        the C++ flavour (UTF-8 boundary mask included)."""
        if not chunk:
            return []
        raw = chunk.encode("utf-8")
        N = len(raw)
        # Mask of "is this byte a codepoint start?"
        is_cp = [False] * (N + 1)
        i = 0
        while i < N:
            is_cp[i] = True
            c0 = raw[i]
            if c0 < 0x80:
                cp_len = 1
            elif (c0 >> 5) == 0b110:
                cp_len = 2
            elif (c0 >> 4) == 0b1110:
                cp_len = 3
            elif (c0 >> 3) == 0b11110:
                cp_len = 4
            else:
                cp_len = 1
            i += cp_len
        is_cp[N] = True

        neg_inf = float("-inf")
        dp = [neg_inf] * (N + 1)
        dp[0] = 0.0
        back: list[tuple[int, int]] = [(-1, -1)] * (N + 1)
        unk_id = self._piece_to_id.get(self._unk_token, -1)

        for i in range(1, N + 1):
            if not is_cp[i]:
                continue
            j_min = max(0, i - self._max_piece_bytes)
            for j in range(j_min, i):
                if not is_cp[j]:
                    continue
                if dp[j] == neg_inf:
                    continue
                sub_bytes = raw[j:i]
                try:
                    sub = sub_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                tid = self._piece_to_id.get(sub)
                if tid is not None:
                    score = dp[j] + self._pieces[tid][1]
                    if score > dp[i]:
                        dp[i] = score
                        back[i] = (j, tid)
                elif unk_id >= 0:
                    # Single-codepoint UNK fallback.
                    c0 = raw[j]
                    if c0 < 0x80:
                        cp_len = 1
                    elif (c0 >> 5) == 0b110:
                        cp_len = 2
                    elif (c0 >> 4) == 0b1110:
                        cp_len = 3
                    elif (c0 >> 3) == 0b11110:
                        cp_len = 4
                    else:
                        cp_len = 1
                    if i - j == cp_len:
                        score = dp[j] + self._unk_log_prob
                        if score > dp[i]:
                            dp[i] = score
                            back[i] = (j, unk_id)
        if dp[N] == neg_inf:
            return []
        ids: list[int] = []
        i = N
        while i > 0:
            j, pid = back[i]
            if pid >= 0:
                ids.append(pid)
            i = j
        ids.reverse()
        return ids

    @override
    def _encode_chunk_ids(self, chunk: str) -> list[int]:
        """Python Viterbi over one chunk."""
        return self._viterbi_encode_chunk(chunk)

    @override
    def _decode_one(self, ids: list[int]) -> str:
        """Concatenate pieces and convert ``▁`` markers back to spaces."""
        decoded = self._decode_with_pipeline(ids)
        if decoded is not None:
            return decoded
        # Standard SentencePiece decode: join surface forms, replace
        # the ▁ marker with a space.
        raw = "".join(self._id_to_piece[i] for i in ids if i in self._id_to_piece)
        return raw.replace(SentencePiecePreTokenizer.SP_SPACE, " ")

    def train(
        self,
        corpus: Iterable[str],
        *,
        vocab_size: int = 30_000,
    ) -> None:
        """Re-train this tokenizer from scratch on ``corpus``."""
        prepared: list[str] = []
        for doc in corpus:
            chunks = self._prepare_chunks(doc)
            prepared.append(" ".join(chunks))
        cpp = _C_engine.utils.tokenizer.Unigram([], self._unk_token, self._unk_log_prob)
        cpp.train(prepared, vocab_size)
        self._pieces = [(p, lp) for p, lp in cpp.pieces()]
        self._rebuild_tables()
        self._refresh_special_ids()


# ── Fast (C++-backed) Unigram ──────────────────────────────────────


class UnigramTokenizerFast(_UnigramCommonMixin, Tokenizer):
    r"""C++-backed Unigram tokenizer.

    Identical algorithm + on-disk format to
    :class:`UnigramTokenizer`; the per-chunk Viterbi loop runs in
    C++ via the engine ``Unigram`` binding.  Encode outputs are
    bit-identical for the same pieces + same normalizer + same
    pre-tokenizer.

    Use this in production training / inference.  Use
    :class:`UnigramTokenizer` for debugging / extending the
    algorithm with custom Python-only normalizers without touching
    C++.

    Parameters
    ----------
    Same as :class:`UnigramTokenizer`.  The C++ backend is
    constructed transparently in ``__init__`` and held as `_cpp`.

    See Also
    --------
    UnigramTokenizer : Pure-Python reference flavour.

    Examples
    --------
    >>> from lucid.utils.tokenizer import UnigramTokenizerFast
    >>> tok = UnigramTokenizerFast(pieces=[])
    >>> tok.train(["the quick brown fox", "the lazy dog sleeps"],
    ...           vocab_size=48)
    >>> tok.decode(tok.encode("the quick dog"))
    ' the quick dog'
    """

    def __init__(
        self,
        pieces: list[tuple[str, float]],
        *,
        unk_token: str = "<unk>",
        unk_log_prob: float = -100.0,
        normalizer: Normalizer | None = None,
        pre_tokenizer: PreTokenizer | None = None,
        special_tokens: SpecialTokens | None = None,
    ) -> None:
        r"""Construct a C++-backed Unigram tokenizer.

        Parameters
        ----------
        pieces : list of (str, float)
            Ordered ``(piece, log_prob)`` list passed to the C++
            backend; index = token id.
        unk_token : str, default "<unk>"
            Fallback piece string for unmatchable regions.
        unk_log_prob : float, default -100.0
            Log probability assigned to UNK substitutions in C++.
        normalizer : Normalizer or None, optional, keyword-only
            Pre-encode normalisation.  Defaults to :class:`NFKC`.
        pre_tokenizer : PreTokenizer or None, optional, keyword-only
            Chunk splitter.  Defaults to
            :class:`SentencePiecePreTokenizer`.
        special_tokens : SpecialTokens or None, optional, keyword-only
            Special-token registry.  Defaults to
            ``SpecialTokens(unk=unk_token)``.

        Notes
        -----
        Constructs the C++ ``Unigram`` backend once and caches it
        on `_cpp`; the Python-side `_id_to_piece` reverse map mirrors
        the pieces list for fast decode.
        """
        self._pieces = list(pieces)
        self._unk_token = unk_token
        self._unk_log_prob = unk_log_prob
        self._init_pipeline()
        self._cpp = _C_engine.utils.tokenizer.Unigram(
            [(p, lp) for p, lp in self._pieces], unk_token, unk_log_prob
        )
        self._id_to_piece = {i: p for i, (p, _) in enumerate(self._pieces)}
        self._index_unk()
        self._normalizer = normalizer if normalizer is not None else NFKC()
        self._pre_tokenizer = (
            pre_tokenizer if pre_tokenizer is not None else SentencePiecePreTokenizer()
        )
        if special_tokens is None:
            special_tokens = SpecialTokens(unk=unk_token)
        super().__init__(special_tokens=special_tokens)

    @override
    @property
    def vocab_size(self) -> int:
        r"""Number of pieces in the live C++ vocabulary.

        Returns
        -------
        int
            ``self._cpp.vocab_size()``.
        """
        return self._cpp.vocab_size()

    @override
    def get_vocab(self) -> dict[str, int]:
        r"""Return a piece → id map from the C++ backend.

        Returns
        -------
        dict[str, int]
            Fresh dict built from ``self._cpp.get_vocab()``, plus any
            added tokens declared outside the piece list.
        """
        return self._with_added(dict(self._cpp.get_vocab()))

    @override
    def _encode_chunk_ids(self, chunk: str) -> list[int]:
        """C++ Viterbi over one chunk."""
        return list(self._cpp.encode(chunk))

    @override
    def _decode_one(self, ids: list[int]) -> str:
        """C++ decode + ``▁`` → space replacement for parity."""
        decoded = self._decode_with_pipeline(ids)
        if decoded is not None:
            return decoded
        raw = self._cpp.decode(list(ids))
        return raw.replace(SentencePiecePreTokenizer.SP_SPACE, " ")

    def train(
        self,
        corpus: Iterable[str],
        *,
        vocab_size: int = 30_000,
    ) -> None:
        """Re-train in C++ (EM with vocab pruning, Kudo 2018)."""
        prepared: list[str] = []
        for doc in corpus:
            chunks = self._prepare_chunks(doc)
            prepared.append(" ".join(chunks))
        self._cpp.train(prepared, vocab_size)
        self._pieces = [(p, lp) for p, lp in self._cpp.pieces()]
        self._id_to_piece = {i: p for i, (p, _) in enumerate(self._pieces)}
        self._index_unk()
        self._refresh_special_ids()
