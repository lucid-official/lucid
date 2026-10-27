"""Byte-Pair Encoding (Sennrich et al. 2016).

This file holds **both flavours** of the BPE tokenizer:

* :class:`BPETokenizer` — pure-Python reference implementation.
  Slower (per-token Python loop) but easy to read, easy to debug,
  easy to extend with new normalizers / pre-tokenizers without
  touching C++.  Use this when training a new tokenizer or
  experimenting with custom configs.

* :class:`BPETokenizerFast` — C++-backed via
  ``lucid._C.engine.utils.tokenizer.BPE``.  Same vocab / merges
  format, same encode output bit-for-bit (verified by parity
  tests).  Use this in production training / inference where
  encode throughput matters.

Both share the HF-compatible on-disk format:

* ``vocab.json`` — JSON dict ``{token: id}``.
* ``merges.txt`` — one merge per line ``"left right"`` (rank = line #
  with line 0 = highest priority); optional ``#version: 0.2`` header.
* ``tokenizer.json`` — unified single-file format (HF Fast tokenizers
  / Rust ``tokenizers`` crate).  When present, takes precedence over
  the legacy pair.

Vocab files from any published Hugging Face BPE checkpoint (GPT,
RoBERTa, BART, distilGPT, ...) load without modification.
"""

import json
import os
from typing import Callable, Iterable, override

from lucid._C import engine as _C_engine

from lucid.utils.tokenizer._base import SpecialTokens, Tokenizer
from lucid.utils.tokenizer._normalizers import NFC, Normalizer
from lucid.utils.tokenizer._pre_tokenizers import (
    PreTokenizer,
    WhitespaceSplit,
)

# ── Vocab loader helpers (shared between BPE + BPEFast) ─────────────


def _load_vocab_json(path: str) -> dict[str, int]:
    """Parse a HF-compatible ``vocab.json``.

    Format: ``{"token_str": int_id, ...}``.  Returns the dict
    verbatim; the algorithm-specific BPE constructor turns it into
    a merge table.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise TypeError(
            f"_load_vocab_json: expected a JSON object at {path}, " f"got {type(raw).__name__}"
        )
    return {str(k): int(v) for k, v in raw.items()}


def _load_merges_txt(path: str) -> list[tuple[str, str]]:
    """Parse a HF-compatible ``merges.txt``.

    Format: ``"left right"`` per line, with an optional ``#version: ``
    header that's silently skipped.  Empty lines + ``#``-comments
    are skipped.  Line order = merge rank (lower index = higher
    priority).
    """
    merges: list[tuple[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split(" ")
            if len(parts) != 2:
                # Some merges may legitimately contain a space if the
                # left half ends in / right half starts with whitespace
                # after byte-encoding.  Use rsplit as a defence: the
                # last space is the separator.  This matches HF's
                # tokenizers crate.
                idx = line.rfind(" ")
                if idx <= 0:
                    raise ValueError(f"_load_merges_txt: malformed line {line!r} " f"at {path}")
                parts = [line[:idx], line[idx + 1 :]]
            merges.append((parts[0], parts[1]))
    return merges


def _load_unified_tokenizer_json(
    path: str,
) -> tuple[dict[str, int], list[tuple[str, str]]]:
    """Parse a Hugging Face Fast-tokenizers ``tokenizer.json``.

    The Rust ``tokenizers`` crate emits this as a single file with
    ``model.vocab`` (dict) + ``model.merges`` (list of "left right"
    strings).  We extract just those two pieces here; normalizer /
    pre-tokenizer config is parsed separately by the
    ``BPETokenizer.from_pretrained`` entry point if present.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    model = data.get("model", {})
    vocab = {str(k): int(v) for k, v in model.get("vocab", {}).items()}
    merges_raw = model.get("merges", [])
    merges: list[tuple[str, str]] = []
    for entry in merges_raw:
        # HF formats: either "left right" string OR ["left", "right"] list
        # (the newer schema as of tokenizers 0.20+).
        if isinstance(entry, list) and len(entry) == 2:
            merges.append((str(entry[0]), str(entry[1])))
        elif isinstance(entry, str):
            idx = entry.rfind(" ")
            if idx <= 0:
                raise ValueError(
                    f"_load_unified_tokenizer_json: malformed merge " f"{entry!r} in {path}"
                )
            merges.append((entry[:idx], entry[idx + 1 :]))
        else:
            raise ValueError(
                f"_load_unified_tokenizer_json: unrecognised merge "
                f"entry type {type(entry).__name__} in {path}"
            )
    return vocab, merges


def _save_vocab_json(vocab: dict[str, int], path: str) -> None:
    """Write a vocab dict back to ``vocab.json`` in HF format."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def _save_merges_txt(merges: list[tuple[str, str]], path: str, version: str = "0.2") -> None:
    """Write a merge list to ``merges.txt`` with the HF header."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"#version: {version}\n")
        for left, right in merges:
            f.write(f"{left} {right}\n")


# ── Shared mixin: normalizer + pre-tokenizer chain ──────────────────


class _BPECommonMixin(Tokenizer):
    """Holds the normalizer + pre-tokenizer chain and persistence that both
    :class:`BPETokenizer` and :class:`BPETokenizerFast` use.

    Subclasses still implement their own ``_encode_one`` (because
    the algorithm itself lives in different places — Python loop vs
    C++ call) but the surface preprocessing and persistence are shared.
    """

    _vocab: dict[str, int]
    _merges: list[tuple[str, str]]
    _id_to_token: dict[int, str]
    _normalizer: Normalizer | None
    _pre_tokenizer: PreTokenizer

    def _prepare_chunks(self, text: str) -> list[str]:
        """Apply normalizer + pre-tokenizer, return chunk strings."""
        if self._normalizer is not None:
            text = self._normalizer(text)
        return [chunk for chunk, _ in self._pre_tokenizer(text)]

    @override
    @property
    def algo(self) -> str:
        """Algorithm identifier (``"bpe"``)."""
        return "bpe"

    @override
    def get_vocab(self) -> dict[str, int]:
        """Return a shallow copy of the token → id map."""
        return dict(self._vocab)

    @override
    def id_to_token(self, token_id: int) -> str | None:
        """Look up the surface string for a token id."""
        return self._id_to_token.get(token_id)

    @override
    def save(self, directory: str) -> None:
        """Persist as both legacy (``vocab.json`` + ``merges.txt``)
        AND unified (``tokenizer.json``) formats.
        """
        os.makedirs(directory, exist_ok=True)
        _save_vocab_json(self._vocab, os.path.join(directory, "vocab.json"))
        _save_merges_txt(self._merges, os.path.join(directory, "merges.txt"))
        super().save(directory)

    @override
    def _save_extras(self) -> dict[str, object]:
        """Add ``model`` block to the unified ``tokenizer.json``."""
        return {
            "model": {
                "type": "BPE",
                "vocab": self._vocab,
                "merges": [list(m) for m in self._merges],
            }
        }


def _load_bpe_checkpoint(
    directory: str,
    special_tokens: SpecialTokens | None,
    caller: str,
) -> tuple[
    dict[str, int],
    list[tuple[str, str]],
    SpecialTokens | None,
    dict[str, object],
]:
    """Load vocab, merges, special tokens, and optional model dict from ``directory``."""
    unified = os.path.join(directory, "tokenizer.json")
    model_data: dict[str, object] = {}
    if os.path.isfile(unified):
        vocab, merges = _load_unified_tokenizer_json(unified)
        with open(unified, encoding="utf-8") as f:
            data = json.load(f)
        raw_model = data.get("model", {})
        if isinstance(raw_model, dict):
            model_data = raw_model
    else:
        vj = os.path.join(directory, "vocab.json")
        mt = os.path.join(directory, "merges.txt")
        if not (os.path.isfile(vj) and os.path.isfile(mt)):
            raise FileNotFoundError(
                f"{caller}.from_file: neither tokenizer.json "
                f"nor (vocab.json + merges.txt) found in {directory}"
            )
        vocab = _load_vocab_json(vj)
        merges = _load_merges_txt(mt)
    st = special_tokens
    if st is None:
        sp_path = os.path.join(directory, "special_tokens_map.json")
        if os.path.isfile(sp_path):
            with open(sp_path, encoding="utf-8") as f:
                sp = json.load(f)
            st = _special_tokens_from_map(sp)
    return vocab, merges, st, model_data


def _load_bpe_from_file[T](
    cls: Callable[..., T],
    directory: str,
    *,
    normalizer: Normalizer | None = None,
    pre_tokenizer: PreTokenizer | None = None,
    special_tokens: SpecialTokens | None = None,
) -> T:
    """Instantiate a BPE tokenizer subclass from ``directory``."""
    vocab, merges, st, _ = _load_bpe_checkpoint(
        directory, special_tokens, getattr(cls, "__name__", "BPETokenizer")
    )
    return cls(
        vocab,
        merges,
        normalizer=normalizer,
        pre_tokenizer=pre_tokenizer,
        special_tokens=st,
    )


# ── Pure-Python reference impl ──────────────────────────────────────


class BPETokenizer(_BPECommonMixin, Tokenizer):
    r"""Reference (pure-Python) BPE tokenizer.

    Implements the classical Sennrich-2016 BPE encode loop directly
    in Python.  Easy to read, easy to step through with a debugger,
    works on every platform (no C++ build needed).

    For production / latency-sensitive use, prefer the matching
    :class:`BPETokenizerFast` — same algorithm, same vocab format,
    bit-identical encode outputs, but the hot loop is in C++.

    Parameters
    ----------
    vocab : dict[str, int]
        Token-string → id map.  Must include every single character
        that any merge could produce (so the algorithm has a valid
        starting symbol sequence).
    merges : list[tuple[str, str]]
        Ordered BPE merge pairs — index = rank (lower = higher
        priority, applied first during encoding).
    normalizer : Normalizer, optional
        Pre-encode text normalisation chain.  Default
        :class:`~lucid.utils.tokenizer._normalizers.NFC`.
    pre_tokenizer : PreTokenizer, optional
        Chunk-splitter that turns the normalised text into the
        sequence of strings the BPE algorithm processes one at a
        time.  Default
        :class:`~lucid.utils.tokenizer._pre_tokenizers.WhitespaceSplit`.
    special_tokens : SpecialTokens, optional
        Special-token registry — see
        :class:`lucid.utils.tokenizer.SpecialTokens`.

    Examples
    --------
    Trained in place from an empty vocabulary and no merges:

    >>> from lucid.utils.tokenizer import BPETokenizer
    >>> tok = BPETokenizer(vocab={}, merges=[])
    >>> tok.train(["the quick brown fox", "the lazy dog sleeps"],
    ...           vocab_size=80)
    >>> tok.decode(tok.encode("the quick dog"))
    'thequickdog'

    The whitespace is gone, and that is the algorithm rather than a
    defect: raw BPE has no notion of a word boundary. Use
    :class:`ByteLevelBPETokenizer` where the round trip has to be exact.
    """

    def __init__(
        self,
        vocab: dict[str, int],
        merges: list[tuple[str, str]],
        *,
        normalizer: Normalizer | None = None,
        pre_tokenizer: PreTokenizer | None = None,
        special_tokens: SpecialTokens | None = None,
    ) -> None:
        r"""Construct a pure-Python BPE tokenizer.

        Parameters
        ----------
        vocab : dict[str, int]
            Token-string → id map; must cover every starting symbol
            referenced by ``merges``.
        merges : list of (str, str)
            Ordered merge pairs; index = rank (lower wins).
        normalizer : Normalizer or None, optional, keyword-only
            Pre-encode normalisation chain.  Defaults to :class:`NFC`.
        pre_tokenizer : PreTokenizer or None, optional, keyword-only
            Chunk splitter.  Defaults to :class:`WhitespaceSplit`.
        special_tokens : SpecialTokens or None, optional, keyword-only
            Special-token registry.

        Notes
        -----
        Mutates internal state: copies ``vocab`` / ``merges``, builds
        the `_pair_to_merge` compiled lookup, and registers special
        tokens via the base :class:`Tokenizer` constructor.
        """
        self._vocab: dict[str, int] = dict(vocab)
        self._merges: list[tuple[str, str]] = list(merges)
        self._normalizer = normalizer if normalizer is not None else NFC()
        self._pre_tokenizer = pre_tokenizer if pre_tokenizer is not None else WhitespaceSplit()
        # Compiled merge table: ``(left_id, right_id) -> (merged_id, rank)``.
        self._pair_to_merge: dict[tuple[int, int], tuple[int, int]] = {}
        self._rebuild_tables()
        super().__init__(special_tokens=special_tokens)

    def _rebuild_tables(self) -> None:
        r"""Recompute the `_pair_to_merge` cache + reverse vocab.

        Called after construction and after :meth:`train`; subclasses
        adding mutating operations must call this themselves.

        Notes
        -----
        Mutates `_id_to_token` and `_pair_to_merge` in place.  Merges
        whose left, right, or merged-string ids are missing from the
        vocab are silently skipped (matches the reference framework).
        """
        self._id_to_token: dict[int, str] = {v: k for k, v in self._vocab.items()}
        self._pair_to_merge = {}
        for rank, (left, right) in enumerate(self._merges):
            la = self._vocab.get(left)
            rb = self._vocab.get(right)
            merged = self._vocab.get(left + right)
            if la is None or rb is None or merged is None:
                # Malformed merge — skip silently (matches HF behaviour).
                continue
            self._pair_to_merge[(la, rb)] = (merged, rank)

    # ── Tokenizer interface ────────────────────────────────────────

    @override
    @property
    def vocab_size(self) -> int:
        r"""Number of tokens currently registered in the vocab.

        Returns
        -------
        int
            ``len(self._vocab)`` — includes special tokens that were
            registered into the main vocab.
        """
        return len(self._vocab)

    @override
    def _encode_one(self, text: str) -> list[int]:
        r"""Algorithm-specific encode: chunk → ids via BPE merges."""
        all_ids: list[int] = []
        for chunk in self._prepare_chunks(text):
            all_ids.extend(self._encode_chunk(chunk))
        return all_ids

    @override
    def _decode_one(self, ids: list[int]) -> str:
        r"""Concatenate surface strings for ``ids``; unknown ids are
        silently dropped.
        """
        out: list[str] = []
        for i in ids:
            tok = self._id_to_token.get(i)
            if tok is not None:
                out.append(tok)
        return "".join(out)

    def _encode_chunk(self, chunk: str) -> list[int]:
        """Apply BPE merges to one pre-tokenized chunk.

        1. Split into per-codepoint ids via the vocab (drop unknowns
           or fall back to UNK if defined).
        2. Greedily apply the lowest-rank merge until no more apply.

        O(N · M) where N = chunk length and M = applicable merges.
        N is typically tiny (~5–15 codepoints) so the constant
        factor is dominated by Python's interpreter overhead;
        :class:`BPETokenizerFast` recovers an order of magnitude on
        long-document encoding.
        """
        ids: list[int] = []
        unk_id = self.unk_token_id
        for ch in chunk:
            tid = self._vocab.get(ch)
            if tid is not None:
                ids.append(tid)
            elif unk_id is not None:
                ids.append(unk_id)
        return self._merge_ids(ids)

    def _merge_ids(self, ids: list[int]) -> list[int]:
        """Greedily apply the lowest-rank merge until none applies.

        Split out from :meth:`_encode_chunk` so a subclass can change how
        the symbol sequence is *seeded* without restating the merge loop —
        GPT-1's end-of-word marker is exactly that case, and a second copy
        of this loop would be free to drift from the first.

        Args:
            ids: Seed symbol ids, in order.

        Returns:
            The merged id sequence.  Fewer than two symbols cannot merge,
            so they are returned unchanged.
        """
        if len(ids) < 2:
            return ids
        while True:
            best_rank = -1
            best_pos = -1
            best_merged = -1
            for k in range(len(ids) - 1):
                entry = self._pair_to_merge.get((ids[k], ids[k + 1]))
                if entry is None:
                    continue
                m_id, rank = entry
                if best_rank == -1 or rank < best_rank:
                    best_rank = rank
                    best_pos = k
                    best_merged = m_id
            if best_pos == -1:
                break
            ids[best_pos] = best_merged
            del ids[best_pos + 1]
        return ids

    # ── Training ────────────────────────────────────────────────────

    def train(
        self,
        corpus: Iterable[str],
        *,
        vocab_size: int = 30_000,
    ) -> None:
        """Re-train this tokenizer from scratch on ``corpus``.

        Implements the classical Sennrich-2016 training loop:

        1. Pre-tokenize each document (using the configured
           pre-tokenizer chain) into a frequency-counted set of
           words.
        2. Seed the vocab with every distinct codepoint appearing
           in any word.
        3. Iteratively count adjacent symbol-pair frequencies + merge
           the most frequent pair, growing the vocab by 1 each step
           until ``vocab_size`` is reached or no pair appears more
           than once.

        Replaces `_vocab` + `_merges` in place; previous contents
        are discarded.

        Parameters
        ----------
        corpus : iterable of str
            Each item is one document (or chunk thereof).  Generators
            are consumed exactly once.
        vocab_size : int, default 30 000
            Target total vocab size (chars + merges).  Stops early if
            no more merges occur.
        """
        # 1. Pre-tokenize + count.
        word_freq: dict[str, int] = {}
        for doc in corpus:
            for chunk in self._prepare_chunks(doc):
                word_freq[chunk] = word_freq.get(chunk, 0) + 1
        # 2. Seed vocab with characters.
        new_vocab: dict[str, int] = {}
        next_id = 0
        # Per-word symbol sequence + frequency (parallel arrays).
        words: list[tuple[list[str], int]] = []
        for w, freq in word_freq.items():
            syms = list(w)  # one entry per codepoint
            for s in syms:
                if s not in new_vocab:
                    new_vocab[s] = next_id
                    next_id += 1
            words.append((syms, freq))
        # 3. Iteratively merge most frequent pair.
        new_merges: list[tuple[str, str]] = []
        while len(new_vocab) < vocab_size:
            pair_freq: dict[tuple[str, str], int] = {}
            for syms, freq in words:
                for k in range(len(syms) - 1):
                    pair = (syms[k], syms[k + 1])
                    pair_freq[pair] = pair_freq.get(pair, 0) + freq
            if not pair_freq:
                break
            # Pick max-count, tie-break on lexicographic order for
            # determinism.
            best_pair = max(pair_freq.items(), key=lambda kv: (kv[1], kv[0]))
            if best_pair[1] < 2:
                break
            left, right = best_pair[0]
            merged = left + right
            new_merges.append((left, right))
            new_vocab[merged] = next_id
            next_id += 1
            # Apply merge across all word symbol sequences.
            new_words: list[tuple[list[str], int]] = []
            for syms, freq in words:
                out: list[str] = []
                k = 0
                while k < len(syms):
                    if k + 1 < len(syms) and syms[k] == left and syms[k + 1] == right:
                        out.append(merged)
                        k += 2
                    else:
                        out.append(syms[k])
                        k += 1
                new_words.append((out, freq))
            words = new_words
        # Commit + rebuild tables + re-resolve special-token ids.
        self._vocab = new_vocab
        self._merges = new_merges
        self._rebuild_tables()
        self._refresh_special_ids()

    @classmethod
    def from_file(
        cls,
        directory: str,
        *,
        normalizer: Normalizer | None = None,
        pre_tokenizer: PreTokenizer | None = None,
        special_tokens: SpecialTokens | None = None,
    ) -> BPETokenizer:
        """Load a pure-Python BPE tokenizer from ``directory``.

        Args:
            directory: Folder with ``tokenizer.json`` or ``vocab.json`` + ``merges.txt``.
            normalizer: Optional pre-encode normaliser override.
            pre_tokenizer: Optional pre-tokenizer splitter override.
            special_tokens: Optional special-token registry override.
        """
        return _load_bpe_from_file(
            cls,
            directory,
            normalizer=normalizer,
            pre_tokenizer=pre_tokenizer,
            special_tokens=special_tokens,
        )

    from_pretrained = from_file  # alias matching HF naming


def _special_tokens_from_map(sp: dict[str, object]) -> SpecialTokens:
    """Parse ``special_tokens_map.json`` into a :class:`SpecialTokens`.

    Recognises HF's standard keys (``pad_token`` / ``unk_token`` /
    ``bos_token`` / ``eos_token`` / ``cls_token`` / ``sep_token`` /
    ``mask_token``) — each can be either a plain string or
    ``{"content": "<tok>", ...}``-style dict.
    """

    def _stringify(v: object) -> str | None:
        if v is None:
            return None
        if isinstance(v, str):
            return v
        if isinstance(v, dict) and "content" in v:
            return str(v["content"])
        return None

    extra = {}
    add = sp.get("additional_special_tokens", []) or []
    if isinstance(add, list):
        for i, v in enumerate(add):
            s = _stringify(v)
            if s is not None:
                extra[f"additional_{i}"] = s
    return SpecialTokens(
        pad=_stringify(sp.get("pad_token")),
        unk=_stringify(sp.get("unk_token")),
        bos=_stringify(sp.get("bos_token")),
        eos=_stringify(sp.get("eos_token")),
        mask=_stringify(sp.get("mask_token")),
        sep=_stringify(sp.get("sep_token")),
        cls=_stringify(sp.get("cls_token")),
        extra=extra,
    )


# ── Fast C++-backed implementation ──────────────────────────────────


class BPETokenizerFast(_BPECommonMixin, Tokenizer):
    r"""C++-backed BPE tokenizer.

    Identical algorithm + vocab format to :class:`BPETokenizer`; the
    hot loop (per-chunk merge application) runs in C++ via the engine
    ``BPE`` binding.  Encode outputs are bit-identical for the same
    vocab + same merges + same normalizer + same pre-tokenizer.

    Use this in production training / inference.  Use
    :class:`BPETokenizer` for debugging / extending the algorithm
    with custom Python-only normalizers without touching C++.

    Parameters
    ----------
    Same as :class:`BPETokenizer`.  The C++ backend is constructed
    transparently in ``__init__`` and held as `_cpp`.

    Examples
    --------
    Given the same vocabulary and merges, this and
    :class:`BPETokenizer` encode identically — so one can be trained and
    the other used to run it:

    >>> from lucid.utils.tokenizer import BPETokenizer, BPETokenizerFast
    >>> slow = BPETokenizer(vocab={}, merges=[])
    >>> slow.train(["the quick brown fox", "the lazy dog sleeps"],
    ...            vocab_size=80)
    >>> fast = BPETokenizerFast(vocab=dict(slow._vocab),
    ...                         merges=list(slow._merges))
    >>> fast.encode("the quick dog") == slow.encode("the quick dog")
    True

    Trained separately they will not match, and that is the training
    rather than the encoder: two runs settle on different merge orders,
    both correct, and the ids follow the merges.
    """

    def __init__(
        self,
        vocab: dict[str, int],
        merges: list[tuple[str, str]],
        *,
        normalizer: Normalizer | None = None,
        pre_tokenizer: PreTokenizer | None = None,
        special_tokens: SpecialTokens | None = None,
    ) -> None:
        r"""Construct a C++-backed BPE tokenizer.

        Parameters
        ----------
        vocab : dict[str, int]
            Token-string → id map; copied into both Python and C++
            state.
        merges : list of (str, str)
            Ordered merge pairs; rank = list index.
        normalizer : Normalizer or None, optional, keyword-only
            Pre-encode normalisation chain.  Defaults to :class:`NFC`.
        pre_tokenizer : PreTokenizer or None, optional, keyword-only
            Chunk splitter.  Defaults to :class:`WhitespaceSplit`.
        special_tokens : SpecialTokens or None, optional, keyword-only
            Special-token registry; mirrored into the C++ side via
            :meth:`_sync_special_tokens_to_cpp`.

        Notes
        -----
        Builds the C++ ``BPE`` backend once and caches it on `_cpp`;
        the Python-side `_id_to_token` is kept in sync so decode
        avoids a binding round-trip.
        """
        self._vocab: dict[str, int] = dict(vocab)
        self._merges: list[tuple[str, str]] = list(merges)
        self._normalizer = normalizer if normalizer is not None else NFC()
        self._pre_tokenizer = pre_tokenizer if pre_tokenizer is not None else WhitespaceSplit()
        # Construct the C++ tokenizer from the same vocab + merges.
        self._cpp = _C_engine.utils.tokenizer.BPE(self._vocab, self._merges)
        # Reverse lookup table for decode (the C++ side also has one,
        # but Python-side access avoids a binding round-trip for the
        # decode hot path).
        self._id_to_token: dict[int, str] = {v: k for k, v in self._vocab.items()}
        super().__init__(special_tokens=special_tokens)
        self._sync_special_tokens_to_cpp()

    def _sync_special_tokens_to_cpp(self) -> None:
        r"""Mirror `_special_ids` into the C++ tokenizer's
        ``SpecialTokens`` registry so the C++ encode fall-back path
        (UNK substitution) has the same view of the world.
        """
        st_cpp = _C_engine.utils.tokenizer.SpecialTokens()
        for name in ("pad", "unk", "bos", "eos", "mask", "sep", "cls"):
            tid = self._special_ids.get(name)
            if tid is not None:
                setattr(st_cpp, name, tid)
        # The engine ``SpecialTokens.extra`` binding maps name → token id;
        # its generated stub annotates the value as str, so silence the
        # int-value mismatch here.
        st_cpp.extra = {
            k: v  # type: ignore[misc]
            for k, v in self._special_ids.items()
            if k not in ("pad", "unk", "bos", "eos", "mask", "sep", "cls")
        }
        self._cpp.special_tokens = st_cpp

    # ── Tokenizer interface ────────────────────────────────────────

    @override
    @property
    def vocab_size(self) -> int:
        r"""Number of tokens in the C++ side vocab.

        Returns
        -------
        int
            Result of ``self._cpp.vocab_size()`` — authoritative
            count for the live tokenizer (kept in sync with the
            Python cache after every :meth:`train`).
        """
        return self._cpp.vocab_size()

    @override
    def _encode_one(self, text: str) -> list[int]:
        r"""C++ encode path: normalize + pre-tokenize in Python,
        per-chunk merge in C++.
        """
        all_ids: list[int] = []
        for chunk in self._prepare_chunks(text):
            all_ids.extend(self._cpp.encode(chunk))
        return all_ids

    @override
    def _decode_one(self, ids: list[int]) -> str:
        r"""C++ decode for parity with the Python flavour.

        Returns the joined token-string surface form; special-token
        stripping is handled by the :class:`Tokenizer` base.
        """
        return self._cpp.decode(list(ids))

    # ── Training ────────────────────────────────────────────────────

    def train(
        self,
        corpus: Iterable[str],
        *,
        vocab_size: int = 30_000,
    ) -> None:
        """Re-train in C++.

        Materialises ``corpus`` into a list before handing off (the
        C++ binding takes ``std::vector<std::string>``); for very
        large corpora the user is responsible for chunking.  After
        training, the Python-side vocab / merges caches are refreshed
        from the C++ side so subsequent encodes see the new state.

        Parameters
        ----------
        corpus : iterable of str
            Documents to train on — each item is one document (or
            chunk thereof).  Generators are consumed exactly once
            and materialised into a list before crossing into C++.
        vocab_size : int, optional, keyword-only, default=30000
            Target total vocab size (characters + merges).  The C++
            trainer stops early if no symbol pair appears more than
            once.
        """
        corpus_list = list(corpus)
        self._cpp.train(corpus_list, vocab_size)
        self._vocab = dict(self._cpp.get_vocab())
        self._merges = [(a, b) for a, b in self._cpp.merges()]
        self._id_to_token = {v: k for k, v in self._vocab.items()}
        self._refresh_special_ids()
        self._sync_special_tokens_to_cpp()

    @classmethod
    def from_file(
        cls,
        directory: str,
        *,
        normalizer: Normalizer | None = None,
        pre_tokenizer: PreTokenizer | None = None,
        special_tokens: SpecialTokens | None = None,
    ) -> BPETokenizerFast:
        """Load a C++-backed BPE tokenizer from ``directory``.

        Args:
            directory: Checkpoint directory path on disk.
            normalizer: Custom text normaliser to apply before splitting.
            pre_tokenizer: Custom chunk splitter for the C++ merge loop.
            special_tokens: Custom special-token configuration.
        """
        return _load_bpe_from_file(
            cls,
            directory,
            normalizer=normalizer,
            pre_tokenizer=pre_tokenizer,
            special_tokens=special_tokens,
        )

    from_pretrained = from_file
