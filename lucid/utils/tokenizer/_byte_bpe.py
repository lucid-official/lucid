"""ByteLevelBPETokenizer — Byte-Pair Encoding over byte-mapped text.

The GPT-2 / RoBERTa / BART tokenizer.  Two key differences from
classical :class:`BPETokenizer`:

1. **Pre-tokenizer** is :class:`ByteLevel` — every UTF-8 byte gets
   re-mapped to a printable Unicode codepoint via the GPT-2
   byte-to-unicode table (bytes 0x00-0x20 + 0x7F-0xA0 get shifted
   into a "safe" Unicode block; everything else maps to itself).
   This means the BPE algorithm never sees raw bytes — it operates
   over a printable Unicode alphabet of size 256.  Bonus: no UNK
   needed, because every byte sequence can be encoded.

2. **Decode** must reverse the byte-mapping.  Joining token surface
   forms gives back the byte-mapped string; passing that through
   :meth:`ByteLevel.decode_bytes` recovers the original UTF-8 bytes,
   which we then UTF-8-decode back to ``str``.  Without this step
   the decoded output looks like ``"Ġworld"`` instead of ``" world"``.

Inherits from :class:`BPETokenizer` / :class:`BPETokenizerFast` —
the merge table + training loop are exactly the same.  Per the user
request, no new C++ class — just a subclass that swaps in the right
pre-tokenizer + post-decode.
"""

from typing import Callable, Iterable, Self, override

from lucid.utils.tokenizer._base import SpecialTokens
from lucid.utils.tokenizer._bpe import (
    BPETokenizer,
    BPETokenizerFast,
    _BPECommonMixin,
    _load_bpe_checkpoint,
)
from lucid.utils.tokenizer._normalizers import NFC, Normalizer
from lucid.utils.tokenizer._pre_tokenizers import ByteLevel


class _ByteLevelDecodeMixin(_BPECommonMixin):
    """Shared decode and persistence override for both BBPE flavours.

    Joins token surface forms, runs them through
    :meth:`ByteLevel.decode_bytes` to undo the GPT-2 byte-to-unicode
    mapping, then UTF-8-decodes back to ``str``.  Without this
    reverse-mapping step the decoded output would look like
    ``"Ġworld"`` instead of ``" world"``.
    """

    _add_prefix_space: bool

    @override
    def _decode_one(self, ids: list[int]) -> str:
        """Reverse the byte-mapping and UTF-8 decode joined tokens."""
        # 1. Join the byte-mapped tokens.
        text = "".join(self._id_to_token[i] for i in ids if i in self._id_to_token)
        # 2. Reverse the byte-to-unicode mapping → raw UTF-8 bytes.
        try:
            raw = ByteLevel.decode_bytes(text)
        except KeyError:
            # Decoded id mapped to a token that isn't in the GPT-2
            # byte alphabet (most likely a user-added special).  Fall
            # back to the raw joined text (the special's surface
            # form is the right rendering anyway).
            return text
        # 3. UTF-8 decode (errors=replace so we never raise).
        return raw.decode("utf-8", errors="replace")

    @override
    @property
    def algo(self) -> str:
        """Algorithm identifier (always ``"byte_bpe"``)."""
        return "byte_bpe"

    @override
    def _save_extras(self) -> dict[str, object]:
        """Emit the ByteLevelBPE ``model`` block for ``tokenizer.json``."""
        return {
            "model": {
                "type": "ByteLevelBPE",
                "vocab": self._vocab,
                "merges": [list(m) for m in self._merges],
                "add_prefix_space": self._add_prefix_space,
            }
        }


def _load_byte_bpe_from_file[T](
    cls: Callable[..., T],
    directory: str,
    *,
    add_prefix_space: bool = False,
    normalizer: Normalizer | None = None,
    special_tokens: SpecialTokens | None = None,
) -> T:
    """Shared checkpoint loader for :class:`ByteLevelBPETokenizer` and :class:`ByteLevelBPETokenizerFast`."""
    vocab, merges, st, model_data = _load_bpe_checkpoint(
        directory, special_tokens, getattr(cls, "__name__", "ByteLevelBPETokenizer")
    )
    resolved_prefix = bool(model_data.get("add_prefix_space", add_prefix_space))
    return cls(
        vocab, merges, add_prefix_space=resolved_prefix, normalizer=normalizer, special_tokens=st
    )


# ── Pure-Python BBPE ────────────────────────────────────────────────


class ByteLevelBPETokenizer(_ByteLevelDecodeMixin, BPETokenizer):
    r"""Pure-Python byte-level BPE (GPT-2 / RoBERTa convention).

    Subclasses :class:`BPETokenizer` and swaps in a
    :class:`~lucid.utils.tokenizer._pre_tokenizers.ByteLevel`
    pre-tokenizer so the underlying BPE algorithm sees a printable
    Unicode alphabet of size 256 instead of raw bytes.  The GPT-2
    byte-to-unicode mapping shifts the unprintable bytes
    (``0x00``–``0x20``, ``0x7F``–``0xA0``) into a safe Unicode block
    so every byte sequence is representable and **no UNK token is
    needed**.

    The merge table + training loop are inherited unchanged; the
    only differences are the pre-tokenizer (encode side) and the
    byte-mapping reversal in :meth:`~ByteLevelBPETokenizer.decode`
    (decode side).

    For production / latency-sensitive use, prefer
    :class:`ByteLevelBPETokenizerFast`.

    Parameters
    ----------
    vocab : dict[str, int]
        Token-string → id map.  Tokens are the GPT-2 byte-mapped
        Unicode strings (e.g. ``"Ġworld"`` for `` world``), **not**
        raw bytes.
    merges : list[tuple[str, str]]
        Ordered BPE merge pairs — index = rank (lower = higher
        priority).  Same format as plain BPE.
    add_prefix_space : bool, default False
        Whether to prepend a space to the input before
        pre-tokenizing (so the first word also gets a ``Ġ`` prefix).
        GPT-2 default is ``False``; RoBERTa default is ``True``.
    normalizer : Normalizer, optional
        Pre-encode normalisation chain.  Default
        :class:`~lucid.utils.tokenizer._normalizers.NFC` (matches
        GPT-2).
    special_tokens : SpecialTokens, optional
        Special-token registry — see
        :class:`lucid.utils.tokenizer.SpecialTokens`.

    Notes
    -----
    Any published Hugging Face GPT-2 / RoBERTa / BART / distilGPT
    checkpoint loads without modification via
    :meth:`~ByteLevelBPETokenizer.from_pretrained`.

    See Also
    --------
    ByteLevelBPETokenizerFast : C++-backed equivalent with the same API.
    BPETokenizer : Classical (non byte-level) BPE base.

    Examples
    --------
    >>> from lucid.utils.tokenizer import ByteLevelBPETokenizer
    >>> tok = ByteLevelBPETokenizer(vocab={}, merges=[])
    >>> tok.train(["the quick brown fox", "the lazy dog sleeps"],
    ...           vocab_size=300)
    >>> tok.decode(tok.encode("the quick dog"))
    'the quick dog'

    Exact, unlike raw :class:`BPETokenizer` — the byte-level alphabet
    keeps the spaces that plain BPE drops.
    """

    def __init__(
        self,
        vocab: dict[str, int],
        merges: list[tuple[str, str]],
        *,
        add_prefix_space: bool = False,
        normalizer: Normalizer | None = None,
        special_tokens: SpecialTokens | None = None,
    ) -> None:
        r"""Construct a pure-Python byte-level BPE tokenizer.

        Parameters
        ----------
        vocab : dict[str, int]
            GPT-2 byte-mapped token → id map.
        merges : list of (str, str)
            BPE merge pairs over the byte-mapped alphabet.
        add_prefix_space : bool, default False
            Prepend a space before pre-tokenizing (GPT-2: ``False``,
            RoBERTa: ``True``).
        normalizer : Normalizer or None, optional, keyword-only
            Pre-encode normaliser.  Defaults to :class:`NFC`.
        special_tokens : SpecialTokens or None, optional, keyword-only
            Special-token registry.

        Notes
        -----
        The pre-tokenizer is fixed to :class:`ByteLevel` with the
        chosen ``add_prefix_space`` — that's the whole point of
        byte-level BPE.  Forwards everything else to
        :meth:`BPETokenizer.__init__`.
        """
        self._add_prefix_space = add_prefix_space
        super().__init__(
            vocab,
            merges,
            normalizer=normalizer if normalizer is not None else NFC(),
            pre_tokenizer=ByteLevel(add_prefix_space=add_prefix_space),
            special_tokens=special_tokens,
        )

    @override
    def train(self, corpus: Iterable[str], *, vocab_size: int = 50_000) -> None:
        """Re-train this tokenizer from scratch on ``corpus``.

        Delegates to the parent :meth:`BPETokenizer.train`.  The
        configured :class:`ByteLevel` pre-tokenizer means the
        algorithm sees byte-mapped chunks throughout training, so
        the resulting vocab + merges are in GPT-2 byte-mapped form
        (compatible with Hugging Face GPT-2 / RoBERTa export).

        Parameters
        ----------
        corpus : iterable of str
            Each item is one document.  Generators are consumed
            exactly once.
        vocab_size : int, default 50 000
            Target total vocab size.  GPT-2's published vocab is
            50 257; RoBERTa's is 50 265.
        """
        super().train(corpus, vocab_size=vocab_size)

    @override
    @classmethod
    def from_file(  # type: ignore[override]
        cls,
        directory: str,
        *,
        add_prefix_space: bool = False,
        normalizer: Normalizer | None = None,
        special_tokens: SpecialTokens | None = None,
    ) -> Self:
        """Load from a Hugging Face-compatible directory."""
        return _load_byte_bpe_from_file(
            cls,
            directory,
            add_prefix_space=add_prefix_space,
            normalizer=normalizer,
            special_tokens=special_tokens,
        )

    from_pretrained = from_file  # type: ignore[assignment]


# ── Fast (C++-backed) BBPE ──────────────────────────────────────────


class ByteLevelBPETokenizerFast(_ByteLevelDecodeMixin, BPETokenizerFast):
    r"""C++-backed byte-level BPE tokenizer.

    Identical configuration semantics to
    :class:`ByteLevelBPETokenizer`; uses the engine ``BPE`` binding
    for the merge hot loop and the same :class:`ByteLevel`
    pre-tokenizer in Python.
    Encode outputs are bit-identical to the pure-Python flavour for
    the same vocab + same merges.

    Parameters
    ----------
    Same as :class:`ByteLevelBPETokenizer`.

    See Also
    --------
    ByteLevelBPETokenizer : Pure-Python reference; same vocab format.

    Examples
    --------
    >>> from lucid.utils.tokenizer import ByteLevelBPETokenizerFast
    >>> tok = ByteLevelBPETokenizerFast(vocab={}, merges=[])
    >>> tok.train(["the quick brown fox", "the lazy dog sleeps"],
    ...           vocab_size=300)
    >>> tok.decode(tok.encode("the quick dog"))
    'the quick dog'
    """

    def __init__(
        self,
        vocab: dict[str, int],
        merges: list[tuple[str, str]],
        *,
        add_prefix_space: bool = False,
        normalizer: Normalizer | None = None,
        special_tokens: SpecialTokens | None = None,
    ) -> None:
        r"""Construct a C++-backed byte-level BPE tokenizer.

        Parameters
        ----------
        vocab : dict[str, int]
            GPT-2 byte-mapped token → id map.
        merges : list of (str, str)
            BPE merge pairs over the byte-mapped alphabet.
        add_prefix_space : bool, default False
            Prepend a space before pre-tokenizing (GPT-2 default
            ``False``; RoBERTa default ``True``).
        normalizer : Normalizer or None, optional, keyword-only
            Pre-encode normaliser.  Defaults to :class:`NFC`.
        special_tokens : SpecialTokens or None, optional, keyword-only
            Special-token registry.

        Notes
        -----
        Fixes the pre-tokenizer to :class:`ByteLevel` and forwards to
        :meth:`BPETokenizerFast.__init__` — which constructs the C++
        ``BPE`` backend and caches it on `_cpp`.
        """
        self._add_prefix_space = add_prefix_space
        super().__init__(
            vocab,
            merges,
            normalizer=normalizer if normalizer is not None else NFC(),
            pre_tokenizer=ByteLevel(add_prefix_space=add_prefix_space),
            special_tokens=special_tokens,
        )

    @override
    def train(self, corpus: Iterable[str], *, vocab_size: int = 50_000) -> None:
        """Re-train in C++ via the parent
        :meth:`BPETokenizerFast.train`.

        The ByteLevel pre-tokenizer normalizes every byte to its
        printable form **before** the corpus reaches the C++
        training loop, so the resulting vocab is in byte-mapped form
        (Hugging Face GPT-2 compatible).

        Note: the C++ ``BPE::train`` does its OWN whitespace split
        on the raw input, so we pre-encode the corpus through the
        ByteLevel pre-tokenizer here in Python before handing off —
        otherwise the training would learn merges on raw bytes
        rather than byte-mapped codepoints.

        Parameters
        ----------
        corpus : iterable of str
            Each item is one document.
        vocab_size : int, default 50 000
            Target total vocab size.
        """
        encoded_corpus: list[str] = []
        for doc in corpus:
            if self._normalizer is not None:
                doc = self._normalizer(doc)
            chunks = [chunk for chunk, _ in self._pre_tokenizer(doc)]
            encoded_corpus.append(" ".join(chunks))
        self._cpp.train(encoded_corpus, vocab_size)
        self._vocab = dict(self._cpp.get_vocab())
        self._merges = [(a, b) for a, b in self._cpp.merges()]
        self._id_to_token = {v: k for k, v in self._vocab.items()}
        self._refresh_special_ids()
        self._sync_special_tokens_to_cpp()

    @override
    @classmethod
    def from_file(  # type: ignore[override]
        cls,
        directory: str,
        *,
        add_prefix_space: bool = False,
        normalizer: Normalizer | None = None,
        special_tokens: SpecialTokens | None = None,
    ) -> Self:
        """Load from a Hugging Face-compatible directory."""
        return _load_byte_bpe_from_file(
            cls,
            directory,
            add_prefix_space=add_prefix_space,
            normalizer=normalizer,
            special_tokens=special_tokens,
        )

    from_pretrained = from_file  # type: ignore[assignment]
