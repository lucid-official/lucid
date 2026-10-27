"""ByteTokenizer — fixed 256-vocab byte-level tokenizer.

Every UTF-8 byte maps to its own id (id == byte value).  The vocab
is fixed at 256 so :meth:`train` is a no-op and the OOV concept
doesn't apply — every input string is, by definition, a sequence of
bytes the tokenizer can encode.

Why use it?
-----------
* **Robust multilingual baseline** — any text in any encoding
  round-trips byte-perfectly without an UNK fallback.
* **No training needed** — the vocab is universal.
* **Used by ByT5** + various byte-level LM research.

Trade-off: sequences are ~4× longer than a subword tokenizer would
produce for the same text, so attention costs more.  Use only when
you don't need subword compression (small inputs, multilingual
robustness more important than throughput).
"""

import json
import os
from typing import Iterable, override

from lucid._C import engine as _C_engine

from lucid.utils.tokenizer._base import SpecialTokens, Tokenizer


def _build_byte_vocab() -> dict[str, int]:
    """The canonical 256-entry vocab — byte ``b`` ↔ Latin-1 char
    ``chr(b)``.  Matches the C++ ByteTokenizer's internal table."""
    return {chr(b): b for b in range(256)}


# ── Pure-Python reference impl ──────────────────────────────────────


class ByteTokenizer(Tokenizer):
    """Reference (pure-Python) byte-level tokenizer.

    Vocab is fixed at 256 (byte value → id == byte value).  Equivalent
    to :class:`ByteTokenizerFast` but loops in Python — useful when
    debugging downstream consumers without a C++ dependency.

    Parameters
    ----------
    special_tokens : SpecialTokens, optional
        Special-token registry; the canonical 7 slots can hold
        printable byte values (``chr(b)`` for some ``b``).  IDs
        beyond 255 are NOT supported on the default vocab — extend
        the vocab manually before adding such specials.

    Examples
    --------
    Nothing to train — every byte is already a token, so the round trip
    is exact for any input at all:

    >>> from lucid.utils.tokenizer import ByteTokenizer
    >>> tok = ByteTokenizer()
    >>> tok.encode("the quick dog")[:4]
    [116, 104, 101, 32]
    >>> tok.decode(tok.encode("héllo — 안녕"))
    'héllo — 안녕'
    """

    def __init__(self, special_tokens: SpecialTokens | None = None) -> None:
        r"""Construct a pure-Python byte tokenizer.

        Parameters
        ----------
        special_tokens : SpecialTokens or None, optional
            Optional special-token registry.  The byte vocab itself
            is universal (256 entries) and is built unconditionally.
        """
        self._vocab = _build_byte_vocab()
        self._id_to_token = {v: k for k, v in self._vocab.items()}
        super().__init__(special_tokens=special_tokens)

    @override
    @property
    def vocab_size(self) -> int:
        """Fixed vocab size = 256 (one id per possible byte value)."""
        return 256

    @override
    @property
    def algo(self) -> str:
        """Algorithm identifier (always ``"byte"``)."""
        return "byte"

    @override
    def get_vocab(self) -> dict[str, int]:
        """Return the canonical 256-entry byte → id map."""
        return dict(self._vocab)

    @override
    def id_to_token(self, token_id: int) -> str | None:
        """Inverse of the byte vocab; returns the Latin-1 char for
        valid byte ids, ``None`` otherwise."""
        return self._id_to_token.get(token_id)

    @override
    def _encode_one(self, text: str) -> list[int]:
        """Encode by UTF-8 byte expansion (id = byte value)."""
        return list(text.encode("utf-8"))

    @override
    def _decode_one(self, ids: list[int]) -> str:
        """Decode by re-assembling bytes; malformed UTF-8 is replaced."""
        # Filter to valid byte range, then UTF-8 decode (replace
        # malformed sub-sequences so the round-trip never raises).
        bs = bytes(i for i in ids if 0 <= i < 256)
        return bs.decode("utf-8", errors="replace")

    def train(self, corpus: Iterable[str], *, vocab_size: int = 256) -> None:
        """No-op — the byte vocab is fixed at 256.

        Parameters
        ----------
        corpus : iterable of str
            Ignored; accepted only for API uniformity with the rest
            of the tokenizer family.
        vocab_size : int, default 256
            Ignored; the vocab is always 256.
        """
        # Intentionally silent; the contract is "byte vocab is fixed".

    @override
    def save(self, directory: str) -> None:
        """Persist as ``tokenizer.json`` only (no per-algorithm
        legacy file — there's no vocab.txt convention for byte
        tokenizers).

        Parameters
        ----------
        directory : str
            Output directory (created if missing).  Produces
            ``tokenizer.json`` + ``special_tokens_map.json``.
        """
        super().save(directory)

    @override
    def _save_extras(self) -> dict[str, object]:
        """Emit the unified-format ``model`` block for byte tokenizers."""
        return {"model": {"type": "Byte"}}

    @classmethod
    def from_file(
        cls, directory: str, *, special_tokens: SpecialTokens | None = None
    ) -> ByteTokenizer:
        """Load from a directory containing ``special_tokens_map.json``.

        The vocab itself is not read from disk (always the canonical
        256-entry table); we only consult ``special_tokens_map.json``
        so user-defined specials survive a round-trip.

        Parameters
        ----------
        directory : str
            Directory previously written by :meth:`save`.
        special_tokens : SpecialTokens, optional
            Override for the on-disk special tokens.

        Returns
        -------
        ByteTokenizer
            New instance with the loaded special-token registry.
        """
        st = special_tokens
        if st is None:
            sp_path = os.path.join(directory, "special_tokens_map.json")
            if os.path.isfile(sp_path):
                with open(sp_path, encoding="utf-8") as f:
                    sp = json.load(f)
                from lucid.utils.tokenizer._bpe import _special_tokens_from_map

                st = _special_tokens_from_map(sp)
        return cls(special_tokens=st)

    from_pretrained = from_file


# ── Fast C++-backed implementation ──────────────────────────────────


class ByteTokenizerFast(Tokenizer):
    """C++-backed byte tokenizer; bit-identical to :class:`ByteTokenizer`.

    Same semantics as :class:`ByteTokenizer`, hot loop in C++.
    Speedup vs the Python flavour is modest (the algorithm is already
    a trivial UTF-8 encode loop) but the API uniformity matters: any
    code that swaps ``XxxTokenizer`` ↔ ``XxxTokenizerFast`` works
    consistently across the family.

    Parameters
    ----------
    special_tokens : SpecialTokens, optional
        Special-token registry; see :class:`ByteTokenizer` for the
        constraints on default-vocab compatibility.

    Examples
    --------
    >>> from lucid.utils.tokenizer import ByteTokenizerFast
    >>> tok = ByteTokenizerFast()
    >>> tok.decode(tok.encode("the quick dog"))
    'the quick dog'
    """

    def __init__(self, special_tokens: SpecialTokens | None = None) -> None:
        r"""Construct a C++-backed byte tokenizer.

        Parameters
        ----------
        special_tokens : SpecialTokens or None, optional
            Optional special-token registry.

        Notes
        -----
        Instantiates the C++ ``ByteTokenizer`` backend; the canonical
        256-entry Python vocab is also cached for ``get_vocab`` /
        ``id_to_token`` parity.
        """
        self._cpp = _C_engine.utils.tokenizer.ByteTokenizer()
        self._vocab = _build_byte_vocab()
        self._id_to_token = {v: k for k, v in self._vocab.items()}
        super().__init__(special_tokens=special_tokens)

    @override
    @property
    def vocab_size(self) -> int:
        """Fixed vocab size (256), as reported by the C++ backend."""
        return self._cpp.vocab_size()

    @override
    @property
    def algo(self) -> str:
        """Algorithm identifier (always ``"byte"``)."""
        return "byte"

    @override
    def get_vocab(self) -> dict[str, int]:
        """Return the canonical 256-entry byte → id map."""
        return dict(self._vocab)

    @override
    def id_to_token(self, token_id: int) -> str | None:
        """Return the Latin-1 char for ``token_id`` or ``None``."""
        return self._id_to_token.get(token_id)

    @override
    def _encode_one(self, text: str) -> list[int]:
        """UTF-8 byte expansion (Python ``str.encode`` is the hot path)."""
        # The C++ ByteTokenizer.encode iterates byte-by-byte; that's
        # marginally slower than ``text.encode('utf-8')`` here, so
        # use Python's built-in for the hot path.  Output is bit-
        # identical to the C++ version (same UTF-8 byte sequence).
        return list(text.encode("utf-8"))

    @override
    def _decode_one(self, ids: list[int]) -> str:
        """Inverse byte expansion; malformed UTF-8 replaced."""
        bs = bytes(i for i in ids if 0 <= i < 256)
        return bs.decode("utf-8", errors="replace")

    def train(self, corpus: Iterable[str], *, vocab_size: int = 256) -> None:
        """No-op — the byte vocab is fixed at 256.  See
        :meth:`ByteTokenizer.train`."""

    @override
    def save(self, directory: str) -> None:
        """Persist as ``tokenizer.json`` + ``special_tokens_map.json``.

        Parameters
        ----------
        directory : str
            Output directory (created if missing).
        """
        super().save(directory)

    @override
    def _save_extras(self) -> dict[str, object]:
        """Emit the unified-format ``model`` block for byte tokenizers."""
        return {"model": {"type": "Byte"}}

    @classmethod
    def from_file(
        cls, directory: str, *, special_tokens: SpecialTokens | None = None
    ) -> ByteTokenizerFast:
        """Load from ``directory`` (only the special-token map is read).

        Parameters
        ----------
        directory : str
            Directory previously written by :meth:`save`.
        special_tokens : SpecialTokens, optional
            Override for the on-disk special tokens.

        Returns
        -------
        ByteTokenizerFast
            New instance with the loaded special-token registry.
        """
        st = special_tokens
        if st is None:
            sp_path = os.path.join(directory, "special_tokens_map.json")
            if os.path.isfile(sp_path):
                with open(sp_path, encoding="utf-8") as f:
                    sp = json.load(f)
                from lucid.utils.tokenizer._bpe import _special_tokens_from_map

                st = _special_tokens_from_map(sp)
        return cls(special_tokens=st)

    from_pretrained = from_file
