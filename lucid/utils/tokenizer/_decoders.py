"""Decoders — turn a sequence of token surfaces back into text.

Encoding rewrites the text before the algorithm sees it: SentencePiece
checkpoints turn spaces into ``▁``, byte-level ones map every byte onto a
printable character, Llama-style vocabularies spell unknown characters as
``<0x..>`` byte tokens.  Decoding has to undo exactly the rewrite that was
applied, and a Hugging Face ``tokenizer.json`` records which one that was
in its ``decoder`` block — so the block is read rather than assumed.

The interface mirrors Hugging Face's: :meth:`Decoder.decode_chain` maps a
list of token surfaces to a list of strings, which lets a
:class:`Sequence` hand one stage's output to the next, and
:meth:`Decoder.decode` joins the final list.  Several stages act per token
on purpose — :class:`Metaspace` treats the first token differently from
the rest — so collapsing to one string early would change the result.
"""

import re
from abc import ABC, abstractmethod
from typing import override

from lucid.utils.tokenizer._config import (
    as_list,
    as_object,
    get_bool,
    get_int,
    get_str,
    pattern_from_config,
    pattern_to_config,
    type_of,
)
from lucid.utils.tokenizer._pre_tokenizers import ByteLevel as _ByteLevelTable
from lucid.utils.tokenizer._pre_tokenizers import prepend_scheme_from_config


class Decoder(ABC):
    """Abstract base — every decoder implements :meth:`decode_chain`.

    See Also
    --------
    Metaspace, Replace, Strip, Fuse, ByteFallback, ByteLevel, Sequence
    """

    @abstractmethod
    def decode_chain(self, tokens: list[str]) -> list[str]:
        """Rewrite a list of token surfaces.

        Parameters
        ----------
        tokens : list of str
            Token surfaces in sequence order.

        Returns
        -------
        list of str
            The rewritten pieces; their concatenation is the text.
        """

    @abstractmethod
    def to_config(self) -> dict[str, object]:
        """Return the ``tokenizer.json`` block that rebuilds this decoder.

        Returns
        -------
        dict of str to object
            A block :func:`decoder_from_config` accepts.
        """

    def decode(self, tokens: list[str]) -> str:
        """Decode token surfaces to text.

        Parameters
        ----------
        tokens : list of str
            Token surfaces in sequence order.

        Returns
        -------
        str
            ``"".join(self.decode_chain(tokens))``.
        """
        return "".join(self.decode_chain(tokens))


class Metaspace(Decoder):
    r"""Undo the SentencePiece ``▁`` word marker.

    Every ``▁`` becomes a space, except in the first token when a prefix
    was prepended at encode time — there every ``▁`` is dropped, which
    removes the space the encoder invented before the first word.

    Parameters
    ----------
    replacement : str, default "▁"
        The marker character.
    prepend_scheme : {"always", "first", "never"}, default "always"
        The scheme the matching pre-tokenizer used.  Anything but
        ``"never"`` means the first token's marker is dropped.
    split : bool, default True
        Stored for serialisation; decoding does not depend on it.

    Examples
    --------
    >>> from lucid.utils.tokenizer._decoders import Metaspace
    >>> Metaspace().decode(["▁hello", "▁wor", "ld"])
    'hello world'
    """

    def __init__(
        self, replacement: str = "▁", prepend_scheme: str = "always", split: bool = True
    ) -> None:
        r"""Record the marker and the scheme.

        Parameters
        ----------
        replacement : str, default "▁"
            The marker character.
        prepend_scheme : {"always", "first", "never"}, default "always"
            The encode-side prepend scheme.
        split : bool, default True
            Stored for serialisation.
        """
        if len(replacement) != 1:
            raise ValueError(f"Metaspace: replacement must be one character, got {replacement!r}")
        if prepend_scheme not in ("always", "first", "never"):
            raise ValueError(f"Metaspace: unknown prepend_scheme {prepend_scheme!r}")
        self._replacement = replacement
        self._prepend_scheme = prepend_scheme
        self._split = split

    @override
    def decode_chain(self, tokens: list[str]) -> list[str]:
        """Map markers to spaces, dropping them in the first token.

        Parameters
        ----------
        tokens : list of str
            Token surfaces.

        Returns
        -------
        list of str
            One string per token.
        """
        out: list[str] = []
        strip_first = self._prepend_scheme != "never"
        for i, token in enumerate(tokens):
            if i == 0 and strip_first:
                out.append(token.replace(self._replacement, ""))
            else:
                out.append(token.replace(self._replacement, " "))
        return out

    @override
    def to_config(self) -> dict[str, object]:
        """Return the ``Metaspace`` decoder block.

        Returns
        -------
        dict of str to object
            The Hugging Face schema for this decoder.
        """
        return {
            "type": "Metaspace",
            "replacement": self._replacement,
            "prepend_scheme": self._prepend_scheme,
            "split": self._split,
        }


class Replace(Decoder):
    r"""Replace a literal or a pattern inside every token.

    Parameters
    ----------
    pattern : str or re.Pattern
        A literal substring, or a compiled expression.
    content : str
        The replacement, inserted verbatim.

    Examples
    --------
    >>> from lucid.utils.tokenizer._decoders import Replace
    >>> Replace("▁", " ").decode(["▁a", "▁b"])
    ' a b'
    """

    def __init__(self, pattern: str | re.Pattern[str], content: str) -> None:
        r"""Record the pattern and the replacement.

        Parameters
        ----------
        pattern : str or re.Pattern
            Literal or compiled expression.
        content : str
            Replacement text.
        """
        self._pattern = pattern
        self._content = content

    @override
    def decode_chain(self, tokens: list[str]) -> list[str]:
        """Apply the replacement to each token.

        Parameters
        ----------
        tokens : list of str
            Token surfaces.

        Returns
        -------
        list of str
            One string per token.
        """
        if isinstance(self._pattern, str):
            return [t.replace(self._pattern, self._content) for t in tokens]
        # A function replacement keeps ``content`` literal; a string one
        # would interpret any backslash in it as a group reference.
        content = self._content
        return [self._pattern.sub(lambda _m: content, t) for t in tokens]

    @override
    def to_config(self) -> dict[str, object]:
        """Return the ``Replace`` decoder block.

        Returns
        -------
        dict of str to object
            The Hugging Face schema for this decoder.
        """
        return {
            "type": "Replace",
            "pattern": pattern_to_config(self._pattern),
            "content": self._content,
        }


class Strip(Decoder):
    r"""Strip up to ``start`` leading and ``stop`` trailing ``content`` chars
    from every token.

    Parameters
    ----------
    content : str, default " "
        The character to strip.
    start : int, default 0
        At most this many are removed from each token's start.
    stop : int, default 0
        At most this many are removed from each token's end.

    Examples
    --------
    >>> from lucid.utils.tokenizer._decoders import Strip
    >>> Strip(" ", start=1).decode(["  hi"])
    ' hi'
    """

    def __init__(self, content: str = " ", start: int = 0, stop: int = 0) -> None:
        r"""Record what to strip and how much.

        Parameters
        ----------
        content : str, default " "
            The character to strip.
        start : int, default 0
            Maximum removed from each token's start.
        stop : int, default 0
            Maximum removed from each token's end.
        """
        if len(content) != 1:
            raise ValueError(f"Strip: content must be one character, got {content!r}")
        self._content = content
        self._start = start
        self._stop = stop

    @override
    def decode_chain(self, tokens: list[str]) -> list[str]:
        """Strip each token.

        Parameters
        ----------
        tokens : list of str
            Token surfaces.

        Returns
        -------
        list of str
            One string per token.
        """
        out: list[str] = []
        for token in tokens:
            begin = 0
            while begin < min(self._start, len(token)) and token[begin] == self._content:
                begin += 1
            end = len(token)
            removed = 0
            while removed < self._stop and end > begin and token[end - 1] == self._content:
                end -= 1
                removed += 1
            out.append(token[begin:end])
        return out

    @override
    def to_config(self) -> dict[str, object]:
        """Return the ``Strip`` decoder block.

        Returns
        -------
        dict of str to object
            The Hugging Face schema for this decoder.
        """
        return {"type": "Strip", "content": self._content, "start": self._start, "stop": self._stop}


class Fuse(Decoder):
    r"""Join every token into one.

    Matters only inside a :class:`Sequence`: a :class:`Strip` after it acts
    on the whole text instead of on each token.

    Examples
    --------
    >>> from lucid.utils.tokenizer._decoders import Fuse
    >>> Fuse().decode_chain(["a", "b"])
    ['ab']
    """

    @override
    def decode_chain(self, tokens: list[str]) -> list[str]:
        """Concatenate all tokens.

        Parameters
        ----------
        tokens : list of str
            Token surfaces.

        Returns
        -------
        list of str
            A one-element list.
        """
        return ["".join(tokens)]

    @override
    def to_config(self) -> dict[str, object]:
        """Return the ``Fuse`` decoder block.

        Returns
        -------
        dict of str to object
            ``{"type": "Fuse"}``.
        """
        return {"type": "Fuse"}


class ByteFallback(Decoder):
    r"""Reassemble ``<0x..>`` byte tokens into characters.

    Consecutive byte tokens are gathered and decoded as UTF-8 together, so
    a character spelled over three byte tokens comes back whole.  A run
    that is not valid UTF-8 becomes one U+FFFD per byte, as in Hugging
    Face.

    Examples
    --------
    >>> from lucid.utils.tokenizer._decoders import ByteFallback
    >>> ByteFallback().decode(["a", "<0xC3>", "<0xA9>"])
    'aé'
    """

    @override
    def decode_chain(self, tokens: list[str]) -> list[str]:
        """Replace runs of byte tokens with the text they spell.

        Parameters
        ----------
        tokens : list of str
            Token surfaces.

        Returns
        -------
        list of str
            Tokens with each byte run collapsed.
        """
        out: list[str] = []
        pending = bytearray()

        def flush() -> None:
            if not pending:
                return
            try:
                out.append(bytes(pending).decode("utf-8"))
            except UnicodeDecodeError:
                out.extend("�" for _ in pending)
            pending.clear()

        for token in tokens:
            value = _byte_token_value(token)
            if value is not None:
                pending.append(value)
                continue
            flush()
            out.append(token)
        flush()
        return out

    @override
    def to_config(self) -> dict[str, object]:
        """Return the ``ByteFallback`` decoder block.

        Returns
        -------
        dict of str to object
            ``{"type": "ByteFallback"}``.
        """
        return {"type": "ByteFallback"}


def _byte_token_value(token: str) -> int | None:
    """Return the byte a ``<0xHH>`` token stands for, or ``None``."""
    if len(token) == 6 and token.startswith("<0x") and token.endswith(">"):
        digits = token[3:5]
        if all(c in "0123456789abcdefABCDEF" for c in digits):
            return int(digits, 16)
    return None


class ByteLevel(Decoder):
    r"""Undo the GPT-2 byte-to-character mapping.

    The tokens are joined, mapped back to the bytes they stand for, and
    decoded as UTF-8 with invalid sequences replaced.

    Examples
    --------
    >>> from lucid.utils.tokenizer._decoders import ByteLevel
    >>> ByteLevel().decode(["Hello", "Ġworld"])
    'Hello world'
    """

    @override
    def decode_chain(self, tokens: list[str]) -> list[str]:
        """Map the joined tokens back through the byte table.

        Parameters
        ----------
        tokens : list of str
            Byte-mapped token surfaces.

        Returns
        -------
        list of str
            A one-element list with the decoded text.
        """
        raw = bytearray()
        decoder = _ByteLevelTable._byte_decoder or _ensure_byte_tables()
        for token in tokens:
            mapped = [decoder.get(ch) for ch in token]
            if all(b is not None for b in mapped):
                raw.extend(b for b in mapped if b is not None)
            else:
                # A token with any character outside the table did not come
                # from the byte mapping — an added token, typically — so its
                # own UTF-8 is kept whole, as Hugging Face does.
                raw.extend(token.encode("utf-8"))
        return [raw.decode("utf-8", errors="replace")]

    @override
    def to_config(self) -> dict[str, object]:
        """Return the ``ByteLevel`` decoder block.

        Returns
        -------
        dict of str to object
            ``{"type": "ByteLevel"}``.
        """
        return {"type": "ByteLevel"}


def _ensure_byte_tables() -> dict[str, int]:
    """Build the shared byte tables if nothing has yet, and return one."""
    _ByteLevelTable._build_byte_tables_()
    return _ByteLevelTable._byte_decoder


class Sequence(Decoder):
    r"""Apply several decoders in order.

    Parameters
    ----------
    decoders : list of Decoder
        Applied left to right, each to the previous one's output.

    Examples
    --------
    The Llama-style chain: markers to spaces, bytes reassembled, the
    tokens fused, one leading space dropped.

    >>> from lucid.utils.tokenizer._decoders import (
    ...     ByteFallback, Fuse, Replace, Sequence, Strip,
    ... )
    >>> chain = Sequence(
    ...     [Replace("▁", " "), ByteFallback(), Fuse(), Strip(" ", start=1)]
    ... )
    >>> chain.decode(["▁caf", "<0xC3>", "<0xA9>"])
    'café'
    """

    def __init__(self, decoders: list[Decoder]) -> None:
        r"""Snapshot the decoders.

        Parameters
        ----------
        decoders : list of Decoder
            Applied left to right.
        """
        self._decoders = list(decoders)

    @override
    def decode_chain(self, tokens: list[str]) -> list[str]:
        """Thread the tokens through every decoder.

        Parameters
        ----------
        tokens : list of str
            Token surfaces.

        Returns
        -------
        list of str
            The last decoder's output.
        """
        out = list(tokens)
        for decoder in self._decoders:
            out = decoder.decode_chain(out)
        return out

    @override
    def to_config(self) -> dict[str, object]:
        """Return the ``Sequence`` decoder block.

        Returns
        -------
        dict of str to object
            ``{"type": "Sequence", "decoders": [...]}``.
        """
        return {"type": "Sequence", "decoders": [d.to_config() for d in self._decoders]}


def decoder_from_config(config: object) -> Decoder | None:
    """Build a decoder from a ``tokenizer.json`` ``decoder`` block.

    Parameters
    ----------
    config : dict or None
        The block, or ``None`` for the JSON ``null``.

    Returns
    -------
    Decoder or None
        The decoder, or ``None`` when the block is ``null``.

    Raises
    ------
    ValueError
        For an unsupported ``type`` or a malformed block.

    Examples
    --------
    >>> from lucid.utils.tokenizer._decoders import decoder_from_config
    >>> t5 = decoder_from_config(
    ...     {"type": "Metaspace", "replacement": "▁", "add_prefix_space": True}
    ... )
    >>> t5.decode(["▁hello", "▁world"])
    'hello world'
    """
    if config is None:
        return None
    where = "decoder"
    cfg = as_object(config, where)
    kind = type_of(cfg, where)
    if kind == "Metaspace":
        return Metaspace(
            replacement=get_str(cfg, "replacement", where, "▁"),
            prepend_scheme=prepend_scheme_from_config(cfg, where),
            split=get_bool(cfg, "split", where, True),
        )
    if kind == "Replace":
        return Replace(
            pattern_from_config(cfg.get("pattern"), f"{where}.pattern"),
            get_str(cfg, "content", where, ""),
        )
    if kind == "Strip":
        return Strip(
            content=get_str(cfg, "content", where, " "),
            start=get_int(cfg, "start", where, 0),
            stop=get_int(cfg, "stop", where, 0),
        )
    if kind == "Fuse":
        return Fuse()
    if kind == "ByteFallback":
        return ByteFallback()
    if kind == "ByteLevel":
        return ByteLevel()
    if kind == "Sequence":
        decoders: list[Decoder] = []
        for raw in as_list(cfg.get("decoders"), f"{where}.decoders"):
            inner = decoder_from_config(raw)
            if inner is not None:
                decoders.append(inner)
        return Sequence(decoders)
    raise ValueError(
        f"{where}: unsupported type {kind!r}; supported are Metaspace, Replace, "
        f"Strip, Fuse, ByteFallback, ByteLevel and Sequence"
    )
