"""Post-processors — frame an encoded sequence with its special tokens.

A post-processor runs after the algorithm has turned text into ids and
decides which special tokens surround them: ``[CLS] … [SEP]`` for BERT,
``… </s>`` for T5, nothing at all for GPT-2.  The choice belongs to the
checkpoint rather than the algorithm — two Unigram vocabularies can frame
their sequences differently — which is why a Hugging Face
``tokenizer.json`` carries it as a ``post_processor`` block of its own,
and why it is read from there instead of being guessed from which special
tokens a ``special_tokens_map.json`` happens to name.  Guessing is how a
T5-style vocabulary whose map lists a ``bos_token`` came out with a BOS
the model never saw in training.

Ids are stored rather than looked up.  The block names each special token
together with the id it emits, and a published tokenizer is reproduced as
published even where its vocabulary would resolve the surface differently.

Every processor works on a list of *encodings* — one id list per input
sequence — and returns the list of segments to concatenate.  Working on
segments instead of one flat list is what lets :class:`Sequence` chain
processors the way Hugging Face does: a :class:`ByteLevel` step passes a
pair through untouched and the :class:`TemplateProcessing` after it still
sees two sequences.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import override

from lucid.utils.tokenizer._config import (
    as_list,
    as_object,
    get_bool,
    get_int,
    get_str,
    type_of,
)


class PostProcessor(ABC):
    """Abstract base — every post-processor implements
    :meth:`process_encodings` and :meth:`to_config`.

    See Also
    --------
    TemplateProcessing, BertProcessing, RobertaProcessing, ByteLevel, Sequence
    """

    @abstractmethod
    def process_encodings(self, encodings: list[list[int]]) -> list[list[int]]:
        """Frame one or two id sequences.

        Parameters
        ----------
        encodings : list of list of int
            One entry for a single sequence, two for a pair.

        Returns
        -------
        list of list of int
            The segments of the framed result, in order.  Their
            concatenation is what :meth:`process` returns.
        """

    @abstractmethod
    def to_config(self) -> dict[str, object]:
        """Return the ``tokenizer.json`` block that rebuilds this processor.

        Returns
        -------
        dict of str to object
            A block :func:`post_processor_from_config` accepts.
        """

    def process(self, ids: list[int], pair_ids: list[int] | None = None) -> list[int]:
        """Frame ``ids`` (and ``pair_ids``) into one flat id list.

        Parameters
        ----------
        ids : list of int
            The first sequence, without special tokens.
        pair_ids : list of int, optional
            The second sequence of a pair.

        Returns
        -------
        list of int
            The framed sequence.
        """
        encodings = [list(ids)] if pair_ids is None else [list(ids), list(pair_ids)]
        return [i for segment in self.process_encodings(encodings) for i in segment]

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        """Count the special tokens framing adds.

        Parameters
        ----------
        pair : bool, default False
            Count for a pair of sequences instead of a single one.

        Returns
        -------
        int
            Length of the framed empty sequence (or pair).
        """
        return len(self.process([], [] if pair else None))


@dataclass(frozen=True, slots=True)
class _Piece:
    """One slot of a template: a sequence (``"A"``/``"B"``) or a special."""

    kind: str  # "A", "B" or "special"
    name: str  # the special token's surface; empty for sequences
    type_id: int


def _parse_piece(text: str) -> _Piece:
    """Parse one whitespace-free template item such as ``$B:1``.

    The grammar is the one Hugging Face accepts: ``$`` alone or ``$A`` is
    the first sequence, ``$B`` the second, ``$<n>`` the first sequence with
    type id ``n``; anything else names a special token; a ``:<n>`` suffix
    sets the type id.
    """
    type_id = 0
    body = text
    head, sep, tail = text.rpartition(":")
    if sep and tail.isdigit() and head:
        body, type_id = head, int(tail)
    if body.startswith("$"):
        ident = body[1:]
        if ident in ("", "A", "a"):
            return _Piece("A", "", type_id)
        if ident in ("B", "b"):
            return _Piece("B", "", type_id)
        if ident.isdigit():
            return _Piece("A", "", int(ident))
        raise ValueError(f"TemplateProcessing: unknown sequence reference {text!r}")
    return _Piece("special", body, type_id)


def _parse_template(template: str | list[str]) -> list[_Piece]:
    """Parse a template given as one string or a list of items."""
    items = template.split() if isinstance(template, str) else list(template)
    return [_parse_piece(item) for item in items]


def _check_template(pieces: list[_Piece], sequences: tuple[str, ...], which: str) -> None:
    """Require each sequence in ``sequences`` exactly once and no other."""
    for kind in ("A", "B"):
        count = sum(1 for p in pieces if p.kind == kind)
        expected = 1 if kind in sequences else 0
        if count != expected:
            raise ValueError(
                f"TemplateProcessing: the {which} template must reference "
                f"${kind} {expected} time(s), found {count}"
            )


class TemplateProcessing(PostProcessor):
    r"""Frame sequences by a template of sequence slots and special tokens.

    Parameters
    ----------
    single : str or list of str
        Template for one sequence, e.g. ``"[CLS] $A [SEP]"`` or
        ``["$A", "</s>"]``.  ``$A`` (or ``$``) is the sequence; any other
        item names a special token; a ``:<n>`` suffix records a type id.
    pair : str or list of str, optional
        Template for a pair, referencing ``$A`` and ``$B`` once each.
        Defaults to ``"$A $B:1"`` — the two sequences with no special
        tokens — as Hugging Face does.
    special_tokens : dict of str to int or list of int, optional
        The id(s) each named special token emits.  Every special token the
        templates name must be present.

    Raises
    ------
    ValueError
        If a template misuses ``$A`` / ``$B`` or names a special token that
        ``special_tokens`` does not define.

    Examples
    --------
    T5 appends ``</s>`` and prepends nothing:

    >>> from lucid.utils.tokenizer._post_processors import TemplateProcessing
    >>> t5 = TemplateProcessing(
    ...     single="$A </s>", pair="$A </s> $B </s>", special_tokens={"</s>": 1}
    ... )
    >>> t5.process([7102])
    [7102, 1]
    >>> t5.process([3, 9], [3, 115])
    [3, 9, 1, 3, 115, 1]
    >>> t5.num_special_tokens_to_add()
    1
    """

    def __init__(
        self,
        single: str | list[str],
        pair: str | list[str] | None = None,
        special_tokens: dict[str, int | list[int]] | None = None,
    ) -> None:
        r"""Parse and validate the templates.

        Parameters
        ----------
        single : str or list of str
            Template for one sequence.
        pair : str or list of str, optional
            Template for a pair.
        special_tokens : dict of str to int or list of int, optional
            The id(s) each named special token emits.
        """
        specials: dict[str, list[int]] = {}
        for name, ids in (special_tokens or {}).items():
            specials[name] = [ids] if isinstance(ids, int) else list(ids)
        single_pieces = _parse_template(single)
        pair_pieces = None if pair is None else _parse_template(pair)
        self._init_pieces(single_pieces, pair_pieces, specials)

    def _init_pieces(
        self,
        single: list[_Piece],
        pair: list[_Piece] | None,
        specials: dict[str, list[int]],
    ) -> None:
        """Validate parsed templates and store them."""
        if pair is None:
            # Hugging Face's default when no pair template is given, checked
            # against its serialised form: the two sequences back to back,
            # the second with type id 1, and no special tokens at all.
            pair = [_Piece("A", "", 0), _Piece("B", "", 1)]
        _check_template(single, ("A",), "single")
        _check_template(pair, ("A", "B"), "pair")
        for piece in single + pair:
            if piece.kind == "special" and piece.name not in specials:
                raise ValueError(
                    f"TemplateProcessing: special token {piece.name!r} is used "
                    f"in a template but has no id in special_tokens"
                )
        self._single = single
        self._pair = pair
        self._specials = specials

    @classmethod
    def _from_pieces(
        cls,
        single: list[_Piece],
        pair: list[_Piece] | None,
        specials: dict[str, list[int]],
    ) -> TemplateProcessing:
        """Build from already-parsed pieces (the ``tokenizer.json`` path)."""
        obj = cls.__new__(cls)
        obj._init_pieces(single, pair, specials)
        return obj

    @override
    def process_encodings(self, encodings: list[list[int]]) -> list[list[int]]:
        """Substitute the encodings into the matching template.

        Parameters
        ----------
        encodings : list of list of int
            One or two sequences.

        Returns
        -------
        list of list of int
            One segment per template item.

        Raises
        ------
        ValueError
            If given anything other than one or two sequences.
        """
        if len(encodings) == 1:
            template = self._single
        elif len(encodings) == 2:
            template = self._pair
        else:
            raise ValueError(f"TemplateProcessing: expected 1 or 2 sequences, got {len(encodings)}")
        out: list[list[int]] = []
        for piece in template:
            if piece.kind == "A":
                out.append(list(encodings[0]))
            elif piece.kind == "B":
                out.append(list(encodings[1]))
            else:
                out.append(list(self._specials[piece.name]))
        return out

    @override
    def to_config(self) -> dict[str, object]:
        """Return the ``TemplateProcessing`` block for ``tokenizer.json``.

        Returns
        -------
        dict of str to object
            The Hugging Face schema: ``single`` / ``pair`` piece lists and
            the ``special_tokens`` id map.
        """

        def dump(pieces: list[_Piece]) -> list[object]:
            out: list[object] = []
            for p in pieces:
                if p.kind == "special":
                    out.append({"SpecialToken": {"id": p.name, "type_id": p.type_id}})
                else:
                    out.append({"Sequence": {"id": p.kind, "type_id": p.type_id}})
            return out

        return {
            "type": "TemplateProcessing",
            "single": dump(self._single),
            "pair": dump(self._pair),
            "special_tokens": {
                name: {"id": name, "ids": list(ids), "tokens": [name] * len(ids)}
                for name, ids in self._specials.items()
            },
        }


class BertProcessing(PostProcessor):
    r"""BERT's framing: ``[CLS] A [SEP]`` and ``[CLS] A [SEP] B [SEP]``.

    Parameters
    ----------
    sep : tuple of (str, int)
        The separator's surface and id.
    cls : tuple of (str, int)
        The classification token's surface and id.

    Examples
    --------
    >>> from lucid.utils.tokenizer._post_processors import BertProcessing
    >>> bert = BertProcessing(sep=("[SEP]", 102), cls=("[CLS]", 101))
    >>> bert.process([7592])
    [101, 7592, 102]
    >>> bert.process([7592], [2088])
    [101, 7592, 102, 2088, 102]
    """

    def __init__(self, sep: tuple[str, int], cls: tuple[str, int]) -> None:
        r"""Record the two framing tokens.

        Parameters
        ----------
        sep : tuple of (str, int)
            The separator's surface and id.
        cls : tuple of (str, int)
            The classification token's surface and id.
        """
        self._sep = (str(sep[0]), int(sep[1]))
        self._cls = (str(cls[0]), int(cls[1]))

    @override
    def process_encodings(self, encodings: list[list[int]]) -> list[list[int]]:
        """Wrap the first sequence in CLS/SEP and close the second with SEP.

        Parameters
        ----------
        encodings : list of list of int
            One or two sequences.

        Returns
        -------
        list of list of int
            The framed segments.
        """
        cls_id, sep_id = self._cls[1], self._sep[1]
        out = [[cls_id, *encodings[0], sep_id]]
        if len(encodings) > 1:
            out.append([*encodings[1], sep_id])
        return out

    @override
    def to_config(self) -> dict[str, object]:
        """Return the ``BertProcessing`` block for ``tokenizer.json``.

        Returns
        -------
        dict of str to object
            ``{"type": "BertProcessing", "sep": [s, id], "cls": [c, id]}``.
        """
        return {
            "type": "BertProcessing",
            "sep": list(self._sep),
            "cls": list(self._cls),
        }


class RobertaProcessing(PostProcessor):
    r"""RoBERTa's framing: ``<s> A </s>`` and ``<s> A </s></s> B </s>``.

    Parameters
    ----------
    sep : tuple of (str, int)
        The separator's surface and id.
    cls : tuple of (str, int)
        The start token's surface and id.
    trim_offsets : bool, default True
        Kept for ``tokenizer.json`` round trips; affects offsets only.
    add_prefix_space : bool, default True
        Kept for ``tokenizer.json`` round trips; affects offsets only.

    Examples
    --------
    >>> from lucid.utils.tokenizer._post_processors import RobertaProcessing
    >>> roberta = RobertaProcessing(sep=("</s>", 2), cls=("<s>", 0))
    >>> roberta.process([10], [20])
    [0, 10, 2, 2, 20, 2]
    """

    def __init__(
        self,
        sep: tuple[str, int],
        cls: tuple[str, int],
        trim_offsets: bool = True,
        add_prefix_space: bool = True,
    ) -> None:
        r"""Record the framing tokens and the offset flags.

        Parameters
        ----------
        sep : tuple of (str, int)
            The separator's surface and id.
        cls : tuple of (str, int)
            The start token's surface and id.
        trim_offsets : bool, default True
            Offsets-only flag, stored for serialisation.
        add_prefix_space : bool, default True
            Offsets-only flag, stored for serialisation.
        """
        self._sep = (str(sep[0]), int(sep[1]))
        self._cls = (str(cls[0]), int(cls[1]))
        self._trim_offsets = trim_offsets
        self._add_prefix_space = add_prefix_space

    @override
    def process_encodings(self, encodings: list[list[int]]) -> list[list[int]]:
        """Frame as RoBERTa does, doubling the separator between a pair.

        Parameters
        ----------
        encodings : list of list of int
            One or two sequences.

        Returns
        -------
        list of list of int
            The framed segments.
        """
        cls_id, sep_id = self._cls[1], self._sep[1]
        out = [[cls_id, *encodings[0], sep_id]]
        if len(encodings) > 1:
            out.append([sep_id, *encodings[1], sep_id])
        return out

    @override
    def to_config(self) -> dict[str, object]:
        """Return the ``RobertaProcessing`` block for ``tokenizer.json``.

        Returns
        -------
        dict of str to object
            The Hugging Face schema for this processor.
        """
        return {
            "type": "RobertaProcessing",
            "sep": list(self._sep),
            "cls": list(self._cls),
            "trim_offsets": self._trim_offsets,
            "add_prefix_space": self._add_prefix_space,
        }


class ByteLevel(PostProcessor):
    r"""GPT-2's post-processor — adds no special tokens.

    Hugging Face uses this stage only to trim whitespace out of token
    *offsets*; the ids pass through unchanged, which is exactly why a
    GPT-2 encode carries no ``<|endoftext|>``.

    Parameters
    ----------
    add_prefix_space : bool, default True
        Offsets-only flag, stored for serialisation.
    trim_offsets : bool, default True
        Offsets-only flag, stored for serialisation.
    use_regex : bool, default True
        Offsets-only flag, stored for serialisation.

    Examples
    --------
    >>> from lucid.utils.tokenizer._post_processors import ByteLevel
    >>> ByteLevel().process([464, 3290])
    [464, 3290]
    """

    def __init__(
        self,
        add_prefix_space: bool = True,
        trim_offsets: bool = True,
        use_regex: bool = True,
    ) -> None:
        r"""Record the offset flags.

        Parameters
        ----------
        add_prefix_space : bool, default True
            Stored for serialisation.
        trim_offsets : bool, default True
            Stored for serialisation.
        use_regex : bool, default True
            Stored for serialisation.
        """
        self._add_prefix_space = add_prefix_space
        self._trim_offsets = trim_offsets
        self._use_regex = use_regex

    @override
    def process_encodings(self, encodings: list[list[int]]) -> list[list[int]]:
        """Return the encodings unchanged.

        Parameters
        ----------
        encodings : list of list of int
            One or two sequences.

        Returns
        -------
        list of list of int
            Copies of the inputs.
        """
        return [list(e) for e in encodings]

    @override
    def to_config(self) -> dict[str, object]:
        """Return the ``ByteLevel`` block for ``tokenizer.json``.

        Returns
        -------
        dict of str to object
            The Hugging Face schema for this processor.
        """
        return {
            "type": "ByteLevel",
            "add_prefix_space": self._add_prefix_space,
            "trim_offsets": self._trim_offsets,
            "use_regex": self._use_regex,
        }


class Sequence(PostProcessor):
    r"""Apply several post-processors in order.

    An empty sequence adds nothing, which is also what a ``tokenizer.json``
    with ``"post_processor": null`` means.

    Parameters
    ----------
    processors : list of PostProcessor
        Applied left to right, each to the previous one's segments.

    Examples
    --------
    >>> from lucid.utils.tokenizer._post_processors import Sequence
    >>> Sequence([]).process([5, 6])
    [5, 6]
    """

    def __init__(self, processors: list[PostProcessor]) -> None:
        r"""Snapshot the processors.

        Parameters
        ----------
        processors : list of PostProcessor
            Applied left to right.
        """
        self._processors = list(processors)

    @override
    def process_encodings(self, encodings: list[list[int]]) -> list[list[int]]:
        """Thread the segments through every processor.

        Parameters
        ----------
        encodings : list of list of int
            One or two sequences.

        Returns
        -------
        list of list of int
            The last processor's segments.
        """
        out = [list(e) for e in encodings]
        for processor in self._processors:
            out = processor.process_encodings(out)
        return out

    @override
    def to_config(self) -> dict[str, object]:
        """Return the ``Sequence`` block for ``tokenizer.json``.

        Returns
        -------
        dict of str to object
            ``{"type": "Sequence", "processors": [...]}``.
        """
        return {
            "type": "Sequence",
            "processors": [p.to_config() for p in self._processors],
        }


def _token_and_id(value: object, where: str) -> tuple[str, int]:
    """Decode a ``["[SEP]", 102]`` pair."""
    items = as_list(value, where)
    if (
        len(items) != 2
        or not isinstance(items[0], str)
        or not isinstance(items[1], int)
        or isinstance(items[1], bool)
    ):
        raise ValueError(f"{where}: expected [token, id], got {value!r}")
    return items[0], items[1]


def _pieces_from_config(value: object, where: str) -> list[_Piece]:
    """Decode a template's list of ``Sequence`` / ``SpecialToken`` objects."""
    pieces: list[_Piece] = []
    for raw in as_list(value, where):
        entry = as_object(raw, where)
        if "Sequence" in entry:
            seq = as_object(entry["Sequence"], where)
            ident = get_str(seq, "id", where, "A")
            if ident not in ("A", "B"):
                raise ValueError(f"{where}: unknown sequence id {ident!r}")
            pieces.append(_Piece(ident, "", get_int(seq, "type_id", where, 0)))
        elif "SpecialToken" in entry:
            special = as_object(entry["SpecialToken"], where)
            name = get_str(special, "id", where, "")
            pieces.append(_Piece("special", name, get_int(special, "type_id", where, 0)))
        else:
            raise ValueError(f"{where}: unknown template piece {entry!r}")
    return pieces


def post_processor_from_config(config: object) -> PostProcessor:
    """Build a post-processor from a ``tokenizer.json`` ``post_processor`` block.

    Parameters
    ----------
    config : dict or None
        The block.  ``None`` — the JSON ``null`` a tokenizer without a
        post-processor serialises — builds one that adds nothing.

    Returns
    -------
    PostProcessor
        The processor the block describes.

    Raises
    ------
    ValueError
        For an unsupported ``type`` or a malformed block.

    Examples
    --------
    >>> from lucid.utils.tokenizer._post_processors import (
    ...     post_processor_from_config,
    ... )
    >>> block = {
    ...     "type": "TemplateProcessing",
    ...     "single": [
    ...         {"Sequence": {"id": "A", "type_id": 0}},
    ...         {"SpecialToken": {"id": "</s>", "type_id": 0}},
    ...     ],
    ...     "pair": [
    ...         {"Sequence": {"id": "A", "type_id": 0}},
    ...         {"SpecialToken": {"id": "</s>", "type_id": 0}},
    ...         {"Sequence": {"id": "B", "type_id": 0}},
    ...         {"SpecialToken": {"id": "</s>", "type_id": 0}},
    ...     ],
    ...     "special_tokens": {"</s>": {"id": "</s>", "ids": [1], "tokens": ["</s>"]}},
    ... }
    >>> post_processor_from_config(block).process([2948])
    [2948, 1]
    >>> post_processor_from_config(None).process([2948])
    [2948]
    """
    if config is None:
        return Sequence([])
    where = "post_processor"
    cfg = as_object(config, where)
    kind = type_of(cfg, where)
    if kind == "TemplateProcessing":
        single = _pieces_from_config(cfg.get("single"), f"{where}.single")
        pair = (
            None
            if cfg.get("pair") is None
            else _pieces_from_config(cfg.get("pair"), f"{where}.pair")
        )
        specials: dict[str, list[int]] = {}
        raw_specials = cfg.get("special_tokens")
        for name, raw in as_object(raw_specials or {}, where).items():
            entry = as_object(raw, f"{where}.special_tokens")
            ids = as_list(entry.get("ids"), f"{where}.special_tokens[{name!r}].ids")
            clean: list[int] = []
            for i in ids:
                if not isinstance(i, int) or isinstance(i, bool):
                    raise TypeError(f"{where}: special token id {i!r} is not an int")
                clean.append(i)
            specials[get_str(entry, "id", where, name)] = clean
        return TemplateProcessing._from_pieces(single, pair, specials)
    if kind == "BertProcessing":
        return BertProcessing(
            sep=_token_and_id(cfg.get("sep"), f"{where}.sep"),
            cls=_token_and_id(cfg.get("cls"), f"{where}.cls"),
        )
    if kind == "RobertaProcessing":
        return RobertaProcessing(
            sep=_token_and_id(cfg.get("sep"), f"{where}.sep"),
            cls=_token_and_id(cfg.get("cls"), f"{where}.cls"),
            trim_offsets=get_bool(cfg, "trim_offsets", where, True),
            add_prefix_space=get_bool(cfg, "add_prefix_space", where, True),
        )
    if kind == "ByteLevel":
        return ByteLevel(
            add_prefix_space=get_bool(cfg, "add_prefix_space", where, True),
            trim_offsets=get_bool(cfg, "trim_offsets", where, True),
            use_regex=get_bool(cfg, "use_regex", where, True),
        )
    if kind == "Sequence":
        return Sequence(
            [
                post_processor_from_config(p)
                for p in as_list(cfg.get("processors"), f"{where}.processors")
            ]
        )
    raise ValueError(
        f"{where}: unsupported type {kind!r}; supported are TemplateProcessing, "
        f"BertProcessing, RobertaProcessing, ByteLevel and Sequence"
    )
