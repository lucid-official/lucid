"""Base :class:`Tokenizer` ABC for the lucid.utils.tokenizer family.

Every concrete tokenizer — pure-Python (``XxxTokenizer``) and
C++-backed (``XxxTokenizerFast``) — derives from :class:`Tokenizer`
and implements the four required hooks (``encode_one`` /
``decode_one`` / ``vocab_size`` / ``algo``).  The base class layers
the universal user-facing API on top:

* :meth:`encode` / :meth:`decode` — single-string convenience that
  manage special tokens + apply the normalizer / pre-tokenizer chain.
* :meth:`encode_batch` / :meth:`decode_batch` — batch versions.
* :meth:`__call__` — Hugging Face-style entry point with padding,
  truncation, ``return_tensors='lucid'``, and attention mask
  computation.
* :meth:`save` / :meth:`from_file` / :meth:`from_pretrained` —
  vocab + config persistence using the HF-compatible
  ``vocab.json`` + ``merges.txt`` / unified ``tokenizer.json``
  formats.

The pure-Python tokenizers (slow but easy to debug + easy to train new
algorithms in) and the Fast (C++) tokenizers cohabit the same file
(``_bpe.py``, ``_wordpiece.py``, ...) and share the same vocabs +
formats, so swapping ``BPETokenizer`` ↔ ``BPETokenizerFast`` is a
one-line code change with bit-identical encode outputs.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, ClassVar, Iterable, Literal, override

from lucid.utils.tokenizer._post_processors import (
    PostProcessor,
    Sequence,
    TemplateProcessing,
    _Piece,
)

# Canonical special-token slot names.  Every algorithm honours this
# scheme so attention-mask / special-token-mask computation can stay
# uniform across BPE / WordPiece / Unigram.
_CANONICAL_SPECIAL_SLOTS = ("pad", "unk", "bos", "eos", "mask", "sep", "cls")


@dataclass(slots=True)
class SpecialTokens:
    """Canonical special-token registry.

    Each slot is either ``None`` (the algorithm doesn't define that
    token — e.g. GPT-2 has no ``pad`` by default) or the token's
    string surface form.  The ``extra`` map holds any non-canonical
    specials (e.g. ``<|endoftext|>`` for GPT-2, ``<|im_start|>``
    for ChatML) keyed by name.

    String values are surface forms (what appears in text); the
    corresponding ids are looked up in the tokenizer's vocab at
    construction time.

    Examples
    --------
    Naming a token here is half of it — the other half is that the
    token has to be in the vocabulary, and ``train`` builds the
    vocabulary from the corpus without reserving anything. Seed it:

    >>> from lucid.utils.tokenizer import SpecialTokens, WordTokenizer
    >>> marks = SpecialTokens(unk="<unk>", pad="<pad>")
    >>> tok = WordTokenizer(vocab={"<unk>": 0}, special_tokens=marks)
    >>> tok.unk_token_id
    0
    >>> tok.encode("a word it has never seen")
    [0, 0, 0, 0, 0, 0]

    Without the seed ``unk_token_id`` is ``None`` and an out-of-vocabulary
    word raises rather than being replaced, which is the safer of the two
    since a silent unknown is indistinguishable from a known token.
    """

    pad: str | None = None
    unk: str | None = None
    bos: str | None = None
    eos: str | None = None
    mask: str | None = None
    sep: str | None = None
    cls: str | None = None
    extra: dict[str, str] = field(default_factory=dict)

    def all_tokens(self) -> list[str]:
        """Return every special-token surface form (canonical + extra)."""
        out: list[str] = []
        for name in _CANONICAL_SPECIAL_SLOTS:
            v = getattr(self, name)
            if v is not None:
                out.append(v)
        out.extend(self.extra.values())
        return out


def registry_key(name: str) -> str:
    """A tokenizer name as the registry keys it: case, ``-``, ``_`` and spaces do not tell names apart."""
    return name.lower().replace("-", "").replace("_", "").replace(" ", "")


class Tokenizer(ABC):
    r"""Abstract base for every tokenizer in ``lucid.utils.tokenizer``.

    Subclasses implement four hooks:

    * :meth:`_encode_one(text) -> list[int]` — algorithm-specific
      encode for one pre-normalized + pre-tokenized chunk.  The base
      class handles normalization, pre-tokenization, and special-token
      insertion around this hook.
    * :meth:`_decode_one(ids) -> str` — algorithm-specific decode of
      one id sequence (special tokens NOT yet stripped — the base
      class handles ``skip_special_tokens``).
    * :attr:`vocab_size` — total number of distinct ids.
    * :attr:`algo` — short algorithm name for serialisation
      (``"bpe"``, ``"wordpiece"``, ``"unigram"``, ``"byte_bpe"``).

    The base class layers
    :meth:`encode` / :meth:`decode` / :meth:`encode_batch` /
    :meth:`decode_batch` / :meth:`__call__` on top — those are
    fully generic and don't need to be overridden.

    See Also
    --------
    :class:`BPETokenizer` : reference Python BPE implementation.
    :class:`BPETokenizerFast` : C++-backed wrapper around the same
        algorithm with identical encode outputs.

    Examples
    --------
    Every tokenizer here is one of these, so the three calls below are
    the whole surface a caller needs — build or train, encode, decode:

    >>> from lucid.utils.tokenizer import WordTokenizer
    >>> tok = WordTokenizer()
    >>> tok.train(["the quick brown fox", "the lazy dog"], vocab_size=32)
    >>> tok.decode(tok.encode("the quick dog"))
    'the quick dog'

    Subclassing one means supplying the algorithm and inheriting the
    rest — the special-token handling, ``save`` / ``from_pretrained``,
    and the batch helpers.
    """

    #: The tokenizer classes by registry key; a class statement fills it and
    #: :func:`~lucid.utils.tokenizer._registry.resolve_tokenizer` reads it.
    _named: ClassVar[dict[str, type[Tokenizer]]] = {}

    @override
    def __init_subclass__(cls, *, names: tuple[str, ...] = (), **kwargs: object) -> None:
        """Register the class under its own name and every alias in ``names``.

        A name beginning with ``_`` is private and not registered; a name taken
        before now means this class.
        """
        super().__init_subclass__(**kwargs)
        for name in (cls.__name__, *names):
            if not name.startswith("_"):
                Tokenizer._named[registry_key(name)] = cls

    # Subclasses populate at construction time.
    _special: SpecialTokens
    _special_ids: dict[str, int]

    #: An explicitly configured post-processor — read from a
    #: ``tokenizer.json`` or assigned through :attr:`post_processor`.  When
    #: ``None`` the class's :attr:`_special_layout` frames the sequence.
    #: A class attribute rather than one set in ``__init__`` so a subclass
    #: may assign it before calling ``super().__init__``.
    _post_processor: PostProcessor | None = None

    #: How ``encode(add_special_tokens=True)`` frames a sequence when no
    #: post-processor is configured, using whatever the special-token
    #: registry resolves at that moment: ``"bos_eos"`` wraps in BOS/EOS,
    #: ``"cls_sep"`` gives BERT's ``[CLS] … [SEP]``, ``"none"`` adds
    #: nothing (GPT-2).  Resolved per call rather than frozen at
    #: construction, so framing follows the ids the vocabulary resolves now
    #: — after ``train`` as well as before it.
    _special_layout: ClassVar[Literal["bos_eos", "cls_sep", "none"]] = "bos_eos"

    def __init__(self, special_tokens: SpecialTokens | None = None) -> None:
        """Initialise the special-token registry; subclasses must
        populate the vocab + algorithm state before calling this.
        """
        self._special = special_tokens or SpecialTokens()
        self._special_ids = {}
        self._refresh_special_ids()

    # ── Required overrides ──────────────────────────────────────────

    @abstractmethod
    def _encode_one(self, text: str) -> list[int]:
        """Algorithm-specific encode for a single text chunk.

        The base class has already applied the normalizer +
        pre-tokenizer chain; ``text`` here is a single
        algorithm-ready chunk (e.g. one whitespace-split word).
        Returns the raw id sequence WITHOUT special tokens
        (the base class injects BOS/EOS/CLS/SEP as needed).
        """

    @abstractmethod
    def _decode_one(self, ids: list[int]) -> str:
        """Algorithm-specific decode for a single id sequence.

        Special tokens are NOT stripped here — the base
        :meth:`decode` handles ``skip_special_tokens``.
        """

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Total number of distinct token ids (including specials)."""

    @property
    @abstractmethod
    def algo(self) -> str:
        """Short algorithm name (``"bpe"`` / ``"wordpiece"`` / ...).

        Used by :meth:`save` to route serialisation to the right
        format and by :meth:`from_pretrained` to pick the right
        subclass when loading from a generic path.
        """

    # ── Vocab introspection ─────────────────────────────────────────

    def get_vocab(self) -> dict[str, int]:
        """Return the token-string → id map.

        Default returns an empty dict; subclasses with a real vocab
        override this.  Used by :meth:`save` and by special-token-id
        resolution in :meth:`_refresh_special_ids`.
        """
        return {}

    def id_to_token(self, token_id: int) -> str | None:
        """Return the surface form for ``token_id`` or ``None`` if
        out of range.

        Default does an O(N) reverse scan of :meth:`get_vocab`;
        subclasses with an O(1) reverse table should override.
        """
        for tok, tid in self.get_vocab().items():
            if tid == token_id:
                return tok
        return None

    def convert_tokens_to_ids(self, tokens: str | Iterable[str]) -> int | list[int]:
        """Map one or many token strings to their ids.

        Unknown tokens raise :class:`KeyError` (callers wanting an
        UNK fallback should do their own ``.get(token, unk_id)``
        lookup).
        """
        vocab = self.get_vocab()
        if isinstance(tokens, str):
            return vocab[tokens]
        return [vocab[t] for t in tokens]

    def convert_ids_to_tokens(self, ids: int | Iterable[int]) -> str | list[str]:
        """Inverse of :meth:`convert_tokens_to_ids`."""
        if isinstance(ids, int):
            t = self.id_to_token(ids)
            if t is None:
                raise KeyError(f"id {ids} out of vocab range")
            return t
        out: list[str] = []
        for i in ids:
            t = self.id_to_token(i)
            if t is None:
                raise KeyError(f"id {i} out of vocab range")
            out.append(t)
        return out

    # ── Special tokens ──────────────────────────────────────────────

    @property
    def special_tokens(self) -> SpecialTokens:
        """The registry of canonical + extra special tokens."""
        return self._special

    @property
    def pad_token_id(self) -> int | None:
        """Id of the ``pad`` special token (``None`` if undefined)."""
        return self._special_ids.get("pad")

    @property
    def unk_token_id(self) -> int | None:
        """Id of the ``unk`` special token (``None`` if undefined)."""
        return self._special_ids.get("unk")

    @property
    def bos_token_id(self) -> int | None:
        """Id of the ``bos`` special token (``None`` if undefined)."""
        return self._special_ids.get("bos")

    @property
    def eos_token_id(self) -> int | None:
        """Id of the ``eos`` special token (``None`` if undefined)."""
        return self._special_ids.get("eos")

    @property
    def mask_token_id(self) -> int | None:
        """Id of the ``mask`` special token (``None`` if undefined)."""
        return self._special_ids.get("mask")

    @property
    def cls_token_id(self) -> int | None:
        """Id of the ``cls`` special token (``None`` if undefined)."""
        return self._special_ids.get("cls")

    @property
    def sep_token_id(self) -> int | None:
        """Id of the ``sep`` special token (``None`` if undefined)."""
        return self._special_ids.get("sep")

    @property
    def all_special_ids(self) -> list[int]:
        """Every special-token id (canonical + extras), deduped."""
        return sorted(set(self._special_ids.values()))

    def _refresh_special_ids(self) -> None:
        """Re-resolve special-token surface forms → ids via the vocab.

        Called automatically after construction + after ``train`` /
        vocab-changing operations.  Subclasses that mutate the vocab
        outside ``__init__`` must call this themselves.
        """
        vocab = self.get_vocab()
        self._special_ids = {}
        for name in _CANONICAL_SPECIAL_SLOTS:
            tok = getattr(self._special, name)
            if tok is not None and tok in vocab:
                self._special_ids[name] = vocab[tok]
        for name, tok in self._special.extra.items():
            if tok in vocab:
                self._special_ids[name] = vocab[tok]

    # ── Universal encode/decode ─────────────────────────────────────

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        """Encode a single string to a list of token ids.

        When ``add_special_tokens=True`` the sequence is framed by the
        tokenizer's :attr:`post_processor` if one is configured (a
        ``tokenizer.json`` carries it — T5's appends only ``</s>``),
        and otherwise by the class's default layout: BOS/EOS, BERT's
        ``[CLS] … [SEP]``, or nothing for GPT-2.

        Parameters
        ----------
        text : str
            Input text to tokenise.  Normalisation + pre-tokenisation
            are applied internally by the algorithm-specific
            :meth:`_encode_one` hook before the BPE / WordPiece /
            Unigram merge loop runs.
        add_special_tokens : bool, optional, keyword-only, default=True
            When ``True``, frame the ids with special tokens as
            described above, via
            :meth:`_build_inputs_with_special_tokens`.  Pass ``False``
            to return the raw merge-loop output untouched (useful for
            offline analysis, alignment, or batching pipelines that
            inject specials elsewhere).

        Returns
        -------
        list of int
            Token ids for ``text`` in left-to-right order.
        """
        ids = self._encode_one(text)
        if add_special_tokens:
            ids = self._build_inputs_with_special_tokens(ids)
        return ids

    def decode(self, ids: Iterable[int] | list[int], *, skip_special_tokens: bool = True) -> str:
        """Decode a sequence of ids back to text.

        When ``skip_special_tokens=True`` every id in
        :attr:`all_special_ids` is dropped before the
        algorithm-specific :meth:`_decode_one` runs.

        Parameters
        ----------
        ids : iterable of int
            Token ids to decode (typically a model's argmax / sample
            output).  Consumed exactly once when an iterator is passed.
        skip_special_tokens : bool, optional, keyword-only, default=True
            When ``True``, every id in :attr:`all_special_ids`
            (BOS/EOS/PAD/CLS/SEP/MASK + any extras) is filtered out
            before :meth:`_decode_one` reassembles the surface string.

        Returns
        -------
        str
            Reconstructed text — exact reverse of :meth:`encode` for
            lossless algorithms (Unigram / byte-BPE); lossy for
            normalising flavours (NFC + Lowercase).
        """
        ids_list = list(ids)
        if skip_special_tokens:
            specials = set(self.all_special_ids)
            ids_list = [i for i in ids_list if i not in specials]
        return self._decode_one(ids_list)

    def encode_batch(
        self, texts: Iterable[str] | list[str], *, add_special_tokens: bool = True
    ) -> list[list[int]]:
        """Vectorised :meth:`encode` over a list of strings.

        Default loops through :meth:`encode`; subclasses with a true
        batched fast path (the C++ Fast tokenizers) override.

        Parameters
        ----------
        texts : iterable of str
            Input strings — one entry per sequence to encode.  An
            iterable is consumed exactly once.
        add_special_tokens : bool, optional, keyword-only, default=True
            Forwarded to :meth:`encode` for each item — when ``True``,
            every sequence is wrapped with its canonical BOS/EOS or
            CLS/SEP markers.

        Returns
        -------
        list of list of int
            One id sequence per input string, same order as ``texts``.
        """
        return [self.encode(t, add_special_tokens=add_special_tokens) for t in texts]

    def decode_batch(
        self,
        batch_ids: Iterable[Iterable[int]] | list[list[int]],
        *,
        skip_special_tokens: bool = True,
    ) -> list[str]:
        """Vectorised :meth:`decode` over a list of id sequences."""
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in batch_ids]

    @property
    def post_processor(self) -> PostProcessor | None:
        """The configured post-processor, or ``None`` for the class default.

        Assign a :class:`~lucid.utils.tokenizer._post_processors.PostProcessor`
        to change how :meth:`encode` frames sequences — for example a
        :class:`~lucid.utils.tokenizer._post_processors.TemplateProcessing`
        reproducing a checkpoint's template — or ``None`` to go back to
        the class's default layout.

        Returns
        -------
        PostProcessor or None
            The processor loaded from ``tokenizer.json`` or assigned here.

        Examples
        --------
        >>> from lucid.utils.tokenizer import SpecialTokens, WordTokenizer
        >>> from lucid.utils.tokenizer._post_processors import TemplateProcessing
        >>> tok = WordTokenizer(
        ...     {"<unk>": 0, "</s>": 1, "hi": 2},
        ...     special_tokens=SpecialTokens(unk="<unk>", eos="</s>"),
        ... )
        >>> tok.encode("hi")
        [2, 1]
        >>> tok.post_processor = TemplateProcessing("$A", special_tokens={})
        >>> tok.encode("hi")
        [2]
        """
        return self._post_processor

    @post_processor.setter
    def post_processor(self, value: PostProcessor | None) -> None:
        """Install ``value`` (or clear it with ``None``)."""
        self._post_processor = value

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        """Count the special tokens ``encode(add_special_tokens=True)`` adds.

        Parameters
        ----------
        pair : bool, default False
            Count for a pair of sequences instead of one.

        Returns
        -------
        int
            How many ids framing adds around the content — the amount
            :meth:`__call__` reserves out of ``max_length`` when it
            truncates.

        Examples
        --------
        >>> from lucid.utils.tokenizer import SpecialTokens, WordTokenizer
        >>> tok = WordTokenizer(
        ...     {"<unk>": 0, "<s>": 1, "</s>": 2},
        ...     special_tokens=SpecialTokens(unk="<unk>", bos="<s>", eos="</s>"),
        ... )
        >>> tok.num_special_tokens_to_add()
        2
        """
        return len(self._build_inputs_with_special_tokens([], [] if pair else None))

    def _special_slot(self, name: str) -> tuple[str, int] | None:
        """``(surface, id)`` of a canonical special token, if it resolves."""
        surface = getattr(self._special, name)
        tid = self._special_ids.get(name)
        if surface is None or tid is None:
            return None
        return surface, tid

    def _default_post_processor(self) -> PostProcessor:
        """The framing :attr:`_special_layout` describes, from the live registry.

        Each framing token is added only if it resolves to an id, so a
        vocabulary without ``[CLS]`` — one trained from a corpus that never
        contained it, say — is left unframed at that end rather than
        failing.
        """
        if self._special_layout == "none":
            return Sequence([])
        if self._special_layout == "cls_sep":
            head, tail = self._special_slot("cls"), self._special_slot("sep")
            # BERT's pair: [CLS] A [SEP] B [SEP] — no second [CLS].
            pair_head = None
        else:
            head, tail = self._special_slot("bos"), self._special_slot("eos")
            # Each sequence of a pair framed alike: BOS A EOS BOS B EOS.
            pair_head = head
        specials: dict[str, list[int]] = {}
        for slot in (head, tail):
            if slot is not None:
                specials[slot[0]] = [slot[1]]

        # Pieces rather than a template string: a surface such as "$x" or
        # "<t>:1" would otherwise be read as template syntax.
        def framed(seq: str, first: tuple[str, int] | None) -> list[_Piece]:
            return [
                *([_Piece("special", first[0], 0)] if first else []),
                _Piece(seq, "", 0),
                *([_Piece("special", tail[0], 0)] if tail else []),
            ]

        single = framed("A", head)
        return TemplateProcessing._from_pieces(single, single + framed("B", pair_head), specials)

    def _build_inputs_with_special_tokens(
        self, ids: list[int], pair_ids: list[int] | None = None
    ) -> list[int]:
        """Frame a raw id sequence (or pair) with special tokens.

        Uses the configured :attr:`post_processor` when there is one and
        the class's :attr:`_special_layout` otherwise.  Subclasses with a
        layout of their own override this; :meth:`num_special_tokens_to_add`
        measures whatever it returns, so truncation stays correct.
        """
        processor = self._post_processor
        if processor is None:
            processor = self._default_post_processor()
        return processor.process(ids, pair_ids)

    # ── HF-style __call__ ───────────────────────────────────────────

    def __call__(
        self,
        text: str | list[str],
        *,
        add_special_tokens: bool = True,
        padding: bool | str = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
        return_attention_mask: bool = True,
    ) -> dict[str, object]:
        """Hugging-Face-style entry point.

        Parameters
        ----------
        text : str or list of str
            Single string → returns dict whose values are 1-D
            sequences.  List → returns dict whose values are 2-D
            batched sequences (rectangular when padded).
        add_special_tokens : bool, default True
            Whether to wrap each sequence with BOS/EOS / CLS/SEP.
        padding : bool or str, default False
            ``False`` — no padding.  ``True`` or ``"longest"`` — pad
            every sequence to the longest in the batch.
            ``"max_length"`` — pad every sequence to ``max_length``.
        truncation : bool, default False
            When ``True``, truncate each sequence to ``max_length``
            (no-op if ``max_length`` is ``None``).  The content is cut,
            never the framing: ``max_length`` counts the special tokens
            and they stay, so an ``EOS`` survives truncation.
        max_length : int, optional
            Bound for both truncation + ``padding="max_length"``.
        return_tensors : {"lucid"} or None
            When ``"lucid"``, wrap each output value as a
            :class:`lucid.Tensor` (dtype int32) and stack the batch
            into a 2-D tensor.  When ``None``, return plain Python
            lists.
        return_attention_mask : bool, default True
            When ``True`` (and ``padding`` is set), include an
            ``"attention_mask"`` entry: 1 for real tokens, 0 for
            padding.

        Returns
        -------
        dict
            Always contains ``"input_ids"``.  Optionally
            ``"attention_mask"`` (when padding + return_attention_mask)
            and ``"special_tokens_mask"`` (always, for downstream
            BERT-style MLM consumers).
        """
        # Normalise to batched form so we can treat single-string and
        # list-of-strings inputs uniformly internally; un-batch the
        # output at the very end if the user passed a single string.
        is_batched = isinstance(text, list)
        texts = text if isinstance(text, list) else [text]
        if not is_batched and not isinstance(text, str):
            raise TypeError(
                f"Tokenizer.__call__: ``text`` must be str or list[str], "
                f"got {type(text).__name__}"
            )

        # Encode each; truncate the content, then frame it.  Cutting the
        # framed sequence instead is how ``</s>`` used to fall off the end:
        # the special tokens are part of the length budget, not of what
        # gets cut.
        encoded: list[list[int]]
        if truncation and max_length is not None:
            content = self.encode_batch(texts, add_special_tokens=False)
            if add_special_tokens:
                budget = max_length - self.num_special_tokens_to_add()
                if budget < 0:
                    raise ValueError(
                        f"Tokenizer.__call__: max_length={max_length} leaves "
                        f"no room for the {self.num_special_tokens_to_add()} "
                        f"special tokens this tokenizer adds."
                    )
                encoded = [self._build_inputs_with_special_tokens(ids[:budget]) for ids in content]
            else:
                encoded = [ids[:max_length] for ids in content]
        else:
            encoded = self.encode_batch(texts, add_special_tokens=add_special_tokens)

        # Padding.
        pad_id = self.pad_token_id
        attention_masks: list[list[int]] | None = None
        if padding is True or padding == "longest":
            target_len = max((len(ids) for ids in encoded), default=0)
            attention_masks = []
            for i, ids in enumerate(encoded):
                pad_n = target_len - len(ids)
                attention_masks.append([1] * len(ids) + [0] * pad_n)
                if pad_n > 0:
                    if pad_id is None:
                        raise ValueError(
                            "Tokenizer.__call__: padding requested but "
                            f"this tokenizer ({self.algo}) has no pad "
                            "token defined.  Set ``special_tokens.pad`` "
                            "or pad manually."
                        )
                    encoded[i] = ids + [pad_id] * pad_n
        elif padding == "max_length":
            if max_length is None:
                raise ValueError(
                    "Tokenizer.__call__: padding='max_length' requires " "max_length to be set."
                )
            attention_masks = []
            for i, ids in enumerate(encoded):
                pad_n = max_length - len(ids)
                if pad_n < 0:
                    raise ValueError(
                        f"Tokenizer.__call__: sequence longer than "
                        f"max_length={max_length} after truncation."
                        " Set truncation=True."
                    )
                attention_masks.append([1] * len(ids) + [0] * pad_n)
                if pad_n > 0:
                    if pad_id is None:
                        raise ValueError(
                            "Tokenizer.__call__: padding='max_length' "
                            f"requested but tokenizer ({self.algo}) has "
                            "no pad token defined."
                        )
                    encoded[i] = ids + [pad_id] * pad_n
        elif padding is not False:
            raise ValueError(
                f"Tokenizer.__call__: unrecognised padding={padding!r}; "
                "expected False / True / 'longest' / 'max_length'."
            )

        # Special-tokens mask — 1 where the id is a special token.
        specials = set(self.all_special_ids)
        sp_masks = [[1 if i in specials else 0 for i in ids] for ids in encoded]

        out: dict[str, object] = {"input_ids": encoded, "special_tokens_mask": sp_masks}
        if attention_masks is not None and return_attention_mask:
            out["attention_mask"] = attention_masks

        # Un-batch when the user passed a single string.
        if not is_batched:
            out = {k: v[0] for k, v in out.items()}

        # Tensor wrap.
        if return_tensors is not None:
            if return_tensors != "lucid":
                raise ValueError(
                    f"Tokenizer.__call__: unsupported "
                    f"return_tensors={return_tensors!r}; only 'lucid' "
                    "is supported."
                )
            out = _wrap_lucid_tensors(out, is_batched)

        return out

    # ── Persistence ─────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        """Persist the tokenizer to ``directory``.

        Default writes ``tokenizer.json`` (unified format) +
        ``special_tokens_map.json``.  Subclasses with algorithm-
        specific legacy files (``vocab.json`` + ``merges.txt`` for
        BPE, ``vocab.txt`` for WordPiece) override this hook to
        emit those alongside.
        """
        import json
        import os

        os.makedirs(directory, exist_ok=True)
        # Universal tokenizer.json with algo + vocab + algorithm-
        # specific extras delegated to subclass via ``_save_extras``.
        unified: dict[str, object] = {"algo": self.algo, "vocab": self.get_vocab()}
        extras = self._save_extras()
        if extras:
            unified.update(extras)
        with open(os.path.join(directory, "tokenizer.json"), "w") as f:
            json.dump(unified, f, indent=2, ensure_ascii=False)
        # Special-tokens map.
        sp: dict[str, object] = {}
        for name in _CANONICAL_SPECIAL_SLOTS:
            v = getattr(self._special, name)
            if v is not None:
                sp[name + "_token"] = v
        if self._special.extra:
            sp["additional_special_tokens"] = list(self._special.extra.values())
        with open(os.path.join(directory, "special_tokens_map.json"), "w") as f:
            json.dump(sp, f, indent=2, ensure_ascii=False)

    def _save_extras(self) -> dict[str, object]:
        """Hook for subclasses to add algorithm-specific entries to
        the unified ``tokenizer.json``.  Default: nothing extra.
        """
        return {}


# ── Tensor-wrap helper (module-level so other tokenizers can reuse) ──


def _wrap_lucid_tensors(out: dict[str, object], is_batched: bool) -> dict[str, object]:
    """Convert plain int lists in ``out`` to :class:`lucid.Tensor` (I32).

    Single-string outputs become 1-D ``(seq_len,)`` tensors; batched
    outputs become 2-D ``(batch, seq_len)`` tensors (rectangular by
    construction since the caller already padded).
    """
    import lucid

    wrapped: dict[str, object] = {}
    for k, v in out.items():
        if is_batched:
            # v is list[list[int]] — rectangular if padding was applied.
            # If not padded, raise (Lucid tensors must be rectangular).
            assert isinstance(v, list)
            lens = {len(row) for row in v}
            if len(lens) > 1:
                raise ValueError(
                    f"Tokenizer.__call__: return_tensors='lucid' requires "
                    f"rectangular batches but '{k}' has varying lengths "
                    f"{sorted(lens)}.  Set padding=True."
                )
            wrapped[k] = lucid.tensor(v, dtype=lucid.int32)
        else:
            # v is list[int] — 1-D.
            wrapped[k] = lucid.tensor(v, dtype=lucid.int32)
    return wrapped


# Build / register a Callable hook so vendored Fast wrappers can drop
# in custom logic later (e.g. C++ batched encode).  Currently a no-op
# placeholder — subclasses just override ``encode_batch`` directly.
_ENCODE_HOOK: Callable[[Tokenizer, list[str]], list[list[int]]] | None = None
