"""``lucid.utils.tokenizer`` — text tokenization sub-package.

Every tokenizer comes in two flavours:

* ``XxxTokenizer`` — pure-Python reference implementation.  Easy to
  read, easy to debug, easy to extend with custom normalizers.
* ``XxxTokenizerFast`` — C++-backed via
  ``lucid._C.engine.utils.tokenizer``.  Same vocab format, same
  encode output bit-for-bit, but the algorithm hot loop runs in C++.

Both flavours share the Hugging Face-compatible on-disk format
(``vocab.json`` + ``merges.txt`` legacy or unified
``tokenizer.json``) so any published HF checkpoint loads without
modification.  The :class:`Tokenizer` base class layers a uniform
HF-style API on top — :meth:`~Tokenizer.encode` /
:meth:`~Tokenizer.decode` / batch versions / HF-style
:meth:`~Tokenizer.__call__` with padding + truncation +
``return_tensors='lucid'``.

Algorithms
----------

==========  ===============  =============================
algo        modules           used by
==========  ===============  =============================
BPE         ``_bpe``         (raw BPE — base for byte-BPE)
ByteLevel   ``_byte_bpe``    GPT, GPT-2, RoBERTa, BART
WordPiece   ``_wordpiece``   BERT, RoFormer, DistilBERT
Unigram     ``_unigram``     T5, LLaMA, Mistral, mBART
==========  ===============  =============================

Each is exported from this package with a ``Fast`` variant beside it —
:class:`ByteLevelBPETokenizer`, :class:`WordPieceTokenizer`,
:class:`UnigramTokenizer`.

Pipeline stages
---------------

A Hugging Face ``tokenizer.json`` names four stages besides the model, and
each has a sub-module here with a ``*_from_config`` factory that reads its
block: :mod:`normalizers` (text clean-up, including SentencePiece's
compiled ``Precompiled`` map), :mod:`pre_tokenizers` (splitting, including
``Metaspace``), :mod:`post_processors` (the special tokens that frame a
sequence — assign one to :attr:`Tokenizer.post_processor`) and
:mod:`decoders` (ids back to text).

Per-model wrappers live in each ``lucid.models.text.<family>``
package's ``_tokenizer/`` directory and subclass the algorithm
matching that family (e.g. ``BERTTokenizer`` ←
:class:`WordPieceTokenizer`).
"""

from lucid.utils.tokenizer._base import SpecialTokens, Tokenizer
from lucid.utils.tokenizer._bpe import BPETokenizer, BPETokenizerFast
from lucid.utils.tokenizer._byte import ByteTokenizer, ByteTokenizerFast
from lucid.utils.tokenizer._byte_bpe import ByteLevelBPETokenizer, ByteLevelBPETokenizerFast
from lucid.utils.tokenizer._char import CharTokenizer, CharTokenizerFast
from lucid.utils.tokenizer._regex import RegexTokenizer, RegexTokenizerFast
from lucid.utils.tokenizer._unigram import UnigramTokenizer, UnigramTokenizerFast
from lucid.utils.tokenizer._whitespace import WhitespaceTokenizer, WhitespaceTokenizerFast
from lucid.utils.tokenizer._word import WordTokenizer, WordTokenizerFast
from lucid.utils.tokenizer._wordpiece import WordPieceTokenizer, WordPieceTokenizerFast
from lucid.utils.tokenizer import _decoders as decoders
from lucid.utils.tokenizer import _normalizers as normalizers
from lucid.utils.tokenizer import _post_processors as post_processors
from lucid.utils.tokenizer import _pre_tokenizers as pre_tokenizers

__all__ = [
    # Base
    "Tokenizer",
    "SpecialTokens",
    # Tier 0 — primitive / no-vocab
    "ByteTokenizer",
    "ByteTokenizerFast",
    "CharTokenizer",
    "CharTokenizerFast",
    # Tier 1 — rule-based / vocab-lookup
    "WhitespaceTokenizer",
    "WhitespaceTokenizerFast",
    "WordTokenizer",
    "WordTokenizerFast",
    "RegexTokenizer",
    "RegexTokenizerFast",
    # Tier 2 — subword
    "BPETokenizer",
    "BPETokenizerFast",
    "ByteLevelBPETokenizer",
    "ByteLevelBPETokenizerFast",
    "WordPieceTokenizer",
    "WordPieceTokenizerFast",
    "UnigramTokenizer",
    "UnigramTokenizerFast",
    # Sub-modules — the four ``tokenizer.json`` pipeline stages
    "normalizers",
    "pre_tokenizers",
    "post_processors",
    "decoders",
]
