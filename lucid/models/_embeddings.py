"""Surgery on a model's vocabulary-sized tables: the pieces ``resize_token_embeddings`` is built from.

A model's vocabulary lives in three places that have to agree after a resize: the
table that looks token ids up, the projection that scores them (tied to the table
or not), and the config's size fields.  :class:`~lucid.models.PretrainedModel`
decides *which* tables a model has; this module only builds the replacements --
each on the device and dtype of what it replaces, keeping every row both have,
and leaving the rest to the family's own initialiser -- so the base class owns
the rule and no family repeats the arithmetic.
"""

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import lucid
import lucid.nn as nn
from lucid._tensor.tensor import Tensor

__all__ = ["VocabSized", "fit_vocab_bias", "resized_embedding", "resized_projection"]

#: What a family hands in to initialise a table it has just allocated, the way it
#: initialises the tables it builds itself.
type InitTable = Callable[[nn.Embedding | nn.Linear], None]


@runtime_checkable
class VocabSized(Protocol):
    """A module that owns a vocabulary-sized parameter beside the projection it holds.

    BERT's prediction head keeps its output bias on itself, not on the ``Linear``
    that decodes, so replacing that ``Linear`` leaves a bias of the old length
    behind.  The holder fits it.
    """

    def fit_vocab(self, vocab: int) -> None: ...


def _parameter_pairs(
    new: nn.Embedding | nn.Linear, old: nn.Embedding | nn.Linear
) -> list[tuple[nn.Parameter, nn.Parameter]]:
    """``(new, old)`` for the weight, and for the bias when both have one."""
    pairs = [(new.weight, old.weight)]
    if not (isinstance(new, nn.Linear) and isinstance(old, nn.Linear)):
        return pairs
    if new.bias is not None and old.bias is not None:
        pairs.append((new.bias, old.bias))
    return pairs


def _inherit(new: nn.Embedding | nn.Linear, old: nn.Embedding | nn.Linear) -> None:
    """Write the rows ``new`` and ``old`` share into ``new``, with ``old``'s train mode and freezing."""
    new.train(old.training)
    with lucid.no_grad():
        for fresh, kept in _parameter_pairs(new, old):
            keep = min(int(kept.shape[0]), int(fresh.shape[0]))
            fresh[:keep] = kept[:keep]
    for fresh, kept in _parameter_pairs(new, old):
        fresh.requires_grad = kept.requires_grad


def resized_embedding(old: nn.Embedding, num_tokens: int, init: InitTable) -> nn.Embedding:
    """A table of ``num_tokens`` rows holding the leading rows of ``old``.

    Rows past ``old``'s are initialised by ``init``, which runs on the whole new
    table first so that the family's rule (not the framework's default) draws
    them; the kept rows are copied over it afterwards.  ``padding_idx`` stays
    while it is still a row of the new table.

    Parameters
    ----------
    old : nn.Embedding
        The table to replace; it is not modified.
    num_tokens : int
        Row count of the new table.
    init : InitTable
        The family's initialiser for a table it has just allocated.

    Returns
    -------
    nn.Embedding
        The new table, on ``old``'s device and dtype, in its train mode and with
        its ``requires_grad``.
    """
    padding = (
        old.padding_idx if old.padding_idx is not None and old.padding_idx < num_tokens else None
    )
    new = nn.Embedding(
        num_tokens,
        old.embedding_dim,
        padding_idx=padding,
        max_norm=old.max_norm,
        norm_type=old.norm_type,
        scale_grad_by_freq=old.scale_grad_by_freq,
        sparse=old.sparse,
        device=old.weight.device,
        dtype=old.weight.dtype,
    )
    init(new)
    _inherit(new, old)
    return new


def resized_projection(old: nn.Linear, num_tokens: int, init: InitTable) -> nn.Linear:
    """A ``Linear`` scoring ``num_tokens`` tokens, holding the leading rows of ``old``.

    The counterpart of :func:`resized_embedding` for an output head that is not
    tied to the input table, so it is a table of its own to resize.

    Parameters
    ----------
    old : nn.Linear
        The head to replace; it is not modified.
    num_tokens : int
        Output width of the new head.
    init : InitTable
        The family's initialiser for a table it has just allocated.

    Returns
    -------
    nn.Linear
        The new head, with the bias ``old`` had, on its device and dtype.
    """
    new = nn.Linear(
        old.in_features,
        num_tokens,
        bias=old.bias is not None,
        device=old.weight.device,
        dtype=old.weight.dtype,
    )
    init(new)
    _inherit(new, old)
    return new


def fit_vocab_bias(bias: Tensor, vocab: int) -> Tensor:
    """``bias`` fitted to ``vocab`` entries: the leading ones kept, any new ones zero.

    Parameters
    ----------
    bias : Tensor
        A one-dimensional output bias, one entry per token.
    vocab : int
        The number of tokens the head now scores.

    Returns
    -------
    Tensor
        ``bias`` itself if it already has ``vocab`` entries, else a new parameter
        of that size on the same device and dtype.
    """
    have = int(bias.shape[0])
    if have == vocab:
        return bias
    kept = bias.detach()[: min(have, vocab)].clone()
    if vocab > have:
        pad = lucid.zeros(vocab - have, dtype=kept.dtype, device=kept.device.type)
        kept = lucid.cat([kept, pad])
    return nn.Parameter(kept, requires_grad=bias.requires_grad)
