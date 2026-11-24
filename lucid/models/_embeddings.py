"""Surgery on a model's vocabulary-sized tables: the pieces ``resize_token_embeddings`` is built from.

A model's vocabulary lives in three places that have to agree after a resize: the
table that looks token ids up, the projection that scores them (tied to the table
or not), and the config's size fields.  :class:`~lucid.models.PretrainedModel`
decides *which* tables a model has; this module only builds the replacements --
each on the device and dtype of what it replaces, keeping every row both have,
and drawing the rest from the old rows' own statistics or the family's own
initialiser -- so the base class owns the rule and no family repeats the arithmetic.
"""

from collections.abc import Callable
from typing import Final, Protocol, TypeGuard, runtime_checkable

import lucid
import lucid.nn as nn
from lucid._tensor.tensor import Tensor

__all__ = [
    "VocabSized",
    "fit_vocab_bias",
    "resized_embedding",
    "resized_projection",
    "resized_vocab_size",
]

#: How much of the old rows' covariance the new rows are drawn with: they land within a
#: hair of the old rows' mean, the reference's value for ``mean_resizing``.
_COVARIANCE_SCALE: Final = 1e-9

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

    def fit_vocab(self, vocab: int, *, mean_resizing: bool = False) -> None: ...


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


def _mean_rows(weight: Tensor, added: int) -> Tensor:
    """``added`` rows drawn from the statistics of the rows of ``weight``.

    The rows' mean and covariance (in float32) give a normal distribution; its
    covariance is scaled by ``1e-9``, so each new row sits a hair from the mean
    rather than at a random point of the embedding space.  A covariance that is not
    positive definite (fewer rows than dimensions, say) has no such distribution,
    and every new row is the mean.  The draw follows ``lucid.manual_seed``.
    """
    rows = weight.detach().float()
    mean = rows.mean(dim=0)
    centered = rows - mean
    covariance = (centered.mT @ centered) / int(rows.shape[0])
    factor, info = lucid.linalg.cholesky_ex(_COVARIANCE_SCALE * covariance)
    if int(info.item()) != 0:
        return mean.unsqueeze(0).repeat(added, 1).to(weight.dtype)
    noise = lucid.randn(added, int(rows.shape[1]), device=rows.device.type)
    return (mean + noise @ factor.mT).to(weight.dtype)


def _mean_entries(bias: Tensor, added: int) -> Tensor:
    """``added`` entries drawn from a normal with the mean and standard deviation of ``bias``.

    The standard deviation is the sample one, as the reference's; a bias of one entry
    has none, and its entries repeat it.  The draw follows ``lucid.manual_seed``.
    """
    entries = bias.detach().float()
    device = entries.device.type
    spread = entries.std() if int(entries.shape[0]) > 1 else lucid.zeros((), device=device)
    noise = lucid.randn(added, device=device)
    return (entries.mean() + noise * spread).to(bias.dtype)


def _draw_added(new: nn.Embedding | nn.Linear, old: nn.Embedding | nn.Linear) -> None:
    """Give the rows ``new`` has past ``old``'s the mean-resizing draw: weight rows, then bias entries."""
    first = int(old.weight.shape[0])
    added = int(new.weight.shape[0]) - first
    if added <= 0:
        return
    with lucid.no_grad():
        for position, (fresh, kept) in enumerate(_parameter_pairs(new, old)):
            fresh[first:] = _mean_rows(kept, added) if position == 0 else _mean_entries(kept, added)


def _is_int(value: object) -> TypeGuard[int]:
    """An ``int`` that is not a ``bool``."""
    return isinstance(value, int) and not isinstance(value, bool)


def _pad_multiple(value: object) -> int | None:
    """``value`` as a multiple to round up to: ``None`` for none, else a positive ``int`` or ``ValueError``."""
    if value is None:
        return None
    if _is_int(value) and value > 0:
        return value
    raise ValueError(
        f"Asking to pad the embedding matrix to a multiple of `{value}`, "
        f"which is not a positive integer. Please make sure to pass a positive integer"
    )


def resized_vocab_size(current: int, new_num_tokens: object, pad_to_multiple_of: object) -> int:
    """The row count a resize asks for: ``new_num_tokens`` (``current`` if ``None``), rounded up to a multiple.

    Both requests are the caller's, so they are checked here rather than trusted.

    Parameters
    ----------
    current : int
        The table's present row count.
    new_num_tokens : int or None
        The requested size, or ``None`` to keep ``current``.
    pad_to_multiple_of : int or None
        Round the size up to a multiple of this, as the reference does to keep a
        table's shape friendly to the hardware; ``None`` for no rounding.

    Returns
    -------
    int
        The row count to resize to.

    Raises
    ------
    TypeError
        If ``new_num_tokens`` is neither an ``int`` nor ``None``.
    ValueError
        If ``new_num_tokens`` is below 1, or ``pad_to_multiple_of`` is not a
        positive ``int``.
    """
    multiple = _pad_multiple(pad_to_multiple_of)
    if new_num_tokens is None:
        size = current
    elif not _is_int(new_num_tokens):
        raise TypeError(
            f"resize_token_embeddings: new_num_tokens must be an int or None, "
            f"got {type(new_num_tokens).__name__}"
        )
    elif new_num_tokens < 1:
        raise ValueError(
            f"resize_token_embeddings: new_num_tokens must be positive, got {new_num_tokens}"
        )
    else:
        size = new_num_tokens
    return size if multiple is None else -(-size // multiple) * multiple


def resized_embedding(
    old: nn.Embedding, num_tokens: int, init: InitTable, *, mean_resizing: bool = False
) -> nn.Embedding:
    """A table of ``num_tokens`` rows holding the leading rows of ``old``.

    The new table is initialised by ``init`` first, so that the family's rule (not
    the framework's default) is what draws any row the old table lacks; the kept
    rows are copied over it afterwards.  With ``mean_resizing`` those new rows are
    redrawn from the statistics of the old ones instead (see :func:`_mean_rows`).
    ``padding_idx`` stays while it is still a row of the new table.

    Parameters
    ----------
    old : nn.Embedding
        The table to replace; it is not modified.
    num_tokens : int
        Row count of the new table.
    init : InitTable
        The family's initialiser for a table it has just allocated.
    mean_resizing : bool, optional, keyword-only
        Draw the new rows from the old rows' mean and covariance.  Default: ``False``.

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
    if mean_resizing:
        _draw_added(new, old)
    return new


def resized_projection(
    old: nn.Linear, num_tokens: int, init: InitTable, *, mean_resizing: bool = False
) -> nn.Linear:
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
    mean_resizing : bool, optional, keyword-only
        Draw the new rows, and the new bias entries, from the old ones' statistics.
        Default: ``False``.

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
    if mean_resizing:
        _draw_added(new, old)
    return new


def fit_vocab_bias(bias: Tensor, vocab: int, *, mean_resizing: bool = False) -> Tensor:
    """``bias`` fitted to ``vocab`` entries: the leading ones kept, any new ones zero.

    A bias that follows a tied head gets zeros; one that belongs to a head of its own
    gets, with ``mean_resizing``, draws from the old entries' mean and spread.

    Parameters
    ----------
    bias : Tensor
        A one-dimensional output bias, one entry per token.
    vocab : int
        The number of tokens the head now scores.
    mean_resizing : bool, optional, keyword-only
        Draw the new entries from the old ones' mean and spread instead of zeros.

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
        if mean_resizing:
            pad = _mean_entries(bias, vocab - have)
        else:
            pad = lucid.zeros(vocab - have, dtype=kept.dtype, device=kept.device.type)
        kept = lucid.cat([kept, pad])
    return nn.Parameter(kept, requires_grad=bias.requires_grad)
