"""How a map-style dataset hands the loader a whole batch.

Two mechanisms, deliberately kept apart.

``__getitems__(indices)`` is the public protocol.  It returns a **list of
samples** that then goes through ``collate_fn`` — a dataset that can read a
batch cheaper than ``batch_size`` calls to ``__getitem__`` still leaves
collation to the loader.

``_collated_fetcher()`` is Lucid's own shortcut: a dataset that knows how
to produce the *already default-collated* batch (``TensorDataset`` does it
with one fancy-index per tensor) returns a callable for it.  The loader
takes that only when the collate function is the default one.

A shortcut is a promise that it returns what ``__getitem__`` followed by
``default_collate`` would.  A subclass that overrides ``__getitem__`` (or
``__getitems__``) breaks the promise without touching the method that makes
it, so the promise does not inherit: :func:`collated_fetcher` honours it
only for a class whose ``__getitem__`` and ``__getitems__`` are the ones
defined next to the ``_collated_fetcher`` it is about to call.  That one
rule lives here and nowhere else.
"""

from collections.abc import Callable
from typing import Protocol, runtime_checkable

_BatchFetcher = Callable[[list[int]], object]


class _MapStyle(Protocol):
    """Anything a sample can be read from by integer index."""

    def __getitem__(self, index: int, /) -> object: ...


@runtime_checkable
class _CollatedFetch(Protocol):
    """A dataset that can produce an already-collated batch."""

    def _collated_fetcher(self) -> _BatchFetcher | None:
        """Return the batch callable, or ``None`` when this instance has none."""
        ...


def collated_fetcher(dataset: object) -> _BatchFetcher | None:
    """The already-collated batch fast path of ``dataset``, if it is still honest.

    Returns ``None`` when the dataset offers no shortcut, when its instance
    has none to give, or when its class overrides ``__getitem__`` or
    ``__getitems__`` after the class that wrote the shortcut.
    """
    if not isinstance(dataset, _CollatedFetch):
        return None
    cls = type(dataset)
    owner = next(k for k in cls.__mro__ if "_collated_fetcher" in vars(k))
    for name in ("__getitem__", "__getitems__"):
        if getattr(cls, name, None) is not getattr(owner, name, None):
            return None
    return dataset._collated_fetcher()


def fetch_samples(dataset: _MapStyle, indices: list[int]) -> list[object]:
    """The samples at ``indices``, through ``__getitems__`` when the dataset has one.

    The result is a list of samples for ``collate_fn`` — never a batch —
    whichever way it was read, so a caller treats both kinds of dataset alike.

    Parameters
    ----------
    dataset : Dataset
        A map-style dataset (anything indexable by integer).
    indices : list of int
        Positions to read, in the order the samples come back.

    Returns
    -------
    list of object
        One sample per index.
    """
    getitems = getattr(dataset, "__getitems__", None)
    if callable(getitems):
        return list(getitems(indices))
    return [dataset[i] for i in indices]
