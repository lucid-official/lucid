"""
Sampler classes for DataLoader.

Every random sampler draws from Lucid's Philox generators — the default
one (seeded by :func:`lucid.manual_seed`) or an explicit
:class:`lucid.Generator` — so shuffles are reproducible with the rest of
a run.  :class:`DistributedSampler` and :class:`RASampler` keep their
own ``seed + epoch`` streams, which every replica must agree on without
sharing a generator.
"""

import bisect
import itertools
import math
import random
from typing import Iterator, Sequence, cast, override

import lucid
from lucid._factories.random import Generator
from lucid._tensor.tensor import Tensor
from lucid.utils.data.dataset import Dataset
from lucid.utils.data._validate import _check_bool, _check_positive_int
from lucid.utils.data._rng import (
    _as_generator,
    _draw_seed,
    _host_float64,
    _permutation,
    _uniform_doubles,
    _uniform_indices,
)


class Sampler:
    """Abstract base class for index samplers used by :class:`DataLoader`.

    A sampler iterates over integer indices into a map-style dataset.
    Subclasses must implement :meth:`__iter__` (yielding indices) and
    :meth:`__len__` (total number of indices produced per epoch).

    Notes
    -----
    Samplers decouple *which* samples are visited from *how* they are
    fetched — the dataset answers ``__getitem__(i)`` and the sampler
    decides the sequence of ``i`` values.  This separation is what lets
    :class:`DataLoader` swap iteration policies (sequential / random /
    weighted / distributed) without touching the dataset.

    Examples
    --------
    >>> class Even(Sampler):
    ...     def __init__(self, n): self.n = n
    ...     def __iter__(self): return iter(range(0, self.n, 2))
    ...     def __len__(self): return (self.n + 1) // 2
    >>> list(Even(6))
    [0, 2, 4]
    """

    def __iter__(self) -> Iterator[int]:
        """Yield integer sample indices.

        Returns
        -------
        Iterator[int]
            Iterator over ``0 <= i < len(data_source)`` values, in an
            order defined by the concrete sampler.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """Return the number of indices yielded in one full pass."""
        raise NotImplementedError


class SequentialSampler(Sampler):
    """Yield indices in fixed order ``0, 1, ..., len(data_source) - 1``.

    The default sampler used when ``shuffle=False`` and no explicit
    sampler is supplied to :class:`DataLoader`.

    Parameters
    ----------
    data_source : Dataset
        Dataset whose ``__len__`` determines the index range.

    Notes
    -----
    Order is fully deterministic — no RNG is consulted — so two passes
    over the sampler always yield the same index sequence.  This is the
    right choice for evaluation / inference loops where ordering must
    line up with external bookkeeping (e.g. per-row metric arrays).

    Examples
    --------
    >>> import lucid
    >>> from lucid.utils.data import SequentialSampler, TensorDataset
    >>> my_dataset = TensorDataset(lucid.randn(10, 3))
    >>> sampler = SequentialSampler(my_dataset)
    >>> list(sampler)[:5]
    [0, 1, 2, 3, 4]
    """

    def __init__(self, data_source: Dataset) -> None:
        """Store ``data_source`` for length introspection.

        Parameters
        ----------
        data_source : Dataset
            Dataset to be iterated sequentially.
        """
        self.data_source = data_source

    @override
    def __iter__(self) -> Iterator[int]:
        """Yield ``0, 1, ..., len(data_source) - 1`` in order."""
        return iter(range(len(self.data_source)))

    @override
    def __len__(self) -> int:
        """Return ``len(data_source)``."""
        return len(self.data_source)


class RandomSampler(Sampler):
    """Yield indices in random order, with or without replacement.

    Parameters
    ----------
    data_source : Dataset
        Dataset whose ``__len__`` defines the index range ``[0, n)``.
    replacement : bool, optional
        If ``True``, indices are drawn with replacement (any index can
        appear multiple times). If ``False`` (default), indices are a
        random permutation of ``range(n)`` and each index appears exactly
        once per epoch.
    num_samples : int, optional
        Number of indices to draw per epoch. Defaults to
        ``len(data_source)``.  Without replacement a value above ``n``
        chains whole permutations (every index appears
        ``num_samples // n`` times, plus a partial pass); a value below
        ``n`` truncates the permutation.
    generator : lucid.Generator or int, optional
        Stream the order is drawn from.  A :class:`lucid.Generator`
        advances from epoch to epoch, so a fixed one reproduces the whole
        sequence of epochs; an ``int`` seeds a private generator once,
        with the same effect.  ``None`` (default) draws a fresh seed from
        the default generator every epoch, so :func:`lucid.manual_seed`
        reproduces the order.

    Notes
    -----
    Without replacement, each epoch is a fresh uniform permutation of
    ``range(n)`` — equivalent to shuffling.  With replacement, indices
    are i.i.d. uniform draws and ``num_samples`` controls the per-epoch
    budget independently of ``n``; this lets the caller oversample
    (``num_samples > n``) for stochastic training regimes.

    Examples
    --------
    >>> import lucid
    >>> from lucid.utils.data import RandomSampler, TensorDataset
    >>> my_dataset = TensorDataset(lucid.randn(10, 3))
    >>> sampler = RandomSampler(my_dataset)
    >>> for idx in sampler:
    ...     x = my_dataset[idx]
    >>> sorted(sampler) == list(range(10))       # a permutation of the indices
    True
    """

    def __init__(
        self,
        data_source: Dataset,
        replacement: bool = False,
        num_samples: int | None = None,
        generator: Generator | int | None = None,
    ) -> None:
        """Configure the random sampler.

        Parameters
        ----------
        data_source : Dataset
            Source dataset.
        replacement : bool
            See class docstring.
        num_samples : int, optional
            See class docstring.
        generator : lucid.Generator or int, optional
            See class docstring.

        Raises
        ------
        TypeError
            If ``replacement`` is not a ``bool``, or ``generator`` is
            neither a :class:`lucid.Generator`, a seed, nor ``None``.
        ValueError
            If ``num_samples`` is not a positive integer.
        """
        self.data_source = data_source
        _check_bool("replacement", replacement)
        self.replacement = replacement
        if num_samples is not None:
            # Caught here rather than at the first ``len()``, where it
            # surfaced as ``__len__() should return >= 0`` — an error about
            # the protocol rather than about the argument that caused it.
            _check_positive_int("num_samples", num_samples)
        self._num_samples = num_samples
        self.generator: Generator | None = _as_generator(generator, "RandomSampler")

    @property
    def num_samples(self) -> int:
        """Effective number of samples per epoch.

        Returns the explicit ``num_samples`` argument if provided,
        otherwise ``len(data_source)``.
        """
        return self._num_samples if self._num_samples is not None else len(self.data_source)

    @override
    def __iter__(self) -> Iterator[int]:
        """Yield ``num_samples`` indices according to the configured strategy.

        With ``replacement=True`` each index is an independent uniform
        draw from ``range(n)``.  With ``replacement=False`` the indices
        are a fresh permutation of ``range(n)`` (chained, then truncated,
        to reach ``num_samples``).  Both draw from ``generator``, or
        — when it is ``None`` — from a per-epoch generator seeded off the
        default one.
        """
        n = len(self.data_source)
        if n == 0:
            if self.num_samples:
                raise ValueError(f"cannot draw {self.num_samples} samples from an empty dataset.")
            return
        rng = self.generator
        if rng is None:
            rng = Generator(_draw_seed(None))
        if self.replacement:
            yield from _uniform_indices(n, self.num_samples, rng)
            return
        for _ in range(self.num_samples // n):
            yield from _permutation(n, rng)
        remainder = self.num_samples % n
        if remainder:
            yield from _permutation(n, rng)[:remainder]

    @override
    def __len__(self) -> int:
        """Return :attr:`num_samples`."""
        return self.num_samples


class SubsetRandomSampler(Sampler):
    """Yield a random permutation of a fixed list of indices each epoch.

    Useful when the caller already knows which subset of the dataset
    should be visited (e.g., a precomputed train/val split) and only
    wants shuffling among that subset.

    Parameters
    ----------
    indices : list of int
        Indices into the parent dataset to sample from.
    generator : lucid.Generator or int, optional
        Stream the order is drawn from; advances from epoch to epoch.
        ``None`` (default) draws from the default generator, so
        :func:`lucid.manual_seed` reproduces the order.

    Notes
    -----
    The index pool itself is fixed at construction time; only the
    *order* changes between epochs.  Useful with precomputed
    cross-validation folds — store the per-fold index list once, then
    instantiate one :class:`SubsetRandomSampler` per fold.

    Examples
    --------
    >>> fold_indices = [3, 5, 7, 9, 11]
    >>> sampler = SubsetRandomSampler(fold_indices)
    >>> sorted(list(sampler)) == fold_indices
    True
    """

    def __init__(self, indices: Sequence[int], generator: Generator | int | None = None) -> None:
        """Store the index pool and optional generator handle.

        Parameters
        ----------
        indices : sequence of int
            Indices to sample from.
        generator : lucid.Generator or int, optional
            See class docstring.
        """
        self.indices = list(indices)
        self.generator: Generator | None = _as_generator(generator, "SubsetRandomSampler")

    @override
    def __iter__(self) -> Iterator[int]:
        """Yield ``self.indices`` in a freshly shuffled order each epoch."""
        indices = self.indices
        for i in _permutation(len(indices), self.generator):
            yield indices[i]

    @override
    def __len__(self) -> int:
        """Return the size of the index pool."""
        return len(self.indices)


class WeightedRandomSampler(Sampler):
    r"""Yield indices drawn proportionally to user-supplied weights.

    Each index ``i`` is selected with probability proportional to
    ``weights[i]``. Internally the weights are normalised by their sum so
    they need not form a true probability distribution on input.

    Parameters
    ----------
    weights : sequence of float or Tensor
        Non-negative, finite weight per index. The effective probability
        of index ``i`` is :math:`p_i = w_i / \sum_j w_j`.
    num_samples : int
        Number of indices to draw per epoch.
    replacement : bool, optional
        If ``True`` (default), draws are independent with replacement —
        the same index may appear multiple times.  If ``False``, each
        index is drawn at most once: the pool shrinks as indices are
        taken, so ``num_samples`` may not exceed ``len(weights)`` and is
        refused at construction if it does.
    generator : lucid.Generator or int, optional
        Stream the draws come from; advances from epoch to epoch, so each
        epoch is a new draw and a fixed generator reproduces the sequence
        of epochs.  ``None`` (default) draws from the default generator,
        so :func:`lucid.manual_seed` reproduces the draws.

    Notes
    -----
    Each index :math:`i` is selected with probability

    .. math::

        P(i) = \frac{w_i}{\sum_j w_j},

    so the user-supplied weights need not be normalised.  The classic
    use case is *class-imbalance correction*: set ``w_i = 1 /
    class_count[label_i]`` so under-represented classes are upsampled
    to roughly uniform frequency.  Sampling with replacement is the
    default — it preserves the target marginal exactly and is the only
    fully consistent option when ``num_samples`` exceeds the number of
    nonzero-weight indices.

    With replacement each draw inverts the cumulative weights by binary
    search, :math:`O(n + k \log n)` per epoch for ``k`` draws.  Without
    replacement the indices are ranked by the exponential keys
    :math:`\log(u_i) / w_i` (Efraimidis & Spirakis, 2006) and the
    ``num_samples`` largest are kept, in the order one-at-a-time draws
    would pick them.  When fewer than ``num_samples`` weights are
    positive, the zero-weight indices follow the positive ones (lowest
    index first), so an epoch always yields exactly ``num_samples``
    indices and ``len(DataLoader)`` stays true.

    Examples
    --------
    >>> # 3 classes, counts [900, 90, 10]; upweight rare classes
    >>> import lucid
    >>> from lucid.utils.data import TensorDataset, WeightedRandomSampler
    >>> labels = [0] * 900 + [1] * 90 + [2] * 10
    >>> my_dataset = TensorDataset(lucid.randn(1000, 4), lucid.tensor(labels))
    >>> weights = [1/900]*900 + [1/90]*90 + [1/10]*10
    >>> sampler = WeightedRandomSampler(weights, num_samples=1000, generator=0)
    >>> for idx in sampler:
    ...     x, y = my_dataset[idx]
    >>> counts = [0, 0, 0]
    >>> for idx in sampler:
    ...     counts[labels[idx]] += 1
    >>> all(250 < c < 420 for c in counts)       # the classes now arrive evenly
    True
    """

    def __init__(
        self,
        weights: Sequence[float] | Tensor,
        num_samples: int,
        replacement: bool = True,
        generator: Generator | int | None = None,
    ) -> None:
        """Store sampling configuration.

        Parameters
        ----------
        weights : sequence of float or Tensor
            See class docstring.
        num_samples : int
            See class docstring.
        replacement : bool
            See class docstring.
        generator : lucid.Generator or int, optional
            See class docstring.

        Raises
        ------
        TypeError
            If ``replacement`` is not a ``bool``.
        ValueError
            If ``num_samples`` is not a positive integer, exceeds
            ``len(weights)`` without replacement, or a weight is negative
            or not finite, or — with replacement — every weight is zero.
        """
        values = (
            cast(list[float], weights.reshape(-1).tolist())
            if isinstance(weights, Tensor)
            else weights
        )
        self.weights: list[float] = [float(w) for w in values]
        _check_positive_int("num_samples", num_samples)
        _check_bool("replacement", replacement)
        if not replacement and num_samples > len(self.weights):
            # There are not that many distinct indices to give.  Refused
            # here rather than silently repeating, which is what this did.
            raise ValueError(
                f"cannot draw {num_samples} indices without replacement from "
                f"{len(self.weights)} weights — pass replacement=True or "
                f"lower num_samples."
            )
        if any(w < 0 for w in self.weights):
            raise ValueError("weights must be non-negative.")
        if not all(math.isfinite(w) for w in self.weights):
            raise ValueError("weights must be finite.")
        if replacement and not any(w > 0 for w in self.weights):
            raise ValueError("weights must contain a positive entry to draw with replacement.")
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator: Generator | None = _as_generator(generator, "WeightedRandomSampler")

    @override
    def __iter__(self) -> Iterator[int]:
        """Yield ``num_samples`` indices proportional to the weights.

        With ``replacement=True`` each index comes from an inverse-CDF
        lookup (binary search over the running weight sums).  With
        ``replacement=False`` the indices with the largest exponential
        keys are taken.  See the class notes.
        """
        weights = self.weights
        if self.replacement:
            cumulative = list(itertools.accumulate(weights))
            total = cumulative[-1]
            # The last index a draw may land on: past it every weight is
            # zero, and a draw rounding up to ``total`` must not reach them.
            last = max(i for i, w in enumerate(weights) if w > 0)
            draws = cast(list[float], _uniform_doubles(self.num_samples, self.generator).tolist())
            for u in draws:
                yield min(bisect.bisect_right(cumulative, u * total), last)
            return
        positive = sum(1 for w in weights if w > 0)
        take = min(self.num_samples, positive)
        if take:
            # Efraimidis–Spirakis: the ``take`` largest keys log(u) / w are a
            # weighted draw without replacement, already in draw order.  A
            # zero weight's key is -inf, so it is never among them.
            w = _host_float64(weights)
            noise = _uniform_doubles(len(weights), self.generator)
            keys = noise.clamp(min=2.0**-60).log() / w
            _, order = lucid.topk(keys, take)
            yield from cast(list[int], order.tolist())
        # ``__len__`` promises ``num_samples`` items, and a DataLoader sizes
        # its epoch by it, so a pool with fewer positive weights is topped up
        # with zero-weight indices — last, in index order, as the weighted
        # draw has nothing to rank them by.
        if take < self.num_samples:
            zero = (i for i, wi in enumerate(weights) if wi <= 0)
            yield from itertools.islice(zero, self.num_samples - take)

    @override
    def __len__(self) -> int:
        """Return :attr:`num_samples`."""
        return self.num_samples


class BatchSampler(Sampler):
    """Group an inner sampler's indices into mini-batches.

    Wraps an existing :class:`Sampler` and packs its emitted indices into
    fixed-size lists. This is what :class:`DataLoader` uses internally to
    convert a per-sample index stream into per-batch index lists.

    Parameters
    ----------
    sampler : Sampler
        Underlying per-sample index sampler.
    batch_size : int
        Number of indices per yielded batch.
    drop_last : bool
        If ``True``, drop the trailing batch when the inner sampler's
        length is not divisible by ``batch_size``. If ``False``, yield
        the short final batch.

    Notes
    -----
    The inner sampler is consumed lazily — :class:`BatchSampler` simply
    accumulates ``batch_size`` indices then yields the list and starts a
    new batch.  ``drop_last=True`` produces ``floor(n / batch_size)``
    batches of uniform size (the usual choice for training, where
    short batches mess with BatchNorm statistics); ``drop_last=False``
    produces ``ceil(n / batch_size)`` batches with the final one
    possibly short (the usual choice for evaluation, where every sample
    must be visited).

    Examples
    --------
    >>> import lucid
    >>> from lucid.utils.data import BatchSampler, SequentialSampler, TensorDataset
    >>> my_dataset = TensorDataset(lucid.randn(10, 3))
    >>> inner = SequentialSampler(my_dataset)   # 10 items
    >>> bs = BatchSampler(inner, batch_size=4, drop_last=False)
    >>> [b for b in bs]
    [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    """

    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool) -> None:
        """Store the inner sampler and batching configuration.

        Parameters
        ----------
        sampler : Sampler
            Inner per-sample sampler.
        batch_size : int
            Target batch size.
        drop_last : bool
            Whether to discard the final short batch.

        Raises
        ------
        ValueError
            If ``batch_size`` is not a positive integer (a ``bool`` is not).
        TypeError
            If ``drop_last`` is not a ``bool``.
        """
        self.sampler = sampler
        # Otherwise the first ``len()`` divides by it and the caller gets a
        # ``ZeroDivisionError`` from inside the sampler — or ``True`` quietly
        # becomes a batch size of 1.
        _check_positive_int("batch_size", batch_size)
        _check_bool("drop_last", drop_last)
        self.batch_size = batch_size
        self.drop_last = drop_last

    @override
    def __iter__(self) -> Iterator[list[int]]:  # type: ignore[override]
        """Yield successive batches of indices.

        Yields
        ------
        list of int
            Lists of length ``batch_size``, except possibly the last
            (which is dropped if ``drop_last`` is ``True``).
        """
        batch: list[int] = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    @override
    def __len__(self) -> int:
        """Return the number of batches produced per epoch.

        Computed as ``floor(n / batch_size)`` when ``drop_last`` is
        ``True``, else ``ceil(n / batch_size)``.
        """
        n = len(self.sampler)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size


class DistributedSampler(Sampler):
    """Subset-and-shuffle sampler for distributed training.

    Lucid is single-process, single-machine — there is no real distributed
    backend to coordinate with — but ``DistributedSampler`` is part of the
    standard ``DataLoader`` surface and user code routinely instantiates it
    even in single-rank contexts (e.g. ``num_replicas=1, rank=0``).  This
    implementation supports exactly that: it partitions the dataset into
    ``num_replicas`` slabs and yields the indices belonging to ``rank``.
    With the default ``num_replicas=1`` it degenerates to a plain
    sequential / random sampler that respects ``shuffle`` and ``seed``.

    A multi-process backend would require a process group + collective
    communication; the surface stays compatible should that land later.

    Notes
    -----
    The index range ``range(len(dataset))`` is partitioned into
    ``num_replicas`` interleaved slabs — rank ``r`` receives every
    ``num_replicas``-th index starting at ``r``.  Per-epoch shuffles
    are driven by ``random.Random(seed + epoch)``, so every replica
    sees a different slab while remaining globally deterministic.
    Call :meth:`set_epoch` once per epoch to rotate the shuffle —
    forgetting to do so produces identical orderings each pass.

    Examples
    --------
    >>> import lucid
    >>> from lucid.utils.data import DistributedSampler, TensorDataset
    >>> my_dataset = TensorDataset(lucid.randn(10, 3))
    >>> sampler = DistributedSampler(my_dataset, num_replicas=4, rank=2)
    >>> num_epochs = 2
    >>> for epoch in range(num_epochs):
    ...     sampler.set_epoch(epoch)
    ...     for idx in sampler:
    ...         x = my_dataset[idx]
    >>> len(sampler)                      # ceil(10 / 4) indices for this rank
    3
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        """Configure the distributed sampler.

        Parameters
        ----------
        dataset : Dataset
            Dataset whose ``__len__`` defines the index range.
        num_replicas : int, optional
            Number of participating replicas (default ``1`` — degenerate
            single-process case).
        rank : int, optional
            Replica id in ``[0, num_replicas)``.
        shuffle : bool, optional
            If ``True``, indices are shuffled by ``random.Random(seed +
            epoch)`` before slabbing.
        seed : int, optional
            Base seed for the shuffling RNG. Combined with
            :meth:`set_epoch` for deterministic per-epoch shuffles.
        drop_last : bool, optional
            If ``True``, drop the trailing remainder so every replica
            sees the same number of samples without padding. If
            ``False``, wrap-pad the index list.

        Raises
        ------
        ValueError
            If ``num_replicas < 1`` or ``rank`` is out of range.
        """
        if num_replicas < 1:
            raise ValueError(f"num_replicas must be >= 1, got {num_replicas}")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank {rank} is out of range for num_replicas={num_replicas}")
        self.dataset: Dataset = dataset
        self.num_replicas: int = num_replicas
        self.rank: int = rank
        self.shuffle: bool = shuffle
        self.seed: int = seed
        self.drop_last: bool = drop_last
        self.epoch: int = 0

        # Split the index range into ``num_replicas`` evenly-sized slabs.
        # With ``drop_last=False`` we wrap-pad so every replica sees the
        # same number of indices; with ``drop_last=True`` we round down.
        n: int = len(dataset)
        if drop_last:
            self.num_samples: int = n // num_replicas
            self.total_size: int = self.num_samples * num_replicas
        else:
            self.num_samples = (n + num_replicas - 1) // num_replicas
            self.total_size = self.num_samples * num_replicas

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch number — affects the shuffling RNG seed."""
        self.epoch = int(epoch)

    @override
    def __iter__(self) -> Iterator[int]:
        """Yield this replica's slab of indices for the current epoch.

        Indices are optionally shuffled with seed ``self.seed + self.epoch``,
        then either truncated to ``total_size`` (``drop_last=True``) or
        wrap-padded (``drop_last=False``). The replica picks every
        ``num_replicas``-th index starting at ``rank``.
        """
        n: int = len(self.dataset)
        indices: list[int] = list(range(n))
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(indices)
        if self.drop_last:
            indices = indices[: self.total_size]
        else:
            # Wrap-pad so the slab division is even.
            padding: int = self.total_size - n
            if padding > 0:
                indices += indices[:padding]
        # Slab pick: take every ``num_replicas``-th index starting at ``rank``.
        return iter(indices[self.rank : self.total_size : self.num_replicas])

    @override
    def __len__(self) -> int:
        """Return per-replica sample count — same on every rank."""
        return self.num_samples


class RASampler(Sampler):
    r"""Repeated-augmentation sampler (Hoffer et al., 2020 — arXiv:1901.09335).

    Each dataset sample appears ``num_repeats`` times consecutively in
    the index stream, so that — when combined with a stochastic
    augmentation pipeline — the model sees several different augmented
    views of the same image inside each batch.  Empirically lifts
    ImageNet accuracy at no extra labelled-data cost.

    The sampler is also distributed-aware (single-process here, but
    sharding by ``rank`` is preserved for parity with the reference
    framework — see :class:`DistributedSampler`).  Per-epoch shuffling
    is deterministic given ``seed`` + the epoch counter
    (:meth:`set_epoch`).

    Parameters
    ----------
    dataset : Dataset
        Dataset whose ``__len__`` defines the unique-index range.
    num_replicas : int, optional, default=1
        Number of distributed replicas.  ``1`` is the single-process
        case (the default Lucid context).
    rank : int, optional, default=0
        Replica id in ``[0, num_replicas)``.
    shuffle : bool, optional, default=True
        If ``True``, the unique indices are shuffled before being
        repeated.  Identical seeds reproduce the same epoch order.
    seed : int, optional, default=0
        Base seed for the per-epoch RNG (combined with the epoch
        counter via :meth:`set_epoch`).
    num_repeats : int, optional, default=3
        How many consecutive copies of each index to emit.  The
        original paper recommends ``3`` for ImageNet.

    Notes
    -----
    Two derived sizes (matching the reference-framework convention):

    * ``num_samples = ceil(len(dataset) * num_repeats / num_replicas)``
      — number of indices this replica's slab contains *before*
      truncation.  Used as ``total_size = num_samples * num_replicas``
      for the full repeated-and-padded index list.
    * ``num_selected_samples = floor(len(dataset) / num_replicas)``
      — number of indices this replica actually *yields* per epoch.
      Iteration truncates the slab to this length so the effective
      epoch size (per rank) matches an un-repeated pass — repetition
      diversifies *within* the epoch rather than extending it.

    Examples
    --------
    >>> import lucid
    >>> from lucid.utils.data import RASampler, DataLoader, TensorDataset
    >>> dataset = TensorDataset(lucid.randn(12, 3), lucid.randint(0, 2, (12,)))
    >>> sampler = RASampler(dataset, num_repeats=3, shuffle=True, seed=0)
    >>> loader = DataLoader(dataset, batch_size=4, sampler=sampler)
    >>> for x, y in loader:                      # each batch may contain
    ...     pass                                 # ≤3 augmented views of
    ...     # ...train step...                   # the same underlying image
    >>> len(sampler)                             # an un-repeated pass per epoch
    12
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0,
        num_repeats: int = 3,
    ) -> None:
        if num_replicas < 1:
            raise ValueError(f"num_replicas must be >= 1, got {num_replicas}")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")
        if num_repeats < 1:
            raise ValueError(f"num_repeats must be >= 1, got {num_repeats}")
        self.dataset: Dataset = dataset
        self.num_replicas: int = num_replicas
        self.rank: int = rank
        self.shuffle: bool = shuffle
        self.seed: int = seed
        self.num_repeats: int = num_repeats
        self.epoch: int = 0

        n: int = len(dataset)
        # Full repeated-and-padded index list size, split evenly across replicas.
        self.num_samples: int = int(math.ceil(n * num_repeats / num_replicas))
        self.total_size: int = self.num_samples * num_replicas
        # Effective per-epoch length per rank — matches an un-repeated pass.
        self.num_selected_samples: int = int(math.floor(n / num_replicas))

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch — affects the shuffling RNG seed."""
        self.epoch = int(epoch)

    @override
    def __iter__(self) -> Iterator[int]:
        n: int = len(self.dataset)
        indices: list[int] = list(range(n))
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(indices)
        # Repeat each unique index ``num_repeats`` times in-place — the
        # consecutive repeats are the whole point.
        repeated: list[int] = [idx for idx in indices for _ in range(self.num_repeats)]
        # Wrap-pad to ``total_size`` so the replica slabs divide evenly.
        if len(repeated) < self.total_size:
            repeated = repeated + repeated[: self.total_size - len(repeated)]
        else:
            repeated = repeated[: self.total_size]
        # Replica's slab.
        slab = repeated[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        # Truncate so the epoch size per rank matches an un-repeated pass.
        return iter(slab[: self.num_selected_samples])

    @override
    def __len__(self) -> int:
        """Return per-replica yield count — ``floor(len(dataset) / num_replicas)``."""
        return self.num_selected_samples
