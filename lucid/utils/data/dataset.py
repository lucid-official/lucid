"""
Dataset base classes and implementations.
"""

from typing import Iterator, TYPE_CHECKING, override

from lucid._factories.random import Generator
from lucid.utils.data._rng import _as_generator, _permutation

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


class Dataset:
    """Abstract base class for map-style datasets.

    Subclasses must implement :meth:`__len__` (total number of samples) and
    :meth:`__getitem__` (sample retrieval by integer index). Together these
    two methods constitute the *map-style* dataset protocol used by
    :class:`~lucid.utils.data.DataLoader`.

    Notes
    -----
    Map-style datasets are random-access: any index ``0 <= i < len(ds)``
    can be fetched at any time, which is what lets samplers (such as
    :class:`~lucid.utils.data.RandomSampler` or
    :class:`~lucid.utils.data.BatchSampler`) drive iteration. If the data
    source does not support random access (e.g., a streaming log), use
    :class:`IterableDataset` instead.

    Examples
    --------
    >>> class Squares(Dataset):
    ...     def __init__(self, n): self.n = n
    ...     def __len__(self): return self.n
    ...     def __getitem__(self, i): return i * i
    >>> ds = Squares(5)
    >>> ds[3]
    9
    """

    def __getitem__(self, index: int) -> Tensor | tuple[Tensor, ...]:
        """Retrieve a single sample by integer index.

        Parameters
        ----------
        index : int
            Position of the sample to return. Implementations are expected
            to support ``0 <= index < len(self)``; negative indexing is not
            part of the protocol.

        Returns
        -------
        Tensor or tuple of Tensor
            The sample at the given index. Multi-output datasets typically
            return a tuple such as ``(input, target)``.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns
        -------
        int
            Total number of samples available via :meth:`__getitem__`.
        """
        raise NotImplementedError

    def __add__(self, other: Dataset) -> ConcatDataset:
        """Concatenate this dataset with another via the ``+`` operator.

        Parameters
        ----------
        other : Dataset
            Dataset whose samples should follow those of ``self``.

        Returns
        -------
        ConcatDataset
            A :class:`ConcatDataset` wrapping ``[self, other]``.
        """
        return ConcatDataset([self, other])


class IterableDataset:
    """Base class for iterable-style (stream) datasets.

    Subclasses must implement :meth:`__iter__`, yielding samples one at a
    time. Unlike :class:`Dataset`, iterable datasets do not support random
    access via integer indices and therefore cannot be used with samplers;
    :class:`~lucid.utils.data.DataLoader` consumes them sequentially.

    Notes
    -----
    Use :class:`IterableDataset` when:

    * The data source is a stream (network socket, generator, log tail).
    * The dataset size is unknown a priori.
    * Random access is prohibitively expensive.

    Otherwise prefer :class:`Dataset` — it composes with shuffling and
    distributed sampling, which is impossible for iterable streams.

    Examples
    --------
    >>> class CountUp(IterableDataset):
    ...     def __init__(self, n): self.n = n
    ...     def __iter__(self):
    ...         for i in range(self.n):
    ...             yield i
    """

    def __iter__(self) -> Iterator[Tensor | tuple[Tensor, ...]]:
        """Yield samples one at a time.

        Returns
        -------
        Iterator
            Iterator producing individual samples. The iterator should
            terminate (raise ``StopIteration``) when the stream is
            exhausted; infinite streams are permitted but require the
            consumer to externally bound iteration.
        """
        raise NotImplementedError

    def __add__(self, other: IterableDataset) -> IterableDataset:
        """Concatenation via ``+`` — not supported for iterable datasets.

        Use :class:`ChainDataset` to chain iterable datasets end-to-end.

        Raises
        ------
        NotImplementedError
            Always; iterable datasets cannot be concatenated through ``+``.
        """
        raise NotImplementedError("Concatenation of IterableDatasets is not supported.")


class TensorDataset(Dataset):
    """Dataset wrapping one or more Tensors, indexed along their first axis.

    Each sample is the tuple ``(t1[i], t2[i], ...)`` where ``t1, t2, ...``
    are the wrapped tensors. All tensors must agree in their first
    dimension (the sample axis); subsequent dimensions are independent.

    Parameters
    ----------
    *tensors : Tensor
        One or more tensors of identical leading-dimension size. The
        dataset length equals ``tensors[0].shape[0]``.

    Raises
    ------
    ValueError
        If no tensors are provided or the leading dimensions disagree.

    Examples
    --------
    >>> X = lucid.randn(100, 4)
    >>> y = lucid.randint(0, 3, (100,))
    >>> ds = TensorDataset(X, y)
    >>> x_i, y_i = ds[0]

    Notes
    -----
    All wrapped tensors must share the same length along axis 0 — that
    shared length defines :meth:`__len__`.  The underlying tensors are
    held *by reference* rather than copied, so any mutation visible on
    the source tensors is also visible through the dataset.  This keeps
    construction O(1) but means the caller is responsible for not
    invalidating the buffers (e.g. by resizing) during iteration.
    """

    def __init__(self, *tensors: Tensor) -> None:
        """Initialise the instance.  See the class docstring for parameter semantics."""
        if not tensors:
            raise ValueError("TensorDataset requires at least one tensor")
        n = tensors[0].shape[0]
        for t in tensors[1:]:
            if t.shape[0] != n:
                raise ValueError(
                    f"All tensors must have the same size in the first dimension: "
                    f"expected {n}, got {t.shape[0]}"
                )
        self.tensors = tensors

    @override
    def __getitem__(self, index: int) -> tuple[Tensor, ...]:
        """Return ``(t[index] for t in self.tensors)`` as a tuple.

        Parameters
        ----------
        index : int
            Sample index along the leading dimension.

        Returns
        -------
        tuple of Tensor
            One element per wrapped tensor, in registration order.
        """
        return tuple(t[index] for t in self.tensors)

    def __getitems__(self, indices: list[int]) -> tuple[Tensor, ...]:
        """Batched fetch — return one fancy-indexed tensor per wrapped tensor.

        This is the 3.2.0 vectorised fast path consumed by
        :class:`~lucid.utils.data.DataLoader`.  Unlike :meth:`__getitem__`,
        which returns a *single* sample tuple and forces the data loader
        to call this method ``batch_size`` times then run ``collate_fn``
        to stack them, ``__getitems__`` returns the **already-batched**
        tensors in one call — one ``t[indices]`` fancy-index per wrapped
        tensor.  The data loader treats the result as already-collated
        and bypasses ``collate_fn`` entirely (a documented short-circuit
        that's safe because the tensors share the same leading-dim
        contract as the per-sample path's stacked output).

        On LeNet-5/MNIST the per-sample path costs ~30 ms / batch in
        Python overhead (60 k calls + 1 stack); the batched path is
        ~1 ms / batch (1 fancy-index + 0 stack).

        Parameters
        ----------
        indices : list of int
            Sample indices to fetch.  Order is preserved.

        Returns
        -------
        tuple of Tensor
            One batched tensor per wrapped tensor, in registration order.
            Each output has shape ``(len(indices), *t.shape[1:])``.
        """
        # Convert the Python list of indices to a Lucid index tensor once,
        # then reuse it across every wrapped tensor.  Fancy indexing copies
        # into a fresh contiguous buffer — that's the batched output the
        # data loader yields directly.
        #
        # Per-tensor device placement: the index tensor must live on the
        # same device as the tensor being indexed, otherwise the engine's
        # ``index_select`` path raises ``bad_variant_access`` on the
        # CPU↔GPU storage mismatch.  We cache one index tensor per device
        # encountered in ``self.tensors`` so a mixed-device dataset still
        # works without re-uploading indices N times.
        from lucid._factories.converters import tensor as _tensor

        cpu_idx = _tensor(indices)
        # Single-device fast path (the common case)
        first_dev = self.tensors[0]._impl.device
        if all(t._impl.device == first_dev for t in self.tensors):
            idx = cpu_idx if first_dev.name == "CPU" else cpu_idx.to(device=first_dev)
            return tuple(t[idx] for t in self.tensors)
        # Mixed-device fallback: cache one index tensor per device
        idx_cache: dict[object, Tensor] = {}
        out: list[Tensor] = []
        for t in self.tensors:
            dev = t._impl.device
            cached = idx_cache.get(dev)
            if cached is None:
                cached = cpu_idx if dev.name == "CPU" else cpu_idx.to(device=dev)
                idx_cache[dev] = cached
            out.append(t[cached])
        return tuple(out)

    @override
    def __len__(self) -> int:
        """Return the leading-dimension size shared by all wrapped tensors."""
        return self.tensors[0].shape[0]


class ConcatDataset(Dataset):
    """Dataset formed by concatenating several map-style datasets end-to-end.

    Sample ``i`` is fetched from the first child whose cumulative length
    exceeds ``i``, with the relative index translated accordingly.

    Parameters
    ----------
    datasets : list of Dataset
        Child datasets, concatenated in order. Each child must implement
        :meth:`__len__` and :meth:`__getitem__`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.utils.data import ConcatDataset, TensorDataset
    >>> ds_a = TensorDataset(lucid.zeros(3, 2))
    >>> ds_b = TensorDataset(lucid.ones(4, 2))
    >>> ds_c = TensorDataset(lucid.ones(5, 2))
    >>> combined = ConcatDataset([ds_a, ds_b, ds_c])
    >>> len(combined) == len(ds_a) + len(ds_b) + len(ds_c)
    True

    Notes
    -----
    A list of cumulative length offsets is maintained internally at
    construction time so that index resolution reduces to a single
    bounded scan (and could be upgraded to a binary search for O(log n)
    lookup on large child lists).  Children are stored by reference;
    no per-sample copy is made.
    """

    def __init__(self, datasets: list[Dataset]) -> None:
        """Initialise the instance.  See the class docstring for parameter semantics."""
        self.datasets = list(datasets)
        self.cumulative_sizes = self._cumsum([len(d) for d in datasets])

    @staticmethod
    def _cumsum(sizes: list[int]) -> list[int]:
        result = []
        total = 0
        for s in sizes:
            total += s
            result.append(total)
        return result

    @override
    def __len__(self) -> int:
        """Return the total length — sum of all child dataset lengths."""
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    @override
    def __getitem__(self, idx: int) -> Tensor | tuple[Tensor, ...]:
        """Return the sample at the given global index.

        Parameters
        ----------
        idx : int
            Index into the concatenated dataset. Negative indices are
            translated relative to the total length.

        Returns
        -------
        Tensor or tuple of Tensor
            Sample fetched from the appropriate child dataset.
        """
        if idx < 0:
            idx = len(self) + idx
        dataset_idx = 0
        while (
            dataset_idx < len(self.cumulative_sizes) and idx >= self.cumulative_sizes[dataset_idx]
        ):
            dataset_idx += 1
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]


class Subset(Dataset):
    """View into a parent dataset restricted to a list of indices.

    Useful for train/val splits, k-fold cross-validation, or any time a
    contiguous subset of a larger dataset is needed without copying the
    underlying data.

    Parameters
    ----------
    dataset : Dataset
        Parent dataset to view into.
    indices : list of int
        Indices into ``dataset`` that compose this subset. Order is
        preserved; duplicates are allowed.

    Examples
    --------
    >>> import lucid
    >>> from lucid.utils.data import Subset, TensorDataset
    >>> X, y = lucid.randn(100, 4), lucid.zeros(100)
    >>> full = TensorDataset(X, y)
    >>> train = Subset(full, list(range(0, 80)))
    >>> val = Subset(full, list(range(80, 100)))
    >>> len(train), len(val)
    (80, 20)

    Notes
    -----
    A :class:`Subset` is purely an index-restricted *view* — the parent
    dataset is held by reference and no sample is copied.  Subsets may
    be nested freely (a :class:`Subset` of a :class:`Subset`) because
    the parent only needs to satisfy the map-style protocol.
    """

    def __init__(self, dataset: Dataset, indices: list[int]) -> None:
        """Initialise the instance.  See the class docstring for parameter semantics."""
        self.dataset = dataset
        self.indices = indices

    @override
    def __len__(self) -> int:
        """Return the number of indices in the subset."""
        return len(self.indices)

    @override
    def __getitem__(self, idx: int) -> Tensor | tuple[Tensor, ...]:
        """Return ``dataset[indices[idx]]``.

        Parameters
        ----------
        idx : int
            Position within the subset (not the parent dataset).

        Returns
        -------
        Tensor or tuple of Tensor
            Sample from the parent dataset at the remapped index.
        """
        return self.dataset[self.indices[idx]]


def random_split(
    dataset: Dataset, lengths: list[int] | list[float], generator: Generator | int | None = None
) -> list[Subset]:
    r"""Randomly split a dataset into non-overlapping :class:`Subset` views.

    Shuffles ``range(len(dataset))`` and slices it into chunks of the
    requested lengths, wrapping each slice in a :class:`Subset`.  The
    children do not copy the underlying samples — they hold the parent
    dataset by reference.

    Parameters
    ----------
    dataset : Dataset
        Source dataset to split.
    lengths : list of int or list of float
        Either absolute split sizes summing to ``len(dataset)``, or
        fractions in ``[0, 1]`` summing (approximately) to ``1.0``.  In
        the fractional case, rounding error is absorbed by the final
        split so the totals stay consistent.
    generator : lucid.Generator or int, optional
        Stream the permutation is drawn from.  ``None`` (default) draws
        from the default generator, so :func:`lucid.manual_seed`
        reproduces the split; an ``int`` seeds a private generator, so the
        same ``int`` always gives the same split.

    Returns
    -------
    list of Subset
        One :class:`Subset` per requested split, in registration order.

    Raises
    ------
    ValueError
        If fractional ``lengths`` do not sum to ``1.0`` (within ``1e-6``)
        or integer ``lengths`` do not sum to ``len(dataset)``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.utils.data import TensorDataset, random_split
    >>> X, y = lucid.randn(100, 4), lucid.zeros(100)
    >>> full = TensorDataset(X, y)
    >>> train, val, test = random_split(full, [0.8, 0.1, 0.1])
    >>> len(train), len(val), len(test)
    (80, 10, 10)

    Notes
    -----
    The split is permutation-based: ``range(len(dataset))`` is permuted
    once with :func:`lucid.randperm` and then sliced into the requested
    chunks.  Reproducibility is obtained by seeding the default generator
    via :func:`lucid.manual_seed`, or by passing an explicit
    ``generator``; the same generator state always yields the same
    partition.

    >>> lucid.manual_seed(0)
    >>> first = random_split(full, [50, 50])[0].indices
    >>> lucid.manual_seed(0)
    >>> first == random_split(full, [50, 50])[0].indices
    True
    """
    rng = _as_generator(generator, "random_split")
    n = len(dataset)

    if all(isinstance(x, float) for x in lengths):
        total = sum(lengths)
        if abs(total - 1.0) > 1e-6:
            raise ValueError("Fraction lengths must sum to 1.0")
        int_lengths = [round(x * n) for x in lengths]
        # Fix rounding error on last element
        int_lengths[-1] = n - sum(int_lengths[:-1])
        lengths = int_lengths

    lengths_int: list[int] = [int(x) for x in lengths]
    if sum(lengths_int) != n:
        raise ValueError(
            f"Sum of split lengths ({sum(lengths_int)}) must equal dataset length ({n})"
        )

    indices = _permutation(n, rng)

    result = []
    offset = 0
    for length in lengths_int:
        result.append(Subset(dataset, indices[offset : offset + length]))
        offset += length
    return result


class ChainDataset(IterableDataset):
    r"""Concatenate multiple :class:`IterableDataset` instances end-to-end.

    Iteration walks the chain in registration order: every element from
    the first child is yielded first, then every element from the second,
    and so on.  The iterable equivalent of :class:`ConcatDataset` for
    map-style datasets.

    Each child must be an :class:`IterableDataset`; map-style datasets
    must be iterated through a sampler-driven :class:`DataLoader`
    instead.

    Parameters
    ----------
    datasets : list of IterableDataset
        Iterable datasets to chain in order.  Each element must be an
        :class:`IterableDataset` instance — passing a :class:`Dataset`
        raises ``TypeError`` at construction time.

    Notes
    -----
    Unlike :class:`ConcatDataset`, no random access is possible: total
    length is unknown a priori and the chain may be infinite if any
    child is.  Useful for stitching together streaming shards
    (e.g. multiple files in a sharded log) into one logical stream.

    Examples
    --------
    >>> from lucid.utils.data import ChainDataset, IterableDataset
    >>> class CountUp(IterableDataset):
    ...     def __init__(self, n):
    ...         self.n = n
    ...     def __iter__(self):
    ...         yield from range(self.n)
    >>> a = CountUp(3)              # yields 0, 1, 2
    >>> b = CountUp(2)              # yields 0, 1
    >>> chain = ChainDataset([a, b])
    >>> list(chain)
    [0, 1, 2, 0, 1]
    """

    def __init__(self, datasets: list[IterableDataset]) -> None:
        """Build a chain over the given iterable datasets.

        Parameters
        ----------
        datasets : list of IterableDataset
            Iterable datasets to chain in order. Each must be an
            :class:`IterableDataset` instance.

        Raises
        ------
        TypeError
            If any element is not an :class:`IterableDataset`.
        """
        bad: list[type] = [type(d) for d in datasets if not isinstance(d, IterableDataset)]
        if bad:
            raise TypeError(f"ChainDataset requires IterableDataset children, got {bad}")
        self.datasets: list[IterableDataset] = list(datasets)

    @override
    def __iter__(self) -> Iterator[Tensor | tuple[Tensor, ...]]:
        """Iterate through each child dataset in registration order.

        Yields
        ------
        Tensor or tuple of Tensor
            Each sample produced by each child, in sequence.
        """
        for d in self.datasets:
            yield from d


class StackDataset(Dataset):
    r"""Bundle several map-style datasets so each index returns a stacked tuple.

    The bundled datasets must agree in length.  Item ``i`` is the tuple
    ``(d_1[i], d_2[i], \dots, d_K[i])`` for each child dataset — handy
    when paired modalities (image, caption, label) live in separate
    sources but share a common index.

    Accepts either positional children (``StackDataset(d1, d2)``) or
    keyword children (``StackDataset(image=d1, label=d2)``); the two
    forms are mutually exclusive.  Positional construction returns
    tuples; keyword construction returns dicts keyed by the supplied
    names.

    Parameters
    ----------
    *args : Dataset
        Positional child datasets.  Cannot be combined with ``**kwargs``.
    **kwargs : Dataset
        Keyword child datasets.  Cannot be combined with ``*args``.

    Notes
    -----
    Equivalent in spirit to a per-index ``zip`` across child datasets,
    but exposed as a :class:`Dataset` so it can drive a sampler-based
    :class:`DataLoader`.  All children must satisfy ``len(d_k) == n``
    for some shared ``n``; otherwise construction raises ``ValueError``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.utils.data import StackDataset, TensorDataset
    >>> X_img, y_lbl = lucid.randn(8, 3, 4, 4), lucid.randint(0, 10, (8,))
    >>> images = TensorDataset(X_img)
    >>> labels = TensorDataset(y_lbl)
    >>> ds = StackDataset(image=images, label=labels)
    >>> sample = ds[0]
    >>> sorted(sample.keys())
    ['image', 'label']
    """

    def __init__(self, *args: Dataset, **kwargs: Dataset) -> None:
        """Bundle child datasets either positionally or by keyword.

        Parameters
        ----------
        *args : Dataset
            Positional child datasets. The resulting samples are tuples.
        **kwargs : Dataset
            Keyword child datasets. The resulting samples are dicts keyed
            by the keyword names.

        Raises
        ------
        ValueError
            If both positional and keyword children are given, if no
            children are given, or if the children disagree in length.
        """
        if args and kwargs:
            raise ValueError(
                "StackDataset takes either positional or keyword child datasets, not both"
            )
        children: tuple[Dataset, ...] = args if args else tuple(kwargs.values())
        if not children:
            raise ValueError("StackDataset requires at least one child dataset")
        n: int = len(children[0])
        for child in children[1:]:
            if len(child) != n:
                raise ValueError(
                    f"StackDataset children must agree in length; " f"saw {n} then {len(child)}"
                )
        self.datasets: tuple[Dataset, ...] = children
        self._keys: tuple[str, ...] | None = tuple(kwargs.keys()) if kwargs else None
        self._n: int = n

    @override
    def __len__(self) -> int:
        """Return the common length shared by all bundled child datasets."""
        return self._n

    @override
    def __getitem__(self, idx: int) -> tuple[object, ...] | dict[str, object]:  # type: ignore[override]
        """Return one sample drawn from each child at index ``idx``.

        Parameters
        ----------
        idx : int
            Index into the bundled datasets.

        Returns
        -------
        tuple or dict
            A tuple of child samples if the dataset was built positionally,
            otherwise a dict keyed by the keyword names supplied at
            construction time.
        """
        items: tuple[object, ...] = tuple(d[idx] for d in self.datasets)
        if self._keys is not None:
            return dict(zip(self._keys, items, strict=True))
        return items
