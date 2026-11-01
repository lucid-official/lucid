"""Datasets the DataLoader worker tests hand to ``spawn``-ed processes.

A worker unpickles its dataset by importing the class's module, so these
live in an importable module of their own rather than inside the test
file (which would drag pytest into every worker).
"""

import os
import random
import time
from typing import Iterator

import lucid
from lucid.utils.data import Dataset, IterableDataset, get_worker_info


def as_is(sample: object) -> object:
    """A ``collate_fn`` that hands samples through untouched (and pickles)."""
    return sample


class Indices(Dataset):
    """``dataset[i] == i`` — for checking which samples came out."""

    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> int:
        return index


class Draws(Dataset):
    """Each sample is a draw from every RNG a transform might use, plus the
    id of the worker that made it."""

    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> tuple[float, float, int]:
        info = get_worker_info()
        worker = -1 if info is None else info.id
        return (float(lucid.rand(1).item()), random.random(), worker)


class Info(Dataset):
    """Each sample is what ``get_worker_info`` reported for it."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.offset = 0

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> tuple[int, int, int, int]:
        info = get_worker_info()
        assert info is not None and info.dataset is self
        return (info.id, info.num_workers, info.seed, self.offset)


def offset_by_worker(worker_id: int) -> None:
    """A ``worker_init_fn`` that leaves a mark on its worker's dataset copy."""
    info = get_worker_info()
    assert info is not None and info.id == worker_id
    setattr(info.dataset, "offset", 1000 + worker_id)


def failing_init(worker_id: int) -> None:
    raise ValueError(f"init failed in {worker_id}")


class Raises(Dataset):
    """Raises ``KeyError`` for one index, returns the index otherwise."""

    def __init__(self, n: int, bad: int) -> None:
        self.n = n
        self.bad = bad

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> int:
        if index == self.bad:
            raise KeyError(f"no sample {index}")
        return index


class StopsAt(Raises):
    """A map-style ``__getitem__`` that leaks ``StopIteration`` for one index."""

    def __getitem__(self, index: int) -> int:
        if index == self.bad:
            raise StopIteration
        return index


class Unpicklable(Dataset):
    """Its samples cannot be sent back from a worker."""

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> object:
        return lambda: index


class Dies(Dataset):
    """The worker process exits outright while fetching one index."""

    def __init__(self, n: int, bad: int) -> None:
        self.n = n
        self.bad = bad

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> int:
        if index == self.bad:
            os._exit(3)
        return index


class Slow(Dataset):
    """Takes ``delay`` seconds per sample."""

    def __init__(self, n: int, delay: float) -> None:
        self.n = n
        self.delay = delay

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> int:
        time.sleep(self.delay)
        return index


class Pids(Dataset):
    """Each sample is the pid of the process that fetched it."""

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> int:
        return os.getpid()


class Heavy(Dataset):
    """4 MiB per sample: a few prefetched results fill the result pipe."""

    def __len__(self) -> int:
        return 16

    def __getitem__(self, index: int) -> bytes:
        return bytes(4 << 20)


class Stream(IterableDataset):
    """``range(n)``, sharded across workers when ``shard`` is set."""

    def __init__(self, n: int, shard: bool) -> None:
        self.n = n
        self.shard = shard

    def __iter__(self) -> Iterator[int]:  # type: ignore[override]
        info = get_worker_info()
        if info is None or not self.shard:
            yield from range(self.n)
        else:
            yield from range(info.id, self.n, info.num_workers)


class FirstSlow(Dataset):
    """``dataset[i] == i``, except that sample 0 takes ``delay`` seconds."""

    def __init__(self, n: int, delay: float) -> None:
        self.n = n
        self.delay = delay

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> int:
        if index == 0:
            time.sleep(self.delay)
        return index
