"""DistributedSampler pads by cycling, so every rank gets what ``len()`` says (LCD-802).

With ``drop_last=False`` and a dataset smaller than ``num_replicas``, padding
by ``indices[:padding]`` alone left the last ranks empty while ``len()``
reported one sample.  The shards are checked against the reference over the
whole (dataset size, world size, shuffle, drop_last) grid.
"""

from collections import Counter
from types import ModuleType

import pytest

from lucid.utils.data import DistributedSampler, RASampler
from lucid.utils.data.sampler import _fit_to_total_size

_SIZES = range(1, 11)
_WORLDS = range(1, 9)


def _ours(n, world, rank, **kw):
    return DistributedSampler(list(range(n)), num_replicas=world, rank=rank, **kw)


@pytest.mark.parametrize("drop_last", [False, True])
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("world", _WORLDS)
@pytest.mark.parametrize("n", _SIZES)
def test_every_rank_yields_exactly_len(n, world, shuffle, drop_last):
    shards = [list(_ours(n, world, r, shuffle=shuffle, drop_last=drop_last)) for r in range(world)]
    for r, shard in enumerate(shards):
        assert len(shard) == len(_ours(n, world, r, shuffle=shuffle, drop_last=drop_last))
    if drop_last:
        assert len({len(s) for s in shards}) == 1
        seen = Counter(i for s in shards for i in s)
        assert set(seen.values()) <= {1}
    else:
        assert all(len(s) >= 1 for s in shards)
        assert {i for s in shards for i in s} == set(range(n))
        # padding repeats an index only as often as the cycle needs
        total = sum(len(s) for s in shards)
        assert max(Counter(i for s in shards for i in s).values()) <= -(-total // n)


@pytest.mark.parity
@pytest.mark.parametrize("drop_last", [False, True])
@pytest.mark.parametrize("world", _WORLDS)
@pytest.mark.parametrize("n", _SIZES)
def test_unshuffled_shards_match_the_reference(ref: ModuleType, n, world, drop_last):
    for r in range(world):
        theirs = ref.utils.data.distributed.DistributedSampler(
            list(range(n)), num_replicas=world, rank=r, shuffle=False, drop_last=drop_last
        )
        assert list(_ours(n, world, r, shuffle=False, drop_last=drop_last)) == list(theirs)


@pytest.mark.parity
@pytest.mark.parametrize("drop_last", [False, True])
@pytest.mark.parametrize("world", _WORLDS)
@pytest.mark.parametrize("n", _SIZES)
def test_shuffled_shard_sizes_match_the_reference(ref: ModuleType, n, world, drop_last):
    for r in range(world):
        theirs = ref.utils.data.distributed.DistributedSampler(
            list(range(n)), num_replicas=world, rank=r, shuffle=True, drop_last=drop_last
        )
        assert len(list(_ours(n, world, r, shuffle=True, drop_last=drop_last))) == len(list(theirs))


def test_the_reported_example():
    shards = [list(_ours(3, 8, r, shuffle=False)) for r in range(8)]
    assert shards == [[0], [1], [2], [0], [1], [2], [0], [1]]


@pytest.mark.parametrize("have, total", [(3, 8), (1, 8), (5, 5), (8, 3), (4, 0), (7, 8)])
def test_fit_to_total_size_always_reaches_the_total(have, total):
    out = _fit_to_total_size(list(range(have)), total)
    assert len(out) == total
    assert out[: min(have, total)] == list(range(min(have, total)))


def test_fit_to_total_size_refuses_to_invent_indices():
    with pytest.raises(RuntimeError, match="empty"):
        _fit_to_total_size([], 4)


@pytest.mark.parametrize("world", _WORLDS)
@pytest.mark.parametrize("n", _SIZES)
def test_ra_sampler_shards_match_len(n, world):
    for r in range(world):
        sampler = RASampler(list(range(n)), num_replicas=world, rank=r, shuffle=False)
        assert len(list(sampler)) == len(sampler)
