"""The samplers, on which indices they actually produce.

``utils/data/sampler.py`` sat at 70.1%.  A sampler decides what a run
sees, and it fails in the one direction nothing downstream can detect:
the batches have the right shape and the right count either way, and a
model trained on a third of its data twice over still converges.

So the assertions are on the multiset of indices — every one, exactly
once, or exactly as many times as asked — rather than on how many came
out.
"""

import collections

import numpy as np
import pytest

import lucid
from lucid.utils.data import (
    BatchSampler,
    DistributedSampler,
    RandomSampler,
    Sampler,
    SequentialSampler,
    SubsetRandomSampler,
    TensorDataset,
    WeightedRandomSampler,
)


def _dataset(n):
    return TensorDataset(lucid.tensor(np.arange(n, dtype=np.float64)))


# ── the deterministic ones ────────────────────────────────────────────────────


def test_sequential_is_the_range():
    assert list(SequentialSampler(_dataset(5))) == [0, 1, 2, 3, 4]
    assert len(SequentialSampler(_dataset(5))) == 5


def test_subset_random_stays_inside_its_subset():
    indices = [7, 2, 5]
    drawn = list(SubsetRandomSampler(indices))
    assert sorted(drawn) == sorted(indices)
    assert len(SubsetRandomSampler(indices)) == 3


# ── random ────────────────────────────────────────────────────────────────────


def test_random_without_replacement_is_a_permutation():
    """Not merely the right length: drawing with replacement would also
    yield twenty indices and lose a third of the data to duplicates."""
    lucid.manual_seed(0)
    assert sorted(RandomSampler(_dataset(20))) == list(range(20))


def test_random_reshuffles_on_each_pass():
    lucid.manual_seed(0)
    sampler = RandomSampler(_dataset(50))
    assert list(sampler) != list(sampler)


def test_random_with_replacement_honours_num_samples():
    lucid.manual_seed(0)
    drawn = list(RandomSampler(_dataset(5), replacement=True, num_samples=11))
    assert len(drawn) == 11
    assert all(0 <= i < 5 for i in drawn)


def test_random_length_follows_num_samples():
    assert len(RandomSampler(_dataset(7))) == 7
    assert len(RandomSampler(_dataset(7), replacement=True, num_samples=3)) == 3


def test_random_refuses_a_non_positive_num_samples():
    """It used to surface at the first ``len()`` as ``__len__() should
    return >= 0`` — an error about the protocol rather than about the
    argument that caused it."""
    with pytest.raises(ValueError, match="num_samples"):
        RandomSampler(_dataset(4), replacement=True, num_samples=-1)
    with pytest.raises(ValueError, match="num_samples"):
        RandomSampler(_dataset(4), replacement=True, num_samples=0)


# ── weighted ──────────────────────────────────────────────────────────────────


def test_a_zero_weight_is_never_drawn():
    lucid.manual_seed(0)
    drawn = list(WeightedRandomSampler([0.0, 0.0, 1.0, 0.0], 50, replacement=True))
    assert set(drawn) == {2}


def test_the_weights_set_the_proportions():
    lucid.manual_seed(0)
    drawn = list(WeightedRandomSampler([1.0, 3.0], 8000, replacement=True))
    assert abs(drawn.count(1) / len(drawn) - 0.75) < 0.03


def test_without_replacement_really_is_without_replacement():
    """It was not.

    The branch called ``random.choices``, which samples *with*
    replacement, under a comment claiming the opposite — five draws from
    two items returned ``[0, 1, 1, 0, 0]``.  A class-balancing sampler
    configured this way silently oversamples exactly the classes it was
    asked to visit once, and the epoch still has the length it should.
    """
    for seed in range(8):
        drawn = list(
            WeightedRandomSampler([1.0, 2.0, 3.0, 4.0], 4, replacement=False, generator=seed)
        )
        assert sorted(drawn) == [0, 1, 2, 3], drawn


def test_a_partial_draw_without_replacement_is_still_unique():
    for seed in range(8):
        drawn = list(
            WeightedRandomSampler([1.0, 1.0, 1.0, 1.0, 1.0], 3, replacement=False, generator=seed)
        )
        assert len(drawn) == 3
        assert len(set(drawn)) == 3


def test_without_replacement_still_lets_the_weights_decide_the_order():
    """Uniqueness must not be bought by ignoring the weights — a heavily
    weighted index should almost always come out first."""
    first = collections.Counter(
        next(iter(WeightedRandomSampler([1.0, 100.0], 2, replacement=False, generator=s)))
        for s in range(400)
    )
    assert first[1] > 350


def test_the_generator_makes_the_draw_reproducible():
    """The without-replacement branch reached for the module-level
    ``random`` rather than the seeded one, so ``generator`` did nothing
    there and an epoch could not be replayed."""
    weights = [1.0, 2.0, 3.0, 4.0]
    same = [
        list(WeightedRandomSampler(weights, 4, replacement=False, generator=7)) for _ in range(2)
    ]
    other = list(WeightedRandomSampler(weights, 4, replacement=False, generator=9))
    assert same[0] == same[1]
    assert same[0] != other


def test_weighted_length_agrees_with_what_it_yields():
    sampler = WeightedRandomSampler([1.0] * 4, 7, replacement=True)
    assert len(sampler) == 7 == len(list(sampler))


def test_asking_for_more_than_there_are_without_replacement_is_refused():
    """There are not that many distinct indices to give.  Refused rather
    than quietly repeating, which is what it did."""
    with pytest.raises(ValueError, match="without replacement"):
        WeightedRandomSampler([1.0, 1.0], 5, replacement=False)


def test_weighted_refuses_a_negative_weight():
    with pytest.raises(ValueError, match="non-negative"):
        WeightedRandomSampler([1.0, -1.0], 2)


def test_weighted_refuses_a_non_positive_num_samples():
    with pytest.raises(ValueError, match="num_samples"):
        WeightedRandomSampler([1.0, 1.0], 0)


def test_a_draw_stops_when_every_remaining_weight_is_zero():
    """Falling back to a uniform draw over the items the caller weighted
    out would be worse than a short epoch."""
    drawn = list(WeightedRandomSampler([1.0, 0.0, 0.0], 3, replacement=False))
    assert drawn == [0]


# ── batching ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("drop_last,sizes", [(False, [3, 3, 3, 1]), (True, [3, 3, 3])])
def test_batch_sampler_partitions_the_indices(drop_last, sizes):
    batches = list(BatchSampler(SequentialSampler(_dataset(10)), 3, drop_last))
    assert [len(b) for b in batches] == sizes
    flat = [i for batch in batches for i in batch]
    assert flat == list(range(sum(sizes)))


@pytest.mark.parametrize("drop_last,count", [(False, 4), (True, 3)])
def test_batch_sampler_length_agrees_with_what_it_yields(drop_last, count):
    sampler = BatchSampler(SequentialSampler(_dataset(10)), 3, drop_last)
    assert len(sampler) == count == len(list(sampler))


def test_batch_sampler_refuses_a_non_positive_batch_size():
    """The first ``len()`` divided by it, so the caller got a
    ``ZeroDivisionError`` from inside the sampler."""
    with pytest.raises(ValueError, match="batch_size"):
        BatchSampler(SequentialSampler(_dataset(4)), 0, False)
    with pytest.raises(ValueError, match="batch_size"):
        BatchSampler(SequentialSampler(_dataset(4)), -1, False)


def test_a_batch_larger_than_the_dataset_is_one_short_batch():
    assert [len(b) for b in BatchSampler(SequentialSampler(_dataset(3)), 10, False)] == [3]
    assert list(BatchSampler(SequentialSampler(_dataset(3)), 10, True)) == []


# ── extension ─────────────────────────────────────────────────────────────────


def test_a_user_subclass_composes_with_the_batch_sampler():
    """``Sampler`` is a documented extension point; the contract is
    ``__iter__`` and ``__len__`` and nothing more."""

    class EveryOther(Sampler):
        def __init__(self, count):
            self.count = count

        def __iter__(self):
            return iter(range(0, self.count, 2))

        def __len__(self):
            return (self.count + 1) // 2

    assert list(EveryOther(7)) == [0, 2, 4, 6]
    assert [len(b) for b in BatchSampler(EveryOther(7), 3, False)] == [3, 1]


# ── sharding ──────────────────────────────────────────────────────────────────


def test_the_distributed_shards_partition_the_dataset():
    """Every index goes to exactly one rank.  An overlap silently trains
    on duplicates and a gap silently drops data, and both leave every
    rank with the same batch count."""
    shards = [sorted(DistributedSampler(_dataset(10), num_replicas=2, rank=r)) for r in (0, 1)]
    assert not (set(shards[0]) & set(shards[1]))
    assert sorted(shards[0] + shards[1]) == list(range(10))


def test_every_distributed_rank_gets_the_same_count():
    """Ranks step in lockstep, so an uneven split deadlocks or truncates."""
    lengths = {len(DistributedSampler(_dataset(10), num_replicas=3, rank=r)) for r in range(3)}
    assert len(lengths) == 1


def test_a_single_replica_is_the_whole_dataset():
    assert sorted(DistributedSampler(_dataset(6), num_replicas=1, rank=0)) == list(range(6))
