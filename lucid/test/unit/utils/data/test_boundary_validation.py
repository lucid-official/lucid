"""LCD-742: ``lucid.utils.data`` refuses bad arguments where they enter.

One defect class — a value the reference framework rejects at the
boundary slid past and came out later as a wrong answer — swept as
tables, with the reference as the oracle where it is installed
(``pytest -m parity``).
"""

from types import ModuleType

import pytest

from lucid.utils.data import (
    BatchSampler,
    ConcatDataset,
    DataLoader,
    RandomSampler,
    SequentialSampler,
    WeightedRandomSampler,
    default_collate,
    random_split,
)


class _Range:
    """A map-style dataset of ``range(n)`` — plain ints, no tensors needed."""

    def __init__(self, n: int, offset: int = 0) -> None:
        self.n, self.offset = n, offset

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> int:
        return self.offset + i


# ── WeightedRandomSampler: __iter__ always yields __len__ items ───────────────


@pytest.mark.parametrize(
    "weights, num_samples, positive",
    [
        ([0.0, 1.0, 1.0], 3, {1, 2}),
        ([1.0, 1.0, 0.0, 0.0], 4, {0, 1}),
        ([0.0, 0.0, 2.0], 2, {2}),
        ([1.0, 2.0, 3.0], 2, {0, 1, 2}),
        ([0.0, 1.0, 0.0, 1.0], 1, {1, 3}),
    ],
)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_unweighted_tail_keeps_the_epoch_length(weights, num_samples, positive, seed):
    sampler = WeightedRandomSampler(weights, num_samples, replacement=False, generator=seed)
    drawn = list(sampler)
    assert len(drawn) == len(sampler) == num_samples
    assert len(set(drawn)) == num_samples  # still without replacement
    head = drawn[: min(num_samples, len(positive))]
    tail = drawn[len(head) :]
    assert set(head) <= positive  # weighted draws first
    assert tail == sorted(tail) and all(weights[i] == 0 for i in tail)  # zero-weight last


def test_a_dataloader_over_a_zero_weight_pool_yields_len_batches():
    sampler = WeightedRandomSampler([0.0, 1.0, 1.0], 3, replacement=False, generator=0)
    loader = DataLoader(_Range(3), sampler=sampler, batch_size=1)
    assert len(loader) == len(list(loader)) == 3


# ── ConcatDataset ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "index, expected",
    [(0, 1), (2, 3), (3, 4), (4, 5), (-1, 5), (-2, 4), (-5, 1)],
)
def test_concat_resolves_indices_across_children(index, expected):
    data = ConcatDataset([_Range(3, 1), _Range(2, 4)])
    assert data[index] == expected


@pytest.mark.parametrize("index", [-6, -7, -100])
def test_concat_negative_index_below_minus_len_is_a_value_error(index):
    data = ConcatDataset([_Range(3), _Range(2)])
    with pytest.raises(ValueError, match="should not exceed dataset length"):
        data[index]


def test_concat_index_past_the_end_is_an_index_error():
    with pytest.raises(IndexError):
        ConcatDataset([_Range(3), _Range(2)])[5]


def test_concat_skips_an_empty_child():
    data = ConcatDataset([_Range(2), _Range(0), _Range(2, 10)])
    assert [data[i] for i in range(4)] == [0, 1, 10, 11]


def test_concat_refuses_an_empty_list():
    with pytest.raises(ValueError, match="at least one dataset"):
        ConcatDataset([])


# ── default_collate ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("leaf", [None, object(), {1, 2}])
def test_default_collate_refuses_an_unsupported_leaf(leaf):
    with pytest.raises(TypeError, match="cannot collate"):
        default_collate([leaf, leaf])


def test_default_collate_refuses_an_unsupported_leaf_nested_in_a_container():
    with pytest.raises(TypeError, match="NoneType"):
        default_collate([(1, None), (2, None)])


# ── sampler constructors ──────────────────────────────────────────────────────


@pytest.mark.parametrize("bad", ["yes", 1, 0, None])
def test_random_sampler_replacement_must_be_a_bool(bad):
    with pytest.raises(TypeError, match="replacement"):
        RandomSampler(_Range(4), replacement=bad)


@pytest.mark.parametrize("bad", ["yes", 1, None])
def test_weighted_sampler_replacement_must_be_a_bool(bad):
    with pytest.raises(TypeError, match="replacement"):
        WeightedRandomSampler([1.0, 1.0], 2, replacement=bad)


@pytest.mark.parametrize("bad", [True, 0, -1, 2.0, "3"])
def test_batch_sampler_batch_size_must_be_a_positive_int(bad):
    with pytest.raises(ValueError, match="batch_size"):
        BatchSampler(SequentialSampler(_Range(4)), bad, False)


@pytest.mark.parametrize("bad", [1, 0, "no", None])
def test_batch_sampler_drop_last_must_be_a_bool(bad):
    with pytest.raises(TypeError, match="drop_last"):
        BatchSampler(SequentialSampler(_Range(4)), 2, bad)


@pytest.mark.parametrize("bad", [True, 1.5, "2"])
def test_num_samples_must_be_a_positive_int(bad):
    with pytest.raises(ValueError, match="num_samples"):
        RandomSampler(_Range(4), num_samples=bad)
    with pytest.raises(ValueError, match="num_samples"):
        WeightedRandomSampler([1.0, 1.0], bad)


# ── random_split fractions ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "n, fractions, sizes",
    [
        (10, [0.33, 0.33, 0.34], [4, 3, 3]),
        (10, [0.5, 0.5], [5, 5]),
        (10, [0.8, 0.1, 0.1], [8, 1, 1]),
        (7, [0.25, 0.25, 0.5], [2, 2, 3]),
        (3, [0.4, 0.3, 0.3], [2, 1, 0]),
        (100, [0.7, 0.2, 0.1], [70, 20, 10]),
    ],
)
def test_random_split_gives_the_floor_remainder_round_robin(n, fractions, sizes):
    parts = random_split(_Range(n), fractions, generator=0)
    assert [len(p) for p in parts] == sizes
    assert sorted(i for p in parts for i in p.indices) == list(range(n))


@pytest.mark.parity
@pytest.mark.parametrize(
    "n, fractions",
    [(10, [0.33, 0.33, 0.34]), (11, [0.2, 0.3, 0.5]), (13, [0.1, 0.1, 0.1, 0.7]), (5, [0.5, 0.5])],
)
def test_random_split_sizes_match_the_reference(ref: ModuleType, n, fractions):
    ours = random_split(_Range(n), fractions, generator=0)
    theirs = ref.utils.data.random_split(list(range(n)), fractions)
    assert [len(p) for p in ours] == [len(p) for p in theirs]


@pytest.mark.parity
@pytest.mark.parametrize("index", [-6, -7, 5, 0, -5])
def test_concat_indexing_matches_the_reference(ref: ModuleType, index):
    theirs = ref.utils.data.ConcatDataset([[1, 2, 3], [4, 5]])
    ours = ConcatDataset([_Range(3, 1), _Range(2, 4)])
    try:
        want = theirs[index]
    except (ValueError, IndexError) as exc:
        with pytest.raises(type(exc)):
            ours[index]
    else:
        assert ours[index] == want


@pytest.mark.parity
def test_empty_concat_is_refused_by_both(ref: ModuleType):
    with pytest.raises(AssertionError):
        ref.utils.data.ConcatDataset([])
    with pytest.raises(ValueError):
        ConcatDataset([])


@pytest.mark.parity
def test_none_batch_is_refused_by_both(ref: ModuleType):
    with pytest.raises(TypeError):
        ref.utils.data.default_collate([None, None])
    with pytest.raises(TypeError):
        default_collate([None, None])
