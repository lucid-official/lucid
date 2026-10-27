"""DataLoader arguments ported training scripts pass, and what they mean.

Each of these failed on code written for the reference framework:

* ``pin_memory=True`` raised ``TypeError`` — the docstring said it was
  accepted, and nearly every training script passes it.
* A dataset returning ``y[i]`` from a NumPy label array produced a list of
  ``np.int64`` instead of a label tensor.
* ``batch_size=None`` (no automatic batching) raised ``TypeError``.
* ``WeightedRandomSampler`` scanned the whole CDF per draw: 1.7 s for one
  10k epoch.

Arguments are checked where they enter, with the exception type naming
what is wrong, rather than surfacing later from inside an iterator.
"""

import inspect
import multiprocessing
import time
import warnings

import numpy as np
import pytest

import lucid
from lucid.utils.data import (
    DataLoader,
    Dataset,
    IterableDataset,
    RandomSampler,
    WeightedRandomSampler,
    default_collate,
    default_convert,
)

from lucid.test.unit.utils.data._worker_datasets import Indices, Stream

# The released positional order — what a positional call binds to.
_RELEASED = [
    "dataset",
    "batch_size",
    "shuffle",
    "sampler",
    "batch_sampler",
    "num_workers",
    "collate_fn",
    "drop_last",
    "timeout",
    "worker_init_fn",
    "multiprocessing_context",
    "generator",
    "prefetch_factor",
    "persistent_workers",
]


class _Labels(Dataset):
    """``__getitem__`` indexes NumPy arrays, as most ported datasets do."""

    def __init__(self, dtype):
        self.x = np.arange(12, dtype=np.float32).reshape(6, 2)
        self.y = np.arange(6).astype(dtype)

    def __len__(self):
        return 6

    def __getitem__(self, i):
        return self.x[i], self.y[i]


# ── pin_memory ────────────────────────────────────────────────────────────────


def test_pin_memory_is_accepted_quietly_and_changes_nothing():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pinned = DataLoader(Indices(5), batch_size=2, pin_memory=True)
        batches = [b.tolist() for b in pinned]
    assert pinned.pin_memory is True
    assert batches == [[0, 1], [2, 3], [4]]


def test_the_released_positional_order_is_unchanged():
    params = list(inspect.signature(DataLoader).parameters.values())
    positional = [p.name for p in params if p.kind is p.POSITIONAL_OR_KEYWORD]
    assert positional == _RELEASED
    pin = inspect.signature(DataLoader).parameters["pin_memory"]
    assert pin.kind is pin.KEYWORD_ONLY and pin.default is False


# ── NumPy scalars ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "np_dtype,dtype",
    [
        (np.int64, lucid.int64),
        (np.int32, lucid.int32),
        (np.float32, lucid.float32),
        (np.float64, lucid.float64),
        (np.bool_, lucid.bool_),
    ],
)
def test_numpy_scalar_labels_collate_into_a_tensor_of_their_dtype(np_dtype, dtype):
    x, y = next(iter(DataLoader(_Labels(np_dtype), batch_size=3)))
    assert isinstance(y, lucid.Tensor)
    assert y.dtype == dtype
    assert y.tolist() == np.arange(3).astype(np_dtype).tolist()
    assert x.shape == (3, 2)


def test_default_convert_makes_a_numpy_scalar_a_zero_d_tensor():
    out = default_convert(np.int64(7))
    assert isinstance(out, lucid.Tensor)
    assert (out.shape, out.dtype, out.item()) == ((), lucid.int64, 7)
    assert default_convert(np.float32(1.5)).dtype == lucid.float32


def test_a_numpy_string_array_is_left_as_it_is():
    batch = [np.array(["a"]), np.array(["b"])]
    assert default_collate(batch) is batch


# ── batch_size=None ───────────────────────────────────────────────────────────


def test_batch_size_none_yields_one_converted_sample_per_step():
    loader = DataLoader(Indices(4), batch_size=None)
    samples = list(loader)
    assert len(loader) == 4
    assert all(isinstance(s, lucid.Tensor) and s.shape == () for s in samples)
    assert [int(s) for s in samples] == [0, 1, 2, 3]
    assert loader.batch_sampler is None


def test_batch_size_none_honours_shuffle_and_a_custom_collate():
    lucid.manual_seed(0)
    seen = list(DataLoader(Indices(6), batch_size=None, shuffle=True, collate_fn=str))
    assert sorted(seen) == [str(i) for i in range(6)]


def test_batch_size_none_streams_an_iterable_dataset_sample_by_sample():
    assert [int(s) for s in DataLoader(Stream(3, shard=False), batch_size=None)] == [0, 1, 2]


def test_batch_size_none_with_drop_last_is_refused():
    with pytest.raises(ValueError, match="drop_last"):
        DataLoader(Indices(4), batch_size=None, drop_last=True)


# ── arguments are checked where they enter ───────────────────────────────────


@pytest.mark.parametrize("batch_size", [0, -1, 2.5, True])
def test_a_batch_size_that_is_not_a_positive_integer_is_refused(batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        DataLoader(Indices(4), batch_size=batch_size)


def test_an_iterable_loader_checks_its_batch_size_too():
    """No sampler validates it on this path; a 0 batched nothing, silently."""
    with pytest.raises(ValueError, match="batch_size"):
        DataLoader(Stream(4, shard=False), batch_size=0)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"num_workers": -1}, "num_workers"),
        ({"num_workers": 1.5}, "num_workers"),
        ({"timeout": -1.0}, "timeout"),
        ({"timeout": float("nan")}, "timeout"),
        ({"prefetch_factor": 0, "num_workers": 1}, "prefetch_factor"),
        ({"prefetch_factor": 2.5, "num_workers": 1}, "prefetch_factor"),
        # Options of the worker pool, given without one (the reference
        # refuses these too).
        ({"prefetch_factor": 2}, "prefetch_factor"),
        ({"multiprocessing_context": "spawn"}, "multiprocessing_context"),
        ({"persistent_workers": True}, "persistent_workers"),
        ({"multiprocessing_context": "no-such-method", "num_workers": 1}, "no-such-method"),
    ],
)
def test_a_bad_value_is_refused_at_construction(kwargs, match):
    with pytest.raises(ValueError, match=match):
        DataLoader(Indices(4), **kwargs)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"timeout": None}, "timeout"),
        ({"timeout": "5"}, "timeout"),
        ({"collate_fn": 3}, "collate_fn"),
        ({"worker_init_fn": "init", "num_workers": 1}, "worker_init_fn"),
        ({"multiprocessing_context": object(), "num_workers": 1}, "multiprocessing_context"),
        ({"generator": object(), "shuffle": True}, "generator"),
        # ``generator=True`` reads as a switch; taking it as the seed 1
        # would turn a typo into a fixed shuffle.
        ({"generator": True, "shuffle": True}, "generator"),
    ],
)
def test_a_wrong_type_is_refused_at_construction(kwargs, match):
    with pytest.raises(TypeError, match=match):
        DataLoader(Indices(4), **kwargs)


def test_a_multiprocessing_context_object_is_accepted():
    context = multiprocessing.get_context("spawn")
    loader = DataLoader(Indices(4), num_workers=1, multiprocessing_context=context)
    assert loader.multiprocessing_context is context


def test_stop_iteration_from_collate_fn_is_an_error_not_an_end():
    def collate(batch):
        raise StopIteration

    for dataset in (Indices(4), Stream(4, shard=False)):
        with pytest.raises(RuntimeError, match="raised StopIteration"):
            list(DataLoader(dataset, batch_size=2, collate_fn=collate))


# ── samplers ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("replacement", [True, False])
def test_weighted_sampling_is_not_quadratic(replacement):
    """It was 1.7 s (with) and 1.2 s (without replacement) per 10k epoch."""
    sampler = WeightedRandomSampler([1.0] * 10_000, 10_000, replacement=replacement)
    start = time.perf_counter()
    drawn = list(sampler)
    assert time.perf_counter() - start < 0.5
    assert len(drawn) == 10_000
    if not replacement:
        assert sorted(drawn) == list(range(10_000))


def test_weighted_weights_may_be_a_tensor():
    sampler = WeightedRandomSampler(lucid.tensor([0.0, 1.0, 0.0]), 5)
    assert list(sampler) == [1] * 5


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_weighted_refuses_a_weight_that_is_not_finite(bad):
    with pytest.raises(ValueError, match="finite"):
        WeightedRandomSampler([1.0, bad], 2)


def test_weighted_refuses_all_zero_weights_with_replacement():
    with pytest.raises(ValueError, match="positive"):
        WeightedRandomSampler([0.0, 0.0], 2)


def test_random_sampler_chains_permutations_past_the_dataset_size():
    """``len`` said ``num_samples``; the sampler stopped at ``n``."""
    sampler = RandomSampler(Indices(4), num_samples=10)
    drawn = list(sampler)
    assert len(drawn) == len(sampler) == 10
    assert sorted(drawn[:4]) == sorted(drawn[4:8]) == [0, 1, 2, 3]


def test_random_sampler_over_an_empty_dataset():
    """Nothing to draw: an empty epoch by default, an error when a positive
    ``num_samples`` was asked for (``len`` would otherwise be a lie)."""
    assert list(RandomSampler(Indices(0))) == []
    for replacement in (True, False):
        sampler = RandomSampler(Indices(0), replacement=replacement, num_samples=3)
        with pytest.raises(ValueError, match="empty"):
            list(sampler)


class _NoLen(IterableDataset):
    def __iter__(self):
        yield from range(3)


def test_an_iterable_loader_still_refuses_a_length():
    with pytest.raises(TypeError, match="no\\s+length"):
        len(DataLoader(_NoLen(), batch_size=None))
