"""The DataLoader, on the promises an epoch depends on.

``utils/data/dataloader.py`` sat at 50.2%.  A loader is the one
component whose mistakes are invisible by construction: it produces
tensors of the right shape and dtype whatever it does with the indices,
and a run that quietly saw two thirds of its data, or the same sample
three times, still trains and still converges — just to somewhere else.

So the checks here are on the set of samples that came out, not on the
shape of any one batch.
"""

import numpy as np
import pytest

import lucid
from lucid.utils.data import (
    BatchSampler,
    ChainDataset,
    DataLoader,
    IterableDataset,
    SequentialSampler,
    SubsetRandomSampler,
    TensorDataset,
    default_collate,
)


def _dataset(n=10):
    values = np.arange(n, dtype=np.float64)
    return TensorDataset(lucid.tensor(values), lucid.tensor(values * 10))


def _seen(loader):
    """Every sample that came out, in order."""
    out = []
    for batch in loader:
        out.extend(np.asarray(batch[0].numpy()).ravel().tolist())
    return out


def _sizes(loader):
    return [int(np.asarray(b[0].numpy()).size) for b in loader]


class _Counting(IterableDataset):
    def __init__(self, start, stop):
        self.start, self.stop = start, stop

    def __iter__(self):
        return iter(range(self.start, self.stop))


# ── the set of samples ────────────────────────────────────────────────────────


def test_every_sample_comes_out_exactly_once():
    """The property a training run rests on, and the one a shape check
    cannot see."""
    assert sorted(_seen(DataLoader(_dataset(10), batch_size=3))) == list(range(10))


def test_without_shuffle_the_order_is_the_dataset_s():
    assert _seen(DataLoader(_dataset(10), batch_size=3)) == list(range(10))


def test_the_last_batch_is_short_rather_than_dropped():
    assert _sizes(DataLoader(_dataset(10), batch_size=3)) == [3, 3, 3, 1]


def test_drop_last_drops_it():
    assert _sizes(DataLoader(_dataset(10), batch_size=3, drop_last=True)) == [3, 3, 3]


@pytest.mark.parametrize(
    "batch_size,drop_last,batches",
    [(3, False, 4), (3, True, 3), (10, False, 1), (1, False, 10), (20, False, 1)],
)
def test_len_agrees_with_what_it_yields(batch_size, drop_last, batches):
    """A ``len`` that disagrees silently truncates a progress bar, a
    learning-rate schedule, or an epoch."""
    loader = DataLoader(_dataset(10), batch_size=batch_size, drop_last=drop_last)
    assert len(loader) == batches
    assert len(list(loader)) == len(loader)


def test_an_empty_dataset_yields_nothing():
    loader = DataLoader(_dataset(0), batch_size=3)
    assert list(loader) == []
    assert len(loader) == 0


def test_a_batch_larger_than_the_dataset_yields_one_short_batch():
    assert _sizes(DataLoader(_dataset(3), batch_size=10)) == [3]


# ── shuffling ─────────────────────────────────────────────────────────────────


def test_shuffle_is_a_permutation_and_not_a_resample():
    """Sampling with replacement would have the right length and lose a
    third of the data to duplicates."""
    lucid.manual_seed(0)
    assert sorted(_seen(DataLoader(_dataset(20), batch_size=3, shuffle=True))) == list(range(20))


def test_shuffle_actually_changes_the_order():
    lucid.manual_seed(0)
    assert _seen(DataLoader(_dataset(50), batch_size=5, shuffle=True)) != list(range(50))


def test_each_epoch_reshuffles():
    """A loader that permuted once and reused the order would look
    shuffled for one epoch and be an ordered loader thereafter."""
    lucid.manual_seed(0)
    loader = DataLoader(_dataset(50), batch_size=5, shuffle=True)
    assert _seen(loader) != _seen(loader)


def test_shuffle_and_an_explicit_sampler_are_mutually_exclusive():
    with pytest.raises(ValueError):
        DataLoader(_dataset(6), batch_size=2, shuffle=True, sampler=SequentialSampler(_dataset(6)))


# ── samplers ──────────────────────────────────────────────────────────────────


def test_an_explicit_sampler_decides_which_samples_appear():
    loader = DataLoader(_dataset(10), batch_size=2, sampler=SubsetRandomSampler([0, 2, 4, 6]))
    assert sorted(_seen(loader)) == [0, 2, 4, 6]


def test_an_explicit_batch_sampler_decides_the_batches():
    sampler = BatchSampler(SequentialSampler(_dataset(6)), 4, False)
    assert _sizes(DataLoader(_dataset(6), batch_sampler=sampler)) == [4, 2]


def test_a_batch_sampler_excludes_the_other_batching_arguments():
    sampler = BatchSampler(SequentialSampler(_dataset(6)), 4, False)
    with pytest.raises(ValueError):
        DataLoader(_dataset(6), batch_size=2, batch_sampler=sampler)


# ── collation ─────────────────────────────────────────────────────────────────


def test_a_custom_collate_is_actually_called():
    """It was not.

    ``TensorDataset`` implements ``__getitems__``, a fast path that
    fetches and collates a whole batch in one fancy-index — six times
    faster, and worth having.  But the loader took it whenever the
    dataset offered it, which meant a caller-supplied ``collate_fn`` was
    accepted, stored, and never invoked once: variable-length sequences,
    dicts and graphs all came back default-collated instead.  The fast
    path now yields when the collate is not the default one.
    """
    calls = []

    def collate(items):
        calls.append(len(items))
        return {"n": len(items)}

    batches = list(DataLoader(_dataset(6), batch_size=2, collate_fn=collate))
    assert calls == [2, 2, 2]
    assert batches == [{"n": 2}, {"n": 2}, {"n": 2}]


def test_the_fast_path_is_still_taken_for_the_default_collate():
    """The fix must not cost everyone the optimisation it guards."""
    loader = DataLoader(_dataset(6), batch_size=2)
    assert next(iter(loader))[0].shape == (2,)
    explicit = DataLoader(_dataset(6), batch_size=2, collate_fn=default_collate)
    assert next(iter(explicit))[0].shape == (2,)


def test_a_custom_collate_receives_one_entry_per_sample():
    shapes = []
    DataLoader(
        _dataset(5), batch_size=2, collate_fn=lambda items: shapes.append(len(items))
    ).__iter__().__next__()
    assert shapes == [2]


# ── iterable-style datasets ───────────────────────────────────────────────────


def test_an_iterable_dataset_streams():
    """The class docstring always said both dataset styles were accepted.
    Until now the constructor built a ``SequentialSampler`` regardless and
    the first iteration died on ``len()``."""
    batches = [np.asarray(b.numpy()).tolist() for b in DataLoader(_Counting(0, 7), batch_size=3)]
    assert batches == [[0, 1, 2], [3, 4, 5], [6]]


def test_drop_last_applies_to_an_iterable_dataset_too():
    batches = list(DataLoader(_Counting(0, 7), batch_size=3, drop_last=True))
    assert len(batches) == 2


def test_an_iterable_dataset_shorter_than_one_batch():
    assert len(list(DataLoader(_Counting(0, 2), batch_size=10))) == 1


def test_an_empty_iterable_dataset_yields_nothing():
    assert list(DataLoader(_Counting(0, 0), batch_size=3)) == []


def test_a_chained_iterable_dataset_streams_through_the_loader():
    chained = ChainDataset([_Counting(0, 3), _Counting(10, 12)])
    flat = [v for b in DataLoader(chained, batch_size=2) for v in np.asarray(b.numpy()).ravel()]
    assert flat == [0, 1, 2, 10, 11]


def test_a_custom_collate_applies_to_an_iterable_dataset():
    assert list(DataLoader(_Counting(0, 5), batch_size=2, collate_fn=list)) == [[0, 1], [2, 3], [4]]


def test_shuffling_an_iterable_dataset_is_refused():
    """There are no indices to permute.  Silently ignoring it would be
    worse: the caller thinks the data is shuffled and it is not."""
    with pytest.raises(ValueError, match="shuffle"):
        DataLoader(_Counting(0, 5), batch_size=2, shuffle=True)


def test_a_sampler_over_an_iterable_dataset_is_refused():
    with pytest.raises(ValueError, match="sampler"):
        DataLoader(_Counting(0, 5), batch_size=2, sampler=SequentialSampler(_dataset(5)))


def test_the_length_of_an_iterable_loader_is_refused_rather_than_guessed():
    with pytest.raises(TypeError, match="no length"):
        len(DataLoader(_Counting(0, 5), batch_size=2))


# ── it still drives a training loop ───────────────────────────────────────────


def test_a_loader_feeds_a_training_loop():
    lucid.manual_seed(0)
    values = np.random.default_rng(0).standard_normal((32, 4)).astype(np.float32)
    dataset = TensorDataset(lucid.tensor(values), lucid.tensor(values[:, :1]))
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = lucid.nn.Linear(4, 1)
    optimiser = lucid.optim.SGD(model.parameters(), lr=0.05)

    def epoch_loss():
        return sum(float(np.asarray(((model(x) - y) ** 2).mean().numpy())) for x, y in loader)

    first = epoch_loss()
    for _ in range(5):
        for x, y in loader:
            optimiser.zero_grad()
            ((model(x) - y) ** 2).mean().backward()
            optimiser.step()
    assert epoch_loss() < first
