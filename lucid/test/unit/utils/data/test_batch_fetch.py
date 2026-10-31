"""How a dataset hands the loader a whole batch (LCD-801).

Two mechanisms: the public ``__getitems__(indices)``, which returns a list of
*samples* that still go through ``collate_fn``, and Lucid's private
``_collated_fetcher`` shortcut, which returns the already-collated batch.  The
shortcut is honoured only for the class that defines the ``__getitem__`` it
mirrors; a subclass that overrides ``__getitem__`` is read through its own
override, as in the reference.
"""

from types import ModuleType

import pytest

import lucid
from lucid.utils.data import (
    DataLoader,
    Dataset,
    StackDataset,
    Subset,
    TensorDataset,
    random_split,
)
from lucid.utils.data._batched import collated_fetcher


def _flat(batches):
    """Batches as nested Python lists — comparable across both frameworks."""

    def one(b):
        if isinstance(b, dict):
            return {k: one(v) for k, v in b.items()}
        if isinstance(b, (tuple, list)):
            return [one(v) for v in b]
        return b.tolist()

    return [one(b) for b in batches]


def _tensors(n: int = 6):
    return lucid.arange(float(n)), lucid.arange(n) * 3


def _ref_tensors(ref: ModuleType, n: int = 6):
    return ref.arange(float(n)), ref.arange(n) * 3


@pytest.fixture
def getitem_calls(monkeypatch):
    """Count ``TensorDataset.__getitem__`` calls without changing what it does."""
    calls: list[int] = []
    original = TensorDataset.__getitem__

    def counting(self, index):
        calls.append(index)
        return original(self, index)

    monkeypatch.setattr(TensorDataset, "__getitem__", counting)
    return calls


# ── a subclass that overrides __getitem__ ─────────────────────────────────────


class _Scaled(TensorDataset):
    def __getitem__(self, i):
        x, y = super().__getitem__(i)
        return x * 10, y


@pytest.mark.parity
def test_a_tensor_dataset_subclass_override_is_honoured(ref: ModuleType):
    ours = _flat(DataLoader(_Scaled(*_tensors()), batch_size=2))

    class RefScaled(ref.utils.data.TensorDataset):
        def __getitem__(self, i):
            x, y = super().__getitem__(i)
            return x * 10, y

    theirs = _flat(ref.utils.data.DataLoader(RefScaled(*_ref_tensors(ref)), batch_size=2))
    assert ours == theirs
    assert ours[0][0] == [0.0, 10.0]


def test_the_shortcut_does_not_inherit():
    assert collated_fetcher(TensorDataset(*_tensors())) is not None
    assert collated_fetcher(_Scaled(*_tensors())) is None


def test_a_subclass_that_defines_getitems_uses_its_own():
    class Own(TensorDataset):
        def __getitems__(self, indices):
            return [(lucid.tensor(-1.0), lucid.tensor(-1)) for _ in indices]

    assert collated_fetcher(Own(*_tensors())) is None
    batches = _flat(DataLoader(Own(*_tensors()), batch_size=3))
    assert batches[0][0] == [-1.0, -1.0, -1.0]


def test_the_plain_tensor_dataset_never_calls_getitem(getitem_calls):
    batches = _flat(DataLoader(TensorDataset(*_tensors()), batch_size=2))
    assert batches[1][0] == [2.0, 3.0]
    assert getitem_calls == []


# ── the public __getitems__ contract: a list of samples, then collate_fn ──────


class _ListGetItems(Dataset):
    def __init__(self):
        self.getitem_calls = 0

    def __len__(self):
        return 5

    def __getitem__(self, i):
        self.getitem_calls += 1
        return float(i), i * 2

    def __getitems__(self, indices):
        return [(float(i) + 100.0, i * 2) for i in indices]


@pytest.mark.parity
def test_getitems_returns_samples_that_are_collated(ref: ModuleType):
    ours = _flat(DataLoader(_ListGetItems(), batch_size=2))

    class RefListGetItems(ref.utils.data.Dataset):
        def __len__(self):
            return 5

        def __getitem__(self, i):
            raise AssertionError("__getitems__ should be used")

        def __getitems__(self, indices):
            return [(float(i) + 100.0, i * 2) for i in indices]

    theirs = _flat(ref.utils.data.DataLoader(RefListGetItems(), batch_size=2))
    assert ours == theirs
    assert ours[0] == [[100.0, 101.0], [0, 2]]


def test_getitems_samples_reach_a_custom_collate_fn():
    seen = []

    def collate(samples):
        seen.append(samples)
        return len(samples)

    ds = _ListGetItems()
    assert list(DataLoader(ds, batch_size=2, collate_fn=collate)) == [2, 2, 1]
    assert seen[0] == [(100.0, 0), (101.0, 2)]
    assert ds.getitem_calls == 0


# ── Subset ────────────────────────────────────────────────────────────────────


@pytest.mark.parity
def test_subset_batches_match_the_reference(ref: ModuleType):
    picks = [5, 1, 4, 0]
    ours = _flat(DataLoader(Subset(TensorDataset(*_tensors()), picks), batch_size=3))
    theirs = _flat(
        ref.utils.data.DataLoader(
            ref.utils.data.Subset(ref.utils.data.TensorDataset(*_ref_tensors(ref)), picks),
            batch_size=3,
        )
    )
    assert ours == theirs


@pytest.mark.parity
def test_subset_getitems_returns_the_remapped_samples(ref: ModuleType):
    picks = [5, 1, 4, 0]
    ours = Subset(TensorDataset(*_tensors()), picks).__getitems__([3, 0])
    theirs = ref.utils.data.Subset(
        ref.utils.data.TensorDataset(*_ref_tensors(ref)), picks
    ).__getitems__([3, 0])
    assert [[t.tolist() for t in s] for s in ours] == [[t.tolist() for t in s] for s in theirs]


def test_subset_of_a_tensor_dataset_keeps_the_fast_path(getitem_calls):
    nested = Subset(Subset(TensorDataset(*_tensors()), [5, 4, 3, 2]), [3, 0, 1])
    assert _flat(DataLoader(nested, batch_size=2))[0][0] == [2.0, 5.0]
    assert getitem_calls == []


def test_random_split_parts_keep_the_fast_path(getitem_calls):
    parts = random_split(TensorDataset(*_tensors(10)), [6, 4], generator=0)
    rows = [v for part in parts for b in DataLoader(part, batch_size=3) for v in b[0].tolist()]
    assert sorted(rows) == [float(i) for i in range(10)]
    assert getitem_calls == []


def test_a_subset_subclass_override_is_honoured():
    class Doubled(Subset):
        def __getitem__(self, idx):
            x, y = super().__getitem__(idx)
            return x * 2, y

    ds = Doubled(TensorDataset(*_tensors()), [3, 1])
    assert collated_fetcher(ds) is None
    assert _flat(DataLoader(ds, batch_size=2))[0][0] == [6.0, 2.0]


def test_a_subset_of_a_plain_dataset_has_no_shortcut():
    plain = _ListGetItems()
    assert collated_fetcher(Subset(plain, [0, 1])) is None
    assert _flat(DataLoader(Subset(plain, [4, 0]), batch_size=2))[0][0] == [104.0, 100.0]


# ── StackDataset ──────────────────────────────────────────────────────────────


@pytest.mark.parity
def test_stack_dataset_batches_match_the_reference(ref: ModuleType):
    ours = _flat(
        DataLoader(
            StackDataset(TensorDataset(lucid.arange(6.0)), TensorDataset(lucid.arange(6) * 3)),
            batch_size=4,
        )
    )
    theirs = _flat(
        ref.utils.data.DataLoader(
            ref.utils.data.StackDataset(
                ref.utils.data.TensorDataset(ref.arange(6.0)),
                ref.utils.data.TensorDataset(ref.arange(6) * 3),
            ),
            batch_size=4,
        )
    )
    assert ours == theirs


@pytest.mark.parity
def test_keyed_stack_dataset_batches_match_the_reference(ref: ModuleType):
    ours = _flat(
        DataLoader(
            StackDataset(a=TensorDataset(lucid.arange(5.0)), b=TensorDataset(lucid.arange(5))),
            batch_size=2,
        )
    )
    theirs = _flat(
        ref.utils.data.DataLoader(
            ref.utils.data.StackDataset(
                a=ref.utils.data.TensorDataset(ref.arange(5.0)),
                b=ref.utils.data.TensorDataset(ref.arange(5)),
            ),
            batch_size=2,
        )
    )
    assert ours == theirs
    assert set(ours[0]) == {"a", "b"}


@pytest.mark.parity
def test_stack_dataset_getitems_matches_the_reference(ref: ModuleType):
    ours = StackDataset(a=TensorDataset(lucid.arange(4.0)), b=TensorDataset(lucid.arange(4)))
    theirs = ref.utils.data.StackDataset(
        a=ref.utils.data.TensorDataset(ref.arange(4.0)),
        b=ref.utils.data.TensorDataset(ref.arange(4)),
    )
    got = [{k: [t.tolist() for t in v] for k, v in s.items()} for s in ours.__getitems__([2, 0])]
    want = [{k: [t.tolist() for t in v] for k, v in s.items()} for s in theirs.__getitems__([2, 0])]
    assert got == want


def test_stack_dataset_keeps_the_fast_path(getitem_calls):
    ds = StackDataset(TensorDataset(lucid.arange(6.0)), TensorDataset(lucid.arange(6)))
    assert collated_fetcher(ds) is not None
    assert _flat(DataLoader(ds, batch_size=3))[0] == [[[0.0, 1.0, 2.0]], [[0, 1, 2]]]
    assert getitem_calls == []


def test_stack_dataset_with_a_plain_child_falls_back_to_samples():
    ds = StackDataset(TensorDataset(lucid.arange(5.0)), _ListGetItems())
    assert collated_fetcher(ds) is None
    batch = _flat(DataLoader(ds, batch_size=2))[0]
    assert batch[0] == [[0.0, 1.0]]
    assert batch[1] == [[100.0, 101.0], [0, 2]]


def test_stack_dataset_refuses_a_child_returning_the_wrong_count():
    class Short(_ListGetItems):
        def __getitems__(self, indices):
            return super().__getitems__(indices)[:-1]

    ds = StackDataset(TensorDataset(lucid.arange(5.0)), Short())
    with pytest.raises(ValueError, match="returned"):
        ds.__getitems__([0, 1])
