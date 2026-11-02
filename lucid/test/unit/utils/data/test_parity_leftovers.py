"""LCD-803: the ``lucid.utils.data`` behaviours the reference framework defines
and Lucid still answered differently.

One table, two readings.  Each case is a function of a *surface* — the
data module plus a tensor constructor — so the identical code runs on Lucid
and on the reference.  ``EXPECTED`` is Lucid's outcome written out (always
checked); the ``parity`` test runs the same case on the reference and holds
the two to each other, except where the reference's own outcome is an
accident Lucid keeps different on purpose (``CLASS_ONLY``).
"""

import collections
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType

import pytest

import lucid
from lucid.utils import data as lucid_data

from lucid.test.unit.utils.data import _worker_datasets as wd


@dataclass(frozen=True)
class Surface:
    """What a case needs: the data package and a way to make a tensor."""

    data: ModuleType
    tensor: Callable[..., object]
    ones: Callable[..., object]


LUCID = Surface(lucid_data, lucid.tensor, lucid.ones)


def _iterable(s: Surface, n: int, sized: bool = True) -> object:
    """An iterable-style dataset over ``range(n)`` built on ``s``'s base class."""

    class Stream(s.data.IterableDataset):  # type: ignore[name-defined]
        def __iter__(self):
            yield from range(n)

    if sized:
        Stream.__len__ = lambda self: n  # type: ignore[attr-defined]
    return Stream()


# ── cases ─────────────────────────────────────────────────────────────────────


def _concat_iterables(s: Surface) -> object:
    return list(_iterable(s, 2) + _iterable(s, 2))  # type: ignore[operator]


def _chain_len(s: Surface) -> object:
    return len(s.data.ChainDataset([_iterable(s, 2), _iterable(s, 3)]))


def _loader_len(s: Surface, **kw: object) -> object:
    return len(s.data.DataLoader(_iterable(s, 5), **kw))


def _loader_len_unsized(s: Surface) -> object:
    return len(s.data.DataLoader(_iterable(s, 5, sized=False), batch_size=2))


def _weights_2d(s: Surface) -> object:
    return s.data.WeightedRandomSampler(s.ones(2, 2), 3)


def _weights_0d(s: Surface) -> object:
    return s.data.WeightedRandomSampler(s.tensor(1.0), 3)


def _tensor_dataset_0d(s: Surface) -> object:
    return s.data.TensorDataset(s.tensor(1.0))


def _tensor_dataset_second_0d(s: Surface) -> object:
    return s.data.TensorDataset(s.ones(3), s.tensor(1.0))


def _dataset_getitem(s: Surface) -> object:
    return s.data.Dataset()[0]


def _split_fraction_out_of_range(s: Surface) -> object:
    return s.data.random_split(range(5), [1.5, -0.5])


def _split_negative_fraction(s: Surface) -> object:
    return s.data.random_split(range(5), [0.5, 0.75, -0.25])


def _split_fractions_miss_one(s: Surface) -> object:
    return s.data.random_split(range(5), [0.5, 0.2])


def _split_sizes_miss_length(s: Surface) -> object:
    return s.data.random_split(range(5), [2, 2])


def _split_fractions_fine(s: Surface) -> object:
    return [len(p) for p in s.data.random_split(range(10), [0.33, 0.33, 0.34])]


def _reassign(attr: str, value: object) -> Callable[[Surface], object]:
    def case(s: Surface) -> object:
        loader = s.data.DataLoader(list(range(4)), batch_size=2)
        setattr(loader, attr, value)
        return "assigned"

    return case


def _reassign_free(s: Surface) -> object:
    loader = s.data.DataLoader(list(range(4)), batch_size=2)
    loader.num_workers = 0
    loader.collate_fn = lambda b: b
    loader.timeout = 1
    return "assigned"


def _loader_accepts(s: Surface) -> object:
    loader = s.data.DataLoader(
        list(range(4)), batch_size=2, pin_memory_device="mps", in_order=False
    )
    return (loader.pin_memory_device, loader.in_order, [b.tolist() for b in loader])


def _loader_defaults(s: Surface) -> object:
    loader = s.data.DataLoader(list(range(4)), batch_size=2)
    return (loader.pin_memory_device, loader.in_order)


def _collate_ordered_dict(s: Surface) -> object:
    batch = [collections.OrderedDict(b=i, a=i + 1) for i in range(3)]
    out = s.data.default_collate(batch)
    return (type(out).__name__, list(out), {k: v.tolist() for k, v in out.items()})


def _collate_defaultdict(s: Surface) -> object:
    out = s.data.default_collate([collections.defaultdict(list, x=1) for _ in range(2)])
    return (type(out).__name__, out.default_factory.__name__)


def _collate_ragged_lists(s: Surface) -> object:
    return s.data.default_collate([[1, 2], [3]])


def _collate_ragged_tuples(s: Surface) -> object:
    return s.data.default_collate([(1, 2, 3), (4, 5)])


def _convert_ordered_dict(s: Surface) -> object:
    out = s.data.default_convert(collections.OrderedDict(b=s.tensor(1), a=s.tensor(2)))
    return (type(out).__name__, list(out))


CASES: dict[str, Callable[[Surface], object]] = {
    "iterable_plus_iterable": _concat_iterables,
    "chain_len": _chain_len,
    "loader_len_batched": lambda s: _loader_len(s, batch_size=2),
    "loader_len_drop_last": lambda s: _loader_len(s, batch_size=2, drop_last=True),
    "loader_len_unbatched": lambda s: _loader_len(s, batch_size=None),
    "loader_len_exact_multiple": lambda s: _loader_len(s, batch_size=5),
    "weights_2d": _weights_2d,
    "weights_0d": _weights_0d,
    "tensor_dataset_0d": _tensor_dataset_0d,
    "tensor_dataset_second_0d": _tensor_dataset_second_0d,
    "dataset_getitem": _dataset_getitem,
    "split_fraction_out_of_range": _split_fraction_out_of_range,
    "split_negative_fraction": _split_negative_fraction,
    "split_fractions_miss_one": _split_fractions_miss_one,
    "split_sizes_miss_length": _split_sizes_miss_length,
    "split_fractions_fine": _split_fractions_fine,
    "reassign_batch_size": _reassign("batch_size", 3),
    "reassign_batch_sampler": _reassign("batch_sampler", None),
    "reassign_sampler": _reassign("sampler", None),
    "reassign_drop_last": _reassign("drop_last", True),
    "reassign_dataset": _reassign("dataset", [1]),
    "reassign_persistent_workers": _reassign("persistent_workers", True),
    "reassign_free_attributes": _reassign_free,
    "loader_accepts_in_order_and_pin_memory_device": _loader_accepts,
    "loader_defaults": _loader_defaults,
    "collate_ordered_dict": _collate_ordered_dict,
    "collate_defaultdict": _collate_defaultdict,
    "collate_ragged_lists": _collate_ragged_lists,
    "collate_ragged_tuples": _collate_ragged_tuples,
    "convert_ordered_dict": _convert_ordered_dict,
}

_ASSIGN_MSG = "{} attribute should not be set after DataLoader is initialized"
_SUM_MSG = "Sum of input lengths does not equal the length of the input dataset!"
_RAGGED_MSG = "each element in list of batch should be of equal size"
_ZERO_D = "TensorDataset tensor {} is 0-d; it has no first dimension to index samples along."

EXPECTED: dict[str, tuple[object, ...]] = {
    "iterable_plus_iterable": ("ok", [0, 1, 0, 1]),
    "chain_len": ("ok", 5),
    "loader_len_batched": ("ok", 3),
    "loader_len_drop_last": ("ok", 2),
    "loader_len_unbatched": ("ok", 5),
    "loader_len_exact_multiple": ("ok", 1),
    "weights_2d": (
        "raises",
        "ValueError",
        "weights should be a 1d sequence but given weights have shape (2, 2)",
    ),
    "weights_0d": (
        "raises",
        "ValueError",
        "weights should be a 1d sequence but given weights have shape ()",
    ),
    "tensor_dataset_0d": ("raises", "IndexError", _ZERO_D.format(0)),
    "tensor_dataset_second_0d": ("raises", "IndexError", _ZERO_D.format(1)),
    "dataset_getitem": (
        "raises",
        "NotImplementedError",
        "Subclasses of Dataset should implement __getitem__.",
    ),
    "split_fraction_out_of_range": (
        "raises",
        "ValueError",
        "Fraction at index 0 is not between 0 and 1",
    ),
    "split_negative_fraction": (
        "raises",
        "ValueError",
        "Fraction at index 2 is not between 0 and 1",
    ),
    "split_fractions_miss_one": ("raises", "ValueError", _SUM_MSG),
    "split_sizes_miss_length": ("raises", "ValueError", _SUM_MSG),
    "split_fractions_fine": ("ok", [4, 3, 3]),
    "reassign_batch_size": ("raises", "ValueError", _ASSIGN_MSG.format("batch_size")),
    "reassign_batch_sampler": ("raises", "ValueError", _ASSIGN_MSG.format("batch_sampler")),
    "reassign_sampler": ("raises", "ValueError", _ASSIGN_MSG.format("sampler")),
    "reassign_drop_last": ("raises", "ValueError", _ASSIGN_MSG.format("drop_last")),
    "reassign_dataset": ("raises", "ValueError", _ASSIGN_MSG.format("dataset")),
    "reassign_persistent_workers": (
        "raises",
        "ValueError",
        _ASSIGN_MSG.format("persistent_workers"),
    ),
    "reassign_free_attributes": ("ok", "assigned"),
    "loader_accepts_in_order_and_pin_memory_device": ("ok", ("mps", False, [[0, 1], [2, 3]])),
    "loader_defaults": ("ok", ("", True)),
    "collate_ordered_dict": ("ok", ("OrderedDict", ["b", "a"], {"b": [0, 1, 2], "a": [1, 2, 3]})),
    "collate_defaultdict": ("ok", ("defaultdict", "list")),
    "collate_ragged_lists": ("raises", "RuntimeError", _RAGGED_MSG),
    "collate_ragged_tuples": ("raises", "RuntimeError", _RAGGED_MSG),
    "convert_ordered_dict": ("ok", ("OrderedDict", ["b", "a"])),
}

# The reference raises, but with a message of its own choosing (its error
# for a 0-d tensor is whatever ``size(0)`` says); the class is the contract.
CLASS_ONLY = {"tensor_dataset_0d", "tensor_dataset_second_0d"}


def _outcome(case: Callable[[Surface], object], surface: Surface) -> tuple[object, ...]:
    try:
        return ("ok", case(surface))
    except Exception as exc:  # noqa: BLE001 — the outcome *is* the exception
        return ("raises", type(exc).__name__, str(exc))


def test_the_table_covers_every_case():
    assert set(CASES) == set(EXPECTED)


@pytest.mark.parametrize("name", sorted(CASES))
def test_lucid_outcome(name):
    assert _outcome(CASES[name], LUCID) == EXPECTED[name]


@pytest.mark.parity
@pytest.mark.parametrize("name", sorted(CASES))
def test_matches_the_reference(ref: ModuleType, name):
    reference = Surface(ref.utils.data, ref.tensor, ref.ones)
    ours, theirs = EXPECTED[name], _outcome(CASES[name], reference)
    if name in CLASS_ONLY:
        ours, theirs = ours[:2], theirs[:2]
    assert theirs == ours


# ── Lucid-only edges ──────────────────────────────────────────────────────────


def test_an_unsized_iterable_loader_still_has_no_length():
    with pytest.raises(TypeError, match="no length"):
        _loader_len_unsized(LUCID)


def test_chaining_with_a_map_style_dataset_is_refused_by_the_chain():
    class Stream(lucid_data.IterableDataset):
        def __iter__(self):
            yield 1

    with pytest.raises(TypeError, match="IterableDataset"):
        Stream() + lucid_data.TensorDataset(lucid.ones(2))  # type: ignore[operator]


def test_a_chain_with_an_unsized_child_has_no_length():
    with pytest.raises(TypeError, match="no length"):
        len(lucid_data.ChainDataset([_iterable(LUCID, 2), _iterable(LUCID, 2, sized=False)]))


def test_a_list_of_lists_of_weights_is_refused():
    with pytest.raises(ValueError, match="1d sequence"):
        lucid_data.WeightedRandomSampler([[1.0, 2.0], [3.0, 4.0]], 3)


def test_a_flat_weight_tensor_still_works():
    sampler = lucid_data.WeightedRandomSampler(lucid.tensor([0.0, 1.0, 0.0]), 4, generator=0)
    assert list(sampler) == [1, 1, 1, 1]


@pytest.mark.parametrize("bad", [1, None, "yes"])
def test_in_order_must_be_a_bool(bad):
    with pytest.raises(TypeError, match="in_order"):
        lucid_data.DataLoader([1, 2], in_order=bad)


@pytest.mark.parametrize("bad", [1, None, True])
def test_pin_memory_device_must_be_a_str(bad):
    with pytest.raises(TypeError, match="pin_memory_device"):
        lucid_data.DataLoader([1, 2], pin_memory_device=bad)


def test_a_subclass_cannot_reassign_a_frozen_attribute_either():
    class Loader(lucid_data.DataLoader):
        pass

    loader = Loader([1, 2], batch_size=1)
    with pytest.raises(ValueError, match="Loader is initialized"):
        loader.batch_size = 2


def test_iterable_loader_length_with_a_chain_of_sized_streams():
    chain = lucid_data.ChainDataset([_iterable(LUCID, 4), _iterable(LUCID, 3)])
    assert len(lucid_data.DataLoader(chain, batch_size=3)) == 3
    assert len(list(lucid_data.DataLoader(chain, batch_size=3))) == 3


# ── in_order=False ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("in_order", [True, False])
def test_every_batch_arrives_once_whatever_the_order(in_order):
    loader = lucid_data.DataLoader(
        wd.Indices(9), batch_size=2, num_workers=2, in_order=in_order, collate_fn=wd.as_is
    )
    seen = [i for batch in loader for i in batch]
    assert sorted(seen) == list(range(9))
    if in_order:
        assert seen == list(range(9))


def test_out_of_order_does_not_wait_for_a_slow_batch():
    """Sample 0 sleeps; with ``in_order=False`` the batches behind it come first."""
    loader = lucid_data.DataLoader(
        wd.FirstSlow(6, delay=3.0), batch_size=1, num_workers=2, in_order=False, collate_fn=wd.as_is
    )
    seen = [i for batch in loader for i in batch]
    assert sorted(seen) == [0, 1, 2, 3, 4, 5]
    assert seen[0] != 0 and seen.index(0) > 0


def test_in_order_waits_for_the_slow_batch():
    loader = lucid_data.DataLoader(
        wd.FirstSlow(4, delay=1.0), batch_size=1, num_workers=2, collate_fn=wd.as_is
    )
    assert [i for batch in loader for i in batch] == [0, 1, 2, 3]


def test_a_persistent_out_of_order_pool_runs_every_epoch():
    loader = lucid_data.DataLoader(
        wd.Indices(6),
        batch_size=2,
        num_workers=2,
        in_order=False,
        persistent_workers=True,
        collate_fn=wd.as_is,
    )
    for _ in range(2):
        assert sorted(i for batch in loader for i in batch) == list(range(6))
