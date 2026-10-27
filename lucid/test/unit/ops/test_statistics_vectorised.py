"""bincount, histogram and the index builders run as tensor ops (API-05).

``bincount``, ``histogram`` and ``combinations`` read their input one
element at a time with ``.item()`` — 0.5 s for a 20k ``bincount`` on the
CPU and 18 s on Metal, where the reference takes 0.02 ms (LCD-165) — and
answered in other dtypes than the reference: ``float64`` weighted counts,
which Metal cannot hold, and ``int64`` histograms.  ``tril_indices`` /
``triu_indices`` looped over every cell of the matrix the same way.

Each is checked against the reference on both devices, and for the class
— a per-element host read — by counting the host reads a large input
costs: a constant, whatever the input's length.
"""

import math
from collections.abc import Callable
from types import ModuleType

import pytest

import lucid
from lucid.test._helpers.compare import assert_close

# ── bincount ────────────────────────────────────────────────────────────────

_COUNTS = [0, 1, 1, 3, 3, 3, 7]


@pytest.mark.parity
@pytest.mark.parametrize("minlength", [0, 3, 12])
@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
def test_bincount_counts_as_the_reference(
    ref: ModuleType, device: str, minlength: int, index_dtype: str
) -> None:
    want = ref.bincount(ref.tensor(_COUNTS, dtype=getattr(ref, index_dtype)), minlength=minlength)
    got = lucid.bincount(
        lucid.tensor(_COUNTS, dtype=getattr(lucid, index_dtype), device=device),
        minlength=minlength,
    )
    assert got.dtype == lucid.int64
    assert got.device.type == device
    assert got.tolist() == want.tolist()


@pytest.mark.parity
@pytest.mark.parametrize("weight_dtype", ["float32", "float64", "float16", "int64"])
def test_bincount_sums_weights_in_the_reference_dtype(
    ref: ModuleType, device: str, weight_dtype: str
) -> None:
    if device == "metal" and weight_dtype == "float64":
        pytest.skip("Metal holds no float64")
    weights = [1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    if weight_dtype == "int64":
        weights = [int(w) for w in weights]
    want = ref.bincount(
        ref.tensor(_COUNTS),
        weights=ref.tensor(weights, dtype=getattr(ref, weight_dtype)),
    )
    got = lucid.bincount(
        lucid.tensor(_COUNTS, device=device),
        weights=lucid.tensor(weights, dtype=getattr(lucid, weight_dtype), device=device),
    )
    # The reference sums anything but float32 / float64 in float64, which
    # Metal does not hold: there the sums are float32.
    expected = str(want.dtype).removeprefix(ref.__name__ + ".")
    if device == "metal" and expected == "float64":
        expected = "float32"
    assert got.dtype == getattr(lucid, expected)
    assert_close(got, want.double().numpy(), rtol=1e-3)


def test_bincount_of_nothing_is_minlength_zeros(device: str) -> None:
    empty = lucid.tensor([], dtype=lucid.int64, device=device)
    assert lucid.bincount(empty).tolist() == []
    assert lucid.bincount(empty, minlength=3).tolist() == [0, 0, 0]


@pytest.mark.parametrize(
    ("values", "kwargs", "error"),
    [
        ([-1, 1], {}, ValueError),
        ([[1, 1]], {}, ValueError),
        ([1.0, 1.0], {}, TypeError),
        ([True, False], {}, TypeError),
        ([1], {"minlength": -1}, ValueError),
    ],
    ids=["negative", "2-d", "float", "bool", "negative-minlength"],
)
def test_bincount_refuses_what_the_reference_refuses(
    device: str, values: list[object], kwargs: dict[str, int], error: type
) -> None:
    with pytest.raises(error):
        lucid.bincount(lucid.tensor(values, device=device), **kwargs)


def test_bincount_refuses_weights_of_another_length(device: str) -> None:
    with pytest.raises(ValueError, match="as long as input"):
        lucid.bincount(
            lucid.tensor([1, 2], device=device),
            weights=lucid.tensor([1.0], device=device),
        )


# ── histogram ───────────────────────────────────────────────────────────────

_VALUES = [0.25, -1.5, 2.0, 0.75, 1.0, -0.5, 1.75, 2.0, 0.0, -1.5]

_HISTOGRAMS: dict[str, Callable[[ModuleType, object], object]] = {
    "bins": lambda m, v: m.histogram(v, bins=4),
    "range": lambda m, v: m.histogram(v, bins=5, range=(-1.0, 1.5)),
    "density": lambda m, v: m.histogram(v, bins=3, density=True),
    "weight": lambda m, v: m.histogram(v, bins=4, weight=v * v),
    "weight-density": lambda m, v: m.histogram(v, bins=4, weight=v * v, density=True),
    "edges": lambda m, v: m.histogram(v, bins=m.tensor([-2.0, 0.0, 0.5, 3.0])),
    "edges-density": lambda m, v: m.histogram(
        v, bins=m.tensor([-2.0, 0.0, 0.5, 3.0]), density=True
    ),
    "one-bin": lambda m, v: m.histogram(v, bins=1),
}


@pytest.mark.parity
@pytest.mark.parametrize("name", list(_HISTOGRAMS))
def test_histogram_matches_the_reference(ref: ModuleType, device: str, name: str) -> None:
    want_hist, want_edges = _HISTOGRAMS[name](ref, ref.tensor(_VALUES))
    hist, edges = _HISTOGRAMS[name](_OnDevice(device), lucid.tensor(_VALUES, device=device))
    for got in (hist, edges):
        assert got.dtype == lucid.float32
        assert got.device.type == device
    assert_close(hist, want_hist.numpy())
    assert_close(edges, want_edges.numpy())


@pytest.mark.parity
@pytest.mark.parametrize("values", [[1.0, 1.0], [], [3.0]], ids=["equal", "empty", "single"])
def test_histogram_of_a_point_or_nothing(ref: ModuleType, device: str, values: list[float]) -> None:
    want_hist, want_edges = ref.histogram(ref.tensor(values), bins=2)
    hist, edges = lucid.histogram(lucid.tensor(values, device=device), bins=2)
    assert hist.tolist() == want_hist.tolist()
    assert edges.tolist() == want_edges.tolist()


@pytest.mark.parametrize(
    ("values", "kwargs"),
    [
        ([1.0], {"range": (1.0, float("nan"))}),
        ([1.0], {"range": (float("-inf"), 1.0)}),
        ([1.0], {"range": (3.0, 1.0)}),
        ([1.0], {"bins": 0}),
        ([1.0], {"bins": [1.0]}),
        ([1.0, 2.0], {"weight": lucid.ones(3)}),
    ],
    ids=["nan-range", "inf-range", "reversed-range", "no-bins", "one-edge", "weight"],
)
def test_histogram_refuses_what_the_reference_refuses(
    device: str, values: list[float], kwargs: dict[str, object]
) -> None:
    with pytest.raises(ValueError):
        lucid.histogram(lucid.tensor(values, device=device), **kwargs)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_range_read_from_non_finite_values_gives_nan_bins(device: str, bad: float) -> None:
    # The reference refuses; refusing would read the extremes back to the
    # host, which the histogram does not do.  NaN bins say it instead.
    hist, _ = lucid.histogram(lucid.tensor([1.0, bad], device=device), bins=3)
    assert all(math.isnan(v) for v in hist.tolist())


def test_histogram_bins_the_edges_as_the_edges_say(device: str) -> None:
    # (v - lo) * bins / (hi - lo) rounds: of the values sitting exactly on
    # the edges of 7 bins over [-1.3, 2.9], three land one bin low.  The
    # edges themselves decide, so each value opens its own bin.
    _, edges = lucid.histogram(lucid.zeros(1, device=device), bins=7, range=(-1.3, 2.9))
    hist, _ = lucid.histogram(edges, bins=7, range=(-1.3, 2.9))
    assert hist.tolist() == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0]


# ── combinations and the triangle indices ───────────────────────────────────


@pytest.mark.parity
@pytest.mark.parametrize("with_replacement", [False, True])
@pytest.mark.parametrize("r", [0, 1, 2, 3, 5])
def test_combinations_match_the_reference(
    ref: ModuleType, device: str, r: int, with_replacement: bool
) -> None:
    values = [3.0, 1.0, 4.0, 1.5]
    want = ref.combinations(ref.tensor(values), r, with_replacement=with_replacement)
    got = lucid.combinations(
        lucid.tensor(values, device=device), r, with_replacement=with_replacement
    )
    assert tuple(got.shape) == tuple(want.shape)
    assert got.device.type == device
    assert got.tolist() == want.tolist()


@pytest.mark.parity
def test_combinations_carry_the_gradient(ref: ModuleType, device: str) -> None:
    values = [3.0, 1.0, 4.0]
    x = lucid.tensor(values, device=device, requires_grad=True)
    rx = ref.tensor(values, requires_grad=True)
    (lucid.combinations(x, 2) ** 2).sum().backward()
    (ref.combinations(rx, 2) ** 2).sum().backward()
    assert x.grad is not None
    assert_close(x.grad, rx.grad.numpy())


def test_combinations_refuse_a_negative_length_and_a_matrix(device: str) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        lucid.combinations(lucid.tensor([1.0, 2.0], device=device), -1)
    with pytest.raises(ValueError, match="1-D"):
        lucid.combinations(lucid.zeros(1, 3, device=device))


@pytest.mark.parity
@pytest.mark.parametrize("which", ["tril_indices", "triu_indices"])
@pytest.mark.parametrize(
    ("row", "col", "offset"),
    [(3, 3, 0), (3, 5, 1), (4, 2, -1), (0, 3, 0), (3, 3, -5), (2, 4, 9)],
)
def test_triangle_indices_match_the_reference(
    ref: ModuleType, device: str, which: str, row: int, col: int, offset: int
) -> None:
    want = getattr(ref, which)(row, col, offset)
    got = getattr(lucid, which)(row, col, offset, device=device)
    assert got.dtype == lucid.int64
    assert got.device.type == device
    assert tuple(got.shape) == tuple(want.shape)
    assert got.tolist() == want.tolist()


# ── the class: host reads do not grow with the input ───────────────────────


def _count_host_reads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every ``Tensor.item`` / ``Tensor.tolist`` from now on."""
    seen: list[str] = []
    item, tolist = lucid.Tensor.item, lucid.Tensor.tolist

    def counted_item(self: lucid.Tensor) -> object:
        seen.append("item")
        return item(self)

    def counted_tolist(self: lucid.Tensor) -> object:
        seen.append("tolist")
        return tolist(self)

    monkeypatch.setattr(lucid.Tensor, "item", counted_item)
    monkeypatch.setattr(lucid.Tensor, "tolist", counted_tolist)
    return seen


_BULK: dict[str, Callable[[str, int], object]] = {
    "bincount": lambda d, n: lucid.bincount(lucid.arange(n, device=d) % 7),
    "bincount-weights": lambda d, n: lucid.bincount(
        lucid.arange(n, device=d) % 7, weights=lucid.ones(n, device=d)
    ),
    "histogram": lambda d, n: lucid.histogram(lucid.arange(float(n), device=d), bins=9),
    "histogram-edges": lambda d, n: lucid.histogram(
        lucid.arange(float(n), device=d), bins=[0.0, 5.0, float(n)]
    ),
    "combinations": lambda d, n: lucid.combinations(lucid.arange(float(n), device=d), 3),
    "tril_indices": lambda d, n: lucid.tril_indices(n, n, device=d),
}


@pytest.mark.parametrize("name", list(_BULK))
def test_host_reads_do_not_grow_with_the_input(
    monkeypatch: pytest.MonkeyPatch, device: str, name: str
) -> None:
    seen = _count_host_reads(monkeypatch)
    _BULK[name](device, 10)
    small = len(seen)
    _BULK[name](device, 40)
    assert len(seen) - small == small <= 2
    if name.startswith("histogram"):
        # Its output size is known before any value is read.
        assert small == 0


class _OnDevice(ModuleType):
    """``lucid`` with ``tensor`` placing its result on one device."""

    def __init__(self, device: str) -> None:
        super().__init__("lucid_on_device")
        self._device = device

    def tensor(self, values: object, **kwargs: object) -> lucid.Tensor:
        return lucid.tensor(values, device=self._device, **kwargs)

    def __getattr__(self, name: str) -> object:
        return getattr(lucid, name)
