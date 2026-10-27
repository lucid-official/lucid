"""A split's sizes must sum to the axis it cuts (LCD-490).

``split`` turns its sizes into the cut points of the engine's ``split_at``,
which checks only that each cut lies inside the axis.  A size list that did
not sum to the axis passed that check, and the last piece made up the
difference: ``[2, 3]`` of a length-6 axis came back ``[2, 4]``, of a
length-4 one ``[2, 2]``.  The sizes are checked where they become cut points
now, so every spelling refuses them as the reference does — the free
function, the method and ``CatTransform``.  ``tensor_split`` and its
siblings take cut *indices* and are permissive by design.  They used to
reach ``split`` through a size list that relied on the old slack, and now
cut the reference's slices themselves.
"""

from collections.abc import Callable
from types import ModuleType

import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid.distributions.transforms import AffineTransform, CatTransform, ExpTransform

_SHAPE = (4, 6)

_Outcome = tuple[str, object]


def _outcome(call: Callable[[], object]) -> _Outcome:
    """The pieces' values, or the refusal's message."""
    try:
        pieces = call()
    except RuntimeError as exc:
        return "refused", str(exc).splitlines()[0]
    assert isinstance(pieces, (list, tuple))
    return "pieces", [(tuple(p.shape), p.tolist()) for p in pieces]


def _pair(
    ref: ModuleType, device: str, shape: tuple[int, ...] = _SHAPE
) -> tuple[lucid.Tensor, object]:
    n = 1
    for s in shape:
        n *= s
    mine = lucid.arange(n, dtype=lucid.float32, device=device).reshape(shape)
    theirs = ref.arange(n, dtype=ref.float32).reshape(shape)
    return mine, theirs


def _agree(got: _Outcome, want: _Outcome, call: Callable[[], object]) -> None:
    assert got == want
    if got[0] == "refused":
        # Lucid's ``InvalidArgument`` is the reference's ``RuntimeError`` and
        # Lucid's old ``ValueError`` at once, so either ``except`` still works.
        with pytest.raises(_C_engine.InvalidArgument):
            call()
        with pytest.raises(ValueError):
            call()


_SPELLINGS: dict[str, Callable[..., object]] = {
    "free": lambda lib, x, sizes, dim: lib.split(x, sizes, dim),
    "free-keyword": lambda lib, x, sizes, dim: lib.split(x, sizes, dim=dim),
    "method": lambda lib, x, sizes, dim: x.split(sizes, dim),
}

#: (sizes, dim) on a (4, 6) tensor: short of, equal to and past the axis,
#: negative entries, zero entries and no entries at all.
_SIZE_LISTS = [
    ([2, 3], -1),
    ([2, 5], -1),
    ([2, 4], -1),
    ([2, 4], 1),
    ([1, 2], 0),
    ([1, 3], 0),
    ([3, 3], 0),
    ([0, 6, 0], -1),
    ([0, 0, 4], 0),
    ([-1, 7], -1),
    ([7, -1], 1),
    ([-1, 3], -1),
    ([], 0),
    ((2, 4), -1),
]


@pytest.mark.parity
@pytest.mark.parametrize("spelling", list(_SPELLINGS))
@pytest.mark.parametrize(("sizes", "dim"), _SIZE_LISTS, ids=str)
def test_a_size_list_splits_as_the_reference_splits(
    ref: ModuleType, device: str, spelling: str, sizes: list[int], dim: int
) -> None:
    split = _SPELLINGS[spelling]
    x, r = _pair(ref, device)
    want = _outcome(lambda: split(ref, r, sizes, dim))
    got = _outcome(lambda: split(lucid, x, sizes, dim))
    _agree(got, want, lambda: split(lucid, x, sizes, dim))


@pytest.mark.parity
@pytest.mark.parametrize("spelling", list(_SPELLINGS))
@pytest.mark.parametrize(
    ("shape", "size", "dim"),
    [
        # An int size that does not divide leaves a shorter last piece.
        (_SHAPE, 4, -1),
        (_SHAPE, 3, 0),
        (_SHAPE, 6, -1),
        (_SHAPE, 7, 0),
        (_SHAPE, 1, 0),
        (_SHAPE, 0, -1),
        (_SHAPE, -1, 0),
        ((0, 6), 2, 0),
        ((0, 6), 0, 0),
        ((4, 0), 0, -1),
    ],
    ids=str,
)
def test_a_split_size_splits_as_the_reference_splits(
    ref: ModuleType, device: str, spelling: str, shape: tuple[int, ...], size: int, dim: int
) -> None:
    split = _SPELLINGS[spelling]
    x, r = _pair(ref, device, shape)
    want = _outcome(lambda: split(ref, r, size, dim))
    got = _outcome(lambda: split(lucid, x, size, dim))
    _agree(got, want, lambda: split(lucid, x, size, dim))


@pytest.mark.parity
@pytest.mark.parametrize("method", [False, True], ids=["free", "method"])
@pytest.mark.parametrize(
    ("shape", "chunks", "dim"),
    [
        (_SHAPE, 4, -1),
        (_SHAPE, 4, 0),
        ((7, 2), 4, 0),
        ((0, 2), 3, 0),
        (_SHAPE, 0, 0),
        (_SHAPE, -1, -1),
    ],
    ids=str,
)
def test_chunk_splits_as_the_reference_splits(
    ref: ModuleType, device: str, method: bool, shape: tuple[int, ...], chunks: int, dim: int
) -> None:
    x, r = _pair(ref, device, shape)

    def chunk(lib: ModuleType, t: lucid.Tensor) -> object:
        return t.chunk(chunks, dim) if method else lib.chunk(t, chunks, dim)

    want = _outcome(lambda: chunk(ref, r))
    got = _outcome(lambda: chunk(lucid, x))
    _agree(got, want, lambda: chunk(lucid, x))


#: Cut indices are slices: past either end, below the previous cut, negative.
_INDICES = [
    [2, 5],
    [5, 2],
    [2, 15],
    [12, 15],
    [-2],
    [-20, 3],
    [3, -20],
    [4, 1, 5],
    [],
    3,
    12,
    0,
    -1,
]


@pytest.mark.parity
@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("indices", _INDICES, ids=str)
def test_tensor_split_slices_as_the_reference_slices(
    ref: ModuleType, device: str, indices: int | list[int], dim: int
) -> None:
    x, r = _pair(ref, device, (10, 6) if dim == 0 else (2, 10))
    want = _outcome(lambda: ref.tensor_split(r, indices, dim))
    got = _outcome(lambda: lucid.tensor_split(x, indices, dim))
    _agree(got, want, lambda: lucid.tensor_split(x, indices, dim))


@pytest.mark.parity
@pytest.mark.parametrize(
    ("op", "shape"),
    [("vsplit", (4, 6)), ("hsplit", (4, 6)), ("hsplit", (6,)), ("dsplit", (2, 3, 6))],
)
@pytest.mark.parametrize("indices", [[1, 9], [5, 2], [-1], []], ids=str)
def test_the_named_splits_slice_as_the_reference_slices(
    ref: ModuleType, device: str, op: str, shape: tuple[int, ...], indices: list[int]
) -> None:
    x, r = _pair(ref, device, shape)
    want = _outcome(lambda: getattr(ref, op)(r, indices))
    got = _outcome(lambda: getattr(lucid, op)(x, indices))
    assert got == want


@pytest.mark.parity
def test_overlapping_slices_each_carry_their_gradient(ref: ModuleType, device: str) -> None:
    """``[5, 2]`` reads ``x[2:5]`` twice, so those elements get two gradients."""
    x = lucid.arange(10, dtype=lucid.float32, device=device).requires_grad_(True)
    r = ref.arange(10, dtype=ref.float32).requires_grad_(True)
    for lib, t in ((lucid, x), (ref, r)):
        pieces = lib.tensor_split(t, [5, 2])
        sum(((i + 1) * p).sum() for i, p in enumerate(pieces)).backward()
    assert x.grad is not None and r.grad is not None
    assert x.grad.tolist() == r.grad.tolist()


@pytest.mark.parametrize(
    ("n", "sizes"), [(6, [2, 3]), (4, [2, 3]), (6, [2, 5]), (6, [-1, 7])], ids=str
)
def test_a_size_list_that_misses_the_axis_is_refused(device: str, n: int, sizes: list[int]) -> None:
    """The defect itself, held without the reference installed."""
    x = lucid.zeros(n, device=device)
    with pytest.raises(_C_engine.InvalidArgument, match="split_with_sizes expects"):
        x.split(sizes)
    with pytest.raises(_C_engine.InvalidArgument, match="split_with_sizes expects"):
        lucid.split(x, sizes)


@pytest.mark.parametrize("sizes", [[2.0, 4.0], [True, 5], True], ids=str)
def test_a_size_that_is_not_an_integer_is_refused(sizes: object) -> None:
    """A float or a bool was read as an int: ``[2.9, 4.1]`` passed as ``[2, 4]``."""
    with pytest.raises(TypeError, match="split_size"):
        lucid.zeros(6).split(sizes)


def test_cat_transform_refuses_lengths_that_miss_the_axis(device: str) -> None:
    """``CatTransform`` cuts by ``split``: lengths that miss the axis are refused."""
    transform = CatTransform([ExpTransform(), AffineTransform(0.0, 2.0)], lengths=[2, 3])
    ok = transform(lucid.zeros(5, device=device))
    assert ok.tolist() == [1.0, 1.0, 0.0, 0.0, 0.0]
    with pytest.raises(_C_engine.InvalidArgument, match="sum exactly to 4"):
        transform(lucid.zeros(4, device=device))
    with pytest.raises(_C_engine.InvalidArgument, match="sum exactly to 6"):
        transform.inv(lucid.zeros(6, device=device))
