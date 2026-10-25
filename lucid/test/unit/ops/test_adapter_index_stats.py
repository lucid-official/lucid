"""The index and statistics adapters answer as the reference does (LCD-336).

Each one used to re-decide a rule its engine op or the reference already
fixes: ``take`` took a 1-D index only, ``masked_fill`` passed an
unbroadcast mask to a kernel that walks both buffers side by side,
``where(cond)`` and ``nonzero(as_tuple=True)`` had no spelling at all,
``histc`` counted NaN in the first bin and widened a one-point range to
``[v, v + 1]``, ``unique`` took none of its options, and ``scatter`` /
``scatter_add`` narrowed an ``int64`` index to ``int32`` and cast a
``src`` of another dtype.  Every case runs on both devices against the
reference.
"""

from types import ModuleType
from typing import Protocol

import pytest

import lucid
from lucid._C import engine as _C_engine

NAN = float("nan")


class _Array(Protocol):
    """What the comparison reads off a reference tensor."""

    shape: tuple[int, ...]
    dtype: object

    def reshape(self, *shape: int) -> _Array: ...

    def tolist(self) -> object: ...


def _same(got: object, want: _Array | tuple[_Array, ...]) -> None:
    """``got`` (Lucid) holds ``want``'s (reference) values, dtype, shape and nesting."""
    if isinstance(want, tuple):
        assert isinstance(got, tuple) and len(got) == len(want)
        for g, w in zip(got, want, strict=True):
            _same(g, w)
        return
    assert isinstance(got, lucid.Tensor)
    assert tuple(got.shape) == tuple(want.shape)
    assert str(got.dtype).rsplit(".")[-1] == str(want.dtype).rsplit(".")[-1]
    flat = want.reshape(-1).tolist()
    assert got.reshape(-1).tolist() == pytest.approx(flat, nan_ok=True)


@pytest.mark.parametrize(
    "index", [[[1, 2], [3, -1]], 4, [], [[[0]], [[5]]], [0, -6]], ids=str
)
def test_take_answers_in_the_index_shape(
    ref: ModuleType, device: str, index: object
) -> None:
    data = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
    got = lucid.take(
        lucid.tensor(data, device=device),
        lucid.tensor(index, dtype=lucid.int64, device=device),
    )
    _same(got, ref.take(ref.tensor(data), ref.tensor(index, dtype=ref.int64)))


@pytest.mark.parametrize(
    "mask_shape", [(2, 1, 1, 4), (), (4,), (3, 1, 4), (2, 3, 4, 4)], ids=str
)
def test_masked_fill_broadcasts_the_mask(
    ref: ModuleType, device: str, mask_shape: tuple[int, ...]
) -> None:
    x = [float(i) for i in range(2 * 3 * 4 * 4)]
    n = 1
    for s in mask_shape:
        n *= s
    bits = [i % 3 == 0 for i in range(n)]
    want = (
        ref.tensor(x)
        .reshape(2, 3, 4, 4)
        .masked_fill(ref.tensor(bits).reshape(mask_shape), -1.0)
    )
    got = lucid.masked_fill(
        lucid.tensor(x, device=device).reshape(2, 3, 4, 4),
        lucid.tensor(bits, device=device).reshape(mask_shape),
        -1.0,
    )
    _same(got, want)


def test_masked_fill_broadcasts_the_input_too(ref: ModuleType, device: str) -> None:
    want = ref.zeros(4).masked_fill(ref.ones(2, 4, dtype=ref.bool), 1.0)
    got = lucid.zeros(4, device=device).masked_fill(
        lucid.ones(2, 4, dtype=lucid.bool, device=device), 1.0
    )
    _same(got, want)


def test_masked_fill_refuses_a_mask_that_does_not_broadcast(device: str) -> None:
    with pytest.raises(ValueError, match="do not broadcast"):
        lucid.masked_fill(
            lucid.zeros(4, device=device),
            lucid.ones(3, dtype=lucid.bool, device=device),
            1.0,
        )


@pytest.mark.parametrize(
    "cond", [[[1, 0], [0, 1]], [0, 0, 0], [[True]], 3.0, 0.0], ids=str
)
def test_where_alone_and_nonzero_as_tuple(
    ref: ModuleType, device: str, cond: object
) -> None:
    c = lucid.tensor(cond, device=device)
    want = ref.nonzero(ref.tensor(cond), as_tuple=True)
    _same(lucid.where(c), ref.where(ref.tensor(cond)))
    _same(lucid.nonzero(c, as_tuple=True), want)
    _same(c.nonzero(as_tuple=True), want)
    _same(lucid.nonzero(c), ref.nonzero(ref.tensor(cond)))


def test_where_with_one_branch_is_refused(device: str) -> None:
    c = lucid.tensor([True, False], device=device)
    with pytest.raises(TypeError, match="cond alone"):
        lucid.where(c, lucid.ones(2, device=device))


@pytest.mark.parametrize(
    ("data", "bins", "lo", "hi"),
    [
        ([NAN, 1.0, 2.0], 2, 0.0, 2.0),
        ([3.0, 3.0], 4, 0.0, 0.0),
        ([3.0], 3, 5.0, 5.0),
        ([1.0, 3.0], 4, 2.0, 2.0),
        ([float("inf"), 1.0, 2.0], 2, 0.0, 2.0),
        ([-float("inf"), NAN, 0.5], 2, 0.0, 1.0),
        ([5.0, -1.0, 1.0], 2, 0.0, 2.0),
        ([], 3, 0.0, 0.0),
        ([0.25, 0.5, 0.75, 1.0], 4, 0.0, 0.0),
    ],
    ids=str,
)
def test_histc(
    ref: ModuleType, device: str, data: list[float], bins: int, lo: float, hi: float
) -> None:
    got = lucid.histc(lucid.tensor(data, device=device), bins, lo, hi)
    _same(got, ref.histc(ref.tensor(data), bins, lo, hi))


@pytest.mark.parametrize(
    ("data", "lo", "hi"),
    [
        ([NAN, 1.0], 0.0, 0.0),
        ([float("inf"), 1.0], 0.0, 0.0),
        ([1.0], 0.0, float("inf")),
    ],
    ids=str,
)
def test_histc_refuses_a_range_that_is_not_finite(
    device: str, data: list[float], lo: float, hi: float
) -> None:
    with pytest.raises(ValueError, match="not finite"):
        lucid.histc(lucid.tensor(data, device=device), 2, lo, hi)


def test_histc_refuses_min_above_max(device: str) -> None:
    with pytest.raises(ValueError, match="at least min"):
        lucid.histc(lucid.tensor([1.0], device=device), 2, 3.0, 1.0)


_UNIQUE_DATA = {
    "ints": [3, 1, 3, 2, 1, 1, 3, 2, 2, 1, 0, 3, 3, 1],
    "matrix": [[1, 2], [1, 2], [0, 5], [1, 2], [0, 4]],
    "floats": [1.0, -0.0, 0.0, -1.0, 2.5, 1.0],
    "scalar": 5,
    "empty": [],
}


@pytest.mark.parametrize("name", list(_UNIQUE_DATA))
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("counts", [False, True])
def test_unique_flat(
    ref: ModuleType, device: str, name: str, inverse: bool, counts: bool
) -> None:
    data = _UNIQUE_DATA[name]
    got = lucid.unique(
        lucid.tensor(data, device=device),
        return_inverse=inverse,
        return_counts=counts,
    )
    want = ref.unique(ref.tensor(data), return_inverse=inverse, return_counts=counts)
    _same(got, want)


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_unique_along_a_dim(ref: ModuleType, device: str, dim: int) -> None:
    data = [[[1, 2], [1, 2]], [[0, 5], [0, 5]], [[1, 2], [1, 2]]]
    got = lucid.unique(
        lucid.tensor(data, device=device),
        dim=dim,
        return_inverse=True,
        return_counts=True,
    )
    want = ref.unique(
        ref.tensor(data), dim=dim, return_inverse=True, return_counts=True
    )
    _same(got, want)


@pytest.mark.parametrize(("shape", "dim"), [((0, 3), 0), ((2, 0, 3), 1)], ids=str)
def test_unique_along_an_empty_dim(
    ref: ModuleType, device: str, shape: tuple[int, ...], dim: int
) -> None:
    got = lucid.unique(
        lucid.zeros(*shape, device=device),
        dim=dim,
        return_inverse=True,
        return_counts=True,
    )
    want = ref.unique(
        ref.zeros(*shape), dim=dim, return_inverse=True, return_counts=True
    )
    _same(got, want)


def test_unique_sorted_false_still_answers(ref: ModuleType, device: str) -> None:
    got = lucid.unique(lucid.tensor([3, 1, 3], device=device), sorted=False)
    assert sorted(got.tolist()) == sorted(ref.unique(ref.tensor([3, 1, 3])).tolist())


@pytest.mark.parametrize("name", list(_UNIQUE_DATA))
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("counts", [False, True])
def test_unique_consecutive_flat(
    ref: ModuleType, device: str, name: str, inverse: bool, counts: bool
) -> None:
    data = _UNIQUE_DATA[name]
    x = lucid.tensor(data, device=device)
    rx = ref.tensor(data)
    want = ref.unique_consecutive(rx, return_inverse=inverse, return_counts=counts)
    _same(
        lucid.unique_consecutive(x, return_inverse=inverse, return_counts=counts),
        want,
    )
    _same(x.unique_consecutive(return_inverse=inverse, return_counts=counts), want)
    _same(
        x.unique(return_inverse=inverse, return_counts=counts),
        ref.unique(rx, return_inverse=inverse, return_counts=counts),
    )


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_unique_consecutive_along_a_dim(ref: ModuleType, device: str, dim: int) -> None:
    data = [[[1, 2], [1, 2]], [[1, 2], [1, 2]], [[0, 5], [0, 5]], [[1, 2], [1, 2]]]
    x = lucid.tensor(data, device=device)
    want = ref.unique_consecutive(
        ref.tensor(data), dim=dim, return_inverse=True, return_counts=True
    )
    _same(
        lucid.unique_consecutive(x, dim=dim, return_inverse=True, return_counts=True),
        want,
    )
    _same(x.unique_consecutive(dim=dim, return_inverse=True, return_counts=True), want)


@pytest.mark.parametrize("op", ["unique", "unique_consecutive"])
@pytest.mark.parametrize("as_method", [False, True])
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("counts", [False, True])
@pytest.mark.parametrize("dim", [None, 0, 1])
def test_unique_propagates_requires_grad_and_refuses_backward(
    device: str,
    op: str,
    as_method: bool,
    inverse: bool,
    counts: bool,
    dim: int | None,
) -> None:
    x = lucid.tensor(
        [[3.0, 1.0], [3.0, 1.0], [2.0, 1.0]], device=device, requires_grad=True
    )
    fn = getattr(x, op) if as_method else lambda **kw: getattr(lucid, op)(x, **kw)
    out = fn(return_inverse=inverse, return_counts=counts, dim=dim)
    if isinstance(out, tuple):
        values = out[0]
        for aux in out[1:]:
            assert not aux.requires_grad
            assert aux.grad_fn is None
    else:
        values = out
    assert values.requires_grad
    assert values.grad_fn is not None
    with pytest.raises(
        NotImplementedError, match=rf"the derivative for '{op}' is not implemented"
    ):
        values.sum().backward()

    with lucid.no_grad():
        out_ng = fn(return_inverse=inverse, return_counts=counts, dim=dim)
    values_ng = out_ng[0] if isinstance(out_ng, tuple) else out_ng
    assert not values_ng.requires_grad
    assert values_ng.grad_fn is None


@pytest.mark.parametrize("op", ["unique", "unique_consecutive"])
def test_unique_refuses_zero_sized_unselected_dim_and_0d_dim(
    device: str, op: str
) -> None:
    fn = getattr(lucid, op)
    with pytest.raises(IndexError, match="tensor has no dimensions"):
        fn(lucid.tensor(5.0, device=device), dim=0)
    with pytest.raises(RuntimeError, match="0 sized dimensions"):
        fn(lucid.zeros(2, 0, device=device), dim=0)


@pytest.mark.parametrize("op", ["scatter", "scatter_add"])
def test_scatter_passes_the_index_at_full_width(device: str, op: str) -> None:
    base = lucid.zeros(4, device=device)
    index = lucid.tensor([2**40], dtype=lucid.int64, device=device)
    call = lambda: getattr(base, op)(0, index, lucid.ones(1, device=device))
    if device == "cpu":
        with pytest.raises(IndexError):
            call()
    else:
        # Metal reads no index back: one past the axis writes nothing, and
        # never the slot a narrowed 2**40 named (slot 0).
        assert call().tolist() == [0.0, 0.0, 0.0, 0.0]


@pytest.mark.parametrize("op", ["scatter", "scatter_add"])
def test_scatter_refuses_a_src_of_another_dtype(ref: ModuleType, op: str) -> None:
    index = [1, 3]
    with pytest.raises(RuntimeError):
        getattr(ref.zeros(4), op)(0, ref.tensor(index), ref.ones(2, dtype=ref.float64))
    with pytest.raises(_C_engine.DtypeMismatch):
        getattr(lucid.zeros(4), op)(
            0, lucid.tensor(index), lucid.ones(2, dtype=lucid.float64)
        )


# The reference warns that ``reduce=`` with a tensor src is on its way out.
@pytest.mark.filterwarnings("ignore:The reduce argument:UserWarning")
def test_scatter_reduce_modes_and_a_number(ref: ModuleType, device: str) -> None:
    index = [[0, 1, 0], [2, 2, 1]]
    src = [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]
    for reduce in ("add", "multiply"):
        want = ref.ones(3, 3).scatter(
            0, ref.tensor(index), ref.tensor(src), reduce=reduce
        )
        got = lucid.ones(3, 3, device=device).scatter(
            0,
            lucid.tensor(index, device=device),
            lucid.tensor(src, device=device),
            reduce=reduce,
        )
        _same(got, want)
    want = ref.zeros(3, 3).scatter(0, ref.tensor(index), 9.0)
    got = lucid.zeros(3, 3, device=device).scatter(
        0, lucid.tensor(index, device=device), 9.0
    )
    _same(got, want)
    with pytest.raises(ValueError, match="'add' or 'multiply'"):
        lucid.zeros(3, device=device).scatter(
            0,
            lucid.tensor([0], device=device),
            lucid.ones(1, device=device),
            reduce="max",
        )


_SCATTERS = {
    "scatter": lambda b, i, s: lucid.scatter(b, 0, i, s),
    "scatter_add": lambda b, i, s: lucid.scatter_add(b, 0, i, s),
    "reduce_add": lambda b, i, s: lucid.scatter(b, 0, i, s, reduce="add"),
    "reduce_multiply": lambda b, i, s: lucid.scatter(b, 0, i, s, reduce="multiply"),
    "scatter_reduce_amax": lambda b, i, s: lucid.scatter_reduce(b, 0, i, s, "amax"),
}


@pytest.mark.parametrize("index_dtype", ["int64", "int32"])
@pytest.mark.parametrize("name", list(_SCATTERS))
def test_an_index_written_after_forward_is_refused_in_backward(
    device: str, name: str, index_dtype: str
) -> None:
    # The index now reaches the node as the caller's own tensor — no
    # narrowing copy in between — so a write into it after forward would
    # move the gradient to other slots.  The node's version guard refuses
    # it instead (LCD-320, LCD-351).
    base = lucid.ones(4, device=device, requires_grad=True)
    src = lucid.tensor([10.0, 20.0], device=device, requires_grad=True)
    index = lucid.tensor([0, 1], dtype=getattr(lucid, index_dtype), device=device)
    out = _SCATTERS[name](base, index, src)
    index.add_(2)
    with pytest.raises(_C_engine.VersionMismatch):
        out.sum().backward()


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_half_data_dependent_ops_match_reference(
    ref: ModuleType, device: str, dtype_name: str
) -> None:
    dt = getattr(lucid, dtype_name)
    ref_dt = getattr(ref, dtype_name)
    vals = [0.5, 1.5, 2.5, 0.0, -0.0, 1.5]
    x = lucid.tensor(vals, dtype=dt, device=device)
    rx = ref.tensor(vals, dtype=ref_dt)

    # histc: explicit range, auto-range, and direct _C_engine.histc auto-range
    _same(
        lucid.histc(x, bins=4, min=0.0, max=3.0),
        ref.histc(rx, bins=4, min=0.0, max=3.0),
    )
    _same(lucid.histc(x, bins=4), ref.histc(rx, bins=4))
    _same(
        lucid._dispatch._wrap(_C_engine.histc(x._impl, 4, 0.0, 0.0)),
        ref.histc(rx, bins=4),
    )

    # nonzero (2-D matrix + as_tuple=True) and argwhere
    mat = x.reshape(2, 3)
    rmat = rx.reshape(2, 3)
    _same(lucid.nonzero(mat), ref.nonzero(rmat))
    _same(lucid.nonzero(mat, as_tuple=True), ref.nonzero(rmat, as_tuple=True))
    _same(lucid.argwhere(mat), ref.argwhere(rmat))

    # unique, masked_select, bincount(weights=half), searchsorted, bucketize, sort/topk
    _same(
        lucid.unique(x, return_inverse=True, return_counts=True),
        ref.unique(rx, return_inverse=True, return_counts=True),
    )
    mask = lucid.tensor([True, False, True, False, False, True], device=device)
    rmask = ref.tensor([True, False, True, False, False, True])
    _same(lucid.masked_select(x, mask), ref.masked_select(rx, rmask))

    bins_idx = lucid.tensor([0, 1, 1, 2, 0, 3], dtype=lucid.int64, device=device)
    rbins_idx = ref.tensor([0, 1, 1, 2, 0, 3], dtype=ref.int64)
    if device == "cpu":
        _same(lucid.bincount(bins_idx, weights=x), ref.bincount(rbins_idx, weights=rx))
    else:
        got_bc = lucid.bincount(bins_idx, weights=x)
        assert got_bc.device.type == "metal"
        assert got_bc.tolist() == pytest.approx(
            ref.bincount(rbins_idx, weights=rx).tolist()
        )

    grid = lucid.tensor([0.0, 1.0, 2.0, 3.0], dtype=dt, device=device)
    rgrid = ref.tensor([0.0, 1.0, 2.0, 3.0], dtype=ref_dt)
    _same(lucid.searchsorted(grid, x), ref.searchsorted(rgrid, rx))
    _same(lucid.bucketize(x, grid), ref.bucketize(rx, rgrid))
    _same(lucid.sort(x), ref.sort(rx))
    _same(lucid.topk(x, 3), ref.topk(rx, 3))

    # histogram2d and histogramdd on half
    h2d, _, _ = lucid.histogram2d(x[:4], x[2:], bins=2)
    assert h2d.sum().item() == 4.0
    hdd, _ = lucid.histogramdd(mat, bins=2)
    assert hdd.sum().item() == 2.0


@pytest.mark.parametrize("dtype_name", ["int8", "int16", "complex64"])
def test_nonzero_supports_narrow_int_and_complex(
    ref: ModuleType, device: str, dtype_name: str
) -> None:
    dt = getattr(lucid, dtype_name)
    ref_dt = getattr(ref, dtype_name)
    vals = [0, 1, 0, -2]
    x = lucid.tensor(vals, dtype=dt, device=device).reshape(2, 2)
    rx = ref.tensor(vals, dtype=ref_dt).reshape(2, 2)
    _same(lucid.nonzero(x), ref.nonzero(rx))


@pytest.mark.parametrize("right", [False, True])
def test_searchsorted_and_bucketize_duplicates_nan_empty_and_2d(
    ref: ModuleType, device: str, right: bool
) -> None:
    # 1-D with duplicates and NaN
    seq_vals = [1.0, 2.0, 2.0, 4.0, NAN]
    q_vals = [-1.0, 1.0, 2.0, 3.0, 5.0, NAN]
    seq = lucid.tensor(seq_vals, device=device)
    q = lucid.tensor(q_vals, device=device)
    rseq = ref.tensor(seq_vals)
    rq = ref.tensor(q_vals)
    _same(
        lucid.searchsorted(seq, q, right=right), ref.searchsorted(rseq, rq, right=right)
    )
    _same(lucid.bucketize(q, seq, right=right), ref.bucketize(rq, rseq, right=right))

    # Empty sequence and empty queries
    empty = lucid.zeros(0, device=device)
    rempty = ref.zeros(0)
    _same(
        lucid.searchsorted(empty, q, right=right),
        ref.searchsorted(rempty, rq, right=right),
    )
    _same(
        lucid.searchsorted(seq, empty, right=right),
        ref.searchsorted(rseq, rempty, right=right),
    )

    # N-D row-wise searchsorted
    seq_2d = lucid.tensor([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]], device=device)
    q_2d = lucid.tensor([[0.0, 3.0, 7.0], [2.5, 4.0, 5.5]], device=device)
    rseq_2d = ref.tensor([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])
    rq_2d = ref.tensor([[0.0, 3.0, 7.0], [2.5, 4.0, 5.5]])
    _same(
        lucid.searchsorted(seq_2d, q_2d, right=right),
        ref.searchsorted(rseq_2d, rq_2d, right=right),
    )


def test_searchsorted_1m_by_1m_completes_in_bounded_memory() -> None:
    # Previously materialised a 1M x 1M broadcast comparison (1 TB -> OOM-killed, LCD-218).
    seq = lucid.arange(1_000_000, dtype=lucid.float32)
    q = lucid.linspace(-10.0, 1_000_010.0, 1_000_000, dtype=lucid.float32)
    out = lucid.searchsorted(seq, q)
    assert out.shape == (1_000_000,)
    assert out.dtype == lucid.int64
    assert int(out[0].item()) == 0
    assert int(out[-1].item()) == 1_000_000
