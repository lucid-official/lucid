"""The index composites answer as the reference does (API-05).

``lucid._ops.composite.indexing`` and ``take_along_dim`` re-implemented
what the indexing owners already decide, and each got a piece wrong:

* an ``int64`` index was narrowed to ``int32`` before the engine checked
  it, so ``2**40`` wrapped to 0 and wrote the first element (LCD-273);
* ``take_along_dim`` did not broadcast and ``masked_scatter`` did not
  broadcast its mask — both answered a silently wrong shape (LCD-163);
* ``index_put`` read a bool mask as the integers 0 and 1 rather than
  through ``_normalize_key`` — ``[5, 5, 0, 0]`` for ``[0, 0, 5, 5]``;
* a ``source`` of another dtype was cast where the reference refuses
  (LCD-324), and ``scatter_reduce`` with ``include_self=False`` raised on
  a bool or complex input.

Each class is pinned across every composite it touches, on both devices,
values and gradients, against the reference.
"""

from collections.abc import Callable
from types import ModuleType

import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid.test._helpers.compare import assert_close

# ── int64 indices reach the engine at full width ────────────────────────────

_WIDE = [2**31, 2**32 + 1, 2**40, -(2**40)]


def _ix(values: list[int], device: str) -> lucid.Tensor:
    return lucid.tensor(values, dtype=lucid.int64, device=device)


def _base(device: str) -> lucid.Tensor:
    return lucid.tensor([1.0, 2.0, 3.0, 4.0], device=device)


def _one(device: str) -> lucid.Tensor:
    return lucid.tensor([9.0], device=device)


_WRITES: dict[str, Callable[[str, int], lucid.Tensor]] = {
    "index_fill": lambda d, b: lucid.index_fill(_base(d), 0, _ix([b], d), 9.0),
    "index_add": lambda d, b: lucid.index_add(_base(d), 0, _ix([b], d), _one(d)),
    "index_copy": lambda d, b: lucid.index_copy(_base(d), 0, _ix([b], d), _one(d)),
    "scatter_reduce_sum": lambda d, b: lucid.scatter_reduce(
        _base(d), 0, _ix([b], d), _one(d), "sum"
    ),
    "scatter_reduce_mean_noself": lambda d, b: lucid.scatter_reduce(
        _base(d), 0, _ix([b], d), _one(d), "mean", include_self=False
    ),
    "scatter_reduce_amax": lambda d, b: lucid.scatter_reduce(
        _base(d), 0, _ix([b], d), _one(d), "amax"
    ),
    "scatter_reduce_prod": lambda d, b: lucid.scatter_reduce(
        _base(d), 0, _ix([b], d), _one(d), "prod"
    ),
    "index_put": lambda d, b: lucid.index_put(_base(d), (_ix([b], d),), _one(d)),
    "index_put_accumulate": lambda d, b: lucid.index_put(
        _base(d), (_ix([b], d),), _one(d), accumulate=True
    ),
    "index_put_": lambda d, b: _base(d).index_put_((_ix([b], d),), _one(d)),
    "put": lambda d, b: lucid.put(_base(d), _ix([b], d), _one(d)),
    "put_accumulate": lambda d, b: lucid.put(
        _base(d), _ix([b], d), _one(d), accumulate=True
    ),
}


@pytest.mark.parametrize("bad", _WIDE, ids=[str(b) for b in _WIDE])
@pytest.mark.parametrize("name", list(_WRITES))
def test_a_wide_index_is_never_wrapped_into_range(
    device: str, name: str, bad: int
) -> None:
    write = _WRITES[name]
    if device == "cpu":
        with pytest.raises(IndexError):
            write(device, bad)
        return
    # Metal drops an out-of-range write (AE-4); wrapped, it landed at 0.
    out = write(device, bad)
    assert out.tolist() == [1.0, 2.0, 3.0, 4.0]


@pytest.mark.parametrize("bad", _WIDE, ids=[str(b) for b in _WIDE])
def test_take_along_dim_never_wraps_a_wide_index(device: str, bad: int) -> None:
    x = lucid.tensor([[1.0, 2.0, 3.0]], device=device)
    if device == "cpu":
        with pytest.raises(IndexError):
            lucid.take_along_dim(x, _ix([[bad]], device), 1)
        return
    assert lucid.take_along_dim(x, _ix([[bad]], device), 1).tolist() != [[1.0]]


# ── a source of another dtype is refused ────────────────────────────────────

# name -> the write, given the framework module ``m``; each writes ``s``
# into ``x``.  The same call runs on both frameworks.
_SOURCED: dict[str, Callable[..., object]] = {
    "index_add": lambda m, x, s: m.index_add(x, 0, m.tensor([0]), s),
    "index_copy": lambda m, x, s: m.index_copy(x, 0, m.tensor([0]), s),
    "scatter_reduce": lambda m, x, s: m.scatter_reduce(x, 0, m.tensor([0]), s, "sum"),
    "index_put": lambda m, x, s: m.index_put(x, (m.tensor([0]),), s),
    "put": lambda m, x, s: m.put(x, m.tensor([0]), s),
    "masked_scatter": lambda m, x, s: m.masked_scatter(x, m.tensor([True, False]), s),
}

_DTYPE_PAIRS = [("float32", "float16"), ("int32", "int64"), ("float32", "int32")]


@pytest.mark.parity
@pytest.mark.parametrize(
    ("self_dt", "src_dt"), _DTYPE_PAIRS, ids=[f"{a}<-{b}" for a, b in _DTYPE_PAIRS]
)
@pytest.mark.parametrize("name", list(_SOURCED))
def test_a_source_of_another_dtype_is_refused(
    ref: ModuleType, device: str, name: str, self_dt: str, src_dt: str
) -> None:
    write = _SOURCED[name]
    with pytest.raises(RuntimeError):
        write(
            ref,
            ref.zeros(2, dtype=getattr(ref, self_dt)),
            ref.ones(1, dtype=getattr(ref, src_dt)),
        )
    x = lucid.zeros(2, dtype=getattr(lucid, self_dt), device=device)
    s = lucid.ones(1, dtype=getattr(lucid, src_dt), device=device)
    with pytest.raises(_C_engine.DtypeMismatch, match=name):
        write(_OnDevice(device), x, s)


class _OnDevice(ModuleType):
    """``lucid`` with ``tensor`` placing its result on one device."""

    def __init__(self, device: str) -> None:
        super().__init__("lucid_on_device")
        self._device = device

    def tensor(self, values: object, **kwargs: object) -> lucid.Tensor:
        return lucid.tensor(values, device=self._device, **kwargs)

    def __getattr__(self, name: str) -> object:
        return getattr(lucid, name)


def test_a_matching_source_still_writes(device: str) -> None:
    x = lucid.zeros(3, dtype=lucid.int32, device=device)
    src = lucid.tensor([7, 8], dtype=lucid.int32, device=device)
    idx = _ix([2, 0], device)
    assert lucid.index_add(x, 0, idx, src).tolist() == [8, 0, 7]
    assert lucid.index_copy(x, 0, idx, src).tolist() == [8, 0, 7]
    assert lucid.put(x, idx, src).tolist() == [8, 0, 7]
    assert lucid.index_put(x, (idx,), src).tolist() == [8, 0, 7]


# ── broadcasting and keys: values and gradients against the reference ──────

# Each case builds its operands with ``m.tensor`` and returns the result and
# the operands that require grad; the same case runs on both frameworks.
_Case = Callable[[ModuleType], tuple[object, list[object]]]


def _take_along_broadcast_index(m: ModuleType) -> tuple[object, list[object]]:
    x = m.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], requires_grad=True)
    return m.take_along_dim(x, m.tensor([[1]]), 1), [x]


def _take_along_broadcast_both(m: ModuleType) -> tuple[object, list[object]]:
    x = m.tensor([[[0.0, 1.0]], [[2.0, 3.0]], [[4.0, 5.0]]], requires_grad=True)
    return m.take_along_dim(x, m.tensor([[[1], [0]]]), 2), [x]


def _take_along_flat(m: ModuleType) -> tuple[object, list[object]]:
    x = m.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], requires_grad=True)
    return m.take_along_dim(x, m.tensor([[1, 5], [5, 0]])), [x]


def _take_along_negative(m: ModuleType) -> tuple[object, list[object]]:
    x = m.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], requires_grad=True)
    return m.take_along_dim(x, m.tensor([[-1], [-3]]), 1), [x]


def _masked_scatter_row_mask(m: ModuleType) -> tuple[object, list[object]]:
    x = m.tensor([[0.5, 1.5], [2.5, 3.5]], requires_grad=True)
    src = m.tensor([10.0, 20.0, 30.0, 40.0], requires_grad=True)
    return m.masked_scatter(x, m.tensor([True, False]), src), [x, src]


def _masked_scatter_wider_mask(m: ModuleType) -> tuple[object, list[object]]:
    x = m.tensor([0.5, 1.5], requires_grad=True)
    src = m.tensor([[10.0, 20.0], [30.0, 40.0]], requires_grad=True)
    mask = m.tensor([[True, False], [False, True]])
    return m.masked_scatter(x, mask, src), [x, src]


def _masked_scatter_attention_mask(m: ModuleType) -> tuple[object, list[object]]:
    x = m.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], requires_grad=True)
    src = m.tensor([7.0, 8.0, 9.0, 10.0, 11.0, 12.0], requires_grad=True)
    mask = m.tensor([[[False, True, True]]])
    return m.masked_scatter(x, mask, src), [x, src]


def _masked_scatter_nothing(m: ModuleType) -> tuple[object, list[object]]:
    x = m.tensor([0.5, 1.5], requires_grad=True)
    src = m.tensor([10.0], requires_grad=True)
    return m.masked_scatter(x, m.tensor([False, False]), src), [x, src]


def _index_put_mask(m: ModuleType) -> tuple[object, list[object]]:
    x = m.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    v = m.tensor(5.0, requires_grad=True)
    return m.index_put(x, (m.tensor([False, False, True, True]),), v), [x, v]


def _index_put_mask_accumulate(m: ModuleType) -> tuple[object, list[object]]:
    x = m.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    v = m.tensor([5.0, 6.0], requires_grad=True)
    mask = m.tensor([True, False, False, True])
    return m.index_put(x, (mask,), v, accumulate=True), [x, v]


def _index_put_2d_mask(m: ModuleType) -> tuple[object, list[object]]:
    x = m.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    v = m.tensor([7.0, 8.0], requires_grad=True)
    mask = m.tensor([[True, False], [False, True]])
    return m.index_put(x, (mask,), v), [x, v]


def _index_put_broadcast_indices(m: ModuleType) -> tuple[object, list[object]]:
    x = m.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    v = m.tensor([[9.0, 8.0]], requires_grad=True)
    rows = m.tensor([[0], [1]])
    cols = m.tensor([[0, 2]])
    return m.index_put(x, (rows, cols), v), [x, v]


def _index_put_duplicates_accumulate(m: ModuleType) -> tuple[object, list[object]]:
    x = m.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    v = m.tensor([5.0, 6.0, 7.0], requires_grad=True)
    return m.index_put(x, (m.tensor([1, 1, 3]),), v, accumulate=True), [x, v]


def _index_put_rows(m: ModuleType) -> tuple[object, list[object]]:
    x = m.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)
    v = m.tensor([9.0, 8.0], requires_grad=True)
    return m.index_put(x, (m.tensor([2, 0]),), v), [x, v]


def _index_add_rows(m: ModuleType) -> tuple[object, list[object]]:
    x = m.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)
    src = m.tensor([[10.0, 20.0], [30.0, 40.0]], requires_grad=True)
    return m.index_add(x, 0, m.tensor([2, 2]), src, alpha=2.0), [x, src]


def _index_copy_cols(m: ModuleType) -> tuple[object, list[object]]:
    x = m.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    src = m.tensor([[10.0], [20.0]], requires_grad=True)
    return m.index_copy(x, 1, m.tensor([1]), src), [x, src]


def _index_fill_negative(m: ModuleType) -> tuple[object, list[object]]:
    x = m.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    return m.index_fill(x, 1, m.tensor([-1, 0]), -2.0), [x]


def _put_negative(m: ModuleType) -> tuple[object, list[object]]:
    x = m.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    src = m.tensor([[9.0, 8.0]], requires_grad=True)
    return m.put(x, m.tensor([[-1, 0]]), src), [x, src]


_CASES: dict[str, _Case] = {
    "take_along_dim-broadcast-index": _take_along_broadcast_index,
    "take_along_dim-broadcast-both": _take_along_broadcast_both,
    "take_along_dim-dim-none": _take_along_flat,
    "take_along_dim-negative": _take_along_negative,
    "masked_scatter-row-mask": _masked_scatter_row_mask,
    "masked_scatter-wider-mask": _masked_scatter_wider_mask,
    "masked_scatter-attention-mask": _masked_scatter_attention_mask,
    "masked_scatter-nothing": _masked_scatter_nothing,
    "index_put-mask": _index_put_mask,
    "index_put-mask-accumulate": _index_put_mask_accumulate,
    "index_put-2d-mask": _index_put_2d_mask,
    "index_put-broadcast-indices": _index_put_broadcast_indices,
    "index_put-duplicates-accumulate": _index_put_duplicates_accumulate,
    "index_put-rows": _index_put_rows,
    "index_add-rows": _index_add_rows,
    "index_copy-cols": _index_copy_cols,
    "index_fill-negative": _index_fill_negative,
    "put-negative": _put_negative,
}


@pytest.mark.parity
@pytest.mark.parametrize("name", list(_CASES))
def test_values_and_gradients_match_the_reference(
    ref: ModuleType, device: str, name: str
) -> None:
    ours, our_leaves = _CASES[name](_OnDevice(device))
    theirs, their_leaves = _CASES[name](ref)
    assert isinstance(ours, lucid.Tensor)
    assert tuple(ours.shape) == tuple(theirs.shape)
    assert ours.device.type == device
    assert_close(ours, theirs.detach().numpy())
    # A weight per element, so each gradient entry is told apart.
    weight = [float(k + 1) for k in range(ours.numel())]
    (ours * lucid.tensor(weight, device=device).reshape(ours.shape)).sum().backward()
    (theirs * ref.tensor(weight).reshape(theirs.shape)).sum().backward()
    for leaf, their_leaf in zip(our_leaves, their_leaves, strict=True):
        assert isinstance(leaf, lucid.Tensor) and leaf.grad is not None
        assert_close(leaf.grad, their_leaf.grad.numpy())


def test_masked_scatter_never_returns_its_input(device: str) -> None:
    x = lucid.zeros(2, device=device)
    out = lucid.masked_scatter(x, lucid.tensor([False, False], device=device), x)
    assert out is not x
    out[0] = 1.0
    assert x.tolist() == [0.0, 0.0]


def test_masked_scatter_refuses_a_short_source_and_a_non_bool_mask(
    device: str,
) -> None:
    x = lucid.zeros(3, device=device)
    with pytest.raises(_C_engine.ShapeMismatch, match="selects 2"):
        lucid.masked_scatter(
            x, lucid.tensor([True, True, False], device=device), _one(device)
        )
    with pytest.raises(_C_engine.DtypeMismatch, match="bool"):
        lucid.masked_scatter(x, lucid.tensor([1, 0, 0], device=device), x)


def test_take_along_dim_refuses_indices_of_another_rank(device: str) -> None:
    x = lucid.zeros(3, 2, device=device)
    with pytest.raises(_C_engine.ShapeMismatch, match="same number of dims"):
        lucid.take_along_dim(x, _ix([1], device), 1)


def test_put_refuses_a_source_of_another_size(device: str) -> None:
    with pytest.raises(IndexError, match="same number of elements"):
        lucid.put(_base(device), _ix([0, 1], device), _one(device))


# ── scatter_reduce on bool and complex ──────────────────────────────────────

_ODD_DTYPES: dict[str, tuple[str, list[object], list[int], list[object]]] = {
    "bool-amax": ("amax", [True, False], [0], [False]),
    "bool-amin": ("amin", [True, False], [1], [True]),
    "bool-sum": ("sum", [True, False], [1, 1], [True, True]),
    "bool-prod": ("prod", [True, False], [1, 1], [True, True]),
    "complex-prod": ("prod", [1 + 1j, 2 + 0j], [0], [2 + 0j]),
    "complex-sum": ("sum", [1 + 1j, 2 + 0j], [0, 0], [2 + 0j, 4j]),
    "complex-mean": ("mean", [1 + 1j, 2 + 0j], [0, 0], [2 + 0j, 4j]),
}


@pytest.mark.parity
@pytest.mark.parametrize("include_self", [False, True])
@pytest.mark.parametrize("name", list(_ODD_DTYPES))
def test_scatter_reduce_takes_bool_and_complex(
    ref: ModuleType, device: str, name: str, include_self: bool
) -> None:
    if name.startswith("complex") and device == "metal":
        pytest.skip("Metal's scatter kernels hold no complex dtype")
    reduce, base, index, src = _ODD_DTYPES[name]
    want = ref.scatter_reduce(
        ref.tensor(base),
        0,
        ref.tensor(index),
        ref.tensor(src),
        reduce,
        include_self=include_self,
    )
    got = lucid.scatter_reduce(
        lucid.tensor(base, device=device),
        0,
        _ix(index, device),
        lucid.tensor(src, device=device),
        reduce,
        include_self=include_self,
    )
    assert got.tolist() == pytest.approx(want.tolist())


def test_complex_amax_is_refused_without_self(device: str) -> None:
    if device == "metal":
        pytest.skip("Metal's scatter kernels hold no complex dtype")
    z = lucid.tensor([1 + 1j, 2 + 0j], device=device)
    with pytest.raises(NotImplementedError, match="no order"):
        lucid.scatter_reduce(z, 0, _ix([0], device), z[:1], "amax", include_self=False)


# ── deterministic duplicate-index overwrite (last write wins) ──────────────

_DUP_OPS: dict[str, Callable[[ModuleType, str | None], object]] = {
    "scatter": lambda m, dev: m.scatter(
        m.zeros(4, **({"device": dev} if dev else {})),
        0,
        m.tensor([2, 0, 2, 1, 2, 0] * 512, **({"device": dev} if dev else {})),
        m.arange(1.0, 3073.0, **({"device": dev} if dev else {})),
    ),
    "scatter_": lambda m, dev: m.zeros(
        4, **({"device": dev} if dev else {})
    ).scatter_(
        0,
        m.tensor([2, 0, 2, 1, 2, 0] * 512, **({"device": dev} if dev else {})),
        m.arange(1.0, 3073.0, **({"device": dev} if dev else {})),
    ),
    "scatter-2d-dim1": lambda m, dev: m.scatter(
        m.zeros(3, 4, **({"device": dev} if dev else {})),
        1,
        m.tensor([[2, 0, 2, 1, 2, 0] * 256] * 3, **({"device": dev} if dev else {})),
        m.arange(1.0, 4609.0, **({"device": dev} if dev else {})).reshape(3, 1536),
    ),
    "index_copy": lambda m, dev: m.index_copy(
        m.zeros(4, 3, **({"device": dev} if dev else {})),
        0,
        m.tensor([2, 0, 2, 1, 2, 0] * 256, **({"device": dev} if dev else {})),
        m.arange(1.0, 4609.0, **({"device": dev} if dev else {})).reshape(1536, 3),
    ),
    "index_copy_": lambda m, dev: m.zeros(
        3, 4, **({"device": dev} if dev else {})
    ).index_copy_(
        1,
        m.tensor([2, 0, 2, 1, 2, 0] * 256, **({"device": dev} if dev else {})),
        m.arange(1.0, 4609.0, **({"device": dev} if dev else {})).reshape(3, 1536),
    ),
    "put": lambda m, dev: m.put(
        m.zeros(2, 2, **({"device": dev} if dev else {})),
        m.tensor([2, 0, 2, 1, 2, 0] * 512, **({"device": dev} if dev else {})),
        m.arange(1.0, 3073.0, **({"device": dev} if dev else {})),
    ),
    "put_": lambda m, dev: m.zeros(
        2, 2, **({"device": dev} if dev else {})
    ).put_(
        m.tensor([2, 0, 2, 1, 2, 0] * 512, **({"device": dev} if dev else {})),
        m.arange(1.0, 3073.0, **({"device": dev} if dev else {})),
    ),
    "index_put": lambda m, dev: m.index_put(
        m.zeros(4, 3, **({"device": dev} if dev else {})),
        (m.tensor([2, 0, 2, 1, 2, 0] * 256, **({"device": dev} if dev else {})),),
        m.arange(1.0, 4609.0, **({"device": dev} if dev else {})).reshape(1536, 3),
    ),
    "index_put_": lambda m, dev: m.zeros(
        4, 3, **({"device": dev} if dev else {})
    ).index_put_(
        (m.tensor([2, 0, 2, 1, 2, 0] * 256, **({"device": dev} if dev else {})),),
        m.arange(1.0, 4609.0, **({"device": dev} if dev else {})).reshape(1536, 3),
    ),
}


@pytest.mark.parity
@pytest.mark.parametrize("name", list(_DUP_OPS))
def test_duplicate_index_overwrite_is_last_write_wins_under_deterministic(
    ref: ModuleType, device: str, name: str
) -> None:
    fn = _DUP_OPS[name]
    prev = lucid.are_deterministic_algorithms_enabled()
    lucid.use_deterministic_algorithms(True)
    try:
        got = fn(lucid, device)
        want = fn(ref, None)
        assert isinstance(got, lucid.Tensor)
        assert got.tolist() == want.tolist()
    finally:
        lucid.use_deterministic_algorithms(prev)
