"""Tensors that read one CPU buffer see each other's in-place writes.

A view family is the set of CPU tensors built over one buffer by
``TensorImpl.make_view`` — reshape-family views, ``.data``, and the views
these tests build directly with ``TensorImpl._make_view``.  Its members
share a version counter, and an in-place write to any of them goes into the
buffer instead of a new storage slot, so the others see it.  Where that
cannot be done safely the write is refused rather than left to diverge.
"""

from collections.abc import Callable

import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid._dispatch import _wrap


def _view(base: lucid.Tensor, *shape: int) -> lucid.Tensor:
    """A dense view of ``base`` under another shape, as reshape makes one."""
    strides: list[int] = []
    step = 1
    for n in reversed(shape):
        strides.append(step)
        step *= n
    impl = _C_engine.TensorImpl._make_view(base._impl, list(shape), strides[::-1], 0)
    return _wrap(impl)


def _flat(t: lucid.Tensor) -> list[float]:
    return [float(v) for v in t.reshape(-1).tolist()]


@pytest.fixture
def x() -> lucid.Tensor:
    return lucid.arange(6).float()


_WRITES: dict[str, Callable[[lucid.Tensor], object]] = {
    "add_": lambda t: t.add_(1.0),
    "mul_": lambda t: t.mul_(2.0),
    "iadd": lambda t: t.__iadd__(1.0),
    "exp_": lambda t: t.exp_(),
    "clamp_": lambda t: t.clamp_(min=2.0),
    "zero_": lambda t: t.zero_(),
    "fill_": lambda t: t.fill_(7.0),
    "copy_": lambda t: t.copy_(lucid.full(t.shape, 7.0)),
    "setitem-all": lambda t: t.__setitem__(..., 7.0),
    "setitem-first": lambda t: t.__setitem__(0, 7.0),
}


def _written(write: Callable[[lucid.Tensor], object], *shape: int) -> list[float]:
    """What ``write`` leaves in a tensor of ``shape`` that nothing else reads."""
    ref = lucid.arange(6).float().reshape(*shape).clone()
    write(ref)
    return _flat(ref)


def test_a_view_joins_its_base_s_family(x: lucid.Tensor) -> None:
    assert not x._impl.is_aliased()
    v = _view(x, 2, 3)
    assert x._impl.is_aliased() and v._impl.is_aliased()
    del v
    assert not x._impl.is_aliased()


@pytest.mark.parametrize("name", list(_WRITES))
def test_a_write_through_a_view_reaches_the_base(x: lucid.Tensor, name: str) -> None:
    v = _view(x, 2, 3)
    _WRITES[name](v)
    assert _flat(x) == _written(_WRITES[name], 2, 3)
    assert _flat(v) == _flat(x)


@pytest.mark.parametrize("name", list(_WRITES))
def test_a_write_through_the_base_reaches_the_view(x: lucid.Tensor, name: str) -> None:
    v = _view(x, 2, 3)
    _WRITES[name](x)
    assert _flat(v) == _written(_WRITES[name], 6)


def test_every_member_s_version_moves_with_a_write(x: lucid.Tensor) -> None:
    v = _view(x, 2, 3)
    before = v._impl.version
    x.add_(1.0)
    assert v._impl.version != before
    before = x._impl.version
    v.mul_(2.0)
    assert x._impl.version != before


def test_a_tensor_whose_views_are_gone_writes_as_before(x: lucid.Tensor) -> None:
    v = _view(x, 2, 3)
    del v
    x.add_(1.0)
    assert _flat(x) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_dot_data_is_a_member_of_the_family() -> None:
    # ``x.data.exp_()`` swapped the alias's slot and left ``x`` alone, and
    # ``x.data.add_(1)`` was refused outright; both now reach ``x``.
    x = lucid.zeros(3)
    x.data.add_(1.0)
    assert _flat(x) == [1.0, 1.0, 1.0]
    x.data.mul_(2.0)
    x.data.exp_()
    assert _flat(x) == pytest.approx([7.389056] * 3)


def test_dot_data_keeps_a_view_s_offset(x: lucid.Tensor) -> None:
    v = _wrap(_C_engine.TensorImpl._make_view(x._impl, [3], [1], 2))
    assert _flat(v.data) == [2.0, 3.0, 4.0]


def test_a_node_holding_the_buffer_stops_the_write(x: lucid.Tensor) -> None:
    # The product saved ``v``'s storage for its backward; writing the new
    # values under it would change what that backward reads.
    v = _view(x, 2, 3)
    w = lucid.ones(2, 3, requires_grad=True)
    y = (v * w).sum()
    with pytest.raises(_C_engine.NotImplementedError, match="also holds that storage"):
        with lucid.no_grad():
            x.add_(1.0)
    assert _flat(x) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    y.backward()
    assert _flat(w.grad) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_a_numpy_array_over_the_buffer_stops_the_write(x: lucid.Tensor) -> None:
    v = _view(x, 2, 3)
    arr = x.numpy()
    with pytest.raises(_C_engine.NotImplementedError, match="also holds that storage"):
        v.add_(1.0)
    assert arr.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_a_write_that_would_change_the_dtype_is_refused() -> None:
    xi = lucid.tensor([0, 1, 2, 3, 4, 5], dtype=lucid.int64)
    v = _view(xi, 2, 3)
    with pytest.raises(RuntimeError, match="can't be cast"):
        xi += 0.5
    assert xi.dtype == lucid.int64
    assert _flat(v) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    xf = lucid.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=lucid.float16)
    vf = _view(xf, 2, 3)
    xf += lucid.ones(6, dtype=lucid.float32)
    assert xf.dtype == lucid.float16
    assert _flat(vf) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def _strided(
    base: lucid.Tensor, shape: list[int], stride: list[int], offset: int = 0
) -> lucid.Tensor:
    return _wrap(_C_engine.TensorImpl._make_view(base._impl, shape, stride, offset))


def test_a_write_through_a_view_at_an_offset_lands_there(x: lucid.Tensor) -> None:
    _strided(x, [3], [1], 2).add_(1.0)
    assert _flat(x) == [0.0, 1.0, 3.0, 4.0, 5.0, 5.0]


def test_a_write_through_a_transposed_view_reaches_every_element(
    x: lucid.Tensor,
) -> None:
    t = _strided(x, [3, 2], [1, 3])  # x viewed as (2, 3), transposed
    t.mul_(10.0)
    assert _flat(x) == [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]


def test_copy_into_a_transposed_view_follows_its_strides(x: lucid.Tensor) -> None:
    t = _strided(x, [3, 2], [1, 3])
    t.copy_(lucid.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    assert _flat(x) == [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
    assert t.tolist() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]


def test_an_expanded_view_is_read_but_not_written(x: lucid.Tensor) -> None:
    e = _strided(x, [2, 3], [0, 1])  # the first three elements, twice
    assert e.tolist() == [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]
    with pytest.raises(Exception, match="overlap"):
        e.add_(1.0)
    assert _flat(x) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_a_transpose_is_a_view_of_its_input() -> None:
    x = lucid.arange(6).float().reshape(2, 3)
    t = x.t()
    assert not t.is_contiguous()
    t.mul_(10.0)
    assert _flat(x) == [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    x.add_(1.0)
    assert t.tolist() == [[1.0, 31.0], [11.0, 41.0], [21.0, 51.0]]


def test_a_recorded_write_through_a_transpose_reaches_the_base_s_gradient() -> None:
    w = lucid.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    h = w * 1.0
    t = h.t()
    t.mul_(2.0)
    assert _flat(h) == [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    (h * lucid.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])).sum().backward()
    assert _flat(w.grad) == [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]


def test_a_recorded_write_through_the_base_reaches_a_transpose_s_gradient() -> None:
    w = lucid.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    h = w * 1.0
    t = h.t()
    h.mul_(3.0)
    (t * lucid.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])).sum().backward()
    assert _flat(w.grad) == [3.0, 6.0, 9.0, 12.0, 15.0, 18.0]


def test_a_constant_written_through_a_strided_column_stops_its_gradient() -> None:
    w = lucid.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    h = w * 1.0
    col = _strided(h, [2], [3], 1)  # column 1
    col.fill_(0.0)
    assert _flat(h) == [1.0, 0.0, 3.0, 4.0, 0.0, 6.0]
    h.sum().backward()
    assert _flat(w.grad) == [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]


def test_copy_into_a_strided_column_sends_the_gradient_to_the_source() -> None:
    w = lucid.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    v = lucid.tensor([7.0, 8.0], requires_grad=True)
    h = w * 1.0
    col = _strided(h, [2], [3], 1)
    col.copy_(v)
    assert _flat(h) == [1.0, 7.0, 3.0, 4.0, 8.0, 6.0]
    (h * lucid.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])).sum().backward()
    assert _flat(w.grad) == [1.0, 0.0, 3.0, 4.0, 0.0, 6.0]
    assert _flat(v.grad) == [2.0, 5.0]


def test_a_strided_splice_differentiates_twice() -> None:
    v = lucid.tensor([1.0, 2.0], requires_grad=True)
    h = lucid.zeros(2, 3)
    col = _strided(h, [2], [3], 1)
    col.copy_(v * v)
    (g,) = lucid.autograd.grad(h.sum(), v, create_graph=True)
    assert g is not None
    assert _flat(g) == [2.0, 4.0]
    g.sum().backward()
    assert _flat(v.grad) == [2.0, 2.0]


def test_a_repeated_element_s_gradient_adds_up() -> None:
    # ``e`` reads ``h`` twice over; after ``h`` doubles, each of ``w``'s
    # elements reaches ``e.sum()`` twice, at twice its value.
    w = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True)
    h = w * 1.0
    e = _strided(h, [2, 3], [0, 1])
    h.mul_(2.0)
    e.sum().backward()
    assert _flat(w.grad) == [4.0, 4.0, 4.0]


def test_a_leaf_s_view_is_written_only_outside_autograd() -> None:
    p = lucid.zeros(6, requires_grad=True)
    v = _view(p, 2, 3)
    with pytest.raises(Exception, match="view of a leaf"):
        v.add_(1.0)
    assert _flat(p) == [0.0] * 6
    with lucid.no_grad():
        v.add_(1.0)
    assert _flat(p) == [1.0] * 6


def test_a_recorded_write_re_derives_the_other_views() -> None:
    # ``v`` held ``h``'s values before the write and holds three times them
    # after it; its gradient has to say so, or it reaches ``w`` unscaled.
    w = lucid.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], requires_grad=True)
    h = w * 1.0
    v = _view(h, 2, 3)
    h.mul_(3.0)
    assert v.requires_grad
    (v * 2.0).sum().backward()
    assert _flat(w.grad) == [6.0] * 6


def test_a_zero_gradient_write_reaches_an_earlier_view() -> None:
    w = lucid.tensor([1.5, 2.5, 3.5, 4.5, 5.5, 6.5], requires_grad=True)
    h = w * 1.0
    v = _view(h, 2, 3)
    # floor's gradient is zero; the write stays in the graph, as in the
    # reference framework, and the view taken before it follows it there.
    h.floor_()
    assert h.requires_grad and v.requires_grad
    assert _flat(v) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    v.sum().backward()
    assert w.grad is not None
    assert _flat(w.grad) == [0.0] * 6


# ── reshape-family ops make these views on the CPU ────────────────────────────


@pytest.mark.parametrize(
    "make",
    [
        lambda t: t.reshape(2, 3),
        lambda t: t.view(3, 2),
        lambda t: t.reshape(1, 6).squeeze(0),
        lambda t: t.unsqueeze(0),
        lambda t: t.reshape(2, 3).flatten(),
    ],
    ids=["reshape", "view", "squeeze", "unsqueeze", "flatten"],
)
def test_reshape_family_ops_view_a_dense_cpu_tensor(
    x: lucid.Tensor, make: Callable[[lucid.Tensor], lucid.Tensor]
) -> None:
    v = make(x)
    assert v.data_ptr() == x.data_ptr()
    v.mul_(2.0)
    assert _flat(x) == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]


def test_a_write_through_a_metal_reshape_reaches_its_base() -> None:
    # An MLX array cannot see a write made through another; the reshape is a
    # member of its base's view family in the engine, and the write lands in
    # the family's root.
    x = lucid.arange(6).float().to("metal")
    v = x.reshape(2, 3)
    v.add_(1.0)
    assert _flat(x) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_a_reshape_under_no_grad_does_not_require_grad() -> None:
    p = lucid.zeros(6, requires_grad=True)
    with lucid.no_grad():
        v = p.reshape(2, 3)
    assert not v.requires_grad


def test_gradients_flow_through_a_reshape_view() -> None:
    w = lucid.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], requires_grad=True)
    (w.reshape(2, 3) * 2.0).sum().backward()
    assert _flat(w.grad) == [2.0] * 6


def test_a_recorded_write_through_a_reshape_reaches_the_base_s_gradient() -> None:
    # ``h`` holds three times its old values after the write through ``v``;
    # its gradient has to say so.
    w = lucid.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], requires_grad=True)
    h = w * 1.0
    v = h.reshape(2, 3)
    v.mul_(3.0)
    h.sum().backward()
    assert _flat(w.grad) == [3.0] * 6
    assert _flat(h) == [3.0, 6.0, 9.0, 12.0, 15.0, 18.0]


def test_a_parameter_s_view_is_written_only_outside_autograd() -> None:
    p = lucid.zeros(6, requires_grad=True)
    v = p.reshape(2, 3)
    with pytest.raises(Exception, match="view of a leaf"):
        v.add_(1.0)
    with lucid.no_grad():
        v.add_(1.0)
    assert _flat(p) == [1.0] * 6


# ── a detached alias (``.data``) writes untracked ─────────────────────────────


def test_a_write_through_dot_data_is_untracked_even_while_recording() -> None:
    # The idiom for rescaling a parameter without autograd seeing it; the
    # PlaNet tests saturate a head this way.
    p = lucid.ones(3, requires_grad=True)
    p.data.copy_(p.data * 60.0)
    p.data.add_(1.0)
    assert _flat(p) == [61.0] * 3
    assert p.requires_grad and p.is_leaf


def test_a_write_through_dot_data_leaves_the_views_graphs_alone() -> None:
    w = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True)
    h = w * 1.0
    v = h.reshape(1, 3)
    h.data.mul_(2.0)
    assert _flat(v) == [2.0, 4.0, 6.0]
    v.sum().backward()
    assert _flat(w.grad) == [1.0] * 3


def test_a_view_of_a_batch_norm_running_stat_sees_the_update() -> None:
    # The running statistics' storage slot was replaced on every training
    # step, so a view taken of them kept the values it started with.
    bn = lucid.nn.BatchNorm1d(3)
    view = bn.running_mean.reshape(1, 3)
    bn.train()
    bn(lucid.randn(4, 3))
    assert _flat(bn.running_mean) != [0.0, 0.0, 0.0]
    assert _flat(view) == _flat(bn.running_mean)


def test_detach_shares_the_tensor_s_storage() -> None:
    # detach() copied, where the reference and its own docstring share.
    w = lucid.zeros(3)
    d = w.detach()
    d.add_(1.0)
    assert _flat(w) == [1.0] * 3
    lucid.detach(w).mul_(2.0)
    assert _flat(w) == [2.0] * 3
    assert not d.requires_grad and d.grad_fn is None


# ── leading-dim slices are views on the CPU ───────────────────────────────────


@pytest.fixture
def grid() -> lucid.Tensor:
    return lucid.arange(12).float().reshape(4, 3)


@pytest.mark.parametrize(
    ("take", "rows"),
    [
        (lambda t: t[1], [1]),
        (lambda t: t[1:3], [1, 2]),
        (lambda t: lucid.narrow(t, 0, 1, 2), [1, 2]),
        (lambda t: lucid.chunk(t, 2)[1], [2, 3]),
        (lambda t: lucid.unbind(t, 0)[1], [1]),
    ],
    ids=["int", "slice", "narrow", "chunk", "unbind"],
)
def test_a_leading_dim_slice_is_a_view(
    grid: lucid.Tensor, take: Callable[[lucid.Tensor], lucid.Tensor], rows: list[int]
) -> None:
    piece = take(grid)
    assert piece._impl.is_aliased()
    piece.mul_(10.0)
    for r in range(4):
        scale = 10.0 if r in rows else 1.0
        assert _flat(grid)[3 * r : 3 * r + 3] == [scale * (3 * r + c) for c in range(3)]


def test_a_slice_along_another_dim_is_a_view(grid: lucid.Tensor) -> None:
    column = grid[:, 1]
    column.add_(100.0)
    assert _flat(grid) == [float(v) + (100.0 if v % 3 == 1 else 0.0) for v in range(12)]


def test_writes_reach_between_a_tensor_and_its_slices(grid: lucid.Tensor) -> None:
    row = grid[2]
    grid[2] = 7.0
    assert _flat(row) == [7.0, 7.0, 7.0]
    row[0] = 5.0
    assert _flat(grid)[6:9] == [5.0, 7.0, 7.0]


def test_gradients_flow_through_a_slice_view() -> None:
    w = lucid.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)
    (w[1:3] * 2.0).sum().backward()
    assert w.grad.tolist() == [[0.0, 0.0], [2.0, 2.0], [2.0, 2.0]]


def test_a_recorded_write_through_a_slice_reaches_the_base_s_gradient() -> None:
    # The base's graph takes the written row in (CopySlices); the rest of
    # it still answers to what it was before.
    w = lucid.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    h = w * 1.0
    s = h[1]
    s.mul_(3.0)
    assert _flat(h) == [1.0, 2.0, 9.0, 12.0]
    h.sum().backward()
    assert _flat(w.grad) == [1.0, 1.0, 3.0, 3.0]


def test_a_recorded_write_through_the_base_reaches_a_slice_s_gradient() -> None:
    w = lucid.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    h = w * 1.0
    s = h[1]
    h.mul_(2.0)
    assert _flat(s) == [6.0, 8.0]
    s.sum().backward()
    assert _flat(w.grad) == [0.0, 0.0, 2.0, 2.0]


def test_slices_written_in_turn() -> None:
    # The node keeps no tensor, so it holds nothing that would stop the
    # second write to the buffer.
    w = lucid.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    h = w * 1.0
    top, bottom = h[0], h[1]
    top.mul_(2.0)
    bottom.mul_(3.0)
    assert _flat(h) == [2.0, 4.0, 9.0, 12.0]
    h.sum().backward()
    assert _flat(w.grad) == [2.0, 2.0, 3.0, 3.0]


def test_overlapping_slices_share_only_what_the_write_reached() -> None:
    w = lucid.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
    h = w * 1.0
    first, last = h[0:2], h[1:3]
    first.mul_(10.0)
    assert _flat(last) == [20.0, 3.0]
    last.sum().backward()
    assert _flat(w.grad) == [0.0, 10.0, 1.0]


def test_a_plain_base_takes_on_the_graph_written_into_its_slice() -> None:
    h = lucid.zeros(2, 2)
    v = lucid.tensor([5.0, 6.0], requires_grad=True)
    h[1].add_(v)
    assert h.requires_grad
    (h * lucid.tensor([[1.0, 2.0], [3.0, 4.0]])).sum().backward()
    assert _flat(v.grad) == [3.0, 4.0]


def test_a_constant_written_through_a_slice_stops_its_gradient() -> None:
    w = lucid.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    h = w * 1.0
    h[1].zero_()
    h.sum().backward()
    assert _flat(w.grad) == [1.0, 1.0, 0.0, 0.0]


def test_a_written_slice_differentiates_twice() -> None:
    # h = [w0, w1 * w1]: the gradient is [1, 2 w1], and its own is [0, 2].
    w = lucid.tensor([[1.0], [2.0]], requires_grad=True)
    h = w * 1.0
    s = h[1]
    s.mul_(w[1])
    (g,) = lucid.autograd.grad(h.sum(), w, create_graph=True)
    assert g is not None
    assert _flat(g) == [1.0, 4.0]
    g.sum().backward()
    assert _flat(w.grad) == [0.0, 2.0]


def test_copy_into_a_slice_sends_the_gradient_to_the_source() -> None:
    w = lucid.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    v = lucid.tensor([5.0, 6.0], requires_grad=True)
    h = w * 1.0
    h[0].copy_(v)
    assert _flat(h) == [5.0, 6.0, 3.0, 4.0]
    (h * lucid.tensor([[1.0, 2.0], [3.0, 4.0]])).sum().backward()
    assert _flat(w.grad) == [0.0, 0.0, 3.0, 4.0]
    assert _flat(v.grad) == [1.0, 2.0]


def test_a_tensor_that_takes_on_a_graph_is_no_longer_a_leaf() -> None:
    x = lucid.zeros(3)
    x.add_(lucid.ones(3, requires_grad=True))
    assert not x.is_leaf


def test_a_slice_given_a_graph_leaves_its_base_writable() -> None:
    # The slice stayed marked a leaf, so the next write into its family
    # took it for a parameter's view and was refused.
    v = lucid.ones(3, requires_grad=True)
    h = lucid.zeros(2, 3)
    s = h[1]
    s.add_(v)
    h.mul_(2.0)
    (h * lucid.arange(6).float().reshape(2, 3)).sum().backward()
    assert _flat(v.grad) == [6.0, 8.0, 10.0]


def test_a_slice_keeps_its_buffer_alive() -> None:
    row = lucid.arange(12).float().reshape(4, 3)[3]
    assert _flat(row) == [9.0, 10.0, 11.0]
