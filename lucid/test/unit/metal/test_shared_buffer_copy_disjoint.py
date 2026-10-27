"""Copies and copy-like MLX results over a Metal shared buffer stay disjoint from later writes.

When MLX evaluates ``copy``, ``contiguous``, ``reshape``, ``slice``, a
single-input ``concatenate`` / ``stack``, or a sum over a size-1 axis, its
kernel aliases the input's ``array::Data`` via ``copy_shared_buffer`` rather
than allocating a new buffer.  On a regular MLX array that is invisible because
nothing mutates an MLX array in place; on a Metal shared buffer a later host
write (from Metal or CPU, through the base tensor, a reshape, or an offset
view) overwrites those bytes in place.  ``gpu::settle_buffer_readers`` now
rebinds every evaluated op result that aliases the shared buffer onto a fresh
snapshot of the pre-write bytes before the write lands, and optimizer state
clones / overwrites use ``mlx::core::copy`` through ``gpu::wrap_mlx_array``.
"""

from collections.abc import Callable

import pytest

import lucid
import lucid.metal as metal
from lucid._C import engine as _C_engine
from lucid._dispatch import _wrap
from lucid.test._fixtures.devices import metal_available

pytestmark = pytest.mark.skipif(not metal_available(), reason="metal unavailable")

PRE = [1.0, 2.0, 3.0, 4.0]

Builder = Callable[[lucid.Tensor], lucid.Tensor]


def _flat(t: lucid.Tensor) -> list[float]:
    return [float(v) for v in t.reshape(-1).tolist()]


def _buffer() -> lucid.Tensor:
    s = metal.shared_tensor((4,))
    s.copy_(lucid.tensor(PRE))
    return s


def _member(base: lucid.Tensor, target: str) -> lucid.Tensor:
    if target == "base":
        return base
    if target == "reshape":
        return base.reshape(2, 2)
    return _wrap(_C_engine.TensorImpl._make_view(base._impl, [2], [1], 1))


def _copy_into_metal(g: lucid.Tensor) -> lucid.Tensor:
    dst = lucid.zeros(g.shape, dtype=g.dtype, device="metal")
    with lucid.no_grad():
        dst.copy_(g)
    return dst


def _optim_momentum_snapshot(g: lucid.Tensor) -> lucid.Tensor:
    p = lucid.ones(g.shape, dtype=g.dtype, device="metal", requires_grad=True)
    p.grad = g
    eng = _C_engine.SGD([p._impl], 0.1, 0.9, 0.0, 0.0, False)
    eng.step()
    bufs = eng.state_buffers()
    snap_impl = bufs[0][1][0]
    assert snap_impl is not None
    return _wrap(snap_impl)


BUILDERS: dict[str, Builder] = {
    "clone": lambda g: g.clone(),
    "copy_into_metal": _copy_into_metal,
    "optim_state_snapshot": _optim_momentum_snapshot,
    "cat_single": lambda g: lucid.cat([g]),
    "stack_single": lambda g: lucid.stack([g]),
    "sum_unit_axis": lambda g: g.reshape(4, 1).sum(1),
    "dlpack_mul": lambda g: lucid.from_dlpack(g) * 1.0,
    "dlpack_clone": lambda g: lucid.from_dlpack(g).clone(),
}


@pytest.mark.parametrize("kind", list(BUILDERS))
@pytest.mark.parametrize("evaluated_before_write", [False, True])
@pytest.mark.parametrize("write_side", ["metal", "cpu"])
@pytest.mark.parametrize("target", ["base", "view", "reshape"])
def test_shared_buffer_copy_is_disjoint_from_later_writes(
    kind: str,
    evaluated_before_write: bool,
    write_side: str,
    target: str,
) -> None:
    s = _buffer()
    g = s.to("metal")
    out = BUILDERS[kind](g)
    if evaluated_before_write:
        assert _flat(out) == PRE

    writer_root = g if write_side == "metal" else s
    with lucid.no_grad():
        _member(writer_root, target).add_(10.0)

    assert _flat(out) == PRE


@pytest.mark.parametrize("evaluated_before_write", [False, True])
@pytest.mark.parametrize("write_side", ["metal", "cpu"])
def test_clone_of_shared_buffer_view_and_reshape_stay_disjoint(
    evaluated_before_write: bool,
    write_side: str,
) -> None:
    s = _buffer()
    g = s.to("metal")
    v_reshape = g.reshape(2, 2)
    v_view = _member(g, "view")
    c_reshape = v_reshape.clone()
    c_view = v_view.clone()
    if evaluated_before_write:
        assert _flat(c_reshape) == PRE
        assert _flat(c_view) == [2.0, 3.0]

    writer = g if write_side == "metal" else s
    with lucid.no_grad():
        writer.add_(10.0)

    assert _flat(c_reshape) == PRE
    assert _flat(c_view) == [2.0, 3.0]
    assert _flat(v_reshape) == [v + 10.0 for v in PRE]
    assert _flat(v_view) == [12.0, 13.0]


@pytest.mark.parametrize("evaluated_before_write", [False, True])
def test_view_of_clone_stays_disjoint_after_shared_buffer_write(
    evaluated_before_write: bool,
) -> None:
    s = _buffer()
    g = s.to("metal")
    c = g.clone()
    cv = c.reshape(2, 2)
    if evaluated_before_write:
        assert _flat(cv) == PRE

    with lucid.no_grad():
        g.add_(10.0)

    assert _flat(cv) == PRE
    assert _flat(c) == PRE


def test_sgd_live_momentum_and_loaded_state_stay_disjoint_from_grad_shared_buffer() -> None:
    s = _buffer()
    g = s.to("metal")
    p = lucid.nn.Parameter(lucid.ones((4,), device="metal"))
    p.grad = g
    opt = lucid.optim.SGD([p], lr=0.1, momentum=0.9)
    opt.step()

    p_eng = lucid.ones((4,), device="metal", requires_grad=True)
    eng2 = _C_engine.SGD([p_eng._impl], 0.1, 0.9, 0.0, 0.0, False)
    eng2.load_state_buffers([("momentum_buffer", [g._impl])])

    with lucid.no_grad():
        g.add_(10.0)

    snap_after = opt.state_dict()["state"][0]["momentum_buffer"]
    loaded_impl = eng2.state_buffers()[0][1][0]
    assert loaded_impl is not None
    assert [float(v) for v in snap_after.reshape(-1).tolist()] == PRE
    assert _flat(_wrap(loaded_impl)) == PRE
