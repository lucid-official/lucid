"""A host write into a Metal shared buffer comes after the GPU reads before it.

A Metal op records a graph node and runs when a value is read.  A result built
over a shared buffer, and not yet run when the buffer is written, used to run
afterwards and read the new bytes: ``y = g * 1.0; g.add_(3.0)`` left ``y`` at
3.  Every writer into the buffer now runs those readers first
(``gpu::settle_buffer_readers``), so a result keeps the value it was built
from, whichever alias — the Metal tensor, a view of it, a reshape of it, the
CPU tensor over the same pages — takes the write.  On the CPU, where ops run
eagerly, the same code always held.
"""

from collections.abc import Callable

import numpy as np
import pytest

import lucid
import lucid.metal as metal
from lucid._C import engine as _C_engine
from lucid._dispatch import _wrap
from lucid.test._fixtures.devices import metal_available

pytestmark = pytest.mark.skipif(not metal_available(), reason="metal unavailable")

PRE = [1.0, 2.0, 3.0, 4.0]

Write = Callable[[lucid.Tensor], object]

WRITES: dict[str, Write] = {
    "add_": lambda t: t.add_(10.0),
    "mul_": lambda t: t.mul_(3.0),
    "copy_": lambda t: t.copy_(lucid.full(t.shape, 7.0, device=t.device.type)),
    "fill_": lambda t: t.fill_(5.0),
    "setitem": lambda t: t.__setitem__(0, 9.0),
}


def _flat(t: lucid.Tensor) -> list[float]:
    return [float(v) for v in t.reshape(-1).tolist()]


def _member(base: lucid.Tensor, target: str) -> lucid.Tensor:
    """The alias of ``base`` that takes the write."""
    if target == "base":
        return base
    if target == "reshape":
        return base.reshape(2, 2)
    # Elements 1 and 2: a member of base's family at an offset.
    return _wrap(_C_engine.TensorImpl._make_view(base._impl, [2], [1], 1))


def _buffer() -> lucid.Tensor:
    s = metal.shared_tensor((4,))
    s.copy_(lucid.tensor(PRE))
    return s


def _expected(target: str, name: str) -> list[float]:
    """The write applied to an ordinary CPU tensor: the values it leaves."""
    ref = lucid.tensor(PRE)
    with lucid.no_grad():
        WRITES[name](_member(ref, target))
    return _flat(ref)


@pytest.mark.parametrize("name", list(WRITES))
@pytest.mark.parametrize("target", ["base", "view", "reshape"])
@pytest.mark.parametrize("device", ["cpu", "metal"])
def test_a_reader_built_before_a_write_keeps_its_value(device: str, target: str, name: str) -> None:
    g = _buffer().to(device)
    y = g * 2.0 + 1.0
    t = _member(g, target)
    with lucid.no_grad():
        WRITES[name](t)
    assert _flat(y) == [2.0 * v + 1.0 for v in PRE]
    # The write itself lands, in the buffer every alias reads.
    assert _flat(g) == _expected(target, name)
    assert g.is_shared


@pytest.mark.parametrize("name", list(WRITES))
@pytest.mark.parametrize("target", ["base", "view", "reshape"])
def test_a_write_through_the_cpu_alias_waits_for_metal_readers(target: str, name: str) -> None:
    s = _buffer()
    g = s.to("metal")
    y = g * 2.0 + 1.0
    with lucid.no_grad():
        WRITES[name](_member(s, target))
    assert _flat(y) == [2.0 * v + 1.0 for v in PRE]
    assert _flat(g) == _expected(target, name)


def test_a_reader_of_a_reshape_member_keeps_its_value() -> None:
    g = _buffer().to("metal")
    y = g.reshape(2, 2).sum(0)
    with lucid.no_grad():
        g.add_(10.0)
    assert y.tolist() == [4.0, 6.0]


#: Engine view-family members whose geometry is not a run of the buffer.
FAMILY_MEMBERS: dict[str, Callable[[lucid.Tensor], lucid.Tensor]] = {
    "strided": lambda g: _wrap(_C_engine.TensorImpl._make_view(g._impl, [2], [2], 0)),
    "transpose": lambda g: _wrap(_C_engine.TensorImpl._make_view(g._impl, [2, 2], [1, 2], 0)),
}
#: The same geometries, made by ops: values of their own on Metal so far.
OP_VIEWS: dict[str, Callable[[lucid.Tensor], lucid.Tensor]] = {
    "step": lambda g: g[::2],
    "T": lambda g: g.reshape(2, 2).T,
}
STRIDED_PRE = {"strided": [1.0, 3.0], "transpose": [1.0, 3.0, 2.0, 4.0]}
STRIDED_PRE.update(step=STRIDED_PRE["strided"], T=STRIDED_PRE["transpose"])


def _write_through(writer: str, s: lucid.Tensor, g: lucid.Tensor) -> None:
    with lucid.no_grad():
        (g if writer == "metal" else s).add_(10.0)


@pytest.mark.parametrize("writer", ["metal", "cpu"])
@pytest.mark.parametrize("order", ["empty", "filled", "reader"])
@pytest.mark.parametrize("member", list(FAMILY_MEMBERS))
def test_a_strided_member_shows_the_write_its_earlier_reader_does_not(
    member: str, order: str, writer: str
) -> None:
    s = _buffer()
    g = s.to("metal")
    v = FAMILY_MEMBERS[member](g)
    y = v * 1.0 if order == "reader" else None
    if order == "filled":
        v.tolist()  # the member's packed elements are cached now
    _write_through(writer, s, g)
    assert _flat(v) == [x + 10.0 for x in STRIDED_PRE[member]]
    if y is not None:
        assert _flat(y) == STRIDED_PRE[member]


@pytest.mark.parametrize("writer", ["metal", "cpu"])
@pytest.mark.parametrize("order", ["empty", "filled"])
@pytest.mark.parametrize("op", list(OP_VIEWS))
def test_an_op_view_shows_a_later_write(op: str, order: str, writer: str) -> None:
    s = _buffer()
    g = s.to("metal")
    v = OP_VIEWS[op](g)
    if order == "filled":
        v.tolist()
    _write_through(writer, s, g)
    assert _flat(v) == [x + 10.0 for x in STRIDED_PRE[op]]


def test_a_reader_keeps_its_value_when_the_search_gives_up() -> None:
    # Past the walk budget every pending array runs, the reader included.
    g = _buffer().to("metal")
    y = g * 2.0
    unrelated = lucid.zeros(4, device="metal")
    for _ in range(2_000):
        unrelated = unrelated + 1.0
    for step in range(3):
        with lucid.no_grad():
            g.add_(1.0)
        assert _flat(g) == [v + step + 1.0 for v in PRE]
    assert _flat(y) == [2.0 * v for v in PRE]
    assert _flat(unrelated) == [2000.0] * 4


def test_a_deep_lazy_chain_is_searched_without_recursion() -> None:
    g = _buffer().to("metal")
    y = g
    for _ in range(20_000):
        y = y + 0.0
    with lucid.no_grad():
        g.fill_(0.0)
    assert _flat(y) == PRE


def test_pending_work_that_does_not_read_the_buffer_still_runs_correctly() -> None:
    g = _buffer().to("metal")
    other = lucid.arange(4.0, device="metal") * 2.0
    with lucid.no_grad():
        g.fill_(0.0)
    assert other.tolist() == [0.0, 2.0, 4.0, 6.0]
    assert _flat(g) == [0.0] * 4


def test_a_numpy_alias_sees_writes_and_orders_after_synchronize() -> None:
    # Host reads stay zero-copy: the NumPy array is the buffer, so a write
    # through any alias shows in it.  A write through the NumPy array itself
    # goes past the engine — lucid.metal.synchronize() first runs pending
    # GPU readers, as shared_tensor's docstring asks.
    s = _buffer()
    g = s.to("metal")
    host = s.numpy()
    with lucid.no_grad():
        g.add_(1.0)
    assert host.tolist() == [v + 1.0 for v in PRE]
    y = g * 1.0
    metal.synchronize()
    host[:] = np.float32(0.0)
    assert _flat(y) == [v + 1.0 for v in PRE]
    assert _flat(g) == [0.0] * 4


def test_a_clone_does_not_follow_later_writes() -> None:
    # A Metal clone of a shared buffer once aliased it (LCD-380).
    g = _buffer().to("metal")
    c = g.clone()
    c.tolist()
    with lucid.no_grad():
        g.add_(10.0)
    assert _flat(c) == PRE
