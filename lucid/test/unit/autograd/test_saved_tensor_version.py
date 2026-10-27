"""A write into a tensor saved for backward is caught, whether or not it requires grad.

``w * buf`` saves ``buf`` to differentiate with respect to ``w``, though
``buf`` itself never required grad.  Its version counter lived in the
autograd metadata, which a tensor like ``buf`` never had, so an in-place
write left the count where the node had recorded it.  On the CPU the write
lands in the very bytes the node holds, and backward computed ``w``'s
gradient from the values written after the forward, in silence:

    out = (w * buf).sum(); buf.fill_(5.0); out.backward()  ->  w.grad == 5

The reference refuses that backward, and so does Lucid now, on both devices.
The ordinary paths — a write to a tensor nothing saved, a write before the
forward, a ``no_grad`` update between steps, an optimiser loop — are
unaffected.
"""

from collections.abc import Callable

import pytest

import lucid
import lucid.nn as nn
from lucid._C import engine as _C_engine
from lucid.autograd.graph import allow_mutation_on_saved_tensors

# Every in-place writer that reached the buffer without moving its count.
WRITES: dict[str, Callable[[lucid.Tensor, str], object]] = {
    "fill_": lambda t, d: t.fill_(5.0),
    "zero_": lambda t, d: t.zero_(),
    "copy_": lambda t, d: t.copy_(lucid.full((3,), 5.0, device=d)),
    "index_add_": lambda t, d: t.index_add_(
        0, lucid.tensor([0], device=d), lucid.tensor([4.0], device=d)
    ),
    "index_copy_": lambda t, d: t.index_copy_(
        0, lucid.tensor([0], device=d), lucid.tensor([5.0], device=d)
    ),
    "index_fill_": lambda t, d: t.index_fill_(0, lucid.tensor([0], device=d), 5.0),
    "index_put_": lambda t, d: t.index_put_(
        (lucid.tensor([0], device=d),), lucid.tensor([5.0], device=d)
    ),
}


def _saved(device: str) -> tuple[lucid.Tensor, lucid.Tensor, lucid.Tensor]:
    """``w``, a buffer ``w * buf`` saved, and the loss built from them."""
    w = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True, device=device)
    buf = lucid.tensor([1.0, 1.0, 1.0], device=device)
    return w, buf, (w * buf).sum()


@pytest.mark.parametrize("write", list(WRITES), ids=list(WRITES))
def test_a_write_into_a_saved_tensor_without_grad_is_refused(write: str, device: str) -> None:
    w, buf, out = _saved(device)
    WRITES[write](buf, device)
    with pytest.raises(_C_engine.VersionMismatch):
        out.backward()
    assert w.grad is None


@pytest.mark.parametrize("write", ["fill_", "copy_", "index_fill_"])
def test_a_write_through_an_alias_of_the_saved_tensor_is_refused(write: str, device: str) -> None:
    # A view and ``detach()`` read the saved tensor's buffer.  This held
    # before and must keep holding: on the CPU the write itself is refused,
    # since a node holds the bytes it would change; on metal it is copied
    # back into the base, whose count moves, so backward refuses.
    for alias in (lambda t: t.view(3), lambda t: t.detach()):
        _, buf, out = _saved(device)
        if device == "cpu":
            with pytest.raises(_C_engine.NotImplementedError, match="saved for backward"):
                WRITES[write](alias(buf), device)
            continue
        WRITES[write](alias(buf), device)
        with pytest.raises(_C_engine.VersionMismatch):
            out.backward()


def test_zeroing_a_gradient_another_op_saved_is_refused(device: str) -> None:
    # ``x.grad`` is one tensor while it lives, so the node that read it and
    # the ``zero_`` that follows touch the same buffer/family; backward used to
    # return ``w.grad == 0`` from the zeroed values.
    x = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True, device=device)
    (x * 2.0).sum().backward()
    w = lucid.ones(3, requires_grad=True, device=device)
    y = (w * x.grad).sum()
    x.grad.zero_()
    with pytest.raises(_C_engine.VersionMismatch):
        y.backward()


def test_the_count_moves_without_making_the_tensor_require_grad(device: str) -> None:
    t = lucid.zeros(3, device=device)
    before = t._version
    assert before == t._impl.version
    t.fill_(1.0)
    assert t._version == t._impl.version > before
    assert not t.requires_grad
    assert t.is_leaf
    assert t.grad is None


@pytest.mark.parametrize("write", list(WRITES), ids=list(WRITES))
def test_a_write_into_a_tensor_nothing_saved_still_runs(write: str, device: str) -> None:
    w = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True, device=device)
    scale = lucid.tensor([2.0, 2.0, 2.0], device=device)
    unrelated = lucid.ones(3, device=device)
    out = (w * scale).sum()
    WRITES[write](unrelated, device)
    out.backward()
    assert w.grad is not None
    assert w.grad.tolist() == [2.0, 2.0, 2.0]


def test_a_write_before_the_forward_is_what_the_forward_reads(device: str) -> None:
    w = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True, device=device)
    buf = lucid.ones(3, device=device)
    buf.fill_(4.0)
    (w * buf).sum().backward()
    assert w.grad is not None
    assert w.grad.tolist() == [4.0, 4.0, 4.0]


def test_an_inplace_op_that_gives_the_saved_tensor_a_new_buffer_is_refused(
    device: str,
) -> None:
    # ``exp_`` leaves the saved bytes alone and hands ``buf`` a new buffer,
    # so the gradient would have come out right; the reference refuses the
    # backward all the same, because ``buf`` is no longer what was saved.
    w, buf, out = _saved(device)
    buf.exp_()
    with pytest.raises(_C_engine.VersionMismatch):
        out.backward()


def test_a_no_grad_update_between_steps_still_runs(device: str) -> None:
    # A running statistic kept by hand: saved by the forward, updated under
    # no_grad after the backward that needed it.
    lucid.manual_seed(0)
    model = nn.Linear(3, 3).to(device)
    running = lucid.ones(3, device=device)
    opt = lucid.optim.SGD(model.parameters(), lr=0.1)
    x = lucid.randn(4, 3, device=device)
    for _ in range(3):
        opt.zero_grad()
        loss = (model(x) * running).pow(2).mean()
        loss.backward()
        opt.step()
        with lucid.no_grad():
            running.mul_(0.9)
            running.add_(x.mean(0) * 0.1)
    assert all(p.grad is not None for p in model.parameters())


def test_the_opt_out_still_lets_backward_read_the_new_values() -> None:
    w, buf, out = _saved("cpu")
    with allow_mutation_on_saved_tensors():
        buf.fill_(5.0)
        out.backward()
    assert w.grad is not None
    assert w.grad.tolist() == [5.0, 5.0, 5.0]
