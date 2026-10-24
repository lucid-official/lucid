"""An in-place write through a view of a Metal tensor reaches the tensor.

On Metal a view is its own MLX array, so ``x[0].add_(1)``, ``x.t()[1].mul_(2)``,
``r[1][0] = v``, ``p.data.sub_(lr * g)`` and ``p.grad.mul_(s)`` changed the
view and nothing else.  No error was raised, while the CPU and the reference
framework wrote through, and a hand-written optimizer step on Metal trained
nothing.  A Metal view is now a member of its base's view family in the
engine, as a CPU view is, so the write lands in the base.
Every case here is held to what the CPU does with the same code.

Found while verifying that compiled steps follow writes through views: they
did, because eager Metal and compiled Metal both did nothing.
"""

from collections.abc import Callable

import numpy as np
import pytest

import lucid
import lucid.nn as nn
from lucid.test._fixtures.devices import metal_available

pytestmark = pytest.mark.skipif(not metal_available(), reason="metal unavailable")


def _rows(dev: str) -> lucid.Tensor:
    return lucid.arange(12.0).reshape(3, 4).to(dev)


def _row_mul(b: lucid.Tensor) -> None:
    b[1].mul_(2.0)


def _held_row_add(b: lucid.Tensor) -> None:
    v = b[2]
    v.add_(100.0)


def _column_fill(b: lucid.Tensor) -> None:
    b[:, 0].fill_(-1.0)


def _transpose_row(b: lucid.Tensor) -> None:
    b.t()[3].mul_(10.0)


def _flat_element(b: lucid.Tensor) -> None:
    b.view(-1)[5].add_(0.5)


def _chained_setitem(b: lucid.Tensor) -> None:
    b[1][0] = -5.0


def _permute(b: lucid.Tensor) -> None:
    b.reshape(2, 3, 2).permute(2, 0, 1)[1].zero_()


def _split_piece(b: lucid.Tensor) -> None:
    _, tail = b.reshape(-1).split([4, 8])
    tail.mul_(-1.0)


def _diagonal(b: lucid.Tensor) -> None:
    b[:, :3].diagonal().fill_(0.0)


def _new_axis_and_narrow(b: lucid.Tensor) -> None:
    b[None, 1].add_(7.0)
    b.narrow(1, 1, 2).mul_(2.0)


def _augmented(b: lucid.Tensor) -> None:
    row = b[0]
    row += 50.0


def _T_property(b: lucid.Tensor) -> None:
    b.T[1].sub_(3.0)


WRITES: dict[str, Callable[[lucid.Tensor], None]] = {
    "row mul_": _row_mul,
    "held row add_": _held_row_add,
    "column fill_": _column_fill,
    "transpose row": _transpose_row,
    "flat element": _flat_element,
    "chained setitem": _chained_setitem,
    "permute": _permute,
    "split piece": _split_piece,
    "diagonal": _diagonal,
    "new axis, narrow": _new_axis_and_narrow,
    "augmented": _augmented,
    "T property": _T_property,
}


@pytest.mark.parametrize("name", list(WRITES))
def test_a_write_through_a_view_matches_the_cpu(name: str) -> None:
    cpu, metal = _rows("cpu"), _rows("metal")
    WRITES[name](cpu)
    WRITES[name](metal)
    assert metal.to("cpu").tolist() == cpu.tolist()


def test_data_detach_and_grad_write_through() -> None:
    p = lucid.arange(4.0).to("metal").requires_grad_()
    p.data.mul_(10.0)
    assert p.tolist() == [0.0, 10.0, 20.0, 30.0]
    q = lucid.arange(4.0).to("metal")
    q.detach().add_(100.0)
    assert q.tolist() == [100.0, 101.0, 102.0, 103.0]
    g = lucid.ones(3).to("metal").requires_grad_()
    (g * 2.0).sum().backward()
    g.grad.mul_(10.0)
    assert g.grad.tolist() == [20.0, 20.0, 20.0]
    g.grad.zero_()
    assert g.grad.tolist() == [0.0, 0.0, 0.0]


def test_a_hand_written_sgd_step_trains_on_metal_as_on_the_cpu() -> None:
    def losses(dev: str) -> list[float]:
        lucid.manual_seed(0)
        layer = nn.Linear(4, 1).to(dev)
        x = lucid.randn(32, 4).to(dev)
        y = lucid.randn(32, 1).to(dev)
        seen = []
        for _ in range(20):
            for p in layer.parameters():
                p.grad = None
            loss = ((layer(x) - y) ** 2).mean()
            loss.backward()
            seen.append(float(loss.item()))
            for p in layer.parameters():
                p.data.sub_(0.1 * p.grad)
        return seen

    cpu, metal = losses("cpu"), losses("metal")
    assert metal[-1] < metal[0]  # it trains at all
    np.testing.assert_allclose(metal, cpu, rtol=1e-4)


def test_gradients_through_view_writes_match_the_cpu() -> None:
    x0 = np.random.default_rng(0).standard_normal((3, 3)).astype(np.float32)

    def f(x: lucid.Tensor) -> lucid.Tensor:
        y = x * 2.0
        y[0].mul_(3.0)
        y[:, 1].add_(x[:, 0] * 5.0)
        y.t()[2].sub_(x[1])
        flat = y.reshape(-1)
        flat[3:6].mul_(0.5)
        return (y * y).sum()

    grads = {}
    for dev in ("cpu", "metal"):
        x = lucid.tensor(x0).to(dev).requires_grad_()
        f(x).backward()
        grads[dev] = x.grad.numpy()
    np.testing.assert_allclose(grads["metal"], grads["cpu"], rtol=1e-6, atol=1e-6)


def test_overlapping_and_leaf_views_are_refused() -> None:
    expanded = lucid.ones(1, 3).to("metal").expand(2, 3)
    with pytest.raises(RuntimeError, match="overlap"):
        expanded.add_(1.0)
    leaf = lucid.ones(3).to("metal").requires_grad_()
    with pytest.raises(RuntimeError, match="leaf tensor that requires grad"):
        leaf[0].mul_(2.0)
    with lucid.no_grad():
        leaf[0].mul_(2.0)
    assert leaf.tolist() == [2.0, 1.0, 1.0]


def test_a_compiled_step_carries_a_buffer_view_write() -> None:
    class Counter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)
            self.register_buffer("seen", lucid.zeros(3, 4))

        def forward(self, x: lucid.Tensor) -> lucid.Tensor:
            with lucid.no_grad():
                self.seen[1].add_(1.0)
                self.seen.t()[0].mul_(2.0)
            return self.lin(x)

    def run(compiled: bool) -> list[list[float]]:
        lucid.manual_seed(0)
        model = Counter().to("metal")
        step = (
            lucid.compile.make_step(model, lambda out: (out * out).mean())
            if compiled
            else None
        )
        for _ in range(3):
            x = lucid.randn(2, 4).to("metal")
            loss = step(x) if step is not None else (model(x) ** 2).mean()
            loss.backward()
        return model.seen.to("cpu").tolist()

    eager = run(False)
    assert eager[1] != [0.0, 0.0, 0.0, 0.0]  # the writes happen at all
    assert run(True) == eager


def test_non_viewable_reshape_does_not_write_back_on_metal() -> None:
    x = lucid.arange(6.0, device="metal").reshape(2, 3)
    before = x.tolist()
    flat = x.t().flatten()
    flat.add_(10.0)
    assert x.tolist() == before
    assert flat.tolist() == [
        0.0 + 10.0,
        3.0 + 10.0,
        1.0 + 10.0,
        4.0 + 10.0,
        2.0 + 10.0,
        5.0 + 10.0,
    ]

    y = lucid.arange(12.0, device="metal").reshape(3, 4)
    y_before = y.tolist()
    narrow_flat = y.narrow(1, 0, 2).reshape(-1)
    narrow_flat.fill_(0.0)
    assert y.tolist() == y_before


@pytest.mark.parametrize("dev", ["cpu", "metal"])
def test_view_and_view_as_refuse_incompatible_strides(dev: str) -> None:
    x = lucid.arange(6.0, device=dev).reshape(2, 3)
    proto = lucid.zeros(6, device=dev)
    with pytest.raises(RuntimeError, match="use .reshape"):
        x.transpose(0, 1).view(-1)
    with pytest.raises(RuntimeError, match="use .reshape"):
        x.transpose(0, 1).view_as(proto)
    with pytest.raises(RuntimeError, match="use .reshape"):
        lucid.view(x.transpose(0, 1), (-1,))
