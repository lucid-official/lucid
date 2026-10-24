"""Tensors nested in list / tuple / dict arguments are bound on every call.

The cache key already described nested tensors, so a second call with new
tensors of the same shapes hit the cached executable — but feed binding
looked only at top-level arguments, so a nested tensor was pinned at its
trace-time value.  ``lucid.compile`` then returned the first call's answer
for every later call, without a word; ``make_step`` with a list of
targets crashed in backward.  Both now bind every tensor by its position
in the call (``leaf_tensors``).
"""

import numpy as np
import pytest

import lucid
import lucid.nn as nn

from lucid.test.unit.compile._helpers import COMPILE_DEVICE


def _metal_ok() -> bool:
    try:
        lucid.zeros(1).to(COMPILE_DEVICE)
    except Exception:  # noqa: BLE001 — any failure means no Metal here
        return False
    return True


pytestmark = pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")


def _t(*shape: int) -> lucid.Tensor:
    return lucid.randn(*shape).to(COMPILE_DEVICE)


class _ListIn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, xs: list[lucid.Tensor]) -> lucid.Tensor:
        return self.lin(xs[0]) + xs[1]


class _DictIn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(
        self, x: lucid.Tensor, *, extra: dict[str, lucid.Tensor]
    ) -> lucid.Tensor:
        return self.lin(x) * extra["scale"] + extra["shift"]


@pytest.mark.usefixtures("inference")
def test_list_argument_is_fresh_every_call() -> None:
    lucid.manual_seed(0)
    model = _ListIn().to(COMPILE_DEVICE).eval()
    cm = lucid.compile(model)
    cm([_t(2, 4), _t(2, 4)])
    for _ in range(2):
        xs = [_t(2, 4), _t(2, 4)]
        assert np.allclose(cm(xs).numpy(), model(xs).numpy(), atol=1e-5)
    assert not cm.cache_info()["eager_only"]


@pytest.mark.usefixtures("inference")
def test_dict_keyword_argument_is_fresh_every_call() -> None:
    lucid.manual_seed(0)
    model = _DictIn().to(COMPILE_DEVICE).eval()
    cm = lucid.compile(model)
    cm(_t(2, 4), extra={"shift": _t(2, 4), "scale": _t(2, 4)})
    x, extra = _t(2, 4), {"shift": _t(2, 4), "scale": _t(2, 4)}
    assert np.allclose(
        cm(x, extra=extra).numpy(), model(x, extra=extra).numpy(), atol=1e-5
    )


@pytest.mark.usefixtures("inference")
def test_nested_alias_is_part_of_the_key() -> None:
    """One tensor at two nested positions at trace time, two tensors later."""
    lucid.manual_seed(0)
    model = _ListIn().to(COMPILE_DEVICE).eval()
    cm = lucid.compile(model)
    same = _t(2, 4)
    cm([same, same])
    xs = [_t(2, 4), _t(2, 4)]
    assert np.allclose(cm(xs).numpy(), model(xs).numpy(), atol=1e-5)


def test_make_step_takes_a_list_of_targets() -> None:
    lucid.manual_seed(0)
    model = nn.Linear(4, 4).to(COMPILE_DEVICE)

    def loss_fn(out: lucid.Tensor, targets: list[lucid.Tensor]) -> lucid.Tensor:
        return ((out - targets[0]) ** 2).sum() + (out * targets[1]).sum()

    step = lucid.compile.make_step(model, loss_fn)
    x = _t(2, 4)
    step(x, [_t(2, 4), _t(2, 4)]).backward()
    targets = [_t(2, 4), _t(2, 4)]
    for p in model.parameters():
        p.grad = None
    loss = step(x, targets)
    loss.backward()
    got = [p.grad.numpy().copy() for p in model.parameters()]
    for p in model.parameters():
        p.grad = None
    want = loss_fn(model(x), targets)
    want.backward()
    assert np.isclose(float(loss.item()), float(want.item()), rtol=1e-5)
    for g, p in zip(got, model.parameters()):
        assert np.allclose(g, p.grad.numpy(), rtol=1e-4, atol=1e-5)
    assert not step.eager_only
