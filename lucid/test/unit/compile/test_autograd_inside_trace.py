"""A function that runs a backward pass inside itself is not compiled.

``autograd.grad`` / ``backward`` inside a traced function — MeanFlow's
``jvp``, a gradient penalty, a score-matching loss — computes storage the
tracer never records, so an executable would replay trace-time values; and
``make_step`` hung trying to build a VJP of the recorded backward.  The
trace is now marked, every compile entry point refuses it, and the call
runs eager with eager's answer.
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


class _Penalised(nn.Module):
    """Output plus the squared norm of its gradient with respect to the input."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        with lucid.enable_grad():
            x = x.detach().requires_grad_(True)
            y = self.lin(x).tanh()
            (g,) = lucid.autograd.grad(y.sum(), x, create_graph=True)
        return y.sum(dim=1) + (g * g).sum(dim=1)


def _setup() -> tuple[nn.Module, lucid.Tensor, lucid.Tensor]:
    lucid.manual_seed(0)
    model = _Penalised().to(COMPILE_DEVICE)
    return (model, lucid.randn(3, 4).to(COMPILE_DEVICE), lucid.randn(3, 4).to(COMPILE_DEVICE))


@pytest.mark.usefixtures("inference")
def test_forward_runs_eager_with_fresh_values() -> None:
    model, x1, x2 = _setup()
    cm = lucid.compile(model)
    cm(x1)
    assert np.allclose(cm(x2).numpy(), model(x2).numpy(), atol=1e-6)
    assert cm.cache_info()["eager_only"]


def test_training_step_runs_eager_with_eagers_gradients() -> None:
    model, x1, x2 = _setup()

    def loss_fn(y: lucid.Tensor) -> lucid.Tensor:
        return y.sum()

    step = lucid.compile.make_step(model, loss_fn)
    step(x1).backward()
    for p in model.parameters():
        p.grad = None
    loss = step(x2)
    loss.backward()
    got = [p.grad.numpy().copy() for p in model.parameters()]
    for p in model.parameters():
        p.grad = None
    want = loss_fn(model(x2))
    want.backward()
    assert step.eager_only
    assert np.isclose(float(loss.item()), float(want.item()), rtol=1e-6)
    for g, p in zip(got, model.parameters()):
        assert np.allclose(g, p.grad.numpy(), atol=1e-6)
