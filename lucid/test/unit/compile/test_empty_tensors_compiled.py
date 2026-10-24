"""A trace holding a zero-size tensor runs eager instead of aborting.

MPSGraph does not decline a constant or placeholder with a zero-length
axis — it aborts the process ("shape[0] (0) should be a strictly
positive value for constant operation").  A compiled Fast R-CNN step
died that way on an image with no proposals.  Every builder now checks
the trace first and leaves such a call to eager.  Were the check lost,
this module would abort rather than fail — loud either way.
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


class _WithEmpty(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        none = lucid.zeros(0, 4, device=x.device)
        return self.lin(lucid.cat([x, none], dim=0))


@pytest.mark.usefixtures("inference")
def test_forward_with_an_empty_tensor_runs_eager() -> None:
    lucid.manual_seed(0)
    model = _WithEmpty().to(COMPILE_DEVICE).eval()
    x = lucid.randn(3, 4).to(COMPILE_DEVICE)
    cm = lucid.compile(model)
    assert np.allclose(cm(x).numpy(), model(x).numpy())
    assert cm.cache_info()["eager_only"]


def test_training_step_with_an_empty_tensor_runs_eager() -> None:
    lucid.manual_seed(0)
    model = _WithEmpty().to(COMPILE_DEVICE)
    x = lucid.randn(3, 4).to(COMPILE_DEVICE)
    step = lucid.compile.make_step(model, lambda y: (y * y).sum())
    loss = step(x)
    loss.backward()
    assert step.eager_only
    assert all(p.grad is not None for p in model.parameters())
