"""Parity: Adam and AdamW, with and without AMSGrad, against the reference.

``amsgrad=True`` used to be accepted by both optimizers and then dropped —
the engine constructor never received it — so the run was plain Adam under
an AMSGrad label.  Both now keep the running maximum of the second moment
and correct it for bias after the maximum, as the reference does.

The gradients start large and then shrink, so the running maximum and the
current second moment part ways — quickly, with beta2 at 0.5; a run where
they coincide would pass with AMSGrad still missing.
"""

from typing import Any

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.optim as optim

_RNG = np.random.default_rng(0)
_W0 = _RNG.standard_normal((4, 3)).astype(np.float32)
_GRADS = [
    _RNG.standard_normal((4, 3)).astype(np.float32) * s for s in (5.0, 0.1, 0.1, 3.0, 0.05, 0.05)
]


@pytest.mark.parametrize("device", ["cpu", "metal"])
@pytest.mark.parametrize("amsgrad", [True, False])
@pytest.mark.parametrize("name", ["Adam", "AdamW"])
def test_adam_family_matches_the_reference(name: str, amsgrad: bool, device: str, ref: Any) -> None:
    mine = nn.Parameter(lucid.tensor(_W0.copy(), device=device))
    theirs = ref.nn.Parameter(ref.tensor(_W0.copy()))
    opt_mine = getattr(optim, name)(
        [mine], lr=0.1, betas=(0.9, 0.5), weight_decay=0.01, amsgrad=amsgrad
    )
    opt_theirs = getattr(ref.optim, name)(
        [theirs], lr=0.1, betas=(0.9, 0.5), weight_decay=0.01, amsgrad=amsgrad
    )
    for g in _GRADS:
        mine.grad = lucid.tensor(g, device=device)
        opt_mine.step()
        theirs.grad = ref.tensor(g)
        opt_theirs.step()
    np.testing.assert_allclose(mine.numpy(), theirs.detach().numpy(), rtol=0, atol=1e-6)
