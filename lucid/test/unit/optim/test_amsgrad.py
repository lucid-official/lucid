"""AMSGrad is a real variant of Adam and AdamW, and its state survives a save.

``amsgrad=True`` used to be accepted and then dropped on the way to the
engine, so an AMSGrad run was plain Adam.  These pin what makes it AMSGrad
without the reference framework: the parameters go somewhere plain Adam
does not, the running maximum is part of the saved state, and a run resumed
from that state continues exactly.  Agreement with the reference itself is
in ``lucid/test/parity/optim/test_parity_adam_amsgrad.py``.
"""

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.optim as optim

_RNG = np.random.default_rng(1)
_W0 = _RNG.standard_normal((4, 3)).astype(np.float32)
# Large first, then small: the running maximum stays up while ``v`` decays —
# fast, with beta2 at 0.5, so the two variants end far apart.
_BETAS = (0.9, 0.5)
_GRADS = [
    _RNG.standard_normal((4, 3)).astype(np.float32) * s for s in (5.0, 0.1, 0.1, 3.0, 0.05, 0.05)
]


def _run(cls: type, amsgrad: bool, device: str, grads: list) -> tuple:
    p = nn.Parameter(lucid.tensor(_W0.copy(), device=device))
    opt = cls([p], lr=0.1, betas=_BETAS, amsgrad=amsgrad)
    for g in grads:
        p.grad = lucid.tensor(g, device=device)
        opt.step()
    return p, opt


@pytest.mark.parametrize("device", ["cpu", "metal"])
@pytest.mark.parametrize("cls", [optim.Adam, optim.AdamW], ids=["Adam", "AdamW"])
def test_amsgrad_is_not_plain_adam(cls: type, device: str) -> None:
    ams, _ = _run(cls, True, device, _GRADS)
    plain, _ = _run(cls, False, device, _GRADS)
    assert float(np.abs(ams.numpy() - plain.numpy()).max()) > 1e-3


@pytest.mark.parametrize("cls", [optim.Adam, optim.AdamW], ids=["Adam", "AdamW"])
def test_the_running_maximum_is_saved_and_restored(cls: type) -> None:
    p, opt = _run(cls, True, "cpu", _GRADS[:3])
    state = opt.state_dict()
    per_param = next(iter(state["state"].values()))
    assert "max_exp_avg_sq" in per_param
    assert np.all(np.asarray(per_param["max_exp_avg_sq"]) >= np.asarray(per_param["exp_avg_sq"]))

    resumed = nn.Parameter(lucid.tensor(p.numpy().copy()))
    opt2 = cls([resumed], lr=0.1, betas=_BETAS, amsgrad=True)
    opt2.load_state_dict(state)
    for g in _GRADS[3:]:
        p.grad = lucid.tensor(g)
        opt.step()
        resumed.grad = lucid.tensor(g)
        opt2.step()
    np.testing.assert_array_equal(resumed.numpy(), p.numpy())


def test_plain_adam_saves_no_running_maximum() -> None:
    _, opt = _run(optim.Adam, False, "cpu", _GRADS[:2])
    per_param = next(iter(opt.state_dict()["state"].values()))
    assert "max_exp_avg_sq" not in per_param
