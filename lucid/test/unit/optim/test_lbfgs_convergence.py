"""LBFGS must actually converge, on both devices.

Found 2026-07-26 by coverage-directed probing — ``lucid/optim/lbfgs.py`` was
the least-covered file in ``optim`` (21% of statements executed).

``_add_to_params`` updated each parameter with
``p._impl = lucid.add(p, lucid.mul(...))._impl``.  Two defects in that one line:

1. ``add``/``mul`` are differentiable, so the result carries a ``grad_fn`` and
   the parameter stopped being a **leaf**.  Backward then flowed *through* it
   and accumulated nothing, so every step after the first read a zero gradient,
   tripped the ``g_norm <= tol_grad`` early return, and LBFGS stalled — the loss
   was bit-identical across every step while the true optimum was 10x lower.
   Same family as the ``.to(dtype)`` leaf bug fixed the same day.
2. the ``alpha`` scalar (and the other scalar helpers in the two-loop
   recursion) were created on the default device, so a Metal parameter raised
   ``DeviceMismatch``.
"""

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.optim as optim

DEVICES = ["cpu", "metal"]


def _problem():
    x = np.random.default_rng(0).standard_normal((8, 4)).astype(np.float32)
    y = np.random.default_rng(1).standard_normal((8, 1)).astype(np.float32)
    design = np.hstack([x, np.ones((8, 1), dtype=np.float32)])
    sol, *_ = np.linalg.lstsq(design, y, rcond=None)
    return x, y, float(((design @ sol - y) ** 2).mean())


def _fit(device, steps=25, lr=0.5):
    x_np, y_np, best = _problem()
    lucid.manual_seed(0)
    model = nn.Linear(4, 1).to(device)
    opt = optim.LBFGS(model.parameters(), lr=lr, max_iter=20)
    x = lucid.tensor(x_np, device=device)
    y = lucid.tensor(y_np, device=device)

    def closure():
        opt.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        return loss

    start = float(((model(x) - y) ** 2).mean().item())
    for _ in range(steps):
        opt.step(closure)
    return start, float(((model(x) - y) ** 2).mean().item()), best, model, opt, closure


@pytest.mark.parametrize("device", DEVICES)
def test_lbfgs_reaches_the_least_squares_optimum(device):
    start, final, best, *_ = _fit(device)
    assert final < start, f"{device}: LBFGS did not reduce the loss at all"
    assert final - best < 0.05, f"{device}: stalled at {final:.6f}, optimum is {best:.6f}"


@pytest.mark.parametrize("device", DEVICES)
def test_parameters_stay_leaves_across_steps(device):
    """The stall's mechanism: a non-leaf parameter stops receiving gradients."""
    from lucid._dispatch import _unwrap

    _, _, _, model, opt, closure = _fit(device, steps=3)
    for name, p in model.named_parameters():
        assert _unwrap(p).is_leaf, f"{device}: {name} is no longer a leaf"
    closure()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{device}: {name} stopped receiving gradients"
        assert np.abs(p.grad.numpy()).max() > 0.0, f"{device}: {name} grad is all zero"


@pytest.mark.parametrize("device", DEVICES)
def test_loss_is_not_frozen_across_steps(device):
    """The visible symptom was a bit-identical loss on every step."""
    x_np, y_np, _ = _problem()
    lucid.manual_seed(0)
    model = nn.Linear(4, 1).to(device)
    # One iteration per step: with ``max_iter=20`` a single step solves this
    # least-squares problem outright (LCD-222), and the loss of an optimum
    # stays put for the right reason.
    opt = optim.LBFGS(model.parameters(), lr=0.5, max_iter=1)
    x = lucid.tensor(x_np, device=device)
    y = lucid.tensor(y_np, device=device)
    seen = []
    for _ in range(4):

        def closure():
            opt.zero_grad()
            loss = ((model(x) - y) ** 2).mean()
            loss.backward()
            return loss

        opt.step(closure)
        seen.append(float(((model(x) - y) ** 2).mean().item()))
    assert len(set(seen)) > 1, f"{device}: loss frozen at {seen[0]}"


def test_cpu_and_metal_agree():
    _, cpu_final, _, *_ = _fit("cpu")
    _, metal_final, _, *_ = _fit("metal")
    assert abs(cpu_final - metal_final) < 1e-3
