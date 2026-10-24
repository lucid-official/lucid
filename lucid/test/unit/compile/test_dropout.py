"""Dropout policy in compile mode.

Contract:

  * **eval mode** or **p == 0** — identity passthrough.  Bit-exact with
    eager because both reduce to a clone.
  * **training mode** with **p > 0** — under ``lucid.compile`` and
    ``make_step`` the scaled mask is a feed drawn again on every call,
    through the same routine and generator eager dropout uses, and the op
    is an ordinary multiply (``lucid/_C/compile/RngFeeds.h``).  Masks
    differ call to call, and under the same seed the compiled output is
    eager's own.  It used to fall back to eager — which left every model
    with dropout training compiled not at all.
  * :func:`fused_step` keeps its own route: the ``dropout_stateful`` op
    threads an MPSGraph Philox state through
    ``compile_generic_fused_step_with_vars``.
"""

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid.test.unit.compile._helpers import (
    COMPILE_DEVICE,
    assert_compile_parity,
    metal_tensor,
)


def _dropout_model(p: float) -> nn.Module:
    class _M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(8, 16)
            self.drop = nn.Dropout(p=p)
            self.fc2 = nn.Linear(16, 4)

        def forward(self, x: lucid.Tensor) -> lucid.Tensor:
            return self.fc2(self.drop(self.fc1(x).relu()))

    return _M().to(COMPILE_DEVICE)


@pytest.mark.usefixtures("inference")
def test_dropout_eval_mode_compiles_clean() -> None:
    """In ``.eval()`` dropout is identity; trace must compile fully."""
    model = _dropout_model(p=0.5)
    model.eval()
    x = metal_tensor(4, 8)
    assert_compile_parity(model, x, atol=1e-4, rtol=1e-5)

    # Make sure no eager-only fallback was inserted.
    cm = lucid.compile(model.eval())
    cm(x)
    info = cm.cache_info()
    assert info["entries"] >= 1, "dropout-in-eval should compile"


def test_dropout_zero_prob_compiles_clean() -> None:
    """``p == 0`` is identity even in training mode — must compile."""
    model = _dropout_model(p=0.0)
    model.train()
    x = metal_tensor(4, 8)
    # Compile mode + eager should both be deterministic + identical.
    assert_compile_parity(model, x, atol=1e-4, rtol=1e-5)


@pytest.mark.usefixtures("inference")
def test_dropout_training_lucid_compile_draws_eagers_masks() -> None:
    """Training-mode dropout compiles, and draws exactly eager's mask."""
    model = _dropout_model(p=0.5)
    model.train()
    x = metal_tensor(4, 8)
    cm = lucid.compile(model)
    cm(x)  # trace
    assert not cm.cache_info()["eager_only"]
    for seed in (3, 4):
        lucid.manual_seed(seed)
        got = cm(x).numpy()
        lucid.manual_seed(seed)
        want = model(x).numpy()
        assert np.allclose(got, want, rtol=1e-5, atol=1e-6)
        # Same zeros: the mask itself is eager's, not a statistical twin.
        assert np.array_equal(got == 0, want == 0)


def test_dropout_training_make_step_matches_eager() -> None:
    """A compiled training step through dropout: eager's loss and gradients."""
    model = _dropout_model(p=0.3)
    model.train()
    x = metal_tensor(16, 8)

    def loss_fn(y: lucid.Tensor) -> lucid.Tensor:
        return (y * y).sum()

    step = lucid.compile.make_step(model, loss_fn)
    step(x).backward()  # trace
    assert not step.eager_only
    for seed in (5, 6):
        for p in model.parameters():
            p.grad = None
        lucid.manual_seed(seed)
        loss = step(x)
        loss.backward()
        got = float(loss.item())
        step_grads = [p.grad.numpy().copy() for p in model.parameters()]
        for p in model.parameters():
            p.grad = None
        lucid.manual_seed(seed)
        want = loss_fn(model(x))
        want.backward()
        assert np.isclose(got, float(want.item()), rtol=1e-5)
        for g, p in zip(step_grads, model.parameters()):
            assert np.allclose(g, p.grad.numpy(), rtol=1e-4, atol=1e-5)


@pytest.mark.usefixtures("inference")
def test_dropout_training_produces_random_outputs() -> None:
    """Training-mode dropout must still randomise (even on eager path).

    Two calls with the same input must produce *different* outputs —
    if the compile pipeline silently routed training-mode dropout
    through the deterministic stateless RNG path, both calls would
    return identical tensors and this assertion would fail.
    """
    model = _dropout_model(p=0.5)
    model.train()
    cm = lucid.compile(model)
    x = metal_tensor(64, 8)  # large enough that random masks differ
    a = cm(x).detach().clone()
    b = cm(x).detach().clone()
    diff = float((a - b).abs().max().item())
    assert diff > 0.0, (
        "training-mode dropout produced identical outputs across two "
        "calls — RNG state is stuck.  Either the eager fallback "
        "regressed or compile took a deterministic RNG path."
    )


def _engine_drop_path(t: lucid.Tensor) -> lucid.Tensor:
    from lucid._C import engine as _C_engine
    from lucid._dispatch import _unwrap, _wrap

    return _wrap(_C_engine.nn.drop_path(_unwrap(t), 0.5, True))


#: Masks that broadcast — one draw per channel, or per sample — rather than
#: one per element.  They had no traced form and ran eager.
_BROADCAST_MASKS = {
    "dropout1d": (lambda t: F.dropout1d(t, 0.5, training=True), (4, 6, 5)),
    "dropout2d": (lambda t: F.dropout2d(t, 0.5, training=True), (4, 6, 3, 3)),
    "dropout3d": (lambda t: F.dropout3d(t, 0.5, training=True), (2, 6, 2, 3, 3)),
    "drop_path": (_engine_drop_path, (8, 3, 4)),
}


@pytest.mark.parametrize("name", list(_BROADCAST_MASKS))
def test_broadcast_mask_dropout_compiles_and_draws_eagers_masks(name: str) -> None:
    fn, shape = _BROADCAST_MASKS[name]
    x = metal_tensor(*shape)
    compiled = lucid.compile(fn)
    compiled(x)  # trace
    assert not compiled.cache_info()["eager_only"]
    for seed in (3, 4):
        lucid.manual_seed(seed)
        got = compiled(x).numpy()
        lucid.manual_seed(seed)
        want = fn(x).numpy()
        assert np.allclose(got, want, rtol=1e-5, atol=1e-6)
        assert np.array_equal(got == 0, want == 0)


@pytest.mark.parametrize("name", list(_BROADCAST_MASKS))
def test_broadcast_mask_dropout_make_step_matches_eager(name: str) -> None:
    fn, shape = _BROADCAST_MASKS[name]

    class _M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(lucid.ones(shape[1]).to(COMPILE_DEVICE))

        def forward(self, x: lucid.Tensor) -> lucid.Tensor:
            view = (1, shape[1]) + (1,) * (len(shape) - 2)
            return fn(x * self.scale.reshape(*view))

    model = _M().train()
    x = metal_tensor(*shape)
    step = lucid.compile.make_step(model, lambda y: (y * y).sum())
    step(x).backward()  # trace
    assert not step.eager_only.snapshot()
    for seed in (5, 6):
        model.scale.grad = None
        lucid.manual_seed(seed)
        loss = step(x)
        loss.backward()
        got = (float(loss.item()), model.scale.grad.numpy().copy())
        model.scale.grad = None
        lucid.manual_seed(seed)
        eager = model(x)
        eager_loss = (eager * eager).sum()
        eager_loss.backward()
        assert np.isclose(got[0], float(eager_loss.item()), rtol=1e-5)
        assert np.allclose(got[1], model.scale.grad.numpy(), rtol=1e-4, atol=1e-5)
